/* rad_block_dspark.h -- the DSpark block-diffusion drafter, fp8 weights over a bf16 checkpoint.
 *
 * ============================== WHAT IT IS ==============================
 *
 * `openbmb/MiniCPM5-2B-DSpark`: five plain Qwen3 attention layers that propose a whole block of
 * tokens in one forward pass, conditioned on the target trunk's residual stream at five
 * intermediate layers. Published acceptance length 5.52 at T = 0 (math 6.05 / code 6.11 /
 * general 4.16) versus MTP's 3.74-5.02.
 *
 * Not rad_block_dflash2.h despite the name: DSpark subclasses SpecForge's `DFlashDraftModel`,
 * which is plain Qwen3 attention (the checkpoint has q/k/v/o, q_norm/k_norm, gate/up/down and no
 * conv or selection tensors), while `Dflash2` is the conv-based drafter. This block reuses
 * Dflash2's structure (Config / Wire / declare / context / step) and its selector.
 *
 * ============================== THE FORWARD PASS ==============================
 *
 * The published repo has no reference code (`auto_map: dspark.DSparkDraftModel`);
 * `RadixArk/Qwen3.8-27B-DSpark` ships `dspark.py` and `dflash.py`, which this implements:
 *
 *     target_hidden = hidden_norm(fc(concat of the trunk's residual stream after layers
 *                                    1, 10, 20, 30, 39))          [n_taps * n_embd] -> [n_embd]
 *     h = target.embed_tokens([anchor, MASK, MASK, ...])            `block` noise rows
 *     per layer (x5):
 *         h_n = input_layernorm(h)
 *         q   = q_norm(q_proj(h_n))                                 q from the noise rows only
 *         k   = k_norm(concat(k_proj(target_hidden), k_proj(h_n)))  context first, then noise
 *         v   =        concat(v_proj(target_hidden), v_proj(h_n))
 *         rope(q, k);  attention is non-causal, mask None at inference
 *         h  += o_proj(attn);  h += mlp(post_attention_layernorm(h))
 *     logits = target.lm_head(norm(h)) + markov_w2 @ markov_w1[prev_token]
 *
 * `fc` then `hidden_norm`; `target_hidden` is computed once and each layer projects it with its
 * own k_proj and v_proj. The drafter uses the target's embedding table and lm_head, so the
 * vocabularies must agree.
 *
 * ============================== DUAL-SOURCE K/V AS A PAGED CACHE ==============================
 *
 * The reference appends context and noise rows to its cache, attends over both, then
 * `crop(start)`s the noise. Here that is a paged KV group: context written per committed token,
 * noise rows written past `seqused`, where the next context pass overwrites them. Two passes, as
 * in Dflash2:
 *
 *   Context pass, one row per committed token: the five k/v projections over the projected trunk
 *   taps, stored. The drafter's layers never run over the context.
 *
 *   Query pass, `block` rows per sequence, through the five layers, attending to the context and
 *   to each other.
 *
 * `k_norm` is per head over head_dim, so norming the two halves separately is bit-identical to
 * norming the concatenation. `attn_paged` with `causal = 0` serves the non-causal block. The
 * shape (16 heads, 2 KV heads, head_dim 128, GQA 8) matches the target, so the trunk's attention
 * and norm/rope rows serve it.
 *
 * ============================== ROW ALIGNMENT ==============================
 *
 * `block` is both the number of noise rows and the number of drafts (seven in, seven out, verify
 * width eight). Row j of [anchor, MASK x (block-1)] predicts position start + j + 1, the anchor
 * row included, so `steps() == block` here and `block - 1` in Dflash2, whose anchor row predicts
 * nothing. Getting this wrong only shifts proposals by one and drops acceptance to chance. The
 * core stages the block against `RadDrafterDecl::block` and runs the head over the last `depth`
 * rows, which covers both conventions (rad_builder.h).
 *
 * ============================== THE MARKOV HEAD USES Dflash2'S SELECTOR ==============================
 *
 * DSpark adds a rank-256 learned bigram bias to the draft logits:
 *
 *     logits[l] += markov_w2 @ markov_w1[prev(l)]      prev(0) = the anchor,
 *                                                      prev(l) = the token chosen at l - 1
 *
 * This is serial in l. Literally it is seven [rank, n_vocab] GEMMs with an argmax between each,
 * reading 66 MiB of `markov_w2` seven times a step. `dflash_select` scores
 *
 *     unary[l][c] + sum_r pred[prev][r] * hp[l][r] * succ[cand[l][c]][r]
 *
 * over the base head's top K, reading the predecessor after the previous step's argmax inside one
 * workgroup. With `hp` absent (treated as ones) it is the bigram `<pred[prev], succ[cand]>`;
 * passing `markov_w1` as `pred` and `markov_w2` as `succ` makes the walk the greedily decoded
 * Markov head. The gather is K * rank * 2 bytes a position (8 KiB at K = 16).
 *
 * Only the candidate set is approximated: the bias re-ranks the base head's top K, where the
 * reference could promote any of 130560 tokens. Since every draft is verified, this costs only
 * acceptance.
 *
 * The confidence head (`AcceptRatePredictor`) is not implemented: it only lets a serving loop
 * shorten the block adaptively, and this engine's depth is fixed at declare (1, 3 or 7). It does
 * not change what the drafter proposes.
 *
 * ============================== PLAIN NORMS ==============================
 *
 * `w`, not `1 + w`: the reference uses `Qwen3RMSNorm` (`model_type: qwen3`), so this geometry's
 * `wadd` and `wadd_qk` are zero. That matches the MiniCPM5 target but is the drafter's own
 * setting.
 *
 * ============================== PRECISION ==============================
 *
 *   what the recipe made them            fc, q/k/v/o, gate_up, down -- every linear
 *   bf16, no plane                       every norm, and both Markov codebooks
 *
 * The checkpoint is dense bf16; the linears are rad_fp8.h's DenseFP8 over weights the recipe
 * quantises, because the drafter is read whole every decode step (324M parameters: 648 MB at
 * bf16, 324 MB at eight bits). DenseFP8 reads block fp8 or MXFP4 against E4M3 activations, or a
 * rotated int8 form (IQ8R and the narrower grids under the same rotation) against int8 ones
 * (Config::act_i8); one format for all of them, since they share the activation pairs. This can
 * only change acceptance.
 *
 * The codebooks are bf16 and replicated: they are gathered by global token id from a merged
 * candidate set. 66.8 MiB each; only K rows are read per position, so the cost is residency.
 *
 * ============================== NOT DONE ==============================
 *
 *   The q/k norms and ropes are four launches per layer where `qk_norm_rope` would be one. That
 *   op takes a fused [q|k] row; the query pass would need a [q_dim + kv_dim] buffer, a column-
 *   slice store and a contiguous q copy, as in rad_block_attn_fp8.h. Worth twenty launches a step.
 *
 *   The all-reduce is not folded into the following norm (kArNone), so a TP2 query pass issues
 *   ten collectives. At 28 KiB per message for one sequence (under the lossy floor), a fold
 *   would save launches, not bytes.
 */
#ifndef RAD_BLOCK_DSPARK_H
#define RAD_BLOCK_DSPARK_H

#include "rad_arch.h"
#include "rad_fp8.h"

namespace rad {
namespace arch {

struct Dspark {
    enum { MAX_LAYERS = 8 };
    /* The 2-bit draft head's scale group, fixed by the w2a8 row. */
    enum { DS_W2_GROUP = 128 };

    /* The drafter config beyond its geometry, read by the plugin from `draft.*` metadata keys. */
    struct Config {
        int64_t n_taps = 0;     /* how many trunk layers `fc` concatenates */
        int64_t block  = 0;     /* noise rows per sequence, and drafts (one per row) */
        int64_t rank   = 0;     /* markov_rank: the codebooks' width */
        int64_t top_k  = 0;     /* candidates the Markov walk re-ranks */
        int64_t window = 0;     /* sliding attention width, 0 for none */
        int     causal = 0;
        /* The drafter's linears are served at i8a8h (DenseFP8::w8h: IQ8R, or a rotated int8 or
         * narrower form at group 128), learned by the plugin from `probe_i8` before it declares the
         * activation pairs: they are then int8 codes with one f32 scale per (row, act_g8), written
         * by had_quant_act_i8g and the rotating norm. Every linear must agree. `fc_mapped`:
         * probe_i8 made `dspark.fc.weight`'s name map. */
        bool    act_i8    = false;
        int64_t act_g8    = RAD_FP8_BLOCK;
        bool    fc_mapped = false;
        /* Every row predicts, so `block`, not `block - 1` (see ROW ALIGNMENT). */
        int64_t steps() const { return block; }
    };

    struct Wire {
        /* ---- the context pass, one row per committed token */
        rad_buf src = 0;   /* [max_tok, n_taps * n_embd]  the trunk's taps, read only */
        rad_buf c   = 0;   /* [max_tok, n_embd]           fc's output, then normed in place */
        /* K and V are separate buffers because of the per-head k norm: `rmsnorm` takes one row
         * pitch, and K heads inside a fused [K|V] row are head_dim apart within a token but
         * 2*kv_dim apart between tokens. libr4d refuses such an operand (libr4d/r4d_model.cpp,
         * rows_pitch). */
        rad_buf ck  = 0;   /* [max_tok, kv_dim]           one layer's K, reused across layers */
        rad_buf cv  = 0;   /* [max_tok, kv_dim]           one layer's V, reused across layers */
        /* The codes and scales the linears read: e4m3 | f32 [.., /128], or with Config::act_i8
         * i8 | f32 [.., /act_g8]. The same holds for the three ActFP8 pairs below, whose `q` and
         * `s` are then the int8 pair. */
        rad_buf srcq = 0, srcs = 0;
        rad_buf cq   = 0, cs   = 0;

        /* ---- the query pass, `block` rows per sequence */
        rad_buf x = 0;     /* [qrows, n_embd]  the drafter's residual stream */
        /* Normed input and the block's delta share one activation, as in the trunk
         * (rad_block_attn_fp8.h): o_proj and down_proj write `h.x`, and the next norm folds it
         * into the stream before norming, writing `h` again. */
        ActFP8  h{};       /* [qrows, n_embd] */
        rad_buf q = 0;     /* [qrows, q_dim] */
        rad_buf k = 0;     /* [qrows, kv_dim]   separate from v, see `ck` */
        rad_buf v = 0;     /* [qrows, kv_dim] */
        ActFP8  attn{};    /* [qrows, q_dim] */
        rad_buf gate_up = 0; /* [qrows, 2 * n_ff] */
        ActFP8  ffn{};     /* [qrows, n_ff] */

        /* ---- the head and the Markov walk, one row per draft position */
        rad_buf sel_h = 0; /* [srows, n_embd]  the gathered final hidden states */
        rad_buf dq    = 0; /* [srows, n_embd]  i8   the 2-bit head's quantised activation */
        rad_buf ds    = 0; /* [srows, 1]       f32  one scale a row */
        rad_buf dsum  = 0; /* [n_embd/128, pad] f32 GROUP-major, quant_act_i8's asum plane */
        rad_buf dlog  = 0; /* [srows, head_rows(g)] bf16  the coarse logits */
        rad_buf cand  = 0; /* [srows, top_k] i32  candidate token ids, global */
        rad_buf unary = 0; /* [srows, top_k] f32  and their logits */
        /* Vocab-parallel pairs, declared only when the head is sharded: (int32 id, f32 value)
         * pairs stored as f32, this rank's top-R and the all-gathered world * R. */
        rad_buf dpair = 0; /* [srows, top_k, 2]         f32 */
        rad_buf dgath = 0; /* [srows, world * top_k, 2] f32 */
        rad_buf tokens = 0;/* [max_seqs, steps] i32  the draft; the engine reads it by name */
    };

    /* rad_arch.h's vocab-head rule, asked through the block by whoever sizes `dlog` and the pair
     * planes. */
    static bool    head_sharded(const Geom& g) { return vocab_head_sharded(g); }
    static int64_t head_rows(const Geom& g)    { return vocab_head_rows(g); }

    Geom    g{};      /* the drafter's geometry, not the trunk's */
    Config  cfg{};
    Wire    w{};
    int     first_layer = 0;   /* absolute, for the KV group's layer_slot lookup */
    int64_t qrows = 0, crows = 0, srows = 0;
    int64_t gated_rows = 0;    /* the band op_gug was declared over; 0 when there is no fold */
    rad_kvgroup kv = 0;

    /* ---- model-scope weights */
    DenseFP8   w_fc{};
    rad_weight w_hnorm = 0, w_norm = 0;
    rad_weight w_mk1 = 0, w_mk2 = 0;         /* markov_w1 (pred), markov_w2 (succ) */
    rad_weight w_tok = 0;                    /* the target's embedding table */
    rad_weight w_dh = 0, w_dhs = 0;          /* the 2-bit copy of the target's lm_head */

    /* ---- per-layer weights. q, k and v are separate: the context pass needs only k and v, and a
     * block-scaled weight's layout follows its op's N, so k and v cannot be one weight issued a
     * half at a time. */
    rad_weight w_anorm[MAX_LAYERS] = {}, w_fnorm[MAX_LAYERS] = {};
    DenseFP8   w_q[MAX_LAYERS]{}, w_k[MAX_LAYERS]{}, w_v[MAX_LAYERS]{}, w_o[MAX_LAYERS]{};
    rad_weight w_qn[MAX_LAYERS] = {}, w_kn[MAX_LAYERS] = {};
    DenseFP8   w_gu[MAX_LAYERS]{}, w_dn[MAX_LAYERS]{};

    /* ---- context-pass ops */
    rad_op op_qsrc = 0, op_cfc = 0, op_cnorm = 0, op_qc = 0, op_crope = 0, op_cstore = 0;
    rad_op op_ck[MAX_LAYERS] = {}, op_cv[MAX_LAYERS] = {}, op_cknorm[MAX_LAYERS] = {};

    /* ---- query-pass ops. Weightless ones are declared once and issued per layer; ops with weights
     * are per layer because a weight's prefetch interval and residency class come from the op
     * that declares it. */
    rad_op op_embed = 0, op_embed_ar = 0;
    /* The fused norm / residual add / quantise, declared directly rather than via rad_fp8.h's
     * NormQuantFP8: this block uses none of its fused-collective forms, and an array of the wrapper
     * (which nests a Geom) makes g++ 16.1 ICE in build_vec_init when the plugin assigns
     * `m = Model{}`. */
    rad_op op_anorm[MAX_LAYERS] = {}, op_fnorm[MAX_LAYERS] = {}, op_outnorm = 0;
    rad_op op_qat = 0;
    rad_op op_qp[MAX_LAYERS] = {}, op_kp[MAX_LAYERS] = {}, op_vp[MAX_LAYERS] = {};
    rad_op op_op[MAX_LAYERS] = {};
    rad_op op_qn[MAX_LAYERS] = {}, op_kn[MAX_LAYERS] = {};
    rad_op op_gu[MAX_LAYERS] = {}, op_dn[MAX_LAYERS] = {};
    rad_op op_gug[MAX_LAYERS] = {};   /* the gate_up|silu|quant fold; 0 when nothing serves it */
    rad_op op_qrope = 0, op_krope = 0, op_store = 0, op_attn = 0;
    rad_op op_act = 0, op_qff = 0, op_ar = 0;

    /* ---- the head and the Markov walk */
    rad_op op_gather = 0, op_dquant = 0, op_dlogits = 0, op_dtopk = 0;
    rad_op op_dgather = 0, op_dmerge = 0;   /* the vocab-parallel pair; 0 when the head is whole */
    rad_op op_sel = 0;

    /* Whether the container's drafter is served at i8a8h and at which scale group, from its `fc`,
     * whose name map this makes (set Config::fc_mapped). Asked before the activation pairs are
     * declared, since it decides their dtype and scale grid. */
    static int probe_i8(RadBuilder* b, Names& nm, const char* ck, bool* i8, int64_t* g8) {
        const char* fwn = nm.f("dspark.fc.weight");
        RAD_ARCH_TRY(map_linear(b, fwn, { nm.ckpt("%s.fc.weight", ck) }));
        RadEncoding fe{};
        *i8 = weight_enc(b, fwn, &fe) && linear_is_w8h(fe);
        *g8 = *i8 ? linear_w8h_group(fe) : (int64_t)RAD_FP8_BLOCK;
        return RAD_OK;
    }

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int base_layer,
                 rad_kvgroup group, rad_weight tok_embd, const char* ck_prefix,
                 const char* ck_lm, const char* ck_lm_alt, const Wire& wire);
    /* The context pass, one row per committed token; `batch` carries those rows and the draft KV
     * group's slot mapping. */
    void context(RadCtx* c, const RadBatch* batch) const;
    /* The query pass, the head and the Markov walk, `block` rows per sequence. */
    void step(RadCtx* c, const RadBatch* batch) const;

private:
    int64_t q_dim()  const { return g.n_head * g.head_dim; }
    int64_t kv_dim() const { return g.n_head_kv * g.head_dim; }
};

/* ------------------------------------------------------------------------------ declare */

inline int Dspark::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c,
                           int base_layer, rad_kvgroup group, rad_weight tok_embd,
                           const char* ck, const char* ck_lm, const char* ck_lm_alt,
                           const Wire& wire) {
    g = geom; cfg = c; w = wire; first_layer = base_layer; kv = group; w_tok = tok_embd;
    if (!ck || !ck_lm || !w_tok) return RAD_E_INVAL;
    if (g.n_layer <= 0 || g.n_layer > MAX_LAYERS) {
        fprintf(stderr, "radiance: dspark has %lld layers; this block carries at most %d\n",
                (long long)g.n_layer, MAX_LAYERS);
        return RAD_E_UNSUPPORTED;
    }
    if (cfg.block < 2 || cfg.n_taps <= 0 || cfg.rank <= 0 || cfg.top_k <= 0) return RAD_E_INVAL;

    const int64_t qd = q_dim(), kvd = kv_dim();

    crows = g.max_tok;
    qrows = g.max_seqs * cfg.block;
    if (qrows > g.max_tok) qrows = g.max_tok;
    /* Every query row predicts, so srows equals the block's rows; kept separate from qrows because
     * Dflash2's convention makes them differ. */
    srows = g.max_seqs * cfg.steps();
    if (qrows <= 0 || srows <= 0) return RAD_E_INVAL;

    /* The linears read the activation pairs the plugin declared: int8 ones take rotated int8
     * weights (cfg.act_i8), E4M3 ones block fp8 or MXFP4. */
    w_fc.w8h_ok = cfg.act_i8;
    for (int i = 0; i < MAX_LAYERS; ++i)
        w_q[i].w8h_ok = w_k[i].w8h_ok = w_v[i].w8h_ok = w_o[i].w8h_ok = w_gu[i].w8h_ok =
            w_dn[i].w8h_ok = cfg.act_i8;

    /* ---------------------------------------------------------------- model-scope weights */
    RAD_ARCH_TRY(w_fc.declare(b, nm, nm.f("dspark.fc"), g.n_embd, cfg.n_taps * g.n_embd,
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model(),
                              { nm.ckpt("%s.fc.weight", ck) }, {}, cfg.fc_mapped));
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.hidden_norm.weight"),
                          nm.ckpt("%s.hidden_norm.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.norm.weight"), nm.ckpt("%s.norm.weight", ck)));

    w_hnorm = decl_w(b, nm.f("dspark.hidden_norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    w_norm  = decl_w(b, nm.f("dspark.norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    /* The Markov codebooks, bf16, replicated over the whole vocabulary. `markov_w1` (nn.Embedding)
     * and `markov_w2` (nn.Linear) both store [vocab, rank], so `w2 @ w1[prev]` is
     * `<w1[prev], w2[cand]>` and they map to the selector's `pred` and `succ`. RAD_ACCESS_VOCAB:
     * reads gather K rows per position. */
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.markov_w1"),
                          nm.ckpt("%s.markov_head.markov_w1.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.markov_w2"),
                          nm.ckpt("%s.markov_head.markov_w2.weight", ck)));

    w_mk1 = decl_w(b, nm.f("dspark.markov_w1"), RAD_BF16, {g.n_vocab_all, cfg.rank},
                   RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    w_mk2 = decl_w(b, nm.f("dspark.markov_w2"), RAD_BF16, {g.n_vocab_all, cfg.rank},
                   RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());

    /* The 2-bit draft head over the target's lm_head (as in rad_block_mtp_fp8.h):
     * `dspark.draft_head.weight` is that tensor under its own name, quantised by the recipe to u2
     * codes with an f16 scale and u8 zero per group of 128. One checkpoint tensor, two views.
     *
     * The alternative source covers a tied head (small LLaMA derivatives ship no `lm_head.weight`
     * and reuse the embedding); it must match how the trunk declares its own head. */
    const bool dh_shard = head_sharded(g);
    const int  dh_split = dh_shard ? RAD_SHARD_ROW : RAD_SHARD_NONE;
    const char* dhn = nm.f("dspark.draft_head.weight");
    if (ck_lm_alt && *ck_lm_alt) RAD_ARCH_TRY(map_copy_alt(b, dhn, ck_lm, ck_lm_alt));
    else                         RAD_ARCH_TRY(map_copy(b, dhn, ck_lm));
    w_dh  = decl_view(b, dhn, nullptr, "codes", RAD_U2, {head_rows(g), g.n_embd},
                      RAD_ACCESS_PER_TOKEN, dh_split, grp_model());
    w_dhs = decl_view(b, nm.f("dspark.draft_head.scale"), dhn, "scale,zero", RAD_F16,
                      {head_rows(g), g.n_embd}, RAD_ACCESS_PER_TOKEN, dh_split, grp_model());
    if (!w_dh || !w_dhs) return RAD_E_INVAL;

    /* ---------------------------------------------------------------- per-layer weights */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        const RadWeightGroup grp = grp_layer(first_layer + i);
        RAD_ARCH_TRY(map_copy(b, nm.f("dspark.blk.%d.attn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.input_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dspark.blk.%d.ffn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.post_attention_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dspark.blk.%d.attn_q_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.q_norm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dspark.blk.%d.attn_k_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.k_norm.weight", ck, i)));

        w_anorm[i] = decl_w(b, nm.f("dspark.blk.%d.attn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        w_fnorm[i] = decl_w(b, nm.f("dspark.blk.%d.ffn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        RAD_ARCH_TRY(w_q[i].declare(b, nm, nm.f("dspark.blk.%d.attn_q", i), qd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp,
                                    { nm.ckpt("%s.layers.%d.self_attn.q_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_k[i].declare(b, nm, nm.f("dspark.blk.%d.attn_k", i), kvd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp,
                                    { nm.ckpt("%s.layers.%d.self_attn.k_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_v[i].declare(b, nm, nm.f("dspark.blk.%d.attn_v", i), kvd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp,
                                    { nm.ckpt("%s.layers.%d.self_attn.v_proj.weight", ck, i) }));
        w_qn[i] = decl_w(b, nm.f("dspark.blk.%d.attn_q_norm.weight", i), RAD_F32, {g.head_dim},
                         RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        w_kn[i] = decl_w(b, nm.f("dspark.blk.%d.attn_k_norm.weight", i), RAD_F32, {g.head_dim},
                         RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
        RAD_ARCH_TRY(w_o[i].declare(b, nm, nm.f("dspark.blk.%d.attn_output", i), g.n_embd, qd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp,
                                    { nm.ckpt("%s.layers.%d.self_attn.o_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_gu[i].declare(b, nm, nm.f("dspark.blk.%d.ffn_gate_up", i),
                                     2 * g.n_ff, g.n_embd,
                                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp,
                                     { nm.ckpt("%s.layers.%d.mlp.gate_proj.weight", ck, i),
                                       nm.ckpt("%s.layers.%d.mlp.up_proj.weight", ck, i) },
                                     { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(w_dn[i].declare(b, nm, nm.f("dspark.blk.%d.ffn_down", i), g.n_embd, g.n_ff,
                                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp,
                                     { nm.ckpt("%s.layers.%d.mlp.down_proj.weight", ck, i) }));
    }

    /* Non-causal: each query row sees its whole block plus the context (`causal` comes from the
     * config's `is_causal: false`).
     *
     * Declared before both kv_stores, which read the resolved `block_size` back with
     * rad_kv_block_size(); before this declaration it returns zero. */
    op_attn = rw(b, RAD_OP(b, "attn_paged",
                    RAD_PARAMS(RAD_RANGE("q_len", 1, cfg.block), RAD_INT("head_dim", g.head_dim),
                               RAD_INT("gqa", g.gqa()), RAD_INT("causal", cfg.causal),
                               RAD_INT("window", cfg.window), RAD_INT("n_head", g.n_head),
                               RAD_INT("block_size", RAD_KV_BLOCK),
                               RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                               RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                    RAD_NOWEIGHTS),
                {w.q}, {w.attn.x});

    /* ---------------------------------------------------------------- the context pass
     *
     * Every linear is a DenseFP8 GEMM over the pair its quantiser writes: `quant_act_fp8` codes
     * for block fp8 or MXFP4, had_quant_act_i8g's at the weights' scale group for a rotated int8
     * drafter (cfg.act_i8). The linears read one activation pair, so they are all i8a8h forms at
     * one group or none is. There is no `fp8a16` band: that kernel is M <= 1, and neither pass
     * runs at one row. */
    const char* const qop = cfg.act_i8 ? "had_quant_act_i8g" : "quant_act_fp8";
    const int64_t     qgr = cfg.act_i8 ? cfg.act_g8 : (int64_t)RAD_FP8_BLOCK;
    {
        auto fits = [&](const DenseFP8& d) {
            return d.w8h == cfg.act_i8 && (!d.w8h || d.g8 == cfg.act_g8);
        };
        bool same = fits(w_fc);
        for (int64_t l = 0; l < g.n_layer; ++l)
            same = same && fits(w_q[l]) && fits(w_k[l]) && fits(w_v[l]) && fits(w_o[l]) &&
                   fits(w_gu[l]) && fits(w_dn[l]);
        if (!same) {
            fprintf(stderr, "radiance: the drafter's linears are not all rotated int8 at one scale "
                            "group or all block fp8 / MXFP4; they read one activation pair, so the "
                            "recipe must give them one format\n");
            return RAD_E_DTYPE;
        }
    }
    op_qsrc = rw(b, RAD_OP(b, qop,
                    RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", cfg.n_taps * g.n_embd),
                               RAD_INT("group", qgr), RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.src}, {w.srcq, w.srcs});
    op_cfc = w_fc.gemm(b, crows, g.n_embd, cfg.n_taps * g.n_embd, {w.srcq, w.srcs}, {w.c});
    /* In place, like the per-head q/k norms below: a row's scale is computed before the row is
     * written. */
    op_cnorm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_hnorm)),
                 {w.c}, {w.c});
    /* One quantise for all layers: every layer's k and v read the same `c`. */
    op_qc = rw(b, RAD_OP(b, qop,
                  RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", g.n_embd),
                             RAD_INT("group", qgr), RAD_STR("dtype", g.dtype)),
                  RAD_NOWEIGHTS),
              {w.c}, {w.cq, w.cs});
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        /* One op per weight: an op's declared weight list gives each weight its prefetch interval
         * and residency class. */
        op_ck[i] = w_k[i].gemm(b, crows, kvd, g.n_embd, {w.cq, w.cs}, {w.ck});
        op_cv[i] = w_v[i].gemm(b, crows, kvd, g.n_embd, {w.cq, w.cs}, {w.cv});
        op_cknorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, crows * g.n_head_kv),
                                        RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                        RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(w_kn[i])),
                         {w.ck}, {w.ck});
    }
    /* K-only rope: `rope` rotates the first n_head + n_head_kv heads, so a K-only tensor is
     * n_head = n_head_kv, n_head_kv = 0. Weightless, so one op serves every layer.
     *
     * Each context row is rotated by its own absolute position, as in the reference's single
     * concatenated rope (k gets the full position range, q the last q_len). */
    op_crope = rw(b, decl_rotary(b, g, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head_kv), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.ck}, {w.ck});
    op_cstore = rw(b, RAD_OP(b, "kv_store",
                      RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                 RAD_INT("n_head_kv", g.n_head_kv),
                                 RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                 RAD_STR("kv_dtype", g.kv_dtype)),
                      RAD_NOWEIGHTS),
                  {w.ck, w.cv}, {});

    /* ---------------------------------------------------------------- the query pass */
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_embd", g.n_embd),
                                RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                RAD_INT("vocab_offset", g.vocab_off)),
                     RAD_WEIGHTS(w_tok)),
                 {}, {w.x});
    /* The target's embedding is vocab-sharded, so the lookup is partial. The all-reduce is exact:
     * each row comes from one rank and the others add zero, so a lossy wire would only lose
     * precision. */
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                            RAD_PARAMS(RAD_INT("world_size", g.world),
                                       RAD_RANGE("numel", g.n_embd, qrows * g.n_embd),
                                       RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                            RAD_NOWEIGHTS),
                        {w.x}, {w.x});

    /* Norm, residual add and quantise are one op, `rmsnorm_quant_fp8` (what rad_fp8.h's
     * NormQuantFP8 wraps), or its rotating int8 twin `rmsnorm_had_quant_i8g` for a rotated int8
     * drafter, which takes the same operands. Layer 0 has no delta to fold, each later norm folds the previous
     * block's delta, and op_outnorm folds the last into the final norm, so there is no `add` op.
     * The final norm's fp8 output is unused (the head runs its own i8 quantiser), a few KiB of
     * waste that saves a launch: libr4d has no plain `rmsnorm_add` row.
     *
     * The op computes residual_out = x + residual, y = rmsnorm(residual_out) * w. A folded site
     * passes the previous block's output as `x` and the stream as `residual`; layer 0 passes the
     * stream as `x` and no residual. `residual` is INOUT, so it is in both the read and write sets;
     * otherwise the planner could reorder a reader in front of it. */
    auto decl_norm = [&](rad_weight gain, bool fold) -> rad_op {
        rad_op h = RAD_OP(b, cfg.act_i8 ? "rmsnorm_had_quant_i8g" : "rmsnorm_quant_fp8",
                          RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                     RAD_F64("eps", g.eps), RAD_INT("group", qgr),
                                     RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd)),
                          RAD_WEIGHTS(gain));
        if (!h) return 0;
        rad_buf rd[2] = { fold ? w.h.x : w.x, w.x };
        rad_buf wr[4] = { w.h.q, w.h.s, w.h.x, w.x };
        rad_op_reads (b, h, rd, fold ? 2 : 1);
        rad_op_writes(b, h, wr, fold ? 4 : 3);
        return h;
    };
    op_qat = rw(b, RAD_OP(b, qop,
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", qd),
                              RAD_INT("group", qgr), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.attn.x}, {w.attn.q, w.attn.s});
    if (!op_qat) return RAD_E_INVAL;

    /* Invariant: the norms are declared where step() issues them. Declaration order is the buffer
     * planner's liveness (a transient lives over the op indices that touch it, and disjoint spans
     * may share bytes; core/build/rad_bufplan.cpp). The norms are the only ops touching the stream
     * `x` after the embedding, so their positions define its lifetime: each layer's two norms sit
     * in the loop, the final one after every op the layers issue, including the shared handles
     * below. Declared up front, the stream's lifetime would end before the layers and the planner
     * could overlay a feed-forward temporary on it. */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_anorm[i] = decl_norm(w_anorm[i], i != 0);
        if (!op_anorm[i]) return RAD_E_INVAL;
        op_qp[i] = w_q[i].gemm(b, qrows, qd, g.n_embd, {w.h.q, w.h.s}, {w.q});
        op_kp[i] = w_k[i].gemm(b, qrows, kvd, g.n_embd, {w.h.q, w.h.s}, {w.k});
        op_vp[i] = w_v[i].gemm(b, qrows, kvd, g.n_embd, {w.h.q, w.h.s}, {w.v});
        op_qn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head), RAD_INT("n", g.head_dim),
                                    RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_qn[i])),
                     {w.q}, {w.q});
        op_kn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head_kv),
                                    RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                    RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_kn[i])),
                     {w.k}, {w.k});
        op_op[i] = w_o[i].gemm(b, qrows, g.n_embd, qd, {w.attn.q, w.attn.s}, {w.h.x});
        op_fnorm[i] = decl_norm(w_fnorm[i], true);
        if (!op_fnorm[i]) return RAD_E_INVAL;
        op_gu[i] = w_gu[i].gemm(b, qrows, 2 * g.n_ff, g.n_embd, {w.h.q, w.h.s}, {w.gate_up});
        /* The fold is an fp8a8 tile, so any other gate_up takes the pair. */
        if (!w_gu[i].mx && !w_gu[i].m2x && !w_gu[i].w8h)
            op_gug[i] = decl_gated_fold(b, qrows, 2 * g.n_ff, g.n_embd, w_gu[i].w, w_gu[i].s,
                                        {w.h.q, w.h.s}, {w.ffn.q, w.ffn.s}, &gated_rows);
        op_dn[i] = w_dn[i].gemm(b, qrows, g.n_embd, g.n_ff, {w.ffn.q, w.ffn.s}, {w.h.x});
    }

    op_qrope = rw(b, decl_rotary(b, g, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.q});
    op_krope = rw(b, decl_rotary(b, g, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head_kv), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.k}, {w.k});
    op_store = rw(b, RAD_OP(b, "kv_store",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head_kv", g.n_head_kv),
                                RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.k, w.v}, {});
    op_act = rw(b, RAD_OP(b, "silu_mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                              RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.gate_up}, {w.ffn.x});
    op_qff = rw(b, RAD_OP(b, qop,
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                              RAD_INT("group", qgr), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.ffn.x}, {w.ffn.q, w.ffn.s});
    /* One collective, issued after both the out and the down projection (same shape, same buffer).
     * Exact or lossy per the deployment, as for the trunk. */
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                      RAD_PARAMS(RAD_INT("world_size", g.world),
                                 RAD_RANGE("numel", g.n_embd, qrows * g.n_embd),
                                 RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                 RAD_INT("min_bytes", g.wire_min_bytes)),
                      RAD_NOWEIGHTS),
                  {w.h.x}, {w.h.x});
    op_outnorm = decl_norm(w_norm, true);
    if (!op_outnorm) return RAD_E_INVAL;

    /* ---------------------------------------------------------------- the head and the walk
     *
     * Every row predicts, so the head runs over all `steps` rows, anchor included (see ROW
     * ALIGNMENT). The gather numbers rows s * steps + l as the walk reads them; for this
     * convention it is the identity, but `out_ids` is the contract, so it is issued anyway. */
    op_gather = rw(b, RAD_OP(b, "gather_rows",
                      RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_embd),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.h.x}, {w.sel_h});
    op_dquant = rw(b, RAD_OP(b, "quant_act_i8",
                      RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_embd),
                                 RAD_INT("group", DS_W2_GROUP), RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.sel_h}, {w.dq, w.ds, w.dsum});
    op_dlogits = rw(b, RAD_OP(b, "gemm_nt_q",
                       RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                                  RAD_INT("K", g.n_embd), RAD_INT("group", DS_W2_GROUP),
                                  RAD_STR("dtype", "w2a8")),
                       RAD_WEIGHTS(w_dh, w_dhs)),
                   {w.dq, w.ds, w.dsum}, {w.dlog});
    /* Sharded, the top-R runs over this rank's plane, reports global ids and also writes the pair
     * plane the gather moves. Replicated, `vocab_off` is omitted: an optional parameter present
     * with value 0 is a different declaration from an absent one. */
    op_dtopk = dh_shard
        ? rw(b, RAD_OP(b, "row_topk",
                RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                           RAD_INT("R", cfg.top_k), RAD_INT("vocab_off", g.vocab_off),
                           RAD_STR("dtype", g.dtype)),
                RAD_NOWEIGHTS),
             {w.dlog}, {w.cand, w.unary, w.dpair})
        : rw(b, RAD_OP(b, "row_topk",
                RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                           RAD_INT("R", cfg.top_k), RAD_STR("dtype", g.dtype)),
                RAD_NOWEIGHTS),
             {w.dlog}, {w.cand, w.unary});
    if (dh_shard) {
        /* Row-interleaved, so one draft position's pairs from every rank are adjacent for
         * `row_topk_merge`, which reduces them to exactly the unsharded global top-R, ties
         * included; the replicated codebooks are then indexed by global id. */
        op_dgather = rw(b, RAD_OP(b, "all_gather",
                           RAD_PARAMS(RAD_INT("world_size", g.world),
                                      RAD_INT("numel", srows * cfg.top_k * 2),
                                      RAD_STR("dtype", "f32"),
                                      RAD_INT("row", cfg.top_k * 2)),
                           RAD_NOWEIGHTS),
                       {w.dpair}, {w.dgath});
        op_dmerge = rw(b, RAD_OP(b, "row_topk_merge",
                          RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("R", cfg.top_k),
                                     RAD_INT("world_size", g.world),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {w.dgath}, {w.cand, w.unary});
    }
    /* The Markov walk: `dflash_select` with `hp` absent is the bigram
     * `unary + <markov_w1[prev], markov_w2[cand]>`, decoded greedily (see the file header). The
     * absent `hp` is passed as RAD_NONE at the issue site.
     *
     * `anchor_stride` is `block`: the operand is the step's token_ids, `block` per sequence with
     * the anchor first. */
    op_sel = rw(b, RAD_OP(b, "dflash_select",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", cfg.steps()),
                              RAD_INT("top_k", cfg.top_k), RAD_INT("rank", cfg.rank),
                              RAD_INT("n_vocab", g.n_vocab_all),
                              RAD_INT("anchor_stride", cfg.block), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_mk1, w_mk2)),
               {w.cand, w.unary}, {w.tokens});

    return RAD_OK;
}

/* ------------------------------------------------------------------------------ the context pass */

inline void Dspark::context(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;

    RAD_ISSUE_N(c, op_qsrc, T, brows(w.src, T), brows(w.srcq, T), brows(w.srcs, T));
    RAD_ISSUE_N(c, op_cfc, T, brows(w.srcq, T), brows(w.srcs, T), RAD_W(w_fc.w), RAD_W(w_fc.s),
                brows(w.c, T), RAD_NONE, w_fc.bref());
    RAD_ISSUE_N(c, op_cnorm, T, brows(w.c, T), RAD_W(w_hnorm), brows(w.c, T));
    RAD_ISSUE_N(c, op_qc, T, brows(w.c, T), brows(w.cq, T), brows(w.cs, T));

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        RAD_ISSUE_N(c, op_ck[i], T, brows(w.cq, T), brows(w.cs, T),
                    RAD_W(w_k[i].w), RAD_W(w_k[i].s), brows(w.ck, T), RAD_NONE, w_k[i].bref());
        RAD_ISSUE_N(c, op_cv[i], T, brows(w.cq, T), brows(w.cs, T),
                    RAD_W(w_v[i].w), RAD_W(w_v[i].s), brows(w.cv, T), RAD_NONE, w_v[i].bref());
        RAD_ISSUE_N(c, op_cknorm[i], T * g.n_head_kv,
                    brows(w.ck, T), RAD_W(w_kn[i]), brows(w.ck, T));
        RAD_ISSUE_N(c, op_crope, T, brows(w.ck, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_cstore, T, brows(w.ck, T), brows(w.cv, T),
                    praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
    }
}

/* ------------------------------------------------------------------------------ the query pass */

inline void Dspark::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    /* The fused norm. Folded, `x` is the previous block's delta in `h.x` and the stream is the
     * residual; unfolded (layer 0), `x` is the stream and the residual is RAD_NONE. */
    auto issue_norm = [&](rad_op op, bool fold, rad_weight gain, int64_t rows) {
        RAD_ISSUE_N(c, op, rows, brows(fold ? w.h.x : w.x, rows),
                    fold ? brows(w.x, rows) : RAD_NONE, RAD_W(gain),
                    brows(w.h.q, rows), brows(w.h.s, rows), brows(w.h.x, rows));
    };

    RAD_ISSUE_N(c, op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok),
                brows(w.x, T));
    if (op_embed_ar)
        RAD_ISSUE_N(c, op_embed_ar, (int64_t)T * g.n_embd, brows(w.x, T), RAD_NONE);

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;

        /* ---- attention. The norm folds the previous feed-forward delta into the stream and
         * quantises; at layer 0 it norms the embedding. */
        issue_norm(op_anorm[i], i != 0, w_anorm[i], T);
        RAD_ISSUE_N(c, op_qp[i], T, brows(w.h.q, T), brows(w.h.s, T),
                    RAD_W(w_q[i].w), RAD_W(w_q[i].s), brows(w.q, T), RAD_NONE, w_q[i].bref());
        RAD_ISSUE_N(c, op_kp[i], T, brows(w.h.q, T), brows(w.h.s, T),
                    RAD_W(w_k[i].w), RAD_W(w_k[i].s), brows(w.k, T), RAD_NONE, w_k[i].bref());
        RAD_ISSUE_N(c, op_vp[i], T, brows(w.h.q, T), brows(w.h.s, T),
                    RAD_W(w_v[i].w), RAD_W(w_v[i].s), brows(w.v, T), RAD_NONE, w_v[i].bref());
        RAD_ISSUE_N(c, op_qn[i], T * g.n_head, brows(w.q, T), RAD_W(w_qn[i]), brows(w.q, T));
        RAD_ISSUE_N(c, op_kn[i], T * g.n_head_kv, brows(w.k, T), RAD_W(w_kn[i]), brows(w.k, T));
        RAD_ISSUE_N(c, op_qrope, T, brows(w.q, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_krope, T, brows(w.k, T), praw(batch->positions, RAD_I32, T));
        /* Noise rows go into the cache past `seqused`; the next context pass overwrites them,
         * which is the reference's `crop(start)`. */
        RAD_ISSUE_N(c, op_store, T, brows(w.k, T), brows(w.v, T),
                    praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
        RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                    brows(w.q, T), kv_cache(kv, first_layer + i),
                    praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(w.attn.x, T));
        RAD_ISSUE_N(c, op_qat, T, brows(w.attn.x, T), brows(w.attn.q, T), brows(w.attn.s, T));
        RAD_ISSUE_N(c, op_op[i], T, brows(w.attn.q, T), brows(w.attn.s, T),
                    RAD_W(w_o[i].w), RAD_W(w_o[i].s), brows(w.h.x, T), RAD_NONE, w_o[i].bref());
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);

        /* ---- the feed-forward, behind the same fused norm-add-quantise */
        issue_norm(op_fnorm[i], true, w_fnorm[i], T);
        if (op_gug[i] && T <= gated_rows) {
            RAD_ISSUE_N(c, op_gug[i], T, brows(w.h.q, T), brows(w.h.s, T),
                        RAD_W(w_gu[i].w), RAD_W(w_gu[i].s), brows(w.ffn.q, T), brows(w.ffn.s, T),
                        RAD_NONE);
        } else {
            RAD_ISSUE_N(c, op_gu[i], T, brows(w.h.q, T), brows(w.h.s, T),
                        RAD_W(w_gu[i].w), RAD_W(w_gu[i].s), brows(w.gate_up, T), RAD_NONE,
                        w_gu[i].bref());
            RAD_ISSUE_N(c, op_act, T, brows(w.gate_up, T), brows(w.ffn.x, T));
            RAD_ISSUE_N(c, op_qff, T, brows(w.ffn.x, T), brows(w.ffn.q, T), brows(w.ffn.s, T));
        }
        RAD_ISSUE_N(c, op_dn[i], T, brows(w.ffn.q, T), brows(w.ffn.s, T),
                    RAD_W(w_dn[i].w), RAD_W(w_dn[i].s), brows(w.h.x, T), RAD_NONE, w_dn[i].bref());
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
    }

    /* The last layer's delta folds into the final norm. */
    issue_norm(op_outnorm, true, w_norm, T);

    /* The predicting rows (all of them), named by the scheduler through out_ids. */
    const int64_t S = batch->n_out;
    if (S <= 0) return;
    RAD_ISSUE_N(c, op_gather, S, RAD_B(w.h.x), praw(batch->out_ids, RAD_I32, S),
                brows(w.sel_h, S));

    /* `dsum` is group-major, [n_embd/group][ceil(M/16)*16], so it has no row slice; pass it
     * whole. */
    RAD_ISSUE_N(c, op_dquant, S, brows(w.sel_h, S), brows(w.dq, S), brows(w.ds, S),
                RAD_B(w.dsum));
    /* The trailing operand is mxfp4's optional reference, absent here. Operands are positional,
     * so an absent optional one is RAD_NONE, not a shorter list. */
    RAD_ISSUE_N(c, op_dlogits, S, brows(w.dq, S), brows(w.ds, S), RAD_W(w_dh), RAD_W(w_dhs),
                brows(w.dlog, S), RAD_B(w.dsum), RAD_NONE);
    if (op_dmerge) {
        /* Sharded, this rank's top-R are only candidates; the merge overwrites `cand`/`unary`. */
        RAD_ISSUE_N(c, op_dtopk, S, brows(w.dlog, S), brows(w.cand, S), brows(w.unary, S),
                    brows(w.dpair, S));
        RAD_ISSUE(c, op_dgather, brows(w.dpair, S), brows(w.dgath, S));
        RAD_ISSUE_N(c, op_dmerge, S, brows(w.dgath, S), brows(w.cand, S), brows(w.unary, S));
    } else {
        RAD_ISSUE_N(c, op_dtopk, S, brows(w.dlog, S), brows(w.cand, S), brows(w.unary, S),
                    RAD_NONE);
    }
    /* RAD_NONE in the third slot is the absent hidden projection: the bigram form. */
    RAD_ISSUE_N(c, op_sel, batch->n_seq, brows(w.cand, S), brows(w.unary, S), RAD_NONE,
                praw(batch->token_ids, RAD_I32, T), RAD_W(w_mk1), RAD_W(w_mk2),
                brows(w.tokens, batch->n_seq));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_DSPARK_H */
