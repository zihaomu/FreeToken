/* rad_block_muse_dflash.h -- Meta Muse Glimmer's DFlash block-diffusion drafter, bf16 weights.
 *
 * ============================== WHAT IT IS ==============================
 *
 * `meta-models/Muse-Glimmer-30B-assistant`: five plain attention layers that denoise a whole
 * block of masked positions in one forward pass, conditioned on the target trunk's residual
 * stream at layers 1, 13, 25, 37 and 49. The target then verifies the block in one parallel
 * pass, so every proposal is checked and a bad draft costs speed, never text.
 *
 * Structurally it is the DSpark drafter (rad_block_dspark_bf16.h) exactly -- fc over the
 * concatenated taps, one hidden norm, q from the noise rows and k/v projected from the shared
 * context, plain q/k gains, SwiGLU, the target's own output table -- with three differences this
 * block exists to state:
 *
 *   * **No Markov head.** DSpark adds markov_w2 @ markov_w1[prev] to the draft logits; the Muse
 *     drafter has no such tensors, so its head is the plain full-vocabulary argmax of
 *     target.lm_head(output_norm(h)). `cfg.rank == 0` is the honest spelling of that.
 *   * **The anchor row does not predict.** The block is [the last accepted token, MASK x 15] and
 *     the reference takes logits from rows 1..15 (`last_hidden_state[:, 1:]`), so `steps()` is
 *     `block - 1`, as in Dflash2 rather than Dspark.
 *   * **No per-tap norm and no output gate** (`tap_norm = 0`, `gate = 0`): the reference has one
 *     RMS norm over the fused 5 x n_embd context, and no gate on the attention output.
 *
 * ============================== THE REFERENCE ==============================
 *
 * transformers' `MuseGlimmerAssistantModel.forward` plus the `DFlashTokenCandidateGenerator` that
 * drives it:
 *
 *     ctx   = rmsnorm(fc(concat of the trunk's residual stream after layers 1,13,25,37,49))
 *     h     = target.embed_tokens([anchor, MASK x (block-1)])      the raw table, the trunk's
 *                                                                  embedding norm BYPASSED
 *     per layer (x5):
 *         h_n = input_layernorm(h)
 *         q   = q_norm(q_proj(h_n))
 *         k,v = k_norm(k_proj(concat(ctx, h))), v_proj(concat(ctx, h))
 *         rope(q,k, theta 500000); attention is bidirectional over the block and causal to the
 *         context, inside a sliding window of 2048
 *         h  += o_proj(attn);  h += down(act(gate(h_n2)) * up(h_n2)), act = silu
 *     logits = target.lm_head(norm(h))[1:]                       15 drafts
 *
 * Two passes, as in Dflash2: the context pass projects and stores K/V for every committed token
 * once per step; the query pass runs the 16 block rows through the five layers. The context K/V
 * lives in the drafter's own paged group, written past `seqused` for the block rows so every row
 * sees its siblings, and overwritten by the next context pass (the reference's cache.crop).
 *
 * ============================== PRECISION ==============================
 *
 * Block fp8 or OCP-MXFP4 weights (`DenseFP8`, whichever the recipe wrote), read against an E4M3
 * activation quantised per row and 128 of K; the residual stream stays f32, which is what the
 * reference's autocast does and what keeps five stacked layers of 6656-wide adds from drifting.
 * The per-head q/k gains and both norms stay bf16.
 *
 * ============================== SHARDING AND BATCHES ==============================
 *
 *   The five layers split like the trunk's: q/k/v and gate_up by output rows, o and down by input
 *   columns, with an all_reduce after each of those two branches (a partial sum without it drafts
 *   nonsense -- measured as position-0 acceptance collapsing from 61% to 5%). The block embeds
 *   from the trunk's vocab-sharded table, so its lookup is partial too: an exact all_reduce sums
 *   the ranks' rows after `embed_lookup`, as the trunk's own embedding does. The head is the
 *   trunk's vocab-sharded table, so the draft argmax is distributed too: `row_topk` over this
 *   rank's slice reports global ids into a pair plane, `all_gather` moves the pairs and
 *   `pair_argmax` picks the winner, which is exactly `rad_gemma4_dflash.h`'s walk. kTopR is 2
 *   because the gather moves 16-byte rows.
 *
 *   Every buffer is sized [max_seqs * steps, ...] and the predicting rows are gathered by
 *   `RadBatch::out_ids`, so a batch of sequences drafts side by side; the proposal plane is
 *   sequence-major and the engine reads it with `proposal_pitch = steps`.
 */
#ifndef RAD_BLOCK_MUSE_DFLASH_H
#define RAD_BLOCK_MUSE_DFLASH_H

#include "rad_arch.h"
#include "rad_fp8.h"   /* DenseFP8: the block fp8 / MXFP4 linear */

namespace rad {
namespace arch {

/* The builder refuses a declaration it cannot serve and says why in its own words; this names which
 * handle did not come back, so that message reads against a line of this file. An op that resolves
 * to no kernel is recorded and handed back null -- it is only an error if the block issues it -- so
 * without this a missing kernel surfaces as a bare "invalid argument" at startup. */
#define MD_REFUSED(...) ::rad::arch::muse_drafter_refused({ __VA_ARGS__ })
inline int muse_drafter_refused(std::initializer_list<const char*> what) {
    std::string list;
    for (const char* w : what) { if (!list.empty()) list += ", "; list += w; }
    std::fprintf(stderr, "Muse Glimmer drafter: the builder gave no handle for %s\n", list.c_str());
    return RAD_E_INVAL;
}

struct MuseDflash {
    enum { MAX_LAYERS = 8 };

    /* The drafter's config beyond its geometry. Everything here comes from the container's
     * `draft.` keys; nothing is defaulted quietly, because a silently ignored gate or window only
     * moves acceptance, which is the one number the user cannot see. */
    struct Config {
        int64_t n_taps  = 0;    /* how many trunk layers `fc` concatenates */
        int64_t block   = 0;    /* noise rows a sequence, and drafts (one per row) */
        int64_t rank    = 0;    /* markov_rank: the codebooks' width */
        int64_t window  = 0;    /* reference's anchor-aligned n_swa (configured history + 1) */
        int64_t rot     = 0;    /* rotary dimensions a head; 0 for all of head_dim */
        int     causal  = 0;
        int     tap_norm = 0;   /* `fc_norm`: one RMS gain a tap before the concatenation */
        int     qk_norm  = 0;   /* per-head q/k gains, and no output gate */
        int     gate     = 0;   /* one sigmoid gate a head on the attention output */
        const char* act = "gelu";   /* the feed-forward's activation */
        int64_t steps() const { return block - 1; }   /* the anchor row does not predict */
    };

    struct Wire {
        /* ---- the context pass, one row per committed token */
        rad_buf src = 0;    /* [max_tok, n_taps * n_embd]  the trunk's taps, read only */
        rad_buf srcn = 0;   /* [max_tok, n_taps * n_embd]  each tap normed, same layout */
        rad_buf c = 0, cn = 0; /* [max_tok, n_embd] fc output and separately normed context */
        rad_buf ck = 0, ckn = 0; /* [max_tok, kv_dim] projected and optionally normed K */
        rad_buf cv = 0;     /* [max_tok, kv_dim]           and its V */

        /* ---- the query pass, `block` rows per sequence */
        rad_buf x = 0;      /* [qrows, n_embd] f32   the drafter's residual stream */
        rad_buf delta = 0;  /* [qrows, n_embd] f32   the block a projection just wrote */
        rad_buf h = 0;      /* [qrows, n_embd] bf16  the normed stream, and every linear's output */
        rad_buf q = 0, k = 0, v = 0, qn = 0, kn = 0;
        rad_buf gate = 0, gaten = 0; /* [qrows, n_head] projection and sigmoid */
        rad_buf attn = 0, attng = 0; /* [qrows, q_dim] attention and gated attention */
        rad_buf gate_up = 0; rad_buf ff = 0;

        /* ---- fp8 codes and scales of the activations the linears read (`quant_act_fp8`), one
         * pair per activation rather than per linear: q, k and v all read the same `h`. The
         * scale plane is one f32 a 128-value block (`fp8_blocks`), which is the grid a
         * block-scaled fp8 GEMM reads (`DenseFP8`). */
        rad_buf srcq = 0, srcs = 0;  /* [max_tok, n_taps * n_embd] e4m3 | [.., /128] f32 */
        rad_buf cnq  = 0, cns  = 0;  /* [max_tok, n_embd]         the normed context, k/v's input */
        rad_buf hq   = 0, hs   = 0;  /* [qrows,   n_embd]         the normed stream */
        rad_buf atq  = 0, ats  = 0;  /* [qrows,   q_dim]          the attention output, o's input */
        rad_buf ffq  = 0, ffs  = 0;  /* [qrows,   n_ff]           the SwiGLU output, down's input */

        /* ---- the head and the walk, one row per draft position */
        rad_buf hg = 0;     /* [max_seqs*depth, n_embd] the predicting rows, gathered by out_ids */
        rad_buf cand = 0;   /* [srows, kTopR] i32  this rank's candidate ids, when sharded */
        rad_buf unary = 0;  /* [srows, kTopR] f32  their values */
        rad_buf dpair = 0;  /* [srows, kTopR, 2] f32  the gather's pair plane */
        rad_buf dgath = 0;  /* [srows, world * kTopR, 2] f32 */
        rad_buf dlog = 0;   /* [srows, n_vocab] f32 full-vocabulary draft logits */
        rad_buf dlog_all = 0; /* [srows, n_vocab_all] bf16, the ranks' logits gathered */
        rad_buf dlog_bf = 0; /* [srows, n_vocab] bf16, row_topk's operand */
        rad_buf best_score = 0; /* [max_seqs, ceil(n_vocab/64)] f32 partial argmax scores */
        rad_buf best_token = 0; /* [max_seqs, ceil(n_vocab/64)] i32 partial argmax ids */
        rad_buf tokens = 0; /* [max_seqs, steps] i32  the draft; the engine reads it by name */
    };

    Geom    g{};        /* the drafter's geometry, not the trunk's */
    Config  cfg{};
    Wire    w{};
    int     first_layer = 0;   /* absolute, for the KV group's layer_slot lookup */
    /* The short block this drafter also serves (rad_drafter_short_block): 4 rows, 3 drafts. The
     * whole block is staged and verified per sequence, so at several sequences its 13 extra rows
     * cost more than the drafts they add. */
    int64_t block_lo = 0;
    int64_t long_steps() const { return cfg.steps(); }
    int64_t qrows = 0, crows = 0, srows = 0;
    rad_kvgroup kv = 0;

    /* ---- model-scope weights */
    DenseFP8 w_fc{};                         /* block fp8 or MXFP4, per the recipe */
    rad_weight b_fc = 0;                     /* the same linear in bf16, when nothing quantised it */
    bool     quant = false;                  /* the container's linears are quantised */
    /* Test switch: declare the column-split linears (o, down) replicated instead of split plus an
     * all_reduce, to tell a wrong reduce from a wrong split. */
    static constexpr bool kRepOut = false;
    /* Test switch: at a vocab-sharded head, all-gather the draft logits and take a plain argmax
     * instead of the top-2 pair walk. Same answer by construction, so a difference is the walk's. */
    static constexpr bool kGatherLogits = true;
    rad_weight w_hnorm = 0, w_norm = 0;
    rad_weight w_fcnorm[MAX_LAYERS] = {};      /* one row of `fc_norm` a tap */
    rad_weight w_mk1 = 0, w_mk2 = 0;           /* markov_w1 (pred), markov_w2 (succ) */
    rad_weight w_tok = 0;                      /* the target's embedding table */
    rad_weight w_head = 0;                     /* the target's output table */

    /* ---- per-layer weights */
    rad_weight w_anorm[MAX_LAYERS] = {}, w_fnorm[MAX_LAYERS] = {};
    DenseFP8 w_q[MAX_LAYERS]{}, w_k[MAX_LAYERS]{}, w_v[MAX_LAYERS]{}, w_o[MAX_LAYERS]{};
    rad_weight b_q[MAX_LAYERS] = {}, b_k[MAX_LAYERS] = {}, b_v[MAX_LAYERS] = {}, b_o[MAX_LAYERS] = {};
    rad_weight w_qn[MAX_LAYERS] = {}, w_kn[MAX_LAYERS] = {}, w_gate[MAX_LAYERS] = {};
    DenseFP8 w_gu[MAX_LAYERS]{}, w_dn[MAX_LAYERS]{};
    rad_weight b_gu[MAX_LAYERS] = {}, b_dn[MAX_LAYERS] = {};

    /* ---- the context pass */
    rad_op op_tapn[MAX_LAYERS] = {};   /* one per tap, not per layer */
    rad_op op_cfc = 0, op_cnorm = 0;
    rad_op op_ck[MAX_LAYERS] = {}, op_cv[MAX_LAYERS] = {}, op_cknorm[MAX_LAYERS] = {};
    rad_op op_crope = 0, op_cstore = 0;

    /* ---- the query pass. The weightless ops are declared once and issued per layer; the ones
     * with weights are per layer, since a weight's prefetch interval and residency class come
     * from the op that declares it. */
    rad_op op_embed = 0, op_embed_ar = 0, op_widen = 0, op_add = 0;
    rad_op op_anorm[MAX_LAYERS] = {}, op_fnorm[MAX_LAYERS] = {}, op_outnorm = 0;
    rad_op op_qp[MAX_LAYERS] = {}, op_kp[MAX_LAYERS] = {}, op_vp[MAX_LAYERS] = {};
    rad_op op_qn[MAX_LAYERS] = {}, op_kn[MAX_LAYERS] = {}, op_gp[MAX_LAYERS] = {};
    rad_op op_op[MAX_LAYERS] = {}, op_gu[MAX_LAYERS] = {}, op_dn[MAX_LAYERS] = {};
    rad_op op_widen2 = 0, op_widen3 = 0;
    rad_op op_qrope = 0, op_krope = 0, op_store = 0, op_attn = 0;
    /* one activation quantiser an input */
    rad_op op_qsrc = 0, op_qcn = 0, op_qh = 0, op_qat = 0, op_qff = 0;
    /* The fused forms, where a kernel serves them: norm+quantise in one launch, and the
     * gate_up|silu|quantise fold. 0 falls back to the pair/triple. */
    /* Bisect switches: which of the two fusions this build takes. */
    rad_op op_gug[MAX_LAYERS] = {};
    int64_t gated_rows = 0;
    rad_op op_sig = 0, op_gate_heads = 0, op_act = 0;

    /* ---- the head and the walk */
    rad_op op_dgather_rows = 0, op_dlogits = 0, op_markov = 0, op_dlogbf = 0;
    rad_op op_dgather = 0, op_pick = 0;
    rad_op op_dgath_lg = 0, op_tok_all = 0;   /* the gathered-logits path, when declared */
    /* The column-split output projections leave each rank a partial sum, which the whole world
     * must add: the same all_reduce the trunk runs after its own attention and MLP branches. */
    rad_op op_ar = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int base_layer,
                 rad_kvgroup group, rad_weight tok_embd, const char* ck, rad_weight head_w,
                 const Wire& wire);
    /* The context pass, one row per committed token. */
    void context(RadCtx* c, const RadBatch* batch) const;
    /* The query pass, the head and the Markov walk, `block` rows per sequence. */
    void step(RadCtx* c, const RadBatch* batch) const;

private:
    int64_t q_dim()  const { return g.n_head * g.head_dim; }
    int64_t kv_dim() const { return g.n_head_kv * g.head_dim; }
};

/* ------------------------------------------------------------------------------ declare */

inline int MuseDflash::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c,
                               int base_layer, rad_kvgroup group, rad_weight tok_embd,
                               const char* ck, rad_weight head_w, const Wire& wire) {
    g = geom; cfg = c; w = wire; first_layer = base_layer; kv = group; w_tok = tok_embd;
    w_head = head_w;
    if (!ck || !w_head || !w_tok) return MD_REFUSED("ck", "w_head", "w_tok");
    if (g.n_layer <= 0 || g.n_layer > MAX_LAYERS) {
        fprintf(stderr, "radiance: the bf16 Muse Glimmer drafter has %lld layers; this block carries at "
                        "most %d\n", (long long)g.n_layer, MAX_LAYERS);
        return RAD_E_UNSUPPORTED;
    }
    if (cfg.n_taps <= 0 || cfg.n_taps > MAX_LAYERS || cfg.block < 2 || cfg.rank < 0)
        return RAD_E_INVAL;
    if (cfg.qk_norm && cfg.gate) {
        fprintf(stderr, "radiance: the drafter declares both per-head q/k norms and a per-head "
                        "output gate; no published drafter ships both, and this block would have "
                        "to guess which one the checkpoint trained with\n");
        return RAD_E_UNSUPPORTED;
    }
    /* At tp > 1 the head is vocab-sharded and the draft argmax runs over the ranks' slices
     * (row_topk with vocab_off -> all_gather -> pair_argmax): every rank holds the same global
     * top-1, so the proposal matches the replicated one. */

    const int64_t qd = q_dim(), kvd = kv_dim(), E = g.n_embd;
    crows = g.max_tok;
    qrows = g.max_seqs * cfg.block;
    if (qrows > g.max_tok) qrows = g.max_tok;
    srows = g.max_seqs * cfg.steps();       /* every row predicts */
    if (qrows <= 0 || srows <= 0 || cfg.rot < 0 || cfg.rot > g.head_dim || cfg.rot % 2)
        return RAD_E_INVAL;
    const int64_t rot = cfg.rot ? cfg.rot : g.head_dim;

    /* Vocab-sharded, the head's rows are this rank's slice and the top-1 must be taken over
     * every rank's slice. kTopR = 2 because the gather moves 16-byte rows. */
    const bool sharded = vocab_head_sharded(g);
    const int64_t kTopR = sharded ? 2 : 1;

    /* One op declaration per kernel shape, shared by the layers: `rmsnorm`'s and `gemm_nt`'s
     * parameters are geometry, and only the weights differ per layer. */
    auto gemm = [&](int64_t rows, int64_t n, int64_t k, rad_weight wt, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", n),
                                       RAD_INT("K", k), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(wt)),
                   {x}, {y});
    };
    /* A linear over a quantised weight: DenseFP8 picks `fp8a8` or `mxfp4a8` from the container's
     * encoding of that weight, and both read the same E4M3 activation pair. */
    auto qgemm = [&](int64_t rows, int64_t n, int64_t k, DenseFP8& wt,
                     rad_buf xq, rad_buf xs, rad_buf y) {
        return wt.gemm(b, rows, n, k, {xq, xs}, {y});
    };
    /* One quantiser per linear input. `h` takes two a layer, because the attention and
     * feed-forward norms each rewrite it. */
    auto qact = [&](rad_buf x, rad_buf xq, rad_buf xs, int64_t rows, int64_t n) {
        return rw(b, RAD_OP(b, "quant_act_fp8",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                       RAD_INT("group", RAD_FP8_BLOCK),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  {x}, {xq, xs});
    };

    /* ---------------------------------------------------------------- model-scope weights */
    {
        const char* wn = nm.f("dspark.fc.weight");
        RAD_ARCH_TRY(map_linear(b, wn, { nm.ckpt("%s.encoder.fc.weight", ck) }));
        RadEncoding enc{};
        /* `weight_enc` is true for a plain bf16 plane too, so ask which encoding: block fp8
         * (e4m3 codes, bf16 scale per 128x128), OCP-MXFP4 or M2XFP4 are what `DenseFP8` reads. */
        const RadEncPlane* sc = nullptr;
        const bool known = weight_enc(b, wn, &enc);
        const bool blk_fp8 = known && rad_enc_is(&enc, "affine") && enc.n_planes == 2 &&
                             !enc.transform[0] && enc.plane[0].dtype == RAD_F8E4M3 &&
                             (sc = rad_enc_plane(&enc, "scale")) &&
                             sc->block[0] == RAD_FP8_BLOCK && sc->block[1] == RAD_FP8_BLOCK;
        quant = blk_fp8 || (known && (linear_is_mxfp4(enc) || linear_is_m2xfp4(enc)));
        if (quant) {
            RAD_ARCH_TRY(w_fc.declare(b, nm, nm.f("dspark.fc"), E, cfg.n_taps * E,
                                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model(),
                                      { nm.ckpt("%s.encoder.fc.weight", ck) }, {}, true));
        } else {
            b_fc = decl_w(b, wn, RAD_BF16, {E, cfg.n_taps * E}, RAD_ACCESS_PER_TOKEN,
                          RAD_SHARD_NONE, grp_model());
        }
    }
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.hidden_norm.weight"),
                          nm.ckpt("%s.encoder.output_norm_enc.weight", ck)));
    w_hnorm = decl_w(b, nm.f("dspark.hidden_norm.weight"), RAD_BF16, {E},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.norm.weight"), nm.ckpt("%s.norm.weight", ck)));
    w_norm = decl_w(b, nm.f("dspark.norm.weight"), RAD_BF16, {E},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    if (!(quant ? (w_fc.w && w_fc.s) : b_fc) || !w_hnorm || !w_norm)
        return MD_REFUSED("w_fc", "w_hnorm", "w_norm");
    fprintf(stderr, "I speculation: 'muse_dflash' reads %s linears\n",
            quant ? "quantised (block fp8 / MXFP4)" : "bf16");

    /* `fc_norm` is [n_taps, n_embd]: one gain row a tap, read as that tap's slice. */
    if (cfg.tap_norm) {
        const char* src = nm.ckpt("%s.fc_norm.weight", ck);
        for (int64_t t = 0; t < cfg.n_taps; ++t) {
            const int i = (int)t;
            RAD_ARCH_TRY(map_rows(b, nm.f("dspark.fc_norm.%d.weight", i),
                                  { { src, (int)t, 1 } }));
            w_fcnorm[i] = decl_w(b, nm.f("dspark.fc_norm.%d.weight", i), RAD_BF16, {E},
                                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
            if (!w_fcnorm[i]) return MD_REFUSED("w_fcnorm[i]");
        }
    }

    /* The Markov codebooks, bf16, replicated over the whole vocabulary: both are [vocab, rank], so
     * `w2 @ w1[prev]` is `<w1[prev], w2[token]>`. RAD_ACCESS_VOCAB: each walk position reads one
     * predecessor row and scans every successor row. */
    /* The Markov codebooks exist only in drafters that were trained with one; the Muse assistant
     * has none, and `rank == 0` leaves the head a plain full-vocabulary argmax below. */
    if (cfg.rank > 0) {
        RAD_ARCH_TRY(map_copy(b, nm.f("dspark.markov_w1"),
                              nm.ckpt("%s.markov_head.markov_w1.weight", ck)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dspark.markov_w2"),
                              nm.ckpt("%s.markov_head.markov_w2.weight", ck)));
        w_mk1 = decl_w(b, nm.f("dspark.markov_w1"), RAD_BF16, {g.n_vocab_all, cfg.rank},
                       RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
        w_mk2 = decl_w(b, nm.f("dspark.markov_w2"), RAD_BF16, {g.n_vocab_all, cfg.rank},
                       RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
        if (!w_mk1 || !w_mk2) return MD_REFUSED("w_mk1", "w_mk2");
    }

    /* The target's output table under the drafter's own name: one stored tensor, two arch weights,
     * as rad_block_dspark.h does for its 2-bit copy. bf16 and replicated here, since the drafter
     * reads it as the head its full-vocabulary walk scores (see NOT DONE for sharding). */
    /* One stored tensor, two arch weights: the trunk's output table under the drafter's own name.
     * The encoding is whatever the recipe planned for the trunk's head (`output.weight`), so an
     * int8 or MXFP4 head is read by the same `logits_gemm` the trunk uses and nothing is stored
     * twice. */
    /* The head is the trunk's own declared output table, passed in: one stored tensor, read by the
     * same `logits_gemm` the trunk uses, so an int8 or MXFP4 head costs the drafter nothing and the
     * draft logits match the trunk's scale exactly. */

    /* ---------------------------------------------------------------- per-layer weights
     *
     * One name map and one declaration a weight, spelled out: the list is the readable form of
     * which checkpoint tensor feeds which matmul, and a drafter has few enough of them that a
     * table would only hide it. */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        const RadWeightGroup grp = grp_layer(first_layer + i);

        /* A norm gain: f32, read whole, never sharded. */
        auto norm_w = [&](rad_weight* out, const char* suffix, const char* from,
                          std::initializer_list<int64_t> shape) -> int {
            const char* name = nm.f("dspark.blk.%d.%s", i, suffix);
            RAD_ARCH_TRY(map_copy(b, name, nm.ckpt("%s.layers.%d.%s", ck, i, from)));
            *out = decl_w(b, name, RAD_BF16, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
            return *out ? RAD_OK : RAD_E_INVAL;
        };
        /* A linear, block fp8 or MXFP4 as the recipe wrote it (`DenseFP8::declare` reads which
         * from the container's encoding and maps the name itself). More than one source is a
         * concatenation, whose row parts say where each piece begins, so `act_mul` reads
         * [gate | up] as one weight. */
        auto lin_w = [&](DenseFP8* out, rad_weight* bout, const char* suffix,
                         std::initializer_list<const char*> srcs,
                         std::initializer_list<int64_t> shape,
                         std::initializer_list<int64_t> parts = {},
                         int shard = RAD_SHARD_NONE) -> int {
            int64_t N = 0, K = 0;
            auto it = shape.begin();
            if (it != shape.end()) N = *it++;
            if (it != shape.end()) K = *it;
            const char* base = nm.f("dspark.blk.%d.%s", i, suffix);
            const char* wn   = nm.f("%s.weight", base);
            RAD_ARCH_TRY(map_linear(b, wn, srcs));
            if (quant)
                return out->declare(b, nm, base, N, K, RAD_ACCESS_PER_TOKEN, shard, grp, srcs,
                                    parts, true);
            *bout = decl_w(b, wn, RAD_BF16, {N, K}, RAD_ACCESS_PER_TOKEN, shard, grp, 0, parts);
            return *bout ? RAD_OK : RAD_E_INVAL;
        };
        /* The output gate, when a drafter ships one, stays bf16: it is [n_head, n_embd] and no
         * recipe quantises it. */
        auto lin_b = [&](rad_weight* out, const char* suffix,
                         std::initializer_list<const char*> srcs,
                         std::initializer_list<int64_t> shape) -> int {
            const char* name = nm.f("dspark.blk.%d.%s", i, suffix);
            RAD_ARCH_TRY(map_copy(b, name, *srcs.begin()));
            *out = decl_w(b, name, RAD_BF16, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
            return *out ? RAD_OK : RAD_E_INVAL;
        };

#define SRC(path) nm.ckpt("%s.layers.%d." path, ck, i)
        RAD_ARCH_TRY(norm_w(&w_anorm[i], "attn_norm.weight", "input_layernorm.weight", {E}));
        RAD_ARCH_TRY(norm_w(&w_fnorm[i], "ffn_norm.weight", "post_attention_layernorm.weight", {E}));
        /* Row-split outputs shard by head; column-split inputs shard the residual axis. The
         * attention kernel reads a per-rank head slice, so q/k/v split rows and o splits its
         * input columns, which is the same plan the trunk's own layers take. */
        RAD_ARCH_TRY(lin_w(&w_q[i], &b_q[i], "attn_q", { SRC("self_attn.q_proj.weight") }, {qd, E}, {},
                           RAD_SHARD_ROW));
        RAD_ARCH_TRY(lin_w(&w_k[i], &b_k[i], "attn_k", { SRC("self_attn.k_proj.weight") }, {kvd, E}, {},
                           RAD_SHARD_ROW));
        RAD_ARCH_TRY(lin_w(&w_v[i], &b_v[i], "attn_v", { SRC("self_attn.v_proj.weight") }, {kvd, E}, {},
                           RAD_SHARD_ROW));
        RAD_ARCH_TRY(lin_w(&w_o[i], &b_o[i], "attn_output", { SRC("self_attn.o_proj.weight") }, {E, qd}, {},
                           kRepOut ? RAD_SHARD_NONE : RAD_SHARD_COL));
        RAD_ARCH_TRY(lin_w(&w_gu[i], &b_gu[i], "ffn_gate_up",
                           { SRC("mlp.gate_proj.weight"), SRC("mlp.up_proj.weight") },
                           {2 * g.n_ff, E}, { g.n_ff, g.n_ff }, RAD_SHARD_ROW));
        RAD_ARCH_TRY(lin_w(&w_dn[i], &b_dn[i], "ffn_down", { SRC("mlp.down_proj.weight") },
                           {E, g.n_ff}, {}, kRepOut ? RAD_SHARD_NONE : RAD_SHARD_COL));
        if (cfg.gate)
            RAD_ARCH_TRY(lin_b(&w_gate[i], "attn_gate", { SRC("self_attn.g_proj.weight") },
                               {g.n_head, E}));
        if (cfg.qk_norm) {
            RAD_ARCH_TRY(norm_w(&w_qn[i], "attn_q_norm.weight", "self_attn.q_norm.weight",
                                {g.head_dim}));
            RAD_ARCH_TRY(norm_w(&w_kn[i], "attn_k_norm.weight", "self_attn.k_norm.weight",
                                {g.head_dim}));
        }
#undef SRC
    }

    /* Non-causal: every query row sees its whole block and the context the trunk committed
     * (`causal` comes from `draft.is_causal`). Declared before both kv_stores, which read the
     * resolved `block_size` back with rad_kv_block_size(); before it, that returns zero. */
    /* `attn_paged` at causal=0 is the bidirectional block over a windowed key plane. The
     * anchor-window variant has no gqa 4 band at head_dim 128 (Muse's 32/8 split), and its
     * mask is anchor-aligned where the reference's is distance-aligned. */
    op_attn = rw(b, RAD_OP(b, "attn_paged",
                    RAD_PARAMS(RAD_RANGE("q_len", 1, cfg.block), RAD_INT("head_dim", g.head_dim),
                               RAD_INT("gqa", g.gqa()), RAD_INT("causal", cfg.causal),
                               RAD_INT("window", cfg.window), RAD_INT("n_head", g.n_head),
                               RAD_INT("block_size", RAD_KV_BLOCK),
                               RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                               RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", g.kv_dtype)),
                    RAD_NOWEIGHTS),
                {cfg.qk_norm ? w.qn : w.q}, {w.attn});
    if (!op_attn) return MD_REFUSED("op_attn");

    /* ---------------------------------------------------------------- the context pass
     *
     * One row per committed token, run once per trunk step: the taps become the drafter's own K
     * and V. Nothing here is per query row, so its cost follows the trunk's token count, not the
     * block's. */
    /* `fc_norm` exists only in drafters that norm each tap on its own; without it `fc` reads the
     * trunk's stream directly, so neither the per-tap norm nor its output buffer is declared. */
    if (cfg.tap_norm)
        for (int64_t t = 0; t < cfg.n_taps; ++t) {
            const int i = (int)t;
            op_tapn[i] = rw(b, RAD_OP(b, "rmsnorm",
                                      RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", E),
                                                 RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                                 RAD_STR("dtype", "bf16")),
                                      RAD_WEIGHTS(w_fcnorm[i])),
                            {w.src}, {w.srcn});
            if (!op_tapn[i]) return MD_REFUSED("op_tapn[i]");
        }
    /* The query pass quantises `h` after each norm and `ff` before down; `attn`'s quantiser is
     * declared in the query pass below, where step() issues it. */
    op_qh = qact(w.h, w.hq, w.hs, qrows, E);
    op_qff = qact(w.ff, w.ffq, w.ffs, qrows, g.n_ff);
    op_qsrc = qact(cfg.tap_norm ? w.srcn : w.src, w.srcq, w.srcs, crows, cfg.n_taps * E);
    op_cfc = quant ? qgemm(crows, E, cfg.n_taps * E, w_fc, w.srcq, w.srcs, w.c)
                   : gemm(crows, E, cfg.n_taps * E, b_fc, cfg.tap_norm ? w.srcn : w.src, w.c);
    op_cnorm = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", E),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w_hnorm)),
                   {w.c}, {w.cn});
    op_qcn = qact(w.cn, w.cnq, w.cns, crows, E);
    if (!op_qsrc || !op_qcn || !op_cfc || !op_cnorm)
        return MD_REFUSED("op_qsrc", "op_qcn", "op_cfc", "op_cnorm");

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        /* One op per weight: an op's declared weight list gives each weight its prefetch interval
         * and residency class. */
        op_ck[i] = quant ? qgemm(crows, kvd, E, w_k[i], w.cnq, w.cns, w.ck)
                         : gemm(crows, kvd, E, b_k[i], w.cn, w.ck);
        op_cv[i] = quant ? qgemm(crows, kvd, E, w_v[i], w.cnq, w.cns, w.cv)
                         : gemm(crows, kvd, E, b_v[i], w.cn, w.cv);
        if (!op_ck[i] || !op_cv[i]) return MD_REFUSED("op_ck[i]", "op_cv[i]");
        if (cfg.qk_norm) {
            op_cknorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                                        RAD_PARAMS(RAD_RANGE("M", 1, crows * g.n_head_kv),
                                                   RAD_INT("n", g.head_dim),
                                                   RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                                   RAD_STR("dtype", "bf16")),
                                        RAD_WEIGHTS(w_kn[i])),
                              {w.ck}, {w.ckn});
            if (!op_cknorm[i]) return MD_REFUSED("op_cknorm[i]");
        }
    }

    /* K-only rope: `rope` rotates the first n_head + n_head_kv heads, so a K-only tensor is
     * n_head = n_head_kv, n_head_kv = 0. Weightless, so one op serves every layer. Each context
     * row is rotated by its own absolute position, as the reference's single concatenated rope
     * does. */
    op_crope = rw(b, decl_rope(b, g, crows, g.head_dim, g.n_head_kv, 0, rot, "neox"),
                  {cfg.qk_norm ? w.ckn : w.ck}, {cfg.qk_norm ? w.ckn : w.ck});
    op_cstore = rw(b, RAD_OP(b, "kv_store",
                      RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                 RAD_INT("n_head_kv", g.n_head_kv),
                                 RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                 RAD_STR("kv_dtype", g.kv_dtype)),
                      RAD_NOWEIGHTS),
                  {cfg.qk_norm ? w.ckn : w.ck, w.cv}, {});
    if (!op_crope || !op_cstore) return MD_REFUSED("op_crope", "op_cstore");

    /* ---------------------------------------------------------------- the query pass
     *
     * `block` rows per sequence, through the drafter's layers. The residual stream is f32 and the
     * linears read bf16, exactly as the trunk keeps them: `rmsnorm` reads the stream and writes the
     * activations, `cast` widens a projection's output, and `add` keeps the sum in f32. */
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_embd", E),
                                RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
                                RAD_INT("vocab_offset", g.vocab_off)),
                     RAD_WEIGHTS(w_tok)),
                 {}, {w.h});
    /* The target's embedding table is vocab-sharded, so the lookup is partial: this rank's slice
     * writes its rows, `embed_lookup` writes zero rows for the rest, and the exact all-reduce
     * below sums the ranks into h -- the same reduce the trunk's own embedding runs, and dflash2's
     * and dspark's. Exact, because a lossy wire would move the one embedded anchor row the whole
     * block conditions on. */
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                            RAD_PARAMS(RAD_INT("world_size", g.world),
                                       RAD_RANGE("numel", E, qrows * E),
                                       RAD_STR("dtype", "bf16"), RAD_INT("exact", 1)),
                            RAD_NOWEIGHTS),
                        {w.h}, {w.h});
    op_widen = rw(b, RAD_OP(b, "cast",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", E),
                                       RAD_STR("from", "bf16"), RAD_STR("to", "f32")),
                            RAD_NOWEIGHTS),
                  {w.h}, {w.x});
    /* One widen and one add serve both residual sites a layer: same shapes, same buffers. */
    op_widen2 = rw(b, RAD_OP(b, "cast",
                             RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", E),
                                        RAD_STR("from", "bf16"), RAD_STR("to", "f32")),
                             RAD_NOWEIGHTS),
                   {w.h}, {w.delta});
    op_add = rw(b, RAD_OP(b, "add",
                          RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", E),
                                     RAD_STR("dtype", "f32")),
                          RAD_NOWEIGHTS),
                {w.x, w.delta}, {w.x});
    /* `attn`'s quantiser, declared here rather than with the others so the attention buffer's
     * declared life reaches into the query pass: declared with them, it ends before the stream's
     * begins, and the planner could lay `attn` over the residual the adds still read (the same
     * liveness rule the norms below follow). */
    op_qat = qact(cfg.gate ? w.attng : w.attn, w.atq, w.ats, qrows, qd);
    if (!op_embed || !op_widen || !op_widen2 || !op_add || !op_qat)
        return MD_REFUSED("op_embed", "op_widen", "op_widen2", "op_add", "op_qat");
    if (g.world > 1 && !op_embed_ar) return MD_REFUSED("op_embed_ar");

    /* Invariant: the norms are declared where step() issues them. Declaration order is the buffer
     * planner's liveness (core/build/rad_bufplan.cpp), and the stream `x` lives over the ops that
     * touch it: each layer's two norms in the loop, the final one after everything the layers
     * issue. Declared up front, the stream's lifetime would end before the layers and the planner
     * could overlay a feed-forward temporary on it. */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_anorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", E),
                                              RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_WEIGHTS(w_anorm[i])),
                         {w.x}, {w.h});
        op_qp[i] = quant ? qgemm(qrows, qd, E, w_q[i], w.hq, w.hs, w.q)
                         : gemm(qrows, qd, E, b_q[i], w.h, w.q);
        op_kp[i] = quant ? qgemm(qrows, kvd, E, w_k[i], w.hq, w.hs, w.k)
                         : gemm(qrows, kvd, E, b_k[i], w.h, w.k);
        op_vp[i] = quant ? qgemm(qrows, kvd, E, w_v[i], w.hq, w.hs, w.v)
                         : gemm(qrows, kvd, E, b_v[i], w.h, w.v);
        op_op[i] = quant ? qgemm(qrows, E, qd, w_o[i], w.atq, w.ats, w.h)
                         : gemm(qrows, E, qd, b_o[i], cfg.gate ? w.attng : w.attn, w.h);
        op_fnorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", E),
                                              RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_WEIGHTS(w_fnorm[i])),
                         {w.x}, {w.h});
        op_gu[i] = quant ? qgemm(qrows, 2 * g.n_ff, E, w_gu[i], w.hq, w.hs, w.gate_up)
                         : gemm(qrows, 2 * g.n_ff, E, b_gu[i], w.h, w.gate_up);
        /* One launch for gate_up, silu and the E4M3 quantise (`gemm_nt_q_gated` /
         * `gemm_nt_q_gated_mx`): it writes the quantised SwiGLU output down reads, so `w.ff`
         * stays for the fallback. */
        if (quant)
            op_gug[i] = (w_gu[i].mx || w_gu[i].m2x)
                        ? decl_gated_fold_mx(b, qrows, 2 * g.n_ff, E, w_gu[i].w, w_gu[i].s,
                                             w_gu[i].ref, {w.hq, w.hs}, {w.ffq, w.ffs},
                                             &gated_rows)
                        : decl_gated_fold(b, qrows, 2 * g.n_ff, E, w_gu[i].w, w_gu[i].s,
                                          {w.hq, w.hs}, {w.ffq, w.ffs}, &gated_rows, true);
        op_dn[i] = quant ? qgemm(qrows, E, g.n_ff, w_dn[i], w.ffq, w.ffs, w.h)
                         : gemm(qrows, E, g.n_ff, b_dn[i], w.ff, w.h);
        if (!op_anorm[i] || !op_qp[i] || !op_kp[i] || !op_vp[i] || !op_op[i] || !op_fnorm[i] ||
            !op_gu[i] || !op_dn[i])
            return RAD_E_INVAL;
        if (cfg.qk_norm) {
            op_qn[i] = rw(b, RAD_OP(b, "rmsnorm",
                                    RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head),
                                               RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                               RAD_F64("wadd", 0.0), RAD_STR("dtype", "bf16")),
                                    RAD_WEIGHTS(w_qn[i])),
                          {w.q}, {w.qn});
            op_kn[i] = rw(b, RAD_OP(b, "rmsnorm",
                                    RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head_kv),
                                               RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                               RAD_F64("wadd", 0.0), RAD_STR("dtype", "bf16")),
                                    RAD_WEIGHTS(w_kn[i])),
                          {w.k}, {w.kn});
            if (!op_qn[i] || !op_kn[i]) return MD_REFUSED("op_qn[i]", "op_kn[i]");
        }
        if (cfg.gate) {
            op_gp[i] = gemm(qrows, g.n_head, E, w_gate[i], w.h, w.gate);
            if (!op_gp[i]) return MD_REFUSED("op_gp[i]");
        }
    }

    op_qrope = rw(b, decl_rope(b, g, qrows, g.head_dim, g.n_head, 0, rot, "neox"),
                  {cfg.qk_norm ? w.qn : w.q}, {cfg.qk_norm ? w.qn : w.q});
    op_krope = rw(b, decl_rope(b, g, qrows, g.head_dim, g.n_head_kv, 0, rot, "neox"),
                  {cfg.qk_norm ? w.kn : w.k}, {cfg.qk_norm ? w.kn : w.k});
    op_store = rw(b, RAD_OP(b, "kv_store",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head_kv", g.n_head_kv),
                                RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {cfg.qk_norm ? w.kn : w.k, w.v}, {});
    /* The gate the trunk uses too: one sigmoid a head, then one scale a head's whole row. */
    if (cfg.gate) {
        op_sig = rw(b, RAD_OP(b, "sigmoid",
                              RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_head),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    {w.gate}, {w.gaten});
        op_gate_heads = rw(b, RAD_OP(b, "scale_rows",
                                     RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head),
                                                RAD_INT("n", g.head_dim), RAD_STR("act", "none"),
                                                RAD_STR("dtype", "bf16")),
                                     RAD_NOWEIGHTS),
                           {w.attn, w.gaten}, {w.attng});
        if (!op_sig || !op_gate_heads) return MD_REFUSED("op_sig", "op_gate_heads");
    }
    /* `act_mul` with the activation the drafter was trained with: exact-erf gelu for the Spark
     * drafts (`hidden_activation: gelu`), silu for a Qwen3-shaped one. `gelu_mul` is the tanh
     * approximation and is not what the reference runs. */
    op_act = rw(b, RAD_OP(b, "act_mul",
                          RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                                     RAD_STR("act", cfg.act), RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS),
                {w.gate_up}, {w.ff});
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                          RAD_PARAMS(RAD_INT("world_size", g.world),
                                     RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                     RAD_STR("dtype", "bf16"), RAD_INT("exact", g.wire_exact),
                                     RAD_INT("min_bytes", g.wire_min_bytes)),
                          RAD_NOWEIGHTS),
                    {w.h}, {w.h});
    op_outnorm = rw(b, RAD_OP(b, "rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", E),
                                         RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                         RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(w_norm)),
                    {w.x}, {w.h});
    if (!op_qrope || !op_krope || !op_store || !op_act || !op_outnorm)
        return MD_REFUSED("op_qrope", "op_krope", "op_store", "op_act", "op_outnorm");

    /* ---------------------------------------------------------------- the head and full-vocabulary walk
     *
     * Every row predicts, so the head scores all `steps` rows, anchor included. The row order is
     * sequence-major, which is the selector's [sequence, position, vocabulary] contract. The
     * greedy Markov op scans all vocabulary entries at every position and feeds each winning id
     * into the next position's bigram bias.
     */
    /* The block's 16 rows are [anchor, MASK x 15] and the anchor row predicts nothing: the engine
     * names the predicting rows in RadBatch::out_ids -- one entry per drafting sequence, `block`
     * staged rows, `out_ids` each entry's last `steps` rows (the scheduler's block_query) -- and
     * the gather below picks them out of the staged block. "The last S rows" of the block is only
     * the same list when EVERY staged sequence drafts; with several, it drops each sequence's
     * first mask row and reads its neighbour's anchor, whose logits then pose as that
     * sequence's first draft. */
    op_dgather_rows = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", E),
                                    RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                     {w.h}, {w.hg});
    op_dlogits = rw(b, RAD_OP(b, "logits_gemm",
                              RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n_vocab", g.n_vocab),
                                         RAD_INT("n_embd", E), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(w_head)),
                   {w.hg}, {w.dlog});
    if (cfg.rank > 0) {
        op_markov = rw(b, RAD_OP(b, "dflash_markov_greedy_f32",
                                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs),
                                            RAD_INT("steps", cfg.steps()), RAD_INT("rank", cfg.rank),
                                            RAD_INT("n_vocab", g.n_vocab_all),
                                            RAD_INT("anchor_stride", cfg.block),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_WEIGHTS(w_mk1, w_mk2)),
                      {w.dlog}, {w.best_score, w.best_token, w.tokens});
    } else {
        /* No bigram bias: the draft is the argmax of the target's own head over the predicting
         * rows, exactly `main_model_output_embeddings(...).argmax(-1)` in the reference. R = 1
         * writes one id a row, which is the [sequences, steps] proposal plane. */
        op_dlogbf = rw(b, RAD_OP(b, "cast",
                                 RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_vocab),
                                            RAD_STR("from", "f32"), RAD_STR("to", "bf16")),
                                 RAD_NOWEIGHTS),
                       {w.dlog}, {w.dlog_bf});
        if (sharded && kGatherLogits) {
            op_dgath_lg = rw(b, RAD_OP(b, "all_gather",
                            RAD_PARAMS(RAD_INT("world_size", g.world),
                                       RAD_INT("numel", srows * g.n_vocab),
                                       RAD_STR("dtype", "bf16"), RAD_INT("row", g.n_vocab)),
                            RAD_NOWEIGHTS),
                        {w.dlog_bf}, {w.dlog_all});
            if (op_dgath_lg)
                op_tok_all = rw(b, RAD_OP(b, "row_topk",
                                RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", g.n_vocab_all),
                                           RAD_INT("R", 1), RAD_INT("vocab_off", 0),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                            {w.dlog_all}, {w.tokens, w.best_score});
        }
        op_markov = sharded
            ? rw(b, RAD_OP(b, "row_topk",
                    RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", g.n_vocab),
                               RAD_INT("R", kTopR), RAD_INT("vocab_off", g.vocab_off),
                               RAD_STR("dtype", "bf16")),
                    RAD_NOWEIGHTS),
                 {w.dlog_bf}, {w.cand, w.unary, w.dpair})
            : rw(b, RAD_OP(b, "row_topk",
                    RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", g.n_vocab),
                               RAD_INT("R", 1), RAD_STR("dtype", "bf16")),
                    RAD_NOWEIGHTS),
                 {w.dlog_bf}, {w.tokens, w.best_score});
    }
    if (sharded) {
        /* Row-interleaved, so one draft position's pairs from every rank are adjacent. */
        op_dgather = rw(b, RAD_OP(b, "all_gather",
                        RAD_PARAMS(RAD_INT("world_size", g.world),
                                   RAD_INT("numel", srows * kTopR * 2), RAD_STR("dtype", "f32"),
                                   RAD_INT("row", kTopR * 2)),
                        RAD_NOWEIGHTS),
                    {w.dpair}, {w.dgath});
        op_pick = rw(b, RAD_OP(b, "pair_argmax",
                     RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.world * kTopR),
                                RAD_STR("dtype", "f32")),
                     RAD_NOWEIGHTS),
                 {w.dgath}, {w.tokens});
    }
    if (!op_dgather_rows || !op_dlogits || !op_markov || (c.rank <= 0 && !op_dlogbf) ||
        (sharded && (!op_dgather || !op_pick)))
        return MD_REFUSED("op_dgather_rows", "op_dlogits", "op_markov");

    return RAD_OK;
}

/* ------------------------------------------------------------------------------ the context pass */

inline void MuseDflash::context(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;
    const int64_t E = g.n_embd;

    /* Each tap's slice of the residual the trunk just tapped, normed with its own gain into the
     * same slice layout of one row-per-token buffer. */
    for (int64_t t = 0; t < cfg.n_taps; ++t) {
        const int i = (int)t;
        if (!cfg.tap_norm) continue;
        RAD_ISSUE_N(c, op_tapn[i], T, bcol(w.src, t * E, E, T), RAD_W(w_fcnorm[i]),
                    bcol(w.srcn, t * E, E, T));
    }
    if (quant) {
        RAD_ISSUE_N(c, op_qsrc, T, brows(cfg.tap_norm ? w.srcn : w.src, T), brows(w.srcq, T),
                    brows(w.srcs, T));
        RAD_ISSUE_N(c, op_cfc, T, brows(w.srcq, T), brows(w.srcs, T), RAD_W(w_fc.w),
                    RAD_W(w_fc.s), brows(w.c, T), RAD_NONE, w_fc.bref());
    } else {
        RAD_ISSUE_N(c, op_cfc, T, brows(cfg.tap_norm ? w.srcn : w.src, T), RAD_W(b_fc),
                    brows(w.c, T));
    }
    RAD_ISSUE_N(c, op_cnorm, T, brows(w.c, T), RAD_W(w_hnorm), brows(w.cn, T));
    if (quant)
        RAD_ISSUE_N(c, op_qcn, T, brows(w.cn, T), brows(w.cnq, T), brows(w.cns, T));

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        if (quant) {
            RAD_ISSUE_N(c, op_ck[i], T, brows(w.cnq, T), brows(w.cns, T), RAD_W(w_k[i].w),
                        RAD_W(w_k[i].s), brows(w.ck, T), RAD_NONE, w_k[i].bref());
            RAD_ISSUE_N(c, op_cv[i], T, brows(w.cnq, T), brows(w.cns, T), RAD_W(w_v[i].w),
                        RAD_W(w_v[i].s), brows(w.cv, T), RAD_NONE, w_v[i].bref());
        } else {
            RAD_ISSUE_N(c, op_ck[i], T, brows(w.cn, T), RAD_W(b_k[i]), brows(w.ck, T));
            RAD_ISSUE_N(c, op_cv[i], T, brows(w.cn, T), RAD_W(b_v[i]), brows(w.cv, T));
        }
        if (cfg.qk_norm)
            RAD_ISSUE_N(c, op_cknorm[i], T * g.n_head_kv, brows(w.ck, T), RAD_W(w_kn[i]),
                        brows(w.ckn, T));
        RAD_ISSUE_N(c, op_crope, T, brows(cfg.qk_norm ? w.ckn : w.ck, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_cstore, T, brows(cfg.qk_norm ? w.ckn : w.ck, T), brows(w.cv, T),
                    praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
    }
}

/* ------------------------------------------------------------------------------ the query pass */

inline void MuseDflash::step(RadCtx* c, const RadBatch* batch) const {
    const bool sharded = vocab_head_sharded(g);
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;
    const RadOperand pos = praw(batch->positions, RAD_I32, T);

    /* The block the core staged: the token the sampler just chose, then the mask token in every
     * unfilled slot. */
    RAD_ISSUE_N(c, op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok), brows(w.h, T));
    if (op_embed_ar) RAD_ISSUE_N(c, op_embed_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    RAD_ISSUE_N(c, op_widen, T, brows(w.h, T), brows(w.x, T));

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;

        RAD_ISSUE_N(c, op_anorm[i], T, brows(w.x, T), RAD_W(w_anorm[i]), brows(w.h, T));
        if (quant) {
            RAD_ISSUE_N(c, op_qh, T, brows(w.h, T), brows(w.hq, T), brows(w.hs, T));
            RAD_ISSUE_N(c, op_qp[i], T, brows(w.hq, T), brows(w.hs, T), RAD_W(w_q[i].w),
                        RAD_W(w_q[i].s), brows(w.q, T), RAD_NONE, w_q[i].bref());
            RAD_ISSUE_N(c, op_kp[i], T, brows(w.hq, T), brows(w.hs, T), RAD_W(w_k[i].w),
                        RAD_W(w_k[i].s), brows(w.k, T), RAD_NONE, w_k[i].bref());
            RAD_ISSUE_N(c, op_vp[i], T, brows(w.hq, T), brows(w.hs, T), RAD_W(w_v[i].w),
                        RAD_W(w_v[i].s), brows(w.v, T), RAD_NONE, w_v[i].bref());
        } else {
            RAD_ISSUE_N(c, op_qp[i], T, brows(w.h, T), RAD_W(b_q[i]), brows(w.q, T));
            RAD_ISSUE_N(c, op_kp[i], T, brows(w.h, T), RAD_W(b_k[i]), brows(w.k, T));
            RAD_ISSUE_N(c, op_vp[i], T, brows(w.h, T), RAD_W(b_v[i]), brows(w.v, T));
        }
        if (cfg.qk_norm) {
            RAD_ISSUE_N(c, op_qn[i], T * g.n_head, brows(w.q, T), RAD_W(w_qn[i]), brows(w.qn, T));
            RAD_ISSUE_N(c, op_kn[i], T * g.n_head_kv, brows(w.k, T), RAD_W(w_kn[i]),
                        brows(w.kn, T));
        }
        RAD_ISSUE_N(c, op_qrope, T, brows(cfg.qk_norm ? w.qn : w.q, T), pos);
        RAD_ISSUE_N(c, op_krope, T, brows(cfg.qk_norm ? w.kn : w.k, T), pos);
        /* The block's own K and V go into the cache past `seqused`, which is how every row of the
         * block attends to every other; the next context pass overwrites them, which is the
         * reference's `crop(start)`. */
        RAD_ISSUE_N(c, op_store, T, brows(cfg.qk_norm ? w.kn : w.k, T), brows(w.v, T), praw(slot, RAD_I32, T),
                    kv_cache(kv, first_layer + i));
        RAD_ISSUE_N(c, op_attn, batch->max_q_len, brows(cfg.qk_norm ? w.qn : w.q, T), kv_cache(kv, first_layer + i),
                    praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq), RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1), brows(w.attn, T));
        if (cfg.gate) {
            /* One gate a head, applied to that head's whole row of the attention output, before
             * o_proj. View [rows, heads * dim] and [rows, heads] as one row a head. */
            RAD_ISSUE_N(c, op_gp[i], T, brows(w.h, T), RAD_W(w_gate[i]), brows(w.gate, T));
            RAD_ISSUE_N(c, op_sig, T, brows(w.gate, T), brows(w.gaten, T));
            const RadOperand heads = RAD_P_T2(rad_buf_ptr(c, w.attn), RAD_BF16, T * g.n_head,
                                              g.head_dim);
            const RadOperand gates = RAD_P_T2(rad_buf_ptr(c, w.gaten), RAD_BF16, T * g.n_head, 1);
            const RadOperand gated = RAD_P_T2(rad_buf_ptr(c, w.attng), RAD_BF16, T * g.n_head,
                                              g.head_dim);
            RAD_ISSUE_N(c, op_gate_heads, T * g.n_head, heads, gates, RAD_NONE, gated);
        }
        if (quant) {
            RAD_ISSUE_N(c, op_qat, T, brows(cfg.gate ? w.attng : w.attn, T), brows(w.atq, T),
                        brows(w.ats, T));
            RAD_ISSUE_N(c, op_op[i], T, brows(w.atq, T), brows(w.ats, T), RAD_W(w_o[i].w),
                        RAD_W(w_o[i].s), brows(w.h, T), RAD_NONE, w_o[i].bref());
        } else {
            RAD_ISSUE_N(c, op_op[i], T, brows(cfg.gate ? w.attng : w.attn, T), RAD_W(b_o[i]),
                        brows(w.h, T));
        }
        if (op_ar && !kRepOut) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
        RAD_ISSUE_N(c, op_widen2, T, brows(w.h, T), brows(w.delta, T));
        RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.delta, T), brows(w.x, T));

        RAD_ISSUE_N(c, op_fnorm[i], T, brows(w.x, T), RAD_W(w_fnorm[i]), brows(w.h, T));
        if (quant)
            RAD_ISSUE_N(c, op_qh, T, brows(w.h, T), brows(w.hq, T), brows(w.hs, T));
        if (op_gug[i] && T <= gated_rows) {
            /* The MXFP4 fold takes the reference plane as a third weight; the fp8 tile does
             * not, and its schema is one operand shorter. */
            if (w_gu[i].mx || w_gu[i].m2x)
                RAD_ISSUE_N(c, op_gug[i], T, brows(w.hq, T), brows(w.hs, T), RAD_W(w_gu[i].w),
                            RAD_W(w_gu[i].s), RAD_W(w_gu[i].ref), brows(w.ffq, T),
                            brows(w.ffs, T), RAD_NONE);
            else
                RAD_ISSUE_N(c, op_gug[i], T, brows(w.hq, T), brows(w.hs, T), RAD_W(w_gu[i].w),
                            RAD_W(w_gu[i].s), brows(w.ffq, T), brows(w.ffs, T), RAD_NONE);
        } else if (quant) {
            RAD_ISSUE_N(c, op_gu[i], T, brows(w.hq, T), brows(w.hs, T), RAD_W(w_gu[i].w),
                        RAD_W(w_gu[i].s), brows(w.gate_up, T), RAD_NONE, w_gu[i].bref());
            RAD_ISSUE_N(c, op_act, T, bcol(w.gate_up, 0, g.n_ff, T),
                        bcol(w.gate_up, g.n_ff, g.n_ff, T), brows(w.ff, T));
            RAD_ISSUE_N(c, op_qff, T, brows(w.ff, T), brows(w.ffq, T), brows(w.ffs, T));
        } else {
            RAD_ISSUE_N(c, op_gu[i], T, brows(w.h, T), RAD_W(b_gu[i]), brows(w.gate_up, T));
            RAD_ISSUE_N(c, op_act, T, bcol(w.gate_up, 0, g.n_ff, T),
                        bcol(w.gate_up, g.n_ff, g.n_ff, T), brows(w.ff, T));
        }
        if (quant)
            RAD_ISSUE_N(c, op_dn[i], T, brows(w.ffq, T), brows(w.ffs, T), RAD_W(w_dn[i].w),
                        RAD_W(w_dn[i].s), brows(w.h, T), RAD_NONE, w_dn[i].bref());
        else
            RAD_ISSUE_N(c, op_dn[i], T, brows(w.ff, T), RAD_W(b_dn[i]), brows(w.h, T));
        if (op_ar && !kRepOut) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
        RAD_ISSUE_N(c, op_widen2, T, brows(w.h, T), brows(w.delta, T));
        RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.delta, T), brows(w.x, T));
    }

    RAD_ISSUE_N(c, op_outnorm, T, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));

    /* The predicting rows are the engine's `out_ids`: one entry per drafting sequence, `block`
     * staged rows, `out_ids` its last `steps` rows (the scheduler's block_query; the gather reads
     * them into `hg` and the head scores the gathered rows). Not "the last S rows" of the staged
     * block: that list coincides with out_ids only while EVERY staged sequence drafts -- with
     * several, it drops each sequence's first mask row and reads its neighbour's anchor, whose
     * logits then pose as that sequence's first draft and collapse a batch's acceptance (measured
     * at two sequences: the drafts came out shifted by one, the anchor's posing as draft 0). */
    const int64_t S = batch->n_out;
    if (S <= 0) return;
    RAD_ISSUE_N(c, op_dgather_rows, S, brows(w.h, T), praw(batch->out_ids, RAD_I32, S),
                brows(w.hg, S));
    RAD_ISSUE_N(c, op_dlogits, S, brows(w.hg, S), RAD_W(w_head), brows(w.dlog, S));
    if (cfg.rank > 0)
        RAD_ISSUE_N(c, op_markov, batch->n_seq, brows(w.dlog, S),
                    praw(batch->token_ids, RAD_I32, T), RAD_W(w_mk1), RAD_W(w_mk2),
                    brows(w.best_score, batch->n_seq), brows(w.best_token, batch->n_seq),
                    brows(w.tokens, batch->n_seq));
    else if (sharded) {
        RAD_ISSUE_N(c, op_dlogbf, S, brows(w.dlog, S), brows(w.dlog_bf, S));
        if (op_tok_all) {
            RAD_ISSUE_N(c, op_dgath_lg, S * g.n_vocab, brows(w.dlog_bf, S), brows(w.dlog_all, S));
            RAD_ISSUE_N(c, op_tok_all, S, brows(w.dlog_all, S), brows(w.tokens, S),
                        brows(w.best_score, S));
            return;
        }
        RAD_ISSUE_N(c, op_markov, S, brows(w.dlog_bf, S), brows(w.cand, S), brows(w.unary, S),
                    brows(w.dpair, S));
        RAD_ISSUE(c, op_dgather, brows(w.dpair, S), brows(w.dgath, S));
        RAD_ISSUE_N(c, op_pick, S, brows(w.dgath, S), brows(w.tokens, S));
    } else {
        RAD_ISSUE_N(c, op_dlogbf, S, brows(w.dlog, S), brows(w.dlog_bf, S));
        RAD_ISSUE_N(c, op_markov, S, brows(w.dlog_bf, S), brows(w.tokens, S),
                    brows(w.best_score, S));
    }
}

/* ================================================================== the trunk's side of it
 *
 * The taps, and one call a target plugin makes to put a drafter on. Both live here because both
 * are the drafter's contract with the trunk: which residual streams it reads, where its blocks
 * live, and how deep a proposal it makes. */

/* The residual streams a drafter reads, captured as the trunk passes them. */
struct MuseTaps {
    enum { MAX_TAPS = MuseDflash::MAX_LAYERS };

    int     n = 0;
    int     layer[MAX_TAPS] = {};   /* the trunk layer whose OUTPUT is tapped */
    rad_buf src = 0;                /* [rows, n * n_embd] bf16, one column slice a tap */
    rad_op  op = 0;                 /* the residual stream, copied into that slice */

    /* Read `draft.target_layer_ids`, which names the layers whose output is tapped. Zero taps when
     * the container carries no drafter, or one that reads no trunk stream. Out-of-range ids are
     * dropped, not clamped: a drafter naming a layer this model lacks belongs to another target. */
    int read(const RadModelMeta* meta, int64_t n_layer) {
        n = 0;
        const char* ids = rad_meta_gets(meta, "draft.target_layer_ids", nullptr);
        if (!ids) ids = rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr);
        if (!ids) ids = rad_meta_gets(meta, "draft.gguf_target_layers", nullptr);
        n = parse_layer_ids(ids, n_layer, layer, MAX_TAPS);
        return n;
    }

    /* The copy that saves a tap. The residual's dtype is the plugin's (`g.act_dtype`), so it is
     * passed in: a bf16 trunk copies bf16 to bf16 (the strided row copy), and an f32 one narrows. */
    int declare(RadBuilder* b, Names& nm, const Geom& g, int64_t rows, rad_buf residual,
                uint32_t res_dtype) {
        if (n <= 0) return RAD_OK;
        /* bf16, not the trunk's f32: `fc` is a bf16 GEMM, and the reference runs the drafter under
         * bf16 autocast, so the narrower tap is both the operand the GEMM reads and half the
         * memory (5 x 6656 x max_tok values are not cheap). */
        src = decl_b(b, nm.f("draft_src"), RAD_BF16, {rows, (int64_t)n * g.n_embd});
        if (!src) return MD_REFUSED("src");
        /* Declared over the trunk's own residual buffer: the op's operand pair is the stream and a
         * tap's slice, and every issue site passes the slice it is writing. */
        op = rw(b, RAD_OP(b, "cast",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                     RAD_STR("from", rad_dtype_name(res_dtype)),
                                     RAD_STR("to", "bf16")),
                          RAD_NOWEIGHTS),
                {residual}, {src});
        return op ? RAD_OK : RAD_E_INVAL;
    }

    /* The tap that follows trunk layer `l`, if any. Issued on prefill and decode alike: the
     * drafter conditions on every committed token's residual stream. */
    void step(RadCtx* c, rad_buf residual, int64_t rows, int64_t l, int64_t n_embd) const {
        if (!op) return;
        for (int t = 0; t < n; ++t)
            if (layer[t] == l)
                RAD_ISSUE(c, op, brows(residual, rows),
                          bcol(src, (int64_t)t * n_embd, n_embd, rows));
    }
};

/* Put a Muse Glimmer drafter on a trunk: its config from the container's `draft.` keys, its own KV
 * group, its graph, and the drafter declaration the core drives. `base_layer` is the absolute
 * layer its first block takes, one past the trunk's last.
 *
 * Nothing is defaulted quietly. A `draft.` key that describes a drafter this file does not
 * implement is refused by name, because a silently ignored gate, window or block length only moves
 * acceptance -- the one number a user cannot see in the output.
 *
 * The plugin calls it last, after its own head and logits: a drafter's passes run after the whole
 * trunk, and the buffer planner lays an op's lifetimes out in declaration order, so declaring it
 * earlier would let a trunk temporary land on the stream the drafter is still reading. */
inline int declare_muse_dflash(MuseDflash* ds, MuseDflash::Wire* w, rad_kvgroup* kv, RadBuilder* b,
                          Names& nm, const RadModelMeta* meta, const RadBuildCtx* ctx,
                          const Geom& target, const MuseTaps& taps, int base_layer,
                          rad_weight tok_embd, rad_weight head_w) {
    Geom dg{};
    RAD_ARCH_TRY(drafter_geom(dg, target, meta, "DFlash"));
    /* The width the drafter concatenates and projects is the trunk's, and `drafter_geom` keeps the
     * target's -- so a draft that disagrees is a draft for another model, and its `fc` would be
     * read at a shape it was not trained at. */
    const int64_t want_embd = rad_meta_geti(meta, "draft.hidden_size",
                                            rad_meta_geti(meta, "draft.dflash_config.hidden_size",
                                                          dg.n_embd));
    if (want_embd != dg.n_embd) {
        fprintf(stderr, "radiance: the drafter reads a %lld-wide trunk and this target is %lld "
                        "wide; they are not the same model\n", (long long)want_embd,
                (long long)dg.n_embd);
        return RAD_E_UNSUPPORTED;
    }

    MuseDflash::Config c{};
    c.n_taps  = taps.n;
    c.rank    = rad_meta_geti(meta, "draft.markov_rank", 0);
    c.causal  = rad_meta_geti(meta, "draft.is_causal", 0) ? 1 : 0;
    c.tap_norm = rad_meta_geti(meta, "draft.fc_norm", 0) ? 1 : 0;
    /* The Muse assistant always carries per-head q/k gains (its checkpoint has
     * `layers.N.self_attn.{q,k}_norm.weight` in every layer), and its config has no key to say so:
     * default them on rather than silently run a different attention. */
    c.qk_norm  = rad_meta_geti(meta, "draft.qk_norm", 1) ? 1 : 0;
    c.gate     = rad_meta_geti(meta, "draft.headwise_attn_output_gate", 0) ? 1 : 0;
    const char* act = rad_meta_gets(meta, "draft.hidden_act", "silu");
    c.act = (!std::strcmp(act, "gelu") || !std::strcmp(act, "none")) ? "gelu"
            : (!std::strcmp(act, "silu")) ? "silu" : nullptr;
    if (!c.act) {
        fprintf(stderr, "radiance: the drafter's feed-forward is '%s'; this block runs exact-erf "
                        "gelu (what the Spark drafts train with) or silu, and the two are not the "
                        "same function\n", act);
        return RAD_E_UNSUPPORTED;
    }
    const char* gate_mode = rad_meta_gets(meta, "draft.gate_attn_act_mode", "sigmoid");
    if (c.gate && std::strcmp(gate_mode, "sigmoid")) {
        fprintf(stderr, "radiance: the drafter gates its attention output with '%s'; this block "
                        "applies a sigmoid\n", gate_mode);
        return RAD_E_UNSUPPORTED;
    }
    /* Every row predicts, the anchor included. Without it a block row's worth of proposals is
     * missing and the head reads rows the drafter never filled. */
    if (!rad_meta_geti(meta, "draft.sample_from_anchor", 1)) {
        fprintf(stderr, "radiance: the drafter declares sample_from_anchor false, whose bonus "
                        "anchor slot this block has no rows for\n");
        return RAD_E_UNSUPPORTED;
    }
    /* The reference applies the bigram bias to the full vocabulary before each greedy choice. */
    if (c.rank < 0 || c.rank > 512 || (c.rank > 0 && c.rank % 32)) {
        fprintf(stderr, "radiance: the drafter's Markov head has rank %lld; a bigram walk takes a "
                        "multiple of 32 up to 512, and 0 means the checkpoint has none (the Muse "
                        "Glimmer assistant does not)\n", (long long)c.rank);
        return RAD_E_UNSUPPORTED;
    }

    /* The block is both the number of noise rows and the number of drafts: every row, the anchor
     * included, predicts the token after it, so the depth is `block_size` rather than the
     * `block_size - 1` a DFlash2 container's is. A block-diffusion drafter is trained for one
     * block length and has no partial block, so any other depth is refused. */
    const int64_t trained = rad_meta_geti(meta, "draft.block_size", 0);
    if (trained < 2) {
        fprintf(stderr, "radiance: this drafter declares no draft.block_size; the number of masked "
                        "positions it denoises is not a number this engine may choose.\n");
        return RAD_E_FORMAT;
    }
    if ((long long)ctx->max_spec != trained - 1) {
        fprintf(stderr, "radiance: the drafter denoises a block of %lld positions, of which %lld "
                        "predict, and this serve asks for %d. A block-diffusion drafter has a fixed "
                        "block; run it at %lld.\n",
                (long long)trained, (long long)trained - 1, ctx->max_spec,
                (long long)trained - 1);
        return RAD_E_UNSUPPORTED;
    }
    c.block = trained;

    /* The reference gives every row of the block the anchor's history. `window` is the configured
     * number of preceding positions; llama.cpp sets n_swa to window + 1 and masks when
     * anchor - key >= n_swa. Keep that exact width in attention, the KV group and scheduler. */
    int64_t window = 0;
    /* Its config names `layer_types: sliding_attention` and a `sliding_window`, but no
     * `use_sliding_window` flag: take the window whenever the drafter states one. */
    const int64_t sw = rad_meta_geti(meta, "draft.sliding_window",
                                     rad_meta_geti(meta, "draft.dflash_config.swa_window_size", 0));
    if (rad_meta_geti(meta, "draft.use_sliding_window", sw > 0 ? 1 : 0))
        window = sw;
    if (window < 0 || (window && window < c.block)) {
        fprintf(stderr, "radiance: the drafter's %lld-token window does not cover the %lld-token "
                        "block it denoises\n", (long long)window, (long long)c.block);
        return RAD_E_UNSUPPORTED;
    }
    if (window > 0 && window == INT64_MAX) return RAD_E_INVAL;
    /* The reference masks with |q - k| <= sliding_window, with no anchor-aligned +1. */
    c.window = window;
    const double prf = rad_meta_getf(meta, "draft.partial_rotary_factor",
                                     rad_meta_getf(meta,
                                         "draft.rope_parameters.partial_rotary_factor", 1.0));
    const double rot = (double)dg.head_dim * prf;
    if (!(prf > 0.0) && prf != 0.0) {
        fprintf(stderr, "radiance: the drafter's partial_rotary_factor is %g\n", prf);
        return RAD_E_INVAL;
    }
    if (std::floor(rot) != rot || (int64_t)rot % 2 || rot > (double)dg.head_dim) {
        fprintf(stderr, "radiance: the drafter rotates %g of %lld head dimensions\n", rot,
                (long long)dg.head_dim);
        return RAD_E_INVAL;
    }
    c.rot = (int64_t)rot ? (int64_t)rot : dg.head_dim;

    RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, dg, c.window, base_layer, kv));

    /* Activations: one row per committed token for the context pass, one row per block position
     * for the query pass. Both are small; the drafter's cost is its weights. */
    const int64_t qd = dg.n_head * dg.head_dim, kvd = dg.n_head_kv * dg.head_dim;
    int64_t qr = dg.max_seqs * c.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr = dg.max_seqs * c.steps();
    MuseDflash::Wire& dw = *w;
    dw.src   = taps.src;
    if (c.tap_norm)
        dw.srcn = decl_b(b, nm.f("ds_srcn"), RAD_BF16, {dg.max_tok, c.n_taps * dg.n_embd});
    dw.c     = decl_b(b, nm.f("ds_c"),  RAD_BF16, {dg.max_tok, dg.n_embd});
    dw.cn    = decl_b(b, nm.f("ds_cn"), RAD_BF16, {dg.max_tok, dg.n_embd});
    dw.ck    = decl_b(b, nm.f("ds_ck"), RAD_BF16, {dg.max_tok, kvd});
    dw.cv    = decl_b(b, nm.f("ds_cv"), RAD_BF16, {dg.max_tok, kvd});
    if (c.qk_norm) dw.ckn = decl_b(b, nm.f("ds_ckn"), RAD_BF16, {dg.max_tok, kvd});
    dw.x     = decl_b(b, nm.f("ds_x"), RAD_F32, {qr, dg.n_embd});
    dw.delta = decl_b(b, nm.f("ds_delta"), RAD_F32, {qr, dg.n_embd});
    dw.h     = decl_b(b, nm.f("ds_h"), RAD_BF16, {qr, dg.n_embd});
    dw.q     = decl_b(b, nm.f("ds_q"), RAD_BF16, {qr, qd});
    dw.k     = decl_b(b, nm.f("ds_k"), RAD_BF16, {qr, kvd});
    dw.v     = decl_b(b, nm.f("ds_v"), RAD_BF16, {qr, kvd});
    if (c.qk_norm) {
        dw.qn = decl_b(b, nm.f("ds_qn"), RAD_BF16, {qr, qd});
        dw.kn = decl_b(b, nm.f("ds_kn"), RAD_BF16, {qr, kvd});
    }
    dw.gate  = decl_b(b, nm.f("ds_gate"), RAD_BF16, {qr, dg.n_head});
    dw.attn  = decl_b(b, nm.f("ds_attn"), RAD_BF16, {qr, qd});
    if (c.gate) {
        dw.gaten = decl_b(b, nm.f("ds_gaten"), RAD_BF16, {qr, dg.n_head});
        dw.attng = decl_b(b, nm.f("ds_attng"), RAD_BF16, {qr, qd});
    }
    dw.gate_up = decl_b(b, nm.f("ds_gate_up"), RAD_BF16, {qr, 2 * dg.n_ff});
    dw.ff    = decl_b(b, nm.f("ds_ff"), RAD_BF16, {qr, dg.n_ff});
    /* The linears are block fp8 or MXFP4, so every activation feeding one has an E4M3 plane and
     * an f32 scale a 128 values. */
    dw.srcq = decl_b(b, nm.f("ds_srcq"), RAD_F8E4M3, {dg.max_tok, c.n_taps * dg.n_embd});
    dw.srcs = decl_b(b, nm.f("ds_srcs"), RAD_F32, {dg.max_tok, fp8_blocks(c.n_taps * dg.n_embd)});
    dw.cnq  = decl_b(b, nm.f("ds_cnq"),  RAD_F8E4M3, {dg.max_tok, dg.n_embd});
    dw.cns  = decl_b(b, nm.f("ds_cns"),  RAD_F32,    {dg.max_tok, fp8_blocks(dg.n_embd)});
    dw.hq   = decl_b(b, nm.f("ds_hq"),   RAD_F8E4M3, {qr, dg.n_embd});
    dw.hs   = decl_b(b, nm.f("ds_hs"),   RAD_F32,    {qr, fp8_blocks(dg.n_embd)});
    dw.atq  = decl_b(b, nm.f("ds_atq"),  RAD_F8E4M3, {qr, qd});
    dw.ats  = decl_b(b, nm.f("ds_ats"),  RAD_F32,    {qr, fp8_blocks(qd)});
    dw.ffq  = decl_b(b, nm.f("ds_ffq"),  RAD_F8E4M3, {qr, dg.n_ff});
    dw.ffs  = decl_b(b, nm.f("ds_ffs"),  RAD_F32,    {qr, fp8_blocks(dg.n_ff)});
    dw.hg    = decl_b(b, nm.f("ds_hg"), RAD_BF16, {sr, dg.n_embd});
    dw.dlog  = decl_b(b, nm.f("ds_dlog"), RAD_F32, {sr, dg.n_vocab});
    /* Vocab-sharded, the head's rows are this rank's slice and the top-1 must be taken over every
     * rank's slice: row_topk reports global ids into a pair plane, all_gather moves the pairs, and
     * pair_argmax picks the winner -- the same walk rad_gemma4_dflash.h runs. kTopR = 2 because the
     * gather moves 16-byte rows. */
    const bool sharded = vocab_head_sharded(dg);
    const int64_t kTopR = sharded ? 2 : 1;
    if (sharded) {
        dw.cand   = decl_b(b, nm.f("ds_cand"),  RAD_I32, {sr, kTopR});
        dw.unary  = decl_b(b, nm.f("ds_unary"), RAD_F32, {sr, kTopR});
        dw.dpair  = decl_b(b, nm.f("ds_dpair"), RAD_F32, {sr, kTopR, 2});
        dw.dgath  = decl_b(b, nm.f("ds_dgath"), RAD_F32, {sr, (int64_t)dg.world * kTopR, 2});
    }
    /* row_topk serves bf16 only (its band list), and logits_gemm writes f32: the draft's argmax
     * reads this narrower copy. One row's scores are 202048 values, so the pair costs 18 MB a
     * draft step against the ~50 MB of head GEMM the same step runs. */
    if (c.rank <= 0) {
        dw.dlog_bf = decl_b(b, nm.f("ds_dlog_bf"), RAD_BF16, {sr, dg.n_vocab});
        if (vocab_head_sharded(dg))
            dw.dlog_all = decl_b(b, nm.f("ds_dlog_all"), RAD_BF16, {sr, dg.n_vocab_all});
    }
    const int64_t n_markov_tiles = (dg.n_vocab_all + 63) / 64;
    /* row_topk's outputs are [M, R]: one row a predicting row, so the no-markov path lays the
     * proposal and its score out as [rows, 1], which is the same bytes as [max_seqs, steps] in
     * sequence-major order and what RadDrafterDecl::proposal_pitch reads. */
    dw.best_score = c.rank > 0
                        ? decl_b(b, nm.f("ds_markov_score"), RAD_F32, {dg.max_seqs, n_markov_tiles})
                        : decl_b(b, nm.f("ds_markov_score"), RAD_F32, {sr, 1});
    dw.best_token = decl_b(b, nm.f("ds_markov_token"), RAD_I32,
                           {dg.max_seqs, n_markov_tiles});
    /* The proposal buffer (RadDrafterDecl::proposal): a block pass proposes through the drafter's
     * own walk, never through the sampler's token buffer. */
    dw.tokens = c.rank > 0
                    ? decl_b(b, nm.f("ds_tokens"), RAD_I32, {dg.max_seqs, c.steps()})
                    : decl_b(b, nm.f("ds_tokens"), RAD_I32, {sr, 1});
    if ((c.tap_norm && !dw.srcn) || !dw.c || !dw.cn || !dw.ck || !dw.cv ||
        (c.qk_norm && (!dw.ckn || !dw.qn || !dw.kn)) ||
        (c.gate && (!dw.gaten || !dw.attng)) || !dw.x || !dw.delta || !dw.h || !dw.q || !dw.k ||
        !dw.v || !dw.gate || !dw.attn || !dw.gate_up || !dw.ff || !dw.hg || !dw.dlog ||
        !dw.best_score || !dw.best_token || !dw.tokens || !dw.srcq || !dw.srcs || !dw.cnq ||
        !dw.cns || !dw.hq || !dw.hs || !dw.atq || !dw.ats || !dw.ffq || !dw.ffs)
        return RAD_E_INVAL;

    RAD_ARCH_TRY(ds->declare(b, nm, dg, c, base_layer, *kv, tok_embd, "draft", head_w, dw));
    fprintf(stderr, "I speculation: 'muse_dflash' %s its gate_up|silu|quant fold (%lld row(s))\n",
            ds->gated_rows ? "fuses" : "does not fuse", (long long)ds->gated_rows);

    RadDrafterDecl drf{};
    drf.name           = "muse_dflash";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = c.block;
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = c.steps();
    drf.window         = c.window;
    drf.mask_token     = (int32_t)rad_meta_geti(meta, "draft.mask_token_id", -1);
    if (drf.mask_token < 0)
        drf.mask_token = (int32_t)rad_meta_geti(meta, "draft.dflash_config.mask_token_id", -1);
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: this drafter declares no mask_token_id, which every unfilled "
                        "block position embeds\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));
    /* A short block, so the core can drop from 16 rows to 4 once several sequences decode: every op
     * above is ranged by rows, so the same graph serves both. Declared after the drafter, as the
     * ABI requires. */
    if (c.block > 4) {
        ds->block_lo = 4;
        RAD_ARCH_TRY(rad_drafter_short_block(b, ds->block_lo));
    }

    rad_note(b, "Muse Glimmer drafter: %lld layers at absolute %d..%d, %lld heads / %lld kv x %lld, "
                "ffn %lld, %s attention with a %lld-token anchor-aligned window (%lld configured "
                "history + 1), %lld of %lld rotary dimensions, %s, %s a head, %d trunk tap(s); one "
                "forward over %lld noise row(s), the anchor included, proposes %lld tokens",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head, (long long)dg.n_head_kv, (long long)dg.head_dim,
             (long long)dg.n_ff, c.causal ? "causal" : "non-causal", (long long)c.window,
             (long long)window, (long long)c.rot, (long long)dg.head_dim, c.act,
             c.gate ? "sigmoid gate" : "no gate", (int)c.n_taps, (long long)c.block,
             (long long)c.steps());
    rad_note(b, "the rank-%lld Markov bigram head greedily scans all %lld vocabulary tokens at "
                "each draft position; confidence is not applied to logits%s",
             (long long)c.rank, (long long)dg.n_vocab_all,
             rad_meta_geti(meta, "draft.has_confidence_head", 0)
                 ? " (its reference role is adaptive depth, which this fixed-depth graph does not use)"
                 : "");
    return RAD_OK;
}


}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MUSE_DFLASH_H */