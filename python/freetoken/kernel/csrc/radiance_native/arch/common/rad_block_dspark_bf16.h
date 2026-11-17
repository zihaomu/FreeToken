/* rad_block_dspark_bf16.h -- the DSpark drafter over bf16 weights, at the widths a Spark layer
 * uses: a few plain attention layers that denoise a whole block of masked positions in one pass,
 * conditioned on the trunk's residual stream at a handful of layers, with a rank-N Markov head
 * over the target's own output table.
 *
 * ============================== WHY A SECOND DSPARK BLOCK ==============================
 *
 * rad_block_dspark.h is the same drafter shape for a block-fp8 release: its linears are DenseFP8
 * behind `quant_act_fp8`, its feed-forward is `silu_mul`, its norms carry q/k gains, and its draft
 * head is a 2-bit copy of the target's. The published Spark drafts (`dflash` GGUFs, and the HF
 * Spark drafter releases behind them) ship the other way: bf16 linears, no q/k norms, one sigmoid
 * gate a head on the attention output, exact-erf GELU, one RMS gain per tap, and a sliding window.
 * Those are different ops, not different parameters, so this block declares its own graph and
 * uses an explicit anchor-window attention op and a full-vocabulary Markov selector so neither
 * semantic distinction is silently folded into the generic per-row window or candidate-lattice ops.
 *
 * Nothing here is quantised. The drafter is 624M parameters read whole a draft pass, so bf16 costs
 * 1.2 GiB of residency and one extra weight pass a step; quantising it is a separate measurement
 * (it can only move acceptance, since every draft is verified by the target).
 *
 * ============================== THE FORWARD PASS ==============================
 *
 * The reference is llama.cpp's `llama_model_dflash::graph<false>` (XHToken/llama.cpp
 * 22c0ae1871b, the branch the Spark draft cards name), which mirrors the HF export:
 *
 *     tap(l)   = rmsnorm(residual after trunk layer l) * fc_norm[l]      one gain row per tap
 *     ctx      = rmsnorm(fc(concat_l tap(l)))                            hidden_norm
 *     per draft layer (x5):
 *         h_n  = input_layernorm(h)
 *         q,k,v = q_proj/k_proj/v_proj(h_n);  k_ctx, v_ctx = k_proj/v_proj(ctx)
 *         rope(q, k);  attention over [the layer's context K/V, this block] non-causally,
 *                      inside a `window` positions
 *         h += o_proj(sigmoid(g_proj(h)) * attn)          one gate a head, no q/k norms
 *         h += mlp(post_attention_layernorm(h))           gelu(gate) * up, then down
 *     logits = target.lm_head(output_norm(h)) + markov_w2 @ markov_w1[prev]
 *
 * The two projections of `ctx` are the drafter's own k_proj and v_proj, one pair per layer, and
 * they run over every committed token; the drafter's layers never see the context rows. That is
 * the context pass below, and the block above is the query pass -- the same split
 * rad_block_dspark.h makes, and the same `RAD_DRAFT_BLOCK` drive.
 *
 * ============================== THE TAPS ARE NORMED, NOT JUST READ ==================
 *
 * `fc_norm` is [n_taps, n_embd]: the residual entering each tap layer is normed with its own gain
 * before the concatenation, so `fc` sees n_taps unit-scaled streams instead of one sum whose
 * magnitudes the trunk happens to have. One `rmsnorm` per tap over that tap's column slice of the
 * tap buffer, into the same slice layout of a normed copy; `fc` then reads one contiguous
 * [n_taps * n_embd] row.
 *
 * ============================== THE SHARED ANCHOR WINDOW ==========================
 *
 * The pinned llama.cpp oracle increments the configured history from 512 to 513 and repeats the
 * anchor row's mask for every row in the denoising block. `attn_paged_anchor_window` implements
 * that exact predicate: every row sees keys through the end of the staged context whose absolute
 * position is less than 513 behind the anchor. The query row does not move the lower cutoff. The
 * same 513-token width is passed to the KV group and scheduler, so rollback, cache recycling and
 * block staging all retain the exact history the drafter reads.
 *
 * ============================== ROW ALIGNMENT ==============================
 *
 * Every row predicts, the anchor included (`dflash.sample_from_anchor`, which the GGUF import
 * refuses when false): row j of [anchor, MASK x (block-1)] proposes the token at the anchor's
 * position plus j + 1, so `steps()` is `block` and the head runs over all the block's rows. The
 * core stages the block against `RadDrafterDecl::block` and runs the head over the last `depth`
 * rows, which covers this and Dflash2's shift-by-one convention (abi/rad_builder.h).
 *
 * ============================== THE MARKOV HEAD ==============================
 *
 * The draft logits get a learned bigram bias
 *
 *     score[l, token] = logits[l, token] + markov_w2[token] . markov_w1[prev(l)]
 *     prev(0) = the anchor; prev(l) = the greedy token chosen at l - 1
 *
 * `dflash_markov_greedy` scans the full vocabulary at every position and greedily chooses the
 * largest biased score, following the structure of the pinned reference's serial Markov walk.
 * Base logits are F32; reduction order and tie behavior can still differ from the reference. It
 * does not prune to the base head's top K, use a joint-path heap, or apply confidence to logits.
 *
 * The confidence head (`conf_proj`, one sigmoid a position predicting acceptance) is imported and
 * ignored: it exists to let a serving loop cut the block short, this engine's depth is fixed at
 * declare, and it changes nothing the drafter proposes.
 *
 * ============================== PRECISION ==============================
 *
 * bf16 for every weight and activation except the residual stream, which is f32 exactly as the
 * Spark trunk keeps it (rad_block_dspark.h's fp8 graphs fold the residual into the norm; here the
 * norms read the f32 stream and write bf16, and `add` keeps the sum in f32). The reference runs
 * bf16 weights against an f32 residual, so rounding the drafter's stream to bf16 would be a third
 * opinion on every layer's output for no saved bytes worth having at `block` rows.
 *
 * ============================== NOT DONE ==============================
 *
 *   The head and full-vocabulary Markov scan are read replicated, so a vocab-sharded deployment
 *   cannot draft with them. The plugin refuses `world > 1` rather than scoring only one rank's
 *   slice of the vocabulary.
 *
 *   The drafter's linears are four GEMMs and one attention a layer with no folding: the norms are
 *   plain `rmsnorm`, not a fused norm-add-quantise, because nothing here quantises. At seven rows
 *   a pass the launches, not the bytes, are the cost.
 */
#ifndef RAD_BLOCK_DSPARK_BF16_H
#define RAD_BLOCK_DSPARK_BF16_H

#include "rad_arch.h"

namespace rad {
namespace arch {

/* The builder refuses a declaration it cannot serve and says why in its own words; this names which
 * handle did not come back, so that message reads against a line of this file. An op that resolves
 * to no kernel is recorded and handed back null -- it is only an error if the block issues it -- so
 * without this a missing kernel surfaces as a bare "invalid argument" at startup. */
#define DS_REFUSED(...) ::rad::arch::drafter_refused({ __VA_ARGS__ })
inline int drafter_refused(std::initializer_list<const char*> what) {
    std::string list;
    for (const char* w : what) { if (!list.empty()) list += ", "; list += w; }
    std::fprintf(stderr, "DSpark drafter: the builder gave no handle for %s\n", list.c_str());
    return RAD_E_INVAL;
}

struct DsparkBF16 {
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
        int64_t steps() const { return block; }   /* every row predicts */
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

        /* ---- the head and the walk, one row per draft position */
        rad_buf dlog = 0;   /* [srows, n_vocab] f32 full-vocabulary draft logits */
        rad_buf best_score = 0; /* [max_seqs, ceil(n_vocab/64)] f32 partial argmax scores */
        rad_buf best_token = 0; /* [max_seqs, ceil(n_vocab/64)] i32 partial argmax ids */
        rad_buf tokens = 0; /* [max_seqs, steps] i32  the draft; the engine reads it by name */
    };

    Geom    g{};        /* the drafter's geometry, not the trunk's */
    Config  cfg{};
    Wire    w{};
    int     first_layer = 0;   /* absolute, for the KV group's layer_slot lookup */
    int64_t qrows = 0, crows = 0, srows = 0;
    rad_kvgroup kv = 0;

    /* ---- model-scope weights */
    rad_weight w_fc = 0, w_hnorm = 0, w_norm = 0;
    rad_weight w_fcnorm[MAX_LAYERS] = {};      /* one row of `fc_norm` a tap */
    rad_weight w_mk1 = 0, w_mk2 = 0;           /* markov_w1 (pred), markov_w2 (succ) */
    rad_weight w_tok = 0;                      /* the target's embedding table */
    rad_weight w_head = 0;                     /* the target's output table */

    /* ---- per-layer weights */
    rad_weight w_anorm[MAX_LAYERS] = {}, w_fnorm[MAX_LAYERS] = {};
    rad_weight w_q[MAX_LAYERS] = {}, w_k[MAX_LAYERS] = {}, w_v[MAX_LAYERS] = {}, w_o[MAX_LAYERS] = {};
    rad_weight w_qn[MAX_LAYERS] = {}, w_kn[MAX_LAYERS] = {}, w_gate[MAX_LAYERS] = {};
    rad_weight w_gu[MAX_LAYERS] = {}, w_dn[MAX_LAYERS] = {};

    /* ---- the context pass */
    rad_op op_tapn[MAX_LAYERS] = {};   /* one per tap, not per layer */
    rad_op op_cfc = 0, op_cnorm = 0;
    rad_op op_ck[MAX_LAYERS] = {}, op_cv[MAX_LAYERS] = {}, op_cknorm[MAX_LAYERS] = {};
    rad_op op_crope = 0, op_cstore = 0;

    /* ---- the query pass. The weightless ops are declared once and issued per layer; the ones
     * with weights are per layer, since a weight's prefetch interval and residency class come
     * from the op that declares it. */
    rad_op op_embed = 0, op_widen = 0, op_add = 0;
    rad_op op_anorm[MAX_LAYERS] = {}, op_fnorm[MAX_LAYERS] = {}, op_outnorm = 0;
    rad_op op_qp[MAX_LAYERS] = {}, op_kp[MAX_LAYERS] = {}, op_vp[MAX_LAYERS] = {};
    rad_op op_qn[MAX_LAYERS] = {}, op_kn[MAX_LAYERS] = {}, op_gp[MAX_LAYERS] = {};
    rad_op op_op[MAX_LAYERS] = {}, op_gu[MAX_LAYERS] = {}, op_dn[MAX_LAYERS] = {};
    rad_op op_widen2 = 0, op_widen3 = 0;
    rad_op op_qrope = 0, op_krope = 0, op_store = 0, op_attn = 0;
    rad_op op_sig = 0, op_gate_heads = 0, op_act = 0;

    /* ---- the head and the walk */
    rad_op op_dlogits = 0, op_markov = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int base_layer,
                 rad_kvgroup group, rad_weight tok_embd, const char* ck, const char* ck_head,
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

inline int DsparkBF16::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c,
                               int base_layer, rad_kvgroup group, rad_weight tok_embd,
                               const char* ck, const char* ck_head, const Wire& wire) {
    g = geom; cfg = c; w = wire; first_layer = base_layer; kv = group; w_tok = tok_embd;
    if (!ck || !ck_head || !w_tok) return DS_REFUSED("ck", "ck_head", "w_tok");
    if (g.n_layer <= 0 || g.n_layer > MAX_LAYERS) {
        fprintf(stderr, "radiance: the bf16 DSpark drafter has %lld layers; this block carries at "
                        "most %d\n", (long long)g.n_layer, MAX_LAYERS);
        return RAD_E_UNSUPPORTED;
    }
    if (cfg.n_taps <= 0 || cfg.n_taps > MAX_LAYERS || cfg.block < 2 || cfg.rank <= 0)
        return RAD_E_INVAL;
    if (cfg.qk_norm && cfg.gate) {
        fprintf(stderr, "radiance: the drafter declares both per-head q/k norms and a per-head "
                        "output gate; no published drafter ships both, and this block would have "
                        "to guess which one the checkpoint trained with\n");
        return RAD_E_UNSUPPORTED;
    }
    /* The full-vocabulary head and Markov scan are read replicated; partial-vocabulary drafting
     * would need an exact distributed argmax and replicated codebook access. */
    if (g.world > 1) {
        fprintf(stderr, "radiance: the bf16 DSpark drafter reads the target's whole output table "
                        "and cannot draft at tp %d; run it at tp 1\n", g.world);
        return RAD_E_UNSUPPORTED;
    }

    const int64_t qd = q_dim(), kvd = kv_dim(), E = g.n_embd;
    crows = g.max_tok;
    qrows = g.max_seqs * cfg.block;
    if (qrows > g.max_tok) qrows = g.max_tok;
    srows = g.max_seqs * cfg.steps();       /* every row predicts */
    if (qrows <= 0 || srows <= 0 || cfg.rot < 0 || cfg.rot > g.head_dim || cfg.rot % 2)
        return RAD_E_INVAL;
    const int64_t rot = cfg.rot ? cfg.rot : g.head_dim;

    /* One op declaration per kernel shape, shared by the layers: `rmsnorm`'s and `gemm_nt`'s
     * parameters are geometry, and only the weights differ per layer. */
    auto gemm = [&](int64_t rows, int64_t n, int64_t k, rad_weight wt, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", n),
                                       RAD_INT("K", k), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(wt)),
                   {x}, {y});
    };

    /* ---------------------------------------------------------------- model-scope weights */
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.fc.weight"), nm.ckpt("%s.fc.weight", ck)));
    w_fc = decl_w(b, nm.f("dspark.fc.weight"), RAD_BF16, {E, cfg.n_taps * E},
                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.hidden_norm.weight"),
                          nm.ckpt("%s.hidden_norm.weight", ck)));
    w_hnorm = decl_w(b, nm.f("dspark.hidden_norm.weight"), RAD_BF16, {E},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.norm.weight"), nm.ckpt("%s.norm.weight", ck)));
    w_norm = decl_w(b, nm.f("dspark.norm.weight"), RAD_BF16, {E},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    if (!w_fc || !w_hnorm || !w_norm) return DS_REFUSED("w_fc", "w_hnorm", "w_norm");

    /* `fc_norm` is [n_taps, n_embd]: one gain row a tap, read as that tap's slice. */
    if (cfg.tap_norm) {
        const char* src = nm.ckpt("%s.fc_norm.weight", ck);
        for (int64_t t = 0; t < cfg.n_taps; ++t) {
            const int i = (int)t;
            RAD_ARCH_TRY(map_rows(b, nm.f("dspark.fc_norm.%d.weight", i),
                                  { { src, (int)t, 1 } }));
            w_fcnorm[i] = decl_w(b, nm.f("dspark.fc_norm.%d.weight", i), RAD_BF16, {E},
                                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
            if (!w_fcnorm[i]) return DS_REFUSED("w_fcnorm[i]");
        }
    }

    /* The Markov codebooks, bf16, replicated over the whole vocabulary: both are [vocab, rank], so
     * `w2 @ w1[prev]` is `<w1[prev], w2[token]>`. RAD_ACCESS_VOCAB: each walk position reads one
     * predecessor row and scans every successor row. */
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.markov_w1"),
                          nm.ckpt("%s.markov_head.markov_w1.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.markov_w2"),
                          nm.ckpt("%s.markov_head.markov_w2.weight", ck)));
    w_mk1 = decl_w(b, nm.f("dspark.markov_w1"), RAD_BF16, {g.n_vocab_all, cfg.rank},
                   RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    w_mk2 = decl_w(b, nm.f("dspark.markov_w2"), RAD_BF16, {g.n_vocab_all, cfg.rank},
                   RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    if (!w_mk1 || !w_mk2) return DS_REFUSED("w_mk1", "w_mk2");

    /* The target's output table under the drafter's own name: one stored tensor, two arch weights,
     * as rad_block_dspark.h does for its 2-bit copy. bf16 and replicated here, since the drafter
     * reads it as the head its full-vocabulary walk scores (see NOT DONE for sharding). */
    RAD_ARCH_TRY(map_copy(b, nm.f("dspark.head.weight"), ck_head));
    w_head = decl_w(b, nm.f("dspark.head.weight"), RAD_BF16, {g.n_vocab_all, E},
                    RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    if (!w_head) return DS_REFUSED("w_head");

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
        /* A linear: bf16. More than one source is a concatenation, whose row parts say where each
         * piece begins (RadWeightDecl::row_parts), so `act_mul` can read [gate | up] as one
         * weight. */
        auto lin_w = [&](rad_weight* out, const char* suffix,
                         std::initializer_list<const char*> srcs,
                         std::initializer_list<int64_t> shape,
                         std::initializer_list<int64_t> parts = {}) -> int {
            const char* name = nm.f("dspark.blk.%d.%s", i, suffix);
            if (srcs.size() > 1) RAD_ARCH_TRY(map_concat(b, name, srcs));
            else                 RAD_ARCH_TRY(map_copy(b, name, *srcs.begin()));
            *out = decl_w(b, name, RAD_BF16, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp, 0,
                          parts);
            return *out ? RAD_OK : RAD_E_INVAL;
        };

#define SRC(path) nm.ckpt("%s.layers.%d." path, ck, i)
        RAD_ARCH_TRY(norm_w(&w_anorm[i], "attn_norm.weight", "input_layernorm.weight", {E}));
        RAD_ARCH_TRY(norm_w(&w_fnorm[i], "ffn_norm.weight", "post_attention_layernorm.weight", {E}));
        RAD_ARCH_TRY(lin_w(&w_q[i], "attn_q", { SRC("self_attn.q_proj.weight") }, {qd, E}));
        RAD_ARCH_TRY(lin_w(&w_k[i], "attn_k", { SRC("self_attn.k_proj.weight") }, {kvd, E}));
        RAD_ARCH_TRY(lin_w(&w_v[i], "attn_v", { SRC("self_attn.v_proj.weight") }, {kvd, E}));
        RAD_ARCH_TRY(lin_w(&w_o[i], "attn_output", { SRC("self_attn.o_proj.weight") }, {E, qd}));
        RAD_ARCH_TRY(lin_w(&w_gu[i], "ffn_gate_up",
                           { SRC("mlp.gate_proj.weight"), SRC("mlp.up_proj.weight") },
                           {2 * g.n_ff, E}, { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(lin_w(&w_dn[i], "ffn_down", { SRC("mlp.down_proj.weight") },
                           {E, g.n_ff}));
        if (cfg.gate)
            RAD_ARCH_TRY(lin_w(&w_gate[i], "attn_gate", { SRC("self_attn.g_proj.weight") },
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
    op_attn = rw(b, RAD_OP(b, cfg.window > 0 ? "attn_paged_anchor_window" : "attn_paged",
                    RAD_PARAMS(RAD_RANGE("q_len", 1, cfg.block), RAD_INT("head_dim", g.head_dim),
                               RAD_INT("gqa", g.gqa()), RAD_INT("causal", cfg.causal),
                               RAD_INT("window", cfg.window), RAD_INT("n_head", g.n_head),
                               RAD_INT("block_size", RAD_KV_BLOCK),
                               RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                               RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", g.kv_dtype)),
                    RAD_NOWEIGHTS),
                {cfg.qk_norm ? w.qn : w.q}, {w.attn});
    if (!op_attn) return DS_REFUSED("op_attn");

    /* ---------------------------------------------------------------- the context pass
     *
     * One row per committed token, run once per trunk step: the taps become the drafter's own K
     * and V. Nothing here is per query row, so its cost follows the trunk's token count, not the
     * block's. */
    for (int64_t t = 0; t < cfg.n_taps; ++t) {
        const int i = (int)t;
        op_tapn[i] = rw(b, RAD_OP(b, "rmsnorm",
                                  RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", E),
                                             RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                             RAD_STR("dtype", "bf16")),
                                  RAD_WEIGHTS(w_fcnorm[i])),
                        {w.src}, {w.srcn});
        if (!op_tapn[i]) return DS_REFUSED("op_tapn[i]");
    }
    op_cfc = gemm(crows, E, cfg.n_taps * E, w_fc, w.srcn, w.c);
    op_cnorm = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", E),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w_hnorm)),
                   {w.c}, {w.cn});
    if (!op_cfc || !op_cnorm) return DS_REFUSED("op_cfc", "op_cnorm");

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        /* One op per weight: an op's declared weight list gives each weight its prefetch interval
         * and residency class. */
        op_ck[i] = gemm(crows, kvd, E, w_k[i], w.cn, w.ck);
        op_cv[i] = gemm(crows, kvd, E, w_v[i], w.cn, w.cv);
        if (!op_ck[i] || !op_cv[i]) return DS_REFUSED("op_ck[i]", "op_cv[i]");
        if (cfg.qk_norm) {
            op_cknorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                                        RAD_PARAMS(RAD_RANGE("M", 1, crows * g.n_head_kv),
                                                   RAD_INT("n", g.head_dim),
                                                   RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                                   RAD_STR("dtype", "bf16")),
                                        RAD_WEIGHTS(w_kn[i])),
                              {w.ck}, {w.ckn});
            if (!op_cknorm[i]) return DS_REFUSED("op_cknorm[i]");
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
    if (!op_crope || !op_cstore) return DS_REFUSED("op_crope", "op_cstore");

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
    if (!op_embed || !op_widen || !op_widen2 || !op_add)
        return DS_REFUSED("op_embed", "op_widen", "op_widen2", "op_add");

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
        op_qp[i] = gemm(qrows, qd, E, w_q[i], w.h, w.q);
        op_kp[i] = gemm(qrows, kvd, E, w_k[i], w.h, w.k);
        op_vp[i] = gemm(qrows, kvd, E, w_v[i], w.h, w.v);
        op_op[i] = gemm(qrows, E, qd, w_o[i], cfg.gate ? w.attng : w.attn, w.h);
        op_fnorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", E),
                                              RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_WEIGHTS(w_fnorm[i])),
                         {w.x}, {w.h});
        op_gu[i] = gemm(qrows, 2 * g.n_ff, E, w_gu[i], w.h, w.gate_up);
        op_dn[i] = gemm(qrows, E, g.n_ff, w_dn[i], w.ff, w.h);
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
            if (!op_qn[i] || !op_kn[i]) return DS_REFUSED("op_qn[i]", "op_kn[i]");
        }
        if (cfg.gate) {
            op_gp[i] = gemm(qrows, g.n_head, E, w_gate[i], w.h, w.gate);
            if (!op_gp[i]) return DS_REFUSED("op_gp[i]");
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
        if (!op_sig || !op_gate_heads) return DS_REFUSED("op_sig", "op_gate_heads");
    }
    /* `act_mul` with the activation the drafter was trained with: exact-erf gelu for the Spark
     * drafts (`hidden_activation: gelu`), silu for a Qwen3-shaped one. `gelu_mul` is the tanh
     * approximation and is not what the reference runs. */
    op_act = rw(b, RAD_OP(b, "act_mul",
                          RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                                     RAD_STR("act", cfg.act), RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS),
                {w.gate_up}, {w.ff});
    op_outnorm = rw(b, RAD_OP(b, "rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", E),
                                         RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                         RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(w_norm)),
                    {w.x}, {w.h});
    if (!op_qrope || !op_krope || !op_store || !op_act || !op_outnorm)
        return DS_REFUSED("op_qrope", "op_krope", "op_store", "op_act", "op_outnorm");

    /* ---------------------------------------------------------------- the head and full-vocabulary walk
     *
     * Every row predicts, so the head scores all `steps` rows, anchor included. The row order is
     * sequence-major, which is the selector's [sequence, position, vocabulary] contract. The
     * greedy Markov op scans all vocabulary entries at every position and feeds each winning id
     * into the next position's bigram bias.
     */
    op_dlogits = rw(b, RAD_OP(b, "logits_gemm",
                              RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n_vocab", g.n_vocab),
                                         RAD_INT("n_embd", E), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(w_head)),
                   {w.h}, {w.dlog});
    op_markov = rw(b, RAD_OP(b, "dflash_markov_greedy_f32",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs),
                                        RAD_INT("steps", cfg.steps()), RAD_INT("rank", cfg.rank),
                                        RAD_INT("n_vocab", g.n_vocab_all),
                                        RAD_INT("anchor_stride", cfg.block),
                                        RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w_mk1, w_mk2)),
                  {w.dlog}, {w.best_score, w.best_token, w.tokens});
    if (!op_dlogits || !op_markov) return DS_REFUSED("op_dlogits", "op_markov");

    return RAD_OK;
}

/* ------------------------------------------------------------------------------ the context pass */

inline void DsparkBF16::context(RadCtx* c, const RadBatch* batch) const {
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
    RAD_ISSUE_N(c, op_cfc, T, cfg.tap_norm ? brows(w.srcn, T) : brows(w.src, T), RAD_W(w_fc),
                brows(w.c, T));
    RAD_ISSUE_N(c, op_cnorm, T, brows(w.c, T), RAD_W(w_hnorm), brows(w.cn, T));

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        RAD_ISSUE_N(c, op_ck[i], T, brows(w.cn, T), RAD_W(w_k[i]), brows(w.ck, T));
        RAD_ISSUE_N(c, op_cv[i], T, brows(w.cn, T), RAD_W(w_v[i]), brows(w.cv, T));
        if (cfg.qk_norm)
            RAD_ISSUE_N(c, op_cknorm[i], T * g.n_head_kv, brows(w.ck, T), RAD_W(w_kn[i]),
                        brows(w.ckn, T));
        RAD_ISSUE_N(c, op_crope, T, brows(cfg.qk_norm ? w.ckn : w.ck, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_cstore, T, brows(cfg.qk_norm ? w.ckn : w.ck, T), brows(w.cv, T),
                    praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
    }
}

/* ------------------------------------------------------------------------------ the query pass */

inline void DsparkBF16::step(RadCtx* c, const RadBatch* batch) const {
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
    RAD_ISSUE_N(c, op_widen, T, brows(w.h, T), brows(w.x, T));

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;

        RAD_ISSUE_N(c, op_anorm[i], T, brows(w.x, T), RAD_W(w_anorm[i]), brows(w.h, T));
        RAD_ISSUE_N(c, op_qp[i], T, brows(w.h, T), RAD_W(w_q[i]), brows(w.q, T));
        RAD_ISSUE_N(c, op_kp[i], T, brows(w.h, T), RAD_W(w_k[i]), brows(w.k, T));
        RAD_ISSUE_N(c, op_vp[i], T, brows(w.h, T), RAD_W(w_v[i]), brows(w.v, T));
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
        RAD_ISSUE_N(c, op_op[i], T, brows(cfg.gate ? w.attng : w.attn, T), RAD_W(w_o[i]), brows(w.h, T));
        RAD_ISSUE_N(c, op_widen2, T, brows(w.h, T), brows(w.delta, T));
        RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.delta, T), brows(w.x, T));

        RAD_ISSUE_N(c, op_fnorm[i], T, brows(w.x, T), RAD_W(w_fnorm[i]), brows(w.h, T));
        RAD_ISSUE_N(c, op_gu[i], T, brows(w.h, T), RAD_W(w_gu[i]), brows(w.gate_up, T));
        RAD_ISSUE_N(c, op_act, T, bcol(w.gate_up, 0, g.n_ff, T),
                    bcol(w.gate_up, g.n_ff, g.n_ff, T), brows(w.ff, T));
        RAD_ISSUE_N(c, op_dn[i], T, brows(w.ff, T), RAD_W(w_dn[i]), brows(w.h, T));
        RAD_ISSUE_N(c, op_widen2, T, brows(w.h, T), brows(w.delta, T));
        RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.delta, T), brows(w.x, T));
    }

    RAD_ISSUE_N(c, op_outnorm, T, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));

    /* The predicting rows, named by the scheduler through `out_ids`. */
    const int64_t S = batch->n_out;
    if (S <= 0) return;
    RAD_ISSUE_N(c, op_dlogits, S, brows(w.h, S), RAD_W(w_head), brows(w.dlog, S));
    RAD_ISSUE_N(c, op_markov, batch->n_seq, brows(w.dlog, S),
                praw(batch->token_ids, RAD_I32, T), RAD_W(w_mk1), RAD_W(w_mk2),
                brows(w.best_score, batch->n_seq), brows(w.best_token, batch->n_seq),
                brows(w.tokens, batch->n_seq));
}

/* ================================================================== the trunk's side of it
 *
 * The taps, and one call a target plugin makes to put a drafter on. Both live here because both
 * are the drafter's contract with the trunk: which residual streams it reads, where its blocks
 * live, and how deep a proposal it makes. */

/* The residual streams a drafter reads, captured as the trunk passes them. */
struct DsparkTaps {
    enum { MAX_TAPS = DsparkBF16::MAX_LAYERS };

    int     n = 0;
    int     layer[MAX_TAPS] = {};   /* the trunk layer whose OUTPUT is tapped */
    rad_buf src = 0;                /* [rows, n * n_embd] f32, one column slice a tap */
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

    /* The copy that saves a tap. A drafter reads the trunk's own residual stream, which is f32
     * here and stays f32: the widening the fp8 graphs do is theirs, not this one's. */
    int declare(RadBuilder* b, Names& nm, const Geom& g, int64_t rows, rad_buf residual) {
        if (n <= 0) return RAD_OK;
        src = decl_b(b, nm.f("draft_src"), RAD_F32, {rows, (int64_t)n * g.n_embd});
        if (!src) return DS_REFUSED("src");
        /* Declared over the trunk's own residual buffer: the op's operand pair is the stream and a
         * tap's slice, and every issue site passes the slice it is writing. */
        op = rw(b, RAD_OP(b, "cast",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                     RAD_STR("from", "f32"), RAD_STR("to", "f32")),
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

/* Put a DSpark drafter on a trunk: its config from the container's `draft.` keys, its own KV
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
inline int declare_dspark(DsparkBF16* ds, DsparkBF16::Wire* w, rad_kvgroup* kv, RadBuilder* b,
                          Names& nm, const RadModelMeta* meta, const RadBuildCtx* ctx,
                          const Geom& target, const DsparkTaps& taps, int base_layer,
                          rad_weight tok_embd, const char* ck_head) {
    Geom dg{};
    RAD_ARCH_TRY(drafter_geom(dg, target, meta, "DSpark"));
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

    DsparkBF16::Config c{};
    c.n_taps  = taps.n;
    c.rank    = rad_meta_geti(meta, "draft.markov_rank", 0);
    c.causal  = rad_meta_geti(meta, "draft.is_causal", 0) ? 1 : 0;
    c.tap_norm = rad_meta_geti(meta, "draft.fc_norm", 0) ? 1 : 0;
    c.qk_norm  = rad_meta_geti(meta, "draft.qk_norm", 0) ? 1 : 0;
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
    if (c.rank <= 0 || c.rank % 32 || c.rank > 512) {
        fprintf(stderr, "radiance: the drafter's Markov head has rank %lld; the walk that applies "
                        "the bigram bias takes a multiple of 32 up to 512\n", (long long)c.rank);
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
    if ((long long)ctx->max_spec != trained) {
        fprintf(stderr, "radiance: the drafter was trained at block_size %lld, which is %lld draft "
                        "tokens, and this serve asks for %d. A block-diffusion drafter denoises a "
                        "fixed number of mask positions; run it at %lld.\n",
                (long long)trained, (long long)trained, ctx->max_spec, (long long)trained);
        return RAD_E_UNSUPPORTED;
    }
    c.block = trained;

    /* The reference gives every row of the block the anchor's history. `window` is the configured
     * number of preceding positions; llama.cpp sets n_swa to window + 1 and masks when
     * anchor - key >= n_swa. Keep that exact width in attention, the KV group and scheduler. */
    int64_t window = 0;
    if (rad_meta_geti(meta, "draft.use_sliding_window", 0))
        window = rad_meta_geti(meta, "draft.sliding_window", 0);
    if (window < 0 || (window && window < c.block)) {
        fprintf(stderr, "radiance: the drafter's %lld-token window does not cover the %lld-token "
                        "block it denoises\n", (long long)window, (long long)c.block);
        return RAD_E_UNSUPPORTED;
    }
    if (window > 0 && window == INT64_MAX) return RAD_E_INVAL;
    c.window = window > 0 ? window + 1 : 0;  /* llama.cpp dflash: n_swa += 1 */
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
    DsparkBF16::Wire& dw = *w;
    dw.src   = taps.src;
    dw.srcn  = decl_b(b, nm.f("ds_srcn"), RAD_BF16, {dg.max_tok, c.n_taps * dg.n_embd});
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
    dw.dlog  = decl_b(b, nm.f("ds_dlog"), RAD_F32, {sr, dg.n_vocab});
    const int64_t n_markov_tiles = (dg.n_vocab_all + 63) / 64;
    dw.best_score = decl_b(b, nm.f("ds_markov_score"), RAD_F32,
                           {dg.max_seqs, n_markov_tiles});
    dw.best_token = decl_b(b, nm.f("ds_markov_token"), RAD_I32,
                           {dg.max_seqs, n_markov_tiles});
    /* The proposal buffer (RadDrafterDecl::proposal): a block pass proposes through the drafter's
     * own walk, never through the sampler's token buffer. */
    dw.tokens = decl_b(b, nm.f("ds_tokens"), RAD_I32, {dg.max_seqs, c.steps()});
    if (!dw.srcn || !dw.c || !dw.cn || !dw.ck || !dw.cv ||
        (c.qk_norm && (!dw.ckn || !dw.qn || !dw.kn)) ||
        (c.gate && (!dw.gaten || !dw.attng)) || !dw.x || !dw.delta || !dw.h || !dw.q || !dw.k ||
        !dw.v || !dw.gate || !dw.attn || !dw.gate_up || !dw.ff || !dw.dlog ||
        !dw.best_score || !dw.best_token || !dw.tokens)
        return RAD_E_INVAL;

    RAD_ARCH_TRY(ds->declare(b, nm, dg, c, base_layer, *kv, tok_embd, "draft", ck_head, dw));

    RadDrafterDecl drf{};
    drf.name           = "dspark";
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

    rad_note(b, "DSpark drafter: %lld layers at absolute %d..%d, %lld heads / %lld kv x %lld, "
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

#endif /* RAD_BLOCK_DSPARK_BF16_H */