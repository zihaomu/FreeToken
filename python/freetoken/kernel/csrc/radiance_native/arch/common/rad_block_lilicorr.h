/* rad_block_lilicorr.h -- LiLiCorr's head: the candidate correlator after a DFlash2 backbone.
 *
 * NVIDIA's drafter for Nemotron-3.5-Bolt (vLLM's model_executor/models/lilicorr.py). The backbone is
 * rad_block_dflash2.h's, with attention sinks (Dflash2::Config::lili). Its final hidden states at
 * the block's draft rows go here instead of to DFlash2's selector:
 *
 *   candidates  each draft position's top-k from the target's own lm_head (log-softmax values,
 *               vocab-parallel: each rank's shard, all-gathered, merged)
 *   lattice     steps x k rows a sequence, row s * k + j:
 *                 token_proj(embed(cand)) + pass_hidden_proj(h_s)
 *                 + feature_mlp(layernorm(five rank features)) + slot[s] + rank[j]
 *   2 layers    rmsnorm, attention over the lattice with a learned slot bias (lili_attn), the
 *               in/out projections; rmsnorm, silu MLP with biases; residuals
 *   factor      x = output_norm(lattice), a = anchor_norm(context_proj(the anchor's context row))
 *               f = silu(W [x | a | x * a] + b), [out | in] = edge heads of f
 *   walk        normalised dot products, start from the anchor's own outgoing head, greedy
 *               (lili_walk)
 *
 * The anchor's context row is hidden_norm(fc(taps)) at the last committed position, which the
 * context pass left in Dflash2's `c` (RadBatch::anchor_rows names the row). The head reads the
 * target's embedding and lm_head, and its own 25 linears are bf16: ModelOpt ships them NVFP4,
 * which no kernel serves, and the recipe decodes them (`cast dtype=bf16`).
 *
 * Every rank computes the whole lattice (vLLM: "candidate IDs and embeddings are replicated across
 * TP ranks"); only the head's logits and the embedding lookup are per rank, with the merge in
 * lili_cand and an exact all-reduce after the lookup.
 */
#ifndef RAD_BLOCK_LILICORR_H
#define RAD_BLOCK_LILICORR_H

#include "rad_arch.h"

namespace rad {
namespace arch {

struct LiliCorr {
    enum { MAX_LAYERS = 4 };

    struct Config {
        int64_t steps   = 0;      /* draft positions a sequence: the block less the anchor */
        int64_t k       = 8;      /* candidates a position (lilicorr_candidate_topk) */
        int64_t hs      = 1024;   /* the lattice's width (lilicorr_hidden_size) */
        int64_t n_head  = 8;
        int64_t mlp     = 2048;   /* hs * lilicorr_mlp_ratio */
        int64_t fdim    = 1024;   /* lilicorr_factor_dim */
        int64_t n_layer = 2;
        double  eps     = 1e-6;   /* rms_norm_eps, the lattice's RMS norms */
        double  vec_eps = 1e-4;   /* lilicorr_vector_eps, F.normalize's floor */
        double  scale   = 8.0;    /* lilicorr_logit_scale */
    };

    /* row_k: the pairs a rank's logprob row carries, k rounded up so 2 + 2 row_k floats are whole
     * 16-byte words (the all-gather's unit): 9 for k = 8. */
    static int64_t row_k(int64_t k) { return (k % 2) ? k : k + 1; }

    struct Wire {
        /* Dflash2's */
        rad_buf sel_h  = 0;   /* [srows, n_embd]  the drafter's final hidden states, draft rows */
        rad_buf c      = 0;   /* [max_tok, n_embd] hidden_norm(fc(taps)), the context pass's rows */
        rad_buf tokens = 0;   /* [max_seqs, steps] i32, the proposal */
        /* the head's own */
        rad_buf logits = 0;   /* [srows, n_vocab] f32, this rank's shard */
        rad_buf stats  = 0;   /* [srows, 2 + 2 row_k] f32 */
        rad_buf sgath  = 0;   /* [srows, world * (2 + 2 row_k)] f32, world > 1 */
        rad_buf pas    = 0;   /* [srows, hs] */
        rad_buf cand   = 0;   /* [srows * k] i32 */
        rad_buf feat   = 0;   /* [lrows, hs] */
        rad_buf hh     = 0;   /* [lrows, hs]  the lattice's residual stream; starts as lili_cand's base */
        rad_buf emb    = 0;   /* [lrows, n_embd] the candidates' embeddings */
        rad_buf y      = 0;   /* [lrows, hs] */
        rad_buf qkv    = 0;   /* [lrows, 3 hs] */
        rad_buf att    = 0;   /* [lrows, hs] */
        rad_buf up     = 0;   /* [lrows, mlp] */
        rad_buf ups    = 0;   /* [lrows, mlp] */
        rad_buf cat    = 0;   /* [lrows, 3 hs] */
        rad_buf fh     = 0;   /* [lrows, hs] */
        rad_buf fhs    = 0;   /* [lrows, hs] */
        rad_buf edges  = 0;   /* [lrows, 2 fdim] */
        rad_buf anc    = 0;   /* [max_seqs, n_embd] */
        rad_buf ast    = 0;   /* [max_seqs, hs] */
        rad_buf an     = 0;   /* [max_seqs, hs] */
        rad_buf aout   = 0;   /* [max_seqs, fdim] */
    };

    Geom    g{};
    Config  cfg{};
    Wire    w{};
    int64_t srows = 0, lrows = 0, rk = 0;

    rad_weight w_lm = 0, w_tok = 0;          /* the target's */
    rad_weight w_tokp = 0, b_tokp = 0, w_pass = 0, b_pass = 0, w_ctx = 0, b_ctx = 0;
    rad_weight w_fdn = 0, b_fdn = 0, ln_w = 0, ln_b = 0, up_w = 0, up_b = 0, w_slot = 0, w_rank = 0;
    rad_weight w_anorm[MAX_LAYERS] = {}, w_in[MAX_LAYERS] = {}, b_in[MAX_LAYERS] = {};
    rad_weight w_out[MAX_LAYERS] = {}, b_out[MAX_LAYERS] = {}, w_mnorm[MAX_LAYERS] = {};
    rad_weight w_up[MAX_LAYERS] = {}, b_up[MAX_LAYERS] = {}, w_dn[MAX_LAYERS] = {};
    rad_weight b_dn[MAX_LAYERS] = {};
    rad_weight w_rel = 0, w_same = 0, w_onorm = 0, w_annorm = 0;
    rad_weight w_fac = 0, b_fac = 0, w_edge = 0, b_edge = 0, w_aout = 0, b_aout = 0;

    rad_op op_logits = 0, op_stats = 0, op_sgath = 0, op_pas = 0, op_cand = 0;
    rad_op op_emb = 0, op_emb_ar = 0, op_tokp = 0, op_fdn = 0;
    rad_op op_anorm[MAX_LAYERS] = {}, op_in[MAX_LAYERS] = {}, op_attn = 0;
    rad_op op_out[MAX_LAYERS] = {}, op_mnorm[MAX_LAYERS] = {}, op_up[MAX_LAYERS] = {};
    rad_op op_dn[MAX_LAYERS] = {}, op_silu_up = 0;
    rad_op op_onorm = 0, op_anc = 0, op_ctx = 0, op_annorm = 0, op_factor = 0, op_fac = 0;
    rad_op op_silu_f = 0, op_edge = 0, op_aout = 0, op_walk = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, rad_weight lm_head,
                 rad_weight tok_embd, const char* ck, const Wire& wire);
    /* After Dflash2::step, on the same query batch: `sel_h` holds the draft rows. */
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int LiliCorr::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c,
                             rad_weight lm_head, rad_weight tok_embd, const char* ck,
                             const Wire& wire) {
    g = geom; cfg = c; w = wire; w_lm = lm_head; w_tok = tok_embd;
    if (!w_lm || !w_tok || !ck || cfg.steps < 1 || cfg.k < 2 || cfg.n_layer < 1 ||
        cfg.n_layer > MAX_LAYERS || cfg.hs % cfg.n_head)
        return RAD_E_INVAL;
    const int64_t K = cfg.k, S = cfg.steps, hs = cfg.hs, fd = cfg.fdim, E = g.n_embd;
    srows = g.max_seqs * S;
    lrows = srows * K;
    rk    = row_k(K);
    const int ACC = RAD_ACCESS_PER_TOKEN;
    const RadWeightGroup grp = grp_model();

    /* The head's weights, all replicated. Linears and tables bf16, biases, norms and the
     * feature layer f32. */
    auto wt = [&](const char* dn, const char* src, uint32_t dt,
                  std::initializer_list<int64_t> shp) -> rad_weight {
        if (map_copy(b, dn, src) < 0) return 0;
        return decl_w(b, dn, dt, shp, ACC, RAD_SHARD_NONE, grp);
    };
    const char* L = nm.ckpt("%s.lilicorr", ck);
    auto src = [&](const char* tail) { return nm.ckpt("%s.%s", L, tail); };
    w_tokp = wt(nm.f("dflash.lili.token_proj.weight"), src("token_proj.weight"), RAD_BF16, { hs, E });
    b_tokp = wt(nm.f("dflash.lili.token_proj.bias"), src("token_proj.bias"), RAD_F32, { hs });
    w_pass = wt(nm.f("dflash.lili.pass_proj.weight"), src("pass_hidden_proj.weight"), RAD_BF16,
                { hs, E });
    b_pass = wt(nm.f("dflash.lili.pass_proj.bias"), src("pass_hidden_proj.bias"), RAD_F32, { hs });
    w_ctx  = wt(nm.f("dflash.lili.context_proj.weight"), src("context_proj.weight"), RAD_BF16,
                { hs, E });
    b_ctx  = wt(nm.f("dflash.lili.context_proj.bias"), src("context_proj.bias"), RAD_F32, { hs });
    ln_w   = wt(nm.f("dflash.lili.feature_norm.weight"), src("feature_norm.weight"), RAD_F32, { 5 });
    ln_b   = wt(nm.f("dflash.lili.feature_norm.bias"), src("feature_norm.bias"), RAD_F32, { 5 });
    up_w   = wt(nm.f("dflash.lili.feature_up.weight"), src("feature_mlp.up_proj.weight"), RAD_F32,
                { hs, 5 });
    up_b   = wt(nm.f("dflash.lili.feature_up.bias"), src("feature_mlp.up_proj.bias"), RAD_F32,
                { hs });
    w_fdn  = wt(nm.f("dflash.lili.feature_down.weight"), src("feature_mlp.down_proj.weight"),
                RAD_BF16, { hs, hs });
    b_fdn  = wt(nm.f("dflash.lili.feature_down.bias"), src("feature_mlp.down_proj.bias"), RAD_F32,
                { hs });
    /* [1, 1, steps, 1, hs] and [1, 1, 1, k, hs], read as tables (a reshape of one whole plane). */
    w_slot = wt(nm.f("dflash.lili.slot_embedding"), src("slot_embedding"), RAD_BF16, { S, hs });
    w_rank = wt(nm.f("dflash.lili.rank_embedding"), src("rank_embedding"), RAD_BF16, { K, hs });
    w_rel  = wt(nm.f("dflash.lili.relative_slot_bias"), src("relative_slot_bias"), RAD_F32,
                { cfg.n_head, 2 * (S + 1) - 1 });
    w_same = wt(nm.f("dflash.lili.same_slot_bias"), src("same_slot_bias"), RAD_F32, { cfg.n_head });
    for (int64_t l = 0; l < cfg.n_layer; ++l) {
        const int i = (int)l;
        auto ls = [&](const char* tail) { return nm.ckpt("%s.layers.%d.%s", L, i, tail); };
        w_anorm[i] = wt(nm.f("dflash.lili.blk.%d.attn_norm.weight", i), ls("attn_norm.weight"),
                        RAD_F32, { hs });
        w_in[i]  = wt(nm.f("dflash.lili.blk.%d.attn_in.weight", i), ls("attn.in_proj_weight"),
                      RAD_BF16, { 3 * hs, hs });
        b_in[i]  = wt(nm.f("dflash.lili.blk.%d.attn_in.bias", i), ls("attn.in_proj_bias"), RAD_F32,
                      { 3 * hs });
        w_out[i] = wt(nm.f("dflash.lili.blk.%d.attn_out.weight", i), ls("attn.out_proj.weight"),
                      RAD_BF16, { hs, hs });
        b_out[i] = wt(nm.f("dflash.lili.blk.%d.attn_out.bias", i), ls("attn.out_proj.bias"),
                      RAD_F32, { hs });
        w_mnorm[i] = wt(nm.f("dflash.lili.blk.%d.mlp_norm.weight", i), ls("mlp_norm.weight"),
                        RAD_F32, { hs });
        w_up[i]  = wt(nm.f("dflash.lili.blk.%d.mlp_up.weight", i), ls("mlp.up_proj.weight"),
                      RAD_BF16, { cfg.mlp, hs });
        b_up[i]  = wt(nm.f("dflash.lili.blk.%d.mlp_up.bias", i), ls("mlp.up_proj.bias"), RAD_F32,
                      { cfg.mlp });
        w_dn[i]  = wt(nm.f("dflash.lili.blk.%d.mlp_down.weight", i), ls("mlp.down_proj.weight"),
                      RAD_BF16, { hs, cfg.mlp });
        b_dn[i]  = wt(nm.f("dflash.lili.blk.%d.mlp_down.bias", i), ls("mlp.down_proj.bias"),
                      RAD_F32, { hs });
        if (!w_anorm[i] || !w_in[i] || !b_in[i] || !w_out[i] || !b_out[i] || !w_mnorm[i] ||
            !w_up[i] || !b_up[i] || !w_dn[i] || !b_dn[i])
            return RAD_E_INVAL;
    }
    w_onorm  = wt(nm.f("dflash.lili.output_norm.weight"), src("output_norm.weight"), RAD_F32, { hs });
    w_annorm = wt(nm.f("dflash.lili.anchor_norm.weight"), src("anchor_norm.weight"), RAD_F32, { hs });
    w_fac    = wt(nm.f("dflash.lili.factor_proj.weight"), src("factor_input_proj.weight"), RAD_BF16,
                  { hs, 3 * hs });
    b_fac    = wt(nm.f("dflash.lili.factor_proj.bias"), src("factor_input_proj.bias"), RAD_F32,
                  { hs });
    /* [out_head | in_head] as one linear, as vLLM fuses them (`_fused_edge_weight`). */
    RAD_ARCH_TRY(map_concat(b, nm.f("dflash.lili.edge.weight"),
                            { src("out_head.weight"), src("in_head.weight") }, 0));
    RAD_ARCH_TRY(map_concat(b, nm.f("dflash.lili.edge.bias"),
                            { src("out_head.bias"), src("in_head.bias") }, 0));
    w_edge = decl_w(b, nm.f("dflash.lili.edge.weight"), RAD_BF16, { 2 * fd, hs }, ACC,
                    RAD_SHARD_NONE, grp);
    b_edge = decl_w(b, nm.f("dflash.lili.edge.bias"), RAD_F32, { 2 * fd }, ACC, RAD_SHARD_NONE,
                    grp);
    w_aout = wt(nm.f("dflash.lili.anchor_out.weight"), src("anchor_out_head.weight"), RAD_BF16,
                { fd, hs });
    b_aout = wt(nm.f("dflash.lili.anchor_out.bias"), src("anchor_out_head.bias"), RAD_F32, { fd });
    if (!w_tokp || !b_tokp || !w_pass || !b_pass || !w_ctx || !b_ctx || !ln_w || !ln_b || !up_w ||
        !up_b || !w_fdn || !b_fdn || !w_slot || !w_rank || !w_rel || !w_same || !w_onorm ||
        !w_annorm || !w_fac || !b_fac || !w_edge || !b_edge || !w_aout || !b_aout)
        return RAD_E_INVAL;

    /* ---------------------------------------------------------------- ops */
    auto gemm_b = [&](int64_t rows, int64_t N, int64_t Kd, rad_weight wgt, rad_weight bias)
        -> rad_op {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", Kd),
                                 RAD_STR("dtype", "bf16"), RAD_STR("act", "none")),
                      RAD_WEIGHTS(wgt, bias));
    };
    auto rms = [&](int64_t rows, int64_t n, rad_weight gain) -> rad_op {
        return RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", cfg.eps),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(gain));
    };
    auto silu = [&](int64_t rows, int64_t n) -> rad_op {
        return RAD_OP(b, "silu",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    };
    const int64_t sw = 2 + 2 * rk;

    /* The candidates: this rank's shard of the target's logits, its logprob row, the exchange. */
    op_logits = rw(b, RAD_OP(b, "logits_gemm",
                             RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n_vocab", g.n_vocab),
                                        RAD_INT("n_embd", E), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w_lm)),
                   { w.sel_h }, { w.logits });
    /* Replicated, `vocab_off` is omitted: an optional parameter present at 0 is a different
     * declaration from an absent one. */
    op_stats = g.world > 1
        ? rw(b, RAD_OP(b, "sample_logprobs",
                       RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n_vocab", g.n_vocab),
                                  RAD_INT("k", rk), RAD_INT("vocab_off", g.vocab_off)),
                       RAD_NOWEIGHTS),
             { w.logits }, { w.stats })
        : rw(b, RAD_OP(b, "sample_logprobs",
                       RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n_vocab", g.n_vocab),
                                  RAD_INT("k", rk)),
                       RAD_NOWEIGHTS),
             { w.logits }, { w.stats });
    if (g.world > 1)
        op_sgath = rw(b, RAD_OP(b, "all_gather",
                                RAD_PARAMS(RAD_INT("world_size", g.world),
                                           RAD_INT("numel", srows * sw), RAD_STR("dtype", "f32"),
                                           RAD_INT("row", sw)),
                                RAD_NOWEIGHTS),
                      { w.stats }, { w.sgath });
    op_pas  = rw(b, gemm_b(srows, hs, E, w_pass, b_pass), { w.sel_h }, { w.pas });
    op_cand = rw(b, RAD_OP(b, "lili_cand",
                           RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("k", K),
                                      RAD_INT("world", g.world), RAD_INT("steps", S),
                                      RAD_INT("n", hs), RAD_STR("dtype", "bf16"),
                                      RAD_F64("eps", 1e-5), RAD_INT("row_k", rk)),
                           RAD_WEIGHTS(ln_w, ln_b, up_w, up_b, w_slot, w_rank)),
                 { g.world > 1 ? w.sgath : w.stats, w.pas }, { w.cand, w.feat, w.hh });
    /* The lattice: candidates' embeddings (the trunk's vocab-sharded table, exact reduce), the
     * token and feature projections added onto lili_cand's base in their GEMMs' residual. */
    op_emb = rw(b, RAD_OP(b, "embed_lookup",
                          RAD_PARAMS(RAD_RANGE("M", 1, lrows), RAD_INT("n_embd", E),
                                     RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
                                     RAD_INT("vocab_offset", g.vocab_off)),
                          RAD_WEIGHTS(w_tok)),
                { w.cand }, { w.emb });
    if (g.world > 1)
        op_emb_ar = rw(b, RAD_OP(b, "all_reduce",
                                 RAD_PARAMS(RAD_INT("world_size", g.world),
                                            RAD_RANGE("numel", E, lrows * E),
                                            RAD_STR("dtype", "bf16"), RAD_INT("exact", 1)),
                                 RAD_NOWEIGHTS),
                       { w.emb }, { w.emb });
    op_tokp = rw(b, gemm_b(lrows, hs, E, w_tokp, b_tokp), { w.emb, w.hh }, { w.hh });
    op_fdn  = rw(b, gemm_b(lrows, hs, hs, w_fdn, b_fdn), { w.feat, w.hh }, { w.hh });
    for (int64_t l = 0; l < cfg.n_layer; ++l) {
        const int i = (int)l;
        op_anorm[i] = rw(b, rms(lrows, hs, w_anorm[i]), { w.hh }, { w.y });
        op_in[i]    = rw(b, gemm_b(lrows, 3 * hs, hs, w_in[i], b_in[i]), { w.y }, { w.qkv });
        op_out[i]   = rw(b, gemm_b(lrows, hs, hs, w_out[i], b_out[i]), { w.att, w.hh }, { w.hh });
        op_mnorm[i] = rw(b, rms(lrows, hs, w_mnorm[i]), { w.hh }, { w.y });
        op_up[i]    = rw(b, gemm_b(lrows, cfg.mlp, hs, w_up[i], b_up[i]), { w.y }, { w.up });
        op_dn[i]    = rw(b, gemm_b(lrows, hs, cfg.mlp, w_dn[i], b_dn[i]), { w.ups, w.hh }, { w.hh });
    }
    op_attn = rw(b, RAD_OP(b, "lili_attn",
                           RAD_PARAMS(RAD_RANGE("M", 1, lrows), RAD_INT("steps", S), RAD_INT("k", K),
                                      RAD_INT("n_head", cfg.n_head),
                                      RAD_INT("head_dim", hs / cfg.n_head),
                                      RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(w_rel, w_same)),
                 { w.qkv }, { w.att });
    op_silu_up = rw(b, silu(lrows, cfg.mlp), { w.up }, { w.ups });
    op_onorm   = rw(b, rms(lrows, hs, w_onorm), { w.hh }, { w.y });
    /* The anchor: the context pass's row at each drafting sequence's last committed position. */
    op_anc = rw(b, RAD_OP(b, "gather_rows",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("n", E),
                                     RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS),
                { w.c }, { w.anc });
    op_ctx    = rw(b, gemm_b(g.max_seqs, hs, E, w_ctx, b_ctx), { w.anc }, { w.ast });
    op_annorm = rw(b, rms(g.max_seqs, hs, w_annorm), { w.ast }, { w.an });
    op_factor = rw(b, RAD_OP(b, "lili_factor",
                             RAD_PARAMS(RAD_RANGE("M", 1, lrows), RAD_INT("L", S * K),
                                        RAD_INT("n", hs), RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { w.y, w.an }, { w.cat });
    op_fac    = rw(b, gemm_b(lrows, hs, 3 * hs, w_fac, b_fac), { w.cat }, { w.fh });
    op_silu_f = rw(b, silu(lrows, hs), { w.fh }, { w.fhs });
    op_edge   = rw(b, gemm_b(lrows, 2 * fd, hs, w_edge, b_edge), { w.fhs }, { w.edges });
    op_aout   = rw(b, gemm_b(g.max_seqs, fd, hs, w_aout, b_aout), { w.an }, { w.aout });
    op_walk   = rw(b, RAD_OP(b, "lili_walk",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", S),
                                        RAD_INT("k", K), RAD_INT("n", fd), RAD_STR("dtype", "bf16"),
                                        RAD_F64("eps", cfg.vec_eps), RAD_F64("scale", cfg.scale)),
                             RAD_NOWEIGHTS),
                   { w.edges, w.aout, w.cand }, { w.tokens });
    if (!op_logits || !op_stats || (g.world > 1 && (!op_sgath || !op_emb_ar)) || !op_pas ||
        !op_cand || !op_emb || !op_tokp || !op_fdn || !op_attn || !op_silu_up || !op_onorm ||
        !op_anc || !op_ctx || !op_annorm || !op_factor || !op_fac || !op_silu_f || !op_edge ||
        !op_aout || !op_walk)
        return RAD_E_INVAL;
    return RAD_OK;
}

inline void LiliCorr::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t n = batch->n_seq, S = n * cfg.steps, Lr = S * cfg.k;
    if (n <= 0) return;
    if (!batch->anchor_rows) {
        rad_step_fail(c, "lilicorr: the query pass carries no anchor rows (RadBatch::anchor_rows): "
                         "this engine does not position a block drafter's anchor");
        return;
    }
    const int64_t E = g.n_embd, hs = cfg.hs, fd = cfg.fdim;
    const int64_t sw = 2 + 2 * rk;

    RAD_ISSUE_N(c, op_logits, S, brows(w.sel_h, S), RAD_W(w_lm), brows(w.logits, S));
    RAD_ISSUE_N(c, op_stats, S, brows(w.logits, S), RAD_NONE, brows(w.stats, S));
    if (op_sgath) RAD_ISSUE(c, op_sgath, brows(w.stats, S), brows(w.sgath, S));
    RAD_ISSUE_N(c, op_pas, S, brows(w.sel_h, S), RAD_W(w_pass), RAD_W(b_pass), RAD_NONE,
                brows(w.pas, S));
    RAD_ISSUE_N(c, op_cand, S, op_sgath ? brows(w.sgath, S) : brows(w.stats, S), brows(w.pas, S),
                RAD_W(ln_w), RAD_W(ln_b), RAD_W(up_w), RAD_W(up_b), RAD_W(w_slot), RAD_W(w_rank),
                brow_slice(w.cand, 0, Lr, 1), brows(w.feat, Lr), brows(w.hh, Lr));
    (void)sw;

    RAD_ISSUE_N(c, op_emb, Lr, brow_slice(w.cand, 0, Lr, 1), RAD_W(w_tok), brows(w.emb, Lr));
    if (op_emb_ar) RAD_ISSUE_N(c, op_emb_ar, Lr * E, brows(w.emb, Lr), RAD_NONE);
    RAD_ISSUE_N(c, op_tokp, Lr, brows(w.emb, Lr), RAD_W(w_tokp), RAD_W(b_tokp), brows(w.hh, Lr),
                brows(w.hh, Lr));
    RAD_ISSUE_N(c, op_fdn, Lr, brows(w.feat, Lr), RAD_W(w_fdn), RAD_W(b_fdn), brows(w.hh, Lr),
                brows(w.hh, Lr));
    for (int64_t l = 0; l < cfg.n_layer; ++l) {
        const int i = (int)l;
        RAD_ISSUE_N(c, op_anorm[i], Lr, brows(w.hh, Lr), RAD_W(w_anorm[i]), brows(w.y, Lr));
        RAD_ISSUE_N(c, op_in[i], Lr, brows(w.y, Lr), RAD_W(w_in[i]), RAD_W(b_in[i]), RAD_NONE,
                    brows(w.qkv, Lr));
        RAD_ISSUE_N(c, op_attn, Lr, brows(w.qkv, Lr), RAD_W(w_rel), RAD_W(w_same), brows(w.att, Lr));
        RAD_ISSUE_N(c, op_out[i], Lr, brows(w.att, Lr), RAD_W(w_out[i]), RAD_W(b_out[i]),
                    brows(w.hh, Lr), brows(w.hh, Lr));
        RAD_ISSUE_N(c, op_mnorm[i], Lr, brows(w.hh, Lr), RAD_W(w_mnorm[i]), brows(w.y, Lr));
        RAD_ISSUE_N(c, op_up[i], Lr, brows(w.y, Lr), RAD_W(w_up[i]), RAD_W(b_up[i]), RAD_NONE,
                    brows(w.up, Lr));
        RAD_ISSUE_N(c, op_silu_up, Lr, brows(w.up, Lr), brows(w.ups, Lr));
        RAD_ISSUE_N(c, op_dn[i], Lr, brows(w.ups, Lr), RAD_W(w_dn[i]), RAD_W(b_dn[i]),
                    brows(w.hh, Lr), brows(w.hh, Lr));
    }
    RAD_ISSUE_N(c, op_onorm, Lr, brows(w.hh, Lr), RAD_W(w_onorm), brows(w.y, Lr));

    RAD_ISSUE_N(c, op_anc, n, RAD_B(w.c), praw(batch->anchor_rows, RAD_I32, n), brows(w.anc, n));
    RAD_ISSUE_N(c, op_ctx, n, brows(w.anc, n), RAD_W(w_ctx), RAD_W(b_ctx), RAD_NONE, brows(w.ast, n));
    RAD_ISSUE_N(c, op_annorm, n, brows(w.ast, n), RAD_W(w_annorm), brows(w.an, n));
    RAD_ISSUE_N(c, op_factor, Lr, brows(w.y, Lr), brows(w.an, n), brows(w.cat, Lr));
    RAD_ISSUE_N(c, op_fac, Lr, brows(w.cat, Lr), RAD_W(w_fac), RAD_W(b_fac), RAD_NONE,
                brows(w.fh, Lr));
    RAD_ISSUE_N(c, op_silu_f, Lr, brows(w.fh, Lr), brows(w.fhs, Lr));
    RAD_ISSUE_N(c, op_edge, Lr, brows(w.fhs, Lr), RAD_W(w_edge), RAD_W(b_edge), RAD_NONE,
                brows(w.edges, Lr));
    RAD_ISSUE_N(c, op_aout, n, brows(w.an, n), RAD_W(w_aout), RAD_W(b_aout), RAD_NONE,
                brows(w.aout, n));
    RAD_ISSUE_N(c, op_walk, n, brows(w.edges, Lr), brows(w.aout, n),
                brow_slice(w.cand, 0, Lr, 1), brows(w.tokens, n), RAD_NONE);
    (void)E; (void)hs; (void)fd;
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_LILICORR_H */
