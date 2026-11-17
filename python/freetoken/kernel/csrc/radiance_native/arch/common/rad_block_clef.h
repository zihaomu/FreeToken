/* rad_block_clef.h -- clef's joint schema head, inside the backbone's plugin.
 *
 * Cloudflare's clef and clef-flash decide every question of a request jointly from the backbone's
 * final hidden state at EVERY prompt position (joint_schema_model.py, JointSchemaHead). This block
 * runs that head on the card, after the trunk, so /v1/systemone needs no second process:
 *
 *   per token, every step      hn = layernorm(h); hn is stored in a paged KV group of its own,
 *                              so a prefix hit keeps the rows the trunk did not recompute
 *   per request, once          every row of hn is gathered back out of the group's blocks, the
 *                              memory projection and the six cross-attention layers run over all
 *                              of them, the question and option spans are averaged, and the
 *                              routing, decoder layers and scorer give three numbers an option
 *                              (its lexical prior, its cosine and its residual) into the request's
 *                              logits row, where the server reads them (core/server/clef.cpp)
 *
 * The spans, the option tokens and the question types come as the request's words
 * (abi/rad_clef.h, RadBatch::plugin_words), laid out as the kernels read them. The head's own
 * tensors are the release's joint_head.safetensors, added to the container at convert, under their
 * own names. Every op here has a device kernel and a host reference, so the same block serves
 * the tiny CPU test and the card.
 *
 * Differences from the reference, all within bf16: the cosine and the prior are means of
 * products of RMS-normalised rows (an identity, with eps 1e-12 against torch's max(norm, 1e-12));
 * the routing softmax is formed in f32 (seg_softmax_pool). The learned scalars (prior_logit_scale,
 * joint_logit_scale, residual_gate) are applied by the server from the container header.
 */
#pragma once

#include <arch/rad_arch.h>
#include <rad_clef.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace rad {
namespace arch {

struct ClefHead {
    bool on = false;

    int64_t E = 0;          /* the backbone's hidden size: the head's input width */
    int64_t W = 0;          /* the head's width */
    int64_t H = 0;          /* heads */
    int64_t HD = 0;         /* W / H */
    int64_t F = 0;          /* feedforward */
    int64_t bs = 0;         /* the cache's block size */
    int64_t nblk = 0;       /* blocks a context holds at most */
    int64_t NM = 0;         /* nblk * bs: rows a request holds at most */
    int64_t n_vocab = 0;    /* this rank's logits width */
    int     layer = 0;      /* the group's bound layer index */
    double  eps = 1e-5;     /* torch.nn.LayerNorm's default */

    rad_kvgroup kv = 0;

    /* ---- weights */
    rad_weight hn_w = 0, hn_b = 0, w_mem = 0, w_q = 0, w_oq = 0, w_glob = 0, w_octx = 0,
               w_olex = 0, w_type = 0, osn_w = 0, osn_b = 0, fn_w = 0, fn_b = 0, on_w = 0,
               on_b = 0, rs1_w = 0, rs1_b = 0, rs2_w = 0, rs2_b = 0;
    struct Ev {
        rad_weight qn_w = 0, qn_b = 0, mn_w = 0, mn_b = 0, wq = 0, bq = 0, wk = 0, bk = 0, wv = 0,
                   bv = 0, wo = 0, bo = 0, fn_w = 0, fn_b = 0, w1 = 0, b1 = 0, w2 = 0, b2 = 0;
        rad_op ln_q = 0, ln_m = 0, pq = 0, pk = 0, pv = 0, attn = 0, po = 0, ln_f = 0, f1 = 0,
               f2 = 0;
    };
    struct Dec {
        rad_weight n1w = 0, n1b = 0, n2w = 0, n2b = 0, n3w = 0, n3b = 0, sa_w = 0, sa_b = 0,
                   so_w = 0, so_b = 0, cq_w = 0, cq_b = 0, ck_w = 0, ck_b = 0, cv_w = 0, cv_b = 0,
                   co_w = 0, co_b = 0, w1 = 0, b1 = 0, w2 = 0, b2 = 0;
        rad_op ln1 = 0, qkv = 0, sattn = 0, so = 0, ln2 = 0, pq = 0, pk = 0, pv = 0, cattn = 0,
               co = 0, ln3 = 0, f1 = 0, f2 = 0;
    };
    std::vector<Ev>  ev;
    std::vector<Dec> dec;

    /* ---- buffers */
    rad_buf b_hn = 0, b_all = 0, b_mem = 0, b_mn = 0, b_k = 0, b_v = 0, b_seg = 0, b_qv = 0,
            b_oc = 0, b_g = 0, b_lx = 0, b_lex = 0, b_x = 0, b_t1 = 0, b_t2 = 0, b_ff = 0,
            b_qq = 0, b_bf = 0, b_fr = 0, b_sc = 0, b_sum = 0, b_f = 0, b_fq = 0, b_f1 = 0,
            b_f2 = 0, b_gw = 0, b_fn = 0, b_on = 0, b_na = 0, b_nb = 0, b_an = 0, b_ln = 0,
            b_p = 0, b_neg1 = 0, b_d = 0, b_nd = 0, b_abs = 0, b_prod = 0, b_feat = 0, b_r = 0,
            b_out3 = 0;

    /* ---- ops, one a use site: the arena plans from each op's read and write sets */
    rad_op op_hn = 0, op_store = 0, op_gather = 0, op_mem = 0, op_seg = 0, op_g_q = 0, op_g_o = 0,
           op_g_last = 0, op_lex = 0, op_lexmean = 0, op_octx = 0, op_olex = 0, op_add_x1 = 0,
           op_oq = 0, op_g_oq = 0, op_add_x2 = 0, op_qproj = 0, op_g_fr = 0, op_mul_sc = 0,
           op_score = 0, op_route = 0, op_ln_osn = 0, op_add_f1 = 0, op_glob = 0, op_add_fg = 0,
           op_type = 0, op_add_f2 = 0, op_ln_fn = 0, op_g_fr2 = 0, op_ln_on = 0, op_rms_a = 0,
           op_rms_b = 0, op_mul_cos = 0, op_mean_cos = 0, op_g_an = 0, op_add_an = 0,
           op_rms_an = 0, op_rms_lex = 0, op_mul_pri = 0, op_mean_pri = 0, op_neg1 = 0,
           op_nd = 0, op_d = 0, op_relu_d = 0, op_nd2 = 0, op_relu_nd = 0, op_abs = 0,
           op_prod = 0, op_cp_fr = 0, op_cp_on = 0, op_cp_prod = 0, op_cp_abs = 0, op_rs1 = 0,
           op_rs2 = 0, op_cast = 0;

    static bool present(const RadModelMeta* meta) {
        const char* t = rad_meta_gets(meta, "decision.type", "");
        return t && !std::strcmp(t, "clef");
    }

    /* Declared after the trunk's layers and lm_head: the trunk's cache block size is known then,
     * and the head's rows read the final norm's output `b_h` and write `b_logits`. */
    int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx, Names& nm,
                const Geom& g, rad_buf b_h, rad_buf b_logits, rad_weight w_lm_head,
                rad_kvgroup kv_trunk, int bind_layer);

    /* Every trunk step: the final norm's rows into the head's cache. */
    void step_tokens(RadCtx* c, const RadBatch* batch, rad_buf b_h);
    /* After the lm_head: the head over every sequence that carries words. */
    void step_heads(RadCtx* c, const RadBatch* batch, rad_buf b_logits, rad_weight w_lm_head);

private:
    const RadKVGroupBatch* group_of(const RadBatch* batch) const {
        for (int i = 0; i < batch->n_kv_groups; ++i)
            if (batch->kv[i].group == kv) return &batch->kv[i];
        return nullptr;
    }
};

inline int ClefHead::declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                             Names& nm, const Geom& g, rad_buf b_h, rad_buf b_logits,
                             rad_weight w_lm_head, rad_kvgroup kv_trunk, int bind_layer) {
    (void)ctx;
    E  = g.n_embd;
    W  = rad_meta_geti(meta, "decision.clef.width", 0);
    H  = rad_meta_geti(meta, "decision.clef.heads", 0);
    F  = rad_meta_geti(meta, "decision.clef.feedforward", 0);
    const long long R = rad_meta_geti(meta, "decision.clef.routing_layers", 0);
    const long long D = rad_meta_geti(meta, "decision.clef.layers", 0);
    const long long hidden = rad_meta_geti(meta, "decision.clef.hidden_size", 0);
    if (W <= 0 || H <= 0 || F <= 0 || R <= 0 || D <= 0 || W % H || hidden != E) {
        std::fprintf(stderr, "radiance: clef: the container's decision.clef.* keys do not describe "
                             "the head (hidden_size %lld against the backbone's %lld, width %lld, "
                             "heads %lld, feedforward %lld, routing_layers %lld, layers %lld); "
                             "rad-convert sets them from joint_head_config.json\n",
                     hidden, (long long)E, (long long)W, (long long)H, (long long)F, R, D);
        return RAD_E_INVAL;
    }
    HD = W / H;
    if (g.world != 1) {
        std::fprintf(stderr, "radiance: clef's head runs on one rank: its rows are replicated and "
                             "the option tokens read the whole lm_head\n");
        return RAD_E_UNSUPPORTED;
    }
    if (weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16) != RAD_BF16) {
        std::fprintf(stderr, "radiance: clef's head averages lm_head rows as bf16; this container's "
                             "lm_head is quantised. Keep output.weight bf16 in the recipe\n");
        return RAD_E_UNSUPPORTED;
    }
    if (E % 2) return RAD_E_INVAL;
    bs = rad_kv_block_size(b, kv_trunk);
    if (bs <= 0) return RAD_E_STATE;
    /* Rows a request may hold: the context, or the head's own bound where that is less (the
     * server refuses a longer prompt before it is admitted). */
    const int64_t rows = g.max_ctx < RAD_CLEF_MAX_PROMPT ? g.max_ctx : RAD_CLEF_MAX_PROMPT;
    nblk = (rows + bs - 1) / bs;
    NM   = nblk * bs;
    n_vocab = g.n_vocab;
    layer = bind_layer;
    const int64_t QM = RAD_CLEF_MAX_QUESTIONS, OM = RAD_CLEF_MAX_OPTIONS, LM = RAD_CLEF_MAX_LEX;
    const int64_t SM = 2 * (QM + OM) + 1;
    if (n_vocab < 3 * OM) {
        std::fprintf(stderr, "radiance: clef's head writes %lld values into a logits row of %lld\n",
                     (long long)(3 * OM), (long long)n_vocab);
        return RAD_E_UNSUPPORTED;
    }

    /* ---- the cache: one FULL group, hn's two halves as a slot's K and V, so a block holds
     * bs rows of E in position order and a gather of a sequence's blocks is its rows. Bound past
     * the trunk, so the scheduler counts it as the head's; cached with the trunk's groups because
     * its block size is theirs (core/mem/prefix.cpp). */
    RadKVGroupDecl kd{};
    kd.kind      = RAD_KV_FULL;
    kd.dtype     = RAD_BF16;
    kd.n_head_kv = 1;
    kd.head_dim  = E / 2;
    kv = rad_decl_kv_group(b, nm.f("clef.kv"), &kd);
    if (!kv) return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_bind_layer_kv(b, layer, kv));
    RAD_ARCH_TRY(rad_declare_plugin_words(b, rad_clef_max_words()));

    /* ---- weights: the release's names, bf16, read once a request. */
    const int ACC = RAD_ACCESS_PER_REQUEST;
    auto bw = [&](const char* dn, const char* src, std::initializer_list<int64_t> shp) -> rad_weight {
        if (map_copy(b, dn, src) < 0) return 0;
        return decl_w(b, dn, RAD_BF16, shp, ACC, RAD_SHARD_NONE, grp_model());
    };
    /* A packed in_proj's third: rows [k W, (k+1) W) of the weight, and of the bias. */
    auto bw_rows = [&](const char* dn, const char* src, int k, int64_t cols) -> rad_weight {
        if (map_rows(b, dn, { { src, (int)(k * W), W } }) < 0) return 0;
        return cols ? decl_w(b, dn, RAD_BF16, { W, cols }, ACC, RAD_SHARD_NONE, grp_model())
                    : decl_w(b, dn, RAD_BF16, { W }, ACC, RAD_SHARD_NONE, grp_model());
    };
    hn_w   = bw(nm.f("clef.hidden_norm.weight"), nm.ckpt("hidden_norm.weight"), { E });
    hn_b   = bw(nm.f("clef.hidden_norm.bias"),   nm.ckpt("hidden_norm.bias"),   { E });
    w_mem  = bw(nm.f("clef.memory_projection.weight"), nm.ckpt("memory_projection.weight"), { W, E });
    w_q    = bw(nm.f("clef.question_projection.weight"), nm.ckpt("question_projection.weight"), { W, E });
    w_oq   = bw(nm.f("clef.option_question_projection.weight"), nm.ckpt("option_question_projection.weight"), { W, E });
    w_glob = bw(nm.f("clef.global_projection.weight"), nm.ckpt("global_projection.weight"), { W, E });
    w_octx = bw(nm.f("clef.option_context_projection.weight"), nm.ckpt("option_context_projection.weight"), { W, E });
    w_olex = bw(nm.f("clef.option_lexical_projection.weight"), nm.ckpt("option_lexical_projection.weight"), { W, E });
    w_type = bw(nm.f("clef.type_embedding.weight"), nm.ckpt("type_embedding.weight"), { 3, W });
    osn_w  = bw(nm.f("clef.option_summary_norm.weight"), nm.ckpt("option_summary_norm.weight"), { W });
    osn_b  = bw(nm.f("clef.option_summary_norm.bias"),   nm.ckpt("option_summary_norm.bias"),   { W });
    fn_w   = bw(nm.f("clef.field_norm.weight"), nm.ckpt("field_norm.weight"), { W });
    fn_b   = bw(nm.f("clef.field_norm.bias"),   nm.ckpt("field_norm.bias"),   { W });
    on_w   = bw(nm.f("clef.option_norm.weight"), nm.ckpt("option_norm.weight"), { W });
    on_b   = bw(nm.f("clef.option_norm.bias"),   nm.ckpt("option_norm.bias"),   { W });
    /* The scorer's first linear reads [field | option | field*option | |field-option|], one row
     * of 4 W an option, which the blocks below assemble in `feat`. */
    rs1_w = bw(nm.f("clef.residual_scorer.0.weight"), nm.ckpt("residual_scorer.0.weight"), { W, 4 * W });
    rs1_b = bw(nm.f("clef.residual_scorer.0.bias"),   nm.ckpt("residual_scorer.0.bias"),   { W });
    rs2_w = bw(nm.f("clef.residual_scorer.3.weight"), nm.ckpt("residual_scorer.3.weight"), { 1, W });
    rs2_b = bw(nm.f("clef.residual_scorer.3.bias"),   nm.ckpt("residual_scorer.3.bias"),   { 1 });
    if (!hn_w || !hn_b || !w_mem || !w_q || !w_oq || !w_glob || !w_octx || !w_olex || !w_type ||
        !osn_w || !osn_b || !fn_w || !fn_b || !on_w || !on_b || !rs1_w || !rs1_b || !rs2_w || !rs2_b)
        return RAD_E_INVAL;

    ev.assign((size_t)R, Ev{});
    for (int i = 0; i < (int)R; ++i) {
        Ev& e = ev[(size_t)i];
        const char* p = "evidence_layers";
        e.qn_w = bw(nm.f("clef.ev.%d.query_norm.weight", i), nm.ckpt("%s.%d.query_norm.weight", p, i), { W });
        e.qn_b = bw(nm.f("clef.ev.%d.query_norm.bias", i),   nm.ckpt("%s.%d.query_norm.bias", p, i),   { W });
        e.mn_w = bw(nm.f("clef.ev.%d.memory_norm.weight", i), nm.ckpt("%s.%d.memory_norm.weight", p, i), { W });
        e.mn_b = bw(nm.f("clef.ev.%d.memory_norm.bias", i),   nm.ckpt("%s.%d.memory_norm.bias", p, i),   { W });
        e.wq = bw_rows(nm.f("clef.ev.%d.q.weight", i), nm.ckpt("%s.%d.attention.in_proj_weight", p, i), 0, W);
        e.wk = bw_rows(nm.f("clef.ev.%d.k.weight", i), nm.ckpt("%s.%d.attention.in_proj_weight", p, i), 1, W);
        e.wv = bw_rows(nm.f("clef.ev.%d.v.weight", i), nm.ckpt("%s.%d.attention.in_proj_weight", p, i), 2, W);
        e.bq = bw_rows(nm.f("clef.ev.%d.q.bias", i), nm.ckpt("%s.%d.attention.in_proj_bias", p, i), 0, 0);
        e.bk = bw_rows(nm.f("clef.ev.%d.k.bias", i), nm.ckpt("%s.%d.attention.in_proj_bias", p, i), 1, 0);
        e.bv = bw_rows(nm.f("clef.ev.%d.v.bias", i), nm.ckpt("%s.%d.attention.in_proj_bias", p, i), 2, 0);
        e.wo = bw(nm.f("clef.ev.%d.o.weight", i), nm.ckpt("%s.%d.attention.out_proj.weight", p, i), { W, W });
        e.bo = bw(nm.f("clef.ev.%d.o.bias", i),   nm.ckpt("%s.%d.attention.out_proj.bias", p, i),   { W });
        e.fn_w = bw(nm.f("clef.ev.%d.ff_norm.weight", i), nm.ckpt("%s.%d.feedforward_norm.weight", p, i), { W });
        e.fn_b = bw(nm.f("clef.ev.%d.ff_norm.bias", i),   nm.ckpt("%s.%d.feedforward_norm.bias", p, i),   { W });
        e.w1 = bw(nm.f("clef.ev.%d.ff1.weight", i), nm.ckpt("%s.%d.feedforward.0.weight", p, i), { F, W });
        e.b1 = bw(nm.f("clef.ev.%d.ff1.bias", i),   nm.ckpt("%s.%d.feedforward.0.bias", p, i),   { F });
        e.w2 = bw(nm.f("clef.ev.%d.ff2.weight", i), nm.ckpt("%s.%d.feedforward.3.weight", p, i), { W, F });
        e.b2 = bw(nm.f("clef.ev.%d.ff2.bias", i),   nm.ckpt("%s.%d.feedforward.3.bias", p, i),   { W });
        if (!e.qn_w || !e.qn_b || !e.mn_w || !e.mn_b || !e.wq || !e.wk || !e.wv || !e.bq || !e.bk ||
            !e.bv || !e.wo || !e.bo || !e.fn_w || !e.fn_b || !e.w1 || !e.b1 || !e.w2 || !e.b2)
            return RAD_E_INVAL;
    }
    dec.assign((size_t)D, Dec{});
    for (int i = 0; i < (int)D; ++i) {
        Dec& d = dec[(size_t)i];
        const char* p = "layers";
        d.n1w = bw(nm.f("clef.dec.%d.norm1.weight", i), nm.ckpt("%s.%d.norm1.weight", p, i), { W });
        d.n1b = bw(nm.f("clef.dec.%d.norm1.bias", i),   nm.ckpt("%s.%d.norm1.bias", p, i),   { W });
        d.n2w = bw(nm.f("clef.dec.%d.norm2.weight", i), nm.ckpt("%s.%d.norm2.weight", p, i), { W });
        d.n2b = bw(nm.f("clef.dec.%d.norm2.bias", i),   nm.ckpt("%s.%d.norm2.bias", p, i),   { W });
        d.n3w = bw(nm.f("clef.dec.%d.norm3.weight", i), nm.ckpt("%s.%d.norm3.weight", p, i), { W });
        d.n3b = bw(nm.f("clef.dec.%d.norm3.bias", i),   nm.ckpt("%s.%d.norm3.bias", p, i),   { W });
        /* Self-attention keeps its packed in_proj: one GEMM, then column slices. */
        d.sa_w = bw(nm.f("clef.dec.%d.sa.in.weight", i), nm.ckpt("%s.%d.self_attn.in_proj_weight", p, i), { 3 * W, W });
        d.sa_b = bw(nm.f("clef.dec.%d.sa.in.bias", i),   nm.ckpt("%s.%d.self_attn.in_proj_bias", p, i),   { 3 * W });
        d.so_w = bw(nm.f("clef.dec.%d.sa.o.weight", i), nm.ckpt("%s.%d.self_attn.out_proj.weight", p, i), { W, W });
        d.so_b = bw(nm.f("clef.dec.%d.sa.o.bias", i),   nm.ckpt("%s.%d.self_attn.out_proj.bias", p, i),   { W });
        d.cq_w = bw_rows(nm.f("clef.dec.%d.ca.q.weight", i), nm.ckpt("%s.%d.multihead_attn.in_proj_weight", p, i), 0, W);
        d.ck_w = bw_rows(nm.f("clef.dec.%d.ca.k.weight", i), nm.ckpt("%s.%d.multihead_attn.in_proj_weight", p, i), 1, W);
        d.cv_w = bw_rows(nm.f("clef.dec.%d.ca.v.weight", i), nm.ckpt("%s.%d.multihead_attn.in_proj_weight", p, i), 2, W);
        d.cq_b = bw_rows(nm.f("clef.dec.%d.ca.q.bias", i), nm.ckpt("%s.%d.multihead_attn.in_proj_bias", p, i), 0, 0);
        d.ck_b = bw_rows(nm.f("clef.dec.%d.ca.k.bias", i), nm.ckpt("%s.%d.multihead_attn.in_proj_bias", p, i), 1, 0);
        d.cv_b = bw_rows(nm.f("clef.dec.%d.ca.v.bias", i), nm.ckpt("%s.%d.multihead_attn.in_proj_bias", p, i), 2, 0);
        d.co_w = bw(nm.f("clef.dec.%d.ca.o.weight", i), nm.ckpt("%s.%d.multihead_attn.out_proj.weight", p, i), { W, W });
        d.co_b = bw(nm.f("clef.dec.%d.ca.o.bias", i),   nm.ckpt("%s.%d.multihead_attn.out_proj.bias", p, i),   { W });
        d.w1 = bw(nm.f("clef.dec.%d.ff1.weight", i), nm.ckpt("%s.%d.linear1.weight", p, i), { F, W });
        d.b1 = bw(nm.f("clef.dec.%d.ff1.bias", i),   nm.ckpt("%s.%d.linear1.bias", p, i),   { F });
        d.w2 = bw(nm.f("clef.dec.%d.ff2.weight", i), nm.ckpt("%s.%d.linear2.weight", p, i), { W, F });
        d.b2 = bw(nm.f("clef.dec.%d.ff2.bias", i),   nm.ckpt("%s.%d.linear2.bias", p, i),   { W });
        if (!d.n1w || !d.n1b || !d.n2w || !d.n2b || !d.n3w || !d.n3b || !d.sa_w || !d.sa_b ||
            !d.so_w || !d.so_b || !d.cq_w || !d.ck_w || !d.cv_w || !d.cq_b || !d.ck_b || !d.cv_b ||
            !d.co_w || !d.co_b || !d.w1 || !d.b1 || !d.w2 || !d.b2)
            return RAD_E_INVAL;
    }

    /* ---- buffers, bf16, at the bounds the words are refused past. */
    b_hn   = decl_b(b, nm.f("clef.hn"),   RAD_BF16, { g.max_tok, E });
    b_all  = decl_b(b, nm.f("clef.all"),  RAD_BF16, { NM, E });
    b_mem  = decl_b(b, nm.f("clef.mem"),  RAD_BF16, { NM, W });
    b_mn   = decl_b(b, nm.f("clef.mn"),   RAD_BF16, { NM, W });
    b_k    = decl_b(b, nm.f("clef.k"),    RAD_BF16, { NM, W });
    b_v    = decl_b(b, nm.f("clef.v"),    RAD_BF16, { NM, W });
    b_seg  = decl_b(b, nm.f("clef.seg"),  RAD_BF16, { SM, E });
    b_qv   = decl_b(b, nm.f("clef.qv"),   RAD_BF16, { QM, E });
    b_oc   = decl_b(b, nm.f("clef.oc"),   RAD_BF16, { OM, E });
    b_g    = decl_b(b, nm.f("clef.g"),    RAD_BF16, { 1, E });
    b_lx   = decl_b(b, nm.f("clef.lx"),   RAD_BF16, { LM, E });
    b_lex  = decl_b(b, nm.f("clef.lex"),  RAD_BF16, { OM, E });
    b_x    = decl_b(b, nm.f("clef.x"),    RAD_BF16, { OM, W });
    b_t1   = decl_b(b, nm.f("clef.t1"),   RAD_BF16, { OM, W });
    b_t2   = decl_b(b, nm.f("clef.t2"),   RAD_BF16, { OM, W });
    b_ff   = decl_b(b, nm.f("clef.ff"),   RAD_BF16, { OM, F });
    b_qq   = decl_b(b, nm.f("clef.qq"),   RAD_BF16, { QM, W });
    b_bf   = decl_b(b, nm.f("clef.bf"),   RAD_BF16, { QM, W });
    b_fr   = decl_b(b, nm.f("clef.fr"),   RAD_BF16, { OM, W });
    b_sc   = decl_b(b, nm.f("clef.sc"),   RAD_BF16, { OM, 1 });
    b_sum  = decl_b(b, nm.f("clef.sum"),  RAD_BF16, { QM, W });
    b_f    = decl_b(b, nm.f("clef.f"),    RAD_BF16, { QM, W });
    b_fq   = decl_b(b, nm.f("clef.fq"),   RAD_BF16, { QM, 3 * W });
    b_f1   = decl_b(b, nm.f("clef.f1"),   RAD_BF16, { QM, W });
    b_f2   = decl_b(b, nm.f("clef.f2"),   RAD_BF16, { QM, W });
    b_gw   = decl_b(b, nm.f("clef.gw"),   RAD_BF16, { 1, W });
    b_fn   = decl_b(b, nm.f("clef.fn"),   RAD_BF16, { QM, W });
    b_on   = decl_b(b, nm.f("clef.on"),   RAD_BF16, { OM, W });
    b_na   = decl_b(b, nm.f("clef.na"),   RAD_BF16, { OM, W });
    b_nb   = decl_b(b, nm.f("clef.nb"),   RAD_BF16, { OM, W });
    b_an   = decl_b(b, nm.f("clef.an"),   RAD_BF16, { OM, E });
    b_ln   = decl_b(b, nm.f("clef.ln"),   RAD_BF16, { OM, E });
    b_p    = decl_b(b, nm.f("clef.p"),    RAD_BF16, { OM, E });
    b_neg1 = decl_b(b, nm.f("clef.neg1"), RAD_BF16, { 1, W });
    b_d    = decl_b(b, nm.f("clef.d"),    RAD_BF16, { OM, W });
    b_nd   = decl_b(b, nm.f("clef.nd"),   RAD_BF16, { OM, W });
    b_abs  = decl_b(b, nm.f("clef.abs"),  RAD_BF16, { OM, W });
    b_prod = decl_b(b, nm.f("clef.prod"), RAD_BF16, { OM, W });
    b_feat = decl_b(b, nm.f("clef.feat"), RAD_BF16, { OM, 4 * W });
    b_r    = decl_b(b, nm.f("clef.r"),    RAD_BF16, { OM, W });
    b_out3 = decl_b(b, nm.f("clef.out3"), RAD_BF16, { 3 * OM, 1 });
    if (!b_hn || !b_all || !b_mem || !b_mn || !b_k || !b_v || !b_seg || !b_qv || !b_oc || !b_g ||
        !b_lx || !b_lex || !b_x || !b_t1 || !b_t2 || !b_ff || !b_qq || !b_bf || !b_fr || !b_sc ||
        !b_sum || !b_f || !b_fq || !b_f1 || !b_f2 || !b_gw || !b_fn || !b_on || !b_na || !b_nb ||
        !b_an || !b_ln || !b_p || !b_neg1 || !b_d || !b_nd || !b_abs || !b_prod || !b_feat || !b_r ||
        !b_out3)
        return RAD_E_INVAL;

    /* ---- ops. The lambdas state each kind once; the row bound is the op's band. */
    auto lnorm = [&](int64_t rows, int64_t n, rad_weight w, rad_weight bias) {
        return RAD_OP(b, "layernorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w, bias));
    };
    auto gemm = [&](int64_t rows, int64_t N, int64_t K, rad_weight w) {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto gemmb = [&](int64_t rows, int64_t N, int64_t K, const char* act, rad_weight w,
                     rad_weight bias) {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16"), RAD_STR("act", act)),
                      RAD_WEIGHTS(w, bias));
    };
    auto unary = [&](const char* op, int64_t rows, int64_t n) {
        return RAD_OP(b, op,
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    };
    auto gather = [&](int64_t rows, int64_t n) { return unary("gather_rows", rows, n); };
    auto cross = [&](int64_t rows) {
        return RAD_OP(b, "attn_cross",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", HD), RAD_INT("gqa", 1),
                                 RAD_INT("causal", 0), RAD_STR("dtype", "bf16"),
                                 RAD_F64("scale", 1.0 / std::sqrt((double)HD))),
                      RAD_NOWEIGHTS);
    };
    auto rmsp = [&](int64_t rows, int64_t n) {
        return RAD_OP(b, "rmsnorm_p",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", 1e-12),
                                 RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    };
    /* The mean of a row: a cosine from two RMS-normed rows, or with `mult` the dot product
     * scaled (mult = n / sqrt(W) gives options . field / sqrt(W)). */
    auto rowmean = [&](int64_t rows, int64_t n, double mult) {
        return RAD_OP(b, "pool_cols",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_INT("k", n),
                                 RAD_F64("mult", mult), RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    };

    /* Every step: the final norm's rows into the cache. kv_store carries the block size and so is
     * the group's consumer (core/build/rad_builder.cpp, find_kv_consumer); its write is the pool. */
    op_hn = rw(b, lnorm(g.max_tok, E, hn_w, hn_b), { b_h }, { b_hn });
    op_store = rw(b, RAD_OP(b, "kv_store",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", E / 2),
                                       RAD_INT("n_head_kv", 1), RAD_INT("block_size", bs),
                                       RAD_STR("kv_dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { b_hn }, {});
    if (rad_kv_block_size(b, kv) != bs) {
        std::fprintf(stderr, "radiance: clef's cache took block size %lld, the trunk's is %lld\n",
                     (long long)rad_kv_block_size(b, kv), (long long)bs);
        return RAD_E_STATE;
    }

    /* Per request. A sequence's blocks are rows of bs * E: the gather reads the pool as
     * [n_blocks, bs * E] and writes `all` as [nblk, bs * E], which the ops below read as rows of E
     * (a contiguous buffer at another width, core/runtime and libr4d's r4d_rows_pitch). */
    op_gather  = rw(b, gather(nblk, bs * E), {}, { b_all });
    op_mem     = rw(b, gemm(NM, W, E, w_mem), { b_all }, { b_mem });
    op_seg     = rw(b, unary("seg_mean", NM, E), { b_all }, { b_seg });
    op_g_q     = rw(b, gather(QM, E), { b_seg }, { b_qv });
    op_g_o     = rw(b, gather(OM, E), { b_seg }, { b_oc });
    op_g_last  = rw(b, gather(1, E), { b_all }, { b_g });
    op_lex     = rw(b, RAD_OP(b, "embed_lookup",
                              RAD_PARAMS(RAD_RANGE("M", 1, LM), RAD_INT("n_embd", E),
                                         RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
                                         RAD_INT("vocab_offset", g.vocab_off)),
                              RAD_WEIGHTS(w_lm_head)),
                    {}, { b_lx });
    op_lexmean = rw(b, unary("seg_mean", LM, E), { b_lx }, { b_lex });
    op_octx    = rw(b, gemm(OM, W, E, w_octx), { b_oc }, { b_x });
    op_olex    = rw(b, gemm(OM, W, E, w_olex), { b_lex }, { b_t1 });
    op_add_x1  = rw(b, unary("add", OM, W), { b_x, b_t1 }, { b_x });
    op_oq      = rw(b, gemm(QM, W, E, w_oq), { b_qv }, { b_qq });
    op_g_oq    = rw(b, gather(OM, W), { b_qq }, { b_t1 });
    op_add_x2  = rw(b, unary("add", OM, W), { b_x, b_t1 }, { b_x });
    for (Ev& e : ev) {
        e.ln_q = rw(b, lnorm(OM, W, e.qn_w, e.qn_b), { b_x }, { b_t1 });
        e.ln_m = rw(b, lnorm(NM, W, e.mn_w, e.mn_b), { b_mem }, { b_mn });
        e.pq   = rw(b, gemmb(OM, W, W, "none", e.wq, e.bq), { b_t1 }, { b_t2 });
        e.pk   = rw(b, gemmb(NM, W, W, "none", e.wk, e.bk), { b_mn }, { b_k });
        e.pv   = rw(b, gemmb(NM, W, W, "none", e.wv, e.bv), { b_mn }, { b_v });
        e.attn = rw(b, cross(OM), { b_t2, b_k, b_v }, { b_t1 });
        e.po   = rw(b, gemmb(OM, W, W, "none", e.wo, e.bo), { b_t1, b_x }, { b_x });
        e.ln_f = rw(b, lnorm(OM, W, e.fn_w, e.fn_b), { b_x }, { b_t1 });
        e.f1   = rw(b, gemmb(OM, F, W, "gelu", e.w1, e.b1), { b_t1 }, { b_ff });
        e.f2   = rw(b, gemmb(OM, W, F, "none", e.w2, e.b2), { b_ff, b_x }, { b_x });
    }
    op_qproj   = rw(b, gemm(QM, W, E, w_q), { b_qv }, { b_bf });
    op_g_fr    = rw(b, gather(OM, W), { b_bf }, { b_fr });
    op_mul_sc  = rw(b, unary("mul", OM, W), { b_x, b_fr }, { b_t1 });
    op_score   = rw(b, rowmean(OM, W, (double)W / std::sqrt((double)W)), { b_t1 }, { b_sc });
    op_route   = rw(b, unary("seg_softmax_pool", OM, W), { b_sc, b_x }, { b_sum });
    op_ln_osn  = rw(b, lnorm(QM, W, osn_w, osn_b), { b_sum }, { b_f1 });
    op_add_f1  = rw(b, unary("add", QM, W), { b_bf, b_f1 }, { b_f });
    op_glob    = rw(b, gemm(1, W, E, w_glob), { b_g }, { b_gw });
    op_add_fg  = rw(b, unary("add", QM, W), { b_f, b_gw }, { b_f });
    op_type    = rw(b, RAD_OP(b, "embed_lookup",
                              RAD_PARAMS(RAD_RANGE("M", 1, QM), RAD_INT("n_embd", W),
                                         RAD_INT("n_vocab", 3), RAD_STR("dtype", "bf16"),
                                         RAD_INT("vocab_offset", 0)),
                              RAD_WEIGHTS(w_type)),
                    {}, { b_f1 });
    op_add_f2  = rw(b, unary("add", QM, W), { b_f, b_f1 }, { b_f });
    for (Dec& d : dec) {
        d.ln1   = rw(b, lnorm(QM, W, d.n1w, d.n1b), { b_f }, { b_f1 });
        d.qkv   = rw(b, gemmb(QM, 3 * W, W, "none", d.sa_w, d.sa_b), { b_f1 }, { b_fq });
        d.sattn = rw(b, RAD_OP(b, "attn_dense",
                               RAD_PARAMS(RAD_RANGE("M", 1, QM), RAD_INT("head_dim", HD),
                                          RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                          RAD_STR("dtype", "bf16"),
                                          RAD_F64("scale", 1.0 / std::sqrt((double)HD)),
                                          RAD_INT("max_seqlen", QM)),
                               RAD_NOWEIGHTS),
                      { b_fq }, { b_f1 });
        d.so    = rw(b, gemmb(QM, W, W, "none", d.so_w, d.so_b), { b_f1, b_f }, { b_f });
        d.ln2   = rw(b, lnorm(QM, W, d.n2w, d.n2b), { b_f }, { b_f1 });
        d.pq    = rw(b, gemmb(QM, W, W, "none", d.cq_w, d.cq_b), { b_f1 }, { b_f2 });
        d.pk    = rw(b, gemmb(NM, W, W, "none", d.ck_w, d.ck_b), { b_mem }, { b_k });
        d.pv    = rw(b, gemmb(NM, W, W, "none", d.cv_w, d.cv_b), { b_mem }, { b_v });
        d.cattn = rw(b, cross(QM), { b_f2, b_k, b_v }, { b_f1 });
        d.co    = rw(b, gemmb(QM, W, W, "none", d.co_w, d.co_b), { b_f1, b_f }, { b_f });
        d.ln3   = rw(b, lnorm(QM, W, d.n3w, d.n3b), { b_f }, { b_f1 });
        d.f1    = rw(b, gemmb(QM, F, W, "gelu", d.w1, d.b1), { b_f1 }, { b_ff });
        d.f2    = rw(b, gemmb(QM, W, F, "none", d.w2, d.b2), { b_ff, b_f }, { b_f });
    }
    op_ln_fn   = rw(b, lnorm(QM, W, fn_w, fn_b), { b_f }, { b_fn });
    /* The scorer, an option at a time against its question's field. */
    op_g_fr2   = rw(b, gather(OM, W), { b_fn }, { b_fr });
    op_ln_on   = rw(b, lnorm(OM, W, on_w, on_b), { b_x }, { b_on });
    op_rms_a   = rw(b, rmsp(OM, W), { b_fr }, { b_na });
    op_rms_b   = rw(b, rmsp(OM, W), { b_on }, { b_nb });
    op_mul_cos = rw(b, unary("mul", OM, W), { b_na, b_nb }, { b_t1 });
    op_mean_cos = rw(b, rowmean(OM, W, 1.0), { b_t1 }, { b_out3 });
    op_g_an    = rw(b, gather(OM, E), { b_qv }, { b_an });
    op_add_an  = rw(b, unary("add", OM, E), { b_an, b_g }, { b_an });
    op_rms_an  = rw(b, rmsp(OM, E), { b_an }, { b_p });
    op_rms_lex = rw(b, rmsp(OM, E), { b_lex }, { b_ln });
    op_mul_pri = rw(b, unary("mul", OM, E), { b_p, b_ln }, { b_an });
    op_mean_pri = rw(b, rowmean(OM, E, 1.0), { b_an }, { b_out3 });
    op_neg1    = rw(b, RAD_OP(b, "fill",
                              RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", W),
                                         RAD_STR("dtype", "bf16"), RAD_F64("value", -1.0)),
                              RAD_NOWEIGHTS),
                    {}, { b_neg1 });
    op_nd      = rw(b, unary("mul", OM, W), { b_on, b_neg1 }, { b_nd });
    op_d       = rw(b, unary("add", OM, W), { b_fr, b_nd }, { b_d });
    op_relu_d  = rw(b, unary("relu", OM, W), { b_d }, { b_abs });
    op_nd2     = rw(b, unary("mul", OM, W), { b_d, b_neg1 }, { b_nd });
    op_relu_nd = rw(b, unary("relu", OM, W), { b_nd }, { b_nd });
    op_abs     = rw(b, unary("add", OM, W), { b_abs, b_nd }, { b_abs });
    op_prod    = rw(b, unary("mul", OM, W), { b_fr, b_on }, { b_prod });
    /* The four feature blocks side by side in `feat`: the field rows are gathered straight into
     * the first, the other three copied in (cast at one dtype is the copy op), so the scorer's
     * first linear is one GEMM over 4 W, rounded once as the reference rounds it. */
    auto copy = [&]() {
        return RAD_OP(b, "cast",
                      RAD_PARAMS(RAD_RANGE("M", 1, OM), RAD_INT("n", W), RAD_STR("from", "bf16"),
                                 RAD_STR("to", "bf16")),
                      RAD_NOWEIGHTS);
    };
    op_cp_fr   = rw(b, copy(), { b_fr }, { b_feat });
    op_cp_on   = rw(b, copy(), { b_on }, { b_feat });
    op_cp_prod = rw(b, copy(), { b_prod }, { b_feat });
    op_cp_abs  = rw(b, copy(), { b_abs }, { b_feat });
    op_rs1     = rw(b, gemmb(OM, W, 4 * W, "gelu", rs1_w, rs1_b), { b_feat }, { b_r });
    op_rs2     = rw(b, gemmb(OM, 1, W, "none", rs2_w, rs2_b), { b_r }, { b_out3 });
    op_cast    = rw(b, RAD_OP(b, "cast",
                              RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 3 * OM),
                                         RAD_STR("from", "bf16"), RAD_STR("to", "f32")),
                              RAD_NOWEIGHTS),
                    { b_out3 }, { b_logits });

    rad_note(b, "clef's joint schema head on the card: width %lld, %lld heads of %lld, "
                "feedforward %lld, %lld routing + %lld decoder layers; hn cached as a FULL group "
                "of block %lld (%lld rows a context); at most %d questions, %d options, %d option "
                "tokens a request; three values an option into the logits row",
             (long long)W, (long long)H, (long long)HD, (long long)F, R, D, (long long)bs,
             (long long)NM, (int)QM, (int)OM, (int)LM);
    on = true;
    return RAD_OK;
}

inline void ClefHead::step_tokens(RadCtx* c, const RadBatch* batch, rad_buf b_h) {
    if (!on || batch->n_tok <= 0) return;
    const RadKVGroupBatch* kb = group_of(batch);
    if (!kb) { rad_step_fail(c, "clef: the head's cache group is not in the step batch"); return; }
    const int64_t T = batch->n_tok;
    RAD_ISSUE_N(c, op_hn, T, brows(b_h, T), RAD_W(hn_w), RAD_W(hn_b), brows(b_hn, T));
    RAD_ISSUE_N(c, op_store, T, bcol(b_hn, 0, E / 2, T), bcol(b_hn, E / 2, E / 2, T),
                praw(kb->slot_mapping, RAD_I32, T), kv_cache(kv, layer));
}

inline void ClefHead::step_heads(RadCtx* c, const RadBatch* batch, rad_buf b_logits,
                                 rad_weight w_lm_head) {
    if (!on || !batch->plugin_cu_h || !batch->plugin_words_h) return;
    const RadKVGroupBatch* kb = group_of(batch);
    if (!kb) { rad_step_fail(c, "clef: the head's cache group is not in the step batch"); return; }
    const int64_t QM = RAD_CLEF_MAX_QUESTIONS, OM = RAD_CLEF_MAX_OPTIONS, LM = RAD_CLEF_MAX_LEX;
    const int64_t SM = 2 * (QM + OM) + 1;
    for (int64_t s = 0; s < batch->n_seq; ++s) {
        const int32_t w0 = batch->plugin_cu_h[s], w1 = batch->plugin_cu_h[s + 1];
        if (w1 <= w0) continue;
        if (w1 - w0 < RAD_CLEF_HDR) { rad_step_fail(c, "clef: a request's words have no header"); return; }
        const int32_t* hw = batch->plugin_words_h + w0;
        const int32_t* dw = batch->plugin_words + w0;
        const RadClefWords L = rad_clef_words_read(hw);
        const int64_t N = L.n, Q = L.q, O = L.o, Lx = L.l, S = L.s;
        const int32_t row = hw[RAD_CLEF_W_ROW];
        const int64_t nb = (N + bs - 1) / bs;
        if (w1 - w0 < L.total || N < 1 || N > NM || Q < 1 || Q > QM || O < 1 || O > OM ||
            Lx < 1 || Lx > LM || S < 1 || S > SM || row < 0 || row >= batch->n_out ||
            nb > kb->max_blocks) {
            rad_step_fail(c, "clef: a request's words are out of the head's bounds");
            return;
        }
        auto dev = [&](int32_t off, int64_t n) { return praw(dw + off, RAD_I32, n); };
        auto out3 = [&](int64_t at, int64_t rows) { return bcol_at(b_out3, at, 1, 0, 1, rows); };
        const int32_t* bt = kb->block_table + s * kb->block_table_pitch;

        /* Every row of hn, from the sequence's blocks. */
        RAD_ISSUE_N(c, op_gather, nb, kv_cache(kv, layer), praw(bt, RAD_I32, nb),
                    brows(b_all, nb * bs));
        RAD_ISSUE_N(c, op_mem, N, brows(b_all, N), RAD_W(w_mem), brows(b_mem, N));
        /* The question, option and global vectors. */
        RAD_ISSUE_N(c, op_seg, N, brows(b_all, N), dev(L.seg_cu, S + 1), brows(b_seg, S));
        RAD_ISSUE_N(c, op_g_q, Q, brows(b_seg, S), dev(L.q_seg, Q), brows(b_qv, Q));
        RAD_ISSUE_N(c, op_g_o, O, brows(b_seg, S), dev(L.o_seg, O), brows(b_oc, O));
        RAD_ISSUE_N(c, op_g_last, 1, brows(b_all, N), dev(L.last, 1), brows(b_g, 1));
        RAD_ISSUE_N(c, op_lex, Lx, dev(L.lex_ids, Lx), RAD_W(w_lm_head), brows(b_lx, Lx));
        RAD_ISSUE_N(c, op_lexmean, Lx, brows(b_lx, Lx), dev(L.lex_cu, O + 1), brows(b_lex, O));
        /* Option queries, then the evidence layers over the memory. */
        RAD_ISSUE_N(c, op_octx, O, brows(b_oc, O), RAD_W(w_octx), brows(b_x, O));
        RAD_ISSUE_N(c, op_olex, O, brows(b_lex, O), RAD_W(w_olex), brows(b_t1, O));
        RAD_ISSUE_N(c, op_add_x1, O, brows(b_x, O), brows(b_t1, O), brows(b_x, O));
        RAD_ISSUE_N(c, op_oq, Q, brows(b_qv, Q), RAD_W(w_oq), brows(b_qq, Q));
        RAD_ISSUE_N(c, op_g_oq, O, brows(b_qq, Q), dev(L.o_q, O), brows(b_t1, O));
        RAD_ISSUE_N(c, op_add_x2, O, brows(b_x, O), brows(b_t1, O), brows(b_x, O));
        for (const Ev& e : ev) {
            RAD_ISSUE_N(c, e.ln_q, O, brows(b_x, O), RAD_W(e.qn_w), RAD_W(e.qn_b), brows(b_t1, O));
            RAD_ISSUE_N(c, e.ln_m, N, brows(b_mem, N), RAD_W(e.mn_w), RAD_W(e.mn_b), brows(b_mn, N));
            RAD_ISSUE_N(c, e.pq, O, brows(b_t1, O), RAD_W(e.wq), RAD_W(e.bq), RAD_NONE, brows(b_t2, O));
            RAD_ISSUE_N(c, e.pk, N, brows(b_mn, N), RAD_W(e.wk), RAD_W(e.bk), RAD_NONE, brows(b_k, N));
            RAD_ISSUE_N(c, e.pv, N, brows(b_mn, N), RAD_W(e.wv), RAD_W(e.bv), RAD_NONE, brows(b_v, N));
            RAD_ISSUE_N(c, e.attn, O, brows(b_t2, O), brows(b_k, N), brows(b_v, N),
                        dev(L.cu_o, 2), dev(L.cu_k, 2), brows(b_t1, O));
            RAD_ISSUE_N(c, e.po, O, brows(b_t1, O), RAD_W(e.wo), RAD_W(e.bo), brows(b_x, O), brows(b_x, O));
            RAD_ISSUE_N(c, e.ln_f, O, brows(b_x, O), RAD_W(e.fn_w), RAD_W(e.fn_b), brows(b_t1, O));
            RAD_ISSUE_N(c, e.f1, O, brows(b_t1, O), RAD_W(e.w1), RAD_W(e.b1), RAD_NONE, brows(b_ff, O));
            RAD_ISSUE_N(c, e.f2, O, brows(b_ff, O), RAD_W(e.w2), RAD_W(e.b2), brows(b_x, O), brows(b_x, O));
        }
        /* Base fields, each question's routing over its options, and the field vectors. */
        RAD_ISSUE_N(c, op_qproj, Q, brows(b_qv, Q), RAD_W(w_q), brows(b_bf, Q));
        RAD_ISSUE_N(c, op_g_fr, O, brows(b_bf, Q), dev(L.o_q, O), brows(b_fr, O));
        RAD_ISSUE_N(c, op_mul_sc, O, brows(b_x, O), brows(b_fr, O), brows(b_t1, O));
        RAD_ISSUE_N(c, op_score, O, brows(b_t1, O), brows(b_sc, O));
        RAD_ISSUE_N(c, op_route, O, brows(b_sc, O), brows(b_x, O), dev(L.opt_cu, Q + 1), brows(b_sum, Q));
        RAD_ISSUE_N(c, op_ln_osn, Q, brows(b_sum, Q), RAD_W(osn_w), RAD_W(osn_b), brows(b_f1, Q));
        RAD_ISSUE_N(c, op_add_f1, Q, brows(b_bf, Q), brows(b_f1, Q), brows(b_f, Q));
        RAD_ISSUE_N(c, op_glob, 1, brows(b_g, 1), RAD_W(w_glob), brows(b_gw, 1));
        RAD_ISSUE_N(c, op_add_fg, Q, brows(b_f, Q), brows(b_gw, 1), brows(b_f, Q));
        RAD_ISSUE_N(c, op_type, Q, dev(L.type, Q), RAD_W(w_type), brows(b_f1, Q));
        RAD_ISSUE_N(c, op_add_f2, Q, brows(b_f, Q), brows(b_f1, Q), brows(b_f, Q));
        for (const Dec& d : dec) {
            RAD_ISSUE_N(c, d.ln1, Q, brows(b_f, Q), RAD_W(d.n1w), RAD_W(d.n1b), brows(b_f1, Q));
            RAD_ISSUE_N(c, d.qkv, Q, brows(b_f1, Q), RAD_W(d.sa_w), RAD_W(d.sa_b), RAD_NONE, brows(b_fq, Q));
            RAD_ISSUE_N(c, d.sattn, Q, bcol(b_fq, 0, W, Q), bcol(b_fq, W, W, Q),
                        bcol(b_fq, 2 * W, W, Q), dev(L.cu_q, 2), brows(b_f1, Q));
            RAD_ISSUE_N(c, d.so, Q, brows(b_f1, Q), RAD_W(d.so_w), RAD_W(d.so_b), brows(b_f, Q), brows(b_f, Q));
            RAD_ISSUE_N(c, d.ln2, Q, brows(b_f, Q), RAD_W(d.n2w), RAD_W(d.n2b), brows(b_f1, Q));
            RAD_ISSUE_N(c, d.pq, Q, brows(b_f1, Q), RAD_W(d.cq_w), RAD_W(d.cq_b), RAD_NONE, brows(b_f2, Q));
            RAD_ISSUE_N(c, d.pk, N, brows(b_mem, N), RAD_W(d.ck_w), RAD_W(d.ck_b), RAD_NONE, brows(b_k, N));
            RAD_ISSUE_N(c, d.pv, N, brows(b_mem, N), RAD_W(d.cv_w), RAD_W(d.cv_b), RAD_NONE, brows(b_v, N));
            RAD_ISSUE_N(c, d.cattn, Q, brows(b_f2, Q), brows(b_k, N), brows(b_v, N),
                        dev(L.cu_q, 2), dev(L.cu_k, 2), brows(b_f1, Q));
            RAD_ISSUE_N(c, d.co, Q, brows(b_f1, Q), RAD_W(d.co_w), RAD_W(d.co_b), brows(b_f, Q), brows(b_f, Q));
            RAD_ISSUE_N(c, d.ln3, Q, brows(b_f, Q), RAD_W(d.n3w), RAD_W(d.n3b), brows(b_f1, Q));
            RAD_ISSUE_N(c, d.f1, Q, brows(b_f1, Q), RAD_W(d.w1), RAD_W(d.b1), RAD_NONE, brows(b_ff, Q));
            RAD_ISSUE_N(c, d.f2, Q, brows(b_ff, Q), RAD_W(d.w2), RAD_W(d.b2), brows(b_f, Q), brows(b_f, Q));
        }
        RAD_ISSUE_N(c, op_ln_fn, Q, brows(b_f, Q), RAD_W(fn_w), RAD_W(fn_b), brows(b_fn, Q));
        /* The scorer: the cosine, the lexical prior and the residual, one value an option, into
         * out3's three stretches (prior | cosine | residual). */
        RAD_ISSUE_N(c, op_g_fr2, O, brows(b_fn, Q), dev(L.o_q, O), brows(b_fr, O));
        RAD_ISSUE_N(c, op_ln_on, O, brows(b_x, O), RAD_W(on_w), RAD_W(on_b), brows(b_on, O));
        RAD_ISSUE_N(c, op_rms_a, O, brows(b_fr, O), brows(b_na, O));
        RAD_ISSUE_N(c, op_rms_b, O, brows(b_on, O), brows(b_nb, O));
        RAD_ISSUE_N(c, op_mul_cos, O, brows(b_na, O), brows(b_nb, O), brows(b_t1, O));
        RAD_ISSUE_N(c, op_mean_cos, O, brows(b_t1, O), out3(OM, O));
        RAD_ISSUE_N(c, op_g_an, O, brows(b_qv, Q), dev(L.o_q, O), brows(b_an, O));
        RAD_ISSUE_N(c, op_add_an, O, brows(b_an, O), brows(b_g, 1), brows(b_an, O));
        RAD_ISSUE_N(c, op_rms_an, O, brows(b_an, O), brows(b_p, O));
        RAD_ISSUE_N(c, op_rms_lex, O, brows(b_lex, O), brows(b_ln, O));
        RAD_ISSUE_N(c, op_mul_pri, O, brows(b_p, O), brows(b_ln, O), brows(b_an, O));
        RAD_ISSUE_N(c, op_mean_pri, O, brows(b_an, O), out3(0, O));
        RAD_ISSUE_N(c, op_neg1, 1, brows(b_neg1, 1));
        RAD_ISSUE_N(c, op_nd, O, brows(b_on, O), brows(b_neg1, 1), brows(b_nd, O));
        RAD_ISSUE_N(c, op_d, O, brows(b_fr, O), brows(b_nd, O), brows(b_d, O));
        RAD_ISSUE_N(c, op_relu_d, O, brows(b_d, O), brows(b_abs, O));
        RAD_ISSUE_N(c, op_nd2, O, brows(b_d, O), brows(b_neg1, 1), brows(b_nd, O));
        RAD_ISSUE_N(c, op_relu_nd, O, brows(b_nd, O), brows(b_nd, O));
        RAD_ISSUE_N(c, op_abs, O, brows(b_abs, O), brows(b_nd, O), brows(b_abs, O));
        RAD_ISSUE_N(c, op_prod, O, brows(b_fr, O), brows(b_on, O), brows(b_prod, O));
        /* [field | option | field*option | |field-option|], then the scorer. The field rows go
         * in again (a copy into the first block) so every block arrives the same way. */
        RAD_ISSUE_N(c, op_cp_fr, O, brows(b_fr, O), bcol(b_feat, 0, W, O));
        RAD_ISSUE_N(c, op_cp_on, O, brows(b_on, O), bcol(b_feat, W, W, O));
        RAD_ISSUE_N(c, op_cp_prod, O, brows(b_prod, O), bcol(b_feat, 2 * W, W, O));
        RAD_ISSUE_N(c, op_cp_abs, O, brows(b_abs, O), bcol(b_feat, 3 * W, W, O));
        RAD_ISSUE_N(c, op_rs1, O, brows(b_feat, O), RAD_W(rs1_w), RAD_W(rs1_b), RAD_NONE, brows(b_r, O));
        RAD_ISSUE_N(c, op_rs2, O, brows(b_r, O), RAD_W(rs2_w), RAD_W(rs2_b), RAD_NONE, out3(2 * OM, O));
        /* The three stretches, as f32, into the request's logits row (one row of 3 OM). */
        RAD_ISSUE_N(c, op_cast, 1, brows(b_out3, 3 * OM),
                    bcol_at(b_logits, row, n_vocab, 0, 3 * OM, 1));
    }
}

/* The zero logits rows a trunk leaves when it skips its lm_head for rows a decision head answers
 * (the qwen35 plugins' `op_logits_fill`): declared here, by the plugin that wants the skip, so a
 * trunk serving a plain container declares the same graph as before. */
inline rad_op clef_logits_fill(RadBuilder* b, Names& nm, const Geom& g, rad_buf b_logits) {
    return rw(b, RAD_OP(b, "fill",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_vocab),
                                   RAD_STR("dtype", "f32"), RAD_F64("value", 0.0)),
                        RAD_NOWEIGHTS),
              {}, { b_logits });
}

}  /* namespace arch */
}  /* namespace rad */
