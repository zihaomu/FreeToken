/* rad_block_museglimmer_vision.h -- Muse Glimmer's vision tower (transformers'
 * MuseGlimmerVisionModel, an InternVit-shaped stack), its adapter and the perception norm, served
 * by museglimmer.
 *
 * What it computes, per encoder pass (RadBatch::enc), in the reference's order:
 *
 *   x  = patches @ Wp^T                          the patch embedding as a linear layer, bias-less;
 *                                                a patch row is (temporal, channel, y, x)
 *   x += position table resampled to the grid    `grid_embed_pad`, bilinear at half-pixel
 *                                                centres, align_corners=False, out-of-range taps
 *                                                zero-weighted (grid_sample padding zeros)
 *   x  = LN_pre(x)                               LayerNorm, eps `layer_norm_eps`
 *   x  = x[window order]                         `window_order`: the 32x32-patch window permutation
 *                                                (per frame, ragged edges smaller windows), its
 *                                                inverse, per-slot rotary positions and the
 *                                                per-window cumulative boundaries
 *   depth times:
 *     h  = LN1(x)                                LayerNorm, eps 1e-5 (the blocks hard-code it)
 *     q|k|v = qkv(h)                             the reference's three biased projections, joined
 *                                                q first into one GEMM as the reference's fused
 *                                                tensors would read
 *     rope(q, k)                                 `rope` axial, the whole head, sections
 *                                                head_dim/4 a side on theta^(-j/(head_dim/4)),
 *                                                positions (column + 1, row + 1) -- HF's
 *                                                position_ids.flip(-1) + 1
 *     x += proj(attn(q, k, v))                   non-causal, scale 1/sqrt(head_dim). A
 *                                                window_attention layer attends within one window
 *                                                (cu = the per-window boundaries in permuted
 *                                                order); a full_attention layer attends within
 *                                                its frame (cu = the frame boundaries). The
 *                                                window permutation never moves a patch across
 *                                                frames, so the frame boundaries among permuted
 *                                                rows are the boundaries among natural rows:
 *                                                `enc_cu` is passed unchanged, exactly what the
 *                                                reference passes (modeling_muse_glimmer.py maps
 *                                                full_attention to the frame cu_seqlens while the
 *                                                rows stay permuted).
 *     x += fc2(gelu(fc1(LN2(x))))                both biased; `gelu` the exact erf form the
 *                                                config's hidden_act names, in gemm_nt_bias's
 *                                                epilogue
 *   x  = x[inv]                                  the window order undone (gather_rows)
 *   x  = LN_post(x)                              LayerNorm, eps `layer_norm_eps`
 *   y  = pixel_shuffle(x)                        the merge (`pixtral_merge`): each 2x2 block's
 *                                                four patches fold into one row of merge^2 *
 *                                                width, channels interleaved -- the reference's
 *                                                [block, merge^2, dim].permute(0, 2, 1)
 *   y  = gelu(fc2(gelu(fc1(y))))                 the adapter: both linears bias-less, TWO
 *                                                activations (projector_hidden_act)
 *   y  = y @ Wproj^T                             the projection, bias-less, the model's width
 *   out = rms(y)                                 `perception_emb_norm`, a weightless RMS norm at
 *                                                the TEXT model's rms_norm_eps
 *
 * The reference permutes AFTER ln_pre, so the tower works in window order and un-permutes before
 * ln_post, exactly as above. The processor stages patches in the reference's raster order; the
 * merge is a real fold (not a view: the reference interleaves the four patches' channels, which
 * four consecutive rows do not spell), and its output is what the adapter reads.
 *
 * Every rank runs the whole tower with replicated weights, as rad_block_vit.h's does, and the
 * weights are RAD_ACCESS_PER_REQUEST. Activations are sized by the pass (RadBuildCtx::
 * max_enc_patches rows), except in a sizing declare (shape_probe), where rows follow the probe's
 * step; the attention's longest segment stays at the real pass so the kernel rows do not move.
 */
#ifndef RAD_BLOCK_MUSEGLIMMER_VISION_H
#define RAD_BLOCK_MUSEGLIMMER_VISION_H

#include "rad_arch.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace rad {
namespace arch {

struct MuseTower {
    /* Geometry, from the container's vision_config (never aliased: only the full prefix). The
     * adapter's widths are top-level keys of the family config. */
    int64_t depth = 0, width = 0, heads = 0, head_dim = 0, ff = 0, patch = 0, temporal = 0;
    int64_t merge = 0, side = 0, npos = 0, proj_ff = 0, out_hidden = 0, n_embd = 0;
    int64_t patch_dim = 0, P = 0;
    double  eps = 1e-5;          /* ln_pre / ln_post; the blocks hard-code 1e-5 */
    float   theta = 10000.0f;
    bool    on = false;

    struct Layer {
        bool full = false;     /* full_attention: frame-wide attention, NoPE-free (rotary runs
                                * on every layer of this tower, full or window) */
        rad_weight n1w = 0, n1b = 0, qkv_w = 0, qkv_b = 0, out_w = 0, out_b = 0;
        rad_weight n2w = 0, n2b = 0, up_w = 0, up_b = 0, down_w = 0, down_b = 0;
        rad_op ln1 = 0, qkv = 0, rope = 0, attn = 0, proj = 0, ln2 = 0, up = 0, down = 0;
        int64_t max_seg = 0;   /* the kernel band's longest segment */
    };
    std::vector<Layer> L;
    rad_weight w_patch = 0, w_pos = 0, lw_pre = 0, lb_pre = 0, lw_post = 0, lb_post = 0;
    rad_weight w_a1 = 0, w_a2 = 0, w_proj = 0;
    rad_op op_patch = 0, op_pos = 0, op_lpre = 0, op_wo = 0, op_gidx = 0, op_ginv = 0;
    rad_op op_lpost = 0, op_merge = 0, op_m1 = 0, op_g1 = 0, op_m2 = 0, op_g2 = 0, op_proj = 0,
           op_perc = 0;
    rad_buf b_x = 0, b_h = 0, b_p = 0, b_qkv = 0, b_att = 0, b_ff = 0;
    rad_buf b_idx = 0, b_inv = 0, b_pos = 0, b_cuw = 0;
    rad_buf b_mh = 0, b_m1 = 0, b_m2 = 0, b_out = 0;
    char    sec[48] = {};

    /* Whether the container carries this family's tower. */
    static bool present(const RadModelMeta* m) {
        return rad_meta_geti(m, "vision_config.num_hidden_layers", 0) > 0;
    }

    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, int first_group);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int MuseTower::declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx,
                              Names& nm, int64_t model_embd, int first_group) {
    on = false;
    if (!present(m) || ctx->max_enc_patches <= 0) return RAD_OK;

    depth    = rad_meta_geti(m, "vision_config.num_hidden_layers", 0);
    width    = rad_meta_geti(m, "vision_config.hidden_size", 0);
    heads    = rad_meta_geti(m, "vision_config.num_attention_heads", 0);
    ff       = rad_meta_geti(m, "vision_config.intermediate_size", 0);
    patch    = rad_meta_geti(m, "vision_config.patch_size", 0);
    temporal = rad_meta_geti(m, "vision_config.patch_temporal", 2);
    merge    = rad_meta_geti(m, "vision_config.merge_size", 2);
    side     = rad_meta_geti(m, "vision_config.pos_emb_height", 0);
    npos     = rad_meta_geti(m, "vision_config.pos_emb_height", 0) *
               rad_meta_geti(m, "vision_config.pos_emb_width", 0);
    eps      = rad_meta_getf(m, "vision_config.layer_norm_eps", 1e-5);
    theta    = (float)rad_meta_getf(m, "vision_config.rope_parameters.rope_theta", 10000.0);
    proj_ff    = rad_meta_geti(m, "projector_hidden_size", 4096);
    out_hidden = rad_meta_geti(m, "out_hidden_size", 6144);
    n_embd     = model_embd;
    if (depth <= 0 || width <= 0 || heads <= 0 || ff <= 0 || patch <= 0 || temporal <= 0 ||
        merge <= 0 || side <= 0 || proj_ff <= 0 || out_hidden <= 0 || width % heads ||
        rad_meta_geti(m, "vision_config.pos_emb_width", 0) != side) {
        fprintf(stderr, "radiance: museglimmer: the container's vision_config is incomplete or "
                        "inconsistent (depth %lld, hidden %lld, heads %lld, ff %lld, patch %lld, "
                        "temporal %lld, merge %lld, position grid %lld x %lld, projector %lld, "
                        "merged %lld)\n", (long long)depth, (long long)width, (long long)heads,
                (long long)ff, (long long)patch, (long long)temporal, (long long)merge,
                (long long)side, (long long)rad_meta_geti(m, "vision_config.pos_emb_width", 0),
                (long long)proj_ff, (long long)out_hidden);
        return RAD_E_INVAL;
    }
    /* The merge is `pixtral_merge`, whose kernels fold a 2x2 block: the only factor Muse Glimmer
     * ships and the only one the op serves. */
    if (merge != 2) {
        fprintf(stderr, "radiance: museglimmer: this vision tower states merge_size %lld; the "
                        "merge op folds a 2x2 block\n", (long long)merge);
        return RAD_E_UNSUPPORTED;
    }
    /* The exact-erf GELU is what this tower and its adapter state; another activation is another
     * tower (the reference would compute a different function). */
    const char* act = rad_meta_gets(m, "vision_config.hidden_act", "gelu");
    const char* pact = rad_meta_gets(m, "projector_hidden_act", "gelu");
    const char* rt = rad_meta_gets(m, "vision_config.rope_parameters.rope_type", "axial");
    if (std::strcmp(act, "gelu") || std::strcmp(pact, "gelu")) {
        fprintf(stderr, "radiance: museglimmer: this vision tower has hidden_act '%s' and "
                        "projector_hidden_act '%s'; it serves the exact gelu both call for\n",
                act, pact);
        return RAD_E_UNSUPPORTED;
    }
    /* `axial` is Meta's spelling and `default` NVIDIA's ModelOpt export of the same tower: the
     * reference's vision embedder has one rotary (axial), so both name it. Anything else is
     * another tower. */
    if (std::strcmp(rt, "axial") && std::strcmp(rt, "default")) {
        fprintf(stderr, "radiance: museglimmer: this vision tower states rope_type '%s'; it "
                        "serves the axial 2-D rotary the tower's rotary embedder raises on "
                        "leaving\n", rt);
        return RAD_E_UNSUPPORTED;
    }
    /* The tower's full layers: every fourth, counted from the start, plus the last (the config's
     * own rule, MuseGlimmerVisionConfig.__post_init__). The shipped array must say the same
     * thing; a window layer where the rule puts a full one changes what the model saw. */
    std::vector<int> full;
    const char* kinds = rad_meta_gets(m, "vision_config.layer_types", "");
    if (kinds && *kinds) {
        RAD_ARCH_TRY(parse_layer_kinds("museglimmer vision", kinds, {"full_attention"},
                                       "window_attention", &full));
        if ((int64_t)full.size() != depth) {
            fprintf(stderr, "radiance: museglimmer: vision_config.layer_types names %zu layers "
                            "and the tower has %lld\n", full.size(), (long long)depth);
            return RAD_E_INVAL;
        }
        for (int64_t l = 0; l < depth; ++l) {
            const bool rule = is_full_attention(l, 4) || l == depth - 1;
            if ((full[(size_t)l] != 0) != rule) {
                fprintf(stderr, "radiance: museglimmer: vision_config.layer_types makes tower "
                                "layer %lld %s and the family rule (every fourth from the first, "
                                "plus the last) makes it %s\n", (long long)l,
                        full[(size_t)l] ? "full" : "window", rule ? "full" : "window");
                return RAD_E_UNSUPPORTED;
            }
        }
    } else {
        full.assign((size_t)depth, 0);
        for (int64_t l = 0; l < depth; ++l)
            full[(size_t)l] = is_full_attention(l, 4) || l == depth - 1;
    }

    head_dim  = width / heads;
    patch_dim = 3 * temporal * patch * patch;
    /* The 2-D rotary: half the head a side, each on its own ladder (arange(0, head_dim/2, 2)
     * frequencies, sections head_dim/4 a side). */
    if (head_dim % 4) return RAD_E_SHAPE;
    std::snprintf(sec, sizeof sec, "%lld %lld", (long long)(head_dim / 4),
                  (long long)(head_dim / 4));
    /* The merger's rows replace token embeddings, so they must be the model's width. */
    if (out_hidden != merge * merge * width) {
        fprintf(stderr, "radiance: museglimmer: out_hidden_size %lld is not merge %lld squared "
                        "times the tower's width %lld, whose view the adapter's first layer "
                        "reads\n", (long long)out_hidden, (long long)merge, (long long)width);
        return RAD_E_INVAL;
    }
    /* The tower emits merged rows the width of the model's tokens: the caller passes the model's
     * width and the projection is declared at it, so no separate check exists. */

    const int64_t mm = merge * merge;
    P = ctx->max_enc_patches;
    if (ctx->shape_probe) {
        /* A sizing declare: rows follow the probe's step (rounded to whole merged rows), keeping
         * the activations out of the levels a decode step runs at. The kernel bands stay at the
         * real pass: the engine runs the real declare's kernels at every level. */
        const int64_t r = std::min<int64_t>(P, ctx->max_tok);
        P = std::max<int64_t>(mm, r / mm * mm);
    }
    if (P % mm) return RAD_E_SHAPE;
    const int64_t R = P / mm;
    const int64_t ACC = RAD_ACCESS_PER_REQUEST;
    /* The window side in patches (the reference states it in pixels and divides by patch_size).
     * A window_attention layer's longest segment is one window. */
    const int64_t win_seg = side * side;

    /* ---- weights, replicated. Model-level groups for the stem and the adapter; one group per
     * block from `first_group` on, so a tower block is its own movement unit. */
    auto bw = [&](const char* dn, const char* src, std::initializer_list<int64_t> shp,
                  RadWeightGroup g) -> rad_weight {
        if (map_copy(b, dn, src) < 0) return 0;
        return decl_w(b, dn, RAD_BF16, shp, ACC, RAD_SHARD_NONE, g);
    };
    w_patch = bw(nm.f("v.patch.weight"),
                 nm.ckpt("model.vision_tower.patch_embedder.patch_embedding.weight"),
                 { width, patch_dim }, grp_model());
    w_pos   = bw(nm.f("v.pos.weight"),
                 nm.ckpt("model.vision_tower.patch_embedder.position_embedding_table.weight"),
                 { npos, width }, grp_model());
    lw_pre  = bw(nm.f("v.lnpre.weight"), nm.ckpt("model.vision_tower.ln_pre.weight"), { width },
                 grp_model());
    lb_pre  = bw(nm.f("v.lnpre.bias"), nm.ckpt("model.vision_tower.ln_pre.bias"), { width },
                 grp_model());
    L.assign((size_t)depth, Layer{});
    for (int64_t i = 0; i < depth; ++i) {
        Layer& l = L[(size_t)i];
        const RadWeightGroup g = grp_layer(first_group + (int)i);
        const int li = (int)i;
        auto src = [&](const char* s) { return nm.ckpt("model.vision_tower.layers.%d.%s", li, s); };
        auto dn  = [&](const char* s) { return nm.f("v.blk.%d.%s", li, s); };
        l.n1w   = bw(dn("norm1.weight"), src("norm1.weight"), { width }, g);
        l.n1b   = bw(dn("norm1.bias"), src("norm1.bias"), { width }, g);
        /* The reference's three biased projections, joined q | k | v into one GEMM. */
        if (map_concat(b, dn("qkv.weight"),
                       { src("attn.q_proj.weight"), src("attn.k_proj.weight"),
                         src("attn.v_proj.weight") }) < 0)
            return RAD_E_INVAL;
        l.qkv_w = decl_w(b, dn("qkv.weight"), RAD_BF16, { 3 * width, width }, ACC,
                         RAD_SHARD_NONE, g, 0, { width, width, width });
        if (map_concat(b, dn("qkv.bias"),
                       { src("attn.q_proj.bias"), src("attn.k_proj.bias"),
                         src("attn.v_proj.bias") }) < 0)
            return RAD_E_INVAL;
        l.qkv_b = decl_w(b, dn("qkv.bias"), RAD_BF16, { 3 * width }, ACC, RAD_SHARD_NONE, g, 0,
                         { width, width, width });
        l.out_w = bw(dn("proj.weight"), src("attn.proj.weight"), { width, width }, g);
        l.out_b = bw(dn("proj.bias"), src("attn.proj.bias"), { width }, g);
        l.n2w   = bw(dn("norm2.weight"), src("norm2.weight"), { width }, g);
        l.n2b   = bw(dn("norm2.bias"), src("norm2.bias"), { width }, g);
        l.up_w  = bw(dn("fc1.weight"), src("mlp.fc1.weight"), { ff, width }, g);
        l.up_b  = bw(dn("fc1.bias"), src("mlp.fc1.bias"), { ff }, g);
        l.down_w= bw(dn("fc2.weight"), src("mlp.fc2.weight"), { width, ff }, g);
        l.down_b= bw(dn("fc2.bias"), src("mlp.fc2.bias"), { width }, g);
        if (!l.n1w || !l.n1b || !l.qkv_w || !l.qkv_b || !l.out_w || !l.out_b || !l.n2w || !l.n2b ||
            !l.up_w || !l.up_b || !l.down_w || !l.down_b)
            return RAD_E_INVAL;
        l.full   = full[(size_t)i] != 0;
        l.max_seg = l.full ? ctx->max_enc_patches : win_seg;
    }
    lw_post = bw(nm.f("v.lnpost.weight"), nm.ckpt("model.vision_tower.ln_post.weight"), { width },
                 grp_model());
    lb_post = bw(nm.f("v.lnpost.bias"), nm.ckpt("model.vision_tower.ln_post.bias"), { width },
                 grp_model());
    w_a1    = bw(nm.f("v.mlp.fc1.weight"), nm.ckpt("model.vision_adapter.fc1.weight"),
                 { proj_ff, out_hidden }, grp_model());
    w_a2    = bw(nm.f("v.mlp.fc2.weight"), nm.ckpt("model.vision_adapter.fc2.weight"),
                 { proj_ff, proj_ff }, grp_model());
    w_proj  = bw(nm.f("v.proj.weight"), nm.ckpt("model.vision_projection.weight"),
                 { n_embd, proj_ff }, grp_model());
    if (!w_patch || !w_pos || !lw_pre || !lb_pre || !lw_post || !lb_post || !w_a1 || !w_a2 ||
        !w_proj)
        return RAD_E_INVAL;

    /* ---- activations. `mh` holds the merged rows: `pixtral_merge` folds each 2x2 block's four
     * patch rows into one row of mm * width in the reference's pixel_shuffle layout, and the
     * adapter's first layer reads it. */
    b_x   = decl_b(b, nm.f("v.x"),   RAD_BF16, { P, width });
    b_h   = decl_b(b, nm.f("v.h"),   RAD_BF16, { P, width });
    b_p   = decl_b(b, nm.f("v.p"),   RAD_BF16, { P, width });
    b_qkv = decl_b(b, nm.f("v.qkv"), RAD_BF16, { P, 3 * width });
    b_att = decl_b(b, nm.f("v.att"), RAD_BF16, { P, width });
    b_ff  = decl_b(b, nm.f("v.ff"),  RAD_BF16, { P, ff });
    b_idx = decl_b(b, nm.f("v.idx"), RAD_I32,  { P });
    b_inv = decl_b(b, nm.f("v.inv"), RAD_I32,  { P });
    b_pos = decl_b(b, nm.f("v.pos"), RAD_I32,  { 2, P });
    b_cuw = decl_b(b, nm.f("v.cuw"), RAD_I32,  { P + 1 });
    b_mh  = decl_b(b, nm.f("v.mh"),  RAD_BF16, { R, mm * width });
    b_m1  = decl_b(b, nm.f("v.m1"),  RAD_BF16, { R, proj_ff });
    b_m2  = decl_b(b, nm.f("v.m2"),  RAD_BF16, { R, proj_ff });
    b_out = decl_b(b, nm.f("v.out"), RAD_BF16, { R, n_embd });
    if (!b_x || !b_h || !b_p || !b_qkv || !b_att || !b_ff || !b_idx || !b_inv || !b_pos || !b_cuw ||
        !b_mh || !b_m1 || !b_m2 || !b_out)
        return RAD_E_INVAL;

    auto gemm_b = [&](int64_t rows, int64_t N, int64_t K, const char* a, rad_weight w,
                      rad_weight bias) -> rad_op {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16"), RAD_STR("act", a)),
                      RAD_WEIGHTS(w, bias));
    };
    auto gemm = [&](int64_t rows, int64_t N, int64_t K, rad_weight w) -> rad_op {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto lnorm = [&](int64_t rows, int64_t n, double e, rad_weight w, rad_weight bias) -> rad_op {
        return RAD_OP(b, "layernorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", e),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w, bias));
    };
    auto attn_op = [&](int64_t max_seg) -> rad_op {
        return RAD_OP(b, "attn_dense",
                      RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", head_dim),
                                 RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                 RAD_STR("dtype", "bf16"),
                                 RAD_F64("scale", 1.0 / std::sqrt((double)head_dim)),
                                 RAD_INT("max_seqlen", max_seg)),
                      RAD_NOWEIGHTS);
    };

    /* The stem is bias-less (the reference's patch_embedding states no bias). */
    op_patch = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("N", width),
                                       RAD_INT("K", patch_dim), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w_patch)),
                  {}, { b_x });
    op_pos   = rw(b, RAD_OP(b, "grid_embed_pad",
                            RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                       RAD_INT("side", side), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w_pos)),
                  { b_x }, { b_x });
    op_lpre  = rw(b, lnorm(P, width, eps, lw_pre, lb_pre), { b_x }, { b_h });
    op_wo    = rw(b, RAD_OP(b, "window_order",
                            RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("window", side)),
                            RAD_NOWEIGHTS),
                  {}, { b_idx, b_inv, b_pos, b_cuw });
    op_gidx  = rw(b, RAD_OP(b, "gather_rows",
                            RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { b_h, b_idx }, { b_p });
    for (Layer& l : L) {
        l.ln1  = rw(b, lnorm(P, width, 1e-5, l.n1w, l.n1b), { b_p }, { b_h });
        l.qkv  = rw(b, gemm_b(P, 3 * width, width, "none", l.qkv_w, l.qkv_b), { b_h }, { b_qkv });
        l.rope = rw(b, RAD_OP(b, "rope",
                              RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", head_dim),
                                         RAD_INT("n_head", heads), RAD_INT("n_head_kv", heads),
                                         RAD_F64("theta", theta), RAD_F64("scale", 1.0),
                                         RAD_STR("mode", "axial"),
                                         RAD_INT("rotary_dim", head_dim),
                                         RAD_STR("sections", sec)),
                              RAD_NOWEIGHTS),
                    { b_qkv, b_pos }, { b_qkv });
        /* A window layer's segment bounds are the order op's per-window cu; a full layer's are
         * the pass's frame bounds, raw (the permutation is within frames, so they read the same
         * values in permuted order). */
        l.attn = rw(b, attn_op(l.max_seg),
                    l.full ? std::initializer_list<rad_buf>{ b_qkv }
                           : std::initializer_list<rad_buf>{ b_qkv, b_cuw },
                    { b_att });
        l.proj = rw(b, gemm_b(P, width, width, "none", l.out_w, l.out_b), { b_att, b_p }, { b_p });
        l.ln2  = rw(b, lnorm(P, width, 1e-5, l.n2w, l.n2b), { b_p }, { b_h });
        l.up   = rw(b, gemm_b(P, ff, width, "gelu", l.up_w, l.up_b), { b_h }, { b_ff });
        l.down = rw(b, gemm_b(P, width, ff, "none", l.down_w, l.down_b), { b_ff, b_p }, { b_p });
    }
    op_ginv  = rw(b, RAD_OP(b, "gather_rows",
                            RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { b_p, b_inv }, { b_x });
    op_lpost = rw(b, lnorm(P, width, eps, lw_post, lb_post), { b_x }, { b_x });
    /* The reference's pixel_shuffle: each 2x2 block's four patches fold into one row with the
     * channels interleaved (block, channel, sub-patch), which is `pixtral_merge`'s layout. A view
     * over four consecutive rows cannot produce it: the concatenated form is a different tensor. */
    op_merge = rw(b, RAD_OP(b, "pixtral_merge",
                            RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                       RAD_INT("merge", merge), RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { b_x }, { b_mh });
    /* The adapter: both linears bias-less, gelu between and after (the reference's
     * act(fc2(act(fc1(x))))). */
    op_m1    = rw(b, gemm(R, proj_ff, mm * width, w_a1), { b_mh }, { b_m1 });
    op_g1    = rw(b, RAD_OP(b, "gelu",
                            RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", proj_ff),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { b_m1 }, { b_m2 });
    op_m2    = rw(b, gemm(R, proj_ff, proj_ff, w_a2), { b_m2 }, { b_m1 });
    op_g2    = rw(b, RAD_OP(b, "gelu",
                            RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", proj_ff),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { b_m1 }, { b_m2 });
    op_proj  = rw(b, gemm(R, n_embd, proj_ff, w_proj), { b_m2 }, { b_out });
    /* perception_emb_norm: MuseGlimmerRMSNorm(eps = the TEXT model's rms_norm_eps, with_scale
     * False) -- no gain operand, no centered 1+. */
    op_perc  = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", n_embd),
                                       RAD_F64("eps", m->rms_eps), RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { b_out }, { b_out });

    if (!ctx->shape_probe) {
        RadEncoderDecl e{};
        e.name        = "vision";
        e.modalities  = RAD_MM_IMAGE | RAD_MM_VIDEO;
        e.patch_dim   = patch_dim;
        e.merge       = mm;
        e.max_patches = ctx->max_enc_patches;
        e.n_embd      = n_embd;
        e.out         = b_out;
        RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        int64_t nfull = 0;
        for (int64_t l = 0; l < depth; ++l) nfull += full[(size_t)l];
        rad_note(b, "vision tower: %lld blocks of %lld (%lld heads x %lld), ff %lld, patch %lld x "
                    "%lld x %lld, merge %lld, windows of %lld patches (%lld full attention); "
                    "%lld patches a pass -> %lld rows of %lld",
                 (long long)depth, (long long)width, (long long)heads, (long long)head_dim,
                 (long long)ff, (long long)temporal, (long long)patch, (long long)patch,
                 (long long)merge, (long long)side, (long long)nfull, (long long)P, (long long)R,
                 (long long)n_embd);
    }
    on = true;
    return RAD_OK;
}

/* Stage taps, the same protocol as rad_block_sam.h's: RADIANCE_DEBUG_MM_BUFFERS names stages
 * (comma separated, matched as substrings) and RADIANCE_DEBUG_MM_DUMP's directory receives
 * tap.<tag>.bf16 -- raw bf16 rows, one file a stage. The copy is issued on the pass's stream
 * behind the producing op and synced, so it reads that op's output rather than whatever writes
 * the buffer later (the after-pass dump in core/engine_mm.cpp cannot see intermediates). */
inline void muse_tap(RadCtx* c, const char* tag, rad_buf b, int64_t rows, int64_t cols) {
    static const char* const dir = std::getenv("RADIANCE_DEBUG_MM_DUMP");
    static const char* const env = std::getenv("RADIANCE_DEBUG_MM_BUFFERS");
    if (!dir || !*dir || !env || !*env || rows <= 0 || cols <= 0 || rad_rank(c) != 0) return;
    bool want = false;
    const size_t tl = std::strlen(tag);
    for (const char* s = env; *s && !want;) {
        const char* e = s;
        while (*e && *e != ',') ++e;
        const size_t l = (size_t)(e - s);
        for (size_t i = 0; l && i + l <= tl && !want; ++i)
            want = std::strncmp(tag + i, s, l) == 0;
        s = *e ? e + 1 : e;
    }
    if (!want) return;
    void* p = rad_buf_ptr(c, b);
    if (!p) return;
    std::vector<uint16_t> h((size_t)(rows * cols));
    if (rad_memcpy_async(h.data(), p, (size_t)(rows * cols * 2), rad_stream(c)) < 0) return;
    rad_stream_sync(rad_stream(c));
    char path[512];
    std::snprintf(path, sizeof path, "%s/tap.%s.bf16", dir, tag);
    if (FILE* f = std::fopen(path, "wb")) {
        std::fwrite(h.data(), sizeof(uint16_t), h.size(), f);
        std::fclose(f);
    }
}

inline void MuseTower::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg;
    const int64_t mm = merge * merge, Rn = Pn / mm;
    if (!on || Pn <= 0 || Pn > P || Pn % mm || S <= 0 || !batch->enc_coord || !batch->enc_cu ||
        !batch->enc_seg_h || !batch->enc_seg_w || !batch->enc_pixels) {
        rad_step_fail(c, "an encoder pass this program's vision tower cannot serve: no tower, or "
                         "more patches than it was declared for, or not whole merged rows");
        return;
    }
    /* Windows per frame, from the pass's host grids: the window layers' cu is the per-window
     * boundaries among permuted rows, `n_win + 1` entries of the order op's output. */
    int64_t n_win = 0;
    for (int64_t s = 0; s < S; ++s)
        n_win += (batch->enc_seg_h[s] + side - 1) / side * ((batch->enc_seg_w[s] + side - 1) / side);
    if (n_win > P) {
        rad_step_fail(c, "an encoder pass whose window count exceeds the tower's declared rows");
        return;
    }
    /* Patches, coordinates (planes 0-3: row, column, segment height, segment width, in patches)
     * and the frame boundaries, staged by the core. The full layers read the frame boundaries
     * unchanged: the window permutation is within frames, so the boundaries among permuted rows
     * are these values (the reference passes its cu_seqlens the same way). */
    const RadOperand pix = praw2(batch->enc_pixels, RAD_BF16, Pn, patch_dim);
    const RadOperand crd = praw2(batch->enc_coord, RAD_I32, 4, Pn);
    const RadOperand cu  = praw(batch->enc_cu, RAD_I32, S + 1);
    RadOperand cuw = RAD_B(b_cuw);
    cuw.rows = n_win + 1;
    RadOperand pos = RAD_B(b_pos);
    pos.rows = 2;
    pos.cols = Pn;

    RAD_ISSUE_N(c, op_patch, Pn, pix, RAD_W(w_patch), brows(b_x, Pn));
    RAD_ISSUE_N(c, op_pos, Pn, brows(b_x, Pn), RAD_W(w_pos), crd);
    muse_tap(c, "v.patchpos", b_x, Pn, width);
    RAD_ISSUE_N(c, op_lpre, Pn, brows(b_x, Pn), RAD_W(lw_pre), RAD_W(lb_pre), brows(b_h, Pn));
    muse_tap(c, "v.lnpre", b_h, Pn, width);
    RAD_ISSUE_N(c, op_wo, Pn, crd, cu, brows(b_idx, Pn), brows(b_inv, Pn), pos, cuw);
    RAD_ISSUE_N(c, op_gidx, Pn, brows(b_h, Pn), brows(b_idx, Pn), brows(b_p, Pn));
    for (size_t li = 0; li < L.size(); ++li) {
        const Layer& l = L[li];
        RAD_ISSUE_N(c, l.ln1, Pn, brows(b_p, Pn), RAD_W(l.n1w), RAD_W(l.n1b), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.qkv, Pn, brows(b_h, Pn), RAD_W(l.qkv_w), RAD_W(l.qkv_b), RAD_NONE,
                    brows(b_qkv, Pn));
        RAD_ISSUE_N(c, l.rope, Pn, brows(b_qkv, Pn), pos);
        RAD_ISSUE_N(c, l.attn, Pn,
                    bcol(b_qkv, 0, width, Pn), bcol(b_qkv, width, width, Pn),
                    bcol(b_qkv, 2 * width, width, Pn),
                    l.full ? cu : cuw, brows(b_att, Pn));
        RAD_ISSUE_N(c, l.proj, Pn, brows(b_att, Pn), RAD_W(l.out_w), RAD_W(l.out_b),
                    brows(b_p, Pn), brows(b_p, Pn));
        RAD_ISSUE_N(c, l.ln2, Pn, brows(b_p, Pn), RAD_W(l.n2w), RAD_W(l.n2b), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.up, Pn, brows(b_h, Pn), RAD_W(l.up_w), RAD_W(l.up_b), RAD_NONE,
                    brows(b_ff, Pn));
        RAD_ISSUE_N(c, l.down, Pn, brows(b_ff, Pn), RAD_W(l.down_w), RAD_W(l.down_b),
                    brows(b_p, Pn), brows(b_p, Pn));
        if (li == 0) muse_tap(c, "v.blk0", b_p, Pn, width);
        if (li + 1 == L.size()) muse_tap(c, "v.blkL", b_p, Pn, width);
    }
    RAD_ISSUE_N(c, op_ginv, Pn, brows(b_p, Pn), brows(b_inv, Pn), brows(b_x, Pn));
    muse_tap(c, "v.ginv", b_x, Pn, width);
    /* ln_post over the P patches, then the reference's pixel_shuffle into merged rows. */
    RAD_ISSUE_N(c, op_lpost, Pn, brows(b_x, Pn), RAD_W(lw_post), RAD_W(lb_post), brows(b_x, Pn));
    muse_tap(c, "v.lnpost", b_x, Pn, width);
    RAD_ISSUE_N(c, op_merge, Pn, brows(b_x, Pn), crd, brows(b_mh, Rn));
    muse_tap(c, "v.merge", b_mh, Rn, mm * width);
    RAD_ISSUE_N(c, op_m1, Rn, brows(b_mh, Rn), RAD_W(w_a1), brows(b_m1, Rn));
    muse_tap(c, "v.fc1", b_m1, Rn, proj_ff);
    RAD_ISSUE_N(c, op_g1, Rn, brows(b_m1, Rn), brows(b_m2, Rn));
    RAD_ISSUE_N(c, op_m2, Rn, brows(b_m2, Rn), RAD_W(w_a2), brows(b_m1, Rn));
    RAD_ISSUE_N(c, op_g2, Rn, brows(b_m1, Rn), brows(b_m2, Rn));
    RAD_ISSUE_N(c, op_proj, Rn, brows(b_m2, Rn), RAD_W(w_proj), brows(b_out, Rn));
    muse_tap(c, "v.proj", b_out, Rn, n_embd);
    RAD_ISSUE_N(c, op_perc, Rn, brows(b_out, Rn), RAD_NONE, brows(b_out, Rn));
    muse_tap(c, "v.out", b_out, Rn, n_embd);
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MUSEGLIMMER_VISION_H */
