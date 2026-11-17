/* rad_block_glm_vision.h -- GLM-5.3's vision tower (transformers' Glm5NextVisionModel), served by
 * glm5next.
 *
 * What it computes, per encoder pass (RadBatch::enc):
 *
 *   x  = patches @ Wp^T + bp                     the Conv3d patch embedding as a linear layer; a
 *                                                patch row is (channel, frame, y, x), no position
 *                                                table
 *   depth times:
 *     h  = RMS(x)                                gain only
 *     q, k, v = qkv(h)                           with bias; q and k RMS-normed per head with a
 *                                                gain, then a 2-D rotary over (row, column):
 *                                                `axial` over NeoX pairs (i, i + head_dim / 2),
 *                                                the row on the first head_dim / 4 frequencies and
 *                                                the column on the next, each on its own ladder
 *     x += proj(attn(q, k, v))                   full attention within a segment (one picture)
 *     x += down(silu(min(g, L)) * clamp(u, -L, L)), g | u = gate_up(RMS(x))   all with bias
 *   y  = RMS(x)                                  post_layernorm
 *   d  = conv2d(y viewed [rows / 2, 2, 2, width], 2 x 2, stride 2) + bd   the downsample: one row
 *                                                a merged block of merge x merge patches
 *   out = down'(silu(min(g', L)) * clamp(u', -L, L)), g' | u' = gate_up'(gelu(LN(proj'(d))))
 *                                                the merger: no biases but the LayerNorm's, exact
 *                                                GELU, the clip of the blocks
 *
 * which is the reference module for module. The processor stages patches in merge-block order (the
 * merge x merge patches of one output row consecutive), so the downsample's input view is the
 * reference's own: `conv2d_unfold` writes each merged block's receptive field as one row, its
 * columns in the checkpoint weight's order (channel, kernel row, kernel column), and the weight is
 * read as stored. Every linear with a bias is `gemm_nt_bias`, its residual in the epilogue.
 *
 * Every rank runs the whole tower with replicated weights, as rad_block_vit.h's does, and the
 * weights are RAD_ACCESS_PER_REQUEST. Activations are sized by the pass (RadBuildCtx::
 * max_enc_patches rows), except in a sizing declare (shape_probe), where rows follow the probe's
 * step and nothing is declared to the core. */
#ifndef RAD_BLOCK_GLM_VISION_H
#define RAD_BLOCK_GLM_VISION_H

#include "rad_arch.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace rad {
namespace arch {

struct GlmVision {
    /* Geometry, from the container's vision_config. */
    int64_t depth = 0, width = 0, heads = 0, head_dim = 0, ff = 0, patch = 0, temporal = 0;
    int64_t chans = 0, merge = 0, n_embd = 0, proj_ff = 0, patch_dim = 0, P = 0;
    double  eps = 1e-5, theta = 10000.0, limit = 0.0;
    bool    on = false;

    struct Layer {
        rad_weight n1 = 0, qkv_w = 0, qkv_b = 0, q_norm = 0, k_norm = 0, out_w = 0, out_b = 0;
        rad_weight n2 = 0, gu_w = 0, gu_b = 0, down_w = 0, down_b = 0;
        rad_op op_n1 = 0, op_qkv = 0, op_qk = 0, op_attn = 0, op_proj = 0, op_n2 = 0, op_gu = 0;
        rad_op op_act = 0, op_down = 0;
    };
    std::vector<Layer> L;
    rad_weight w_patch = 0, b_patch = 0, w_post = 0, w_ds = 0, b_ds = 0;
    rad_weight w_mproj = 0, w_mln = 0, b_mln = 0, w_mgu = 0, w_mdown = 0;
    rad_op op_patch = 0, op_post = 0, op_unfold = 0, op_ds = 0, op_mproj = 0, op_mln = 0;
    rad_op op_gelu = 0, op_mgu = 0, op_mact = 0, op_mdown = 0;
    rad_buf b_x = 0, b_h = 0, b_qkv = 0, b_att = 0, b_gu = 0, b_ff = 0, b_post = 0, b_unf = 0;
    rad_buf b_d = 0, b_m = 0, b_mgu = 0, b_mff = 0, b_out = 0;
    char    sec[48] = {};

    /* Whether the container carries GLM-5.3's tower. */
    static bool present(const RadModelMeta* m) {
        const char* t = rad_meta_gets(m, "vision_config.model_type", "");
        return t && !std::strcmp(t, "glm5_next_vision");
    }

    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, int first_group);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int GlmVision::declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx,
                              Names& nm, int64_t model_embd, int first_group) {
    on = false;
    if (!present(m) || ctx->max_enc_patches <= 0) return RAD_OK;

    depth    = rad_meta_geti(m, "vision_config.depth", 0);
    width    = rad_meta_geti(m, "vision_config.hidden_size", 0);
    heads    = rad_meta_geti(m, "vision_config.num_heads", 0);
    ff       = rad_meta_geti(m, "vision_config.intermediate_size", 0);
    patch    = rad_meta_geti(m, "vision_config.patch_size", 0);
    temporal = rad_meta_geti(m, "vision_config.temporal_patch_size", 0);
    chans    = rad_meta_geti(m, "vision_config.in_channels", 3);
    merge    = rad_meta_geti(m, "vision_config.spatial_merge_size", 0);
    n_embd   = rad_meta_geti(m, "vision_config.out_hidden_size", 0);
    proj_ff  = rad_meta_geti(m, "vision_config.projection_intermediate_size", 0);
    eps      = rad_meta_getf(m, "vision_config.rms_norm_eps", 1e-5);
    limit    = rad_meta_getf(m, "vision_config.swiglu_limit", 0.0);
    /* The checkpoint states no rotary; the reference's default is axial at 10000. */
    theta    = rad_meta_getf(m, "vision_config.rope_parameters.rope_theta", 10000.0);
    const char* rt  = rad_meta_gets(m, "vision_config.rope_parameters.rope_type", "axial");
    const char* act = rad_meta_gets(m, "vision_config.hidden_act", "");
    const bool  bias = rad_meta_geti(m, "vision_config.attention_bias", 1) != 0;
    if (depth <= 0 || width <= 0 || heads <= 0 || ff <= 0 || patch <= 0 || temporal <= 0 ||
        chans <= 0 || merge <= 0 || n_embd <= 0 || proj_ff <= 0 || width % heads ||
        (width / heads) % 4 || !(eps > 0) || !(theta > 0)) {
        fprintf(stderr, "radiance: glm5next: the container's vision_config is incomplete or "
                        "inconsistent (depth %lld, hidden %lld, heads %lld, ff %lld, patch %lld, "
                        "merge %lld, out %lld, projection %lld)\n", (long long)depth,
                (long long)width, (long long)heads, (long long)ff, (long long)patch,
                (long long)merge, (long long)n_embd, (long long)proj_ff);
        return RAD_E_INVAL;
    }
    /* The clipped SwiGLU, the axial rotary and the biased linears are what this tower is; a
     * checkpoint that turns one of them off is another tower. */
    if (std::strcmp(act, "silu") || !(limit > 0) || !std::isfinite(limit) ||
        std::strcmp(rt, "axial") || !bias) {
        fprintf(stderr, "radiance: glm5next: this vision tower has a '%s' activation, swiglu_limit "
                        "%g, a '%s' rotary or no attention bias; it serves silu, a positive "
                        "limit, axial and biased linears\n", act, limit, rt);
        return RAD_E_UNSUPPORTED;
    }
    /* The merger's rows replace token embeddings, so they must be the model's width. */
    if (n_embd != model_embd) {
        fprintf(stderr, "radiance: glm5next: the vision tower emits %lld-wide rows and the model "
                        "embeds %lld-wide tokens\n", (long long)n_embd, (long long)model_embd);
        return RAD_E_INVAL;
    }
    head_dim  = width / heads;
    patch_dim = chans * temporal * patch * patch;
    std::snprintf(sec, sizeof sec, "%lld %lld", (long long)(head_dim / 4),
                  (long long)(head_dim / 4));

    const int64_t mm = merge * merge;
    P = ctx->max_enc_patches;
    /* The longest segment attention may see: the real pass size, even in a sizing declare (see
     * rad_block_vit.h). */
    const int64_t max_seg = P;
    if (ctx->shape_probe) {
        const int64_t r = std::min<int64_t>(P, ctx->max_tok);
        P = std::max<int64_t>(mm, r / mm * mm);
    }
    if (P % mm) return RAD_E_SHAPE;
    const int64_t R = P / mm;
    const int ACC = RAD_ACCESS_PER_REQUEST;

    /* ---- weights, replicated: model-level groups for the stem, downsample and merger, one group
     * a block from `first_group` on. gate and up are joined into one GEMM, gate first, as
     * swiglu_clamped splits its operand. */
    auto copy = [&](const char* dn, const char* src, std::initializer_list<int64_t> shp,
                    RadWeightGroup g) -> rad_weight {
        if (map_copy(b, dn, src) < 0) return 0;
        return decl_w(b, dn, RAD_BF16, shp, ACC, RAD_SHARD_NONE, g);
    };
    auto join = [&](const char* dn, const char* gate, const char* up,
                    std::initializer_list<int64_t> shp, RadWeightGroup g) -> rad_weight {
        if (map_concat(b, dn, { gate, up }) < 0) return 0;
        return decl_w(b, dn, RAD_BF16, shp, ACC, RAD_SHARD_NONE, g);
    };
    const char* V = "model.visual";
    w_patch = copy(nm.f("v.patch.weight"), nm.ckpt("%s.patch_embed.proj.weight", V),
                   { width, patch_dim }, grp_model());
    b_patch = copy(nm.f("v.patch.bias"), nm.ckpt("%s.patch_embed.proj.bias", V), { width },
                   grp_model());
    L.assign((size_t)depth, Layer{});
    for (int64_t i = 0; i < depth; ++i) {
        Layer& l = L[(size_t)i];
        const RadWeightGroup g = grp_layer(first_group + (int)i);
        const int li = (int)i;
        auto src = [&](const char* s) { return nm.ckpt("%s.blocks.%d.%s", V, li, s); };
        auto dn  = [&](const char* s) { return nm.f("v.blk.%d.%s", li, s); };
        l.n1     = copy(dn("norm1.weight"), src("norm1.weight"), { width }, g);
        l.qkv_w  = copy(dn("qkv.weight"), src("attn.qkv.weight"), { 3 * width, width }, g);
        l.qkv_b  = copy(dn("qkv.bias"), src("attn.qkv.bias"), { 3 * width }, g);
        l.q_norm = copy(dn("q_norm.weight"), src("attn.q_norm.weight"), { head_dim }, g);
        l.k_norm = copy(dn("k_norm.weight"), src("attn.k_norm.weight"), { head_dim }, g);
        l.out_w  = copy(dn("proj.weight"), src("attn.proj.weight"), { width, width }, g);
        l.out_b  = copy(dn("proj.bias"), src("attn.proj.bias"), { width }, g);
        l.n2     = copy(dn("norm2.weight"), src("norm2.weight"), { width }, g);
        l.gu_w   = join(dn("gate_up.weight"), src("mlp.gate_proj.weight"), src("mlp.up_proj.weight"),
                        { 2 * ff, width }, g);
        l.gu_b   = join(dn("gate_up.bias"), src("mlp.gate_proj.bias"), src("mlp.up_proj.bias"),
                        { 2 * ff }, g);
        l.down_w = copy(dn("down.weight"), src("mlp.down_proj.weight"), { width, ff }, g);
        l.down_b = copy(dn("down.bias"), src("mlp.down_proj.bias"), { width }, g);
        if (!l.n1 || !l.qkv_w || !l.qkv_b || !l.q_norm || !l.k_norm || !l.out_w || !l.out_b ||
            !l.n2 || !l.gu_w || !l.gu_b || !l.down_w || !l.down_b)
            return RAD_E_INVAL;
    }
    w_post  = copy(nm.f("v.post_norm.weight"), nm.ckpt("%s.post_layernorm.weight", V), { width },
                   grp_model());
    /* [out, width, merge, merge] read as [out, width * merge^2]: conv2d_unfold's column order. */
    w_ds    = copy(nm.f("v.downsample.weight"), nm.ckpt("%s.downsample.weight", V),
                   { n_embd, mm * width }, grp_model());
    b_ds    = copy(nm.f("v.downsample.bias"), nm.ckpt("%s.downsample.bias", V), { n_embd },
                   grp_model());
    w_mproj = copy(nm.f("v.merger.proj.weight"), nm.ckpt("%s.merger.proj.weight", V),
                   { n_embd, n_embd }, grp_model());
    w_mln   = copy(nm.f("v.merger.norm.weight"),
                   nm.ckpt("%s.merger.post_projection_norm.weight", V), { n_embd }, grp_model());
    b_mln   = copy(nm.f("v.merger.norm.bias"), nm.ckpt("%s.merger.post_projection_norm.bias", V),
                   { n_embd }, grp_model());
    w_mgu   = join(nm.f("v.merger.gate_up.weight"), nm.ckpt("%s.merger.gate_proj.weight", V),
                   nm.ckpt("%s.merger.up_proj.weight", V), { 2 * proj_ff, n_embd }, grp_model());
    w_mdown = copy(nm.f("v.merger.down.weight"), nm.ckpt("%s.merger.down_proj.weight", V),
                   { n_embd, proj_ff }, grp_model());
    if (!w_patch || !b_patch || !w_post || !w_ds || !b_ds || !w_mproj || !w_mln || !b_mln ||
        !w_mgu || !w_mdown)
        return RAD_E_INVAL;

    /* ---- activations. `post` is declared at merge x width: the norm writes P rows of `width`
     * and conv2d_unfold reads the same bytes as P / merge rows of `merge` columns (dh rows of dw
     * columns), the reference's view() with no copy. */
    b_x    = decl_b(b, nm.f("v.x"),    RAD_BF16, { P, width });
    b_h    = decl_b(b, nm.f("v.h"),    RAD_BF16, { P, width });
    b_qkv  = decl_b(b, nm.f("v.qkv"),  RAD_BF16, { P, 3 * width });
    b_att  = decl_b(b, nm.f("v.att"),  RAD_BF16, { P, width });
    b_gu   = decl_b(b, nm.f("v.gu"),   RAD_BF16, { P, 2 * ff });
    b_ff   = decl_b(b, nm.f("v.ff"),   RAD_BF16, { P, ff });
    b_post = decl_b(b, nm.f("v.post"), RAD_BF16, { P / merge, merge * width });
    b_unf  = decl_b(b, nm.f("v.unf"),  RAD_BF16, { R, mm * width });
    b_d    = decl_b(b, nm.f("v.d"),    RAD_BF16, { R, n_embd });
    b_m    = decl_b(b, nm.f("v.m"),    RAD_BF16, { R, n_embd });
    b_mgu  = decl_b(b, nm.f("v.mgu"),  RAD_BF16, { R, 2 * proj_ff });
    b_mff  = decl_b(b, nm.f("v.mff"),  RAD_BF16, { R, proj_ff });
    b_out  = decl_b(b, nm.f("v.out"),  RAD_BF16, { R, n_embd });
    if (!b_x || !b_h || !b_qkv || !b_att || !b_gu || !b_ff || !b_post || !b_unf || !b_d || !b_m ||
        !b_mgu || !b_mff || !b_out)
        return RAD_E_INVAL;

    auto gemm_b = [&](int64_t rows, int64_t N, int64_t K, rad_weight w, rad_weight bw) -> rad_op {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16"), RAD_STR("act", "none")),
                      RAD_WEIGHTS(w, bw));
    };
    auto gemm = [&](int64_t rows, int64_t N, int64_t K, rad_weight w) -> rad_op {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto rms = [&](int64_t rows, rad_weight w) -> rad_op {
        return RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                 RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto swiglu = [&](int64_t rows, int64_t n) -> rad_op {
        return RAD_OP(b, "swiglu_clamped",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                 RAD_F64("limit", limit), RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    };

    op_patch = rw(b, gemm_b(P, width, patch_dim, w_patch, b_patch), {}, { b_x });
    for (Layer& l : L) {
        l.op_n1   = rw(b, rms(P, l.n1), { b_x }, { b_h });
        l.op_qkv  = rw(b, gemm_b(P, 3 * width, width, l.qkv_w, l.qkv_b), { b_h }, { b_qkv });
        l.op_qk   = rw(b, RAD_OP(b, "qk_norm_rope",
                                 RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", head_dim),
                                            RAD_INT("n_head", heads), RAD_INT("n_head_kv", heads),
                                            RAD_F64("theta", theta), RAD_F64("eps", eps),
                                            RAD_STR("mode", "axial"),
                                            RAD_INT("rotary_dim", head_dim),
                                            RAD_STR("sections", sec)),
                                 RAD_WEIGHTS(l.q_norm, l.k_norm)),
                       { b_qkv }, { b_qkv });
        l.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", head_dim),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)head_dim)),
                                            RAD_INT("max_seqlen", max_seg)),
                                 RAD_NOWEIGHTS),
                       { b_qkv }, { b_att });
        l.op_proj = rw(b, gemm_b(P, width, width, l.out_w, l.out_b), { b_att, b_x }, { b_x });
        l.op_n2   = rw(b, rms(P, l.n2), { b_x }, { b_h });
        l.op_gu   = rw(b, gemm_b(P, 2 * ff, width, l.gu_w, l.gu_b), { b_h }, { b_gu });
        l.op_act  = rw(b, swiglu(P, ff), { b_gu }, { b_ff });
        l.op_down = rw(b, gemm_b(P, width, ff, l.down_w, l.down_b), { b_ff, b_x }, { b_x });
    }
    op_post   = rw(b, rms(P, w_post), { b_x }, { b_post });
    /* Input rows are P / merge (one kernel row of merge patches each) and a segment's rows are its
     * patches over merge, so a picture of whole merged blocks starts on an even row. */
    op_unfold = rw(b, RAD_OP(b, "conv2d_unfold",
                             RAD_PARAMS(RAD_RANGE("M", 1, P / merge), RAD_INT("width", merge),
                                        RAD_INT("cin", width), RAD_INT("kh", merge),
                                        RAD_INT("kw", merge), RAD_INT("stride", merge),
                                        RAD_INT("pad", 0), RAD_INT("cu_div", merge),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { b_post }, { b_unf });
    op_ds    = rw(b, gemm_b(R, n_embd, mm * width, w_ds, b_ds), { b_unf }, { b_d });
    op_mproj = rw(b, gemm(R, n_embd, n_embd, w_mproj), { b_d }, { b_m });
    /* torch.nn.LayerNorm's default eps: the merger states none. */
    op_mln   = rw(b, RAD_OP(b, "layernorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", n_embd),
                                       RAD_F64("eps", 1e-5), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w_mln, b_mln)),
                  { b_m }, { b_d });
    op_gelu  = rw(b, RAD_OP(b, "gelu",
                            RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", n_embd),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { b_d }, { b_m });
    op_mgu   = rw(b, gemm(R, 2 * proj_ff, n_embd, w_mgu), { b_m }, { b_mgu });
    op_mact  = rw(b, swiglu(R, proj_ff), { b_mgu }, { b_mff });
    op_mdown = rw(b, gemm(R, n_embd, proj_ff, w_mdown), { b_mff }, { b_out });

    if (!ctx->shape_probe) {
        RadEncoderDecl e{};
        e.name        = "vision";
        e.modalities  = RAD_MM_IMAGE;
        e.patch_dim   = patch_dim;
        e.merge       = mm;
        e.max_patches = P;
        e.n_embd      = n_embd;
        e.out         = b_out;
        RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        rad_note(b, "vision tower: %lld blocks of %lld (%lld heads x %lld), ff %lld clipped at %g, "
                    "patch %lld x %lld x %lld, %lld x %lld downsample, merger %lld; %lld patches a "
                    "pass -> %lld rows of %lld", (long long)depth, (long long)width,
                 (long long)heads, (long long)head_dim, (long long)ff, limit, (long long)temporal,
                 (long long)patch, (long long)patch, (long long)merge, (long long)merge,
                 (long long)proj_ff, (long long)P, (long long)R, (long long)n_embd);
    }
    on = true;
    return RAD_OK;
}

inline void GlmVision::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg;
    const int64_t mm = merge * merge, Rn = Pn / mm;
    if (!on || Pn <= 0 || Pn > P || Pn % mm || S <= 0) {
        rad_step_fail(c, "an encoder pass this program's vision tower cannot serve: no tower, or "
                         "more patches than it was declared for, or not whole merged rows");
        return;
    }
    /* Patches, coordinates (planes 0-1 a dense [2, P] of (row, column)) and segment boundaries,
     * staged by the core. */
    const RadOperand pix = praw2(batch->enc_pixels, RAD_BF16, Pn, patch_dim);
    const RadOperand rc  = praw2(batch->enc_coord, RAD_I32, 2, Pn);
    const RadOperand cu  = praw(batch->enc_cu, RAD_I32, S + 1);

    RAD_ISSUE_N(c, op_patch, Pn, pix, RAD_W(w_patch), RAD_W(b_patch), RAD_NONE, brows(b_x, Pn));
    for (const Layer& l : L) {
        RAD_ISSUE_N(c, l.op_n1, Pn, brows(b_x, Pn), RAD_W(l.n1), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.op_qkv, Pn, brows(b_h, Pn), RAD_W(l.qkv_w), RAD_W(l.qkv_b), RAD_NONE,
                    brows(b_qkv, Pn));
        RAD_ISSUE_N(c, l.op_qk, Pn, brows(b_qkv, Pn), rc, RAD_W(l.q_norm), RAD_W(l.k_norm));
        RAD_ISSUE_N(c, l.op_attn, Pn, bcol(b_qkv, 0, width, Pn), bcol(b_qkv, width, width, Pn),
                    bcol(b_qkv, 2 * width, width, Pn), cu, brows(b_att, Pn));
        RAD_ISSUE_N(c, l.op_proj, Pn, brows(b_att, Pn), RAD_W(l.out_w), RAD_W(l.out_b),
                    brows(b_x, Pn), brows(b_x, Pn));
        RAD_ISSUE_N(c, l.op_n2, Pn, brows(b_x, Pn), RAD_W(l.n2), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.op_gu, Pn, brows(b_h, Pn), RAD_W(l.gu_w), RAD_W(l.gu_b), RAD_NONE,
                    brows(b_gu, Pn));
        RAD_ISSUE_N(c, l.op_act, Pn, brows(b_gu, Pn), brows(b_ff, Pn));
        RAD_ISSUE_N(c, l.op_down, Pn, brows(b_ff, Pn), RAD_W(l.down_w), RAD_W(l.down_b),
                    brows(b_x, Pn), brows(b_x, Pn));
    }
    /* The norm writes Pn rows of `width`; the unfold reads them as Pn / merge rows. */
    RAD_ISSUE_N(c, op_post, Pn, brows(b_x, Pn), RAD_W(w_post), brows(b_post, Pn / merge));
    RAD_ISSUE_N(c, op_unfold, Pn / merge, brows(b_post, Pn / merge), cu, RAD_NONE,
                brows(b_unf, Rn));
    RAD_ISSUE_N(c, op_ds, Rn, brows(b_unf, Rn), RAD_W(w_ds), RAD_W(b_ds), RAD_NONE,
                brows(b_d, Rn));
    RAD_ISSUE_N(c, op_mproj, Rn, brows(b_d, Rn), RAD_W(w_mproj), brows(b_m, Rn));
    RAD_ISSUE_N(c, op_mln, Rn, brows(b_m, Rn), RAD_W(w_mln), RAD_W(b_mln), brows(b_d, Rn));
    RAD_ISSUE_N(c, op_gelu, Rn, brows(b_d, Rn), brows(b_m, Rn));
    RAD_ISSUE_N(c, op_mgu, Rn, brows(b_m, Rn), RAD_W(w_mgu), brows(b_mgu, Rn));
    RAD_ISSUE_N(c, op_mact, Rn, brows(b_mgu, Rn), brows(b_mff, Rn));
    RAD_ISSUE_N(c, op_mdown, Rn, brows(b_mff, Rn), RAD_W(w_mdown), brows(b_out, Rn));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_GLM_VISION_H */
