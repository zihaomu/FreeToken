/* rad_block_vit.h -- the Qwen-VL vision tower (Qwen2.5-VL / Qwen3-VL / Qwen3.5 / Qwen3.8), and the
 * scatter that puts its rows into a language model's embeddings.
 *
 * WHAT IT COMPUTES, per encoder pass (RadBatch::enc):
 *
 *   x  = patches @ Wp^T + bp                     the Conv3d patch embedding, as a linear layer:
 *                                                a patch row is (channel, frame, y, x)
 *   x += pos_embed resampled to each segment's grid   bilinear, corners aligned (grid_embed)
 *   depth times:
 *     x += proj(attn(rope(qkv(LN1(x)))))         full attention within a segment (one image, or
 *                                                one temporal group of a video), 2-D rotary over
 *                                                (row, column) -- `axial` with half the rotated
 *                                                width a side, each on its own ladder
 *     x += fc2(gelu_tanh(fc1(LN2(x))))
 *   out = fc2'(gelu(fc1'(LN(x) viewed [P/merge^2, merge^2 * width])))   the patch merger
 *
 * which is transformers' Qwen3_5VisionModel. Qwen3-VL's deepstack taps (the residual stream after
 * named blocks, each through its own merger whose norm is taken after the 2x2 view, over
 * merge^2 * width, added to the language model's hidden state after its first layers) are declared
 * only for a plugin that passes `deepstack`; otherwise a tower with taps is refused. Every linear
 * is `gemm_nt_bias` with activation and residual in the epilogue, rounded where the reference's
 * separate tensors are.
 *
 * Every rank runs the whole tower with replicated weights: every rank embeds every token, and
 * sharding would cost a collective per layer for once-per-picture compute. Weights are
 * RAD_ACCESS_PER_REQUEST, so placement may keep them off the card when VRAM is short.
 *
 * Activations are sized by the pass (RadBuildCtx::max_enc_patches rows), except in a sizing
 * declare (shape_probe), which uses the probe's step to keep the tower out of the smaller arena
 * levels; an encoder pass always runs at the full plan (Engine::arena_step).
 */
#ifndef RAD_BLOCK_VIT_H
#define RAD_BLOCK_VIT_H

#include "rad_arch.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace rad {
namespace arch {

struct VisionTower {
    /* Geometry, from the container's vision_config. */
    int64_t depth = 0, width = 0, heads = 0, ff = 0, patch = 0, temporal = 0, chans = 0;
    int64_t merge = 0, side = 0, n_embd = 0, head_dim = 0, patch_dim = 0, P = 0;
    float   eps = 1e-6f, theta = 10000.0f;
    bool    on = false;

    struct Layer {
        rad_weight n1w = 0, n1b = 0, n2w = 0, n2b = 0;
        rad_weight qkv_w = 0, qkv_b = 0, out_w = 0, out_b = 0;
        rad_weight up_w = 0, up_b = 0, down_w = 0, down_b = 0;
        rad_op ln1 = 0, qkv = 0, rope = 0, attn = 0, proj = 0, ln2 = 0, up = 0, down = 0;
    };
    std::vector<Layer> L;
    rad_weight w_patch = 0, b_patch = 0, w_pos = 0;
    rad_weight mn_w = 0, mn_b = 0, m1_w = 0, m1_b = 0, m2_w = 0, m2_b = 0;
    rad_op op_patch = 0, op_pos = 0, op_mln = 0, op_m1 = 0, op_m2 = 0;
    rad_buf b_x = 0, b_h = 0, b_qkv = 0, b_att = 0, b_ff = 0, b_mh = 0, b_mf = 0, b_out = 0;
    char    sec[48] = {};

    /* Deepstack: tap t is the residual stream after block `ds_layer[t]`, merged into `b_ds[t]`
     * ([P / merge^2, n_embd], one row per merged token, in `b_out`'s order). */
    enum { MAX_DS = 4 };
    int        n_ds = 0;
    int        ds_layer[MAX_DS] = {};
    rad_weight ds_nw[MAX_DS] = {}, ds_nb[MAX_DS] = {}, ds_1w[MAX_DS] = {}, ds_1b[MAX_DS] = {};
    rad_weight ds_2w[MAX_DS] = {}, ds_2b[MAX_DS] = {};
    rad_op     ds_ln[MAX_DS] = {}, ds_1[MAX_DS] = {}, ds_2[MAX_DS] = {};
    rad_buf    b_ds[MAX_DS] = {};

    /* Whether the container carries a vision tower. */
    static bool present(const RadModelMeta* m) {
        return rad_meta_geti(m, "vision_config.depth", 0) > 0;
    }

    /* With `gate_weights`, a checkpoint or container without the tower's weights declares nothing
     * and leaves `on` false, for a model whose config names a tower its weights may not have. */
    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, int first_group, bool deepstack = false,
                bool gate_weights = false);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int VisionTower::declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx,
                                Names& nm, int64_t model_embd, int first_group, bool deepstack,
                                bool gate_weights) {
    on = false;
    if (!present(m) || ctx->max_enc_patches <= 0) return RAD_OK;

    /* Whether the stem is there: its map first, then its encoding, which convert answers from the
     * checkpoint and load from the container alike. A map left behind is inert. */
    if (gate_weights) {
        if (map_copy(b, nm.f("v.patch.weight"), nm.ckpt("model.visual.patch_embed.proj.weight")) < 0)
            return RAD_E_INVAL;
        RadEncoding enc;
        if (rad_weight_encoding(b, nm.f("v.patch.weight"), &enc, nullptr, nullptr) != RAD_OK) {
            std::fprintf(stderr, "radiance: vision_config names a tower the checkpoint does not "
                                 "ship; declaring without it\n");
            return RAD_OK;
        }
    }

    depth    = rad_meta_geti(m, "vision_config.depth", 0);
    width    = rad_meta_geti(m, "vision_config.hidden_size", 0);
    heads    = rad_meta_geti(m, "vision_config.num_heads", 0);
    ff       = rad_meta_geti(m, "vision_config.intermediate_size", 0);
    patch    = rad_meta_geti(m, "vision_config.patch_size", 0);
    temporal = rad_meta_geti(m, "vision_config.temporal_patch_size", 0);
    chans    = rad_meta_geti(m, "vision_config.in_channels", 3);
    merge    = rad_meta_geti(m, "vision_config.spatial_merge_size", 0);
    n_embd   = rad_meta_geti(m, "vision_config.out_hidden_size", 0);
    const int64_t npos = rad_meta_geti(m, "vision_config.num_position_embeddings", 0);
    theta    = (float)rad_meta_getf(m, "vision_config.rope_parameters.rope_theta",
                                    rad_meta_getf(m, "vision_config.rope_theta", 10000.0));
    side = (int64_t)std::llround(std::sqrt((double)npos));
    if (depth <= 0 || width <= 0 || heads <= 0 || ff <= 0 || patch <= 0 || temporal <= 0 ||
        merge <= 0 || n_embd <= 0 || side <= 0 || side * side != npos || width % heads) {
        fprintf(stderr, "radiance: the container's vision_config is incomplete or inconsistent "
                        "(depth %lld, hidden %lld, heads %lld, ff %lld, patch %lld, merge %lld, "
                        "positions %lld)\n", (long long)depth, (long long)width, (long long)heads,
                (long long)ff, (long long)patch, (long long)merge, (long long)npos);
        return RAD_E_INVAL;
    }
    /* The merger's rows replace token embeddings, so they must be the model's width. */
    if (n_embd != model_embd) {
        fprintf(stderr, "radiance: the vision tower emits %lld-wide rows and the model embeds "
                        "%lld-wide tokens\n", (long long)n_embd, (long long)model_embd);
        return RAD_E_INVAL;
    }
    /* Deepstack feeds tower layers into the language model's early layers; a plugin that does not
     * consume it is refused the tower. */
    const char* ds = rad_meta_gets(m, "vision_config.deepstack_visual_indexes", "");
    n_ds = 0;
    for (const char* q = ds; q && *q;) {
        if (*q < '0' || *q > '9') { ++q; continue; }
        char* e = nullptr;
        const long v = std::strtol(q, &e, 10);
        q = e;
        if (!deepstack) {
            fprintf(stderr, "radiance: this vision tower has deepstack taps (%s), which this plugin "
                            "does not serve\n", ds);
            return RAD_E_UNSUPPORTED;
        }
        if (n_ds >= MAX_DS || v < 0 || v >= depth) {
            fprintf(stderr, "radiance: deepstack taps '%s' do not fit a %lld-block tower\n", ds,
                    (long long)depth);
            return RAD_E_INVAL;
        }
        ds_layer[n_ds++] = (int)v;
    }
    const char* act = rad_meta_gets(m, "vision_config.hidden_act", "gelu_pytorch_tanh");
    if (std::strcmp(act, "gelu_pytorch_tanh") && std::strcmp(act, "gelu")) {
        fprintf(stderr, "radiance: vision_config.hidden_act '%s' is not served\n", act);
        return RAD_E_UNSUPPORTED;
    }
    const char* mlp_act = std::strcmp(act, "gelu") ? "gelu_tanh" : "gelu";
    head_dim  = width / heads;
    patch_dim = chans * temporal * patch * patch;
    /* The 2-D rotary: half the head a side, each on its own ladder. */
    if (head_dim % 4) return RAD_E_SHAPE;
    std::snprintf(sec, sizeof sec, "%lld %lld", (long long)(head_dim / 4), (long long)(head_dim / 4));

    const int64_t mm = merge * merge;
    P = ctx->max_enc_patches;
    /* The longest segment attention may see: the real pass size, even in a sizing declare. It
     * selects the kernel, and the engine runs the real declare's kernel at every level; only the
     * ranged row counts may follow the probe's step. */
    const int64_t max_seg = P;
    if (ctx->shape_probe) {
        /* A sizing declare: rows follow the probe's step (rounded to whole merged rows), keeping
         * the activations out of the levels a decode step runs at. */
        const int64_t r = std::min<int64_t>(P, ctx->max_tok);
        P = std::max<int64_t>(mm, r / mm * mm);
    }
    if (P % mm) return RAD_E_SHAPE;
    const int64_t R = P / mm;
    const int ACC = RAD_ACCESS_PER_REQUEST;

    /* ---- weights, replicated. Model-level groups for the stem and merger; one group per block
     * from `first_group` on (past every language layer), so a tower block is its own movement
     * unit. */
    auto bw = [&](const char* dn, const char* src, uint32_t dt, std::initializer_list<int64_t> shp,
                  RadWeightGroup g) -> rad_weight {
        /* map first: the weight's encoding is looked up through it */
        if (map_copy(b, dn, src) < 0) return 0;
        return decl_w(b, dn, dt, shp, ACC, RAD_SHARD_NONE, g);
    };
    /* Mapped above with `gate_weights`. */
    w_patch = gate_weights
        ? decl_w(b, nm.f("v.patch.weight"), RAD_BF16, { width, patch_dim },
                 RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, grp_model())
        : bw(nm.f("v.patch.weight"), nm.ckpt("model.visual.patch_embed.proj.weight"),
             RAD_BF16, { width, patch_dim }, grp_model());
    b_patch = bw(nm.f("v.patch.bias"), nm.ckpt("model.visual.patch_embed.proj.bias"),
                 RAD_BF16, { width }, grp_model());
    w_pos   = bw(nm.f("v.pos.weight"), nm.ckpt("model.visual.pos_embed.weight"),
                 RAD_BF16, { npos, width }, grp_model());
    L.assign((size_t)depth, Layer{});
    for (int64_t i = 0; i < depth; ++i) {
        Layer& l = L[(size_t)i];
        const RadWeightGroup g = grp_layer(first_group + (int)i);
        const int li = (int)i;
        l.n1w    = bw(nm.f("v.blk.%d.norm1.weight", li), nm.ckpt("model.visual.blocks.%d.norm1.weight", li), RAD_BF16, { width }, g);
        l.n1b    = bw(nm.f("v.blk.%d.norm1.bias", li),   nm.ckpt("model.visual.blocks.%d.norm1.bias", li),   RAD_BF16, { width }, g);
        l.qkv_w  = bw(nm.f("v.blk.%d.qkv.weight", li),   nm.ckpt("model.visual.blocks.%d.attn.qkv.weight", li), RAD_BF16, { 3 * width, width }, g);
        l.qkv_b  = bw(nm.f("v.blk.%d.qkv.bias", li),     nm.ckpt("model.visual.blocks.%d.attn.qkv.bias", li),   RAD_BF16, { 3 * width }, g);
        l.out_w  = bw(nm.f("v.blk.%d.proj.weight", li),  nm.ckpt("model.visual.blocks.%d.attn.proj.weight", li), RAD_BF16, { width, width }, g);
        l.out_b  = bw(nm.f("v.blk.%d.proj.bias", li),    nm.ckpt("model.visual.blocks.%d.attn.proj.bias", li),   RAD_BF16, { width }, g);
        l.n2w    = bw(nm.f("v.blk.%d.norm2.weight", li), nm.ckpt("model.visual.blocks.%d.norm2.weight", li), RAD_BF16, { width }, g);
        l.n2b    = bw(nm.f("v.blk.%d.norm2.bias", li),   nm.ckpt("model.visual.blocks.%d.norm2.bias", li),   RAD_BF16, { width }, g);
        l.up_w   = bw(nm.f("v.blk.%d.fc1.weight", li),   nm.ckpt("model.visual.blocks.%d.mlp.linear_fc1.weight", li), RAD_BF16, { ff, width }, g);
        l.up_b   = bw(nm.f("v.blk.%d.fc1.bias", li),     nm.ckpt("model.visual.blocks.%d.mlp.linear_fc1.bias", li),   RAD_BF16, { ff }, g);
        l.down_w = bw(nm.f("v.blk.%d.fc2.weight", li),   nm.ckpt("model.visual.blocks.%d.mlp.linear_fc2.weight", li), RAD_BF16, { width, ff }, g);
        l.down_b = bw(nm.f("v.blk.%d.fc2.bias", li),     nm.ckpt("model.visual.blocks.%d.mlp.linear_fc2.bias", li),   RAD_BF16, { width }, g);
    }
    mn_w = bw(nm.f("v.merger.norm.weight"), nm.ckpt("model.visual.merger.norm.weight"), RAD_BF16, { width }, grp_model());
    mn_b = bw(nm.f("v.merger.norm.bias"),   nm.ckpt("model.visual.merger.norm.bias"),   RAD_BF16, { width }, grp_model());
    m1_w = bw(nm.f("v.merger.fc1.weight"),  nm.ckpt("model.visual.merger.linear_fc1.weight"), RAD_BF16, { mm * width, mm * width }, grp_model());
    m1_b = bw(nm.f("v.merger.fc1.bias"),    nm.ckpt("model.visual.merger.linear_fc1.bias"),   RAD_BF16, { mm * width }, grp_model());
    m2_w = bw(nm.f("v.merger.fc2.weight"),  nm.ckpt("model.visual.merger.linear_fc2.weight"), RAD_BF16, { n_embd, mm * width }, grp_model());
    m2_b = bw(nm.f("v.merger.fc2.bias"),    nm.ckpt("model.visual.merger.linear_fc2.bias"),   RAD_BF16, { n_embd }, grp_model());
    if (!w_patch || !b_patch || !w_pos || !mn_w || !mn_b || !m1_w || !m1_b || !m2_w || !m2_b)
        return RAD_E_INVAL;
    for (int t = 0; t < n_ds; ++t) {
        const char* pre = "model.visual.deepstack_merger_list";
        ds_nw[t] = bw(nm.f("v.ds.%d.norm.weight", t), nm.ckpt("%s.%d.norm.weight", pre, t), RAD_BF16, { mm * width }, grp_model());
        ds_nb[t] = bw(nm.f("v.ds.%d.norm.bias", t),   nm.ckpt("%s.%d.norm.bias", pre, t),   RAD_BF16, { mm * width }, grp_model());
        ds_1w[t] = bw(nm.f("v.ds.%d.fc1.weight", t),  nm.ckpt("%s.%d.linear_fc1.weight", pre, t), RAD_BF16, { mm * width, mm * width }, grp_model());
        ds_1b[t] = bw(nm.f("v.ds.%d.fc1.bias", t),    nm.ckpt("%s.%d.linear_fc1.bias", pre, t),   RAD_BF16, { mm * width }, grp_model());
        ds_2w[t] = bw(nm.f("v.ds.%d.fc2.weight", t),  nm.ckpt("%s.%d.linear_fc2.weight", pre, t), RAD_BF16, { n_embd, mm * width }, grp_model());
        ds_2b[t] = bw(nm.f("v.ds.%d.fc2.bias", t),    nm.ckpt("%s.%d.linear_fc2.bias", pre, t),   RAD_BF16, { n_embd }, grp_model());
        if (!ds_nw[t] || !ds_nb[t] || !ds_1w[t] || !ds_1b[t] || !ds_2w[t] || !ds_2b[t])
            return RAD_E_INVAL;
    }

    /* ---- activations. `mh`, the merger's input, is declared at the merged width: the norm writes
     * P rows of `width` and the first merger layer reads the same bytes as P/mm rows of
     * mm * width (the reference's view(), no copy). */
    b_x   = decl_b(b, nm.f("v.x"),   RAD_BF16, { P, width });
    b_h   = decl_b(b, nm.f("v.h"),   RAD_BF16, { P, width });
    b_qkv = decl_b(b, nm.f("v.qkv"), RAD_BF16, { P, 3 * width });
    b_att = decl_b(b, nm.f("v.att"), RAD_BF16, { P, width });
    b_ff  = decl_b(b, nm.f("v.ff"),  RAD_BF16, { P, ff });
    b_mh  = decl_b(b, nm.f("v.mh"),  RAD_BF16, { R, mm * width });
    b_mf  = decl_b(b, nm.f("v.mf"),  RAD_BF16, { R, mm * width });
    b_out = decl_b(b, nm.f("v.out"), RAD_BF16, { R, n_embd });
    if (!b_x || !b_h || !b_qkv || !b_att || !b_ff || !b_mh || !b_mf || !b_out) return RAD_E_INVAL;

    auto gemm = [&](int64_t rows, int64_t N, int64_t K, const char* a, rad_weight w,
                    rad_weight bias) -> rad_op {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16"), RAD_STR("act", a)),
                      RAD_WEIGHTS(w, bias));
    };
    auto lnorm = [&](int64_t rows, rad_weight w, rad_weight bias) -> rad_op {
        return RAD_OP(b, "layernorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                 RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w, bias));
    };

    op_patch = rw(b, gemm(P, width, patch_dim, "none", w_patch, b_patch), {}, { b_x });
    op_pos = rw(b, RAD_OP(b, "grid_embed",
                          RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                     RAD_INT("side", side), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w_pos)),
                { b_x }, { b_x });
    for (Layer& l : L) {
        l.ln1  = rw(b, lnorm(P, l.n1w, l.n1b), { b_x }, { b_h });
        l.qkv  = rw(b, gemm(P, 3 * width, width, "none", l.qkv_w, l.qkv_b), { b_h }, { b_qkv });
        l.rope = rw(b, RAD_OP(b, "rope",
                              RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", head_dim),
                                         RAD_INT("n_head", heads), RAD_INT("n_head_kv", heads),
                                         RAD_F64("theta", theta), RAD_F64("scale", 1.0),
                                         RAD_STR("mode", "axial"),
                                         RAD_INT("rotary_dim", head_dim),
                                         RAD_STR("sections", sec)),
                              RAD_NOWEIGHTS),
                    { b_qkv }, { b_qkv });
        l.attn = rw(b, RAD_OP(b, "attn_dense",
                              RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", head_dim),
                                         RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                         RAD_STR("dtype", "bf16"),
                                         RAD_F64("scale", 1.0 / std::sqrt((double)head_dim)),
                                         RAD_INT("max_seqlen", max_seg)),
                              RAD_NOWEIGHTS),
                    { b_qkv }, { b_att });
        l.proj = rw(b, gemm(P, width, width, "none", l.out_w, l.out_b), { b_att, b_x }, { b_x });
        l.ln2  = rw(b, lnorm(P, l.n2w, l.n2b), { b_x }, { b_h });
        l.up   = rw(b, gemm(P, ff, width, mlp_act, l.up_w, l.up_b), { b_h }, { b_ff });
        l.down = rw(b, gemm(P, width, ff, "none", l.down_w, l.down_b), { b_ff, b_x }, { b_x });
    }
    op_mln = rw(b, lnorm(P, mn_w, mn_b), { b_x }, { b_mh });
    op_m1  = rw(b, gemm(R, mm * width, mm * width, "gelu", m1_w, m1_b), { b_mh }, { b_mf });
    op_m2  = rw(b, gemm(R, n_embd, mm * width, "none", m2_w, m2_b), { b_mf }, { b_out });
    /* A deepstack merger normalises after the view, over R rows of merge^2 * width. Its ops are
     * declared after the tower and issued between blocks; their working planes are consumed before
     * the next block. Each tap's output must survive until the language model reads it, so it is
     * PERSIST. */
    for (int t = 0; t < n_ds; ++t) {
        b_ds[t] = decl_b(b, nm.f("v.ds.%d.out", t), RAD_BF16, { R, n_embd }, RAD_BUF_PERSIST);
        if (!b_ds[t]) return RAD_E_INVAL;
        ds_ln[t] = rw(b, RAD_OP(b, "layernorm",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", mm * width),
                                           RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                                RAD_WEIGHTS(ds_nw[t], ds_nb[t])),
                      { b_x }, { b_mh });
        ds_1[t] = rw(b, gemm(R, mm * width, mm * width, "gelu", ds_1w[t], ds_1b[t]), { b_mh }, { b_mf });
        ds_2[t] = rw(b, gemm(R, n_embd, mm * width, "none", ds_2w[t], ds_2b[t]), { b_mf }, { b_ds[t] });
    }

    if (!ctx->shape_probe) {
        RadEncoderDecl e{};
        e.name        = "vision";
        e.modalities  = RAD_MM_IMAGE | RAD_MM_VIDEO;
        e.patch_dim   = patch_dim;
        e.merge       = mm;
        e.max_patches = P;
        e.n_embd      = n_embd;
        e.out         = b_out;
        RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        rad_note(b, "vision tower: %lld blocks of %lld (%lld heads x %lld), ff %lld, patch %lld x "
                    "%lld x %lld, merge %lld; %lld patches a pass -> %lld rows of %lld",
                 (long long)depth, (long long)width, (long long)heads, (long long)head_dim,
                 (long long)ff, (long long)temporal, (long long)patch, (long long)patch,
                 (long long)merge, (long long)P, (long long)R, (long long)n_embd);
    }
    on = true;
    return RAD_OK;
}

inline void VisionTower::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg;
    const int64_t mm = merge * merge, Rn = Pn / mm;
    if (!on || Pn <= 0 || Pn > P || Pn % mm || S <= 0) {
        rad_step_fail(c, "an encoder pass this program's vision tower cannot serve: no tower, or "
                         "more patches than it was declared for, or not whole merged rows");
        return;
    }
    /* Patches, coordinates (planes 0-1 a dense [2, P] of (row, column); all four a dense [4, P])
     * and segment boundaries, staged by the core. */
    const RadOperand pix = praw2(batch->enc_pixels, RAD_BF16, Pn, patch_dim);
    const RadOperand rc  = praw2(batch->enc_coord, RAD_I32, 2, Pn);
    const RadOperand crd = praw2(batch->enc_coord, RAD_I32, 4, Pn);
    const RadOperand cu  = praw(batch->enc_cu, RAD_I32, S + 1);

    RAD_ISSUE_N(c, op_patch, Pn, pix, RAD_W(w_patch), RAD_W(b_patch), RAD_NONE, brows(b_x, Pn));
    RAD_ISSUE_N(c, op_pos, Pn, brows(b_x, Pn), RAD_W(w_pos), crd);
    for (size_t li = 0; li < L.size(); ++li) {
        const Layer& l = L[li];
        RAD_ISSUE_N(c, l.ln1, Pn, brows(b_x, Pn), RAD_W(l.n1w), RAD_W(l.n1b), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.qkv, Pn, brows(b_h, Pn), RAD_W(l.qkv_w), RAD_W(l.qkv_b), RAD_NONE,
                    brows(b_qkv, Pn));
        RAD_ISSUE_N(c, l.rope, Pn, brows(b_qkv, Pn), rc);
        RAD_ISSUE_N(c, l.attn, Pn,
                    bcol(b_qkv, 0, width, Pn), bcol(b_qkv, width, width, Pn),
                    bcol(b_qkv, 2 * width, width, Pn), cu, brows(b_att, Pn));
        RAD_ISSUE_N(c, l.proj, Pn, brows(b_att, Pn), RAD_W(l.out_w), RAD_W(l.out_b),
                    brows(b_x, Pn), brows(b_x, Pn));
        RAD_ISSUE_N(c, l.ln2, Pn, brows(b_x, Pn), RAD_W(l.n2w), RAD_W(l.n2b), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.up, Pn, brows(b_h, Pn), RAD_W(l.up_w), RAD_W(l.up_b), RAD_NONE,
                    brows(b_ff, Pn));
        RAD_ISSUE_N(c, l.down, Pn, brows(b_ff, Pn), RAD_W(l.down_w), RAD_W(l.down_b),
                    brows(b_x, Pn), brows(b_x, Pn));
        for (int t = 0; t < n_ds; ++t) {
            if (ds_layer[t] != (int)li) continue;
            RAD_ISSUE_N(c, ds_ln[t], Rn, brows(b_x, Pn), RAD_W(ds_nw[t]), RAD_W(ds_nb[t]),
                        brows(b_mh, Rn));
            RAD_ISSUE_N(c, ds_1[t], Rn, brows(b_mh, Rn), RAD_W(ds_1w[t]), RAD_W(ds_1b[t]), RAD_NONE,
                        brows(b_mf, Rn));
            RAD_ISSUE_N(c, ds_2[t], Rn, brows(b_mf, Rn), RAD_W(ds_2w[t]), RAD_W(ds_2b[t]), RAD_NONE,
                        brows(b_ds[t], Rn));
        }
    }
    /* The merger norm writes P rows of `width`; the next layer reads them as P/mm merged rows. */
    RAD_ISSUE_N(c, op_mln, Pn, brows(b_x, Pn), RAD_W(mn_w), RAD_W(mn_b), brows(b_mh, Rn));
    RAD_ISSUE_N(c, op_m1, Rn, brows(b_mh, Rn), RAD_W(m1_w), RAD_W(m1_b), RAD_NONE,
                brows(b_mf, Rn));
    RAD_ISSUE_N(c, op_m2, Rn, brows(b_mf, Rn), RAD_W(m2_w), RAD_W(m2_b), RAD_NONE,
                brows(b_out, Rn));
}

/* ================================================================== the encoder rows
 *
 * Scatters encoder rows over the token embeddings they replace (RadBatch::mm_rows): one
 * `scatter_rows` of `n_embd` bf16 rows into the embedding buffer.
 *
 * Issue it before the embedding's reduction. The lookup is vocab-sharded, so each rank holds a
 * partial sum; the core gives rank 0 the encoder rows and other ranks zeros, so the following
 * reduction (an all_reduce, or the first layer's fused norm) counts each row once. */
struct MediaRows {
    rad_op  op = 0;
    int64_t n_embd = 0;

    int declare(RadBuilder* b, int64_t max_tok, int64_t width, rad_buf embd) {
        n_embd = width;
        op = rw(b, RAD_OP(b, "scatter_rows",
                          RAD_PARAMS(RAD_RANGE("M", 1, max_tok), RAD_INT("n", width),
                                     RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS),
                { embd }, { embd });
        return op ? RAD_OK : RAD_E_INVAL;
    }

    /* `embd` narrowed to this pass's tokens; nothing is issued on a pass with no media rows. */
    void step(RadCtx* c, const RadBatch* batch, rad_buf embd, int64_t T) const {
        if (!op || batch->n_mm_rows <= 0) return;
        RAD_ISSUE_N(c, op, batch->n_mm_rows,
                    praw2(batch->mm_embd, RAD_BF16, batch->n_mm_rows, n_embd),
                    praw(batch->mm_rows, RAD_I32, batch->n_mm_rows), brows(embd, T));
    }
};

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_VIT_H */
