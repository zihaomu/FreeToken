/* rad_block_gemma4_vision.h -- Gemma 4's vision tower (transformers' Gemma4VisionModel), served by
 * gemma4_bf16 (the 31B's tower: 27 layers of width 1152) and embgemma2 (16 layers of width 768).
 *
 *   a patch is 16 x 16 x 3, staged by the core's Gemma 4 processor as 2 * (u / 255) - 1 in bf16,
 *   (y, x, channel); patch embedding a [width, 768] linear plus a learned row for the patch's
 *   column and one for its row (a [2, npos, width] table), the two summed first
 *   layers: sandwich norms (gain w, not 1 + w), q/k RMS-normed with a gain and v without one, a
 *   2-D rotary over two neox blocks of a head -- x (the column) over the first, y over the second,
 *   each block at theta^(-j / (hd / 4)) (rope_type "default"; EmbeddingGemma 2 spells the same
 *   rotary "axial") -- bidirectional attention within an image at scale 1, a gated gelu-tanh MLP
 *   pooling: the mean of each pool x pool window times sqrt(width) (the processor lays a window's
 *   patches out consecutively, raster order of windows, as the reference's position pooling)
 *   standardisation (vision_config.standardize), when set: (z - std_bias) * std_scale per channel.
 *   The reference does it in f32 because a few channels carry ~5e4 with a bias that cancels them;
 *   here z is the pooler's bf16 row, so those channels keep bf16's spacing at that magnitude
 *   (about 1e-2 after the scale). The bias row is made on the card: fill(-1) times std_bias, added
 *   to z, then times std_scale.
 *   the multimodal embedder: an RMS norm with no gain and a width -> n_embd linear. Its rows
 *   replace the image's placeholder embeddings (which the text model scales by sqrt(n_embd); the
 *   image rows are not scaled).
 *
 * Clipped linears (vision_config.use_clipped_linears, Gemma 4 E2B/E4B): every projection clamps its
 * input and its output to bounds the checkpoint stores (`<proj>.input_min/max`, `output_min/max`;
 * transformers' Gemma4ClippableLinear). The tower keeps its fused q|k|v and gate|up GEMMs, so it
 * takes the input bounds of q_proj (and of gate_proj) for the fused input, q_proj's output bounds for
 * q's columns, k_proj's for k's and v's, and gate_proj's output bounds for gate|up. That is exact
 * where the fused parts share their bounds, as NVIDIA's E4B does in all 16 layers (k and v share the
 * input bounds of q and their output bounds, up shares gate's). The bound weights are one bf16
 * element each, read by `clamp`.
 *
 * One encoder pass carries RadBuildCtx::max_enc_patches patches over any number of segments (an
 * image, or one frame of a video). Every rank runs the whole tower (engine_mm.cpp), so its weights
 * are replicated (RAD_SHARD_NONE) and each rank writes the rows into its own copy of the embedding
 * stream. */
#ifndef RAD_BLOCK_GEMMA4_VISION_H
#define RAD_BLOCK_GEMMA4_VISION_H

#include "rad_arch.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace rad {
namespace arch {

struct Gemma4Vision {
    struct Layer {
        rad_weight in_norm = 0, qkv = 0, q_norm = 0, k_norm = 0, o = 0, post_attn = 0;
        rad_weight pre_ff = 0, gate_up = 0, down = 0, post_ff = 0;
        rad_op op_in = 0, op_qkv = 0, op_qk = 0, op_attn = 0, op_o = 0, op_pa = 0, op_add1 = 0;
        rad_op op_pf = 0, op_gu = 0, op_act = 0, op_down = 0, op_pff = 0, op_add2 = 0;
        /* clipped linears: the bounds (lo, hi) of the nine clamps, and the clamps */
        enum { C_QKV_IN, C_Q_OUT, C_KV_OUT, C_O_IN, C_O_OUT, C_GU_IN, C_GU_OUT, C_DN_IN, C_DN_OUT, C_N };
        rad_weight lo[C_N] = {}, hi[C_N] = {};
        rad_op     op_c[C_N] = {};
    };
    bool on = false, standardize = false, clip = false;
    /* The language model reads a picture's rows both ways (RadEncoderDecl::bidirectional); the
     * plugin sets it before declare. */
    bool bidirectional = false;
    int64_t P = 0, R = 0, width = 0, heads = 0, hd = 0, ff = 0, patch_dim = 0, pool = 0;
    int64_t npos = 0, n_embd = 0;
    rad_weight w_in = 0, w_px = 0, w_py = 0, w_proj = 0, w_sbias = 0, w_sscale = 0;
    rad_op op_patch = 0, op_px = 0, op_py = 0, op_padd = 0, op_add = 0, op_pool = 0, op_norm = 0;
    rad_op op_out = 0, op_sfill = 0, op_sbias = 0, op_sadd = 0, op_sscale = 0;
    rad_buf b_x = 0, b_h = 0, b_o = 0, b_qkv = 0, b_att = 0, b_gu = 0, b_ff = 0, b_px = 0,
            b_py = 0, b_pool = 0, b_sb = 0, b_pn = 0, b_out = 0;
    std::vector<Layer> L;

    static bool present(const RadModelMeta* m) {
        const char* t = rad_meta_gets(m, "vision_config.model_type", "");
        return t && !std::strcmp(t, "gemma4_vision");
    }

    /* `pre` is the checkpoint prefix of the tower and its embedder ("model." for Gemma 4's
     * conditional-generation checkpoints, "" for EmbeddingGemma 2); `who` names the plugin in
     * refusals. The tower's layers take weight groups from `first_group`. */
    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, int first_group, const char* pre, const char* who) {
        on = false;
        if (!present(m) || ctx->max_enc_patches <= 0) return RAD_OK;
        width = rad_meta_geti(m, "vision_config.hidden_size", 0);
        heads = rad_meta_geti(m, "vision_config.num_attention_heads", 0);
        hd    = rad_meta_geti(m, "vision_config.head_dim", 0);
        ff    = rad_meta_geti(m, "vision_config.intermediate_size", 0);
        pool  = rad_meta_geti(m, "vision_config.pooling_kernel_size", 0);
        npos  = rad_meta_geti(m, "vision_config.position_embedding_size", 0);
        const int64_t depth = rad_meta_geti(m, "vision_config.num_hidden_layers", 0);
        const int64_t patch = rad_meta_geti(m, "vision_config.patch_size", 0);
        const int64_t kvh   = rad_meta_geti(m, "vision_config.num_key_value_heads", heads);
        const double  eps   = rad_meta_getf(m, "vision_config.rms_norm_eps", 1e-6);
        const double  theta = rad_meta_getf(m, "vision_config.rope_parameters.rope_theta", 0.0);
        const char*   rt    = rad_meta_gets(m, "vision_config.rope_parameters.rope_type", "");
        const char*   act   = rad_meta_gets(m, "vision_config.hidden_activation", "");
        standardize = rad_meta_geti(m, "vision_config.standardize", 0) != 0;
        if (width <= 0 || heads <= 0 || hd <= 0 || ff <= 0 || pool <= 0 || npos <= 0 ||
            depth <= 0 || patch <= 0 || !(theta > 0) || heads * hd != width || kvh != heads ||
            hd % 4) {
            fprintf(stderr, "radiance: %s: the container's vision_config is incomplete or not the "
                            "shape this tower serves (hidden %lld, heads %lld x %lld, kv %lld, ff "
                            "%lld, pooling %lld, layers %lld)\n", who, (long long)width,
                    (long long)heads, (long long)hd, (long long)kvh, (long long)ff,
                    (long long)pool, (long long)depth);
            return RAD_E_INVAL;
        }
        /* Any rotary or activation other than the 2-D one and gelu-tanh is refused by name. */
        if ((std::strcmp(rt, "default") && std::strcmp(rt, "axial")) ||
            std::strcmp(act, "gelu_pytorch_tanh")) {
            fprintf(stderr, "radiance: %s: this vision tower uses a '%s' rotary or a '%s' "
                            "activation, which it does not serve\n", who, rt, act);
            return RAD_E_UNSUPPORTED;
        }
        clip = rad_meta_geti(m, "vision_config.use_clipped_linears", 0) != 0;
        n_embd = model_embd;
        patch_dim = 3 * patch * patch;
        const int64_t kk = pool * pool;
        P = ctx->max_enc_patches;
        if (P % kk) P -= P % kk;
        R = P / kk;
        const int A = RAD_ACCESS_PER_REQUEST, N = RAD_SHARD_NONE;
        char V[64];
        std::snprintf(V, sizeof V, "%svision_tower", pre);

        RAD_ARCH_TRY(map_copy(b, nm.f("v.patch.weight"), nm.ckpt("%s.patch_embedder.input_proj.weight", V)));
        RAD_ARCH_TRY(map_slice(b, nm.f("v.pos_x.weight"),
                               nm.ckpt("%s.patch_embedder.position_embedding_table", V), 0));
        RAD_ARCH_TRY(map_slice(b, nm.f("v.pos_y.weight"),
                               nm.ckpt("%s.patch_embedder.position_embedding_table", V), 1));
        RAD_ARCH_TRY(map_copy(b, nm.f("v.embed.weight"), nm.ckpt("%sembed_vision.embedding_projection.weight", pre)));
        w_in   = decl_w(b, nm.f("v.patch.weight"), RAD_BF16, {width, patch_dim}, A, N, grp_model());
        w_px   = decl_w(b, nm.f("v.pos_x.weight"), RAD_BF16, {npos, width}, A, N, grp_model());
        w_py   = decl_w(b, nm.f("v.pos_y.weight"), RAD_BF16, {npos, width}, A, N, grp_model());
        w_proj = decl_w(b, nm.f("v.embed.weight"), RAD_BF16, {n_embd, width}, A, N, grp_model());
        if (standardize) {
            RAD_ARCH_TRY(map_copy(b, nm.f("v.std_bias"), nm.ckpt("%s.std_bias", V)));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.std_scale"), nm.ckpt("%s.std_scale", V)));
            w_sbias  = decl_w(b, nm.f("v.std_bias"), RAD_BF16, {width}, A, N, grp_model());
            w_sscale = decl_w(b, nm.f("v.std_scale"), RAD_BF16, {width}, A, N, grp_model());
        }
        L.assign((size_t)depth, Layer{});
        for (int64_t i = 0; i < depth; ++i) {
            Layer& l = L[(size_t)i];
            const long long li = (long long)i;
            const RadWeightGroup g = grp_layer(first_group + (int)i);
            auto src = [&](const char* s) { return nm.ckpt("%s.encoder.layers.%lld.%s", V, li, s); };
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.in_norm", li), src("input_layernorm.weight")));
            RAD_ARCH_TRY(map_concat(b, nm.f("v.blk.%lld.qkv", li),
                                    {src("self_attn.q_proj.linear.weight"),
                                     src("self_attn.k_proj.linear.weight"),
                                     src("self_attn.v_proj.linear.weight")}));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.q_norm", li), src("self_attn.q_norm.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.k_norm", li), src("self_attn.k_norm.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.o", li), src("self_attn.o_proj.linear.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.post_attn", li), src("post_attention_layernorm.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.pre_ff", li), src("pre_feedforward_layernorm.weight")));
            RAD_ARCH_TRY(map_concat(b, nm.f("v.blk.%lld.gate_up", li),
                                    {src("mlp.gate_proj.linear.weight"),
                                     src("mlp.up_proj.linear.weight")}));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.down", li), src("mlp.down_proj.linear.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.post_ff", li), src("post_feedforward_layernorm.weight")));
            l.in_norm   = decl_w(b, nm.f("v.blk.%lld.in_norm", li),   RAD_BF16, {width}, A, N, g);
            l.qkv       = decl_w(b, nm.f("v.blk.%lld.qkv", li),       RAD_BF16, {3 * width, width}, A, N, g);
            l.q_norm    = decl_w(b, nm.f("v.blk.%lld.q_norm", li),    RAD_BF16, {hd}, A, N, g);
            l.k_norm    = decl_w(b, nm.f("v.blk.%lld.k_norm", li),    RAD_BF16, {hd}, A, N, g);
            l.o         = decl_w(b, nm.f("v.blk.%lld.o", li),         RAD_BF16, {width, width}, A, N, g);
            l.post_attn = decl_w(b, nm.f("v.blk.%lld.post_attn", li), RAD_BF16, {width}, A, N, g);
            l.pre_ff    = decl_w(b, nm.f("v.blk.%lld.pre_ff", li),    RAD_BF16, {width}, A, N, g);
            l.gate_up   = decl_w(b, nm.f("v.blk.%lld.gate_up", li),   RAD_BF16, {2 * ff, width}, A, N, g);
            l.down      = decl_w(b, nm.f("v.blk.%lld.down", li),      RAD_BF16, {width, ff}, A, N, g);
            l.post_ff   = decl_w(b, nm.f("v.blk.%lld.post_ff", li),   RAD_BF16, {width}, A, N, g);
            if (clip) {
                /* clamp k -> the checkpoint projection whose bound it takes, and which bound */
                static const char* const proj[Layer::C_N] = {
                    "self_attn.q_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj",
                    "self_attn.o_proj", "mlp.gate_proj", "mlp.gate_proj", "mlp.down_proj",
                    "mlp.down_proj" };
                static const bool in_side[Layer::C_N] = { true, false, false, true, false, true,
                                                          false, true, false };
                for (int k = 0; k < Layer::C_N; ++k) {
                    const char* side = in_side[k] ? "input" : "output";
                    const char* lo_n = nm.f("v.blk.%lld.clip%d.lo", li, k);
                    const char* hi_n = nm.f("v.blk.%lld.clip%d.hi", li, k);
                    RAD_ARCH_TRY(map_copy(b, lo_n, nm.ckpt("%s.encoder.layers.%lld.%s.%s_min", V, li,
                                                           proj[k], side)));
                    RAD_ARCH_TRY(map_copy(b, hi_n, nm.ckpt("%s.encoder.layers.%lld.%s.%s_max", V, li,
                                                           proj[k], side)));
                    l.lo[k] = decl_w(b, lo_n, RAD_BF16, {1}, A, N, g);
                    l.hi[k] = decl_w(b, hi_n, RAD_BF16, {1}, A, N, g);
                    if (!l.lo[k] || !l.hi[k]) return RAD_E_INVAL;
                }
            }
        }

        b_x    = decl_b(b, nm.f("v.x"),    RAD_BF16, {P, width});
        b_h    = decl_b(b, nm.f("v.h"),    RAD_BF16, {P, width});
        b_o    = decl_b(b, nm.f("v.o"),    RAD_BF16, {P, width});
        b_qkv  = decl_b(b, nm.f("v.qkv"),  RAD_BF16, {P, 3 * width});
        b_att  = decl_b(b, nm.f("v.att"),  RAD_BF16, {P, width});
        b_gu   = decl_b(b, nm.f("v.gu"),   RAD_BF16, {P, 2 * ff});
        b_ff   = decl_b(b, nm.f("v.ff"),   RAD_BF16, {P, ff});
        b_px   = decl_b(b, nm.f("v.px"),   RAD_BF16, {P, width});
        b_py   = decl_b(b, nm.f("v.py"),   RAD_BF16, {P, width});
        b_pool = decl_b(b, nm.f("v.pool"), RAD_BF16, {R, width});
        b_sb   = decl_b(b, nm.f("v.sb"),   RAD_BF16, {R, width});
        b_pn   = decl_b(b, nm.f("v.pn"),   RAD_BF16, {R, width});
        b_out  = decl_b(b, nm.f("v.out"),  RAD_BF16, {R, n_embd});

        auto rms = [&](int64_t rows, rad_weight w) {
            if (!w)
                return RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                                       RAD_F64("eps", eps), RAD_F64("wadd", 0.0),
                                                       RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS);
            return RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                                   RAD_F64("eps", eps), RAD_F64("wadd", 0.0),
                                                   RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w));
        };
        auto gemm = [&](int64_t rows, int64_t Nn, int64_t K, rad_weight w) {
            return RAD_OP(b, "gemm_nt", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", Nn),
                                                   RAD_INT("K", K), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w));
        };
        auto add = [&](int64_t rows) {
            return RAD_OP(b, "add", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                               RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS);
        };
        auto lookup = [&](rad_weight w) {
            return RAD_OP(b, "embed_lookup",
                          RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n_embd", width),
                                     RAD_INT("n_vocab", npos), RAD_STR("dtype", "bf16"),
                                     RAD_INT("vocab_offset", 0), RAD_F64("mult", 1.0)),
                          RAD_WEIGHTS(w));
        };
        auto clamp = [&](int64_t n, rad_weight lo, rad_weight hi) {
            return RAD_OP(b, "clamp", RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", n),
                                                 RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(lo, hi));
        };
        auto gain = [&](rad_weight w) {
            return RAD_OP(b, "mul_weight", RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", width),
                                                      RAD_STR("dtype", "bf16"), RAD_INT("scalar", 0)),
                          RAD_WEIGHTS(w));
        };
        op_patch = rw(b, gemm(P, width, patch_dim, w_in), {}, {b_x});
        op_px    = rw(b, lookup(w_px), {}, {b_px});
        op_py    = rw(b, lookup(w_py), {}, {b_py});
        op_padd  = rw(b, add(P), {b_px, b_py}, {b_px});
        op_add   = rw(b, add(P), {b_x, b_px}, {b_x});
        /* Two neox blocks of hd / 4 pairs, x (staged plane 1, the column) then y (plane 0). */
        char sec[48];
        std::snprintf(sec, sizeof sec, "%lld %lld", (long long)(hd / 4), (long long)(hd / 4));
        for (Layer& l : L) {
            /* the clamps, each in place on the one buffer its linear reads or writes (each of those
             * is read by that linear alone, or by what follows it) */
            if (clip) {
                using C = Layer;
                const struct { int k; int64_t n; rad_buf buf; } cl[C::C_N] = {
                    { C::C_QKV_IN, width, b_h }, { C::C_Q_OUT, width, b_qkv },
                    { C::C_KV_OUT, 2 * width, b_qkv }, { C::C_O_IN, width, b_att },
                    { C::C_O_OUT, width, b_o }, { C::C_GU_IN, width, b_h },
                    { C::C_GU_OUT, 2 * ff, b_gu }, { C::C_DN_IN, ff, b_ff },
                    { C::C_DN_OUT, width, b_o } };
                for (const auto& e : cl)
                    l.op_c[e.k] = rw(b, clamp(e.n, l.lo[e.k], l.hi[e.k]), {e.buf}, {e.buf});
            }
            l.op_in   = rw(b, rms(P, l.in_norm), {b_x}, {b_h});
            l.op_qkv  = rw(b, gemm(P, 3 * width, width, l.qkv), {b_h}, {b_qkv});
            l.op_qk   = rw(b, RAD_OP(b, "qk_norm_rope",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", hd),
                                                RAD_INT("n_head", heads), RAD_INT("n_head_kv", heads),
                                                RAD_F64("theta", theta), RAD_F64("eps", eps),
                                                RAD_STR("mode", "blocks"), RAD_INT("v_norm", 1),
                                                RAD_STR("sections", sec), RAD_STR("comps", "1 0")),
                                     RAD_WEIGHTS(l.q_norm, l.k_norm)),
                           {b_qkv}, {b_qkv});
            l.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", hd),
                                                RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                                RAD_STR("dtype", "bf16"), RAD_F64("scale", 1.0),
                                                RAD_INT("max_seqlen", P), RAD_INT("window", 0)),
                                     RAD_NOWEIGHTS),
                           {b_qkv}, {b_att});
            l.op_o    = rw(b, gemm(P, width, width, l.o), {b_att}, {b_o});
            l.op_pa   = rw(b, rms(P, l.post_attn), {b_o}, {b_o});
            l.op_add1 = rw(b, add(P), {b_x, b_o}, {b_x});
            l.op_pf   = rw(b, rms(P, l.pre_ff), {b_x}, {b_h});
            l.op_gu   = rw(b, gemm(P, 2 * ff, width, l.gate_up), {b_h}, {b_gu});
            l.op_act  = rw(b, RAD_OP(b, "act_mul",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", ff),
                                                RAD_STR("act", "gelu_tanh"), RAD_STR("dtype", "bf16")),
                                     RAD_NOWEIGHTS),
                           {b_gu}, {b_ff});
            l.op_down = rw(b, gemm(P, width, ff, l.down), {b_ff}, {b_o});
            l.op_pff  = rw(b, rms(P, l.post_ff), {b_o}, {b_o});
            l.op_add2 = rw(b, add(P), {b_x, b_o}, {b_x});
        }
        op_pool = rw(b, RAD_OP(b, "pool_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                          RAD_INT("k", kk), RAD_F64("mult", std::sqrt((double)width)),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     {b_x}, {b_pool});
        if (standardize) {
            op_sfill  = rw(b, RAD_OP(b, "fill",
                                     RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", width),
                                                RAD_STR("dtype", "bf16"), RAD_F64("value", -1.0)),
                                     RAD_NOWEIGHTS),
                           {}, {b_sb});
            op_sbias  = rw(b, gain(w_sbias), {b_sb}, {b_sb});
            op_sadd   = rw(b, add(R), {b_pool, b_sb}, {b_pool});
            op_sscale = rw(b, gain(w_sscale), {b_pool}, {b_pool});
        }
        op_norm = rw(b, rms(R, 0), {b_pool}, {b_pn});
        op_out  = rw(b, gemm(R, n_embd, width, w_proj), {b_pn}, {b_out});

        RadEncoderDecl e{};
        e.name        = "vision";
        e.modalities  = RAD_MM_IMAGE | RAD_MM_VIDEO;
        e.patch_dim   = patch_dim;
        e.merge       = kk;
        e.max_patches = P;
        e.n_embd      = n_embd;
        e.out         = b_out;
        e.bidirectional = bidirectional ? 1 : 0;
        RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        rad_note(b, "vision tower: %lld layers of %lld (%lld heads x %lld), ff %lld, patch %lld, "
                    "pooled %lldx%lld%s%s; %lld patches a pass -> %lld rows of %lld",
                 (long long)depth, (long long)width, (long long)heads, (long long)hd, (long long)ff,
                 (long long)patch, (long long)pool, (long long)pool,
                 standardize ? ", standardised" : "", clip ? ", clipped linears" : "",
                 (long long)P, (long long)R,
                 (long long)n_embd);
        on = true;
        return RAD_OK;
    }

    void step(RadCtx* c, const RadBatch* batch) const {
        const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg, kk = pool * pool;
        if (!on || Pn <= 0 || Pn > P || Pn % kk || S <= 0) {
            rad_step_fail(c, "an encoder pass this program's vision tower cannot serve: no tower, "
                             "more patches than it was declared for, or not whole pooling windows");
            return;
        }
        const int64_t Rn = Pn / kk;
        const int32_t* crd = batch->enc_coord;
        const RadOperand pix = praw2(batch->enc_pixels, RAD_BF16, Pn, patch_dim);
        const RadOperand rc  = praw2(crd, RAD_I32, 2, Pn);
        const RadOperand cu  = praw(batch->enc_cu, RAD_I32, S + 1);

        RAD_ISSUE_N(c, op_patch, Pn, pix, RAD_W(w_in), brows(b_x, Pn));
        RAD_ISSUE_N(c, op_px, Pn, praw(crd + Pn, RAD_I32, Pn), RAD_W(w_px), brows(b_px, Pn));
        RAD_ISSUE_N(c, op_py, Pn, praw(crd, RAD_I32, Pn), RAD_W(w_py), brows(b_py, Pn));
        RAD_ISSUE_N(c, op_padd, Pn, brows(b_px, Pn), brows(b_py, Pn), brows(b_px, Pn));
        RAD_ISSUE_N(c, op_add, Pn, brows(b_x, Pn), brows(b_px, Pn), brows(b_x, Pn));
        /* a clamp of operand `x` in place, when the tower clips */
        auto cl = [&](const Layer& l, int k, RadOperand x) {
            if (clip) RAD_ISSUE_N(c, l.op_c[k], Pn, x, RAD_W(l.lo[k]), RAD_W(l.hi[k]), x);
        };
        for (const Layer& l : L) {
            RAD_ISSUE_N(c, l.op_in, Pn, brows(b_x, Pn), RAD_W(l.in_norm), brows(b_h, Pn));
            cl(l, Layer::C_QKV_IN, brows(b_h, Pn));
            RAD_ISSUE_N(c, l.op_qkv, Pn, brows(b_h, Pn), RAD_W(l.qkv), brows(b_qkv, Pn));
            cl(l, Layer::C_Q_OUT, bcol(b_qkv, 0, width, Pn));
            cl(l, Layer::C_KV_OUT, bcol(b_qkv, width, 2 * width, Pn));
            RAD_ISSUE_N(c, l.op_qk, Pn, brows(b_qkv, Pn), rc, RAD_W(l.q_norm), RAD_W(l.k_norm));
            RAD_ISSUE_N(c, l.op_attn, Pn, bcol(b_qkv, 0, width, Pn), bcol(b_qkv, width, width, Pn),
                        bcol(b_qkv, 2 * width, width, Pn), cu, brows(b_att, Pn));
            cl(l, Layer::C_O_IN, brows(b_att, Pn));
            RAD_ISSUE_N(c, l.op_o, Pn, brows(b_att, Pn), RAD_W(l.o), brows(b_o, Pn));
            cl(l, Layer::C_O_OUT, brows(b_o, Pn));
            RAD_ISSUE_N(c, l.op_pa, Pn, brows(b_o, Pn), RAD_W(l.post_attn), brows(b_o, Pn));
            RAD_ISSUE_N(c, l.op_add1, Pn, brows(b_x, Pn), brows(b_o, Pn), brows(b_x, Pn));
            RAD_ISSUE_N(c, l.op_pf, Pn, brows(b_x, Pn), RAD_W(l.pre_ff), brows(b_h, Pn));
            cl(l, Layer::C_GU_IN, brows(b_h, Pn));
            RAD_ISSUE_N(c, l.op_gu, Pn, brows(b_h, Pn), RAD_W(l.gate_up), brows(b_gu, Pn));
            cl(l, Layer::C_GU_OUT, brows(b_gu, Pn));
            RAD_ISSUE_N(c, l.op_act, Pn, bcol(b_gu, 0, ff, Pn), bcol(b_gu, ff, ff, Pn),
                        brows(b_ff, Pn));
            cl(l, Layer::C_DN_IN, brows(b_ff, Pn));
            RAD_ISSUE_N(c, l.op_down, Pn, brows(b_ff, Pn), RAD_W(l.down), brows(b_o, Pn));
            cl(l, Layer::C_DN_OUT, brows(b_o, Pn));
            RAD_ISSUE_N(c, l.op_pff, Pn, brows(b_o, Pn), RAD_W(l.post_ff), brows(b_o, Pn));
            RAD_ISSUE_N(c, l.op_add2, Pn, brows(b_x, Pn), brows(b_o, Pn), brows(b_x, Pn));
        }
        RAD_ISSUE_N(c, op_pool, Pn, brows(b_x, Pn), brows(b_pool, Rn));
        if (standardize) {
            RAD_ISSUE_N(c, op_sfill, Rn, brows(b_sb, Rn));
            RAD_ISSUE_N(c, op_sbias, Rn, brows(b_sb, Rn), RAD_W(w_sbias), brows(b_sb, Rn));
            RAD_ISSUE_N(c, op_sadd, Rn, brows(b_pool, Rn), brows(b_sb, Rn), brows(b_pool, Rn));
            RAD_ISSUE_N(c, op_sscale, Rn, brows(b_pool, Rn), RAD_W(w_sscale), brows(b_pool, Rn));
        }
        RAD_ISSUE_N(c, op_norm, Rn, brows(b_pool, Rn), RAD_NONE, brows(b_pn, Rn));
        RAD_ISSUE_N(c, op_out, Rn, brows(b_pn, Rn), RAD_W(w_proj), brows(b_out, Rn));
    }
};

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_GEMMA4_VISION_H */
