/* rad_block_gemma4_audio.h -- Gemma 4's audio encoder (USM conformer), shared by the plugins whose
 * checkpoints carry it: EmbeddingGemma 2 (embgemma2) and Gemma 4 E2B/E4B (gemma4_bf16).
 *
 *   a frame is 128 log-mel bins (the processor's), four frames an output row
 *   subsampling: two 3x3 stride-2 convolutions over (time, frequency) -- 1 -> 128 -> 32 channels,
 *   each input masked to the clip's real frames first, each followed by a channel LayerNorm (no
 *   bias) and a ReLU -- then 32 x 32 features a row through a 1024 linear
 *   12 conformer blocks: a feed-forward at half weight, local attention (11 frames back, a
 *   relative-position term, a softplus per-dimension query scale, a softcap of 50), a gated
 *   depthwise causal convolution, a second half-weight feed-forward, a norm; their linears are
 *   clipped: input and output clamped to bounds the checkpoint stores
 *   1024 -> 1536 with a bias, then the multimodal embedder: an RMS norm with no gain, 1536 -> the
 *   text model's width.
 *
 * The blocks' clamps at +-1e10 ("gradient clipping") are omitted: they are ten orders of magnitude
 * beyond the activations' range and change nothing. One clip is one segment; a pass carries up to
 * RadBuildCtx::max_audio_frames frames. */
#ifndef RAD_BLOCK_GEMMA4_AUDIO_H
#define RAD_BLOCK_GEMMA4_AUDIO_H

#include "rad_arch.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace rad {
namespace arch {

struct Gemma4Audio {
    /* A clipped linear: the weight and its four bounds. */
    struct Clip {
        rad_weight w = 0, in_lo = 0, in_hi = 0, out_lo = 0, out_hi = 0;
        rad_op op_in = 0, op_mm = 0, op_out = 0;
    };
    struct FF {
        rad_weight pre = 0, post = 0;
        Clip l1, l2;
        rad_op op_pre = 0, op_act = 0, op_post = 0, op_add = 0;
    };
    struct Layer {
        FF ff1, ff2;
        rad_weight n_pre_attn = 0, n_post_attn = 0, n_out = 0, w_rel = 0, pds = 0;
        Clip q, k, v, post;
        rad_weight lc_pre = 0, lc_dw = 0, lc_norm = 0;
        Clip lc_start, lc_end;
        rad_op op_npa = 0, op_rel = 0, op_attn = 0, op_npo = 0, op_add_attn = 0;
        rad_op op_lc_pre = 0, op_glu = 0, op_dw = 0, op_lc_norm = 0, op_lc_act = 0, op_lc_add = 0;
        rad_op op_out = 0;
    };
    bool on = false;
    int64_t P = 0, R = 0, H = 0, heads = 0, hd = 0, ffw = 0, n_embd = 0, out_dims = 0;
    int64_t c0 = 0, c1 = 0, nmel = 0, merge = 4, left = 0, chunk = 0, n_rel = 0;
    double eps = 1e-6, softcap = 0;
    rad_weight w_c0 = 0, w_c1 = 0, w_n0 = 0, w_n1 = 0, w_in = 0, w_out = 0, b_outw = 0, w_emb = 0;
    rad_op op_c0 = 0, op_n0 = 0, op_r0 = 0, op_c1 = 0, op_n1 = 0, op_r1 = 0, op_in = 0;
    rad_op op_relpos = 0, op_outp = 0, op_en = 0, op_emb = 0;
    rad_buf b_s0 = 0, b_s1 = 0, b_x = 0, b_h = 0, b_hc = 0, b_f = 0, b_q = 0, b_k = 0, b_v = 0;
    rad_buf b_att = 0, b_o = 0, b_g = 0, b_u = 0, b_relpos = 0, b_relk = 0, b_o2 = 0, b_on = 0;
    rad_buf b_out = 0;
    std::vector<Layer> L;

    /* A feature-extractor key: preprocessor_config.json's (`preprocessor.`), else
     * processor_config.json's `feature_extractor` object (`processor.feature_extractor.`). */
    static int64_t fe_i(const RadModelMeta* m, const char* k, int64_t d) {
        char a[96], b[128];
        std::snprintf(a, sizeof a, "preprocessor.%s", k);
        std::snprintf(b, sizeof b, "processor.feature_extractor.%s", k);
        return rad_meta_geti(m, a, rad_meta_geti(m, b, d));
    }

    static bool present(const RadModelMeta* m) {
        const char* t = rad_meta_gets(m, "audio_config.model_type", "");
        return t && !std::strcmp(t, "gemma4_audio");
    }
    /* The longest clip in frames: 30 s at the extractor's 10 ms hop, rounded up to whole rows. */
    static int64_t frames_for(const RadModelMeta* m) {
        const int64_t rate = fe_i(m, "sampling_rate", 16000);
        const int64_t fl = fe_i(m, "frame_length", 320);
        const int64_t hop = fe_i(m, "hop_length", 160);
        if (rate <= 0 || hop <= 0) return 0;
        const int64_t n = 30 * rate;
        const int64_t real = (n + fl / 2 - fl - 1) / hop + 1;
        return (real + 3) / 4 * 4;
    }

    /* `pre` is the checkpoint prefix of the encoder and its embedder ("model." for Gemma 4's
     * conditional-generation checkpoints, "" for EmbeddingGemma 2); `who` names the plugin in
     * refusals. The encoder's layers take weight groups from `first_group`. */
    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, int first_group, const char* pre, const char* who) {
        on = false;
        if (!present(m) || ctx->max_audio_frames <= 0) return RAD_OK;
        const char* A = "audio_config";
        auto gi = [&](const char* k, int64_t d) {
            char key[96];
            std::snprintf(key, sizeof key, "%s.%s", A, k);
            return rad_meta_geti(m, key, d);
        };
        auto gf = [&](const char* k, double d) {
            char key[96];
            std::snprintf(key, sizeof key, "%s.%s", A, k);
            return rad_meta_getf(m, key, d);
        };
        H        = gi("hidden_size", 0);
        heads    = gi("num_attention_heads", 0);
        out_dims = gi("output_proj_dims", 0);
        chunk    = gi("attention_chunk_size", 0);
        /* transformers masks a key unless 0 <= q - k < attention_context_left - 1, so the
         * furthest key back is context_left - 2 frames: 11 for this checkpoint, the query and
         * eleven before it. */
        left     = gi("attention_context_left", 0) - 2;
        softcap  = gf("attention_logit_cap", 0.0);
        eps      = gf("rms_norm_eps", 1e-6);
        const int64_t depth = gi("num_hidden_layers", 0);
        const int64_t kconv = gi("conv_kernel_size", 0);
        const int64_t right = gi("attention_context_right", 0);
        const double  resw  = gf("residual_weight", 0.0);
        const char*   act   = rad_meta_gets(m, "audio_config.hidden_act", "");
        const char*   chs   = rad_meta_gets(m, "audio_config.subsampling_conv_channels", "");
        nmel = fe_i(m, "feature_size", 0);
        if (std::sscanf(chs ? chs : "", "%lld %lld", (long long*)&c0, (long long*)&c1) != 2)
            c0 = c1 = 0;
        hd = heads > 0 ? H / heads : 0;
        ffw = 4 * H;
        n_rel = left + 1;
        if (H <= 0 || heads <= 0 || hd * heads != H || out_dims <= 0 || depth <= 0 || kconv <= 0 ||
            chunk <= 0 || left < 0 || right != 0 || !(softcap > 0) || resw != 0.5 ||
            std::strcmp(act ? act : "", "silu") || c0 <= 0 || c1 <= 0 || nmel != 128 ||
            (nmel / 4) * c1 != H) {
            fprintf(stderr, "radiance: %s: the container's audio_config is not the "
                            "conformer this plugin serves (hidden %lld, heads %lld, output %lld, "
                            "layers %lld, conv %lld, left %lld right %lld, residual %g, act %s, "
                            "subsampling %lld/%lld over %lld bins)\n", who, (long long)H,
                    (long long)heads, (long long)out_dims, (long long)depth, (long long)kconv,
                    (long long)left, (long long)right, resw, act ? act : "", (long long)c0,
                    (long long)c1, (long long)nmel);
            return RAD_E_UNSUPPORTED;
        }
        n_embd = model_embd;
        P = ctx->max_audio_frames;
        if (P % merge) return RAD_E_SHAPE;
        R = P / merge;
        const int AC = RAD_ACCESS_PER_REQUEST, N = RAD_SHARD_NONE;
        char T[64];
        std::snprintf(T, sizeof T, "%saudio_tower", pre);

        /* ---- weights */
        auto w = [&](const char* dn, const char* src, std::initializer_list<int64_t> shp,
                     RadWeightGroup g) -> rad_weight {
            if (map_copy(b, dn, src) < 0) return 0;
            return decl_w(b, dn, RAD_BF16, shp, AC, N, g);
        };
        auto clip = [&](Clip& c, const char* dn, const char* src, int64_t out, int64_t in,
                        RadWeightGroup g) {
            char d1[160], s1[200];
            std::snprintf(d1, sizeof d1, "%s.w", dn); std::snprintf(s1, sizeof s1, "%s.linear.weight", src);
            c.w = w(nm.f("%s", d1), nm.ckpt("%s", s1), {out, in}, g);
            const char* bn[4] = { "input_min", "input_max", "output_min", "output_max" };
            rad_weight* dst[4] = { &c.in_lo, &c.in_hi, &c.out_lo, &c.out_hi };
            for (int i = 0; i < 4; ++i) {
                std::snprintf(d1, sizeof d1, "%s.%s", dn, bn[i]);
                std::snprintf(s1, sizeof s1, "%s.%s", src, bn[i]);
                *dst[i] = w(nm.f("%s", d1), nm.ckpt("%s", s1), {1}, g);
            }
        };
        w_c0 = w(nm.f("a.sscp.c0"), nm.ckpt("%s.subsample_conv_projection.layer0.conv.weight", T), {c0, 1, 3, 3}, grp_model());
        w_n0 = w(nm.f("a.sscp.n0"), nm.ckpt("%s.subsample_conv_projection.layer0.norm.weight", T), {c0}, grp_model());
        w_c1 = w(nm.f("a.sscp.c1"), nm.ckpt("%s.subsample_conv_projection.layer1.conv.weight", T), {c1, c0, 3, 3}, grp_model());
        w_n1 = w(nm.f("a.sscp.n1"), nm.ckpt("%s.subsample_conv_projection.layer1.norm.weight", T), {c1}, grp_model());
        w_in = w(nm.f("a.sscp.proj"), nm.ckpt("%s.subsample_conv_projection.input_proj_linear.weight", T), {H, H}, grp_model());
        w_out  = w(nm.f("a.out.w"), nm.ckpt("%s.output_proj.weight", T), {out_dims, H}, grp_model());
        b_outw = w(nm.f("a.out.b"), nm.ckpt("%s.output_proj.bias", T), {out_dims}, grp_model());
        w_emb  = w(nm.f("a.embed"), nm.ckpt("%sembed_audio.embedding_projection.weight", pre), {n_embd, out_dims}, grp_model());
        L.assign((size_t)depth, Layer{});
        for (int64_t i = 0; i < depth; ++i) {
            Layer& l = L[(size_t)i];
            const long long li = (long long)i;
            const RadWeightGroup g = grp_layer(first_group + (int)i);
            char d[96], s[160];
            auto D = [&](const char* x) { std::snprintf(d, sizeof d, "a.blk.%lld.%s", li, x); return nm.f("%s", d); };
            auto S = [&](const char* x) { std::snprintf(s, sizeof s, "%s.layers.%lld.%s", T, li, x); return nm.ckpt("%s", s); };
            for (int f = 0; f < 2; ++f) {
                FF& ff = f ? l.ff2 : l.ff1;
                const char* fn = f ? "feed_forward2" : "feed_forward1";
                char px[64]; std::snprintf(px, sizeof px, "%s.pre_layer_norm.weight", fn);
                ff.pre = w(D(px), S(px), {H}, g);
                std::snprintf(px, sizeof px, "%s.post_layer_norm.weight", fn);
                ff.post = w(D(px), S(px), {H}, g);
                char dn[96], sn[160];
                std::snprintf(dn, sizeof dn, "a.blk.%lld.%s.l1", li, fn);
                std::snprintf(sn, sizeof sn, "%s.layers.%lld.%s.ffw_layer_1", T, li, fn);
                clip(ff.l1, dn, sn, ffw, H, g);
                std::snprintf(dn, sizeof dn, "a.blk.%lld.%s.l2", li, fn);
                std::snprintf(sn, sizeof sn, "%s.layers.%lld.%s.ffw_layer_2", T, li, fn);
                clip(ff.l2, dn, sn, H, ffw, g);
            }
            l.n_pre_attn  = w(D("norm_pre_attn"), S("norm_pre_attn.weight"), {H}, g);
            l.n_post_attn = w(D("norm_post_attn"), S("norm_post_attn.weight"), {H}, g);
            l.n_out       = w(D("norm_out"), S("norm_out.weight"), {H}, g);
            l.w_rel       = w(D("rel_k"), S("self_attn.relative_k_proj.weight"), {H, H}, g);
            l.pds         = w(D("per_dim_scale"), S("self_attn.per_dim_scale"), {hd}, g);
            const char* qn[4] = { "q_proj", "k_proj", "v_proj", "post" };
            Clip* qc[4] = { &l.q, &l.k, &l.v, &l.post };
            for (int j = 0; j < 4; ++j) {
                char dn[96], sn[160];
                std::snprintf(dn, sizeof dn, "a.blk.%lld.attn.%s", li, qn[j]);
                std::snprintf(sn, sizeof sn, "%s.layers.%lld.self_attn.%s", T, li, qn[j]);
                clip(*qc[j], dn, sn, H, H, g);
            }
            l.lc_pre  = w(D("lconv.pre"), S("lconv1d.pre_layer_norm.weight"), {H}, g);
            l.lc_dw   = w(D("lconv.dw"), S("lconv1d.depthwise_conv1d.weight"), {H, 1, kconv}, g);
            l.lc_norm = w(D("lconv.norm"), S("lconv1d.conv_norm.weight"), {H}, g);
            {
                char dn[96], sn[160];
                std::snprintf(dn, sizeof dn, "a.blk.%lld.lconv.start", li);
                std::snprintf(sn, sizeof sn, "%s.layers.%lld.lconv1d.linear_start", T, li);
                clip(l.lc_start, dn, sn, 2 * H, H, g);
                std::snprintf(dn, sizeof dn, "a.blk.%lld.lconv.end", li);
                std::snprintf(sn, sizeof sn, "%s.layers.%lld.lconv1d.linear_end", T, li);
                clip(l.lc_end, dn, sn, H, H, g);
            }
        }

        /* ---- activations: the subsampler's planes at their own widths, the blocks' at R rows */
        b_s0     = decl_b(b, nm.f("a.s0"),     RAD_BF16, {P / 2, (nmel / 2) * c0});
        b_s1     = decl_b(b, nm.f("a.s1"),     RAD_BF16, {R, (nmel / 4) * c1});
        b_x      = decl_b(b, nm.f("a.x"),      RAD_BF16, {R, H});
        b_h      = decl_b(b, nm.f("a.h"),      RAD_BF16, {R, H});
        b_hc     = decl_b(b, nm.f("a.hc"),     RAD_BF16, {R, H});
        b_f      = decl_b(b, nm.f("a.f"),      RAD_BF16, {R, ffw});
        b_q      = decl_b(b, nm.f("a.q"),      RAD_BF16, {R, H});
        b_k      = decl_b(b, nm.f("a.k"),      RAD_BF16, {R, H});
        b_v      = decl_b(b, nm.f("a.v"),      RAD_BF16, {R, H});
        b_att    = decl_b(b, nm.f("a.att"),    RAD_BF16, {R, H});
        b_o      = decl_b(b, nm.f("a.o"),      RAD_BF16, {R, H});
        b_g      = decl_b(b, nm.f("a.g"),      RAD_BF16, {R, 2 * H});
        b_u      = decl_b(b, nm.f("a.u"),      RAD_BF16, {R, H});
        b_relpos = decl_b(b, nm.f("a.relpos"), RAD_BF16, {n_rel, H});
        b_relk   = decl_b(b, nm.f("a.relk"),   RAD_BF16, {n_rel, H});
        b_o2     = decl_b(b, nm.f("a.o2"),     RAD_BF16, {R, out_dims});
        b_on     = decl_b(b, nm.f("a.on"),     RAD_BF16, {R, out_dims});
        b_out    = decl_b(b, nm.f("a.out"),    RAD_BF16, {R, n_embd});

        auto gemm = [&](int64_t rows, int64_t Nn, int64_t K, rad_weight wt) {
            return RAD_OP(b, "gemm_nt", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", Nn),
                                                   RAD_INT("K", K), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(wt));
        };
        auto rms = [&](int64_t rows, int64_t n, rad_weight wt, double mult) {
            if (!wt)
                return RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                       RAD_F64("eps", eps), RAD_F64("wadd", 0.0),
                                                       RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS);
            return RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                   RAD_F64("eps", eps), RAD_F64("wadd", 0.0),
                                                   RAD_STR("dtype", "bf16"), RAD_F64("mult", mult)),
                          RAD_WEIGHTS(wt));
        };
        auto clamp = [&](int64_t rows, int64_t n, rad_weight lo, rad_weight hi) {
            return RAD_OP(b, "clamp", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                 RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(lo, hi));
        };
        auto unary = [&](const char* op, int64_t rows, int64_t n) {
            return RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                            RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS);
        };
        auto add = [&](int64_t rows, int64_t n) { return unary("add", rows, n); };
        auto lnorm = [&](int64_t rows, int64_t n, rad_weight g) {
            return RAD_OP(b, "layernorm", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                     RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(g));
        };
        auto conv = [&](int64_t rows, int64_t width, int64_t ci, int64_t co, int64_t cu_div,
                        int64_t vstride, rad_weight wt) {
            return RAD_OP(b, "conv2d", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("width", width),
                                                  RAD_INT("cin", ci), RAD_INT("cout", co),
                                                  RAD_INT("kh", 3), RAD_INT("kw", 3),
                                                  RAD_INT("stride", 2), RAD_INT("pad", 1),
                                                  RAD_INT("cu_div", cu_div),
                                                  RAD_INT("valid_stride", vstride),
                                                  RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(wt));
        };
        /* A clipped linear over rows of `in` into `out`: the input clamp writes a copy (the input
         * feeds several projections, each with its own bounds), the output clamp is in place. The
         * copy is a whole [R, n_in] buffer, since a GEMM's input must be contiguous. */
        auto clipped = [&](Clip& c, int64_t n_out, int64_t n_in, rad_buf in, rad_buf tmp,
                           rad_buf out) {
            c.op_in  = rw(b, clamp(R, n_in, c.in_lo, c.in_hi), {in}, {tmp});
            c.op_mm  = rw(b, gemm(R, n_out, n_in, c.w), {tmp}, {out});
            c.op_out = rw(b, clamp(R, n_out, c.out_lo, c.out_hi), {out}, {out});
        };

        op_c0 = rw(b, conv(P, nmel, 1, c0, 1, 1, w_c0), {}, {b_s0});
        op_n0 = rw(b, lnorm(P / 2 * (nmel / 2), c0, w_n0), {b_s0}, {b_s0});
        op_r0 = rw(b, unary("relu", P / 2 * (nmel / 2), c0), {b_s0}, {b_s0});
        op_c1 = rw(b, conv(P / 2, nmel / 2, c0, c1, 2, 2, w_c1), {b_s0}, {b_s1});
        op_n1 = rw(b, lnorm(R * (nmel / 4), c1, w_n1), {b_s1}, {b_s1});
        op_r1 = rw(b, unary("relu", R * (nmel / 4), c1), {b_s1}, {b_s1});
        op_in = rw(b, gemm(R, H, H, w_in), {b_s1}, {b_x});
        op_relpos = rw(b, RAD_OP(b, "rel_sinusoid",
                                 RAD_PARAMS(RAD_RANGE("M", 1, n_rel), RAD_INT("n", H),
                                            RAD_F64("max_timescale", 10000.0),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       {}, {b_relpos});
        for (Layer& l : L) {
            auto ffw_block = [&](FF& ff) {
                ff.op_pre = rw(b, rms(R, H, ff.pre, 1.0), {b_x}, {b_h});
                clipped(ff.l1, ffw, H, b_h, b_hc, b_f);
                ff.op_act = rw(b, unary("silu", R, ffw), {b_f}, {b_f});
                /* the activation's only reader is this linear, so its clamp is in place */
                clipped(ff.l2, H, ffw, b_f, b_f, b_o);
                ff.op_post = rw(b, rms(R, H, ff.post, 0.5), {b_o}, {b_o});
                ff.op_add  = rw(b, add(R, H), {b_x, b_o}, {b_x});
            };
            ffw_block(l.ff1);
            l.op_npa = rw(b, rms(R, H, l.n_pre_attn, 1.0), {b_x}, {b_h});
            clipped(l.q, H, H, b_h, b_hc, b_q);
            clipped(l.k, H, H, b_h, b_hc, b_k);
            clipped(l.v, H, H, b_h, b_hc, b_v);
            l.op_rel  = rw(b, gemm(n_rel, H, H, l.w_rel), {b_relpos}, {b_relk});
            l.op_attn = rw(b, RAD_OP(b, "attn_local_rel",
                                     RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("head_dim", hd),
                                                RAD_INT("n_head", heads), RAD_INT("left", left),
                                                RAD_INT("right", 0), RAD_F64("softcap", softcap),
                                                RAD_INT("cu_div", merge), RAD_STR("dtype", "bf16")),
                                     RAD_WEIGHTS(l.pds)),
                           {b_q, b_k, b_v, b_relk}, {b_att});
            clipped(l.post, H, H, b_att, b_hc, b_o);
            l.op_npo = rw(b, rms(R, H, l.n_post_attn, 1.0), {b_o}, {b_o});
            l.op_add_attn = rw(b, add(R, H), {b_x, b_o}, {b_x});
            l.op_lc_pre = rw(b, rms(R, H, l.lc_pre, 1.0), {b_x}, {b_h});
            clipped(l.lc_start, 2 * H, H, b_h, b_hc, b_g);
            /* GLU: the first half times the sigmoid of the second. */
            l.op_glu = rw(b, RAD_OP(b, "act_mul",
                                    RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", H),
                                               RAD_STR("act", "sigmoid"), RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS),
                          {b_g}, {b_u});
            l.op_dw = rw(b, RAD_OP(b, "conv1d_dw",
                                   RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", H),
                                              RAD_INT("k", kconv), RAD_INT("cu_div", merge),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_WEIGHTS(l.lc_dw)),
                         {b_u}, {b_o});
            l.op_lc_norm = rw(b, rms(R, H, l.lc_norm, 1.0), {b_o}, {b_o});
            l.op_lc_act  = rw(b, unary("silu", R, H), {b_o}, {b_o});
            clipped(l.lc_end, H, H, b_o, b_hc, b_u);
            l.op_lc_add = rw(b, add(R, H), {b_x, b_u}, {b_x});
            ffw_block(l.ff2);
            l.op_out = rw(b, rms(R, H, l.n_out, 1.0), {b_x}, {b_x});
        }
        op_outp = rw(b, RAD_OP(b, "gemm_nt_bias",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("N", out_dims),
                                          RAD_INT("K", H), RAD_STR("dtype", "bf16"),
                                          RAD_STR("act", "none")),
                               RAD_WEIGHTS(w_out, b_outw)),
                     {b_x}, {b_o2});
        op_en  = rw(b, rms(R, out_dims, 0, 1.0), {b_o2}, {b_on});
        op_emb = rw(b, gemm(R, n_embd, out_dims, w_emb), {b_on}, {b_out});

        RadEncoderDecl e{};
        e.name        = "audio";
        e.modalities  = RAD_MM_AUDIO;
        e.patch_dim   = nmel;
        e.merge       = merge;
        e.max_patches = P;
        e.n_embd      = n_embd;
        e.out         = b_out;
        RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        rad_note(b, "audio encoder: %lld conformer blocks of %lld (%lld heads x %lld), "
                    "subsampling %lld/%lld channels, local attention %lld back; %lld frames a "
                    "pass -> %lld rows of %lld", (long long)depth, (long long)H, (long long)heads,
                 (long long)hd, (long long)c0, (long long)c1, (long long)left, (long long)P,
                 (long long)R, (long long)n_embd);
        on = true;
        return RAD_OK;
    }

    void step(RadCtx* c, const RadBatch* batch) const {
        const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg;
        if (!on || Pn <= 0 || Pn > P || Pn % merge || S <= 0) {
            rad_step_fail(c, "an audio pass this program's audio encoder cannot serve: none "
                             "declared, more frames than it was declared for, or not whole rows");
            return;
        }
        const int64_t Rn = Pn / merge;
        const int32_t* crd = batch->enc_coord;
        const RadOperand frames = praw2(batch->enc_pixels, RAD_BF16, Pn, nmel);
        const RadOperand valid = praw(crd + Pn, RAD_I32, Pn);
        const RadOperand cu = praw(batch->enc_cu, RAD_I32, S + 1);
        auto clipped = [&](const Clip& cl, rad_buf in, int64_t n_in, rad_buf tmp, rad_buf out,
                           int64_t n_out) {
            RAD_ISSUE_N(c, cl.op_in, Rn, bcol(in, 0, n_in, Rn), RAD_W(cl.in_lo), RAD_W(cl.in_hi),
                        bcol(tmp, 0, n_in, Rn));
            RAD_ISSUE_N(c, cl.op_mm, Rn, bcol(tmp, 0, n_in, Rn), RAD_W(cl.w), bcol(out, 0, n_out, Rn));
            RAD_ISSUE_N(c, cl.op_out, Rn, bcol(out, 0, n_out, Rn), RAD_W(cl.out_lo), RAD_W(cl.out_hi),
                        bcol(out, 0, n_out, Rn));
        };

        RAD_ISSUE_N(c, op_c0, Pn, frames, RAD_W(w_c0), cu, valid, brows(b_s0, Pn / 2), RAD_NONE);
        RAD_ISSUE_N(c, op_n0, Pn / 2 * (nmel / 2), brows(b_s0, Pn / 2), RAD_W(w_n0), RAD_NONE,
                    brows(b_s0, Pn / 2));
        RAD_ISSUE_N(c, op_r0, Pn / 2 * (nmel / 2), brows(b_s0, Pn / 2), brows(b_s0, Pn / 2));
        RAD_ISSUE_N(c, op_c1, Pn / 2, brows(b_s0, Pn / 2), RAD_W(w_c1), cu, valid, brows(b_s1, Rn),
                    RAD_NONE);
        RAD_ISSUE_N(c, op_n1, Rn * (nmel / 4), brows(b_s1, Rn), RAD_W(w_n1), RAD_NONE,
                    brows(b_s1, Rn));
        RAD_ISSUE_N(c, op_r1, Rn * (nmel / 4), brows(b_s1, Rn), brows(b_s1, Rn));
        RAD_ISSUE_N(c, op_in, Rn, brows(b_s1, Rn), RAD_W(w_in), brows(b_x, Rn));
        RAD_ISSUE_N(c, op_relpos, n_rel, brows(b_relpos, n_rel));
        for (const Layer& l : L) {
            auto ffw_block = [&](const FF& ff) {
                RAD_ISSUE_N(c, ff.op_pre, Rn, brows(b_x, Rn), RAD_W(ff.pre), brows(b_h, Rn));
                clipped(ff.l1, b_h, H, b_hc, b_f, ffw);
                RAD_ISSUE_N(c, ff.op_act, Rn, brows(b_f, Rn), brows(b_f, Rn));
                clipped(ff.l2, b_f, ffw, b_f, b_o, H);
                RAD_ISSUE_N(c, ff.op_post, Rn, brows(b_o, Rn), RAD_W(ff.post), brows(b_o, Rn));
                RAD_ISSUE_N(c, ff.op_add, Rn, brows(b_x, Rn), brows(b_o, Rn), brows(b_x, Rn));
            };
            ffw_block(l.ff1);
            RAD_ISSUE_N(c, l.op_npa, Rn, brows(b_x, Rn), RAD_W(l.n_pre_attn), brows(b_h, Rn));
            clipped(l.q, b_h, H, b_hc, b_q, H);
            clipped(l.k, b_h, H, b_hc, b_k, H);
            clipped(l.v, b_h, H, b_hc, b_v, H);
            RAD_ISSUE_N(c, l.op_rel, n_rel, brows(b_relpos, n_rel), RAD_W(l.w_rel),
                        brows(b_relk, n_rel));
            RAD_ISSUE_N(c, l.op_attn, Rn, brows(b_q, Rn), brows(b_k, Rn), brows(b_v, Rn),
                        brows(b_relk, n_rel), RAD_W(l.pds), cu, brows(b_att, Rn));
            clipped(l.post, b_att, H, b_hc, b_o, H);
            RAD_ISSUE_N(c, l.op_npo, Rn, brows(b_o, Rn), RAD_W(l.n_post_attn), brows(b_o, Rn));
            RAD_ISSUE_N(c, l.op_add_attn, Rn, brows(b_x, Rn), brows(b_o, Rn), brows(b_x, Rn));
            RAD_ISSUE_N(c, l.op_lc_pre, Rn, brows(b_x, Rn), RAD_W(l.lc_pre), brows(b_h, Rn));
            clipped(l.lc_start, b_h, H, b_hc, b_g, 2 * H);
            RAD_ISSUE_N(c, l.op_glu, Rn, bcol(b_g, H, H, Rn), bcol(b_g, 0, H, Rn), brows(b_u, Rn));
            RAD_ISSUE_N(c, l.op_dw, Rn, brows(b_u, Rn), RAD_W(l.lc_dw), cu, brows(b_o, Rn),
                        RAD_NONE);
            RAD_ISSUE_N(c, l.op_lc_norm, Rn, brows(b_o, Rn), RAD_W(l.lc_norm), brows(b_o, Rn));
            RAD_ISSUE_N(c, l.op_lc_act, Rn, brows(b_o, Rn), brows(b_o, Rn));
            clipped(l.lc_end, b_o, H, b_hc, b_u, H);
            RAD_ISSUE_N(c, l.op_lc_add, Rn, brows(b_x, Rn), brows(b_u, Rn), brows(b_x, Rn));
            ffw_block(l.ff2);
            RAD_ISSUE_N(c, l.op_out, Rn, brows(b_x, Rn), RAD_W(l.n_out), brows(b_x, Rn));
        }
        RAD_ISSUE_N(c, op_outp, Rn, brows(b_x, Rn), RAD_W(w_out), RAD_W(b_outw), RAD_NONE,
                    brows(b_o2, Rn));
        RAD_ISSUE_N(c, op_en, Rn, brows(b_o2, Rn), RAD_NONE, brows(b_on, Rn));
        RAD_ISSUE_N(c, op_emb, Rn, brows(b_on, Rn), RAD_W(w_emb), brows(b_out, Rn));
    }
};

}  /* namespace arch */
}  /* namespace rad */

#endif
