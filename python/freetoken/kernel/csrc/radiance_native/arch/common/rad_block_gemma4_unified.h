/* rad_block_gemma4_unified.h -- the media embedders of Gemma 4 12B ("unified", transformers'
 * Gemma4UnifiedVisionEmbedder and the audio Gemma4UnifiedMultimodalEmbedder), served by gemma4_bf16.
 * Neither has an encoder: the language model sees each patch or audio frame after a few norms and
 * linears.
 *
 *   image: a patch is 48 x 48 x 3 (3 x 3 of the processor's 16-pixel patches), staged by the core's
 *   Gemma 4 processor as u / 255 in bf16, (y, x, channel); a LayerNorm over the patch, a linear with
 *   a bias, a LayerNorm, plus a learned row for the patch's column and one for its row (the
 *   checkpoint's [npos, 2, width] table, read as its two taps of dimension 1), a LayerNorm, then the
 *   multimodal embedder: an RMS norm with no gain and a width -> n_embd linear. The LayerNorms are
 *   nn.LayerNorm's default eps 1e-5, which no config states. One patch is one row.
 *   audio: a frame is 640 samples at 16 kHz as they are, the last zero-padded, in bf16; an RMS norm
 *   with no gain and a 640 -> n_embd linear. One frame is one row.
 *
 * Their rows replace the placeholders' embeddings unscaled, as the vision tower's do. Every rank
 * runs them whole (RAD_SHARD_NONE). */
#ifndef RAD_BLOCK_GEMMA4_UNIFIED_H
#define RAD_BLOCK_GEMMA4_UNIFIED_H

#include "rad_arch.h"

#include <cstdio>
#include <cstring>

namespace rad {
namespace arch {

struct Gemma4UnifiedVision {
    bool on = false;
    /* The language model reads a picture's rows both ways (RadEncoderDecl::bidirectional); the
     * plugin sets it before declare. */
    bool bidirectional = false;
    int64_t P = 0, width = 0, patch_dim = 0, npos = 0, n_embd = 0;
    rad_weight w_ln1 = 0, w_ln1b = 0, w_in = 0, w_inb = 0, w_ln2 = 0, w_ln2b = 0, w_px = 0, w_py = 0;
    rad_weight w_pn = 0, w_pnb = 0, w_proj = 0;
    rad_op op_ln1 = 0, op_in = 0, op_ln2 = 0, op_px = 0, op_py = 0, op_padd = 0, op_add = 0;
    rad_op op_pn = 0, op_norm = 0, op_out = 0;
    rad_buf b_n = 0, b_x = 0, b_h = 0, b_px = 0, b_py = 0, b_out = 0;

    static bool present(const RadModelMeta* m) {
        const char* t = rad_meta_gets(m, "vision_config.model_type", "");
        return t && !std::strcmp(t, "gemma4_unified_vision");
    }

    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, const char* who) {
        on = false;
        if (!present(m) || ctx->max_enc_patches <= 0) return RAD_OK;
        width = rad_meta_geti(m, "vision_config.mm_embed_dim", 0);
        npos  = rad_meta_geti(m, "vision_config.mm_posemb_size", 0);
        const int64_t side = rad_meta_geti(m, "vision_config.model_patch_size", 0);
        const int64_t outd = rad_meta_geti(m, "vision_config.output_proj_dims", width);
        const double  eps  = rad_meta_getf(m, "vision_config.rms_norm_eps", 1e-6);
        if (width <= 0 || npos <= 0 || side <= 0 || outd != width) {
            fprintf(stderr, "radiance: %s: the container's vision_config is incomplete or not the "
                            "embedder this plugin serves (mm_embed_dim %lld, mm_posemb_size %lld, "
                            "model_patch_size %lld, output_proj_dims %lld)\n", who,
                    (long long)width, (long long)npos, (long long)side, (long long)outd);
            return RAD_E_UNSUPPORTED;
        }
        n_embd = model_embd;
        patch_dim = 3 * side * side;
        P = ctx->max_enc_patches;
        const int A = RAD_ACCESS_PER_REQUEST, N = RAD_SHARD_NONE;
        const char* V = "model.vision_embedder";
        struct { const char* to; const char* from; } cp[] = {
            { "v.ln1.weight", "patch_ln1.weight" }, { "v.ln1.bias", "patch_ln1.bias" },
            { "v.in.weight", "patch_dense.weight" }, { "v.in.bias", "patch_dense.bias" },
            { "v.ln2.weight", "patch_ln2.weight" }, { "v.ln2.bias", "patch_ln2.bias" },
            { "v.pn.weight", "pos_norm.weight" }, { "v.pn.bias", "pos_norm.bias" } };
        for (const auto& e : cp)
            RAD_ARCH_TRY(map_copy(b, nm.f("%s", e.to), nm.ckpt("%s.%s", V, e.from)));
        RAD_ARCH_TRY(map_tap(b, nm.f("v.pos_x.weight"), nm.ckpt("%s.pos_embedding", V), 1, 0));
        RAD_ARCH_TRY(map_tap(b, nm.f("v.pos_y.weight"), nm.ckpt("%s.pos_embedding", V), 1, 1));
        RAD_ARCH_TRY(map_copy(b, nm.f("v.embed.weight"), nm.ckpt("model.embed_vision.embedding_projection.weight")));
        w_ln1  = decl_w(b, nm.f("v.ln1.weight"), RAD_BF16, {patch_dim}, A, N, grp_model());
        w_ln1b  = decl_w(b, nm.f("v.ln1.bias"), RAD_BF16, {patch_dim}, A, N, grp_model());
        w_in   = decl_w(b, nm.f("v.in.weight"), RAD_BF16, {width, patch_dim}, A, N, grp_model());
        w_inb   = decl_w(b, nm.f("v.in.bias"), RAD_BF16, {width}, A, N, grp_model());
        w_ln2  = decl_w(b, nm.f("v.ln2.weight"), RAD_BF16, {width}, A, N, grp_model());
        w_ln2b  = decl_w(b, nm.f("v.ln2.bias"), RAD_BF16, {width}, A, N, grp_model());
        w_px   = decl_w(b, nm.f("v.pos_x.weight"), RAD_BF16, {npos, width}, A, N, grp_model());
        w_py   = decl_w(b, nm.f("v.pos_y.weight"), RAD_BF16, {npos, width}, A, N, grp_model());
        w_pn   = decl_w(b, nm.f("v.pn.weight"), RAD_BF16, {width}, A, N, grp_model());
        w_pnb   = decl_w(b, nm.f("v.pn.bias"), RAD_BF16, {width}, A, N, grp_model());
        w_proj = decl_w(b, nm.f("v.embed.weight"), RAD_BF16, {n_embd, width}, A, N, grp_model());

        b_n   = decl_b(b, nm.f("v.n"),   RAD_BF16, {P, patch_dim});
        b_x   = decl_b(b, nm.f("v.x"),   RAD_BF16, {P, width});
        b_h   = decl_b(b, nm.f("v.h"),   RAD_BF16, {P, width});
        b_px  = decl_b(b, nm.f("v.px"),  RAD_BF16, {P, width});
        b_py  = decl_b(b, nm.f("v.py"),  RAD_BF16, {P, width});
        b_out = decl_b(b, nm.f("v.out"), RAD_BF16, {P, n_embd});

        auto ln = [&](int64_t n, rad_weight w, rad_weight bias) {
            return RAD_OP(b, "layernorm", RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", n),
                                                     RAD_F64("eps", 1e-5), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w, bias));
        };
        auto lookup = [&](rad_weight w) {
            return RAD_OP(b, "embed_lookup",
                          RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n_embd", width),
                                     RAD_INT("n_vocab", npos), RAD_STR("dtype", "bf16"),
                                     RAD_INT("vocab_offset", 0), RAD_F64("mult", 1.0)),
                          RAD_WEIGHTS(w));
        };
        auto add = [&]() {
            return RAD_OP(b, "add", RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                               RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS);
        };
        op_ln1  = rw(b, ln(patch_dim, w_ln1, w_ln1b), {}, {b_n});
        op_in   = rw(b, RAD_OP(b, "gemm_nt_bias",
                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("N", width),
                                          RAD_INT("K", patch_dim), RAD_STR("dtype", "bf16"),
                                          RAD_STR("act", "none")),
                               RAD_WEIGHTS(w_in, w_inb)),
                     {b_n}, {b_x});
        op_ln2  = rw(b, ln(width, w_ln2, w_ln2b), {b_x}, {b_h});
        op_px   = rw(b, lookup(w_px), {}, {b_px});
        op_py   = rw(b, lookup(w_py), {}, {b_py});
        op_padd = rw(b, add(), {b_px, b_py}, {b_px});
        op_add  = rw(b, add(), {b_h, b_px}, {b_h});
        op_pn   = rw(b, ln(width, w_pn, w_pnb), {b_h}, {b_x});
        op_norm = rw(b, RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                          RAD_F64("eps", eps), RAD_F64("wadd", 0.0),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     {b_x}, {b_h});
        op_out  = rw(b, RAD_OP(b, "gemm_nt",
                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("N", n_embd),
                                          RAD_INT("K", width), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(w_proj)),
                     {b_h}, {b_out});

        RadEncoderDecl e{};
        e.name        = "vision";
        e.modalities  = RAD_MM_IMAGE | RAD_MM_VIDEO;
        e.patch_dim   = patch_dim;
        e.merge       = 1;
        e.max_patches = P;
        e.n_embd      = n_embd;
        e.out         = b_out;
        e.bidirectional = bidirectional ? 1 : 0;
        RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        rad_note(b, "vision embedder (no tower): %lldx%lld patches of %lld values -> %lld, "
                    "positions %lld a side; %lld patches a pass -> %lld rows of %lld",
                 (long long)side, (long long)side, (long long)patch_dim, (long long)width,
                 (long long)npos, (long long)P, (long long)P, (long long)n_embd);
        on = true;
        return RAD_OK;
    }

    void step(RadCtx* c, const RadBatch* batch) const {
        const int64_t Pn = batch->enc_n_patch;
        if (!on || Pn <= 0 || Pn > P || batch->enc_n_seg <= 0) {
            rad_step_fail(c, "an encoder pass this program's vision embedder cannot serve: none "
                             "declared, or more patches than it was declared for");
            return;
        }
        const int32_t* crd = batch->enc_coord;
        RAD_ISSUE_N(c, op_ln1, Pn, praw2(batch->enc_pixels, RAD_BF16, Pn, patch_dim), RAD_W(w_ln1),
                    RAD_W(w_ln1b), brows(b_n, Pn));
        RAD_ISSUE_N(c, op_in, Pn, brows(b_n, Pn), RAD_W(w_in), RAD_W(w_inb), RAD_NONE, brows(b_x, Pn));
        RAD_ISSUE_N(c, op_ln2, Pn, brows(b_x, Pn), RAD_W(w_ln2), RAD_W(w_ln2b), brows(b_h, Pn));
        RAD_ISSUE_N(c, op_px, Pn, praw(crd + Pn, RAD_I32, Pn), RAD_W(w_px), brows(b_px, Pn));
        RAD_ISSUE_N(c, op_py, Pn, praw(crd, RAD_I32, Pn), RAD_W(w_py), brows(b_py, Pn));
        RAD_ISSUE_N(c, op_padd, Pn, brows(b_px, Pn), brows(b_py, Pn), brows(b_px, Pn));
        RAD_ISSUE_N(c, op_add, Pn, brows(b_h, Pn), brows(b_px, Pn), brows(b_h, Pn));
        RAD_ISSUE_N(c, op_pn, Pn, brows(b_h, Pn), RAD_W(w_pn), RAD_W(w_pnb), brows(b_x, Pn));
        RAD_ISSUE_N(c, op_norm, Pn, brows(b_x, Pn), RAD_NONE, brows(b_h, Pn));
        RAD_ISSUE_N(c, op_out, Pn, brows(b_h, Pn), RAD_W(w_proj), brows(b_out, Pn));
    }
};

struct Gemma4UnifiedAudio {
    bool on = false;
    int64_t P = 0, frame = 0, n_embd = 0;
    rad_weight w_proj = 0;
    rad_op op_norm = 0, op_out = 0;
    rad_buf b_n = 0, b_out = 0;

    static bool present(const RadModelMeta* m) {
        const char* t = rad_meta_gets(m, "audio_config.model_type", "");
        return t && !std::strcmp(t, "gemma4_unified_audio");
    }
    /* Samples a frame: the extractor's audio_samples_per_token, else the embedder's input width. */
    static int64_t frame_for(const RadModelMeta* m) {
        return rad_meta_geti(m, "processor.feature_extractor.audio_samples_per_token",
                             rad_meta_geti(m, "audio_config.audio_embed_dim", 0));
    }
    /* The longest clip in frames: processor_config.json's audio_seq_length, else 30 s. */
    static int64_t frames_for(const RadModelMeta* m) {
        const int64_t f = frame_for(m);
        const int64_t rate = rad_meta_geti(m, "processor.feature_extractor.sampling_rate", 16000);
        if (f <= 0 || rate <= 0) return 0;
        return rad_meta_geti(m, "processor.audio_seq_length", (30 * rate + f - 1) / f);
    }

    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, const char* who) {
        on = false;
        if (!present(m) || ctx->max_audio_frames <= 0) return RAD_OK;
        frame = frame_for(m);
        const int64_t width = rad_meta_geti(m, "audio_config.audio_embed_dim", 0);
        const double  eps   = rad_meta_getf(m, "audio_config.rms_norm_eps", 1e-6);
        if (frame <= 0 || width != frame) {
            fprintf(stderr, "radiance: %s: the container's audio frame (%lld samples) is not the "
                            "embedder's input (audio_embed_dim %lld)\n", who, (long long)frame,
                    (long long)width);
            return RAD_E_UNSUPPORTED;
        }
        n_embd = model_embd;
        P = ctx->max_audio_frames;
        RAD_ARCH_TRY(map_copy(b, nm.f("a.embed.weight"), nm.ckpt("model.embed_audio.embedding_projection.weight")));
        w_proj = decl_w(b, nm.f("a.embed.weight"), RAD_BF16, {n_embd, frame}, RAD_ACCESS_PER_REQUEST,
                        RAD_SHARD_NONE, grp_model());
        b_n   = decl_b(b, nm.f("a.n"),   RAD_BF16, {P, frame});
        b_out = decl_b(b, nm.f("a.out"), RAD_BF16, {P, n_embd});
        op_norm = rw(b, RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", frame),
                                          RAD_F64("eps", eps), RAD_F64("wadd", 0.0),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     {}, {b_n});
        op_out  = rw(b, RAD_OP(b, "gemm_nt",
                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("N", n_embd),
                                          RAD_INT("K", frame), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(w_proj)),
                     {b_n}, {b_out});
        RadEncoderDecl e{};
        e.name        = "audio";
        e.modalities  = RAD_MM_AUDIO;
        e.patch_dim   = frame;
        e.merge       = 1;
        e.max_patches = P;
        e.n_embd      = n_embd;
        e.out         = b_out;
        RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        rad_note(b, "audio embedder (no encoder): frames of %lld samples -> %lld; %lld frames a pass",
                 (long long)frame, (long long)n_embd, (long long)P);
        on = true;
        return RAD_OK;
    }

    void step(RadCtx* c, const RadBatch* batch) const {
        const int64_t Pn = batch->enc_n_patch;
        if (!on || Pn <= 0 || Pn > P || batch->enc_n_seg <= 0) {
            rad_step_fail(c, "an audio pass this program's audio embedder cannot serve: none "
                             "declared, or more frames than it was declared for");
            return;
        }
        RAD_ISSUE_N(c, op_norm, Pn, praw2(batch->enc_pixels, RAD_BF16, Pn, frame), RAD_NONE,
                    brows(b_n, Pn));
        RAD_ISSUE_N(c, op_out, Pn, brows(b_n, Pn), RAD_W(w_proj), brows(b_out, Pn));
    }
};

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_GEMMA4_UNIFIED_H */
