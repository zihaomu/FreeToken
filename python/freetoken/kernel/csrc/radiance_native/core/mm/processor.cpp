/* processor.cpp -- see processor.h. */
#include "mm/processor.h"
#include "util/rowpool.h"
#include "util/simd.h"
#include "mm/imaging.h"

#include "rad_internal.h"
#include "rad_plugin.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <sstream>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#include <immintrin.h>
#define RAD_MM_HAVE_AVX2 1
#endif

namespace rad {
namespace mm {

/* ================================================================== configuration */

static bool read_triple(const RadModelMeta& m, const char* key, float out[3]) {
    const char* v = rad_meta_gets(&m, key, nullptr);
    if (!v || !*v) return false;
    std::istringstream is(v);
    float t[3];
    for (int i = 0; i < 3; ++i)
        if (!(is >> t[i])) return false;
    for (int i = 0; i < 3; ++i) out[i] = t[i];
    return true;
}

static bool soft_budget_ok(int32_t n) {
    return n == 70 || n == 140 || n == 280 || n == 560 || n == 1120;
}

/* A flag a container flattened from JSON: `true`/`false`, or a number. */
static bool meta_bool(const RadModelMeta& m, const char* key, bool dflt) {
    const char* v = rad_meta_gets(&m, key, nullptr);
    if (!v || !*v) return dflt;
    if (!std::strcmp(v, "true")) return true;
    if (!std::strcmp(v, "false")) return false;
    return std::strtol(v, nullptr, 10) != 0;
}

/* Muse Glimmer's shipped processor config (processor.image_mean, processor.image_std), over the
 * reference class's IMAGENET_STANDARD defaults. */
static const float kMuseDefaultMean[3] = { 0.5f, 0.5f, 0.5f };
static const float kMuseDefaultStd[3]  = { 0.5f, 0.5f, 0.5f };

int VisionConfig::from_meta(const RadModelMeta& m, std::string* why) {
    /* Only the requested noncausal Mistral3/Ministral3 wrapper selects this new path. */
    if (!std::strcmp(rad_meta_gets(&m, "model_type", ""), "mistral3") &&
        !std::strcmp(rad_meta_gets(&m, "text_config.model_type", ""), "ministral3") &&
        !std::strcmp(rad_meta_gets(&m, "vision_config.model_type", ""), "pixtral") &&
        !meta_bool(m, "is_causal", true) && !meta_bool(m, "text_config.is_causal", true)) {
        family = Family::Pixtral;
        patch = (int32_t)rad_meta_geti(&m, "vision_config.patch_size", 0);
        merge = (int32_t)rad_meta_geti(&m, "spatial_merge_size", 0);
        temporal = 1;
        if (patch != 14 || merge != 2) {
            if (why) *why = "Nemotron 3.5's Pixtral processor requires patch 14 and merge 2";
            return RAD_E_UNSUPPORTED;
        }
        pixtral_longest_edge =
            (int32_t)rad_meta_geti(&m, "processor.image_processor.size.longest_edge", 1120);
        tok_image = (int32_t)rad_meta_geti(&m, "image_token_index", -1);
        image_token = rad_meta_gets(&m, "processor.image_token", "[IMG]");
        image_break_token = rad_meta_gets(&m, "processor.image_break_token", "[IMG_BREAK]");
        image_end_token = rad_meta_gets(&m, "processor.image_end_token", "[IMG_END]");
        const char* root = "processor.image_processor.";
        for (const char* flag : {"do_resize", "do_convert_rgb", "do_rescale", "do_normalize"}) {
            if (!meta_bool(m, (std::string(root) + flag).c_str(), true)) {
                if (why)
                    *why = "Pixtral requires resize, RGB conversion, rescaling and normalization";
                return RAD_E_UNSUPPORTED;
            }
        }
        if (rad_meta_geti(&m, "processor.image_processor.resample", 3) != 3 ||
            rad_meta_getf(&m, "processor.image_processor.rescale_factor", 1.0 / 255) != 1.0 / 255) {
            if (why) *why = "Pixtral supports bicubic RGB rescaling by 1/255";
            return RAD_E_UNSUPPORTED;
        }
        const float clip_mean[] = {.48145466f, .4578275f, .40821073f};
        const float clip_std[] = {.26862954f, .26130258f, .27577711f};
        std::copy(clip_mean, clip_mean + 3, mean);
        std::copy(clip_std, clip_std + 3, stdv);
        read_triple(m, "processor.image_processor.image_mean", mean);
        read_triple(m, "processor.image_processor.image_std", stdv);
        image_min_pixels = video_min_pixels = (int64_t)factor() * factor();
        image_max_pixels = video_max_pixels = (int64_t)pixtral_longest_edge * pixtral_longest_edge;
        return check(why);
    }
    /* Unlimited-OCR: one processor class, two modes, both stated in preprocessor_config.json and
     * processor_config.json (`processor.*`). Every key has the reference's default, so a container
     * converted before the tower landed still states a processor. */
    if (!std::strcmp(rad_meta_gets(&m, "model_type", ""), "unlimited_ocr")) {
        family = Family::UnlimitedOcr;
        patch      = (int32_t)rad_meta_geti(&m, "processor.patch_size",
                                            rad_meta_geti(&m, "vision_config.patch_size", 16));
        downsample = (int32_t)rad_meta_geti(&m, "processor.downsample_ratio", 4);
        merge      = downsample;
        temporal   = 1;
        base_size  = (int32_t)rad_meta_geti(&m, "preprocessor.size.height", 1024);
        tile       = (int32_t)rad_meta_geti(&m, "preprocessor.tile_size", 640);
        /* The base mode's whole view: what the reference states for its input, and what the
         * preprocessor's `size` is. Multi-page inference may ask for the tile instead. */
        image_size = (int32_t)rad_meta_geti(&m, "preprocessor.size.width", base_size);
        crop_mode  = meta_bool(m, "preprocessor.crop_to_patches", true);
        if (const char* t = rad_meta_gets(&m, "processor.image_token", nullptr); t && *t)
            image_token = t;
        min_tiles  = (int32_t)rad_meta_geti(&m, "preprocessor.min_patches", 2);
        max_tiles  = (int32_t)rad_meta_geti(&m, "preprocessor.max_patches", 32);
        tok_image  = (int32_t)rad_meta_geti(&m, "image_token_id", -1);
        if (!read_triple(m, "preprocessor.image_mean", mean)) mean[0] = mean[1] = mean[2] = 0.5f;
        if (!read_triple(m, "preprocessor.image_std", stdv)) stdv[0] = stdv[1] = stdv[2] = 0.5f;
        /* 3 is PIL's bicubic, the only filter this processor stages. */
        const int64_t resample = rad_meta_geti(&m, "preprocessor.resample", 3);
        if (resample != 3) {
            if (why) *why = "the container resamples pictures with filter " +
                            std::to_string(resample) + "; this processor stages PIL's bicubic (3)";
            return RAD_E_UNSUPPORTED;
        }
        /* No video; the bands only have to be consistent for check(). */
        const int64_t f2 = (int64_t)factor() * factor();
        image_min_pixels = video_min_pixels = f2;
        image_max_pixels = video_max_pixels = (int64_t)base_size * base_size;
        min_frames       = 1;
        std::string bad;
        if (check(&bad) < 0) {
            if (why) *why = "the container's Unlimited-OCR processor configuration is "
                            "inconsistent: " + bad;
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }
    /* LFM2-VL states its processor twice, in config.json and in processor_config.json's
     * image_processor; the latter is what the reference processor reads, so it goes first. */
    if (!std::strcmp(rad_meta_gets(&m, "model_type", ""), "lfm2_vl")) {
        family = Family::Lfm2;
        auto key = [](const char* k) { return std::string("processor.image_processor.") + k; };
        auto geti = [&](const char* k, int64_t dflt) {
            return rad_meta_geti(&m, key(k).c_str(), rad_meta_geti(&m, k, dflt));
        };
        auto getb = [&](const char* k, bool dflt) {
            return meta_bool(m, key(k).c_str(), meta_bool(m, k, dflt));
        };
        patch    = (int32_t)geti("encoder_patch_size", rad_meta_geti(&m, "vision_config.patch_size", 0));
        merge    = (int32_t)geti("downsample_factor", 0);
        temporal = 1;
        tok_image = (int32_t)rad_meta_geti(&m, "image_token_id", -1);
        if (patch <= 0 || merge <= 0 || tok_image < 0) {
            if (why) *why = "the container states no LFM2-VL processor geometry (encoder_patch_size, "
                            "downsample_factor, image_token_id)";
            return RAD_E_NOTFOUND;
        }
        const int64_t f2 = (int64_t)factor() * factor();
        image_min_pixels = geti("min_image_tokens", 64) * f2;
        image_max_pixels = geti("max_image_tokens", 256) * f2;
        video_min_pixels = image_min_pixels;           /* no video; consistent for check() */
        video_max_pixels = image_max_pixels;
        /* The markers around a picture and before each tile are what the model was trained on;
         * a checkpoint that turns them off spells its pictures differently, which is not served. */
        if (!meta_bool(m, "use_image_special_tokens", true)) {
            if (why) *why = "use_image_special_tokens is off; this processor writes LFM2-VL's "
                            "picture markers and serves only the checkpoints that read them";
            return RAD_E_UNSUPPORTED;
        }
        const bool split = getb("do_image_splitting", true);
        min_tiles = split ? (int32_t)geti("min_tiles", 2) : 1;
        max_tiles = split ? (int32_t)geti("max_tiles", 10) : 1;
        tile      = (int32_t)geti("tile_size", 512);
        use_thumbnail = getb("use_thumbnail", true);
        max_pixels_tolerance = rad_meta_getf(&m, key("max_pixels_tolerance").c_str(),
                                             rad_meta_getf(&m, "max_pixels_tolerance", 2.0));
        max_segment_patches = geti("max_num_patches", 1024);
        if (!read_triple(m, key("image_mean").c_str(), mean)) read_triple(m, "preprocessor.image_mean", mean);
        if (!read_triple(m, key("image_std").c_str(), stdv)) read_triple(m, "preprocessor.image_std", stdv);
        std::string bad;
        if (check(&bad) < 0) {
            if (why) *why = "the container's LFM2-VL processor configuration is inconsistent: " + bad;
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }
    const char* mt = rad_meta_gets(&m, "vision_config.model_type", "");
    /* GLM-5.3 states its processor only in processor_config.json's image_processor, with a token
     * budget and no pixel band; the tower's geometry is the fallback, as the processor class
     * defaults to it. */
    if (mt && !std::strcmp(mt, "glm5_next_vision")) {
        family = Family::Glm5Next;
        auto geti = [&](const char* k, const char* tower, int64_t dflt) {
            const std::string key = std::string("processor.image_processor.") + k;
            return rad_meta_geti(&m, key.c_str(), rad_meta_geti(&m, tower, dflt));
        };
        patch    = (int32_t)geti("patch_size", "vision_config.patch_size", 0);
        temporal = (int32_t)geti("temporal_patch_size", "vision_config.temporal_patch_size", 0);
        merge    = (int32_t)geti("merge_size", "vision_config.spatial_merge_size", 0);
        if (patch <= 0 || temporal <= 0 || merge <= 0) {
            if (why) *why = "the container states no vision tower geometry (vision_config."
                            "patch_size, temporal_patch_size, spatial_merge_size)";
            return RAD_E_NOTFOUND;
        }
        /* The expand factor multiplies the canvas's alignment without changing the patches; the
         * checkpoints ship 1, and another one is a canvas this processor does not make. */
        const int64_t expand =
            rad_meta_geti(&m, "processor.image_processor.patch_expand_factor", 1);
        if (expand != 1) {
            if (why) *why = "processor.image_processor.patch_expand_factor is " +
                            std::to_string(expand) + "; this processor serves 1";
            return RAD_E_UNSUPPORTED;
        }
        tok_image = (int32_t)rad_meta_geti(&m, "image_token_id", -1);
        tok_video = (int32_t)rad_meta_geti(&m, "video_token_id", -1);
        tok_start = (int32_t)rad_meta_geti(&m, "image_start_token_id", -1);
        tok_end   = (int32_t)rad_meta_geti(&m, "image_end_token_id", -1);
        if (tok_image < 0 || tok_start < 0 || tok_end < 0) {
            if (why) *why = "the container states no media token ids (image_token_id, "
                            "image_start_token_id, image_end_token_id)";
            return RAD_E_NOTFOUND;
        }
        min_image_tokens = (int32_t)rad_meta_geti(&m, "processor.image_processor.min_image_tokens",
                                                  16);
        max_image_tokens = (int32_t)rad_meta_geti(&m, "processor.image_processor.max_image_tokens",
                                                  8000);
        /* OpenAI CLIP's, the processor class's default. */
        const float clip_mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };
        const float clip_std[3]  = { 0.26862954f, 0.26130258f, 0.27577711f };
        std::copy(clip_mean, clip_mean + 3, mean);
        std::copy(clip_std, clip_std + 3, stdv);
        read_triple(m, "processor.image_processor.image_mean", mean);
        read_triple(m, "processor.image_processor.image_std", stdv);
        /* The pixel bands are QwenVL's; here they only have to be consistent for check(). */
        const int64_t f2 = (int64_t)factor() * factor();
        image_min_pixels = video_min_pixels = std::max<int64_t>(1, min_image_tokens) * f2;
        image_max_pixels = video_max_pixels =
            std::max<int64_t>(image_min_pixels, (int64_t)max_image_tokens * f2);
        std::string bad;
        if (check(&bad) < 0) {
            if (why)
                *why = "the container's GLM-5.3 processor configuration is inconsistent: " + bad;
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }
    /* NVIDIA's Nemotron-H Omni states its tile budget in the processor configs (preprocessor and
     * processor_config's image_processor carry the same keys) and its token ids in config.json,
     * except <img> and </img>, which the vocabulary names (Processor::init). */
    if (mt && !std::strcmp(mt, "radio")) {
        family = Family::Radio;
        auto geti = [&](const char* k, int64_t dflt) {
            const std::string a = std::string("processor.image_processor.") + k;
            const std::string b = std::string("preprocessor.") + k;
            return rad_meta_geti(&m, a.c_str(), rad_meta_geti(&m, b.c_str(), dflt));
        };
        patch = (int32_t)geti("patch_size", rad_meta_geti(&m, "vision_config.patch_size", 0));
        const double ratio = rad_meta_getf(&m, "downsample_ratio", 0.0);
        merge = ratio > 0.0 ? (int32_t)std::lround(1.0 / ratio) : 0;
        temporal = 1;
        radio_tubelet = (int32_t)rad_meta_geti(
            &m, "processor.video_temporal_patch_dim",
            rad_meta_geti(&m, "video_temporal_patch_size",
                          rad_meta_geti(&m, "vision_config.video_temporal_patch_size", 0)));
        radio_prefix = (int32_t)(rad_meta_geti(&m, "vision_config.num_cls_tokens", 0) +
                                 rad_meta_geti(&m, "vision_config.num_registers", 0));
        tok_image = (int32_t)rad_meta_geti(&m, "img_context_token_id", -1);
        if (patch <= 0 || merge <= 0 || radio_tubelet <= 0 || radio_prefix <= 0 || tok_image < 0) {
            if (why) *why = "the container states no Nemotron-H Omni processor geometry "
                            "(vision_config.patch_size, downsample_ratio, video_temporal_patch_size, "
                            "num_cls_tokens + num_registers, img_context_token_id)";
            return RAD_E_NOTFOUND;
        }
        radio_min_patches   = (int32_t)geti("min_num_patches", 1024);
        radio_max_patches   = (int32_t)geti("max_num_patches", 13312);
        radio_max_len       = (int32_t)geti("max_model_len", 16384);
        radio_video_patches = (int32_t)geti("video_target_num_patches", 1024);
        if (!read_triple(m, "processor.image_processor.norm_mean", mean) &&
            !read_triple(m, "preprocessor.norm_mean", mean))
            read_triple(m, "norm_mean", mean);
        if (!read_triple(m, "processor.image_processor.norm_std", stdv) &&
            !read_triple(m, "preprocessor.norm_std", stdv))
            read_triple(m, "norm_std", stdv);
        /* The model card's serving default: 2 frames a second, at most 128 (64 tubelets). */
        fps        = 2.0;
        min_frames = radio_tubelet;
        max_frames = 128;
        max_frame_rows = 0;
        /* The pixel bands only have to be consistent for check(); the tile rule sizes by patches. */
        const int64_t p2 = (int64_t)patch * patch;
        image_min_pixels = (int64_t)radio_min_patches * p2;
        image_max_pixels = std::max<int64_t>(image_min_pixels, (int64_t)radio_max_patches * p2);
        video_min_pixels = video_max_pixels = (int64_t)radio_video_patches * p2;
        std::string bad;
        if (radio_min_patches <= 0 || radio_video_patches <= 0 || radio_max_len <= 0)
            bad = "min_num_patches, video_target_num_patches and max_model_len must be positive";
        if (bad.empty() && check(&bad) < 0) bad = "the container's Nemotron-H Omni processor "
                                                  "configuration is inconsistent: " + bad;
        if (!bad.empty()) {
            if (why) *why = bad;
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }
    /* Meta's Muse Glimmer 30B: the vision tower states the patch geometry and the processor
     * configs the token caps; the markers are numbered by config.json where they are numbered at
     * all, and the vocabulary spells the rest (resolved at Processor::init). */
    if (mt && !std::strcmp(mt, "muse_glimmer_vision")) {
        family = Family::MuseGlimmer;
        patch    = (int32_t)rad_meta_geti(&m, "vision_config.patch_size", 0);
        temporal = (int32_t)rad_meta_geti(&m, "vision_config.patch_temporal",
                                          rad_meta_geti(&m, "vision_config.temporal_patch_size", 0));
        merge    = (int32_t)rad_meta_geti(&m, "vision_config.merge_size",
                                          rad_meta_geti(&m, "vision_config.spatial_merge_size", 0));
        if (patch <= 0 || temporal <= 0 || merge <= 0) {
            if (why) *why = "the container states no vision tower geometry (vision_config."
                            "patch_size, patch_temporal, merge_size)";
            return RAD_E_NOTFOUND;
        }
        tok_image = (int32_t)rad_meta_geti(&m, "image_token_id", -1);
        tok_video = (int32_t)rad_meta_geti(&m, "video_token_id", -1);
        if (tok_image < 0 || tok_video < 0) {
            if (why) *why = "the container states no media token ids (image_token_id, "
                            "video_token_id)";
            return RAD_E_NOTFOUND;
        }
        /* The markers: a container that numbers them wins; the rest the vocabulary resolves. */
        tok_start        = (int32_t)rad_meta_geti(&m, "image_start_token_id", -1);
        tok_end          = (int32_t)rad_meta_geti(&m, "image_end_token_id", -1);
        tok_video_start  = (int32_t)rad_meta_geti(&m, "video_start_token_id", -1);
        tok_video_end    = (int32_t)rad_meta_geti(&m, "video_end_token_id", -1);
        tok_video_sep    = (int32_t)rad_meta_geti(&m, "video_frame_separator_token_id", -1);
        /* The token caps, in merged tokens: one per picture and one per video frame. */
        max_image_tokens = (int32_t)rad_meta_geti(&m, "processor.image_processor.max_image_tokens",
                                                  4096);
        max_frame_rows   = (int32_t)rad_meta_geti(&m, "processor.video_processor.max_video_frame_tokens",
                                                  144);
        max_frames       = (int32_t)rad_meta_geti(&m, "processor.video_processor.num_frames", 96);
        fps              = rad_meta_getf(&m, "processor.video_processor.fps", 2.0);
        min_frames       = temporal;
        /* The shipped checkpoints (mean and deviation 0.5) over the reference class's
         * IMAGENET_STANDARD defaults: the config wins, as it does for every family. */
        if (!read_triple(m, "processor.image_mean", mean)) std::copy(std::begin(kMuseDefaultMean),
                                                                     std::end(kMuseDefaultMean), mean);
        if (!read_triple(m, "processor.image_std", stdv)) std::copy(std::begin(kMuseDefaultStd),
                                                                    std::end(kMuseDefaultStd), stdv);
        /* The pixel bands only have to be consistent for check(); the caps size by tokens. */
        const int64_t f2 = (int64_t)factor() * factor();
        image_min_pixels = video_min_pixels = f2;
        image_max_pixels = std::max<int64_t>(f2, (int64_t)std::max(1, max_image_tokens) * f2);
        video_max_pixels = std::max<int64_t>(f2, (int64_t)std::max(1, max_frame_rows) * f2);
        std::string bad;
        if (max_image_tokens < 1 || max_frame_rows < 1)
            bad = "max_image_tokens and max_video_frame_tokens must be positive";
        if (bad.empty() && check(&bad) < 0) bad = "the container's Muse Glimmer processor "
                                                  "configuration is inconsistent: " + bad;
        if (!bad.empty()) {
            if (why) *why = bad;
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }
    /* Gemma 4 12B ("unified") has no tower: its embedder takes a whole 48 x 48 pooling window as
     * one patch, so a patch is a row (merge 1), and its values are u / 255. The sizes are the
     * same: a window is 48 pixels a side either way. */
    const bool unified = mt && !std::strcmp(mt, "gemma4_unified_vision");
    if (mt && (!std::strcmp(mt, "gemma4_vision") || unified)) {
        family   = Family::Gemma4;
        unit_pixels = unified;
        patch    = (int32_t)rad_meta_geti(&m, unified ? "vision_config.model_patch_size"
                                                      : "vision_config.patch_size", 0);
        merge    = unified ? 1 : (int32_t)rad_meta_geti(&m, "vision_config.pooling_kernel_size", 0);
        temporal = 1;
        if (patch <= 0 || merge <= 0) {
            if (why) *why = "the container states no vision tower geometry (vision_config."
                            "patch_size, pooling_kernel_size)";
            return RAD_E_NOTFOUND;
        }
        tok_image = (int32_t)rad_meta_geti(&m, "image_token_id", -1);
        tok_video = (int32_t)rad_meta_geti(&m, "video_token_id", -1);
        tok_start = (int32_t)rad_meta_geti(&m, "boi_token_id", -1);
        tok_end   = (int32_t)rad_meta_geti(&m, "eoi_token_id", -1);
        if (tok_image < 0 || tok_video < 0 || tok_start < 0 || tok_end < 0) {
            if (why) *why = "the container states no media token ids (image_token_id, "
                            "video_token_id, boi_token_id, eoi_token_id)";
            return RAD_E_NOTFOUND;
        }
        /* The budgets from the processor config, the model's stated default when it is absent. */
        image_soft_tokens = (int32_t)rad_meta_geti(
            &m, "processor.image_processor.max_soft_tokens",
            rad_meta_geti(&m, "vision_soft_tokens_per_image",
                          rad_meta_geti(&m, "vision_config.default_output_length",
                                        rad_meta_geti(&m, "vision_config.num_soft_tokens", 280))));
        video_soft_tokens = (int32_t)rad_meta_geti(&m, "processor.video_processor.max_soft_tokens",
                                                   70);
        fps        = rad_meta_getf(&m, "processor.video_processor.fps", 1.0);
        min_frames = 1;
        max_frames = (int32_t)rad_meta_geti(&m, "processor.video_processor.max_frames", 32);
        max_frame_rows = 0;
        /* The pixel bands are QwenVL's; here they only have to be consistent for check(). */
        const int64_t f2 = (int64_t)factor() * factor();
        image_min_pixels = video_min_pixels = f2;
        image_max_pixels = (int64_t)image_soft_tokens * f2;
        video_max_pixels = (int64_t)video_soft_tokens * f2 * max_frames;
        std::string bad;
        if (check(&bad) < 0) {
            if (why) *why = "the container's processor configuration is inconsistent: " + bad;
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }

    patch    = (int32_t)rad_meta_geti(&m, "vision_config.patch_size", 0);
    temporal = (int32_t)rad_meta_geti(&m, "vision_config.temporal_patch_size", 0);
    merge    = (int32_t)rad_meta_geti(&m, "vision_config.spatial_merge_size", 0);
    if (patch <= 0 || temporal <= 0 || merge <= 0) {
        if (why) *why = "the container states no vision tower geometry (vision_config.patch_size, "
                        "temporal_patch_size, spatial_merge_size)";
        return RAD_E_NOTFOUND;
    }
    tok_image = (int32_t)rad_meta_geti(&m, "image_token_id", -1);
    tok_video = (int32_t)rad_meta_geti(&m, "video_token_id", -1);
    tok_start = (int32_t)rad_meta_geti(&m, "vision_start_token_id", -1);
    tok_end   = (int32_t)rad_meta_geti(&m, "vision_end_token_id", -1);
    if (tok_image < 0 || tok_video < 0 || tok_start < 0 || tok_end < 0) {
        if (why) *why = "the container states no media token ids (image_token_id, video_token_id, "
                        "vision_start_token_id, vision_end_token_id)";
        return RAD_E_NOTFOUND;
    }
    /* Processor-only keys come from the checkpoint's preprocessor configs, which a container may
     * not carry; the defaults are what the Qwen3.x checkpoints ship. */
    image_min_pixels = rad_meta_geti(&m, "preprocessor.size.shortest_edge",
                                     rad_meta_geti(&m, "preprocessor.min_pixels", 65536));
    image_max_pixels = rad_meta_geti(&m, "preprocessor.size.longest_edge",
                                     rad_meta_geti(&m, "preprocessor.max_pixels", 16777216));
    video_min_pixels = rad_meta_geti(&m, "video_preprocessor.size.shortest_edge", 4096);
    video_max_pixels = rad_meta_geti(&m, "video_preprocessor.size.longest_edge", 25165824);
    fps        = rad_meta_getf(&m, "video_preprocessor.fps", 2.0);
    min_frames = (int32_t)rad_meta_geti(&m, "video_preprocessor.min_frames", 4);
    max_frames = (int32_t)rad_meta_geti(&m, "video_preprocessor.max_frames", 768);
    max_frame_rows = (int32_t)rad_meta_geti(&m, "video_preprocessor.max_video_tokens", 768);
    read_triple(m, "preprocessor.image_mean", mean);
    read_triple(m, "preprocessor.image_std", stdv);
    std::string bad;
    if (check(&bad) < 0) {
        if (why) *why = "the container's preprocessor configuration is inconsistent: " + bad;
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

int VisionConfig::check(std::string* why) const {
    auto no = [why](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    if (image_min_pixels <= 0) return no("image_min_pixels must be positive");
    if (image_max_pixels < image_min_pixels)
        return no("image_max_pixels " + std::to_string(image_max_pixels) + " is below image_min_pixels " +
                  std::to_string(image_min_pixels));
    if (video_min_pixels <= 0) return no("video_min_pixels must be positive");
    if (video_max_pixels < video_min_pixels)
        return no("video_max_pixels " + std::to_string(video_max_pixels) + " is below video_min_pixels " +
                  std::to_string(video_min_pixels));
    if (!(fps > 0.0)) return no("fps must be positive");
    if (min_frames < 1) return no("min_frames must be at least 1");
    for (int c = 0; c < 3; ++c)
        if (!(stdv[c] > 0.0f)) return no("image_std must be positive");
    if (max_frames < min_frames)
        return no("max_frames " + std::to_string(max_frames) + " is below min_frames " +
                  std::to_string(min_frames));
    if (family == Family::Pixtral &&
        (patch != 14 || merge != 2 || temporal != 1 || tok_image < 0 ||
         pixtral_longest_edge < factor() || pixtral_longest_edge % factor() ||
         pixtral_longest_edge > 1540 || image_token.empty() || image_break_token.empty() ||
         image_end_token.empty()))
        return no(
            "Pixtral requires patch geometry, merge 2, image token ids and a whole merged-patch longest edge");
    if (family == Family::Pixtral)
        for (int i = 0; i < 3; ++i)
            if (!std::isfinite(mean[i]) || !std::isfinite(stdv[i]))
                return no("Pixtral requires finite image means and deviations");
    if (family == Family::Lfm2) {
        if (min_tiles < 1 || max_tiles < min_tiles || tile <= 0 || tile % factor())
            return no("the tiling (min_tiles " + std::to_string(min_tiles) + ", max_tiles " +
                      std::to_string(max_tiles) + ", tile_size " + std::to_string(tile) +
                      ") is not a range of whole merged patches");
        if (!(max_pixels_tolerance > 0.0) || max_segment_patches <= 0)
            return no("max_pixels_tolerance and max_num_patches must be positive");
    }
    if (family == Family::UnlimitedOcr) {
        if (patch <= 0 || downsample <= 0 || merge != downsample)
            return no("the tower's geometry (patch_size " + std::to_string(patch) +
                      ", downsample_ratio " + std::to_string(downsample) +
                      ") does not cut a whole merged patch");
        if (base_size <= 0 || base_size % factor() || image_size <= 0 || image_size % factor())
            return no("the global view's side (" + std::to_string(base_size) + " and " +
                      std::to_string(image_size) + " pixels) is not a whole number of merged "
                      "patches");
        if (min_tiles < 1 || max_tiles < min_tiles || tile <= 0 || tile % factor())
            return no("the tiling (min_patches " + std::to_string(min_tiles) + ", max_patches " +
                      std::to_string(max_tiles) + ", tile_size " + std::to_string(tile) +
                      ") is not a range of whole merged patches");
    }
    if (family == Family::Glm5Next && (min_image_tokens < 1 || max_image_tokens < min_image_tokens))
        return no("the image token band [" + std::to_string(min_image_tokens) + ", " +
                  std::to_string(max_image_tokens) + "] is empty");
    if (family == Family::Gemma4) {
        if (!soft_budget_ok(image_soft_tokens))
            return no("the image soft-token budget " + std::to_string(image_soft_tokens) +
                      " is not one of 70, 140, 280, 560, 1120");
        if (!soft_budget_ok(video_soft_tokens))
            return no("the video frame soft-token budget " + std::to_string(video_soft_tokens) +
                      " is not one of 70, 140, 280, 560, 1120");
    }
    return RAD_OK;
}

/* ================================================================== sizes and sampling */

/* Python's round(): half-to-even on the double, so a size landing on .5 rounds as the Python
 * reference does. */
static inline int64_t pyround(double x) { return (int64_t)std::nearbyint(x); }

/* The step the references share after rounding each side to `factor`: a size past `max_px`
 * shrinks by the area ratio and rounds down, one under `min_px` grows and rounds up. `t` frames
 * (rounded to `tb`) count toward the area; a picture passes 1 and 1. */
static void fit_pixels(int64_t t, int64_t tb, int64_t h, int64_t w, int32_t factor,
                       int64_t min_px, int64_t max_px, int64_t* hb, int64_t* wb) {
    const double f = factor;
    if (tb * *hb * *wb > max_px) {
        const double beta = std::sqrt((double)(t * h * w) / (double)max_px);
        *hb = std::max<int64_t>(factor, (int64_t)std::floor((double)h / beta / f) * factor);
        *wb = std::max<int64_t>(factor, (int64_t)std::floor((double)w / beta / f) * factor);
    } else if (tb * *hb * *wb < min_px) {
        const double beta = std::sqrt((double)min_px / (double)(t * h * w));
        *hb = (int64_t)std::ceil((double)h * beta / f) * factor;
        *wb = (int64_t)std::ceil((double)w * beta / f) * factor;
    }
}

int smart_resize_image(int64_t h, int64_t w, int32_t factor, int64_t min_px, int64_t max_px,
                       int64_t* oh, int64_t* ow, std::string* why) {
    if (h <= 0 || w <= 0) { if (why) *why = "the picture has no size"; return RAD_E_INVAL; }
    if ((double)std::max(h, w) / (double)std::min(h, w) > 200.0) {
        if (why) *why = "the picture's aspect ratio is over 200:1";
        return RAD_E_INVAL;
    }
    const double f = factor;
    int64_t hb = pyround((double)h / f) * factor;
    int64_t wb = pyround((double)w / f) * factor;
    fit_pixels(1, 1, h, w, factor, min_px, max_px, &hb, &wb);
    *oh = hb;
    *ow = wb;
    return RAD_OK;
}

int smart_resize_video(int64_t t, int64_t h, int64_t w, int32_t tfactor, int32_t factor,
                       int64_t min_px, int64_t max_px, int64_t* oh, int64_t* ow, std::string* why) {
    if (t < tfactor) { if (why) *why = "too few frames"; return RAD_E_INVAL; }
    if (h < factor || w < factor) {
        const double s = std::max((double)factor / (double)h, (double)factor / (double)w);
        h = (int64_t)((double)h * s);
        w = (int64_t)((double)w * s);
    }
    if (h <= 0 || w <= 0) { if (why) *why = "the video has no size"; return RAD_E_INVAL; }
    if ((double)std::max(h, w) / (double)std::min(h, w) > 200.0) {
        if (why) *why = "the video's aspect ratio is over 200:1";
        return RAD_E_INVAL;
    }
    const double f = factor;
    int64_t hb = pyround((double)h / f) * factor;
    int64_t wb = pyround((double)w / f) * factor;
    const int64_t tb = pyround((double)t / (double)tfactor) * tfactor;
    fit_pixels(t, tb, h, w, factor, min_px, max_px, &hb, &wb);
    *oh = hb;
    *ow = wb;
    return RAD_OK;
}

/* The reference's frame choice: as many frames as `fps` gives over the clip's duration, clamped to
 * [min_frames, max_frames] and to the stream's length, spread evenly (numpy's linspace, rounded
 * half-to-even). A stream with no stated rate is taken at 24 fps, as the reference does. */
std::vector<int64_t> sample_frames(const VideoInfo& info, double fps, int32_t min_frames,
                                   int32_t max_frames) {
    const int64_t total = info.n_frames;
    if (total <= 0) return {};
    const double vfps = info.fps > 0.0 ? info.fps : 24.0;
    int64_t n = (int64_t)((double)total / vfps * fps);
    n = std::min<int64_t>(std::max<int64_t>(n, min_frames), max_frames);
    n = std::min<int64_t>(n, total);
    std::vector<int64_t> idx((size_t)n);
    if (n == 1) { idx[0] = 0; return idx; }
    const double step = (double)(total - 1) / (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) idx[(size_t)i] = (int64_t)std::nearbyint((double)i * step);
    idx[(size_t)n - 1] = total - 1;
    return idx;
}

int gemma4_size(int64_t h, int64_t w, int32_t patch, int32_t pool, int64_t max_patches,
                int64_t* oh, int64_t* ow, std::string* why) {
    if (h <= 0 || w <= 0) { if (why) *why = "the picture has no size"; return RAD_E_INVAL; }
    /* get_aspect_ratio_preserving_size, in Python's doubles. */
    const double total = (double)h * (double)w;
    const double target = (double)max_patches * (double)patch * (double)patch;
    const double factor = std::sqrt(target / total);
    const int64_t side = (int64_t)patch * pool;
    int64_t th = (int64_t)std::floor(factor * (double)h / (double)side) * side;
    int64_t tw = (int64_t)std::floor(factor * (double)w / (double)side) * side;
    if (th == 0 && tw == 0) {
        if (why) *why = "the picture is too small to make one pooled patch";
        return RAD_E_INVAL;
    }
    const int64_t max_side = (max_patches / ((int64_t)pool * pool)) * side;
    if (th == 0) {
        th = side;
        tw = std::min<int64_t>((int64_t)std::floor((double)w / (double)h) * side, max_side);
    } else if (tw == 0) {
        tw = side;
        th = std::min<int64_t>((int64_t)std::floor((double)h / (double)w) * side, max_side);
    }
    if ((double)th * (double)tw > target) {
        if (why) *why = "the picture's aspect ratio does not fit the patch budget";
        return RAD_E_INVAL;
    }
    *oh = th;
    *ow = tw;
    return RAD_OK;
}

/* Glm5NextImageProcessor's smart_resize and the size its resize() gives the picture, in Python's
 * doubles. A token is `temporal` frames of factor x factor pixels, and a picture counts as
 * `temporal` frames. The canvas's sides are rounded up to whole factors; under the floor the
 * picture is scaled up by the area ratio and rounded up again; over the budget a binary search over
 * the picture's height finds the largest canvas inside it. The picture then scales by the smaller
 * side ratio, never up while it reaches the floor, truncated. */
int glm5next_size(int64_t h, int64_t w, int32_t temporal, int32_t factor, int64_t min_tok,
                  int64_t max_tok, int64_t* oh, int64_t* ow, int64_t* ch, int64_t* cw,
                  std::string* why) {
    if (h <= 0 || w <= 0) { if (why) *why = "the picture has no size"; return RAD_E_INVAL; }
    const int64_t per_tok = (int64_t)temporal * factor * factor;
    const int64_t min_px = min_tok * per_tok, max_px = max_tok * per_tok;
    if (max_px < per_tok) {
        if (why) *why = "the token budget is less than one merged patch";
        return RAD_E_INVAL;
    }
    auto align = [factor](int64_t v) { return (v + factor - 1) / factor * factor; };
    int64_t ah = align(h), aw = align(w);
    if (temporal * ah * aw < min_px) {
        const double scale = std::sqrt((double)min_px / ((double)temporal * (double)h * (double)w));
        ah = align(std::max<int64_t>(1, (int64_t)std::ceil((double)h * scale)));
        aw = align(std::max<int64_t>(1, (int64_t)std::ceil((double)w * scale)));
    }
    if (temporal * ah * aw > max_px) {
        int64_t lo = 1, hi = h, bh = factor, bw = factor;
        while (lo <= hi) {
            const int64_t mid = (lo + hi) / 2;
            const int64_t wid = std::max<int64_t>(1, (int64_t)std::floor((double)(w * mid) /
                                                                         (double)h));
            const int64_t th = align(mid), tw = align(wid);
            if (temporal * th * tw <= max_px) { bh = th; bw = tw; lo = mid + 1; }
            else hi = mid - 1;
        }
        ah = bh;
        aw = bw;
    }
    double scale = std::min((double)ah / (double)h, (double)aw / (double)w);
    if (temporal * h * w >= per_tok * min_tok) scale = std::min(1.0, scale);
    *oh = ah;
    *ow = aw;
    *ch = std::max<int64_t>(1, std::min<int64_t>(ah, (int64_t)std::floor((double)h * scale)));
    *cw = std::max<int64_t>(1, std::min<int64_t>(aw, (int64_t)std::floor((double)w * scale)));
    return RAD_OK;
}

/* The reference's (EmbeddingGemma2VideoProcessor.sample_frames): the stream's duration at `fps`,
 * one frame each 1/fps seconds as int(i * native_fps / fps), then `max_frames` of those chosen by
 * numpy's linspace truncated to integers. */
std::vector<int64_t> sample_frames_gemma4(const VideoInfo& info, double fps, int32_t max_frames) {
    const int64_t total = info.n_frames;
    if (total <= 0) return {};
    std::vector<int64_t> idx;
    if (info.fps > 0.0) {
        const double step = info.fps / fps;
        const double duration = (double)total / info.fps;
        const int64_t n = std::max<int64_t>(1, (int64_t)(duration * fps));
        for (int64_t i = 0; i < n; ++i)
            idx.push_back(std::min<int64_t>(total - 1, (int64_t)((double)i * step)));
    } else {
        for (int64_t i = 0; i < total; ++i) idx.push_back(i);
    }
    if (max_frames > 0 && (int64_t)idx.size() > max_frames) {
        std::vector<int64_t> keep((size_t)max_frames);
        const double span = (double)(idx.size() - 1);
        for (int32_t i = 0; i < max_frames; ++i) {
            const double t = max_frames == 1 ? 0.0 : span * (double)i / (double)(max_frames - 1);
            keep[(size_t)i] = idx[(size_t)(int64_t)t];
        }
        idx.swap(keep);
    }
    return idx;
}

/* CPython's tuple hash (tuplehash, the xxHash-based combination of 3.8 and later) for the
 * candidates' (grid height, grid width) pairs. Lane = hash(int) = the value itself. */
static inline uint64_t py_tuple_hash(int64_t a, int64_t b) {
    constexpr uint64_t P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full,
                       P5 = 0x27D4EB2F165667C5ull;
    uint64_t acc = P5;
    for (int64_t v : { a, b }) {
        acc += (uint64_t)v * P2;
        acc = (acc << 31) | (acc >> 33);
        acc *= P1;
    }
    acc += (uint64_t)2 ^ (P5 ^ 3527539ull);
    return acc;
}

/* CPython's set of those pairs, as an 8-slot open table (PySet_MINSIZE) with its probing, and
 * iteration in slot order -- what `list(set(...))` gives, which fixes which of equally-near
 * candidate grids Python's `min` keeps. */
class PyCandidateSet {
public:
    void add(int64_t h, int64_t w) {
        for (int32_t s = 0; s < 8; ++s)
            if (slots_[s].used && slots_[s].h == h && slots_[s].w == w) return;
        uint64_t hash = py_tuple_hash(h, w), perturb = hash;
        int32_t i = (int32_t)(hash & 7);
        for (;;) {
            if (!slots_[i].used) {
                slots_[i] = { true, h, w };
                return;
            }
            if (slots_[i].h == h && slots_[i].w == w) return;
            perturb >>= 5;
            i = (int32_t)(((uint64_t)i * 5 + perturb + 1) & 7);
        }
    }
    /* In slot order, i from 0 to 7: the next used slot's grid, or false at the end. */
    bool next(int32_t* i, int64_t* h, int64_t* w) const {
        for (; *i < 8; ++*i)
            if (slots_[*i].used) {
                *h = slots_[*i].h;
                *w = slots_[*i].w;
                ++*i;
                return true;
            }
        return false;
    }

private:
    struct Slot { bool used = false; int64_t h = 0, w = 0; };
    Slot slots_[8];
};

/* MuseGlimmer{Image,Video}Processor.smart_resize, in Python's doubles: the ideal merged-token
 * sides are capped to the token budget by area, then the whole grids at their floor and ceiling
 * are candidates, and the one whose aspect is nearest the picture's own wins. Ties are real --
 * every square picture's floor and ceiling grids are both exactly nearest -- and the reference
 * resolves them by iterating the candidates in a CPython set's order, so the set's slot order is
 * simulated above. The fallback for an empty candidate set is the reference's own, kept for
 * parity; it cannot fire for a budget of at least one token. */
int muse_size(int64_t h, int64_t w, int32_t factor, int64_t max_tokens, int64_t* oh, int64_t* ow,
              std::string* why) {
    if (h <= 0 || w <= 0) { if (why) *why = "the picture has no size"; return RAD_E_INVAL; }
    if (max_tokens < 1) { if (why) *why = "the token budget is empty"; return RAD_E_INVAL; }
    const double f = factor;
    double ih = (double)h / f;
    double iw = (double)w / f;
    const double ratio = ih > 0.0 ? iw / ih : 1.0;
    if (ih * iw > (double)max_tokens) {
        ih = std::sqrt((double)max_tokens / ratio);
        iw = ih * ratio;
    }
    const int64_t fh = (int64_t)std::floor(ih), ch = (int64_t)std::ceil(ih);
    const int64_t fw = (int64_t)std::floor(iw), cw = (int64_t)std::ceil(iw);
    PyCandidateSet cand;
    cand.add(fh, fw);
    cand.add(fh, cw);
    cand.add(ch, fw);
    cand.add(ch, cw);
    const double aspect = (double)h / (double)w;
    int64_t gh = -1, gw = -1;
    double best = 0.0;
    for (int32_t i = 0;;) {
        int64_t ch2 = 0, cw2 = 0;
        if (!cand.next(&i, &ch2, &cw2)) break;
        if (ch2 < 1 || cw2 < 1 || (double)ch2 * (double)cw2 > (double)max_tokens) continue;
        const double diff = std::fabs((double)ch2 / (double)cw2 - aspect);
        if (gh < 0 || diff < best) {
            best = diff;
            gh = ch2;
            gw = cw2;
        }
    }
    if (gh < 0) {
        gh = std::max<int64_t>(1, pyround(ih));
        gw = std::max<int64_t>(1, pyround(iw));
    }
    *oh = gh * factor;
    *ow = gw * factor;
    return RAD_OK;
}

/* MuseGlimmerVideoProcessor.sample_frames: torch.linspace(0, total - 1, n).long(), which
 * truncates, against the .long() of a float tensor (half ties go toward zero). */
std::vector<int64_t> sample_frames_muse(const VideoInfo& info, double fps, int32_t temporal,
                                        int32_t num_frames) {
    const int64_t total = info.n_frames;
    if (total <= 0) return {};
    const double vfps = info.fps > 0.0 ? info.fps : 24.0;
    int64_t n = (int64_t)((double)total * fps / vfps);   /* Python's int(): toward zero */
    n = std::min<int64_t>(std::min<int64_t>(n, num_frames), total);
    n = std::max<int64_t>(temporal, (n / temporal) * temporal);
    n = std::min<int64_t>(n, total);
    std::vector<int64_t> idx((size_t)n);
    if (n == 1) { idx[0] = 0; return idx; }
    const double step = (double)(total - 1) / (double)(n - 1);
    for (int64_t i = 0; i < n; ++i) idx[(size_t)i] = (int64_t)((double)i * step);
    return idx;
}

/* ================================================================== the resample
 *
 * Pillow's ImagingResample for 8-bit RGB: coefficients in double, normalised per output pixel and
 * rounded to 22-bit fixed point; a horizontal pass over only the source rows the vertical pass
 * reads, into an 8-bit intermediate; then the vertical pass. The intermediate rounding is part of
 * the function: a single-pass float resample is a different function from the one the model was
 * trained against. */
namespace {

constexpr int kPrec = 32 - 8 - 2;

inline double bicubic(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

/* ATen's HelperInterpLanczos filter, which it credits to Pillow's Resample.c: sin(pi x) / (pi x)
 * windowed by the same at a third. |x| >= 3 contributes nothing. */
inline double sinc_lanczos(double x) {
    if (x == 0.0) return 1.0;
    x *= 3.14159265358979323846;
    return std::sin(x) / x;
}

inline double lanczos(double x) {
    if (x < 0.0) x = -x;
    if (x >= 3.0) return 0.0;
    return sinc_lanczos(x) * sinc_lanczos(x / 3.0);
}

inline uint8_t clip8(int32_t v, int prec) {
    const int32_t s = v >> prec;
    return (uint8_t)(s < 0 ? 0 : (s > 255 ? 255 : s));
}

/* One axis: `bounds` gets (first, count) per output pixel and `kk` its fixed-point weights,
 * `ksize` per row; `*prec` gets their fraction bits. PIL uses a fixed 22; with `torch`, ATen's
 * per-axis choice: the most bits at which the largest weight still fits an int16. */
int coeffs(int32_t in_size, int32_t out_size, std::vector<int32_t>* bounds,
           std::vector<int32_t>* kk, bool torch, int* prec, double (*filter)(double),
           double support_scale = 2.0) {
    const double scale = (double)in_size / (double)out_size;
    const double fscale = scale < 1.0 ? 1.0 : scale;
    const double support = support_scale * fscale;
    const int ksize = (int)std::ceil(support) * 2 + 1;
    std::vector<double> k((size_t)ksize), w((size_t)out_size * (size_t)ksize, 0.0);
    bounds->assign((size_t)out_size * 2, 0);
    kk->assign((size_t)out_size * (size_t)ksize, 0);
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        const double ss = 1.0 / fscale;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double ww = 0.0;
        int x = 0;
        for (; x < xmax; ++x) {
            const double w = filter((x + xmin - center + 0.5) * ss);
            k[(size_t)x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; ++x)
            if (ww != 0.0) k[(size_t)x] /= ww;
        for (; x < ksize; ++x) k[(size_t)x] = 0.0;
        for (x = 0; x < ksize; ++x) w[(size_t)xx * ksize + x] = k[(size_t)x];
        (*bounds)[(size_t)xx * 2]     = xmin;
        (*bounds)[(size_t)xx * 2 + 1] = xmax;
    }
    *prec = kPrec;
    if (torch) {
        double largest = 0.0;
        for (double v : w) largest = std::max(largest, v);
        for (*prec = 0; *prec < 22; ++*prec)
            if ((int)(0.5 + largest * (1 << (*prec + 1))) >= (1 << 15)) break;
    }
    for (size_t i = 0; i < w.size(); ++i) {
        const double v = w[i];
        (*kk)[i] = v < 0 ? (int32_t)(-0.5 + v * (1 << *prec)) : (int32_t)(0.5 + v * (1 << *prec));
    }
    return ksize;
}

}  /* namespace */

/* ================================================================== AVX2 resample helpers
 *
 * The two passes below are fixed-point int32 per tap. Integer addition is exact and associative at
 * these magnitudes (the coefficients of one output sum to 1 << prec and 255 << prec is about
 * 1.07e9, so no wrapper), so evaluating eight output pixels at once in any tap order is the same
 * function as the scalar loop; the rounding is only the final shift and clip. The horizontal pass
 * has variable tap counts and arbitrary stride-3 starts, so its lanes gather one 32-bit word per
 * tap (the RGB bytes of a pixel) and mask off inactive taps; the vertical pass reads whole rows,
 * where eight pixels are 24 contiguous bytes.
 *
 * Dispatch is local rather than through libavx: this file is core host code, and the AVX2 bodies
 * are target-attributed so the baseline object they live in still cannot execute AVX2 before the
 * check. mm_force_scalar() selects the scalar path for the bit-exactness test. */

static std::atomic<bool> g_force_scalar{ false };

void mm_force_scalar(bool on) { g_force_scalar.store(on, std::memory_order_relaxed); }

static bool mm_use_avx2() {
#ifdef RAD_MM_HAVE_AVX2
    static const bool cpu = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2");
    }();
    return cpu && !g_force_scalar.load(std::memory_order_relaxed);
#else
    return false;
#endif
}

#if defined(RAD_MM_HAVE_AVX2)
#define RAD_MM_AVX2 __attribute__((target("avx2")))
#define RAD_MM_INLINE_AVX2 __attribute__((target("avx2"), always_inline)) inline
#define RAD_MM_EXTRACT16(v) _mm256_extracti128_si256((v), 1)

/* Eight RGB pixels from 24 contiguous bytes (bytes 0..23 of p). */
RAD_MM_INLINE_AVX2 void load_rgb8(const uint8_t* p, __m256i* r, __m256i* g, __m256i* b) {
    const __m128i a = _mm_loadu_si128((const __m128i*)p);
    const __m128i c = _mm_loadl_epi64((const __m128i*)(p + 16));
    const __m128i mr0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i mg0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i mb0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i mr1 = _mm_setr_epi8(2, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i mg1 = _mm_setr_epi8(0, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i mb1 = _mm_setr_epi8(1, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i rb = _mm_or_si128(_mm_shuffle_epi8(a, mr0), _mm_slli_si128(_mm_shuffle_epi8(c, mr1), 6));
    const __m128i gb = _mm_or_si128(_mm_shuffle_epi8(a, mg0), _mm_slli_si128(_mm_shuffle_epi8(c, mg1), 5));
    const __m128i bb = _mm_or_si128(_mm_shuffle_epi8(a, mb0), _mm_slli_si128(_mm_shuffle_epi8(c, mb1), 5));
    *r = _mm256_cvtepu8_epi32(rb);
    *g = _mm256_cvtepu8_epi32(gb);
    *b = _mm256_cvtepu8_epi32(bb);
}

/* Eight rounded, clipped int32 sums written as eight RGB triplets (exactly 24 bytes). */
RAD_MM_INLINE_AVX2 void store_rgb8(uint8_t* o, __m256i s0, __m256i s1, __m256i s2, int prec) {
    const __m256i sh = _mm256_set1_epi32(prec);
    const __m256i v0 = _mm256_min_epi32(_mm256_max_epi32(_mm256_srav_epi32(s0, sh), _mm256_setzero_si256()),
                                        _mm256_set1_epi32(255));
    const __m256i v1 = _mm256_min_epi32(_mm256_max_epi32(_mm256_srav_epi32(s1, sh), _mm256_setzero_si256()),
                                        _mm256_set1_epi32(255));
    const __m256i v2 = _mm256_min_epi32(_mm256_max_epi32(_mm256_srav_epi32(s2, sh), _mm256_setzero_si256()),
                                        _mm256_set1_epi32(255));
    const __m128i r16 = _mm_packus_epi32(_mm256_castsi256_si128(v0), RAD_MM_EXTRACT16(v0));
    const __m128i g16 = _mm_packus_epi32(_mm256_castsi256_si128(v1), RAD_MM_EXTRACT16(v1));
    const __m128i b16 = _mm_packus_epi32(_mm256_castsi256_si128(v2), RAD_MM_EXTRACT16(v2));
    const __m128i r8 = _mm_packus_epi16(r16, r16);
    const __m128i g8 = _mm_packus_epi16(g16, g16);
    const __m128i b8 = _mm_packus_epi16(b16, b16);
    /* Layout [r0..r3 g0..g3 b0..b3] then shuffle to 12 RGB bytes; twice for eight pixels.
     * Mask the low four bytes before shifting, or each shift would overwrite the previous
     * channel's bytes. */
    const __m128i low4 = _mm_set_epi64x(0, 0xffffffffll);
    const __m128i m = _mm_setr_epi8(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, -1, -1, -1, -1);
    const __m128i lo = _mm_shuffle_epi8(
        _mm_or_si128(_mm_or_si128(_mm_and_si128(r8, low4),
                                  _mm_slli_si128(_mm_and_si128(g8, low4), 4)),
                     _mm_slli_si128(_mm_and_si128(b8, low4), 8)),
        m);
    const __m128i hi = _mm_shuffle_epi8(
        _mm_or_si128(_mm_or_si128(_mm_and_si128(_mm_srli_si128(r8, 4), low4),
                                  _mm_slli_si128(_mm_and_si128(_mm_srli_si128(g8, 4), low4), 4)),
                     _mm_slli_si128(_mm_and_si128(_mm_srli_si128(b8, 4), low4), 8)),
        m);
    _mm_storel_epi64((__m128i*)o, lo);
    _mm_storeu_si32(o + 8, _mm_srli_si128(lo, 8));
    _mm_storel_epi64((__m128i*)(o + 12), hi);
    _mm_storeu_si32(o + 20, _mm_srli_si128(hi, 8));
}

/* One horizontal row, eight output pixels a step. plans is [group][tap] of eight (offset, coeff,
 * mask) int32s. The gather reads four bytes at the last input pixel's B channel, so the caller
 * keeps the buffer's final row on the scalar path. */
RAD_MM_AVX2 void hpass_row(const uint8_t* row, uint8_t* o, int32_t ngroups, int32_t ksh,
                           int prec, const int32_t* plans) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i round = _mm256_set1_epi32(1 << (prec - 1));
    const __m256i mask8 = _mm256_set1_epi32(0xff);
    for (int32_t g = 0; g < ngroups; ++g) {
        __m256i s0 = round, s1 = round, s2 = round;
        const int32_t* pl = plans + (size_t)g * (size_t)ksh * 24;
        for (int32_t t = 0; t < ksh; ++t, pl += 24) {
            const __m256i off = _mm256_loadu_si256((const __m256i*)pl);
            const __m256i coef = _mm256_loadu_si256((const __m256i*)(pl + 8));
            const __m256i msk = _mm256_loadu_si256((const __m256i*)(pl + 16));
            const __m256i word = _mm256_mask_i32gather_epi32(zero, (const int*)row, off, msk, 1);
            const __m256i r = _mm256_and_si256(word, mask8);
            const __m256i gg = _mm256_and_si256(_mm256_srli_epi32(word, 8), mask8);
            const __m256i bb = _mm256_and_si256(_mm256_srli_epi32(word, 16), mask8);
            s0 = _mm256_add_epi32(s0, _mm256_mullo_epi32(r, coef));
            s1 = _mm256_add_epi32(s1, _mm256_mullo_epi32(gg, coef));
            s2 = _mm256_add_epi32(s2, _mm256_mullo_epi32(bb, coef));
        }
        store_rgb8(o + (size_t)g * 24, s0, s1, s2, prec);
    }
}

/* One vertical row: every tap is a whole source row, eight output pixels at a time. */
RAD_MM_AVX2 void vpass_row(const Frame& src, int32_t ymin, int32_t ymax, const int32_t* k,
                           uint8_t* o, int prec) {
    const __m256i round = _mm256_set1_epi32(1 << (prec - 1));
    int32_t xx = 0;
    for (; xx + 8 <= src.w; xx += 8) {
        __m256i s0 = round, s1 = round, s2 = round;
        for (int32_t y = 0; y < ymax; ++y) {
            const uint8_t* p = src.rgb.data() + ((size_t)(y + ymin) * (size_t)src.w + xx) * 3;
            __m256i r, g, b;
            load_rgb8(p, &r, &g, &b);
            const __m256i kk = _mm256_set1_epi32(k[y]);
            s0 = _mm256_add_epi32(s0, _mm256_mullo_epi32(r, kk));
            s1 = _mm256_add_epi32(s1, _mm256_mullo_epi32(g, kk));
            s2 = _mm256_add_epi32(s2, _mm256_mullo_epi32(b, kk));
        }
        store_rgb8(o + (size_t)xx * 3, s0, s1, s2, prec);
    }
    for (; xx < src.w; ++xx) {
        int32_t s0 = 1 << (prec - 1), s1 = s0, s2 = s0;
        for (int32_t y = 0; y < ymax; ++y) {
            const uint8_t* p = src.rgb.data() + ((size_t)(y + ymin) * (size_t)src.w + xx) * 3;
            s0 += p[0] * k[y];
            s1 += p[1] * k[y];
            s2 += p[2] * k[y];
        }
        o[xx * 3 + 0] = clip8(s0, prec);
        o[xx * 3 + 1] = clip8(s1, prec);
        o[xx * 3 + 2] = clip8(s2, prec);
    }
}

/* One patch row: for each channel, P strided source bytes through that channel's LUT. lut32 is
 * the same LUT as 32-bit words, so one gather gets all three channels' table entries. */
RAD_MM_AVX2 void patch_row(const uint8_t* row, uint16_t* dst, int32_t P, size_t chstride,
                           const uint32_t* lut32, const uint16_t* lut) {
    for (int32_t x = 0; x + 8 <= P; x += 8) {
        __m256i r, g, b;
        load_rgb8(row + (size_t)x * 3, &r, &g, &b);
        const __m256i idx[3] = { r, g, b };
        for (int ch = 0; ch < 3; ++ch) {
            const __m256i ii = _mm256_add_epi32(idx[ch], _mm256_set1_epi32(ch * 256));
            const __m256i v = _mm256_i32gather_epi32((const int*)lut32, ii, 4);
            const __m128i h = _mm_packus_epi32(_mm256_castsi256_si128(v), RAD_MM_EXTRACT16(v));
            _mm_storeu_si128((__m128i*)(dst + (size_t)ch * chstride + x), h);
        }
    }
    for (int ch = 0; ch < 3; ++ch)
        for (int32_t x = (P / 8) * 8; x < P; ++x)
            dst[(size_t)ch * chstride + x] = lut[ch * 256 + row[(size_t)x * 3 + ch]];
}
#endif  /* RAD_MM_HAVE_AVX2 */

/* One pass over an axis. `horizontal` reads rows [y0, y0 + rows) of `src` and writes w columns;
 * `vertical` reads rows from `shift` on and writes h rows. Both accumulate in fixed point and
 * round each output to eight bits, as Pillow does -- the intermediate's rounding is part of the
 * function, and the scalar tap order must not change.
 *
 * Both passes are row-independent: every output row reads only its own input rows and writes only
 * its own output row, in the same per-tap order, so splitting the rows across threads is bit-exact
 * (the goldens pin it). */
static void run_rows(int32_t rows, const std::function<void(int32_t)>& row) {
    row_pool_run(rows, 8, [&row](int64_t y) { row((int32_t)y); });
}

static void pass_horizontal(const Frame& src, int32_t y0, int32_t rows, int32_t w,
                            const std::vector<int32_t>& bh, const std::vector<int32_t>& kh,
                            int32_t ksh, int prec, Frame* out) {
    out->w = w;
    out->h = rows;
    out->rgb.assign((size_t)w * (size_t)rows * 3, 0);
    const bool simd = mm_use_avx2();
    const int32_t ngroups = simd ? w / 8 : 0;
    std::vector<int32_t> plans;
    if (ngroups) {
        plans.resize((size_t)ngroups * (size_t)ksh * 24);
        for (int32_t g = 0; g < ngroups; ++g)
            for (int32_t t = 0; t < ksh; ++t) {
                int32_t* p = plans.data() + ((size_t)g * (size_t)ksh + (size_t)t) * 24;
                for (int32_t j = 0; j < 8; ++j) {
                    const int32_t xx = g * 8 + j;
                    const int32_t xmin = bh[(size_t)xx * 2], xmax = bh[(size_t)xx * 2 + 1];
                    const bool on = t < xmax;
                    p[j] = on ? (xmin + t) * 3 : 0;
                    p[8 + j] = on ? kh[(size_t)xx * (size_t)ksh + (size_t)t] : 0;
                    p[16 + j] = on ? -1 : 0;
                }
            }
    }
    auto scalar_row = [&](const uint8_t* row, uint8_t* o, int32_t x0) {
        for (int32_t xx = x0; xx < w; ++xx) {
            const int32_t xmin = bh[(size_t)xx * 2], xmax = bh[(size_t)xx * 2 + 1];
            const int32_t* k = kh.data() + (size_t)xx * (size_t)ksh;
            int32_t s0 = 1 << (prec - 1), s1 = s0, s2 = s0;
            for (int32_t x = 0; x < xmax; ++x) {
                const uint8_t* p = row + (size_t)(x + xmin) * 3;
                s0 += p[0] * k[x];
                s1 += p[1] * k[x];
                s2 += p[2] * k[x];
            }
            o[xx * 3 + 0] = clip8(s0, prec);
            o[xx * 3 + 1] = clip8(s1, prec);
            o[xx * 3 + 2] = clip8(s2, prec);
        }
    };
    run_rows(rows, [&](int32_t yy) {
        const uint8_t* row = src.rgb.data() + (size_t)(yy + y0) * (size_t)src.w * 3;
        uint8_t* o = out->rgb.data() + (size_t)yy * (size_t)w * 3;
        /* The gather reads one byte past the row, which stays in the buffer on every row but its
         * last; that one stays scalar. */
        if (!ngroups || yy + y0 == src.h - 1) { scalar_row(row, o, 0); return; }
#ifdef RAD_MM_HAVE_AVX2
        hpass_row(row, o, ngroups, ksh, prec, plans.data());
        scalar_row(row, o, ngroups * 8);
#endif
    });
}

static void pass_vertical(const Frame& src, int32_t shift, int32_t h,
                          const std::vector<int32_t>& bv, const std::vector<int32_t>& kv,
                          int32_t ksv, int prec, Frame* out) {
    out->w = src.w;
    out->h = h;
    out->rgb.assign((size_t)src.w * (size_t)h * 3, 0);
    const bool simd = mm_use_avx2();
    run_rows(h, [&](int32_t yy) {
        const int32_t ymin = bv[(size_t)yy * 2] - shift, ymax = bv[(size_t)yy * 2 + 1];
        const int32_t* k = kv.data() + (size_t)yy * (size_t)ksv;
        uint8_t* o = out->rgb.data() + (size_t)yy * (size_t)src.w * 3;
#ifdef RAD_MM_HAVE_AVX2
        if (simd) { vpass_row(src, ymin, ymax, k, o, prec); return; }
#endif
        (void)simd;
        for (int32_t xx = 0; xx < src.w; ++xx) {
            int32_t s0 = 1 << (prec - 1), s1 = s0, s2 = s0;
            for (int32_t y = 0; y < ymax; ++y) {
                const uint8_t* p = src.rgb.data() + ((size_t)(y + ymin) * (size_t)src.w + xx) * 3;
                s0 += p[0] * k[y];
                s1 += p[1] * k[y];
                s2 += p[2] * k[y];
            }
            o[xx * 3 + 0] = clip8(s0, prec);
            o[xx * 3 + 1] = clip8(s1, prec);
            o[xx * 3 + 2] = clip8(s2, prec);
        }
    });
}

static void resize_bicubic_impl(const Frame& in, int32_t w, int32_t h, Frame* out, bool torch,
                                double (*filter)(double) = bicubic, double support_scale = 2.0) {
    if (in.w == w && in.h == h) { *out = in; return; }
    std::vector<int32_t> bh, kh, bv, kv;
    int ph = kPrec, pv = kPrec;
    const int ksh = coeffs(in.w, w, &bh, &kh, torch, &ph, filter, support_scale);
    const int ksv = coeffs(in.h, h, &bv, &kv, torch, &pv, filter, support_scale);
    const bool need_h = w != in.w, need_v = h != in.h;

    /* Pillow's pass order (libImaging/Resample.c): vertical first when the height shrinks and by
     * more than twice what the width does, so the horizontal pass runs over fewer rows. The
     * intermediate is eight bits, so the order changes the picture: both are the reference's.
     * torchvision's on a uint8 tensor (ATen's CPU kernel) is horizontal first at every size. */
    const bool horizontal_first =
        torch || !((int64_t)in.h - h > 0 && (int64_t)in.h - h > 2 * ((int64_t)in.w - w));

    Frame tmp;
    if (need_h && need_v) {
        if (horizontal_first) {
            /* Only the source rows the vertical pass reads are resampled horizontally. */
            const int32_t y0 = bv[0];
            const int32_t rows = bv[(size_t)h * 2 - 2] + bv[(size_t)h * 2 - 1] - y0;
            pass_horizontal(in, y0, rows, w, bh, kh, ksh, ph, &tmp);
            pass_vertical(tmp, y0, h, bv, kv, ksv, pv, out);
        } else {
            pass_vertical(in, 0, h, bv, kv, ksv, pv, &tmp);
            pass_horizontal(tmp, 0, h, w, bh, kh, ksh, ph, out);
        }
        return;
    }
    if (need_h) {
        pass_horizontal(in, 0, in.h, w, bh, kh, ksh, ph, out);
        return;
    }
    pass_vertical(in, 0, h, bv, kv, ksv, pv, out);
}

void resize_bicubic(const Frame& in, int32_t w, int32_t h, Frame* out) {
    resize_bicubic_impl(in, w, h, out, false);
}

void resize_bicubic_torch(const Frame& in, int32_t w, int32_t h, Frame* out) {
    resize_bicubic_impl(in, w, h, out, true);
}

/* The Muse Glimmer reference's resize filter: torchvision's tvF.resize on the uint8 pixels with
 * InterpolationMode.LANCZOS and antialias=True, CPU (the TorchvisionBackend refuses it anywhere
 * else and falls back to bicubic). On CPU torchvision sends a uint8 tensor to torch's native
 * uint8 kernel, which is PIL's two-pass fixed-point scheme with ATen's parameter choices: weights
 * in double, normalised, quantised to int16 at an adaptive precision (the most bits at which the
 * largest weight still fits, at most 22), the horizontal pass first, its output rounded to eight
 * bits and clamped, then the vertical pass; each pass rounds half up and clamps 0..255. That is
 * resize_bicubic_impl's `torch` branch with the Lanczos filter in place of bicubic -- the same
 * quantisation, pass order and rounding this file already implements for torchvision's bicubic
 * (coeffs, kPrec, pass_horizontal, pass_vertical). */
void resize_lanczos_torch(const Frame& in, int32_t w, int32_t h, Frame* out) {
    resize_bicubic_impl(in, w, h, out, true, lanczos, 3.0);
}

/* Python's round(): the nearest integer, a tie to the even one. The default floating-point
 * environment rounds that way, which nearbyint follows. */
static inline int64_t py_round(double x) { return (int64_t)std::nearbyint(x); }

void radio_size(int64_t w, int64_t h, const VisionConfig& c, int32_t* wp, int32_t* hp) {
    const int64_t div = c.merge, lo = c.radio_min_patches;
    const int64_t hi = c.radio_max_patches > 0 ? c.radio_max_patches : INT64_MAX;
    const int64_t budget = std::max<int64_t>((int64_t)(c.radio_max_len - 4) * div * div, lo);
    const int64_t avail = std::max<int64_t>(std::min<int64_t>(budget, hi), lo);
    /* round(x + 0.5): x + 1 rounded down for a fraction, and for a whole x, x or x + 1 by parity
     * (Python's tie to even), which the reference's comment does not say but its code does. */
    const int64_t ch = py_round((double)h / (double)c.patch + 0.5);
    const int64_t cw = py_round((double)w / (double)c.patch + 0.5);
    const double factor = std::min(std::sqrt((double)avail / (double)(ch * cw)), 1.0);
    int64_t th = (int64_t)std::floor(factor * (double)ch);
    int64_t tw = (int64_t)std::floor(factor * (double)cw);
    if (avail > lo && th * tw < lo) {
        const double up = std::sqrt((double)lo / (double)(th * tw));
        th = (int64_t)std::ceil(up * (double)th);
        tw = (int64_t)std::ceil(up * (double)tw);
    }
    if (const int64_t r = th % div) th = (th + div - r) * tw <= avail ? th + div - r
                                                                       : std::max(div, th - r);
    if (const int64_t r = tw % div) tw = th * (tw + div - r) <= avail ? tw + div - r
                                                                       : std::max(div, tw - r);
    *wp = (int32_t)tw;
    *hp = (int32_t)th;
}

void radio_video_size(int64_t w, int64_t h, const VisionConfig& c, int32_t* wp, int32_t* hp) {
    const int64_t div = c.merge, target = c.radio_video_patches;
    const double aspect = (double)w / (double)std::max<int64_t>(h, 1);
    int64_t ph = std::max<int64_t>(py_round(std::sqrt((double)target / aspect)), 1);
    int64_t pw = std::max<int64_t>(py_round(std::sqrt((double)target * aspect)), 1);
    if (div > 1) {
        const int64_t rh = ph % div, rw = pw % div;
        const int64_t phu = ph + (rh ? div - rh : 0), pwu = pw + (rw ? div - rw : 0);
        if (phu * pwu <= target) {
            ph = phu;
            pw = pwu;
        } else {
            ph = std::max(div, ph - rh);
            pw = std::max(div, pw - rw);
        }
    }
    *wp = (int32_t)pw;
    *hp = (int32_t)ph;
}

/* ATen's antialiased bicubic (Keys, a = -0.5) on float, one axis: output i reads inputs
 * [lo, lo + n) at the filter's normalised weights, the support widened by the scale when
 * shrinking. */
namespace {
struct AaTaps { std::vector<int32_t> lo, n; std::vector<float> w; int32_t width = 0; };
inline float aa_cubic(float x) {
    const float a = -0.5f;
    x = std::fabs(x);
    if (x < 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f) return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
    return 0.0f;
}
AaTaps aa_taps_f32(int32_t in, int32_t out) {
    AaTaps t;
    const float scale = (float)in / (float)out;
    const float support = scale >= 1.0f ? 2.0f * scale : 2.0f;
    const float inv = scale >= 1.0f ? 1.0f / scale : 1.0f;
    t.width = (int32_t)std::ceil(support) * 2 + 1;
    t.lo.resize((size_t)out);
    t.n.resize((size_t)out);
    t.w.assign((size_t)out * t.width, 0.0f);
    for (int32_t i = 0; i < out; ++i) {
        const float center = scale * ((float)i + 0.5f);
        const int32_t lo = std::max((int32_t)(center - support + 0.5f), 0);
        const int32_t n = std::min((int32_t)(center + support + 0.5f), in) - lo;
        float* w = &t.w[(size_t)i * t.width];
        float total = 0.0f;
        for (int32_t j = 0; j < n && j < t.width; ++j) {
            w[j] = aa_cubic(((float)(j + lo) - center + 0.5f) * inv);
            total += w[j];
        }
        for (int32_t j = 0; j < n && j < t.width; ++j) w[j] = total != 0.0f ? w[j] / total : 0.0f;
        t.lo[(size_t)i] = lo;
        t.n[(size_t)i] = std::min(n, t.width);
    }
    return t;
}
}  /* namespace */

void resize_bicubic_aa_f32(const Frame& in, int32_t w, int32_t h, std::vector<float>* out) {
    std::vector<float> src((size_t)in.w * in.h * 3);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (float)in.rgb[i];
    /* Width first, then height; a side that keeps its size is skipped (its taps are the identity). */
    std::vector<float> mid;
    const std::vector<float>* cur = &src;
    int32_t cw = in.w;
    if (w != in.w) {
        const AaTaps t = aa_taps_f32(in.w, w);
        mid.assign((size_t)w * in.h * 3, 0.0f);
        for (int32_t y = 0; y < in.h; ++y)
            for (int32_t x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c) {
                    float acc = 0.0f;
                    const float* wt = &t.w[(size_t)x * t.width];
                    for (int32_t j = 0; j < t.n[(size_t)x]; ++j)
                        acc += src[((size_t)y * in.w + t.lo[(size_t)x] + j) * 3 + c] * wt[j];
                    mid[((size_t)y * w + x) * 3 + c] = acc;
                }
        cur = &mid;
        cw = w;
    }
    if (h != in.h) {
        const AaTaps t = aa_taps_f32(in.h, h);
        out->assign((size_t)cw * h * 3, 0.0f);
        for (int32_t y = 0; y < h; ++y) {
            const float* wt = &t.w[(size_t)y * t.width];
            for (int32_t x = 0; x < cw; ++x)
                for (int c = 0; c < 3; ++c) {
                    float acc = 0.0f;
                    for (int32_t j = 0; j < t.n[(size_t)y]; ++j)
                        acc += (*cur)[((size_t)(t.lo[(size_t)y] + j) * cw + x) * 3 + c] * wt[j];
                    (*out)[((size_t)y * cw + x) * 3 + c] = acc;
                }
        }
        return;
    }
    *out = *cur;
}

void cap_area(const Frame& in, int64_t max_pixels, Frame* out) {
    if (max_pixels <= 0 || (int64_t)in.w * in.h <= max_pixels) { *out = in; return; }
    const double scale = std::sqrt((double)max_pixels / ((double)in.w * in.h));
    resize_bicubic(in, std::max(1, (int)(in.w * scale)), std::max(1, (int)(in.h * scale)), out);
}

/* ================================================================== patches */

namespace {

inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

/* Two independent 64-bit lanes over the patch bytes and the grid. The prefix cache folds the key
 * into a chained 128-bit block identity, and a collision would serve one picture's cache for
 * another, so both lanes are kept. */
void content_key(const Item& it, uint64_t* hi, uint64_t* lo) {
    uint64_t a = 0x6a09e667f3bcc908ull ^ mix64((uint64_t)it.grid_t << 42 ^
                                               (uint64_t)it.grid_h << 21 ^ (uint64_t)it.grid_w);
    uint64_t b = 0xbb67ae8584caa73bull ^
                 mix64(it.kind == Kind::Image ? 1u : it.kind == Kind::Video ? 2u : 3u);
    /* The tiling: the same patches cut into different segments are different encoder inputs. */
    for (const Item::Segment& sg : it.segments)
        a = mix64(a ^ ((uint64_t)(uint32_t)sg.h << 32 | (uint32_t)sg.w)) + (uint64_t)sg.patch0;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(it.pixels.data());
    const size_t n = it.pixels.size() * sizeof(uint16_t);
    /* The pixels' whole 64-byte stripes through eight independent lanes (util/simd.h, XXH3's
     * accumulation), which run in vector registers; a chain of mixes a word would hash at about
     * a byte a cycle. The lanes fold into both halves; the tail mixes as below. */
    uint64_t lane[8] = { 0xc2b2ae3d27d4eb4full, 0x9e3779b185ebca87ull, 0xc2b2ae3d27d4eb4full ^ n,
                         0x165667b19e3779f9ull, 0x85ebca77c2b2ae63ull, 0x27d4eb2f165667c5ull,
                         0x9e3779b97f4a7c15ull ^ n, 0x94d049bb133111ebull };
    const size_t stripes = n / 64;
    simd::stripe_acc_u64x8(lane, p, stripes);
    for (int j = 0; j < 8; ++j) {
        a = mix64(a ^ lane[j]) + (uint64_t)j;
        b = mix64(b + lane[j]) ^ (uint64_t)(j + 8);
    }
    size_t i = stripes * 64;
    for (; i + 16 <= n; i += 16) {
        uint64_t w0, w1;
        std::memcpy(&w0, p + i, 8);
        std::memcpy(&w1, p + i + 8, 8);
        a = mix64(a ^ w0) + w1;
        b = mix64(b ^ w1) ^ w0;
    }
    for (; i < n; ++i) { a = mix64(a ^ p[i]); b = mix64(b + p[i]); }
    *hi = mix64(a ^ (uint64_t)n);
    *lo = mix64(b + (uint64_t)n);
}

/* After its content key, the item's expansion as one run of `rows` copies of `tok` between the
 * `start` and `end` markers. */
static void one_run(Item* it, int32_t start, int64_t rows, int32_t tok, int32_t end) {
    content_key(*it, &it->key_hi, &it->key_lo);
    it->expand.reserve((size_t)rows + 2);
    it->expand.push_back(start);
    it->seg_at.push_back(1);
    it->expand.insert(it->expand.end(), (size_t)rows, tok);
    it->expand.push_back(end);
}

/* The patch rows, in merge-block order, from `frames` (temporal * grid_t of them, already resized
 * to grid_h * patch by grid_w * patch). Each row is (channel, frame, y, x); each value is
 * normalised as the reference does, (x - 255 mean) / (255 std) in f32, and rounded to bf16. */
void patchify(const VisionConfig& c, const std::vector<const Frame*>& frames, Item* it) {
    const int P = c.patch, T = c.temporal, M = c.merge;
    const int gh = it->grid_h, gw = it->grid_w;
    float m255[3], s255[3];
    for (int ch = 0; ch < 3; ++ch) {
        /* torch.tensor(mean) * (1 / rescale_factor), in f32: the fused rescale's own constants.
         * 1 / (1 / 255) rounds to 255 in f32. */
        m255[ch] = c.mean[ch] * 255.0f;
        s255[ch] = c.stdv[ch] * 255.0f;
    }
    /* One lookup per (channel, byte): the normalisation is a function of the byte alone. */
    uint16_t lut[3][256];
    for (int ch = 0; ch < 3; ++ch)
        for (int v = 0; v < 256; ++v)
            lut[ch][v] = rad_f32_to_bf16(((float)v - m255[ch]) / s255[ch]);

    const int64_t dim = it->patch_dim;
    it->pixels.assign((size_t)it->n_patch * (size_t)dim, 0);
    uint16_t* dst = it->pixels.data();
    for (int g = 0; g < it->grid_t; ++g)
        for (int bh = 0; bh < gh / M; ++bh)
            for (int bw = 0; bw < gw / M; ++bw)
                for (int mh = 0; mh < M; ++mh)
                    for (int mw = 0; mw < M; ++mw) {
                        const int py0 = (bh * M + mh) * P, px0 = (bw * M + mw) * P;
                        for (int ch = 0; ch < 3; ++ch)
                            for (int t = 0; t < T; ++t) {
                                const Frame& f = *frames[(size_t)(g * T + t)];
                                for (int y = 0; y < P; ++y) {
                                    const uint8_t* row = f.rgb.data() +
                                        ((size_t)(py0 + y) * f.w + px0) * 3 + ch;
                                    for (int x = 0; x < P; ++x) *dst++ = lut[ch][row[x * 3]];
                                }
                            }
                    }
}

/* Muse Glimmer's patch rows, in raster order like the reference's processor emits them (row by
 * row, not merge-block order): the tower's patch GEMM reads each row as one patch and its
 * coordinate planes/merge op need the raster grid, and the merge is `pixtral_merge`, which packs
 * each 2x2 block into one row of the reference's pixel_shuffle layout (channel-major). Each
 * flattened patch is (temporal, channel), not (channel, temporal), and each value is normalised
 * through the same per-(channel, byte) lookup as patchify's -- the fused (u - 255 mean) /
 * (255 std) in f32 is the form the reference's rescale-and-normalise gives, byte for byte,
 * rounded to bf16.
 */
void patchify_muse(const VisionConfig& c, const std::vector<const Frame*>& frames, Item* it) {
    const int P = c.patch, T = c.temporal, M = c.merge;
    const int gh = it->grid_h, gw = it->grid_w;
    float m255[3], s255[3];
    for (int ch = 0; ch < 3; ++ch) {
        m255[ch] = c.mean[ch] * 255.0f;
        s255[ch] = c.stdv[ch] * 255.0f;
    }
    uint16_t lut[3][256];
    for (int ch = 0; ch < 3; ++ch)
        for (int v = 0; v < 256; ++v)
            lut[ch][v] = rad_f32_to_bf16(((float)v - m255[ch]) / s255[ch]);
    const int64_t dim = it->patch_dim;
    it->pixels.assign((size_t)it->n_patch * (size_t)dim, 0);
    it->patch_raster = true;
    uint16_t* dst = it->pixels.data();
    for (int g = 0; g < it->grid_t; ++g)
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x) {
                const int py0 = y * P, px0 = x * P;
                for (int t = 0; t < T; ++t) {
                    const Frame& f = *frames[(size_t)(g * T + t)];
                    for (int ch = 0; ch < 3; ++ch)
                        for (int yy = 0; yy < P; ++yy) {
                            const uint8_t* row = f.rgb.data() +
                                ((size_t)(py0 + yy) * f.w + px0) * 3 + ch;
                            for (int xx = 0; xx < P; ++xx) *dst++ = lut[ch][row[xx * 3]];
                        }
                }
            }
}

/* Gemma 4's patch rows, in merge-block order like patchify's: each pooling window's patches are
 * consecutive, so the pooler is a mean of k^2 consecutive rows (attention and the position terms
 * read each patch's own coordinates, so the order is otherwise free). Each row is (y, x, channel);
 * each value is 2 * (p - 0.5) in f32 (the model's input scaling), rounded to bf16, where p is the
 * processor's rescale u * (1/255). The processors AutoProcessor loads take that product in f32 for
 * pictures (Gemma4ImageProcessor) and in double, rounded to f32, for video (Gemma 4's and
 * EmbeddingGemma 2's video processors); `video` picks which, and the two differ for u = 127. One
 * frame per segment. With `unit_pixels` (the 12B, merge 1: a row is one 48 x 48 window) the value
 * is p itself, rounded to bf16. */
void patchify_gemma4(const VisionConfig& c, const std::vector<const Frame*>& frames, bool video,
                     Item* it) {
    const int P = c.patch, M = c.merge;
    const int gh = it->grid_h, gw = it->grid_w;
    uint16_t lut[256];
    for (int v = 0; v < 256; ++v) {
        const float p = video ? (float)((double)v * (1.0 / 255.0))
                              : (float)v * (float)(1.0 / 255.0);
        lut[v] = rad_f32_to_bf16(c.unit_pixels ? p : 2.0f * (p - 0.5f));
    }
    const int64_t dim = it->patch_dim;
    it->pixels.assign((size_t)it->n_patch * (size_t)dim, 0);
    uint16_t* dst = it->pixels.data();
    for (int g = 0; g < it->grid_t; ++g) {
        const Frame& f = *frames[(size_t)g];
        for (int bh = 0; bh < gh / M; ++bh)
            for (int bw = 0; bw < gw / M; ++bw)
                for (int mh = 0; mh < M; ++mh)
                    for (int mw = 0; mw < M; ++mw) {
                        const int py0 = (bh * M + mh) * P, px0 = (bw * M + mw) * P;
                        for (int y = 0; y < P; ++y) {
                            const uint8_t* row = f.rgb.data() + ((size_t)(py0 + y) * f.w + px0) * 3;
                            for (int x = 0; x < 3 * P; ++x) *dst++ = lut[row[x]];
                        }
                    }
    }
}

/* Radio's patch rows (Family::Radio), segment by segment: `n_prefix` zero rows, then the patches
 * in merge-block order. A row is [the tubelet's frames (frame, channel, y, x) | the picture's
 * (channel, y, x)]: a picture fills the second part, a video tubelet the first, the rest zero.
 * Each value is (u / 255 - mean) / std in f32 on the resized float pixels, rounded to bf16, as the
 * reference processor's tensor ops and the model's cast to bf16 give it. `frames` holds
 * grid_t * tubelet frames of [grid_h * patch, grid_w * patch, 3] floats for a video, one for a
 * picture. */
void patchify_radio(const VisionConfig& c, const std::vector<const std::vector<float>*>& frames,
                    bool video, Item* it) {
    const int P = c.patch, M = c.merge, T = c.radio_tubelet;
    const int gh = it->grid_h, gw = it->grid_w, W = gw * P;
    const int64_t dim = it->patch_dim, pp = 3LL * P * P;
    it->pixels.assign((size_t)it->n_patch * (size_t)dim, 0);
    uint16_t* dst = it->pixels.data();
    auto norm = [&](float u, int ch) {
        return rad_f32_to_bf16((u / 255.0f - c.mean[ch]) / c.stdv[ch]);
    };
    for (int g = 0; g < it->grid_t; ++g) {
        dst += (size_t)it->seg_prefix * dim;
        for (int bh = 0; bh < gh / M; ++bh)
            for (int bw = 0; bw < gw / M; ++bw)
                for (int mh = 0; mh < M; ++mh)
                    for (int mw = 0; mw < M; ++mw) {
                        const int py0 = (bh * M + mh) * P, px0 = (bw * M + mw) * P;
                        const int nf = video ? T : 1;
                        uint16_t* o = dst + (video ? 0 : (int64_t)T * pp);
                        for (int t = 0; t < nf; ++t) {
                            const std::vector<float>& f = *frames[(size_t)(g * nf + t)];
                            for (int ch = 0; ch < 3; ++ch)
                                for (int y = 0; y < P; ++y)
                                    for (int x = 0; x < P; ++x)
                                        *o++ = norm(f[((size_t)(py0 + y) * W + px0 + x) * 3 + ch], ch);
                        }
                        dst += dim;
                    }
    }
}

/* Radio's item geometry: finish_item's, with the prefix rows counted in the patches. */
void finish_item_radio(const VisionConfig& c, Item* it) {
    it->patch_dim  = c.patch_dim();
    it->seg_prefix = c.radio_prefix;
    it->rows_h     = it->grid_h / c.merge;
    it->rows_w     = it->grid_w / c.merge;
    it->seg_rows   = it->rows_h * it->rows_w;
    it->n_seg      = it->grid_t;
    it->n_patch    = (int64_t)it->grid_t * (c.radio_prefix + (int64_t)it->grid_h * it->grid_w);
}

void finish_item(const VisionConfig& c, Item* it) {
    it->patch_dim = c.patch_dim();
    it->n_patch = (int64_t)it->grid_t * it->grid_h * it->grid_w;
    it->rows_h = it->grid_h / c.merge;
    it->rows_w = it->grid_w / c.merge;
    it->seg_rows = it->rows_h * it->rows_w;
    it->n_seg = it->grid_t;
}

}  /* namespace */

/* ================================================================== the processor */

int Processor::init(const VisionConfig& cfg,
                    std::function<std::vector<int32_t>(const std::string&)> encode_text,
                    std::function<int32_t(const std::string&)> token_id) {
    if (cfg.patch <= 0 || cfg.temporal <= 0 || cfg.merge <= 0 || cfg.max_patches <= 0)
        return RAD_E_INVAL;
    cfg_ = cfg;
    if (cfg.family == Family::Pixtral) {
        if (!token_id) return RAD_E_INVAL;
        cfg_.tok_image = token_id(cfg.image_token);
        cfg_.tok_break = token_id(cfg.image_break_token);
        cfg_.tok_end = token_id(cfg.image_end_token);
        if (cfg_.tok_image < 0 || cfg_.tok_image != cfg.tok_image || cfg_.tok_break < 0 ||
            cfg_.tok_end < 0)
            return RAD_E_NOTFOUND;
        encode_text_ = std::move(encode_text);
        return cfg_.check(nullptr);
    }
    if (cfg.family == Family::Lfm2) {
        /* The config names the markers rather than numbering them. Resolve all of them here so a
         * vocabulary missing one is refused at startup, not on the first large picture. */
        if (!token_id) return RAD_E_INVAL;
        cfg_.tok_start = token_id("<|image_start|>");
        cfg_.tok_end   = token_id("<|image_end|>");
        tok_thumbnail_ = token_id("<|img_thumbnail|>");
        if (cfg_.tok_start < 0 || cfg_.tok_end < 0 || tok_thumbnail_ < 0) return RAD_E_NOTFOUND;
        tok_tiles_.assign((size_t)cfg.max_tiles * cfg.max_tiles, -1);
        for (int32_t r = 1; r <= cfg.max_tiles; ++r)
            for (int32_t c = 1; r * c <= cfg.max_tiles; ++c) {
                const int32_t id = token_id("<|img_row_" + std::to_string(r) + "_col_" +
                                            std::to_string(c) + "|>");
                if (id < 0) return RAD_E_NOTFOUND;
                tok_tiles_[(size_t)(r - 1) * cfg.max_tiles + (c - 1)] = id;
            }
        /* A tile and the largest thumbnail are each one segment, of the tower and of a pass. */
        const int64_t seg = std::max<int64_t>((int64_t)(cfg.tile / cfg.patch) * (cfg.tile / cfg.patch),
                                              cfg.image_max_pixels / ((int64_t)cfg.patch * cfg.patch));
        if (seg > cfg.max_patches || seg > cfg.max_segment_patches) return RAD_E_INVAL;
        encode_text_ = std::move(encode_text);
        return RAD_OK;
    }
    if (cfg.family == Family::Radio) {
        /* The markers around a run are vocabulary words, not config ids. A tile is one segment of
         * one pass with its prefix rows, so the picture budget is lowered to what a pass holds;
         * a video frame's target must fit as it is. */
        if (!token_id) return RAD_E_INVAL;
        cfg_.tok_start = token_id("<img>");
        cfg_.tok_end   = token_id("</img>");
        if (cfg_.tok_start < 0 || cfg_.tok_end < 0) return RAD_E_NOTFOUND;
        const int64_t cap = cfg.max_patches - cfg.radio_prefix;
        if (cap < cfg.radio_min_patches || cap < cfg.radio_video_patches) return RAD_E_INVAL;
        if (cfg_.radio_max_patches > cap) cfg_.radio_max_patches = (int32_t)cap;
        /* A video's budget: max_frames frames in tubelets, each its prefix plus the frame target
         * (the video tile never exceeds it). */
        if (cfg_.max_video_patches <= 0)
            cfg_.max_video_patches = (int64_t)((cfg.max_frames + cfg.radio_tubelet - 1) /
                                               cfg.radio_tubelet) *
                                     (cfg.radio_prefix + cfg.radio_video_patches);
        encode_text_ = std::move(encode_text);
        return RAD_OK;
    }
    if (cfg.family == Family::Glm5Next) {
        /* A token is merge^2 patches and an image is one segment of one pass, so the budget is
         * lowered to what a pass holds. */
        const int64_t pass_tok = cfg.max_patches / ((int64_t)cfg.merge * cfg.merge);
        if (pass_tok < cfg.min_image_tokens) return RAD_E_INVAL;
        if (cfg_.max_image_tokens > pass_tok) cfg_.max_image_tokens = (int32_t)pass_tok;
        encode_text_ = std::move(encode_text);
        return RAD_OK;
    }
    if (cfg.family == Family::Gemma4) {
        /* A budget is a patch count, and an image is one segment of one pass. */
        if ((int64_t)cfg.image_soft_tokens * cfg.merge * cfg.merge > cfg.max_patches ||
            (int64_t)cfg.video_soft_tokens * cfg.merge * cfg.merge > cfg.max_patches)
            return RAD_E_INVAL;
        encode_text_ = std::move(encode_text);
        return RAD_OK;
    }
    if (cfg.family == Family::MuseGlimmer) {
        /* The markers the container does not number are vocabulary words, refused here so a
         * missing one fails at startup, not on the first media. An image and a video frame are
         * one segment each of merged tokens, so the caps are lowered to what a pass holds. */
        if (cfg_.tok_start < 0 || cfg_.tok_end < 0 || cfg_.tok_video_start < 0 ||
            cfg_.tok_video_end < 0 || cfg_.tok_video_sep < 0) {
            if (!token_id) return RAD_E_INVAL;
            if (cfg_.tok_start < 0) cfg_.tok_start = token_id("<|image_start|>");
            if (cfg_.tok_end < 0) cfg_.tok_end = token_id("<|image_end|>");
            if (cfg_.tok_video_start < 0) cfg_.tok_video_start = token_id("<|vid_start|>");
            if (cfg_.tok_video_end < 0) cfg_.tok_video_end = token_id("<|vid_end|>");
            if (cfg_.tok_video_sep < 0) cfg_.tok_video_sep = token_id("<|vid_frame_separator|>");
            if (cfg_.tok_start < 0 || cfg_.tok_end < 0 || cfg_.tok_video_start < 0 ||
                cfg_.tok_video_end < 0 || cfg_.tok_video_sep < 0)
                return RAD_E_NOTFOUND;
        }
        const int64_t pass_tok = cfg.max_patches / ((int64_t)cfg.merge * cfg.merge);
        if (pass_tok < 1 || (int64_t)cfg_.max_frame_rows * cfg.merge * cfg.merge > cfg.max_patches)
            return RAD_E_INVAL;
        if (cfg_.max_image_tokens > pass_tok) cfg_.max_image_tokens = (int32_t)pass_tok;
        if (cfg_.max_video_patches <= 0)
            cfg_.max_video_patches =
                (int64_t)((cfg.max_frames + cfg.temporal - 1) / cfg.temporal) *
                (int64_t)cfg_.max_frame_rows * cfg.merge * cfg.merge;
        encode_text_ = std::move(encode_text);
        return RAD_OK;
    }
    if (cfg.family == Family::UnlimitedOcr) {
        /* The vocabulary spells the marker (`processor.image_token`), and every placeholder of a
         * picture is that same id: there is nothing else to resolve. */
        if (!token_id) return RAD_E_INVAL;
        if (cfg_.tok_image < 0) cfg_.tok_image = token_id(cfg_.image_token);
        if (cfg_.tok_image < 0) return RAD_E_NOTFOUND;
        /* A tile's line and the global view are each one segment, so each must fit one pass. */
        const int64_t q = cfg.tile / cfg.patch;
        const int64_t g = std::max(cfg.base_size, cfg.image_size) / cfg.patch;
        if (q * (q + 4) > cfg.max_patches || g * (g + 4) > cfg.max_patches) return RAD_E_INVAL;
        encode_text_ = std::move(encode_text);
        return RAD_OK;
    }
    /* The largest image one encoder pass takes, in pixels. Lowering the band's ceiling to it makes
     * smart_resize keep every image inside a pass. */
    const int64_t pass_px = cfg.max_patches * cfg.patch * cfg.patch;
    if (cfg_.image_max_pixels > pass_px) cfg_.image_max_pixels = pass_px;
    if (cfg_.image_min_pixels > cfg_.image_max_pixels) cfg_.image_min_pixels = cfg_.image_max_pixels;
    if (cfg_.max_video_patches <= 0)
        cfg_.max_video_patches = cfg_.video_max_pixels / ((int64_t)cfg.patch * cfg.patch);
    encode_text_ = std::move(encode_text);
    return RAD_OK;
}

void lfm2_size(int64_t h, int64_t w, int32_t factor, int64_t min_px, int64_t max_px,
               int64_t* oh, int64_t* ow) {
    const double f = factor;
    int64_t hb = std::max<int64_t>(factor, pyround((double)h / f) * factor);
    int64_t wb = std::max<int64_t>(factor, pyround((double)w / f) * factor);
    fit_pixels(1, 1, h, w, factor, min_px, max_px, &hb, &wb);
    *oh = hb;
    *ow = wb;
}

bool lfm2_too_large(int64_t h, int64_t w, int32_t patch, int32_t factor, int64_t max_px,
                    double tolerance) {
    const int64_t hb = std::max<int64_t>(patch, pyround((double)h / factor) * factor);
    const int64_t wb = std::max<int64_t>(patch, pyround((double)w / factor) * factor);
    return (double)(hb * wb) > (double)max_px * tolerance;
}

/* The reference walks the ratios sorted by tile count, and within one count in Python set order;
 * this walks by columns. The two differ only on an exact tie between two nearest grids of one
 * count. */
void lfm2_grid(int64_t h, int64_t w, int32_t min_tiles, int32_t max_tiles, int32_t tile,
               int32_t* cols, int32_t* rows) {
    const double aspect = (double)w / (double)h;
    double best = INFINITY;
    *cols = *rows = 1;
    for (int32_t n = min_tiles; n <= max_tiles; ++n)
        for (int32_t c = 1; c <= n; ++c) {
            if (n % c) continue;
            const int32_t r = n / c;
            const double diff = std::fabs(aspect - (double)c / (double)r);
            if (diff < best) {
                best = diff; *cols = c; *rows = r;
            } else if (diff == best && (double)(w * h) > 0.5 * tile * tile * (double)n) {
                *cols = c; *rows = r;
            }
        }
}

/* Unlimited-OCR's ratios of one tile count tie only when they are each other's transpose: two
 * ratios of the same count are equally near an aspect only if one mirrors the other about it. So
 * the reference's set order matters for exactly the (c, r) pairs below, the tall ones (c < r)
 * its `sorted(set(...), key=area)` visits after their wide transpose -- the order CPython's set
 * has for the 2..32-tile range the checkpoint states. It comes from the reference itself
 * (tests/mkuocrgolden.py dumps it); outside that range the wide side is taken as last. */
static constexpr int8_t kTallLast[][2] = {
    {1, 3},  {1, 4},  {1, 5},  {1, 6},  {1, 7},  {1, 9},  {1, 13}, {1, 14},
    {1, 16}, {1, 18}, {1, 20}, {1, 22}, {1, 27}, {1, 29}, {2, 3},  {2, 4},
    {2, 5},  {2, 6},  {2, 7},  {2, 8},  {2, 9},  {2, 14}, {2, 15}, {2, 16},
    {3, 5},  {3, 7},  {3, 9},  {4, 5},  {4, 6},  {4, 7},  {4, 8},
};
bool uocr_tall_last(int32_t c, int32_t r) {
    for (const auto& p : kTallLast)
        if (p[0] == c && p[1] == r) return true;
    return false;
}

/* One n defers at most its divisor pairs with c < n/c, at most the divisors below its square
 * root. Over the 2..32-tile range the checkpoints state the table is built for, no n defers
 * more than this; uocr_grid's deferred buffer is sized from it and the static_assert below
 * keeps the bound checked against the table, not empirical. */
static constexpr int32_t deferred_pairs_max(int32_t max_n) {
    int32_t best = 0;
    for (int32_t n = 2; n <= max_n; ++n) {
        int32_t cnt = 0;
        for (const auto& p : kTallLast)
            if (p[0] < p[1] && (int32_t)p[0] * (int32_t)p[1] == n) ++cnt;
        if (cnt > best) best = cnt;
    }
    return best;
}
static_assert(deferred_pairs_max(32) <= 8,
              "uocr_grid's 8-slot deferred buffer must cover the stated 2..32 tile range");

void uocr_grid(int64_t h, int64_t w, int32_t min_tiles, int32_t max_tiles, int32_t tile,
               int32_t* cols, int32_t* rows) {
    *cols = *rows = 1;
    if (h <= 0 || w <= 0 || min_tiles < 1 || max_tiles < min_tiles) return;
    const double aspect = (double)w / (double)h;
    const double area = (double)w * (double)h;
    const double half = 0.5 * (double)tile * (double)tile;
    double best = INFINITY;
    for (int32_t n = min_tiles; n <= max_tiles; ++n) {
        int32_t deferred[8];      /* the transposed ratios the reference visits last */
        int32_t n_deferred = 0;
        auto visit = [&](int32_t c, int32_t r) {
            const double diff = std::fabs(aspect - (double)c / (double)r);
            if (diff < best) {
                best = diff; *cols = c; *rows = r;
            } else if (diff == best && area > half * (double)n) {
                *cols = c; *rows = r;
            }
        };
        for (int32_t c = 1; c <= n; ++c) {
            if (n % c) continue;
            if (c < n / c && uocr_tall_last(c, n / c)) {
                /* Defensive: the static_assert proves 8 covers the stated 2..32 range. Beyond
                 * it the reference's tie order is not guaranteed, and a drop is a last resort
                 * against overflow, not a designed path. */
                if (n_deferred < 8) deferred[n_deferred++] = c;
                continue;
            }
            visit(c, n / c);
        }
        for (int32_t k = 0; k < n_deferred; ++k) visit(deferred[k], n / deferred[k]);
    }
}

/* LFM2-VL's image processor (transformers' Lfm2VlImageProcessor), segment by segment: the tiles
 * in row-major order, then the thumbnail. Within a segment, patches are in merge-block order (each
 * merge x merge block consecutive, dy then dx, as the projector's pixel unshuffle reads them, so
 * its merge is a view); each patch is (y, x, channel). */
int Processor::image_lfm2(const Frame& f, std::shared_ptr<Item>* out, std::string* why) const {
    const VisionConfig& c = cfg_;
    const int32_t F = c.factor(), P = c.patch, M = c.merge;
    int64_t th = 0, tw = 0;
    lfm2_size(f.h, f.w, F, c.image_min_pixels, c.image_max_pixels, &th, &tw);
    const bool large = lfm2_too_large(f.h, f.w, P, F, c.image_max_pixels, c.max_pixels_tolerance);
    int32_t cols = 1, rows = 1;
    if (large && !(c.min_tiles == 1 && c.max_tiles == 1))
        lfm2_grid(f.h, f.w, c.min_tiles, c.max_tiles, c.tile, &cols, &rows);
    const bool tiled = cols * rows > 1;
    if (th / P * (tw / P) > std::min(c.max_patches, c.max_segment_patches)) {
        if (why) *why = "the picture's aspect ratio needs more patches than one encoder segment holds";
        return RAD_E_INVAL;
    }

    uint16_t lut[3][256];
    for (int ch = 0; ch < 3; ++ch)
        for (int v = 0; v < 256; ++v)
            lut[ch][v] = rad_f32_to_bf16(((float)v - 255.0f * c.mean[ch]) / (255.0f * c.stdv[ch]));

    auto it = std::make_shared<Item>();
    it->kind = Kind::Image;
    it->patch_dim = c.patch_dim();
    it->rope_1d = true;
    it->expand.push_back(c.tok_start);
    /* One segment of `frame`'s gh x gw patches from (x0, y0), its placeholders after `marker`. */
    auto append = [&](const Frame& frame, int x0, int y0, int gh, int gw, int32_t marker) {
        it->segments.push_back({ gh, gw, it->n_patch });
        it->n_patch += (int64_t)gh * gw;
        if (marker >= 0) it->expand.push_back(marker);
        it->seg_at.push_back((int32_t)it->expand.size());
        it->expand.insert(it->expand.end(), (size_t)gh * gw / (M * M), c.tok_image);
        const size_t at = it->pixels.size();
        it->pixels.resize(at + (size_t)gh * gw * it->patch_dim);
        uint16_t* dst = it->pixels.data() + at;
        for (int bh = 0; bh < gh / M; ++bh)
            for (int bw = 0; bw < gw / M; ++bw)
                for (int mh = 0; mh < M; ++mh)
                    for (int mw = 0; mw < M; ++mw)
                        for (int y = 0; y < P; ++y) {
                            const uint8_t* row = frame.rgb.data() +
                                ((size_t)(y0 + (bh * M + mh) * P + y) * frame.w +
                                 x0 + (bw * M + mw) * P) * 3;
                            for (int x = 0; x < 3 * P; ++x) *dst++ = lut[x % 3][row[x]];
                        }
    };
    if (tiled) {
        Frame big;
        resize_bicubic_torch(f, cols * c.tile, rows * c.tile, &big);
        const int g = c.tile / P;
        for (int32_t r = 0; r < rows; ++r)
            for (int32_t col = 0; col < cols; ++col)
                append(big, col * c.tile, r * c.tile, g, g,
                       tok_tiles_[(size_t)r * c.max_tiles + col]);
    }
    if (!tiled || c.use_thumbnail) {
        Frame small;
        resize_bicubic_torch(f, (int32_t)tw, (int32_t)th, &small);
        append(small, 0, 0, (int)(th / P), (int)(tw / P), tiled ? tok_thumbnail_ : -1);
    }
    it->expand.push_back(c.tok_end);
    it->n_seg = it->grid_t = (int32_t)it->segments.size();
    it->grid_h = it->segments[0].h;
    it->grid_w = it->segments[0].w;
    it->rows_h = it->grid_h / M;
    it->rows_w = it->grid_w / M;
    it->seg_rows = it->rows_h * it->rows_w;
    if (it->n_seg == 1) it->segments.clear();     /* one segment is the uniform form */
    content_key(*it, &it->key_hi, &it->key_lo);
    *out = std::move(it);
    return RAD_OK;
}

int uocr_mode(const VisionConfig& c, const ImageMode& mode, bool* crop, int32_t* base_size,
              int32_t* image_size, std::string* why) {
    auto no = [why](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    if (c.family != Family::UnlimitedOcr) return no("this model has one picture mode, not two");
    const int32_t F = c.factor();
    const bool cr = mode.kind == ImageMode::Base   ? false
                  : mode.kind == ImageMode::Gundam ? true
                  : mode.several                   ? false
                                                    : c.crop_mode;
    const int32_t bs = mode.base_size > 0 ? mode.base_size : c.base_size;
    const int32_t is = mode.image_size > 0 ? mode.image_size : c.image_size;
    if (cr) {
        /* The crops are the tile the checkpoint was trained at; the reference's image_size is
         * that same number, so a request may only restate it. */
        if (bs % F)
            return no("base_size " + std::to_string(bs) + " is not a whole number of merged "
                      "patches (" + std::to_string(F) + ")");
        if (bs < 4 * F)
            return no("base_size " + std::to_string(bs) + " is under two output rows");
        if (bs / c.patch != 64 && bs / c.patch != 40)
            return no("the DeepEncoder serves base grids 64 and 40 (base_size 1024 or 640); "
                      "base_size " + std::to_string(bs) + " is neither");
        if (mode.image_size > 0 && mode.image_size != c.tile)
            return no("crop mode cuts pictures into " + std::to_string(c.tile) + "-pixel tiles; "
                      "image_size " + std::to_string(mode.image_size) + " is not one");
        *crop = true; *base_size = bs; *image_size = c.tile;
        return RAD_OK;
    }
    if (is % F)
        return no("image_size " + std::to_string(is) + " is not a whole number of merged patches "
                  "(" + std::to_string(F) + ")");
    if (is < 4 * F) return no("image_size " + std::to_string(is) + " is under two output rows");
    if (is / c.patch != 64 && is / c.patch != 40)
        return no("the DeepEncoder serves base grids 64 and 40 (image_size 1024 or 640); "
                  "image_size " + std::to_string(is) + " is neither");
    if (mode.base_size > 0 && mode.base_size != bs)
        return no("the base mode pads a page to image_size, not base_size");
    *crop = false; *base_size = bs; *image_size = is;
    return RAD_OK;
}

int64_t uocr_tokens(const VisionConfig& c, const ImageMode& mode, int32_t cols, int32_t rows) {
    bool crop = false;
    int32_t bs = 0, is = 0;
    if (uocr_mode(c, mode, &crop, &bs, &is, nullptr) < 0) return -1;
    const int32_t P = c.patch, D = c.downsample;
    /* The global view: `g` lines of `g` features, each followed by a newline, then the view
     * separator (`[tok] * g + [tok]) * g + [tok]`). */
    const int32_t g = (crop ? bs : is) / P / D;
    int64_t t = (int64_t)(g + 1) * g + 1;
    /* The local views, when the picture is tiled: one line per feature row holding every tile's
     * row, a newline after it (`[tok] * q * cols + [tok]) * q * rows`). */
    if (crop && (int64_t)cols * rows > 1) {
        const int32_t q = c.tile / P / D;
        t += (int64_t)(q * cols + 1) * q * rows;
    }
    return t;
}

void pad_bicubic(const Frame& in, int32_t w, int32_t h, uint8_t fill, Frame* out) {
    /* ImageOps.pad: contain() fits the picture inside w x h at its own aspect (PIL's bicubic,
     * rounded as its round() does), then it is pasted centred on `fill`. */
    int32_t nw = w, nh = h;
    if (in.w > 0 && in.h > 0) {
        const double im_ratio = (double)in.w / (double)in.h;
        const double dst_ratio = (double)w / (double)h;
        if (im_ratio != dst_ratio) {
            if (im_ratio > dst_ratio) {
                nh = (int32_t)pyround((double)in.h / (double)in.w * (double)w);
                if (nh == 0) nh = 1;
            } else {
                nw = (int32_t)pyround((double)in.w / (double)in.h * (double)h);
                if (nw == 0) nw = 1;
            }
        }
    }
    const Frame* src = &in;
    Frame fitted;
    if (nw != in.w || nh != in.h) {
        resize_bicubic(in, nw, nh, &fitted);
        src = &fitted;
    }
    out->w = w;
    out->h = h;
    out->rgb.assign((size_t)w * (size_t)h * 3, fill);
    const int32_t x = nw != w ? (int32_t)pyround((double)(w - nw) * 0.5) : 0;
    const int32_t y = nw != w ? 0 : (int32_t)pyround((double)(h - nh) * 0.5);
    for (int32_t yy = 0; yy < src->h; ++yy)
        std::memcpy(out->rgb.data() + ((size_t)(y + yy) * (size_t)w + (size_t)x) * 3,
                    src->rgb.data() + (size_t)yy * (size_t)src->w * 3, (size_t)src->w * 3);
}

/* Unlimited-OCR's picture processor, the reference's infer() and infer_multi() exactly: the
 * global view is the picture ImageOps.pad-ed to base_size (in the base mode, to image_size,
 * squared first when that side is a tile or less), and in the gundam mode a picture larger than a
 * tile is cut the way dynamic_preprocess cuts it. Each view is staged as its feature grid plus
 * `downsample` patch columns nothing is read from, because the reference's layout puts a newline
 * after every line of features: one output row per downsample x downsample patches, and the last
 * column's rows are where those newlines go. The segments are, in the reference's order, the
 * local grid's lines (one per tile row; a line wider than a pass is cut into chunks of whole
 * tiles, its last chunk carrying the newline column), then the global view's grid, then the view
 * separator's single row. */
int Processor::image_unlimitedocr(const Frame& f, const ImageMode& mode,
                                  std::shared_ptr<Item>* out, std::string* why) const {
    const VisionConfig& c = cfg_;
    const int32_t P = c.patch, M = c.merge;
    bool crop = false;
    int32_t base_size = 0, image_size = 0;
    RAD_TRY(uocr_mode(c, mode, &crop, &base_size, &image_size, why));
    if (f.w <= 0 || f.h <= 0) {
        if (why) *why = "the picture has no size";
        return RAD_E_INVAL;
    }
    const int32_t q = c.tile / P;                  /* a tile's patch rows */
    int32_t cols = 1, rows = 1;
    const bool tiled = crop && (f.w > c.tile || f.h > c.tile);
    if (tiled) uocr_grid(f.h, f.w, c.min_tiles, c.max_tiles, c.tile, &cols, &rows);

    /* The global view. The base mode squares a page of a tile or less before padding it, as the
     * reference's multi-page path does. */
    Frame global;
    if (crop) {
        pad_bicubic(f, base_size, base_size, 127, &global);
    } else if (image_size <= c.tile) {
        /* The reference's multi-page path squares the page first, so a non-square page is
         * aspect-distorted by this resize, and the pad below is a no-op by construction (the
         * square already is the pad target). Square-first kept for reference parity. */
        Frame sq;
        resize_bicubic(f, image_size, image_size, &sq);
        pad_bicubic(sq, image_size, image_size, 127, &global);
    } else {
        pad_bicubic(f, image_size, image_size, 127, &global);
    }
    const int32_t g = (crop ? base_size : image_size) / P;
    if ((int64_t)g * (g + 4) > c.max_patches) {
        if (why) *why = "the global view's " + std::to_string((long long)g * (g + 4)) +
                        " patches are more than one encoder pass holds (" +
                        std::to_string(c.max_patches) + ")";
        return RAD_E_INVAL;
    }

    /* ToTensor then Normalize, in f32: (u / 255 - mean) / std, rounded to bf16. */
    uint16_t lut[3][256];
    uint32_t lut32[3 * 256];
    for (int ch = 0; ch < 3; ++ch)
        for (int v = 0; v < 256; ++v) {
            const float p = (float)v / 255.0f;
            lut[ch][v] = rad_f32_to_bf16((p - c.mean[ch]) / c.stdv[ch]);
            lut32[ch * 256 + v] = lut[ch][v];
        }
    const bool simd = mm_use_avx2();

    auto it = std::make_shared<Item>();
    it->kind = Kind::Image;
    it->patch_dim = c.patch_dim();
    it->rope_1d = true;
    it->patch_raster = true;
    int64_t seg_row = 0;
    /* One segment of `gh` x `gw` patches from (x0, y0) of `src`, of which the first `real_w`
     * patches a row carry pixels; the rest are the newline column. */
    auto append = [&](const Frame& src, int32_t x0, int32_t y0, int32_t gh, int32_t gw,
                      int32_t real_w, bool sep = false) {
        it->segments.push_back({ gh, gw, it->n_patch, sep });
        it->seg_at.push_back((int32_t)seg_row);
        it->expand.insert(it->expand.end(), (size_t)(gh / M) * (size_t)(gw / M), c.tok_image);
        seg_row += (int64_t)(gh / M) * (gw / M);
        it->n_patch += (int64_t)gh * gw;
        const size_t at = it->pixels.size();
        it->pixels.resize(at + (size_t)gh * (size_t)gw * (size_t)it->patch_dim, 0);
        uint16_t* dst = it->pixels.data() + at;
        /* Each patch row writes its own disjoint run of dst, in the same order, so splitting the
         * rows across threads is bit-exact. */
        run_rows(gh, [&](int32_t py) {
            for (int32_t px = 0; px < real_w; ++px) {
                uint16_t* pp = dst + ((size_t)py * (size_t)gw + (size_t)px) * (size_t)it->patch_dim;
                const int32_t py0 = y0 + py * P, px0 = x0 + px * P;
                for (int y = 0; y < P; ++y) {
                    const uint8_t* row = src.rgb.data() +
                        ((size_t)(py0 + y) * (size_t)src.w + (size_t)px0) * 3;
#ifdef RAD_MM_HAVE_AVX2
                    if (simd) {
                        patch_row(row, pp + (size_t)y * P, P, (size_t)P * P, lut32, &lut[0][0]);
                        continue;
                    }
#endif
                    (void)simd;
                    for (int ch = 0; ch < 3; ++ch) {
                        const uint8_t* r = row + (size_t)ch;
                        uint16_t* q = pp + ((size_t)ch * P + (size_t)y) * P;
                        for (int x = 0; x < P; ++x) q[x] = lut[ch][r[(size_t)x * 3]];
                    }
                }
            }
        });
    };

    if (tiled) {
        Frame big;
        resize_bicubic(f, cols * c.tile, rows * c.tile, &big);
        /* A chunk of a line must fit one pass: q rows of q patches a tile, plus the newline
         * column its last chunk holds. */
        int64_t per_row = c.max_patches / q;
        const int64_t chunk = per_row > 4 ? (per_row - 4) / q : 0;
        if (chunk < 1) {
            if (why) *why = "one encoder pass holds no line of " + std::to_string(q) +
                            "-patch tiles (" + std::to_string(c.max_patches) + " patches)";
            return RAD_E_INVAL;
        }
        for (int32_t r = 0; r < rows; ++r)
            for (int32_t c0 = 0; c0 < cols; c0 += (int32_t)chunk) {
                const int32_t n = std::min<int32_t>((int32_t)chunk, cols - c0);
                const bool last = c0 + n == cols;
                append(big, c0 * c.tile, r * c.tile, q, q * n + (last ? 4 : 0), q * n);
            }
    }
    append(global, 0, 0, g, g + 4, g);
    /* The view separator: one output row, from a grid of exactly one merged cell and no pixels. */
    append(global, 0, 0, M, M, 0, /*sep=*/true);

    it->n_seg = (int32_t)it->segments.size();
    it->grid_t = 1;
    it->grid_h = it->segments[0].h;
    it->grid_w = it->segments[0].w;
    it->rows_h = it->grid_h / M;
    it->rows_w = it->grid_w / M;
    it->seg_rows = it->rows_h * it->rows_w;
    content_key(*it, &it->key_hi, &it->key_lo);
    *out = std::move(it);
    return RAD_OK;
}

/* GLM-5.3's image processor (transformers' Glm5NextImageProcessor, torchvision backend): the
 * picture, resized only when its size changes, in the top left of a canvas of zero bytes, which
 * the normalisation turns into each channel's -mean / std as it does the reference's padding. */
int Processor::image_glm5next(const Frame& f, std::shared_ptr<Item>* out, std::string* why) const {
    const VisionConfig& c = cfg_;
    int64_t h = 0, w = 0, ch = 0, cw = 0;
    RAD_TRY(glm5next_size(f.h, f.w, c.temporal, c.factor(), c.min_image_tokens,
                          c.max_image_tokens, &h, &w, &ch, &cw, why));
    Frame r;
    const Frame* src = &f;
    if (ch != f.h || cw != f.w) {
        resize_bicubic_torch(f, (int32_t)cw, (int32_t)ch, &r);
        src = &r;
    }
    Frame canvas;
    canvas.w = (int32_t)w;
    canvas.h = (int32_t)h;
    canvas.rgb.assign((size_t)w * h * 3, 0);
    for (int64_t y = 0; y < ch; ++y)
        std::memcpy(canvas.rgb.data() + (size_t)y * w * 3, src->rgb.data() + (size_t)y * cw * 3,
                    (size_t)cw * 3);
    auto it = std::make_shared<Item>();
    it->kind = Kind::Image;
    it->grid_t = 1;
    it->grid_h = (int32_t)(h / c.patch);
    it->grid_w = (int32_t)(w / c.patch);
    it->rope_1d = true;
    finish_item(c, it.get());
    if (it->n_patch > c.max_patches) {
        if (why) *why = "the image resizes to more patches than one encoder pass holds";
        return RAD_E_INVAL;
    }
    std::vector<const Frame*> frames((size_t)c.temporal, &canvas);
    patchify(c, frames, it.get());
    one_run(it.get(), c.tok_start, it->seg_rows, c.tok_image, c.tok_end);
    *out = std::move(it);
    return RAD_OK;
}

/* The official PIL processor: cap the longest edge without enlarging, floor the aspect-ratio
 * resize, then ceil both sides to patch * merge. Patch rows are raster order, (channel, y, x).
 * The merger needs that order; merge-block ordering here would silently rotate the wrong grid. */
int Processor::image_pixtral(const Frame& f, std::shared_ptr<Item>* out, std::string* why) const {
    const VisionConfig& c = cfg_;
    if (!out || f.h <= 0 || f.w <= 0 || (int64_t)f.h * f.w > c.max_source_pixels ||
        f.rgb.size() != (uint64_t)f.h * f.w * 3) {
        if (why) *why = "invalid Pixtral RGB frame or source pixel limit exceeded";
        return RAD_E_INVAL;
    }
    const double ratio =
        std::max((double)f.h / c.pixtral_longest_edge, (double)f.w / c.pixtral_longest_edge);
    int64_t h = f.h, w = f.w;
    if (ratio > 1) {
        h = (int64_t)std::floor(h / ratio);
        w = (int64_t)std::floor(w / ratio);
    }
    const int64_t factor = c.factor();
    h = ((h + factor - 1) / factor) * factor;
    w = ((w + factor - 1) / factor) * factor;
    if (h <= 0 || w <= 0 || (h / c.patch) * (w / c.patch) > c.max_patches) {
        if (why) *why = "Pixtral image is empty after resize or exceeds one encoder pass";
        return RAD_E_INVAL;
    }
    Frame resized;
    resize_bicubic(f, (int32_t)w, (int32_t)h, &resized);
    auto it = std::make_shared<Item>();
    it->grid_t = 1;
    it->grid_h = (int32_t)(h / c.patch);
    it->grid_w = (int32_t)(w / c.patch);
    it->rope_1d = true;
    it->patch_raster = true;
    finish_item(c, it.get());
    uint16_t lut[3][256];
    for (int ch = 0; ch < 3; ++ch)
        for (int v = 0; v < 256; ++v) {
            const float scaled = (float)((double)v * (1.0 / 255));
            lut[ch][v] = rad_f32_to_bf16((scaled - c.mean[ch]) / c.stdv[ch]);
        }
    it->pixels.resize((size_t)it->n_patch * it->patch_dim);
    auto* dst = it->pixels.data();
    for (int py = 0; py < it->grid_h; ++py)
        for (int px = 0; px < it->grid_w; ++px)
            for (int ch = 0; ch < 3; ++ch)
                for (int y = 0; y < c.patch; ++y)
                    for (int x = 0; x < c.patch; ++x)
                        *dst++ = lut[ch][resized.rgb[
                            ((size_t)(py * c.patch + y) * w + px * c.patch + x) * 3 + ch]];
    it->seg_at.push_back(0);
    for (int y = 0; y < it->rows_h; ++y) {
        for (int x = 0; x < it->rows_w; ++x) {
            it->row_tokens.push_back((int32_t)it->expand.size());
            it->expand.push_back(c.tok_image);
        }
        it->expand.push_back(y + 1 == it->rows_h ? c.tok_end : c.tok_break);
    }
    content_key(*it, &it->key_hi, &it->key_lo);
    *out = std::move(it);
    return RAD_OK;
}

/* NVIDIA's Nemotron-H Omni image processor: one tile, sized by radio_size, resized with torch's
 * antialiased bicubic on float pixels (none when the size already fits). */
int Processor::image_radio(const Frame& f, std::shared_ptr<Item>* out, std::string* why) const {
    const VisionConfig& c = cfg_;
    int32_t wp = 0, hp = 0;
    radio_size(f.w, f.h, c, &wp, &hp);
    std::vector<float> px;
    resize_bicubic_aa_f32(f, wp * c.patch, hp * c.patch, &px);
    auto it = std::make_shared<Item>();
    it->kind   = Kind::Image;
    it->grid_t = 1;
    it->grid_h = hp;
    it->grid_w = wp;
    it->rope_1d = true;
    finish_item_radio(c, it.get());
    if (it->n_patch > c.max_patches) {
        if (why) *why = "the image resizes to more patches than one encoder pass holds";
        return RAD_E_INVAL;
    }
    patchify_radio(c, { &px }, false, it.get());
    one_run(it.get(), c.tok_start, it->seg_rows, c.tok_image, c.tok_end);
    *out = std::move(it);
    return RAD_OK;
}

/* Its video path: every frame at the video tile (radio_video_size of the first), `tubelet`
 * frames a segment (the last repeated to fill one), each run labelled as the reference processor
 * labels a tubelet -- "Frame i sampled at T seconds and frame i+1 sampled at T' seconds: ", T the
 * frame's source index times int(1000 / fps) ms -- the runs joined by a newline. No pruning (EVS):
 * every tubelet keeps all its rows. */
int Processor::video_radio(const Video& v, std::shared_ptr<Item>* out, std::string* why) const {
    const VisionConfig& c = cfg_;
    if (v.frames.empty()) { if (why) *why = "the video decoded to no frames"; return RAD_E_INVAL; }
    int32_t wp = 0, hp = 0;
    radio_video_size(v.frames[0].w, v.frames[0].h, c, &wp, &hp);
    const int32_t T = c.radio_tubelet;
    const int64_t nf = (int64_t)v.frames.size();
    auto it = std::make_shared<Item>();
    it->kind   = Kind::Video;
    it->grid_t = (int32_t)((nf + T - 1) / T);
    it->grid_h = hp;
    it->grid_w = wp;
    it->rope_1d = true;
    finish_item_radio(c, it.get());
    if (c.radio_prefix + (int64_t)hp * wp > c.max_patches) {
        if (why) *why = "a video frame resizes to more patches than one encoder pass holds";
        return RAD_E_INVAL;
    }
    if (it->n_patch > c.max_video_patches) {
        if (why) *why = "the video is over the server's patch budget for one video";
        return RAD_E_INVAL;
    }
    std::vector<std::vector<float>> resized((size_t)nf);
    for (int64_t i = 0; i < nf; ++i)
        resize_bicubic_aa_f32(v.frames[(size_t)i], wp * c.patch, hp * c.patch, &resized[(size_t)i]);
    std::vector<const std::vector<float>*> frames;
    for (int64_t i = 0; i < (int64_t)it->grid_t * T; ++i)
        frames.push_back(&resized[(size_t)std::min(i, nf - 1)]);
    patchify_radio(c, frames, true, it.get());
    content_key(*it, &it->key_hi, &it->key_lo);

    const int64_t frame_ms = v.info.fps > 0.0 ? (int64_t)(1000.0 / v.info.fps) : 0;
    for (int32_t g = 0; g < it->grid_t; ++g) {
        std::string label = g > 0 ? "\n" : "";
        for (int32_t j = 0; j < T; ++j) {
            const int64_t fi = (int64_t)g * T + j;
            if (fi >= nf) break;
            char part[96];
            if (frame_ms > 0)
                std::snprintf(part, sizeof part, "%s%s %lld sampled at %.2f seconds",
                              j ? " and " : "", j ? "frame" : "Frame", (long long)(fi + 1),
                              (double)(v.index[(size_t)fi] * frame_ms) / 1000.0);
            else
                std::snprintf(part, sizeof part, "%s%s %lld", j ? " and " : "",
                              j ? "frame" : "Frame", (long long)(fi + 1));
            label += part;
        }
        label += ": ";
        const std::vector<int32_t> tt = encode_text_ ? encode_text_(label) : std::vector<int32_t>{};
        if (tt.empty()) {
            if (why) *why = "the tokenizer produced nothing for a video frame label";
            return RAD_E_STATE;
        }
        it->expand.insert(it->expand.end(), tt.begin(), tt.end());
        it->expand.push_back(c.tok_start);
        it->seg_at.push_back((int32_t)it->expand.size());
        it->expand.insert(it->expand.end(), (size_t)it->seg_rows, c.tok_image);
        it->expand.push_back(c.tok_end);
    }
    *out = std::move(it);
    return RAD_OK;
}

/* Meta's Muse Glimmer image processor (transformers' MuseGlimmerImageProcessor, Torchvision
 * backend): the picture at muse_size's grid under the token cap, resized with torchvision's
 * antialiased Lanczos on the uint8 pixels, values (u / 255 - mean) / std in bf16. One segment. */
int Processor::image_muse(const Frame& f, std::shared_ptr<Item>* out, std::string* why) const {
    const VisionConfig& c = cfg_;
    int64_t h = 0, w = 0;
    RAD_TRY(muse_size(f.h, f.w, c.factor(), c.max_image_tokens, &h, &w, why));
    Frame r;
    resize_lanczos_torch(f, (int32_t)w, (int32_t)h, &r);
    auto it = std::make_shared<Item>();
    it->kind = Kind::Image;
    it->grid_t = 1;
    it->grid_h = (int32_t)(h / c.patch);
    it->grid_w = (int32_t)(w / c.patch);
    finish_item(c, it.get());
    if (it->n_patch > c.max_patches) {
        if (why) *why = "the image resizes to more patches than one encoder pass holds";
        return RAD_E_INVAL;
    }
    /* An image fills the temporal depth by repetition, as the reference's expand does. */
    std::vector<const Frame*> frames((size_t)c.temporal, &r);
    patchify_muse(c, frames, it.get());
    one_run(it.get(), c.tok_start, it->seg_rows, c.tok_image, c.tok_end);
    *out = std::move(it);
    return RAD_OK;
}

/* Its video processor: every frame at the first frame's muse_size grid under the per-frame token
 * cap (the reference resizes the whole stack once), the frames padded to whole temporal groups by
 * repeating the last, and per group one "Time: T s" stamp -- the group's first frame's time at
 * the stream's rate, 24 fps when the stream states none -- between <|vid_start|> and
 * <|vid_end|>, the groups separated by <|vid_frame_separator|>. */
int Processor::video_muse(const Video& v, std::shared_ptr<Item>* out, std::string* why) const {
    const VisionConfig& c = cfg_;
    if (v.frames.empty()) { if (why) *why = "the video decoded to no frames"; return RAD_E_INVAL; }
    int64_t h = 0, w = 0;
    RAD_TRY(muse_size(v.frames[0].h, v.frames[0].w, c.factor(), c.max_frame_rows, &h, &w, why));

    /* Pad the frames and their indices to whole temporal groups by repeating the last. */
    std::vector<const Frame*> frames;
    std::vector<int64_t> idx = v.index;
    for (const Frame& f : v.frames) frames.push_back(&f);
    while (frames.size() % (size_t)c.temporal != 0 || frames.size() < (size_t)c.temporal) {
        frames.push_back(frames.back());
        idx.push_back(idx.empty() ? 0 : idx.back());
    }

    auto it = std::make_shared<Item>();
    it->kind = Kind::Video;
    it->grid_t = (int32_t)(frames.size() / (size_t)c.temporal);
    it->grid_h = (int32_t)(h / c.patch);
    it->grid_w = (int32_t)(w / c.patch);
    finish_item(c, it.get());
    if ((int64_t)it->grid_h * it->grid_w > c.max_patches) {
        if (why) *why = "a video frame resizes to more patches than one encoder pass holds";
        return RAD_E_INVAL;
    }
    if (it->n_patch > c.max_video_patches) {
        if (why) *why = "the video is over the server's patch budget for one video";
        return RAD_E_INVAL;
    }

    /* Resize each distinct frame once; padding repeats share the resized copy. */
    std::vector<Frame> resized(v.frames.size());
    for (size_t i = 0; i < v.frames.size(); ++i)
        resize_lanczos_torch(v.frames[i], (int32_t)w, (int32_t)h, &resized[i]);
    std::vector<const Frame*> rframes;
    for (const Frame* f : frames) rframes.push_back(&resized[(size_t)(f - v.frames.data())]);
    patchify_muse(c, rframes, it.get());
    content_key(*it, &it->key_hi, &it->key_lo);

    const double vfps = v.info.fps > 0.0 ? v.info.fps : 24.0;
    it->expand.push_back(c.tok_video_start);
    for (int32_t g = 0; g < it->grid_t; ++g) {
        char ts[64];
        std::snprintf(ts, sizeof ts, "Time: %.1fs", (double)idx[(size_t)g * c.temporal] / vfps);
        const std::vector<int32_t> tt = encode_text_ ? encode_text_(ts) : std::vector<int32_t>{};
        if (tt.empty()) {
            if (why) *why = "the tokenizer produced nothing for a video timestamp";
            return RAD_E_STATE;
        }
        it->expand.insert(it->expand.end(), tt.begin(), tt.end());
        it->seg_at.push_back((int32_t)it->expand.size());
        it->expand.insert(it->expand.end(), (size_t)it->seg_rows, c.tok_video);
        it->expand.push_back(g + 1 == it->grid_t ? c.tok_video_end : c.tok_video_sep);
    }
    *out = std::move(it);
    return RAD_OK;
}

int Processor::image_from(const Frame& f, std::shared_ptr<Item>* out, std::string* why,
                          const ImageMode& mode) const {
    const VisionConfig& c = cfg_;
    if (mode.kind != ImageMode::Default && c.family != Family::UnlimitedOcr) {
        if (why) *why = "this model has one picture mode; images_config names one of two";
        return RAD_E_UNSUPPORTED;
    }
    if (c.family == Family::Gemma4) return image_gemma4(f, c.image_soft_tokens, out, why);
    if (c.family == Family::Pixtral) return image_pixtral(f, out, why);
    if (c.family == Family::Lfm2) return image_lfm2(f, out, why);
    if (c.family == Family::Glm5Next) return image_glm5next(f, out, why);
    if (c.family == Family::Radio) return image_radio(f, out, why);
    if (c.family == Family::MuseGlimmer) return image_muse(f, out, why);
    if (c.family == Family::UnlimitedOcr) return image_unlimitedocr(f, mode, out, why);
    int64_t h = 0, w = 0;
    RAD_TRY(smart_resize_image(f.h, f.w, c.factor(), c.image_min_pixels, c.image_max_pixels,
                               &h, &w, why));
    Frame r;
    resize_bicubic(f, (int32_t)w, (int32_t)h, &r);
    auto it = std::make_shared<Item>();
    it->kind = Kind::Image;
    it->grid_t = 1;
    it->grid_h = (int32_t)(h / c.patch);
    it->grid_w = (int32_t)(w / c.patch);
    finish_item(c, it.get());
    if (it->n_patch > c.max_patches) {
        if (why) *why = "the image resizes to more patches than one encoder pass holds";
        return RAD_E_INVAL;
    }
    /* An image fills the temporal depth by repetition. */
    std::vector<const Frame*> frames((size_t)c.temporal, &r);
    patchify(c, frames, it.get());
    one_run(it.get(), c.tok_start, it->seg_rows, c.tok_image, c.tok_end);
    *out = std::move(it);
    return RAD_OK;
}

int Processor::image_gemma4(const Frame& f, int32_t soft, std::shared_ptr<Item>* out,
                            std::string* why) const {
    const VisionConfig& c = cfg_;
    int64_t h = 0, w = 0;
    RAD_TRY(gemma4_size(f.h, f.w, c.patch, c.merge, (int64_t)soft * c.merge * c.merge, &h, &w,
                        why));
    Frame r;
    const Frame* src = &f;
    if (h != f.h || w != f.w) {
        resize_bicubic(f, (int32_t)w, (int32_t)h, &r);
        src = &r;
    }
    auto it = std::make_shared<Item>();
    it->kind = Kind::Image;
    it->grid_t = 1;
    it->grid_h = (int32_t)(h / c.patch);
    it->grid_w = (int32_t)(w / c.patch);
    finish_item(c, it.get());
    if (it->n_patch > c.max_patches) {
        if (why) *why = "the image's soft-token budget is more patches than one encoder pass holds";
        return RAD_E_INVAL;
    }
    patchify_gemma4(c, { src }, /*video=*/false, it.get());
    one_run(it.get(), c.tok_start, it->seg_rows, c.tok_image, c.tok_end);
    *out = std::move(it);
    return RAD_OK;
}

int Processor::video_gemma4(const Video& v, std::shared_ptr<Item>* out, std::string* why) const {
    const VisionConfig& c = cfg_;
    if (v.frames.empty()) { if (why) *why = "the video decoded to no frames"; return RAD_E_INVAL; }
    int64_t h = 0, w = 0;
    RAD_TRY(gemma4_size(v.frames[0].h, v.frames[0].w, c.patch, c.merge,
                        (int64_t)c.video_soft_tokens * c.merge * c.merge, &h, &w, why));
    auto it = std::make_shared<Item>();
    it->kind = Kind::Video;
    it->grid_t = (int32_t)v.frames.size();
    it->grid_h = (int32_t)(h / c.patch);
    it->grid_w = (int32_t)(w / c.patch);
    finish_item(c, it.get());
    if ((int64_t)it->grid_h * it->grid_w > c.max_patches) {
        if (why) *why = "a video frame's soft-token budget is more patches than one encoder pass "
                        "holds";
        return RAD_E_INVAL;
    }
    std::vector<Frame> resized(v.frames.size());
    std::vector<const Frame*> rf;
    for (size_t i = 0; i < v.frames.size(); ++i) {
        if (v.frames[i].h == h && v.frames[i].w == w) { rf.push_back(&v.frames[i]); continue; }
        resize_bicubic(v.frames[i], (int32_t)w, (int32_t)h, &resized[i]);
        rf.push_back(&resized[i]);
    }
    patchify_gemma4(c, rf, /*video=*/true, it.get());
    content_key(*it, &it->key_hi, &it->key_lo);
    /* One <boi> placeholders <eoi> run per frame, back to back: the reference's visual-only form
     * (add_timestamps off). */
    for (int32_t g = 0; g < it->grid_t; ++g) {
        it->expand.push_back(c.tok_start);
        it->seg_at.push_back((int32_t)it->expand.size());
        it->expand.insert(it->expand.end(), (size_t)it->seg_rows, c.tok_video);
        it->expand.push_back(c.tok_end);
    }
    *out = std::move(it);
    return RAD_OK;
}

int Processor::video_from(const Video& v, std::shared_ptr<Item>* out, std::string* why) const {
    const VisionConfig& c = cfg_;
    if (c.family == Family::Gemma4) return video_gemma4(v, out, why);
    if (c.family == Family::Radio) return video_radio(v, out, why);
    if (c.family == Family::MuseGlimmer) return video_muse(v, out, why);
    if (c.family == Family::Glm5Next || c.family == Family::Pixtral) {
        if (why) *why = "this model's processor takes pictures, not video";
        return RAD_E_UNSUPPORTED;
    }
    if (v.frames.empty()) { if (why) *why = "the video decoded to no frames"; return RAD_E_INVAL; }
    /* Pad the frames and their indices to whole temporal groups by repeating the last; a clip
     * shorter than one group becomes one group. */
    std::vector<const Frame*> frames;
    std::vector<int64_t> idx = v.index;
    for (const Frame& f : v.frames) frames.push_back(&f);
    while (frames.size() % (size_t)c.temporal != 0 || frames.size() < (size_t)c.temporal) {
        frames.push_back(frames.back());
        idx.push_back(idx.back());
    }
    const int64_t nf = (int64_t)v.frames.size() < c.temporal ? c.temporal
                                                              : (int64_t)v.frames.size();

    /* Per-frame cap: at most `max_frame_rows` output rows' worth of pixels per frame and at most
     * the video budget's even share, but never under the band's floor. */
    int64_t max_px = c.video_max_pixels;
    if (c.max_frame_rows > 0) {
        const int64_t f2 = (int64_t)c.factor() * c.factor();
        const int64_t cap = (int64_t)c.max_frame_rows * f2;
        const int64_t share = c.video_max_pixels / nf;
        const int64_t per = std::max<int64_t>(std::min(cap, share),
                                              (int64_t)((double)c.video_min_pixels * 1.05));
        max_px = per * nf;
    }
    int64_t h = 0, w = 0;
    RAD_TRY(smart_resize_video(nf, v.frames[0].h, v.frames[0].w, c.temporal, c.factor(),
                               c.video_min_pixels, max_px, &h, &w, why));

    auto it = std::make_shared<Item>();
    it->kind = Kind::Video;
    it->grid_t = (int32_t)(frames.size() / (size_t)c.temporal);
    it->grid_h = (int32_t)(h / c.patch);
    it->grid_w = (int32_t)(w / c.patch);
    finish_item(c, it.get());
    if ((int64_t)it->grid_h * it->grid_w > c.max_patches) {
        if (why) *why = "a video frame resizes to more patches than one encoder pass holds";
        return RAD_E_INVAL;
    }
    if (it->n_patch > c.max_video_patches) {
        if (why) *why = "the video is over the server's patch budget for one video";
        return RAD_E_INVAL;
    }

    /* Resize each distinct frame once; padding repeats share the resized copy. */
    std::vector<Frame> resized(v.frames.size());
    for (size_t i = 0; i < v.frames.size(); ++i)
        resize_bicubic(v.frames[i], (int32_t)w, (int32_t)h, &resized[i]);
    std::vector<const Frame*> rframes;
    for (const Frame* f : frames) rframes.push_back(&resized[(size_t)(f - v.frames.data())]);
    patchify(c, rframes, it.get());
    content_key(*it, &it->key_hi, &it->key_lo);

    /* Timestamped expansion: per temporal group, "<T seconds>" then pads between vision_start and
     * vision_end, T being the mean of the group's first and last frame times. It replaces the
     * template's whole <|vision_start|><|video_pad|><|vision_end|>, as the model's own processor
     * does (transformers 4.57, qwen-vl-utils). transformers 5.x replaces only the pad and leaves
     * an outer start/end pair the model was not trained with. */
    const double vfps = v.info.fps > 0.0 ? v.info.fps : 24.0;
    for (int32_t g = 0; g < it->grid_t; ++g) {
        const double t0 = (double)idx[(size_t)g * c.temporal] / vfps;
        const double t1 = (double)idx[(size_t)g * c.temporal + c.temporal - 1] / vfps;
        char ts[64];
        std::snprintf(ts, sizeof ts, "<%.1f seconds>", (t0 + t1) / 2.0);
        const std::vector<int32_t> tt = encode_text_ ? encode_text_(ts) : std::vector<int32_t>{};
        if (tt.empty()) {
            if (why) *why = "the tokenizer produced nothing for a video timestamp";
            return RAD_E_STATE;
        }
        it->expand.insert(it->expand.end(), tt.begin(), tt.end());
        it->expand.push_back(c.tok_start);
        it->seg_at.push_back((int32_t)it->expand.size());
        it->expand.insert(it->expand.end(), (size_t)it->seg_rows, c.tok_video);
        it->expand.push_back(c.tok_end);
    }
    *out = std::move(it);
    return RAD_OK;
}

int Processor::image(const uint8_t* bytes, size_t n, int32_t image_tokens,
                     std::shared_ptr<Item>* out, std::string* why, int64_t max_pixels,
                     const ImageMode& mode) const {
    if (mode.kind != ImageMode::Default && cfg_.family != Family::UnlimitedOcr) {
        if (why) *why = "this model has one picture mode; images_config names one of two";
        return RAD_E_UNSUPPORTED;
    }
    if (image_tokens != 0) {
        if (cfg_.family == Family::Glm5Next) {
            if (why) *why = "this model's image token budget is the server's (--image-tokens), "
                            "not a request's";
            return RAD_E_UNSUPPORTED;
        }
        if (cfg_.family != Family::Gemma4) {
            if (why) *why = "this model's processor sizes an image by its pixels, not by a token "
                            "budget";
            return RAD_E_UNSUPPORTED;
        }
        if (!soft_budget_ok(image_tokens) ||
            (int64_t)image_tokens * cfg_.merge * cfg_.merge > cfg_.max_patches) {
            if (why) *why = "the image token budget must be one of 70, 140, 280, 560, 1120 and "
                            "fit one encoder pass (" + std::to_string(cfg_.max_patches) +
                            " patches)";
            return RAD_E_INVAL;
        }
    }
    Frame f;
    RAD_TRY(decode_image(bytes, n, cfg_.max_source_pixels, &f, why));
    if (max_pixels > 0 && (int64_t)f.w * f.h > max_pixels) {
        Frame capped;
        cap_area(f, max_pixels, &capped);
        f = std::move(capped);
    }
    if (cfg_.family == Family::Gemma4)
        return image_gemma4(f, image_tokens ? image_tokens : cfg_.image_soft_tokens, out, why);
    return image_from(f, out, why, mode);
}

namespace {
/* A picture's sides for a video encoder: the longer one scaled to at most `max_side`, aspect
 * kept, then each floored to a multiple of `ratio`, at least one multiple. */
void encoder_sides(int32_t fw, int32_t fh, int32_t ratio, int32_t max_side, int32_t* w,
                   int32_t* h) {
    const int32_t longest = std::max(fw, fh);
    const double  scale = longest > max_side ? (double)max_side / (double)longest : 1.0;
    auto snap = [&](int32_t s) {
        const int64_t v = (int64_t)std::floor(s * scale) / ratio * ratio;
        return (int32_t)std::min<int64_t>(std::max<int64_t>(v, ratio), max_side / ratio * ratio);
    };
    *w = snap(fw);
    *h = snap(fh);
}
}  /* namespace */

int Processor::encoder_image(const uint8_t* bytes, size_t n, int32_t ratio, int32_t max_side,
                             Frame* out, std::string* why) const {
    if (ratio < 1 || max_side < ratio) {
        if (why) *why = "the encoder's side ratio and maximum are not usable";
        return RAD_E_INVAL;
    }
    /* image()'s decode guard. The encoder runs a picture whole, not in ComfyUI's 256-pixel tiles,
     * so a larger one is shrunk to its bound. */
    Frame f;
    RAD_TRY(decode_image(bytes, n, cfg_.max_source_pixels, &f, why));
    int32_t w = 0, h = 0;
    encoder_sides(f.w, f.h, ratio, max_side, &w, &h);
    if (w == f.w && h == f.h) {
        *out = std::move(f);
        return RAD_OK;
    }
    resize_lanczos_rgb(f, w, h, out);
    return RAD_OK;
}

int Processor::encoder_image_canvas(const uint8_t* bytes, size_t n, int32_t w, int32_t h,
                                    bool cover, Frame* out, std::string* why) const {
    if (w < 1 || h < 1) {
        if (why) *why = "the keyframe canvas is empty";
        return RAD_E_INVAL;
    }
    /* image()'s decode guard. */
    Frame f;
    RAD_TRY(decode_image(bytes, n, cfg_.max_source_pixels, &f, why));
    if (!cover) {
        /* A stretch, ComfyUI's first-frame resize. */
        if (w == f.w && h == f.h) { *out = std::move(f); return RAD_OK; }
        resize_lanczos_rgb(f, w, h, out);
        return RAD_OK;
    }
    /* Scaled to cover the canvas, aspect kept, then cropped at the centre: ComfyUI's last-frame
     * resize. */
    const double s = std::max((double)w / (double)f.w, (double)h / (double)f.h);
    int32_t cw = (int32_t)std::llround((double)f.w * s);
    int32_t ch = (int32_t)std::llround((double)f.h * s);
    if (cw < w) cw = w;
    if (ch < h) ch = h;
    Frame big;
    resize_lanczos_rgb(f, cw, ch, &big);
    const int32_t x0 = (cw - w) / 2, y0 = (ch - h) / 2;
    out->w = w;
    out->h = h;
    out->rgb.resize((size_t)w * h * 3);
    for (int32_t y = 0; y < h; ++y)
        for (int32_t x = 0; x < w * 3; ++x)
            out->rgb[(size_t)y * w * 3 + x] =
                big.rgb[(size_t)(y0 + y) * cw * 3 + (size_t)(x0 * 3 + x)];
    return RAD_OK;
}

namespace {
/* The frames a reference video keeps, spread over the stream: the most, within the stream and
 * `max_frames`, that is 5 modulo 17, ComfyUI's trim; a shorter stream keeps what it has. */
struct VaeSampleArgs { int32_t max_frames; };
std::vector<int64_t> vae_sample_cb(const VideoInfo& info, const void* user) {
    const VaeSampleArgs* a = static_cast<const VaeSampleArgs*>(user);
    const int64_t avail = info.n_frames > 0 ? info.n_frames : 1;
    const int64_t cap = a->max_frames > 0 ? std::min<int64_t>(a->max_frames, avail) : avail;
    int64_t want = cap - (((cap - 5) % 17) + 17) % 17;   /* the largest v <= cap, v % 17 == 5 */
    if (want < 5) want = std::min<int64_t>(cap, 5);
    const int64_t take = std::max<int64_t>(1, std::min(want, avail));
    std::vector<int64_t> idx((size_t)take);
    for (int64_t i = 0; i < take; ++i)
        idx[(size_t)i] = take > 1 ? (i * (avail - 1)) / (take - 1) : 0;
    return idx;
}
}  /* namespace */

int Processor::encoder_video(const uint8_t* bytes, size_t n, int32_t ratio, int32_t max_side,
                             int32_t max_frames, std::vector<Frame>* out, std::string* why) const {
    if (ratio < 1 || max_side < ratio) {
        if (why) *why = "the encoder's side ratio and maximum are not usable";
        return RAD_E_INVAL;
    }
    /* image()'s decode guard. */
    Video v;
    const VaeSampleArgs a{ max_frames };
    RAD_TRY(decode_video(bytes, n, vae_sample_cb, &a, cfg_.max_source_pixels, &v, why));
    if (v.frames.empty()) {
        if (why) *why = "the video holds no frames";
        return RAD_E_INVAL;
    }
    int32_t w = 0, h = 0;
    encoder_sides(v.frames.front().w, v.frames.front().h, ratio, max_side, &w, &h);
    out->clear();
    out->reserve(v.frames.size());
    for (const Frame& f : v.frames) {
        Frame r;
        resize_lanczos_rgb(f, w, h, &r);
        out->push_back(std::move(r));
    }
    return RAD_OK;
}

int Processor::encoder_audio(const uint8_t* bytes, size_t n, int32_t rate, int64_t max_samples,
                             std::vector<float>* out, std::string* why) const {
    return decode_audio(bytes, n, rate, max_samples, out, why);
}

namespace {
struct SampleArgs { double fps; int32_t lo, hi; bool gemma4; int32_t temporal; };
std::vector<int64_t> sample_cb(const VideoInfo& info, const void* user) {
    const SampleArgs* a = static_cast<const SampleArgs*>(user);
    return a->gemma4 ? sample_frames_gemma4(info, a->fps, a->hi)
                     : sample_frames(info, a->fps, a->lo, a->hi);
}
std::vector<int64_t> muse_sample_cb(const VideoInfo& info, const void* user) {
    const SampleArgs* a = static_cast<const SampleArgs*>(user);
    return sample_frames_muse(info, a->fps, a->temporal, a->hi);
}
}  /* namespace */

int Processor::video(const uint8_t* bytes, size_t n, std::shared_ptr<Item>* out,
                     std::string* why) const {
    if (cfg_.family == Family::Lfm2 || cfg_.family == Family::Glm5Next ||
        cfg_.family == Family::Pixtral) {
        if (why) *why = "this model's processor takes pictures, not video";
        return RAD_E_UNSUPPORTED;
    }
    Video v;
    if (cfg_.family == Family::MuseGlimmer) {
        const SampleArgs a{ cfg_.fps, cfg_.min_frames, cfg_.max_frames, false, cfg_.temporal };
        RAD_TRY(decode_video(bytes, n, muse_sample_cb, &a, cfg_.max_source_pixels, &v, why));
        return video_from(v, out, why);
    }
    const SampleArgs a{ cfg_.fps, cfg_.min_frames, cfg_.max_frames,
                        cfg_.family == Family::Gemma4, cfg_.temporal };
    RAD_TRY(decode_video(bytes, n, sample_cb, &a, cfg_.max_source_pixels, &v, why));
    return video_from(v, out, why);
}

/* ================================================================== audio */

/* A list a container flattened from a JSON array ("1 2 3"), or nothing when it is absent or null. */
static std::vector<double> read_list(const RadModelMeta& m, const char* key) {
    std::vector<double> v;
    const char* s = rad_meta_gets(&m, key, nullptr);
    if (!s || !*s) return v;
    std::istringstream is(s);
    double x;
    while (is >> x) v.push_back(x);
    return v;
}

int64_t AudioConfig::chunk_rows(int64_t frames) {
    for (int i = 0; i < 3; ++i) frames = frames > 0 ? (frames - 1) / 2 + 1 : 0;
    return frames;
}

/* Qwen3ASRFeatureExtractor's framing, from processor_config.json's `feature_extractor` and the
 * encoder's audio_config. The mel band (0 to 8000 Hz) is the extractor's own constant, not a
 * key. */
static int qwen3asr_from_meta(const RadModelMeta& m, AudioConfig* c, std::string* why) {
    auto gi = [&](const char* k, int64_t d) {
        char key[128];
        std::snprintf(key, sizeof key, "processor.feature_extractor.%s", k);
        return rad_meta_geti(&m, key, d);
    };
    c->family       = AudioFamily::Qwen3ASR;
    c->rate         = (int32_t)gi("sampling_rate", 16000);
    c->hop          = (int32_t)gi("hop_length", 160);
    c->fft          = (int32_t)gi("n_fft", 400);
    c->frame_length = c->fft;
    c->n_mels       = (int32_t)gi("feature_size", rad_meta_geti(&m, "audio_config.num_mel_bins", 128));
    c->min_samples  = gi("min_length", 8000);
    c->chunk        = (int32_t)(2 * gi("n_window", rad_meta_geti(&m, "audio_config.n_window", 50)));
    c->min_freq     = 0.0;
    c->max_freq     = 8000.0;
    c->tok_audio    = (int32_t)rad_meta_geti(&m, "audio_token_id", -1);
    if (rad_meta_getf(&m, "processor.feature_extractor.dither", 0.0) != 0.0) {
        if (why) *why = "the feature extractor dithers its input (feature_extractor.dither), which "
                        "no deterministic front end reproduces";
        return RAD_E_UNSUPPORTED;
    }
    if (c->rate <= 0 || c->hop <= 0 || c->fft < 2 || c->n_mels <= 0 || c->chunk <= 0 ||
        c->min_samples < 0 || c->tok_audio < 0 || 2.0 * c->max_freq > (double)c->rate) {
        if (why) *why = "the container's Qwen3-ASR feature configuration is incomplete or "
                        "inconsistent (rate, hop, n_fft, mel bins, n_window, audio_token_id)";
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* Gemma4UnifiedAudioFeatureExtractor's framing (Gemma 4 12B): processor_config.json's
 * `feature_extractor` and the clip cap `audio_seq_length` (in frames). The frame's width is what the
 * embedder takes, so it stands in `n_mels` as a mel spectrum's bins do. */
static int unified_from_meta(const RadModelMeta& m, AudioConfig* c, std::string* why) {
    c->family       = AudioFamily::Gemma4Unified;
    c->rate         = (int32_t)rad_meta_geti(&m, "processor.feature_extractor.sampling_rate", 16000);
    c->frame_length = (int32_t)rad_meta_geti(
        &m, "processor.feature_extractor.audio_samples_per_token",
        rad_meta_geti(&m, "audio_config.audio_samples_per_token",
                      rad_meta_geti(&m, "audio_config.audio_embed_dim", 0)));
    c->hop          = c->frame_length;
    c->n_mels       = c->frame_length;
    c->max_samples  = rad_meta_geti(&m, "processor.audio_seq_length", 750) * c->frame_length;
    c->tok_audio    = (int32_t)rad_meta_geti(&m, "audio_token_id", -1);
    c->tok_start    = (int32_t)rad_meta_geti(&m, "boa_token_id", -1);
    c->tok_end      = (int32_t)rad_meta_geti(&m, "eoa_token_id",
                                             rad_meta_geti(&m, "eoa_token_index", -1));
    if (c->rate <= 0 || c->frame_length <= 0 || c->max_samples <= 0 || c->tok_audio < 0 ||
        c->tok_start < 0 || c->tok_end < 0) {
        if (why) *why = "the container's audio framing is incomplete (sampling_rate, "
                        "audio_samples_per_token, audio_seq_length, audio_token_id, boa_token_id, "
                        "eoa_token_id)";
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

int AudioConfig::from_meta(const RadModelMeta& m, std::string* why) {
    const char* mt = rad_meta_gets(&m, "audio_config.model_type", "");
    if (mt && !std::strcmp(mt, "qwen3_asr_encoder")) return qwen3asr_from_meta(m, this, why);
    if (mt && !std::strcmp(mt, "gemma4_unified_audio")) return unified_from_meta(m, this, why);
    if (!mt || std::strcmp(mt, "gemma4_audio")) {
        if (why) *why = "the container states no audio tower this processor knows (audio_config."
                        "model_type gemma4_audio, gemma4_unified_audio or qwen3_asr_encoder)";
        return RAD_E_NOTFOUND;
    }
    /* preprocessor_config.json's keys (`preprocessor.`), else processor_config.json's
     * `feature_extractor` object, where Gemma 4 E2B/E4B keep them. */
    auto key = [&m](const char* k) {
        std::string pp = std::string("preprocessor.") + k;
        if (rad_meta_gets(&m, pp.c_str(), nullptr)) return pp;
        return std::string("processor.feature_extractor.") + k;
    };
    rate         = (int32_t)rad_meta_geti(&m, key("sampling_rate").c_str(), 16000);
    frame_length = (int32_t)rad_meta_geti(&m, key("frame_length").c_str(), 320);
    hop          = (int32_t)rad_meta_geti(&m, key("hop_length").c_str(), 160);
    n_mels       = (int32_t)rad_meta_geti(&m, key("feature_size").c_str(), 128);
    fft          = (int32_t)rad_meta_geti(&m, key("fft_length").c_str(), 0);
    if (fft <= 0) {
        fft = 1;
        while (fft < frame_length) fft *= 2;
    }
    if (rad_meta_geti(&m, key("fft_overdrive").c_str(), 0)) fft *= 2;
    min_freq    = rad_meta_getf(&m, key("min_frequency").c_str(), 0.0);
    max_freq    = rad_meta_getf(&m, key("max_frequency").c_str(), 8000.0);
    mel_floor   = rad_meta_getf(&m, key("mel_floor").c_str(), 1e-3);
    preemphasis = rad_meta_getf(&m, key("preemphasis").c_str(), 0.0);
    preemphasis_htk = rad_meta_geti(&m, key("preemphasis_htk_flavor").c_str(), 1) != 0;
    input_scale = rad_meta_getf(&m, key("input_scale_factor").c_str(), 1.0);
    bin_mean = read_list(m, key("per_bin_mean").c_str());
    bin_std  = read_list(m, key("per_bin_stddev").c_str());
    max_samples = 480000;
    tok_audio = (int32_t)rad_meta_geti(&m, "audio_token_id", -1);
    tok_start = (int32_t)rad_meta_geti(&m, "boa_token_id", -1);
    tok_end   = (int32_t)rad_meta_geti(&m, "eoa_token_id", rad_meta_geti(&m, "eoa_token_index", -1));
    /* Dither adds Gaussian noise of this deviation to every sample, a new draw each call, so no
     * deterministic front end can reproduce it. */
    if (rad_meta_getf(&m, key("dither").c_str(), 0.0) != 0.0) {
        if (why) *why = "the feature extractor dithers its input (dither), which no "
                        "deterministic front end reproduces";
        return RAD_E_UNSUPPORTED;
    }
    if (rate <= 0 || frame_length <= 1 || hop <= 0 || n_mels <= 0 || fft < frame_length ||
        !(max_freq > min_freq) || !(mel_floor > 0.0) || tok_audio < 0 || tok_start < 0 ||
        tok_end < 0 || (!bin_mean.empty() && (int64_t)bin_mean.size() != n_mels) ||
        (!bin_std.empty() && (int64_t)bin_std.size() != n_mels)) {
        if (why) *why = "the container's audio feature configuration is incomplete or "
                        "inconsistent (rate, frame and hop lengths, mel bins, frequencies, "
                        "audio_token_id, boa_token_id, eoa_token_id)";
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

int64_t AudioConfig::real_frames(int64_t n) const {
    /* Frame i's last sample is (frame_length / 2 zeros) + i * hop + frame_length, and it is real
     * when that is inside the clip: i * hop + frame_length < frame_length / 2 + n. */
    const int64_t room = n + frame_length / 2 - frame_length - 1;
    return room < 0 ? 0 : room / hop + 1;
}

namespace {

/* An in-place radix-2 complex FFT, n a power of two: the frame's real spectrum in double, as the
 * reference's numpy rfft computes it. */
void fft_inplace(std::vector<double>& re, std::vector<double>& im,
                 const std::vector<double>& tw_re, const std::vector<double>& tw_im,
                 const std::vector<int32_t>& brev) {
    const size_t n = re.size();
    for (size_t i = 1; i < n; ++i) {
        const size_t j = (size_t)brev[i];
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t step = n / len;   /* stage len's k-th twiddle is the table's k * step */
        for (size_t i = 0; i < n; i += len)
            for (size_t k = 0; k < len / 2; ++k) {
                const double wr = tw_re[k * step], wi = tw_im[k * step];
                const size_t a = i + k, b = i + k + len / 2;
                const double xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
            }
    }
}

/* A mixed-radix complex FFT over any length (Whisper's 400 points are 2^4 * 5^2). Decimation in
 * time over `radix` (the length's prime factors, smallest first): n points at stride s
 * are p transforms of n / p points at stride s * p, combined by p-point DFTs with twiddles. `tw`
 * is exp(-2 pi i k / N) for k < N, N the outermost length, read at stride N / n by a level of n
 * points; `work` is scratch, 2n doubles a level, the levels below reusing what follows. */
void fft_mixed(const double* in_re, const double* in_im, int64_t stride, int64_t n,
               double* out_re, double* out_im, const int32_t* radix,
               const std::vector<double>& tw_re, const std::vector<double>& tw_im, double* work) {
    if (n == 1) { out_re[0] = in_re[0]; out_im[0] = in_im[0]; return; }
    const int64_t p = radix[0], m = n / p;
    for (int64_t q = 0; q < p; ++q)
        fft_mixed(in_re + q * stride, in_im + q * stride, stride * p, m, out_re + q * m,
                  out_im + q * m, radix + 1, tw_re, tw_im, work + 2 * n);
    /* X[k + s m] = sum_q W_n^(q (k + s m)) Y_q[k]: Y_q is periodic in m. */
    const int64_t step = (int64_t)tw_re.size() / n;
    double* t_re = work;
    double* t_im = work + n;
    for (int64_t k = 0; k < m; ++k)
        for (int64_t s = 0; s < p; ++s) {
            const int64_t at = k + s * m;
            double ar = 0.0, ai = 0.0;
            for (int64_t q = 0; q < p; ++q) {
                const int64_t e = (q * at) % n * step;
                const double yr = out_re[q * m + k], yi = out_im[q * m + k];
                ar += yr * tw_re[(size_t)e] - yi * tw_im[(size_t)e];
                ai += yr * tw_im[(size_t)e] + yi * tw_re[(size_t)e];
            }
            t_re[at] = ar;
            t_im[at] = ai;
        }
    std::memcpy(out_re, t_re, (size_t)n * sizeof(double));
    std::memcpy(out_im, t_im, (size_t)n * sizeof(double));
}

/* numpy.linspace's values: start + i * step, the last exactly `stop`. */
std::vector<double> linspace(double a, double b, int64_t n) {
    std::vector<double> v((size_t)n);
    const double step = n > 1 ? (b - a) / (double)(n - 1) : 0.0;
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = a + (double)i * step;
    if (n > 1) v[(size_t)n - 1] = b;
    return v;
}

}  /* namespace */

int Processor::init_audio(const AudioConfig& cfg, std::string* why) {
    auto refuse = [&](int rc, std::string w) {
        if (why) *why = std::move(w);
        return rc;
    };
    if (cfg.merge <= 0 || cfg.max_frames <= 0 || cfg.max_frames % cfg.merge)
        return refuse(RAD_E_INVAL, "the audio encoder's pass (" + std::to_string(cfg.max_frames) +
                                       " frames) is not whole rows of " +
                                       std::to_string(cfg.merge) + " frames");
    /* The 12B's frames are samples: no window, no FFT, no mel bank. */
    if (cfg.family == AudioFamily::Gemma4Unified) {
        acfg_ = cfg;
        audio_on_ = true;
        return RAD_OK;
    }
    const bool qwen = cfg.family == AudioFamily::Qwen3ASR;
    /* Gemma 4's FFT is radix 2; Qwen3-ASR's runs mixed radix over any length. */
    if (!qwen && (cfg.fft < 2 || (cfg.fft & (cfg.fft - 1)) != 0))
        return refuse(RAD_E_UNSUPPORTED, "the feature extractor's " + std::to_string(cfg.fft) +
                                             "-point FFT is not a power of two, which this "
                                             "extractor's radix-2 transform needs");
    acfg_ = cfg;
    if (qwen) {
        /* A whole chunk is staged as merge * chunk_rows(chunk) frames, which must hold it; a
         * clip's longest is as many whole chunks as one pass takes. */
        const int64_t staged = (int64_t)cfg.merge * AudioConfig::chunk_rows(cfg.chunk);
        if (staged < cfg.chunk || cfg.max_frames < staged)
            return refuse(RAD_E_INVAL, "a whole " + std::to_string(cfg.chunk) + "-frame chunk is "
                                       "staged as " + std::to_string(staged) + " frames, which "
                                       "does not hold it or does not fit one " +
                                       std::to_string(cfg.max_frames) + "-frame encoder pass");
        if (cfg.tok_start < 0 || cfg.tok_end < 0)
            return refuse(RAD_E_INVAL, "the clip's opening or closing token is unresolved");
        acfg_.max_samples = cfg.max_frames / staged * cfg.chunk * cfg.hop;
    }
    /* np.hanning(frame_length + 1)[:-1] -- torch.hann_window's periodic window -- stored as
     * float32. */
    hann_.resize((size_t)cfg.frame_length);
    for (int32_t i = 0; i < cfg.frame_length; ++i)
        hann_[(size_t)i] = (float)(0.5 - 0.5 * std::cos(2.0 * M_PI * (double)i /
                                                        (double)cfg.frame_length));
    /* transformers' mel_filter_bank: triangles between consecutive mel points, in Hz, over the
     * FFT bins' frequencies -- Gemma 4's on the HTK scale with no area normalisation, Qwen3-ASR's
     * (Whisper's) on the Slaney scale, each triangle scaled to unit area by 2 / (its width in
     * Hz). */
    const int64_t nb = cfg.fft / 2 + 1, nm = cfg.n_mels;
    auto htk2mel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
    auto mel2htk = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };
    const double slaney_step = 27.0 / std::log(6.4);
    auto slaney2mel = [&](double f) {
        return f >= 1000.0 ? 15.0 + std::log(f / 1000.0) * slaney_step : 3.0 * f / 200.0;
    };
    auto mel2slaney = [&](double m) {
        return m >= 15.0 ? 1000.0 * std::exp((m - 15.0) / slaney_step) : 200.0 * m / 3.0;
    };
    const std::vector<double> fft_f = linspace(0.0, (double)(cfg.rate / 2), nb);
    std::vector<double> filt = qwen ? linspace(slaney2mel(cfg.min_freq), slaney2mel(cfg.max_freq), nm + 2)
                                    : linspace(htk2mel(cfg.min_freq), htk2mel(cfg.max_freq), nm + 2);
    for (double& f : filt) f = qwen ? mel2slaney(f) : mel2htk(f);
    mel_.assign((size_t)(nb * nm), 0.0);
    mel_lo_.assign((size_t)nm, 0);
    mel_hi_.assign((size_t)nm, 0);
    for (int64_t i = 0; i < nm; ++i) {
        const double enorm = qwen ? 2.0 / (filt[(size_t)i + 2] - filt[(size_t)i]) : 1.0;
        int32_t lo = -1, hi = 0;
        for (int64_t k = 0; k < nb; ++k) {
            const double down = -(filt[(size_t)i] - fft_f[(size_t)k]) /
                                (filt[(size_t)i + 1] - filt[(size_t)i]);
            const double up = (filt[(size_t)i + 2] - fft_f[(size_t)k]) /
                              (filt[(size_t)i + 2] - filt[(size_t)i + 1]);
            const double w = std::max(0.0, std::min(down, up)) * enorm;
            mel_[(size_t)(i * nb + k)] = w;
            if (w != 0.0) { if (lo < 0) lo = (int32_t)k; hi = (int32_t)k + 1; }
        }
        /* A triangle covers a few bins; the sum skips the zeros. */
        mel_lo_[(size_t)i] = lo < 0 ? 0 : lo;
        mel_hi_[(size_t)i] = hi;
    }
    /* Precomputed twiddles: evaluating sin and cos per butterfly costs over 100 ms on a 30 s
     * clip. Gemma 4's power of two runs radix 2 over half a turn; Qwen3-ASR's FFT runs mixed
     * radix over the length's prime factors (a power of two included) and a whole turn. */
    radix_.clear();
    brev_.clear();
    const bool pow2 = !qwen;
    if (pow2) {
        brev_.assign((size_t)cfg.fft, 0);
        for (int32_t i = 1, j = 0; i < cfg.fft; ++i) {
            int32_t bit = cfg.fft >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            brev_[(size_t)i] = j;
        }
    } else {
        int32_t r = cfg.fft;
        for (int32_t f = 2; f * f <= r; ++f)
            while (r % f == 0) { radix_.push_back(f); r /= f; }
        if (r > 1) radix_.push_back(r);
    }
    const int32_t n_tw = pow2 ? cfg.fft / 2 : cfg.fft;
    tw_re_.resize((size_t)n_tw);
    tw_im_.resize((size_t)n_tw);
    for (int32_t k = 0; k < n_tw; ++k) {
        const double ang = -2.0 * M_PI * (double)k / (double)cfg.fft;
        tw_re_[(size_t)k] = std::cos(ang);
        tw_im_[(size_t)k] = std::sin(ang);
    }
    audio_on_ = true;
    return RAD_OK;
}

/* Qwen3-ASR's clip (AudioFamily::Qwen3ASR): Whisper's log-mel over the whole clip, then staged a
 * chunk at a time. The spectrum is computed in double, two real frames a complex transform as the
 * Gemma 4 path does; the reference computes in float32, and the difference is far below the bf16
 * the features are staged in. */
int Processor::audio_qwen3asr(const std::vector<float>& samples, std::shared_ptr<Item>* out,
                              std::string* why) const {
    const AudioConfig& c = acfg_;
    if ((int64_t)samples.size() > c.max_samples) {
        if (why) *why = "the clip is longer than one audio encoder pass holds (" +
                        std::to_string(c.max_samples / c.rate) + " s)";
        return RAD_E_INVAL;
    }
    /* A clip shorter than min_samples is zero-padded to it first; the reflected padding is of the
     * padded clip. */
    const int64_t n = std::max<int64_t>((int64_t)samples.size(), c.min_samples);
    const int64_t pad = c.fft / 2;
    if (n <= pad) {
        if (why) *why = "the clip is too short to frame";
        return RAD_E_INVAL;
    }
    /* torch.stft(center=True) makes 1 + n / hop frames and the extractor drops the last. */
    const int64_t real = n / c.hop;
    const int64_t full = real / c.chunk, rem = real % c.chunk;
    const int64_t rows_full = AudioConfig::chunk_rows(c.chunk);
    const int64_t rows_rem = AudioConfig::chunk_rows(rem);
    const int64_t staged_full = (int64_t)c.merge * rows_full;
    const int64_t rows = full * rows_full + rows_rem;
    const int64_t frames = full * staged_full + (int64_t)c.merge * rows_rem;
    if (frames > c.max_frames) {
        if (why) *why = "the clip is more frames than one audio encoder pass holds";
        return RAD_E_INVAL;
    }

    /* x[q] of the padded clip, reflected at both ends: torch's "reflect" leaves the edge sample
     * out, so x[-1] is x[1] and x[n] is x[n - 2]. */
    auto sample = [&](int64_t p) -> double {
        int64_t q = p - pad;
        if (q < 0) q = -q;
        if (q >= n) q = 2 * (n - 1) - q;
        return q < (int64_t)samples.size() ? (double)samples[(size_t)q] : 0.0;
    };
    const int64_t N = c.fft, nb = N / 2 + 1, nm = c.n_mels;
    std::vector<double> re((size_t)N), im((size_t)N), ore((size_t)N), oim((size_t)N);
    std::vector<double> work((size_t)(4 * N));
    std::vector<double> pw((size_t)nb), pw2((size_t)nb);
    auto window = [&](int64_t t, double* dst) {
        for (int64_t j = 0; j < N; ++j)
            dst[j] = (double)((float)sample(t * c.hop + j) * hann_[(size_t)j]);
    };
    std::vector<double> logmel((size_t)(real * nm));
    double top = -INFINITY;
    auto emit = [&](int64_t t, const double* p) {
        double* dst = logmel.data() + t * nm;
        for (int64_t b = 0; b < nm; ++b) {
            double s = 0.0;
            const double* w = mel_.data() + (size_t)b * nb;
            for (int32_t k = mel_lo_[(size_t)b]; k < mel_hi_[(size_t)b]; ++k) s += p[k] * w[k];
            const double v = std::log10(std::max(s, 1e-10));
            dst[b] = v;
            if (v > top) top = v;
        }
    };
    for (int64_t t = 0; t < real; t += 2) {
        const bool pair = t + 1 < real;
        window(t, re.data());
        if (pair) window(t + 1, im.data());
        else std::fill(im.begin(), im.end(), 0.0);
        fft_mixed(re.data(), im.data(), 1, N, ore.data(), oim.data(), radix_.data(), tw_re_,
                  tw_im_, work.data());
        for (int64_t k = 0; k < nb; ++k) {
            const size_t a = (size_t)k, b = k ? (size_t)(N - k) : 0;
            const double ar = 0.5 * (ore[a] + ore[b]), ai = 0.5 * (oim[a] - oim[b]);
            const double br = 0.5 * (oim[a] + oim[b]), bi = -0.5 * (ore[a] - ore[b]);
            pw[a]  = ar * ar + ai * ai;
            pw2[a] = br * br + bi * bi;
        }
        emit(t, pw.data());
        if (pair) emit(t + 1, pw2.data());
    }

    auto it = std::make_shared<Item>();
    it->kind = Kind::Audio;
    it->grid_t = 1;
    it->grid_h = (int32_t)frames;
    it->grid_w = 1;
    it->patch_dim = nm;
    it->n_patch = frames;
    it->valid = (int32_t)real;
    it->chunk_staged = (int32_t)staged_full;
    it->chunk_read = c.chunk;
    it->rows_h = (int32_t)rows;
    it->rows_w = 1;
    it->seg_rows = (int32_t)rows;
    it->n_seg = rows > 0 ? 1 : 0;
    /* Chunk k's real frames at row k * staged_full; everything else stays zero -- the reference's
     * padding inside the last chunk, and the alignment past a whole one. */
    it->pixels.assign((size_t)(frames * nm), 0);
    const double floor_v = top - 8.0;
    for (int64_t t = 0; t < real; ++t) {
        uint16_t* dst = it->pixels.data() + ((t / c.chunk) * staged_full + t % c.chunk) * nm;
        const double* src = logmel.data() + t * nm;
        for (int64_t b = 0; b < nm; ++b)
            dst[b] = rad_f32_to_bf16((float)((std::max(src[b], floor_v) + 4.0) / 4.0));
    }
    content_key(*it, &it->key_hi, &it->key_lo);
    it->expand.push_back(c.tok_start);
    it->seg_at.push_back(1);
    it->expand.insert(it->expand.end(), (size_t)rows, c.tok_audio);
    it->expand.push_back(c.tok_end);
    *out = std::move(it);
    return RAD_OK;
}

/* The 12B's clip (AudioFamily::Gemma4Unified): ceil(n / frame_length) frames of samples, the last
 * zero-padded, each sample rounded to bf16 as the reference casts the frames for its embedder; one
 * frame a placeholder. The reference cuts nothing; a clip past `max_samples` is cut there. */
int Processor::audio_unified(const std::vector<float>& samples, std::shared_ptr<Item>* out,
                             std::string* why) const {
    const AudioConfig& c = acfg_;
    const int64_t n = std::min<int64_t>((int64_t)samples.size(), c.max_samples);
    const int64_t fl = c.frame_length, rows = (n + fl - 1) / fl;
    if (rows * c.merge > c.max_frames) {
        if (why) *why = "the clip is more frames than one audio encoder pass holds";
        return RAD_E_INVAL;
    }
    auto it = std::make_shared<Item>();
    it->kind = Kind::Audio;
    it->grid_t = 1;
    it->grid_h = (int32_t)rows;
    it->grid_w = 1;
    it->patch_dim = fl;
    it->n_patch = rows;
    it->valid = (int32_t)rows;
    it->rows_h = (int32_t)rows;
    it->rows_w = 1;
    it->seg_rows = (int32_t)rows;
    it->n_seg = rows > 0 ? 1 : 0;
    it->pixels.assign((size_t)(rows * fl), 0);
    for (int64_t i = 0; i < n; ++i) it->pixels[(size_t)i] = rad_f32_to_bf16(samples[(size_t)i]);
    one_run(it.get(), c.tok_start, rows, c.tok_audio, c.tok_end);
    if (rows == 0) it->encoded = true;
    *out = std::move(it);
    return RAD_OK;
}

int Processor::audio_from(const std::vector<float>& samples, std::shared_ptr<Item>* out,
                          std::string* why) const {
    if (!audio_on_) { if (why) *why = "this program declares no audio encoder"; return RAD_E_UNSUPPORTED; }
    const AudioConfig& c = acfg_;
    if (c.family == AudioFamily::Qwen3ASR) return audio_qwen3asr(samples, out, why);
    if (c.family == AudioFamily::Gemma4Unified) return audio_unified(samples, out, why);
    const int64_t n = std::min<int64_t>((int64_t)samples.size(), c.max_samples);
    const int64_t real = c.real_frames(n);
    const int64_t rows = (real + c.merge - 1) / c.merge;
    const int64_t frames = rows * c.merge;
    if (frames > c.max_frames) {
        if (why) *why = "the clip is more frames than one audio encoder pass holds";
        return RAD_E_INVAL;
    }
    auto it = std::make_shared<Item>();
    it->kind = Kind::Audio;
    it->grid_t = 1;
    it->grid_h = (int32_t)frames;
    it->grid_w = 1;
    it->patch_dim = c.n_mels;
    it->n_patch = frames;
    it->valid = (int32_t)real;
    it->rows_h = (int32_t)rows;
    it->rows_w = 1;
    it->seg_rows = (int32_t)rows;
    it->n_seg = rows > 0 ? 1 : 0;
    it->pixels.assign((size_t)(frames * c.n_mels), 0);

    const int64_t pad = c.frame_length / 2;
    const int64_t nb = c.fft / 2 + 1;
    std::vector<double> re((size_t)c.fft), im((size_t)c.fft), mag((size_t)nb);
    std::vector<float> fr((size_t)c.frame_length + 1);
    auto sample = [&](int64_t p) -> float {     /* the zero-led clip, zero past its end */
        const int64_t q = p - pad;
        return q >= 0 && q < n ? samples[(size_t)q] * (float)c.input_scale : 0.0f;
    };
    /* Frame i's windowed samples into dst, zero past the frame. */
    auto window = [&](int64_t i, double* dst) {
        for (int32_t j = 0; j <= c.frame_length; ++j) fr[(size_t)j] = sample(i * c.hop + j);
        for (int32_t j = 0; j < c.frame_length; ++j) {
            float v;
            if (c.preemphasis > 0.0) {
                const float pe = (float)c.preemphasis;
                v = c.preemphasis_htk ? (j == 0 ? fr[0] * (1.0f - pe) : fr[(size_t)j] - pe * fr[(size_t)j - 1])
                                      : fr[(size_t)j + 1] - pe * fr[(size_t)j];
            } else {
                v = fr[(size_t)j];
            }
            dst[j] = (double)(v * hann_[(size_t)j]);   /* the product in float32 */
        }
        for (int32_t j = c.frame_length; j < c.fft; ++j) dst[j] = 0.0;
    };
    /* Frame i's mel bins from its magnitudes. */
    auto emit = [&](int64_t i, const double* m) {
        uint16_t* dst = it->pixels.data() + i * c.n_mels;
        for (int32_t b = 0; b < c.n_mels; ++b) {
            double s = 0.0;
            const double* w = mel_.data() + (size_t)b * nb;
            for (int32_t k = mel_lo_[(size_t)b]; k < mel_hi_[(size_t)b]; ++k) s += m[k] * w[k];
            double v = std::log(s + c.mel_floor);
            if (!c.bin_mean.empty()) v -= c.bin_mean[(size_t)b];
            if (!c.bin_std.empty())  v /= c.bin_std[(size_t)b];
            dst[b] = rad_f32_to_bf16((float)v);
        }
    };
    /* Two real frames per complex transform: frame a in the real part, frame b in the imaginary,
     * separated by symmetry as A[k] = (Z[k] + conj Z[n-k]) / 2 and B[k] = (Z[k] - conj Z[n-k]) / 2i.
     * This halves the transforms; magnitudes match one transform per frame to a few ulps of double.
     * |X| is sqrt(re^2 + im^2), within an ulp of numpy's hypot and far below the float32 the
     * features are rounded to. */
    std::vector<double> mag2((size_t)nb);
    const int64_t N = c.fft;
    for (int64_t i = 0; i < real; i += 2) {
        const bool pair = i + 1 < real;
        window(i, re.data());
        if (pair) window(i + 1, im.data());
        else std::fill(im.begin(), im.end(), 0.0);
        fft_inplace(re, im, tw_re_, tw_im_, brev_);
        for (int64_t k = 0; k < nb; ++k) {
            const size_t a = (size_t)k, b = k ? (size_t)(N - k) : 0;
            const double ar = 0.5 * (re[a] + re[b]), ai = 0.5 * (im[a] - im[b]);
            const double br = 0.5 * (im[a] + im[b]), bi = -0.5 * (re[a] - re[b]);
            mag[a]  = std::sqrt(ar * ar + ai * ai);
            mag2[a] = std::sqrt(br * br + bi * bi);
        }
        emit(i, mag.data());
        if (pair) emit(i + 1, mag2.data());
    }
    one_run(it.get(), c.tok_start, rows, c.tok_audio, c.tok_end);
    /* A clip too short for one real frame is <boa><eoa> and nothing to encode. */
    if (rows == 0) it->encoded = true;
    *out = std::move(it);
    return RAD_OK;
}

int Processor::audio(const uint8_t* bytes, size_t n, std::shared_ptr<Item>* out,
                     std::string* why) const {
    if (!audio_on_) { if (why) *why = "this program declares no audio encoder"; return RAD_E_UNSUPPORTED; }
    std::vector<float> s;
    /* Gemma 4's extractor truncates at max_samples; Qwen3-ASR's does not, so one sample more is
     * decoded and a clip past a pass is refused rather than cut. */
    const bool whole = acfg_.family == AudioFamily::Qwen3ASR;
    RAD_TRY(decode_audio(bytes, n, acfg_.rate, acfg_.max_samples + (whole ? 1 : 0), &s, why));
    return audio_from(s, out, why);
}

}  /* namespace mm */
}  /* namespace rad */
