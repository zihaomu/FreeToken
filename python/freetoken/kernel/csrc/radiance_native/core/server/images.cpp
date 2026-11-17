/* images.cpp -- OpenAI's images API over an image generator (core/gen): /v1/images/generations
 * and /v1/images/edits.
 *
 * Request: the prompt goes into the model's own template (ImageGenInfo), including the fixed
 * system turn the generator drops; a transparent background is requested in the model's trained
 * wording. Each edit picture is resized to the model's reference area at its own aspect (PIL's
 * premultiplied Lanczos, as diffusers does) and read twice: as RGBA by the generator's encoder,
 * and composited on white by the conditioning network's. With a mask, the first picture is
 * repainted where the mask is transparent; the rest is restored exactly after decoding.
 *
 * Extensions, under vLLM-Omni's names: `seed`, `negative_prompt` with `true_cfg_scale`
 * (guidance, in the model card's terms; `guidance_scale` is the same field under diffusers' name),
 * `num_inference_steps`, and `width` / `height` instead of `size`. A fixed-schedule (distilled)
 * model accepts only its own step count. A model whose card guides by default does so with its
 * own negative prompt unless the request names one or turns guidance off.
 *
 * Refused: `response_format: url`, since no image is stored to serve later; only b64_json.
 *
 * Streaming follows OpenAI: `image_generation.partial_image` (`image_edit.partial_image`) events
 * for the `partial_images` previews (each step's decoded estimate of the final latent), and one
 * `.completed` event per image.
 */
#include "server.h"
#include "loras.h"
#include "oai.h"

#include "gen/gen.h"
#include "mm/imaging.h"
#include "mm/media.h"
#include "mm/processor.h"
#include "rad_internal.h"
#include "rad_plugin.h"

#include "util/json.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <random>

namespace rad {
namespace server {

namespace {

/* The largest picture a request may upload, before any resize. */
constexpr int64_t kMaxSourcePixels = (int64_t)1 << 26;

/* Below this alpha a pixel counts as transparent, both in a mask and when deciding whether a
 * picture has transparency. A threshold, not 255, because a decoder's alpha on an opaque picture is
 * not exactly 255, so this model's own output stays opaque when edited. */
constexpr uint8_t kTransparentBelow = 128;

struct ImgReq {
    bool        edit = false;
    std::string prompt, negative;
    bool        has_neg = false;
    double      cfg = 1.0;
    int32_t     w = 0, h = 0, n = 1, steps = 0, previews = 0;
    int32_t     sampler = 0, scheduler = 0;
    bool        seed_given = false;
    uint64_t    seed = 0;
    std::string background = "auto", format = "png", quality = "auto";
    int32_t     compression = -1;
    bool        stream = false;
    int32_t     rebalance = 0;   /* RadGenBatch::cond_opts: RAD_GEN_COND_REBALANCE and its preset */
    std::vector<mm::FrameRGBA> images;
    bool        has_mask = false;
    mm::FrameRGBA mask;
};

/* A multipart field's text as the JSON value the same field takes in a JSON body. */
json form_value(const std::string& key, const std::string& v) {
    static const char* ints[] = { "n", "seed", "num_inference_steps", "partial_images",
                                  "output_compression", "width", "height" };
    for (const char* k : ints)
        if (key == k) {
            char* e = nullptr;
            const long long x = std::strtoll(v.c_str(), &e, 10);
            if (e && *e == 0 && !v.empty()) return json(x);
            return json(v);
        }
    if (key == "true_cfg_scale" || key == "guidance_scale") {
        char* e = nullptr;
        const double x = std::strtod(v.c_str(), &e);
        if (e && *e == 0 && !v.empty()) return json(x);
        return json(v);
    }
    if (key == "stream" || key == "rebalance") {
        if (v == "true" || v == "1") return json(true);
        if (v == "false" || v == "0") return json(false);
    }
    /* `loras` as JSON text, or one adapter's name. Bounded like a request body: a form field
     * under the body cap can be brackets alone, and the deep tree's teardown would take the
     * process with it. Over-deep or over-budget reads as plain text, as a discarded parse did. */
    if ((key == "loras" || key == "lora") && !v.empty() && (v[0] == '[' || v[0] == '{')) {
        bool over = false, deep = false;
        json j = parse_json_bounded(v, 0, &over, &deep);
        if (!j.is_discarded() && !over && !deep) return j;
    }
    return json(v);
}

/* diffusers' calculate_dimensions: a target area at an aspect, each side rounded to `mult`. */
void dims_for(double area, double ratio, int32_t mult, int32_t* w, int32_t* h) {
    const double wd = std::sqrt(area * ratio);
    const double hd = wd / ratio;
    *w = (int32_t)std::lround(wd / mult) * mult;
    *h = (int32_t)std::lround(hd / mult) * mult;
    if (*w < mult) *w = mult;
    if (*h < mult) *h = mult;
}

std::string replace_all(std::string s, const std::string& what, const std::string& with) {
    for (size_t p = s.find(what); p != std::string::npos; p = s.find(what, p + with.size()))
        s.replace(p, what.size(), with);
    return s;
}

/* A picture as the generator's encoder reads it: each channel u -> 2 * (u / 255) - 1 in f32, then
 * bf16 (VaeImageProcessor's preprocess, cast to the VAE's dtype). An RGB model reads the colour
 * channels and not the alpha, as PIL's convert("RGB") drops it. */
gen::Pixels to_pixels(const mm::FrameRGBA& f, int ch) {
    gen::Pixels p;
    p.w = f.w;
    p.h = f.h;
    p.ch = ch;
    p.bf16.resize((size_t)f.w * f.h * ch);
    for (size_t px = 0; px < (size_t)f.w * f.h; ++px)
        for (int c = 0; c < ch; ++c)
            p.bf16[px * ch + c] =
                rad_f32_to_bf16(2.0f * ((float)f.rgba[px * 4 + c] / 255.0f) - 1.0f);
    return p;
}

/* The centred w x h window of a picture (diffusers' resize_mode "crop" after its resize). */
mm::FrameRGBA crop_centre(const mm::FrameRGBA& f, int32_t w, int32_t h) {
    mm::FrameRGBA o;
    o.w = w;
    o.h = h;
    o.rgba.resize((size_t)w * h * 4);
    const int32_t x0 = (f.w - w) / 2, y0 = (f.h - h) / 2;
    for (int32_t y = 0; y < h; ++y)
        std::memcpy(&o.rgba[(size_t)y * w * 4], &f.rgba[((size_t)(y0 + y) * f.w + x0) * 4],
                    (size_t)w * 4);
    return o;
}

/* The decoded RGBA flattened on white (no alpha), for a jpeg or an opaque background. */
std::vector<uint8_t> on_white(const std::vector<uint8_t>& rgba, int32_t w, int32_t h) {
    mm::FrameRGBA f;
    f.w = w;
    f.h = h;
    f.rgba = rgba;
    mm::Frame o;
    mm::composite_white(f, &o);
    return std::move(o.rgb);
}

}  /* namespace */

HttpResponse Server::handle_images(const HttpRequest& req, bool edit) {
    if (!deps_.images || !deps_.tok)
        return error_response(err_not_implemented(
            "the images API requires an image generation model; this one generates text"),
            req.peer);
    const ImageGenInfo& info = deps_.images->info();
    const char* ep = edit ? "/v1/images/edits" : "/v1/images/generations";
    if (edit && info.max_refs <= 0)
        return error_response(err_not_implemented("this model makes images from text only"),
                              req.peer);

    /* ---- the body, JSON or (an edit's usual form) multipart, as one JSON object */
    json body = json::object();
    std::vector<std::vector<uint8_t>> files;
    std::vector<uint8_t> mask_bytes;
    bool have_mask = false;
    if (!req.form.empty()) {
        for (const FormPart& p : req.form) {
            if (p.name == "image" || p.name == "image[]") {
                files.emplace_back(p.data.begin(), p.data.end());
            } else if (p.name == "mask") {
                mask_bytes.assign(p.data.begin(), p.data.end());
                have_mask = true;
            } else {
                body[p.name] = form_value(p.name, p.data);
            }
        }
    } else {
        ApiError err;
        if (parse_request_object(req.body, lim_, &body, &err) < 0)
            return error_response(err, req.peer);
        /* An edit's pictures as JSON: `images: [{image_url: "data:..."}]` and `mask: {image_url}`
         * (the newer API form), or `image` as one data URL or a list. */
        auto url_of = [](const json& v, std::string* u) {
            if (v.is_string()) { *u = v.get<std::string>(); return true; }
            if (v.is_object() && v.contains("image_url") && v["image_url"].is_string()) {
                *u = v["image_url"].get<std::string>();
                return true;
            }
            if (v.is_object() && v.contains("image_url") && v["image_url"].is_object() &&
                v["image_url"].contains("url") && v["image_url"]["url"].is_string()) {
                *u = v["image_url"]["url"].get<std::string>();
                return true;
            }
            return false;
        };
        for (const char* key : { "images", "image" }) {
            if (!body.contains(key)) continue;
            const json v = body[key];
            body.erase(key);
            std::vector<json> list;
            if (v.is_array()) list.assign(v.begin(), v.end());
            else list.push_back(v);
            for (const json& e : list) {
                std::string u;
                if (!url_of(e, &u))
                    return error_response(err_bad(key, "a picture is a data: URL, or an object "
                                                       "with an image_url"), req.peer);
                std::vector<uint8_t> bytes;
                ApiError err;
                if (decode_media_url(u, key, &bytes, &err) < 0) return error_response(err, req.peer);
                files.push_back(std::move(bytes));
            }
        }
        if (body.contains("mask")) {
            std::string u;
            if (!url_of(body["mask"], &u))
                return error_response(err_bad("mask", "the mask is a data: URL, or an object with "
                                                      "an image_url"), req.peer);
            body.erase("mask");
            ApiError err;
            if (decode_media_url(u, "mask", &mask_bytes, &err) < 0)
                return error_response(err, req.peer);
            have_mask = true;
        }
    }

    /* ---- the fields */
    ImgReq r;
    r.edit = edit;
    static const char* kKnown[] = {
        "prompt", "model", "n", "size", "quality", "background", "output_format",
        "output_compression", "response_format", "stream", "partial_images", "moderation",
        "style", "user", "input_fidelity", "seed", "negative_prompt", "true_cfg_scale",
        "guidance_scale",
        "num_inference_steps", "width", "height", "loras", "lora", "sampler", "scheduler",
        "rebalance",
    };
    for (auto it = body.begin(); it != body.end(); ++it) {
        bool known = false;
        for (const char* k : kKnown) known = known || it.key() == k;
        if (!known)
            return error_response(err_bad(it.key(), "'" + it.key() + "' is not a field of " +
                                                        std::string(ep)), req.peer);
    }
    auto str = [&](const char* k, std::string* out) -> int {
        if (!body.contains(k) || body[k].is_null()) return 0;
        if (!body[k].is_string()) return -1;
        *out = body[k].get<std::string>();
        return 1;
    };
    auto num = [&](const char* k, int64_t lo, int64_t hi, int64_t* out) -> int {
        if (!body.contains(k) || body[k].is_null()) return 0;
        if (!body[k].is_number_integer()) return -1;
        const int64_t v = body[k].get<int64_t>();
        if (v < lo || v > hi) return -1;
        *out = v;
        return 1;
    };
    if (str("prompt", &r.prompt) < 0)
        return error_response(err_bad("prompt", "expected a string"), req.peer);
    if (!body.contains("prompt"))
        return error_response(err_bad("prompt", "the prompt is required"), req.peer);
    /* Runtime LoRA adapters: `loras` and <lora:...> tags, which leave the prompt. */
    std::vector<std::pair<std::string, double>> loras;
    {
        std::string field, why;
        if (parse_loras(body, &r.prompt, &loras, &field, &why) < 0)
            return error_response(err_bad(field.c_str(), why), req.peer);
        if (!loras.empty() && !opt_.gen_lora_acquire)
            return error_response(err_unsupported(field.c_str(), "this server was started without "
                                                                 "adapters (--lora, --lora-dir)"),
                                  req.peer);
    }
    /* An empty prompt becomes " ", as in the diffusers pipelines, unless the model pads a short
     * prompt itself (ImageGenInfo::min_tokens), as ComfyUI's tokenizers do. */
    if (r.prompt.empty() && info.min_tokens == 0) r.prompt = " ";
    std::string rf;
    if (str("response_format", &rf) < 0 || (!rf.empty() && rf != "b64_json" && rf != "url"))
        return error_response(err_bad("response_format", "expected b64_json"), req.peer);
    if (rf == "url") {
        ApiError e = err_unsupported("response_format",
                                     "url is not served: this server keeps no image to serve "
                                     "later; ask for b64_json");
        e.supported = { "b64_json" };
        return error_response(e, req.peer);
    }
    int64_t v = 0;
    if (num("n", 1, info.max_lanes, &v) < 0)
        return error_response(
            err_bad("n", "expected an integer in 1.." + std::to_string(info.max_lanes)), req.peer);
    r.n = body.contains("n") && !body["n"].is_null() ? (int32_t)v : 1;
    if (str("background", &r.background) < 0 ||
        (r.background != "auto" && r.background != "transparent" && r.background != "opaque")) {
        ApiError e = err_bad("background", "expected auto, transparent or opaque");
        e.supported = { "auto", "transparent", "opaque" };
        return error_response(e, req.peer);
    }
    if (str("output_format", &r.format) < 0 ||
        (r.format != "png" && r.format != "jpeg" && r.format != "webp")) {
        ApiError e = err_bad("output_format", "expected png, jpeg or webp");
        e.supported = { "png", "jpeg", "webp" };
        return error_response(e, req.peer);
    }
    if (r.background == "transparent" && r.format == "jpeg")
        return error_response(err_bad("background", "jpeg has no transparency; ask for png or "
                                                    "webp"), req.peer);
    if (r.background == "transparent" && info.pixel_channels < 4)
        return error_response(err_bad("background", "this model makes opaque images"), req.peer);
    if (num("output_compression", 0, 100, &v) < 0)
        return error_response(err_bad("output_compression", "expected an integer in 0..100"),
                              req.peer);
    if (body.contains("output_compression")) r.compression = (int32_t)v;
    if (body.contains("stream") && !body["stream"].is_null()) {
        if (!body["stream"].is_boolean())
            return error_response(err_bad("stream", "expected a boolean"), req.peer);
        r.stream = body["stream"].get<bool>();
    }
    if (num("partial_images", 0, 3, &v) < 0)
        return error_response(err_bad("partial_images", "expected an integer in 0..3"), req.peer);
    if (body.contains("partial_images")) r.previews = (int32_t)v;
    if (r.previews > 0 && !r.stream)
        return error_response(err_bad("partial_images", "partial images are sent on a stream; "
                                                        "set stream"), req.peer);
    if (str("quality", &r.quality) < 0 ||
        (r.quality != "auto" && r.quality != "low" && r.quality != "medium" &&
         r.quality != "high" && r.quality != "standard" && r.quality != "hd")) {
        ApiError e = err_bad("quality", "expected auto, low, medium or high (or standard, hd)");
        e.supported = { "auto", "low", "medium", "high", "standard", "hd" };
        return error_response(e, req.peer);
    }
    /* The step count: the request's, or what its quality asks for, or the model's. */
    int64_t steps = 0;
    if (num("num_inference_steps", 1, 200, &steps) < 0)
        return error_response(err_bad("num_inference_steps", "expected an integer in 1..200"),
                              req.peer);
    if (info.fixed_schedule) {
        if (steps != 0 && steps != info.fixed_steps)
            return error_response(err_unsupported("num_inference_steps",
                                                  "this model samples on its own " +
                                                      std::to_string(info.fixed_steps) +
                                                      "-step schedule and takes no other"),
                                  req.peer);
        steps = 0;
    } else if (steps == 0) {
        if (r.quality == "low") steps = std::max<int64_t>(1, info.default_steps / 2);
        else if (r.quality == "medium") steps = std::max<int64_t>(1, info.default_steps * 3 / 4);
    }
    r.steps = (int32_t)steps;
    /* A discrete-time model's sampler and scheduler, by ComfyUI's names. */
    {
        std::string sname, kname;
        if (str("sampler", &sname) < 0)
            return error_response(err_bad("sampler", "expected a string"), req.peer);
        if (str("scheduler", &kname) < 0)
            return error_response(err_bad("scheduler", "expected a string"), req.peer);
        if (!info.samplers && (!sname.empty() || !kname.empty()))
            return error_response(err_unsupported(sname.empty() ? "scheduler" : "sampler",
                                                  "this model samples with flow-matching Euler "
                                                  "on its own schedule"), req.peer);
        r.sampler = info.default_sampler;
        r.scheduler = info.default_scheduler;
        if (!sname.empty()) {
            r.sampler = gen::sampler_id(sname);
            if (r.sampler < 0 || !(info.samplers & (1u << r.sampler))) {
                ApiError e = err_bad("sampler", "'" + sname + "' is not a sampler this model serves");
                for (int32_t i = 0; i < RAD_GEN_SAMPLER_COUNT; ++i)
                    if (info.samplers & (1u << i)) e.supported.push_back(gen::sampler_name(i));
                return error_response(e, req.peer);
            }
        }
        if (!kname.empty()) {
            r.scheduler = gen::scheduler_id(kname);
            if (r.scheduler < 0) {
                ApiError e = err_bad("scheduler", "'" + kname + "' is not a scheduler this model "
                                                  "serves");
                for (int32_t i = 0; i < RAD_GEN_SCHED_COUNT; ++i)
                    e.supported.push_back(gen::scheduler_name(i));
                return error_response(e, req.peer);
            }
        }
    }
    /* -1 asks for a random seed, as llama-server and the chat endpoints read it. */
    if (body.contains("seed") && !body["seed"].is_null()) {
        if (!body["seed"].is_number_integer())
            return error_response(err_bad("seed", "expected an integer"), req.peer);
        r.seed = (uint64_t)body["seed"].get<int64_t>();
        r.seed_given = body["seed"].is_number_unsigned() || body["seed"].get<int64_t>() != -1;
    }
    /* The conditioning at the plugin's fixed rebalance gains (Krea 2: its text taps). true or
     * "legacy" ConditioningKrea2Rebalance's defaults; "balanced", "detail", "subtle" the
     * comfyui-krea2-conditioning presets at the taps' RMS, on a model that declares presets
     * (RAD_GEN_COND_REBALANCE_PRESET). `false` is accepted everywhere. */
    if (body.contains("rebalance") && !body["rebalance"].is_null()) {
        const json& rb = body["rebalance"];
        static const char* const kPresets[] = { "legacy", "balanced", "detail", "subtle" };
        bool ok = rb.is_boolean();
        if (ok) r.rebalance = rb.get<bool>() ? RAD_GEN_COND_REBALANCE : 0;
        else if (rb.is_string())
            for (int i = 0; i < 4; ++i)
                if (rb.get<std::string>() == kPresets[i]) {
                    r.rebalance = RAD_GEN_COND_REBALANCE | (i << RAD_GEN_COND_REBALANCE_PRESET_SHIFT);
                    ok = true;
                }
        if (!ok)
            return error_response(err_bad("rebalance", "expected true, false, \"legacy\", "
                                                       "\"balanced\", \"detail\" or \"subtle\""),
                                  req.peer);
        if (r.rebalance && !(info.cond_opts & RAD_GEN_COND_REBALANCE))
            return error_response(err_unsupported("rebalance", "this model has no rebalance"),
                                  req.peer);
        if ((r.rebalance & RAD_GEN_COND_REBALANCE_PRESET) &&
            !(info.cond_opts & RAD_GEN_COND_REBALANCE_PRESET))
            return error_response(err_unsupported("rebalance", "this model's rebalance has no "
                                                               "presets; use true"),
                                  req.peer);
    }
    if (str("negative_prompt", &r.negative) < 0)
        return error_response(err_bad("negative_prompt", "expected a string"), req.peer);
    /* The scale in the model card's terms: off at 0 for a card that states it as
     * cond + s (cond - uncond), at 1 for uncond + s (cond - uncond). */
    const double cfg_off = info.cfg_original ? 0.0 : 1.0;
    const char* cfg_key = body.contains("guidance_scale") && !body["guidance_scale"].is_null()
                              ? "guidance_scale" : "true_cfg_scale";
    const bool cfg_given = body.contains(cfg_key) && !body[cfg_key].is_null();
    if (cfg_given && body.contains("true_cfg_scale") && body.contains("guidance_scale") &&
        !body["true_cfg_scale"].is_null())
        return error_response(err_bad("guidance_scale", "true_cfg_scale and guidance_scale are "
                                                        "the same field; name one"), req.peer);
    r.cfg = info.default_cfg;
    if (cfg_given) {
        if (!body[cfg_key].is_number())
            return error_response(err_bad(cfg_key, "expected a number"), req.peer);
        r.cfg = body[cfg_key].get<double>();
        if (!(r.cfg >= cfg_off && r.cfg <= 30.0))
            return error_response(err_bad(cfg_key, "expected a number in " +
                                                       std::string(cfg_off > 0 ? "1" : "0") +
                                                       "..30"), req.peer);
    }
    /* Guidance needs a scale past its off value and a negative prompt: the request's (empty
     * becomes " " as the prompt does), or the model's own. */
    const bool guided = r.cfg > cfg_off;
    const bool named_neg = body.contains("negative_prompt") && !body["negative_prompt"].is_null();
    r.has_neg = guided && (named_neg || info.has_default_negative);
    if (r.has_neg && !named_neg) r.negative = info.default_negative;
    else if (r.has_neg && r.negative.empty() && info.min_tokens == 0) r.negative = " ";
    if (guided && !r.has_neg)
        return error_response(err_bad(cfg_key, "guidance needs a negative_prompt"), req.peer);

    /* ---- the pictures */
    if (edit) {
        if (files.empty())
            return error_response(err_bad("image", "an edit needs at least one picture"), req.peer);
        if ((int64_t)files.size() > info.max_refs)
            return error_response(err_bad("image", "at most " + std::to_string(info.max_refs) +
                                                       " pictures"), req.peer);
        for (size_t i = 0; i < files.size(); ++i) {
            mm::FrameRGBA f;
            std::string why;
            if (mm::decode_image_rgba(files[i].data(), files[i].size(), kMaxSourcePixels, &f,
                                      &why) < 0)
                return error_response(err_bad("image", "picture " + std::to_string(i + 1) +
                                                           ": " + why), req.peer);
            r.images.push_back(std::move(f));
        }
        if (have_mask) {
            std::string why;
            if (mm::decode_image_rgba(mask_bytes.data(), mask_bytes.size(), kMaxSourcePixels,
                                      &r.mask, &why) < 0)
                return error_response(err_bad("mask", why), req.peer);
            if (r.mask.w != r.images[0].w || r.mask.h != r.images[0].h)
                return error_response(err_bad("mask", "the mask must be the first picture's "
                                                      "size"), req.peer);
            r.has_mask = true;
        }
    } else if (!files.empty() || have_mask) {
        return error_response(err_bad("image", "pictures go to /v1/images/edits"), req.peer);
    }
    /* `auto`: an edit of a picture with transparency keeps it; anything else is opaque, since an
     * image generated without transparency has only decoder noise in its alpha. */
    if (r.background == "auto") {
        bool alpha = false;
        if (r.format != "jpeg" && info.pixel_channels >= 4)
            for (const mm::FrameRGBA& f : r.images)
                for (size_t i = 3; i < f.rgba.size() && !alpha; i += 4)
                    alpha = f.rgba[i] < kTransparentBelow;
        r.background = alpha ? "transparent" : "opaque";
    }

    /* ---- the size: `width`/`height`, `size` "WxH", or auto (the native area, at the last
     * picture's aspect for an edit). Sides are rounded down to the model's multiple. */
    const int32_t mult = (int32_t)info.size_multiple;
    std::string size = "auto";
    if (str("size", &size) < 0)
        return error_response(err_bad("size", "expected \"WxH\" or \"auto\""), req.peer);
    int64_t bw = 0, bh = 0;
    if (num("width", 1, 16384, &bw) < 0)
        return error_response(err_bad("width", "expected an integer in 1..16384"), req.peer);
    if (num("height", 1, 16384, &bh) < 0)
        return error_response(err_bad("height", "expected an integer in 1..16384"), req.peer);
    if ((bw > 0) != (bh > 0))
        return error_response(err_bad(bw ? "height" : "width", "width and height go together"),
                              req.peer);
    if (bw > 0) {
        r.w = (int32_t)bw;
        r.h = (int32_t)bh;
    } else if (size != "auto") {
        int a = 0, b = 0;
        char x = 0;
        if (std::sscanf(size.c_str(), "%d%c%d", &a, &x, &b) != 3 || x != 'x' || a <= 0 || b <= 0)
            return error_response(err_bad("size", "expected \"WxH\" or \"auto\""), req.peer);
        r.w = a;
        r.h = b;
    } else if (edit) {
        const mm::FrameRGBA& last = r.images.back();
        dims_for((double)info.default_width * info.default_height,
                 (double)last.w / (double)last.h, mult, &r.w, &r.h);
    } else {
        r.w = (int32_t)info.default_width;
        r.h = (int32_t)info.default_height;
    }
    r.w = r.w / mult * mult;
    r.h = r.h / mult * mult;
    if (r.w <= 0 || r.h <= 0)
        return error_response(err_bad("size", "each side at least " + std::to_string(mult) +
                                                  " pixels"), req.peer);
    const int64_t ls = info.latent_scale;
    if ((r.w / ls) * (r.h / ls) > info.max_tokens)
        return error_response(err_bad("size", std::to_string(r.w) + "x" + std::to_string(r.h) +
                                                  " is larger than this model makes (" +
                                                  std::to_string(info.max_tokens * ls * ls) +
                                                  " pixels)"), req.peer);
    if (!r.seed_given) {
        /* 53 bits: the seed is returned for reuse, and JavaScript reads JSON numbers as doubles,
         * so a wider seed would not round-trip. */
        std::random_device rd;
        r.seed = (((uint64_t)rd() << 32) ^ rd()) & (((uint64_t)1 << 53) - 1);
    }

    /* ---- the request: prompts rendered and tokenised, pictures resized */
    auto g = std::make_unique<gen::Request>();
    g->width = r.w;
    g->height = r.h;
    g->n = r.n;
    g->steps = r.steps;
    g->sampler = r.sampler;
    g->scheduler = r.scheduler;
    g->seed = r.seed;
    g->previews = r.previews;
    g->cond_opts = r.rebalance;
    /* The driver guides as uncond + cfg (cond - uncond); a card's cond + s (cond - uncond) is
     * that at s + 1. */
    g->cfg = (float)(info.cfg_original ? r.cfg + 1.0 : r.cfg);
    g->has_neg = r.has_neg;
    g->ticket = std::make_shared<gen::Ticket>();

    std::vector<std::shared_ptr<mm::Item>> items;
    const int pch = (int)info.pixel_channels;
    for (const mm::FrameRGBA& f : r.images) {
        int32_t rw = 0, rh = 0;
        mm::FrameRGBA rs;
        if (info.ref_shrink_only) {
            /* Shrunk to the reference area only when larger, then cut to the size multiple. */
            const double area = (double)f.w * f.h;
            if (area > (double)info.ref_area) {
                const double k = std::sqrt((double)info.ref_area / area);
                rw = std::max<int32_t>(1, (int32_t)(f.w * k));
                rh = std::max<int32_t>(1, (int32_t)(f.h * k));
                mm::resize_lanczos_rgba(f, rw, rh, &rs);
            } else {
                rs = f;
            }
            const int32_t cw = rs.w / mult * mult, ch = rs.h / mult * mult;
            if (cw <= 0 || ch <= 0)
                return error_response(err_bad("image", "a picture must be at least " +
                                                           std::to_string(mult) + " pixels a side"),
                                      req.peer);
            if (cw != rs.w || ch != rs.h) rs = crop_centre(rs, cw, ch);
        } else {
            dims_for((double)info.ref_area, (double)f.w / (double)f.h, mult, &rw, &rh);
            mm::resize_lanczos_rgba(f, rw, rh, &rs);
        }
        g->refs.push_back(to_pixels(rs, pch));
        if (info.refs_encoder_only) continue;
        mm::Frame rgb;
        mm::composite_white(rs, &rgb);
        std::shared_ptr<mm::Item> it;
        std::string why;
        if (deps_.images->picture(rgb, &it, &why) < 0 || !it)
            return error_response(err_bad("image", "the picture could not be used: " + why),
                                  req.peer);
        items.push_back(std::move(it));
    }
    /* Repainting: the first picture at the output size, and per latent token the fraction of
     * its ls x ls pixels the mask leaves transparent. Those are repainted, the rest kept. */
    std::vector<uint8_t> keep_px, src_px;
    if (r.has_mask) {
        mm::FrameRGBA src, msk;
        mm::resize_lanczos_rgba(r.images[0], r.w, r.h, &src);
        mm::resize_lanczos_rgba(r.mask, r.w, r.h, &msk);
        g->inpaint = true;
        g->inp_src = to_pixels(src, pch);
        const int32_t lw = r.w / (int32_t)ls, lh = r.h / (int32_t)ls;
        g->inp_mask.assign((size_t)lw * lh, 0.0f);
        keep_px.assign((size_t)r.w * r.h, 0);
        for (int32_t y = 0; y < r.h; ++y)
            for (int32_t x = 0; x < r.w; ++x) {
                const bool edit_px = msk.rgba[((size_t)y * r.w + x) * 4 + 3] < kTransparentBelow;
                keep_px[(size_t)y * r.w + x] = edit_px ? 0 : 1;
                if (edit_px) g->inp_mask[(size_t)(y / ls) * lw + x / ls] += 1.0f;
            }
        for (float& m : g->inp_mask) m /= (float)(ls * ls);
        src_px = std::move(src.rgba);
    }

    auto render = [&](const std::string& prompt, const char* field, gen::Branch* out,
                      ApiError* err) -> int {
        /* A prompt spelling the media marker would be scanned for pictures this request never
         * uploaded, and the scan would walk off the item list: refuse the text, as the chat and
         * embeddings paths refuse marker text (oai.cpp). */
        if (prompt.find(RAD_MEDIA_MARKER) != std::string::npos) {
            *err = err_bad(field, std::string("contains \"") + RAD_MEDIA_MARKER +
                                        "\": it names a picture this request did not upload");
            return RAD_E_INVAL;
        }
        std::string text = r.background == "transparent" && !info.tmpl_transparent.empty()
                               ? replace_all(info.tmpl_transparent, "{prompt}", prompt)
                               : prompt;
        std::string tmpl = edit ? info.tmpl_edit : info.tmpl_text;
        if (edit) {
            std::string imgs;
            for (size_t i = 0; i < items.size(); ++i) {
                if (i) imgs += info.tmpl_image_sep;
                imgs += replace_all(replace_all(info.tmpl_image, "{n}", std::to_string(i + 1)),
                                    "{image}", RAD_MEDIA_MARKER);
            }
            tmpl = replace_all(tmpl, "{images}", imgs);
        }
        const std::string rendered = replace_all(tmpl, "{prompt}", text);
        /* Tokenised around the markers, each marker replaced by its picture's token run: the
         * chat path's tokenize_with_media, with the pictures already processed. */
        auto pm = std::make_shared<mm::PromptMedia>();
        size_t pos = 0, mi = 0;
        for (;;) {
            const size_t at = rendered.find(RAD_MEDIA_MARKER, pos);
            const std::string seg = rendered.substr(pos, at == std::string::npos ? std::string::npos
                                                                                 : at - pos);
            if (!seg.empty()) {
                auto ids = deps_.tok->encode(seg, false, /*parse_special=*/true);
                out->tokens.insert(out->tokens.end(), ids.begin(), ids.end());
            }
            if (at == std::string::npos) break;
            if (mi >= items.size()) {
                /* Unreachable for any request: the prompt text is refused above, so only a model
                 * template that spells the marker past its pictures could reach it. Belt for the
                 * day one does. */
                *err = err_internal("the render template spells the media marker more times "
                                    "than the request has pictures");
                return RAD_E_STATE;
            }
            mm::Placed pl;
            pl.item = items[mi++];
            pl.tok = (int32_t)out->tokens.size();
            out->tokens.insert(out->tokens.end(), pl.item->expand.begin(), pl.item->expand.end());
            pm->items.push_back(std::move(pl));
            pos = at + std::strlen(RAD_MEDIA_MARKER);
        }
        if (!pm->items.empty()) {
            pm->layout();
            out->media = std::move(pm);
        }
        /* The fixed turn the generator drops; the prompt must start with it. */
        const auto drop = deps_.tok->encode(info.drop_prefix, false, true);
        if (drop.size() > out->tokens.size() ||
            !std::equal(drop.begin(), drop.end(), out->tokens.begin())) {
            /* The model's own templates disagree; nothing the request did. */
            *err = err_internal("the prompt does not start with the model's fixed turn");
            return RAD_E_STATE;
        }
        out->n_drop = (int32_t)drop.size();
        /* The reference tokenizer's minimum length: an empty prompt is padding the model reads. */
        while ((int64_t)out->tokens.size() - out->n_drop < info.min_tokens)
            out->tokens.push_back(info.pad_token);
        if (info.aux_tokens) {
            if (deps_.images->encode_aux(prompt, &out->aux) < 0) {
                *err = err_internal("the prompt could not be read by the model's second "
                                    "tokenizer");
                return RAD_E_STATE;
            }
            if ((int64_t)out->aux.size() > info.aux_max_tokens) {
                *err = err_bad("prompt", "the prompt is " + std::to_string(out->aux.size()) +
                                             " tokens to this model's second tokenizer, which "
                                             "reads at most " +
                                             std::to_string(info.aux_max_tokens));
                return RAD_E_INVAL;
            }
        }
        /* A prompt read in windows: runs of at most chunk_tokens after the prefix, each between
         * its markers and padded to the window. */
        if (info.chunk_tokens > 0) {
            const size_t W = (size_t)info.chunk_tokens;
            std::vector<int32_t> win(out->tokens.begin(), out->tokens.begin() + out->n_drop);
            const size_t n_body = out->tokens.size() - (size_t)out->n_drop;
            size_t at = 0;
            do {
                const size_t take = std::min(W, n_body - at);
                win.push_back(info.chunk_bos);
                win.insert(win.end(), out->tokens.begin() + out->n_drop + (std::ptrdiff_t)at,
                           out->tokens.begin() + out->n_drop + (std::ptrdiff_t)(at + take));
                win.push_back(info.chunk_eos);
                win.resize(win.size() + (W - take), info.chunk_pad);
                at += take;
            } while (at < n_body);
            out->tokens = std::move(win);
        }
        if ((int64_t)out->tokens.size() > info.max_cond_tokens) {
            *err = err_bad("prompt", "the prompt is " + std::to_string(out->tokens.size()) +
                                         " tokens with its pictures; this deployment conditions on "
                                         "at most " + std::to_string(info.max_cond_tokens));
            return RAD_E_INVAL;
        }
        return RAD_OK;
    };
    {
        ApiError err;
        if (render(r.prompt, "prompt", &g->pos, &err) < 0) return error_response(err, req.peer);
        if (r.has_neg && render(r.negative, "negative_prompt", &g->neg, &err) < 0)
            return error_response(err, req.peer);
    }

    /* ---- the adapters' slot, held until the request is done */
    std::string lora_note;
    if (!loras.empty()) {
        std::string why;
        const int s = opt_.gen_lora_acquire(loras, &g->lora_slot, &g->lora_hold, &lora_note, &why);
        if (s < 0) {
            ApiError e = err_bad("loras", why);
            if (s == RAD_E_NOTFOUND) { e.status = 404; e.type = "not_found_error"; }
            else if (s != RAD_E_INVAL) { e.status = 503; e.type = "server_error"; }
            return error_response(e, req.peer);
        }
    }

    /* ---- admitted and queued */
    if (auto no = admit(1, req.peer)) return *no;
    std::shared_ptr<gen::Ticket> ticket = g->ticket;
    {
        std::string why;
        const int s = deps_.images->submit(std::move(g), &why);
        if (s < 0) {
            release(1);
            ApiError e = err_bad("", why.empty() ? std::string(rad_strerror(s)) : why);
            if (s != RAD_E_INVAL) { e.status = 503; e.type = "server_error"; }
            return error_response(e, req.peer);
        }
    }
    std::string lora_s;
    for (const auto& [n, w] : loras) lora_s += (lora_s.empty() ? " loras=" : ",") + n + fmt(":%g", w);
    RAD_INFO("images: %s %dx%d n=%d steps=%d seed=%llu%s%s%s%s", edit ? "edit" : "generation", r.w,
             r.h, r.n, r.steps, (unsigned long long)r.seed, r.has_neg ? " guided" : "",
             r.has_mask ? " masked" : "", r.stream ? " streamed" : "", lora_s.c_str());

    /* ---- one finished picture to a file */
    const std::string size_s = std::to_string(r.w) + "x" + std::to_string(r.h);
    const bool flatten = r.format == "jpeg" || r.background == "opaque";
    const std::string fmt = r.format;
    const int32_t comp = r.compression;
    auto encode = [fmt, comp, keep_px, src_px, flatten](gen::Image& im, std::string* b64,
                                                        std::string* why) -> int {
        std::vector<uint8_t>& px = im.px;
        const size_t ch = (size_t)im.ch;
        if (!keep_px.empty() && !im.preview)
            for (size_t i = 0; i < keep_px.size(); ++i)
                if (keep_px[i]) std::memcpy(&px[i * ch], &src_px[i * 4], ch);
        std::vector<uint8_t> file;
        int s;
        if (ch == 3) {
            s = mm::encode_image(px.data(), im.w, im.h, 3, fmt, comp, &file, why);
        } else if (flatten) {
            const std::vector<uint8_t> rgb = on_white(px, im.w, im.h);
            s = mm::encode_image(rgb.data(), im.w, im.h, 3, fmt, comp, &file, why);
        } else {
            s = mm::encode_image(px.data(), im.w, im.h, 4, fmt, comp, &file, why);
        }
        if (s < 0) return s;
        *b64 = base64_encode(file.data(), file.size());
        return RAD_OK;
    };
    const std::string bg_out = flatten ? "opaque" : "transparent";
    /* The pictures' share of the input: their token runs, in every branch that carries them. */
    int64_t image_tok = 0;
    for (const auto& it : items) image_tok += (int64_t)it->expand.size();
    if (r.has_neg) image_tok *= 2;
    auto usage = [ticket, image_tok]() {
        json u;
        const int64_t in = ticket->in_tokens, out = ticket->out_tokens;
        u["total_tokens"] = in + out;
        u["input_tokens"] = in;
        u["output_tokens"] = out;
        u["input_tokens_details"] = json{ { "text_tokens", in - image_tok },
                                          { "image_tokens", image_tok } };
        return u;
    };
    const int64_t created = (int64_t)std::time(nullptr);

    if (!r.stream) {
        std::vector<gen::Image> done;
        for (;;) {
            gen::Image im;
            bool to = false;
            if (ticket->next(&im, 200, &to)) {
                if (!im.preview) done.push_back(std::move(im));
                continue;
            }
            if (!to) break;
            if (req.peer_gone()) {
                ticket->cancel.store(true);
                release(1);
                metrics_.observe_http(499);
                return HttpResponse::text(499, "client closed request\n");
            }
            if (stopping_.load()) {
                ticket->cancel.store(true);
                release(1);
                return error_response(err_overloaded("the server is shutting down"), req.peer);
            }
        }
        release(1);
        if (ticket->status() < 0 || (int32_t)done.size() != r.n) {
            /* RAD_E_INVAL is the driver's admission refusal (a request this deployment can never
             * serve, e.g. conditioning longer than the cache), so a 400; anything else failed
             * while running. */
            ApiError e;
            const bool client = ticket->status() == RAD_E_INVAL;
            e.status = client ? 400 : 500;
            e.type = client ? "invalid_request_error" : "server_error";
            e.message = ticket->why().empty() ? "the image generator failed" : ticket->why();
            return error_response(e, req.peer);
        }
        std::sort(done.begin(), done.end(),
                  [](const gen::Image& a, const gen::Image& b) { return a.index < b.index; });
        json out;
        out["created"] = created;
        json data = json::array();
        for (gen::Image& im : done) {
            std::string b64, why;
            if (encode(im, &b64, &why) < 0) {
                ApiError e;
                e.status = 500;
                e.type = "server_error";
                e.message = "the picture could not be encoded: " + why;
                return error_response(e, req.peer);
            }
            data.push_back(json{ { "b64_json", std::move(b64) }, { "revised_prompt", r.prompt },
                                 { "seed", im.seed } });
        }
        out["data"] = std::move(data);
        out["background"] = bg_out;
        out["output_format"] = r.format;
        out["quality"] = r.quality;
        out["size"] = size_s;
        out["usage"] = usage();
        metrics_.observe_http(200);
        return HttpResponse::json_body(200, json_dump(out));
    }

    /* ---- streamed: partial images as previews arrive, one completed event per image */
    HttpResponse res;
    res.status = 200;
    res.content_type = "text/event-stream";
    res.set_header("Cache-Control", "no-cache");
    const std::string kind = edit ? "image_edit" : "image_generation";
    auto finished = std::make_shared<bool>(false);
    auto released = std::make_shared<bool>(false);
    const std::string quality = r.quality;
    res.next = [this, ticket, kind, encode, created, size_s, bg_out, fmt, quality, usage, finished,
                released](std::string& chunk) mutable -> bool {
        if (*finished) return false;
        gen::Image im;
        bool to = false;
        if (ticket->next(&im, 100, &to)) {
            std::string b64, why;
            json ev;
            if (encode(im, &b64, &why) < 0) {
                ev["type"] = "error";
                ev["error"] = json{ { "message", "the picture could not be encoded: " + why } };
            } else {
                ev["type"] = kind + (im.preview ? ".partial_image" : ".completed");
                ev["b64_json"] = std::move(b64);
                ev["created_at"] = created;
                ev["size"] = size_s;
                ev["quality"] = quality;
                ev["background"] = bg_out;
                ev["output_format"] = fmt;
                ev["index"] = im.index;
                if (im.preview) ev["partial_image_index"] = im.preview_no;
                else { ev["usage"] = usage(); ev["seed"] = im.seed; }
            }
            chunk = "event: " + ev["type"].get<std::string>() + "\ndata: " + json_dump(ev) + "\n\n";
            return true;
        }
        if (to) { chunk.clear(); return !stopping_.load(); }
        *finished = true;
        if (!*released) { *released = true; release(1); }
        if (ticket->status() < 0) {
            json ev;
            ev["type"] = "error";
            ev["error"] = json{ { "message", ticket->why() },
                                { "type", ticket->status() == RAD_E_INVAL ? "invalid_request_error"
                                                                          : "server_error" } };
            chunk = "event: error\ndata: " + json_dump(ev) + "\n\n";
            return true;
        }
        chunk.clear();
        return false;
    };
    res.on_close = [this, ticket, released](bool ok) {
        if (!ok) ticket->cancel.store(true);
        if (!*released) { *released = true; release(1); }
    };
    metrics_.observe_http(200);
    return res;
}

}  /* namespace server */
}  /* namespace rad */
