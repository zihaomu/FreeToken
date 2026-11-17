/* videos.cpp -- OpenAI's video endpoints over a video generator (core/gen/video_driver.h): a video
 * and its soundtrack from a prompt.
 *
 * A video takes minutes, so it is a job, as OpenAI's API has it: POST /v1/videos queues one and
 * answers with the video object at once, GET /v1/videos/{id} reports its status and progress, and
 * GET /v1/videos/{id}/content downloads the MP4 once it is completed (`variant` thumbnail gives
 * the first frame as WebP). GET /v1/videos lists the jobs, newest first, and DELETE removes one,
 * cancelling it when it is still running. A finished video is kept for `video_ttl_s` seconds
 * (`expires_at`), at most `max_videos` of them.
 *
 * The request takes OpenAI's fields (`prompt`, `model`, `seconds`, `size`) as JSON or as a
 * multipart form, and SGLang's for the same model: `task` (t2va), `target` {`short_edge`,
 * `aspect_ratio`, `duration_seconds`}, `conditions` (empty), `seed`, and `num_inference_steps`.
 * `size` is the frame's width x height, multiples of the model's size multiple; `aspect_ratio`
 * (`w:h`) instead lays the model's own canvas over that ratio: the short edge (the model's, or
 * `short_edge`), the area capped at the model's largest, both sides rounded to the multiple.
 * A video's length is a whole number of the model's chunks: `seconds` is rounded up to the next.
* A model with the VAEs' encoders also takes keyframes, form file parts `first_frame` and
 * `last_frame` (tasks i2va, l2va, fl2va), or reference media (task ref2va): `ref_image_N` (at
 * most 9), `ref_video_N` (3) with a soundtrack `ref_video_audio_N` and `ref_audio_N` (3), N from 1
 * without gaps within each kind. The parts carry the bytes: `input_reference` (but for i2va on a
 * model that starts from a picture) and non-empty `conditions` are refused.
 * A model with a length head (LTX-2.5's duration head) chooses the length of a request that names
 * none, or says `"seconds": "auto"`; the video object reports it once chosen. A model with more
 * than one decoder takes `decoder`, one of their names (LTX-2.5: `vae`, the default, or
 * `diffusion`).
 * A model that can start from a picture takes OpenAI's `input_reference`, a file in the form or a
 * data URL (a string or {image_url}) in JSON, as the video's first frame, prepared as the model's
 * pipeline prepares it: an H.264 round trip at the model's quality, then a bilinear resize to the
 * size the first steps run at. Non-empty `conditions` are refused.
 *
 * A guided model takes `negative_prompt` and its guidance terms, each the model's default when
 * absent: `guidance_scale`, `stg_scale`, `modality_scale` and `guidance_rescale` for the video,
 * the same with `audio_` in front for the soundtrack. A scale of 1 turns that modality's
 * unconditional branch off; the negative prompt is the model's own unless named.
 *
 * A model with a latent upscaler also takes `latent_upscale`, a factor from 1 to its largest: the
 * first steps run at `size` divided by it and the latent is enlarged to `size` for the last
 * `upscale_refine_steps` (the model's default when absent; abi/rad_builder.h, RadVideoGenDecl).
 * The video object of an upscaled video says how it was made: the factor, the refine steps and the
 * small size (`upscale_from`). Keyframes ride it: each is prepared at both sizes and pinned at the
 * one the steps run at.
 *
 * The prompt is the model's template with `{prompt}` filled; the template is tokenised with its
 * special tokens and the prompt without, so a caller cannot write control tokens into it.
 */
#include <atomic>

#include "front.h"
#include "server.h"
#include "loras.h"
#include "oai.h"

#include "gen/video.h"
#include "mm/imaging.h"
#include "mm/media.h"
#include "mm/mp4.h"
#include "rad_internal.h"

#include "util/json.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <thread>
#include <vector>

namespace rad {
namespace server {

struct Server::VideoJob {
    std::string id, model, prompt, seconds, size;
    int64_t     created_at = 0;
    uint64_t    seed = 0;
    int32_t     sampler = 0;   /* the request's RAD_SAMPLER_* */
    double      shift_video = 0, shift_audio = 0;   /* 0: the model's */
    std::shared_ptr<gen::VideoTicket> ticket;
    std::thread worker;
    /* Under Server::videos_m_ from here on. */
    std::string status = "queued";
    int         progress = 0;
    /* The steps done of the total (0 before the first), and the first one's steady-clock time in
     * ns, which the per-step log line measures its rate from. */
    int         steps_done = 0, steps_total = 0;
    int64_t     step_ns0 = 0;
    int64_t     completed_at = 0, expires_at = 0;
    bool        failed = false;
    std::string err_code, err_message;
    std::string mp4;
    std::vector<uint8_t> first;   /* the first frame, RGB, for the thumbnail */
    /* A job with reference media or keyframes: its task and its parts in conditioning order. */
    std::string task;
    std::vector<std::string> refs;
    /* Latent upscaling, as the generator resolved it: the factor, the evaluations after it and the
     * small size; refine 0 when the video is not upscaled. */
    double      upscale = 0.0;
    int32_t     refine = 0;
    std::string upscale_from;
    /* A guided model's terms as resolved against its defaults, the negative prompt when a scale
     * used it, and whether the video started from a picture. */
    bool        guided = false;
    double      guidance[8] = {};
    std::string negative;
    bool        has_negative = false;
    bool        from_image = false;
    std::string decoder;          /* a decoder the request named other than the default */
    int32_t     w = 0, h = 0;
    bool        worker_done = false;   /* its thread has nothing left to do but return */
    /* Deleted, by the server or through the Front: nothing answers for it from here on. */
    std::atomic<bool> deleted{false};
};

namespace {

/* The largest picture `input_reference` may decode to, which bounds the memory a request can make
 * the server allocate before the resize. */
constexpr int64_t kMaxRefPixels = (int64_t)1 << 26;

std::string new_video_id() {
    static const char* hex = "0123456789abcdef";
    std::random_device rd;
    std::string id = "video_";
    for (int i = 0; i < 32; ++i) id.push_back(hex[rd() & 15]);
    return id;
}

/* "1344x768" -> (1344, 768); false for anything else. */
bool parse_size(const std::string& s, int64_t* w, int64_t* h) {
    const size_t x = s.find_first_of("xX");
    if (x == std::string::npos || x == 0 || x + 1 >= s.size()) return false;
    char* e = nullptr;
    *w = std::strtoll(s.c_str(), &e, 10);
    if (e != s.c_str() + x) return false;
    *h = std::strtoll(s.c_str() + x + 1, &e, 10);
    return *e == '\0' && *w > 0 && *h > 0;
}

/* "16:9" -> 16/9 as two positives; false for anything else. */
bool parse_ratio(const std::string& s, double* a, double* b) {
    const size_t c = s.find(':');
    if (c == std::string::npos) return false;
    char* e = nullptr;
    *a = std::strtod(s.c_str(), &e);
    if (e != s.c_str() + c) return false;
    *b = std::strtod(s.c_str() + c + 1, &e);
    return *e == '\0' && *a > 0 && *b > 0;
}

/* The model's canvas for a ratio: the short edge, the area capped, both sides rounded to the
 * multiple (ties to even, as the reference's float round). */
void canvas_for(double rw, double rh, int64_t short_edge, int64_t max_area, int64_t mult,
                int64_t* w, int64_t* h) {
    const double ratio = rw / rh;
    double W = ratio >= 1.0 ? short_edge * ratio : (double)short_edge;
    double H = ratio >= 1.0 ? (double)short_edge : short_edge / ratio;
    if (W * H > (double)max_area) {
        const double sc = std::sqrt((double)max_area / (W * H));
        W *= sc;
        H *= sc;
    }
    *w = std::max<int64_t>(mult, (int64_t)std::nearbyint(W / mult) * mult);
    *h = std::max<int64_t>(mult, (int64_t)std::nearbyint(H / mult) * mult);
}

/* The most reference media of each kind a request carries, ComfyUI's ref2va workflow's; a model
 * may take fewer. */
constexpr int32_t kMaxRefImages = 9, kMaxRefVideos = 3, kMaxRefAudios = 3;

/* A reference part's kind and number from its name; false for any other name. `ref_video_audio_`
 * is tried before `ref_video_`, which it starts with. */
bool parse_ref_name(const std::string& name, gen::VideoRefMedia::Kind* kind, int32_t* index) {
    static const struct { const char* pfx; gen::VideoRefMedia::Kind kind; } kKinds[] = {
        { "ref_image_",       gen::VideoRefMedia::IMAGE       },
        { "ref_video_audio_", gen::VideoRefMedia::VIDEO_AUDIO },
        { "ref_video_",       gen::VideoRefMedia::VIDEO       },
        { "ref_audio_",       gen::VideoRefMedia::AUDIO       },
    };
    for (const auto& k : kKinds) {
        const size_t n = std::strlen(k.pfx);
        if (name.compare(0, n, k.pfx) != 0 || name.size() == n) continue;
        char* e = nullptr;
        const long v = std::strtol(name.c_str() + n, &e, 10);
        if (*e == '\0' && v >= 1 && v <= 99) { *kind = k.kind; *index = (int32_t)v; return true; }
    }
    return false;
}

/* A ref2va request's parts, refused naming the part when one is empty, over its kind's count,
 * numbered with a gap or twice, or a soundtrack with no video of its number. On return `refs` is
 * sorted by (kind, number), the order of the conditioning. */
int validate_refs(std::vector<gen::VideoRefMedia>& refs, const VideoGenInfo& info, ApiError* err) {
    if (refs.empty()) {
        *err = err_bad("task", "ref2va takes at least one reference part: ref_image_N, "
                               "ref_video_N, ref_video_audio_N or ref_audio_N");
        return -1;
    }
    const struct {
        const char* pfx;
        gen::VideoRefMedia::Kind kind;
        int32_t cap;
    } kinds[] = {
        { "ref_image",       gen::VideoRefMedia::IMAGE,
          std::min(kMaxRefImages, (int32_t)info.max_ref_images) },
        { "ref_video",       gen::VideoRefMedia::VIDEO,
          std::min(kMaxRefVideos, (int32_t)info.max_ref_videos) },
        { "ref_video_audio", gen::VideoRefMedia::VIDEO_AUDIO,
          std::min(kMaxRefVideos, (int32_t)info.max_ref_videos) },
        { "ref_audio",       gen::VideoRefMedia::AUDIO,
          std::min(kMaxRefAudios, (int32_t)info.max_ref_audios) },
    };
    for (const auto& k : kinds) {
        std::vector<bool> got(k.cap + 1, false);
        int n = 0;
        for (const gen::VideoRefMedia& m : refs) {
            if (m.kind != k.kind) continue;
            ++n;
            if (m.data.empty())
                return *err = err_bad(m.name, "'" + m.name + "' carries an empty file"), -1;
            if (m.index > k.cap)
                return *err = err_bad(m.name, k.cap == 0
                                                  ? "this model takes no parts named " +
                                                    std::string(k.pfx) + "_N"
                                                  : "this model takes at most " +
                                                    std::to_string(k.cap) + " parts named " +
                                                    std::string(k.pfx) + "_N"), -1;
            if (got[m.index])
                return *err = err_bad(m.name, "two parts are numbered " + std::string(k.pfx) +
                                                  "_" + std::to_string(m.index)), -1;
            got[m.index] = true;
        }
        for (int i = 1; i <= n; ++i)
            if (!got[i])
                return *err = err_bad(std::string(k.pfx) + "_" + std::to_string(i),
                                      "the " + std::string(k.pfx) + "_N parts must be numbered 1 "
                                      "to " + std::to_string(n) + " without gaps or repeats"), -1;
    }
    for (const gen::VideoRefMedia& m : refs) {
        if (m.kind != gen::VideoRefMedia::VIDEO_AUDIO) continue;
        bool paired = false;
        for (const gen::VideoRefMedia& v : refs)
            paired = paired || (v.kind == gen::VideoRefMedia::VIDEO && v.index == m.index);
        if (!paired)
            return *err = err_bad(m.name, "'" + m.name + "' is the soundtrack of a reference "
                                          "video; give ref_video_" + std::to_string(m.index) +
                                          " with it"), -1;
    }
    std::stable_sort(refs.begin(), refs.end(),
                     [](const gen::VideoRefMedia& a, const gen::VideoRefMedia& b) {
                         return a.kind != b.kind ? a.kind < b.kind : a.index < b.index;
                     });
    return 0;
}

}  /* namespace */

/* A video model's capabilities on /v1/models and /server_info: the counts validate_refs applies,
 * the samplers and the model's shifts. */
void Server::video_caps(json& caps, const IVideoGen* gen) {
    const VideoGenInfo& info = gen->info();
    caps["video_max_text"]       = info.max_text;
    caps["video_max_ref_images"] = std::min(kMaxRefImages, (int32_t)info.max_ref_images);
    caps["video_max_ref_videos"] = std::min(kMaxRefVideos, (int32_t)info.max_ref_videos);
    caps["video_max_ref_audios"] = std::min(kMaxRefAudios, (int32_t)info.max_ref_audios);
    caps["video_samplers"] = json::array({ "euler", "res_multistep" });
    caps["video_shift"]       = info.shift_video;
    caps["video_audio_shift"] = info.shift_audio;
}

HttpResponse Server::handle_video_create(const HttpRequest& req) {
    if (!deps_.videos || !deps_.tok)
        return error_response(err_not_implemented(
            "/v1/videos requires a video generation model; this one does not make video"),
                              req.peer);
    const VideoGenInfo& info = deps_.videos->info();

    const bool takes_refs = info.max_ref_images > 0 || info.max_ref_videos > 0 ||
                            info.max_ref_audios > 0;
    /* JSON, or a form whose fields are strings (OpenAI's clients send the create call as a form,
     * since it may carry a file). Reference media and keyframes are file parts, collected here and
     * checked once the task is known. */
    std::vector<gen::VideoRefMedia> refs;
    /* The keyframes, `first_frame` and `last_frame`. */
    std::vector<gen::VideoRefMedia> keys;
    json body = json::object();
    std::vector<uint8_t> ref_bytes;
    bool has_ref = false;
    if (!req.form.empty()) {
        for (const FormPart& p : req.form) {
            if (p.name == "input_reference" && !p.filename.empty()) {
                ref_bytes.assign(p.data.begin(), p.data.end());
                has_ref = true;
                continue;
            }
            if (p.name == "first_frame" || p.name == "last_frame") {
                if (p.filename.empty())
                    return error_response(err_bad(p.name, "'" + p.name + "' must carry an image "
                                                          "file"), req.peer);
                gen::VideoRefMedia m;
                m.kind = p.name == "first_frame" ? gen::VideoRefMedia::FIRST_FRAME
                                                 : gen::VideoRefMedia::LAST_FRAME;
                m.name = p.name;
                m.filename = p.filename;
                m.content_type = p.content_type;
                m.data = p.data;
                keys.push_back(std::move(m));
                continue;
            }
            if (!p.filename.empty()) {
                gen::VideoRefMedia m;
                if (!parse_ref_name(p.name, &m.kind, &m.index))
                    return error_response(err_bad(p.name, "'" + p.name + "' takes text, not a "
                                                          "file"), req.peer);
                m.name = p.name;
                m.filename = p.filename;
                m.content_type = p.content_type;
                m.data = p.data;
                refs.push_back(std::move(m));
                continue;
            }
            json v = p.data;
            if (p.name == "target" || p.name == "conditions" ||
                ((p.name == "loras" || p.name == "lora") && !p.data.empty() &&
                 (p.data[0] == '[' || p.data[0] == '{'))) {
                /* Bounded like every other untrusted parse: a form field under the body cap can
                 * be brackets alone, and the deep tree's teardown would take the process with
                 * it. Over-budget or over-deep reads as a bad parse. */
                bool over = false, deep = false;
                v = parse_json_bounded(p.data, 0, &over, &deep);
                if (v.is_discarded() || over || deep)
                    return error_response(err_bad(p.name, "expected JSON"), req.peer);
            }
            body[p.name] = v;
        }
    } else {
        ApiError err;
        if (parse_request_object(req.body, lim_, &body, &err) < 0)
            return error_response(err, req.peer);
    }
    static const char* kKnown[] = {
        "model", "prompt", "seconds", "size", "input_reference", "seed", "num_inference_steps",
        "task", "target", "conditions", "user", "latent_upscale", "upscale_refine_steps",
        "sampler", "shift", "audio_shift", "first_frame_mode",
        "negative_prompt", "decoder", "loras", "lora",
        /* RadVideoGenBatch::guidance_h's order */
        "guidance_scale", "stg_scale", "modality_scale", "guidance_rescale",
        "audio_guidance_scale", "audio_stg_scale", "audio_modality_scale", "audio_guidance_rescale",
    };
    for (auto it = body.begin(); it != body.end(); ++it) {
        bool known = false;
        for (const char* k : kKnown) known = known || it.key() == k;
        if (!known)
            return error_response(err_bad(it.key(), "'" + it.key() + "' is not a field of "
                                                     "/v1/videos"), req.peer);
    }
    auto bad = [&](const char* k, const std::string& why) { return error_response(err_bad(k, why), req.peer); };
    auto has = [&](const json& o, const char* k) { return o.contains(k) && !o[k].is_null(); };
    /* A number, or a string holding one (a form's fields are strings). */
    auto number = [&](const json& v, double* out) {
        if (v.is_number()) { *out = v.get<double>(); return true; }
        if (!v.is_string()) return false;
        const std::string s = v.get<std::string>();
        char* e = nullptr;
        *out = std::strtod(s.c_str(), &e);
        return !s.empty() && *e == '\0';
    };

    if (!has(body, "prompt") || !body["prompt"].is_string() ||
        body["prompt"].get<std::string>().find_first_not_of(" \t\r\n") == std::string::npos)
        return bad("prompt", "the prompt is required, as a non-empty string");
    std::string prompt = body["prompt"].get<std::string>();
    /* Runtime LoRA adapters: `loras` and <lora:...> tags, which leave the prompt. */
    std::vector<std::pair<std::string, double>> loras;
    {
        std::string field, why;
        if (parse_loras(body, &prompt, &loras, &field, &why) < 0) return bad(field.c_str(), why);
        if (!loras.empty() && !opt_.gen_lora_acquire)
            return error_response(err_unsupported(field.c_str(), "this server was started without "
                                                                 "adapters (--lora, --lora-dir)"),
                                  req.peer);
    }
    if (has(body, "input_reference")) {
        /* JSON: a data URL, as a string or an image_url object. */
        const json& v = body["input_reference"];
        std::string u;
        if (v.is_string()) u = v.get<std::string>();
        else if (v.is_object() && v.contains("image_url") && v["image_url"].is_string())
            u = v["image_url"].get<std::string>();
        else if (v.is_object() && v.contains("image_url") && v["image_url"].is_object() &&
                 v["image_url"].contains("url") && v["image_url"]["url"].is_string())
            u = v["image_url"]["url"].get<std::string>();
        else
            return bad("input_reference", "a picture is a file, a data: URL, or an object with an "
                                          "image_url");
        ApiError err;
        if (decode_media_url(u, "input_reference", &ref_bytes, &err) < 0)
            return error_response(err, req.peer);
        has_ref = true;
    }
    if (has_ref && !info.image_cond)
        return error_response(err_unsupported("input_reference", takes_refs
                                              ? "this server reads reference media as ref_image_N "
                                                "file parts"
                                              : "this model makes video from text only"), req.peer);
    /* With `input_reference` the task is i2va. */
    if (has_ref && has(body, "task") && (!body["task"].is_string() ||
                                         body["task"].get<std::string>() != "i2va")) {
        ApiError e = err_unsupported("task", "a video from a picture is i2va");
        e.supported = { "i2va" };
        return error_response(e, req.peer);
    }
    /* Otherwise t2va, or what the model takes: ref2va with reference media, the keyframe tasks
     * with a video encoder. */
    bool ref2va = false;
    std::string kf_task;   /* a keyframe task named */
    if (!has_ref && has(body, "task")) {
        const std::string task = body["task"].is_string() ? body["task"].get<std::string>() : "";
        if (task == "ref2va" && takes_refs) ref2va = true;
        /* The parts say which frames there are; a named task is checked against them below. */
        else if (info.max_enc_side > 0 && (task == "i2va" || task == "l2va" || task == "fl2va"))
            kf_task = task;
        else if (task != "t2va") {
            std::string what = "this server makes video from text";
            std::vector<std::string> sup{ "t2va" };
            if (info.image_cond) { what += ", from a picture"; sup.push_back("i2va"); }
            if (info.max_enc_side > 0) {
                what += ", or between keyframes";
                for (const char* k : { "i2va", "l2va", "fl2va" })
                    if (std::find(sup.begin(), sup.end(), std::string(k)) == sup.end())
                        sup.push_back(k);
            }
            if (takes_refs) { what += ", from reference media"; sup.push_back("ref2va"); }
            what += ": ";
            for (size_t i = 0; i < sup.size(); ++i) what += (i ? ", " : "") + sup[i];
            ApiError e = err_unsupported("task", what);
            e.supported = sup;
            return error_response(e, req.peer);
        }
    }
    if (has(body, "conditions") && (!body["conditions"].is_array() || !body["conditions"].empty()))
        return error_response(err_unsupported("conditions", takes_refs
                                              ? "reference media arrive as ref_* file parts; "
                                                "conditions must be empty"
                                              : "this server makes video from text only; "
                                                "conditions must be empty"), req.peer);
    /* Reference parts without task ref2va are a mistake about the task. */
    if (!refs.empty() && !ref2va)
        return bad("task", "reference files are given: send them with task ref2va, or send "
                           "none");
    if (ref2va) {
        ApiError e;
        if (validate_refs(refs, info, &e) < 0) return error_response(e, req.peer);
    }
    /* At most one keyframe of each, not with reference media, and only with a video encoder. */
    {
        int n_first = 0, n_last = 0;
        for (const gen::VideoRefMedia& m : keys)
            if (m.kind == gen::VideoRefMedia::FIRST_FRAME) ++n_first; else ++n_last;
        if (n_first > 1) return bad("first_frame", "at most one first_frame part");
        if (n_last > 1) return bad("last_frame", "at most one last_frame part");
        /* A keyframe's <Picture N> counts the pictures present, as ComfyUI numbers them: a last
         * frame alone is <Picture 1>. */
        for (gen::VideoRefMedia& m : keys)
            m.index = m.kind == gen::VideoRefMedia::LAST_FRAME && n_first ? 2 : 1;
        if (!keys.empty()) {
            if (info.max_enc_side <= 0)
                return bad("first_frame", "this model has no video encoder, so it cannot pin a "
                                          "keyframe");
            if (!refs.empty())
                return bad("first_frame", "a keyframe completion conditions on its pinned frames, "
                                          "not on reference media: give first_frame/last_frame or "
                                          "the ref_* parts, not both");
        }
        /* A named keyframe task carries its frames. */
        if (!kf_task.empty()) {
            if ((kf_task == "i2va" || kf_task == "fl2va") && n_first < 1)
                return bad("first_frame", "task '" + kf_task + "' pins a first frame: give a "
                                          "first_frame part");
            if ((kf_task == "l2va" || kf_task == "fl2va") && n_last < 1)
                return bad("last_frame", "task '" + kf_task + "' pins a last frame: give a "
                                         "last_frame part");
        }
    }
    /* How the first frame is fitted to the video: stretched, as ComfyUI does, or `center`, scaled
     * to cover it and cropped. The last frame is always cropped, as ComfyUI does. */
    bool first_center = false;
    if (has(body, "first_frame_mode")) {
        const std::string fm = body["first_frame_mode"].is_string()
                             ? body["first_frame_mode"].get<std::string>() : "";
        if (fm.empty() || fm == "stretch")
            ;
        else if (fm == "center")
            first_center = true;
        else
            return bad("first_frame_mode", "expected \"stretch\" (the reference's own resize) "
                                           "or \"center\" (keep the anchor's aspect, crop the "
                                           "canvas)");
    }
    json target = json::object();
    if (has(body, "target")) {
        if (!body["target"].is_object()) return bad("target", "expected an object");
        target = body["target"];
        for (auto it = target.begin(); it != target.end(); ++it)
            if (it.key() != "short_edge" && it.key() != "aspect_ratio" &&
                it.key() != "duration_seconds")
                return bad("target", "'" + it.key() + "' is not a field of target");
    }

    /* ---- the size */
    int64_t W = info.default_width, H = info.default_height;
    const int64_t short_def = std::min(info.default_width, info.default_height);
    if (has(body, "size")) {
        if (!body["size"].is_string() || !parse_size(body["size"].get<std::string>(), &W, &H))
            return bad("size", "expected WIDTHxHEIGHT, e.g. " + std::to_string(info.default_width) +
                                   "x" + std::to_string(info.default_height));
        if (has(target, "aspect_ratio") || has(target, "short_edge"))
            return bad("size", "give size or target's aspect_ratio and short_edge, not both");
    } else if (has(target, "aspect_ratio") || has(target, "short_edge")) {
        double rw = (double)info.default_width, rh = (double)info.default_height;
        if (has(target, "aspect_ratio")) {
            if (!target["aspect_ratio"].is_string())
                return bad("target", "aspect_ratio is a string, w:h or auto");
            const std::string ar = target["aspect_ratio"].get<std::string>();
            if (ar != "auto" && !parse_ratio(ar, &rw, &rh))
                return bad("target", "aspect_ratio must be w:h (16:9) or auto");
        }
        int64_t se = short_def;
        if (has(target, "short_edge")) {
            double v = 0;
            if (!number(target["short_edge"], &v) || v < 1 || v != std::floor(v))
                return bad("target", "short_edge must be a positive whole number of pixels");
            se = (int64_t)v;
        }
        if (std::max(rw, rh) > (double)info.max_aspect * std::min(rw, rh))
            return bad("target", "the aspect ratio may be at most " +
                                     std::to_string(info.max_aspect) + ":1 either way");
        canvas_for(rw, rh, se, info.max_area, info.size_multiple, &W, &H);
    }
    if (W % info.size_multiple || H % info.size_multiple || W < info.min_side ||
        H < info.min_side || W > info.max_area || H > info.max_area || W * H > info.max_area ||
        std::max(W, H) > info.max_aspect * std::min(W, H)) {
        int64_t sw = 0, sh = 0;
        canvas_for((double)W, (double)H, short_def, info.max_area, info.size_multiple, &sw, &sh);
        return bad("size", std::to_string(W) + "x" + std::to_string(H) + " is not a size this "
                   "model makes: sides are multiples of " + std::to_string(info.size_multiple) +
                   " from " + std::to_string(info.min_side) + ", at most " +
                   std::to_string(info.max_area) + " pixels and " +
                   std::to_string(info.max_aspect) + ":1; its canvas at that ratio is " +
                   std::to_string(sw) + "x" + std::to_string(sh));
    }

    /* ---- the length, in whole chunks */
    const double fps = (double)info.fps_num / (double)info.fps_den;
    const double min_s = (double)(info.frames_per_chunk * info.min_chunks + info.frames_tail) / fps;
    const double max_s = (double)(info.frames_per_chunk * info.max_chunks + info.frames_tail) / fps;
    int64_t chunks = info.default_chunks;
    const bool has_sec = has(body, "seconds"), has_dur = has(target, "duration_seconds");
    if (has_sec && has_dur)
        return bad("seconds", "give seconds or target's duration_seconds, not both");
    /* A model with a length head chooses the length of a request that names none, or `auto`. */
    const bool auto_len = info.length_head &&
        (!(has_sec || has_dur) ||
         (has_sec && body["seconds"].is_string() && body["seconds"].get<std::string>() == "auto"));
    if (auto_len) {
        chunks = 0;
    } else if (has_sec || has_dur) {
        double sec = 0;
        if (!number(has_sec ? body["seconds"] : target["duration_seconds"], &sec) || !(sec > 0))
            return bad(has_sec ? "seconds" : "target", "the length is a positive number of seconds");
        const int64_t want = (int64_t)std::ceil(sec * fps - 1e-9);
        chunks = std::max<int64_t>(info.min_chunks,
                                   (want - info.frames_tail + info.frames_per_chunk - 1) /
                                       info.frames_per_chunk);
        if (chunks > info.max_chunks)
            return bad(has_sec ? "seconds" : "target",
                       "this model makes " + rad::fmt("%.2f", min_s) + " to " +
                           rad::fmt("%.2f", max_s) + " seconds");
    }
    /* The length the job reports until the model has chosen one. */
    const int64_t frames =
        info.frames_per_chunk * (chunks > 0 ? chunks : info.default_chunks) + info.frames_tail;

    std::unique_ptr<gen::VideoRequest> r = std::make_unique<gen::VideoRequest>();
    r->width = (int32_t)W;
    r->height = (int32_t)H;
    r->chunks = (int32_t)chunks;
    /* ---- the decoder, by name, the model's first when absent */
    std::string decoder;
    if (has(body, "decoder")) {
        if (!body["decoder"].is_string())
            return bad("decoder", "the decoder is a name");
        decoder = body["decoder"].get<std::string>();
        std::string names;
        int32_t idx = -1;
        for (size_t i = 0; i < info.decoders.size(); ++i) {
            names += (i ? ", " : "") + info.decoders[i];
            if (info.decoders[i] == decoder) idx = (int32_t)i;
        }
        if (idx < 0)
            return bad("decoder", info.decoders.empty()
                                      ? std::string("this model has one decoder and takes no name")
                                      : "this model decodes with " + names);
        if (W * H > info.decoder_max_area[(size_t)idx])
            return bad("decoder", "the " + decoder + " decoder makes frames of " +
                                      std::to_string(info.decoder_max_area[(size_t)idx]) +
                                      " pixels at most");
        r->decoder = idx;
        if (idx == 0) decoder.clear();
    }
    if (has(body, "num_inference_steps")) {
        double v = 0;
        const char* unit = info.step_evals ? " model evaluations"
                                           : " sigma grid points, the last one 0";
        const int64_t lo = info.step_evals ? 1 : 2;
        if (info.fixed_steps && (!number(body["num_inference_steps"], &v) ||
                                 v != (double)info.max_steps))
            return bad("num_inference_steps",
                       "this model runs the schedule it was trained on: " +
                           std::to_string(info.max_steps) + unit);
        if (!number(body["num_inference_steps"], &v) || v != std::floor(v) || v < lo ||
            v > (double)info.max_steps)
            return bad("num_inference_steps", "expected an integer in " + std::to_string(lo) +
                                                  ".." + std::to_string(info.max_steps) + " (" +
                                                  (unit + 1) + ")");
        r->steps = (int32_t)v;
    }
    if (has(body, "sampler")) {
        const std::string sm = body["sampler"].is_string() ? body["sampler"].get<std::string>() : "";
        if (sm.empty() || sm == "euler") r->sampler = RAD_SAMPLER_EULER;
        else if (sm == "res_multistep") r->sampler = RAD_SAMPLER_RES_MULTISTEP;
        else {
            ApiError e = err_unsupported("sampler", "the sampler is euler (first order) or "
                                                   "res_multistep (second order on the clean "
                                                   "estimate)");
            e.supported = { "euler", "res_multistep" };
            return error_response(e, req.peer);
        }
    }
    /* Each stream's schedule shift; any positive shift is safe, and (0, 100] refuses typos. */
    if (has(body, "shift")) {
        double v = 0;
        if (!number(body["shift"], &v) || !(v > 0.0) || v > 100.0)
            return bad("shift", "the video schedule shift: a number in (0, 100], or omit it to run "
                                "the model's trained shift " + rad::fmt("%g", info.shift_video));
        r->shift_video = v;
    }
    if (has(body, "audio_shift")) {
        double v = 0;
        if (!number(body["audio_shift"], &v) || !(v > 0.0) || v > 100.0)
            return bad("audio_shift", "the soundtrack schedule shift: a number in (0, 100], or "
                                      "omit it to run the model's trained shift " +
                                      rad::fmt("%g", info.shift_audio));
        r->shift_audio = v;
    }
    /* ---- guidance */
    static const char* kGuide[8] = {
        "guidance_scale", "stg_scale", "modality_scale", "guidance_rescale",
        "audio_guidance_scale", "audio_stg_scale", "audio_modality_scale", "audio_guidance_rescale",
    };
    for (int i = 0; i < 8; ++i) {
        if (!has(body, kGuide[i])) continue;
        if (!info.guidance)
            return error_response(err_unsupported(kGuide[i], "this model takes no guidance"),
                                  req.peer);
        double v = 0;
        const bool rescale = i % 4 == 3;
        if (!number(body[kGuide[i]], &v) || !(v >= 0.0) || v > (rescale ? 1.0 : 100.0))
            return bad(kGuide[i], rescale ? "expected a number from 0 to 1"
                                          : "expected a number from 0 to 100");
        r->guidance[i] = v;
    }
    std::string negative = info.default_negative;
    if (has(body, "negative_prompt")) {
        if (!info.guidance)
            return error_response(err_unsupported("negative_prompt", "this model takes no "
                                                  "guidance"), req.peer);
        if (!body["negative_prompt"].is_string())
            return bad("negative_prompt", "expected a string");
        negative = body["negative_prompt"].get<std::string>();
        r->has_neg = true;
    }
    if (has(body, "latent_upscale")) {
        double v = 0;
        if (!(info.upscale_max > 0.0))
            return error_response(err_unsupported("latent_upscale", "this model has no latent "
                                                  "upscaler"), req.peer);
        if (!number(body["latent_upscale"], &v) || !(v >= 1.0) || v > info.upscale_max)
            return bad("latent_upscale", "expected a factor from 1 to " +
                                             rad::fmt("%g", info.upscale_max));
        r->upscale = v > 1.0 ? v : 0.0;
    }
    if (has(body, "upscale_refine_steps")) {
        double v = 0;
        if (r->upscale <= 1.0)
            return bad("upscale_refine_steps", "counts the steps after a latent upscale; give "
                                               "latent_upscale above 1 with it");
        if (!number(body["upscale_refine_steps"], &v) || v != std::floor(v) || v < 1 || v > 1e6)
            return bad("upscale_refine_steps", "expected a positive integer");
        r->refine = (int32_t)v;
    }
    bool seed_given = false;
    if (has(body, "seed")) {
        double v = 0;
        if (!number(body["seed"], &v) || v != std::floor(v)) return bad("seed", "expected an integer");
        r->seed = (uint64_t)(int64_t)v;
        seed_given = v != -1;
    }
    if (!seed_given) {
        /* 53 bits, as the other generators choose one: JavaScript reads a number as a double. */
        std::random_device rd;
        r->seed = (((uint64_t)rd() << 32) ^ rd()) & (((uint64_t)1 << 53) - 1);
    }
    r->refs = std::move(refs);
    /* The keyframes after the references, the first before the last. */
    std::stable_sort(keys.begin(), keys.end(), [](const gen::VideoRefMedia& a,
                                                  const gen::VideoRefMedia& b) {
        return a.kind < b.kind;
    });
    r->refs.insert(r->refs.end(), keys.begin(), keys.end());
    const bool has_keys = !keys.empty();

    /* Pixels as the VAE encoders take them: RGB in 0..255, frame after frame. */
    auto pixels = [](gen::VideoRefMedia& m, const std::vector<mm::Frame>& fr) {
        m.enc_frames = (int32_t)fr.size();
        m.enc_h = fr.front().h;
        m.enc_w = fr.front().w;
        m.enc_pixels.clear();
        for (const mm::Frame& f : fr) m.enc_pixels.insert(m.enc_pixels.end(), f.rgb.begin(),
                                                          f.rgb.end());
    };

    /* A keyframe is decoded once, onto the output canvas, and that frame serves both the VAE pin
     * and the vision tower, as in ComfyUI: the tower sees the pixels that are pinned, not the file
     * at its own size. The first frame is stretched unless first_frame_mode=center, the last is
     * centre-cropped. `canvas` parallels r->refs. */
    std::vector<mm::Frame> canvas;
    if (has_keys && deps_.mm) {
        canvas.resize(r->refs.size());
        for (size_t i = 0; i < r->refs.size(); ++i) {
            gen::VideoRefMedia& m = r->refs[i];
            if (m.kind != gen::VideoRefMedia::FIRST_FRAME &&
                m.kind != gen::VideoRefMedia::LAST_FRAME)
                continue;
            std::string why;
            if (deps_.mm->encoder_image_canvas((const uint8_t*)m.data.data(), m.data.size(),
                                               (int32_t)W, (int32_t)H,
                                               m.kind == gen::VideoRefMedia::LAST_FRAME ||
                                                   first_center,
                                               &canvas[i], &why) < 0)
                return bad(m.name.c_str(), "'" + m.name + "' could not be decoded: " + why);
            pixels(m, { canvas[i] });
            /* A latent upscale's first steps pin it at their own size, prepared the same way. */
            if (r->upscale > 1.0) {
                RadVideoGenDecl geo{};   /* video_small_size reads the size multiple only */
                geo.size_multiple = info.size_multiple;
                int32_t sw = 0, sh = 0;
                gen::video_small_size(geo, r->width, r->height, r->upscale, &sw, &sh);
                mm::Frame small;
                if (deps_.mm->encoder_image_canvas((const uint8_t*)m.data.data(), m.data.size(),
                                                   sw, sh,
                                                   m.kind == gen::VideoRefMedia::LAST_FRAME ||
                                                       first_center,
                                                   &small, &why) < 0)
                    return bad(m.name.c_str(), "'" + m.name + "' could not be decoded: " + why);
                m.small_h = small.h;
                m.small_w = small.w;
                m.small_pixels.assign(small.rgb.begin(), small.rgb.end());
            }
        }
    }

    /* ---- the prompt: the template's text with special tokens, the media in ComfyUI's order, the
     * prompt without. A picture is its `<Picture N>: ` label, encoded on its own, then its
     * tokens; a video its soundtrack's `<Audio N>: ` label if it has one, then its `<Video N>: `;
     * a soundtrack its `<Audio N>: `, numbered with the videos' soundtracks. Only the pictures are
     * shown to the text encoder: videos and soundtracks reach the model through the VAEs. */
    {
        static const std::string kSlot = "{prompt}";
        const std::string& t = info.tmpl;
        const size_t p = t.find(kSlot);
        auto put = [&](std::vector<int32_t>* out, const std::string& s, bool special) {
            if (s.empty()) return;
            const auto ids = deps_.tok->encode(s, false, special);
            out->insert(out->end(), ids.begin(), ids.end());
        };
        auto render = [&](const std::string& text, std::vector<int32_t>* out) {
            put(out, t.substr(0, p), true);
            put(out, text, false);
            put(out, t.substr(p + kSlot.size()), true);
        };
        if ((ref2va || has_keys) && deps_.mm && deps_.mm->accepts(MediaKind::Image)) {
            /* Every picture is processed before any token is laid down. ComfyUI also shows the text
             * encoder a video's frames at 2 fps, which this does not. */
            std::vector<std::shared_ptr<mm::Item>> pics;
            MediaOpts mo;
            mo.max_pixels = (int64_t)W * (int64_t)H;   /* scaled to the video's area */
            /* A keyframe is shown to the tower as well as pinned, as its canvas frame. */
            auto anchor = [](const gen::VideoRefMedia& m) {
                return m.kind == gen::VideoRefMedia::IMAGE ||
                       m.kind == gen::VideoRefMedia::FIRST_FRAME ||
                       m.kind == gen::VideoRefMedia::LAST_FRAME;
            };
            auto keyframe = [](const gen::VideoRefMedia& m) {
                return m.kind == gen::VideoRefMedia::FIRST_FRAME ||
                       m.kind == gen::VideoRefMedia::LAST_FRAME;
            };
            for (size_t i = 0; i < r->refs.size(); ++i) {
                const gen::VideoRefMedia& m = r->refs[i];
                if (!anchor(m)) continue;
                std::shared_ptr<mm::Item> it;
                std::string why;
                int rc;
                if (keyframe(m) && !canvas.empty() && canvas[i].w > 0)
                    rc = deps_.mm->prepare_frame(canvas[i], &it, &why);
                else {
                    const std::vector<uint8_t> bytes(m.data.begin(), m.data.end());
                    rc = deps_.mm->prepare(MediaKind::Image, bytes, mo, &it, &why);
                }
                if (rc < 0 || !it)
                    return bad(m.name.c_str(), "'" + m.name + "' could not be used: " + why);
                pics.push_back(std::move(it));
            }
            auto has_soundtrack = [&](int32_t idx) {
                for (const gen::VideoRefMedia& m : r->refs)
                    if (m.kind == gen::VideoRefMedia::VIDEO_AUDIO && m.index == idx) return true;
                return false;
            };
            put(&r->tokens, t.substr(0, p), true);
            std::shared_ptr<mm::PromptMedia> media;
            if (!pics.empty()) media = std::make_shared<mm::PromptMedia>();
            size_t i = 0;
            for (const gen::VideoRefMedia& m : r->refs) {
                if (!anchor(m)) continue;
                put(&r->tokens, "<Picture " + std::to_string(m.index) + ">: ", false);
                mm::Placed pl;
                pl.item = pics[i++];
                pl.tok = (int32_t)r->tokens.size();
                r->tokens.insert(r->tokens.end(), pl.item->expand.begin(),
                                 pl.item->expand.end());
                media->items.push_back(std::move(pl));
            }
            int audio_ord = 0;
            for (const gen::VideoRefMedia& m : r->refs) {
                if (m.kind != gen::VideoRefMedia::VIDEO) continue;
                if (has_soundtrack(m.index))
                    put(&r->tokens, "<Audio " + std::to_string(++audio_ord) + ">: ", false);
                put(&r->tokens, "<Video " + std::to_string(m.index) + ">: ", false);
            }
            for (const gen::VideoRefMedia& m : r->refs) {
                if (m.kind != gen::VideoRefMedia::AUDIO) continue;
                put(&r->tokens, "<Audio " + std::to_string(++audio_ord) + ">: ", false);
            }
            if (media) { media->layout(); r->media = std::move(media); }
            put(&r->tokens, prompt, false);
            put(&r->tokens, t.substr(p + kSlot.size()), true);
        } else {
            render(prompt, &r->tokens);
        }
        if ((int64_t)r->tokens.size() > info.max_text)
            return bad("prompt", "the prompt" + std::string((ref2va || has_keys)
                                                                ? " with its media" : "") +
                                     " is " + std::to_string(r->tokens.size()) +
                                     " tokens; this model takes at most " +
                                     std::to_string(info.max_text));
        if (info.guidance) {
            render(negative, &r->neg_tokens);
            if ((int64_t)r->neg_tokens.size() > info.max_text)
                return bad("negative_prompt", "the negative prompt is " +
                                                  std::to_string(r->neg_tokens.size()) +
                                                  " tokens; this model takes at most " +
                                                  std::to_string(info.max_text));
        }
    }

    /* ---- the first frame, at the size the first steps run at */
    if (has_ref) {
        int32_t fw = r->width, fh = r->height;
        if (r->upscale > 1.0) {
            RadVideoGenDecl geo{};   /* video_small_size reads the size multiple only */
            geo.size_multiple = info.size_multiple;
            gen::video_small_size(geo, r->width, r->height, r->upscale, &fw, &fh);
        }
        mm::Frame f;
        std::string why;
        if (mm::decode_image(ref_bytes.data(), ref_bytes.size(), kMaxRefPixels, &f, &why) < 0)
            return bad("input_reference", "the picture could not be read: " + why);
        if (info.image_crf > 0) {
            mm::Frame rt;
            const int e = mm::h264_roundtrip(f, info.image_crf, &rt, &why);
            if (e < 0) {
                ApiError err = err_bad("input_reference", "the picture could not be prepared: " + why);
                /* A build without an H.264 codec cannot prepare one at all: 501, not a transient 503. */
                if (e != RAD_E_INVAL) { err.status = e == RAD_E_UNSUPPORTED ? 501 : 503; err.type = "server_error"; }
                return error_response(err, req.peer);
            }
            f = std::move(rt);
        }
        mm::FrameRGBA a, sized;
        a.w = f.w;
        a.h = f.h;
        a.rgba.resize((size_t)f.w * f.h * 4);
        for (size_t i = 0, n = (size_t)f.w * f.h; i < n; ++i) {
            std::memcpy(&a.rgba[4 * i], &f.rgb[3 * i], 3);
            a.rgba[4 * i + 3] = 255;
        }
        mm::resize_bilinear_rgba(a, fw, fh, &sized);
        r->image.resize((size_t)fw * fh * 3);
        for (size_t i = 0, n = (size_t)fw * fh; i < n; ++i)
            std::memcpy(&r->image[3 * i], &sized.rgba[4 * i], 3);
    }

    /* ---- every reference as its VAE encoder takes it (the keyframes are done above): pictures and
     * video frames fitted to the encoder's bounds, soundtracks as mono samples at the model's rate,
     * one copy a channel. The driver pads both to the encoders' clips and hops. */
    if ((ref2va || !keys.empty()) && deps_.mm) {
        for (gen::VideoRefMedia& m : r->refs) {
            std::string why;
            const uint8_t* p = (const uint8_t*)m.data.data();
            const size_t   n = m.data.size();
            std::vector<mm::Frame> fr(1);
            int st = 0;
            if (m.kind == gen::VideoRefMedia::FIRST_FRAME || m.kind == gen::VideoRefMedia::LAST_FRAME)
                continue;
            if (m.kind == gen::VideoRefMedia::IMAGE)
                st = deps_.mm->encoder_image(p, n, (int32_t)info.enc_ratio,
                                             (int32_t)info.max_enc_side, &fr[0], &why);
            else if (m.kind == gen::VideoRefMedia::VIDEO)
                st = deps_.mm->encoder_video(p, n, (int32_t)info.enc_ratio,
                                             (int32_t)info.max_enc_side,
                                             (int32_t)info.max_enc_frames, &fr, &why);
            else {
                std::vector<float> mono;
                if (deps_.mm->encoder_audio(p, n, (int32_t)info.sample_rate, info.max_enc_samples,
                                            &mono, &why) < 0 || mono.empty())
                    return bad(m.name.c_str(), "'" + m.name + "' could not be decoded: " + why);
                m.enc_channels = (int32_t)info.channels;
                m.enc_n = (int64_t)mono.size();
                m.enc_samples.clear();
                for (int32_t c = 0; c < m.enc_channels; ++c)
                    m.enc_samples.insert(m.enc_samples.end(), mono.begin(), mono.end());
                continue;
            }
            if (st < 0 || fr.empty())
                return bad(m.name.c_str(), "'" + m.name + "' could not be decoded: " + why);
            pixels(m, fr);
        }
    }

    /* ---- admitted and queued: the job holds its place until its file is written */
    if (!loras.empty()) {
        std::string why, note;
        const int s = opt_.gen_lora_acquire(loras, &r->lora_slot, &r->lora_hold, &note, &why);
        if (s < 0) {
            ApiError e = err_bad("loras", why);
            if (s == RAD_E_NOTFOUND) { e.status = 404; e.type = "not_found_error"; }
            else if (s != RAD_E_INVAL) { e.status = 503; e.type = "server_error"; }
            return error_response(e, req.peer);
        }
    }
    if (auto no = admit(1, req.peer)) return *no;
    auto j = std::make_shared<VideoJob>();
    j->id = new_video_id();
    j->model = opt_.model_id;
    j->prompt = prompt;
    j->seconds = auto_len ? std::string("auto") : rad::fmt("%.2f", (double)frames / fps);
    j->size = std::to_string(W) + "x" + std::to_string(H);
    j->created_at = unix_now();
    j->seed = r->seed;
    j->sampler = r->sampler;
    j->shift_video = r->shift_video;
    j->shift_audio = r->shift_audio;
    for (const gen::VideoRefMedia& m : r->refs) j->refs.push_back(m.name);
    if (ref2va) j->task = "ref2va";
    else if (keys.size() == 2) j->task = "fl2va";
    else if (!keys.empty())
        j->task = keys[0].kind == gen::VideoRefMedia::FIRST_FRAME ? "i2va" : "l2va";
    j->ticket = std::make_shared<gen::VideoTicket>();
    r->ticket = j->ticket;
    const int64_t n_tok = (int64_t)r->tokens.size();
    const int32_t steps = r->steps;
    j->guided = info.guidance;
    for (int i = 0; i < 8; ++i)
        j->guidance[i] = std::isnan(r->guidance[i]) ? info.guidance_default[i] : r->guidance[i];
    j->has_negative = info.guidance && (j->guidance[0] != 1.0 || j->guidance[4] != 1.0);
    if (j->has_negative) j->negative = negative;
    j->from_image = !r->image.empty();
    j->decoder = decoder;
    const double upscale = r->upscale;
    {
        std::string w;
        const int s = deps_.videos->submit(std::move(r), &w);
        if (s < 0) {
            release(1);
            ApiError e = err_bad("", w.empty() ? std::string(rad_strerror(s)) : w);
            if (s != RAD_E_INVAL) { e.status = 503; e.type = "server_error"; }
            return error_response(e, req.peer);
        }
    }
    const gen::VideoTicket& tk = *j->ticket;
    if (tk.refine > 0) {
        j->upscale = upscale;
        j->refine = tk.refine;
        j->upscale_from = std::to_string(tk.small_width) + "x" + std::to_string(tk.small_height);
    }
    RAD_INFO("video %s: %s, %s, %lld prompt tokens, %d steps%s%s, seed=%llu",
             j->id.c_str(), j->size.c_str(),
             auto_len ? "the length the model chooses"
                      : rad::fmt("%lld frames (%s s)", (long long)frames, j->seconds.c_str()).c_str(),
             (long long)n_tok, steps > 0 ? steps : (int)tk.steps,
             tk.refine > 0 ? rad::fmt(" (the first %d at %dx%d, upscaled for the last %d)",
                                      (int)(tk.steps - tk.refine), (int)tk.small_width,
                                      (int)tk.small_height, (int)tk.refine).c_str()
                           : "",
             j->refs.empty() ? "" : rad::fmt(", %zu reference parts", j->refs.size()).c_str(),
             (unsigned long long)j->seed);
    std::lock_guard<std::mutex> lk(videos_m_);
    sweep_videos_locked(unix_now());
    videos_.push_back(j);
    if (deps_.background) deps_.background->fetch_add(1);
    j->worker = std::thread([this, j] {
        video_work(j);
        if (deps_.background) deps_.background->fetch_sub(1);
    });
    return HttpResponse::json_body(200, json_dump(video_json(*j)));
}

/* The job's thread. Progress is the steps' share of 90, then the frames' of the last 10. */
void Server::video_work(std::shared_ptr<VideoJob> j) {
    gen::VideoTicket& t = *j->ticket;
    const VideoGenInfo& info = deps_.videos->info();
    mm::Mp4Writer mp4;
    std::string why;
    int st = mp4.open(t.width, t.height, info.fps_num, info.fps_den, t.sample_rate, t.channels,
                      &why);
    int32_t frames = 0;
    bool aborted = false;
    while (st >= 0) {
        gen::VideoEvent e;
        bool to = false;
        if (!t.next(&e, 500, &to)) {
            if (!to) break;
            if (stopping_.load()) {
                t.cancel.store(true);
                aborted = true;
                break;
            }
            continue;
        }
        if (e.kind == gen::VideoEvent::STEP) {
            const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            std::lock_guard<std::mutex> lk(videos_m_);
            j->status = "in_progress";
            j->progress = t.steps > 0 ? (int)(90LL * e.steps_done / t.steps) : 0;
            j->steps_done = e.steps_done;
            j->steps_total = t.steps;
            if (!j->step_ns0) j->step_ns0 = now;
            /* A line a step, its rate the mean since the first step, so the eta of an upscaled
             * video, whose small steps are faster, is an estimate. */
            const double el = (double)(now - j->step_ns0) / 1e9;
            const double per = e.steps_done > 1 ? el / (e.steps_done - 1) : 0.0;
            if (per > 0)
                RAD_INFO("video %s: step %d/%d, %.2f s/step, eta %.0f s",
                         j->id.c_str(), e.steps_done, t.steps, per,
                         per * (double)(t.steps - e.steps_done));
            else
                RAD_INFO("video %s: step %d/%d", j->id.c_str(), e.steps_done, t.steps);
        } else if (e.kind == gen::VideoEvent::LENGTH) {
            std::lock_guard<std::mutex> lk(videos_m_);
            j->seconds = rad::fmt("%.2f", (double)e.n * info.fps_den / info.fps_num);
        } else if (e.kind == gen::VideoEvent::AUDIO) {
            if (t.channels > 0)
                st = mp4.add_audio(e.samples.data(), (int64_t)e.samples.size() / t.channels, &why);
        } else if (e.kind == gen::VideoEvent::FRAMES) {
            if (e.first == 0 && e.n > 0) {
                std::lock_guard<std::mutex> lk(videos_m_);
                j->first.assign(e.rgb.begin(), e.rgb.begin() + (size_t)t.width * t.height * 3);
                j->w = t.width;
                j->h = t.height;
            }
            st = mp4.add_frames(e.rgb.data(), e.n, &why);
            frames += e.n;
            std::lock_guard<std::mutex> lk(videos_m_);
            j->progress = 90 + (t.frames > 0 ? (int)(10LL * frames / t.frames) : 0);
        }
    }
    if (st < 0) t.cancel.store(true);
    std::string file;
    if (st >= 0 && !aborted && t.status() >= 0 && frames == t.frames)
        st = mp4.finish(&file, &why);
    release(1);

    std::lock_guard<std::mutex> lk(videos_m_);
    const int64_t now = unix_now();
    if (st >= 0 && !aborted && t.status() >= 0 && frames == t.frames) {
        j->status = "completed";
        j->progress = 100;
        j->completed_at = now;
        j->expires_at = now + opt_.video_ttl_s;
        j->mp4 = std::move(file);
        auto sec = [](int64_t a, int64_t b) { return (double)(b - a) / 1e9; };
        const std::string steps = t.t_upscaled_ns > 0
            ? rad::fmt("small steps %.1f s, upscale %.2f s, full steps %.1f s",
                       sec(t.t_cond_ns, t.t_small_ns), sec(t.t_small_ns, t.t_upscaled_ns),
                       sec(t.t_upscaled_ns, t.t_denoised_ns))
            : rad::fmt("steps %.1f s", sec(t.t_cond_ns, t.t_denoised_ns));
        RAD_INFO("video %s: completed, %.1f MB (%s), %.1f s: queue and prompt %.1f s, %s, "
                 "decode %.1f s", j->id.c_str(), (double)j->mp4.size() / 1e6,
                 mp4.video_encoder().c_str(), sec(t.t_submit_ns, t.t_done_ns),
                 sec(t.t_submit_ns, t.t_cond_ns), steps.c_str(), sec(t.t_denoised_ns, t.t_done_ns));
    } else {
        j->status = "failed";
        j->failed = true;
        j->completed_at = now;
        if (st < 0) {
            j->err_code = "encoding_failed";
            j->err_message = "the video could not be written: " + why;
        } else if (aborted || t.status() == RAD_E_CANCELLED) {
            j->err_code = "cancelled";
            j->err_message = aborted ? "this model is stopping" : "the video was cancelled";
        } else {
            j->err_code = t.status() == RAD_E_INVAL ? "invalid_request" : "generation_failed";
            j->err_message = t.why().empty() ? std::string(rad_strerror(t.status())) : t.why();
        }
        RAD_WARN("video %s: failed: %s", j->id.c_str(), j->err_message.c_str());
    }
    j->worker_done = true;
    /* A finished job no longer changes, so it can be read without this server, which may be gone
     * by the time its client comes for it. */
    if (deps_.front && !j->deleted.load())
        deps_.front->keep_job(j->id, j->expires_at ? j->expires_at : now + opt_.video_ttl_s,
                              opt_.max_videos, [j](const HttpRequest& q) -> HttpResponse {
            if (q.method == "DELETE") {
                j->deleted.store(true);
                return video_deleted(j->id);
            }
            if (q.path.size() > 8 && q.path.compare(q.path.size() - 8, 8, "/content") == 0) {
                HttpResponse res;
                ApiError e;
                if (video_content(*j, q, &res, &e)) return res;
                return HttpResponse::json_body(e.status, error_body(q.path, e.status, e.type.c_str(), e.message));
            }
            return HttpResponse::json_body(200, json_dump(video_json(*j)));
        });
}

void Server::sweep_videos_locked(int64_t now) {
    /* Deleted jobs whose threads are done are joined. */
    for (size_t i = 0; i < videos_gone_.size();) {
        if (!videos_gone_[i]->worker_done) { ++i; continue; }
        if (videos_gone_[i]->worker.joinable()) videos_gone_[i]->worker.join();
        videos_gone_.erase(videos_gone_.begin() + (std::ptrdiff_t)i);
    }
    /* Expired files go first, then the oldest finished jobs past the count. A running job is
     * never dropped here: it holds an admission place and its thread. */
    auto drop = [&](size_t i) {
        if (videos_[i]->worker.joinable()) videos_[i]->worker.join();
        videos_.erase(videos_.begin() + (std::ptrdiff_t)i);
    };
    for (size_t i = 0; i < videos_.size();) {
        const VideoJob& j = *videos_[i];
        const bool done = j.status == "completed" || j.status == "failed";
        if (done && j.deleted.load()) drop(i);
        else if (done && j.expires_at > 0 && j.expires_at <= now) drop(i);
        else if (done && j.failed && j.completed_at + opt_.video_ttl_s <= now) drop(i);
        else ++i;
    }
    int64_t finished = 0;
    for (const auto& j : videos_) finished += j->status == "completed" || j->status == "failed";
    for (size_t i = 0; i < videos_.size() && finished > opt_.max_videos;) {
        const VideoJob& j = *videos_[i];
        if (j.status == "completed" || j.status == "failed") {
            drop(i);
            --finished;
        } else {
            ++i;
        }
    }
}

void Server::stop_videos() {
    std::vector<std::shared_ptr<VideoJob>> all;
    {
        std::lock_guard<std::mutex> lk(videos_m_);
        all = videos_;
        all.insert(all.end(), videos_gone_.begin(), videos_gone_.end());
    }
    for (auto& j : all) j->ticket->cancel.store(true);
    for (auto& j : all)
        if (j->worker.joinable()) j->worker.join();
}

/* The video object, OpenAI's shape. */
json Server::video_json(const VideoJob& j) {
    json o;
    o["id"] = j.id;
    o["object"] = "video";
    o["model"] = j.model;
    o["status"] = j.status;
    o["progress"] = j.progress;
    if (j.steps_total > 0) {
        o["steps_done"] = j.steps_done;
        o["steps_total"] = j.steps_total;
    }
    o["created_at"] = j.created_at;
    o["completed_at"] = j.completed_at ? json(j.completed_at) : json(nullptr);
    o["expires_at"] = j.expires_at ? json(j.expires_at) : json(nullptr);
    o["prompt"] = j.prompt;
    o["seconds"] = j.seconds;
    o["size"] = j.size;
    o["seed"] = j.seed;
    o["sampler"] = j.sampler == RAD_SAMPLER_RES_MULTISTEP ? "res_multistep" : "euler";
    if (j.shift_video > 0.0) o["shift"] = j.shift_video;
    if (j.shift_audio > 0.0) o["audio_shift"] = j.shift_audio;
    if (j.refine > 0) {
        o["latent_upscale"] = j.upscale;
        o["upscale_refine_steps"] = j.refine;
        o["upscale_from"] = j.upscale_from;
    }
    if (!j.refs.empty()) {
        o["task"] = j.task;
        o["refs"] = j.refs;
    }
    if (j.guided) {
        static const char* kGuide[8] = {
            "guidance_scale", "stg_scale", "modality_scale", "guidance_rescale",
            "audio_guidance_scale", "audio_stg_scale", "audio_modality_scale",
            "audio_guidance_rescale",
        };
        for (int i = 0; i < 8; ++i) o[kGuide[i]] = j.guidance[i];
        if (j.has_negative) o["negative_prompt"] = j.negative;
    }
    if (j.from_image) o["input_reference"] = true;
    if (!j.decoder.empty()) o["decoder"] = j.decoder;
    o["remixed_from_video_id"] = nullptr;
    if (j.failed) {
        json e;
        e["code"] = j.err_code;
        e["message"] = j.err_message;
        o["error"] = e;
    } else {
        o["error"] = nullptr;
    }
    return o;
}

HttpResponse Server::handle_video_get(const HttpRequest& req) {
    if (!deps_.videos)
        return error_response(err_not_implemented("/v1/videos requires a video generation model"),
                              req.peer);
    const std::string id = req.param("id");
    std::lock_guard<std::mutex> lk(videos_m_);
    for (const auto& j : videos_)
        if (j->id == id && !j->deleted.load())
            return HttpResponse::json_body(200, json_dump(video_json(*j)));
    ApiError e = err_bad("id", "no video '" + id + "'; it may have expired");
    e.status = 404;
    e.type = "not_found_error";
    return error_response(e, req.peer);
}

HttpResponse Server::handle_video_list(const HttpRequest& req) {
    if (!deps_.videos)
        return error_response(err_not_implemented("/v1/videos requires a video generation model"),
                              req.peer);
    int64_t limit = 20;
    if (auto it = req.query.find("limit"); it != req.query.end()) {
        char* e = nullptr;
        limit = std::strtoll(it->second.c_str(), &e, 10);
        if (*e || limit < 1 || limit > 100)
            return error_response(err_bad("limit", "expected an integer in 1..100"), req.peer);
    }
    const auto ord = req.query.find("order");
    const bool asc = ord != req.query.end() && ord->second == "asc";
    if (ord != req.query.end() && ord->second != "asc" && ord->second != "desc")
        return error_response(err_bad("order", "expected asc or desc"), req.peer);
    const auto after = req.query.find("after");
    std::lock_guard<std::mutex> lk(videos_m_);
    sweep_videos_locked(unix_now());
    std::vector<std::shared_ptr<VideoJob>> v = videos_;
    if (!asc) std::reverse(v.begin(), v.end());
    size_t at = 0;
    if (after != req.query.end()) {
        while (at < v.size() && v[at]->id != after->second) ++at;
        if (at == v.size())
            return error_response(err_bad("after", "no video '" + after->second + "'"), req.peer);
        ++at;
    }
    json data = json::array();
    for (size_t i = at; i < v.size() && (int64_t)data.size() < limit; ++i)
        data.push_back(video_json(*v[i]));
    json o;
    o["object"] = "list";
    o["data"] = data;
    o["first_id"] = data.empty() ? json(nullptr) : data.front()["id"];
    o["last_id"] = data.empty() ? json(nullptr) : data.back()["id"];
    o["has_more"] = at + data.size() < v.size();
    return HttpResponse::json_body(200, json_dump(o));
}

HttpResponse Server::handle_video_delete(const HttpRequest& req) {
    if (!deps_.videos)
        return error_response(err_not_implemented("/v1/videos requires a video generation model"),
                              req.peer);
    const std::string id = req.param("id");
    std::lock_guard<std::mutex> lk(videos_m_);
    std::shared_ptr<VideoJob> j;
    for (size_t i = 0; i < videos_.size(); ++i)
        if (videos_[i]->id == id) {
            j = videos_[i];
            videos_.erase(videos_.begin() + (std::ptrdiff_t)i);
            break;
        }
    if (!j) {
        ApiError e = err_bad("id", "no video '" + id + "'");
        e.status = 404;
        e.type = "not_found_error";
        return error_response(e, req.peer);
    }
    /* A running job is cancelled, and its thread ends once the driver has seen that (at its next
     * pass, which may be seconds away); it is joined then. */
    j->ticket->cancel.store(true);
    j->deleted.store(true);
    videos_gone_.push_back(j);
    sweep_videos_locked(unix_now());
    return video_deleted(id);
}

HttpResponse Server::video_deleted(const std::string& id) {
    json o;
    o["id"] = id;
    o["object"] = "video.deleted";
    o["deleted"] = true;
    return HttpResponse::json_body(200, json_dump(o));
}

bool Server::video_content(const VideoJob& j, const HttpRequest& req, HttpResponse* res, ApiError* err) {
    std::string variant = "video";
    if (auto it = req.query.find("variant"); it != req.query.end()) variant = it->second;
    if (variant != "video" && variant != "thumbnail") {
        *err = err_unsupported("variant", "this server gives the video or its thumbnail");
        err->supported = { "video", "thumbnail" };
        return false;
    }
    if (variant == "thumbnail") {
        if (j.first.empty()) {
            *err = err_bad("id", "video '" + j.id + "' has no frames yet");
            return false;
        }
        std::vector<uint8_t> file;
        std::string why;
        if (mm::encode_image(j.first.data(), j.w, j.h, 3, "webp", 90, &file, &why) < 0) {
            *err = err_internal("the thumbnail could not be encoded: " + why);
            return false;
        }
        res->content_type = "image/webp";
        res->body.assign((const char*)file.data(), file.size());
        return true;
    }
    if (j.status != "completed") {
        *err = err_bad("id", "video '" + j.id + "' is " + j.status +
                                 (j.failed ? ": " + j.err_message : ""));
        return false;
    }
    res->content_type = "video/mp4";
    res->body = j.mp4;
    return true;
}

HttpResponse Server::handle_video_content(const HttpRequest& req) {
    if (!deps_.videos)
        return error_response(err_not_implemented("/v1/videos requires a video generation model"),
                              req.peer);
    const std::string id = req.param("id");
    std::lock_guard<std::mutex> lk(videos_m_);
    for (const auto& j : videos_)
        if (j->id == id && !j->deleted.load()) {
            HttpResponse res;
            ApiError e;
            return video_content(*j, req, &res, &e) ? res : error_response(e, req.peer);
        }
    ApiError e = err_bad("id", "no video '" + id + "'; it may have expired");
    e.status = 404;
    e.type = "not_found_error";
    return error_response(e, req.peer);
}

}  /* namespace server */
}  /* namespace rad */
