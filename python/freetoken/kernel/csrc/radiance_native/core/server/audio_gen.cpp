/* audio_gen.cpp -- POST /v1/audio/speech over an audio generator (core/gen/audio_driver.h): music
 * from a description and lyrics, or speech from text and a description of the voice.
 *
 * The request is OpenAI's speech request: `input` is the text the audio carries (a song's lyrics,
 * as SGLang-Omni reads it for a music model, or the words to speak) and `instructions` describes
 * it (the piece, or the voice). A model with a template for requests without instructions takes
 * them as optional. `voice` is accepted and ignored (the instructions say who sings or speaks),
 * `speed` must be 1. The length is `max_new_tokens`, in the model's frames as SGLang-Omni counts
 * it, or `duration` in seconds; without either the audio may run to the model's longest, and the
 * model usually ends it before that. Also read: `seed` (-1 or absent picks one),
 * `guidance_scale` (the model's guidance: the denoiser's for a music model, the frame model's for
 * a speech one) and, for a model with a denoiser, `num_inference_steps`.
 *
 * References, for a model that clones a voice (RadAudioGenDecl::tmpl_ref): `references` is an
 * array of {"audio", "text"}, a clip of the voice (a data: URL or the base64 of an audio file,
 * decoded to mono at the model's rate and padded to whole frames) and what it says; at most the
 * model's count of them and seconds together. The prompt is then the references' template, each
 * reference rendered from the model's item, its transcript and a row a frame of its clip.
 *
 * Prompt: the plugin's template with its slots filled, each text first rewritten by the plugin
 * into the form the model was trained on (rad_arch_text_prepare). The template is tokenised with
 * its special tokens and the texts without, so a caller cannot write the model's control tokens
 * into a lyric; a model that declares `input_special` has its input tokenised with them too (its
 * speaker tags), its prepare() refusing the control tokens. A model that declares the
 * unconditional branch's own template gets that one rendered from the same texts.
 *
 * Response: the whole piece, once it is finished, as `response_format` wav (16-bit PCM, the
 * model's rate and channels) or pcm (the same samples raw, little-endian). Streamed as the
 * vocoder plays each window, in two forms (vLLM-Omni's reading of OpenAI's `stream_format`):
 * `stream_format` "audio" sends the bytes themselves as they are made (a wav's header then states
 * no length, as a stream's cannot), and `stream` true or `stream_format` "sse" sends Server-Sent
 * Events, `speech.audio.delta` with a chunk of base64 PCM and `speech.audio.done` with the usage,
 * or `speech.audio.error`.
 */
#include "server.h"
#include "oai.h"

#include "gen/audio.h"
#include "mm/media.h"
#include "rad_internal.h"

#include "util/json.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <random>

namespace rad {
namespace server {

namespace {

void put_le(std::string* s, uint32_t v, int bytes) {
    for (int i = 0; i < bytes; ++i) s->push_back((char)((v >> (8 * i)) & 0xff));
}

}  /* namespace */

/* Samples in [-1, 1] as 16-bit PCM, as libsndfile writes float data to PCM_16 (scaled by 32767,
 * rounded, clipped). */
void append_pcm16(const std::vector<float>& x, std::string* out) {
    const size_t at = out->size();
    out->resize(at + 2 * x.size());
    char* p = &(*out)[at];
    for (size_t i = 0; i < x.size(); ++i) {
        const float v = std::min(1.0f, std::max(-1.0f, x[i]));
        const int16_t q = (int16_t)std::lrint(v * 32767.0f);
        std::memcpy(p + 2 * i, &q, 2);
    }
}

/* A RIFF/WAVE header for `bytes` of 16-bit PCM; -1 states no length, as a stream's header must,
 * with both sizes at their largest (how ffmpeg and libsndfile read a WAV of unknown length). */
std::string wav_header(int64_t bytes, int64_t rate, int64_t channels) {
    std::string h;
    h += "RIFF";
    put_le(&h, bytes < 0 ? 0xffffffffu : (uint32_t)(36 + bytes), 4);
    h += "WAVEfmt ";
    put_le(&h, 16, 4);
    put_le(&h, 1, 2);                                     /* PCM */
    put_le(&h, (uint32_t)channels, 2);
    put_le(&h, (uint32_t)rate, 4);
    put_le(&h, (uint32_t)(rate * channels * 2), 4);       /* byte rate */
    put_le(&h, (uint32_t)(channels * 2), 2);              /* block align */
    put_le(&h, 16, 2);                                    /* bits a sample */
    h += "data";
    put_le(&h, bytes < 0 ? 0xffffffffu : (uint32_t)bytes, 4);
    return h;
}

/* The request's description of the audio, `text` its input: the instructions, the tempo, the
 * length, the denoiser's steps, the guidance and the seed, and the prompt rendered from the
 * model's templates. */
int Server::speech_request(const json& body, const std::string& text,
                           std::unique_ptr<gen::AudioRequest>* out, ApiError* err) {
    const AudioGenInfo& info = deps_.audio->info();
    auto fail = [&](ApiError e) {
        *err = std::move(e);
        return -1;
    };
    auto str = [&](const char* k, std::string* v) -> int {
        if (!body.contains(k) || body[k].is_null()) return 0;
        if (!body[k].is_string()) return -1;
        *v = body[k].get<std::string>();
        return 1;
    };
    auto blank = [](const std::string& t) {
        return t.find_first_not_of(" \t\r\n") == std::string::npos;
    };
    std::string inst;
    const int has_inst = str("instructions", &inst);
    if (has_inst < 0) return fail(err_bad("instructions", "expected a string"));
    if (has_inst > 0 && blank(inst)) inst.clear();
    if (inst.empty() && info.tmpl_plain.empty())
        return fail(err_bad("instructions", "this model needs instructions, as a non-empty "
                                            "string: for a song its genre, mood, vocals and "
                                            "instruments"));
    if (!inst.empty() && info.tmpl.empty())
        return fail(err_unsupported("instructions", "this model takes no instructions; its "
                                                    "direction is written in the input"));
    if (body.contains("speed") && !body["speed"].is_null() &&
        !(body["speed"].is_number() && body["speed"].get<double>() == 1.0))
        return fail(err_unsupported("speed", "this model keeps its own tempo; the instructions "
                                             "set it"));

    auto r = std::make_unique<gen::AudioRequest>();
    /* The length: frames, or seconds, or the model's longest. */
    const bool has_frames = body.contains("max_new_tokens") && !body["max_new_tokens"].is_null();
    const bool has_dur = body.contains("duration") && !body["duration"].is_null();
    if (has_frames && has_dur)
        return fail(err_bad("duration", "give max_new_tokens or duration, not both"));
    r->max_frames = info.max_frames;
    if (has_frames) {
        if (!body["max_new_tokens"].is_number_integer() ||
            body["max_new_tokens"].get<int64_t>() < 1 ||
            body["max_new_tokens"].get<int64_t>() > info.max_frames)
            return fail(err_bad("max_new_tokens", "expected an integer in 1.." +
                                    std::to_string(info.max_frames) + " (frames, " +
                                    rad::fmt("%g", (double)info.frame_rate_num /
                                                       (double)info.frame_rate_den) +
                                    " a second)"));
        r->max_frames = body["max_new_tokens"].get<int64_t>();
    }
    if (has_dur) {
        const double max_s = (double)info.max_frames * info.frame_rate_den / info.frame_rate_num;
        if (!body["duration"].is_number() || !(body["duration"].get<double>() > 0) ||
            body["duration"].get<double>() > max_s)
            return fail(err_bad("duration", "expected seconds in (0, " +
                                    std::to_string((int64_t)max_s) + "]"));
        r->max_frames = std::max<int64_t>(
            1, std::min<int64_t>(info.max_frames,
                                 (int64_t)std::ceil(body["duration"].get<double>() *
                                                    info.frame_rate_num / info.frame_rate_den)));
    }
    if (body.contains("num_inference_steps") && !body["num_inference_steps"].is_null()) {
        if (info.default_steps <= 0)
            return fail(err_unsupported("num_inference_steps", "this model has no denoiser"));
        if (!body["num_inference_steps"].is_number_integer() ||
            body["num_inference_steps"].get<int64_t>() < 1 ||
            body["num_inference_steps"].get<int64_t>() > 1000)
            return fail(err_bad("num_inference_steps", "expected an integer in 1..1000"));
        r->steps = (int32_t)body["num_inference_steps"].get<int64_t>();
    }
    if (body.contains("guidance_scale") && !body["guidance_scale"].is_null()) {
        if (!body["guidance_scale"].is_number() || !(body["guidance_scale"].get<double>() >= 1.0) ||
            body["guidance_scale"].get<double>() > 30.0)
            return fail(err_bad("guidance_scale", "expected a number in [1, 30]"));
        if (info.branches == 1 && body["guidance_scale"].get<double>() != 1.0)
            return fail(err_unsupported("guidance_scale", "this model runs without guidance"));
        r->cfg = (float)body["guidance_scale"].get<double>();
    }
    bool seed_given = false;
    if (body.contains("seed") && !body["seed"].is_null()) {
        if (!body["seed"].is_number_integer()) return fail(err_bad("seed", "expected an integer"));
        r->seed = (uint64_t)body["seed"].get<int64_t>();
        seed_given = body["seed"].is_number_unsigned() || body["seed"].get<int64_t>() != -1;
    }
    if (!seed_given) {
        /* 53 bits, as the images endpoints choose one: JavaScript reads a number as a double. */
        std::random_device rd;
        r->seed = (((uint64_t)rd() << 32) ^ rd()) & (((uint64_t)1 << 53) - 1);
    }

    /* ---- the references: a clip of a voice and what it says, each */
    struct Ref { std::string text; int64_t frames = 0; };
    std::vector<Ref> refs;
    if (body.contains("references") && !body["references"].is_null()) {
        const json& rs = body["references"];
        if (!rs.is_array())
            return fail(err_bad("references", "expected an array of {\"audio\", \"text\"}"));
        if (!rs.empty() && info.tmpl_ref.empty())
            return fail(err_unsupported("references", "this model takes no reference audio; its "
                                                      "voice is described in the input"));
        if ((int64_t)rs.size() > info.ref_max)
            return fail(err_bad("references", "at most " + std::to_string(info.ref_max) +
                                                  " references a request"));
        const int64_t hop = std::max<int64_t>(1, info.ref_hop);
        r->ref_cu.assign(1, 0);
        for (size_t i = 0; i < rs.size(); ++i) {
            const std::string at = "references[" + std::to_string(i) + "]";
            const json& e = rs[i];
            if (!e.is_object())
                return fail(err_bad(at, "expected an object {\"audio\", \"text\"}"));
            for (auto it = e.begin(); it != e.end(); ++it)
                if (it.key() != "audio" && it.key() != "text")
                    return fail(err_bad(at + "." + it.key(), "a reference is its audio and its "
                                                             "text"));
            if (!e.contains("audio") || !e["audio"].is_string())
                return fail(err_bad(at + ".audio", "the clip is required, as a data: URL or "
                                                   "base64 of an audio file"));
            if (!e.contains("text") || !e["text"].is_string() || blank(e["text"].get<std::string>()))
                return fail(err_bad(at + ".text", "the clip's transcript is required, as a "
                                                  "non-empty string"));
            /* The file: a data: URL, or its base64 alone (as fish-speech's server takes it). */
            const std::string& a = e["audio"].get_ref<const std::string&>();
            std::vector<uint8_t> bytes;
            if (a.rfind("data:", 0) == 0) {
                ApiError ue;
                if (decode_media_url(a, at + ".audio", &bytes, &ue) < 0) return fail(ue);
            } else if (!base64_decode(a, &bytes)) {
                return fail(err_bad(at + ".audio", "invalid base64"));
            }
            /* Mono at the model's rate; one sample past what is left of the budget is decoded,
             * so a clip too long is refused rather than cut. */
            const int64_t left = info.ref_max_samples - (int64_t)r->ref_audio.size();
            std::vector<float> pcm;
            std::string dwhy;
            if (left < hop)
                return fail(err_bad(at + ".audio", "the references are longer than this model "
                                                   "takes"));
            if (mm::decode_audio(bytes.data(), bytes.size(), (int32_t)info.ref_rate, left + 1,
                                 &pcm, &dwhy) < 0)
                return fail(err_bad(at + ".audio", "the clip could not be used: " + dwhy));
            if (pcm.empty()) return fail(err_bad(at + ".audio", "the clip holds no audio"));
            const int64_t padded = ((int64_t)pcm.size() + hop - 1) / hop * hop;
            if (padded > left)
                return fail(err_bad(at + ".audio", "the references hold more than " +
                                                       rad::fmt("%g", (double)info.ref_max_samples /
                                                                          (double)info.ref_rate) +
                                                       " seconds together; this model takes no "
                                                       "more"));
            pcm.resize((size_t)padded, 0.0f);
            r->ref_audio.insert(r->ref_audio.end(), pcm.begin(), pcm.end());
            r->ref_cu.push_back((int32_t)r->ref_audio.size());
            Ref ref;
            std::string why;
            if (deps_.audio->prepare(RAD_TEXT_REFERENCE, e["text"].get<std::string>(), &ref.text,
                                     &why) < 0)
                return fail(err_bad(at + ".text", why));
            ref.frames = padded / hop;
            refs.push_back(std::move(ref));
        }
        if (refs.empty()) r->ref_cu.clear();
    }
    const bool ref_direction = !refs.empty() && !inst.empty();
    if (ref_direction && info.tmpl_ref_instruct.empty())
        return fail(err_unsupported("instructions", "a request with references takes no "
                                                    "instructions; the clips set the voice"));
    /* A model whose negative template omits the reference has not declared reference CFG.
     * In particular Breeze's ref_clone_tata is an unguided continuation, not voice design. */
    if (!refs.empty() && info.branches != 1 && r->cfg > 1.0f &&
        (!ref_direction || info.neg_tmpl_ref.empty()))
        return fail(err_unsupported("guidance_scale", "voice cloning requires guidance_scale 1; "
                                                     "this model has no guided reference template"));

    /* ---- the prompt */
    std::string i2, t2, why;
    if (!inst.empty() && deps_.audio->prepare(RAD_TEXT_INSTRUCTIONS, inst, &i2, &why) < 0)
        return fail(err_bad("instructions", why));
    if (deps_.audio->prepare(RAD_TEXT_INPUT, text, &t2, &why) < 0)
        return fail(err_bad("input", why));
    auto encode = [&](const std::string& t, bool special, std::vector<int32_t>* ids_out) {
        if (t.empty()) return;
        const auto ids = deps_.tok->encode(t, false, /*parse_special=*/special);
        ids_out->insert(ids_out->end(), ids.begin(), ids.end());
    };
    auto subst = [](std::string t, const std::string& key, const std::string& v) {
        for (size_t p = t.find(key); p != std::string::npos; p = t.find(key, p + v.size()))
            t.replace(p, key.size(), v);
        return t;
    };
    /* One reference (ref_item): its header, the transcript in it, tokenised as one segment with
     * its special tokens (the speaker tags prepare() allows), a row a frame of its clip, and what
     * follows the clip as another segment. */
    auto render_ref = [&](size_t i, std::vector<int32_t>* ids_out,
                          std::vector<int32_t>* frame_map, int64_t& frame0) {
        static const std::string kAudio = "{audio}";
        const std::string idx = std::to_string(i + 1);
        const size_t a = info.ref_item.find(kAudio);
        encode(subst(subst(info.ref_item.substr(0, a), "{index}", idx), "{text}", refs[i].text),
               true, ids_out);
        frame_map->resize(ids_out->size(), -1);
        for (int64_t f = 0; f < refs[i].frames; ++f) {
            ids_out->push_back(0);
            frame_map->push_back((int32_t)(frame0 + f));
        }
        frame0 += refs[i].frames;
        encode(subst(info.ref_item.substr(a + kAudio.size()), "{index}", idx), true, ids_out);
    };
    /* A template a segment at a time: its own text with special tokens, each slot without. */
    auto render = [&](const std::string& t, std::vector<int32_t>* ids_out,
                      std::vector<int32_t>* frame_map) {
        static const std::string kInst = "{instructions}", kInput = "{input}",
                                 kRefs = "{references}";
        size_t at = 0;
        int64_t frame0 = 0;
        while (at <= t.size()) {
            const size_t a = t.find(kInst, at), b = t.find(kInput, at), c = t.find(kRefs, at);
            const size_t p = std::min(a, std::min(b, c));
            encode(t.substr(at, p == std::string::npos ? std::string::npos : p - at), true,
                   ids_out);
            if (p == std::string::npos) break;
            if (p == c) {
                for (size_t i = 0; i < refs.size(); ++i) render_ref(i, ids_out, frame_map, frame0);
                at = p + kRefs.size();
                continue;
            }
            const bool is_inst = p == a;
            /* The input keeps its special tokens only for a model that asks (its speaker tags);
             * the plugin's prepare() has refused the ones a caller may not write. */
            encode(is_inst ? i2 : t2, !is_inst && info.input_special, ids_out);
            at = p + (is_inst ? kInst : kInput).size();
        }
        if (frame_map) frame_map->resize(ids_out->size(), -1);
    };
    render(ref_direction ? info.tmpl_ref_instruct : !refs.empty() ? info.tmpl_ref :
           inst.empty() ? info.tmpl_plain : info.tmpl, &r->tokens,
           refs.empty() ? nullptr : &r->ref_frame);
    if (ref_direction)
        render(info.neg_tmpl_ref, &r->neg_tokens, &r->neg_ref_frame);
    else if (!info.neg_tmpl.empty()) render(info.neg_tmpl, &r->neg_tokens, nullptr);
    const size_t n_prompt = std::max(r->tokens.size(), r->neg_tokens.size());
    if ((int64_t)n_prompt > info.max_prompt)
        return fail(err_bad("input", "the prompt is " + std::to_string(n_prompt) +
                                         " tokens; this model takes at most " +
                                         std::to_string(info.max_prompt)));
    *out = std::move(r);
    return 0;
}

HttpResponse Server::handle_speech(const HttpRequest& req) {
    if (!deps_.audio || !deps_.tok)
        return error_response(err_not_implemented(
            opt_.image_model ? "/v1/audio/speech requires an audio generation model; this one "
                               "makes images"
                             : "/v1/audio/speech requires an audio generation model; this one "
                               "generates text"), req.peer);
    const AudioGenInfo& info = deps_.audio->info();

    json body = json::object();
    {
        ApiError err;
        if (parse_request_object(req.body, lim_, &body, &err) < 0)
            return error_response(err, req.peer);
    }
    static const char* kKnown[] = {
        "model", "input", "instructions", "voice", "response_format", "speed", "stream",
        "stream_format", "seed", "max_new_tokens", "duration", "num_inference_steps",
        "guidance_scale", "user", "references",
    };
    for (auto it = body.begin(); it != body.end(); ++it) {
        bool known = false;
        for (const char* k : kKnown) known = known || it.key() == k;
        if (!known)
            return error_response(err_bad(it.key(), "'" + it.key() + "' is not a field of "
                                                     "/v1/audio/speech"), req.peer);
    }
    auto str = [&](const char* k, std::string* out) -> int {
        if (!body.contains(k) || body[k].is_null()) return 0;
        if (!body[k].is_string()) return -1;
        *out = body[k].get<std::string>();
        return 1;
    };
    std::string text, fmt;
    if (str("input", &text) <= 0 || text.find_first_not_of(" \t\r\n") == std::string::npos)
        return error_response(err_bad("input", "the text is required, as a non-empty string"),
                              req.peer);
    /* How the audio comes back: whole, as raw bytes while it is made, or as events. */
    bool stream = false;
    if (body.contains("stream") && !body["stream"].is_null()) {
        if (!body["stream"].is_boolean())
            return error_response(err_bad("stream", "expected a boolean"), req.peer);
        stream = body["stream"].get<bool>();
    }
    std::string sfmt;
    if (str("stream_format", &sfmt) < 0 || (!sfmt.empty() && sfmt != "sse" && sfmt != "audio")) {
        ApiError e = err_unsupported("stream_format", "a stream is sent as sse or audio");
        e.supported = { "sse", "audio" };
        return error_response(e, req.peer);
    }
    if (stream && sfmt == "audio")
        return error_response(err_bad("stream", "stream sends Server-Sent Events and "
                                                "stream_format audio sends the bytes themselves; "
                                                "give one"), req.peer);
    const bool sse = stream || sfmt == "sse", raw = sfmt == "audio";
    if (str("response_format", &fmt) < 0) fmt = "?";
    if (fmt.empty()) fmt = sse ? "pcm" : "wav";
    if (fmt != "wav" && fmt != "pcm") {
        ApiError e = err_unsupported("response_format", "this server writes wav or pcm");
        e.supported = { "wav", "pcm" };
        return error_response(e, req.peer);
    }
    if (sse && fmt != "pcm")
        return error_response(err_unsupported("response_format", "an event carries raw PCM; set "
                                                                 "response_format pcm"), req.peer);
    std::unique_ptr<gen::AudioRequest> r;
    {
        ApiError err;
        if (speech_request(body, text, &r, &err) < 0) return error_response(err, req.peer);
    }

    /* ---- admitted and queued */
    if (auto no = admit(1, req.peer)) return *no;
    r->ticket = std::make_shared<gen::AudioTicket>();
    std::shared_ptr<gen::AudioTicket> ticket = r->ticket;
    const int64_t frames_max = r->max_frames;
    const uint64_t seed = r->seed;
    const int64_t n_tok = (int64_t)r->tokens.size();
    {
        std::string w;
        const int s = deps_.audio->submit(std::move(r), &w);
        if (s < 0) {
            release(1);
            ApiError e = err_bad("", w.empty() ? std::string(rad_strerror(s)) : w);
            if (s != RAD_E_INVAL) { e.status = 503; e.type = "server_error"; }
            return error_response(e, req.peer);
        }
    }
    RAD_INFO("audio: %lld prompt tokens, up to %lld frames, seed=%llu", (long long)n_tok,
             (long long)frames_max, (unsigned long long)seed);

    if (sse || raw) return stream_speech(ticket, info, n_tok, seed, sse, fmt == "wav");

    /* ---- the windows as they are played, in order */
    std::vector<gen::AudioChunk> chunks;
    for (;;) {
        gen::AudioChunk c;
        bool to = false;
        if (ticket->next(&c, 200, &to)) {
            chunks.push_back(std::move(c));
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
    if (ticket->status() < 0 || chunks.empty()) {
        /* RAD_E_INVAL is the driver's refusal of this request (or the model ending the piece
         * before it began), so a 400; anything else failed while running. */
        ApiError e;
        const bool client = ticket->status() == RAD_E_INVAL;
        e.status = client ? 400 : 500;
        e.type = client ? "invalid_request_error" : "server_error";
        e.message = ticket->why().empty() ? "the audio generator failed" : ticket->why();
        return error_response(e, req.peer);
    }
    std::sort(chunks.begin(), chunks.end(), [](const gen::AudioChunk& a, const gen::AudioChunk& b) {
        return a.window < b.window;
    });
    std::string pcm;
    for (const gen::AudioChunk& c : chunks) append_pcm16(c.samples, &pcm);
    const double secs = (double)pcm.size() / 2.0 / (double)(info.channels * info.sample_rate);
    auto sec = [](int64_t a, int64_t b) { return a > 0 && b >= a ? (double)(b - a) * 1e-9 : 0.0; };
    RAD_INFO("audio: %.1f s from %lld frames in %lld windows, %.1f s: queue and prompt %.1f s, "
             "first audio %.1f s, frames %.1f s%s", secs, (long long)ticket->frames,
             (long long)ticket->windows, sec(ticket->t_submit_ns, ticket->t_done_ns),
             sec(ticket->t_submit_ns, ticket->t_prompt_ns),
             sec(ticket->t_submit_ns, ticket->t_first_audio_ns),
             sec(ticket->t_prompt_ns, ticket->t_frames_ns),
             ticket->model_ended ? "" : " (the request's length)");
    HttpResponse res;
    res.status = 200;
    if (fmt == "wav") {
        res.content_type = "audio/wav";
        res.body = wav_header((int64_t)pcm.size(), info.sample_rate, info.channels) + pcm;
    } else {
        res.content_type = "audio/pcm";
        res.body = std::move(pcm);
    }
    res.set_header("X-Audio-Sample-Rate", std::to_string(info.sample_rate));
    res.set_header("X-Audio-Channels", std::to_string(info.channels));
    res.set_header("X-Seed", std::to_string(seed));
    metrics_.observe_http(200);
    return res;
}

/* The piece as it is played: each window's samples once every earlier window has gone (the driver
 * may finish them out of order), as bytes or as `speech.audio.delta` events. */
HttpResponse Server::stream_speech(std::shared_ptr<gen::AudioTicket> ticket,
                                   const AudioGenInfo& info, int64_t n_tok, uint64_t seed,
                                   bool sse, bool wav) {
    HttpResponse res;
    res.status = 200;
    res.content_type = sse ? "text/event-stream" : wav ? "audio/wav" : "audio/pcm";
    if (sse) res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Audio-Sample-Rate", std::to_string(info.sample_rate));
    res.set_header("X-Audio-Channels", std::to_string(info.channels));
    res.set_header("X-Seed", std::to_string(seed));
    struct State {
        std::map<int32_t, std::vector<float>> held;   /* windows ahead of the next one */
        int32_t next = 0;
        bool header = false, finished = false, released = false;
    };
    auto st = std::make_shared<State>();
    st->header = !wav || sse;
    const int64_t rate = info.sample_rate, channels = info.channels;
    res.next = [this, ticket, st, sse, rate, channels, n_tok](std::string& chunk) -> bool {
        if (st->finished) return false;
        chunk.clear();
        if (!st->header) {
            chunk = wav_header(-1, rate, channels);
            st->header = true;
        }
        gen::AudioChunk c;
        bool to = false;
        if (ticket->next(&c, 100, &to)) {
            st->held[c.window] = std::move(c.samples);
            std::string pcm;
            for (auto it = st->held.find(st->next); it != st->held.end();
                 it = st->held.find(st->next)) {
                append_pcm16(it->second, &pcm);
                st->held.erase(it);
                ++st->next;
            }
            if (pcm.empty()) return true;
            if (!sse) {
                chunk += pcm;
                return true;
            }
            json ev;
            ev["type"] = "speech.audio.delta";
            ev["audio"] = base64_encode(pcm.data(), pcm.size());
            ev["response_format"] = "pcm";
            ev["sample_rate"] = rate;
            ev["channels"] = channels;
            chunk += "event: speech.audio.delta\ndata: " + json_dump(ev) + "\n\n";
            return true;
        }
        if (to) return !stopping_.load();
        st->finished = true;
        if (!st->released) { st->released = true; release(1); }
        if (!sse) return !chunk.empty();
        json ev;
        if (ticket->status() < 0 || st->next == 0) {
            ev["type"] = "speech.audio.error";
            ev["error"] = json{
                { "message", ticket->why().empty() ? "the audio generator failed" : ticket->why() },
                { "type", ticket->status() == RAD_E_INVAL ? "invalid_request_error"
                                                          : "server_error" } };
        } else {
            ev["type"] = "speech.audio.done";
            ev["usage"] = json{ { "input_tokens", n_tok }, { "output_tokens", ticket->frames },
                                { "total_tokens", n_tok + ticket->frames } };
        }
        chunk += "event: " + ev["type"].get<std::string>() + "\ndata: " + json_dump(ev) + "\n\n";
        return true;
    };
    res.on_close = [this, ticket, st](bool ok) {
        if (!ok) ticket->cancel.store(true);
        if (!st->released) { st->released = true; release(1); }
    };
    metrics_.observe_http(200);
    return res;
}

}  /* namespace server */
}  /* namespace rad */
