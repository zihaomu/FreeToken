/* speech.cpp -- OpenAI's POST /v1/audio/transcriptions, over the model's own speech format.
 *
 * What this route knows about the model is what its plugin states: the turn a transcription is
 * asked in, how a language is forced and named, and where the transcript starts in the reply
 * (rad_arch_audio_format, copied into SpeechFormat at startup). This file knows OpenAI's side: the
 * form, the response shapes and the stream's events.
 *
 * Request: multipart/form-data, as OpenAI's is: `file` (required), `model`, `language` (a code or
 * a name the model knows), `prompt` (context for the recognition: a name, a domain),
 * `response_format` (json, text or verbose_json), `stream`, and `temperature`, which must be 0
 * because the decode is greedy, as the reference model's is. Word timestamps are refused rather
 * than answered with nothing.
 *
 * A recording longer than one encoder pass is cut into clips, each ending at the quietest 100 ms of
 * the last five seconds before the pass would be full, so a cut lands in a pause where there is
 * one. Each clip is one prompt; the clips run a wave at a time and their transcripts are joined
 * with a space.
 *
 * Prompt: the plugin's template with its three slots filled: the request's `prompt` as plain text
 * (tokenised with no special tokens parsed, so a caller cannot write the model's control tokens
 * into it), the clip's placeholder run, and the forced-language text or nothing.
 */
#include "server.h"
#include "server_priv.h"

#include "mm/processor.h"
#include "rad_internal.h"

#include "util/json.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <ctime>

namespace rad {
namespace server {

std::string SpeechFormat::language_name(const std::string& lang) const {
    auto lower = [](std::string s) {
        for (char& c : s) c = (char)std::tolower((unsigned char)c);
        return s;
    };
    const std::string l = lower(lang);
    for (const auto& p : languages)
        if (lower(p.first) == l || lower(p.second) == l) return p.second;
    return std::string();
}

namespace {

/* The most a clip's transcript may run to. Thirty seconds of the fastest speech is a few hundred
 * tokens; the bound is there for a model that loops, not for one that talks. */
constexpr int32_t kMaxTokensPerClip = 512;
/* The longest recording one request transcribes. */
constexpr int64_t kMaxSeconds = 3600;
/* Clips submitted together. */
constexpr size_t kWave = 8;

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

/* Where a long recording is cut: [start, end) clips of at most `max_n` samples, each ending at the
 * quietest point in the last five seconds (or half a clip, if that is shorter) before the clip
 * would be full. */
std::vector<std::pair<int64_t, int64_t>> cut_clips(const std::vector<float>& x, int64_t max_n,
                                                   int32_t rate) {
    std::vector<std::pair<int64_t, int64_t>> out;
    const int64_t n = (int64_t)x.size();
    const int64_t search = std::min<int64_t>(max_n / 2, (int64_t)rate * 5);
    int64_t start = 0;
    while (n - start > max_n) {
        const int64_t hi = start + max_n;
        const int64_t best = speech_quiet_point(x, hi - search, hi, rate);
        out.emplace_back(start, best);
        start = best;
    }
    out.emplace_back(start, n);
    return out;
}

/* What one clip's reply says: the transcript, and the language when the model named it. A reply
 * the language was forced in is the transcript alone. */
struct Heard {
    std::string language;
    std::string text;
};

Heard read_reply(const SpeechFormat& f, const std::string& reply, const std::string& forced) {
    if (!forced.empty() || f.transcript_open.empty()) return { forced, trim(reply) };
    const size_t at = reply.find(f.transcript_open);
    if (at == std::string::npos) return { std::string(), trim(reply) };
    Heard h;
    h.text = trim(reply.substr(at + f.transcript_open.size()));
    const std::string head = reply.substr(0, at);
    const size_t lead = f.language_lead.empty() ? std::string::npos : head.rfind(f.language_lead);
    const std::string said = trim(lead == std::string::npos ? head
                                                            : head.substr(lead + f.language_lead.size()));
    /* A name the model does not list (its own "None" for a clip with no speech) is no
     * language. */
    h.language = f.language_name(said);
    return h;
}

std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

}  /* namespace */

/* ------------------------------------------------------------------ shared with realtime.cpp */

int speech_prompt(const SpeechFormat& f, ITokenizer& tok, const std::string& context,
                  const std::string& language_text, const std::shared_ptr<mm::Item>& item,
                  PromptInput* out) {
    auto pm = std::make_shared<mm::PromptMedia>();
    const std::string& t = f.transcribe_prompt;
    auto add = [&](const std::string& text, bool special) {
        if (text.empty()) return;
        const std::vector<int32_t> ids = tok.encode(text, /*add_special=*/false, special);
        out->tokens.insert(out->tokens.end(), ids.begin(), ids.end());
        out->text += text;
    };
    size_t pos = 0;
    bool placed = false;
    while (pos < t.size()) {
        size_t at = t.find('{', pos);
        if (at == std::string::npos) at = t.size();
        add(t.substr(pos, at - pos), true);
        if (at == t.size()) break;
        if (!t.compare(at, 9, "{context}")) {
            add(context, false);
            pos = at + 9;
        } else if (!t.compare(at, 10, "{language}")) {
            add(language_text, true);
            pos = at + 10;
        } else if (!t.compare(at, 7, "{audio}")) {
            mm::Placed pl;
            pl.item = item;
            pl.tok = (int32_t)out->tokens.size();
            out->tokens.insert(out->tokens.end(), item->expand.begin(), item->expand.end());
            out->text += RAD_MEDIA_MARKER;
            pm->items.push_back(std::move(pl));
            placed = true;
            pos = at + 7;
        } else {
            add("{", true);
            pos = at + 1;
        }
    }
    if (!placed) return RAD_E_INVAL;
    pm->layout();
    out->media = std::move(pm);
    return RAD_OK;
}

std::string speech_language_text(const SpeechFormat& f, const std::string& name) {
    std::string t = f.language_prompt;
    const size_t at = t.find("{language}");
    if (at != std::string::npos) t.replace(at, 10, name);
    return t;
}

size_t speech_transcript_start(const SpeechFormat& f, const std::string& text, bool forced,
                               bool closed) {
    size_t s = 0;
    if (!forced && !f.transcript_open.empty()) {
        const size_t at = text.find(f.transcript_open);
        if (at == std::string::npos) return closed ? 0 : std::string::npos;
        s = at + f.transcript_open.size();
    }
    while (s < text.size() && std::isspace((unsigned char)text[s])) ++s;
    return s;
}

int64_t speech_quiet_point(const std::vector<float>& x, int64_t lo, int64_t hi, int32_t rate) {
    const int64_t win = std::max<int64_t>(2, rate / 10);
    lo = std::max<int64_t>(lo, 0);
    int64_t best = hi;
    double best_e = INFINITY;
    for (int64_t a = lo; a + win <= hi; a += win / 2) {
        double e = 0.0;
        for (int64_t i = a; i < a + win; ++i) e += (double)x[(size_t)i] * x[(size_t)i];
        if (e < best_e) { best_e = e; best = a + win / 2; }
    }
    return best;
}

/* One transcription request: the recording, where it is cut, and what each clip heard. Clips
 * are prepared a wave at a time, so a long recording holds its samples and one wave's features,
 * not every clip's at once. The request holds `places` admission places from its first wave to
 * its last, so a later wave is never refused once the first has run. Shared between the handler
 * and, for a stream, the transport's callbacks. */
struct Server::TranscriptionJob {
    OaiRequest              base;        /* everything but the prompts and max_tokens */
    std::vector<float>      samples;     /* the whole recording at the model's rate */
    std::vector<std::pair<int64_t, int64_t>> cuts;   /* each clip's [start, end) in samples */
    int32_t                 rate = 0;
    std::string             context;     /* the request's `prompt`, plain text */
    std::string             language_text;
    std::string             forced;      /* the forced language's name, "" for none */
    std::vector<std::string> replies;    /* each clip's whole reply, once its wave is done */
    std::vector<Heard>      heard;
    int64_t n_prompt = 0, n_gen = 0;
    size_t  next = 0;                    /* the first clip not yet submitted */
    int64_t places = 0;                  /* admission places held, 0 once released */

    /* The wave in flight. */
    OaiRequest r;
    std::vector<Generation> gens;
    bool    running = false;

    /* Streaming: how much of the stream's text has gone out, and the clip it is reading. */
    size_t  stream_clip = 0;
    size_t  stream_off = 0;
    bool    stream_started = false;
    bool    done_sent = false;
    std::string streamed;
};

HttpResponse Server::handle_transcription(const HttpRequest& req) {
    if (!deps_.sched || !deps_.tok)
        return error_response(err_not_implemented("the server has no engine attached"), req.peer);
    if (!deps_.speech || !deps_.speech->transcribes() || !deps_.mm ||
        !deps_.mm->accepts(MediaKind::Audio) || deps_.mm->audio_rate() <= 0)
        return error_response(err_not_implemented(
                                  "this model does not transcribe: its architecture plugin states "
                                  "no transcription format (rad_arch_audio_format), or it was "
                                  "started without its audio encoder"),
                              req.peer);
    const SpeechFormat& f = *deps_.speech;

    /* ---- the form */
    if (req.form.empty())
        return error_response(err_bad("file", "expected multipart/form-data with a `file` part, "
                                              "as OpenAI's transcription API takes"),
                              req.peer);
    auto field = [&](const char* k) {
        const FormPart* p = req.part(k);
        return p ? trim(p->data) : std::string();
    };
    const FormPart* file = req.part("file");
    if (!file || file->data.empty())
        return error_response(err_bad("file", "the audio file is missing or empty"), req.peer);
    const std::string fmt = field("response_format").empty() ? "json" : field("response_format");
    if (fmt != "json" && fmt != "text" && fmt != "verbose_json")
        return error_response(err_unsupported("response_format",
                                              "json, text or verbose_json; not '" + fmt + "'"),
                              req.peer);
    const std::string stream_s = lower(field("stream"));
    const bool stream = stream_s == "true" || stream_s == "1";
    if (!stream_s.empty() && !stream && stream_s != "false" && stream_s != "0")
        return error_response(err_bad("stream", "expected true or false"), req.peer);
    if (const std::string t = field("temperature"); !t.empty()) {
        char* end = nullptr;
        const double v = std::strtod(t.c_str(), &end);
        if (end == t.c_str() || *end != '\0' || !std::isfinite(v))
            return error_response(err_bad("temperature", "expected a number, not '" + t + "'"),
                                  req.peer);
        if (v != 0.0)
            return error_response(err_unsupported("temperature",
                                                  "this route decodes greedily, as the model's "
                                                  "reference does; send 0 or leave it out"),
                                  req.peer);
    }
    for (const FormPart& p : req.form)
        if ((p.name == "timestamp_granularities[]" || p.name == "timestamp_granularities") &&
            trim(p.data) == "word")
            return error_response(err_unsupported("timestamp_granularities",
                                                   "word timestamps are not produced; verbose_json "
                                                   "carries one segment a clip"),
                                  req.peer);
    std::string forced;
    if (!field("language").empty()) {
        forced = f.language_name(field("language"));
        if (forced.empty()) {
            std::string known;
            for (const auto& l : f.languages) known += (known.empty() ? "" : ", ") + l.first;
            return error_response(err_bad("language", "this model names no language '" +
                                                          field("language") + "'; it knows " +
                                                          known),
                                  req.peer);
        }
    }

    /* ---- the recording, and where it is cut into clips one encoder pass each */
    auto job = std::make_shared<TranscriptionJob>();
    job->rate = deps_.mm->audio_rate();
    const int32_t rate = job->rate;
    const int64_t max_clip = deps_.mm->audio_max_samples();
    {
        std::string why;
        const std::vector<uint8_t> bytes(file->data.begin(), file->data.end());
        const int s = deps_.mm->decode_audio(bytes, kMaxSeconds * rate, &job->samples, &why);
        if (s < 0) return error_response(err_bad("file", "the audio could not be used: " + why),
                                         req.peer);
    }
    if (job->samples.empty())
        return error_response(err_bad("file", "the recording holds no audio"), req.peer);
    job->cuts = cut_clips(job->samples, max_clip > 0 ? max_clip : (int64_t)job->samples.size(),
                          rate);
    job->context = field("prompt");
    job->forced = forced;
    job->language_text = forced.empty() ? std::string() : speech_language_text(f, forced);
    job->replies.assign(job->cuts.size(), std::string());
    job->heard.assign(job->cuts.size(), Heard{});

    OaiRequest& b = job->base;
    b.endpoint = Endpoint::Completion;
    b.shape = ApiShape::OpenAI;
    b.model = opt_.model_id;
    b.id = random_id("tr-");
    b.created = (int64_t)std::time(nullptr);
    b.peer = req.peer;
    b.n = 1;
    b.sp = SamplingParams{};
    b.sp.temp = 0.0f;
    b.sp.top_k = 1;

    /* A wave is at most kWave clips and never more than the admission queue holds. */
    const size_t wave = (size_t)std::max<int64_t>(
        1, std::min<int64_t>({ (int64_t)kWave, (int64_t)job->cuts.size(), queue_depth_ }));
    if (auto no = admit((int64_t)wave, req.peer)) return *no;
    job->places = (int64_t)wave;

    sweep_abandoned();
    Server* self = this;
    const SpeechFormat* fp = &f;
    auto release_job = [self, job]() {
        if (job->places > 0) { self->release(job->places); job->places = 0; }
    };
    /* Prepare and submit the next wave; false with `*err` set when it could not be. */
    auto start_wave = [self, job, fp, wave](ApiError* err) -> bool {
        const size_t n = std::min(wave, job->cuts.size() - job->next);
        job->r = job->base;
        job->r.prompts.clear();
        int64_t longest = 0;
        for (size_t i = job->next; i < job->next + n; ++i) {
            const std::vector<float> clip(job->samples.begin() + job->cuts[i].first,
                                          job->samples.begin() + job->cuts[i].second);
            std::shared_ptr<mm::Item> item;
            std::string why;
            if (self->deps_.mm->prepare_audio(clip, &item, &why) < 0 || !item) {
                *err = err_bad("file", "the audio could not be used: " + why);
                return false;
            }
            PromptInput p;
            if (speech_prompt(*fp, *self->deps_.tok, job->context, job->language_text, item,
                              &p) < 0) {
                *err = err_not_implemented("the model's transcription template has no {audio} "
                                           "slot");
                return false;
            }
            longest = std::max<int64_t>(longest, (int64_t)p.tokens.size());
            job->n_prompt += (int64_t)p.tokens.size();
            job->r.prompts.push_back(std::move(p));
        }
        job->r.max_tokens = kMaxTokensPerClip;
        if (self->lim_.max_ctx > 0) {
            if (longest >= self->lim_.max_ctx) {
                *err = err_bad("file", "a clip's prompt (" + std::to_string(longest) +
                                           " tokens) is longer than the context");
                return false;
            }
            job->r.max_tokens = (int32_t)std::min<int64_t>(job->r.max_tokens,
                                                           self->lim_.max_ctx - longest);
        }
        if (job->next == 0) self->log_admitted(job->r);
        job->gens.clear();
        if (self->submit_all(job->r, &job->gens, err) < 0) {
            self->retire(job->gens);
            return false;
        }
        job->running = true;
        return true;
    };
    /* A wave that has finished: its replies kept, its sequences retired. Error when one failed. */
    auto finish_wave = [self, job, fp](std::string* failure) -> bool {
        Finish overall = Finish::Stop;
        for (const auto& g : job->gens) {
            const size_t k = job->next + (size_t)g.prompt_index;
            job->replies[k] = g.text;
            job->heard[k] = read_reply(*fp, g.text, job->forced);
            job->n_gen += g.sink->n_produced();
            if (g.finish == Finish::Error) {
                overall = Finish::Error;
                if (failure) *failure = engine_failure(*g.sink);
            }
        }
        self->record(job->r, job->gens, overall);
        self->retire(job->gens);
        job->next += job->r.prompts.size();
        job->running = false;
        return overall != Finish::Error;
    };
    auto joined = [job]() {
        std::string t;
        for (const Heard& h : job->heard) {
            if (h.text.empty()) continue;
            if (!t.empty()) t += ' ';
            t += h.text;
        }
        return t;
    };
    auto usage = [job]() {
        json u;
        u["type"] = "tokens";
        u["input_tokens"] = job->n_prompt;
        u["output_tokens"] = job->n_gen;
        u["total_tokens"] = job->n_prompt + job->n_gen;
        return u;
    };
    auto internal = [](const std::string& why) {
        ApiError e;
        e.status = 500;
        e.type = "internal_server_error";
        e.message = why;
        return e;
    };

    if (!stream) {
        while (job->next < job->cuts.size()) {
            ApiError err;
            if (!start_wave(&err)) {
                release_job();
                return error_response(err, req.peer);
            }
            for (;;) {
                pump(job->gens, job->r);
                if (all_done(job->gens)) break;
                if (req.peer_gone() || stopping_.load()) {
                    cancel_all(job->gens, req.peer_gone());
                    record(job->r, job->gens, Finish::Cancelled);
                    retire(job->gens);
                    release_job();
                    if (req.peer_gone()) {
                        metrics_.observe_http(499);
                        return HttpResponse::text(499, "client closed request\n");
                    }
                    return error_response(err_overloaded("the server is shutting down"), req.peer);
                }
                for (auto& g : job->gens) if (!g.closed) { g.sink->wait(); break; }
            }
            std::string failure;
            if (!finish_wave(&failure)) {
                release_job();
                return error_response(internal(failure), req.peer);
            }
        }
        release_job();
        const std::string text = joined();
        metrics_.observe_http(200);
        if (fmt == "text") return HttpResponse::text(200, text + "\n");
        json o;
        if (fmt == "verbose_json") {
            std::string lang = forced;
            for (const Heard& h : job->heard)
                if (lang.empty() && !h.language.empty()) lang = h.language;
            o["task"] = "transcribe";
            o["language"] = lower(lang);
            o["duration"] = (double)job->samples.size() / rate;
            o["text"] = text;
            json segs = json::array();
            for (size_t i = 0; i < job->heard.size(); ++i) {
                json s;
                s["id"] = (int64_t)i;
                s["start"] = (double)job->cuts[i].first / rate;
                s["end"] = (double)job->cuts[i].second / rate;
                s["text"] = job->heard[i].text;
                segs.push_back(std::move(s));
            }
            o["segments"] = std::move(segs);
        } else {
            o["text"] = text;
        }
        o["usage"] = usage();
        return HttpResponse::json_body(200, json_dump(o));
    }

    /* ---- the stream: transcript.text.delta events as each clip's transcript settles, in clip
     * order, then transcript.text.done with the whole text. */
    {
        ApiError err;
        if (!start_wave(&err)) {
            release_job();
            return error_response(err, req.peer);
        }
    }
    HttpResponse res;
    res.status = 200;
    res.content_type = "text/event-stream; charset=utf-8";
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_header("X-Accel-Buffering", "no");
    auto peer_closed = req.closed;
    res.next = [self, job, peer_closed, fp, start_wave, finish_wave, joined, usage, internal,
                release_job](std::string& out) -> bool {
        out.clear();
        if (job->done_sent) return false;
        if (peer_closed && peer_closed()) return false;
        auto delta = [&](const std::string& d) {
            if (d.empty()) return;
            json o;
            o["type"] = "transcript.text.delta";
            o["delta"] = d;
            out += sse_event(json_dump(o));
            job->streamed += d;
        };
        const bool progress = job->running && self->pump(job->gens, job->r);
        /* Whatever of the clip being read has settled; the clips after it wait their turn. */
        while (job->stream_clip < job->cuts.size()) {
            const size_t k = job->stream_clip;
            const bool in_wave = k >= job->next && k < job->next + job->gens.size() && job->running;
            const Generation* g = in_wave ? &job->gens[k - job->next] : nullptr;
            const bool closed = g ? g->closed : k < job->next;
            const std::string& text = g ? g->text : job->replies[k];
            const size_t s = speech_transcript_start(*fp, text, !job->forced.empty(), closed);
            if (s != std::string::npos) {
                std::string end = closed ? trim(text.substr(std::min(s, text.size())))
                                         : text.substr(s);
                if (closed && end.size() < job->stream_off) end.clear();
                if (end.size() > job->stream_off) {
                    std::string d = end.substr(job->stream_off);
                    if (!job->stream_started && !job->streamed.empty() && job->stream_off == 0)
                        d = " " + d;
                    if (job->stream_off == 0 && !d.empty()) job->stream_started = true;
                    job->stream_off = end.size();
                    delta(d);
                }
            }
            if (!closed) break;
            ++job->stream_clip;
            job->stream_off = 0;
            job->stream_started = false;
        }
        if (job->running && self->all_done(job->gens)) {
            std::string failure;
            if (!finish_wave(&failure)) {
                out += sse_error(internal(failure));
                job->done_sent = true;
                release_job();
                return true;
            }
            if (job->next < job->cuts.size()) {
                ApiError err;
                if (!start_wave(&err)) {
                    out += sse_error(err);
                    job->done_sent = true;
                    release_job();
                    return true;
                }
            }
            return true;
        }
        if (!job->running && job->stream_clip >= job->cuts.size()) {
            release_job();
            json o;
            o["type"] = "transcript.text.done";
            o["text"] = joined();
            o["usage"] = usage();
            out += sse_event(json_dump(o));
            job->done_sent = true;
            self->metrics_.observe_http(200);
            return true;
        }
        if (!progress && out.empty() && job->running)
            for (auto& g : job->gens) if (!g.closed) { g.sink->wait(); break; }
        return true;
    };
    res.on_close = [self, job, release_job](bool) {
        if (job->running) {
            self->cancel_all(job->gens, true);
            self->record(job->r, job->gens, Finish::Cancelled);
            self->retire(job->gens);
            job->running = false;
        }
        release_job();
    };
    return res;
}

}  /* namespace server */
}  /* namespace rad */
