/* realtime.cpp -- OpenAI's realtime transcription session, over a WebSocket at /v1/realtime.
 *
 * The upload route (speech.cpp) transcribes a recording once it has ended. This one is sent the
 * audio as it is captured and answers with the words as they settle, as OpenAI's live
 * transcription does: deltas while the audio arrives, and a completed transcript for each turn
 * once it is committed.
 *
 * Protocol: OpenAI's realtime one in its transcription form. The client sends `session.update`
 * (session.type "transcription", or the beta's `transcription_session.update`) and
 * `input_audio_buffer.append` / `.commit` / `.clear`. The server sends `session.created` /
 * `.updated`, `input_audio_buffer.committed` / `.cleared` / `.speech_started` /
 * `.speech_stopped`, `conversation.item.input_audio_transcription.delta` / `.completed` /
 * `.failed`, and `error`. Audio is 16-bit little-endian mono PCM, base64 in an append or the same
 * bytes as a binary frame, at the session's rate (24 kHz unless it names another), resampled to
 * the model's.
 *
 * Settling: the encoder takes a clip whole, so a turn in progress is heard again from its start
 * each time enough new audio has come in, one short pass whose prompt is the turn so far. Each
 * pass is held to what the earlier ones settled (the settled reply tokens follow the prompt as the
 * start of the reply, so the pass decodes only what comes after them) and settles all of its reply
 * but the last few tokens, which the end of the audio can still change: a word cut off mid-sound,
 * or the full stop the model puts after what it has heard so far. Settled text is sent as a delta
 * and never taken back; the rest is sent as `radiance.input_audio_transcription.tentative`, which
 * each pass replaces. OpenAI's protocol has no such event, and a client that does not know it
 * ignores it.
 *
 * A turn ends at the client's commit, or at a pause with server_vad turn detection, and is then
 * heard once more, whole, for its completed transcript. A turn that reaches the length of one
 * encoder pass is committed at its quietest point in the last five seconds, and the rest goes on
 * as the next turn.
 *
 * The model's side comes from its plugin (RadAudioFormat) through the helpers the upload route
 * uses; this file knows OpenAI's side.
 *
 * Threads: the connection's thread reads messages and queues them. The session's thread does
 * everything else (the audio, the passes, every event sent), so the session's state has one owner
 * and its events go out in the order they happen.
 */
#include "server.h"
#include "server_priv.h"
#include "oai.h"

#include "mm/processor.h"
#include "mm/resample.h"
#include "rad_internal.h"

#include "util/json.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <mutex>
#include <thread>

namespace rad {
namespace server {

namespace {

using ojson = Json;
using Clock = std::chrono::steady_clock;

/* A pass once this much audio has come in since the turn's last one began. */
constexpr double kStepSeconds = 0.5;
/* No pass hears less than this. */
constexpr double kFirstPassSeconds = 0.5;
/* Nothing settles before the turn has this much audio: the first words of a turn heard on their
 * own are where the model guesses most. */
constexpr double kSettleAfterSeconds = 1.0;
/* The most one pass may decode. Thirty seconds of the fastest speech is a few hundred tokens. */
constexpr int32_t kMaxTokensPerPass = 512;
/* The client messages a session may have queued and not yet taken in. */
constexpr size_t kMaxQueuedBytes = 16u << 20;
/* Committed turns not yet transcribed: past either bound the session has fallen behind its
 * audio for good, and is closed with an error rather than left to grow. */
constexpr double kMaxBehindSeconds = 120.0;
constexpr size_t kMaxBehindTurns = 64;
/* A client commit needs at least this much audio, as OpenAI's does. */
constexpr int64_t kMinCommitMs = 100;
/* Pause detection: a frame, the speech that starts a turn, and the level nothing under counts as
 * speech however quiet the room. */
constexpr int32_t kFrameMs = 20;
constexpr int kStartFrames = 3;
constexpr double kSpeechFloorDb = -55.0;

/* The reply tokens left unsettled, by OpenAI's `delay`: how far behind the speech the settled
 * text may run. A settled word is never taken back, even when the rest of the sentence would have
 * changed it, so the default leaves a few words open: on LibriSpeech, four tokens settled
 * "The English forwarded" once as "The English voted", and six did not. The tentative text shows
 * the open words meanwhile. */
int unsettled_for(const std::string& delay) {
    if (delay == "minimal") return 3;
    if (delay == "low") return 4;
    if (delay == "high") return 8;
    if (delay == "xhigh") return 10;
    return 6;   /* medium */
}

std::string rtrim(std::string s) {
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
    return s;
}

/* A string member, or `def` when it is absent or not a string. */
std::string str_of(const ojson& o, const char* k, const std::string& def) {
    return o.contains(k) && o[k].is_string() ? o[k].get<std::string>() : def;
}

ojson event(const char* type) {
    ojson o;
    o["type"] = type;
    o["event_id"] = random_id("event_");
    return o;
}

}  /* namespace */

/* One turn: its audio at the model's rate, and what of its transcript has gone out. */
struct RtTurn {
    std::string id;
    std::vector<float> audio;
    int64_t start_ms = 0;               /* on the session's clock */
    bool    committed = false;
    bool    dropped = false;            /* cleared before it was committed */
    Clock::time_point committed_at;
    std::vector<int32_t> settled;       /* reply tokens every later pass is held to */
    std::string sent;                   /* transcript text sent as deltas */
    std::string tentative;              /* the tentative text last sent */
    size_t  heard = 0;                  /* audio.size() when its last pass began */
    int64_t n_prompt = 0, n_gen = 0;
    int     passes = 0;
};

struct Server::RealtimeSession {
    std::string id;
    std::string peer;

    /* The connection's thread to the session's: messages, and whether the peer is gone. */
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::pair<std::string, bool>> inbox;   /* (message, binary) */
    size_t inbox_bytes = 0;
    bool   peer_gone = false;
    bool   overflow = false;

    /* Everything below is the session thread's alone. */
    std::string model;
    bool        beta = false;            /* answers in the beta's event names */
    int32_t     rate = 24000;            /* the client's audio */
    std::string language_code;           /* as the client named it, "" for none */
    std::string language;                /* the model's name for it, "" to detect */
    std::string prompt;
    std::vector<std::string> keywords;
    std::string delay;
    int         unsettled = 6;
    bool        vad = false;
    double      threshold = 0.5;
    int         prefix_ms = 300;
    int         silence_ms = 500;

    int32_t model_rate = 16000;
    std::unique_ptr<mm::Resampler> rs;
    int64_t clock = 0;                   /* samples at the model's rate heard so far */
    std::vector<float> frame;            /* pause detection's frame in progress */
    std::vector<float> recent;           /* the audio before speech, for the turn's lead-in */
    double floor_db = NAN;
    bool   speaking = false;
    int    run = 0, quiet = 0;

    std::shared_ptr<RtTurn> open;        /* the turn the audio goes into */
    std::deque<std::shared_ptr<RtTurn>> turns;   /* committed, not yet completed */
    std::string last_item;

    struct Pass {
        std::shared_ptr<RtTurn> turn;
        bool   final = false;
        size_t heard = 0;
        OaiRequest r;
        std::vector<Generation> gens;
        bool   running = false;
    } pass;

    int64_t n_turns = 0;

    int64_t ms(int64_t samples) const { return samples * 1000 / model_rate; }
    std::string context() const {
        std::string c = prompt;
        if (!keywords.empty()) {
            if (!c.empty()) c += "\n";
            for (size_t i = 0; i < keywords.size(); ++i) c += (i ? ", " : "") + keywords[i];
        }
        return c;
    }
};

/* ------------------------------------------------------------------ the connection's thread */

void Server::handle_realtime(const HttpRequest& req, WsConn& ws) {
    auto refuse = [&](const char* type, const char* code, const std::string& why, int close_code) {
        ojson o = event("error");
        ojson e;
        e["type"] = type;
        e["code"] = code;
        e["message"] = why;
        e["param"] = nullptr;
        e["event_id"] = nullptr;
        o["error"] = std::move(e);
        ws.send(json_dump(o));
        RAD_INFO("realtime refused from %s: %s", req.peer.c_str(), why.c_str());
        ws.close(close_code, code);
    };
    if (stopping_.load()) return refuse("server_error", "shutting_down",
                                        "the server is shutting down", 1001);
    if (!deps_.sched || !deps_.tok || !deps_.speech || !deps_.speech->transcribes() || !deps_.mm ||
        !deps_.mm->accepts(MediaKind::Audio) || deps_.mm->audio_rate() <= 0)
        return refuse("invalid_request_error", "model_not_supported",
                      "this model does not transcribe: its architecture plugin states no "
                      "transcription format (rad_arch_audio_format), or it was started without "
                      "its audio encoder", 1008);
    /* An open turn is cut at the encoder's clip length; without one it would grow for as long as
     * the client appends audio. */
    if (deps_.mm->audio_max_samples() <= 0)
        return refuse("invalid_request_error", "model_not_supported",
                      "this model's audio encoder states no clip length, which a realtime turn "
                      "is cut at", 1008);
    /* A session holds a connection's thread and one of its own for its whole life, and runs one
     * pass at a time: more sessions than the engine has sequences would only queue. */
    const int64_t cap = std::max<int64_t>(1, opt_.max_seqs);
    if (realtime_sessions_.fetch_add(1) >= cap) {
        realtime_sessions_.fetch_sub(1);
        return refuse("server_error", "session_limit",
                      fmt("this server holds %lld realtime sessions at once; try again later",
                          (long long)cap), 1013);
    }
    struct Count {
        std::atomic<int64_t>& n;
        ~Count() { n.fetch_sub(1); }
    } count{ realtime_sessions_ };
    /* One admission place for the session's whole life, which its passes run in, so a session
     * counts against the queue (and the HTTP workers sized from it) like any request. */
    if (admit(1, req.peer))
        return refuse("server_error", "rate_limit_exceeded",
                      "the admission queue is full; try again later", 1013);
    struct Place {
        Server& srv;
        ~Place() { srv.release(1); }
    } place{ *this };

    RealtimeSession s;
    s.id = random_id("sess_");
    s.peer = req.peer;
    s.model = opt_.model_id;
    s.model_rate = deps_.mm->audio_rate();
    std::thread worker([this, &s, &ws] { realtime_run(s, ws); });
    /* The session's thread ends once the peer is gone, however this one leaves. */
    struct Join {
        RealtimeSession& s;
        std::thread& t;
        ~Join() {
            {
                std::lock_guard<std::mutex> lk(s.m);
                s.peer_gone = true;
            }
            s.cv.notify_one();
            if (t.joinable()) t.join();
        }
    } join{ s, worker };

    std::string msg;
    bool binary = false;
    while (ws.read(&msg, &binary)) {
        std::lock_guard<std::mutex> lk(s.m);
        if (s.inbox_bytes + msg.size() > kMaxQueuedBytes) {
            s.overflow = true;
            s.cv.notify_one();
            break;
        }
        s.inbox_bytes += msg.size();
        s.inbox.emplace_back(std::move(msg), binary);
        s.cv.notify_one();
    }
}

/* ------------------------------------------------------------------ the session's thread */

void Server::realtime_run(RealtimeSession& s, WsConn& ws) {
    bool server_closed = true;
    try {
        server_closed = realtime_loop(s, ws);
    } catch (const std::exception& e) {
        /* e.what() is server-internal text that may quote client frames: sanitized to the log,
         * and the socket's client gets the generic sentence, not the reason. */
        RAD_ERR("realtime %s failed: %s", s.id.c_str(), log_sanitize(e.what()).c_str());
        ojson o = event("error");
        ojson err;
        err["type"] = "server_error";
        err["code"] = "internal_error";
        err["message"] = "the session failed; the reason is in the server log";
        err["param"] = nullptr;
        err["event_id"] = nullptr;
        o["error"] = std::move(err);
        ws.send(json_dump(o));
        ws.close(1011, "internal error");
    }
    if (s.pass.running) {
        cancel_all(s.pass.gens, !server_closed);
        record(s.pass.r, s.pass.gens, Finish::Cancelled, /*quiet=*/true);
        retire(s.pass.gens);
        s.pass.running = false;
    }
    RAD_INFO("realtime %s closed: %lld turns, %.1fs of audio", s.id.c_str(),
             (long long)s.n_turns, (double)s.clock / s.model_rate);
}

bool Server::realtime_loop(RealtimeSession& s, WsConn& ws) {
    const SpeechFormat& f = *deps_.speech;
    const int64_t max_clip = deps_.mm->audio_max_samples();
    const int64_t frame_n = (int64_t)s.model_rate * kFrameMs / 1000;
    /* Where a turn that reaches the length of a pass is cut: its last five seconds, or half of it
     * if that is less. Nothing in this stretch settles, so whatever has gone out as a delta was
     * heard before the cut, wherever in the stretch the cut lands. */
    const int64_t cut_zone =
        max_clip > 0 ? std::min<int64_t>(max_clip / 2, 5 * (int64_t)s.model_rate) : 0;
    s.rs.reset(new mm::Resampler(s.rate, s.model_rate));

    auto send = [&](const ojson& o) { ws.send(json_dump(o)); };
    auto error = [&](const char* type, const std::string& code, const std::string& why,
                     const std::string& param, const ojson& client_event) {
        ojson o = event("error");
        ojson e;
        e["type"] = type;
        e["code"] = code;
        e["message"] = why;
        e["param"] = param.empty() ? ojson(nullptr) : ojson(param);
        e["event_id"] = client_event.is_string() ? client_event : ojson(nullptr);
        o["error"] = std::move(e);
        send(o);
    };

    /* ---- the session as the client sees it */
    auto session_json = [&]() {
        ojson tr;
        tr["model"] = s.model;
        tr["language"] = s.language_code.empty() ? ojson(nullptr) : ojson(s.language_code);
        tr["prompt"] = s.prompt;
        ojson td = nullptr;
        if (s.vad) {
            td = ojson::object();
            td["type"] = "server_vad";
            td["threshold"] = s.threshold;
            td["prefix_padding_ms"] = s.prefix_ms;
            td["silence_duration_ms"] = s.silence_ms;
        }
        ojson o;
        if (s.beta) {
            o["id"] = s.id;
            o["object"] = "realtime.transcription_session";
            o["input_audio_format"] = "pcm16";
            o["input_audio_transcription"] = std::move(tr);
            o["turn_detection"] = std::move(td);
            o["input_audio_noise_reduction"] = nullptr;
            return o;
        }
        if (!s.keywords.empty()) tr["keywords"] = s.keywords;
        if (!s.delay.empty()) tr["delay"] = s.delay;
        ojson fmt_o;
        fmt_o["type"] = "audio/pcm";
        fmt_o["rate"] = s.rate;
        ojson in;
        in["format"] = std::move(fmt_o);
        in["transcription"] = std::move(tr);
        in["turn_detection"] = std::move(td);
        in["noise_reduction"] = nullptr;
        o["type"] = "transcription";
        o["object"] = "realtime.transcription_session";
        o["id"] = s.id;
        o["audio"]["input"] = std::move(in);
        return o;
    };

    /* ---- turns */
    auto new_turn = [&](int64_t start_ms) {
        auto t = std::make_shared<RtTurn>();
        t->id = random_id("item_");
        t->start_ms = start_ms;
        return t;
    };
    auto commit_open = [&]() {
        std::shared_ptr<RtTurn> t = std::move(s.open);
        s.open.reset();
        t->committed = true;
        t->committed_at = Clock::now();
        ojson o = event("input_audio_buffer.committed");
        o["previous_item_id"] = s.last_item.empty() ? ojson(nullptr) : ojson(s.last_item);
        o["item_id"] = t->id;
        send(o);
        s.last_item = t->id;
        s.turns.push_back(std::move(t));
        ++s.n_turns;
    };
    /* A turn as long as one pass takes is committed at its quietest point in the last five
     * seconds of that pass; what follows that point goes on as the next turn. */
    auto bound_open = [&]() {
        while (s.open && max_clip > 0 && (int64_t)s.open->audio.size() >= max_clip) {
            std::vector<float>& a = s.open->audio;
            const int64_t cut = speech_quiet_point(a, max_clip - cut_zone, max_clip, s.model_rate);
            std::vector<float> rest(a.begin() + cut, a.end());
            a.resize((size_t)cut);
            const int64_t start = s.open->start_ms + s.ms(cut);
            commit_open();
            s.open = new_turn(start);
            s.open->audio = std::move(rest);
            if (s.vad) {
                ojson o = event("input_audio_buffer.speech_started");
                o["audio_start_ms"] = start;
                o["item_id"] = s.open->id;
                send(o);
            }
        }
    };

    /* ---- pause detection: a frame's level against the room's, tracked as its quietest */
    auto vad_frame = [&](const float* x, int64_t n) {
        double e = 0.0;
        for (int64_t i = 0; i < n; ++i) e += (double)x[i] * x[i];
        const double db = 10.0 * std::log10(e / (double)n + 1e-12);
        if (std::isnan(s.floor_db) || db < s.floor_db) s.floor_db = db;
        else s.floor_db += s.speaking ? 0.002 : 0.02;   /* 0.1 dB/s in speech, 1 dB/s out of it */
        const double margin = 6.0 + 18.0 * s.threshold;
        const bool voiced = db > std::max(s.floor_db + margin, kSpeechFloorDb);
        s.clock += n;
        if (!s.speaking) {
            s.recent.insert(s.recent.end(), x, x + n);
            s.run = voiced ? s.run + 1 : 0;
            const size_t lead = (size_t)((int64_t)s.model_rate * s.prefix_ms / 1000 +
                                         kStartFrames * n);
            if (s.run >= kStartFrames) {
                const size_t take = std::min(lead, s.recent.size());
                s.open = new_turn(s.ms(s.clock) - s.ms((int64_t)take));
                s.open->audio.assign(s.recent.end() - (std::ptrdiff_t)take, s.recent.end());
                s.recent.clear();
                s.speaking = true;
                s.quiet = 0;
                ojson o = event("input_audio_buffer.speech_started");
                o["audio_start_ms"] = s.open->start_ms;
                o["item_id"] = s.open->id;
                send(o);
            } else if (s.recent.size() > lead + (size_t)s.model_rate) {
                s.recent.erase(s.recent.begin(), s.recent.end() - (std::ptrdiff_t)lead);
            }
            return;
        }
        s.open->audio.insert(s.open->audio.end(), x, x + n);
        s.quiet = voiced ? 0 : s.quiet + 1;
        if ((int64_t)s.quiet * kFrameMs >= s.silence_ms) {
            s.speaking = false;
            s.run = 0;
            ojson o = event("input_audio_buffer.speech_stopped");
            o["audio_end_ms"] = s.ms(s.clock);
            o["item_id"] = s.open->id;
            send(o);
            commit_open();
            return;
        }
        bound_open();
    };

    /* Audio at the model's rate, into the open turn or through pause detection. */
    auto take = [&](const std::vector<float>& x) {
        if (x.empty()) return;
        if (!s.vad) {
            if (!s.open) s.open = new_turn(s.ms(s.clock));
            s.open->audio.insert(s.open->audio.end(), x.begin(), x.end());
            s.clock += (int64_t)x.size();
            bound_open();
            return;
        }
        size_t i = 0;
        while (i < x.size()) {
            const size_t k = std::min(x.size() - i, (size_t)frame_n - s.frame.size());
            s.frame.insert(s.frame.end(), x.begin() + (std::ptrdiff_t)i,
                           x.begin() + (std::ptrdiff_t)(i + k));
            i += k;
            if ((int64_t)s.frame.size() == frame_n) {
                vad_frame(s.frame.data(), frame_n);
                s.frame.clear();
            }
        }
    };
    auto append_pcm = [&](const uint8_t* p, size_t n, const ojson& eid) {
        if (n % 2) {
            error("invalid_request_error", "invalid_value",
                  "the audio is 16-bit PCM, two bytes a sample; this chunk has an odd length",
                  "audio", eid);
            return;
        }
        std::vector<float> in(n / 2), out;
        for (size_t i = 0; i < n / 2; ++i)
            in[i] = (float)(int16_t)(uint16_t)(p[2 * i] | (p[2 * i + 1] << 8)) / 32768.0f;
        s.rs->push(in.data(), in.size(), &out);
        take(out);
    };
    /* The pause detector's half-heard frame, into the open turn when there is one. */
    auto flush_frame = [&]() {
        if (s.speaking && s.open) s.open->audio.insert(s.open->audio.end(), s.frame.begin(),
                                                      s.frame.end());
        s.clock += (int64_t)s.frame.size();
        s.frame.clear();
    };

    /* ---- session.update: checked whole, then applied whole */
    auto update = [&](const ojson& j, bool beta) {
        const ojson eid = j.contains("event_id") ? j["event_id"] : ojson(nullptr);
        const char* ack = beta ? "transcription_session.updated" : "session.updated";
        if (!j.contains("session") || !j["session"].is_object())
            return error("invalid_request_error", "missing_required_parameter",
                         "the update carries no session object", "session", eid);
        const ojson& se = j["session"];
        int32_t rate = s.rate;
        std::string code = s.language_code, name = s.language, prompt = s.prompt, delay = s.delay;
        std::vector<std::string> keywords = s.keywords;
        bool vad = s.vad;
        double threshold = s.threshold;
        int prefix = s.prefix_ms, silence = s.silence_ms;
        std::string model = s.model;
        auto bad = [&](const std::string& param, const std::string& why) {
            error("invalid_request_error", "invalid_value", why, param, eid);
        };
        const ojson* tr = nullptr;
        const ojson* td = nullptr;
        bool has_td = false;
        if (!beta) {
            if (se.contains("type") && se["type"] != "transcription")
                return bad("session.type", "this server's realtime sessions transcribe: "
                                           "session.type is \"transcription\"");
            if (se.contains("audio") && se["audio"].is_object() && se["audio"].contains("input") &&
                se["audio"]["input"].is_object()) {
                const ojson& in = se["audio"]["input"];
                if (in.contains("format") && in["format"].is_object()) {
                    const ojson& fo = in["format"];
                    if (str_of(fo, "type", "audio/pcm") != "audio/pcm")
                        return bad("session.audio.input.format.type",
                                   "the audio is \"audio/pcm\": 16-bit little-endian mono");
                    if (fo.contains("rate")) {
                        if (!fo["rate"].is_number_integer() ||
                            !mm::Resampler::takes(fo["rate"].get<int32_t>()))
                            return bad("session.audio.input.format.rate",
                                       "the rate is one of " + mm::Resampler::rates() + " Hz");
                        rate = fo["rate"].get<int32_t>();
                    }
                }
                if (in.contains("transcription") && in["transcription"].is_object())
                    tr = &in["transcription"];
                if (in.contains("turn_detection")) { td = &in["turn_detection"]; has_td = true; }
            }
        } else {
            if (se.contains("input_audio_format") && se["input_audio_format"] != "pcm16")
                return bad("session.input_audio_format",
                           "the audio is \"pcm16\": 16-bit little-endian mono at 24 kHz");
            if (se.contains("input_audio_format")) rate = 24000;
            if (se.contains("input_audio_transcription") &&
                se["input_audio_transcription"].is_object())
                tr = &se["input_audio_transcription"];
            if (se.contains("turn_detection")) { td = &se["turn_detection"]; has_td = true; }
        }
        if (tr) {
            const ojson& t = *tr;
            model = str_of(t, "model", model);
            std::string want = str_of(t, "language", "");
            bool detect = t.contains("language") && t["language"].is_null();
            /* The live models' `languages`: one is a forced language; several, detection. */
            if (t.contains("languages") && t["languages"].is_array()) {
                if (t["languages"].size() == 1 && t["languages"][0].is_string())
                    want = t["languages"][0].get<std::string>();
                else
                    detect = true;
            }
            if (detect && want.empty()) {
                code.clear();
                name.clear();
            }
            if (!want.empty()) {
                const std::string n = f.language_name(want);
                if (n.empty()) {
                    std::string known;
                    for (const auto& l : f.languages)
                        known += (known.empty() ? "" : ", ") + l.first;
                    return bad("session.audio.input.transcription.language",
                               "this model names no language '" + want + "'; it knows " + known);
                }
                code = want;
                name = n;
            }
            prompt = str_of(t, "prompt", prompt);
            if (t.contains("keywords") && t["keywords"].is_array()) {
                keywords.clear();
                for (const auto& k : t["keywords"])
                    if (k.is_string()) keywords.push_back(k.get<std::string>());
            }
            if (t.contains("delay") && t["delay"].is_string()) {
                delay = str_of(t, "delay", "");
                if (delay != "minimal" && delay != "low" && delay != "medium" && delay != "high" &&
                    delay != "xhigh")
                    return bad("session.audio.input.transcription.delay",
                               "delay is minimal, low, medium, high or xhigh");
            }
        }
        if (has_td) {
            const ojson& t = *td;
            if (t.is_null()) {
                vad = false;
            } else if (t.is_object()) {
                const std::string type = str_of(t, "type", "server_vad");
                if (type != "server_vad")
                    return bad("turn_detection.type",
                               "turn detection is server_vad, or null for turns the client "
                               "commits; not '" + type + "'");
                vad = true;
                if (t.contains("threshold") && t["threshold"].is_number()) {
                    threshold = t["threshold"].get<double>();
                    if (!(threshold >= 0.0 && threshold <= 1.0))
                        return bad("turn_detection.threshold", "threshold is between 0 and 1");
                }
                if (t.contains("prefix_padding_ms") && t["prefix_padding_ms"].is_number_integer())
                    prefix = std::clamp(t["prefix_padding_ms"].get<int>(), 0, 2000);
                if (t.contains("silence_duration_ms") &&
                    t["silence_duration_ms"].is_number_integer())
                    silence = std::clamp(t["silence_duration_ms"].get<int>(), 100, 5000);
            } else {
                return bad("turn_detection", "turn detection is an object or null");
            }
        }

        /* Applied. Audio already in at the old rate comes out before the new rate's. */
        if (rate != s.rate) {
            std::vector<float> tail;
            s.rs->flush(&tail);
            take(tail);
            s.rate = rate;
            s.rs.reset(new mm::Resampler(s.rate, s.model_rate));
        }
        if (vad != s.vad) {
            if (s.vad) flush_frame();            /* off: the open turn waits for a commit */
            s.speaking = vad && s.open && !s.open->audio.empty();
            s.run = s.quiet = 0;
            s.recent.clear();
            s.vad = vad;
        }
        s.beta = beta;
        s.model = model;
        s.language_code = code;
        s.language = name;
        s.prompt = prompt;
        s.keywords = keywords;
        s.delay = delay;
        s.unsettled = unsettled_for(delay);
        s.threshold = threshold;
        s.prefix_ms = prefix;
        s.silence_ms = silence;
        ojson o = event(ack);
        o["session"] = session_json();
        send(o);
    };

    auto handle = [&](const std::string& msg, bool binary) {
        if (binary) return append_pcm((const uint8_t*)msg.data(), msg.size(), nullptr);
        /* Bounded like a request body: 2 MiB of brackets is a legal WebSocket frame, and the
         * recursive teardown of a deep tree would take the whole process down with it. */
        bool over = false, deep = false;
        const ojson j = parse_json_bounded(msg, 0, &over, &deep);
        if (j.is_discarded() || over || deep || !j.is_object() || !j.contains("type") ||
            !j["type"].is_string())
            return error("invalid_request_error", "invalid_json",
                         "a client event is a JSON object with a string `type`", "", nullptr);
        const std::string type = j["type"].get<std::string>();
        const ojson eid = j.contains("event_id") ? j["event_id"] : ojson(nullptr);
        if (type == "input_audio_buffer.append") {
            std::vector<uint8_t> pcm;
            if (!j.contains("audio") || !j["audio"].is_string() ||
                !base64_decode(j["audio"].get<std::string>(), &pcm))
                return error("invalid_request_error", "invalid_value",
                             "`audio` is base64 of 16-bit little-endian PCM", "audio", eid);
            return append_pcm(pcm.data(), pcm.size(), eid);
        }
        if (type == "input_audio_buffer.commit") {
            /* The turn ends here: what the resampler still holds back belongs to it. */
            std::vector<float> tail;
            s.rs->flush(&tail);
            take(tail);
            if (s.vad) flush_frame();
            if (!s.open || s.ms((int64_t)s.open->audio.size()) < kMinCommitMs)
                return error("invalid_request_error", "input_audio_buffer_commit_empty",
                             fmt("the buffer holds less than %lld ms of audio",
                                 (long long)kMinCommitMs), "", eid);
            commit_open();
            s.speaking = false;
            s.run = s.quiet = 0;
            return;
        }
        if (type == "input_audio_buffer.clear") {
            if (s.open) s.open->dropped = true;
            s.open.reset();
            s.frame.clear();
            s.recent.clear();
            s.speaking = false;
            s.run = s.quiet = 0;
            send(event("input_audio_buffer.cleared"));
            return;
        }
        if (type == "session.update") return update(j, false);
        if (type == "transcription_session.update") return update(j, true);
        error("invalid_request_error", "unknown_event",
              "this session takes session.update, input_audio_buffer.append, "
              "input_audio_buffer.commit and input_audio_buffer.clear; not '" + type + "'",
              "type", eid);
    };

    /* ---- passes */
    auto detok = [&](const std::vector<int32_t>& ids, bool flush) {
        std::unique_ptr<IDetokenizer> d = deps_.tok->detokenizer({});
        std::string t;
        for (int32_t id : ids) t += d->push(id);
        if (flush) t += d->flush();
        return t;
    };
    auto usage = [](const RtTurn& t) {
        ojson u;
        u["type"] = "tokens";
        u["input_tokens"] = t.n_prompt;
        u["output_tokens"] = t.n_gen;
        u["total_tokens"] = t.n_prompt + t.n_gen;
        return u;
    };
    auto delta = [&](RtTurn& t, const std::string& d) {
        if (d.empty()) return;
        ojson o = event("conversation.item.input_audio_transcription.delta");
        o["item_id"] = t.id;
        o["content_index"] = 0;
        o["delta"] = d;
        send(o);
        t.sent += d;
    };
    auto failed = [&](RtTurn& t, const std::string& why) {
        ojson o = event("conversation.item.input_audio_transcription.failed");
        o["item_id"] = t.id;
        o["content_index"] = 0;
        ojson e;
        e["type"] = "server_error";
        e["code"] = nullptr;
        e["message"] = why;
        e["param"] = nullptr;
        o["error"] = std::move(e);
        send(o);
        RAD_WARN("realtime %s %s failed: %s", s.id.c_str(), t.id.c_str(), why.c_str());
    };
    auto drop_front = [&](const std::shared_ptr<RtTurn>& t) {
        if (!s.turns.empty() && s.turns.front() == t) s.turns.pop_front();
    };

    /* What to hear next: the oldest committed turn whole, else the open turn once enough new
     * audio has come in. */
    auto next_work = [&](bool* final) -> std::shared_ptr<RtTurn> {
        if (s.pass.running) return nullptr;
        if (!s.turns.empty()) { *final = true; return s.turns.front(); }
        if (!s.open) return nullptr;
        const size_t n = s.open->audio.size();
        if ((double)n < kFirstPassSeconds * s.model_rate ||
            (double)n < (double)s.open->heard + kStepSeconds * s.model_rate)
            return nullptr;
        *final = false;
        return s.open;
    };

    auto start_pass = [&](const std::shared_ptr<RtTurn>& t, bool final) {
        const size_t n = t->audio.size();
        std::shared_ptr<mm::Item> item;
        std::string why;
        PromptInput p;
        const int st = deps_.mm->prepare_audio(t->audio, &item, &why);
        if (st < 0 || !item ||
            speech_prompt(f, *deps_.tok, s.context(),
                          s.language.empty() ? std::string() : speech_language_text(f, s.language),
                          item, &p) < 0) {
            t->heard = n;
            if (final) {
                failed(*t, st < 0 ? "the audio could not be used: " + why
                                  : "the model's transcription template has no {audio} slot");
                drop_front(t);
            }
            return;
        }
        p.tokens.insert(p.tokens.end(), t->settled.begin(), t->settled.end());
        OaiRequest& r = s.pass.r;
        r = OaiRequest{};
        r.endpoint = Endpoint::Completion;
        r.shape = ApiShape::OpenAI;
        r.model = opt_.model_id;
        r.id = random_id("rt-");
        r.created = (int64_t)std::time(nullptr);
        r.peer = s.peer;
        r.n = 1;
        r.sp = SamplingParams{};
        r.sp.temp = 0.0f;
        r.sp.top_k = 1;
        r.return_tokens = true;
        r.max_tokens = kMaxTokensPerPass;
        if (lim_.max_ctx > 0) {
            if ((int64_t)p.tokens.size() >= lim_.max_ctx) {
                t->heard = n;
                if (final) {
                    failed(*t, "the turn's prompt is longer than the context");
                    drop_front(t);
                }
                return;
            }
            r.max_tokens = (int32_t)std::min<int64_t>(r.max_tokens,
                                                      lim_.max_ctx - (int64_t)p.tokens.size());
        }
        r.prompts.assign(1, std::move(p));
        s.pass.gens.clear();
        ApiError err;
        if (submit_all(r, &s.pass.gens, &err) < 0) {
            retire(s.pass.gens);
            t->heard = n;
            if (final) {
                failed(*t, err.message);
                drop_front(t);
            }
            return;
        }
        s.pass.turn = t;
        s.pass.final = final;
        s.pass.heard = n;
        s.pass.running = true;
        t->heard = n;
        ++t->passes;
    };

    auto finish_pass = [&]() {
        Generation& g = s.pass.gens[0];
        std::shared_ptr<RtTurn> t = s.pass.turn;
        const Finish fin = g.finish;
        const std::string why = fin == Finish::Error ? engine_failure(*g.sink) : std::string();
        t->n_prompt += (int64_t)s.pass.r.prompts[0].tokens.size();
        t->n_gen += g.sink->n_produced();
        std::vector<int32_t> ids = t->settled;
        ids.insert(ids.end(), g.tokens.begin(), g.tokens.end());
        /* The end of the turn, which the decoder renders as nothing, is no part of the reply. */
        while (!ids.empty() && deps_.tok->reply_bytes(ids.back()).empty()) ids.pop_back();
        record(s.pass.r, s.pass.gens, fin, /*quiet=*/true);
        retire(s.pass.gens);
        s.pass.running = false;
        s.pass.turn.reset();
        if (t->dropped) return;
        const bool forced = !s.language.empty();
        if (fin == Finish::Error) {
            if (s.pass.final) {
                failed(*t, why);
                drop_front(t);
            }
            return;
        }
        if (s.pass.final) {
            const std::string text = detok(ids, /*flush=*/true);
            const size_t at = speech_transcript_start(f, text, forced, /*closed=*/true);
            const std::string transcript = rtrim(text.substr(std::min(at, text.size())));
            if (transcript.starts_with(t->sent)) delta(*t, transcript.substr(t->sent.size()));
            ojson o = event("conversation.item.input_audio_transcription.completed");
            o["item_id"] = t->id;
            o["content_index"] = 0;
            o["transcript"] = transcript;
            o["usage"] = usage(*t);
            send(o);
            RAD_INFO("realtime %s %s: %.1fs of audio, %d passes, prompt=%lld gen=%lld, final "
                     "%.2fs after the commit", s.id.c_str(), t->id.c_str(),
                     (double)t->audio.size() / s.model_rate, t->passes, (long long)t->n_prompt,
                     (long long)t->n_gen,
                     std::chrono::duration<double>(Clock::now() - t->committed_at).count());
            drop_front(t);
            return;
        }
        /* An interim pass: settle all but the last few tokens -- none in the turn's first second,
         * none once the turn is where it may be cut (cut_zone), and none of a reply that ran to its
         * bound, which is a model looping. */
        if ((double)s.pass.heard >= kSettleAfterSeconds * s.model_rate && fin != Finish::Length &&
            (max_clip <= 0 || (int64_t)s.pass.heard <= max_clip - cut_zone) &&
            ids.size() > (size_t)s.unsettled)
            t->settled.assign(ids.begin(),
                              ids.begin() + (std::ptrdiff_t)std::max(t->settled.size(),
                                                                     ids.size() - s.unsettled));
        const std::string settled_text = detok(t->settled, /*flush=*/false);
        const size_t a = speech_transcript_start(f, settled_text, forced, /*closed=*/false);
        const std::string settled = a == std::string::npos ? std::string()
                                                           : rtrim(settled_text.substr(a));
        if (settled.size() > t->sent.size() && settled.starts_with(t->sent))
            delta(*t, settled.substr(t->sent.size()));
        const std::string all_text = detok(ids, /*flush=*/false);
        const size_t b = speech_transcript_start(f, all_text, forced, /*closed=*/false);
        const std::string all = b == std::string::npos ? std::string()
                                                       : rtrim(all_text.substr(b));
        const std::string tentative = all.starts_with(t->sent) ? all.substr(t->sent.size())
                                                                : std::string();
        if (tentative != t->tentative) {
            ojson o = event("radiance.input_audio_transcription.tentative");
            o["item_id"] = t->id;
            o["content_index"] = 0;
            o["text"] = tentative;
            send(o);
            t->tentative = tentative;
        }
    };

    /* ---- the loop */
    RAD_INFO("realtime %s open from %s", s.id.c_str(), s.peer.c_str());
    {
        ojson o = event("session.created");
        o["session"] = session_json();
        send(o);
    }
    for (;;) {
        std::deque<std::pair<std::string, bool>> got;
        bool gone = false, overflow = false;
        {
            std::unique_lock<std::mutex> lk(s.m);
            bool final = false;
            if (s.inbox.empty() && !s.peer_gone && !s.overflow && !s.pass.running &&
                !next_work(&final))
                s.cv.wait_for(lk, std::chrono::milliseconds(100));
            got.swap(s.inbox);
            s.inbox_bytes = 0;
            gone = s.peer_gone;
            overflow = s.overflow;
        }
        if (overflow) {
            /* The reader stopped at the message that overflowed; what it queued before is
             * dropped with the session. */
            error("invalid_request_error", "buffer_overflow",
                  fmt("the session fell more than %zu MiB of messages behind",
                      kMaxQueuedBytes >> 20), "", nullptr);
            ws.close(1008, "buffer overflow");
            return true;
        }
        for (const auto& m : got) handle(m.first, m.second);
        if (gone) return false;
        {
            double behind = 0.0;
            for (const auto& t : s.turns) behind += (double)t->audio.size() / s.model_rate;
            if (s.turns.size() > kMaxBehindTurns || behind > kMaxBehindSeconds) {
                error("invalid_request_error", "buffer_overflow",
                      fmt("the session is %zu committed turns (%.0f s of audio) behind its "
                          "transcription; the most is %zu turns or %.0f s",
                          s.turns.size(), behind, kMaxBehindTurns, kMaxBehindSeconds),
                      "", nullptr);
                ws.close(1008, "buffer overflow");
                return true;
            }
        }
        if (stopping_.load()) {
            error("server_error", "shutting_down", "the server is shutting down", "", nullptr);
            ws.close(1001, "closing");
            return true;
        }
        if (s.pass.running) {
            pump(s.pass.gens, s.pass.r);
            if (all_done(s.pass.gens)) {
                finish_pass();
                continue;
            }
            bool idle;
            {
                std::lock_guard<std::mutex> lk(s.m);
                idle = s.inbox.empty() && !s.peer_gone;
            }
            if (idle) s.pass.gens[0].sink->wait(10);
            continue;
        }
        sweep_abandoned();
        bool final = false;
        if (std::shared_ptr<RtTurn> t = next_work(&final)) start_pass(t, final);
    }
}

}  /* namespace server */
}  /* namespace rad */
