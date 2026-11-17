/* speech_stream.cpp -- vLLM-Omni's realtime speech session, over a WebSocket at
 * /v1/audio/speech/stream.
 *
 * The POST route (audio_gen.cpp) speaks a text it is given whole. This one is sent the text as it
 * is produced, a language model's reply token by token or a typist's keystrokes, and speaks each
 * sentence as soon as it is complete: the first audio waits for the first sentence, not the text.
 *
 * Protocol: vLLM-Omni's, in its buffered text mode. The client sends `session.config` first (the
 * POST route's fields but `input`, and `response_format` pcm or wav, `stream_audio`,
 * `split_granularity` none / sentence / clause, `text_input_mode` buffered), then `input.text`
 * {text} pieces, `input.done` to end an utterance, and `session.close`. The server answers each
 * sentence with `audio.start` {utterance_index, sentence_index, sentence_text, format,
 * sample_rate, channels}, its audio as binary messages of 16-bit little-endian PCM (one with
 * `stream_audio` false; several, as the vocoder plays it, with it true; a wav begins with a header
 * that states its length, or none while it streams), and `audio.done` {utterance_index,
 * sentence_index, total_bytes, error}; once an utterance's last sentence is spoken,
 * `session.done` {utterance_index, total_sentences}. The session stays open for the next
 * utterance until `session.close` or the socket's close. A refusal or failure is `error`
 * {message, param}; the session goes on.
 *
 * A sentence is one request, built from the session's fields as the POST route builds one
 * (Server::speech_request), and the sentences of a session are spoken one at a time in order.
 *
 * Threads: the connection's thread reads messages and queues them. The session's thread does
 * everything else, so the session's state has one owner and its messages go out in the order they
 * happen.
 */
#include "server.h"
#include "server_priv.h"
#include "oai.h"
#include "text_split.h"

#include "gen/audio.h"
#include "rad_internal.h"

#include "util/json.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <thread>

namespace rad {
namespace server {

namespace {

using ojson = Json;

/* The client messages a session may have queued and not yet taken in. */
constexpr size_t kMaxQueuedBytes = 16u << 20;
/* An utterance's text not yet spoken: a client that never ends a sentence is refused past it. */
constexpr size_t kMaxPendingBytes = 128u << 10;

ojson event(const char* type) {
    ojson o;
    o["type"] = type;
    return o;
}

/* The fields that shape the session rather than each sentence's request. */
bool session_field(const std::string& k) {
    return k == "type" || k == "stream_audio" || k == "split_granularity" ||
           k == "text_input_mode" || k == "response_format";
}

}  /* namespace */

struct Server::SpeechSession {
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
    bool      configured = false;
    json      fields = json::object();   /* every sentence's request, but its input */
    bool      stream_audio = false;
    bool      wav = false;
    TextSplit split = TextSplit::Sentence;

    std::string pending;                 /* the utterance's text not yet a sentence */
    std::deque<std::string> sentences;   /* complete, not yet spoken */
    bool    done_asked = false;          /* input.done: the utterance ends once spoken */
    int64_t utterance = 0;
    int64_t sentence = 0;                /* the utterance's next sentence index */
    int64_t spoken = 0;                  /* sentences over the session's life */

    /* The sentence being spoken. */
    struct Current {
        bool    active = false;
        std::shared_ptr<gen::AudioTicket> ticket;
        int64_t index = 0;
        std::map<int32_t, std::vector<float>> held;   /* chunks ahead of the next one */
        int32_t next = 0;
        std::string pcm;                 /* the whole sentence, when it is sent at once */
        int64_t bytes = 0;               /* binary bytes sent for it */
    } cur;
};

/* ------------------------------------------------------------------ the connection's thread */

void Server::handle_speech_stream(const HttpRequest& req, WsConn& ws) {
    auto refuse = [&](const std::string& why, int close_code, const char* reason) {
        ojson o = event("error");
        o["message"] = why;
        ws.send(json_dump(o));
        RAD_INFO("speech stream refused from %s: %s", req.peer.c_str(), why.c_str());
        ws.close(close_code, reason);
    };
    if (stopping_.load()) return refuse("the server is shutting down", 1001, "shutting down");
    if (!deps_.audio || !deps_.tok)
        return refuse("/v1/audio/speech/stream requires an audio generation model; this one "
                      "makes none", 1008, "model not supported");
    /* A session speaks one sentence at a time: more sessions than the generator has lanes would
     * only queue. */
    const int64_t cap = std::max<int64_t>(1, deps_.audio->info().max_lanes);
    if (speech_sessions_.fetch_add(1) >= cap) {
        speech_sessions_.fetch_sub(1);
        return refuse(fmt("this server holds %lld speech sessions at once; try again later",
                          (long long)cap), 1013, "session limit");
    }
    struct Count {
        std::atomic<int64_t>& n;
        ~Count() { n.fetch_sub(1); }
    } count{ speech_sessions_ };
    /* One admission place for the session's whole life, as the realtime transcription session
     * holds one. */
    if (admit(1, req.peer))
        return refuse("the admission queue is full; try again later", 1013, "rate limited");
    struct Place {
        Server& srv;
        ~Place() { srv.release(1); }
    } place{ *this };

    SpeechSession s;
    s.id = random_id("speech_");
    s.peer = req.peer;
    std::thread worker([this, &s, &ws] { speech_stream_run(s, ws); });
    /* The session's thread ends once the peer is gone, however this one leaves. */
    struct Join {
        SpeechSession& s;
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

void Server::speech_stream_run(SpeechSession& s, WsConn& ws) {
    try {
        speech_stream_loop(s, ws);
    } catch (const std::exception& e) {
        /* e.what() is server-internal text that may quote client frames: sanitized to the log,
         * and the socket's client gets the generic sentence, not the reason. */
        RAD_ERR("speech stream %s failed: %s", s.id.c_str(), log_sanitize(e.what()).c_str());
        ojson o = event("error");
        o["message"] = "the session failed; the reason is in the server log";
        ws.send(json_dump(o));
        ws.close(1011, "internal error");
    }
    if (s.cur.active) s.cur.ticket->cancel.store(true);
    RAD_INFO("speech stream %s closed: %lld utterances, %lld sentences", s.id.c_str(),
             (long long)s.utterance, (long long)s.spoken);
}

bool Server::speech_stream_loop(SpeechSession& s, WsConn& ws) {
    const AudioGenInfo& info = deps_.audio->info();
    auto send = [&](const ojson& o) { ws.send(json_dump(o)); };
    auto error = [&](const std::string& why, const std::string& param = std::string()) {
        ojson o = event("error");
        o["message"] = why;
        o["param"] = param.empty() ? ojson(nullptr) : ojson(param);
        send(o);
    };
    auto binary = [&](const std::string& bytes) {
        if (bytes.empty()) return;
        ws.send_binary(bytes);
        s.cur.bytes += (int64_t)bytes.size();
    };

    /* ---- session.config: checked whole, then applied whole, between utterances only */
    auto configure = [&](const json& j) {
        if (s.cur.active || !s.sentences.empty() || !s.pending.empty() || s.done_asked)
            return error("session.config applies between utterances; this one is still being "
                         "spoken");
        static const char* kFields[] = {
            "type", "model", "instructions", "voice", "response_format", "speed", "seed",
            "max_new_tokens", "duration", "num_inference_steps", "guidance_scale", "user",
            "references", "stream_audio", "split_granularity", "text_input_mode",
        };
        json fields = json::object();
        for (auto it = j.begin(); it != j.end(); ++it) {
            bool known = false;
            for (const char* k : kFields) known = known || it.key() == k;
            if (it.key() == "input")
                return error("the text arrives in input.text messages, not in the configuration",
                             "input");
            if (!known)
                return error("'" + it.key() + "' is not a field of session.config", it.key());
            if (!session_field(it.key())) fields[it.key()] = it.value();
        }
        if (j.contains("text_input_mode") && !j["text_input_mode"].is_null()) {
            const json& m = j["text_input_mode"];
            if (!m.is_string() || (m != "buffered" && m != "commitment"))
                return error("text_input_mode is buffered", "text_input_mode");
            if (m == "commitment")
                return error("text_input_mode commitment is not served here; buffered is",
                             "text_input_mode");
        }
        TextSplit split = TextSplit::Sentence;
        if (j.contains("split_granularity") && !j["split_granularity"].is_null()) {
            const json& g = j["split_granularity"];
            if (g == "none") split = TextSplit::None;
            else if (g == "clause") split = TextSplit::Clause;
            else if (g != "sentence")
                return error("split_granularity is none, sentence or clause", "split_granularity");
        }
        bool stream_audio = false;
        if (j.contains("stream_audio") && !j["stream_audio"].is_null()) {
            if (!j["stream_audio"].is_boolean())
                return error("stream_audio is a boolean", "stream_audio");
            stream_audio = j["stream_audio"].get<bool>();
        }
        bool wav = false;
        if (j.contains("response_format") && !j["response_format"].is_null()) {
            const json& f = j["response_format"];
            if (f != "pcm" && f != "wav")
                return error("response_format is pcm or wav", "response_format");
            wav = f == "wav";
        }
        /* The request fields as every sentence will be made with them, tried on a stand-in. */
        std::unique_ptr<gen::AudioRequest> probe;
        ApiError err;
        if (speech_request(fields, "a", &probe, &err) < 0) return error(err.message, err.param);
        s.fields = std::move(fields);
        s.split = split;
        s.stream_audio = stream_audio;
        s.wav = wav;
        s.configured = true;
    };

    auto take = [&](bool final) {
        for (std::string& u : take_text_units(&s.pending, s.split, final))
            s.sentences.push_back(std::move(u));
    };

    /* Returns true when the session is over. */
    auto handle = [&](const std::string& msg, bool bin) -> bool {
        if (bin) {
            error("this session takes JSON messages: session.config, input.text, input.done and "
                  "session.close");
            return false;
        }
        /* Bounded like a request body: a deep bracket nest is a legal frame, and tearing its
         * tree down recurses once per level. */
        bool over = false, deep = false;
        const json j = parse_json_bounded(msg, 0, &over, &deep);
        if (j.is_discarded() || over || deep || !j.is_object() || !j.contains("type") ||
            !j["type"].is_string()) {
            error("a client message is a JSON object with a string `type`");
            return false;
        }
        const std::string type = j["type"].get<std::string>();
        if (type == "session.config") {
            configure(j);
            return false;
        }
        if (type == "session.close") {
            ws.close(1000, "session closed");
            return true;
        }
        if (type != "input.text" && type != "input.done") {
            error("this session takes session.config, input.text, input.done and session.close; "
                  "not '" + type + "'", "type");
            return false;
        }
        if (!s.configured) {
            error("send session.config before any text");
            return false;
        }
        if (type == "input.done") {
            take(true);
            s.done_asked = true;
            return false;
        }
        if (!j.contains("text") || !j["text"].is_string()) {
            error("input.text carries `text`, a string", "text");
            return false;
        }
        const std::string& t = j["text"].get_ref<const std::string&>();
        if (s.pending.size() + t.size() > kMaxPendingBytes) {
            error(fmt("the utterance holds more than %zu KiB of text without a sentence's end; "
                      "this piece was dropped", kMaxPendingBytes >> 10), "text");
            return false;
        }
        s.pending += t;
        take(false);
        return false;
    };

    /* ---- a sentence: begun, its chunks sent as they come in order, and ended */
    auto done = [&](bool failed) {
        ojson o = event("audio.done");
        o["utterance_index"] = s.utterance;
        o["sentence_index"] = s.cur.index;
        o["total_bytes"] = s.cur.bytes;
        o["error"] = failed;
        send(o);
        s.cur = SpeechSession::Current{};
        ++s.spoken;
    };
    auto begin = [&]() {
        const std::string text = std::move(s.sentences.front());
        s.sentences.pop_front();
        s.cur = SpeechSession::Current{};
        s.cur.index = s.sentence++;
        ojson o = event("audio.start");
        o["utterance_index"] = s.utterance;
        o["sentence_index"] = s.cur.index;
        o["sentence_text"] = text;
        o["format"] = s.wav ? "wav" : "pcm";
        o["sample_rate"] = info.sample_rate;
        o["channels"] = info.channels;
        send(o);
        std::unique_ptr<gen::AudioRequest> r;
        ApiError err;
        if (speech_request(s.fields, text, &r, &err) < 0) {
            error(err.message, err.param);
            return done(true);
        }
        r->ticket = std::make_shared<gen::AudioTicket>();
        std::shared_ptr<gen::AudioTicket> ticket = r->ticket;
        std::string why;
        const int st = deps_.audio->submit(std::move(r), &why);
        if (st < 0) {
            error(why.empty() ? std::string(rad_strerror(st)) : why);
            return done(true);
        }
        s.cur.active = true;
        s.cur.ticket = std::move(ticket);
        if (s.stream_audio && s.wav) binary(wav_header(-1, info.sample_rate, info.channels));
    };
    auto poll = [&](int wait_ms) {
        gen::AudioChunk c;
        bool to = false;
        for (int ms = wait_ms;; ms = 0) {
            if (s.cur.ticket->next(&c, ms, &to)) {
                s.cur.held[c.window] = std::move(c.samples);
                std::string pcm;
                for (auto it = s.cur.held.find(s.cur.next); it != s.cur.held.end();
                     it = s.cur.held.find(s.cur.next)) {
                    append_pcm16(it->second, &pcm);
                    s.cur.held.erase(it);
                    ++s.cur.next;
                }
                if (s.stream_audio) binary(pcm);
                else s.cur.pcm += pcm;
                continue;
            }
            if (to) return;
            /* The sentence has ended. */
            gen::AudioTicket& t = *s.cur.ticket;
            if (t.status() < 0 || s.cur.next == 0) {
                error(t.why().empty() ? "the audio generator failed" : t.why());
                return done(true);
            }
            if (!s.stream_audio) {
                const std::string& pcm = s.cur.pcm;
                binary(s.wav ? wav_header((int64_t)pcm.size(), info.sample_rate, info.channels) +
                                   pcm
                             : pcm);
            }
            return done(false);
        }
    };

    /* ---- the loop */
    RAD_INFO("speech stream %s open from %s", s.id.c_str(), s.peer.c_str());
    for (;;) {
        std::deque<std::pair<std::string, bool>> got;
        bool gone = false, overflow = false;
        {
            std::unique_lock<std::mutex> lk(s.m);
            const bool work = s.cur.active || !s.sentences.empty() || s.done_asked;
            if (s.inbox.empty() && !s.peer_gone && !s.overflow && !work)
                s.cv.wait_for(lk, std::chrono::milliseconds(100));
            got.swap(s.inbox);
            s.inbox_bytes = 0;
            gone = s.peer_gone;
            overflow = s.overflow;
        }
        if (overflow) {
            error(fmt("the session fell more than %zu MiB of messages behind",
                      kMaxQueuedBytes >> 20));
            ws.close(1008, "buffer overflow");
            return true;
        }
        for (const auto& m : got)
            if (handle(m.first, m.second)) return true;
        if (gone) return false;
        if (stopping_.load()) {
            error("the server is shutting down");
            ws.close(1001, "closing");
            return true;
        }
        if (!s.cur.active && !s.sentences.empty()) begin();
        if (s.cur.active) {
            bool idle;
            {
                std::lock_guard<std::mutex> lk(s.m);
                idle = s.inbox.empty() && !s.peer_gone;
            }
            poll(idle ? 10 : 0);
        }
        if (!s.cur.active && s.sentences.empty() && s.done_asked) {
            ojson o = event("session.done");
            o["utterance_index"] = s.utterance;
            o["total_sentences"] = s.sentence;
            send(o);
            ++s.utterance;
            s.sentence = 0;
            s.done_asked = false;
        }
    }
}

}  /* namespace server */
}  /* namespace rad */
