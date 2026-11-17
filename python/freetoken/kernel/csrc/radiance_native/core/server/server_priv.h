/* server_priv.h -- the Server's per-request state, shared by the units that serve requests
 * (server.cpp, speech.cpp, realtime.cpp). Not part of the server's interface: nothing outside
 * core/server includes it. */
#pragma once
#include "server.h"
#include "rad_internal.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rad {
namespace mm { struct Item; }
namespace server {

/* ------------------------------------------------------------------ per-choice state */

struct Server::Generation {
    std::shared_ptr<Sink>         sink;
    /* The runtime LoRA adapter this choice runs with, held so it cannot be unloaded under it. */
    std::shared_ptr<void>         lora_hold;
    std::unique_ptr<IDetokenizer> detok;
    std::unique_ptr<ReplyStream> delta;   /* chat streaming only, in the request's API shape */
    uint64_t    id = 0;
    int         index = 0;
    int         prompt_index = 0;
    std::string text;
    std::vector<LogprobEntry> logprobs;
    size_t      lp_sent = 0;         /* streaming: how many of `logprobs` a chunk already carried */
    /* Tracks every delivered token against the reasoning block, for
     * completion_tokens_details.reasoning_tokens. */
    ReasoningCount think;
    Finish      finish = Finish::None;
    bool        closed = false;      /* the server has decided this choice is over */
    size_t      scanned = 0;         /* how far the stop-string scan has reached */
    /* OaiRequest::stops_after: where in `text` the reasoning block's closer ends (npos until it
     * has been generated) and how far the search for it has reached. */
    size_t      stops_floor = std::string::npos;
    size_t      closer_scanned = 0;
    size_t      stop_cut = std::string::npos;
    std::string stop_word;           /* the stop string at stop_cut, "" when the decoder cut it */
    size_t      emitted = 0;         /* streaming: text already handed to the delta stream */
    std::vector<int32_t> tokens;     /* llama-server's native shape: the ids generated */
    size_t      tokens_sent = 0;     /* ...and how many of them a stream chunk carried */
    int32_t     progress_sent = -1;  /* streaming: the prompt_progress last sent, -1 for none */
    bool        retired = false;     /* reap() has been said for this id, at most once */
};

struct Server::StreamState {
    OaiRequest              r;
    std::vector<Generation> gens;
    std::function<bool()>   peer_closed;
    bool opened = false;
    bool done_sent = false;
    bool released = false;
};

/* The failure message for the client: the failing component's sentence and the status. A failed
 * request always carries a negative status (RAD_E_STATE if none was set), so this never reads
 * "ok". */
inline std::string engine_failure(const Sink& s) {
    const int st = s.status() < 0 ? s.status() : RAD_E_STATE;
    const char* why = s.why();
    return why ? fmt("the engine failed this request: %s (%s)", why, rad_strerror(st))
               : fmt("the engine failed this request: %s", rad_strerror(st));
}

/* ------------------------------------------------------------------ speech (speech.cpp)
 * The model's transcription format, used by the upload route and the realtime one alike. */

/* The plugin's template with its three slots filled (`context` as plain text, the clip's
 * placeholder run, and `language_text`, from speech_language_text or ""), as tokens with the clip
 * placed. RAD_E_INVAL when the template has no {audio} slot. */
int speech_prompt(const SpeechFormat& f, ITokenizer& tok, const std::string& context,
                  const std::string& language_text, const std::shared_ptr<mm::Item>& item,
                  PromptInput* out);
/* What a forced language puts in the {language} slot: the plugin's language_prompt naming it. */
std::string speech_language_text(const SpeechFormat& f, const std::string& name);
/* Where the transcript starts in a reply so far: past the transcript's opener, once it has been
 * generated, and past the whitespace after it. npos while a reply that is not `closed` has not
 * named it yet; a closed reply that never named it is all transcript. */
size_t speech_transcript_start(const SpeechFormat& f, const std::string& text, bool forced,
                               bool closed);
/* The centre of the quietest 100 ms of x[lo, hi): where a recording too long for one pass is cut,
 * between words when there is a pause to cut in. */
int64_t speech_quiet_point(const std::vector<float>& x, int64_t lo, int64_t hi, int32_t rate);

/* ---- audio_gen.cpp: what the audio generator's routes write */
/* Samples in [-1, 1] as 16-bit little-endian PCM, appended. */
void append_pcm16(const std::vector<float>& x, std::string* out);
/* A RIFF/WAVE header for `bytes` of 16-bit PCM; -1 states no length, as a stream's must. */
std::string wav_header(int64_t bytes, int64_t rate, int64_t channels);

}  /* namespace server */
}  /* namespace rad */
