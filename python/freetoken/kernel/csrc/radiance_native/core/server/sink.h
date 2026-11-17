// Modified for FreeToken: native integration, diagnostics and Qwen FP8 KV support.
/* core/server/sink.h -- the per-request channel from the scheduler thread to the HTTP thread.
 *
 * `Request::sink` is the void* hook for this (rad_core.h). One scheduler thread writes and one
 * HTTP thread reads, so it is a lock-free single-producer single-consumer ring: relaxed load of
 * the head, store into a slot, release store of the head. The producer never allocates or
 * blocks, so a slow HTTP client cannot delay a step for the whole batch.
 *
 * Only the consumer takes the mutex. The producer calls notify_one() without it, which can lose
 * a wake-up between the consumer's predicate test and its sleep, so the consumer waits at most
 * RAD_SINK_POLL_MS. A lost wake-up costs at most that much latency on one token; in exchange the
 * scheduler never waits on a lock an HTTP thread holds. The same timeout is the streaming path's
 * client-liveness poll interval.
 *
 * Tokens, not text: detokenisation is stateful and per-request (spec §12) and belongs on the HTTP
 * thread, and the producer's write stays a fixed-size SinkToken.
 */
#pragma once
#include "rad_core.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

namespace rad {
namespace server {

/* How long the consumer sleeps before re-testing. Also the connection-liveness poll interval. */
static constexpr int RAD_SINK_POLL_MS = 25;

enum class Finish : int {
    None = 0,      /* still running */
    Stop,          /* EOS */
    StopString,    /* a stop string matched */
    Length,        /* max_tokens */
    ToolCalls,     /* the parser found a complete tool call and the model stopped */
    Cancelled,     /* client went away, or /cancel */
    Error,         /* the engine failed the request; `status` says why */
};

/* The OpenAI `finish_reason` string. Cancelled has no OpenAI spelling; it is "abort" as in vLLM,
 * which dashboards key off. */
const char* finish_reason_oai(Finish f);

struct SinkToken {
    int32_t token = 0;
    float   logprob = 0.0f;   /* the token's logprob; 0 unless the request asked for logprobs */
    int32_t n_alt = 0;        /* how many alternatives read() wrote for this token */
    bool    accepted_draft = false;
};

/* One of a token's most likely alternatives, with its logprob. */
struct SinkAlt {
    int32_t id = 0;
    float   logprob = 0.0f;
};

class Sink {
public:
    // FreeToken opt-in accounting reuses producer timestamps, never adds GPU synchronization.
    struct CommitGroup { int64_t time_ns; uint32_t tokens; uint32_t accepted_draft; };
    struct VerifyRound { int32_t proposed; int32_t accepted_by_sampler; uint64_t produced_total_after; int64_t last_commit_since_start_ns; };
    struct RoundLedger {
        bool enabled = false;
        std::vector<CommitGroup> commits;
        std::vector<VerifyRound> verifications;
    };
    RoundLedger round_ledger() const;
    /* `capacity` is rounded up to a power of two. 8192 tokens is 80 s of slack at 100 tok/s, far
     * more than a live consumer falls behind. Overflow fails the request rather than dropping
     * tokens from the middle of the stream. */
    explicit Sink(size_t capacity = 8192);

    /* Waits until the producer has stopped touching this object.
     *
     * Observing `done()` is not enough: the producer publishes the terminal state and only then
     * calls notify_one(), and destroying the condition_variable in between is a use-after-free in
     * pthread_cond_signal. The producer's last act is to set a quiesced flag, which this waits
     * for. The window is a few instructions wide. */
    ~Sink();

    /* ------------------------------------------------------------------ producer: scheduler */

    /* Returns false only on overflow, which also marks the request failed. `lp` is the token's
     * logprobs when the request asked for them (enable_logprobs), and null otherwise. `now`, a
     * sink_now_ns() reading, is the token's time; negative reads the clock. A step's tokens share
     * one reading: where the clocksource is HPET (the TSC marked unstable) a read is a system call
     * of about a microsecond, a token's whole cost here. */
    bool push(int32_t token, const TokenLogprobs* lp = nullptr, int64_t now = -1,
              bool accepted_draft = false);

    /* Terminal and idempotent: the first call wins, so a cancel racing a natural stop does not
     * produce two finish reasons. */
    void finish(Finish f);
    /* `why` is a client-readable sentence that outlives the sink (a literal), or null. */
    void fail(int rad_status, const char* why = nullptr);

    /* The scheduler polls this to learn the client hung up, then frees the blocks (spec §14).
     * IScheduler::cancel() also tells it; this is the cheap path with no lookup. */
    bool cancel_requested() const { return cancel_.load(std::memory_order_acquire); }

    /* Admission accounting, set once by the scheduler when the prompt is placed. */
    void set_prompt_stats(int32_t n_prompt, int32_t n_cached);

    /* Marks the first step the scheduler gives this request, where its prompt phase starts, so
     * queue time is not counted as prefill. A preempted request keeps its first mark. */
    void mark_admitted();
    /* One verify step: `n` draft tokens proposed, `accepted` taken. Summed per request for
     * /metrics' draft counters. */
    void add_drafts(int32_t n, int32_t accepted);
    /* Prompt tokens processed of `total`, `cached` of them from the prefix cache. Set at
     * admission and after every prefill step; streamed as llama-server's `prompt_progress` when
     * a request asks for it. Wakes the consumer. */
    void set_prompt_progress(int32_t processed, int32_t total, int32_t cached);
    int32_t progress_processed() const { return prog_done_.load(std::memory_order_acquire); }
    int32_t progress_total() const    { return prog_total_.load(std::memory_order_acquire); }
    int32_t progress_cached() const   { return prog_cached_.load(std::memory_order_acquire); }

    /* ------------------------------------------------------------------ consumer: HTTP */

    /* Copies up to `max` tokens out and returns how many; never blocks. With `alts`, token i's
     * n_alt alternatives go to alts[i * alt_width()], which needs room for max * alt_width(). */
    size_t read(SinkToken* out, size_t max, SinkAlt* alts = nullptr);

    /* Sleeps until there is something to read, the request is terminal, or `timeout_ms` elapses.
     * Returns true if there is now something to look at. */
    bool wait(int timeout_ms = RAD_SINK_POLL_MS);

    bool   done() const { return finish_.load(std::memory_order_acquire) != (int)Finish::None; }
    Finish finish_reason() const { return (Finish)finish_.load(std::memory_order_acquire); }
    int    status() const { return status_.load(std::memory_order_acquire); }
    const char* why() const { return why_.load(std::memory_order_acquire); }
    bool   overflowed() const { return overflow_.load(std::memory_order_acquire); }

    /* Called by the HTTP thread when the peer disappears. Frees nothing: the server also calls
     * IScheduler::cancel(id); this only stops the producer sooner. */
    void request_cancel();

    /* Timings, in steady-clock nanoseconds since the sink was constructed. -1 until observed. */
    int64_t t_first_token_ns() const { return t_first_.load(std::memory_order_acquire); }
    int64_t t_last_token_ns() const  { return t_last_.load(std::memory_order_acquire); }
    int64_t t_admitted_ns() const    { return t_admit_.load(std::memory_order_acquire); }
    int64_t age_ns() const;
    /* The steady clock at construction, which the times above count from; puts two sinks'
     * times on one axis. */
    int64_t t_start_ns() const { return t_start_ns_; }

    int64_t n_drafted() const        { return n_draft_.load(std::memory_order_acquire); }
    int64_t n_draft_accepted() const { return n_draft_acc_.load(std::memory_order_acquire); }

    int64_t n_produced() const { return (int64_t)head_.load(std::memory_order_acquire); }
    int32_t n_prompt() const { return n_prompt_.load(std::memory_order_acquire); }
    int32_t n_cached() const { return n_cached_.load(std::memory_order_acquire); }

    /* Allocates the alternatives ring, `n` per token, beside the token ring. Only requests that
     * ask for logprobs get one: at RAD_MAX_LOGPROBS it is twenty times the token ring. The
     * consumer calls this before handing the request to the scheduler. */
    void enable_logprobs(int n);
    int  alt_width() const { return alt_w_; }

    /* The engine's request id, so a dropped connection can reach IScheduler::cancel. */
    uint64_t id = 0;

    /* An embedding request's pooled vector; empty for generating requests. Written once by the
     * engine before the request finishes and read only after done(): finish()'s release store
     * and done()'s acquire load publish it. */
    void set_embedding(const float* v, int64_t n) { embd_.assign(v, v + n); }
    const std::vector<float>& embedding() const { return embd_; }

private:
    void notify();

    std::vector<SinkToken> ring_;
    size_t                 mask_ = 0;

    std::atomic<uint64_t> head_{0};      /* produced count; producer writes, both read */
    std::atomic<uint64_t> tail_{0};      /* consumed count; consumer writes, both read */

    std::atomic<int>      finish_{(int)Finish::None};
    std::atomic<int>      status_{RAD_OK};
    std::atomic<const char*> why_{nullptr};
    std::atomic<bool>     overflow_{false};
    std::atomic<bool>     cancel_{false};
    std::atomic<bool>     quiesced_{false};   /* the producer will not touch this object again */

    std::atomic<int64_t>  t_first_{-1};
    std::atomic<int64_t>  t_last_{-1};
    std::atomic<int64_t>  t_admit_{-1};
    std::atomic<int64_t>  n_draft_{0};
    std::atomic<int64_t>  n_draft_acc_{0};
    int64_t               t_start_ns_ = 0;

    std::atomic<int32_t>  n_prompt_{0};
    std::atomic<int32_t>  n_cached_{0};
    std::atomic<int32_t>  prog_done_{-1};
    std::atomic<int32_t>  prog_total_{0};
    std::atomic<int32_t>  prog_cached_{0};
    std::vector<SinkAlt>  alts_;                 /* [capacity * alt_w_], parallel to ring_ */
    std::vector<int32_t>  alt_n_;                /* [capacity] */
    int                   alt_w_ = 0;
    std::vector<float>    embd_;

    std::mutex              m_;
    std::condition_variable cv_;
    std::atomic<bool>       asleep_{false};   /* the consumer is inside cv_.wait_for */

    bool record_rounds_ = false;
    mutable std::mutex rounds_mu_;
    std::vector<CommitGroup> commit_groups_;
    std::vector<VerifyRound> verify_rounds_;

};

/* The scheduler's call sites, so core/sched needs no casts. A null sink means nobody is
 * listening (an offline or benchmark request). */
inline bool sink_push(void* s, int32_t token, const TokenLogprobs* lp = nullptr, int64_t now = -1,
                      bool accepted_draft = false) {
    return s ? static_cast<Sink*>(s)->push(token, lp, now, accepted_draft) : true;
}
/* The clock a sink's times are on, for push's `now`. */
int64_t sink_now_ns();
inline void sink_finish(void* s, Finish f) {
    if (s) static_cast<Sink*>(s)->finish(f);
}
inline void sink_fail(void* s, int rad_status, const char* why = nullptr) {
    if (s) static_cast<Sink*>(s)->fail(rad_status, why);
}
inline bool sink_cancelled(const void* s) {
    return s && static_cast<const Sink*>(s)->cancel_requested();
}
inline void sink_prompt_stats(void* s, int32_t n_prompt, int32_t n_cached) {
    if (s) static_cast<Sink*>(s)->set_prompt_stats(n_prompt, n_cached);
}
inline void sink_admitted(void* s) {
    if (s) static_cast<Sink*>(s)->mark_admitted();
}
inline void sink_prompt_progress(void* s, int32_t processed, int32_t total, int32_t cached) {
    if (s) static_cast<Sink*>(s)->set_prompt_progress(processed, total, cached);
}
inline void sink_drafts(void* s, int32_t n, int32_t accepted) {
    if (s) static_cast<Sink*>(s)->add_drafts(n, accepted);
}
inline void sink_embedding(void* s, const float* v, int64_t n) {
    if (s) static_cast<Sink*>(s)->set_embedding(v, n);
}

}  /* namespace server */
}  /* namespace rad */
