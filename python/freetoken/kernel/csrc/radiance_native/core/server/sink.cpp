// Modified for FreeToken: native integration, diagnostics and Qwen FP8 KV support.
#include "sink.h"
#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <thread>

namespace rad {
namespace server {

static int64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

int64_t sink_now_ns() { return now_ns(); }

const char* finish_reason_oai(Finish f) {
    switch (f) {
        case Finish::Stop:
        case Finish::StopString: return "stop";
        case Finish::Length:     return "length";
        case Finish::ToolCalls:  return "tool_calls";
        case Finish::Cancelled:  return "abort";   /* vLLM's spelling; dashboards key off it */
        case Finish::Error:      return "error";
        case Finish::None:       return "";
    }
    return "";
}

static size_t round_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

Sink::Sink(size_t capacity) {
    const char* rounds = std::getenv("FREETOKEN_NATIVE_ROUND_LEDGER");
    record_rounds_ = rounds && rounds[0] == '1' && rounds[1] == '\0';
    if (record_rounds_) { commit_groups_.reserve(1024); verify_rounds_.reserve(1024); }
    size_t cap = round_pow2(capacity < 16 ? 16 : capacity);
    ring_.resize(cap);
    mask_ = cap - 1;
    t_start_ns_ = now_ns();
}

Sink::~Sink() {
    /* Only a sink the producer terminated can have a notify in flight. */
    if (finish_.load(std::memory_order_acquire) == (int)Finish::None) return;
    for (int spins = 0; !quiesced_.load(std::memory_order_acquire); ++spins) {
        /* The window is a few instructions wide, so a short spin covers it; the sleep keeps a
         * descheduled producer from burning a core. */
        if (spins > 256) std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}

int64_t Sink::age_ns() const { return now_ns() - t_start_ns_; }

void Sink::notify() {
    /* Deliberately without m_ (see sink.h): a lost wake-up is covered by the consumer's bounded
     * wait, and the scheduler thread never waits on an HTTP thread. */
    cv_.notify_one();
}

void Sink::enable_logprobs(int n) {
    alt_w_ = n > 0 ? n : 0;
    alts_.assign(ring_.size() * (size_t)alt_w_, SinkAlt{});
    alt_n_.assign(ring_.size(), 0);
}

bool Sink::push(int32_t token, const TokenLogprobs* lp, int64_t now, bool accepted_draft) {
    const uint64_t h = head_.load(std::memory_order_relaxed);
    const uint64_t t = tail_.load(std::memory_order_acquire);
    if (h - t > mask_) {
        /* The consumer is not draining; fail rather than leave a hole in the stream. */
        overflow_.store(true, std::memory_order_release);
        fail(RAD_E_FULL);
        return false;
    }
    const size_t slot = (size_t)(h & mask_);
    ring_[slot] = SinkToken{ token, lp ? lp->logprob : 0.0f, 0, accepted_draft };
    if (alt_w_ > 0) {
        const int n = lp ? std::min(lp->n_top, alt_w_) : 0;
        for (int j = 0; j < n; ++j)
            alts_[slot * (size_t)alt_w_ + (size_t)j] = SinkAlt{ lp->top_id[j], lp->top_lp[j] };
        alt_n_[slot] = n;
    }
    head_.store(h + 1, std::memory_order_release);

    const int64_t t_now = (now >= 0 ? now : now_ns()) - t_start_ns_;
    if (h == 0) t_first_.store(t_now, std::memory_order_release);
    t_last_.store(t_now, std::memory_order_release);
    if (record_rounds_) {
        std::lock_guard<std::mutex> lock(rounds_mu_);
        if (commit_groups_.empty() || commit_groups_.back().time_ns != t_now)
            commit_groups_.push_back({t_now, 0, 0});
        ++commit_groups_.back().tokens;
        commit_groups_.back().accepted_draft += accepted_draft ? 1 : 0;
    }


    /* A wake only while the consumer is asleep (asleep_): at hundreds of streams a wake per
     * token per stream, for consumers already awake draining, is a syscall each on the
     * scheduler thread. Every push while it sleeps wakes it, so a wake that lands after its
     * predicate test but before it sleeps (notify() takes no lock) is repaired by the next
     * push, not left to the bounded wait. The fence orders the head store before the flag's
     * load, against the consumer's flag store before its predicate's head load. */
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (asleep_.load(std::memory_order_relaxed)) notify();
    return true;
}

void Sink::finish(Finish f) {
    int expected = (int)Finish::None;
    /* First terminal state wins: a cancel racing an EOS must not be reported twice. */
    if (finish_.compare_exchange_strong(expected, (int)f, std::memory_order_acq_rel)) {
        notify();
        /* The producer's last write to this object, which the destructor waits on. It must
         * follow the notify: it publishes that the condition variable is no longer in use. */
        quiesced_.store(true, std::memory_order_release);
    }
}

void Sink::fail(int rad_status, const char* why) {
    status_.store(rad_status, std::memory_order_release);
    why_.store(why, std::memory_order_release);
    finish(Finish::Error);
}

void Sink::set_prompt_stats(int32_t n_prompt, int32_t n_cached) {
    n_prompt_.store(n_prompt, std::memory_order_release);
    n_cached_.store(n_cached, std::memory_order_release);
}

void Sink::set_prompt_progress(int32_t processed, int32_t total, int32_t cached) {
    prog_total_.store(total, std::memory_order_release);
    prog_cached_.store(cached, std::memory_order_release);
    prog_done_.store(processed, std::memory_order_release);
    notify();
}

void Sink::mark_admitted() {
    if (t_admit_.load(std::memory_order_relaxed) < 0)
        t_admit_.store(now_ns() - t_start_ns_, std::memory_order_release);
}

/* One producer, so a load and a store suffice. */
void Sink::add_drafts(int32_t n, int32_t accepted) {
    n_draft_.store(n_draft_.load(std::memory_order_relaxed) + n, std::memory_order_release);
    n_draft_acc_.store(n_draft_acc_.load(std::memory_order_relaxed) + accepted,
                       std::memory_order_release);
    if (record_rounds_) {
        std::lock_guard<std::mutex> lock(rounds_mu_);
        verify_rounds_.push_back({n, accepted, head_.load(std::memory_order_acquire),
                                  t_last_.load(std::memory_order_acquire)});
    }
}

Sink::RoundLedger Sink::round_ledger() const {
    if (!record_rounds_) return {};
    std::lock_guard<std::mutex> lock(rounds_mu_);
    return {true, commit_groups_, verify_rounds_};
}

size_t Sink::read(SinkToken* out, size_t max, SinkAlt* alts) {
    const uint64_t t = tail_.load(std::memory_order_relaxed);
    const uint64_t h = head_.load(std::memory_order_acquire);
    size_t n = (size_t)(h - t);
    if (n > max) n = max;
    for (size_t i = 0; i < n; ++i) {
        const size_t slot = (size_t)((t + i) & mask_);
        out[i] = ring_[slot];
        if (!alts || alt_w_ <= 0) continue;
        out[i].n_alt = alt_n_[slot];
        for (int j = 0; j < alt_n_[slot]; ++j)
            alts[i * (size_t)alt_w_ + (size_t)j] = alts_[slot * (size_t)alt_w_ + (size_t)j];
    }
    if (n) tail_.store(t + n, std::memory_order_release);
    return n;
}

bool Sink::wait(int timeout_ms) {
    auto ready = [&] {
        return head_.load(std::memory_order_acquire) != tail_.load(std::memory_order_relaxed)
            || finish_.load(std::memory_order_acquire) != (int)Finish::None
            || cancel_.load(std::memory_order_acquire);
    };
    if (ready()) return true;
    std::unique_lock<std::mutex> lk(m_);
    asleep_.store(true, std::memory_order_seq_cst);
    cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
    asleep_.store(false, std::memory_order_relaxed);
    return ready();
}

void Sink::request_cancel() {
    cancel_.store(true, std::memory_order_release);
    notify();
}

}  /* namespace server */
}  /* namespace rad */
