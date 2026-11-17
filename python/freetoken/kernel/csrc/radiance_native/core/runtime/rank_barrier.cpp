/* rank_barrier.cpp -- see rank_barrier.h for why this is a condvar and not an IPC handshake. */
#include "rank_barrier.h"

namespace rad {

void RankBarrier::arrive_and_wait() {
    std::unique_lock<std::mutex> lk(m_);
    if (aborted_) return;

    const uint64_t my_gen = gen_;
    if (++waiting_ == n_) {
        /* Last one in resets the counter and flips the generation, so a rank that re-enters at
         * once waits on the new generation instead of racing through. */
        waiting_ = 0;
        ++gen_;
        lk.unlock();
        cv_.notify_all();
        return;
    }
    cv_.wait(lk, [&] { return aborted_ || gen_ != my_gen; });
}

void RankBarrier::abort() {
    {
        std::lock_guard<std::mutex> lk(m_);
        aborted_ = true;
        waiting_ = 0;
        ++gen_;
    }
    cv_.notify_all();
}

}  /* namespace rad */
