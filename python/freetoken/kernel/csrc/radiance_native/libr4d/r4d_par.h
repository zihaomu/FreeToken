/* r4d_par.h -- the load-time relayouts' split of one weight over the host's cores. */
#pragma once

#include <algorithm>
#include <thread>
#include <vector>

namespace r4d_par {

/* Runs f(lo, hi) over [0, n) in parallel chunks, one thread for every `per_thread` items' worth of
 * work and at most 16. A model's weights relayout at load, gigabytes of them, so a large one is
 * split over the host's cores; the loader already runs several relayouts side by side, so a small
 * one stays on the calling thread rather than starting threads that would only contend. */
template <typename F>
void par_for(long long n, long long per_thread, F f) {
    const long long hc = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    const long long nt = std::max(1ll, std::min(hc, n / std::max(1ll, per_thread)));
    if (nt == 1) {
        f(0ll, n);
        return;
    }
    const long long per = (n + nt - 1) / nt;
    std::vector<std::thread> th;
    for (long long lo = 0; lo < n; lo += per) th.emplace_back(f, lo, std::min(n, lo + per));
    for (std::thread& t : th) t.join();
}

}  // namespace r4d_par
