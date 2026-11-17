// Cost of a scattered KV eviction. The prefix cache is block-granular and its blocks are scattered
// through the paged pool, so a bulk device-to-host bandwidth figure does not price it. Compares one
// contiguous copy of N*B bytes, N copies of B bytes from scattered offsets with a sync after each,
// and the same N copies issued back-to-back on one stream (the batched form an eviction would use).
//
//   clang++ -x hip -O3 -std=c++17 --offload-arch=gfx1201 blockcopy.hip -o blockcopy
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
    const long long bytes = (argc > 1 ? std::atoll(argv[1]) : 1382) * 1024LL * 1024LL; /* ~1.35 GiB */
    const int  blk   = argc > 2 ? std::atoi(argv[2]) : 122880;   /* bytes a block */
    const int  iters = argc > 3 ? std::atoi(argv[3]) : 5;
    const long long n = bytes / blk;

    void* dev = nullptr;
    if (hipMalloc(&dev, (size_t)bytes) != hipSuccess) { std::printf("hipMalloc %lld failed\n", bytes); return 1; }
    void* host = nullptr;
    if (hipHostMalloc(&host, (size_t)bytes) != hipSuccess) { std::printf("hipHostMalloc failed\n"); return 1; }
    hipStream_t s; hipStreamCreate(&s);

    /* Shuffled order: a pool hands out blocks in free order, uncorrelated with address. */
    std::vector<long long> order((size_t)n);
    for (long long i = 0; i < n; ++i) order[(size_t)i] = i;
    unsigned seed = 12345;
    for (long long i = n - 1; i > 0; --i) {
        seed = seed * 1103515245u + 12345u;
        std::swap(order[(size_t)i], order[(size_t)(seed % (unsigned)(i + 1))]);
    }

    std::printf("%lld MiB, block %d B, %lld blocks, %d iters\n", bytes >> 20, blk, n, iters);

    auto run = [&](const char* nm, bool scattered, bool sync_each) {
        double best = 1e30;
        for (int it = 0; it < iters; ++it) {
            const auto t0 = clk::now();
            if (!scattered) {
                hipMemcpyAsync(host, dev, (size_t)bytes, hipMemcpyDeviceToHost, s);
            } else {
                for (long long i = 0; i < n; ++i) {
                    const long long off = order[(size_t)i] * blk;
                    hipMemcpyAsync((char*)host + off, (char*)dev + off, (size_t)blk,
                                   hipMemcpyDeviceToHost, s);
                    if (sync_each) hipStreamSynchronize(s);
                }
            }
            hipStreamSynchronize(s);
            best = std::min(best, ms_since(t0, clk::now()));
        }
        std::printf("  %-28s %8.1f ms   %6.2f GB/s\n", nm, best,
                    (double)bytes / (best * 1e6));
    };

    run("one contiguous copy", false, false);
    run("scattered, batched on a stream", true, false);
    run("scattered, sync each block", true, true);
    return 0;
}
