// Does a launch cost more when the module registers more kernels, and does crossing into a shared
// object cost anything?
//
// An engine launch costs roughly twice what the same shape costs in a self-contained binary, and
// the excess is roughly constant across kernel shapes: a fixed per-issue tax. The per-launch error
// check, a full command ring and the argument count do not account for it.
//
// This isolates the module. libr4d is a .so holding every kernel instantiation, entered through a
// function pointer; a self-contained harness compiles a handful of kernels into the binary. HIP
// resolves a launch by looking the host stub up in the table __hipRegisterFunction fills, so a
// bigger table is a plausible tax, and then the lever is fewer instantiations.
//
//   see build_launchreg.sh
#include <hip/hip_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <vector>

__global__ void local_k(float* p) { if (threadIdx.x == 0) p[blockIdx.x] = 1.0f; }

using clk = std::chrono::steady_clock;
static double us_since(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}
static double med(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

typedef int (*LaunchFn)(void*, void*);
typedef int (*DistinctFn)(void*, void*, int);
typedef int (*CountFn)(void);

struct Lib {
    const char* path;
    void*       h    = nullptr;
    LaunchFn    fn   = nullptr;
    DistinctFn  dfn  = nullptr;
    int         nreg = 0;
};

int main(int argc, char** argv) {
    const int N     = argc > 1 ? std::atoi(argv[1]) : 45;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 300;

    float* d = nullptr;
    if (hipMalloc(&d, 1024 * sizeof(float)) != hipSuccess) { std::printf("hipMalloc\n"); return 1; }
    hipStream_t s;
    if (hipStreamCreate(&s) != hipSuccess) { std::printf("hipStreamCreate\n"); return 1; }

    Lib few{"./liblr_few.so"}, many{"./liblr_many.so"};
    for (Lib* L : {&few, &many}) {
        L->h = dlopen(L->path, RTLD_NOW);
        if (!L->h) { std::printf("dlopen %s: %s\n", L->path, dlerror()); return 1; }
        L->fn  = (LaunchFn)dlsym(L->h, "lr_launch");
        L->dfn = (DistinctFn)dlsym(L->h, "lr_launch_distinct");
        CountFn c = (CountFn)dlsym(L->h, "lr_registered");
        if (!L->fn || !L->dfn || !c) { std::printf("dlsym %s\n", L->path); return 1; }
        L->nreg = c();
    }
    std::printf("%d launches a sequence, %d iterations, medians\n", N, iters);
    std::printf("  few  = %d instantiations, many = %d\n", few.nreg, many.nreg);

    /* Warm every path: clocks, lazy module load for both objects, and the code itself. */
    for (int w = 0; w < 400; ++w) {
        for (int i = 0; i < N; ++i) hipLaunchKernelGGL(local_k, dim3(1), dim3(64), 0, s, d);
        for (int i = 0; i < N; ++i) few.fn(d, s);
        for (int i = 0; i < N; ++i) many.fn(d, s);
        many.dfn(d, s, N);
        hipStreamSynchronize(s);
    }

    std::vector<double> a_loc, a_few, a_many, a_dist;
    for (int it = 0; it < iters; ++it) {          /* interleaved, one loop, shared clock state */
        auto t0 = clk::now();
        for (int i = 0; i < N; ++i) hipLaunchKernelGGL(local_k, dim3(1), dim3(64), 0, s, d);
        auto t1 = clk::now();
        hipStreamSynchronize(s);
        a_loc.push_back(us_since(t0, t1));

        t0 = clk::now();
        for (int i = 0; i < N; ++i) few.fn(d, s);
        t1 = clk::now();
        hipStreamSynchronize(s);
        a_few.push_back(us_since(t0, t1));

        t0 = clk::now();
        for (int i = 0; i < N; ++i) many.fn(d, s);
        t1 = clk::now();
        hipStreamSynchronize(s);
        a_many.push_back(us_since(t0, t1));

        t0 = clk::now();
        many.dfn(d, s, N);
        t1 = clk::now();
        hipStreamSynchronize(s);
        a_dist.push_back(us_since(t0, t1));
    }

    /* ---- A second submitting thread. --tp 2 runs two rank threads, each issuing onto its own
     * stream. A process-wide lock on HIP's submit path would add a fixed tax to every launch,
     * the same shape as the excess. The second thread runs the same loop on its own stream and
     * is not measured; only rank 0 is. */
    hipStream_t s2;
    if (hipStreamCreate(&s2) != hipSuccess) { std::printf("second stream\n"); return 1; }
    std::atomic<bool> run{true};
    std::atomic<long> other{0};
    std::thread rank1([&] {
        while (run.load(std::memory_order_relaxed)) {
            for (int i = 0; i < N; ++i) hipLaunchKernelGGL(local_k, dim3(1), dim3(64), 0, s2, d);
            hipStreamSynchronize(s2);
            other.fetch_add(N, std::memory_order_relaxed);
        }
    });
    std::vector<double> a_tp2;
    for (int w = 0; w < 200; ++w) {
        for (int i = 0; i < N; ++i) hipLaunchKernelGGL(local_k, dim3(1), dim3(64), 0, s, d);
        hipStreamSynchronize(s);
    }
    for (int it = 0; it < iters; ++it) {
        const auto t0 = clk::now();
        for (int i = 0; i < N; ++i) hipLaunchKernelGGL(local_k, dim3(1), dim3(64), 0, s, d);
        const auto t1 = clk::now();
        hipStreamSynchronize(s);
        a_tp2.push_back(us_since(t0, t1));
    }
    run.store(false);
    rank1.join();

    const double loc = med(a_loc);
    auto row = [&](const char* nm, std::vector<double>& v) {
        const double m = med(v);
        std::printf("  %-22s %7.1f us  (%5.3f us a launch)  %+6.3f us a launch vs in-binary\n",
                    nm, m, m / N, (m - loc) / N);
    };
    row("in-binary", a_loc);
    row("so, few kernels", a_few);
    row("so, many kernels", a_many);
    row("so, 45 DISTINCT", a_dist);
    row("in-binary, 2 threads", a_tp2);
    std::printf("  the second thread submitted %ld launches while that arm ran\n",
                other.load());
    return 0;
}
