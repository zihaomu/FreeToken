// Host cost of submitting a dispatch, and what capturing a fixed sequence saves.
//
// The device-side gap between dependent kernels is well under a microsecond on this part. In the
// MTP draft loop the host is on the critical path instead: each pass embeds the token the previous
// pass sampled, so nothing overlaps the submit and the device idles through it. The trunk runs
// over a thousand dispatches ahead of the device and never pays this. So this times the submit
// loop itself, separate from the sync.
//
// Every arm runs once per iteration in one loop: clock and scheduler drift between runs exceeds
// the differences being measured.
//
//   individual   45 separate hipLaunchKernelGGL, as an uncaptured pass issues them.
//   graph        one captured graph replayed. The floor; unreachable, since the pass is not static.
//   graph+Kset   replay after rewriting K nodes. About ten of the draft pass's 45 issues move an
//                operand pointer every step, and three (qsa_work, qsa_score, qsa_select) also
//                move a shape, because the QSA extents grow with context.
//   graph+Kg+s   the same, reading the params back from the node first, because the engine holds
//                no launch geometry (the shim marshals its own arguments). If this costs what
//                Kset costs, no ABI change is needed.
//   recap+updat  re-issue the whole pass under capture and hipGraphExecUpdate the exec. Needs no
//                knowledge of which arg moved, so it is the smallest engine change, if appending
//                nodes is cheap enough.
//
//   clang++ -x hip -O3 -std=c++17 --offload-arch=gfx1201 graphrate.hip -o graphrate
//   ./graphrate 45 300 10
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void tiny(float* p) { if (threadIdx.x == 0) p[blockIdx.x] = 1.0f; }

/* `tiny` with a real argument list, to separate HIP's argument marshalling from the shim's own
 * prologue (band checks, grid derivation, template switch) as the cause of an engine issue costing
 * several times `tiny`. The signature matches r4d_qsa_score_bf16: six pointers, nine scalars. */
__global__ void fat(const float* q, const float* bk, const int* bt, const int* cu, const int* nc,
                    float* out, int n, int heads, int nb, int nseq, int bpw, int q_ld, int bk_ld,
                    int bt_ld, int out_ld) {
    if (threadIdx.x == 0 && q && bk && bt && cu && nc)
        out[blockIdx.x] = (float)(n + heads + nb + nseq + bpw + q_ld + bk_ld + bt_ld + out_ld);
}

/* Slow enough to back the queue up. The other arms sync each iteration, so the command ring is
 * empty and a submit costs only the runtime's write. A serving engine runs far ahead of a busy
 * device, and a submit into a full ring waits for the device to retire a slot; if that dominates,
 * host-side shim tidying cannot help. */
__global__ void slow(float* p, int n) {
    float a = (float)threadIdx.x;
    for (int i = 0; i < n; ++i) a = fmaf(a, 1.0000001f, 1.0f);
    if (a == 12345.678f) p[blockIdx.x] = a;
}

using clk = std::chrono::steady_clock;
static double us_since(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

/* Median, not mean: a single descheduled iteration skews a mean. */
static double med(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    const int N     = argc > 1 ? std::atoi(argv[1]) : 45;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 300;
    const int K     = argc > 3 ? std::atoi(argv[3]) : 10;
    float* d = nullptr;
    if (hipMalloc(&d, 1024 * sizeof(float)) != hipSuccess) { std::printf("hipMalloc\n"); return 1; }
    hipStream_t s;
    if (hipStreamCreate(&s) != hipSuccess) { std::printf("hipStreamCreate\n"); return 1; }

    std::printf("%d kernels a sequence, %d iterations, %d nodes updated, medians\n", N, iters, K);

    hipGraph_t     g  = nullptr;
    hipGraphExec_t ge = nullptr;
    if (hipStreamBeginCapture(s, hipStreamCaptureModeGlobal) != hipSuccess) {
        std::printf("  capture unsupported\n"); return 0;
    }
    for (int i = 0; i < N; ++i) hipLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, s, d);
    if (hipStreamEndCapture(s, &g) != hipSuccess) { std::printf("  end capture\n"); return 1; }
    if (hipGraphInstantiate(&ge, g, nullptr, nullptr, 0) != hipSuccess) {
        std::printf("  instantiate\n"); return 1;
    }
    size_t nn = 0;
    if (hipGraphGetNodes(g, nullptr, &nn) != hipSuccess || nn == 0) {
        std::printf("  GetNodes\n"); return 1;
    }
    std::vector<hipGraphNode_t> nodes(nn);
    if (hipGraphGetNodes(g, nodes.data(), &nn) != hipSuccess) { std::printf("  nodes\n"); return 1; }
    const int k = std::min(K, (int)nn);
    std::vector<hipKernelNodeParams> cached(k);
    for (int i = 0; i < k; ++i)
        if (hipGraphKernelNodeGetParams(nodes[i], &cached[i]) != hipSuccess) {
            std::printf("  GetParams -- update arms unavailable\n"); return 1;
        }

    std::vector<double> a_emit;
    std::vector<double> a_frag;
    std::vector<double> a_ind, a_fat, a_err, a_gra, a_set, a_gs, a_rec, a_cap, a_upd;
    a_ind.reserve(iters);

    /* Warm the clocks, the runtime's lazy module load and every path below. The cards
     * idle-suspend, and a cold start times the clock governor ramping (up to ~2x). */
    for (int w = 0; w < 400; ++w) {
        for (int i = 0; i < N; ++i) hipLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, s, d);
        for (int i = 0; i < N; ++i)
            hipLaunchKernelGGL(fat, dim3(1), dim3(64), 0, s, d, d, (const int*)d,
                               (const int*)d, (const int*)d, d, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        hipStreamSynchronize(s);
        for (int i = 0; i < k; ++i) hipGraphExecKernelNodeSetParams(ge, nodes[i], &cached[i]);
        hipGraphLaunch(ge, s);
        hipStreamSynchronize(s);
        hipGraph_t g2 = nullptr;
        hipStreamBeginCapture(s, hipStreamCaptureModeGlobal);
        for (int i = 0; i < N; ++i) hipLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, s, d);
        hipStreamEndCapture(s, &g2);
        hipGraphExecUpdateResult r; hipGraphNode_t bad = nullptr;
        hipGraphExecUpdate(ge, g2, &bad, &r);
        hipGraphDestroy(g2);
    }

    /* Four graphs over the same 45 nodes: three all-reduces and a gather cut the real pass into
     * segments. A graph launch has a roughly fixed cost whatever it contains, paid per segment,
     * and that may eat the saving. */
    hipGraphExec_t gef[4] = {nullptr, nullptr, nullptr, nullptr};
    {
        const int seg[4] = {3, 17, 13, 12};
        for (int q = 0; q < 4; ++q) {
            hipGraph_t gq = nullptr;
            hipStreamBeginCapture(s, hipStreamCaptureModeGlobal);
            for (int i = 0; i < seg[q]; ++i)
                hipLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, s, d);
            hipStreamEndCapture(s, &gq);
            hipGraphInstantiate(&gef[q], gq, nullptr, nullptr, 0);
            hipGraphDestroy(gq);
        }
    }

    for (int it = 0; it < iters; ++it) {

        /* ---- individual, one argument */
        auto t0 = clk::now();
        for (int i = 0; i < N; ++i) hipLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, s, d);
        auto t1 = clk::now();
        hipStreamSynchronize(s);
        a_ind.push_back(us_since(t0, t1));

        /* ---- fifteen arguments plus the hipGetLastError check libr4d makes after every launch;
         * one op's path can make several. Prices that check against the plain 15-argument arm. */
        t0 = clk::now();
        for (int i = 0; i < N; ++i) {
            hipLaunchKernelGGL(fat, dim3(1), dim3(64), 0, s, d, d, (const int*)d,
                               (const int*)d, (const int*)d, d, 1, 2, 3, 4, 5, 6, 7, 8, 9);
            if (hipGetLastError() != hipSuccess) std::printf("  launch failed\n");
        }
        t1 = clk::now();
        hipStreamSynchronize(s);
        a_err.push_back(us_since(t0, t1));

        /* ---- fifteen arguments through hipLaunchKernel + kernelParams, as R4D_EMIT's launch path
         * does. Every ordinary dispatch pays this while a graph only pays back on replay, so if
         * it costs more than `fat` it loses more on the normal path than it saves on capture. */
        t0 = clk::now();
        for (int i = 0; i < N; ++i) {
            void* pv[15];
            void* p0 = d; const int* p1 = (const int*)d;
            int   i1 = 1, i2 = 2, i3 = 3, i4 = 4, i5 = 5, i6 = 6, i7 = 7, i8 = 8, i9 = 9;
            pv[0] = &p0; pv[1] = &p0; pv[2] = &p1; pv[3] = &p1; pv[4] = &p1; pv[5] = &p0;
            pv[6] = &i1; pv[7] = &i2; pv[8] = &i3; pv[9] = &i4; pv[10] = &i5;
            pv[11] = &i6; pv[12] = &i7; pv[13] = &i8; pv[14] = &i9;
            hipLaunchKernel((const void*)fat, dim3(1), dim3(64), pv, 0, s);
        }
        t1 = clk::now();
        hipStreamSynchronize(s);
        a_emit.push_back(us_since(t0, t1));

        /* ---- individual, fifteen arguments */
        t0 = clk::now();
        for (int i = 0; i < N; ++i)
            hipLaunchKernelGGL(fat, dim3(1), dim3(64), 0, s, d, d, (const int*)d,
                               (const int*)d, (const int*)d, d, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        t1 = clk::now();
        hipStreamSynchronize(s);
        a_fat.push_back(us_since(t0, t1));

        /* ---- the same 45 nodes as four graphs, plus the five dispatches the cuts leave behind */
        t0 = clk::now();
        for (int q = 0; q < 4; ++q) hipGraphLaunch(gef[q], s);
        for (int i = 0; i < 5; ++i) hipLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, s, d);
        t1 = clk::now();
        hipStreamSynchronize(s);
        a_frag.push_back(us_since(t0, t1));

        /* ---- graph replay */

        t0 = clk::now();
        hipGraphLaunch(ge, s);
        t1 = clk::now();
        hipStreamSynchronize(s);
        a_gra.push_back(us_since(t0, t1));

        /* ---- graph + K set */
        t0 = clk::now();
        for (int i = 0; i < k; ++i) hipGraphExecKernelNodeSetParams(ge, nodes[i], &cached[i]);
        hipGraphLaunch(ge, s);
        t1 = clk::now();
        hipStreamSynchronize(s);
        a_set.push_back(us_since(t0, t1));

        /* ---- graph + K get+set */
        t0 = clk::now();
        for (int i = 0; i < k; ++i) {
            hipKernelNodeParams p;
            hipGraphKernelNodeGetParams(nodes[i], &p);
            hipGraphExecKernelNodeSetParams(ge, nodes[i], &p);
        }
        hipGraphLaunch(ge, s);
        t1 = clk::now();
        hipStreamSynchronize(s);
        a_gs.push_back(us_since(t0, t1));

        /* ---- recapture + ExecUpdate */
        t0 = clk::now();
        hipGraph_t g2 = nullptr;
        hipStreamBeginCapture(s, hipStreamCaptureModeGlobal);
        for (int i = 0; i < N; ++i) hipLaunchKernelGGL(tiny, dim3(1), dim3(64), 0, s, d);
        hipStreamEndCapture(s, &g2);
        auto tc = clk::now();
        hipGraphExecUpdateResult r; hipGraphNode_t bad = nullptr;
        const hipError_t e = hipGraphExecUpdate(ge, g2, &bad, &r);
        auto tu = clk::now();
        hipGraphLaunch(ge, s);
        t1 = clk::now();
        hipStreamSynchronize(s);
        hipGraphDestroy(g2);
        if (e != hipSuccess && it == 0) std::printf("  ExecUpdate says %d\n", (int)e);
        a_cap.push_back(us_since(t0, tc));
        a_upd.push_back(us_since(tc, tu));
        a_rec.push_back(us_since(t0, t1));
    }

    /* ---- Submit under backpressure: the device cannot keep up, so the ring stays full. */
    std::vector<double> a_bp;
    for (int w = 0; w < 20; ++w) {
        for (int i = 0; i < N; ++i) hipLaunchKernelGGL(slow, dim3(8), dim3(256), 0, s, d, 4000);
        hipStreamSynchronize(s);
    }
    for (int it = 0; it < iters; ++it) {
        /* No sync: the queue stays deep across iterations, as in a decode step. */
        const auto t0 = clk::now();
        for (int i = 0; i < N; ++i) hipLaunchKernelGGL(slow, dim3(8), dim3(256), 0, s, d, 4000);
        const auto t1 = clk::now();
        a_bp.push_back(us_since(t0, t1));
    }
    hipStreamSynchronize(s);

    const double ind = med(a_ind);
    auto row = [&](const char* nm, std::vector<double>& v) {
        const double m = med(v);
        std::printf("  %-12s submit %7.1f us  (%5.2f us a dispatch)   %+6.1f us vs individual\n",
                    nm, m, m / N, m - ind);
    };
    row("individual", a_ind);
    row("ind 15 args", a_fat);
    row("emit 15 args", a_emit);
    row("+GetLastError", a_err);
    row("graph", a_gra);
    row("4 graphs+5", a_frag);
    row("graph+set", a_set);
    row("graph+get+set", a_gs);
    row("recap+update", a_rec);
    row("full ring", a_bp);
    std::printf("  %-12s of which capture %6.1f us (%4.2f us a node), ExecUpdate %6.1f us\n",
                "", med(a_cap), med(a_cap) / N, med(a_upd));
    return 0;
}
