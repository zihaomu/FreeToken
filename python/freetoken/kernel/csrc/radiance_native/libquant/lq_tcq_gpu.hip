/* lq_tcq_gpu.hip -- GPTQ over a trellis grid on the card. See lq_gpu.h.
 *
 * One workgroup is one run of tcq_rows rows, and takes a 128-column block's columns in order as
 * the host loop does (lq_gptq.cpp, gptq_quantise_tcq): for each column the seam window's free
 * Viterbi, the tail-biting search from and to its state, the codes, the decoded values, the errors
 * and their feedback into the block's later columns. A second kernel applies the block's errors to
 * the columns after it (gptq_tail's arithmetic). The host chooses the next block's dyn=1 scales
 * from its columns while the card finishes the rest of the tail.
 *
 * The states are spread so the threads holding a group hold all its predecessors: state
 * st = q + t * Q with Q = 2^(L-k), t < 2^k, the group q's t-th, sits with q's lanes (one, or for
 * the wide codes a few adjacent ones, each with a share of t). A step is then each group's
 * cheapest predecessor (within a thread or across its lanes), one barrier, and each state's cost
 * from its own group's cheapest, read from LDS.
 */
#include "lq_gpu.h"
#include "lq_gptq.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace lq {

namespace {

std::vector<int>& devices() {
    static std::vector<int> d;
    return d;
}

/* tcq_viterbi's cost of a state no path may start in. */
constexpr float kUnreached = 3.0e38f;

/* Everything one block launch reads, for the device's rows [0, nl). */
struct TcqArgs {
    float*         w;       /* [nl, K] the weight as the loop has left it */
    int32_t*       codes;   /* [nl, K] */
    float*         err;     /* [nl, 128] this block's errors, or null without a Hessian */
    const float*   u;       /* [K, K] the factor, or null */
    const float*   lut;     /* [2^L] the states' values */
    const float*   s;       /* [nl] the scale each row's values take in this block */
    const float*   rt;      /* [nl] tcq_rb's ratio, or null */
    uint8_t*       back;    /* per run, [rows, Q] each group's cheapest predecessor */
    int64_t        K, nl, b0, b1;
    int            T;
    bool           fused;   /* tcq_rb's rounding of the exact product (a computed code) */
};

/* A thread's E states' table values. With PACK they sit four to a register as the int8 they are
 * (the table holds integers in -127..127), for the wide forms. */
template <int E, int PACK>
struct Lut {
    /* PACK 1: four int8 a register; 2: two int16 a register, for a wide code (tcq_code=mad16,
     * values past int8); 0: one float a state. */
    static constexpr int NP = PACK == 2 ? E / 2 : PACK ? E / 4 : 1;
    float    f[PACK ? 1 : E];
    uint32_t p[NP];
    __device__ void set(int e, float v) {
        if (PACK == 2) {
            const uint32_t b = (uint32_t)(uint16_t)(int16_t)(int)v;
            if (e % 2 == 0) p[e / 2] = b;
            else p[e / 2] |= b << 16;
        } else if (PACK) {
            const uint32_t b = (uint32_t)(uint8_t)(int8_t)(int)v;
            if (e % 4 == 0) p[e / 4] = b;
            else p[e / 4] |= b << (8 * (e % 4));
        } else {
            f[e] = v;
        }
    }
    __device__ float at(int e) const {
        if (PACK == 2) return (float)(int16_t)(p[e / 2] >> (16 * (e % 2)));
        if (PACK) return (float)(int8_t)(p[e / 4] >> (8 * (e % 4)));
        return f[e];
    }
};

/* A workgroup barrier that orders LDS alone: __syncthreads() also waits for every global store to
 * land and invalidates the vector cache. */
__device__ inline void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

/* (cost, state) a is cheaper than b: the lower cost, a NaN above every number, then the lower
 * state. A total order, so any reduction order finds tcq_viterbi's state. */
__device__ inline bool cheaper(float ac, int ai, float bc, int bi) {
    const bool an = ac != ac, bn = bc != bc;
    if (an != bn) return bn;
    if (an) return ai < bi;
    return ac < bc || (ac == bc && ai < bi);
}

/* One run's block of columns for 2^L states and KB-bit codes: M groups a thread, SH lanes a group,
 * the table packed or not, WPE waves an EU (the compiler's register budget, 1 for its own), HR
 * tcq_rb, NB the groups' cheapest held in one LDS buffer or two (one: a second barrier a step, for
 * the state counts whose two do not fit). The geometry is compile-time, so a state's LDS address
 * is an immediate offset from the thread's. */
template <int L, int KB, int M, int SH, int PACK, int WPE, bool HR, int NB = 2>
__global__ void __launch_bounds__((1 << (L - KB)) * SH / M)
__attribute__((amdgpu_waves_per_eu(WPE))) tcq_block(TcqArgs a) {
    /* SH lanes share a group's predecessors, BT each; a thread holds M groups' worth. */
    constexpr int B = 1 << KB, Q = 1 << (L - KB), TH = Q * SH / M, BT = B / SH, E = M * BT;
    constexpr int G = TH / SH;                    /* groups a slot of the thread's */
    /* The group of state st is st >> KB; it splits into the thread's part and a constant when
     * G and Q are multiples of B. */
    constexpr bool split = G % B == 0 && Q % B == 0;
    extern __shared__ float lds[];
    const int j = (int)threadIdx.x, gj = j / SH, hj = j % SH;
    const int64_t K = a.K;
    float* bestl = lds;                           /* [NB][Q] */
    float* xs = bestl + NB * Q;                   /* [T] the column */
    float* ss = xs + a.T;                         /* [T] */
    float* rs = ss + a.T;                         /* [T] */
    float* es = rs + a.T;                         /* [T] the rows' errors */
    int* path = (int*)(es + a.T);                 /* [T] */
    float* redc = (float*)(path + a.T);           /* [32] */
    int* redi = (int*)(redc + 32);                /* [32] */

    const int64_t n0 = (int64_t)blockIdx.x * a.T;
    const int n = (int)(a.nl - n0 < a.T ? a.nl - n0 : a.T);
    uint8_t* back = a.back + (size_t)blockIdx.x * (size_t)a.T * (size_t)Q;
    constexpr int c = (L + KB - 1) / KB;
    const int h = 8 * c < n / 2 ? 8 * c : n / 2;
    /* The thread's state e: group gj + (e / BT) * G, predecessor hj * BT + e % BT. */
    auto state = [&](int e) { return gj + (e / BT) * G + (hj * BT + e % BT) * Q; };
    auto group = [&](int e) {
        if constexpr (split)
            return (gj >> KB) + hj * BT * (Q / B) + ((e / BT) * G + (e % BT) * Q) / B;
        else return state(e) >> KB;
    };

    Lut<E, PACK> lv;
#pragma unroll
    for (int e = 0; e < E; ++e) lv.set(e, a.lut[state(e)]);
    float cost[E];
    int buf = 0;

    /* A state's value in row r: s * lut, or under tcq_rb s * rint(lut * rt) (a computed code's
     * rounding of the exact product, rb_round). */
    auto value = [&](int e, float s, float rt) {
        return s * (HR ? (a.fused ? __builtin_fmaf(lv.at(e), rt, 12582912.0f) - 12582912.0f
                                  : __builtin_rintf(lv.at(e) * rt))
                       : lv.at(e));
    };
    /* Row r's first costs: from any state (start < 0), or from predecessor state `start`. */
    auto first = [&](int r, int start) {
        const float x = xs[r], s = ss[r], rt = HR ? rs[r] : 0.0f;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const float d = x - value(e, s, rt);
            cost[e] = start < 0 || group(e) == (start & (Q - 1)) ? d * d : kUnreached;
        }
    };
    /* One trellis step into row r (tcq_step): each group's cheapest predecessor, the first on a
     * tie, into LDS and `from`; then each state's squared error plus its group's cheapest.
     * Shared by SH lanes, the first scans from its first predecessor as tcq_step does, the others
     * from +inf, which no cost is below, and the lanes combine in predecessor order taking the
     * later only when strictly cheaper: the first of the lowest, as one scan would find. */
    auto step = [&](int r, uint8_t* from) {
        float* bl = bestl + (NB == 2 ? buf : 0) * Q;
        /* one buffer: every thread has read the last step's before any writes this one's */
        if constexpr (NB == 1) lds_barrier();
        /* A packed table is unpacked where it is used: hoisted out of the row loop it would hold
         * a register a state again. */
        if constexpr (PACK) {
#pragma unroll
            for (int q = 0; q < Lut<E, PACK>::NP; ++q) asm volatile("" : "+v"(lv.p[q]));
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float b = SH == 1 || hj == 0 ? cost[m * BT] : __builtin_inff();
            int bt = SH == 1 || hj == 0 ? 0 : -1;
#pragma unroll
            for (int t = SH == 1 ? 1 : 0; t < BT; ++t) {
                const bool lt = cost[m * BT + t] < b;
                b = lt ? cost[m * BT + t] : b;
                bt = lt ? hj * BT + t : bt;
            }
#pragma unroll
            for (int o = 1; o < SH; o <<= 1) {
                const float ob = __shfl_xor(b, o);
                const int obt = __shfl_xor(bt, o);
                const bool hi = (j & o) != 0;
                const float lb = hi ? ob : b, hb = hi ? b : ob;
                const int lt = hi ? obt : bt, ht = hi ? bt : obt;
                b = hb < lb ? hb : lb;
                bt = hb < lb ? ht : lt;
            }
            if (SH == 1 || hj == 0) {
                bl[gj + m * G] = b;
                from[gj + m * G] = (uint8_t)bt;
            }
        }
        /* The step exchanges only LDS; `from` is read after the run's last step, behind a full
         * barrier, so the stores to it need not land before this one. */
        lds_barrier();
        const float x = xs[r], s = ss[r], rt = HR ? rs[r] : 0.0f;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const float d = x - value(e, s, rt);
            cost[e] = d * d + bl[group(e)];
        }
        buf ^= 1;
    };
    auto prev = [&](int st, int r) {
        const int q = st >> KB;
        return q + (int)back[(size_t)r * Q + q] * Q;
    };

    for (int64_t i = a.b0; i < a.b1; ++i) {
        for (int r = j; r < n; r += TH) {
            xs[r] = a.w[(n0 + r) * K + i];
            ss[r] = a.s[n0 + r];
            if (HR) rs[r] = a.rt[n0 + r];
        }
        __syncthreads();

        /* The seam: the best free path over the run's last h rows then its first h, and its
         * state at the seam (grid_tcq_run). Window row ii is run row ii < h ? n - h + ii : ii - h. */
        first(n - h, -1);
        for (int ii = 1; ii < 2 * h; ++ii) step(ii < h ? n - h + ii : ii - h, back + (size_t)ii * Q);
        {
            float bc = cost[0];
            int bi = state(0);
#pragma unroll
            for (int e = 1; e < E; ++e)
                if (cheaper(cost[e], state(e), bc, bi)) { bc = cost[e]; bi = state(e); }
            for (int o = warpSize / 2; o > 0; o /= 2) {
                const float oc = __shfl_xor(bc, o);
                const int oi = __shfl_xor(bi, o);
                if (cheaper(oc, oi, bc, bi)) { bc = oc; bi = oi; }
            }
            if (j % warpSize == 0) { redc[j / warpSize] = bc; redi[j / warpSize] = bi; }
            /* tcq_viterbi scans from state 0 and moves only to a strictly lower cost, so a NaN at
             * state 0 keeps it; state 0 is thread 0's e = 0. */
            const bool nan0 = j == 0 && cost[0] != cost[0];
            __syncthreads();
            if (j == 0) {
                for (int wv = 1; wv < (TH + warpSize - 1) / warpSize; ++wv)
                    if (cheaper(redc[wv], redi[wv], bc, bi)) { bc = redc[wv]; bi = redi[wv]; }
                int st = nan0 ? 0 : bi;
                for (int r = 2 * h - 1; r >= h; --r) st = prev(st, r);
                path[0] = st;
            }
            __syncthreads();
        }
        const int seam = path[0];
        __syncthreads();

        /* The run, from and to the seam's state. */
        first(0, seam);
        for (int r = 1; r < n; ++r) step(r, back + (size_t)r * Q);
        __syncthreads();
        if (j == 0) {
            int st = seam;
            path[n - 1] = st;
            for (int r = n - 1; r > 0; --r) {
                st = prev(st, r);
                path[r - 1] = st;
            }
        }
        __syncthreads();

        for (int r = j; r < n; r += TH) {
            const int st = path[r];
            const float v = a.lut[st];
            const float dq = (HR ? (a.fused ? __builtin_fmaf(v, rs[r], 12582912.0f) - 12582912.0f
                                            : __builtin_rintf(v * rs[r]))
                                 : v) * ss[r];
            a.codes[(n0 + r) * K + i] = st & (B - 1);
            a.w[(n0 + r) * K + i] = dq;
            if (a.u) {
                const float e = (xs[r] - dq) / a.u[i * K + i];
                a.err[(n0 + r) * 128 + (i - a.b0)] = e;
                es[r] = e;
            }
        }
        __syncthreads();
        if (a.u && i + 1 < a.b1) {
            const int cols = (int)(a.b1 - i - 1);
            const float* ur = a.u + i * K + i + 1;
            for (int x = j; x < n * cols; x += TH) {
                const int r = x / cols, cc = x % cols;
                float* p = a.w + (n0 + r) * K + i + 1 + cc;
                *p = *p - es[r] * ur[cc];
            }
        }
        __syncthreads();
    }
}

/* gptq_tail on the card: w[r][j] -= e[r][t] * u[b0 + t][j] for t < nb, j over [j0, j1), each
 * element's terms one at a time in ascending t, a product then a subtraction. */
constexpr int kTailRows = 32, kTailCols = 256;
__global__ void __launch_bounds__(kTailCols) tcq_tail(float* w, const float* err, const float* u,
                                                      int64_t K, int64_t nl, int64_t j0, int64_t j1,
                                                      int64_t b0, int nb) {
    __shared__ float e[kTailRows][128];
    const int64_t r0 = (int64_t)blockIdx.y * kTailRows;
    const int rows = (int)(nl - r0 < kTailRows ? nl - r0 : kTailRows);
    for (int x = (int)threadIdx.x; x < kTailRows * 128; x += kTailCols) {
        const int r = x / 128, t = x % 128;
        e[r][t] = r < rows && t < nb ? err[(r0 + r) * 128 + t] : 0.0f;
    }
    __syncthreads();
    const int64_t jj = j0 + (int64_t)blockIdx.x * kTailCols + threadIdx.x;
    if (jj >= j1) return;
    float acc[kTailRows];
#pragma unroll
    for (int r = 0; r < kTailRows; ++r) acc[r] = r < rows ? w[(r0 + r) * K + jj] : 0.0f;
    for (int t = 0; t < nb; ++t) {
        const float ut = u[(b0 + t) * K + jj];
#pragma unroll
        for (int r = 0; r < kTailRows; ++r) acc[r] = acc[r] - e[r][t] * ut;
    }
#pragma unroll
    for (int r = 0; r < kTailRows; ++r)
        if (r < rows) w[(r0 + r) * K + jj] = acc[r];
}

using BlockKernel = void (*)(TcqArgs);

/* The forms built, each the fastest of its sweep over groups a thread (M), lanes a group (SH), a
 * packed table and an occupancy target (R9700, a 96-run weight of 5120 columns): fewer threads a
 * run let more runs share a WGP, until the states a thread holds spill. 1024 threads at most, the
 * wide codes' groups split over lanes. The 2^10-state forms are the default sizing, for tests. */
/* `wide`: the forms for a code whose values pass int8 (tcq_code=mad16), the table packed two int16
 * a register. */
struct Form { int L, k; BlockKernel plain, rb; int threads, nb; bool wide; };
#define LQ_FORM(l, kb, m, sh, pack, wpe)                                                         \
    { l, kb, tcq_block<l, kb, m, sh, pack, wpe, false>, tcq_block<l, kb, m, sh, pack, wpe, true>,  \
      (1 << (l - kb)) * sh / m, 2, (pack) == 2 }
#define LQ_FORM1(l, kb, m, sh, pack, wpe)                                                        \
    { l, kb, tcq_block<l, kb, m, sh, pack, wpe, false, 1>,                                        \
      tcq_block<l, kb, m, sh, pack, wpe, true, 1>, (1 << (l - kb)) * sh / m, 1, (pack) == 2 }
const Form kForms[] = {
    LQ_FORM(10, 2, 1, 1, false, 1), LQ_FORM(10, 3, 2, 1, false, 1), LQ_FORM(10, 4, 1, 1, false, 1),
    LQ_FORM(10, 5, 1, 1, false, 1),
    LQ_FORM(12, 2, 4, 1, false, 16), LQ_FORM(12, 3, 2, 1, false, 1), LQ_FORM(12, 4, 1, 1, false, 1),
    LQ_FORM(12, 5, 1, 1, false, 1), LQ_FORM(12, 6, 1, 2, false, 1), LQ_FORM(12, 7, 1, 8, false, 1),
    LQ_FORM(14, 2, 8, 1, false, 12), LQ_FORM(14, 3, 4, 1, false, 12), LQ_FORM(14, 4, 2, 1, false, 12),
    LQ_FORM(14, 5, 1, 1, false, 12), LQ_FORM(14, 6, 1, 2, false, 12), LQ_FORM(14, 7, 1, 2, true, 1),
    /* 2^16 states: a run's two cost buffers fit the LDS from 4-bit codes up, one at 3 bits; 2^15
     * states at 2 bits in one; not yet swept */
    LQ_FORM1(15, 2, 8, 1, true, 1), LQ_FORM1(16, 3, 8, 1, true, 1), LQ_FORM(16, 4, 4, 1, true, 1), LQ_FORM(16, 5, 2, 1, true, 1), LQ_FORM(16, 6, 1, 1, true, 1),
    LQ_FORM(16, 7, 1, 2, true, 1),
    /* the wide code's forms: the sweep's shapes, the table at int16 */
    LQ_FORM1(16, 3, 8, 1, 2, 1), LQ_FORM(16, 4, 4, 1, 2, 1), LQ_FORM(16, 5, 2, 1, 2, 1),
    LQ_FORM(16, 6, 1, 1, 2, 1), LQ_FORM(16, 7, 1, 2, 2, 1),
};
#undef LQ_FORM
#undef LQ_FORM1

const Form* pick_form(int L, int k, bool wide) {
    for (const Form& f : kForms)
        if (f.L == L && f.k == k && f.wide == wide) return &f;
    return nullptr;
}

std::string hip_fail(const char* what, hipError_t e) {
    return std::string(what) + ": " + hipGetErrorString(e);
}

#define LQ_HIP(x)                                                                                 \
    do {                                                                                          \
        const hipError_t e_ = (x);                                                                \
        if (e_ != hipSuccess) { *why = hip_fail(#x, e_); return RAD_E_DEVICE; }                   \
    } while (0)

/* A device allocation freed with its scope. */
template <typename T>
struct DevBuf {
    T* p = nullptr;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { if (p) (void)hipFree(p); }
    hipError_t alloc(size_t n) { return hipMalloc((void**)&p, n * sizeof(T) + (n ? 0 : 1)); }
};

/* Page-locked host memory freed with its scope. Every copy goes through it: the runtime's own
 * staging for pageable memory gave wrong bytes under copies from several threads to two cards at
 * once (two weights each split over both). */
template <typename T>
struct HostBuf {
    T* p = nullptr;
    HostBuf() = default;
    HostBuf(const HostBuf&) = delete;
    HostBuf& operator=(const HostBuf&) = delete;
    ~HostBuf() { if (p) (void)hipHostFree(p); }
    hipError_t alloc(size_t n) { return hipHostMalloc((void**)&p, n * sizeof(T) + (n ? 0 : 1)); }
};

/* `bytes` from host `src` to device `dst` (or back with `down`) through `bounce`, kBounce bytes of
 * pinned memory, a chunk at a time on stream `s`. */
constexpr size_t kBounce = (size_t)32 << 20;
hipError_t bounce_copy(void* dst, const void* src, size_t bytes, bool down, uint8_t* bounce,
                       hipStream_t s) {
    for (size_t off = 0; off < bytes; off += kBounce) {
        const size_t n = std::min(kBounce, bytes - off);
        hipError_t e;
        if (down) {
            e = hipMemcpyAsync(bounce, (const uint8_t*)src + off, n, hipMemcpyDeviceToHost, s);
            if (e == hipSuccess) e = hipStreamSynchronize(s);
            if (e != hipSuccess) return e;
            std::memcpy((uint8_t*)dst + off, bounce, n);
        } else {
            std::memcpy(bounce, (const uint8_t*)src + off, n);
            e = hipMemcpyAsync((uint8_t*)dst + off, bounce, n, hipMemcpyHostToDevice, s);
            if (e == hipSuccess) e = hipStreamSynchronize(s);
            if (e != hipSuccess) return e;
        }
    }
    return hipSuccess;
}

struct Streams {
    hipStream_t s0 = nullptr, s1 = nullptr;
    hipEvent_t  next = nullptr, down = nullptr, rest = nullptr;
    ~Streams() {
        if (s0) (void)hipStreamSynchronize(s0);
        if (s1) (void)hipStreamSynchronize(s1);
        for (hipEvent_t e : { next, down, rest })
            if (e) (void)hipEventDestroy(e);
        for (hipStream_t s : { s0, s1 })
            if (s) (void)hipStreamDestroy(s);
    }
};

/* Rows [ra, rb) of the weight on card `dev`: ra and rb are multiples of tcq_rows (rb may be N). */
int run_rows(const Grid& g, const Form& f, float* w, int32_t* codes, Scales& scales, int64_t ra,
             int64_t rb, int64_t K, const float* u, bool dynamic, int dev, std::string* why) {
    LQ_HIP(hipSetDevice(dev));
    const int T = (int)g.tcq_rows, L = g.tcq_L, k = g.tcq_k;
    const int Q = 1 << (L - k), TH = f.threads;
    int lanes = 0;
    LQ_HIP(hipDeviceGetAttribute(&lanes, hipDeviceAttributeWarpSize, dev));
    if (lanes <= 0 || TH % lanes) {
        *why = "tcq=" + std::to_string(L) + " with " + std::to_string(k) + "-bit codes is " +
               std::to_string(TH) + " threads a run, not whole waves of " + std::to_string(lanes);
        return RAD_E_UNSUPPORTED;
    }
    const int64_t nl = rb - ra, runs = (nl + T - 1) / T, nb = (K + 127) / 128;
    const size_t lds = (size_t)(f.nb * Q + 5 * T + 64) * sizeof(float);

    DevBuf<float> dw, derr, du, dlut, ds, drt;
    DevBuf<int32_t> dcodes;
    DevBuf<uint8_t> dback;
    LQ_HIP(dw.alloc((size_t)(nl * K)));
    LQ_HIP(dcodes.alloc((size_t)(nl * K)));
    LQ_HIP(dlut.alloc(g.tcq_lut.size()));
    LQ_HIP(ds.alloc((size_t)(nl * nb)));
    if (g.tcq_rb) LQ_HIP(drt.alloc((size_t)(nl * nb)));
    LQ_HIP(dback.alloc((size_t)runs * (size_t)T * (size_t)Q));
    if (u) {
        LQ_HIP(derr.alloc((size_t)(2 * nl * 128)));
        LQ_HIP(du.alloc((size_t)(K * K)));
    }
    Streams st;
    LQ_HIP(hipStreamCreateWithFlags(&st.s0, hipStreamNonBlocking));
    LQ_HIP(hipStreamCreateWithFlags(&st.s1, hipStreamNonBlocking));
    LQ_HIP(hipEventCreateWithFlags(&st.next, hipEventDisableTiming));
    LQ_HIP(hipEventCreateWithFlags(&st.down, hipEventDisableTiming));
    LQ_HIP(hipEventCreateWithFlags(&st.rest, hipEventDisableTiming));

    HostBuf<uint8_t> bounce;
    HostBuf<float> hs, hrt, hblk;
    LQ_HIP(bounce.alloc(kBounce));
    LQ_HIP(hs.alloc((size_t)(nl * nb)));
    if (g.tcq_rb) LQ_HIP(hrt.alloc((size_t)(nl * nb)));
    if (dynamic) LQ_HIP(hblk.alloc((size_t)nl * 128));
    LQ_HIP(bounce_copy(dw.p, w + ra * K, (size_t)(nl * K) * sizeof(float), false, bounce.p, st.s0));
    LQ_HIP(bounce_copy(dlut.p, g.tcq_lut.data(), g.tcq_lut.size() * sizeof(float), false,
                       bounce.p, st.s0));
    if (u)
        LQ_HIP(bounce_copy(du.p, u, (size_t)(K * K) * sizeof(float), false, bounce.p, st.s0));

    std::vector<float> dummy(g.tcq_rb ? 0 : (size_t)T);
    int64_t down_c0 = 0, down_c1 = 0;   /* the columns the last block's download holds */
    bool rest_pending = false;
    for (int64_t bi = 0; bi < nb; ++bi) {
        const int64_t b0 = bi * 128, b1 = std::min(K, b0 + 128);
        if (dynamic) {
            /* The block's columns as the card left them, then each row's scale from them. */
            if (bi > 0) {
                LQ_HIP(hipEventSynchronize(st.down));
                const int64_t wc = down_c1 - down_c0;
                for (int64_t r = 0; r < nl; ++r)
                    std::memcpy(w + (ra + r) * K + down_c0, hblk.p + r * 128,
                                (size_t)wc * sizeof(float));
            }
            gptq_tcq_dyn_scales(g, w, K, scales, ra, rb, b0, b1);
        }
        float* hsb = hs.p + bi * nl;
        float* hrb = g.tcq_rb ? hrt.p + bi * nl : nullptr;
        for (int64_t r = ra; r < rb; r += T) {
            const int64_t n = std::min(rb, r + T) - r;
            gptq_tcq_col_scales(g, scales, r, n, b0, hsb + (r - ra),
                                hrb ? hrb + (r - ra) : dummy.data());
        }
        LQ_HIP(hipMemcpyAsync(ds.p + bi * nl, hsb, (size_t)nl * sizeof(float),
                              hipMemcpyHostToDevice, st.s0));
        if (hrb)
            LQ_HIP(hipMemcpyAsync(drt.p + bi * nl, hrb, (size_t)nl * sizeof(float),
                                  hipMemcpyHostToDevice, st.s0));

        TcqArgs a{};
        a.w = dw.p;
        a.codes = dcodes.p;
        a.err = u ? derr.p + (bi & 1) * nl * 128 : nullptr;
        a.u = u ? du.p : nullptr;
        a.lut = dlut.p;
        a.s = ds.p + bi * nl;
        a.rt = hrb ? drt.p + bi * nl : nullptr;
        a.back = dback.p;
        a.K = K;
        a.nl = nl;
        a.b0 = b0;
        a.b1 = b1;
        a.T = T;
        a.fused = g.tcq_mad;
        void* args[] = { &a };
        LQ_HIP(hipLaunchKernel((const void*)(hrb ? f.rb : f.plain), dim3((unsigned)runs),
                               dim3((unsigned)TH), args, lds, st.s0));
        if (!u || b1 >= K) continue;

        /* The tail: the next block's columns first, so the host can take them while the card
         * applies the rest. The rest of block bi and the next block's run touch different
         * columns; the next block's first part waits for it. */
        const int64_t c1 = std::min(K, b1 + 128);
        const dim3 tb((unsigned)((c1 - b1 + kTailCols - 1) / kTailCols),
                      (unsigned)((nl + kTailRows - 1) / kTailRows));
        if (rest_pending) LQ_HIP(hipStreamWaitEvent(st.s0, st.rest, 0));
        hipLaunchKernelGGL(tcq_tail, tb, dim3(kTailCols), 0, st.s0, dw.p, a.err, du.p, K, nl, b1,
                           c1, b0, (int)(b1 - b0));
        LQ_HIP(hipGetLastError());
        LQ_HIP(hipEventRecord(st.next, st.s0));
        LQ_HIP(hipStreamWaitEvent(st.s1, st.next, 0));
        if (dynamic) {
            LQ_HIP(hipMemcpy2DAsync(hblk.p, 128 * sizeof(float), dw.p + b1,
                                    (size_t)K * sizeof(float), (size_t)(c1 - b1) * sizeof(float),
                                    (size_t)nl, hipMemcpyDeviceToHost, st.s1));
            LQ_HIP(hipEventRecord(st.down, st.s1));
            down_c0 = b1;
            down_c1 = c1;
        }
        if (c1 < K) {
            const dim3 rbk((unsigned)((K - c1 + kTailCols - 1) / kTailCols),
                           (unsigned)((nl + kTailRows - 1) / kTailRows));
            hipLaunchKernelGGL(tcq_tail, rbk, dim3(kTailCols), 0, st.s1, dw.p, a.err, du.p, K, nl,
                               c1, K, b0, (int)(b1 - b0));
            LQ_HIP(hipGetLastError());
        }
        LQ_HIP(hipEventRecord(st.rest, st.s1));
        rest_pending = true;
    }
    LQ_HIP(hipStreamSynchronize(st.s1));
    LQ_HIP(hipStreamSynchronize(st.s0));
    LQ_HIP(bounce_copy(w + ra * K, dw.p, (size_t)(nl * K) * sizeof(float), true, bounce.p, st.s0));
    LQ_HIP(bounce_copy(codes + ra * K, dcodes.p, (size_t)(nl * K) * sizeof(int32_t), true,
                       bounce.p, st.s0));
    return RAD_OK;
}

}  /* namespace */

int gpu_set_devices(const int* ids, int n) {
    int count = 0;
    if (n > 0 && hipGetDeviceCount(&count) != hipSuccess) count = 0;
    for (int i = 0; i < n; ++i)
        if (ids[i] < 0 || ids[i] >= count) {
            std::fprintf(stderr, "libquant: card %d is not there (%d visible)\n", ids[i], count);
            return RAD_E_DEVICE;
        }
    devices().assign(ids, ids + (n > 0 ? n : 0));
    return RAD_OK;
}

bool gpu_enabled() { return !devices().empty(); }

int gptq_tcq_gpu(const Grid& g, float* w, int32_t* codes, Scales& scales, int64_t N, int64_t K,
                 const float* u, bool dynamic, std::string* why) {
    const Form* f = pick_form(g.tcq_L, g.tcq_k, g.tcq_wide);
    const int64_t T = g.tcq_rows;
    if (!f) {
        *why = "the card has no trellis kernel for tcq=" + std::to_string(g.tcq_L) + " with " +
               std::to_string(g.tcq_k) + "-bit codes";
        return RAD_E_UNSUPPORTED;
    }
    const int Q = 1 << (g.tcq_L - g.tcq_k);
    if ((size_t)(f->nb * Q + 5 * T + 64) * sizeof(float) > 65536) {
        *why = "a run's working set does not fit the card's 64 KB of LDS (tcq_rows=" +
               std::to_string(T) + ")";
        return RAD_E_UNSUPPORTED;
    }
    if (N % T == 1) {
        *why = "a run of one row (rows " + std::to_string(N) + ", tcq_rows " + std::to_string(T) +
               ") has no seam";
        return RAD_E_UNSUPPORTED;
    }
    /* A 128-column block's scale is one per row (gptq_tcq_col_scales at the block's first
     * column stands for all its columns). */
    if (!(scales.bc % 128 == 0 || scales.bc >= K)) {
        *why = "the card takes scale groups of a multiple of 128 columns, not " +
               std::to_string(scales.bc);
        return RAD_E_UNSUPPORTED;
    }

    /* A weight of many runs is split across the cards; a small one goes whole to the next card
     * in turn, so weights quantised side by side spread over them. */
    const std::vector<int>& dv = devices();
    const int64_t runs = (N + T - 1) / T;
    std::vector<std::pair<int64_t, int64_t>> part;
    std::vector<int> on;
    if (dv.size() > 1 && runs >= 32) {
        for (size_t d = 0; d < dv.size(); ++d) {
            const int64_t a = runs * (int64_t)d / (int64_t)dv.size() * T;
            const int64_t b = std::min(N, runs * (int64_t)(d + 1) / (int64_t)dv.size() * T);
            if (b > a) { part.emplace_back(a, b); on.push_back(dv[d]); }
        }
    } else {
        static std::atomic<unsigned> turn{0};
        part.emplace_back(0, N);
        on.push_back(dv[turn.fetch_add(1, std::memory_order_relaxed) % dv.size()]);
    }
    std::vector<int> st(part.size(), RAD_OK);
    std::vector<std::string> wy(part.size());
    if (part.size() == 1) {
        st[0] = run_rows(g, *f, w, codes, scales, 0, N, K, u, dynamic, on[0], &wy[0]);
    } else {
        std::vector<std::thread> th;
        for (size_t d = 0; d < part.size(); ++d)
            th.emplace_back([&, d] {
                st[d] = run_rows(g, *f, w, codes, scales, part[d].first, part[d].second, K, u,
                                 dynamic, on[d], &wy[d]);
            });
        for (std::thread& t : th) t.join();
    }
    for (size_t d = 0; d < part.size(); ++d)
        if (st[d] != RAD_OK) {
            *why = "card " + std::to_string(on[d]) + ": " + wy[d];
            return st[d];
        }
    return RAD_OK;
}

}  /* namespace lq */
