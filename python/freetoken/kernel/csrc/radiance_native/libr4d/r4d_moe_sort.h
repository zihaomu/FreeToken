// r4d_moe_sort.h -- the routed slots' stable counting sort in one workgroup, shared by every
// kernel that ends in it: moe_scatter's small form and router_topk_scatter (r4d_moe.hip), and
// GLM's router_sigmoid_topk_scatter (r4d_glm.hip). One definition, so a fused selection and the
// standalone sort place every slot alike; tests/r4d_emu runs it from here.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>

#include "r4d_args.h"
#include "r4d_common.h"

// Internal linkage, as each unit's own definitions had: the same code generation in every kernel
// that takes the sort.
namespace r4d_moesort {
namespace {

constexpr uint32_t kSortWave = 32;
/* Expert-count ceiling, a hard limit of these kernels: the router keeps a lane's logits in
 * registers (MAXOWN) and both scatter forms keep n_expert + 1 counters in LDS. 512 covers
 * Qwen3.8-Flash-Next (512 experts of 640, top-10) for 4 KiB of scatter LDS and a second router
 * instantiation. */
constexpr uint32_t kMaxExpert = 512;

/* Protected experts of a slice, sorted last. A layer may keep a few experts (those whose output
 * dwarfs the rest) at higher precision, and one grouped GEMM takes one weight format, so the sort
 * places their runs after all others, in the listed order, and two GEMMs serve [0, n - p) and
 * [n - p, n) of the offsets. Local indices, strictly ascending; p <= kMaxProtect. With none, an
 * expert's place is itself. */
constexpr uint32_t kMaxProtect = 8;
struct Protect {
    uint32_t n;
    uint16_t e[kMaxProtect];
};

/* Local expert `e`'s place in the sort: the protected ones last, the rest closed up in order. */
__device__ __forceinline__ int32_t sort_place(int32_t e, uint32_t n_local, const Protect& p) {
    int32_t below = 0;
    for (uint32_t j = 0; j < p.n; ++j) {
        if ((int32_t)p.e[j] == e) return (int32_t)(n_local - p.n + j);
        below += (int32_t)p.e[j] < e;
    }
    return e - below;
}

/* `protect`, a comma list of local expert indices: strictly ascending, inside the slice, at most
 * kMaxProtect. Absent or empty is none. */
inline int protect_of(const RadArgs* a, long long n_local, Protect* p) {
    *p = Protect{};
    const char* s = r4d_gets(a, "protect");
    if (!s) return RAD_OK;
    long long prev = -1;
    while (*s) {
        char* end = nullptr;
        const long long v = std::strtoll(s, &end, 10);
        if (end == s || v <= prev || v >= n_local || p->n >= kMaxProtect)
            return R4D_NO(RAD_E_SHAPE);
        p->e[p->n++] = (uint16_t)v;
        prev = v;
        s = end;
        if (*s == ',') ++s;
        else if (*s) return R4D_NO(RAD_E_SHAPE);
    }
    return RAD_OK;
}

/* ================================================================== the small sort
 *
 * One workgroup does the whole sort, for T <= kScatterSmall, which covers every decode and verify
 * step (eight sequences at depth 3 and top_k 10 is T = 320), in one launch instead of three.
 * Histogram in LDS, offsets an exclusive prefix over n_expert <= kMaxExpert counts, and each slot
 * placed by its rank inside its expert's run.
 *
 * On one workgroup the critical path (barriers, serial LDS chains) is the cost, not the work:
 *
 *   - The prefix is a DPP scan inside each wave plus one exchange of wave totals through LDS: one
 *     barrier per tile of 32 counts per wave (at most two tiles at eight waves for 512 experts).
 *   - Placement groups first and orders after. The histogram atomic hands each slot a ticket in its
 *     expert's run (arrival order); slots are gathered into runs in ticket order, then each slot's
 *     place is the number of the run's slots with a lower index. That count is as long as the run
 *     (a row picks an expert at most once), far shorter than counting all earlier slots.
 *
 * Stable and byte-identical: a slot's place is its position among the same expert's slots in
 * ascending i, as in the reference, whatever the ticket order. Every barrier orders LDS only, so
 * none waits on or invalidates the vector memory path. */
constexpr uint32_t kScatterSmall = 1024;

/* The small sort's LDS (about 16 KiB), defined once for the standalone and fused kernels. NW is the
 * workgroup's wave count, which sizes the scan's exchange. */
template <uint32_t NW>
struct ScatterLds {
    static constexpr uint32_t kTiles = (kMaxExpert + NW * kSortWave - 1) / (NW * kSortWave);
    int32_t e[kScatterSmall];                    /* the slot list: expert in the slice, or -1 */
    int32_t tk[kScatterSmall];                   /* the slot's ticket inside its expert's run */
    int32_t run[kScatterSmall];                  /* slots grouped by expert, tickets' order */
    int32_t cnt[kMaxExpert + 1];
    int32_t off[kMaxExpert + 1];
    int32_t wt[kTiles][NW];                      /* each wave's total, a slot per tile */
};

/* Inclusive prefix sum over a wave32 with DPP row shifts: four steps within each 16-lane row, then
 * the upper row adds the lower row's total. Lanes shifted in from outside the row read zero
 * (bound_ctrl). */
__device__ __forceinline__ int32_t wave_incl_scan_i32(int32_t x, uint32_t lane) {
    x += __builtin_amdgcn_update_dpp(0, x, 0x111, 0xF, 0xF, true);    /* row_shr:1 */
    x += __builtin_amdgcn_update_dpp(0, x, 0x112, 0xF, 0xF, true);    /* row_shr:2 */
    x += __builtin_amdgcn_update_dpp(0, x, 0x114, 0xF, 0xF, true);    /* row_shr:4 */
    x += __builtin_amdgcn_update_dpp(0, x, 0x118, 0xF, 0xF, true);    /* row_shr:8 */
    const int32_t lo = __builtin_amdgcn_readlane(x, 15);
    return lane >= 16u ? x + lo : x;
}

/* The sort, from a filled slot list on: histogram, offsets and counts, placement. The caller has
 * written `sl.e[0, T)` and zeroed `sl.cnt[0, n_expert)`, followed by a barrier. Shared by the
 * standalone sort and the fused kernels. The workgroup is exactly NW waves.
 *
 * INV also writes the inverse: `inv[i]` is slot i's row in `sorted`, -1 for a dropped slot, the
 * table moe_gather would otherwise build from `sorted`. */
template <uint32_t NW, bool INV = false>
__device__ __forceinline__ void scatter_small_place(
    int32_t* __restrict__ sorted, int32_t* __restrict__ off, int32_t* __restrict__ cnt,
    uint32_t T, uint32_t n_expert, ScatterLds<NW>& sl, int32_t* __restrict__ inv = nullptr) {
    const uint32_t tid = threadIdx.x, lane = tid % kSortWave, wave = tid / kSortWave;

    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = sl.e[i];
        if (e >= 0) sl.tk[i] = atomicAdd(&sl.cnt[e], 1);
    }
    lds_barrier();

    /* Every thread sums all NW wave totals, so `carry` (finally the live slot count) is the same
     * in every thread. */
    int32_t carry = 0;
    for (uint32_t b = 0, t = 0; b < n_expert; b += NW * kSortWave, ++t) {
        const uint32_t e = b + tid;
        const int32_t  v = (e < n_expert) ? sl.cnt[e] : 0;
        const int32_t  x = wave_incl_scan_i32(v, lane);
        if (lane == kSortWave - 1) sl.wt[t][wave] = x;
        lds_barrier();
        int32_t below = 0, tot = 0;
#pragma unroll
        for (uint32_t k = 0; k < NW; ++k) {
            const int32_t c = sl.wt[t][k];
            below += (k < wave) ? c : 0;
            tot   += c;
        }
        if (e < n_expert) {
            const int32_t o = carry + below + x - v;      /* inclusive -> exclusive */
            sl.off[e] = o;
            off[e]    = o;
            cnt[e]    = v;
        }
        carry += tot;
    }
    if (tid == 0) off[n_expert] = carry;
    lds_barrier();

    for (uint32_t i = (uint32_t)carry + tid; i < T; i += blockDim.x) sorted[i] = -1;
    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = sl.e[i];
        if (e >= 0) sl.run[sl.off[e] + sl.tk[i]] = (int32_t)i;
    }
    lds_barrier();
    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = sl.e[i];
        if (e < 0) {
            if constexpr (INV) inv[i] = -1;
            continue;
        }
        const int32_t o = sl.off[e], n = sl.cnt[e];
        int32_t rank = 0;
        for (int32_t k = 0; k < n; ++k) rank += sl.run[o + k] < (int32_t)i;
        sorted[o + rank] = (int32_t)i;
        if constexpr (INV) inv[i] = o + rank;
    }
}

/* moe_scatter's small form: the slot list from `ids`, then the sort. A template only so a unit that
 * includes this header and does not launch it compiles no kernel. */
constexpr uint32_t kScatWaves = 8;          /* 256 threads */
template <bool INV>
__global__ __launch_bounds__(kScatWaves * kSortWave) void scatter_small_kernel(
    const int32_t* __restrict__ ids, int32_t* __restrict__ sorted, int32_t* __restrict__ off,
    int32_t* __restrict__ cnt, uint32_t T, uint32_t n_expert, int32_t base, Protect prot,
    int32_t* __restrict__ inv) {
    __shared__ ScatterLds<kScatWaves> sl;
    const uint32_t tid = threadIdx.x;
    for (uint32_t e = tid; e < n_expert; e += blockDim.x) sl.cnt[e] = 0;
    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = ids[i] - base;
        sl.e[i] = (e >= 0 && e < (int32_t)n_expert) ? sort_place(e, n_expert, prot) : -1;
    }
    lds_barrier();
    scatter_small_place<kScatWaves, INV>(sorted, off, cnt, T, n_expert, sl, inv);
}

}  // namespace
}  // namespace r4d_moesort
