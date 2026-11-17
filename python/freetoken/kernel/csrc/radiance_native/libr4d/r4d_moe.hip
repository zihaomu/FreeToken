// R4D mixture of experts: the router, the scatter/gather pair, and the grouped block-scaled FP8
// GEMM, in one translation unit because the first three exist to feed the GEMM.
//
// ---- GEMM shape ---------------------------------------------------------------
//
// Qwen3.6-35B-A3B: 256 experts, 8 per token, 512 intermediate columns each. A decode step at eight
// requests routes 64 (token, slot) rows across up to 64 experts and reads each one's gate_up
// [1024, 2048] and down [2048, 512] (3 MB an expert, 24 MB a layer). The grouped GEMM is
// bandwidth-bound, which shapes everything below:
//
//   * A block owns a contiguous range of sorted rows and loops over the experts in it; no padded
//     sort. A block straddling two experts stages two weight tiles, so weight traffic scales with
//     (block, expert) incidences, at most `n_expert + T/FM`.
//
//     At prefill a block map gives one run per block anyway (see the AL note beside
//     moe_blockmap_kernel): at a 2048 chunk the uniform grid has 414 incidences against 254, and a
//     duplicate staging of a pooled expert is another pass over PCIe. Worth 3.6% of prefill,
//     bit-identical. At decode the (tile, run) grid already gives one run per block.
//
//   * FM is banded: decode runs are a row or two and get 16-row tiles, one run per block; prefill
//     runs are tens of rows and get 32-row tiles, a wave skipping the 16-row M tiles its run does
//     not reach. See r4d_moe_decode_band and the kernel's prefill-tile note.
//
//   * Within a 16-row tile dead rows are masked, not skipped: free on a weight-bound kernel, and it
//     keeps the K loop and its barriers wave-uniform.
//
// ---- the weight is a pointer table --------------------------------------------------------
//
// `Wtab[e]`, not `W + e * stride`: a layer's experts are separate movement units (spec §4.2) that
// may sit in VRAM, behind the link or on disk, so no stride describes them (RAD_OPK_WTAB,
// abi/rad_runtime.h). The stacked rank-3 [n_expert, N, K] form is accepted too, for callers with
// no engine (rad-kbench fixtures); a zero leading stride distinguishes the table.
//
// ---- determinism ------------------------------------------------------------------------------
//
// The scatter is a stable counting sort with no atomic ordering: a slot's position is
// `expert_offset[e] + blocks_before[b][e] + rank_within_block`, all computed by reduction. An
// atomic cursor would make `sorted_tok`, and every row of the GEMM output, depend on scheduling,
// breaking the byte-level reproducibility of spec §17.
#include "r4d.h"
#include "r4d_common.h"
#include "r4d_chain.h"
#include "r4d_fp8_wmma.h"
#include "r4d_plugin.h"
#include "r4d_emit.h"
#include "r4d_gemm_args.h"
#include "r4d_moe_sort.h"
#include "r4d_moe_gather.h"

#include <hip/hip_bf16.h>

#include <cstdlib>

#include <algorithm>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>

namespace {

constexpr uint32_t kWave  = 32;
using r4d_moesort::kMaxExpert;
using r4d_moesort::kMaxProtect;
using r4d_moesort::Protect;
using r4d_moesort::sort_place;
using r4d_moesort::protect_of;
using r4d_moesort::kScatterSmall;
using r4d_moesort::ScatterLds;
using r4d_moesort::wave_incl_scan_i32;
using r4d_moesort::scatter_small_place;
using r4d_moesort::kScatWaves;
using r4d_moesort::scatter_small_kernel;
using r4d_moegather::kGatherSlots;
using r4d_moegather::kGatherSelfT;
using r4d_moegather::gather_kernel;
using r4d_moegather::r4dchain_moe_gather;
constexpr uint32_t kBlock = 128;          // the fp8 scale block, along K and along N
/* Run-map entry that owns no expert: rows above `off[n_expert]` are dropped slots whose output must
 * be zeroed. Declared here because the GEMM reads it; the map is built further down. */
constexpr uint32_t kMapTail = 0xffffffffu;
constexpr uint32_t kFK    = 128;          // K bytes staged per pass; one scale block exactly

using r4d_fp8::i2v;
using r4d_fp8::f8v;
using r4d_fp8::mma;
using r4d_fp8::acc_zero;
using r4d_fp8::acc_val;
using r4d_fp8::mma8;
using r4d_fp8::acc_row;
using r4d_fp8::frag_c;
using r4d_fp8::frag;
using r4d_fp8::bscale;
using r4d_fp8::Stage8;
using r4d_fp8::stage8_load;
using r4d_fp8::stage8_store;
using r4d_fp8::lds_pitch;

#define R4D_MOE_CHK() do { hipError_t e_ = hipGetLastError();                       \
                           if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

/* An entry of a weight-table operand: `tab` is the pointer array when `stride0` is 0, else the base
 * of a stacked plane. Resolved on the device: the table is in device memory, the mover may rewrite
 * it, and a host read would need a sync. */
__device__ __forceinline__ const uint8_t* wtab_at(const void* const* tab, size_t stride0,
                                                  uint32_t e) {
    if (stride0 == 0) return (const uint8_t*)tab[e];
    return (const uint8_t*)tab + (size_t)e * stride0;
}

/* One sortable u32 for (value, expert id), which removes the branches from the selection; possible
 * because the logits are bf16.
 *
 * The selection wants the largest value, ties to the lower id. As a two-register comparison gfx12
 * compiles it into exec-mask branches, and the kernel runs on one workgroup with nothing to hide
 * them. Instead the order-preserving map of the bf16 (flip the sign bit if positive, invert all
 * bits if negative; monotone including -inf) goes in the top half and `~id` in the low half, so a
 * lower id wins ties. Selection is then an unsigned max, the wave reduction carries one register,
 * and an entry is removed by zeroing its key.
 *
 * Same answer as the comparison form, with two special cases handled:
 *   - -0.0 compares equal to +0.0 but has a different pattern, so it is folded to +0.0.
 *   - NaN never wins a float comparison, but a positive NaN's pattern sorts above +inf, so NaN gets
 *     key 0 (never selected; an all-NaN row takes the `nothing left` exit).
 *
 * Inverting the map returns the original bf16, so the softmax input is unchanged.
 *
 * Note: rad-kbench's 6.354e-05 on the M=512 router case is a one-ulp difference in the weight, not
 * the selection (expert_ids are exact); every case from M=1 to M=256 is exact. */
__device__ __forceinline__ uint32_t topk_key(uint16_t raw, uint32_t e) {
    const bool nan = ((raw & 0x7F80u) == 0x7F80u) && (raw & 0x7Fu);
    uint16_t   u   = (raw == 0x8000u) ? (uint16_t)0u : raw;          /* -0.0 folds to +0.0 */
    const uint16_t k = (u & 0x8000u) ? (uint16_t)~u : (uint16_t)(u | 0x8000u);
    return nan ? 0u : (((uint32_t)k << 16) | (~e & 0xFFFFu));
}
__device__ __forceinline__ uint16_t topk_key_raw(uint32_t key) {
    const uint16_t k = (uint16_t)(key >> 16);
    return (k & 0x8000u) ? (uint16_t)(k & 0x7FFFu) : (uint16_t)~k;
}

/* The selection's wave minimum, on r4d_xor_lane's DPP/permlane exchange rather than `__shfl_xor`
 * (an LDS round trip per stage on gfx1201, run top_k times per row with nothing to hide it). An
 * unsigned minimum is order-independent. */
__device__ __forceinline__ uint32_t wave_min_u32(uint32_t b) {
    uint32_t o;
    o = r4d_xor_lane_u32<16>(b); b = b < o ? b : o;
    o = r4d_xor_lane_u32<8>(b);  b = b < o ? b : o;
    o = r4d_xor_lane_u32<4>(b);  b = b < o ? b : o;
    o = r4d_xor_lane_u32<2>(b);  b = b < o ? b : o;
    o = r4d_xor_lane_u32<1>(b);  b = b < o ? b : o;
    return b;
}

/* ================================================================== router_topk
 *
 * Softmax over all n_expert logits, then the top_k of them by value, then -- if `norm` -- the kept
 * weights renormalised to sum to one.
 *
 * One wave per row; the selection is k passes of a wave reduction, not a sort. That matches the
 * reference exactly (largest remaining value, ties to the lower id); a sort with another tie rule
 * would be a different router on tied rows, which are not rare. Selection runs on the logits:
 * softmax is monotone, so the order is the same and the comparison exact (as in libref). */
/* MAXOWN is how many of a row's logits one lane holds (n_expert <= MAXOWN * kWave). It is a
 * template parameter because they are registers: 16 per lane at 512 experts, 8 at 256. */
/* One row per wave, with nothing outside the wave: reductions are butterflies over the row's 32
 * lanes and stores are addressed by `m`, so the grid shape cannot change the answer. That lets
 * `router_topk_scatter_kernel` run this same code from a different grid, byte-identically. */
/* In the fused kernel a row also hands its slots to the sort through `e`: each slot's expert
 * relative to the slice, or -1 for another rank's. The standalone selection passes a null `e`. */
struct SlotSink {
    int32_t* e;
    int32_t  base;
    uint32_t n_local;
    Protect  prot;
};

template <uint32_t MAXOWN>
__device__ __forceinline__ void router_topk_row(
    const uint16_t* __restrict__ lg, int32_t* __restrict__ ids, uint16_t* __restrict__ w,
    uint32_t m, uint32_t n_expert, uint32_t top_k, uint32_t lg_ld, int norm, uint32_t lane,
    SlotSink snk) {

    const uint16_t* row = lg + (size_t)m * lg_ld;
    /* Up to MAXOWN experts per lane; n_expert <= MAXOWN * kWave is checked on the host, which also
     * picks the instantiation.
     *
     * The trip count is MAXOWN, not a lane-derived `n_own`: a lane-derived count makes every later
     * `v[i]` a divergent register index, which gfx12 implements as a v_movrels waterfall loop,
     * costing tens of microseconds to rank 512 numbers. Unrolled to the constant bound, the array
     * stays in registers. Slots past `n_expert` hold -INFINITY with key 0, which keeps the result
     * bit-identical: fmaxf ignores -inf, __expf(-inf - mx) is exactly zero, and the selection
     * never picks key 0 while a real entry remains. */
    /* All MAXOWN loads issue before any is waited on. A guarded `if (ok) raw = row[e]` would put
     * each load in its own exec-masked branch and serialise the round trips, so a slot past
     * `n_expert` reads the row's first logit (always in bounds) and the select discards it. */
    uint16_t raw[MAXOWN];
#pragma unroll
    for (uint32_t i = 0; i < MAXOWN; ++i) {
        const uint32_t e = lane + i * kWave;
        raw[i] = row[e < n_expert ? e : 0u];
    }
    float    v[MAXOWN];
    uint32_t key[MAXOWN];    /* (order-preserving value, ~id); 0 = absent or already chosen */
#pragma unroll
    for (uint32_t i = 0; i < MAXOWN; ++i) {
        const uint32_t e  = lane + i * kWave;
        const bool     ok = e < n_expert;
        v[i]   = ok ? bf16_to_f32(raw[i]) : -INFINITY;
        key[i] = topk_key(raw[i], e) & (0u - (uint32_t)ok);
    }

    float mx = -INFINITY;
#pragma unroll
    for (uint32_t i = 0; i < MAXOWN; ++i) mx = fmaxf(mx, v[i]);
    mx = fmaxf(mx, r4d_xor_lane<16>(mx));
    mx = fmaxf(mx, r4d_xor_lane<8>(mx));
    mx = fmaxf(mx, r4d_xor_lane<4>(mx));
    mx = fmaxf(mx, r4d_xor_lane<2>(mx));
    mx = fmaxf(mx, r4d_xor_lane<1>(mx));

    float sum = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < MAXOWN; ++i) sum += __expf(v[i] - mx);
    sum = r4d_wave_sum(sum);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    /* ---- the selection stays in registers.
     *
     * The reference stores each selected probability as bf16 and, with `norm`, rescales the stored
     * value. Doing that literally would chain top_k stores and dependent loads on one lane. Instead
     * round r parks its winner on lane r (top_k <= kWave; both launchers refuse more).
     *
     * Round r takes the largest key below round r-1's, which equals zeroing the last winner (keys
     * are unique) but costs one subtraction: with P the last winner minus one, P - key is at most P
     * for keys not above P and wraps past P for taken keys, so the smallest P - key names the
     * largest untaken key. Key 0 (absent or NaN) yields P itself and is chosen only when nothing
     * real is left, the `nothing left` exit.
     *
     * Probabilities are computed after the loop on each winner's lane with the same expression, so
     * the same bits. `wsum` adds them in round order, as the reference does, and the
     * renormalisation reads the produced bf16, not the f32 behind it. */
    uint32_t P     = 0xFFFFFFFFu;     /* round 0 may take any key */
    uint32_t mykey = 0u;
    uint32_t nsel  = 0;
    for (uint32_t r = 0; r < top_k; ++r) {
        uint32_t d = 0xFFFFFFFFu;
#pragma unroll
        for (uint32_t i = 0; i < MAXOWN; ++i) {
            const uint32_t t = P - key[i];
            d = d < t ? d : t;
        }
        const uint32_t best = P - wave_min_u32(d);
        if ((best & 0xFFFFu) == 0u) break;    /* fewer live entries than top_k: nothing left */
        if (lane == r) mykey = best;
        P = best - 1u;
        ++nsel;
    }
    const float    p  = __expf(bf16_to_f32(topk_key_raw(mykey)) - mx) * inv;
    const uint16_t pb = f32_to_bf16(p);
    float wsum = 0.0f;
    for (uint32_t r = 0; r < nsel; ++r)
        wsum += __builtin_bit_cast(float,
                    __builtin_amdgcn_readlane(__builtin_bit_cast(int, p), (int)r));
    if (lane < nsel) {
        ids[(size_t)m * top_k + lane] = (int32_t)(~mykey & 0xFFFFu);
        w[(size_t)m * top_k + lane]   = (norm && wsum > 0.0f)
                                      ? f32_to_bf16(bf16_to_f32(pb) * (1.0f / wsum)) : pb;
    }
    if (snk.e && lane < top_k) {
        const int32_t e  = (int32_t)(~mykey & 0xFFFFu) - snk.base;
        const bool    ok = lane < nsel && e >= 0 && e < (int32_t)snk.n_local;
        snk.e[m * top_k + lane] = ok ? sort_place(e, snk.n_local, snk.prot) : -1;
    }
}

template <uint32_t NW, uint32_t MAXOWN>
__global__ __launch_bounds__(NW * kWave) void router_topk_kernel(
    const uint16_t* __restrict__ lg, int32_t* __restrict__ ids, uint16_t* __restrict__ w,
    uint32_t M, uint32_t n_expert, uint32_t top_k, uint32_t lg_ld, int norm) {
    const uint32_t wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
    const uint32_t m = blockIdx.x * NW + wave;
    if (m >= M) return;
    router_topk_row<MAXOWN>(lg, ids, w, m, n_expert, top_k, lg_ld, norm, lane,
                            SlotSink{nullptr, 0, 0, Protect{}});
}

/* ================================================================== moe_scatter
 *
 * The small form is one workgroup and the sort of r4d_moe_sort.h; the general form below takes
 * three passes. */
/* ================================================================== router_topk + moe_scatter
 *
 * One launch for the pair: at decode each half is a single 256-thread workgroup moving about a
 * kilobyte, so each launch is mostly dispatch floor and barriers, once per MoE layer per step.
 *
 * Slots reach the sort through LDS, not `expert_ids`: each row's lanes write their slot's expert
 * (relative to the slice, or -1) into the sort's slot list, so the one barrier between the halves
 * orders LDS only. There is no barrier before the rows (each thread zeroes the histogram on its
 * way in), since a top barrier would wait for the workgroup's last wave to launch.
 *
 * Byte-identical to the unfused pair: the selection and placement are the same device functions,
 * wave-local and stable. The one difference is a row with fewer live experts than top_k (NaN
 * logits): the standalone sort reads its unselected slots from whatever `expert_ids` held, this
 * one records them as dropped.
 *
 * The rows are a loop, not an early return, because every thread must reach the barrier. T =
 * M * top_k must fit kScatterSmall, checked by the shim and by the architecture before declaring
 * the op.
 */
/* Two different expert counts: the selection runs over the whole (replicated) router, `n_expert`,
 * while the sort runs over this rank's slice, `n_local`, dropping the rest (512 against 254 at two
 * ranks here). Passing one count to both would let the sort index its histogram out of range. */
template <uint32_t MAXOWN, uint32_t NW>
__global__ __launch_bounds__(NW * kWave) void router_topk_scatter_kernel(
    const uint16_t* __restrict__ lg, int32_t* __restrict__ ids, uint16_t* __restrict__ w,
    int32_t* __restrict__ sorted, int32_t* __restrict__ off, int32_t* __restrict__ cnt,
    uint32_t M, uint32_t n_expert, uint32_t n_local, uint32_t top_k, uint32_t lg_ld, int norm,
    int32_t base, Protect prot) {
    __shared__ ScatterLds<NW> sl;
    const uint32_t wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
    for (uint32_t e = threadIdx.x; e < n_local; e += blockDim.x) sl.cnt[e] = 0;
    const SlotSink snk{sl.e, base, n_local, prot};
    for (uint32_t m = wave; m < M; m += NW)
        router_topk_row<MAXOWN>(lg, ids, w, m, n_expert, top_k, lg_ld, norm, lane, snk);
    lds_barrier();
    scatter_small_place(sorted, off, cnt, M * top_k, n_local, sl);
}

/* The general form, in three passes so it stays stable without an atomic deciding order. `bcnt` is
 * [n_tile, n_expert] in scratch. */
constexpr uint32_t kScatterTile = 256;

__global__ __launch_bounds__(kScatterTile) void scatter_hist_kernel(
    const int32_t* __restrict__ ids, int32_t* __restrict__ bcnt, uint32_t T, uint32_t n_expert,
    int32_t base, Protect prot) {
    extern __shared__ int32_t s_h[];
    const uint32_t tid = threadIdx.x, b = blockIdx.x;
    for (uint32_t e = tid; e < n_expert; e += blockDim.x) s_h[e] = 0;
    __syncthreads();
    const uint32_t i = b * kScatterTile + tid;
    if (i < T) {
        const int32_t e = ids[i] - base;
        if (e >= 0 && e < (int32_t)n_expert) atomicAdd(&s_h[sort_place(e, n_expert, prot)], 1);
    }
    __syncthreads();
    for (uint32_t e = tid; e < n_expert; e += blockDim.x)
        bcnt[(size_t)b * n_expert + e] = s_h[e];
}

/* One workgroup, one thread per expert: the exclusive scan across tiles within each expert, in
 * place (`bcnt[b][e]` becomes the count of expert e in tiles before b), each expert's total, and
 * the exclusive scan of the totals across experts.
 *
 * A thread's column of `bcnt` is loaded kScanBatch at a time before use; one at a time it would be
 * 80 dependent round trips at a 2048-token chunk and top 10. The prefix across experts is the small
 * sort's (DPP scan per wave plus one exchange). Integer sums are order-independent. */
constexpr uint32_t kScanBatch = 16;

__global__ __launch_bounds__(kMaxExpert) void scatter_scan_kernel(
    int32_t* __restrict__ bcnt, int32_t* __restrict__ off, int32_t* __restrict__ cnt,
    uint32_t n_tile, uint32_t n_expert) {
    constexpr uint32_t NW = kMaxExpert / kWave;
    __shared__ int32_t s_wt[NW];
    const uint32_t e = threadIdx.x, lane = e % kWave, wave = e / kWave;
    int32_t run = 0;
    if (e < n_expert) {
        for (uint32_t b0 = 0; b0 < n_tile; b0 += kScanBatch) {
            /* Every load in bounds (past the end re-reads the last row, discarded), so no load sits
             * in a branch. */
            int32_t v[kScanBatch];
#pragma unroll
            for (uint32_t k = 0; k < kScanBatch; ++k) {
                const uint32_t b = b0 + k < n_tile ? b0 + k : n_tile - 1u;
                v[k] = bcnt[(size_t)b * n_expert + e];
            }
#pragma unroll
            for (uint32_t k = 0; k < kScanBatch; ++k) {
                if (b0 + k >= n_tile) break;
                bcnt[(size_t)(b0 + k) * n_expert + e] = run;
                run += v[k];
            }
        }
        cnt[e] = run;
    }
    const int32_t x = wave_incl_scan_i32(run, lane);
    if (lane == kWave - 1) s_wt[wave] = x;
    lds_barrier();
    int32_t below = 0, tot = 0;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) {
        const int32_t c = s_wt[w];
        below += (w < wave) ? c : 0;
        tot   += c;
    }
    if (e < n_expert) off[e] = below + x - run;     /* inclusive -> exclusive */
    if (e == 0) off[n_expert] = tot;
}

/* Placement and tail. Each routed slot goes to its expert's offset plus the expert's count in
 * earlier tiles plus its rank in this tile. Sorted rows past the live count are -1; no placement
 * lands there (all placed slots are below `off[n_expert]`), so each thread writes its own index
 * without races. INV also writes each slot's row, -1 when dropped (scatter_small_place). */
template <bool INV>
__global__ __launch_bounds__(kScatterTile) void scatter_place_kernel(
    const int32_t* __restrict__ ids, const int32_t* __restrict__ bcnt,
    const int32_t* __restrict__ off, int32_t* __restrict__ sorted,
    uint32_t T, uint32_t n_expert, int32_t base, Protect prot, int32_t* __restrict__ inv) {
    __shared__ int32_t s_id[kScatterTile];
    const uint32_t tid = threadIdx.x, b = blockIdx.x;
    const uint32_t i = b * kScatterTile + tid;
    /* Everything works in sort places (local index, protected experts last), the indices the
     * offsets and weight tables use. */
    const int32_t  l = (i < T) ? ids[i] - base : -1;
    const int32_t  e = (l >= 0 && l < (int32_t)n_expert) ? sort_place(l, n_expert, prot) : -1;
    s_id[tid] = e;
    const uint32_t live = (uint32_t)off[n_expert];
    if (i >= live && i < T) sorted[i] = -1;
    __syncthreads();
    if constexpr (INV) if (i < T && e < 0) inv[i] = -1;
    if (i >= T || e < 0) return;
    /* Rank within the tile: earlier slots of this tile with the same expert. A serial scan of at
     * most 256 LDS words per thread, the price of stability. */
    int32_t rank = 0;
    for (uint32_t j = 0; j < tid; ++j) rank += (s_id[j] == e);
    const int32_t at = off[e] + bcnt[(size_t)b * n_expert + e] + rank;
    sorted[at] = (int32_t)i;
    if constexpr (INV) inv[i] = at;
}

/* ================================================================== the 4-bit expert weight
 *
 * Sixteen packed bytes expand to the thirty-two E4M3 bytes of a Stage8; everything downstream (LDS
 * pitch, frag(), WMMA, store) is the fp8 kernel's. The stored plane is described in
 * r4d_layout.cpp's moe w4 section.
 *
 * Byte b of a packed dword holds element b in its low nibble and b+4 in its high one, so the two
 * masks yield elements 0..3 and 4..7 in order with no interleave (as in the dense w4a8 kernel).
 *
 * Fifteen VALU per eight weights. The kernel is DRAM-bound by over an order of magnitude (a
 * prefill layer: ~3 GiB of expert weight for 134 GFLOP), so the dequant is free and halving the
 * weight stream is the win.
 */
__device__ __forceinline__ uint32_t w4_e4m3(uint32_t nib, const uint32_t (&tab)[4]) {
    /* Sixteen levels from eight-byte permutes. V_PERM_B32 selects from a 64-bit source, so one
     * table holds eight entries; two tables and a selector between them cover a nibble:
     *
     *   m   = nib & 7                      the index within a half, the same for both
     *   lo  = perm(tab[1], tab[0], m)      E4M3 of entries 0..7   (codes 0..7)
     *   hi  = perm(tab[3], tab[2], m)      E4M3 of entries 8..15  (codes 8..15)
     *   sel = 0x03020100 + ((nib & 8) >> 1)   byte i picks lo[i], or hi[i] four bytes further on
     *   out = perm(hi, lo, sel)
     *
     * The add cannot carry: each selector byte is 0..3 before it and 4..7 after.
     *
     * The table is an argument (r4d_args.h): the integers -8..7 of a two's-complement nibble (w4a8,
     * w4a8h) or the w4nl levels (w4nla8h). Every entry is exact in E4M3, so codes expand losslessly
     * and the f32_16x16x16_fp8_fp8 inner loop is reused unchanged; only staging differs. */
    const uint32_t m   = nib & 0x07070707u;
    const uint32_t lo  = __builtin_amdgcn_perm(tab[1], tab[0], m);
    const uint32_t hi  = __builtin_amdgcn_perm(tab[3], tab[2], m);
    const uint32_t sel = 0x03020100u + ((nib & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(hi, lo, sel);
}

__device__ __forceinline__ Stage8 w4_expand(uint4 p, const uint32_t (&tab)[4]) {
    const uint32_t d[4] = { p.x, p.y, p.z, p.w };
    uint32_t o[8];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        o[2 * i]     = w4_e4m3(d[i] & 0x0F0F0F0Fu, tab);
        o[2 * i + 1] = w4_e4m3((d[i] >> 4) & 0x0F0F0F0Fu, tab);
    }
    Stage8 r;
    r.v = make_uint4(o[0], o[1], o[2], o[3]);
    r.w = make_uint4(o[4], o[5], o[6], o[7]);
    return r;
}

/* w5nl64a8h's 32 codes: magnitudes via w4_expand (`tab` holds the sixteen magnitudes), then each
 * byte's sign from `sg`. Output dword m holds elements 4m..4m+3, and the relayout stores element
 * 4m+b's sign at bit 7-m of byte b of `sg` (r4d_layout.cpp), so shifting by m puts dword m's signs
 * at bit 7 of their bytes (E4M3's sign bit) and the mask keeps them. */
__device__ __forceinline__ Stage8 w5_expand(uint4 p, uint32_t sg, const uint32_t (&tab)[4]) {
    Stage8 r = w4_expand(p, tab);
    uint32_t o[8] = { r.v.x, r.v.y, r.v.z, r.v.w, r.w.x, r.w.y, r.w.z, r.w.w };
#pragma unroll
    for (uint32_t m = 0; m < 8; ++m)
        o[m] |= (sg << m) & 0x80808080u;
    r.v = make_uint4(o[0], o[1], o[2], o[3]);
    r.w = make_uint4(o[4], o[5], o[6], o[7]);
    return r;
}

/* ---- the codebook levels against an int8 activation ---------------------------------------------
 *
 * The iu8 WMMA wants the levels as int8. The two's-complement nibble is its own value
 * (kW4IntPermI8). w4nl's magnitudes are 1.125, 3.25, 5.5, 8, 11, 14, 18 and 22: the unit that makes
 * all of them integers is 1/8, at which 22 is 176 and past int8, and a second WMMA over the other
 * half of each level costs a third of the kernel's time. So the levels are read as the int8 set
 * nearest them at one unit, r4d_args.h's kW4nlI8 times kW4nlI8Unit (folded into the scale): each
 * within 1.2% of its level. */
static constexpr uint32_t kW4IntPermI8[4] = { 0x03020100u, 0x07060504u, 0xFBFAF9F8u, 0xFFFEFDFCu };
static constexpr uint32_t kW4nlPermI8[4] = {
    w4nl_i8_dword(0), w4nl_i8_dword(1), w4nl_i8_dword(2), w4nl_i8_dword(3) };

/* The staged bytes of a four-bit row segment: E4M3 from `tab`, or at I8 the int8 levels, `tab`'s
 * for the integer grid and w4nl's at the subgroup-scaled forms (W4 2 and 3). */
template <int W4, bool I8>
__device__ __forceinline__ Stage8 moe_w4_bytes(uint4 p, const uint32_t (&tab)[4]) {
    if constexpr (I8 && (W4 == 2 || W4 == 3)) return w4_expand(p, kW4nlPermI8);
    else return w4_expand(p, tab);
}

/* The packed half of a Stage8's worth of K, or zeros for a row this block does not own. Zeros
 * expand to entry 0 (E4M3 zero under -8..7, 1.125 under w4nl); either way the column contributes
 * nothing, since staging gives an unowned column a weight scale of zero. */
__device__ __forceinline__ uint4 w4_load(const uint8_t* __restrict__ src, bool live) {
    return live ? *(const uint4*)src : make_uint4(0, 0, 0, 0);
}

/* ---- UW: the weight stream from a uniform global address ----------------------------------------
 *
 * The weight base pointer is loaded through the table with an index from LDS, so the compiler
 * treats it as a divergent generic pointer and every weight load becomes a flat_load. On gfx12 a
 * flat load may reach LDS, so the waitcnt pass waits for a load count of zero before every weight
 * load, serialising a pass's activation and weight loads into three round trips.
 *
 * Under UW the table entries are made uniform with readfirstlane (`e` is block-uniform) and loaded
 * through address_space(1), as in the dense w4 kernels: SGPR base, global_load, no premature wait.
 * Bit-identical. A pooled expert's host-memory entry is still a global address. */
typedef unsigned r4d_moe_u4v __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) const r4d_moe_u4v* r4d_moe_gu4;
typedef __attribute__((address_space(1))) const uint32_t*    r4d_moe_gu32;
typedef __attribute__((address_space(1))) const uint16_t*    r4d_moe_gu16;

template <bool UW, typename P>
__device__ __forceinline__ const P* uw_ptr(const P* p) {
    if constexpr (!UW) return p;
    const uint64_t v  = (uint64_t)(uintptr_t)p;
    const uint32_t lo = (uint32_t)__builtin_amdgcn_readfirstlane((int)(uint32_t)v);
    const uint32_t hi = (uint32_t)__builtin_amdgcn_readfirstlane((int)(uint32_t)(v >> 32));
    return (const P*)(uintptr_t)(((uint64_t)hi << 32) | lo);
}
template <bool UW>
__device__ __forceinline__ uint4 w4_load_w(const uint8_t* __restrict__ src, bool live) {
    if constexpr (!UW) return w4_load(src, live);
    if (!live) return make_uint4(0, 0, 0, 0);
    const r4d_moe_u4v v = *(r4d_moe_gu4)src;
    return make_uint4(v.x, v.y, v.z, v.w);
}
template <bool UW>
__device__ __forceinline__ Stage8 stage8_load_w(const uint8_t* __restrict__ src, bool live) {
    if constexpr (!UW) return stage8_load(src, live);
    Stage8 r;
    r.v = make_uint4(0, 0, 0, 0);
    r.w = make_uint4(0, 0, 0, 0);
    if (live) {
        const r4d_moe_u4v a = *(r4d_moe_gu4)src, b = *(r4d_moe_gu4)(src + 16);
        r.v = make_uint4(a.x, a.y, a.z, a.w);
        r.w = make_uint4(b.x, b.y, b.z, b.w);
    }
    return r;
}
/* A 32-K row segment from the per-128-scale I4 fragment tiles, for the LDS-staged kernels.
 * Two 64-bit loads gather its four dwords; subgroup-scaled codebooks retain their row loads. */
template <int W4, bool UW>
__device__ __forceinline__ uint4 moe_w4_load(const uint8_t* base, uint32_t row, uint32_t ld,
                                           uint32_t k, bool live) {
    if constexpr (W4 != 1) return w4_load_w<UW>(base + (size_t)row * ld + k / 2, live);
    if (!live) return make_uint4(0, 0, 0, 0);
    const size_t word = (((size_t)(row / 16) * (ld / 32) + k / 64) * 32 + row % 16) * 4 +
                        (k % 64) / 16;
    typedef __attribute__((address_space(1))) const uint64_t* gu64;
    const auto* p = (const uint64_t*)(base + word * 4);
    const uint64_t lo = UW ? *(gu64)p : *p;
    const uint64_t hi = UW ? *(gu64)(p + 32) : p[32];
    return make_uint4((uint32_t)lo, (uint32_t)hi, (uint32_t)(lo >> 32), (uint32_t)(hi >> 32));
}

/* A staged tile row as loaded, or the zero a dead row stages. */
__device__ __forceinline__ Stage8 stage8_keep(const Stage8& s, bool live) {
    Stage8 r;
    r.v = live ? s.v : make_uint4(0, 0, 0, 0);
    r.w = live ? s.w : make_uint4(0, 0, 0, 0);
    return r;
}
template <bool UW>
__device__ __forceinline__ uint32_t u32_load_w(const uint32_t* p) {
    if constexpr (!UW) return *p;
    return *(r4d_moe_gu32)p;
}
/* A 16-bit scale loaded zero-extended into a whole register. As a 16-bit value LLVM 24 loads it into
 * a register half (global_load_d16_b16), which depends on the register's last write and forces a
 * full wait at the surrounding branch joins (25-30% on the down projection's four-pass blocks).
 * Widened, it is a global_load_u16. */
template <bool UW>
__device__ __forceinline__ uint32_t u16_load_w(const uint16_t* p) {
    if constexpr (!UW) return *p;
    return *(r4d_moe_gu16)p;
}

/* ================================================================== moe_gemm_q
 *
 * y[i, :] = a[row(i), :] @ w[e]^T, in SORTED order, both operands E4M3 with a scale per 128 along
 * K (and, for the weight, per 128 along N).
 *
 * `a_sorted` picks row(i). gate_up reads the block input, one row per token, so row(i) is
 * `sorted[i] / top_k`. down reads gate_up's output, already one row per (token, slot) in sorted
 * order, so row(i) is i; gathering again would read another slot's activation. Only `s_tok`'s
 * resolution differs.
 *
 * MT   M tiles a block, 16 rows each; the banded knob.
 * NTL  N tiles a block, 16 columns each.
 * WGM  waves along M; WGN waves along N. WGM * WGN waves a block, a wave owning MT/WGM M tiles and
 *      NTL/WGN N tiles.
 *
 * SK   skip, per wave, the M tiles that start at or past the end of the block's run.
 *
 * ---- prefill tile: 32 rows x 64 columns, waves skipping rows their run does not reach ----
 *
 * (128 columns over eight waves where the weight's parts allow: see kPrefillCols.)
 *
 * At prefill a block serves one run (AL), averaging T / n_expert rows (forty at a 2048 chunk on
 * Qwen3.8-Flash-Next). A 128-row tile is then mostly dead rows, masked at load and store but still
 * computed; reshaping it only moves the staging depth. Two changes remove the dead work, both
 * bit-identical:
 *
 *   SK     a wave skips fragment reads, WMMA and scale fold for M tiles starting at or past its
 *          run's end (wave-uniform; it still stages and meets every barrier).
 *   FM 32  the tile matches a typical run; a longer run spans blocks and re-stages from L2.
 *
 * Measured (us per call, gate_up / down, 40 rows per run): 128 x 64 with 8 waves 3346 / 2081;
 * 32 x 64 with 4 waves and SK 1552 / 987. 16- and 64-row tiles at 128 columns, and 32 x 128 at
 * small runs, are slower, so the tile keeps 64 columns (and the op's "N a multiple of 64").
 *
 * At prefill the link is not the bottleneck: per-layer GEMM time does not correlate with how many
 * experts are pooled. The deployment uses a 2048 chunk; larger chunks do not help (the checkpoint
 * interval clamps them, and a larger activation arena costs expert residency).
 *
 * ES   one run per block instead of all of a tile's runs (below).
 *
 * ---- ES: spreading a decode step's runs over blocks -----------------------------------
 *
 * At decode T is about eleven rows, one 16-row tile, so a row-range grid collapses to (1, N/FN)
 * (20 workgroups for a 1280-column gate_up) and walks the experts serially (~89 GB/s). Runs are
 * independent (run j reads expert e(j) and writes rows [rs, re), partitioning the tile), so
 * `blockIdx.x` carries (tile, run) and each block does one run. Weight traffic and accumulation
 * order are unchanged, so output is bit-identical. A tile has at most FM runs (each owns a row), so
 * FM blocks per tile suffice; extra blocks exit at the loop head. The split is unconditional inside
 * the decode band (r4d_moe_decode_band).
 *
 * At decode the link binds, not this kernel: a gate/up dispatch reads ~19.7 MB; the fastest run at
 * the part's 576 GB/s cold-stream peak, and the mean (270 GB/s) gap is entirely experts not
 * resident in the slab, streamed over the link at ~26 GB/s (~63 us per 1.64 MB expert). The lever
 * is the miss rate, not this kernel's arithmetic.
 */
/* One class of experts: even- or odd-numbered, with separate weight tables because every entry of
 * one table has one geometry. Expert e is entry e / 2 of class e % 2. The classes differ when a rank
 * holds an uneven slice of each expert's width (three 128-column blocks of one parity, two of the
 * other). `moe_gemm_q_mod4` has four classes, by expert number mod 4 (entry e / 4 of class e % 4),
 * so four ranks take a five-block expert's extra block in turn.
 *
 * `n` is the weight's rows and `k` its K; `w_ld` the packed byte pitch of a weight row and `s_ld`
 * the scale plane's row pitch in elements. */
struct MoeCls {
    const void* const* w;
    const void* const* s;
    uint32_t w_s0, s_s0;   /* bytes an entry of a stacked plane; 0 marks a pointer table */
    uint32_t n, k, w_ld, s_ld;
};
/* W4: the code table (r4d_args.h's kW4IntPerm, kW4nlPerm or kW5nlMagPerm), shared by all classes. */
struct MoeTab { uint32_t v[4]; };

/* W4 is the weight's form: 0 block E4M3; 1 four-bit with one bf16 scale per (row, 128 of K); 2 the
 * same codes with one E4M3 scale per (row, 64 of K) times the fixed kW4G64Scale (w4nl64a8h); 3 the
 * same per (row, 32 of K) (w4nl32a8h); 4 five-bit codes, form 2's magnitudes and scales plus a sign
 * bit per code from the plane after a row's nibbles (w5nl64a8h). At 2-4 a staged 128-wide tile holds
 * two or four scale groups, each in its own accumulator; a tile's E4M3 scales are one 16- or 32-bit
 * load.
 *
 * FK is the K staged per pass: kFK (one 128-wide scale block) or, at W4 2-4 only, two blocks, which
 * doubles the bytes in flight per barrier pair and halves the passes. Each scale group keeps its
 * own accumulator and the fold runs the two blocks in K order, so FK 256 is bit-identical to 128.
 *
 * UW: uniform global weight loads (see the UW note above). Used by the decode form and the
 * run-aligned prefill form (whose first-pass prefetch at the map read needs it); the uniform
 * prefill grid keeps plain loads, as it was never measured with UW.
 *
 * RS finds an ES block's run in one round trip (see the RS note at the run search), for up to
 * kMaxExpert experts; wider counts use the chunked search, which numbers runs the same way.
 *
 * Waves per SIMD (R4D_WAVES), as tuned: ES prefill 9-11 (5 at FK 256, register-bound), aligned
 * 9-12, uniform grid 8-9. Unconstrained, LLVM 24 spends ~30 more VGPRs on the aligned split-K form
 * for one to three fewer waves (16-52% slower prefill tiles). */
constexpr uint32_t moe_waves(uint32_t ntl, int w4, bool es, bool al, uint32_t fk, bool rs) {
    return es ? (fk == 256u ? 5u : w4 == 0 ? (rs ? 10u : 11u) : w4 == 1 ? 11u : w4 == 3 ? 9u : 10u)
         : al ? (ntl == 16u ? (w4 == 1 ? 12u : 9u) : ntl == 8u ? (w4 <= 1 ? 10u : 9u) : 9u)
              : (w4 <= 1 ? 9u : 8u);
}

#define R4D_MOEGQ_PARAMS                                                                        \
    const uint8_t* __restrict__ Aq, const float* __restrict__ As, const MoeCls c_even,          \
    const MoeCls c_odd, const MoeCls c_2, const MoeCls c_3,                                     \
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,                        \
    uint16_t* __restrict__ Y, const uint32_t* __restrict__ map, uint32_t T, uint32_t N,         \
    uint32_t n_expert, uint32_t top_k, uint32_t a_ld, uint32_t as_ld, uint32_t y_ld,            \
    uint32_t mode, const MoeTab tab
template <uint32_t MT, uint32_t NTL, uint32_t WGM, uint32_t WGN, int W4 = 0,
          bool ES = false, bool AL = false, bool SK = false, uint32_t FK = kFK, bool UW = false,
          bool RS = false, bool I8 = false>
__device__ __forceinline__ void moe_gemm_q_body(R4D_MOEGQ_PARAMS) {
    /* `mode`: bit 0 `a_sorted`, bits 1-2 the class shift (1 for two classes, 2 for four), bit 3
     * the activation's scale per 256 of K instead of per 128 (`a_group`: block b reads scale
     * b >> 1), bits 8+ `parts`. */
    const uint32_t a_sorted = mode & 1u, cshift = (mode >> 1) & 3u, parts = mode >> 8;
    const uint32_t ash = (mode >> 3) & 1u;
    constexpr uint32_t FM = MT * 16, FN = NTL * 16;
    constexpr uint32_t NTHR = WGM * WGN * kWave;
    constexpr uint32_t WM = MT / WGM, WN = NTL / WGN;
    static_assert(MT % WGM == 0 && NTL % WGN == 0, "the wave grid must divide the tile");
    /* The block-fp8 weight has one scale per 128x128 tile; a w4 weight one per row, so its block
     * may span two fp8 scale rows (the aligned form's 256-column down tile). */
    /* I8: the activation is int8 and the WMMA the iu8 one, its int32 sums read as numbers at the
     * fold (r4d_fp8_wmma.h, mma8). The weight is then int8 too: at W4 0 the codes themselves with
     * one bf16 scale per (row, 128 of K), the four-bit forms' scale grid, and at W4 1 to 3 the
     * levels as int8 (moe_w4_bytes). kRowS: the weight's scale varies per row. */
    constexpr bool kRowS = W4 != 0 || I8;
    static_assert(!I8 || W4 <= 3, "the five-bit levels have no int8 form");
    static_assert(kRowS || kBlock % FN == 0,
                  "a block's N rows must sit in one 128-row weight scale row");
    /* Scale groups per 128-wide block: one, or the 64- or 32-column groups at W4 2-4. */
    constexpr uint32_t kSub = W4 == 3 ? 4u : (W4 == 2 || W4 == 4) ? 2u : 1u;
    /* What multiplies a subgroup's E4M3 scale: the format's fixed second level, and at I8 the
     * levels' unit. */
    constexpr float kSubScale = I8 ? kW4G64Scale * kW4nlI8Unit : kW4G64Scale;
    /* 128-wide blocks per staged pass (one, or two at FK 256), and scale groups per pass. */
    constexpr uint32_t kBpS  = FK / kBlock;
    constexpr uint32_t kSubT = kSub * kBpS;
    static_assert(FK == kFK || (FK == 2 * kFK && kSub > 1),
                  "FK is one scale block, or two where the fold already sums groups (W4 2 to 4)");
    static_assert(FK % (kSubT * r4d_fp8::kK) == 0, "a scale group is whole WMMA K steps");
    static_assert(FM + FN <= NTHR, "a thread stages each row's and each column's scale: see "
                                   "THE SCALES RIDE WITH THE TILE");

    __shared__ uint8_t s_a[FM * lds_pitch<FK>()];
    __shared__ uint8_t s_b[FN * lds_pitch<FK>()];
    __shared__ int32_t s_tok[FM];
    /* The staged K block's scales: one per activation row, and one per weight column (w4) or one
     * per block (fp8), for each 128-wide block of the pass. See "the scales ride with the tile" at
     * the K loop. */
    __shared__ float   s_as[kBpS * FM];
    __shared__ float   s_ws[kSubT * FN];

    const uint32_t tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
    const uint32_t wgm = wave / WGN, wgn = wave % WGN;
    /* With ES the x axis is (tile, run) with run minor, so a tile's blocks, which share activation
     * rows, land on neighbouring CUs. */
    const uint32_t jrun = ES ? (blockIdx.x % FM) : 0u;
    /* Under AL the map names the block: entry i is (expert, first row) and the block serves only
     * that run. The grid is the upper bound `n_expert + T/FM`; blocks past the real count exit. */
    const uint32_t n0 = blockIdx.y * FN;
    uint32_t r0, rN, e_al = 0;
    [[maybe_unused]] uint32_t al_end = 0;      /* AL: the run's end, off[e_al + 1] */
    if constexpr (AL) {
        if (blockIdx.x >= map[0]) return;
        const uint32_t* ent = map + 1 + 2 * blockIdx.x;
        e_al = ent[0];
        r0   = ent[1];
        /* A tail block owns no run: it zeroes the rows past the live count and leaves before the
         * early weight loads below. Block-uniform, so those loads stay on one path. */
        if (e_al == kMapTail) {
            const uint32_t live = (uint32_t)off[n_expert];
            const uint32_t rT = (r0 + FM < T) ? (r0 + FM) : T;
            for (uint32_t r = (r0 > live ? r0 : live); r < rT; ++r)
                for (uint32_t c = tid; c < FN && n0 + c < N; c += NTHR)
                    Y[(size_t)r * y_ld + n0 + c] = 0;
            return;
        }
        /* The run's end, clamped to the output's rows (an operand may name fewer rows than its
         * buffer, never more). */
        const uint32_t oe = (uint32_t)off[e_al + 1];
        al_end = oe < T ? oe : T;
        rN = (r0 + FM < al_end) ? (r0 + FM) : al_end;
    } else {
        r0 = (ES ? (blockIdx.x / FM) : blockIdx.x) * FM;
        if (r0 >= T) return;
        rN = (r0 + FM < T) ? (r0 + FM) : T;
    }

    /* The aligned form's first pass of weight (`ew_`, the early weight), issued as soon as the map
     * names the expert. Nothing the weight's address needs comes later, and in the run loop it
     * would wait behind four round trips (tokens, barrier, live count, table entry) on a block
     * whose whole K is one or two passes at the down projection.
     *
     * Loads go out in the order their waits want: the block's sorted rows and the live count, then
     * the (uniform, scalar) table entries and the first pass's weight. The vector-load counter
     * retires in order, so waiting for the tokens leaves the weight in flight, and the token
     * barrier orders LDS only. The weight uses UW's uniform global base (a flat load would also
     * count against the LDS counter). The first issue() takes these registers. Bit-identical.
     *
     * Every early load is unconditional: a load under an exec mask makes the waitcnt pass wait for
     * all of them. A dead row reads the expert's row 0 and a dead token slot a real one; the zero
     * is selected where the value is used. */
    constexpr uint32_t kEwTPR = FK / 32, kEwRPP = NTHR / kEwTPR;
    constexpr uint32_t kEwBP = (FN + kEwRPP - 1) / kEwRPP;
    [[maybe_unused]] uint4    ew_b4[AL && W4 ? kEwBP : 1];
    [[maybe_unused]] uint32_t ew_sp[AL && W4 == 4 ? kEwBP : 1];
    [[maybe_unused]] Stage8   ew_b8[AL && !W4 ? kEwBP : 1];
    [[maybe_unused]] bool     ew_ok[AL ? kEwBP : 1];
    [[maybe_unused]] const uint8_t*  ew_wp  = nullptr;
    [[maybe_unused]] const uint16_t* ew_wsp = nullptr;
    [[maybe_unused]] int32_t  ew_f = -1;
    [[maybe_unused]] uint32_t ew_live = 0;
    if constexpr (AL) {
        static_assert(!ES && !RS && UW && FM <= NTHR,
                      "the aligned form issues its first pass early, on UW's uniform weight base");
        {
            const uint32_t i = r0 + tid % FM;           /* a real row whatever the thread */
            ew_f = sorted[i < T ? i : T - 1u];
        }
        ew_live = (uint32_t)off[n_expert];
        /* Keep these two loads first: the counter retires in order. */
        __builtin_amdgcn_sched_barrier(0);
        const uint32_t cls = e_al & ((1u << cshift) - 1u);
        const MoeCls& C = cls == 0u ? c_even : cls == 1u ? c_odd : cls == 2u ? c_2 : c_3;
        const uint32_t fy = N / parts, part = n0 / fy, j0 = n0 - part * fy;
        const uint32_t fc = C.n / parts, wn0 = part * fc + j0;
        ew_wp  = uw_ptr<true>(wtab_at(C.w, C.w_s0, e_al >> cshift));
        ew_wsp = uw_ptr<true>((const uint16_t*)wtab_at(C.s, C.s_s0, e_al >> cshift));
        const uint32_t sn = tid / kEwTPR, sk = (tid % kEwTPR) * 32u;
        /* When a staging pass covers the tile's rows exactly every thread's rows exist, so there
         * is no guard and no exec mask around a load. */
#pragma unroll
        for (uint32_t p = 0; p < kEwBP; ++p) {
            const uint32_t r = sn + p * kEwRPP;
            if constexpr (FN % kEwRPP != 0) { if (r >= FN) continue; }
            ew_ok[p] = (j0 + r) < fc;
            const uint8_t* wr = ew_wp + (size_t)(ew_ok[p] ? wn0 + r : 0u) * C.w_ld;
            if constexpr (W4) {
                ew_b4[p] = moe_w4_load<W4, true>(ew_wp, ew_ok[p] ? wn0 + r : 0u,
                                                C.w_ld, sk, true);
                if constexpr (W4 == 4)
                    ew_sp[p] = u32_load_w<true>((const uint32_t*)(wr + (C.k >> 1) + (sk >> 3)));
            } else
                ew_b8[p] = stage8_load_w<true>(wr + sk, true);
        }
    }

    /* RS: this thread's NPT experts' offsets (plus the next), which of them hold a run meeting the
     * tile, and the wave's running count of those, carried to the run search below. */
    constexpr uint32_t NPT = RS ? kMaxExpert / NTHR : 1;
    static_assert(!RS || (ES && kMaxExpert % NTHR == 0 && FM <= NTHR),
                  "RS is the decode grid's search: the offsets divide over the block's threads");
    __shared__ uint32_t s_rc[RS ? NTHR / kWave : 1];
    __shared__ uint32_t s_sel[RS ? 3 : 1];
    [[maybe_unused]] uint32_t rs_ov[NPT + 1], rs_pm = 0, rs_cnt = 0, rs_inc = 0;

    uint32_t live;
    if constexpr (RS) {
        /* Every load the search needs, issued at once (sorted rows, this thread's offsets, live
         * count): one round trip, then one barrier publishes tokens and wave counts together. */
        int32_t f = -1;
        if (tid < FM && r0 + tid < rN) f = sorted[r0 + tid];
#pragma unroll
        for (uint32_t k = 0; k <= NPT; ++k) {
            const uint32_t x = tid * NPT + k;
            rs_ov[k] = x <= n_expert ? (uint32_t)off[x] : 0u;
        }
        live = (uint32_t)off[n_expert];
        if (tid < FM)
            s_tok[tid] = f < 0 ? -1 : (a_sorted ? (int32_t)(r0 + tid) : f / (int32_t)top_k);
#pragma unroll
        for (uint32_t k = 0; k < NPT; ++k) {
            const uint32_t x = tid * NPT + k;
            const bool p = x < n_expert && rs_ov[k] < rs_ov[k + 1] && rs_ov[k] < rN &&
                           rs_ov[k + 1] > r0;
            rs_pm |= (uint32_t)p << k;
            rs_cnt += p ? 1u : 0u;
        }
        rs_inc = (uint32_t)wave_incl_scan_i32((int32_t)rs_cnt, lane);
        if (lane == kWave - 1) s_rc[wave] = rs_inc;
        /* Both RS barriers order LDS only. */
        lds_barrier();
    } else if constexpr (AL) {
        /* Tokens and live count were loaded ahead of the early weight (see above). A row past the
         * block's is a dropped slot, as in the plain form. */
        if (tid < FM) {
            const int32_t f = r0 + tid < rN ? ew_f : -1;
            s_tok[tid] = f < 0 ? -1 : (a_sorted ? (int32_t)(r0 + tid) : f / (int32_t)top_k);
        }
        lds_barrier();
        live = ew_live;
    } else {
    /* The token each of the block's sorted rows reads its activation from, resolved once. -1 marks
     * a dropped slot, whose output row is zero. */
    for (uint32_t r = tid; r < FM; r += NTHR) {
        const uint32_t i = r0 + r;
        int32_t t = -1;
        if (i < rN) {
            const int32_t f = sorted[i];
            t = f < 0 ? -1 : (a_sorted ? (int32_t)i : f / (int32_t)top_k);
        }
        s_tok[r] = t;
    }
    __syncthreads();

    live = (uint32_t)off[n_expert];
    }
    /* Rows past the live count are dropped slots, zeroed here. No run owns them, so under ES only
     * one of the tile's blocks writes them (identical concurrent writes are still a race). Under AL
     * the tail blocks zeroed them at the map read. */
    if constexpr (!AL) {
        if (!ES || jrun == 0)
            for (uint32_t r = (r0 > live ? r0 : live); r < rN; ++r)
                for (uint32_t c = tid; c < FN && n0 + c < N; c += NTHR)
                    Y[(size_t)r * y_ld + n0 + c] = 0;
    }

    /* The expert of the block's first run. The row-tile grid binary-searches the offsets for its
     * first row's expert, and the loop below walks runs from there.
     *
     * Under ES that walk would dominate: at decode a few dozen slots spread over hundreds of
     * experts, so reaching run `jrun` would step over many empty experts one dependent load at a
     * time. Instead the block finds its tile's runs in one parallel pass: each thread tests one
     * expert for rows meeting this tile, and a ballot plus wave counts place hits in expert (= row)
     * order, so run `jrun` is entry `jrun`. Chunks wholly before or after the tile are skipped with
     * one broadcast load, and the pass stops at the block's run. Bit-identical.
     */
    uint32_t e = 0;
    [[maybe_unused]] uint32_t rs_first = 0, rs_end = 0;
    if constexpr (AL) {
        e = e_al;                      /* the map already named it */
    } else if constexpr (RS) {
        /* RS: the same search in one round trip and two barriers, where the chunked pass costs four
         * to six round trips to L2 before a block knows its expert. Each thread holds NPT
         * consecutive experts' offsets from the batch above; hits are numbered in expert order (a
         * thread's own, then a DPP running count over the wave and the waves before it), the same
         * numbering as the chunked pass. The owner of run `jrun` hands the expert and the run's
         * bounds over LDS, so `off[e]` and `off[e + 1]` are not reloaded. Bit-identical. */
        uint32_t before = rs_inc - rs_cnt, nrun = 0;
        for (uint32_t w = 0; w < NTHR / kWave; ++w) {
            if (w < wave) before += s_rc[w];
            nrun += s_rc[w];
        }
        if (jrun >= before && jrun < before + rs_cnt) {
            uint32_t m = jrun - before;
#pragma unroll
            for (uint32_t k = 0; k < NPT; ++k) {
                if (!((rs_pm >> k) & 1u)) continue;
                if (m == 0) {
                    s_sel[0] = tid * NPT + k;
                    s_sel[1] = rs_ov[k] > r0 ? rs_ov[k] : r0;
                    s_sel[2] = rs_ov[k + 1];
                }
                --m;
            }
        }
        lds_barrier();
        if (jrun >= nrun) return;       /* the tile has fewer runs than this block's index */
        e = s_sel[0];
        rs_first = s_sel[1];
        rs_end = s_sel[2];
    } else if constexpr (ES) {
        /* The chunked search serves the decode form above kMaxExpert experts (RS's registers are
         * sized for kMaxExpert), walking NTHR experts at a time. r4d_selftest checks it at 640
         * experts against the uniform prefill grid. */
        __shared__ int32_t  s_run[FM];
        __shared__ uint32_t s_wc[NTHR / kWave];
        uint32_t nrun = 0;
        for (uint32_t c = 0; c < n_expert && nrun <= jrun; c += NTHR) {
            const uint32_t ce = c + NTHR < n_expert ? c + NTHR : n_expert;
            if ((uint32_t)off[ce] <= r0) continue;          /* every expert here ends before it */
            if ((uint32_t)off[c] >= rN) break;              /* and every one from here after it */
            const uint32_t x = c + tid;
            bool p = false;
            if (x < ce) {
                const uint32_t a = (uint32_t)off[x], b = (uint32_t)off[x + 1];
                p = a < b && a < rN && b > r0;
            }
            const unsigned long long mk = __ballot(p);
            if (lane == 0) s_wc[wave] = (uint32_t)__popcll(mk);
            __syncthreads();
            uint32_t before = nrun, tot = 0;
            for (uint32_t w = 0; w < NTHR / kWave; ++w) {
                if (w < wave) before += s_wc[w];
                tot += s_wc[w];
            }
            const uint32_t pos = before + (uint32_t)__popcll(mk & ((1ull << lane) - 1ull));
            if (p && pos < FM) s_run[pos] = (int32_t)x;
            __syncthreads();
            nrun += tot;
        }
        if (jrun >= nrun) return;       /* the tile has fewer runs than this block's index */
        e = (uint32_t)s_run[jrun];
    } else {
        uint32_t lo = 0, hi = n_expert;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if ((uint32_t)off[mid] <= r0) lo = mid; else hi = mid;
        }
        e = lo;
    }

    /* The output is `parts` stacked parts of N / parts columns, as is every weight (n / parts rows
     * per part). A weight part narrower than the output's fills the part's leading columns and the
     * rest are zero: a rank's two-block slice of a gate/up pair in the three-block layout the gated
     * quantiser reads, whose output there must be zero. With one part this is the whole weight at
     * column n0. */
    const uint32_t fy = N / parts;
    const uint32_t part = n0 / fy, j0 = n0 - part * fy;

    constexpr uint32_t kSeg = 32;                     /* bytes a thread stages, contiguous */
    constexpr uint32_t kTPR = FK / kSeg;              /* threads covering one staged row */
    constexpr uint32_t kRPP = NTHR / kTPR;            /* rows one staging pass covers */
    constexpr uint32_t kAP  = (FM + kRPP - 1) / kRPP;
    constexpr uint32_t kBP  = (FN + kRPP - 1) / kRPP;
    const uint32_t sn = tid / kTPR, sk = (tid % kTPR) * kSeg;

    /* Under ES the loop starts at the block's run: its first row in the tile, run count `jrun`. */
    uint32_t rs = r0, run = 0;
    if constexpr (RS) {
        rs = rs_first;
        run = jrun;
    } else if constexpr (ES) {
        const uint32_t a = (uint32_t)off[e];
        rs = a > r0 ? a : r0;
        run = jrun;
    }
    while (rs < rN && rs < live) {
        /* This run: rows [rs, re) all belong to expert `e`. The advance is at the top so `e` never
         * passes the last expert (`off[e + 1]` is read only for e < n_expert). Under RS the search
         * supplied the run's end. */
        if constexpr (!RS && !AL) {
            while (e < n_expert && (uint32_t)off[e + 1] <= rs) ++e;
            if (e >= n_expert) break;
        }
        /* AL: the map's run is e_al's from r0, with its end already loaded. */
        const uint32_t ehi = RS ? rs_end : AL ? al_end : (uint32_t)off[e + 1];
        const uint32_t re  = rN < live ? (rN < ehi ? rN : ehi) : (live < ehi ? live : ehi);
        if (re <= rs) break;
        /* Never true (under ES the loop starts at the block's run), but kept for the fp8 weight
         * (W4 0) for the back edge it creates: with it LLVM 24's waitcnt placement serialises the
         * four-bit forms' first pass, without it the fp8 form's K loop (11-13% slower at one and two
         * tokens). Each form is built the way it measured faster. */
        if (ES && W4 == 0 && run != jrun) { rs = re; ++run; continue; }

        /* The expert's class and where this block's columns sit in its weight. Block-uniform: a pad
         * tile zeroes the run's rows and moves on without LDS or a barrier. `cshift` is 1 for two
         * classes, 2 for four; c_2 and c_3 are unused at two. */
        const uint32_t cls = e & ((1u << cshift) - 1u);
        const MoeCls& C = cls == 0u ? c_even : cls == 1u ? c_odd : cls == 2u ? c_2 : c_3;
        const uint32_t fc = C.n / parts;
        if (j0 >= fc) {
            const uint32_t nr = re - rs;
            for (uint32_t x = tid; x < nr * FN; x += NTHR) {
                const uint32_t c = x % FN;
                if (n0 + c < N) Y[(size_t)(rs + x / FN) * y_ld + n0 + c] = 0;
            }
            rs = re;
            ++run;
            if (ES || AL) break;
            continue;
        }
        const uint32_t wn0 = part * fc + j0;          /* the weight row of the block's column 0 */
        const uint32_t K = C.k, w_ld = C.w_ld, ws_ld = C.s_ld;
        const uint32_t kb_n = wn0 / kBlock;           /* the weight scale's N row for this block */
        if constexpr (kRowS) (void)kb_n;              /* these scales vary per ROW, not per N block */

        /* Both table entries load before either is made uniform, so the two readfirstlanes wait on
         * one round trip. */
        const uint8_t*  Wp0;
        const uint16_t* Wsp0;
        if constexpr (AL) {
            Wp0  = ew_wp;                 /* loaded at the map read; AL's run is e_al's */
            Wsp0 = ew_wsp;
        } else {
            Wp0  = wtab_at(C.w, C.w_s0, e >> cshift);
            Wsp0 = (const uint16_t*)wtab_at(C.s, C.s_s0, e >> cshift);
        }
        const uint8_t*  Wp   = uw_ptr<UW>(Wp0);
        const uint16_t* Wsp  = uw_ptr<UW>(Wsp0);

        /* SK: a wave whose M tiles are all past the run has nothing to accumulate (wave-uniform
         * test); it still stages and meets every barrier, skipping only fragment reads, WMMA and
         * scale fold. */
        const bool wlive = !SK || (r0 + wgm * WM * 16 < re);
        f8v raw[WM][WN], out[WM][WN];
        /* The other scale groups' accumulators at W4 2-4 (`raw` is the first's), for every block of
         * the pass; one unused element otherwise. Each starts fresh at its group's first WMMA, so
         * none is zeroed. */
        f8v rx[kSubT > 1 ? kSubT - 1 : 1][kSubT > 1 ? WM : 1][kSubT > 1 ? WN : 1];
#pragma unroll
        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
            for (uint32_t j = 0; j < WN; ++j) { raw[i][j] = acc_zero(); out[i][j] = acc_zero(); }

        /* The K loop is software pipelined. Staged straight through, each of K/kFK = 20 iterations
         * exposes its global latency, with under two blocks per CU at decode to cover it (~245 GB/s
         * against the part's 486 cold). So the tile is loaded into registers one iteration ahead:
         * tile i+1's loads issue right after tile i reaches LDS and are collected at the next LDS
         * store, a barrier and a WMMA block later, for sixteen VGPRs. Bit-identical.
         *
         * Do not go deeper than one tile: a two-slot ring puts the arrays in scratch (running
         * index) or costs 16 VGPRs and a doubled body (unrolled). At decode the kernel is
         * critical-path bound on the 20-step K chain, which only a split-K would shorten, and that
         * is not bit-identical. */
        /* The scales ride with the tile: the fold needs each row's activation scale and each
         * column's weight scale, which loaded after the WMMA would be scattered dependent loads. So
         * the thread staging a row or column loads its scale with the tile, one K block ahead, and
         * the fold reads them from LDS. Bit-identical. */
        Stage8 apre[kAP];
        uint4  bpre4[kBP];      /* W4: packed nibbles, expanded at store time as before */
        uint32_t spre[W4 == 4 ? kBP : 1];   /* W4 4: the same 32 elements' sign bits */
        Stage8 bpre8[kBP];      /* otherwise: the fp8 bytes themselves */
        /* kUL: whether each staged row is live. Under kUL every row's tile is loaded (a dead row
         * reads a row that exists) and the zero is selected at the LDS store. A guarded load writes
         * its zero into registers the previous pass's load filled, and the waitcnt pass then waits
         * for every load in flight (15% of the aligned down projection's time); unconditional loads
         * retire in order and need no wait. ES keeps the guarded loads (1.5% faster there). */
        constexpr bool kUL = UW && AL;
        [[maybe_unused]] bool aok[kUL ? kAP : 1];
        [[maybe_unused]] bool bok[kUL ? kBP : 1];
        /* thread tid < FM: its row's activation scale, one per 128-wide block of the pass */
        float    pas[kBpS] = {};
        /* thread FM + c: column c's weight scale (w4) or the block's (fp8), one per 128-wide block
         * of the pass. At W4 3 it is 32 bits (four packed E4M3 groups). Otherwise 16-bit, held
         * zero-extended in a whole register under AL (see u16_load_w); the other forms measured 1%
         * slower that way. */
        using PwsT = std::conditional_t<AL || W4 == 3, uint32_t, uint16_t>;
        PwsT pws[kBpS] = {};
        /* UW: the thread's scale as loaded and whether it is live; the zero is selected where it is
         * staged. One register per role, deliberately uninitialised: an initialised role is
         * provably zero in the other role's branch, and the compiler's write of that zero would
         * wait for the other branch's load (both branches run in one wave). */
        float pau[UW ? kBpS : 1];
        PwsT  pwu[UW ? kBpS : 1];
        bool  pok = false;
        /* `first` marks the cc 0 pass as a type, not a runtime test: the compiler cannot rule out
         * `cc == 0` in issue(c0 + FK) (unsigned wrap) and would keep the ew_ registers live across
         * every pass, starving the WMMA block of sixteen VGPRs. */
        auto issue = [&](uint32_t cc, auto first) {
            const uint32_t kbk = cc / kBlock;
            if constexpr (UW) {
                /* Under UW the scale loads are unconditional (from an existing row when this one is
                 * dead) and the zero is selected at the LDS store; selecting it here would write a
                 * register another lane's load is filling and force a wait before the pass's other
                 * loads. */
                if (tid < FM) {
                    const int32_t tok = s_tok[tid];
                    pok = tok >= 0 && (r0 + tid) >= rs && (r0 + tid) < re;
                    const size_t ar = (size_t)(pok ? tok : 0) * as_ld;
#pragma unroll
                    for (uint32_t h = 0; h < kBpS; ++h) pau[h] = As[ar + ((kbk + h) >> ash)];
                } else if (tid < FM + FN) {
                    const uint32_t nn = tid - FM;
                    pok = j0 + nn < fc;
                    const size_t wr = (size_t)(pok ? wn0 + nn : wn0) * ws_ld + kbk;
#pragma unroll
                    for (uint32_t h = 0; h < kBpS; ++h) {
                        if constexpr (W4 == 3)
                            pwu[h] = u32_load_w<UW>((const uint32_t*)Wsp + wr + h);
                        else if constexpr (kRowS)
                            pwu[h] = u16_load_w<UW>(Wsp + wr + h);
                        else
                            pwu[h] = u16_load_w<UW>(Wsp + (size_t)kb_n * ws_ld + kbk + h);
                    }
                }
            } else if (tid < FM) {
                const int32_t tok = s_tok[tid];
                const bool    ok  = tok >= 0 && (r0 + tid) >= rs && (r0 + tid) < re;
#pragma unroll
                for (uint32_t h = 0; h < kBpS; ++h)
                    pas[h] = ok ? As[(size_t)tok * as_ld + ((kbk + h) >> ash)] : 0.0f;
            } else if (tid < FM + FN) {
                const uint32_t nn = tid - FM;
#pragma unroll
                for (uint32_t h = 0; h < kBpS; ++h) {
                    if constexpr (W4 == 3)
                        pws[h] = j0 + nn < fc
                                     ? u32_load_w<UW>((const uint32_t*)Wsp +
                                                      (size_t)(wn0 + nn) * ws_ld + kbk + h)
                                     : 0u;
                    else if constexpr (kRowS)
                        pws[h] = j0 + nn < fc
                                     ? u16_load_w<UW>(Wsp + (size_t)(wn0 + nn) * ws_ld + kbk + h)
                                     : 0u;
                    else if (nn == 0)
                        pws[h] = u16_load_w<UW>(Wsp + (size_t)kb_n * ws_ld + kbk + h);
                }
            }
#pragma unroll
            for (uint32_t p = 0; p < kAP; ++p) {
                const uint32_t r = sn + p * kRPP;
                if (r >= FM) continue;
                const int32_t tok = s_tok[r];
                const bool    ok  = tok >= 0 && (r0 + r) >= rs && (r0 + r) < re;
                const uint8_t* src = Aq + (size_t)(ok ? tok : 0) * a_ld + cc + sk;
                if constexpr (kUL) {
                    aok[p]  = ok;
                    apre[p] = stage8_load_w<UW>(src, true);
                } else
                    apre[p] = stage8_load(src, ok);
            }
            if constexpr (AL && decltype(first)::value) {
                /* the first pass's weight is already in flight */
                static_assert(kEwBP == kBP && kEwRPP == kRPP,
                              "the early weight stages the pass issue() would");
#pragma unroll
                for (uint32_t p = 0; p < kBP; ++p) {
                    if (sn + p * kRPP >= FN) continue;
                    if constexpr (kUL) {
                        /* loaded from a row that exists; the store selects the dead rows' zero */
                        bok[p] = ew_ok[p];
                        if constexpr (W4) {
                            bpre4[p] = ew_b4[p];
                            if constexpr (W4 == 4) spre[p] = ew_sp[p];
                        } else
                            bpre8[p] = ew_b8[p];
                        continue;
                    }
                    /* The dead rows' zeros, as w4_load / stage8_load give them. */
                    if constexpr (W4) {
                        bpre4[p] = ew_ok[p] ? ew_b4[p] : make_uint4(0, 0, 0, 0);
                        if constexpr (W4 == 4) spre[p] = ew_ok[p] ? ew_sp[p] : 0u;
                    } else {
                        bpre8[p].v = ew_ok[p] ? ew_b8[p].v : make_uint4(0, 0, 0, 0);
                        bpre8[p].w = ew_ok[p] ? ew_b8[p].w : make_uint4(0, 0, 0, 0);
                    }
                }
                return;
            }
#pragma unroll
            for (uint32_t p = 0; p < kBP; ++p) {
                const uint32_t r = sn + p * kRPP;
                if (r >= FN) continue;
                const bool ok = (j0 + r) < fc;
                /* W4: half the bytes of the row, expanded on the way in; `w_ld` is already the
                 * packed byte pitch. At W4 4 the row's sign plane (one bit per element) follows its
                 * K/2 bytes of nibbles. Under kUL a dead row reads the tile's first row and the
                 * store selects its zero (see bok). */
                const uint8_t* wr = Wp + (size_t)(kUL && !ok ? wn0 : wn0 + r) * w_ld;
                if constexpr (kUL) bok[p] = ok;
                if constexpr (W4) {
                    bpre4[p] = moe_w4_load<W4, UW>(Wp, kUL && !ok ? wn0 : wn0 + r,
                                                  w_ld, cc + sk, kUL || ok);
                    if constexpr (W4 == 4)
                        spre[p] = kUL || ok ? u32_load_w<UW>((const uint32_t*)(wr + (K >> 1) +
                                                                             ((cc + sk) >> 3)))
                                           : 0u;
                } else
                    bpre8[p] = stage8_load_w<UW>(wr + cc + sk, kUL || ok);
            }
        };
        /* `s_tok` is settled by the barrier above the run loop and never rewritten, so priming
         * before the loop's own barrier is safe. */
        issue(0u, std::true_type{});

        /* The K loop's barriers order LDS only; no thread reads another's global writes inside it.
         * __syncthreads() fences every address space, which on gfx12 waits for the prefetch right
         * after issuing it, so under UW the barriers are lds_barrier() and the loads stay in flight
         * until the next pass's LDS store.
         *
         * That collection is one explicit wait where the stores begin. Left to the waitcnt pass,
         * the wait sits inside the stores' exec-masked branches, and a wave skipping one would
         * guard the next pass's load with a full wait, serialising the loads again. */
        auto kbar = [] {
            if constexpr (UW) lds_barrier();
            else __syncthreads();
        };
        for (uint32_t c0 = 0; c0 < K; c0 += FK) {
            kbar();                                /* the previous run's reads are done */
            /* The wait, as a global release fence: the only form of `s_wait_loadcnt 0` the waitcnt
             * pass recognises, so it knows the prefetch has landed. No invalidate (that is the
             * acquire half). */
            if constexpr (UW) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
#pragma unroll
            for (uint32_t p = 0; p < kAP; ++p) {
                const uint32_t r = sn + p * kRPP;
                if (r >= FM) continue;
                stage8_store<FK>(s_a, r, sk, kUL ? stage8_keep(apre[p], aok[p]) : apre[p]);
            }
#pragma unroll
            for (uint32_t p = 0; p < kBP; ++p) {
                const uint32_t r = sn + p * kRPP;
                if (r >= FN) continue;
                /* a dead row's zero, where the packed form held it before (see bok) */
                const bool live = !kUL || bok[p];
                const uint4 b4 = live ? bpre4[p] : make_uint4(0, 0, 0, 0);
                if constexpr (W4 == 4)
                    stage8_store<FK>(s_b, r, sk, w5_expand(b4, live ? spre[p] : 0u, tab.v));
                else if constexpr (W4)
                    stage8_store<FK>(s_b, r, sk, moe_w4_bytes<W4, I8>(b4, tab.v));
                else
                    stage8_store<FK>(s_b, r, sk, kUL ? stage8_keep(bpre8[p], live) : bpre8[p]);
            }
            /* The scales as issued; under UW the dead ones are zeroed here (see issue). */
            auto pas_v = [&](uint32_t h) {
                if constexpr (UW) return pok ? pau[h] : 0.0f;
                else return pas[h];
            };
            auto pws_v = [&](uint32_t h) {
                if constexpr (UW) return pok ? pwu[h] : (PwsT)0;
                else return pws[h];
            };
            if (tid < FM) {
#pragma unroll
                for (uint32_t h = 0; h < kBpS; ++h) s_as[h * FM + tid] = pas_v(h);
            } else if (tid < FM + FN) {
                if constexpr (kSub > 1) {
                    /* The tile's scale groups in column order, first in the low byte; at FK 256
                     * the first block's groups, then the second's. */
#pragma unroll
                    for (uint32_t h = 0; h < kBpS; ++h)
#pragma unroll
                        for (uint32_t q = 0; q < kSub; ++q)
                            s_ws[(h * kSub + q) * FN + tid - FM] =
                                r4d_fp8::e4m3((uint32_t)((pws_v(h) >> (8 * q)) & 0xFFu)) *
                                kSubScale;
                } else {
                    s_ws[tid - FM] = (kRowS || tid == FM) ? bscale(pws_v(0)) : 0.0f;
                }
            }
            if (c0 + FK < K) issue(c0 + FK, std::false_type{});
            kbar();

            if (!wlive) continue;
            /* One K step of the tile into `acc`; `fresh` starts from zero instead of the held value
             * (same sum, no reset). */
            auto kstep = [&](uint32_t ks, f8v (&acc)[WM][WN], bool fresh = false) {
                i2v af[WM], bf[WN];
#pragma unroll
                for (uint32_t i = 0; i < WM; ++i) {
                    if (SK && r0 + (wgm * WM + i) * 16 >= re) continue;
                    af[i] = frag<FK>(s_a, (wgm * WM + i) * 16, ks, lane);
                }
#pragma unroll
                for (uint32_t j = 0; j < WN; ++j)
                    bf[j] = frag<FK>(s_b, (wgn * WN + j) * 16, ks, lane);
#pragma unroll
                for (uint32_t i = 0; i < WM; ++i) {
                    if (SK && r0 + (wgm * WM + i) * 16 >= re) continue;
#pragma unroll
                    for (uint32_t j = 0; j < WN; ++j)
                        acc[i][j] = mma8<I8>(af[i], bf[j], fresh ? acc_zero() : acc[i][j]);
                }
            };
            /* At W4 2-4 each scale group of the tile has its own accumulator, so no group's WMMA
             * waits on the previous group's fold; all fold once at the end of the tile, and each
             * group's first step starts its accumulator. At FK 256 group g of the pass is group
             * g % kSub of block g / kSub, in K order. */
            if constexpr (kSubT > 1) {
                constexpr uint32_t kG = FK / kSubT;
#pragma unroll
                for (uint32_t ks = 0; ks < kG; ks += r4d_fp8::kK) kstep(ks, raw, ks == 0);
#pragma unroll
                for (uint32_t q = 1; q < kSubT; ++q)
#pragma unroll
                    for (uint32_t ks = q * kG; ks < (q + 1) * kG; ks += r4d_fp8::kK)
                        kstep(ks, rx[q - 1], ks == q * kG);
            } else {
#pragma unroll
                for (uint32_t ks = 0; ks < FK; ks += r4d_fp8::kK) kstep(ks, raw);
            }
            {
                /* The forms differ only in what the weight scale varies over:
                 *
                 * fp8:  one bf16 per 128x128 tile, uniform over the block, folded into the
                 *       activation scale before the accumulate.
                 * w4:   one bf16 per (row, 128 of K); a 128x128 block scale would spend all sixteen
                 *       levels on its largest row. A fragment's lanes hold sixteen consecutive
                 *       output columns, so the scale is a per-lane constant per N tile.
                 * W4 2-4: groups of 64 or 32 K, two or four per tile, each sum under its scale.
                 *
                 * The fp8 form is written out separately because a * (b * c) and (a * b) * c differ
                 * in the last bit, and its output is byte-checked. */
                float as[kBpS][WM][8];
#pragma unroll
                for (uint32_t h = 0; h < kBpS; ++h)
#pragma unroll
                    for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t)
                            as[h][i][t] = s_as[h * FM + (wgm * WM + i) * 16 + acc_row(lane, t)];
                if constexpr (kSub > 1) {
                    float wq[kSubT][WN];
#pragma unroll
                    for (uint32_t q = 0; q < kSubT; ++q)
#pragma unroll
                        for (uint32_t j = 0; j < WN; ++j)
                            wq[q][j] = s_ws[q * FN + (wgn * WN + j) * 16 + frag_c(lane)];
#pragma unroll
                    for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                        for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                float v = acc_val<I8>(raw[i][j][t]) * wq[0][j];
#pragma unroll
                                for (uint32_t q = 1; q < kSub; ++q)
                                    v += acc_val<I8>(rx[q - 1][i][j][t]) * wq[q][j];
                                out[i][j][t] += v * as[0][i][t];
                                if constexpr (kBpS > 1) {
                                    /* FK 256: the second 128-wide block, folded after the first
                                     * with the expression a separate 128 pass would use, so `out`
                                     * sees the same updates. */
                                    float v1 = acc_val<I8>(rx[kSub - 1][i][j][t]) * wq[kSub][j];
#pragma unroll
                                    for (uint32_t q = 1; q < kSub; ++q)
                                        v1 += acc_val<I8>(rx[kSub + q - 1][i][j][t]) * wq[kSub + q][j];
                                    out[i][j][t] += v1 * as[1][i][t];
                                }
                            }
                } else if constexpr (kRowS) {
                    float wsj[WN];
#pragma unroll
                    for (uint32_t j = 0; j < WN; ++j)
                        wsj[j] = s_ws[(wgn * WN + j) * 16 + frag_c(lane)];
#pragma unroll
                    for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                        for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                out[i][j][t] += acc_val<I8>(raw[i][j][t]) * as[0][i][t] * wsj[j];
                                raw[i][j][t] = 0.0f;
                            }
                } else {
                    const float ws = s_ws[0];
#pragma unroll
                    for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t) as[0][i][t] *= ws;
#pragma unroll
                    for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                        for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                out[i][j][t] += raw[i][j][t] * as[0][i][t];
                                raw[i][j][t] = 0.0f;
                            }
                }
            }
        }

        /* The store: for a fixed accumulator slot sixteen lanes hold sixteen consecutive columns,
         * so it is coalesced across the wave. */
#pragma unroll
        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
            for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    const uint32_t r = (wgm * WM + i) * 16 + acc_row(lane, t);
                    const uint32_t c = (wgn * WN + j) * 16 + frag_c(lane);
                    const uint32_t row = r0 + r;
                    if (r >= FM || row < rs || row >= re || n0 + c >= N) continue;
                    Y[(size_t)row * y_ld + n0 + c] = f32_to_bf16(out[i][j][t]);
                }

        rs = re;
        ++run;
        /* One run per block under ES and AL. Under AL a second iteration would exit at once, but
         * without the break LLVM 24 hoists the early weight's selects to the top and waits for
         * that weight before issuing the pass's other loads. */
        if (ES || AL) break;
    }
}

/* The kernel around the body, whose blocks return where their work ends: CH `r4dchain` is the
 * chained form (r4d_chain.h), every block waiting before its first read and adding once after
 * its last write. */
template <uint32_t MT, uint32_t NTL, uint32_t WGM, uint32_t WGN, int W4, bool ES, bool AL, bool SK,
          uint32_t FK, bool UW, bool RS, bool I8, class CH>
__global__ __launch_bounds__(WGM * WGN * kWave) R4D_WAVES(moe_waves(NTL, W4, ES, AL, FK, RS))
void moe_gemm_q_kernel(R4D_MOEGQ_PARAMS) {
    if (CH::on) r4d_chain_wait();
    moe_gemm_q_body<MT, NTL, WGM, WGN, W4, ES, AL, SK, FK, UW, RS, I8>(
        Aq, As, c_even, c_odd, c_2, c_3, sorted, off, Y, map, T, N, n_expert, top_k, a_ld, as_ld,
        y_ld, mode, tab);
    if (CH::on) r4d_chain_done();
}
#undef R4D_MOEGQ_PARAMS

/* ================================================================== moe_gemm_q, the down decode form
 *
 * The sorted-operand projection at decode: one block per run tile, G column tiles per block.
 *
 * The (tile, run) grid gives down (K 128 or 256 per rank, N the model width) a block per 16-row
 * tile, run slot and 64 columns: 25 600 blocks at sixteen sequences for ~63 MB of weight, most
 * finding no run, the rest staging 4-8 KB behind a search, a table load and a barrier pair. A run
 * crossing a tile boundary is staged twice.
 *
 * Here a block owns one 16-row tile of one run (tiles start at the run's first row, as in the
 * run-aligned prefill form) and G 64-column tiles of output:
 *
 *   - The grid is (run tile, column group). A run of n rows is ceil(n / 16) tiles, at most
 *     min(T, n_expert) + T/16 in all; block j finds the j-th in expert order in one round trip
 *     (four offsets per thread, a DPP count and one exchange), as RS does. One extra last block
 *     zeroes the rows past the live count.
 *   - A run of up to sixteen rows is always one tile, so an expert's weight is staged once per
 *     column tile; a longer run gets one block per tile, side by side.
 *   - All G column tiles' passes form one pipeline over (column tile, K block), each loaded one
 *     pass ahead, so the search/table/first-load chain is paid once per G tiles.
 *
 * G comes from the launcher (kMoeDownDecG2Rows, kMoeDownDecG4Rows): one at a few sequences, up to
 * four at sixteen; at most kMoeDownDecMaxG so the column tiles are bits of one word. Loads are UW
 * throughout (uniform bases, LDS-only barriers), since a barrier waiting on the prefetch would
 * serialise the pipeline.
 *
 * Bit-identical to the (tile, run) form: each output element is the same WMMA chain and fold; the
 * selftest compares them byte for byte.
 *
 * Sorted operand only: a run's activation rows are [rs, re) of `Aq` with no gather. Every row in a
 * run is a placed slot (the scatter fills exactly [0, off[n_expert])), so `sorted` is not read. */
#define R4D_MOEDD_PARAMS                                                                        \
    const uint8_t* __restrict__ Aq, const float* __restrict__ As, const MoeCls c_even,          \
    const MoeCls c_odd, const MoeCls c_2, const MoeCls c_3, const int32_t* __restrict__ off,    \
    uint16_t* __restrict__ Y, uint32_t T, uint32_t N, uint32_t n_expert, uint32_t G,            \
    uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t mode, const MoeTab tab
template <int W4, bool I8 = false>
__device__ __forceinline__ void moe_down_dec_body(R4D_MOEDD_PARAMS) {
    constexpr bool UW = true;
    constexpr uint32_t FM = 16, FN = 64, FK = kFK, NTHR = 4 * kWave;
    constexpr uint32_t kSub = W4 == 3 ? 4u : (W4 == 2 || W4 == 4) ? 2u : 1u;
    /* moe_gemm_q_kernel's I8, with its per-row scale and its subgroup unit. */
    constexpr bool kRowS = W4 != 0 || I8;
    constexpr float kSubScale = I8 ? kW4G64Scale * kW4nlI8Unit : kW4G64Scale;
    static_assert(!I8 || W4 <= 3, "the five-bit levels have no int8 form");
    constexpr uint32_t NPT = kMaxExpert / NTHR;
    static_assert(kMaxExpert % NTHR == 0, "the offsets divide over the block's threads");
    static_assert(FK % (kSub * r4d_fp8::kK) == 0, "a scale group is whole WMMA K steps");
    static_assert(FM + FN <= NTHR, "a thread stages each row's and each column's scale");
    const uint32_t cshift = (mode >> 1) & 3u, parts = mode >> 8, ash = (mode >> 3) & 1u;

    __shared__ uint8_t  s_a[FM * lds_pitch<FK>()];
    __shared__ uint8_t  s_b[FN * lds_pitch<FK>()];
    __shared__ float    s_as[FM];
    __shared__ float    s_ws[kSub * FN];
    __shared__ uint32_t s_rc[NTHR / kWave];
    __shared__ uint32_t s_sel[3];

    const uint32_t tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
    /* The block's columns: G 64-wide tiles from cg0, the last group cut at N. */
    const uint32_t cg0 = blockIdx.y * G * FN;
    if (cg0 >= N) return;               /* the launcher's grid has none; block-uniform */
    const uint32_t cend = cg0 + G * FN < N ? cg0 + G * FN : N;
    const uint32_t ng = (cend - cg0 + FN - 1) / FN;

    /* The last block owns no run: it zeroes every column of the rows past the live count (dropped
     * slots, or protected experts' rows their own GEMM writes afterwards). */
    if (blockIdx.x == gridDim.x - 1) {
        const uint32_t live = (uint32_t)off[n_expert];
        for (uint32_t r = live; r < T; ++r)
            for (uint32_t c = cg0 + tid; c < cend; c += NTHR) Y[(size_t)r * y_ld + c] = 0;
        return;
    }

    /* The tile: the j-th 16-row run tile in expert, then row, order. Each thread holds NPT
     * consecutive experts' offsets plus the next; tiles are numbered by thread, then over the wave
     * (DPP) and the waves before it. Both barriers order LDS.
     *
     * Every offset is clamped to T, so no run extends past the output (an operand may name fewer
     * rows than its buffer, never more) and the tile count stays within the grid. */
    /* Unconditional loads (a thread past the experts re-reads off[n_expert], dropped by a select),
     * so all NPT + 1 issue in one round instead of one round trip each. */
    uint32_t ov[NPT + 1];
#pragma unroll
    for (uint32_t k = 0; k <= NPT; ++k) {
        const uint32_t x = tid * NPT + k;
        const uint32_t raw = (uint32_t)off[x <= n_expert ? x : n_expert];
        const uint32_t o = x <= n_expert ? raw : 0u;
        ov[k] = o < T ? o : T;
    }
    uint32_t nt[NPT], cnt = 0;
#pragma unroll
    for (uint32_t k = 0; k < NPT; ++k) {
        nt[k] = tid * NPT + k < n_expert ? (ov[k + 1] - ov[k] + FM - 1) / FM : 0u;
        cnt += nt[k];
    }
    const uint32_t inc = (uint32_t)wave_incl_scan_i32((int32_t)cnt, lane);
    if (lane == kWave - 1) s_rc[wave] = inc;
    lds_barrier();
    uint32_t before = inc - cnt, ntile = 0;
#pragma unroll
    for (uint32_t w = 0; w < NTHR / kWave; ++w) {
        const uint32_t c = s_rc[w];
        before += w < wave ? c : 0u;
        ntile += c;
    }
    const uint32_t j = blockIdx.x;
    if (j >= ntile) return;             /* fewer tiles than slots: block-uniform */
    if (j >= before && j < before + cnt) {
        uint32_t m = j - before;
        bool hit = false;
#pragma unroll
        for (uint32_t k = 0; k < NPT; ++k) {
            if (!hit && m < nt[k]) {
                s_sel[0] = tid * NPT + k;
                s_sel[1] = ov[k] + m * FM;
                s_sel[2] = ov[k + 1];
                hit = true;
            }
            if (!hit) m -= nt[k];
        }
    }
    lds_barrier();
    /* Block-uniform from here (readfirstlane), so the class, table entry and pass bounds are
     * scalar. The block's rows are [rs, re), at most a tile. */
    const uint32_t e  = (uint32_t)__builtin_amdgcn_readfirstlane((int)s_sel[0]);
    const uint32_t rs = (uint32_t)__builtin_amdgcn_readfirstlane((int)s_sel[1]);
    const uint32_t r1 = (uint32_t)__builtin_amdgcn_readfirstlane((int)s_sel[2]);
    const uint32_t re = rs + FM < r1 ? rs + FM : r1;

    const uint32_t cls = e & ((1u << cshift) - 1u);
    const MoeCls& C = cls == 0u ? c_even : cls == 1u ? c_odd : cls == 2u ? c_2 : c_3;
    const uint32_t fy = N / parts, fc = C.n / parts;
    const uint32_t K = C.k, w_ld = C.w_ld, ws_ld = C.s_ld;
    const uint8_t*  Wp0  = wtab_at(C.w, C.w_s0, e >> cshift);
    const uint16_t* Wsp0 = (const uint16_t*)wtab_at(C.s, C.s_s0, e >> cshift);
    const uint8_t*  Wp   = uw_ptr<UW>(Wp0);
    const uint16_t* Wsp  = uw_ptr<UW>(Wsp0);

    /* A column tile past the class's width holds no weight: its rows are zeroed and it takes no
     * pass. The others are marked in `npm`. */
    uint32_t npm = 0;
    for (uint32_t g = 0; g < ng; ++g) {
        const uint32_t n0 = cg0 + g * FN, part = n0 / fy, j0 = n0 - part * fy;
        if (j0 < fc) { npm |= 1u << g; continue; }
        const uint32_t nr = re - rs;
        for (uint32_t x = tid; x < nr * FN; x += NTHR) {
            const uint32_t c = x % FN;
            if (n0 + c < N) Y[(size_t)(rs + x / FN) * y_ld + n0 + c] = 0;
        }
    }
    if (!npm) return;

    /* Passes over (column tile, K block), K innermost. The last pass's successor is never
     * issued. */
    const uint32_t nK = K / FK;
    const uint32_t P = (uint32_t)__builtin_popcount(npm) * nK;
    struct Pass { uint32_t g, kp; };
    auto advance = [&](Pass s) {
        if (++s.kp < nK) return s;
        s.kp = 0;
        const uint32_t rest = npm & ~((2u << s.g) - 1u);
        s.g = rest ? (uint32_t)__builtin_ctz(rest) : s.g;
        return s;
    };

    constexpr uint32_t kSeg = 32;                     /* bytes a thread stages, contiguous */
    constexpr uint32_t kTPR = FK / kSeg;              /* threads covering one staged row */
    constexpr uint32_t kRPP = NTHR / kTPR;            /* rows one staging pass covers */
    constexpr uint32_t kAP  = (FM + kRPP - 1) / kRPP;
    constexpr uint32_t kBP  = (FN + kRPP - 1) / kRPP;
    const uint32_t sn = tid / kTPR, sk = (tid % kTPR) * kSeg;

    /* Operands one pass ahead, as in moe_gemm_q_kernel's UW path: unconditional scale loads, the
     * zero selected at the LDS store, scale-role registers uninitialised (see the note there). */
    Stage8 apre[kAP];
    uint4  bpre4[kBP];
    uint32_t spre[W4 == 4 ? kBP : 1];
    Stage8 bpre8[kBP];
    using PwsT = uint32_t;              /* a 16-bit scale zero-extended: see u16_load_w */
    float pau[1];
    PwsT  pwu[1];
    bool  pok = false;
    auto issue = [&](const Pass& s) {
        const uint32_t cc = s.kp * FK, kbk = s.kp;    /* FK is one 128-wide scale block */
        const uint32_t n0 = cg0 + s.g * FN, part = n0 / fy, j0 = n0 - part * fy;
        const uint32_t wn0 = part * fc + j0;          /* the weight row of the tile's column 0 */
        const uint32_t kb_n = wn0 / kBlock;
        if constexpr (kRowS) (void)kb_n;
        if (tid < FM) {
            const uint32_t row = rs + tid;
            pok = row < re;
            pau[0] = As[(size_t)(pok ? row : 0u) * as_ld + (kbk >> ash)];
        } else if (tid < FM + FN) {
            const uint32_t nn = tid - FM;
            pok = j0 + nn < fc;
            const size_t wr = (size_t)(pok ? wn0 + nn : wn0) * ws_ld + kbk;
            if constexpr (W4 == 3)
                pwu[0] = u32_load_w<UW>((const uint32_t*)Wsp + wr);
            else if constexpr (kRowS)
                pwu[0] = u16_load_w<UW>(Wsp + wr);
            else
                pwu[0] = u16_load_w<UW>(Wsp + (size_t)kb_n * ws_ld + kbk);
        }
#pragma unroll
        for (uint32_t p = 0; p < kAP; ++p) {
            const uint32_t r = sn + p * kRPP;
            if (r >= FM) continue;
            const uint32_t row = rs + r;
            const bool ok = row < re;
            apre[p] = stage8_load(Aq + (size_t)(ok ? row : 0u) * a_ld + cc + sk, ok);
        }
#pragma unroll
        for (uint32_t p = 0; p < kBP; ++p) {
            const uint32_t r = sn + p * kRPP;
            if (r >= FN) continue;
            const bool ok = (j0 + r) < fc;
            if constexpr (W4) {
                const uint8_t* wr = Wp + (size_t)(wn0 + r) * w_ld;
                bpre4[p] = moe_w4_load<W4, UW>(Wp, wn0 + r, w_ld, cc + sk, ok);
                if constexpr (W4 == 4)
                    spre[p] = ok ? u32_load_w<UW>((const uint32_t*)(wr + (K >> 1) +
                                                                    ((cc + sk) >> 3)))
                                 : 0u;
            } else
                bpre8[p] = stage8_load_w<UW>(Wp + (size_t)(wn0 + r) * w_ld + cc + sk, ok);
        }
    };

    f8v raw = acc_zero(), out = acc_zero();
    /* The other scale groups' accumulators at W4 2 to 4; each starts fresh at its group's first
     * WMMA, so none is zeroed. */
    f8v rx[kSub > 1 ? kSub - 1 : 1];
    Pass cur{(uint32_t)__builtin_ctz(npm), 0u};
    issue(cur);
    for (uint32_t p = 0; p < P; ++p) {
        lds_barrier();                         /* the previous pass's reads are done */
        /* The prefetch collected in one wait, as at moe_gemm_q_kernel's UW K loop head. */
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
#pragma unroll
        for (uint32_t q = 0; q < kAP; ++q) {
            const uint32_t r = sn + q * kRPP;
            if (r >= FM) continue;
            stage8_store<FK>(s_a, r, sk, apre[q]);
        }
#pragma unroll
        for (uint32_t q = 0; q < kBP; ++q) {
            const uint32_t r = sn + q * kRPP;
            if (r >= FN) continue;
            if constexpr (W4 == 4)
                stage8_store<FK>(s_b, r, sk, w5_expand(bpre4[q], spre[q], tab.v));
            else if constexpr (W4)
                stage8_store<FK>(s_b, r, sk, moe_w4_bytes<W4, I8>(bpre4[q], tab.v));
            else
                stage8_store<FK>(s_b, r, sk, bpre8[q]);
        }
        if (tid < FM) {
            s_as[tid] = pok ? pau[0] : 0.0f;
        } else if (tid < FM + FN) {
            const PwsT pws = pok ? pwu[0] : (PwsT)0;
            if constexpr (kSub > 1) {
#pragma unroll
                for (uint32_t q = 0; q < kSub; ++q)
                    s_ws[q * FN + tid - FM] =
                        r4d_fp8::e4m3((uint32_t)((pws >> (8 * q)) & 0xFFu)) * kSubScale;
            } else {
                s_ws[tid - FM] = (kRowS || tid == FM) ? bscale(pws) : 0.0f;
            }
        }
        const Pass nxt = advance(cur);
        if (p + 1 < P) issue(nxt);
        lds_barrier();

        /* moe_gemm_q_kernel's K step and fold at one M tile by one N tile a wave. */
        auto kstep = [&](uint32_t ks, f8v& acc, bool fresh = false) {
            const i2v af = frag<FK>(s_a, 0, ks, lane);
            const i2v bf = frag<FK>(s_b, wave * 16, ks, lane);
            acc = mma8<I8>(af, bf, fresh ? acc_zero() : acc);
        };
        if constexpr (kSub > 1) {
            constexpr uint32_t kG = FK / kSub;
#pragma unroll
            for (uint32_t ks = 0; ks < kG; ks += r4d_fp8::kK) kstep(ks, raw, ks == 0);
#pragma unroll
            for (uint32_t q = 1; q < kSub; ++q)
#pragma unroll
                for (uint32_t ks = q * kG; ks < (q + 1) * kG; ks += r4d_fp8::kK)
                    kstep(ks, rx[q - 1], ks == q * kG);
        } else {
#pragma unroll
            for (uint32_t ks = 0; ks < FK; ks += r4d_fp8::kK) kstep(ks, raw);
        }
        {
            float as[8];
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) as[t] = s_as[acc_row(lane, t)];
            if constexpr (kSub > 1) {
                float wq[kSub];
#pragma unroll
                for (uint32_t q = 0; q < kSub; ++q) wq[q] = s_ws[q * FN + wave * 16 + frag_c(lane)];
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    float v = acc_val<I8>(raw[t]) * wq[0];
#pragma unroll
                    for (uint32_t q = 1; q < kSub; ++q) v += acc_val<I8>(rx[q - 1][t]) * wq[q];
                    out[t] += v * as[t];
                }
            } else if constexpr (kRowS) {
                const float wsj = s_ws[wave * 16 + frag_c(lane)];
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    out[t] += acc_val<I8>(raw[t]) * as[t] * wsj;
                    raw[t] = 0.0f;
                }
            } else {
                const float ws = s_ws[0];
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) as[t] *= ws;
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    out[t] += raw[t] * as[t];
                    raw[t] = 0.0f;
                }
            }
        }

        /* The tile's last K block: its rows of the run, then a fresh accumulator. */
        if (cur.kp + 1 == nK) {
            const uint32_t n0 = cg0 + cur.g * FN;
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                const uint32_t row = rs + acc_row(lane, t);
                const uint32_t c = wave * 16 + frag_c(lane);
                if (row >= re || n0 + c >= N) continue;
                Y[(size_t)row * y_ld + n0 + c] = f32_to_bf16(out[t]);
            }
            out = acc_zero();
        }
        cur = nxt;
    }
}

/* The kernel around the body, as moe_gemm_q_kernel around its own: CH `r4dchain` is the chained
 * form. */
template <int W4, bool I8, class CH>
__global__ __launch_bounds__(4 * kWave)
void moe_down_dec_kernel(R4D_MOEDD_PARAMS) {
    if (CH::on) r4d_chain_wait();
    moe_down_dec_body<W4, I8>(Aq, As, c_even, c_odd, c_2, c_3, off, Y, T, N, n_expert, G, a_ld,
                              as_ld, y_ld, mode, tab);
    if (CH::on) r4d_chain_done();
}
#undef R4D_MOEDD_PARAMS

/* One or a few routed rows at I4 with BF16 scales per 128 K (signed or a 16-entry codebook).
 * A block owns one sorted row and 64 columns;
 * the two halves of each wave read the same activation row and their own eight K bytes directly
 * into WMMA registers. Fragment-tiled weights are coalesced 128-bit loads, spent four times, so no
 * retiling, LDS staging, run scan or workgroup barrier is needed. Only accumulator row zero is
 * folded/stored. Keep the same eight WMMAs per group and raw * activation_scale * weight_scale
 * order as the tiled kernels; the discarded padded rows cannot affect that row.
 *
 * Sorted rows are resolved independently, so singleton, repeated and empty expert runs, both
 * activation orders, parity/mod4 classes and dropped slots all retain their ordinary meaning. */
template <bool I8 = false>
__global__ __launch_bounds__(4 * kWave) void moe_w4_rows_kernel(
    const uint8_t* __restrict__ Aq, const float* __restrict__ As,
    const MoeCls c_even, const MoeCls c_odd, const MoeCls c_2, const MoeCls c_3,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ Y, uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k,
    uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t mode, const MoeTab tab) {
    const uint32_t row = blockIdx.x, n0 = blockIdx.y * 64u;
    const uint32_t tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
    if (row >= T || n0 >= N) return;
    if (row >= (uint32_t)off[n_expert]) {
        if (tid < 64 && n0 + tid < N) Y[(size_t)row * y_ld + n0 + tid] = 0;
        return;
    }
    uint32_t lo = 0, hi = n_expert;
    while (lo + 1 < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if ((uint32_t)off[mid] <= row) lo = mid;
        else hi = mid;
    }
    const uint32_t e = (uint32_t)__builtin_amdgcn_readfirstlane((int)lo);
    const uint32_t cshift = (mode >> 1) & 3u, parts = mode >> 8, ash = (mode >> 3) & 1u;
    const uint32_t cls = e & ((1u << cshift) - 1u);
    const MoeCls& C = cls == 0 ? c_even : cls == 1 ? c_odd : cls == 2 ? c_2 : c_3;
    const uint32_t fy = N / parts, fc = C.n / parts;
    const uint32_t part = n0 / fy, j0 = n0 - part * fy;
    if (j0 >= fc) {
        if (tid < 64 && n0 + tid < N) Y[(size_t)row * y_ld + n0 + tid] = 0;
        return;
    }
    const uint32_t arow = (mode & 1u) ? row : (uint32_t)sorted[row] / top_k;
    const uint32_t col = wave * 16 + frag_c(lane), wn0 = part * fc + j0;
    const uint32_t nt = (uint32_t)__builtin_amdgcn_readfirstlane((int)(wn0 / 16 + wave));
    const uint8_t* Wp = uw_ptr<true>(wtab_at(C.w, C.w_s0, e >> cshift));
    const uint16_t* Ws = uw_ptr<true>((const uint16_t*)wtab_at(C.s, C.s_s0, e >> cshift));
    float out = 0.f;
    for (uint32_t g = 0; g < C.k / kBlock; ++g) {
        const float as = As[(size_t)arow * as_ld + (g >> ash)];
        const float ws = bscale(u16_load_w<true>(Ws + (size_t)(wn0 + col) * C.s_ld + g));
        f8v raw = acc_zero();
#pragma unroll
        for (uint32_t b = 0; b < 2; ++b) {
            const r4d_moe_u4v words = *((r4d_moe_gu4)Wp +
                ((size_t)nt * (C.w_ld / 32) + 2 * g + b) * kWave + lane);
#pragma unroll
            for (uint32_t s = 0; s < 4; ++s) {
                const uint32_t kk = g * kBlock + b * 64 + s * 16 + r4d_fp8::frag_k0(lane);
                const uint2 a = *(const uint2*)(Aq + (size_t)arow * a_ld + kk);
                const uint32_t w = words[s];
                const i2v af = { (int)a.x, (int)a.y };
                const i2v bf = { (int)w4_e4m3(w & 0x0f0f0f0fu, tab.v),
                                 (int)w4_e4m3((w >> 4) & 0x0f0f0f0fu, tab.v) };
                raw = mma8<I8>(af, bf, raw);
            }
        }
        out += acc_val<I8>(raw[0]) * as * ws;
    }
    if (lane < 16 && n0 + col < N) Y[(size_t)row * y_ld + n0 + col] = f32_to_bf16(out);
}

/* ================================================================== moe_gemm, bf16
 *
 * y[i, :] = a[row(i), :] @ w[e]^T in sorted order, bf16 operands and one f32 accumulator over all
 * of K: the expert GEMM for a model served from its bf16 checkpoint (the reference deployment, and
 * the forward pass a calibration run records Hessians through).
 *
 * Grouping is moe_gemm_q_kernel's (AL at prefill, ES at decode, SK, `a_sorted`), so the forms are
 * byte-identical to each other for the same reason. Differences:
 *   - One table, one geometry: no parity classes or `parts`, since a bf16 plane has no scale
 *     blocks to respect.
 *   - The plane is the checkpoint's row-major [N, K] as stored, with no relayout, so an expert can
 *     go from disk through a staging buffer straight to this kernel.
 *   - K is staged kBK16 = 64 elements (128 bytes) per pass with r4d_gemm_bf16_nt_tiled.hip's LDS
 *     pitch of 72, so a fragment's sixteen rows hit distinct bank pairs; the next pass is
 *     prefetched into registers.
 *   - Nothing to fold: the WMMA accumulates across every pass of K.
 */
constexpr uint32_t kBK16 = 64;            // bf16 K elements staged per pass
constexpr uint32_t kP16  = kBK16 + 8;     // the LDS row pitch, in elements

typedef __bf16 mb_v8bf __attribute__((ext_vector_type(8)));
typedef float  mb_v8f  __attribute__((ext_vector_type(8)));

__device__ __forceinline__ mb_v8f mb_mma(mb_v8bf a, mb_v8bf b, mb_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

/* A 16x16 bf16 operand fragment from an LDS tile whose row 0 is `tile`, advanced to the k step.
 * Lane l reads row l % 16 at k = 4 * (l / 16) .. +3 and eight past that (the wave32 bf16 WMMA
 * operand layout, r4d_gdn_wmma.h): two 8-byte reads 16 bytes apart, one ds_load_2addr_b64. */
__device__ __forceinline__ mb_v8bf mb_frag(const uint16_t* __restrict__ tile, uint32_t lane) {
    const uint2* pw = (const uint2*)(tile + (lane & 15u) * kP16 + 4u * (lane >> 4));
    union { uint2 w[2]; mb_v8bf f; } u;
    u.w[0] = pw[0];
    u.w[1] = pw[2];
    return u.f;
}

#define R4D_MOE16_PARAMS                                                                    \
    const uint16_t* __restrict__ A, const void* const* __restrict__ Wt, size_t w_s0,            \
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,                        \
    uint16_t* __restrict__ Y, uint32_t T, uint32_t N, uint32_t K, uint32_t n_expert,            \
    uint32_t top_k, uint32_t a_sorted, uint32_t a_ld, uint32_t w_ld, uint32_t y_ld,             \
    const uint32_t* __restrict__ map
template <uint32_t MT, uint32_t NTL, uint32_t WGM, uint32_t WGN, bool ES = false, bool AL = false,
          bool SK = false>
__device__ __forceinline__ void moe_gemm_bf16_body(R4D_MOE16_PARAMS) {
    constexpr uint32_t FM = MT * 16, FN = NTL * 16;
    constexpr uint32_t NTHR = WGM * WGN * kWave;
    constexpr uint32_t WM = MT / WGM, WN = NTL / WGN;
    /* Sixteen-byte chunks: eight per staged row, and how many a thread carries per tile. */
    constexpr uint32_t kCPR = kBK16 / 8;
    constexpr uint32_t kAC = (FM * kCPR + NTHR - 1) / NTHR;
    constexpr uint32_t kBC = (FN * kCPR + NTHR - 1) / NTHR;
    static_assert(MT % WGM == 0 && NTL % WGN == 0, "the wave grid must divide the tile");

    __shared__ __attribute__((aligned(16))) uint16_t s_a[FM * kP16];
    __shared__ __attribute__((aligned(16))) uint16_t s_b[FN * kP16];
    __shared__ int32_t s_tok[FM];

    const uint32_t tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
    const uint32_t wgm = wave / WGN, wgn = wave % WGN;
    /* The block's rows and run, as in moe_gemm_q_kernel. */
    const uint32_t jrun = ES ? (blockIdx.x % FM) : 0u;
    uint32_t r0, rN, e_al = 0;
    bool tail = false;
    if constexpr (AL) {
        if (blockIdx.x >= map[0]) return;
        const uint32_t* ent = map + 1 + 2 * blockIdx.x;
        e_al = ent[0];
        r0   = ent[1];
        tail = (e_al == kMapTail);
        const uint32_t lim = tail ? T : (uint32_t)off[e_al + 1];
        rN = (r0 + FM < lim) ? (r0 + FM) : lim;
    } else {
        r0 = (ES ? (blockIdx.x / FM) : blockIdx.x) * FM;
        if (r0 >= T) return;
        rN = (r0 + FM < T) ? (r0 + FM) : T;
    }
    const uint32_t n0 = blockIdx.y * FN;

    for (uint32_t r = tid; r < FM; r += NTHR) {
        const uint32_t i = r0 + r;
        int32_t t = -1;
        if (i < rN) {
            const int32_t f = sorted[i];
            t = f < 0 ? -1 : (a_sorted ? (int32_t)i : f / (int32_t)top_k);
        }
        s_tok[r] = t;
    }
    __syncthreads();

    /* Dropped slots past the live count, zeroed by exactly one of a tile's blocks. */
    const uint32_t live = (uint32_t)off[n_expert];
    if (!ES || jrun == 0)
    for (uint32_t r = (r0 > live ? r0 : live); r < rN; ++r)
        for (uint32_t c = tid; c < FN && n0 + c < N; c += NTHR)
            Y[(size_t)r * y_ld + n0 + c] = 0;
    if constexpr (AL) { if (tail) return; }

    /* The expert of the block's first run, found the three ways moe_gemm_q_kernel finds it. */
    uint32_t e = 0;
    if constexpr (AL) {
        e = e_al;
    } else if constexpr (ES) {
        __shared__ int32_t  s_run[FM];
        __shared__ uint32_t s_wc[NTHR / kWave];
        uint32_t nrun = 0;
        for (uint32_t c = 0; c < n_expert && nrun <= jrun; c += NTHR) {
            const uint32_t ce = c + NTHR < n_expert ? c + NTHR : n_expert;
            if ((uint32_t)off[ce] <= r0) continue;
            if ((uint32_t)off[c] >= rN) break;
            const uint32_t x = c + tid;
            bool p = false;
            if (x < ce) {
                const uint32_t lo = (uint32_t)off[x], hi = (uint32_t)off[x + 1];
                p = lo < hi && lo < rN && hi > r0;
            }
            const unsigned long long mk = __ballot(p);
            if (lane == 0) s_wc[wave] = (uint32_t)__popcll(mk);
            __syncthreads();
            uint32_t before = nrun, tot = 0;
            for (uint32_t w = 0; w < NTHR / kWave; ++w) {
                if (w < wave) before += s_wc[w];
                tot += s_wc[w];
            }
            const uint32_t pos = before + (uint32_t)__popcll(mk & ((1ull << lane) - 1ull));
            if (p && pos < FM) s_run[pos] = (int32_t)x;
            __syncthreads();
            nrun += tot;
        }
        if (jrun >= nrun) return;
        e = (uint32_t)s_run[jrun];
    } else {
        uint32_t lo = 0, hi = n_expert;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if ((uint32_t)off[mid] <= r0) lo = mid; else hi = mid;
        }
        e = lo;
    }

    uint32_t rs = r0, run = 0;
    if constexpr (ES) {
        const uint32_t s0 = (uint32_t)off[e];
        rs = s0 > r0 ? s0 : r0;
        run = jrun;
    } else {
        /* A call may serve a trailing segment of the sort (a layer's protected experts, after the
         * four-bit GEMM served the rest); then off[0] > 0 and the rows below belong to another
         * call. The block map and run search never reach them; the plain row grid must skip them. */
        const uint32_t s0 = (uint32_t)off[0];
        rs = s0 > r0 ? s0 : r0;
    }
    while (rs < rN && rs < live) {
        while (e < n_expert && (uint32_t)off[e + 1] <= rs) ++e;
        if (e >= n_expert) break;
        const uint32_t ehi = (uint32_t)off[e + 1];
        const uint32_t re  = rN < live ? (rN < ehi ? rN : ehi) : (live < ehi ? live : ehi);
        if (re <= rs) break;
        if (ES && run != jrun) { rs = re; ++run; continue; }

        const uint16_t* Wp = (const uint16_t*)wtab_at(Wt, w_s0, e);
        const bool wlive = !SK || (r0 + wgm * WM * 16 < re);
        mb_v8f acc[WM][WN];
#pragma unroll
        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
            for (uint32_t j = 0; j < WN; ++j) acc[i][j] = (mb_v8f)(0.0f);

        /* Chunk c of a tile is row c / kCPR, elements (c % kCPR) * 8. A row outside the run, a
         * dropped slot or a column past N stages zeros. Every load is issued (a dead row reads an
         * existing row) and the store selects the zero, since a conditional load would serialise
         * the pass's loads (moe_gemm_q_kernel's K-loop note). */
        uint4 apre[kAC], bpre[kBC];
        bool aok[kAC], bok[kBC];
        auto issue = [&](uint32_t k0) {
#pragma unroll
            for (uint32_t p = 0; p < kAC; ++p) {
                const uint32_t c = tid + p * NTHR;
                if (c >= FM * kCPR) continue;
                const uint32_t r = c / kCPR, kk = (c % kCPR) * 8;
                const int32_t tok = s_tok[r];
                aok[p] = tok >= 0 && (r0 + r) >= rs && (r0 + r) < re;
                apre[p] = *(const uint4*)(A + (size_t)(aok[p] ? tok : 0) * a_ld + k0 + kk);
            }
#pragma unroll
            for (uint32_t p = 0; p < kBC; ++p) {
                const uint32_t c = tid + p * NTHR;
                if (c >= FN * kCPR) continue;
                const uint32_t r = c / kCPR, kk = (c % kCPR) * 8;
                bok[p] = (n0 + r) < N;
                bpre[p] = *(const uint4*)(Wp + (size_t)(bok[p] ? n0 + r : N - 1) * w_ld + k0 + kk);
            }
        };
        issue(0);

        /* The K loop's barriers order LDS only, so they are lds_barrier() (__syncthreads() would
         * wait for the prefetch right after issuing it). The prefetch is collected by one global
         * release fence where the stores begin (moe_gemm_q_kernel's K-loop note). */
        for (uint32_t k0 = 0; k0 < K; k0 += kBK16) {
            lds_barrier();                         /* the previous pass's fragments are read */
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
#pragma unroll
            for (uint32_t p = 0; p < kAC; ++p) {
                const uint32_t c = tid + p * NTHR;
                if (c >= FM * kCPR) continue;
                *(uint4*)(s_a + (c / kCPR) * kP16 + (c % kCPR) * 8) =
                    aok[p] ? apre[p] : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (uint32_t p = 0; p < kBC; ++p) {
                const uint32_t c = tid + p * NTHR;
                if (c >= FN * kCPR) continue;
                *(uint4*)(s_b + (c / kCPR) * kP16 + (c % kCPR) * 8) =
                    bok[p] ? bpre[p] : make_uint4(0, 0, 0, 0);
            }
            if (k0 + kBK16 < K) issue(k0 + kBK16);
            lds_barrier();

            if (!wlive) continue;
#pragma unroll
            for (uint32_t ks = 0; ks < kBK16; ks += 16) {
                mb_v8bf af[WM], bf[WN];
#pragma unroll
                for (uint32_t i = 0; i < WM; ++i) {
                    if (SK && r0 + (wgm * WM + i) * 16 >= re) continue;
                    af[i] = mb_frag(s_a + (wgm * WM + i) * 16 * kP16 + ks, lane);
                }
#pragma unroll
                for (uint32_t j = 0; j < WN; ++j)
                    bf[j] = mb_frag(s_b + (wgn * WN + j) * 16 * kP16 + ks, lane);
#pragma unroll
                for (uint32_t i = 0; i < WM; ++i) {
                    if (SK && r0 + (wgm * WM + i) * 16 >= re) continue;
#pragma unroll
                    for (uint32_t j = 0; j < WN; ++j) acc[i][j] = mb_mma(af[i], bf[j], acc[i][j]);
                }
            }
        }

        /* Lane l holds column l % 16 and rows 8 * (l / 16) + t (the layout of acc_row / frag_c).
         * Rounded to nearest-even. */
#pragma unroll
        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
            for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    const uint32_t r = (wgm * WM + i) * 16 + 8 * (lane >> 4) + t;
                    const uint32_t c = (wgn * WN + j) * 16 + (lane & 15u);
                    const uint32_t row = r0 + r;
                    if (r >= FM || row < rs || row >= re || n0 + c >= N) continue;
                    Y[(size_t)row * y_ld + n0 + c] = f32_to_bf16(acc[i][j][t]);
                }

        rs = re;
        ++run;
        if (ES) break;
    }
}

/* The kernel around the body, as moe_gemm_q_kernel around its own: CH `r4dchain` is the chained
 * form, which the launcher takes at decode heights. */
template <uint32_t MT, uint32_t NTL, uint32_t WGM, uint32_t WGN, bool ES, bool AL, bool SK, class CH>
__global__ __launch_bounds__(WGM * WGN * kWave)
void moe_gemm_bf16_kernel(R4D_MOE16_PARAMS) {
    if (CH::on) r4d_chain_wait();
    moe_gemm_bf16_body<MT, NTL, WGM, WGN, ES, AL, SK>(A, Wt, w_s0, sorted, off, Y, T, N, K, n_expert, top_k, a_sorted, a_ld, w_ld, y_ld, map);
    if (CH::on) r4d_chain_done();
}
#undef R4D_MOE16_PARAMS

/* ================================================================== moe_gather
 *
 * y[t, :] = sum over a token's slots of expert_w[t, j] * y_expert[position of (t, j)], summed in
 * ascending slot order for reproducibility (expert-sorted order would differ in the last bits).
 *
 * The inverse permutation is built by its own pass: `sorted` permutes the live slots, so
 * `inv[sorted[i]] = i` is one scattered store per row. An unclaimed (dropped) slot keeps -1 and
 * contributes nothing. */
__global__ void gather_inv_kernel(int32_t* __restrict__ inv, uint32_t T) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T) inv[i] = -1;
}

__global__ void gather_scatter_kernel(const int32_t* __restrict__ sorted,
                                      int32_t* __restrict__ inv, uint32_t T) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T) return;
    const int32_t f = sorted[i];
    if (f >= 0 && f < (int32_t)T) inv[f] = (int32_t)i;
}

/* ---- prefill gather: a wave per token, eight columns per lane ---------------------------------
 *
 * At prefill this op reads `ye` (105 MB per layer at a 2048 chunk of Qwen3.8-Flash-Next on four
 * ranks), and gather_kernel moves two bytes per lane with slot loads serialised on the slot table
 * (about two thirds of cold-stream rate).
 *
 * Here a wave owns one token and 256 columns, eight per lane: the slot table is read first (scalar
 * loads), then every slot's sixteen bytes are issued before any is used. Arithmetic is
 * gather_kernel's (ascending slot order, dropped slots skipped, f32 `acc += w * x`, the shared
 * arm's two narrows), so the bits match.
 */
constexpr uint32_t kGatherWideSlots = 16;      /* top_k at most this: the slots live in registers */
constexpr uint32_t kGatherWideCols  = 8;       /* columns a lane, one sixteen-byte load */
static_assert(kGatherWideSlots <= kWave, "slot j is read off lane j");

template <uint32_t NW>
__global__ __launch_bounds__(NW * kWave) void gather_wide_kernel(
    const uint16_t* __restrict__ ye, const uint16_t* __restrict__ ew,
    const int32_t* __restrict__ inv, uint16_t* __restrict__ y,
    uint32_t M, uint32_t n, uint32_t top_k, uint32_t ye_ld, uint32_t ew_ld, uint32_t y_ld,
    const uint16_t* __restrict__ sh, const uint16_t* __restrict__ sg,
    uint32_t sh_ld, uint32_t sg_ld, int sig) {
    const uint32_t lane = threadIdx.x % kWave;
    const uint32_t wave = (uint32_t)__builtin_amdgcn_readfirstlane((int)(threadIdx.x / kWave));
    const uint32_t t = blockIdx.x * NW + wave;
    const uint32_t c = (blockIdx.y * kWave + lane) * kGatherWideCols;
    /* No barrier below, so a wave past the last token simply leaves. */
    if (t >= M) return;

    /* The token's slots and weights in one load: lane j fetches slot j (lanes past top_k re-fetch
     * slot 0, unused) and every lane reads them off the lanes into scalar registers. Every lane
     * takes part (a lane past the last column leaves only afterwards), since values are read off
     * any lane. */
    const uint32_t jl = lane < top_k ? lane : 0u;
    const int32_t  il = inv[(size_t)t * top_k + jl];
    const int32_t  wl = (int32_t)ew[(size_t)t * ew_ld + jl];
    int32_t si[kGatherWideSlots];
    float   sw[kGatherWideSlots];
#pragma unroll
    for (uint32_t j = 0; j < kGatherWideSlots; ++j) {
        si[j] = __builtin_amdgcn_readlane(il, (int)j);
        sw[j] = bf16_to_f32((uint16_t)__builtin_amdgcn_readlane(wl, (int)j));
    }
    if (c >= n) return;
    /* No break or continue in these loops: an early exit keeps the slot loop rolled and `yv` goes
     * to scratch. Uniform branches over a constant trip count keep every slot in a register. A slot
     * is live exactly where gather_kernel's loop would use it. */
    bool  lv[kGatherWideSlots];
    uint4 yv[kGatherWideSlots];
#pragma unroll
    for (uint32_t j = 0; j < kGatherWideSlots; ++j) {
        lv[j] = j < top_k && si[j] >= 0;
        yv[j] = make_uint4(0, 0, 0, 0);
        if (lv[j]) yv[j] = *(const uint4*)(ye + (size_t)si[j] * ye_ld + c);
    }
    /* The shared arm after the slots, so its loads do not delay theirs. */
    float gate = 0.0f;
    uint4 shv  = make_uint4(0, 0, 0, 0);
    if (sh) {
        const float v0 = bf16_to_f32(sg[(size_t)t * sg_ld]);
        gate = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
        shv  = *(const uint4*)(sh + (size_t)t * sh_ld + c);
    }

    float acc[kGatherWideCols];
#pragma unroll
    for (uint32_t q = 0; q < kGatherWideCols; ++q) acc[q] = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kGatherWideSlots; ++j) {
        if (lv[j]) {
            const float    wj   = sw[j];
            const uint32_t d[4] = { yv[j].x, yv[j].y, yv[j].z, yv[j].w };
#pragma unroll
            for (uint32_t q = 0; q < 4; ++q) {
                acc[2 * q]     += wj * bf16_to_f32((uint16_t)(d[q] & 0xFFFFu));
                acc[2 * q + 1] += wj * bf16_to_f32((uint16_t)(d[q] >> 16));
            }
        }
    }

    const uint32_t s[4] = { shv.x, shv.y, shv.z, shv.w };
    uint32_t o[4];
#pragma unroll
    for (uint32_t q = 0; q < 4; ++q) {
        uint16_t lo = f32_to_bf16(acc[2 * q]), hi = f32_to_bf16(acc[2 * q + 1]);
        if (sh) {
            /* The shared arm, folded with gather_kernel's two narrows. */
            const uint16_t plo = f32_to_bf16(bf16_to_f32((uint16_t)(s[q] & 0xFFFFu)) * gate);
            const uint16_t phi = f32_to_bf16(bf16_to_f32((uint16_t)(s[q] >> 16)) * gate);
            lo = f32_to_bf16(bf16_to_f32(plo) + bf16_to_f32(lo));
            hi = f32_to_bf16(bf16_to_f32(phi) + bf16_to_f32(hi));
        }
        o[q] = (uint32_t)lo | ((uint32_t)hi << 16);
    }
    *(uint4*)(y + (size_t)t * y_ld + c) = make_uint4(o[0], o[1], o[2], o[3]);
}

/* ================================================================== scale_rows
 *
 * y[m, i] = x[m, i] * act(s[m]). One workgroup per row, grid-strided along it, with the scalar read
 * once. Used only for the shared expert's gate; docs/OPS.md explains why `mul` cannot serve it. */
/* With `ad`, the fused gate-and-add form `y = bf16(x * act(s)) + ad`. The middle narrow matches the
 * unfused pair, which stores the product as bf16 before adding; an f32 product would change the
 * last bit of every channel. */
template <uint32_t NTHR>
__global__ __launch_bounds__(NTHR) void scale_rows_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ sc,
    const uint16_t* __restrict__ ad, uint16_t* __restrict__ y,
    uint32_t M, uint32_t n, uint32_t x_ld, uint32_t s_ld, uint32_t a_ld, uint32_t y_ld, int sig) {
    const uint32_t m = blockIdx.x;
    if (m >= M) return;
    const float v0 = bf16_to_f32(sc[(size_t)m * s_ld]);
    const float v  = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    for (uint32_t i = threadIdx.x; i < n; i += NTHR) {
        const uint16_t p = f32_to_bf16(bf16_to_f32(x[(size_t)m * x_ld + i]) * v);
        y[(size_t)m * y_ld + i] =
            ad ? f32_to_bf16(bf16_to_f32(p) + bf16_to_f32(ad[(size_t)m * a_ld + i])) : p;
    }
}

}  /* namespace */

/* The gather's form, pinnable for tests: 0 the launcher's rule, 1 gather_kernel (a column per
 * thread), 2 gather_wide_kernel (a wave per token). The rule takes 2 past the self-scan bound where
 * the shape allows (r4d_rad_moe_gather); an impossible pin is refused. Both are bit-identical
 * (r4d_selftest). */
static int g_moe_gather_pin = 0;
extern "C" void r4d_moe_gather_pin_form(int form) { g_moe_gather_pin = form; }

/* ============================================================== the plugin ABI entry points */

extern "C" int r4d_rad_router_topk(const RadArgs* a, RadStream s) {
    const RadTensor* lg  = r4d_opt(a, 0);
    const RadTensor* ids = r4d_opt(a, 1);
    const RadTensor* w   = r4d_opt(a, 2);
    if (!lg || !ids || !w || !lg->data || !ids->data || !w->data) return R4D_NO(RAD_E_INVAL);
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k    = r4d_geti_or(a, "top_k", 0);
    const long long norm     = r4d_geti_or(a, "norm", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || top_k > n_expert)
        return R4D_NO(RAD_E_SHAPE);
    /* The key packs `~e` into sixteen bits, which the kMaxExpert check above guarantees. top_k is
     * bounded by the wave, since round r parks its winner on lane r. */
    if (top_k > (long long)kWave) return R4D_NO(RAD_E_SHAPE);
    if (lg->dtype != RAD_BF16 || w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (ids->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    long long M = 0;
    const long long lg_ld = r4d_rows_pitch(lg, n_expert, &M);
    if (lg_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(ids) || !r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(ids) < M * top_k || r4d_numel(w) < M * top_k) return R4D_NO(RAD_E_SHAPE);

    constexpr uint32_t NW = 4;
    const dim3 grid((unsigned)((M + NW - 1) / NW)), blk(NW * kWave);
    /* The narrowest instantiation covering n_expert; the wider costs eight registers per lane. */
    if (n_expert <= (long long)(8 * kWave))
        hipLaunchKernelGGL((router_topk_kernel<NW, 8>), grid, blk, 0, (hipStream_t)s,
                           (const uint16_t*)lg->data, (int32_t*)ids->data, (uint16_t*)w->data,
                           (uint32_t)M, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)lg_ld,
                           (int)norm);
    else
        hipLaunchKernelGGL((router_topk_kernel<NW, 16>), grid, blk, 0, (hipStream_t)s,
                           (const uint16_t*)lg->data, (int32_t*)ids->data, (uint16_t*)w->data,
                           (uint32_t)M, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)lg_ld,
                           (int)norm);
    R4D_MOE_CHK();
    return RAD_OK;
}

/* The pair in one launch. Operands are the two ops' own, selection first: logits -> expert_ids,
 * expert_w, sorted_tok, expert_offset, expert_count. The checks are the two shims' checks.
 *
 * It refuses, rather than falls back, when the sort would not fit one workgroup, so the op's cost
 * does not hinge on a hidden bound; the architecture checks the same before declaring it
 * (`moe_fuse_rows`). */
extern "C" int r4d_rad_router_topk_scatter(const RadArgs* a, RadStream stream) {
    int emit_rc = RAD_OK;
    const RadTensor* lg     = r4d_opt(a, 0);
    const RadTensor* ids    = r4d_opt(a, 1);
    const RadTensor* w      = r4d_opt(a, 2);
    const RadTensor* sorted = r4d_opt(a, 3);
    const RadTensor* off    = r4d_opt(a, 4);
    const RadTensor* cnt    = r4d_opt(a, 5);
    if (!lg || !ids || !w || !sorted || !off || !cnt) return R4D_NO(RAD_E_INVAL);
    if (!lg->data || !ids->data || !w->data || !sorted->data || !off->data || !cnt->data)
        return R4D_NO(RAD_E_INVAL);

    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    /* Absent means no expert parallelism: the slice is the whole router. */
    const long long n_local  = r4d_geti_or(a, "n_local", n_expert);
    const long long top_k    = r4d_geti_or(a, "top_k", 0);
    const long long norm     = r4d_geti_or(a, "norm", 0);
    const long long base     = r4d_geti_or(a, "expert_base", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || top_k > n_expert)
        return R4D_NO(RAD_E_SHAPE);
    if (top_k > (long long)kWave) return R4D_NO(RAD_E_SHAPE);
    if (n_local <= 0 || n_local > n_expert) return R4D_NO(RAD_E_SHAPE);
    if (base < 0 || base + n_local > n_expert) return R4D_NO(RAD_E_SHAPE);
    Protect prot;
    const int prc = protect_of(a, n_local, &prot);
    if (prc != RAD_OK) return prc;
    if (lg->dtype != RAD_BF16 || w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (ids->dtype != RAD_I32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    long long M = 0;
    const long long lg_ld = r4d_rows_pitch(lg, n_expert, &M);
    if (lg_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(ids) || !r4d_contig(w) || !r4d_contig(sorted) || !r4d_contig(off) ||
        !r4d_contig(cnt)) return R4D_NO(RAD_E_STRIDE);

    const long long T = M * top_k;
    if (T > (long long)kScatterSmall) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(ids) < T || r4d_numel(w) < T) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(sorted) < T) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(off) < n_local + 1 || r4d_numel(cnt) < n_local) return R4D_NO(RAD_E_SHAPE);

#define R4D_RTS_L(OWN, NWV)                                                                     \
    emit_rc = R4D_EMIT(stream, (router_topk_scatter_kernel<OWN, NWV>), dim3(1),                 \
                       dim3((NWV) * kWave), 0,                                                  \
                       (const uint16_t*)lg->data, (int32_t*)ids->data,                          \
                       (uint16_t*)w->data, (int32_t*)sorted->data, (int32_t*)off->data,         \
                       (int32_t*)cnt->data, (uint32_t)M, (uint32_t)n_expert,                    \
                       (uint32_t)n_local, (uint32_t)top_k,                                      \
                       (uint32_t)lg_ld, (int)norm, (int32_t)base, prot)
    /* The narrowest instantiation, as in the standalone selection.
     *
     * One wave per row, rounded to a power of two between 8 and 32 waves: a wave selects its rows
     * serially, so a 32-row verify step at eight waves runs four selections back to back (8.6 us
     * against 5.4 at 32 waves), while extra waves cost ~0.4 us at four rows. Above 32 rows the
     * waves wrap. */
    const int nw = M <= 8 ? 8 : M <= 16 ? 16 : 32;
    if (n_expert <= (long long)(8 * kWave)) {
        if      (nw == 8)  R4D_RTS_L(8, 8);
        else if (nw == 16) R4D_RTS_L(8, 16);
        else               R4D_RTS_L(8, 32);
    } else {
        if      (nw == 8)  R4D_RTS_L(16, 8);
        else if (nw == 16) R4D_RTS_L(16, 16);
        else               R4D_RTS_L(16, 32);
    }
#undef R4D_RTS_L
    if (emit_rc != RAD_OK) return emit_rc;
    R4D_MOE_CHK();
    return RAD_OK;
}

/* Scratch for the general scatter: the per-tile histogram plane (none for the small form). Sized
 * for the shape asked about, since the resolved band sizes the arena. */
static int scatter_dims(const RadArgs* a, long long* T, long long* n_expert) {
    const RadTensor* ids = r4d_opt(a, 0);
    *n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 1);
    if (*n_expert <= 0 || *n_expert > (long long)kMaxExpert || top_k <= 0)
        return R4D_NO(RAD_E_SHAPE);
    *T = ids && ids->data ? r4d_numel(ids) : r4d_geti_or(a, "M", 0) * top_k;
    if (*T <= 0) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}

extern "C" int64_t r4d_rad_moe_scatter_scratch(const RadArgs* a) {
    long long T = 0, n_expert = 0;
    const int rc = scatter_dims(a, &T, &n_expert);
    if (rc != RAD_OK) return rc;
    if (T <= (long long)kScatterSmall) return 0;
    const long long n_tile = (T + kScatterTile - 1) / kScatterTile;
    return n_tile * n_expert * (long long)sizeof(int32_t);
}

/* moe_scatter, and with `inv` (moe_scatter_inv's fifth operand) the inverse beside it. */
static int moe_scatter_body(const RadArgs* a, RadStream s, const RadTensor* inv) {
    const RadTensor* ids    = r4d_opt(a, 0);
    const RadTensor* sorted = r4d_opt(a, 1);
    const RadTensor* off    = r4d_opt(a, 2);
    const RadTensor* cnt    = r4d_opt(a, 3);
    if (!ids || !sorted || !off || !cnt) return R4D_NO(RAD_E_INVAL);
    if (!ids->data || !sorted->data || !off->data || !cnt->data) return R4D_NO(RAD_E_INVAL);
    if (ids->dtype != RAD_I32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(ids) || !r4d_contig(sorted) || !r4d_contig(off) || !r4d_contig(cnt))
        return R4D_NO(RAD_E_STRIDE);

    long long T = 0, n_expert = 0;
    const int rc = scatter_dims(a, &T, &n_expert);
    if (rc != RAD_OK) return rc;
    if (r4d_numel(sorted) < T) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(off) < n_expert + 1 || r4d_numel(cnt) < n_expert) return R4D_NO(RAD_E_SHAPE);
    /* This rank's first expert, subtracted from every routed id so offsets, counts and the weight
     * table are indexed locally. 0 without expert parallelism. */
    const long long base = r4d_geti_or(a, "expert_base", 0);
    if (base < 0) return R4D_NO(RAD_E_SHAPE);
    Protect prot;
    const int prc = protect_of(a, n_expert, &prot);
    if (prc != RAD_OK) return prc;

    if (inv && (inv->dtype != RAD_I32 || !r4d_contig(inv) || r4d_numel(inv) < T))
        return R4D_NO(RAD_E_SHAPE);

    const int32_t* idp = (const int32_t*)ids->data;
    int32_t* sp = (int32_t*)sorted->data;
    int32_t* op = (int32_t*)off->data;
    int32_t* cp = (int32_t*)cnt->data;
    int32_t* ip = inv ? (int32_t*)inv->data : nullptr;

    if (T <= (long long)kScatterSmall) {
        if (ip)
            hipLaunchKernelGGL(scatter_small_kernel<true>, dim3(1), dim3(kScatWaves * kWave), 0,
                               (hipStream_t)s, idp, sp, op, cp, (uint32_t)T, (uint32_t)n_expert,
                               (int32_t)base, prot, ip);
        else
            hipLaunchKernelGGL(scatter_small_kernel<false>, dim3(1), dim3(kScatWaves * kWave), 0,
                               (hipStream_t)s, idp, sp, op, cp, (uint32_t)T, (uint32_t)n_expert,
                               (int32_t)base, prot, ip);
        R4D_MOE_CHK();
        return RAD_OK;
    }

    const long long n_tile = (T + kScatterTile - 1) / kScatterTile;
    const int64_t need = n_tile * n_expert * (long long)sizeof(int32_t);
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);
    int32_t* bcnt = (int32_t*)a->scratch;

    hipLaunchKernelGGL(scatter_hist_kernel, dim3((unsigned)n_tile), dim3(kScatterTile),
                       (unsigned)(n_expert * sizeof(int32_t)), (hipStream_t)s,
                       idp, bcnt, (uint32_t)T, (uint32_t)n_expert, (int32_t)base, prot);
    R4D_MOE_CHK();
    hipLaunchKernelGGL(scatter_scan_kernel, dim3(1), dim3(kMaxExpert), 0, (hipStream_t)s,
                       bcnt, op, cp, (uint32_t)n_tile, (uint32_t)n_expert);
    R4D_MOE_CHK();
    if (ip)
        hipLaunchKernelGGL(scatter_place_kernel<true>, dim3((unsigned)n_tile), dim3(kScatterTile),
                           0, (hipStream_t)s, idp, bcnt, op, sp, (uint32_t)T, (uint32_t)n_expert,
                           (int32_t)base, prot, ip);
    else
        hipLaunchKernelGGL(scatter_place_kernel<false>, dim3((unsigned)n_tile), dim3(kScatterTile),
                           0, (hipStream_t)s, idp, bcnt, op, sp, (uint32_t)T, (uint32_t)n_expert,
                           (int32_t)base, prot, ip);
    R4D_MOE_CHK();
    return RAD_OK;
}

extern "C" int r4d_rad_moe_scatter(const RadArgs* a, RadStream s) {
    return moe_scatter_body(a, s, nullptr);
}

/* moe_scatter that also writes, for every slot, its row in `sorted_tok` (-1 when dropped): the
 * table moe_gather_inv reads instead of rebuilding it from `sorted_tok`. Same sort, same scratch. */
extern "C" int r4d_rad_moe_scatter_inv(const RadArgs* a, RadStream s) {
    const RadTensor* inv = r4d_opt(a, 4);
    if (!inv) return R4D_NO(RAD_E_INVAL);
    return moe_scatter_body(a, s, inv);
}

/* The weight-table operand from the core's tensor. A zero leading stride marks a pointer array
 * (abi/rad_runtime.h, RAD_WTAB); otherwise it is a stacked plane with the stride in elements of the
 * weight's dtype (bytes for E4M3). */
static int wtab_of(const RadTensor* t, uint32_t dtype, int bits, long long n_expert,
                   long long rows, long long cols, const void* const** tab, size_t* stride0,
                   long long* ld) {
    if (!t || !t->data || t->rank != 3) return R4D_NO(RAD_E_SHAPE);
    if (t->dtype != dtype) return R4D_NO(RAD_E_DTYPE);
    if (t->shape[0] < n_expert || t->shape[1] < rows || t->shape[2] < cols)
        return R4D_NO(RAD_E_SHAPE);
    if (t->stride[2] != 1) return R4D_NO(RAD_E_STRIDE);
    *tab     = (const void* const*)t->data;
    /* The element width is passed in: for a plugin-private dtype such as the 4-bit plane,
     * rad_dtype_bits() returns 0 and a stacked stride would collapse to the zero that marks a
     * pointer array. Bits are multiplied before the divide so a 4-bit plane keeps its half-byte. */
    *stride0 = (size_t)((t->stride[0] * (int64_t)bits) / 8);
    *ld      = t->stride[1];
    return RAD_OK;
}

/* ---- AL: one run per block, so a streamed expert's weight crosses the link once -----------------
 *
 * The uniform row grid serves a run straddling a tile boundary with two blocks, staging its weight
 * twice. For a resident expert the second staging is an L2/MALL hit, but for a pooled one it is
 * another pass over PCIe (~26 GB/s against VRAM's 576). At a 2048 chunk the grid has 414
 * incidences over 254 experts, so a third of the link traffic would be re-reads.
 *
 * So a block map gives one run per block: entry i names (expert, first row) and the block serves
 * rows [first, min(first + FM, off[e+1])); an expert is staged by two blocks only if its run is
 * longer than FM. This costs no extra activation traffic (stage8_load skips rows outside the run),
 * no extra arithmetic (the uniform grid already runs a full tile per covered run), and no
 * determinism (each output row still comes from one block and one expert in the same K order).
 *
 * The grid is an upper bound, `n_expert + T/FM` (each expert one block plus one per FM rows), since
 * the true count needs the offsets on the host; the map's first slot holds the real count and
 * extra blocks exit at the head.
 *
 * Rows past `off[n_expert]` (dropped slots) must still be zeroed: the map carries them as entries
 * with expert `kMapTail`, whose block zeroes and returns. */

/* The prefill tile's rows; see the prefill-tile note at the kernel. */
constexpr long long kPrefillRows = 32;
/* ...and 128 columns over eight waves where a column tile cannot straddle two stacked weight parts.
 * Each wave's share is unchanged (one 16-row by two 16-column tiles), so every element's arithmetic
 * is the same; the wider block stages the activation once for twice the columns, with half the
 * blocks (~5% faster per call at 512-4096 tokens). 128 columns over only four waves is no better. */
constexpr long long kPrefillCols = 128;

/* Where the (tile, run) decode grid ends and the run-aligned prefill grid begins: an average of 1.5
 * routed rows per expert. At file scope because the scratch hook and the launcher must agree on
 * the band.
 *
 * The crossover is a run length, not a token count: 16-row tiles win for runs of a row or two,
 * 32-row tiles for several rows. Measured at 512 experts (gate_up + down, us):
 *
 *      T (rows)         640    800    1000   1280   2560   5120
 *      decode form      804    900    1006   1113   1541   2224
 *      prefill form     826    900     981   1053   1247   1413
 *
 * So the band is T <= 1.5 * n_expert (768 rows here). Decode is at most
 * `max_num_seqs * (1 + n_spec) * top_k` (320 rows at eight sequences, depth 3, top 10). */
static inline bool r4d_moe_decode_band(long long T, long long n_expert) {
    return 2 * T <= 3 * n_expert;
}
/* Row limit, per weight format, for staging two 128-wide K blocks per pass (FK 256) in the decode
 * form of the token-operand projection (gate_up). Only formats that already fold scale groups
 * (W4 2-4) with every class's K a multiple of 256 qualify. Bit-identical either way; a constant, so
 * recorded and live passes take the same kernel.
 *
 * Measured (gate_up, us, one block / two): the 4-bit formats gain at 10-20 rows but not at 40-160
 * (e.g. w4nl64a8h 17.9/15.6 at 10 rows, 36.4/37.3 at 40); the 5-bit format is 2-3x faster with two
 * blocks from 40 rows on (122.3/43.1 at 40) and faster at every row count. In a whole decode step
 * the 4-bit formats still take two blocks up to 80 rows (two sequences at depth 3): on Flash-Next
 * (w4nl64a8h) a one-stream step at depth 3 is 1.1% shorter at TP4 and no slower at TP2. */
template <int W4>
constexpr long long kMoeGuFk256Rows = W4 == 4 ? (1ll << 40) : (W4 == 2 || W4 == 3) ? 80 : 0;
/* The decode form's staged K, pinnable for tests: 0 kMoeGuFk256Rows's rule, 1 one 128-wide block
 * per pass, 2 two wherever allowed. r4d_selftest compares both byte for byte. */
static int g_moe_fk_pin = 0;
template <int W4>
static inline bool r4d_moe_gu_fk256(long long T, bool a_sorted) {
    if (a_sorted || g_moe_fk_pin == 1) return false;
    if (g_moe_fk_pin == 2) return true;
    return T <= kMoeGuFk256Rows<W4>;
}
/* The sorted-operand projection's decode kernel and its column tiles per block. The decode form of
 * down (sorted operand, one part, at most kMaxExpert experts) is moe_down_dec_kernel; any other
 * shape takes the (tile, run) form. G follows the routed-row count: one below kMoeDownDecG2Rows,
 * two below kMoeDownDecG4Rows, four above, keeping a few thousand working blocks from one sequence
 * to sixteen. Bit-identical at every G; constants, so recorded and live passes agree.
 *
 * At TP4 (MTP depth 3) this saves 0.9%, 3.1% and 7.8% of a decode step at 4, 8 and 16 sequences.
 * At TP2 differences are within noise (a rank holding twice each expert's width already keeps the
 * (tile, run) grid busy), so the TP4 thresholds stand. */
constexpr long long kMoeDownDecG2Rows = 200;
constexpr long long kMoeDownDecG4Rows = 480;
/* G's bound: the kernel keeps its column tiles in one 32-bit mask. */
constexpr long long kMoeDownDecMaxG = 16;
/* The sorted operand's decode kernel, pinnable for tests: 0 the rule above, -1 the (tile, run)
 * form, n > 0 moe_down_dec_kernel with n column tiles per block (capped at kMoeDownDecMaxG). A pin
 * on an unsupported shape is refused. */
static int g_moe_down_dec_pin = 0;
static inline long long r4d_moe_down_dec_g(long long T) {
    if (g_moe_down_dec_pin < 0) return 0;
    if (g_moe_down_dec_pin > 0) return g_moe_down_dec_pin;
    return T >= kMoeDownDecG4Rows ? 4 : T >= kMoeDownDecG2Rows ? 2 : 1;
}
/* The map's upper bound, defined once so the scratch hook and the launcher agree exactly (or the
 * kernel writes past its arena). Each expert contributes one block plus one per FM rows beyond the
 * first; the dropped-slot tail at most T/FM plus one. */
static inline long long r4d_blockmap_cap(long long T, long long n_expert, long long FM) {
    return n_expert + (T + FM - 1) / FM + 1;
}
/* One count slot, then two uint32 per entry. */
static inline long long r4d_blockmap_bytes(long long cap) {
    return (1 + 2 * cap) * (long long)sizeof(uint32_t);
}

/* One workgroup: the job is a scan over at most kMaxExpert counters, cheaper than a second launch.
 * NT threads, thread tid owning experts tid, tid + NT, ...; offsets are read once, in parallel.
 *
 * The prefix is parallel (a DPP scan per wave and one exchange of wave totals, 256 counts per tile;
 * three tiles at 512 experts), not 513 dependent LDS round trips on one thread ahead of a GEMM that
 * waits for the map. Integer sums are order-independent. */
template <class CH>
__global__ __launch_bounds__(256) void moe_blockmap_kernel(
    const int32_t* __restrict__ off, uint32_t n_expert, uint32_t T, uint32_t FM, uint32_t cap,
    uint32_t* __restrict__ map) {
    constexpr uint32_t NT = 256, NW = NT / kWave, kTiles = (kMaxExpert + NT) / NT;
    if (CH::on) r4d_chain_wait();
    __shared__ uint32_t s_cnt[kMaxExpert + 1];
    __shared__ int32_t  s_wt[kTiles][NW];
    const uint32_t tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;

    /* Every offset this thread needs in one round of loads: both ends of each expert it owns
     * (e = tid + i * NT), index clamped so every load is unconditional, all issued before the first
     * wait. Counts and map rows come from these registers; offsets never pass through LDS. */
    int32_t lo[kTiles], hi[kTiles];
#pragma unroll
    for (uint32_t i = 0; i < kTiles; ++i) {
        const uint32_t e = tid + i * NT;
        lo[i] = off[e < n_expert ? e : n_expert];
        hi[i] = off[e + 1 < n_expert ? e + 1 : n_expert];
    }
    const uint32_t live = (uint32_t)off[n_expert];
    /* Blocks this expert needs; a zero-row run needs none. */
#pragma unroll
    for (uint32_t i = 0; i < kTiles; ++i) {
        const uint32_t e = tid + i * NT;
        if (e < n_expert) s_cnt[e] = ((uint32_t)(hi[i] - lo[i]) + FM - 1) / FM;
    }
    /* The tail of dropped slots, carried as one more "expert" so the scan places it too. */
    if (tid == 0) s_cnt[n_expert] = T > live ? (T - live + FM - 1) / FM : 0u;
    __syncthreads();

    /* Exclusive scan over the n_expert + 1 counts, NT at a time. Each tile has its own exchange
     * slots, so a tile's writes never meet the previous tile's reads. */
    uint32_t carry = 0;
    for (uint32_t b = 0, tl = 0; b <= n_expert; b += NT, ++tl) {
        const uint32_t e = b + tid;
        const int32_t  v = e <= n_expert ? (int32_t)s_cnt[e] : 0;
        const int32_t  x = wave_incl_scan_i32(v, lane);
        if (lane == kWave - 1) s_wt[tl][wave] = x;
        lds_barrier();
        int32_t below = 0, tot = 0;
#pragma unroll
        for (uint32_t k = 0; k < NW; ++k) {
            const int32_t c = s_wt[tl][k];
            below += (k < wave) ? c : 0;
            tot   += c;
        }
        if (e <= n_expert) s_cnt[e] = carry + (uint32_t)(below + x - v);
        carry += (uint32_t)tot;
    }
    if (tid == 0) map[0] = carry < cap ? carry : cap;    /* the count, clamped to the grid */
    __syncthreads();

#pragma unroll
    for (uint32_t i = 0; i < kTiles; ++i) {
        const uint32_t e = tid + i * NT;
        if (e >= n_expert) continue;
        const uint32_t run = (uint32_t)(hi[i] - lo[i]);
        uint32_t o = s_cnt[e];
        for (uint32_t r = 0; r < run; r += FM, ++o) {
            if (o >= cap) break;
            map[1 + 2 * o + 0] = e;
            map[1 + 2 * o + 1] = (uint32_t)lo[i] + r;
        }
    }
    if (tid == 0) {
        uint32_t o = s_cnt[n_expert];
        for (uint32_t r = live; r < T; r += FM, ++o) {
            if (o >= cap) break;
            map[1 + 2 * o + 0] = kMapTail;
            map[1 + 2 * o + 1] = r;
        }
    }
    if (CH::on) r4d_chain_done();
}

/* Launch form: 0 the rule; for tests, 1 pins the run-aligned prefill form, 2 the (tile, run)
 * decode form, 3 the uniform prefill grid, whatever the band. r4d_selftest --perfmoe requires all
 * three to match byte for byte. */
static int g_moe_prefill_form = 0;
/* The aligned form's column tile, pinnable for tests: 0 the launcher's rule, 1 never the down
 * projection's 256-column tile (use 128 or 64), 2 the 256-column tile (refused where the shape
 * cannot take it or the launch is not aligned). r4d_selftest compares them byte for byte. */
static int g_moe_down_tile_pin = 0;

/* One body for the fp8 and w4 launchers, which differ only in the weight dtype and its two
 * pitches, so their shape checks cannot drift apart.
 *
 * `ncls` is the class count: 2 for `moe_gemm_q` (odd class = operands 6 and 7 at N_odd/K_odd), 4
 * for `moe_gemm_q_mod4` (classes 1..3 = operands 6/7, 8/9, 10/11 at N_1/K_1 .. N_3/K_3, `y` at 12). */
/* I8: the int8 activation's rows (dtypes i8a8 and i8a8h at W4 0, the four-bit forms' `a8i8`
 * names), the kernels' I8. */
template <int W4, bool I8 = false>
static int moe_gemm_launch(const RadArgs* a, RadStream stream, int ncls = 2) {
    int emit_rc = RAD_OK;
    if (ncls != 2 && ncls != 4) return R4D_NO(RAD_E_INVAL);
    const RadTensor* Aq     = r4d_opt(a, 0);
    const RadTensor* As     = r4d_opt(a, 1);
    const RadTensor* sorted = r4d_opt(a, 4);
    const RadTensor* off    = r4d_opt(a, 5);
    const RadTensor* Wc[4]  = { r4d_opt(a, 2), r4d_opt(a, 6),
                                ncls == 4 ? r4d_opt(a, 8) : nullptr,
                                ncls == 4 ? r4d_opt(a, 10) : nullptr };
    const RadTensor* Wsc[4] = { r4d_opt(a, 3), r4d_opt(a, 7),
                                ncls == 4 ? r4d_opt(a, 9) : nullptr,
                                ncls == 4 ? r4d_opt(a, 11) : nullptr };
    const RadTensor* Y      = r4d_opt(a, ncls == 4 ? 12 : 8);
    if (!Aq || !As || !sorted || !off || !Y) return R4D_NO(RAD_E_INVAL);
    for (int q = 0; q < ncls; ++q)
        if (!Wc[q] || !Wsc[q]) return R4D_NO(RAD_E_INVAL);
    if (!Aq->data || !As->data || !sorted->data || !off->data || !Y->data)
        return R4D_NO(RAD_E_INVAL);

    /* Each class's weight geometry (defaulting to class 0's); the output is as wide as the widest
     * class and the activation as long as the longest K. */
    static const char* const kNk[4][2] = { { "N", "K" }, { "N_1", "K_1" }, { "N_2", "K_2" },
                                           { "N_3", "K_3" } };
    long long Nc[4] = { 0, 0, 0, 0 }, Kc[4] = { 0, 0, 0, 0 };
    Nc[0] = r4d_geti_or(a, "N", 0);
    Kc[0] = r4d_geti_or(a, "K", 0);
    for (int q = 1; q < ncls; ++q) {
        Nc[q] = r4d_geti_or(a, ncls == 4 ? kNk[q][0] : "N_odd", Nc[0]);
        Kc[q] = r4d_geti_or(a, ncls == 4 ? kNk[q][1] : "K_odd", Kc[0]);
    }
    const long long parts = r4d_geti_or(a, "parts", 1);
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0);
    const long long group = r4d_geti_or(a, "group", kBlock);
    if (parts <= 0 || n_expert <= 0 || top_k <= 0) return R4D_NO(RAD_E_SHAPE);
    if (group != (long long)kBlock) return R4D_NO(RAD_E_SHAPE);
    /* `a_group`: the K one activation scale covers, 128 or, for an int8 activation quantised as
     * IQ8R's are, 256 (two of the kernel's K blocks under one scale). */
    const long long a_group = r4d_geti_or(a, "a_group", kBlock);
    if (a_group != (long long)kBlock && !(I8 && a_group == 2 * (long long)kBlock))
        return R4D_NO(RAD_E_UNSUPPORTED);
    long long N = 0, K = 0;
    for (int q = 0; q < ncls; ++q) {
        if (Nc[q] <= 0 || Kc[q] <= 0) return R4D_NO(RAD_E_SHAPE);
        /* A part of every class is whole 64-column tiles, so a tile is all weight or all padding. */
        if (Kc[q] % a_group || Nc[q] % (64 * parts)) return R4D_NO(RAD_E_SHAPE);
        if (Nc[q] > N) N = Nc[q];
        if (Kc[q] > K) K = Kc[q];
    }
    if (Aq->dtype != (I8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_F8E4M3) || As->dtype != RAD_F32 ||
        Y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (sorted->dtype != RAD_I32 || off->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    long long T = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &T);
    if (y_ld < 0 || T <= 0) return R4D_NO(RAD_E_STRIDE);
    long long M = 0;
    const long long a_ld = r4d_rows_pitch(Aq, K, &M);
    if (a_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    long long Ms = 0;
    const long long as_ld = r4d_rows_pitch(As, K / a_group, &Ms);
    if (as_ld < 0 || Ms < M) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(sorted) < T || r4d_numel(off) < n_expert + 1) return R4D_NO(RAD_E_SHAPE);

    /* The activation's row count is checked, not inferred: a short operand means the caller passed
     * the gate_up projection's buffer to the down projection. */
    const char* ord = r4d_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (M < (a_sorted ? T : (T + top_k - 1) / top_k)) return R4D_NO(RAD_E_SHAPE);

    /* The code table, from the dtype the layout hook accepted: w4nla8h's levels, w5nl64a8h's
     * magnitudes, or two's complement for w4a8 and w4a8h. */
    const char* wdt = r4d_gets(a, "dtype");
    const uint32_t* tab = I8 ? kW4IntPermI8
                        : W4 == 4 ? kW5nlMagPerm
                        : (W4 && wdt && (!std::strcmp(wdt, "w4nla8h") ||
                                         !std::strcmp(wdt, "w4nl64a8h") ||
                                         !std::strcmp(wdt, "w4nl32a8h"))) ? kW4nlPerm : kW4IntPerm;

    /* One class's tables. The kernel wants the weight pitch in bytes: half the element count for a
     * 4-bit plane, five eighths for the 5-bit one. `wtab_of` reports the declared element stride,
     * so the scaling happens here. */
    auto cls_of = [&](const RadTensor* w, const RadTensor* ws, long long cnt, long long n,
                      long long k, MoeCls* c) -> int {
        long long w_ld = 0, ws_ld = 0;
        size_t w_s0 = 0, s_s0 = 0;
        int rc = wtab_of(w, W4 == 4 ? (uint32_t)R4D_DT_MOEW5
                            : W4 ? (uint32_t)R4D_DT_MOEW4
                            : I8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_F8E4M3,
                         W4 == 4 ? 5 : W4 ? 4 : 8, cnt, n, k, &c->w, &w_s0, &w_ld);
        if (rc != RAD_OK) return rc;
        if constexpr (W4 >= 2) {
            /* One E4M3 byte per (row, group of 64 or 32 K). The pitch is derived, not read: the
             * scale view arrives at the weight's logical [n, k] (row stride k) while the stored row
             * is k/64 or k/32 bytes (r4d_layout.cpp). A tile's scales are one 16- or 32-bit word,
             * so the kernel's pitch is k/128 words. */
            rc = wtab_of(ws, RAD_F8E4M3, 8, cnt, n, k / (kBlock / (W4 == 3 ? 4 : 2)), &c->s,
                         &s_s0, &ws_ld);
            if (rc != RAD_OK) return rc;
            /* A 5-bit row is its nibbles then its sign bits; the kernel finds the signs K/2 bytes
             * in. */
            if (W4 == 4 ? (w_ld % 8 || k % 32) : (w_ld % 2)) return R4D_NO(RAD_E_STRIDE);
            w_ld = W4 == 4 ? w_ld / 8 * 5 : w_ld / 2;
            ws_ld = k / kBlock;
        } else if constexpr (W4) {
            /* The scale plane is one bf16 per (row, 128 of K), not per 128x128 tile. */
            rc = wtab_of(ws, RAD_BF16, 16, cnt, n, k / kBlock, &c->s, &s_s0, &ws_ld);
            if (rc != RAD_OK) return rc;
            if (w_ld < k || w_ld % 64) return R4D_NO(RAD_E_STRIDE); // whole 64-K fragment tiles
            w_ld /= 2;
        } else if constexpr (I8) {
            /* Int8 codes row-major, one bf16 scale per (row, 128 of K). */
            rc = wtab_of(ws, RAD_BF16, 16, cnt, n, k / kBlock, &c->s, &s_s0, &ws_ld);
            if (rc != RAD_OK) return rc;
        } else {
            rc = wtab_of(ws, RAD_BF16, 16, cnt, (n + kBlock - 1) / kBlock, k / kBlock, &c->s,
                         &s_s0, &ws_ld);
            if (rc != RAD_OK) return rc;
        }
        /* A stacked entry is one expert's plane, far below 4 GiB; a pointer table is 0. */
        if (w_s0 > 0xFFFFFFFFull || s_s0 > 0xFFFFFFFFull) return R4D_NO(RAD_E_STRIDE);
        c->w_s0 = (uint32_t)w_s0;
        c->s_s0 = (uint32_t)s_s0;
        c->n = (uint32_t)n;
        c->k = (uint32_t)k;
        c->w_ld = (uint32_t)w_ld;
        c->s_ld = (uint32_t)ws_ld;
        return RAD_OK;
    };
    /* Class q holds experts e with e % ncls == q at entry e / ncls, ceil((n - q) / ncls) of them.
     * At two classes the last two stay zero and unread. */
    MoeCls cl[4]{};
    for (int q = 0; q < ncls; ++q) {
        const int rc = cls_of(Wc[q], Wsc[q], (n_expert + ncls - 1 - q) / ncls, Nc[q], Kc[q],
                              &cl[q]);
        if (rc != RAD_OK) return rc;
    }
    const MoeCls& ce = cl[0];
    const MoeCls& co = cl[1];
    const MoeCls& c2 = cl[2];
    const MoeCls& c3 = cl[3];
    const uint32_t cshift = ncls == 4 ? 2u : 1u;
    if (parts > 0xFFFFFF) return R4D_NO(RAD_E_SHAPE);
    const uint32_t mode = (a_sorted ? 1u : 0u) | (cshift << 1) |
                          (a_group != (long long)kBlock ? 8u : 0u) | ((uint32_t)parts << 8);
    MoeTab tabv{};
    for (int i = 0; i < 4; ++i) tabv.v[i] = tab[i];

    /* The one-token per-128-scale I4 path needs no LDS tile or run scan. Pins select the original
     * forms, which are also its byte-for-byte regression oracle. Larger row counts retain the
     * existing decode/prefill bands; a small row count alone is not evidence of their crossover. */
    if constexpr (W4 == 1) {
        if (T <= top_k && T <= 32 && g_moe_prefill_form == 0 &&
            g_moe_down_dec_pin == 0 && g_moe_fk_pin == 0) {
            emit_rc = R4D_EMIT(stream, (moe_w4_rows_kernel<I8>),
                               dim3((unsigned)T, (unsigned)(N / 64)), dim3(4 * kWave), 0,
                               (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                               (const int32_t*)sorted->data, (const int32_t*)off->data,
                               (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N,
                               (uint32_t)n_expert, (uint32_t)top_k,
                               (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv);
            if (emit_rc != RAD_OK) return emit_rc;
            R4D_MOE_CHK();
            return RAD_OK;
        }
    }

    /* The band: a decode step takes the (tile, run) form (the kernel's ES note) and runs of several
     * rows take the aligned 32-row form; r4d_moe_decode_band has the crossover. */
    const uint32_t ny = (uint32_t)(N / 64);
    const int form = g_moe_prefill_form;
    /* The decode band's launch has a chained form (r4d_chain.h); the prefill forms have none. */
    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    if ((form == 0 && r4d_moe_decode_band(T, n_expert)) || form == 2) {
        /* The decode grid is (tile, run) wide (the kernel's ES note); decode has one tile of rows,
         * which alone would give only N/64 workgroups. */
        const unsigned tiles = (unsigned)((T + 15) / 16);
        /* The sorted operand's decode kernel (moe_down_dec_kernel, kMoeDownDecG2Rows): the grid is
         * the run tiles' upper bound (one per run plus one per 16 rows) plus one tail block, by
         * column groups. */
        const bool dn_dec = a_sorted && parts == 1 && n_expert <= (long long)kMaxExpert;
        const long long g2 = r4d_moe_down_dec_g(T);
        if (!dn_dec && g_moe_down_dec_pin > 0) return R4D_NO(RAD_E_UNSUPPORTED);
        if (dn_dec && g2 > 0) {
            const unsigned slots  = (unsigned)(std::min(T, n_expert) + (T + 15) / 16);
            const unsigned groups = (unsigned)((N + 64 * g2 - 1) / (64 * g2));
#define R4D_MOE_DD(CH_)                                                                          \
            emit_rc = R4D_EMIT(stream, (moe_down_dec_kernel<W4, I8, CH_>),                       \
                               dim3(slots + 1u, groups), dim3(4 * kWave), 0,                     \
                               (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3, \
                               (const int32_t*)off->data, (uint16_t*)Y->data,                    \
                               (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)g2,       \
                               (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv)
            if (chain) R4D_MOE_DD(r4dchain); else R4D_MOE_DD(R4dUnchained);
#undef R4D_MOE_DD
            if (emit_rc != RAD_OK) return emit_rc;
            R4D_MOE_CHK();
            return RAD_OK;
        }
        /* Instantiation: UW always; RS up to kMaxExpert experts, the chunked search above; two
         * 128-wide K blocks per pass where kMoeGuFk256Rows admits the row count, every class's K is
         * whole 256-wide passes and the format folds scale groups. All are bit-identical; a
         * recorded pass sees a different choice as a new node. */
        const bool rs = n_expert <= (long long)kMaxExpert;
        auto pick = [&](auto fk) {
            constexpr uint32_t F = decltype(fk)::value;
            if (chain)
                return rs ? &moe_gemm_q_kernel<1, 4, 1, 4, W4, true, false, false, F, true, true, I8,
                                               r4dchain>
                          : &moe_gemm_q_kernel<1, 4, 1, 4, W4, true, false, false, F, true, false, I8,
                                               r4dchain>;
            return rs ? &moe_gemm_q_kernel<1, 4, 1, 4, W4, true, false, false, F, true, true, I8,
                                           R4dUnchained>
                      : &moe_gemm_q_kernel<1, 4, 1, 4, W4, true, false, false, F, true, false, I8,
                                           R4dUnchained>;
        };
        auto kern = pick(std::integral_constant<uint32_t, kFK>{});
        if constexpr (W4 >= 2) {
            bool fk2 = r4d_moe_gu_fk256<W4>(T, a_sorted);
            for (int q = 0; q < ncls; ++q) fk2 = fk2 && Kc[q] % (2 * (long long)kBlock) == 0;
            if (fk2) kern = pick(std::integral_constant<uint32_t, 2 * kFK>{});
        }
        emit_rc = R4D_EMIT(stream, kern,
                           dim3(tiles * 16u, ny), dim3(4 * kWave), 0,
                           (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                           (const int32_t*)sorted->data, (const int32_t*)off->data,
                           (uint16_t*)Y->data, nullptr,
                           (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,
                           (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv);
    } else {
        /* The prefill form is run-aligned when map scratch is present, uniform otherwise (e.g. a
         * caller that never asked the scratch hook); the answer is the same.
         *
         * The map builder's LDS tables hold kMaxExpert + 1 entries, so wider expert counts take the
         * uniform grid regardless of scratch. */
        const long long cap = r4d_blockmap_cap(T, n_expert, kPrefillRows);
        uint32_t* map = (form != 3 && n_expert <= (long long)kMaxExpert && a->scratch &&
                         a->scratch_bytes >= r4d_blockmap_bytes(cap))
                            ? (uint32_t*)a->scratch : nullptr;
        /* The 128-column tile where no column tile straddles two parts of the weight. */
        const bool wide_n = (N / parts) % kPrefillCols == 0;
        const unsigned nyw = (unsigned)(N / kPrefillCols);
        /* The down projection's 256-column tile: the sorted operand, one part, a w4 weight with
         * per-row scales and every class's width whole tiles take 256 columns over sixteen waves
         * instead of 128 over eight. Each wave's share is unchanged, so every element is the same
         * sum; the wider block stages the activation and run prologue once for twice the columns.
         * Other shapes keep the 128- or 64-column tile. */
        bool dn_wide = (W4 != 0 || I8) && a_sorted && parts == 1 &&
                       N % (2 * kPrefillCols) == 0;
        for (int q = 0; q < ncls; ++q) dn_wide = dn_wide && Nc[q] % (2 * kPrefillCols) == 0;
        const int tpin = g_moe_down_tile_pin;
        if (tpin < 0 || tpin > 2) return R4D_NO(RAD_E_INVAL);
        if (tpin == 2 && (!map || !dn_wide)) return R4D_NO(RAD_E_UNSUPPORTED);
        if (tpin == 1) dn_wide = false;
        if (map) {
            emit_rc = R4D_EMIT(stream, moe_blockmap_kernel<R4dUnchained>, dim3(1), dim3(256), 0,
                               (const int32_t*)off->data, (uint32_t)n_expert, (uint32_t)T,
                               (uint32_t)kPrefillRows, (uint32_t)cap, map);
            if (emit_rc != RAD_OK) return emit_rc;
            /* Every aligned instantiation uses UW and issues its first pass at the map read (see
             * the early-weight note at the kernel). */
            if (dn_wide) {
                /* Only a weight with a scale per row takes it (dn_wide is false otherwise); the
                 * block-fp8 instantiation would not compile, its scale being per 128x128 tile. */
                if constexpr (W4 != 0 || I8)
                    emit_rc = R4D_EMIT(stream, (moe_gemm_q_kernel<kPrefillRows / 16, 2 * kPrefillCols / 16, 2, 8, W4, false, true, true, kFK, true, false, I8, R4dUnchained>),
                               dim3((unsigned)cap, (unsigned)(N / (2 * kPrefillCols))),
                               dim3(16 * kWave), 0,
                               (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                               (const int32_t*)sorted->data, (const int32_t*)off->data,
                               (uint16_t*)Y->data, (const uint32_t*)map,
                               (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,
                               (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv);
            } else if (wide_n) {
                emit_rc = R4D_EMIT(stream, (moe_gemm_q_kernel<kPrefillRows / 16, kPrefillCols / 16, 2, 4, W4, false, true, true, kFK, true, false, I8, R4dUnchained>),
                               dim3((unsigned)cap, nyw), dim3(8 * kWave), 0,
                               (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                               (const int32_t*)sorted->data, (const int32_t*)off->data,
                               (uint16_t*)Y->data, (const uint32_t*)map,
                               (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,
                               (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv);
            } else {
                emit_rc = R4D_EMIT(stream, (moe_gemm_q_kernel<kPrefillRows / 16, 4, 2, 2, W4, false, true, true, kFK, true, false, I8, R4dUnchained>),
                               dim3((unsigned)cap, ny), dim3(4 * kWave), 0,
                               (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                               (const int32_t*)sorted->data, (const int32_t*)off->data,
                               (uint16_t*)Y->data, (const uint32_t*)map,
                               (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,
                               (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv);
            }
        } else if (wide_n) {
            emit_rc = R4D_EMIT(stream, (moe_gemm_q_kernel<kPrefillRows / 16, kPrefillCols / 16, 2, 4, W4, false, false, true, kFK, false, false, I8, R4dUnchained>),
                               dim3((unsigned)((T + kPrefillRows - 1) / kPrefillRows), nyw),
                               dim3(8 * kWave), 0,
                               (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                               (const int32_t*)sorted->data, (const int32_t*)off->data,
                               (uint16_t*)Y->data, nullptr,
                               (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,
                               (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv);
        } else {
            emit_rc = R4D_EMIT(stream, (moe_gemm_q_kernel<kPrefillRows / 16, 4, 2, 2, W4, false, false, true, kFK, false, false, I8, R4dUnchained>),
                               dim3((unsigned)((T + kPrefillRows - 1) / kPrefillRows), ny),
                               dim3(4 * kWave), 0,
                               (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                               (const int32_t*)sorted->data, (const int32_t*)off->data,
                               (uint16_t*)Y->data, nullptr,
                               (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,
                               (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv);
        }
    }
    if (emit_rc != RAD_OK) return emit_rc;
    R4D_MOE_CHK();
    return RAD_OK;
}

/* Bytes for the run map, zero where it is not used.
 *
 * T comes from the parameters (`M * top_k`, the routed row count), not an operand, since the hook
 * runs at declare before tensors are bound; the launcher derives the same count from Y. If they
 * ever disagree it is safe: the launcher checks the scratch it was handed and falls back to the
 * uniform grid when it is short.
 */
static int64_t moe_gemm_map_scratch(const RadArgs* a) {
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k    = r4d_geti_or(a, "top_k", 0);
    const long long M        = r4d_geti_or(a, "M", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || M <= 0) return 0;
    const long long T = M * top_k;
    if (r4d_moe_decode_band(T, n_expert)) return 0;      /* the (tile, run) grid needs no map */
    return r4d_blockmap_bytes(r4d_blockmap_cap(T, n_expert, kPrefillRows));
}

/* The run map and the decode band for another unit's grouped GEMM over the same sort
 * (r4d_moe_iqr.hip): the map's entry bound and bytes at `fm` rows a block, and its one launch. */
long long r4d_moe_blockmap_cap(long long T, long long n_expert, long long fm) {
    return r4d_blockmap_cap(T, n_expert, fm);
}
long long r4d_moe_blockmap_bytes(long long cap) { return r4d_blockmap_bytes(cap); }
int r4d_moe_blockmap_emit(RadStream s, const int32_t* off, uint32_t n_expert, uint32_t T,
                          uint32_t fm, uint32_t cap, uint32_t* map) {
    return R4D_EMIT(s, moe_blockmap_kernel<R4dUnchained>, dim3(1), dim3(256), 0, off, n_expert, T, fm,
                    cap, map);
}
bool r4d_moe_in_decode_band(long long T, long long n_expert) {
    return r4d_moe_decode_band(T, n_expert);
}

extern "C" int64_t r4d_rad_moe_gemm_q_scratch(const RadArgs* a)    { return moe_gemm_map_scratch(a); }
extern "C" int64_t r4d_rad_moe_gemm_w4a8_scratch(const RadArgs* a) { return moe_gemm_map_scratch(a); }

extern "C" void r4d_moe_gemm_pin_prefill(int form) { g_moe_prefill_form = form; }
extern "C" void r4d_moe_gemm_pin_fk(int pin) { g_moe_fk_pin = pin; }
extern "C" void r4d_moe_gemm_pin_down_tile(int pin) { g_moe_down_tile_pin = pin; }
extern "C" void r4d_moe_gemm_pin_down_dec(int pin) {
    g_moe_down_dec_pin = pin < (int)kMoeDownDecMaxG ? pin : (int)kMoeDownDecMaxG;
}

/* The four-bit forms (an E4M3 scale per 64 or 32 of K for w4nl64a8h / w4nl32a8h, else a bf16
 * scale per 128) and the five-bit w5nl64a8h. */
extern "C" int r4d_rad_moe_gemm_w4a8(const RadArgs* a, RadStream stream) {
    const char* wdt = r4d_gets(a, "dtype");
    if (wdt && !std::strcmp(wdt, "w4nl64a8h")) return moe_gemm_launch<2>(a, stream);
    if (wdt && !std::strcmp(wdt, "w4nl32a8h")) return moe_gemm_launch<3>(a, stream);
    if (wdt && !std::strcmp(wdt, "w5nl64a8h")) return moe_gemm_launch<4>(a, stream);
    return moe_gemm_launch<1>(a, stream);
}

/* The int8 activation's rows. `i8a8` and `i8a8h`: int8 experts, row-major codes with a bf16 scale
 * per (row, 128 of K). The four-bit experts keep their stored planes: `w4a8i8` and `w4a8i8h` the
 * integer grid with a bf16 scale per (row, 128), `w4nl64a8i8h` and `w4nl32a8i8h` the w4nl codebook
 * with an E4M3 scale per (row, 64 or 32 of K). */
static int moe_gemm_w4a8i8(const RadArgs* a, RadStream stream, int ncls) {
    const char* wdt = r4d_gets(a, "dtype");
    if (wdt && !std::strcmp(wdt, "w4nl64a8i8h")) return moe_gemm_launch<2, true>(a, stream, ncls);
    if (wdt && !std::strcmp(wdt, "w4nl32a8i8h")) return moe_gemm_launch<3, true>(a, stream, ncls);
    return moe_gemm_launch<1, true>(a, stream, ncls);
}
extern "C" int r4d_rad_moe_gemm_i8a8(const RadArgs* a, RadStream s) {
    return moe_gemm_launch<0, true>(a, s);
}
extern "C" int r4d_rad_moe_gemm_i8a8_mod4(const RadArgs* a, RadStream s) {
    return moe_gemm_launch<0, true>(a, s, 4);
}
extern "C" int r4d_rad_moe_gemm_w4a8i8(const RadArgs* a, RadStream s) {
    return moe_gemm_w4a8i8(a, s, 2);
}
extern "C" int r4d_rad_moe_gemm_w4a8i8_mod4(const RadArgs* a, RadStream s) {
    return moe_gemm_w4a8i8(a, s, 4);
}
extern "C" int r4d_rad_moe_gemm_w4a8_mod4(const RadArgs* a, RadStream stream) {
    const char* wdt = r4d_gets(a, "dtype");
    if (wdt && !std::strcmp(wdt, "w4nl64a8h")) return moe_gemm_launch<2>(a, stream, 4);
    if (wdt && !std::strcmp(wdt, "w4nl32a8h")) return moe_gemm_launch<3>(a, stream, 4);
    if (wdt && !std::strcmp(wdt, "w5nl64a8h")) return moe_gemm_launch<4>(a, stream, 4);
    return moe_gemm_launch<1>(a, stream, 4);
}

/* `moe_gemm` at bf16: operands a, w (one table), sorted_tok, expert_offset -> y. Band and forms
 * follow moe_gemm_launch's rule, so the same scratch hook sizes the run map.
 *
 * Every load is sixteen bytes, hence the alignment checks: row pitches must be multiples of eight
 * elements and stacked experts 16-byte aligned. Pointer-table entries live in device memory and
 * cannot be checked here; their allocators hand out movement-unit-aligned addresses. */
extern "C" int r4d_rad_moe_gemm_bf16(const RadArgs* a, RadStream stream) {
    int emit_rc = RAD_OK;
    const RadTensor* A      = r4d_opt(a, 0);
    const RadTensor* W      = r4d_opt(a, 1);
    const RadTensor* sorted = r4d_opt(a, 2);
    const RadTensor* off    = r4d_opt(a, 3);
    const RadTensor* Y      = r4d_opt(a, 4);
    if (!A || !W || !sorted || !off || !Y) return R4D_NO(RAD_E_INVAL);
    if (!A->data || !sorted->data || !off->data || !Y->data) return R4D_NO(RAD_E_INVAL);
    const char* dt = r4d_gets(a, "dtype");
    if (!dt || std::strcmp(dt, "bf16")) return R4D_NO(RAD_E_DTYPE);

    const long long N = r4d_geti_or(a, "N", 0);
    const long long K = r4d_geti_or(a, "K", 0);
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0);
    if (N <= 0 || K <= 0 || n_expert <= 0 || top_k <= 0) return R4D_NO(RAD_E_SHAPE);
    /* Column tiles are 64 wide and all weight; a K pass is 64 deep. */
    if (N % 64 || K % (long long)kBK16) return R4D_NO(RAD_E_SHAPE);
    if (A->dtype != RAD_BF16 || Y->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (sorted->dtype != RAD_I32 || off->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    long long T = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &T);
    if (y_ld < 0 || T <= 0) return R4D_NO(RAD_E_STRIDE);
    long long M = 0;
    const long long a_ld = r4d_rows_pitch(A, K, &M);
    if (a_ld < 0 || M <= 0 || a_ld % 8 || ((uintptr_t)A->data % 16)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(sorted) < T || r4d_numel(off) < n_expert + 1) return R4D_NO(RAD_E_SHAPE);

    /* The activation's row count is checked, not inferred (see moe_gemm_launch). */
    const char* ord = r4d_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (M < (a_sorted ? T : (T + top_k - 1) / top_k)) return R4D_NO(RAD_E_SHAPE);

    const void* const* tab = nullptr;
    size_t w_s0 = 0;
    long long w_ld = 0;
    int rc = wtab_of(W, RAD_BF16, 16, n_expert, N, K, &tab, &w_s0, &w_ld);
    if (rc != RAD_OK) return rc;
    if (w_ld % 8 || w_s0 % 16) return R4D_NO(RAD_E_STRIDE);

    const uint32_t ny = (uint32_t)(N / 64);
    const int form = g_moe_prefill_form;
    /* Chained forms (r4d_chain.h) up to a decode step's routed slots, in either band: a layer's
     * few plain experts beside a quantised table are a table of at most eight, which a verify
     * step's forty slots put in the prefill form, run map included. */
    const bool chain = r4d_geti_or(a, "chain", 0) != 0 && T <= 1024;
    if ((form == 0 && r4d_moe_decode_band(T, n_expert)) || form == 2) {
        const unsigned tiles = (unsigned)((T + 15) / 16);
#define R4D_MOE16D(CH_)                                                                          \
        R4D_EMIT(stream, (moe_gemm_bf16_kernel<1, 4, 1, 4, true, false, false, CH_>),            \
                 dim3(tiles * 16u, ny), dim3(4 * kWave), 0,                                      \
                 (const uint16_t*)A->data, tab, w_s0,                                            \
                 (const int32_t*)sorted->data, (const int32_t*)off->data,                        \
                 (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)K,                      \
                 (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)(a_sorted ? 1 : 0),              \
                 (uint32_t)a_ld, (uint32_t)w_ld, (uint32_t)y_ld, nullptr)
        if (chain) emit_rc = R4D_MOE16D(r4dchain);
        else       emit_rc = R4D_MOE16D(R4dUnchained);
#undef R4D_MOE16D
    } else {
        const long long cap = r4d_blockmap_cap(T, n_expert, kPrefillRows);
        uint32_t* map = (form != 3 && n_expert <= (long long)kMaxExpert && a->scratch &&
                         a->scratch_bytes >= r4d_blockmap_bytes(cap))
                            ? (uint32_t*)a->scratch : nullptr;
        /* The 128-column, eight-wave tile wherever the width allows, as in the fp8 forms. */
        const bool wide_n = N % kPrefillCols == 0;
        const unsigned nyw = (unsigned)(N / kPrefillCols);
        const unsigned gx = map ? (unsigned)cap : (unsigned)((T + kPrefillRows - 1) / kPrefillRows);
        if (map) {
#define R4D_MOEBM(CH_)                                                                           \
            R4D_EMIT(stream, moe_blockmap_kernel<CH_>, dim3(1), dim3(256), 0,                    \
                     (const int32_t*)off->data, (uint32_t)n_expert, (uint32_t)T,                 \
                     (uint32_t)kPrefillRows, (uint32_t)cap, map)
            if (chain) emit_rc = R4D_MOEBM(r4dchain);
            else       emit_rc = R4D_MOEBM(R4dUnchained);
#undef R4D_MOEBM
            if (emit_rc != RAD_OK) return emit_rc;
        }
#define R4D_MOE16C(AL_, NTL_, WGN_, NYG_, CH_)                                                   \
        R4D_EMIT(stream,                                                                       \
                 (moe_gemm_bf16_kernel<kPrefillRows / 16, NTL_, 2, WGN_, false, AL_, true,      \
                                       CH_>),                                                   \
                 dim3(gx, NYG_), dim3(2 * WGN_ * kWave), 0,                                      \
                 (const uint16_t*)A->data, tab, w_s0,                                            \
                 (const int32_t*)sorted->data, (const int32_t*)off->data,                        \
                 (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)K,                      \
                 (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)(a_sorted ? 1 : 0),              \
                 (uint32_t)a_ld, (uint32_t)w_ld, (uint32_t)y_ld, (const uint32_t*)map)
#define R4D_MOE16(AL_, NTL_, WGN_, NYG_)                                                         \
        (chain ? R4D_MOE16C(AL_, NTL_, WGN_, NYG_, r4dchain)                                    \
               : R4D_MOE16C(AL_, NTL_, WGN_, NYG_, R4dUnchained))
        if (map)  emit_rc = wide_n ? R4D_MOE16(true, kPrefillCols / 16, 4, nyw)
                                   : R4D_MOE16(true, 4, 2, ny);
        else      emit_rc = wide_n ? R4D_MOE16(false, kPrefillCols / 16, 4, nyw)
                                   : R4D_MOE16(false, 4, 2, ny);
#undef R4D_MOE16
#undef R4D_MOE16C
    }
    if (emit_rc != RAD_OK) return emit_rc;
    R4D_MOE_CHK();
    return RAD_OK;
}

extern "C" int64_t r4d_rad_moe_gemm_bf16_scratch(const RadArgs* a) {
    return moe_gemm_map_scratch(a);
}

extern "C" int64_t r4d_rad_moe_gather_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 1);
    if (M <= 0 || top_k <= 0) return R4D_NO(RAD_E_SHAPE);
    return M * top_k * (long long)sizeof(int32_t);
}

/* moe_gather, or with `given` moe_gather_inv, whose third operand is the inverse moe_scatter_inv
 * wrote: no prepass and no self-scan, and the same kernel at the same T, so the same sums. */
static int moe_gather_body(const RadArgs* a, RadStream s, bool given) {
    const RadTensor* ye     = r4d_opt(a, 0);
    const RadTensor* ew     = r4d_opt(a, 1);
    const RadTensor* sorted = r4d_opt(a, 2);
    const RadTensor* sh     = r4d_opt(a, 3);
    const RadTensor* sg     = r4d_opt(a, 4);
    const RadTensor* Y      = r4d_opt(a, 5);
    if (!ye || !ew || !sorted || !Y) return R4D_NO(RAD_E_INVAL);
    if (!ye->data || !ew->data || !sorted->data || !Y->data) return R4D_NO(RAD_E_INVAL);
    const long long n = r4d_geti_or(a, "n", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0);
    if (n <= 0 || top_k <= 0) return R4D_NO(RAD_E_SHAPE);
    if (ye->dtype != RAD_BF16 || (ew->dtype != RAD_BF16 && ew->dtype != RAD_F32) || Y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (sorted->dtype != RAD_I32 || !r4d_contig(sorted)) return R4D_NO(RAD_E_STRIDE);
    /* The shared arm is a pair: the plane and its per-row gate, or neither. */
    if ((sh != nullptr) != (sg != nullptr)) return R4D_NO(RAD_E_INVAL);
    if (sh && (sh->dtype != RAD_BF16 || sg->dtype != RAD_BF16)) return R4D_NO(RAD_E_DTYPE);

    long long M = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    if (y_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    long long Te = 0;
    const long long ye_ld = r4d_rows_pitch(ye, n, &Te);
    if (ye_ld < 0) return R4D_NO(RAD_E_STRIDE);
    long long Mw = 0;
    const long long ew_ld = r4d_rows_pitch(ew, top_k, &Mw);
    if (ew_ld < 0 || Mw < M) return R4D_NO(RAD_E_STRIDE);
    const long long T = M * top_k;
    if (Te < T || r4d_numel(sorted) < T) return R4D_NO(RAD_E_SHAPE);

    /* SELF: small enough for the gather to build its own slot table, skipping the two prepass
     * kernels (see gather_kernel). Bounded on T because each of the M*gy blocks repeats the scan;
     * at the cap it is under half a megabyte of L2 traffic. Prefill takes the prepass. A given
     * table replaces both, and `small` keeps the kernel choice the self-scan would make. */
    const bool small = T <= (long long)kGatherSelfT && top_k <= (long long)kGatherSlots;
    const bool self = small && !given;

    const int64_t need = self || given ? 0 : T * (long long)sizeof(int32_t);
    int32_t* inv = given ? (int32_t*)sorted->data : nullptr;
    if (!self && !given) {
        if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);
        inv = (int32_t*)a->scratch;
        const unsigned blocks = (unsigned)((T + 255) / 256);
        /* Two passes: an unclaimed slot must read -1, and the clear cannot be fused into the
         * scatter because no thread knows which entries nobody will write. */
        hipLaunchKernelGGL(gather_inv_kernel, dim3(blocks), dim3(256), 0, (hipStream_t)s,
                           inv, (uint32_t)T);
        R4D_MOE_CHK();
        hipLaunchKernelGGL(gather_scatter_kernel, dim3(blocks), dim3(256), 0, (hipStream_t)s,
                           (const int32_t*)sorted->data, inv, (uint32_t)T);
        R4D_MOE_CHK();
    }
    long long shrows = 0, sgrows = 0, sh_ld = 0, sg_ld = 0;
    int sig = 0;
    if (sh) {
        sh_ld = r4d_rows_pitch(sh, n, &shrows);
        sg_ld = r4d_rows_pitch(sg, 1, &sgrows);
        if (sh_ld < 0 || sg_ld < 0 || shrows < M || sgrows < M) return R4D_NO(RAD_E_STRIDE);
        const char* const act = r4d_gets(a, "act");
        if (act && !std::strcmp(act, "sigmoid"))   sig = 1;
        else if (act && std::strcmp(act, "none"))  return R4D_NO(RAD_E_UNSUPPORTED);
    }

    /* The prefill form (gather_wide_kernel, a wave per token, sixteen bytes per lane) past the
     * self-scan bound, where top_k fits kGatherWideSlots and every row is 16-byte aligned. Same
     * sums as gather_kernel, which serves everything else. */
    auto al16 = [](const void* p, long long ld) {
        return ((uintptr_t)p % 16) == 0 && ld % (long long)kGatherWideCols == 0;
    };
    const bool wide_ok = ew->dtype == RAD_BF16 && !small && top_k <= (long long)kGatherWideSlots &&
                         n % (long long)kGatherWideCols == 0 && al16(ye->data, ye_ld) &&
                         al16(Y->data, y_ld) && (!sh || al16(sh->data, sh_ld));
    const int gpin = g_moe_gather_pin;
    if (gpin < 0 || gpin > 2) return R4D_NO(RAD_E_INVAL);
    if (gpin >= 2 && !wide_ok) return R4D_NO(RAD_E_UNSUPPORTED);
    if ((gpin == 0 && wide_ok) || gpin >= 2) {
        constexpr uint32_t NW = 8;
        const unsigned gw = (unsigned)((n + kWave * kGatherWideCols - 1) / (kWave * kGatherWideCols));
        hipLaunchKernelGGL((gather_wide_kernel<NW>), dim3((unsigned)((M + NW - 1) / NW), gw),
                           dim3(NW * kWave), 0, (hipStream_t)s,
                           (const uint16_t*)ye->data, (const uint16_t*)ew->data, inv,
                           (uint16_t*)Y->data, (uint32_t)M, (uint32_t)n, (uint32_t)top_k,
                           (uint32_t)ye_ld, (uint32_t)ew_ld, (uint32_t)y_ld,
                           sh ? (const uint16_t*)sh->data : nullptr,
                           sh ? (const uint16_t*)sg->data : nullptr,
                           (uint32_t)sh_ld, (uint32_t)sg_ld, sig);
        R4D_MOE_CHK();
        return RAD_OK;
    }

    const unsigned gy = (unsigned)((n + 255) / 256);
    /* The chained form where the row resolved with `chain` and this launch is the op's only one
     * (no prepass before it: a decode step). */
    const bool chain = r4d_geti_or(a, "chain", 0) != 0 && (self || given) && M <= 64;
#define R4D_MG(...)                                                                              \
    hipLaunchKernelGGL((__VA_ARGS__), dim3((unsigned)M, gy), dim3(256), 0, (hipStream_t)s,       \
                       (const uint16_t*)ye->data, (const void*)ew->data, inv,                    \
                       (uint16_t*)Y->data, (uint32_t)M, (uint32_t)n, (uint32_t)top_k,            \
                       (uint32_t)ye_ld, (uint32_t)ew_ld, (uint32_t)y_ld,                         \
                       self ? (const int32_t*)sorted->data : nullptr, (uint32_t)T,               \
                       sh ? (const uint16_t*)sh->data : nullptr,                                 \
                       sh ? (const uint16_t*)sg->data : nullptr,                                 \
                       (uint32_t)sh_ld, (uint32_t)sg_ld, sig)
    if (ew->dtype == RAD_F32) {
        if (chain) R4D_MG(r4dchain_moe_gather<256, true>);
        else       R4D_MG(gather_kernel<256, true>);
    } else {
        if (chain) R4D_MG(r4dchain_moe_gather<256>);
        else       R4D_MG(gather_kernel<256>);
    }
#undef R4D_MG
    R4D_MOE_CHK();
    return RAD_OK;
}

extern "C" int r4d_rad_moe_gather(const RadArgs* a, RadStream s) {
    return moe_gather_body(a, s, false);
}

extern "C" int r4d_rad_moe_gather_inv(const RadArgs* a, RadStream s) {
    return moe_gather_body(a, s, true);
}

extern "C" int r4d_rad_scale_rows(const RadArgs* a, RadStream s) {
    const RadTensor* x  = r4d_opt(a, 0);
    const RadTensor* sc = r4d_opt(a, 1);
    const RadTensor* ad = r4d_opt(a, 2);          /* optional: the fused gate-and-add form */
    const RadTensor* y  = r4d_opt(a, 3);
    if (!x || !sc || !y || !x->data || !sc->data || !y->data) return R4D_NO(RAD_E_INVAL);
    if (ad && !ad->data) ad = nullptr;
    if (x->dtype != RAD_BF16 || sc->dtype != RAD_BF16 || y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (ad && ad->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    /* Read as a name, not via r4d_act_code (whose vocabulary is silu/sigmoid, not sigmoid/none), so
     * "none" is distinguishable from a typo. An unknown name is refused. */
    const char* act = r4d_gets(a, "act");
    const int sig = (act && !std::strcmp(act, "sigmoid")) ? 1 : 0;
    if (act && std::strcmp(act, "sigmoid") && std::strcmp(act, "none")) return R4D_NO(RAD_E_INVAL);

    long long M = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    if (y_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    long long Mx = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &Mx);
    if (x_ld < 0 || Mx < M) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(sc) < M) return R4D_NO(RAD_E_SHAPE);
    long long a_ld = 0;
    if (ad) {
        long long Ma = 0;
        a_ld = r4d_rows_pitch(ad, n, &Ma);
        if (a_ld < 0 || Ma < M) return R4D_NO(RAD_E_STRIDE);
    }
    /* One scalar per row, as [M] or [M, 1]; the row stride is dim 0's in both. */
    const long long s_ld = r4d_stride(sc, 0) > 0 ? r4d_stride(sc, 0) : 1;

    hipLaunchKernelGGL((scale_rows_kernel<256>), dim3((unsigned)M), dim3(256), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (const uint16_t*)sc->data,
                       ad ? (const uint16_t*)ad->data : nullptr, (uint16_t*)y->data,
                       (uint32_t)M, (uint32_t)n, (uint32_t)x_ld, (uint32_t)(s_ld ? s_ld : 1),
                       (uint32_t)a_ld, (uint32_t)y_ld, sig);
    R4D_MOE_CHK();
    return RAD_OK;
}

/* ================================================================== router GEMM + router_topk_scatter
 *
 * The router GEMM, its selection and its sort in one launch, plus, where the layer has a shared
 * expert, that expert's one-column gate. rad_block_moe_fp8.h issues it up to kRgMaxM rows wherever
 * it declares router_topk_scatter.
 *
 * At decode the router GEMM and router_topk_scatter are two dependent dispatches per routed layer
 * with almost no work in the second. The selection needs a row's every logit, so it runs in the
 * workgroup that finishes last, named by an arrival counter (the pattern of gemm_fp8a8's split-K
 * finisher and hc_up_mix's quantiser: self-resetting, never shared by concurrent launches). Here
 * the counter is the op instance's, taken at init (see the arrival-counter note).
 *
 * The shared expert's gate (gemm_nt at N = 1 over the same input) rides along as one more
 * workgroup (`skg` > 0), removing its separate launch.
 *
 * Byte-identical to the launches it replaces:
 *   - The GEMM is r4d_gemm_bf16_nt_m64_kernel's arithmetic: an output element is SK partial tiles,
 *     each one wave's WMMA chain over a contiguous K slice, summed in ascending split order from
 *     0.f and narrowed by the language's conversion. Only SK matters, so the split counts come from
 *     the same law (cfg_for with the same N-tile count), and a wave may run several (tile, split)
 *     items, each into its own LDS slot.
 *   - The selection and sort are router_topk_scatter_kernel's body: wave-local and stable, so the
 *     last block's wave count changes no bit.
 *
 * A tuned point for the unfused router would break the first, so this row is NOTUNE. Decode only:
 * at most kRgMaxM rows (four 16-row tiles); beyond that the architecture issues the pair. */
constexpr long long kRgMaxM = 64;

namespace {

constexpr int kRgKU = 8;                  /* r4d_gemm_bf16_nt_m64's KU: the unroll, not the order */

typedef __bf16 rg_v8bf __attribute__((ext_vector_type(8)));
typedef float  rg_v8f  __attribute__((ext_vector_type(8)));
/* Global-address-space pointer for the fragment loads, as r4d_gemm_bf16_nt_m64.hip holds it. */
typedef __attribute__((address_space(1))) const unsigned long long* rg_gu2;

__device__ __forceinline__ rg_v8f rg_mma(rg_v8bf a, rg_v8bf b, rg_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
/* r4d_gemm_bf16_nt_m64.hip's fragU: k step `u` of a K-contiguous row, two runs of four. */
__device__ __forceinline__ rg_v8bf rg_fragU(rg_gu2 p, int u) {
    union { unsigned long long w[2]; rg_v8bf f; } x;
    x.w[0] = p[4 * u];
    x.w[1] = p[4 * u + 2];
    return x.f;
}
/* r4d_gdn_wmma.h's rowGo: this lane's row, clamped to `lim`, as a 32-bit element offset. */
__device__ __forceinline__ unsigned rg_rowGo(unsigned rowstride, int row0, int lim, int lane) {
    int r = row0 + (lane & 15);
    r = r < lim ? r : lim;
    return (unsigned)r * rowstride + 4u * (unsigned)(lane >> 4);
}

/* One (tile, split) item: the 16 x 16 partial tile r4d_gemm_bf16_nt_m64_kernel's wave (n0, ks)
 * accumulates at MT = 1, with the same loads and WMMA chain from zero. */
__device__ __forceinline__ rg_v8f rg_item(const uint16_t* __restrict__ A,
                                          const uint16_t* __restrict__ W, int M, int K, int N,
                                          int n0, int m0, int ks, int SK, int lane) {
    const int Ks = K / SK, ksteps = Ks / 16, kbase = ks * Ks;
    rg_v8f acc = (rg_v8f)(0.f);
    const unsigned woff = rg_rowGo((unsigned)K, n0, N - 1, lane) + (unsigned)kbase;
    const unsigned aoff = rg_rowGo((unsigned)K, m0, M - 1, lane) + (unsigned)kbase;
    if (n0 < N) {
        rg_gu2 pw = (rg_gu2)(const unsigned long long*)(W + woff);
        rg_gu2 pa = (rg_gu2)(const unsigned long long*)(A + aoff);
        int s = 0;
        for (; s + kRgKU <= ksteps; s += kRgKU) {
#pragma unroll
            for (int u = 0; u < kRgKU; u++) {
                const rg_v8bf b = rg_fragU(pw, u);
                acc = rg_mma(rg_fragU(pa, u), b, acc);
            }
            pw += kRgKU * 4;
            pa += kRgKU * 4;
        }
        for (; s < ksteps; s++) {
            const rg_v8bf b = rg_fragU(pw, 0);
            pw += 4;
            acc = rg_mma(rg_fragU(pa, 0), b, acc);
            pa += 4;
        }
    }
    return acc;
}

/* Launch geometry: `sk` the router's split, `tpb` its 16-column tiles per block, `nrb` the
 * router's blocks along x, `skg` the gate's split (0: no gate; its block is x = nrb). Rows are
 * tiled along y, one 16-row tile per block. */
struct RgGeom {
    uint32_t M, K, n_expert, n_local, top_k, lg_ld;
    int32_t  norm, base;
    uint32_t sk, tpb, nrb, skg;
};

/* `lg` is neither const nor __restrict__: every block writes it and the last one reads it back. */
template <uint32_t MAXOWN, uint32_t NW, class CH>
__global__ __launch_bounds__(NW * kWave) void router_gemm_topk_scatter_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ W,
    const uint16_t* __restrict__ Wg, uint16_t* lg, uint16_t* __restrict__ gate,
    int32_t* __restrict__ ids, uint16_t* __restrict__ w, int32_t* __restrict__ sorted,
    int32_t* __restrict__ off, int32_t* __restrict__ cnt, unsigned int* __restrict__ ctr,
    RgGeom g, Protect prot) {
    /* The GEMM's partial tiles, then (last block only, after two barriers) the sort's LDS; the
     * shim sizes it for the larger. */
    if (CH::on) r4d_chain_wait();
    extern __shared__ __align__(16) unsigned char rg_smem[];
    float* const red = (float*)rg_smem;
    const uint32_t tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;

    const bool      gb  = g.skg > 0 && blockIdx.x == g.nrb;
    const int       SK  = (int)(gb ? g.skg : g.sk);
    const uint32_t  nti = gb ? 1u : g.tpb;
    const int       N   = gb ? 1 : (int)g.n_expert;
    const uint32_t  nt0 = gb ? 0u : blockIdx.x * g.tpb;
    const uint16_t* Wb  = gb ? Wg : W;
    const int       m0  = (int)(blockIdx.y * 16u);

    const uint32_t items = nti * (uint32_t)SK;
    for (uint32_t it = wave; it < items; it += NW) {
        const uint32_t j = it / (uint32_t)SK, ks = it % (uint32_t)SK;
        const rg_v8f acc = rg_item(A, Wb, (int)g.M, (int)g.K, N, (int)((nt0 + j) * 16u), m0,
                                   (int)ks, SK, (int)lane);
        float* const slot = red + (size_t)it * 256;           /* it == j * SK + ks */
#pragma unroll
        for (int e = 0; e < 8; e++) slot[e * (int)kWave + (int)lane] = acc[e];
    }
    __syncthreads();
    /* The reduction, as in r4d_gemm_bf16_nt_m64_kernel: splits in ascending order from 0.f, and
     * the language's float-to-bf16 conversion (not f32_to_bf16, which differs on NaN). */
    for (uint32_t idx = tid; idx < nti * 256u; idx += NW * kWave) {
        const uint32_t j = idx >> 8, o = idx & 255u;
        float v = 0.f;
        for (int s = 0; s < SK; s++) v += red[((size_t)j * (size_t)SK + (size_t)s) * 256 + o];
        const uint32_t l = o & 31u;
        const int n = (int)((nt0 + j) * 16u + (l & 15u));
        const int m = m0 + (int)(8u * (l >> 4) + (o >> 5));
        if (n < N && m < (int)g.M) {
            if (gb) ((__hip_bfloat16*)gate)[m] = (__hip_bfloat16)v;
            else    ((__hip_bfloat16*)lg)[(size_t)m * g.lg_ld + (size_t)n] = (__hip_bfloat16)v;
        }
    }

    /* ---- the last block ranks. Every store is made visible device-wide before the arrival is
     * published; the block completing the count resets it for the next launch and acquires the
     * other blocks' logits. */
    __threadfence();
    __syncthreads();
    __shared__ unsigned int s_arrived;
    if (tid == 0)
        s_arrived = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (s_arrived + 1u == gridDim.x * gridDim.y) {
        if (tid == 0) __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __threadfence();

        /* router_topk_scatter_kernel, from here to the end. */
        ScatterLds<NW>& sl = *reinterpret_cast<ScatterLds<NW>*>(rg_smem);
        for (uint32_t e = tid; e < g.n_local; e += blockDim.x) sl.cnt[e] = 0;
        const SlotSink snk{sl.e, g.base, g.n_local, prot};
        for (uint32_t m = wave; m < g.M; m += NW)
            router_topk_row<MAXOWN>(lg, ids, w, m, g.n_expert, g.top_k, g.lg_ld, g.norm, lane, snk);
        lds_barrier();
        scatter_small_place(sorted, off, cnt, g.M * g.top_k, g.n_local, sl);
    }
    if (CH::on) r4d_chain_done();
}

/* The arrival counter belongs to the op instance and is taken at init, not on the launch path.
 *
 * It must persist between launches, zeroed (the last block resets it; a non-zero start would rank
 * before the GEMM finished), so scratch cannot hold it: other ops write the arena between
 * launches. Allocating on first use (as gemm_fp8a8's split_ctr and hc_bf16's mix counters do)
 * escapes the VRAM budget. Init runs once per resolved instance
 * during declare, before the engine measures free VRAM, so its allocation is counted.
 *
 * One counter per instance (op, band, domain, rank). Each routed layer declares its own op, and an
 * instance's launches are ordered since they write the same declared outputs. Counters are 64-byte
 * slots of one zeroed block per device, freed with its last instance. */
constexpr uint32_t kRgCtrStride = 16;       /* uint32 words a slot: one 64-byte line each */
constexpr uint32_t kRgCtrSlots  = 4096;     /* instances a device; a model declares hundreds */

struct RgCtrBlock {
    unsigned int* base = nullptr;
    uint32_t next = 0;                      /* slots handed out since the block was made */
    uint32_t live = 0;                      /* instances holding one */
};
struct RgInst {
    int dev = 0;
    unsigned int* ctr = nullptr;
};
std::mutex g_rg_mu;
std::map<int, RgCtrBlock> g_rg_blocks;

}  /* namespace */

extern "C" int r4d_rad_router_gemm_topk_scatter_init(const RadParam*, int, int, int,
                                                     void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return RAD_E_DEVICE;
    std::lock_guard<std::mutex> lk(g_rg_mu);
    RgCtrBlock& blk = g_rg_blocks[dev];
    if (!blk.base) {
        const size_t bytes = (size_t)kRgCtrSlots * kRgCtrStride * sizeof(unsigned int);
        void* p = nullptr;
        if (hipMalloc(&p, bytes) != hipSuccess) return RAD_E_NOMEM;
        /* Zero before any launch reads it. In the engine hipMemset is the AQL backend's and has
         * finished when it returns (core/device/aql.cpp); under HIP's own runtime (r4d_selftest) it
         * runs on the null stream, which a non-blocking stream does not wait for, so wait here. */
        if (hipMemset(p, 0, bytes) != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
            (void)hipFree(p);
            return RAD_E_DEVICE;
        }
        blk.base = (unsigned int*)p;
        blk.next = 0;
    }
    /* A slot is never reused while its block lives: one returned by fini may not be zero if its
     * last launch did not finish. */
    if (blk.next >= kRgCtrSlots) return RAD_E_FULL;
    RgInst* in = new RgInst;
    in->dev = dev;
    in->ctr = blk.base + (size_t)blk.next * kRgCtrStride;
    ++blk.next;
    ++blk.live;
    *out = in;
    return RAD_OK;
}

extern "C" void r4d_rad_router_gemm_topk_scatter_fini(void* inst) {
    RgInst* in = (RgInst*)inst;
    if (!in) return;
    std::lock_guard<std::mutex> lk(g_rg_mu);
    auto it = g_rg_blocks.find(in->dev);
    if (it != g_rg_blocks.end() && it->second.live > 0 && --it->second.live == 0) {
        int prev = 0;
        const bool moved = hipGetDevice(&prev) == hipSuccess && prev != in->dev &&
                           hipSetDevice(in->dev) == hipSuccess;
        (void)hipFree(it->second.base);
        if (moved) (void)hipSetDevice(prev);
        g_rg_blocks.erase(it);
    }
    delete in;
}

/* Operands: a, b (the router), gate_b? -> logits, expert_ids, expert_w, sorted_tok,
 * expert_offset, expert_count, gate?. The two optional ones come together or not at all. The
 * replaced shims' checks are all made here, plus the row bound. */
extern "C" int r4d_rad_router_gemm_topk_scatter(const RadArgs* a, RadStream stream) {
    int emit_rc = RAD_OK;
    const RadTensor* x      = r4d_opt(a, 0);
    const RadTensor* wr     = r4d_opt(a, 1);
    const RadTensor* wg     = r4d_opt(a, 2);
    const RadTensor* lg     = r4d_opt(a, 3);
    const RadTensor* ids    = r4d_opt(a, 4);
    const RadTensor* w      = r4d_opt(a, 5);
    const RadTensor* sorted = r4d_opt(a, 6);
    const RadTensor* off    = r4d_opt(a, 7);
    const RadTensor* cnt    = r4d_opt(a, 8);
    const RadTensor* gate   = r4d_opt(a, 9);
    if (!x || !wr || !lg || !ids || !w || !sorted || !off || !cnt) return R4D_NO(RAD_E_INVAL);
    if ((wg != nullptr) != (gate != nullptr)) return R4D_NO(RAD_E_INVAL);

    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long n_local  = r4d_geti_or(a, "n_local", n_expert);
    const long long top_k    = r4d_geti_or(a, "top_k", 0);
    const long long norm     = r4d_geti_or(a, "norm", 0);
    const long long base     = r4d_geti_or(a, "expert_base", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || top_k > n_expert)
        return R4D_NO(RAD_E_SHAPE);
    if (top_k > (long long)kWave) return R4D_NO(RAD_E_SHAPE);
    if (n_local <= 0 || n_local > n_expert) return R4D_NO(RAD_E_SHAPE);
    if (base < 0 || base + n_local > n_expert) return R4D_NO(RAD_E_SHAPE);
    Protect prot;
    const int prc = protect_of(a, n_local, &prot);
    if (prc != RAD_OK) return prc;
    if (x->dtype != RAD_BF16 || wr->dtype != RAD_BF16 || lg->dtype != RAD_BF16 ||
        w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (wg && (wg->dtype != RAD_BF16 || gate->dtype != RAD_BF16)) return R4D_NO(RAD_E_DTYPE);
    if (ids->dtype != RAD_I32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    /* The GEMM's operands, as gemm_ops reads them: A [M, K] and the output contiguous, A and the
     * weights 16-byte aligned. */
    long long M = 0, K = 0;
    if (r4d_rows_of(x, &M, &K) != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(x)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_geti_or(a, "K", K) != K) return R4D_NO(RAD_E_SHAPE);
    if (M > kRgMaxM) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(wr->data, 16) ||
        (wg && !r4d_aligned(wg->data, 16))) return R4D_NO(RAD_E_ALIGN);
    if (r4d_numel(wr) < n_expert * K || (wg && r4d_numel(wg) < K)) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(lg) || r4d_numel(lg) < M * n_expert) return R4D_NO(RAD_E_STRIDE);
    if (gate && (!r4d_contig(gate) || r4d_numel(gate) < M)) return R4D_NO(RAD_E_STRIDE);

    const long long T = M * top_k;
    if (T > (long long)kScatterSmall) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(ids) || !r4d_contig(w) || !r4d_contig(sorted) || !r4d_contig(off) ||
        !r4d_contig(cnt)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(ids) < T || r4d_numel(w) < T) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(sorted) < T) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(off) < n_local + 1 || r4d_numel(cnt) < n_local) return R4D_NO(RAD_E_SHAPE);

    /* The unfused launches' splits: gemm_bf16_nt_m16/_m64 ask cfg_for with ceil(N/16) tiles, so
     * the router asks with its width and the gate with one tile. */
    const long long ntiles = (n_expert + 15) / 16;
    Cfg cr{}, cg{};
    int rc = cfg_for(a, K, M, 16, BR_FLAT, &cr, ntiles);
    if (rc != RAD_OK) return rc;
    if (wg) {
        rc = cfg_for(a, K, M, 16, BR_FLAT, &cg, 1);
        if (rc != RAD_OK) return rc;
    }
    /* The instance's counter (see the arrival-counter note). Without init there is none and the
     * op is refused. */
    const RgInst* const inst = (const RgInst*)a->instance;
    if (!inst || !inst->ctr) return R4D_NO(RAD_E_STATE);
    unsigned int* const ctr = inst->ctr;

    /* The wave count follows router_topk_scatter's rule (a wave per row, up to 32); the GEMM takes
     * as many 16-column tiles per block as fill those waves at its split. */
    const uint32_t nw = M <= 8 ? 8u : M <= 16 ? 16u : 32u;
    long long tpb = (long long)nw / cr.sk;
    if (tpb < 1) tpb = 1;
    if (tpb > ntiles) tpb = ntiles;
    const long long nrb = (ntiles + tpb - 1) / tpb;
    size_t lds = (size_t)(tpb * cr.sk) * 256 * sizeof(float);
    if (wg) lds = std::max(lds, (size_t)cg.sk * 256 * sizeof(float));
    const size_t sort_lds = nw == 8 ? sizeof(ScatterLds<8>)
                          : nw == 16 ? sizeof(ScatterLds<16>) : sizeof(ScatterLds<32>);
    lds = std::max(lds, sort_lds);

    RgGeom g{};
    g.M = (uint32_t)M;               g.K = (uint32_t)K;
    g.n_expert = (uint32_t)n_expert; g.n_local = (uint32_t)n_local;
    g.top_k = (uint32_t)top_k;       g.lg_ld = (uint32_t)n_expert;
    g.norm = (int32_t)norm;          g.base = (int32_t)base;
    g.sk = (uint32_t)cr.sk;          g.tpb = (uint32_t)tpb;
    g.nrb = (uint32_t)nrb;           g.skg = wg ? (uint32_t)cg.sk : 0u;
    const dim3 grid((unsigned)(nrb + (wg ? 1 : 0)), (unsigned)((M + 15) / 16));

#define R4D_RG_E(OWN, NWV, CH_)                                                                 \
    emit_rc = R4D_EMIT(stream, (router_gemm_topk_scatter_kernel<OWN, NWV, CH_>), grid,          \
                       dim3((NWV) * kWave), lds,                                                \
                       (const uint16_t*)x->data, (const uint16_t*)wr->data,                     \
                       wg ? (const uint16_t*)wg->data : (const uint16_t*)nullptr,               \
                       (uint16_t*)lg->data, gate ? (uint16_t*)gate->data : (uint16_t*)nullptr,  \
                       (int32_t*)ids->data, (uint16_t*)w->data, (int32_t*)sorted->data,         \
                       (int32_t*)off->data, (int32_t*)cnt->data, ctr, g, prot)
#define R4D_RG_L(OWN, NWV)                                                                      \
    do { if (chain) R4D_RG_E(OWN, NWV, r4dchain); else R4D_RG_E(OWN, NWV, R4dUnchained); } while (0)
    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    if (n_expert <= (long long)(8 * kWave)) {
        if      (nw == 8)  R4D_RG_L(8, 8);
        else if (nw == 16) R4D_RG_L(8, 16);
        else               R4D_RG_L(8, 32);
    } else {
        if      (nw == 8)  R4D_RG_L(16, 8);
        else if (nw == 16) R4D_RG_L(16, 16);
        else               R4D_RG_L(16, 32);
    }
#undef R4D_RG_L
#undef R4D_RG_E
    if (emit_rc != RAD_OK) return emit_rc;
    R4D_MOE_CHK();
    return RAD_OK;
}
