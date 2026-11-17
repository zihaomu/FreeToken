// Block-scaled FP8 weight, fp8 activation. Two entry points, two kernels with one operand mapping:
//
//   r4d_gemm_fp8a8_nt_m16   M <= 64 (decode and speculative verify). A wave's 16 weight rows
//                           against up to four 16-token tiles, split-K over the caller's scratch.
//   r4d_gemm_fp8a8_tiled    any M (prefill). 128x128 tile, eight waves, each 2 weight tiles x 4
//                           activation tiles; the weight from registers, the activation from LDS.
//
// Reads block-scaled FP8 checkpoints as shipped (DeepSeek-style, e.g. Qwen3.8-27B-FP8):
// `<linear>.weight` is F8_E4M3 [N][K] and `<linear>.weight_scale_inv` is a BF16 scale per
// 128x128 block, [ceil(N/128)][ceil(K/128)]. No repacking and no scale conversion, so the
// checkpoint never has to be dequantised to bf16 (2x the weight bytes).
//
// Properties of the K loop:
//   1. The bf16 scale decodes as `__uint_as_float(s << 16)` (r4d_fp8_wmma.h).
//   2. The weight is the WMMA A operand on both tiles, so D is C^T of `C[M,N] = A[M,K] @ W[N,K]^T`
//      and a lane's eight accumulator slots are eight weight rows of one activation row.
//   3. Split-K uses the caller's scratch (nothing on a launch path allocates), sized by
//      r4d_gemm_fp8a8_nt_m16_scratch(). The summation order is fixed.
//   4. C is bf16, like every GEMM in this library.
//
// Load-bearing: the padded LDS row pitch (r4d_fp8_wmma.h `lds_pitch`), the one-K-tile-ahead
// register prefetch, the two accumulator sets, the staging assignment (`kTPR`) and, on the prefill
// tile, keeping the weight out of LDS. Reasons are at each.
//
// Entry points return a negative code rather than throwing: they are reachable from a kernel
// plugin, and an exception must not cross the .so boundary.

#include "r4d.h"
#include "r4d_keyed.h"
#include "r4d_fp8_args.h"
#include "r4d_emit.h"
#include "r4d_chain.h"
#include "r4d_fp8_wmma.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <type_traits>

namespace {

using namespace r4d_fp8;

constexpr uint32_t kWave = 32;

// The scale block on both axes. The checkpoint's scale grid, the activation quantiser's group
// (r4d_quant_act_fp8) and this fold must agree, so it is one constant, and not a build flag: it is
// a property of the checkpoint.
constexpr uint32_t kBlock = 128;

// Narrow-tile K width: one gfx1201 cacheline, 256 B. `kTPR` is FK/32, so this puts eight threads
// on a row, which makes the weight stream coalesce (64 B would put two and cost a third of the
// bandwidth; 128 B is also full speed; 512 B needs 64 VGPRs of prefetch and does not build).
constexpr uint32_t kFKNarrow = 256;

// ---- the prefill tile: its geometry and body (shared with the IQR tile) ------------------------
#include "r4d_gemm_8a8_prefill.h"

// Blocks the split aims for. A fixed tuning default, not derived from the CU count: `ksplit` sets
// the order K partials are summed in, so deriving it would make output depend on the card.
//
// `gx * mw` is N/16, so the rule depends on N alone. A target of 512 leaves the attention
// projections (4096-5120 rows) at ~60% of their split speed; 256 leaves the 2048-3072-row
// projections 3-12% short. Above 8192 rows splitting costs, and the guard below stops it there.
// 128 holds across the whole M band: the split's cost (an M*N f32 partial plane) grows with M while
// its benefit does not, so a wider band wants a lower target, and 128 is already below every trunk
// grid. Do not re-sweep it.
constexpr uint32_t kSplitTarget = 128u;
/* A lower target at M > 8 would split N = 512 (the attention k/v projections) three ways instead
 * of four and save ~0.1 ms per step in the engine, but a split repartitions K, which changes the
 * emitted text at M > 8 (short prefill chunks). Not worth the output change.
 *
 * Any such rule would also have to fix the scratch query: it is asked once at declare with the
 * band's upper bound and is safe only because `ksplit * M * N` falls with M. A larger split at
 * M <= 8 can exceed the bound's answer (e.g. 3*9 < 4*8), and the launch would silently fall back
 * to ksplit 1. */
// Split cap when the output is thin (`M*N < 8192`), where the reduce could be thinner than the GEMM
// the split widened. The reduce keeps eight loads in flight, which lets the cap sit this high.
constexpr uint32_t kSplitThinCap = 64u;

// ---- fragment-order weight ---------------------------------------------------------------------
//
// A 16x16 fragment from a row-major [N][K] plane is 8 bytes every `w_ld`, so the weight must go
// global -> LDS -> fragment (a staging pass, a barrier per K tile, 36 KiB of LDS). In fragment
// order (r4d_fp8_frag.h, applied by the layout hook at convert) the bytes are contiguous and none
// of that is needed. The order is a container property, so the prefill tile and matvec serve it
// too. The win comes from the request width: 8 bytes per lane (a 256 B request) loses to staging;
// 32 bytes per lane (1 KiB per wave) wins everywhere, by ~13% on gate_up's decode shape.
// ---- the body ----------------------------------------------------------------------------------
//
// fp8 x fp8 -> f32 with no conversion in the loop: both operands are E4M3 bytes, staged as raw
// bytes (no 16-bit WMMA, no in-loop v_cvt).
//
// Both operands carry a scale per 128 along K, constant together over a 128-column block, so they
// are applied once per block: `raw` accumulates 128 columns unscaled and then folds into `out`
// (64 extra VGPRs, no per-element arithmetic in the inner loop).
//
// ---- the weight is the A operand ---------------------------------------------------------------
//
// On both tiles: a lane's A fragment bytes are its 32 bytes of the fragment-order plane, and the
// activation is B. The scales differ in shape (weight: one per 128 rows; activation: one per token),
// and the accumulator is column-distributed (a lane owns one B row and eight A rows), so with the
// weight on A a lane's eight accumulator slots share one activation scale, and a wave whose weight
// rows sit in one 128-row scale row folds with one weight scale. D is [N][M]: the narrow tile's
// epilogue transposes on the way out (sixteen lanes write addresses `N` apart), which at M <= 64 is
// negligible against the weight read; the prefill tile stores C row by row, eight consecutive
// weight rows of one activation row as one 16-byte store.
// ---- GATED: SwiGLU and the quantiser folded into the epilogue ---------------------------------
//
// Replaces `gated_quant_fp8` re-reading this kernel's [M, 2n] bf16 output and writing [M, n] E4M3
// plus a scale per (token, 128 columns). It fits the narrow tile (and cannot fit the prefill tile;
// see the 8-accumulator cap): a block owns FM = 256 weight rows, 16 per wave. The first half of the
// waves takes gate rows `[c0, c0+128)`, the second half up rows `[n + c0, n + c0 + 128)`, so:
//
//   * a block emits exactly 128 output columns, one scale group, and the amax closes in the block;
//   * waves w and w + MW/2 hold the same 16 output columns for the same tokens in the same lane
//     (the accumulator mapping does not depend on the weight rows), so the gate stash is a
//     straight per-lane copy through LDS with no bank conflict;
//   * `acc_row(lane, t) = (lane >> 4) * 8 + t` is eight consecutive output columns for one token,
//     so the E4M3 store is one 8-byte access;
//   * block count and total WMMA are unchanged.
//
// The stash reuses `s_b`, dead after the K loop (16 KiB at NT = 4 of the 16.5 KiB it holds); only
// the 2 KiB amax plane is new.
//
// Byte-identical to the unfused pair: the accumulator is rounded to bf16 first and silu and the
// multiply run on the round-tripped values. The amax is an fmaxf reduction, exact and order-free.
//
// The narrow path's weight loads carry no nontemporal hint: the activation every block re-reads
// was just written by the previous kernel and is resident, so protecting it does not move a step.

/* I8 is the same kernel at int8 codes (gemm_i8a8_*): i8 operands in the same staging and fragment
 * order, v_wmma_i32_16x16x16_iu8, and a weight scale per row per 128 K (bf16, tile-major
 * [N/16][K/128][16], r4d_fp8_layout.cpp) instead of per 128x128 tile. Only the fold differs: one
 * weight scale per accumulator row, eight a lane, read as one 16-byte load. */
/* Waves per SIMD for each narrow tile shape (R4D_WAVES): pins the register budget each shape was
 * tuned at (the NT accumulators and K-step unroll are sized against it), so a compiler release
 * cannot change occupancy. Shapes with one or two waves per block are LDS-bound and use that
 * count. */
constexpr uint32_t f8_waves(bool full, uint32_t nt, uint32_t mw, uint32_t fk, bool i8,
                            bool gated = false) {
    if (mw == 1u) return nt == 1u ? 8u : nt == 2u ? 4u : nt == 3u ? 3u : 2u;
    if (mw == 2u) return nt == 1u ? 12u : nt == 2u ? 8u : nt == 3u ? 5u : 4u;
    if (!i8) {
        if (fk == 128u)
            return nt == 1u ? 16u : nt == 2u ? (full ? 16u : 12u) : nt == 3u ? 10u
                 : (full && mw >= 8u) ? 10u : 9u;
        /* The gated three-tile form pins 8 waves, not 9. At 9 waves (168 registers) its epilogue
         * spills 57 registers at the full tile (48 rows) and 10 below it, 70 with the LDS-staged
         * scales; the full-tile spill makes gate_up 51% slower than at 56 rows. At 8 waves (192
         * registers) it spills 9 and 10. The gated form is always MW 16, four waves a SIMD a
         * block, so 8 and 9 waves both hold two blocks a WGP and no occupancy is lost. */
        return nt == 1u ? 12u : nt == 2u ? 10u : nt == 3u ? (gated ? 8u : 9u)
             : mw == 4u ? 7u : 8u;
    }
    if (fk == 128u)
        return nt == 1u ? 16u : nt == 2u ? (full ? 10u : 12u) : nt == 3u ? (full ? 9u : 10u)
             : full ? 7u : 9u;
    /* 10, not 12. The fold's LDS-staged scales (gemm_fp8a8_kernel) need a few more
     * registers than the 120 that 12 waves leave, and spill there. gfx12 allocates registers 24 at a time, so
     * 11 waves leave 120 as well and 10 is the next step (144). No decode shape puts more than 8 of
     * these waves on a SIMD, so the lower pin costs no occupancy. */
    if (nt == 1u) return 10u;
    if (nt == 2u) return full ? 9u : mw == 4u ? 9u : 10u;
    if (nt == 3u) return full ? (mw == 16u ? 8u : 7u) : (mw == 4u ? 8u : 9u);
    return full ? (mw == 4u ? 5u : 6u) : (mw == 4u ? 7u : 8u);
}

// ---- BIAS: gemm_nt_q_bias's epilogue on the narrow tile ---------------------------------------
//
// With BIAS (never split) the store adds the bf16 `Bias` of each weight row to the f32 accumulator
// before its one rounding, libref's order, and writes C at row pitch `c_ld`, so C may be a column
// slice of a wider buffer (a modulation table's block of columns).
//
// The narrow tile: MW waves a block, a wave one 16-row weight tile (A, fragments straight from the
// plane into registers) against NT 16-token tiles (B, staged through LDS).
/* The body. gemm_fp8a8_kernel and its chained form r4dchain_gemm_fp8a8 (r4d_chain.h) wrap it
 * after it: its blocks return early (an empty K slice, the finisher's losers), and every block of
 * a chained launch must still count itself in. */
template <bool Full, uint32_t NT, uint32_t MW, uint32_t FK,
          bool GATED = false, bool I8 = false, bool BIAS = false, uint32_t SB = kBlock>
__device__ __forceinline__ void gemm_fp8a8_body(
    const uint8_t* __restrict__ Ap, const void* __restrict__ Sa,
    const uint8_t* __restrict__ Bp, const void* __restrict__ Sb,
    void* __restrict__ Y, void* __restrict__ C, unsigned int* __restrict__ Ctr,
    uint32_t rows, uint32_t cols, uint32_t nb,
    uint32_t a_ld, uint32_t b_ld, uint32_t sa_ld, uint32_t sb_ld, uint32_t ck,
    uint8_t* __restrict__ Q = nullptr, float* __restrict__ Sc = nullptr,
    uint16_t* __restrict__ Yb = nullptr,
    uint32_t q_ld = 0, uint32_t s_ld = 0, uint32_t y_ld = 0,
    const uint16_t* __restrict__ Bias = nullptr, uint32_t c_ld = 0) {
    constexpr uint32_t FN = NT * 16;              // B rows a block
    constexpr uint32_t FM = MW * 16;              // A rows a block
    // The weight scale is folded as one scalar per wave, indexed at `(m0 + wm) / kBlock`, so a
    // wave's 16 rows (not the block's FM) must sit in one 128-row scale row; m0 is FM-aligned and FM
    // a multiple of 16, so they never straddle one. Requiring it of the block would cap MW at 8.

    // One buffer: two (74 KiB) would be over the 64 KiB limit, and decode waits on HBM anyway, so
    // the two-barrier structure is kept by the barrier at the loop foot. No LDS for A: a lane loads
    // its fragment bytes straight from global, so the tile needs 4 KiB of LDS instead of 38.
    __shared__ uint8_t s_b[FN * lds_pitch<FK>()];

    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    // Split-K. `ck` is a multiple of both 128 (the scale block) and FK, and the launcher derives
    // ksplit from it, so every slice has work and starts on a scale boundary.
    const uint32_t kbeg = ck ? blockIdx.z * ck : 0u;
    const uint32_t kend = ck ? min(cols, kbeg + ck) : cols;
    if (kbeg >= kend) return;

    const uint32_t m0 = blockIdx.x * FM;
    const uint32_t n0 = blockIdx.y * FN;
    // Wave grid MW(M) x 1(N): a wave owns 16 A rows and all FN B rows.
    const uint32_t wm = wave * 16u, wn = 0;
    /* GATED changes which weight rows a wave owns before the epilogue: the gate and up halves are
     * `rows/2` apart, so later uses of `m0 + wm` (fragment tile, liveness, A-side scale) go through
     * wrow0. A wave still owns 16 consecutive rows, so they stay in one scale row. */
    constexpr uint32_t kGHalf = GATED ? (MW / 2u) : MW;
    const uint32_t wrow0 =
        GATED ? ((wave < kGHalf ? 0u : rows / 2u) + blockIdx.x * (FM / 2u)
                     + (wave % kGHalf) * 16u)
              : (m0 + wm);
    static_assert(!GATED || (MW % 2u) == 0u, "the gated fold splits the waves in half");
    static_assert(!(GATED && I8), "the gated epilogue quantises to E4M3 and has no int8 form");
    static_assert(!BIAS || (!GATED && !I8), "the bias epilogue is the fp8 narrow tile's");
    /* SB: the K extent of one scale on both sides. 128, or 256 for an int8 weight at IQ8R's group
     * (R4D_I8_GROUP_WIDE), whose int32 sums then run over two rotation blocks before one fold
     * (|sum| <= 256 * 127 * 127 < 2^24, exact in the f32 conversion). The staged tile must not
     * be wider than the group's two halves meet in: FK 256 with SB 256 folds once a tile. */
    static_assert(SB == kBlock || (I8 && !GATED && SB == 256u && FK == 256u),
                  "a 256-wide scale group is the int8 tile's at the 256-byte K step");

    f8v raw[NT], out[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) { raw[j] = acc_zero(); out[j] = acc_zero(); }

    // ---- staging assignment (a bandwidth decision) ------------------------------------------------
    //
    // A thread stages one 32-byte granule pair, and the threads sharing a row are consecutive, so
    // `kTPR` threads cover a row and one wave load asks for `32 * 32 / kTPR` rows of `kTPR * 32`
    // contiguous bytes. Streaming 5120-byte rows on gfx1201:
    //
    //     threads per row    2         4         8        16        32
    //     GB/s             422       626       633       635       636      (plain stream: 638)
    //
    // Two threads per row leaves a third of the bandwidth unused; four recover it. kTPR = FK/32 is 8
    // on the 256-byte K tile (fetch-bound). A pass covers blockDim/kTPR rows.
    constexpr uint32_t kSeg = 32;                       // bytes a thread stages, contiguous
    constexpr uint32_t kTPR = FK / kSeg;                // threads covering one row
    constexpr uint32_t kRPP = (MW * kWave) / kTPR;      // rows one pass covers
    // Fragment-order supertiles per staged K tile: one 32-byte load per lane, four fragments each
    // (see r4d_fp8_frag.h).
    constexpr uint32_t kAF  = FK / r4d_fp8::kSuperK;
    constexpr uint32_t kBP  = (FN + kRPP - 1) / kRPP;   // passes over the B tile
    static_assert(kTPR >= 1 && kRPP >= 1, "FK must be at least 32 B and no wider than the block");

    const uint32_t sn = tid / kTPR, sk = (tid % kTPR) * kSeg;

    const uint8_t* bsrc = Bp + (size_t)(n0 + sn) * b_ld + sk;
    // A as fragments: a wave owns one 16-row weight tile, so its A operand for 64 k is this lane's
    // 32 bytes. `ksuper` is the plane's tile stride; `arow` is out of range only when N is not a
    // multiple of FM.
    const uint32_t ksuper = cols / r4d_fp8::kSuperK;
    const uint32_t arow   = wrow0 / 16u;
    const bool     alive  = wrow0 < rows;
    const uint8_t* afrag  = Ap + (size_t)lane * r4d_fp8::kLaneBytes;

    // One K tile ahead in registers: the load for tile c0+FK is issued before tile c0's WMMA and
    // consumed after it, hiding memory latency under arithmetic. `sk` is the same column for every
    // pass, so only row existence varies per pass.
    const bool k_live = Full || (kbeg + sk + kSeg <= cols);
    /* `pf` is read straight from registers by the mma loop, so its prefetch goes to `pfn` and the
     * loop foot rotates them; prefetching into `pf` would overwrite the current tile (a bug hidden
     * whenever split-K gives a block only one K iteration). */
    Stage8 pb[kBP];
    Stage8 pf[kAF], pfn[kAF];
#pragma unroll
    for (uint32_t p = 0; p < kBP; ++p) {
        const uint32_t r = sn + p * kRPP;
        pb[p] = stage8_load(bsrc + (size_t)p * kRPP * b_ld + kbeg,
                            k_live && r < FN && (Full || (n0 + r) < nb));
    }
#pragma unroll
    for (uint32_t p = 0; p < kAF; ++p) {
        const uint32_t st = kbeg / r4d_fp8::kSuperK + p;
        pf[p] = stage8_load(
            afrag + (size_t)(arow * ksuper + st) * r4d_fp8::kTileBytes, alive && st < ksuper);
    }

    /* The fold's scales are staged through LDS one K tile ahead, beside the B tile. A fold that
     * loaded its own scales after the next tile's prefetch was issued would wait for that whole
     * prefetch (loads return in order), twice a tile at FK = 256. Instead each thread loads one
     * weight-scale dword and one activation scale of tile c0+FK with its B prefetch (a few each
     * only in the selftest's one- and two-wave pins) and stores them with the B tile at the top of
     * the next pass, so a fold reads LDS. That costs two registers a thread; holding a tile's
     * scales per lane (8 + 2*NT registers) spills at the decode shapes' register pins. The staged
     * values are the ones a direct load reads, so results are bit-identical. A dead wave's fp8
     * block scale reads as 0; its rows are never stored.
     *
     * A tile holds kFolds = FK / kBlock scale blocks, and the fold at `ks` uses block ks / kBlock
     * of its tile. s_ws: I8, each wave's 16 bf16 row scales a block, [MW][kFolds][8] dwords
     * (r4d_fp8_layout.cpp's tile-major plane); fp8, each wave's one bf16 block scale,
     * [MW][kFolds]. s_as: each B row's f32 scale a block, [kFolds][FN]. */
    constexpr uint32_t kFolds = (FK + SB - 1) / SB;
    constexpr uint32_t kWS = I8 ? MW * kFolds * 8u : MW * kFolds;
    constexpr uint32_t kAS = kFolds * FN;
    constexpr uint32_t kWSPer = (kWS + MW * kWave - 1) / (MW * kWave);
    constexpr uint32_t kASPer = (kAS + MW * kWave - 1) / (MW * kWave);
    __shared__ __attribute__((aligned(16))) uint32_t s_ws[kWS];
    __shared__ float s_as[kAS];
    const uint32_t nkb8 = (cols + SB - 1) / SB;
    auto load_ws = [&](uint32_t c, uint32_t i) -> uint32_t {
        if (i >= kWS) return 0u;
        if constexpr (I8) {
            const uint32_t tr = (m0 + (i / (kFolds * 8u)) * 16u) / 16u;
            const uint32_t kb = c / SB + (i / 8u) % kFolds;
            return ((Full || tr * 16u < rows) && kb < nkb8)
                ? ((const uint32_t*)Sa)[((size_t)tr * nkb8 + kb) * 8u + i % 8u] : 0u;
        } else {
            const uint32_t w = i / kFolds, kb = c / kBlock + i % kFolds;
            const uint32_t wr = GATED ? ((w < kGHalf ? 0u : rows / 2u) + blockIdx.x * (FM / 2u)
                                         + (w % kGHalf) * 16u)
                                      : (m0 + w * 16u);
            return ((Full || wr < rows) && kb < nkb8)
                ? (uint32_t)((const uint16_t*)Sa)[(size_t)(wr / kBlock) * sa_ld + kb] : 0u;
        }
    };
    auto load_as = [&](uint32_t c, uint32_t i) -> float {
        if (i >= kAS) return 0.0f;
        const uint32_t tok = n0 + i % FN, kb = c / SB + i / FN;
        return ((Full || tok < nb) && kb < nkb8) ? ((const float*)Sb)[(size_t)tok * sb_ld + kb]
                                                 : 0.0f;
    };
    uint32_t qws[kWSPer];
    float qas[kASPer];
    auto load_q = [&](uint32_t c) {
#pragma unroll
        for (uint32_t r = 0; r < kWSPer; ++r) qws[r] = load_ws(c, tid + r * MW * kWave);
#pragma unroll
        for (uint32_t r = 0; r < kASPer; ++r) qas[r] = load_as(c, tid + r * MW * kWave);
    };
    load_q(kbeg);
    // The B rows this lane owns within the tile, one per N tile (column `frag_c` of each of its NT
    // accumulators, 16 apart). A per-B-row scale must be looked up for each, not just the first.
    const uint32_t bl0 = wn + frag_c(lane);

    for (uint32_t c0 = kbeg; c0 < kend; c0 += FK) {
#pragma unroll
        for (uint32_t p = 0; p < kBP; ++p) {
            const uint32_t r = sn + p * kRPP;
            if (r < FN) stage8_store<FK>(s_b, r, sk, pb[p]);
        }
#pragma unroll
        for (uint32_t r = 0; r < kWSPer; ++r)
            if (tid + r * MW * kWave < kWS) s_ws[tid + r * MW * kWave] = qws[r];
#pragma unroll
        for (uint32_t r = 0; r < kASPer; ++r)
            if (tid + r * MW * kWave < kAS) s_as[tid + r * MW * kWave] = qas[r];
        __syncthreads();
        const uint32_t c1 = c0 + FK;
        if (c1 < kend) {
            const bool live = Full || (c1 + sk + kSeg <= cols);
#pragma unroll
            for (uint32_t p = 0; p < kBP; ++p) {
                const uint32_t r = sn + p * kRPP;
                if (r < FN)
                    pb[p] = stage8_load(bsrc + (size_t)p * kRPP * b_ld + c1,
                                        live && (Full || (n0 + r) < nb));
            }
#pragma unroll
            for (uint32_t p = 0; p < kAF; ++p) {
                const uint32_t st = c1 / r4d_fp8::kSuperK + p;
                pfn[p] = stage8_load(
                    afrag + (size_t)(arow * ksuper + st) * r4d_fp8::kTileBytes,
                    alive && st < ksuper);
            }
            load_q(c1);
        }

#pragma unroll
        for (uint32_t ks = 0; ks < FK; ks += kK) {
            // A comes from registers: the 32 bytes this lane loaded for the 64-k supertile are its
            // four fragments (+0, +8, +16, +24). `ks` is constant after unrolling, so the selection
            // folds away.
            const i2v af = frag_of(pf[ks / r4d_fp8::kSuperK], (ks % r4d_fp8::kSuperK) / 16u);
            i2v bf[NT];
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) bf[j] = frag<FK>(s_b, wn + j * 16, ks, lane);
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) raw[j] = mma8<I8>(af, bf[j], raw[j]);

            // End of a 128-column scale block: fold both scales in and reset. Inside the ks loop
            // because at FK = 256 one staged tile spans two scale blocks.
            //
            // `c0` is a runtime value, so the compiler if-converts this test at every unrolled ks.
            // Gating it at the compile-time-known positions is legal and byte-identical but raises
            // the register live range of the fold; see libr4d/isa/README.md.
            if ((c0 + ks + kK) % SB == 0 || c0 + ks + kK >= kend) {
                /* This tile's scale block for `ks`, staged with the tile. */
                const uint32_t f = ks / SB < kFolds ? ks / SB : kFolds - 1u;
                if constexpr (I8) {
                    // A is the int8 weight with a bf16 scale per row. A lane's eight accumulator
                    // slots are eight consecutive rows of one 16-row tile, adjacent in the
                    // tile-major [N/16][K/128][16] plane: one 16-byte load. B is the activation.
                    float wsr[8];
                    {
                        const uint4 w8 =
                            *(const uint4*)&s_ws[(wave * kFolds + f) * 8u + (lane >> 4) * 4u];
                        const uint32_t d[4] = { w8.x, w8.y, w8.z, w8.w };
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t)
                            wsr[t] = bscale((uint16_t)(d[t / 2] >> (16 * (t % 2))));
                    }
#pragma unroll
                    for (uint32_t j = 0; j < NT; ++j) {
                        const float s = s_as[f * FN + bl0 + j * 16u];
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t) {
                            out[j][t] += acc_val<I8>(raw[j][t]) * (wsr[t] * s);
                            raw[j][t] = 0.0f;
                        }
                    }
                } else {
                    // A is the weight: one bf16 scale for this wave's 16 rows, which sit in one
                    // scale row. B is the activation: one f32 scale per (token, 128-column block),
                    // one per lane per tile.
                    const float ws = bscale((uint16_t)s_ws[wave * kFolds + f]);
#pragma unroll
                    for (uint32_t j = 0; j < NT; ++j) {
                        const float s = ws * s_as[f * FN + bl0 + j * 16u];
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t) {
                            out[j][t] += raw[j][t] * s;
                            raw[j][t] = 0.0f;
                        }
                    }
                }
            }
        }
        // Wait until every wave has finished reading before the tile and its scales are
        // overwritten. The prefetched fragments become current.
#pragma unroll
        for (uint32_t p = 0; p < kAF; ++p) pf[p] = pfn[p];
        __syncthreads();
    }

    if constexpr (GATED) {
        constexpr uint32_t kStash = kGHalf * kWave * NT * 8u;          // bf16 gate values
        static_assert(kStash * 2u <= FN * lds_pitch<FK>(),
                      "the gate stash must fit in the staging buffer it reuses");
        uint16_t* const gst = (uint16_t*)&s_b[0];
        __shared__ float s_amx[kGHalf][FN];
        const uint32_t half = wave / kGHalf, w2 = wave % kGHalf;
        const uint32_t c = frag_c(lane);

        // s_b is still being read by the last K tile's fragments when this is reached.
        __syncthreads();
        if (half == 0u) {
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                if (!Full && n0 + wn + j * 16u >= nb) continue;
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t)
                    gst[((w2 * kWave + lane) * NT + j) * 8u + t] = f32_to_bf16(out[j][t]);
            }
        }
        __syncthreads();

        // Same arithmetic and order as r4d_fused_quant_fp8.hip's gated_quant_fp8_kernel: gate and up
        // both round-tripped through bf16, silu in f32, product rounded to bf16, then quantised.
        float v[NT][8];
        if (half != 0u) {
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                /* Skip a token tile wholly past the end (wave-uniform test); the per-lane guard in
                 * the store is divergent and saves nothing. */
                if (!Full && n0 + wn + j * 16u >= nb) continue;
                float m = 0.0f;
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    const float g = bf16_to_f32(gst[((w2 * kWave + lane) * NT + j) * 8u + t]);
                    const float a = g / (1.0f + expf(-g));
                    const uint16_t h =
                        f32_to_bf16(a * bf16_to_f32(f32_to_bf16(out[j][t])));
                    v[j][t] = bf16_to_f32(h);
                    m = fmaxf(m, fabsf(v[j][t]));
                }
                // A wave's 16 output columns for one token are in lanes `c` and `c ^ 16`; the other
                // 112 columns of the group are in the other waves. fmaxf is exact and order-free.
                m = fmaxf(m, r4d_xor_lane<16>(m));
                if (lane < 16u) s_amx[w2][j * 16u + lane] = m;
            }
        }
        __syncthreads();
        if (half != 0u) {

            const uint32_t col0 = blockIdx.x * (FM / 2u) + w2 * 16u + (lane >> 4) * 8u;
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                const uint32_t tok = n0 + wn + j * 16u + c;
                if (!Full && tok >= nb) continue;
                float mm = 0.0f;
#pragma unroll
                for (uint32_t u = 0; u < kGHalf; ++u) mm = fmaxf(mm, s_amx[u][j * 16u + c]);
                const float sc = mm > 0.0f ? mm * (1.0f / 448.0f) : 1.0f;
                if (w2 == 0u && (lane >> 4) == 0u) Sc[(size_t)tok * s_ld + blockIdx.x] = sc;
                const float inv = 1.0f / sc;
                // The clamp is required (see r4d_q8.h): the builtin returns NaN out of range, and
                // `amax * (1/scale)` can land an ulp above 448.
                float qc[8];
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t)
                    qc[t] = fminf(fmaxf(v[j][t] * inv, -448.0f), 448.0f);
                int pk[2] = { 0, 0 };
#pragma unroll
                for (uint32_t d = 0; d < 2; ++d) {
                    pk[d] = __builtin_amdgcn_cvt_pk_fp8_f32(qc[d * 4 + 0], qc[d * 4 + 1], pk[d],
                                                            false);
                    pk[d] = __builtin_amdgcn_cvt_pk_fp8_f32(qc[d * 4 + 2], qc[d * 4 + 3], pk[d],
                                                            true);
                }
                // Eight consecutive columns: one 8-byte store, aligned since `q_ld` is a multiple of
                // 128 and col0 of 8.
                *(uint2*)(Q + (size_t)tok * q_ld + col0) =
                    make_uint2((uint32_t)pk[0], (uint32_t)pk[1]);
                if (Yb) {
                    /* Packed in whole registers (bf16x2_rne, f32_to_bf16's bits); a uint16_t[8]
                     * would be assembled from register halves. */
                    uint16_t* const yr = Yb + (size_t)tok * y_ld + col0;
                    *(uint4*)yr = make_uint4(bf16x2_rne(v[j][0], v[j][1]), bf16x2_rne(v[j][2], v[j][3]),
                                             bf16x2_rne(v[j][4], v[j][5]), bf16x2_rne(v[j][6], v[j][7]));
                }

            }
        }
        return;
    }

    // Split-K writes an f32 partial plane; without it the store is the final bf16 C.
    float* const part = ck ? (float*)Y + (size_t)blockIdx.z * rows * nb : nullptr;
    uint16_t* const c16 = (uint16_t*)Y;

    const uint32_t c = frag_c(lane);
    /* ---- eight consecutive elements in one store ---------------------------------------------
     *
     * `acc_row(lane, t)` is `(lane>>4)*8 + t`, so with the output index `col * rows + r` a lane's
     * eight slots are eight consecutive addresses in one 16-byte piece; one store instead of eight
     * removes the epilogue's M-proportional cost. Bit-identical. Requires `rows` % 8 == 0 for
     * 16-byte alignment (`r` is already a multiple of 8). */
    const bool vec8 = BIAS ? (c_ld & 7u) == 0u && ((uintptr_t)Y & 15u) == 0u : (rows & 7u) == 0u;
    /* C's row pitch: N, or the caller's with BIAS. The split-K partial plane (never with BIAS)
     * stays dense at N. */
    const uint32_t cld = BIAS ? c_ld : rows;
    /* The split-K and vec8 tests are hoisted out of the store loop as template arguments;
     * otherwise the compiler keeps a compare and branch at every store and assembles the vec8
     * words from register halves. */
    auto store = [&](auto split, auto v8) {
        constexpr bool CK = decltype(split)::value;
        constexpr bool V8 = decltype(v8)::value;
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            const uint32_t col = n0 + wn + j * 16 + c;
            if (!Full && col >= nb) continue;
            const uint32_t rb = m0 + wm;
            const uint32_t r0 = rb + acc_row(lane, 0);
            /* The bias of each of this lane's eight weight rows (zero past N). */
            f8v ov = out[j];
            if constexpr (BIAS) {
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    const uint32_t r = r0 + t;
                    ov[t] += (Full || r < rows) ? bf16_to_f32(Bias[r]) : 0.0f;
                }
            }
            if (V8 && (Full || r0 + 8u <= rows)) {
                const size_t o = (size_t)col * cld + r0;
                if constexpr (CK) {
                    __attribute__((aligned(16))) float pv[8];
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) pv[t] = ov[t];
                    *(uint4*)(part + o)     = *(const uint4*)pv;
                    *(uint4*)(part + o + 4) = *(const uint4*)(pv + 4);
                } else {
                    /* Packed in whole registers (bf16x2_rne, f32_to_bf16's bits); a
                     * uint16_t[8] would be assembled from register halves. */
                    *(uint4*)(c16 + o) = make_uint4(bf16x2_rne(ov[0], ov[1]), bf16x2_rne(ov[2], ov[3]),
                                                    bf16x2_rne(ov[4], ov[5]), bf16x2_rne(ov[6], ov[7]));
                }
                continue;
            }
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                const uint32_t r = rb + acc_row(lane, t);
                if (!Full && r >= rows) continue;
                // A held the weight, so [r][col] is [N][M] and C is [M][N].
                const size_t o = (size_t)col * cld + r;
                if constexpr (CK) part[o] = ov[t];
                else              c16[o]  = f32_to_bf16(ov[t]);
            }
        }
    };
    if (vec8) {
        if (ck) store(std::true_type{}, std::true_type{});
        else    store(std::false_type{}, std::true_type{});
    } else {
        if (ck) store(std::true_type{}, std::false_type{});
        else    store(std::false_type{}, std::false_type{});
    }

    // ---- split finisher ---------------------------------------------------------------------------
    //
    // Dependent dispatches cost ~2.8 us each on this part (2.3 in a hipGraph), so a separate reduce
    // launch per split GEMM is expensive. Instead the last z block of a column reduces it:
    // `ctr[blockIdx.x]` counts arrivals, and the block that reaches gridDim.z resets it and sums.
    // The z loop runs in fixed ascending order with no atomic on an accumulator, so the result is
    // bit-identical to the separate reduce and deterministic.
    //
    // The counters are a small allocation keyed on (stream, output) (`split_ctr`), not the caller's
    // scratch, which other ops overwrite. The self-reset keeps them zero, which is sound only while
    // launches sharing a counter block never overlap; see split_ctr.
    if (ck && Ctr) {
        // Every thread's stores must be visible device-wide before the arrival is published: a
        // per-thread fence, then the barrier.
        __threadfence();
        __syncthreads();
        __shared__ unsigned int s_arrived;
        if (tid == 0)
            s_arrived = __hip_atomic_fetch_add(&Ctr[blockIdx.x], 1u, __ATOMIC_ACQ_REL,
                                               __HIP_MEMORY_SCOPE_AGENT);
        __syncthreads();
        if (s_arrived + 1u == gridDim.z) {
            if (tid == 0)
                __hip_atomic_store(&Ctr[blockIdx.x], 0u, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
            __threadfence();                       // acquire the other slices' partials
            // This block owns A rows [m0, m0+FM) and all nb output columns, exactly what the stores
            // above wrote.
            const uint32_t rlo = m0, rhi = min(m0 + FM, rows);
            const float* const w0 = (const float*)Y;
            uint16_t* const cy = (uint16_t*)C;
            /* Consecutive threads take consecutive weight rows, which are adjacent in a
             * partial plane ([nb][rows]): four rows a thread as one 16-byte load, kFinU quads in
             * flight at each z. Walking the token index fastest would put consecutive threads
             * `rows` floats apart, a cache line and a round trip per element. Both branches sum
             * each element from 0.0f in ascending z, so they are bit-identical. */
            const uint32_t nr = rhi - rlo;
            const size_t zs = (size_t)rows * nb;
            if ((rows & 3u) == 0u && (nr & 3u) == 0u && ((uintptr_t)C & 7u) == 0u
                && ((uintptr_t)Y & 15u) == 0u) {
                constexpr uint32_t kFinU = 4;
                const uint32_t nq = nr / 4u, total = nq * nb;
                for (uint32_t b = tid; b < total; b += kFinU * MW * kWave) {
                    size_t o[kFinU];
                    float4 acc[kFinU];
#pragma unroll
                    for (uint32_t u = 0; u < kFinU; ++u) {
                        const uint32_t idx = b + u * MW * kWave;
                        o[u] = idx < total ? (size_t)(idx / nq) * rows + rlo + (idx % nq) * 4u : 0u;
                        acc[u] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                    for (uint32_t z = 0; z < gridDim.z; ++z) {
                        float4 v[kFinU];
#pragma unroll
                        for (uint32_t u = 0; u < kFinU; ++u)
                            v[u] = b + u * MW * kWave < total ? *(const float4*)(w0 + z * zs + o[u])
                                                              : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
                        for (uint32_t u = 0; u < kFinU; ++u) {
                            acc[u].x += v[u].x; acc[u].y += v[u].y;
                            acc[u].z += v[u].z; acc[u].w += v[u].w;
                        }
                    }
#pragma unroll
                    for (uint32_t u = 0; u < kFinU; ++u) {
                        if (b + u * MW * kWave >= total) continue;
                        *(uint2*)(cy + o[u]) = make_uint2(bf16x2_rne(acc[u].x, acc[u].y),
                                                          bf16x2_rne(acc[u].z, acc[u].w));
                    }
                }
            } else {
                for (uint32_t idx = tid; idx < nr * nb; idx += MW * kWave) {
                    const size_t o = (size_t)(idx / nr) * rows + rlo + idx % nr;
                    float acc = 0.0f;
                    for (uint32_t z = 0; z < gridDim.z; ++z) acc += w0[z * zs + o];
                    cy[o] = f32_to_bf16(acc);
                }
            }
        }
    }
}

#define R4D_F8K_ARGS Ap, Sa, Bp, Sb, Y, C, Ctr, rows, cols, nb, a_ld, b_ld, sa_ld, sb_ld, ck, Q,  \
    Sc, Yb, q_ld, s_ld, y_ld, Bias, c_ld
template <bool Full, uint32_t NT, uint32_t MW, uint32_t FK,
          bool GATED = false, bool I8 = false, bool BIAS = false, uint32_t SB = kBlock>
__global__ __launch_bounds__(MW * kWave) R4D_WAVES(f8_waves(Full, NT, MW, FK, I8, GATED))
void gemm_fp8a8_kernel(
    const uint8_t* __restrict__ Ap, const void* __restrict__ Sa,
    const uint8_t* __restrict__ Bp, const void* __restrict__ Sb,
    void* __restrict__ Y, void* __restrict__ C, unsigned int* __restrict__ Ctr,
    uint32_t rows, uint32_t cols, uint32_t nb,
    uint32_t a_ld, uint32_t b_ld, uint32_t sa_ld, uint32_t sb_ld, uint32_t ck,
    uint8_t* __restrict__ Q = nullptr, float* __restrict__ Sc = nullptr,
    uint16_t* __restrict__ Yb = nullptr,
    uint32_t q_ld = 0, uint32_t s_ld = 0, uint32_t y_ld = 0,
    const uint16_t* __restrict__ Bias = nullptr, uint32_t c_ld = 0) {
    gemm_fp8a8_body<Full, NT, MW, FK, GATED, I8, BIAS, SB>(R4D_F8K_ARGS);
}
template <bool Full, uint32_t NT, uint32_t MW, uint32_t FK,
          bool GATED = false, bool I8 = false, bool BIAS = false, uint32_t SB = kBlock>
__global__ __launch_bounds__(MW * kWave) R4D_WAVES(f8_waves(Full, NT, MW, FK, I8, GATED))
void r4dchain_gemm_fp8a8(
    const uint8_t* __restrict__ Ap, const void* __restrict__ Sa,
    const uint8_t* __restrict__ Bp, const void* __restrict__ Sb,
    void* __restrict__ Y, void* __restrict__ C, unsigned int* __restrict__ Ctr,
    uint32_t rows, uint32_t cols, uint32_t nb,
    uint32_t a_ld, uint32_t b_ld, uint32_t sa_ld, uint32_t sb_ld, uint32_t ck,
    uint8_t* __restrict__ Q = nullptr, float* __restrict__ Sc = nullptr,
    uint16_t* __restrict__ Yb = nullptr,
    uint32_t q_ld = 0, uint32_t s_ld = 0, uint32_t y_ld = 0,
    const uint16_t* __restrict__ Bias = nullptr, uint32_t c_ld = 0) {
    r4d_chain_wait();
    gemm_fp8a8_body<Full, NT, MW, FK, GATED, I8, BIAS, SB>(R4D_F8K_ARGS);
    r4d_chain_done();
}
#undef R4D_F8K_ARGS

// ---- the K-parallel int8 tile: M <= 16 ---------------------------------------------------------
//
// The narrow tile walks a wave down all of K, and a weight of few rows fills the card only through
// split-K: partial planes in global scratch and a finisher behind arrival counters. That holds an
// int8 GEMM near 18 us at K = 2560 whatever its rows (256 rows: 18.5 us, where the IQR 16-token
// tile takes 5). Here a block is one 16-row weight tile and its KW waves stride K by scale group
// (wave w takes groups w, w + KW, ..), as the IQR tile's do: the weight's fragments and the
// activation's are read straight from global memory, a group's int32 sums fold once with its
// (row, group) and (token, group) scales, and the waves' partials meet in LDS in ascending wave
// order. No scratch, no counters and no block barrier in the K loop. A block multiplies its weight
// tile against NT token tiles of sixteen; token columns past M read token M - 1 and are not stored. The sum order is this tile's own (per wave, then across waves,
// whatever N and M are), so its output is the product within f32 rounding, not the narrow tile's
// bytes.
#define R4D_I8KPAR_PARAMS                                                                      \
    const uint8_t* __restrict__ Ap, const uint16_t* __restrict__ Sa,                           \
    const int8_t* __restrict__ Bp, const float* __restrict__ Sb, uint16_t* __restrict__ C,     \
    uint32_t rows, uint32_t cols, uint32_t M, uint32_t b_ld, uint32_t sb_ld
template <uint32_t KW, uint32_t SB, uint32_t NT>
__device__ __forceinline__ void gemm_i8_kpar_body(R4D_I8KPAR_PARAMS) {
    constexpr uint32_t kUG = SB / kBlock;   /* 128-K units a scale group */
    __shared__ float s_red[KW][NT][8 * kWave];
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1);
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const uint32_t rt = blockIdx.x, ngrp = cols / SB, nunit = cols / kBlock;
    const uint32_t ksuper = cols / r4d_fp8::kSuperK;
    /* Token tile j is tokens 16 j .. 16 j + 15; a column past M reads token M - 1. */
    const uint32_t col = frag_c(lane);
    const int8_t* xrow[NT];
    const float* xsr[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        const uint32_t tok = min(j * 16u + col, M - 1u);
        xrow[j] = Bp + (size_t)tok * b_ld + frag_k0(lane);
        xsr[j] = Sb + (size_t)tok * sb_ld;
    }
    const uint8_t* const arow = Ap + (size_t)rt * ksuper * r4d_fp8::kTileBytes +
                                (size_t)lane * r4d_fp8::kLaneBytes;
    /* Unit u of the tile: the weight's two 64-K supertiles, a token tile's eight fragments. */
    auto aload = [&](uint32_t u, Stage8 (&a)[2]) {
#pragma unroll
        for (uint32_t p = 0; p < 2; ++p)
            a[p] = stage8_load(arow + (size_t)(2u * u + p) * r4d_fp8::kTileBytes, true);
    };
    auto bload = [&](uint32_t j, uint32_t u, i2v (&b)[8]) {
#pragma unroll
        for (uint32_t ks = 0; ks < 8; ++ks) {
            const uint2 v = *(const uint2*)(xrow[j] + u * kBlock + ks * kK);
            b[ks][0] = (int)v.x;
            b[ks][1] = (int)v.y;
        }
    };

    f8v out[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) out[j] = acc_zero();
    /* The weight and the first token tile are asked for a unit ahead of their WMMA steps, each
     * further tile's fragments a tile ahead. */
    Stage8 ac[2], an[2];
    i2v bc[8], bn[8], bq[2][8];
    if (wave < ngrp) {
        aload(wave * kUG, ac);
        bload(0u, wave * kUG, bc);
    }
    for (uint32_t g = wave; g < ngrp; g += KW) {
        f8v raw[NT];   /* int32 sums */
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) raw[j] = acc_zero();
#pragma unroll
        for (uint32_t h = 0; h < kUG; ++h) {
            const uint32_t u = g * kUG + h;
            const uint32_t un = h + 1u < kUG ? u + 1u : (g + KW) * kUG;
            if (un < nunit) {
                aload(un, an);
                bload(0u, un, bn);
            }
#pragma unroll
            for (uint32_t ks = 0; ks < 8; ++ks) bq[0][ks] = bc[ks];
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                if (j + 1u < NT) bload(j + 1u, u, bq[(j + 1u) & 1u]);
#pragma unroll
                for (uint32_t ks = 0; ks < kBlock / kK; ++ks)
                    raw[j] = mma8<true>(frag_of(ac[ks / 4u], ks % 4u), bq[j & 1u][ks], raw[j]);
            }
#pragma unroll
            for (uint32_t p = 0; p < 2; ++p) ac[p] = an[p];
#pragma unroll
            for (uint32_t ks = 0; ks < 8; ++ks) bc[ks] = bn[ks];
        }
        /* The lane's eight rows' bf16 scales of this group are consecutive in the tile-major
         * plane: one 16-byte load. */
        const uint4 q4 = *(const uint4*)(Sa + ((size_t)rt * ngrp + g) * 16u + acc_row(lane, 0));
        const uint32_t qd[4] = { q4.x, q4.y, q4.z, q4.w };
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            const float as = xsr[j][g];
#pragma unroll
            for (uint32_t t = 0; t < 4; ++t) {
                out[j][2 * t] = __builtin_fmaf(acc_val<true>(raw[j][2 * t]),
                                               __uint_as_float(qd[t] << 16) * as, out[j][2 * t]);
                out[j][2 * t + 1] = __builtin_fmaf(acc_val<true>(raw[j][2 * t + 1]),
                                                   __uint_as_float(qd[t] & 0xFFFF0000u) * as,
                                                   out[j][2 * t + 1]);
            }
        }
    }
    if constexpr (KW > 1u) {
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) s_red[wave][j][t * kWave + lane] = out[j][t];
        __syncthreads();
        if (wave != 0u) return;
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                float a = 0.0f;
#pragma unroll
                for (uint32_t w = 0; w < KW; ++w) a += s_red[w][j][t * kWave + lane];
                out[j][t] = a;
            }
    }
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        const uint32_t tok = j * 16u + col;
        if (tok < M)
            *(uint4*)(C + (size_t)tok * rows + rt * 16u + acc_row(lane, 0)) =
                make_uint4(bf16x2_rne(out[j][0], out[j][1]), bf16x2_rne(out[j][2], out[j][3]),
                           bf16x2_rne(out[j][4], out[j][5]), bf16x2_rne(out[j][6], out[j][7]));
    }
}

// The kernel around the body, whose waves past the first return after the fold: CH `r4dchain` is
// the chained form (r4d_chain.h).
template <uint32_t KW, uint32_t SB, uint32_t NT, class CH>
__global__ __launch_bounds__(KW * kWave) R4D_WAVES(6)
void gemm_i8_kpar_kernel(R4D_I8KPAR_PARAMS) {
    if (CH::on) r4d_chain_wait();
    gemm_i8_kpar_body<KW, SB, NT>(Ap, Sa, Bp, Sb, C, rows, cols, M, b_ld, sb_ld);
    if (CH::on) r4d_chain_done();
}
#undef R4D_I8KPAR_PARAMS

// Sum the split-K partials into bf16 C, in fixed z order with no atomics. Latency bound, so eight
// loads are issued before their adds; the order of the adds is unchanged.
__global__ void splitk_reduce_kernel(const float* __restrict__ ws, uint16_t* __restrict__ y,
                                     uint32_t n_out, uint32_t ksplit) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t)n_out;
    if (i >= n) return;
    const float* __restrict__ p = ws + i;
    constexpr uint32_t U = 8;
    float acc = 0.0f;
    uint32_t z = 0;
    for (; z + U <= ksplit; z += U) {
        float v[U];
#pragma unroll
        for (uint32_t u = 0; u < U; ++u) v[u] = p[(size_t)(z + u) * n];
#pragma unroll
        for (uint32_t u = 0; u < U; ++u) acc += v[u];
    }
    for (; z < ksplit; ++z) acc += p[(size_t)z * n];
    y[i] = f32_to_bf16(acc);
}

template <bool I8, int RB = 0, uint32_t SB = kBlock>
__global__ __launch_bounds__(kPWaves * kWave) R4D_WAVES(6)
void gemm_8a8_prefill_kernel(const uint8_t* __restrict__ A, const float* __restrict__ Sa,
                             const uint8_t* __restrict__ W, const uint16_t* __restrict__ Sw,
                             uint16_t* __restrict__ C, uint32_t M, uint32_t K, uint32_t N,
                             uint32_t a_ld, uint32_t sa_ld, uint32_t sw_ld) {
    gemm_8a8_prefill_body<I8, RB, PrefillPlain, false, SB>(A, Sa, W, Sw, C, M, K, N, a_ld, sa_ld,
                                                          sw_ld, PrefillPlain{});
}

// The narrow tile's K width is kFKNarrow (256 B); 128 B loses at every wave count (see `kTPR`), and
// is used only via narrow_fk for a K that 256 does not divide.

// Double-buffering the narrow tile is ~10% slower even where it fits (MW = 2): the tile is short
// of LDS, not waiting on the barrier.

// The finisher's arrival counters, one per grid.x column (see the finisher). Zeroed once and kept
// zero by the finisher. If allocation fails, the caller falls back to the separate reduce kernel,
// which computes the same bytes.
constexpr uint32_t kMaxCtr = 8192;

/* One counter block per (stream, output), so concurrent launches never share one (sharing would
 * mean a hang or a wrong answer). Launches overlap across streams (`rad_lane`) or on one stream
 * (rad_kernel_concurrent); overlapping launches on one stream write disjoint outputs, and launches
 * sharing an output on one stream are ordered by their dependency. A step has a few dozen distinct
 * outputs at fixed arena addresses, so this is a few dozen 32 KiB blocks, allocated once. A block
 * lives as long as its output's memory (r4d_keyed.h): a recorded pass replays the pointer.
 *
 * The lock is not held while a block is made. Zeroing it waits for the card, and under tensor
 * parallelism the card can be inside an all-reduce whose other half the peer rank has yet to issue;
 * if the peer's first launch of a split GEMM then waited on this lock, neither rank would advance. */
unsigned int* split_ctr(RadStream s, const void* out) {
  return r4d::keyed_block(r4d::Keyed::SplitFp8, s, out,
                          (size_t)kMaxCtr * sizeof(unsigned int));
}
// The narrow tile's wave count MW, which is its M tile: the wave grid is MW x 1, so a block covers
// MW*16 weight rows and the grid over N is N/(MW*16). The rule is fitted together with
// kSplitTarget (which shapes split decides whether narrow or wide blocks win) on the shapes the
// engine actually issues (R4D_F8_MWLOG; n_ff is already per rank).
//
//   - M <= 16: MW = 4, best or within a microsecond on every decode shape.
//   - M > 16: MW = 8. The weight is streamed once whatever MW is, but every block stages the whole
//     [M, K] activation, so a call moves N*K + M*K*N/(MW*16) bytes; the second term matches the
//     weight at M = 64 (served from cache, but it is the drop from 586 to 460 GB/s on gate_up).
//     MW = 8 halves it at the same N/16 waves. The crossover is where NT leaves 1.
//   - M > 32 and N >= 2048: MW = 16 (FM = 256, legal because the fold reads the weight scale per
//     wave). Halves the re-read again; narrower N would leave too few blocks for 32 WGPs.
//
// Token tiles NT = ceil(M/16), including 3: rounding 33..48 up to a 64-token tile would spend
// three-eighths of the work at M = 40 on padding.
//
// Neither MW nor NT changes the output bits: `narrow_ksplit` depends on `gx * mw` = N/16 (the rule
// requires `n % 128` so that holds), so every element is one wave's accumulation over the same K
// tiles in the same order. The selftest checks this with the pin_mw / pin_nt hooks.
//
// This part shows occasional correlated +5 us outliers across repetitions; a cell that breaks its
// row's monotonicity needs re-measuring.
/* Test hook behind r4d_gemm_fp8a8_pin_mw: zero is the rule, anything else overrides every call.
 * Separate from the environment pin, which is read once per process, so a bit check can compare
 * both settings in one process. */
std::atomic<uint32_t> g_mw_force{0u};

uint32_t narrow_mw(uint32_t n, uint32_t m) {
    uint32_t mw = 4u;
    if (m > 16u && (n % 128u) == 0u) mw = 8u;
    if (m > 32u && n >= 2048u && (n % 256u) == 0u) mw = 16u;
    /* Regrouping the same waves into more or fewer blocks (e.g. N = 2560: 40 blocks at MW 4) is
     * slower both ways: the cost is per-block work (activation re-read, scale plane, prologue), not
     * scheduling balance. */
    if (const uint32_t f = g_mw_force.load(std::memory_order_relaxed)) mw = f;
    /* `R4D_F8_MWLOG` prints each distinct (N, M) once, up to 64 (n_ff in a container is already
     * per rank, so the shapes must be read, not derived). */
    static const bool mwlog = [] { const char* v = std::getenv("R4D_F8_MWLOG");
                                   return v && std::strtoul(v, nullptr, 10) != 0; }();
    if (mwlog) {
        static std::mutex lm;
        static uint64_t seen[64];
        static int nseen = 0;
        const uint64_t key = ((uint64_t)n << 32) | m;
        std::lock_guard<std::mutex> lk(lm);
        int i = 0;
        while (i < nseen && seen[i] != key) ++i;
        if (i == nseen && nseen < 64) {
            seen[nseen++] = key;
            std::fprintf(stderr, "[mwlog] N=%u M=%u -> mw=%u\n", n, m, mw);
        }
    }
    return mw;
}

// Raising the split target so N = 2560 (160 waves, 40 blocks) splits does not change its time, so
// the narrow grid is not parallelism-starved at 128. A split shows in grid.z, not grid.x.

/* The narrow tile's K width: 256, or 128 when K is a multiple of 128 but not 256 (e.g. K = 640, a
 * shared expert's down projection), which would otherwise fall to the prefill kernel. At 128 the
 * fold takes the `FK <= kBlock` branch. One function, because the scratch query and the launch
 * must agree. */
inline uint32_t narrow_fk(uint32_t K) { return (K % kFKNarrow == 0u) ? kFKNarrow : 128u; }

/* Test hook behind r4d_gemm_i8a8_pin_split_waves: zero is each caller's own target, anything else
 * overrides every call's. The selftest pins the IQR narrow tile's target around its int8
 * reference, so the two sum their K partials in one order and compare to the byte. */
std::atomic<uint32_t> g_split_waves_force{0u};

// The split, given the narrow grid. Shared by the scratch query and the launch so they agree on the
// number of partial planes. `short_unsplit` is the `gemm_nt_q_short` row's opt-in (see below).
// `target` is the waves the split aims for: kSplitTarget for the fp8a8 and i8a8 tiles; the IQR
// narrow tile (r4d_gemm_iqr.hip), whose waves decode before they multiply, asks for more.
uint32_t narrow_ksplit(uint32_t N, uint32_t K, uint32_t M, uint32_t* ck_out,
                       bool short_unsplit = false, uint32_t target = kSplitTarget) {
    *ck_out = 0;
    if (const uint32_t f = g_split_waves_force.load(std::memory_order_relaxed)) target = f;
    const uint32_t mw = narrow_mw(N, M);
    const uint32_t fm = mw * 16u;   // the narrow tile is one M tile a wave
    const uint32_t gx = (N + fm - 1) / fm;              // the M grid runs over the weight rows
    if (K < 512u || gx * mw >= target) return 1;
    /* A narrow weight (N <= 512, the attention k/v projections) at more than 40 tokens: the split
     * runs into a cliff there (gfx1200, cold weights: N = 512, K = 2048, M = 44..64 split 51-54 us,
     * unsplit 23-26; K = 5120, M = 41..64 58-63 against 40-48), while at K >= 8192 the split still
     * wins. Unsplit changes the order these outputs are summed in, so a k/v row at 41..64 tokens
     * differs from the split result in its last bits; the rows are as exact either way. Hence an
     * opt-in: only `gemm_nt_q_short`, whose declaration the model owns, takes it, and every other
     * caller keeps the split and its bits. */
    if (short_unsplit && N <= 512u && M > 40u && K <= 5120u) return 1;
    const uint32_t want = (target + gx * mw - 1) / (gx * mw);
    uint32_t ckw = ((K / want) + kBlock - 1u) / kBlock * kBlock;
    if (ckw < kBlock) ckw = kBlock;
    uint32_t k2 = (K + ckw - 1) / ckw;
    if (k2 > kSplitThinCap && (uint64_t)N * M < 8192) k2 = kSplitThinCap;
    if (k2 < 2) return 1;
    // Round the slice to the staged tile, not the 128-column scale block: a slice that is only a
    // multiple of 128 would run one 256-wide tile past its end into the next slice.

    const uint32_t fk = narrow_fk(K);
    uint32_t ck = ((K + k2 - 1) / k2 + fk - 1u) / fk * fk;
    k2 = (K + ck - 1) / ck;
    if (k2 < 2) return 1;
    *ck_out = ck;
    return k2;
}

}  // namespace

/* The narrow tile's grid and split rules and its finisher's counters, for the IQR narrow tile
 * (r4d_gemm_iqr.hip): its split sums each output as this tile's does. */
uint32_t r4d_8a8_narrow_mw(uint32_t n, uint32_t m) { return narrow_mw(n, m); }
uint32_t r4d_8a8_narrow_ksplit(uint32_t N, uint32_t K, uint32_t M, uint32_t* ck, uint32_t target) {
    return narrow_ksplit(N, K, M, ck, false, target);
}
unsigned int* r4d_8a8_split_ctr(RadStream s, const void* out) { return split_ctr(s, out); }

// ================================================================== the entry points

extern "C" int r4d_gemm_fp8a8_block(void)          { return (int)kBlock; }
static int gemm_fp8a8_nt_m16_emit(long a, long ascale, long wq, long ws, long c,
                                  int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                  long scratch, long scratch_bytes, RadStream stream,
                                  int i8 = 0, int split_n = 0, bool chain = false,
                                  bool short_unsplit = false);

extern "C" int r4d_gemm_fp8a8_nt_m16(long a, long ascale, long wq, long ws, long c,
                                     int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                     long scratch, long scratch_bytes, long stream) {
    return gemm_fp8a8_nt_m16_emit(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld,
                                  scratch, scratch_bytes, (RadStream)(uintptr_t)stream);
}

extern "C" int r4d_gemm_fp8a8_nt_m16_max_m(void)   { return 64; }

/* See r4d.h. Only instantiated wave counts are accepted; anything else clears the override rather
 * than falling through to the dispatch's last arm. */
extern "C" void r4d_gemm_i8a8_pin_split_waves(int waves) {
    g_split_waves_force.store(waves > 0 ? (uint32_t)waves : 0u, std::memory_order_relaxed);
}

extern "C" void r4d_gemm_fp8a8_pin_mw(int mw) {
    const uint32_t v = (mw == 1 || mw == 2 || mw == 4 || mw == 8 || mw == 16) ? (uint32_t)mw : 0u;
    g_mw_force.store(v, std::memory_order_relaxed);
}

/* The same hook for the token-tile count. `nt` may be pinned up but never below ceil(M/16): NT*16
 * is how many token columns the epilogue writes, so a lower pin would drop rows. */
std::atomic<uint32_t> g_nt_force{0u};

extern "C" void r4d_gemm_fp8a8_pin_nt(int nt) {
    const uint32_t v = (nt == 1 || nt == 2 || nt == 3 || nt == 4) ? (uint32_t)nt : 0u;
    g_nt_force.store(v, std::memory_order_relaxed);
}

// Bytes of f32 scratch one r4d_gemm_fp8a8_nt_m16 call needs, or 0 when it will not split. The
// caller may pass more, or none (the launch then runs unsplit: correct but slow, and unreported).
// The bound is max_m() so the query and the band cannot drift apart.
//
// `split_n` (0: N) is the width the split is chosen for. The split depends on N alone, and it
// fixes the order K partials are summed in, so a row slice of a wider linear (N rows of a
// split_n-row weight) passes the whole width to sum each output as the whole linear does: the
// slice's outputs are then the linear's to the bit.
static long m16_scratch(int M, int K, int N, int split_n, bool short_unsplit = false) {
    if (M <= 0 || K <= 0 || N <= 0 || M > r4d_gemm_fp8a8_nt_m16_max_m()) return 0;
    uint32_t ck = 0;
    const uint32_t k2 = narrow_ksplit((uint32_t)(split_n > 0 ? split_n : N), (uint32_t)K,
                                      (uint32_t)M, &ck, short_unsplit);
    return k2 > 1 ? (long)k2 * M * N * (long)sizeof(float) : 0;
}
extern "C" long r4d_gemm_fp8a8_nt_m16_scratch(int M, int K, int N) {
    return m16_scratch(M, K, N, 0);
}

// C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, M <= 64.
//
//   a       [M][K]                     E4M3, row-major, `a_ld` bytes a row (0 = K)
//   ascale  [M][ceil(K/128)]           f32,  `sa_ld` elements a row (0 = ceil(K/128))
//   wq      [N][K]                     E4M3, row-major, `w_ld` bytes a row (0 = K)
//   ws      [ceil(N/128)][ceil(K/128)] bf16, `sw_ld` elements a row (0 = ceil(K/128))
//   c       [M][N]                     bf16
//
// Returns 0, or a negative code for a shape it does not serve. `scratch` may be 0. `split_n` is
// r4d_gemm_fp8a8_nt_m16_scratch's. `i8`: 0 the fp8 tile, 1 the int8 tile at a scale per 128 of K,
// 2 the int8 tile at a scale per 256 (IQ8R, K a multiple of 256).
static int gemm_fp8a8_nt_m16_emit(long a, long ascale, long wq, long ws, long c,
                                  int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                  long scratch, long scratch_bytes, RadStream stream, int i8,
                                  int split_n, bool chain, bool short_unsplit) {
    int emit_rc = RAD_OK;
    if (!a || !ascale || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    // K must be a multiple of 128; the staged tile is 256 bytes where K allows and 128 otherwise
    // (narrow_fk). The declared constraint must match this guard (cGemmNtFp8A8M16 in r4d_rows.cpp):
    // the resolver offers a call by what a row claims, not by what the entry point accepts.
    if (K % 128) return -4;
    // A wave owns 16 weight rows and the transposing store writes `N`-strided columns.
    if (N % 16) return -5;
    if (!a_ld)  a_ld  = K;
    if (!w_ld)  w_ld  = K;
    const int grp = i8 == 2 ? (int)R4D_I8_GROUP_WIDE : (int)kBlock;
    if (K % grp) return -4;
    const int nkb = (K + grp - 1) / grp;
    if (!sa_ld) sa_ld = nkb;
    if (!sw_ld) sw_ld = nkb;

    /* An int8 weight to 16 tokens: the K-parallel tile (its 16-byte stores want an aligned C). */
    /* Measured against the narrow tile at Flash-Next's per-rank shapes, weights cold (us, narrow
     * -> this): to 16 tokens 768 x 2560 8.5 -> 6.1, 512 x 2560 6.3 -> 4.4, 2560 x 3072 16.7 ->
     * 15.9, 8192 x 2560 38.0 -> 36.8; at 64 tokens 768 x 2560 10.9 -> 7.5, 2560 x 3072 22.8 ->
     * 18.5, 6144 x 2560 level, and 8192 x 2560 41.8 -> 42.8, so from 49 tokens a weight of 8192
     * rows and up keeps the narrow tile. Never slower elsewhere. A call under
     * r4d_gemm_i8a8_pin_split_waves keeps the narrow tile too: the pin asks for the narrow
     * tile's split, whose bytes the IQR narrow tile is compared with. */
    const bool split_pinned = g_split_waves_force.load(std::memory_order_relaxed) != 0u;
    if (i8 && !split_pinned && M <= 64 && (M <= 48 || N < 8192) &&
        ((uintptr_t)c & 15u) == 0u) {
        const uint32_t ngrp = (uint32_t)(K / grp);
        const uint32_t knt = ((uint32_t)M + 15u) / 16u;
        const dim3 kgrid((uint32_t)N / 16u);
#define R4D_I8K_E(kw_, sb_, nt_, ch_)                                                          \
        emit_rc = R4D_EMIT(stream, (gemm_i8_kpar_kernel<kw_, sb_, nt_, ch_>), kgrid,          \
                           dim3(kw_ * kWave), 0, (const uint8_t*)wq, (const uint16_t*)ws,      \
                           (const int8_t*)a, (const float*)ascale, (uint16_t*)c, (uint32_t)N,  \
                           (uint32_t)K, (uint32_t)M, (uint32_t)a_ld, (uint32_t)sa_ld)
#define R4D_I8K(kw_, sb_, nt_)                                                                 \
        do { if (chain) R4D_I8K_E(kw_, sb_, nt_, r4dchain);                                    \
             else       R4D_I8K_E(kw_, sb_, nt_, R4dUnchained); } while (0)
#define R4D_I8K_NT(kw_, sb_)                                                                   \
        do { if (knt == 1u)      R4D_I8K(kw_, sb_, 1u);                                        \
             else if (knt == 2u) R4D_I8K(kw_, sb_, 2u);                                        \
             else if (knt == 3u) R4D_I8K(kw_, sb_, 3u);                                        \
             else                R4D_I8K(kw_, sb_, 4u); } while (0)
#define R4D_I8K_KW(sb_)                                                                        \
        do { if (ngrp >= 4u)      R4D_I8K_NT(4u, sb_);                                         \
             else if (ngrp >= 2u) R4D_I8K_NT(2u, sb_);                                         \
             else                 R4D_I8K_NT(1u, sb_); } while (0)
        if (i8 == 2) R4D_I8K_KW(256u); else R4D_I8K_KW(128u);
#undef R4D_I8K_KW
#undef R4D_I8K_NT
#undef R4D_I8K
#undef R4D_I8K_E
        if (emit_rc != RAD_OK || hipGetLastError() != hipSuccess) return -6;
        return 0;
    }

    const uint32_t mw = narrow_mw((uint32_t)N, (uint32_t)M);
    const uint32_t fm = mw * 16u;   // the narrow tile is one M tile a wave
    const uint32_t gx = ((uint32_t)N + fm - 1) / fm;

    uint32_t ck = 0;
    uint32_t ksplit = narrow_ksplit((uint32_t)(split_n > 0 ? split_n : N), (uint32_t)K,
                                    (uint32_t)M, &ck, short_unsplit);
    const long need = ksplit > 1 ? (long)ksplit * M * N * (long)sizeof(float) : 0;
    if (need && (!scratch || scratch_bytes < need)) { ksplit = 1; ck = 0; }

    const dim3 grid(gx, 1, ksplit);
    void* const dst = ksplit > 1 ? (void*)scratch : (void*)c;
    /* Token tiles per wave: the activation is the B operand, so NT tiles cover NT*16 tokens with
     * one weight read. NT = ceil(M/16), including 3 (see the note above narrow_mw). The gate stash
     * scales with NT on both sides (4096*NT bytes against FN * (256+8) = 4224*NT), so it fits. */
    uint32_t nt = (M > 48) ? 4u : ((M > 32) ? 3u : ((M > 16) ? 2u : 1u));
    if (const uint32_t f = g_nt_force.load(std::memory_order_relaxed)) { if ((uint32_t)M <= f * 16u) nt = f; }
    /* `Full` drops the column guard, so it must mean all NT tiles are live, not just M % 16 == 0.
     * Otherwise (e.g. M = 48, NT = 4) the epilogue writes past a split-K scratch sized for M, over
     * the arrival counter, and the finisher hangs. */
    const bool full = ((uint32_t)N % fm == 0) && ((uint32_t)M % (nt * 16u) == 0) &&
                      (K % (int)kBlock == 0);
    // The finisher needs one counter per column; beyond kMaxCtr or on allocation failure, the
    // separate reduce kernel computes the same bytes.
    unsigned int* const ctr =
        (ksplit > 1 && gx <= kMaxCtr) ? split_ctr(stream, (const void*)c) : nullptr;

#define R4D_F8D(f, mw_, nt_, fk_, i8_, sb_)                                                    \
    emit_rc = R4D_EMIT(stream,                                                                 \
                       (chain ? r4dchain_gemm_fp8a8<f, nt_, mw_, fk_, false, i8_, false, sb_>  \
                              : gemm_fp8a8_kernel<f, nt_, mw_, fk_, false, i8_, false, sb_>), \
                       grid, dim3(mw_ * kWave),                                                \
                       0, (const uint8_t*)wq, (const void*)ws,                                 \
                       (const uint8_t*)a, (const void*)ascale, dst, (void*)c, ctr,              \
                       (uint32_t)N, (uint32_t)K,                                                \
                       (uint32_t)M, (uint32_t)w_ld, (uint32_t)a_ld, (uint32_t)sw_ld,            \
                       (uint32_t)sa_ld, ck,                                                     \
                       /* The eight defaulted parameters, spelled out: r4d_emit passes          \
                        * kernelParams as an array, which cannot fill defaults, and asserts the  \
                        * arity. */                                                              \
                       (uint8_t*)nullptr, (float*)nullptr, (uint16_t*)nullptr,                  \
                       (uint32_t)0, (uint32_t)0, (uint32_t)0, (const uint16_t*)nullptr,          \
                       (uint32_t)0)
#define R4D_F8D_NT(f, mw_, fk_, i8_, sb_)                                                      \
    do { if (nt == 4u)      R4D_F8D(f, mw_, 4u, fk_, i8_, sb_);                                \
         else if (nt == 3u) R4D_F8D(f, mw_, 3u, fk_, i8_, sb_);                                \
         else if (nt == 2u) R4D_F8D(f, mw_, 2u, fk_, i8_, sb_);                                \
         else               R4D_F8D(f, mw_, 1u, fk_, i8_, sb_); } while (0)
#define R4D_F8D_MW(fk_, i8_, sb_)                                                              \
    do { if (full) {                                                                           \
             if (mw == 4u)      R4D_F8D_NT(true, 4u, fk_, i8_, sb_);                           \
             else if (mw == 8u) R4D_F8D_NT(true, 8u, fk_, i8_, sb_);                           \
             else               R4D_F8D_NT(true, 16u, fk_, i8_, sb_);                          \
         } else {                                                                              \
             if (mw == 4u)      R4D_F8D_NT(false, 4u, fk_, i8_, sb_);                          \
             else if (mw == 8u) R4D_F8D_NT(false, 8u, fk_, i8_, sb_);                          \
             else               R4D_F8D_NT(false, 16u, fk_, i8_, sb_);                         \
         } } while (0)
    const uint32_t fk = narrow_fk((uint32_t)K);
    if (mw == 1u || mw == 2u) {
        /* Only the selftest's pin reaches these (narrow_mw answers 4, 8 or 16), and only the fp8
         * set at the 256-wide K step builds them, to keep the object size down. */
        if (i8 || fk != kFKNarrow) return -4;
        if (full) { if (mw == 1u) R4D_F8D_NT(true, 1u, kFKNarrow, false, kBlock);
                    else          R4D_F8D_NT(true, 2u, kFKNarrow, false, kBlock); }
        else      { if (mw == 1u) R4D_F8D_NT(false, 1u, kFKNarrow, false, kBlock);
                    else          R4D_F8D_NT(false, 2u, kFKNarrow, false, kBlock); }
    } else if (fk == kFKNarrow) {
        if (i8 == 2)  R4D_F8D_MW(kFKNarrow, true, 256u);
        else if (i8)  R4D_F8D_MW(kFKNarrow, true, kBlock);
        else          R4D_F8D_MW(kFKNarrow, false, kBlock);
    } else {
        if (i8 == 2) return -4;   /* K % 256 was checked above; unreachable */
        if (i8) R4D_F8D_MW(128u, true, kBlock); else R4D_F8D_MW(128u, false, kBlock);
    }
#undef R4D_F8D_MW
#undef R4D_F8D_NT
#undef R4D_F8D
    if (emit_rc != RAD_OK) return -6;
    if (hipGetLastError() != hipSuccess) return -6;

    if (ksplit > 1 && !ctr) {
        const uint32_t n = (uint32_t)M * (uint32_t)N;
        if (R4D_EMIT(stream, splitk_reduce_kernel, dim3((n + 255) / 256), dim3(256), 0,
                     (const float*)scratch, (uint16_t*)c, n, ksplit) != RAD_OK) return -7;
    }
    return 0;
}

// The same GEMM with gated_quant_fp8's pass folded into its epilogue (see `GATED` on the kernel).
// Every unmet shape condition is a refusal: a gated call running ungated would write the wrong
// buffer.
//
//   q       [M][nout]                  E4M3, `q_ld` bytes a row (0 = nout)
//   s       [M][ceil(nout/128)]        f32,  `s_ld` elements a row
//   y       [M][nout] or 0             bf16, the silu product the unfused pair also wrote
//
// The weight is the gate_up plane [2*nout][K], gate first, as the unfused GEMM reads it.
extern "C" int r4d_gemm_fp8a8_gated_nt_m16_max_m(void) { return 64; }

// The same GEMM for prefill row counts: same operands and layouts, the 128x128 prefill tile
// (gemm_8a8_prefill_kernel). No split-K: the grid is ceil(M/128) x N/128, hundreds of blocks at any
// prefill shape.
template <bool I8, int RB = 0, uint32_t SB = kBlock>
static int gemm_8a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int a_ld, int sa_ld, int w_ld, int sw_ld, long stream);

extern "C" int r4d_gemm_fp8a8_tiled(long a, long ascale, long wq, long ws, long c,
                                    int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                    long stream) {
    return gemm_8a8_tiled<false>(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, stream);
}

// The int8 twins: i8 codes for both operands and a bf16 weight scale per row per 128 K (ws is
// [N][ceil(K/128)]); otherwise as the fp8 entry points.
extern "C" int r4d_gemm_i8a8_tiled(long a, long ascale, long wq, long ws, long c,
                                   int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                   long stream) {
    return gemm_8a8_tiled<true>(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, stream);
}

// The same with the weight's scale one per 8 rows and 128 K (each row of the plane holding its
// block's value), folded as the fp8 tile folds (gemm_8a8_prefill_kernel's RB).
extern "C" int r4d_gemm_i8a8_rb8_tiled(long a, long ascale, long wq, long ws, long c,
                                       int M, int K, int N, int a_ld, int sa_ld, int w_ld,
                                       int sw_ld, long stream) {
    return gemm_8a8_tiled<true, 8>(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld,
                                   stream);
}

// The same at a weight scale per 32 rows, a wave's: the fp8 tile's fold.
extern "C" int r4d_gemm_i8a8_rb32_tiled(long a, long ascale, long wq, long ws, long c,
                                        int M, int K, int N, int a_ld, int sa_ld, int w_ld,
                                        int sw_ld, long stream) {
    return gemm_8a8_tiled<true, 32>(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld,
                                    stream);
}

extern "C" int r4d_gemm_i8a8_nt_m16(long a, long ascale, long wq, long ws, long c,
                                    int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                    long scratch, long scratch_bytes, long stream) {
    return gemm_fp8a8_nt_m16_emit(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld,
                                  scratch, scratch_bytes, (RadStream)(uintptr_t)stream, true);
}

template <bool I8, int RB, uint32_t SB>
static int gemm_8a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int a_ld, int sa_ld, int w_ld, int sw_ld, long stream) {
    if (!a || !ascale || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    // Required: the fold assumes whole SB-column scale blocks, and a block's 128 weight rows are
    // one scale row.
    if (K % (int)SB) return -4;
    if (N % (int)kPBN) return -5;
    // C is written 16 bytes at a time.
    if ((uintptr_t)c & 15u) return -3;
    if (!a_ld)  a_ld  = K;
    (void)w_ld;   // the fragment-order plane has no row pitch
    const int nkb = (K + (int)SB - 1) / (int)SB;
    if (!sa_ld) sa_ld = nkb;
    if (!sw_ld) sw_ld = nkb;

    const dim3 grid(((uint32_t)M + kPBM - 1) / kPBM, (uint32_t)N / kPBN, 1);
    const int rc = R4D_EMIT((RadStream)(uintptr_t)stream, (gemm_8a8_prefill_kernel<I8, RB, SB>), grid,
                            dim3(kPWaves * kWave), 0, (const uint8_t*)a, (const float*)ascale,
                            (const uint8_t*)wq, (const uint16_t*)ws, (uint16_t*)c, (uint32_t)M,
                            (uint32_t)K, (uint32_t)N, (uint32_t)a_ld, (uint32_t)sa_ld,
                            (uint32_t)sw_ld);
    return rc == RAD_OK && hipGetLastError() == hipSuccess ? 0 : -6;
}

/* gemm_nt_q_bias's shim: gemm_nt_q's four operands, then the bias and C (oGemmNtQBias in
 * r4d_rows.cpp). Not fp8_ops: C is operand 5 and may be a column slice of a wider buffer, which
 * fp8_ops refuses for the unbiased kernels' dense store. */
enum { GQB_A = 0, GQB_AS = 1, GQB_B = 2, GQB_BS = 3, GQB_BIAS = 4, GQB_Y = 5 };

/* The gated GEMM's shim. Not fp8_ops: that derives M and N from operand 4, which here is the
 * quantiser's [M, nout] output rather than the GEMM's [M, 2*nout]. Operand indices are
 * `gemm_nt_q_gated`'s (oGemmNtQGated in r4d_rows.cpp). */
enum { GG_A = 0, GG_AS = 1, GG_B = 2, GG_BS = 3, GG_Q = 4, GG_S = 5, GG_Y = 6 };

/* ---- the int8 twins' shims: the same operands with i8 codes and the weight's scale plane per
 * row, which fp8_ops checks at [N][K/128] when asked (FP8_OPS_ROW_SCALE). */
extern "C" int64_t r4d_rad_gemm_i8a8_m16_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), N = r4d_geti_or(a, "N", 0),
                    K = r4d_geti_or(a, "K", 0), sn = r4d_geti_or(a, "split_n", 0);
    if (M < 1 || N < 1 || K < 1) return 0;
    if (M > r4d_gemm_fp8a8_nt_m16_max_m()) return 0;
    return (int64_t)m16_scratch((int)M, (int)K, (int)N, (int)sn);
}

extern "C" int r4d_rad_gemm_i8a8_m16(const RadArgs* a, RadStream stream) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1, FP8_OPS_ROW_SCALE);
    if (rc != RAD_OK) return rc;
    if (g.M > r4d_gemm_fp8a8_nt_m16_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (g.a->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    const int e = gemm_fp8a8_nt_m16_emit(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)a->scratch, (long)a->scratch_bytes, stream,
        g.group == R4D_I8_GROUP_WIDE ? 2 : 1,
        (int)r4d_geti_or(a, "split_n", 0), r4d_geti_or(a, "chain", 0) != 0);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_gemm_i8a8_tiled(const RadArgs* a, RadStream s) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1, FP8_OPS_ROW_SCALE);
    if (rc != RAD_OK) return rc;
    if (g.a->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    const int e = g.group == R4D_I8_GROUP_WIDE
        ? gemm_8a8_tiled<true, 0, 256u>(
              r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
              (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
              (long)(uintptr_t)s)
        : r4d_gemm_i8a8_tiled(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_gemm_i8a8_rb8_tiled(const RadArgs* a, RadStream s) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1, FP8_OPS_ROW_SCALE);
    if (rc != RAD_OK) return rc;
    if (g.a->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    const int e = r4d_gemm_i8a8_rb8_tiled(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_gemm_i8a8_rb32_tiled(const RadArgs* a, RadStream s) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1, FP8_OPS_ROW_SCALE);
    if (rc != RAD_OK) return rc;
    if (g.a->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    const int e = r4d_gemm_i8a8_rb32_tiled(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

