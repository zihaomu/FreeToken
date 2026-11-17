// Block-scaled FP8 weight, bf16 activation, few rows: the decode form of the checkpoint's dense
// format, `<linear>.weight` as F8_E4M3 [N][K] with `<linear>.weight_scale_inv` in BF16 on a
// [ceil(N/128)][ceil(K/128)] grid. Also the fp8 lm_head (logits_gemm), whose scales sit in each
// row's tail (r4d_fp8_lm.h).
//
// Unlike r4d_gemm_fp8a8_nt_m16 at M = 1, this reads the bf16 activation directly: no E4M3
// quantiser launch in front of the linear, and no 16-row tile of wasted matrix work.
//
// The scale is bf16 per 128x128 tile (r4d_fp8_wmma.h); the activation and C are bf16 (logits: f32).
//
// `rb` (rows per wave) is a tuning argument: blocking trades waves for memory-level parallelism,
// and the best value depends on the shape and the inner loop. 0 takes rb_default; 1, 2, 4 and 8
// pin it.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.
//
// ---- the multi-row loop is issue-bound (gfx1201) -----------------------------------------------
//
// At MT = 8 the loop is ~394 instructions per 16 weight bytes per lane, ~89 of them widening bf16
// activations to f32 (v_fma_f32 cannot read bf16); the instruction count accounts for the rate.
// v_pk_fma_f32 does not exist on RDNA4. v_dot2_f32_bf16 does, and E4M3 -> bf16 is exact, but a loop
// around it (215 instructions) is ~30% slower: dot instructions run at a fraction of v_fmac_f32's
// rate on this part. The instruction count is not the cost.
//
// lm_wmma_kernel below decodes E4M3 to bf16 fragments for the bf16 WMMA and holds close to stream
// rate from M = 1 to 16, so r4d_logits_gemm_fp8 uses it above M = 2.

#include "r4d.h"
#include "r4d_fp8_args.h"
#include "r4d_args.h"
#include "r4d_fp8_lm.h"
#include "r4d_fp8_wmma.h"
#include "r4d_chain.h"
#include "r4d_iqr.h"

#include <algorithm>

/* Rows per wave for the multi-row matvec (see rows_dot_mt). RB divides the activation traffic but
 * RB * MT is the accumulator count:
 *
 *   - MT <= 2: the activation is already amortised, so RB = 1;
 *   - middle of the band: the activation re-read dominates, RB = 2 then 4;
 *   - MT = 32: RB = 4 would be 128 accumulators per lane, so RB = 2. */
static uint32_t mvm_rb(int mt) {
    if (mt <= 2) return 1u;
    if (mt <= 8) return 2u;
    if (mt <= 16) return 4u;
    return 2u;
}

/* Scale blocks per staging round, by MT = ceil(M / 16) (see lm_wmma_kernel): two at MT <= 2, one
 * from MT = 3, where the wider tile's LDS costs more than the row straddle. */
static uint32_t lm_fkb_rule(uint32_t mt) { return mt <= 2u ? 2u : 1u; }

namespace {

using namespace r4d_fp8;

constexpr int kWave = 32;
constexpr int kDenseWaves = 8;
constexpr int kDenseThreads = kDenseWaves * kWave;
constexpr uint32_t kBlock = 128;   // the scale block; see r4d_gemm_fp8a8.hip
// The widest token count r4d_gemm_fp8a16_nt_mt serves: a register budget (MT accumulators plus the
// decoded weight window). 8 covers n_spec up to 7.
constexpr uint32_t kMaxMt = 8;

// The activation, seen as f32 by the dot. `XV` is anything with `operator[](uint32_t) -> float`,
// so a fused input transform (e.g. an RMS norm) could reuse the same loop.
struct Bf16V {
    const uint16_t* __restrict__ x;
    __device__ __forceinline__ float operator[](uint32_t i) const {
        return __uint_as_float((uint32_t)x[i] << 16);
    }
};

// One output row, reduced across the wave.
//
// 32 lanes x uint4 = 512 contiguous bytes per pass, four scale tiles: lanes 0-7 fall in the first
// tile, 8-15 the second, and so on. A lane's 16 columns share one scale, so each lane applies its
// own and the row needs a single wave reduction at the end.
template <class XV>
__device__ __forceinline__ float row_dot(const uint8_t* __restrict__ W,
                                         const uint16_t* __restrict__ S, const XV& xv,
                                         uint32_t row, uint32_t cols, uint32_t w_ld,
                                         uint32_t sc_cols, int lane) {
    const uint32_t stride = kWave * 16u;
    const uint32_t full = (cols / stride) * stride;
    const uint8_t* w = W + (size_t)row * w_ld;
    const uint16_t* srow = S + (size_t)(row / kBlock) * sc_cols;
    float acc = 0.0f;
    for (uint32_t c = (uint32_t)lane * 16u; c < full; c += stride) {
        const uint4 packed = *(const uint4*)(w + c);
        const uint32_t* pw = (const uint32_t*)&packed;
        const float scale = bscale(srow[c / kBlock]);
        float a0 = 0.0f, a1 = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const uint32_t p = pw[k];
            const uint32_t c4 = c + (uint32_t)k * 4u;
            // Two v_cvt_pk_f32_fp8 for four bytes; handles subnormal weights (see e4m3()).
            const v2f e01 = e4m3_pk<false>(p);
            const v2f e23 = e4m3_pk<true>(p);
            a0 = fmaf(e01.x, xv[c4 + 0], a0);
            a1 = fmaf(e01.y, xv[c4 + 1], a1);
            a0 = fmaf(e23.x, xv[c4 + 2], a0);
            a1 = fmaf(e23.y, xv[c4 + 3], a1);
        }
        acc = fmaf(a0 + a1, scale, acc);
    }
    // Tail, when K is not a multiple of 512. Same per-lane-scale rule, one column at a time.
    for (uint32_t c = full + (uint32_t)lane; c < cols; c += kWave)
        acc = fmaf(e4m3(w[c]) * xv[c], bscale(srow[c / kBlock]), acc);
    return r4d_wave_sum(acc);
}

// The same dot, RB rows at a time: one activation load feeds RB weight loads, which issue back to
// back. The one-row form has only one weight load per six memory instructions on the critical
// path, and a narrow matrix has too few iterations to hide it. `r0` is RB-aligned and RB divides
// 128, so the rows share a scale row. Summation order per row is unchanged.
template <uint32_t RB, class XV>
__device__ __forceinline__ void rows_dot(const uint8_t* __restrict__ W,
                                         const uint16_t* __restrict__ S, const XV& xv,
                                         uint32_t row0, uint32_t nr, uint32_t cols, uint32_t w_ld,
                                         uint32_t sc_cols, int lane, float* __restrict__ acc) {
    const uint32_t stride = kWave * 16u;
    const uint32_t full = (cols / stride) * stride;
    const uint8_t* w[RB];
    // Rows past `N` re-read row0 rather than branch; the caller discards them.
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) w[r] = W + (size_t)(row0 + (r < nr ? r : 0)) * w_ld;
    const uint16_t* srow = S + (size_t)(row0 / kBlock) * sc_cols;
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) acc[r] = 0.0f;

    for (uint32_t c = (uint32_t)lane * 16u; c < full; c += stride) {
        uint4 packed[RB];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) packed[r] = *(const uint4*)(w[r] + c);
        float xr[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) xr[k] = xv[c + (uint32_t)k];
        const float scale = bscale(srow[c / kBlock]);
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) {
            const uint32_t* pw = (const uint32_t*)&packed[r];
            float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t p = pw[k];
                const v2f e01 = e4m3_pk<false>(p);
                const v2f e23 = e4m3_pk<true>(p);
                a0 = fmaf(e01.x, xr[k * 4 + 0], a0);
                a1 = fmaf(e01.y, xr[k * 4 + 1], a1);
                a0 = fmaf(e23.x, xr[k * 4 + 2], a0);
                a1 = fmaf(e23.y, xr[k * 4 + 3], a1);
            }
            acc[r] = fmaf(a0 + a1, scale, acc[r]);
        }
    }
    for (uint32_t c = full + (uint32_t)lane; c < cols; c += kWave) {
        const float sc = bscale(srow[c / kBlock]);
        const float xc = xv[c];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) acc[r] = fmaf(e4m3(w[r][c]) * xc, sc, acc[r]);
    }
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) acc[r] = r4d_wave_sum(acc[r]);
}

// The int8 head's codes (logits_gemm_i8): a signed byte per weight, exact in f32 and bf16, where
// the E4M3 head's are decoded by v_cvt_pk_f32_fp8.
template <bool I8, bool HI>
__device__ __forceinline__ v2f code_pk(uint32_t p) {
    if constexpr (I8) {
        v2f r;
        r.x = (float)(int8_t)(p >> (HI ? 16 : 0));
        r.y = (float)(int8_t)(p >> (HI ? 24 : 8));
        return r;
    } else {
        return e4m3_pk<HI>(p);
    }
}
template <bool I8>
__device__ __forceinline__ float code1(uint8_t b) {
    if constexpr (I8) return (float)(int8_t)b;
    else return e4m3(b);
}

/* An IQR head (r4d_iqr.h's lm row: TB-bit codes indexing an int8 table of 2^TB entries, the row's
 * bf16 scales after its codes): TB = 0 is a byte a code (E4M3, or int8 with I8). The table sits
 * after the last row (lm_tab), and a kernel holds it in sixteen dwords (`tv`) for byte permutes:
 * a lookup of four codes is one v_perm a group of eight entries, the groups chosen bytewise by the
 * codes' upper bits. */
template <uint32_t TB>
__device__ __forceinline__ void lm_table(const uint8_t* W, uint32_t rows, uint32_t w_ld,
                                         uint32_t (&tv)[16]) {
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) tv[i] = 0u;
    if constexpr (TB != 0) {
        const uint32_t* t = (const uint32_t*)(W + ((size_t)rows * w_ld + 255u) / 256u * 256u);
#pragma unroll
        for (uint32_t i = 0; i < ((1u << TB) + 3u) / 4u; ++i) tv[i] = t[i];
    }
}
/* The table entries of four codes, one a byte of `x`. */
template <uint32_t TB>
__device__ __forceinline__ uint32_t lm_lookup4(uint32_t x, const uint32_t (&tv)[16]) {
    const uint32_t lo = x & 0x07070707u;
    uint32_t g[8];
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i)
        if (i < ((1u << TB) + 7u) / 8u) g[i] = __builtin_amdgcn_perm(tv[2 * i + 1], tv[2 * i], lo);
#pragma unroll
    for (uint32_t b = 3; b < TB; ++b) {
        /* bytes whose bit b is set take the upper half of each pair of groups */
        const uint32_t m = ((x >> b) & 0x01010101u) * 0xFFu;
        const uint32_t half = 1u << (b - 3);
#pragma unroll
        for (uint32_t i = 0; i < 8; i += 2 * half)
            if (i + half < ((1u << TB) + 7u) / 8u) g[i] = (g[i + half] & m) | (g[i] & ~m);
    }
    return g[0];
}
/* Sixteen consecutive codes from column c (a multiple of 16) of a row, as int8 bytes. */
template <uint32_t TB>
__device__ __forceinline__ uint4 lm_codes16(const uint8_t* __restrict__ row, uint32_t c,
                                            const uint32_t (&tv)[16]) {
    if constexpr (TB == 0) {
        return *(const uint4*)(row + c);
    } else {
        /* 2*TB bytes from offset 2*TB*(c/16), read from the dword-aligned window over them */
        constexpr uint32_t ND = (2u * TB + 3u + 3u) / 4u;
        const uint32_t off = c / 16u * 2u * TB;
        const uint32_t* p = (const uint32_t*)(row + (off & ~3u));
        uint32_t w[ND];
#pragma unroll
        for (uint32_t i = 0; i < ND; ++i) w[i] = p[i];
        const uint32_t sh = (off & 3u) * 8u;
        uint32_t d[ND];
#pragma unroll
        for (uint32_t i = 0; i < ND; ++i)
            d[i] = __builtin_amdgcn_alignbit(i + 1u < ND ? w[i + 1u] : 0u, w[i], sh);
        uint32_t o[4];
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q) {
            uint32_t x = 0;
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t bit = (q * 4u + j) * TB, i = bit >> 5, s = bit & 31u;
                const uint32_t v = (s + TB <= 32u ? d[i] >> s
                                                  : __builtin_amdgcn_alignbit(d[i + 1u], d[i], s)) &
                                   ((1u << TB) - 1u);
                x |= v << (8u * j);
            }
            o[q] = lm_lookup4<TB>(x, tv);
        }
        return make_uint4(o[0], o[1], o[2], o[3]);
    }
}
/* One code's value, for a ragged tail. */
template <uint32_t TB, bool I8>
__device__ __forceinline__ float lm_code1(const uint8_t* __restrict__ row, uint32_t c,
                                          const uint32_t (&tv)[16]) {
    if constexpr (TB == 0) {
        return code1<I8>(row[c]);
    } else {
        const uint32_t bit = c * TB;
        const uint32_t w = (uint32_t)row[bit >> 3] | (uint32_t)row[(bit >> 3) + 1] << 8;
        const uint32_t code = (w >> (bit & 7u)) & ((1u << TB) - 1u);
        return (float)(int8_t)(tv[code / 4u] >> (8u * (code % 4u)));
    }
}

// MT token rows against RB weight rows per wave: a speculative verify step (M = 1 + n_spec) reads
// the weight once for all MT rows instead of falling to r4d_gemm_fp8a8_nt_m16.
//
// Each wave reads all of X for its rows, so past MT = 4 the L1 traffic of X (MT x 32 bytes per 16
// weight bytes) is the cost, not HBM or registers. RB consecutive rows per wave divide it by RB.
//
// The E4M3 decode is hoisted out of the token loop. The a0/a1 split (even columns, odd columns, per
// lane, one wave reduction) keeps row_dot's summation order, so MT = 1 is bit-identical to row_dot
// and RB changes only which wave owns a row.
template <uint32_t MT, uint32_t RB, bool SROW, bool I8 = false, uint32_t TB = 0>
__device__ __forceinline__ void rows_dot_mt(const uint8_t* __restrict__ W,
                                            const uint16_t* __restrict__ S,
                                            const uint16_t* __restrict__ X, uint32_t a_ld,
                                            uint32_t row0, uint32_t rows, uint32_t cols,
                                            uint32_t w_ld, uint32_t sc_cols, uint32_t m_real,
                                            int lane, float* __restrict__ acc) {
    uint32_t tv[16];
    lm_table<TB>(W, rows, w_ld, tv);
    const uint32_t stride = kWave * 16u;
    const uint32_t full = (cols / stride) * stride;
    // The activation row a token slot reads. Above kMaxMt, M is rounded up to a multiple of four and
    // X holds only `m_real` rows: a padded slot re-reads the last real row and its outputs are
    // dropped. At or below kMaxMt the test folds away.
    auto xrow = [m_real](uint32_t m) -> uint32_t {
        return (MT > kMaxMt && m >= m_real) ? m_real - 1u : m;
    };
    const uint8_t*  w[RB];
    const uint16_t* srow[RB];
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) {
        // Clamped, not skipped: a tail wave recomputes the last row and the store drops it, keeping
        // the loop uniform.
        const uint32_t rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
        w[r] = W + (size_t)rr * w_ld;
        srow[r] = SROW ? (const uint16_t*)(w[r] + (TB ? cols / 8u * TB : cols))
                       : S + (size_t)(rr / kBlock) * sc_cols;
    }
#pragma unroll
    for (uint32_t i = 0; i < RB * MT; ++i) acc[i] = 0.0f;

    /* At MT <= 2 (kPre) each pass's weight chunk, scale and activation are loaded one pass ahead,
     * so the loads run under the previous pass's arithmetic; otherwise the compiler waits on them
     * before converting anything. Wider tiles cannot hold the activation a pass (MT = 32 would
     * spill), so they load in the pass. The last pass reloads its own chunk rather than reading
     * past the row. */
    constexpr bool kPre = MT <= 2u;
    [[maybe_unused]] uint4 wn[RB];
    [[maybe_unused]] uint32_t sn[RB];
    [[maybe_unused]] uint4 xn[kPre ? MT : 1][2];
    auto load = [&](uint32_t c) {
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) {
            wn[r] = lm_codes16<TB>(w[r], c, tv);
            sn[r] = srow[r][c / kBlock];
        }
        if constexpr (kPre) {
#pragma unroll
            for (uint32_t m = 0; m < MT; ++m) {
                const uint4* xm = (const uint4*)(X + (size_t)xrow(m) * a_ld + c);
                xn[m][0] = xm[0];
                xn[m][1] = xm[1];
            }
        }
    };
    if constexpr (kPre)
        if ((uint32_t)lane * 16u < full) load((uint32_t)lane * 16u);
    for (uint32_t c = (uint32_t)lane * 16u; c < full; c += stride) {
        float wf[RB][16];
        float sc[RB];
        [[maybe_unused]] uint4 wc[kPre ? RB : 1];
        [[maybe_unused]] uint32_t scr[kPre ? RB : 1];
        [[maybe_unused]] uint4 xc[kPre ? MT : 1][2];
        if constexpr (kPre) {
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) { wc[r] = wn[r]; scr[r] = sn[r]; }
#pragma unroll
            for (uint32_t m = 0; m < MT; ++m) { xc[m][0] = xn[m][0]; xc[m][1] = xn[m][1]; }
            load(c + stride < full ? c + stride : c);
        }
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) {
            uint4 packed;
            if constexpr (kPre) packed = wc[r];
            else packed = lm_codes16<TB>(w[r], c, tv);
            const uint32_t* pw = (const uint32_t*)&packed;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const v2f e01 = code_pk<I8, false>(pw[k]);
                const v2f e23 = code_pk<I8, true>(pw[k]);
                wf[r][k * 4 + 0] = e01.x;
                wf[r][k * 4 + 1] = e01.y;
                wf[r][k * 4 + 2] = e23.x;
                wf[r][k * 4 + 3] = e23.y;
            }
            if constexpr (kPre) sc[r] = bscale(scr[r]);
            else sc[r] = bscale(srow[r][c / kBlock]);
        }
#pragma unroll
        for (uint32_t m = 0; m < MT; ++m) {
            /* The sixteen bf16 as eight dwords, widened by a shift (even element) and a mask (odd).
             * Read a uint16_t at a time, the compiler spends two instructions per element. */
            uint4 x0, x1;
            if constexpr (kPre) {
                x0 = xc[m][0];
                x1 = xc[m][1];
            } else {
                const uint4* xm = (const uint4*)(X + (size_t)xrow(m) * a_ld + c);
                x0 = xm[0];
                x1 = xm[1];
            }
            const uint32_t xw[8] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };
            float xv[16];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                xv[2 * k]     = __uint_as_float(xw[k] << 16);
                xv[2 * k + 1] = __uint_as_float(xw[k] & 0xffff0000u);
            }
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) {
                float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                for (int k = 0; k < 16; k += 2) {
                    a0 = fmaf(wf[r][k],     xv[k],     a0);
                    a1 = fmaf(wf[r][k + 1], xv[k + 1], a1);
                }
                acc[r * MT + m] = fmaf(a0 + a1, sc[r], acc[r * MT + m]);
            }
        }
    }
    for (uint32_t c = full + (uint32_t)lane; c < cols; c += kWave) {
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) {
            const float wv = lm_code1<TB, I8>(w[r], c, tv), s = bscale(srow[r][c / kBlock]);
#pragma unroll
            for (uint32_t m = 0; m < MT; ++m)
                acc[r * MT + m] =
                    fmaf(wv * __uint_as_float((uint32_t)X[(size_t)xrow(m) * a_ld + c] << 16), s,
                         acc[r * MT + m]);
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < RB * MT; ++i) acc[i] = r4d_wave_sum(acc[i]);
}

// bf16 for a GEMM output, f32 for logits (a bf16 logit's 8-bit mantissa is too coarse for a
// nucleus cut; see r4d_model.cpp).
template <class OT> __device__ __forceinline__ OT mvm_store(float v);
template <> __device__ __forceinline__ uint16_t mvm_store<uint16_t>(float v) { return f32_to_bf16(v); }
template <> __device__ __forceinline__ float mvm_store<float>(float v) { return v; }

template <uint32_t MT, bool SROW, class OT, uint32_t RB = 1, bool I8 = false, uint32_t TB = 0>
__global__ __launch_bounds__(kDenseThreads) void mvm_kernel(
    const uint8_t* __restrict__ W, const uint16_t* __restrict__ S,
    const uint16_t* __restrict__ X, uint32_t a_ld, OT* __restrict__ out, uint32_t c_ld,
    uint32_t rows, uint32_t cols, uint32_t w_ld, uint32_t sc_cols, uint32_t m_real) {
    const int tid = threadIdx.x;
    const int lane = tid & (kWave - 1);
    const int wave = tid / kWave;
    const uint32_t step = gridDim.x * kDenseWaves * RB;
    for (uint32_t r0 = ((uint32_t)blockIdx.x * kDenseWaves + (uint32_t)wave) * RB; r0 < rows;
         r0 += step) {
        float acc[RB * MT];
        rows_dot_mt<MT, RB, SROW, I8, TB>(W, S, X, a_ld, r0, rows, cols, w_ld, sc_cols, m_real,
                                          lane, acc);
        // C is [M][N]: lane 0 writes MT scattered values `c_ld` apart, negligible against the
        // weight read. Rows m >= m_real are padding (M rounded up) and are discarded.
        if (lane == 0) {
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) {
                if (r0 + r >= rows) break;      /* the clamped duplicate, dropped here */
#pragma unroll
                for (uint32_t m = 0; m < MT; ++m)
                    if (m < m_real) out[(size_t)m * c_ld + r0 + r] = mvm_store<OT>(acc[r * MT + m]);
            }
        }
    }
}

template <uint32_t RB, class XV>
__global__ __launch_bounds__(kDenseThreads) void mv_rb_kernel(
    const uint8_t* __restrict__ W, const uint16_t* __restrict__ S, XV xv,
    uint16_t* __restrict__ out, uint32_t rows, uint32_t cols, uint32_t w_ld, uint32_t sc_cols) {
    const int tid = threadIdx.x;
    const int lane = tid & (kWave - 1);
    const int wave = tid / kWave;
    const uint32_t step = gridDim.x * kDenseWaves * RB;
    for (uint32_t r0 = ((uint32_t)blockIdx.x * kDenseWaves + (uint32_t)wave) * RB; r0 < rows;
         r0 += step) {
        const uint32_t nr = min(RB, rows - r0);
        float acc[RB];
        rows_dot<RB>(W, S, xv, r0, nr, cols, w_ld, sc_cols, lane, acc);
        if (lane == 0) {
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r)
                if (r < nr) out[r0 + r] = f32_to_bf16(acc[r]);
        }
    }
}

template <class XV>
__global__ __launch_bounds__(kDenseThreads) void mv_kernel(
    const uint8_t* __restrict__ W, const uint16_t* __restrict__ S, XV xv,
    uint16_t* __restrict__ out, uint32_t rows, uint32_t cols, uint32_t w_ld, uint32_t sc_cols) {
    const int tid = threadIdx.x;
    const int lane = tid & (kWave - 1);
    const int wave = tid / kWave;
    for (uint32_t row = blockIdx.x * kDenseWaves + wave; row < rows;
         row += gridDim.x * kDenseWaves) {
        const float total = row_dot(W, S, xv, row, cols, w_ld, sc_cols, lane);
        if (lane == 0) out[row] = f32_to_bf16(total);
    }
}

// Default RB: wide rows already have enough loads in flight, so blocking would only remove waves;
// narrow rows cannot hide a load behind two inner iterations, so block more when rows are few.
uint32_t rb_default(uint32_t rows, uint32_t cols) {
    if (cols > 2048u) return 1u;
    return rows >= 8192u ? 2u : 4u;
}

// ---------------------------------------------------------------------------- the lm head, on WMMA
//
// The scalar matvec's rate falls steeply with M (see rows_dot_mt); this is the fp8a8 tile's shape
// for a bf16 activation, one weight read for up to 64 rows (e.g. eight sequences at DFlash2
// depth 7).
//
// No precision is lost: the bf16 WMMA accumulates in f32, the activation is already bf16, an E4M3
// code is exact in bf16 (its low sixteen f32 mantissa bits are zero), and the block scale is
// applied after each 128-wide K block's accumulation. Only the summation order differs from the
// matvec.
//
// No re-layout is needed (r4d_fp8_wmma.h's map): with c = l % 16 and h = l / 16 the B operand
// wants B[8h + j][c] = W[c][8h + j], eight consecutive K bytes of one weight row, and A wants
// A[c][8h + j], sixteen contiguous bytes; both use the same K labelling. Accumulator slot j of
// lane l is D[8h + j][c], so a lane owns one output column and one block scale per K block.
typedef short lm_s8v __attribute__((ext_vector_type(8)));
typedef unsigned lm_u4v __attribute__((ext_vector_type(4)));

/* Two decoded codes as one bf16 pair: the high halves of both floats, by one v_perm_b32. A code
 * decodes to a float whose low sixteen bits are zero (E4M3: three mantissa bits, int8: seven), so
 * its bf16 is exactly the high half and no rounding is needed. Building it from f32_to_bf16 gives
 * the same bits but more instructions. The floats are read with __float_as_uint by value:
 * __builtin_bit_cast of a vector element miscompiles. */
__device__ __forceinline__ uint32_t lm_bf16_pair(v2f p) {
    return __builtin_amdgcn_perm(__float_as_uint(p.y), __float_as_uint(p.x), 0x07060302u);
}

template <bool I8 = false>
__device__ __forceinline__ lm_s8v e4m3x8_to_bf16(uint32_t lo, uint32_t hi) {
    const v2f a01 = code_pk<I8, false>(lo), a23 = code_pk<I8, true>(lo);
    const v2f b01 = code_pk<I8, false>(hi), b23 = code_pk<I8, true>(hi);
    lm_u4v r;
    r.x = lm_bf16_pair(a01);
    r.y = lm_bf16_pair(a23);
    r.z = lm_bf16_pair(b01);
    r.w = lm_bf16_pair(b23);
    return __builtin_bit_cast(lm_s8v, r);
}

// MT is ceil(M / 16): the 16-row activation tiles sharing one weight read.
//
// The weight is staged through LDS. Read straight from global, each wave would issue sixteen
// 8-byte requests per K step whose 128 B lines must stay resident for the next seven steps: a
// 16 KiB working set per workgroup against a 32 KiB L1, which thrashes. The LDS pitch is
// lds_pitch<FK>() (r4d_fp8_wmma.h), so sixteen rows land on distinct banks.
//
// FK (= FKB * 128) is the K one staging round covers, and the tile's main tunable. The stored row
// is [K bytes of E4M3][nkb bf16 scales], only 16-byte aligned (5200 bytes at K = 5120), so a
// 128-byte round straddles two cache lines on every row; that costs latency (a second transaction
// on the critical path), not traffic. FK = 256 reads three lines per 256 bytes instead of four and
// at MT <= 2 brings the tile within a few percent of stream rate. From MT = 3 the doubled LDS
// costs more workgroups per WGP than the straddle costs (lm_fkb_rule). Padding the stored stride
// to 128 bytes (+3.4% bytes at K = 5120) is slower than FKB = 2.
//
// Alternatives that lose and are not in the code:
//   - staging the activation in LDS too: a wash at FKB = 2 and over the 64 KiB limit at MT = 4;
//   - double-buffering the staged tile: matches FK at FKB = 1 (it hides the straddle rather than
//     removing it), loses at MT >= 3, and does not fit at FKB = 2. Restore it if a K that is not a
//     multiple of 256 ships;
//   - NW = 2 weight tiles per wave: slower at every M, spills at MT = 4;
//   - non-temporal weight loads: about 2x slower here (they lose coalescing on these unaligned
//     rows; the 128-aligned fragment-order planes are a different case).
//
// FK changes no arithmetic: the scale loop is nested inside the staged tile, so every element
// accumulates its K in the same order and applies each block scale after that block.
//
// MT >= 3 remains well short of stream rate, and the following were measured and ruled out: the
// register file (a seventh wave via a shallower unroll is ~12% slower), staging latency, the
// straddle, the activation loads (M = 56 and M = 64 take the same time), the WMMA issue rate, and
// the activation re-read (a wider workgroup reading it fewer times is slower: NW = 16 does not
// tile a WGP, and NW = 12 still loses). The cause is unknown.
//
// Waves per SIMD (R4D_WAVES): FKB = 2 holds 33 KB of LDS, capping a SIMD at six waves; at one
// block per round, MT = 3 targets seven and MT = 4 six. Without the target the scheduler trades
// the activation loads' overlap for occupancy, which cost 40% at MT = 4.
constexpr uint32_t lm_waves(uint32_t mt, uint32_t fkb) {
    return fkb == 2u ? 6u : mt >= 4u ? 6u : mt == 3u ? 7u : mt == 2u ? 9u : 12u;
}

// The body; lm_wmma_kernel and its chained form (RadKernelChainFn) wrap it.
template <uint32_t MT, uint32_t FKB, bool I8, uint32_t MH = 1, uint32_t TB = 0>
__device__ __forceinline__ void lm_wmma_body(
    const uint8_t* __restrict__ W, const uint16_t* __restrict__ A, uint32_t a_ld,
    float* __restrict__ C, uint32_t c_ld,
    uint32_t rows, uint32_t cols, uint32_t w_ld, uint32_t m_real) {
    constexpr uint32_t kFK    = FKB * kBlock;              /* K a staging round */
    /* Weight rows a workgroup covers: 16 a wave, the waves split into MH halves along M that share
     * the staged tile (MH = 2: 64 rows, each half four 16-row activation tiles). */
    constexpr uint32_t kRows  = kDenseWaves * 16u / MH;
    constexpr uint32_t kPitch = lds_pitch<kFK>();          /* BYTES a staged weight row */
    constexpr uint32_t kWUnits = kRows * (kFK / 16u);      /* 16-byte staging units, weight */
    constexpr bool     kPre    = MT <= 3u;                  /* activation a K step ahead, below */

    __shared__ uint8_t s_w[kRows * kPitch];

    const uint32_t tid  = (uint32_t)threadIdx.x;
    const uint32_t lane = tid & (kWave - 1u);
    const uint32_t wv   = tid / kWave;
    const uint32_t wn   = wv % (kDenseWaves / MH);                /* this wave's 16 weight rows */
    const uint32_t mo   = (wv / (kDenseWaves / MH)) * 16u * MT;   /* its M half's first row */
    const uint32_t c    = lane & 15u;          /* this lane's weight row inside its 16-row tile */
    const uint32_t h    = lane >> 4;           /* which half of the K step */
    uint32_t tv[16];
    lm_table<TB>(W, rows, w_ld, tv);

    for (uint32_t base = (uint32_t)blockIdx.x * kRows; base < rows; base += gridDim.x * kRows) {
        /* Clamped, not skipped: a tail workgroup recomputes the last row; the store drops it. */
        const uint32_t n   = base + wn * 16u + c;
        const uint32_t nc  = (n < rows) ? n : (rows - 1u);
        const uint16_t* scl =
            (const uint16_t*)(W + (size_t)nc * w_ld + (TB ? cols / 8u * TB : cols));

        f8v acc[MT];
#pragma unroll
        for (uint32_t t = 0; t < MT; ++t) acc[t] = (f8v)(0.0f);

        for (uint32_t kt = 0; kt < cols; kt += kFK) {
            /* At MT <= 3 (kPre) activation fragments are loaded one K step ahead, so their round
             * trip runs under a step's WMMAs; the first step's are issued here, while the tile
             * stages. At MT = 4 holding them costs too many VGPRs, so they load in place. The last
             * step reloads its own fragments rather than reading past the tile. */
            [[maybe_unused]] lm_s8v an[kPre ? MT : 1];
            auto aload = [&](uint32_t kf) {
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t) {
                    const uint32_t m = mo + t * 16u + c, mr = (m < m_real) ? m : (m_real - 1u);
                    an[t] = *(const lm_s8v*)(A + (size_t)mr * a_ld + kt + kf + h * 8u);
                }
            };
            if constexpr (kPre) aload(0u);
            /* The tile's block scales are read ahead of the staging barrier so they arrive while
             * the tile stages and computes, rather than as an exposed round trip after the WMMAs. */
            uint16_t sraw[FKB];
#pragma unroll
            for (uint32_t sb = 0; sb < FKB; ++sb) sraw[sb] = scl[(kt + sb * kBlock) / kBlock];
            /* Before the store: the previous tile's fragment reads may still be in flight. Both
             * barriers order LDS only (no thread reads global memory another writes); on gfx12
             * __syncthreads() would also wait for the scale and fragment loads in flight. */
            lds_barrier();
/* Staging loads in flight is a register decision: a round is kFK/32 uint4 units per thread (eight
 * at FKB = 2), and a full unroll holds them all live. At MT >= 3 that spills, so the unroll
 * narrows. */
#pragma unroll MT >= 3 ? 4 : 8
            for (uint32_t i = 0; i < kWUnits / kDenseThreads; ++i) {
                const uint32_t u = tid + i * kDenseThreads;
                const uint32_t r = u / (kFK / 16u), off = (u % (kFK / 16u)) * 16u;
                const uint32_t g = base + r;
                const uint8_t* src = W + (size_t)((g < rows) ? g : (rows - 1u)) * w_ld;
                *(uint4*)(s_w + (size_t)r * kPitch + off) = lm_codes16<TB>(src, kt + off, tv);
            }
            lds_barrier();

            /* One scale block at a time inside the staged tile, so each block scale multiplies the
             * same partial a one-block round forms. */
#pragma unroll
            for (uint32_t sb = 0; sb < FKB; ++sb) {
                f8v part[MT];
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t) part[t] = (f8v)(0.0f);
/* Four K steps in flight below MT = 4; at MT = 4 the registers (249 VGPRs) favour two. */
#pragma unroll MT >= 4 ? 2 : 4
                for (uint32_t ks = 0; ks < kBlock; ks += 16u) {
                    const uint32_t kf = sb * kBlock + ks;          /* K inside the staged tile */
                    [[maybe_unused]] lm_s8v ac[kPre ? MT : 1];
                    if constexpr (kPre) {
#pragma unroll
                        for (uint32_t t = 0; t < MT; ++t) ac[t] = an[t];
                        aload(kf + 16u < kFK ? kf + 16u : kf);
                    }
                    const i2v raw = frag<kFK>(s_w, wn * 16u, kf, lane);
                    const lm_s8v bf =
                        e4m3x8_to_bf16<I8>((uint32_t)raw[0], (uint32_t)raw[1]);
/* The activation load is unconditional: a padding row re-reads the last real row and a select
 * zeroes it. A guarded load would sit in its own branch and serialise the loads (MT round trips
 * per K step). */
#pragma unroll
                    for (uint32_t t = 0; t < MT; ++t) {
                        const uint32_t m = mo + t * 16u + c;
                        [[maybe_unused]] const uint32_t mr = (m < m_real) ? m : (m_real - 1u);
                        lm_s8v af;
                        if constexpr (kPre) af = ac[t];
                        else af = *(const lm_s8v*)(A + (size_t)mr * a_ld + kt + kf + h * 8u);
                        if (m >= m_real) af = (lm_s8v)(short)0;
                        part[t] =
                            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(af, bf, part[t]);
                    }
                }
                const float sc = bscale(sraw[sb]);
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t)
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[t][j] = fmaf(part[t][j], sc, acc[t][j]);
            }
        }

        if (n >= rows) continue;
#pragma unroll
        for (uint32_t t = 0; t < MT; ++t)
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t m = mo + t * 16u + h * 8u + j;
                if (m < m_real) C[(size_t)m * c_ld + n] = acc[t][j];
            }
    }
}

#define R4D_LMW_PARAMS const uint8_t* __restrict__ W, const uint16_t* __restrict__ A,           \
    uint32_t a_ld, float* __restrict__ C, uint32_t c_ld, uint32_t rows, uint32_t cols,          \
    uint32_t w_ld, uint32_t m_real
template <uint32_t MT, uint32_t FKB, bool I8 = false, uint32_t MH = 1, uint32_t TB = 0>
__global__ __launch_bounds__(kDenseThreads) R4D_WAVES(lm_waves(MT, FKB)) void lm_wmma_kernel(
    R4D_LMW_PARAMS) {
    lm_wmma_body<MT, FKB, I8, MH, TB>(W, A, a_ld, C, c_ld, rows, cols, w_ld, m_real);
}
template <uint32_t MT, uint32_t FKB, bool I8 = false, uint32_t MH = 1, uint32_t TB = 0>
__global__ __launch_bounds__(kDenseThreads) R4D_WAVES(lm_waves(MT, FKB)) void r4dchain_lm_wmma(
    R4D_LMW_PARAMS) {
    r4d_chain_wait();
    lm_wmma_body<MT, FKB, I8, MH, TB>(W, A, a_ld, C, c_ld, rows, cols, w_ld, m_real);
    r4d_chain_done();
}
#undef R4D_LMW_PARAMS
}  // namespace

// C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T for M <= kMaxMt, one weight read for all M rows.
//
//   a    [M][K]                     bf16, `a_ld` elements a row (0 = K)
//   wq   [N][K]                     E4M3, row-major, `w_ld` bytes a row (0 = K)
//   ws   [ceil(N/128)][ceil(K/128)] bf16, `sw_ld` elements a row (0 = ceil(K/128))
//   c    [M][N]                     bf16, `c_ld` elements a row (0 = N)
//
// M is a template argument with every value 1..8 instantiated, so e.g. M = 3 does not pay for a
// padded fourth row.
extern "C" int r4d_gemm_fp8a16_nt_mt(long a, long wq, long ws, long c,
                                     int M, int K, int N, int a_ld, int c_ld,
                                     int w_ld, int sw_ld, long stream) {
    if (!a || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (M > (int)kMaxMt) return -3;
    if (K % (int)kBlock) return -4;
    if (!w_ld) w_ld = K;
    if (!a_ld) a_ld = K;
    if (!c_ld) c_ld = N;
    const int nkb = K / (int)kBlock;
    if (!sw_ld) sw_ld = nkb;

    const uint32_t rows = (uint32_t)N, cols = (uint32_t)K;
    const uint32_t rb = mvm_rb(M);
    const uint32_t blocks =
        std::min<uint32_t>(2048u, (rows + kDenseWaves * rb - 1) / (kDenseWaves * rb));
#define R4D_MVMT_RB(NM, RBV)                                                                    \
    hipLaunchKernelGGL((mvm_kernel<NM, false, uint16_t, RBV>), dim3(blocks), dim3(kDenseThreads), \
                       0,                                                                        \
                       (hipStream_t)stream, (const uint8_t*)wq, (const uint16_t*)ws,             \
                       (const uint16_t*)a, (uint32_t)a_ld, (uint16_t*)c, (uint32_t)c_ld,         \
                       rows, cols, (uint32_t)w_ld, (uint32_t)sw_ld, (uint32_t)M)
#define R4D_MVMT(NM)                                                                            \
    do { if (rb == 4) R4D_MVMT_RB(NM, 4);                                                       \
         else if (rb == 2) R4D_MVMT_RB(NM, 2);                                                  \
         else R4D_MVMT_RB(NM, 1); } while (0)
    switch (M) {
        case 1: R4D_MVMT(1); break;
        case 2: R4D_MVMT(2); break;
        case 3: R4D_MVMT(3); break;
        case 4: R4D_MVMT(4); break;
        case 5: R4D_MVMT(5); break;
        case 6: R4D_MVMT(6); break;
        case 7: R4D_MVMT(7); break;
        case 8: R4D_MVMT(8); break;
        default: return -3;
    }
#undef R4D_MVMT
#undef R4D_MVMT_RB
    return hipGetLastError() == hipSuccess ? 0 : -6;
}


// C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, f32 out, with the scales in each row's own tail.
//
// The fp8 lm_head: halves the bytes of the largest single weight read in a decode step (a bf16
// head already runs at stream rate).
//
// The scales ride in the row because `logits_gemm` has one weight operand, and lm_head is
// row-sharded under TP: the loader takes a rank's rows as a plain byte range, so a separate scale
// plane appended to the stream would be left behind. Each row repeats its block's scales (80 bytes
// per 5120 at K = 5120, 1.6%).
//
//   a    [M][K]          bf16, `a_ld` elements a row (0 = K)
//   wq   [N][lm_row_bytes(K)]  E4M3 codes, nkb bf16 scales, pad; `w_ld` BYTES a row (0 = that)
//   c    [M][N]          f32,  `c_ld` elements a row (0 = N)
//
// The WMMA path takes M > 2 in passes of up to 128 rows, each reading the weight once (past 64
// rows in two M halves over one staged tile); the matvec takes M <= 2.
//
// This row must serve every M: all bands of one op read the same stored bytes, so a band falling
// to another kernel (e.g. the bf16 projection) would misread the E4M3 codes.
// core/build/rad_builder.cpp refuses such a split at declare time.
template <bool I8, uint32_t TB = 0>
static int logits_gemm_8(long a, long wq, long c, int M, int K, int N, int a_ld, int c_ld,
                         int w_ld, long stream, bool chain = false);

extern "C" int r4d_logits_gemm_fp8(long a, long wq, long c, int M, int K, int N,
                                   int a_ld, int c_ld, int w_ld, long stream) {
    return logits_gemm_8<false>(a, wq, c, M, K, N, a_ld, c_ld, w_ld, stream);
}

// The int8 head: the same stored row (codes, bf16 scales in the tail, pad) with i8 codes and each
// row's own scale per 128 columns, where the E4M3 head repeats its block's.
extern "C" int r4d_logits_gemm_i8(long a, long wq, long c, int M, int K, int N,
                                  int a_ld, int c_ld, int w_ld, long stream) {
    return logits_gemm_8<true>(a, wq, c, M, K, N, a_ld, c_ld, w_ld, stream);
}

/* `chain`: the WMMA pass's chained form (the plugin entry, where the row resolved with `chain`);
 * the matvec below three rows stays plain. */
template <bool I8, uint32_t TB>
static int logits_gemm_8(long a, long wq, long c, int M, int K, int N, int a_ld, int c_ld,
                         int w_ld, long stream, bool chain) {
    if (!a || !wq || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (K % (int)kBlock) return -4;
    if (!a_ld) a_ld = K;
    if (!c_ld) c_ld = N;
    if (!w_ld) w_ld = (int)r4d_fp8::lm_row_bytes(K);
    /* Both paths load the weight as uint4 at `row * w_ld + 16k`, so w_ld must be a multiple of 16
     * (the layout pads it; see r4d_fp8_lm.h). */
    if (w_ld % 16) return -5;

    const uint32_t rows = (uint32_t)N, cols = (uint32_t)K;

    /* WMMA from M = 3, the matvec at or below. The tile's rate is nearly flat in M; the matvec is
     * marginally ahead at M = 2 and clearly at M = 1. Both read the same stored bytes. */
    if (M > 2) {
        /* Passes of up to 128 rows. Up to 64, one tile of four WMMA accumulator tiles (32
         * registers) a wave; past 64, the same four a wave in two M halves over one staged weight
         * tile (MH = 2), so the head is read once a pass. */
        for (int m0 = 0; m0 < M; m0 += 128) {
            const int Mc = (M - m0) < 128 ? (M - m0) : 128;
            if (Mc > 64) {
                uint32_t fkb = lm_fkb_rule(4u);
                while (fkb > 1u && (cols % (fkb * kBlock)) != 0u) fkb >>= 1;
                const uint32_t per = kDenseWaves * 16u / 2u;
                const uint32_t blocks = std::min<uint32_t>(4096u, (rows + per - 1u) / per);
                const uint16_t* ac = (const uint16_t*)a + (size_t)m0 * a_ld;
                float* cc = (float*)c + (size_t)m0 * c_ld;
                if (fkb == 2u)
                    hipLaunchKernelGGL((chain ? r4dchain_lm_wmma<4, 2, I8, 2, TB> : lm_wmma_kernel<4, 2, I8, 2, TB>),
                                       dim3(blocks), dim3(kDenseThreads), 0,
                                       (hipStream_t)stream, (const uint8_t*)wq, ac, (uint32_t)a_ld, cc,
                                       (uint32_t)c_ld, rows, cols, (uint32_t)w_ld, (uint32_t)Mc);
                else
                    hipLaunchKernelGGL((chain ? r4dchain_lm_wmma<4, 1, I8, 2, TB> : lm_wmma_kernel<4, 1, I8, 2, TB>),
                                       dim3(blocks), dim3(kDenseThreads), 0,
                                       (hipStream_t)stream, (const uint8_t*)wq, ac, (uint32_t)a_ld, cc,
                                       (uint32_t)c_ld, rows, cols, (uint32_t)w_ld, (uint32_t)Mc);
                if (hipGetLastError() != hipSuccess) return -6;
                continue;
            }
            const uint32_t mt = (uint32_t)((Mc + 15) / 16);
            /* FKB = 4 would need kRows * (512 + 8) = 66560 bytes of LDS, over the 64 KiB limit. */
            uint32_t fkb = lm_fkb_rule(mt);
            /* FK must divide K, or a staging round would read the row's scales as codes. */
            while (fkb > 1u && (cols % (fkb * kBlock)) != 0u) fkb >>= 1;
            const uint32_t per = kDenseWaves * 16u;      /* weight rows a workgroup covers */
            const uint32_t blocks = std::min<uint32_t>(4096u, (rows + per - 1u) / per);
            const uint16_t* ac = (const uint16_t*)a + (size_t)m0 * a_ld;
            float* cc = (float*)c + (size_t)m0 * c_ld;
#define R4D_LMW2(MTV, FKV)                                                                  \
    hipLaunchKernelGGL((chain ? r4dchain_lm_wmma<MTV, FKV, I8, 1, TB> : lm_wmma_kernel<MTV, FKV, I8, 1, TB>), \
                       dim3(blocks), dim3(kDenseThreads), 0,                                 \
                       (hipStream_t)stream, (const uint8_t*)wq, ac, (uint32_t)a_ld, cc,      \
                       (uint32_t)c_ld, rows, cols, (uint32_t)w_ld, (uint32_t)Mc)
#define R4D_LMW(MTV) do { if (fkb == 2u) R4D_LMW2(MTV, 2); else R4D_LMW2(MTV, 1); } while (0)
            switch (mt) {
                case 1:  R4D_LMW(1); break;
                case 2:  R4D_LMW(2); break;
                case 3:  R4D_LMW(3); break;
                default: R4D_LMW(4); break;
            }
#undef R4D_LMW
#undef R4D_LMW2
            if (hipGetLastError() != hipSuccess) return -6;
        }
        return 0;
    }

    const uint32_t blocks = std::min<uint32_t>(2048u, (rows + kDenseWaves - 1) / kDenseWaves);
#define R4D_LGF8(NM)                                                                            \
    hipLaunchKernelGGL((mvm_kernel<NM, true, float, 1, I8, TB>), dim3(blocks), dim3(kDenseThreads),  \
                       0, (hipStream_t)stream, (const uint8_t*)wq, (const uint16_t*)nullptr,     \
                       (const uint16_t*)a, (uint32_t)a_ld, (float*)c, (uint32_t)c_ld,            \
                       rows, cols, (uint32_t)w_ld, 0u, (uint32_t)M)
    if (M == 1) R4D_LGF8(1);
    else        R4D_LGF8(2);
#undef R4D_LGF8
    return hipGetLastError() == hipSuccess ? 0 : -6;
}

// C[1,N] = A[1,K] @ dequant(Wq)[N,K]^T.
//
//   a    [K]                        bf16
//   wq   [N][K]                     E4M3, row-major, `w_ld` bytes a row (0 = K)
//   ws   [ceil(N/128)][ceil(K/128)] bf16, `sw_ld` elements a row (0 = ceil(K/128))
//   c    [N]                        bf16
//
// No LDS staging: with uint4 loads, 16 lanes would hit the same bank, and `a` stays in L1 anyway.
//
// Returns 0, or a negative code for a shape it does not serve.
extern "C" int r4d_gemm_fp8a16_nt_m1(long a, long wq, long ws, long c,
                                     int M, int K, int N, int w_ld, int sw_ld, int rb,
                                     long stream) {
    if (!a || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (M > 1) return -3;
    // `srow[c / 128]` reads one scale per 128 columns; a ragged last block would read past the
    // plane.
    if (K % (int)kBlock) return -4;
    if (rb != 0 && rb != 1 && rb != 2 && rb != 4 && rb != 8) return -5;
    if (!w_ld) w_ld = K;
    const int nkb = K / (int)kBlock;
    if (!sw_ld) sw_ld = nkb;

    const Bf16V xv{ (const uint16_t*)a };
    const uint32_t rows = (uint32_t)N, cols = (uint32_t)K;
    uint32_t r = rb ? (uint32_t)rb : rb_default(rows, cols);

    if (r > 1u) {
        const uint32_t per = kDenseWaves * r;
        const uint32_t blocks = std::min<uint32_t>(2048u, (rows + per - 1) / per);
#define R4D_MVRB(NB)                                                                          \
    hipLaunchKernelGGL((mv_rb_kernel<NB, Bf16V>), dim3(blocks), dim3(kDenseThreads), 0,        \
                       (hipStream_t)stream, (const uint8_t*)wq, (const uint16_t*)ws, xv,       \
                       (uint16_t*)c, rows, cols, (uint32_t)w_ld, (uint32_t)sw_ld)
        if (r == 2u)      { R4D_MVRB(2); }
        else if (r == 4u) { R4D_MVRB(4); }
        else              { R4D_MVRB(8); }
#undef R4D_MVRB
        return hipGetLastError() == hipSuccess ? 0 : -6;
    }
    const uint32_t blocks = std::min<uint32_t>(2048u, (rows + kDenseWaves - 1) / kDenseWaves);
    hipLaunchKernelGGL((mv_kernel<Bf16V>), dim3(blocks), dim3(kDenseThreads), 0,
                       (hipStream_t)stream, (const uint8_t*)wq, (const uint16_t*)ws, xv,
                       (uint16_t*)c, rows, cols, (uint32_t)w_ld, (uint32_t)sw_ld);
    return hipGetLastError() == hipSuccess ? 0 : -6;
}

// ============================================================== the plugin ABI entry point
// Helpers from libr4d/r4d_args.h; see the note in r4d_model_bf16.hip.
enum { LG_X = 0, LG_W, LG_Y };

/* ------------------------------------------------------------------ logits_gemm, fp8 weight
 *
 * The weight arrives as [n_vocab][r4d_fp8::lm_row_bytes(n_embd)]: E4M3 codes, the row's bf16
 * scales in its tail, then the 16-byte alignment pad (see r4d_rad_layout_fp8_logits).
 */
static int rad_logits_gemm_8(const RadArgs* a, RadStream s, bool i8) {
    const RadTensor* x = r4d_opt(a, LG_X);
    const RadTensor* w = r4d_opt(a, LG_W);
    const RadTensor* y = r4d_opt(a, LG_Y);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (w->dtype != (i8 ? RAD_I8 : RAD_F8E4M3)) return R4D_NO(RAD_E_DTYPE);
    if (y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);

    const long long n_vocab = r4d_geti_or(a, "n_vocab", r4d_dim(w, 0));
    const long long n_embd  = r4d_geti_or(a, "n_embd", 0);
    if (n_vocab <= 0 || n_embd <= 0) return R4D_NO(RAD_E_SHAPE);
    if (n_embd % 128) return R4D_NO(RAD_E_SHAPE);
    const long long stride = r4d_fp8::lm_row_bytes(n_embd);
    /* Check the declared shape: core/runtime/ctx.cpp builds the RadTensor from `decl.shape`, so it
     * arrives as [n_vocab][n_embd] even though the plane is wider. The stored stride is derived
     * above by the layout hook's rule; the engine's layout-tag check guarantees they agree. */
    if (w->rank != 2 || w->shape[0] != n_vocab || w->shape[1] != n_embd)
        return R4D_NO(RAD_E_SHAPE);

    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n_embd, &M);
    const long long y_ld = r4d_rows_pitch(y, n_vocab, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(w->data, 16)) return R4D_NO(RAD_E_ALIGN);

    /* The chained form at decode heights, where the row resolved with `chain`. */
    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
    return (i8 ? logits_gemm_8<true> : logits_gemm_8<false>)(
               r4d_addr(x), r4d_addr(w), r4d_addr(y), (int)M, (int)n_embd, (int)n_vocab,
               (int)x_ld, (int)y_ld, (int)stride, (long)(uintptr_t)s, ch) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* The int8 head: the same operands and stored row, i8 codes (r4d_rad_layout_i8_logits). */
extern "C" int r4d_rad_logits_gemm_i8(const RadArgs* a, RadStream s) {
    return rad_logits_gemm_8(a, s, true);
}

/* An IQR head (r4d_iqr.h's lm form): the int8 head's kernels with each sixteen codes read at their
 * width and looked up in the row form's table (lm_codes16), so the arithmetic is the int8 head's
 * over the same values. */
extern "C" int r4d_rad_logits_gemm_iqr(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, LG_X);
    const RadTensor* w = r4d_opt(a, LG_W);
    const RadTensor* y = r4d_opt(a, LG_Y);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || y->dtype != RAD_F32 || !r4d_iqr::is_weight_dtype(w->dtype))
        return R4D_NO(RAD_E_DTYPE);
    const uint32_t word = w->dtype & r4d_iqr::kWordMask;
    if (!r4d_iqr::lm_built(word)) return R4D_NO(RAD_E_DTYPE);
    const r4d_iqr::Form f = r4d_iqr::form(word);
    const long long n_vocab = r4d_geti_or(a, "n_vocab", r4d_dim(w, 0));
    const long long n_embd  = r4d_geti_or(a, "n_embd", 0);
    if (n_vocab <= 0 || n_embd <= 0 || n_embd % 128) return R4D_NO(RAD_E_SHAPE);
    if (w->rank != 2 || w->shape[0] != n_vocab || w->shape[1] != n_embd) return R4D_NO(RAD_E_SHAPE);
    const long long stride = r4d_iqr::lm_row(f, n_embd);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n_embd, &M);
    const long long y_ld = r4d_rows_pitch(y, n_vocab, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(w->data, 256)) return R4D_NO(RAD_E_ALIGN);
    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
    int (*go)(long, long, long, int, int, int, int, int, int, long, bool) =
        f.bits == 2u ? logits_gemm_8<true, 2> : f.bits == 3u ? logits_gemm_8<true, 3>
      : f.bits == 4u ? logits_gemm_8<true, 4> : f.bits == 5u ? logits_gemm_8<true, 5>
                     : logits_gemm_8<true, 6>;
    return go(r4d_addr(x), r4d_addr(w), r4d_addr(y), (int)M, (int)n_embd, (int)n_vocab, (int)x_ld,
              (int)y_ld, (int)stride, (long)(uintptr_t)s, ch) == 0 ? RAD_OK : RAD_E_DEVICE;
}

// ============================================================== the plugin ABI entry points
//
// The shared operand reader is r4d_fp8_args.h; see libr4d/r4d_args.h.
/* ================================================================== the fp8 GEMMs */

