// Per-(row, 128-column) symmetric E4M3 quantisation of a bf16 activation, the input to the fp8a8
// GEMMs. The checkpoint's activation_scheme is "dynamic" (no stored activation scale), and a GEMM
// cannot compute one without a reduction across every M block.
//
// Output is plain row-major, unlike r4d_quant_act_i8 (which writes gemm_w4a8_nt_m64's register
// fragment order): the fp8a8 GEMMs stage both operands through the same `stage8` copy and fragment
// read, and row-major lets the activation be the B operand of the narrow instantiation.
//
// The group is fixed at 128, the checkpoint's weight-scale block width (unlike
// R4D_GEMM_W4A8_GROUP); another group would make the GEMM's fold read the wrong scale. No LDS: the
// input is already row-major per token.
//
// Not obvious:
//   - The scale is `amax / 448` (E4M3's largest finite value), a dequant multiplier like the
//     checkpoint's `weight_scale_inv`, so the GEMM's fold only multiplies.
//   - A zero block gets scale 1 and quantises to zeros.
//   - The clamp is required: `amax * (1/scale)` can land an ulp above 448, and v_cvt_pk_fp8_f32
//     returns NaN for out-of-range input instead of saturating, so every block's maximum would
//     become NaN.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_fp8_args.h"
#include "r4d_fp8_wmma.h"
#include "r4d_fwht.h"
#include "r4d_chain.h"

namespace {

constexpr uint32_t kWave = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kGroup = 128;

// One block per row; a wave owns one 128-column group, a lane four consecutive columns (one dword
// store, 128 contiguous bytes per wave, two v_cvt_pk_fp8_f32).
//
// ROT applies r4d_fwht_wave<128> in registers between the load and the absmax; the mapping (four
// values per lane) is the one it expects, so no LDS or barrier is needed, and the absmax sees the
// rotated values. The sqrt(128) factor of the unnormalised transform is not removed here:
// <Hw, Ha> = 128 <w, a>, and r4d_layout.cpp folds that into the stored weight scale at convert.
//
// I8 is the same quantiser at int8 codes (quant_act_i8g, had_quant_act_i8g): scale amax/127,
// round-to-nearest-even into -127..127, same row-major layout and per-(row, 128) scales.
//
// CHAIN brackets the body with the chained launch's wait and add (r4d_chain.h); no block returns
// early. quant_act_fp8_kernel and r4dchain_quant_act_fp8 below are the plain and chained forms.
//
// SUB is the 128-column blocks one scale covers: 1, or 2 for IQ8R's 256-wide group (int8 only),
// where a wave owns both blocks of a group, rotates each on its own (the rotation stays 128 wide)
// and takes the amax over the pair. `nkb` counts scale groups.
template <bool ROT, bool I8 = false, bool CHAIN = false, uint32_t SUB = 1>
__device__ __forceinline__ void quant_act_fp8_body(
    const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s,
    uint32_t K, uint32_t nkb, uint32_t x_ld, uint32_t q_ld, uint32_t s_ld) {
    static_assert(SUB == 1 || (SUB == 2 && I8), "a 256 group is the int8 quantiser's");
    if (CHAIN) r4d_chain_wait();
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const size_t xrow = (size_t)row * x_ld;
    const size_t qrow = (size_t)row * q_ld;

    for (uint32_t kb = wave; kb < nkb; kb += kWaves) {
        float v[SUB][4];
        float m = 0.0f;
#pragma unroll
        for (uint32_t h = 0; h < SUB; ++h) {
            const uint32_t c0 = (kb * SUB + h) * kGroup + lane * 4u;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t c = c0 + (uint32_t)j;
                v[h][j] = c < K ? __uint_as_float((uint32_t)x[xrow + c] << 16) : 0.0f;
            }
            // The rotated launcher requires K % 128 == 0: a partial block would rotate real values
            // with padding zeros, and the weight has no matching block.
            if constexpr (ROT) r4d_fwht_wave<(int)kGroup>(v[h], (int)lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(v[h][j]));
        }
        m = r4d_wave_max(m);
        constexpr float kTop = I8 ? 127.0f : 448.0f;
        const float sc = m > 0.0f ? m * (1.0f / kTop) : 1.0f;
        if (lane == 0) s[(size_t)row * s_ld + kb] = sc;
        const float inv = 1.0f / sc;
#pragma unroll
        for (uint32_t h = 0; h < SUB; ++h) {
            const uint32_t c0 = (kb * SUB + h) * kGroup + lane * 4u;
            int pk = 0;
            if constexpr (I8) {
                uint32_t b = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = (int)rintf(fminf(fmaxf(v[h][j] * inv, -127.0f), 127.0f));
                    b |= ((uint32_t)c & 0xFFu) << (8 * j);
                }
                pk = (int)b;
            } else {
                pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[h][0] * inv, -448.0f), 448.0f),
                                                     fminf(fmaxf(v[h][1] * inv, -448.0f), 448.0f),
                                                     pk, false);
                pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[h][2] * inv, -448.0f), 448.0f),
                                                     fminf(fmaxf(v[h][3] * inv, -448.0f), 448.0f),
                                                     pk, true);
            }
            // A ragged last group is written zero-padded; the GEMM's bounds guard keeps the
            // padding out of the accumulators. The caller must own K rounded up to 4 bytes of `q`.
            if (c0 < K) *(uint32_t*)(q + qrow + c0) = (uint32_t)pk;
        }
    }
    if (CHAIN) r4d_chain_done();
}

#define R4D_QAF_PARAMS const uint16_t* __restrict__ x, uint8_t* __restrict__ q,                  \
    float* __restrict__ s, uint32_t K, uint32_t nkb, uint32_t x_ld, uint32_t q_ld, uint32_t s_ld
template <bool ROT, bool I8 = false, uint32_t SUB = 1>
__global__ __launch_bounds__(kThreads) void quant_act_fp8_kernel(R4D_QAF_PARAMS) {
    quant_act_fp8_body<ROT, I8, false, SUB>(x, q, s, K, nkb, x_ld, q_ld, s_ld);
}
template <bool ROT, bool I8 = false, uint32_t SUB = 1>
__global__ __launch_bounds__(kThreads) void r4dchain_quant_act_fp8(R4D_QAF_PARAMS) {
    quant_act_fp8_body<ROT, I8, true, SUB>(x, q, s, K, nkb, x_ld, q_ld, s_ld);
}
#undef R4D_QAF_PARAMS

}  // namespace

extern "C" int r4d_quant_act_fp8_group(void) { return (int)kGroup; }

//   a  [M][K]                bf16, `x_ld` elements a row (0 = K)
//   q  [M][K]                E4M3, `q_ld` bytes a row (0 = K), rounded up to 4
//   s  [M][ceil(K/128)]      f32,  `s_ld` elements a row (0 = ceil(K/128))
//
// `rot` runs the 128-wide Hadamard first and additionally requires K % 128 == 0 (see the kernel).
// `wide` (int8 only): one scale per 256 columns (IQ8R), K a multiple of 256, `s` [M][K/256].
static int quant_act_fp8_launch(long a, long q, long s, int M, int K,
                                int x_ld, int q_ld, int s_ld, RadStream stream, bool rot,
                                bool i8 = false, bool chain = false, bool wide = false) {
    int emit_rc = RAD_OK;
    if (!a || !q || !s) return -1;
    if (M <= 0 || K <= 0) return -2;
    if (K % 4) return -3;              // a lane owns four columns; see the kernel
    if (rot && (K % (int)kGroup)) return -5;
    if (wide && (!i8 || K % (2 * (int)kGroup))) return -5;
    if (!x_ld) x_ld = K;
    if (!q_ld) q_ld = K;
    const int grp = wide ? 2 * (int)kGroup : (int)kGroup;
    const int nkb = (K + grp - 1) / grp;
    if (!s_ld) s_ld = nkb;
    /* The chained form at decode heights, where the row resolved with `chain`. */
    const bool ch = chain && M <= 64;
#define R4D_QA(ROT_, I8_, SUB_)                                                               \
    (ch ? R4D_EMIT(stream, (r4dchain_quant_act_fp8<ROT_, I8_, SUB_>), dim3((uint32_t)M),       \
                   dim3(kThreads), 0, (const uint16_t*)a, (uint8_t*)q, (float*)s, (uint32_t)K, \
                   (uint32_t)nkb, (uint32_t)x_ld, (uint32_t)q_ld, (uint32_t)s_ld)               \
        : R4D_EMIT(stream, (quant_act_fp8_kernel<ROT_, I8_, SUB_>), dim3((uint32_t)M),         \
                   dim3(kThreads), 0, (const uint16_t*)a, (uint8_t*)q, (float*)s, (uint32_t)K, \
                   (uint32_t)nkb, (uint32_t)x_ld, (uint32_t)q_ld, (uint32_t)s_ld))
    if (wide)    emit_rc = rot ? R4D_QA(true, true, 2u) : R4D_QA(false, true, 2u);
    else if (i8) emit_rc = rot ? R4D_QA(true, true, 1u) : R4D_QA(false, true, 1u);
    else         emit_rc = rot ? R4D_QA(true, false, 1u) : R4D_QA(false, false, 1u);
#undef R4D_QA
    if (emit_rc != RAD_OK) return -4;
    return (hipGetLastError() != hipSuccess) ? -4 : 0;
}

extern "C" int r4d_quant_act_fp8(long a, long q, long s, int M, int K,
                                 int x_ld, int q_ld, int s_ld, long stream) {
    return quant_act_fp8_launch(a, q, s, M, K, x_ld, q_ld, s_ld,
                                (RadStream)(uintptr_t)stream, false);
}

// ============================================================== the plugin ABI entry points
//
// The shared operand reader is r4d_fp8_args.h; see libr4d/r4d_args.h.
/* ================================================================== the fp8 activation quantiser */

static int rad_quant_act_fp8(const RadArgs* a, RadStream stream, bool rot, bool i8 = false) {
    const RadTensor* x  = r4d_opt(a, QF_X);
    const RadTensor* q  = r4d_opt(a, QF_Q);
    const RadTensor* sc = r4d_opt(a, QF_S);
    if (!x || !q || !sc) return R4D_NO(RAD_E_INVAL);
    /* Rows are dim 0 and everything after is the row, as r4d_row_pitch reads it. A GDN block's
     * out-projection input is [tokens, heads, width] (the scan kernels require that rank) and its
     * K is heads*width; one row per head would put scales on a grid the GEMM does not read.
     * `q` and `sc` are written for the GEMM and stay rank 2. */
    if (x->rank < 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = x->shape[0], K = r4d_numel(x) / (x->shape[0] > 0 ? x->shape[0] : 1);
    /* The group is fixed by the weight's scale grid: 128, or 256 for the int8 quantisers (IQ8R).
     * Any other declared group is refused. */
    const long long g0 = r4d_quant_act_fp8_group();
    const long long grp = r4d_geti_or(a, "group", g0);
    const bool wide = i8 && grp == R4D_I8_GROUP_WIDE;
    if (grp != g0 && !wide) return R4D_NO(RAD_E_SHAPE);
    const long long nkb = (K + grp - 1) / grp;
    if (q->shape[0] != M || q->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != nkb) return R4D_NO(RAD_E_SHAPE);
    const long long xp = r4d_row_pitch(x), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    if (xp < 0 || qp < 0 || sp < 0) return R4D_NO(RAD_E_STRIDE);
    const int e = quant_act_fp8_launch(r4d_addr(x), r4d_addr(q), r4d_addr(sc), (int)M, (int)K,
                                       (int)xp, (int)qp, (int)sp, stream, rot, i8,
                                       r4d_geti_or(a, "chain", 0) != 0, wide);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_quant_act_fp8(const RadArgs* a, RadStream stream) {
    return rad_quant_act_fp8(a, stream, false);
}

/* The int8 pair, for the int8 trunk GEMMs (gemm_i8a8_*). Separate ops for the same reason as the
 * rotated form: different numbers in a buffer of the same shape. */
extern "C" int r4d_rad_quant_act_i8g(const RadArgs* a, RadStream stream) {
    return rad_quant_act_fp8(a, stream, false, true);
}

extern "C" int r4d_rad_had_quant_act_i8g(const RadArgs* a, RadStream stream) {
    return rad_quant_act_fp8(a, stream, true, true);
}

// ============================================================== quant_cols_fp8
// E4M3 codes with one scale per column over every row: an attention's V for its fp8 P.V. The
// formula is quant_act_fp8's (scale amax / 448, 1 for an all-zero column, codes of x / scale
// clamped to +-448) with the group running down a column. A column's scale needs every row, so
// two passes over the rows, split into kColChunks chunks:
//   1. column amax partials, a workgroup a block of columns and a chunk of rows, the bf16
//      magnitudes compared as integers (their order is the values');
//   2. a workgroup the same block and chunk: the scales from the chunks' partials, then the codes,
//      the chunk's rows walked from its end, which the first pass read last and the last-level
//      cache still holds.
// A thread owns eight columns: one 16-byte load and one 8-byte store a row.
namespace {

constexpr int kColChunks  = 32;
constexpr int kColThreads = 128;

typedef unsigned short r4d_u16x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ r4d_u16x2 col_mag(uint32_t w) {
    return __builtin_bit_cast(r4d_u16x2, w & 0x7fff7fffu);
}

__global__ __launch_bounds__(kColThreads) void quant_cols_amax_kernel(
    const uint16_t* __restrict__ x, int M, int n, int x_ld, float* __restrict__ part) {
    const int c8 = blockIdx.x * kColThreads + threadIdx.x;
    if (c8 * 8 >= n) return;
    const int r0 = (int)((long)M * blockIdx.y / kColChunks);
    const int r1 = (int)((long)M * (blockIdx.y + 1) / kColChunks);
    const uint16_t* p = x + (long)c8 * 8;
    r4d_u16x2 m[4] = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } };
    int r = r0;
    for (; r + 8 <= r1; r += 8) {
        uint4 u[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) u[k] = *(const uint4*)(p + (long)(r + k) * x_ld);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            m[0] = __builtin_elementwise_max(m[0], col_mag(u[k].x));
            m[1] = __builtin_elementwise_max(m[1], col_mag(u[k].y));
            m[2] = __builtin_elementwise_max(m[2], col_mag(u[k].z));
            m[3] = __builtin_elementwise_max(m[3], col_mag(u[k].w));
        }
    }
    for (; r < r1; ++r) {
        const uint4 u = *(const uint4*)(p + (long)r * x_ld);
        m[0] = __builtin_elementwise_max(m[0], col_mag(u.x));
        m[1] = __builtin_elementwise_max(m[1], col_mag(u.y));
        m[2] = __builtin_elementwise_max(m[2], col_mag(u.z));
        m[3] = __builtin_elementwise_max(m[3], col_mag(u.w));
    }
    float* d = part + (long)blockIdx.y * n + c8 * 8;
    *(float4*)d = make_float4(bf16_to_f32(m[0].x), bf16_to_f32(m[0].y), bf16_to_f32(m[1].x),
                              bf16_to_f32(m[1].y));
    *(float4*)(d + 4) = make_float4(bf16_to_f32(m[2].x), bf16_to_f32(m[2].y),
                                    bf16_to_f32(m[3].x), bf16_to_f32(m[3].y));
}

template <bool HI>
__device__ __forceinline__ int col_code2(float a, float b, float ia, float ib, int old) {
    return __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(a * ia, -448.0f), 448.0f),
                                           fminf(fmaxf(b * ib, -448.0f), 448.0f), old, HI);
}

__global__ __launch_bounds__(kColThreads) void quant_cols_code_kernel(
    const uint16_t* __restrict__ x, int M, int n, int x_ld, const float* __restrict__ part,
    uint8_t* __restrict__ y, int y_ld, float* __restrict__ scale) {
    const int c8 = blockIdx.x * kColThreads + threadIdx.x;
    if (c8 * 8 >= n) return;
    float am[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int ch = 0; ch < kColChunks; ++ch) {
        const float4 a = *(const float4*)(part + (long)ch * n + c8 * 8);
        const float4 b = *(const float4*)(part + (long)ch * n + c8 * 8 + 4);
        am[0] = fmaxf(am[0], a.x); am[1] = fmaxf(am[1], a.y);
        am[2] = fmaxf(am[2], a.z); am[3] = fmaxf(am[3], a.w);
        am[4] = fmaxf(am[4], b.x); am[5] = fmaxf(am[5], b.y);
        am[6] = fmaxf(am[6], b.z); am[7] = fmaxf(am[7], b.w);
    }
    float sc[8], inv[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        sc[e] = am[e] > 0.0f ? am[e] * (1.0f / 448.0f) : 1.0f;
        inv[e] = 1.0f / sc[e];
    }
    if (blockIdx.y == 0) {
        *(float4*)(scale + c8 * 8) = make_float4(sc[0], sc[1], sc[2], sc[3]);
        *(float4*)(scale + c8 * 8 + 4) = make_float4(sc[4], sc[5], sc[6], sc[7]);
    }
    const int r0 = (int)((long)M * blockIdx.y / kColChunks);
    const int r1 = (int)((long)M * (blockIdx.y + 1) / kColChunks);
    const uint16_t* p = x + (long)c8 * 8;
    uint8_t* q = y + (long)c8 * 8;
    auto code = [&](uint4 u) {
        int lo = col_code2<false>(bf16w_lo(u.x), bf16w_hi(u.x), inv[0], inv[1], 0);
        lo = col_code2<true>(bf16w_lo(u.y), bf16w_hi(u.y), inv[2], inv[3], lo);
        int hi = col_code2<false>(bf16w_lo(u.z), bf16w_hi(u.z), inv[4], inv[5], 0);
        hi = col_code2<true>(bf16w_lo(u.w), bf16w_hi(u.w), inv[6], inv[7], hi);
        return make_uint2((uint32_t)lo, (uint32_t)hi);
    };
    int r = r1;
    for (; r - 8 >= r0; r -= 8) {
        uint4 u[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) u[k] = *(const uint4*)(p + (long)(r - 1 - k) * x_ld);
        #pragma unroll
        for (int k = 0; k < 8; ++k) *(uint2*)(q + (long)(r - 1 - k) * y_ld) = code(u[k]);
    }
    for (; r > r0; --r)
        *(uint2*)(q + (long)(r - 1) * y_ld) = code(*(const uint4*)(p + (long)(r - 1) * x_ld));
}

}  // namespace

/* The partial maxima of every chunk, kColChunks rows of n floats. */
extern "C" int64_t r4d_rad_quant_cols_fp8_scratch(const RadArgs* a) {
    const long long n = r4d_geti_or(a, "n", 0);
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    return (int64_t)kColChunks * n * (int64_t)sizeof(float);
}

extern "C" int r4d_rad_quant_cols_fp8(const RadArgs* a, RadStream stream) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1), *sc = r4d_opt(a, 2);
    if (!x || !y || !sc) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || y->dtype != RAD_F8E4M3 || sc->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    if (n <= 0 || n % 8) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, my = 0;
    const long long xl = r4d_rows_pitch(x, n, &M), yl = r4d_rows_pitch(y, n, &my);
    if (xl < 0 || yl < 0 || (xl & 7) || (yl & 7) || ((uintptr_t)x->data & 15) ||
        ((uintptr_t)y->data & 7) || !r4d_contig(sc) || ((uintptr_t)sc->data & 15))
        return R4D_NO(RAD_E_STRIDE);
    if (my < M || r4d_numel(sc) < n || M > 0x7fffffffLL || n > 0x7fffffffLL)
        return R4D_NO(RAD_E_SHAPE);
    if (!a->scratch || ((uintptr_t)a->scratch & 15) ||
        a->scratch_bytes < (int64_t)kColChunks * n * (int64_t)sizeof(float))
        return R4D_NO(RAD_E_SCRATCH);
    const dim3 grid((unsigned)((n / 8 + kColThreads - 1) / kColThreads), (unsigned)kColChunks);
    hipLaunchKernelGGL(quant_cols_amax_kernel, grid, dim3(kColThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)x->data, (int)M, (int)n, (int)xl, (float*)a->scratch);
    hipLaunchKernelGGL(quant_cols_code_kernel, grid, dim3(kColThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)x->data, (int)M, (int)n, (int)xl,
                       (const float*)a->scratch, (uint8_t*)y->data, (int)yl, (float*)sc->data);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}
