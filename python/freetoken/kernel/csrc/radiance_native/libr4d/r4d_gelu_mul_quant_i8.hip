// r4d_gelu_mul_quant_i8.hip - r4d_gelu_mul_quant_i8: gelu(gate) * up and the per-row int8 quantiser
// r4d_gemm_w4a8_nt_m64 reads, in one pass over the row.
//
// The down projection's input of a GELU-gated feed-forward (Gemma) on a w4a8 grid, replacing the
// gelu_mul + quant_act_i8 pair and byte-identical to it: the product is formed as gelu_mul_kernel
// forms it (tanh form, same constants, rounded to bf16), and the absmax, scale and codes are
// quant_act_i8's over those bf16 values. The absmax is a max, so partitioning does not change it.
//
// `gelu_mul_quant_i8g` is the same product with quant_act_i8g's quantiser instead (GRP): row-major
// codes and an f32 scale per (row, 128 columns), for a down projection at i8a8, w4a8i8_asym or
// w5a8i8; byte-identical to gelu_mul + quant_act_i8g.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_chain.h"
#include "r4d_q8.h"
#include <cstdint>

namespace {

// CHAIN brackets the body with the chained launch's wait and add (r4d_chain.h); every block runs
// to the end. The plain kernel and the chained one (r4dchain*) follow. GRP: quant_act_i8g's codes
// and n / 128 scales a row (r4d_q8.h's q8_row_i8g) instead of quant_act_i8's.
template <uint32_t THR, bool CHAIN, bool GRP = false>
__device__ __forceinline__ void gelu_mul_quant_i8_body(
    const uint16_t* __restrict__ gu, signed char* __restrict__ q, float* __restrict__ S,
    uint16_t* __restrict__ yo, uint32_t n, uint32_t gu_ld, uint32_t y_ld) {
    if (CHAIN) r4d_chain_wait();
    extern __shared__ uint16_t ybuf[];
    __shared__ float lm[THR / 32];
    const uint32_t row = blockIdx.x;
    const uint16_t* __restrict__ gr = gu + (size_t)row * gu_ld;
    signed char* __restrict__ qr = q + (size_t)row * n;

    float m = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const float g = bf16_to_f32(gr[i]);
        const float k0 = 0.7978845608028654f, k1 = 0.044715f;
        const float gelu = 0.5f * g * (1.0f + tanhf(k0 * (g + k1 * g * g * g)));
        const uint16_t y = f32_to_bf16(gelu * bf16_to_f32(gr[n + i]));
        ybuf[i] = y;
        if (yo) yo[(size_t)row * y_ld + i] = y;
        m = fmaxf(m, fabsf(bf16_to_f32(y)));
    }
    if constexpr (GRP) {
        __syncthreads();   // publishes ybuf to every thread
        q8_row_i8g<THR>(ybuf, qr, S + (size_t)row * (n / 128u), n);
        if (CHAIN) r4d_chain_done();
        return;
    }
    m = r4d_wave_max(m);
    if ((threadIdx.x & 31) == 0) lm[threadIdx.x >> 5] = m;
    __syncthreads();   // also publishes ybuf to every thread
    m = lm[0];
#pragma unroll
    for (uint32_t i = 1; i < THR / 32; i++) m = fmaxf(m, lm[i]);
    const float scale = fmaxf(m, 1e-8f) / 127.f;
    const float inv = 1.f / scale;
    if (threadIdx.x == 0) S[row] = scale;
    // quant_act_i8's fragment order, see r4d_quant_act_i8.hip
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const uint32_t s = i >> 4, j = i & 15;
        const uint32_t src = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
        const int t = __float2int_rn(bf16_to_f32(ybuf[src]) * inv);
        qr[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
    }
    if (CHAIN) r4d_chain_done();
}

template <uint32_t THR, bool GRP = false>
__global__ __launch_bounds__(THR) void gelu_mul_quant_i8_kernel(
    const uint16_t* __restrict__ gu, signed char* __restrict__ q, float* __restrict__ S,
    uint16_t* __restrict__ yo, uint32_t n, uint32_t gu_ld, uint32_t y_ld) {
    gelu_mul_quant_i8_body<THR, false, GRP>(gu, q, S, yo, n, gu_ld, y_ld);
}
template <uint32_t THR, bool GRP = false>
__global__ __launch_bounds__(THR) void r4dchain_gelu_mul_quant_i8(
    const uint16_t* __restrict__ gu, signed char* __restrict__ q, float* __restrict__ S,
    uint16_t* __restrict__ yo, uint32_t n, uint32_t gu_ld, uint32_t y_ld) {
    gelu_mul_quant_i8_body<THR, true, GRP>(gu, q, S, yo, n, gu_ld, y_ld);
}

}  // namespace

// Operands (r4d_rows.cpp, oGeluMulQuantI8): gate_up [M][2n] bf16 (gate first), q [M][n] int8 in the
// fragment order, scale [M][1] f32, out_bf16 [M][n] bf16 optional. gelu_mul_quant_i8g (`grp`): q
// row-major, scale [M][n / 128], n a multiple of 128.
enum { GM_GU = 0, GM_Q, GM_S, GM_Y };

static int rad_gelu_mul_quant_i8(const RadArgs* a, RadStream stream, bool grp) {
    int emit_rc = RAD_OK;
    const RadTensor* gu = r4d_opt(a, GM_GU);
    const RadTensor* q  = r4d_opt(a, GM_Q);
    const RadTensor* s  = r4d_opt(a, GM_S);
    const RadTensor* y  = r4d_opt(a, GM_Y);
    if (!gu || !q || !s) return R4D_NO(RAD_E_INVAL);
    if (gu->dtype != RAD_BF16 || q->dtype != RAD_I8 || s->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    long long M = 0, n = 0;
    if (r4d_rows_of(q, &M, &n) != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    if (n % 16 || (grp && n % 128) || !r4d_contig(q)) return R4D_NO(RAD_E_SHAPE);
    long long Mg = 0;
    const long long gp = r4d_rows_pitch(gu, 2 * n, &Mg);
    if (gp < 0) return R4D_NO(RAD_E_STRIDE);
    if (Mg < M) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(s) != M * (grp ? n / 128 : 1) || !r4d_contig(s)) return R4D_NO(RAD_E_SHAPE);
    if (y && r4d_numel(y) != M * n) return R4D_NO(RAD_E_SHAPE);
    const long long yp = y ? r4d_row_pitch(y) : n;
    if (yp < 0) return R4D_NO(RAD_E_STRIDE);
    const size_t lds = (size_t)n * sizeof(uint16_t);
    if (lds > 60 * 1024) return R4D_NO(RAD_E_SHAPE);
    const int thr = r4d_act_threads((int)M, (int)n);
    const bool chain = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;   /* RadKernelChainFn, decode */
#define R4D_GMQ_C(T, C, G) R4D_EMIT(stream, (C ? r4dchain_gelu_mul_quant_i8<T, G> : gelu_mul_quant_i8_kernel<T, G>), \
    dim3((uint32_t)M), dim3(T), lds, \
    (const uint16_t*)r4d_p(gu), (signed char*)r4d_wp(q), (float*)r4d_wp(s), (uint16_t*)r4d_wp(y),           \
    (uint32_t)n, (uint32_t)gp, (uint32_t)yp)
#define R4D_GMQ_G(T, G) emit_rc = chain ? R4D_GMQ_C(T, true, G) : R4D_GMQ_C(T, false, G)
#define R4D_GMQ_T(T) do { if (grp) R4D_GMQ_G(T, true); else R4D_GMQ_G(T, false); } while (0)
    if (thr == 1024)     R4D_GMQ_T(1024);
    else if (thr == 512) R4D_GMQ_T(512);
    else if (thr == 256) R4D_GMQ_T(256);
    else return R4D_NO(RAD_E_INVAL);
#undef R4D_GMQ_T
#undef R4D_GMQ_G
#undef R4D_GMQ_C
    return emit_rc;
}

extern "C" int r4d_rad_gelu_mul_quant_i8(const RadArgs* a, RadStream stream) {
    return rad_gelu_mul_quant_i8(a, stream, false);
}

/* quant_act_i8g's grid: its own op, as quant_act_i8g is beside quant_act_i8. */
extern "C" int r4d_rad_gelu_mul_quant_i8g(const RadArgs* a, RadStream stream) {
    return rad_gelu_mul_quant_i8(a, stream, true);
}
