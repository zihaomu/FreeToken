// r4d_gemm_f32_nt_bias.hip -- gemm_nt_bias at f32: C = res + act(A @ W^T + bias) with an f32
// activation and weight, f32 accumulation, and the result f32 or rounded to bf16 at the store.
// The epilogue is libref's: the biased product, the activation of it and the residual sum each
// rounded to C's dtype, as the unfused chain's tensors would be; at an f32 C only the arithmetic
// rounds. A diffusion transformer's patch projections and output heads, its timestep MLP, and a
// video decoder's 1x1 projections run here: K from 24 to a few thousand, N from 24 to 5376, M up
// to a hundred thousand rows.
//
// The tile is r4d_f32_tile.h's at 128 rows: 128 columns where N is wider than 96, 96 where it is
// wider than 64, 64 where it is wider than 32, 32 otherwise, so a narrow head (N 32 or 96) fills
// its tile. No split-K: the narrow-N shapes have hundreds of row tiles.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_f32_tile.h"

#include <cmath>
#include <cstring>

namespace {

constexpr int kTM = 8, kBM = 16 * kTM;

__device__ __forceinline__ float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}
__device__ __forceinline__ float gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f, k1 = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x * x * x)));
}

// C's element: f32 as it is, or bf16 rounded to nearest even.
__device__ __forceinline__ float rd(const float*, float v) { return v; }
__device__ __forceinline__ float rd(const uint16_t*, float v) { return bf16_to_f32(f32_to_bf16(v)); }
__device__ __forceinline__ float ld1(const float* p) { return *p; }
__device__ __forceinline__ float ld1(const uint16_t* p) { return bf16_to_f32(*p); }
__device__ __forceinline__ void st1(float* p, float v) { *p = v; }
__device__ __forceinline__ void st1(uint16_t* p, float v) { *p = f32_to_bf16(v); }
// Four (or two) adjacent elements in one store: 16 (8) bytes at f32, 8 (4) at bf16.
__device__ __forceinline__ void stv(float* p, int cnt, const float* v) {
    if (cnt == 4) *(float4*)p = make_float4(v[0], v[1], v[2], v[3]);
    else          *(float2*)p = make_float2(v[0], v[1]);
}
__device__ __forceinline__ void stv(uint16_t* p, int cnt, const float* v) {
    if (cnt == 4) *(uint2*)p = make_uint2(bf16x2_rne(v[0], v[1]), bf16x2_rne(v[2], v[3]));
    else          *(uint32_t*)p = bf16x2_rne(v[0], v[1]);
}

// `act` 0 none, 1 exact GELU, 2 tanh GELU. `R` may be `C` itself (a residual updated in place):
// each element is read by the thread that stores it, before the store. `vec` says C's rows take
// whole-run stores (a pitch of whole runs at an aligned base).
template <int TN, typename TY>
__global__ __launch_bounds__(r4d_f32::kThr) void gemm_f32_bias_kernel(
    const float* __restrict__ A, int lda, const float* __restrict__ W, int K,
    const void* __restrict__ bias, int bias_f32, const TY* R, int ldr, TY* C, int ldc, int M,
    int N, int act, int vec) {
    constexpr int BN = 16 * TN, CW = r4d_f32::run_of(TN);
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * kBM, n0 = blockIdx.x * BN;
    r4d_f32::RowsLoader<kTM> la{ A, lda, M, K };
    r4d_f32::RowsLoader<TN> lw{ W, K, N, K };
    la.init(m0 + ty, tx);
    lw.init(n0 + ty, tx);
    /* The bias of this thread's TN columns (runs of CW, 16 CW apart), read once. */
    float bc[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int n = n0 + (j / CW) * 16 * CW + tx * CW + j % CW;
        const int nc = n < N ? n : N - 1;
        bc[j] = !bias ? 0.0f : bias_f32 ? ((const float*)bias)[nc] : bf16_to_f32(((const uint16_t*)bias)[nc]);
    }
    auto ep = [&](int r, int c, int cnt, float4 q, int g) {
        const int m = m0 + r, n = n0 + c;
        if (m >= M || n >= N) return;
        float v[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
        for (int e = 0; e < CW; ++e) {
            float t = rd(C, v[e] + bc[g * CW + e]);
            if (act == 1) t = rd(C, gelu_erf(t));
            else if (act == 2) t = rd(C, gelu_tanh(t));
            if (R && n + e < N) t = ld1(R + (size_t)m * ldr + n + e) + t;
            v[e] = t;
        }
        TY* dst = C + (size_t)m * ldc + n;
        if (vec && n + cnt <= N) {
            stv(dst, cnt, v);
        } else {
#pragma unroll
            for (int e = 0; e < 4; ++e)
                if (e < cnt && n + e < N) st1(dst + e, v[e]);
        }
    };
    r4d_f32::tile<kTM, TN>(la, lw, K, ep);
}

}  // namespace

// ============================================================== the plugin ABI entry point
enum { GF_A = 0, GF_W, GF_BIAS, GF_RES, GF_Y };

static int gemm_f32_emit(const RadArgs* a, RadStream stream, bool plain) {
    const RadTensor* A = r4d_opt(a, GF_A);
    const RadTensor* W = r4d_opt(a, GF_W);
    const RadTensor* B = r4d_opt(a, GF_BIAS);
    const RadTensor* R = r4d_opt(a, GF_RES);
    const RadTensor* Y = r4d_opt(a, GF_Y);
    if (!A || !W || !Y || (!B && !plain)) return R4D_NO(RAD_E_INVAL);
    /* f32 A and W; C f32 or bf16, the residual C's dtype; the bias f32 or bf16. */
    if (A->dtype != RAD_F32 || W->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (Y->dtype != RAD_F32 && Y->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (B && B->dtype != RAD_F32 && B->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (R && R->dtype != Y->dtype) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(W) || (B && !r4d_contig(B))) return R4D_NO(RAD_E_STRIDE);

    const long long K = r4d_geti_or(a, "K", r4d_dim(W, (int)W->rank - 1));
    const long long N = r4d_geti_or(a, "N", r4d_numel(W) / (K > 0 ? K : 1));
    if (K <= 0 || N <= 0 || r4d_numel(W) < N * K || (B && r4d_numel(B) < N)) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Ma = 0, Mr = 0;
    const long long ldc = r4d_rows_pitch(Y, N, &M);
    const long long lda = r4d_rows_pitch(A, K, &Ma);
    const long long ldr = R ? r4d_rows_pitch(R, N, &Mr) : 0;
    if (ldc < 0 || lda < 0 || ldr < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Ma < M || (R && Mr < M)) return R4D_NO(RAD_E_SHAPE);
    /* The tile's loaders index A and W with 32-bit element offsets. */
    if (M > 0x7fffffffll || N > 0x7fffffffll || K > 0x7fffffffll || lda > 0x7fffffffll ||
        ldc > 0x7fffffffll || ldr > 0x7fffffffll || (M + kBM - 1) / kBM > 65535 ||
        M * lda >= (1ll << 32) || N * K >= (1ll << 32))
        return R4D_NO(RAD_E_SHAPE);

    const char* act_s = r4d_gets(a, "act");
    int act = 0;
    if (act_s && *act_s) {
        if (!std::strcmp(act_s, "gelu")) act = 1;
        else if (!std::strcmp(act_s, "gelu_tanh")) act = 2;
        else if (std::strcmp(act_s, "none")) return R4D_NO(RAD_E_SCHEMA);
    }
    const int bias_f32 = B && B->dtype == RAD_F32;
    const bool f32 = Y->dtype == RAD_F32;
    /* Whole-run stores where every row starts aligned to a run of four. */
    const unsigned run_bytes = f32 ? 16u : 8u;
    const int vec = (ldc % 4) == 0 && r4d_aligned(r4d_p(Y), run_bytes);

    const int tn = N > 96 ? 8 : N > 64 ? 6 : N > 32 ? 4 : 2;
    const dim3 grid((unsigned)((N + 16 * tn - 1) / (16 * tn)), (unsigned)((M + kBM - 1) / kBM));
    const dim3 block(r4d_f32::kThr);
#define GF_L(TN, TY)                                                                              \
    return R4D_EMIT(stream, (gemm_f32_bias_kernel<TN, TY>), grid, block, 0,                       \
                    (const float*)r4d_p(A), (int)lda, (const float*)r4d_p(W), (int)K, r4d_p(B),  \
                    bias_f32, (const TY*)r4d_p(R), (int)ldr, (TY*)r4d_wp(Y), (int)ldc, (int)M,   \
                    (int)N, act, vec)
    if (f32) {
        if (tn == 8) GF_L(8, float);
        if (tn == 6) GF_L(6, float);
        if (tn == 4) GF_L(4, float);
        GF_L(2, float);
    }
    if (tn == 8) GF_L(8, uint16_t);
    if (tn == 6) GF_L(6, uint16_t);
    if (tn == 4) GF_L(4, uint16_t);
    GF_L(2, uint16_t);
#undef GF_L
}

extern "C" int r4d_rad_gemm_f32_nt_bias(const RadArgs* a, RadStream stream) {
    return gemm_f32_emit(a, stream, false);
}

extern "C" int r4d_rad_gemm_f32_nt(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return R4D_NO(RAD_E_INVAL);
    const RadTensor t[] = {a->t[0], a->t[1], {}, {}, a->t[2]};
    RadArgs b = *a; b.t = t; b.n_t = 5;
    return gemm_f32_emit(&b, stream, true);
}
