// H += X^T X, in f32, over an E4M3 or int8 activation with a per-(row, group) scale, or over a plain
// bf16 one.
//
// Purpose: the device half of GPTQ's Gram matrix `H = sum_t x_t x_t^T` over a calibration corpus,
// for the 4-bit expert format (docs/MOE-W4.md).
//
// Not a GEMM: `gemm_nt` contracts the last axis of both operands, this contracts the first.
// Expressing it as a GEMM would need a transposed copy of X (a 26 MB scratch and a transpose).
//
// E4M3 input: the taps are the MoE block's own quantiser outputs (`had_quant_act_fp8` for gate_up,
// `gated_had_quant_fp8` for down), the bytes the expert GEMM actually contracts, and the only form
// the rotated path stores, contracted by a scalar kernel. A plain bf16 activation (a bf16-served
// model's) goes through the matrix cores instead: see gram_accum_bf16_wmma.
//
// The gate_up tap is the unsorted activation. Every token is routed to exactly top_k experts, so
// the expert GEMM's input distribution gives H_sorted = top_k * H_token, and GPTQ is invariant to a
// positive scale on H (damping is a fraction of the trace, LDLQ uses ratios). Taking M token rows
// instead of M*top_k gathered ones saves a factor of top_k.
//
// Accumulation is f32, not f64: a block keeps its tile in registers across the M loop and adds to
// global once per launch. The consumer damps H by a percent of its trace, far above this rounding;
// f64 would double 1.3 GB of state.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_fp8_wmma.h"
#include "r4d_gdn_wmma.h"

namespace {

// A 64-wide C tile, 256 threads, 4x4 accumulators per thread (16x16 threads x 16 = 64*64). A
// 64-aligned tile lies inside one scale group, so a row's scale is one load per tile; the launcher
// refuses a group the tile does not divide.
constexpr int kTile    = 64;
constexpr int kThreads = 256;
constexpr int kMB      = 32;    // M rows staged a pass; 2 * 32 * 64 * 4 = 16 KiB of LDS
constexpr int kTT      = 4;

// I8: the codes are int8 (an int8 trunk's activation, quant_act_i8g's), otherwise E4M3; both carry an f32
// scale per (row, group).
template <bool I8>
__global__ __launch_bounds__(kThreads) void gram_accum_kernel(
    const void* __restrict__ qv, const float* __restrict__ sc, float* __restrict__ h,
    uint32_t M, uint32_t K, uint32_t q_ld, uint32_t s_ld, uint32_t h_ld, uint32_t grp,
    uint32_t upper) {
    // `upper`: only the tiles on or above the diagonal; H is symmetric, and a caller that reads
    // only k2 >= k1 halves the work.
    if (upper && blockIdx.y < blockIdx.x) return;

    __shared__ float sa[kMB][kTile];
    __shared__ float sb[kMB][kTile];

    const uint32_t k1_0 = blockIdx.x * (uint32_t)kTile;
    const uint32_t k2_0 = blockIdx.y * (uint32_t)kTile;
    const uint32_t tx = threadIdx.x & 15u, ty = threadIdx.x >> 4;   // 16 x 16
    // Both tiles sit inside one scale group each -- see kTile's note.
    const uint32_t g1 = k1_0 / grp, g2 = k2_0 / grp;

    float acc[kTT][kTT];
#pragma unroll
    for (int i = 0; i < kTT; ++i)
#pragma unroll
        for (int j = 0; j < kTT; ++j) acc[i][j] = 0.0f;

    for (uint32_t m0 = 0; m0 < M; m0 += (uint32_t)kMB) {
        __syncthreads();
        // 32 rows x 16 slots = 512 per tile, two per thread. An fp8 slot is a dword of four E4M3
        // codes, decoded by two v_cvt_pk_f32_fp8.
        for (uint32_t idx = threadIdx.x; idx < (uint32_t)(kMB * kTile / 4);
             idx += (uint32_t)kThreads) {
            const uint32_t mi = idx >> 4, cq = (idx & 15u) * 4u;
            const uint32_t m = m0 + mi;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
            if (m < M) {
                const size_t qrow = (size_t)m * q_ld;
                const uint32_t c1 = k1_0 + cq, c2 = k2_0 + cq;
                if (c1 < K) {
                    // The row is owned out to K rounded up to four bytes (quant_act_fp8's
                    // contract), so a dword load at c1 < K stays inside the allocation.
                    const uint32_t w = *(const uint32_t*)((const uint8_t*)qv + qrow + c1);
                    const float s = sc[(size_t)m * s_ld + g1];
                    if constexpr (I8) {
                        a0 = (float)(int8_t)(w & 0xffu) * s;         a1 = (float)(int8_t)(w >> 8) * s;
                        a2 = (float)(int8_t)(w >> 16) * s;           a3 = (float)(int8_t)(w >> 24) * s;
                    } else {
                        const v2f lo = r4d_fp8::e4m3_pk<false>(w), hi = r4d_fp8::e4m3_pk<true>(w);
                        a0 = lo.x * s; a1 = lo.y * s; a2 = hi.x * s; a3 = hi.y * s;
                    }
                }
                if (c2 < K) {
                    const uint32_t w = *(const uint32_t*)((const uint8_t*)qv + qrow + c2);
                    const float s = sc[(size_t)m * s_ld + g2];
                    if constexpr (I8) {
                        b0 = (float)(int8_t)(w & 0xffu) * s;         b1 = (float)(int8_t)(w >> 8) * s;
                        b2 = (float)(int8_t)(w >> 16) * s;           b3 = (float)(int8_t)(w >> 24) * s;
                    } else {
                        const v2f lo = r4d_fp8::e4m3_pk<false>(w), hi = r4d_fp8::e4m3_pk<true>(w);
                        b0 = lo.x * s; b1 = lo.y * s; b2 = hi.x * s; b3 = hi.y * s;
                    }
                }
            }
            // Columns past K stage as zeros and contribute nothing; the store masks them anyway.
            sa[mi][cq + 0] = a0; sa[mi][cq + 1] = a1; sa[mi][cq + 2] = a2; sa[mi][cq + 3] = a3;
            sb[mi][cq + 0] = b0; sb[mi][cq + 1] = b1; sb[mi][cq + 2] = b2; sb[mi][cq + 3] = b3;
        }
        __syncthreads();

        const uint32_t rows = M - m0 < (uint32_t)kMB ? M - m0 : (uint32_t)kMB;
        for (uint32_t r = 0; r < rows; ++r) {
            float u[kTT], v[kTT];
#pragma unroll
            for (int i = 0; i < kTT; ++i) u[i] = sa[r][ty * kTT + i];
#pragma unroll
            for (int j = 0; j < kTT; ++j) v[j] = sb[r][tx * kTT + j];
#pragma unroll
            for (int i = 0; i < kTT; ++i)
#pragma unroll
                for (int j = 0; j < kTT; ++j) acc[i][j] = fmaf(u[i], v[j], acc[i][j]);
        }
    }

    // One block owns one C tile, so the read-modify-write is race-free. The caller zeroes H once
    // and accumulates the whole corpus into it.
#pragma unroll
    for (int i = 0; i < kTT; ++i) {
        const uint32_t k1 = k1_0 + ty * kTT + (uint32_t)i;
        if (k1 >= K) continue;
#pragma unroll
        for (int j = 0; j < kTT; ++j) {
            const uint32_t k2 = k2_0 + tx * kTT + (uint32_t)j;
            if (k2 < K) h[(size_t)k1 * h_ld + k2] += acc[i][j];
        }
    }
}

// The bf16 form on the matrix cores. One 256-thread block owns a 128x128 C tile; its eight waves
// each own 32x64 of it (2x4 WMMA tiles of 16x16). The contraction runs over the M rows: a pass
// stages 64 rows of both column strips into LDS row-major ([row][column], as they lie in global)
// and each wave reads its fragments transposed (fragT), eight 16-bit reads a fragment. A transpose
// at staging would cost the same reads as stores with an 8-way bank conflict. Products of bf16
// are exact in f32 and WMMA accumulates in f32, so this matches the scalar form up to summation
// order, and an entry and its mirror are the same products in the same order (symmetric to the
// bit). `upper` keeps the scalar form's 64-tile rule, so both forms touch the same entries.
constexpr int kWT  = 128;           // C tile
constexpr int kWMB = 64;            // M rows staged a pass: four 16-deep WMMA steps
constexpr int kWP  = kWT + 8;       // LDS pitch: the two half-waves' rows land 16 banks apart

__device__ __forceinline__ v8bf gram_frag(const unsigned short* src, int base, int kbase, int lane) {
    return fragT(src, kWP, base, kbase, lane);
}

__global__ __launch_bounds__(kThreads) void gram_accum_bf16_wmma(
    const uint16_t* __restrict__ x, float* __restrict__ h, uint32_t M, uint32_t K, uint32_t x_ld,
    uint32_t h_ld, uint32_t upper) {
    if (upper && blockIdx.y < blockIdx.x) return;
    __shared__ __attribute__((aligned(16))) unsigned short sa[kWMB * kWP];
    __shared__ __attribute__((aligned(16))) unsigned short sb[kWMB * kWP];
    const uint32_t k1_0 = blockIdx.x * (uint32_t)kWT, k2_0 = blockIdx.y * (uint32_t)kWT;
    const bool diag = blockIdx.x == blockIdx.y;   // both strips are the same columns
    const int lane = (int)(threadIdx.x & 31u), wave = (int)(threadIdx.x >> 5);
    const int wr = (wave & 3) * 32, wc = (wave >> 2) * 64;

    v8f acc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.0f;

    for (uint32_t m0 = 0; m0 < M; m0 += (uint32_t)kWMB) {
        __syncthreads();
        // 64 rows x 32 slots of four columns = 2048 a strip, eight per thread. K is a multiple
        // of four, so a slot is wholly inside or wholly past the row.
        for (uint32_t idx = threadIdx.x; idx < (uint32_t)(kWMB * kWT / 4); idx += (uint32_t)kThreads) {
            const uint32_t mi = idx >> 5, c = (idx & 31u) * 4u;
            const uint32_t m = m0 + mi;
            uint2 wa = make_uint2(0u, 0u), wb = make_uint2(0u, 0u);
            if (m < M) {
                const uint16_t* row = x + (size_t)m * x_ld;
                if (k1_0 + c < K) wa = *(const uint2*)(row + k1_0 + c);
                if (!diag && k2_0 + c < K) wb = *(const uint2*)(row + k2_0 + c);
            }
            *(uint2*)(sa + mi * kWP + c) = wa;
            if (!diag) *(uint2*)(sb + mi * kWP + c) = wb;
        }
        __syncthreads();
        const unsigned short* B = diag ? sa : sb;
        // Rows past M were staged as zeros, so a whole pass runs and adds nothing for them.
#pragma unroll
        for (int ks = 0; ks < kWMB / 16; ++ks) {
            v8bf fa[2], fb[4];
#pragma unroll
            for (int i = 0; i < 2; ++i) fa[i] = gram_frag(sa, wr + i * 16, ks * 16, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) fb[j] = gram_frag(B, wc + j * 16, ks * 16, lane);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = mma(fa[i], fb[j], acc[i][j]);
        }
    }

    // D: column n = lane % 16 of the B strip (k2), rows m = 8 * (lane / 16) + e of the A strip
    // (k1). One block owns the tile, so the read-modify-write is race-free.
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t k2 = k2_0 + (uint32_t)(wc + j * 16 + (lane & 15));
            if (k2 >= K) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t k1 = k1_0 + (uint32_t)(wr + i * 16 + 8 * (lane >> 4) + e);
                if (k1 >= K || (upper && k2 / (uint32_t)kTile < k1 / (uint32_t)kTile)) continue;
                h[(size_t)k1 * h_ld + k2] += acc[i][j][e];
            }
        }
}

}  // namespace

extern "C" int r4d_gram_accum_fp8_tile(void) { return kTile; }

//   q  [M][K]                E4M3, `q_ld` bytes a row (0 = K), owned out to a multiple of 4
//   s  [M][ceil(K/grp)]      f32,  `s_ld` elements a row (0 = ceil(K/grp))
//   h  [K][K]                f32,  `h_ld` elements a row (0 = K), read and written; with `upper`
//                            only the 64-wide tiles on or above the diagonal are added to
namespace {
int gram_accum_codes(bool i8, long q, long s, long h, int M, int K, int grp, int q_ld, int s_ld,
                     int h_ld, int upper, long stream) {
    if (!q || !s || !h) return -1;
    if (M <= 0 || K <= 0) return -2;
    if (K % 4) return -3;                       // a staging slot is one dword of four codes
    if (grp <= 0 || (grp % kTile)) return -5;   // a tile must lie inside one scale group
    if (!q_ld) q_ld = K;
    if (q_ld % 4) return -6;                    // ...and each row's dwords must stay aligned
    if (!s_ld) s_ld = (K + grp - 1) / grp;
    if (!h_ld) h_ld = K;
    const uint32_t nt = (uint32_t)((K + kTile - 1) / kTile);
    if (i8)
        hipLaunchKernelGGL(gram_accum_kernel<true>, dim3(nt, nt), dim3(kThreads), 0,
                           (hipStream_t)stream, (const void*)q, (const float*)s, (float*)h,
                           (uint32_t)M, (uint32_t)K, (uint32_t)q_ld, (uint32_t)s_ld,
                           (uint32_t)h_ld, (uint32_t)grp, (uint32_t)(upper != 0));
    else
        hipLaunchKernelGGL(gram_accum_kernel<false>, dim3(nt, nt), dim3(kThreads), 0,
                           (hipStream_t)stream, (const void*)q, (const float*)s, (float*)h,
                           (uint32_t)M, (uint32_t)K, (uint32_t)q_ld, (uint32_t)s_ld,
                           (uint32_t)h_ld, (uint32_t)grp, (uint32_t)(upper != 0));
    return hipGetLastError() == hipSuccess ? 0 : -4;
}
}  // namespace

extern "C" int r4d_gram_accum_fp8(long q, long s, long h, int M, int K, int grp,
                                  int q_ld, int s_ld, int h_ld, int upper, long stream) {
    return gram_accum_codes(false, q, s, h, M, K, grp, q_ld, s_ld, h_ld, upper, stream);
}

//   q  [M][K]                int8, otherwise as r4d_gram_accum_fp8
extern "C" int r4d_gram_accum_i8(long q, long s, long h, int M, int K, int grp,
                                 int q_ld, int s_ld, int h_ld, int upper, long stream) {
    return gram_accum_codes(true, q, s, h, M, K, grp, q_ld, s_ld, h_ld, upper, stream);
}

//   a  [M][K]                bf16, `a_ld` elements a row (0 = K)
//   h  [K][K]                f32,  `h_ld` elements a row (0 = K), read and written; `upper` as above
extern "C" int r4d_gram_accum_bf16(long a, long h, int M, int K, int a_ld, int h_ld, int upper,
                                   long stream) {
    if (!a || !h) return -1;
    if (M <= 0 || K <= 0) return -2;
    if (K % 4) return -3;                       // a staging slot is four elements of one row
    if (!a_ld) a_ld = K;
    if (a_ld % 4 || (a % 8)) return -6;         // ...loaded as two aligned dwords
    if (!h_ld) h_ld = K;
    const uint32_t nt = (uint32_t)((K + kWT - 1) / kWT);
    hipLaunchKernelGGL(gram_accum_bf16_wmma, dim3(nt, nt), dim3(kThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)a, (float*)h, (uint32_t)M, (uint32_t)K, (uint32_t)a_ld,
                       (uint32_t)h_ld, (uint32_t)(upper != 0));
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// ============================================================== the plugin ABI entry point

enum { GA_Q = 0, GA_S = 1, GA_H = 2 };

extern "C" int r4d_rad_gram_accum(const RadArgs* a, RadStream s) {
    const RadTensor* q = r4d_opt(a, GA_Q);
    const RadTensor* sc = r4d_opt(a, GA_S);
    const RadTensor* h = r4d_opt(a, GA_H);
    if (!q || !sc || !h) return R4D_NO(RAD_E_INVAL);
    if ((q->dtype != RAD_F8E4M3 && q->dtype != RAD_I8) || sc->dtype != RAD_F32 ||
        h->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    /* Rows are dim 0 and everything after is the row, as in r4d_rad_quant_act_fp8; `sc` and `h`
     * are rank 2 by definition. */
    if (q->rank < 2 || sc->rank != 2 || h->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = q->shape[0];
    if (M < 1 || r4d_numel(q) % M) return R4D_NO(RAD_E_SHAPE);
    const long long K = r4d_numel(q) / M;
    const long long grp = r4d_geti_or(a, "group", R4D_FP8_BLOCK);
    const long long nkb = (K + grp - 1) / grp;
    if (sc->shape[0] != M || sc->shape[1] != nkb) return R4D_NO(RAD_E_SHAPE);
    if (h->shape[0] != K || h->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    const long long qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc), hp = r4d_row_pitch(h);
    if (qp < 0 || sp < 0 || hp < 0) return R4D_NO(RAD_E_STRIDE);
    const int e = gram_accum_codes(q->dtype == RAD_I8, r4d_addr(q), r4d_addr(sc), r4d_addr(h),
                                   (int)M, (int)K, (int)grp, (int)qp, (int)sp, (int)hp,
                                   (int)r4d_geti_or(a, "upper", 0), (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_gram_accum_bf16(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, GA_Q);
    const RadTensor* h = r4d_opt(a, GA_H);
    if (!x || !h) return R4D_NO(RAD_E_INVAL);
    /* A scale operand means the caller believes the input is quantised; refuse it. */
    if (r4d_opt(a, GA_S)) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || h->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (x->rank < 2 || h->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = x->shape[0];
    if (M < 1 || r4d_numel(x) % M) return R4D_NO(RAD_E_SHAPE);
    const long long K = r4d_numel(x) / M;
    if (h->shape[0] != K || h->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    const long long xp = r4d_row_pitch(x), hp = r4d_row_pitch(h);
    if (xp < 0 || hp < 0) return R4D_NO(RAD_E_STRIDE);
    const int e = r4d_gram_accum_bf16(r4d_addr(x), r4d_addr(h), (int)M, (int)K, (int)xp, (int)hp,
                                      (int)r4d_geti_or(a, "upper", 0), (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : R4D_NO(RAD_E_SHAPE);
}
