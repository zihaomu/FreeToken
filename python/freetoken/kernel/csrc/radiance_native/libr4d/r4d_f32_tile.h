// r4d_f32_tile.h -- the register-blocked f32 GEMM tile under libr4d's f32 GEMM (gemm_nt_bias at
// f32) and its f32 convolutions (conv1d and conv_t1d as implicit GEMMs, r4d_audio.hip).
//
// One 256-thread workgroup computes a BM x BN block (BM = 16 TM, BN = 16 TN) of C = A . W^T over
// K. Both operands are staged through LDS k-major, sixteen K a slab and two slabs of LDS, so the
// next slab's global loads are in flight under the current slab's FMAs and one barrier a slab
// suffices. A thread owns TM x TN accumulators and reads its operands from LDS as float4 (or
// float2) rows; every product is a fused multiply-add in ascending K, one rounding a product and
// sum, against libref's separate multiply and add in the same order.
//
// The operands come from loader policies: a thread stages rows `r0 + 16 j` of A (j < TM) and of W
// (j < TN) at one K column a slab, `kk = 16 slab + (tid % 16)`, so a GEMM reads rows in place and
// a convolution gathers the input row a tap reads, with its sequence's zero padding, from the same
// slot. A row or column past the operand reads zero. The epilogue is the caller's: it receives
// each run of contiguous columns a thread owns, in registers.
//
// Thread (tx, ty) = (tid % 16, tid / 16) owns rows g 16 RW + ty RW + r and columns
// g 16 CW + tx CW + c, RW = run_of(TM) and CW = run_of(TN): a wave's sixteen tx read 16 CW
// contiguous floats of W's slab row, conflict-free, and the two ty values of a wave read two A
// vectors, each broadcast. The LDS pitch is padded by four floats, which spreads the staging
// stores (a wave's 16 K columns of one row) over distinct banks.
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace r4d_f32 {

constexpr int kThr = 256;
constexpr int kBK  = 16;   // K a staged slab

// The run a thread owns along an axis of `t` values: four where `t` is whole fours, else two (a
// tile of 6 columns is three runs of two, so a 96-wide output fills its tile exactly).
constexpr int run_of(int t) { return t % 4 == 0 ? 4 : 2; }

// A plain row-major f32 operand: row r, column k at p[r * ld + k], zero past `rows` or `K`. Every
// load is issued, at an address clamped into the operand, and the zero selected after it: a load
// guarded by a branch serialises the slab's loads behind one another. Offsets are 32-bit, so the
// operand spans fewer than 2^32 elements (the shims check).
template <int CNT>
struct RowsLoader {
    const float* p;
    int ld, rows, K;
    int k = 0;
    uint32_t off[CNT];
    bool live[CNT];
    __device__ void init(int first_row, int kk) {
        k = kk;
#pragma unroll
        for (int j = 0; j < CNT; ++j) {
            const int r = first_row + 16 * j;
            live[j] = r < rows;
            off[j] = (uint32_t)(r < rows ? r : rows - 1) * (uint32_t)ld;
        }
    }
    __device__ void fetch(float (&v)[CNT]) const {
        const bool kl = k < K;
        const uint32_t kc = (uint32_t)(kl ? k : K - 1);
#pragma unroll
        for (int j = 0; j < CNT; ++j) {
            const float x = p[off[j] + kc];
            v[j] = (live[j] && kl) ? x : 0.0f;
        }
    }
    __device__ void advance() { k += kBK; }
};

template <int TM, int TN, class LA, class LW, class EP>
__device__ __forceinline__ void tile(LA& la, LW& lw, int K, EP& ep) {
    constexpr int BM = 16 * TM, BN = 16 * TN;
    constexpr int AP = BM + 4, WP = BN + 4;
    constexpr int RW = run_of(TM), RG = TM / RW;
    constexpr int CW = run_of(TN), CG = TN / CW;
    static_assert(RW * RG == TM && CW * CG == TN && (RW == 2 || RW == 4) && (CW == 2 || CW == 4),
                  "a thread's rows and columns are whole runs of two or four");
    __shared__ __attribute__((aligned(16))) float sA[2][kBK][AP];
    __shared__ __attribute__((aligned(16))) float sW[2][kBK][WP];

    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    float ra[TM], rw[TN];
    la.fetch(ra);
    lw.fetch(rw);
    const int ns = (K + kBK - 1) / kBK;
    for (int s = 0; s < ns; ++s) {
        const int b = s & 1;
#pragma unroll
        for (int j = 0; j < TM; ++j) sA[b][tx][ty + 16 * j] = ra[j];
#pragma unroll
        for (int j = 0; j < TN; ++j) sW[b][tx][ty + 16 * j] = rw[j];
        __syncthreads();
        if (s + 1 < ns) {
            la.advance();
            lw.advance();
            la.fetch(ra);
            lw.fetch(rw);
        }
#pragma unroll
        for (int k = 0; k < kBK; ++k) {
            float a[TM], w[TN];
#pragma unroll
            for (int g = 0; g < RG; ++g) {
                const float* src = &sA[b][k][g * 16 * RW + ty * RW];
                if constexpr (RW == 4) {
                    const float4 q = *(const float4*)src;
                    a[g * 4] = q.x; a[g * 4 + 1] = q.y; a[g * 4 + 2] = q.z; a[g * 4 + 3] = q.w;
                } else {
                    const float2 q = *(const float2*)src;
                    a[g * 2] = q.x; a[g * 2 + 1] = q.y;
                }
            }
#pragma unroll
            for (int g = 0; g < CG; ++g) {
                const float* src = &sW[b][k][g * 16 * CW + tx * CW];
                if constexpr (CW == 4) {
                    const float4 q = *(const float4*)src;
                    w[g * 4] = q.x; w[g * 4 + 1] = q.y; w[g * 4 + 2] = q.z; w[g * 4 + 3] = q.w;
                } else {
                    const float2 q = *(const float2*)src;
                    w[g * 2] = q.x; w[g * 2 + 1] = q.y;
                }
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_fmaf(a[i], w[j], acc[i][j]);
        }
        /* One barrier a slab: the next slab goes to the other buffer, which every thread finished
         * reading before it passed this slab's barrier. */
    }

    /* A run of CW columns as one float4 (the last two lanes unused at CW 2), so the accumulators
     * stay in registers through the caller's store; `gj` says which of the thread's CG runs it
     * is, for state the caller keeps per run. */
#pragma unroll
    for (int gi = 0; gi < RG; ++gi)
#pragma unroll
        for (int r = 0; r < RW; ++r)
#pragma unroll
            for (int gj = 0; gj < CG; ++gj) {
                const float* q = acc[gi * RW + r] + gj * CW;
                float4 v;
                if constexpr (CW == 4) v = make_float4(q[0], q[1], q[2], q[3]);
                else                   v = make_float4(q[0], q[1], 0.0f, 0.0f);
                ep(gi * 16 * RW + ty * RW + r, gj * 16 * CW + tx * CW, CW, v, gj);
            }
}

}  // namespace r4d_f32
