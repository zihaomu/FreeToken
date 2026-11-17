// r4d_moe_gather.h -- moe_gather's kernel for decode and short passes, beside its launcher in
// r4d_moe.hip; tests/r4d_emu runs it from here against the inverse the sort writes.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

#include "r4d_common.h"

// Internal linkage, as the unit's own definition had.
#include "r4d_chain.h"

namespace r4d_moegather {
namespace {

/* The slot table is hoisted: `inv[t * top_k + j]` and `ew[t * ew_ld + j]` do not depend on the
 * column, and read inside the column loop each `inv` load would serialise the `ye` load behind it
 * (110 dependent latencies per layer at decode). Read once into LDS, the `ye` loads pipeline.
 *
 * The column range is also split over blocks, since decode has one token. Neither change touches
 * the sum: each column's slots are still summed in ascending order within one block. */
constexpr uint32_t kGatherSlots = 64;
/* When small, the inverse permutation is built in this kernel and the two prepass kernels are not
 * launched. A block owns one token, so it scans `sorted` for entries in [t*top_k, (t+1)*top_k),
 * giving exactly the table `inv` would hold. Every block repeats the scan, so the launcher caps it
 * (r4d_rad_moe_gather); at decode T = 40 and the scan is one pass, against two dependent dispatches
 * (~2.1 us each) of single-block kernels.
 *
 * `sorted == nullptr` selects the prepass path, used at prefill. */
constexpr uint32_t kGatherSelfT = 512;

template <bool F32>
__device__ float gather_weight(const void* ew, size_t i) {
    if constexpr (F32) return ((const float*)ew)[i];
    else return bf16_to_f32(((const uint16_t*)ew)[i]);
}
/* The body, over the block's two shared tables, which the kernel that calls it owns: a function
 * because a block past the last token returns at once, and a chained block must still add to its
 * queue's counter (r4d_chain.h). */
#define R4D_MG_PARAMS                                                                            \
    const uint16_t* __restrict__ ye, const void* __restrict__ ew,                                \
    const int32_t* __restrict__ inv, uint16_t* __restrict__ y, uint32_t M, uint32_t n,           \
    uint32_t top_k, uint32_t ye_ld, uint32_t ew_ld, uint32_t y_ld,                               \
    const int32_t* __restrict__ sorted, uint32_t T, const uint16_t* __restrict__ sh,             \
    const uint16_t* __restrict__ sg, uint32_t sh_ld, uint32_t sg_ld, int sig
#define R4D_MG_ARGS                                                                              \
    ye, ew, inv, y, M, n, top_k, ye_ld, ew_ld, y_ld, sorted, T, sh, sg, sh_ld, sg_ld, sig
template <uint32_t NTHR, bool F32>
__device__ __forceinline__ void gather_body(R4D_MG_PARAMS, int32_t* s_i, float* s_w) {
    const uint32_t t = blockIdx.x;
    if (t >= M) return;
    const bool cached = top_k <= kGatherSlots;
    /* The shared arm's scalar, read once per block as scale_rows reads it once per row. */
    float gate = 0.0f;
    if (sh) {
        const float v0 = bf16_to_f32(sg[(size_t)t * sg_ld]);
        gate = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    }
    if (sorted) {
        /* An unclaimed slot must read -1, so fill and scan are separated by a barrier (as the
         * prepass is two kernels). */
        for (uint32_t j = threadIdx.x; j < top_k; j += NTHR) s_i[j] = -1;
        __syncthreads();
        const int32_t lo = (int32_t)(t * top_k), hi = lo + (int32_t)top_k;
        for (uint32_t i = threadIdx.x; i < T; i += NTHR) {
            const int32_t f = sorted[i];
            if (f >= lo && f < hi) s_i[f - lo] = (int32_t)i;
        }
        for (uint32_t j = threadIdx.x; j < top_k; j += NTHR)
            s_w[j] = gather_weight<F32>(ew, (size_t)t * ew_ld + j);
        __syncthreads();
    } else if (cached) {
        for (uint32_t j = threadIdx.x; j < top_k; j += NTHR) {
            s_i[j] = inv[t * top_k + j];
            s_w[j] = gather_weight<F32>(ew, (size_t)t * ew_ld + j);
        }
        __syncthreads();
    }
    for (uint32_t c = blockIdx.y * NTHR + threadIdx.x; c < n; c += NTHR * gridDim.y) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < top_k; ++j) {
            const int32_t i = cached ? s_i[j] : inv[t * top_k + j];
            if (i < 0) continue;
            const float wj = cached ? s_w[j] : gather_weight<F32>(ew, (size_t)t * ew_ld + j);
            if constexpr (F32) acc = fmaf(wj, bf16_to_f32(ye[(size_t)i * ye_ld + c]), acc);
            else acc += wj * bf16_to_f32(ye[(size_t)i * ye_ld + c]);
        }
        const uint16_t g = f32_to_bf16(acc);
        if (!sh) { y[(size_t)t * y_ld + c] = g; continue; }
        /* The shared arm, folded with the unfused pair's two narrows: `scale_rows` reads the gather
         * as bf16 and narrows its own product before adding. */
        const uint16_t p = f32_to_bf16(bf16_to_f32(sh[(size_t)t * sh_ld + c]) * gate);
        y[(size_t)t * y_ld + c] = f32_to_bf16(bf16_to_f32(p) + bf16_to_f32(g));
    }
}
template <uint32_t NTHR, bool F32 = false>
__global__ __launch_bounds__(NTHR) void gather_kernel(R4D_MG_PARAMS) {
    __shared__ int32_t s_i[kGatherSlots];
    __shared__ float   s_w[kGatherSlots];
    gather_body<NTHR, F32>(R4D_MG_ARGS, s_i, s_w);
}
/* The chained form, taken where the gather is the op's one launch (a decode step). */
template <uint32_t NTHR, bool F32 = false>
__global__ __launch_bounds__(NTHR) void r4dchain_moe_gather(R4D_MG_PARAMS) {
    __shared__ int32_t s_i[kGatherSlots];
    __shared__ float   s_w[kGatherSlots];
    r4d_chain_wait();
    gather_body<NTHR, F32>(R4D_MG_ARGS, s_i, s_w);
    r4d_chain_done();
}
#undef R4D_MG_ARGS
#undef R4D_MG_PARAMS

}  // namespace
}  // namespace r4d_moegather
