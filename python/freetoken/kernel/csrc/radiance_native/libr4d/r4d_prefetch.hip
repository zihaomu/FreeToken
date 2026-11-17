// r4d_prefetch.hip - r4d_rad_prefetch: read a weight's leading bytes into the on-die cache.
//
// A decode step at a few rows streams each weight once from memory, and the all-reduce between two
// GEMMs leaves memory idle for its whole round trip: ~20 us at a DFlash2 verify's 8 rows on 4x
// R9700. A read issued on the second lane in that window puts the next GEMM's first bytes in the
// 64 MB Infinity Cache, and the GEMM then streams them from there. On four cards (a microbenchmark
// of the 8-row two-shot and a GEMM-like stream of a cold 29 MB weight) 16 MB read beside the
// all-reduce took a layer from 84.0 us to 70.1; the same read on the same stream gained nothing.
//
// It writes nothing at all. Each lane folds its loads into one word and hands it to an empty
// `asm volatile`, which the compiler can neither drop nor see into, so the loads stay.
//
// Nothing here reaches HIP's runtime: no allocation at launch and no `__device__` word, which HIP's
// runtime registers and places itself (the engine records only fat binaries and functions). Such a
// word cost ~8% of a DFlash2 step on 4x R9700, for a reason not pinned down.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_emit.h"
#include <cstdint>

namespace {

__global__ __launch_bounds__(256) void prefetch_kernel(const int4* __restrict__ p, size_t n16) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nt  = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (size_t i = tid; i < n16; i += nt) {
        const int4 v = p[i];
        acc ^= (unsigned)(v.x ^ v.y ^ v.z ^ v.w);
    }
    asm volatile("" ::"v"(acc));
}

/* The chained form (RadKernelChainFn): it waits for the kernels before it on its queue, as every
 * chained form does, but adds its blocks to the counter at once, before reading: nothing after it
 * reads anything it writes, so the next kernel need not wait for its reads. Issued on lane 0
 * between a GEMM and the collective after it (where launches chain), it starts reading as the
 * GEMM's last block ends, with no event between the queues and no barrier bit on the GEMM. */
__global__ __launch_bounds__(256) void r4dchain_prefetch(const int4* __restrict__ p, size_t n16) {
    r4d_chain_wait();
    r4d_chain_done();
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nt  = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (size_t i = tid; i < n16; i += nt) {
        const int4 v = p[i];
        acc ^= (unsigned)(v.x ^ v.y ^ v.z ^ v.w);
    }
    asm volatile("" ::"v"(acc));
}

}  // namespace

extern "C" int r4d_rad_layout_any(const RadParam*, int, int, const RadEncoding*, const int*,
                                  const RadTensor*, int, RadLayout*) {
    return RAD_LAYOUT_ANY;
}

extern "C" int r4d_rad_prefetch(const RadArgs* a, RadStream stream) {
    const RadTensor* w = r4d_opt(a, 0);
    if (!w || !w->data) return R4D_NO(RAD_E_INVAL);
    long long bytes = r4d_geti_or(a, "bytes", 0);
    if (bytes <= 0) return RAD_OK;
    // Never past what the tensor says it holds, where its dtype says how wide an element is.
    const int bits = rad_dtype_bits(w->dtype);
    if (bits > 0) {
        long long n = 1;
        for (uint32_t i = 0; i < w->rank; ++i) n *= w->shape[i];
        const long long have = n * bits / 8;
        if (have > 0 && bytes > have) bytes = have;
    }
    if ((uintptr_t)w->data % 16) return R4D_NO(RAD_E_ALIGN);
    int nb = r4d_cfg_prefetch_nb(a);
    if (nb < 1) nb = 64;
    const bool ch = r4d_geti_or(a, "chain", 0) != 0;
    return R4D_EMIT(stream, (ch ? r4dchain_prefetch : prefetch_kernel), dim3((unsigned)nb), dim3(256), 0,
                    (const int4*)w->data, (size_t)(bytes / 16));
}
