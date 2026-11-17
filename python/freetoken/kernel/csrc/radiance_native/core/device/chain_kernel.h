/* chain_kernel.h -- the chained form's wait and completion for the core's own kernels
 * (RadKernelChainFn, abi/rad_abi.h).
 *
 * The backend gives a kernel named r4dchain* a record at RAD_CHAIN_KARG_OFF and drops its barrier
 * bit behind another chained kernel. Before the first read one lane waits for the queue's counter
 * to reach the record's target; after the last write the block releases and adds 1. A record that
 * does not check out means: run plain. Every block must reach rad_chain_done exactly once.
 */
#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>

#include "rad_abi.h"

struct RadChainRec {
    unsigned long long  magic, target;
    unsigned long long* ctr;
    unsigned long long  check;
};

__device__ __forceinline__ const RadChainRec* rad_chain_rec() {
    const RadChainRec* r = (const RadChainRec*)((const char*)__builtin_amdgcn_kernarg_segment_ptr() +
                                                RAD_CHAIN_KARG_OFF);
    const unsigned long long magic = r->magic, target = r->target, check = r->check;
    if (magic != RAD_CHAIN_MAGIC ||
        check != (magic ^ target ^ (unsigned long long)(uintptr_t)r->ctr ^ RAD_CHAIN_CHECK))
        return nullptr;
    return r;
}

__device__ __forceinline__ void rad_chain_wait() {
    const RadChainRec* r = rad_chain_rec();
    if (!r) return;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        while (__hip_atomic_load(r->ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < r->target)
            __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

__device__ __forceinline__ void rad_chain_done() {
    const RadChainRec* r = rad_chain_rec();
    if (!r) return;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        __hip_atomic_fetch_add(r->ctr, 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
