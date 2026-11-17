/* advance.hip -- the advance kernel: one thread per row of the batch it patches. See advance.h.
 *
 * Launched through hipLaunchKernel, which the AQL backend intercepts, so it is one dispatch on the
 * rank's own queue, in stream order with its pass.
 */
#include <hip/hip_runtime.h>

#include <cstdlib>

#include "advance_impl.h"
#include "device/chain_kernel.h"
#include "rad_abi.h"

extern "C" int rad_dev_chain_enabled(void);   /* core/device/device.h */

namespace {

constexpr int kThreads = 64;

__global__ __launch_bounds__(kThreads) void rad_step_advance(rad::AdvanceArgs a) {
    const int i = (int)(blockIdx.x * kThreads + threadIdx.x);
    if (i < a.n_seq) rad::adv_row(a, i);
}

/* The chained forms (chain_kernel.h), launched where launches chain. */
__global__ __launch_bounds__(kThreads) void r4dchain_step_advance(rad::AdvanceArgs a) {
    rad_chain_wait();
    const int i = (int)(blockIdx.x * kThreads + threadIdx.x);
    if (i < a.n_seq) rad::adv_row(a, i);
    rad_chain_done();
}

/* ADV_COMPACT: block (x, y) is work item x (a pool, layer and head) of batch row y; its threads
 * split the K|V row into 16-byte chunks and its last words. Chained like the advance where launches chain, every
 * block waiting for the kernels before it and counting itself in after its moves. */
constexpr int kCompactThreads = 64;
__device__ __forceinline__ void step_compact_body(const rad::AdvanceArgs& a) {
    int p = 0, l = 0, h = 0;
    if (!rad::adv_compact_item(a, (int)blockIdx.x, &p, &l, &h)) return;
    rad::adv_compact(a, (int)blockIdx.y, p, l, h, (int)threadIdx.x, kCompactThreads);
}
__global__ __launch_bounds__(kCompactThreads) void rad_step_compact(rad::AdvanceArgs a) {
    step_compact_body(a);
}
__global__ __launch_bounds__(kCompactThreads) void r4dchain_step_compact(rad::AdvanceArgs a) {
    rad_chain_wait();
    step_compact_body(a);
    rad_chain_done();
}

}  // namespace

extern "C" int rad_step_compact_launch(const rad::AdvanceArgs* args, void* stream) {
    if (!args || args->n_seq <= 0) return 0;
    const int items = rad::adv_compact_items(*args);
    if (items <= 0) return 0;
    if (!rad_dev_chain_enabled())
        hipLaunchKernelGGL(rad_step_compact, dim3((unsigned)items, (unsigned)args->n_seq),
                           dim3(kCompactThreads), 0, (hipStream_t)stream, *args);
    else
        hipLaunchKernelGGL(r4dchain_step_compact, dim3((unsigned)items, (unsigned)args->n_seq),
                           dim3(kCompactThreads), 0, (hipStream_t)stream, *args);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

extern "C" int rad_step_advance_launch(const rad::AdvanceArgs* args, void* stream) {
    if (!args || args->n_seq <= 0) return 0;
    const unsigned blocks = (unsigned)((args->n_seq + kThreads - 1) / kThreads);
    /* The chained form only where launches chain (rad_dev_chain_enabled): there the backend writes
     * its record; elsewhere a launch carries only these arguments, so the plain kernel runs. */
    if (!rad_dev_chain_enabled())
        hipLaunchKernelGGL(rad_step_advance, dim3(blocks), dim3(kThreads), 0, (hipStream_t)stream,
                           *args);
    else
        hipLaunchKernelGGL(r4dchain_step_advance, dim3(blocks), dim3(kThreads), 0,
                           (hipStream_t)stream, *args);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}
