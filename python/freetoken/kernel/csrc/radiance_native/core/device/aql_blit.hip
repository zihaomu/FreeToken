/* aql_blit.hip -- copy and fill kernels behind rad_memcpy_async, rad_memcpy_2d_async and
 * rad_memset_async on the AQL backend.
 *
 * A copy is a kernel on the stream's own queue, like HIP's blit path. The DMA engine would be a
 * second queue to order against with signals, and its latency exceeds a small copy's run time.
 * Every address is GPU-reachable (device, peer or pinned host memory); the caller stages
 * pageable host memory.
 *
 * The launchers go through hipLaunchKernel, which the backend intercepts, so these kernels are
 * dispatched like any kernel library's.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "aql.h"
#include "chain_kernel.h"
#include "device.h"

namespace {

constexpr int kThreads = 256;
constexpr uint64_t kMaxBlocks = 2048;

typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

/* Four loads in flight per thread before the first store: a copy from pinned host memory is
 * link-latency bound, and one outstanding load per thread leaves most of the link idle. */
__global__ __launch_bounds__(kThreads) void aql_copy16(const u32x4* __restrict__ src,
                                                       u32x4* __restrict__ dst, uint64_t n) {
    const uint64_t stride = (uint64_t)gridDim.x * kThreads;
    uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        const u32x4 a = src[i], b = src[i + stride], c = src[i + 2 * stride], d = src[i + 3 * stride];
        dst[i] = a; dst[i + stride] = b; dst[i + 2 * stride] = c; dst[i + 3 * stride] = d;
    }
    for (; i < n; i += stride) dst[i] = src[i];
}

/* aql_copy16 as a chained form (chain_kernel.h), launched where launches chain: a card-side copy
 * between two kernels of a pass (a routed layer's counts into the core's row) then costs no
 * barrier on either side. */
__global__ __launch_bounds__(kThreads) void r4dchain_copy16(const u32x4* __restrict__ src,
                                                            u32x4* __restrict__ dst, uint64_t n) {
    rad_chain_wait();
    const uint64_t stride = (uint64_t)gridDim.x * kThreads;
    uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        const u32x4 a = src[i], b = src[i + stride], c = src[i + 2 * stride], d = src[i + 3 * stride];
        dst[i] = a; dst[i + stride] = b; dst[i + 2 * stride] = c; dst[i + 3 * stride] = d;
    }
    for (; i < n; i += stride) dst[i] = src[i];
    rad_chain_done();
}

/* A copy across the link (pinned host memory) is one workgroup of four waves.
 *
 * The link delivers ~29 GB/s regardless of grid size, so the copy should hold as few waves as
 * possible: every wave it holds is denied to compute kernels for the copy's duration (a wide link
 * copy beside each layer slows a staged prefill by ~15%). Link bandwidth by grid, one card,
 * 89 MB from pinned host memory:
 *
 *     16 x 256 threads, 4 loads in flight    28.9 GB/s
 *      1 x 256 threads, 16 in flight         28.9
 *      1 x 128 threads, 16 in flight         28.9           <- this
 *      1 x  64 threads, 16 in flight         16.8
 *      1 x  64 threads,  8 in flight         10.4
 *
 * 32 KiB in flight is several times the link's bandwidth-delay product; fewer threads leave the
 * link idle and 32 loads per thread spill. Non-temporal loads and stores make no difference.
 *
 * The last round clamps its load indices instead of guarding them, so all its loads are in flight
 * at once; a guarded load waits for the one before it. */
constexpr int kLinkThreads = 128, kLinkDepth = 16;
constexpr uint64_t kLinkByteBlocks = 16;

__global__ __launch_bounds__(kLinkThreads) void aql_copy_link(const u32x4* __restrict__ src,
                                                              u32x4* __restrict__ dst, uint64_t n) {
    uint64_t i = threadIdx.x;
    for (; i + (uint64_t)(kLinkDepth - 1) * kLinkThreads < n; i += (uint64_t)kLinkDepth * kLinkThreads) {
        u32x4 v[kLinkDepth];
#pragma unroll
        for (int k = 0; k < kLinkDepth; ++k) v[k] = src[i + (uint64_t)k * kLinkThreads];
#pragma unroll
        for (int k = 0; k < kLinkDepth; ++k) dst[i + (uint64_t)k * kLinkThreads] = v[k];
    }
    if (i >= n) return;
    u32x4 v[kLinkDepth];
#pragma unroll
    for (int k = 0; k < kLinkDepth; ++k) {
        const uint64_t j = i + (uint64_t)k * kLinkThreads;
        v[k] = src[j < n ? j : n - 1];
    }
#pragma unroll
    for (int k = 0; k < kLinkDepth; ++k) {
        const uint64_t j = i + (uint64_t)k * kLinkThreads;
        if (j < n) dst[j] = v[k];
    }
}

__global__ __launch_bounds__(kThreads) void aql_copy1(const unsigned char* __restrict__ src,
                                                      unsigned char* __restrict__ dst, uint64_t n) {
    const uint64_t stride = (uint64_t)gridDim.x * kThreads;
    for (uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x; i < n; i += stride)
        dst[i] = src[i];
}

__global__ __launch_bounds__(kThreads) void aql_copy2d16(const u32x4* __restrict__ src,
                                                         u32x4* __restrict__ dst, uint64_t spitch16,
                                                         uint64_t dpitch16, uint64_t w16,
                                                         uint64_t height) {
    const uint64_t cstride = (uint64_t)gridDim.x * kThreads;
    for (uint64_t r = blockIdx.y; r < height; r += gridDim.y) {
        const u32x4* s = src + r * spitch16;
        u32x4* d = dst + r * dpitch16;
        for (uint64_t c = (uint64_t)blockIdx.x * kThreads + threadIdx.x; c < w16; c += cstride)
            d[c] = s[c];
    }
}

__global__ __launch_bounds__(kThreads) void aql_copy2d1(const unsigned char* __restrict__ src,
                                                        unsigned char* __restrict__ dst,
                                                        uint64_t spitch, uint64_t dpitch,
                                                        uint64_t width, uint64_t height) {
    const uint64_t cstride = (uint64_t)gridDim.x * kThreads;
    for (uint64_t r = blockIdx.y; r < height; r += gridDim.y) {
        const unsigned char* s = src + r * spitch;
        unsigned char* d = dst + r * dpitch;
        for (uint64_t c = (uint64_t)blockIdx.x * kThreads + threadIdx.x; c < width; c += cstride)
            d[c] = s[c];
    }
}

/* Ranged copy: the first `prefix_blocks` workgroups copy the prefix as aql_copy16 does, and each
 * later workgroup copies one range. Ranges are only 4-byte aligned, so they move in words, 16 per
 * thread per round, with all loads issued before the first store (each is a read across the
 * link). Loads past the range end clamp to its last word instead of branching, since a guarded
 * load waits for the one before it. */
constexpr int kRangeWords = 16;

__global__ __launch_bounds__(kThreads) void aql_copy_ranges(const u32x4* __restrict__ src16,
                                                            u32x4* __restrict__ dst16, uint64_t n16,
                                                            unsigned prefix_blocks,
                                                            const RadCopyRange* __restrict__ r) {
    if (blockIdx.x < prefix_blocks) {
        const uint64_t stride = (uint64_t)prefix_blocks * kThreads;
        uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x;
        for (; i + 3 * stride < n16; i += 4 * stride) {
            const u32x4 a = src16[i], b = src16[i + stride], c = src16[i + 2 * stride],
                        d = src16[i + 3 * stride];
            dst16[i] = a; dst16[i + stride] = b; dst16[i + 2 * stride] = c; dst16[i + 3 * stride] = d;
        }
        for (; i < n16; i += stride) dst16[i] = src16[i];
        return;
    }
    const RadCopyRange g = r[blockIdx.x - prefix_blocks];
    const unsigned* s4 = (const unsigned*)((const char*)src16 + g.src);
    unsigned* d4 = (unsigned*)((char*)dst16 + g.dst);
    const int64_t n = g.bytes / 4;
    for (int64_t base = 0; base < n; base += (int64_t)kThreads * kRangeWords) {
        unsigned v[kRangeWords];
#pragma unroll
        for (int k = 0; k < kRangeWords; ++k) {
            const int64_t w = base + (int64_t)k * kThreads + threadIdx.x;
            v[k] = s4[w < n ? w : n - 1];
        }
#pragma unroll
        for (int k = 0; k < kRangeWords; ++k) {
            const int64_t w = base + (int64_t)k * kThreads + threadIdx.x;
            if (w < n) d4[w] = v[k];
        }
    }
}

__global__ __launch_bounds__(kThreads) void aql_fill16(u32x4* __restrict__ dst, unsigned v,
                                                       uint64_t n) {
    const uint64_t stride = (uint64_t)gridDim.x * kThreads;
    const u32x4 w = { v, v, v, v };
    for (uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x; i < n; i += stride) dst[i] = w;
}

__global__ __launch_bounds__(kThreads) void aql_fill1(unsigned char* __restrict__ dst,
                                                      unsigned char v, uint64_t n) {
    const uint64_t stride = (uint64_t)gridDim.x * kThreads;
    for (uint64_t i = (uint64_t)blockIdx.x * kThreads + threadIdx.x; i < n; i += stride) dst[i] = v;
}

inline unsigned blocks_for(uint64_t n, uint64_t per_block, uint64_t cap = kMaxBlocks) {
    uint64_t b = (n + per_block - 1) / per_block;
    if (b < 1) b = 1;
    if (b > cap) b = cap;
    return (unsigned)b;
}

inline bool aligned16(uint64_t v) { return (v & 15u) == 0; }

inline int launched() { return hipGetLastError() == hipSuccess ? 0 : -1; }

}  // namespace

extern "C" int rad_aql_blit_copy(void* dst, const void* src, uint64_t bytes, void* stream,
                                 bool link) {
    if (bytes == 0) return 0;
    const hipStream_t s = (hipStream_t)stream;
    if (aligned16((uint64_t)(uintptr_t)dst | (uint64_t)(uintptr_t)src | bytes)) {
        const uint64_t n = bytes / 16;
        if (link)
            hipLaunchKernelGGL(aql_copy_link, dim3(1), dim3(kLinkThreads), 0, s,
                               (const u32x4*)src, (u32x4*)dst, n);
        else if (rad_dev_chain_enabled())
            hipLaunchKernelGGL(r4dchain_copy16, dim3(blocks_for(n, 4 * kThreads)), dim3(kThreads),
                               0, s, (const u32x4*)src, (u32x4*)dst, n);
        else
            hipLaunchKernelGGL(aql_copy16, dim3(blocks_for(n, 4 * kThreads)), dim3(kThreads), 0,
                               s, (const u32x4*)src, (u32x4*)dst, n);
    } else {
        /* Unaligned: byte copies have one byte per thread in flight, so across the link they
         * use 16 workgroups instead of the aligned path's one. */
        const uint64_t cap = link ? kLinkByteBlocks : kMaxBlocks;
        hipLaunchKernelGGL(aql_copy1, dim3(blocks_for(bytes, 4 * kThreads, cap)), dim3(kThreads),
                           0, s, (const unsigned char*)src, (unsigned char*)dst, bytes);
    }
    return launched();
}

extern "C" int rad_aql_blit_copy2d(void* dst, uint64_t dpitch, const void* src, uint64_t spitch,
                                   uint64_t width, uint64_t height, void* stream) {
    if (width == 0 || height == 0) return 0;
    const hipStream_t s = (hipStream_t)stream;
    const unsigned by = (unsigned)(height < 65535 ? height : 65535);
    if (aligned16((uint64_t)(uintptr_t)dst | (uint64_t)(uintptr_t)src | dpitch | spitch | width)) {
        const uint64_t w16 = width / 16;
        hipLaunchKernelGGL(aql_copy2d16, dim3(blocks_for(w16, kThreads), by), dim3(kThreads), 0, s,
                           (const u32x4*)src, (u32x4*)dst, spitch / 16, dpitch / 16, w16, height);
    } else {
        hipLaunchKernelGGL(aql_copy2d1, dim3(blocks_for(width, kThreads), by), dim3(kThreads), 0, s,
                           (const unsigned char*)src, (unsigned char*)dst, spitch, dpitch, width,
                           height);
    }
    return launched();
}

extern "C" int rad_aql_blit_ranges(void* dst, const void* src, uint64_t prefix,
                                   const RadCopyRange* ranges, int n, void* stream) {
    const uint64_t n16 = prefix / 16;
    const unsigned pb = n16 ? blocks_for(n16, 4 * kThreads) : 0;
    if (pb == 0 && n == 0) return 0;
    hipLaunchKernelGGL(aql_copy_ranges, dim3(pb + (unsigned)n), dim3(kThreads), 0,
                       (hipStream_t)stream, (const u32x4*)src, (u32x4*)dst, n16, pb, ranges);
    return launched();
}

extern "C" int rad_aql_blit_fill(void* dst, int value, uint64_t bytes, void* stream) {
    if (bytes == 0) return 0;
    const hipStream_t s = (hipStream_t)stream;
    const unsigned char b = (unsigned char)value;
    if (aligned16((uint64_t)(uintptr_t)dst | bytes)) {
        const unsigned v = 0x01010101u * b;
        const uint64_t n = bytes / 16;
        hipLaunchKernelGGL(aql_fill16, dim3(blocks_for(n, 4 * kThreads)), dim3(kThreads), 0, s,
                           (u32x4*)dst, v, n);
    } else {
        hipLaunchKernelGGL(aql_fill1, dim3(blocks_for(bytes, 4 * kThreads)), dim3(kThreads), 0, s,
                           (unsigned char*)dst, b, bytes);
    }
    return launched();
}
