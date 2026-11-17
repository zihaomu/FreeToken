// r4d_emit.h -- the one launch path every libr4d kernel goes through.
//
// `hipLaunchKernel` takes the arguments as an array of pointers (kernelParams) and copies each
// slot's bytes without conversion, so a `long long` passed for a `uint32_t` parameter would fill a
// four-byte slot with eight bytes and corrupt every later argument. r4d_emit takes the kernel
// typed, converts every argument to its declared parameter type, and checks the arity at compile
// time. It also traces each dispatch under RAD_R4D_TRACE and feeds r4d_record_dispatches (r4d.h).

#ifndef R4D_EMIT_H
#define R4D_EMIT_H

#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <tuple>

#include "rad_abi.h"
#include "r4d_args.h"

/* r4d_record_dispatches' log (r4d_rows.cpp): while `r4d_dispatch_log` is set, each launch appends
 * its function and counts it in `r4d_dispatch_n`, which may run past `r4d_dispatch_cap`. */
extern const void** r4d_dispatch_log;
extern int          r4d_dispatch_cap;
extern int          r4d_dispatch_n;

/* `func` is the kernel's host stub; for a template, the instantiation's own symbol. */
template <class... P, class... A>
inline int r4d_emit(RadStream stream, void (*kernel)(P...), dim3 grid, dim3 block,
                    uint32_t shared_bytes, const A&... args) {
    static_assert(sizeof...(P) == sizeof...(A),
                  "a kernel argument list must match the kernel's parameter list");
    const void* func = (const void*)kernel;
    std::tuple<P...> vals(static_cast<P>(args)...);
    void* params[sizeof...(P) + 1];
    std::apply([&](P&... v) {
        int i = 0;
        ((params[i++] = (void*)&v), ...);
    }, vals);
    params[sizeof...(P)] = nullptr;

    if (r4d_dispatch_log) {
        if (r4d_dispatch_n < r4d_dispatch_cap) r4d_dispatch_log[r4d_dispatch_n] = func;
        ++r4d_dispatch_n;
    }
    /* Trace the dispatch before submitting it: a fault inside the driver leaves only this frame,
     * with the geometry in reused registers and the kernel a stub address. An `emit` with no
     * `ret` after it is the faulting dispatch; flushed because the fault may be next. Each line is
     * one write, since every rank is a thread and piecewise fprintf calls would interleave. */
    if (r4d_trace_on()) {
        static constexpr size_t kSize[] = { sizeof(P)..., 0 };
        char line[1024];
        int  o = snprintf(line, sizeof line,
                          "[r4d] emit %p grid=%ux%ux%u block=%ux%ux%u lds=%u np=%u",
                          func, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                          shared_bytes, (unsigned)sizeof...(P));
        for (size_t k = 0; k < sizeof...(P) && o > 0 && o < (int)sizeof line - 32; ++k) {
            unsigned long long v = 0;
            memcpy(&v, params[k], kSize[k] > 8 ? 8 : kSize[k]);
            o += snprintf(line + o, sizeof line - (size_t)o, " %u:%llx", (unsigned)kSize[k], v);
        }
        std::fprintf(stderr, "%s\n", line);
        std::fflush(stderr);
    }
    const int rc = hipLaunchKernel(func, grid, block, params, shared_bytes,
                                   (hipStream_t)stream) == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    if (r4d_trace_on()) { std::fprintf(stderr, "[r4d] ret %p %d\n", func, rc); std::fflush(stderr); }
    return rc;
}

/* `kernel` is named, not passed as a value, so a template instantiation is spelled at the call
 * site exactly as the triple-chevron spelled it. A test may define R4D_EMIT to drive the same
 * sources through a host SIMT emulator, as with R4D_KLAUNCH (r4d_launch.h). */
#ifndef R4D_EMIT
#define R4D_EMIT(stream, kernel, grid, block, lds, ...) \
    r4d_emit((stream), (kernel), (grid), (block), (uint32_t)(lds), __VA_ARGS__)
#endif

#endif  /* R4D_EMIT_H */
