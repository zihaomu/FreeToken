/* wtab.h -- writing entries of a device weight table in stream order.
 *
 * A routed layer's weight table is a device array of expert pointers; the mover changes a few
 * entries on most steps. The new values travel in the kernel arguments, which the backend copies
 * at enqueue. A copy from a host array would read it when the card reaches the copy, and since the
 * host runs ahead, it could then hold the next step's pointers, including one whose transfer this
 * step is not ordered behind.
 */
#pragma once
#include <cstdint>

#include "rad_device.h"

namespace rad {

/* One dispatch's worth of writes: dst[i] = val[i] for i < n. Sized so the argument block stays
 * inside the backend's per-dispatch argument slot with the implicit arguments beside it. */
constexpr int kWtabPatch = 24;

struct WtabPatch {
    uint32_t n   = 0;
    uint32_t pad = 0;
    void**   dst[kWtabPatch] = {};
    void*    val[kWtabPatch] = {};
};

/* Enqueue `p` on `s`. Under the host backend, whose device memory is host memory, the writes
 * happen here once `s` has run everything issued before them. */
int wtab_patch(const WtabPatch& p, RadStream s);

}  /* namespace rad */

/* The launcher in wtab.hip: one kernel onto `stream` through hipLaunchKernel. */
extern "C" int rad_wtab_patch_launch(const rad::WtabPatch* p, void* stream);
