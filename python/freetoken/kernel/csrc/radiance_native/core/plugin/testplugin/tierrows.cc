/* tierrows.cc -- device-domain byte-row gather and scatter on the host backend, for the KV tiers'
 * transfer test (tests/mem_test.cpp).
 *
 * The tiers move KV through `gather_rows` and `scatter_rows` resolved in the DEVICE domain, and the
 * reference library registers those ops in the host domain only -- so without this, the one piece
 * of the tier that can put another conversation's context into a restored entry (the addressing:
 * layer-major strides, staging pitch, arena runs, a rank's window of a shared slot) could only be
 * tested on a card. Here the host backend's memory stands in for the card's.
 *
 * STREAM ORDER IS KEPT BY WAITING. The host backend runs each stream on a worker thread, and a
 * launch here runs on the caller's thread; draining the stream first puts the kernel exactly where
 * a device launch would sit -- after everything enqueued before it and before everything enqueued
 * after it returns. The test executable exports the device API for that call (tests/CMakeLists.txt).
 *
 * It lives in its own directory, under a name no hierarchy lists, so nothing but the test that asks
 * for it by path ever loads it: a kernel that computes, registered in the device domain, is exactly
 * what a real run must never select by accident.
 */
#include "rad_abi.h"
#include "rad_device.h"
#include "rad_abi_layout.h"

#include <cstdint>
#include <cstring>

#define NELEM(a) (int)(sizeof(a) / sizeof((a)[0]))

namespace {

/* Rows of bytes: shape[0] rows of shape[1] bytes, rows `stride[0]` bytes apart. */
struct Rows {
    char*   p;
    int64_t n, width, pitch;
};
bool rows_of(const RadTensor& t, Rows* out) {
    if (t.rank != 2 || t.dtype != RAD_U8 || !t.data || t.stride[1] != 1) return false;
    out->p = (char*)t.data;
    out->n = t.shape[0];
    out->width = t.shape[1];
    out->pitch = t.stride[0];
    return true;
}

int copy_rows(const RadArgs* a, RadStream s, bool scatter) {
    if (!a || a->n_t != 3) return RAD_E_INVAL;
    if (rad_stream_sync(s) != RAD_OK) return RAD_E_DEVICE;
    /* gather: pool, idx, pack.   scatter: pack, idx, pool. */
    Rows pool, pack;
    if (!rows_of(a->t[scatter ? 2 : 0], &pool) || !rows_of(a->t[scatter ? 0 : 2], &pack))
        return RAD_E_SHAPE;
    const RadTensor& idx = a->t[1];
    if (idx.dtype != RAD_I32 || !idx.data || idx.shape[0] < pack.n || pack.width != pool.width)
        return RAD_E_SHAPE;
    const int32_t* ix = (const int32_t*)idx.data;
    for (int64_t i = 0; i < pack.n; ++i) {
        const int32_t r = ix[i];
        if (r >= pool.n) return RAD_E_INVAL;
        char* row = pack.p + i * pack.pitch;
        if (r < 0) {
            if (!scatter) std::memset(row, 0, (size_t)pack.width);
            continue;
        }
        char* slot = pool.p + (int64_t)r * pool.pitch;
        if (scatter) std::memcpy(slot, row, (size_t)pack.width);
        else         std::memcpy(row, slot, (size_t)pack.width);
    }
    return RAD_OK;
}

int t_gather(const RadArgs* a, RadStream s)  { return copy_rows(a, s, false); }
int t_scatter(const RadArgs* a, RadStream s) { return copy_rows(a, s, true); }

}  /* namespace */

extern "C" {

static const RadParamSpec kP[] = {
    { "M", RAD_P_INT, RAD_REQUIRED }, { "n", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec kGather[]  = {
    { "x", RAD_OPD_IN, 0 }, { "idx", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadOperandSpec kScatter[] = {
    { "v", RAD_OPD_IN, 0 }, { "idx", RAD_OPD_IN, 0 }, { "x", RAD_OPD_INOUT, 0 },
};
static const RadOpSchema kSchemas[] = {
    { "gather_rows", kP, NELEM(kP), kGather, NELEM(kGather), "y[i,:] = x[idx[i],:]" },
    { "scatter_rows", kP, NELEM(kP), kScatter, NELEM(kScatter), "x[idx[i],:] = v[i,:]" },
};
static const RadKernelInfo kKernels[] = {
    { .name = "t_rows_gather", .op = "gather_rows", .family = "elem",
      .computes = "byte rows, on the host backend's memory", .shape = "any", .dtypes = "u8",
      .domain = RAD_DOMAIN_DEVICE, .priority = 0, .launch = t_gather },
    { .name = "t_rows_scatter", .op = "scatter_rows", .family = "elem",
      .computes = "byte rows, on the host backend's memory", .shape = "any", .dtypes = "u8",
      .domain = RAD_DOMAIN_DEVICE, .priority = 0, .launch = t_scatter },
};
static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "radtest_rows", "0.1",
    "device-domain byte-row gather/scatter for the KV tier transfer test", "host",
};

uint32_t             rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
int                  rad_kernel_schema_count(void) { return NELEM(kSchemas); }
const RadOpSchema*   rad_kernel_schema_at(int i) {
    return (i < 0 || i >= NELEM(kSchemas)) ? nullptr : &kSchemas[i];
}
int                  rad_kernel_count(void) { return NELEM(kKernels); }
const RadKernelInfo* rad_kernel_at(int i) {
    return (i < 0 || i >= NELEM(kKernels)) ? nullptr : &kKernels[i];
}

}  /* extern "C" */
