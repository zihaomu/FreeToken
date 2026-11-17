/* lq_gpu.cpp -- the host-only build's answer to lq_gpu.h: no card path (lq_tcq_gpu.hip is built
 * only where radiance has a device build). */
#include "lq_gpu.h"

#ifndef LQ_CARD
namespace lq {

int gpu_set_devices(const int*, int n) { return n > 0 ? RAD_E_UNSUPPORTED : RAD_OK; }

bool gpu_enabled() { return false; }

int gptq_tcq_gpu(const Grid&, float*, int32_t*, Scales&, int64_t, int64_t, const float*, bool,
                 std::string* why) {
    *why = "this libquant was built without a device path";
    return RAD_E_UNSUPPORTED;
}

}  /* namespace lq */
#endif
