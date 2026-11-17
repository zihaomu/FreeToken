/* advance.cpp -- issuing an advance. See advance.h. */
#include "advance_impl.h"

#include "device/device.h"

namespace rad {

int step_advance(const AdvanceArgs& a, RadStream s) {
    if (a.n_seq <= 0) return RAD_OK;
    /* Host backend streams are queues run by worker threads (device/host.cpp), and device memory
     * is host memory. Drain the stream first so the rows run where the kernel would in stream
     * order; otherwise they can read a sampled token or staged row whose copy is still queued. */
    if (device_is_host()) {
        RAD_TRY(rad_stream_sync(s));
        for (int i = 0; i < a.n_seq; ++i) adv_row(a, i);
        return RAD_OK;
    }
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    return rad_step_advance_launch(&a, (void*)s) == 0 ? RAD_OK : RAD_E_DEVICE;
#else
    (void)s;
    return RAD_E_UNSUPPORTED;
#endif
}

int step_compact(const AdvanceArgs& a, RadStream s) {
    if (a.n_seq <= 0 || a.n_pool <= 0) return RAD_OK;
    if (device_is_host()) {
        RAD_TRY(rad_stream_sync(s));
        const int items = adv_compact_items(a);
        for (int i = 0; i < a.n_seq; ++i)
            for (int w = 0; w < items; ++w) {
                int p = 0, l = 0, h = 0;
                if (adv_compact_item(a, w, &p, &l, &h)) adv_compact(a, i, p, l, h, 0, 1);
            }
        return RAD_OK;
    }
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    return rad_step_compact_launch(&a, (void*)s) == 0 ? RAD_OK : RAD_E_DEVICE;
#else
    (void)s;
    return RAD_E_UNSUPPORTED;
#endif
}

}  /* namespace rad */
