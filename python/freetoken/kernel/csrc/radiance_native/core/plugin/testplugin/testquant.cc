/* testquant.cc -- a quantiser plugin with a scheme of its own, for tests/quant_test.cpp.
 *
 * "radtest-halves" stores each value as the nearest multiple of one half, as an i8 code that is
 * twice the value, and nothing else: one plane, no scale, a scheme the core does not define. What
 * it exists to show is that a weight written that way is READ through the quantiser that wrote it
 * -- enc_decode_rows asks the loaded quantisers' decode hooks for any scheme but plain and affine
 * (abi/rad_quant.h) -- and that a decoder answers RAD_E_UNSUPPORTED for a scheme it did not write.
 */
#include "rad_quant.h"
#include "rad_abi_layout.h"

#include <cmath>
#include <cstring>

namespace {

const char kScheme[] = "radtest-halves";

int enc(const RadParam*, int, const RadQuantWeight* w, RadEncoding* out) {
    if (w->rank < 2) return RAD_E_UNSUPPORTED;
    rad_enc_clear(out);
    rad_enc_copy_str(out->scheme, kScheme);
    rad_enc_add_plane(out, "codes", RAD_I8, 1, 1);
    return RAD_OK;
}

int64_t row_block(const RadParam*, int, const RadQuantWeight*) { return 1; }

int quantize(const RadParam*, int, const RadQuantWeight* w, float* src, int64_t, int64_t rows,
             void* const* planes) {
    int8_t* q = static_cast<int8_t*>(planes[0]);
    for (int64_t i = 0; i < rows * w->cols; ++i) {
        const float v = std::nearbyint(src[i] * 2.0f);
        q[i] = (int8_t)(v > 127.f ? 127.f : v < -128.f ? -128.f : v);
    }
    return RAD_OK;
}

int decode(const RadEncoding* e, const RadQuantWeight* w, const void* const* planes, int64_t row0,
           int64_t rows, float* out) {
    if (std::strcmp(e->scheme, kScheme) != 0) return RAD_E_UNSUPPORTED;
    const int8_t* q = static_cast<const int8_t*>(planes[0]) + row0 * w->cols;
    for (int64_t i = 0; i < rows * w->cols; ++i) out[i] = 0.5f * (float)q[i];
    return RAD_OK;
}

const RadQuantizerInfo kQuant[] = {{
    "radtest-halves", "the nearest multiple of one half, as twice it in i8: a scheme of its own",
    nullptr, 0, enc, row_block, quantize, decode,
}};

const RadPluginInfo kInfo = {
    RAD_PLUGIN_QUANT, "radtest_quant", "0.1", "a quantiser whose scheme only it decodes", "host",
};

}  // namespace

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
extern "C" int rad_quant_count(void) { return 1; }
extern "C" const RadQuantizerInfo* rad_quant_at(int i) { return i == 0 ? &kQuant[0] : nullptr; }
