/* rad_abi_layout.h -- the plugin ABI's layout fingerprint: a hash of the size of every struct abi/
 * exports and of the offset and size of each of its fields (rad_abi_layout_rows.h), as the
 * translation unit that includes this header sees them.
 *
 * RAD_ABI_VERSION names a layout, but two trees can give the same number to different layouts, and
 * a plugin built against one then reads the other's structs at the wrong offsets, with no error. A
 * plugin exports its own fingerprint (RAD_PLUGIN_ABI_LAYOUT_EXPORT, once in the plugin), and the
 * loader refuses one whose fingerprint differs from the engine's under the same version. C and C++.
 */
#ifndef RAD_ABI_LAYOUT_H
#define RAD_ABI_LAYOUT_H

#include "rad_abi.h"
#include "rad_builder.h"
#include "rad_device.h"
#include "rad_encoding.h"
#include "rad_format.h"
#include "rad_philox.h"
#include "rad_planes.h"
#include "rad_plugin.h"
#include "rad_quant.h"
#include "rad_runtime.h"
#include "rad_sample.h"
#include "rad_spechub.h"
#include "rad_types.h"
#include "rad_yarn.h"

#include <stddef.h>
#include <stdint.h>

/* FNV-1a over the eight bytes of `v`, little end first. */
static inline uint64_t rad_abi_layout_mix(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h ^= (v >> (8 * i)) & 0xffu;
        h *= 0x100000001b3ull;
    }
    return h;
}

/* The fingerprint of the abi/ headers this translation unit was compiled against. */
static inline uint64_t rad_abi_layout(void) {
    uint64_t h = 0xcbf29ce484222325ull;
    h = rad_abi_layout_mix(h, RAD_ABI_VERSION);
#define ABI_STRUCT(T) h = rad_abi_layout_mix(h, sizeof(T));
#define ABI_FIELD(T, f)                                         \
    h = rad_abi_layout_mix(h, offsetof(T, f));                  \
    h = rad_abi_layout_mix(h, sizeof(((T*)0)->f));
#include "rad_abi_layout_rows.h"
#undef ABI_STRUCT
#undef ABI_FIELD
    return h;
}

/* The export the loader reads beside rad_plugin_abi_version: written once in a plugin, at file
 * scope. */
#ifdef __cplusplus
#define RAD_PLUGIN_ABI_LAYOUT_EXPORT \
    extern "C" uint64_t rad_plugin_abi_layout(void) { return rad_abi_layout(); }
#else
#define RAD_PLUGIN_ABI_LAYOUT_EXPORT \
    uint64_t rad_plugin_abi_layout(void) { return rad_abi_layout(); }
#endif

#endif /* RAD_ABI_LAYOUT_H */
