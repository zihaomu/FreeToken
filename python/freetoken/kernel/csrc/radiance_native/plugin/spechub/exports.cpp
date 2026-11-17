/* plugin/spechub/exports.cpp -- the plugin's export block.
 *
 * The whole kernel-plugin set (schema count/at, kernel count/at, ABI version, info) and nothing else.
 * The rows themselves, their schemas and their argument readers are in spechub.cpp; the launches are
 * spechub.hip (device) and ref_spechub.cpp (host).
 *
 * Reference status is per row: the library carries device rows and their host oracle, so
 * `rad_kernel_reference` marks the device rows 0 and the host rows 1, and `rad_plugin_reference`
 * stays 0. Marking the whole library would make the reference gate refuse the device rows. On a
 * machine with no card only the reference rows exist, and the gate refuses the coupling there.
 *
 * The ops are declared only under --draft-coupling on; the plugin never reads the flag.
 */
#include "spechub.h"
#include "rad_abi_layout.h"

#ifdef RAD_SPECHUB_WITH_DEVICE
#include "../../libr4d/r4d_emit.h"   /* only for the three globals it declares */

/* r4d_emit.h's dispatch recorder, defined here because plugins load RTLD_LOCAL and cannot
 * resolve libr4d's copy. Nothing in this plugin sets the log, so the recorder stays inert. */
const void** r4d_dispatch_log = nullptr;
int          r4d_dispatch_cap = 0;
int          r4d_dispatch_n   = 0;
#endif

namespace {
const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "spechub", "0.1.0",
    "Maximal coupling of speculative drafts: Q draws (MTP, DFlash2), coupled verify", "gfx1201",
};
}  /* namespace */

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
/* Mixed library: 0, and the per-row answer below does the classifying. */
extern "C" int rad_plugin_reference(void) { return 0; }

/* Row layout is [device 0..2, ref 0..2] with a card, [ref 0..2] without; row/op index 2 is
 * dflash_couple_propose. The device rows are the production kernels; the host rows are the oracle
 * rad-kbench checks them against and the only thing a card-less machine can run. */
extern "C" int rad_kernel_reference(int i) {
#ifdef RAD_SPECHUB_WITH_DEVICE
    return (i >= 0 && i < RAD_SPECHUB_N_OPS) ? 0 : 1;
#else
    (void)i;
    return 1;
#endif
}

extern "C" int rad_kernel_schema_count(void) { return rad_spechub_schema_count(); }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) { return rad_spechub_schema_at(i); }

/* The device rows and the host references beside them, in that order. A card-less machine loads the
 * same library and resolves only the reference rows. */
extern "C" int rad_kernel_count(void) {
#ifdef RAD_SPECHUB_WITH_DEVICE
    return 2 * RAD_SPECHUB_N_OPS;
#else
    return RAD_SPECHUB_N_OPS;
#endif
}

namespace {
RadKernelInfo ref_row(int op, RadLaunchFn launch, const char* name) {
    RadKernelInfo x = rad_spechub_row(op, launch);
    x.name = name;
    x.domain = RAD_DOMAIN_HOST;
    x.priority = 0;
    return x;
}
}  /* namespace */

extern "C" const RadKernelInfo* rad_kernel_at(int i) {
#ifdef RAD_SPECHUB_WITH_DEVICE
    static const RadKernelInfo dev[RAD_SPECHUB_N_OPS] = {
        rad_spechub_row(0, rad_mine_mtp_couple_propose_launch),
        rad_spechub_row(1, rad_mine_mtp_couple_verify_launch),
        rad_spechub_row(2, rad_mine_dflash_couple_launch),
    };
    if (i >= 0 && i < RAD_SPECHUB_N_OPS) return &dev[i];
    i -= RAD_SPECHUB_N_OPS;
#endif
    static const RadKernelInfo ref[RAD_SPECHUB_N_OPS] = {
        ref_row(0, rad_ref_mtp_couple_propose, "ref_mtp_couple_propose"),
        ref_row(1, rad_ref_mtp_couple_verify, "ref_mtp_couple_verify"),
        ref_row(2, rad_ref_dflash_couple, "ref_dflash_couple_propose"),
    };
    if (i < 0 || i >= RAD_SPECHUB_N_OPS) return nullptr;
    return &ref[i];
}
