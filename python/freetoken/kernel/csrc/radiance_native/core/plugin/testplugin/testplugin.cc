/* testplugin.cc -- in-tree kernel plugins that exist to be loaded by tests/plugin_test.cpp.
 *
 * It computes nothing: every launch returns RAD_E_UNSUPPORTED, because a test kernel that returned
 * RAD_OK would be a slow path nobody knows about, waiting to be selected by a real model. What it
 * does have is a registry: rows that exercise every constraint operator, two plugins' worth of
 * hierarchy, a priority tie, a schema conflict that must be refused, and an op with kernels at
 * M<=16, M<=64 and unconstrained -- which is spec.md §2.2's worked example, so the test can assert
 * the exact band table and the paragraph becomes a regression test.
 *
 * One source, several .so files, selected by -D: kernel plugins that load, kernel plugins that must
 * be refused or that decline the machine, and architecture plugins that claim one id between them
 * or one family at several quantisations. The rows are worth reading side by side, and near-copies
 * of this file would drift.
 *
 * The extension is .cc, not .cpp, and that is load-bearing: core/CMakeLists.txt globs
 * every core/plugin .cpp file into the rad_core static library, and a plugin's exports
 * (rad_kernel_count and friends) do not belong inside the engine. The alternative is an
 * exclusion rule in a file this component does not own.
 */
#include "rad_abi.h"
#include "rad_planes.h"

#include <cstdlib>
#include <cstring>

#if defined(RADTEST_ARCH) || defined(RADTEST_ARCH2) || defined(RADTEST_ARCHQ)
#define RADTEST_IS_ARCH 1
#include "rad_builder.h"
#include "rad_runtime.h"
#endif
#include "rad_abi_layout.h"

#if !defined(RADTEST_BASE) && !defined(RADTEST_OVER) && !defined(RADTEST_CONFLICT) && \
    !defined(RADTEST_BADABI) && !defined(RADTEST_BADLAYOUT) && !defined(RADTEST_IS_ARCH) && \
    !defined(RADTEST_EXTRA) && \
    !defined(RADTEST_BADTUNE) && !defined(RADTEST_DECLINE)
#define RADTEST_BASE 1
#endif

#define NELEM(a) (int)(sizeof(a) / sizeof((a)[0]))

/* ------------------------------------------------------------------ hooks */
namespace {

int  g_open_calls = 0;
int  g_init_calls = 0;
int  g_fini_calls = 0;

struct TestInstance {
    long long m;        /* the frozen geometry's ranged value, so a test can prove which band's
                         * upper bound reached init */
    int rank, world;
};

/* Every row shares one launch, and it refuses. Nothing here can run. */
[[maybe_unused]] int t_launch(const RadArgs* a, RadStream s) {
    (void)a; (void)s;
    return RAD_E_UNSUPPORTED;
}

[[maybe_unused]] int t_init(const RadParam* p, int n_p, int rank, int world_size, void** out) {
    TestInstance* inst = (TestInstance*)std::calloc(1, sizeof(TestInstance));
    if (!inst) return RAD_E_NOMEM;
    inst->m = -1;
    inst->rank = rank;
    inst->world = world_size;
    for (int i = 0; i < n_p; ++i)
        if (p[i].kind == RAD_P_INT && p[i].key && std::strcmp(p[i].key, "M") == 0)
            inst->m = p[i].ival;
    *out = inst;
    ++g_init_calls;
    return RAD_OK;
}

[[maybe_unused]] void t_fini(void* inst) {
    ++g_fini_calls;
    std::free(inst);
}

/* A kernel whose one-time setup fails. A negative init aborts startup naming the kernel, and that
 * path needs a kernel that takes it. */
[[maybe_unused]] int t_init_fails(const RadParam* p, int n_p, int rank, int world_size, void** out) {
    (void)p; (void)n_p; (void)rank; (void)world_size; (void)out;
    return RAD_E_NOMEM;
}

}  /* namespace */

extern "C" {

int radtest_open_calls(void) { return g_open_calls; }
int radtest_init_calls(void) { return g_init_calls; }
int radtest_fini_calls(void) { return g_fini_calls; }
long long radtest_instance_m(void* inst) { return inst ? ((TestInstance*)inst)->m : -1; }

/* radtest_decline is a library for another machine: it declines this one at open. */
int rad_plugin_open(void) {
    ++g_open_calls;
#if defined(RADTEST_DECLINE)
    return RAD_E_UNSUPPORTED;
#else
    return RAD_OK;
#endif
}

/* ------------------------------------------------------------------ operands */
/* Roles matter to the core (a weight operand is resolved through a handle, an output feeds the
 * buffer planner); names are diagnostics only, which is why two plugins may spell them
 * differently and still agree on the schema. */
static const RadOperandSpec kOpdGemm[] = {
    { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadOperandSpec kOpdNorm[] = {
    { "x", RAD_OPD_IN, 0 }, { "w", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadOperandSpec kOpdAttn[] = {
    { "q", RAD_OPD_IN, 0 }, { "kv_cache", RAD_OPD_INOUT, 0 }, { "out", RAD_OPD_OUT, 0 },
};
/* A grouped GEMM's operand list in miniature: one fixed weight, then four classes of experts as
 * table pairs -- codes, then scales -- with an activation operand between the first pair and the
 * rest, the way `moe_gemm_q_mod4` lays its classes out. It is here for the declaration's division
 * of one weight list over several table positions (tests/build_test.cpp). */
static const RadOperandSpec kOpdMoe[] = {
    { "a", RAD_OPD_IN, 0 },       { "g", RAD_OPD_WEIGHT, 0 },
    { "w", RAD_OPD_WTAB, 0 },     { "w_scale", RAD_OPD_WTAB, 0 },
    { "sorted", RAD_OPD_IN, 0 },
    { "w_1", RAD_OPD_WTAB, 0 },   { "w_1_scale", RAD_OPD_WTAB, 0 },
    { "w_2", RAD_OPD_WTAB, 0 },   { "w_2_scale", RAD_OPD_WTAB, 0 },
    { "w_3", RAD_OPD_WTAB, 0 },   { "w_3_scale", RAD_OPD_WTAB, 0 },
    { "y", RAD_OPD_OUT, 0 },
};

/* ------------------------------------------------------------------ schemas */
static const RadParamSpec kPGemm[] = {
    { "M", RAD_P_INT, RAD_REQUIRED },
    { "N", RAD_P_INT, RAD_REQUIRED },
    { "K", RAD_P_INT, RAD_REQUIRED },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadParamSpec kPNorm[] = {
    { "M", RAD_P_INT, RAD_REQUIRED },
    { "n", RAD_P_INT, RAD_REQUIRED },
    /* A float slot. Constraints never match on one; it is here so the schema check has something
     * to refuse an integer spelling of. */
    { "eps", RAD_P_F64, RAD_OPTIONAL },
    { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadParamSpec kPTie[] = {
    { "M", RAD_P_INT, RAD_REQUIRED },
    { "n", RAD_P_INT, RAD_REQUIRED },
};
static const RadParamSpec kPAttn[] = {
    { "q_len", RAD_P_INT, RAD_REQUIRED },
    { "head_dim", RAD_P_INT, RAD_REQUIRED },
    { "window", RAD_P_INT, RAD_REQUIRED },
    /* Supplied by the KERNEL, which is what makes t_attn_paged reachable at all: its
     * block_size == 16 would otherwise refuse every query that did not already know the answer
     * (rad_abi.h RAD_DERIVED, spec §7.2). */
    { "block_size", RAD_P_INT, RAD_DERIVED },
    { "causal", RAD_P_INT, RAD_REQUIRED },
    { "n_head", RAD_P_INT, RAD_REQUIRED },
    { "kv_dtype", RAD_P_STR, RAD_REQUIRED },
};

/* The conflicting spelling: same op name, no dtype. Since arguments are positional, a call built
 * against this schema and served by a kernel selected against the other one is not a diagnosable
 * failure -- it is silent numerical garbage, which is why the load refuses it (§2.3). */
static const RadParamSpec kPGemmWrong[] = {
    { "M", RAD_P_INT, RAD_REQUIRED },
    { "N", RAD_P_INT, RAD_REQUIRED },
    { "K", RAD_P_INT, RAD_REQUIRED },
};

/* ------------------------------------------------------------------ constraints */
/* Between them these rows use every operator the ABI has: EQ, LE, GE, DIV and IN. */
static const RadConstraint kCGemmM16[] = {
    RAD_CLE("M", 16), RAD_CDIV("K", 256), RAD_CIN("dtype", "bf16 w4a8"),
};
static const RadConstraint kCGemmM64[] = {
    RAD_CLE("M", 64), RAD_CDIV("K", 16), RAD_CIN("dtype", "bf16 w4a8"),
};
static const RadConstraint kCNorm4096[] = {
    RAD_CEQ("n", 4096),
};
static const RadConstraint kCTie[] = {
    RAD_CGE("M", 1),
};
/* A kernel that reaches another card's memory: a constraint on the machine, not the geometry. */
static const RadConstraint kCPeer[] = {
    RAD_CGE("M", 1),
    RAD_CEQ(RAD_FACT_P2P, 1),
};
/* Constraint ORDER is the order the miss message reports in: the first unmet one is what a reader
 * is told, so the cheapest and most identifying predicate goes first. */
static const RadConstraint kCAttn[] = {
    RAD_CEQ("head_dim", 128), RAD_CGE("window", 0), RAD_CEQ("block_size", 16),
    RAD_CIN("causal", "0 1"), RAD_CLE("q_len", 16), RAD_CDIV("n_head", 4),
    RAD_CIN("kv_dtype", "bf16 fp8_e4m3"),
};
/* The other half of the derived story: a row that admits every block size and names none. ref's
 * rows are exactly this -- every extent is read off the operands at launch -- so the winner has no
 * RAD_C_EQ to read back and the derived key stays UNSET. "any" is not a number. */
static const RadConstraint kCAttnAny[] = {
    RAD_CEQ("head_dim", 128), RAD_CGE("block_size", 1), RAD_CIN("kv_dtype", "bf16 fp8_e4m3"),
};
static const RadConstraint kCSkinnyM64[] = { RAD_CLE("M", 64) };
static const RadConstraint kCSkinnyBig[] = { RAD_CGE("M", 4096) };
static const RadConstraint kCEqM32[]     = { RAD_CEQ("M", 32) };

/* A TUNABLE AXIS, declared the way an out-of-tree library declares one: the legal values and a
 * default, not a list of pre-named combinations. `deflt` is what runs when this machine has no
 * measurement for an instantiation, and the loader refuses the plugin if it is not one of the
 * values (§15). */
static const int64_t kBnValues[] = { 64, 128 };
static const RadTunable kTGemmM64[] = {
    { "bn", kBnValues, NELEM(kBnValues), 64, "columns of C one block covers" },
};

/* ================================================================== radtest_base */
#if defined(RADTEST_BASE)

static const RadOpSchema kSchemas[] = {
    { "t_gemm",   kPGemm, NELEM(kPGemm), kOpdGemm, NELEM(kOpdGemm), "the §2.2 worked example" },
    { "t_norm",   kPNorm, NELEM(kPNorm), kOpdNorm, NELEM(kOpdNorm), "hierarchy vs priority" },
    { "t_tie",    kPTie,  NELEM(kPTie),  kOpdNorm, NELEM(kOpdNorm), "a declaration-order tie" },
    { "t_attn",   kPAttn, NELEM(kPAttn), kOpdAttn, NELEM(kOpdAttn), "every constraint operator" },
    { "t_attn_ref", kPAttn, NELEM(kPAttn), kOpdAttn, NELEM(kOpdAttn),
      "a derived key its kernel does not pin" },
    { "t_skinny", kPGemm, NELEM(kPGemm), kOpdGemm, NELEM(kOpdGemm), "a range with a hole in it" },
    { "t_eq",     kPTie,  NELEM(kPTie),  kOpdNorm, NELEM(kOpdNorm), "an EQ band boundary" },
    { "t_peer",   kPTie,  NELEM(kPTie),  kOpdNorm, NELEM(kOpdNorm),
      "a peer kernel with a fallback beneath it" },
    { "t_peer_only", kPTie, NELEM(kPTie), kOpdNorm, NELEM(kOpdNorm),
      "a peer kernel and nothing else" },
    { "t_moe", kPTie, NELEM(kPTie), kOpdMoe, NELEM(kOpdMoe),
      "one weight list over eight table operands" },
};

static const RadKernelInfo kKernels[] = {
    { .name = "t_gemm_m16", .op = "t_gemm", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "M <= 16, K divisible by 256",
      .dtypes = "bf16 / w4a8", .domain = RAD_DOMAIN_DEVICE, .priority = 30,
      .constraints = kCGemmM16, .n_constraints = NELEM(kCGemmM16),
      .init = t_init, .fini = t_fini, .launch = t_launch },

    { .name = "t_gemm_m64", .op = "t_gemm", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "M <= 64, K divisible by 16",
      .dtypes = "bf16 / w4a8", .domain = RAD_DOMAIN_DEVICE, .priority = 20,
      .constraints = kCGemmM64, .n_constraints = NELEM(kCGemmM64),
      .tunables = kTGemmM64, .n_tunables = NELEM(kTGemmM64),
      .init = t_init, .fini = t_fini, .launch = t_launch },

    /* Unconstrained, and lowest priority: the prefill row that serves whatever the two skinny rows
     * above it will not. It is what closes the last band of the bucket table. */
    { .name = "t_gemm_any", .op = "t_gemm", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10,
      .init = t_init, .fini = t_fini, .launch = t_launch },

    /* A CPU kernel is an ordinary kernel with domain = host, resolved through the same hierarchy
     * and the same constraints (§2.1). */
    { .name = "t_gemm_host", .op = "t_gemm", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_HOST, .priority = 10, .launch = t_launch },

    /* Very specialised and very high priority -- and still beaten by a plugin above it, which is
     * the point of the t_norm rows. */
    { .name = "t_norm_specialised", .op = "t_norm", .family = "norm",
      .computes = "nothing; a registry row", .shape = "n == 4096", .dtypes = "bf16",
      .domain = RAD_DOMAIN_DEVICE, .priority = 100,
      .constraints = kCNorm4096, .n_constraints = NELEM(kCNorm4096), .launch = t_launch },
    { .name = "t_norm_host", .op = "t_norm", .family = "norm",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_HOST, .priority = 10, .launch = t_launch },

    /* Peer access is the machine's: the direct row needs it, the staged one does not. */
    { .name = "t_peer_direct", .op = "t_peer", .family = "ar",
      .computes = "nothing; a registry row", .shape = "M >= 1, cards reach each other",
      .dtypes = "any", .domain = RAD_DOMAIN_DEVICE, .priority = 20,
      .constraints = kCPeer, .n_constraints = NELEM(kCPeer), .launch = t_launch },
    { .name = "t_peer_staged", .op = "t_peer", .family = "ar",
      .computes = "nothing; a registry row", .shape = "M >= 1", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10,
      .constraints = kCTie, .n_constraints = NELEM(kCTie), .launch = t_launch },
    { .name = "t_peer_only_direct", .op = "t_peer_only", .family = "ar",
      .computes = "nothing; a registry row", .shape = "M >= 1, cards reach each other",
      .dtypes = "any", .domain = RAD_DOMAIN_DEVICE, .priority = 10,
      .constraints = kCPeer, .n_constraints = NELEM(kCPeer), .launch = t_launch },

    /* Same op, same priority, same constraints. Ties break on declaration order, so t_tie_first
     * wins -- and if it ever does not, the tie rule has silently changed. */
    { .name = "t_tie_first", .op = "t_tie", .family = "norm",
      .computes = "nothing; a registry row", .shape = "M >= 1", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10,
      .constraints = kCTie, .n_constraints = NELEM(kCTie), .launch = t_launch },
    { .name = "t_tie_second", .op = "t_tie", .family = "norm",
      .computes = "nothing; a registry row", .shape = "M >= 1", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10,
      .constraints = kCTie, .n_constraints = NELEM(kCTie), .launch = t_launch },

    /* Device only: nothing serves t_attn on the host, which is the asymmetric-absence case. */
    { .name = "t_attn_paged", .op = "t_attn", .family = "attn",
      .computes = "nothing; a registry row",
      .shape = "head_dim 128, block 16, q_len <= 16, n_head divisible by 4",
      .dtypes = "bf16 / fp8_e4m3 kv", .domain = RAD_DOMAIN_DEVICE, .priority = 10,
      .constraints = kCAttn, .n_constraints = NELEM(kCAttn), .launch = t_launch },

    { .name = "t_attn_ref_any", .op = "t_attn_ref", .family = "attn",
      .computes = "nothing; a registry row", .shape = "head_dim 128, any block size",
      .dtypes = "bf16 / fp8_e4m3 kv", .domain = RAD_DOMAIN_DEVICE, .priority = 10,
      .constraints = kCAttnAny, .n_constraints = NELEM(kCAttnAny), .launch = t_launch },

    /* M <= 64 and M >= 4096, with nothing in between: a range with a hole in it, reported as that
     * band rather than as the whole op (§2.2). */
    { .name = "t_skinny_m64", .op = "t_skinny", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "M <= 64", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10,
      .constraints = kCSkinnyM64, .n_constraints = NELEM(kCSkinnyM64), .launch = t_launch },
    { .name = "t_skinny_big", .op = "t_skinny", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "M >= 4096", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 5,
      .constraints = kCSkinnyBig, .n_constraints = NELEM(kCSkinnyBig), .launch = t_launch },

    { .name = "t_eq_m32", .op = "t_eq", .family = "norm",
      .computes = "nothing; a registry row", .shape = "M == 32", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10,
      .constraints = kCEqM32, .n_constraints = NELEM(kCEqM32),
      .init = t_init_fails, .launch = t_launch },

    { .name = "t_moe_any", .op = "t_moe", .family = "moe",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10, .launch = t_launch },
};

/* The name is a -D so that two of these can load side by side, and one of them can declare itself
 * a reference implementation -- the only way to observe the loader's rule that a reference library
 * sorts below every other kernel plugin. */
#ifndef RADTEST_BASE_NAME
#define RADTEST_BASE_NAME "radtest_base"
#endif
static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, RADTEST_BASE_NAME, "0.1",
    "test rows: the §2.2 band table, a priority tie, every constraint operator", "host",
};
#ifdef RADTEST_REFERENCE
extern "C" int rad_plugin_reference(void) { return 1; }
#endif

/* ================================================================== radtest_extra */
/* Rows for rules the base plugin's ops do not reach: a constraint on the ranged key that is not an
 * interval, a tunable axis with a cross-axis test, a kernel that relayouts its weight, a kernel
 * that reads any stored form, and a capacity parameter. In testkernels/extra, so loading testkernels/ alone is unchanged. */
#elif defined(RADTEST_EXTRA)

static const RadOperandSpec kOpdCap[] = {
    { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadParamSpec kPCap[] = {
    { "M", RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
    { "cap", RAD_P_INT, RAD_REQUIRED, RAD_PROLE_CAPACITY },
    { "n", RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
};

static const RadOpSchema kSchemas[] = {
    { "t_div",  kPTie,  NELEM(kPTie),  kOpdNorm, NELEM(kOpdNorm),
      "IN and DIV on the ranged key" },
    { "t_tune", kPGemm, NELEM(kPGemm), kOpdGemm, NELEM(kOpdGemm),
      "a tunable axis with a cross-axis test" },
    { "t_lay",  kPGemm, NELEM(kPGemm), kOpdGemm, NELEM(kOpdGemm),
      "a kernel that stores its weight in its own layout" },
    { "t_join", kPGemm, NELEM(kPGemm), kOpdGemm, NELEM(kOpdGemm),
      "a weight of several planes: a device kernel's own form, a host kernel's joined one" },
    { "t_any",  kPGemm, NELEM(kPGemm), kOpdGemm, NELEM(kOpdGemm),
      "a kernel that reads its weight's stored bytes uninterpreted (a prefetch)" },
    { "t_cap",  kPCap,  NELEM(kPCap),  kOpdCap,  NELEM(kOpdCap),
      "a capacity a sizing declare may shrink" },
};

/* M in {8, 24}, and M divisible by 16 up to 64. */
static const RadConstraint kCDivIn[]  = { RAD_CIN("M", "8 24") };
static const RadConstraint kCDivM16[] = { RAD_CDIV("M", 16), RAD_CLE("M", 64) };

/* bn = 128 only where the output has at least 128 columns. */
static int t_tune_valid(const RadParam* p, int n_p, const RadParam* choice, int n_choice) {
    long long n = 0, bn = 0;
    for (int i = 0; i < n_p; ++i)
        if (p[i].key && std::strcmp(p[i].key, "N") == 0) n = p[i].ival;
    for (int i = 0; i < n_choice; ++i)
        if (choice[i].key && std::strcmp(choice[i].key, "bn") == 0) bn = choice[i].ival;
    return bn <= n ? 1 : 0;
}

/* Stores a bf16 weight, or fp8 codes with their scales, in a "tiled" form of its own, and reads no
 * other encoding (plain fp8 among them). */
static int t_lay_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                        const int* sel, const RadTensor* planes, int n_planes, RadLayout* out) {
    (void)p; (void)n_p; (void)operand;
    if (!enc || !sel || !planes || n_planes < 1 || !out) return RAD_E_INVAL;
    const uint32_t codes = enc->plane[sel[0]].dtype;
    const bool bf16 = n_planes == 1 && codes == RAD_BF16;
    const bool fp8 = n_planes == 2 && codes == RAD_F8E4M3;
    if (!bf16 && !fp8) return RAD_E_DTYPE;
    out->tag = bf16 ? "tiled" : "tiled-fp8";
    out->dtype = codes;
    out->rank = 2;
    out->shape[0] = planes[0].shape[0];
    out->shape[1] = planes[0].shape[1];
    out->align = 4096;
    int64_t bytes = 0;
    for (int i = 0; i < n_planes; ++i)
        bytes += rad_dtype_bytes(planes[i].dtype, rad_tensor_numel(&planes[i]));
    out->bytes = bytes > 0 ? bytes : 1;
    return RAD_OK;
}

/* The joined form every host library shares (abi/rad_planes.h). */
static int t_join_host_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                              const int* sel, const RadTensor* planes, int n_planes,
                              RadLayout* out) {
    (void)p; (void)n_p; (void)operand;
    return rad_planes_layout(enc, sel, planes, n_planes, out);
}

/* Reads the stored bytes without interpreting them: any form, and claims none. */
static int t_any_layout(const RadParam*, int, int, const RadEncoding*, const int*,
                        const RadTensor*, int, RadLayout*) {
    return RAD_LAYOUT_ANY;
}

static const RadConstraint kCLayTiled[] = { RAD_CLE("M", 16) };
static const RadConstraint kCLayPlain[] = { RAD_CLE("M", 4096) };

static const RadKernelInfo kKernels[] = {
    { .name = "t_div_in", .op = "t_div", .family = "norm",
      .computes = "nothing; a registry row", .shape = "M in {8, 24}", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 40,
      .constraints = kCDivIn, .n_constraints = NELEM(kCDivIn), .launch = t_launch },
    { .name = "t_div_m16", .op = "t_div", .family = "norm",
      .computes = "nothing; a registry row", .shape = "M <= 64, M divisible by 16",
      .dtypes = "any", .domain = RAD_DOMAIN_DEVICE, .priority = 30,
      .constraints = kCDivM16, .n_constraints = NELEM(kCDivM16), .launch = t_launch },
    { .name = "t_div_any", .op = "t_div", .family = "norm",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10, .launch = t_launch },

    { .name = "t_tune_picky", .op = "t_tune", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10,
      .tunables = kTGemmM64, .n_tunables = NELEM(kTGemmM64), .tune_valid = t_tune_valid,
      .launch = t_launch },

    { .name = "t_lay_tiled", .op = "t_lay", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "M <= 16", .dtypes = "bf16 / fp8 + scale",
      .domain = RAD_DOMAIN_DEVICE, .priority = 20,
      .constraints = kCLayTiled, .n_constraints = NELEM(kCLayTiled),
      .launch = t_launch, .layout = t_lay_layout },
    { .name = "t_lay_plain", .op = "t_lay", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "M <= 4096", .dtypes = "as stored",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10,
      .constraints = kCLayPlain, .n_constraints = NELEM(kCLayPlain), .launch = t_launch },
    { .name = "t_lay_host", .op = "t_lay", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "as stored",
      .domain = RAD_DOMAIN_HOST, .priority = 10, .launch = t_launch },

    { .name = "t_join_tiled", .op = "t_join", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "M <= 16", .dtypes = "fp8 + scale",
      .domain = RAD_DOMAIN_DEVICE, .priority = 20,
      .constraints = kCLayTiled, .n_constraints = NELEM(kCLayTiled),
      .launch = t_launch, .layout = t_lay_layout },
    { .name = "t_join_host", .op = "t_join", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "planes joined",
      .domain = RAD_DOMAIN_HOST, .priority = 10, .launch = t_launch,
      .layout = t_join_host_layout },

    { .name = "t_any_read", .op = "t_any", .family = "misc",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "any stored form",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10, .launch = t_launch,
      .layout = t_any_layout },

    { .name = "t_cap_any", .op = "t_cap", .family = "norm",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10, .launch = t_launch },
};

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "radtest_extra", "0.1",
    "non-interval ranged constraints, a cross-axis tune test, layout hooks, a capacity", "host",
};

/* ================================================================== radtest_badtune_* */
/* A tunable axis the loader must refuse: a default that is not one of its values
 * (RADTEST_BADTUNE=1), or one axis declared twice (2). */
#elif defined(RADTEST_BADTUNE)

static const RadOpSchema kSchemas[] = {
    { "t_gemm", kPGemm, NELEM(kPGemm), kOpdGemm, NELEM(kOpdGemm), "never registered" },
};
#if RADTEST_BADTUNE == 1
static const RadTunable kTBad[] = {
    { "bn", kBnValues, NELEM(kBnValues), 96, "a default the axis does not offer" },
};
#else
static const RadTunable kTBad[] = {
    { "bn", kBnValues, NELEM(kBnValues), 64, "the axis" },
    { "bn", kBnValues, NELEM(kBnValues), 128, "the same axis again" },
};
#endif
static const RadKernelInfo kKernels[] = {
    { .name = "t_badtune_gemm", .op = "t_gemm", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 10,
      .tunables = kTBad, .n_tunables = NELEM(kTBad), .launch = t_launch },
};
static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "radtest_badtune", "0.1", "a tunable axis the loader refuses", "host",
};

/* ================================================================== radtest_decline */
#elif defined(RADTEST_DECLINE)

static const RadOpSchema kSchemas[] = {
    { "t_gemm", kPGemm, NELEM(kPGemm), kOpdGemm, NELEM(kOpdGemm), "never registered" },
};
static const RadKernelInfo kKernels[] = {
    { .name = "t_decline_gemm", .op = "t_gemm", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 1000, .launch = t_launch },
};
static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "radtest_decline", "0.1", "declines this machine at open", "host",
};

/* ================================================================== radtest_over */
#elif defined(RADTEST_OVER)

/* The same schemas, declared again with the same shape. That is expected and not an error: every
 * plugin declares the ops it implements, and an override that did not would be relying on a plugin
 * below it that a user may well have removed. */
static const RadOpSchema kSchemas[] = {
    { "t_gemm", kPGemm, NELEM(kPGemm), kOpdGemm, NELEM(kOpdGemm), "re-declared, same shape" },
    { "t_norm", kPNorm, NELEM(kPNorm), kOpdNorm, NELEM(kOpdNorm), "re-declared, same shape" },
};

static const RadKernelInfo kKernels[] = {
    /* Unconstrained and priority 1 -- as unspecialised as a row can be. Placed above
     * radtest_base in the hierarchy it still wins t_norm outright, because the hierarchy is
     * absolute and priority only orders rows WITHIN one plugin. */
    { .name = "t_over_norm", .op = "t_norm", .family = "norm",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 1, .launch = t_launch },
};

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "radtest_over", "0.1",
    "one generic row that must still beat a specialised one below it", "host",
};

/* ================================================================== radtest_conflict */
#elif defined(RADTEST_CONFLICT)

static const RadOpSchema kSchemas[] = {
    { "t_gemm", kPGemmWrong, NELEM(kPGemmWrong), kOpdGemm, NELEM(kOpdGemm),
      "the same op name with a different parameter list" },
};

static const RadKernelInfo kKernels[] = {
    { .name = "t_conflict_gemm", .op = "t_gemm", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 1000, .launch = t_launch },
};

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "radtest_conflict", "0.1",
    "declares t_gemm with a different schema; must be refused at load", "host",
};

/* ================================================================== radtest_arch, radtest_arch2 */
/* Two architecture plugins claiming ONE architecture id. An architecture id is what selects the
 * plugin a model is built from, so two claimants is a coin toss over which graph gets built --
 * refused at load, and the second one loses (§2.4). They export no kernel symbols at all, which is
 * how the loader learns that an arch plugin is not a kernel plugin with missing exports. */
#elif defined(RADTEST_IS_ARCH)

/* RADTEST_ARCHQ plugins are one family at several quantisations, each named, with its
 * quantisation, by -D. */
#if defined(RADTEST_ARCHQ)
#define RADTEST_ARCH_NAME RADTEST_ARCHQ_NAME
#define RADTEST_ARCH_ID "radtest.q"
#define RADTEST_ARCH_QUANT RADTEST_ARCHQ_QUANT
#elif defined(RADTEST_ARCH)
#define RADTEST_ARCH_NAME "radtest_arch"
#else
#define RADTEST_ARCH_NAME "radtest_arch2"
#endif
#ifndef RADTEST_ARCH_ID
#define RADTEST_ARCH_ID "radtest.v1"
#endif
#ifndef RADTEST_ARCH_QUANT
#define RADTEST_ARCH_QUANT ""
#endif

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_ARCH, RADTEST_ARCH_NAME, "0.1", "claims an architecture id", "host",
};

/* ================================================================== radtest_badabi */
#else

static const RadOpSchema kSchemas[] = {
    { "t_gemm", kPGemm, NELEM(kPGemm), kOpdGemm, NELEM(kOpdGemm), "never registered" },
};

static const RadKernelInfo kKernels[] = {
    { .name = "t_badabi_gemm", .op = "t_gemm", .family = "gemm",
      .computes = "nothing; a registry row", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_DEVICE, .priority = 1000, .launch = t_launch },
};

#if defined(RADTEST_BADLAYOUT)
static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "radtest_badlayout", "0.1",
    "reports this engine's ABI version over another layout; must be refused at load", "host",
};
#else
static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "radtest_badabi", "0.1",
    "reports an ABI version this engine does not speak; must be refused at load", "host",
};
#endif

#endif

/* ------------------------------------------------------------------ the exports */
#if defined(RADTEST_BADABI)
/* Deliberately wrong. A plugin that does not report exactly RAD_ABI_VERSION is refused at load, by
 * name, with both versions printed. */
uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION + 7u; }
#else
uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
#endif
#if defined(RADTEST_BADLAYOUT)
/* Deliberately wrong: the right version over another layout, as a plugin built against another
 * tree's headers at the same number reports it. Refused at load with both fingerprints. */
uint64_t rad_plugin_abi_layout(void) { return rad_abi_layout() ^ 1u; }
#else
RAD_PLUGIN_ABI_LAYOUT_EXPORT
#endif

const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }

#if defined(RADTEST_IS_ARCH)
const char* rad_arch_id(void) { return RADTEST_ARCH_ID; }
/* The other half of the selection key (spec §2.4); "" is an unquantised container. */
const char* rad_arch_quant(void) { return RADTEST_ARCH_QUANT; }
#if defined(RADTEST_ARCHQ_NO_IMPORTS)
/* Quantised, but never the plugin an imported checkpoint is handed to. */
int rad_arch_serves_imports(void) { return 0; }
#endif
/* The core loads and identifies architecture plugins; the build component drives them. Neither
 * hook is called by anything this test exercises, and declare refuses rather than pretending. */
int  rad_arch_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    (void)b; (void)meta; (void)ctx;
    return RAD_E_UNSUPPORTED;
}
void rad_arch_step(RadCtx* c, const RadBatch* batch) { (void)c; (void)batch; }

#if defined(RADTEST_ARCH)
/* The optional reply-format export, declared the way an out-of-tree plugin would declare its
 * model's syntax: static storage, struct_size as compiled. radtest_arch2 leaves it out. */
static const char* const kTestBreaks[] = { "<call>" };
static const char* const kTestTriggers[] = { "<call>" };
static const RadChatFormat kTestFormat = {
    sizeof(RadChatFormat), "radtest-format",
    "<reason>", "</reason>", kTestBreaks, 1, RAD_CHAT_THINK_FROM_PROMPT,
    "", "", "",
    "<call>", "</call>", "<fn ", ">", "</fn>",
    "<arg ", "", "=", "</arg>",
    RAD_CHAT_VALUE_CDATA, RAD_CHAT_VALUE_JSON, 0,
    kTestTriggers, 1,
};
const RadChatFormat* rad_arch_chat_format(const RadModelMeta* meta) {
    (void)meta;
    return &kTestFormat;
}
#endif
#else
int                  rad_kernel_schema_count(void) { return NELEM(kSchemas); }
const RadOpSchema*   rad_kernel_schema_at(int i) {
    return (i < 0 || i >= NELEM(kSchemas)) ? nullptr : &kSchemas[i];
}
int                  rad_kernel_count(void) { return NELEM(kKernels); }
const RadKernelInfo* rad_kernel_at(int i) {
    return (i < 0 || i >= NELEM(kKernels)) ? nullptr : &kKernels[i];
}
#endif

}  /* extern "C" */
