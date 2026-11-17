/* kbarch.cc -- an architecture small enough to declare from a synthetic container, for
 * tests/kbench_mutant_test.cpp: rad-kbench records a fixture from it on the host backend, from the
 * container and again from the container's skeleton, with libref the only kernel library.
 *
 * It declares a norm with a gain and a projection, both read from the container, and an add. It
 * never runs a step. Loaded only by path, from a test's own home.
 */
#include "rad_abi.h"
#include "rad_builder.h"
#include "rad_abi_layout.h"

#include <cstring>

namespace {

RadParam p_int(const char* k, long long v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}
RadParam p_range(const char* k, long long lo, long long hi) {
    RadParam p = p_int(k, lo);
    p.kind = RAD_P_RANGE;
    p.ihi = hi;
    return p;
}
RadParam p_str(const char* k, const char* v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_STR;
    p.sval = v;
    return p;
}
RadParam p_f64(const char* k, double v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_F64;
    p.dval = v;
    return p;
}

}  /* namespace */

extern "C" {

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_ARCH, "radtest_kbarch", "0.1",
    "declares a norm, a projection and an add from a synthetic container, for rad-kbench's tests",
    "host",
};

uint32_t             rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
const char*          rad_arch_id(void) { return "radtest.kbench"; }
const char*          rad_arch_quant(void) { return ""; }

int rad_arch_declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx) {
    const int64_t n = m->n_embd, N = m->n_ff;
    if (n <= 0 || N <= 0 || ctx->max_tok <= 0) return RAD_E_INVAL;
    RadWeightDecl g{};
    g.dtype = RAD_BF16;
    g.rank = 1;
    g.shape[0] = n;
    g.access = RAD_ACCESS_PER_TOKEN;
    g.shard = RAD_SHARD_NONE;
    g.group.layer = -1;
    g.group.expert = -1;
    const rad_weight wn = rad_decl_weight(b, "norm.weight", &g);
    RadWeightDecl w = g;
    w.rank = 2;
    w.shape[0] = N;
    w.shape[1] = n;
    const rad_weight wp = rad_decl_weight(b, "proj.weight", &w);
    if (!wn || !wp) return RAD_E_INVAL;

    const RadParam pn[] = { p_range("M", 1, ctx->max_tok), p_int("n", n), p_f64("eps", 1e-6),
                            p_str("dtype", "bf16") };
    const RadParam pg[] = { p_range("M", 1, ctx->max_tok), p_int("N", N), p_int("K", n),
                            p_str("dtype", "bf16") };
    const RadParam pa[] = { p_range("M", 1, ctx->max_tok), p_int("n", N), p_str("dtype", "bf16") };
    if (!rad_decl_op(b, "rmsnorm", pn, 4, &wn, 1)) return RAD_E_INVAL;
    if (!rad_decl_op(b, "gemm_nt", pg, 4, &wp, 1)) return RAD_E_INVAL;
    if (!rad_decl_op(b, "add", pa, 3, nullptr, 0)) return RAD_E_INVAL;
    return RAD_OK;
}

void rad_arch_step(struct RadCtx* c, const struct RadBatch* batch) {
    (void)c;
    (void)batch;
}

}  /* extern "C" */
