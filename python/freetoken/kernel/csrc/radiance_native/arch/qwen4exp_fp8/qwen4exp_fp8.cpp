// Reference architecture composition; mathematical blocks are shared with FreeToken v2.
#include <arch/freetoken_qwen_blocks.h>
namespace qwen4exp_fp8 {
static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    return freetoken_qwen_blocks::lower_blocks(b, meta, ctx);
}
static void step(RadCtx* c, const RadBatch* batch) { freetoken_qwen_blocks::step(c, batch); }
static int probe(const RadModelMeta* m, RadArchProbe* out) { return freetoken_qwen_blocks::probe(m, out); }
static uint64_t pass_key(RadCtx* c, const RadBatch* batch, const int32_t* used) {
    return freetoken_qwen_blocks::pass_key(c, batch, used);
}
}

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(qwen4exp_fp8)
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(qwen4exp_fp8, "qwen4exp", "", "0.1.0",
                "Qwen4-Exp (Qwen3.8-Flash-Next): hybrid Gated DeltaNet / gated attention, a gated "
                "residual four streams wide and 512 routed experts, from a dense bf16 checkpoint "
                "quantised ahead of time by a recipe")
extern "C" uint64_t rad_arch_pass_key(RadCtx* c, const RadBatch* batch,
                                      const int32_t* used_max_h) {
    return qwen4exp_fp8::pass_key(c, batch, used_max_h);
}
#endif
