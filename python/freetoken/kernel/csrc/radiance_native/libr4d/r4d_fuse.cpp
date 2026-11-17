/* r4d_fuse.cpp -- the chain of unfused ops each fused kernel computes in one launch.
 *
 * `RadKernelInfo::replaces` (rad_abi.h) makes a fused row's claim to be byte-identical to the ops
 * it replaces executable. Most of libr4d's fusions have no libref kernel, so this is their only
 * oracle: a caller runs the fused row against its chain built from this plugin's own unfused rows,
 * and expects identical bytes, not a tolerance.
 *
 * The authority for each chain is the device kernel, not the row's prose, and each chain names the
 * source that decides it. Exactness often comes down to one `f2bf`: the fp8 folds round their
 * intermediate to bf16 and read it back so the fused and unfused forms agree; the int8 folds do
 * not, because their reference is a different pair. A kernel that is merely close to a sequence
 * publishes no hook (rad_abi.h): a false failure is worse than a skip.
 *
 * ============================================================ operand presence
 *
 * `rmsnorm_quant_fp8` computes `add` + `rmsnorm` + quantiser when given a residual, and `rmsnorm`
 * + quantiser when not, so the chain length depends on an operand. That is why the chain is a hook
 * rather than a table, and why the hook receives `present`: one bit per fused operand, read with
 * RAD_FUSE_HAS(). Two alternatives are wrong:
 *
 *   - Deriving it from a parameter: there is none. `wadd` is GemmaRMSNorm's `1 +`, a property of
 *     the gain, and arch/common/rad_fp8.h passes it in both forms.
 *   - Marking the operands RAD_FUSE_NONE: that marks an optional operand a step does not pass.
 *     Here the operands that vary are required ones (`add`'s `b`, `rmsnorm`'s `x`) and the chain
 *     length itself changes.
 *
 * A caller that does not know passes RAD_FUSE_PRESENT_ALL, meaning every operand this plugin's own
 * description lists; r4d_shape_rmsnorm_quant_fp8 publishes all six as present. Where a fold's step
 * carries the same optional operands as the fold (ar_rmsnorm_quant_fp8 below), absence propagates
 * through the wiring and `present` need not be read. Prefer that shape where there is a choice.
 *
 * ============================================================ rows with no hook
 *
 * These rows carry a null `replaces`; the caller counts them as uncovered.
 *
 *   rmsnorm_had_quant_i8, gated_had_quant_i8, gdn_gated_norm_had_quant_i8
 *       `rmsnorm` + `had_quant_act_i8` is expressible but does not hold.
 *       r4d_rmsnorm_had_quant_i8.hip runs the FWHT on `x * (w + wadd)` before the norm closes
 *       (valid: the transform is linear and the int8 codes cancel the per-row `inv_rms`) and folds
 *       `inv_rms` into `S[row]`. Its codes come from values never normalised or rounded to bf16,
 *       while the unfused pair rotates a stored bf16 norm output. r4d_gated_had_quant_i8.hip
 *       likewise computes silu(x) in fp32, which `silu_mul` + `had_quant_act_i8` does not.
 *
 *   had_quant_act_i8, quant_act_i8, quant_act_fp8
 *       Primitive ops, not fusions. A one-step chain naming the op itself would check nothing.
 *
 *   ar_ln_had_quant_i8
 *       `all_reduce` + `rmsnorm_had_quant_i8` is expressible but the arithmetic differs: the kernel
 *       computes `t = mine + peer + residual` in f32 with no bf16 round after the reduce, while the
 *       standalone all-reduce stores bf16 (the compressed wire rounds to bf16, to match the
 *       compressed all-reduce). The fp8 sibling ar_rmsnorm_quant_fp8 rounds there and has a hook.
 *
 *   qk_norm_rope_gate
 *       Not byte-identical to the four kernels it replaces (docs/OPS.md): it folds a sum of
 *       squares over one wave per head where `rmsnorm` folds over kR4dSumLanes' 256 lanes, so
 *       the two differ by a ULP.
 *
 *   sample_chain
 *       Stands beside the decomposed sampling stages rather than replacing them: one workgroup
 *       over an L2-resident row does not round like separate launches, and the fused row also
 *       writes `tok_excl` and `p_query`, which no chain of them produces.
 *
 *   dflash_conv, dflash_select, the five gdn_* rows
 *       No unfused equivalent in the vocabulary. The GDN preamble's parts (a causal conv against a
 *       state cache, an L2 norm, a per-chunk gate cumsum) have no op names.
 *
 * No tuned axis is read here: this plugin's axes (tile shapes, split counts, wave counts,
 * all-reduce launch geometry) do not change what a kernel computes. An axis that did would be read
 * from `p`, where the core appends the chosen point.
 *
 * Pure host code (no r4d.h, no HIP), like r4d_rows.cpp, r4d_layout.cpp and r4d_shapes.cpp, so the
 * tables compile and can be read without ROCm.
 */

#include "r4d_plugin.h"

#include <cstring>
#include <initializer_list>

namespace {

/* ==================================================================== reading the fused geometry
 *
 * Not r4d_pgeti_or(): that reads RAD_P_INT only, which suits a launch (every range has collapsed
 * to the band's value), but this hook, like `opd_shape`, can see ranges still open. A RAD_P_RANGE
 * reads as its high bound, as in the operand descriptions the caller's buffers were sized from. */
int64_t pi(const RadParam* p, int n_p, const char* k, int64_t dflt = 0) {
    if (!p || !k) return dflt;
    for (int i = 0; i < n_p; ++i) {
        if (!p[i].key || std::strcmp(p[i].key, k) != 0) continue;
        if (p[i].kind == RAD_P_INT)   return (int64_t)p[i].ival;
        if (p[i].kind == RAD_P_RANGE) return (int64_t)p[i].ihi;
        return dflt;
    }
    return dflt;
}

/* Fused operand i when `present` has it, else RAD_FUSE_NONE: the wiring of an optional operand. */
int16_t opt_opd(uint64_t present, int i) {
    return RAD_FUSE_HAS(present, i) ? (int16_t)i : (int16_t)RAD_FUSE_NONE;
}

/* ==================================================================== building one step
 *
 * A step names its op, then its parameters in the step op's spelling, then its operand wiring in
 * the step op's schema order, so the operand count can be checked by eye against r4d_rows.cpp
 * (the ABI checks it too).
 *
 * `ok` records a required parameter the fused geometry lacks, and wire() refuses the step. A step
 * resolved at the wrong geometry would produce a difference that reads as a kernel fault.
 */
struct Step {
    RadFuseStep* s;
    bool         ok = true;

    Step(RadFuseStep* out, const char* op) : s(out) {
        *s = RadFuseStep{};
        s->op = op;
    }

    /* Copies a fused-op parameter verbatim, kind included. `M` is ranged on every op here, and
     * collapsing it would select a different band, which for a banded family like the fp8 GEMMs
     * is a different kernel and summation order. `eps` and `wadd` are RAD_P_F64. */
    Step& mirror(const RadParam* p, int n_p, const char* key) {
        const RadParam* f = rad_param_find(p, n_p, key);
        if (!f || s->n_p >= RAD_MAX_FUSE_PARAMS) { ok = false; return *this; }
        s->p[s->n_p++] = *f;
        return *this;
    }

    /* As mirror(), for a parameter optional in both ops: absent stays absent. A constraint whose
     * key the geometry lacks does not hold, so inventing `wadd = 0` could route the step to a
     * different row. */
    Step& mirror_opt(const RadParam* p, int n_p, const char* key) {
        const RadParam* f = rad_param_find(p, n_p, key);
        if (!f) return *this;
        return mirror(p, n_p, key);
    }

    /* Built field by field: rad_types.h's RAD_INT / RAD_STR are C compound literals, only a GNU
     * extension in C++. */
    Step& pint(const char* key, long long v) {
        if (s->n_p >= RAD_MAX_FUSE_PARAMS) { ok = false; return *this; }
        RadParam q{};
        q.key  = key;
        q.kind = RAD_P_INT;
        q.ival = v;
        s->p[s->n_p++] = q;
        return *this;
    }

    Step& pf64(const char* key, double v) {
        if (s->n_p >= RAD_MAX_FUSE_PARAMS) { ok = false; return *this; }
        RadParam q{};
        q.key  = key;
        q.kind = RAD_P_F64;
        q.dval = v;
        s->p[s->n_p++] = q;
        return *this;
    }

    Step& pstr(const char* key, const char* v) {
        if (!v || s->n_p >= RAD_MAX_FUSE_PARAMS) { ok = false; return *this; }
        RadParam q{};
        q.key  = key;
        q.kind = RAD_P_STR;
        q.sval = v;
        s->p[s->n_p++] = q;
        return *this;
    }

    /* One entry per operand of the step's op, in schema order: a fused-kernel operand index,
     * RAD_FUSE_TMP(t) for a temporary between steps, or RAD_FUSE_NONE for an optional operand the
     * step does not pass. */
    int wire(std::initializer_list<int> from) {
        if (!ok) return RAD_E_UNSUPPORTED;
        if (from.size() > RAD_MAX_FUSE_OPD) return RAD_E_INVAL;
        for (int v : from) s->from[s->n_from++] = (int16_t)v;
        return RAD_OK;
    }
};

/* Whether a gated fold's `act` (vocabulary spelling) is `want`. `act` is optional and the shims
 * default it differently: r4d_fused_quant_fp8.hip uses `mode` 0 (silu) for `gated_quant_fp8`, which feeds
 * down_proj, and `mode` 1 (sigmoid) for `gate_quant_fp8`, which feeds a gated attention's o_proj.
 * Each hook passes its row's default. */
bool act_is(const RadParam* p, int n_p, const char* dflt, const char* want) {
    const char* a = r4d_pgets(p, n_p, "act");
    return std::strcmp(a ? a : dflt, want) == 0;
}

/* The fp8 folds quantise only on the checkpoint's scale grid: the shims and the operand
 * descriptions test `group` against r4d_quant_act_fp8_group(). Any other group has no fused kernel
 * and so no chain. */
bool fp8_geometry_ok(const RadParam* p, int n_p, int64_t M, int64_t n) {
    return M > 0 && n > 0 && pi(p, n_p, "group", R4D_FP8_BLOCK) == R4D_FP8_BLOCK;
}

}  /* namespace */

/* ==========================================================================================
 * rmsnorm_quant_fp8 -- the RMS norm and the fp8 activation quantiser in one pass.
 * (kernel: libr4d/r4d_fused_quant_fp8.hip, rmsnorm_quant_fp8_kernel and its shim)
 *
 * Operands: [0] x  [1] residual?  [2] w  [3] q  [4] scale  [5] out_bf16?
 *
 * The chain is `add` + `rmsnorm` + `quant_act_fp8`, not `rmsnorm_add` + `quant_act_fp8`. The two
 * differ in which value the sum of squares is taken over:
 *
 *   this kernel   `h = f2bf(bf2f(rr[i]) + bf2f(xr[i])); rr[i] = h; v = bf2f(h); ss += v * v;`
 *                 The sum is rounded to bf16, stored, and read back before the variance: the
 *                 `add` + `rmsnorm` pair bit for bit.
 *
 *   rmsnorm_add   `float v = ldt(x, ..) + ldt(res, ..); ss += v * v; stt(ro, .., v);`
 *                 (libref/ref_elementwise.cpp, matching vLLM's fused_add_rms_norm). The variance
 *                 uses the unrounded f32 sum, half a ULP away on every channel.
 *
 * All three steps are libr4d rows (add_bf16, rmsnorm_bf16, quant_act_fp8), so the check stays
 * within libr4d; `rmsnorm_add` has no libr4d row.
 *
 * Both kernels fold the sum of squares in kR4dSumLanes' partition at whatever width
 * `r4d_act_threads(M, n)` gives them (r4d_common.h), so the width does not decide the bytes; `M`
 * is mirrored verbatim so both launch the same shape.
 *
 * Step 0 writes the sum into the fused residual operand, both its own `b` and the fused kernel's
 * in-place residual, so both sides leave the same buffers behind. `add_bf16` is safe in place:
 * one element per thread, no neighbours.
 *
 * The norm writes fused operand 5 when present. The kernel stores exactly the bf16 it quantises
 * (`h = f2bf(bf2f(nr[c]) * rs * (g + wadd)); if (yr) yr[c] = h; v[j] = bf2f(h);`), so the codes
 * are taken over that buffer.
 */
extern "C" int r4d_fuse_rmsnorm_quant_fp8(const RadParam* p, int n_p, uint64_t present, int step,
                                          RadFuseStep* out) {

    if (!out) return RAD_E_INVAL;
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    if (!fp8_geometry_ok(p, n_p, M, n)) return RAD_E_UNSUPPORTED;

    /* `residual` decides whether there is an `add` step; `out_bf16` decides whether the norm
     * output is a caller-visible buffer or a temporary. */
    const bool has_res = RAD_FUSE_HAS(present, 1) != 0;
    const bool has_out = RAD_FUSE_HAS(present, 5) != 0;
    const int16_t mid = has_out ? (int16_t)5 : RAD_FUSE_TMP(0);
    /* Without a residual there is no `add`, so the steps shift down by one. */
    const int s = has_res ? step : step + 1;

    switch (s) {
    case 0: {
        /* residual = x + residual, rounded to bf16; the norm reads this store back. */
        Step st(out, "add");
        st.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "dtype");
        return st.wire({ 0, 1, 1 });                /* a, b, y */
    }
    case 1: {
        /* The norm of the stored sum, or of `x` without a residual. `wadd` only if supplied. */
        Step st(out, "rmsnorm");
        st.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "eps")
          .mirror(p, n_p, "dtype").mirror_opt(p, n_p, "wadd");
        return st.wire({ has_res ? (int16_t)1 : (int16_t)0, 2, mid });   /* x, w, y */
    }
    case 2: {
        /* The codes, taken over the bf16 the norm wrote. */
        Step st(out, "quant_act_fp8");
        st.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "group").mirror(p, n_p, "dtype");
        return st.wire({ mid, 3, 4 });              /* x, q, scale */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * gated_quant_fp8 -- an elementwise gated product and the fp8 quantiser in one pass.
 * (kernel: libr4d/r4d_fused_quant_fp8.hip, gated_quant_fp8_kernel)
 *
 * Operands: [0] gate_up  [1] q  [2] scale  [3] out_bf16?
 *
 * Only `act` silu has a chain. `gate_up` is one [M, 2n] buffer (gate half, then up half), and the
 * wiring cannot name half an operand. `silu_mul` splits the packed plane itself; there is no
 * `sigmoid_mul`.
 *
 * The kernel also keeps the activation in f32 through the product and rounds once:
 *
 *     const float a = ACT == 0 ? g / (1.0f + expf(-g)) : 1.0f / (1.0f + expf(-g));
 *     const uint16_t h = f2bf(a * bf2f(gr[n + c]));
 *
 * That matches `silu_mul_bf16` in r4d_model_bf16.hip element for element, and differs from
 * `sigmoid` + `mul`, which round the activation in between. A future `sigmoid_mul` op would have to
 * round once too. The sigmoid gate is reached through `gate_quant_fp8` instead (cGemmNtQGatedFp8
 * and the arch use it), whose gate is a separate operand.
 */
extern "C" int r4d_fuse_gated_quant_fp8(const RadParam* p, int n_p, uint64_t present, int step,
                                        RadFuseStep* out) {

    if (!out) return RAD_E_INVAL;
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    if (!fp8_geometry_ok(p, n_p, M, n)) return RAD_E_UNSUPPORTED;
    /* The shim (r4d_fused_quant_fp8.hip) defaults this row's `mode` to 0 (silu). */
    if (!act_is(p, n_p, "silu", "silu")) return RAD_E_UNSUPPORTED;

    switch (step) {
    case 0: {
        /* silu(gate) * up over the packed [M, 2n] plane, into the bf16 the codes are taken over. */
        Step s(out, "silu_mul");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "dtype");
        return s.wire({ 0, 3 });                    /* gate_up, y */
    }
    case 1: {
        Step s(out, "quant_act_fp8");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "group").mirror(p, n_p, "dtype");
        return s.wire({ 3, 1, 2 });                 /* x, q, scale */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * router_sigmoid_topk_scatter -- GLM's selection, the sort over its ids and the shared gate's fill.
 * (kernel: libr4d/r4d_glm.hip, router_sigmoid_scatter_kernel)
 *
 * Operands: [0] logits  [1] bias  [2] ids  [3] weights  [4] sorted_tok  [5] expert_offset
 *           [6] expert_count  [7] gate?  [8] inverse?
 *
 * The sort runs over this rank's slice, so `moe_scatter`'s `n_expert` is the fused op's `n_local`;
 * with the inverse it is `moe_scatter_inv`. The fill steps in only with the gate.
 */
extern "C" int r4d_fuse_router_sigmoid_topk_scatter(const RadParam* p, int n_p, uint64_t present,
                                                    int step, RadFuseStep* out) {

    if (!out) return RAD_E_INVAL;
    const int64_t ne = pi(p, n_p, "n_expert"), nl = pi(p, n_p, "n_local", ne);
    if (ne <= 0 || nl <= 0 || nl > ne) return RAD_E_UNSUPPORTED;

    switch (step) {
    case 0: {
        Step s(out, "router_sigmoid_topk");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n_expert").mirror(p, n_p, "top_k")
         .mirror(p, n_p, "norm").mirror(p, n_p, "route_scale").mirror(p, n_p, "dtype");
        return s.wire({ 0, 1, 2, 3 });              /* logits, bias, ids, weights */
    }
    case 1: {
        const bool inv = RAD_FUSE_HAS(present, 8) != 0;
        Step s(out, inv ? "moe_scatter_inv" : "moe_scatter");
        s.mirror(p, n_p, "M").pint("n_expert", nl).mirror(p, n_p, "top_k")
         .mirror_opt(p, n_p, "expert_base").mirror_opt(p, n_p, "protect");
        if (inv) return s.wire({ 2, 4, 5, 6, 8 });  /* expert_ids, sorted_tok, offset, count, inverse */
        return s.wire({ 2, 4, 5, 6 });              /* expert_ids, sorted_tok, offset, count */
    }
    case 2: {
        if (!RAD_FUSE_HAS(present, 7)) return RAD_E_NOTFOUND;
        Step s(out, "fill");
        s.mirror(p, n_p, "M").pint("n", 1).pstr("dtype", "bf16").pf64("value", 1.0);
        return s.wire({ 7 });                       /* y */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * gate_quant_fp8 -- the same epilogue with the gate in its own operand.
 * (kernel: libr4d/r4d_fused_quant_fp8.hip, gate_quant_fp8_kernel)
 *
 * Operands: [0] gate  [1] x  [2] q  [3] scale  [4] out_bf16?
 *
 * Unlike the packed sibling above, this has a chain. The kernel rounds the activation before the
 * multiply, because its reference is `sigmoid` then `mul`, two kernels with a bf16 buffer between:
 *
 *     const uint16_t ab = f2bf(ACT == 0 ? gv / (1.0f + expf(-gv)) : 1.0f / (1.0f + expf(-gv)));
 *     const uint16_t h  = f2bf(bf2f(ab) * bf2f(xr[c]));
 *
 * `sigmoid_bf16` (`f32_to_bf16(1.0f / (1.0f + expf(-bf2f(xr[i]))))`) and `mul_bf16`
 * (`f32_to_bf16(bf2f(a) * bf2f(b))`) match those two lines term for term.
 *
 * `act` silu is declined: the chain's `silu` step (silu_bf16 or libref) computes libref's
 * `x * act_sigmoid(x)`, two f32 roundings, where this kernel does one divide, so many elements
 * would differ in the last bit. Accepting silu needs the kernel to take the sigmoid first as well.
 * The gated attention's gate is a sigmoid, so nothing issues this row with silu and the shim's
 * `mode` defaults to 1.
 *
 * `act` none is the plain product: the chain is `mul` then `quant_act_fp8`, and the kernel rounds
 * the bf16 product once, exactly as mul_bf16 does. The product is TMP(0) or `out_bf16`.
 *
 * TMP(0) is act(gate), an [M, n] bf16 plane the fusion never writes. `out_bf16` is an operand: the
 * kernel stores the product there when asked and takes the codes over those bytes.
 */
extern "C" int r4d_fuse_gate_quant_fp8(const RadParam* p, int n_p, uint64_t present, int step,
                                       RadFuseStep* out) {

    if (!out) return RAD_E_INVAL;
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    if (!fp8_geometry_ok(p, n_p, M, n)) return RAD_E_UNSUPPORTED;
    /* The shim (r4d_fused_quant_fp8.hip) defaults this row's `mode` to 1 (sigmoid). */
    const bool none = act_is(p, n_p, "sigmoid", "none");
    if (!none && !act_is(p, n_p, "sigmoid", "sigmoid")) return RAD_E_UNSUPPORTED;
    const int16_t mid = RAD_FUSE_HAS(present, 4) ? (int16_t)4 : RAD_FUSE_TMP(0);

    if (none) {
        switch (step) {
        case 0: {
            /* gate * x, rounded to bf16 once: mul_bf16's store. */
            Step s(out, "mul");
            s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "dtype");
            return s.wire({ 0, 1, mid });           /* a, b, y */
        }
        case 1: {
            Step s(out, "quant_act_fp8");
            s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "group")
             .mirror(p, n_p, "dtype");
            return s.wire({ mid, 2, 3 });           /* x, q, scale */
        }
        default:
            return RAD_E_NOTFOUND;
        }
    }

    switch (step) {
    case 0: {
        /* sigmoid(gate) rounded to bf16: the buffer the unfused pair had between them. */
        Step s(out, "sigmoid");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "dtype");
        return s.wire({ 0, RAD_FUSE_TMP(0) });      /* x, y */
    }
    case 1: {
        Step s(out, "mul");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "dtype");
        return s.wire({ RAD_FUSE_TMP(0), 1, 4 });   /* a, b, y */
    }
    case 2: {
        Step s(out, "quant_act_fp8");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "group").mirror(p, n_p, "dtype");
        return s.wire({ 4, 2, 3 });                 /* x, q, scale */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * gemm_nt_q_gated at i8a8h -- the same fold over an IQR gate_up weight (r4d_iqr.h) under the
 * rotated int8 trunk. (row: gemm_iqr_gated, M <= 16; kernel: libr4d/r4d_gemm_iqr.hip, the gated
 * forms of its GEMV and 16-token tile, each byte-identical to the GEMM row gemm_iqr_nt_m64 serves
 * at that M followed by gated_had_quant_i8g.)
 *
 * Operands: [0] a  [1] a_scale  [2] b  [3] b_scale  [4] q  [5] scale  [6] out_bf16
 *
 * As gemm_nt_q_gated's chain, with `dtype` i8a8h on the GEMM and the rotated int8 quantiser.
 */
extern "C" int r4d_fuse_gemm_nt_q_gated_iqr(const RadParam* p, int n_p, uint64_t present, int step,
                                            RadFuseStep* out) {
    (void)present;
    if (!out) return RAD_E_INVAL;
    const int64_t M = pi(p, n_p, "M"), N = pi(p, n_p, "N"), K = pi(p, n_p, "K");
    const char* dt = r4d_pgets(p, n_p, "dtype");
    if (M <= 0 || N <= 0 || K <= 0 || N % 2) return RAD_E_UNSUPPORTED;
    if (pi(p, n_p, "group", R4D_FP8_BLOCK) != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    if (!dt || std::strcmp(dt, "i8a8h")) return RAD_E_UNSUPPORTED;
    if (!act_is(p, n_p, "silu", "silu")) return RAD_E_UNSUPPORTED;

    switch (step) {
    case 0: {
        Step s(out, "gemm_nt_q");
        s.mirror(p, n_p, "M").mirror(p, n_p, "N").mirror(p, n_p, "K")
         .mirror(p, n_p, "group").mirror(p, n_p, "dtype");
        /* a, a_scale, b, b_scale, y, a_sum, b_ref */
        return s.wire({ 0, 1, 2, 3, RAD_FUSE_TMP(0), RAD_FUSE_NONE, RAD_FUSE_NONE });
    }
    case 1: {
        Step s(out, "gated_had_quant_i8g");
        s.mirror(p, n_p, "M").pint("n", (long long)(N / 2)).mirror(p, n_p, "group")
         .pstr("dtype", "bf16").mirror(p, n_p, "act");
        return s.wire({ RAD_FUSE_TMP(0), 4, 5, 6 });   /* gate_up, q, scale, out_bf16 */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * gdn_conv_recurrent_update -- the decode convolution folded into the recurrent update's prologue.
 * (row: gdn_conv_recurrent_update_k128_v128_bf16; kernel: libr4d/
 * r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip, r4d_gdn_recurrent_update_kernel under CV)
 *
 * Operands: [0] x  [1] conv_w  [2] conv_b  [3] conv_state  [4] conv_idx  [5] a  [6] b  [7] A_log
 *           [8] dt_bias  [9] state  [10] state_idx  [11] num_accepted  [12] cu  [13] z
 *           [14] norm_w  [15] o  [16] o_q  [17] o_scale
 *
 * The pair's q, k and v are the three temporaries. Every parameter has the same name in the step
 * op, so each step mirrors its own: the convolution's four, then the recurrent update's. */
extern "C" int r4d_fuse_gdn_conv_recurrent_update(const RadParam* p, int n_p, uint64_t present,
                                                  int step, RadFuseStep* out) {
    if (!out) return RAD_E_INVAL;
    const int16_t cb = opt_opd(present, 2), oq = opt_opd(present, 16), os = opt_opd(present, 17);
    switch (step) {
    case 0: {
        Step s(out, "gdn_conv_update");
        s.mirror(p, n_p, "q_len").mirror(p, n_p, "head_k").mirror(p, n_p, "head_v")
         .mirror(p, n_p, "conv_width");
        /* x, w, b, conv_state, state_idx, num_accepted, cu, q, k, v */
        return s.wire({ 0, 1, cb, 3, 4, 11, 12, RAD_FUSE_TMP(0), RAD_FUSE_TMP(1),
                        RAD_FUSE_TMP(2) });
    }
    case 1: {
        Step s(out, "gdn_recurrent_update");
        s.mirror(p, n_p, "q_len").mirror(p, n_p, "head_k").mirror(p, n_p, "head_v")
         .mirror_opt(p, n_p, "n_head_v").mirror_opt(p, n_p, "scale").mirror_opt(p, n_p, "eps")
         .mirror_opt(p, n_p, "l2_eps").mirror_opt(p, n_p, "softplus_thr")
         .mirror_opt(p, n_p, "act").mirror_opt(p, n_p, "state_fp16")
         .mirror_opt(p, n_p, "state_form");
        /* q, k, v, a, b, A_log, dt_bias, state, state_idx, num_accepted, cu, z, norm_w, o, o_q,
         * o_scale */
        return s.wire({ RAD_FUSE_TMP(0), RAD_FUSE_TMP(1), RAD_FUSE_TMP(2), 5, 6, 7, 8, 9, 10, 11,
                        12, 13, 14, 15, oq, os });
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * qk_norm_rope_gate_kv_store -- the gated-attention prologue with the cache store folded on.
 * (rows: qk_norm_rope_gate_kv_store_fp8 and _bf16; kernel: libr4d/r4d_qk_norm_rope_gate.hip,
 * r4d_qk_norm_rope_gate_kernel under KV)
 *
 * Operands: [0] q_gate?  [1] k  [2] cos_sin  [3] positions  [4] q_w?  [5] k_w  [6] q_out?
 *           [7] k_out  [8] gate_out?  [9] v  [10] slot_mapping  [11] kv_cache
 *
 * No temporaries: the store reads the prologue's own `k_out`, which the fold writes as well. */
extern "C" int r4d_fuse_qk_norm_rope_gate_kv_store(const RadParam* p, int n_p, uint64_t present,
                                                   int step, RadFuseStep* out) {
    if (!out) return RAD_E_INVAL;
    auto opt = [present](int i) { return opt_opd(present, i); };
    switch (step) {
    case 0: {
        Step s(out, "qk_norm_rope_gate");
        s.mirror(p, n_p, "M").mirror(p, n_p, "head_dim").mirror(p, n_p, "n_head")
         .mirror(p, n_p, "n_head_kv").mirror(p, n_p, "rot").mirror(p, n_p, "q_dtype")
         .mirror_opt(p, n_p, "eps").mirror_opt(p, n_p, "wadd").mirror_opt(p, n_p, "cs_f32")
         .mirror_opt(p, n_p, "pos_i64");
        return s.wire({ opt(0), 1, 2, 3, opt(4), 5, opt(6), 7, opt(8) });
    }
    case 1: {
        Step s(out, "kv_store");
        s.mirror(p, n_p, "M").mirror(p, n_p, "head_dim").mirror(p, n_p, "n_head_kv")
         .mirror(p, n_p, "block_size").mirror(p, n_p, "kv_dtype")
         .mirror_opt(p, n_p, "k_scale").mirror_opt(p, n_p, "v_scale");
        /* k, v, slot_mapping, kv_cache */
        return s.wire({ 7, 9, 10, 11 });
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * qk_norm_rope_gate_kv_store_streamed -- the same prologue with the streamed store folded on.
 * (rows: qk_norm_rope_gate_kv_store_streamed_{fp8,bf16}; kernel: r4d_qk_norm_rope_gate.hip under
 * QK_KV_STREAMED)
 *
 * Operands: [0] q_gate?  [1] k  [2] cos_sin  [3] positions  [4] q_w?  [5] k_w  [6] q_out?
 *           [7] k_out  [8] gate_out?  [9] v  [10] slot_mapping  [11] kv_cache
 *           [12] res  [13] host
 *
 * The store is `kv_store_streamed`, whose `k` is the prologue's own `k_out`. */
extern "C" int r4d_fuse_qk_norm_rope_gate_kv_store_streamed(const RadParam* p, int n_p,
                                                            uint64_t present, int step,
                                                            RadFuseStep* out) {
    if (!out) return RAD_E_INVAL;
    auto opt = [present](int i) { return opt_opd(present, i); };
    switch (step) {
    case 0: {
        Step s(out, "qk_norm_rope_gate");
        s.mirror(p, n_p, "M").mirror(p, n_p, "head_dim").mirror(p, n_p, "n_head")
         .mirror(p, n_p, "n_head_kv").mirror(p, n_p, "rot").mirror(p, n_p, "q_dtype")
         .mirror_opt(p, n_p, "eps").mirror_opt(p, n_p, "wadd").mirror_opt(p, n_p, "cs_f32")
         .mirror_opt(p, n_p, "pos_i64");
        return s.wire({ opt(0), 1, 2, 3, opt(4), 5, opt(6), 7, opt(8) });
    }
    case 1: {
        Step s(out, "kv_store_streamed");
        s.mirror(p, n_p, "M").mirror(p, n_p, "head_dim").mirror(p, n_p, "n_head_kv")
         .mirror(p, n_p, "block_size").mirror(p, n_p, "kv_dtype")
         .mirror_opt(p, n_p, "k_scale").mirror_opt(p, n_p, "v_scale");
        /* k_out, v, slot_mapping, kv_cache, res, host */
        return s.wire({ 7, 9, 10, 11, 12, 13 });
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * qkv_norm_rope_kv_store -- qkv_norm_rope, then kv_store of its finished k and v.
 * (rows: qkv_norm_rope_kv_store_fp8 and _bf16; kernel: libr4d/r4d_qkv_norm_rope.hip,
 * qkv_norm_rope_kernel under KV)
 *
 * Operands: [0] q  [1] k  [2] v  [3] positions  [4] q_w  [5] k_w  [6] slot_mapping  [7] kv_cache
 *
 * No temporaries: the prologue rewrites q, k and v in place and the store reads k and v as the
 * prologue left them, which is what the kernel stores. */
extern "C" int r4d_fuse_qkv_norm_rope_kv_store(const RadParam* p, int n_p, uint64_t present,
                                               int step, RadFuseStep* out) {
    (void)present;
    if (!out) return RAD_E_INVAL;
    switch (step) {
    case 0: {
        Step s(out, "qkv_norm_rope");
        s.mirror(p, n_p, "M").mirror(p, n_p, "head_dim").mirror(p, n_p, "n_head")
         .mirror(p, n_p, "n_head_kv").mirror(p, n_p, "theta").mirror(p, n_p, "eps")
         .mirror_opt(p, n_p, "scale").mirror_opt(p, n_p, "mode").mirror_opt(p, n_p, "rotary_dim")
         .mirror_opt(p, n_p, "ladder").mirror_opt(p, n_p, "wadd").mirror_opt(p, n_p, "v_from_k")
         .mirror(p, n_p, "dtype").mirror_opt(p, n_p, "v_norm");
        return s.wire({ 0, 1, 2, 3, 4, 5 });
    }
    case 1: {
        Step s(out, "kv_store");
        s.mirror(p, n_p, "M").mirror(p, n_p, "head_dim").mirror(p, n_p, "n_head_kv")
         .mirror(p, n_p, "block_size").mirror(p, n_p, "kv_dtype")
         .mirror_opt(p, n_p, "k_scale").mirror_opt(p, n_p, "v_scale");
        /* k, v, slot_mapping, kv_cache */
        return s.wire({ 1, 2, 6, 7 });
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * add_rmsnorm -- add (x += a), then rmsnorm of the x it wrote.
 * (row: add_rmsnorm_bf16; kernel: libr4d/r4d_model_bf16.hip, add_rmsnorm_kernel)
 *
 * Operands: [0] x  [1] a  [2] w  [3] y
 *
 * No temporaries: the add writes x in place and the norm reads it. */
extern "C" int r4d_fuse_add_rmsnorm(const RadParam* p, int n_p, uint64_t present, int step,
                                    RadFuseStep* out) {
    (void)present;
    if (!out) return RAD_E_INVAL;
    switch (step) {
    case 0: {
        Step s(out, "add");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "dtype");
        /* a = x, b = the addend, y = x */
        return s.wire({ 0, 1, 0 });
    }
    case 1: {
        Step s(out, "rmsnorm");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "eps").mirror(p, n_p, "dtype")
         .mirror_opt(p, n_p, "wadd").mirror_opt(p, n_p, "mult");
        return s.wire({ 0, 2, 3 });
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * attn_paged_gate_quant -- attn_paged, then gate_quant_fp8 over its output.
 * (row: attn_decode_gq_h256_gqa12_fp8kv; kernels: libr4d/r4d_attn_decode_h256_gqa6.hip, the
 * `_gq` split-KV merges)
 *
 * Operands: [0] q  [1] kv_cache  [2] block_table  [3] seqused  [4] k_descale?  [5] v_descale?
 *           [6] cu_seqlens?  [7] out  [8] gate  [9] q8  [10] q8_scale
 *
 * No temporaries: gate_quant_fp8 reads `out` and writes its bf16 product back over it, the
 * in-place pair the fold replaces. Its rows are attn_paged's (token, head) pairs, so its `M` is the
 * band's query rows times the head count (a bound, which is how its shim reads it) and its `n` is
 * `head_dim`. */
static int fuse_attn_paged_gate(const RadParam* p, int n_p, uint64_t present, int step,
                                RadFuseStep* out, const char* gate_op) {
    if (!out) return RAD_E_INVAL;
    auto opt = [present](int i) { return opt_opd(present, i); };
    const int64_t q_len = pi(p, n_p, "q_len"), nh = pi(p, n_p, "n_head");
    const int64_t hd = pi(p, n_p, "head_dim"), seqs = pi(p, n_p, "max_seqs", 1);
    if (q_len <= 0 || nh <= 0 || hd <= 0 || seqs <= 0) return RAD_E_UNSUPPORTED;
    switch (step) {
    case 0: {
        Step s(out, "attn_paged");
        s.mirror(p, n_p, "q_len").mirror(p, n_p, "head_dim").mirror(p, n_p, "gqa")
         .mirror(p, n_p, "block_size").mirror(p, n_p, "causal").mirror(p, n_p, "window")
         .mirror(p, n_p, "q_dtype").mirror(p, n_p, "kv_dtype").mirror_opt(p, n_p, "scale")
         .mirror_opt(p, n_p, "max_ctx").mirror_opt(p, n_p, "n_head")
         .mirror_opt(p, n_p, "max_seqs");
        return s.wire({ 0, 1, 2, 3, opt(4), opt(5), opt(6), 7 });
    }
    case 1: {
        Step s(out, gate_op);
        s.pint("M", (long long)(seqs * q_len * nh)).pint("n", (long long)hd)
         .mirror(p, n_p, "group").pstr("dtype", "bf16").mirror(p, n_p, "act");
        return s.wire({ 8, 7, 9, 10, 7 });   /* gate, x, q, scale, out_bf16 */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}
extern "C" int r4d_fuse_attn_paged_gate_quant(const RadParam* p, int n_p, uint64_t present,
                                              int step, RadFuseStep* out) {
    return fuse_attn_paged_gate(p, n_p, present, step, out, "gate_quant_fp8");
}
/* attn_paged_gate_had_quant -- the same two steps with gate_had_quant_i8g as the second. */
extern "C" int r4d_fuse_attn_paged_gate_had_quant(const RadParam* p, int n_p, uint64_t present,
                                                  int step, RadFuseStep* out) {
    return fuse_attn_paged_gate(p, n_p, present, step, out, "gate_had_quant_i8g");
}

/* ==========================================================================================
 * ar_gather_hc_write -- moe_gather, then ar_hc_write.
 * (kernel: libr4d/r4d_ar_oneshot_2rank_exact.hip, r4d_ar_gather_hc_write_kernel, and at four ranks
 * libr4d/r4d_ar_oneshot_Nrank_exact.hip, r4d_ar_gather_hc_write_wide_kernel; shim: r4d_ar_entry.hip)
 *
 * Operands: [0] y_expert  [1] expert_w  [2] sorted_tok  [3] shared?  [4] shared_gate?  [5] y (inout)
 *           [6] inj  [7] h (inout)
 *
 * Two declared ops with nothing in between: the gather writes `y`, the message that is then
 * reduced in place and written into the streams. The shared-arm pair is optional at the same
 * positions in both ops, so its absence propagates through the wiring. `wire` and `world_size` go
 * to the reducing step unchanged; `world_size` 4 selects ar_hc_write_4rank, as the four-rank row
 * runs.
 */
extern "C" int r4d_fuse_ar_gather_hc_write(const RadParam* p, int n_p, uint64_t present, int step,
                                           RadFuseStep* out) {
    if (!out) return RAD_E_INVAL;
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), k = pi(p, n_p, "top_k");
    if (M <= 0 || n <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    switch (step) {
    case 0: {
        Step s(out, "moe_gather");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "top_k")
         .mirror(p, n_p, "dtype").mirror_opt(p, n_p, "act");
        return s.wire({ 0, 1, 2, 3, 4, 5 });   /* y_expert, expert_w, sorted_tok, shared, gate, y */
    }
    case 1: {
        Step s(out, "ar_hc_write");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "hc")
         .mirror(p, n_p, "world_size").mirror(p, n_p, "dtype").mirror_opt(p, n_p, "wire");
        return s.wire({ 5, 6, 7 });            /* y, inj, h */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * ar_rmsnorm_quant_fp8 -- the two-rank all-reduce with the norm and the quantiser folded on.
 * (kernel: libr4d/r4d_fused_quant_fp8.hip, ar_rmsnorm_quant_fp8_kernel; shim: r4d_ar_entry.hip)
 *
 * Operands: [0] x (inout)  [1] residual?  [2] w  [3] q  [4] scale  [5] out_bf16?
 *
 * The second step is `rmsnorm_quant_fp8`, an op with the same two optional operands in the same
 * positions, so `residual` and `out_bf16` are wired straight through and absence propagates
 * without reading `present`.
 *
 * Both wires match exactly. On the exact wire, pass one stores the standalone one-shot's f32 sum
 * as bf16 back into `x` (`a = f2bf(bf2f(xr[i]) + bf2f(pr[i])); xr[i] = a;`). On the compressed
 * wire the reduce loop stores the inverse-rotated sum into `x` as bf16, as the standalone
 * compressed all-reduce does. After the handshake both run `norm_quant_row`, the device function
 * of rmsnorm_quant_fp8. Contrast ar_ln_had_quant_i8, which does not round and has no hook.
 *
 * Three parameters are not copies, and each selects which all-reduce row serves the step:
 *
 *   `numel`      is M * n. `all_reduce` declares the largest message the instance carries, as an
 *                element count, so the band's high bound (how `pi` reads a range) is right.
 *
 *   `exact`      is the fused `wire` inverted. It is required on `all_reduce` and separates
 *                cArExact from cArWht6.
 *
 *   `min_bytes`  is 0, on the compressed wire only. The wht6 shim serves a message below its floor
 *                with the exact kernel (both share the instance's scratch and flag space), but the
 *                fused kernel has no such fallback. Without the key a small message could go out
 *                exact on one side of the comparison and rotated on the other.
 *
 * `hops` is not emitted: no libr4d row reads it, and two ranks have only the one-shot. `y` is
 * RAD_FUSE_NONE: the fused kernel reduces in place into `x`, as every all-reduce row does.
 */
extern "C" int r4d_fuse_ar_rmsnorm_quant_fp8(const RadParam* p, int n_p, uint64_t present, int step,
                                             RadFuseStep* out) {

    if (!out) return RAD_E_INVAL;
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    const int64_t world = pi(p, n_p, "world_size");
    const int64_t wire = pi(p, n_p, "wire", 0);
    if (!fp8_geometry_ok(p, n_p, M, n) || world <= 0) return RAD_E_UNSUPPORTED;
    /* The entry point serves wires 0 and 6 only; any other has no kernel and so no chain. */
    if (wire != 0 && wire != 6) return RAD_E_UNSUPPORTED;

    switch (step) {
    case 0: {
        Step s(out, "all_reduce");
        s.mirror(p, n_p, "world_size").pint("numel", (long long)(M * n))
         .mirror(p, n_p, "dtype").pint("exact", wire == 0 ? 1 : 0);
        if (wire != 0) s.pint("min_bytes", 0);
        return s.wire({ 0, RAD_FUSE_NONE });        /* x (inout), y */
    }
    case 1: {
        Step s(out, "rmsnorm_quant_fp8");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "eps")
         .mirror(p, n_p, "group").mirror(p, n_p, "dtype").mirror_opt(p, n_p, "wadd");
        return s.wire({ 0, 1, 2, 3, 4, 5 });        /* x, residual, w, q, scale, out_bf16 */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}
