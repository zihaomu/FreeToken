/* ref_moe.cpp -- routing, the scatter/gather pair, the grouped GEMM, and the exact row top-k.
 *
 * ============================== What sorted_tok carries ==============================
 *
 * The flattened (token, slot) index, token * top_k + slot, grouped by expert; not the bare token
 * id, with which moe_gather could not find which of a token's routing weights applies to a row.
 * This is vLLM's `sorted_token_ids` convention.
 *
 * A slot whose expert id is outside [0, n_expert) is dropped: not counted, and the tail of
 * sorted_tok from expert_offset[n_expert] to M*top_k is filled with -1. A negative id marks a
 * padded batch row.
 *
 * The resulting histogram is what rad_route_report hands the heat engine; it is read back
 * asynchronously and consumed on the next step, without blocking the compute stream (spec §5.5).
 */
#include "ref_common.h"
#include "ref_gemm.h"
#include "ref_ops.h"
#include "ref_planes.h"
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <cstdlib>
#include <vector>

using namespace ref;

/* ------------------------------------------------------------------ router_topk */
/* Softmax over all n_expert logits, then the top_k by probability, then (with `norm`) renormalise
 * the kept weights to sum to one. Softmax-then-select is the order Mixtral, Qwen3 and every model in
 * scope use; selecting first would be a different router. */
int ref_router_topk(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *ids = &a->t[1], *w = &a->t[2];
    const int64_t n_expert = p_int(a, "n_expert", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    const int64_t top_k    = p_int(a, "top_k", 1);
    const int     norm     = (int)p_int(a, "norm", 0);
    if (n_expert <= 0 || top_k <= 0 || top_k > n_expert) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_expert;
    if (M <= 0 || numel(ids) < M * top_k || numel(w) < M * top_k) return RAD_E_SHAPE;

    REF_PAR_FOR(M * n_expert * (top_k + 2))
    for (int64_t m = 0; m < M; ++m) {
        const int64_t lb = rowoff(lg, m, n_expert), ls = laststride(lg);
        float mx = -INFINITY;
        for (int64_t e = 0; e < n_expert; ++e) {
            float v = ldt(lg, lb + e * ls);
            if (v > mx) mx = v;
        }
        float sum = 0.0f;
        for (int64_t e = 0; e < n_expert; ++e) sum += ref_exp(ldt(lg, lb + e * ls) - mx);
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

        /* Select on the logits: softmax is monotone, so the order is the same and the comparison
         * avoids an exp. */
        float pv = INFINITY;
        int64_t pi = -1, kept = 0;
        float wsum = 0.0f;
        for (int64_t r = 0; r < top_k; ++r) {
            if (!topk_next(lg, lb, ls, n_expert, &pv, &pi)) break;
            const float p = ref_exp(pv - mx) * inv;
            st_int(ids->dtype, ids->data, rowoff(ids, m, top_k) + r * laststride(ids), pi);
            stt(w, rowoff(w, m, top_k) + r * laststride(w), p);
            wsum += p;
            ++kept;
        }
        if (norm && wsum > 0.0f) {
            const float n2 = 1.0f / wsum;
            for (int64_t r = 0; r < kept; ++r) {
                const int64_t off = rowoff(w, m, top_k) + r * laststride(w);
                stt(w, off, ldt(w, off) * n2);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ moe_scatter */
/* `protect`: local experts the sort places last, in the order listed, the rest closed up; the
 * layout a layer serving a few experts with a second grouped GEMM needs (libr4d's sort_place).
 * Returns false unless the list is strictly ascending within [0, n). */
static bool sort_places(const char* s, int64_t n, std::vector<int64_t>* place) {
    std::vector<int64_t> prot;
    for (const char* c = s ? s : ""; *c;) {
        char* end = nullptr;
        const long long v = std::strtoll(c, &end, 10);
        if (end == c || v < 0 || v >= n || (!prot.empty() && v <= prot.back())) return false;
        prot.push_back(v);
        c = end;
        if (*c == ',') ++c;
        else if (*c) return false;
    }
    place->assign((size_t)n, 0);
    int64_t next = 0;
    for (int64_t e = 0; e < n; ++e)
        if (!std::binary_search(prot.begin(), prot.end(), e)) (*place)[(size_t)e] = next++;
    for (size_t j = 0; j < prot.size(); ++j) (*place)[(size_t)prot[j]] = next++;
    return true;
}

/* A stable counting sort of the (token, slot) pairs by expert id. Serial: a parallel stable version
 * would need per-thread histograms and a prefix pass, more code than the work it saves. */
int ref_moe_scatter(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *ids = &a->t[0], *sorted = &a->t[1], *off = &a->t[2], *cnt = &a->t[3];
    const int64_t n_expert = p_int(a, "n_expert", 0);
    const int64_t top_k    = p_int(a, "top_k", 1);
    /* This rank's first expert, subtracted from every routed id so the sort runs over local
     * indices, as the weight table, offsets and downstream GEMM expect. */
    const int64_t base     = p_int(a, "expert_base", 0);
    if (n_expert <= 0 || top_k <= 0 || base < 0) return RAD_E_SHAPE;
    const int64_t M = numel(ids) / top_k;
    const int64_t T = M * top_k;
    if (M <= 0 || numel(sorted) < T) return RAD_E_SHAPE;
    if (numel(off) < n_expert + 1 || numel(cnt) < n_expert) return RAD_E_SHAPE;
    std::vector<int64_t> place;
    if (!sort_places(p_str(a, "protect", nullptr), n_expert, &place)) return RAD_E_SHAPE;
    /* A slot's place in the sort, or -1 for an expert outside this call's slice. */
    auto at = [&](int64_t i) -> int64_t {
        const int64_t e = ld_int(ids->dtype, ids->data, offlin(ids, i)) - base;
        return (e < 0 || e >= n_expert) ? -1 : place[(size_t)e];
    };

    for (int64_t e = 0; e < n_expert; ++e) st_int(cnt->dtype, cnt->data, offlin(cnt, e), 0);
    for (int64_t i = 0; i < T; ++i) {
        const int64_t e = at(i);
        if (e < 0) continue;
        const int64_t o = offlin(cnt, e);
        st_int(cnt->dtype, cnt->data, o, ld_int(cnt->dtype, cnt->data, o) + 1);
    }
    int64_t run = 0;
    for (int64_t e = 0; e < n_expert; ++e) {
        st_int(off->dtype, off->data, offlin(off, e), run);
        run += ld_int(cnt->dtype, cnt->data, offlin(cnt, e));
    }
    st_int(off->dtype, off->data, offlin(off, n_expert), run);

    /* Each slot goes to its expert's cursor, in ascending slot order. The cursors are a copy, so
     * the offsets (an output) are never mutated. */
    for (int64_t i = run; i < T; ++i) st_int(sorted->dtype, sorted->data, offlin(sorted, i), -1);
    std::vector<int64_t> cur((size_t)n_expert);
    for (int64_t e = 0; e < n_expert; ++e)
        cur[(size_t)e] = ld_int(off->dtype, off->data, offlin(off, e));
    for (int64_t i = 0; i < T; ++i) {
        const int64_t e = at(i);
        if (e >= 0) st_int(sorted->dtype, sorted->data, offlin(sorted, cur[(size_t)e]++), i);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ router_topk_scatter */
/* The pair as one op, and the reference is the pair: the fused device kernel must be
 * byte-identical to these two launches, in order, over the operands the fused op carries. The
 * selection runs over the whole router's `n_expert` logits; the sort runs over this rank's slice,
 * the fused op's `n_local` standing where `moe_scatter`'s `n_expert` names the slice it owns. */
int ref_router_topk_scatter(const RadArgs* a, RadStream s) {
    if (!have(a, 6)) return RAD_E_INVAL;
    const int64_t ne = p_int(a, "n_expert", 0);
    const int64_t nl = p_int(a, "n_local", ne);
    const int64_t k  = p_int(a, "top_k", 1);
    if (ne <= 0 || k <= 0 || k > ne || nl <= 0 || nl > ne) return RAD_E_SHAPE;

    const RadParam rp[] = { RAD_INT("n_expert", ne), RAD_INT("top_k", k),
                            RAD_INT("norm", p_int(a, "norm", 0)),
                            RAD_STR("dtype", p_str(a, "dtype", "bf16")) };
    const RadArgs ra{ a->t, 3, rp, 4, nullptr, 0, nullptr, a->rank, a->world_size };
    const int rc = ref_router_topk(&ra, s);
    if (rc != RAD_OK) return rc;

    /* The fused op's operands with `expert_w` removed -- the sort's input is the selection's
     * output, named once, and `moe_scatter` takes it in operand slot 0. RadArgs reads a
     * contiguous tensor array, so the four descriptors are laid out here, by value. */
    const RadTensor t2[] = { a->t[1], a->t[3], a->t[4], a->t[5] };
    const RadParam sp[]   = { RAD_INT("n_expert", nl), RAD_INT("top_k", k),
                              RAD_INT("expert_base", p_int(a, "expert_base", 0)),
                              RAD_STR("protect", p_str(a, "protect", nullptr)) };
    const RadArgs sa{ t2, 4, sp, 4, nullptr, 0, nullptr, a->rank, a->world_size };
    return ref_moe_scatter(&sa, s);
}

/* ------------------------------------------------------------------ router_gemm_topk_scatter */
/* The router GEMM in front of the pair, and with `gate_b`/`gate` present the shared expert's
 * one-column gate over the same input. The reference is the launches it stands in for, and the
 * fused kernel must be byte-identical to them: the router's `gemm_nt`, the gate's `gemm_nt` at
 * N = 1, then `router_topk_scatter` over the logits. `gate_b` and `gate` come together or not at
 * all, and the operand count says which. */
int ref_router_gemm_topk_scatter(const RadArgs* a, RadStream s) {
    /* NOT `have`: the required operands are not a prefix -- `gate_b` sits between the router's
     * weight and the outputs, and it and `gate` are null slots in the gate-absent form. The nine
     * required slots are checked by name; the pair is checked to come together or not at all. */
    if (!a || !a->t || a->n_t < 9) return RAD_E_INVAL;
    const bool gate_b = a->n_t > 2 && a->t[2].data != nullptr;
    const bool gate   = a->n_t > 9 && a->t[9].data != nullptr;
    if (gate_b != gate) return RAD_E_INVAL;
    for (int i : {0, 1, 3, 4, 5, 6, 7, 8})
        if (!a->t[i].data) return RAD_E_INVAL;
    const int64_t ne = p_int(a, "n_expert", 0);
    const int64_t nl = p_int(a, "n_local", ne);
    const int64_t k  = p_int(a, "top_k", 1);
    const int64_t K  = p_int(a, "K", 0);
    if (ne <= 0 || k <= 0 || k > ne || nl <= 0 || nl > ne || K <= 0) return RAD_E_SHAPE;
    const char* dt = p_str(a, "dtype", "bf16");

    /* The router: a [M, K] x b [n_expert, K]^T -> logits. gemm_nt takes a, b, y contiguously. */
    const RadTensor g3[] = { a->t[0], a->t[1], a->t[3] };
    const RadParam  gp[] = { RAD_INT("N", ne), RAD_INT("K", K), RAD_STR("dtype", dt) };
    const RadArgs   ga{ g3, 3, gp, 3, nullptr, 0, nullptr, a->rank, a->world_size };
    const int gc = ref_gemm_nt(&ga, s);
    if (gc != RAD_OK) return gc;

    /* The shared expert's gate: the same arithmetic at N = 1, beside the router, over the same
     * `a`. The pair was checked to come together. */
    if (gate) {
        const RadTensor p3[] = { a->t[0], a->t[2], a->t[9] };
        const RadParam  pp[] = { RAD_INT("N", 1), RAD_INT("K", K), RAD_STR("dtype", dt) };
        const RadArgs   pa{ p3, 3, pp, 3, nullptr, 0, nullptr, a->rank, a->world_size };
        const int pc = ref_gemm_nt(&pa, s);
        if (pc != RAD_OK) return pc;
    }

    /* The pair over the logits the GEMM just wrote. Every parameter the pair reads is a parameter
     * of this op, under the same name -- the fused lists are unions, so `a`'s list passes through. */
    const RadTensor t6[] = { a->t[3], a->t[4], a->t[5], a->t[6], a->t[7], a->t[8] };
    const RadArgs   fa{ t6, 6, a->p, a->n_p, nullptr, 0, nullptr, a->rank, a->world_size };
    return ref_router_topk_scatter(&fa, s);
}

/* ------------------------------------------------------------------ moe_gemm, moe_gemm_q */
/* y[i, :] = a[row(i), :] @ w[e]^T, e the expert whose offset range contains i. `a_order` picks
 * row(i): `sorted_tok[i] / top_k` for one activation row per token, i itself for one per
 * (token, slot). The output is in sorted order (one row per (token, slot), grouped as moe_scatter
 * left them) so the expert loop is contiguous; moe_gather restores token order.
 *
 * The weight operand is a table (RAD_OPD_WTAB), read through ref_gemm.h's expert_entry(): experts
 * are separate movement units, so one may be in VRAM while the next is streamed. The stacked form
 * is also accepted for callers without an engine (rad-kbench fixtures).
 *
 * The quantised arm uses ref_gemm.h's Scales and quant_dot, accumulating raw codes and scaling once
 * per group exactly as gemm_nt_q does. Only the weight scale grid is block-shaped (one scale per
 * 128x128 tile); the activation's is per row per K group (rblk = 1), which make_scales derives from
 * the extents. */
/* quant_tile with the loaders resolved at compile time for the grouped GEMM's dtype pairs (the
 * same floats as Dyn, which serves the rest). */
typedef float* (*TileFn)(const Mat&, const Mat&, const Scales&, const Scales&, const int64_t*,
                         int64_t, int64_t, int64_t, int64_t, float*);

#define REF_TILE_ROW(DA)                                                   \
    case DA:                                                               \
        switch (db) {                                                      \
            case RAD_F8E4M3:  return &quant_tile<Fix<DA>, Fix<RAD_F8E4M3>>; \
            case RAD_I4:      return &quant_tile<Fix<DA>, Fix<RAD_I4>>;     \
            case RAD_I8:      return &quant_tile<Fix<DA>, Fix<RAD_I8>>;     \
            case RAD_BF16:    return &quant_tile<Fix<DA>, Fix<RAD_BF16>>;   \
            case RAD_F32:     return &quant_tile<Fix<DA>, Fix<RAD_F32>>;    \
            default:          return &quant_tile<Dyn, Dyn>;                 \
        }

static TileFn pick_tile(uint32_t da, uint32_t db) {
    switch (da) {
        REF_TILE_ROW(RAD_F8E4M3)
        REF_TILE_ROW(RAD_BF16)
        REF_TILE_ROW(RAD_F32)
        REF_TILE_ROW(RAD_I8)
        default: return &quant_tile<Dyn, Dyn>;
    }
}
#undef REF_TILE_ROW

/* ------------------------------------------------------------------ the codebook experts
 *
 * Qwen3.8-Flash-Next's experts (docs/MOE-W4.md) are u4 or u5 codes indexing a table of 16 or 32
 * f32 levels, with a scale per (row, group of K): bf16 per 128 (`w4nla8h`), or E4M3 per 64 or 32
 * times one fixed f32 (`w4nl64a8h`, `w4nl32a8h`, `w5nl64a8h`). Element (n, k) of an expert is
 * table[code(n, k)] * scale(n, k / group), times `scale.1` where there is one. The codes view takes
 * the planes `codes,table` and the two-level forms' scale view `scale,scale.1`; such a weight is
 * stored in the joined form of abi/rad_planes.h, which these rows' layout hooks ask for. A view of
 * one plane (w4nla8h's bf16 scale, every view of the other dtypes) is stored as it is.
 *
 * The arithmetic is quant_dot's: per weight group, the sub-sum of table values times activation
 * codes, then `(sub * sa) * sb` with sb the scale times its level in f32. The logical form the
 * oracle and rad-kbench pass (enc_selection_logical: table entries for codes, scale products for
 * scales) goes through the same chain, so the two readings agree bit for bit wherever the logical
 * operands hold the values exactly, as E4M3 and bf16 do for the shipped tables and scales. libr4d
 * folds each 64- or 32-wide group's sum under its scale first and the activation's scale per 128
 * after (`(sub0 * sb0 + sub1 * sb1) * sa`), which differs in the last bits. */

/* What a codebook `dtype` stores: the code width, the scale's group along K and dtype, and whether
 * the scale view carries the level above it. `bits` is 0 for any other dtype. */
struct CbForm { int bits; int64_t group; uint32_t sdt; bool level; };
static CbForm cb_form(const RadParam* p, int n_p) {
    const RadParam* d = rad_param_find(p, n_p, "dtype");
    const char* s = d && d->kind == RAD_P_STR ? d->sval : nullptr;
    if (s && !std::strcmp(s, "w4nla8h"))   return CbForm{ 4, 128, RAD_BF16, false };
    if (s && !std::strcmp(s, "w4nl64a8h")) return CbForm{ 4, 64, RAD_F8E4M3, true };
    if (s && !std::strcmp(s, "w4nl32a8h")) return CbForm{ 4, 32, RAD_F8E4M3, true };
    if (s && !std::strcmp(s, "w5nl64a8h")) return CbForm{ 5, 64, RAD_F8E4M3, true };
    /* The same stored experts under their names against an int8 activation. */
    if (s && !std::strcmp(s, "w4nl64a8i8h")) return CbForm{ 4, 64, RAD_F8E4M3, true };
    if (s && !std::strcmp(s, "w4nl32a8i8h")) return CbForm{ 4, 32, RAD_F8E4M3, true };
    return CbForm{ 0, 0, 0, false };
}

/* Whether a selection of several planes for `operand` is what `dtype` names: the codes operands of
 * both grouped GEMMs are 2, 6, 8 and 10 and their scales 3, 7, 9 and 11. RAD_E_DTYPE otherwise,
 * which refuses the weight at declare by name. */
static int cb_check(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                    const int* sel, int n_planes) {
    const CbForm f = cb_form(p, n_p);
    const bool codes = operand == 2 || operand == 6 || operand == 8 || operand == 10;
    const bool scale = operand == 3 || operand == 7 || operand == 9 || operand == 11;
    if (!f.bits || (!codes && !scale) || !enc || !sel || n_planes != 2 ||
        std::strcmp(enc->transform, "fwht128") != 0)
        return RAD_E_DTYPE;
    for (int i = 0; i < 2; ++i)
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return RAD_E_SHAPE;
    const RadEncPlane& a = enc->plane[sel[0]];
    const RadEncPlane& b = enc->plane[sel[1]];
    if (codes)
        return !std::strcmp(a.role, "codes") && a.kind == RAD_PLANE_TILED &&
               a.dtype == (f.bits == 5 ? (uint32_t)RAD_U5 : (uint32_t)RAD_U4) &&
               a.block[0] == 1 && a.block[1] == 1 && !std::strcmp(b.role, "table") &&
               b.kind == RAD_PLANE_TABLE && b.dtype == RAD_F32 && b.extent[0] == 1 &&
               b.extent[1] == ((int64_t)1 << f.bits)
                   ? RAD_OK : RAD_E_DTYPE;
    return f.level && !std::strcmp(a.role, "scale") && a.kind == RAD_PLANE_TILED &&
           a.dtype == f.sdt && a.block[0] == 1 && a.block[1] == f.group &&
           !std::strcmp(b.role, "scale.1") && b.kind == RAD_PLANE_TILED && b.dtype == RAD_F32 &&
           b.block[0] == 0 && b.block[1] == 0
               ? RAD_OK : RAD_E_DTYPE;
}

extern "C" int ref_moe_cb_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                 const int* sel, const RadTensor* planes, int n_planes,
                                 RadLayout* out) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const int rc = cb_check(p, n_p, operand, enc, sel, n_planes);
    return rc < 0 ? rc : rad_planes_layout(enc, sel, planes, n_planes, out);
}

extern "C" int ref_moe_cb_relayout(const RadParam* p, int n_p, int operand,
                                   const RadEncoding* enc, const int* sel, const RadTensor* planes,
                                   int n_planes, void* dst, int64_t dst_bytes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const int rc = cb_check(p, n_p, operand, enc, sel, n_planes);
    return rc < 0 ? rc : rad_planes_relayout(enc, sel, planes, n_planes, dst, dst_bytes);
}

extern "C" int ref_moe_cb_unrelayout(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const void* src,
                                     int64_t src_bytes, const RadTensor* planes, int n_planes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const int rc = cb_check(p, n_p, operand, enc, sel, n_planes);
    return rc < 0 ? rc : rad_planes_unrelayout(enc, sel, src, src_bytes, planes, n_planes);
}

/* The joined form's header when weight entry `t` holds one. An entry smaller than a header is a
 * plain plane, and is not read past its end. */
static const RadPlanesHeader* cb_joined(const RadTensor& t) {
    int64_t n = 1;
    for (uint32_t i = 0; i < t.rank; ++i) n *= t.shape[i];
    if (!t.data || rad_dtype_bytes(t.dtype, n) < (int64_t)sizeof(RadPlanesHeader)) return nullptr;
    return rad_planes_header(t.data);
}

/* A codebook expert's codes, which LdCb reads as table values. The RadTensor comes first, so a Mat
 * over `t` reaches the rest: element i is (i / cols, i % cols) of a plane whose rows start
 * `row_bytes` apart, the layout rad_planes.h keeps. */
struct CbTensor {
    RadTensor t;              /* `data`: the codes plane, from the tile's first weight row */
    float     tab[32];
    int64_t   cols = 0, row_bytes = 0;
    int       bits = 0;
};
struct LdCb {
    static inline float ld(const RadTensor* t, int64_t i) {
        const CbTensor* c = reinterpret_cast<const CbTensor*>(t);
        const int64_t r = i / c->cols, k = i - r * c->cols;
        return c->tab[rad_load_bits((const uint8_t*)c->t.data + r * c->row_bytes, c->bits, k)];
    }
};

/* An expert's codes when its entry is the joined form: 1 with `out` filled, 0 for a plain plane
 * (any dtype but u4 and u5), RAD_E_DTYPE for a form these kernels do not read and RAD_E_SHAPE for
 * codes not covering [n, k]. A u4 or u5 entry is always joined: only its table says what a code
 * means. */
static int cb_codes(const RadTensor& w, int64_t n, int64_t k, CbTensor* out) {
    if (w.dtype != RAD_U4 && w.dtype != RAD_U5) return 0;
    const RadPlanesHeader* h = rad_planes_header(w.data);
    const RadPlanesEntry *ec = nullptr, *et = nullptr;
    const void* codes = h ? rad_planes_find(w.data, "codes", &ec) : nullptr;
    const void* tab = h ? rad_planes_find(w.data, "table", &et) : nullptr;
    if (!codes || !tab || h->n_planes != 2) return RAD_E_DTYPE;
    const int bits = ec->dtype == RAD_U4 ? 4 : ec->dtype == RAD_U5 ? 5 : 0;
    if (!bits || ec->kind != RAD_PLANE_TILED || ec->block[0] != 1 || ec->block[1] != 1 ||
        et->kind != RAD_PLANE_TABLE || et->dtype != RAD_F32 ||
        et->rows * et->cols != ((int64_t)1 << bits))
        return RAD_E_DTYPE;
    if (ec->rows < n || ec->cols < k) return RAD_E_SHAPE;
    out->t = w;
    out->t.data = const_cast<void*>(codes);
    std::memcpy(out->tab, tab, sizeof(float) << bits);
    out->cols = ec->cols;
    out->row_bytes = ec->row_bytes;
    out->bits = bits;
    return 1;
}

/* An expert's scales beside joined codes: the scale times the level above it. */
struct CbScale {
    Scales s;
    float  mul = 1.0f;
    inline float at(int64_t row, int64_t g) const { return (s.p ? s.grp(row, g) : 1.0f) * mul; }
};

/* From the joined `scale,scale.1` form when the entry holds one, else the plane as make_scales
 * reads it (w4nla8h's bf16 scale, or none). RAD_E_DTYPE for a joined form of other planes. */
static int cb_scale(const RadTensor* ws, int64_t n, int64_t k, CbScale* out) {
    out->mul = 1.0f;
    const RadPlanesHeader* h = ws ? cb_joined(*ws) : nullptr;
    if (!h) {
        out->s = make_scales(ws, n, k, false);
        return RAD_OK;
    }
    const RadPlanesEntry *es = nullptr, *e1 = nullptr;
    const void* sp = rad_planes_find(ws->data, "scale", &es);
    const void* lp = rad_planes_find(ws->data, "scale.1", &e1);
    if (!sp || !lp || h->n_planes != 2) return RAD_E_DTYPE;
    const uint32_t dt = es->dtype;
    if (es->kind != RAD_PLANE_TILED || es->block[0] < 1 || es->block[1] < 1 ||
        (dt != RAD_F8E4M3 && dt != RAD_BF16 && dt != RAD_F16 && dt != RAD_F32) ||
        e1->kind != RAD_PLANE_TILED || e1->dtype != RAD_F32 || e1->block[0] != 0 ||
        e1->block[1] != 0 || e1->rows != 1 || e1->cols != 1)
        return RAD_E_DTYPE;
    if ((n + es->block[0] - 1) / es->block[0] > es->rows ||
        (k + es->block[1] - 1) / es->block[1] > es->cols)
        return RAD_E_SHAPE;
    Scales s{};
    s.p = sp;
    s.dt = dt;
    s.rs = es->cols;
    s.cs = 1;
    s.group = es->block[1];
    s.rblk = es->block[0];
    s.e8m0 = false;
    out->s = s;
    std::memcpy(&out->mul, lp, sizeof out->mul);
    return 1;
}

static TileFn pick_cb_tile(uint32_t da) {
    switch (da) {
        case RAD_F8E4M3: return &quant_tile<Fix<RAD_F8E4M3>, LdCb>;
        case RAD_BF16:   return &quant_tile<Fix<RAD_BF16>, LdCb>;
        case RAD_F32:    return &quant_tile<Fix<RAD_F32>, LdCb>;
        default:         return &quant_tile<Dyn, LdCb>;
    }
}

static int moe_gemm_impl(const RadArgs* a, bool quant, int ncls = 2) {
    const int n_opd = quant ? (ncls == 4 ? 13 : 9) : 5;
    if (!have(a, n_opd)) return RAD_E_INVAL;
    const RadTensor* A      = &a->t[0];
    const RadTensor* as_    = quant ? t_in(a, 1) : nullptr;
    const RadTensor* W      = &a->t[quant ? 2 : 1];
    const RadTensor* ws_    = quant ? t_in(a, 3) : nullptr;
    const RadTensor* sorted = &a->t[quant ? 4 : 2];
    const RadTensor* off    = &a->t[quant ? 5 : 3];
    const RadTensor* Y      = &a->t[quant ? (ncls == 4 ? 12 : 8) : 4];
    /* The quantised arm's experts are two tables by parity (see ref_registry.cpp), or four by
     * expert number mod 4 for `moe_gemm_q_mod4`; the plain arm has one, every expert in class 0. */
    const RadTensor* W1     = quant ? &a->t[6] : W;
    const RadTensor* ws1_   = quant ? t_in(a, 7) : nullptr;
    const bool two = quant;
    const int  nc  = quant ? ncls : 1;
    const RadTensor* Wq[4]  = { W, W1, quant && nc == 4 ? &a->t[8] : W,
                                quant && nc == 4 ? &a->t[10] : W };
    const RadTensor* wsq[4] = { ws_, ws1_, quant && nc == 4 ? t_in(a, 9) : nullptr,
                                quant && nc == 4 ? t_in(a, 11) : nullptr };

    /* "sorted": `a` has one row per (token, slot) in scatter order (the down projection, reading
     * the first projection's output). "token" (default, gate_up): one row per token, gathered
     * through sorted_tok. See ref_registry.cpp for why this is stated, not derived. */
    const char* ord = p_str(a, "a_order", "token");
    const bool  sorted_a = ord && std::strcmp(ord, "sorted") == 0;
    if (ord && !sorted_a && std::strcmp(ord, "token") != 0) return RAD_E_UNSUPPORTED;

    const int64_t N0 = p_int(a, "N", W->rank >= 3 ? W->shape[1] : 0);
    const int64_t K0 = p_int(a, "K", W->rank >= 3 ? W->shape[2] : 0);
    /* Each class's geometry, defaulting to class 0's: N_odd/K_odd at two classes, N_q/K_q at
     * four. */
    static const char* const kN4[4] = { "N", "N_1", "N_2", "N_3" };
    static const char* const kK4[4] = { "K", "K_1", "K_2", "K_3" };
    int64_t Nq[4] = { N0, N0, N0, N0 }, Kq[4] = { K0, K0, K0, K0 };
    for (int q = 1; q < nc; ++q) {
        Nq[q] = p_int(a, nc == 4 ? kN4[q] : "N_odd", N0);
        Kq[q] = p_int(a, nc == 4 ? kK4[q] : "K_odd", K0);
    }
    const int64_t parts = two ? p_int(a, "parts", 1) : 1;
    const int64_t n_expert = p_int(a, "n_expert", W->rank >= 3 ? W->shape[0] : 0);
    const int64_t top_k = p_int(a, "top_k", 1);
    if (parts <= 0 || n_expert <= 0 || top_k <= 0) return RAD_E_SHAPE;
    /* The output is the widest class's N and the activation the longest class's K. Class q holds
     * ceil((n - q) / nc) experts. */
    int64_t N = 0, K = 0;
    for (int q = 0; q < nc; ++q) {
        if (Nq[q] <= 0 || Kq[q] <= 0 || Nq[q] % parts) return RAD_E_SHAPE;
        if (Nq[q] > N) N = Nq[q];
        if (Kq[q] > K) K = Kq[q];
        const int64_t neq = (n_expert + nc - 1 - q) / nc;
        if (neq > 0 && (Wq[q]->rank < 3 || Wq[q]->shape[0] < neq)) return RAD_E_SHAPE;
    }
    const int64_t T = numel(Y) / N;
    if (T <= 0 || numel(sorted) < T || numel(off) < n_expert + 1) return RAD_E_SHAPE;

    /* `y` has one row per (token, slot); `a` has T rows under "sorted" and M = T / top_k under
     * "token", checked against the matching count (M rows under "sorted" means the down projection
     * was handed gate_up's input buffer). */
    const int64_t M  = T / top_k;
    const int64_t AR = sorted_a ? T : M;
    if (M <= 0 || numel(A) < AR * K) return RAD_E_SHAPE;

    const Mat ma = as_mat(A, K);
    const Scales SA = make_scales(as_, AR, K, false);
    /* The activation's scales are per `a_group` of K where the caller states one (an int8
     * activation at one scale per 256 against experts scaled per 128), else per `group`. */
    const int64_t gp = p_int(a, "a_group", p_int(a, "group", 0));
    if (quant && gp > 0 && SA.p && SA.group != gp) return RAD_E_SHAPE;

    const int64_t y_rs = row_stride(Y, N);
    const int64_t y_cs = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;

    /* Per-expert views, built once outside the row loop (and validated there, since an OpenMP body
     * cannot return). Expert e is entry e / nc of class e % nc. */
    std::vector<RadTensor> we((size_t)n_expert), wse((size_t)n_expert);
    std::vector<Mat>       mw((size_t)n_expert);
    std::vector<Scales>    SW((size_t)n_expert);
    std::vector<int64_t>   en((size_t)n_expert), ek((size_t)n_expert);
    /* Codebook experts (joined codes): every expert of the call, or none. */
    std::vector<CbTensor>  cbt;
    std::vector<CbScale>   cbs;
    bool codebook = false;
    for (int64_t e = 0; e < n_expert; ++e) {
        const int q = (int)(e % nc);
        const RadTensor* tw = Wq[q];
        const RadTensor* ts = wsq[q];
        const int64_t ix = e / nc;
        const int64_t n = Nq[q], k = Kq[q];
        en[(size_t)e] = n;
        ek[(size_t)e] = k;
        we[(size_t)e] = expert_entry(tw, ix);
        if (!we[(size_t)e].data) return RAD_E_STRIDE;
        if (we[(size_t)e].rank < 2 || we[(size_t)e].shape[0] < n ||
            we[(size_t)e].shape[1] < k) return RAD_E_SHAPE;
        if (quant && ts) {
            wse[(size_t)e] = expert_entry(ts, ix);
            if (!wse[(size_t)e].data) return RAD_E_STRIDE;
        }
        if (quant) {
            CbTensor ct;
            const int jc = cb_codes(we[(size_t)e], n, k, &ct);
            if (jc < 0) return jc;
            if (e == 0 && jc > 0) {
                codebook = true;
                cbt.resize((size_t)n_expert);
                cbs.resize((size_t)n_expert);
            }
            if (codebook != (jc > 0)) return RAD_E_DTYPE;
            if (codebook) {
                cbt[(size_t)e] = ct;
                const int js = cb_scale(ts ? &wse[(size_t)e] : nullptr, n, k, &cbs[(size_t)e]);
                if (js < 0) return js;
                continue;
            }
            /* A joined scale is read only beside joined codes. */
            if (ts && cb_joined(wse[(size_t)e])) return RAD_E_DTYPE;
        }
        mw[(size_t)e] = as_mat(&we[(size_t)e], k);
        SW[(size_t)e] = make_scales(quant && ts ? &wse[(size_t)e] : nullptr, n, k,
                                    we[(size_t)e].dtype == RAD_FP4E2M1);
    }

    const int64_t fy = N / parts;

    /* Each row's expert (the binary search over the offsets) and activation row, or -1 for a row
     * that is all zeros: a dropped slot, or a token past `a`. Zeros are written, never stale
     * arena bytes, and so are each part's columns past a narrower class's width. */
    std::vector<int64_t> ex((size_t)T, -1), tk((size_t)T, 0);
    REF_PAR_FOR(T * N)
    for (int64_t i = 0; i < T; ++i) {
        const int64_t flat = ld_int(sorted->dtype, sorted->data, offlin(sorted, i));
        int64_t e = -1;
        if (flat >= 0) {
            int64_t lo = 0, hi = n_expert;
            while (lo + 1 < hi) {
                const int64_t mid = (lo + hi) / 2;
                if (ld_int(off->dtype, off->data, offlin(off, mid)) <= i) lo = mid; else hi = mid;
            }
            const int64_t tok = sorted_a ? i : flat / top_k;
            if (tok < AR) { e = lo; tk[(size_t)i] = tok; }
        }
        ex[(size_t)i] = e;
        const int64_t fc = e >= 0 ? en[(size_t)e] / parts : 0;
        for (int64_t n = 0; n < N; ++n)
            if (n - (n / fy) * fy >= fc) stt(Y, i * y_rs + n * y_cs, 0.0f);
    }

    /* The live rows grouped by expert, ascending within one, and the tiles over them: one per
     * (expert, part, column block, row group). Output column pp * fy + j is weight row
     * pp * fc + j of the expert's [N_e, K_e] plane. */
    std::vector<int64_t> start((size_t)n_expert + 1, 0), order((size_t)T), arow((size_t)T);
    for (int64_t i = 0; i < T; ++i)
        if (ex[(size_t)i] >= 0) ++start[(size_t)ex[(size_t)i] + 1];
    for (int64_t e = 0; e < n_expert; ++e) start[(size_t)e + 1] += start[(size_t)e];
    {
        std::vector<int64_t> cur(start.begin(), start.end() - 1);
        for (int64_t i = 0; i < T; ++i) {
            const int64_t e = ex[(size_t)i];
            if (e < 0) continue;
            const int64_t at = cur[(size_t)e]++;
            order[(size_t)at] = i;
            arow[(size_t)at] = tk[(size_t)i];
        }
    }
    struct Item { int64_t e, pp, j0, r0, R; };
    std::vector<Item> items;
    int64_t rg_max = 1;
    for (int64_t e = 0; e < n_expert; ++e) {
        const int64_t cnt = start[(size_t)e + 1] - start[(size_t)e];
        if (cnt <= 0) continue;
        const int64_t fc = en[(size_t)e] / parts;
        const TileGrid tg = tile_grid(cnt, fc * parts);
        if (tg.rg > rg_max) rg_max = tg.rg;
        for (int64_t pp = 0; pp < parts; ++pp)
            for (int64_t j0 = 0; j0 < fc; j0 += REF_NB)
                for (int64_t r0 = 0; r0 < cnt; r0 += tg.rg)
                    items.push_back({ e, pp, j0, r0, (r0 + tg.rg < cnt) ? tg.rg : cnt - r0 });
    }
    std::vector<TileFn> tile((size_t)n_expert);
    for (int64_t e = 0; e < n_expert; ++e)
        tile[(size_t)e] = codebook ? pick_cb_tile(A->dtype)
                                   : pick_tile(A->dtype, we[(size_t)e].dtype);

    REF_PAR_REGION(T * N * K)
    {
        /* A codebook tile's scales are its REF_NB weight rows' groups, at most K of them. */
        const size_t wsf = tile_ws_floats(rg_max);
        float* const ws = ws_get(wsf + (codebook ? (size_t)(REF_NB * K) : 0));
        REF_FOR_DYN
        for (int64_t it = 0; it < (int64_t)items.size(); ++it) {
            const Item& w = items[(size_t)it];
            const int64_t e = w.e, fc = en[(size_t)e] / parts, ke = ek[(size_t)e];
            const int64_t nb = (w.j0 + REF_NB < fc) ? REF_NB : fc - w.j0;
            const int64_t base = start[(size_t)e] + w.r0;
            if (codebook) {
                /* The tile's weight rows as their own [nb, ke] operand: codes from its first row,
                 * and its scales, each times its level, as an f32 [nb, ngrp] stream. */
                const int64_t wr0 = w.pp * fc + w.j0;
                CbTensor lt = cbt[(size_t)e];
                lt.t.data = (uint8_t*)lt.t.data + wr0 * lt.row_bytes;
                const Mat B{ &lt.t, lt.cols, 1 };
                const CbScale& cs = cbs[(size_t)e];
                const int64_t g = cs.s.p && cs.s.group > 0 ? cs.s.group : ke;
                const int64_t ngrp = (ke + g - 1) / g;
                float* const sb = ws + wsf;
                for (int64_t j = 0; j < nb; ++j)
                    for (int64_t gi = 0; gi < ngrp; ++gi) sb[j * ngrp + gi] = cs.at(wr0 + j, gi);
                Scales SB{};
                SB.p = cs.s.p || cs.mul != 1.0f ? sb : nullptr;
                SB.dt = RAD_F32;
                SB.rs = ngrp;
                SB.cs = 1;
                SB.group = g;
                SB.rblk = 1;
                if (SA.p && SA.group < g) {
                    for (int64_t r = 0; r < w.R; ++r)
                        for (int64_t j = 0; j < nb; ++j)
                            stt(Y, order[(size_t)(base + r)] * y_rs + (w.pp * fy + w.j0 + j) * y_cs,
                                quant_dot<Dyn, LdCb>(ma, B, SA, SB, arow[(size_t)(base + r)], j,
                                                     ke));
                    continue;
                }
                const float* acc = tile[(size_t)e](ma, B, SA, SB, arow.data() + base, w.R, 0, nb,
                                                   ke, ws);
                for (int64_t r = 0; r < w.R; ++r) {
                    const int64_t i = order[(size_t)(base + r)];
                    for (int64_t j = 0; j < nb; ++j)
                        stt(Y, i * y_rs + (w.pp * fy + w.j0 + j) * y_cs, acc[r * REF_NB + j]);
                }
                continue;
            }
            const Mat&    B  = mw[(size_t)e];
            const Scales& SB = SW[(size_t)e];
            const int64_t g  = SB.group > 0 ? SB.group : ke;
            if (SA.p && SA.group < g) {
                /* An activation grid finer than the weight's: quant_dot per element. */
                for (int64_t r = 0; r < w.R; ++r)
                    for (int64_t j = 0; j < nb; ++j)
                        stt(Y, order[(size_t)(base + r)] * y_rs + (w.pp * fy + w.j0 + j) * y_cs,
                            quant_dot(ma, B, SA, SB, arow[(size_t)(base + r)],
                                      w.pp * fc + w.j0 + j, ke));
                continue;
            }
            const float* acc = tile[(size_t)e](ma, B, SA, SB, arow.data() + base, w.R,
                                               w.pp * fc + w.j0, nb, ke, ws);
            for (int64_t r = 0; r < w.R; ++r) {
                const int64_t i = order[(size_t)(base + r)];
                for (int64_t j = 0; j < nb; ++j)
                    stt(Y, i * y_rs + (w.pp * fy + w.j0 + j) * y_cs, acc[r * REF_NB + j]);
            }
        }
    }
    return RAD_OK;
}

int ref_moe_gemm(const RadArgs* a, RadStream)   { return moe_gemm_impl(a, false); }
int ref_moe_gemm_q(const RadArgs* a, RadStream s) {
    const char* dt = p_str(a, "dtype", "");
    if (dt && std::strcmp(dt, "w4a4") == 0) return ref_moe_gemm_w4a4(a, s);
    /* w4a8 over bf16 scales has its own body, summed in the order libr3d's kernel sums. Any other
     * four-bit expert (f16 scales, a code plane with a zero point) is a form that body does not
     * read, and the general one does. */
    if (dt && std::strcmp(dt, "w4a8") == 0) {
        const int rc = ref_moe_gemm_w4a8(a, s);
        if (rc != RAD_E_DTYPE) return rc;
    }
    return moe_gemm_impl(a, true);
}
int ref_moe_gemm_q_mod4(const RadArgs* a, RadStream) { return moe_gemm_impl(a, true, 4); }

/* ------------------------------------------------------------------ moe_gemm_w4a8 */
/* moe_gemm_q dtype w4a8: y[i, :] = a[sorted_tok[i] / top_k, :] @ w[e(i)]^T. The two's-complement
 * int4 experts (nibbles, -8..7) carry one bf16 scale per (row, 128 of K); the output is bf16 in
 * sorted order. The experts are two tables by parity (the even in `w` / `w_scale`, the odd in
 * `w_odd` / `w_odd_scale`, expert e at entry e/2 of class e%2). Two activation arms:
 *   * int8 a: each 128-block accumulates its raw product sum in i32 (exact), then folds
 *     (sub * a_scale) * w_scale in f32 in ascending block order -- the exact arithmetic and order
 *     the libr3d kernel runs, so the two agree bit for bit.
 *   * E4M3 a: the a and w codes decode exactly to f32 and the 128-block dot is summed in f32, the
 *     scales folded the same way; the kernel's f16 wave-matrix arm matches this to the f16 store
 *     ulp.
 * A row no expert owns (a dropped slot) is left as the caller set it. */
int ref_moe_gemm_w4a8(const RadArgs* a, RadStream) {
    if (!have(a, 9)) return RAD_E_INVAL;
    const RadTensor* A   = &a->t[0];
    const RadTensor* AS  = &a->t[1];
    const RadTensor* W   = &a->t[2];
    const RadTensor* WS  = &a->t[3];
    const RadTensor* ST  = &a->t[4];
    const RadTensor* OF  = &a->t[5];
    const RadTensor* WO  = &a->t[6];
    const RadTensor* WOS = &a->t[7];
    const RadTensor* Y   = &a->t[8];
    const bool e4m3 = (A->dtype == RAD_F8E4M3);
    if (A->dtype != RAD_I8 && !e4m3) return RAD_E_DTYPE;
    if (AS->dtype != RAD_F32 || W->dtype != RAD_I4 || WS->dtype != RAD_BF16 || Y->dtype != RAD_BF16)
        return RAD_E_DTYPE;
    if (e4m3 && (WO->dtype != RAD_I4 || WOS->dtype != RAD_BF16)) return RAD_E_DTYPE;

    const char* ord = p_str(a, "a_order", "token");
    const bool sorted_a = ord && std::strcmp(ord, "sorted") == 0;
    if (ord && !sorted_a && std::strcmp(ord, "token") != 0) return RAD_E_UNSUPPORTED;

    const int64_t n_expert = p_int(a, "n_expert", W->rank >= 3 ? W->shape[0] * 2 : 0);
    const int64_t top_k    = p_int(a, "top_k", 1);
    if (n_expert <= 0 || top_k <= 0) return RAD_E_SHAPE;
    const int64_t N = W->rank >= 3 ? W->shape[1] : 0;
    const int64_t K = W->rank >= 3 ? W->shape[2] : 0;
    const int64_t T = Y->rank >= 2 ? Y->shape[0] : 0;
    if (N <= 0 || K <= 0 || T <= 0 || K % 128) return RAD_E_SHAPE;
    if (p_int(a, "N", N) != N || p_int(a, "K", K) != K) return RAD_E_SHAPE;

    const int64_t n_even = (n_expert + 1) / 2, n_odd = n_expert / 2;
    if (W->rank < 3 || W->shape[0] < n_even || W->shape[1] < N || W->shape[2] < K) return RAD_E_SHAPE;
    if (WS->rank < 3 || WS->shape[0] < n_even || WS->shape[1] < N || WS->shape[2] < K / 128)
        return RAD_E_SHAPE;
    if (n_odd > 0) {
        if (WO->rank < 3 || WO->shape[0] < n_odd || WO->shape[1] < N || WO->shape[2] < K)
            return RAD_E_SHAPE;
        if (WOS->rank < 3 || WOS->shape[0] < n_odd || WOS->shape[1] < N || WOS->shape[2] < K / 128)
            return RAD_E_SHAPE;
    }
    const int64_t M = A->rank >= 2 ? A->shape[0] : 0;
    const int64_t ar = sorted_a ? T : M;
    if (M <= 0 || numel(AS) < ar * (K / 128) || numel(ST) < T || numel(OF) < n_expert + 1)
        return RAD_E_SHAPE;

    const int64_t a_ld  = A->rank >= 2 ? A->stride[0] : K;
    const int64_t y_ld  = Y->rank >= 2 ? Y->stride[0] : N;
    const int64_t w_ld  = W->stride[1];                 // nibbles per weight row
    const int64_t ws_ld = WS->stride[1];                // scales per weight row
    const int8_t*  ap   = (const int8_t*)A->data;
    const uint8_t* ap8  = (const uint8_t*)A->data;
    const float*   asp  = (const float*)AS->data;
    const int32_t* st   = (const int32_t*)ST->data;
    const int32_t* of   = (const int32_t*)OF->data;
    const uint8_t* wp   = (const uint8_t*)W->data;
    const uint16_t* wsp = (const uint16_t*)WS->data;
    const uint8_t* wop  = (const uint8_t*)WO->data;
    const uint16_t* wosp = (const uint16_t*)WOS->data;
    uint16_t* yp = (uint16_t*)Y->data;

    for (int64_t i = 0; i < T; ++i) {
        const int32_t f = st[i];
        const int64_t src = sorted_a ? i : (f >= 0 ? (int64_t)f / top_k : -1);
        if (src < 0 || (!sorted_a && (int64_t)f / top_k >= M)) continue;  // a dropped slot
        int64_t e = 0;
        while (e + 1 < n_expert && of[e + 1] <= i) ++e;
        if (i < of[e] || i >= of[e + 1]) continue;        // no expert owns this row
        const int64_t cls = e & 1, e2 = e >> 1;
        const uint8_t*  wt  = cls ? wop  : wp;
        const uint16_t* wst = cls ? wosp : wsp;
        const int64_t wrow0 = (size_t)e2 * (size_t)N * w_ld / 2;
        const int64_t wsrow0 = (size_t)e2 * (size_t)N * ws_ld;
        for (int64_t n = 0; n < N; ++n) {
            const uint8_t*  wrow = wt  + wrow0 + (size_t)n * w_ld / 2;
            const uint16_t* wsrow = wst + wsrow0 + (size_t)n * ws_ld;
            const float* asr = asp + src * (K / 128);
            float acc = 0.f;
            for (int64_t b = 0; b < K / 128; ++b) {
                const int64_t k0 = b * 128;
                float term;
                float wscale;
                const uint32_t ws_bits = (uint32_t)wsrow[b] << 16;   /* bf16 -> f32: align to high 16 bits */
                std::memcpy(&wscale, &ws_bits, 4);
                if (e4m3) {
                    float dot = 0.f;
                    for (int64_t k = 0; k < 128; ++k)
                        dot += fp8e4m3_to_f32(ap8[(size_t)src * a_ld + k0 + k]) * w4_code(wrow, k0 + k);
                    term = dot * asr[b] * wscale;
                } else {
                    int32_t sub = 0;
                    for (int64_t k = 0; k < 128; ++k)
                        sub += (int32_t)ap[(size_t)src * a_ld + k0 + k] * (int32_t)w4_code(wrow, k0 + k);
                    term = (float)sub * asr[b] * wscale;
                }
                acc += term;
            }
            yp[i * y_ld + n] = f32_to_bf16(acc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ moe_gemm_w4a4 */
/* moe_gemm_q dtype w4a4: y[i, :] = a[sorted_tok[i] / top_k, :] @ w[e(i)]^T. Both the activation and
 * the experts are two's-complement int4 (nibbles, -8..7) packed two per byte; the activation carries
 * one f32 scale per (row, 128 of K), each expert one bf16 scale per (row, 128 of K); the output is
 * bf16 in sorted order. The experts are two tables by parity (the even in `w` / `w_scale`, the odd in
 * `w_odd` / `w_odd_scale`, expert e at entry e/2 of class e%2). Each 128-block accumulates its raw
 * product sum in i32 (exact: both codes widen to int8), then folds (sub * a_scale) * w_scale in f32
 * in ascending block order -- the exact arithmetic and order the libr3d kernel runs, so the two agree
 * bit for bit. A row no expert owns (a dropped slot) is left as the caller set it. */
int ref_moe_gemm_w4a4(const RadArgs* a, RadStream) {
    if (!have(a, 9)) return RAD_E_INVAL;
    const RadTensor* A   = &a->t[0];
    const RadTensor* AS  = &a->t[1];
    const RadTensor* W   = &a->t[2];
    const RadTensor* WS  = &a->t[3];
    const RadTensor* ST  = &a->t[4];
    const RadTensor* OF  = &a->t[5];
    const RadTensor* WO  = &a->t[6];
    const RadTensor* WOS = &a->t[7];
    const RadTensor* Y   = &a->t[8];
    if (A->dtype != RAD_I4) return RAD_E_DTYPE;
    if (AS->dtype != RAD_F32 || W->dtype != RAD_I4 || WS->dtype != RAD_BF16 || Y->dtype != RAD_BF16)
        return RAD_E_DTYPE;
    if (WO->dtype != RAD_I4 || WOS->dtype != RAD_BF16) return RAD_E_DTYPE;

    const char* ord = p_str(a, "a_order", "token");
    const bool sorted_a = ord && std::strcmp(ord, "sorted") == 0;
    if (ord && !sorted_a && std::strcmp(ord, "token") != 0) return RAD_E_UNSUPPORTED;

    const int64_t n_expert = p_int(a, "n_expert", W->rank >= 3 ? W->shape[0] * 2 : 0);
    const int64_t top_k    = p_int(a, "top_k", 1);
    if (n_expert <= 0 || top_k <= 0) return RAD_E_SHAPE;
    const int64_t N = W->rank >= 3 ? W->shape[1] : 0;
    const int64_t K = W->rank >= 3 ? W->shape[2] : 0;
    const int64_t T = Y->rank >= 2 ? Y->shape[0] : 0;
    if (N <= 0 || K <= 0 || T <= 0 || K % 128) return RAD_E_SHAPE;
    if (p_int(a, "N", N) != N || p_int(a, "K", K) != K) return RAD_E_SHAPE;

    const int64_t n_even = (n_expert + 1) / 2, n_odd = n_expert / 2;
    if (W->rank < 3 || W->shape[0] < n_even || W->shape[1] < N || W->shape[2] < K) return RAD_E_SHAPE;
    if (WS->rank < 3 || WS->shape[0] < n_even || WS->shape[1] < N || WS->shape[2] < K / 128)
        return RAD_E_SHAPE;
    if (n_odd > 0) {
        if (WO->rank < 3 || WO->shape[0] < n_odd || WO->shape[1] < N || WO->shape[2] < K)
            return RAD_E_SHAPE;
        if (WOS->rank < 3 || WOS->shape[0] < n_odd || WOS->shape[1] < N || WOS->shape[2] < K / 128)
            return RAD_E_SHAPE;
    }
    const int64_t M = A->rank >= 2 ? A->shape[0] : 0;
    const int64_t ar = sorted_a ? T : M;
    if (M <= 0 || numel(AS) < ar * (K / 128) || numel(ST) < T || numel(OF) < n_expert + 1)
        return RAD_E_SHAPE;

    const int64_t a_ld  = A->rank >= 2 ? A->stride[0] / 2 : K / 2;   // BYTES per activation row (K int4 = K/2 bytes)
    const int64_t y_ld  = Y->rank >= 2 ? Y->stride[0] : N;
    const int64_t w_ld  = W->stride[1];                 // nibbles per weight row
    const int64_t ws_ld = WS->stride[1];                // scales per weight row
    const uint8_t* ap   = (const uint8_t*)A->data;
    const float*   asp  = (const float*)AS->data;
    const int32_t* st   = (const int32_t*)ST->data;
    const int32_t* of   = (const int32_t*)OF->data;
    const uint8_t* wp   = (const uint8_t*)W->data;
    const uint16_t* wsp = (const uint16_t*)WS->data;
    const uint8_t* wop  = (const uint8_t*)WO->data;
    const uint16_t* wosp = (const uint16_t*)WOS->data;
    uint16_t* yp = (uint16_t*)Y->data;

    for (int64_t i = 0; i < T; ++i) {
        const int32_t f = st[i];
        const int64_t src = sorted_a ? i : (f >= 0 ? (int64_t)f / top_k : -1);
        if (src < 0 || (!sorted_a && (int64_t)f / top_k >= M)) continue;  // a dropped slot
        int64_t e = 0;
        while (e + 1 < n_expert && of[e + 1] <= i) ++e;
        if (i < of[e] || i >= of[e + 1]) continue;        // no expert owns this row
        const int64_t cls = e & 1, e2 = e >> 1;
        const uint8_t*  wt  = cls ? wop  : wp;
        const uint16_t* wst = cls ? wosp : wsp;
        const int64_t wrow0 = (size_t)e2 * (size_t)N * w_ld / 2;
        const int64_t wsrow0 = (size_t)e2 * (size_t)N * ws_ld;
        for (int64_t n = 0; n < N; ++n) {
            const uint8_t*  wrow = wt  + wrow0 + (size_t)n * w_ld / 2;
            const uint16_t* wsrow = wst + wsrow0 + (size_t)n * ws_ld;
            const float* asr = asp + src * (K / 128);
            float acc = 0.f;
            for (int64_t b = 0; b < K / 128; ++b) {
                const int64_t k0 = b * 128;
                float wscale;
                const uint32_t ws_bits = (uint32_t)wsrow[b] << 16;   /* bf16 -> f32: high 16 bits */
                std::memcpy(&wscale, &ws_bits, 4);
                int32_t sub = 0;
                for (int64_t k = 0; k < 128; ++k)
                    sub += (int32_t)w4_code(ap + (size_t)src * a_ld, k0 + k)
                         * (int32_t)w4_code(wrow, k0 + k);
                acc += (float)sub * asr[b] * wscale;
            }
            yp[i * y_ld + n] = f32_to_bf16(acc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ moe_gather */
/* y[t] = sum over the token's slots of expert_w[t, j] * y_expert[position of (t, j)].
 *
 * Summation is in ascending slot j; walking the expert-sorted rows would sum in expert-id order and
 * differ in the last bits.
 *
 * The inverse permutation is rebuilt per block of tokens into a fixed stack array, one pass over
 * sorted_tok per block, so the op needs no scratch or allocation. */
int ref_moe_gather(const RadArgs* a, RadStream) {
    /* Operands are positional: an absent shared pair is two null slots, which `have` would
     * refuse. */
    if (a->n_t < 6) return RAD_E_INVAL;
    const RadTensor *ye = &a->t[0], *ew = &a->t[1], *sorted = &a->t[2], *Y = &a->t[5];
    if (!ye->data || !ew->data || !sorted->data || !Y->data) return RAD_E_INVAL;
    /* The shared arm, folded: y = bf16(shared * act(gate)) + bf16(gathered). A pair or neither. */
    const RadTensor *sh = t_in(a, 3), *sg = t_in(a, 4);
    if ((sh != nullptr) != (sg != nullptr)) return RAD_E_INVAL;
    const char* const act = p_str(a, "act", "none");
    const bool sig = act && !std::strcmp(act, "sigmoid");
    if (sh && act && std::strcmp(act, "sigmoid") && std::strcmp(act, "none")) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0);
    const int64_t top_k = p_int(a, "top_k", 1);
    if (n <= 0 || top_k <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(Y) / n;
    const int64_t T = M * top_k;
    if (M <= 0 || numel(sorted) < T || numel(ew) < T || numel(ye) < T * n) return RAD_E_SHAPE;

    enum { INV_MAX = 4096 };
    const int64_t tb = INV_MAX / top_k > 0 ? INV_MAX / top_k : 1;
    const int64_t nblk = (M + tb - 1) / tb;

    const int64_t ye_rs = ye->rank >= 2 ? ye->stride[ye->rank - 2] : n;
    const int64_t ye_cs = ye->rank >= 2 ? ye->stride[ye->rank - 1] : 1;
    const int64_t y_rs  = Y->rank >= 2 ? Y->stride[Y->rank - 2] : n;
    const int64_t y_cs  = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;

    REF_PAR_FOR(T * n)
    for (int64_t b = 0; b < nblk; ++b) {
        int32_t inv[INV_MAX];
        const int64_t t0 = b * tb, t1 = (t0 + tb < M) ? t0 + tb : M;
        const int64_t span = (t1 - t0) * top_k;
        for (int64_t i = 0; i < span; ++i) inv[i] = -1;
        for (int64_t i = 0; i < T; ++i) {
            const int64_t f = ld_int(sorted->dtype, sorted->data, offlin(sorted, i));
            if (f < t0 * top_k || f >= t1 * top_k) continue;
            inv[f - t0 * top_k] = (int32_t)i;
        }
        for (int64_t t = t0; t < t1; ++t) {
            float gate = 0.0f;
            if (sh) {
                const float v0 = ldt(sg, rowoff(sg, t, 1));
                gate = sig ? act_sigmoid(v0) : v0;
            }
            for (int64_t c = 0; c < n; ++c) {
                float acc = 0.0f;
                for (int64_t j = 0; j < top_k; ++j) {
                    const int32_t i = inv[(t - t0) * top_k + j];
                    if (i < 0) continue;
                    acc += ldt(ew, rowoff(ew, t, top_k) + j * laststride(ew)) *
                           ldt(ye, (int64_t)i * ye_rs + c * ye_cs);
                }
                const int64_t yo = t * y_rs + c * y_cs;
                /* The stores round as the unfused pair does: the gather's result and scale_rows'
                 * product are each rounded, then their sum. */
                stt(Y, yo, acc);
                if (!sh) continue;
                const float g = ldt(Y, yo);
                stt(Y, yo, ldt(sh, rowoff(sh, t, n) + c * laststride(sh)) * gate);
                stt(Y, yo, ldt(Y, yo) + g);
            }
        }
    }
    return RAD_OK;
}

/* One step of a selection sort over `n` candidates: the unused one with the largest value, ties to
 * the lower id, skipping pads (id < 0). Returns its index, or -1 when none is left. */
template <class Id, class Val>
static int64_t next_best(int64_t n, const Id& id, const Val& val,
                         const std::vector<unsigned char>& used) {
    int64_t best = -1;
    for (int64_t c = 0; c < n; ++c) {
        if (used[(size_t)c] || id(c) < 0) continue;
        if (best < 0 || val(c) > val(best) || (val(c) == val(best) && id(c) < id(best))) best = c;
    }
    return best;
}

/* ------------------------------------------------------------------ row_topk */
/* Exact per-row top-R, descending, ties to the lower column: the router's selection without the
 * softmax, for drafter candidate sets and speculative trees. */
int ref_row_topk(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *idx = &a->t[1], *val = &a->t[2];
    /* Vocab-parallel form: `vocab_off` shifts reported columns to global ids; `pairs` is the
     * [M, R, 2] plane an all_gather moves (int32 id bits in the first float slot, then the value).
     * Both absent is the single-rank top-R. */
    const RadTensor* pairs = t_in(a, 3);
    const int64_t off = p_int(a, "vocab_off", 0);
    const int64_t N = p_int(a, "N", x->rank >= 2 ? x->shape[x->rank - 1] : 0);
    const int64_t R = p_int(a, "R", 1);
    if (N <= 0 || R <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(x) / N;
    if (M <= 0 || numel(idx) < M * R || numel(val) < M * R) return RAD_E_SHAPE;
    if (pairs && (pairs->dtype != RAD_F32 || numel(pairs) < M * R * 2)) return RAD_E_SHAPE;

    REF_PAR_FOR(M * N * R)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t xb = rowoff(x, m, N), xs = laststride(x);
        float pv = INFINITY;
        int64_t pi = -1;
        for (int64_t r = 0; r < R; ++r) {
            const bool ok = topk_next(x, xb, xs, N, &pv, &pi);
            const int64_t id = ok ? pi + off : -1;
            st_int(idx->dtype, idx->data, rowoff(idx, m, R) + r * laststride(idx), id);
            stt(val, rowoff(val, m, R) + r * laststride(val), ok ? pv : -INFINITY);
            if (pairs) {
                float* const q = (float*)pairs->data + (m * R + r) * 2;
                const int32_t i32 = (int32_t)id;
                std::memcpy(q, &i32, sizeof i32);
                q[1] = ok ? pv : -INFINITY;
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ row_topk_merge */
/* The other half of the vocab-parallel top-R: the R largest of world_size gathered per-rank
 * candidate sets, in the single-rank path's descending (idx, val) form.
 *
 * Ties break on the lower id, matching a dense top-R over the whole vocabulary; rank order would
 * disagree when two shards hold the same logit.
 *
 * An unfilled slot carries id -1 and is dropped; -inf cannot be the sentinel, since masked logits
 * legitimately take it. */
int ref_row_topk_merge(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *g = &a->t[0], *idx = &a->t[1], *val = &a->t[2];
    const int64_t R = p_int(a, "R", 1);
    const int64_t W = p_int(a, "world_size", 1);
    if (R <= 0 || W <= 0) return RAD_E_SHAPE;
    /* Rows come from the operand; `M` is a band ceiling (see band_rows). */
    const int64_t M = numel(g) / (W * R * 2);
    if (M <= 0) return RAD_E_SHAPE;
    if (g->dtype != RAD_F32 || numel(g) < M * W * R * 2) return RAD_E_SHAPE;
    if (numel(idx) < M * R || numel(val) < M * R) return RAD_E_SHAPE;

    REF_PAR_FOR(M * W * R * R)
    for (int64_t m = 0; m < M; ++m) {
        const float* const row = (const float*)g->data + m * W * R * 2;
        auto id = [&](int64_t c) {
            int32_t v;
            std::memcpy(&v, row + c * 2, sizeof v);
            return v;
        };
        auto value = [&](int64_t c) { return row[c * 2 + 1]; };
        /* Selection sort over W*R entries (32 at world 2, R 16). */
        std::vector<unsigned char> used((size_t)(W * R), 0);
        for (int64_t r = 0; r < R; ++r) {
            const int64_t best = next_best(W * R, id, value, used);
            if (best >= 0) used[(size_t)best] = 1;
            st_int(idx->dtype, idx->data, rowoff(idx, m, R) + r * laststride(idx),
                   best >= 0 ? (int64_t)id(best) : -1);
            stt(val, rowoff(val, m, R) + r * laststride(val), best >= 0 ? value(best) : -INFINITY);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ pair_argmax */
/* The top-1 of a gathered pair plane as a proposal: one id per row at the output's pitch, so a block
 * drafter's per-position picks land contiguous in the token buffer it verifies. `pairs` holds `n`
 * (int32 id, f32 value) pairs per row, a negative id being a pad; largest value wins, ties to the
 * lower id, matching an argmax over the unsharded row.
 *
 * Not row_topk_merge with R = 1: a rank sending R = 1 sends 8 bytes, which a row-interleaved
 * all_gather of 16-byte words cannot carry, so ranks send R >= 2 and this takes the top-1 of all
 * `n`. A row with no valid pair writes -1 and -inf. `val` is optional. */
int ref_pair_argmax(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *p = &a->t[0], *idx = &a->t[1];
    const RadTensor* val = t_in(a, 2);
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    if (p->dtype != RAD_F32) return RAD_E_DTYPE;   /* the id is carried by bit pattern */
    /* Rows come from the operand; `M` is a band ceiling. */
    const int64_t M = numel(p) / (n * 2);
    if (M <= 0 || numel(idx) < M || (val && numel(val) < M)) return RAD_E_SHAPE;
    const float* const pf = (const float*)p->data;

    REF_PAR_FOR(M * n)
    for (int64_t m = 0; m < M; ++m) {
        int32_t bid = -1;
        float   bv  = -INFINITY;
        for (int64_t j = 0; j < n; ++j) {
            int32_t id;
            std::memcpy(&id, pf + offlin(p, (m * n + j) * 2), sizeof id);
            if (id < 0) continue;                          /* a pad, not a candidate */
            const float v = pf[offlin(p, (m * n + j) * 2 + 1)];
            if (bid < 0 || v > bv || (v == bv && id < bid)) { bid = id; bv = v; }
        }
        st_int(idx->dtype, idx->data, offlin(idx, m), bid);
        if (val) stt(val, offlin(val, m), bv);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ logit_rerank */
/* Rescore R candidates per row exactly against the full-precision head.
 *
 * A speculative draft head may be coarse (the target verifies every proposal, so a worse draft
 * costs only acceptance), but a heavily quantised head often disagrees with the exact one on the
 * top-1. Rescoring its top few candidates against the exact weight recovers most of that, reading
 * R rows of the head instead of all of it.
 *
 * `vocab_off` is the shard's first column; ids are global, as row_topk reports and row_topk_merge
 * consumes them. A candidate outside this rank's rows becomes a pad.
 *
 * Output order matches row_topk: descending value, ties to the lower id, so the downstream merge
 * resolves ties consistently. Pads sort last with id -1. */
int ref_logit_rerank(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *in = &a->t[2];
    const RadTensor *idx = &a->t[3], *val = &a->t[4];
    const RadTensor *pairs = (a->n_t > 5 && a->t[5].data) ? &a->t[5] : nullptr;
    const int64_t R    = p_int(a, "R", 1);
    const int64_t K    = p_int(a, "n_embd", 0);
    const int64_t NV   = p_int(a, "n_vocab", 0);
    const int64_t voff = p_int(a, "vocab_off", 0);
    if (R <= 0 || K <= 0 || NV <= 0) return RAD_E_SHAPE;
    /* Rows come from the operand; `M` is a band ceiling. */
    const int64_t M = numel(x) / K;
    if (M <= 0 || numel(in) < M * R) return RAD_E_SHAPE;
    if (numel(idx) < M * R || numel(val) < M * R) return RAD_E_SHAPE;
    /* A head of codes and scale (ref_planes.h) is read element by element as the core's decoder
     * gives it, so its scores are those of the decoded f32 head. */
    ScaledRows sr;
    const bool joined = is_joined(w);
    if (joined) RAD_REF_TRY(scaled_rows(w, NV, K, &sr));
    else if (numel(w) < NV * K) return RAD_E_SHAPE;
    /* The pair plane is f32 words, ids carried by bit pattern, as row_topk writes it. */
    if (pairs && (pairs->dtype != RAD_F32 || numel(pairs) < M * R * 2)) return RAD_E_SHAPE;
    const int64_t ws = laststride(w);

    REF_PAR_FOR(M * R * K)
    for (int64_t m = 0; m < M; ++m) {
        std::vector<int32_t> id((size_t)R);
        std::vector<float>   sc((size_t)R);
        for (int64_t r = 0; r < R; ++r) {
            const int64_t g = ld_int(in->dtype, in->data, rowoff(in, m, R) + r * laststride(in));
            const int64_t local = g - voff;
            if (g < 0 || local < 0 || local >= NV) {
                id[(size_t)r] = -1;
                sc[(size_t)r] = -INFINITY;
                continue;
            }
            double acc = 0.0;                       /* summed in double, in k order */
            const int64_t wb = joined ? 0 : rowoff(w, local, K);
            for (int64_t k = 0; k < K; ++k)
                acc += (double)ldt(x, rowoff(x, m, K) + k * laststride(x)) *
                       (double)(joined ? sr.at(local, k) : ldt(w, wb + k * ws));
            id[(size_t)r] = (int32_t)g;
            sc[(size_t)r] = (float)acc;
        }
        std::vector<unsigned char> used((size_t)R, 0);
        for (int64_t o = 0; o < R; ++o) {
            const int64_t best = next_best(R, [&](int64_t c) { return id[(size_t)c]; },
                                           [&](int64_t c) { return sc[(size_t)c]; }, used);
            const int32_t oid = best >= 0 ? id[(size_t)best] : -1;
            const float   ov  = best >= 0 ? sc[(size_t)best] : -INFINITY;
            if (best >= 0) used[(size_t)best] = 1;
            st_int(idx->dtype, idx->data, rowoff(idx, m, R) + o * laststride(idx), (int64_t)oid);
            stt(val, rowoff(val, m, R) + o * laststride(val), ov);
            if (pairs) {
                float* p = (float*)pairs->data + (m * R + o) * 2;
                std::memcpy(p, &oid, sizeof oid);
                p[1] = ov;
            }
        }
    }
    return RAD_OK;
}
