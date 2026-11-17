/* avx_moe.cpp -- routing, the scatter/gather pair, the grouped GEMM, and the exact row top-R.
 *
 * sorted_tok holds the flattened (token, slot) index, token * top_k + slot, grouped by expert, not
 * the bare token id, so `moe_gather` knows which routing weight applies to each row. This is vLLM's
 * `sorted_token_ids` convention. A slot whose expert id is outside [0, n_expert) is dropped: not
 * counted, and the tail from expert_offset[n_expert] to M*top_k is filled with -1 (so padded batch
 * rows are expressible).
 *
 * Three loops are restructured relative to libref, each with an identical result:
 *
 *   moe_scatter is O(T + n_expert) instead of O(T * n_expert): one pass with a per-expert write
 *   cursor instead of rescanning the ids per expert. Same stable order (ascending i within an
 *   expert).
 *
 *   moe_gather builds the inverse permutation once, O(T). libref rebuilds it per token block in a
 *   4096-entry stack array to avoid allocating, at (M / block) passes over sorted_tok.
 *
 *   moe_gemm runs every expert's rows as one batch of products on the GEMM's tile drivers
 *   (avx_gemm_tile.cpp), each (expert, part) a job; libref re-reads the expert's weight for every
 *   row. Where the drivers decline, it gathers activation rows into a panel and runs a dot over
 *   each weight row packed once per expert.
 *
 * moe_gather's summation order is unchanged: ascending slot j. Summing in expert-sorted order
 * instead would differ in the last bits.
 */
#include "avx_gemm.h"
#include "rad_planes.h"

#include <algorithm>
#include <cstdlib>
#include <type_traits>
#include <vector>

using namespace avx;

/* ================================================================== operand readings
 * Mat, Scales, make_scales and widen_cols are avx_gemm.h's (libref/ref_gemm.h's contract);
 * `row_stride` and `grid_block` come from avx_common.h. */

/* Expert e of a RAD_OPD_WTAB operand: `data` is a host array of shape[0] pointers, one per expert,
 * marked by stride[0] == 0. The stacked form, a dense rank-3 [n_expert, N, K], is also accepted
 * (what callers outside the engine, such as a checker, pass). */
static RadTensor expert_entry(const RadTensor* w, int64_t e) {
    RadTensor t{};
    if (!w || !w->data || w->rank < 2 || e < 0 || e >= w->shape[0]) return t;
    t.dtype = w->dtype;
    t.rank = w->rank - 1;
    for (uint32_t d = 0; d + 1 < w->rank; ++d) {
        t.shape[d] = w->shape[d + 1];
        t.stride[d] = w->stride[d + 1];
    }
    if (w->stride[0] == 0) {
        void* const* tab = (void* const*)w->data;
        t.data = tab[e];
    } else {
        /* Offset in bits, since a packed dtype's element offset can be mid-byte; a plane must
         * start on a byte. */
        const int64_t bits = rad_dtype_bits(w->dtype);
        const int64_t nbits = bits * w->stride[0] * e;
        if (bits <= 0 || (nbits & 7)) return t;      /* data stays null: unreadable */
        t.data = (void*)((const char*)w->data + (nbits >> 3));
    }
    return t;
}

static inline void pack_row(const Mat& m, int64_t r, int64_t K, float* dst) {
    widen_cols(m, r, 0, K, dst);
}

/* ------------------------------------------------------------------ router_topk */
/* Softmax over all n_expert logits, then the top_k by probability, then (if `norm`) renormalise
 * the kept weights to sum to one. Softmax before selection is the order Mixtral, Qwen3 and the
 * other supported models use; the order is part of the op.
 *
 * Selection compares logits, not probabilities: softmax is monotone, so the order is the same and
 * the comparison is exact. */
AVX_KERNEL(router_topk) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *ids = &a->t[1], *w = &a->t[2];
    const int64_t n_expert = p_int(a, "n_expert", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    const int64_t top_k = p_int(a, "top_k", 1);
    const int norm = (int)p_int(a, "norm", 0);
    if (n_expert <= 0 || top_k <= 0 || top_k > n_expert) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_expert;
    if (M <= 0 || numel(ids) < M * top_k || numel(w) < M * top_k) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M * n_expert >= 1024)
    for (int64_t m = 0; m < M; ++m) {
        float* buf = scratch_f32((size_t)n_expert);
        const float* v = in_row(lg, rowoff(lg, m, n_expert), n_expert, buf);
        const float mx = row_max(v, n_expert);
        float sum = 0.0f;
        for (int64_t e = 0; e < n_expert; ++e) sum += sc_exp(v[e] - mx);
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

        float pv = INFINITY;
        int64_t pi = -1, kept = 0;
        float wsum = 0.0f;
        for (int64_t r = 0; r < top_k; ++r) {
            if (!topk_next_f32(v, n_expert, &pv, &pi)) break;
            const float p = sc_exp(pv - mx) * inv;
            st_int_dt(ids->dtype, ids->data, rowoff(ids, m, top_k) + r * laststride(ids), pi);
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
/* `protect`: local experts the sort places last, in the listed order, with the rest closed up;
 * the layout for a layer that serves a few experts with a second grouped GEMM (libr4d's
 * sort_place). Returns false unless the list is strictly ascending within [0, n). */
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

/* Stable counting sort of the (token, slot) pairs by expert id: one placement pass with a
 * per-expert write cursor. Within an expert, rows are appended in ascending i, as in libref.
 *
 * Serial: O(T) over small integers. A stable parallel version needs per-thread histograms and a
 * prefix pass, which costs more than it saves. */
AVX_KERNEL(moe_scatter) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *ids = &a->t[0], *sorted = &a->t[1], *off = &a->t[2], *cnt = &a->t[3];
    const int64_t n_expert = p_int(a, "n_expert", 0);
    const int64_t top_k = p_int(a, "top_k", 1);
    /* This rank's first expert, subtracted from every routed id so the sort uses local indices
     * (as the weight table, offsets and GEMM do). */
    const int64_t base = p_int(a, "expert_base", 0);
    if (n_expert <= 0 || top_k <= 0 || base < 0) return RAD_E_SHAPE;
    const int64_t M = numel(ids) / top_k;
    const int64_t T = M * top_k;
    if (M <= 0 || numel(sorted) < T) return RAD_E_SHAPE;
    if (numel(off) < n_expert + 1 || numel(cnt) < n_expert) return RAD_E_SHAPE;
    std::vector<int64_t> place;
    if (!sort_places(p_str(a, "protect", nullptr), n_expert, &place)) return RAD_E_SHAPE;
    /* A slot's place in the sort, or -1 for an expert outside this call's slice. */
    auto at = [&](int64_t i) -> int64_t {
        const int64_t e = ldi(ids, offlin(ids, i)) - base;
        return (e < 0 || e >= n_expert) ? -1 : place[(size_t)e];
    };

    int64_t* cur = (int64_t*)scratch_raw((size_t)(n_expert + 1) * sizeof(int64_t));
    for (int64_t e = 0; e < n_expert; ++e) cur[e] = 0;

    for (int64_t i = 0; i < T; ++i) {
        const int64_t e = at(i);
        if (e < 0) continue;
        ++cur[e];
    }
    for (int64_t e = 0; e < n_expert; ++e)
        st_int_dt(cnt->dtype, cnt->data, offlin(cnt, e), cur[e]);
    int64_t run = 0;
    for (int64_t e = 0; e < n_expert; ++e) {
        st_int_dt(off->dtype, off->data, offlin(off, e), run);
        const int64_t c = cur[e];
        cur[e] = run;                /* now the write cursor */
        run += c;
    }
    st_int_dt(off->dtype, off->data, offlin(off, n_expert), run);

    for (int64_t i = run; i < T; ++i)
        st_int_dt(sorted->dtype, sorted->data, offlin(sorted, i), -1);
    for (int64_t i = 0; i < T; ++i) {
        const int64_t e = at(i);
        if (e < 0) continue;
        st_int_dt(sorted->dtype, sorted->data, offlin(sorted, cur[e]), i);
        ++cur[e];
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the codebook experts
 *
 * Qwen3.8-Flash-Next's served experts (docs/MOE-W4.md): u4 or u5 codes indexing a table of 16 or
 * 32 f32 levels, a scale per (row, group of K), bf16 per 128 (w4nla8h) or E4M3 per 64 or 32 times
 * one fixed f32 (w4nl64a8h, w4nl32a8h, w5nl64a8h). The codes view, and the two-level forms' scale
 * view, arrive in the joined form of abi/rad_planes.h, which the registry rows' layout hooks ask
 * for; the planes accepted and refused are libref's (libref/ref_moe.cpp).
 *
 * Arithmetic: each activation row is multiplied by its own scale as it is packed, and the weight's
 * scale is applied to the table (one register multiply a group) or, in AVX-512's line decode, to
 * each lane's partial sum once a line; the dot runs over all of K with one horizontal sum per
 * output. libref sums each weight group and scales the sum, `(sub * sa) * sb`; the two differ in
 * rounding order only.
 *
 * Decode: codes are expanded through the table in registers, never into an f32 copy of an expert.
 * u4 at AVX-512 takes a cache line (128 codes) at a time: see cb_fused_lines. Elsewhere a chunk of
 * kCbCH codes: sixteen bytes of u4 widen to sixteen dword lanes, and one vpermps each on the low
 * and high nibbles gives the chunk's even and odd elements (the 16-entry table is one register);
 * u5 places each code's two bytes in its lane with vpshufb, aligns it with a variable shift and
 * looks it up in the 32 entries with vpermt2ps. AVX2 does the same over 8-entry vpermps, two (u4)
 * or four (u5) of them blended on the index bits. The activation is packed in the matching order.
 * The scalar and AVX levels, and a geometry whose K or scale group is not whole chunks, decode a
 * weight row through the table into L1 and take the generic dot.
 *
 * Two forms over an expert's rows. With at most kCbFused rows (decode and short verify steps) each
 * weight row is decoded once and multiplied into every row's accumulators as it is decoded, the
 * next rows' codes prefetched a pass ahead. With more, kCbNR weight rows at a time are decoded over
 * all of K into a panel (L2-resident at Flash-Next's K), and a kCbMR x kCbNR register block
 * multiplies it into every row of a block of up to kCbRB, one horizontal sum per output.
 *
 * The activation rows a call reads are packed once (times their scales, in the decode's order)
 * and shared by every expert that reads them. Work is (expert, block of its rows, kCbNC or, for
 * the panel form, kCbNCP weight rows) items, taken dynamically. Each output is computed by one
 * item in an order fixed by the call's geometry, so the result does not depend on the thread
 * count. */

/* The panel form's register block: 3 x 8 accumulators in AVX-512's 32 registers, 2 x 4 in AVX2's
 * 16. Both divide kCbNC and the experts' widths. */
#if AVX_LEVEL == 3
constexpr int64_t kCbMR = 3, kCbNR = 8;
#else
constexpr int64_t kCbMR = 2, kCbNR = 4;
#endif
constexpr int64_t kCbFused = 4, kCbRB = 64, kCbNC = 64, kCbNCP = 256;
constexpr int64_t kCbCH = 2 * VF_N;           /* codes a decode step expands */

/* One expert's codes, table and scales, read from its joined form. */
struct CbExp {
    const uint8_t* codes = nullptr;
    int64_t row_bytes = 0, n = 0, k = 0;
    int     bits = 0;
    float   tab[32];
    Scales  s{};              /* the scale plane; `s.p` null for none */
    float   mul = 1.0f;       /* the level above the scale, 1 for none */
    int64_t group = 0;        /* K elements a scale covers */
    inline float sc(int64_t row, int64_t g) const {
        return (s.p ? s.grp(row, g) : 1.0f) * mul;
    }
    int64_t groups() const { return (k + group - 1) / group; }
};

/* Weight row `row`'s scales times their level into `dst`, the values sc() gives: one row
 * conversion where the scales are contiguous, instead of a dtype switch a group. */
static void cb_scales(const CbExp& x, int64_t row, float* dst) {
    const int64_t ng = x.groups();
    if (x.s.p && x.s.cs == 1) {
        const int64_t base = (x.s.rblk > 1 ? row / x.s.rblk : row) * x.s.rs;
        row_to_f32(x.s.dt, (const uint8_t*)x.s.p + base * (rad_dtype_bits(x.s.dt) / 8), dst, ng);
        if (x.mul != 1.0f)
            for (int64_t g = 0; g < ng; ++g) dst[g] *= x.mul;
        return;
    }
    for (int64_t g = 0; g < ng; ++g) dst[g] = x.sc(row, g);
}

/* The joined form's header when weight entry `t` holds one; an entry smaller than a header is a
 * plain plane and is not read past its end (libref's cb_joined). */
static const RadPlanesHeader* cb_joined(const RadTensor& t) {
    int64_t n = 1;
    for (uint32_t i = 0; i < t.rank; ++i) n *= t.shape[i];
    if (!t.data || rad_dtype_bytes(t.dtype, n) < (int64_t)sizeof(RadPlanesHeader)) return nullptr;
    return rad_planes_header(t.data);
}

/* An expert's codes from its joined form: 1, 0 for a plain plane (any dtype but u4 and u5),
 * RAD_E_DTYPE for planes these kernels do not read, RAD_E_SHAPE for codes not covering [n, k]. */
static int cb_codes(const RadTensor& w, int64_t n, int64_t k, CbExp* out) {
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
    out->codes = (const uint8_t*)codes;
    out->row_bytes = ec->row_bytes;
    out->n = n;
    out->k = k;
    out->bits = bits;
    std::memcpy(out->tab, tab, sizeof(float) << bits);
    return 1;
}

/* An expert's scales beside joined codes: the joined `scale,scale.1` form, else the plane as
 * make_scales reads it (w4nla8h's bf16 scale, or none). RAD_E_DTYPE for other joined planes. */
static int cb_scale(const RadTensor* ws, CbExp* x) {
    x->mul = 1.0f;
    const RadPlanesHeader* h = ws ? cb_joined(*ws) : nullptr;
    if (!h) {
        x->s = make_scales(ws, x->n, x->k, false);
        x->group = x->s.p && x->s.group > 0 ? x->s.group : x->k;
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
    if ((x->n + es->block[0] - 1) / es->block[0] > es->rows ||
        (x->k + es->block[1] - 1) / es->block[1] > es->cols)
        return RAD_E_SHAPE;
    x->s.p = sp;
    x->s.dt = dt;
    x->s.rs = es->cols;
    x->s.cs = 1;
    x->s.group = es->block[1];
    x->s.rblk = es->block[0];
    x->group = es->block[1];
    std::memcpy(&x->mul, lp, sizeof x->mul);
    return 1;
}

/* ---- the in-register decode: a chunk of kCbCH codes at `p` into two vectors, `v0` and `v1`.
 * For u4 those are the chunk's even and odd elements; for u5 its two halves in element order. The
 * table registers hold the table times the current group's scale. */
#if AVX_LEVEL == 3
struct CbDec {
    __m512 t0, t1;            /* entries 0..15 and 16..31 */
    __m512i shuf, shift;      /* u5: each lane's two source bytes, and its bit offset in them */
};
static inline CbDec cb_dec_init() {
    alignas(64) int8_t sh[64];
    alignas(64) int32_t sv[16];
    for (int j = 0; j < 16; ++j) {
        const int b = (5 * j) >> 3;
        sh[4 * j] = (int8_t)b;
        sh[4 * j + 1] = (int8_t)(b + 1);
        sh[4 * j + 2] = sh[4 * j + 3] = (int8_t)0x80;
        sv[j] = (5 * j) & 7;
    }
    CbDec d;
    d.t0 = d.t1 = _mm512_setzero_ps();
    d.shuf = _mm512_load_si512(sh);
    d.shift = _mm512_load_si512(sv);
    return d;
}
template <int BITS> static inline void cb_dec_scale(CbDec& d, const float* tab, float s) {
    const __m512 vs = _mm512_set1_ps(s);
    d.t0 = _mm512_mul_ps(_mm512_loadu_ps(tab), vs);
    if (BITS == 5) d.t1 = _mm512_mul_ps(_mm512_loadu_ps(tab + 16), vs);
}
template <int BITS> static inline void cb_dec(const uint8_t* p, const CbDec& d, vf& v0, vf& v1);
template <> inline void cb_dec<4>(const uint8_t* p, const CbDec& d, vf& v0, vf& v1) {
    const __m512i x = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)p));
    v0 = _mm512_permutexvar_ps(x, d.t0);
    v1 = _mm512_permutexvar_ps(_mm512_srli_epi32(x, 4), d.t0);
}
/* Reads 16 bytes at `p` and at `p + 10`: 6 past the chunk's 20. */
template <> inline void cb_dec<5>(const uint8_t* p, const CbDec& d, vf& v0, vf& v1) {
    const __m512i b0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)p));
    const __m512i b1 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)(p + 10)));
    v0 = _mm512_permutex2var_ps(d.t0, _mm512_srlv_epi32(_mm512_shuffle_epi8(b0, d.shuf), d.shift),
                                d.t1);
    v1 = _mm512_permutex2var_ps(d.t0, _mm512_srlv_epi32(_mm512_shuffle_epi8(b1, d.shuf), d.shift),
                                d.t1);
}
#elif AVX_LEVEL == 2
struct CbDec {
    __m256 t[4];              /* entries 0..7, 8..15, 16..23, 24..31 */
    __m256i shuf, shift;      /* u5, as at AVX-512, eight lanes over 5 bytes */
};
static inline CbDec cb_dec_init() {
    alignas(32) int8_t sh[32];
    alignas(32) int32_t sv[8];
    for (int j = 0; j < 8; ++j) {
        const int b = (5 * j) >> 3;
        sh[4 * j] = (int8_t)b;
        sh[4 * j + 1] = (int8_t)(b + 1);
        sh[4 * j + 2] = sh[4 * j + 3] = (int8_t)0x80;
        sv[j] = (5 * j) & 7;
    }
    CbDec d;
    d.t[0] = d.t[1] = d.t[2] = d.t[3] = _mm256_setzero_ps();
    d.shuf = _mm256_load_si256((const __m256i*)sh);
    d.shift = _mm256_load_si256((const __m256i*)sv);
    return d;
}
template <int BITS> static inline void cb_dec_scale(CbDec& d, const float* tab, float s) {
    const __m256 vs = _mm256_set1_ps(s);
    for (int i = 0; i < (BITS == 5 ? 4 : 2); ++i)
        d.t[i] = _mm256_mul_ps(_mm256_loadu_ps(tab + 8 * i), vs);
}
/* Entry x & 15 of the 16 in (a, b): vpermps reads the low three bits, bit 3 picks the half. */
static inline __m256 cb_lut16(__m256 a, __m256 b, __m256i x) {
    return _mm256_blendv_ps(_mm256_permutevar8x32_ps(a, x), _mm256_permutevar8x32_ps(b, x),
                            _mm256_castsi256_ps(_mm256_slli_epi32(x, 28)));
}
template <int BITS> static inline void cb_dec(const uint8_t* p, const CbDec& d, vf& v0, vf& v1);
template <> inline void cb_dec<4>(const uint8_t* p, const CbDec& d, vf& v0, vf& v1) {
    const __m256i x = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)p));
    v0 = cb_lut16(d.t[0], d.t[1], x);
    v1 = cb_lut16(d.t[0], d.t[1], _mm256_srli_epi32(x, 4));
}
static inline __m256 cb_u5_half(const uint8_t* p, const CbDec& d) {
    const __m256i b = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i*)p));
    const __m256i x = _mm256_srlv_epi32(_mm256_shuffle_epi8(b, d.shuf), d.shift);
    return _mm256_blendv_ps(cb_lut16(d.t[0], d.t[1], x), cb_lut16(d.t[2], d.t[3], x),
                            _mm256_castsi256_ps(_mm256_slli_epi32(x, 27)));
}
/* Reads 8 bytes at `p` and at `p + 5`: 3 past the chunk's 10. */
template <> inline void cb_dec<5>(const uint8_t* p, const CbDec& d, vf& v0, vf& v1) {
    v0 = cb_u5_half(p, d);
    v1 = cb_u5_half(p + 5, d);
}
#endif

#if AVX_LEVEL >= 2
/* Chunk `c` of a row of `nch`: its bytes, or for u5's last chunk, whose decode reads past it, a
 * copy in `bounce`. */
template <int BITS>
static inline const uint8_t* cb_chunk(const uint8_t* row, int64_t c, int64_t nch,
                                      uint8_t* bounce) {
    constexpr int64_t cb = kCbCH * BITS / 8;
    const uint8_t* q = row + c * cb;
    if (BITS == 5 && c + 1 == nch) {
        std::memcpy(bounce, q, (size_t)cb);
        return bounce;
    }
    return q;
}

/* Weight rows wr .. wr + W - 1 against R packed activation rows `ar`, decoded and multiplied in
 * one pass; out[w * R + r]. `sv` holds the W rows' scales (cb_scales), `ng` a row. Two weight rows
 * give the FMA chains of one-row decode enough independence to cover their latency. */
template <int BITS, int W, int R>
static void cb_fused(const CbExp& x, int64_t wr, const float* const* ar, const float* sv,
                     int64_t ng, float* out) {
    vf acc[W][R][2];
    for (int w = 0; w < W; ++w)
        for (int r = 0; r < R; ++r) acc[w][r][0] = acc[w][r][1] = vf_zero();
    CbDec d[W];
    for (int w = 0; w < W; ++w) d[w] = cb_dec_init();
    const int64_t nch = x.k / kCbCH, cpg = x.group / kCbCH;
    alignas(16) uint8_t bounce[W][32] = {};
    for (int64_t c = 0; c < nch;) {
        for (int w = 0; w < W; ++w) cb_dec_scale<BITS>(d[w], x.tab, sv[w * ng + c / cpg]);
        const int64_t ce = c + cpg < nch ? c + cpg : nch;
        for (; c < ce; ++c) {
            const int64_t a = c * kCbCH;
            constexpr int64_t cb = kCbCH * BITS / 8;
            /* The next W rows' line at this column, one pass ahead of the decode. */
            if (((c * cb) & 63) < cb)
                for (int w = 0; w < W; ++w)
                    _mm_prefetch((const char*)(x.codes + (wr + W + w) * x.row_bytes + c * cb),
                                 _MM_HINT_T0);
            for (int w = 0; w < W; ++w) {
                vf v0, v1;
                cb_dec<BITS>(cb_chunk<BITS>(x.codes + (wr + w) * x.row_bytes, c, nch, bounce[w]),
                             d[w], v0, v1);
                for (int r = 0; r < R; ++r) {
                    acc[w][r][0] = vf_fma(v0, vf_loadu(ar[r] + a), acc[w][r][0]);
                    acc[w][r][1] = vf_fma(v1, vf_loadu(ar[r] + a + VF_N), acc[w][r][1]);
                }
            }
        }
    }
    for (int w = 0; w < W; ++w)
        for (int r = 0; r < R; ++r) out[w * R + r] = vf_hsum(vf_add(acc[w][r][0], acc[w][r][1]));
}

/* Weight rows [wr, wr + nr), chunks [c0, c1), decoded into `P`, row i at P + i * ldp; `sv` holds
 * the rows' scales as for cb_fused. */
template <int BITS>
static void cb_panel(const CbExp& x, int64_t wr, int64_t nr, int64_t c0, int64_t c1,
                     const float* sv, int64_t ng, float* P, int64_t ldp) {
    const int64_t nch = x.k / kCbCH, cpg = x.group / kCbCH;
    alignas(16) uint8_t bounce[32] = {};
    CbDec d = cb_dec_init();
    for (int64_t i = 0; i < nr; ++i) {
        const uint8_t* row = x.codes + (wr + i) * x.row_bytes;
        float* dst = P + i * ldp;
        for (int64_t c = c0; c < c1;) {
            cb_dec_scale<BITS>(d, x.tab, sv[i * ng + c / cpg]);
            const int64_t ce = (c / cpg + 1) * cpg < c1 ? (c / cpg + 1) * cpg : c1;
            for (; c < ce; ++c) {
                vf v0, v1;
                cb_dec<BITS>(cb_chunk<BITS>(row, c, nch, bounce), d, v0, v1);
                vf_storeu(dst + (c - c0) * kCbCH, v0);
                vf_storeu(dst + (c - c0) * kCbCH + VF_N, v1);
            }
        }
    }
}
#endif

#if AVX_LEVEL == 3
/* ---- u4 at AVX-512 a cache line at a time: 128 codes, of which dword lane j holds codes 8j to
 * 8j + 7 as nibbles. A shift by 4m and one vpermps, which reads the low four bits of each lane,
 * give code 8j + m in lane j: eight vpermps a line and no widening. The activation is packed to
 * match (vector m of a line holds its elements 8j + m), and since a lane's eight codes share a
 * scale group, each lane's partial sum is multiplied by its group's scale once a line. */
constexpr int64_t kCbLine = 128;

/* Lane j's scale in line c of a row whose scales are `sv`: group (128c + 8j) / group. */
static inline __m512 cb_line_scale(const float* sv, int64_t c, int64_t group) {
    if (group % kCbLine == 0) return _mm512_set1_ps(sv[c * kCbLine / group]);
    if (group == 64)
        return _mm512_mask_mov_ps(_mm512_set1_ps(sv[2 * c]), 0xFF00, _mm512_set1_ps(sv[2 * c + 1]));
    __m512 v = _mm512_set1_ps(sv[4 * c]);
    v = _mm512_mask_mov_ps(v, 0x00F0, _mm512_set1_ps(sv[4 * c + 1]));
    v = _mm512_mask_mov_ps(v, 0x0F00, _mm512_set1_ps(sv[4 * c + 2]));
    return _mm512_mask_mov_ps(v, 0xF000, _mm512_set1_ps(sv[4 * c + 3]));
}

/* Whether expert `x` takes the line form: u4, whole lines, and a group whole lanes divide. */
static inline bool cb_lines(const CbExp& x) {
    return x.bits == 4 && x.k % kCbLine == 0 &&
           (x.group % kCbLine == 0 || x.group == 64 || x.group == 32);
}

/* cb_fused in lines: weight rows wr .. wr + W - 1 against R rows `ar`, out[w * R + r]. Two
 * partial chains a row cover the FMA latency with the decode's shuffles. */
template <int W, int R>
static void cb_fused_lines(const CbExp& x, int64_t wr, const float* const* ar, const float* sv,
                           int64_t ng, float* out) {
    __m512 acc[W][R];
    for (int w = 0; w < W; ++w)
        for (int r = 0; r < R; ++r) acc[w][r] = _mm512_setzero_ps();
    const __m512 T = _mm512_loadu_ps(x.tab);
    const int64_t nl = x.k / kCbLine;
    for (int64_t c = 0; c < nl; ++c) {
        const int64_t a = c * kCbLine;
        /* The next W rows' line at this column, one pass ahead of the decode. */
        for (int w = 0; w < W; ++w)
            _mm_prefetch((const char*)(x.codes + (wr + W + w) * x.row_bytes + c * 64), _MM_HINT_T0);
        auto act = [&](int r, int m) { return _mm512_loadu_ps(ar[r] + a + 16 * m); };
        for (int w = 0; w < W; ++w) {
            const __m512i q = _mm512_loadu_si512(x.codes + (wr + w) * x.row_bytes + c * 64);
            __m512 s0[R], s1[R];
            {
                const __m512 v0 = _mm512_permutexvar_ps(q, T);
                const __m512 v1 = _mm512_permutexvar_ps(_mm512_srli_epi32(q, 4), T);
                for (int r = 0; r < R; ++r) {
                    s0[r] = _mm512_mul_ps(v0, act(r, 0));
                    s1[r] = _mm512_mul_ps(v1, act(r, 1));
                }
            }
            for (int m = 2; m < 8; m += 2) {
                const __m512 v0 = _mm512_permutexvar_ps(_mm512_srli_epi32(q, 4 * m), T);
                const __m512 v1 = _mm512_permutexvar_ps(_mm512_srli_epi32(q, 4 * m + 4), T);
                for (int r = 0; r < R; ++r) {
                    s0[r] = _mm512_fmadd_ps(v0, act(r, m), s0[r]);
                    s1[r] = _mm512_fmadd_ps(v1, act(r, m + 1), s1[r]);
                }
            }
            const __m512 sc = cb_line_scale(sv + w * ng, c, x.group);
            for (int r = 0; r < R; ++r)
                acc[w][r] = _mm512_fmadd_ps(_mm512_add_ps(s0[r], s1[r]), sc, acc[w][r]);
        }
    }
    for (int w = 0; w < W; ++w)
        for (int r = 0; r < R; ++r) out[w * R + r] = _mm512_reduce_add_ps(acc[w][r]);
}

/* cb_panel in lines: weight rows [wr, wr + nr) over all of K into `P`, row i at P + i * ldp, each
 * lane times its scale. */
static void cb_panel_lines(const CbExp& x, int64_t wr, int64_t nr, const float* sv, int64_t ng,
                           float* P, int64_t ldp) {
    const __m512 T = _mm512_loadu_ps(x.tab);
    const int64_t nl = x.k / kCbLine;
    for (int64_t i = 0; i < nr; ++i) {
        const uint8_t* row = x.codes + (wr + i) * x.row_bytes;
        float* dst = P + i * ldp;
        for (int64_t c = 0; c < nl; ++c) {
            const __m512i q = _mm512_loadu_si512(row + c * 64);
            const __m512 sc = cb_line_scale(sv + i * ng, c, x.group);
            for (int m = 0; m < 8; ++m)
                _mm512_storeu_ps(dst + c * kCbLine + 16 * m,
                                 _mm512_mul_ps(_mm512_permutexvar_ps(
                                                   _mm512_srli_epi32(q, 4 * m), T), sc));
        }
    }
}
#endif

/* MB x NB dot products over [0, kl), kl a multiple of VF_N, added into `out`. */
template <int MB, int NB>
static inline void cb_block(const float* const* A, const float* B, int64_t ldb, int64_t kl,
                            float out[MB][NB]) {
    vf acc[MB][NB];
    for (int i = 0; i < MB; ++i)
        for (int j = 0; j < NB; ++j) acc[i][j] = vf_zero();
    for (int64_t k = 0; k < kl; k += VF_N) {
        vf av[MB];
        for (int i = 0; i < MB; ++i) av[i] = vf_loadu(A[i] + k);
        for (int j = 0; j < NB; ++j) {
            const vf bv = vf_loadu(B + j * ldb + k);
            for (int i = 0; i < MB; ++i) acc[i][j] = vf_fma(av[i], bv, acc[i][j]);
        }
    }
    for (int i = 0; i < MB; ++i)
        for (int j = 0; j < NB; ++j) out[i][j] += vf_hsum(acc[i][j]);
}

/* A call's operands and geometry, as moe_gemm_impl read them. */
struct CbCall {
    const RadTensor *A = nullptr, *sorted = nullptr, *off = nullptr, *Y = nullptr;
    const RadTensor* Wq[4] = {};
    const RadTensor* wsq[4] = {};
    int64_t Nq[4] = {}, Kq[4] = {};
    int     nc = 2;
    int64_t parts = 1, n_expert = 0, top_k = 1, N = 0, K = 0, T = 0, AR = 0, y_rs = 0, y_cs = 1;
    bool    sorted_a = false;
    Mat     ma{};
    Scales  SA{};
};

/* Live rows [r0, r1) of the list (one expert's) against its weight rows [n0, n1). */
struct CbItem { int64_t e, r0, r1, n0, n1; };

/* How a call decodes, the same for all its experts: a weight row through the table into L1, the
 * register decode in chunks of kCbCH, or AVX-512's u4 lines. */
enum CbPath { kCbGeneric = 0, kCbChunks = 1, kCbLines = 2 };
/* Row `tok` of the activation as f32 times its own scale, `k` elements in the order the call's
 * decode produces the weights' (`path`, and `bits` for the chunk decode). `tmp` holds k floats. */
static void cb_pack(const CbCall& c, int64_t tok, int64_t k, int bits, float* dst, float* tmp,
                    int path) {
    pack_row(c.ma, tok, k, tmp);
    if (c.SA.p)
        for (int64_t g0 = 0; g0 < k; g0 += c.SA.group) {
            const int64_t g1 = g0 + c.SA.group < k ? g0 + c.SA.group : k;
            const float s = c.SA.grp(tok, g0 / c.SA.group);
            const vf vs = vf_set1(s);
            int64_t i = g0;
            for (; i + VF_N <= g1; i += VF_N) vf_storeu(tmp + i, vf_mul(vf_loadu(tmp + i), vs));
            for (; i < g1; ++i) tmp[i] *= s;
        }
    if (path == kCbLines) {
        /* Vector m of each 128-element line holds elements 8j + m. */
        for (int64_t c0 = 0; c0 < k; c0 += 128)
            for (int64_t m = 0; m < 8; ++m)
                for (int64_t j = 0; j < 16; ++j) dst[c0 + 16 * m + j] = tmp[c0 + 8 * j + m];
    } else if (path == kCbChunks && bits == 4) {
        for (int64_t c0 = 0; c0 < k; c0 += kCbCH)
            for (int64_t j = 0; j < VF_N; ++j) {
                dst[c0 + j] = tmp[c0 + 2 * j];
                dst[c0 + VF_N + j] = tmp[c0 + 2 * j + 1];
            }
    } else {
        std::memcpy(dst, tmp, (size_t)k * sizeof(float));
    }
}

/* The scales of weight rows [n0, n1) into `sv`, ng a row: one conversion when the plane's rows are
 * contiguous, as a joined form's are. */
static void cb_scale_rows(const CbExp& x, int64_t n0, int64_t n1, float* sv) {
    const int64_t ng = x.groups();
    if (x.s.p && x.s.cs == 1 && x.s.rblk == 1 && x.s.rs == ng) {
        row_to_f32(x.s.dt, (const uint8_t*)x.s.p + n0 * ng * (rad_dtype_bits(x.s.dt) / 8), sv,
                   (n1 - n0) * ng);
        if (x.mul != 1.0f)
            for (int64_t i = 0; i < (n1 - n0) * ng; ++i) sv[i] *= x.mul;
        return;
    }
    for (int64_t wr = n0; wr < n1; ++wr) cb_scales(x, wr, sv + (wr - n0) * ng);
}

/* An item's products, store(r, wr, value) for its rows and weight rows: the generic decode, the
 * fused form or the panel form. `ar` points at the packed rows, `tmp` holds kCbNR rows of K floats
 * and `sv` the item's weight rows' scales. */
template <class Store>
static void cb_compute(const CbExp& x, const CbItem& it, int64_t R, const float* const* ar,
                       float* tmp, const float* sv, int64_t ng, const Store& store, int path) {
    const int64_t k = x.k;
    auto rs = [&](int64_t wr) { return sv + (wr - it.n0) * ng; };
    if (path == kCbGeneric) {
        /* A weight row through the table into L1, scaled per group, then the generic dot. */
        for (int64_t wr = it.n0; wr < it.n1; ++wr) {
            const uint8_t* row = x.codes + wr * x.row_bytes;
            for (int64_t g0 = 0; g0 < k; g0 += x.group) {
                const float s = rs(wr)[g0 / x.group];
                const int64_t g1 = g0 + x.group < k ? g0 + x.group : k;
                for (int64_t i = g0; i < g1; ++i) tmp[i] = x.tab[rad_load_bits(row, x.bits, i)] * s;
            }
            for (int64_t r = 0; r < R; ++r) store(r, wr, dotf(ar[r], tmp, k));
        }
        return;
    }
#if AVX_LEVEL == 3
    if (path == kCbLines && R <= kCbFused) {
        /* Four weight rows at a time for one row, two for more: eight or more partial chains. */
        float o[4 * kCbFused];
        auto run = [&](auto w, auto rr) {
            constexpr int Wn = decltype(w)::value, Rn = decltype(rr)::value;
            int64_t wr = it.n0;
            for (; wr + Wn <= it.n1; wr += Wn) {
                cb_fused_lines<Wn, Rn>(x, wr, ar, rs(wr), ng, o);
                for (int ww = 0; ww < Wn; ++ww)
                    for (int r = 0; r < Rn; ++r) store(r, wr + ww, o[ww * Rn + r]);
            }
            for (; wr < it.n1; ++wr) {
                cb_fused_lines<1, Rn>(x, wr, ar, rs(wr), ng, o);
                for (int r = 0; r < Rn; ++r) store(r, wr, o[r]);
            }
        };
        using W4 = std::integral_constant<int, 4>;
        using W2 = std::integral_constant<int, 2>;
        switch (R) {
            case 1:  run(W4{}, std::integral_constant<int, 1>{}); break;
            case 2:  run(W2{}, std::integral_constant<int, 2>{}); break;
            case 3:  run(W2{}, std::integral_constant<int, 3>{}); break;
            default: run(W2{}, std::integral_constant<int, 4>{}); break;
        }
        return;
    }
#endif
#if AVX_LEVEL >= 2
    if (R <= kCbFused) {
        float o[2 * kCbFused];
        auto run = [&](auto bits, auto w, auto rr) {
            constexpr int B = decltype(bits)::value, Wn = decltype(w)::value;
            constexpr int Rn = decltype(rr)::value;
            int64_t wr = it.n0;
            for (; wr + Wn <= it.n1; wr += Wn) {
                cb_fused<B, Wn, Rn>(x, wr, ar, rs(wr), ng, o);
                for (int ww = 0; ww < Wn; ++ww)
                    for (int r = 0; r < Rn; ++r) store(r, wr + ww, o[ww * Rn + r]);
            }
            for (; wr < it.n1; ++wr) {
                cb_fused<B, 1, Rn>(x, wr, ar, rs(wr), ng, o);
                for (int r = 0; r < Rn; ++r) store(r, wr, o[r]);
            }
        };
        using I4 = std::integral_constant<int, 4>;
        using I5 = std::integral_constant<int, 5>;
        /* AVX2's sixteen registers hold two weight rows' accumulators for up to two rows. */
        using W2 = std::integral_constant<int, 2>;
        using WB = std::integral_constant<int, AVX_LEVEL == 3 ? 2 : 1>;
        auto by_r = [&](auto bits) {
            switch (R) {
                case 1:  run(bits, W2{}, std::integral_constant<int, 1>{}); break;
                case 2:  run(bits, W2{}, std::integral_constant<int, 2>{}); break;
                case 3:  run(bits, WB{}, std::integral_constant<int, 3>{}); break;
                default: run(bits, WB{}, std::integral_constant<int, 4>{}); break;
            }
        };
        if (x.bits == 5) by_r(I5{});
        else             by_r(I4{});
        return;
    }
    /* The panel form: kCbNR weight rows over all of K, against every row in kCbMR blocks, so each
     * output takes one horizontal sum. */
    float* const P = tmp;
    const int64_t nch = k / kCbCH;
    for (int64_t wr0 = it.n0; wr0 < it.n1; wr0 += kCbNR) {
        const int64_t nr = it.n1 - wr0 < kCbNR ? it.n1 - wr0 : kCbNR;
#if AVX_LEVEL == 3
        if (path == kCbLines)  cb_panel_lines(x, wr0, nr, rs(wr0), ng, P, k);
        else
#endif
        if (x.bits == 5) cb_panel<5>(x, wr0, nr, 0, nch, rs(wr0), ng, P, k);
        else             cb_panel<4>(x, wr0, nr, 0, nch, rs(wr0), ng, P, k);
        int64_t r0 = 0;
        for (; r0 + kCbMR <= R; r0 += kCbMR) {
            if (nr == kCbNR) {
                float o[kCbMR][kCbNR] = {};
                cb_block<kCbMR, kCbNR>(ar + r0, P, k, k, o);
                for (int64_t i = 0; i < kCbMR; ++i)
                    for (int64_t j = 0; j < kCbNR; ++j) store(r0 + i, wr0 + j, o[i][j]);
                continue;
            }
            for (int64_t j = 0; j < nr; ++j) {
                float o[kCbMR][1] = {};
                cb_block<kCbMR, 1>(ar + r0, P + j * k, k, k, o);
                for (int64_t i = 0; i < kCbMR; ++i) store(r0 + i, wr0 + j, o[i][0]);
            }
        }
        for (; r0 < R; ++r0) {
            if (nr == kCbNR) {
                float o[1][kCbNR] = {};
                cb_block<1, kCbNR>(ar + r0, P, k, k, o);
                for (int64_t j = 0; j < kCbNR; ++j) store(r0, wr0 + j, o[0][j]);
                continue;
            }
            for (int64_t j = 0; j < nr; ++j) {
                float o[1][1] = { { 0.0f } };
                cb_block<1, 1>(ar + r0, P + j * k, k, k, o);
                store(r0, wr0 + j, o[0][0]);
            }
        }
    }
#else
    (void)R; (void)ar; (void)tmp; (void)sv; (void)ng; (void)store;
#endif
}

/* One item: its products and the outputs written. `apk` holds the call's packed activation rows,
 * `kp` floats apart. Scratch: a decoded weight row or the panel (_b), the outputs and the weight
 * rows' scales (_c). */
static void cb_item(const CbCall& c, const CbExp* ex, const int64_t* lr, const int64_t* atok,
                    const float* apk, int64_t kp, const CbItem& it, int path) {
    const CbExp& x = ex[it.e];
    const int64_t R = it.r1 - it.r0, k = x.k, nw = it.n1 - it.n0;
    const float* ar[kCbRB];
    for (int64_t r = 0; r < R; ++r) ar[r] = apk + atok[lr[it.r0 + r]] * kp;
    float* tmp = scratch_f32_b((size_t)(kCbNR * k));
    const int64_t ng = x.groups();
    float* const ob = scratch_f32_c((size_t)(R * nw + nw * ng));
    float* const sv = ob + R * nw;
    cb_scale_rows(x, it.n0, it.n1, sv);
    auto store = [&](int64_t r, int64_t wr, float v) { ob[r * nw + wr - it.n0] = v; };
    cb_compute(x, it, R, ar, tmp, sv, ng, store, path);

    /* Weight row wr is output column (wr / fc) * fy + wr % fc, so each part's rows are one run of
     * columns, converted at once where the output is dense. */
    const int64_t fc = x.n / c.parts, fy = c.N / c.parts;
    const int ybits = rad_dtype_bits(c.Y->dtype);
    const bool direct = c.y_cs == 1 && ybits % 8 == 0;
    for (int64_t r = 0; r < R; ++r) {
        const int64_t yr = lr[it.r0 + r] * c.y_rs;
        for (int64_t w0 = it.n0; w0 < it.n1;) {
            const int64_t w1 = std::min(it.n1, (w0 / fc + 1) * fc);
            const int64_t col = (w0 / fc) * fy + w0 % fc;
            const float* v = ob + r * nw + (w0 - it.n0);
            if (direct)
                row_from_f32(c.Y->dtype, v, (uint8_t*)c.Y->data + (yr + col) * (ybits / 8),
                             w1 - w0);
            else
                for (int64_t i = 0; i < w1 - w0; ++i) stt(c.Y, yr + (col + i) * c.y_cs, v[i]);
            w0 = w1;
        }
    }
}

static int moe_codebook(const CbCall& c) {
    const int64_t T = c.T, ne = c.n_expert, nc = c.nc, K = c.K, AR = c.AR;
    /* Upper bound on the items: each expert's row blocks, the last one partial, by its weight
     * blocks. */
    const int64_t nblk = (c.N + kCbNC - 1) / kCbNC;
    const int64_t max_items = (T / kCbRB + ne + 1) * nblk;
    auto pad64 = [](size_t x) { return (x + 63) & ~(size_t)63; };
    const size_t o_ex = 0;
    const size_t o_tok = pad64(o_ex + (size_t)ne * sizeof(CbExp));
    const size_t o_exp = pad64(o_tok + (size_t)T * sizeof(int64_t));
    const size_t o_lr = pad64(o_exp + (size_t)T * sizeof(int64_t));
    const size_t o_it = pad64(o_lr + (size_t)T * sizeof(int64_t));
    const size_t o_use = pad64(o_it + (size_t)max_items * sizeof(CbItem));
    const size_t o_apk = pad64(o_use + (size_t)AR);
    uint8_t* base = (uint8_t*)scratch_raw(o_apk + (size_t)(AR * K) * sizeof(float) + 64);
    CbExp*   ex = (CbExp*)(base + o_ex);
    int64_t* atok = (int64_t*)(base + o_tok);
    int64_t* aexp = (int64_t*)(base + o_exp);
    int64_t* lr = (int64_t*)(base + o_lr);
    CbItem*  items = (CbItem*)(base + o_it);
    uint8_t* used = base + o_use;
    float*   apk = (float*)(base + o_apk);

    /* Every expert, in libref's order of checks; refused before any parallel region. And the
     * fastest decode every expert admits; the chunk decode packs u4 and u5 rows differently, so
     * a call mixing them takes the generic one. */
    int path = AVX_LEVEL == 3 ? kCbLines : AVX_LEVEL == 2 ? kCbChunks : kCbGeneric;
    for (int64_t e = 0; e < ne; ++e) {
        const int q = (int)(e % nc);
        const int64_t ix = e / nc, n = c.Nq[q], k = c.Kq[q];
        const RadTensor w = expert_entry(c.Wq[q], ix);
        if (!w.data) return RAD_E_STRIDE;
        if (w.rank < 2 || w.shape[0] < n || w.shape[1] < k) return RAD_E_SHAPE;
        RadTensor s{};
        if (c.wsq[q]) {
            s = expert_entry(c.wsq[q], ix);
            if (!s.data) return RAD_E_STRIDE;
        }
        CbExp& x = ex[e];
        x = CbExp{};
        const int jc = cb_codes(w, n, k, &x);
        if (jc < 0) return jc;
        if (jc == 0) return RAD_E_DTYPE;
        const int js = cb_scale(c.wsq[q] ? &s : nullptr, &x);
        if (js < 0) return js;
#if AVX_LEVEL == 3
        if (path == kCbLines && !cb_lines(x)) path = kCbChunks;
#endif
        if (path == kCbChunks &&
            (k % kCbCH != 0 || x.group % kCbCH != 0 || x.bits != ex[0].bits))
            path = kCbGeneric;
    }

    /* Each row's token, -1 for a dropped slot or a token past `a`, and its expert (libref's
     * binary search, as one sweep over the offsets). */
    std::memset(used, 0, (size_t)AR);
    for (int64_t i = 0; i < T; ++i) {
        const int64_t flat = ldi(c.sorted, offlin(c.sorted, i));
        const int64_t tok = flat < 0 ? -1 : (c.sorted_a ? i : flat / c.top_k);
        atok[i] = (tok >= 0 && tok < AR) ? tok : -1;
        if (atok[i] >= 0) used[atok[i]] = 1;
    }
    {
        int64_t e = 0;
        for (int64_t i = 0; i < T; ++i) {
            while (e + 1 < ne && ldi(c.off, offlin(c.off, e + 1)) <= i) ++e;
            aexp[i] = e;
        }
    }

    /* Every activation row a live row reads, packed once for all its experts' items. */
    AVX_PARALLEL_FOR_IF(AR * K >= AVX_PAR_MIN)
    for (int64_t t = 0; t < AR; ++t) {
        if (!used[t]) continue;
        cb_pack(c, t, K, ex[0].bits, apk + t * K, scratch_f32_b((size_t)K), path);
    }

    /* Zeros: every column of a dropped row, and each part's columns past a narrower class's. */
    const int64_t fy = c.N / c.parts;
    AVX_PARALLEL_FOR_IF(T * c.N >= AVX_PAR_MIN)
    for (int64_t i = 0; i < T; ++i) {
        const int64_t fc = atok[i] < 0 ? 0 : ex[aexp[i]].n / c.parts;
        for (int64_t pp = 0; fc < fy && pp < c.parts; ++pp)
            for (int64_t n = pp * fy + fc; n < (pp + 1) * fy; ++n)
                stt(c.Y, i * c.y_rs + n * c.y_cs, 0.0f);
    }

    /* The live rows, grouped by expert since the experts ascend with the row, and the items. */
    int64_t nl = 0, n_items = 0;
    double work = 0;
    for (int64_t i = 0; i < T; ++i)
        if (atok[i] >= 0) lr[nl++] = i;
    for (int64_t l0 = 0; l0 < nl;) {
        const int64_t e = aexp[lr[l0]];
        int64_t l1 = l0;
        while (l1 < nl && aexp[lr[l1]] == e) ++l1;
        {
            /* The panel form's items are wider: each re-reads its rows once a panel. */
            const int64_t w = l1 - l0 > kCbFused ? kCbNCP : kCbNC;
            for (int64_t r0 = l0; r0 < l1; r0 += kCbRB)
                for (int64_t n0 = 0; n0 < ex[e].n; n0 += w)
                    items[n_items++] = CbItem{ e, r0, r0 + kCbRB < l1 ? r0 + kCbRB : l1, n0,
                                               n0 + w < ex[e].n ? n0 + w : ex[e].n };
        }
        work += (double)(l1 - l0) * (double)ex[e].n * (double)ex[e].k;
        l0 = l1;
    }

#ifdef _OPENMP
    AVX_PRAGMA(omp parallel for schedule(dynamic, 1)
               if(work >= (double)AVX_PAR_MIN || avx_par_forced()) num_threads(avx_team_size()))
#endif
    for (int64_t it = 0; it < n_items; ++it)
        cb_item(c, ex, lr, atok, apk, K, items[it], path);
    return RAD_OK;
}

/* ------------------------------------------------------------------ moe_gemm, moe_gemm_q */
/* y[i, :] = a[row(i), :] @ w[e]^T, where e is the expert whose offset range contains i, and
 * `a_order` picks row(i): `sorted_tok[i] / top_k` when the activation has one row per token, i when
 * it has one row per (token, slot). The output is in sorted order, so each expert's rows are
 * contiguous; moe_gather restores token order.
 *
 * `a_order` is explicit rather than inferred from the row count (M vs M*top_k): gate_up reads one
 * row per token, down reads one per (token, slot), and a meaning that depended on the extent would
 * be invisible to a checker that sizes the operand one way. */
static int moe_gemm_impl(const RadArgs* a, bool quant, int ncls = 2) {
    const int n_opd = quant ? (ncls == 4 ? 13 : 9) : 5;
    if (!rad_args_have(a, n_opd)) return RAD_E_INVAL;
    const RadTensor* A = &a->t[0];
    const RadTensor* as_ = quant ? rad_arg_in(a, 1) : nullptr;
    const RadTensor* W = &a->t[quant ? 2 : 1];
    const RadTensor* ws_ = quant ? rad_arg_in(a, 3) : nullptr;
    const RadTensor* sorted = &a->t[quant ? 4 : 2];
    const RadTensor* off = &a->t[quant ? 5 : 3];
    const RadTensor* Y = &a->t[quant ? (ncls == 4 ? 12 : 8) : 4];
    /* Quantised experts are split into tables by parity: expert e is entry e / 2 of class e % 2
     * (for `moe_gemm_q_mod4`, entry e / 4 of class e % 4). The plain arm has one table. */
    const bool two = quant;
    const int  nc  = quant ? ncls : 1;
    const RadTensor* W1 = two ? &a->t[6] : W;
    const RadTensor* ws1_ = two ? rad_arg_in(a, 7) : nullptr;
    const RadTensor* Wq[4]  = { W, W1, quant && nc == 4 ? &a->t[8] : W,
                                quant && nc == 4 ? &a->t[10] : W };
    const RadTensor* wsq[4] = { ws_, ws1_, quant && nc == 4 ? rad_arg_in(a, 9) : nullptr,
                                quant && nc == 4 ? rad_arg_in(a, 11) : nullptr };

    const char* ord = p_str(a, "a_order", "token");
    const bool sorted_a = ord && std::strcmp(ord, "sorted") == 0;
    if (ord && !sorted_a && std::strcmp(ord, "token") != 0) return RAD_E_UNSUPPORTED;

    const int64_t N0 = p_int(a, "N", W->rank >= 3 ? W->shape[1] : 0);
    const int64_t K0 = p_int(a, "K", W->rank >= 3 ? W->shape[2] : 0);
    /* Per-class geometry, defaulting to class 0's: N_odd/K_odd for two classes, N_1..N_3 /
     * K_1..K_3 for four. */
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
    /* Output width is the widest class's N, activation width the largest K; class q holds
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

    const int64_t M = T / top_k;
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

    /* u4 or u5 codes are the codebook experts' joined form, which only their own path reads. */
    if (quant && (Wq[0]->dtype == RAD_U4 || Wq[0]->dtype == RAD_U5)) {
        CbCall c;
        c.A = A; c.sorted = sorted; c.off = off; c.Y = Y;
        for (int q = 0; q < 4; ++q) {
            c.Wq[q] = Wq[q]; c.wsq[q] = wsq[q]; c.Nq[q] = Nq[q]; c.Kq[q] = Kq[q];
        }
        c.nc = nc; c.parts = parts; c.n_expert = n_expert; c.top_k = top_k;
        c.N = N; c.K = K; c.T = T; c.AR = AR; c.y_rs = y_rs; c.y_cs = y_cs;
        c.sorted_a = sorted_a;
        c.ma = ma; c.SA = SA;
        return moe_codebook(c);
    }

    /* The per-expert readings and the rows' sources and experts, which the parallel regions read,
     * in vectors this thread keeps: the tile drivers take scratch_raw blocks of their own, and a
     * second scratch_raw call may free the first under the worker threads. */
    static thread_local std::vector<RadTensor> we_v, wse_v;
    static thread_local std::vector<Mat> mw_v;
    static thread_local std::vector<Scales> sw_v;
    static thread_local std::vector<int64_t> atok_v, aexp_v;
    we_v.assign((size_t)n_expert, RadTensor{});
    wse_v.assign((size_t)n_expert, RadTensor{});
    mw_v.assign((size_t)n_expert, Mat{});
    sw_v.assign((size_t)n_expert, Scales{});
    atok_v.resize((size_t)T);
    aexp_v.resize((size_t)T);
    RadTensor* we  = we_v.data();
    RadTensor* wse = wse_v.data();
    Mat*       mw  = mw_v.data();
    Scales*    SW  = sw_v.data();
    int64_t*   atok = atok_v.data();
    int64_t*   aexp = aexp_v.data();

    /* Per-expert readings, built once. Unreadable tables (null `data`) are refused here, before
     * any parallel region, since an OpenMP body cannot return. */
    bool fine_a = false;
    for (int64_t e = 0; e < n_expert; ++e) {
        const int q = (int)(e % nc);
        const RadTensor* tw = Wq[q];
        const RadTensor* ts = wsq[q];
        const int64_t ix = e / nc;
        const int64_t n = Nq[q], k = Kq[q];
        we[e] = expert_entry(tw, ix);
        if (!we[e].data) return RAD_E_STRIDE;
        if (we[e].rank < 2 || we[e].shape[0] < n || we[e].shape[1] < k) return RAD_E_SHAPE;
        if (quant && ts) {
            wse[e] = expert_entry(ts, ix);
            if (!wse[e].data) return RAD_E_STRIDE;
        }
        if (quant) {
            /* Expert 0 is plain, so codebook codes or a joined scale here mix two forms in one
             * call, refused as libref refuses them. */
            CbExp cx;
            const int jc = cb_codes(we[e], n, k, &cx);
            if (jc < 0) return jc;
            if (jc > 0 || (ts && cb_joined(wse[e]))) return RAD_E_DTYPE;
        }
        mw[e] = as_mat(&we[e], k);
        SW[e] = make_scales(quant && ts ? &wse[e] : nullptr, n, k, we[e].dtype == RAD_FP4E2M1);
        /* An activation scale grid finer than an expert's cannot be hoisted out of the group. No
         * supported format has one, but libref answers it by scaling each element, so the panel
         * below is scaled as it is packed and the dot reads no activation scale (`SAg`). */
        const int64_t g = SW[e].p && SW[e].group > 0 ? SW[e].group : k;
        if (SA.p && SA.group < g) fine_a = true;
    }
    Scales SAg = SA;
    if (fine_a) SAg.p = nullptr;

    /* Each sorted row's activation row (-1: a dropped slot) and owning expert, in one linear
     * sweep over the offsets (rows are expert-sorted). */
    for (int64_t i = 0; i < T; ++i) {
        const int64_t flat = ldi(sorted, offlin(sorted, i));
        const int64_t tok = flat < 0 ? -1 : (sorted_a ? i : flat / top_k);
        atok[i] = (tok >= 0 && tok < AR) ? tok : -1;
    }
    {
        int64_t e = 0;
        for (int64_t i = 0; i < T; ++i) {
            while (e + 1 < n_expert && ldi(off, offlin(off, e + 1)) <= i) ++e;
            aexp[i] = e;
        }
    }

    /* Output column n is weight row (n / fy) * fc + n % fy of its expert, and zero past the end of
     * a narrower class's part. */
    const int64_t fy = N / parts;

    /* The tile drivers: one job per (run of an expert's rows, part), every job's tasks in one
     * parallel region, the activation rows read through atok. Dropped slots and the columns past
     * a narrower class's part are zero, written after. */
    {
        static thread_local std::vector<TileJob> jobs;
        jobs.clear();
        for (int64_t i0 = 0; i0 < T;) {
            int64_t i1 = i0 + 1;
            while (i1 < T && aexp[i1] == aexp[i0]) ++i1;
            const int64_t e = aexp[i0];
            const int q = (int)(e % nc);
            const int64_t fc = Nq[q] / parts;
            for (int64_t pp = 0; pp < parts; ++pp) {
                TileJob jb{};
                jb.mb = mw[e];
                jb.SB = SW[e];
                jb.b0 = pp * fc;
                jb.a0 = i0; jb.M = i1 - i0; jb.N = fc; jb.K = Kq[q];
                jb.y0 = i0; jb.yc0 = pp * fy;
                jobs.push_back(jb);
            }
            i0 = i1;
        }
        const Scales* sap = SA.p ? &SA : nullptr;
        if (gemm_q_bf16_jobs(ma, atok, T, sap, jobs.data(), (int64_t)jobs.size(), Y, N) ||
            gemm_tile_f32_jobs(ma, atok, T, sap, jobs.data(), (int64_t)jobs.size(), Y, N)) {
            AVX_PARALLEL_FOR_IF(T * N >= AVX_PAR_MIN)
            for (int64_t i = 0; i < T; ++i) {
                const int64_t fc = atok[i] < 0 ? 0 : Nq[(int)(aexp[i] % nc)] / parts;
                if (fc >= fy) continue;
                for (int64_t pp = 0; pp < parts; ++pp)
                    for (int64_t n = pp * fy + fc; n < (pp + 1) * fy; ++n)
                        stt(Y, i * y_rs + n * y_cs, 0.0f);
            }
            return RAD_OK;
        }
    }

    /* Activation panel: each output row's source row widened to f32 once, so the N loop reuses
     * it. Dropped slots have a negative token, and their output rows are zeroed below. */
    float* apanel = (float*)scratch_raw((size_t)T * (size_t)K * sizeof(float) + 64);
    AVX_PARALLEL_FOR_IF(T * K >= AVX_PAR_MIN)
    for (int64_t i = 0; i < T; ++i) {
        if (atok[i] < 0) continue;
        float* row = apanel + i * K;
        pack_row(ma, atok[i], K, row);
        if (fine_a)
            for (int64_t k = 0; k < K; ++k) row[k] *= SA.grp(atok[i], k / SA.group);
    }

    AVX_PARALLEL_FOR_IF(T * N * K >= AVX_PAR_MIN)
    for (int64_t n = 0; n < N; ++n) {
        float* bbuf = scratch_f32((size_t)K);
        int64_t packed_e = -1;
        const int64_t pp = n / fy, j = n - pp * fy;
        for (int64_t i = 0; i < T; ++i) {
            if (atok[i] < 0) { stt(Y, i * y_rs + n * y_cs, 0.0f); continue; }
            const int64_t e = aexp[i];
            const int q = (int)(e % nc);
            const int64_t fc = Nq[q] / parts, ke = Kq[q];
            if (j >= fc) { stt(Y, i * y_rs + n * y_cs, 0.0f); continue; }
            const int64_t wr = pp * fc + j;
            /* Repack the weight row only when the expert changes: at most n_expert times per
             * column, since rows are expert-sorted. */
            if (e != packed_e) { pack_row(mw[e], wr, ke, bbuf); packed_e = e; }
            const Scales& SB = SW[e];
            const int64_t g = SB.p && SB.group > 0 ? SB.group : ke;
            const int64_t ngrp = (ke + g - 1) / g;
            const float* av = apanel + i * K;
            float acc = 0.0f;
            for (int64_t gi = 0; gi < ngrp; ++gi) {
                const int64_t k0 = gi * g, k1 = (k0 + g < ke) ? k0 + g : ke;
                const float sb = SB.p ? SB.grp(wr, gi) : 1.0f;
                const float sa = (SAg.p && SAg.group > 0) ? SAg.grp(atok[i], k0 / SAg.group)
                                                          : 1.0f;
                acc += dotf(av + k0, bbuf + k0, k1 - k0) * sa * sb;
            }
            stt(Y, i * y_rs + n * y_cs, acc);
        }
    }
    return RAD_OK;
}

AVX_KERNEL(moe_gemm)   { return moe_gemm_impl(a, false); }
AVX_KERNEL(moe_gemm_q) { return moe_gemm_impl(a, true); }
AVX_KERNEL(moe_gemm_q_mod4) { return moe_gemm_impl(a, true, 4); }

/* ------------------------------------------------------------------ moe_gather */
/* y[t] = sum over the token's slots of expert_w[t, j] * y_expert[position of (t, j)], summed in
 * ascending slot j (see the file header). The inverse permutation is built once per batch. */
AVX_KERNEL(moe_gather) {
    /* NOT `rad_args_have(a, 6)`: the required operands are not a prefix -- the shared pair at 3
     * and 4 is a NULL SLOTS form for a layer whose shared arm runs beside the block, and the
     * prefix test reads a legal unfolded call as a missing operand. */
    if (!a || !a->t || a->n_t < 6 || !rad_args_have(a, 3) || !rad_arg_in(a, 5)) return RAD_E_INVAL;
    const RadTensor *ye = &a->t[0], *ew = &a->t[1], *sorted = &a->t[2], *Y = &a->t[5];
    /* Optional shared-expert fold: y = narrow(shared * act(gate)) + narrow(gathered). Both
     * operands or neither, as in libref. */
    if (!ye->data || !ew->data || !sorted->data || !Y->data) return RAD_E_INVAL;
    const RadTensor *sh = rad_arg_in(a, 3), *sg = rad_arg_in(a, 4);
    if ((sh != nullptr) != (sg != nullptr)) return RAD_E_INVAL;
    const char* act = p_str(a, "act", "none");
    const bool sig = act && !std::strcmp(act, "sigmoid");
    if (sh && act && std::strcmp(act, "sigmoid") && std::strcmp(act, "none")) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0);
    const int64_t top_k = p_int(a, "top_k", 1);
    if (n <= 0 || top_k <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(Y) / n;
    const int64_t T = M * top_k;
    if (M <= 0 || numel(sorted) < T || numel(ew) < T || numel(ye) < T * n) return RAD_E_SHAPE;

    int32_t* inv = (int32_t*)scratch_raw((size_t)T * sizeof(int32_t) + 64);
    for (int64_t i = 0; i < T; ++i) inv[i] = -1;
    for (int64_t i = 0; i < T; ++i) {
        const int64_t f = ldi(sorted, offlin(sorted, i));
        if (f >= 0 && f < T) inv[f] = (int32_t)i;
    }

    const int64_t ye_rs = ye->rank >= 2 ? ye->stride[ye->rank - 2] : n;
    const int64_t ye_cs = ye->rank >= 2 ? ye->stride[ye->rank - 1] : 1;
    /* libref's Y addressing for the fold: stride[rank-2] / stride[rank-1], rather than out_row's
     * rowoff/laststride. They agree for rank-2 [M, n]; this matches the reference for higher
     * ranks too. */
    const int64_t y_rs = Y->rank >= 2 ? Y->stride[Y->rank - 2] : n;
    const int64_t y_cs = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;

    AVX_PARALLEL_FOR_IF(M * n * top_k >= AVX_PAR_MIN)
    for (int64_t t = 0; t < M; ++t) {
        float* acc = scratch_f32((size_t)n);
        float* row = scratch_f32_b((size_t)n);
        for (int64_t c = 0; c < n; ++c) acc[c] = 0.0f;
        for (int64_t j = 0; j < top_k; ++j) {
            const int32_t i = inv[t * top_k + j];
            if (i < 0) continue;
            const float w = ldt(ew, rowoff(ew, t, top_k) + j * laststride(ew));
            const float* v;
            if (ye_cs == 1) {
                v = row_f32_in(ye->dtype, byte_at(ye->data, ye->dtype, (int64_t)i * ye_rs), row, n);
            } else {
                for (int64_t c = 0; c < n; ++c)
                    row[c] = rad_load_f32(ye->data, ye->dtype, (int64_t)i * ye_rs + c * ye_cs);
                v = row;
            }
            const vf vw = vf_set1(w);
            int64_t c = 0;
            for (; c + VF_N <= n; c += VF_N)
                vf_storeu(acc + c, vf_fma(vw, vf_loadu(v + c), vf_loadu(acc + c)));
            for (; c < n; ++c) acc[c] += w * v[c];
        }
        if (!sh) {
            out_row(Y, rowoff(Y, t, n), n, acc);
            continue;
        }
        /* The unfused pair's roundings, in libref's order: store the gathered row into Y, store
         * the shared product into Y, store their sum. The per-token gate uses libm's expf, as the
         * oracle does. */
        const float v0 = ldt(sg, t * laststride(sg));
        const float gate = sig ? 1.0f / (1.0f + sc_exp(-v0)) : v0;
        const int64_t shb = rowoff(sh, t, n), shs = laststride(sh);
        for (int64_t c = 0; c < n; ++c) {
            const int64_t yo = t * y_rs + c * y_cs;
            stt(Y, yo, acc[c]);
            const float g = ldt(Y, yo);
            stt(Y, yo, ldt(sh, shb + c * shs) * gate);
            stt(Y, yo, ldt(Y, yo) + g);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ logit_rerank */
/* Rescores R candidate ids against the full-precision head and re-sorts them: the second half of
 * a coarse top-R. `idx_in` holds global ids, `vocab_off` is this rank's first column, and an id
 * outside the head's rows becomes a pad that sorts last with -inf.
 *
 * The dots are f32 vector sums (libref sums in f64 in order); at K = 2048 they differ ~1e-6
 * relative, inside the f32 tolerance of 1e-5. The selection is exact: descending, ties to the
 * lower id, pads dropped, the same rule as row_topk.
 *
 * `head` is read flat ([NV, K] row-major, unit stride), as the reference reads it: candidate g's
 * row starts at element g * K. */
AVX_KERNEL(logit_rerank) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *in = &a->t[2];
    const RadTensor *idx = &a->t[3], *val = &a->t[4];
    const RadTensor* pairs = (a->n_t > 5 && a->t[5].data) ? &a->t[5] : nullptr;
    const int64_t R    = p_int(a, "R", 1);
    const int64_t K    = p_int(a, "n_embd", 0);
    const int64_t NV   = p_int(a, "n_vocab", 0);
    const int64_t voff = p_int(a, "vocab_off", 0);
    if (R <= 0 || K <= 0 || NV <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(x) / K;
    if (M <= 0 || numel(in) < M * R) return RAD_E_SHAPE;
    if (numel(idx) < M * R || numel(val) < M * R) return RAD_E_SHAPE;
    /* A head of codes and scale, joined (avx_planes.h), decodes a candidate's row as the core's
     * decoder does, so its scores are those of the decoded f32 head. */
    avxp::ScaledRows sr;
    const bool joined = avxp::is_joined(w);
    if (joined) {
        if (const int st = avxp::scaled_rows(w, NV, K, &sr); st != RAD_OK) return st;
    } else if (numel(w) < NV * K) {
        return RAD_E_SHAPE;
    }

    AVX_PARALLEL_FOR_IF(M * R * K >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* xb = scratch_f32((size_t)K);
        float* wb = scratch_f32_b((size_t)K);
        float* sc = scratch_f32_c((size_t)R);
        const float* xv = in_row(x, rowoff(x, m, K), K, xb);
        /* One raw block for `id` and `used`: scratch_raw has a single slot, so a second call
         * would return the same pointer. */
        int32_t* id = (int32_t*)scratch_raw((size_t)R * sizeof(int32_t) + (size_t)R + 64);
        for (int64_t r = 0; r < R; ++r) {
            const int64_t g = ldi(in, rowoff(in, m, R) + r * laststride(in));
            const int64_t local = g - voff;
            if (g < 0 || local < 0 || local >= NV) {
                id[r] = -1;
                sc[r] = -INFINITY;
                continue;
            }
            if (joined) scaled_row(sr, local, K, wb);
            else row_to_f32(w->dtype, byte_at(w->data, w->dtype, local * K), wb, K);
            id[r] = (int32_t)g;
            sc[r] = dotf(xv, wb, K);
        }
        uint8_t* used = (uint8_t*)(id + R);
        for (int64_t r = 0; r < R; ++r) used[r] = 0;
        for (int64_t o = 0; o < R; ++o) {
            int64_t best = -1;
            for (int64_t c = 0; c < R; ++c) {
                if (used[c] || id[c] < 0) continue;
                if (best < 0 || sc[c] > sc[best] ||
                    (sc[c] == sc[best] && id[c] < id[best])) best = c;
            }
            const int32_t oid = best >= 0 ? id[best] : -1;
            const float ov = best >= 0 ? sc[best] : -INFINITY;
            if (best >= 0) used[best] = 1;
            st_int_dt(idx->dtype, idx->data, rowoff(idx, m, R) + o * laststride(idx), oid);
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

/* ------------------------------------------------------------------ row_topk */
/* Exact per-row top-R, descending, ties to the lower column: the router's selection without the
 * softmax (for drafter candidate sets and speculative trees).
 *
 * Vocab-parallel form: `vocab_off` shifts reported columns to global ids, and `pairs` is the
 * [M, R, 2] plane an all_gather moves (int32 id bit-copied into the first float, then the value).
 * Without either it is the single-rank top-R. */
AVX_KERNEL(row_topk) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *idx = &a->t[1], *val = &a->t[2];
    const RadTensor* pairs = rad_arg_in(a, 3);
    const int64_t off = p_int(a, "vocab_off", 0);
    const int64_t N = p_int(a, "N", x->rank >= 2 ? x->shape[x->rank - 1] : 0);
    const int64_t R = p_int(a, "R", 1);
    if (N <= 0 || R <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(x) / N;
    if (M <= 0 || numel(idx) < M * R || numel(val) < M * R) return RAD_E_SHAPE;
    if (pairs && (pairs->dtype != RAD_F32 || numel(pairs) < M * R * 2)) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M * N >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* buf = scratch_f32((size_t)N);
        const float* v = in_row(x, rowoff(x, m, N), N, buf);
        float pv = INFINITY;
        int64_t pi = -1;
        for (int64_t r = 0; r < R; ++r) {
            const bool ok = topk_next_f32(v, N, &pv, &pi);
            const int64_t id = ok ? pi + off : -1;
            st_int_dt(idx->dtype, idx->data, rowoff(idx, m, R) + r * laststride(idx), id);
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
/* The R largest of world_size all-gathered per-rank candidate sets, in the single-rank path's
 * descending (idx, val) form.
 *
 * Ties break on the lower id, so the merge matches a dense top-R over the whole vocabulary exactly.
 * Unfilled slots carry id -1 and are dropped (an id sentinel, since -inf is a legitimate masked
 * logit).
 *
 * Scalar: W*R is small (32 at world 2, R 16) and the (value, id) tie rule has no vector max. */
AVX_KERNEL(row_topk_merge) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *g = &a->t[0], *idx = &a->t[1], *val = &a->t[2];
    const int64_t R = p_int(a, "R", 1);
    const int64_t W = p_int(a, "world_size", 1);
    if (R <= 0 || W <= 0) return RAD_E_SHAPE;
    /* Rows come from the operand; `M` is only the band's ceiling. */
    const int64_t M = numel(g) / (W * R * 2);
    if (M <= 0) return RAD_E_SHAPE;
    if (g->dtype != RAD_F32 || numel(g) < M * W * R * 2) return RAD_E_SHAPE;
    if (numel(idx) < M * R || numel(val) < M * R) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M * W * R * R >= 4096)
    for (int64_t m = 0; m < M; ++m) {
        const float* const row = (const float*)g->data + m * W * R * 2;
        unsigned char* used = (unsigned char*)scratch_f32((size_t)(W * R) / 4 + 4);
        for (int64_t c = 0; c < W * R; ++c) used[c] = 0;
        for (int64_t r = 0; r < R; ++r) {
            int64_t best = -1;
            float bv = 0.0f;
            int32_t bid = 0;
            for (int64_t c = 0; c < W * R; ++c) {
                if (used[c]) continue;
                int32_t id;
                std::memcpy(&id, row + c * 2, sizeof id);
                if (id < 0) continue;                      /* pad */
                const float v = row[c * 2 + 1];
                if (best < 0 || v > bv || (v == bv && id < bid)) { best = c; bv = v; bid = id; }
            }
            if (best >= 0) used[best] = 1;
            st_int_dt(idx->dtype, idx->data, rowoff(idx, m, R) + r * laststride(idx),
                      best >= 0 ? (int64_t)bid : -1);
            stt(val, rowoff(val, m, R) + r * laststride(val), best >= 0 ? bv : -INFINITY);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ pair_argmax */
/* row_topk_merge at R = 1, one id per row at the output's own pitch: the vocab-sharded draft
 * head's proposal. Same pads and tie rule; -1 and -inf for a row with nothing valid; `val` is
 * optional. Scalar: `n` is world_size * R (16 pairs at TP8, R 2). */
AVX_KERNEL(pair_argmax) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *p = &a->t[0], *idx = &a->t[1];
    const RadTensor* val = rad_arg_in(a, 2);
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    if (p->dtype != RAD_F32) return RAD_E_DTYPE;   /* the id is carried by bit pattern */
    const int64_t M = numel(p) / (n * 2);
    if (M <= 0 || numel(idx) < M || (val && numel(val) < M)) return RAD_E_SHAPE;
    const float* const pf = (const float*)p->data;

    AVX_PARALLEL_FOR_IF(M * n >= 4096)
    for (int64_t m = 0; m < M; ++m) {
        int32_t bid = -1;
        float   bv  = -INFINITY;
        for (int64_t j = 0; j < n; ++j) {
            int32_t id;
            std::memcpy(&id, pf + offlin(p, (m * n + j) * 2), sizeof id);
            if (id < 0) continue;                          /* pad */
            const float v = pf[offlin(p, (m * n + j) * 2 + 1)];
            if (bid < 0 || v > bv || (v == bv && id < bid)) { bid = id; bv = v; }
        }
        st_int_dt(idx->dtype, idx->data, offlin(idx, m), bid);
        if (val) stt(val, offlin(val, m), bv);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ router_topk_scatter */
/* The pair as one op: `router_topk` then `moe_scatter` over the ids it just wrote. The selection
 * reads the whole router's `n_expert` logits; the sort runs over this rank's slice, which is the
 * fused op's `n_local` standing where `moe_scatter`'s `n_expert` names the slice it owns.
 *
 * The operands line up with `router_topk`'s own first three slots (logits, ids, weights), so the
 * selection is the sibling kernel called on this very RadArgs -- no copy, no re-derivation of the
 * softmax -- and the sort is `moe_scatter` over the four operands it takes. */
AVX_KERNEL(router_topk_scatter) {
    if (!rad_args_have(a, 6)) return RAD_E_INVAL;
    const int64_t ne = p_int(a, "n_expert", 0);
    const int64_t nl = p_int(a, "n_local", ne);
    const int64_t k  = p_int(a, "top_k", 1);
    if (ne <= 0 || k <= 0 || k > ne || nl <= 0 || nl > ne) return RAD_E_SHAPE;

    const int rc = AVX_FN(avx_router_topk)(a, stream);
    if (rc != RAD_OK) return rc;

    const RadTensor t2[] = { a->t[1], a->t[3], a->t[4], a->t[5] };
    const RadParam sp[]   = { RAD_INT("n_expert", nl), RAD_INT("top_k", k),
                              RAD_INT("expert_base", p_int(a, "expert_base", 0)),
                              RAD_STR("protect", p_str(a, "protect", nullptr)) };
    const RadArgs sa{ t2, 4, sp, 4, a->scratch, a->scratch_bytes, a->instance, a->rank,
                      a->world_size };
    return AVX_FN(avx_moe_scatter)(&sa, stream);
}

/* ------------------------------------------------------------------ router_gemm_topk_scatter */
/* The router GEMM in front of the pair, and with `gate_b`/`gate` present the shared expert's
 * one-column gate over the same input, at N = 1. `gate_b` and `gate` come together or not at all,
 * and the operand count says which. */
extern "C" int AVX_FN(avx_gemm_nt)(const RadArgs*, RadStream);

AVX_KERNEL(router_gemm_topk_scatter) {
    /* NOT `rad_args_have`: the required operands are not a prefix -- `gate_b` sits between the
     * router's weight and the outputs, and it and `gate` are null slots in the gate-absent form. */
    if (!a || !a->t || a->n_t < 9) return RAD_E_INVAL;
    const bool gate_b = a->t[2].data != nullptr;
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

    const RadTensor g3[] = { a->t[0], a->t[1], a->t[3] };
    const RadParam  gp[] = { RAD_INT("N", ne), RAD_INT("K", K), RAD_STR("dtype", dt) };
    const RadArgs   ga{ g3, 3, gp, 3, a->scratch, a->scratch_bytes, a->instance, a->rank,
                        a->world_size };
    const int gc = AVX_FN(avx_gemm_nt)(&ga, stream);
    if (gc != RAD_OK) return gc;

    if (gate) {
        const RadTensor p3[] = { a->t[0], a->t[2], a->t[9] };
        const RadParam  pp[] = { RAD_INT("N", 1), RAD_INT("K", K), RAD_STR("dtype", dt) };
        const RadArgs   pa{ p3, 3, pp, 3, a->scratch, a->scratch_bytes, a->instance, a->rank,
                            a->world_size };
        const int pc = AVX_FN(avx_gemm_nt)(&pa, stream);
        if (pc != RAD_OK) return pc;
    }

    /* The pair over the logits the GEMM just wrote. Every parameter the pair reads is a parameter
     * of this op, under the same name -- the fused lists are unions. */
    const RadTensor t6[] = { a->t[3], a->t[4], a->t[5], a->t[6], a->t[7], a->t[8] };
    const RadArgs   fa{ t6, 6, a->p, a->n_p, a->scratch, a->scratch_bytes, a->instance, a->rank,
                        a->world_size };
    return AVX_FN(avx_router_topk_scatter)(&fa, stream);
}
