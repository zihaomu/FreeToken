/* avx_gdn.cpp -- the gated delta net: the two convolution forms, the triangular solve, the chunked
 * scan, the decode recurrence, and the gated norm.
 *
 * These ops contract heads (128 elements of k against 128 of q, or a [128, 128] state against a
 * key), and libref reads every operand of every product through `ldt`, a dtype switch per element.
 * In the quadratic ops that is the whole cost (gdn_kkt_solve at chunk 64, head_k 128: 1 M switched
 * loads over 8 K distinct elements). Here each chunk's q, k and v rows are widened once into f32
 * panels and every dot is a vector reduction. The algorithms are libref's, step for step:
 *
 *   The decay is applied per element as e^{g_i - g_j}, not split around a per-chunk reference.
 *   Only i > j is used and g decreases along the chunk, so the exponent is <= 0 and cannot
 *   overflow; the split form can, at spans this model reaches.
 *
 *   The triangular inverse is a forward substitution. I + M is unit lower triangular, so the
 *   inverse is unique and libr4d's blocked WMMA merge agrees with it.
 *
 *   The conv output is rounded before the L2 norm, as the unfused pair does (the conv writes a
 *   tensor that the prep reads back): it is written to q/k/v and re-read for the norm.
 *
 *   The L2 norm does not divide by the head width: rsqrt(sum(x^2) + l2_eps), l2_eps 1e-6 (FLA's
 *   and libr4d's L2NORM_EPS).
 *
 * Rollback contract (spec §10): speculative rejection without recompute. The recurrent state and
 * conv window have already absorbed the rejected tokens, so `num_accepted` is required and a
 * rejection is a change of read offset:
 *
 *   conv state   read at `num_accepted - 1`, rewritten shifted down by one, so slots
 *                0..conv_width-3 take the tail of the old history and this step's token t lands at
 *                slot t + conv_width - 2.
 *   linear state read from state_idx[s, num_accepted - 1]; every candidate token writes its own
 *                state to state_idx[s, t], since which survives is known only after this layer.
 *
 * `num_accepted` is one-based: the number of tokens the previous step committed, so 1 is a plain
 * decode with no drafts and 0 is invalid.
 *
 * `state_form` names the state contract: "per_candidate" (above), or "anchor", one committed state
 * a sequence plus each token's replay factors beside it, in libr4d's cache layout (AnchorLayout,
 * at gdn_recurrent_update). Both forms give the same `o`; an implementation handed the other form
 * would continue from the wrong state, so the form is a parameter, never inferred.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <vector>

using namespace avx;

/* The view helpers are `view2` / `view3` / `view4`, not libref's `v2` / `v3` / `v4`: those names
 * collide with the per-level inline namespaces (AVX_SUFFIX: sc / v1 / v2 / v3) at three of the four
 * levels.
 *
 * Deepest convolution history held on the stack. libr4d's width is 4; a wider request returns
 * RAD_E_UNSUPPORTED rather than convolving a truncated window. Same bound as libref. */
enum { AVX_CONV_HIST_MAX = 16 };

struct V3 { const RadTensor* t; int64_t s0, s1, s2; };
struct V2 { const RadTensor* t; int64_t s0, s1; };
struct V4 { const RadTensor* t; int64_t s0, s1, s2, s3; };

static inline V3 view3(const RadTensor* t, int64_t d1, int64_t d2) {
    V3 v{ t, d1 * d2, d2, 1 };
    if (t->rank >= 3) {
        v.s0 = t->stride[t->rank - 3];
        v.s1 = t->stride[t->rank - 2];
        v.s2 = t->stride[t->rank - 1];
    }
    return v;
}
/* view3 for an operand that may be a column view of a wider row (the fp8 delta net's output gate is
 * the tail of a fused [q|k|v|z] projection). At rank 2 the token stride is stride[0], where view3
 * would assume d1*d2. */
static inline V3 view3_rows(const RadTensor* t, int64_t d1, int64_t d2) {
    if (t->rank != 2) return view3(t, d1, d2);
    return V3{ t, t->stride[0], d2 * t->stride[1], t->stride[1] };
}
static inline V2 view2(const RadTensor* t, int64_t d1) {
    V2 v{ t, d1, 1 };
    if (t->rank >= 2) { v.s0 = t->stride[t->rank - 2]; v.s1 = t->stride[t->rank - 1]; }
    return v;
}
static inline int64_t at3(const V3& v, int64_t i, int64_t j, int64_t k) {
    return i * v.s0 + j * v.s1 + k * v.s2;
}
static inline int64_t at2(const V2& v, int64_t i, int64_t j) { return i * v.s0 + j * v.s1; }

/* [N, heads, head_v, head_k], the shape of the chunked scan's h0/ht and of the paged recurrent
 * state. Slot and head strides come from the tensor, not the shape: a paged state cache is padded
 * so its page matches the attention page. */
static inline V4 view4(const RadTensor* t, int64_t d1, int64_t d2, int64_t d3) {
    V4 v{ t, d1 * d2 * d3, d2 * d3, d3, 1 };
    if (t->rank >= 4) {
        v.s0 = t->stride[t->rank - 4];
        v.s1 = t->stride[t->rank - 3];
        v.s2 = t->stride[t->rank - 2];
        v.s3 = t->stride[t->rank - 1];
    }
    return v;
}
static inline int64_t at4(const V4& v, int64_t i, int64_t j, int64_t k, int64_t l) {
    return i * v.s0 + j * v.s1 + k * v.s2 + l * v.s3;
}

static inline int64_t cu_at(const RadTensor* cu, int64_t i) { return ldi(cu, offlin(cu, i)); }

/* The conv state is [n_slots, conv_dim, state_len]; a flattened tensor is read as the dense layout
 * that shape implies. */
static inline V3 conv_state_view(const RadTensor* cs, int64_t conv_dim, int64_t width,
                                 int64_t* state_len) {
    *state_len = cs->rank >= 3 ? cs->shape[cs->rank - 1] : (width > 1 ? width - 1 : 1);
    return view3(cs, conv_dim, *state_len);
}

/* Two-sided softplus with FLA's and libr4d's large-x cutoff: above the threshold log1p(e^x) equals
 * x in f32, and evaluating it would overflow. */
static inline float softplus(float x, float thr) {
    if (x > thr) return x;
    return x > 0.0f ? x + sc_log1p(sc_exp(-x)) : sc_log1p(sc_exp(x));
}

/* A dot product with two accumulators: the delta net's summation order, which differs from
 * avx_vec.h's four-accumulator dotf. */
static inline float dot2(const float* a, const float* b, int64_t n) {
    vf a0 = vf_zero(), a1 = vf_zero();
    int64_t i = 0;
    for (; i + 2 * VF_N <= n; i += 2 * VF_N) {
        a0 = vf_fma(vf_loadu(a + i), vf_loadu(b + i), a0);
        a1 = vf_fma(vf_loadu(a + i + VF_N), vf_loadu(b + i + VF_N), a1);
    }
    for (; i + VF_N <= n; i += VF_N) a0 = vf_fma(vf_loadu(a + i), vf_loadu(b + i), a0);
    float s = vf_hsum(vf_add(a0, a1));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* Widen and transpose a [cbw, width] tap block starting at channel c0 into tapT[j*cbw + c], so the
 * token loops read tap columns contiguously. The vector form (width 4, dense f32 or bf16 rows,
 * AVX2+) widens 8 rows into a 32-float staging block and scatters them into the four columns;
 * loads and stores only, so bit-exact. Everything else, and the ragged tail, is scalar. */
static inline void fill_tapT(const RadTensor* w, V2 wv, int64_t c0, int64_t cbw, int64_t width,
                             float* tapT) {
#if AVX_LEVEL >= 2
    if (width == 4 && wv.s0 == 4 && wv.s1 == 1 &&
        (w->dtype == RAD_F32 || w->dtype == RAD_BF16)) {
        const bool f32 = (w->dtype == RAD_F32);
        int64_t c = 0;
        for (; c + 8 <= cbw; c += 8) {
            alignas(32) float stg[8 * 4];
            if (f32) {
                const float* base = (const float*)w->data + (c0 + c) * 4;
                _mm256_store_ps(stg, _mm256_loadu_ps(base));
                _mm256_store_ps(stg + 8, _mm256_loadu_ps(base + 8));
                _mm256_store_ps(stg + 16, _mm256_loadu_ps(base + 16));
                _mm256_store_ps(stg + 24, _mm256_loadu_ps(base + 24));
            } else {
                /* bf16 -> f32 is a zero-extend and shift (not cvtph, which reads f16), as in
                 * cvt_bf16_to_f32. */
                const uint16_t* base = (const uint16_t*)w->data + (c0 + c) * 4;
                for (int k = 0; k < 4; ++k) {
                    __m128i h = _mm_loadu_si128((const __m128i*)(base + k * 8));
                    __m256i e = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
                    _mm256_store_ps(stg + k * 8, _mm256_castsi256_ps(e));
                }
            }
            for (int64_t j = 0; j < 4; ++j)
                for (int64_t k = 0; k < 8; ++k) tapT[j * cbw + c + k] = stg[k * 4 + j];
        }
        for (; c < cbw; ++c)
            for (int64_t j = 0; j < width; ++j)
                tapT[j * cbw + c] = rad_load_f32(w->data, w->dtype, at2(wv, c0 + c, j));
        return;
    }
#endif
    for (int64_t c = 0; c < cbw; ++c)
        for (int64_t j = 0; j < width; ++j)
            tapT[j * (size_t)cbw + c] = ldt(w, at2(wv, c0 + c, j));
}

static inline void axpy(float* dst, const float* src, float c, int64_t n) {
    const vf vc = vf_set1(c);
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N)
        vf_storeu(dst + i, vf_fma(vc, vf_loadu(src + i), vf_loadu(dst + i)));
    for (; i < n; ++i) dst[i] += c * src[i];
}

/* A [rows, width] panel of an operand read as [token, head, d], widened to f32 once (see the file
 * header). */
static void panel3(const RadTensor* t, const V3& v, int64_t t0, int64_t rows, int64_t h,
                   int64_t width, float* dst) {
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t b = at3(v, t0 + r, h, 0);
        if (v.s2 == 1) row_to_f32(t->dtype, byte_at(t->data, t->dtype, b), dst + r * width, width);
        else for (int64_t d = 0; d < width; ++d) dst[r * width + d] = ldt(t, b + d * v.s2);
    }
}

/* Writes one token's convolution output over channel panel [c0, c0 + cbw) into q, k and v, each
 * piece at its head and its offset within the head. Panels are 1024 channels, so an edge falls
 * mid-head when the head width does not divide 1024 (e.g. 96 or 192); runs are cut at head
 * boundaries so each is contiguous within one head's row. */
static inline void store_pieces(const float* acc, int64_t cbw, int64_t c0, int64_t bos, int64_t t,
                                const RadTensor* q, const RadTensor* k, const RadTensor* v,
                                V3 qv, V3 kv, V3 vv, int64_t Hg, int64_t K, int64_t V,
                                int64_t conv_dim) {
    const RadTensor* outs[3] = { q, k, v };
    const V3 ovs[3] = { qv, kv, vv };
    const int64_t edges[4] = { 0, Hg * K, 2 * Hg * K, conv_dim };
    for (int r = 0; r < 3; ++r) {
        const int64_t lo = c0 > edges[r] ? c0 : edges[r];
        const int64_t hi = c0 + cbw < edges[r + 1] ? c0 + cbw : edges[r + 1];
        const int64_t kd = (r < 2) ? K : V;
        for (int64_t ch = lo; ch < hi;) {
            const int64_t oh = (ch - edges[r]) / kd, d0 = (ch - edges[r]) % kd;
            const int64_t len = hi - ch < kd - d0 ? hi - ch : kd - d0;
            if (ovs[r].s2 == 1)
                row_from_f32(outs[r]->dtype, acc + (ch - c0),
                             (void*)byte_at(outs[r]->data, outs[r]->dtype,
                                            at3(ovs[r], bos + t, oh, d0)),
                             len);
            else
                for (int64_t d = 0; d < len; ++d)
                    stt(outs[r], at3(ovs[r], bos + t, oh, d0 + d), acc[ch - c0 + d]);
            ch += len;
        }
    }
}

/* ------------------------------------------------------------------ gdn_conv_prep */
/* Everything between the qkv projection and the chunked scan: the width-`conv_width` depthwise
 * causal convolution with silu and its state cache, the q/k/v split, the L2 norm over each q and k
 * head, the gate with its per-chunk cumsum, and beta. */
AVX_KERNEL(gdn_conv_prep) {
    if (a->n_t < 16) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *cs = &a->t[5], *cu = &a->t[6];
    const RadTensor *q = &a->t[11], *k = &a->t[12], *v = &a->t[13];
    const RadTensor *g = &a->t[14], *beta = &a->t[15];
    const RadTensor* bias = rad_arg_in(a, 2);
    const RadTensor* A_log = rad_arg_in(a, 3);
    const RadTensor* dt_bias = rad_arg_in(a, 4);
    const RadTensor* ta = rad_arg_in(a, 7);
    const RadTensor* tb = rad_arg_in(a, 8);
    const RadTensor* sidx = rad_arg_in(a, 9);
    const RadTensor* hini = rad_arg_in(a, 10);
    if (!x->data || !w->data || !cu->data || !q->data || !k->data || !v->data ||
        !g->data || !beta->data) return RAD_E_INVAL;
    /* Both or neither: with only one, the other would be read from a fused row that may not
     * exist. */
    if ((ta != nullptr) != (tb != nullptr)) return RAD_E_INVAL;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    /* `chunk` is RAD_DERIVED but no operand's shape records it, and the gate cumsum restarts at
     * each chunk boundary, so a missing value is refused rather than defaulted. */
    const int64_t chunk = p_int(a, "chunk", 0), width = p_int(a, "conv_width", 0);
    const float l2eps = p_f32(a, "l2_eps", 1e-6f);
    const float spthr = p_f32(a, "softplus_thr", 20.0f);
    if (K <= 0 || V <= 0 || chunk <= 0 || width <= 0) return RAD_E_SHAPE;
    if (width - 1 > AVX_CONV_HIST_MAX) return RAD_E_UNSUPPORTED;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_E_SHAPE;                  /* zero rows is not a call (docs/OPS.md) */
    const int64_t Hg = numel(q) / (T * K);
    const int64_t H = numel(v) / (T * V);
    if (Hg <= 0 || H <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    const int64_t conv_dim = 2 * Hg * K + H * V;
    if (numel(w) < conv_dim * width) return RAD_E_SHAPE;
    if (numel(g) < T * H || numel(beta) < T * H) return RAD_E_SHAPE;

    const int64_t xrow = x->rank >= 2 ? x->stride[x->rank - 2] : (conv_dim + (ta ? 0 : 2 * H));
    const int64_t xcol = laststride(x);
    /* Bound by the view, not numel: `x` may be a column slice of a wider row, whose numel counts
     * only the named elements. At rank >= 2 the shape gives the extent. */
    const int64_t xneed = conv_dim + (ta ? 0 : 2 * H);
    if (x->rank >= 2) {
        if (x->shape[x->rank - 1] < xneed || x->shape[x->rank - 2] < T) return RAD_E_SHAPE;
    } else if (numel(x) < (T - 1) * xrow + xneed) {
        return RAD_E_SHAPE;
    }
    if (ta && (numel(ta) < T * H || numel(tb) < T * H)) return RAD_E_SHAPE;

    int64_t state_len = 0;
    const V3 cvs = conv_state_view(cs, conv_dim, width, &state_len);
    if (cs->data && state_len < width - 1) return RAD_E_SHAPE;

    const V3 qv = view3(q, Hg, K), kv = view3(k, Hg, K), vv = view3(v, H, V);
    const V2 wv = view2(w, width);
    const V2 gv = view2(g, H), bv = view2(beta, H);
    const V2 av = ta ? view2(ta, H) : V2{ nullptr, 0, 0 };
    const V2 bgv = tb ? view2(tb, H) : V2{ nullptr, 0, 0 };

    /* ---- phase 1: the convolution, straight into q / k / v ---------------------------------- */
    /* Tokens outer, channels vectorised over contiguous rows (channel-outer would read x
     * strided, one cache line per element, with a dtype switch and scalar exp each). Per element
     * the order is bias, taps ascending, silu last, with separate mul and add rather than FMA, to
     * keep the reference's roundings. vf_silu is ~1 ulp from libm, below what the bf16 store
     * keeps. Panels are 1024 channels; store_pieces handles edges inside a head. */
    const int64_t CB = 1024;
    const int64_t npan = (conv_dim + CB - 1) / CB;
    AVX_PARALLEL_FOR_IF(N * npan >= 4)
    for (int64_t ncb = 0; ncb < N * npan; ++ncb) {
        const int64_t n = ncb / npan, pan = ncb % npan;
        const int64_t c0 = pan * CB, cbw = (c0 + CB < conv_dim) ? CB : conv_dim - c0;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        if (slen <= 0) continue;
        const int64_t slot = sidx ? ldi(sidx, offlin(sidx, n)) : n;
        /* `has_init` zero: the slot holds no history and the window before the first token is
         * zeros, so a new sequence on a recycled slot does not read the previous one's tail. */
        const bool warm = !hini || ldi(hini, offlin(hini, n)) != 0;

        float* acc = scratch_f32((size_t)cbw);
        float* xr = scratch_f32_b((size_t)cbw);
        /* Taps transposed to [width][cbw], widened once per panel. */
        float* tapT = scratch_f32_c((size_t)width * (size_t)cbw);
        fill_tapT(w, wv, c0, cbw, width, tapT);
        /* Bias, widened once per panel into the raw scratch slot (the three f32 slots are in
         * use); each token's accumulator starts as a copy. */
        float* bs = (float*)scratch_raw((size_t)cbw * sizeof(float) + 64);
        if (bias) {
            if (laststride(bias) == 1)
                row_to_f32(bias->dtype, byte_at(bias->data, bias->dtype, c0), bs, cbw);
            else
                for (int64_t c = 0; c < cbw; ++c) bs[c] = ldt(bias, (c0 + c) * laststride(bias));
        } else {
            for (int64_t c = 0; c < cbw; ++c) bs[c] = 0.0f;
        }

        for (int64_t t = 0; t < slen; ++t) {
            /* Per token: bias, then taps ascending. */
            int64_t c = 0;
            for (; c + VF_N <= cbw; c += VF_N) vf_storeu(acc + c, vf_loadu(bs + c));
            for (; c < cbw; ++c) acc[c] = bs[c];
            for (int64_t j = 0; j < width; ++j) {
                const int64_t src = t - (width - 1) + j;
                if (src >= 0) {
                    if (xcol == 1)
                        row_to_f32(x->dtype,
                                   byte_at(x->data, x->dtype, (bos + src) * xrow + c0), xr,
                                   cbw);
                    else
                        for (int64_t c = 0; c < cbw; ++c)
                            xr[c] = ldt(x, (bos + src) * xrow + (c0 + c) * xcol);
                } else if (warm && cs->data && slot >= 0) {
                    if (cvs.s1 == 1)
                        row_to_f32(cs->dtype,
                                   byte_at(cs->data, cs->dtype, at3(cvs, slot, c0, src + width - 1)),
                                   xr, cbw);
                    else
                        for (int64_t c = 0; c < cbw; ++c)
                            xr[c] = ldt(cs, at3(cvs, slot, c0 + c, src + width - 1));
                } else {
                    std::memset(xr, 0, (size_t)cbw * sizeof(float));
                }
                /* Separate mul and add, not FMA, to match the reference bit for bit. */
                const float* tc = tapT + j * cbw;
                c = 0;
                for (; c + VF_N <= cbw; c += VF_N)
                    vf_storeu(acc + c, vf_add(vf_mul(vf_loadu(xr + c), vf_loadu(tc + c)),
                                              vf_loadu(acc + c)));
                for (; c < cbw; ++c) acc[c] += xr[c] * tc[c];
            }
            /* Silu in place, then split into q/k/v. */
            c = 0;
            for (; c + VF_N <= cbw; c += VF_N)
                vf_storeu(acc + c, vf_silu(vf_loadu(acc + c)));
            for (; c < cbw; ++c) acc[c] /= (1.0f + sc_exp(-acc[c]));
            store_pieces(acc, cbw, c0, bos, t, q, k, v, qv, kv, vv, Hg, K, V, conv_dim);
        }
    }

    /* ---- phase 2: the L2 norm over each q and k head, reading the rounded conv output -------- */
    AVX_PARALLEL_FOR_IF(T * Hg * 2 >= 32)
    for (int64_t th = 0; th < T * Hg * 2; ++th) {
        const int64_t which = th / (T * Hg);
        const int64_t rest = th % (T * Hg);
        const int64_t t = rest / Hg, h = rest % Hg;
        const RadTensor* z = which ? k : q;
        const V3& zv = which ? kv : qv;
        const int64_t b = at3(zv, t, h, 0);
        float* buf = scratch_f32((size_t)K);
        const float* u = (zv.s2 == 1) ? row_f32_in(z->dtype, byte_at(z->data, z->dtype, b), buf, K)
                                      : buf;
        if (zv.s2 != 1) for (int64_t d = 0; d < K; ++d) buf[d] = ldt(z, b + d * zv.s2);
        /* L2 norm, not RMS: no division by K. */
        const float inv = 1.0f / std::sqrt(row_sumsq(u, K) + l2eps);
        if (zv.s2 == 1) {
            float* o2 = scratch_f32_b((size_t)K);
            const vf vi = vf_set1(inv);
            int64_t d = 0;
            for (; d + VF_N <= K; d += VF_N) vf_storeu(o2 + d, vf_mul(vf_loadu(u + d), vi));
            for (; d < K; ++d) o2[d] = u[d] * inv;
            out_row(z, b, K, o2);
        } else {
            for (int64_t d = 0; d < K; ++d) stt(z, b + d * zv.s2, buf[d] * inv);
        }
    }

    /* ---- phase 3: the gate, its per-chunk cumsum, and beta ----------------------------------- */
    /* The cumsum restarts per chunk and per sequence, as the scan consumes it: g[j] is the decay
     * from the start of j's chunk through j. A running sum, so sequential in t. */
    AVX_PARALLEL_FOR_IF(N * H >= 4)
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t n = nh / H, h = nh % H;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        const float alog = A_log ? ldt(A_log, h * laststride(A_log)) : 0.0f;
        const float dtb = dt_bias ? ldt(dt_bias, h * laststride(dt_bias)) : 0.0f;
        const float negexp = -sc_exp(alog);
        float run = 0.0f;
        for (int64_t t = 0; t < slen; ++t) {
            if (t % chunk == 0) run = 0.0f;
            const float av_ = ta ? ldt(ta, at2(av, bos + t, h))
                                 : ldt(x, (bos + t) * xrow + (conv_dim + h) * xcol);
            const float bvl = tb ? ldt(tb, at2(bgv, bos + t, h))
                                 : ldt(x, (bos + t) * xrow + (conv_dim + H + h) * xcol);
            run += negexp * softplus(av_ + dtb, spthr);
            stt(g, at2(gv, bos + t, h), run);
            stt(beta, at2(bv, bos + t, h), 1.0f / (1.0f + sc_exp(-bvl)));
        }
    }

    /* ---- phase 4: rewrite the conv state from the sequence's last width-1 tokens -------------- */
    /* After phase 1, which has read the old history. A sequence shorter than width-1 keeps the
     * old state ahead of its tokens, so a sequence's next prefill chunk continues from this one. */
    if (cs->data && width > 1) {
        AVX_PARALLEL_FOR_IF(N * conv_dim >= 32)
        for (int64_t nc = 0; nc < N * conv_dim; ++nc) {
            const int64_t n = nc / conv_dim, c = nc % conv_dim;
            const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
            const int64_t slot = sidx ? ldi(sidx, offlin(sidx, n)) : n;
            if (slen <= 0 || slot < 0) continue;
            const bool warm = !hini || ldi(hini, offlin(hini, n)) != 0;
            float keep[AVX_CONV_HIST_MAX];
            const int64_t st = width - 1;
            for (int64_t i = 0; i < st; ++i) {
                const int64_t src = slen - (width - 1) + i;
                keep[i] = src >= 0 ? ldt(x, (bos + src) * xrow + c * xcol)
                                   : (warm ? ldt(cs, at3(cvs, slot, c, src + width - 1)) : 0.0f);
            }
            for (int64_t i = 0; i < st; ++i) stt(cs, at3(cvs, slot, c, i), keep[i]);
        }
    }
    return RAD_OK;
}

/* Silu in place over a channel panel, then store_pieces. */
static inline void silu_store_pieces(float* acc, int64_t cbw, int64_t c0, int64_t bos, int64_t t,
                                     const RadTensor* q, const RadTensor* k, const RadTensor* v,
                                     V3 qv, V3 kv, V3 vv, int64_t Hg, int64_t K, int64_t V,
                                     int64_t conv_dim) {
    /* Exact silu only: vf_silu_fast (as silu_mul uses for bf16 outputs) made no measurable
     * difference here, since the token loop is frontend-bound. */
    int64_t c = 0;
    for (; c + VF_N <= cbw; c += VF_N) vf_storeu(acc + c, vf_silu(vf_loadu(acc + c)));
    for (; c < cbw; ++c) acc[c] /= (1.0f + sc_exp(-acc[c]));
    store_pieces(acc, cbw, c0, bos, t, q, k, v, qv, kv, vv, Hg, K, V, conv_dim);
}

/* ------------------------------------------------------------------ gdn_conv_update */
/* The convolution for a decode step against the rolling window: the read starts at
 * num_accepted - 1 and the rewrite shifts the window down by one (the rollback contract in the
 * file header).
 *
 * No L2 norm here; on the decode path the recurrent update applies it, as in libr4d. */
AVX_KERNEL(gdn_conv_update) {
    if (a->n_t < 10) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *cs = &a->t[3], *cu = &a->t[6];
    const RadTensor *q = &a->t[7], *k = &a->t[8], *v = &a->t[9];
    const RadTensor* bias = rad_arg_in(a, 2);
    const RadTensor* sidx = rad_arg_in(a, 4);
    const RadTensor* nacc = rad_arg_in(a, 5);
    if (!x->data || !w->data || !cs->data || !cu->data || !q->data || !k->data || !v->data)
        return RAD_E_INVAL;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const int64_t width = p_int(a, "conv_width", 0);
    if (K <= 0 || V <= 0 || width <= 0) return RAD_E_SHAPE;
    if (width - 1 > AVX_CONV_HIST_MAX) return RAD_E_UNSUPPORTED;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_E_SHAPE;
    const int64_t Hg = numel(q) / (T * K);
    const int64_t H = numel(v) / (T * V);
    if (Hg <= 0 || H <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    const int64_t conv_dim = 2 * Hg * K + H * V;
    if (numel(w) < conv_dim * width) return RAD_E_SHAPE;

    const int64_t xrow = x->rank >= 2 ? x->stride[x->rank - 2] : conv_dim;
    const int64_t xcol = laststride(x);
    int64_t state_len = 0;
    const V3 cvs = conv_state_view(cs, conv_dim, width, &state_len);

    /* A state too shallow for this step's offset is refused, not clamped (that would silently
     * truncate history). Checked before the loop; an OpenMP body cannot return. */
    for (int64_t n = 0; n < N; ++n) {
        const int64_t slen = cu_at(cu, n + 1) - cu_at(cu, n);
        const int64_t off = nacc ? ldi(nacc, offlin(nacc, n)) - 1 : 0;
        if (off < 0) return RAD_E_INVAL;
        if (off + width - 1 > state_len) return RAD_E_SHAPE;
        if (width > 1 && (width - 2) + slen > state_len) return RAD_E_SHAPE;
    }

    const V3 qv = view3(q, Hg, K), kv = view3(k, Hg, K), vv = view3(v, H, V);
    const V2 wv = view2(w, width);

    /* Tokens outer, channels vectorised, as in gdn_conv_prep (at decode slen is 1 but conv_dim
     * is 10240). */
    const int64_t CB = 1024;
    const int64_t npan = (conv_dim + CB - 1) / CB;
    AVX_PARALLEL_FOR_IF(N * npan >= 4)
    for (int64_t ncb = 0; ncb < N * npan; ++ncb) {
        const int64_t n = ncb / npan, pan = ncb % npan;
        const int64_t c0 = pan * CB, cbw = (c0 + CB < conv_dim) ? CB : conv_dim - c0;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        if (slen <= 0) continue;
        const int64_t slot = sidx ? ldi(sidx, offlin(sidx, n)) : n;
        if (slot < 0) continue;
        const int64_t off = nacc ? ldi(nacc, offlin(nacc, n)) - 1 : 0;

        float* acc = scratch_f32((size_t)cbw);
        float* xr = scratch_f32_b((size_t)cbw);
        float* tapT = scratch_f32_c((size_t)width * (size_t)cbw);
        fill_tapT(w, wv, c0, cbw, width, tapT);
        /* One raw block (single scratch slot): the bias panel, then the width-1 history rows of
         * this panel, widened once. Only the first width-1 tokens read history. */
        float* raw = (float*)scratch_raw(((size_t)cbw + (size_t)(width - 1) * (size_t)cbw) *
                                         sizeof(float) + 64);
        float* bs = raw;
        float* hseg = raw + cbw;
        if (bias) {
            if (laststride(bias) == 1)
                row_to_f32(bias->dtype, byte_at(bias->data, bias->dtype, c0), bs, cbw);
            else
                for (int64_t c = 0; c < cbw; ++c) bs[c] = ldt(bias, (c0 + c) * laststride(bias));
        } else {
            for (int64_t c = 0; c < cbw; ++c) bs[c] = 0.0f;
        }
        for (int64_t i = 0; i < width - 1; ++i) {
            if (cvs.s1 == 1)
                row_to_f32(cs->dtype, byte_at(cs->data, cs->dtype, at3(cvs, slot, c0, off + i)),
                           hseg + i * cbw, cbw);
            else
                for (int64_t c = 0; c < cbw; ++c)
                    hseg[i * cbw + c] = ldt(cs, at3(cvs, slot, c0 + c, off + i));
        }
        for (int64_t t = 0; t < slen; ++t) {
            int64_t c = 0;
            for (; c + VF_N <= cbw; c += VF_N) vf_storeu(acc + c, vf_loadu(bs + c));
            for (; c < cbw; ++c) acc[c] = bs[c];
            for (int64_t j = 0; j < width; ++j) {
                const int64_t src = t - (width - 1) + j;
                if (src >= 0) {
                    if (xcol == 1)
                        row_to_f32(x->dtype,
                                   byte_at(x->data, x->dtype, (bos + src) * xrow + c0), xr,
                                   cbw);
                    else
                        for (int64_t c = 0; c < cbw; ++c)
                            xr[c] = ldt(x, (bos + src) * xrow + (c0 + c) * xcol);
                } else {
                    /* src + width - 1 is in [0, width - 2]: the history segment. */
                    const float* hs = hseg + (src + width - 1) * cbw;
                    for (int64_t c = 0; c < cbw; ++c) xr[c] = hs[c];
                }
                /* Separate mul and add, not FMA, to match the reference's two roundings. */
                const float* tc = tapT + j * cbw;
                c = 0;
                for (; c + VF_N <= cbw; c += VF_N)
                    vf_storeu(acc + c, vf_add(vf_mul(vf_loadu(xr + c), vf_loadu(tc + c)),
                                              vf_loadu(acc + c)));
                for (; c < cbw; ++c) acc[c] += xr[c] * tc[c];
            }
            silu_store_pieces(acc, cbw, c0, bos, t, q, k, v, qv, kv, vv, Hg, K, V, conv_dim);
        }
        /* Shift down by one: the tail of the window this step read (at `off`, not zero), then
         * this step's tokens. Slot i takes slot off + i + 1, which no earlier iteration has
         * written, so the ascending walk is safe in place. Byte moves when both sides are
         * channel-contiguous with the same dtype; scalar otherwise. */
        if (width > 1) {
            const uint32_t csbits = rad_dtype_bits(cs->dtype);
            const size_t elbytes = (size_t)csbits / 8;
            /* memmove: the ranges can overlap when the history stride is short. Whole-byte
             * dtypes only. */
            if (cvs.s1 == 1 && xcol == 1 && x->dtype == cs->dtype && csbits % 8 == 0 &&
                elbytes > 0) {
                for (int64_t i = 0; i + 1 < width - 1; ++i)
                    std::memmove((char*)cs->data + (at3(cvs, slot, c0, i) * (int64_t)elbytes),
                                 (const char*)cs->data +
                                     (at3(cvs, slot, c0, off + i + 1) * (int64_t)elbytes),
                                 (size_t)cbw * elbytes);
                for (int64_t t = 0; t < slen; ++t)
                    std::memmove((char*)cs->data +
                                     (at3(cvs, slot, c0, t + width - 2) * (int64_t)elbytes),
                                 (const char*)x->data +
                                     (((bos + t) * xrow + c0) * (int64_t)elbytes),
                                 (size_t)cbw * elbytes);
            } else {
                for (int64_t c = 0; c < cbw; ++c) {
                    for (int64_t i = 0; i + 1 < width - 1; ++i)
                        stt(cs, at3(cvs, slot, c0 + c, i),
                            ldt(cs, at3(cvs, slot, c0 + c, off + i + 1)));
                    for (int64_t t = 0; t < slen; ++t)
                        stt(cs, at3(cvs, slot, c0 + c, t + width - 2),
                            ldt(x, (bos + t) * xrow + (c0 + c) * xcol));
                }
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_kkt_solve */
/* A = (I + strict_lower(diag(beta) . K K^T . e^{g_i - g_j}))^{-1}, per chunk, per v head. The
 * chunk's k rows, `g` and `beta` are widened once (see the file header). */
AVX_KERNEL(gdn_kkt_solve) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *k = &a->t[0], *beta = &a->t[1], *g = &a->t[2], *cu = &a->t[3], *A = &a->t[4];

    const int64_t K = p_int(a, "head_k", 0);
    /* RAD_DERIVED: taken from A's shape [T, heads, chunk] when absent. */
    const int64_t chunk = p_int(a, "chunk", A->rank >= 3 ? A->shape[A->rank - 1] : 0);
    if (K <= 0 || chunk <= 0) return RAD_E_SHAPE;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_E_SHAPE;
    const int64_t H = numel(beta) / T;
    const int64_t Hg = numel(k) / (T * K);
    if (H <= 0 || Hg <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    if (numel(A) < T * H * chunk) return RAD_E_SHAPE;
    const int64_t grp = H / Hg;

    const V3 kv = view3(k, Hg, K), Av = view3(A, H, chunk);
    const V2 gv = view2(g, H), bv = view2(beta, H);

    AVX_PARALLEL_FOR_IF(N * H >= 4)
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t n = nh / H, h = nh % H, hk = h / grp;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        /* One scratch call, carved by hand; a second could reallocate the first. */
        const size_t need = (size_t)(chunk * chunk + chunk + chunk * K + 2 * chunk + chunk) + 16;
        float* wsp = scratch_f32(need);
        float* Am = wsp;                          /* [chunk, chunk] the inverse, in f32 */
        float* row = Am + chunk * chunk;          /* [chunk] one row of M */
        float* kp = row + chunk;                  /* [chunk, K] the chunk's k rows */
        float* gp = kp + chunk * K;               /* [chunk] */
        float* bp = gp + chunk;                   /* [chunk] */
        float* ev = bp + chunk;                   /* [chunk] e^{g_i - g_j} for this i */

        for (int64_t c0 = 0; c0 < slen; c0 += chunk) {
            const int64_t L = (c0 + chunk < slen) ? chunk : slen - c0;
            panel3(k, kv, bos + c0, L, hk, K, kp);
            for (int64_t i = 0; i < L; ++i) {
                gp[i] = ldt(g, at2(gv, bos + c0 + i, h));
                bp[i] = ldt(beta, at2(bv, bos + c0 + i, h));
            }
            for (int64_t i = 0; i < L; ++i) {
                const int64_t ti = bos + c0 + i;
                /* e^{g_i - g_j} over the row with vf_exp (~1 ulp from libm, within tolerance). */
                {
                    const vf vg = vf_set1(gp[i]);
                    int64_t j = 0;
                    for (; j + VF_N <= i; j += VF_N)
                        vf_storeu(ev + j, vf_exp(vf_sub(vg, vf_loadu(gp + j))));
                    for (; j < i; ++j) ev[j] = sc_exp(gp[i] - gp[j]);
                }
                for (int64_t j = 0; j < i; ++j)
                    row[j] = bp[i] * dot2(kp + i * K, kp + j * K, K) * ev[j];
                /* (I + M) A = I, row by row: A[i][j] = delta - sum_{j<=l<i} M[i][l] A[l][j].
                 * l outer so the run over j is contiguous vector work. Each (i, j) lane still
                 * subtracts in ascending l with mul+sub (not FMA), so the result is bit-identical
                 * to the scalar loop. Terms with l < j read Am[l][j] = 0, so starting the sum at
                 * l = 0 changes nothing. */
                for (int64_t j = 0; j <= i; ++j) Am[i * chunk + j] = (i == j) ? 1.0f : 0.0f;
                for (int64_t l = 0; l < i; ++l) {
                    const vf rl = vf_set1(row[l]);
                    const float* Aml = Am + l * chunk;
                    float* Ami = Am + i * chunk;
                    int64_t j = 0;
                    for (; j + VF_N <= i + 1; j += VF_N)
                        vf_storeu(Ami + j,
                                  vf_sub(vf_loadu(Ami + j), vf_mul(rl, vf_loadu(Aml + j))));
                    for (; j <= i; ++j) Ami[j] -= row[l] * Aml[j];
                }
                for (int64_t j = i + 1; j < chunk; ++j) Am[i * chunk + j] = 0.0f;
                if (Av.s2 == 1)
                    row_from_f32(A->dtype, Am + i * chunk,
                                 (void*)byte_at(A->data, A->dtype, at3(Av, ti, h, 0)), chunk);
                else
                    for (int64_t j = 0; j < chunk; ++j)
                        stt(A, at3(Av, ti, h, j), Am[i * chunk + j]);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_chunk_scan */
AVX_KERNEL(gdn_chunk_scan) {
    if (a->n_t < 10) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *A = &a->t[3];
    const RadTensor *g = &a->t[4], *beta = &a->t[5], *cu = &a->t[7];
    const RadTensor *o = &a->t[8], *ht = &a->t[9];
    const RadTensor* h0 = rad_arg_in(a, 6);
    /* Optional state slot per sequence, for a state pool not indexed by batch position; absent,
     * sequence n owns slot n. */
    const RadTensor* sidx = rad_arg_in(a, 10);
    if (!q->data || !k->data || !v->data || !A->data || !g->data || !beta->data ||
        !cu->data || !o->data) return RAD_E_INVAL;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const int64_t chunk = p_int(a, "chunk", A->rank >= 3 ? A->shape[A->rank - 1] : 0);
    const float scale = p_f32(a, "scale", 1.0f);
    if (K <= 0 || V <= 0 || chunk <= 0) return RAD_E_SHAPE;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_E_SHAPE;
    const int64_t H = numel(beta) / T;
    const int64_t Hg = numel(q) / (T * K);
    if (H <= 0 || Hg <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    const int64_t grp = H / Hg;
    if (numel(o) < T * H * V || numel(A) < T * H * chunk) return RAD_E_SHAPE;
    if (!sidx) {
        if (ht->data && numel(ht) < N * H * V * K) return RAD_E_SHAPE;
        if (h0 && numel(h0) < N * H * V * K) return RAD_E_SHAPE;
    } else if (sidx->rank < 1 || sidx->shape[0] < N) {
        return RAD_E_SHAPE;
    }

    const V3 qv = view3(q, Hg, K), kv = view3(k, Hg, K), vv = view3(v, H, V), ov = view3(o, H, V);
    const V3 Av = view3(A, H, chunk);
    const V2 gv = view2(g, H), bv = view2(beta, H);
    const V4 h0v = h0 ? view4(h0, H, V, K) : V4{ nullptr, 0, 0, 0, 0 };
    const V4 htv = ht->data ? view4(ht, H, V, K) : V4{ nullptr, 0, 0, 0, 0 };

    AVX_PARALLEL_FOR_IF(N * H >= 4)
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t n = nh / H, h = nh % H, hk = h / grp;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        /* Column 0 of state_idx is the slot. A row may be wider (a speculative decode kernel
         * keeps more slots beside it); the scan writes only this one. */
        const int64_t sw = sidx && sidx->rank >= 1 ? sidx->stride[0] : 1;
        const int64_t sn = sidx ? ldi(sidx, n * sw) : n;
        if (sn < 0) continue;                  /* padded row: no state */

        /* One scratch request, carved by hand: the q/k/v panels, plus Ap/expv holding the A
         * weights and gate exponentials once per chunk (scalar exps in the O loop were most of the
         * time at model shape). */
        const size_t need = (size_t)(V * K + 3 * chunk * V + 2 * chunk * K + 3 * chunk +
                                     V + K + chunk * chunk + chunk + K * V) + 32;
        float* wsp = scratch_f32(need);
        float* S = wsp;                            /* [V, K] the recurrent state */
        float* Wm = S + V * K;                     /* [chunk, V] */
        float* Vp = Wm + chunk * V;                /* [chunk, V] */
        float* row = Vp + chunk * V;               /* [chunk] one row of sA */
        float* qp = row + chunk;                   /* [chunk, K] */
        float* kp = qp + chunk * K;                /* [chunk, K] */
        float* vp = kp + chunk * K;                /* [chunk, V] */
        float* gp = vp + chunk * V;                /* [chunk] */
        float* bp = gp + chunk;                    /* [chunk] */
        /* Own buffers for the output row and the decayed key, not aliases of `Wm` or `row`,
         * which would only fit while K <= chunk, a condition the op does not promise. */
        float* acc = bp + chunk;                   /* [V] the output row being formed */
        float* gk = acc + V;                       /* [K] e^{g_last - g_j} . k[j] */
        float* Ap = gk + K;                        /* [chunk, chunk] A weights, one row per i */
        float* expv = Ap + chunk * chunk;          /* [chunk] e^{g_j} */

        /* Initial state, widened a row at a time. */
        if (h0 && h0v.s3 == 1) {
            for (int64_t e = 0; e < V; ++e)
                row_to_f32(h0->dtype,
                           byte_at(h0->data, h0->dtype, at4(h0v, sn, h, e, 0)), S + e * K, K);
        } else {
            for (int64_t e = 0; e < V; ++e)
                for (int64_t d = 0; d < K; ++d)
                    S[e * K + d] = h0 ? ldt(h0, at4(h0v, sn, h, e, d)) : 0.0f;
        }

        for (int64_t c0 = 0; c0 < slen; c0 += chunk) {
            const int64_t L = (c0 + chunk < slen) ? chunk : slen - c0;
            panel3(q, qv, bos + c0, L, hk, K, qp);
            panel3(k, kv, bos + c0, L, hk, K, kp);
            panel3(v, vv, bos + c0, L, h, V, vp);
            for (int64_t j = 0; j < L; ++j) {
                gp[j] = ldt(g, at2(gv, bos + c0 + j, h));
                bp[j] = ldt(beta, at2(bv, bos + c0 + j, h));
            }
            const float glast = gp[L - 1];
            /* e^{g_j} once per j, vectorised (vf_exp is ~1 ulp from libm, within tolerance). */
            {
                int64_t j = 0;
                for (; j + VF_N <= L; j += VF_N)
                    vf_storeu(expv + j, vf_exp(vf_loadu(gp + j)));
                for (; j < L; ++j) expv[j] = sc_exp(gp[j]);
            }
            /* The chunk's A rows, widened once for the triangle below. */
            if (Av.s2 == 1) {
                for (int64_t i = 0; i < L; ++i)
                    row_to_f32(A->dtype,
                               byte_at(A->data, A->dtype, at3(Av, bos + c0 + i, h, 0)),
                               Ap + i * chunk, chunk);
            }

            /* W = beta . (V - e^g . (K S)). Transposed-S k-outer forms (streaming Wm, or
             * register-blocked) measured slower than this dot loop. */
            for (int64_t j = 0; j < L; ++j) {
                const float ge = expv[j];
                for (int64_t e = 0; e < V; ++e)
                    Wm[j * V + e] =
                        bp[j] * (vp[j * V + e] - ge * dot2(kp + j * K, S + e * K, K));
            }
            /* V' = A W; A is unit lower triangular, so only j <= i contributes. Vectorised over e,
             * summing in libref's j order. */
            for (int64_t i = 0; i < L; ++i) {
                const int64_t ti = bos + c0 + i;
                for (int64_t e = 0; e < V; ++e) Vp[i * V + e] = 0.0f;
                if (Av.s2 == 1) {
                    for (int64_t j = 0; j <= i; ++j)
                        axpy(Vp + i * V, Wm + j * V, Ap[i * chunk + j], V);
                } else {
                    for (int64_t j = 0; j <= i; ++j)
                        axpy(Vp + i * V, Wm + j * V, ldt(A, at3(Av, ti, h, j)), V);
                }
            }
            /* O = scale . e^{g_i} . (Q S^T) + sA V' */
            for (int64_t i = 0; i < L; ++i) {
                const int64_t ti = bos + c0 + i;
                const float gi = gp[i];
                const float ge = sc_exp(gi);
                /* e^{gi - g_j} over the row, vectorised, before the dots. */
                {
                    const vf vg = vf_set1(gi), vs = vf_set1(scale);
                    int64_t j = 0;
                    for (; j + VF_N <= i + 1; j += VF_N)
                        vf_storeu(row + j,
                                  vf_mul(vs, vf_exp(vf_sub(vg, vf_loadu(gp + j)))));
                    for (; j <= i; ++j) row[j] = scale * sc_exp(gi - gp[j]);
                }
                for (int64_t j = 0; j <= i; ++j)
                    row[j] *= dot2(qp + i * K, kp + j * K, K);
                for (int64_t e = 0; e < V; ++e)
                    acc[e] = scale * ge * dot2(qp + i * K, S + e * K, K);
                for (int64_t j = 0; j <= i; ++j) axpy(acc, Vp + j * V, row[j], V);
                if (ov.s2 == 1) out_row(o, at3(ov, ti, h, 0), V, acc);
                else for (int64_t e = 0; e < V; ++e) stt(o, at3(ov, ti, h, e), acc[e]);
            }
            /* S = e^{g_last} S + V'^T (e^{g_last - g_j} . K) */
            {
                const float el = sc_exp(glast);
                for (int64_t e = 0; e < V; ++e) {
                    const vf ve = vf_set1(el);
                    int64_t d = 0;
                    for (; d + VF_N <= K; d += VF_N)
                        vf_storeu(S + e * K + d, vf_mul(vf_loadu(S + e * K + d), ve));
                    for (; d < K; ++d) S[e * K + d] *= el;
                }
                /* e^{g_last - g_j} over the chunk, vectorised into `row` (free here). */
                {
                    const vf vg = vf_set1(glast);
                    int64_t j = 0;
                    for (; j + VF_N <= L; j += VF_N)
                        vf_storeu(row + j, vf_exp(vf_sub(vg, vf_loadu(gp + j))));
                    for (; j < L; ++j) row[j] = sc_exp(glast - gp[j]);
                }
                for (int64_t j = 0; j < L; ++j) {
                    const float gs = row[j];
                    /* gs * k[j] formed once per j, in libref's association `Vp * (gs * k)`;
                     * (Vp * gs) * k would differ in the last bits. */
                    const vf vg = vf_set1(gs);
                    int64_t d = 0;
                    for (; d + VF_N <= K; d += VF_N)
                        vf_storeu(gk + d, vf_mul(vf_loadu(kp + j * K + d), vg));
                    for (; d < K; ++d) gk[d] = kp[j * K + d] * gs;
                    for (int64_t e = 0; e < V; ++e) axpy(S + e * K, gk, Vp[j * V + e], K);
                }
            }
        }

        if (ht->data) {
            if (htv.s3 == 1) {
                for (int64_t e = 0; e < V; ++e)
                    row_from_f32(ht->dtype, S + e * K,
                                 (void*)byte_at(ht->data, ht->dtype, at4(htv, sn, h, e, 0)),
                                 K);
            } else {
                for (int64_t e = 0; e < V; ++e)
                    for (int64_t d = 0; d < K; ++d)
                        stt(ht, at4(htv, sn, h, e, d), S[e * K + d]);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the anchor state form
 *
 * docs/OPS.md `state_form`, libref's reading (ref_gdn.cpp) of libr4d's layout: under `anchor` a
 * sequence keeps one committed state, and each step records its tokens' rank-1 update factors
 * beside it; the next step replays the first `num_accepted` of them onto the state before running
 * its own tokens.
 *
 *   state_idx  six columns: 0..2 the anchor, 3..5 the factor block. A paged caller's window can
 *              shift by -1, 0 or +1 pages a step; the anchor is published at column 2 and each
 *              factor copy is stamped with its column (1, 2, 3), so the next step reads the stamp
 *              at column 4 and the anchor at the column it implies. A zeroed block reads as no
 *              shift.
 *   factors    per token, inside the head's [V, K] block of the factor slot: the normalised k once
 *              per 32-row group of V, then v, then (g, beta) once per group, each group's rows
 *              replaying from their own copy; room for `tmax` tokens, the stamp just past them.
 *
 * libr4d's kernel is K = V = 128, where the block holds kTmax = 16 tokens and the stamp sits at
 * 16 * fstr; square heads from 64 up hold 16 too, so there the layout is libr4d's byte for byte.
 * A smaller head's block holds fewer tokens' factors (a 16 x 16 head: 7), so `tmax` is as many as
 * fit before the stamp, and a step or an accepted prefix longer than that is refused (libref's
 * rule, docs/OPS.md). */
namespace {
struct AnchorLayout {
    static constexpr int64_t kTmax = 16;
    int64_t groups, fstr, vb, gb, tmax, mark;
    AnchorLayout(int64_t K, int64_t V)
        : groups((V + 31) / 32), fstr(groups * K + V + 2 * groups), vb(groups * K),
          gb(groups * K + V), tmax(std::min<int64_t>(kTmax, (V * K - 1) / fstr)),
          mark(tmax * fstr) {}
    /* The anchor's column, from the stamp at the factor block `fb`'s `mark`. */
    int anchor_col(const RadTensor* st, int64_t fb) const {
        const int m = (int)(ldt(st, fb + mark) + 0.5f) - 1;
        return 3 - ((m >= 0 && m <= 2) ? m : 1);
    }
};
}  // namespace

/* ------------------------------------------------------------------ gdn_recurrent_update */
/* Decode's counterpart of the chunked scan, against the paged state cache: the initial state is
 * read from the slot the last accepted token wrote, and every candidate token writes its own slot
 * (see the rollback contract), or under `anchor` the replayed state is published and each token's
 * factors are written instead. The gated RMS norm is folded into the epilogue, as in libr4d. */
AVX_KERNEL(gdn_recurrent_update) {
    if (a->n_t < 14) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *av = &a->t[3], *bvt = &a->t[4];
    const RadTensor *state = &a->t[7], *o = &a->t[13];
    const RadTensor* A_log = rad_arg_in(a, 5);
    const RadTensor* dt_bias = rad_arg_in(a, 6);
    const RadTensor* sidx = rad_arg_in(a, 8);
    const RadTensor* nacc = rad_arg_in(a, 9);
    const RadTensor* cu = rad_arg_in(a, 10);
    const RadTensor* z = rad_arg_in(a, 11);
    const RadTensor* nw = rad_arg_in(a, 12);
    /* Optional fused quantiser for the out projection (o_q and o_scale). */
    const RadTensor* oq = (a->n_t > 14 && a->t[14].data) ? &a->t[14] : nullptr;
    const RadTensor* osc = (a->n_t > 15 && a->t[15].data) ? &a->t[15] : nullptr;
    if (!q->data || !k->data || !v->data || !av->data || !bvt->data || !state->data || !o->data ||
        !cu) return RAD_E_INVAL;

    const char* sf = p_str(a, "state_form", "per_candidate");
    const bool anchor = sf && std::strcmp(sf, "anchor") == 0;
    if (!anchor && sf && std::strcmp(sf, "per_candidate") != 0) return RAD_E_UNSUPPORTED;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const float scale = p_f32(a, "scale", 1.0f);
    const float l2eps = p_f32(a, "l2_eps", 1e-6f);
    const float neps = p_f32(a, "eps", 1e-6f);
    const float spthr = p_f32(a, "softplus_thr", 20.0f);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int nact = p_enum(a, "act", kActs, 2, 0);
    if (nact != 0 && nact != 1) return RAD_E_INVAL;
    if (K <= 0 || V <= 0) return RAD_E_SHAPE;

    /* Head and token counts come from v ([T, n_head_v, head_v]); `n_head_v` overrides for a
     * flattened view. */
    int64_t Hv = p_int(a, "n_head_v", 0);
    if (Hv <= 0) Hv = v->rank >= 3 ? v->shape[v->rank - 2] : 0;
    if (Hv <= 0) return RAD_E_SHAPE;
    const int64_t Tk = numel(v) / (Hv * V);
    if (Tk <= 0) return RAD_E_SHAPE;
    const int64_t Hg = numel(q) / (Tk * K);
    if (Hg <= 0 || Hv % Hg != 0) return RAD_E_SHAPE;
    const int64_t grp = Hv / Hg;

    /* The sequence count comes from cu ([N+1]) only. */
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    if (cu_at(cu, n_seq) != Tk) return RAD_E_SHAPE;
    const int64_t sw = sidx ? (sidx->rank >= 2 ? sidx->shape[sidx->rank - 1]
                                               : (numel(sidx) / n_seq > 0 ? numel(sidx) / n_seq
                                                                          : 1))
                            : 1;

    const V3 qv = view3(q, Hg, K), kv = view3(k, Hg, K), vv = view3(v, Hv, V), ov = view3(o, Hv, V);
    const V3 zv = z ? view3_rows(z, Hv, V) : vv;
    const V2 a2 = view2(av, Hv), b2 = view2(bvt, Hv);
    const V2 si = sidx ? view2(sidx, sw) : V2{ nullptr, 0, 0 };
    /* The anchor form keeps its factors inside a head's [V, K] block of a state slot, at libr4d's
     * offsets, so the block must be dense and large enough. Checked before any write. */
    const AnchorLayout al(K, V);
    if (anchor) {
        if (!sidx || sw < 6) return RAD_E_SHAPE;
        if (state->rank >= 4 && (state->stride[3] != 1 || state->stride[2] != K))
            return RAD_E_STRIDE;
        if (al.tmax < 1) return RAD_E_SHAPE;
        for (int64_t n = 0; n < n_seq; ++n) {
            const int64_t na = nacc ? ldi(nacc, offlin(nacc, n)) : 1;
            if (cu_at(cu, n + 1) - cu_at(cu, n) > al.tmax || na > al.tmax) return RAD_E_SHAPE;
        }
    }

    AVX_PARALLEL_FOR_IF(n_seq * Hv >= 4)
    for (int64_t nh = 0; nh < n_seq * Hv; ++nh) {
        const int64_t n = nh / Hv, h = nh % Hv, hk = h / grp;
        const int64_t bos = cu_at(cu, n), q_len = cu_at(cu, n + 1) - bos;
        const int64_t sstride = state->rank >= 4 ? state->stride[0] : (int64_t)Hv * V * K;
        const int64_t hstride = state->rank >= 4 ? state->stride[1] : (int64_t)V * K;
        const int64_t vstride = state->rank >= 4 ? state->stride[2] : K;
        const int64_t dstride = state->rank >= 4 ? state->stride[3] : 1;
        /* Under `anchor` a sequence with no rows this step still folds its accepted prefix in. */
        if (q_len <= 0 && !anchor) continue;
        const int64_t na = nacc ? ldi(nacc, offlin(nacc, n)) : 1;
        const int64_t rd = na > 0 ? na - 1 : 0;
        int64_t col[6] = { 0, 0, 0, 0, 0, 0 };
        int64_t slot_in = 0;
        if (anchor) {
            bool pad = false;
            for (int c = 0; c < 6; ++c) {
                col[c] = ldi(sidx, at2(si, n, c));
                pad = pad || col[c] < 0;
            }
            if (pad) continue;
            slot_in = col[al.anchor_col(state, col[4] * sstride + h * hstride)];
        } else {
            slot_in = sidx ? ldi(sidx, at2(si, n, rd < sw ? rd : sw - 1)) : n;
            if (slot_in < 0) continue;
        }
        /* The factor blocks: column 4's is read, and 3, 4 and 5 are written (each a page the
         * window can shift onto). */
        const int64_t fb = anchor ? col[4] * sstride + h * hstride : 0;
        const int64_t fbs[3] = { anchor ? col[3] * sstride + h * hstride : 0, fb,
                                 anchor ? col[5] * sstride + h * hstride : 0 };
        const bool dup_f = anchor && (col[3] != col[4] || col[5] != col[4]);

        /* vrow/zrow hold the per-token value and gate rows, widened once. */
        const size_t need = (size_t)(V * K + 2 * K + 5 * V) + 16;
        float* wsp = scratch_f32(need);
        float* S = wsp;
        float* qn = S + V * K;
        float* kn = qn + K;
        float* uu = kn + K;
        float* orow = uu + V;
        float* gainv = orow + V;
        float* vrow = gainv + V;
        float* zrow = vrow + V;

        /* Widen the state a row at a time (one dtype switch per row, avx_vec.h). */
        if (dstride == 1) {
            for (int64_t e = 0; e < V; ++e)
                row_to_f32(state->dtype,
                           byte_at(state->data, state->dtype,
                                   slot_in * sstride + h * hstride + e * vstride),
                           S + e * K, K);
        } else {
            for (int64_t e = 0; e < V; ++e)
                for (int64_t d = 0; d < K; ++d)
                    S[e * K + d] = ldt(state, slot_in * sstride + h * hstride + e * vstride +
                                              d * dstride);
        }

        const float alog = A_log ? ldt(A_log, h * laststride(A_log)) : 0.0f;
        const float dtb = dt_bias ? ldt(dt_bias, h * laststride(dt_bias)) : 0.0f;
        const float negexp = -sc_exp(alog);
        /* Norm gain, read once per head. */
        if (nw) {
            if (laststride(nw) == 1) row_to_f32(nw->dtype, nw->data, gainv, V);
            else for (int64_t e = 0; e < V; ++e) gainv[e] = ldt(nw, e * laststride(nw));
        }

        if (anchor) {
            /* Replay the previous step's accepted tokens from their factors, with the token
             * loop's own arithmetic, so the replayed state is the one per_candidate would hold.
             * Then publish it at column 2 and stamp each factor copy with its column. */
            for (int64_t t = 0; t < na; ++t) {
                const int64_t f = fb + t * al.fstr;
                for (int64_t e = 0; e < V; ++e) vrow[e] = ldt(state, f + al.vb + e);
                /* Each 32-row group of V replays from its own copy of k and its own (g, beta), as
                 * libr4d's kernel and libref read them; writers store equal copies. */
                for (int64_t gi = 0; gi < al.groups; ++gi) {
                    const int64_t e0 = 32 * gi, e1 = std::min<int64_t>(V, e0 + 32);
                    for (int64_t d = 0; d < K; ++d) kn[d] = ldt(state, f + gi * K + d);
                    const float gval = ldt(state, f + al.gb + 2 * gi);
                    const float bt = ldt(state, f + al.gb + 2 * gi + 1);
                    const float dec = sc_exp(gval);
                    const vf vdec = vf_set1(dec);
                    for (int64_t e = e0; e < e1; ++e) {
                        float* Se = S + e * K;
                        int64_t d = 0;
                        for (; d + VF_N <= K; d += VF_N)
                            vf_storeu(Se + d, vf_mul(vf_loadu(Se + d), vdec));
                        for (; d < K; ++d) Se[d] *= dec;
                        uu[e] = bt * (vrow[e] - dot2(Se, kn, K));
                    }
                    for (int64_t e = e0; e < e1; ++e) axpy(S + e * K, kn, uu[e], K);
                }
            }
            for (int64_t e = 0; e < V; ++e)        /* the anchor block is dense (checked) */
                row_from_f32(state->dtype, S + e * K,
                             (void*)byte_at(state->data, state->dtype,
                                            col[2] * sstride + h * hstride + e * K),
                             K);
            for (int c = 0; c < 3; ++c) stt(state, fbs[c] + al.mark, (float)(c + 1));
        }

        for (int64_t t = 0; t < q_len; ++t) {
            const int64_t tok = bos + t;
            const float gval = negexp * softplus(ldt(av, at2(a2, tok, h)) + dtb, spthr);
            const float bt = 1.0f / (1.0f + sc_exp(-ldt(bvt, at2(b2, tok, h))));

            if (qv.s2 == 1 && kv.s2 == 1) {
                row_to_f32(q->dtype, byte_at(q->data, q->dtype, at3(qv, tok, hk, 0)), qn, K);
                row_to_f32(k->dtype, byte_at(k->data, k->dtype, at3(kv, tok, hk, 0)), kn, K);
            } else {
                for (int64_t d = 0; d < K; ++d) {
                    qn[d] = ldt(q, at3(qv, tok, hk, d));
                    kn[d] = ldt(k, at3(kv, tok, hk, d));
                }
            }
            /* Value row, widened once per token. */
            if (vv.s2 == 1)
                row_to_f32(v->dtype, byte_at(v->data, v->dtype, at3(vv, tok, h, 0)), vrow, V);
            else
                for (int64_t e = 0; e < V; ++e) vrow[e] = ldt(v, at3(vv, tok, h, e));
            /* L2 norms (no division by K), with `scale` folded into q's in libref's order. */
            const float qs = 1.0f / std::sqrt(row_sumsq(qn, K) + l2eps) * scale;
            const float ks = 1.0f / std::sqrt(row_sumsq(kn, K) + l2eps);
            {
                const vf vq = vf_set1(qs), vk = vf_set1(ks);
                int64_t d = 0;
                for (; d + VF_N <= K; d += VF_N) {
                    vf_storeu(qn + d, vf_mul(vf_loadu(qn + d), vq));
                    vf_storeu(kn + d, vf_mul(vf_loadu(kn + d), vk));
                }
                for (; d < K; ++d) { qn[d] *= qs; kn[d] *= ks; }
            }

            const float dec = sc_exp(gval);
            const vf vdec = vf_set1(dec);
            for (int64_t e = 0; e < V; ++e) {
                float* Se = S + e * K;
                int64_t d = 0;
                for (; d + VF_N <= K; d += VF_N)
                    vf_storeu(Se + d, vf_mul(vf_loadu(Se + d), vdec));
                for (; d < K; ++d) Se[d] *= dec;
                uu[e] = bt * (vrow[e] - dot2(Se, kn, K));
            }
            for (int64_t e = 0; e < V; ++e) {
                float* Se = S + e * K;
                axpy(Se, kn, uu[e], K);
                orow[e] = dot2(Se, qn, K);
            }

            /* This token's factors, for the next step to replay: the normalised k once per
             * 32-row group, v, then the decay exponent and beta once per group. */
            if (anchor)
                for (int c = 0; c < 3; ++c) {
                    if (c != 1 && !dup_f) continue;
                    const int64_t f = fbs[c] + t * al.fstr;
                    for (int64_t gi = 0; gi < al.groups; ++gi) {
                        for (int64_t d = 0; d < K; ++d) stt(state, f + gi * K + d, kn[d]);
                        stt(state, f + al.gb + 2 * gi, gval);
                        stt(state, f + al.gb + 2 * gi + 1, bt);
                    }
                    for (int64_t e = 0; e < V; ++e) stt(state, f + al.vb + e, vrow[e]);
                }

            /* Gate row, widened once per token. */
            if (z) {
                if (zv.s2 == 1)
                    row_to_f32(z->dtype, byte_at(z->data, z->dtype, at3(zv, tok, h, 0)), zrow,
                               V);
                else
                    for (int64_t e = 0; e < V; ++e) zrow[e] = ldt(z, at3(zv, tok, h, e));
            }
            /* Gated RMS norm over the head's V channels. */
            const float sc = 1.0f / std::sqrt(row_sumsq(orow, V) / (float)V + neps);
            for (int64_t e = 0; e < V; ++e) {
                float r = orow[e] * sc;
                if (nw) r *= gainv[e];
                if (z) {
                    const float zz = zrow[e];
                    const float sg = 1.0f / (1.0f + sc_exp(-zz));
                    r *= (nact == 1 ? sg : zz * sg);
                }
                orow[e] = r;
            }
            if (ov.s2 == 1) out_row(o, at3(ov, tok, h, 0), V, orow);
            else for (int64_t e = 0; e < V; ++e) stt(o, at3(ov, tok, h, e), orow[e]);

            /* Quantise `o` as stored (read back), as the standalone quant_act_fp8 would see it.
             * One group per (token, head): head_v is the fp8 block width. */
            if (oq && osc) {
                if (ov.s2 == 1)
                    row_to_f32(o->dtype, byte_at(o->data, o->dtype, at3(ov, tok, h, 0)), uu, V);
                else
                    for (int64_t e = 0; e < V; ++e) uu[e] = ldt(o, at3(ov, tok, h, e));
                const float amax = row_amax(uu, V);
                const float qsc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
                const float inv = 1.0f / qsc;
                const int64_t qb = rowoff(oq, tok, Hv * V), qs2 = laststride(oq);
                if (qs2 == 1) {
                    const vf vi = vf_set1(inv), vlo = vf_set1(-448.0f), vhi = vf_set1(448.0f);
                    int64_t e = 0;
                    for (; e + VF_N <= V; e += VF_N) {
                        vf u = vf_mul(vf_loadu(uu + e), vi);
                        vf_storeu(uu + e, vf_min(vf_max(u, vlo), vhi));
                    }
                    for (; e < V; ++e) {
                        float u = uu[e] * inv;
                        uu[e] = u > 448.0f ? 448.0f : (u < -448.0f ? -448.0f : u);
                    }
                    row_from_f32(oq->dtype, uu,
                                 (void*)byte_at(oq->data, oq->dtype, qb + h * V), V);
                } else {
                    for (int64_t e = 0; e < V; ++e) {
                        float u = uu[e] * inv;
                        u = u > 448.0f ? 448.0f : (u < -448.0f ? -448.0f : u);
                        rad_store_f32(oq->data, oq->dtype, qb + (h * V + e) * qs2, u);
                    }
                }
                stt(osc, rowoff(osc, tok, Hv) + h * laststride(osc), qsc);
            }

            if (anchor) continue;
            const int64_t so = sidx ? ldi(sidx, at2(si, n, t < sw ? t : sw - 1)) : n;
            if (so >= 0) {
                if (dstride == 1) {
                    for (int64_t e = 0; e < V; ++e)
                        row_from_f32(state->dtype, S + e * K,
                                     (void*)byte_at(state->data, state->dtype,
                                                    so * sstride + h * hstride + e * vstride),
                                     K);
                } else {
                    for (int64_t e = 0; e < V; ++e)
                        for (int64_t d = 0; d < K; ++d)
                            stt(state, so * sstride + h * hstride + e * vstride + d * dstride,
                                S[e * K + d]);
                }
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_conv_recurrent_update */
/* Defined as gdn_conv_update into a staging q, k and v, then gdn_recurrent_update over them, and
 * implemented as exactly that pair of this level's kernels. The staging geometry comes from `o`
 * ([T, Hv, head_v]) and `x` (2 Hg head_k + Hv head_v columns).
 *
 * The staging uses x's dtype; sub-byte dtypes are refused. It is a per-thread buffer that only
 * grows. */
AVX_KERNEL(gdn_conv_recurrent_update) {
    if (a->n_t < 16) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *o = &a->t[15];
    if (!x->data || !o->data || o->rank != 3 || x->rank < 2) return RAD_E_SHAPE;
    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const int64_t T = o->shape[0], Hv = o->shape[1];
    if (K <= 0 || V <= 0 || T <= 0 || Hv <= 0 || o->shape[2] != V) return RAD_E_SHAPE;
    const int64_t conv_dim = x->shape[x->rank - 1];
    if (conv_dim <= Hv * V || (conv_dim - Hv * V) % (2 * K)) return RAD_E_SHAPE;
    const int64_t Hg = (conv_dim - Hv * V) / (2 * K);
    const int64_t eb = (int64_t)rad_dtype_bits(x->dtype) / 8;
    if (eb <= 0 || (int64_t)rad_dtype_bits(x->dtype) % 8) return RAD_E_DTYPE;

    static thread_local std::vector<uint8_t> stage;
    const int64_t nq = T * Hg * K, nv = T * Hv * V;
    if (stage.size() < (size_t)((2 * nq + nv) * eb)) stage.resize((size_t)((2 * nq + nv) * eb));
    auto staged = [&](int64_t off, int64_t h, int64_t w) {
        RadTensor t{};
        t.data = stage.data() + off * eb;
        t.dtype = x->dtype;
        t.rank = 3;
        t.shape[0] = T; t.shape[1] = h; t.shape[2] = w;
        t.stride[0] = h * w; t.stride[1] = w; t.stride[2] = 1;
        return t;
    };
    const RadTensor tq = staged(0, Hg, K), tk = staged(nq, Hg, K), tv = staged(2 * nq, Hv, V);

    RadParam cp[4] = {};
    const char* ck[4] = { "q_len", "head_k", "head_v", "conv_width" };
    for (int i = 0; i < 4; ++i) {
        const RadParam* f = rad_param_find(a->p, a->n_p, ck[i]);
        if (!f) return RAD_E_INVAL;
        cp[i] = *f;
    }
    /* x, w, b, conv_state, state_idx, num_accepted, cu, q, k, v */
    RadTensor ct[10] = { a->t[0], a->t[1], a->t[2], a->t[3], a->t[4], a->t[11], a->t[12], tq, tk,
                         tv };
    RadArgs ca = *a;
    ca.t = ct; ca.n_t = 10; ca.p = cp; ca.n_p = 4;
    const int rc = AVX_FN(avx_gdn_conv_update)(&ca, stream);
    if (rc != RAD_OK) return rc;

    /* q, k, v, a, b, A_log, dt_bias, state, state_idx, num_accepted, cu, z, norm_w, o, o_q,
     * o_scale. The parameters pass as they are: the recurrence reads its own by name. */
    RadTensor rt[16] = { tq, tk, tv, a->t[5], a->t[6], a->t[7], a->t[8], a->t[9], a->t[10],
                         a->t[11], a->t[12], a->t[13], a->t[14], a->t[15],
                         a->n_t > 16 ? a->t[16] : RadTensor{}, a->n_t > 17 ? a->t[17] : RadTensor{} };
    RadArgs ra = *a;
    ra.t = rt; ra.n_t = 16;
    return AVX_FN(avx_gdn_recurrent_update)(&ra, stream);
}

/* ------------------------------------------------------------------ gdn_gated_rmsnorm */
/* out = x * rsqrt(mean_channels(x^2) + eps) * w * act(z), one row per (token, head): the prefill
 * form of the epilogue the recurrent update folds in.
 *
 * The gate may be a rank-2 column slice of a wider projection row (the fp8 delta net fuses
 * [q|k|v|z] into one GEMM) with heads*ch as its last axis, which rowoff() would read as a dense row.
 * Handled here since it is specific to this operand. */
static inline int64_t gate_rowoff(const RadTensor* t, int64_t r, int64_t ch) {
    if (t->rank == 2 && ch > 0 && t->shape[1] != ch && t->shape[1] % ch == 0) {
        const int64_t heads = t->shape[1] / ch;
        return (r / heads) * t->stride[0] + (r % heads) * ch * t->stride[1];
    }
    return rowoff(t, r, ch);
}

AVX_KERNEL(gdn_gated_rmsnorm) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *z = &a->t[1], *w = &a->t[2], *o = &a->t[3];
    const int64_t ch = p_int(a, "channels", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, 0);
    if (act != 0 && act != 1) return RAD_E_INVAL;
    if (ch <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / ch;
    if (rows <= 0 || numel(o) < rows * ch || numel(z) < rows * ch || numel(w) < ch)
        return RAD_E_SHAPE;

    float* gain = (float*)scratch_raw((size_t)ch * sizeof(float) + 64);
    {
        const int64_t ws = laststride(w);
        if (ws == 1) row_to_f32(w->dtype, w->data, gain, ch);
        else for (int64_t i = 0; i < ch; ++i) gain[i] = ldt(w, i * ws);
    }

    AVX_PARALLEL_FOR_IF(rows * ch >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, ch), zb = gate_rowoff(z, r, ch), ob = rowoff(o, r, ch);
        const int64_t zs = laststride(z), os = laststride(o);
        float* xbuf = scratch_f32((size_t)ch);
        float* zbuf = scratch_f32_b((size_t)ch);
        float* obuf = scratch_f32_c((size_t)ch);
        const float* xv = in_row(x, xb, ch, xbuf);
        const float* zv2 = (zs == 1) ? in_row(z, zb, ch, zbuf) : zbuf;
        if (zs != 1) for (int64_t i = 0; i < ch; ++i) zbuf[i] = ldt(z, zb + i * zs);

        const float sc = 1.0f / std::sqrt(row_sumsq(xv, ch) / (float)ch + eps);
        const vf vs = vf_set1(sc);
        int64_t i = 0;
        for (; i + VF_N <= ch; i += VF_N) {
            const vf zz = vf_loadu(zv2 + i);
            const vf gate = (act == 1) ? vf_sigmoid(zz) : vf_silu(zz);
            vf_storeu(obuf + i,
                      vf_mul(vf_mul(vf_mul(vf_loadu(xv + i), vs), vf_loadu(gain + i)), gate));
        }
        for (; i < ch; ++i) {
            const float zz = zv2[i];
            const float sg = 1.0f / (1.0f + sc_exp(-zz));
            obuf[i] = xv[i] * sc * gain[i] * (act == 1 ? sg : zz * sg);
        }
        if (os == 1) out_row(o, ob, ch, obuf);
        else for (int64_t j = 0; j < ch; ++j) stt(o, ob + j * os, obuf[j]);
    }
    return RAD_OK;
}
