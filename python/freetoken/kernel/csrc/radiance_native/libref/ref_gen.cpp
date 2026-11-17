/* ref_gen.cpp -- the image generator's ops (docs/OPS.md, "Image generation").
 *
 * Every explicit rounding follows the upstream model, which runs the diffusion transformer and its
 * decoder in bf16 and so rounds every tensor it materialises: `LN(x) * (1 + scale)` is three bf16
 * tensors, not one f32 expression. Rounding at the same points lets a fused kernel be held to it.
 *
 * Per-request geometry (an image's height and width) comes from the operands, never the
 * parameters, which are fixed at declare: a 2-D image operand is [height, width * channels], dense.
 */
#include "ref_common.h"
#include "ref_gemm.h"
#include "ref_ops.h"
#include "rad_philox.h"

#include <algorithm>
#include <vector>
#include <string>

using namespace ref;

namespace {

/* An entry's flags (docs/OPS.md, flow_euler): RadGenBatch's RAD_GEN_F_* values, restated because a
 * kernel library compiles against rad_abi.h, not the run phase's header. */
enum { kFirst = 1, kInpaint = 2, kPreview = 4 };

inline float rbf(float v) { return round_dt(RAD_BF16, v); }

/* A space-separated list of floats, each rounded to bf16 as the upstream bf16 tensor holds it, or
 * kept in f32 with `bf16` false. */
int parse_list(const char* s, std::vector<float>* out, bool bf16 = true) {
    out->clear();
    while (s && *s) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end == s) break;
        out->push_back(bf16 ? rbf((float)v) : (float)v);
        s = end;
    }
    return (int)out->size();
}

}  /* namespace */

/* ------------------------------------------------------------------ ln_mod */
/* y = bf16( bf16(LN(x)) * bf16(1 + s[seq]) ): a layer norm with no gain or bias, then the row's
 * sequence's modulation row. `s` is [n_seq, >= n] at its own row pitch (a column slice of the
 * modulation projection). */
int ref_ln_mod(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *s = &a->t[1], *cu = &a->t[2], *y = &a->t[3];
    const int64_t n = p_int(a, "n", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n), n_seq = numel(cu) - 1;
    if (rows <= 0 || n_seq <= 0 || rows_of(x, n) < rows) return RAD_E_SHAPE;
    const int64_t s_rs = row_stride(s, n), s_cs = laststride(s);
    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t e = seq_of(cu, n_seq, r);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) sum += ldt(x, xb + i * xs);
        const float mean = sum / (float)n;
        float var = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            const float d = ldt(x, xb + i * xs) - mean;
            var += d * d;
        }
        const float rs = 1.0f / std::sqrt(var / (float)n + eps);
        for (int64_t i = 0; i < n; ++i) {
            const float ln = round_dt(y->dtype, (ldt(x, xb + i * xs) - mean) * rs);
            const float sc = round_dt(y->dtype, 1.0f + ldt(s, e * s_rs + i * s_cs));
            stt(y, yb + i * ys, ln * sc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gate_add */
/* x = bf16( x + bf16( bf16(tanh(g[seq])) * y ) ): a residual added through its sequence's gate.
 * With `act` none the gate is g itself, unrounded: x = bf16(x + bf16(g[seq] * y)). */
int ref_gate_add(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *y = &a->t[0], *g = &a->t[1], *cu = &a->t[2], *x = &a->t[3];
    const int64_t n = p_int(a, "n", 0);
    static const char* kActs[] = { "tanh", "none" };
    const int act = p_enum(a, "act", kActs, 2, 0);
    if (act < 0 || act > 1) return RAD_E_INVAL;
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n), n_seq = numel(cu) - 1;
    if (rows <= 0 || n_seq <= 0 || rows_of(x, n) < rows) return RAD_E_SHAPE;
    const int64_t g_rs = row_stride(g, n), g_cs = laststride(g);
    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t e = seq_of(cu, n_seq, r);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) {
            const float gv = ldt(g, e * g_rs + i * g_cs);
            const float t = act ? gv : round_dt(x->dtype, ref_tanh(gv));
            const float p = round_dt(x->dtype, t * ldt(y, yb + i * ys));
            stt(x, xb + i * xs, ldt(x, xb + i * xs) + p);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rms_mod / rms_mod_quant_fp8 */
/* The checks both forms share: `s` holds [shift | scale] in each of its n_seq rows, `g` a gate
 * row per sequence when `gated`, and every row operand at least `rows` rows. With `mapped`, the
 * sequences address table rows through the column map, so `s` and `g` may hold fewer rows than
 * there are sequences; the map's bound is the caller's. */
static int rms_mod_check(const RadTensor* y, const RadTensor* g, const RadTensor* s,
                         const RadTensor* w, const RadTensor* x, int64_t n, int64_t n_seq,
                         int64_t rows, bool gated, bool mapped) {
    if (rows <= 0 || n_seq <= 0 || rows_of(x, n) < rows || (w && numel(w) < n) ||
        (!mapped && numel(s) < n_seq * 2 * n))
        return RAD_E_SHAPE;
    if (gated && (rows_of(y, n) < rows || (!mapped && numel(g) < n_seq * n))) return RAD_E_SHAPE;
    return RAD_OK;
}

/* Row r of sequence e into o[0, n). With `gated` the branch is added first through the gate row,
 * x = bf16(x + bf16(g[e] * y[r])) (no tanh), and stored; then an RMS norm with a gain and the
 * sequence's shift and scale, each step a tensor the upstream bf16 model stores:
 *
 *   n1 = bf16(x * rsqrt(mean(x^2) + eps))   n2 = bf16(n1 * w)
 *   m  = bf16(n2 * bf16(1 + scale[e]))      z  = bf16(m + shift[e])
 *
 * The sum of squares is f32 in ascending order, read from the stored residual. `s` is
 * [n_seq, >= 2n] at its own row pitch, shift = s[e, 0:n] and scale = s[e, n:2n] (a column slice
 * of the modulation table). The residual rounds to x's dtype and the norm chain to `dt`.
 *
 * `wadd` is added to the gain in f32 (a zero-centred norm's 1 + w), and with `once` the norm and
 * its gain round together, n2 = bf16(x rsqrt(...) (w + wadd)), as an f32 RMS norm with a gain
 * stores its output (Krea 2's). With `layer` the norm is a LayerNorm (FLUX's, which has no gain:
 * `w` absent), and with `none` there is no norm and no gain, n2 = x (LTX-2's text keys and values,
 * modulated as they are). */
static void rms_mod_row(const RadTensor* y, const RadTensor* g, const RadTensor* s,
                        const RadTensor* w, const RadTensor* x, int64_t r, int64_t e, int64_t n,
                        float eps, bool gated, uint32_t dt, float* o, float wadd = 0.0f,
                        bool once = false, bool layer = false, bool none = false) {
    const int64_t xb = rowoff(x, r, n), xs = laststride(x);
    if (gated) {
        const int64_t g_rs = row_stride(g, n), g_cs = laststride(g);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t i = 0; i < n; ++i) {
            const float p = round_dt(x->dtype, ldt(g, e * g_rs + i * g_cs) * ldt(y, yb + i * ys));
            stt(x, xb + i * xs, ldt(x, xb + i * xs) + p);
        }
    }
    /* RMS: x rsqrt(mean(x^2) + eps). Layer: (x - mean) rsqrt(var + eps), the biased variance of
     * the centred values; both sums f32 in ascending order. */
    float mu = 0.0f;
    if (layer) {
        for (int64_t i = 0; i < n; ++i) mu += ldt(x, xb + i * xs);
        mu /= (float)n;
    }
    float ss = 0.0f;
    for (int64_t i = 0; i < n && !none; ++i) { const float v = ldt(x, xb + i * xs) - mu; ss += v * v; }
    const float rs = 1.0f / std::sqrt(ss / (float)n + eps);
    const int64_t s_rs = row_stride(s, 2 * n), s_cs = laststride(s), ws = w ? laststride(w) : 0;
    for (int64_t i = 0; i < n; ++i) {
        const float v = ldt(x, xb + i * xs) - mu;
        /* With no gain the norm is rounded once, as a norm without a weight stores it. */
        const float n2 = none ? round_dt(dt, v)
                       : !w ? round_dt(dt, v * rs)
                       : once ? round_dt(dt, v * rs * (ldt(w, i * ws) + wadd))
                              : round_dt(dt, round_dt(dt, v * rs) * (ldt(w, i * ws) + wadd));
        const float sc = round_dt(dt, 1.0f + ldt(s, e * s_rs + (n + i) * s_cs));
        const float m  = round_dt(dt, n2 * sc);
        o[i] = round_dt(dt, m + ldt(s, e * s_rs + i * s_cs));
    }
}

/* A diffusion transformer's adaptive RMS norm with a shift (MiniMax-H3's
 * `norm(h) * (1 + scale) + shift`), the gated residual of the branch before it optionally first;
 * the norm chain rounds to z's dtype.
 *
 * The optional trailing `colmap` (slot 7, or 9 for the quantising form), one i32 a sequence, maps a
 * sequence to its row of the modulation table, so sequences that share a cell read one row. */
int ref_rms_mod(const RadArgs* a, RadStream) {
    if (a->n_t < 7) return RAD_E_INVAL;
    const RadTensor *y = t_in(a, 0), *g = t_in(a, 1), *s = &a->t[2], *w = t_in(a, 3);
    const RadTensor *cu = &a->t[4], *x = &a->t[5], *z = &a->t[6], *cm = t_in(a, 7);
    const int64_t n = p_int(a, "n", 0);
    const bool gated = p_int(a, "gated", 0) != 0;
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    const bool once = p_int(a, "norm_once", 0) != 0;
    const char* nk = p_str(a, "norm", "rms");
    const bool layer = std::strcmp(nk, "layer") == 0, none = std::strcmp(nk, "none") == 0;
    if (!layer && !none && std::strcmp(nk, "rms") != 0) return RAD_E_INVAL;
    if (none && t_in(a, 3)) return RAD_E_INVAL;
    if (n <= 0 || (gated && (!y || !g))) return RAD_E_SHAPE;
    const int64_t rows = rows_of(z, n), n_seq = numel(cu) - 1;
    if (cm && (cm->dtype != RAD_I32 || numel(cm) < n_seq)) return RAD_E_SHAPE;
    RAD_REF_TRY(rms_mod_check(y, g, s, w, x, n, n_seq, rows, gated, cm != nullptr));
    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        std::vector<float> o((size_t)n);
        int64_t e = seq_of(cu, n_seq, r);
        if (cm) e = (int64_t)ldt(cm, e);
        rms_mod_row(y, g, s, w, x, r, e, n, eps, gated, z->dtype, o.data(), wadd, once, layer, none);
        const int64_t zb = rowoff(z, r, n), zs = laststride(z);
        for (int64_t i = 0; i < n; ++i) stt(z, zb + i * zs, o[(size_t)i]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ timestep_embed */
/* y[m] = [cos(t' f_0) .. cos(t' f_{h-1}) | sin(t' f_0) .. sin(t' f_{h-1})], h = n / 2, where
 * f_i = exp(-ln(max_period) * i / h) in f32 and t' = scale * t -- or, with `round` 1, the value
 * diffusers' pipeline hands its transformer: t rounded to bf16 at scale, divided back and rounded
 * again, then scaled: scale * bf16(bf16(t * scale) / scale). t is raised to `clamp_min` first.
 * With `curve` = G > 0 and a `table` [G, n] f32 operand instead, y[m] is the table at
 * clamp01(max(t[m], clamp_min)) (G - 1), between the two rows around it: a curve-form
 * checkpoint's embeddings. */
int ref_timestep_embed(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *t = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    const float scale = p_f32(a, "scale", 1000.0f);
    const double mp = p_f32(a, "max_period", 10000.0f);
    const int round = (int)p_int(a, "round", 0);
    const float clamp = p_f32(a, "clamp_min", 0.0f);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t grid = p_int(a, "curve", 0);
    if (!grid && (n & 1)) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(t) < rows) return RAD_E_SHAPE;
    if (grid > 0) {
        const RadTensor* tab = t_in(a, 2);
        if (!tab) return RAD_E_INVAL;
        if (tab->dtype != (uint32_t)RAD_F32 || numel(tab) < grid * n) return RAD_E_SHAPE;
        for (int64_t m = 0; m < rows; ++m) {
            const float tv = std::fmax(ldt(t, offlin(t, m)), clamp);
            const float pos = std::fmin(std::fmax(tv, 0.0f), 1.0f) * (float)(grid - 1);
            float i0f = std::floor(pos);
            if (i0f > (float)(grid - 2)) i0f = (float)(grid - 2);
            const float fr = pos - i0f;
            const int64_t i0 = (int64_t)i0f;
            const int64_t yb = rowoff(y, m, n), ys = laststride(y);
            for (int64_t i = 0; i < n; ++i) {
                const float a0 = ldt(tab, offlin(tab, i0 * n + i));
                const float a1 = ldt(tab, offlin(tab, (i0 + 1) * n + i));
                stt(y, yb + i * ys, a0 + fr * (a1 - a0));
            }
        }
        return RAD_OK;
    }
    const int64_t h = n / 2;
    const float coef = (float)(-std::log(mp));
    for (int64_t m = 0; m < rows; ++m) {
        const float tv = std::fmax(ldt(t, offlin(t, m)), clamp);
        const float tt = round == 3 ? rbf(scale * rbf(tv)) : round == 2 ? scale * rbf(tv)
                       : round ? scale * rbf(rbf(tv * scale) / scale) : scale * tv;
        const int64_t yb = rowoff(y, m, n), ys = laststride(y);
        for (int64_t i = 0; i < h; ++i) {
            const float f = ref_exp(coef * (float)i / (float)h);
            const float arg = tt * f;
            stt(y, yb + i * ys, ref_cos(arg));
            stt(y, yb + (h + i) * ys, ref_sin(arg));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gather_rows2 */
/* y[i] = idx[i] >= 0 ? a[idx[i]] : b[-1 - idx[i]] -- a sequence assembled from two sources. */
int ref_gather_rows2(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *idx = &a->t[2], *y = &a->t[3];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(idx) < rows) return RAD_E_SHAPE;
    const int64_t ra = rows_of(A, n), rbn = rows_of(B, n);
    REF_PAR_FOR(rows * n)
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t r = ld_int(idx->dtype, idx->data, offlin(idx, i));
        const RadTensor* src = r >= 0 ? A : B;
        const int64_t row = r >= 0 ? r : -1 - r;
        const int64_t yb = rowoff(y, i, n), ys = laststride(y);
        if (row >= (r >= 0 ? ra : rbn)) {
            for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, 0.0f);
            continue;
        }
        const int64_t sb = rowoff(src, row, n), ss = laststride(src);
        for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, ldt(src, sb + c * ss));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ lane_gather / lane_store */
/* lane_gather: y[cu[e] + k] = src[lane[e] * stride + k] for the rows of each entry. */
int ref_lane_gather(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *src = &a->t[0], *lane = &a->t[1], *cu = &a->t[2], *y = &a->t[3];
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1, src_rows = rows_of(src, n);
    if (n_seq <= 0 || numel(lane) < n_seq || cu_at(cu, n_seq) <= cu_at(cu, 0)) return RAD_E_SHAPE;
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t L = ld_int(lane->dtype, lane->data, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), b1 = cu_at(cu, e + 1);
        if (L < 0 || b1 - b0 > stride || (L + 1) * stride > src_rows) return RAD_E_SHAPE;
        for (int64_t k = 0; k < b1 - b0; ++k) {
            const int64_t sb = rowoff(src, L * stride + k, n), ss = laststride(src);
            const int64_t yb = rowoff(y, b0 + k, n), ys = laststride(y);
            for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, ldt(src, sb + c * ss));
        }
    }
    return RAD_OK;
}

/* lane_store: every listed lane's rows [0, rows) become src's rows. */
int ref_lane_store(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *src = &a->t[0], *lane = &a->t[1], *dst = &a->t[2];
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(src, n), dst_rows = rows_of(dst, n), nl = numel(lane);
    if (rows <= 0 || rows > stride) return RAD_E_SHAPE;
    for (int64_t l = 0; l < nl; ++l) {
        const int64_t L = ld_int(lane->dtype, lane->data, offlin(lane, l));
        if (L < 0 || (L + 1) * stride > dst_rows) return RAD_E_SHAPE;
        for (int64_t k = 0; k < rows; ++k) {
            const int64_t sb = rowoff(src, k, n), ss = laststride(src);
            const int64_t db = rowoff(dst, L * stride + k, n), ds = laststride(dst);
            for (int64_t c = 0; c < n; ++c) stt(dst, db + c * ds, ldt(src, sb + c * ss));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ randn_lanes */
/* The first step of an image: its lane's rows [0, cu[e+1] - cu[e]) become the seed's normal
 * stream (abi/rad_philox.h), element (k, c) the stream's element k * n + c. Entries whose flags
 * lack RAD_GEN_F_FIRST are left alone. */
int ref_randn_lanes(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *fl = &a->t[0], *seed = &a->t[1], *lane = &a->t[2], *cu = &a->t[3];
    const RadTensor *dst = &a->t[4];
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1, dst_rows = rows_of(dst, n);
    if (n_seq <= 0 || numel(fl) < n_seq || numel(lane) < n_seq || numel(seed) < 2 * n_seq)
        return RAD_E_SHAPE;
    if (cu_at(cu, n_seq) <= cu_at(cu, 0)) return RAD_E_SHAPE;
    for (int64_t e = 0; e < n_seq; ++e) {
        if (!(ld_int(fl->dtype, fl->data, offlin(fl, e)) & kFirst)) continue;
        const int64_t L = ld_int(lane->dtype, lane->data, offlin(lane, e));
        const int64_t len = cu_at(cu, e + 1) - cu_at(cu, e);
        if (L < 0 || len > stride || (L + 1) * stride > dst_rows) return RAD_E_SHAPE;
        const uint64_t sd = (uint64_t)(uint32_t)ld_int(seed->dtype, seed->data, offlin(seed, 2 * e)) |
                            ((uint64_t)(uint32_t)ld_int(seed->dtype, seed->data, offlin(seed, 2 * e + 1)) << 32);
        REF_PAR_FOR(len * n * 8)
        for (int64_t k = 0; k < len; ++k) {
            const int64_t db = rowoff(dst, L * stride + k, n), ds = laststride(dst);
            for (int64_t c = 0; c < n; ++c)
                stt(dst, db + c * ds, rad_philox_normal(sd, (uint64_t)(k * n + c)));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ flow_euler */
/* One Euler step of a flow-matching sampler for every entry that moves a lane (pair != -2):
 *
 *   v'  = pair >= 0 ? bf16(neg + bf16(cfg * bf16(pos - neg))) : pos   the guided velocity
 *   x'  = bf16(x + bf16(bf16(sigma_next - sigma) * v'))              diffusers' step, its roundings
 *
 * and, by flag, kPreview writes bf16(x - sigma * v') -- the step's estimate of the clean
 * latent -- to the lane's preview rows; kInpaint puts the kept region back afterwards:
 * x' = bf16(m x' + (1 - m)((1 - sigma_next) src + sigma_next noise)), m the lane's mask row and
 * noise the image's own stream (randn_lanes). At the last step sigma_next is 0 and the kept
 * region is the picture's latent exactly. */
int ref_flow_euler(const RadArgs* a, RadStream) {
    /* inp, mask and preview are optional: a sampler without repaint or previews passes none. */
    if (!have(a, 9) || !t_in(a, 11)) return RAD_E_INVAL;
    const RadTensor *v = &a->t[0], *sg = &a->t[1], *sgn = &a->t[2], *cfg = &a->t[3];
    const RadTensor *pair = &a->t[4], *lane = &a->t[5], *fl = &a->t[6], *seed = &a->t[7];
    const RadTensor *cu = &a->t[8];
    const RadTensor *inp = t_in(a, 9), *mask = t_in(a, 10);
    const RadTensor *st = t_in(a, 11);
    const RadTensor *pv = t_in(a, 12);
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1, st_rows = rows_of(st, n);
    if (n_seq <= 0 || cu_at(cu, n_seq) <= cu_at(cu, 0)) return RAD_E_SHAPE;
    const uint32_t dt = st->dtype;
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t pr = ld_int(pair->dtype, pair->data, offlin(pair, e));
        if (pr == -2) continue;
        const int64_t L = ld_int(lane->dtype, lane->data, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), len = cu_at(cu, e + 1) - b0;
        const int64_t p0 = pr >= 0 ? cu_at(cu, pr) : 0;
        if (L < 0 || len > stride || (L + 1) * stride > st_rows) return RAD_E_SHAPE;
        if (pr >= 0 && cu_at(cu, pr + 1) - p0 != len) return RAD_E_SHAPE;
        const float s  = ldt(sg, offlin(sg, e)), sn = ldt(sgn, offlin(sgn, e));
        const float w  = ldt(cfg, offlin(cfg, e));
        const int64_t f = ld_int(fl->dtype, fl->data, offlin(fl, e));
        const uint64_t sd = (uint64_t)(uint32_t)ld_int(seed->dtype, seed->data, offlin(seed, 2 * e)) |
                            ((uint64_t)(uint32_t)ld_int(seed->dtype, seed->data, offlin(seed, 2 * e + 1)) << 32);
        const float dts = round_dt(dt, sn - s);
        if ((f & kInpaint) && (!inp || !mask)) return RAD_E_INVAL;
        if ((f & kPreview) && !pv) return RAD_E_INVAL;
        REF_PAR_FOR(len * n * 8)
        for (int64_t k = 0; k < len; ++k) {
            const int64_t vb = rowoff(v, b0 + k, n), vs = laststride(v);
            const int64_t nb = pr >= 0 ? rowoff(v, p0 + k, n) : 0;
            const int64_t sb = rowoff(st, L * stride + k, n), ss = laststride(st);
            for (int64_t c = 0; c < n; ++c) {
                float vv = ldt(v, vb + c * vs);
                if (pr >= 0) {
                    const float vn = ldt(v, nb + c * vs);
                    vv = round_dt(dt, vn + round_dt(dt, w * round_dt(dt, vv - vn)));
                }
                const float x = ldt(st, sb + c * ss);
                if (f & kPreview) {
                    const int64_t pb = rowoff(pv, L * stride + k, n);
                    stt(pv, pb + c * laststride(pv), x - s * vv);
                }
                float xn = round_dt(dt, x + round_dt(dt, dts * vv));
                if (f & kInpaint) {
                    const float m = ldt(mask, offlin(mask, L * stride + k));
                    const float src = ldt(inp, rowoff(inp, L * stride + k, n) + c * laststride(inp));
                    const float nz = rad_philox_normal(sd, (uint64_t)(k * n + c));
                    const float kept = (1.0f - sn) * src + sn * nz;
                    xn = round_dt(dt, m * xn + (1.0f - m) * kept);
                }
                stt(st, sb + c * ss, xn);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ flow_x0_step */
/* One sampler step over a lane's rows in f32, as diffusers' MiniMaxH3Scheduler.step takes it: the
 * velocity points toward the data, so the clean estimate is d = x + (1 - t) v, and the step blends
 * the sample toward it by the schedule's ratio. Each product and sum is one f32 rounding:
 *
 *   st = f32(1 - t)   d = f32(x + f32(st * v))   r = f32(sigma_next / sigma)
 *   x' = f32(f32(r * x) + f32(f32(1 - r) * d))
 *
 * t, sigma and sigma_next are one value each: t the timestep the transformer was conditioned on,
 * sigma and sigma_next the schedule's entries, kept apart because 1 - (1 - sigma) is not exact in
 * f32. x' is stored at x's dtype. */
int ref_flow_x0_step(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *v = &a->t[0], *t = &a->t[1], *sg = &a->t[2], *sgn = &a->t[3], *x = &a->t[4];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || rows_of(v, n) < rows || numel(t) < 1 || numel(sg) < 1 || numel(sgn) < 1)
        return RAD_E_SHAPE;
    const float st = 1.0f - ldt(t, offlin(t, 0));
    const float r = ldt(sgn, offlin(sgn, 0)) / ldt(sg, offlin(sg, 0));
    const float omr = 1.0f - r;
    REF_PAR_FOR(rows * n)
    for (int64_t k = 0; k < rows; ++k) {
        const int64_t vb = rowoff(v, k, n), vs = laststride(v);
        const int64_t xb = rowoff(x, k, n), xs = laststride(x);
        for (int64_t c = 0; c < n; ++c) {
            const float xv = ldt(x, xb + c * xs);
            const float d = xv + st * ldt(v, vb + c * vs);
            stt(x, xb + c * xs, r * xv + omr * d);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ flow_res_multistep */
/* k-diffusion's res_multistep at eta 0 (DPM-Solver++ 2M) on the clean estimate d = x + sigma v:
 * x' = r x + hb1 d + hb2 x0, r = sigma_next / sigma and x0 the last step's estimate, which d then
 * replaces; with `order` < 2, Euler, x' = r x + (1 - r) d. sigma, r, hb1, hb2 and order are one
 * f32 each, x and x0 f32. */
int ref_flow_res_multistep(const RadArgs* a, RadStream) {
    if (!have(a, 8)) return RAD_E_INVAL;
    const RadTensor *v = &a->t[0], *x0 = &a->t[1], *sg = &a->t[2], *rr = &a->t[3];
    const RadTensor *hb1 = &a->t[4], *hb2 = &a->t[5], *ord = &a->t[6], *x = &a->t[7];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || rows_of(v, n) < rows || rows_of(x0, n) < rows || numel(sg) < 1 ||
        numel(rr) < 1 || numel(hb1) < 1 || numel(hb2) < 1 || numel(ord) < 1)
        return RAD_E_SHAPE;
    const float sgv = ldt(sg, offlin(sg, 0));
    const float r = ldt(rr, offlin(rr, 0));
    const float b1 = ldt(hb1, offlin(hb1, 0));
    const float b2 = ldt(hb2, offlin(hb2, 0));
    const bool two = ldt(ord, offlin(ord, 0)) >= 1.5f;
    const float omr = 1.0f - r;
    REF_PAR_FOR(rows * n)
    for (int64_t k = 0; k < rows; ++k) {
        const int64_t vb = rowoff(v, k, n), vs = laststride(v);
        const int64_t xb = rowoff(x, k, n), xs = laststride(x);
        const int64_t ob = rowoff(x0, k, n), os = laststride(x0);
        for (int64_t c = 0; c < n; ++c) {
            const float xv = ldt(x, xb + c * xs);
            const float d = xv + sgv * ldt(v, vb + c * vs);
            const float nx = two ? r * xv + b1 * d + b2 * ldt(x0, ob + c * os)
                                 : r * xv + omr * d;
            stt(x, xb + c * xs, nx);
            stt(x0, ob + c * os, d);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ kd_input, kd_step */
/* A k-diffusion sampler over a discrete-time (VP-trained) denoiser, run in the variance-exploding
 * form x = x0 + sigma eps: the lane holds x in f32, the model reads c_in x with
 * c_in = 1 / sqrt(sigma^2 + 1), and a step turns the model's output into the clean estimate D and
 * moves x along the schedule (docs/OPS.md, "Image generation"). Every product and sum is one f32
 * rounding in the order written; libm's log, expm1 and sqrt. */

namespace {

enum { kSamplerEuler = 0, kSamplerEulerAncestral = 1, kSamplerDpmpp2m = 2 };

/* One entry's per-step constants, from its sigmas and guidance. */
struct KdEntry {
    float s = 0, sn = 0, w = 0;
    float cskip = 0, cout = 0;          /* v-prediction's c_skip and c_out */
    float sup = 0, sdown = 0;           /* the ancestral step's split of sigma_next */
    float ratio = 0, em = 0;            /* DPM++ 2M: sigma_next / sigma, expm1(-h) */
    float c1 = 0, c2 = 0;               /* DPM++ 2M's second-order blend of D and the last D */
    bool  second = false;
};

KdEntry kd_entry(float s, float sn, float sp, float w, int64_t sampler, bool first) {
    KdEntry k;
    k.s = s;
    k.sn = sn;
    k.w = w;
    const float s2p = s * s + 1.0f;
    k.cskip = 1.0f / s2p;
    k.cout = s / std::sqrt(s2p);
    if (sampler == kSamplerEulerAncestral) {
        const float s2 = s * s, sn2 = sn * sn;
        const float up2 = sn2 * (s2 - sn2) / s2;
        float sup = std::sqrt(up2);
        if (!(sup < sn)) sup = sn;
        const float dn2 = sn2 - sup * sup;
        k.sup = sup;
        k.sdown = std::sqrt(dn2 > 0.0f ? dn2 : 0.0f);
    } else if (sampler == kSamplerDpmpp2m && sn > 0.0f) {
        const float t = -std::log(s), tn = -std::log(sn);
        const float h = tn - t;
        k.ratio = sn / s;
        k.em = std::expm1(-h);
        if (!first) {
            const float hl = t - (-std::log(sp));
            const float r = hl / h;
            k.c2 = 1.0f / (2.0f * r);
            k.c1 = 1.0f + k.c2;
            k.second = true;
        }
    }
    return k;
}

/* The clean estimate from the model's output `o` at x. */
inline float kd_denoised(const KdEntry& k, bool v_pred, float x, float o) {
    return v_pred ? x * k.cskip - o * k.cout : x - k.s * o;
}

/* x after the step, from x, the guided estimate D, the lane's last estimate and the noise. */
inline float kd_next(const KdEntry& k, int64_t sampler, float x, float D, float hist, float nz) {
    if (sampler == kSamplerEuler) {
        const float d = (x - D) / k.s;
        return x + (k.sn - k.s) * d;
    }
    if (sampler == kSamplerEulerAncestral) {
        const float d = (x - D) / k.s;
        float xn = x + d * (k.sdown - k.s);
        if (k.sn > 0.0f) xn = xn + nz * k.sup;
        return xn;
    }
    if (k.sn == 0.0f) return D;
    const float Dd = k.second ? k.c1 * D - k.c2 * hist : D;
    return k.ratio * x - k.em * Dd;
}

inline uint64_t kd_seed(const RadTensor* seed, int64_t e) {
    return (uint64_t)(uint32_t)ld_int(seed->dtype, seed->data, offlin(seed, 2 * e)) |
           ((uint64_t)(uint32_t)ld_int(seed->dtype, seed->data, offlin(seed, 2 * e + 1)) << 32);
}

}  /* namespace */

/* The model's input for every entry that moves a lane (pair != -2): on a first step the lane
 * becomes the seed's normal stream times sqrt(1 + sigma^2) (k-diffusion's start at the schedule's
 * largest sigma), then y = rd(x / sqrt(sigma^2 + 1)) at the entry's rows and, when it is guided, at
 * its pair's rows too (the negative entry reads the same latent). */
int ref_kd_input(const RadArgs* a, RadStream) {
    if (!have(a, 8)) return RAD_E_INVAL;
    const RadTensor *sg = &a->t[0], *pair = &a->t[1], *lane = &a->t[2], *fl = &a->t[3];
    const RadTensor *seed = &a->t[4], *cu = &a->t[5], *st = &a->t[6], *y = &a->t[7];
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1, st_rows = rows_of(st, n), y_rows = rows_of(y, n);
    if (n_seq <= 0 || cu_at(cu, n_seq) <= cu_at(cu, 0)) return RAD_E_SHAPE;
    if (numel(sg) < n_seq || numel(pair) < n_seq || numel(lane) < n_seq || numel(fl) < n_seq ||
        numel(seed) < 2 * n_seq)
        return RAD_E_SHAPE;
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t pr = ld_int(pair->dtype, pair->data, offlin(pair, e));
        if (pr == -2) continue;
        const int64_t L = ld_int(lane->dtype, lane->data, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), len = cu_at(cu, e + 1) - b0;
        if (L < 0 || len > stride || (L + 1) * stride > st_rows || b0 < 0 || b0 + len > y_rows)
            return RAD_E_SHAPE;
        if (pr >= n_seq || (pr >= 0 && (cu_at(cu, pr + 1) - cu_at(cu, pr) != len ||
                                        cu_at(cu, pr) < 0 || cu_at(cu, pr) + len > y_rows)))
            return RAD_E_SHAPE;
        if (ld_int(fl->dtype, fl->data, offlin(fl, e)) & kInpaint) return RAD_E_UNSUPPORTED;
    }
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t pr = ld_int(pair->dtype, pair->data, offlin(pair, e));
        if (pr == -2) continue;
        const int64_t L = ld_int(lane->dtype, lane->data, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), len = cu_at(cu, e + 1) - b0;
        const int64_t p0 = pr >= 0 ? cu_at(cu, pr) : 0;
        const float s = ldt(sg, offlin(sg, e));
        const float init = std::sqrt(1.0f + s * s), cin = 1.0f / std::sqrt(s * s + 1.0f);
        const bool first = (ld_int(fl->dtype, fl->data, offlin(fl, e)) & kFirst) != 0;
        const uint64_t sd = kd_seed(seed, e);
        REF_PAR_FOR(len * n * 8)
        for (int64_t k = 0; k < len; ++k) {
            const int64_t sb = rowoff(st, L * stride + k, n), ss = laststride(st);
            const int64_t yb = rowoff(y, b0 + k, n), ys = laststride(y);
            const int64_t pb = pr >= 0 ? rowoff(y, p0 + k, n) : 0;
            for (int64_t c = 0; c < n; ++c) {
                if (first) stt(st, sb + c * ss, rad_philox_normal(sd, (uint64_t)(k * n + c)) * init);
                const float v = ldt(st, sb + c * ss) * cin;
                stt(y, yb + c * ys, v);
                if (pr >= 0) stt(y, pb + c * ys, v);
            }
        }
    }
    return RAD_OK;
}

/* One sampler step for every entry that moves a lane (pair != -2), x the lane's f32 state:
 *
 *   D = eps: x - sigma o      v: x / (sigma^2 + 1) - o sigma / sqrt(sigma^2 + 1)
 *   D = pair >= 0 ? D(o_u) + cfg (D(o_c) - D(o_u)) : D(o_c)
 *
 * then by the entry's sampler: 0 Euler, x + (sigma_next - sigma)(x - D) / sigma; 1 Euler ancestral
 * (eta 1): the step to sigma_down plus sigma_up times the seed's stream 1 + step; 2 DPM++ 2M:
 * (sigma_next / sigma) x - expm1(-h) D', D' blending D with the lane's last estimate after the first
 * step, and D itself at sigma_next 0. Flag 4 writes D to the preview; every step leaves D in
 * `hist` for the next. */
int ref_kd_step(const RadArgs* a, RadStream) {
    if (!have(a, 14)) return RAD_E_INVAL;
    const RadTensor *O = &a->t[0], *sg = &a->t[1], *sgn = &a->t[2], *sgp = &a->t[3];
    const RadTensor *cfg = &a->t[4], *pair = &a->t[5], *lane = &a->t[6], *fl = &a->t[7];
    const RadTensor *seed = &a->t[8], *smp = &a->t[9], *stp = &a->t[10], *cu = &a->t[11];
    const RadTensor *st = &a->t[12], *hs = &a->t[13], *pv = t_in(a, 14);
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    const char* pred = p_str(a, "pred", "eps");
    const bool v_pred = pred && !std::strcmp(pred, "v");
    if (!v_pred && (!pred || std::strcmp(pred, "eps"))) return RAD_E_INVAL;
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1, st_rows = rows_of(st, n), o_rows = rows_of(O, n);
    if (n_seq <= 0 || cu_at(cu, n_seq) <= cu_at(cu, 0)) return RAD_E_SHAPE;
    for (const RadTensor* t : { sg, sgn, sgp, cfg, pair, lane, fl, smp, stp })
        if (numel(t) < n_seq) return RAD_E_SHAPE;
    if (numel(seed) < 2 * n_seq || rows_of(hs, n) < st_rows || (pv && rows_of(pv, n) < st_rows))
        return RAD_E_SHAPE;
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t pr = ld_int(pair->dtype, pair->data, offlin(pair, e));
        if (pr == -2) continue;
        const int64_t L = ld_int(lane->dtype, lane->data, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), len = cu_at(cu, e + 1) - b0;
        if (L < 0 || len > stride || (L + 1) * stride > st_rows || b0 < 0 || b0 + len > o_rows)
            return RAD_E_SHAPE;
        if (pr >= n_seq || (pr >= 0 && (cu_at(cu, pr + 1) - cu_at(cu, pr) != len ||
                                        cu_at(cu, pr) < 0 || cu_at(cu, pr) + len > o_rows)))
            return RAD_E_SHAPE;
        const int64_t f = ld_int(fl->dtype, fl->data, offlin(fl, e));
        const int64_t sm = ld_int(smp->dtype, smp->data, offlin(smp, e));
        if ((f & kInpaint) || sm < kSamplerEuler || sm > kSamplerDpmpp2m) return RAD_E_UNSUPPORTED;
        if ((f & kPreview) && !pv) return RAD_E_INVAL;
    }
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t pr = ld_int(pair->dtype, pair->data, offlin(pair, e));
        if (pr == -2) continue;
        const int64_t L = ld_int(lane->dtype, lane->data, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), len = cu_at(cu, e + 1) - b0;
        const int64_t p0 = pr >= 0 ? cu_at(cu, pr) : 0;
        const int64_t f = ld_int(fl->dtype, fl->data, offlin(fl, e));
        const int64_t sm = ld_int(smp->dtype, smp->data, offlin(smp, e));
        const uint64_t stream = (uint64_t)(uint32_t)(1 + ld_int(stp->dtype, stp->data, offlin(stp, e)));
        const KdEntry K = kd_entry(ldt(sg, offlin(sg, e)), ldt(sgn, offlin(sgn, e)),
                                   ldt(sgp, offlin(sgp, e)), ldt(cfg, offlin(cfg, e)), sm,
                                   (f & kFirst) != 0);
        const uint64_t sd = kd_seed(seed, e);
        REF_PAR_FOR(len * n * 8)
        for (int64_t k = 0; k < len; ++k) {
            const int64_t ob = rowoff(O, b0 + k, n), os = laststride(O);
            const int64_t ub = pr >= 0 ? rowoff(O, p0 + k, n) : 0;
            const int64_t sb = rowoff(st, L * stride + k, n), ss = laststride(st);
            const int64_t hb = rowoff(hs, L * stride + k, n), hss = laststride(hs);
            for (int64_t c = 0; c < n; ++c) {
                const float x = ldt(st, sb + c * ss);
                float D = kd_denoised(K, v_pred, x, ldt(O, ob + c * os));
                if (pr >= 0) {
                    const float Du = kd_denoised(K, v_pred, x, ldt(O, ub + c * os));
                    D = Du + K.w * (D - Du);
                }
                if (f & kPreview)
                    stt(pv, rowoff(pv, L * stride + k, n) + c * laststride(pv), D);
                const float nz = sm == kSamplerEulerAncestral && K.sn > 0.0f
                                     ? rad_philox_normal_s(sd, stream, (uint64_t)(k * n + c)) : 0.0f;
                stt(st, sb + c * ss, kd_next(K, sm, x, D, ldt(hs, hb + c * hss), nz));
                stt(hs, hb + c * hss, D);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ chan_affine */
/* A per-channel affine from two parameter lists (a latent's mean and deviation are configuration,
 * not weights), each value rounded to bf16 as upstream holds it: mode `fma` is
 * bf16(bf16(x * mul) + add), `inv` is bf16(bf16(x - add) / mul). With `round` 0 the lists stay
 * f32 and so does the arithmetic, rounded once into y: rd(f32(x * mul) + add) and
 * rd(f32(x - add) / mul), as an f32 model denormalises an f32 latent. */
int ref_chan_affine(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), rnd = p_int(a, "round", 1);
    if (rnd < 0 || rnd > 1) return RAD_E_INVAL;
    std::vector<float> mul, add;
    if (n <= 0 || parse_list(p_str(a, "mul", ""), &mul, rnd) != n ||
        parse_list(p_str(a, "add", ""), &add, rnd) != n)
        return RAD_E_SHAPE;
    const char* mode = p_str(a, "mode", "fma");
    const bool inv = mode && !std::strcmp(mode, "inv");
    if (!inv && mode && std::strcmp(mode, "fma")) return RAD_E_INVAL;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || rows_of(x, n) < rows) return RAD_E_SHAPE;
    /* The inner product's or difference's rounding: y's dtype, or none. */
    const uint32_t mid = rnd ? y->dtype : (uint32_t)RAD_F32;
    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t c = 0; c < n; ++c) {
            const float v = ldt(x, xb + c * xs);
            stt(y, yb + c * ys, inv ? round_dt(mid, v - add[c]) / mul[c]
                                    : round_dt(mid, v * mul[c]) + add[c]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv2d_nhwc */
/* An image convolution, channels innermost: x is [Hin, Win * cin], y [Hout, Wout * cout], both
 * dense, and the sizes are read off them. w is PyTorch's [cout, cin, k, k] and b [cout].
 *
 *   up 2      the input is first upsampled 2x nearest-neighbour (`nearest-exact` at an exact
 *             factor of two is the same): the conv reads x[y / 2, x / 2]
 *   pad same  k / 2 zeros on every side
 *   pad down  no zeros above and left, one below and right: ZeroPad2d((0, 1, 0, 1)) then the conv,
 *             as a stride-2 downsample does
 *   pad causal  k - 1 zeros above, none below, k / 2 a side across: a convolution causal along
 *             the rows (LTX-2's audio VAE, whose rows are time)
 *
 * f32 accumulation over (kh, kw, cin), the bias added, one rounding. */
int ref_conv2d_nhwc(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *b = &a->t[2], *y = &a->t[3];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0), k = p_int(a, "k", 0);
    const int64_t stride = p_int(a, "stride", 1), up = p_int(a, "up", 1);
    const char* pad = p_str(a, "pad", "same");
    const bool down = pad && !std::strcmp(pad, "down");
    const bool causal = pad && !std::strcmp(pad, "causal");
    if (cin <= 0 || cout <= 0 || k <= 0 || stride <= 0 || up <= 0) return RAD_E_SHAPE;
    if (x->rank != 2 || y->rank != 2 || x->shape[1] % cin || y->shape[1] % cout) return RAD_E_SHAPE;
    const int64_t Hin = x->shape[0], Win = x->shape[1] / cin;
    const int64_t Ho = y->shape[0], Wo = y->shape[1] / cout;
    if (Hin <= 0 || Win <= 0 || Ho <= 0 || Wo <= 0) return RAD_E_SHAPE;
    const int64_t Hv = Hin * up, Wv = Win * up;
    const int64_t pt = down ? 0 : causal ? k - 1 : k / 2, pb = down ? 1 : causal ? 0 : k / 2;
    const int64_t pl = down ? 0 : k / 2, pr = down ? 1 : k / 2;
    if ((Hv + pt + pb - k) / stride + 1 != Ho || (Wv + pl + pr - k) / stride + 1 != Wo)
        return RAD_E_SHAPE;
    if (numel(w) < cout * cin * k * k || numel(b) < cout) return RAD_E_SHAPE;
    const int64_t xr = x->stride[0], yr = y->stride[0];
    REF_PAR_FOR(Ho * Wo * cout * cin * k * k)
    for (int64_t oy = 0; oy < Ho; ++oy)
        for (int64_t ox = 0; ox < Wo; ++ox)
            for (int64_t co = 0; co < cout; ++co) {
                float acc = 0.0f;
                for (int64_t kh = 0; kh < k; ++kh) {
                    const int64_t iy = oy * stride + kh - pt;
                    if (iy < 0 || iy >= Hv) continue;
                    for (int64_t kw = 0; kw < k; ++kw) {
                        const int64_t ix = ox * stride + kw - pl;
                        if (ix < 0 || ix >= Wv) continue;
                        const int64_t xb = (iy / up) * xr + (ix / up) * cin;
                        for (int64_t ci = 0; ci < cin; ++ci)
                            acc += ldt(w, ((co * cin + ci) * k + kh) * k + kw) * ldt(x, xb + ci);
                    }
                }
                stt(y, oy * yr + ox * cout + co, acc + ldt(b, co));
            }
    return RAD_OK;
}

/* ------------------------------------------------------------------ im2col_nhwc */
/* conv2d_nhwc's receptive fields as rows, so a convolution is a GEMM against its own weight read as
 * [cout, cin k k]: x is [Hin, Win * cin] (rows at their own pitch), y [Ho * Wo, cin * k * k], and
 * row oy * Wo + ox, column (ci k + i) k + j holds the value conv2d_nhwc multiplies weight
 * [o, ci, i, j] by -- x at the tap's position read through `up` and `pad` as conv2d_nhwc reads it,
 * 0 in the padding. The output size follows from x's; y must have exactly Ho * Wo rows. A copy:
 * the values are stored as they are (rounded only when y is narrower than x). */
int ref_im2col_nhwc(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t cin = p_int(a, "cin", 0), k = p_int(a, "k", 0);
    const int64_t stride = p_int(a, "stride", 1), up = p_int(a, "up", 1);
    const char* pad = p_str(a, "pad", "same");
    const bool down = pad && !std::strcmp(pad, "down");
    const bool causal = pad && !std::strcmp(pad, "causal");
    if (cin <= 0 || k <= 0 || stride <= 0 || up <= 0) return RAD_E_SHAPE;
    if (x->rank != 2 || x->shape[1] % cin) return RAD_E_SHAPE;
    const int64_t Hin = x->shape[0], Win = x->shape[1] / cin, kk = cin * k * k;
    if (Hin <= 0 || Win <= 0) return RAD_E_SHAPE;
    const int64_t Hv = Hin * up, Wv = Win * up;
    const int64_t pt = down ? 0 : causal ? k - 1 : k / 2, pb = down ? 1 : causal ? 0 : k / 2;
    const int64_t pl = down ? 0 : k / 2, pr = down ? 1 : k / 2;
    const int64_t Ho = (Hv + pt + pb - k) / stride + 1, Wo = (Wv + pl + pr - k) / stride + 1;
    if (Ho <= 0 || Wo <= 0 || rows_of(y, kk) != Ho * Wo) return RAD_E_SHAPE;
    const int64_t xr = x->stride[0], xs = laststride(x), ys = laststride(y);
    REF_PAR_FOR(Ho * Wo * kk)
    for (int64_t oy = 0; oy < Ho; ++oy)
        for (int64_t ox = 0; ox < Wo; ++ox) {
            const int64_t yb = rowoff(y, oy * Wo + ox, kk);
            for (int64_t ci = 0; ci < cin; ++ci)
                for (int64_t i = 0; i < k; ++i) {
                    const int64_t iy = oy * stride + i - pt;
                    for (int64_t j = 0; j < k; ++j) {
                        const int64_t ix = ox * stride + j - pl;
                        const bool in = iy >= 0 && iy < Hv && ix >= 0 && ix < Wv;
                        const float v = in ? ldt(x, (iy / up) * xr + ((ix / up) * cin + ci) * xs)
                                           : 0.0f;
                        stt(y, yb + ((ci * k + i) * k + j) * ys, v);
                    }
                }
        }
    return RAD_OK;
}

/* ------------------------------------------------------------------ l2norm_chan */
/* An image norm over each pixel's channels, as upstream's three bf16 tensors:
 * n1 = bf16(x / max(||x||, 1e-12)), n2 = bf16(n1 * sqrt(n)), y = bf16(n2 * gamma), then `act`
 * (none, or silu: bf16(y * sigmoid(y))). */
int ref_l2norm_chan(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *g = &a->t[1], *y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || numel(g) < n) return RAD_E_SHAPE;
    const char* act = p_str(a, "act", "none");
    const bool silu = act && !std::strcmp(act, "silu");
    if (!silu && act && std::strcmp(act, "none")) return RAD_E_INVAL;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || rows_of(x, n) < rows) return RAD_E_SHAPE;
    const float sq = (float)std::sqrt((double)n);
    const uint32_t dt = y->dtype;
    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        float ss = 0.0f;
        for (int64_t c = 0; c < n; ++c) { const float v = ldt(x, xb + c * xs); ss += v * v; }
        const float nrm = std::max(std::sqrt(ss), 1e-12f);
        for (int64_t c = 0; c < n; ++c) {
            const float n1 = round_dt(dt, ldt(x, xb + c * xs) / nrm);
            const float n2 = round_dt(dt, n1 * sq);
            float v = round_dt(dt, n2 * ldt(g, c));
            if (silu) v = v / (1.0f + ref_exp(-v));
            stt(y, yb + c * ys, v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ d2s_add / s2d_avg_add */
/* d2s_add, the decoder's upsampling shortcut (Wan's DupUp3D on one frame): x [H, W * cin] repeated
 * `rep = cout * F / cin` times per channel and dealt into a 2x larger y [2H, 2W * cout], added:
 * y[2r+i, 2c+j, o] += x[r, c, (o * F + off + 2i + j) / rep], F = 4 * ft, off = 4 when ft is 2
 * (a first chunk keeps the second of the two frames a temporal factor makes). */
int ref_d2s_add(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0), ft = p_int(a, "ft", 1);
    if (cin <= 0 || cout <= 0 || (ft != 1 && ft != 2)) return RAD_E_SHAPE;
    if (x->rank != 2 || y->rank != 2 || x->shape[1] % cin || y->shape[1] % cout) return RAD_E_SHAPE;
    const int64_t H = x->shape[0], W = x->shape[1] / cin;
    if (H <= 0 || W <= 0) return RAD_E_SHAPE;
    if (y->shape[0] != 2 * H || y->shape[1] / cout != 2 * W) return RAD_E_SHAPE;
    const int64_t F = 4 * ft, off = ft == 2 ? 4 : 0;
    if ((cout * F) % cin) return RAD_E_SHAPE;
    const int64_t rep = cout * F / cin;
    const int64_t xr = x->stride[0], yr = y->stride[0];
    REF_PAR_FOR(4 * H * W * cout)
    for (int64_t oy = 0; oy < 2 * H; ++oy)
        for (int64_t ox = 0; ox < 2 * W; ++ox)
            for (int64_t o = 0; o < cout; ++o) {
                const int64_t i = oy & 1, j = ox & 1;
                const int64_t ci = (o * F + off + 2 * i + j) / rep;
                const int64_t yo = oy * yr + ox * cout + o;
                stt(y, yo, ldt(y, yo) + ldt(x, (oy / 2) * xr + (ox / 2) * cin + ci));
            }
    return RAD_OK;
}

/* s2d_avg_add, the encoder's downsampling shortcut (Wan's AvgDown3D on one frame): each 2x2 block
 * of x [2H, 2W * cin] dealt into channels and averaged in groups of gs = cin * F / cout into
 * y [H, W * cout], added. A dealt channel (c, t, i, j) sits at c * F + 4t + 2i + j; with ft 2,
 * frame t == 0 is upstream's zero padding frame, which the mean still counts. */
int ref_s2d_avg_add(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0), ft = p_int(a, "ft", 1);
    if (cin <= 0 || cout <= 0 || (ft != 1 && ft != 2)) return RAD_E_SHAPE;
    if (x->rank != 2 || y->rank != 2 || x->shape[1] % cin || y->shape[1] % cout) return RAD_E_SHAPE;
    const int64_t H = y->shape[0], W = y->shape[1] / cout;
    if (H <= 0 || W <= 0) return RAD_E_SHAPE;
    if (x->shape[0] != 2 * H || x->shape[1] / cin != 2 * W) return RAD_E_SHAPE;
    const int64_t F = 4 * ft;
    if ((cin * F) % cout) return RAD_E_SHAPE;
    const int64_t gs = cin * F / cout;
    const int64_t xr = x->stride[0], yr = y->stride[0];
    REF_PAR_FOR(H * W * cout * gs)
    for (int64_t oy = 0; oy < H; ++oy)
        for (int64_t ox = 0; ox < W; ++ox)
            for (int64_t o = 0; o < cout; ++o) {
                float acc = 0.0f;
                for (int64_t gg = 0; gg < gs; ++gg) {
                    const int64_t g = o * gs + gg;
                    const int64_t c = g / F, rem = g % F, t = rem / 4, i = (rem % 4) / 2, j = rem % 2;
                    if (ft == 2 && t == 0) continue;
                    acc += ldt(x, (2 * oy + i) * xr + (2 * ox + j) * cin + c);
                }
                const int64_t yo = oy * yr + ox * cout + o;
                stt(y, yo, ldt(y, yo) + round_dt(y->dtype, acc / (float)gs));
            }
    return RAD_OK;
}

/* ------------------------------------------------------------------ to_rgba8 */
/* A decoded image to bytes the way diffusers' postprocess makes them: v = clamp(x, -1, 1),
 * u = clamp(bf16(bf16(v / 2) + 0.5), 0, 1), byte = round-half-even(f32(u) * 255). */
int ref_to_rgba8(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || rows_of(x, n) < rows || y->dtype != RAD_U8) return RAD_E_SHAPE;
    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t c = 0; c < n; ++c) {
            float v = ldt(x, xb + c * xs);
            v = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
            float u = rbf(rbf(v * 0.5f) + 0.5f);
            u = u < 0.0f ? 0.0f : (u > 1.0f ? 1.0f : u);
            ((uint8_t*)y->data)[yb + c * ys] = (uint8_t)std::nearbyintf(u * 255.0f);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gelu_tanh, scatter_add_rows */
int ref_gelu_tanh(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || rows_of(x, n) < rows) return RAD_E_SHAPE;
    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, act_gelu_tanh(ldt(x, xb + c * xs)));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gelu_quick */
/* CLIP's quick_gelu as torch evaluates it in a reduced dtype, three stored tensors:
 * a = rd(1.702 x), s = rd(sigmoid(a)), y = rd(x s). */
int ref_gelu_quick(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || rows_of(x, n) < rows) return RAD_E_SHAPE;
    const uint32_t dt = y->dtype;
    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t c = 0; c < n; ++c) {
            const float v = ldt(x, xb + c * xs);
            const float t = round_dt(dt, 1.702f * v);
            const float s = round_dt(dt, 1.0f / (1.0f + std::exp(-t)));
            stt(y, yb + c * ys, v * s);
        }
    }
    return RAD_OK;
}

/* x[idx[i]] = x[idx[i]] + v[i], rounded to x's dtype; a negative index is a skip. */
int ref_scatter_add_rows(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *v = &a->t[0], *idx = &a->t[1], *x = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t M = rows_of(v, n), dst_rows = rows_of(x, n);
    if (M <= 0 || numel(idx) < M) return RAD_E_SHAPE;
    for (int64_t m = 0; m < M; ++m)
        if (ld_int(idx->dtype, idx->data, offlin(idx, m)) >= dst_rows) return RAD_E_INVAL;
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r = ld_int(idx->dtype, idx->data, offlin(idx, m));
        if (r < 0) continue;
        const int64_t vb = rowoff(v, m, n), vs = laststride(v);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) stt(x, xb + i * xs, ldt(x, xb + i * xs) + ldt(v, vb + i * vs));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ attn_prefix_self */
/* Attention over two key sources: the sequence's prefix (`seqused[s]` positions in the paged cache)
 * and the step's own rows (`k` and `v` beside `q`, not cached). Self row i sees every prefix key
 * and its own rows j <= i with `causal`, all of them otherwise.
 *
 * For rows recomputed each step against a fixed prefix: an image's latent rows against the
 * conditioning, or one conditioning segment against the segments before it. */
/* quant_act_i8g's rule for one row of `n`: the scale (amax/127, 1 for an all-zero row) and the
 * codes rint(clamp(x * (1/scale), +-127)), in f32 as the device computes them. `code` may be `x`:
 * each code is written after its value is read. */
float ref::ref_q8_row(const float* x, int64_t n, float* code) {
    float am = 0.0f;
    for (int64_t d = 0; d < n; ++d) am = std::max(am, std::fabs(x[d]));
    const float sc = am > 0.0f ? am * (1.0f / 127.0f) : 1.0f;
    const float inv = 1.0f / sc;
    for (int64_t d = 0; d < n; ++d)
        code[d] = std::rint(std::min(std::max(x[d] * inv, -127.0f), 127.0f));
    return sc;
}

int ref_attn_prefix_self(const RadArgs* a, RadStream) {
    if (a->n_t < 12) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *cu = &a->t[6], *o = &a->t[11];
    /* The paged prefix (cache, block table, per-sequence prefix lengths), all three or none: with
     * none every sequence's prefix is empty. */
    const RadTensor *kv = t_in(a, 3), *bt = t_in(a, 4), *su = t_in(a, 5);
    if ((kv == nullptr) != (bt == nullptr) || (kv == nullptr) != (su == nullptr)) return RAD_E_INVAL;
    const RadTensor *kdes = t_in(a, 7), *vdes = t_in(a, 8);
    /* The int8 Q.K form: own keys' codes and scales are given; everything else is quantised here. */
    const RadTensor *k8 = t_in(a, 9), *ks = t_in(a, 10);
    const char* qk = p_str(a, "qk", "bf16");
    const bool i8 = std::strcmp(qk, "i8") == 0;
    if (i8 != (k8 != nullptr) || i8 != (ks != nullptr)) return RAD_E_INVAL;
    if (i8 && (kdes || vdes)) return RAD_E_UNSUPPORTED;
    /* The fp8 P.V form: `v` E4M3 codes with a scale per column in `v_scale` (quant_cols_fp8), each
     * weight exp(s - max) taken to E4M3 at 448 times its value, o = sum(w v) v_scale /
     * (448 sum exp(s - max)); prefix-free only. */
    const RadTensor* vsc = a->n_t > 12 ? t_in(a, 12) : nullptr;
    const char* pv = p_str(a, "pv", "bf16");
    const bool pv8 = std::strcmp(pv, "fp8") == 0;
    if ((!pv8 && std::strcmp(pv, "bf16") != 0) || pv8 != (vsc != nullptr)) return RAD_E_INVAL;
    if (pv8 && (kv || kdes || vdes)) return RAD_E_UNSUPPORTED;
    const int64_t head_dim = p_int(a, "head_dim", 0), gqa = p_int(a, "gqa", 1);
    const int causal = (int)p_int(a, "causal", 0);
    const int64_t block_size = !kv ? 1 : kv->rank == 4 ? kv->shape[2] : p_int(a, "block_size", 0);
    /* The widest head held in the stack rows below. */
    constexpr int64_t kHdMax = 1024;
    if (head_dim <= 0 || gqa <= 0 || block_size <= 0) return RAD_E_SHAPE;
    if (head_dim > kHdMax) return RAD_E_UNSUPPORTED;
    /* Without a cache the key heads are the own keys' width over head_dim. */
    const int64_t kv_heads = kv ? (kv->rank == 4 ? kv->shape[1] : 0)
                           : k->rank >= 3 ? k->shape[k->rank - 2]
                           : k->rank == 2 ? k->shape[1] / head_dim : 0;
    if (kv_heads <= 0) return RAD_E_SHAPE;
    const int64_t q_heads = kv_heads * gqa, n_seq = numel(cu) - 1;
    if (n_seq <= 0 || (su && numel(su) < n_seq)) return RAD_E_SHAPE;
    /* The rows are cu's span, which need not start at 0: a segment of a longer sequence is passed
     * as its two bounding offsets. */
    const int64_t r_first = cu_at(cu, 0), rows = cu_at(cu, n_seq) - r_first;
    if (rows <= 0) return RAD_E_SHAPE;
    const int64_t max_blocks = !bt ? 0 : bt->rank >= 2 ? bt->shape[1] : numel(bt) / n_seq;
    const int64_t bt_rs = bt && bt->rank >= 2 ? bt->stride[0] : max_blocks;
    const int64_t bt_cs = bt && bt->rank >= 2 ? bt->stride[1] : 1;
    /* Every prefix key must have a block-table entry. */
    for (int64_t s = 0; su && s < n_seq; ++s)
        if (ld_int(su->dtype, su->data, offlin(su, s)) > max_blocks * block_size) return RAD_E_SHAPE;
    const int64_t q_rs = row_stride(q, q_heads * head_dim), k_rs = row_stride(k, kv_heads * head_dim);
    const int64_t v_rs = row_stride(v, kv_heads * head_dim), o_rs = row_stride(o, q_heads * head_dim);
    const int64_t k8_rs = i8 ? row_stride(k8, kv_heads * head_dim) : 0;
    const int64_t ks_rs = i8 ? row_stride(ks, kv_heads) : 0;
    const int64_t sb = kv ? kv->stride[0] : 0, sh = kv ? kv->stride[1] : 0;
    const int64_t so = kv ? kv->stride[2] : 0;
    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)head_dim));
    REF_PAR_FOR_DYN(rows * q_heads * head_dim * 64)
    for (int64_t idx = 0; idx < rows * q_heads; ++idx) {
        const int64_t h = idx % q_heads, r = r_first + idx / q_heads, kh = h / gqa;
        const int64_t s = seq_of(cu, n_seq, r), b0 = cu_at(cu, s), b1 = cu_at(cu, s + 1);
        const int64_t i = r - b0, P = su ? ld_int(su->dtype, su->data, offlin(su, s)) : 0;
        const float kds = kdes ? ldt(kdes, s * kv_heads + kh) : 1.0f;
        const float vds = vdes ? ldt(vdes, s * kv_heads + kh) : 1.0f;
        const int64_t qb = r * q_rs + h * head_dim;
        const int64_t nself = causal ? i + 1 : b1 - b0;
        auto pkey = [&](int64_t j, int64_t d, bool val) -> float {
            const int64_t blk = ld_int(bt->dtype, bt->data, s * bt_rs + (j / block_size) * bt_cs);
            if (blk < 0) return 0.0f;
            return ldt(kv, blk * sb + kh * sh + (j % block_size) * so + (val ? head_dim : 0) + d) *
                   (val ? vds : kds);
        };
        float xq[kHdMax], cq[kHdMax], ck[kHdMax];
        float sq = 1.0f;
        if (i8) {
            for (int64_t d = 0; d < head_dim; ++d) xq[d] = ldt(q, qb + d);
            sq = ref_q8_row(xq, head_dim, cq);
        }
        auto dot = [&](int64_t j) -> float {
            if (i8) {
                float sk = 0.0f;
                if (j < P) {
                    for (int64_t d = 0; d < head_dim; ++d) ck[d] = pkey(j, d, false);
                    sk = ref_q8_row(ck, head_dim, ck);
                } else {
                    const int64_t row = b0 + j - P;
                    for (int64_t d = 0; d < head_dim; ++d)
                        ck[d] = ldt(k8, row * k8_rs + kh * head_dim + d);
                    sk = ldt(ks, row * ks_rs + kh);
                }
                /* Integers below 2^24 in magnitude: the f32 sum is exact. */
                float acc = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) acc += cq[d] * ck[d];
                return acc * sq * sk * scale;
            }
            float acc = 0.0f;
            if (j < P) {
                for (int64_t d = 0; d < head_dim; ++d) acc += ldt(q, qb + d) * pkey(j, d, false);
            } else {
                const int64_t kb = (b0 + j - P) * k_rs + kh * head_dim;
                for (int64_t d = 0; d < head_dim; ++d) acc += ldt(q, qb + d) * ldt(k, kb + d);
            }
            return acc * scale;
        };
        const int64_t nk = P + nself;
        float mx = -INFINITY;
        for (int64_t j = 0; j < nk; ++j) mx = std::max(mx, dot(j));
        float acc[kHdMax];
        for (int64_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;
        float sum = 0.0f;
        for (int64_t j = 0; j < nk; ++j) {
            const float p = ref_exp(dot(j) - mx);
            sum += p;
            const float wgt = pv8 ? round_dt(RAD_F8E4M3, 448.0f * p) : p;
            if (j < P) {
                for (int64_t d = 0; d < head_dim; ++d) acc[d] += wgt * pkey(j, d, true);
            } else {
                const int64_t vb = (b0 + j - P) * v_rs + kh * head_dim;
                for (int64_t d = 0; d < head_dim; ++d) acc[d] += wgt * ldt(v, vb + d);
            }
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        const int64_t ob = r * o_rs + h * head_dim;
        if (pv8) {
            const float inv8 = sum > 0.0f ? 1.0f / (448.0f * sum) : 0.0f;
            for (int64_t d = 0; d < head_dim; ++d)
                stt(o, ob + d, acc[d] * ldt(vsc, offlin(vsc, kh * head_dim + d)) * inv8);
        } else {
            for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d, acc[d] * inv);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ table_add */
/* y[r, l n + c] = x[r, c] + t[l, c] for every row of `x` and every row l of the table: a shared
 * modulation plus each block's own learned table (Krea 2's, LTX-2's), all blocks in one pass. The
 * sum is f32, rounded once to y's dtype. `x` and `y` are read at their own row pitches. */
int ref_table_add(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *T = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", 0), L = p_int(a, "L", 0);
    if (n <= 0 || L <= 0 || numel(T) < L * n) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, L * n);
    if (rows <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;
    const int64_t xs = laststride(X), ys = laststride(Y), ts = laststride(T);
    REF_PAR_FOR(rows * L * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, n), yb = rowoff(Y, r, L * n);
        for (int64_t l = 0; l < L; ++l)
            for (int64_t c = 0; c < n; ++c)
                stt(Y, yb + (l * n + c) * ys,
                    ldt(X, xb + c * xs) + ldt(T, (l * n + c) * ts));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ mix_rows */
/* y[r, c] = sum_i w[i] x[r k + i, c], i < k: a Linear(k, 1) across k consecutive rows (Krea 2's
 * text fusion folds twelve tapped layers of a token into one row). f32 sum in ascending i, rounded
 * once to y's dtype. */
int ref_mix_rows(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *W = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", 0), k = p_int(a, "k", 0);
    if (n <= 0 || k <= 0 || numel(W) < k) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || rows_of(X, n) < rows * k) return RAD_E_SHAPE;
    const int64_t xs = laststride(X), ys = laststride(Y), ws = laststride(W);
    REF_PAR_FOR(rows * n * k)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t yb = rowoff(Y, r, n);
        for (int64_t c = 0; c < n; ++c) {
            float acc = 0.0f;
            for (int64_t i = 0; i < k; ++i)
                acc += ldt(W, i * ws) * ldt(X, rowoff(X, r * k + i, n) + c * xs);
            stt(Y, yb + c * ys, acc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ chan_stats */
/* A latent normalised or denormalised by per-channel statistics a VAE keeps as weights: `mean` and
 * `s` [n], `s` a standard deviation, or with `var` 1 a variance (std = sqrt(s + eps), a BatchNorm's
 * running statistics). Each statistic is rounded to y's dtype as the reference casts it, and
 * every step is a tensor the reference stores: `denorm` y = rd(rd(x std) + mean), `norm`
 * y = rd(rd(x - mean) / std). */
int ref_chan_stats(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Mn = &a->t[1], *Sd = &a->t[2], *Y = &a->t[3];
    const int64_t n = p_int(a, "n", 0);
    const bool var = p_int(a, "var", 0) != 0;
    const float eps = p_f32(a, "eps", 0.0f);
    const char* mode = p_str(a, "mode", "denorm");
    const bool norm = std::strcmp(mode, "norm") == 0;
    if (!norm && std::strcmp(mode, "denorm") != 0) return RAD_E_INVAL;
    if (n <= 0 || numel(Mn) < n || numel(Sd) < n) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;
    const uint32_t dt = Y->dtype;
    std::vector<float> mu((size_t)n), sd((size_t)n);
    for (int64_t c = 0; c < n; ++c) {
        mu[(size_t)c] = round_dt(dt, ldt(Mn, c * laststride(Mn)));
        const float v = ldt(Sd, c * laststride(Sd));
        sd[(size_t)c] = round_dt(dt, var ? std::sqrt(v + eps) : v);
    }
    const int64_t xs = laststride(X), ys = laststride(Y);
    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, n), yb = rowoff(Y, r, n);
        for (int64_t c = 0; c < n; ++c) {
            const float x = ldt(X, xb + c * xs);
            const float y = norm ? round_dt(dt, round_dt(dt, x - mu[(size_t)c]) / sd[(size_t)c])
                                 : round_dt(dt, round_dt(dt, x * sd[(size_t)c]) + mu[(size_t)c]);
            stt(Y, yb + c * ys, y);
        }
    }
    return RAD_OK;
}
