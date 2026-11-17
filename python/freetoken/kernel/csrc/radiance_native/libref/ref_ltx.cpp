/* ref_ltx.cpp -- LTX-2's ops (docs/OPS.md, "LTX-2"): its rotary over fractional coordinates, the
 * per-head attention gate, and the sampler step that combines a step's guidance passes in the
 * clean-estimate space.
 *
 * Every rounding follows the upstream model, which runs its transformer in bf16 and its rotary,
 * guidance and sampler in f32 (diffusers' transformer_ltx2.py and pipeline_ltx2.py).
 */
#include "ref_common.h"
#include "ref_gemm.h"
#include "ref_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace ref;

namespace {

inline int64_t ldi(const RadTensor* t, int64_t i) { return ld_int(t->dtype, t->data, offlin(t, i)); }

/* torch.linspace(0, 1, steps) in f64: the first half up from the start, the second down from the
 * end, as torch computes it. */
double linspace01(int64_t i, int64_t steps) {
    if (steps <= 1) return 0.0;
    const double step = 1.0 / (double)(steps - 1);
    return i < steps / 2 ? step * (double)i : 1.0 - step * (double)(steps - 1 - i);
}

int parse_list_f32(const char* s, std::vector<float>* out) {
    out->clear();
    while (s && *s) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end == s) break;
        out->push_back((float)v);
        s = end;
    }
    return (int)out->size();
}

/* Element (d, r) of a [C, M] operand at its own strides; an integer operand reads as its value. */
float pos_at(const RadTensor* p, int64_t d, int64_t r) {
    const int64_t off = p->rank >= 2 ? d * p->stride[0] + r * p->stride[1] : r * laststride(p);
    return ldt(p, off);
}

}  /* namespace */

/* ------------------------------------------------------------------ rope_split */
/* LTX-2's rotary (LTX2AudioVideoRotaryPosEmbed, "split"): over the full width `dim` of all heads,
 * P = dim / 2 pairs, the first `pad` = P - steps n_dims of them unrotated and the rest the
 * frequency ladder freq[k] = f32(theta^linspace(0, 1, steps)[k] pi / 2) (f64 before the cast),
 * steps = P / n_dims, its pairs interleaving the coordinates: pair pad + k n_dims + d turns by
 * coordinate d at freq[k]. A coordinate c becomes the angle a = f32(f32(2 f32(c / max[d])) - 1)
 * freq[k]. Head h owns pairs [h hd/2, (h+1) hd/2) and rotates its halves against each other,
 *
 *   o1 = f32(f32(x1 cos a) - f32(x2 sin a))   o2 = f32(f32(x2 cos a) + f32(x1 sin a))
 *
 * rounded to x's dtype. `head0` is the first head this operand holds (a rank's share). `x` is
 * [M, >= n_head head_dim] at its own row pitch; `positions` [n_dims, M], f32 or integer, at its own
 * strides. */
int ref_rope_split(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *P = &a->t[1];
    const int64_t nh = p_int(a, "n_head", 0), hd = p_int(a, "head_dim", 0);
    const int64_t dim = p_int(a, "dim", 0), nd = p_int(a, "n_dims", 0), h0 = p_int(a, "head0", 0);
    const double theta = (double)p_f32(a, "theta", 10000.0f);
    std::vector<float> mx;
    if (nh <= 0 || hd <= 0 || (hd & 1) || nd <= 0 || dim < (h0 + nh) * hd ||
        parse_list_f32(p_str(a, "max", ""), &mx) != (int)nd)
        return RAD_E_SHAPE;
    const int64_t P2 = dim / 2, steps = P2 / nd, pad = P2 - steps * nd, half = hd / 2;
    const int64_t rows = rows_of(X, nh * hd);
    if (rows <= 0 || P->rank < 1) return RAD_E_SHAPE;
    std::vector<float> freq((size_t)steps);
    for (int64_t k = 0; k < steps; ++k)
        freq[(size_t)k] = (float)(std::pow(theta, linspace01(k, steps)) * M_PI / 2.0);
    const int64_t xs = laststride(X);
    REF_PAR_FOR(rows * nh * hd)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, nh * hd);
        for (int64_t h = 0; h < nh; ++h)
            for (int64_t i = 0; i < half; ++i) {
                const int64_t p = (h0 + h) * half + i;
                if (p < pad) continue;   /* cos 1, sin 0: the value as it is */
                const int64_t j = p - pad, d = j % nd, k = j / nd;
                const float g = pos_at(P, d, r) / mx[(size_t)d];
                const float ang = (2.0f * g - 1.0f) * freq[(size_t)k];
                const float c = (float)std::cos((double)ang), s = (float)std::sin((double)ang);
                const int64_t o1 = xb + (h * hd + i) * xs, o2 = xb + (h * hd + half + i) * xs;
                const float x1 = ldt(X, o1), x2 = ldt(X, o2);
                const float a1 = x1 * c, b1 = x2 * s, a2 = x2 * c, b2 = x1 * s;
                stt(X, o1, a1 - b1);
                stt(X, o2, a2 + b2);
            }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ head_gate */
/* LTX-2's gated attention: each head's output times 2 sigmoid of its logit,
 * x = rd(x rd(2 rd(sigmoid(g)))) (the sigmoid of the bf16 logit rounded, doubled exactly). `g` is
 * [M, >= n_head], `x` [M, >= n_head head_dim], each at its own row pitch. */
int ref_head_gate(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *G = &a->t[0], *X = &a->t[1];
    const int64_t nh = p_int(a, "n_head", 0), hd = p_int(a, "head_dim", 0);
    if (nh <= 0 || hd <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(X, nh * hd);
    if (rows <= 0 || rows_of(G, nh) < rows) return RAD_E_SHAPE;
    const int64_t xs = laststride(X), gs = laststride(G);
    REF_PAR_FOR(rows * nh * hd)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, nh * hd), gb = rowoff(G, r, nh);
        for (int64_t h = 0; h < nh; ++h) {
            const float lg = ldt(G, gb + h * gs);
            const float sg = round_dt(G->dtype, 1.0f / (1.0f + std::exp(-lg)));
            const float gate = round_dt(X->dtype, 2.0f * sg);
            for (int64_t i = 0; i < hd; ++i) {
                const int64_t o = xb + (h * hd + i) * xs;
                stt(X, o, ldt(X, o) * gate);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ltx_guide */
/* One LTX-2 sampler step over a modality's latent `x` (f32 [M, n]) from its guidance passes'
 * velocities (each [M, n] at its own pitch, read as f32 as the pipeline upcasts them):
 *
 *   x0_p = f32(x - f32(v_p sigma))                                 for every pass p
 *   g    = x0_c [+ f32(f32(gs - 1) f32(x0_c - x0_u))] [+ f32(stg f32(x0_c - x0_s))]
 *               [+ f32(f32(ms - 1) f32(x0_c - x0_m))]              left to right, f32
 *   with rescale r > 0:  g = f32(f32(r f32(g f32(std(x0_c) / std(g)))) + f32(f32(1 - r) g))
 *   v    = f32(f32(x - g) / sigma)        x = f32(x + f32(f32(sigma_next - sigma) v))
 *
 * A bracketed term is taken when its pass is bound and its coefficient is not zero (gs, ms 1;
 * stg 0). std is the unbiased deviation over all M n values, accumulated in f64. `scales` is f32
 * [4] (gs, stg, ms, r) and `sched` f32 [2] (sigma, sigma_next), both read on the device; `stats` is
 * the implementation's scratch (libref leaves it untouched). */
int ref_ltx_guide(const RadArgs* a, RadStream) {
    if (a->n_t < 8) return RAD_E_INVAL;
    const RadTensor *Vc = &a->t[0], *Vu = t_in(a, 1), *Vs = t_in(a, 2), *Vm = t_in(a, 3);
    const RadTensor *Sc = &a->t[4], *Sh = &a->t[5], *X = &a->t[7];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || numel(Sc) < 4 || numel(Sh) < 2) return RAD_E_SHAPE;
    const int64_t rows = rows_of(X, n);
    if (rows <= 0 || rows_of(Vc, n) < rows || (Vu && rows_of(Vu, n) < rows) ||
        (Vs && rows_of(Vs, n) < rows) || (Vm && rows_of(Vm, n) < rows))
        return RAD_E_SHAPE;
    const float gs = ldt(Sc, 0), stg = ldt(Sc, laststride(Sc)), ms = ldt(Sc, 2 * laststride(Sc));
    const float rr = ldt(Sc, 3 * laststride(Sc));
    const float sig = ldt(Sh, 0), sig_n = ldt(Sh, laststride(Sh));
    const bool use_u = Vu && gs != 1.0f, use_s = Vs && stg != 0.0f, use_m = Vm && ms != 1.0f;
    const float cu = gs - 1.0f, cm = ms - 1.0f;
    const int64_t N = rows * n;
    std::vector<float> x0c((size_t)N), g((size_t)N);
    auto x0 = [&](const RadTensor* V, int64_t r, int64_t c, float x) {
        return x - ldt(V, rowoff(V, r, n) + c * laststride(V)) * sig;
    };
    REF_PAR_FOR(N)
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < n; ++c) {
            const float x = ldt(X, rowoff(X, r, n) + c * laststride(X));
            const float xc = x0(Vc, r, c, x);
            float v = xc;
            if (use_u) v = v + cu * (xc - x0(Vu, r, c, x));
            if (use_s) v = v + stg * (xc - x0(Vs, r, c, x));
            if (use_m) v = v + cm * (xc - x0(Vm, r, c, x));
            x0c[(size_t)(r * n + c)] = xc;
            g[(size_t)(r * n + c)] = v;
        }
    float fac = 1.0f, keep = 1.0f;
    if (rr > 0.0f) {
        auto stdev = [&](const std::vector<float>& v) {
            double m = 0.0;
            for (float e : v) m += e;
            m /= (double)N;
            double ss = 0.0;
            for (float e : v) ss += ((double)e - m) * ((double)e - m);
            return (float)std::sqrt(ss / (double)(N > 1 ? N - 1 : 1));
        };
        fac = stdev(x0c) / stdev(g);
        keep = (float)(1.0 - (double)rr);
    }
    const float dt = sig_n - sig;
    REF_PAR_FOR(N)
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < n; ++c) {
            float v = g[(size_t)(r * n + c)];
            if (rr > 0.0f) v = rr * (v * fac) + keep * v;
            const int64_t o = rowoff(X, r, n) + c * laststride(X);
            const float x = ldt(X, o);
            const float vel = (x - v) / sig;
            stt(X, o, x + dt * vel);
        }
    return RAD_OK;
}

/* ------------------------------------------------------------------ pixel_norm */
/* An RMS norm over each row's n values as a model computes it in bf16 tensors, every step stored:
 *
 *   m = rd(sum_i rd(x_i^2) / n)        (f32 sum in ascending i)
 *   rsqrt form: z_i = rd(x_i rd(1 / sqrt(rd(m + eps))))
 *   div form:   z_i = rd(x_i / rd(sqrt(rd(m + eps))))
 *
 * then with `act` silu z = rd(z / (1 + exp(-z))), and with `mult` z = rd(z mult) (the text
 * connectors' scale to the DiT's width). rd is y's dtype. LTX-2's VAE PerChannelRMSNorm
 * and audio PixelNorm are the div form; its text connectors' per-layer norm the rsqrt form, whose
 * output `ileave` interleaves: row r = t L + l of x (L = ileave) goes to columns c L + l of y's row
 * t, the layer index innermost as the reference flattens [C, L]. */
int ref_pixel_norm(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), L = p_int(a, "ileave", 1);
    const float eps = p_f32(a, "eps", 1e-6f), mult = p_f32(a, "mult", 1.0f);
    static const char* const kForms[] = { "rsqrt", "div" };
    static const char* const kActs[] = { "none", "silu" };
    const int form = p_enum(a, "form", kForms, 2, 0), act = p_enum(a, "act", kActs, 2, 0);
    if (form < 0 || act < 0 || n <= 0 || L <= 0) return RAD_E_INVAL;
    const int64_t rows = rows_of(Y, n * L) * L;
    if (rows <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;
    const uint32_t dt = Y->dtype;
    const int64_t xs = laststride(X), ys = laststride(Y);
    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, n), yb = rowoff(Y, r / L, n * L), l = r % L;
        float ss = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            const float v = ldt(X, xb + i * xs);
            ss += round_dt(dt, v * v);
        }
        const float v = round_dt(dt, round_dt(dt, ss / (float)n) + eps);
        const float s = form == 0 ? round_dt(dt, 1.0f / std::sqrt(v)) : round_dt(dt, std::sqrt(v));
        for (int64_t i = 0; i < n; ++i) {
            const float x = ldt(X, xb + i * xs);
            float z = round_dt(dt, form == 0 ? x * s : x / s);
            if (act == 1) z = round_dt(dt, z / (1.0f + std::exp(-z)));
            if (mult != 1.0f) z = round_dt(dt, z * mult);
            stt(Y, yb + (i * L + l) * ys, z);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ d2s3d / s2d3d */
/* LTX-2's video VAE resampling over voxel rows (row (t, i, j) of a T x H x W volume, channels
 * innermost); `geom` is x's volume, i32 on the device. The packed channel of (c, a, b, d) is
 * ((c st + a) sh + b) sw + d, a the frame offset, b the row's, d the column's.
 *
 * d2s3d, the upsampler's depth to space: y[(t st + a - drop, i sh + b, j sw + d), c] = x[(t, i,
 * j), packed(c, a, b, d)], the first `drop` frames dropped (default st - 1, the causal mapping;
 * T st - drop out). A copy. */
int ref_d2s3d(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *Y = &a->t[2];
    const int64_t C = p_int(a, "C", 0), st = p_int(a, "st", 1), sh = p_int(a, "sh", 1);
    const int64_t sw = p_int(a, "sw", 1), drop = p_int(a, "drop", st - 1);
    if (C <= 0 || st <= 0 || sh <= 0 || sw <= 0 || drop < 0 || drop >= st || numel(G) < 3)
        return RAD_E_SHAPE;
    const int64_t T = ldi(G, 0), H = ldi(G, 1), W = ldi(G, 2), P = st * sh * sw;
    const int64_t To = T * st - drop, Ho = H * sh, Wo = W * sw;
    if (T <= 0 || H <= 0 || W <= 0 || rows_of(X, C * P) < T * H * W ||
        rows_of(Y, C) < To * Ho * Wo)
        return RAD_E_SHAPE;
    const int64_t xs = laststride(X), ys = laststride(Y);
    REF_PAR_FOR(To * Ho * Wo * C)
    for (int64_t r = 0; r < To * Ho * Wo; ++r) {
        const int64_t tt = r / (Ho * Wo) + drop, ii = (r / Wo) % Ho, jj = r % Wo;
        const int64_t t = tt / st, aa = tt % st, i = ii / sh, b = ii % sh, j = jj / sw, d = jj % sw;
        const int64_t xb = rowoff(X, (t * H + i) * W + j, C * P), yb = rowoff(Y, r, C);
        for (int64_t c = 0; c < C; ++c)
            stt(Y, yb + c * ys, ldt(X, xb + (((c * st + aa) * sh + b) * sw + d) * xs));
    }
    return RAD_OK;
}

/* s2d3d, the downsampler's space to depth: the volume with `tpre` frames prepended (frames 0 ..
 * tpre - 1 again, the reference's hs[:, :, :st-1]) cut into st x sh x sw boxes, each box's values
 * a packed row. `mode` copy writes the packed row (C st sh sw channels); `mean` averages each run
 * of `group` consecutive packed channels (f32 sum, divided, rounded once; the residual's mean).
 * With `add` 1 the result is added into y, y = rd(y + v). */
int ref_s2d3d(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *Y = &a->t[2];
    const int64_t C = p_int(a, "C", 0), st = p_int(a, "st", 1), sh = p_int(a, "sh", 1);
    const int64_t sw = p_int(a, "sw", 1), tpre = p_int(a, "tpre", 0), grp = p_int(a, "group", 1);
    const bool add = p_int(a, "add", 0) != 0;
    static const char* const kModes[] = { "copy", "mean" };
    const int mode = p_enum(a, "mode", kModes, 2, 0);
    if (mode < 0) return RAD_E_INVAL;
    if (C <= 0 || st <= 0 || sh <= 0 || sw <= 0 || tpre < 0 || grp <= 0 || numel(G) < 3)
        return RAD_E_SHAPE;
    const int64_t T = ldi(G, 0), H = ldi(G, 1), W = ldi(G, 2), P = C * st * sh * sw;
    const int64_t Tv = T + tpre;
    if (T <= 0 || Tv % st || H % sh || W % sw || (mode == 1 && P % grp)) return RAD_E_SHAPE;
    const int64_t To = Tv / st, Ho = H / sh, Wo = W / sw, Co = mode == 1 ? P / grp : P;
    if (rows_of(X, C) < T * H * W || rows_of(Y, Co) < To * Ho * Wo) return RAD_E_SHAPE;
    const int64_t xs = laststride(X), ys = laststride(Y);
    const uint32_t dt = Y->dtype;
    REF_PAR_FOR(To * Ho * Wo * P)
    for (int64_t r = 0; r < To * Ho * Wo; ++r) {
        const int64_t t = r / (Ho * Wo), i = (r / Wo) % Ho, j = r % Wo;
        const int64_t yb = rowoff(Y, r, Co);
        auto val = [&](int64_t q) {   /* packed channel q of this box */
            const int64_t d = q % sw, b = (q / sw) % sh, aa = (q / (sw * sh)) % st;
            const int64_t c = q / (sw * sh * st);
            const int64_t v = t * st + aa, tr = v < tpre ? v : v - tpre;
            return ldt(X, rowoff(X, (tr * H + i * sh + b) * W + j * sw + d, C) + c * xs);
        };
        for (int64_t o = 0; o < Co; ++o) {
            float v;
            if (mode == 0) {
                v = val(o);
            } else {
                float s = 0.0f;
                for (int64_t q = o * grp; q < (o + 1) * grp; ++q) s += val(q);
                v = round_dt(dt, s / (float)grp);
            }
            const int64_t oy = yb + o * ys;
            stt(Y, oy, add ? round_dt(dt, ldt(Y, oy)) + round_dt(dt, v) : v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ pix_patch */
/* LTX-2's patchifier over dense 2-D operands whose shapes give the sizes. A patch of
 * p x p pixels of C channels is one row of C p p values, channel c's offset (ph, pw) at
 * c p p + pw p + ph with `order` wh (LTX-2's: the column offset outer) or c p p + ph p + pw with
 * hw.
 *
 * pix_patch: an image x [H, W C] to patch rows y [H / p, (W / p) C p p]. A copy. */
static inline int64_t patch_col(int64_t c, int64_t ph, int64_t pw, int64_t p, bool wh) {
    return c * p * p + (wh ? pw * p + ph : ph * p + pw);
}

int ref_pix_patch(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t p = p_int(a, "p", 0), C = p_int(a, "C", 0);
    const bool wh = std::strcmp(p_str(a, "order", "wh"), "hw") != 0;
    if (p <= 0 || C <= 0 || X->rank != 2 || Y->rank != 2 || X->shape[1] % C) return RAD_E_SHAPE;
    const int64_t H = X->shape[0], W = X->shape[1] / C;
    if (H <= 0 || W <= 0 || H % p || W % p || Y->shape[0] != H / p ||
        Y->shape[1] != (W / p) * C * p * p)
        return RAD_E_SHAPE;
    const int64_t xr = X->stride[0], yr = Y->stride[0];
    REF_PAR_FOR(H * W * C)
    for (int64_t y = 0; y < H; ++y)
        for (int64_t x = 0; x < W; ++x)
            for (int64_t c = 0; c < C; ++c)
                stt(Y, (y / p) * yr + (x / p) * C * p * p + patch_col(c, y % p, x % p, p, wh),
                    ldt(X, y * xr + x * C + c));
    return RAD_OK;
}

/* ------------------------------------------------------------------ vol_index */
/* A volume's voxel coordinates, the rotary positions of attention over it: geom = (T, H, W), i32
 * on the device; y i32 [3, n] component-major (at its own row pitch), voxel m = (t H + i) W + j's
 * (t, i, j) at column m, for m < min(n, T H W). */
int ref_vol_index(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *G = &a->t[0], *Y = &a->t[1];
    if (numel(G) < 3 || Y->rank != 2 || Y->shape[0] != 3) return RAD_E_SHAPE;
    const int64_t T = ldi(G, 0), H = ldi(G, 1), W = ldi(G, 2), n = Y->shape[1];
    if (T <= 0 || H <= 0 || W <= 0 || n <= 0) return RAD_E_SHAPE;
    const int64_t m1 = std::min(n, T * H * W);
    for (int64_t m = 0; m < m1; ++m) {
        const int64_t v[3] = { m / (H * W), (m / W) % H, m % W };
        for (int64_t c = 0; c < 3; ++c)
            st_int(Y->dtype, Y->data, c * Y->stride[0] + m * Y->stride[1], v[c]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ tile_grid / tile_crop / tile_stitch */
/* A decoder run in tiles, as diffusers cuts and blends them: LTX-2's VAE (`tiled_decode`, rule
 * `stride`) and LTX-2.5's diffusion decoder (`_tile_intervals`, rule `intervals`). Along an axis
 * of L cells a tile starts at every multiple of `stride` below L; under `intervals` one tile covers
 * an axis of `tile` or fewer cells, and while more than one start is left and the last leaves fewer
 * than `min` cells, that start is dropped. A tile but the last is min(tile, L - start) long; the
 * last runs to L. Tiles are raster-ordered over (time band, row band, column band); without
 * `tile_t` there is one time band of every frame.
 *
 * tile_grid: the boxes of the grid geom = (T, H, W), i32 on the device, into y i32 [M, 8], row k
 * (t, h, w, t0, y0, x0, k, count); the last time band is `tail` cells longer than its cut (frames
 * past T a decoder carries, which its crop takes and its output drops). Rows past the count are
 * zero, tiles past M not written. A row's (t, h, w) is a decoder's volume operand and its first six
 * entries tile_crop's box. */
struct TileCut {
    int64_t tile = 1, stride = 1, min = 1;
    bool    intervals = false;
    int64_t count(int64_t L) const {
        if (intervals && L <= tile) return 1;
        int64_t n = (L + stride - 1) / stride;
        while (intervals && n > 1 && L - (n - 1) * stride < min) --n;
        return n;
    }
    int64_t len(int64_t L, int64_t k) const {
        return k == count(L) - 1 ? L - k * stride : std::min(tile, L - k * stride);
    }
};

static bool tile_cuts(const RadArgs* a, TileCut* s, TileCut* t) {
    static const char* const kRules[] = { "stride", "intervals" };
    const int rule = p_enum(a, "rule", kRules, 2, 0);
    s->tile = p_int(a, "tile", 0);
    s->stride = p_int(a, "stride", 0);
    s->min = p_int(a, "min", 1);
    s->intervals = rule == 1;
    t->tile = p_int(a, "tile_t", 1 << 30);
    t->stride = p_int(a, "stride_t", 1 << 30);
    t->min = p_int(a, "min_t", 1);
    t->intervals = rule == 1;
    return rule >= 0 && s->tile > 0 && s->stride > 0 && s->stride <= s->tile && s->min > 0 &&
           t->tile > 0 && t->stride > 0 && t->stride <= t->tile && t->min > 0;
}

int ref_tile_grid(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *G = &a->t[0], *Y = &a->t[1];
    TileCut cs, ct;
    const int64_t n = numel(Y) / 8, tail = p_int(a, "tail", 0);
    if (!tile_cuts(a, &cs, &ct) || tail < 0 || n <= 0 || numel(G) < 3) return RAD_E_SHAPE;
    const int64_t T = ldi(G, 0), H = ldi(G, 1), W = ldi(G, 2);
    if (T <= 0 || H <= 0 || W <= 0) return RAD_E_SHAPE;
    const int64_t nt = ct.count(T), ni = cs.count(H), nj = cs.count(W), count = nt * ni * nj;
    for (int64_t k = 0; k < n; ++k) {
        int64_t row[8] = {};
        if (k < count) {
            const int64_t kt = k / (ni * nj), i = (k / nj) % ni, j = k % nj;
            const int64_t v[8] = { ct.len(T, kt) + (kt == nt - 1 ? tail : 0), cs.len(H, i),
                                   cs.len(W, j), kt * ct.stride, i * cs.stride, j * cs.stride, k,
                                   count };
            std::memcpy(row, v, sizeof row);
        }
        for (int64_t e = 0; e < 8; ++e) st_int(Y->dtype, Y->data, offlin(Y, k * 8 + e), row[e]);
    }
    return RAD_OK;
}

/* tile_crop: one tile's voxels out of a volume. x [>= T H W, C] is the volume (row (t, yy, xx)
 * row-major), geom its (T, H, W); box = (t, h, w, t0, y0, x0), i32 on the device (a tile_grid
 * row); y [t h w, C], y[(a, i, j)] = x[(t0 + a, y0 + i, x0 + j)]. A copy. */
int ref_tile_crop(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *B = &a->t[2], *Y = &a->t[3];
    const int64_t C = p_int(a, "C", 0);
    if (C <= 0 || numel(G) < 3 || numel(B) < 6) return RAD_E_SHAPE;
    const int64_t T = ldi(G, 0), H = ldi(G, 1), W = ldi(G, 2);
    const int64_t t = ldi(B, 0), h = ldi(B, 1), w = ldi(B, 2);
    const int64_t t0 = ldi(B, 3), y0 = ldi(B, 4), x0 = ldi(B, 5);
    if (t <= 0 || h <= 0 || w <= 0 || t0 < 0 || y0 < 0 || x0 < 0 || t0 + t > T || y0 + h > H ||
        x0 + w > W || rows_of(X, C) < T * H * W || rows_of(Y, C) < t * h * w)
        return RAD_E_SHAPE;
    const int64_t xs = laststride(X), ys = laststride(Y);
    REF_PAR_FOR(t * h * w * C)
    for (int64_t r = 0; r < t * h * w; ++r) {
        const int64_t f = r / (h * w), i = (r / w) % h, j = r % w;
        const int64_t xb = rowoff(X, ((t0 + f) * H + y0 + i) * W + x0 + j, C), yb = rowoff(Y, r, C);
        for (int64_t c = 0; c < C; ++c) stt(Y, yb + c * ys, ldt(X, xb + c * xs));
    }
    return RAD_OK;
}

/* tile_stitch: decoded tiles to RGB bytes. geom = (T, H, W) is the grid the tiles cut (tail cells
 * excluded); a cell is `scale` patches of p x p pixels a side and `scale_t` frames, the first cell
 * of the video one frame (the causal mapping: T cells are scale_t T - scale_t + 1 frames). x is
 * every tile's decoder output in tile order, back to back: tile (kt, i, j) with cells [t0, t1) in
 * time is F = scale_t (t1 - t0), less scale_t - 1 when t0 is 0, frames of (h scale) x (w scale)
 * rows of patches (pix_patch's columns, C p p a row). `first` is the first frame y holds, i32 [1].
 *
 * A pixel (Y, X) of frame n is time band kt's whose kept frames hold n (a band but the last keeps
 * stride_t scale_t frames, the first scale_t - 1 fewer), then tile (i, j) = (Y / s, X / s) of that
 * band, s = stride scale p, the last band of an axis keeping the rest. With R a tile's decoded
 * value, every product and sum rounded to x's dtype (rd) and mix(a, b, k, e) =
 * rd(rd(a f32(1 - k / e)) + rd(b f32(k / e))) (the quotients in f64), a band's picture S is
 * diffusers' in-place blend_v then blend_h of its tiles:
 *
 *   ev = min(rows of tile (i-1, j), rows of tile (i, j), (tile - stride) scale p), eh likewise
 *   up(c)   = R(i-1, c)[rows(i-1) - ev + ly, .]          the tile above's bottom margin
 *   A       = mix(up(j-1)[., cols(j-1) - eh + lx], up(j)[., lx], lx, eh) when j > 0, lx < eh,
 *             else up(j)[., lx]                           (it was blended with its own left)
 *   V       = mix(A, R(i, j)[ly, lx], ly, ev) when i > 0 and ly < ev, else R(i, j)[ly, lx]
 *   L       = mix(up(j-1)[., cols(j-1) - eh + lx], R(i, j-1)[ly, cols(j-1) - eh + lx], ly, ev)
 *             when i > 0 and ly < ev, else R(i, j-1)[ly, cols(j-1) - eh + lx]
 *   S       = mix(L, V, lx, eh) when j > 0 and lx < eh, else V
 *
 * and across time blend_t: frame f of band kt, f < et = min(F of band kt-1, F of band kt,
 * (tile_t - stride_t) scale_t), is mix(S_{kt-1}[F_{kt-1} - et + f], S_kt[f], f, et). The byte is
 * to_rgba8's, round(255 clamp(rd(rd(clamp(v, -1, 1) / 2) + 0.5), 0, 1)). y is u8
 * [n (H scale p), (W scale p) C], frames first .. first + n - 1. A tile above, to the left or
 * before is never short enough for its own blends to reach the margin read here (a tile not last
 * on its axis spans more than `stride`), so these are the reference's values exactly. */
struct TileStore {
    TileCut cs, ct;
    int64_t T, H, W, sc, st, nt, ni, nj, hsum, wsum;
    TileStore(const TileCut& s, const TileCut& t, int64_t T_, int64_t H_, int64_t W_, int64_t sc_,
              int64_t st_)
        : cs(s), ct(t), T(T_), H(H_), W(W_), sc(sc_), st(st_), nt(t.count(T_)), ni(s.count(H_)),
          nj(s.count(W_)), hsum(0), wsum(0) {
        for (int64_t i = 0; i < ni; ++i) hsum += cs.len(H, i);
        for (int64_t j = 0; j < nj; ++j) wsum += cs.len(W, j);
    }
    int64_t frames(int64_t kt) const {
        return st * ct.len(T, kt) - (kt == 0 ? st - 1 : 0);
    }
    /* Tile (kt, i, j)'s first patch row. */
    int64_t offset(int64_t kt, int64_t i, int64_t j) const {
        int64_t before = 0;
        for (int64_t q = 0; q < kt; ++q) before += frames(q);
        int64_t left = 0, above = 0;
        for (int64_t q = 0; q < j; ++q) left += cs.len(W, q);
        for (int64_t q = 0; q < i; ++q) above += cs.len(H, q);
        return sc * sc * (before * hsum * wsum + frames(kt) * (above * wsum + cs.len(H, i) * left));
    }
    int64_t rows() const { return offset(nt, 0, 0); }
};

int ref_tile_stitch(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *Fst = &a->t[2], *Y = &a->t[3];
    TileCut cs, ct;
    const int64_t sc = p_int(a, "scale", 0), st = p_int(a, "scale_t", 1);
    const int64_t p = p_int(a, "p", 0), C = p_int(a, "C", 0);
    const bool wh = std::strcmp(p_str(a, "order", "wh"), "hw") != 0;
    if (!tile_cuts(a, &cs, &ct) || sc <= 0 || st <= 0 || p <= 0 || C <= 0 || numel(G) < 3 ||
        numel(Fst) < 1 || Y->dtype != RAD_U8 || Y->rank != 2)
        return RAD_E_SHAPE;
    const int64_t T = ldi(G, 0), H = ldi(G, 1), W = ldi(G, 2), f0 = ldi(Fst, 0);
    const int64_t Hpx = H * sc * p, Wpx = W * sc * p, P = C * p * p;
    if (T <= 0 || H <= 0 || W <= 0 || Y->shape[1] != Wpx * C || Y->shape[0] % Hpx) return RAD_E_SHAPE;
    const int64_t nf = Y->shape[0] / Hpx;
    const TileStore ts(cs, ct, T, H, W, sc, st);
    if (nf <= 0 || f0 < 0 || f0 + nf > st * T - (st - 1) || rows_of(X, P) < ts.rows())
        return RAD_E_SHAPE;
    const int64_t s = cs.stride * sc * p, E = (cs.tile - cs.stride) * sc * p;
    const int64_t Et = (ct.tile - ct.stride) * st, xs = laststride(X);
    const uint32_t dt = X->dtype;
    auto R = [&](int64_t kt, int64_t i, int64_t j, int64_t f, int64_t yy, int64_t xx, int64_t c) {
        const int64_t wp = cs.len(W, j) * sc, hp = cs.len(H, i) * sc;
        const int64_t row = ts.offset(kt, i, j) + (f * hp + yy / p) * wp + xx / p;
        return ldt(X, rowoff(X, row, P) + patch_col(c, yy % p, xx % p, p, wh) * xs);
    };
    auto mix = [&](float va, float vb, int64_t k, int64_t e) {
        const float wa = (float)(1.0 - (double)k / (double)e), wb = (float)((double)k / (double)e);
        return round_dt(dt, round_dt(dt, va * wa) + round_dt(dt, vb * wb));
    };
    /* Band kt's spatially stitched value at its frame f. */
    auto S = [&](int64_t kt, int64_t f, int64_t Yp, int64_t Xp, int64_t c) {
        const int64_t i = std::min(Yp / s, ts.ni - 1), ly = Yp - i * s;
        const int64_t j = std::min(Xp / s, ts.nj - 1), lx = Xp - j * s;
        const int64_t ha = i > 0 ? cs.len(H, i - 1) * sc * p : 0;
        const int64_t ev = i > 0 ? std::min(std::min(ha, cs.len(H, i) * sc * p), E) : 0;
        const int64_t wl = j > 0 ? cs.len(W, j - 1) * sc * p : 0;
        const int64_t eh = j > 0 ? std::min(std::min(wl, cs.len(W, j) * sc * p), E) : 0;
        const bool vb = i > 0 && ly < ev, hb = j > 0 && lx < eh;
        float V = R(kt, i, j, f, ly, lx, c);
        if (vb) {
            const int64_t ya = ha - ev + ly;
            float A = R(kt, i - 1, j, f, ya, lx, c);
            if (hb) A = mix(R(kt, i - 1, j - 1, f, ya, wl - eh + lx, c), A, lx, eh);
            V = mix(A, V, ly, ev);
        }
        if (hb) {
            float L = R(kt, i, j - 1, f, ly, wl - eh + lx, c);
            if (vb) L = mix(R(kt, i - 1, j - 1, f, ha - ev + ly, wl - eh + lx, c), L, ly, ev);
            V = mix(L, V, lx, eh);
        }
        return V;
    };
    uint8_t* yb = (uint8_t*)Y->data;
    const int64_t yr = Y->stride[0], band = ct.stride * st;
    REF_PAR_FOR(nf * Hpx * Wpx * C)
    for (int64_t r = 0; r < nf * Hpx; ++r) {
        const int64_t n = f0 + r / Hpx, Yp = r % Hpx;
        const int64_t kt = std::min((n + st - 1) / band, ts.nt - 1);
        const int64_t f = n - (kt == 0 ? 0 : kt * band - (st - 1));
        const int64_t et = kt > 0 ? std::min(std::min(ts.frames(kt - 1), ts.frames(kt)), Et) : 0;
        for (int64_t Xp = 0; Xp < Wpx; ++Xp)
            for (int64_t c = 0; c < C; ++c) {
                float v = S(kt, f, Yp, Xp, c);
                if (kt > 0 && f < et) v = mix(S(kt - 1, ts.frames(kt - 1) - et + f, Yp, Xp, c), v, f, et);
                const float cl = std::min(std::max(v, -1.0f), 1.0f);
                float u = round_dt(dt, round_dt(dt, cl / 2.0f) + 0.5f);
                u = std::min(std::max(u, 0.0f), 1.0f);
                yb[r * yr + Xp * C + c] = (uint8_t)std::nearbyint(u * 255.0f);
            }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ stft_logmel */
/* LTX-2's bandwidth extender's input: each channel's causal log-mel spectrogram, as its MelSTFT
 * computes it in the vocoder's dtype. x [S, ch] (time rows, a channel a column), `basis`
 * [2F, 1, nfft] (the windowed DFT, real rows then imaginary), `mel` [n_mel, F]; frame t of channel
 * h reads samples t hop + j - (nfft - hop), zero before the start:
 *
 *   s_k  = rd(sum_j basis[k, j] x[.])      mag_k = rd(sqrt(rd(rd(re_k^2) + rd(im_k^2))))
 *   y[t, h n_mel + m] = rd(log(max(rd(sum_k mel[m, k] mag_k), rd(1e-5))))
 *
 * sums f32 in ascending order, rd y's dtype. y is [S / hop, ch n_mel]. */
int ref_stft_logmel(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *B = &a->t[1], *Mb = &a->t[2], *Y = &a->t[3];
    const int64_t hop = p_int(a, "hop", 0), nfft = p_int(a, "nfft", 0), nm = p_int(a, "n_mel", 0);
    const int64_t ch = p_int(a, "ch", 0);
    if (hop <= 0 || nfft < hop || nm <= 0 || ch <= 0) return RAD_E_SHAPE;
    const int64_t F = nfft / 2 + 1, S = rows_of(X, ch), frames = rows_of(Y, ch * nm);
    if (S <= 0 || frames <= 0 || frames * hop > S || numel(B) < 2 * F * nfft || numel(Mb) < nm * F)
        return RAD_E_SHAPE;
    const uint32_t dt = Y->dtype;
    const int64_t pad = nfft - hop, xs = laststride(X), ys = laststride(Y);
    const float floor_v = round_dt(dt, 1e-5f);
    REF_PAR_FOR(frames * ch * 2 * F * nfft)
    for (int64_t t = 0; t < frames; ++t)
        for (int64_t h = 0; h < ch; ++h) {
            std::vector<float> mag((size_t)F);
            for (int64_t k = 0; k < F; ++k) {
                float re = 0.0f, im = 0.0f;
                for (int64_t j = 0; j < nfft; ++j) {
                    const int64_t s = t * hop + j - pad;
                    if (s < 0) continue;
                    const float v = ldt(X, rowoff(X, s, ch) + h * xs);
                    re += ldt(B, k * nfft + j) * v;
                    im += ldt(B, (F + k) * nfft + j) * v;
                }
                re = round_dt(dt, re);
                im = round_dt(dt, im);
                mag[(size_t)k] = round_dt(dt, std::sqrt(round_dt(dt, round_dt(dt, re * re) +
                                                                     round_dt(dt, im * im))));
            }
            const int64_t yb = rowoff(Y, t, ch * nm);
            for (int64_t m = 0; m < nm; ++m) {
                float s = 0.0f;
                for (int64_t k = 0; k < F; ++k) s += ldt(Mb, m * F + k) * mag[(size_t)k];
                const float mel = std::max(round_dt(dt, s), floor_v);
                stt(Y, yb + (h * nm + m) * ys, std::log(mel));
            }
        }
    return RAD_OK;
}

/* ------------------------------------------------------------------ bwe_up */
/* LTX-2's bandwidth extender's output: the low-rate waveform x [S, ch] raised `ratio` times by its
 * Hann-windowed sinc resampler (UpSample1d, width ceil(6 / 0.99), its taps in x's dtype), added to
 * the extender's residual `res` [S ratio, ch], clamped to [-1, 1]:
 *
 *   f[n] = rd(sinc(u) cos^2(clamp(u, -6, 6) pi / 12) 0.99 / ratio), u = f32(n / ratio - w) 0.99
 *   up[o] = rd(ratio rd(sum f[k] x[clamp(i - w, 0, S - 1)])), i ratio + k = o + 2 w ratio
 *   y[o] = clamp(rd(res[o] + up[o]), -1, 1)
 *
 * the filter's arithmetic f32 as torch builds it, rd x's dtype; y is f32 or x's dtype. */
int ref_bwe_up(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Rz = &a->t[1], *Y = &a->t[2];
    const int64_t r = p_int(a, "ratio", 0), ch = p_int(a, "ch", 0);
    if (r <= 0 || ch <= 0) return RAD_E_SHAPE;
    const int64_t S = rows_of(X, ch), O = rows_of(Y, ch);
    if (S <= 0 || O <= 0 || O > S * r || rows_of(Rz, ch) < O) return RAD_E_SHAPE;
    const uint32_t dt = X->dtype;
    const int64_t w = (int64_t)std::ceil(6.0 / 0.99), K = 2 * w * r + 1, off = 2 * w * r;
    std::vector<float> f((size_t)K);
    for (int64_t n = 0; n < K; ++n) {
        const float u = ((float)n / (float)r - (float)w) * 0.99f;
        const float uc = std::min(std::max(u, -6.0f), 6.0f);
        const float cw = std::cos(uc * (float)M_PI / 6.0f / 2.0f);
        const float pu = (float)M_PI * u;
        const float sinc = u == 0.0f ? 1.0f : std::sin(pu) / pu;
        f[(size_t)n] = round_dt(dt, sinc * (cw * cw) * 0.99f / (float)r);
    }
    const int64_t xs = laststride(X), rs = laststride(Rz), ys = laststride(Y);
    REF_PAR_FOR(O * ch * K)
    for (int64_t o = 0; o < O; ++o)
        for (int64_t h = 0; h < ch; ++h) {
            const int64_t u = o + off;
            float acc = 0.0f;
            for (int64_t k = u % r; k < K; k += r) {
                const int64_t i = (u - k) / r;   /* the padded input's index */
                const int64_t s = std::min(std::max<int64_t>(i - w, 0), S - 1);
                acc += f[(size_t)k] * ldt(X, rowoff(X, s, ch) + h * xs);
            }
            const float up = round_dt(dt, (float)r * round_dt(dt, acc));
            const float v = round_dt(dt, ldt(Rz, rowoff(Rz, o, ch) + h * rs) + up);
            stt(Y, rowoff(Y, o, ch) + h * ys, std::min(std::max(v, -1.0f), 1.0f));
        }
    return RAD_OK;
}
