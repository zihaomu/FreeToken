/* avx_ltx.cpp -- LTX-2's ops (docs/OPS.md, "LTX-2"): its rotary over fractional coordinates, the
 * per-head attention gate, the guided sampler step, its VAE's norm, resampling and patches, and its
 * vocoder's spectrogram and resampler. libref's arithmetic, element by element
 * (libref/ref_ltx.cpp states the roundings): LTX-2 is a GPU model, and these run on the CPU only
 * where a whole model does, so they are written for agreement rather than speed.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace avx;

namespace {

/* A product the compiler may not fuse into the add after it (libavx builds with
 * -ffp-contract=fast; libref rounds the product first). */
inline float keep(float x) { __asm__("" : "+x"(x)); return x; }

double linspace01(int64_t i, int64_t steps) {
    if (steps <= 1) return 0.0;
    const double step = 1.0 / (double)(steps - 1);
    return i < steps / 2 ? step * (double)i : 1.0 - step * (double)(steps - 1 - i);
}

int parse_floats(const char* s, std::vector<float>* out) {
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

inline int64_t at_i(const RadTensor* t, int64_t i) {
    return (int64_t)ld_elem(t->data, t->dtype, i * laststride(t));
}

float pos_at(const RadTensor* p, int64_t d, int64_t r) {
    const int64_t off = p->rank >= 2 ? d * p->stride[0] + r * p->stride[1] : r * laststride(p);
    return ldt(p, off);
}

}  /* namespace */

/* ------------------------------------------------------------------ rope_split */
AVX_KERNEL(rope_split) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *P = &a->t[1];
    const int64_t nh = p_int(a, "n_head", 0), hd = p_int(a, "head_dim", 0);
    const int64_t dim = p_int(a, "dim", 0), nd = p_int(a, "n_dims", 0), h0 = p_int(a, "head0", 0);
    const double theta = (double)p_f32(a, "theta", 10000.0f);
    std::vector<float> mx;
    if (nh <= 0 || hd <= 0 || (hd & 1) || nd <= 0 || dim < (h0 + nh) * hd ||
        parse_floats(p_str(a, "max", ""), &mx) != (int)nd)
        return RAD_E_SHAPE;
    const int64_t P2 = dim / 2, steps = P2 / nd, pad = P2 - steps * nd, half = hd / 2;
    const int64_t rows = rows_of(X, nh * hd);
    if (rows <= 0 || P->rank < 1) return RAD_E_SHAPE;
    std::vector<float> freq((size_t)steps);
    for (int64_t k = 0; k < steps; ++k)
        freq[(size_t)k] = (float)(std::pow(theta, linspace01(k, steps)) * M_PI / 2.0);
    const int64_t xs = laststride(X);
    AVX_PARALLEL_FOR_IF(rows * nh * hd >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, nh * hd);
        for (int64_t h = 0; h < nh; ++h)
            for (int64_t i = 0; i < half; ++i) {
                const int64_t p = (h0 + h) * half + i;
                if (p < pad) continue;
                const int64_t j = p - pad, d = j % nd, k = j / nd;
                const float g = pos_at(P, d, r) / mx[(size_t)d];
                const float ang = (2.0f * g - 1.0f) * freq[(size_t)k];
                const float c = (float)std::cos((double)ang), s = (float)std::sin((double)ang);
                const int64_t o1 = xb + (h * hd + i) * xs, o2 = xb + (h * hd + half + i) * xs;
                const float x1 = ldt(X, o1), x2 = ldt(X, o2);
                st_elem(X->data, X->dtype, o1, keep(x1 * c) - keep(x2 * s));
                st_elem(X->data, X->dtype, o2, keep(x2 * c) + keep(x1 * s));
            }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ head_gate */
AVX_KERNEL(head_gate) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *G = &a->t[0], *X = &a->t[1];
    const int64_t nh = p_int(a, "n_head", 0), hd = p_int(a, "head_dim", 0);
    if (nh <= 0 || hd <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(X, nh * hd);
    if (rows <= 0 || rows_of(G, nh) < rows) return RAD_E_SHAPE;
    const int64_t xs = laststride(X), gs = laststride(G);
    AVX_PARALLEL_FOR_IF(rows * nh * hd >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, nh * hd), gb = rowoff(G, r, nh);
        for (int64_t h = 0; h < nh; ++h) {
            const float lg = ldt(G, gb + h * gs);
            const float sg = narrow1(G->dtype, 1.0f / (1.0f + std::exp(-lg)));
            const float gate = narrow1(X->dtype, 2.0f * sg);
            for (int64_t i = 0; i < hd; ++i) {
                const int64_t o = xb + (h * hd + i) * xs;
                st_elem(X->data, X->dtype, o, ldt(X, o) * gate);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ltx_guide */
AVX_KERNEL(ltx_guide) {
    if (a->n_t < 8) return RAD_E_INVAL;
    const RadTensor *Vc = &a->t[0], *Vu = rad_arg_in(a, 1), *Vs = rad_arg_in(a, 2);
    const RadTensor *Vm = rad_arg_in(a, 3), *Sc = &a->t[4], *Sh = &a->t[5], *X = &a->t[7];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || numel(Sc) < 4 || numel(Sh) < 2) return RAD_E_SHAPE;
    const int64_t rows = rows_of(X, n);
    if (rows <= 0 || rows_of(Vc, n) < rows || (Vu && rows_of(Vu, n) < rows) ||
        (Vs && rows_of(Vs, n) < rows) || (Vm && rows_of(Vm, n) < rows))
        return RAD_E_SHAPE;
    const int64_t ss = laststride(Sc);
    const float gs = ldt(Sc, 0), stg = ldt(Sc, ss), ms = ldt(Sc, 2 * ss), rr = ldt(Sc, 3 * ss);
    const float sig = ldt(Sh, 0), sig_n = ldt(Sh, laststride(Sh));
    const bool use_u = Vu && gs != 1.0f, use_s = Vs && stg != 0.0f, use_m = Vm && ms != 1.0f;
    const float cu = gs - 1.0f, cm = ms - 1.0f;
    const int64_t N = rows * n;
    std::vector<float> x0c((size_t)N), g((size_t)N);
    auto x0 = [&](const RadTensor* V, int64_t r, int64_t c, float x) {
        return x - keep(ldt(V, rowoff(V, r, n) + c * laststride(V)) * sig);
    };
    AVX_PARALLEL_FOR_IF(N >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < n; ++c) {
            const float x = ldt(X, rowoff(X, r, n) + c * laststride(X));
            const float xc = x0(Vc, r, c, x);
            float v = xc;
            if (use_u) v = v + keep(cu * (xc - x0(Vu, r, c, x)));
            if (use_s) v = v + keep(stg * (xc - x0(Vs, r, c, x)));
            if (use_m) v = v + keep(cm * (xc - x0(Vm, r, c, x)));
            x0c[(size_t)(r * n + c)] = xc;
            g[(size_t)(r * n + c)] = v;
        }
    float fac = 1.0f, kept = 1.0f;
    if (rr > 0.0f) {
        auto stdev = [&](const std::vector<float>& v) {
            double m = 0.0;
            for (float e : v) m += e;
            m /= (double)N;
            double sq = 0.0;
            for (float e : v) sq += ((double)e - m) * ((double)e - m);
            return (float)std::sqrt(sq / (double)(N > 1 ? N - 1 : 1));
        };
        fac = stdev(x0c) / stdev(g);
        kept = (float)(1.0 - (double)rr);
    }
    const float dt = sig_n - sig;
    AVX_PARALLEL_FOR_IF(N >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < n; ++c) {
            float v = g[(size_t)(r * n + c)];
            if (rr > 0.0f) v = keep(rr * keep(v * fac)) + keep(kept * v);
            const int64_t o = rowoff(X, r, n) + c * laststride(X);
            const float x = ldt(X, o);
            const float vel = (x - v) / sig;
            st_elem(X->data, X->dtype, o, x + keep(dt * vel));
        }
    return RAD_OK;
}

/* ------------------------------------------------------------------ pixel_norm */
AVX_KERNEL(pixel_norm) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), L = p_int(a, "ileave", 1);
    const float eps = p_f32(a, "eps", 1e-6f), mult = p_f32(a, "mult", 1.0f);
    const char* fs = p_str(a, "form", "rsqrt");
    const char* as = p_str(a, "act", "none");
    const int form = !std::strcmp(fs, "rsqrt") ? 0 : !std::strcmp(fs, "div") ? 1 : -1;
    const int act = !std::strcmp(as, "none") ? 0 : !std::strcmp(as, "silu") ? 1 : -1;
    if (form < 0 || act < 0 || n <= 0 || L <= 0) return RAD_E_INVAL;
    const int64_t rows = rows_of(Y, n * L) * L;
    if (rows <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;
    const uint32_t dt = Y->dtype;
    const int64_t xs = laststride(X), ys = laststride(Y);
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, n), yb = rowoff(Y, r / L, n * L), l = r % L;
        float ss = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            const float v = ldt(X, xb + i * xs);
            ss += narrow1(dt, v * v);
        }
        const float v = narrow1(dt, narrow1(dt, ss / (float)n) + eps);
        const float sc = form == 0 ? narrow1(dt, 1.0f / std::sqrt(v)) : narrow1(dt, std::sqrt(v));
        for (int64_t i = 0; i < n; ++i) {
            const float x = ldt(X, xb + i * xs);
            float z = narrow1(dt, form == 0 ? x * sc : x / sc);
            if (act == 1) z = narrow1(dt, z / (1.0f + std::exp(-z)));
            if (mult != 1.0f) z = narrow1(dt, z * mult);
            st_elem(Y->data, dt, yb + (i * L + l) * ys, z);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ d2s3d */
AVX_KERNEL(d2s3d) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *Y = &a->t[2];
    const int64_t C = p_int(a, "C", 0), st = p_int(a, "st", 1), sh = p_int(a, "sh", 1);
    const int64_t sw = p_int(a, "sw", 1), drop = p_int(a, "drop", st - 1);
    if (C <= 0 || st <= 0 || sh <= 0 || sw <= 0 || drop < 0 || drop >= st || numel(G) < 3)
        return RAD_E_SHAPE;
    const int64_t T = at_i(G, 0), H = at_i(G, 1), W = at_i(G, 2), P = st * sh * sw;
    const int64_t To = T * st - drop, Ho = H * sh, Wo = W * sw;
    if (T <= 0 || H <= 0 || W <= 0 || rows_of(X, C * P) < T * H * W ||
        rows_of(Y, C) < To * Ho * Wo)
        return RAD_E_SHAPE;
    const int64_t xs = laststride(X), ys = laststride(Y);
    AVX_PARALLEL_FOR_IF(To * Ho * Wo * C >= AVX_PAR_MIN)
    for (int64_t r = 0; r < To * Ho * Wo; ++r) {
        const int64_t tt = r / (Ho * Wo) + drop, ii = (r / Wo) % Ho, jj = r % Wo;
        const int64_t t = tt / st, aa = tt % st, i = ii / sh, b = ii % sh, j = jj / sw, d = jj % sw;
        const int64_t xb = rowoff(X, (t * H + i) * W + j, C * P), yb = rowoff(Y, r, C);
        for (int64_t c = 0; c < C; ++c)
            st_elem(Y->data, Y->dtype, yb + c * ys,
                    ldt(X, xb + (((c * st + aa) * sh + b) * sw + d) * xs));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ s2d3d */
AVX_KERNEL(s2d3d) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *Y = &a->t[2];
    const int64_t C = p_int(a, "C", 0), st = p_int(a, "st", 1), sh = p_int(a, "sh", 1);
    const int64_t sw = p_int(a, "sw", 1), tpre = p_int(a, "tpre", 0), grp = p_int(a, "group", 1);
    const bool add = p_int(a, "add", 0) != 0;
    const char* ms = p_str(a, "mode", "copy");
    const int mode = !std::strcmp(ms, "copy") ? 0 : !std::strcmp(ms, "mean") ? 1 : -1;
    if (mode < 0) return RAD_E_INVAL;
    if (C <= 0 || st <= 0 || sh <= 0 || sw <= 0 || tpre < 0 || grp <= 0 || numel(G) < 3)
        return RAD_E_SHAPE;
    const int64_t T = at_i(G, 0), H = at_i(G, 1), W = at_i(G, 2), P = C * st * sh * sw;
    const int64_t Tv = T + tpre;
    if (T <= 0 || Tv % st || H % sh || W % sw || (mode == 1 && P % grp)) return RAD_E_SHAPE;
    const int64_t To = Tv / st, Ho = H / sh, Wo = W / sw, Co = mode == 1 ? P / grp : P;
    if (rows_of(X, C) < T * H * W || rows_of(Y, Co) < To * Ho * Wo) return RAD_E_SHAPE;
    const int64_t xs = laststride(X), ys = laststride(Y);
    const uint32_t dt = Y->dtype;
    AVX_PARALLEL_FOR_IF(To * Ho * Wo * P >= AVX_PAR_MIN)
    for (int64_t r = 0; r < To * Ho * Wo; ++r) {
        const int64_t t = r / (Ho * Wo), i = (r / Wo) % Ho, j = r % Wo;
        const int64_t yb = rowoff(Y, r, Co);
        auto val = [&](int64_t q) {
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
                v = narrow1(dt, s / (float)grp);
            }
            const int64_t oy = yb + o * ys;
            st_elem(Y->data, dt, oy, add ? narrow1(dt, ldt(Y, oy)) + narrow1(dt, v) : v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ pix_patch */
static inline int64_t patch_col(int64_t c, int64_t ph, int64_t pw, int64_t p, bool wh) {
    return c * p * p + (wh ? pw * p + ph : ph * p + pw);
}

AVX_KERNEL(pix_patch) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t p = p_int(a, "p", 0), C = p_int(a, "C", 0);
    const bool wh = std::strcmp(p_str(a, "order", "wh"), "hw") != 0;
    if (p <= 0 || C <= 0 || X->rank != 2 || Y->rank != 2 || X->shape[1] % C) return RAD_E_SHAPE;
    const int64_t H = X->shape[0], W = X->shape[1] / C;
    if (H <= 0 || W <= 0 || H % p || W % p || Y->shape[0] != H / p ||
        Y->shape[1] != (W / p) * C * p * p)
        return RAD_E_SHAPE;
    const int64_t xr = X->stride[0], yr = Y->stride[0];
    AVX_PARALLEL_FOR_IF(H * W * C >= AVX_PAR_MIN)
    for (int64_t y = 0; y < H; ++y)
        for (int64_t x = 0; x < W; ++x)
            for (int64_t c = 0; c < C; ++c)
                st_elem(Y->data, Y->dtype,
                        (y / p) * yr + (x / p) * C * p * p + patch_col(c, y % p, x % p, p, wh),
                        ldt(X, y * xr + x * C + c));
    return RAD_OK;
}

/* ------------------------------------------------------------------ vol_index */
AVX_KERNEL(vol_index) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *G = &a->t[0], *Y = &a->t[1];
    if (numel(G) < 3 || Y->rank != 2 || Y->shape[0] != 3) return RAD_E_SHAPE;
    const int64_t T = at_i(G, 0), H = at_i(G, 1), W = at_i(G, 2), n = Y->shape[1];
    if (T <= 0 || H <= 0 || W <= 0 || n <= 0) return RAD_E_SHAPE;
    const int64_t m1 = std::min(n, T * H * W);
    for (int64_t m = 0; m < m1; ++m) {
        const int64_t v[3] = { m / (H * W), (m / W) % H, m % W };
        for (int64_t c = 0; c < 3; ++c)
            st_int_dt(Y->dtype, Y->data, c * Y->stride[0] + m * Y->stride[1], v[c]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ tile_grid / tile_crop / tile_stitch */
/* diffusers' tiled decodes: libref/ref_ltx.cpp states the cuts and the blends. */
namespace {
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

bool tile_cuts(const RadArgs* a, TileCut* s, TileCut* t) {
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

}  /* namespace */

AVX_KERNEL(tile_grid) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *G = &a->t[0], *Y = &a->t[1];
    TileCut cs, ct;
    const int64_t n = numel(Y) / 8, tail = p_int(a, "tail", 0);
    if (!tile_cuts(a, &cs, &ct) || tail < 0 || n <= 0 || numel(G) < 3) return RAD_E_SHAPE;
    const int64_t T = at_i(G, 0), H = at_i(G, 1), W = at_i(G, 2);
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
        for (int64_t e = 0; e < 8; ++e) st_int_dt(Y->dtype, Y->data, offlin(Y, k * 8 + e), row[e]);
    }
    return RAD_OK;
}

AVX_KERNEL(tile_crop) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *B = &a->t[2], *Y = &a->t[3];
    const int64_t C = p_int(a, "C", 0);
    if (C <= 0 || numel(G) < 3 || numel(B) < 6) return RAD_E_SHAPE;
    const int64_t T = at_i(G, 0), H = at_i(G, 1), W = at_i(G, 2);
    const int64_t t = at_i(B, 0), h = at_i(B, 1), w = at_i(B, 2);
    const int64_t t0 = at_i(B, 3), y0 = at_i(B, 4), x0 = at_i(B, 5);
    if (t <= 0 || h <= 0 || w <= 0 || t0 < 0 || y0 < 0 || x0 < 0 || t0 + t > T || y0 + h > H ||
        x0 + w > W || rows_of(X, C) < T * H * W || rows_of(Y, C) < t * h * w)
        return RAD_E_SHAPE;
    const int64_t xs = laststride(X), ys = laststride(Y);
    AVX_PARALLEL_FOR_IF(t * h * w * C >= AVX_PAR_MIN)
    for (int64_t r = 0; r < t * h * w; ++r) {
        const int64_t f = r / (h * w), i = (r / w) % h, j = r % w;
        const int64_t xb = rowoff(X, ((t0 + f) * H + y0 + i) * W + x0 + j, C), yb = rowoff(Y, r, C);
        for (int64_t c = 0; c < C; ++c) st_elem(Y->data, Y->dtype, yb + c * ys, ldt(X, xb + c * xs));
    }
    return RAD_OK;
}

namespace {
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
}  /* namespace */

AVX_KERNEL(tile_stitch) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *Fst = &a->t[2], *Y = &a->t[3];
    TileCut cs, ct;
    const int64_t sc = p_int(a, "scale", 0), st = p_int(a, "scale_t", 1);
    const int64_t p = p_int(a, "p", 0), C = p_int(a, "C", 0);
    const bool wh = std::strcmp(p_str(a, "order", "wh"), "hw") != 0;
    if (!tile_cuts(a, &cs, &ct) || sc <= 0 || st <= 0 || p <= 0 || C <= 0 || numel(G) < 3 ||
        numel(Fst) < 1 || Y->dtype != RAD_U8 || Y->rank != 2)
        return RAD_E_SHAPE;
    const int64_t T = at_i(G, 0), H = at_i(G, 1), W = at_i(G, 2), f0 = at_i(Fst, 0);
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
        return narrow1(dt, narrow1(dt, keep(va * wa)) + narrow1(dt, keep(vb * wb)));
    };
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
    AVX_PARALLEL_FOR_IF(nf * Hpx * Wpx * C >= AVX_PAR_MIN)
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
                float u = narrow1(dt, narrow1(dt, cl / 2.0f) + 0.5f);
                u = std::min(std::max(u, 0.0f), 1.0f);
                yb[r * yr + Xp * C + c] = (uint8_t)std::nearbyint(u * 255.0f);
            }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ stft_logmel */
AVX_KERNEL(stft_logmel) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *B = &a->t[1], *Mb = &a->t[2], *Y = &a->t[3];
    const int64_t hop = p_int(a, "hop", 0), nfft = p_int(a, "nfft", 0), nm = p_int(a, "n_mel", 0);
    const int64_t ch = p_int(a, "ch", 0);
    if (hop <= 0 || nfft < hop || nm <= 0 || ch <= 0) return RAD_E_SHAPE;
    const int64_t F = nfft / 2 + 1, S = rows_of(X, ch), frames = rows_of(Y, ch * nm);
    if (S <= 0 || frames <= 0 || frames * hop > S || numel(B) < 2 * F * nfft || numel(Mb) < nm * F)
        return RAD_E_SHAPE;
    const uint32_t dt = Y->dtype;
    const int64_t pad = nfft - hop, xs = laststride(X), ys = laststride(Y);
    const float floor_v = narrow1(dt, 1e-5f);
    AVX_PARALLEL_FOR_IF(frames * ch * F * nfft >= AVX_PAR_MIN)
    for (int64_t t = 0; t < frames; ++t)
        for (int64_t h = 0; h < ch; ++h) {
            std::vector<float> mag((size_t)F);
            for (int64_t k = 0; k < F; ++k) {
                float re = 0.0f, im = 0.0f;
                for (int64_t j = 0; j < nfft; ++j) {
                    const int64_t s = t * hop + j - pad;
                    if (s < 0) continue;
                    const float v = ldt(X, rowoff(X, s, ch) + h * xs);
                    re += keep(ldt(B, k * nfft + j) * v);
                    im += keep(ldt(B, (F + k) * nfft + j) * v);
                }
                re = narrow1(dt, re);
                im = narrow1(dt, im);
                mag[(size_t)k] = narrow1(dt, std::sqrt(narrow1(dt, narrow1(dt, re * re) +
                                                                   narrow1(dt, im * im))));
            }
            const int64_t yb = rowoff(Y, t, ch * nm);
            for (int64_t m = 0; m < nm; ++m) {
                float s = 0.0f;
                for (int64_t k = 0; k < F; ++k) s += keep(ldt(Mb, m * F + k) * mag[(size_t)k]);
                const float mel = std::max(narrow1(dt, s), floor_v);
                st_elem(Y->data, dt, yb + (h * nm + m) * ys, std::log(mel));
            }
        }
    return RAD_OK;
}

/* ------------------------------------------------------------------ bwe_up */
AVX_KERNEL(bwe_up) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Rz = &a->t[1], *Y = &a->t[2];
    const int64_t r = p_int(a, "ratio", 0), ch = p_int(a, "ch", 0);
    if (r <= 0 || ch <= 0) return RAD_E_SHAPE;
    const int64_t S = rows_of(X, ch), O = rows_of(Y, ch);
    if (S <= 0 || O <= 0 || O > S * r || rows_of(Rz, ch) < O) return RAD_E_SHAPE;
    const uint32_t dt = X->dtype;
    const int64_t w = (int64_t)std::ceil(6.0 / 0.99), K = 2 * w * r + 1, off = 2 * w * r;
    std::vector<float> f((size_t)K);
    for (int64_t n = 0; n < K; ++n) {
        const float u = keep(keep((float)n / (float)r - (float)w) * 0.99f);
        const float uc = std::min(std::max(u, -6.0f), 6.0f);
        const float cw = std::cos(keep(keep(uc * (float)M_PI) / 6.0f) / 2.0f);
        const float pu = keep((float)M_PI * u);
        const float sinc = u == 0.0f ? 1.0f : std::sin(pu) / pu;
        f[(size_t)n] = narrow1(dt, keep(keep(sinc * keep(cw * cw)) * 0.99f) / (float)r);
    }
    const int64_t xs = laststride(X), rs = laststride(Rz), ys = laststride(Y);
    AVX_PARALLEL_FOR_IF(O * ch * K >= AVX_PAR_MIN)
    for (int64_t o = 0; o < O; ++o)
        for (int64_t h = 0; h < ch; ++h) {
            const int64_t u = o + off;
            float acc = 0.0f;
            for (int64_t k = u % r; k < K; k += r) {
                const int64_t i = (u - k) / r;
                const int64_t s = std::min(std::max<int64_t>(i - w, 0), S - 1);
                acc += keep(f[(size_t)k] * ldt(X, rowoff(X, s, ch) + h * xs));
            }
            const float up = narrow1(dt, (float)r * narrow1(dt, acc));
            const float v = narrow1(dt, ldt(Rz, rowoff(Rz, o, ch) + h * rs) + up);
            st_elem(Y->data, Y->dtype, rowoff(Y, o, ch) + h * ys, std::min(std::max(v, -1.0f), 1.0f));
        }
    return RAD_OK;
}
