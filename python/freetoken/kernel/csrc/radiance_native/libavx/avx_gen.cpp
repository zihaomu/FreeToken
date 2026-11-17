/* avx_gen.cpp -- the image generator's ops (docs/OPS.md, "Image generation") except
 * attn_prefix_self (avx_gen_attn.cpp): the diffusion transformer's adaptive norms and gated
 * residual, the timestep embedding, the per-image latent lanes and the flow-matching sampler, and
 * the image decoder's convolution, channel norm, resampling shortcuts and pixel output; and a video
 * generator's modulated RMS norms and sampler step ("Video generation").
 *
 * Each op rounds where libref/ref_gen.cpp rounds, which is where the upstream bf16 model stores a
 * tensor. Rows are computed in f32 scratch and every such rounding is a pass through the stored
 * dtype (`narrow_row`), so a fused kernel produces the bytes of the unfused chain.
 *
 * Bit-exact with libref, not only within tolerance: randn_lanes, flow_euler and flow_x0_step (a
 * seed gives the same picture on every plugin, and the sampler's state feeds every later step),
 * gate_add, chan_affine, to_rgba8, s2d_avg_add, d2s_add, the lane copies and timestep_embed.
 * Where libref multiplies and then adds in two roundings, `keep` stops the compiler from fusing
 * the pair (this plugin is built with -ffp-contract=fast, libref without contraction). The norms
 * reduce in a different order from libref's and are held to the tolerance table.
 *
 * Per-entry tables (a sequence's modulation row, its tanh gate) are built once per launch before
 * the parallel region, in scratch_raw; the row loops use only scratch_f32 / _b / _c, so the
 * calling thread's table is never reallocated under its readers.
 */
#include "avx_vec.h"
#include "avx_common.h"
#include "rad_philox.h"

#include <algorithm>
#include <vector>

using namespace avx;

#include "avx_had.h"

namespace avx {
inline namespace AVX_SUFFIX {
/* gemm_nt's product over operands assembled by a composite kernel (avx_gemm.cpp). */
int gemm_nt_into(const RadTensor* A, const RadTensor* B, const RadTensor* bias, const RadTensor* Y,
                 int64_t M, int64_t N, int64_t K);
}  /* inline namespace AVX_SUFFIX */
}  /* namespace avx */

namespace {

/* An entry's flags (docs/OPS.md, flow_euler): RadGenBatch's RAD_GEN_F_* values, restated as
 * libref restates them, since a kernel library compiles against rad_abi.h only. */
enum { kFirst = 1, kInpaint = 2, kPreview = 4 };

/* Entry `i` of a cu_seqlens operand, and the entry owning row `r` (libref's linear scan). */
inline int64_t cu_at(const RadTensor* cu, int64_t i) { return ldi(cu, offlin(cu, i)); }
inline int64_t seq_of(const RadTensor* cu, int64_t n_seq, int64_t r) {
    int64_t s = 0;
    while (s + 1 < n_seq && cu_at(cu, s + 1) <= r) ++s;
    return s;
}

/* v[0, n) rounded through dtype `dt` in place, as a store and reload rounds it. `tmp` holds n
 * floats. */
inline void narrow_row(uint32_t dt, float* v, int64_t n, float* tmp) {
    if (dt == RAD_F32) return;
    row_from_f32(dt, v, tmp, n);
    row_to_f32(dt, tmp, v, n);
}

/* A value the compiler may not fuse into the next add. */
#if AVX_LEVEL == 0
inline vf keep(vf x) { __asm__("" : "+x"(x)); return x; }
#elif AVX_LEVEL == 3
inline vf keep(vf x) { __asm__("" : "+v"(x)); return x; }
inline float keep(float x) { __asm__("" : "+v"(x)); return x; }
#else
inline vf keep(vf x) { __asm__("" : "+x"(x)); return x; }
inline float keep(float x) { __asm__("" : "+x"(x)); return x; }
#endif

/* Sum of (x[i] - mean)^2 with four accumulators: layer norm's second pass. */
inline float dev_sumsq(const float* x, float mean, int64_t n) {
    const vf vm = vf_set1(mean);
    vf a0 = vf_zero(), a1 = vf_zero(), a2 = vf_zero(), a3 = vf_zero();
    int64_t i = 0;
    for (; i + 4 * VF_N <= n; i += 4 * VF_N) {
        const vf d0 = vf_sub(vf_loadu(x + i), vm), d1 = vf_sub(vf_loadu(x + i + VF_N), vm);
        const vf d2 = vf_sub(vf_loadu(x + i + 2 * VF_N), vm);
        const vf d3 = vf_sub(vf_loadu(x + i + 3 * VF_N), vm);
        a0 = vf_fma(d0, d0, a0);
        a1 = vf_fma(d1, d1, a1);
        a2 = vf_fma(d2, d2, a2);
        a3 = vf_fma(d3, d3, a3);
    }
    for (; i + VF_N <= n; i += VF_N) {
        const vf d = vf_sub(vf_loadu(x + i), vm);
        a0 = vf_fma(d, d, a0);
    }
    float s = vf_hsum(vf_add(vf_add(a0, a1), vf_add(a2, a3)));
    for (; i < n; ++i) { const float d = x[i] - mean; s += d * d; }
    return s;
}

/* out = (x - mean) * rs: the normalised row before its rounding, as libref forms it. */
inline void center_scale(const float* x, float mean, float rs, float* out, int64_t n) {
    const vf vm = vf_set1(mean), vr = vf_set1(rs);
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) vf_storeu(out + i, vf_mul(vf_sub(vf_loadu(x + i), vm), vr));
    for (; i < n; ++i) out[i] = (x[i] - mean) * rs;
}

inline void mul_rows(float* v, const float* w, int64_t n) {
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) vf_storeu(v + i, vf_mul(vf_loadu(v + i), vf_loadu(w + i)));
    for (; i < n; ++i) v[i] *= w[i];
}

inline void mul_into(float* out, const float* a, const float* b, int64_t n) {
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) vf_storeu(out + i, vf_mul(vf_loadu(a + i), vf_loadu(b + i)));
    for (; i < n; ++i) out[i] = a[i] * b[i];
}

inline void add_rows(float* v, const float* w, int64_t n) {
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) vf_storeu(v + i, vf_add(vf_loadu(v + i), vf_loadu(w + i)));
    for (; i < n; ++i) v[i] += w[i];
}

/* An entry's modulation row through `f`, rounded to `dt`, for every entry: [n_seq][n] floats in
 * scratch_raw. `s` is [n_seq, >= n] at its own row pitch (a column slice of the shared
 * modulation projection), read as libref reads it. `f` 0 is 1 + s, 1 is tanh(s), the exact libm
 * tanh: a gate is per entry and column, so n_seq * n tanh calls per launch, not one per element.
 * `f` 2 is s itself, unrounded. */
float* entry_rows(const RadTensor* s, int64_t n_seq, int64_t n, uint32_t dt, int f) {
    float* t = (float*)scratch_raw((size_t)(n_seq * n + 16) * sizeof(float));
    const int64_t rs = row_stride(s, n), cs = laststride(s);
    for (int64_t e = 0; e < n_seq; ++e) {
        float* row = t + e * n;
        for (int64_t i = 0; i < n; ++i) {
            const float v = ldt(s, e * rs + i * cs);
            row[i] = f == 2 ? v : narrow1(dt, f ? sc_tanh(v) : 1.0f + v);
        }
    }
    return t;
}

/* A row copied between operands, converted through f32 as libref's element loop is. Whole-byte
 * float rows of one dtype are copied as bytes, which is the same thing for every non-NaN value. */
inline void copy_row(const RadTensor* src, int64_t sb, const RadTensor* dst, int64_t db, int64_t n,
                     float* scr) {
    const uint32_t dt = src->dtype;
    const bool flt = dt == RAD_F32 || dt == RAD_BF16 || dt == RAD_F16 || dt == RAD_F8E4M3 ||
                     dt == RAD_F8E5M2;
    if (flt && dt == dst->dtype && laststride(src) == 1 && laststride(dst) == 1) {
        const size_t w = (size_t)(rad_dtype_bits(dt) / 8);
        std::memcpy((char*)dst->data + (size_t)db * w, (const char*)src->data + (size_t)sb * w,
                    (size_t)n * w);
        return;
    }
    out_row(dst, db, n, in_row(src, sb, n, scr));
}

/* Elements [e0, e0 + cnt) of `seed`'s normal stream into out (abi/rad_philox.h): the same
 * arithmetic as rad_philox_normal, with one Philox block and one log per element pair instead of
 * one each per element. */
void philox_normals(uint64_t seed, uint64_t e0, int64_t cnt, float* out) {
    uint64_t e = e0;
    int64_t i = 0;
    while (i < cnt) {
        const uint64_t g = e >> 2;
        uint32_t c[4] = { (uint32_t)g, (uint32_t)(g >> 32), 0u, 0u };
        rad_philox4x32_10(c, (uint32_t)seed, (uint32_t)(seed >> 32));
        for (int pair = (int)((e & 3u) >> 1); pair < 2 && i < cnt; ++pair) {
            const uint32_t wa = c[pair * 2], wb = c[pair * 2 + 1];
            const float ua = ((float)(wa >> 8) + 0.5f) * (1.0f / 16777216.0f);
            const float ub = ((float)(wb >> 8) + 0.5f) * (1.0f / 16777216.0f);
            const float r = sqrtf(-2.0f * logf(ua));
            const float t = 6.28318530717958647692f * ub;
            if ((e & 1u) == 0) { out[i++] = r * cosf(t); ++e; }
            if (i < cnt) { out[i++] = r * sinf(t); ++e; }
        }
    }
}

inline uint64_t seed_of(const RadTensor* seed, int64_t e) {
    return (uint64_t)(uint32_t)ldi(seed, offlin(seed, 2 * e)) |
           ((uint64_t)(uint32_t)ldi(seed, offlin(seed, 2 * e + 1)) << 32);
}

/* The bf16 values of a space-separated list (libref's parse_list), or with `bf16` false its f32
 * values: up to `cap` into out. */
int64_t parse_list(const char* s, float* out, int64_t cap, bool bf16 = true) {
    int64_t k = 0;
    while (s && *s) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end == s) break;
        if (k < cap) out[k] = bf16 ? narrow1(RAD_BF16, (float)v) : (float)v;
        ++k;
        s = end;
    }
    return k;
}

}  /* namespace */

/* ------------------------------------------------------------------ ln_mod */
/* y = bf16( bf16(LN(x)) * bf16(1 + s[seq]) ), the roundings in y's dtype. */
AVX_KERNEL(ln_mod) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *S = &a->t[1], *cu = &a->t[2], *Y = &a->t[3];
    const int64_t n = p_int(a, "n", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n), n_seq = numel(cu) - 1;
    if (rows <= 0 || n_seq <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;
    const float* gain = entry_rows(S, n_seq, n, Y->dtype, 0);

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(3 * n + 48));
        float *xs = buf, *d = buf + n + 16, *tmp = d + n + 16;
        const float* xv = in_row(X, rowoff(X, r, n), n, xs);
        const float mean = row_sum(xv, n) / (float)n;
        const float rs = 1.0f / std::sqrt(dev_sumsq(xv, mean, n) / (float)n + eps);
        center_scale(xv, mean, rs, d, n);
        narrow_row(Y->dtype, d, n, tmp);
        mul_rows(d, gain + seq_of(cu, n_seq, r) * n, n);
        out_row(Y, rowoff(Y, r, n), n, d);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gate_add */
/* x = bf16( x + bf16( bf16(tanh(g[seq])) * y ) ), the roundings in x's dtype; with `act` none the
 * gate is g itself, unrounded. */
AVX_KERNEL(gate_add) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *Yv = &a->t[0], *G = &a->t[1], *cu = &a->t[2], *X = &a->t[3];
    const int64_t n = p_int(a, "n", 0);
    static const char* kActs[] = { "tanh", "none" };
    const int act = p_enum(a, "act", kActs, 2, 0);
    if (act < 0 || act > 1) return RAD_E_INVAL;
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Yv, n), n_seq = numel(cu) - 1;
    if (rows <= 0 || n_seq <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;
    const float* gate = entry_rows(G, n_seq, n, X->dtype, act ? 2 : 1);

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(4 * n + 64));
        float *ys = buf, *xs = buf + n + 16, *p = xs + n + 16, *tmp = p + n + 16;
        const float* yv = in_row(Yv, rowoff(Yv, r, n), n, ys);
        const float* xv = in_row(X, rowoff(X, r, n), n, xs);
        const float* t = gate + seq_of(cu, n_seq, r) * n;
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) vf_storeu(p + i, vf_mul(vf_loadu(t + i), vf_loadu(yv + i)));
        for (; i < n; ++i) p[i] = t[i] * yv[i];
        narrow_row(X->dtype, p, n, tmp);
        add_rows(p, xv, n);
        out_row(X, rowoff(X, r, n), n, p);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ln_mod_quant_fp8 */
/* gate_add (with `gated` 1), ln_mod and quant_act_fp8 in one pass over each row: the residual
 * rounded to x's dtype and stored, the norm to bf16, the codes as quant_act_fp8 writes them over
 * that bf16 row (avx_had.h: fp8_emit), and the bf16 row into `z` when bound. */
/* ------------------------------------------------------------------ rms_mod / rms_mod_quant_fp8 */
/* Both forms: with `gated` the branch added through the gate row (no tanh) and the residual stored
 * in x's dtype; then n1 = rd(x rsqrt(mean(x^2) + eps)), n2 = rd(n1 w), m = rd(n2 rd(1 + scale)),
 * z = rd(m + shift), rd the norm chain's dtype (z's for rms_mod, bf16 for the quantised form,
 * libref's rule); `norm` none takes n2 = x. The sum of squares is the vector reduction, held to the tolerance table; every
 * rounding is a pass through the stored dtype, so the rest is libref's arithmetic. The per-entry
 * rows (1 + scale rounded, the shift and the gate) and the gain are built once a launch. */
static int rms_mod_run(const RadArgs* a, bool quant) {
    if (a->n_t < (quant ? 9 : 7)) return RAD_E_INVAL;
    const RadTensor *Yv = rad_arg_in(a, 0), *G = rad_arg_in(a, 1), *S = &a->t[2];
    const RadTensor* W = rad_arg_in(a, 3);
    const RadTensor *cu = &a->t[4], *X = &a->t[5];
    const RadTensor *Q = quant ? &a->t[6] : nullptr, *SC = quant ? &a->t[7] : nullptr;
    const RadTensor* Z = quant ? rad_arg_in(a, 8) : &a->t[6];
    const int64_t n = p_int(a, "n", 0), grp = quant ? p_int(a, "group", 0) : 1;
    const bool gated = p_int(a, "gated", 0) != 0;
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    const bool once = p_int(a, "norm_once", 0) != 0;
    const char* nk = p_str(a, "norm", "rms");
    const bool layer = std::strcmp(nk, "layer") == 0, none = std::strcmp(nk, "none") == 0;
    /* The optional trailing column map (slot 9 quantising, slot 7 plain): a sequence reads the
     * modulation table at its mapped row, so `s` and `g` may hold fewer rows than there are
     * sequences; the map's bound is the issuer's. */
    const RadTensor* cm = rad_arg_in(a, quant ? 9 : 7);
    if ((!layer && !none && std::strcmp(nk, "rms") != 0) || (none && W)) return RAD_E_INVAL;
    if (n <= 0 || grp <= 0 || n % grp || (gated && (!Yv || !G))) return RAD_E_SHAPE;
    const int64_t rows = rows_of(quant ? Q : Z, n), n_seq = numel(cu) - 1, ngrp = n / grp;
    if (cm && (cm->dtype != RAD_I32 || numel(cm) < n_seq)) return RAD_E_SHAPE;
    if (rows <= 0 || n_seq <= 0 || rows_of(X, n) < rows || (W && numel(W) < n) ||
        (!cm && numel(S) < n_seq * 2 * n))
        return RAD_E_SHAPE;
    if (gated && (rows_of(Yv, n) < rows || (!cm && numel(G) < n_seq * n))) return RAD_E_SHAPE;
    if (quant && (numel(SC) < rows * ngrp || (Z && rows_of(Z, n) < rows))) return RAD_E_SHAPE;
    const uint32_t dt = quant ? (uint32_t)RAD_BF16 : Z->dtype;
    /* The tables from one scratch_raw call, [gain | shift | gate] a sequence and the weight row:
     * a second call may reallocate the first. */
    const int64_t tw = gated ? 3 * n : 2 * n;
    float* tabs = (float*)scratch_raw((size_t)(n_seq * tw + n + 32) * sizeof(float));
    {
        const int64_t s_rs = row_stride(S, 2 * n), s_cs = laststride(S);
        const int64_t g_rs = gated ? row_stride(G, n) : 0, g_cs = gated ? laststride(G) : 0;
        for (int64_t e = 0; e < n_seq; ++e) {
            const int64_t col = cm ? (int64_t)ldt(cm, e) : e;  /* the sequence's table row */
            for (int64_t i = 0; i < n; ++i) {
                float* row = tabs + e * tw;
                row[i] = narrow1(dt, 1.0f + ldt(S, col * s_rs + (n + i) * s_cs));
                row[n + i] = ldt(S, col * s_rs + i * s_cs);
                if (gated) row[2 * n + i] = ldt(G, col * g_rs + i * g_cs);
            }
        }
        for (int64_t i = 0; i < n; ++i)
            tabs[n_seq * tw + i] = W ? ldt(W, i * laststride(W)) + wadd : 1.0f;
    }
    const float* wrow = tabs + n_seq * tw;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(5 * n + 80));
        float *xs = buf, *v = buf + n + 16, *o = v + n + 16, *codes = o + n + 16;
        float* tmp = codes + n + 16;
        const float* ent = tabs + seq_of(cu, n_seq, r) * tw;
        const float* xv = in_row(X, rowoff(X, r, n), n, xs);
        if (gated) {
            const float* yv = in_row(Yv, rowoff(Yv, r, n), n, codes);
            mul_into(v, ent + 2 * n, yv, n);
            narrow_row(X->dtype, v, n, tmp);
            add_rows(v, xv, n);
            narrow_row(X->dtype, v, n, tmp);
            out_row(X, rowoff(X, r, n), n, v);
        } else {
            std::memcpy(v, xv, (size_t)n * sizeof(float));
        }
        if (layer) {
            /* Centred first, the mean an f32 sum in ascending order (libref's). */
            float mu = 0.0f;
            for (int64_t k = 0; k < n; ++k) mu += v[k];
            mu /= (float)n;
            for (int64_t k = 0; k < n; ++k) v[k] -= mu;
        }
        /* `none`: no norm, the residual as it is (libref's n2 = x). */
        const float rs = none ? 1.0f : 1.0f / std::sqrt(row_sumsq(v, n) / (float)n + eps);
        const vf vr = vf_set1(rs);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) vf_storeu(o + i, vf_mul(vf_loadu(v + i), vr));
        for (; i < n; ++i) o[i] = v[i] * rs;
        if (W) {
            if (!once) narrow_row(dt, o, n, tmp);
            mul_rows(o, wrow, n);
        }
        narrow_row(dt, o, n, tmp);
        mul_rows(o, ent, n);
        narrow_row(dt, o, n, tmp);
        add_rows(o, ent + n, n);
        if (!quant) {
            out_row(Z, rowoff(Z, r, n), n, o);
            continue;
        }
        narrow_row(dt, o, n, tmp);
        fp8_emit(o, codes, n, grp, Q, SC, Z, r);
    }
    return RAD_OK;
}

AVX_KERNEL(rms_mod) { return rms_mod_run(a, false); }
/* ------------------------------------------------------------------ timestep_embed */
/* y[m] = [cos(t' f_0) .. cos(t' f_{h-1}) | sin(t' f_0) .. sin(t' f_{h-1})] with libref's libm
 * calls: a pass embeds one row per entry, so this is a few thousand transcendentals. With
 * `curve` = G > 0 and a [G, n] f32 `table`, y[m] instead interpolates linearly between the
 * table's two rows around clamp01(max(t[m], clamp_min)) * (G - 1): a checkpoint's timestep
 * embedding precomputed on a grid, in place of the sinusoid and its MLP. */
AVX_KERNEL(timestep_embed) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *T = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    const float scale = p_f32(a, "scale", 1000.0f);
    const double mp = p_f32(a, "max_period", 10000.0f);
    const int round = (int)p_int(a, "round", 0);
    const float clamp = p_f32(a, "clamp_min", 0.0f);
    const int64_t grid = p_int(a, "curve", 0);
    if (n <= 0 || (!grid && (n & 1))) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(T) < rows) return RAD_E_SHAPE;
    if (grid > 0) {
        const RadTensor* tab = rad_arg_in(a, 2);
        if (!tab) return RAD_E_INVAL;
        if (tab->dtype != (uint32_t)RAD_F32 || numel(tab) < grid * n) return RAD_E_SHAPE;
        float* buf = scratch_f32((size_t)n + 16);
        for (int64_t m = 0; m < rows; ++m) {
            const float tv = std::fmax(ldt(T, offlin(T, m)), clamp);
            const float pos = std::fmin(std::fmax(tv, 0.0f), 1.0f) * (float)(grid - 1);
            float i0f = std::floor(pos);
            if (i0f > (float)(grid - 2)) i0f = (float)(grid - 2);
            const float fr = pos - i0f;
            const int64_t i0 = (int64_t)i0f;
            for (int64_t i = 0; i < n; ++i) {
                const float a0 = ldt(tab, offlin(tab, i0 * n + i));
                const float a1 = ldt(tab, offlin(tab, (i0 + 1) * n + i));
                buf[i] = a0 + fr * (a1 - a0);
            }
            out_row(Y, rowoff(Y, m, n), n, buf);
        }
        return RAD_OK;
    }
    const int64_t h = n / 2;
    const float coef = (float)(-std::log(mp));
    float* buf = scratch_f32((size_t)(2 * n + 16));
    float *f = buf, *row = buf + n;
    for (int64_t i = 0; i < h; ++i) f[i] = sc_exp(coef * (float)i / (float)h);
    for (int64_t m = 0; m < rows; ++m) {
        const float tv = std::fmax(ldt(T, offlin(T, m)), clamp);   /* clamp_min: a floor */
        const float tt = round == 3 ? narrow1(RAD_BF16, scale * narrow1(RAD_BF16, tv))
                       : round == 2 ? scale * narrow1(RAD_BF16, tv)
                       : round ? scale * narrow1(RAD_BF16, narrow1(RAD_BF16, tv * scale) / scale)
                               : scale * tv;
        for (int64_t i = 0; i < h; ++i) {
            const float arg = tt * f[i];
            row[i] = sc_cos(arg);
            row[h + i] = sc_sin(arg);
        }
        out_row(Y, rowoff(Y, m, n), n, row);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gather_rows2 */
/* y[i] = idx[i] >= 0 ? a[idx[i]] : b[-1 - idx[i]], zeros past either source. */
AVX_KERNEL(gather_rows2) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *I = &a->t[2], *Y = &a->t[3];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(I) < rows) return RAD_E_SHAPE;
    const int64_t ra = rows_of(A, n), rb = rows_of(B, n);

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t i = 0; i < rows; ++i) {
        float* scr = scratch_f32((size_t)n + 16);
        const int64_t r = ldi(I, offlin(I, i));
        const RadTensor* src = r >= 0 ? A : B;
        const int64_t row = r >= 0 ? r : -1 - r;
        if (row >= (r >= 0 ? ra : rb)) {
            for (int64_t c = 0; c < n; ++c) scr[c] = 0.0f;
            out_row(Y, rowoff(Y, i, n), n, scr);
            continue;
        }
        copy_row(src, rowoff(src, row, n), Y, rowoff(Y, i, n), n, scr);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ lane_gather / lane_store */
/* lane_gather: y[cu[e] + k] = src[lane[e] * stride + k]. Every entry is checked before any row is
 * written (libref checks each as it reaches it), and an entry whose rows lie past y's end is
 * refused rather than written past it. */
AVX_KERNEL(lane_gather) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *src = &a->t[0], *lane = &a->t[1], *cu = &a->t[2], *Y = &a->t[3];
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1, src_rows = rows_of(src, n), y_rows = rows_of(Y, n);
    if (n_seq <= 0 || numel(lane) < n_seq || cu_at(cu, n_seq) <= cu_at(cu, 0)) return RAD_E_SHAPE;
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t L = ldi(lane, offlin(lane, e)), b0 = cu_at(cu, e), b1 = cu_at(cu, e + 1);
        if (L < 0 || b1 - b0 > stride || (L + 1) * stride > src_rows) return RAD_E_SHAPE;
        if (b1 > b0 && (b0 < 0 || b1 > y_rows)) return RAD_E_SHAPE;
    }
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t L = ldi(lane, offlin(lane, e)), b0 = cu_at(cu, e), len = cu_at(cu, e + 1) - b0;
        AVX_PARALLEL_FOR_IF(len * n >= AVX_PAR_MIN)
        for (int64_t k = 0; k < len; ++k) {
            float* scr = scratch_f32((size_t)n + 16);
            copy_row(src, rowoff(src, L * stride + k, n), Y, rowoff(Y, b0 + k, n), n, scr);
        }
    }
    return RAD_OK;
}

/* lane_store: every listed lane's rows [0, rows) become src's rows. */
AVX_KERNEL(lane_store) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *src = &a->t[0], *lane = &a->t[1], *dst = &a->t[2];
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(src, n), dst_rows = rows_of(dst, n), nl = numel(lane);
    if (rows <= 0 || rows > stride) return RAD_E_SHAPE;
    for (int64_t l = 0; l < nl; ++l) {
        const int64_t L = ldi(lane, offlin(lane, l));
        if (L < 0 || (L + 1) * stride > dst_rows) return RAD_E_SHAPE;
    }
    for (int64_t l = 0; l < nl; ++l) {
        const int64_t L = ldi(lane, offlin(lane, l));
        AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
        for (int64_t k = 0; k < rows; ++k) {
            float* scr = scratch_f32((size_t)n + 16);
            copy_row(src, rowoff(src, k, n), dst, rowoff(dst, L * stride + k, n), n, scr);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ randn_lanes */
/* An image's first step: its lane's rows [0, cu[e+1] - cu[e]) become the seed's normal stream,
 * element (k, c) the stream's k * n + c, bit for bit libref's. */
AVX_KERNEL(randn_lanes) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *fl = &a->t[0], *seed = &a->t[1], *lane = &a->t[2], *cu = &a->t[3];
    const RadTensor* dst = &a->t[4];
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1, dst_rows = rows_of(dst, n);
    if (n_seq <= 0 || numel(fl) < n_seq || numel(lane) < n_seq || numel(seed) < 2 * n_seq)
        return RAD_E_SHAPE;
    if (cu_at(cu, n_seq) <= cu_at(cu, 0)) return RAD_E_SHAPE;
    for (int64_t e = 0; e < n_seq; ++e) {
        if (!(ldi(fl, offlin(fl, e)) & kFirst)) continue;
        const int64_t L = ldi(lane, offlin(lane, e)), len = cu_at(cu, e + 1) - cu_at(cu, e);
        if (L < 0 || len > stride || (L + 1) * stride > dst_rows) return RAD_E_SHAPE;
    }
    for (int64_t e = 0; e < n_seq; ++e) {
        if (!(ldi(fl, offlin(fl, e)) & kFirst)) continue;
        const int64_t L = ldi(lane, offlin(lane, e)), len = cu_at(cu, e + 1) - cu_at(cu, e);
        const uint64_t sd = seed_of(seed, e);
        AVX_PARALLEL_FOR_IF(len * n * 8 >= AVX_PAR_MIN)
        for (int64_t k = 0; k < len; ++k) {
            float* row = scratch_f32((size_t)n + 16);
            philox_normals(sd, (uint64_t)(k * n), n, row);
            out_row(dst, rowoff(dst, L * stride + k, n), n, row);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ flow_euler */
/* One Euler step for every entry that moves a lane (pair != -2), libref's roundings in the state's
 * dtype:
 *
 *   v'  = pair >= 0 ? bf16(neg + bf16(cfg * bf16(v - neg))) : v
 *   x'  = bf16(x + bf16(bf16(sigma_next - sigma) * v'))
 *
 * flag 4 writes x - sigma v' to the lane's preview, and flag 2 restores the kept region,
 * x' = bf16(m x' + (1 - m)((1 - sigma_next) src + sigma_next noise)). inp, mask and preview are
 * optional, as in libref: an entry whose flags need one that is absent is refused. Every entry is
 * checked before any lane is written. */
AVX_KERNEL(flow_euler) {
    if (!rad_args_have(a, 9) || !rad_arg_in(a, 11)) return RAD_E_INVAL;
    const RadTensor *V = &a->t[0], *sg = &a->t[1], *sgn = &a->t[2], *cfg = &a->t[3];
    const RadTensor *pair = &a->t[4], *lane = &a->t[5], *fl = &a->t[6], *seed = &a->t[7];
    const RadTensor *cu = &a->t[8], *inp = rad_arg_in(a, 9), *mask = rad_arg_in(a, 10);
    const RadTensor *st = &a->t[11], *pv = rad_arg_in(a, 12);
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1, st_rows = rows_of(st, n), v_rows = rows_of(V, n);
    if (n_seq <= 0 || cu_at(cu, n_seq) <= cu_at(cu, 0)) return RAD_E_SHAPE;
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t pr = ldi(pair, offlin(pair, e));
        if (pr == -2) continue;
        const int64_t L = ldi(lane, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), len = cu_at(cu, e + 1) - b0;
        if (L < 0 || len > stride || (L + 1) * stride > st_rows) return RAD_E_SHAPE;
        if (pr >= n_seq || (pr >= 0 && cu_at(cu, pr + 1) - cu_at(cu, pr) != len))
            return RAD_E_SHAPE;
        if (len > 0 && (b0 < 0 || b0 + len > v_rows)) return RAD_E_SHAPE;
        if (pr >= 0 && len > 0 && (cu_at(cu, pr) < 0 || cu_at(cu, pr) + len > v_rows))
            return RAD_E_SHAPE;
        const int64_t f = ldi(fl, offlin(fl, e));
        if (((f & kInpaint) && (!inp || !mask)) || ((f & kPreview) && !pv)) return RAD_E_INVAL;
    }
    const uint32_t dt = st->dtype;
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t pr = ldi(pair, offlin(pair, e));
        if (pr == -2) continue;
        const int64_t L = ldi(lane, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), len = cu_at(cu, e + 1) - b0;
        const int64_t p0 = pr >= 0 ? cu_at(cu, pr) : 0;
        const float s = ldt(sg, offlin(sg, e)), sn = ldt(sgn, offlin(sgn, e));
        const float w = ldt(cfg, offlin(cfg, e));
        const int64_t f = ldi(fl, offlin(fl, e));
        const uint64_t sd = seed_of(seed, e);
        const float dts = narrow1(dt, sn - s);
        const float one_sn = 1.0f - sn;
        AVX_PARALLEL_FOR_IF(len * n * 8 >= AVX_PAR_MIN)
        for (int64_t k = 0; k < len; ++k) {
            float* buf = scratch_f32((size_t)(5 * n + 80));
            float *vv = buf, *xs = vv + n + 16, *u = xs + n + 16, *nz = u + n + 16;
            float* tmp = nz + n + 16;
            const float* vin = in_row(V, rowoff(V, b0 + k, n), n, u);
            std::memcpy(vv, vin, (size_t)n * sizeof(float));
            if (pr >= 0) {
                const float* vn = in_row(V, rowoff(V, p0 + k, n), n, u);
                int64_t c = 0;
                for (; c + VF_N <= n; c += VF_N)
                    vf_storeu(vv + c, vf_sub(vf_loadu(vv + c), vf_loadu(vn + c)));
                for (; c < n; ++c) vv[c] -= vn[c];
                narrow_row(dt, vv, n, tmp);
                const vf vw = vf_set1(w);
                for (c = 0; c + VF_N <= n; c += VF_N) vf_storeu(vv + c, vf_mul(vw, vf_loadu(vv + c)));
                for (; c < n; ++c) vv[c] *= w;
                narrow_row(dt, vv, n, tmp);
                add_rows(vv, vn, n);
                narrow_row(dt, vv, n, tmp);
            }
            const int64_t sb = rowoff(st, L * stride + k, n);
            const float* x = in_row(st, sb, n, xs);
            if (f & kPreview) {
                const vf vs = vf_set1(s);
                int64_t c = 0;
                for (; c + VF_N <= n; c += VF_N)
                    vf_storeu(tmp + c, vf_sub(vf_loadu(x + c), keep(vf_mul(vs, vf_loadu(vv + c)))));
                for (; c < n; ++c) tmp[c] = x[c] - keep(s * vv[c]);
                out_row(pv, rowoff(pv, L * stride + k, n), n, tmp);
            }
            {
                const vf vd = vf_set1(dts);
                int64_t c = 0;
                for (; c + VF_N <= n; c += VF_N) vf_storeu(u + c, vf_mul(vd, vf_loadu(vv + c)));
                for (; c < n; ++c) u[c] = dts * vv[c];
                narrow_row(dt, u, n, tmp);
                add_rows(u, x, n);
                narrow_row(dt, u, n, tmp);
            }
            if (f & kInpaint) {
                const float m = ldt(mask, offlin(mask, L * stride + k));
                const float one_m = 1.0f - m;
                const float* src = in_row(inp, rowoff(inp, L * stride + k, n), n, xs);
                philox_normals(sd, (uint64_t)(k * n), n, nz);
                const vf v1 = vf_set1(one_sn), v2 = vf_set1(sn), vm = vf_set1(m), v3 = vf_set1(one_m);
                int64_t c = 0;
                for (; c + VF_N <= n; c += VF_N) {
                    const vf kept = vf_add(keep(vf_mul(v1, vf_loadu(src + c))),
                                           keep(vf_mul(v2, vf_loadu(nz + c))));
                    vf_storeu(u + c, vf_add(keep(vf_mul(vm, vf_loadu(u + c))),
                                            keep(vf_mul(v3, kept))));
                }
                for (; c < n; ++c) {
                    const float kept = keep(one_sn * src[c]) + keep(sn * nz[c]);
                    u[c] = keep(m * u[c]) + keep(one_m * kept);
                }
                narrow_row(dt, u, n, tmp);
            }
            out_row(st, sb, n, u);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ kd_input, kd_step */
/* The k-diffusion sampler over a discrete-time denoiser (libref/ref_gen.cpp): libref's f32 steps
 * bit for bit, element by element, `kd_keep` holding every product to its own rounding before the
 * sum that follows it. The lanes are a few thousand rows of four channels, so a scalar loop costs
 * nothing next to the denoiser. */
namespace {

enum { kSamplerEuler = 0, kSamplerEulerAncestral = 1, kSamplerDpmpp2m = 2 };

/* A product the compiler may not fuse into the next add (any ISA level: a scalar in an xmm). */
inline float kd_keep(float x) { __asm__("" : "+x"(x)); return x; }

struct KdEntry {
    float s = 0, sn = 0, w = 0;
    float cskip = 0, cout = 0;
    float sup = 0, sdown = 0;
    float ratio = 0, em = 0;
    float c1 = 0, c2 = 0;
    bool  second = false;
};

KdEntry kd_entry(float s, float sn, float sp, float w, int64_t sampler, bool first) {
    KdEntry k;
    k.s = s;
    k.sn = sn;
    k.w = w;
    const float s2p = kd_keep(s * s) + 1.0f;
    k.cskip = 1.0f / s2p;
    k.cout = s / std::sqrt(s2p);
    if (sampler == kSamplerEulerAncestral) {
        const float s2 = kd_keep(s * s), sn2 = kd_keep(sn * sn);
        const float up2 = kd_keep(sn2 * (s2 - sn2)) / s2;
        float sup = std::sqrt(up2);
        if (!(sup < sn)) sup = sn;
        const float dn2 = sn2 - kd_keep(sup * sup);
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
            k.c2 = 1.0f / kd_keep(2.0f * r);
            k.c1 = 1.0f + k.c2;
            k.second = true;
        }
    }
    return k;
}

inline float kd_denoised(const KdEntry& k, bool v_pred, float x, float o) {
    return v_pred ? kd_keep(x * k.cskip) - kd_keep(o * k.cout) : x - kd_keep(k.s * o);
}

inline float kd_next(const KdEntry& k, int64_t sampler, float x, float D, float hist, float nz) {
    if (sampler == kSamplerEuler) {
        const float d = (x - D) / k.s;
        return x + kd_keep((k.sn - k.s) * d);
    }
    if (sampler == kSamplerEulerAncestral) {
        const float d = (x - D) / k.s;
        float xn = x + kd_keep(d * (k.sdown - k.s));
        if (k.sn > 0.0f) xn = xn + kd_keep(nz * k.sup);
        return xn;
    }
    if (k.sn == 0.0f) return D;
    const float Dd = k.second ? kd_keep(k.c1 * D) - kd_keep(k.c2 * hist) : D;
    return kd_keep(k.ratio * x) - kd_keep(k.em * Dd);
}

}  /* namespace */

/* The model's input per entry that moves a lane: a first step's noise times sqrt(1 + sigma^2)
 * into the lane, then rd(x / sqrt(sigma^2 + 1)) at the entry's rows and its pair's. Every entry is
 * checked before any is written. */
AVX_KERNEL(kd_input) {
    if (!rad_args_have(a, 8)) return RAD_E_INVAL;
    const RadTensor *sg = &a->t[0], *pair = &a->t[1], *lane = &a->t[2], *fl = &a->t[3];
    const RadTensor *seed = &a->t[4], *cu = &a->t[5], *st = &a->t[6], *Y = &a->t[7];
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1, st_rows = rows_of(st, n), y_rows = rows_of(Y, n);
    if (n_seq <= 0 || cu_at(cu, n_seq) <= cu_at(cu, 0)) return RAD_E_SHAPE;
    if (numel(sg) < n_seq || numel(pair) < n_seq || numel(lane) < n_seq || numel(fl) < n_seq ||
        numel(seed) < 2 * n_seq)
        return RAD_E_SHAPE;
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t pr = ldi(pair, offlin(pair, e));
        if (pr == -2) continue;
        const int64_t L = ldi(lane, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), len = cu_at(cu, e + 1) - b0;
        if (L < 0 || len > stride || (L + 1) * stride > st_rows || b0 < 0 || b0 + len > y_rows)
            return RAD_E_SHAPE;
        if (pr >= n_seq || (pr >= 0 && (cu_at(cu, pr + 1) - cu_at(cu, pr) != len ||
                                        cu_at(cu, pr) < 0 || cu_at(cu, pr) + len > y_rows)))
            return RAD_E_SHAPE;
        if (ldi(fl, offlin(fl, e)) & kInpaint) return RAD_E_UNSUPPORTED;
    }
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t pr = ldi(pair, offlin(pair, e));
        if (pr == -2) continue;
        const int64_t L = ldi(lane, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), len = cu_at(cu, e + 1) - b0;
        const int64_t p0 = pr >= 0 ? cu_at(cu, pr) : 0;
        const float s = ldt(sg, offlin(sg, e));
        const float init = std::sqrt(1.0f + kd_keep(s * s));
        const float cin = 1.0f / std::sqrt(kd_keep(s * s) + 1.0f);
        const bool first = (ldi(fl, offlin(fl, e)) & kFirst) != 0;
        const uint64_t sd = seed_of(seed, e);
        AVX_PARALLEL_FOR_IF(len * n * 8 >= AVX_PAR_MIN)
        for (int64_t k = 0; k < len; ++k) {
            const int64_t sb = rowoff(st, L * stride + k, n), ss = laststride(st);
            const int64_t yb = rowoff(Y, b0 + k, n), ys = laststride(Y);
            const int64_t pb = pr >= 0 ? rowoff(Y, p0 + k, n) : 0;
            for (int64_t c = 0; c < n; ++c) {
                if (first) stt(st, sb + c * ss, rad_philox_normal(sd, (uint64_t)(k * n + c)) * init);
                const float v = ldt(st, sb + c * ss) * cin;
                stt(Y, yb + c * ys, v);
                if (pr >= 0) stt(Y, pb + c * ys, v);
            }
        }
    }
    return RAD_OK;
}

/* One sampler step per entry that moves a lane, libref's arithmetic (docs/OPS.md, kd_step). */
AVX_KERNEL(kd_step) {
    if (!rad_args_have(a, 14)) return RAD_E_INVAL;
    const RadTensor *O = &a->t[0], *sg = &a->t[1], *sgn = &a->t[2], *sgp = &a->t[3];
    const RadTensor *cfg = &a->t[4], *pair = &a->t[5], *lane = &a->t[6], *fl = &a->t[7];
    const RadTensor *seed = &a->t[8], *smp = &a->t[9], *stp = &a->t[10], *cu = &a->t[11];
    const RadTensor *st = &a->t[12], *hs = &a->t[13], *pv = rad_arg_in(a, 14);
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
        const int64_t pr = ldi(pair, offlin(pair, e));
        if (pr == -2) continue;
        const int64_t L = ldi(lane, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), len = cu_at(cu, e + 1) - b0;
        if (L < 0 || len > stride || (L + 1) * stride > st_rows || b0 < 0 || b0 + len > o_rows)
            return RAD_E_SHAPE;
        if (pr >= n_seq || (pr >= 0 && (cu_at(cu, pr + 1) - cu_at(cu, pr) != len ||
                                        cu_at(cu, pr) < 0 || cu_at(cu, pr) + len > o_rows)))
            return RAD_E_SHAPE;
        const int64_t f = ldi(fl, offlin(fl, e)), sm = ldi(smp, offlin(smp, e));
        if ((f & kInpaint) || sm < kSamplerEuler || sm > kSamplerDpmpp2m) return RAD_E_UNSUPPORTED;
        if ((f & kPreview) && !pv) return RAD_E_INVAL;
    }
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t pr = ldi(pair, offlin(pair, e));
        if (pr == -2) continue;
        const int64_t L = ldi(lane, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), len = cu_at(cu, e + 1) - b0;
        const int64_t p0 = pr >= 0 ? cu_at(cu, pr) : 0;
        const int64_t f = ldi(fl, offlin(fl, e)), sm = ldi(smp, offlin(smp, e));
        const uint64_t stream = (uint64_t)(uint32_t)(1 + ldi(stp, offlin(stp, e)));
        const KdEntry K = kd_entry(ldt(sg, offlin(sg, e)), ldt(sgn, offlin(sgn, e)),
                                   ldt(sgp, offlin(sgp, e)), ldt(cfg, offlin(cfg, e)), sm,
                                   (f & kFirst) != 0);
        const uint64_t sd = seed_of(seed, e);
        AVX_PARALLEL_FOR_IF(len * n * 8 >= AVX_PAR_MIN)
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
                    D = Du + kd_keep(K.w * (D - Du));
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

/* ------------------------------------------------------------------ flow_x0_step */
/* x' = f32(f32(r x) + f32(f32(1 - r) d)) with d = f32(x + f32((1 - t) v)) and
 * r = sigma_next / sigma: libref's f32 steps bit for bit, `keep` holding each product to its own
 * rounding. */
AVX_KERNEL(flow_x0_step) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *V = &a->t[0], *T = &a->t[1], *sg = &a->t[2], *sgn = &a->t[3], *X = &a->t[4];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(X, n);
    if (rows <= 0 || rows_of(V, n) < rows || numel(T) < 1 || numel(sg) < 1 || numel(sgn) < 1)
        return RAD_E_SHAPE;
    const float st = 1.0f - ldt(T, offlin(T, 0));
    const float r = ldt(sgn, offlin(sgn, 0)) / ldt(sg, offlin(sg, 0));
    const float omr = 1.0f - r;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t k = 0; k < rows; ++k) {
        float* buf = scratch_f32((size_t)(3 * n + 48));
        float *vs = buf, *xs = buf + n + 16, *o = xs + n + 16;
        const float* v = in_row(V, rowoff(V, k, n), n, vs);
        const float* x = in_row(X, rowoff(X, k, n), n, xs);
        const vf vst = vf_set1(st), vr = vf_set1(r), vo = vf_set1(omr);
        int64_t c = 0;
        for (; c + VF_N <= n; c += VF_N) {
            const vf xv = vf_loadu(x + c);
            const vf d = vf_add(xv, keep(vf_mul(vst, vf_loadu(v + c))));
            vf_storeu(o + c, vf_add(keep(vf_mul(vr, xv)), keep(vf_mul(vo, d))));
        }
        for (; c < n; ++c) {
            const float d = x[c] + keep(st * v[c]);
            o[c] = keep(r * x[c]) + keep(omr * d);
        }
        out_row(X, rowoff(X, k, n), n, o);
    }
    return RAD_OK;
}

/* x' = f32(f32(r x) + f32(hb1 d) + f32(hb2 x0)) when order >= 2, else the Euler form
 * f32(f32(r x) + f32((1 - r) d)), with d = f32(x + f32(sigma v)): libref's step bit for bit, `keep`
 * holding each product to its own rounding. x0 is the previous estimate (read here, d written back). */
AVX_KERNEL(flow_res_multistep) {
    if (!rad_args_have(a, 8)) return RAD_E_INVAL;
    const RadTensor *V = &a->t[0], *X0 = &a->t[1], *SG = &a->t[2], *RR = &a->t[3];
    const RadTensor *HB1 = &a->t[4], *HB2 = &a->t[5], *ORD = &a->t[6], *X = &a->t[7];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(X, n);
    if (rows <= 0 || rows_of(V, n) < rows || rows_of(X0, n) < rows || numel(SG) < 1 ||
        numel(RR) < 1 || numel(HB1) < 1 || numel(HB2) < 1 || numel(ORD) < 1)
        return RAD_E_SHAPE;
    const float sg = ldt(SG, offlin(SG, 0));
    const float r = ldt(RR, offlin(RR, 0));
    const float b1 = ldt(HB1, offlin(HB1, 0));
    const float b2 = ldt(HB2, offlin(HB2, 0));
    const bool two = ldt(ORD, offlin(ORD, 0)) >= 1.5f;
    const float omr = 1.0f - r;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t k = 0; k < rows; ++k) {
        float* buf = scratch_f32((size_t)(5 * n + 64));
        float *vs = buf, *xs = buf + n + 16, *h0s = xs + n + 16, *o = h0s + n + 16, *o0 = o + n + 16;
        const float* v = in_row(V, rowoff(V, k, n), n, vs);
        const float* x = in_row(X, rowoff(X, k, n), n, xs);
        const float* h0 = in_row(X0, rowoff(X0, k, n), n, h0s);
        const vf vsg = vf_set1(sg), vr = vf_set1(r), vb1 = vf_set1(b1), vb2 = vf_set1(b2),
                 vo = vf_set1(omr);
        int64_t c = 0;
        for (; c + VF_N <= n; c += VF_N) {
            const vf xv = vf_loadu(x + c);
            const vf d = vf_add(xv, keep(vf_mul(vsg, vf_loadu(v + c))));
            const vf a = keep(vf_mul(vr, xv));
            vf xo = two ? vf_add(vf_add(a, keep(vf_mul(vb1, d))), keep(vf_mul(vb2, vf_loadu(h0 + c))))
                        : vf_add(a, keep(vf_mul(vo, d)));
            vf_storeu(o + c, xo);
            vf_storeu(o0 + c, d);
        }
        for (; c < n; ++c) {
            const float d = x[c] + keep(sg * v[c]);
            const float a = keep(r * x[c]);
            o[c] = two ? (a + keep(b1 * d)) + keep(b2 * h0[c]) : a + keep(omr * d);
            o0[c] = d;
        }
        out_row(X, rowoff(X, k, n), n, o);
        out_row(X0, rowoff(X0, k, n), n, o0);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ chan_affine */
/* A per-channel affine from two lists of n bf16 values: `fma` is bf16(bf16(x * mul) + add), `inv`
 * is bf16(bf16(x - add) / mul), the inner rounding in y's dtype. */
AVX_KERNEL(chan_affine) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), rnd = p_int(a, "round", 1);
    if (rnd < 0 || rnd > 1) return RAD_E_INVAL;
    if (n <= 0) return RAD_E_SHAPE;
    const char* ms = p_str(a, "mul", ""), *as = p_str(a, "add", "");
    /* Each number takes at least two characters but the last, so this bounds the count. */
    const int64_t cap = (int64_t)((std::strlen(ms) + std::strlen(as)) / 2 + 2);
    float* lists = (float*)scratch_raw((size_t)(2 * cap + 16) * sizeof(float));
    float *mul = lists, *add = lists + cap;
    if (parse_list(ms, mul, cap, rnd) != n || parse_list(as, add, cap, rnd) != n)
        return RAD_E_SHAPE;
    const char* mode = p_str(a, "mode", "fma");
    const bool inv = mode && !std::strcmp(mode, "inv");
    if (!inv && mode && std::strcmp(mode, "fma")) return RAD_E_INVAL;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(3 * n + 48));
        float *xs = buf, *t = buf + n + 16, *tmp = t + n + 16;
        const float* x = in_row(X, rowoff(X, r, n), n, xs);
        int64_t c = 0;
        if (inv) {
            for (; c + VF_N <= n; c += VF_N)
                vf_storeu(t + c, vf_sub(vf_loadu(x + c), vf_loadu(add + c)));
            for (; c < n; ++c) t[c] = x[c] - add[c];
            if (rnd) narrow_row(Y->dtype, t, n, tmp);
            for (c = 0; c + VF_N <= n; c += VF_N)
                vf_storeu(t + c, vf_div(vf_loadu(t + c), vf_loadu(mul + c)));
            for (; c < n; ++c) t[c] /= mul[c];
        } else if (rnd) {
            for (; c + VF_N <= n; c += VF_N)
                vf_storeu(t + c, vf_mul(vf_loadu(x + c), vf_loadu(mul + c)));
            for (; c < n; ++c) t[c] = x[c] * mul[c];
            narrow_row(Y->dtype, t, n, tmp);
            add_rows(t, add, n);
        } else {
            for (; c + VF_N <= n; c += VF_N)
                vf_storeu(t + c, vf_add(keep(vf_mul(vf_loadu(x + c), vf_loadu(mul + c))),
                                        vf_loadu(add + c)));
            for (; c < n; ++c) t[c] = keep(x[c] * mul[c]) + add[c];
        }
        out_row(Y, rowoff(Y, r, n), n, t);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ im2col_nhwc */
/* conv2d_nhwc's receptive fields as rows (libref's definition): y [Ho * Wo, cin * k * k], column
 * (ci k + i) k + j of row oy * Wo + ox the input at that tap, 0 in the padding. Each output row is
 * filled tap by tap from the input pixel's channel run; bf16 into bf16 copies the bits. Parallel
 * over output image rows. */
AVX_KERNEL(im2col_nhwc) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t cin = p_int(a, "cin", 0), k = p_int(a, "k", 0);
    const int64_t stride = p_int(a, "stride", 1), up = p_int(a, "up", 1);
    const char* pad = p_str(a, "pad", "same");
    const bool down = pad && !std::strcmp(pad, "down");
    const bool causal = pad && !std::strcmp(pad, "causal");
    if (cin <= 0 || k <= 0 || stride <= 0 || up <= 0) return RAD_E_SHAPE;
    if (X->rank != 2 || X->shape[1] % cin) return RAD_E_SHAPE;
    const int64_t Hin = X->shape[0], Win = X->shape[1] / cin, kk = cin * k * k;
    if (Hin <= 0 || Win <= 0) return RAD_E_SHAPE;
    const int64_t Hv = Hin * up, Wv = Win * up;
    const int64_t pt = down ? 0 : causal ? k - 1 : k / 2, pb = down ? 1 : causal ? 0 : k / 2;
    const int64_t pl = down ? 0 : k / 2, pr = down ? 1 : k / 2;
    const int64_t Ho = (Hv + pt + pb - k) / stride + 1, Wo = (Wv + pl + pr - k) / stride + 1;
    if (Ho <= 0 || Wo <= 0 || rows_of(Y, kk) != Ho * Wo) return RAD_E_SHAPE;
    const int64_t xr = X->stride[0], xs = laststride(X), ys = laststride(Y);
    const bool bits = X->dtype == RAD_BF16 && Y->dtype == RAD_BF16;
    AVX_PARALLEL_FOR_IF(Ho * Wo * kk >= AVX_PAR_MIN)
    for (int64_t oy = 0; oy < Ho; ++oy)
        for (int64_t ox = 0; ox < Wo; ++ox) {
            const int64_t yb = rowoff(Y, oy * Wo + ox, kk);
            for (int64_t i = 0; i < k; ++i) {
                const int64_t iy = oy * stride + i - pt;
                for (int64_t j = 0; j < k; ++j) {
                    const int64_t ix = ox * stride + j - pl;
                    const bool in = iy >= 0 && iy < Hv && ix >= 0 && ix < Wv;
                    const int64_t xb = in ? (iy / up) * xr + (ix / up) * cin * xs : 0;
                    const int64_t yt = yb + (i * k + j) * ys;
                    for (int64_t ci = 0; ci < cin; ++ci) {
                        const int64_t yo = yt + ci * k * k * ys;
                        if (bits) {
                            ((uint16_t*)Y->data)[yo] =
                                in ? ((const uint16_t*)X->data)[xb + ci * xs] : (uint16_t)0;
                        } else {
                            stt(Y, yo, in ? ldt(X, xb + ci * xs) : 0.0f);
                        }
                    }
                }
            }
        }
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv2d_nhwc */
/* An image convolution as GEMMs over patch rows. x [Hin, Win * cin] and y [Ho, Wo * cout] are
 * images whose sizes come from the operands, rows at their own pitch and channels contiguous, as
 * libref reads them; w is PyTorch's [cout, cin, k, k] and b [cout].
 *
 * The weight is re-laid once per launch as [cout][kh][kw][cin], so an output pixel's patch row is
 * its k * k input pixels' channel runs back to back: k * k contiguous copies, zeros outside the
 * image, the upsample read as x[y / 2, x / 2]. A band of output rows' patches is one gemm_nt_into
 * call (A the patches, B the weight, the bias added before the one rounding), which takes the bf16
 * outer-product path where the CPU has AVX512_BF16. A 1 x 1 convolution reads x itself as A. A
 * layer narrower than one 48-column panel (the decoder's last, four channels) computes
 * y^T = W A^T into f32 instead, so the parallel axis is the pixels rather than four channels, and
 * adds the bias as it rounds into y.
 *
 * Patch rows stay in x's dtype when it is bf16 (exact copies) and are f32 otherwise; the weight
 * likewise. They live in thread_local vectors that keep their high-water size, so the GEMM's own
 * scratch arenas stay free for it. */
AVX_KERNEL(conv2d_nhwc) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *W = &a->t[1], *B = &a->t[2], *Y = &a->t[3];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0), k = p_int(a, "k", 0);
    const int64_t stride = p_int(a, "stride", 1), up = p_int(a, "up", 1);
    const char* pad = p_str(a, "pad", "same");
    const bool down = pad && !std::strcmp(pad, "down");
    if (cin <= 0 || cout <= 0 || k <= 0 || stride <= 0 || up <= 0) return RAD_E_SHAPE;
    if (X->rank != 2 || Y->rank != 2 || X->shape[1] % cin || Y->shape[1] % cout) return RAD_E_SHAPE;
    const int64_t Hin = X->shape[0], Win = X->shape[1] / cin;
    const int64_t Ho = Y->shape[0], Wo = Y->shape[1] / cout;
    if (Hin <= 0 || Win <= 0 || Ho <= 0 || Wo <= 0) return RAD_E_SHAPE;
    const int64_t Hv = Hin * up, Wv = Win * up;
    /* `causal` pads k - 1 rows above and none below (libref's). */
    const bool causal = pad && !std::strcmp(pad, "causal");
    const int64_t pt = down ? 0 : causal ? k - 1 : k / 2, pb = down ? 1 : causal ? 0 : k / 2;
    const int64_t pl = down ? 0 : k / 2, pr = down ? 1 : k / 2;
    if ((Hv + pt + pb - k) / stride + 1 != Ho || (Wv + pl + pr - k) / stride + 1 != Wo)
        return RAD_E_SHAPE;
    if (numel(W) < cout * cin * k * k || numel(B) < cout) return RAD_E_SHAPE;
    const int64_t xr = X->stride[0], yr = Y->stride[0];
    const int64_t KK = k * k * cin;
    const bool xb = X->dtype == RAD_BF16;
    const uint32_t adt = xb ? (uint32_t)RAD_BF16 : (uint32_t)RAD_F32;
    const size_t ael = xb ? 2 : 4;
    const bool narrow = cout < 48;
    static thread_local std::vector<uint8_t> wbuf, pbuf;
    static thread_local std::vector<float> ybuf, bbuf;

    /* ---- the weight, [cout][kh][kw][cin] */
    const bool wb16 = W->dtype == RAD_BF16;
    const uint32_t wdt = wb16 ? (uint32_t)RAD_BF16 : (uint32_t)RAD_F32;
    const size_t wel = wb16 ? 2 : 4;
    const bool w_as_is = k == 1 && (wb16 || W->dtype == RAD_F32);
    if (!w_as_is) {
        wbuf.resize((size_t)(cout * KK) * wel + 64);
        uint8_t* wp = wbuf.data();
        AVX_PARALLEL_FOR_IF(cout * KK >= AVX_PAR_MIN)
        for (int64_t co = 0; co < cout; ++co)
            for (int64_t ci = 0; ci < cin; ++ci)
                for (int64_t t = 0; t < k * k; ++t) {
                    const int64_t src = (co * cin + ci) * k * k + t, dst = co * KK + t * cin + ci;
                    if (wb16) ((uint16_t*)wp)[dst] = ((const uint16_t*)W->data)[src];
                    else      ((float*)wp)[dst] = ld_elem(W->data, W->dtype, src);
                }
    }
    RadTensor wt{};
    wt.data = w_as_is ? W->data : (void*)wbuf.data();
    wt.dtype = wdt;
    wt.rank = 2;
    wt.shape[0] = cout; wt.shape[1] = KK;
    wt.stride[0] = KK; wt.stride[1] = 1;

    /* ---- bands of whole output rows, or one row a band when y's rows are not back to back */
    const bool y_dense = yr == Wo * cout;
    const bool direct = k == 1 && stride == 1 && up == 1 && !down && xr == Win * cin &&
                        (xb || X->dtype == RAD_F32);
    /* A band's patches: at least 8 Mi elements (thousands of pixels, enough M for the GEMM's
     * tiles to fill the threads) and eight times the weight, so the GEMM's per-call weight pack
     * is amortised, but at most 32 Mi (64 MiB of bf16 the buffer keeps after the call). */
    int64_t band = Ho;
    if (!direct) {
        const int64_t want = std::min<int64_t>((int64_t)1 << 25,
                                               std::max<int64_t>((int64_t)1 << 23, 8 * cout * KK));
        band = std::max<int64_t>(1, want / std::max<int64_t>(1, Wo * KK));
    }
    if (!y_dense) band = 1;
    band = std::min(band, Ho);
    if (!direct) pbuf.resize((size_t)(band * Wo * KK) * ael + 64);
    if (narrow) {
        ybuf.resize((size_t)(band * Wo * cout) + 16);
        bbuf.resize((size_t)cout + 16);
        for (int64_t co = 0; co < cout; ++co) bbuf[(size_t)co] = ldt(B, co);
    }
    char* pp = (char*)pbuf.data();
    float* yp = ybuf.data();
    const float* bp = bbuf.data();
    for (int64_t oy0 = 0; oy0 < Ho; oy0 += band) {
        const int64_t nb = std::min(band, Ho - oy0), P = nb * Wo;
        RadTensor at{};
        at.dtype = adt;
        at.rank = 2;
        at.shape[0] = P; at.shape[1] = KK;
        at.stride[0] = KK; at.stride[1] = 1;
        if (direct) {
            at.data = (char*)X->data + (size_t)(oy0 * xr) * ael;
        } else {
            at.data = pp;
            AVX_PARALLEL_FOR_IF(P * KK >= AVX_PAR_MIN)
            for (int64_t p = 0; p < P; ++p) {
                const int64_t oy = oy0 + p / Wo, ox = p % Wo;
                char* dst = pp + (size_t)(p * KK) * ael;
                for (int64_t kh = 0; kh < k; ++kh) {
                    const int64_t iy = oy * stride + kh - pt;
                    for (int64_t kw = 0; kw < k; ++kw) {
                        const int64_t ix = ox * stride + kw - pl;
                        char* d = dst + (size_t)((kh * k + kw) * cin) * ael;
                        if (iy < 0 || iy >= Hv || ix < 0 || ix >= Wv) {
                            std::memset(d, 0, (size_t)cin * ael);
                            continue;
                        }
                        const int64_t xo = (iy / up) * xr + (ix / up) * cin;
                        if (xb) std::memcpy(d, (const uint16_t*)X->data + xo, (size_t)cin * 2);
                        else    row_to_f32(X->dtype, byte_at(X->data, X->dtype, xo), (float*)d, cin);
                    }
                }
            }
        }
        RadTensor yt{};
        yt.rank = 2;
        if (narrow) {
            /* y^T [cout, P] = W A^T into f32 pixel-major rows, then the bias and the rounding. */
            yt.data = yp;
            yt.dtype = RAD_F32;
            yt.shape[0] = cout; yt.shape[1] = P;
            yt.stride[0] = 1; yt.stride[1] = cout;
            const int st = gemm_nt_into(&wt, &at, nullptr, &yt, cout, P, KK);
            if (st < 0) return st;
            AVX_PARALLEL_FOR_IF(P * cout >= AVX_PAR_MIN)
            for (int64_t p = 0; p < P; ++p) {
                float* row = yp + p * cout;
                add_rows(row, bp, cout);
                const int64_t yo = (oy0 + p / Wo) * yr + (p % Wo) * cout;
                row_from_f32(Y->dtype, row, (void*)byte_at(Y->data, Y->dtype, yo), cout);
            }
        } else {
            yt.data = (void*)byte_at(Y->data, Y->dtype, oy0 * yr);
            yt.dtype = Y->dtype;
            yt.shape[0] = P; yt.shape[1] = cout;
            yt.stride[0] = cout; yt.stride[1] = 1;
            const int st = gemm_nt_into(&at, &wt, B, &yt, P, cout, KK);
            if (st < 0) return st;
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ l2norm_chan */
/* n1 = bf16(x / max(||x||, 1e-12)), n2 = bf16(n1 * sqrt(n)), y = bf16(n2 * gamma), then `act`
 * none or silu (bf16(y * sigmoid(y))), the roundings in y's dtype. */
AVX_KERNEL(l2norm_chan) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || numel(G) < n) return RAD_E_SHAPE;
    const char* act = p_str(a, "act", "none");
    const bool silu = act && !std::strcmp(act, "silu");
    if (!silu && act && std::strcmp(act, "none")) return RAD_E_INVAL;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;
    const float sq = (float)std::sqrt((double)n);
    const uint32_t dt = Y->dtype;
    /* gamma read at linear offsets, as libref reads its [C, 1, 1(, 1)] weight. */
    float* gam = (float*)scratch_raw((size_t)(n + 16) * sizeof(float));
    for (int64_t c = 0; c < n; ++c) gam[c] = ld_elem(G->data, G->dtype, c);

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(3 * n + 48));
        float *xs = buf, *v = buf + n + 16, *tmp = v + n + 16;
        const float* x = in_row(X, rowoff(X, r, n), n, xs);
        const float nrm = std::max(std::sqrt(row_sumsq(x, n)), 1e-12f);
        const vf vn = vf_set1(nrm), vq = vf_set1(sq);
        int64_t c = 0;
        for (; c + VF_N <= n; c += VF_N) vf_storeu(v + c, vf_div(vf_loadu(x + c), vn));
        for (; c < n; ++c) v[c] = x[c] / nrm;
        narrow_row(dt, v, n, tmp);
        for (c = 0; c + VF_N <= n; c += VF_N) vf_storeu(v + c, vf_mul(vf_loadu(v + c), vq));
        for (; c < n; ++c) v[c] *= sq;
        narrow_row(dt, v, n, tmp);
        mul_rows(v, gam, n);
        narrow_row(dt, v, n, tmp);
        if (silu) {
            const vf one = vf_set1(1.0f);
            for (c = 0; c + VF_N <= n; c += VF_N) {
                const vf y = vf_loadu(v + c);
                vf_storeu(v + c, vf_div(y, vf_add(one, vf_exp(vf_sub(vf_zero(), y)))));
            }
            for (; c < n; ++c) v[c] = v[c] / (1.0f + sc_exp(-v[c]));
        }
        out_row(Y, rowoff(Y, r, n), n, v);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ d2s_add / s2d_avg_add */
/* d2s_add: y [2H, 2W * cout] += x [H, W * cin] repeated and dealt into 2x2 blocks,
 * y[2r+i, 2c+j, o] += x[r, c, (o F + off + 2i + j) / rep] (docs/OPS.md). Output row by output row:
 * the input row and the output row in f32, each output pixel's channels a gather of its input
 * pixel's through a per-(i, j) channel map, which is the identity when cin == cout. */
AVX_KERNEL(d2s_add) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0), ft = p_int(a, "ft", 1);
    if (cin <= 0 || cout <= 0 || (ft != 1 && ft != 2)) return RAD_E_SHAPE;
    if (X->rank != 2 || Y->rank != 2 || X->shape[1] % cin || Y->shape[1] % cout) return RAD_E_SHAPE;
    const int64_t H = X->shape[0], W = X->shape[1] / cin;
    if (H <= 0 || W <= 0) return RAD_E_SHAPE;
    if (Y->shape[0] != 2 * H || Y->shape[1] / cout != 2 * W) return RAD_E_SHAPE;
    const int64_t F = 4 * ft, off = ft == 2 ? 4 : 0;
    if ((cout * F) % cin) return RAD_E_SHAPE;
    const int64_t rep = cout * F / cin;
    const int64_t xr = X->stride[0], yr = Y->stride[0];
    int32_t* map = (int32_t*)scratch_raw((size_t)(4 * cout + 16) * sizeof(int32_t));
    bool ident = true;
    for (int64_t ij = 0; ij < 4; ++ij)
        for (int64_t o = 0; o < cout; ++o) {
            map[ij * cout + o] = (int32_t)((o * F + off + ij) / rep);
            ident = ident && map[ij * cout + o] == o;
        }
    const int64_t xn = W * cin, yn = 2 * W * cout;

    AVX_PARALLEL_FOR_IF(2 * H * yn >= AVX_PAR_MIN)
    for (int64_t oy = 0; oy < 2 * H; ++oy) {
        float* buf = scratch_f32((size_t)(xn + yn + 32));
        float *xs = buf, *ys = buf + xn + 16;
        const float* xrow = in_row(X, (oy / 2) * xr, xn, xs);
        const float* yrow = in_row(Y, oy * yr, yn, ys);
        if (yrow != ys) std::memcpy(ys, yrow, (size_t)yn * sizeof(float));
        const int64_t i = oy & 1;
        for (int64_t ox = 0; ox < 2 * W; ++ox) {
            const float* xp = xrow + (ox / 2) * cin;
            float* yp = ys + ox * cout;
            if (ident) {
                add_rows(yp, xp, cout);
                continue;
            }
            const int32_t* m = map + (2 * i + (ox & 1)) * cout;
            for (int64_t o = 0; o < cout; ++o) yp[o] += xp[m[o]];
        }
        out_row(Y, oy * yr, yn, ys);
    }
    return RAD_OK;
}

/* s2d_avg_add: y [H, W * cout] += the mean of each output channel's gs = cin F / cout dealt
 * values of x [2H, 2W * cin]; a dealt channel (c, t, i, j) is c F + 4t + 2i + j, and with ft 2
 * the frame t = 0 is the zero frame the mean still counts. Each sum runs in libref's order, over
 * the two input rows of an output row widened once. */
AVX_KERNEL(s2d_avg_add) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0), ft = p_int(a, "ft", 1);
    if (cin <= 0 || cout <= 0 || (ft != 1 && ft != 2)) return RAD_E_SHAPE;
    if (X->rank != 2 || Y->rank != 2 || X->shape[1] % cin || Y->shape[1] % cout) return RAD_E_SHAPE;
    const int64_t H = Y->shape[0], W = Y->shape[1] / cout;
    if (H <= 0 || W <= 0) return RAD_E_SHAPE;
    if (X->shape[0] != 2 * H || X->shape[1] / cin != 2 * W) return RAD_E_SHAPE;
    const int64_t F = 4 * ft;
    if ((cin * F) % cout) return RAD_E_SHAPE;
    const int64_t gs = cin * F / cout;
    const int64_t xr = X->stride[0], yr = Y->stride[0];
    const int64_t xn = 2 * W * cin, yn = W * cout;
    /* Each output channel's dealt values, as offsets into the two widened input rows (the second
     * row starts at xn); -1 marks a zero-frame value. */
    int32_t* rel = (int32_t*)scratch_raw((size_t)(cout * gs + 16) * sizeof(int32_t));
    for (int64_t o = 0; o < cout; ++o)
        for (int64_t gg = 0; gg < gs; ++gg) {
            const int64_t g = o * gs + gg;
            const int64_t c = g / F, rem = g % F, t = rem / 4, i = (rem % 4) / 2, j = rem % 2;
            rel[o * gs + gg] = (ft == 2 && t == 0) ? -1 : (int32_t)(i * xn + j * cin + c);
        }
    const float fgs = (float)gs;

    AVX_PARALLEL_FOR_IF(H * yn * gs >= AVX_PAR_MIN)
    for (int64_t oy = 0; oy < H; ++oy) {
        float* buf = scratch_f32((size_t)(2 * xn + 2 * yn + 64));
        float *xs = buf, *ys = buf + 2 * xn + 16, *tmp = ys + yn + 16;
        for (int64_t i = 0; i < 2; ++i) {
            const float* row = in_row(X, (2 * oy + i) * xr, xn, xs + i * xn);
            if (row != xs + i * xn) std::memcpy(xs + i * xn, row, (size_t)xn * sizeof(float));
        }
        for (int64_t ox = 0; ox < W; ++ox) {
            const float* xp = xs + 2 * ox * cin;
            for (int64_t o = 0; o < cout; ++o) {
                const int32_t* rr = rel + o * gs;
                float acc = 0.0f;
                for (int64_t gg = 0; gg < gs; ++gg)
                    if (rr[gg] >= 0) acc += xp[rr[gg]];
                tmp[ox * cout + o] = acc / fgs;
            }
        }
        narrow_row(Y->dtype, tmp, yn, ys);
        const float* yrow = in_row(Y, oy * yr, yn, ys);
        add_rows(tmp, yrow, yn);
        out_row(Y, oy * yr, yn, tmp);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ to_rgba8 */
/* v = clamp(x, -1, 1), u = clamp(bf16(bf16(v / 2) + 0.5), 0, 1), byte = round-half-even(u * 255),
 * as diffusers' postprocess makes them. A NaN pixel is 0, as libref's conversion leaves it. */
AVX_KERNEL(to_rgba8) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || rows_of(X, n) < rows || Y->dtype != RAD_U8) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(3 * n + 48));
        float *xs = buf, *u = buf + n + 16, *tmp = u + n + 16;
        const float* x = in_row(X, rowoff(X, r, n), n, xs);
        const vf lo = vf_set1(-1.0f), hi = vf_set1(1.0f), half = vf_set1(0.5f);
        int64_t c = 0;
        /* max(x, -1) takes -1 for a NaN lane, which ends at byte 0. */
        for (; c + VF_N <= n; c += VF_N)
            vf_storeu(u + c, vf_mul(vf_min(vf_max(vf_loadu(x + c), lo), hi), half));
        for (; c < n; ++c) {
            float v = x[c];
            v = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
            u[c] = v != v ? -0.5f : v * 0.5f;
        }
        narrow_row(RAD_BF16, u, n, tmp);
        for (c = 0; c + VF_N <= n; c += VF_N) vf_storeu(u + c, vf_add(vf_loadu(u + c), half));
        for (; c < n; ++c) u[c] += 0.5f;
        narrow_row(RAD_BF16, u, n, tmp);
        const vf zero = vf_zero(), k255 = vf_set1(255.0f);
        for (c = 0; c + VF_N <= n; c += VF_N)
            vf_storeu(u + c, vf_mul(vf_min(vf_max(vf_loadu(u + c), zero), hi), k255));
        for (; c < n; ++c) {
            const float v = u[c] < 0.0f ? 0.0f : (u[c] > 1.0f ? 1.0f : u[c]);
            u[c] = v * 255.0f;
        }
        const int64_t yb = rowoff(Y, r, n), ys = laststride(Y);
        uint8_t* out = (uint8_t*)Y->data;
        c = 0;
        if (ys == 1) {
#if AVX_LEVEL == 3
            for (; c + 16 <= n; c += 16)
                _mm_storeu_si128((__m128i*)(out + yb + c),
                                 _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(_mm512_loadu_ps(u + c))));
#elif AVX_LEVEL == 2
            for (; c + 8 <= n; c += 8) {
                const __m256i q = _mm256_cvtps_epi32(_mm256_loadu_ps(u + c));
                __m128i p = _mm_packus_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
                p = _mm_packus_epi16(p, p);
                _mm_storel_epi64((__m128i*)(out + yb + c), p);
            }
#endif
        }
        for (; c < n; ++c) out[yb + c * ys] = (uint8_t)std::nearbyintf(u[c]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gelu_tanh */
/* 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), tanh as (1 - e^(-2z)) / (1 + e^(-2z)) on
 * vf_exp, torch's form (avx_elementwise.cpp: act_mul says why that form). */
AVX_KERNEL(gelu_tanh) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(2 * n + 32));
        float *xs = buf, *y = buf + n + 16;
        const float* x = in_row(X, rowoff(X, r, n), n, xs);
        const vf one = vf_set1(1.0f), half = vf_set1(0.5f);
        const vf m2k0 = vf_set1(-2.0f * 0.7978845608028654f), k1 = vf_set1(0.044715f);
        int64_t c = 0;
        for (; c + VF_N <= n; c += VF_N) {
            const vf g = vf_loadu(x + c);
            const vf inner = vf_fma(vf_mul(k1, g), vf_mul(g, g), g);
            const vf e = vf_exp(vf_mul(m2k0, inner));
            const vf t = vf_div(vf_sub(one, e), vf_add(one, e));
            vf_storeu(y + c, vf_mul(vf_mul(half, g), vf_add(one, t)));
        }
        for (; c < n; ++c) {
            const float g = x[c];
            y[c] = 0.5f * g * (1.0f + sc_tanh(0.7978845608028654f * (g + 0.044715f * g * g * g)));
        }
        out_row(Y, rowoff(Y, r, n), n, y);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gelu_quick */
/* CLIP's quick_gelu in libref's three roundings: a = rd(1.702 x), s = rd(sigmoid(a)),
 * y = rd(x s), libm's exp, element by element: a text encoder's feed-forward, small beside its
 * GEMMs. */
AVX_KERNEL(gelu_quick) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;
    const uint32_t dt = Y->dtype;
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, n), xs = laststride(X);
        const int64_t yb = rowoff(Y, r, n), ys = laststride(Y);
        for (int64_t c = 0; c < n; ++c) {
            const float v = ldt(X, xb + c * xs);
            const float t = narrow1(dt, 1.702f * v);
            const float s = narrow1(dt, 1.0f / (1.0f + std::exp(-t)));
            stt(Y, yb + c * ys, v * s);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ scatter_add_rows */
/* x[idx[i]] = x[idx[i]] + v[i], rounded to x's dtype; a negative index adds nothing. Every index
 * is checked before the first write. Rows run in parallel only when no destination repeats: a
 * repeated one adds in index order, as libref adds it. */
AVX_KERNEL(scatter_add_rows) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *V = &a->t[0], *I = &a->t[1], *X = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t M = rows_of(V, n), dst_rows = rows_of(X, n);
    if (M <= 0 || numel(I) < M) return RAD_E_SHAPE;
    for (int64_t m = 0; m < M; ++m)
        if (ldi(I, offlin(I, m)) >= dst_rows) return RAD_E_INVAL;
    uint8_t* seen = (uint8_t*)scratch_raw((size_t)dst_rows + 64);
    std::memset(seen, 0, (size_t)dst_rows);
    bool distinct = true;
    for (int64_t m = 0; m < M && distinct; ++m) {
        const int64_t r = ldi(I, offlin(I, m));
        if (r < 0) continue;
        distinct = !seen[r];
        seen[r] = 1;
    }

    auto add_row = [&](int64_t m) {
        const int64_t r = ldi(I, offlin(I, m));
        if (r < 0) return;
        float* buf = scratch_f32((size_t)(2 * n + 32));
        float *vs = buf, *xs = buf + n + 16;
        const float* v = in_row(V, rowoff(V, m, n), n, vs);
        const float* x = in_row(X, rowoff(X, r, n), n, xs);
        if (x != xs) std::memcpy(xs, x, (size_t)n * sizeof(float));
        add_rows(xs, v, n);
        out_row(X, rowoff(X, r, n), n, xs);
    };
    if (!distinct) {
        for (int64_t m = 0; m < M; ++m) add_row(m);
        return RAD_OK;
    }
    AVX_PARALLEL_FOR_IF(M * n >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) add_row(m);
    return RAD_OK;
}

/* ------------------------------------------------------------------ table_add */
/* libref's table_add: y[r, l n + c] = x[r, c] + t[l, c], f32 sum rounded once to y's dtype. */
AVX_KERNEL(table_add) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *T = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", 0), L = p_int(a, "L", 0);
    if (n <= 0 || L <= 0 || numel(T) < L * n) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, L * n);
    if (rows <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(rows * L * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(3 * n + 64));
        float *xs = buf, *ts = buf + n + 16, *p = ts + n + 16;
        const float* xv = in_row(X, rowoff(X, r, n), n, xs);
        for (int64_t l = 0; l < L; ++l) {
            const float* tv = in_row(T, l * n * laststride(T), n, ts);
            for (int64_t c = 0; c < n; ++c) p[c] = xv[c] + tv[c];
            out_row(Y, rowoff(Y, r, L * n) + l * n * laststride(Y), n, p);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ mix_rows */
/* libref's mix_rows: y[r, c] = sum_i w[i] x[r k + i, c], f32 in ascending i, rounded once. */
AVX_KERNEL(mix_rows) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *W = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", 0), k = p_int(a, "k", 0);
    if (n <= 0 || k <= 0 || numel(W) < k) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || rows_of(X, n) < rows * k) return RAD_E_SHAPE;
    std::vector<float> w((size_t)k);
    for (int64_t i = 0; i < k; ++i) w[(size_t)i] = ldt(W, i * laststride(W));
    AVX_PARALLEL_FOR_IF(rows * n * k >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(2 * n + 64));
        float *xs = buf, *acc = buf + n + 16;
        for (int64_t c = 0; c < n; ++c) acc[c] = 0.0f;
        for (int64_t i = 0; i < k; ++i) {
            const float* xv = in_row(X, rowoff(X, r * k + i, n), n, xs);
            for (int64_t c = 0; c < n; ++c) acc[c] += w[(size_t)i] * xv[c];
        }
        out_row(Y, rowoff(Y, r, n), n, acc);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ chan_stats */
/* libref's chan_stats: per-channel statistics held as weights, every step rounded as libref. */
AVX_KERNEL(chan_stats) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
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
        mu[(size_t)c] = narrow1(dt, ldt(Mn, c * laststride(Mn)));
        const float v = ldt(Sd, c * laststride(Sd));
        sd[(size_t)c] = narrow1(dt, var ? std::sqrt(v + eps) : v);
    }
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(2 * n + 32));
        float *xs = buf, *o = buf + n + 16;
        const float* xv = in_row(X, rowoff(X, r, n), n, xs);
        for (int64_t c = 0; c < n; ++c)
            o[c] = norm ? narrow1(dt, narrow1(dt, xv[c] - mu[(size_t)c]) / sd[(size_t)c])
                        : narrow1(dt, narrow1(dt, xv[c] * sd[(size_t)c]) + mu[(size_t)c]);
        out_row(Y, rowoff(Y, r, n), n, o);
    }
    return RAD_OK;
}
