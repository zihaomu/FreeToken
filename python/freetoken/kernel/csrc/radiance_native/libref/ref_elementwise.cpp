/* ref_elementwise.cpp -- the norms and the elementwise ops.
 *
 * The subtle part is where fused forms round: `rmsnorm_add` is one pass in every real
 * implementation, so the reference must state which intermediates are rounded, or a correct fused
 * kernel fails its tolerance.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace ref;

/* ------------------------------------------------------------------ rmsnorm */
/* y = x * rsqrt(mean(x^2) + eps) * (w + w_add)
 *
 * The mean is over the last n elements of a row, accumulated in f32 in ascending index order.
 * `w_add` is GemmaRMSNorm's `1 +`: Qwen3.5, Qwen3-Next and Gemma store a zero-centred gain and use
 * 1 + w, and forming that in the weight's dtype first loses 0.3% per channel. Default 0. An absent
 * gain `w` is 1. */
static int rms_rows(const RadTensor* x, const RadTensor* w, const RadTensor* y, int64_t n,
                    float eps, float wadd, float mult) {
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(y) < rows * n || (w && numel(w) < n)) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), yb = rowoff(y, r, n);
        const int64_t xs = laststride(x), ys = laststride(y), ws = w ? laststride(w) : 0;
        float ss = 0.0f;
        for (int64_t i = 0; i < n; ++i) { float v = ldt(x, xb + i * xs); ss += v * v; }
        const float s = 1.0f / std::sqrt(ss / (float)n + eps);
        for (int64_t i = 0; i < n; ++i)
            stt(y, yb + i * ys,
                ldt(x, xb + i * xs) * s * (w ? ldt(w, i * ws) + wadd : 1.0f) * mult);
    }
    return RAD_OK;
}

int ref_rmsnorm(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor* w = a->t[1].data ? &a->t[1] : nullptr;
    const float wadd = p_f32(a, "wadd", 0.0f);
    /* `wadd` without a gain is refused. */
    if (!w && wadd != 0.0f) return RAD_E_INVAL;
    /* `mult` scales the normed row before its single rounding, for a residual branch weighted by
     * a constant (Gemma 4's audio feed-forward uses 0.5). */
    return rms_rows(&a->t[0], w, &a->t[2], p_int(a, "n", w && w->rank ? w->shape[w->rank - 1] : 0),
                    p_f32(a, "eps", 1e-6f), wadd, p_f32(a, "mult", 1.0f));
}

/* ------------------------------------------------------------------ rmsnorm_p */
/* y = x * rsqrt(mean(x^2) + eps): rmsnorm with no gain. A separate op because Gemma normalises the
 * value with a bare RMS (llama.cpp's ggml_rms_norm(V, eps)) and the checkpoint has no plane for a
 * gain; rmsnorm would need a phantom weight. */
int ref_rmsnorm_p(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    return rms_rows(&a->t[0], nullptr, &a->t[1], p_int(a, "n", 0), p_f32(a, "eps", 1e-6f), 0.0f,
                    1.0f);
}

/* ------------------------------------------------------------------ rmsnorm_add */
/* residual_out = x + residual;  y = rmsnorm(residual_out) * (w + w_add)
 *
 * The rounding is the contract: the variance comes from the unrounded f32 sum, but y reads the sum
 * back from residual_out after it is narrowed to that dtype. vLLM's fused_add_rms_norm and every
 * fused kernel here do the same, since the residual is stored anyway. Keeping f32 throughout would
 * be half a ULP better and disagree with every kernel.
 */
int ref_rmsnorm_add(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *res = &a->t[1], *w = &a->t[2], *y = &a->t[3], *ro = &a->t[4];
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(y) < rows * n || numel(ro) < rows * n ||
        numel(res) < rows * n || numel(w) < n) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), rb = rowoff(res, r, n);
        const int64_t yb = rowoff(y, r, n), ob = rowoff(ro, r, n);
        const int64_t xs = laststride(x), rs = laststride(res);
        const int64_t ys = laststride(y), os = laststride(ro), ws = laststride(w);
        float ss = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            float v = ldt(x, xb + i * xs) + ldt(res, rb + i * rs);
            ss += v * v;
            stt(ro, ob + i * os, v);
        }
        const float s = 1.0f / std::sqrt(ss / (float)n + eps);
        for (int64_t i = 0; i < n; ++i)
            stt(y, yb + i * ys, ldt(ro, ob + i * os) * s * (ldt(w, i * ws) + wadd));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ layernorm */
/* y = (x - mean) * rsqrt(var + eps) * w + b, with the biased variance (divide by n, not n-1), as
 * transformers use. `b` is optional: a bias-free ViT passes a null operand. */
int ref_layernorm(const RadArgs* a, RadStream) {
    if (a->n_t < 4 || !a->t[0].data || !a->t[1].data || !a->t[3].data) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *y = &a->t[3];
    const RadTensor* b = t_in(a, 2);
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps = p_f32(a, "eps", 1e-5f);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(y) < rows * n || numel(w) < n) return RAD_E_SHAPE;
    if (b && numel(b) < n) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), yb = rowoff(y, r, n);
        const int64_t xs = laststride(x), ys = laststride(y), ws = laststride(w);
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) sum += ldt(x, xb + i * xs);
        const float mean = sum / (float)n;
        float ss = 0.0f;
        for (int64_t i = 0; i < n; ++i) { float d = ldt(x, xb + i * xs) - mean; ss += d * d; }
        const float s = 1.0f / std::sqrt(ss / (float)n + eps);
        for (int64_t i = 0; i < n; ++i) {
            float v = (ldt(x, xb + i * xs) - mean) * s * ldt(w, i * ws);
            if (b) v += ldt(b, i * laststride(b));
            stt(y, yb + i * ys, v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ grid_embed */
/* A learned side x side table of position rows, resampled per row of x to that row's grid size and
 * added in: the vision tower's position embedding, trained at one grid and applied to any image
 * shape.
 *
 * Bilinear with corners aligned: grid row r of height g sits at table coordinate
 * r * (side - 1) / max(g - 1, 1) in f32, with taps at floor and floor + 1, clamped and weighted by
 * distance. The four products are summed in f32 in the order (top-left, top-right, bottom-left,
 * bottom-right), rounded to x's dtype as a tensor of its own, then added to x and rounded again. */
int ref_grid_embed(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *tab = &a->t[1], *crd = &a->t[2];
    const int64_t n = p_int(a, "n", tab->rank ? tab->shape[tab->rank - 1] : 0);
    const int64_t side = p_int(a, "side", 0);
    if (n <= 0 || side <= 0 || numel(tab) < side * side * n) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < rows)
        return RAD_E_SHAPE;
    const int64_t cs = crd->stride[0], ms = crd->stride[1];
    const int64_t ts = laststride(tab);

    REF_PAR_FOR(rows * n * 4)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t gr = ld_int(crd->dtype, crd->data, 0 * cs + r * ms);
        const int64_t gc = ld_int(crd->dtype, crd->data, 1 * cs + r * ms);
        const int64_t gh = ld_int(crd->dtype, crd->data, 2 * cs + r * ms);
        const int64_t gw = ld_int(crd->dtype, crd->data, 3 * cs + r * ms);
        int64_t th[2], tw[2];
        float wh[2], ww[2];
        auto axis = [&](int64_t idx, int64_t size, int64_t* t, float* w) {
            const float src = (float)idx * (float)(side - 1) / (float)(size - 1 > 1 ? size - 1 : 1);
            const float fl = std::floor(src);
            for (int k = 0; k < 2; ++k) {
                int64_t tap = (int64_t)fl + k;
                t[k] = tap < 0 ? 0 : (tap > side - 1 ? side - 1 : tap);
                const float d = std::fabs(src - fl - (float)k);
                w[k] = 1.0f - d > 0.0f ? 1.0f - d : 0.0f;
            }
        };
        axis(gr, gh, th, wh);
        axis(gc, gw, tw, ww);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int ki = 0; ki < 2; ++ki)
                for (int kj = 0; kj < 2; ++kj)
                    acc += ldt(tab, rowoff(tab, th[ki] * side + tw[kj], n) + i * ts) *
                           (wh[ki] * ww[kj]);
            const float pe = round_dt(x->dtype, acc);
            stt(x, xb + i * xs, ldt(x, xb + i * xs) + pe);
        }
    }
    return RAD_OK;
}

/* grid_embed_aa: the table resampled as torch's F.interpolate(mode="bilinear", align_corners=False,
 * antialias=True), SigLIP2's position embedding. Per axis, a triangle filter on half-pixel centres,
 * widened by the scale when shrinking, with in-table taps normalised to sum to one. Separable, f32
 * throughout, rounded once before the add. Out-of-grid coordinates are clamped. */
static void aa_taps(int64_t at, int64_t size, int64_t side, float* w, int64_t* lo, int64_t* cnt) {
    if (size < 1) size = 1;
    at = at < 0 ? 0 : (at >= size ? size - 1 : at);
    const float scale = (float)side / (float)size;
    const float support = scale > 1.0f ? scale : 1.0f, inv = 1.0f / support;
    const float center = scale * ((float)at + 0.5f);
    int64_t a = (int64_t)(center - support + 0.5f), b = (int64_t)(center + support + 0.5f);
    if (a < 0) a = 0;
    if (b > side) b = side;
    float total = 0.0f;
    for (int64_t j = a; j < b; ++j) {
        const float d = std::fabs(((float)j - center + 0.5f) * inv);
        w[j - a] = d < 1.0f ? 1.0f - d : 0.0f;
        total += w[j - a];
    }
    for (int64_t j = a; j < b; ++j) w[j - a] = total != 0.0f ? w[j - a] / total : 0.0f;
    *lo = a;
    *cnt = b > a ? b - a : 0;
}

int ref_grid_embed_aa(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *tab = &a->t[1], *crd = &a->t[2];
    const int64_t n = p_int(a, "n", tab->rank ? tab->shape[tab->rank - 1] : 0);
    const int64_t side = p_int(a, "side", 0);
    if (n <= 0 || side <= 0 || numel(tab) < side * side * n) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < rows)
        return RAD_E_SHAPE;
    const int64_t cs = crd->stride[0], ms = crd->stride[1];
    const int64_t ts = laststride(tab);

    REF_PAR_FOR(rows * n * 16)
    for (int64_t r = 0; r < rows; ++r) {
        int64_t g[4];
        for (int c = 0; c < 4; ++c) g[c] = ld_int(crd->dtype, crd->data, c * cs + r * ms);
        std::vector<float> wh((size_t)side), ww((size_t)side);
        int64_t h0 = 0, nh = 0, w0 = 0, nw = 0;
        aa_taps(g[0], g[2], side, wh.data(), &h0, &nh);
        aa_taps(g[1], g[3], side, ww.data(), &w0, &nw);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int64_t y = 0; y < nh; ++y) {
                float row = 0.0f;
                for (int64_t c = 0; c < nw; ++c)
                    row += ldt(tab, rowoff(tab, (h0 + y) * side + w0 + c, n) + i * ts) *
                           ww[(size_t)c];
                acc += row * wh[(size_t)y];
            }
            const float pe = round_dt(x->dtype, acc);
            stt(x, xb + i * xs, ldt(x, xb + i * xs) + pe);
        }
    }
    return RAD_OK;
}

/* grid_embed_cpe: RADIO's cropped position embedding (CPE). The table resampled as torch's
 * F.interpolate(mode="bilinear", align_corners=False) -- without antialias -- onto a square of side
 * d = max(grid height, grid width), of which the grid keeps the top-left h x w. Per axis two taps at
 * src = (i + 0.5) * side / d - 0.5, clamped at 0, the upper tap clamped into the table, and blended
 * along the row first as torch's bilinear kernel does. f32, rounded once before the add. */
static void cpe_taps(int64_t at, int64_t d, int64_t side, int64_t t[2], float w[2]) {
    if (d < 1) d = 1;
    at = at < 0 ? 0 : (at >= d ? d - 1 : at);
    float src = (float)side / (float)d * ((float)at + 0.5f) - 0.5f;
    if (src < 0.0f) src = 0.0f;
    int64_t i0 = (int64_t)src;
    if (i0 > side - 1) i0 = side - 1;
    t[0] = i0;
    t[1] = i0 < side - 1 ? i0 + 1 : i0;
    w[1] = src - (float)i0;
    w[0] = 1.0f - w[1];
}

int ref_grid_embed_cpe(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *tab = &a->t[1], *crd = &a->t[2];
    const int64_t n = p_int(a, "n", tab->rank ? tab->shape[tab->rank - 1] : 0);
    const int64_t side = p_int(a, "side", 0);
    if (n <= 0 || side <= 0 || numel(tab) < side * side * n) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < rows)
        return RAD_E_SHAPE;
    const int64_t cs = crd->stride[0], ms = crd->stride[1];
    const int64_t ts = laststride(tab);
    /* The prefix rows, when n_prefix names some; without it the operand is not read. */
    const int64_t npre = p_int(a, "n_prefix", 0);
    const RadTensor* pre = npre > 0 && a->n_t > 3 && a->t[3].data ? &a->t[3] : nullptr;
    if (npre > 0 && (!pre || numel(pre) < npre * n)) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n * 4)
    for (int64_t r = 0; r < rows; ++r) {
        int64_t g[4];
        for (int c = 0; c < 4; ++c) g[c] = ld_int(crd->dtype, crd->data, c * cs + r * ms);
        const int64_t xb0 = rowoff(x, r, n), xs0 = laststride(x);
        if (g[0] < 0) {
            /* A row ahead of its segment's patches: the learned row its column names. */
            const int64_t k = g[1] < 0 ? 0 : (g[1] >= npre ? npre - 1 : g[1]);
            for (int64_t i = 0; npre > 0 && i < n; ++i) {
                const float pe = round_dt(x->dtype, ldt(pre, rowoff(pre, k, n) + i * laststride(pre)));
                stt(x, xb0 + i * xs0, ldt(x, xb0 + i * xs0) + pe);
            }
            continue;
        }
        const int64_t d = g[2] > g[3] ? g[2] : g[3];
        int64_t th[2], tw[2];
        float wh[2], ww[2];
        cpe_taps(g[0], d, side, th, wh);
        cpe_taps(g[1], d, side, tw, ww);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        auto at = [&](int64_t y, int64_t c, int64_t i) {
            return ldt(tab, rowoff(tab, y * side + c, n) + i * ts);
        };
        for (int64_t i = 0; i < n; ++i) {
            const float r0 = ww[0] * at(th[0], tw[0], i) + ww[1] * at(th[0], tw[1], i);
            const float r1 = ww[0] * at(th[1], tw[0], i) + ww[1] * at(th[1], tw[1], i);
            const float pe = round_dt(x->dtype, wh[0] * r0 + wh[1] * r1);
            stt(x, xb + i * xs, ldt(x, xb + i * xs) + pe);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ grid_embed_pad */
/* The same learned table and operands as grid_embed's, resampled in the half-pixel-centre form
 * (align_corners=False) with grid_sample's padding_mode="zeros": taps whose UNCLAMPED index falls
 * outside [0, side-1) contribute nothing rather than reading the edge. Per axis,
 * src = (i + 0.5) * side / size - 0.5; floor(src) and floor(src) + 1, each carrying the linear hat
 * weight (1 - |src - tap|)+ masked to zero off the table. For grids no larger than the table the
 * mask never fires and the op agrees with grid_embed_aa's centre positions (not grid_embed's
 * corners-aligned ones); it differs from them only when a grid upsamples past the table, which is
 * the common case for Muse. The four products are summed in f32 in grid_embed's order (top-left,
 * top-right, bottom-left, bottom-right), rounded to x's dtype, then added and rounded again. */
int ref_grid_embed_pad(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *tab = &a->t[1], *crd = &a->t[2];
    const int64_t n = p_int(a, "n", tab->rank ? tab->shape[tab->rank - 1] : 0);
    const int64_t side = p_int(a, "side", 0);
    if (n <= 0 || side <= 0 || numel(tab) < side * side * n) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < rows)
        return RAD_E_SHAPE;
    const int64_t cs = crd->stride[0], ms = crd->stride[1];
    const int64_t ts = laststride(tab);

    REF_PAR_FOR(rows * n * 4)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t gr = ld_int(crd->dtype, crd->data, 0 * cs + r * ms);
        const int64_t gc = ld_int(crd->dtype, crd->data, 1 * cs + r * ms);
        const int64_t gh = ld_int(crd->dtype, crd->data, 2 * cs + r * ms);
        const int64_t gw = ld_int(crd->dtype, crd->data, 3 * cs + r * ms);
        int64_t th[2], tw[2];
        float wh[2], ww[2];
        auto axis = [&](int64_t idx, int64_t size, int64_t* t, float* w) {
            if (size < 1) size = 1;
            const float src = (float)side / (float)size * ((float)idx + 0.5f) - 0.5f;
            const float fl = std::floor(src);
            for (int k = 0; k < 2; ++k) {
                const int64_t raw = (int64_t)fl + k;
                t[k] = raw < 0 ? 0 : (raw > side - 1 ? side - 1 : raw);
                const float d = std::fabs(src - fl - (float)k);
                const float u = 1.0f - d > 0.0f ? 1.0f - d : 0.0f;
                w[k] = raw >= 0 && raw <= side - 1 ? u : 0.0f;
            }
        };
        axis(gr, gh, th, wh);
        axis(gc, gw, tw, ww);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int ki = 0; ki < 2; ++ki)
                for (int kj = 0; kj < 2; ++kj)
                    acc += ldt(tab, rowoff(tab, th[ki] * side + tw[kj], n) + i * ts) *
                           (wh[ki] * ww[kj]);
            const float pe = round_dt(x->dtype, acc);
            stt(x, xb + i * xs, ldt(x, xb + i * xs) + pe);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ window_order */
/* The window-attention permutation of one encoder pass, computed entirely from the on-device
 * coordinate planes: HF's get_vision_window_index as a gather-index computation. Windows partition
 * each segment (one segment == one frame): window row = row/window, window col = col/window,
 * windows scanned window-row-major and ragged at the segment's right and bottom edges; within a
 * window patches keep raster order, and windows never span segments.
 *
 * `crd` is [4, P] i32 -- row, column, segment height, segment width per patch -- and `cu` is [S+1]
 * i32, the per-frame segment boundaries in patches. Outputs, all i32: `idx` [P], slot -> source
 * patch (the gather index); `inv` [P], source patch -> slot; `pos` [2, P], each slot's rope
 * positions (col + 1, row + 1) -- Muse's rotary wants the axes swapped and offset by one relative
 * to the coordinate planes (HF's position_ids.flip(-1) + 1) -- laid out component-major like the
 * planes; and `cuw` [P+1], the cumulative window boundaries in the permuted order, n_windows + 1
 * entries used, the rest holding the total. `M` is the patches this pass covers, 1..P: slots past
 * M are the identity, so a caller may reuse buffers sized for a bigger pass. `window` <= 0 makes
 * the permutation the identity and cuw = cu. */
int ref_window_order(const RadArgs* a, RadStream) {
    if (!have(a, 6)) return RAD_E_INVAL;
    const RadTensor *crd = &a->t[0], *cu = &a->t[1];
    RadTensor *idx = const_cast<RadTensor*>(&a->t[2]), *inv = const_cast<RadTensor*>(&a->t[3]),
              *pos = const_cast<RadTensor*>(&a->t[4]), *cuw = const_cast<RadTensor*>(&a->t[5]);
    const int64_t M = p_int(a, "M", 0);
    const int64_t win = p_int(a, "window", 0);
    if (M <= 0 || crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < M) return RAD_E_SHAPE;
    if (numel(cu) < 2 || numel(idx) < M || numel(inv) < M || numel(pos) < 2 * M ||
        numel(cuw) < M + 1)
        return RAD_E_SHAPE;
    const int64_t P = crd->shape[1];
    const int64_t cs = crd->stride[0], ms = crd->stride[1];
    auto gx = [&](int c, int64_t p) {
        return ld_int(crd->dtype, crd->data, (int64_t)c * cs + p * ms);
    };

    /* The identity: every slot its own patch, cuw = cu, the tail padded with the last boundary. */
    auto identity = [&]() {
        for (int64_t p = 0; p < P; ++p) {
            st_int(idx->dtype, idx->data, p, p);
            st_int(inv->dtype, inv->data, p, p);
            if (p < M) {
                st_int(pos->dtype, pos->data, p, gx(1, p) + 1);
                st_int(pos->dtype, pos->data, P + p, gx(0, p) + 1);
            } else {
                st_int(pos->dtype, pos->data, p, 1);
                st_int(pos->dtype, pos->data, P + p, 1);
            }
        }
        int64_t last = 0;
        const int64_t k = numel(cu) < M + 1 ? numel(cu) : M + 1;
        for (int64_t i = 0; i < k; ++i) {
            last = ld_int(cu->dtype, cu->data, i);
            st_int(cuw->dtype, cuw->data, i, last);
        }
        for (int64_t i = k; i < M + 1; ++i) st_int(cuw->dtype, cuw->data, i, last);
    };
    if (win <= 0) {
        identity();
        return RAD_OK;
    }

    /* Slot arithmetic: patch p's segment, its window and its rank in the window follow from the
     * coordinates alone, so a first walk writes the window boundaries and a second scatters. */
    std::vector<int64_t> seg(M), seg_nw, seg_first_win;
    std::vector<int64_t> wbase((size_t)M);      /* each window's base slot; windows <= M */
    int64_t s = 0, slot = 0, nwin = 0;
    const int64_t S = numel(cu) - 1;
    while (s < S) {
        const int64_t start = ld_int(cu->dtype, cu->data, s);
        if (start >= M) break;
        int64_t end = ld_int(cu->dtype, cu->data, s + 1);
        if (end > M) end = M;
        const int64_t gh = start < end ? gx(2, start) : 1;
        const int64_t gw = start < end ? gx(3, start) : 1;
        const int64_t nh = (gh + win - 1) / win, nw = (gw + win - 1) / win;
        seg_first_win.push_back(nwin);
        seg_nw.push_back(nw);
        for (int64_t p = start; p < end; ++p) seg[p] = (int64_t)seg_first_win.size() - 1;
        for (int64_t wr = 0; wr < nh; ++wr)
            for (int64_t wc = 0; wc < nw; ++wc) {
                const int64_t rows_in = std::min(win, gh - wr * win);
                const int64_t cols_in = std::min(win, gw - wc * win);
                st_int(cuw->dtype, cuw->data, nwin, slot);
                wbase[(size_t)nwin] = slot;
                ++nwin;
                slot += rows_in * cols_in;
            }
        ++s;
    }
    st_int(cuw->dtype, cuw->data, nwin, slot);
    for (int64_t i = nwin + 1; i < M + 1; ++i) st_int(cuw->dtype, cuw->data, i, slot);

    for (int64_t p = 0; p < P; ++p) {
        if (p >= M) {
            st_int(idx->dtype, idx->data, p, p);
            st_int(inv->dtype, inv->data, p, p);
            st_int(pos->dtype, pos->data, p, 1);
            st_int(pos->dtype, pos->data, P + p, 1);
            continue;
        }
        const int64_t r = gx(0, p), c = gx(1, p);
        const int64_t gw = gx(3, p);
        int64_t wr = r / win, wc = c / win;
        if (wr < 0) wr = 0;
        if (wc < 0) wc = 0;
        const int64_t cols_in = std::min(win, gw - wc * win);
        const int64_t wid = seg_first_win[(size_t)seg[p]] + wr * seg_nw[(size_t)seg[p]] + wc;
        const int64_t at = wbase[(size_t)wid] + (r - wr * win) * cols_in + (c - wc * win);
        st_int(idx->dtype, idx->data, at, p);
        st_int(inv->dtype, inv->data, p, at);
        st_int(pos->dtype, pos->data, at, c + 1);
        st_int(pos->dtype, pos->data, P + at, r + 1);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ add / mul */
/* Elementwise over the output's extent. `b` may be a single row of exactly `n` elements, which
 * broadcasts along rows (a bias add); nothing else broadcasts, so a shape bug cannot pass as a
 * plausible number. `add` with `div` stores rd(rd(a + b) / div), rd y's dtype: the last step of a
 * mean summed pairwise, (b0 + b1 + b2) / 3 as a vocoder averages its residual blocks. */
static int elementwise2(const RadArgs* a, bool mul) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", numel(Y));
    const float div = mul ? 1.0f : p_f32(a, "div", 1.0f);
    if (!(div != 0.0f) || !std::isfinite(div)) return RAD_E_INVAL;
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(A) < rows * n) return RAD_E_SHAPE;
    const bool bcast = numel(B) == n && rows > 1;
    if (!bcast && numel(B) < rows * n) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t ab = rowoff(A, r, n), yb = rowoff(Y, r, n);
        const int64_t bb = bcast ? 0 : rowoff(B, r, n);
        const int64_t as = laststride(A), bs = laststride(B), ys = laststride(Y);
        for (int64_t i = 0; i < n; ++i) {
            float u = ldt(A, ab + i * as), v = ldt(B, bb + i * bs);
            float s = mul ? u * v : u + v;
            if (div != 1.0f) s = round_dt(Y->dtype, s) / div;
            stt(Y, yb + i * ys, s);
        }
    }
    return RAD_OK;
}

int ref_add(const RadArgs* a, RadStream) { return elementwise2(a, false); }
int ref_mul(const RadArgs* a, RadStream) { return elementwise2(a, true); }

/* ------------------------------------------------------------------ mul_weight */
/* y = x * w, `w` a weight of one element or one row: a learned gain such as a per-layer residual
 * scalar (Gemma 4's layer_scalar) or a per-channel layer scale (vision towers). Separate from `mul`
 * because the schema fixes whether an operand is a weight or an activation, and `mul`'s `b` is an
 * activation. The f32 product is rounded once to y's dtype. */
int ref_mul_weight(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *W = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(X) < rows * n) return RAD_E_SHAPE;
    const bool scalar = numel(W) == 1;
    if (!scalar && numel(W) != n) return RAD_E_SHAPE;
    const int64_t ws = scalar ? 0 : laststride(W);

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, n), yb = rowoff(Y, r, n);
        const int64_t xs = laststride(X), ys = laststride(Y);
        for (int64_t i = 0; i < n; ++i)
            stt(Y, yb + i * ys, ldt(X, xb + i * xs) * ldt(W, i * ws));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ scale_rows */
/* y[m, i] = x[m, i] * act(s[m]): one scalar per row, broadcast across it (a per-token gate).
 * Separate from `mul`, which broadcasts one row down the rows; one op with both rules would be
 * ambiguous when rows == n.
 *
 * The activation applies to the scalar, not the product: Qwen3.5-MoE's shared expert is scaled by
 * `sigmoid(shared_expert_gate(x))`.
 *
 * With `add` present it is the fused gate-and-add, y = narrow(x * act(s)) + add. The unfused pair
 * stores the product to a bf16 buffer before adding, so the product is rounded first; this writes
 * the same two stores through `y`. `add` is read before `y` is written so the two may alias. */
int ref_scale_rows(const RadArgs* a, RadStream) {
    /* `add` (operand 2) is optional; the other three are required. */
    if (a->n_t < 4 || !t_in(a, 0) || !t_in(a, 1) || !t_in(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *sc = &a->t[1], *y = &a->t[3];
    const RadTensor *ad = t_in(a, 2);
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(x) < rows * n || numel(sc) < rows) return RAD_E_SHAPE;
    if (ad && numel(ad) < rows * n) return RAD_E_SHAPE;
    const char* act = p_str(a, "act", "none");
    const bool sig = act && !std::strcmp(act, "sigmoid");
    if (act && std::strcmp(act, "sigmoid") && std::strcmp(act, "none")) return RAD_E_INVAL;

    const int64_t ss = laststride(sc);
    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const float v0 = ldt(sc, r * ss);
        const float v  = sig ? act_sigmoid(v0) : v0;
        const int64_t xb = rowoff(x, r, n), yb = rowoff(y, r, n);
        const int64_t xs = laststride(x), ys = laststride(y);
        const int64_t ab = ad ? rowoff(ad, r, n) : 0;
        const int64_t as = ad ? laststride(ad) : 1;
        for (int64_t i = 0; i < n; ++i) {
            const float av = ad ? ldt(ad, ab + i * as) : 0.0f;
            stt(y, yb + i * ys, ldt(x, xb + i * xs) * v);
            if (ad) stt(y, yb + i * ys, ldt(y, yb + i * ys) + av);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ silu_mul */
/* y[.., i] = silu(gate_up[.., i]) * gate_up[.., n + i]. The gate is the first half, the order in
 * which rad-convert fuses gate and up. */
int ref_silu_mul(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *gu = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(gu) < rows * 2 * n) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t gb = rowoff(gu, r, 2 * n), yb = rowoff(y, r, n);
        const int64_t gs = laststride(gu), ys = laststride(y);
        for (int64_t i = 0; i < n; ++i)
            stt(y, yb + i * ys, act_silu(ldt(gu, gb + i * gs)) * ldt(gu, gb + (n + i) * gs));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ act_mul */
/* y = act(a) * b, the gated product with its halves in two operands: two column slices of a fused
 * gate_up, or a gate whose multiplier is another tensor (Gemma's per-layer-input block multiplies
 * gelu(gate(x)) by a per-layer embedding). With `relu` and the same tensor in both operands it is
 * relu(x)^2, Nemotron-H's ungated expert activation.
 *
 * The activation is rounded to y's dtype before the product, as the unfused pair stores it. */
int ref_act_mul(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(A) < rows * n || numel(B) < rows * n) return RAD_E_SHAPE;
    static const char* kActs[] = { "silu", "gelu", "gelu_tanh", "sigmoid", "quick_gelu", "relu" };
    const int act = p_enum(a, "act", kActs, 6, -1);
    if (act < 0) return RAD_E_INVAL;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t ab = rowoff(A, r, n), bb = rowoff(B, r, n), yb = rowoff(Y, r, n);
        const int64_t as = laststride(A), bs = laststride(B), ys = laststride(Y);
        for (int64_t i = 0; i < n; ++i) {
            const float g = ldt(A, ab + i * as);
            /* quick_gelu is x * sigmoid(1.702 x) with the scaled value rounded to y's dtype
             * first, exactly the unfused mul-then-act_mul pair this replaces. */
            const float v = act == 0 ? act_silu(g) : act == 1 ? act_gelu(g)
                          : act == 2 ? act_gelu_tanh(g)
                          : act == 3 ? act_sigmoid(g)
                          : act == 5 ? (g > 0.0f ? g : 0.0f)
                                     : act_sigmoid(round_dt(Y->dtype,
                                                            g * round_dt(RAD_BF16, 1.702f)));
            stt(Y, yb + i * ys, round_dt(Y->dtype, v) * ldt(B, bb + i * bs));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ seg_mean */
/* y[s, :] = mean of x's rows [cu[s], cu[s+1]): sentence-embedding mean pooling over a ragged
 * batch. Summed in double and divided once, so only the final rounding to y's dtype depends on the
 * segment length. An empty segment writes zeros. */
int ref_seg_mean(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *cu = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    const int64_t S = numel(cu) - 1;
    if (n <= 0 || S <= 0 || numel(Y) < S * n) return RAD_E_SHAPE;
    const int64_t total = cu_at(cu, S);
    if (total <= 0 || numel(X) < total * n) return RAD_E_SHAPE;

    REF_PAR_FOR((total + S) * n)
    for (int64_t s = 0; s < S; ++s) {
        const int64_t b0 = cu_at(cu, s);
        const int64_t b1 = cu_at(cu, s + 1);
        const int64_t yb = rowoff(Y, s, n), ys = laststride(Y), xs = laststride(X);
        for (int64_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (int64_t r = b0; r < b1; ++r) acc += ldt(X, rowoff(X, r, n) + i * xs);
            stt(Y, yb + i * ys, b1 > b0 ? (float)(acc / (double)(b1 - b0)) : 0.0f);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ seg_softmax_pool */
/* y[s, :] = sum over the segment's rows r of softmax(score)[r] * v[r, :]: one row a cu_seqlens
 * segment, the weights a softmax over the segment's own scores (f32, the maximum subtracted), the
 * sum in f32 and rounded once. A query reading its few candidates with learned scores (clef's
 * routing over a question's options). An empty segment writes zeros. */
int ref_seg_softmax_pool(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *Sc = &a->t[0], *V = &a->t[1], *cu = &a->t[2], *Y = &a->t[3];
    const int64_t n = p_int(a, "n", 0);
    const int64_t S = numel(cu) - 1;
    if (n <= 0 || S <= 0 || numel(Y) < S * n) return RAD_E_SHAPE;
    const int64_t total = cu_at(cu, S);
    if (total <= 0 || numel(V) < total * n || numel(Sc) < total) return RAD_E_SHAPE;

    REF_PAR_FOR((total + S) * n)
    for (int64_t s = 0; s < S; ++s) {
        const int64_t b0 = cu_at(cu, s);
        const int64_t b1 = cu_at(cu, s + 1);
        const int64_t yb = rowoff(Y, s, n), ys = laststride(Y), vs = laststride(V);
        float m = -INFINITY, z = 0.0f;
        for (int64_t r = b0; r < b1; ++r) m = std::max(m, ldt(Sc, rowoff(Sc, r, 1)));
        for (int64_t r = b0; r < b1; ++r) z += std::exp(ldt(Sc, rowoff(Sc, r, 1)) - m);
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int64_t r = b0; r < b1; ++r)
                acc += std::exp(ldt(Sc, rowoff(Sc, r, 1)) - m) / z *
                       ldt(V, rowoff(V, r, n) + i * vs);
            stt(Y, yb + i * ys, b1 > b0 ? acc : 0.0f);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ pool_rows */
/* Row r of y is the mean of x's rows [r*k, r*k + k), summed in f32 and rounded to y's dtype, then
 * multiplied by `mult` and rounded again, as the unfused pair stores the mean. For vision poolers
 * whose processor lays each k-patch window out consecutively. */
int ref_pool_rows(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), k = p_int(a, "k", 0);
    const float mult = p_f32(a, "mult", 1.0f);
    if (n <= 0 || k <= 0) return RAD_E_SHAPE;
    const int64_t rows_in = rows_of(X, n);
    if (rows_in <= 0 || rows_in % k) return RAD_E_SHAPE;
    const int64_t rows = rows_in / k;
    if (numel(Y) < rows * n) return RAD_E_SHAPE;
    const float inv = 1.0f / (float)k;

    REF_PAR_FOR(rows_in * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t yb = rowoff(Y, r, n), ys = laststride(Y), xs = laststride(X);
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int64_t j = 0; j < k; ++j) acc += ldt(X, rowoff(X, r * k + j, n) + i * xs);
            stt(Y, yb + i * ys, acc * inv);
            stt(Y, yb + i * ys, ldt(Y, yb + i * ys) * mult);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ pool_cols */
/* Column c of row m of y is the mean of x[m]'s columns [c*k, c*k + k), summed in f32 and rounded
 * to y's dtype, then multiplied by `mult` and rounded again: torch's adaptive_avg_pool1d over the
 * feature axis. */
int ref_pool_cols(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), k = p_int(a, "k", 0);
    const float mult = p_f32(a, "mult", 1.0f);
    if (n <= 0 || k <= 0 || n % k) return RAD_E_SHAPE;
    const int64_t M = rows_of(X, n), nw = n / k;
    if (M <= 0 || numel(Y) < M * nw) return RAD_E_SHAPE;
    const float inv = 1.0f / (float)k;

    REF_PAR_FOR(M * nw * k)
    for (int64_t r = 0; r < M; ++r) {
        const int64_t yb = rowoff(Y, r, nw), ys = laststride(Y), xs = laststride(X);
        for (int64_t c = 0; c < nw; ++c) {
            float acc = 0.0f;
            for (int64_t j = 0; j < k; ++j) acc += ldt(X, rowoff(X, r, n) + (c * k + j) * xs);
            stt(Y, yb + c * ys, acc * inv);
            stt(Y, yb + c * ys, ldt(Y, yb + c * ys) * mult);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ point_fourier */
/* A point's Fourier features: row m of y is [x | sin(x_i f_j) | cos(x_i f_j)] over x's n values,
 * the sines and the cosines each i-major then j, f_j = 2^j for j < freqs (times pi with `pi` 1),
 * so y has n (2 freqs + 1) columns. The products are f32 (exact without pi: a power of two), the
 * sine and cosine f32 and rounded once to y's dtype, as Hunyuan3D-2's shape VAE embeds a query
 * point. */
int ref_point_fourier(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), F = p_int(a, "freqs", 0);
    const bool pi = p_int(a, "pi", 0) != 0;
    if (n <= 0 || F <= 0 || F > 30) return RAD_E_SHAPE;
    const int64_t w = n * (2 * F + 1);
    const int64_t M = rows_of(X, n);
    if (M <= 0 || numel(Y) < M * w) return RAD_E_SHAPE;
    float f[30];
    for (int64_t j = 0; j < F; ++j)
        f[j] = pi ? (float)std::ldexp(3.14159265358979323846, (int)j) : std::ldexp(1.0f, (int)j);

    REF_PAR_FOR(M * w)
    for (int64_t r = 0; r < M; ++r) {
        const int64_t xb = rowoff(X, r, n), yb = rowoff(Y, r, w);
        const int64_t xs = laststride(X), ys = laststride(Y);
        for (int64_t i = 0; i < n; ++i) {
            const float v = ldt(X, xb + i * xs);
            stt(Y, yb + i * ys, v);
            for (int64_t j = 0; j < F; ++j) {
                const float t = v * f[j];
                stt(Y, yb + (n + i * F + j) * ys, std::sin(t));
                stt(Y, yb + (n + n * F + i * F + j) * ys, std::cos(t));
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ relu */
/* y = max(x, 0) over [M, n], each operand at its own row pitch. */
int ref_relu(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(X) < rows * n) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, n), yb = rowoff(Y, r, n);
        const int64_t xs = laststride(X), ys = laststride(Y);
        for (int64_t i = 0; i < n; ++i) {
            const float v = ldt(X, xb + i * xs);
            stt(Y, yb + i * ys, v > 0.0f ? v : 0.0f);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ clamp */
/* y = min(max(x, lo), hi), the bounds being one-element weights: a clipped linear's input or output
 * range, stated per projection in the checkpoint (Gemma 4's audio tower). */
int ref_clamp(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *L = &a->t[1], *H = &a->t[2], *Y = &a->t[3];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0 || numel(L) < 1 || numel(H) < 1) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(X) < rows * n) return RAD_E_SHAPE;
    const float lo = ldt(L, 0), hi = ldt(H, 0);

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, n), yb = rowoff(Y, r, n);
        const int64_t xs = laststride(X), ys = laststride(Y);
        for (int64_t i = 0; i < n; ++i) {
            float v = ldt(X, xb + i * xs);
            v = v < lo ? lo : v;
            v = v > hi ? hi : v;
            stt(Y, yb + i * ys, v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv2d */
/* A 2-D convolution over (time, width) with channels innermost, segment by segment.
 *
 * x is [M, W * cin]: row t is time t, element (t, w, c) at column w * cin + c. Each cu_seqlens
 * segment [cu[s] / cu_div, cu[s+1] / cu_div) is convolved separately, zero-padded by `pad` on both
 * axes, so clips in one pass do not see each other; its output rows start at
 * cu[s] / (cu_div * stride). `w` is PyTorch's [cout, cin, kh, kw], kh over time; with `t_last` it
 * is [cout, cin, kw, kh], time last (a model that convolves a [mel, time] spectrogram, whose
 * kernel's first spatial axis is the one this op calls width). With the optional `valid` operand,
 * input row t (pass-global) reads as zeros unless valid[t * valid_stride] != 0: Gemma 4's
 * subsampler frame mask, or the chunk boundary that keeps each second of Qwen3-ASR's audio apart.
 * The optional `b` adds a bias per output channel before the one rounding; `act` (`none`, `gelu`)
 * is then applied to the rounded value and rounded again, as gemm_nt_bias's epilogue does. f32
 * accumulation. */
int ref_conv2d(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 5 || !a->t[0].data || !a->t[1].data || !a->t[2].data || !a->t[4].data)
        return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Wt = &a->t[1], *cu = &a->t[2], *Y = &a->t[4];
    const RadTensor* V = a->t[3].data ? &a->t[3] : nullptr;
    const RadTensor* B = a->n_t > 5 && a->t[5].data ? &a->t[5] : nullptr;
    const int64_t W = p_int(a, "width", 0), cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0);
    const int64_t kh = p_int(a, "kh", 0), kw = p_int(a, "kw", 0);
    const int64_t st = p_int(a, "stride", 1), pad = p_int(a, "pad", 0);
    const int64_t cdiv = p_int(a, "cu_div", 1), vst = p_int(a, "valid_stride", 1);
    const bool tlast = p_int(a, "t_last", 0) != 0;
    const char* act = p_str(a, "act", "none");
    const bool gelu = !std::strcmp(act, "gelu");
    if (!gelu && std::strcmp(act, "none")) return RAD_E_UNSUPPORTED;
    if (W <= 0 || cin <= 0 || cout <= 0 || kh <= 0 || kw <= 0 || st <= 0 || pad < 0 || cdiv <= 0 ||
        vst <= 0)
        return RAD_E_SHAPE;
    const int64_t Wo = (W + 2 * pad - kw) / st + 1;
    if (Wo <= 0 || numel(Wt) < cout * cin * kh * kw) return RAD_E_SHAPE;
    if (B && numel(B) < cout) return RAD_E_SHAPE;
    const int64_t S = numel(cu) - 1;
    if (S <= 0) return RAD_E_SHAPE;
    const int64_t rin = W * cin, rout = Wo * cout;
    const int64_t Min = rows_of(X, rin), Mout = rows_of(Y, rout);
    if (Min <= 0 || Mout <= 0 || cu_at(cu, S) <= cu_at(cu, 0)) return RAD_E_SHAPE;   /* zero rows */
    /* Kernel tap (i over time, j over width) of channel pair (co, ci). */
    auto widx = [&](int64_t co, int64_t ci, int64_t i, int64_t j) {
        return tlast ? ((co * cin + ci) * kw + j) * kh + i : ((co * cin + ci) * kh + i) * kw + j;
    };

    for (int64_t s = 0; s < S; ++s) {
        const int64_t c0 = cu_at(cu, s);
        const int64_t c1 = cu_at(cu, s + 1);
        if (c0 % cdiv || c1 % cdiv || (c0 / cdiv) % st || c1 < c0) return RAD_E_SHAPE;
        const int64_t t0 = c0 / cdiv, T = c1 / cdiv - t0;
        if (T <= 0) continue;
        const int64_t To = (T + 2 * pad - kh) / st + 1, o0 = t0 / st;
        if (To <= 0) continue;
        if (t0 + T > Min || o0 + To > Mout) return RAD_E_SHAPE;
        if (V && numel(V) < (t0 + T - 1) * vst + 1) return RAD_E_SHAPE;

        REF_PAR_FOR(To * rout * cin * kh * kw)
        for (int64_t to = 0; to < To; ++to) {
            const int64_t yb = rowoff(Y, o0 + to, rout), ys = laststride(Y), xs = laststride(X);
            for (int64_t wo = 0; wo < Wo; ++wo)
                for (int64_t co = 0; co < cout; ++co) {
                    float acc = 0.0f;
                    for (int64_t i = 0; i < kh; ++i) {
                        const int64_t ti = to * st - pad + i;
                        if (ti < 0 || ti >= T) continue;
                        const int64_t tg = t0 + ti;
                        if (V && ld_int(V->dtype, V->data, offlin(V, tg * vst)) == 0) continue;
                        const int64_t xb = rowoff(X, tg, rin);
                        for (int64_t j = 0; j < kw; ++j) {
                            const int64_t wi = wo * st - pad + j;
                            if (wi < 0 || wi >= W) continue;
                            for (int64_t ci = 0; ci < cin; ++ci)
                                acc += ldt(X, xb + (wi * cin + ci) * xs) * ldt(Wt, widx(co, ci, i, j));
                        }
                    }
                    if (B) acc += ldt(B, offlin(B, co));
                    const int64_t yo = yb + (wo * cout + co) * ys;
                    stt(Y, yo, acc);
                    if (gelu) stt(Y, yo, act_gelu(ldt(Y, yo)));
                }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv2d_unfold */
/* conv2d's receptive fields as rows, so a GEMM against the weight is the convolution: row
 * (o0 + to) * Wo + wo of y holds, for every input channel and kernel tap, the value conv2d would
 * multiply that tap's weight by -- zero where the tap falls in the padding or on a row `valid`
 * masks. Its columns are in the weight's order, (ci, i, j) for PyTorch's [cout, cin, kh, kw] and
 * (ci, j, i) with t_last, so y times the weight viewed as [cout, cin * kh * kw] is conv2d's output
 * with no relayout of either: row (to, wo) of the product is column block wo of conv2d's row to.
 * Segments, cu_div and valid as conv2d's; rows no segment covers are left alone. */
int ref_conv2d_unfold(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 4 || !a->t[0].data || !a->t[1].data || !a->t[3].data) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *cu = &a->t[1], *Y = &a->t[3];
    const RadTensor* V = a->t[2].data ? &a->t[2] : nullptr;
    const int64_t W = p_int(a, "width", 0), cin = p_int(a, "cin", 0);
    const int64_t kh = p_int(a, "kh", 0), kw = p_int(a, "kw", 0);
    const int64_t st = p_int(a, "stride", 1), pad = p_int(a, "pad", 0);
    const int64_t cdiv = p_int(a, "cu_div", 1), vst = p_int(a, "valid_stride", 1);
    const bool tlast = p_int(a, "t_last", 0) != 0;
    if (W <= 0 || cin <= 0 || kh <= 0 || kw <= 0 || st <= 0 || pad < 0 || cdiv <= 0 || vst <= 0)
        return RAD_E_SHAPE;
    const int64_t Wo = (W + 2 * pad - kw) / st + 1, K = cin * kh * kw;
    const int64_t S = numel(cu) - 1;
    if (Wo <= 0 || S <= 0) return RAD_E_SHAPE;
    const int64_t rin = W * cin, Min = rows_of(X, rin), Mout = rows_of(Y, K);
    if (Min <= 0 || Mout <= 0 || cu_at(cu, S) <= cu_at(cu, 0)) return RAD_E_SHAPE;   /* zero rows */

    for (int64_t s = 0; s < S; ++s) {
        const int64_t c0 = cu_at(cu, s);
        const int64_t c1 = cu_at(cu, s + 1);
        if (c0 % cdiv || c1 % cdiv || (c0 / cdiv) % st || c1 < c0) return RAD_E_SHAPE;
        const int64_t t0 = c0 / cdiv, T = c1 / cdiv - t0;
        if (T <= 0) continue;
        const int64_t To = (T + 2 * pad - kh) / st + 1, o0 = t0 / st;
        if (To <= 0) continue;
        if (t0 + T > Min || (o0 + To) * Wo > Mout) return RAD_E_SHAPE;
        if (V && numel(V) < (t0 + T - 1) * vst + 1) return RAD_E_SHAPE;

        REF_PAR_FOR(To * Wo * K)
        for (int64_t to = 0; to < To; ++to) {
            const int64_t ys = laststride(Y), xs = laststride(X);
            for (int64_t wo = 0; wo < Wo; ++wo) {
                const int64_t yb = rowoff(Y, (o0 + to) * Wo + wo, K);
                for (int64_t i = 0; i < kh; ++i) {
                    const int64_t ti = to * st - pad + i;
                    const int64_t tg = t0 + ti;
                    const bool row_ok = ti >= 0 && ti < T &&
                                        !(V && ld_int(V->dtype, V->data, offlin(V, tg * vst)) == 0);
                    for (int64_t j = 0; j < kw; ++j) {
                        const int64_t wi = wo * st - pad + j;
                        const bool ok = row_ok && wi >= 0 && wi < W;
                        for (int64_t ci = 0; ci < cin; ++ci) {
                            const int64_t k = tlast ? (ci * kw + j) * kh + i : (ci * kh + i) * kw + j;
                            const float v = ok ? ldt(X, rowoff(X, tg, rin) + (wi * cin + ci) * xs)
                                               : 0.0f;
                            stt(Y, yb + k * ys, v);
                        }
                    }
                }
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv1d_dw */
/* A causal depthwise convolution over time: y[t, c] = sum_j w[c, j] * x[t - (k-1) + j, c], the
 * rows before the segment's start (cu[s] / cu_div) read as zeros, plus `b` [C] when bound. `w` is
 * PyTorch's depthwise [C, 1, k]. f32 accumulation, one rounding. */
int ref_conv1d_dw(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Wt = &a->t[1], *cu = &a->t[2], *Y = &a->t[3];
    const RadTensor *B = t_in(a, 4);
    const int64_t C = p_int(a, "n", 0), K = p_int(a, "k", 0), cdiv = p_int(a, "cu_div", 1);
    if (C <= 0 || K <= 0 || cdiv <= 0 || numel(Wt) < C * K || (B && numel(B) < C))
        return RAD_E_SHAPE;
    const int64_t S = numel(cu) - 1;
    if (S <= 0) return RAD_E_SHAPE;
    const int64_t Mx = rows_of(X, C), My = rows_of(Y, C);
    if (Mx <= 0 || My <= 0 || cu_at(cu, S) <= cu_at(cu, 0)) return RAD_E_SHAPE;   /* zero rows */

    for (int64_t s = 0; s < S; ++s) {
        const int64_t c0 = cu_at(cu, s);
        const int64_t c1 = cu_at(cu, s + 1);
        if (c0 % cdiv || c1 % cdiv || c1 < c0) return RAD_E_SHAPE;
        const int64_t t0 = c0 / cdiv, t1 = c1 / cdiv;
        if (t1 > Mx || t1 > My) return RAD_E_SHAPE;

        REF_PAR_FOR((t1 - t0) * C * K)
        for (int64_t t = t0; t < t1; ++t) {
            const int64_t yb = rowoff(Y, t, C), ys = laststride(Y), xs = laststride(X);
            for (int64_t c = 0; c < C; ++c) {
                float acc = 0.0f;
                for (int64_t j = 0; j < K; ++j) {
                    const int64_t tt = t - (K - 1) + j;
                    if (tt < t0) continue;
                    acc += ldt(X, rowoff(X, tt, C) + c * xs) * ldt(Wt, c * K + j);
                }
                if (B) acc += ldt(B, c);
                stt(Y, yb + c * ys, acc);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rel_sinusoid */
/* The sinusoidal relative-position table, row d for distance d: sin(d * inv[i]) in the first half,
 * cos(d * inv[i]) in the second, inv[i] = exp(i * -log(max_timescale) / max(n/2 - 1, 1)). Computed
 * in f32 step by step as the upstream torch expressions are, then rounded. */
int ref_rel_sinusoid(const RadArgs* a, RadStream) {
    if (!have(a, 1)) return RAD_E_INVAL;
    const RadTensor* Y = &a->t[0];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || n % 2) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0) return RAD_E_SHAPE;
    const int64_t h = n / 2;
    const double mt = (double)p_f32(a, "max_timescale", 10000.0f);
    const float inc = (float)(std::log(mt / 1.0) / (double)(h - 1 > 1 ? h - 1 : 1));

    REF_PAR_FOR(rows * n)
    for (int64_t d = 0; d < rows; ++d) {
        const int64_t yb = rowoff(Y, d, n), ys = laststride(Y);
        for (int64_t i = 0; i < h; ++i) {
            const float inv = ref_exp((float)i * -inc);
            const float ang = (float)d * inv;
            stt(Y, yb + i * ys, ref_sin(ang));
            stt(Y, yb + (h + i) * ys, ref_cos(ang));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gelu_mul */
/* y[.., i] = gelu_tanh(gate_up[.., i]) * gate_up[.., n + i], gate first as in silu_mul. Gemma's
 * gated FFN, down(gelu_tanh(gate(x)) * up(x)) (gelu_pytorch_tanh in the checkpoint). Unlike
 * gemm_nt's "gelu_tanh" epilogue it multiplies two halves of one fused plane. Deliberately the tanh
 * form, which GELU-gated FFNs are trained with; the exact-erf form is the unary `gelu` op. */
int ref_gelu_mul(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *gu = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(gu) < rows * 2 * n) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t gb = rowoff(gu, r, 2 * n), yb = rowoff(y, r, n);
        const int64_t gs = laststride(gu), ys = laststride(y);
        for (int64_t i = 0; i < n; ++i)
            stt(y, yb + i * ys, act_gelu_tanh(ldt(gu, gb + i * gs)) * ldt(gu, gb + (n + i) * gs));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ seg_sinusoid */
/* A Whisper-style absolute position table restarting every `period` rows of a segment: row r of
 * segment s (rows [cu[s] / cu_div, cu[s+1] / cu_div)) takes position p = (r - cu[s] / cu_div) mod
 * period, and is sin(p * inv[i]) then cos(p * inv[i]) over n/2 timescales, inv[i] as
 * rel_sinusoid's, each step in f32 and rounded once. Qwen3-ASR's encoder adds it to every chunk's
 * rows, which restart at position 0 with each second of audio. A row no segment covers is left
 * untouched. */
int ref_seg_sinusoid(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *cu = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), period = p_int(a, "period", 0), cdiv = p_int(a, "cu_div", 1);
    if (n <= 0 || n % 2 || period <= 0 || cdiv <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n), S = numel(cu) - 1;
    if (rows <= 0 || S <= 0 || cu_at(cu, S) <= cu_at(cu, 0)) return RAD_E_SHAPE;   /* zero rows */
    const int64_t h = n / 2;
    const double mt = (double)p_f32(a, "max_timescale", 10000.0f);
    const float inc = (float)(std::log(mt / 1.0) / (double)(h - 1 > 1 ? h - 1 : 1));

    for (int64_t s = 0; s < S; ++s) {
        const int64_t c0 = cu_at(cu, s);
        const int64_t c1 = cu_at(cu, s + 1);
        if (c0 % cdiv || c1 % cdiv || c1 < c0 || c1 / cdiv > rows) return RAD_E_SHAPE;
        REF_PAR_FOR((c1 / cdiv - c0 / cdiv) * n * 4)
        for (int64_t r = c0 / cdiv; r < c1 / cdiv; ++r) {
            const int64_t p = (r - c0 / cdiv) % period;
            const int64_t yb = rowoff(Y, r, n), ys = laststride(Y);
            for (int64_t i = 0; i < h; ++i) {
                const float inv = ref_exp((float)i * -inc);
                const float ang = (float)p * inv;
                stt(Y, yb + i * ys, ref_sin(ang));
                stt(Y, yb + (h + i) * ys, ref_cos(ang));
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gelu / silu */
static int unary(const RadArgs* a, int which) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t nel = numel(y);
    if (nel <= 0 || numel(x) < nel) return RAD_E_SHAPE;

    REF_PAR_FOR(nel * 4)
    for (int64_t i = 0; i < nel; ++i) {
        float v = ldt(x, offlin(x, i));
        stt(y, offlin(y, i), which == 0 ? act_gelu(v) : which == 1 ? act_silu(v) : act_sigmoid(v));
    }
    return RAD_OK;
}

int ref_gelu(const RadArgs* a, RadStream) { return unary(a, 0); }
int ref_silu(const RadArgs* a, RadStream) { return unary(a, 1); }

/* ------------------------------------------------------------------ sigmoid */
/* Its own op so the selector can name it: a gated shared expert issues it on the router's output
 * (docs/OPS.md, "Elementwise and norms"). */
int ref_sigmoid(const RadArgs* a, RadStream) { return unary(a, 2); }

/* ------------------------------------------------------------------ fill */
/* y = value over [M, n] at y's row pitch, rounded once by the store. */
int ref_fill(const RadArgs* a, RadStream) {
    if (!have(a, 1)) return RAD_E_INVAL;
    const RadTensor* y = &a->t[0];
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0) return RAD_E_SHAPE;
    const float v = p_f32(a, "value", 0.0f);

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t i = 0; i < n; ++i) stt(y, yb + i * ys, v);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ logit_softcap */
/* y = cap * tanh(x / cap), Gemma's final-logit softcap, issued on the logits. Its own op so the
 * selector can name it and so `cap` can be a call parameter. */
int ref_logit_softcap(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t nel = numel(y);
    if (nel <= 0 || numel(x) < nel) return RAD_E_SHAPE;
    const float cap = p_f32(a, "cap", 30.0f);
    if (!(cap > 0.0f)) return RAD_E_INVAL;

    REF_PAR_FOR(nel * 4)
    for (int64_t i = 0; i < nel; ++i) {
        const float v = ldt(x, offlin(x, i));
        stt(y, offlin(y, i), cap * ref_tanh(v / cap));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ scale */
/* y = x * s, s a [1] weight broadcast over all of x: Gemma 4's per-layer residual scalar (llama.cpp
 * `out_scale`). `mul` broadcasts a row and `scale_rows` one scalar per row; this is one scalar per
 * tensor, kept separate so the broadcasts are never ambiguous. `s` is a per-layer checkpoint weight,
 * not a call constant. */
int ref_scale(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *s = &a->t[1], *y = &a->t[2];
    const int64_t nel = numel(y);
    if (nel <= 0 || numel(x) < nel || numel(s) < 1) return RAD_E_SHAPE;
    const float sv = ldt(s, 0);
    REF_PAR_FOR(nel)
    for (int64_t i = 0; i < nel; ++i)
        stt(y, offlin(y, i), ldt(x, offlin(x, i)) * sv);
    return RAD_OK;
}
/* ------------------------------------------------------------------ gather_rows */
/* y[i, :] = x[idx[i], :]. Lets the lm_head GEMM run on one row per sequence instead of every
 * prefill token: at max_tok 8192 and a 248K vocab the full logits buffer would be 8 GiB
 * (docs/OPS.md).
 *
 * `M` is a range parameter, so the row count is y's extent and `idx` is read for that many entries.
 * A negative index writes a zero row (embed_lookup's padding rule; unsampled rows of a step). An
 * index at or past x's row count is a caller bug and returns RAD_E_INVAL. */
int ref_gather_rows(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *idx = &a->t[1], *y = &a->t[2];
    const int64_t n = p_int(a, "n", y->rank >= 1 ? y->shape[y->rank - 1] : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t M = rows_of(y, n);
    const int64_t src_rows = rows_of(x, n);
    if (M <= 0 || src_rows <= 0 || numel(idx) < M) return RAD_E_SHAPE;

    /* Validated before the loop, because an OpenMP body cannot return. */
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r = ld_int(idx->dtype, idx->data, offlin(idx, m));
        if (r >= src_rows) return RAD_E_INVAL;
    }

    REF_PAR_FOR(M * n)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r  = ld_int(idx->dtype, idx->data, offlin(idx, m));
        const int64_t yb = rowoff(y, m, n), ys = laststride(y);
        if (r < 0) { for (int64_t i = 0; i < n; ++i) stt(y, yb + i * ys, 0.0f); continue; }
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) stt(y, yb + i * ys, ldt(x, xb + i * xs));
    }
    return RAD_OK;
}


/* ------------------------------------------------------------------ scatter_rows */
/* The mirror of gather_rows: writes M rows to M arbitrary destinations in one launch instead of M
 * small transfers.
 *
 * `x` is INOUT and the write is partial: unnamed rows keep their contents, since the destination
 * is often a pool shared with other sequences.
 *
 * Duplicate indices are illegal (RAD_OPD_F_IDX_UNIQUE): two rows landing on one destination have no
 * defined winner in parallel, so reference and kernel could legitimately disagree. */
int ref_scatter_rows(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *v = &a->t[0], *idx = &a->t[1], *x = &a->t[2];
    const int64_t n = p_int(a, "n", v->rank >= 1 ? v->shape[v->rank - 1] : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t M = rows_of(v, n);
    const int64_t dst_rows = rows_of(x, n);
    if (M <= 0 || dst_rows <= 0 || numel(idx) < M) return RAD_E_SHAPE;

    /* Validated before the loop, since an OpenMP body cannot return. A negative index is skipped,
     * mirroring gather_rows' zero row: both mean the slot has no counterpart. */
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r = ld_int(idx->dtype, idx->data, offlin(idx, m));
        if (r >= dst_rows) return RAD_E_INVAL;
    }

    REF_PAR_FOR(M * n)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r = ld_int(idx->dtype, idx->data, offlin(idx, m));
        if (r < 0) continue;
        const int64_t vb = rowoff(v, m, n), vs = laststride(v);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) stt(x, xb + i * xs, ldt(v, vb + i * vs));
    }
    return RAD_OK;
}
/* ------------------------------------------------------------------ cast */
/* `from` and `to` exist for the selector (each dtype pair is a different kernel). Ref ignores them
 * and uses the tensors' own dtypes, which describe the bytes. */
int ref_cast(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t nel = numel(y);
    if (nel <= 0 || numel(x) < nel) return RAD_E_SHAPE;

    /* Integer-to-integer casts bypass f32, so ids and indices above 2^24 survive. */
    const bool int_pair = (x->dtype == RAD_I32 || x->dtype == RAD_I64 || x->dtype == RAD_U32 ||
                           x->dtype == RAD_I16) &&
                          (y->dtype == RAD_I32 || y->dtype == RAD_I64 || y->dtype == RAD_U32 ||
                           y->dtype == RAD_I16);
    REF_PAR_FOR(nel)
    for (int64_t i = 0; i < nel; ++i) {
        if (int_pair) st_int(y->dtype, y->data, offlin(y, i), ld_int(x->dtype, x->data, offlin(x, i)));
        else          stt(y, offlin(y, i), ldt(x, offlin(x, i)));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ softmax */
/* Max-subtracted, f32, over the last n elements of a row. Every kernel subtracts the max (a 60-nat
 * logit range would otherwise overflow), so the reference must too. */
int ref_softmax(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(x) < rows * n) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n * 4)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), yb = rowoff(y, r, n);
        const int64_t xs = laststride(x), ys = laststride(y);
        float mx = -INFINITY;
        for (int64_t i = 0; i < n; ++i) { float v = ldt(x, xb + i * xs); if (v > mx) mx = v; }
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            float e = ref_exp(ldt(x, xb + i * xs) - mx);
            sum += e;
            stt(y, yb + i * ys, e);
        }
        const float inv = 1.0f / sum;
        for (int64_t i = 0; i < n; ++i) stt(y, yb + i * ys, ldt(y, yb + i * ys) * inv);
    }
    return RAD_OK;
}
