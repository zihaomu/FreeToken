/* lq_awq.cpp -- AWQ: activation-aware weight quantisation (Lin et al., 2023), on any grid.
 *
 * A few input channels carry most of a layer's activation energy, and a weight column's output
 * error scales with its channel's activation. AWQ scales each column up by s_j before quantising,
 * so important channels use more of the grid, and the reader scales it back down. s_j = a_j^alpha
 * over the channel activation magnitudes a_j, normalised so their geometric middle is one; alpha
 * is picked from `alpha_steps` values in [0, 1) by output error. alpha = 0 is plain rounding, so
 * the search never does worse. A per-block clip search follows (AWQ's auto_clip): each scale
 * block's range is shrunk in steps of 0.05 and the factor with least output error is kept.
 *
 * Output error: the Hessian's quadratic form (lq_hess.h), over `sample_rows` rows for the alpha
 * search and over each block's own columns (H restricted to them) for the clip. This is the error
 * AWQ measures on calibration activations, without needing them. a_j = sqrt(H_jj), the channel's
 * RMS. With no Hessian but an imatrix, the imatrix is the diagonal (a_j = sqrt(importance), error
 * diagonal-weighted); with neither, alpha is 0 and only the plain grid remains. Each fallback is
 * reported on stderr.
 *
 * The scales are stored, not folded. The reference implementation folds 1/s into the preceding
 * norm or projection, which needs the model graph. Here they are the encoding's last scale level,
 * a [* x 1] plane with one `colscale`-typed value per column holding the multiplier c_j = 1/s_j.
 * The weight is divided by the rounded c_j, so the reader multiplies back exactly what was divided.
 *
 * No transform: the statistics describe the input's own channels, which a rotated weight's columns
 * are not.
 */
#include "lq_hess.h"
#include "lq_quant.h"
#include "lq_common.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace lq {

namespace {

const RadQuantOption kAwqOptions[] = {
    LQ_GRID_OPTIONS,
    LQ_CALIB_OPTION("the imatrix's diagonal stands in, and with neither the weight is rounded to "
                    "nearest with unit column scales"),
    { "colscale", RAD_P_STR, "f16", "the column scales' dtype: f32, bf16, f16" },
    { "alpha_steps", RAD_P_INT, "20", "the exponents searched: k / alpha_steps, k = 0 .. steps-1" },
    { "clip", RAD_P_INT, "1", "1: search each scale block's clipping range (AWQ's auto_clip)" },
    { "sample_rows", RAD_P_INT, "512", "rows the alpha search measures the output error over" },
};

int awq_grid(const RadParam* o, int n_o, const RadQuantWeight* w, Grid* g, uint32_t* cdt) {
    const int st = grid_of(o, n_o, w, g);
    if (st != RAD_OK) return st;
    if (g->fwht) return refuse(w, "awq takes no transform= -- its statistics are the input's own "
                                  "channels, and a rotated weight's columns are not those");
    const char* c = rad_param_gets(o, n_o, "colscale", "f16");
    *cdt = rad_dtype_parse(c);
    if (*cdt != RAD_F32 && *cdt != RAD_BF16 && *cdt != RAD_F16)
        return refuse(w, "colscale= must be f32, bf16 or f16");
    if (rad_param_geti(o, n_o, "alpha_steps", 20) < 1) return refuse(w, "alpha_steps= must be >= 1");
    return RAD_OK;
}

/* The column scale is the chain's next level after the grid's own. */
const char* colscale_role(const Grid& g) { return g.lv2 ? "scale.2" : "scale.1"; }

int awq_encoding(const RadParam* o, int n_o, const RadQuantWeight* w, RadEncoding* out) {
    if (declines(w)) return RAD_E_UNSUPPORTED;
    Grid g;
    uint32_t cdt = 0;
    const int st = awq_grid(o, n_o, w, &g, &cdt);
    if (st != RAD_OK) return st;
    *out = grid_encoding(g);
    if (rad_enc_add_plane(out, colscale_role(g), cdt, 0, 1) < 0)
        return refuse(w, "the encoding has no room for the column scales");
    return RAD_OK;
}

int64_t awq_row_block(const RadParam*, int, const RadQuantWeight*) { return 0; }

/* The output error of a candidate, D = w_hat - w over `rows` rows: the Hessian's form, or the
 * diagonal's. */
struct Err {
    const float* H = nullptr;     /* K x K, or null */
    const float* diag = nullptr;  /* K, the imatrix, when there is no H */
    int64_t K = 0;
    double operator()(const float* D, int64_t rows) const {
        if (H) {
            std::vector<float> G((size_t)(rows * K));
            std::vector<double> q((size_t)rows);
            mat_hess(D, rows, K, H, G.data());
            row_dots(D, G.data(), rows, K, q.data());
            double s = 0.0;
            for (double x : q) s += x;
            return s;
        }
        double s = 0.0;
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t j = 0; j < K; ++j) {
                const double d = D[r * K + j];
                s += (diag ? (double)diag[j] : 1.0) * d * d;
            }
        return s;
    }
};

int awq_quantize(const RadParam* o, int n_o, const RadQuantWeight* w, float* src, int64_t row0,
                 int64_t rows, void* const* planes) {
    Grid g;
    uint32_t cdt = 0;
    int st = awq_grid(o, n_o, w, &g, &cdt);
    if (st != RAD_OK) return st;
    if (row0 != 0 || rows != w->rows) return RAD_E_SHAPE;
    const int64_t N = w->rows, K = w->cols;
    const char* dir = rad_param_gets(o, n_o, "calib", nullptr);
    const int steps = (int)rad_param_geti(o, n_o, "alpha_steps", 20);
    const bool clip = rad_param_geti(o, n_o, "clip", 1) != 0;
    const int64_t want = std::max<long long>(1, rad_param_geti(o, n_o, "sample_rows", 512));

    Err err;
    err.K = K;
    const Gram Hg = calib_gram(dir, w->name, K, g.fwht, "awq");
    err.H = Hg ? Hg->data() : nullptr;
    if (!err.H && w->importance) {
        err.diag = w->importance;
        std::fprintf(stderr, "libquant: awq: '%s' has no Hessian; the imatrix's diagonal stands "
                             "in for it\n", w->name);
    }
    const bool have = err.H || err.diag;

    /* Channel activation magnitudes, floored as AWQ floors them. */
    std::vector<double> a((size_t)K, 1.0);
    if (have)
        for (int64_t j = 0; j < K; ++j) {
            const double e = err.H ? (double)err.H[j * K + j] : (double)err.diag[j];
            a[(size_t)j] = std::max(std::sqrt(std::max(e, 0.0)), 1e-4);
        }

    /* The sample: whole row blocks of the scale grid, evenly spread, so a candidate is quantised
     * exactly as the full weight will be. */
    const int64_t unit = g.br > 0 ? g.br : N;
    const int64_t nunits = (N + unit - 1) / unit;
    const int64_t take = std::min<int64_t>(nunits, std::max<int64_t>(1, want / unit));
    std::vector<int64_t> pick;
    for (int64_t i = 0; i < take; ++i) pick.push_back(i * nunits / take);
    std::vector<float> ws;
    const int64_t srows = gather_units(src, N, K, unit, pick, &ws);

    /* c_j for an exponent: the stored column multiplier 1 / s_j, rounded to its dtype. */
    auto colscales = [&](double alpha, std::vector<float>* c) {
        std::vector<double> s((size_t)K);
        double mx = 0.0, mn = 1e300;
        for (int64_t j = 0; j < K; ++j) {
            s[(size_t)j] = std::max(std::pow(a[(size_t)j], alpha), 1e-4);
            mx = std::max(mx, s[(size_t)j]);
            mn = std::min(mn, s[(size_t)j]);
        }
        const double norm = std::sqrt(mx * mn);
        c->resize((size_t)K);
        for (int64_t j = 0; j < K; ++j)
            (*c)[(size_t)j] = round_to(cdt, (float)(norm / s[(size_t)j]));
    };

    /* Search the exponent. */
    std::vector<float> c, best_c, scaled((size_t)(srows * K)), fake((size_t)(srows * K));
    colscales(0.0, &best_c);
    if (have) {
        double best = -1.0;
        int best_k = 0;
        for (int k = 0; k < steps; ++k) {
            colscales((double)k / steps, &c);
            for (int64_t r = 0; r < srows; ++r)
                for (int64_t j = 0; j < K; ++j)
                    scaled[(size_t)(r * K + j)] = ws[(size_t)(r * K + j)] / c[(size_t)j];
            grid_fake(g, scaled.data(), srows, K, fake.data());
            for (int64_t r = 0; r < srows; ++r)
                for (int64_t j = 0; j < K; ++j) {
                    const size_t i = (size_t)(r * K + j);
                    fake[i] = fake[i] * c[(size_t)j] - ws[i];
                }
            const double e = err(fake.data(), srows);
            if (best < 0.0 || e < best) { best = e; best_k = k; best_c = c; }
        }
        std::fprintf(stderr, "libquant: awq: '%s' alpha %.3f\n", w->name, (double)best_k / steps);
    }

    /* Scale the whole weight. */
    for (int64_t r = 0; r < N; ++r)
        for (int64_t j = 0; j < K; ++j) src[r * K + j] /= best_c[(size_t)j];

    /* Clip search per block: quantise the clamped block and measure the error in original units
     * against H over the block's columns. */
    if (clip && have) {
        const int64_t br = g.br > 0 ? g.br : N, bc = g.bc > 0 ? g.bc : K;
        const int64_t nr = (N + br - 1) / br, nc = (K + bc - 1) / bc;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int64_t kb = 0; kb < nr * nc; ++kb) {
            const int64_t r0 = (kb / nc) * br, r1 = std::min(r0 + br, N);
            const int64_t c0 = (kb % nc) * bc, c1 = std::min(c0 + bc, K);
            const int64_t m = r1 - r0, n = c1 - c0;
            std::vector<float> blk((size_t)(m * n)), cl((size_t)(m * n)), dq((size_t)(m * n));
            float lo = 0.0f, hi = 0.0f;
            for (int64_t r = 0; r < m; ++r)
                for (int64_t j = 0; j < n; ++j) {
                    const float v = src[(r0 + r) * K + c0 + j];
                    blk[(size_t)(r * n + j)] = v;
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                }
            const float amax = std::max(-lo, hi);
            double best = -1.0;
            float best_f = 1.0f;
            for (int i = 0; i < 10; ++i) {               /* 1.00, 0.95, ... 0.55 */
                const float f = 1.0f - 0.05f * (float)i;
                const float l = g.asym ? lo * f : -amax * f, h = g.asym ? hi * f : amax * f;
                for (size_t t = 0; t < cl.size(); ++t) cl[t] = std::min(std::max(blk[t], l), h);
                const BlockScale b = grid_block_scale(g, cl.data(), n, 0, m, 0, n);
                double e = 0.0;
                for (int64_t r = 0; r < m; ++r) {
                    for (int64_t j = 0; j < n; ++j) {
                        const size_t t = (size_t)(r * n + j);
                        dq[t] = (grid_decode(g, b, grid_code(g, b, cl[t])) - blk[t]) *
                                best_c[(size_t)(c0 + j)];
                    }
                    const float* d = dq.data() + r * n;
                    if (err.H) {
                        for (int64_t x = 0; x < n; ++x) {
                            double hx = 0.0;
                            const float* hr = err.H + (c0 + x) * K + c0;
                            for (int64_t y = 0; y < n; ++y) hx += (double)hr[y] * d[y];
                            e += d[x] * hx;
                        }
                    } else {
                        for (int64_t x = 0; x < n; ++x) e += (double)err.diag[c0 + x] * d[x] * d[x];
                    }
                }
                if (best < 0.0 || e < best) { best = e; best_f = f; }
            }
            const float l = g.asym ? lo * best_f : -amax * best_f;
            const float h = g.asym ? hi * best_f : amax * best_f;
            for (int64_t r = 0; r < m; ++r)
                for (int64_t j = 0; j < n; ++j) {
                    float& v = src[(r0 + r) * K + c0 + j];
                    v = std::min(std::max(v, l), h);
                }
        }
    }

    /* Quantise the scaled, clipped weight, then store the column scales. A column's importance in
     * the scaled weight is its own times c_j^2, since its error grows by c_j when the reader
     * multiplies the column back. */
    std::vector<float> imp;
    if (w->importance) {
        imp.resize((size_t)K);
        for (int64_t j = 0; j < K; ++j)
            imp[(size_t)j] = w->importance[j] * best_c[(size_t)j] * best_c[(size_t)j];
    }
    st = grid_rtn(g, src, 0, N, K, grid_planes(g, K, planes), imp.empty() ? nullptr : imp.data());
    if (st != RAD_OK) return st;
    const int cp = grid_encoding(g).n_planes;
    for (int64_t j = 0; j < K; ++j) rad_store_f32(planes[cp], cdt, j, best_c[(size_t)j]);
    return RAD_OK;
}

}  /* namespace */

const RadQuantizerInfo& awq_info() {
    static const RadQuantizerInfo q = {
        "awq",
        "activation-aware: each input column scaled by its activation magnitude to a searched "
        "power before the grid, a per-block clip searched after, both against the layer's input "
        "Hessian; the column scales are stored as the encoding's last scale level",
        kAwqOptions, (int)(sizeof kAwqOptions / sizeof kAwqOptions[0]),
        awq_encoding, awq_row_block, awq_quantize, nullptr,
    };
    return q;
}

}  /* namespace lq */
