/* lq_autoround.cpp -- AutoRound: rounding by signed gradient descent (Cheng et al., 2023).
 *
 * Rounding to nearest decides each value alone, but the output error depends on the whole row.
 * AutoRound learns a per-value offset V in [-0.5, 0.5], q = round(w / s + V), and per scale block a
 * range factor in [0.5, 1] (one for a symmetric block, one per end for an asymmetric one). It
 * descends on the output error with the sign of each gradient, through the rounding by the
 * straight-through estimator (d round(x) / dx = 1). The step is `lr` (default 1/iters) decaying
 * linearly to zero. Each row keeps its lowest-error parameters, so the result is never worse than
 * the starting point, round-to-nearest.
 *
 * Error: the Hessian's quadratic form (lq_hess.h) instead of the paper's block output over
 * calibration samples. For one linear layer it is the same function of the weight, and it
 * decomposes over rows, so each row descends alone: no batch, no sample noise, a row block per
 * call. Cost is N K^2 multiply-adds per step, which dominates conversion time at the default 200
 * steps; `iters` trades it.
 *
 * Grid: integer codes, symmetric or with a zero, blocks of one row. A codebook or float code has
 * no rounding to learn, and a second level or a multi-row block would tie rows together.
 */
#include "lq_autoround.h"
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

std::string autoround_refusal(const Grid& g) {
    if (!g.table.empty()) return "a codebook has no rounding to learn";
    if (g.is_float) return "a float code has no rounding to learn";
    if (g.wide) return "16- and 32-bit codes round to within a part in 30000 already";
    if (g.lv2) return "a second scale level ties the blocks of a row together";
    if (g.br != 1) return "a scale block spanning rows ties those rows together; use group= or "
                          "block=1xC";
    if (g.rule == Grid::MX || g.rule == Grid::FIXED) return "rule=mx and rule=fixed have no "
                                                            "range to learn";
    return std::string();
}

namespace {

inline float sgn(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }

/* One row's state through a step: its blocks' ranges and the scale, zero and codes they give. */
struct RowState {
    const Grid* g = nullptr;
    int64_t K = 0, nc = 0, bc = 0;
    std::vector<float> s0, lo0, hi0;   /* the absmax ends a block, from the weight */

    void ends(const float* w) {
        lo0.assign((size_t)nc, 0.0f);
        hi0.assign((size_t)nc, 0.0f);
        s0.assign((size_t)nc, 0.0f);
        for (int64_t j = 0; j < nc; ++j) {
            const int64_t c0 = j * bc, c1 = std::min(c0 + bc, K);
            float lo = 0.0f, hi = 0.0f;
            for (int64_t c = c0; c < c1; ++c) { lo = std::min(lo, w[c]); hi = std::max(hi, w[c]); }
            lo0[(size_t)j] = lo;
            hi0[(size_t)j] = hi;
            s0[(size_t)j] = std::max(-lo, hi) / (float)g->qmax;
        }
    }
    /* block j's scale and zero for range factors (a, b): a on the high end, b on the low */
    BlockScale block(int64_t j, float a, float b) const {
        if (g->asym) {
            const float hi = hi0[(size_t)j] * a, lo = lo0[(size_t)j] * b;
            return grid_finish(*g, (hi - lo) / (float)g->qmax, lo);
        }
        return grid_finish(*g, (double)(s0[(size_t)j] * a), 0.0f);
    }
};

/* The code of `x` with offset `v` against block `b`, and whether the clamp took it. */
inline int code_v(const Grid& g, const BlockScale& b, float x, float v, bool* clamped) {
    const int r = round_away(x * b.inv + v) + (g.asym ? b.zero : 0);
    *clamped = r < g.qmin || r > g.qmax;
    return clampi(r, g.qmin, g.qmax);
}

}  /* namespace */

void autoround_rows(const Grid& g, const float* w, int64_t N, int64_t K, const float* H,
                    const float* diag, int iters, double lr, const Planes& planes) {
    const int64_t bc = g.bc > 0 ? g.bc : K, nc = (K + bc - 1) / bc;
    const size_t NK = (size_t)(N * K), NB = (size_t)(N * nc);
    std::vector<float> V(NK, 0.0f), bestV(NK, 0.0f), D(NK), G(NK);
    std::vector<float> A(NB, 1.0f), B(NB, 1.0f), bestA(NB, 1.0f), bestB(NB, 1.0f);
    std::vector<double> loss((size_t)N), best((size_t)N, -1.0);
    std::vector<RowState> rs((size_t)N);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < N; ++r) {
        rs[(size_t)r].g = &g;
        rs[(size_t)r].K = K;
        rs[(size_t)r].nc = nc;
        rs[(size_t)r].bc = bc;
        rs[(size_t)r].ends(w + r * K);
    }

    /* the residual D = w_hat - w of the current parameters, every row */
    auto forward = [&](const std::vector<float>& v, const std::vector<float>& a,
                       const std::vector<float>& b) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t r = 0; r < N; ++r)
            for (int64_t j = 0; j < nc; ++j) {
                const BlockScale bs = rs[(size_t)r].block(j, a[(size_t)(r * nc + j)],
                                                          b[(size_t)(r * nc + j)]);
                const int64_t c1 = std::min((j + 1) * bc, K);
                for (int64_t c = j * bc; c < c1; ++c) {
                    const size_t i = (size_t)(r * K + c);
                    bool cl = false;
                    const int q = code_v(g, bs, w[i], v[i], &cl);
                    D[i] = (float)(q - (g.asym ? bs.zero : 0)) * bs.s - w[i];
                }
            }
    };
    auto errors = [&]() {
        if (H) {
            mat_hess(D.data(), N, K, H, G.data());
        } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int64_t r = 0; r < N; ++r)
                for (int64_t c = 0; c < K; ++c)
                    G[(size_t)(r * K + c)] = D[(size_t)(r * K + c)] * (diag ? diag[c] : 1.0f);
        }
        row_dots(D.data(), G.data(), N, K, loss.data());
    };
    auto keep = [&]() {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t r = 0; r < N; ++r) {
            if (best[(size_t)r] >= 0.0 && !(loss[(size_t)r] < best[(size_t)r])) continue;
            best[(size_t)r] = loss[(size_t)r];
            std::memcpy(&bestV[(size_t)(r * K)], &V[(size_t)(r * K)], (size_t)K * sizeof(float));
            std::memcpy(&bestA[(size_t)(r * nc)], &A[(size_t)(r * nc)], (size_t)nc * sizeof(float));
            std::memcpy(&bestB[(size_t)(r * nc)], &B[(size_t)(r * nc)], (size_t)nc * sizeof(float));
        }
    };

    for (int t = 0; t < iters; ++t) {
        forward(V, A, B);
        errors();
        keep();
        const float step = (float)(lr * (1.0 - (double)t / iters));
        /* Gradients through the straight-through rounding: dL/dw_hat = 2 G, and w_hat = s (q - z)
         * with q = round(w/s + V) + z clamped. Unclamped, dw_hat/dV = s and dw_hat/ds = (q - z) -
         * w/s; clamped, the first is 0 and the second (q - z). The range factors reach the loss
         * through s only (the zero is held where the rounding put it). */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t r = 0; r < N; ++r)
            for (int64_t j = 0; j < nc; ++j) {
                const size_t k = (size_t)(r * nc + j);
                const RowState& st = rs[(size_t)r];
                const BlockScale bs = st.block(j, A[k], B[k]);
                double gs = 0.0;
                const int64_t c1 = std::min((j + 1) * bc, K);
                for (int64_t c = j * bc; c < c1; ++c) {
                    const size_t i = (size_t)(r * K + c);
                    bool cl = false;
                    const int q = code_v(g, bs, w[i], V[i], &cl);
                    const float gw = 2.0f * G[i];
                    const float qz = (float)(q - (g.asym ? bs.zero : 0));
                    if (!cl) {
                        V[i] = std::min(0.5f, std::max(-0.5f, V[i] - step * sgn(gw * bs.s)));
                        gs += (double)gw * (qz - w[i] * bs.inv);
                    } else {
                        gs += (double)gw * qz;
                    }
                }
                if (g.asym) {
                    const float da = (float)(gs * st.hi0[(size_t)j] / g.qmax);
                    const float db = (float)(-gs * st.lo0[(size_t)j] / g.qmax);
                    A[k] = std::min(1.0f, std::max(0.5f, A[k] - step * sgn(da)));
                    B[k] = std::min(1.0f, std::max(0.5f, B[k] - step * sgn(db)));
                } else {
                    const float da = (float)(gs * st.s0[(size_t)j]);
                    A[k] = std::min(1.0f, std::max(0.5f, A[k] - step * sgn(da)));
                }
            }
    }
    forward(V, A, B);
    errors();
    keep();

    /* Write the best parameters: each block's scale and zero, then the codes. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < N; ++r) {
        uint8_t* cr = planes.codes + r * planes.codes_row;
        uint8_t* sr = planes.scale + r * planes.scale_row;
        uint8_t* zr = planes.zero ? planes.zero + r * planes.zero_row : nullptr;
        for (int64_t j = 0; j < nc; ++j) {
            const size_t k = (size_t)(r * nc + j);
            const BlockScale bs = rs[(size_t)r].block(j, bestA[k], bestB[k]);
            grid_store_scale(g, bs, sr, zr, j);
            const int64_t c1 = std::min((j + 1) * bc, K);
            for (int64_t c = j * bc; c < c1; ++c) {
                bool cl = false;
                const size_t i = (size_t)(r * K + c);
                grid_store_code(g, cr, c, code_v(g, bs, w[i], bestV[i], &cl));
            }
        }
    }
}

namespace {

const RadQuantOption kAutoRoundOptions[] = {
    LQ_GRID_OPTIONS,
    LQ_CALIB_OPTION("the imatrix's diagonal stands in, and with neither the weight is rounded to "
                    "nearest"),
    { "iters", RAD_P_INT, "200", "descent steps; each costs N K^2 multiply-adds" },
    { "lr", RAD_P_F64, "0", "the step, decaying linearly to zero; 0 is 1 / iters" },
};

int ar_grid(const RadParam* o, int n_o, const RadQuantWeight* w, Grid* g) {
    const int st = grid_of(o, n_o, w, g);
    if (st != RAD_OK) return st;
    const std::string no = autoround_refusal(*g);
    if (!no.empty()) return refuse(w, ("autoround: " + no).c_str());
    if (rad_param_geti(o, n_o, "iters", 200) < 0) return refuse(w, "iters= must be >= 0");
    return RAD_OK;
}

int ar_encoding(const RadParam* o, int n_o, const RadQuantWeight* w, RadEncoding* out) {
    if (declines(w)) return RAD_E_UNSUPPORTED;
    Grid g;
    const int st = ar_grid(o, n_o, w, &g);
    if (st != RAD_OK) return st;
    *out = grid_encoding(g);
    return RAD_OK;
}

/* Rows descend independently, so any block size works; 512 keeps the state to a few hundred MB at
 * the widest K while still filling the threads. */
int64_t ar_row_block(const RadParam*, int, const RadQuantWeight*) { return 512; }

int ar_quantize(const RadParam* o, int n_o, const RadQuantWeight* w, float* src, int64_t row0,
                int64_t rows, void* const* planes) {
    Grid g;
    const int st = ar_grid(o, n_o, w, &g);
    if (st != RAD_OK) return st;
    const int64_t K = w->cols;
    grid_transform_rows(g, src, rows, K);
    const char* dir = rad_param_gets(o, n_o, "calib", nullptr);
    const int iters = (int)rad_param_geti(o, n_o, "iters", 200);
    double lr = rad_param_getf(o, n_o, "lr", 0.0);
    if (!(lr > 0.0)) lr = iters > 0 ? 1.0 / iters : 0.0;
    const Gram Hg = calib_gram(dir, w->name, K, g.fwht, "autoround");
    const float* H = Hg ? Hg->data() : nullptr;
    const float* diag = (!H && !g.fwht) ? w->importance : nullptr;
    const Planes p = grid_planes(g, K, planes);
    if (!H && !diag) {
        /* With no error weighting, round-to-nearest is optimal. */
        return grid_rtn(g, src, row0, rows, K, p, nullptr);
    }
    if (!H)
        std::fprintf(stderr, "libquant: autoround: '%s' has no Hessian; the imatrix's diagonal "
                             "stands in for it\n", w->name);
    autoround_rows(g, src, rows, K, H, diag, iters, lr, p);
    return RAD_OK;
}

}  /* namespace */

const RadQuantizerInfo& autoround_info() {
    static const RadQuantizerInfo q = {
        "autoround",
        "rounding by signed gradient descent: an offset a value and a range factor a block, "
        "learned against the layer's input Hessian through the straight-through rounding, the "
        "best of every row kept",
        kAutoRoundOptions, (int)(sizeof kAutoRoundOptions / sizeof kAutoRoundOptions[0]),
        ar_encoding, ar_row_block, ar_quantize, nullptr,
    };
    return q;
}

}  /* namespace lq */
