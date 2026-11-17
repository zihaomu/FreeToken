/* lq_paro.cpp -- ParoQuant: pairwise rotation quantisation (Liang, Chen, Han, Liu; ICLR 2026).
 *
 * An outlier channel stretches the range of every scale group it sits in. ParoQuant transforms each
 * row's input channels before the grid (a channel-wise scaling, then `stages` rounds of independent
 * Givens rotations, each a set of disjoint channel pairs within one group of `rot_group` channels)
 * so a group's values even out, and learns the angles and scales against the output error. The
 * reader undoes the transform on the activation side: the encoding's transform is "givens"
 * (abi/rad_encoding.h), T = D . G_1 ... G_S over row vectors, with scales, pairs and angles stored
 * as its tables.
 *
 * Stage one fits the transform with Adam at `lr` (cosine-decayed to lr/20, no weight decay) over
 * `steps` steps, each on the next `sample_rows` rows; angles start at zero and scales at one, as in
 * the paper. Each step quantises through the grid and the gradient reaches the transform by the
 * straight-through estimator: w_hat = w + E T^-1 with the quantisation error E held, so a step
 * moves T^-1 to send E where the input does not look.
 * Stage two replaces the paper's fine-tuning of rounding and scales with AutoRound
 * (lq_autoround.h) on the transformed weight against T^-1 H T^-T, for `iters` steps; 0 rounds the
 * transformed weight to nearest.
 *
 * Pairs are drawn at random within each rotation group: a channel is paired at most once per stage
 * and, where the group allows, a pair is used at most once over all stages (the paper's diverse
 * independent rotations). The generator is seeded by the weight's name, so converts are repeatable.
 *
 * Output error: the Hessian's quadratic form (lq_hess.h) in the weight's own domain, the
 * layer-local form of the paper's loss; with no Hessian but an imatrix, its diagonal. With neither,
 * the transform stays the identity (reported on stderr).
 *
 * Determinism: gradients are summed over rows in row order regardless of thread count, so the
 * stored file depends only on the weight and the calibration.
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

namespace {

const RadQuantOption kParoOptions[] = {
    LQ_GRID_OPTIONS,
    LQ_CALIB_OPTION("the imatrix's diagonal stands in, and with neither the transform is the "
                    "identity and the weight is rounded to nearest"),
    { "stages", RAD_P_INT, "8", "rounds of independent Givens rotations" },
    { "rot_group", RAD_P_INT, "0",
      "channels a rotation's pairs stay within: 0 is the grid's group, or 128 without one" },
    { "steps", RAD_P_INT, "200", "stage one's Adam steps" },
    { "lr", RAD_P_F64, "0.05", "stage one's learning rate, for the angles and the scales" },
    { "sample_rows", RAD_P_INT, "512", "rows a stage-one step measures the output error over" },
    { "iters", RAD_P_INT, "100",
      "stage two's AutoRound steps on the transformed weight (an integer grid with row blocks); "
      "0 rounds it to nearest" },
};

struct Paro {
    Grid g;
    int stages = 8;
    int64_t gr = 128;
};

int paro_setup(const RadParam* o, int n_o, const RadQuantWeight* w, Paro* p) {
    const int st = grid_of(o, n_o, w, &p->g);
    if (st != RAD_OK) return st;
    if (p->g.fwht) return refuse(w, "paroquant writes its own transform, so it takes no "
                                    "transform=");
    p->stages = (int)rad_param_geti(o, n_o, "stages", 8);
    if (p->stages < 1 || p->stages > 64) return refuse(w, "stages= must be 1 to 64");
    p->gr = rad_param_geti(o, n_o, "rot_group", 0);
    if (p->gr <= 0) p->gr = (p->g.bc > 0 && w->cols % p->g.bc == 0) ? p->g.bc : 128;
    if (p->gr < 2 || (p->gr & 1)) return refuse(w, "rot_group= must be an even channel count");
    if (w->cols % p->gr) {
        char m[160];
        std::snprintf(m, sizeof m, "paroquant: %lld columns are not a whole number of %lld-channel "
                      "rotation groups", (long long)w->cols, (long long)p->gr);
        return refuse(w, m);
    }
    return RAD_OK;
}

int paro_encoding(const RadParam* o, int n_o, const RadQuantWeight* w, RadEncoding* out) {
    if (declines(w)) return RAD_E_UNSUPPORTED;
    Paro p;
    const int st = paro_setup(o, n_o, w, &p);
    if (st != RAD_OK) return st;
    *out = grid_encoding(p.g);
    rad_enc_copy_str(out->transform, "givens");
    if (rad_enc_add_table(out, "t.scale", RAD_F32, 1, w->cols) < 0 ||
        rad_enc_add_table(out, "t.pairs", RAD_I32, p.stages, w->cols) < 0 ||
        rad_enc_add_table(out, "t.angle", RAD_F32, p.stages, w->cols / 2) < 0)
        return refuse(w, "the encoding has no room for the rotation's tables");
    return RAD_OK;
}

int64_t paro_row_block(const RadParam*, int, const RadQuantWeight*) { return 0; }

uint64_t splitmix(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* The pairs, [stages x K]: stage s pairs columns pairs[s][2p] and pairs[s][2p+1]. */
std::vector<int32_t> draw_pairs(const char* name, int stages, int64_t K, int64_t gr) {
    std::vector<int32_t> pairs((size_t)(stages * K));
    for (int64_t g0 = 0; g0 < K; g0 += gr) {
        std::vector<uint8_t> used((size_t)(gr * gr), 0);
        for (int s = 0; s < stages; ++s) {
            uint64_t st = name_seed(name) ^ ((uint64_t)(s + 1) * 0xD1B54A32D192ED03ull) ^
                          ((uint64_t)g0 * 0x8CB92BA72F3D8DD7ull);
            std::vector<int> order((size_t)gr);
            for (int64_t i = 0; i < gr; ++i) order[(size_t)i] = (int)i;
            for (int64_t i = gr - 1; i > 0; --i)
                std::swap(order[(size_t)i], order[(size_t)(splitmix(&st) % (uint64_t)(i + 1))]);
            std::vector<uint8_t> paired((size_t)gr, 0);
            int64_t p = 0;
            for (int64_t x = 0; x < gr; ++x) {
                const int a = order[(size_t)x];
                if (paired[(size_t)a]) continue;
                int b = -1;
                for (int64_t y = x + 1; y < gr && b < 0; ++y) {          /* an unused pair */
                    const int c = order[(size_t)y];
                    if (!paired[(size_t)c] && !used[(size_t)(a * gr + c)]) b = c;
                }
                for (int64_t y = x + 1; y < gr && b < 0; ++y)            /* else any partner */
                    if (!paired[(size_t)order[(size_t)y]]) b = order[(size_t)y];
                paired[(size_t)a] = paired[(size_t)b] = 1;
                used[(size_t)(a * gr + b)] = used[(size_t)(b * gr + a)] = 1;
                pairs[(size_t)(s * K + g0 + 2 * p)] = (int32_t)(g0 + a);
                pairs[(size_t)(s * K + g0 + 2 * p + 1)] = (int32_t)(g0 + b);
                ++p;
            }
        }
    }
    return pairs;
}

/* The transform's tables, and its forward and inverse over one row. */
struct Rot {
    int S = 0;
    int64_t K = 0;
    std::vector<int32_t> pairs;   /* [S x K] */
    std::vector<float> angle;     /* [S x K/2] */
    std::vector<float> scale;     /* [K] */

    /* v = w . D . G_1 ... G_S, in place. */
    void forward(float* x) const {
        for (int64_t k = 0; k < K; ++k) x[k] *= scale[(size_t)k];
        for (int s = 0; s < S; ++s)
            for (int64_t p = 0; p < K / 2; ++p) {
                const int32_t a = pairs[(size_t)(s * K + 2 * p)], b = pairs[(size_t)(s * K + 2 * p + 1)];
                const float t = angle[(size_t)(s * (K / 2) + p)], c = std::cos(t), sn = std::sin(t);
                const float xa = x[a], xb = x[b];
                x[a] = xa * c + xb * sn;
                x[b] = xb * c - xa * sn;
            }
    }
    /* y = e . T^-1, keeping every intermediate in ys[k] (after the k-th inverse stage; k = 0 is
     * e). The result, divided by the scales, goes to `out`. */
    void inverse(const float* e, float* ys, float* out) const {
        std::memcpy(ys, e, (size_t)K * sizeof(float));
        for (int k = 1; k <= S; ++k) {
            const int s = S - k;
            const float* y0 = ys + (size_t)(k - 1) * K;
            float* y1 = ys + (size_t)k * K;
            std::memcpy(y1, y0, (size_t)K * sizeof(float));
            for (int64_t p = 0; p < K / 2; ++p) {
                const int32_t a = pairs[(size_t)(s * K + 2 * p)], b = pairs[(size_t)(s * K + 2 * p + 1)];
                const float t = angle[(size_t)(s * (K / 2) + p)], c = std::cos(t), sn = std::sin(t);
                y1[a] = c * y0[a] - sn * y0[b];
                y1[b] = sn * y0[a] + c * y0[b];
            }
        }
        const float* yl = ys + (size_t)S * K;
        for (int64_t k = 0; k < K; ++k) out[k] = yl[k] / scale[(size_t)k];
    }
};

/* The output error's gradient factor: G = D H, or D times the diagonal. */
void err_grad(const float* D, int64_t R, int64_t K, const float* H, const float* diag, float* G) {
    if (H) { mat_hess(D, R, K, H, G); return; }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < R; ++r)
        for (int64_t k = 0; k < K; ++k) G[r * K + k] = D[r * K + k] * diag[k];
}

/* Stage one: fit the transform. */
void fit(const Paro& p, const float* w, int64_t N, int64_t K, const float* H, const float* diag,
         int steps, double lr, int64_t want, Rot* rot) {
    const Grid& g = p.g;
    const int S = rot->S;
    const int64_t P = (int64_t)S * (K / 2) + K;            /* the angles, then the scales */
    const int64_t unit = g.br > 0 ? g.br : N;
    const int64_t nunits = (N + unit - 1) / unit;
    const int64_t take = std::min<int64_t>(nunits, std::max<int64_t>(1, want / unit));
    std::vector<double> m((size_t)P, 0.0), v((size_t)P, 0.0);
    double norm = 0.0;                                     /* the first step's loss */
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8, lr_min = lr / 20.0;

    std::vector<float> ws, vs, fake, E, D, G, ys, part;
    for (int t = 0; t < steps; ++t) {
        /* This step's rows: the next `take` row units, wrapping. */
        std::vector<int64_t> us;
        for (int64_t i = 0; i < take; ++i) us.push_back((t * take + i) % nunits);
        std::sort(us.begin(), us.end());
        us.erase(std::unique(us.begin(), us.end()), us.end());
        const int64_t R = gather_units(w, N, K, unit, us, &ws);
        vs = ws;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t r = 0; r < R; ++r) rot->forward(&vs[(size_t)(r * K)]);
        fake.resize(vs.size());
        grid_fake(g, vs.data(), R, K, fake.data());
        E.resize(vs.size());
        for (size_t i = 0; i < vs.size(); ++i) E[i] = fake[i] - vs[i];
        D.resize(vs.size());
        ys.resize((size_t)(R * (S + 1) * K));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t r = 0; r < R; ++r)
            rot->inverse(&E[(size_t)(r * K)], &ys[(size_t)(r * (S + 1) * K)], &D[(size_t)(r * K)]);
        G.resize(vs.size());
        err_grad(D.data(), R, K, H, diag, G.data());
        double loss = 0.0;
        {
            std::vector<double> q((size_t)R);
            row_dots(D.data(), G.data(), R, K, q.data());
            for (double x : q) loss += x;
        }
        if (t == 0) norm = loss > 0.0 ? loss : 1.0;

        /* Backward pass, one row at a time into the row's own slot of `part`: dL/dD = 2 G,
         * D = y_S / scale, and each inverse stage (a, b) -> (c a - s b, s a + c b) gives
         * dL/dt = g_a (-y_b) + g_b y_a over its outputs and sends g back by the transpose. */
        part.assign((size_t)(R * P), 0.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t r = 0; r < R; ++r) {
            float* gp = &part[(size_t)(r * P)];
            std::vector<float> gg((size_t)K);
            const float* d = &D[(size_t)(r * K)];
            const float* gr = &G[(size_t)(r * K)];
            for (int64_t k = 0; k < K; ++k) {
                const float sc = rot->scale[(size_t)k];
                const float gk = 2.0f * gr[k];
                gp[(int64_t)S * (K / 2) + k] = gk * (-d[k] / sc);
                gg[(size_t)k] = gk / sc;
            }
            const float* yr = &ys[(size_t)(r * (S + 1) * K)];
            for (int k = S; k >= 1; --k) {
                const int s = S - k;
                const float* y1 = yr + (size_t)k * K;
                for (int64_t q = 0; q < K / 2; ++q) {
                    const int32_t a = rot->pairs[(size_t)(s * K + 2 * q)];
                    const int32_t b = rot->pairs[(size_t)(s * K + 2 * q + 1)];
                    const float th = rot->angle[(size_t)(s * (K / 2) + q)];
                    const float c = std::cos(th), sn = std::sin(th);
                    const float ga = gg[(size_t)a], gb = gg[(size_t)b];
                    gp[s * (K / 2) + q] = ga * (-y1[b]) + gb * y1[a];
                    gg[(size_t)a] = c * ga + sn * gb;
                    gg[(size_t)b] = -sn * ga + c * gb;
                }
            }
        }
        /* Sum over rows in row order, then take the Adam step. */
        const double lt = lr_min + 0.5 * (lr - lr_min) * (1.0 + std::cos(M_PI * t / steps));
        const double c1 = 1.0 - std::pow(b1, t + 1), c2 = 1.0 - std::pow(b2, t + 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < P; ++i) {
            double gsum = 0.0;
            for (int64_t r = 0; r < R; ++r) gsum += (double)part[(size_t)(r * P + i)];
            gsum /= norm;
            m[(size_t)i] = b1 * m[(size_t)i] + (1.0 - b1) * gsum;
            v[(size_t)i] = b2 * v[(size_t)i] + (1.0 - b2) * gsum * gsum;
            const double upd = lt * (m[(size_t)i] / c1) / (std::sqrt(v[(size_t)i] / c2) + eps);
            if (i < (int64_t)S * (K / 2)) {
                rot->angle[(size_t)i] = (float)(rot->angle[(size_t)i] - upd);
            } else {
                float& sc = rot->scale[(size_t)(i - (int64_t)S * (K / 2))];
                sc = std::min(20.0f, std::max(0.05f, (float)(sc - upd)));
            }
        }
        if (t == 0 || t + 1 == steps)
            std::fprintf(stderr, "libquant: paroquant: step %d of %d, output error %.6g of the "
                                 "first step's\n", t + 1, steps, loss / norm);
    }
}

/* T^-1 H T^-T by rotation group. T^-1 is block-diagonal over the groups, so its rows are formed
 * once by sending each unit vector through the inverse, and each block of H is multiplied by them
 * on both sides. */
std::vector<float> transformed_hessian(const Rot& rot, const float* H, const float* diag,
                                       int64_t K, int64_t gr) {
    const int64_t ng = K / gr;
    std::vector<float> M((size_t)(K * gr));                 /* row i: e_i . T^-1, its group's span */
    {
        std::vector<float> e((size_t)K, 0.0f), ys((size_t)((rot.S + 1) * K)), out((size_t)K);
        for (int64_t i = 0; i < K; ++i) {
            e[(size_t)i] = 1.0f;
            rot.inverse(e.data(), ys.data(), out.data());
            const int64_t g0 = (i / gr) * gr;
            for (int64_t j = 0; j < gr; ++j) M[(size_t)(i * gr + j)] = out[(size_t)(g0 + j)];
            e[(size_t)i] = 0.0f;
        }
    }
    std::vector<float> Hv((size_t)(K * K), 0.0f);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
#endif
    for (int64_t I = 0; I < ng; ++I)
        for (int64_t J = 0; J < ng; ++J) {
            if (!H && I != J) continue;
            std::vector<double> t1((size_t)(gr * gr), 0.0);   /* M_I . H_IJ */
            for (int64_t i = 0; i < gr; ++i)
                for (int64_t k = 0; k < gr; ++k) {
                    const double mik = M[(size_t)((I * gr + i) * gr + k)];
                    if (mik == 0.0) continue;
                    for (int64_t j = 0; j < gr; ++j) {
                        const double h = H ? (double)H[(I * gr + k) * K + J * gr + j]
                                           : (k == j ? (double)diag[I * gr + k] : 0.0);
                        t1[(size_t)(i * gr + j)] += mik * h;
                    }
                }
            for (int64_t i = 0; i < gr; ++i)
                for (int64_t j = 0; j < gr; ++j) {
                    double s = 0.0;
                    for (int64_t k = 0; k < gr; ++k)
                        s += t1[(size_t)(i * gr + k)] * (double)M[(size_t)((J * gr + j) * gr + k)];
                    Hv[(size_t)((I * gr + i) * K + J * gr + j)] = (float)s;
                }
        }
    return Hv;
}

int paro_quantize(const RadParam* o, int n_o, const RadQuantWeight* w, float* src, int64_t row0,
                  int64_t rows, void* const* planes) {
    Paro p;
    int st = paro_setup(o, n_o, w, &p);
    if (st != RAD_OK) return st;
    if (row0 != 0 || rows != w->rows) return RAD_E_SHAPE;
    const int64_t N = w->rows, K = w->cols;
    const char* dir = rad_param_gets(o, n_o, "calib", nullptr);
    const int steps = (int)std::max<long long>(0, rad_param_geti(o, n_o, "steps", 200));
    const double lr = rad_param_getf(o, n_o, "lr", 0.05);
    const int64_t want = std::max<long long>(1, rad_param_geti(o, n_o, "sample_rows", 512));
    const int iters = (int)std::max<long long>(0, rad_param_geti(o, n_o, "iters", 100));

    /* Domain 0: the grid carries no Hadamard (paro_setup refuses one), and the fitted rotation is
     * applied to the weight and the Hessian here, after the read. */
    const Gram Hg = calib_gram(dir, w->name, K, 0, "paroquant");
    const float* H = Hg ? Hg->data() : nullptr;
    const float* diag = H ? nullptr : w->importance;
    if (!H && diag)
        std::fprintf(stderr, "libquant: paroquant: '%s' has no Hessian; the imatrix's diagonal "
                             "stands in for it\n", w->name);
    if (!H && !diag)
        std::fprintf(stderr, "libquant: paroquant: '%s' has neither a Hessian nor an imatrix; its "
                             "transform is the identity\n", w->name);

    Rot rot;
    rot.S = p.stages;
    rot.K = K;
    rot.pairs = draw_pairs(w->name, p.stages, K, p.gr);
    rot.angle.assign((size_t)(p.stages * (K / 2)), 0.0f);
    rot.scale.assign((size_t)K, 1.0f);
    if ((H || diag) && steps > 0) fit(p, src, N, K, H, diag, steps, lr, want, &rot);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < N; ++r) rot.forward(src + r * K);

    const Planes pl = grid_planes(p.g, K, planes);
    const std::string no = autoround_refusal(p.g);
    if ((H || diag) && iters > 0 && !no.empty())
        std::fprintf(stderr, "libquant: paroquant: '%s': stage two is AutoRound, and %s; the "
                             "transformed weight is rounded to nearest\n", w->name, no.c_str());
    if ((H || diag) && iters > 0 && no.empty()) {
        const std::vector<float> Hv = transformed_hessian(rot, H, diag, K, p.gr);
        autoround_rows(p.g, src, N, K, Hv.data(), nullptr, iters, 1.0 / iters, pl);
    } else {
        st = grid_rtn(p.g, src, 0, N, K, pl, nullptr);
        if (st != RAD_OK) return st;
    }
    const int base = grid_encoding(p.g).n_planes;
    std::memcpy(planes[base], rot.scale.data(), rot.scale.size() * sizeof(float));
    std::memcpy(planes[base + 1], rot.pairs.data(), rot.pairs.size() * sizeof(int32_t));
    std::memcpy(planes[base + 2], rot.angle.data(), rot.angle.size() * sizeof(float));
    return RAD_OK;
}

}  /* namespace */

const RadQuantizerInfo& paroquant_info() {
    static const RadQuantizerInfo q = {
        "paroquant",
        "pairwise rotation: a channel scaling and rounds of Givens rotations of the input channels, "
        "learned against the layer's input Hessian through the straight-through quantiser, then "
        "AutoRound on the rotated weight; the transform is the encoding's givens tables",
        kParoOptions, (int)(sizeof kParoOptions / sizeof kParoOptions[0]),
        paro_encoding, paro_row_block, paro_quantize, nullptr,
    };
    return q;
}

}  /* namespace lq */
