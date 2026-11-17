/* lq_gptq.cpp -- the Hessian side of GPTQ and the sequential loop. See lq_gptq.h. */
#include "lq_gptq.h"
#include "lq_gpu.h"
#include "lq_hess.h"

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace lq {

namespace {

/* H is damped, inverted and re-factorised into upper U with U^T U = H^-1, the form the sequential
 * loop wants: column i's error is divided by U[i][i] and spread over columns j > i with weight
 * U[i][j] (Frantar et al.'s formulation). Double throughout: a Cholesky is K^3/6 accumulations
 * deep, and f32 loses real conditioning on a matrix near-singular in the directions a rotation
 * flattened. */
#ifdef RAD_LQ_LAPACK
/* LAPACK, column-major, Fortran calling convention (OpenBLAS). */
extern "C" void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
extern "C" void dtrtri_(const char* uplo, const char* diag, const int* n, double* a, const int* lda,
                        int* info);

/* H (damped, row-major) -> U upper with U^T U = H^-1, in f32, by one Cholesky and one triangular
 * inverse instead of a Cholesky, an inverse and a second Cholesky. With J the reversal, J H J =
 * L L^T makes H = M M^T for the upper M = J L J, so H^-1 = M^-T M^-1 and U = M^-1 = J L^-1 J.
 * Reversing a symmetric matrix is reversing its storage, so J H J is `a` read backwards; the factor
 * and the inverse are LAPACK's, in column-major lower, and U[i][j] = L^-1[K-1-i][K-1-j] is read
 * from there. Half the f64 work of the three-step form. */
bool inv_upper_lapack(std::vector<double>& a, int64_t k, float* u) {
    std::reverse(a.begin(), a.end());
    const int n = (int)k;
    int info = 0;
    dpotrf_("L", &n, a.data(), &n, &info);
    if (info != 0) return false;
    dtrtri_("L", "N", &n, a.data(), &n, &info);
    if (info != 0) return false;
    const double* l = a.data();
    bool ok = true;
#pragma omp parallel for schedule(static) reduction(&& : ok)
    for (int64_t i = 0; i < k; ++i) {
        float* ur = u + i * k;
        for (int64_t j = 0; j < i; ++j) ur[j] = 0.0f;
        /* Column-major (r, c) is l[r + c * k]; here r = k-1-i and c = k-1-j. */
        for (int64_t j = i; j < k; ++j) ur[j] = (float)l[(k - 1 - i) + (k - 1 - j) * k];
        ok = ok && ur[i] > 0.0f && std::isfinite(ur[i]);
    }
    return ok;
}

#endif  /* RAD_LQ_LAPACK */

/* The single-threaded loops: the reference, and the only path without OpenBLAS. */
bool chol_lower(std::vector<double>& a, int64_t k) {
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j <= i; ++j) {
            double s = a[(size_t)(i * k + j)];
            for (int64_t t = 0; t < j; ++t) s -= a[(size_t)(i * k + t)] * a[(size_t)(j * k + t)];
            if (i == j) {
                if (!(s > 0.0)) return false;
                a[(size_t)(i * k + i)] = std::sqrt(s);
            } else {
                a[(size_t)(i * k + j)] = s / a[(size_t)(j * k + j)];
            }
        }
        for (int64_t j = i + 1; j < k; ++j) a[(size_t)(i * k + j)] = 0.0;
    }
    return true;
}

/* A = L L^T, factored in place -> A^-1, full and symmetric: L^-1 first, then L^-T L^-1. */
void inv_from_chol(std::vector<double>& l, int64_t k) {
    for (int64_t j = 0; j < k; ++j) {
        l[(size_t)(j * k + j)] = 1.0 / l[(size_t)(j * k + j)];
        for (int64_t i = j + 1; i < k; ++i) {
            double s = 0.0;
            for (int64_t t = j; t < i; ++t) s += l[(size_t)(i * k + t)] * l[(size_t)(t * k + j)];
            l[(size_t)(i * k + j)] = -s / l[(size_t)(i * k + i)];
        }
    }
    std::vector<double> out((size_t)(k * k), 0.0);
    for (int64_t i = 0; i < k; ++i)
        for (int64_t j = 0; j <= i; ++j) {
            double s = 0.0;
            for (int64_t t = i; t < k; ++t) s += l[(size_t)(t * k + i)] * l[(size_t)(t * k + j)];
            out[(size_t)(i * k + j)] = s;
            out[(size_t)(j * k + i)] = s;
        }
    l.swap(out);
}

/* A -> U upper with A = U^T U. */
bool chol_upper(std::vector<double>& a, int64_t k) {
    for (int64_t i = 0; i < k; ++i) {
        double d = a[(size_t)(i * k + i)];
        for (int64_t t = 0; t < i; ++t) {
            const double v = a[(size_t)(t * k + i)];
            d -= v * v;
        }
        if (!(d > 0.0)) return false;
        const double u = std::sqrt(d);
        a[(size_t)(i * k + i)] = u;
        for (int64_t j = i + 1; j < k; ++j) {
            double s = a[(size_t)(i * k + j)];
            for (int64_t t = 0; t < i; ++t) s -= a[(size_t)(t * k + i)] * a[(size_t)(t * k + j)];
            a[(size_t)(i * k + j)] = s / u;
        }
        for (int64_t j = 0; j < i; ++j) a[(size_t)(i * k + j)] = 0.0;
    }
    return true;
}

#ifdef RAD_LQ_LAPACK
/* LAPACK where OpenBLAS was found; RAD_LQ_LAPACK=0 takes the loops instead, the reference the
 * LAPACK path is checked against (quant_algo_test). */
bool use_lapack() {
    const char* e = std::getenv("RAD_LQ_LAPACK");
    return !(e && e[0] == '0');
}
#endif

/* Rows a tile of GPTQ's loop: the tail update keeps a tile's columns in registers across the
 * block's 128 errors. */
constexpr int64_t kTileRows = 8;
constexpr int64_t kTileCols = 32;

/* w[r][j] -= e[r][t] * u[t][j] for t over the block [b0, b1), j over [b1, K), r over `rows`
 * (<= kTileRows) rows of `w` (row stride K) and `e` (row stride 128). Each element takes its terms
 * one at a time in ascending t, a product then a subtraction, as a row running alone would apply
 * them, so the result does not depend on the vector width, the tile or the threads. Built for each
 * of these instruction sets and chosen at load. */
__attribute__((target_clones("avx512f", "avx2", "default")))
void gptq_tail(float* w, const float* e, int64_t rows, const float* u, int64_t K, int64_t b0,
               int64_t b1) {
    int64_t j0 = b1;
    if (rows == kTileRows) {
        for (; j0 + kTileCols <= K; j0 += kTileCols) {
            float acc[kTileRows][kTileCols];
            for (int64_t r = 0; r < kTileRows; ++r)
                for (int64_t c = 0; c < kTileCols; ++c) acc[r][c] = w[r * K + j0 + c];
            for (int64_t t = b0; t < b1; ++t) {
                const float* ut = u + t * K + j0;
                for (int64_t r = 0; r < kTileRows; ++r) {
                    const float er = e[r * 128 + (t - b0)];
                    for (int64_t c = 0; c < kTileCols; ++c) acc[r][c] -= er * ut[c];
                }
            }
            for (int64_t r = 0; r < kTileRows; ++r)
                for (int64_t c = 0; c < kTileCols; ++c) w[r * K + j0 + c] = acc[r][c];
        }
    }
    for (int64_t r = 0; r < rows; ++r) {
        float* wr = w + r * K;
        for (int64_t t = b0; t < b1; ++t) {
            const float er = e[r * 128 + (t - b0)];
            const float* ut = u + t * K;
            for (int64_t j = j0; j < K; ++j) wr[j] -= er * ut[j];
        }
    }
}

/* Layer factors kept at once. */
constexpr size_t kFactors = 4;

/* One layer factor: built by the first caller under its own lock, which later callers wait on. */
struct Slot {
    std::mutex mu;
    bool done = false;
    Gram g;                          /* null = looked and found nothing */
    std::vector<int32_t> order;      /* gptq_factor_ordered's column order */
};

struct Cache {
    std::mutex mu;                   /* the map, held only to find or add a slot */
    std::map<std::string, std::shared_ptr<Slot>> f;
    std::deque<std::string> order;   /* f's keys, oldest first */
};

Cache& cache() {
    static Cache c;
    return c;
}

}  /* namespace */

std::vector<int32_t> gptq_act_order(const char* dir, const char* name, int64_t K, int64_t domain) {
    std::vector<int32_t> perm((size_t)K);
    for (int64_t i = 0; i < K; ++i) perm[(size_t)i] = (int32_t)i;
    if (!dir || !*dir || !name || !*name || K <= 0) return perm;
    std::vector<double> acc;
    if (!calib_sum(dir, calib_resolve(dir, name, K), K, domain, &acc)) return perm;
    /* Largest input energy first, ties in column order. The recursion absorbs error best early,
     * so the costliest columns are quantised while every later column can still take it. */
    std::stable_sort(perm.begin(), perm.end(), [&](int32_t x, int32_t y) {
        return acc[(size_t)x * (size_t)K + (size_t)x] > acc[(size_t)y * (size_t)K + (size_t)y];
    });
    return perm;
}

namespace {
Gram factor_impl(const char* dir, const char* name, int64_t K, int64_t domain, double damp,
                 const int32_t* perm, std::vector<int32_t>* order);
}  /* namespace */

Gram gptq_factor(const char* dir, const char* name, int64_t K, int64_t domain, double damp,
                 const int32_t* perm) {
    return factor_impl(dir, name, K, domain, damp, perm, nullptr);
}

Gram gptq_factor_ordered(const char* dir, const char* name, int64_t K, int64_t domain, double damp,
                         std::vector<int32_t>* order) {
    return factor_impl(dir, name, K, domain, damp, nullptr, order);
}

namespace {
Gram factor_impl(const char* dir, const char* name, int64_t K, int64_t domain, double damp,
                 const int32_t* perm, std::vector<int32_t>* order) {
    if (!dir || !*dir || !name || !*name || K <= 0) return nullptr;
    const std::string stem = calib_resolve(dir, name, K);
    const bool layer = stem == calib_stem(name);
    char dk[32];
    std::snprintf(dk, sizeof dk, "%.17g", damp);
    const std::string key = std::string(dir) + "|" + stem + "|" + std::to_string(K) + "|" +
                            std::to_string(domain) + "|" + dk + (perm ? "|act" : "") +
                            (order ? "|ord" : "");

    /* A layer's factor is shared by its experts, so it is built once and cached, under a lock of
     * its own: weights quantised side by side (rad-convert --gpus) factor different layers at
     * once. An expert's own is built for its one call outside any lock, so experts factor in
     * parallel. A few layer factors are kept, the oldest dropped first: a dense model reads each
     * layer's once, and a 21504-wide factor is 1.85 GB in f32 (every one kept OOM-killed
     * TranslateGemma 27B's conversion at layer 25). A MoE layer's experts run in parallel and
     * alternate between its gate_up and down stems, so one kept factor would be recomputed over
     * and over. */
    Cache& c = cache();
    std::shared_ptr<Slot> slot;
    std::unique_lock<std::mutex> lk;
    if (layer) {
        {
            std::lock_guard<std::mutex> g(c.mu);
            auto it = c.f.find(key);
            if (it == c.f.end()) {
                while (c.order.size() >= kFactors) {
                    c.f.erase(c.order.front());
                    c.order.pop_front();
                }
                it = c.f.emplace(key, std::make_shared<Slot>()).first;
                c.order.push_back(key);
            }
            slot = it->second;
        }
        lk = std::unique_lock<std::mutex>(slot->mu);
        if (slot->done) {
            if (order) *order = slot->order;
            return slot->g;
        }
    }
    auto keep = [&](Gram u) {
        if (slot) {
            slot->g = u;
            slot->done = true;
            if (order) slot->order = *order;
        }
        return u;
    };
    if (order) {
        order->resize((size_t)K);
        for (int64_t i = 0; i < K; ++i) (*order)[(size_t)i] = (int32_t)i;
    }

    std::vector<double> acc;
    const int found = calib_sum(dir, stem, K, domain, &acc);
    if (!found) {
        std::fprintf(stderr, "libquant: no calibration for '%s' in %s (looked for gram.r*.%s.bin); "
                             "rounding it to nearest\n", name, dir, stem.c_str());
        return keep(nullptr);
    }
    /* gptq_factor_ordered: the order from the same sum (gptq_act_order's: descending diagonal,
     * ties in column order), read once for both. */
    if (order) {
        std::stable_sort(order->begin(), order->end(), [&](int32_t x, int32_t y) {
            return acc[(size_t)x * (size_t)K + (size_t)x] > acc[(size_t)y * (size_t)K + (size_t)y];
        });
        perm = order->data();
    }
    /* Under act order the factor is of P H P^T, since the recursion runs over columns in
     * quantisation order. */
    if (perm) {
        std::vector<double> p((size_t)(K * K));
        for (int64_t i = 0; i < K; ++i)
            for (int64_t j = 0; j < K; ++j)
                p[(size_t)(i * K + j)] = acc[(size_t)perm[i] * (size_t)K + (size_t)perm[j]];
        acc.swap(p);
    }

    double tr = 0.0;
    int64_t dead = 0;
    for (int64_t i = 0; i < K; ++i) {
        if (!(acc[(size_t)(i * K + i)] > 0.0)) ++dead;
        tr += acc[(size_t)(i * K + i)];
    }
    /* A dead column is a channel the corpus never exercised: its diagonal is zero, the Cholesky
     * would fail on it, and the calibration says nothing about it. It gets the mean diagonal, so
     * the recursion does not spend other columns' error on it. */
    const double mean = tr / (double)K;
    const double lam  = damp * mean;
    for (int64_t i = 0; i < K; ++i) {
        double& d = acc[(size_t)(i * K + i)];
        if (!(d > 0.0)) d = mean;
        d += lam;
    }
    if (dead)
        std::fprintf(stderr, "libquant: %lld of %lld columns of '%s' were never exercised by the "
                             "calibration corpus\n", (long long)dead, (long long)K, stem.c_str());
    auto u = std::make_shared<std::vector<float>>((size_t)(K * K));
#ifdef RAD_LQ_LAPACK
    if (use_lapack()) {
        if (!inv_upper_lapack(acc, K, u->data())) {
            std::fprintf(stderr, "libquant: the Hessian for '%s' is not positive definite even "
                                 "damped; rounding it to nearest\n", stem.c_str());
            return keep(nullptr);
        }
        if (layer)
            std::fprintf(stderr, "libquant: GPTQ quantises '%s' against %d calibration file(s), "
                                 "K=%lld\n", stem.c_str(), found, (long long)K);
        return keep(u);
    }
#endif
    if (!chol_lower(acc, K)) {
        std::fprintf(stderr, "libquant: the Hessian for '%s' is not positive definite even "
                             "damped; rounding it to nearest\n", stem.c_str());
        return keep(nullptr);
    }
    inv_from_chol(acc, K);
    if (!chol_upper(acc, K)) {
        std::fprintf(stderr, "libquant: the inverse Hessian for '%s' would not factor; rounding "
                             "it to nearest\n", stem.c_str());
        return keep(nullptr);
    }
    for (size_t i = 0; i < u->size(); ++i) (*u)[i] = (float)acc[i];
    if (layer)
        std::fprintf(stderr, "libquant: GPTQ quantises '%s' against %d calibration file(s), "
                             "K=%lld\n", stem.c_str(), found, (long long)K);
    return keep(u);
}
}  /* namespace */

void gptq_tcq_dyn_scales(const Grid& g, const float* w, int64_t K, Scales& scales, int64_t r0,
                         int64_t r1, int64_t b0, int64_t b1) {
    /* Rows are independent. The host loop calls this per run inside its own parallel loop; the
     * card's driver calls it once for all its rows. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 64) if (!omp_in_parallel() && r1 - r0 > 256)
#endif
    for (int64_t r = r0; r < r1; ++r) {
        BlockScale b = grid_block_scale(g, w, K, r, r + 1, b0, b1, nullptr);
        if (g.lv2) {
            const int64_t i2 = g.br2 > 0 ? r / g.br2 : 0, j2 = g.bc2 > 0 ? b0 / g.bc2 : 0;
            b = grid_tcq_block_lv2(g, b.s, scales.s2[(size_t)(i2 * scales.nc2 + j2)]);
        }
        scales.b[(size_t)(r * scales.nc + b0 / 128)] = b;
    }
}

void gptq_tcq_col_scales(const Grid& g, const Scales& scales, int64_t n0, int64_t n, int64_t col,
                         float* s, float* rt) {
    const float cf = scales.colf.empty() ? 1.0f : scales.colf[(size_t)(col / 128)];
    for (int64_t r = 0; r < n; ++r) s[r] = scales.at(n0 + r, col).s * cf;
    /* tcq_rb: each row's values on its row block's grid (Grid::tcq_rb). n0 is a multiple of
     * tcq_rows, which tcq_rb divides, so the blocks are the GEMM's. */
    if (g.tcq_rb)
        for (int64_t r0 = 0; r0 < n; r0 += g.tcq_rb) {
            const int64_t r1 = std::min(n, r0 + g.tcq_rb);
            float bm = 0.0f;
            for (int64_t r = r0; r < r1; ++r) bm = std::max(bm, s[r]);
            for (int64_t r = r0; r < r1; ++r) {
                rt[r] = bm > 0.0f ? s[r] / bm : 0.0f;
                s[r] = bm;
            }
        }
}

namespace {
/* gptq_quantise over a trellis grid. A trellis runs down a column, so the loop turns: each run of
 * tcq_rows rows (a tile, and the parallelism) takes the block's columns in order, the trellis
 * choosing that column's codes for all the tile's rows at once; every row then feeds its error on
 * to the block's later columns as a row alone would, and the tail takes the block's errors as
 * before. All rows of a column share its weight in the Hessian's metric, so the trellis's own
 * squared error is GPTQ's objective for the column. lq_tcq_gpu.hip runs the same loop on the
 * cards. */
void gptq_quantise_tcq(const Grid& g, float* w, int32_t* codes, Scales& scales, int64_t N,
                       int64_t K, const float* u, bool dynamic) {
    const int64_t T = g.tcq_rows, B = 128;
    std::vector<float> err(u ? (size_t)(N * B) : 0);
    for (int64_t b0 = 0; b0 < K; b0 += B) {
        const int64_t b1 = (b0 + B < K) ? b0 + B : K;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int64_t n0 = 0; n0 < N; n0 += T) {
            const int64_t n1 = n0 + T < N ? n0 + T : N, n = n1 - n0;
            thread_local TcqWork work;
            thread_local std::vector<float> x, s, dq, rt;
            thread_local std::vector<int32_t> cs;
            x.resize((size_t)n); s.resize((size_t)n); dq.resize((size_t)n); cs.resize((size_t)n);
            rt.resize(g.tcq_rb ? (size_t)n : 0);
            if (dynamic) gptq_tcq_dyn_scales(g, w, K, scales, n0, n1, b0, b1);
            for (int64_t i = b0; i < b1; ++i) {
                for (int64_t r = 0; r < n; ++r) x[(size_t)r] = w[(size_t)((n0 + r) * K + i)];
                gptq_tcq_col_scales(g, scales, n0, n, i, s.data(), rt.data());
                grid_tcq_run(g, x.data(), s.data(), n, cs.data(), dq.data(), work,
                             g.tcq_rb ? rt.data() : nullptr);
                for (int64_t r = 0; r < n; ++r) {
                    float* wr = w + (size_t)((n0 + r) * K);
                    codes[(size_t)((n0 + r) * K + i)] = cs[(size_t)r];
                    wr[i] = dq[(size_t)r];
                    if (!u) continue;
                    const float* ur = u + (size_t)(i * K);
                    const float e = (x[(size_t)r] - dq[(size_t)r]) / ur[i];
                    err[(size_t)((n0 + r) * B + (i - b0))] = e;
                    for (int64_t j = i + 1; j < b1; ++j) wr[j] -= e * ur[j];
                }
            }
            if (u)
                for (int64_t m0 = n0; m0 < n1; m0 += kTileRows)
                    gptq_tail(w + (size_t)(m0 * K), err.data() + (size_t)(m0 * B),
                              (m0 + kTileRows < n1 ? m0 + kTileRows : n1) - m0, u, K, b0, b1);
        }
    }
}
}  /* namespace */

bool gptq_dynamic_ok(const Grid& g, const Scales& s) {
    return (!g.lv2 || g.tcq_L) && g.rule != Grid::FIXED && s.br == 1 && s.bc == 128;
}

void gptq_tcq_colf(const float* w, Scales& scales, int64_t N, int64_t K) {
    /* Block kb's factor: the root of its share of the weight's energy over the mean block's; then
     * each row's scale, which was its whole row's rms, is taken against the factors, so its values
     * span the table the same over the row. */
    const int64_t nb = K / 128;
    std::vector<double> ms((size_t)(N * nb), 0.0), col((size_t)nb, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < N; ++r)
        for (int64_t b = 0; b < nb; ++b) {
            double ss = 0.0;
            for (int64_t c = b * 128; c < b * 128 + 128; ++c) ss += (double)w[r * K + c] * w[r * K + c];
            ms[(size_t)(r * nb + b)] = ss / 128.0;
        }
    double tot = 0.0;
    for (int64_t b = 0; b < nb; ++b) {
        for (int64_t r = 0; r < N; ++r) col[(size_t)b] += ms[(size_t)(r * nb + b)];
        tot += col[(size_t)b];
    }
    scales.colf.assign((size_t)nb, 1.0f);
    for (int64_t b = 0; b < nb; ++b)
        scales.colf[(size_t)b] = tot > 0.0 && col[(size_t)b] > 0.0
                                     ? (float)std::sqrt(col[(size_t)b] * (double)nb / tot) : 1.0f;
    for (int64_t r = 0; r < N; ++r) {
        double raw = 0.0, adj = 0.0;
        for (int64_t b = 0; b < nb; ++b) {
            const double c = scales.colf[(size_t)b];
            raw += ms[(size_t)(r * nb + b)];
            adj += ms[(size_t)(r * nb + b)] / (c * c);
        }
        BlockScale& bs = scales.b[(size_t)(r * scales.nc)];
        const float f = raw > 0.0 ? (float)std::sqrt(adj / raw) : 1.0f;
        bs.s *= f;
        bs.stored *= f;
        bs.inv = 1.0f / bs.s;
        bs.inv_d = 1.0 / (double)bs.s;
    }
}

int gptq_quantise(const Grid& g, float* w, int32_t* codes, Scales& scales, int64_t N,
                  int64_t K, const float* u, bool dynamic, std::string* why) {
    if (!w || !codes || N <= 0 || K <= 0) return RAD_OK;
    auto one = [&](int64_t n, int64_t i, float v) -> float {
        const BlockScale& b = scales.at(n, i);
        const int c = grid_code(g, b, v);
        codes[(size_t)(n * K + i)] = c;
        return grid_decode(g, b, c);
    };

    /* A vector grid's codeword for the `vdim` columns from i, into codes and `out`. */
    const int64_t V = g.vq.empty() ? 0 : g.vdim;
    auto vec = [&](int64_t n, int64_t i, const float* v, float* out) {
        const BlockScale& b = scales.at(n, i);
        const int c = grid_vq_code(g, b, v);
        for (int64_t j = 0; j < V; ++j) {
            codes[(size_t)(n * K + i + j)] = c;
            out[j] = g.vq[(size_t)(c * V + j)] * b.s;
        }
    };

    if (g.tcq_L && g.tcq_colf) {
        if (scales.bc < K || dynamic || g.lv2 || g.tcq_rb) {
            if (why) *why = "tcq_colf=1 takes one static scale a row (block=1x*, no dyn, scale2 or "
                            "row blocks)";
            return RAD_E_INVAL;
        }
        gptq_tcq_colf(w, scales, N, K);
    }
    if (g.tcq_L) {
        /* Only the last run can be short, and one shorter than a state's codes cannot hold its
         * seam: the decoder refuses it (grid_tcq_decode), so the encoder does too. */
        const int64_t c = (g.tcq_L + g.tcq_k - 1) / g.tcq_k;
        if (N > 0 && N - (N - 1) / g.tcq_rows * g.tcq_rows < c) {
            if (why) *why = "the last trellis run is " +
                            std::to_string(N - (N - 1) / g.tcq_rows * g.tcq_rows) + " row(s) of " +
                            std::to_string(N) + " (tcq_rows " + std::to_string(g.tcq_rows) +
                            "), shorter than the " + std::to_string(c) + " a seam state needs";
            return RAD_E_SHAPE;
        }
        if (gpu_enabled()) {
            std::string w1;
            const int st = gptq_tcq_gpu(g, w, codes, scales, N, K, u, dynamic, &w1);
            if (st != RAD_OK && why) *why = w1;
            return st;
        }
        gptq_quantise_tcq(g, w, codes, scales, N, K, u, dynamic);
        return RAD_OK;
    }
    if (!u) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t n = 0; n < N; ++n) {
            float* wr = w + (size_t)(n * K);
            if (V) {
                for (int64_t i = 0; i < K; i += V) vec(n, i, wr + i, wr + i);
                continue;
            }
            for (int64_t i = 0; i < K; ++i) wr[i] = one(n, i, wr[i]);
        }
        return RAD_OK;
    }

    /* Column blocks outside, row tiles inside. Rows are independent and are the parallelism; with
     * the block loop outside, every tile works on the same columns and the block's 128 x K slice
     * of `u` stays in cache. Within a tile each row runs the block's columns in order, then the
     * tile's tail (the columns after the block) takes the block's errors (gptq_tail). */
    const int64_t B = 128;
    std::vector<float> err((size_t)(N * B));
    for (int64_t b0 = 0; b0 < K; b0 += B) {
        const int64_t b1 = (b0 + B < K) ? b0 + B : K;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t n0 = 0; n0 < N; n0 += kTileRows) {
            const int64_t n1 = n0 + kTileRows < N ? n0 + kTileRows : N;
            for (int64_t n = n0; n < n1; ++n) {
                float* wr = w + (size_t)(n * K);
                float* er = err.data() + (size_t)(n * B);
                if (dynamic)
                    scales.b[(size_t)(n * scales.nc + b0 / 128)] =
                        grid_block_scale(g, w, K, n, n + 1, b0, b1, nullptr);
                /* A vector grid chooses each codeword at its first column from the values the
                 * error has moved the vector to; its columns then feed their error on in turn. */
                float vbuf[16];
                for (int64_t i = b0; i < b1; ++i) {
                    const float* ur = u + (size_t)(i * K);
                    const float d = ur[i];
                    const float v = wr[i];
                    if (V && (i - b0) % V == 0) vec(n, i, wr + i, vbuf);
                    const float dq = V ? vbuf[(i - b0) % V] : one(n, i, v);
                    wr[i] = dq;
                    /* d is U[i][i], strictly positive: the factor refuses otherwise. */
                    const float e = (v - dq) / d;
                    er[i - b0] = e;
                    for (int64_t j = i + 1; j < b1; ++j) wr[j] -= e * ur[j];
                }
            }
            gptq_tail(w + (size_t)(n0 * K), err.data() + (size_t)(n0 * B), n1 - n0, u, K, b0, b1);
        }
    }
    return RAD_OK;
}

namespace {
/* gptq_refine over a trellis grid: a trellis code moves the states of the rows after it, so a
 * column moves whole. Each run of tcq_rows rows takes its columns in order, searching each one
 * again against the exact descent target of its own values given every other column's (the
 * row's gradient at the column less the column's own term, over the diagonal), on the scales the
 * loop chose; the column's new codes are kept when the form, summed over the run's rows, strictly
 * falls. */
void gptq_tcq_refine(const Grid& g, const float* x, float* w, int32_t* codes, const Scales& scales,
                     int64_t N, int64_t K, const float* h, const float* ad, const float* dg,
                     float* e, float* gm, int passes) {
    const int64_t T = g.tcq_rows;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int64_t n0 = 0; n0 < N; n0 += T) {
        const int64_t n1 = n0 + T < N ? n0 + T : N, n = n1 - n0;
        TcqWork work;
        std::vector<float> t((size_t)n), s((size_t)n), dq((size_t)n), rt(g.tcq_rb ? (size_t)n : 0),
                           d((size_t)n);
        std::vector<int32_t> cs((size_t)n);
        for (int64_t r = n0; r < n1; ++r) {
            float* gr = gm + (size_t)(r * K);
            const float* er = e + (size_t)(r * K);
            for (int64_t j = 0; j < K; ++j) gr[j] += ad[(size_t)j] * er[j];
        }
        for (int pass = 0; pass < passes; ++pass) {
            int64_t moved = 0;
            for (int64_t j = 0; j < K; ++j) {
                const float hj = dg[(size_t)j];
                for (int64_t r = 0; r < n; ++r) {
                    const size_t ix = (size_t)((n0 + r) * K + j);
                    t[(size_t)r] = x[ix] - (gm[ix] - e[ix] * hj) / hj;
                }
                gptq_tcq_col_scales(g, scales, n0, n, j, s.data(), rt.data());
                grid_tcq_run(g, t.data(), s.data(), n, cs.data(), dq.data(), work,
                             g.tcq_rb ? rt.data() : nullptr);
                double delta = 0.0;
                bool same = true;
                for (int64_t r = 0; r < n; ++r) {
                    const size_t ix = (size_t)((n0 + r) * K + j);
                    same &= cs[(size_t)r] == codes[ix];
                    d[(size_t)r] = (dq[(size_t)r] - x[ix]) - e[ix];
                    delta += (double)d[(size_t)r] * (2.0f * gm[ix] + d[(size_t)r] * hj);
                }
                if (same || !(delta < 0.0)) continue;
                const float* hr = h + (size_t)(j * K);
                for (int64_t r = 0; r < n; ++r) {
                    const size_t ix = (size_t)((n0 + r) * K + j);
                    codes[ix] = cs[(size_t)r];
                    const float dl = d[(size_t)r];
                    if (dl == 0.0f) continue;
                    float* gr = gm + (size_t)((n0 + r) * K);
                    for (int64_t t2 = 0; t2 < K; ++t2) gr[t2] += dl * hr[t2];
                    gr[j] += ad[(size_t)j] * dl;
                    e[ix] += dl;
                    w[ix] = dq[(size_t)r];
                }
                ++moved;
            }
            if (!moved) break;
        }
    }
}
}  /* namespace */

void gptq_rescale_rows(const float* x, float* w, Scales& scales, int64_t N, int64_t K,
                       const float* h) {
    if (!x || !w || !h || N <= 0 || K <= 0 || scales.nc != 1 || scales.br != 1) return;
    std::vector<float> hx((size_t)(N * K)), hw((size_t)(N * K));
    mat_hess(x, N, K, h, hx.data());
    mat_hess(w, N, K, h, hw.data());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t n = 0; n < N; ++n) {
        float* wr = w + (size_t)(n * K);
        const float* xr = hx.data() + (size_t)(n * K);
        const float* qr = hw.data() + (size_t)(n * K);
        double num = 0.0, den = 0.0;
        for (int64_t j = 0; j < K; ++j) {
            num += (double)wr[j] * xr[j];
            den += (double)wr[j] * qr[j];
        }
        if (!(num > 0.0) || !(den > 0.0)) continue;
        const float a = (float)(num / den);
        BlockScale& b = scales.b[(size_t)n];
        b.s *= a;
        b.stored *= a;
        b.inv = 1.0f / b.s;
        b.inv_d = 1.0 / (double)b.s;
        for (int64_t j = 0; j < K; ++j) wr[j] *= a;
    }
}

void gptq_refine(const Grid& g, const float* x, float* w, int32_t* codes, const Scales& scales,
                 int64_t N, int64_t K, const float* h, double damp, int passes) {
    if (!x || !w || !codes || !h || N <= 0 || K <= 0 || passes <= 0) return;
    /* Same damping as gptq_factor, so both minimise the same form: lambda is `damp` times the mean
     * diagonal, and an unexercised column takes the mean as its diagonal. `ad` is what the form
     * adds to H's diagonal, `dg` the resulting diagonal. */
    double tr = 0.0;
    for (int64_t i = 0; i < K; ++i) tr += h[(size_t)(i * K + i)];
    const double mean = tr / (double)K;
    const float lam = (float)(damp * mean);
    std::vector<float> ad((size_t)K), dg((size_t)K);
    for (int64_t i = 0; i < K; ++i) {
        const float d = h[(size_t)(i * K + i)];
        ad[(size_t)i] = (d > 0.0f ? 0.0f : (float)mean) + lam;
        dg[(size_t)i] = (d > 0.0f ? d : (float)mean) + lam;
    }
    /* G = E (H + diag(ad)) for all rows by the deterministic blocked product; each row then keeps
     * its G current by adding one row of it per code that moves. */
    std::vector<float> e((size_t)(N * K)), gm((size_t)(N * K));
    for (size_t i = 0; i < e.size(); ++i) e[i] = w[i] - x[i];
    mat_hess(e.data(), N, K, h, gm.data());
    if (g.tcq_L) {
        gptq_tcq_refine(g, x, w, codes, scales, N, K, h, ad.data(), dg.data(), e.data(), gm.data(),
                        passes);
        return;
    }
    /* A vector grid's pair products c_a c_b (a <= b) per codeword, for its descent. */
    std::vector<float> cpp;
    if (!g.vq.empty()) {
        const int64_t V = g.vdim;
        for (int64_t k = 0; k < g.ventries; ++k)
            for (int64_t a = 0; a < V; ++a)
                for (int64_t c = a; c < V; ++c)
                    cpp.push_back(g.vq[(size_t)(k * V + a)] * g.vq[(size_t)(k * V + c)]);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t n = 0; n < N; ++n) {
        float* er = e.data() + (size_t)(n * K);
        float* gr = gm.data() + (size_t)(n * K);
        const float* xr = x + (size_t)(n * K);
        float* wr = w + (size_t)(n * K);
        int32_t* cr = codes + (size_t)(n * K);
        for (int64_t j = 0; j < K; ++j) gr[j] += ad[(size_t)j] * er[j];
        const int64_t V = g.vq.empty() ? 0 : g.vdim;
        for (int pass = 0; pass < passes; ++pass) {
            int64_t moved = 0;
            /* A vector grid moves a whole codeword. For codeword k, decoded s c_k against the
             * current values w, the form changes by 2 s c_k.u + s^2 c_k^T A c_k + const, with
             * A = H_VV + diag(ad) and u = g_V - A w; the most negative change is taken when it
             * beats keeping the codeword. c_k^T A c_k is A's upper triangle against the codeword's
             * pair products, which are the same for every vector (cpp). */
            for (int64_t j0 = 0; V && j0 < K; j0 += V) {
                const BlockScale& b = scales.at(n, j0);
                float A[16][16], u[16], up[136];
                for (int64_t a = 0; a < V; ++a)
                    for (int64_t c = 0; c < V; ++c)
                        A[a][c] = h[(size_t)((j0 + a) * K + j0 + c)] + (a == c ? ad[(size_t)(j0 + a)] : 0.0f);
                for (int64_t a = 0; a < V; ++a) {
                    float aw = 0.0f;
                    for (int64_t c = 0; c < V; ++c) aw += A[a][c] * wr[j0 + c];
                    u[a] = gr[j0 + a] - aw;
                }
                int64_t t = 0;
                for (int64_t a = 0; a < V; ++a)
                    for (int64_t c = a; c < V; ++c, ++t) up[t] = a == c ? A[a][a] : 2.0f * A[a][c];
                const int64_t P = t;
                const float s1 = b.s, s2 = b.s * b.s;
                auto f = [&](int64_t k) {
                    const float* ck = g.vq.data() + k * V;
                    const float* pk = cpp.data() + k * P;
                    float lin = 0.0f, quad = 0.0f;
                    for (int64_t a = 0; a < V; ++a) lin += ck[a] * u[a];
                    for (int64_t q = 0; q < P; ++q) quad += pk[q] * up[q];
                    return 2.0f * s1 * lin + s2 * quad;
                };
                const float keep = f(cr[j0]);
                int best = -1;
                float bestv = keep;
                for (int64_t k = 0; k < g.ventries; ++k) {
                    if (k == cr[j0]) continue;
                    const float v = f(k);
                    if (v < bestv) { bestv = v; best = (int)k; }
                }
                if (best < 0) continue;
                for (int64_t a = 0; a < V; ++a) {
                    const float qn = g.vq[(size_t)(best * V + a)] * b.s;
                    const float dl = qn - wr[j0 + a];
                    const float* hr = h + (size_t)((j0 + a) * K);
                    for (int64_t t2 = 0; t2 < K; ++t2) gr[t2] += dl * hr[t2];
                    gr[j0 + a] += ad[(size_t)(j0 + a)] * dl;
                    er[j0 + a] += dl;
                    wr[j0 + a] = qn;
                    cr[j0 + a] = best;
                }
                ++moved;
            }
            for (int64_t j = 0; !V && j < K; ++j) {
                const float hj = dg[(size_t)j];
                /* The row's gradient at j less column j's own term, over the diagonal. */
                const float t = xr[j] - (gr[j] - er[j] * hj) / hj;
                const BlockScale& b = scales.at(n, j);
                const int c = grid_code(g, b, t);
                if (c == cr[j]) continue;
                const float dq = grid_decode(g, b, c);
                const float d = (dq - xr[j]) - er[j];
                /* Only a strict improvement moves a code: the form changes by d (2 g_j + d hj),
                 * g_j the current gradient; a tie keeps GPTQ's choice. */
                if (!(d * (2.0f * gr[j] + d * hj) < 0.0f)) continue;
                const float* hr = h + (size_t)(j * K);
                for (int64_t t2 = 0; t2 < K; ++t2) gr[t2] += d * hr[t2];
                gr[j] += ad[(size_t)j] * d;
                er[j] += d;
                wr[j] = dq;
                cr[j] = c;
                ++moved;
            }
            if (!moved) break;
        }
    }
}

}  /* namespace lq */
