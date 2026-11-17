/* Manifold-constrained hyper-connections. The residual mixing matrix is indexed
 * [source, destination]; transposing it leaves every shape valid and changes the model.
 * The first Sinkhorn pass is softmax + eps, then column normalization. Later passes
 * add eps to both denominators. These are separate operations from Qwen's hc_read/write. */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>
#include <cmath>

using namespace ref;

namespace {
bool matrix(const RadTensor& t, int64_t rows, int64_t cols) {
    return t.data && t.rank == 2 && t.shape[0] == rows && t.shape[1] == cols;
}
}

namespace {
int64_t part_stride(int64_t hc) { return (hc * (hc + 2) + 1 + 3) / 4 * 4; }

bool pre_params(const RadArgs* a, int64_t* n, int64_t* hc, int64_t* iters, float* eps,
                float* heps) {
    *n = p_int(a, "n", 0);
    *hc = p_int(a, "hc", 0);
    *iters = p_int(a, "iterations", 0);
    *eps = p_f32(a, "eps", 0);
    *heps = p_f32(a, "hc_eps", 0);
    return *n > 0 && *hc >= 2 && *hc <= 8 && *iters >= 1 && *iters <= 256 && std::isfinite(*eps) &&
           std::isfinite(*heps) && *eps > 0 && *heps >= 0;
}

/* The outputs' shapes, shared by mhc_pre and mhc_pre_finish. */
bool pre_outputs(const RadTensor& h, const RadTensor& scale, const RadTensor& base,
                 const RadTensor& x, const RadTensor& post, const RadTensor& comb, int64_t rows,
                 int64_t n, int64_t hc) {
    const int64_t mix = hc * (hc + 2);
    return matrix(h, rows, hc * n) && scale.rank == 1 && scale.dtype == RAD_F32 &&
           numel(&scale) == 3 && base.rank == 1 && base.dtype == RAD_F32 && numel(&base) == mix &&
           matrix(x, rows, n) && matrix(post, rows, hc) && matrix(comb, rows, hc * hc) &&
           post.dtype == RAD_F32 && comb.dtype == RAD_F32;
}

/* Row r from its RMS-scaled coefficients: gates, Sinkhorn mixing and the collapse of h. */
void finish_row(const float* coeff, const RadTensor& h, const RadTensor& scale,
                const RadTensor& base, const RadTensor& x, const RadTensor& post,
                const RadTensor& comb, int64_t r, int64_t n, int64_t hc, int64_t iters,
                float heps) {
    float pre[8], c[64];
    const int64_t hb = rowoff(&h, r, hc * n), hs = laststride(&h);
    for (int64_t j = 0; j < hc; ++j) {
        pre[j] =
            1.0f / (1.0f +
                    std::exp(-(coeff[j] * ldt(&scale, 0) + ldt(&base, j * laststride(&base))))) +
            heps;
        const float z = coeff[hc + j] * ldt(&scale, laststride(&scale)) +
                        ldt(&base, (hc + j) * laststride(&base));
        stt(&post, rowoff(&post, r, hc) + j * laststride(&post), 2.0f / (1.0f + std::exp(-z)));
    }
    for (int64_t j = 0; j < hc; ++j) {
        float mx = -INFINITY;
        for (int64_t k = 0; k < hc; ++k) {
            const int64_t q = 2 * hc + j * hc + k;
            c[j * hc + k] =
                coeff[q] * ldt(&scale, 2 * laststride(&scale)) + ldt(&base, q * laststride(&base));
            mx = std::max(mx, c[j * hc + k]);
        }
        float sum = 0;
        for (int64_t k = 0; k < hc; ++k) {
            c[j * hc + k] = std::exp(c[j * hc + k] - mx);
            sum += c[j * hc + k];
        }
        for (int64_t k = 0; k < hc; ++k) c[j * hc + k] = c[j * hc + k] / sum + heps;
    }
    for (int64_t it = 0; it < iters; ++it) {
        if (it)
            for (int64_t j = 0; j < hc; ++j) {
                float sum = 0;
                for (int64_t k = 0; k < hc; ++k) sum += c[j * hc + k];
                for (int64_t k = 0; k < hc; ++k) c[j * hc + k] /= sum + heps;
            }
        for (int64_t k = 0; k < hc; ++k) {
            float sum = 0;
            for (int64_t j = 0; j < hc; ++j) sum += c[j * hc + k];
            for (int64_t j = 0; j < hc; ++j) c[j * hc + k] /= sum + heps;
        }
    }
    const int64_t cb = rowoff(&comb, r, hc * hc), cs = laststride(&comb);
    for (int64_t j = 0; j < hc * hc; ++j) stt(&comb, cb + j * cs, c[j]);
    const int64_t xb = rowoff(&x, r, n), xs = laststride(&x);
    for (int64_t i = 0; i < n; ++i) {
        float v = 0;
        for (int64_t j = 0; j < hc; ++j) v += pre[j] * ldt(&h, hb + (j * n + i) * hs);
        stt(&x, xb + i * xs, v);
    }
}
}

int ref_mhc_pre(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
    int64_t n, hc, iters;
    float eps, heps;
    if (!pre_params(a, &n, &hc, &iters, &eps, &heps)) return RAD_E_SHAPE;
    const auto &h = a->t[0], &fn = a->t[1], &scale = a->t[2], &base = a->t[3];
    const auto &x = a->t[4], &post = a->t[5], &comb = a->t[6];
    const int64_t width = hc * n, mix = hc * (hc + 2);
    if (h.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0];
    if (rows < 1 || !matrix(fn, mix, width) || fn.dtype != RAD_F32 ||
        !pre_outputs(h, scale, base, x, post, comb, rows, n, hc))
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        float coeff[80];
        const int64_t hb = rowoff(&h, r, width), hs = laststride(&h);
        float sq = 0;
        for (int64_t i = 0; i < width; ++i) {
            const float v = ldt(&h, hb + i * hs);
            sq += v * v;
        }
        const float inv = 1.0f / std::sqrt(sq / (float)width + eps);
        for (int64_t j = 0; j < mix; ++j) {
            float dot = 0;
            const int64_t fb = rowoff(&fn, j, width), fs = laststride(&fn);
            for (int64_t i = 0; i < width; ++i) dot += ldt(&h, hb + i * hs) * ldt(&fn, fb + i * fs);
            coeff[j] = dot * inv;
        }
        finish_row(coeff, h, scale, base, x, post, comb, r, n, hc, iters, heps);
    }
    return RAD_OK;
}

int ref_mhc_pre_partial(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0), chunk = p_int(a, "chunk", 0);
    const int64_t first = p_int(a, "first", -1), count = p_int(a, "chunks", 0);
    if (n <= 0 || hc < 2 || hc > 8 || chunk <= 0 || (hc * n) % chunk || first < 0 || count < 1 ||
        first + count > hc * n / chunk)
        return RAD_E_SHAPE;
    const auto &h = a->t[0], &fn = a->t[1], &part = a->t[2];
    const int64_t width = hc * n, mix = hc * (hc + 2), S = part_stride(hc);
    if (h.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0];
    if (rows < 1 || !matrix(h, rows, width) || !matrix(fn, mix, width) || fn.dtype != RAD_F32 ||
        !matrix(part, rows, count * S) || part.dtype != RAD_F32)
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t hb = rowoff(&h, r, width), hs = laststride(&h);
        const int64_t pb = rowoff(&part, r, count * S), ps = laststride(&part);
        for (int64_t c = 0; c < count; ++c) {
            const int64_t lo = (first + c) * chunk;
            for (int64_t j = 0; j < S; ++j) {
                float sum = 0;
                if (j < mix) {
                    const int64_t fb = rowoff(&fn, j, width), fs = laststride(&fn);
                    for (int64_t i = lo; i < lo + chunk; ++i)
                        sum += ldt(&h, hb + i * hs) * ldt(&fn, fb + i * fs);
                } else if (j == mix) {
                    for (int64_t i = lo; i < lo + chunk; ++i) {
                        const float v = ldt(&h, hb + i * hs);
                        sum += v * v;
                    }
                }
                stt(&part, pb + (c * S + j) * ps, sum);
            }
        }
    }
    return RAD_OK;
}

int ref_mhc_pre_finish(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
    int64_t n, hc, iters;
    float eps, heps;
    if (!pre_params(a, &n, &hc, &iters, &eps, &heps)) return RAD_E_SHAPE;
    const int64_t chunk = p_int(a, "chunk", 0);
    if (chunk <= 0 || (hc * n) % chunk) return RAD_E_SHAPE;
    const auto &part = a->t[0], &h = a->t[1], &scale = a->t[2], &base = a->t[3];
    const auto &x = a->t[4], &post = a->t[5], &comb = a->t[6];
    const int64_t width = hc * n, mix = hc * (hc + 2), S = part_stride(hc), chunks = width / chunk;
    if (h.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0];
    if (rows < 1 || !matrix(part, rows, chunks * S) || part.dtype != RAD_F32 ||
        !pre_outputs(h, scale, base, x, post, comb, rows, n, hc))
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        float coeff[80];
        const int64_t pb = rowoff(&part, r, chunks * S), ps = laststride(&part);
        float sq = 0;
        for (int64_t c = 0; c < chunks; ++c) sq += ldt(&part, pb + (c * S + mix) * ps);
        const float inv = 1.0f / std::sqrt(sq / (float)width + eps);
        for (int64_t j = 0; j < mix; ++j) {
            float dot = 0;
            for (int64_t c = 0; c < chunks; ++c) dot += ldt(&part, pb + (c * S + j) * ps);
            coeff[j] = dot * inv;
        }
        finish_row(coeff, h, scale, base, x, post, comb, r, n, hc, iters, heps);
    }
    return RAD_OK;
}

int ref_mhc_post(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    if (n <= 0 || hc < 2 || hc > 8) return RAD_E_SHAPE;
    const auto &x = a->t[0], &h = a->t[1], &post = a->t[2], &comb = a->t[3], &out = a->t[4];
    if (x.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = x.shape[0];
    if (rows < 1 || !matrix(x, rows, n) || !matrix(h, rows, hc * n) || !matrix(post, rows, hc) ||
        !matrix(comb, rows, hc * hc) || !matrix(out, rows, hc * n) || out.data == h.data ||
        post.dtype != RAD_F32 || comb.dtype != RAD_F32)
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t hb = rowoff(&h, r, hc * n), cb = rowoff(&comb, r, hc * hc);
        const int64_t pb = rowoff(&post, r, hc), xb = rowoff(&x, r, n);
        const int64_t ob = rowoff(&out, r, hc * n);
        for (int64_t k = 0; k < hc; ++k)
            for (int64_t i = 0; i < n; ++i) {
                float v = 0;
                for (int64_t j = 0; j < hc; ++j)
                    v += ldt(&comb, cb + (j * hc + k) * laststride(&comb)) *
                         ldt(&h, hb + (j * n + i) * laststride(&h));
                v += ldt(&post, pb + k * laststride(&post)) * ldt(&x, xb + i * laststride(&x));
                stt(&out, ob + (k * n + i) * laststride(&out), v);
            }
    }
    return RAD_OK;
}

int ref_mhc_contract(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    if (n <= 0 || hc < 2 || hc > 8) return RAD_E_SHAPE;
    const auto &h = a->t[0], &x = a->t[1];
    if (h.rank != 2 || h.shape[0] < 1 || !matrix(h, h.shape[0], hc * n) ||
        !matrix(x, h.shape[0], n))
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < h.shape[0]; ++r)
        for (int64_t i = 0; i < n; ++i) {
            float v = 0;
            for (int64_t j = 0; j < hc; ++j)
                v += ldt(&h, rowoff(&h, r, hc * n) + (j * n + i) * laststride(&h));
            stt(&x, rowoff(&x, r, n) + i * laststride(&x), v / (float)hc);
        }
    return RAD_OK;
}
