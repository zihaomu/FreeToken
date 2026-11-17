/* Host mHC and bounded KDA. Widen each activation row once, then reduce the
 * projections and state products with this ISA level's vector operations. */
#include "avx_vec.h"
#include "avx_common.h"
#include <algorithm>

using namespace avx;

namespace {
bool matrix(const RadTensor& t, int64_t r, int64_t c) {
    return t.data && t.rank == 2 && t.shape[0] == r && t.shape[1] == c;
}
float dot(const float* x, const float* y, int64_t n) {
    vf acc = vf_zero();
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) acc = vf_fma(vf_loadu(x + i), vf_loadu(y + i), acc);
    float s = vf_hsum(acc);
    for (; i < n; ++i) s += x[i] * y[i];
    return s;
}
void sinkhorn(float* c, int h, int repeats, float eps) {
    for (int r = 0; r < h; ++r) {
        float max = -INFINITY;
        for (int k = 0; k < h; ++k) max = std::max(max, c[r * h + k]);
        float total = 0;
        for (int k = 0; k < h; ++k) {
            c[r * h + k] = std::exp(c[r * h + k] - max);
            total += c[r * h + k];
        }
        for (int k = 0; k < h; ++k) c[r * h + k] = c[r * h + k] / total + eps;
    }
    for (int it = 0; it < repeats; ++it) {
        if (it)
            for (int r = 0; r < h; ++r) {
                float total = 0;
                for (int k = 0; k < h; ++k) total += c[r * h + k];
                for (int k = 0; k < h; ++k) c[r * h + k] /= total + eps;
            }
        for (int k = 0; k < h; ++k) {
            float total = 0;
            for (int r = 0; r < h; ++r) total += c[r * h + k];
            for (int r = 0; r < h; ++r) c[r * h + k] /= total + eps;
        }
    }
}
bool headed(const RadTensor& t, int64_t tokens, int64_t heads, int64_t dim) {
    return t.data && t.shape[0] == tokens &&
           ((t.rank == 3 && t.shape[1] == heads && t.shape[2] == dim) ||
            (t.rank == 2 && t.shape[1] == heads * dim));
}
int64_t at(const RadTensor& t, int64_t token, int64_t h, int64_t d, int64_t width) {
    return token * t.stride[0] + h * (t.rank == 3 ? t.stride[1] : width * t.stride[1]) +
           d * laststride(&t);
}
}

AVX_KERNEL(gemm_nt_f32) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const auto &x = a->t[0], &w = a->t[1], &y = a->t[2];
    const int64_t n = p_int(a, "N", 0), k = p_int(a, "K", 0);
    if (n < 1 || k < 1 || x.rank != 2 || w.rank != 2 || y.rank != 2 || x.shape[0] < 1 ||
        x.shape[1] != k || w.shape[0] != n || w.shape[1] != k || y.shape[0] != x.shape[0] ||
        y.shape[1] != n)
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || w.dtype != RAD_F32 || y.dtype != RAD_F32) return RAD_E_DTYPE;
    if (!x.data || !w.data || !y.data) return RAD_E_INVAL;
    AVX_PARALLEL_FOR_IF(x.shape[0] * n * k >= AVX_PAR_MIN)
    for (int64_t r = 0; r < x.shape[0]; ++r) {
        float* query = scratch_f32((size_t)k);
        for (int64_t j = 0; j < k; ++j) query[j] = ldt(&x, r * x.stride[0] + j * x.stride[1]);
        for (int64_t col = 0; col < n; ++col) {
            float sum = 0;
            if (w.stride[1] == 1)
                sum = dot(query, (const float*)w.data + col * w.stride[0], k);
            else
                for (int64_t j = 0; j < k; ++j)
                    sum += query[j] * ldt(&w, col * w.stride[0] + j * w.stride[1]);
            stt(&y, r * y.stride[0] + col * y.stride[1], sum);
        }
    }
    return RAD_OK;
}

AVX_KERNEL(router_sigmoid_topk) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const auto &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const int64_t ne = p_int(a, "n_expert", 0), k = p_int(a, "top_k", 0);
    const float factor = p_f32(a, "route_scale", 0);
    if (ne < 1 || ne > 512 || k < 1 || k > ne || !std::isfinite(factor) || factor <= 0 ||
        lg.rank != 2 || lg.shape[0] < 1 || lg.shape[1] != ne || bias.rank != 1 ||
        bias.shape[0] != ne || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 ||
        !matrix(ids, lg.shape[0], k) || !matrix(w, lg.shape[0], k) || !lg.data || !bias.data)
        return RAD_E_SHAPE;
    for (int64_t r = 0; r < lg.shape[0]; ++r) {
        float prob[512], ranked[512], chosen[512];
        int index[512];
        for (int64_t e = 0; e < ne; ++e) {
            prob[e] = 1.0f / (1.0f + std::exp(-ldt(&lg, rowoff(&lg, r, ne) + e * laststride(&lg))));
            ranked[e] = prob[e] + ldt(&bias, e * laststride(&bias));
        }
        float total = 0;
        for (int64_t j = 0; j < k; ++j) {
            int winner = -1;
            for (int64_t e = 0; e < ne; ++e)
                if (ranked[e] > -INFINITY && (winner < 0 || ranked[e] > ranked[winner]))
                    winner = (int)e;
            if (winner < 0) return RAD_E_SHAPE;
            index[j] = winner;
            chosen[j] = prob[winner];
            total += chosen[j];
            ranked[winner] = -INFINITY;
        }
        const float mult = p_int(a, "norm", 0) ? factor / (total + 1e-20f) : factor;
        for (int64_t j = 0; j < k; ++j) {
            st_int_dt(ids.dtype, ids.data, rowoff(&ids, r, k) + j * laststride(&ids), index[j]);
            stt(&w, rowoff(&w, r, k) + j * laststride(&w), chosen[j] * mult);
        }
    }
    return RAD_OK;
}

AVX_KERNEL(swiglu_clamped) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const auto &gu = a->t[0], &y = a->t[1];
    const int64_t n = p_int(a, "n", 0);
    const float limit = p_f32(a, "limit", 0);
    if (n < 1 || !std::isfinite(limit) || limit <= 0 || gu.rank != 2 || y.rank != 2 || !gu.data ||
        !y.data || y.shape[0] < 1 || gu.shape[0] != y.shape[0] || gu.shape[1] != 2 * n ||
        y.shape[1] != n)
        return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(y.shape[0] * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < y.shape[0]; ++r) {
        for (int64_t i = 0; i < n; ++i) {
            const int64_t b = rowoff(&gu, r, 2 * n), s = laststride(&gu);
            const float g = std::min(ldt(&gu, b + i * s), limit);
            const float u = std::clamp(ldt(&gu, b + (n + i) * s), -limit, limit);
            stt(&y, rowoff(&y, r, n) + i * laststride(&y), (g / (1.0f + std::exp(-g))) * u);
        }
    }
    return RAD_OK;
}

AVX_KERNEL(swiglu_step) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const auto &gu = a->t[0], &y = a->t[1];
    const int64_t n = p_int(a, "n", 0);
    const float limit = p_f32(a, "limit", 0);
    if (n < 1 || !std::isfinite(limit) || limit <= 0 || gu.rank != 2 || y.rank != 2 || !gu.data ||
        !y.data || y.shape[0] < 1 || gu.shape[0] != y.shape[0] || gu.shape[1] != 2 * n ||
        y.shape[1] != n)
        return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(y.shape[0] * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < y.shape[0]; ++r) {
        for (int64_t i = 0; i < n; ++i) {
            const int64_t b = rowoff(&gu, r, 2 * n), s = laststride(&gu);
            const float x = ldt(&gu, b + i * s);
            const float g = std::min(x / (1.0f + std::exp(-x)), limit);
            const float u = std::clamp(ldt(&gu, b + (n + i) * s), -limit, limit);
            stt(&y, rowoff(&y, r, n) + i * laststride(&y), g * u);
        }
    }
    return RAD_OK;
}

AVX_KERNEL(router_sigmoid_group_topk) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const auto &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const int64_t ne = p_int(a, "n_expert", 0), k = p_int(a, "top_k", 0),
                  ng = p_int(a, "n_group", 0), kg = p_int(a, "topk_group", 0);
    const float factor = p_f32(a, "route_scale", 0);
    if (ne < 1 || ne > 512 || k < 1 || ng < 1 || ne % ng || ne / ng < 2 || kg < 1 || kg > ng ||
        k > kg * (ne / ng) || !std::isfinite(factor) || factor <= 0 || lg.rank != 2 ||
        lg.shape[0] < 1 || lg.shape[1] != ne || bias.rank != 1 || bias.shape[0] != ne ||
        bias.dtype != RAD_F32 || ids.dtype != RAD_I32 || !matrix(ids, lg.shape[0], k) ||
        !matrix(w, lg.shape[0], k) || !lg.data || !bias.data)
        return RAD_E_SHAPE;
    const int64_t gs = ne / ng;
    for (int64_t r = 0; r < lg.shape[0]; ++r) {
        float prob[512], ranked[512], chosen[512], group[512];
        int index[512];
        bool kept[512] = {};
        for (int64_t e = 0; e < ne; ++e) {
            prob[e] = 1.0f / (1.0f + std::exp(-ldt(&lg, rowoff(&lg, r, ne) + e * laststride(&lg))));
            ranked[e] = prob[e] + ldt(&bias, e * laststride(&bias));
        }
        for (int64_t gi = 0; gi < ng; ++gi) {
            float t0 = -INFINITY, t1 = -INFINITY;
            for (int64_t e = gi * gs; e < (gi + 1) * gs; ++e) {
                if (ranked[e] > t0) { t1 = t0; t0 = ranked[e]; }
                else if (ranked[e] > t1) t1 = ranked[e];
            }
            group[gi] = t0 + t1;
        }
        for (int64_t j = 0; j < kg; ++j) {
            int winner = -1;
            for (int64_t gi = 0; gi < ng; ++gi)
                if (group[gi] > -INFINITY && (winner < 0 || group[gi] > group[winner]))
                    winner = (int)gi;
            if (winner < 0) return RAD_E_SHAPE;
            for (int64_t e = winner * gs; e < (winner + 1) * gs; ++e) kept[e] = true;
            group[winner] = -INFINITY;
        }
        float total = 0;
        for (int64_t j = 0; j < k; ++j) {
            int winner = -1;
            for (int64_t e = 0; e < ne; ++e)
                if (kept[e] && ranked[e] > -INFINITY && (winner < 0 || ranked[e] > ranked[winner]))
                    winner = (int)e;
            if (winner < 0) return RAD_E_SHAPE;
            index[j] = winner;
            chosen[j] = prob[winner];
            total += chosen[j];
            ranked[winner] = -INFINITY;
        }
        const float mult = p_int(a, "norm", 0) ? factor / (total + 1e-20f) : factor;
        for (int64_t j = 0; j < k; ++j) {
            st_int_dt(ids.dtype, ids.data, rowoff(&ids, r, k) + j * laststride(&ids), index[j]);
            stt(&w, rowoff(&w, r, k) + j * laststride(&w), chosen[j] * mult);
        }
    }
    return RAD_OK;
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
    return *n >= 1 && *hc >= 2 && *hc <= 8 && *iters >= 1 && *iters <= 256 && std::isfinite(*eps) &&
           *eps > 0 && std::isfinite(*heps) && *heps >= 0;
}
bool pre_outputs(const RadTensor& h, const RadTensor& scale, const RadTensor& base,
                 const RadTensor& x, const RadTensor& post, const RadTensor& comb, int64_t rows,
                 int64_t n, int64_t hc) {
    const int64_t mix = hc * (hc + 2);
    return matrix(h, rows, hc * n) && scale.rank == 1 && scale.dtype == RAD_F32 &&
           numel(&scale) == 3 && base.rank == 1 && base.dtype == RAD_F32 && numel(&base) == mix &&
           matrix(x, rows, n) && matrix(post, rows, hc) && matrix(comb, rows, hc * hc) &&
           post.dtype == RAD_F32 && comb.dtype == RAD_F32;
}
/* Row r from its RMS-scaled coefficients and the widened residual row hv. */
void finish_row(const float* coeff, const float* hv, const RadTensor& scale, const RadTensor& base,
                const RadTensor& x, const RadTensor& post, const RadTensor& comb, int64_t r,
                int64_t n, int64_t hc, int64_t iters, float heps) {
    float pre[8], c[64];
    for (int64_t j = 0; j < hc; ++j) {
        pre[j] =
            1.0f / (1.0f +
                    std::exp(-(coeff[j] * ldt(&scale, 0) + ldt(&base, j * laststride(&base))))) +
            heps;
        const float z = coeff[hc + j] * ldt(&scale, laststride(&scale)) +
                        ldt(&base, (hc + j) * laststride(&base));
        stt(&post, rowoff(&post, r, hc) + j * laststride(&post), 2.0f / (1.0f + std::exp(-z)));
    }
    for (int64_t j = 0; j < hc * hc; ++j)
        c[j] = coeff[2 * hc + j] * ldt(&scale, 2 * laststride(&scale)) +
               ldt(&base, (2 * hc + j) * laststride(&base));
    sinkhorn(c, (int)hc, (int)iters, heps);
    for (int64_t j = 0; j < hc * hc; ++j)
        stt(&comb, rowoff(&comb, r, hc * hc) + j * laststride(&comb), c[j]);
    for (int64_t i = 0; i < n; ++i) {
        float v = 0;
        for (int64_t j = 0; j < hc; ++j) v += pre[j] * hv[j * n + i];
        stt(&x, rowoff(&x, r, n) + i * laststride(&x), v);
    }
}
/* fn row j over columns [lo, lo+len), contiguous or through the scratch fw. */
float fn_dot(const float* hv, const RadTensor& fn, int64_t j, int64_t width, int64_t lo,
             int64_t len, float* fw) {
    const int64_t fb = rowoff(&fn, j, width);
    if (laststride(&fn) == 1) return dot(hv + lo, (const float*)fn.data + fb + lo, len);
    for (int64_t i = 0; i < len; ++i) fw[i] = ldt(&fn, fb + (lo + i) * laststride(&fn));
    return dot(hv + lo, fw, len);
}
void widen(const RadTensor& h, int64_t r, int64_t width, float* hv) {
    for (int64_t i = 0; i < width; ++i) hv[i] = ldt(&h, rowoff(&h, r, width) + i * laststride(&h));
}
}

AVX_KERNEL(mhc_pre) {
    if (!rad_args_have(a, 7)) return RAD_E_INVAL;
    int64_t n, hc, iters;
    float eps, heps;
    if (!pre_params(a, &n, &hc, &iters, &eps, &heps)) return RAD_E_SHAPE;
    const auto &h = a->t[0], &fn = a->t[1], &scale = a->t[2], &base = a->t[3];
    const auto &x = a->t[4], &post = a->t[5], &comb = a->t[6];
    if (h.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0], width = hc * n, mix = hc * (hc + 2);
    if (rows < 1 || !matrix(fn, mix, width) || fn.dtype != RAD_F32 ||
        !pre_outputs(h, scale, base, x, post, comb, rows, n, hc))
        return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(rows * width * mix >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* scratch = scratch_f32((size_t)(2 * width));
        float* hv = scratch;
        float* fw = hv + width;
        widen(h, r, width, hv);
        const float norm = 1.0f / std::sqrt(dot(hv, hv, width) / (float)width + eps);
        float coeff[80];
        for (int64_t j = 0; j < mix; ++j) coeff[j] = fn_dot(hv, fn, j, width, 0, width, fw) * norm;
        finish_row(coeff, hv, scale, base, x, post, comb, r, n, hc, iters, heps);
    }
    return RAD_OK;
}

AVX_KERNEL(mhc_pre_partial) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0), chunk = p_int(a, "chunk", 0);
    const int64_t first = p_int(a, "first", -1), count = p_int(a, "chunks", 0);
    if (n < 1 || hc < 2 || hc > 8 || chunk < 1 || (hc * n) % chunk || first < 0 || count < 1 ||
        first + count > hc * n / chunk)
        return RAD_E_SHAPE;
    const auto &h = a->t[0], &fn = a->t[1], &part = a->t[2];
    if (h.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0], width = hc * n, mix = hc * (hc + 2), S = part_stride(hc);
    if (rows < 1 || !matrix(h, rows, width) || !matrix(fn, mix, width) || fn.dtype != RAD_F32 ||
        !matrix(part, rows, count * S) || part.dtype != RAD_F32)
        return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(rows * count * chunk * mix >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* scratch = scratch_f32((size_t)(width + chunk));
        float* hv = scratch;
        float* fw = hv + width;
        widen(h, r, width, hv);
        const int64_t pb = rowoff(&part, r, count * S), ps = laststride(&part);
        for (int64_t c = 0; c < count; ++c) {
            const int64_t lo = (first + c) * chunk;
            for (int64_t j = 0; j < S; ++j) {
                const float v = j < mix    ? fn_dot(hv, fn, j, width, lo, chunk, fw)
                                : j == mix ? dot(hv + lo, hv + lo, chunk)
                                           : 0.0f;
                stt(&part, pb + (c * S + j) * ps, v);
            }
        }
    }
    return RAD_OK;
}

AVX_KERNEL(mhc_pre_finish) {
    if (!rad_args_have(a, 7)) return RAD_E_INVAL;
    int64_t n, hc, iters;
    float eps, heps;
    if (!pre_params(a, &n, &hc, &iters, &eps, &heps)) return RAD_E_SHAPE;
    const int64_t chunk = p_int(a, "chunk", 0);
    if (chunk < 1 || (hc * n) % chunk) return RAD_E_SHAPE;
    const auto &part = a->t[0], &h = a->t[1], &scale = a->t[2], &base = a->t[3];
    const auto &x = a->t[4], &post = a->t[5], &comb = a->t[6];
    if (h.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0], width = hc * n, mix = hc * (hc + 2), S = part_stride(hc),
                  chunks = width / chunk;
    if (rows < 1 || !matrix(part, rows, chunks * S) || part.dtype != RAD_F32 ||
        !pre_outputs(h, scale, base, x, post, comb, rows, n, hc))
        return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(rows * width >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* hv = scratch_f32((size_t)width);
        widen(h, r, width, hv);
        const int64_t pb = rowoff(&part, r, chunks * S), ps = laststride(&part);
        float sq = 0;
        for (int64_t c = 0; c < chunks; ++c) sq += ldt(&part, pb + (c * S + mix) * ps);
        const float norm = 1.0f / std::sqrt(sq / (float)width + eps);
        float coeff[80];
        for (int64_t j = 0; j < mix; ++j) {
            float d = 0;
            for (int64_t c = 0; c < chunks; ++c) d += ldt(&part, pb + (c * S + j) * ps);
            coeff[j] = d * norm;
        }
        finish_row(coeff, hv, scale, base, x, post, comb, r, n, hc, iters, heps);
    }
    return RAD_OK;
}

AVX_KERNEL(mhc_post) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    if (n < 1 || hc < 2 || hc > 8) return RAD_E_SHAPE;
    const auto &x = a->t[0], &h = a->t[1], &p = a->t[2], &c = a->t[3], &o = a->t[4];
    if (x.rank != 2) return RAD_E_SHAPE;
    const int64_t rows = x.shape[0];
    if (rows < 1 || !matrix(x, rows, n) || !matrix(h, rows, hc * n) || !matrix(p, rows, hc) ||
        !matrix(c, rows, hc * hc) || !matrix(o, rows, hc * n) || o.data == h.data ||
        p.dtype != RAD_F32 || c.dtype != RAD_F32)
        return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(rows * n * hc >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t dst = 0; dst < hc; ++dst)
            for (int64_t i = 0; i < n; ++i) {
                float v = 0;
                for (int64_t src = 0; src < hc; ++src)
                    v += ldt(&c, rowoff(&c, r, hc * hc) + (src * hc + dst) * laststride(&c)) *
                         ldt(&h, rowoff(&h, r, hc * n) + (src * n + i) * laststride(&h));
                v += ldt(&p, rowoff(&p, r, hc) + dst * laststride(&p)) *
                     ldt(&x, rowoff(&x, r, n) + i * laststride(&x));
                stt(&o, rowoff(&o, r, hc * n) + (dst * n + i) * laststride(&o), v);
            }
    }
    return RAD_OK;
}

AVX_KERNEL(mhc_contract) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), hc = p_int(a, "hc", 0);
    if (n < 1 || hc < 2 || hc > 8) return RAD_E_SHAPE;
    const auto &h = a->t[0], &x = a->t[1];
    if (h.rank != 2 || h.shape[0] < 1 || !matrix(h, h.shape[0], hc * n) ||
        !matrix(x, h.shape[0], n))
        return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(h.shape[0] * n * hc >= AVX_PAR_MIN)
    for (int64_t r = 0; r < h.shape[0]; ++r)
        for (int64_t i = 0; i < n; ++i) {
            float v = 0;
            for (int64_t j = 0; j < hc; ++j)
                v += ldt(&h, rowoff(&h, r, hc * n) + (j * n + i) * laststride(&h));
            stt(&x, rowoff(&x, r, n) + i * laststride(&x), v / (float)hc);
        }
    return RAD_OK;
}

AVX_KERNEL(kda_update) {
    if (!a || a->n_t < 13) return RAD_E_INVAL;
    for (int i = 0; i < 10; ++i)
        if (!rad_arg_in(a, i)) return RAD_E_INVAL;
    if (!rad_arg_in(a, 12)) return RAD_E_INVAL;
    const int64_t H = p_int(a, "heads", 0), K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const float lower = p_f32(a, "lower_bound", 0), eps = p_f32(a, "l2_eps", 1e-6f),
                scale = p_f32(a, "scale", 1);
    const char* mode = p_str(a, "mode", "");
    const bool verify = !std::strcmp(mode, "verify");
    const char* form = p_str(a, "state_form", "per_candidate");
    const bool anchor = !std::strcmp(form, "anchor");
    if (!anchor && std::strcmp(form, "per_candidate")) return RAD_E_INVAL;
    if (anchor && (K != 128 || V != 128)) return RAD_E_UNSUPPORTED;
    if (!verify && std::strcmp(mode, "prefill")) return RAD_E_INVAL;
    if (H < 1 || K < 1 || K > 128 || V < 1 || V > 128 || !std::isfinite(lower) || lower >= 0 ||
        !std::isfinite(eps) || eps <= 0 || !std::isfinite(scale))
        return RAD_E_SHAPE;
    const auto &q = a->t[0], &k = a->t[1], &v = a->t[2], &g = a->t[3], &b = a->t[4];
    const auto &al = a->t[5], &bias = a->t[6], &s = a->t[7], &idx = a->t[8], &cu = a->t[9],
               &out = a->t[12];
    const auto* accepted = rad_arg_in(a, 10);
    const auto* init = rad_arg_in(a, 11);
    if (q.rank < 2 || cu.rank != 1 || cu.dtype != RAD_I32 || cu.shape[0] < 2) return RAD_E_SHAPE;
    const int64_t T = q.shape[0], N = cu.shape[0] - 1, slots = s.shape[0];
    if (T < 1) return RAD_E_SHAPE;
    if (!headed(q, T, H, K) || !headed(k, T, H, K) || !headed(v, T, H, V) || !headed(g, T, H, K) ||
        !headed(b, T, H, 1) || !headed(out, T, H, V) || numel(&al) != H || numel(&bias) != H * K ||
        s.rank != 4 || s.dtype != RAD_F32 || s.shape[1] != H || s.shape[2] != V ||
        s.shape[3] != K || idx.rank != 2 || idx.dtype != RAD_I32 || idx.shape[0] < N ||
        idx.shape[1] < 1 || (accepted && (accepted->dtype != RAD_I32 || numel(accepted) < N)) ||
        (init && (init->dtype != RAD_I32 || numel(init) < N)))
        return RAD_E_SHAPE;
    const int64_t width = idx.shape[1];
    for (int64_t seq = 0; seq < N; ++seq) {
        const int64_t begin = ldi(&cu, seq * cu.stride[0]);
        const int64_t end = ldi(&cu, (seq + 1) * cu.stride[0]);
        const int64_t na = accepted ? ldi(accepted, offlin(accepted, seq)) : 1;
        if (begin < 0 || end < begin || end > T || na < 1 || na > (anchor ? 32 : width) ||
            (verify && end - begin > (anchor ? 32 : width)) || (anchor && width < 2))
            return RAD_E_SHAPE;
        for (int64_t j = 0; j < width; ++j)
            if (ldi(&idx, seq * idx.stride[0] + j * idx.stride[1]) >= slots) return RAD_E_SHAPE;
        if (anchor) {
            const int64_t read = ldi(&idx, seq * idx.stride[0]);
            const int64_t scratch = ldi(&idx, seq * idx.stride[0] + (width / 2) * idx.stride[1]);
            if (read >= 0 && (scratch < 0 || scratch == read)) return RAD_E_SHAPE;
            if (verify && read >= 0 && (!init || ldi(init, offlin(init, seq))))
                for (int64_t h = 0; h < H; ++h) {
                    const float count = ldt(&s, scratch * s.stride[0] + h * s.stride[1]);
                    if (!std::isfinite(count) || count < 0 || count > 32 ||
                        count != std::floor(count) || (count > 0 && na > count))
                        return RAD_E_SHAPE;
                }
        }
    }
    AVX_PARALLEL_FOR_IF(N * H * V * K >= AVX_PAR_MIN)
    for (int64_t sh = 0; sh < N * H; ++sh) {
        const int64_t seq = sh / H, head = sh % H;
        const int64_t begin = ldi(&cu, seq * cu.stride[0]);
        const int64_t end = ldi(&cu, (seq + 1) * cu.stride[0]);
        const int64_t na = accepted ? ldi(accepted, offlin(accepted, seq)) : 1;
        const int64_t read =
            ldi(&idx, seq * idx.stride[0] + (verify && !anchor ? na - 1 : 0) * idx.stride[1]);
        if (read < 0) {
            for (int64_t t = begin; t < end; ++t)
                for (int64_t e = 0; e < V; ++e) stt(&out, at(out, t, head, e, V), 0);
            continue;
        }
        float* S = scratch_f32((size_t)(V * K + 3 * K));
        float* qn = S + V * K;
        float* kn = qn + K;
        float* decay = kn + K;
        const bool warm = !init || ldi(init, offlin(init, seq)) != 0;
        for (int64_t e = 0; e < V; ++e)
            for (int64_t d = 0; d < K; ++d)
                S[e * K + d] = warm ? ldt(&s, read * s.stride[0] + head * s.stride[1] +
                                                  e * s.stride[2] + d * s.stride[3])
                                    : 0;
        const int64_t scratch =
            anchor ? ldi(&idx, seq * idx.stride[0] + (width / 2) * idx.stride[1]) : 0;
        auto factor_at = [&](int64_t i) {
            return scratch * s.stride[0] + head * s.stride[1] + (i / K) * s.stride[2] +
                   (i % K) * s.stride[3];
        };
        constexpr int factors = 385;
        if (anchor && verify) {
            const int replay = warm ? (int)ldt(&s, factor_at(0)) : 0;
            for (int t = 0; t < (replay ? na : 0); ++t) {
                const int f = 1 + t * factors;
                for (int d = 0; d < 128; ++d) {
                    kn[d] = ldt(&s, factor_at(f + d));
                    decay[d] = ldt(&s, factor_at(f + 256 + d));
                }
                const float bt = ldt(&s, factor_at(f + 384));
                for (int e = 0; e < 128; ++e) {
                    float* row = S + e * K;
                    for (int d = 0; d < 128; d += VF_N)
                        vf_storeu(row + d, vf_mul(vf_loadu(row + d), vf_loadu(decay + d)));
                    const float delta = bt * (ldt(&s, factor_at(f + 128 + e)) - dot(row, kn, K));
                    for (int d = 0; d < 128; d += VF_N)
                        vf_storeu(row + d,
                                  vf_fma(vf_set1(delta), vf_loadu(kn + d), vf_loadu(row + d)));
                }
            }
            for (int e = 0; e < 128; ++e)
                for (int d = 0; d < 128; ++d)
                    stt(&s,
                        read * s.stride[0] + head * s.stride[1] + e * s.stride[2] + d * s.stride[3],
                        S[e * K + d]);
        }
        const float gain = std::exp(ldt(&al, head * laststride(&al)));
        for (int64_t t = begin; t < end; ++t) {
            for (int64_t d = 0; d < K; ++d) {
                qn[d] = ldt(&q, at(q, t, head, d, K));
                kn[d] = ldt(&k, at(k, t, head, d, K));
            }
            const float qi = scale / std::sqrt(dot(qn, qn, K) + eps),
                        ki = 1 / std::sqrt(dot(kn, kn, K) + eps);
            for (int64_t d = 0; d < K; ++d) {
                qn[d] *= qi;
                kn[d] *= ki;
                const float z = gain * (ldt(&g, at(g, t, head, d, K)) +
                                        ldt(&bias, (head * K + d) * laststride(&bias)));
                decay[d] = std::exp(lower / (1 + std::exp(-z)));
            }
            const float beta = 1 / (1 + std::exp(-ldt(&b, at(b, t, head, 0, 1))));
            if (anchor && verify) {
                const int f = 1 + (int)(t - begin) * factors;
                for (int d = 0; d < 128; ++d) {
                    stt(&s, factor_at(f + d), kn[d]);
                    stt(&s, factor_at(f + 256 + d), decay[d]);
                }
                for (int e = 0; e < 128; ++e)
                    stt(&s, factor_at(f + 128 + e), ldt(&v, at(v, t, head, e, V)));
                stt(&s, factor_at(f + 384), beta);
            }
            for (int64_t e = 0; e < V; ++e) {
                float* row = S + e * K;
                int64_t d = 0;
                for (; d + VF_N <= K; d += VF_N)
                    vf_storeu(row + d, vf_mul(vf_loadu(row + d), vf_loadu(decay + d)));
                for (; d < K; ++d) row[d] *= decay[d];
                const float delta = beta * (ldt(&v, at(v, t, head, e, V)) - dot(row, kn, K));
                d = 0;
                for (; d + VF_N <= K; d += VF_N)
                    vf_storeu(row + d, vf_fma(vf_set1(delta), vf_loadu(kn + d), vf_loadu(row + d)));
                for (; d < K; ++d) row[d] += delta * kn[d];
                stt(&out, at(out, t, head, e, V), dot(row, qn, K));
            }
            if ((!anchor && verify) || (!verify && t + 1 == end)) {
                const int64_t write =
                    ldi(&idx, seq * idx.stride[0] + (verify ? t - begin : 0) * idx.stride[1]);
                if (write >= 0)
                    for (int64_t e = 0; e < V; ++e)
                        for (int64_t d = 0; d < K; ++d)
                            stt(&s,
                                write * s.stride[0] + head * s.stride[1] + e * s.stride[2] +
                                    d * s.stride[3],
                                S[e * K + d]);
            }
        }
        if (anchor) stt(&s, factor_at(0), verify ? (float)(end - begin) : 0.0f);
    }
    return RAD_OK;
}
