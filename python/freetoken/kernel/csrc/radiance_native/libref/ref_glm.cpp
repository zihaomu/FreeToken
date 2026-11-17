#include "ref_common.h"
#include "ref_ops.h"
#include <algorithm>
#include <cmath>
using namespace ref;

/* Routing and index head gates are F32 projections in GLM, even with BF16 hidden states. */
int ref_gemm_nt_f32(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const auto &x = a->t[0], &w = a->t[1], &y = a->t[2];
    const int64_t n = p_int(a, "N", 0), k = p_int(a, "K", 0);
    if (n < 1 || k < 1 || x.rank != 2 || w.rank != 2 || y.rank != 2 || x.shape[0] < 1 ||
        x.shape[1] != k || w.shape[0] != n || w.shape[1] != k || y.shape[0] != x.shape[0] ||
        y.shape[1] != n)
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || w.dtype != RAD_F32 || y.dtype != RAD_F32) return RAD_E_DTYPE;
    if (!x.data || !w.data || !y.data) return RAD_E_INVAL;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < x.shape[0]; ++r)
        for (int64_t col = 0; col < n; ++col) {
            float sum = 0;
            for (int64_t j = 0; j < k; ++j)
                sum += ldt(&x, r * x.stride[0] + j * x.stride[1]) *
                       ldt(&w, col * w.stride[0] + j * w.stride[1]);
            stt(&y, r * y.stride[0] + col * y.stride[1], sum);
        }
    return RAD_OK;
}

/* Single-group noaux_tc routing: correction changes selection, never the mixture weights. */
int ref_router_sigmoid_topk(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const auto &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &weights = a->t[3];
    const int64_t ne = p_int(a, "n_expert", 0), k = p_int(a, "top_k", 0);
    const float scale = p_f32(a, "route_scale", 0);
    const bool norm = p_int(a, "norm", 0) != 0;
    if (ne < 1 || ne > 512 || k < 1 || k > ne || !std::isfinite(scale) || scale <= 0 ||
        lg.rank != 2 || lg.shape[0] < 1 || lg.shape[1] != ne || bias.rank != 1 ||
        bias.shape[0] != ne || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 || ids.rank != 2 ||
        weights.rank != 2 || ids.shape[0] != lg.shape[0] || weights.shape[0] != lg.shape[0] ||
        ids.shape[1] != k || weights.shape[1] != k || !lg.data || !bias.data || !ids.data ||
        !weights.data)
        return RAD_E_SHAPE;
    for (int64_t r = 0; r < lg.shape[0]; ++r) {
        float scores[512], select[512], chosen[512];
        int index[512];
        for (int64_t e = 0; e < ne; ++e) {
            scores[e] = act_sigmoid(ldt(&lg, rowoff(&lg, r, ne) + e * laststride(&lg)));
            select[e] = scores[e] + ldt(&bias, e * laststride(&bias));
        }
        float sum = 0;
        for (int64_t j = 0; j < k; ++j) {
            int best = -1;
            for (int64_t e = 0; e < ne; ++e)
                if (select[e] > -INFINITY && (best < 0 || select[e] > select[best])) best = (int)e;
            if (best < 0) return RAD_E_SHAPE;
            index[j] = best;
            chosen[j] = scores[best];
            sum += chosen[j];
            select[best] = -INFINITY;
        }
        for (int64_t j = 0; j < k; ++j) {
            st_int(ids.dtype, ids.data, rowoff(&ids, r, k) + j * laststride(&ids), index[j]);
            stt(&weights, rowoff(&weights, r, k) + j * laststride(&weights),
                chosen[j] * (norm ? scale / (sum + 1e-20f) : scale));
        }
    }
    return RAD_OK;
}

/* Group-limited noaux_tc routing (DeepSeek-V3's, Ling-3.0's): the experts fall into `n_group`
 * equal groups, a group ranks by the sum of its two highest `p + bias`, and selection runs over
 * the experts of the `topk_group` best groups only. Weights as in router_sigmoid_topk. */
int ref_router_sigmoid_group_topk(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const auto &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &weights = a->t[3];
    const int64_t ne = p_int(a, "n_expert", 0), k = p_int(a, "top_k", 0),
                  ng = p_int(a, "n_group", 0), kg = p_int(a, "topk_group", 0);
    const float scale = p_f32(a, "route_scale", 0);
    const bool norm = p_int(a, "norm", 0) != 0;
    if (ne < 1 || ne > 512 || k < 1 || ng < 1 || ne % ng || ne / ng < 2 || kg < 1 || kg > ng ||
        k > kg * (ne / ng) || !std::isfinite(scale) || scale <= 0 || lg.rank != 2 ||
        lg.shape[0] < 1 || lg.shape[1] != ne || bias.rank != 1 || bias.shape[0] != ne ||
        bias.dtype != RAD_F32 || ids.dtype != RAD_I32 || ids.rank != 2 || weights.rank != 2 ||
        ids.shape[0] != lg.shape[0] || weights.shape[0] != lg.shape[0] || ids.shape[1] != k ||
        weights.shape[1] != k || !lg.data || !bias.data || !ids.data || !weights.data)
        return RAD_E_SHAPE;
    const int64_t gsz = ne / ng;
    for (int64_t r = 0; r < lg.shape[0]; ++r) {
        float scores[512], select[512], chosen[512], gscore[512];
        int index[512];
        bool keep[512] = {};
        for (int64_t e = 0; e < ne; ++e) {
            scores[e] = act_sigmoid(ldt(&lg, rowoff(&lg, r, ne) + e * laststride(&lg)));
            select[e] = scores[e] + ldt(&bias, e * laststride(&bias));
        }
        for (int64_t gi = 0; gi < ng; ++gi) {
            float t0 = -INFINITY, t1 = -INFINITY;
            for (int64_t e = gi * gsz; e < (gi + 1) * gsz; ++e) {
                if (select[e] > t0) { t1 = t0; t0 = select[e]; }
                else if (select[e] > t1) t1 = select[e];
            }
            gscore[gi] = t0 + t1;
        }
        for (int64_t j = 0; j < kg; ++j) {
            int best = -1;
            for (int64_t gi = 0; gi < ng; ++gi)
                if (gscore[gi] > -INFINITY && (best < 0 || gscore[gi] > gscore[best])) best = (int)gi;
            if (best < 0) return RAD_E_SHAPE;
            for (int64_t e = best * gsz; e < (best + 1) * gsz; ++e) keep[e] = true;
            gscore[best] = -INFINITY;
        }
        float sum = 0;
        for (int64_t j = 0; j < k; ++j) {
            int best = -1;
            for (int64_t e = 0; e < ne; ++e)
                if (keep[e] && select[e] > -INFINITY && (best < 0 || select[e] > select[best]))
                    best = (int)e;
            if (best < 0) return RAD_E_SHAPE;
            index[j] = best;
            chosen[j] = scores[best];
            sum += chosen[j];
            select[best] = -INFINITY;
        }
        for (int64_t j = 0; j < k; ++j) {
            st_int(ids.dtype, ids.data, rowoff(&ids, r, k) + j * laststride(&ids), index[j]);
            stt(&weights, rowoff(&weights, r, k) + j * laststride(&weights),
                chosen[j] * (norm ? scale / (sum + 1e-20f) : scale));
        }
    }
    return RAD_OK;
}

/* Ling-3.0's clamp: the SiLU is taken first and its value clipped above, the up branch on both
 * sides (vLLM's SwigluStepAndMul). swiglu_clamped clips the gate before the SiLU instead. */
int ref_swiglu_step(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const auto &gu = a->t[0], &y = a->t[1];
    const int64_t n = p_int(a, "n", 0);
    const float limit = p_f32(a, "limit", 0);
    if (n < 1 || !std::isfinite(limit) || limit <= 0 || gu.rank != 2 || y.rank != 2 || !gu.data ||
        !y.data || y.shape[0] < 1 || gu.shape[0] != y.shape[0] || gu.shape[1] != 2 * n ||
        y.shape[1] != n)
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < y.shape[0]; ++r)
        for (int64_t i = 0; i < n; ++i) {
            const int64_t b = rowoff(&gu, r, 2 * n), s = laststride(&gu);
            const float g = std::min(act_silu(ldt(&gu, b + i * s)), limit);
            const float u = std::clamp(ldt(&gu, b + (n + i) * s), -limit, limit);
            stt(&y, rowoff(&y, r, n) + i * laststride(&y), g * u);
        }
    return RAD_OK;
}

/* GLM clamps the gate only above, but clamps the up branch on both sides. */
int ref_swiglu_clamped(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const auto &gu = a->t[0], &y = a->t[1];
    const int64_t n = p_int(a, "n", 0);
    const float limit = p_f32(a, "limit", 0);
    if (n < 1 || !std::isfinite(limit) || limit <= 0 || gu.rank != 2 || y.rank != 2 || !gu.data ||
        !y.data || y.shape[0] < 1 || gu.shape[0] != y.shape[0] || gu.shape[1] != 2 * n ||
        y.shape[1] != n)
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < y.shape[0]; ++r)
        for (int64_t i = 0; i < n; ++i) {
            const int64_t b = rowoff(&gu, r, 2 * n), s = laststride(&gu);
            const float g = std::min(ldt(&gu, b + i * s), limit);
            const float u = std::clamp(ldt(&gu, b + (n + i) * s), -limit, limit);
            stt(&y, rowoff(&y, r, n) + i * laststride(&y), act_silu(g) * u);
        }
    return RAD_OK;
}
