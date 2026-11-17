/* avx_lili.cpp -- LiLiCorr's candidate correlator, the contracts of libref/ref_lili.cpp: lili_cand,
 * lili_attn, lili_factor and lili_walk. Small ops at a draft step's size (seven positions of eight
 * candidates a sequence), so the arithmetic is libref's in its order, with the lattice's rows
 * widened once into a panel and the rows or (sequence, head) units shared over the threads.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace avx;

/* ------------------------------------------------------------------ lili_cand */
AVX_KERNEL(lili_cand) {
    if (a->n_t < 11) return RAD_E_INVAL;
    for (int i = 0; i < 11; ++i) if (!a->t[i].data) return RAD_E_INVAL;
    const RadTensor *st = &a->t[0], *pas = &a->t[1], *lnw = &a->t[2], *lnb = &a->t[3];
    const RadTensor *upw = &a->t[4], *upb = &a->t[5], *slot = &a->t[6], *rank = &a->t[7];
    const RadTensor *cand = &a->t[8], *fo = &a->t[9], *bo = &a->t[10];
    const int64_t k = p_int(a, "k", 0), world = p_int(a, "world", 1), steps = p_int(a, "steps", 0);
    const int64_t n = p_int(a, "n", 0);
    const float eps = p_f32(a, "eps", 1e-5f);
    const int64_t rk = p_int(a, "row_k", k);
    if (k < 2 || k > 64 || rk < k || world < 1 || world > 16 || steps < 1 || n <= 0)
        return RAD_E_SHAPE;
    const int64_t w = 2 + 2 * rk;
    const int64_t M = rows_of(st, world * w);
    if (M <= 0 || rows_of(pas, n) < M || numel(cand) < M * k || rows_of(fo, n) < M * k ||
        rows_of(bo, n) < M * k)
        return RAD_E_SHAPE;
    if (numel(lnw) < 5 || numel(lnb) < 5 || numel(upw) < n * 5 || numel(upb) < n ||
        rows_of(slot, n) < steps || rows_of(rank, n) < k)
        return RAD_E_SHAPE;
    if (cand->dtype != RAD_I32) return RAD_E_DTYPE;

    /* The up projection widened once: [n, 5] and its bias. */
    std::vector<float> uw((size_t)(n * 5)), ub((size_t)n);
    for (int64_t i = 0; i < n * 5; ++i) uw[(size_t)i] = ldt(upw, i);
    for (int64_t i = 0; i < n; ++i) ub[(size_t)i] = ldt(upb, i);
    float lw[5], lb[5];
    for (int i = 0; i < 5; ++i) { lw[i] = ldt(lnw, i); lb[i] = ldt(lnb, i); }

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t sb = rowoff(st, m, world * w), ss = laststride(st);
        auto S = [&](int64_t r, int64_t i) { return ldt(st, sb + (r * w + i) * ss); };
        float mx = -INFINITY;
        for (int64_t r = 0; r < world; ++r) mx = std::max(mx, S(r, 0));
        double sum = 0.0;
        for (int64_t r = 0; r < world; ++r)
            if (S(r, 0) > -INFINITY) sum += (double)S(r, 1) * std::exp((double)S(r, 0) - mx);
        const double lse = (double)mx + std::log(sum);
        /* A world-way merge of lists sorted by value, ties to the lower id. */
        float lp[64];
        int64_t ids[64];
        int64_t at[16] = {};
        for (int64_t j = 0; j < k; ++j) {
            int64_t br = -1;
            float bv = 0.0f;
            int64_t bid = 0;
            for (int64_t r = 0; r < world; ++r) {
                while (at[r] < k && (int64_t)S(r, 2 + 2 * at[r]) < 0) ++at[r];
                if (at[r] >= k) continue;
                const float v = S(r, 3 + 2 * at[r]);
                const int64_t id = (int64_t)S(r, 2 + 2 * at[r]);
                if (br < 0 || v > bv || (v == bv && id < bid)) { br = r; bv = v; bid = id; }
            }
            ids[j] = br < 0 ? 0 : bid;
            lp[j] = br < 0 ? -1e4f : (float)((double)bv - lse);
            if (br >= 0) ++at[br];
            ((int32_t*)cand->data)[rowoff(cand, m, k) + j * laststride(cand)] = (int32_t)ids[j];
        }
        const int64_t s = m % steps;
        const int64_t pb = rowoff(pas, m, n), ps = laststride(pas);
        const int64_t slb = rowoff(slot, s, n), sls = laststride(slot);
        for (int64_t j = 0; j < k; ++j) {
            const float f[5] = { lp[j], std::exp(lp[j]), lp[j] - lp[0],
                                 (float)j / (float)(k - 1), j == 0 ? 1.0f : 0.0f };
            float mean = 0.0f;
            for (int i = 0; i < 5; ++i) mean += f[i];
            mean /= 5.0f;
            float var = 0.0f;
            for (int i = 0; i < 5; ++i) var += (f[i] - mean) * (f[i] - mean);
            var /= 5.0f;
            const float rs = 1.0f / std::sqrt(var + eps);
            float y[5];
            for (int i = 0; i < 5; ++i) y[i] = (f[i] - mean) * rs * lw[i] + lb[i];
            const int64_t row = m * k + j;
            const int64_t fb = rowoff(fo, row, n), fs = laststride(fo);
            const int64_t bb = rowoff(bo, row, n), bs = laststride(bo);
            const int64_t rb = rowoff(rank, j, n), rks = laststride(rank);
            for (int64_t c = 0; c < n; ++c) {
                float u = ub[(size_t)c];
                for (int i = 0; i < 5; ++i) u += uw[(size_t)(c * 5 + i)] * y[i];
                stt(fo, fb + c * fs, u / (1.0f + std::exp(-u)));
                stt(bo, bb + c * bs,
                    ldt(pas, pb + c * ps) + ldt(slot, slb + c * sls) + ldt(rank, rb + c * rks));
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ lili_attn */
AVX_KERNEL(lili_attn) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *qkv = &a->t[0], *rel = &a->t[1], *same = &a->t[2], *o = &a->t[3];
    const int64_t steps = p_int(a, "steps", 0), k = p_int(a, "k", 0);
    const int64_t nh = p_int(a, "n_head", 0), hd = p_int(a, "head_dim", 0);
    if (steps < 1 || k < 1 || nh < 1 || hd < 1) return RAD_E_SHAPE;
    const int64_t L = steps * k, n = nh * hd;
    const int64_t M = rows_of(qkv, 3 * n);
    if (M <= 0 || M % L || rows_of(o, n) < M) return RAD_E_SHAPE;
    const int64_t rw = rel->rank ? rel->shape[rel->rank - 1] : 0;
    if (rw < 2 * steps - 1 || rows_of(rel, rw) < nh || numel(same) < nh) return RAD_E_SHAPE;
    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)hd));
    const int64_t qs = laststride(qkv), os = laststride(o);
    const int64_t units = (M / L) * nh;

    AVX_PARALLEL_FOR_IF(units >= 2)
    for (int64_t u = 0; u < units; ++u) {
        const int64_t q0 = u / nh * L, h = u % nh;
        /* The lattice's q, k and v of this head, widened once. */
        float* pan = scratch_f32((size_t)(3 * L * hd + L) + 16);
        float *Q = pan, *K = pan + L * hd, *V = pan + 2 * L * hd, *sc = pan + 3 * L * hd;
        for (int64_t i = 0; i < L; ++i) {
            const int64_t b = rowoff(qkv, q0 + i, 3 * n);
            for (int64_t d = 0; d < hd; ++d) {
                Q[i * hd + d] = ldt(qkv, b + (h * hd + d) * qs);
                K[i * hd + d] = ldt(qkv, b + (n + h * hd + d) * qs);
                V[i * hd + d] = ldt(qkv, b + (2 * n + h * hd + d) * qs);
            }
        }
        const float sm = ldt(same, h);
        const int64_t relb = rowoff(rel, h, rw), rels = laststride(rel);
        for (int64_t i = 0; i < L; ++i) {
            const int64_t si = i / k;
            float mx = -INFINITY;
            for (int64_t j = 0; j < L; ++j) {
                float dot = 0.0f;
                for (int64_t d = 0; d < hd; ++d) dot += Q[i * hd + d] * K[j * hd + d];
                const int64_t sj = j / k;
                float b = ldt(rel, relb + (si - sj + steps) * rels);
                if (si == sj) b += sm;
                sc[j] = dot * scale + b;
                mx = std::max(mx, sc[j]);
            }
            float sum = 0.0f;
            for (int64_t j = 0; j < L; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
            const int64_t ob = rowoff(o, q0 + i, n) + h * hd * os;
            for (int64_t d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int64_t j = 0; j < L; ++j) acc += sc[j] * V[j * hd + d];
                stt(o, ob + d * os, acc / sum);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ lili_factor */
AVX_KERNEL(lili_factor) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *h = &a->t[0], *an = &a->t[1], *y = &a->t[2];
    const int64_t L = p_int(a, "L", 0), n = p_int(a, "n", 0);
    if (L < 1 || n < 1) return RAD_E_SHAPE;
    const int64_t M = rows_of(h, n);
    if (M <= 0 || M % L || rows_of(an, n) < M / L || rows_of(y, 3 * n) < M) return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(M >= 8)
    for (int64_t i = 0; i < M; ++i) {
        const int64_t hb = rowoff(h, i, n), ab = rowoff(an, i / L, n), yb = rowoff(y, i, 3 * n);
        const int64_t hs = laststride(h), as = laststride(an), ys = laststride(y);
        for (int64_t c = 0; c < n; ++c) {
            const float hv = ldt(h, hb + c * hs), av = ldt(an, ab + c * as);
            stt(y, yb + c * ys, hv);
            stt(y, yb + (n + c) * ys, av);
            stt(y, yb + (2 * n + c) * ys, hv * av);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ lili_walk */
AVX_KERNEL(lili_walk) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *ed = &a->t[0], *ao = &a->t[1], *cd = &a->t[2], *tk = &a->t[3];
    const RadTensor* sco = a->n_t > 4 && a->t[4].data ? &a->t[4] : nullptr;
    const int64_t steps = p_int(a, "steps", 0), k = p_int(a, "k", 0), n = p_int(a, "n", 0);
    const float eps = p_f32(a, "eps", 1e-4f), scale = p_f32(a, "scale", 1.0f);
    if (steps < 1 || k < 1 || n < 1) return RAD_E_SHAPE;
    const int64_t M = rows_of(ao, n);
    if (M <= 0 || rows_of(ed, 2 * n) < M * steps * k || numel(cd) < M * steps * k ||
        numel(tk) < M * steps)
        return RAD_E_SHAPE;
    if (cd->dtype != RAD_I32 || tk->dtype != RAD_I32) return RAD_E_DTYPE;
    if (sco && numel(sco) < M * steps * k * k) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t R = steps * k;
        float* pan = scratch_f32((size_t)((2 * R + 1) * n + R * k) + 16);
        float *vo = pan, *vi = pan + R * n, *va = pan + 2 * R * n, *sc = pan + (2 * R + 1) * n;
        auto vec = [&](const RadTensor* t, int64_t row, int64_t pitch, int64_t col0, float* v) {
            const int64_t b = rowoff(t, row, pitch) + col0 * laststride(t), s = laststride(t);
            double ss = 0.0;
            for (int64_t c = 0; c < n; ++c) { v[c] = ldt(t, b + c * s); ss += (double)v[c] * v[c]; }
            const float inv = 1.0f / std::max((float)std::sqrt(ss), eps);
            for (int64_t c = 0; c < n; ++c) v[c] *= inv;
        };
        vec(ao, m, n, 0, va);
        for (int64_t r = 0; r < R; ++r) {
            vec(ed, m * R + r, 2 * n, 0, vo + r * n);
            vec(ed, m * R + r, 2 * n, n, vi + r * n);
        }
        auto dot = [&](const float* x, const float* y) {
            float s = 0.0f;
            for (int64_t c = 0; c < n; ++c) s += x[c] * y[c];
            return s * scale;
        };
        for (int64_t j = 0; j < k; ++j) {
            const float v = dot(va, vi + j * n);
            for (int64_t i = 0; i < k; ++i) sc[i * k + j] = v;
        }
        for (int64_t s = 1; s < steps; ++s)
            for (int64_t i = 0; i < k; ++i)
                for (int64_t j = 0; j < k; ++j)
                    sc[(s * k + i) * k + j] = dot(vo + ((s - 1) * k + i) * n, vi + (s * k + j) * n);
        int64_t prev = 0;
        for (int64_t s = 0; s < steps; ++s) {
            int64_t best = 0;
            for (int64_t j = 1; j < k; ++j)
                if (sc[(s * k + prev) * k + j] > sc[(s * k + prev) * k + best]) best = j;
            const int64_t id = ldi(cd, rowoff(cd, m * steps + s, k) + best * laststride(cd));
            ((int32_t*)tk->data)[rowoff(tk, m, steps) + s * laststride(tk)] = (int32_t)id;
            prev = best;
        }
        if (sco)
            for (int64_t e = 0; e < R * k; ++e) stt(sco, m * R * k + e, sc[e]);
    }
    return RAD_OK;
}
