/* ref_lili.cpp -- LiLiCorr's candidate correlator (NVIDIA's Nemotron-3.5-Bolt drafter; vLLM's
 * model_executor/models/lilicorr.py), the four ops a block drafter's query pass ends in:
 *
 *   lili_cand    the candidates of every draft position from the target head's top-k, their five
 *                rank features through LayerNorm, the up projection and SiLU, and the lattice row's
 *                additive part: the position's own state plus a slot and a rank embedding
 *   lili_attn    attention over each sequence's lattice of steps x k rows, with a learned bias per
 *                head on the slot distance and on the same slot
 *   lili_factor  [h | a | h * a] a row, a = its sequence's anchor
 *   lili_walk    the factored edge scores and the greedy walk over them
 *
 * A sequence's lattice is `steps` draft positions of `k` candidates, row s * k + j for position s's
 * j-th candidate, the sequences one after another. Everything runs in f32 and rounds once to the
 * output's dtype.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace ref;

/* ------------------------------------------------------------------ lili_cand */
/* stats [M, world * (2 + 2 row_k)] f32: each rank's sample_logprobs row (max, sum of exp(l - max),
 * row_k >= k (id, logit) pairs descending), all-gathered row-interleaved; row_k defaults to k (a
 * caller widens it so a row is whole 16-byte words for the all-gather). The ranks merge as one row would:
 * the maximum and the rescaled sum give the log-normaliser, the pairs the k largest by value, ties
 * to the lower id. For candidate j of row m, with lp_j = logit_j - lse:
 *
 *   feat  = [lp_j, exp(lp_j), lp_j - lp_0, j / (k - 1), j == 0]
 *   f     = silu(up_w . layernorm(feat; ln_w, ln_b, eps) + up_b)              -> feat out, row m*k+j
 *   base  = pas[m] + slot[m % steps] + rank[j]                                -> base out, row m*k+j
 *   cand  = id_j                                                              -> cand [M, k] i32
 */
int ref_lili_cand(const RadArgs* a, RadStream) {
    if (a->n_t < 11) return RAD_E_INVAL;
    for (int i = 0; i < 11; ++i) if (!a->t[i].data) return RAD_E_INVAL;
    const RadTensor *st = &a->t[0], *pas = &a->t[1], *lnw = &a->t[2], *lnb = &a->t[3];
    const RadTensor *upw = &a->t[4], *upb = &a->t[5], *slot = &a->t[6], *rank = &a->t[7];
    const RadTensor *cand = &a->t[8], *fo = &a->t[9], *bo = &a->t[10];
    const int64_t k = p_int(a, "k", 0), world = p_int(a, "world", 1), steps = p_int(a, "steps", 0);
    const int64_t n = p_int(a, "n", 0);
    const float eps = p_f32(a, "eps", 1e-5f);
    const int64_t rk = p_int(a, "row_k", k);
    if (k < 2 || k > 64 || rk < k || world < 1 || steps < 1 || n <= 0) return RAD_E_SHAPE;
    const int64_t w = 2 + 2 * rk;
    const int64_t M = rows_of(st, world * w);
    if (M <= 0 || rows_of(pas, n) < M || numel(cand) < M * k || rows_of(fo, n) < M * k ||
        rows_of(bo, n) < M * k)
        return RAD_E_SHAPE;
    if (numel(lnw) < 5 || numel(lnb) < 5 || numel(upw) < n * 5 || numel(upb) < n ||
        rows_of(slot, n) < steps || rows_of(rank, n) < k)
        return RAD_E_SHAPE;
    if (cand->dtype != RAD_I32) return RAD_E_DTYPE;

    REF_PAR_FOR(M * k * n * 8)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t sb = rowoff(st, m, world * w), ss = laststride(st);
        auto S = [&](int64_t r, int64_t i) { return ldt(st, sb + (r * w + i) * ss); };
        /* The log-normaliser over every rank's shard. */
        float mx = -INFINITY;
        for (int64_t r = 0; r < world; ++r) mx = std::max(mx, S(r, 0));
        double sum = 0.0;
        for (int64_t r = 0; r < world; ++r)
            if (S(r, 0) > -INFINITY) sum += (double)S(r, 1) * std::exp((double)S(r, 0) - mx);
        const double lse = (double)mx + std::log(sum);
        /* The k largest pairs of the world * k, by value, ties to the lower id. */
        std::vector<std::pair<float, int64_t>> all;
        all.reserve((size_t)(world * k));
        for (int64_t r = 0; r < world; ++r)
            for (int64_t j = 0; j < k; ++j) {
                const int64_t id = (int64_t)S(r, 2 + 2 * j);
                if (id >= 0) all.push_back({ S(r, 3 + 2 * j), id });
            }
        std::sort(all.begin(), all.end(), [](const auto& x, const auto& y) {
            return x.first > y.first || (x.first == y.first && x.second < y.second);
        });
        float lp[64];
        int64_t ids[64];
        for (int64_t j = 0; j < k; ++j) {
            const bool have_j = j < (int64_t)all.size();
            ids[j] = have_j ? all[(size_t)j].second : 0;
            lp[j] = have_j ? (float)((double)all[(size_t)j].first - lse) : -1e4f;
            ((int32_t*)cand->data)[rowoff(cand, m, k) + j * laststride(cand)] = (int32_t)ids[j];
        }
        const int64_t s = m % steps;
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
            for (int i = 0; i < 5; ++i) y[i] = (f[i] - mean) * rs * ldt(lnw, i) + ldt(lnb, i);
            const int64_t row = m * k + j;
            const int64_t fb = rowoff(fo, row, n), fs = laststride(fo);
            const int64_t bb = rowoff(bo, row, n), bs = laststride(bo);
            for (int64_t c = 0; c < n; ++c) {
                float u = ldt(upb, c);
                for (int i = 0; i < 5; ++i) u += ldt(upw, c * 5 + i) * y[i];
                stt(fo, fb + c * fs, u / (1.0f + std::exp(-u)));
                const float b = ldt(pas, rowoff(pas, m, n) + c * laststride(pas)) +
                                ldt(slot, rowoff(slot, s, n) + c * laststride(slot)) +
                                ldt(rank, rowoff(rank, j, n) + c * laststride(rank));
                stt(bo, bb + c * bs, b);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ lili_attn */
/* qkv [M, 3 * n_head * head_dim] (q | k | v, heads contiguous within each), M = sequences * L with
 * L = steps * k. Row i of a lattice sits at slot s_i = i / k. Per sequence and head:
 *
 *   score(i, j) = q_i . k_j * scale + rel[h][s_i - s_j + steps] + (s_i == s_j ? same[h] : 0)
 *   out_i       = softmax_j(score(i, .)) . v
 *
 * Every row sees every row of its lattice: the bias is a preference, not a mask. */
int ref_lili_attn(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
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

    REF_PAR_FOR(M * nh * L * hd * 2)
    for (int64_t u = 0; u < M * nh; ++u) {
        const int64_t i = u / nh, h = u % nh;
        const int64_t q0 = i / L * L, si = (i - q0) / k;
        const int64_t qb = rowoff(qkv, i, 3 * n) + h * hd * qs;
        std::vector<float> sc((size_t)L);
        float mx = -INFINITY;
        for (int64_t j = 0; j < L; ++j) {
            const int64_t kb = rowoff(qkv, q0 + j, 3 * n) + (n + h * hd) * qs;
            float dot = 0.0f;
            for (int64_t d = 0; d < hd; ++d) dot += ldt(qkv, qb + d * qs) * ldt(qkv, kb + d * qs);
            const int64_t sj = j / k;
            float b = ldt(rel, rowoff(rel, h, rw) + (si - sj + steps) * laststride(rel));
            if (si == sj) b += ldt(same, h);
            sc[(size_t)j] = dot * scale + b;
            mx = std::max(mx, sc[(size_t)j]);
        }
        float sum = 0.0f;
        for (int64_t j = 0; j < L; ++j) { sc[(size_t)j] = std::exp(sc[(size_t)j] - mx); sum += sc[(size_t)j]; }
        const int64_t ob = rowoff(o, i, n) + h * hd * os;
        for (int64_t d = 0; d < hd; ++d) {
            float acc = 0.0f;
            for (int64_t j = 0; j < L; ++j)
                acc += sc[(size_t)j] * ldt(qkv, rowoff(qkv, q0 + j, 3 * n) + (2 * n + h * hd + d) * qs);
            stt(o, ob + d * os, acc / sum);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ lili_factor */
/* y[i] = [h_i | a_s | h_i * a_s] with s = i / L, the factor projection's input. */
int ref_lili_factor(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *h = &a->t[0], *an = &a->t[1], *y = &a->t[2];
    const int64_t L = p_int(a, "L", 0), n = p_int(a, "n", 0);
    if (L < 1 || n < 1) return RAD_E_SHAPE;
    const int64_t M = rows_of(h, n);
    if (M <= 0 || M % L || rows_of(an, n) < M / L || rows_of(y, 3 * n) < M) return RAD_E_SHAPE;
    REF_PAR_FOR(M * n * 3)
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
/* edges [M * steps * k, 2n]: each candidate's outgoing | incoming vector, aout [M, n] the anchor's
 * outgoing one, cand [M * steps, k] i32. Vectors are normalised as F.normalize (v / max(|v|, eps)).
 *
 *   start[j]       = scale * aout . in[0][j]
 *   pair[s][i][j]  = scale * out[s][i] . in[s + 1][j]
 *   walk           p_0 = argmax start, p_s = argmax pair[s - 1][p_{s-1}]  (first index on ties)
 *   tokens[m][s]   = cand[m * steps + s][p_s]
 *
 * `scores` [M, steps, k, k], when present, is vLLM's tensor: row 0 is start for every i. */
int ref_lili_walk(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
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

    REF_PAR_FOR(M * steps * k * k * n)
    for (int64_t m = 0; m < M; ++m) {
        auto vec = [&](const RadTensor* t, int64_t row, int64_t pitch, int64_t col0, float* v) {
            const int64_t b = rowoff(t, row, pitch) + col0 * laststride(t);
            double ss = 0.0;
            for (int64_t c = 0; c < n; ++c) { v[c] = ldt(t, b + c * laststride(t)); ss += (double)v[c] * v[c]; }
            const float inv = 1.0f / std::max((float)std::sqrt(ss), eps);
            for (int64_t c = 0; c < n; ++c) v[c] *= inv;
        };
        std::vector<float> va((size_t)n), vo((size_t)(steps * k * n)), vi((size_t)(steps * k * n));
        vec(ao, m, n, 0, va.data());
        for (int64_t r = 0; r < steps * k; ++r) {
            vec(ed, m * steps * k + r, 2 * n, 0, &vo[(size_t)(r * n)]);
            vec(ed, m * steps * k + r, 2 * n, n, &vi[(size_t)(r * n)]);
        }
        auto dot = [&](const float* x, const float* y) {
            float s = 0.0f;
            for (int64_t c = 0; c < n; ++c) s += x[c] * y[c];
            return s * scale;
        };
        std::vector<float> sc((size_t)(steps * k * k));
        for (int64_t j = 0; j < k; ++j) {
            const float v = dot(va.data(), &vi[(size_t)(j * n)]);
            for (int64_t i = 0; i < k; ++i) sc[(size_t)(i * k + j)] = v;
        }
        for (int64_t s = 1; s < steps; ++s)
            for (int64_t i = 0; i < k; ++i)
                for (int64_t j = 0; j < k; ++j)
                    sc[(size_t)((s * k + i) * k + j)] =
                        dot(&vo[(size_t)(((s - 1) * k + i) * n)], &vi[(size_t)((s * k + j) * n)]);
        int64_t prev = 0;
        for (int64_t s = 0; s < steps; ++s) {
            int64_t best = 0;
            for (int64_t j = 1; j < k; ++j)
                if (sc[(size_t)((s * k + prev) * k + j)] > sc[(size_t)((s * k + prev) * k + best)]) best = j;
            const int64_t id = ld_int(cd->dtype, cd->data, rowoff(cd, m * steps + s, k) + best * laststride(cd));
            ((int32_t*)tk->data)[rowoff(tk, m, steps) + s * laststride(tk)] = (int32_t)id;
            prev = best;
        }
        if (sco)
            for (int64_t e = 0; e < steps * k * k; ++e) stt(sco, m * steps * k * k + e, sc[(size_t)e]);
    }
    return RAD_OK;
}
