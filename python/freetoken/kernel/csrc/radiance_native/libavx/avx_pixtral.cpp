/* Nemotron's text/vision rotary, patch merger and materialized BF16 attention.
 * Separate ops preserve existing rotary, attention and merger contracts. */
#include "avx_vec.h"
#include "avx_common.h"

#include "rad_yarn.h"

using namespace avx;

AVX_KERNEL(pixtral_rope) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *rc = &a->t[1];
    // The issue's row views can be smaller than the resolved M capacity band.
    const int64_t M = x->rank == 2 ? x->shape[0] : 0;
    const int64_t D = p_int(a, "head_dim", 0), H = p_int(a, "n_head", 0);
    const float theta = p_f32(a, "theta", 0);
    if (M <= 0 || M > p_int(a, "M", 0) || D <= 0 || D % 4 || H <= 0 || !(theta > 0) ||
        !std::isfinite(theta) || x->rank != 2 || x->shape[0] < M || x->shape[1] != 3 * H * D ||
        rc->rank != 2 || rc->shape[0] != 2 || rc->shape[1] != M)
        return RAD_E_SHAPE;
    if (rc->dtype != RAD_I32 || (x->dtype != RAD_BF16 && x->dtype != RAD_F32)) return RAD_E_DTYPE;
    for (int64_t r = 0; r < 2; ++r)
        for (int64_t p = 0; p < M; ++p)
            if (ldi(rc, r * rc->stride[0] + p * rc->stride[1]) < 0) return RAD_E_INVAL;
    AVX_PARALLEL_FOR_IF(M * H * D >= AVX_PAR_MIN)
    for (int64_t p = 0; p < M; ++p) {
        for (int64_t i = 0; i < D / 2; ++i) {
            const int64_t axis = i / (D / 4), j = i % (D / 4);
            const float pos = (float)ldi(rc, axis * rc->stride[0] + p * rc->stride[1]);
            const float inv = 1.0f / sc_pow(theta, (float)(4 * j + 2 * axis) / D);
            const float angle = pos * inv;
            const float c = narrow1(x->dtype, sc_cos(angle)), s = narrow1(x->dtype, sc_sin(angle));
            for (int64_t h = 0; h < 2 * H; ++h) {
                const int64_t u = rowoff(x, p, 3 * H * D) + (h * D + i) * laststride(x);
                const int64_t v = u + (D / 2) * laststride(x);
                const float a0 = ldt(x, u), b0 = ldt(x, v);
                /* Transformers narrows cos/sin and each product before adding BF16 tensors. */
                stt(x, u, narrow1(x->dtype, a0 * c) - narrow1(x->dtype, b0 * s));
                stt(x, v, narrow1(x->dtype, b0 * c) + narrow1(x->dtype, a0 * s));
            }
        }
    }
    return RAD_OK;
}

AVX_KERNEL(pixtral_merge) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *rc = &a->t[1], *y = &a->t[2];
    const int64_t M = x->rank == 2 ? x->shape[0] : 0;
    const int64_t N = p_int(a, "n", 0), K = p_int(a, "merge", 0);
    if (M <= 0 || M > p_int(a, "M", 0) || N <= 0 || K <= 0 || K > M || M / K < K || M % (K * K) ||
        x->rank != 2 || x->shape[0] != M || x->shape[1] != N || rc->rank != 2 ||
        rc->shape[0] != 4 || rc->shape[1] != M || y->rank != 2 || y->shape[0] != M / (K * K) ||
        y->shape[1] != N * K * K)
        return RAD_E_SHAPE;
    if (rc->dtype != RAD_I32 || x->dtype != y->dtype ||
        (x->dtype != RAD_BF16 && x->dtype != RAD_F32))
        return RAD_E_DTYPE;
    const auto coord = [&](int axis, int64_t p) {
        return ldi(rc, axis * rc->stride[0] + p * rc->stride[1]);
    };
    /* Verify full raster grids before any write, including variable image widths. The grid at
     * each image's first patch defines its boundary; no image is merged with the next one. */
    for (int64_t p = 0; p < M;) {
        const int64_t h = coord(2, p), w = coord(3, p);
        if (h <= 0 || w <= 0 || h % K || w % K || h > (M - p) / w) return RAD_E_INVAL;
        for (int64_t j = 0; j < h * w; ++j)
            if (coord(0, p + j) != j / w || coord(1, p + j) != j % w || coord(2, p + j) != h ||
                coord(3, p + j) != w)
                return RAD_E_INVAL;
        p += h * w;
    }
    for (int64_t p = 0; p < M;) {
        const int64_t h = coord(2, p), w = coord(3, p);
        for (int64_t r = 0; r < h / K; ++r)
            for (int64_t c = 0; c < w / K; ++c) {
                const int64_t dst = p / (K * K) + r * (w / K) + c;
                for (int64_t n = 0; n < N; ++n)
                    for (int64_t dy = 0; dy < K; ++dy)
                        for (int64_t dx = 0; dx < K; ++dx) {
                            const int64_t src = p + (r * K + dy) * w + c * K + dx;
                            const int64_t col = (n * K + dy) * K + dx;
                            stt(y, rowoff(y, dst, N * K * K) + col * laststride(y),
                                ldt(x, rowoff(x, src, N) + n * laststride(x)));
                        }
            }
        p += h * w;
    }
    return RAD_OK;
}

AVX_KERNEL(ministral_rope) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *pos = &a->t[1];
    const int64_t M = x->rank == 2 ? x->shape[0] : 0;
    const int64_t D = p_int(a, "head_dim", 0), H = p_int(a, "n_head", 0),
                  K = p_int(a, "n_head_kv", 0);
    const float theta = p_f32(a, "theta", 0);
    const RadYarn yarn = rad_yarn_args(a);
    if (M <= 0 || M > p_int(a, "M", 0) || D <= 0 || D > 2048 || D % 2 || H <= 0 || H > 1024 ||
        K <= 0 || K > H || H % K || x->shape[1] != (H + 2 * K) * D || pos->rank != 1 ||
        numel(pos) != M || !(theta > 0) || !std::isfinite(theta) || !rad_yarn_valid(yarn))
        return RAD_E_SHAPE;
    if (pos->dtype != RAD_I32 || (x->dtype != RAD_BF16 && x->dtype != RAD_F32)) return RAD_E_DTYPE;
    for (int64_t p = 0; p < M; ++p)
        if (ldi(pos, offlin(pos, p)) < 0) return RAD_E_INVAL;
    float freq[1024];
    for (int64_t i = 0; i < D / 2; ++i)
        freq[i] = yarn.frequency(1.0f / sc_pow(theta, (float)(2 * i) / (float)D), (int)i);
    AVX_PARALLEL_FOR_IF(M * (H + K) * D >= AVX_PAR_MIN)
    for (int64_t p = 0; p < M; ++p) {
        const float position = (float)(ldi(pos, offlin(pos, p)));
        for (int64_t i = 0; i < D / 2; ++i) {
            const float angle = position * freq[i];
            const float c = narrow1(x->dtype, sc_cos(angle) * yarn.magnitude);
            const float s = narrow1(x->dtype, sc_sin(angle) * yarn.magnitude);
            for (int64_t h = 0; h < H + K; ++h) {
                const int64_t u = rowoff(x, p, (H + 2 * K) * D) + (h * D + i) * laststride(x);
                const int64_t v = u + (D / 2) * laststride(x);
                const float a0 = ldt(x, u), b0 = ldt(x, v);
                stt(x, u, narrow1(x->dtype, a0 * c) - narrow1(x->dtype, b0 * s));
                stt(x, v, narrow1(x->dtype, b0 * c) + narrow1(x->dtype, a0 * s));
            }
        }
    }
    return RAD_OK;
}

/* The eager BF16 attention expression materializes QK, scaled logits and probabilities.
 * Keep it separate from attn_dense's existing fused arithmetic. Only whole, noncausal segments
 * are served. Stack storage is bounded by the model's 32768-token limit. */
AVX_KERNEL(embed_attn) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *cu = &a->t[3], *out = &a->t[4];
    const int64_t M = q->rank == 2 ? q->shape[0] : 0;
    const int64_t D = p_int(a, "head_dim", 0), G = p_int(a, "gqa", 1);
    if (M <= 0 || M > p_int(a, "M", 0) || D <= 0 || D > 2048 || G <= 0 || q->shape[1] <= 0 ||
        q->shape[1] % D || k->rank != 2 || v->rank != 2 || out->rank != 2 || cu->rank != 1 ||
        numel(cu) < 2 || k->shape[0] != M || v->shape[0] != M || out->shape[0] != M)
        return RAD_E_SHAPE;
    const int64_t H = q->shape[1] / D, S = numel(cu) - 1;
    if (H % G || k->shape[1] != H / G * D || v->shape[1] != H / G * D || out->shape[1] != H * D)
        return RAD_E_SHAPE;
    if (p_int(a, "causal", 0) != 0) return RAD_E_UNSUPPORTED;
    if ((q->dtype != RAD_BF16 && q->dtype != RAD_F32) || k->dtype != q->dtype ||
        v->dtype != q->dtype || out->dtype != q->dtype || cu->dtype != RAD_I32)
        return RAD_E_DTYPE;
    auto boundary = [&](int64_t s) { return ldi(cu, offlin(cu, s)); };
    if (boundary(0) != 0 || boundary(S) != M) return RAD_E_INVAL;
    for (int64_t s = 0; s < S; ++s)
        if (boundary(s) >= boundary(s + 1) || boundary(s + 1) - boundary(s) > 32768)
            return RAD_E_INVAL;
    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)D));
    if (!std::isfinite(scale) || !(scale > 0)) return RAD_E_INVAL;
    AVX_PARALLEL_FOR_IF(M * H * D >= AVX_PAR_MIN)
    for (int64_t sh = 0; sh < S * H; ++sh) {
        const int64_t s = sh / H, h = sh % H, kh = h / G;
        const int64_t begin = boundary(s), end = boundary(s + 1);
        float scores[32768], accum[2048];
        for (int64_t i = begin; i < end; ++i) {
            const int64_t qb = rowoff(q, i, H * D) + h * D * laststride(q);
            float maximum = -INFINITY;
            for (int64_t j = begin; j < end; ++j) {
                const int64_t kb = rowoff(k, j, H / G * D) + kh * D * laststride(k);
                float dot = 0;
                for (int64_t d = 0; d < D; ++d)
                    dot += ldt(q, qb + d * laststride(q)) * ldt(k, kb + d * laststride(k));
                const float score = narrow1(q->dtype, narrow1(q->dtype, dot) * scale);
                scores[j - begin] = score;
                maximum = std::max(maximum, score);
            }
            float sum = 0;
            for (int64_t j = begin; j < end; ++j) {
                scores[j - begin] = std::exp(scores[j - begin] - maximum);
                sum += scores[j - begin];
            }
            std::fill_n(accum, D, 0.0f);
            for (int64_t j = begin; j < end; ++j) {
                const float probability = narrow1(q->dtype, scores[j - begin] / sum);
                const int64_t vb = rowoff(v, j, H / G * D) + kh * D * laststride(v);
                for (int64_t d = 0; d < D; ++d)
                    accum[d] += probability * ldt(v, vb + d * laststride(v));
            }
            const int64_t ob = rowoff(out, i, H * D) + h * D * laststride(out);
            for (int64_t d = 0; d < D; ++d)
                stt(out, ob + d * laststride(out), accum[d]);
        }
    }
    return RAD_OK;
}

/* Native masked mean: the sum is stored in BF16 before dividing, then stored in BF16 again.
 * Keep this separate from seg_mean's one-rounding sentence-embedding contract. */
AVX_KERNEL(seg_mean_bf16_sum) {
    if (!rad_args_have(a, 3))
        return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *cu = &a->t[1], *y = &a->t[2];
    if (x->dtype != RAD_BF16 || y->dtype != RAD_BF16 || cu->dtype != RAD_I32)
        return RAD_E_DTYPE;
    if (!x->data || !cu->data || !y->data)
        return RAD_E_INVAL;
    const int64_t M = x->rank == 2 ? x->shape[0] : 0, N = p_int(a, "n", 0);
    const int64_t S = cu->rank == 1 && cu->shape[0] > 1 ? cu->shape[0] - 1 : 0;
    const auto matrix = [](const RadTensor& t, int64_t rows, int64_t cols) {
        constexpr int64_t limit = INT64_MAX / 2;
        if (t.rank != 2 || t.shape[0] != rows || t.shape[1] != cols || t.stride[0] <= 0 ||
            t.stride[1] <= 0 || cols - 1 > limit / t.stride[1])
            return false;
        const int64_t columns = (cols - 1) * t.stride[1];
        return t.stride[0] > columns && rows - 1 <= (limit - columns) / t.stride[0];
    };
    if (M <= 0 || M > INT32_MAX || M > p_int(a, "M", 0) || N <= 0 || S <= 0 || S > M ||
        cu->stride[0] <= 0 || cu->stride[0] > (INT64_MAX / 4) / S || !matrix(*x, M, N) ||
        !matrix(*y, S, N))
        return RAD_E_SHAPE;
    const auto at = [&](int64_t s) { return ldi(cu, s * cu->stride[0]); };
    if (at(0) != 0 || at(S) != M)
        return RAD_E_INVAL;
    for (int64_t s = 0; s < S; ++s)
        if (at(s) < 0 || at(s + 1) <= at(s) || at(s + 1) > M)
            return RAD_E_INVAL;
    for (int64_t s = 0; s < S; ++s)
        for (int64_t c = 0; c < N; ++c) {
            float sum = 0;
            for (int64_t r = at(s); r < at(s + 1); ++r)
                sum += ldt(x, r * x->stride[0] + c * x->stride[1]);
            stt(y, s * y->stride[0] + c * y->stride[1],
                narrow1(RAD_BF16, sum) / (float)(at(s + 1) - at(s)));
        }
    return RAD_OK;
}
