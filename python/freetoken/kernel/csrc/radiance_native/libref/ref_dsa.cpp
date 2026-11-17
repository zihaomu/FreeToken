/* GLM's compressed index keys: channel-wise learned pooling, then a normalized
 * Hadamard and power-of-two E4M3 quantization. Pooling the raw keys by a mean,
 * or omitting either BF16 boundary, changes sparse selection. */
#include "ref_common.h"
#include "ref_ops.h"
#include <algorithm>
#include <cmath>
using namespace ref;

int ref_dsa_query(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const auto &x = a->t[0], &q = a->t[1], &s = a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n < 1 || n > 256 || (n & (n - 1)) || x.rank != 2 || q.rank != 2 || s.rank != 2 ||
        x.shape[0] < 1 || q.shape[0] != x.shape[0] || x.shape[1] != n || q.shape[1] != n ||
        s.shape[0] != x.shape[0] || s.shape[1] != 1)
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || q.dtype != RAD_F8E4M3 || s.dtype != RAD_F32) return RAD_E_DTYPE;
    if (!x.data || !q.data || !s.data) return RAD_E_INVAL;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < x.shape[0]; ++r) {
        float v[256];
        for (int64_t d = 0; d < n; ++d) v[d] = ldt(&x, r * x.stride[0] + d * x.stride[1]);
        for (int64_t h = 1; h < n; h *= 2)
            for (int64_t i = 0; i < n; i += 2 * h)
                for (int64_t j = 0; j < h; ++j) {
                    const float lo = v[i + j], hi = v[i + j + h];
                    v[i + j] = lo + hi;
                    v[i + j + h] = lo - hi;
                }
        float am = 1e-4f;
        const float norm = 1.0f / std::sqrt((float)n);
        for (int64_t d = 0; d < n; ++d) {
            v[d] = bf16_to_f32(f32_to_bf16(v[d] * norm));
            am = std::max(am, std::fabs(v[d]));
        }
        int exponent = 0;
        const float fraction = std::frexp(am * (1.0f / 448.0f), &exponent);
        const float sc = std::ldexp(1.0f, exponent - (fraction == 0.5f));
        stt(&s, r * s.stride[0], sc);
        for (int64_t d = 0; d < n; ++d)
            stt(&q, r * q.stride[0] + d * q.stride[1], std::clamp(v[d] / sc, -448.0f, 448.0f));
    }
    return RAD_OK;
}

int ref_dsa_score(const RadArgs* a, RadStream) {
    if (!a || !a->t || a->n_t < 9) return RAD_E_INVAL;
    RadTensor q = a->t[0], qs = a->t[1];
    const auto& gate = a->t[2];
    RadTensor k = a->t[3], ks = a->t[4];
    const bool packed = k.dtype == RAD_U8;
    if (packed) {
        if (ks.data || k.rank != 4 || k.shape[1] != 1 || k.shape[2] != 4 || k.shape[3] != 34 ||
            k.stride[3] != 1 || k.stride[2] != 34 || k.stride[0] < 136 || k.stride[0] % 4 ||
            ((uintptr_t)k.data & 3))
            return RAD_E_SHAPE;
        ks = {};
        ks.data = k.data ? (uint8_t*)k.data + 128 : nullptr;
        ks.dtype = RAD_F32;
        ks.rank = 1;
        ks.shape[0] = k.shape[0];
        ks.stride[0] = k.stride[0] / 4;
        k.dtype = RAD_F8E4M3;
        k.rank = 2;
        k.shape[1] = 128;
        k.stride[1] = 1;
    }
    const auto &bt = a->t[5], &cu = a->t[6], &nc = a->t[7], &out = a->t[8];
    const RadTensor* cu_part = rad_arg_in(a, 9);
    const int64_t n = p_int(a, "n", 0), h = p_int(a, "heads", 0);
    const int64_t part = p_int(a, "row_part", -1);
    if (h > 0 && q.rank == 2 && qs.rank == 2 && qs.shape[1] == 1 && gate.rank == 2 &&
        q.shape[0] == gate.shape[0] * h && qs.shape[0] == q.shape[0]) {
        q.rank = 3;
        q.shape[0] = gate.shape[0];
        q.shape[2] = q.shape[1];
        q.shape[1] = h;
        q.stride[2] = q.stride[1];
        q.stride[1] = q.stride[0];
        q.stride[0] *= h;
        qs.shape[0] = gate.shape[0];
        qs.shape[1] = h;
        qs.stride[1] = qs.stride[0];
        qs.stride[0] *= h;
    }
    if (n < 1 || h < 1 || h > 64 || q.rank != 3 || q.shape[0] < 1 || q.shape[1] != h ||
        q.shape[2] != n || qs.rank != 2 || gate.rank != 2 || qs.shape[0] != q.shape[0] ||
        gate.shape[0] != q.shape[0] || qs.shape[1] != h || gate.shape[1] != h || k.rank != 2 ||
        k.shape[1] != n || ks.rank != 1 || ks.shape[0] != k.shape[0] || bt.rank != 2 ||
        cu.rank != 1 || nc.rank != 1 || cu.shape[0] != bt.shape[0] + 1 ||
        nc.shape[0] != bt.shape[0] || out.rank != 2 ||
        (part < 0 ? out.shape[0] != q.shape[0]
                  : out.shape[0] < 1 || q.shape[0] < (part + 1) * out.shape[0]) ||
        out.shape[1] > bt.shape[1])
        return RAD_E_SHAPE;
    if (cu_part && (part < 0 || cu_part->rank != 1 || cu_part->shape[0] != cu.shape[0]))
        return part < 0 ? RAD_E_INVAL : RAD_E_SHAPE;
    if (cu_part && cu_part->dtype != RAD_I32) return RAD_E_DTYPE;
    if (q.dtype != RAD_F8E4M3 || k.dtype != RAD_F8E4M3 || qs.dtype != RAD_F32 ||
        ks.dtype != RAD_F32 || gate.dtype != RAD_F32 || bt.dtype != RAD_I32 ||
        cu.dtype != RAD_I32 || nc.dtype != RAD_I32 || out.dtype != RAD_F32)
        return RAD_E_DTYPE;
    for (int i = 0; i < 9; ++i)
        if (!(packed && i == 4) && !a->t[i].data) return RAD_E_INVAL;
    const float norm = 1.0f / std::sqrt((float)(h * n));
    /* A part scores out's rows from step row part * rows; the query operands span the step. */
    const int64_t row0 = part > 0 ? part * out.shape[0] : 0;
    if (cu_part)
        for (int64_t s = 0; s < cu.shape[0]; ++s)
            st_int(RAD_I32, cu_part->data, s * cu_part->stride[0],
                ld_int(cu.dtype, cu.data, s * cu.stride[0]) - row0);
#pragma omp parallel for schedule(static)
    for (int64_t o = 0; o < out.shape[0]; ++o) {
        const int64_t r = row0 + o;
        int64_t seq = -1;
        for (int64_t s = 0; s < bt.shape[0]; ++s)
            if (r >= ld_int(cu.dtype, cu.data, s * cu.stride[0]) &&
                r < ld_int(cu.dtype, cu.data, (s + 1) * cu.stride[0])) {
                seq = s;
                break;
            }
        const int64_t live = seq >= 0 ? ld_int(nc.dtype, nc.data, seq * nc.stride[0]) : 0;
        for (int64_t b = 0; b < out.shape[1]; ++b) {
            const int64_t page =
                seq >= 0 && b < live
                    ? ld_int(bt.dtype, bt.data, seq * bt.stride[0] + b * bt.stride[1])
                    : -1;
            float result = -INFINITY;
            if (page >= 0 && page < k.shape[0]) {
                result = 0;
                for (int64_t head = 0; head < h; ++head) {
                    float dot = 0;
                    for (int64_t d = 0; d < n; ++d)
                        dot += ldt(&q, r * q.stride[0] + head * q.stride[1] + d * q.stride[2]) *
                               ldt(&k, page * k.stride[0] + d * k.stride[1]);
                    result += std::max(dot, 0.0f) *
                              ldt(&qs, r * qs.stride[0] + head * qs.stride[1]) *
                              ldt(&gate, r * gate.stride[0] + head * gate.stride[1]);
                }
                result *= ldt(&ks, page * ks.stride[0]) * norm;
            }
            stt(&out, o * out.stride[0] + b * out.stride[1], result);
        }
    }
    return RAD_OK;
}

int ref_dsa_pool(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const auto &kg = a->t[0], &ape = a->t[1], &q = a->t[2], &s = a->t[3];
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0);
    if (p_int(a, "block_size", 4) != 4) return RAD_E_SHAPE;
    const auto *page = rad_arg_in(a, 4), *cache = rad_arg_in(a, 5);
    if ((page != nullptr) != (cache != nullptr)) return RAD_E_INVAL;
    if (cache &&
        (n != 128 || ratio != 4 || page->dtype != RAD_I32 || page->rank != 1 ||
         page->shape[0] != q.shape[0] || page->stride[0] < 1 || cache->dtype != RAD_U8 ||
         cache->rank != 4 || cache->shape[0] < 1 || cache->shape[1] != 1 || cache->shape[2] != 4 ||
         cache->shape[3] != 34 || cache->stride[3] != 1 || cache->stride[2] != 34 ||
         cache->stride[0] < 136 || cache->stride[0] % 4 || ((uintptr_t)cache->data & 3)))
        return RAD_E_SHAPE;
    if (n < 1 || n > 256 || (n & (n - 1)) || ratio < 2 || ratio > 16 || kg.rank != 2 ||
        ape.rank != 2 || q.rank != 2 || s.rank != 2 || q.shape[0] < 1 || !kg.data || !ape.data ||
        !q.data || !s.data || kg.dtype != RAD_BF16 || ape.dtype != RAD_F32 ||
        q.dtype != RAD_F8E4M3 || s.dtype != RAD_F32 || kg.shape[0] != q.shape[0] * ratio ||
        kg.shape[1] != 2 * n || ape.shape[0] != ratio || ape.shape[1] != n || q.shape[1] != n ||
        s.shape[0] != q.shape[0] || s.shape[1] != 1)
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < q.shape[0]; ++r) {
        float v[256];
        for (int64_t d = 0; d < n; ++d) {
            float gate[16], mx = -INFINITY;
            for (int64_t j = 0; j < ratio; ++j) {
                gate[j] = ldt(&kg, rowoff(&kg, r * ratio + j, 2 * n) + (n + d) * laststride(&kg)) +
                          ldt(&ape, rowoff(&ape, j, n) + d * laststride(&ape));
                mx = std::max(mx, gate[j]);
            }
            float num = 0, den = 0;
            for (int64_t j = 0; j < ratio; ++j) {
                const float p = std::exp(gate[j] - mx);
                num += p * ldt(&kg, rowoff(&kg, r * ratio + j, 2 * n) + d * laststride(&kg));
                den += p;
            }
            v[d] = bf16_to_f32(f32_to_bf16(num / den));
        }
        for (int64_t h = 1; h < n; h *= 2)
            for (int64_t i = 0; i < n; i += 2 * h)
                for (int64_t j = 0; j < h; ++j) {
                    const float x = v[i + j], y = v[i + j + h];
                    v[i + j] = x + y;
                    v[i + j + h] = x - y;
                }
        float am = 1e-4f;
        const float norm = 1.0f / std::sqrt((float)n);
        for (int64_t d = 0; d < n; ++d) {
            v[d] = bf16_to_f32(f32_to_bf16(v[d] * norm));
            am = std::max(am, std::fabs(v[d]));
        }
        const float target = am * (1.0f / 448.0f);
        int exp = 0;
        const float frac = std::frexp(target, &exp);
        const float scale = std::ldexp(1.0f, exp - (frac == 0.5f));
        stt(&s, rowoff(&s, r, 1), scale);
        for (int64_t d = 0; d < n; ++d)
            stt(&q, rowoff(&q, r, n) + d * laststride(&q),
                std::clamp(v[d] / scale, -448.0f, 448.0f));
        if (cache) {
            const int64_t p = ld_int(page->dtype, page->data, r * page->stride[0]);
            if (p >= 0 && p < cache->shape[0]) {
                auto* dst = (uint8_t*)cache->data + p * cache->stride[0];
                for (int d = 0; d < 128; ++d)
                    dst[d] = ((uint8_t*)q.data)[r * q.stride[0] + d * q.stride[1]];
                std::memcpy(dst + 128, &scale, sizeof(scale));
            }
        }
    }
    return RAD_OK;
}
