#include "avx_vec.h"
#include "avx_common.h"
#include <algorithm>
#include <cmath>
using namespace avx;

AVX_KERNEL(mla_project) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    RadTensor x = a->t[0], y = a->t[2];
    const auto& w = a->t[1];
    const int64_t h = p_int(a, "heads", 0), d = p_int(a, "head_dim", 0),
                  l = p_int(a, "latent_dim", 0);
    const char* mode = p_str(a, "mode", "");
    const bool absorb = !std::strcmp(mode, "absorb"), expand = !std::strcmp(mode, "expand");
    auto heads_view = [&](RadTensor& t, int64_t width) {
        if (h > 0 && width > 0 && t.rank == 2 && t.shape[1] == h * width) {
            t.rank = 3;
            t.shape[1] = h;
            t.shape[2] = width;
            t.stride[2] = t.stride[1];
            t.stride[1] *= width;
        }
    };
    heads_view(x, absorb ? d : l);
    heads_view(y, absorb ? l : d);
    if ((!absorb && !expand) || h < 1 || d < 1 || l < 1 || x.rank != 3 || w.rank != 2 ||
        y.rank != 3 || x.shape[0] < 1 || x.shape[1] != h || x.shape[2] != (absorb ? d : l) ||
        w.shape[0] != h * 2 * d || w.shape[1] != l || y.shape[0] != x.shape[0] || y.shape[1] != h ||
        y.shape[2] != (absorb ? l : d))
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || w.dtype != RAD_BF16 || y.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (!x.data || !w.data || !y.data) return RAD_E_INVAL;
    AVX_PARALLEL_FOR_IF(x.shape[0] * h * d * l >= AVX_PAR_MIN)
    for (int64_t job = 0; job < x.shape[0] * h; ++job) {
        const int64_t r = job / h, head = job % h;
        float* tmp = scratch_f32((size_t)(2 * (d + l)));
        float *query = tmp, *weights = tmp + d + l;
        for (int64_t j = 0; j < (absorb ? d : l); ++j)
            query[j] = ldt(&x, r * x.stride[0] + head * x.stride[1] + j * x.stride[2]);
        for (int64_t col = 0; col < (absorb ? l : d); ++col) {
            const int64_t nk = absorb ? d : l;
            for (int64_t j = 0; j < nk; ++j) {
                const int64_t wr = head * 2 * d + (absorb ? j : d + col), wc = absorb ? col : j;
                weights[j] = ldt(&w, wr * w.stride[0] + wc * w.stride[1]);
            }
            vf acc = vf_zero();
            int64_t j = 0;
            for (; j + VF_N <= nk; j += VF_N)
                acc = vf_fma(vf_loadu(query + j), vf_loadu(weights + j), acc);
            float sum = vf_hsum(acc);
            for (; j < nk; ++j) sum += query[j] * weights[j];
            stt(&y, r * y.stride[0] + head * y.stride[1] + col * y.stride[2], sum);
        }
    }
    return RAD_OK;
}

/* libref's mla_attn_paged (libref/ref_mla.cpp): dense causal latent attention over a cache row of
 * the latent then the rotated key part, one sequence's page table, an optional head gate. */
AVX_KERNEL(mla_attn_paged) {
    if (a->n_t < 8 || !rad_arg_in(a, 0) || !rad_arg_in(a, 2) || !rad_arg_in(a, 3) ||
        !rad_arg_in(a, 4) || !rad_arg_in(a, 5) || !rad_arg_in(a, 7))
        return RAD_E_INVAL;
    RadTensor q = a->t[0], qr = a->t[1], o = a->t[7];
    const auto &kv = a->t[2], &bt = a->t[3], &su = a->t[4], &cu = a->t[5];
    const RadTensor* gate = rad_arg_in(a, 6);
    const int64_t h = p_int(a, "heads", 0), L = p_int(a, "latent_dim", 0),
                  R = p_int(a, "rope_dim", 0), bs = p_int(a, "block_size", 0);
    const float scale = p_f32(a, "scale", 0);
    auto heads_view = [&](RadTensor& t, int64_t width) {
        if (h > 0 && width > 0 && t.rank == 2 && t.shape[1] == h * width) {
            t.rank = 3;
            t.shape[1] = h;
            t.shape[2] = width;
            t.stride[2] = t.stride[1];
            t.stride[1] *= width;
        }
    };
    heads_view(q, L);
    heads_view(qr, R);
    heads_view(o, L);
    if (h < 1 || L < 1 || L > 2048 || R < 0 || R > 2048 || bs < 1 || !std::isfinite(scale) ||
        scale <= 0 || q.rank != 3 || q.shape[0] < 1 || q.shape[1] != h || q.shape[2] != L ||
        (R && (!qr.data || qr.rank != 3 || qr.shape[0] != q.shape[0] || qr.shape[1] != h ||
               qr.shape[2] != R)) ||
        kv.rank != 4 || kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] < L + R ||
        bt.rank != 2 || su.rank != 1 || cu.rank != 1 || numel(&cu) != numel(&su) + 1 ||
        o.rank != 3 || o.shape[0] != q.shape[0] || o.shape[1] != h || o.shape[2] != L ||
        (gate && (gate->rank != 2 || gate->shape[0] != q.shape[0] || gate->shape[1] < h)))
        return RAD_E_SHAPE;
    const int64_t n_seq = numel(&su), rows = q.shape[0];
    auto cu_at = [&](int64_t i) { return ld_int_dt(cu.dtype, cu.data, i * laststride(&cu)); };
    if (cu_at(0) != 0 || cu_at(n_seq) != rows) return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(rows * h * (L + R) * bt.shape[1] * bs >= AVX_PAR_MIN)
    for (int64_t job = 0; job < rows * h; ++job) {
        const int64_t r = job / h, head = job % h;
        int64_t s = 0;
        while (s + 1 < n_seq && cu_at(s + 1) <= r) ++s;
        const int64_t qlen = cu_at(s + 1) - cu_at(s), i = r - cu_at(s);
        const int64_t ctx = ld_int_dt(su.dtype, su.data, s * su.stride[0]);
        const int64_t live = std::min(ctx - qlen + i + 1, bt.shape[1] * bs);
        float *tmp = scratch_f32((size_t)(3 * L + 2 * R)), *query = tmp, *key = tmp + L,
              *value = tmp + 2 * L, *qrot = tmp + 3 * L, *krot = qrot + R;
        for (int64_t c = 0; c < L; ++c) {
            query[c] = ldt(&q, r * q.stride[0] + head * q.stride[1] + c * q.stride[2]);
            value[c] = 0;
        }
        for (int64_t c = 0; c < R; ++c)
            qrot[c] = ldt(&qr, r * qr.stride[0] + head * qr.stride[1] + c * qr.stride[2]);
        float max = -INFINITY, den = 0;
        for (int64_t j = 0; j < live; ++j) {
            const int64_t page =
                ld_int_dt(bt.dtype, bt.data, s * bt.stride[0] + (j / bs) * bt.stride[1]);
            if (page < 0 || page >= kv.shape[0]) continue;
            const int64_t base = page * kv.stride[0] + (j % bs) * kv.stride[2];
            for (int64_t c = 0; c < L; ++c) key[c] = ldt(&kv, base + c * kv.stride[3]);
            for (int64_t c = 0; c < R; ++c) krot[c] = ldt(&kv, base + (L + c) * kv.stride[3]);
            vf acc = vf_zero();
            int64_t c = 0;
            for (; c + VF_N <= L; c += VF_N)
                acc = vf_fma(vf_loadu(query + c), vf_loadu(key + c), acc);
            float score = vf_hsum(acc);
            for (; c < L; ++c) score += query[c] * key[c];
            for (c = 0; c < R; ++c) score += qrot[c] * krot[c];
            score *= scale;
            const float next = std::max(max, score), old = std::exp(max - next),
                        w = std::exp(score - next);
            den = den * old + w;
            max = next;
            const vf a = vf_set1(old), b = vf_set1(w);
            c = 0;
            for (; c + VF_N <= L; c += VF_N)
                vf_storeu(value + c, vf_fma(vf_loadu(key + c), b, vf_mul(vf_loadu(value + c), a)));
            for (; c < L; ++c) value[c] = value[c] * old + key[c] * w;
        }
        float g = 1.0f;
        if (gate) {
            const float x = ldt(gate, r * gate->stride[0] + head * gate->stride[1]);
            g = narrow1(RAD_BF16, 1.0f / (1.0f + std::exp(-x)));
        }
        for (int64_t c = 0; c < L; ++c) {
            float v = den > 0 ? value[c] / den : 0;
            if (gate) v = narrow1(RAD_BF16, v) * g;
            stt(&o, r * o.stride[0] + head * o.stride[1] + c * o.stride[2], v);
        }
    }
    return RAD_OK;
}

AVX_KERNEL(mla_attn) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const auto &q = a->t[0], &kv = a->t[1], &bt = a->t[2], &len = a->t[3], &out = a->t[4];
    const int64_t d = p_int(a, "latent_dim", 0), h = p_int(a, "heads", 0),
                  bs = p_int(a, "block_size", 0);
    const float scale = p_f32(a, "scale", 0);
    if (d < 1 || d > 2048 || h < 1 || bs < 1 || !std::isfinite(scale) || scale <= 0 ||
        q.rank != 3 || q.shape[0] < 1 || q.shape[1] != h || q.shape[2] != d || kv.rank != 4 ||
        kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] != d || bt.rank != 2 ||
        bt.shape[0] != q.shape[0] || len.rank != 1 || len.shape[0] != q.shape[0] || out.rank != 3 ||
        out.shape[0] != q.shape[0] || out.shape[1] != h || out.shape[2] != d ||
        bt.dtype != RAD_I32 || len.dtype != RAD_I32)
        return RAD_E_SHAPE;
    for (int i = 0; i < 5; ++i)
        if (!a->t[i].data) return RAD_E_INVAL;
    AVX_PARALLEL_FOR_IF(q.shape[0] * h * d * bt.shape[1] * bs >= AVX_PAR_MIN)
    for (int64_t job = 0; job < q.shape[0] * h; ++job) {
        const int64_t r = job / h, head = job % h;
        const int64_t live = std::clamp(ld_int_dt(len.dtype, len.data, r * len.stride[0]),
                                        (int64_t)0, bt.shape[1] * bs);
        float *tmp = scratch_f32((size_t)(3 * d)), *query = tmp, *key = tmp + d,
              *value = tmp + 2 * d;
        for (int64_t c = 0; c < d; ++c) {
            query[c] = ldt(&q, r * q.stride[0] + head * q.stride[1] + c * q.stride[2]);
            value[c] = 0;
        }
        float max = -INFINITY, den = 0;
        for (int64_t j = 0; j < live; ++j) {
            const int64_t page =
                ld_int_dt(bt.dtype, bt.data, r * bt.stride[0] + (j / bs) * bt.stride[1]);
            if (page < 0 || page >= kv.shape[0]) continue;
            const int64_t base = page * kv.stride[0] + (j % bs) * kv.stride[2];
            for (int64_t c = 0; c < d; ++c) key[c] = ldt(&kv, base + c * kv.stride[3]);
            vf acc = vf_zero();
            int64_t c = 0;
            for (; c + VF_N <= d; c += VF_N)
                acc = vf_fma(vf_loadu(query + c), vf_loadu(key + c), acc);
            float score = vf_hsum(acc);
            for (; c < d; ++c) score += query[c] * key[c];
            score *= scale;
            const float next = std::max(max, score), old = std::exp(max - next),
                        w = std::exp(score - next);
            den = den * old + w;
            max = next;
            const vf a = vf_set1(old), b = vf_set1(w);
            c = 0;
            for (; c + VF_N <= d; c += VF_N)
                vf_storeu(value + c, vf_fma(vf_loadu(key + c), b, vf_mul(vf_loadu(value + c), a)));
            for (; c < d; ++c) value[c] = value[c] * old + key[c] * w;
        }
        for (int64_t c = 0; c < d; ++c)
            stt(&out, r * out.stride[0] + head * out.stride[1] + c * out.stride[2],
                den > 0 ? value[c] / den : 0);
    }
    return RAD_OK;
}
