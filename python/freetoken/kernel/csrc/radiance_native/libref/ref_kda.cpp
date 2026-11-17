/* Bounded per-channel Kimi Delta Attention. Cache rows are complete candidate states,
 * so the accepted-prefix index restores the recurrence without trying to invert a delta.
 * Prefill commits only its final state; verify preserves every candidate for rejection. */
#include "ref_common.h"
#include "ref_ops.h"
#include <cmath>
#include <cstring>

using namespace ref;

namespace {
bool headed(const RadTensor& t, int64_t tokens, int64_t heads, int64_t dim) {
    return t.data && t.shape[0] == tokens &&
           ((t.rank == 3 && t.shape[1] == heads && t.shape[2] == dim) ||
            (t.rank == 2 && t.shape[1] == heads * dim));
}
int64_t off(const RadTensor& t, int64_t token, int64_t head, int64_t d, int64_t width) {
    return token * t.stride[0] + head * (t.rank == 3 ? t.stride[1] : width * t.stride[1]) +
           d * laststride(&t);
}
}

int ref_kda_update(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 13) return RAD_E_INVAL;
    for (int i = 0; i < 10; ++i)
        if (!rad_arg_in(a, i)) return RAD_E_INVAL;
    if (!rad_arg_in(a, 12)) return RAD_E_INVAL;
    const int64_t H = p_int(a, "heads", 0), K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const float lower = p_f32(a, "lower_bound", 0), eps = p_f32(a, "l2_eps", 1e-6f);
    const float scale = p_f32(a, "scale", 1.0f);
    const char* mode = p_str(a, "mode", "");
    const bool verify = !std::strcmp(mode, "verify");
    const char* form = p_str(a, "state_form", "per_candidate");
    const bool anchor = !std::strcmp(form, "anchor");
    if (!anchor && std::strcmp(form, "per_candidate")) return RAD_E_INVAL;
    if (anchor && (K != 128 || V != 128)) return RAD_E_UNSUPPORTED;
    if (!verify && std::strcmp(mode, "prefill")) return RAD_E_INVAL;
    if (H <= 0 || K <= 0 || K > 128 || V <= 0 || V > 128 || !std::isfinite(lower) || lower >= 0 ||
        !std::isfinite(eps) || eps <= 0 || !std::isfinite(scale))
        return RAD_E_SHAPE;
    const auto &q = a->t[0], &k = a->t[1], &v = a->t[2], &g = a->t[3], &beta = a->t[4];
    const auto &alog = a->t[5], &bias = a->t[6], &state = a->t[7], &idx = a->t[8], &cu = a->t[9];
    const RadTensor *accepted = rad_arg_in(a, 10), *init = rad_arg_in(a, 11);
    const auto& out = a->t[12];
    if (q.rank < 2 || cu.rank != 1 || cu.dtype != RAD_I32 || cu.shape[0] < 2) return RAD_E_SHAPE;
    const int64_t T = q.shape[0], N = cu.shape[0] - 1;
    if (T < 1) return RAD_E_SHAPE;
    if (!headed(q, T, H, K) || !headed(k, T, H, K) || !headed(v, T, H, V) || !headed(g, T, H, K) ||
        !headed(beta, T, H, 1) || !headed(out, T, H, V) || numel(&alog) != H ||
        numel(&bias) != H * K || state.rank != 4 || state.dtype != RAD_F32 || state.shape[1] != H ||
        state.shape[2] != V || state.shape[3] != K || idx.dtype != RAD_I32 || idx.rank != 2 ||
        idx.shape[0] < N || idx.shape[1] < 1 ||
        (accepted && (accepted->dtype != RAD_I32 || numel(accepted) < N)) ||
        (init && (init->dtype != RAD_I32 || numel(init) < N)))
        return RAD_E_SHAPE;
    const int64_t width = idx.shape[1], slots = state.shape[0];
    /* Check every address before writing any state, so a malformed batch cannot leave a
     * partially committed prefix when the host reference reports an error. */
    for (int64_t seq = 0; seq < N; ++seq) {
        const int64_t begin = ld_int(cu.dtype, cu.data, seq * cu.stride[0]);
        const int64_t end = ld_int(cu.dtype, cu.data, (seq + 1) * cu.stride[0]);
        const int64_t na =
            accepted ? ld_int(accepted->dtype, accepted->data, offlin(accepted, seq)) : 1;
        if (begin < 0 || end < begin || end > T || na < 1 || na > (anchor ? 32 : width) ||
            (verify && end - begin > (anchor ? 32 : width)) || (anchor && width < 2))
            return RAD_E_SHAPE;
        for (int64_t j = 0; j < width; ++j)
            if (ld_int(idx.dtype, idx.data, seq * idx.stride[0] + j * idx.stride[1]) >= slots)
                return RAD_E_SHAPE;
        if (anchor) {
            const int64_t read = ld_int(idx.dtype, idx.data, seq * idx.stride[0]);
            const int64_t scratch =
                ld_int(idx.dtype, idx.data, seq * idx.stride[0] + (width / 2) * idx.stride[1]);
            if (read >= 0 && (scratch < 0 || scratch == read)) return RAD_E_SHAPE;
            if (verify && read >= 0 &&
                (!init || ld_int(init->dtype, init->data, offlin(init, seq))))
                for (int64_t h = 0; h < H; ++h) {
                    const float count =
                        ldt(&state, scratch * state.stride[0] + h * state.stride[1]);
                    if (!std::isfinite(count) || count < 0 || count > 32 ||
                        count != std::floor(count) || (count > 0 && na > count))
                        return RAD_E_SHAPE;
                }
        }
    }
#pragma omp parallel for schedule(static)
    for (int64_t sh = 0; sh < N * H; ++sh) {
        const int64_t seq = sh / H, head = sh % H;
        const int64_t begin = ld_int(cu.dtype, cu.data, seq * cu.stride[0]);
        const int64_t end = ld_int(cu.dtype, cu.data, (seq + 1) * cu.stride[0]);
        if (begin == end) continue;
        const int64_t na =
            accepted ? ld_int(accepted->dtype, accepted->data, offlin(accepted, seq)) : 1;
        const int64_t read =
            ld_int(idx.dtype, idx.data,
                   seq * idx.stride[0] + (verify && !anchor ? na - 1 : 0) * idx.stride[1]);
        if (read < 0) {
            for (int64_t t = begin; t < end; ++t)
                for (int64_t e = 0; e < V; ++e) stt(&out, off(out, t, head, e, V), 0);
            continue;
        }
        float S[128 * 128], qn[128], kn[128], decay[128];
        const bool initialized = !init || ld_int(init->dtype, init->data, offlin(init, seq)) != 0;
        const int64_t sb = read * state.stride[0] + head * state.stride[1];
        for (int64_t e = 0; e < V; ++e)
            for (int64_t d = 0; d < K; ++d)
                S[e * K + d] =
                    initialized ? ldt(&state, sb + e * state.stride[2] + d * state.stride[3]) : 0;
        const int64_t scratch =
            anchor ? ld_int(idx.dtype, idx.data, seq * idx.stride[0] + (width / 2) * idx.stride[1])
                   : 0;
        auto factor_at = [&](int64_t i) {
            return scratch * state.stride[0] + head * state.stride[1] + (i / K) * state.stride[2] +
                   (i % K) * state.stride[3];
        };
        constexpr int factors = 385;
        if (anchor && verify) {
            const int replay = initialized ? (int)ldt(&state, factor_at(0)) : 0;
            for (int t = 0; t < (replay ? na : 0); ++t) {
                const int f = 1 + t * factors;
                for (int d = 0; d < 128; ++d) {
                    kn[d] = ldt(&state, factor_at(f + d));
                    decay[d] = ldt(&state, factor_at(f + 256 + d));
                }
                const float bt = ldt(&state, factor_at(f + 384));
                for (int e = 0; e < 128; ++e) {
                    float prediction = 0;
                    for (int d = 0; d < 128; ++d) {
                        S[e * K + d] *= decay[d];
                        prediction += S[e * K + d] * kn[d];
                    }
                    const float delta = bt * (ldt(&state, factor_at(f + 128 + e)) - prediction);
                    for (int d = 0; d < 128; ++d) S[e * K + d] += delta * kn[d];
                }
            }
            // Publish only the accepted prefix. Current candidates remain in registers.
            for (int e = 0; e < 128; ++e)
                for (int d = 0; d < 128; ++d)
                    stt(&state, sb + e * state.stride[2] + d * state.stride[3], S[e * K + d]);
        }
        const float gain = std::exp(ldt(&alog, head * laststride(&alog)));
        for (int64_t t = begin; t < end; ++t) {
            float qs = 0, ks = 0;
            for (int64_t d = 0; d < K; ++d) {
                qn[d] = ldt(&q, off(q, t, head, d, K));
                kn[d] = ldt(&k, off(k, t, head, d, K));
                qs += qn[d] * qn[d];
                ks += kn[d] * kn[d];
            }
            const float qi = scale / std::sqrt(qs + eps), ki = 1.0f / std::sqrt(ks + eps);
            for (int64_t d = 0; d < K; ++d) {
                qn[d] *= qi;
                kn[d] *= ki;
                const float z = gain * (ldt(&g, off(g, t, head, d, K)) +
                                        ldt(&bias, (head * K + d) * laststride(&bias)));
                decay[d] = std::exp(lower / (1.0f + std::exp(-z)));
            }
            const float bt = 1.0f / (1.0f + std::exp(-ldt(&beta, off(beta, t, head, 0, 1))));
            if (anchor && verify) {
                const int f = 1 + (int)(t - begin) * factors;
                for (int d = 0; d < 128; ++d) {
                    stt(&state, factor_at(f + d), kn[d]);
                    stt(&state, factor_at(f + 256 + d), decay[d]);
                }
                for (int e = 0; e < 128; ++e)
                    stt(&state, factor_at(f + 128 + e), ldt(&v, off(v, t, head, e, V)));
                stt(&state, factor_at(f + 384), bt);
            }
            for (int64_t e = 0; e < V; ++e) {
                float predicted = 0;
                for (int64_t d = 0; d < K; ++d) {
                    S[e * K + d] *= decay[d];
                    predicted += S[e * K + d] * kn[d];
                }
                const float delta = bt * (ldt(&v, off(v, t, head, e, V)) - predicted);
                float result = 0;
                for (int64_t d = 0; d < K; ++d) {
                    S[e * K + d] += delta * kn[d];
                    result += S[e * K + d] * qn[d];
                }
                stt(&out, off(out, t, head, e, V), result);
            }
            if ((!anchor && verify) || (!verify && t + 1 == end)) {
                const int64_t write =
                    ld_int(idx.dtype, idx.data,
                           seq * idx.stride[0] + (verify ? t - begin : 0) * idx.stride[1]);
                if (write >= 0)
                    for (int64_t e = 0; e < V; ++e)
                        for (int64_t d = 0; d < K; ++d)
                            stt(&state,
                                write * state.stride[0] + head * state.stride[1] +
                                    e * state.stride[2] + d * state.stride[3],
                                S[e * K + d]);
            }
        }
        if (anchor) stt(&state, factor_at(0), verify ? (float)(end - begin) : 0.0f);
    }
    return RAD_OK;
}
