/* avx_ssd.cpp -- the Mamba-2 selective state update (ssd_update), libref/ref_ssd.cpp's arithmetic
 * with the d_state reduction in this ISA level's vectors, in both state forms.
 *
 * A token widens its group's B and C rows once; each head then decays and updates its
 * [head_dim, d_state] state row by row and reads the row against C in the same pass. The anchor
 * form's replay applies a record with the window's own update (row_axpy is row_update without the
 * read), so the committed state is this library's per-candidate state bit for bit. */
#include "avx_vec.h"
#include "avx_common.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace avx;

namespace {
bool headed(const RadTensor& t, int64_t tokens, int64_t heads, int64_t width) {
    return t.data && t.shape[0] == tokens &&
           ((t.rank == 3 && t.shape[1] == heads && t.shape[2] == width) ||
            (t.rank == 2 && t.shape[1] == heads * width));
}
int64_t off(const RadTensor& t, int64_t token, int64_t head, int64_t d, int64_t width) {
    return token * t.stride[0] + head * (t.rank == 3 ? t.stride[1] : width * t.stride[1]) +
           d * laststride(&t);
}
float softplus(float v) { return v > 20.0f ? v : std::log1p(std::exp(v)); }

/* s = s * decay + u * b over one state row, as row_update does it, without the read. */
void row_axpy(float* s, const float* b, float decay, float u, int64_t n) {
    const vf vd = vf_set1(decay), vu = vf_set1(u);
    int64_t d = 0;
    for (; d + VF_N <= n; d += VF_N)
        vf_storeu(s + d, vf_fma(vf_loadu(s + d), vd, vf_mul(vu, vf_loadu(b + d))));
    for (; d < n; ++d) s[d] = s[d] * decay + u * b[d];
}
int64_t anchor_tokens(int64_t P, int64_t Nst) {
    const int64_t fit = (P * Nst - 1) / (1 + P + Nst);
    return fit < 32 ? fit : 32;
}

/* s = s * decay + u * b over one state row, and the row's dot with c. */
float row_update(float* s, const float* b, const float* c, float decay, float u, int64_t n) {
    const vf vd = vf_set1(decay), vu = vf_set1(u);
    vf acc = vf_zero();
    int64_t d = 0;
    for (; d + VF_N <= n; d += VF_N) {
        const vf r = vf_fma(vf_loadu(s + d), vd, vf_mul(vu, vf_loadu(b + d)));
        vf_storeu(s + d, r);
        acc = vf_fma(r, vf_loadu(c + d), acc);
    }
    float result = vf_hsum(acc);
    for (; d < n; ++d) {
        s[d] = s[d] * decay + u * b[d];
        result += s[d] * c[d];
    }
    return result;
}
}  // namespace

AVX_KERNEL(ssd_update) {
    if (!a || a->n_t < 13) return RAD_E_INVAL;
    for (int i = 0; i < 10; ++i)
        if (!rad_arg_in(a, i)) return RAD_E_INVAL;
    if (!rad_arg_in(a, 12)) return RAD_E_INVAL;
    const int64_t H = p_int(a, "heads", 0), P = p_int(a, "head_dim", 0);
    const int64_t Nst = p_int(a, "d_state", 0), G = p_int(a, "groups", 0);
    const char* mode = p_str(a, "mode", "");
    const bool verify = !std::strcmp(mode, "verify");
    if (!verify && std::strcmp(mode, "prefill")) return RAD_E_INVAL;
    const char* form = p_str(a, "state_form", "per_candidate");
    const bool anchor = !std::strcmp(form, "anchor");
    if (!anchor && std::strcmp(form, "per_candidate")) return RAD_E_INVAL;
    if (H < 1 || P < 1 || Nst < 1 || G < 1 || H % G) return RAD_E_SHAPE;
    const int64_t rec = 1 + P + Nst, deepest = anchor_tokens(P, Nst);
    if (anchor && deepest < 1) return RAD_E_SHAPE;
    const auto &x = a->t[0], &B = a->t[1], &C = a->t[2], &dt = a->t[3];
    const auto &al = a->t[4], &dtb = a->t[5], &dskip = a->t[6], &s = a->t[7];
    const auto &idx = a->t[8], &cu = a->t[9], &out = a->t[12];
    const auto* accepted = rad_arg_in(a, 10);
    const auto* init = rad_arg_in(a, 11);
    if (x.rank < 2 || cu.rank != 1 || cu.dtype != RAD_I32 || cu.shape[0] < 2) return RAD_E_SHAPE;
    const int64_t T = x.shape[0], N = cu.shape[0] - 1, slots = s.shape[0];
    if (T < 1) return RAD_E_SHAPE;
    if (!headed(x, T, H, P) || !headed(B, T, G, Nst) || !headed(C, T, G, Nst) ||
        !headed(dt, T, H, 1) || !headed(out, T, H, P) || numel(&al) != H || numel(&dtb) != H ||
        numel(&dskip) != H || s.rank != 4 || s.dtype != RAD_F32 || s.shape[1] != H ||
        s.shape[2] != P || s.shape[3] != Nst || idx.rank != 2 || idx.dtype != RAD_I32 ||
        idx.shape[0] < N || idx.shape[1] < 1 ||
        (accepted && (accepted->dtype != RAD_I32 || numel(accepted) < N)) ||
        (init && (init->dtype != RAD_I32 || numel(init) < N)))
        return RAD_E_SHAPE;
    const int64_t width = idx.shape[1];
    for (int64_t seq = 0; seq < N; ++seq) {
        const int64_t begin = ldi(&cu, seq * cu.stride[0]);
        const int64_t end = ldi(&cu, (seq + 1) * cu.stride[0]);
        const int64_t na = accepted ? ldi(accepted, offlin(accepted, seq)) : 1;
        const int64_t most = anchor ? deepest : width;
        if (begin < 0 || end < begin || end > T || na < 1 || na > most ||
            (verify && end - begin > most) || (anchor && width < 2))
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
                    if (!std::isfinite(count) || count < 0 || count > (float)deepest ||
                        count != std::floor(count) || (count > 0 && na > count))
                        return RAD_E_SHAPE;
                }
        }
    }
    const int64_t per_group = H / G;
    AVX_PARALLEL_FOR_IF(N * H * P * Nst >= AVX_PAR_MIN)
    for (int64_t sh = 0; sh < N * H; ++sh) {
        const int64_t seq = sh / H, head = sh % H, grp = head / per_group;
        const int64_t begin = ldi(&cu, seq * cu.stride[0]);
        const int64_t end = ldi(&cu, (seq + 1) * cu.stride[0]);
        if (begin == end) continue;
        const int64_t na = accepted ? ldi(accepted, offlin(accepted, seq)) : 1;
        const int64_t read =
            ldi(&idx, seq * idx.stride[0] + (verify && !anchor ? na - 1 : 0) * idx.stride[1]);
        if (read < 0) {
            for (int64_t t = begin; t < end; ++t)
                for (int64_t e = 0; e < P; ++e) stt(&out, off(out, t, head, e, P), 0);
            continue;
        }
        const bool initialized = !init || ldi(init, offlin(init, seq)) != 0;
        std::vector<float> S((size_t)(P * Nst)), b((size_t)Nst), c((size_t)Nst);
        const int64_t sb = read * s.stride[0] + head * s.stride[1];
        for (int64_t e = 0; e < P; ++e)
            for (int64_t d = 0; d < Nst; ++d)
                S[e * Nst + d] = initialized ? ldt(&s, sb + e * s.stride[2] + d * s.stride[3]) : 0;
        const int64_t scratch =
            anchor ? ldi(&idx, seq * idx.stride[0] + (width / 2) * idx.stride[1]) : 0;
        auto factor_at = [&](int64_t i) {
            return scratch * s.stride[0] + head * s.stride[1] + (i / Nst) * s.stride[2] +
                   (i % Nst) * s.stride[3];
        };
        if (anchor && verify) {
            /* The accepted prefix of the previous window, onto the committed state, published. */
            const int64_t replay = initialized ? (int64_t)ldt(&s, factor_at(0)) : 0;
            for (int64_t t = 0; t < (replay ? na : 0); ++t) {
                const int64_t f = 1 + t * rec;
                const float decay = ldt(&s, factor_at(f));
                for (int64_t d = 0; d < Nst; ++d) b[(size_t)d] = ldt(&s, factor_at(f + 1 + P + d));
                for (int64_t e = 0; e < P; ++e)
                    row_axpy(&S[e * Nst], b.data(), decay, ldt(&s, factor_at(f + 1 + e)), Nst);
            }
            for (int64_t e = 0; e < P; ++e)
                for (int64_t d = 0; d < Nst; ++d)
                    stt(&s, sb + e * s.stride[2] + d * s.stride[3], S[e * Nst + d]);
        }
        const float A = -std::exp(ldt(&al, head * laststride(&al)));
        const float bias = ldt(&dtb, head * laststride(&dtb));
        const float skip = ldt(&dskip, head * laststride(&dskip));
        for (int64_t t = begin; t < end; ++t) {
            const float step = softplus(ldt(&dt, off(dt, t, head, 0, 1)) + bias);
            const float decay = std::exp(step * A);
            for (int64_t d = 0; d < Nst; ++d) {
                b[d] = ldt(&B, off(B, t, grp, d, Nst));
                c[d] = ldt(&C, off(C, t, grp, d, Nst));
            }
            const int64_t f = 1 + (t - begin) * rec;
            if (anchor && verify) {
                stt(&s, factor_at(f), decay);
                for (int64_t d = 0; d < Nst; ++d) stt(&s, factor_at(f + 1 + P + d), b[d]);
            }
            for (int64_t e = 0; e < P; ++e) {
                const float xe = ldt(&x, off(x, t, head, e, P));
                const float u = step * xe;
                if (anchor && verify) stt(&s, factor_at(f + 1 + e), u);
                const float r = row_update(&S[e * Nst], b.data(), c.data(), decay, u, Nst);
                stt(&out, off(out, t, head, e, P), r + skip * xe);
            }
            if ((verify && !anchor) || (!verify && t + 1 == end)) {
                const int64_t write =
                    ldi(&idx, seq * idx.stride[0] + (verify ? t - begin : 0) * idx.stride[1]);
                if (write >= 0)
                    for (int64_t e = 0; e < P; ++e)
                        for (int64_t d = 0; d < Nst; ++d)
                            stt(&s, write * s.stride[0] + head * s.stride[1] + e * s.stride[2] +
                                        d * s.stride[3],
                                S[e * Nst + d]);
            }
        }
        if (anchor) stt(&s, factor_at(0), verify ? (float)(end - begin) : 0.0f);
    }
    return RAD_OK;
}
