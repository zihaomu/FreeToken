/* ref_ssd.cpp -- the Mamba-2 selective state update (SSD), token by token.
 *
 * ============================== The algorithm ==============================
 *
 * Per token, per head h of `heads`, with the state S shaped [head_dim, d_state] and g = h / (heads /
 * groups) the head's B/C group:
 *
 *     dt'  = softplus(dt[h] + dt_bias[h])            A = -exp(A_log[h])
 *     S   *= e^{dt' A}
 *     S   += dt' . x[h] B[g]^T                        o[h] = S C[g] + D[h] . x[h]
 *
 * A Mamba-2 head is a gated delta net head without the delta: one scalar decay per head and token,
 * no beta, no l2 norm on B or C, and the skip term D. The gdn ops cannot run it (their rank-1 update
 * subtracts S k and their keys are normalised), so it is an op of its own.
 *
 * dt is not clamped. The checkpoint's `time_step_limit` is (0, inf), which is what mamba_ssm's
 * kernels and vLLM pass; transformers' CPU fallback clamps dt at `time_step_min` and is not what
 * the model was trained or is served with. softplus keeps torch's and mamba_ssm's threshold: above
 * 20 it is the identity.
 *
 * The output excludes the model's gated norm.
 *
 * ============================== The layouts ==============================
 *
 *     x, o   [T, heads, head_dim] or [T, heads*head_dim]
 *     B, C   [T, groups, d_state] or [T, groups*d_state]
 *     dt     [T, heads]                       A_log, dt_bias, D   [heads], f32
 *     state  [n_slots, heads, head_dim, d_state], f32             state_idx [N, W]   cu [N+1]
 *
 * The rollback contract is kda_update's, in both forms. `per_candidate`: `mode=verify` reads column
 * `num_accepted - 1` of the sequence's `state_idx` row and writes the state after token t of the
 * window to column t; `mode=prefill` reads column zero and writes only the final state there.
 * `has_init` 0 starts from zero, a negative read slot gives zero output and a negative write slot
 * is skipped.
 *
 * `anchor`: column zero is the committed state and column `width/2` a scratch slot of the same
 * shape. Per head the scratch holds a replay count and then one record per window token: the
 * decay e^{dt' A}, u = dt' x (head_dim) and B (d_state), 1 + head_dim + d_state floats, as many as
 * the [head_dim, d_state] block holds, at most 32 (a 64 x 128 head holds 42). `mode=verify` first
 * replays the previous window's accepted prefix (`num_accepted` records, at most the stored count)
 * onto the committed state, publishes it, then runs the window from it writing each token's record
 * and the window length as the new count. The rank-1 update is replayed with the same arithmetic,
 * so the committed state is the per-candidate form's bit for bit. `mode=prefill` commits its final
 * state and sets the count to zero; a zeroed scratch replays nothing.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace ref;

namespace {

/* Rows of `width` columns per (token, head), whether the tensor is 3-D or flattened to 2-D. */
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

/* The anchor scratch's records: per window token 1 + head_dim + d_state floats behind the count. */
int64_t anchor_tokens(int64_t P, int64_t Nst) {
    const int64_t fit = (P * Nst - 1) / (1 + P + Nst);
    return fit < 32 ? fit : 32;
}

}  // namespace

int ref_ssd_update(const RadArgs* a, RadStream) {
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
    if (H <= 0 || P <= 0 || Nst <= 0 || G <= 0 || H % G) return RAD_E_SHAPE;
    const int64_t rec = 1 + P + Nst, deepest = anchor_tokens(P, Nst);
    if (anchor && deepest < 1) return RAD_E_SHAPE;
    const auto &x = a->t[0], &B = a->t[1], &C = a->t[2], &dt = a->t[3];
    const auto &alog = a->t[4], &dtb = a->t[5], &dskip = a->t[6], &state = a->t[7];
    const auto &idx = a->t[8], &cu = a->t[9];
    const RadTensor *accepted = rad_arg_in(a, 10), *init = rad_arg_in(a, 11);
    const auto& out = a->t[12];
    if (x.rank < 2 || cu.rank != 1 || cu.dtype != RAD_I32 || cu.shape[0] < 2) return RAD_E_SHAPE;
    const int64_t T = x.shape[0], N = cu.shape[0] - 1;
    if (T < 1) return RAD_E_SHAPE;
    if (!headed(x, T, H, P) || !headed(B, T, G, Nst) || !headed(C, T, G, Nst) ||
        !headed(dt, T, H, 1) || !headed(out, T, H, P) || numel(&alog) != H || numel(&dtb) != H ||
        numel(&dskip) != H || state.rank != 4 || state.dtype != RAD_F32 || state.shape[1] != H ||
        state.shape[2] != P || state.shape[3] != Nst || idx.dtype != RAD_I32 || idx.rank != 2 ||
        idx.shape[0] < N || idx.shape[1] < 1 ||
        (accepted && (accepted->dtype != RAD_I32 || numel(accepted) < N)) ||
        (init && (init->dtype != RAD_I32 || numel(init) < N)))
        return RAD_E_SHAPE;
    const int64_t width = idx.shape[1], slots = state.shape[0];
    /* Every address is checked before any state is written, so a malformed batch cannot leave a
     * partly committed prefix behind an error. */
    for (int64_t seq = 0; seq < N; ++seq) {
        const int64_t begin = ld_int(cu.dtype, cu.data, seq * cu.stride[0]);
        const int64_t end = ld_int(cu.dtype, cu.data, (seq + 1) * cu.stride[0]);
        const int64_t na =
            accepted ? ld_int(accepted->dtype, accepted->data, offlin(accepted, seq)) : 1;
        const int64_t most = anchor ? deepest : width;
        if (begin < 0 || end < begin || end > T || na < 1 || na > most ||
            (verify && end - begin > most) || (anchor && width < 2))
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
                    if (!std::isfinite(count) || count < 0 || count > (float)deepest ||
                        count != std::floor(count) || (count > 0 && na > count))
                        return RAD_E_SHAPE;
                }
        }
    }
    const int64_t per_group = H / G;
#pragma omp parallel for schedule(static)
    for (int64_t sh = 0; sh < N * H; ++sh) {
        const int64_t seq = sh / H, head = sh % H, grp = head / per_group;
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
                for (int64_t e = 0; e < P; ++e) stt(&out, off(out, t, head, e, P), 0);
            continue;
        }
        const bool initialized = !init || ld_int(init->dtype, init->data, offlin(init, seq)) != 0;
        std::vector<float> S((size_t)(P * Nst)), b((size_t)Nst), c((size_t)Nst);
        const int64_t sb = read * state.stride[0] + head * state.stride[1];
        for (int64_t e = 0; e < P; ++e)
            for (int64_t d = 0; d < Nst; ++d)
                S[e * Nst + d] =
                    initialized ? ldt(&state, sb + e * state.stride[2] + d * state.stride[3]) : 0;
        const int64_t scratch =
            anchor ? ld_int(idx.dtype, idx.data, seq * idx.stride[0] + (width / 2) * idx.stride[1])
                   : 0;
        auto factor_at = [&](int64_t i) {
            return scratch * state.stride[0] + head * state.stride[1] +
                   (i / Nst) * state.stride[2] + (i % Nst) * state.stride[3];
        };
        if (anchor && verify) {
            /* The accepted prefix of the previous window, onto the committed state, published. */
            const int64_t replay = initialized ? (int64_t)ldt(&state, factor_at(0)) : 0;
            for (int64_t t = 0; t < (replay ? na : 0); ++t) {
                const int64_t f = 1 + t * rec;
                const float decay = ldt(&state, factor_at(f));
                for (int64_t d = 0; d < Nst; ++d) b[(size_t)d] = ldt(&state, factor_at(f + 1 + P + d));
                for (int64_t e = 0; e < P; ++e) {
                    const float u = ldt(&state, factor_at(f + 1 + e));
                    for (int64_t d = 0; d < Nst; ++d) {
                        float& s = S[e * Nst + d];
                        s = s * decay + u * b[d];
                    }
                }
            }
            for (int64_t e = 0; e < P; ++e)
                for (int64_t d = 0; d < Nst; ++d)
                    stt(&state, sb + e * state.stride[2] + d * state.stride[3], S[e * Nst + d]);
        }
        const float A = -std::exp(ldt(&alog, head * laststride(&alog)));
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
                stt(&state, factor_at(f), decay);
                for (int64_t d = 0; d < Nst; ++d) stt(&state, factor_at(f + 1 + P + d), b[d]);
            }
            for (int64_t e = 0; e < P; ++e) {
                const float xe = ldt(&x, off(x, t, head, e, P));
                const float u = step * xe;
                if (anchor && verify) stt(&state, factor_at(f + 1 + e), u);
                float result = 0;
                for (int64_t d = 0; d < Nst; ++d) {
                    float& s = S[e * Nst + d];
                    s = s * decay + u * b[d];
                    result += s * c[d];
                }
                stt(&out, off(out, t, head, e, P), result + skip * xe);
            }
            if ((verify && !anchor) || (!verify && t + 1 == end)) {
                const int64_t write =
                    ld_int(idx.dtype, idx.data,
                           seq * idx.stride[0] + (verify ? t - begin : 0) * idx.stride[1]);
                if (write >= 0)
                    for (int64_t e = 0; e < P; ++e)
                        for (int64_t d = 0; d < Nst; ++d)
                            stt(&state,
                                write * state.stride[0] + head * state.stride[1] +
                                    e * state.stride[2] + d * state.stride[3],
                                S[e * Nst + d]);
            }
        }
        if (anchor) stt(&state, factor_at(0), verify ? (float)(end - begin) : 0.0f);
    }
    return RAD_OK;
}
