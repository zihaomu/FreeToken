/* Mamba-2's selective state update (ssd_update): libref/ref_ssd.cpp's recurrence, a token at a time.
 * One wave owns four rows of a head's [head_dim, d_state] state and keeps each row's 128-wide state
 * in four registers a lane, as r4d_kda.hip does; the rows' output reductions interleave and the
 * next token's operands load while this one is reduced. Prefill runs the same recurrence and
 * commits only its final state; verify writes every candidate's (per_candidate), or each token's
 * record into the scratch slot (anchor, docs/OPS.md): the decay, u = dt' x and B, replayed onto the
 * committed state by a separate dispatch before the next window, as r4d_kda.hip's replay is, so no
 * wave overwrites a record another is still reading. The replay applies a record with the window's
 * own fmaf, so the committed state is the per-candidate one bit for bit. */
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include <climits>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace {
struct SsdMeta {
    long xr, xh, br, bh, cr, ch, dr, dh, os0, os1;
    long ss0, ss1, ss2, is0, is1, cs0, as0, hs0;
    int tokens, slots, width, verify, per_group;
    /* The anchor form: the record length 1 + head_dim + d_state and the most records a head's
     * block holds (at most 32). */
    int anchor, head_dim, rec, deepest;
};

__device__ __forceinline__ float softplus(float v) { return v > 20.0f ? v : log1pf(expf(v)); }

/* Element i of a head's scratch block, rows of 128 at the state's row pitch. */
__device__ __forceinline__ long factor_offset(long base, int i, long pitch) {
    return base + (long)(i / 128) * pitch + (i % 128);
}

/* The previous window's accepted prefix onto the committed state (column 0), published; a cold
 * sequence's committed rows are zeroed instead. One wave a state row. */
__global__ void ssd_replay_kernel(float* state, const int* idx, const int* cu, const int* accepted,
                                  const int* init, SsdMeta m) {
    const int lane = threadIdx.x & 31, e = blockIdx.x * 4 + (threadIdx.x >> 5), head = blockIdx.y,
              seq = blockIdx.z;
    if (cu[(long)seq * m.cs0] == cu[(long)(seq + 1) * m.cs0]) return;
    const int anchor = idx[(long)seq * m.is0],
              scratch = idx[(long)seq * m.is0 + (m.width / 2) * m.is1];
    if (anchor < 0 || anchor >= m.slots || scratch < 0 || scratch >= m.slots || anchor == scratch)
        return;
    const long fb = (long)scratch * m.ss0 + (long)head * m.ss1;
    const long sb = (long)anchor * m.ss0 + (long)head * m.ss1 + (long)e * m.ss2;
    if (init && !init[(long)seq * m.hs0]) {
#pragma unroll
        for (int d = 0; d < 4; ++d) state[sb + lane + d * 32] = 0;
        return;
    }
    const float count = state[fb];
    const int na = accepted ? accepted[(long)seq * m.as0] : 1;
    if (!(count >= 0 && count <= (float)m.deepest) || count != floorf(count) || na < 1 ||
        na > m.deepest || (count > 0 && na > count))
        return;
    float s[4];
#pragma unroll
    for (int d = 0; d < 4; ++d) s[d] = state[sb + lane + d * 32];
    for (int t = 0; t < (count > 0 ? na : 0); ++t) {
        const int f = 1 + t * m.rec;
        const float decay = state[factor_offset(fb, f, m.ss2)];
        const float u = state[factor_offset(fb, f + 1 + e, m.ss2)];
#pragma unroll
        for (int d = 0; d < 4; ++d)
            s[d] = fmaf(s[d], decay,
                        u * state[factor_offset(fb, f + 1 + m.head_dim + lane + 32 * d, m.ss2)]);
    }
#pragma unroll
    for (int d = 0; d < 4; ++d) state[sb + lane + d * 32] = s[d];
}

template <int R>
__global__ void ssd_rows_kernel(const uint16_t* x, const uint16_t* B, const uint16_t* C,
                                const uint16_t* dt, const float* alog, const float* dtb,
                                const float* dskip, float* state, uint16_t* out, const int* idx,
                                const int* cu, const int* accepted_ptr, const int* init,
                                SsdMeta m) {
    const int lane = threadIdx.x & 31, e0 = (blockIdx.x * 4 + (threadIdx.x >> 5)) * R;
    const int head = blockIdx.y, seq = blockIdx.z, grp = head / m.per_group;
    const int begin = cu[(long)seq * m.cs0], end = cu[(long)(seq + 1) * m.cs0];
    const int accepted = accepted_ptr ? accepted_ptr[(long)seq * m.as0] : 1;
    const int most = m.anchor ? m.deepest : m.width;
    if (begin < 0 || end <= begin || end > m.tokens || accepted < 1 || accepted > most ||
        (m.verify && end - begin > most))
        return;
    const int read = idx[(long)seq * m.is0 + (m.verify && !m.anchor ? accepted - 1 : 0) * m.is1];
    if (read >= m.slots) return;
    uint16_t* const out_row = out + (long)head * m.os1 + e0;
    if (read < 0) {
        if (lane < R)
            for (int t = begin; t < end; ++t) out_row[(long)t * m.os0 + lane] = 0;
        return;
    }
    /* Under anchor the replay dispatch has already published the accepted prefix (or zeroed a cold
     * sequence) at column 0. */
    const bool initialized = (m.anchor && m.verify) || !init || init[(long)seq * m.hs0] != 0;
    const int scratch = m.anchor ? idx[(long)seq * m.is0 + (m.width / 2) * m.is1] : 0;
    if (m.anchor && (scratch < 0 || scratch >= m.slots || scratch == read)) return;
    const long fb = (long)scratch * m.ss0 + (long)head * m.ss1;
    const long initial = (long)read * m.ss0 + (long)head * m.ss1 + (long)e0 * m.ss2;
    float s[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int d = 0; d < 4; ++d)
            s[r][d] = initialized ? state[initial + r * m.ss2 + lane + 32 * d] : 0;
    const float A = -expf(alog[head]), bias = dtb[head], skip = dskip[head];
    const uint16_t* const xb = x + (long)head * m.xh + e0;
    const uint16_t* const bb = B + (long)grp * m.bh + lane;
    const uint16_t* const cb = C + (long)grp * m.ch + lane;
    const uint16_t* const db = dt + (long)head * m.dh;
    struct Token {
        uint32_t b[4], c[4], x[R], dt;
    };
    auto fetch = [&](int t, Token& o) {
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            o.b[d] = ldg_u16(bb + (long)t * m.br + 32 * d);
            o.c[d] = ldg_u16(cb + (long)t * m.cr + 32 * d);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) o.x[r] = ldg_u16(xb + (long)t * m.xr + r);
        o.dt = ldg_u16(db + (long)t * m.dr);
    };
    Token cur;
    fetch(begin, cur);
    for (int t = begin; t < end; ++t) {
        Token next;
        /* The last token loads itself again rather than a row past `end`. */
        fetch(t + 1 < end ? t + 1 : t, next);
        const float step = softplus(bf16w_lo(cur.dt) + bias);
        const float decay = expf(step * A);
        float bv[4], cv[4], xv[R], acc[R];
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            bv[d] = bf16w_lo(cur.b[d]);
            cv[d] = bf16w_lo(cur.c[d]);
        }
        const bool record = m.anchor && m.verify;
        const int f = 1 + (t - begin) * m.rec;
        if (record && e0 == 0) {
            if (!lane) state[factor_offset(fb, f, m.ss2)] = decay;
#pragma unroll
            for (int d = 0; d < 4; ++d)
                state[factor_offset(fb, f + 1 + m.head_dim + lane + 32 * d, m.ss2)] = bv[d];
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            xv[r] = bf16w_lo(cur.x[r]);
            const float u = step * xv[r];
            if (record && lane == r) state[factor_offset(fb, f + 1 + e0 + r, m.ss2)] = u;
            acc[r] = 0;
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                s[r][d] = fmaf(s[r][d], decay, u * bv[d]);
                acc[r] += s[r][d] * cv[d];
            }
        }
        float mine = 0;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float y = r4d_wave_sum(acc[r]) + skip * xv[r];
            if (lane == r) mine = y;
        }
        if (lane < R) out_row[(long)t * m.os0 + lane] = f32_to_bf16(mine);
        if ((m.verify && !m.anchor) || (!m.verify && t + 1 == end)) {
            const int write = idx[(long)seq * m.is0 + (m.verify ? t - begin : 0) * m.is1];
            if (write >= 0 && write < m.slots) {
                const long target = (long)write * m.ss0 + (long)head * m.ss1 + (long)e0 * m.ss2;
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int d = 0; d < 4; ++d) state[target + r * m.ss2 + lane + 32 * d] = s[r][d];
            }
        }
        cur = next;
    }
    if (m.anchor && e0 == 0 && !lane) state[fb] = m.verify ? (float)(end - begin) : 0.0f;
}

/* A [tokens, heads, width] bf16 plane as rank 3, or rank 2 [tokens, heads * width]: its row and
 * head pitches. */
bool headed(const RadTensor& t, long tokens, long heads, long width, long* row, long* head) {
    if (!t.data || t.dtype != RAD_BF16 || t.shape[0] != tokens || t.stride[0] > INT_MAX)
        return false;
    if (t.rank == 3 && t.shape[1] == heads && t.shape[2] == width && t.stride[2] == 1 &&
        t.stride[1] >= width && t.stride[1] <= INT_MAX) {
        *row = t.stride[0];
        *head = t.stride[1];
        return true;
    }
    if (t.rank == 2 && t.shape[1] == heads * width && t.stride[1] == 1 &&
        t.stride[0] >= heads * width) {
        *row = t.stride[0];
        *head = width;
        return true;
    }
    return false;
}
bool vector(const RadTensor& t, uint32_t dt, long n) {
    return t.data && t.dtype == dt && t.rank == 1 && t.shape[0] >= n && t.stride[0] == 1;
}

constexpr int kRows = 4, kState = 128;

int body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 13) return RAD_E_INVAL;
    const long H = r4d_geti_or(a, "heads", 0), P = r4d_geti_or(a, "head_dim", 0);
    const long N = r4d_geti_or(a, "d_state", 0), G = r4d_geti_or(a, "groups", 0);
    if (H < 1 || H > 65535 || N != kState || P < 4 * kRows || P % (4 * kRows) || G < 1 || H % G)
        return RAD_E_SHAPE;
    const char* form = r4d_gets(a, "state_form");
    const bool anchor = form && !std::strcmp(form, "anchor");
    if (form && !anchor && std::strcmp(form, "per_candidate")) return RAD_E_INVAL;
    const char* mode = r4d_gets(a, "mode");
    if (!mode || (std::strcmp(mode, "verify") && std::strcmp(mode, "prefill"))) return RAD_E_INVAL;
    SsdMeta m{};
    m.verify = !std::strcmp(mode, "verify");
    m.anchor = anchor ? 1 : 0;
    m.head_dim = (int)P;
    m.rec = (int)(1 + P + N);
    m.deepest = (int)std::min<long>(32, (P * N - 1) / m.rec);
    const auto &state = a->t[7], &idx = a->t[8], &cu = a->t[9];
    const long tokens = a->t[0].shape[0];
    if (tokens < 1 || tokens > INT_MAX || state.rank != 4 || state.dtype != RAD_F32 ||
        !state.data || state.shape[0] < 1 || state.shape[0] > INT_MAX || state.shape[1] != H ||
        state.shape[2] != P || state.shape[3] != N || state.stride[3] != 1 ||
        state.stride[2] < N || state.stride[1] < P * state.stride[2] ||
        state.stride[0] < H * state.stride[1] || cu.rank != 1 || cu.dtype != RAD_I32 ||
        !cu.data || cu.shape[0] < 2 || cu.shape[0] > 65536 || cu.stride[0] < 1 ||
        idx.rank != 2 || idx.dtype != RAD_I32 || !idx.data || idx.shape[0] < cu.shape[0] - 1 ||
        idx.shape[1] < 1 || idx.shape[1] > INT_MAX || idx.stride[1] < 1 ||
        idx.stride[0] < idx.shape[1] * idx.stride[1] || (anchor && idx.shape[1] < 2))
        return RAD_E_SHAPE;
    if (!headed(a->t[0], tokens, H, P, &m.xr, &m.xh) ||
        !headed(a->t[1], tokens, G, N, &m.br, &m.bh) ||
        !headed(a->t[2], tokens, G, N, &m.cr, &m.ch) ||
        !headed(a->t[3], tokens, H, 1, &m.dr, &m.dh) ||
        !headed(a->t[12], tokens, H, P, &m.os0, &m.os1) || !vector(a->t[4], RAD_F32, H) ||
        !vector(a->t[5], RAD_F32, H) || !vector(a->t[6], RAD_F32, H))
        return RAD_E_SHAPE;
    const RadTensor *accepted = rad_arg_in(a, 10), *init = rad_arg_in(a, 11);
    if ((accepted && !vector(*accepted, RAD_I32, cu.shape[0] - 1)) ||
        (init && !vector(*init, RAD_I32, cu.shape[0] - 1)))
        return RAD_E_SHAPE;
    m.tokens = (int)tokens;
    m.slots = (int)state.shape[0];
    m.width = (int)idx.shape[1];
    m.per_group = (int)(H / G);
    m.ss0 = state.stride[0];
    m.ss1 = state.stride[1];
    m.ss2 = state.stride[2];
    m.is0 = idx.stride[0];
    m.is1 = idx.stride[1];
    m.cs0 = cu.stride[0];
    if (accepted) m.as0 = accepted->stride[0];
    if (init) m.hs0 = init->stride[0];
    const unsigned seqs = (unsigned)(cu.shape[0] - 1);
    if (anchor && m.verify) {
        const dim3 rgrid((unsigned)(P / 4), (unsigned)H, seqs);
        const int rc = R4D_EMIT(stream, ssd_replay_kernel, rgrid, dim3(128), 0,
                                (float*)state.data, (const int*)idx.data,
                                (const int*)cu.data,
                                accepted ? (const int*)accepted->data : nullptr,
                                init ? (const int*)init->data : nullptr, m);
        if (rc != RAD_OK) return rc;
    }
    const dim3 grid((unsigned)(P / (4 * kRows)), (unsigned)H, seqs);
    return R4D_EMIT(stream, (ssd_rows_kernel<kRows>), grid, dim3(128), 0,
                    (const uint16_t*)a->t[0].data, (const uint16_t*)a->t[1].data,
                    (const uint16_t*)a->t[2].data, (const uint16_t*)a->t[3].data,
                    (const float*)a->t[4].data, (const float*)a->t[5].data,
                    (const float*)a->t[6].data, (float*)state.data, (uint16_t*)a->t[12].data,
                    (const int*)idx.data, (const int*)cu.data,
                    accepted ? (const int*)accepted->data : nullptr,
                    init ? (const int*)init->data : nullptr, m);
}
}
extern "C" int r4d_rad_ssd_update(const RadArgs* a, RadStream stream) { return body(a, stream); }
