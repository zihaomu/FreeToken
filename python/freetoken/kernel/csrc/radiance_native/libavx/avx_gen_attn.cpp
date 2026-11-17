/* avx_gen_attn.cpp -- attn_prefix_self: attention over a cached conditioning prefix and the pass's
 * own rows (docs/OPS.md, "Image generation"), the diffusion transformer's attention.
 *
 * Keys of entry s, in order: its `seqused[s]` prefix positions in the paged cache, every one
 * visible, then its own rows [cu[s], cu[s+1]) beside q, of which own row i sees j <= i with
 * `causal` and all of them otherwise. A prefix key in a missing block (a negative table entry) is
 * a zero key and a zero value that still takes its share of the softmax, as libref reads it.
 *
 * Flash-attention form. Each (entry, kv head)'s key sequence is packed once per launch into
 * 64-key tiles, then a task is (entry, kv head, 32 query positions) with the gqa query heads of
 * that kv head: per tile, S = Q K^T as broadcast-against-tile products (no horizontal sums), the
 * causal mask, the online softmax (running max moved at tile boundaries), then O += P V. Packs are
 * made for a group of kv heads at a time, at most ~64 MiB, so the packed tiles of the heads in
 * flight stay near cache instead of all heads' at once.
 *
 *   bf16 (AVX512_BF16 and bf16 q, k, v, cache, head_dim a multiple of 64): vdpbf16ps for both
 *         products, P rounded to bf16 for P V as PyTorch's CPU flash attention does (and
 *         avx_attn.cpp: attn_dense_bf16).
 *   f32   every other case: f32 tiles, vf FMAs, P in f32.
 *
 * The int8 score form (`qk` "i8"): the query and each prefix key are quantised per (row, head) by
 * libref's rule (scale amax / 127, 1 for an all-zero row; code rint(clamp(x / scale, +-127))), the
 * own keys arrive as codes and scales, and a score is ((dot * sq) * sk) * scale with the integer
 * dot exact in f32 (|dot| < 2^24 up to head_dim 1024). Codes are exact in bf16 too, so the bf16
 * path computes the same scores; only P V differs from libref, by the softmax order and P's
 * rounding.
 *
 * Results differ from libref's two-pass softmax in summation order and the rescales' roundings,
 * within the attention tolerance. Each output row is one task's, so they do not depend on the
 * thread count.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <vector>

using namespace avx;

namespace {

/* The widest head, libref's (its stack rows, and the int8 dot's exact range). */
enum { PS_HD_MAX = 1024 };
/* Keys per tile and query positions per task. */
enum { KT = 64, QB = 32 };
/* The packed tiles of the kv heads in flight. */
constexpr int64_t kPackBudget = (int64_t)64 << 20;

struct Geo {
    const RadTensor *q, *k, *v, *kv, *bt, *su, *cu, *o, *kdes, *vdes, *k8, *ks;
    bool i8;
    int causal;
    int64_t hd, gqa, kvh, qh, n_seq, bs, max_blocks, bt_rs, bt_cs;
    int64_t q_rs, k_rs, v_rs, o_rs, k8_rs, ks_rs, sb, sh, so;
    int64_t rows_total;    /* the fp8 P.V form's rows */
    float scale;
};

/* f32 to the nearest E4M3 and back. */
inline float e4m3_round(float x) {
    const uint8_t c = rad_f32_to_fp8e4m3(x);
    float y;
    cvt_fp8e4m3_to_f32(&c, &y, 1);
    return y;
}

inline int64_t cu_at(const RadTensor* cu, int64_t i) { return ldi(cu, offlin(cu, i)); }

/* One (entry, kv head)'s key sequence: P prefix keys then the entry's own rows from b0. */
struct Seq {
    int64_t s, kh, P, b0, len, nk, ntile;
    size_t off;            /* bytes into the pack */
};

/* libref's per-row int8 rule: codes into `code` (may be `x`), the scale returned. */
inline float q8_row(const float* x, int64_t n, float* code) {
    const float am = row_amax(x, n);
    const float sc = am > 0.0f ? am * (1.0f / 127.0f) : 1.0f;
    const float inv = 1.0f / sc;
    const vf vi = vf_set1(inv), lo = vf_set1(-127.0f), hi = vf_set1(127.0f);
    int64_t d = 0;
#if AVX_LEVEL >= 1
    for (; d + VF_N <= n; d += VF_N) {
        const vf c = vf_min(vf_max(vf_mul(vf_loadu(x + d), vi), lo), hi);
#if AVX_LEVEL == 3
        vf_storeu(code + d, _mm512_roundscale_ps(c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#else
        vf_storeu(code + d, _mm256_round_ps(c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#endif
    }
#else
    (void)vi; (void)lo; (void)hi;
#endif
    for (; d < n; ++d) code[d] = std::rint(std::min(std::max(x[d] * inv, -127.0f), 127.0f));
    return sc;
}

/* Key (or value) j of `q`'s sequence as f32 into out[0, hd). A prefix key carries its entry's
 * descale per element, as libref applies it. */
void key_f32(const Geo& g, const Seq& q, int64_t j, bool val, float* out) {
    const int64_t hd = g.hd;
    if (j < q.P) {
        const int64_t blk = ldi(g.bt, q.s * g.bt_rs + (j / g.bs) * g.bt_cs);
        if (blk < 0) {
            for (int64_t d = 0; d < hd; ++d) out[d] = 0.0f;
            return;
        }
        const int64_t base = blk * g.sb + q.kh * g.sh + (j % g.bs) * g.so + (val ? hd : 0);
        row_to_f32(g.kv->dtype, byte_at(g.kv->data, g.kv->dtype, base), out, hd);
        const RadTensor* des = val ? g.vdes : g.kdes;
        const float m = des ? ldt(des, q.s * g.kvh + q.kh) : 1.0f;
        if (m != 1.0f)
            for (int64_t d = 0; d < hd; ++d) out[d] *= m;
        return;
    }
    const int64_t row = q.b0 + j - q.P;
    const RadTensor* t = val ? g.v : g.k;
    const int64_t base = row * (val ? g.v_rs : g.k_rs) + q.kh * hd;
    row_to_f32(t->dtype, byte_at(t->data, t->dtype, base), out, hd);
}

/* The int8 form's key j: codes into out and its scale. */
float key_i8(const Geo& g, const Seq& q, int64_t j, float* out) {
    if (j < q.P) {
        key_f32(g, q, j, false, out);
        return q8_row(out, g.hd, out);
    }
    const int64_t row = q.b0 + j - q.P;
    row_to_f32(g.k8->dtype, byte_at(g.k8->data, g.k8->dtype, row * g.k8_rs + q.kh * g.hd), out,
               g.hd);
    return ldt(g.ks, row * g.ks_rs + q.kh);
}

/* The keys a task's last query can see: all, or with `causal` up to its own row. */
inline int64_t visible_keys(const Geo& g, const Seq& q, int64_t q1) {
    return g.causal ? std::min(q.nk, q.P + q1) : q.nk;
}

/* S *= scale, or the int8 form's ((S * sq[r]) * sk) * scale; then -inf past the tile's last key
 * and, with `causal`, for own keys after a row's position. */
void finish_scores(const Geo& g, const Seq& q, float* S, int64_t R, int64_t t, int64_t q0,
                   const float* sq, const float* sk) {
    const vf vs = vf_set1(g.scale);
    for (int64_t r = 0; r < R; ++r) {
        float* sr = S + r * KT;
        if (g.i8) {
            const vf vq = vf_set1(sq[r]);
            int64_t j = 0;
            for (; j + VF_N <= KT; j += VF_N)
                vf_storeu(sr + j, vf_mul(vf_mul(vf_mul(vf_loadu(sr + j), vq), vf_loadu(sk + j)), vs));
            for (; j < KT; ++j) sr[j] = ((sr[j] * sq[r]) * sk[j]) * g.scale;
        } else {
            int64_t j = 0;
            for (; j + VF_N <= KT; j += VF_N) vf_storeu(sr + j, vf_mul(vf_loadu(sr + j), vs));
            for (; j < KT; ++j) sr[j] *= g.scale;
        }
        const int64_t i = q0 + r / g.gqa;
        const int64_t end = g.causal ? std::min(q.nk, q.P + i + 1) : q.nk;
        for (int64_t j = std::max<int64_t>(0, end - t * KT); j < KT; ++j) sr[j] = -INFINITY;
    }
}

/* The online softmax over one tile of R rows: m, l and acc (rows of `ld`) rescaled at a new
 * maximum, S replaced by exp(S - m). A row with nothing visible in the tile becomes zeros. */
void softmax_tile(float* S, int64_t R, float* m, float* l, float* acc, int64_t ld, int64_t width) {
    for (int64_t r = 0; r < R; ++r) {
        float* sr = S + r * KT;
        const float tmax = row_max(sr, KT);
        if (tmax == -INFINITY) {
            for (int64_t j = 0; j < KT; ++j) sr[j] = 0.0f;
            continue;
        }
        if (tmax > m[r]) {
            const float alpha = m[r] == -INFINITY ? 0.0f : sc_exp(m[r] - tmax);
            if (alpha != 1.0f) {
                float* ar = acc + r * ld;
                const vf va = vf_set1(alpha);
                int64_t d = 0;
                for (; d + VF_N <= width; d += VF_N) vf_storeu(ar + d, vf_mul(vf_loadu(ar + d), va));
                for (; d < width; ++d) ar[d] *= alpha;
                l[r] *= alpha;
            }
            m[r] = tmax;
        }
        const vf vm = vf_set1(m[r]);
        int64_t j = 0;
        for (; j + VF_N <= KT; j += VF_N) vf_storeu(sr + j, vf_exp(vf_sub(vf_loadu(sr + j), vm)));
        for (; j < KT; ++j) sr[j] = sc_exp(sr[j] - m[r]);
        l[r] += row_sum(sr, KT);
    }
}

/* out = acc / l (0 for a row that saw nothing), stored to `o` at its row and head. */
void store_rows(const Geo& g, const Seq& q, int64_t q0, int64_t R, float* acc, int64_t ld,
                const float* m, const float* l) {
    for (int64_t r = 0; r < R; ++r) {
        const int64_t row = q.b0 + q0 + r / g.gqa, h = q.kh * g.gqa + r % g.gqa;
        float* ar = acc + r * ld;
        const float inv = (m[r] == -INFINITY || l[r] <= 0.0f) ? 0.0f : 1.0f / l[r];
        const vf vi = vf_set1(inv);
        int64_t d = 0;
        for (; d + VF_N <= g.hd; d += VF_N) vf_storeu(ar + d, vf_mul(vf_loadu(ar + d), vi));
        for (; d < g.hd; ++d) ar[d] *= inv;
        out_row(g.o, row * g.o_rs + h * g.hd, g.hd, ar);
    }
}

/* ================================================================== the f32 form
 *
 * A tile packed as Kt [hd][64] (each dimension's 64 keys contiguous, so a query element broadcast
 * against four vectors gives 4 * VF_N scores) and V [64][hp], hp the head rounded up to 4 * VF_N
 * with zero columns, so P V runs whole vectors. The int8 form's key scales follow, [64]. */
#if AVX_LEVEL == 3
constexpr int RB = 6;
#else
constexpr int RB = 2;
#endif
constexpr int64_t CK = 4 * VF_N;

/* The next row block of a tile: `top` rows, except that the last two blocks are kept at four or
 * more (32 rows go 6 6 6 6 4 4). A one- or two-row block has four or eight accumulators, too few
 * to cover the dot product's latency, and would cost nearly as much as a full one. */
inline int row_block(int64_t left, int top) {
    if (top < 4) return left >= top ? top : (int)left;
    if (left >= top + 4 || left == top) return top;
    return left > top ? (int)(left - 4) : (int)left;
}

/* fn<mb>(args...) for a block of 1..6 rows. */
#define PS_ROWS(fn, mb, ...)                              \
    switch (mb) {                                         \
        case 6: fn<6>(__VA_ARGS__); break;                \
        case 5: fn<5>(__VA_ARGS__); break;                \
        case 4: fn<4>(__VA_ARGS__); break;                \
        case 3: fn<3>(__VA_ARGS__); break;                \
        case 2: fn<2>(__VA_ARGS__); break;                \
        default: fn<1>(__VA_ARGS__); break;               \
    }

inline int64_t f32_hp(int64_t hd) { return (hd + CK - 1) / CK * CK; }
inline size_t f32_tile_floats(int64_t hd) { return (size_t)(hd * KT + KT * f32_hp(hd) + KT); }

void pack_tile_f32(const Geo& g, const Seq& q, int64_t t, float* tile, float* row) {
    const int64_t hd = g.hd, hp = f32_hp(hd);
    float* Kt = tile;
    float* Vt = tile + hd * KT;
    float* sk = Vt + KT * hp;
    for (int64_t jj = 0; jj < KT; ++jj) {
        const int64_t j = t * KT + jj;
        float* vr = Vt + jj * hp;
        if (j >= q.nk) {
            for (int64_t d = 0; d < hd; ++d) Kt[d * KT + jj] = 0.0f;
            for (int64_t d = 0; d < hp; ++d) vr[d] = 0.0f;
            sk[jj] = 0.0f;
            continue;
        }
        sk[jj] = g.i8 ? key_i8(g, q, j, row) : (key_f32(g, q, j, false, row), 1.0f);
        for (int64_t d = 0; d < hd; ++d) Kt[d * KT + jj] = row[d];
        key_f32(g, q, j, true, vr);
        for (int64_t d = hd; d < hp; ++d) vr[d] = 0.0f;
    }
}

/* S[i][0, 64) = Q[i] . K for MB_ rows of Q (pitch ldq) over the packed Kt. */
template <int MB_>
inline void qk_f32(const float* Q, int64_t ldq, const float* Kt, int64_t hd, float* S) {
    for (int64_t c = 0; c < KT; c += CK) {
        vf acc[MB_][4];
        for (int i = 0; i < MB_; ++i)
            for (int v = 0; v < 4; ++v) acc[i][v] = vf_zero();
        for (int64_t d = 0; d < hd; ++d) {
            const float* kr = Kt + d * KT + c;
            const vf k0 = vf_loadu(kr), k1 = vf_loadu(kr + VF_N);
            const vf k2 = vf_loadu(kr + 2 * VF_N), k3 = vf_loadu(kr + 3 * VF_N);
            for (int i = 0; i < MB_; ++i) {
                const vf b = vf_set1(Q[i * ldq + d]);
                acc[i][0] = vf_fma(b, k0, acc[i][0]);
                acc[i][1] = vf_fma(b, k1, acc[i][1]);
                acc[i][2] = vf_fma(b, k2, acc[i][2]);
                acc[i][3] = vf_fma(b, k3, acc[i][3]);
            }
        }
        for (int i = 0; i < MB_; ++i)
            for (int v = 0; v < 4; ++v) vf_storeu(S + i * KT + c + v * VF_N, acc[i][v]);
    }
}

/* acc[i][0, hp) += P[i] . V for MB_ rows over the tile's `kt` keys. */
template <int MB_>
inline void pv_f32(float* acc, int64_t hp, const float* Pm, const float* Vt, int64_t kt) {
    for (int64_t c = 0; c < hp; c += CK) {
        vf o[MB_][4];
        for (int i = 0; i < MB_; ++i)
            for (int v = 0; v < 4; ++v) o[i][v] = vf_loadu(acc + i * hp + c + v * VF_N);
        for (int64_t j = 0; j < kt; ++j) {
            const float* vr = Vt + j * hp + c;
            const vf v0 = vf_loadu(vr), v1 = vf_loadu(vr + VF_N);
            const vf v2 = vf_loadu(vr + 2 * VF_N), v3 = vf_loadu(vr + 3 * VF_N);
            for (int i = 0; i < MB_; ++i) {
                const vf p = vf_set1(Pm[i * KT + j]);
                o[i][0] = vf_fma(p, v0, o[i][0]);
                o[i][1] = vf_fma(p, v1, o[i][1]);
                o[i][2] = vf_fma(p, v2, o[i][2]);
                o[i][3] = vf_fma(p, v3, o[i][3]);
            }
        }
        for (int i = 0; i < MB_; ++i)
            for (int v = 0; v < 4; ++v) vf_storeu(acc + i * hp + c + v * VF_N, o[i][v]);
    }
}

void task_f32(const Geo& g, const Seq& q, const uint8_t* pack, int64_t q0, int64_t q1) {
    const int64_t hd = g.hd, hp = f32_hp(hd), R = (q1 - q0) * g.gqa, R_MAX = QB * g.gqa;
    float* frame = scratch_f32((size_t)(R_MAX * hp + R_MAX * hd + R_MAX * KT + 3 * R_MAX + hp) + 64);
    float* acc = frame;
    float* Q = acc + R_MAX * hp;
    float* S = Q + R_MAX * hd;
    float* m = S + R_MAX * KT;
    float* l = m + R_MAX;
    float* sq = l + R_MAX;
    for (int64_t r = 0; r < R; ++r) {
        const int64_t row = q.b0 + q0 + r / g.gqa, h = q.kh * g.gqa + r % g.gqa;
        float* qr = Q + r * hd;
        row_to_f32(g.q->dtype, byte_at(g.q->data, g.q->dtype, row * g.q_rs + h * hd), qr, hd);
        sq[r] = g.i8 ? q8_row(qr, hd, qr) : 1.0f;
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }
    std::memset(acc, 0, sizeof(float) * (size_t)(R * hp));
    const int64_t nvis = visible_keys(g, q, q1);
    const size_t tf = f32_tile_floats(hd);
    for (int64_t t = 0; t * KT < nvis; ++t) {
        const float* tile = (const float*)(pack + q.off) + (size_t)t * tf;
        const float* Kt = tile;
        const float* Vt = tile + hd * KT;
        const float* sk = Vt + KT * hp;
        for (int64_t r = 0, mb; r < R; r += mb) {
            mb = row_block(R - r, RB);
            PS_ROWS(qk_f32, mb, Q + r * hd, hd, Kt, hd, S + r * KT);
        }
        finish_scores(g, q, S, R, t, q0, sq, sk);
        softmax_tile(S, R, m, l, acc, hp, hp);
        const int64_t kt = std::min<int64_t>(KT, q.nk - t * KT);
        for (int64_t r = 0, mb; r < R; r += mb) {
            mb = row_block(R - r, RB);
            PS_ROWS(pv_f32, mb, acc + r * hp, hp, S + r * KT, Vt, kt);
        }
    }
    store_rows(g, q, q0, R, acc, hp, m, l);
}

/* ================================================================== the bf16 form
 *
 * A tile packed as kp [hd/2][64] u32, key j's dimensions (2d, 2d+1) as one bf16 pair, and
 * vp [32][hd] u32, keys (2k, 2k+1) at column d as one pair; the int8 form's key scales follow,
 * [64] f32. Codes are integers of at most 127 in magnitude, exact in bf16. */
#if AVX_LEVEL == 3
#define PS_BF16 __attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16")))

inline size_t bf16_tile_bytes(int64_t hd) {
    return (size_t)(hd / 2 * KT) * 4 + (size_t)(KT / 2 * hd) * 4 + (size_t)KT * 4;
}

/* A key row's hd bf16 values (the cache's or the own rows' own bits, or exact codes), and for the
 * int8 form its scale. `row` holds hd floats. */
float key_bf16(const Geo& g, const Seq& q, int64_t j, bool val, uint16_t* out, float* row) {
    const int64_t hd = g.hd;
    if (g.i8 && !val) {
        const float sk = key_i8(g, q, j, row);
        row_from_f32(RAD_BF16, row, out, hd);
        return sk;
    }
    const uint16_t* src = nullptr;
    if (j < q.P) {
        const int64_t blk = ldi(g.bt, q.s * g.bt_rs + (j / g.bs) * g.bt_cs);
        if (blk >= 0)
            src = (const uint16_t*)g.kv->data + blk * g.sb + q.kh * g.sh + (j % g.bs) * g.so +
                  (val ? hd : 0);
    } else {
        const int64_t r = q.b0 + j - q.P;
        src = (const uint16_t*)(val ? g.v : g.k)->data + r * (val ? g.v_rs : g.k_rs) + q.kh * hd;
    }
    if (src) std::memcpy(out, src, (size_t)hd * 2);
    else     std::memset(out, 0, (size_t)hd * 2);
    return 1.0f;
}

void pack_tile_bf16(const Geo& g, const Seq& q, int64_t t, uint8_t* tile, float* row,
                    uint16_t* kv2) {
    const int64_t hd = g.hd, D2 = hd / 2;
    uint32_t* kp = (uint32_t*)tile;
    uint32_t* vp = kp + D2 * KT;
    float* sk = (float*)(vp + (KT / 2) * hd);
    uint16_t* ka = kv2;
    uint16_t* va = kv2 + 2 * hd;
    for (int64_t k2 = 0; k2 < KT / 2; ++k2) {
        for (int e = 0; e < 2; ++e) {
            const int64_t jj = 2 * k2 + e, j = t * KT + jj;
            uint16_t* kr = ka + e * hd;
            uint16_t* vr = va + e * hd;
            if (j >= q.nk) {
                std::memset(kr, 0, (size_t)hd * 2);
                std::memset(vr, 0, (size_t)hd * 2);
                sk[jj] = 0.0f;
                continue;
            }
            sk[jj] = key_bf16(g, q, j, false, kr, row);
            key_bf16(g, q, j, true, vr, row);
            for (int64_t d2 = 0; d2 < D2; ++d2)
                kp[d2 * KT + jj] = (uint32_t)kr[2 * d2] | ((uint32_t)kr[2 * d2 + 1] << 16);
        }
        uint32_t* vo = vp + k2 * hd;
        for (int64_t d = 0; d < hd; ++d) vo[d] = (uint32_t)va[d] | ((uint32_t)va[hd + d] << 16);
    }
}

/* S[i][0, 64) = Q[i] . K over the packed tile, MB_ query rows given as bf16 row pointers. */
template <int MB_>
PS_BF16 inline void qk_bf16(const uint16_t* const* qr, const uint32_t* kp, int64_t D2, float* S) {
    __m512 acc[MB_][4];
#pragma GCC unroll 8
    for (int i = 0; i < MB_; ++i)
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) acc[i][v] = _mm512_setzero_ps();
    for (int64_t d2 = 0; d2 < D2; ++d2) {
        const uint32_t* kr = kp + d2 * KT;
        const __m512bh b0 = (__m512bh)_mm512_loadu_si512((const void*)(kr));
        const __m512bh b1 = (__m512bh)_mm512_loadu_si512((const void*)(kr + 16));
        const __m512bh b2 = (__m512bh)_mm512_loadu_si512((const void*)(kr + 32));
        const __m512bh b3 = (__m512bh)_mm512_loadu_si512((const void*)(kr + 48));
#pragma GCC unroll 8
        for (int i = 0; i < MB_; ++i) {
            uint32_t pr;
            std::memcpy(&pr, qr[i] + 2 * d2, 4);
            const __m512bh a = (__m512bh)_mm512_set1_epi32((int)pr);
            acc[i][0] = _mm512_dpbf16_ps(acc[i][0], a, b0);
            acc[i][1] = _mm512_dpbf16_ps(acc[i][1], a, b1);
            acc[i][2] = _mm512_dpbf16_ps(acc[i][2], a, b2);
            acc[i][3] = _mm512_dpbf16_ps(acc[i][3], a, b3);
        }
    }
#pragma GCC unroll 8
    for (int i = 0; i < MB_; ++i)
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) _mm512_storeu_ps(S + i * KT + 16 * v, acc[i][v]);
}

/* acc[i][c0, c0 + 64) += P[i] . V, MB_ rows of bf16 P pairs (pp[i * 32 + k2]) against the
 * tile's value pairs. */
template <int MB_>
PS_BF16 inline void pv_bf16(float* acc, int64_t hd, const uint32_t* pp, const uint32_t* vp,
                            int64_t K2, int64_t c0) {
    __m512 o[MB_][4];
#pragma GCC unroll 8
    for (int i = 0; i < MB_; ++i)
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) o[i][v] = _mm512_loadu_ps(acc + i * hd + c0 + 16 * v);
    for (int64_t k2 = 0; k2 < K2; ++k2) {
        const uint32_t* vr = vp + k2 * hd + c0;
        const __m512bh b0 = (__m512bh)_mm512_loadu_si512((const void*)(vr));
        const __m512bh b1 = (__m512bh)_mm512_loadu_si512((const void*)(vr + 16));
        const __m512bh b2 = (__m512bh)_mm512_loadu_si512((const void*)(vr + 32));
        const __m512bh b3 = (__m512bh)_mm512_loadu_si512((const void*)(vr + 48));
#pragma GCC unroll 8
        for (int i = 0; i < MB_; ++i) {
            const __m512bh a = (__m512bh)_mm512_set1_epi32((int)pp[i * 32 + k2]);
            o[i][0] = _mm512_dpbf16_ps(o[i][0], a, b0);
            o[i][1] = _mm512_dpbf16_ps(o[i][1], a, b1);
            o[i][2] = _mm512_dpbf16_ps(o[i][2], a, b2);
            o[i][3] = _mm512_dpbf16_ps(o[i][3], a, b3);
        }
    }
#pragma GCC unroll 8
    for (int i = 0; i < MB_; ++i)
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) _mm512_storeu_ps(acc + i * hd + c0 + 16 * v, o[i][v]);
}

/* A row of 64 probabilities to 32 bf16 key pairs, round to nearest even. */
PS_BF16 inline void p_pairs(const float* sr, uint32_t* pr) {
    _mm512_storeu_si512((void*)pr,
                        (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(sr + 16), _mm512_loadu_ps(sr)));
    _mm512_storeu_si512((void*)(pr + 16), (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(sr + 48),
                                                                      _mm512_loadu_ps(sr + 32)));
}

PS_BF16 void task_bf16(const Geo& g, const Seq& q, const uint8_t* pack, int64_t q0, int64_t q1) {
    const int64_t hd = g.hd, D2 = hd / 2, R = (q1 - q0) * g.gqa, R_MAX = QB * g.gqa;
    float* frame = scratch_f32((size_t)(R_MAX * hd + R_MAX * KT + 3 * R_MAX + R_MAX * KT / 2 +
                                        R_MAX * hd / 2 + hd) + 64);
    float* acc = frame;
    float* S = acc + R_MAX * hd;
    float* m = S + R_MAX * KT;
    float* l = m + R_MAX;
    float* sq = l + R_MAX;
    uint32_t* pp = (uint32_t*)(sq + R_MAX);
    uint16_t* qcodes = (uint16_t*)(pp + R_MAX * KT / 2);
    float* row = (float*)(qcodes + R_MAX * hd);
    const uint16_t* qrow[16 * QB];
    for (int64_t r = 0; r < R; ++r) {
        const int64_t qrw = q.b0 + q0 + r / g.gqa, h = q.kh * g.gqa + r % g.gqa;
        const uint16_t* src = (const uint16_t*)g.q->data + qrw * g.q_rs + h * hd;
        if (g.i8) {
            row_to_f32(RAD_BF16, src, row, hd);
            sq[r] = q8_row(row, hd, row);
            uint16_t* c = qcodes + r * hd;
            row_from_f32(RAD_BF16, row, c, hd);
            qrow[r] = c;
        } else {
            sq[r] = 1.0f;
            qrow[r] = src;
        }
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }
    std::memset(acc, 0, sizeof(float) * (size_t)(R * hd));
    const int64_t nvis = visible_keys(g, q, q1);
    const size_t tb = bf16_tile_bytes(hd);
    for (int64_t t = 0; t * KT < nvis; ++t) {
        const uint8_t* tile = pack + q.off + (size_t)t * tb;
        const uint32_t* kp = (const uint32_t*)tile;
        const uint32_t* vp = kp + D2 * KT;
        const float* sk = (const float*)(vp + (KT / 2) * hd);
        for (int64_t r = 0, mb; r < R; r += mb) {
            mb = row_block(R - r, 6);
            PS_ROWS(qk_bf16, mb, qrow + r, kp, D2, S + r * KT);
        }
        finish_scores(g, q, S, R, t, q0, sq, sk);
        softmax_tile(S, R, m, l, acc, hd, hd);
        for (int64_t r = 0; r < R; ++r) p_pairs(S + r * KT, pp + r * 32);
        const int64_t K2 = (std::min<int64_t>(KT, q.nk - t * KT) + 1) / 2;
        for (int64_t c0 = 0; c0 < hd; c0 += 64)
            for (int64_t r = 0, mb; r < R; r += mb) {
                mb = row_block(R - r, 6);
                PS_ROWS(pv_bf16, mb, acc + r * hd, hd, pp + r * 32, vp, K2, c0);
            }
    }
    store_rows(g, q, q0, R, acc, hd, m, l);
}
#endif

}  /* namespace */

/* ------------------------------------------------------------------ attn_prefix_self, fp8 P.V */
/* libref's fp8 P.V form, prefix-free, row by row: every score (bf16 or int8 form, libref's
 * order), the max, then p = exp(s - max) in f64 rounded to f32, w = e4m3(448 p), and
 * o = (sum w v) v_scale / (448 sum p), sums in key order. Two passes, as libref's, because the
 * weights are rounded against the row's final max. */
static int ps_pv8(const Geo& g, const RadTensor* vsc) {
    const int64_t hd = g.hd, qh = g.qh;
    AVX_PARALLEL_FOR_IF(g.rows_total * qh * hd >= AVX_PAR_MIN)
    for (int64_t idx = 0; idx < g.rows_total * qh; ++idx) {
        const int64_t h = idx % qh, r = cu_at(g.cu, 0) + idx / qh, kh = h / g.gqa;
        int64_t s = 0;
        while (s + 1 < g.n_seq && r >= cu_at(g.cu, s + 1)) ++s;
        const int64_t b0 = cu_at(g.cu, s), b1 = cu_at(g.cu, s + 1), i = r - b0;
        const int64_t nk = g.causal ? i + 1 : b1 - b0;
        float* qf = scratch_f32((size_t)(3 * hd + nk));
        float* kf = qf + hd;
        float* acc = kf + hd;
        float* sc = acc + hd;
        for (int64_t d = 0; d < hd; ++d) qf[d] = ldt(g.q, r * g.q_rs + h * hd + d);
        float sq = 1.0f;
        if (g.i8) {
            float am = 0.0f;
            for (int64_t d = 0; d < hd; ++d) am = std::max(am, std::fabs(qf[d]));
            sq = am > 0.0f ? am * (1.0f / 127.0f) : 1.0f;
            const float inv = 1.0f / sq;
            for (int64_t d = 0; d < hd; ++d)
                qf[d] = std::rint(std::min(std::max(qf[d] * inv, -127.0f), 127.0f));
        }
        float mx = -INFINITY;
        for (int64_t j = 0; j < nk; ++j) {
            const int64_t row = b0 + j;
            float dot = 0.0f;
            if (g.i8) {
                for (int64_t d = 0; d < hd; ++d) kf[d] = ldt(g.k8, row * g.k8_rs + kh * hd + d);
                for (int64_t d = 0; d < hd; ++d) dot += qf[d] * kf[d];
                dot = dot * sq * ldt(g.ks, row * g.ks_rs + kh) * g.scale;
            } else {
                const int64_t kb = row * g.k_rs + kh * hd;
                for (int64_t d = 0; d < hd; ++d) dot += qf[d] * ldt(g.k, kb + d);
                dot *= g.scale;
            }
            sc[j] = dot;
            mx = std::max(mx, dot);
        }
        for (int64_t d = 0; d < hd; ++d) acc[d] = 0.0f;
        float sum = 0.0f;
        for (int64_t j = 0; j < nk; ++j) {
            const float p = (float)std::exp((double)(sc[j] - mx));
            sum += p;
            const float w = e4m3_round(448.0f * p);
            const int64_t vb = (b0 + j) * g.v_rs + kh * hd;
            for (int64_t d = 0; d < hd; ++d) acc[d] += w * ldt(g.v, vb + d);
        }
        const float inv8 = sum > 0.0f ? 1.0f / (448.0f * sum) : 0.0f;
        for (int64_t d = 0; d < hd; ++d)
            stt(g.o, r * g.o_rs + h * hd + d, acc[d] * ldt(vsc, offlin(vsc, kh * hd + d)) * inv8);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ attn_prefix_self */
AVX_KERNEL(attn_prefix_self) {
    if (a->n_t < 12) return RAD_E_INVAL;
    Geo g{};
    g.q = &a->t[0]; g.k = &a->t[1]; g.v = &a->t[2]; g.cu = &a->t[6]; g.o = &a->t[11];
    /* The paged prefix, all three operands or none (every prefix empty). */
    g.kv = rad_arg_in(a, 3);
    g.bt = rad_arg_in(a, 4);
    g.su = rad_arg_in(a, 5);
    if ((g.kv == nullptr) != (g.bt == nullptr) || (g.kv == nullptr) != (g.su == nullptr))
        return RAD_E_INVAL;
    g.kdes = rad_arg_in(a, 7);
    g.vdes = rad_arg_in(a, 8);
    g.k8 = rad_arg_in(a, 9);
    g.ks = rad_arg_in(a, 10);
    g.i8 = std::strcmp(p_str(a, "qk", "bf16"), "i8") == 0;
    if (g.i8 != (g.k8 != nullptr) || g.i8 != (g.ks != nullptr)) return RAD_E_INVAL;
    if (g.i8 && (g.kdes || g.vdes)) return RAD_E_UNSUPPORTED;
    g.hd = p_int(a, "head_dim", 0);
    g.gqa = p_int(a, "gqa", 1);
    g.causal = (int)p_int(a, "causal", 0);
    g.bs = !g.kv ? 1 : g.kv->rank == 4 ? g.kv->shape[2] : p_int(a, "block_size", 0);
    if (g.hd <= 0 || g.gqa <= 0 || g.bs <= 0) return RAD_E_SHAPE;
    if (g.hd > PS_HD_MAX) return RAD_E_UNSUPPORTED;
    /* Without a cache the key heads are the own keys' width over head_dim. */
    g.kvh = g.kv ? (g.kv->rank == 4 ? g.kv->shape[1] : 0)
          : g.k->rank >= 3 ? g.k->shape[g.k->rank - 2]
          : g.k->rank == 2 ? g.k->shape[1] / g.hd : 0;
    if (g.kvh <= 0) return RAD_E_SHAPE;
    g.qh = g.kvh * g.gqa;
    g.n_seq = numel(g.cu) - 1;
    if (g.n_seq <= 0 || (g.su && numel(g.su) < g.n_seq)) return RAD_E_SHAPE;
    if (cu_at(g.cu, g.n_seq) - cu_at(g.cu, 0) <= 0) return RAD_E_SHAPE;
    g.max_blocks = !g.bt ? 0 : g.bt->rank >= 2 ? g.bt->shape[1] : numel(g.bt) / g.n_seq;
    g.bt_rs = g.bt && g.bt->rank >= 2 ? g.bt->stride[0] : g.max_blocks;
    g.bt_cs = g.bt && g.bt->rank >= 2 ? g.bt->stride[1] : 1;
    for (int64_t s = 0; g.su && s < g.n_seq; ++s)
        if (ldi(g.su, offlin(g.su, s)) > g.max_blocks * g.bs) return RAD_E_SHAPE;
    g.q_rs = row_stride(g.q, g.qh * g.hd);
    g.k_rs = row_stride(g.k, g.kvh * g.hd);
    g.v_rs = row_stride(g.v, g.kvh * g.hd);
    g.o_rs = row_stride(g.o, g.qh * g.hd);
    g.k8_rs = g.i8 ? row_stride(g.k8, g.kvh * g.hd) : 0;
    g.ks_rs = g.i8 ? row_stride(g.ks, g.kvh) : 0;
    if (g.kv) { g.sb = g.kv->stride[0]; g.sh = g.kv->stride[1]; g.so = g.kv->stride[2]; }
    g.scale = p_f32(a, "scale", 1.0f / std::sqrt((float)g.hd));
    /* The fp8 P.V form (libref's ref_attn_prefix_self): prefix-free, its own two-pass loop. */
    {
        const RadTensor* vsc = a->n_t > 12 ? rad_arg_in(a, 12) : nullptr;
        const char* pv = p_str(a, "pv", "bf16");
        const bool pv8 = std::strcmp(pv, "fp8") == 0;
        if ((!pv8 && std::strcmp(pv, "bf16") != 0) || pv8 != (vsc != nullptr)) return RAD_E_INVAL;
        if (pv8) {
            if (g.kv || g.kdes || g.vdes) return RAD_E_UNSUPPORTED;
            g.rows_total = cu_at(g.cu, g.n_seq) - cu_at(g.cu, 0);
            return ps_pv8(g, vsc);
        }
    }

    /* ---- which form, and each (entry, kv head)'s key sequence */
    [[maybe_unused]] bool bf16 = false;
#if AVX_LEVEL == 3
    static const bool have_bf16 = __builtin_cpu_supports("avx512bf16");
    bf16 = have_bf16 && g.hd % 64 == 0 && g.gqa <= 16 && !g.kdes && !g.vdes &&
           g.q->dtype == RAD_BF16 && (!g.kv || g.kv->dtype == RAD_BF16) &&
           g.v->dtype == RAD_BF16 &&
           (g.i8 || g.k->dtype == RAD_BF16);
#endif
    const size_t tile_bytes =
#if AVX_LEVEL == 3
        bf16 ? bf16_tile_bytes(g.hd) :
#endif
        f32_tile_floats(g.hd) * sizeof(float);
    /* Kept at their high-water size; the parallel regions read them through pointers taken on
     * this thread. */
    static thread_local std::vector<Seq> seqs;
    static thread_local std::vector<uint8_t> pack;
    static thread_local std::vector<int64_t> tile_of, task_of;
    seqs.clear();
    for (int64_t kh = 0; kh < g.kvh; ++kh)
        for (int64_t s = 0; s < g.n_seq; ++s) {
            Seq q{};
            q.s = s; q.kh = kh;
            q.P = g.su ? ldi(g.su, offlin(g.su, s)) : 0;
            q.b0 = cu_at(g.cu, s);
            q.len = std::max<int64_t>(0, cu_at(g.cu, s + 1) - q.b0);
            /* A negative prefix count makes key j own row b0 + j - P, as libref indexes it. */
            q.nk = std::max<int64_t>(0, q.P + q.len);
            q.ntile = (q.nk + KT - 1) / KT;
            seqs.push_back(q);
        }

    /* ---- kv heads in groups whose packs fit the budget: pack each group, then run its tasks */
    size_t i0 = 0;
    while (i0 < seqs.size()) {
        size_t i1 = i0, bytes = 0;
        do {
            /* Whole kv heads: an entry's tasks of one head share nothing with another head's. */
            const int64_t kh = seqs[i1].kh;
            size_t hb = 0, j = i1;
            for (; j < seqs.size() && seqs[j].kh == kh; ++j) hb += (size_t)seqs[j].ntile * tile_bytes;
            if (i1 > i0 && bytes + hb > (size_t)kPackBudget) break;
            for (; i1 < j; ++i1) {
                seqs[i1].off = bytes;
                bytes += (size_t)seqs[i1].ntile * tile_bytes;
            }
        } while (i1 < seqs.size());
        pack.resize(bytes + 64);
        uint8_t* pk = pack.data();
        tile_of.assign(1, 0);
        task_of.assign(1, 0);
        for (size_t i = i0; i < i1; ++i) {
            tile_of.push_back(tile_of.back() + seqs[i].ntile);
            task_of.push_back(task_of.back() + (seqs[i].len + QB - 1) / QB);
        }
        const int64_t n_tiles = tile_of.back(), n_tasks = task_of.back();
        const int64_t n_grp = (int64_t)(i1 - i0);
        const Seq* sq0 = seqs.data() + i0;
        const int64_t* t_of = tile_of.data();
        const int64_t* k_of = task_of.data();

        AVX_PARALLEL_FOR_IF(n_tiles * KT * g.hd >= AVX_PAR_MIN)
        for (int64_t u = 0; u < n_tiles; ++u) {
            const int64_t i = (int64_t)(std::upper_bound(t_of, t_of + n_grp + 1, u) - t_of) - 1;
            const Seq& q = sq0[i];
            const int64_t t = u - t_of[i];
            uint8_t* tile = pk + q.off + (size_t)t * tile_bytes;
            float* row = scratch_f32((size_t)(4 * g.hd + 64));
#if AVX_LEVEL == 3
            if (bf16) {
                pack_tile_bf16(g, q, t, tile, row, (uint16_t*)(row + g.hd + 16));
                continue;
            }
#endif
            pack_tile_f32(g, q, t, (float*)tile, row);
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(avx_team_size()) \
    if(n_tasks > 1 || avx_par_forced())
#endif
        for (int64_t u = 0; u < n_tasks; ++u) {
            const int64_t i = (int64_t)(std::upper_bound(k_of, k_of + n_grp + 1, u) - k_of) - 1;
            const Seq& q = sq0[i];
            const int64_t q0 = (u - k_of[i]) * QB, q1 = std::min(q0 + QB, q.len);
#if AVX_LEVEL == 3
            if (bf16) {
                task_bf16(g, q, pk, q0, q1);
                continue;
            }
#endif
            task_f32(g, q, pk, q0, q1);
        }
        i0 = i1;
    }
    return RAD_OK;
}
