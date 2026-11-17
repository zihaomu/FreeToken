/* ref_audio.cpp -- the audio generator's ops (docs/OPS.md, "Audio generation").
 *
 * An audio generator writes a piece frame by frame with a language model and a depth decoder over
 * residual codebooks, sampled with guidance between branches; a denoiser turns runs of frames
 * (windows) into latents; a vocoder of 1-D convolutions plays them. These are the ops that are not
 * already a language model's or an image generator's.
 *
 * Rounding follows the reference models, which run in bf16 and round every tensor they
 * materialise; where an op stands for several reference tensors, the libref function says which
 * roundings it keeps. Per-pass extents (a window's latents, a sequence's rows) come from the
 * operands; parameters are fixed at declare.
 */
#include "ref_common.h"
#include "ref_ops.h"
#include "rad_philox.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ref;

namespace {

/* A window entry's flags (RadAudioGenBatch, RAD_AGEN_F_*), restated because a kernel library
 * compiles against rad_abi.h, not the run phase's header. */
enum { kFirst = 1, kLast = 8, kJoin = 16 };

inline float rbf(float v) { return round_dt(RAD_BF16, v); }

/* The Philox stream an audio window's noise is drawn from, and the one a frame's picks are. */
inline uint64_t window_stream(int64_t k) { return 1 + (uint64_t)k; }
inline uint64_t frame_stream(int64_t f) { return ((uint64_t)1 << 32) + (uint64_t)f; }

inline uint64_t seed_at(const RadTensor* s, int64_t i) {
    return (uint64_t)(uint32_t)ld_int(s->dtype, s->data, offlin(s, 2 * i)) |
           ((uint64_t)(uint32_t)ld_int(s->dtype, s->data, offlin(s, 2 * i + 1)) << 32);
}

inline int64_t ldi(const RadTensor* t, int64_t i) {
    return ld_int(t->dtype, t->data, offlin(t, i));
}

}  /* namespace */

/* ------------------------------------------------------------------ conv1d */
int ref_elu(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    if (numel(y) <= 0 || numel(x) < numel(y)) return RAD_E_SHAPE;
    REF_PAR_FOR(numel(y))
    for (int64_t i = 0; i < numel(y); ++i) {
        const float v = ldt(x, offlin(x, i));
        stt(y, offlin(y, i), v >= 0 ? v : std::expm1(v));
    }
    return RAD_OK;
}

/* Mimi's Euclidean RVQ, not the cosine quantiser used by drama-3. Projections belong to
 * the enclosing split quantiser; each step subtracts its centroid in the projected space. */
int ref_vq_euclidean(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *cb = &a->t[1], *codes = &a->t[2];
    const int64_t n = p_int(a, "n", 0), bins = p_int(a, "bins", 0);
    if (n <= 0 || bins <= 0 || numel(cb) != bins * n || codes->dtype != RAD_I32)
        return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows < 1 || numel(codes) < rows) return RAD_E_SHAPE;
    REF_PAR_FOR(rows * bins * n)
    for (int64_t r = 0; r < rows; ++r) {
        float best = INFINITY;
        int64_t pick = 0;
        for (int64_t k = 0; k < bins; ++k) {
            float d = 0;
            for (int64_t j = 0; j < n; ++j) {
                const float delta = ldt(x, rowoff(x, r, n) + j * laststride(x)) -
                                    ldt(cb, rowoff(cb, k, n) + j * laststride(cb));
                d += delta * delta;
            }
            if (d < best) { best = d; pick = k; }
        }
        st_int(codes->dtype, codes->data, offlin(codes, r), pick);
        for (int64_t j = 0; j < n; ++j) {
            const int64_t at = rowoff(x, r, n) + j * laststride(x);
            stt(x, at, ldt(x, at) - ldt(cb, rowoff(cb, pick, n) + j * laststride(cb)));
        }
    }
    return RAD_OK;
}

/* A 1-D convolution over sequences of rows, channels innermost: x [rows, cin], y [rows, cout],
 * `cu_seqlens` bounding each sequence (row indices of both operands, as every sequenced op reads
 * them), outside of which the input reads zero (each sequence is padded on its own). w is
 * PyTorch's Conv1d weight [cout, cin, k], b [cout], `dil` the dilation and the padding
 * dil (k - 1) / 2 on each side (k odd), so a sequence keeps its length; with `causal` 1 all of it,
 * dil (k - 1), on the left (any k). A `stride` s > 1 comes in two forms. The causal one (a reach
 * of at least s) keeps every s-th output row: the left pad is the reach less s, and a sequence of
 * L rows (every bound a whole number of strides) gives L / s; `cu_div` reads the bounds in units
 * that many input rows apart. With an explicit `pad` (bf16 only) it is PyTorch's zero-padded
 * strided conv over one sequence of all of x, any k: output row m reads
 * x[m * stride + tap * dil - pad], and y has exactly (rows + 2 pad - dil (k - 1) - 1) / stride + 1
 * rows. `res`, when bound, is added after the bias; then `act` (none, tanh, or clamp to [-1, 1],
 * a vocoder's output clamp). f32 accumulation, one rounding. */
int ref_conv1d(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *cu = &a->t[2];
    const RadTensor *res = t_in(a, 3), *y = &a->t[4], *b = t_in(a, 5);
    if (!y->data) return RAD_E_INVAL;
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0), k = p_int(a, "k", 0);
    const int64_t dil = p_int(a, "dil", 1), causal = p_int(a, "causal", 0);
    const int64_t st = p_int(a, "stride", 1), pp = p_int(a, "pad", -1), cd = p_int(a, "cu_div", 1);
    const char* act = p_str(a, "act", "none");
    const bool tanh_out = act && !std::strcmp(act, "tanh");
    const bool clamp_out = act && !std::strcmp(act, "clamp");
    if (!tanh_out && !clamp_out && act && std::strcmp(act, "none")) return RAD_E_INVAL;
    /* The causal strided form or the padded one, never both. */
    const bool padded = pp >= 0 && st > 1;
    if (cin <= 0 || cout <= 0 || k <= 0 || (!causal && st <= 1 && (k & 1) == 0) || dil <= 0 ||
        causal < 0 || causal > 1 || st < 1 || (pp >= 0 && (st <= 1 || causal || cd != 1)) ||
        (st > 1 && pp < 0 && (!causal || dil * (k - 1) + 1 < st)) || cd < 1)
        return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    const int64_t cdv = padded ? 1 : cd;
    /* A stride takes each sequence of L rows to L / stride, so every bound is a whole number of
     * strides (of cu_div units each); the padded form bounds only its one full sequence. */
    for (int64_t e = 0; !padded && e <= n_seq; ++e)
        if (cu_at(cu, e) % (cd * st)) return RAD_E_SHAPE;
    const int64_t r0 = cu_at(cu, 0) / cdv, r1 = cu_at(cu, n_seq) / cdv;
    if (r0 < 0 || r1 <= r0 || rows_of(x, cin) < r1) return RAD_E_SHAPE;
    if (numel(w) < cout * cin * k || (b && numel(b) < cout)) return RAD_E_SHAPE;
    int64_t rmin = r0 / st, rmax = r1 / st;
    if (padded) {
        if (x->dtype != RAD_BF16) return RAD_E_DTYPE;
        if (n_seq != 1 || r0 != 0 || r1 != rows_of(x, cin)) return RAD_E_SHAPE;
        rmin = 0;
        rmax = (r1 + 2 * pp - dil * (k - 1) - 1) / st + 1;
    }
    if (rmin >= rmax) return RAD_E_SHAPE;
    if (padded ? rows_of(y, cout) != rmax : rows_of(y, cout) < rmax) return RAD_E_SHAPE;
    if (res && rows_of(res, cout) < rmax) return RAD_E_SHAPE;
    /* Causal: the reach is dil (k - 1) + 1 input rows and the left pad that less the stride, so
     * output row o of a sequence reads input rows o stride - pad .. o stride + stride - 1. The
     * padded form takes PyTorch's zero padding from the explicit `pad` instead. */
    const int64_t pad = padded ? pp : (causal ? dil * (k - 1) + 1 - st : dil * (k - 1) / 2);
    REF_PAR_FOR((rmax - rmin) * cout * cin * k)
    for (int64_t r = rmin; r < rmax; ++r) {
        const int64_t e = seq_of(cu, n_seq, r * st * cdv);
        const int64_t s0 = cu_at(cu, e) / cdv, s1 = cu_at(cu, e + 1) / cdv;
        const int64_t yb = rowoff(y, r, cout), ys = laststride(y);
        for (int64_t co = 0; co < cout; ++co) {
            float acc = 0.0f;
            for (int64_t kk = 0; kk < k; ++kk) {
                const int64_t ir = r * st + kk * dil - pad;
                if (ir < s0 || ir >= s1) continue;
                const int64_t xb = rowoff(x, ir, cin), xs = laststride(x);
                for (int64_t ci = 0; ci < cin; ++ci)
                    acc += ldt(w, (co * cin + ci) * k + kk) * ldt(x, xb + ci * xs);
            }
            if (b) acc += ldt(b, co);
            if (res) acc += ldt(res, rowoff(res, r, cout) + co * laststride(res));
            if (tanh_out) acc = ref_tanh(acc);
            if (clamp_out) acc = acc < -1.0f ? -1.0f : (acc > 1.0f ? 1.0f : acc);
            stt(y, yb + co * ys, acc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv_t1d */
/* A transposed 1-D convolution of kernel `k` >= stride (2 stride by default) and padding `pad`
 * ((k - stride) / 2 by default, which needs k - stride even; 0 is the causal form, the full
 * output's last k - stride rows dropped), so each sequence grows exactly `stride` times: x
 * [rows, cin] in sequences `cu_seqlens`, y [rows * stride, cout] in the same sequences scaled. w
 * is PyTorch's ConvTranspose1d weight [cin, cout, k], b [cout]; sequence e is input rows
 * [cu[e], cu[e+1]) and output rows [stride cu[e], stride cu[e+1]). Output row o of a sequence
 * reads every input row i of the same sequence with 0 <= (o + pad) - i stride < k: with
 * q = (o + pad) / stride and r = (o + pad) % stride, the taps t = 0, 1, .. while r + t stride < k,
 * y[o] = b + sum_t x[q - t] w[:, :, r + t stride], in that order. f32 accumulation, one
 * rounding. */
int ref_conv_t1d(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *b = &a->t[2], *cu = &a->t[3], *y = &a->t[4];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0), s = p_int(a, "stride", 0);
    /* Without `pad`, the symmetric (k - stride) / 2, which needs k - stride even. */
    const int64_t K = p_int(a, "k", 2 * s), pad = p_int(a, "pad", -1);
    const int64_t P = pad >= 0 ? pad : (K - s) / 2;
    if (cin <= 0 || cout <= 0 || s <= 0 || K < s || (pad < 0 && ((K - s) & 1))) return RAD_E_SHAPE;
    const int64_t taps = (K + s - 1) / s;
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    const int64_t r0 = cu_at(cu, 0), r1 = cu_at(cu, n_seq);
    if (r0 < 0 || r1 <= r0 || rows_of(x, cin) < r1 || rows_of(y, cout) < r1 * s) return RAD_E_SHAPE;
    if (numel(w) < cin * cout * K || numel(b) < cout) return RAD_E_SHAPE;
    REF_PAR_FOR((r1 - r0) * s * cout * cin * taps)
    for (int64_t o = r0 * s; o < r1 * s; ++o) {
        const int64_t e = seq_of(cu, n_seq, o / s);
        const int64_t i0 = cu_at(cu, e), i1 = cu_at(cu, e + 1);
        const int64_t lo = o - i0 * s;   /* the output row within its sequence */
        const int64_t q = (lo + P) / s, rr = (lo + P) % s;
        const int64_t yb = rowoff(y, o, cout), ys = laststride(y);
        for (int64_t co = 0; co < cout; ++co) {
            float acc = ldt(b, co);
            for (int64_t t = 0; rr + t * s < K; ++t) {
                const int64_t qi = q - t, tap = rr + t * s;
                if (qi < 0 || i0 + qi >= i1) continue;
                const int64_t xb = rowoff(x, i0 + qi, cin), xs = laststride(x);
                for (int64_t ci = 0; ci < cin; ++ci)
                    acc += ldt(w, (ci * cout + co) * K + tap) * ldt(x, xb + ci * xs);
            }
            stt(y, yb + co * ys, acc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ snake */
/* Snake, per channel: y = x + sin(alpha x)^2 / (alpha + 1e-9), alpha [n]. With `beta` [n], the
 * SnakeBeta form y = x + (1 / (beta + 1e-9)) sin(alpha x)^2. With `log` 1 both are stored as their
 * logarithms and exponentiated first, as the reference does each pass. f32, one rounding. */
int ref_snake(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *al = &a->t[1], *y = &a->t[2], *be = t_in(a, 3);
    const int64_t n = p_int(a, "n", 0), lg = p_int(a, "log", 0);
    if (n <= 0 || numel(al) < n || (be && numel(be) < n) || lg < 0 || lg > 1) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || rows_of(x, n) < rows) return RAD_E_SHAPE;
    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t c = 0; c < n; ++c) {
            const float v = ldt(x, xb + c * xs);
            const float al_c = lg ? ref_exp(ldt(al, c)) : ldt(al, c);
            const float sn = ref_sin(v * al_c);
            if (be) {
                const float b_c = lg ? ref_exp(ldt(be, c)) : ldt(be, c);
                stt(y, yb + c * ys, v + 1.0f / (b_c + 1e-9f) * (sn * sn));
            } else {
                stt(y, yb + c * ys, v + sn * sn / (al_c + 1e-9f));
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ snake_aa */
/* BigVGAN's alias-free SnakeBeta over each sequence (diffusers' MiniMaxH3AudioActivation1d:
 * UpSample1d, SnakeBeta, DownSample1d), rows samples and channels innermost, all in f32. Per
 * channel c of a sequence of L rows:
 *
 *   xp   the sequence replicate-padded by P = k / ratio - 1 rows on each side
 *   u    ratio * conv_transpose1d(xp, f_up, stride ratio), cropped by P ratio + (k - ratio) / 2
 *        rows at the start and P ratio + (k - ratio + 1) / 2 at the end, so L ratio rows remain:
 *        u_full[o] = sum over input rows i with 0 <= o - i ratio < k of f_up[o - i ratio] xp[i],
 *        in ascending i
 *   a    u + (1 / (B + 1e-9)) sin(A u)^2, A = exp(alpha[c]) and B = exp(beta[c]) with `log` 1
 *        (alpha[c] and beta[c] themselves otherwise), as `snake` evaluates it
 *   y[o] sum over j < k of f_down[j] ap[o ratio + j] in ascending j, ap being a replicate-padded
 *        by k / 2 rows on the right and k / 2 on the left less one for an even k
 *
 * so a sequence keeps its length. f_up and f_down are the checkpoint's [1, 1, k] Kaiser filters,
 * read for every channel. */
int ref_snake_aa(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *al = &a->t[1], *be = &a->t[2], *fu = &a->t[3];
    const RadTensor *fd = &a->t[4], *cu = &a->t[5], *y = &a->t[6];
    const int64_t n = p_int(a, "n", 0), R = p_int(a, "ratio", 0), k = p_int(a, "k", 0);
    const int64_t lg = p_int(a, "log", 0);
    if (n <= 0 || R <= 0 || k < R || lg < 0 || lg > 1) return RAD_E_SHAPE;
    if (numel(al) < n || numel(be) < n || numel(fu) < k || numel(fd) < k) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    const int64_t r0 = cu_at(cu, 0), r1 = cu_at(cu, n_seq);
    if (r0 < 0 || r1 <= r0 || rows_of(x, n) < r1 || rows_of(y, n) < r1) return RAD_E_SHAPE;
    const int64_t P = k / R - 1, crop = P * R + (k - R) / 2, pl = k / 2 - (k % 2 == 0 ? 1 : 0);
    std::vector<float> fup((size_t)k), fdn((size_t)k);
    for (int64_t j = 0; j < k; ++j) {
        fup[(size_t)j] = ldt(fu, offlin(fu, j));
        fdn[(size_t)j] = ldt(fd, offlin(fd, j));
    }
    REF_PAR_FOR_DYN((r1 - r0) * n * k * 4)
    for (int64_t ec = 0; ec < n_seq * n; ++ec) {
        const int64_t e = ec / n, c = ec % n;
        const int64_t s0 = cu_at(cu, e), L = cu_at(cu, e + 1) - s0;
        if (L <= 0) continue;
        const float A = lg ? ref_exp(ldt(al, c)) : ldt(al, c);
        const float B = lg ? ref_exp(ldt(be, c)) : ldt(be, c);
        /* Row i of the padded sequence. */
        auto xp = [&](int64_t i) {
            const int64_t q = std::min(std::max<int64_t>(i - P, 0), L - 1);
            return ldt(x, rowoff(x, s0 + q, n) + c * laststride(x));
        };
        std::vector<float> act((size_t)(L * R));
        for (int64_t m = 0; m < L * R; ++m) {
            const int64_t o = crop + m;
            const int64_t i_lo = o < k ? 0 : (o - k) / R + 1;
            const int64_t i_hi = std::min(o / R, L + 2 * P - 1);
            float acc = 0.0f;
            for (int64_t i = i_lo; i <= i_hi; ++i) acc += fup[(size_t)(o - i * R)] * xp(i);
            const float u = (float)R * acc;
            const float sn = ref_sin(u * A);
            act[(size_t)m] = u + 1.0f / (B + 1e-9f) * (sn * sn);
        }
        for (int64_t o = 0; o < L; ++o) {
            float acc = 0.0f;
            for (int64_t j = 0; j < k; ++j) {
                const int64_t m = std::min(std::max<int64_t>(o * R + j - pl, 0), L * R - 1);
                acc += fdn[(size_t)j] * act[(size_t)m];
            }
            stt(y, rowoff(y, s0 + o, n) + c * laststride(y), acc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ cfg_topk_sample */
/* One pick per lane from `rep` branch rows of logits (rows rep i .. rep i + rep - 1 are lane i's
 * branches, the conditional first), with guidance: g = u + cfg (c - u) between the first two
 * branches, or c alone when rep is 1; `cfg` [n] f32, when bound, is each lane's guidance in place
 * of the parameter. With `penalty` (the repetition penalty) and a lane's history `hist`
 * [lanes, n_vocab] i32 (`lane` [n] naming the row), every id the history holds has its g divided
 * by the penalty when positive and multiplied by it otherwise; frame 0 clears the lane's row
 * first, and the pick is recorded in it. Ids in [sup_lo, sup_hi) are never candidates. With
 * `cond_k` > 0 only the entries where c reaches its cond_k-th largest value are. Of the candidates
 * the `k` largest g (ties to the lower index) are kept, divided by `temp`, softmaxed in f32 and
 * drawn by inverse CDF in descending g with u = rad_philox_uniform_s(seed, 2^32 + frame, stream).
 * With `top_p` < 1 the kept list is cut first where the cumulative probability of the whole
 * candidate set's softmax over temp (summed in index order) passes top_p, as Fish Audio's sampler
 * cuts it: a rank whose inclusive cumulative probability exceeds top_p is dropped, rank 0 never.
 * The pick is written to all of the lane's `rep` slots of `ids`. logits [rep n, n_vocab] f32 or
 * bf16; seed [n, 2]; frame [n]. */
int ref_cfg_topk_sample(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *seed = &a->t[1], *fr = &a->t[2], *ids = &a->t[3];
    const RadTensor *cfgs = t_in(a, 4), *lane = t_in(a, 5), *hist = t_in(a, 6);
    const int64_t V = p_int(a, "n_vocab", 0), K = p_int(a, "k", 0), rep = p_int(a, "rep", 1);
    const int64_t ck = p_int(a, "cond_k", 0), stream = p_int(a, "stream", 0);
    const int64_t sup_lo = p_int(a, "sup_lo", 0), sup_hi = p_int(a, "sup_hi", 0);
    const float cfg = p_f32(a, "cfg", 1.0f), temp = p_f32(a, "temp", 1.0f);
    const float pen = p_f32(a, "penalty", 1.0f);
    const double top_p = (double)p_f32(a, "top_p", 1.0f);
    if (V <= 0 || K <= 0 || K > V || rep < 1 || rep > 2 || ck < 0 || ck > V || !(temp > 0) ||
        sup_lo < 0 || sup_hi < sup_lo || sup_hi > V || !(pen > 0) || (!hist != !lane) ||
        !(top_p >= 0))
        return RAD_E_SHAPE;
    const int64_t n = numel(ids) / rep;
    if (n <= 0 || rows_of(lg, V) < rep * n || numel(fr) < n || numel(seed) < 2 * n ||
        (cfgs && numel(cfgs) < n) || (lane && numel(lane) < n))
        return RAD_E_SHAPE;
    std::vector<float> g((size_t)V), c((size_t)V);
    std::vector<int64_t> idx((size_t)V);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t cb = rowoff(lg, rep * i, V), cs = laststride(lg);
        for (int64_t j = 0; j < V; ++j) c[(size_t)j] = ldt(lg, cb + j * cs);
        if (rep == 2) {
            const float gi = cfgs ? ldt(cfgs, offlin(cfgs, i)) : cfg;
            const int64_t ub = rowoff(lg, rep * i + 1, V);
            for (int64_t j = 0; j < V; ++j) {
                const float u = ldt(lg, ub + j * cs);
                g[(size_t)j] = u + gi * (c[(size_t)j] - u);
            }
        } else {
            g = c;
        }
        /* The lane's history: cleared at its first frame, then each id it holds penalised. */
        int64_t hb = -1;
        if (hist) {
            const int64_t L = ldi(lane, i);
            if (L < 0 || (L + 1) * V > numel(hist)) return RAD_E_SHAPE;
            hb = L * V;
            if (ldi(fr, i) == 0)
                for (int64_t j = 0; j < V; ++j)
                    st_int(hist->dtype, hist->data, offlin(hist, hb + j), 0);
            for (int64_t j = 0; j < V; ++j)
                if (ldi(hist, hb + j) != 0) {
                    float& v = g[(size_t)j];
                    v = v > 0.0f ? v / pen : v * pen;
                }
        }
        /* The candidates: every entry outside the suppressed range, or those at or above the
         * conditional's ck-th largest. */
        int64_t nc = 0;
        auto open = [&](int64_t j) { return j < sup_lo || j >= sup_hi; };
        if (ck > 0) {
            std::vector<float> cc = c;
            std::nth_element(cc.begin(), cc.begin() + (ck - 1), cc.end(), std::greater<float>());
            const float thr = cc[(size_t)(ck - 1)];
            for (int64_t j = 0; j < V; ++j)
                if (c[(size_t)j] >= thr && open(j)) idx[(size_t)nc++] = j;
        } else {
            for (int64_t j = 0; j < V; ++j)
                if (open(j)) idx[(size_t)nc++] = j;
        }
        if (nc == 0) return RAD_E_SHAPE;
        /* The whole candidate set's normaliser, in index order (idx still is), for top_p. */
        double z = 0;
        if (top_p < 1.0) {
            float gm = g[(size_t)idx[0]];
            for (int64_t t = 1; t < nc; ++t) gm = std::max(gm, g[(size_t)idx[(size_t)t]]);
            const float m0 = gm / temp;
            for (int64_t t = 0; t < nc; ++t)
                z += std::exp((double)(g[(size_t)idx[(size_t)t]] / temp - m0));
        }
        int64_t kk = std::min(K, nc);
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.begin() + nc,
                          [&](int64_t p, int64_t q) {
                              return g[(size_t)p] > g[(size_t)q] ||
                                     (g[(size_t)p] == g[(size_t)q] && p < q);
                          });
        const float m = g[(size_t)idx[0]] / temp;
        if (top_p < 1.0) {
            double cum = 0;
            int64_t keep = 1;
            for (int64_t t = 0; t < kk; ++t) {
                cum += std::exp((double)(g[(size_t)idx[(size_t)t]] / temp - m)) / z;
                if (t > 0 && cum > top_p) break;
                keep = t + 1;
            }
            kk = keep;
        }
        double tot = 0;
        std::vector<double> e((size_t)kk);
        for (int64_t t = 0; t < kk; ++t) {
            e[(size_t)t] = std::exp((double)(g[(size_t)idx[(size_t)t]] / temp - m));
            tot += e[(size_t)t];
        }
        const double u = (double)rad_philox_uniform_s(seed_at(seed, i), frame_stream(ldi(fr, i)),
                                                      (uint64_t)stream) * tot;
        int64_t pick = idx[(size_t)(kk - 1)];
        double acc = 0;
        for (int64_t t = 0; t < kk; ++t) {
            acc += e[(size_t)t];
            if (u < acc) { pick = idx[(size_t)t]; break; }
        }
        for (int64_t r = 0; r < rep; ++r)
            st_int(ids->dtype, ids->data, offlin(ids, rep * i + r), pick);
        if (hist) st_int(hist->dtype, hist->data, offlin(hist, hb + pick), 1);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rvq_embed */
/* A frame's embedding from its residual codes: codes [n_codes, rows] (code-major, so each
 * codebook's picks are one dense row), t0 the first codebook's table read at code + off0, t1 the
 * other codebooks' stacked tables, codebook k (k >= 1) at code + (k - 1) vocab. As the reference
 * rounds its tensors: y = bf16(bf16(t0[.] + bf16(sum_k t1[.])) * scale); with `round` 1 the whole
 * sum is one f32 accumulation in codebook order, times scale, rounded once to y's dtype (a sum of
 * every codebook's embedding, as torch's sum over the codebook axis rounds). A first code outside
 * t0 leaves its row unwritten; a later code outside t1 adds nothing. */
int ref_rvq_embed(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *codes = &a->t[0], *t0 = &a->t[1], *t1 = &a->t[2], *y = &a->t[3];
    const int64_t D = p_int(a, "n_embd", 0), K = p_int(a, "n_codes", 0);
    const int64_t voc = p_int(a, "vocab", 0), off0 = p_int(a, "off0", 0);
    const float scale = p_f32(a, "scale", 1.0f);
    const int64_t round1 = p_int(a, "round", 0), clamp = p_int(a, "clamp", 0);
    if (D <= 0 || K < 1 || voc <= 0 || round1 < 0 || round1 > 1 || clamp < 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, D);
    if (rows <= 0 || numel(codes) < K * rows) return RAD_E_SHAPE;
    const int64_t r0 = rows_of(t0, D), r1 = rows_of(t1, D);
    /* `clamp`: a code above clamp - 1 reads row clamp - 1 (a negative one still reads nothing). */
    auto cl = [clamp](int64_t c) { return clamp > 0 && c > clamp - 1 ? clamp - 1 : c; };
    /* Code (k, r): a codebook's row at the plane's own pitch, a frame's code at its own stride. */
    const int64_t kp = codes->rank >= 2 ? codes->stride[codes->rank - 2] : rows;
    const int64_t rp = codes->rank >= 1 ? codes->stride[codes->rank - 1] : 1;
    REF_PAR_FOR(rows * D * K)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t c0 = cl(ld_int(codes->dtype, codes->data, r * rp)) + off0;
        if (c0 < 0 || c0 >= r0) continue;
        std::vector<float> acc((size_t)D, 0.0f);
        const int64_t b0 = rowoff(t0, c0, D), yb = rowoff(y, r, D);
        if (round1)
            for (int64_t d = 0; d < D; ++d) acc[(size_t)d] = ldt(t0, b0 + d * laststride(t0));
        for (int64_t k = 1; k < K; ++k) {
            const int64_t ck = cl(ld_int(codes->dtype, codes->data, k * kp + r * rp)) + (k - 1) * voc;
            if (ck < 0 || ck >= r1) continue;
            const int64_t tb = rowoff(t1, ck, D);
            for (int64_t d = 0; d < D; ++d) acc[(size_t)d] += ldt(t1, tb + d * laststride(t1));
        }
        for (int64_t d = 0; d < D; ++d) {
            const float s = round1 ? acc[(size_t)d]
                                   : rbf(ldt(t0, b0 + d * laststride(t0)) + rbf(acc[(size_t)d]));
            stt(y, yb + d * laststride(y), s * scale);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vq_encode */
/* One step of a residual vector quantiser with factorised, L2-normalised codes (DAC's
 * VectorQuantize, as Fish Audio's codec encodes): per row of x [rows, n], e = in_w x + in_b
 * (in_w [dim, n]); the code is the row of cb [bins, dim] of greatest cosine with e,
 * (e / max(|e|, 1e-12)) . (cb_c / max(|cb_c|, 1e-12)), the lower index on a tie; then
 * x -= out_w cb[code] + out_b (out_w [n, dim], the raw codebook row), rounded once into x. f32,
 * every sum in index order. A chain of these over a codec's codebooks is its residual quantiser,
 * each step quantising what the last left. */
int ref_vq_encode(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *iw = &a->t[1], *ib = &a->t[2], *cb = &a->t[3];
    const RadTensor *ow = &a->t[4], *ob = &a->t[5], *codes = &a->t[6];
    const int64_t n = p_int(a, "n", 0), dim = p_int(a, "dim", 0), bins = p_int(a, "bins", 0);
    if (n <= 0 || dim <= 0 || bins <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(codes) < rows || numel(iw) < dim * n || numel(ib) < dim ||
        numel(cb) < bins * dim || numel(ow) < n * dim || numel(ob) < n)
        return RAD_E_SHAPE;
    /* The codebook normalised once, as F.normalize divides. */
    std::vector<float> cn((size_t)(bins * dim));
    for (int64_t c = 0; c < bins; ++c) {
        float ss = 0.0f;
        for (int64_t j = 0; j < dim; ++j) ss += ldt(cb, c * dim + j) * ldt(cb, c * dim + j);
        const float nrm = std::max(std::sqrt(ss), 1e-12f);
        for (int64_t j = 0; j < dim; ++j) cn[(size_t)(c * dim + j)] = ldt(cb, c * dim + j) / nrm;
    }
    REF_PAR_FOR(rows * (dim * n + bins * dim + n * dim))
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        std::vector<float> e((size_t)dim);
        float ss = 0.0f;
        for (int64_t j = 0; j < dim; ++j) {
            float acc = 0.0f;
            for (int64_t d = 0; d < n; ++d) acc += ldt(iw, j * n + d) * ldt(x, xb + d * xs);
            e[(size_t)j] = acc + ldt(ib, j);
            ss += e[(size_t)j] * e[(size_t)j];
        }
        const float nrm = std::max(std::sqrt(ss), 1e-12f);
        for (int64_t j = 0; j < dim; ++j) e[(size_t)j] /= nrm;
        int64_t best = 0;
        float bs = -INFINITY;
        for (int64_t c = 0; c < bins; ++c) {
            float s = 0.0f;
            for (int64_t j = 0; j < dim; ++j) s += e[(size_t)j] * cn[(size_t)(c * dim + j)];
            if (s > bs) { bs = s; best = c; }
        }
        st_int(codes->dtype, codes->data, offlin(codes, r), best);
        for (int64_t d = 0; d < n; ++d) {
            float q = 0.0f;
            for (int64_t j = 0; j < dim; ++j) q += ldt(ow, d * dim + j) * ldt(cb, best * dim + j);
            stt(x, xb + d * xs, ldt(x, xb + d * xs) - (q + ldt(ob, d)));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ frame_mix_store */
/* A frame's conditioning row, stored into its lane: m = bf16(bf16(sum_l w_l h_l) * bf16(scale))
 * with w = bf16(softmax(bf16(logits))) over the `L` hidden planes of hid [L, rows, n_embd], row
 * rep i of each plane belonging to lane[i]'s frame[i] (a plane holds every branch's row, the
 * conditional first). Stored at dst[lane * stride + frame - 1] when frame >= 1 and
 * status[lane] >= 0 (frame 0 only moves past the prompt; a negative status ended the piece). */
int ref_frame_mix_store(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
    const RadTensor *hid = &a->t[0], *lw = &a->t[1], *sc = &a->t[2], *lane = &a->t[3];
    const RadTensor *fr = &a->t[4], *st = &a->t[5], *dst = &a->t[6];
    const int64_t D = p_int(a, "n_embd", 0), L = p_int(a, "L", 0), stride = p_int(a, "stride", 0);
    const int64_t rep = p_int(a, "rep", 1);
    if (D <= 0 || L <= 0 || stride <= 0 || rep < 1 || numel(lw) < L || numel(sc) < 1)
        return RAD_E_SHAPE;
    const int64_t n = numel(lane), plane = rows_of(hid, D) / L;
    if (n <= 0 || plane < rep * n || numel(fr) < n) return RAD_E_SHAPE;
    std::vector<float> wl((size_t)L);
    float mx = -INFINITY;
    for (int64_t l = 0; l < L; ++l) {
        wl[(size_t)l] = rbf(ldt(lw, l));
        mx = std::max(mx, wl[(size_t)l]);
    }
    double tot = 0;
    for (int64_t l = 0; l < L; ++l) tot += std::exp((double)(wl[(size_t)l] - mx));
    for (int64_t l = 0; l < L; ++l)
        wl[(size_t)l] = rbf((float)(std::exp((double)(wl[(size_t)l] - mx)) / tot));
    const float scale = rbf(ldt(sc, 0));
    const int64_t drows = rows_of(dst, D);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t ln = ldi(lane, i), f = ldi(fr, i);
        if (f < 1 || ln < 0 || ln >= numel(st) || ldi(st, ln) < 0) continue;
        const int64_t row = ln * stride + f - 1;
        if (f > stride || row >= drows) return RAD_E_SHAPE;
        const int64_t yb = rowoff(dst, row, D);
        for (int64_t d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int64_t l = 0; l < L; ++l)
                acc += wl[(size_t)l] *
                       ldt(hid, rowoff(hid, l * plane + rep * i, D) + d * laststride(hid));
            stt(dst, yb + d * laststride(dst), rbf(acc) * scale);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ras_pick */
/* Fish Audio's repetition-aware sampling, the choice between two draws a lane: the first draw
 * ids[rep i] = first[rep i], unless it lies in [lo, hi) and the lane's window (its last `win`
 * picks, `hist` [lanes, win], oldest first, -1 empty) holds it, in which case the alternative
 * draw alt[rep i] is taken. The pick then enters the window, its oldest entry leaving. `clear` 1
 * empties the lane's window first (a piece's first pick). A pick outside [lo, hi) (Fish's end of
 * speech) is never redrawn. */
int ref_ras_pick(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *first = &a->t[0], *alt = &a->t[1], *lane = &a->t[2], *hist = &a->t[3];
    const RadTensor* ids = &a->t[4];
    const int64_t W = p_int(a, "win", 0), lo = p_int(a, "lo", 0), hi = p_int(a, "hi", 0);
    const int64_t rep = p_int(a, "rep", 1), clear = p_int(a, "clear", 0);
    const int64_t n = numel(lane);
    if (W <= 0 || lo < 0 || hi < lo || rep < 1 || clear < 0 || clear > 1 || n <= 0 ||
        numel(first) < rep * n || numel(alt) < rep * n || numel(ids) < rep * n)
        return RAD_E_SHAPE;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t L = ldi(lane, i);
        if (L < 0 || (L + 1) * W > numel(hist)) return RAD_E_SHAPE;
    }
    for (int64_t i = 0; i < n; ++i) {
        const int64_t hb = ldi(lane, i) * W;
        if (clear)
            for (int64_t w = 0; w < W; ++w) st_int(hist->dtype, hist->data, offlin(hist, hb + w), -1);
        const int64_t f = ldi(first, rep * i);
        int64_t pick = f;
        if (f >= lo && f < hi)
            for (int64_t w = 0; w < W; ++w)
                if (ldi(hist, hb + w) == f) { pick = ldi(alt, rep * i); break; }
        for (int64_t r = 0; r < rep; ++r) st_int(ids->dtype, ids->data, offlin(ids, rep * i + r), pick);
        for (int64_t w = 0; w + 1 < W; ++w)
            st_int(hist->dtype, hist->data, offlin(hist, hb + w), ldi(hist, hb + w + 1));
        st_int(hist->dtype, hist->data, offlin(hist, hb + W - 1), pick);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ end_status */
/* A frame's status per lane from its first pick: status[lane[i]] = -1 when ids[rep i] is `end`
 * (the model ended the piece), else the pick. */
int ref_end_status(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *ids = &a->t[0], *lane = &a->t[1], *st = &a->t[2];
    const int64_t end = p_int(a, "end", -1), rep = p_int(a, "rep", 1);
    const int64_t n = numel(lane);
    if (n <= 0 || rep < 1 || numel(ids) < rep * n) return RAD_E_SHAPE;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t ln = ldi(lane, i), v = ldi(ids, rep * i);
        if (ln < 0 || ln >= numel(st)) return RAD_E_SHAPE;
        st_int(st->dtype, st->data, offlin(st, ln), v == end ? -1 : v);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ frame_store */
/* A frame's codes into its lane's history: dst[(lane[i] n_codes + k) stride + frame[i] - skip] =
 * codes[k, rep i] for every k, when frame[i] >= skip and status[lane[i]] >= 0. codes is code-major
 * [n_codes, rows] (a codebook's picks one row, each lane's `rep` times); dst [lanes * n_codes,
 * stride] holds each lane's codebooks as rows over its frames, so a run of frames is a block. */
int ref_frame_store(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *codes = &a->t[0], *lane = &a->t[1], *fr = &a->t[2], *st = &a->t[3];
    const RadTensor *dst = &a->t[4];
    const int64_t K = p_int(a, "n_codes", 0), stride = p_int(a, "stride", 0);
    const int64_t rep = p_int(a, "rep", 1), skip = p_int(a, "skip", 0);
    if (K <= 0 || stride <= 0 || rep < 1 || skip < 0) return RAD_E_SHAPE;
    const int64_t n = numel(lane);
    if (n <= 0 || numel(fr) < n || numel(codes) < K * rep * n) return RAD_E_SHAPE;
    const int64_t kp = codes->rank >= 2 ? codes->stride[codes->rank - 2] : rep * n;
    const int64_t rp = codes->rank >= 1 ? codes->stride[codes->rank - 1] : 1;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t L = ldi(lane, i), f = ldi(fr, i) - skip;
        if (L < 0 || L >= numel(st)) return RAD_E_SHAPE;
        if (f < 0 || ldi(st, L) < 0) continue;
        if (f >= stride || (L + 1) * K * stride > numel(dst)) return RAD_E_SHAPE;
        for (int64_t k = 0; k < K; ++k)
            st_int(dst->dtype, dst->data, offlin(dst, (L * K + k) * stride + f),
                   ld_int(codes->dtype, codes->data, k * kp + rep * i * rp));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ resample_nearest */
/* Rows resampled nearest-neighbour as torch's interpolate(mode="nearest") does, in f32: output row
 * j reads input row min(floor(j * (in / out)), in - 1), `in` and `out` the operands' row counts. */
int ref_resample_nearest(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t in = rows_of(x, n), out = rows_of(y, n);
    if (in <= 0 || out <= 0) return RAD_E_SHAPE;
    const float scale = (float)in / (float)out;
    for (int64_t j = 0; j < out; ++j) {
        const int64_t src = std::min<int64_t>((int64_t)std::floor((float)j * scale), in - 1);
        const int64_t xb = rowoff(x, src, n), yb = rowoff(y, j, n);
        for (int64_t c = 0; c < n; ++c)
            stt(y, yb + c * laststride(y), ldt(x, xb + c * laststride(x)));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ fourier_embed */
/* Random Fourier features of a time in [0, 1]: y = [cos(2 pi t w) | sin(2 pi t w)], w [n / 2]
 * (a trained projection). With `round` 1 every tensor the bf16 reference holds is rounded: t, then
 * bf16(2 pi t), then the product with w, then each cosine and sine. t is f32 [M]. */
int ref_fourier_embed(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *t = &a->t[0], *w = &a->t[1], *y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    const bool rnd = p_int(a, "round", 0) != 0;
    if (n <= 0 || (n & 1) || numel(w) < n / 2) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(t) < rows) return RAD_E_SHAPE;
    const float two_pi = 6.28318530717958647692f;
    for (int64_t r = 0; r < rows; ++r) {
        float tv = ldt(t, offlin(t, r));
        if (rnd) tv = rbf(rbf(tv) * two_pi);
        else tv *= two_pi;
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t i = 0; i < n / 2; ++i) {
            float ang = tv * ldt(w, i);
            if (rnd) ang = rbf(ang);
            stt(y, yb + i * ys, ref_cos(ang));
            stt(y, yb + (n / 2 + i) * ys, ref_sin(ang));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ win_input */
/* A denoiser's input rows for its window entries: entry e's latent rows (lat_cu) are
 * [lat | zeros | cond] from lane win_lane[e] of the latent state `lat` [lanes * stride, n_lat] and
 * of the window conditioning `cond` [lanes * stride, n_cond], `stride` rows a lane. An entry
 * that is an unconditional branch (pair -2) takes zeros for its conditioning.
 * y [rows, n_lat + n_zero + n_cond]. */
int ref_win_input(const RadArgs* a, RadStream) {
    if (!have(a, 6)) return RAD_E_INVAL;
    const RadTensor *lat = &a->t[0], *cond = &a->t[1], *lcu = &a->t[2], *wl = &a->t[3];
    const RadTensor *pair = &a->t[4], *y = &a->t[5];
    const int64_t C = p_int(a, "n_lat", 0), Z = p_int(a, "n_zero", 0), D = p_int(a, "n_cond", 0);
    const int64_t stride = p_int(a, "stride", 0), W = C + Z + D;
    if (C <= 0 || Z < 0 || D <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t S = numel(lcu) - 1;
    if (S <= 0 || numel(wl) < S || numel(pair) < S) return RAD_E_SHAPE;
    if (rows_of(y, W) < cu_at(lcu, S)) return RAD_E_SHAPE;
    for (int64_t e = 0; e < S; ++e) {
        const int64_t L = ldi(wl, e), neg = ldi(pair, e) == -2;
        const int64_t b0 = cu_at(lcu, e), len = cu_at(lcu, e + 1) - cu_at(lcu, e);
        if (L < 0 || len > stride || (L + 1) * stride > rows_of(lat, C) ||
            (len > 0 && (b0 < 0 || b0 + len > rows_of(y, W))) ||
            (len > 0 && !neg && (L + 1) * stride > rows_of(cond, D)))
            return RAD_E_SHAPE;
        for (int64_t k = 0; k < len; ++k) {
            const int64_t yb = rowoff(y, b0 + k, W), ys = laststride(y);
            const int64_t lb = rowoff(lat, L * stride + k, C);
            for (int64_t c = 0; c < C; ++c) stt(y, yb + c * ys, ldt(lat, lb + c * laststride(lat)));
            for (int64_t c = 0; c < Z; ++c) stt(y, yb + (C + c) * ys, 0.0f);
            const int64_t cb = rowoff(cond, L * stride + k, D);
            for (int64_t c = 0; c < D; ++c)
                stt(y, yb + (C + Z + c) * ys, neg ? 0.0f : ldt(cond, cb + c * laststride(cond)));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ win_prep */
/* A window step's start, for each entry that moves its lane (pair != -2):
 *   FIRST  the lane's latent rows become the window's noise, element (k, c) of
 *          rad_philox_normal_s(seed, 1 + win_index, k n + c) as bf16, and with JOIN the first `ov`
 *          rows are kept in `noise`;
 *   JOIN   the first min(ov, rows) rows are bound to the previous window's end:
 *          bf16(bf16(1 - bf16((1 - 1e-6) t)) noise + bf16(t prev)), t = bf16(t[e]).
 * `lat` [lanes * stride, n], `prev` and `noise` [lanes * ov, n]. */
int ref_win_prep(const RadArgs* a, RadStream) {
    if (!have(a, 10)) return RAD_E_INVAL;
    const RadTensor *fl = &a->t[0], *t = &a->t[1], *seed = &a->t[2], *widx = &a->t[3];
    const RadTensor *wl = &a->t[4], *pair = &a->t[5], *lcu = &a->t[6], *prev = &a->t[7];
    const RadTensor *noise = &a->t[8], *lat = &a->t[9];
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0), ov = p_int(a, "ov", 0);
    if (n <= 0 || stride <= 0 || ov < 0 || ov > stride) return RAD_E_SHAPE;
    const int64_t S = numel(lcu) - 1;
    if (S <= 0 || numel(fl) < S || numel(t) < S || numel(widx) < S || numel(wl) < S ||
        numel(pair) < S)
        return RAD_E_SHAPE;
    for (int64_t e = 0; e < S; ++e) {
        if (ldi(pair, e) == -2) continue;
        const int64_t f = ldi(fl, e), L = ldi(wl, e);
        const int64_t len = cu_at(lcu, e + 1) - cu_at(lcu, e);
        if (L < 0 || len > stride || (L + 1) * stride > rows_of(lat, n) ||
            (ov && (L + 1) * ov > rows_of(noise, n)) ||
            ((f & kFirst) && numel(seed) < 2 * (e + 1)) ||
            ((f & kJoin) && ov && len > 0 && (L + 1) * ov > rows_of(prev, n)))
            return RAD_E_SHAPE;
        if (f & kFirst) {
            const uint64_t sd = seed_at(seed, e), sm = window_stream(ldi(widx, e));
            for (int64_t k = 0; k < len; ++k)
                for (int64_t c = 0; c < n; ++c) {
                    const float v = rbf(rad_philox_normal_s(sd, sm, (uint64_t)(k * n + c)));
                    stt(lat, rowoff(lat, L * stride + k, n) + c * laststride(lat), v);
                    if ((f & kJoin) && k < ov)
                        stt(noise, rowoff(noise, L * ov + k, n) + c * laststride(noise), v);
                }
        }
        if (f & kJoin) {
            const float tb = rbf(ldt(t, offlin(t, e)));
            const float wn = rbf(1.0f - rbf((1.0f - 1e-6f) * tb));
            const int64_t m = std::min(ov, len);
            for (int64_t k = 0; k < m; ++k)
                for (int64_t c = 0; c < n; ++c) {
                    const float nz =
                        ldt(noise, rowoff(noise, L * ov + k, n) + c * laststride(noise));
                    const float pv = ldt(prev, rowoff(prev, L * ov + k, n) + c * laststride(prev));
                    stt(lat, rowoff(lat, L * stride + k, n) + c * laststride(lat),
                        rbf(rbf(wn * nz) + rbf(tb * pv)));
                }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ win_finish */
/* A window's end, for each entry that moves its lane and carries LAST: with JOIN its first
 * min(ov, rows) rows become the previous window's end exactly; then the rows
 * [max(0, rows - 2 ov), max(that, rows - ov)) of the latent and of the conditioning are kept as
 * the next window's `prev` and `prev_cond`. `lat` [lanes * stride, n], `cond` [lanes * stride,
 * n_cond], `prev` [lanes * ov, n], `prev_cond` [lanes * ov, n_cond]. */
int ref_win_finish(const RadArgs* a, RadStream) {
    if (!have(a, 8)) return RAD_E_INVAL;
    const RadTensor *fl = &a->t[0], *pair = &a->t[1], *wl = &a->t[2], *lcu = &a->t[3];
    const RadTensor *cond = &a->t[4], *lat = &a->t[5], *prev = &a->t[6], *pc = &a->t[7];
    const int64_t n = p_int(a, "n", 0), D = p_int(a, "n_cond", 0);
    const int64_t stride = p_int(a, "stride", 0), ov = p_int(a, "ov", 0);
    if (n <= 0 || D <= 0 || stride <= 0 || ov <= 0 || ov > stride) return RAD_E_SHAPE;
    const int64_t S = numel(lcu) - 1;
    if (S <= 0 || numel(fl) < S || numel(wl) < S || numel(pair) < S) return RAD_E_SHAPE;
    for (int64_t e = 0; e < S; ++e) {
        const int64_t f = ldi(fl, e);
        if (ldi(pair, e) == -2 || !(f & kLast)) continue;
        const int64_t L = ldi(wl, e), len = cu_at(lcu, e + 1) - cu_at(lcu, e);
        if (L < 0 || len > stride || (L + 1) * stride > rows_of(lat, n) ||
            (L + 1) * ov > rows_of(prev, n) || (L + 1) * ov > rows_of(pc, D) ||
            (len > ov && (L + 1) * stride > rows_of(cond, D)))
            return RAD_E_SHAPE;
        if (f & kJoin)
            for (int64_t k = 0; k < std::min(ov, len); ++k)
                for (int64_t c = 0; c < n; ++c)
                    stt(lat, rowoff(lat, L * stride + k, n) + c * laststride(lat),
                        ldt(prev, rowoff(prev, L * ov + k, n) + c * laststride(prev)));
        const int64_t s0 = std::max<int64_t>(0, len - 2 * ov);
        const int64_t s1 = std::max<int64_t>(s0, len - ov);
        for (int64_t k = s0; k < s1; ++k) {
            for (int64_t c = 0; c < n; ++c)
                stt(prev, rowoff(prev, L * ov + k - s0, n) + c * laststride(prev),
                    ldt(lat, rowoff(lat, L * stride + k, n) + c * laststride(lat)));
            for (int64_t c = 0; c < D; ++c)
                stt(pc, rowoff(pc, L * ov + k - s0, D) + c * laststride(pc),
                    ldt(cond, rowoff(cond, L * stride + k, D) + c * laststride(cond)));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ chan_split */
/* Channel groups to sequences: x [rows, parts * n] becomes y [parts * rows, n], y[p rows + r] =
 * x[r, p n .. p n + n): a stereo latent's two halves as two streams for a vocoder. */
int ref_chan_split(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), P = p_int(a, "parts", 0);
    if (n <= 0 || P <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n) / P;
    if (rows <= 0 || rows_of(x, P * n) < rows) return RAD_E_SHAPE;
    for (int64_t p = 0; p < P; ++p)
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t xb = rowoff(x, r, P * n), yb = rowoff(y, p * rows + r, n);
            for (int64_t c = 0; c < n; ++c)
                stt(y, yb + c * laststride(y), ldt(x, xb + (p * n + c) * laststride(x)));
        }
    return RAD_OK;
}

/* ------------------------------------------------------------------ interleave2 */
/* Two streams of samples interleaved, y [rows, 2] = (a[r], b[r]): a vocoder's left and right
 * outputs as a stereo buffer. */
int ref_interleave2(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x0 = &a->t[0], *x1 = &a->t[1], *y = &a->t[2];
    const int64_t rows = rows_of(y, 2);
    if (rows <= 0 || numel(x0) < rows || numel(x1) < rows) return RAD_E_SHAPE;
    for (int64_t r = 0; r < rows; ++r) {
        stt(y, rowoff(y, r, 2), ldt(x0, offlin(x0, r)));
        stt(y, rowoff(y, r, 2) + laststride(y), ldt(x1, offlin(x1, r)));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ iota */
/* y[i] = start + i * step over y's elements, or start + (i % period) * step with `period`: row
 * indices a model's fixed layout names (a depth decoder's position p in every sequence of its
 * frame), or the positions of equal sequences laid end to end. */
int ref_iota(const RadArgs* a, RadStream) {
    if (!have(a, 1)) return RAD_E_INVAL;
    const RadTensor* y = &a->t[0];
    const int64_t start = p_int(a, "start", 0), step = p_int(a, "step", 1);
    const int64_t period = p_int(a, "period", 0);
    const int64_t n = numel(y);
    if (n <= 0 || period < 0) return RAD_E_SHAPE;
    for (int64_t i = 0; i < n; ++i)
        st_int(y->dtype, y->data, offlin(y, i), start + (period ? i % period : i) * step);
    return RAD_OK;
}

/* ------------------------------------------------------------------ int_affine */
/* y[j] = (src[j] < 0 ? 0 : x[src[j]]) * mul[j] + add[j], the three lists space-separated integers
 * of one length: shapes and sequence bounds a model derives from a pass's device geometry (a
 * decoder's volume at each of its scales). y holds at least the lists' length. */
static int parse_ints(const char* s, int64_t* v, int cap) {
    int n = 0;
    while (s && *s) {
        char* e = nullptr;
        const long long x = std::strtoll(s, &e, 10);
        if (e == s) break;
        if (n == cap) return -1;
        v[n++] = x;
        s = e;
    }
    return n;
}

int ref_int_affine(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    int64_t src[16], mul[16], add[16];
    const int n = parse_ints(p_str(a, "src", ""), src, 16);
    if (n <= 0 || parse_ints(p_str(a, "mul", ""), mul, 16) != n ||
        parse_ints(p_str(a, "add", ""), add, 16) != n || numel(y) < n)
        return RAD_E_SHAPE;
    for (int j = 0; j < n; ++j) {
        if (src[j] >= numel(x)) return RAD_E_SHAPE;
        const int64_t v = src[j] < 0 ? 0 : ld_int(x->dtype, x->data, offlin(x, src[j]));
        st_int(y->dtype, y->data, offlin(y, j), v * mul[j] + add[j]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sub_i32 */
/* y[i] = x[i] - s[0], int32: a device length less a device offset. */
int ref_sub_i32(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *s = &a->t[1], *y = &a->t[2];
    const int64_t n = numel(y);
    if (n <= 0 || numel(x) < n || numel(s) < 1) return RAD_E_SHAPE;
    const int64_t s0 = ldi(s, 0);
    for (int64_t i = 0; i < n; ++i)
        st_int(y->dtype, y->data, offlin(y, i), (int64_t)(int32_t)(ldi(x, i) - s0));
    return RAD_OK;
}

/* ------------------------------------------------------------------ cu_repeat */
/* Sequence bounds of equal sequences: y[j] = j * len[0] * mult over y's elements. A pass's
 * lengths live on the device (a window's frames, the latents a vocoder decodes per channel group),
 * so a model's derived bounds are made there too. */
int ref_cu_repeat(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *len = &a->t[0], *y = &a->t[1];
    const int64_t mult = p_int(a, "mult", 1);
    const int64_t n = numel(y);
    if (n <= 0 || numel(len) < 1 || mult <= 0) return RAD_E_SHAPE;
    const int64_t l = ldi(len, 0);
    for (int64_t j = 0; j < n; ++j) st_int(y->dtype, y->data, offlin(y, j), j * l * mult);
    return RAD_OK;
}
