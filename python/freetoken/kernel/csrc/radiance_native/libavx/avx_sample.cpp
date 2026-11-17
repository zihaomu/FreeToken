/* avx_sample.cpp -- the sampler ops.
 *
 * Most of these work over a candidate set of at most about a thousand entries, where the cost is
 * control flow; they widen candidate values to f32 once and otherwise keep libref's structure. The
 * three that touch the whole vocabulary (a megabyte per row at 248K) are vectorised:
 *
 *   sample_temp     a row multiply.
 *   sample_argmax   maximum, lowest index on a tie, through the same `topk_next_f32` the routers
 *                   use, so greedy and top-1 agree on ties (a greedy request is staged at
 *                   top_k = 1 when lm_head is vocab-sharded).
 *   sample_mask     a vocabulary bitmask. At AVX-512 a grammar half-word is a `__mmask16`, so 16
 *                   tokens take one masked store of -inf; lower levels shift and test per token.
 *
 * Per-request settings (`rep`, `freq`, `pres`, `k`, `p`, seed, position) are fields of a params row
 * (abi/rad_sample.h), not op parameters: parameters are geometry fixed at declare, so they could
 * not differ between requests in one batch.
 *
 * Candidate set: `sample_topk` writes a fixed-width set and the narrowing stages edit it in place.
 * A dropped candidate is index -1, value -infinity; kept candidates stay in place, in descending
 * order. `cand_val` holds logits, and each stage takes its own softmax over the set (as llama.cpp
 * does), so stages compose in any order.
 *
 * A greedy row (RAD_SP_GREEDY) has no candidates: its token comes from sample_argmax, top-k and the
 * merge write it an empty set, and the narrowing stages and the pick skip it (libref's contract).
 */
#include "avx_vec.h"
#include "avx_common.h"

#include "rad_sample.h"

#include <cfloat>
#include <cmath>
#include <vector>

using namespace avx;

/* Validates the params operand. A row array is a byte blob declared as
 * [rows, sizeof(RadSampleParams)/4] u32 (the builder's only way to say "this many bytes a row"), so
 * the dtype must be a 32-bit one; a 16-bit view would pass a byte-count check and misread every
 * row. */
struct Rows { const RadSampleParams* p; int64_t n; };

static inline int rows_of_params(const RadTensor* t, int64_t M, Rows* out) {
    if (!t || !t->data) return RAD_E_INVAL;
    if (t->dtype != RAD_U32 && t->dtype != RAD_I32 && t->dtype != RAD_F32) return RAD_E_DTYPE;
    if (numel(t) * 4 < M * (int64_t)sizeof(RadSampleParams)) return RAD_E_SHAPE;
    out->p = (const RadSampleParams*)t->data;
    out->n = M;
    return RAD_OK;
}

#define AVX_TRY(expr) do { const int _s = (expr); if (_s < 0) return _s; } while (0)

/* History window for the penalty and DRY stages, as in libref (ref_sample.cpp): `hist_off` and
 * `hist_len` address this row of the history plane, clamped to it, and `last_n` keeps the tail of
 * that span. last_n 0 disables the stage for the row; negative keeps the whole span (llama.cpp's
 * repeat_last_n / dry_penalty_last_n). */
static inline bool hist_window(const RadSampleParams& sp, int32_t last_n, int64_t hmax,
                               int64_t* ho_out, int64_t* hl_out) {
    if (last_n == 0) return false;
    int64_t hl = sp.hist_len;
    int64_t ho = sp.hist_off < 0 ? 0 : sp.hist_off;
    if (ho > hmax) ho = hmax;
    if (hl > hmax - ho) hl = hmax - ho;
    if (last_n > 0 && hl > last_n) {
        ho += hl - last_n;
        hl = last_n;
    }
    *ho_out = ho;
    *hl_out = hl;
    return hl > 0;
}

/* ------------------------------------------------------------------ sample_penalties */
/* llama.cpp's ordering, in one pass over the row's history:
 *
 *     repetition   logit = logit > 0 ? logit / rep : logit * rep       (once per distinct token)
 *     frequency    logit -= count * freq
 *     presence     logit -= pres                                       (once per distinct token)
 *
 * The repetition penalty is sign-dependent: dividing a negative logit would raise it.
 *
 * The history is copied into an int array once, so libref's O(hl^2) first-occurrence scan
 * compares integers instead of switching on dtype; same scan, same result.
 *
 * History ids are global and the logits row is this rank's shard: a token's column is its id minus
 * `vocab_off`, and tokens outside the shard belong to another rank. */
AVX_KERNEL(sample_penalties) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *hist = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));
    const int64_t hmax = numel(hist) / M;
    if (hmax <= 0) return RAD_E_SHAPE;
    const int64_t voff = p_int(a, "vocab_off", 0);

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        if (!(sp.flags & RAD_SP_PENALTY)) continue;
        const float rep = sp.rep_penalty, freq = sp.freq_penalty, pres = sp.pres_penalty;
        if (rep == 1.0f && freq == 0.0f && pres == 0.0f) continue;

        int64_t ho = 0, hl = 0;
        if (!hist_window(sp, sp.penalty_last_n, hmax, &ho, &hl)) continue;

        const int64_t hb = rowoff(hist, m, hmax) + ho * laststride(hist);
        const int64_t hs = laststride(hist);
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        int32_t* h = (int32_t*)scratch_f32((size_t)hl);
        for (int64_t i = 0; i < hl; ++i) h[i] = (int32_t)ldi(hist, hb + i * hs);

        for (int64_t i = 0; i < hl; ++i) {
            const int32_t tok = h[i];
            const int64_t col = (int64_t)tok - voff;
            if (tok < 0 || col < 0 || col >= n_vocab) continue;
            /* First occurrence only; `count` carries the multiplicity. */
            bool first = true;
            int64_t count = 0;
            for (int64_t j = 0; j < hl; ++j) {
                if (h[j] != tok) continue;
                if (j < i) { first = false; break; }
                ++count;
            }
            if (!first) continue;
            float v = ldt(lg, lb + col * ls);
            v = v > 0.0f ? v / rep : v * rep;
            v -= (float)count * freq;
            v -= pres;
            stt(lg, lb + col * ls, v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_dry */
/* DRY ("don't repeat yourself"), as in libref (ref_sample.cpp) and llama.cpp: each token that
 * would extend a previously seen suffix of the window is penalised once, by
 * multiplier * base^(L - allowed_length) for the longest repeat L it extends, with repeats capped at
 * the row's `dry_rep_limit`. Positions marked in the optional `breakers` plane hold sequence
 * breakers and are never penalised. The penalty is built by repeated multiplication that stops
 * short of overflow, as other implementations do, so results agree to the bit.
 *
 * The window and the per-position repeat lengths are kept in two int arrays; the longest repeat
 * per token is found by scanning them, with no map. */
static inline float dry_penalty(float multiplier, float base, int64_t e) {
    float pen = multiplier;
    if (base == 1.0f) return pen;
    const float lim = FLT_MAX / base;
    for (int64_t i = 0; i < e; ++i) {
        if (pen > lim) break;
        pen *= base;
    }
    return pen;
}

AVX_KERNEL(sample_dry) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *hist = &a->t[2];
    const RadTensor* brk = rad_arg_in(a, 3);
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));
    const int64_t hmax = numel(hist) / M;
    if (hmax <= 0) return RAD_E_SHAPE;
    if (brk && numel(brk) < M * hmax) return RAD_E_SHAPE;
    const int64_t voff = p_int(a, "vocab_off", 0);

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        if (!(sp.flags & RAD_SP_DRY) || !(sp.dry_multiplier > 0.0f)) continue;
        const int64_t allowed = sp.dry_allowed_length > 0 ? sp.dry_allowed_length : 2;

        int64_t ho = 0, hl = 0;
        if (!hist_window(sp, sp.dry_penalty_last_n, hmax, &ho, &hl)) continue;
        if (hl <= allowed) continue;
        const int64_t cap = sp.dry_rep_limit > 0 && sp.dry_rep_limit < hl ? sp.dry_rep_limit : hl;
        if (cap < allowed) continue;

        const int64_t hb = rowoff(hist, m, hmax) + ho * laststride(hist);
        const int64_t hs = laststride(hist);
        const int64_t bb = brk ? rowoff(brk, m, hmax) + ho * laststride(brk) : 0;
        const int64_t bs = brk ? laststride(brk) : 0;
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        int32_t* h   = (int32_t*)scratch_f32((size_t)hl);
        int32_t* len = (int32_t*)scratch_f32_b((size_t)hl);
        for (int64_t i = 0; i < hl; ++i) h[i] = (int32_t)ldi(hist, hb + i * hs);

        /* len[j]: length of the window suffix that also ends at j, capped; 0 when the token
         * after j is a breaker or outside this rank's shard. */
        for (int64_t j = 0; j + 1 < hl; ++j) {
            const int64_t lim = j + 1 < cap ? j + 1 : cap;
            int64_t n = 0;
            while (n < lim && h[j - n] == h[hl - 1 - n]) ++n;
            const int64_t col = (int64_t)h[j + 1] - voff;
            const bool skip = (brk && ldi(brk, bb + (j + 1) * bs) != 0) || h[j + 1] < 0 ||
                              col < 0 || col >= n_vocab;
            len[j] = skip ? 0 : (int32_t)n;
        }

        const float base = sp.dry_base > 0.0f ? sp.dry_base : 1.75f;
        for (int64_t j = 0; j + 1 < hl; ++j) {
            if (len[j] < allowed) continue;
            /* Penalise each token once, at its longest repeat (earliest position on a tie). */
            const int32_t next = h[j + 1];
            bool rep = true;
            for (int64_t j2 = 0; j2 + 1 < hl && rep; ++j2) {
                if (j2 == j || h[j2 + 1] != next) continue;
                if (len[j2] > len[j] || (len[j2] == len[j] && j2 < j)) rep = false;
            }
            if (!rep) continue;
            const int64_t off = lb + ((int64_t)next - voff) * ls;
            stt(lg, off, ldt(lg, off) - dry_penalty(sp.dry_multiplier, base, len[j] - allowed));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_ngram_ban */
/* libref's ref_sample_ngram_ban: the windowed no-repeat-ngram ban. The row's history is
 * `s[0, hl)` oldest first; with `n` and `w` from the row's `ngram` pair,
 *
 *     search_start = max(0, hl - w)      (w <= 0: the whole span)
 *     search_end   = hl - n + 1
 *     prefix       = s[hl - n + 1, hl)   (empty at n == 1)
 *
 * and every n-gram in [search_start, search_end) whose first n-1 tokens equal the prefix bans the
 * token that follows it, by -infinity. The window bounds the n-gram's START, as the reference's
 * SlidingWindowNoRepeatNgramProcessor does. Ids are global; a token's column is its id less
 * `vocab_off`, and a token outside the shard is another rank's.
 *
 * The work is integer comparisons over at most a window of history, so the rows are parallel and
 * the scan inside a row stays scalar: libref's own loop, bit for bit. A row with n 0 (the
 * default) is skipped before it reads anything.
 *
 * The optional `whitelist` bitmask (a SET bit, the row's own plane row) exempts a token from
 * the ban, the reference's `banned.difference_update(self.whitelist)`; absent, nothing is
 * exempt, and the result is bit-for-bit the ban without it. */
AVX_KERNEL(sample_ngram_ban) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *hist = &a->t[2], *ng = &a->t[3];
    const RadTensor* wl = rad_arg_in(a, 4);
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));
    const int64_t hmax = numel(hist) / M;
    if (hmax <= 0) return RAD_E_SHAPE;
    if (numel(ng) < 2 * M) return RAD_E_SHAPE;
    const int64_t voff = p_int(a, "vocab_off", 0);
    const int64_t wwords = (n_vocab + 31) / 32;
    if (wl && wl->dtype != RAD_I32 && wl->dtype != RAD_U32) return RAD_E_DTYPE;
    if (wl && numel(wl) < M * wwords) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        const int64_t nb = rowoff(ng, m, 2), ns = laststride(ng);
        const int64_t n = ldi(ng, nb);
        const int64_t win_req = ldi(ng, nb + ns);
        if (n <= 0) continue;

        int64_t ho = 0, hl = 0;
        if (!hist_window(sp, -1, hmax, &ho, &hl)) continue;
        if (hl < n) continue;
        const int64_t win = win_req > 0 && win_req < hl ? win_req : hl;
        const int64_t search_start = hl - win;
        const int64_t search_end   = hl - n + 1;
        if (search_end <= search_start) continue;

        const int64_t hb = rowoff(hist, m, hmax) + ho * laststride(hist);
        const int64_t hs = laststride(hist);
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        const int64_t wb = wl ? rowoff(wl, m, wwords) : 0, ws = wl ? laststride(wl) : 0;
        auto tok_at = [&](int64_t i) { return (int64_t)(int32_t)ldi(hist, hb + i * hs); };
        auto exempt = [&](int64_t col) {
            if (!wl) return false;
            const uint32_t word = (uint32_t)ldi(wl, wb + (col / 32) * ws);
            return ((word >> (uint32_t)(col % 32)) & 1u) != 0u;
        };
        const int64_t plen = n - 1, p0 = hl - plen;
        /* The prefix is the same n-1 tokens compared at every search position: read once. */
        std::vector<int64_t> pref((size_t)plen);
        for (int64_t j = 0; j < plen; ++j) pref[(size_t)j] = tok_at(p0 + j);

        for (int64_t idx = search_start; idx < search_end; ++idx) {
            bool match = true;
            for (int64_t j = 0; j < plen; ++j) {
                if (tok_at(idx + j) != pref[(size_t)j]) { match = false; break; }
            }
            if (!match) continue;
            const int64_t tok = tok_at(idx + plen);
            const int64_t col = tok - voff;
            if (tok < 0 || col < 0 || col >= n_vocab) continue;
            if (exempt(col)) continue;
            stt(lg, lb + col * ls, -INFINITY);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_temp */
/* temp <= 0 leaves the row untouched; greedy decoding is `sample_argmax`. */
AVX_KERNEL(sample_temp) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));

    AVX_PARALLEL_FOR_IF(M * n_vocab >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const float t = R.p[m].temp;
        if (!(t > 0.0f) || t == 1.0f) continue;
        const float inv = 1.0f / t;
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        if (ls != 1) {
            for (int64_t i = 0; i < n_vocab; ++i) stt(lg, lb + i * ls, ldt(lg, lb + i * ls) * inv);
            continue;
        }
        float* buf = scratch_f32((size_t)n_vocab);
        const float* v = in_row(lg, lb, n_vocab, buf);
        float* out = (v == buf) ? buf : scratch_f32_b((size_t)n_vocab);
        const vf vi = vf_set1(inv);
        int64_t i = 0;
        for (; i + VF_N <= n_vocab; i += VF_N) vf_storeu(out + i, vf_mul(vf_loadu(v + i), vi));
        for (; i < n_vocab; ++i) out[i] = v[i] * inv;
        out_row(lg, lb, n_vocab, out);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_topk */
AVX_KERNEL(sample_topk) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *ci = &a->t[2], *cv = &a->t[3];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));
    /* The candidate width is the buffer's (the deployment bound), not the request's top_k. */
    const int64_t n_cand = ci->rank >= 2 ? ci->shape[ci->rank - 1] : numel(ci) / M;
    if (n_cand <= 0 || numel(cv) < M * n_cand) return RAD_E_SHAPE;

    /* Optional vocab-parallel pair plane. cand_idx stays local (it indexes this rank's logits);
     * `pairs` holds the same candidates as (u32 global id = local + vocab_off, f32 logit bits),
     * which the all-gather moves and sample_merge_topk reads. */
    const RadTensor* pp = a->n_t > 4 && a->t[4].data ? &a->t[4] : nullptr;
    const int64_t voff = p_int(a, "vocab_off", 0);
    if (pp && numel(pp) < M * 2 * n_cand) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M * n_vocab >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        /* The row's top_k, clamped to the buffer; <= 0 means the full width. A greedy row takes
         * none and its logits are not read. */
        int64_t k = R.p[m].top_k;
        if (k <= 0 || k > n_cand) k = n_cand;
        if (R.p[m].flags & RAD_SP_GREEDY) k = 0;

        const int64_t lb = rowoff(lg, m, n_vocab);
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);
        const int64_t pb = pp ? rowoff(pp, m, 2 * n_cand) : 0, ps = pp ? laststride(pp) : 0;
        float* buf = scratch_f32((size_t)n_vocab);
        const float* v = k > 0 ? in_row(lg, lb, n_vocab, buf) : buf;
        float pv = INFINITY;
        int64_t pi = -1;
        for (int64_t r = 0; r < n_cand; ++r) {
            const bool ok = r < k && topk_next_f32(v, n_vocab, &pv, &pi);
            st_int_dt(ci->dtype, ci->data, ib + r * is, ok ? pi : -1);
            stt(cv, vb + r * vs, ok ? pv : -INFINITY);
            if (!pp) continue;
            /* 0xffffffff marks a hole; the merge reads it as a negative id (cand_idx's -1). */
            uint32_t bits;
            const float val = ok ? pv : -INFINITY;
            std::memcpy(&bits, &val, 4);
            st_int_dt(pp->dtype, pp->data, pb + (2 * r) * ps,
                      ok ? (int64_t)(pi + voff) : (int64_t)0xffffffffu);
            st_int_dt(pp->dtype, pp->data, pb + (2 * r + 1) * ps, (int64_t)bits);
        }
    }
    return RAD_OK;
}

/* ================================================================== the narrowing stages
 *
 * Unlike libref, which reads `cand_val` through `ldt` inside O(n_cand) and O(n_cand^2) loops, all
 * four widen the candidate values to f32 once. Otherwise libref's structure is kept.
 *
 * The typical stage also precomputes each candidate's probability and surprisal distance (libref
 * recomputes exp and log inside the threshold search). The search stays libref's O(n_cand^2) scan:
 * a faster one would sort, which could change which of two equal distances is found first. */
enum Narrow { NR_TOPP, NR_MINP, NR_TYPICAL, NR_XTC };

static int narrow(const RadArgs* a, Narrow which) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *ci = &a->t[0], *cv = &a->t[1], *pr = &a->t[2];
    const int64_t n_cand = p_int(a, "n_cand", ci->rank >= 2 ? ci->shape[ci->rank - 1] : 0);
    if (n_cand <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(ci) / n_cand;
    if (M <= 0 || numel(cv) < M * n_cand) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        if (sp.flags & RAD_SP_GREEDY) continue;
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);

        float* val = scratch_f32((size_t)n_cand);
        float* prob = scratch_f32_b((size_t)n_cand);
        unsigned char* alive = (unsigned char*)scratch_f32_c((size_t)n_cand / 4 + 4);
        for (int64_t i = 0; i < n_cand; ++i) {
            alive[i] = ldi(ci, ib + i * is) >= 0 ? 1 : 0;
            val[i] = alive[i] ? ldt(cv, vb + i * vs) : -INFINITY;
        }

        auto drop = [&](int64_t i) {
            st_int_dt(ci->dtype, ci->data, ib + i * is, -1);
            stt(cv, vb + i * vs, -INFINITY);
        };

        float mx = -INFINITY;
        for (int64_t i = 0; i < n_cand; ++i) if (alive[i] && val[i] > mx) mx = val[i];
        if (mx == -INFINITY) continue;

        if (which == NR_MINP) {
            const float p = sp.min_p;
            if (!(p > 0.0f)) continue;
            /* Keep candidates within a factor p of the most likely. exp(v - max) >= p is
             * prob >= p * max_prob without the normaliser. */
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!alive[i]) continue;
                if (val[i] == mx) continue;                  /* the max always survives */
                if (sc_exp(val[i] - mx) < p) drop(i);
            }
            continue;
        }

        float sum = 0.0f;
        for (int64_t i = 0; i < n_cand; ++i) if (alive[i]) sum += sc_exp(val[i] - mx);
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int64_t i = 0; i < n_cand; ++i)
            prob[i] = alive[i] ? sc_exp(val[i] - mx) * inv : 0.0f;

        if (which == NR_TOPP) {
            const float p = sp.top_p;
            if (!(p < 1.0f) || !(p > 0.0f)) continue;
            /* Candidates arrive in descending order from sample_topk and are not re-sorted. Keep
             * the shortest prefix whose cumulative probability reaches p, at least one. */
            float cum = 0.0f;
            bool done = false;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!alive[i]) continue;
                if (done) { drop(i); continue; }
                cum += prob[i];
                if (cum >= p) done = true;
            }
            continue;
        }

        if (which == NR_TYPICAL) {
            const float p = sp.typical_p;
            if (!(p < 1.0f) || !(p > 0.0f)) continue;
            /* Locally typical sampling: keep the candidates whose surprisal -log(prob) is closest
             * to the entropy H until their mass reaches p. This orders by |surprisal - H|, not by
             * probability, so the selection is a distance threshold rather than a prefix. Ties at
             * the threshold are all kept (may overshoot p), so the result does not depend on
             * candidate index. */
            float H = 0.0f;
            for (int64_t i = 0; i < n_cand; ++i)
                if (alive[i] && prob[i] > 0.0f) H -= prob[i] * sc_log(prob[i]);
            float* dist = val;    /* `val` is dead here; reuse its storage. */
            for (int64_t i = 0; i < n_cand; ++i)
                dist[i] = (alive[i] && prob[i] > 0.0f) ? std::fabs(-sc_log(prob[i]) - H) : 0.0f;

            float cut = 0.0f;
            bool have_cut = false;
            for (int64_t c = 0; c < n_cand; ++c) {
                if (!alive[c] || !(prob[c] > 0.0f)) continue;
                const float dc = dist[c];
                float mass = 0.0f;
                for (int64_t i = 0; i < n_cand; ++i) {
                    if (!alive[i] || !(prob[i] > 0.0f)) continue;
                    if (dist[i] <= dc) mass += prob[i];
                }
                if (mass >= p && (!have_cut || dc < cut)) { cut = dc; have_cut = true; }
            }
            if (!have_cut) continue;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!alive[i]) continue;
                if (!(prob[i] > 0.0f) || dist[i] > cut) drop(i);
            }
            continue;
        }

        /* XTC ("exclude top choices"): with probability `xtc_probability`, drop every candidate
         * above `xtc_threshold` except the least likely of them. The draw comes from the row's
         * own random stream, so it is reproducible. */
        {
            if (!(sp.xtc_probability > 0.0f) || !(sp.xtc_threshold > 0.0f)) continue;
            if (rad_sample_uniform01(sp.seed ^ RAD_SAMPLE_KEY_XTC, sp.pos) >= sp.xtc_probability)
                continue;
            int64_t last_above = -1, n_above = 0;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!alive[i]) continue;
                if (prob[i] >= sp.xtc_threshold) { last_above = i; ++n_above; }
            }
            /* Needs at least two candidates above the threshold. */
            if (n_above < 2) continue;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!alive[i] || i == last_above) continue;
                if (prob[i] >= sp.xtc_threshold) drop(i);
            }
        }
    }
    return RAD_OK;
}

AVX_KERNEL(sample_topp)    { return narrow(a, NR_TOPP); }
AVX_KERNEL(sample_minp)    { return narrow(a, NR_MINP); }
AVX_KERNEL(sample_typical) { return narrow(a, NR_TYPICAL); }
AVX_KERNEL(sample_xtc)     { return narrow(a, NR_XTC); }

/* ------------------------------------------------------------------ sample_mask */
/* Applies a GBNF grammar's token bitmask (the state machine runs on the host; the mask is 19 KiB
 * per sequence at a 151K vocab). A set bit means allowed. Word w bit b covers token 32*w + b, LSB
 * first.
 *
 * At AVX-512 sixteen bits of a mask word are a `__mmask16`, so sixteen tokens take one masked store
 * of -inf with the mask inverted. Below AVX-512 the mask would have to be expanded to a 0/-1 vector
 * and blended, which costs more than the scalar shift-and-test. */
AVX_KERNEL(sample_mask) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *bm = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));
    const int64_t words = (n_vocab + 31) / 32;
    const int64_t mask_rows = numel(bm) / (words > 0 ? words : 1);
    if (mask_rows <= 0) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M * n_vocab >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        if (!(sp.flags & RAD_SP_MASK)) continue;
        const int64_t row = sp.mask_row;
        if (row < 0 || row >= mask_rows) continue;
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        const int64_t bb = rowoff(bm, row, words), bs = laststride(bm);

#if AVX_LEVEL == 3
        if (ls == 1 && lg->dtype == RAD_F32 && bs == 1 &&
            (bm->dtype == RAD_U32 || bm->dtype == RAD_I32)) {
            float* p = (float*)lg->data + lb;
            const uint32_t* w = (const uint32_t*)bm->data + bb;
            const __m512 neg = _mm512_set1_ps(-INFINITY);
            int64_t i = 0;
            for (; i + 32 <= n_vocab; i += 32) {
                const uint32_t bits = w[i / 32];
                /* Store -inf where the grammar bit is clear. */
                _mm512_mask_storeu_ps(p + i, (__mmask16)(~(uint16_t)(bits & 0xffffu)), neg);
                _mm512_mask_storeu_ps(p + i + 16, (__mmask16)(~(uint16_t)(bits >> 16)), neg);
            }
            for (; i < n_vocab; ++i)
                if (!((w[i / 32] >> (i % 32)) & 1u)) p[i] = -INFINITY;
            continue;
        }
#endif
        for (int64_t i = 0; i < n_vocab; ++i) {
            const uint32_t word = (uint32_t)ldi(bm, bb + (i / 32) * bs);
            if (!((word >> (i % 32)) & 1u)) stt(lg, lb + i * ls, -INFINITY);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_pick */
/* Inverse CDF in candidate order, accumulated in f32. If rounding lets `u` (< 1) fall through,
 * the fallback is the last candidate above -infinity, never a grammar-masked one top-k left at
 * -infinity at the tail. Greedy rows are skipped (their token is the argmax's).
 *
 * The draw is specified (abi/rad_sample.h): splitmix64 over (seed * golden ratio + pos), 24 bits as
 * a float in [0, 1), so host, device and reruns produce the same stream. */
AVX_KERNEL(sample_pick) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *ci = &a->t[0], *cv = &a->t[1], *pr = &a->t[2], *tok = &a->t[3];
    const int64_t n_cand = p_int(a, "n_cand", ci->rank >= 2 ? ci->shape[ci->rank - 1] : 0);
    if (n_cand <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(ci) / n_cand;
    if (M <= 0 || numel(cv) < M * n_cand || numel(tok) < M) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        if (R.p[m].flags & RAD_SP_GREEDY) continue;
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);

        float mx = -INFINITY;
        for (int64_t i = 0; i < n_cand; ++i) {
            if (ldi(ci, ib + i * is) < 0) continue;
            const float v = ldt(cv, vb + i * vs);
            if (v > mx) mx = v;
        }
        if (mx == -INFINITY) { st_int_dt(tok->dtype, tok->data, offlin(tok, m), -1); continue; }

        float sum = 0.0f;
        for (int64_t i = 0; i < n_cand; ++i) {
            if (ldi(ci, ib + i * is) < 0) continue;
            sum += sc_exp(ldt(cv, vb + i * vs) - mx);
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        const float u = rad_sample_uniform01(R.p[m].seed ^ RAD_SAMPLE_KEY_PICK, R.p[m].pos);

        float cum = 0.0f;
        int64_t chosen = -1, last = -1;
        for (int64_t i = 0; i < n_cand; ++i) {
            const int64_t id = ldi(ci, ib + i * is);
            if (id < 0) continue;
            const float v = ldt(cv, vb + i * vs);
            if (v > -INFINITY) last = id;
            cum += sc_exp(v - mx) * inv;
            if (u < cum) { chosen = id; break; }
        }
        st_int_dt(tok->dtype, tok->data, offlin(tok, m), chosen >= 0 ? chosen : last);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_argmax */
/* Greedy, ties to the lower token id, through the same `topk_next_f32` as top-k. When lm_head is
 * vocab-sharded this op is unusable (its answer is a local column) and greedy requests run at
 * top_k = 1 instead, so both paths must pick the same token. */
AVX_KERNEL(sample_argmax) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *tok = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0 || numel(tok) < M) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M * n_vocab >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* buf = scratch_f32((size_t)n_vocab);
        const float* v = in_row(lg, rowoff(lg, m, n_vocab), n_vocab, buf);
        float pv = INFINITY;
        int64_t pi = -1;
        const bool ok = topk_next_f32(v, n_vocab, &pv, &pi);
        st_int_dt(tok->dtype, tok->data, offlin(tok, m), ok ? pi : -1);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_logprobs */
/* For rows with RAD_SP_LOGPROBS: max, sum of exp(l - max), and the k largest as (id, logit)
 * pairs, descending, ties to the lower id (libref's contract, ref_sample.cpp). Other rows are not
 * written.
 *
 * One pass for the k largest instead of k passes of topk_next_f32 (`k` is the deployment's maximum
 * top_logprobs). A sorted set is seeded with the first k columns; the rest of the row is compared
 * a vector at a time against the set's smallest value, and lanes strictly above it are inserted.
 * Strictly, because ids increase along the scan and an equal later value loses the tie. Few vectors
 * pass the test after the first few thousand columns.
 *
 * The sum is f32 in four accumulators. Its terms are in (0, 1] and a logprob carries only its
 * relative error, so f32 suffices (libref uses double). Without `params` every row is written. */
AVX_KERNEL(sample_logprobs) {
    if (a->n_t < 3 || !a->t[0].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *st = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    const int64_t k = p_int(a, "k", 0);
    const int64_t voff = p_int(a, "vocab_off", 0);
    if (n_vocab <= 0 || k <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    const int64_t w = 2 + 2 * k;
    if (M <= 0 || numel(st) < M * w) return RAD_E_SHAPE;
    Rows R;
    if (pr->data) AVX_TRY(rows_of_params(pr, M, &R));

    AVX_PARALLEL_FOR_IF(M * n_vocab >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        if (pr->data && !(R.p[m].flags & RAD_SP_LOGPROBS)) continue;
        float* buf = scratch_f32((size_t)n_vocab + 2 * (size_t)k);
        const float* v = in_row(lg, rowoff(lg, m, n_vocab), n_vocab, buf);
        float* tv = buf + n_vocab;                 /* the set's values, descending */
        int32_t* ti = (int32_t*)(tv + k);          /* and their ids */

        const float mx = row_max(v, n_vocab);
        float sum = 0.0f;
        {
            const vf vmx = vf_set1(mx);
            vf s0 = vf_zero(), s1 = vf_zero(), s2 = vf_zero(), s3 = vf_zero();
            int64_t i = 0;
            for (; i + 4 * VF_N <= n_vocab; i += 4 * VF_N) {
                s0 = vf_add(s0, vf_exp(vf_sub(vf_loadu(v + i), vmx)));
                s1 = vf_add(s1, vf_exp(vf_sub(vf_loadu(v + i + VF_N), vmx)));
                s2 = vf_add(s2, vf_exp(vf_sub(vf_loadu(v + i + 2 * VF_N), vmx)));
                s3 = vf_add(s3, vf_exp(vf_sub(vf_loadu(v + i + 3 * VF_N), vmx)));
            }
            for (; i + VF_N <= n_vocab; i += VF_N) s0 = vf_add(s0, vf_exp(vf_sub(vf_loadu(v + i), vmx)));
            sum = vf_hsum(vf_add(vf_add(s0, s1), vf_add(s2, s3)));
            for (; i < n_vocab; ++i) sum += sc_exp(v[i] - mx);
        }

        /* Insert (x, id) into the sorted set of n entries, the set at most k long. */
        int64_t n = 0;
        auto insert = [&](float x, int32_t id) {
            int64_t j = n < k ? n++ : k - 1;
            while (j > 0 && x > tv[j - 1]) { tv[j] = tv[j - 1]; ti[j] = ti[j - 1]; --j; }
            tv[j] = x;
            ti[j] = id;
        };
        const int64_t seed = std::min(k, n_vocab);
        for (int64_t i = 0; i < seed; ++i) insert(v[i], (int32_t)i);
        int64_t i = seed;
#if AVX_LEVEL > 0
        for (; i + VF_N <= n_vocab; i += VF_N) {
            const vf x = vf_loadu(v + i);
#if AVX_LEVEL == 3
            unsigned msk = (unsigned)_mm512_cmp_ps_mask(x, vf_set1(tv[k - 1]), _CMP_GT_OQ);
#else
            unsigned msk = (unsigned)_mm256_movemask_ps(_mm256_cmp_ps(x, vf_set1(tv[k - 1]), _CMP_GT_OQ));
#endif
            /* Lanes in increasing id; the bar can rise after each insertion. */
            while (msk) {
                const int lane = __builtin_ctz(msk);
                msk &= msk - 1;
                if (v[i + lane] > tv[k - 1]) insert(v[i + lane], (int32_t)(i + lane));
            }
        }
#endif
        for (; i < n_vocab; ++i)
            if (v[i] > tv[k - 1]) insert(v[i], (int32_t)i);

        const int64_t sb = rowoff(st, m, w), ss = laststride(st);
        stt(st, sb, mx);
        stt(st, sb + ss, sum);
        for (int64_t j = 0; j < k; ++j) {
            stt(st, sb + (2 + 2 * j) * ss, j < n ? (float)(ti[j] + voff) : -1.0f);
            stt(st, sb + (3 + 2 * j) * ss, j < n ? tv[j] : -INFINITY);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_merge_topk */
/* Vocab-parallel merge. With lm_head vocab-sharded, each rank runs top-k over its own shard and the
 * candidate sets are all-gathered: k * world_size pairs (2 KiB at k=64, world_size=4) instead of
 * the full logits.
 *
 * `gathered` is [M, world_size * n_cand] (token id, logit) pairs as raw u32 words. Ids are already
 * global (each rank adds its vocab offset before the gather). The logit is read by bit pattern,
 * since the buffer's dtype is u32.
 *
 * The row's top_k is re-applied (hence `params`): the merged set holds up to world_size * k
 * candidates and no later stage narrows it. At k = 1 this matters: greedy rows are staged at
 * top_k = 1, and with world_size candidates left sample_pick would draw among them. */
AVX_KERNEL(sample_merge_topk) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *gt = &a->t[0], *pr = &a->t[1], *ci = &a->t[2], *cv = &a->t[3];
    const int64_t n_cand = p_int(a, "n_cand", ci->rank >= 2 ? ci->shape[ci->rank - 1] : 0);
    if (n_cand <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(ci) / n_cand;
    if (M <= 0 || numel(cv) < M * n_cand) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));
    if (gt->dtype != RAD_U32 && gt->dtype != RAD_I32) return RAD_E_DTYPE;
    const int64_t per_row = numel(gt) / M;
    const int64_t n_in = per_row / 2;
    if (n_in <= 0) return RAD_E_SHAPE;
    auto bits_to_f32 = [](int64_t w) {
        const uint32_t u = (uint32_t)w;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    };

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t gb = rowoff(gt, m, per_row), gs = laststride(gt);
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);
        const int32_t tk = R.p[m].top_k;
        const int64_t keep = (R.p[m].flags & RAD_SP_GREEDY) ? 0
                           : (tk > 0 && (int64_t)tk < n_cand) ? (int64_t)tk : n_cand;

        /* Unpack the pairs once (libref reads them inside the O(n_cand * n_in) selection). The
         * selection sort is libref's, ties to the lower token id. */
        float* gv = scratch_f32((size_t)n_in);
        int32_t* gi = (int32_t*)scratch_f32_b((size_t)n_in);
        for (int64_t j = 0; j < n_in; ++j) {
            gi[j] = (int32_t)(uint32_t)ldi(gt, gb + 2 * j * gs);
            gv[j] = bits_to_f32(ldi(gt, gb + (2 * j + 1) * gs));
        }

        float taken_v = INFINITY;
        int64_t taken_i = -1;
        for (int64_t r = 0; r < n_cand; ++r) {
            if (r >= keep) {
                st_int_dt(ci->dtype, ci->data, ib + r * is, -1);
                stt(cv, vb + r * vs, -INFINITY);
                continue;
            }
            float best = -INFINITY;
            int64_t best_id = -1;
            for (int64_t j = 0; j < n_in; ++j) {
                const int64_t id = gi[j];
                const float v = gv[j];
                if (id < 0) continue;
                const bool after = v < taken_v || (v == taken_v && id > taken_i);
                if (!after) continue;
                if (best_id < 0 || v > best || (v == best && id < best_id)) {
                    best = v;
                    best_id = id;
                }
            }
            st_int_dt(ci->dtype, ci->data, ib + r * is, best_id);
            stt(cv, vb + r * vs, best_id >= 0 ? best : -INFINITY);
            if (best_id < 0) continue;
            taken_v = best;
            taken_i = best_id;
        }
    }
    return RAD_OK;
}

/* ================================================================== dflash_select
 *
 * Not a sampler, but it also picks tokens: DFlash2's greedy walk over a candidate lattice. A
 * block-diffusion drafter proposes all positions of a block in one pass, so it cannot condition
 * position l+1 on its pick at l; this walk restores that dependence.
 *
 *   score[l, c] = unary[l, c] + sum_r pred[pid(l)][r] * hp[l][r] * succ[cand[l, c]][r]
 *   pid(0) = anchor[s * anchor_stride],   pid(l) = cand[l - 1, chosen(l - 1)]
 *
 * An absent `hp` is a plane of ones (a plain low-rank bigram). The walk is serial in l, so the
 * parallel axis is the sequence and the vector work is each candidate's rank-R dot. A predecessor
 * or candidate outside the vocabulary contributes nothing. Highest score wins, ties to the lower
 * candidate (libref's rule). */
static inline float dot_r(const float* a, const float* b, int64_t n) {
    vf acc = vf_zero();
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) acc = vf_fma(vf_loadu(a + i), vf_loadu(b + i), acc);
    float s = vf_hsum(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

AVX_KERNEL(dflash_select) {
    if (a->n_t < 7) return RAD_E_INVAL;
    const RadTensor *cand = &a->t[0], *un = &a->t[1], *anc = &a->t[3];
    const RadTensor *pred = &a->t[4], *succ = &a->t[5], *out = &a->t[6];
    const RadTensor* hp = rad_arg_in(a, 2);
    if (!cand->data || !un->data || !anc->data || !pred->data || !succ->data || !out->data)
        return RAD_E_INVAL;

    const int64_t steps = p_int(a, "steps", 0);
    const int64_t K     = p_int(a, "top_k", 0);
    /* Rows come from `cand` ([M * steps, K]); the `M` parameter is the band (avx_common.h:
     * band_rows). */
    const int64_t M     = steps > 0 && K > 0
                        ? band_rows(numel(cand) / (steps * K), p_int(a, "M", 0)) : 0;
    const int64_t R     = p_int(a, "rank", pred->rank >= 2 ? pred->shape[pred->rank - 1] : 0);
    const int64_t nv    = p_int(a, "n_vocab", pred->rank >= 2 ? pred->shape[pred->rank - 2] : 0);
    const int64_t astr  = p_int(a, "anchor_stride", 1);
    if (M <= 0 || steps <= 0 || K <= 0 || R <= 0 || nv <= 0 || astr <= 0) return RAD_E_SHAPE;
    if (numel(un) < M * steps * K) return RAD_E_SHAPE;
    if (numel(anc) < (M - 1) * astr + 1 || numel(out) < M * steps) return RAD_E_SHAPE;
    if (numel(pred) < nv * R || numel(succ) < nv * R) return RAD_E_SHAPE;
    if (hp && numel(hp) < M * steps * R) return RAD_E_SHAPE;

    const int64_t cs = laststride(cand), us = laststride(un), os = laststride(out);

    AVX_PARALLEL_FOR_IF(M * steps * K * R >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* ax = scratch_f32((size_t)R);
        float* rv = scratch_f32_b((size_t)R);
        float* hv = scratch_f32_c((size_t)R);
        int64_t pid = ldi(anc, offlin(anc, m * astr));
        for (int64_t l = 0; l < steps; ++l) {
            const int64_t r  = m * steps + l;
            const int64_t cb = rowoff(cand, r, K), ub = rowoff(un, r, K);
            /* The candidate-independent factor, once per step. */
            const bool ok = pid >= 0 && pid < nv;
            const float* pv = ok ? in_row(pred, rowoff(pred, pid, R), R, rv) : nullptr;
            const float* hw = hp ? in_row(hp, rowoff(hp, r, R), R, hv) : nullptr;
            for (int64_t i = 0; i < R; ++i)
                ax[i] = (ok ? pv[i] : 0.0f) * (hp ? hw[i] : 1.0f);

            int64_t best_c = 0;
            float   best_s = -INFINITY;
            for (int64_t c = 0; c < K; ++c) {
                const int64_t tok = ld_int_dt(cand->dtype, cand->data, cb + c * cs);
                float sc = ldt(un, ub + c * us);
                if (tok >= 0 && tok < nv)
                    sc += dot_r(ax, in_row(succ, rowoff(succ, tok, R), R, rv), R);
                if (sc > best_s) { best_s = sc; best_c = c; }
            }
            pid = ld_int_dt(cand->dtype, cand->data, cb + best_c * cs);
            st_int_dt(out->dtype, out->data, rowoff(out, m, steps) + l * os, pid);
        }
    }
    return RAD_OK;
}

/* ================================================================== dflash_markov_greedy
 *
 * DSpark's walk over the whole vocabulary (libref ref_quant.cpp, ref_markov_greedy): at each step
 * every token scores its draft logit plus sum_r pred[pid][r] * succ[token][r], the greatest wins
 * (ties to the lower id) and becomes the next predecessor. The winners of each 64-token tile are
 * written too, as the device row's decomposition. Scalar and in libref's order (the bf16 form starts
 * the sum at the logit, the f32 form adds it last), so the argmax ties the same way. */
static int markov_greedy(const RadArgs* a, uint32_t logits_dtype) {
    if (a->n_t < 7) return RAD_E_INVAL;
    const RadTensor *logits = &a->t[0], *anc = &a->t[1], *pred = &a->t[2], *succ = &a->t[3];
    const RadTensor *tsc = &a->t[4], *ttok = &a->t[5], *out = &a->t[6];
    if (!logits->data || !anc->data || !pred->data || !succ->data || !tsc->data || !ttok->data ||
        !out->data)
        return RAD_E_INVAL;
    if (logits->dtype != logits_dtype || pred->dtype != RAD_BF16 || succ->dtype != RAD_BF16 ||
        tsc->dtype != RAD_F32 || ttok->dtype != RAD_I32 || out->dtype != RAD_I32)
        return RAD_E_DTYPE;
    if (anc->dtype != RAD_I32) return RAD_E_SHAPE;

    const int64_t steps = p_int(a, "steps", 0);
    const int64_t R     = p_int(a, "rank", 0);
    const int64_t V     = p_int(a, "n_vocab", 0);
    const int64_t astr  = p_int(a, "anchor_stride", 1);
    const int64_t M     = steps > 0 && V > 0
                        ? band_rows(numel(logits) / (steps * V), p_int(a, "M", 0)) : 0;
    if (M <= 0 || steps <= 0 || R <= 0 || V <= 0 || astr <= 0) return RAD_E_SHAPE;
    if (numel(logits) < M * steps * V || numel(anc) < (M - 1) * astr + 1 ||
        numel(pred) < V * R || numel(succ) < V * R || numel(out) < M * steps)
        return RAD_E_SHAPE;
    const int64_t n_tiles = (V + 63) / 64;
    if (numel(tsc) < M * n_tiles || numel(ttok) < M * n_tiles) return RAD_E_SHAPE;

    const int64_t ls = laststride(logits), os = laststride(out);
    const int64_t tss = laststride(tsc), tis = laststride(ttok);
    AVX_PARALLEL_FOR_IF(M * steps * V * R >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* prev = scratch_f32((size_t)R);
        float* sv   = scratch_f32_b((size_t)R);
        int64_t pid = ldi(anc, offlin(anc, m * astr));
        for (int64_t l = 0; l < steps; ++l) {
            const int64_t row = m * steps + l;
            const bool ok = pid >= 0 && pid < V;
            if (ok) {
                const float* pv = in_row(pred, rowoff(pred, pid, R), R, prev);
                if (pv != prev) std::memcpy(prev, pv, (size_t)R * sizeof(float));
            } else {
                for (int64_t r = 0; r < R; ++r) prev[r] = 0.0f;
            }
            const int64_t lb = rowoff(logits, row, V);
            float   gbest = -INFINITY;
            int64_t gtok  = 0;
            for (int64_t tile = 0; tile < n_tiles; ++tile) {
                const int64_t first = tile * 64, end = std::min<int64_t>(first + 64, V);
                float   best = -INFINITY;
                int64_t btok = first;
                for (int64_t tok = first; tok < end; ++tok) {
                    const float base = ldt(logits, lb + tok * ls);
                    const float* su = in_row(succ, rowoff(succ, tok, R), R, sv);
                    float s = logits_dtype == RAD_F32 ? 0.0f : base;
                    for (int64_t r = 0; r < R; ++r) s += prev[r] * su[r];
                    if (logits_dtype == RAD_F32) s += base;
                    if (s > best || (s == best && tok < btok)) { best = s; btok = tok; }
                }
                stt(tsc, rowoff(tsc, m, n_tiles) + tile * tss, best);
                st_int_dt(ttok->dtype, ttok->data, rowoff(ttok, m, n_tiles) + tile * tis, btok);
                if (best > gbest || (best == gbest && btok < gtok)) { gbest = best; gtok = btok; }
            }
            st_int_dt(out->dtype, out->data, rowoff(out, m, steps) + l * os, gtok);
            pid = gtok;
        }
    }
    return RAD_OK;
}

AVX_KERNEL(dflash_markov_greedy) { return markov_greedy(a, RAD_BF16); }
AVX_KERNEL(dflash_markov_greedy_f32) { return markov_greedy(a, RAD_F32); }

/* ================================================================== dflash_tree
 *
 * The same lattice grown best-first into a tree of `nodes` nodes (rad_drafter_tree), libref's rule
 * (ref_quant.cpp, ref_dflash_tree): a taken node's children are the valid candidates at the next
 * position, scored as dflash_select scores them after the node's token, each with log probability
 * parent + log_softmax(score / temp) over them; each round takes the largest untaken child, ties to
 * the one pushed first. Parallel over sequences. */
AVX_KERNEL(dflash_tree) {
    if (a->n_t < 8) return RAD_E_INVAL;
    const RadTensor *cand = &a->t[0], *un = &a->t[1], *anc = &a->t[3];
    const RadTensor *pred = &a->t[4], *succ = &a->t[5], *out = &a->t[6], *par = &a->t[7];
    const RadTensor* hp = rad_arg_in(a, 2);
    if (!cand->data || !un->data || !anc->data || !pred->data || !succ->data || !out->data ||
        !par->data)
        return RAD_E_INVAL;

    const int64_t steps = p_int(a, "steps", 0);
    const int64_t K     = p_int(a, "top_k", 0);
    const int64_t M     = steps > 0 && K > 0
                        ? band_rows(numel(cand) / (steps * K), p_int(a, "M", 0)) : 0;
    const int64_t R     = p_int(a, "rank", pred->rank >= 2 ? pred->shape[pred->rank - 1] : 0);
    const int64_t nv    = p_int(a, "n_vocab", pred->rank >= 2 ? pred->shape[pred->rank - 2] : 0);
    const int64_t astr  = p_int(a, "anchor_stride", 1);
    const int64_t N     = p_int(a, "nodes", steps);
    const float   temp  = p_f32(a, "temp", 1.0f);
    if (M <= 0 || steps <= 0 || K <= 0 || R <= 0 || nv <= 0 || astr <= 0 || N <= 0 ||
        !(temp > 0.0f))
        return RAD_E_SHAPE;
    if (numel(un) < M * steps * K) return RAD_E_SHAPE;
    if (numel(anc) < (M - 1) * astr + 1 || numel(out) < M * N || numel(par) < M * N)
        return RAD_E_SHAPE;
    if (numel(pred) < nv * R || numel(succ) < nv * R) return RAD_E_SHAPE;
    if (hp && numel(hp) < M * steps * R) return RAD_E_SHAPE;

    const int64_t cs = laststride(cand), us = laststride(un), os = laststride(out);
    const int64_t pps = laststride(par);

    AVX_PARALLEL_FOR_IF(M * N * K * R >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        struct Node { int64_t l, tok, parent; float lp; };
        struct Kid  { float lp; int64_t parent, l, tok; bool taken; };
        std::vector<Node>  sel;
        std::vector<Kid>   front;
        std::vector<float> ax((size_t)R), rv((size_t)R), hv((size_t)R), x((size_t)K);
        sel.push_back({ -1, ldi(anc, offlin(anc, m * astr)), -1, 0.0f });

        auto expand = [&](int64_t n) {
            const Node nd = sel[(size_t)n];
            const int64_t l = nd.l + 1;
            if (l >= steps) return;
            const int64_t r  = m * steps + l;
            const int64_t cb = rowoff(cand, r, K), ub = rowoff(un, r, K);
            const bool ok = nd.tok >= 0 && nd.tok < nv;
            const float* pv = ok ? in_row(pred, rowoff(pred, nd.tok, R), R, rv.data()) : nullptr;
            const float* hw = hp ? in_row(hp, rowoff(hp, r, R), R, hv.data()) : nullptr;
            for (int64_t i = 0; i < R; ++i)
                ax[(size_t)i] = (ok ? pv[i] : 0.0f) * (hp ? hw[i] : 1.0f);
            float mx = -INFINITY;
            for (int64_t c = 0; c < K; ++c) {
                const int64_t tok = ld_int_dt(cand->dtype, cand->data, cb + c * cs);
                float sc = ldt(un, ub + c * us);
                const bool valid = tok >= 0 && tok < nv;
                if (valid)
                    sc += dot_r(ax.data(), in_row(succ, rowoff(succ, tok, R), R, rv.data()), R);
                x[(size_t)c] = valid ? sc / temp : -INFINITY;
                if (x[(size_t)c] > mx) mx = x[(size_t)c];
            }
            if (mx == -INFINITY) return;
            float sum = 0.0f;
            for (int64_t c = 0; c < K; ++c)
                if (x[(size_t)c] > -INFINITY) sum += std::exp(x[(size_t)c] - mx);
            const float lse = mx + std::log(sum);
            for (int64_t c = 0; c < K; ++c) {
                if (!(x[(size_t)c] > -INFINITY)) continue;
                front.push_back({ nd.lp + (x[(size_t)c] - lse), n, l,
                                  ld_int_dt(cand->dtype, cand->data, cb + c * cs), false });
            }
        };

        expand(0);
        for (int64_t j = 1; j <= N; ++j) {
            int64_t best = -1;
            for (size_t f = 0; f < front.size(); ++f)
                if (!front[f].taken && (best < 0 || front[f].lp > front[(size_t)best].lp))
                    best = (int64_t)f;
            if (best < 0) {
                sel.push_back({ steps, 0, (int64_t)sel.size() - 1, -INFINITY });
            } else {
                Kid& k = front[(size_t)best];
                k.taken = true;
                sel.push_back({ k.l, k.tok, k.parent, k.lp });
                if (j < N) expand((int64_t)sel.size() - 1);
            }
            const Node& nd = sel.back();
            st_int_dt(out->dtype, out->data, rowoff(out, m, N) + (j - 1) * os, nd.tok);
            st_int_dt(par->dtype, par->data, rowoff(par, m, N) + (j - 1) * pps, nd.parent);
        }
    }
    return RAD_OK;
}
