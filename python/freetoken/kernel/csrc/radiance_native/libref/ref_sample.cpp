/* ref_sample.cpp -- the sampler ops.
 *
 * Sampling runs as kernels, not on the host: at a 151K vocab and batch 256 the logits are ~77 MiB
 * per step, and copying them for a host chain would cost bandwidth and a synchronisation on the step
 * path (spec §13). core/sample/ keeps llama.cpp's chain as a host reference; this is an independent
 * second implementation of the same arithmetic, so comparing the two is meaningful.
 *
 * ============================== Every scalar is per row ==============================
 *
 * Sampling scalars (`rep`, `freq`, `pres`, `k`, `p`, `seed`, `pos`) arrive as data, in a `params`
 * operand of RadSampleParams rows, one per sampled position, uploaded once per step
 * (abi/rad_sample.h). As parameters they would be frozen at declare (see ref_registry.cpp).
 *
 * ============================== The candidate set ==============================
 *
 * `sample_topk` produces a fixed-width candidate set and the narrowing stages work in place. A
 * dropped candidate is index -1, value -infinity; kept candidates stay in place in descending
 * order, so a later stage can stop at the first -1 or skip holes (`sample_pick` skips).
 *
 * `cand_val` carries logits, not probabilities: each narrowing stage takes its own softmax over the
 * set, as llama.cpp does, so stages compose in any order.
 *
 * A row's top_k may be narrower than the buffer (a declare-time bound): top_k 40 in a 1024-wide
 * buffer gives 40 candidates and 984 holes.
 *
 * A greedy row (RAD_SP_GREEDY) has no candidates; its token is sample_argmax's. In a mixed batch
 * every stage runs over every row, so top-k and the merge write it an empty set, narrowing leaves it
 * alone, and the pick must not overwrite its token with a draw.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include "rad_sample.h"

#include <algorithm>
#include <cfloat>
#include <unordered_map>
#include <vector>

using namespace ref;

namespace {

/* The params operand, checked once. */
struct Rows {
    const RadSampleParams* p;
    int64_t                n;
};

static inline int param_rows(const RadTensor* t, int64_t M, Rows* out) {
    if (!t || !t->data) return RAD_E_INVAL;
    /* The row array is a byte blob declared as [rows, sizeof(RadSampleParams)/4] u32, the only way
     * the builder can express a per-row byte size. The dtype must be 32-bit: a 16-bit view could
     * pass a byte-count check and then misread every row. */
    if (t->dtype != RAD_U32 && t->dtype != RAD_I32 && t->dtype != RAD_F32) return RAD_E_DTYPE;
    if (numel(t) * 4 < M * (int64_t)sizeof(RadSampleParams)) return RAD_E_SHAPE;
    out->p = (const RadSampleParams*)t->data;
    out->n = M;
    return RAD_OK;
}

}  /* namespace */

/* ------------------------------------------------------------------ the history window */
/* The window both history stages read: `hist_off` and `hist_len` address the row's own row of the
 * history plane, clamped to it, and `last_n` keeps the tail of that span (the most recent tokens).
 * last_n 0 disables the stage for the row (returns false); negative keeps the whole span, as
 * llama.cpp's repeat_last_n and dry_penalty_last_n. */
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
 *     presence     logit -= pres                                        (once per distinct token)
 *
 * The repetition penalty is sign-dependent: dividing a negative logit would raise it, so it is
 * multiplied instead.
 *
 * History ids are global and the logits row is this rank's vocabulary shard, so a token's column is
 * its id less `vocab_off`; tokens outside the shard are other ranks'. */
int ref_sample_penalties(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *hist = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(param_rows(pr, M, &R));
    const int64_t hmax = numel(hist) / M;
    if (hmax <= 0) return RAD_E_SHAPE;
    const int64_t voff = p_int(a, "vocab_off", 0);

    REF_PAR_FOR(M * (n_vocab + hmax))
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
        for (int64_t i = 0; i < hl; ++i) {
            const int64_t tok = ld_int(hist->dtype, hist->data, hb + i * hs);
            const int64_t col = tok - voff;
            if (tok < 0 || col < 0 || col >= n_vocab) continue;
            /* First occurrence only: penalties are per distinct token, the count carries the
             * multiplicity. O(hist^2) over a short history. */
            bool first = true;
            int64_t count = 0;
            for (int64_t j = 0; j < hl; ++j) {
                const int64_t t2 = ld_int(hist->dtype, hist->data, hb + j * hs);
                if (t2 != tok) continue;
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
/* DRY ("don't repeat yourself"): a penalty on a token that would extend a suffix of the history
 * that occurred before, to stop phrase-level loops. llama.cpp's rule (llama_sampler_dry_apply):
 *
 *     for every earlier position j, n_j is how long a suffix of the window also ends at j, capped
 *     at the breaker limit; the token after j would extend that repeat. Each such token is
 *     penalised once, by multiplier * base^(L - allowed_length), L the longest n_j among the
 *     positions it follows.
 *
 * Only lengths >= `allowed_length` count, and a token that is itself a whole sequence breaker is
 * never penalised. Breaker facts come from the host, which has the vocabulary: the row's
 * `dry_rep_limit` (0 for none) and the optional `breakers` plane, one byte per history position,
 * nonzero for a single-token breaker.
 *
 * The penalty is computed by repeated multiplication, stopping short of overflow, so every
 * implementation computes the same float (llama.cpp clamps the exponent likewise). Columns are
 * global ids less `vocab_off`. Quadratic over the window. */
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

int ref_sample_dry(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *hist = &a->t[2];
    const RadTensor* brk = t_in(a, 3);
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(param_rows(pr, M, &R));
    const int64_t hmax = numel(hist) / M;
    if (hmax <= 0) return RAD_E_SHAPE;
    /* The breaker plane matches the history position for position. */
    if (brk && numel(brk) < M * hmax) return RAD_E_SHAPE;
    const int64_t voff = p_int(a, "vocab_off", 0);

    REF_PAR_FOR(M * (n_vocab + hmax * hmax))
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
        auto tok_at = [&](int64_t i) {
            return ld_int(hist->dtype, hist->data, hb + i * hs);
        };
        auto exempt = [&](int64_t i) {
            return brk && ld_int(brk->dtype, brk->data, bb + i * bs) != 0;
        };

        /* The longest repeat each continuation token would extend. */
        std::unordered_map<int64_t, int64_t> longest;
        for (int64_t j = 0; j + 1 < hl; ++j) {
            const int64_t lim = j + 1 < cap ? j + 1 : cap;
            int64_t n = 0;
            while (n < lim && tok_at(j - n) == tok_at(hl - 1 - n)) ++n;
            if (n < allowed || exempt(j + 1)) continue;
            const int64_t next = tok_at(j + 1);
            const int64_t col  = next - voff;
            if (next < 0 || col < 0 || col >= n_vocab) continue;
            int64_t& L = longest[next];
            if (n > L) L = n;
        }

        const float base = sp.dry_base > 0.0f ? sp.dry_base : 1.75f;
        for (const auto& e : longest) {
            const int64_t off = lb + (e.first - voff) * ls;
            stt(lg, off, ldt(lg, off) - dry_penalty(sp.dry_multiplier, base, e.second - allowed));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_ngram_ban */
/* The windowed no-repeat-ngram ban: Unlimited-OCR's `SlidingWindowNoRepeatNgramProcessor`
 * (modeling_unlimitedocr.py:353-383), which SGLang ships as
 * DeepseekOCRNoRepeatNGramLogitProcessor. With the row's history `s[0, hl)` (oldest first, `hl`
 * committed tokens including the one just emitted), `n` and `w` from the `ngram` plane (an i32
 * pair a row, {n-gram length, window}):
 *
 *     search_start = max(0, hl - w)          (w <= 0: the whole span)
 *     search_end   = hl - n + 1
 *     prefix       = s[hl - n + 1, hl)        (empty at n == 1)
 *     for idx in [search_start, search_end):
 *         if s[idx, idx + n - 1) == prefix:   ban s[idx + n - 1]
 *
 * A banned token's logit becomes -infinity, exactly as the reference writes it, so the argmax and
 * every later stage see the same distribution. The n-gram that spans the window boundary is one
 * of the searched ones, as in the reference: the window is applied to the START index, not to the
 * n-gram's end.
 *
 * Ids are global and columns are global less `vocab_off`; a banned token on another rank's shard
 * is that rank's to mask. Rows whose n is 0 (the default) are untouched, so a request without the
 * ban leaves the logits bit for bit as they were. The scan is O(w * n) a row and compares
 * integers, so every implementation (this one, libavx, libr4d) agrees by construction.
 *
 * The optional `whitelist` bitmask is the reference's whitelist: a token whose column bit is set
 * in the row's own plane row is removed from the banned set before the -inf writes
 * (`banned.difference_update(self.whitelist)`, modeling_unlimitedocr.py:379), so it is never
 * banned. Absent, nothing is exempt -- the default serving form, bit-identical to the ban
 * without the operand. */
int ref_sample_ngram_ban(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *hist = &a->t[2], *ng = &a->t[3];
    const RadTensor* wl = a->n_t > 4 && a->t[4].data ? &a->t[4] : nullptr;
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(param_rows(pr, M, &R));
    const int64_t hmax = numel(hist) / M;
    if (hmax <= 0) return RAD_E_SHAPE;
    /* One {n, window} pair a row. At least M pairs, so a plane shared by sparser row sets is not
     * read past its end. */
    if (numel(ng) < 2 * M) return RAD_E_SHAPE;
    const int64_t voff = p_int(a, "vocab_off", 0);
    const int64_t wwords = (n_vocab + 31) / 32;
    if (wl && wl->dtype != RAD_I32 && wl->dtype != RAD_U32) return RAD_E_DTYPE;
    if (wl && numel(wl) < M * wwords) return RAD_E_SHAPE;

    REF_PAR_FOR(M * hmax)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        const int64_t nb = rowoff(ng, m, 2), ns = laststride(ng);
        const int64_t n = ld_int(ng->dtype, ng->data, nb);
        const int64_t win_req = ld_int(ng->dtype, ng->data, nb + ns);
        if (n <= 0) continue;

        /* The whole staged span; the window below keeps its tail. last_n -1 is hist_window's
         * "the whole span". */
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
        auto tok_at = [&](int64_t i) {
            return ld_int(hist->dtype, hist->data, hb + i * hs);
        };
        /* The reference removes the whitelist from the banned set before it writes; a column
         * whose bit is set in the row's whitelist words is therefore never written. */
        auto exempt = [&](int64_t col) {
            if (!wl) return false;
            const uint32_t word = (uint32_t)ld_int(wl->dtype, wl->data, wb + (col / 32) * ws);
            return ((word >> (uint32_t)(col % 32)) & 1u) != 0u;
        };
        /* n == 1 bans every token in the window: the prefix is empty, as in the reference.
         * The prefix is read once, not per search position: it is the same n-1 tokens
         * compared at every idx, and the row's history is short (the staged span). */
        const int64_t plen = n - 1, p0 = hl - plen;
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
int ref_sample_temp(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(param_rows(pr, M, &R));

    REF_PAR_FOR(M * n_vocab)
    for (int64_t m = 0; m < M; ++m) {
        const float t = R.p[m].temp;
        if (!(t > 0.0f) || t == 1.0f) continue;
        const float inv = 1.0f / t;
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        for (int64_t i = 0; i < n_vocab; ++i) stt(lg, lb + i * ls, ldt(lg, lb + i * ls) * inv);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_topk */
int ref_sample_topk(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *ci = &a->t[2], *cv = &a->t[3];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(param_rows(pr, M, &R));
    /* The candidate width is the buffer's (the declare-time bound), not the request's. */
    const int64_t n_cand = ci->rank >= 2 ? ci->shape[ci->rank - 1] : numel(ci) / M;
    if (n_cand <= 0 || numel(cv) < M * n_cand) return RAD_E_SHAPE;

    /* Optional vocab-parallel pair plane. cand_idx stays local (it indexes this rank's logits
     * row); `pairs` carries the same candidates as (u32 global id, f32 logit) with `vocab_off`
     * added, for the all-gather and sample_merge_topk. */
    const RadTensor* pp = a->n_t > 4 && a->t[4].data ? &a->t[4] : nullptr;
    const int64_t voff = p_int(a, "vocab_off", 0);
    if (pp && numel(pp) < M * 2 * n_cand) return RAD_E_SHAPE;

    REF_PAR_FOR(M * n_vocab * 4)
    for (int64_t m = 0; m < M; ++m) {
        /* The row's top_k, clamped to the buffer; <= 0 means the whole width. A greedy row takes
         * none. */
        int64_t k = R.p[m].top_k;
        if (k <= 0 || k > n_cand) k = n_cand;
        if (R.p[m].flags & RAD_SP_GREEDY) k = 0;

        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);
        const int64_t pb = pp ? rowoff(pp, m, 2 * n_cand) : 0, ps = pp ? laststride(pp) : 0;
        float pv = INFINITY;
        int64_t pi = -1;
        for (int64_t r = 0; r < n_cand; ++r) {
            const bool ok = r < k && topk_next(lg, lb, ls, n_vocab, &pv, &pi);
            st_int(ci->dtype, ci->data, ib + r * is, ok ? pi : -1);
            stt(cv, vb + r * vs, ok ? pv : -INFINITY);
            if (!pp) continue;
            /* 0xffffffff is the hole (a negative signed id to the merge, -1 in cand_idx). */
            uint32_t bits;
            const float val = ok ? pv : -INFINITY;
            std::memcpy(&bits, &val, 4);
            st_int(pp->dtype, pp->data, pb + (2 * r) * ps,
                   ok ? (int64_t)(pi + voff) : (int64_t)0xffffffffu);
            st_int(pp->dtype, pp->data, pb + (2 * r + 1) * ps, (int64_t)bits);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the narrowing stages */
namespace {

enum Narrow { NR_TOPP, NR_MINP, NR_TYPICAL, NR_XTC };

static int narrow(const RadArgs* a, Narrow which) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *ci = &a->t[0], *cv = &a->t[1], *pr = &a->t[2];
    const int64_t n_cand = p_int(a, "n_cand", ci->rank >= 2 ? ci->shape[ci->rank - 1] : 0);
    if (n_cand <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(ci) / n_cand;
    if (M <= 0 || numel(cv) < M * n_cand) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(param_rows(pr, M, &R));

    REF_PAR_FOR(M * n_cand * 4)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        if (sp.flags & RAD_SP_GREEDY) continue;
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);

        auto drop = [&](int64_t i) {
            st_int(ci->dtype, ci->data, ib + i * is, -1);
            stt(cv, vb + i * vs, -INFINITY);
        };
        auto live = [&](int64_t i) {
            return ld_int(ci->dtype, ci->data, ib + i * is) >= 0;
        };

        float mx = -INFINITY;
        for (int64_t i = 0; i < n_cand; ++i)
            if (live(i)) { const float v = ldt(cv, vb + i * vs); if (v > mx) mx = v; }
        if (mx == -INFINITY) continue;

        if (which == NR_MINP) {
            const float p = sp.min_p;
            if (!(p > 0.0f)) continue;
            /* Keep everything within a factor p of the most likely candidate: exp(v - max) >= p
             * is prob >= p * max_prob without forming the normaliser. */
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!live(i)) continue;
                const float v = ldt(cv, vb + i * vs);
                if (v == mx) continue;                       /* the max always survives */
                if (ref_exp(v - mx) < p) drop(i);
            }
            continue;
        }

        /* The other three stages need normalised probabilities. */
        float sum = 0.0f;
        for (int64_t i = 0; i < n_cand; ++i)
            if (live(i)) sum += ref_exp(ldt(cv, vb + i * vs) - mx);
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

        if (which == NR_TOPP) {
            const float p = sp.top_p;
            if (!(p < 1.0f) || !(p > 0.0f)) continue;
            /* Candidates arrive in descending order from sample_topk and are not re-sorted. Keep
             * the shortest prefix whose cumulative probability reaches p, at least one. */
            float cum = 0.0f;
            bool done = false;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!live(i)) continue;
                if (done) { drop(i); continue; }
                cum += ref_exp(ldt(cv, vb + i * vs) - mx) * inv;
                if (cum >= p) done = true;
            }
            continue;
        }

        if (which == NR_TYPICAL) {
            const float p = sp.typical_p;
            if (!(p < 1.0f) || !(p > 0.0f)) continue;
            /* Locally typical sampling: keep the candidates whose surprisal -log(prob) is closest
             * to the entropy until their mass reaches p. That order differs from the candidate
             * order, so this is a threshold on |surprisal - H| rather than a prefix: find the
             * smallest distance whose mass reaches p, keep everything at or below it. Ties are
             * kept (may overshoot p) so the result does not depend on candidate index. */
            /* q[i] is a live candidate's probability (0 for a dropped one), d[i] its distance. */
            std::vector<float> q((size_t)n_cand, 0.0f), d((size_t)n_cand, 0.0f);
            float H = 0.0f;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!live(i)) continue;
                q[(size_t)i] = ref_exp(ldt(cv, vb + i * vs) - mx) * inv;
                if (q[(size_t)i] > 0.0f) H -= q[(size_t)i] * ref_log(q[(size_t)i]);
            }
            for (int64_t i = 0; i < n_cand; ++i)
                if (q[(size_t)i] > 0.0f) d[(size_t)i] = ref_abs(-ref_log(q[(size_t)i]) - H);
            /* Scan the distances that occur (n_cand is at most about a thousand). */
            float cut = 0.0f;
            bool have_cut = false;
            for (int64_t c = 0; c < n_cand; ++c) {
                if (!(q[(size_t)c] > 0.0f)) continue;
                float mass = 0.0f;
                for (int64_t i = 0; i < n_cand; ++i)
                    if (q[(size_t)i] > 0.0f && d[(size_t)i] <= d[(size_t)c]) mass += q[(size_t)i];
                if (mass >= p && (!have_cut || d[(size_t)c] < cut)) {
                    cut = d[(size_t)c];
                    have_cut = true;
                }
            }
            if (!have_cut) continue;
            for (int64_t i = 0; i < n_cand; ++i)
                if (live(i) && (!(q[(size_t)i] > 0.0f) || d[(size_t)i] > cut)) drop(i);
            continue;
        }

        /* XTC ("exclude top choices"): with probability `xtc_probability`, drop every candidate
         * above `xtc_threshold` except the least likely of them, removing the obvious
         * continuations. The draw comes from the row's own stream, so it reproduces. */
        {
            if (!(sp.xtc_probability > 0.0f) || !(sp.xtc_threshold > 0.0f)) continue;
            if (rad_sample_uniform01(sp.seed ^ RAD_SAMPLE_KEY_XTC, sp.pos) >= sp.xtc_probability)
                continue;
            int64_t last_above = -1, n_above = 0;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!live(i)) continue;
                if (ref_exp(ldt(cv, vb + i * vs) - mx) * inv >= sp.xtc_threshold) {
                    last_above = i;
                    ++n_above;
                }
            }
            /* Needs at least two candidates above the threshold. */
            if (n_above < 2) continue;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!live(i) || i == last_above) continue;
                if (ref_exp(ldt(cv, vb + i * vs) - mx) * inv >= sp.xtc_threshold) drop(i);
            }
        }
    }
    return RAD_OK;
}

}  /* namespace */

int ref_sample_topp(const RadArgs* a, RadStream)    { return narrow(a, NR_TOPP); }
int ref_sample_minp(const RadArgs* a, RadStream)    { return narrow(a, NR_MINP); }
int ref_sample_typical(const RadArgs* a, RadStream) { return narrow(a, NR_TYPICAL); }
int ref_sample_xtc(const RadArgs* a, RadStream)     { return narrow(a, NR_XTC); }

/* ------------------------------------------------------------------ sample_mask */
/* GBNF grammar masking. The state machine stays on the host; a token bitmask per sequence (19 KiB
 * at a 151K vocab) is uploaded every step (spec §13).
 *
 * A set bit means allowed. Word w bit b covers token 32*w + b, least significant bit first.
 *
 * Each row names its mask row; unconstrained rows share the plane with `mask_row` -1. */
int ref_sample_mask(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *bm = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(param_rows(pr, M, &R));
    const int64_t words = (n_vocab + 31) / 32;
    const int64_t mask_rows = numel(bm) / (words > 0 ? words : 1);
    if (mask_rows <= 0) return RAD_E_SHAPE;

    REF_PAR_FOR(M * n_vocab)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        if (!(sp.flags & RAD_SP_MASK)) continue;
        const int64_t row = sp.mask_row;
        if (row < 0 || row >= mask_rows) continue;
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        const int64_t bb = rowoff(bm, row, words), bs = laststride(bm);
        for (int64_t i = 0; i < n_vocab; ++i) {
            const uint32_t word = (uint32_t)ld_int(bm->dtype, bm->data, bb + (i / 32) * bs);
            if (!((word >> (i % 32)) & 1u)) stt(lg, lb + i * ls, -INFINITY);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_pick */
int ref_sample_pick(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *ci = &a->t[0], *cv = &a->t[1], *pr = &a->t[2], *tok = &a->t[3];
    const int64_t n_cand = p_int(a, "n_cand", ci->rank >= 2 ? ci->shape[ci->rank - 1] : 0);
    if (n_cand <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(ci) / n_cand;
    if (M <= 0 || numel(cv) < M * n_cand || numel(tok) < M) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(param_rows(pr, M, &R));

    REF_PAR_FOR(M * n_cand * 4)
    for (int64_t m = 0; m < M; ++m) {
        /* A greedy row's token is the argmax's, already written. */
        if (R.p[m].flags & RAD_SP_GREEDY) continue;
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);

        float mx = -INFINITY;
        for (int64_t i = 0; i < n_cand; ++i) {
            if (ld_int(ci->dtype, ci->data, ib + i * is) < 0) continue;
            const float v = ldt(cv, vb + i * vs);
            if (v > mx) mx = v;
        }
        if (mx == -INFINITY) { st_int(tok->dtype, tok->data, offlin(tok, m), -1); continue; }

        float sum = 0.0f;
        for (int64_t i = 0; i < n_cand; ++i) {
            if (ld_int(ci->dtype, ci->data, ib + i * is) < 0) continue;
            sum += ref_exp(ldt(cv, vb + i * vs) - mx);
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        const float u = rad_sample_uniform01(R.p[m].seed ^ RAD_SAMPLE_KEY_PICK, R.p[m].pos);

        /* Inverse CDF in candidate order, accumulating in f32. The f32 probabilities may sum to
         * less than u < 1, so a draw that falls through takes the last candidate above
         * -infinity. Grammar-masked tokens sit at the set's tail at -infinity; they add nothing to
         * `cum` and must never be the fallback. */
        float cum = 0.0f;
        int64_t chosen = -1, last = -1;
        for (int64_t i = 0; i < n_cand; ++i) {
            const int64_t id = ld_int(ci->dtype, ci->data, ib + i * is);
            if (id < 0) continue;
            const float v = ldt(cv, vb + i * vs);
            if (v > -INFINITY) last = id;
            cum += ref_exp(v - mx) * inv;
            if (u < cum) { chosen = id; break; }
        }
        st_int(tok->dtype, tok->data, offlin(tok, m), chosen >= 0 ? chosen : last);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_argmax */
/* Greedy, ties to the lower token id, as in top-k selection, so greedy and top-1 agree. */
int ref_sample_argmax(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *tok = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0 || numel(tok) < M) return RAD_E_SHAPE;

    REF_PAR_FOR(M * n_vocab)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        float best = -INFINITY;
        int64_t bi = -1;
        for (int64_t i = 0; i < n_vocab; ++i) {
            const float v = ldt(lg, lb + i * ls);
            if (bi < 0 || v > best) { best = v; bi = i; }
        }
        st_int(tok->dtype, tok->data, offlin(tok, m), bi);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_logprobs */
/* A summary of the raw distribution for rows with RAD_SP_LOGPROBS: the max, the sum of
 * exp(l - max), and the k largest as (id, logit) pairs, descending, ties to the lower id, ids as
 * floats (exact below 2^24). Read before any sampling stage, as OpenAI's and llama-server's
 * logprobs report. Rows without the flag are not written. Under a split vocabulary ids are global
 * (+ `vocab_off`) and the host merges ranks; an unfilled slot is id -1, logit -infinity. The sum is
 * in double. Without `params` every row is written (a drafter's candidate logprobs). */
int ref_sample_logprobs(const RadArgs* a, RadStream) {
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
    if (pr->data) RAD_REF_TRY(param_rows(pr, M, &R));

    REF_PAR_FOR(M * n_vocab * (k + 2))
    for (int64_t m = 0; m < M; ++m) {
        if (pr->data && !(R.p[m].flags & RAD_SP_LOGPROBS)) continue;
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        const int64_t sb = rowoff(st, m, w), ss = laststride(st);
        float mx = -INFINITY;
        for (int64_t i = 0; i < n_vocab; ++i) mx = std::max(mx, ldt(lg, lb + i * ls));
        double sum = 0.0;
        for (int64_t i = 0; i < n_vocab; ++i) sum += std::exp((double)ldt(lg, lb + i * ls) - mx);
        std::vector<std::pair<float, int64_t>> top;
        top.reserve((size_t)k + 1);
        auto above = [](const std::pair<float, int64_t>& x, const std::pair<float, int64_t>& y) {
            return x.first > y.first || (x.first == y.first && x.second < y.second);
        };
        for (int64_t i = 0; i < n_vocab; ++i) {
            const std::pair<float, int64_t> e{ ldt(lg, lb + i * ls), i };
            if ((int64_t)top.size() == k && !above(e, top.back())) continue;
            top.insert(std::upper_bound(top.begin(), top.end(), e, above), e);
            if ((int64_t)top.size() > k) top.pop_back();
        }
        stt(st, sb, mx);
        stt(st, sb + ss, (float)sum);
        for (int64_t j = 0; j < k; ++j) {
            const bool have_j = j < (int64_t)top.size();
            stt(st, sb + (2 + 2 * j) * ss, have_j ? (float)(top[(size_t)j].second + voff) : -1.0f);
            stt(st, sb + (3 + 2 * j) * ss, have_j ? top[(size_t)j].first : -INFINITY);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_merge_topk */
/* The vocab-parallel merge. lm_head is vocab-sharded, so each rank runs top-k over its shard and
 * the candidate sets are all-gathered: k * world_size pairs, 2 KiB at k=64 and world_size=4, versus
 * ~77 MiB for a full-logits gather (spec §9, §13).
 *
 * `gathered` is [M, world_size * n_cand] (token id, logit) pairs as raw u32 words, ids already
 * global. The output is the single-rank path's descending candidate set, so later stages are
 * unchanged. */
int ref_sample_merge_topk(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *gt = &a->t[0], *pr = &a->t[1], *ci = &a->t[2], *cv = &a->t[3];
    const int64_t n_cand = p_int(a, "n_cand", ci->rank >= 2 ? ci->shape[ci->rank - 1] : 0);
    if (n_cand <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(ci) / n_cand;
    if (M <= 0 || numel(cv) < M * n_cand) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(param_rows(pr, M, &R));
    /* Each entry packs a u32 id and an f32 logit in a u32 buffer, so the logit is read by bit
     * pattern (`ldt` would convert 0x40800000 to 1.08e9, not 4.0). */
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

    REF_PAR_FOR(M * n_in * 4)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t gb = rowoff(gt, m, per_row), gs = laststride(gt);
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);

        /* Selection sort over the gathered pairs, n_cand times, ties to the lower id: O(n*k) as
         * in single-rank top-k, no scratch.
         *
         * The row's own top_k is re-applied: the gather holds up to world_size * k candidates and
         * nothing later narrows them, which at top_k 1 would let sample_pick draw among several
         * instead of returning the argmax. <= 0 means the whole width. */
        const int32_t tk = R.p[m].top_k;
        const int64_t keep = (R.p[m].flags & RAD_SP_GREEDY) ? 0
                           : (tk > 0 && (int64_t)tk < n_cand) ? (int64_t)tk : n_cand;

        float taken_v = INFINITY;
        int64_t taken_i = -1;
        for (int64_t r = 0; r < n_cand; ++r) {
            if (r >= keep) {
                st_int(ci->dtype, ci->data, ib + r * is, -1);
                stt(cv, vb + r * vs, -INFINITY);
                continue;
            }
            float best = -INFINITY;
            int64_t best_id = -1;
            for (int64_t j = 0; j < n_in; ++j) {
                /* As int32, so the 0xffffffff hole reads as -1 and is skipped. */
                const int64_t id = (int32_t)ld_int(gt->dtype, gt->data, gb + 2 * j * gs);
                const float   v  = bits_to_f32(ld_int(gt->dtype, gt->data, gb + (2 * j + 1) * gs));
                if (id < 0) continue;
                const bool after = v < taken_v || (v == taken_v && id > taken_i);
                if (!after) continue;
                if (best_id < 0 || v > best || (v == best && id < best_id)) { best = v; best_id = id; }
            }
            st_int(ci->dtype, ci->data, ib + r * is, best_id);
            stt(cv, vb + r * vs, best_id >= 0 ? best : -INFINITY);
            if (best_id < 0) continue;
            taken_v = best;
            taken_i = best_id;
        }
    }
    return RAD_OK;
}
