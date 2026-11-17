/* rad_spechub.h -- the speculative coupling (linear MTP and DFlash2's block walk): wire rows and
 * the shared arithmetic.
 *
 * The core writes these rows and a kernel plugin reads them, so their layout is ABI. The helpers
 * are header-only and stateless, so the host reference, the device kernel and the sampler share
 * one definition of the wire and of the arithmetic that turns it into a token.
 *
 * A row lives in a U32 plane whose base and row stride are eight-byte aligned, so a state can be
 * loaded whole on the device.
 */
#ifndef RAD_SPECHUB_H
#define RAD_SPECHUB_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* The verify reads a RadSampleParams row. */
#include "rad_sample.h"

/* Linear MTP coupling (--draft-coupling on). One draw per drafting row, staged before the
 * serial passes; one state per (temporal round, row). `base` is the request's PRE-commit output
 * size, so (base, round) names one draft stream that no later step reuses. `card` is the step slot
 * the advance writes, checked against RadBatch::seq_ids. `mode` 1 samples x~Q; 0 keeps the greedy
 * proposal (temperature 0 or top_k 1). */
typedef struct RadMtpCoupleDraw {
    uint64_t seed, base, seq;
    int32_t  card;
    uint32_t mode;
} RadMtpCoupleDraw;
/* status 1: x sampled from the exact q[] stored here; 0: greedy proposal kept; -1: malformed
 * shortlist or row identity, greedy proposal kept. `pos` is the verify row's output position, bound
 * by the host after commit; the device writes 0. ids[count..7] = -1, q[count..7] = 0. */
typedef struct RadMtpCoupleState {
    RadMtpCoupleDraw draw;
    uint64_t pos;
    int32_t  status, round, x, count;
    int32_t  ids[8];
    double   q[8];
} RadMtpCoupleState;
#define RAD_MTP_COUPLE_DRAW_WORDS  (sizeof(RadMtpCoupleDraw) / 4)
#define RAD_MTP_COUPLE_STATE_WORDS (sizeof(RadMtpCoupleState) / 4)
#define RAD_MTP_COUPLE_MAX_ROUNDS 8
/* Draft stream of round r: seed ^ (KEY_DRAFT + r) at coordinate `base`. Acceptance and the
 * residual draw use RAD_SAMPLE_KEY_ACCEPT / RAD_SAMPLE_KEY_RESID at the verify position. */
#define RAD_MTP_COUPLE_KEY_DRAFT 0x4d5450434f55504cull
#ifdef __cplusplus
static_assert(sizeof(RadMtpCoupleDraw) == 32, "MTP coupling draw wire");
static_assert(sizeof(RadMtpCoupleState) == 152, "MTP coupling state wire");
static_assert(offsetof(RadMtpCoupleState, q) % 8 == 0, "MTP coupling q alignment");
#endif

/* ================================================================== the shared arithmetic
 *
 * Q is the FP64 softmax of an FP32 shortlist at temperature 1; P is the FP64 softmax of the
 * sampler's FP32 candidate logits; uniforms are 53-bit. A coupled row therefore realises the
 * filtered target law to FP64 rounding rather than sample_pick's FP32/24-bit law, and is not
 * seed-identical to the ordinary pick.
 */
#ifdef __cplusplus

#if defined(__HIPCC__)
#define RAD_SPECHUB_HD __host__ __device__
#else
#define RAD_SPECHUB_HD
#endif

static inline RAD_SPECHUB_HD double rad_spechub_positive(double a) { return a > 0 ? a : 0; }
static inline RAD_SPECHUB_HD bool rad_spechub_finite(double x) {
    return x == x && x <= 1.7976931348623157e308 && x >= -1.7976931348623157e308;
}

/* A 53-bit uniform in [0, 1): splitmix64 over (seed ^ key, pos), top 53 bits scaled by 2^-53. */
static inline RAD_SPECHUB_HD double rad_spechub_uniform(uint64_t seed, uint64_t pos, uint64_t key) {
    uint64_t z = (seed ^ key) * 0x9e3779b97f4a7c15ull + pos + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return (double)(z >> 11) * 0x1p-53;
}

/* Is `s` a state the coupling's propose kernel could have written for a sampled row? The verifier
 * and the engine's bind check refuse a row by this rule, on the device and on the host. */
static inline RAD_SPECHUB_HD bool rad_mtp_couple_state_valid(const RadMtpCoupleState& s,
                                                             int64_t vocab) {
    if (s.status != 1 || s.draw.mode != 1 || s.count < 1 || s.count > 8 || s.round < 1 ||
        s.round > RAD_MTP_COUPLE_MAX_ROUNDS)
        return false;
    double sum = 0;
    bool has_x = false;
    for (int i = 0; i < 8; ++i) {
        if (i >= s.count) {
            if (s.ids[i] != -1 || s.q[i] != 0) return false;
            continue;
        }
        if (s.ids[i] < 0 || s.ids[i] >= vocab || !rad_spechub_finite(s.q[i]) || !(s.q[i] > 0) ||
            s.q[i] > 1)
            return false;
        for (int j = 0; j < i; ++j)
            if (s.ids[j] == s.ids[i]) return false;
        sum += s.q[i];
        has_x |= s.ids[i] == s.x;
    }
    return has_x && (sum - 1) <= 1e-12 && (1 - sum) <= 1e-12;
}

/* The verify position of coupled round r (1-based): the coordinate the engine's bind and the
 * sampler's row check must agree on.
 *
 * The sampler stages row k of a request at output position `n_out + k`, where `n_out` is the
 * request's committed output length when the step is built; round r is verified at row r - 1. The
 * acceptance and residual uniforms are drawn at this position.
 *
 * `n_out` is read from the request after the drafting step's commit. It is not rebuilt from the
 * draft's pre-commit `base` plus the accepted count: the two differ when the commit emits fewer
 * tokens than the card accepted (a max_tokens or context cut) or a step is carried. */
static inline RAD_SPECHUB_HD uint64_t rad_mtp_couple_verify_pos(uint64_t n_out, int round) {
    return n_out + (round > 0 ? (uint64_t)(round - 1) : 0);
}

/* Would the verify row for draft index `k` (0-based) of a request with committed output length
 * `n_out` take `c` as its coupled state? Same request and seed, the round this row's draft came
 * from, the position it samples, the very token it proposes, and a state the proposer could have
 * written. Anything else is a stale state from another step, and coupling against it would apply a
 * law this step never drew.
 *
 * Asked by the sampler's step build, against the state the engine bound. The engine's own bind
 * does not call it: it checks the readback against the card (draw, round, x, status) and writes
 * the position with rad_mtp_couple_verify_pos, the formula here that both sides must agree on. */
static inline RAD_SPECHUB_HD bool rad_mtp_couple_binds(const RadMtpCoupleState& c, uint64_t seq,
                                                       uint64_t seed, uint64_t n_out, int k,
                                                       int32_t draft_x, int64_t vocab) {
    return c.status == 1 && c.draw.seq == seq && c.draw.seed == seed && c.round == k + 1 &&
           c.pos == rad_mtp_couple_verify_pos(n_out, k + 1) && c.x == draft_x &&
           rad_mtp_couple_state_valid(c, vocab);
}

/* Build the stored Q from a shortlist of eight (id, score) slots.
 *
 * Holes (id < 0) and -inf scores carry no mass. An entry whose FP64 weight or normalized q
 * underflows to zero is dropped, so the stored q[] is exactly the law the draw walks. False on an
 * out-of-range id, NaN/+inf score, duplicate id or empty support (s->count = 0). */
static inline RAD_SPECHUB_HD bool rad_mtp_couple_build_q(const int32_t* ids, const float* val,
                                                         int64_t vocab, RadMtpCoupleState* s) {
    int32_t kid[8];
    float   kv[8];
    int     n = 0;
    s->count = 0;
    for (int i = 0; i < 8; ++i) { s->ids[i] = -1; s->q[i] = 0; }
    for (int i = 0; i < 8; ++i) {
        if (ids[i] < 0) continue;
        if (ids[i] >= vocab || val[i] != val[i] || val[i] == INFINITY) return false;
        if (val[i] == -INFINITY) continue;
        for (int j = 0; j < n; ++j)
            if (kid[j] == ids[i]) return false;
        kid[n] = ids[i];
        kv[n++] = val[i];
    }
    if (!n) return false;
    float top = kv[0];
    for (int j = 1; j < n; ++j)
        if (kv[j] > top) top = kv[j];
    double w[8], sum = 0;
    for (int j = 0; j < n; ++j) { w[j] = ::exp((double)kv[j] - top); sum += w[j]; }
    for (int j = 0; j < n; ++j) {
        const double q = w[j] / sum;
        if (q > 0) { s->ids[s->count] = kid[j]; s->q[s->count++] = q; }
    }
    return s->count > 0;
}

/* Inverse CDF over the stored q with a uniform in [0, 1); index into ids. */
static inline RAD_SPECHUB_HD int rad_mtp_couple_pick(const RadMtpCoupleState& s, double u) {
    double total = 0;
    for (int i = 0; i < s.count; ++i) total += s.q[i];
    double c = u * total;
    for (int i = 0; i < s.count; ++i) { c -= s.q[i]; if (c < 0) return i; }
    return s.count - 1;
}

/* Residual mass r(y) = max(0, P(y) - Q(y)) of one candidate. Q tokens use their P aggregated over
 * duplicate candidates, counted once at the first occurrence (`seen`). */
static inline RAD_SPECHUB_HD double rad_mtp_couple_mass(const RadMtpCoupleState& s, const double* pq,
                                                        int32_t id, float v, float top, double sum,
                                                        unsigned* seen) {
    if (id < 0 || v == -INFINITY) return 0;
    for (int k = 0; k < s.count; ++k)
        if (s.ids[k] == id) {
            if (*seen & (1u << k)) return 0;
            *seen |= 1u << k;
            return rad_spechub_positive(pq[k] - s.q[k]);
        }
    return ::exp((double)v - top) / sum;
}

/* The maximal coupling of the target's law P (over the surviving candidates) with the draft's Q.
 * Writes the emitted token to *out and returns true, or returns false for an invalid or empty
 * target. u_accept and u_resid are independent uniforms in [0, 1).
 *
 * The draft x is kept when P dominates Q on every Q token or when u_accept * Q(x) < P(x); otherwise
 * the token is drawn from the residual (P - Q)+. The emitted token is distributed as P, and it
 * equals x with probability sum min(P, Q).
 *
 * If FP64 cancellation leaves the residual total R at 0 while some Q token has P < Q, P is sampled
 * directly in FP64 with u_resid. That branch depends on (P, Q) only, never on x or a uniform.
 *
 * R sums r over every candidate including x. The residual walk runs only when P(x) < Q(x), where
 * r(x) is exactly +0, so skipping x in the walk leaves its total equal to R bit for bit. */
static inline RAD_SPECHUB_HD bool rad_mtp_couple_decide(const RadMtpCoupleState& s,
                                                        const int32_t* ids, const float* val,
                                                        int64_t n, int64_t vocab, double u_accept,
                                                        double u_resid, int32_t* out) {
    float top = -INFINITY;
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] < 0) continue;
        if (ids[j] >= vocab || val[j] != val[j] || val[j] == INFINITY) return false;
        if (val[j] > top) top = val[j];
    }
    if (top == -INFINITY) return false;
    double sum = 0, pq[8] = {};
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] < 0 || val[j] == -INFINITY) continue;
        const double w = ::exp((double)val[j] - top);
        sum += w;
        for (int k = 0; k < s.count; ++k)
            if (ids[j] == s.ids[k]) { pq[k] += w; break; }
    }
    if (!(sum > 0) || !rad_spechub_finite(sum)) return false;
    int  ix = -1;
    bool dominated = true;
    for (int k = 0; k < s.count; ++k) {
        pq[k] /= sum;
        dominated = dominated && pq[k] >= s.q[k];
        if (s.ids[k] == s.x) ix = k;
    }
    if (ix < 0) return false;   /* unreachable for a rad_mtp_couple_state_valid state */
    unsigned seen = 0;
    double   resid = 0;
    for (int64_t j = 0; j < n; ++j)
        resid += rad_mtp_couple_mass(s, pq, ids[j], val[j], top, sum, &seen);
    if (!dominated && !(resid > 0 && rad_spechub_finite(resid))) {
        double  c = u_resid * sum;
        int32_t last = -1;
        for (int64_t j = 0; j < n; ++j)
            if (ids[j] >= 0 && val[j] != -INFINITY) {
                const double w = ::exp((double)val[j] - top);
                if (w > 0) { last = ids[j]; c -= w; if (c < 0) { *out = last; return true; } }
            }
        *out = last;
        return true;
    }
    if (pq[ix] >= s.q[ix] || u_accept * s.q[ix] < pq[ix]) { *out = s.x; return true; }
    /* Rejected, so P(x) < Q(x): not dominated, resid > 0, and some y != x has mass > 0 below. */
    double  c = u_resid * resid;
    int32_t last = -1;
    seen = 0;
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] == s.x) continue;
        const double m = rad_mtp_couple_mass(s, pq, ids[j], val[j], top, sum, &seen);
        if (m > 0) { last = ids[j]; c -= m; if (c < 0) { *out = last; return true; } }
    }
    *out = last;
    return true;
}

/* ================================================================== DFlash2's coupled walk
 *
 * The block drafter's counterpart of mtp_couple_propose (`dflash_couple_propose`). DFlash2 proposes
 * a block in one pass and `dflash_select` walks its candidate lattice greedily: at draft position l
 * every candidate c of that position scores
 *
 *     score[l, c] = unary[l, c] + sum_r pred[pid][r] * hp[l][r] * succ[cand[l, c]][r]
 *
 * with pid the anchor at l = 0 and the walk's own previous token after it. The coupled walk scores
 * the same lattice but SAMPLES each position instead of taking its argmax:
 *
 *     Q_l = the FP64 softmax of score[l, c] / q_temp over the eight best-scoring candidates whose
 *           ids are in the vocabulary (rad_mtp_couple_build_q), x_l ~ Q_l at the draw's
 *           (seed, base, KEY_DRAFT + l + 1), and pid = x_l for position l + 1.
 *
 * Q_l depends on the committed text (through the drafter's pass and the anchor) and on x_0..x_{l-1},
 * never on the target's verification, and it is stored exactly in the state row the verify reads,
 * which is all the maximal coupling needs to be exact (rad_mtp_couple_decide). The pairwise term is
 * what makes Q_l follow the walk's own earlier samples rather than the drafter's marginal.
 *
 * Bit-identical on host and device: each candidate's edge is RAD_DFLASH_EDGE_PARTS partial sums,
 * each a serial fmaf chain over a contiguous slice of the rank, added in order and then to the unary
 * score. The plugin builds with -ffp-contract=off, so nothing else is fused. */
#define RAD_DFLASH_EDGE_PARTS 8
#define RAD_DFLASH_COUPLE_MAX_K 32
#define RAD_DFLASH_COUPLE_MAX_R 512

static inline RAD_SPECHUB_HD float rad_dflash_bf2f(uint16_t h) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)h << 16;
    return c.f;
}

/* Part `j` of one candidate's edge: a[r] * succ_row[r] over r in [j*R/P, (j+1)*R/P), fmaf in order.
 * `a` is the candidate-independent half, pred[pid][r] * hp[r] (or pred[pid][r] with no hp, 0 for a
 * predecessor outside the vocabulary). */
static inline RAD_SPECHUB_HD float rad_dflash_edge_part(const float* a, const uint16_t* succ_row,
                                                        int64_t R, int j) {
    const int64_t w = R / RAD_DFLASH_EDGE_PARTS;
    float acc = 0.0f;
    for (int64_t r = (int64_t)j * w; r < (int64_t)(j + 1) * w; ++r)
        acc = fmaf(a[r], rad_dflash_bf2f(succ_row[r]), acc);
    return acc;
}

/* One position of the walk, given its K scores and ids (the ids as the candidate plane holds them).
 * Fills `*s` as mtp_couple_propose fills a round's state and returns the proposed token:
 *
 *   mode 0 (a greedy request): status 0, x = `greedy_x`, the token dflash_select already wrote;
 *   mode 1, its card's draw and a Q that builds: status 1, x ~ Q;
 *   mode 1 otherwise: status -1, x = the walk's argmax (ties to the lower index, dflash_select's
 *   rule over these scores), and the ordinary verify applies to it.
 *
 * The eight are taken in descending score, ties to the lower index, among candidates with an id
 * in [0, vocab); a NaN score is never taken (build_q would refuse it). */
static inline RAD_SPECHUB_HD int32_t rad_dflash_couple_step(const float* sc, const int32_t* id,
                                                            int K, float q_temp,
                                                            const RadMtpCoupleDraw& d, bool card_ok,
                                                            int round, int64_t vocab,
                                                            int32_t greedy_x, RadMtpCoupleState* s) {
    s->draw = d;
    s->pos = 0;
    s->round = round;
    s->count = 0;
    for (int i = 0; i < 8; ++i) { s->ids[i] = -1; s->q[i] = 0; }
    if (d.mode != 1) {
        s->status = 0;
        s->x = greedy_x;
        return greedy_x;
    }
    s->status = -1;
    if (card_ok) {
        int32_t kid[8];
        float   kv[8];
        unsigned taken = 0;
        int n = 0;
        for (; n < 8; ++n) {
            int best = -1;
            for (int c = 0; c < K; ++c) {
                if ((taken >> c) & 1u) continue;
                if (id[c] < 0 || id[c] >= vocab || sc[c] != sc[c]) continue;
                if (best < 0 || sc[c] > sc[best]) best = c;
            }
            if (best < 0) break;
            taken |= 1u << best;
            kid[n] = id[best];
            kv[n] = sc[best] / q_temp;
        }
        for (int i = n; i < 8; ++i) { kid[i] = -1; kv[i] = -INFINITY; }
        if (n > 0 && rad_mtp_couple_build_q(kid, kv, vocab, s)) {
            const double u = rad_spechub_uniform(d.seed, d.base,
                                                 RAD_MTP_COUPLE_KEY_DRAFT + (uint64_t)round);
            s->x = s->ids[rad_mtp_couple_pick(*s, u)];
            s->status = 1;
            return s->x;
        }
        s->count = 0;
        for (int i = 0; i < 8; ++i) { s->ids[i] = -1; s->q[i] = 0; }
    }
    int   best = 0;
    float bv = -INFINITY;
    for (int c = 0; c < K; ++c)
        if (sc[c] > bv) { bv = sc[c]; best = c; }
    s->x = id[best];
    return s->x;
}

/* One sequence's whole walk on the host: the reference the device kernel is checked against, and
 * what the tests drive. `cand`/`unary` are `steps` rows at pitches `ld_cand`/`ld_unary`, `hp`
 * `steps` rows at `ld_hp` (null: ones), `tokens` the `steps` tokens dflash_select wrote (rewritten
 * for a sampled row), `state` round 1's row with `st_round` u32 words between rounds. K <= 32, R a
 * multiple of RAD_DFLASH_EDGE_PARTS and at most RAD_DFLASH_COUPLE_MAX_R (the plugin's validation).
 * `pred`/`succ` are [vocab, R]. */
static inline void rad_dflash_couple_walk(const int32_t* cand, const float* unary, int64_t ld_cand,
                                          int64_t ld_unary, const uint16_t* hp, int64_t ld_hp,
                                          int32_t anchor, const uint16_t* pred,
                                          const uint16_t* succ, int K,
                                          int64_t R, int64_t vocab, int64_t steps, float q_temp,
                                          const RadMtpCoupleDraw& d, bool card_ok, int32_t* tokens,
                                          uint32_t* state, int64_t st_round) {
    float   a[RAD_DFLASH_COUPLE_MAX_R], sc[RAD_DFLASH_COUPLE_MAX_K] = {};
    int32_t pid = anchor;
    for (int64_t l = 0; l < steps; ++l) {
        RadMtpCoupleState* s = reinterpret_cast<RadMtpCoupleState*>(state + l * st_round);
        const int32_t* id = cand + l * ld_cand;
        if (d.mode == 1) {
            const float* un = unary + l * ld_unary;
            const bool ok = pid >= 0 && pid < vocab;
            for (int64_t r = 0; r < R; ++r)
                a[r] = ok ? rad_dflash_bf2f(pred[(int64_t)pid * R + r]) *
                                (hp ? rad_dflash_bf2f(hp[l * ld_hp + r]) : 1.0f)
                          : 0.0f;
            for (int c = 0; c < K; ++c) {
                float acc = 0.0f;
                if (id[c] >= 0 && id[c] < vocab)
                    for (int j = 0; j < RAD_DFLASH_EDGE_PARTS; ++j)
                        acc += rad_dflash_edge_part(a, succ + (int64_t)id[c] * R, R, j);
                sc[c] = un[c] + acc;
            }
        }
        pid = rad_dflash_couple_step(sc, id, K, q_temp, d, card_ok, (int)l + 1, vocab, tokens[l], s);
        tokens[l] = pid;
    }
}

#undef RAD_SPECHUB_HD
#endif  /* __cplusplus */
#endif  /* RAD_SPECHUB_H */
