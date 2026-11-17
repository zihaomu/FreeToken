/* ref_ngram.cpp -- the n-gram row ids of Qwen4-Exp's Per-Layer Embedding (PLE).
 *
 * ============================== What it is ==============================
 *
 * PLE gives one layer a second embedding table addressed by a hash of the last `ngram` tokens
 * rather than the token. Qwen3.8-Flash-Next's has 320,001,536 rows of 160 bf16 (102 GB, 51 of the
 * model's 125 billion parameters), read sixteen rows per token. This op computes those row ids; the
 * gather itself is `gather_rows` over [M * heads] indices.
 *
 *   mixed[2gram] = t0*m0 ^ t1*m1                 heads 0..7
 *   mixed[3gram] = t0*m0 ^ t1*m1 ^ t2*m2         heads 8..15
 *   ids[h]       = mixed[ngram(h)] % vocab[h] + offset[h]
 *
 * `m`, `vocab` and `offset` come from the checkpoint (`layer_multipliers`,
 * `ngram_heads_vocab_sizes`, `ngram_heads_offsets`). Upstream derives them from a splitmix64 seed
 * and a prime search; re-deriving them here would be a second source for numbers the container
 * already carries.
 *
 * ============================== Two non-traps ==============================
 *
 * The hash is non-negative, so `%` is unambiguous even though upstream's `torch.remainder` floors
 * (remainder(-5, 3) == 1, C gives -2). `_build_layer_multipliers` bounds each multiplier by
 * `(2^63-1) // vocab_size`, so a real token's product and any XOR of such products have the sign
 * bit clear: `mixed` is in [0, 2^63).
 *
 * The EOS rule collapses to two comparisons. Upstream's `_shift_right_ignore_eos` (a cummax over
 * `where(ids == eos, positions, -1)`, a segment start, and
 * `valid = (position_in_segment >= shift) & (p - shift >= 0)`) reduces to "no eos in [p-s, p-1]";
 * where it would substitute eos because `tok[p-s]` is itself eos, the value is unchanged. So:
 *
 *     t0 = tok[p]
 *     t1 = p >= 1                ? tok[p-1] : eos
 *     t2 = (p >= 2 && t1 != eos) ? tok[p-2] : eos
 *
 * and in general `t[s] = tok[p-s]` until a shift falls back to eos, after which all deeper shifts
 * are eos. A serving engine needs only the `ngram-1` committed ids before the step.
 *
 * ============================== Where those ids come from ==============================
 *
 * An INOUT state: a rolling window of `ngram - 1 + n_spec` ids per slot, read at
 * `num_accepted - 1`, with exactly `ple_conv`'s contract (ref_ple.cpp). With `num_accepted`
 * present (a decode step whose drafts may be rejected) the window is rewritten as gdn_conv_update
 * does: old ids shifted down by one, then every id of the step, so a next step that kept k reads
 * at k - 1. Absent, every id commits and the last `ngram - 1` land at offset zero.
 *
 * It is a state rather than a plain input although no rejection can reach the step's predecessors:
 * nothing in the engine derives "the ids before this step" (`DeriveInput` carries positions and
 * cu_seqlens, no token ids), and the layer's dilated convolution already passes `num_accepted`,
 * `cache_idx` and `has_init` under the same contract. RAD_KV_CONV ("per-sequence, a rolling window
 * of width-1 + num_spec entries") fits, and `conv_state_index` derives the read cursor.
 *
 * The cold value is eos, not zero (zero is a real token id): `has_init` 0 fills the window with
 * `eos`.
 */
#include "ref_common.h"
#include "ref_ops.h"

using namespace ref;

namespace {

/* Torch's int64 multiply and remainder without undefined behaviour: signed overflow is UB in C++
 * but wraps in torch, so the multiply goes through uint64; `torch.remainder` floors where C
 * truncates. On a real checkpoint neither case arises (see the file header); these give torch's
 * answer for drawn test inputs that no vocabulary bounds. */
inline int64_t wrap_mul(int64_t a, int64_t b) {
    return (int64_t)((uint64_t)a * (uint64_t)b);
}
inline int64_t floor_mod(int64_t a, int64_t b) {    /* b > 0 */
    const int64_t r = a % b;
    return r < 0 ? r + b : r;
}

}  // namespace

/* ids[m][h] = (hash of the ngram ending at m) % vocab[h] + offset[h].
 *
 *   tok          [M]              i32/i64  the step's token ids, sequences back to back
 *   state        [n_slots][W]     i32      inout: a rolling window of committed ids, W = ctx +
 *                                          n_spec, slot i holding the id at relative position
 *                                          i - ctx from the step's first row
 *   cu           [n_seq+1]        i32
 *   mult         [ngram]          i64  w
 *   vocab_sizes  [heads]          i64  w
 *   offsets      [heads]          i64  w
 *   cache_idx    [n_seq]          i32  ?   the state slot per sequence
 *   has_init     [n_seq]          i32  ?   0 = the window is eos, not what the slot last held
 *   num_accepted [n_seq]          i32  ?   one-based; shifts the read offset, and its presence
 *                                          makes the write-back the decode layout. Absent reads
 *                                          at offset zero and leaves the prefill layout.
 *   ids          [M][heads]       i32      out
 *   ahead_ids    [A][heads]       i32  ?   out: `tok` then holds A more tokens after the step's,
 *                                          continuing the last sequence, and these are their ids
 *                                          as the step that runs them will compute them. The
 *                                          window is the step's own.
 */
int ref_ngram_ids(const RadArgs* a, RadStream) {
    /* Eleven positions, four optional, so the optionals are tested one by one; `have()` would
     * refuse an absent optional operand. */
    if (a->n_t < 10) return RAD_E_INVAL;
    const RadTensor* tok   = t_in(a, 0);
    const RadTensor* st    = t_in(a, 1);
    const RadTensor* cu    = t_in(a, 2);
    const RadTensor* mult  = t_in(a, 3);
    const RadTensor* vsz   = t_in(a, 4);
    const RadTensor* off_  = t_in(a, 5);
    const RadTensor* sidx  = t_in(a, 6);      /* optional */
    const RadTensor* hini  = t_in(a, 7);      /* optional */
    const RadTensor* nacc  = t_in(a, 8);      /* optional */
    const RadTensor* ids   = t_in(a, 9);
    const RadTensor* ahead = a->n_t > 10 ? t_in(a, 10) : nullptr;     /* optional */
    if (!tok || !st || !cu || !mult || !vsz || !off_ || !ids) return RAD_E_INVAL;

    const int64_t heads = p_int(a, "heads", 0);
    if (heads <= 0) return RAD_E_SHAPE;
    /* The token vector holds the call's tokens plus the `ahead` tokens; `M` is the band (see
     * band_rows in ref_common.h). */
    const int64_t A     = ahead ? numel(ahead) / heads : 0;
    const int64_t M     = band_rows(numel(tok) - A, p_int(a, "M", 0));
    const int64_t ngram = p_int(a, "ngram", 0);
    const int64_t eos   = p_int(a, "eos", -1);
    if (M <= 0 || ngram < 2 || eos < 0) return RAD_E_SHAPE;
    /* Heads are `ngram - 1` blocks of `heads_per_ngram`, block b using the (b+2)-gram, so the
     * head count must divide evenly. */
    if (heads % (ngram - 1)) return RAD_E_SHAPE;
    const int64_t per = heads / (ngram - 1);
    const int64_t ctx = ngram - 1;
    if (ngram > 8) return RAD_E_UNSUPPORTED;      /* the shift window lives in a stack frame */

    if (numel(mult) < ngram || numel(vsz) < heads || numel(off_) < heads) return RAD_E_SHAPE;
    if (numel(ids) < M * heads) return RAD_E_SHAPE;
    for (int64_t h = 0; h < heads; ++h)
        if (ld_int(vsz->dtype, vsz->data, h) <= 0 || ld_int(off_->dtype, off_->data, h) < 0)
            return RAD_E_SHAPE;

    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    const int64_t W = st->rank >= 2 ? st->shape[st->rank - 1] : ctx;
    if (W < ctx) return RAD_E_SHAPE;
    /* Check every sequence before writing any state: its rows, the read window, and the window a
     * decode step leaves (`ctx - 1` ids of history plus every id of the step). */
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t o = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, s)) - 1 : 0;
        if (o < 0) return RAD_E_INVAL;
        if (o + ctx > W) return RAD_E_SHAPE;
        const int64_t bos = cu_at(cu, s), eot = cu_at(cu, s + 1);
        if (bos < 0 || eot < bos || eot > M) return RAD_E_SHAPE;
        if (nacc && eot > bos && ctx - 1 + (eot - bos) > W) return RAD_E_SHAPE;
    }

    const int64_t ids_ld = rowoff(ids, 1, heads);
    const int64_t ahd_ld = ahead ? rowoff(ahead, 1, heads) : 0;

    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t bos = cu_at(cu, s);
        const int64_t T = cu_at(cu, s + 1) - bos;
        if (T <= 0) continue;
        const int64_t slot = sidx ? ld_int(sidx->dtype, sidx->data, offlin(sidx, s)) : s;
        if (slot < 0) continue;
        const bool warm = !hini || ld_int(hini->dtype, hini->data, offlin(hini, s)) != 0;
        const int64_t o = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, s)) - 1 : 0;

        /* The `ctx` committed ids before this step. A cold slot's window is eos, not a recycled
         * slot's previous contents (and not zero, a real token). */
        int64_t prev[8];
        for (int64_t i = 0; i < ctx; ++i)
            prev[i] = warm ? ld_int(st->dtype, st->data, rowoff(st, slot, W) + o + i) : eos;

        /* The sequence's history: `ctx` predecessors, then its own rows. Row `t` is history
         * position `ctx + t`, so every shift is in range without a bounds test. */
        auto hist = [&](int64_t p) -> int64_t {
            return p < ctx ? prev[p] : ld_int(tok->dtype, tok->data, offlin(tok, bos + p - ctx));
        };

        /* The last sequence runs on into the prompt that follows, whose ids land in `ahead`. */
        const int64_t reach = T + (ahead && s + 1 == n_seq ? A : 0);
        for (int64_t t = 0; t < reach; ++t) {
            const int64_t p = ctx + t;

            /* The shifted tokens under the collapsed eos rule (file header): once a shift falls
             * back to eos, every deeper one does too. */
            int64_t sh[8];
            sh[0] = hist(p);
            bool cut = false;
            for (int64_t k = 1; k < ngram; ++k) {
                if (cut || sh[k - 1] == eos) { cut = true; sh[k] = eos; continue; }
                sh[k] = hist(p - k);
            }

            /* One mix per n-gram order, extended in place: the (b+2)-gram's mix is the
             * (b+1)-gram's XOR one more term, as in upstream's inner loop. */
            int64_t mixed = 0;
            int64_t blk = -1;
            for (int64_t k = 0; k < ngram; ++k) {
                const int64_t term = wrap_mul(sh[k], ld_int(mult->dtype, mult->data, k));
                mixed = k == 0 ? term : (mixed ^ term);
                if (k == 0) continue;            /* a 1-gram addresses no head */
                ++blk;                           /* k = 1 is the 2-gram block, k = 2 the 3-gram */
                for (int64_t j = 0; j < per; ++j) {
                    const int64_t h = blk * per + j;
                    const int64_t v = ld_int(vsz->dtype, vsz->data, h);
                    const int64_t ofh = ld_int(off_->dtype, off_->data, h);
                    if (t < T)
                        st_int(ids->dtype, ids->data, (bos + t) * ids_ld + h,
                               floor_mod(mixed, v) + ofh);
                    else
                        st_int(ahead->dtype, ahead->data, (t - T) * ahd_ld + h,
                               floor_mod(mixed, v) + ofh);
                }
            }
        }

        /* Write the window back after all reads, by ple_conv's rule: a decode step (`num_accepted`
         * present) shifts the old ids down by one and appends every id it ran; a step that
         * commits everything leaves its last `ctx` ids at offset zero. */
        const int64_t base = rowoff(st, slot, W);
        if (nacc) {
            for (int64_t i = 0; i + 1 < ctx; ++i)
                st_int(st->dtype, st->data, base + i, prev[i + 1]);
            for (int64_t t = 0; t < T; ++t)
                st_int(st->dtype, st->data, base + ctx - 1 + t,
                       ld_int(tok->dtype, tok->data, offlin(tok, bos + t)));
        } else {
            for (int64_t i = 0; i < ctx; ++i) {
                const int64_t src = T - ctx + i;
                const int64_t v = src >= 0 ? ld_int(tok->dtype, tok->data, offlin(tok, bos + src))
                                           : prev[src + ctx];
                st_int(st->dtype, st->data, base + i, v);
            }
        }
    }
    return RAD_OK;
}
