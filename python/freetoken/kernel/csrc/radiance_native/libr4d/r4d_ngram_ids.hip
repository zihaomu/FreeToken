// r4d_ngram_ids.hip -- the hashed row ids of Qwen4-Exp's Per-Layer Embedding.
//
//   mixed(k) = t0*mult[0] ^ t1*mult[1] ^ ... ^ tk*mult[k]        int64, wrapping
//   ids[h]   = mixed(b+1) % vocab_sizes[h] + offsets[h]          h in block b
//
// `heads` is laid out as `ngram - 1` equal blocks, block b using the (b+2)-gram, and `t_s` is the
// token `s` positions back, or the EOS id if the shift would cross one. libref/ref_ngram.cpp
// derives that EOS rule (a cummax over segment starts reduces to "once a shift falls back, every
// deeper one does"); docs/OPS.md holds the operand contract.
//
// One thread a token, looping its heads. A token's heads share the shifted tokens, the EOS walk
// and the `ngram - 1` hash mixes and differ only in the modulus, so a thread per head would redo
// that work per head, and at decode (one token) would be partial waves either way. At prefill the
// grid is thousands of tokens; at decode the kernel is negligible beside the gather that follows.
//
// `mixed % vocab_sizes[h]` is a 64-bit division by a runtime value (~70 cycles on this part),
// sixteen per token. Magic-number division would need the architecture to compute a multiplier
// and shift per head and pass them as a second weight; not worth it at this head count, but the
// change to make if a model hashes at prefill width with many more heads.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_emit.h"

#include <cstdint>

namespace {

/* The same two operations libref spells out: signed overflow is UB, and torch's `remainder` floors
 * where C truncates. On a real checkpoint neither fires (the multipliers keep a real token's
 * product non-negative), but kernel and oracle must also agree on test inputs. */
__device__ __forceinline__ long long ng_wrap_mul(long long a, long long b) {
    return (long long)((unsigned long long)a * (unsigned long long)b);
}
__device__ __forceinline__ long long ng_floor_mod(long long a, long long b) {
    const long long r = a % b;
    return r < 0 ? r + b : r;
}

enum { kNgMaxShift = 8, kNgThreads = 256 };

/* One workgroup per (token tile, sequence), with the window write-back in the same launch.
 *
 * `s` is blockIdx.y. The fold is legal because only a token with `t - bos < ctx` reads past its
 * sequence's first row into the window, and those are the first `ctx` tokens, all in tile 0. So
 * tile 0's block is the only reader of state[slot]; its threads are ordered by __syncthreads(), and
 * thread 0 writes the new window after it. No other block touches those bytes.
 *
 * At decode this op's cost is the dispatch itself (flat from M=1 to M=512), so avoiding a second
 * launch for the write-back is most of the saving. */
#define NGRAM_IDS_PARAMS                                                                        \
    const int32_t* __restrict__ tok, int32_t* __restrict__ state,                               \
    const int32_t* __restrict__ cu, const long long* __restrict__ mult,                         \
    const long long* __restrict__ vsz, const long long* __restrict__ off,                       \
    const int32_t* __restrict__ sidx, const int32_t* __restrict__ hini,                         \
    const int32_t* __restrict__ nacc, int32_t* __restrict__ ids, uint32_t heads,                \
    uint32_t ngram, uint32_t per, uint32_t W, long long eos, uint32_t ids_ld,                   \
    int32_t* __restrict__ ahead, uint32_t n_ahead, uint32_t ahead_ld
__device__ __forceinline__ void ngram_ids_body(NGRAM_IDS_PARAMS) {

    const uint32_t s   = blockIdx.y;
    const uint32_t bos = (uint32_t)cu[s];
    const uint32_t eot = (uint32_t)cu[s + 1];
    const uint32_t ctx = ngram - 1u;
    /* Only block-uniform exits before the barrier below: an empty sequence and a negative slot are
     * properties of the block, so all its threads leave together. */
    if (eot <= bos) return;
    const int32_t slot = sidx ? sidx[s] : (int32_t)s;
    if (slot < 0) return;

    const uint32_t slen = eot - bos;
    const bool warm = !hini || hini[s] != 0;
    const long long o = nacc ? (long long)nacc[s] - 1 : 0;
    const uint32_t lt = blockIdx.x * kNgThreads + threadIdx.x;   /* this thread's row, in-sequence */

    /* The following prompt continues the last sequence: its tokens sit in `tok` after the step's,
     * so a row past the sequence's end hashes exactly as it will when the next step runs it, and
     * lands in `ahead` instead of `ids`. The window below is the step's own and never sees them. */
    const uint32_t reach = slen + ((ahead && s + 1u == gridDim.y) ? n_ahead : 0u);
    if (lt < reach) {
        const uint32_t t = bos + lt;
        /* The `ctx` committed ids before this sequence's first row. A cold slot's window is EOS,
         * not its stale contents and not zero (a real token id). */
        long long prev[kNgMaxShift];
        for (uint32_t i = 0; i < ctx; ++i)
            prev[i] = warm ? (long long)state[(long long)slot * W + o + i] : eos;

        /* `p` indexes the history `prev ++ this sequence's rows`, so row `lt` sits at `ctx + lt`
         * and every shift is in range without a bounds test. */
        const uint32_t p = ctx + lt;
        long long sh[kNgMaxShift];
        sh[0] = (long long)tok[t];
        bool cut = false;
        for (uint32_t k = 1; k < ngram; ++k) {
            if (cut || sh[k - 1] == eos) { cut = true; sh[k] = eos; continue; }
            const uint32_t q = p - k;
            sh[k] = q < ctx ? prev[q] : (long long)tok[bos + (q - ctx)];
        }

        /* One mix per n-gram order, extended in place: the (b+2)-gram's mix is the (b+1)-gram's
         * XOR one more term, as in the reference's inner loop. */
        long long mixed = 0;
        int blk = -1;
        for (uint32_t k = 0; k < ngram; ++k) {
            const long long term = ng_wrap_mul(sh[k], mult[k]);
            mixed = k == 0 ? term : (mixed ^ term);
            if (k == 0) continue;
            ++blk;
            int32_t* const row = lt < slen ? ids + (long long)t * ids_ld
                                           : ahead + (long long)(lt - slen) * ahead_ld;
            for (uint32_t j = 0; j < per; ++j) {
                const uint32_t h = (uint32_t)blk * per + j;
                row[h] = (int32_t)(ng_floor_mod(mixed, vsz[h]) + off[h]);
            }
        }
    }

    /* The window this step leaves, in libref/ref_ngram.cpp's two layouts.
     *
     * With `num_accepted` this is a decode step whose drafts a verify may reject: the window shifts
     * down by one and every id of the step follows it, so slot i holds position i + 1 of
     * (window read ++ step). A next step that kept k of these ids reads at k - 1 and finds the
     * `ctx` ids ending at the k-th. Slot i is read from o + i + 1, past every slot already written,
     * so the ascending walk is safe in place. Slots past W are not written; the manager sizes the
     * window ctx + n_spec, enough for a step of 1 + n_spec ids.
     *
     * Without it every id commits and the last `ctx` land at offset zero.
     *
     * One thread, not one per slot: a thread per (sequence, k) races within the sequence (at
     * ctx = 2 and slen = 1, thread 0 reads slot o+1 while thread 1 writes slot 1), and at decode
     * the serial loop is only ctx - 1 + slen stores. */
    __syncthreads();
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int32_t* const win = state + (long long)slot * W;
    if (nacc) {
        for (uint32_t k = 0; k + 1 < ctx; ++k)
            win[k] = warm ? win[o + k + 1] : (int32_t)eos;
        for (uint32_t t = 0; t < slen && ctx - 1u + t < W; ++t) win[ctx - 1u + t] = tok[bos + t];
        return;
    }
    long long keep[kNgMaxShift];
    for (uint32_t k = 0; k < ctx; ++k) {
        const long long src = (long long)slen - (long long)ctx + (long long)k;
        keep[k] = src >= 0 ? (long long)tok[bos + src]
                           : (warm ? (long long)win[o + src + ctx] : eos);
    }
    for (uint32_t k = 0; k < ctx; ++k) win[k] = (int32_t)keep[k];
}

/* The kernel around the body, whose blocks return where their work ends: CH `r4dchain` is the
 * chained form (r4d_chain.h). */
template <class CH>
__global__ __launch_bounds__(kNgThreads)
void ngram_ids_kernel(NGRAM_IDS_PARAMS) {
    if (CH::on) r4d_chain_wait();
    ngram_ids_body(tok, state, cu, mult, vsz, off, sidx, hini, nacc, ids, heads, ngram, per, W, eos, ids_ld, ahead, n_ahead, ahead_ld);
    if (CH::on) r4d_chain_done();
}
#undef NGRAM_IDS_PARAMS

/* ngram_hint: the row ids of each sequence's NEWEST token on a draft head's pass, ahead of the
 * verify step that will run it, so the rows can be fetched while the head drafts.
 *
 * A verify step's rows are the token the last verify chose and the head's proposals. The head
 * proposes one a pass: its history pass ends at the chosen token, each later round reads the
 * proposal before it. So all but the last proposal are known a pass or more before ngram_ids
 * hashes them, and a table read from a drive has that long to deliver their rows.
 *
 * One thread a sequence. `tail` holds, per state slot, the `ngram - 1` committed ids up to the
 * chosen token and the same count of ids up to the newest proposal, each newest last:
 *
 *   hist   the pass's last row is the chosen token; rows at its position are repeats of it
 *          (the history pass pads a run to the furthest the verify could have committed), and
 *          the rows before them hold the ids committed just before, in order. Ids further back
 *          come from the committed tail. Both tails then end at the chosen token.
 *   else   the row is the proposal before this round's; the ids before it are the chain tail,
 *          which then ends at it.
 *
 * The hash is ngram_ids'. The ids are a hint: a slot whose tail belongs to another request (the
 * first pass after a prompt) hashes other history and names rows nobody reads, which costs their
 * fetch and nothing else, so there is no cold-window rule here. */
#define NGRAM_HINT_PARAMS                                                                        \
    const int32_t* __restrict__ tok, const int32_t* __restrict__ pos,                            \
    const int32_t* __restrict__ cu, const long long* __restrict__ mult,                          \
    const long long* __restrict__ vsz, const long long* __restrict__ off,                        \
    const int32_t* __restrict__ sidx, int32_t* __restrict__ tail, int32_t* __restrict__ ids,     \
    uint32_t n_seq, uint32_t heads, uint32_t ngram, uint32_t per, long long eos,                 \
    uint32_t ids_ld, uint32_t hist, uint32_t slots
__device__ __forceinline__ void ngram_hint_body(NGRAM_HINT_PARAMS) {
    const uint32_t s = blockIdx.x * kNgThreads + threadIdx.x;
    if (s >= n_seq) return;
    const uint32_t bos = (uint32_t)cu[s], eot = (uint32_t)cu[s + 1];
    const int32_t slot = sidx ? sidx[s] : (int32_t)s;
    int32_t* const row = ids + (long long)s * ids_ld;
    /* A sequence with nothing to hint still gets ids a gather can read: its heads' first rows. */
    if (eot <= bos || slot < 0) {
        for (uint32_t h = 0; h < heads; ++h) row[h] = (int32_t)off[h];
        return;
    }
    const uint32_t ctx = ngram - 1u;
    /* Two slots that meet in `tail` (a state slot past its rows) swap hints, no more. */
    int32_t* const commit = tail + (long long)((uint32_t)slot % slots) * 2 * ctx;
    int32_t* const chain  = commit + ctx;

    /* `back[k]` is the id k + 1 places before the newest. */
    const uint32_t last = eot - 1u;
    const long long x = (long long)tok[last];
    long long back[kNgMaxShift];
    if (hist) {
        uint32_t first = last;
        const int32_t at = pos[last];
        while (first > bos && pos[first - 1u] == at) --first;
        const uint32_t have = first - bos;
        for (uint32_t k = 0; k < ctx; ++k)
            back[k] = k < have ? (long long)tok[first - 1u - k]
                               : (long long)commit[ctx - 1u - (k - have)];
    } else {
        for (uint32_t k = 0; k < ctx; ++k) back[k] = (long long)chain[ctx - 1u - k];
    }
    for (uint32_t k = 0; k < ctx; ++k) {
        const int32_t v = k + 1u < ctx ? (int32_t)back[ctx - 2u - k] : (int32_t)x;
        if (hist) commit[k] = v;
        chain[k] = v;
    }

    long long sh[kNgMaxShift];
    sh[0] = x;
    bool cut = false;
    for (uint32_t k = 1; k < ngram; ++k) {
        if (cut || sh[k - 1] == eos) { cut = true; sh[k] = eos; continue; }
        sh[k] = back[k - 1u];
    }
    long long mixed = 0;
    for (uint32_t k = 0; k < ngram; ++k) {
        const long long term = ng_wrap_mul(sh[k], mult[k]);
        mixed = k == 0 ? term : (mixed ^ term);
        if (k == 0) continue;
        for (uint32_t j = 0; j < per; ++j) {
            const uint32_t h = (k - 1u) * per + j;
            row[h] = (int32_t)(ng_floor_mod(mixed, vsz[h]) + off[h]);
        }
    }
}

template <class CH>
__global__ __launch_bounds__(kNgThreads)
void ngram_hint_kernel(NGRAM_HINT_PARAMS) {
    if (CH::on) r4d_chain_wait();
    ngram_hint_body(tok, pos, cu, mult, vsz, off, sidx, tail, ids, n_seq, heads, ngram, per, eos,
                    ids_ld, hist, slots);
    if (CH::on) r4d_chain_done();
}
#undef NGRAM_HINT_PARAMS

}  // namespace

enum { NG_TOK = 0, NG_STATE, NG_CU, NG_MULT, NG_VSZ, NG_OFF, NG_CIDX, NG_HINIT, NG_NACC, NG_IDS,
       NG_AHEAD };

extern "C" int r4d_rad_ngram_ids(const RadArgs* a, RadStream stream) {
    const RadTensor* tok  = r4d_opt(a, NG_TOK);
    const RadTensor* st   = r4d_opt(a, NG_STATE);
    const RadTensor* cu   = r4d_opt(a, NG_CU);
    const RadTensor* mult = r4d_opt(a, NG_MULT);
    const RadTensor* vsz  = r4d_opt(a, NG_VSZ);
    const RadTensor* off  = r4d_opt(a, NG_OFF);
    const RadTensor* cidx = r4d_opt(a, NG_CIDX);
    const RadTensor* hini = r4d_opt(a, NG_HINIT);
    const RadTensor* nacc = r4d_opt(a, NG_NACC);
    const RadTensor* ids  = r4d_opt(a, NG_IDS);
    const RadTensor* ahd  = r4d_opt(a, NG_AHEAD);
    if (!tok || !st || !cu || !mult || !vsz || !off || !ids) return R4D_NO(RAD_E_INVAL);
    if (tok->dtype != RAD_I32 || st->dtype != RAD_I32 || ids->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    if (cu->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (mult->dtype != RAD_I64 || vsz->dtype != RAD_I64 || off->dtype != RAD_I64)
        return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(tok) || !r4d_contig(st) || !r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);

    const long long M     = r4d_geti_or(a, "M", 0);
    const long long heads = r4d_geti_or(a, "heads", 0);
    const long long ngram = r4d_geti_or(a, "ngram", 0);
    const long long eos   = r4d_geti_or(a, "eos", -1);
    if (M <= 0 || heads <= 0 || ngram < 2 || eos < 0) return R4D_NO(RAD_E_SHAPE);
    if (ngram > kNgMaxShift) return R4D_NO(RAD_E_UNSUPPORTED);
    if (heads % (ngram - 1)) return R4D_NO(RAD_E_SHAPE);
    const long long per = heads / (ngram - 1), ctx = ngram - 1;

    long long rows = 0;
    const long long ids_ld = r4d_rows_pitch(ids, heads, &rows);
    if (ids_ld < 0 || rows <= 0) return R4D_NO(RAD_E_STRIDE);
    /* The row count comes from the operand: `M` is ranged, so the arg holds the band's upper bound
     * and a grid sized by it would run off the arena. */
    const long long T = rows;
    /* The prompt that follows: `ahead`'s rows, whose tokens `tok` holds after the step's. */
    long long A = 0, ahead_ld = heads;
    if (ahd) {
        if (ahd->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
        ahead_ld = r4d_rows_pitch(ahd, heads, &A);
        if (ahead_ld < 0) return R4D_NO(RAD_E_STRIDE);
    }
    if (r4d_numel(tok) < T + A) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(mult) < ngram || r4d_numel(vsz) < heads || r4d_numel(off) < heads)
        return R4D_NO(RAD_E_SHAPE);

    const long long N = r4d_numel(cu) - 1;
    if (N <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long W = st->rank >= 2 ? r4d_dim(st, (int)st->rank - 1) : ctx;
    if (W < ctx) return R4D_NO(RAD_E_SHAPE);

    /* (token tiles, sequences). Tiles are sized by the batch's total token count, an upper bound on
     * any one sequence's; a short sequence's extra blocks find their rows past its end. A
     * per-sequence count would mean reading `cu` on the host. */
    const uint32_t tiles = (uint32_t)((T + A + kNgThreads - 1) / kNgThreads);
#define R4D_NGI(CH_)                                                                             \
    R4D_EMIT(stream, ngram_ids_kernel<CH_>, dim3(tiles, (uint32_t)N), dim3(kNgThreads),         \
             0,                                                                                  \
             (const int32_t*)r4d_addr(tok), (int32_t*)r4d_addr(st),                              \
             (const int32_t*)r4d_addr(cu), (const long long*)r4d_addr(mult),                     \
             (const long long*)r4d_addr(vsz), (const long long*)r4d_addr(off),                   \
             cidx ? (const int32_t*)r4d_addr(cidx) : nullptr,                                    \
             hini ? (const int32_t*)r4d_addr(hini) : nullptr,                                    \
             nacc ? (const int32_t*)r4d_addr(nacc) : nullptr,                                    \
             (int32_t*)r4d_addr(ids), (uint32_t)heads, (uint32_t)ngram,                          \
             (uint32_t)per, (uint32_t)W, eos, (uint32_t)ids_ld,                                  \
             A > 0 ? (int32_t*)r4d_addr(ahd) : nullptr, (uint32_t)A,                             \
             (uint32_t)ahead_ld)
    const int rc = r4d_geti_or(a, "chain", 0) != 0 && T <= 64 ? R4D_NGI(r4dchain)
                                                               : R4D_NGI(R4dUnchained);
#undef R4D_NGI
    return rc;
}

enum { NH_TOK = 0, NH_POS, NH_CU, NH_MULT, NH_VSZ, NH_OFF, NH_CIDX, NH_TAIL, NH_IDS };

extern "C" int r4d_rad_ngram_hint(const RadArgs* a, RadStream stream) {
    const RadTensor* tok  = r4d_opt(a, NH_TOK);
    const RadTensor* pos  = r4d_opt(a, NH_POS);
    const RadTensor* cu   = r4d_opt(a, NH_CU);
    const RadTensor* mult = r4d_opt(a, NH_MULT);
    const RadTensor* vsz  = r4d_opt(a, NH_VSZ);
    const RadTensor* off  = r4d_opt(a, NH_OFF);
    const RadTensor* cidx = r4d_opt(a, NH_CIDX);
    const RadTensor* tail = r4d_opt(a, NH_TAIL);
    const RadTensor* ids  = r4d_opt(a, NH_IDS);
    if (!tok || !pos || !cu || !mult || !vsz || !off || !tail || !ids) return R4D_NO(RAD_E_INVAL);
    if (tok->dtype != RAD_I32 || pos->dtype != RAD_I32 || cu->dtype != RAD_I32 ||
        tail->dtype != RAD_I32 || ids->dtype != RAD_I32 || (cidx && cidx->dtype != RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    if (mult->dtype != RAD_I64 || vsz->dtype != RAD_I64 || off->dtype != RAD_I64)
        return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(tok) || !r4d_contig(pos) || !r4d_contig(cu) || !r4d_contig(tail))
        return R4D_NO(RAD_E_STRIDE);

    const long long heads = r4d_geti_or(a, "heads", 0);
    const long long ngram = r4d_geti_or(a, "ngram", 0);
    const long long eos   = r4d_geti_or(a, "eos", -1);
    const long long hist  = r4d_geti_or(a, "hist", 0);
    if (heads <= 0 || ngram < 2 || eos < 0) return R4D_NO(RAD_E_SHAPE);
    if (ngram > kNgMaxShift) return R4D_NO(RAD_E_UNSUPPORTED);
    if (heads % (ngram - 1)) return R4D_NO(RAD_E_SHAPE);
    const long long per = heads / (ngram - 1);

    /* A row of ids a sequence, counted from the operand (`M` is ranged). */
    const long long N = r4d_numel(cu) - 1;
    long long rows = 0;
    const long long ids_ld = r4d_rows_pitch(ids, heads, &rows);
    if (ids_ld < 0 || N <= 0 || rows < N) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(mult) < ngram || r4d_numel(vsz) < heads || r4d_numel(off) < heads)
        return R4D_NO(RAD_E_SHAPE);
    if (cidx && r4d_numel(cidx) < N) return R4D_NO(RAD_E_SHAPE);
    const long long slots = r4d_numel(tail) / (2 * (ngram - 1));
    if (slots < 1) return R4D_NO(RAD_E_SHAPE);

    const uint32_t blocks = (uint32_t)((N + kNgThreads - 1) / kNgThreads);
#define R4D_NGH(CH_)                                                                             \
    R4D_EMIT(stream, ngram_hint_kernel<CH_>, dim3(blocks), dim3(kNgThreads), 0,                  \
             (const int32_t*)r4d_addr(tok), (const int32_t*)r4d_addr(pos),                       \
             (const int32_t*)r4d_addr(cu), (const long long*)r4d_addr(mult),                     \
             (const long long*)r4d_addr(vsz), (const long long*)r4d_addr(off),                   \
             cidx ? (const int32_t*)r4d_addr(cidx) : nullptr, (int32_t*)r4d_addr(tail),          \
             (int32_t*)r4d_addr(ids), (uint32_t)N, (uint32_t)heads, (uint32_t)ngram,             \
             (uint32_t)per, eos, (uint32_t)ids_ld, (uint32_t)(hist ? 1 : 0), (uint32_t)slots)
    const int rc = r4d_geti_or(a, "chain", 0) != 0 ? R4D_NGH(r4dchain) : R4D_NGH(R4dUnchained);
#undef R4D_NGH
    return rc;
}
