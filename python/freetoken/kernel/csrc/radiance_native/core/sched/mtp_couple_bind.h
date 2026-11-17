/* mtp_couple_bind.h -- the engine's post-commit binding step for a serial drafting row.
 *
 * Engine::finish_serial_drafter calls this after the commit, once the card's proposal states are
 * read back. It decides:
 *
 *   - the position the next build stages the row at: the request's committed output length after
 *     this commit. It is not rebuilt from the draw's pre-commit base plus the accepted count, which
 *     differ when a max_tokens or context cut emits fewer tokens than the card accepted;
 *   - whether the row binds at all: a sequence that is gone from the commit or no longer Running
 *     will not sample again.
 *
 * Both come from Scheduler::committed_seq(), filled by commit under the scheduler lock, never from
 * the plan's StepEntry::req, which reap() may have freed by now. A header of its own so the rule
 * can be tested against a real commit without an Engine (tests/sched_test.cpp).
 */
#pragma once

#include "advance.h"
#include "rad_spechub.h"

#include <cstring>
#include <string>

namespace rad {

/* Where the next build will sample this sequence, after the current step's commit. */
struct MtpCoupleBindPoint {
    bool     found = false;   /* the committed step carried the sequence */
    bool     live  = false;   /* and its request was still Running (not finished/cancelled/failed) */
    uint64_t n_out = 0;       /* its committed output length: what the next build reads as n_out */
};

/* Bind `done` rounds of this row's draw to the positions the next build samples. `rb` is round 1
 * of this row in the engine's readback (stride `rb_stride` states), `tok[rd]` the card's token for
 * round rd, `S` the card's slot. Fills `out[0, done)` and sets `*n_bound = done`; a sequence that
 * is not live binds nothing (`*n_bound = 0`, RAD_OK) and leaves `out` untouched. RAD_E_STATE with
 * `why` for a readback that is not this draw's, or is not a state the proposer could have written.
 * A sampled round the proposer marked status -1 (a shortlist it could not turn into Q) kept the
 * greedy proposal; it is bound as is, the sampler does not couple it, and `*n_malformed` counts it.
 * `base` is never read here: it is the draft's coordinate, not the verify's. */
inline int mtp_couple_bind_rounds(const MtpCoupleBindPoint& p, const RadMtpCoupleDraw& d,
                                  const RadMtpCoupleState* rb, int rb_stride, int done,
                                  const StepSlot& S, int64_t vocab, RadMtpCoupleState* out,
                                  int* n_bound, std::string* why, int* n_malformed = nullptr) {
    *n_bound = 0;
    if (n_malformed) *n_malformed = 0;
    if (!p.live) return RAD_OK;
    const bool sampled = d.mode == 1;
    for (int rd = 1; rd <= done; ++rd) {
        const RadMtpCoupleState& st = rb[(size_t)(rd - 1) * (size_t)rb_stride];
        const bool malformed = sampled && st.status == -1;
        if (std::memcmp(&st.draw, &d, sizeof d) || st.round != rd || st.x != S.tok[rd] ||
            (st.status != (sampled ? 1 : 0) && !malformed) ||
            (sampled && !malformed && !rad_mtp_couple_state_valid(st, vocab))) {
            *why = "MTP coupled: sequence " + std::to_string(d.seq) + " round " +
                   std::to_string(rd) + " returned status " + std::to_string(st.status) +
                   " round " + std::to_string(st.round) + " x " + std::to_string(st.x) +
                   " for card token " + std::to_string(S.tok[rd]) + " (draw " +
                   (std::memcmp(&st.draw, &d, sizeof d) ? "stale" : "current") +
                   "); refusing the step";
            return RAD_E_STATE;
        }
        if (malformed && n_malformed) ++*n_malformed;
        RadMtpCoupleState b = st;
        b.pos = rad_mtp_couple_verify_pos(p.n_out, rd);
        out[rd - 1] = b;
    }
    *n_bound = done;
    return RAD_OK;
}

}  // namespace rad
