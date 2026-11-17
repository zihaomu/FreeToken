/* mtp_couple_plan.h -- which draw the engine stages for each drafting row (serial or block).
 *
 * Engine::stage_couple_draws (core/engine_draft.cpp) reads the draw of every sequence on its card
 * from the step plan and, for each drafting row, the plan entry naming the row's sequence on the
 * row's card. Pure host bookkeeping, so it is tested without a card.
 *
 * `by_card` has one entry per card slot, indexed by StepEntry::card in [0, max_seqs), the range of
 * the StepSlots the advance reads.
 */
#pragma once

#include "batch.h"
#include "scheduler.h"   /* Scheduler::SerialRow */

#include "rad_spechub.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rad {

/* Fills `by_card[0, n_card)` from the plan and writes one draw per drafting row to `stage[0, n)`.
 * RAD_OK, or RAD_E_STATE with `why` set: the plane is too small for the card slots, or a drafting
 * row's card carries no plan entry naming its sequence (a stale row -- the plugin would refuse the
 * staged draw, so it is refused here where the row can be named). */
template <class Row>   /* Scheduler::SerialRow or Scheduler::BlockRow: a sequence and its card */
inline int mtp_couple_plan_stage(const StepPlan& plan, int n_card, const Row* rows,
                                 int n, std::vector<RadMtpCoupleDraw>* by_card,
                                 RadMtpCoupleDraw* stage, std::string* why) {
    if (!by_card || !stage || !why) return RAD_E_INVAL;
    if ((int)by_card->size() < n_card) {
        *why = "the draw plane holds " + std::to_string(by_card->size()) +
               " card slot(s), the step can use " + std::to_string(n_card);
        return RAD_E_STATE;
    }
    for (int c = 0; c < n_card; ++c) (*by_card)[(size_t)c] = RadMtpCoupleDraw{};
    for (const StepEntry& e : plan.e)
        if (e.req && e.card >= 0 && e.card < n_card)
            (*by_card)[(size_t)e.card] = { e.req->sp.seed, (uint64_t)e.req->output.size(), e.seq,
                                           e.card, e.req->sp.greedy() ? 0u : 1u };
    for (int i = 0; i < n; ++i) {
        const Row& row = rows[i];
        if (row.card < 0 || row.card >= n_card || (*by_card)[(size_t)row.card].seq != row.seq) {
            *why = "drafting row " + std::to_string(i) + " (sequence " + std::to_string(row.seq) +
                   ", card " + std::to_string(row.card) + ") has no plan entry on its card";
            return RAD_E_STATE;
        }
        stage[i] = (*by_card)[(size_t)row.card];
    }
    return RAD_OK;
}

}  // namespace rad
