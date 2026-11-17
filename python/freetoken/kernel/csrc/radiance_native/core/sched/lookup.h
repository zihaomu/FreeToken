/* lookup.h -- whether a prompt-lookup proposal is worth its verify rows.
 *
 * A verify row is not free. On a MoE every row reads its own routed experts, so verifying five
 * drafts on one sequence costs a few plain steps (DeepSeek-V4.1 at TP4: 32 ms against 13.5 ms),
 * and prompt lookup proposes on novel text too: any bigram it has seen before is a pattern. On
 * prose those proposals are wrong four times in five and the verifies make decoding slower than
 * not speculating; on code edits and quoted text they are right most of the time and it is twice
 * as fast. So each proposal is verified only when it is expected to pay:
 *
 *     verify k drafts iff  E[accepted] * T(R) > tokens * (T(R + k) - T(R))
 *
 * where T(R) is the measured wall time of a decode step of R query rows, R the rows the step
 * already carries and `tokens` what it already expects to emit. For one sequence that is
 * 1 + E[accepted] > T(1 + k) / T(1).
 *
 * E[accepted] = sum_{j<k} S_j, with S_j the sequence's estimate of P(the proposal's first j + 1
 * tokens all come true). The estimate is fed by every proposal, verified or not: a proposal held
 * back is compared with the tokens the sequence goes on to emit, and its accepted count is their
 * common prefix. That is exactly what a verify would have accepted, at any temperature, because
 * a verify accepts a draft iff it equals the token the target emits at that position and the
 * target's draw there does not depend on the draft (core/sample/accept.h). So a held sequence
 * keeps learning at no cost, and resumes drafting as soon as its text turns repetitive.
 *
 * Estimates are kept per class of proposal: one whose match runs at least kLongMatch tokens back
 * (the sequence is copying a stretch it has seen) and one whose match is shorter (a recurring
 * bigram or trigram). They differ by far more than either varies within a reply, and a new
 * sequence starts each from its own prior: a long match is drafted at once, a short one waits for
 * evidence.
 *
 * T is measured online over pure decode steps, keyed by rows (StepCost, step_cost.h). Before a
 * row count is timed, its cost is the nearest timed count's scaled by the prior shape
 * 1 + kRowCost (R - 1), which is optimistic on purpose: the first verify it lets through times
 * the real cost.
 *
 * Only which proposals are verified changes, never what is emitted: output is the same as with
 * speculation off.
 */
#pragma once
#include <cstdint>

#include "draft.h"
#include "step_cost.h"
#include "ngram.h"

namespace rad {

/* One sequence's record: its estimates and the proposal being compared with its text. Kept in the
 * scheduler's per-request state; it describes the text, so preemption keeps it and a new request
 * starts it over. */
struct LookupSeq {
    bool     started = false;
    float    s[2][RAD_SCHED_MAX_SPEC] = {};   /* by class: P(first j + 1 drafts all come true) */
    uint16_t n[2] = {};                       /* proposals evaluated in each class, saturating */
    /* The proposal under evaluation: its first token is the sequence's token `pend_pos`; -1 none.
     * `pend_done` of its `pend_len` tokens have been compared, all equal so far. */
    int64_t  pend_pos = -1;
    int32_t  pend_len = 0;
    int32_t  pend_done = 0;
    int32_t  pend_cls = 0;
    bool     pend_held = false;               /* it was held back rather than verified */
    bool     last_taken = false;              /* the last offer became the pending one */
    int32_t  pend[RAD_SCHED_MAX_SPEC] = {};
    /* For the request's log line. */
    int64_t  proposed = 0, held = 0, held_drafts = 0, held_hits = 0;
};

class LookupGate {
public:
    /* ---------------------------------------------------------- the policy's constants
     * Fixed; none is a property of a deployment. The deployment's one property, what a verify
     * row costs, is measured (step_time). Chosen on 24 real replies (an essay, a code edit and 22
     * varied prompts, DeepSeek-V4.1 greedy, 256 tokens) replayed through the engine's lookup rule
     * at per-row costs of 0.06 to 0.6 of a plain step: over the prose replies, 0.999 to 1.014x
     * the speed of not speculating (geometric mean) and 0.988x at worst, where verifying every
     * proposal gives 0.85 to 1.015x and 0.73x at worst; on the code edit within 2.1% of verifying
     * every proposal at the costs where that pays, and 11% ahead of it at 0.6. */

    /* A proposal whose match runs this many tokens back is the long class. */
    static constexpr int kLongMatch = 8;
    /* Where each class starts: P(each draft comes true), the same at every position. A long match
     * drafts at once at any row cost under 0.8 of a step; a short one waits until its own record
     * says it pays. */
    static constexpr double kPriorShort = 0.0;
    static constexpr double kPriorLong = 0.8;
    /* The prior weighs as much as two proposals: the third evaluated in a class moves its
     * estimate by a third. */
    static constexpr double kPriorWeight = 2.0;
    /* Then a moving average with a half-life of 8 proposals, short enough to follow a reply from
     * prose into a quoted block and out again within a few lines. */
    static constexpr double kHalfLife = 8.0;
    /* The shape an untimed row count is priced on: each query row adds 6% to a step. */
    static constexpr double kRowCost = 0.06;

    void configure(int max_depth);
    int  max_depth() const { return max_depth_; }

    /* ---------------------------------------------------------- one sequence */
    /* The class of a proposal whose match runs `match` tokens back. */
    static int class_of(int match) { return match >= kLongMatch ? 1 : 0; }
    /* Expected drafts accepted if the first `k` of a class-`cls` proposal are verified. Starts the
     * estimate if needed. */
    double expect(LookupSeq& q, int cls, int k) const;
    /* A proposal of `n` tokens whose first is the sequence's token `pos`: taken for evaluation
     * when none is pending. Counted as proposed either way. */
    void offer(LookupSeq& q, int64_t pos, const int32_t* prop, int n, int match) const;
    /* The proposal offered last was not verified: `n` drafts held back. Counters only. */
    void held(LookupSeq& q, int n) const;
    /* After a commit: compares the pending proposal with the tokens `seq` now holds, and folds it
     * into its class once a token differs or every token has come true. */
    void advance(LookupSeq& q, const TokenSeq& seq) const;

    /* ---------------------------------------------------------- the step */
    /* Whether `k` drafts expected to yield `gain` accepted tokens pay, in a step that so far
     * carries `rows` query rows and expects `tokens` tokens. */
    bool pays(double gain, int k, int64_t rows, double tokens) const;
    /* A pure decode step of `rows` query rows took `ms`, from the end of the step before. */
    void step_time(int64_t rows, double ms);
    /* The expected wall time of a decode step of `rows` rows: ms once anything is timed, else the
     * unitless prior. Only ratios are used. */
    double step_cost(int64_t rows) const;
    int64_t cost_samples(int64_t rows) const;

    /* Across all sequences, for metrics: proposals offered; proposals held back and their drafts;
     * and of the held proposals scored (one at a time per sequence), the drafts that came true:
     * what verifying them would have accepted. Updated by the per-sequence calls above, which are
     * const on the policy, hence mutable. */
    int64_t proposed() const { return proposed_; }
    int64_t held_count() const { return held_; }
    int64_t held_drafts() const { return held_drafts_; }
    int64_t held_hits() const { return held_hits_; }

private:
    int    max_depth_ = 0;
    double alpha_ = 0.0;
    StepCost cost_{kRowCost};
    mutable int64_t proposed_ = 0, held_ = 0, held_drafts_ = 0, held_hits_ = 0;

    void start(LookupSeq& q) const;
    void fold(LookupSeq& q, int cls, int len, int hits) const;
};

}  /* namespace rad */
