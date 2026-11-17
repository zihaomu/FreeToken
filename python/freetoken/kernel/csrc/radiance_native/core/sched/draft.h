/* draft.h -- the speculative draft-depth controller.
 *
 * Static by default: the depth is `--num-speculative-tokens`. A dynamic policy that gates on the
 * drafter's own top-1 confidences cannot work here, because the scheduler picks the depth before
 * the drafter runs; measured acceptance arrives a step late and pooled across sequences. With a
 * draft head whose cost is nearly flat in depth (one extra verify row on a weight-bound step), the
 * best depth is a property of the model pair and is known up front, so it is a flag. The
 * controller still keeps acceptance accounting, a reported metric (spec §14).
 *
 * `--draft-window adaptive` is for when that premise fails: with streamed weights (a MoE reading
 * experts per verify row, a serial head running one pass per round) each draft costs a head round
 * and a verify row, and the best depth depends on the text (deep on code and JSON, shallow on
 * prose). The adaptive policy has two halves:
 *
 *   Per sequence (DraftAdapt): a smoothed estimate of each draft position's conditional
 *   acceptance, fed by the verify outcomes, and the sequence's stop -- the deepest draft whose
 *   chance of being emitted still pays for its verify row (kDraftMinChance).
 *
 *   Per step (plan_step): the head depth D. A serial head runs as many rounds as the deepest
 *   window in the step, so D maximises sum_i E_i(min(D, stop_i)) / T(R(D)), with
 *   E(d) = 1 + sum_k prod_{j<=k} a_j, R(D) the query rows the step would then carry (a row a
 *   sequence plus sum_i min(D, stop_i)) and T measured online by row count (step_time,
 *   step_cost.h). Each sequence then drafts min(D, stop_i).
 *
 * Both move only past a hold, and D also past a hysteresis, because a new window is a new pass
 * shape that must be recorded before it plays (runtime/ctx.h). `max_depth` stays the ceiling and
 * everything is sized for it.
 *
 * The policy is for serial drafters only (Scheduler::init): a block drafter proposes its block in
 * one pass and prompt lookup runs none, so neither pays per-round cost.
 *
 * It changes only how many drafts a step verifies. Acceptance follows core/sample/accept.h, so
 * every emitted token is the target's own; at temperature 0 the text matches a plain run as far
 * as the logits do, and a verify row's logits are computed at an M that moves with the window
 * (see accept.h).
 */
#pragma once
#include "rad_core.h"
#include "step_cost.h"

namespace rad {

/* libr4d's decode kernel serves up to 64 query rows, the verify-step band (spec §10), so 64 is
 * the largest window any batch can carry. */
enum { RAD_SCHED_MAX_SPEC = 64 };

/* One sequence's half of the adaptive policy: the drafter's record on this sequence's text and the
 * resulting stop. Stored in the scheduler's per-request state; the controller does the arithmetic.
 *
 * `depth` 0 means not started: the first ask fills the estimate from the prior and sets the stop.
 * It belongs to the request, not its cache: preemption keeps it, a recycled slot resets it. */
struct DraftAdapt {
    int32_t  depth = 0;                       /* the sequence's stop; 0 = not started */
    int32_t  held  = 0;                       /* verifies since `depth` last moved */
    float    a[RAD_SCHED_MAX_SPEC] = {};      /* conditional acceptance at each draft position */
    uint16_t n[RAD_SCHED_MAX_SPEC] = {};      /* verifies that evaluated each one, saturating */
};

class DraftController {
public:
    struct Params {
        int  max_depth = 0;       /* the deployment's window; 0 disables speculation entirely */
        /* The adaptive policy: `--draft-window adaptive` with a serial drafter, decided by
         * Scheduler::init. Off, the depth is max_depth. */
        bool adapt = false;
    };

    /* ---------------------------------------------------------- the policy's constants
     * Not configurable: each is a property of the policy, not of a deployment. The one
     * deployment property, step cost, is measured (step_time). */

    /* A window moves only for 2% more yield. One draft more or less moves E/T by several percent,
     * while a smoothed estimate wobbles about 1% per verify; a lower threshold would flip D
     * between near-equal neighbours and pay a recording each time. */
    static constexpr double kHyst = 0.02;
    /* And not within 8 verifies (steps, for D) of the last move. A pass shape is recorded on its
     * second sighting and played from the third (runtime/ctx.h), so a shape held 8 steps mostly
     * plays from the recording; 8 verifies at kHalfLife move an estimate about a third of the way
     * to a new rate, a trend rather than one rejection. */
    static constexpr int kHold = 8;
    /* Each position's estimate has a half-life of 16 verifies that evaluate it (about 50 tokens of
     * text): short enough to follow a reply from prose into code and back, long enough that one
     * rejection moves an estimate by about 4%. */
    static constexpr double kHalfLife = 16.0;
    /* An unmeasured position starts at 0.8, optimistic on purpose: a window that starts deep
     * learns every position, one that starts shallow never learns the deep ones. 0.8 lies between
     * typical first-position acceptance on prose and on code, so a wrong start corrects within a
     * hold. */
    static constexpr double kPrior = 0.8;
    /* Once 4 sequences have finished, new ones start from the mean of their final estimates
     * (adapt_retire): a running mean, then a moving average with a half-life of 32 sequences, so
     * the prior follows a change of traffic within a few dozen requests. A sequence counts only
     * after kHalfLife verifies; before that its estimate is mostly the prior itself. */
    static constexpr int    kPriorMinSeqs = 4;
    static constexpr double kPriorHalfLifeSeqs = 32.0;
    /* Never fewer than one draft: at zero no position is evaluated, so the estimate could never
     * recover. */
    static constexpr int kMinDepth = 1;
    /* A sequence stops at the first draft whose chance of being emitted, prod_{j<=d} a_j, is
     * under 1 in 10. Once the head runs D rounds for the deepest neighbour, a sequence's only
     * added cost is its verify row, and a draft pays while p_d / E exceeds the share of the step
     * that row adds. That share is a few percent at any load and E is two to three tokens a
     * sequence, so this is never stricter than D's own trade; it only keeps a sequence that
     * accepts little from riding its neighbours' rounds with rows that are almost never emitted. */
    static constexpr double kDraftMinChance = 0.1;
    /* The shape an untimed row count is priced on (StepCost): each query row adds 12% of a
     * one-row step, a verify row of streamed experts and its share of a head round. Qwen3.8
     * Flash-Next on two R9700s, ms a step at 3 / 4 / 5 rows of one sequence: 11.2 / 12.3 / 13.6.
     * Only ratios are used, and a timed row count replaces it. */
    static constexpr double kRowCost = 0.12;

    int  configure(const Params& p);
    const Params& params() const { return p_; }

    /* The draft depth for the next step; 0 means do not speculate. Constant: under the adaptive
     * policy it is the ceiling, and window_for() is the window. */
    int  next_depth() const { return p_.max_depth; }

    /* ---------------------------------------------------------- the adaptive policy */
    bool adaptive() const { return adapt_; }
    /* The window this sequence may draft into: min(D, its stop), starting its estimate on the
     * first ask; max_depth when the policy is off. Before any step chooses D, D is the ceiling. */
    int  window_for(DraftAdapt& s) const;
    /* As window_for() but never starts an estimate; 0 if not started. For reporting. */
    int  window_of(const DraftAdapt& s) const;
    /* One verify of this sequence: `drafted` proposed, a prefix of `accepted` taken. Updates the
     * estimate, and the stop once past the hold. Returns the stop. */
    int  adapt_observe(DraftAdapt& s, int drafted, int accepted) const;
    /* A finished sequence: its estimate joins the prior that new sequences start from.
     *
     * Per sequence, not per verify. Under this policy, pooled accepted(j) / reached(j) is biased:
     * only well-predicted texts get windows deep enough to measure deep positions. Here each
     * finished sequence counts once at every position, using its edge extrapolation where it never
     * reached (adapt_observe), so shallow texts pull deep positions down as much as deep ones hold
     * them up. */
    void adapt_retire(const DraftAdapt& s);
    /* The prior a sequence starting now is given at position `j`. For tests and logs. */
    double prior_at(int j) const;

    /* D for the step being opened. `seq` holds the estimates of the `n_seq` drafting sequences
     * (started here if needed), `n_rows` all decode rows (the rest emit one token and draft
     * nothing). Moves D past the hold and hysteresis; with nothing to draft, D stays. Returns D. */
    int  plan_step(DraftAdapt* const* seq, int n_seq, int n_rows);
    /* The step depth in force: 0 before the first step chose one. */
    int  step_depth() const { return step_d_; }
    /* The deepest window a step at depth D carries: D capped at the deepest stop it was chosen
     * over; the ceiling before the first choice or with the policy off. Proposals from outside
     * the head (prompt lookup) are held to it, so they add no deeper pass shape. */
    int  step_window() const { return adapt_ && step_w_ > 0 ? step_w_ : p_.max_depth; }

    /* T(R): a drafting decode step of `q_rows` query rows (every sequence's verified token and
     * its drafts) took `ms` since the end of the previous step, the head's rounds included. Fed
     * by Scheduler::step_finished; adaptive policy only. */
    void step_time(int q_rows, double ms);
    /* Expected cost of a step of `q_rows` query rows (StepCost::cost): ms once anything is timed,
     * else the unitless prior. */
    double step_cost(int q_rows) const;
    /* Samples behind a row count: for tests and logs. */
    int64_t cost_samples(int q_rows) const;

    /* E(d) for `s`'s current estimate: expected tokens emitted by a verify of d drafts, the
     * verified token included. For tests and logs. */
    double yield(const DraftAdapt& s, int d) const;
    /* Histogram of verifies by window, counted only under the adaptive policy. */
    int64_t verified_at(int d) const {
        return (d >= 0 && d <= RAD_SCHED_MAX_SPEC) ? by_depth_[d] : 0;
    }

    /* One verify's outcome: `drafted` proposed, a prefix of `accepted` taken. Metrics only. */
    void observe(int drafted, int accepted);

    /* Metrics (spec §14). Cumulative and exact. */
    int64_t drafted()  const { return tot_drafted_; }
    int64_t accepted() const { return tot_accepted_; }
    float   acceptance() const {
        return tot_drafted_ ? (float)((double)tot_accepted_ / (double)tot_drafted_) : 0.0f;
    }

    /* Per-position acceptance, a diagnostic rather than a policy input. The pooled ratio cannot
     * tell a drafter that decays gradually from one that fails at a specific position.
     *
     * `reached`: every earlier position was accepted, so this one was evaluated. `hits`: and it
     * was accepted. Their ratio is the conditional acceptance. Unreached positions are not
     * charged, so a rejection at position 0 does not count against later positions. */
    int64_t reached(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? reach_[j] : 0;
    }
    int64_t hits(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? hit_[j] : 0;
    }
    /* The deepest position any draft has reached; bounds the meaningful part of both arrays. */
    int     depth_seen() const { return seen_; }

private:
    Params  p_;
    int64_t tot_drafted_ = 0, tot_accepted_ = 0;
    int64_t reach_[RAD_SCHED_MAX_SPEC] = {};
    int64_t hit_  [RAD_SCHED_MAX_SPEC] = {};
    int     seen_ = 0;

    /* The adaptive policy as configure() resolved it. */
    bool    adapt_ = false;
    double  alpha_ = 0.0;                         /* the smoothing step from kHalfLife */
    int64_t by_depth_[RAD_SCHED_MAX_SPEC + 1] = {};
    int32_t step_d_ = 0;                          /* D; 0 until a step chose one */
    int32_t step_w_ = 0;                          /* min(D, the deepest stop it saw) */
    int32_t step_held_ = 0;                       /* steps since D last moved */
    /* The prior from finished sequences (adapt_retire), and how many are behind it. */
    float   prior_[RAD_SCHED_MAX_SPEC] = {};
    int64_t prior_n_ = 0;
    /* T(R): drafting steps' times by query rows. */
    StepCost cost_{kRowCost};

    /* Fill a sequence's estimate from the prior and set its first stop. */
    void    start(DraftAdapt& s) const;
    /* The deepest draft whose chance of being emitted is at least kDraftMinChance. */
    int     stop_of(const float* a) const;
};

}  /* namespace rad */
