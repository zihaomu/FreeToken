/* draft.cpp -- the static draft-depth policy and the adaptive one (`--draft-window adaptive`).
 * See draft.h.
 */
#include "draft.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace rad {

namespace {

/* The prior weighs as much as this many verifies. The (n+1)-th verify moves an estimate by
 * 1 / (n + k + 1), a running mean over prior and evidence, until that falls below the half-life's
 * step and the moving average takes over. Without it a sequence would spend its first ~16
 * verifies at the prior's window, most of a short answer. */
constexpr double kPriorWeight = 4.0;

/* The weight of the (n+1)-th sample in an estimate that is a running mean until 1 / (n + 1)
 * falls below a moving average's step at half-life `h` samples, and that average after. */
double settle_weight(double h, int64_t n) {
    return std::fmax(1.0 - std::exp2(-1.0 / h), 1.0 / (double)(n + 1));
}

}  /* namespace */

int DraftController::configure(const Params& p) {
    if (p.max_depth < 0 || p.max_depth > RAD_SCHED_MAX_SPEC) {
        RAD_ERR("draft depth %d is outside 0..%d -- the verify kernel band is %d query rows",
                p.max_depth, RAD_SCHED_MAX_SPEC, RAD_SCHED_MAX_SPEC);
        return RAD_E_INVAL;
    }
    p_ = p;
    tot_drafted_ = tot_accepted_ = 0;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) { reach_[j] = 0; hit_[j] = 0; }
    seen_ = 0;

    adapt_ = p.adapt && p.max_depth > 0;
    alpha_ = 1.0 - std::exp2(-1.0 / kHalfLife);
    step_d_ = 0;
    step_w_ = 0;
    step_held_ = 0;
    prior_n_ = 0;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) prior_[j] = 0.0f;
    for (int d = 0; d <= RAD_SCHED_MAX_SPEC; ++d) by_depth_[d] = 0;
    cost_.reset(kRowCost);
    if (adapt_)
        RAD_INFO("draft depth: adaptive, %d..%d; a sequence stops at the first draft emitted less "
                 "than %.0f%% of the time, and the depth moves at +%.0f%% at most every %d steps",
                 kMinDepth, p_.max_depth, 100.0 * kDraftMinChance, 100.0 * kHyst, kHold);
    return RAD_OK;
}

void DraftController::observe(int drafted, int accepted) {
    if (drafted <= 0) return;
    if (drafted > RAD_SCHED_MAX_SPEC) drafted = RAD_SCHED_MAX_SPEC;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;
    tot_drafted_  += drafted;
    tot_accepted_ += accepted;

    /* Position j is evaluated only if every earlier position was accepted: 0..accepted-1 are
     * evaluated and accepted, position `accepted` (if drafted) evaluated and rejected, and nothing
     * past it is reached. */
    const int last = accepted < drafted ? accepted : drafted - 1;
    for (int j = 0; j <= last; ++j) {
        ++reach_[j];
        if (j < accepted) ++hit_[j];
    }
    if (drafted > seen_) seen_ = drafted;
    if (adapt_) ++by_depth_[drafted];
}

/* ------------------------------------------------------------------ the step's cost */
void DraftController::step_time(int q_rows, double ms) {
    if (!adapt_) return;
    cost_.sample(q_rows, ms);
}

int64_t DraftController::cost_samples(int q_rows) const {
    return q_rows >= 1 ? cost_.samples(q_rows) : 0;
}

double DraftController::step_cost(int q_rows) const { return cost_.cost(q_rows); }

/* ------------------------------------------------------------------ one sequence */
int DraftController::stop_of(const float* a) const {
    double p = 1.0;
    int d = 0;
    while (d < p_.max_depth) {
        p *= (double)a[d];
        if (p < kDraftMinChance) break;
        ++d;
    }
    return d < kMinDepth ? kMinDepth : d;
}

double DraftController::prior_at(int j) const {
    if (j < 0 || j >= RAD_SCHED_MAX_SPEC) return 0.0;
    return prior_n_ >= kPriorMinSeqs ? (double)prior_[j] : kPrior;
}

void DraftController::start(DraftAdapt& s) const {
    /* From finished sequences once there are enough (adapt_retire), else the fixed kPrior. */
    for (int j = 0; j < p_.max_depth; ++j) {
        s.a[j] = (float)prior_at(j);
        s.n[j] = 0;
    }
    s.depth = stop_of(s.a);
    s.held  = 0;
}

void DraftController::adapt_retire(const DraftAdapt& s) {
    if (!adapt_ || s.depth <= 0 || (double)s.n[0] < kHalfLife) return;
    const double w = settle_weight(kPriorHalfLifeSeqs, prior_n_);
    for (int j = 0; j < p_.max_depth; ++j)
        prior_[j] = prior_n_ ? (float)((double)prior_[j] + w * ((double)s.a[j] - (double)prior_[j]))
                             : s.a[j];
    ++prior_n_;
}

double DraftController::yield(const DraftAdapt& s, int d) const {
    if (d > p_.max_depth) d = p_.max_depth;
    double e = 1.0, run = 1.0;
    for (int k = 1; k <= d; ++k) { run *= (double)s.a[k - 1]; e += run; }
    return e;
}

int DraftController::window_of(const DraftAdapt& s) const {
    if (!adapt_) return p_.max_depth;
    if (s.depth <= 0) return 0;
    const int D = step_d_ > 0 ? step_d_ : p_.max_depth;
    return s.depth < D ? s.depth : D;
}

int DraftController::window_for(DraftAdapt& s) const {
    if (!adapt_) return p_.max_depth;
    if (s.depth <= 0) start(s);
    return window_of(s);
}

int DraftController::adapt_observe(DraftAdapt& s, int drafted, int accepted) const {
    if (!adapt_) return p_.max_depth;
    if (s.depth <= 0) start(s);
    if (drafted <= 0) return s.depth;
    const int D = p_.max_depth;
    if (drafted > D) drafted = D;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;

    /* The positions this verify evaluated, by observe()'s rule. Positions past the first
     * rejection were never evaluated and are not updated here. */
    const int last = accepted < drafted ? accepted : drafted - 1;
    for (int j = 0; j <= last; ++j) {
        const double w = std::fmax(alpha_, 1.0 / ((double)s.n[j] + kPriorWeight + 1.0));
        const double x = j < accepted ? 1.0 : 0.0;
        s.a[j] = (float)((double)s.a[j] + w * (x - (double)s.a[j]));
        if (s.n[j] < UINT16_MAX) ++s.n[j];
    }
    /* Positions past the window drift toward its deepest one. Otherwise a sequence held shallow
     * by prose that turns into code would keep its stale deep estimates and never draft deep
     * enough to update them. Assumes acceptance decays smoothly with position; when wrong it costs
     * a move out and back. */
    const double edge = (double)s.a[drafted - 1];
    for (int j = drafted; j < D; ++j)
        s.a[j] = (float)((double)s.a[j] + alpha_ * (edge - (double)s.a[j]));

    /* The stop moves only past the hold: it is this sequence's share of the step's shape. */
    if (s.held < INT32_MAX) ++s.held;
    if (s.held < kHold) return s.depth;
    const int d = stop_of(s.a);
    if (d != s.depth) {
        s.depth = d;
        s.held  = 0;
    }
    return s.depth;
}

/* ------------------------------------------------------------------ the step */
int DraftController::plan_step(DraftAdapt* const* seq, int n_seq, int n_rows) {
    if (!adapt_ || n_seq <= 0) return step_d_;
    if (n_rows < n_seq) n_rows = n_seq;
    const int M = p_.max_depth;

    /* Expected tokens at each head depth, sum_i E_i(min(D, stop_i)) plus one per non-drafting
     * row, and the drafts the step would then verify, sum_i min(D, stop_i). Candidates end at the
     * deepest stop, since a deeper D changes nothing. */
    double tok[RAD_SCHED_MAX_SPEC + 1] = {};
    int    drafts[RAD_SCHED_MAX_SPEC + 1] = {};
    int top = kMinDepth;
    for (int i = 0; i < n_seq; ++i) {
        DraftAdapt& s = *seq[i];
        if (s.depth <= 0) start(s);
        if (s.depth > top) top = s.depth;
        double e = 1.0, run = 1.0;
        for (int d = 1; d <= M; ++d) {
            if (d <= s.depth) { run *= (double)s.a[d - 1]; e += run; }
            tok[d] += e;
            drafts[d] += d <= s.depth ? d : s.depth;
        }
    }
    const double plain = (double)(n_rows - n_seq);
    if (top > M) top = M;

    /* A step's query rows at depth d: a row a decode sequence and its drafts. */
    auto gain = [&](int d) { return (tok[d] + plain) / step_cost(n_rows + drafts[d]); };
    int best = kMinDepth;
    double best_g = -1.0;
    for (int d = kMinDepth; d <= top; ++d) {
        const double g = gain(d);
        if (g > best_g) { best_g = g; best = d; }
    }

    /* Move only after kHold steps and only for kHyst more yield than the current depth. The
     * first drafting step takes the best outright. */
    if (step_d_ <= 0) {
        step_d_ = best;
        step_held_ = 0;
    } else {
        if (step_held_ < INT32_MAX) ++step_held_;
        const int cur = step_d_ < top ? step_d_ : top;
        const double cur_g = gain(cur);
        if (best != cur && step_held_ >= kHold && best_g > cur_g * (1.0 + kHyst)) {
            RAD_DEBUG("draft window: head depth %d -> %d over %d drafting of %d rows (%.3f -> "
                      "%.3f tokens per unit of step cost)", step_d_, best, n_seq, n_rows, cur_g,
                      best_g);
            step_d_ = best;
            step_held_ = 0;
        }
    }
    step_w_ = step_d_ < top ? step_d_ : top;
    return step_d_;
}

}  /* namespace rad */
