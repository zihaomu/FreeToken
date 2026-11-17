/* lookup.cpp -- the prompt-lookup verify gate. See lookup.h. */
#include "lookup.h"

#include <cmath>

namespace rad {

void LookupGate::configure(int max_depth) {
    max_depth_ = max_depth < 0 ? 0 : (max_depth > RAD_SCHED_MAX_SPEC ? RAD_SCHED_MAX_SPEC : max_depth);
    alpha_ = 1.0 - std::exp2(-1.0 / kHalfLife);
    cost_.reset(kRowCost);
    proposed_ = held_ = held_drafts_ = held_hits_ = 0;
}

/* ------------------------------------------------------------------ one sequence */
void LookupGate::start(LookupSeq& q) const {
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) {
        q.s[0][j] = (float)kPriorShort;
        q.s[1][j] = (float)kPriorLong;
    }
    q.n[0] = q.n[1] = 0;
    q.pend_pos = -1;
    q.pend_len = q.pend_done = q.pend_cls = 0;
    q.pend_held = q.last_taken = false;
    q.proposed = q.held = q.held_drafts = q.held_hits = 0;
    q.started = true;
}

double LookupGate::expect(LookupSeq& q, int cls, int k) const {
    if (!q.started) start(q);
    if (cls < 0 || cls > 1) cls = 0;
    if (k > RAD_SCHED_MAX_SPEC) k = RAD_SCHED_MAX_SPEC;
    double e = 0.0;
    for (int j = 0; j < k; ++j) e += (double)q.s[cls][j];
    return e;
}

void LookupGate::offer(LookupSeq& q, int64_t pos, const int32_t* prop, int n, int match) const {
    if (!q.started) start(q);
    if (n <= 0 || !prop) return;
    ++q.proposed;
    ++proposed_;
    q.last_taken = false;
    if (q.pend_pos >= 0) return;   /* one at a time: it resolves within n tokens, usually one */
    if (n > RAD_SCHED_MAX_SPEC) n = RAD_SCHED_MAX_SPEC;
    q.pend_pos = pos;
    q.pend_len = n;
    q.pend_done = 0;
    q.pend_cls = class_of(match);
    q.pend_held = false;
    q.last_taken = true;
    for (int j = 0; j < n; ++j) q.pend[j] = prop[j];
}

void LookupGate::held(LookupSeq& q, int n) const {
    if (n <= 0) return;
    ++q.held;
    q.held_drafts += n;
    ++held_;
    held_drafts_ += n;
    if (q.last_taken) q.pend_held = true;
}

void LookupGate::fold(LookupSeq& q, int cls, int len, int hits) const {
    /* Position j came true iff j < hits: the survival estimate S_j sees every evaluated
     * proposal, not only those that reached j. */
    const double w = std::fmax(alpha_, 1.0 / ((double)q.n[cls] + kPriorWeight + 1.0));
    for (int j = 0; j < len; ++j) {
        const double x = j < hits ? 1.0 : 0.0;
        q.s[cls][j] = (float)((double)q.s[cls][j] + w * (x - (double)q.s[cls][j]));
    }
    if (q.n[cls] < UINT16_MAX) ++q.n[cls];
}

void LookupGate::advance(LookupSeq& q, const TokenSeq& seq) const {
    if (!q.started || q.pend_pos < 0) return;
    const int64_t len = seq.size();
    while (q.pend_done < q.pend_len && q.pend_pos + q.pend_done < len) {
        if (seq[q.pend_pos + q.pend_done] != q.pend[q.pend_done]) break;
        ++q.pend_done;
    }
    const bool missed = q.pend_done < q.pend_len && q.pend_pos + q.pend_done < len;
    if (!missed && q.pend_done < q.pend_len) return;   /* agrees so far; more tokens to come */
    fold(q, q.pend_cls, q.pend_len, q.pend_done);
    if (q.pend_held) {
        q.held_hits += q.pend_done;
        held_hits_ += q.pend_done;
    }
    q.pend_pos = -1;
}

/* ------------------------------------------------------------------ the step */
bool LookupGate::pays(double gain, int k, int64_t rows, double tokens) const {
    if (k <= 0 || !(gain > 0.0)) return false;
    if (rows < 1) rows = 1;
    if (tokens < 1.0) tokens = 1.0;
    const double t0 = step_cost(rows);
    const double t1 = step_cost(rows + k);
    /* (tokens + gain) / t1 > tokens / t0 */
    return gain * t0 > tokens * (t1 - t0);
}

void LookupGate::step_time(int64_t rows, double ms) { cost_.sample(rows, ms); }

int64_t LookupGate::cost_samples(int64_t rows) const { return cost_.samples(rows); }

double LookupGate::step_cost(int64_t rows) const { return cost_.cost(rows); }

}  /* namespace rad */
