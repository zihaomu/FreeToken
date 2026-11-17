/* step_cost.h -- a decode step's wall time as a function of its query rows, measured online.
 *
 * A decode step's cost moves with the rows it carries (each sequence's verified token and its
 * drafts): the routed experts and every other weight read per row. Both speculation policies
 * weigh a draft's expected yield against that cost (lookup.h, draft.h), and both need the cost of
 * row counts they have not run yet. So step times are kept by row count: a timed count is its
 * mean; one between two timed counts lies on the line between them; one outside them is the
 * nearest timed count scaled by a prior shape, 1 + row_cost (R - 1); with nothing timed it is
 * that shape itself, unitless. Only ratios of costs are used.
 *
 * Row counts move by themselves as sequences come and go, so a policy holding one window still
 * times its neighbours' row counts without ever trying them.
 */
#pragma once
#include <cmath>
#include <cstdint>

namespace rad {

class StepCost {
public:
    /* A running mean for the first samples, then a moving average with a half-life of 32 steps:
     * long enough to average out one-off costs (a tier copy, a recording, a prefix-cache trim),
     * short enough to follow a context growing by a few thousand tokens over a reply. */
    static constexpr double kHalfLife = 32.0;
    /* Once a key has 4 samples one step counts for at most 4x its mean: a step that waited on a
     * recording or a host stall says nothing about its row count. Below 4 samples the mean is not
     * a trustworthy reference to clip against. */
    static constexpr double kOutlier = 4.0;
    static constexpr int64_t kClipAfter = 4;
    /* Row counts are kept exactly up to 128, then in bands of 32. */
    enum { kExact = 128, kBand = 32, kKeys = 160 };

    /* `row_cost` is the prior shape's slope: the share of a one-row step each further row adds. */
    explicit StepCost(double row_cost = 0.0) : row_cost_(row_cost) {}

    /* Forgets every sample and takes a slope. */
    void reset(double row_cost) {
        row_cost_ = row_cost;
        for (int k = 0; k < kKeys; ++k) { r_[k] = 0.0; ms_[k] = 0.0; n_[k] = 0; }
    }

    /* A step of `rows` query rows took `ms`. Nonsense is not a sample. */
    void sample(int64_t rows, double ms) {
        if (rows < 1 || !(ms > 0.0) || !std::isfinite(ms)) return;
        const int k = key(rows);
        double&  m = ms_[k];
        double&  r = r_[k];
        int64_t& n = n_[k];
        if (n >= kClipAfter && ms > kOutlier * m) ms = kOutlier * m;
        const double w = std::fmax(1.0 - std::exp2(-1.0 / kHalfLife), 1.0 / (double)(n + 1));
        m = n ? m + w * (ms - m) : ms;
        r = n ? r + w * ((double)rows - r) : (double)rows;
        ++n;
    }

    /* The expected wall time of a step of `rows` rows: ms once anything is timed, else the
     * unitless prior. */
    double cost(int64_t rows) const {
        if (rows < 1) rows = 1;
        const double R = (double)rows;
        /* The timed keys either side, by the mean rows of their samples (monotone in the key). */
        int lo = -1, hi = -1;
        for (int k = 0; k < kKeys; ++k) {
            if (n_[k] <= 0) continue;
            if (r_[k] <= R) lo = k;
            if (r_[k] >= R && hi < 0) hi = k;
        }
        if (lo >= 0 && hi >= 0) {
            const double r0 = r_[lo], r1 = r_[hi];
            if (lo == hi || r1 <= r0) return ms_[lo];
            return ms_[lo] + (ms_[hi] - ms_[lo]) * (R - r0) / (r1 - r0);
        }
        if (lo >= 0) return ms_[lo] * prior(R) / prior(r_[lo]);
        if (hi >= 0) return ms_[hi] * prior(R) / prior(r_[hi]);
        return prior(R);
    }

    /* Samples behind a row count's key: for tests and logs. */
    int64_t samples(int64_t rows) const { return n_[key(rows)]; }

private:
    double  row_cost_ = 0.0;
    double  r_[kKeys] = {};     /* mean rows of the samples in each key */
    double  ms_[kKeys] = {};
    int64_t n_[kKeys] = {};

    static int key(int64_t rows) {
        if (rows < 1) rows = 1;
        if (rows <= kExact) return (int)(rows - 1);
        const int64_t k = kExact + (rows - kExact - 1) / kBand;
        return (int)(k < kKeys ? k : kKeys - 1);
    }
    double prior(double rows) const { return 1.0 + row_cost_ * (rows - 1.0); }
};

}  /* namespace rad */
