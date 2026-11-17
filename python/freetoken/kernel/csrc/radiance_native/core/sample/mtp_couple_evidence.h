/* mtp_couple_evidence.h -- what the completed-verify log line may claim.
 *
 * The line reports a coupled verify only when the build that staged bound rows belongs to the step
 * whose wait just returned, and counts only sequences still live after that wait. It reads
 * Sampler::couple_build(), the snapshot build_step froze, because the drafter's build_greedy
 * clears staged_any_couple() before the wait. The op's domain is printed so a host reference row
 * is not read as a device verify.
 *
 * A pure function of the snapshot, a liveness predicate and the domain, so it is tested without
 * an Engine (tests/sample_test.cpp).
 */
#pragma once

#include "sampler.h"

#include <string>

namespace rad {

struct MtpCoupleEvidence {
    int64_t     rows   = 0;     /* bound rows of sequences still live after the wait */
    bool        device = false; /* the op that ran is a device op */
    std::string seqs;           /* those sequences, comma separated, in build order */
};

/* `live(seq)` answers whether the sequence is still Running; the engine passes a predicate over
 * Scheduler::running(), the tests a set. Empty (rows == 0) when the snapshot is not this step's,
 * or nothing it staged is still live. */
template <class Live>
MtpCoupleEvidence mtp_couple_evidence(const Sampler::CoupleBuild& b, int64_t step, int domain,
                                      Live&& live) {
    MtpCoupleEvidence e;
    if (b.step != step) return e;
    e.device = domain == RAD_DOMAIN_DEVICE;
    for (const auto& s : b.seqs)
        if (live(s.first)) {
            e.rows += s.second;
            if (!e.seqs.empty()) e.seqs += ",";
            e.seqs += std::to_string(s.first);
        }
    return e;
}

}  // namespace rad
