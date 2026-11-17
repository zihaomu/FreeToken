/* bands.cpp -- the bucket table. spec.md §2.2.
 *
 * A parameter may be declared as a range, because some parameters belong to the batch rather
 * than the model. M is the main one: a GEMM sees one row at decode, 64 at a speculative verify
 * and thousands in a prefill chunk, and libr4d has a different kernel for each band. Resolving
 * once at M = max_tok would run the prefill kernel at M = 1.
 *
 * The bucket boundaries are the candidate kernels' own constraint values. There is no
 * power-of-two ladder or bucket-count policy here.
 */
#include "registry.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace rad {

namespace {

/* The DIV a row places on the ranged parameter, or the least common multiple of several; 0 when
 * it places none. */
int64_t ranged_div(const KernelRow* r, std::string_view key) {
    if (!r || !r->info || key.empty()) return 0;
    int64_t d = 0;
    for (int i = 0; i < r->info->n_constraints; ++i) {
        const RadConstraint& c = r->info->constraints[i];
        if (c.op != RAD_C_DIV || !c.key || key != c.key || c.ival <= 1) continue;
        int64_t a = d ? d : 1, b = c.ival;
        while (b) { const int64_t t = a % b; a = b; b = t; }
        d = (d ? d : 1) / a * c.ival;
    }
    return d;
}

/* "M <= 64" for the first band, "M in (16, 64]" for a later one, so the printed table does not
 * read as overlapping. */
std::string band_span(std::string_view key, int64_t lo_excl, int64_t hi, bool first) {
    if (hi == INT64_MAX) return std::string("any shape");   /* the op has no ranged parameter */
    if (first) return fmt("%.*s <= %lld", (int)key.size(), key.data(), (long long)hi);
    return fmt("%.*s in (%lld, %lld]", (int)key.size(), key.data(), (long long)lo_excl,
               (long long)hi);
}

}  /* namespace */

std::vector<Band> Registry::build_bands(std::string_view op, std::string_view ranged_key,
                                        int64_t lo, int64_t hi, const Geometry& base,
                                        const RowFilter* accept) {
    std::vector<Band> out;

    /* No ranged parameter: one band covering everything. hi is INT64_MAX, not the declared value,
     * so the issue path's band scan finds it for any issue count. */
    if (ranged_key.empty()) {
        lo = 0;
        hi = INT64_MAX;
    } else if (hi < lo) {
        RAD_WARN("op '%s' declares the empty range %s in [%lld, %lld]; resolving the single value "
                 "%lld", std::string(op).c_str(), std::string(ranged_key).c_str(),
                 (long long)lo, (long long)hi, (long long)lo);
        hi = lo;
    }

    /* The union of the LE, GE and EQ values that candidate kernels from every plugin and both
     * domains place on the ranged parameter: boundaries belong to the candidate set, not to the
     * kernel that wins a band. An IN on the ranged parameter lists its values, each an EQ with a
     * band of its own, so the row wins only values it serves. A DIV is not an interval and cuts no
     * edge: a row with one is resolved at a band's upper bound, which it divides, and the band
     * records the divisor so the issue path refuses a value inside it the row does not serve
     * (Resolved::ranged_div). */
    std::vector<int64_t> bounds;
    auto add = [&](int64_t b) {
        if (b >= lo && b < hi) bounds.push_back(b);   /* clamped to the declared range; hi closes
                                                       * the last band and is appended below */
    };
    if (!ranged_key.empty()) {
        if (const std::vector<const KernelRow*>* rows = rows_for(op)) {
            for (const KernelRow* r : *rows) {
                for (int i = 0; i < r->info->n_constraints; ++i) {
                    const RadConstraint& c = r->info->constraints[i];
                    if (!c.key || ranged_key != c.key) continue;
                    switch (c.op) {
                        case RAD_C_LE: add(c.ival);     break;
                        /* A kernel that starts at v ends the band below it at v-1. */
                        case RAD_C_GE: add(c.ival - 1); break;
                        /* An EQ needs both edges, so its value gets a band of its own. */
                        case RAD_C_EQ: add(c.ival - 1); add(c.ival); break;
                        case RAD_C_IN:
                            for (std::string_view tok : split_ws(c.sval ? c.sval : "")) {
                                const std::string t(tok);
                                char* end = nullptr;
                                const long long v = std::strtoll(t.c_str(), &end, 10);
                                if (end && *end == '\0' && end != t.c_str()) { add(v - 1); add(v); }
                            }
                            break;
                        default: break;
                    }
                }
            }
        }
    }
    std::sort(bounds.begin(), bounds.end());
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
    bounds.push_back(hi);

    /* Each band is resolved at its upper bound, in both domains: the upper bound is the band's
     * worst case, so a kernel that serves it serves the whole band. */
    int64_t prev = lo - 1;
    for (size_t i = 0; i < bounds.size(); ++i) {
        Band b;
        b.hi = bounds[i];

        Geometry g = base;
        if (!ranged_key.empty()) g.set_i(ranged_key, b.hi);

        SelectionTrace t;
        for (int dom = 0; dom < RAD_N_DOMAINS; ++dom) {
            b.dom[dom] = resolve(op, g, dom, &t, accept);
            if (!b.dom[dom]) b.miss[dom] = t.reason;
            else b.dom[dom].ranged_div = ranged_div(b.dom[dom].row, ranged_key);
        }

        b.span = band_span(ranged_key, prev, b.hi, i == 0);
        if (!b.dom[RAD_DOMAIN_HOST])
            b.miss[RAD_DOMAIN_HOST] += "; site 'host' is unavailable for this band";
        out.push_back(std::move(b));
        prev = bounds[i];
    }

    /* No device kernel is fatal. No host kernel only removes host-site execution for this op;
     * its weights may still live in the host and SSD tiers and stage to the device (§5.1).
     *
     * Neither is reported here: this runs once per declared op (64 times for the same op on a
     * 64-layer model). Declare always completes, and Builder collects the bands' miss strings and
     * reports once per graph (§3.1), a hole in a range as that band. */
    return out;
}

}  /* namespace rad */
