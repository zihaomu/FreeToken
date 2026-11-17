/* rad_graph.cpp -- `--debug-graph` (spec §16): the full record of what runs. Every op in declared
 * order with its bucket table (one line per band: the resolving kernel, variant and plugin), the
 * matched constraints, the weights it touches and the buffer it writes; then the weight manifest,
 * buffer plan, KV groups, tuning requests and miss report.
 *
 * The bucket-table block is exactly the format in spec §2.2:
 *
 *     gemm_w4a8_nt  N=8192 K=5120
 *         M <=   16  ->  gemm_w4a8_nt_m64        (libr4d, prio 20)
 *         M <=   64  ->  gemm_w4a8_nt_m64        (libr4d, prio 20)
 *         M <= 8192  ->  gemm_w4a8_prefill       (libr4d, prio 10)
 *
 * Everything else goes on its own lines at a deeper indent, so the table stays greppable.
 */
#include "rad_build.h"

#include "startup.h"

#include "util/json.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <unordered_map>
#include <cstring>

namespace rad {

/* ordered_json keeps keys in insertion order rather than alphabetical. */
using json = Json;

namespace {

/* A constraint as the kernel wrote it. constraint_why() phrases it as a refusal
 * ("M <= 16, have 8192") for the miss column. */
std::string constraint_str(const RadConstraint& c) {
    switch (c.op) {
        case RAD_C_EQ:  return fmt("%s == %lld", c.key, c.ival);
        case RAD_C_LE:  return fmt("%s <= %lld", c.key, c.ival);
        case RAD_C_GE:  return fmt("%s >= %lld", c.key, c.ival);
        case RAD_C_DIV: return fmt("%s %% %lld == 0", c.key, c.ival);
        case RAD_C_IN:  return fmt("%s in {%s}", c.key, c.sval ? c.sval : "");
        default:        return fmt("%s ?", c.key);
    }
}

/* "(libr4d, prio 20)", plus the tuned point only for a kernel with declared axes, so rows that
 * were tuned stand out (spec §15). */
std::string origin(const Resolved& r) {
    const RadKernelInfo* k = r.row->info;
    if (!k || k->n_tunables <= 0) return fmt("(%s, prio %d)", r.row->plugin.c_str(),
                                             k ? k->priority : 0);
    const char* vn = r.choice.empty() ? "?" : r.choice.c_str();
    /* Same spelling as render_band_table in core/plugin/. */
    return fmt("(%s, prio %d, %s from %s)", r.row->plugin.c_str(), k->priority, vn,
               r.tune_source.empty() ? "axis defaults" : r.tune_source.c_str());
}

/* The left half of a band line, through the arrow. Ranged and fixed forms share column positions
 * so they stack in one table. */
std::string band_prefix(const OpInfo& o, const Band& b) {
    if (o.ranged_key.empty()) return "    always     ->  ";
    if (b.hi == LLONG_MAX)    return fmt("    %s <=  any  ->  ", o.ranged_key.c_str());
    return fmt("    %s <= %4lld  ->  ", o.ranged_key.c_str(), (long long)b.hi);
}

bool domain_has_any(const OpInfo& o, int d) {
    for (const Band& b : o.bands) if (b.dom[d]) return true;
    return false;
}

void dump_bucket_table(std::string& s, const OpInfo& o, int d) {
    for (const Band& b : o.bands) {
        const Resolved& r = b.dom[d];
        if (r) {
            s += band_prefix(o, b) + fmt("%-24s", r.row->info ? r.row->info->name : "?")
               + origin(r) + "\n";
        } else {
            /* Misses are reported per band, not per op (spec §2.2). */
            s += band_prefix(o, b) + fmt("%-24s", "(nothing)")
               + "-- " + (b.miss[d].empty() ? "no kernel matched" : b.miss[d]) + "\n";
        }
    }
}

void dump_matched(std::string& s, const OpInfo& o, int d) {
    /* One line per distinct kernel, not per band. */
    std::vector<const RadKernelInfo*> seen;
    for (const Band& b : o.bands) {
        const Resolved& r = b.dom[d];
        if (!r || !r.row->info) continue;
        if (std::find(seen.begin(), seen.end(), r.row->info) != seen.end()) continue;
        seen.push_back(r.row->info);

        std::string cs;
        for (int i = 0; i < r.row->info->n_constraints; ++i) {
            if (i) cs += ", ";
            cs += constraint_str(r.row->info->constraints[i]);
        }
        if (cs.empty()) cs = "(unconstrained -- serves any geometry)";
        s += fmt("      matched  %s: %s\n", r.row->info->name, cs.c_str());
    }
}

/* Append a table row with trailing spaces trimmed; the tier and site columns are empty at declare
 * (the planner fills them). */
void append_row(std::string& s, std::string row) {
    while (!row.empty() && row.back() == ' ') row.pop_back();
    s += row;
    s += '\n';
}

}  /* namespace */

std::string dump_graph_str(const Program& p, const std::vector<std::string>* errors) {
    std::string s;

    s += fmt("==== declared graph: %s [%s]  rank %d/%d  max_tok %lld  max_seqs %lld%s\n\n",
             p.meta.name ? p.meta.name : "?", p.meta.arch_id ? p.meta.arch_id : "?",
             p.ctx.rank, p.ctx.world_size, (long long)p.ctx.max_tok, (long long)p.ctx.max_seqs,
             (p.ctx.scope && *p.ctx.scope) ? fmt("  scope '%s'", p.ctx.scope).c_str() : "");

    /* ---------------------------------------------------------------- the ops */
    for (size_t i = 1; i < p.ops.size(); ++i) {
        const OpInfo& o = p.ops[i];

        std::string params = o.base.str();
        s += o.op;
        if (!params.empty()) { s += "  "; s += params; }
        s += "\n";

        /* Device first and unlabelled, the §2.2 form. A host-only kernel set (a primary target,
         * not a fallback) gets a labelled host table instead of a device table of holes. */
        bool dev = domain_has_any(o, RAD_DOMAIN_DEVICE);
        bool hst = domain_has_any(o, RAD_DOMAIN_HOST);
        bool printed_device = dev || !hst;

        if (printed_device) dump_bucket_table(s, o, RAD_DOMAIN_DEVICE);
        if (hst) {
            if (printed_device) s += "    host domain:\n";
            dump_bucket_table(s, o, RAD_DOMAIN_HOST);
        }

        if (printed_device) dump_matched(s, o, RAD_DOMAIN_DEVICE);
        if (hst)            dump_matched(s, o, RAD_DOMAIN_HOST);

        if (!o.weights.empty()) {
            std::vector<std::string> names;
            for (rad_weight h : o.weights)
                names.push_back(h < p.weights.size() ? p.weights[h].name : fmt("<bad handle %u>", h));
            s += fmt("      weights  %s\n", join(names).c_str());
        }

        /* Buffers this op writes, read off the plan: a transient whose live range begins here is
         * defined by this op. */
        std::vector<std::string> defs, ends;
        const int32_t nops = (int32_t)p.ops.size() - 1;
        for (size_t bi = 1; bi < p.buffers.size(); ++bi) {
            const BufferInfo& b = p.buffers[bi];
            if (b.decl.kind != RAD_BUF_TRANSIENT) continue;
            /* Whole-program liveness means no declared use; attributing it to the first and last
             * op would be wrong. The buffer plan section counts these. */
            if (b.first_def <= 0 && b.last_use >= nops) continue;
            if (b.first_def == (int32_t)i) defs.push_back(b.name);
            else if (b.last_use == (int32_t)i) ends.push_back(b.name);
        }
        if (!defs.empty()) s += fmt("      writes   %s\n", join(defs).c_str());
        if (!ends.empty()) s += fmt("      last use %s\n", join(ends).c_str());

        s += "\n";
    }

    /* ---------------------------------------------------------------- the weight manifest */
    int64_t wtotal = 0;
    for (size_t i = 1; i < p.weights.size(); ++i) wtotal += p.weights[i].stored_bytes;
    s += fmt("---- weights: %zu, %s held by rank %d of %d\n",
             p.weights.size() - 1, humanb(wtotal).c_str(), p.ctx.rank, p.ctx.world_size);
    append_row(s, fmt("  %-40s %12s %6s  %-12s %6s %7s %6s %6s  %-16s %-5s %-5s",
                      "name", "bytes", "align", "access", "layer", "expert", "first", "last",
                      "layout", "tier", "site"));
    for (size_t i = 1; i < p.weights.size(); ++i) {
        const WeightInfo& w = p.weights[i];
        /* `layout` is the storage form the resolved kernel asked for, which is why stored bytes may
         * differ from logical ones. tier and site stay blank: the planner fills them after
         * declare, and a default would read as a decision. */
        append_row(s, fmt("  %-40s %12s %6lld  %-12s %6d %7d %6d %6d  %-16s %-5s %-5s",
                          w.name.c_str(), humanb(w.stored_bytes).c_str(), (long long)w.align,
                          bld_access_name(w.decl.access), w.decl.group.layer, w.decl.group.expert,
                          w.first_use_op, w.last_use_op,
                          w.layout_tag.empty() ? "as declared" : w.layout_tag.c_str(), "", ""));
    }
    s += "\n";

    /* ---------------------------------------------------------------- the buffer plan */
    int64_t sum = 0, peak = 0, unknown = 0, persist = 0;
    const int32_t last_op = (int32_t)p.ops.size() - 1;
    for (size_t i = 1; i < p.buffers.size(); ++i) {
        const BufferInfo& b = p.buffers[i];
        if (b.decl.domain == RAD_DOMAIN_HOST) continue;
        if (b.decl.kind == RAD_BUF_TRANSIENT) {
            sum += b.bytes;
            if (b.first_def <= 0 && b.last_use >= last_op) ++unknown;
        } else {
            persist += b.bytes;
        }
    }
    for (int32_t t = 1; t <= std::max<int32_t>(last_op, 1); ++t) {
        int64_t live = 0;
        for (size_t i = 1; i < p.buffers.size(); ++i) {
            const BufferInfo& b = p.buffers[i];
            if (b.decl.kind != RAD_BUF_TRANSIENT || b.decl.domain == RAD_DOMAIN_HOST) continue;
            if (b.first_def <= t && t <= b.last_use) live += b.bytes;
        }
        peak = std::max(peak, live);
    }

    s += fmt("---- buffer plan: arena %s, scratch %s\n",
             humanb(p.arena_bytes).c_str(), humanb(p.scratch_bytes).c_str());
    s += fmt("  transient   %12s packed from %s unshared  (peak live %s, waste %s)\n",
             humanb(p.arena_bytes - persist).c_str(), humanb(sum).c_str(),
             humanb(peak).c_str(), humanb(p.arena_bytes - persist - peak).c_str());
    s += fmt("  persist     %12s\n", humanb(persist).c_str());
    /* Two arenas, coloured separately (see plan_buffers). */
    if (p.host_arena_bytes)
        s += fmt("  host arena  %12s   (its own offset space and its own base pointer)\n",
                 humanb(p.host_arena_bytes).c_str());
    if (unknown)
        s += fmt("  %lld transient buffer(s) had no declared use and are assumed live for the "
                 "whole program -- that is the arena's waste, and it goes away when ops name "
                 "their buffer operands\n", (long long)unknown);

    std::vector<const BufferInfo*> big;
    for (size_t i = 1; i < p.buffers.size(); ++i) big.push_back(&p.buffers[i]);
    std::sort(big.begin(), big.end(),
              [](const BufferInfo* a, const BufferInfo* b) { return a->bytes > b->bytes; });
    for (size_t i = 0; i < big.size() && i < 5; ++i)
        s += fmt("    %-32s %12s  %-9s live [%d,%d] @ %lld\n", big[i]->name.c_str(),
                 humanb(big[i]->bytes).c_str(), bld_buf_kind_name(big[i]->decl.kind),
                 big[i]->first_def, big[i]->last_use, (long long)big[i]->arena_offset);
    s += "\n";

    /* ---------------------------------------------------------------- KV groups */
    if (p.kv_groups.size() > 1) {
        s += fmt("---- kv groups: %zu\n", p.kv_groups.size() - 1);
        for (size_t i = 1; i < p.kv_groups.size(); ++i) {
            const KVGroupInfo& g = p.kv_groups[i];
            s += fmt("  %-20s %-7s block %-5lld %12s/block %12s/state  %zu layer(s)\n",
                     g.name.c_str(), bld_kv_kind_name(g.decl.kind), (long long)g.block_size,
                     humanb(g.bytes_per_block).c_str(), humanb(g.bytes_per_state).c_str(),
                     g.layers.size());
        }
        s += "\n";
    }

    /* ---------------------------------------------------------------- the drafter (spec §10) */
    if (p.drafter.kind != RAD_DRAFT_NONE) {
        const bool blk = p.drafter.kind == RAD_DRAFT_BLOCK;
        s += fmt("---- drafter: '%s', %s, %d token(s) a step\n",
                 p.drafter_name.c_str(),
                 blk ? "RAD_DRAFT_BLOCK -- a context pass, then one pass fills the whole block"
                     : "RAD_DRAFT_SERIAL -- one dependent pass a token",
                 p.drafter.depth);
        s += fmt("  proposal   %s\n",
                 p.drafter.proposal
                     ? fmt("%s [%lld a row]", p.buffers[p.drafter.proposal].name.c_str(),
                           (long long)p.drafter.proposal_pitch).c_str()
                     : "the sampler's token buffer (this head's logits go through the chain)");
        if (blk)
            s += fmt("  block      %lld row(s) a sequence, the head over the last %d, mask token "
                     "%d, drafter window %lld\n",
                     (long long)p.drafter.block, p.drafter.depth,
                     (int)p.drafter.mask_token, (long long)p.drafter.window);
        if (p.drafter_shared_kv)
            s += fmt("  cache      none of its own: it reads the trunk's, owes no history, and "
                     "every round runs at the committed position%s\n",
                     p.drafter_pos_offset ? " (rotated one past it)" : "");
        s += "\n";
    }

    /* ---------------------------------------------------------------- tuning (spec §15) */
    if (!p.tune_requests.empty()) {
        s += fmt("---- tuning requests: %zu  (rad-tune benchmarks each on this machine)\n",
                 p.tune_requests.size());
        for (const auto& t : p.tune_requests)
            s += fmt("  %-24s %-14s %2d axes  %s\n", t.kernel.c_str(), t.plugin.c_str(),
                     t.info ? t.info->n_tunables : 0, t.geom.str().c_str());
        s += "\n";
    }

    /* ---------------------------------------------------------------- fusions (spec §2.3)
     *
     * Launches that stand in for several ops. A kernel with `replaces` declares, as data, the
     * conventional chain it computes; rad-check checks it against that chain. This section is
     * coverage, not advice: an op with no fusion is not a defect, and a fusion that did not fire
     * is usually banded on purpose. */
    {
        std::vector<std::string> lines;
        int n_fused = 0, n_ops_replaced = 0;
        for (size_t i = 1; i < p.ops.size(); ++i) {
            const OpInfo& o = p.ops[i];
            for (size_t b = 0; b < o.bands.size(); ++b) {
                for (int d = 0; d < RAD_N_DOMAINS; ++d) {
                    const Resolved& r = o.bands[b].dom[d];
                    if (!r || !r.row || !r.row->info || !r.row->info->replaces) continue;
                    /* Walk the chain; the hook ends it by declining the step past the end. */
                    std::string chain;
                    int n_step = 0;
                    for (int step = 0; step < RAD_MAX_FUSE_OPD; ++step) {
                        RadFuseStep fs{};
                        /* RAD_FUSE_PRESENT_ALL: the declared graph records ops and geometry, not
                         * which optional operands a given issue passes (a per-step property). So
                         * the row means "what this kernel stands in for", described for the operand
                         * set the kernel publishes, not "what ran on the last token". */
                        const int st = r.row->info->replaces(r.geom.params(), r.geom.n_params(),
                                                             RAD_FUSE_PRESENT_ALL,
                                                             step, &fs);
                        if (st != RAD_OK) break;
                        if (!fs.op) break;
                        chain += (chain.empty() ? "" : " -> ");
                        chain += fs.op;
                        ++n_step;
                    }
                    if (n_step < 2) continue;   /* a one-op "chain" is not a fusion */
                    ++n_fused;
                    n_ops_replaced += n_step;
                    lines.push_back(fmt("  %-24s %-28s %s", o.op.c_str(), r.row->info->name,
                                        chain.c_str()));
                }
            }
        }
        if (n_fused > 0) {
            std::sort(lines.begin(), lines.end());
            lines.erase(std::unique(lines.begin(), lines.end()), lines.end());
            s += fmt("---- fusions: %d resolved kernel(s) stand in for %d op(s)\n",
                     n_fused, n_ops_replaced);
            s += "  each row is a kernel and the conventional chain it declares it computes;\n"
                 "  rad-check runs that chain out of the same plugin and compares the bytes\n";
            for (const std::string& l : lines) s += l + "\n";
            s += "\n";
        }
    }

    /* ---------------------------------------------------------------- plugin notes */
    if (!p.notes.empty()) {
        s += "---- notes from the architecture plugin\n";
        for (const auto& n : p.notes) s += "  " + n + "\n";
        s += "\n";
    }

    /* ---------------------------------------------------------------- what did not resolve */
    if (!p.misses.empty()) {
        s += fmt("---- unresolved: %zu op/band/domain combination(s)\n", p.misses.size());
        s += "  no DEVICE kernel is fatal for that band; no HOST kernel removes host-site\n"
             "  placement for that op and nothing else (spec §2.1)\n";
        for (const auto& m : p.misses) s += "  " + m.line() + "\n";
        s += "\n";
    }

    if (errors && !errors->empty()) {
        s += fmt("---- declare errors: %zu\n", errors->size());
        for (const auto& e : *errors) s += "  " + e + "\n";
        s += "\n";
    }

    return s;
}

/* ================================================================== the graph, as JSON
 *
 * dump_graph_str's content and more, for the server's Model view. It walks the same Program with
 * the same helpers (origin, constraint_str, the fuse hook), so the two views cannot disagree.
 *
 * The extra detail lives in the per-op panel: the schema doc and operands, declared geometry,
 * each band's winner and the candidates that lost with their reasons, the kernel's own
 * description, what it stands in for, the weights read (size, layout) and the buffers whose lives
 * begin and end at the op.
 *
 * Ops are grouped by (op name, every band's resolution), so a per-layer op is one entry with a
 * count. Per-occurrence facts (weights, buffers, op index) are the first declaration's, and the
 * page labels them so. */

/* A RadFuseStep's own geometry, which differs from the fused op's: `qk_norm_rope_gate` replaces
 * an rmsnorm at M = T * n_head, a product absent from the fused op's parameters. */
static std::string params_str(const RadParam* p, int n) {
    std::string s;
    for (int i = 0; i < n; ++i) {
        if (!p[i].key) continue;
        if (!s.empty()) s += " ";
        switch (p[i].kind) {
            case RAD_P_INT:   s += fmt("%s=%lld", p[i].key, p[i].ival); break;
            case RAD_P_STR:   s += fmt("%s=%s", p[i].key, p[i].sval ? p[i].sval : ""); break;
            case RAD_P_F64:   s += fmt("%s=%g", p[i].key, p[i].dval); break;
            case RAD_P_RANGE: s += fmt("%s=[%lld,%lld]", p[i].key, p[i].ival, p[i].ihi); break;
            default:          s += p[i].key; break;
        }
    }
    return s;
}

/* The chain this kernel declares it computes, with each step's geometry, which makes the claim
 * checkable and which a reader cannot derive. */
static json fuse_steps(const Resolved& r) {
    json out = json::array();
    if (!r || !r.row || !r.row->info || !r.row->info->replaces) return out;
    for (int step = 0; step < RAD_MAX_FUSE_OPD; ++step) {
        RadFuseStep fs{};
        /* RAD_FUSE_PRESENT_ALL: see the same call in dump_graph_str. */
        if (r.row->info->replaces(r.geom.params(), r.geom.n_params(),
                                  RAD_FUSE_PRESENT_ALL, step, &fs) != RAD_OK) break;
        if (!fs.op) break;
        json j;
        j["op"]     = fs.op;
        j["params"] = params_str(fs.p, fs.n_p);
        out.push_back(std::move(j));
    }
    if (out.size() < 2) return json::array();   /* a one-op "chain" is not a fusion */
    return out;
}

/* The optional ABI hooks a kernel supplies, which decide what the tools can do with it: without
 * `opd_shape` rad-check skips the row, without `unrelayout` a dense oracle has nothing to compare,
 * without `replaces` a fusion is unchecked. */
static json kernel_hooks(const RadKernelInfo* k) {
    json h = json::array();
    if (k->init)       h.push_back("init");
    if (k->scratch)    h.push_back("scratch");
    if (k->layout)     h.push_back("layout");
    if (k->relayout)   h.push_back("relayout");
    if (k->unrelayout) h.push_back("unrelayout");
    if (k->opd_shape)  h.push_back("opd_shape");
    if (k->replaces)   h.push_back("replaces");
    return h;
}

static json resolved_json(const Resolved& r) {
    if (!r || !r.row || !r.row->info) return nullptr;
    const RadKernelInfo* k = r.row->info;
    json j;
    j["kernel"]   = k->name ? k->name : "?";
    j["plugin"]   = r.row->plugin;
    j["priority"] = k->priority;
    j["geometry"] = r.geom.str();
    j["domain"]   = bld_domain_name(k->domain);
    /* The kernel author's own description. */
    if (k->family   && *k->family)   j["family"]   = k->family;
    if (k->computes && *k->computes) j["computes"] = k->computes;
    if (k->shape    && *k->shape)    j["shape"]    = k->shape;
    if (k->dtypes   && *k->dtypes)   j["dtypes"]   = k->dtypes;
    if (r.scratch_bytes) j["scratch"] = r.scratch_bytes;
    if (k->n_tunables > 0) {
        j["tuned"]        = r.choice;
        j["tuned_source"] = r.tune_source.empty() ? "axis defaults" : r.tune_source;
        /* Every axis with its full range, not just the choice, so a reader can tell a tuned
           choice from a forced one (an axis with no alternatives at this shape). */
        json ax = json::array();
        int64_t space = 1;
        for (int i = 0; i < k->n_tunables; ++i) {
            const RadTunable& t = k->tunables[i];
            json a;
            a["key"]     = t.key ? t.key : "?";
            a["default"] = t.deflt;
            if (t.doc && *t.doc) a["doc"] = t.doc;
            json vals = json::array();
            for (int v = 0; v < t.n_values; ++v) vals.push_back(t.values[v]);
            a["values"] = std::move(vals);
            ax.push_back(std::move(a));
            space *= (int64_t)t.n_values;
        }
        j["tunables"]   = std::move(ax);
        j["tune_space"] = space;
    }
    /* The kernel's declared constraints, which selection matched it on. With the band's
     * `considered` list and `priority`, they explain why it won. */
    json cons = json::array();
    for (int i = 0; i < k->n_constraints; ++i) cons.push_back(constraint_str(k->constraints[i]));
    j["constraints"] = std::move(cons);
    j["hooks"] = kernel_hooks(k);

    /* Names only, for the flow's chip and the filter's haystack; the steps carry the geometries. */
    json names = json::array();
    json steps = fuse_steps(r);
    for (const json& st : steps) names.push_back(st["op"]);
    j["replaces"]    = std::move(names);
    j["fuse_steps"]  = std::move(steps);
    return j;
}

/* Every candidate for this band and what refused it (SelectionTrace::candidates; build_bands keeps
 * only the miss reason). Recomputed rather than stored on every Band, which would hold megabytes of
 * strings for the process lifetime; the cost is one pass over one op's rows per distinct band
 * table. `record = false` keeps the selection table's "asked" counts to the engine's own
 * queries. */
static json considered_json(Registry& reg, const OpInfo& o, const Band& b, int dom) {
    Geometry q = o.base;
    if (!o.ranged_key.empty()) q.set_i(o.ranged_key, b.hi == INT64_MAX ? o.range_hi : b.hi);
    SelectionTrace t;
    reg.select(o.op, q, dom, &t, /*record=*/false);

    json out = json::array();
    for (const SelectionTrace::Candidate& c : t.candidates) {
        json j;
        j["kernel"]   = c.kernel;
        j["plugin"]   = c.plugin;
        j["priority"] = c.priority;
        j["ok"]       = c.ok;
        j["why"]      = c.why;
        out.push_back(std::move(j));
    }
    return out;
}

/* The op's schema. The operand list matters most: `RadArgs.t` is positional, so it is the only
 * way to read a fused kernel's `from` wiring. */
static json schema_json(const RadOpSchema* s) {
    if (!s) return nullptr;
    json j;
    if (s->doc && *s->doc) j["doc"] = s->doc;
    json ps = json::array();
    for (int i = 0; i < s->n_params; ++i) {
        json p;
        p["key"]  = s->params[i].key ? s->params[i].key : "?";
        p["type"] = s->params[i].type == RAD_P_STR ? "str" : "int";
        p["need"] = s->params[i].required == RAD_REQUIRED ? "required"
                  : s->params[i].required == RAD_DERIVED  ? "derived" : "optional";
        ps.push_back(std::move(p));
    }
    j["params"] = std::move(ps);
    json os = json::array();
    for (int i = 0; i < s->n_operands; ++i) {
        const int role = s->operands[i].role;
        json p;
        p["name"] = s->operands[i].name ? s->operands[i].name : "?";
        p["role"] = role == RAD_OPD_OUT ? "out" : role == RAD_OPD_INOUT ? "inout"
                  : role == RAD_OPD_WEIGHT ? "weight" : role == RAD_OPD_WTAB ? "wtab" : "in";
        p["optional"] = s->operands[i].optional != 0;
        os.push_back(std::move(p));
    }
    j["operands"] = std::move(os);
    return j;
}

static std::string shape_str(const int64_t* dims, uint32_t rank) {
    if (!rank) return "";
    std::string s;
    for (uint32_t i = 0; i < rank; ++i) s += (i ? " x " : "") + std::to_string((long long)dims[i]);
    return s;
}

/* A weight one declaration of this op reads: its stored size, and the layout the winning kernel
 * demanded (why stored bytes may differ from logical ones). */
static json weight_json(const WeightInfo& w) {
    json j;
    j["name"]    = w.name;
    j["bytes"]   = w.stored_bytes;
    j["logical"] = w.logical_bytes;
    j["dtype"]   = rad_dtype_name(w.stored_dtype);
    j["shape"]   = shape_str(w.stored_shape, w.stored_rank);
    j["layout"]  = w.layout_tag.empty() ? std::string("as declared") : w.layout_tag;
    j["access"]  = bld_access_name(w.decl.access);
    j["tier"]    = tier_name(w.tier);
    j["site"]    = site_name(w.site);
    j["layer"]   = w.decl.group.layer;
    if (w.decl.group.expert >= 0) j["expert"] = w.decl.group.expert;
    return j;
}

std::string dump_graph_json(const Program& p, Registry* r) {
    Registry& reg = r ? *r : rad_tools_registry();
    json root;
    json meta;
    meta["arch"]     = p.meta.arch_id ? p.meta.arch_id : "";
    meta["name"]     = p.meta.name ? p.meta.name : "";
    meta["quant"]    = p.meta.quant ? p.meta.quant : "";
    meta["n_layers"] = p.meta.n_layers;
    meta["n_embd"]   = p.meta.n_embd;
    meta["n_vocab"]  = p.meta.n_vocab;
    meta["rank"]     = p.ctx.rank;
    meta["world"]    = p.ctx.world_size;
    meta["max_tok"]  = p.ctx.max_tok;
    meta["max_seqs"] = p.ctx.max_seqs;
    root["model"] = std::move(meta);

    int64_t wtotal = 0;
    for (size_t i = 1; i < p.weights.size(); ++i) wtotal += p.weights[i].stored_bytes;

    json counts;
    counts["ops"]       = (int64_t)p.ops.size() - 1;
    counts["weights"]   = (int64_t)p.weights.size() - 1;
    counts["buffers"]   = (int64_t)p.buffers.size() - 1;
    counts["kv_groups"] = (int64_t)p.kv_groups.size();
    counts["weight_bytes"]     = wtotal;
    counts["arena_bytes"]      = p.arena_bytes;
    counts["host_arena_bytes"] = p.host_arena_bytes;
    counts["scratch_bytes"]    = p.scratch_bytes;
    root["counts"] = std::move(counts);

    /* The plugin hierarchy: an override is a plugin ahead of the default in this list. */
    json plugs = json::array();
    for (const Plugin& pl : reg.plugins()) {
        json j;
        j["name"]    = pl.name;
        j["order"]   = pl.order;
        j["version"] = pl.version;
        j["kernels"] = pl.n_kernels;
        j["schemas"] = pl.n_schemas;
        if (!pl.description.empty())  j["description"]  = pl.description;
        if (!pl.build_target.empty()) j["build_target"] = pl.build_target;
        if (!pl.arch_id.empty())      j["arch"] = pl.arch_id + (pl.arch_quant.empty()
                                                    ? std::string() : " " + pl.arch_quant);
        plugs.push_back(std::move(j));
    }
    root["plugins"] = std::move(plugs);

    /* One entry per distinct (op, band table), with how many declarations collapsed into it. */
    std::vector<std::string> order;   /* unique keys, first-seen order */
    std::vector<std::string> seq;     /* every declared op, in order, by key */
    std::unordered_map<std::string, json> group;
    std::unordered_map<std::string, int64_t> tally;

    const int32_t nops = (int32_t)p.ops.size() - 1;

    for (size_t i = 1; i < p.ops.size(); ++i) {
        const OpInfo& o = p.ops[i];
        json bands = json::array();
        std::string key = o.op;
        for (const Band& b : o.bands) {
            json jb;
            jb["span"] = b.span;
            jb["hi"]   = b.hi;   /* the page picks the widest band as the node's headline */
            for (int d = 0; d < RAD_N_DOMAINS; ++d) {
                json rj = resolved_json(b.dom[d]);
                const char* dn = bld_domain_name(d);
                if (!rj.is_null()) jb[dn] = std::move(rj);
                else if (!b.miss[d].empty())  jb[std::string(dn) + "_miss"] = b.miss[d];
            }
            /* The grouping key is formed before the candidate lists are added: `considered`
             * depends only on op and band, so hashing it would add cost and nothing else. */
            key += "|" + jb.dump();
            for (int d = 0; d < RAD_N_DOMAINS; ++d)
                jb[std::string(bld_domain_name(d)) + "_considered"] =
                    considered_json(reg, o, b, d);
            bands.push_back(std::move(jb));
        }
        seq.push_back(key);
        auto it = group.find(key);
        if (it != group.end()) { tally[key] += 1; continue; }

        json j;
        j["op"]       = o.op;
        j["index"]    = o.index;
        j["geometry"] = o.base.str();
        j["ranged"]   = o.ranged_key;
        if (!o.ranged_key.empty()) { j["range_lo"] = o.range_lo; j["range_hi"] = o.range_hi; }
        /* Read through an atomic_ref: the step path stores this flag from the engine thread
         * (Ctx::flush_issued); a plain read here would race it. */
        j["issued"] = std::atomic_ref<const bool>(o.issued).load(std::memory_order_relaxed);
        j["schema"]   = schema_json(o.schema ? o.schema : reg.schema(o.op));

        json ws = json::array();
        for (rad_weight w : o.weights)
            if (w && w < p.weights.size()) ws.push_back(weight_json(p.weights[w]));
        j["weights"] = std::move(ws);

        /* Buffers whose lives begin or end at this op, read off the plan as in the text dump.
         * Whole-program buffers (no declared use) are skipped. */
        json writes = json::array(), frees = json::array();
        for (size_t bi = 1; bi < p.buffers.size(); ++bi) {
            const BufferInfo& bf = p.buffers[bi];
            if (bf.decl.kind != RAD_BUF_TRANSIENT) continue;
            if (bf.first_def <= 0 && bf.last_use >= nops) continue;
            if (bf.first_def != (int32_t)i && bf.last_use != (int32_t)i) continue;
            json jb;
            jb["name"]   = bf.name;
            jb["bytes"]  = bf.bytes;
            jb["domain"] = bld_domain_name(bf.decl.domain);
            jb["live"]   = fmt("[%d, %d]", bf.first_def, bf.last_use);
            if (bf.first_def == (int32_t)i) writes.push_back(std::move(jb));
            else                            frees.push_back(std::move(jb));
        }
        j["writes"] = std::move(writes);
        j["frees"]  = std::move(frees);
        j["bands"]  = std::move(bands);
        group.emplace(key, std::move(j));
        order.push_back(key);
        tally[key] = 1;
    }

    std::unordered_map<std::string, int> index_of;
    json ops = json::array();
    for (const std::string& key : order) {
        index_of[key] = (int)ops.size();
        json j = group[key];
        j["count"] = tally[key];
        ops.push_back(std::move(j));
    }

    /* The declared order as indices into `ops`, which the grouping dropped and the diagram needs.
     * One small integer per declared op, far smaller than emitting every op in full. */
    json ord = json::array();
    std::vector<int> first_at((size_t)ops.size(), -1), last_at((size_t)ops.size(), -1);
    for (size_t k = 0; k < seq.size(); ++k) {
        const int gi = index_of[seq[k]];
        ord.push_back(gi);
        if (first_at[(size_t)gi] < 0) first_at[(size_t)gi] = (int)k;
        last_at[(size_t)gi] = (int)k;
    }
    /* First and last position of the group's declarations in the flow. */
    for (size_t k = 0; k < ops.size(); ++k) {
        ops[k]["first_at"] = first_at[k];
        ops[k]["last_at"]  = last_at[k];
    }
    root["ops"]   = std::move(ops);
    root["order"] = std::move(ord);

    json kvs = json::array();
    for (const KVGroupInfo& g : p.kv_groups) {
        if (g.name.empty() || g.layers.empty()) continue;
        json j;
        j["name"]            = g.name;
        j["kind"]            = bld_kv_kind_name(g.decl.kind);
        j["layers"]          = (int64_t)g.layers.size();
        j["block_size"]      = g.block_size;
        j["bytes_per_block"] = g.bytes_per_block;
        j["bytes_per_state"] = g.bytes_per_state;
        kvs.push_back(std::move(j));
    }
    root["kv_groups"] = std::move(kvs);

    if (p.drafter.kind != RAD_DRAFT_NONE) {
        json d;
        d["name"]  = p.drafter_name;
        d["kind"]  = p.drafter.kind == RAD_DRAFT_BLOCK ? "block" : "serial";
        d["depth"] = p.drafter.depth;
        if (p.drafter.kind == RAD_DRAFT_BLOCK) {
            d["window"] = p.drafter.window;
            d["block"]  = p.drafter.block;
        }
        root["drafter"] = std::move(d);
    }

    json notes = json::array();
    for (const std::string& n : p.notes) notes.push_back(n);
    root["notes"] = std::move(notes);

    json misses = json::array();
    for (const Program::Miss& m : p.misses) {
        json j;
        j["op"] = m.op; j["geometry"] = m.geom; j["span"] = m.span;
        j["domain"] = bld_domain_name(m.domain); j["why"] = m.why;
        misses.push_back(std::move(j));
    }
    root["misses"] = std::move(misses);
    return root.dump();
}

}  /* namespace rad */
