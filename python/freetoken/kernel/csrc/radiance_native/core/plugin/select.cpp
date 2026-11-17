/* select.cpp -- the constraint selector, the selection table, and the per-instance init/fini.
 * spec.md §2.1 and §16.
 *
 * The selector finds a kernel compiled for this geometry and domain in the highest plugin that has
 * one. A match is necessary, not sufficient: strides, contiguity, alignment and buffer sizes are
 * properties of a call, and the entry point still rejects what it cannot run.
 */
#include "../device/device.h"
#include "registry.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>
#include <cstring>

namespace rad {

namespace {

/* True when every constraint holds. Otherwise `why` names the first one that failed and what it
 * wanted. */
bool row_matches(const KernelRow& r, const Geometry& g, const Geometry& facts,
                 const std::vector<std::string>& skip, std::string* why) {
    static const size_t kFact = std::strlen(RAD_DEVICE_FACT_PREFIX);
    for (int i = 0; i < r.info->n_constraints; ++i) {
        const RadConstraint& c = r.info->constraints[i];
        /* A constraint on a RAD_DERIVED parameter the caller did not name is the answer, read back
         * afterwards, not a predicate; matching on it would make the kernel unreachable
         * (rad_abi.h, spec §7.2). */
        if (c.key && std::find(skip.begin(), skip.end(), c.key) != skip.end()) continue;
        /* A "device." key is a fact about the machine, read from what the engine measured. */
        if (c.key && std::strncmp(c.key, RAD_DEVICE_FACT_PREFIX, kFact) == 0) {
            if (constraint_holds(c, facts)) continue;
            if (why)
                *why = facts.has(c.key)
                           ? constraint_why(c, facts) + " -- this machine, measured at launch"
                           : std::string(c.key) + " is not known: nothing measured this machine";
            return false;
        }
        if (!constraint_holds(c, g)) {
            if (why) *why = constraint_why(c, g);
            return false;
        }
    }
    if (why) *why = "ok";
    return true;
}

/* The op's RAD_DERIVED keys (from its schema) that the caller did not supply. A supplied one is
 * pinned and matches like any other parameter. */
std::vector<std::string> unsupplied_derived(const RadOpSchema* sch, const Geometry& g) {
    std::vector<std::string> keys;
    if (!sch) return keys;
    for (int i = 0; i < sch->n_params; ++i) {
        if (sch->params[i].required != RAD_DERIVED || !sch->params[i].key) continue;
        if (!g.has(sch->params[i].key)) keys.emplace_back(sch->params[i].key);
    }
    return keys;
}

/* The scan select() and select_in() share: the first plugin in hierarchy order with a match
 * supplies the row, the highest priority within it wins, ties to the row declared first. With
 * `only` set, rows of every other plugin are passed over. Every row considered goes into
 * `t.candidates` and each distinct refusal into `misses`. */
const KernelRow* scan_rows(const std::vector<const KernelRow*>* rows, const Geometry& g,
                           int domain, const Geometry& facts,
                           const std::vector<std::string>& derived_keys,
                           const Registry::RowFilter* accept, std::string_view only,
                           SelectionTrace& t, std::vector<std::string>* misses);

/* The question, normalised so two spellings of it fold together in the table. Only the fold key
 * is sorted; the geometry is displayed as asked. */
std::string fold_key(std::string_view op, int domain, const Geometry& g) {
    const std::string q = g.str();     /* named: split_ws returns views into it */
    std::vector<std::string> kv;
    for (auto tok : split_ws(q)) kv.emplace_back(tok);
    std::sort(kv.begin(), kv.end());
    std::string k = std::string(op) + "/" + domain_name(domain) + "/";
    for (const std::string& s : kv) { k += s; k += ';'; }
    return k;
}

}  /* namespace */

bool kernel_constraint(const KernelRow& r, std::string_view key, int c_op, long long* out) {
    if (!r.info) return false;
    for (int i = 0; i < r.info->n_constraints; ++i) {
        const RadConstraint& c = r.info->constraints[i];
        if (c.op != c_op || !c.key || key != c.key) continue;
        if (out) *out = c.ival;
        return true;
    }
    return false;
}

/* ------------------------------------------------------------------ selection */
const KernelRow* Registry::select(std::string_view op, const Geometry& g, int domain,
                                  SelectionTrace* why, bool record, const RowFilter* accept) {
    SelectionTrace local;
    SelectionTrace& t = why ? *why : local;
    t = SelectionTrace{};
    t.op     = std::string(op);
    t.query  = g.str();
    t.domain = domain;

    const std::vector<std::string> derived_keys = unsupplied_derived(schema(op), g);
    const std::vector<const KernelRow*>* rows = rows_for(op);
    std::vector<std::string> misses;
    const KernelRow* best = scan_rows(rows, g, domain, device_facts_, derived_keys, accept, {},
                                      t, &misses);

    if (best) {
        t.kernel   = best->info->name;
        t.plugin   = best->plugin;
        t.priority = best->info->priority;
        /* Each skipped derived key takes the winner's RAD_C_EQ value. A kernel with no EQ on it
         * (e.g. ref's `block_size >= 1`) leaves it unknown rather than zero; the caller decides. */
        for (const std::string& key : derived_keys) {
            SelectionTrace::DerivedParam d;
            d.key   = key;
            d.known = kernel_constraint(*best, key, RAD_C_EQ, &d.value);
            t.derived.push_back(std::move(d));
        }
    } else if (!rows) {
        const std::string did = near_miss(op);
        t.reason = fmt("no loaded kernel plugin implements op '%s'%s", t.op.c_str(),
                       did.empty() ? "" : fmt("; did you mean '%s'?", did.c_str()).c_str());
    } else if (misses.empty()) {
        t.reason = fmt("no %s kernel for op '%s' in any plugin", domain_name(domain), t.op.c_str());
    } else {
        for (size_t i = 0; i < misses.size(); ++i) t.reason += (i ? "; " : "") + misses[i];
    }

    if (record) record_(t.op, domain, g, t);
    return best;
}

const KernelRow* Registry::select_in(std::string_view plugin, std::string_view op,
                                     const Geometry& g, int domain, const RowFilter* accept,
                                     SelectionTrace* why) const {
    SelectionTrace local;
    SelectionTrace& t = why ? *why : local;
    t = SelectionTrace{};
    t.op     = std::string(op);
    t.query  = g.str();
    t.domain = domain;
    if (plugin.empty()) {
        t.reason = "no plugin named";
        return nullptr;
    }
    const std::vector<std::string> derived_keys = unsupplied_derived(schema(op), g);
    std::vector<std::string> misses;
    const KernelRow* best = scan_rows(rows_for(op), g, domain, device_facts_, derived_keys,
                                      accept, plugin, t, &misses);
    if (best) {
        t.kernel   = best->info->name;
        t.plugin   = best->plugin;
        t.priority = best->info->priority;
        for (const std::string& key : derived_keys) {
            SelectionTrace::DerivedParam d;
            d.key   = key;
            d.known = kernel_constraint(*best, key, RAD_C_EQ, &d.value);
            t.derived.push_back(std::move(d));
        }
    } else {
        for (size_t i = 0; i < misses.size(); ++i) t.reason += (i ? "; " : "") + misses[i];
        if (t.reason.empty())
            t.reason = fmt("no %s kernel for op '%s' in %.*s", domain_name(domain), t.op.c_str(),
                           (int)plugin.size(), plugin.data());
    }
    return best;
}

namespace {

const KernelRow* scan_rows(const std::vector<const KernelRow*>* rows, const Geometry& g,
                           int domain, const Geometry& facts,
                           const std::vector<std::string>& derived_keys,
                           const Registry::RowFilter* accept, std::string_view only,
                           SelectionTrace& t, std::vector<std::string>* misses) {
    const KernelRow* best = nullptr;
    if (!rows) return best;
    bool in_group = false;
    int  cur_order = 0;
    for (const KernelRow* r : *rows) {
        if (!only.empty() && r->plugin != only) continue;
        /* Rows of the other domain are neither candidates nor refusals, but their existence is
         * recorded for the placement planner (§5.1). */
        if (r->info->domain != domain) { t.other_domain_has_rows = true; continue; }

        if (!in_group || r->plugin_order != cur_order) {
            /* The hierarchy is absolute: stop at the first plugin with a match, before a lower
             * plugin can outbid it. */
            if (best) break;
            cur_order = r->plugin_order;
            in_group  = true;
        }

        std::string whynot;
        bool ok = row_matches(*r, g, facts, derived_keys, &whynot);
        if (ok && accept && *accept) {
            whynot = (*accept)(*r, g);
            ok = whynot.empty();
        }
        if (ok) {
            t.candidates.push_back({ r->plugin, r->info->name, r->info->priority, true, "ok" });
            /* Highest declared priority wins; a tie goes to the row declared first. */
            if (!best || r->info->priority > best->info->priority) best = r;
        } else {
            t.candidates.push_back({ r->plugin, r->info->name, r->info->priority, false, whynot });
            if (std::find(misses->begin(), misses->end(), whynot) == misses->end())
                misses->push_back(whynot);
        }
    }
    return best;
}

}  /* namespace */

std::string weight_refusal(const RadKernelInfo* ki, const Geometry& g, int operand,
                           const RadEncoding& enc, const int* sel, const RadTensor* planes,
                           int n_sel, std::string_view name) {
    char es[256];
    if (!ki->layout) {
        if (ki->domain == RAD_DOMAIN_DEVICE && n_sel > 1) {
            rad_enc_format(&enc, es, sizeof es);
            return fmt("reads '%.*s' as stored, and it is %d planes of %s", (int)name.size(),
                       name.data(), n_sel, es);
        }
        return std::string();
    }
    RadLayout L{};
    if (ki->layout(g.params(), g.n_params(), operand, &enc, sel, planes, n_sel, &L) ==
        RAD_E_DTYPE) {
        rad_enc_format(&enc, es, sizeof es);
        return fmt("does not read '%.*s' encoded %s", (int)name.size(), name.data(), es);
    }
    return std::string();
}

/* "gemm_nt:sk=8,mb=1;all_reduce:drain=3" -> the axes asked for `op`, or empty. Parsed on every
 * resolve: resolve runs only at startup, and a parsed-once cache would need a mutex. */
static std::string tune_pin_for(const char* spec, std::string_view op) {
    for (const char* p = spec; *p;) {
        const char* semi = std::strchr(p, ';');
        const char* end  = semi ? semi : p + std::strlen(p);
        if (const char* colon = (const char*)std::memchr(p, ':', (size_t)(end - p))) {
            if (op == std::string_view(p, (size_t)(colon - p)))
                return std::string(colon + 1, (size_t)(end - colon - 1));
        }
        p = semi ? semi + 1 : end;
    }
    return {};
}

/* Every axis and its legal values, for the bad-pin error message. */
static std::string tunable_names(const RadKernelInfo* k) {
    std::string s;
    for (int i = 0; i < k->n_tunables; ++i) {
        if (i) s += ", ";
        s += k->tunables[i].key;
        s += '=';
        for (int j = 0; j < k->tunables[i].n_values; ++j) {
            if (j) s += '|';
            s += std::to_string((long long)k->tunables[i].values[j]);
        }
    }
    return s.empty() ? "(nothing tunable)" : s;
}

/* Whether `v` is one of the axis's declared values. A cached or pinned value outside the list is
 * never applied, so a stale cache cannot launch a shape the kernel cannot serve. */
static bool axis_allows(const RadTunable& t, int64_t v) {
    for (int i = 0; i < t.n_values; ++i) if (t.values[i] == v) return true;
    return false;
}

/* Whether the combination passes the kernel's cross-axis test; no test means yes. */
static bool choice_valid(const RadKernelInfo* k, const Geometry& g,
                         const std::vector<std::pair<std::string, int64_t>>& c) {
    if (!k->tune_valid) return true;
    std::vector<RadParam> cp;
    cp.reserve(c.size());
    for (const auto& e : c) {
        RadParam p{};
        p.key = e.first.c_str();
        p.kind = RAD_P_INT;
        p.ival = e.second;
        cp.push_back(p);
    }
    return k->tune_valid(g.params(), g.n_params(), cp.data(), (int)cp.size()) == 1;
}

Resolved Registry::resolve(std::string_view op, const Geometry& g, int domain, SelectionTrace* why,
                           const RowFilter* accept) {
    SelectionTrace local;
    SelectionTrace& t = why ? *why : local;

    Resolved r;
    r.row = select(op, g, domain, &t, true, accept);
    if (!r.row) return r;

    /* The frozen geometry is the query plus the kernel's derived values, so rad_kv_block_size()
     * and RadArgs.p agree: the kernel launches with the block size it was compiled for, and the
     * block manager sizes itself from the same number. */
    r.geom = g;
    for (const SelectionTrace::DerivedParam& d : t.derived)
        if (d.known) r.geom.set_i(d.key, d.value);
    /* Tuned axes. A kernel declares a space (RadTunable); the point chosen is the cached one when
     * this shape was measured on this machine, else each axis's `deflt` (spec §15).
     *
     * The cache key is the frozen geometry after the derived parameters and before the choices
     * go in: rad-tune keys rows on the geometry the builder recorded, so a key that included its
     * own answer could never match.
     *
     * An axis the cached row does not mention keeps its default, and a row key the kernel no
     * longer declares is ignored (tunecache.h), so a library may add or widen axes between builds
     * and keep the measurements that still apply. */
    r.geom_key = TuneCache::canon_geometry(r.geom);

    const RadKernelInfo* ki = r.row->info;
    if (ki->n_tunables > 0) {
        std::vector<std::pair<std::string, int64_t>> chosen;
        chosen.reserve((size_t)ki->n_tunables);
        for (int i = 0; i < ki->n_tunables; ++i)
            chosen.emplace_back(ki->tunables[i].key, ki->tunables[i].deflt);
        r.tune_source = "axis defaults";

        if (!set_->tune.empty()) {
            if (const TuneRow* tr = set_->tune.find(ki->name, r.row->plugin, r.geom_key)) {
                bool took = false;
                for (const auto& kv : TuneCache::parse_choice(tr->choice))
                    for (int i = 0; i < ki->n_tunables; ++i)
                        if (kv.first == ki->tunables[i].key &&
                            axis_allows(ki->tunables[i], kv.second)) {
                            chosen[(size_t)i].second = kv.second;
                            took = true;
                        }
                if (took) r.tune_source = "tuned " + tr->measured;
            }
        }

        /* RADIANCE_TUNE=op:key=value[,key=value][;op:...] pins axes for every kernel serving that
         * op, for debugging without a rebuild (e.g. the all-reduce's drain depth and acquire
         * ordering are axes).
         *
         * An unknown key or a value the axis does not offer is a hard error: a pin that silently
         * did not take is worse than none. A kernel with no axes is not a failed pin (libref
         * declares none), so a device pin does not remove the host band. */
        static const char* const pin = std::getenv("RADIANCE_TUNE");
        if (pin) {
            const std::string want = tune_pin_for(pin, op);
            for (const auto& kv : TuneCache::parse_choice(want)) {
                int idx = -1;
                for (int i = 0; i < ki->n_tunables; ++i)
                    if (kv.first == ki->tunables[i].key) { idx = i; break; }
                if (idx < 0 || !axis_allows(ki->tunables[idx], kv.second)) {
                    RAD_ERR("RADIANCE_TUNE asks %s=%lld of op '%s', which kernel '%s' does not "
                            "offer. It has: %s", kv.first.c_str(), (long long)kv.second,
                            std::string(op).c_str(), ki->name, tunable_names(ki).c_str());
                    r.row = nullptr;
                    return r;
                }
                chosen[(size_t)idx].second = kv.second;
                r.tune_source = "RADIANCE_TUNE";
            }
        }

        /* A combination the kernel's cross-axis test refuses (a cache from another build, or a
         * pin legal per axis but not together) falls back to all defaults, which the kernel must
         * always be able to run. */
        if (!choice_valid(ki, r.geom, chosen)) {
            std::vector<std::pair<std::string, int64_t>> dflt;
            for (int i = 0; i < ki->n_tunables; ++i)
                dflt.emplace_back(ki->tunables[i].key, ki->tunables[i].deflt);
            if (r.tune_source != "axis defaults")
                RAD_WARN("kernel '%s' refuses the %s axes for %s; falling back to its defaults",
                         ki->name, r.tune_source.c_str(), r.geom_key.c_str());
            chosen = std::move(dflt);
            r.tune_source = "axis defaults";
        }

        for (const auto& kv : chosen) r.geom.set_i(kv.first, kv.second);
        r.choice = TuneCache::canon_choice(chosen);
    }
    /* A chainable row runs its chained form wherever chaining is on (RadKernelChainFn); the
     * parameter is outside the tune cache's key, which was taken above. */
    if (r.row->chain && rad_dev_chain_enabled())
        r.geom.set_i("chain", 1);

    /* scratch_bytes stays 0: the scratch hook needs a RadArgs with the operand tensors, which the
     * builder owns and sizes the arena with. */
    return r;
}

/* ------------------------------------------------------------------ the selection table (§16) */
void Registry::record_(const std::string& op, int domain, const Geometry& g,
                       const SelectionTrace& t) {
    const std::string key = fold_key(op, domain, g);
    std::lock_guard<std::mutex> lk(mu_);
    /* The same question, therefore the same answer. */
    const auto at = log_at_.try_emplace(key, log_.size());
    if (!at.second) { ++log_[at.first->second].count; return; }
    SelectionRecord s;
    s.op       = op;
    s.domain   = domain;
    s.query    = g.str();
    s.key      = key;
    s.kernel   = t.kernel;
    s.plugin   = t.plugin;
    s.priority = t.priority;
    s.reason   = t.reason;
    /* Derived keys are kept apart from the query because the caller did not ask for them. ":="
     * is the kernel's answer; "?" is a winner with no EQ on the key, which the caller must fill. */
    for (const SelectionTrace::DerivedParam& d : t.derived) {
        if (!s.derived.empty()) s.derived += ' ';
        s.derived += d.known ? fmt("%s:=%lld", d.key.c_str(), d.value)
                             : fmt("%s:=?", d.key.c_str());
    }
    log_.push_back(std::move(s));
}

size_t Registry::n_selections() const {
    std::lock_guard<std::mutex> lk(mu_);
    return log_.size();
}

void Registry::clear_selections() {
    std::lock_guard<std::mutex> lk(mu_);
    log_.clear();
    log_at_.clear();
}

std::string Registry::selection_table() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (log_.empty()) return "selection table: the selector was never asked anything\n";

    size_t wop = 2, wdom = 3, wq = 8;
    long long total = 0;
    for (const SelectionRecord& s : log_) {
        wop = std::max(wop, s.op.size());
        wdom = std::max(wdom, std::strlen(domain_name(s.domain)));
        wq = std::max(wq, s.query.size());
        total += s.count;
    }

    std::string out = fmt("selection table -- %zu distinct questions, %lld asked\n",
                          log_.size(), total);
    out += fmt("  %-*s  %-*s  %5s  %-*s  %s\n", (int)wop, "op", (int)wdom, "domain", "asked",
               (int)wq, "geometry", "resolved");
    for (const SelectionRecord& s : log_) {
        std::string got = s.kernel.empty()
            ? fmt("-- none: %s", s.reason.c_str())
            : fmt("%s (%s, prio %d)%s%s", s.kernel.c_str(), s.plugin.c_str(), s.priority,
                  s.derived.empty() ? "" : "  derived ", s.derived.c_str());
        out += fmt("  %-*s  %-*s  %5lld  %-*s  %s\n", (int)wop, s.op.c_str(),
                   (int)wdom, domain_name(s.domain), s.count, (int)wq, s.query.c_str(),
                   got.c_str());
    }
    return out;
}

/* ------------------------------------------------------------------ instances */
/* See registry.h: init_instance. */
int Registry::init_instance(Resolved& r, int rank, int world_size) {
    if (!r.row || !r.row->info) return RAD_E_INVAL;
    const RadKernelInfo* k = r.row->info;
    if (!k->init) { r.instance = nullptr; return RAD_OK; }

    void* inst = nullptr;
    /* The parameters init reads: the frozen geometry, plus the engine's tensor-parallel group when
     * one is set. The extended copy lives as long as the instance, as the frozen geometry does,
     * since an init may keep pointers into what it was handed. */
    const Geometry* g = &r.geom;
    if (tp_group_ >= 0) {
        std::lock_guard<std::mutex> lk(mu_);
        init_geoms_.push_back(r.geom);
        init_geoms_.back().set_i("tp_group", tp_group_);
        g = &init_geoms_.back();
    }
    const int st = k->init(g->params(), g->n_params(), rank, world_size, &inst);
    if (st < 0) {
        /* A negative init aborts startup rather than launching a kernel whose setup failed. */
        RAD_ERR("kernel '%s' (%s) failed to initialise for %s: %s", k->name, r.row->plugin.c_str(),
                r.geom.str().c_str(), rad_strerror(st));
        return st;
    }
    r.instance = inst;
    /* Registered even when the instance is null: fini pairs 1:1 with every successful init. */
    std::lock_guard<std::mutex> lk(mu_);
    instances_.emplace_back(k, inst);
    return RAD_OK;
}

void Registry::fini_instances() {
    std::vector<std::pair<const RadKernelInfo*, void*>> take;
    {
        std::lock_guard<std::mutex> lk(mu_);
        take.swap(instances_);
    }
    for (auto it = take.rbegin(); it != take.rend(); ++it)
        if (it->first->fini) it->first->fini(it->second);
    std::lock_guard<std::mutex> lk(mu_);
    init_geoms_.clear();
}

size_t Registry::n_instances() const {
    std::lock_guard<std::mutex> lk(mu_);
    return instances_.size();
}

}  /* namespace rad */
