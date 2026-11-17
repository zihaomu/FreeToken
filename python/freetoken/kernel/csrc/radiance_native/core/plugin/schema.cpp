/* schema.cpp -- the op-schema registry and the check an architecture plugin's declared parameters
 * go through. spec.md §2.3.
 *
 * Without a schema, a typo and an unimplemented op give the same "nothing resolved". With one, a
 * misspelled op fails declare with the near-miss named. The core learns the op vocabulary from
 * the plugins, so new ops need no core change.
 */
#include "registry.h"

#include <algorithm>
#include <cstring>

namespace rad {

namespace {

/* Levenshtein, O(nm). Runs on short names, once per failed declare. */
int edit_distance(std::string_view a, std::string_view b) {
    const size_t n = b.size();
    std::vector<int> prev(n + 1), cur(n + 1);
    for (size_t j = 0; j <= n; ++j) prev[j] = (int)j;
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = (int)i;
        for (size_t j = 1; j <= n; ++j) {
            const int sub = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
            cur[j] = std::min({ sub, prev[j] + 1, cur[j - 1] + 1 });
        }
        prev.swap(cur);
    }
    return prev[n];
}

/* How far a name may be from a known one and still be suggested: one edit for a short name, up
 * to three for a long one. A wrong suggestion is worse than none. */
bool close_enough(std::string_view a, std::string_view b, int d) {
    const size_t len = std::max(a.size(), b.size());
    const int budget = len <= 4 ? 1 : (len <= 10 ? 2 : 3);
    return d <= budget;
}

std::string suggest(std::string_view want, const std::vector<std::string>& known) {
    std::string best;
    int best_d = INT32_MAX;
    for (const std::string& k : known) {
        const int d = edit_distance(want, k);
        if (d < best_d || (d == best_d && k < best)) { best_d = d; best = k; }
    }
    if (best.empty() || !close_enough(want, best, best_d)) return "";
    return best;
}

}  /* namespace */

const char* param_kind_name(int kind) {
    switch (kind) {
        case RAD_P_INT:   return "an integer";
        case RAD_P_STR:   return "a string";
        case RAD_P_RANGE: return "a range";
        case RAD_P_F64:   return "a float";
        default:          return "of an unknown kind";
    }
}

const RadOpSchema* Registry::schema(std::string_view op) const {
    auto it = set_->schemas.find(std::string(op));
    return it == set_->schemas.end() ? nullptr : it->second.schema;
}

const Plugin* Registry::schema_owner(std::string_view op) const {
    auto it = set_->schemas.find(std::string(op));
    return it == set_->schemas.end() ? nullptr : plugin(it->second.plugin);
}

std::vector<std::string> Registry::op_names() const {
    std::vector<std::string> out;
    out.reserve(set_->schemas.size());
    for (const auto& kv : set_->schemas) out.push_back(kv.first);
    /* Include ops that have rows but no schema (check_complete() reports those), so a near-miss
     * search can still name them. */
    for (const KernelRow& r : set_->rows)
        if (set_->schemas.find(r.info->op) == set_->schemas.end()) out.emplace_back(r.info->op);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::string Registry::near_miss(std::string_view op) const {
    return suggest(op, op_names());
}

int Registry::validate(std::string_view op, const RadParam* p, int n_p, std::string* err) const {
    std::string sink;
    if (!err) err = &sink;
    err->clear();

    const std::string name(op);
    auto it = set_->schemas.find(name);
    if (it == set_->schemas.end()) {
        const std::string did = near_miss(op);
        *err = fmt("no loaded kernel plugin declares op '%s'%s", name.c_str(),
                   did.empty() ? "" : fmt("; did you mean '%s'?", did.c_str()).c_str());
        return RAD_E_NOSCHEMA;
    }
    const RadOpSchema& s = *it->second.schema;

    std::vector<std::string> keys;
    keys.reserve((size_t)s.n_params);
    for (int i = 0; i < s.n_params; ++i) keys.emplace_back(s.params[i].key ? s.params[i].key : "");

    int n_ranged = 0;
    std::string ranged_key;
    for (int i = 0; i < n_p; ++i) {
        if (!p[i].key || !p[i].key[0]) {
            *err = fmt("%s: parameter %d has no name", name.c_str(), i);
            return RAD_E_SCHEMA;
        }
        for (int j = 0; j < i; ++j) {
            if (p[j].key && std::strcmp(p[i].key, p[j].key) == 0) {
                *err = fmt("%s: parameter '%s' is given twice", name.c_str(), p[i].key);
                return RAD_E_SCHEMA;
            }
        }

        const RadParamSpec* spec = nullptr;
        for (int j = 0; j < s.n_params; ++j)
            if (keys[(size_t)j] == p[i].key) { spec = &s.params[j]; break; }
        if (!spec) {
            const std::string did = suggest(p[i].key, keys);
            *err = fmt("%s: no parameter '%s' in this op's schema%s", name.c_str(), p[i].key,
                       did.empty() ? "" : fmt("; did you mean '%s'?", did.c_str()).c_str());
            return RAD_E_SCHEMA;
        }

        /* A range is an integer parameter that has not been collapsed yet, so it satisfies an
         * integer slot and nothing else. */
        const bool int_ok = (p[i].kind == RAD_P_INT || p[i].kind == RAD_P_RANGE);
        if (spec->type == RAD_P_INT && !int_ok) {
            *err = fmt("%s: parameter '%s' is %s, but the schema says it is an integer",
                       name.c_str(), p[i].key, param_kind_name(p[i].kind));
            return RAD_E_SCHEMA;
        }
        if (spec->type == RAD_P_STR && p[i].kind != RAD_P_STR) {
            *err = fmt("%s: parameter '%s' is %s, but the schema says it is a string",
                       name.c_str(), p[i].key, param_kind_name(p[i].kind));
            return RAD_E_SCHEMA;
        }
        if (spec->type == RAD_P_STR && !p[i].sval) {
            *err = fmt("%s: string parameter '%s' is null", name.c_str(), p[i].key);
            return RAD_E_SCHEMA;
        }
        /* A float slot takes only a float: an integer would land in `ival` while the kernel
         * reads `dval` (a silent zero eps). Constraints never match on a float (rad_types.h), so
         * only this check catches it. */
        if (spec->type == RAD_P_F64 && p[i].kind != RAD_P_F64) {
            *err = fmt("%s: parameter '%s' is %s, but the schema says it is a float -- declare it "
                       "with RAD_F64()", name.c_str(), p[i].key, param_kind_name(p[i].kind));
            return RAD_E_SCHEMA;
        }

        if (p[i].kind == RAD_P_RANGE) {
            if (p[i].ihi < p[i].ival) {
                *err = fmt("%s: parameter '%s' is the empty range [%lld, %lld]", name.c_str(),
                           p[i].key, p[i].ival, p[i].ihi);
                return RAD_E_SCHEMA;
            }
            if (++n_ranged == 1) ranged_key = p[i].key;
            else {
                /* At most one ranged parameter per op, so the bucket-table index at issue is a
                 * single integer; the run phase depends on this (docs/OPS.md). */
                *err = fmt("%s: '%s' and '%s' are both ranges; an op may have at most one ranged "
                           "parameter", name.c_str(), ranged_key.c_str(), p[i].key);
                return RAD_E_SCHEMA;
            }
        }
    }

    for (int j = 0; j < s.n_params; ++j) {
        /* Only RAD_REQUIRED. RAD_DERIVED is supplied by the kernel: the caller cannot know, e.g.,
         * the block size before a kernel resolves (rad_abi.h, spec §7.2). A caller may still
         * supply one, which pins it. */
        if (s.params[j].required != RAD_REQUIRED) continue;
        bool given = false;
        for (int i = 0; i < n_p && !given; ++i) given = (keys[(size_t)j] == p[i].key);
        if (!given) {
            *err = fmt("%s: required parameter '%s' was not given (declared by plugin '%s')",
                       name.c_str(), keys[(size_t)j].c_str(), it->second.plugin.c_str());
            return RAD_E_SCHEMA;
        }
    }
    return RAD_OK;
}

}  /* namespace rad */
