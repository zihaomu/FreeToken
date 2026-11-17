/* rope_override.h -- --rope-scaling, applied to a container's free-form metadata at load.
 *
 * The architecture reads static YaRN (arch/common/rad_arch.h, yarn_from) from the first of
 * `text_config.rope_parameters.`, `rope_parameters.`, `text_config.rope_scaling.`, `rope_scaling.`
 * that names a rotary type; a DFlash2 drafter reads `draft.rope_parameters.` /
 * `draft.rope_scaling.`. These functions rewrite those keys so a container gains YaRN without
 * reconversion, and the architecture's one reader and checks apply.
 *
 * The rewrite targets the object the reader takes: a Qwen3.8 container says
 * `text_config.rope_parameters.rope_type = default`, so YaRN under `rope_parameters.` would be
 * ignored.
 *
 * Pure functions over parallel string vectors, tested without an engine (tests/yarn_test.cpp
 * checks them against yarn_from). */
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rad {

struct RopeSpec {
    enum Kind { Keep, None, Yarn } kind = Keep;
    double  factor = 0.0;
    int64_t orig = 0;   /* 0: the container's own trained length */
};

/* "yarn:FACTOR[:ORIG]" with FACTOR > 1 and ORIG > 0, or "none"; empty is Keep. False otherwise. */
inline bool rope_spec_parse(const char* s, RopeSpec* out) {
    RopeSpec r;
    if (!s || !*s) { *out = r; return true; }
    if (!std::strcmp(s, "none")) { r.kind = RopeSpec::None; *out = r; return true; }
    if (std::strncmp(s, "yarn:", 5)) return false;
    char* e = nullptr;
    r.factor = std::strtod(s + 5, &e);
    if (e == s + 5 || !(r.factor > 1.0) || r.factor > 1e6) return false;
    if (*e == ':') {
        char* e2 = nullptr;
        const long long o = std::strtoll(e + 1, &e2, 10);
        if (e2 == e + 1 || *e2 || o <= 0) return false;
        r.orig = o;
    } else if (*e) {
        return false;
    }
    r.kind = RopeSpec::Yarn;
    *out = r;
    return true;
}

struct RopeKeys {
    std::vector<std::string>& k;
    std::vector<std::string>& v;

    int find(const std::string& key) const {
        for (size_t i = 0; i < k.size(); ++i)
            if (k[i] == key) return (int)i;
        return -1;
    }
    const std::string* get(const std::string& key) const {
        const int i = find(key);
        return i < 0 ? nullptr : &v[(size_t)i];
    }
    void set(const std::string& key, const std::string& val) {
        const int i = find(key);
        if (i >= 0) { v[(size_t)i] = val; return; }
        k.push_back(key);
        v.push_back(val);
    }
    void erase(const std::string& key) {
        const int i = find(key);
        if (i < 0) return;
        k.erase(k.begin() + i);
        v.erase(v.begin() + i);
    }
    bool any_prefix(const char* pre) const {
        const size_t n = std::strlen(pre);
        for (const auto& s : k)
            if (!s.compare(0, n, pre)) return true;
        return false;
    }
};

/* The reader's search order; must match yarn_from's. */
inline std::vector<std::string> rope_homes(const char* group) {
    if (!*group)
        return { "text_config.rope_parameters.", "rope_parameters.", "text_config.rope_scaling.",
                 "rope_scaling." };
    return { std::string(group) + "rope_parameters.", std::string(group) + "rope_scaling." };
}

/* The object the reader takes and the key naming its type; empty when none names one (the default
 * rotary). */
inline void rope_authoritative(const RopeKeys& kv, const char* group, std::string* home,
                               std::string* type_key) {
    home->clear();
    type_key->clear();
    for (const auto& h : rope_homes(group))
        for (const char* t : { "rope_type", "type" })
            if (kv.get(h + t)) { *home = h; *type_key = h + t; return; }
}

struct RopeResolved {
    std::string type;      /* "" when nothing names one */
    double      factor = 1.0;
    double      orig = 0.0;
};

inline RopeResolved rope_resolve(std::vector<std::string>& k, std::vector<std::string>& v,
                                 const char* group) {
    RopeKeys kv{ k, v };
    RopeResolved r;
    std::string home, tk;
    rope_authoritative(kv, group, &home, &tk);
    if (home.empty()) return r;
    r.type = *kv.get(tk);
    if (const std::string* f = kv.get(home + "factor")) r.factor = std::strtod(f->c_str(), nullptr);
    if (const std::string* o = kv.get(home + "original_max_position_embeddings"))
        r.orig = std::strtod(o->c_str(), nullptr);
    return r;
}

/* Apply `spec` to one rotary group ("" the target, "draft." a DFlash2 drafter). `trained` is the
 * group's trained length, the default ORIG. False, with *why, when the group's rotary type is one
 * YaRN is not stacked over (e.g. linear interpolation). */
inline bool rope_override_group(std::vector<std::string>& k, std::vector<std::string>& v,
                                const RopeSpec& spec, const char* group, int64_t trained,
                                std::string* why) {
    if (spec.kind == RopeSpec::Keep) return true;
    RopeKeys kv{ k, v };
    std::string home, tk;
    rope_authoritative(kv, group, &home, &tk);
    const std::string type = home.empty() ? std::string() : *kv.get(tk);
    if (spec.kind == RopeSpec::None) {
        if (type == "yarn") kv.set(tk, "default");
        return true;
    }
    /* `mrope` is the legacy Qwen2-VL spelling of default with sections. */
    if (!type.empty() && type != "default" && type != "mrope" && type != "yarn") {
        *why = "the " + std::string(*group ? "drafter's" : "model's") + " rotary is '" + type +
               "' (" + tk + "); YaRN over it is not --rope-scaling's to decide";
        return false;
    }
    if (home.empty()) {
        home = *group ? std::string(group) + "rope_parameters."
             : kv.any_prefix("text_config.") ? "text_config.rope_parameters."
                                             : "rope_parameters.";
        tk = home + "rope_type";
    }
    /* An existing YaRN keeps its trained length unless one is given; keys tied to the old factor
     * are erased so the reader derives them for the new one. */
    int64_t orig = spec.orig;
    if (!orig && type == "yarn")
        if (const std::string* o = kv.get(home + "original_max_position_embeddings"))
            orig = std::atoll(o->c_str());
    if (!orig) orig = trained;
    if (orig <= 0) {
        *why = "no trained length to scale from; give one: yarn:FACTOR:ORIG";
        return false;
    }
    char buf[64];
    kv.set(tk, "yarn");
    std::snprintf(buf, sizeof buf, "%.17g", spec.factor);
    kv.set(home + "factor", buf);
    std::snprintf(buf, sizeof buf, "%lld", (long long)orig);
    kv.set(home + "original_max_position_embeddings", buf);
    for (const char* stale : { "attention_factor", "mscale", "mscale_all_dim" })
        kv.erase(home + stale);
    return true;
}

/* The rotary's identity: every key with a path component containing `rope` or `rotary`
 * (rope_parameters.*, rope_scaling.*, rope_theta, partial_rotary_factor, their text_config. and
 * draft. forms, a GGUF's <arch>.rope.*) as sorted key=value, after --rope-scaling. A persisted
 * prefix store is fingerprinted with it, since the same container with YaRN has identical names and
 * shapes but cached keys would be at the wrong angles. Key order and unrelated keys do not
 * matter. */
inline std::string rope_identity(const std::vector<std::string>& k,
                                 const std::vector<std::string>& v) {
    auto names_rotary = [](const std::string& key) {
        size_t s = 0;
        while (s <= key.size()) {
            size_t e = key.find('.', s);
            if (e == std::string::npos) e = key.size();
            const std::string part = key.substr(s, e - s);
            if (part.find("rope") != std::string::npos || part.find("rotary") != std::string::npos)
                return true;
            s = e + 1;
        }
        return false;
    };
    std::vector<std::string> rows;
    for (size_t i = 0; i < k.size() && i < v.size(); ++i)
        if (names_rotary(k[i])) rows.push_back(k[i] + "=" + v[i]);
    std::sort(rows.begin(), rows.end());
    std::string out;
    for (const auto& r : rows) { out += r; out += ';'; }
    return out;
}

}  /* namespace rad */
