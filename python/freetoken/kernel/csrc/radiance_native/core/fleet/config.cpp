/* fleet/config.cpp -- reading the YAML file (fleet/config.h, docs/MULTIMODEL.md §4).
 *
 * The file is parsed into a tree with locations, anchors and merge keys resolved, then walked level
 * by level. Every scalar keeps its source text, so `temp: 1.0` reaches config_parse as "1.0" and is
 * judged by the same rules as the flag was. Errors name the file and line; the first one stops the
 * load, since a later error after a misread key would only mislead.
 */
#include "fleet/config.h"
#include "util/json.h"

#include <ryml.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace rad::fleet {

namespace {

/* ------------------------------------------------------------------ levels */

/* Options that configure the process rather than a model. They travel in every engine's flag list
 * while one process holds one listener (§5.1), so each engine reads the same values. */
const char* const kServerKeys[] = {
    "host", "port", "api-key", "api-key-env", "allow-host", "trusted-proxy", "http-threads",
    "read-timeout", "write-timeout", "keep-alive-timeout", "max-body-mib", "retry-after", "no-cors",
    "live", "radiance-home", "kernels", "download-dir", "proceed-with-incompatible-configuration",
    "debug-accept-reference-kernels",
};

/* The memory level's options: the per-card pair (gpu-memory-mib, gpu-headroom-mib) and the host
 * headroom. Under memory: they apply to every model; a model that states either half of the card
 * pair itself replaces the memory level's pair, which is how models sharing a card divide it. */
const char* const kMemoryFlagKeys[] = { "gpu-headroom-mib", "gpu-memory-mib", "host-headroom-mib" };

/* Options with no YAML key: the switches the YAML spells differently, and help. */
struct Replaced { const char* flag; const char* instead; };
const Replaced kReplaced[] = {
    { "served-model-name", "the entry's name under models: is the model id; add more with aliases:" },
    { "help",              "run radiance --help for the key reference" },
    { "v",                 "set server.log: debug" },
    { "vv",                "set server.log: trace" },
};

/* Flags given more than once on a command line; a YAML sequence repeats them. */
const char* const kRepeatable[] = { "lora", "allow-host", "trusted-proxy" };

/* The fleet's own keys on a model. `defaults:` may carry every one but aliases (read_model_keys). */
const char* const kModelFleetKeys[] = { "aliases", "priority", "pin", "ttl", "flex", "park" };

const char* const kTopKeys[] = { "server", "cards", "memory", "scheduler", "defaults", "models", "groups" };

template <size_t N>
bool in(const char* const (&set)[N], const std::string& k) {
    for (const char* s : set)
        if (k == s) return true;
    return false;
}

template <size_t N>
std::vector<std::string> as_vector(const char* const (&set)[N]) {
    return std::vector<std::string>(std::begin(set), std::end(set));
}

/* ------------------------------------------------------------------ errors */

/* ryml reports through callbacks that must not return. A throw carries the message out to the
 * single catch in parse(); nothing else in this file throws. */
struct YamlError : std::runtime_error {
    size_t line;
    YamlError(const std::string& m, size_t l) : std::runtime_error(m), line(l) {}
};

[[noreturn]] void on_basic(ryml::csubstr msg, ryml::ErrorDataBasic const&, void*) {
    throw YamlError(std::string(msg.str, msg.len), ryml::npos);
}
[[noreturn]] void on_parse(ryml::csubstr msg, ryml::ErrorDataParse const& e, void*) {
    throw YamlError(std::string(msg.str, msg.len), e.ymlloc.line);
}
[[noreturn]] void on_visit(ryml::csubstr msg, ryml::ErrorDataVisit const&, void*) {
    throw YamlError(std::string(msg.str, msg.len), ryml::npos);
}

std::string str(ryml::csubstr s) { return std::string(s.str, s.len); }

size_t edit_distance(const std::string& a, const std::string& b) {
    std::vector<size_t> row(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) row[j] = j;
    for (size_t i = 1; i <= a.size(); ++i) {
        size_t diag = row[0];
        row[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            const size_t up = row[j];
            row[j] = std::min({ row[j] + 1, row[j - 1] + 1, diag + (a[i - 1] != b[j - 1]) });
            diag = up;
        }
    }
    return row[b.size()];
}

/* " (did you mean X?)" when a candidate is within a third of the key's length, else "". */
std::string suggest(const std::string& key, const std::vector<std::string>& candidates) {
    std::string best;
    size_t      best_d = SIZE_MAX;
    for (const std::string& c : candidates) {
        const size_t d = edit_distance(key, c);
        if (d < best_d) { best_d = d; best = c; }
    }
    if (best.empty() || best_d > std::max<size_t>(2, key.size() / 3)) return "";
    return " (did you mean " + best + "?)";
}

/* ------------------------------------------------------------------ the walk */

class Reader {
public:
    Reader(const std::string& origin, ryml::Parser& parser, const ryml::Tree& tree)
        : origin_(origin), parser_(parser), tree_(tree) {
        size_t n = 0;
        const ConfigFlag* f = config_flags(&n);
        for (size_t i = 0; i < n; ++i) {
            const char* name = f[i].name;
            while (*name == '-') ++name;
            flags_[name] = &f[i];
        }
    }

    int run(FleetConfig* out);

private:
    using Node = ryml::ConstNodeRef;

    /* A key's node, the key itself, and the YAML path it was read from, e.g. "models.qwen.tp".
     * The key is kept on its own because a model id may hold dots. */
    struct Item { Node node; std::string path; std::string key; };

    const std::string& origin_;
    ryml::Parser&      parser_;
    const ryml::Tree&  tree_;
    std::map<std::string, const ConfigFlag*> flags_;

    std::vector<std::string> shared_args_;              /* server flags, in file order */
    std::vector<std::string> memory_args_;              /* memory flags, in file order */
    std::map<std::string, std::string> shared_given_;   /* flag (with dashes) -> YAML path */

    std::string where(Node n) const {
        const ryml::Location loc = n.location(parser_);
        if (loc.line == ryml::npos) return origin_;
        return origin_ + ":" + std::to_string(loc.line + 1);
    }

    int fail(Node n, const std::string& msg) const {
        RAD_ERR("%s: %s", where(n).c_str(), msg.c_str());
        return RAD_E_INVAL;
    }

    /* The map's children in order, refusing a key given twice; YAML parsers keep both silently. */
    int children(Node map, const std::string& path, std::vector<Item>* out) const {
        if (!map.is_map()) return fail(map, (path.empty() ? std::string("the file") : path) + " must be a mapping of keys");
        std::set<std::string> seen;
        for (Node c : map.children()) {
            const std::string k = str(c.key());
            if (!seen.insert(k).second)
                return fail(c, (path.empty() ? k : path + "." + k) + " is given twice");
            out->push_back({ c, path.empty() ? k : path + "." + k, k });
        }
        return RAD_OK;
    }

    bool scalar(Node n) const { return n.has_val() && !n.is_map() && !n.is_seq(); }
    bool null_val(Node n) const { return scalar(n) && !n.is_val_quoted() && (n.val_is_null() || n.val().empty()); }

    int scalar_text(const Item& it, std::string* out) const {
        if (!scalar(it.node)) return fail(it.node, it.path + " takes a single value");
        if (null_val(it.node)) return fail(it.node, it.path + " needs a value");
        *out = str(it.node.val());
        return RAD_OK;
    }

    int boolean(const Item& it, bool* out) const {
        std::string v;
        if (int s = scalar_text(it, &v); s < 0) return s;
        if (v == "true" || v == "True" || v == "TRUE")   { *out = true;  return RAD_OK; }
        if (v == "false" || v == "False" || v == "FALSE") { *out = false; return RAD_OK; }
        return fail(it.node, it.path + " takes true or false, not '" + v + "'");
    }

    int integer(const Item& it, int64_t lo, int64_t hi, int64_t* out) const {
        std::string v;
        if (int s = scalar_text(it, &v); s < 0) return s;
        char* end = nullptr;
        errno = 0;
        const long long n = std::strtoll(v.c_str(), &end, 10);
        if (v.empty() || *end || errno == ERANGE)
            return fail(it.node, it.path + " takes a whole number, not '" + v + "'");
        if (n < lo || n > hi)
            return fail(it.node, it.path + " must be in [" + std::to_string(lo) + ", " + std::to_string(hi) +
                                 "], not " + v);
        *out = n;
        return RAD_OK;
    }

    int number(const Item& it, double* out) const {
        std::string v;
        if (int s = scalar_text(it, &v); s < 0) return s;
        char* end = nullptr;
        errno = 0;
        const double d = std::strtod(v.c_str(), &end);
        if (v.empty() || *end || errno == ERANGE) return fail(it.node, it.path + " takes a number, not '" + v + "'");
        *out = d;
        return RAD_OK;
    }

    int duration(const Item& it, int64_t* out_ms) const {
        std::string v;
        if (int s = scalar_text(it, &v); s < 0) return s;
        if (!parse_duration_ms(v, out_ms))
            return fail(it.node, it.path + " takes a duration such as 250ms, 30s, 30m or 2h, not '" + v + "'");
        return RAD_OK;
    }

    /* A YAML value as JSON: mappings and sequences recursively, plain scalars typed the way YAML's
     * core schema types them (true/false/null/numbers), everything quoted as a string. */
    rad::Json to_json(Node n) const {
        if (n.is_map()) {
            rad::Json j = rad::Json::object();
            for (Node c : n.children()) j[str(c.key())] = to_json(c);
            return j;
        }
        if (n.is_seq()) {
            rad::Json j = rad::Json::array();
            for (Node c : n.children()) j.push_back(to_json(c));
            return j;
        }
        const std::string v = str(n.val());
        if (n.is_val_quoted()) return v;
        if (n.val_is_null() || v.empty() || v == "~" || v == "null") return nullptr;
        if (v == "true") return true;
        if (v == "false") return false;
        char* end = nullptr;
        errno = 0;
        const long long i = std::strtoll(v.c_str(), &end, 10);
        if (!*end && errno != ERANGE) return i;
        const double d = std::strtod(v.c_str(), &end);
        if (!*end && errno != ERANGE) return d;
        return v;
    }

    /* `c`, an entry of a sequence value or a member of a mapping one, as flag text: the entry
     * itself, or KEY=VALUE. `what` names the children in the refusal. */
    int child_text(const Item& it, Node c, const char* what, std::string* out) const {
        if (!scalar(c) || null_val(c)) return fail(c, it.path + " " + what + " must be single values");
        *out = it.node.is_map() ? str(c.key()) + "=" + str(c.val()) : str(c.val());
        return RAD_OK;
    }

    /* One option key as flag arguments, appended to *out. The placeholder in the flag table says
     * how a structured value is spelled: LIST joins a sequence with commas, MAP joins a mapping as
     * KEY=VALUE pairs, JSON takes a mapping or sequence as JSON text, and repeatable flags repeat. */
    int flag_args(const Item& it, std::vector<std::string>* out) const {
        const ConfigFlag* f = flags_.at(it.key);
        const std::string flag = f->name;
        const std::string arg = f->arg ? f->arg : "";
        const bool structured = it.node.is_seq() || it.node.is_map();
        if (!f->arg) {
            bool on = false;
            if (int s = boolean(it, &on); s < 0) return s;
            if (on) out->push_back(flag);
            return RAD_OK;
        }
        if (in(kRepeatable, it.key) && structured) {
            for (Node c : it.node.children()) {
                std::string v;
                if (int s = child_text(it, c, "entries", &v); s < 0) return s;
                out->push_back(flag);
                out->push_back(v);
            }
            return RAD_OK;
        }
        std::string v;
        if (arg == "JSON" && structured) {
            v = to_json(it.node).dump();
        } else if ((arg == "LIST" && it.node.is_seq()) || (arg == "MAP" && it.node.is_map())) {
            for (Node c : it.node.children()) {
                std::string one;
                if (int s = child_text(it, c, arg == "MAP" ? "values" : "entries", &one); s < 0) return s;
                v += (v.empty() ? "" : ",") + one;
            }
        } else if (int s = scalar_text(it, &v); s < 0) {
            return s;
        }
        out->push_back(flag);
        out->push_back(v);
        return RAD_OK;
    }

    /* Where a key that is not valid here belongs, or why it does not exist. */
    std::string misplaced(const std::string& key, const std::string& level) const {
        for (const Replaced& r : kReplaced)
            if (key == r.flag) return key + " is not a key: " + r.instead;
        if (level != "server" && (in(kServerKeys, key) || key == "log")) return key + " belongs under server:";
        if (level != "memory" && level != "model" && in(kMemoryFlagKeys, key))
            return key + " belongs under memory: or a model";
        if (level != "memory" && (key == "host-mib" || key == "disk-tier"))
            return key + " belongs under memory:";
        if (level != "model" && flags_.count(key))
            return key + " is a per-model key; put it under the model or defaults:";
        return "";
    }

    /* A key `level` does not take: where it belongs, else "PATH is not a NOUN key" with the
     * closest of the keys the level does take. */
    int refuse_key(const Item& it, const std::string& level, const std::string& noun,
                   const std::vector<std::string>& valid) const {
        std::string m = misplaced(it.key, level);
        if (m.empty()) m = it.path + " is not a " + noun + " key" + suggest(it.key, valid);
        return fail(it.node, m);
    }

    std::vector<std::string> model_key_names() const {
        std::vector<std::string> v;
        for (const auto& [k, f] : flags_)
            if (misplaced(k, "model").empty()) v.push_back(k);
        for (const char* k : kModelFleetKeys) v.push_back(k);
        return v;
    }

    int read_server(const Item& sec, FleetConfig* out);
    int read_memory(const Item& sec, FleetConfig* out);
    int read_scheduler(const Item& sec, FleetConfig* out);
    int read_cards(const Item& sec, FleetConfig* out);
    int read_groups(const Item& sec, FleetConfig* out);
    int read_model_keys(const Item& sec, bool is_defaults, std::vector<Item>* out);
    int read_fleet_key(const Item& it, ModelEntry* e);
    int build_model(const std::string& name, Node at, const std::vector<Item>& keys, ModelEntry* e);
};

int Reader::read_server(const Item& sec, FleetConfig* out) {
    std::vector<Item> items;
    if (int s = children(sec.node, sec.path, &items); s < 0) return s;
    for (const Item& it : items) {
        const std::string& k = it.key;
        if (k == "log") {
            std::string v;
            if (int s = scalar_text(it, &v); s < 0) return s;
            if      (v == "info")  out->log = Log::Info;
            else if (v == "debug") out->log = Log::Debug;
            else if (v == "trace") out->log = Log::Trace;
            else return fail(it.node, "server.log takes info, debug or trace, not '" + v + "'");
            out->log_set = true;
            continue;
        }
        if (!in(kServerKeys, k)) {
            std::vector<std::string> valid = as_vector(kServerKeys);
            valid.push_back("log");
            return refuse_key(it, "server", "server", valid);
        }
        if (int s = flag_args(it, &shared_args_); s < 0) return s;
        shared_given_[flags_.at(k)->name] = it.path;
    }
    return RAD_OK;
}

int Reader::read_memory(const Item& sec, FleetConfig* out) {
    std::vector<Item> items;
    if (int s = children(sec.node, sec.path, &items); s < 0) return s;
    for (const Item& it : items) {
        const std::string& k = it.key;
        if (k == "host-mib") {
            if (int s = integer(it, 1, INT64_MAX / (1 << 20), &out->host_mib); s < 0) return s;
            continue;
        }
        if (k == "disk-tier") {
            if (int s = scalar_text(it, &out->disk_tier); s < 0) return s;
            continue;
        }
        if (!in(kMemoryFlagKeys, k)) {
            std::vector<std::string> valid = as_vector(kMemoryFlagKeys);
            valid.push_back("host-mib");
            valid.push_back("disk-tier");
            return refuse_key(it, "memory", "memory", valid);
        }
        if (int s = flag_args(it, &memory_args_); s < 0) return s;
        shared_given_[flags_.at(k)->name] = it.path;
    }
    return RAD_OK;
}

int Reader::read_scheduler(const Item& sec, FleetConfig* out) {
    std::vector<Item> items;
    if (int s = children(sec.node, sec.path, &items); s < 0) return s;
    for (const Item& it : items) {
        const std::string& k = it.key;
        int s = RAD_OK;
        if      (k == "load-timeout")  s = duration(it, &out->load_timeout_ms);
        else if (k == "drain-timeout") s = duration(it, &out->drain_timeout_ms);
        else if (k == "keepalive")     s = duration(it, &out->keepalive_ms);
        else if (k == "stuck-exit")    s = duration(it, &out->stuck_exit_ms);
        else if (k == "warm-up")       s = boolean(it, &out->warm_up);
        else {
            return fail(it.node, "scheduler." + k + " is not a scheduler key" +
                                 suggest(k, { "load-timeout", "drain-timeout", "keepalive",
                                              "stuck-exit", "warm-up" }));
        }
        if (s < 0) return s;
    }
    if (out->keepalive_ms <= 0) return fail(sec.node, "scheduler.keepalive must be above zero");
    return RAD_OK;
}

int Reader::read_cards(const Item& sec, FleetConfig* out) {
    if (scalar(sec.node) && str(sec.node.val()) == "all") return RAD_OK;
    if (!sec.node.is_seq() || sec.node.num_children() == 0)
        return fail(sec.node, "cards takes all or a list of card indices");
    std::set<int64_t> seen;
    for (Node c : sec.node.children()) {
        int64_t idx = 0;
        if (int s = integer({ c, sec.path, "cards" }, 0, 63, &idx); s < 0) return s;
        if (!seen.insert(idx).second) return fail(c, "card " + std::to_string(idx) + " is listed twice");
        out->cards.push_back((int)idx);
    }
    return RAD_OK;
}

int Reader::read_groups(const Item& sec, FleetConfig* out) {
    std::vector<Item> items;
    if (int s = children(sec.node, sec.path, &items); s < 0) return s;
    for (const Item& it : items) {
        std::vector<Item> kinds;
        if (int s = children(it.node, it.path, &kinds); s < 0) return s;
        if (kinds.size() != 1)
            return fail(it.node, it.path + " takes exactly one of together: or exclusive:");
        const Item& k = kinds[0];
        Group g;
        g.name = it.key;
        const std::string& kind = k.key;
        if      (kind == "together")  g.kind = Group::Kind::Together;
        else if (kind == "exclusive") g.kind = Group::Kind::Exclusive;
        else return fail(k.node, k.path + ": a group is together: or exclusive:" + suggest(kind, { "together", "exclusive" }));
        if (!k.node.is_seq() || k.node.num_children() < 2)
            return fail(k.node, k.path + " takes a list of at least two models");
        for (Node m : k.node.children()) {
            if (!scalar(m) || null_val(m)) return fail(m, k.path + " entries are model names");
            const std::string name = str(m.val());
            if (std::find(g.members.begin(), g.members.end(), name) != g.members.end())
                return fail(m, k.path + " lists " + name + " twice");
            g.members.push_back(name);
        }
        out->groups.push_back(std::move(g));
    }
    return RAD_OK;
}

/* A model's (or the defaults') keys, checked against the levels; option values are read later. */
int Reader::read_model_keys(const Item& sec, bool is_defaults, std::vector<Item>* out) {
    if (int s = children(sec.node, sec.path, out); s < 0) return s;
    for (const Item& it : *out) {
        const std::string& k = it.key;
        if (in(kModelFleetKeys, k)) {
            if (is_defaults && k == "aliases") return fail(it.node, "aliases name one model; they cannot be a default");
            continue;
        }
        std::string m = misplaced(k, "model");
        if (!m.empty()) return fail(it.node, m);
        if (!flags_.count(k)) return fail(it.node, it.path + " is not a key" + suggest(k, model_key_names()));
    }
    return RAD_OK;
}

int Reader::read_fleet_key(const Item& it, ModelEntry* e) {
    const std::string& k = it.key;
    if (k == "aliases") {
        std::vector<Node> names;
        if (scalar(it.node)) names.push_back(it.node);
        else if (it.node.is_seq()) for (Node c : it.node.children()) names.push_back(c);
        else return fail(it.node, it.path + " takes a name or a list of names");
        for (Node n : names) {
            if (!scalar(n) || null_val(n)) return fail(n, it.path + " entries are names");
            e->aliases.push_back(str(n.val()));
        }
        return RAD_OK;
    }
    if (k == "priority") {
        int64_t p = 0;
        if (int s = integer(it, -1000000, 1000000, &p); s < 0) return s;
        e->priority = (int)p;
        return RAD_OK;
    }
    if (k == "pin") return boolean(it, &e->pin);
    if (k == "ttl") return duration(it, &e->ttl_ms);
    if (k == "flex") {
        std::vector<Item> parts;
        if (int s = children(it.node, it.path, &parts); s < 0) return s;
        for (const Item& p : parts) {
            const std::string& pk = p.key;
            std::vector<Item> fields;
            if (pk != "kv" && pk != "experts")
                return fail(p.node, p.path + ": flex has kv: and experts:" + suggest(pk, { "kv", "experts" }));
            if (int s = children(p.node, p.path, &fields); s < 0) return s;
            for (const Item& f : fields) {
                const std::string& fk = f.key;
                int s = RAD_OK;
                if (pk == "kv" && fk == "min-tokens")          s = integer(f, 1, INT64_MAX, &e->flex.kv_min_tokens);
                else if (pk == "kv" && fk == "max-tokens")     s = integer(f, 1, INT64_MAX, &e->flex.kv_max_tokens);
                else if (pk == "kv" && fk == "weight") {
                    if ((s = number(f, &e->flex.kv_weight)) == RAD_OK && !(e->flex.kv_weight > 0))
                        return fail(f.node, f.path + " must be above zero");
                }
                else if (pk == "experts" && fk == "min-mib")   s = integer(f, 0, INT64_MAX, &e->flex.experts_min_mib);
                else if (pk == "experts" && fk == "max-mib")   s = integer(f, 0, INT64_MAX, &e->flex.experts_max_mib);
                else {
                    const std::vector<std::string> c = pk == "kv"
                        ? std::vector<std::string>{ "min-tokens", "max-tokens", "weight" }
                        : std::vector<std::string>{ "min-mib", "max-mib" };
                    return fail(f.node, f.path + " is not a key" + suggest(fk, c));
                }
                if (s < 0) return s;
            }
        }
        if (e->flex.kv_min_tokens > 0 && e->flex.kv_max_tokens > 0 && e->flex.kv_min_tokens > e->flex.kv_max_tokens)
            return fail(it.node, it.path + ".kv: min-tokens is above max-tokens");
        if (e->flex.experts_min_mib >= 0 && e->flex.experts_max_mib > 0 && e->flex.experts_min_mib > e->flex.experts_max_mib)
            return fail(it.node, it.path + ".experts: min-mib is above max-mib");
        return RAD_OK;
    }
    if (k == "park") {
        std::vector<Item> fields;
        if (int s = children(it.node, it.path, &fields); s < 0) return s;
        for (const Item& f : fields) {
            const std::string& fk = f.key;
            std::string v;
            if (fk != "vram" && fk != "ram" && fk != "restore")
                return fail(f.node, f.path + ": park has vram:, ram: and restore:" +
                                        suggest(fk, { "vram", "ram", "restore" }));
            if (int s = scalar_text(f, &v); s < 0) return s;
            if (fk == "vram") {
                if      (v == "none")    e->park.vram = ParkSpec::Vram::None;
                else if (v == "statics") e->park.vram = ParkSpec::Vram::Statics;
                else {
                    int64_t mib = 0;
                    if (integer(f, 1, INT64_MAX, &mib) < 0)
                        return fail(f.node, f.path + " takes none, statics or a size in MiB");
                    e->park.vram = ParkSpec::Vram::Mib;
                    e->park.vram_mib = mib;
                }
            } else if (fk == "restore") {
                if      (v == "auto") e->park.restore = ParkSpec::Restore::Auto;
                else if (v == "full") e->park.restore = ParkSpec::Restore::Full;
                else if (v == "live") e->park.restore = ParkSpec::Restore::Live;
                else return fail(f.node, f.path + " takes auto, full or live, not '" + v + "'");
            } else {
                if      (v == "required") e->park.ram = ParkSpec::Ram::Required;
                else if (v == "allowed")  e->park.ram = ParkSpec::Ram::Allowed;
                else if (v == "never")    e->park.ram = ParkSpec::Ram::Never;
                else return fail(f.node, f.path + " takes required, allowed or never, not '" + v + "'");
            }
        }
        return RAD_OK;
    }
    return RAD_E_INVAL;
}

/* `keys` is the defaults with the model's own keys over them. */
int Reader::build_model(const std::string& name, Node at, const std::vector<Item>& keys, ModelEntry* e) {
    e->name = name;
    std::vector<const Item*> options;
    for (const Item& it : keys) {
        if (!in(kModelFleetKeys, it.key)) options.push_back(&it);
        else if (int s = read_fleet_key(it, e); s < 0) return s;
    }
    std::sort(options.begin(), options.end(),
              [](const Item* a, const Item* b) { return a->key < b->key; });

    /* A model stating either half of the card pair replaces the memory level's pair. */
    bool own_card = false;
    for (const Item* it : options) own_card |= it->key == "gpu-memory-mib" || it->key == "gpu-headroom-mib";
    std::vector<std::string> args = shared_args_;
    std::map<std::string, std::string> given = shared_given_;
    for (size_t i = 0; i < memory_args_.size();) {
        const std::string& f = memory_args_[i];
        const bool takes = flags_.at(f.substr(2))->arg != nullptr;
        const size_t n = takes ? 2 : 1;
        if (own_card && (f == "--gpu-memory-mib" || f == "--gpu-headroom-mib")) given.erase(f);
        else args.insert(args.end(), memory_args_.begin() + (long)i, memory_args_.begin() + (long)(i + n));
        i += n;
    }
    args.push_back("--served-model-name");
    args.push_back(name);
    given["--served-model-name"] = "models." + name;
    for (const Item* it : options) {
        if (int s = flag_args(*it, &args); s < 0) return s;
        given[flags_.at(it->key)->name] = it->path;
    }

    std::vector<std::string> argv_store{ "radiance" };
    argv_store.insert(argv_store.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (std::string& a : argv_store) argv.push_back(a.data());
    argv.push_back(nullptr);
    Config cfg;
    if (config_parse((int)argv_store.size(), argv.data(), &cfg) != RAD_OK) {
        return fail(at, "models." + name + " refused by the line above (YAML keys are the options it names, "
                        "without the leading --)");
    }
    /* config_parse records every option it read as given on the command line; they came from the
     * file, so each takes its YAML path. Options it filled from the environment keep that source. */
    for (auto& [flag, src] : cfg.given) {
        auto g = given.find(flag);
        if (g != given.end() && src == "command line") src = g->second;
    }
    e->args = std::move(args);
    e->cfg = std::move(cfg);
    return RAD_OK;
}

bool valid_name(const std::string& n) {
    if (n.empty()) return false;
    for (unsigned char ch : n)
        if (ch == ':' || std::isspace(ch) || std::iscntrl(ch)) return false;
    return true;
}

int Reader::run(FleetConfig* out) {
    std::vector<Item> top;
    const Node root = tree_.crootref();
    if (root.is_stream() && root.num_children() != 1)
        return fail(root, "the file must hold exactly one YAML document");
    const Node doc = root.is_stream() ? root.first_child() : root;
    if (int s = children(doc, "", &top); s < 0) return s;

    const Item* defaults = nullptr;
    const Item* models = nullptr;
    for (const Item& it : top) {
        const std::string& k = it.key;
        int s = RAD_OK;
        if      (k == "server")    s = read_server(it, out);
        else if (k == "memory")    s = read_memory(it, out);
        else if (k == "scheduler") s = read_scheduler(it, out);
        else if (k == "cards")     s = read_cards(it, out);
        else if (k == "groups")    s = read_groups(it, out);
        else if (k == "defaults")  defaults = &it;
        else if (k == "models")    models = &it;
        else if (k.rfind("x-", 0) == 0) continue;
        else return refuse_key(it, "top", "top-level", as_vector(kTopKeys));
        if (s < 0) return s;
    }
    if (!models) return fail(doc, "no models: section; the file must name at least one model");

    std::vector<Item> base;
    if (defaults)
        if (int s = read_model_keys(*defaults, true, &base); s < 0) return s;

    std::vector<Item> entries;
    if (int s = children(models->node, models->path, &entries); s < 0) return s;
    if (entries.empty()) return fail(models->node, "models: names no model");

    std::map<std::string, std::string> taken;   /* name or alias -> the model it names */
    for (const Item& ent : entries) {
        const std::string& name = ent.key;
        if (!valid_name(name))
            return fail(ent.node, "'" + name + "' cannot be a model id: ids are non-empty and hold no ':' "
                                  "(reserved for base:adapter) and no whitespace");
        std::vector<Item> own;
        if (int s = read_model_keys(ent, false, &own); s < 0) return s;
        std::vector<Item> merged;
        for (const Item& d : base)
            if (std::none_of(own.begin(), own.end(), [&](const Item& o) { return o.key == d.key; }))
                merged.push_back(d);
        merged.insert(merged.end(), own.begin(), own.end());

        ModelEntry e;
        if (int s = build_model(name, ent.node, merged, &e); s < 0) return s;
        std::vector<std::string> ids{ name };
        ids.insert(ids.end(), e.aliases.begin(), e.aliases.end());
        for (const std::string& n : ids) {
            if (!valid_name(n))
                return fail(ent.node, "models." + name + ": alias '" + n + "' holds ':' or whitespace");
            auto [it, fresh] = taken.emplace(n, name);
            if (!fresh)
                return fail(ent.node, "models." + name + ": '" + n + "' already names model " + it->second);
        }
        out->models.push_back(std::move(e));
    }

    std::map<std::pair<std::string, std::string>, Group::Kind> pairs;
    for (const Group& g : out->groups) {
        for (const std::string& m : g.members)
            if (const ModelEntry* me = out->find(m); !me || me->name != m)
                return fail(doc, "groups." + g.name + " names '" + m + "', which is not a model id");
        for (size_t i = 0; i < g.members.size(); ++i)
            for (size_t j = i + 1; j < g.members.size(); ++j) {
                auto key = std::minmax(g.members[i], g.members[j]);
                auto [it, fresh] = pairs.emplace(key, g.kind);
                if (!fresh && it->second != g.kind)
                    return fail(doc, "groups: " + key.first + " and " + key.second +
                                     " are in both a together and an exclusive group");
            }
    }
    return RAD_OK;
}

}  // namespace

const ModelEntry* FleetConfig::find(const std::string& n) const {
    for (const ModelEntry& m : models) {
        if (m.name == n) return &m;
        for (const std::string& a : m.aliases)
            if (a == n) return &m;
    }
    return nullptr;
}

std::vector<std::string> level_keys(const std::string& level) {
    if (level == "server") return as_vector(kServerKeys);
    if (level == "memory") return as_vector(kMemoryFlagKeys);
    return {};
}

bool parse_duration_ms(const std::string& s, int64_t* out_ms) {
    size_t i = 0;
    while (i < s.size() && std::isdigit((unsigned char)s[i])) ++i;
    if (i == 0 || i > 12) return false;
    const int64_t n = std::stoll(s.substr(0, i));
    const std::string unit = s.substr(i);
    int64_t mul = 0;
    if      (unit == "ms")                 mul = 1;
    else if (unit == "s" || unit.empty())  mul = 1000;
    else if (unit == "m")                  mul = 60 * 1000;
    else if (unit == "h")                  mul = 3600 * 1000;
    else return false;
    *out_ms = n * mul;
    return true;
}

int parse(const std::string& text, const std::string& origin, FleetConfig* out) {
    ryml::Callbacks cb;
    cb.set_error_basic(&on_basic).set_error_parse(&on_parse).set_error_visit(&on_visit);
    FleetConfig fc;
    fc.path = origin;
    try {
        ryml::EventHandlerTree handler(cb);
        ryml::Parser parser(&handler, ryml::ParserOptions().locations(true));
        ryml::Tree tree(cb);
        ryml::parse_in_arena(&parser, ryml::to_csubstr(origin), ryml::to_csubstr(text), &tree);
        /* Anchors, aliases and `<<` merge keys, so a file can share a block between models. */
        tree.resolve();
        if (tree.empty() || !tree.crootref().has_children()) {
            RAD_ERR("%s: the file is empty; it must name at least one model under models:", origin.c_str());
            return RAD_E_INVAL;
        }
        Reader r(origin, parser, tree);
        if (int s = r.run(&fc); s < 0) return s;
    } catch (const YamlError& e) {
        if (e.line != ryml::npos) RAD_ERR("%s:%zu: %s", origin.c_str(), e.line + 1, e.what());
        else                      RAD_ERR("%s: %s", origin.c_str(), e.what());
        return RAD_E_INVAL;
    }
    *out = std::move(fc);
    return RAD_OK;
}

int load(const std::string& path, FleetConfig* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        RAD_ERR("%s: cannot read the configuration: %s", path.c_str(), std::strerror(errno));
        return RAD_E_INVAL;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse(ss.str(), path, out);
}

/* ------------------------------------------------------------------ the key reference */

namespace {

void print_keys(FILE* to, const std::vector<const ConfigFlag*>& flags, const char* indent) {
    for (const ConfigFlag* f : flags) {
        const char* name = f->name;
        while (*name == '-') ++name;
        char left[64];
        snprintf(left, sizeof left, "%s%s: %s", indent, name, f->arg ? f->arg : "true|false");
        std::string help;
        for (const char* p = f->help; *p; ++p) {
            help += *p;
            if (*p != '\n') continue;
            help.append(39, ' ');
            while (p[1] == ' ') ++p;
        }
        fprintf(to, "%-38s %s\n", left, help.c_str());
    }
}

}  // namespace

void print_reference(FILE* to) {
    size_t n = 0;
    const ConfigFlag* f = config_flags(&n);
    std::vector<const ConfigFlag*> server, memory, model;
    for (size_t i = 0; i < n; ++i) {
        const char* name = f[i].name;
        while (*name == '-') ++name;
        bool replaced = false;
        for (const Replaced& r : kReplaced) replaced |= !std::strcmp(name, r.flag);
        if (replaced) continue;
        if (in(kServerKeys, name))          server.push_back(&f[i]);
        else if (in(kMemoryFlagKeys, name)) memory.push_back(&f[i]);
        else                                model.push_back(&f[i]);
    }
    fprintf(to,
        "usage: radiance --config FILE [--check]\n"
        "\n"
        "  --config FILE   the YAML file naming every model this process serves (docs/MULTIMODEL.md)\n"
        "  --check         read and validate the file, print what it configures, serve nothing\n"
        "\n"
        "The file's levels, and the keys each takes:\n\n"
        "server:\n"
        "  log: info|debug|trace                the log level\n");
    print_keys(to, server, "  ");
    fprintf(to,
        "\ncards: all | [N, ...]                  the cards models may use\n"
        "\nmemory:\n"
        "  host-mib: N                          pinned host RAM the process may hold\n"
        "  disk-tier: DIR                       the prefix-cache disk tier and park files\n");
    print_keys(to, memory, "  ");
    fprintf(to,
        "\nscheduler:\n"
        "  load-timeout: DURATION               how long a request waits for its model (default 120s)\n"
        "  drain-timeout: DURATION              how long a model being evicted may finish requests (30s)\n"
        "  keepalive: DURATION                  SSE comment interval while a request waits (10s)\n"
        "  stuck-exit: DURATION                 a device stuck this long ends the process (60s; 0 never)\n"
        "  warm-up: true|false                  fill free VRAM, then host RAM, with models at start (true)\n"
        "\ndefaults:                              any model key, merged under every model\n"
        "\nmodels:\n"
        "  NAME:                                the model id requests name\n"
        "    aliases: [NAME, ...]               more ids for this model\n"
        "    priority: N                        higher keeps VRAM and takes flex from lower\n"
        "    pin: true|false                    never leaves VRAM\n"
        "    ttl: DURATION                      park after this long without requests\n"
        "    flex: {kv: {min-tokens, max-tokens, weight}, experts: {min-mib, max-mib}}\n"
        "    park: {vram: none|statics|MIB, ram: required|allowed|never, restore: auto|full|live}\n");
    print_keys(to, model, "    ");
    fprintf(to,
        "\ngroups:\n"
        "  NAME: {together: [A, B, ...]}        loaded and parked together\n"
        "  NAME: {exclusive: [A, B, ...]}       never resident at the same time\n"
        "\nDurations are 250ms, 30s, 30m or 2h. Top-level keys starting with x- are ignored.\n");
}

/* ------------------------------------------------------------------ the migration hint */

namespace {

/* Plain when YAML would read it back as the same string, double-quoted otherwise. */
std::string yaml_scalar(const std::string& v) {
    bool plain = !v.empty();
    for (unsigned char ch : v)
        if (!(std::isalnum(ch) || ch == '_' || ch == '.' || ch == '/' || ch == '-' || ch == '+' || ch == '='))
            plain = false;
    if (plain && (v[0] == '-' || v == "true" || v == "false" || v == "null")) plain = false;
    if (plain) return v;
    std::string q = "\"";
    /* A double-quoted scalar folds a raw line break into a space: control characters go escaped. */
    for (char ch : v) {
        if      (ch == '\n') q += "\\n";
        else if (ch == '\t') q += "\\t";
        else if (ch == '\r') q += "\\r";
        else {
            if (ch == '"' || ch == '\\') q += '\\';
            q += ch;
        }
    }
    return q + "\"";
}

}  // namespace

std::string yaml_from_args(int argc, char** argv) {
    size_t n = 0;
    const ConfigFlag* f = config_flags(&n);
    std::map<std::string, const ConfigFlag*> table;
    for (size_t i = 0; i < n; ++i) table[f[i].name] = &f[i];

    struct Kv { std::string key; std::vector<std::string> vals; bool sw = false; bool json = false; };
    std::vector<Kv> server, memory, model;
    std::string name, log;
    auto put = [](std::vector<Kv>& to, const std::string& k, const std::string* v) {
        for (Kv& e : to)
            if (e.key == k) { if (v) e.vals.push_back(*v); return; }
        to.push_back({ k, v ? std::vector<std::string>{ *v } : std::vector<std::string>{}, v == nullptr });
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-v")  { log = "debug"; continue; }
        if (a == "-vv") { log = "trace"; continue; }
        auto t = table.find(a);
        if (t == table.end()) {
            /* Not an option: written as a model key anyway, so loading the file refuses it by name
             * (with the closest key) instead of the setting vanishing. */
            if (a.size() > 2 && a.compare(0, 2, "--") == 0) {
                const bool has_val = i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0;
                const std::string v = has_val ? argv[++i] : std::string("true");
                put(model, a.substr(2), &v);
            }
            continue;
        }
        const std::string key = a.substr(2);
        std::string v;
        if (t->second->arg) {
            if (i + 1 >= argc) break;
            v = argv[++i];
        }
        if (key == "served-model-name") { name = v; continue; }
        std::vector<Kv>& to = in(kServerKeys, key) ? server : in(kMemoryFlagKeys, key) ? memory : model;
        put(to, key, t->second->arg ? &v : nullptr);
        /* JSON is YAML's flow style, so a JSON argument is written as it stands, unquoted. */
        if (t->second->arg && !std::strcmp(t->second->arg, "JSON")) {
            const rad::Json j = rad::Json::parse(v, nullptr, false);
            if (!j.is_discarded() && j.is_structured())
                for (Kv& e : to)
                    if (e.key == key) e.json = true;
        }
    }
    if (name.empty()) {
        for (const Kv& e : model)
            if (e.key == "model" && !e.vals.empty()) {
                std::string m = e.vals[0];
                while (!m.empty() && m.back() == '/') m.pop_back();
                const size_t slash = m.rfind('/');
                if (slash != std::string::npos) m = m.substr(slash + 1);
                /* Only a container's extension goes: a checkpoint directory or repo name such as
                 * Qwen3.8-27B-FP8 keeps its dots. */
                if (m.size() > 4 && m.compare(m.size() - 4, 4, ".rad") == 0) m.resize(m.size() - 4);
                name = m;
            }
    }
    if (!valid_name(name)) name = "default";

    std::string y;
    auto emit = [&](const std::vector<Kv>& kvs, const char* indent) {
        for (const Kv& e : kvs) {
            if (e.sw) { y += std::string(indent) + e.key + ": true\n"; continue; }
            if (e.json && e.vals.size() == 1) {
                y += std::string(indent) + e.key + ": " + rad::Json::parse(e.vals[0]).dump() + "\n";
                continue;
            }
            if (e.vals.size() == 1) { y += std::string(indent) + e.key + ": " + yaml_scalar(e.vals[0]) + "\n"; continue; }
            y += std::string(indent) + e.key + ":\n";
            for (const std::string& v : e.vals) y += std::string(indent) + "  - " + yaml_scalar(v) + "\n";
        }
    };
    if (!server.empty() || !log.empty()) {
        y += "server:\n";
        if (!log.empty()) y += "  log: " + log + "\n";
        emit(server, "  ");
    }
    if (!memory.empty()) {
        y += "memory:\n";
        emit(memory, "  ");
    }
    y += "models:\n  " + yaml_scalar(name) + ":\n";
    emit(model, "    ");
    return y;
}

}  // namespace rad::fleet
