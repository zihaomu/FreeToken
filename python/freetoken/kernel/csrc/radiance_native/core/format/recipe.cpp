/* recipe.cpp -- see recipe.h. */
#include "recipe.h"
#include "../plugin/registry.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>

namespace rad {

std::string RecipeRule::options_text() const {
    std::vector<std::pair<std::string, std::string>> o = options;
    std::sort(o.begin(), o.end());
    std::string s;
    for (const auto& [k, v] : o) {
        if (!s.empty()) s += ',';
        s += k + "=" + v;
    }
    return s;
}

bool recipe_glob(const char* p, const char* n) {
    /* Iterative, with one backtrack point: the last `*` and where it was tried from. */
    const char *star = nullptr, *retry = nullptr;
    while (*n) {
        if (*p == '*') { star = p++; retry = n; continue; }
        if (*p && (*p == '?' || *p == *n)) { ++p; ++n; continue; }
        if (!star) return false;
        p = star + 1;
        n = ++retry;
    }
    while (*p == '*') ++p;
    return *p == 0;
}

namespace {

/* Expand `$NAME` and `${NAME}` from the environment. An unset name is refused rather than
 * expanded to empty (e.g. an empty calibration directory). */
bool expand(const std::string& in, std::string* out, std::string* why) {
    out->clear();
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] != '$') { out->push_back(in[i]); continue; }
        size_t b = i + 1, e;
        const bool braced = b < in.size() && in[b] == '{';
        if (braced) {
            e = in.find('}', ++b);
            if (e == std::string::npos) { *why = "an unclosed ${ in '" + in + "'"; return false; }
        } else {
            e = b;
            while (e < in.size() && (std::isalnum((unsigned char)in[e]) || in[e] == '_')) ++e;
        }
        const std::string name = in.substr(b, e - b);
        if (name.empty()) { *why = "a bare $ in '" + in + "'"; return false; }
        const char* v = std::getenv(name.c_str());
        if (!v) { *why = "$" + name + " is not set"; return false; }
        out->append(v);
        i = braced ? e : e - 1;
    }
    return true;
}

bool add_option(RecipeRule* r, const std::string& kv, std::string* why) {
    const size_t eq = kv.find('=');
    if (eq == std::string::npos || eq == 0) {
        *why = "'" + kv + "' is not key=value";
        return false;
    }
    std::string v;
    if (!expand(kv.substr(eq + 1), &v, why)) return false;
    for (const auto& o : r->options)
        if (o.first == kv.substr(0, eq)) {
            *why = "option '" + o.first + "' given twice";
            return false;
        }
    r->options.emplace_back(kv.substr(0, eq), v);
    return true;
}

}  /* namespace */

int Recipe::add_text(const std::string& text, const std::string& origin, std::string* why) {
    size_t at = 0;
    int line = 0;
    while (at <= text.size()) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        std::string l = text.substr(at, nl - at);
        at = nl + 1;
        ++line;
        const size_t hash = l.find('#');
        if (hash != std::string::npos) l.resize(hash);
        std::vector<std::string> words;
        size_t i = 0;
        while (i < l.size()) {
            while (i < l.size() && std::isspace((unsigned char)l[i])) ++i;
            const size_t s = i;
            while (i < l.size() && !std::isspace((unsigned char)l[i])) ++i;
            if (i > s) words.push_back(l.substr(s, i - s));
        }
        if (words.empty()) continue;
        const std::string where = origin + ":" + std::to_string(line);
        if (words.size() < 2) {
            if (why) *why = where + ": a rule is PATTERN QUANTISER [key=value ...]";
            return RAD_E_FORMAT;
        }
        RecipeRule r;
        r.pattern = words[0];
        r.quantizer = words[1];
        r.origin = where;
        for (size_t k = 2; k < words.size(); ++k) {
            std::string w;
            if (!add_option(&r, words[k], &w)) {
                if (why) *why = where + ": " + w;
                return RAD_E_FORMAT;
            }
        }
        rules_.push_back(std::move(r));
    }
    return RAD_OK;
}

int Recipe::add_flag(const std::string& spec, std::string* why) {
    const size_t eq = spec.find('=');
    if (eq == std::string::npos || eq == 0) {
        if (why) *why = "--quant '" + spec + "': PATTERN=QUANTISER[:key=value,...]";
        return RAD_E_FORMAT;
    }
    RecipeRule r;
    r.pattern = spec.substr(0, eq);
    r.origin = "--quant";
    const std::string rest = spec.substr(eq + 1);
    const size_t colon = rest.find(':');
    r.quantizer = rest.substr(0, colon);
    if (r.quantizer.empty()) {
        if (why) *why = "--quant '" + spec + "' names no quantiser";
        return RAD_E_FORMAT;
    }
    if (colon != std::string::npos) {
        const std::string opts = rest.substr(colon + 1);
        size_t at = 0;
        while (at <= opts.size()) {
            size_t c = opts.find(',', at);
            if (c == std::string::npos) c = opts.size();
            std::string w;
            if (c > at && !add_option(&r, opts.substr(at, c - at), &w)) {
                if (why) *why = "--quant '" + spec + "': " + w;
                return RAD_E_FORMAT;
            }
            at = c + 1;
        }
    }
    rules_.push_back(std::move(r));
    return RAD_OK;
}

const RecipeRule* Recipe::match(const std::string& name) const {
    for (const RecipeRule& r : rules_)
        if (recipe_glob(r.pattern.c_str(), name.c_str())) return &r;
    return nullptr;
}

int CompiledRule::build(const RecipeRule& r, const Registry& reg, std::string* why) {
    rule = &r;
    const Registry::QuantRow* row = reg.quantizer(r.quantizer);
    if (!row || !row->info) {
        std::string have;
        for (const auto& qr : reg.quantizers())
            if (qr.info) have += std::string(have.empty() ? "" : ", ") + qr.info->name;
        *why = r.origin + ": no quantiser '" + r.quantizer + "' is loaded (" +
               (have.empty() ? std::string("none are -- is $RADIANCE_HOME/quantizers there?")
                             : "loaded: " + have) + ")";
        return RAD_E_NOTFOUND;
    }
    q = row->info;
    keys.reserve(r.options.size());
    vals.reserve(r.options.size());
    for (const auto& [k, v] : r.options) {
        const RadQuantOption* opt = nullptr;
        for (int i = 0; i < q->n_options; ++i)
            if (k == q->options[i].key) opt = &q->options[i];
        if (!opt) {
            std::string takes;
            for (int i = 0; i < q->n_options; ++i)
                takes += std::string(i ? ", " : "") + q->options[i].key;
            *why = r.origin + ": " + q->name + " takes no option '" + k + "' (it takes " + takes +
                   ")";
            return RAD_E_INVAL;
        }
        char* end = nullptr;
        errno = 0;
        if (opt->type == RAD_P_INT) {
            const long long x = std::strtoll(v.c_str(), &end, 10);
            if (v.empty() || !end || *end || errno == ERANGE) {
                *why = r.origin + ": " + k + "=" + v + " is not a whole number";
                return RAD_E_INVAL;
            }
            keys.push_back(k);
            vals.push_back(v);
            params.push_back(RAD_INT(nullptr, x));
        } else if (opt->type == RAD_P_F64) {
            const double x = std::strtod(v.c_str(), &end);
            if (v.empty() || !end || *end || errno == ERANGE || !std::isfinite(x)) {
                *why = r.origin + ": " + k + "=" + v + " is not a number";
                return RAD_E_INVAL;
            }
            keys.push_back(k);
            vals.push_back(v);
            params.push_back(RAD_F64(nullptr, x));
        } else {
            keys.push_back(k);
            vals.push_back(v);
            params.push_back(RAD_STR(nullptr, nullptr));
        }
    }
    for (size_t i = 0; i < params.size(); ++i) {
        params[i].key = keys[i].c_str();
        if (params[i].kind == RAD_P_STR) params[i].sval = vals[i].c_str();
    }
    return RAD_OK;
}

std::string Recipe::text() const {
    std::string s;
    for (const RecipeRule& r : rules_) {
        s += r.pattern + " " + r.quantizer;
        for (const auto& [k, v] : r.options) s += " " + k + "=" + v;
        s += "\n";
    }
    return s;
}

}  /* namespace rad */
