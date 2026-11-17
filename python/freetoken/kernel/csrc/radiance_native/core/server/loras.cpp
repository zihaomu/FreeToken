/* loras.cpp -- see loras.h. */
#include "loras.h"
#include "rad_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace rad {
namespace server {

namespace {

/* "name" or "name:strength" (the last ':' splits, so a name may hold one). */
bool split_entry(const std::string& s, std::string* name, double* w) {
    *w = 1.0;
    const size_t c = s.rfind(':');
    if (c != std::string::npos && c + 1 < s.size()) {
        char* end = nullptr;
        const double v = std::strtod(s.c_str() + c + 1, &end);
        if (end && *end == '\0' && std::isfinite(v)) {
            *name = s.substr(0, c);
            *w = v;
            return !name->empty();
        }
    }
    *name = s;
    return !name->empty();
}

}  /* namespace */

int parse_loras(const Json& body, std::string* prompt,
                std::vector<std::pair<std::string, double>>* out, std::string* field,
                std::string* why) {
    out->clear();
    const char* key = body.contains("loras") ? "loras" : "lora";
    if (body.contains(key) && !body[key].is_null()) {
        *field = key;
        const Json& v = body[key];
        auto entry = [&](const Json& e) -> bool {
            std::string name;
            double w = 1.0;
            if (e.is_string()) {
                if (!split_entry(e.get<std::string>(), &name, &w)) return false;
            } else if (e.is_object()) {
                if (!e.contains("name") || !e["name"].is_string()) return false;
                name = e["name"].get<std::string>();
                for (const char* k : { "weight", "strength", "scale" })
                    if (e.contains(k) && !e[k].is_null()) {
                        if (!e[k].is_number()) return false;
                        w = e[k].get<double>();
                    }
            } else {
                return false;
            }
            if (name.empty() || !std::isfinite(w)) return false;
            out->push_back({ name, w });
            return true;
        };
        if (v.is_array()) {
            for (const Json& e : v)
                if (!entry(e)) {
                    *why = "each entry is a name, \"name:strength\" or {\"name\", \"weight\"}";
                    return RAD_E_INVAL;
                }
        } else if (v.is_object() && !v.contains("name")) {
            for (auto it = v.begin(); it != v.end(); ++it) {
                if (!it.value().is_number() || !std::isfinite(it.value().get<double>())) {
                    *why = "a {name: strength} object's strengths are numbers";
                    return RAD_E_INVAL;
                }
                out->push_back({ it.key(), it.value().get<double>() });
            }
            /* By name, not as written: the adapters apply in the order they are listed, and a
             * {name: strength} object has none of its own. */
            std::sort(out->begin(), out->end());
        } else if (!entry(v)) {
            *why = "expected a list of adapters (names, \"name:strength\" or {\"name\", \"weight\"})";
            return RAD_E_INVAL;
        }
    }
    /* <lora:name[:strength]> tags in the prompt. */
    if (prompt) {
        std::string p;
        size_t at = 0;
        for (;;) {
            const size_t open = prompt->find("<lora:", at);
            if (open == std::string::npos) { p += prompt->substr(at); break; }
            const size_t close = prompt->find('>', open);
            if (close == std::string::npos) { p += prompt->substr(at); break; }
            std::string name;
            double w = 1.0;
            if (!split_entry(prompt->substr(open + 6, close - open - 6), &name, &w)) {
                *field = "prompt";
                *why = "a <lora:...> tag names no adapter";
                return RAD_E_INVAL;
            }
            out->push_back({ name, w });
            p += prompt->substr(at, open - at);
            at = close + 1;
        }
        if (p.size() != prompt->size()) {
            /* Tidy the gaps the tags leave: runs of spaces, and commas with nothing between. */
            std::string t;
            for (char c : p) {
                if (c == ' ' && (t.empty() || t.back() == ' ')) continue;
                t += c;
            }
            for (size_t k; (k = t.find(", ,")) != std::string::npos;) t.erase(k, 2);
            for (size_t k; (k = t.find(",,")) != std::string::npos;) t.erase(k, 1);
            while (!t.empty() && (t[0] == ' ' || t[0] == ',')) t.erase(0, 1);
            while (!t.empty() && (t.back() == ' ' || t.back() == ',')) t.pop_back();
            *prompt = t.empty() ? std::string(" ") : t;
        }
    }
    return RAD_OK;
}

}  /* namespace server */
}  /* namespace rad */
