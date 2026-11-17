/* gen_config.cpp -- reading a checkpoint's own sampling; see gen_config.h. */
#include "gen_config.h"

#include "rad_internal.h"

#include "util/json.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace rad {
namespace {

/* A numeric field, taken only if in range; otherwise warned about and dropped. */
void take(const char* what, const char* src, double v, double lo, double hi, bool lo_open,
          float* dst) {
    const bool ok = (lo_open ? v > lo : v >= lo) && v <= hi;
    if (ok) { *dst = (float)v; return; }
    RAD_WARN("%s states %s %g, which is outside %s%g, %g]; ignoring it",
             src, what, v, lo_open ? "(" : "[", lo, hi);
}

/* An integer field (a token count), same contract. */
void take_i(const char* what, const char* src, int64_t v, int64_t lo, int64_t hi, int* dst) {
    if (v >= lo && v <= hi) { *dst = (int)v; return; }
    RAD_WARN("%s states %s %lld, which is outside [%lld, %lld]; ignoring it",
             src, what, (long long)v, (long long)lo, (long long)hi);
}

}  /* namespace */

SamplingParams GenSampling::onto(SamplingParams base) const {
    if (temp  >= 0.0f) base.temp  = temp;
    if (top_p >= 0.0f) base.top_p = top_p;
    if (min_p >= 0.0f) base.min_p = min_p;
    if (top_k >= 0)    base.top_k = top_k;
    if (ngram_ban_n >= 0)    base.ngram_ban_n = ngram_ban_n;
    if (ngram_window >= 0)   base.ngram_window = ngram_window;
    if (!ngram_whitelist.empty()) base.ngram_whitelist = ngram_whitelist;
    if (!suppress.empty())   base.suppress_tokens = suppress;
    return base;
}

/* The one-line statement of a served sampling configuration. The wording is pinned by
 * gen_config_test.cpp (gen_config_reads_the_ngram_ban) -- change it there in the same change.
 * It spells the values out for a log reader; `onto` fills the fields from defaults. The two
 * share values, not logic, and the duplication of the ban clause is a string, not a rule. */
std::string GenSampling::describe(const SamplingParams& sp) {
    char buf[320];
    std::snprintf(buf, sizeof buf, "temperature %.2f, top_k %d%s, top_p %.2f, min_p %.2f",
                  (double)sp.temp, sp.top_k,
                  sp.top_k == 0 ? " (the 1024 most likely tokens, the sampler's candidate width)" : "",
                  (double)sp.top_p, (double)sp.min_p);
    std::string s(buf);
    if (sp.ngram_ban_n > 0) {
        char ban[96];
        if (sp.ngram_window > 0)
            std::snprintf(ban, sizeof ban, ", ngram ban n %d over the last %d tokens",
                          sp.ngram_ban_n, sp.ngram_window);
        else
            std::snprintf(ban, sizeof ban, ", ngram ban n %d over the whole sequence",
                          sp.ngram_ban_n);
        s += ban;
        if (!sp.ngram_whitelist.empty())
            s += ", whitelist " + std::to_string(sp.ngram_whitelist.size()) + " ids";
    }
    return s;
}

std::vector<GenFieldSource> gen_sampling_sources(const GenSampling& stated, const std::string& src,
                                                 const SamplingParams& served, const Config& cfg) {
    std::vector<GenFieldSource> out;
    auto field = [&](const char* flag, const char* name, bool given, bool said, std::string value) {
        if (given) return;   /* the option's own row says "command line" */
        std::string why = said        ? "from " + src
                        : src.empty() ? std::string("the model states no sampling of its own, so "
                                                    "this is the built-in default")
                                      : src + " states no " + name +
                                            ", so this is the built-in default";
        out.push_back({ flag, std::move(value), std::move(why) });
    };
    field("--temp", "temperature", cfg.sample_temp >= 0.0f, stated.temp >= 0.0f,
          fmt("%g", (double)served.temp));
    field("--top-k", "top_k", cfg.sample_top_k >= 0, stated.top_k >= 0,
          std::to_string(served.top_k));
    field("--top-p", "top_p", cfg.sample_top_p >= 0.0f, stated.top_p >= 0.0f,
          fmt("%g", (double)served.top_p));
    field("--min-p", "min_p", cfg.sample_min_p >= 0.0f, stated.min_p >= 0.0f,
          fmt("%g", (double)served.min_p));
    return out;
}

bool gen_sampling_parse(const std::string& json_text, const char* src, GenSampling* out) {
    Json j;
    try { j = Json::parse(json_text); }
    catch (const std::exception& e) { RAD_WARN("%s: %s; ignoring it", src, e.what()); return false; }
    if (!j.is_object()) { RAD_WARN("%s: not a JSON object; ignoring it", src); return false; }

    /* `do_sample: false` means greedy, and transformers then ignores every other sampler field,
     * so take nothing else. */
    auto it = j.find("do_sample");
    if (it != j.end() && it->is_boolean() && !it->get<bool>()) { out->temp = 0.0f; return true; }

    it = j.find("temperature");
    if (it != j.end() && it->is_number()) take("temperature", src, it->get<double>(), 0.0, 2.0, false, &out->temp);
    it = j.find("top_p");
    if (it != j.end() && it->is_number()) take("top_p", src, it->get<double>(), 0.0, 1.0, true, &out->top_p);
    it = j.find("min_p");
    if (it != j.end() && it->is_number()) take("min_p", src, it->get<double>(), 0.0, 1.0, false, &out->min_p);
    it = j.find("top_k");
    if (it != j.end() && it->is_number_integer()) {
        const int64_t v = it->get<int64_t>();
        /* -1 is transformers' "no top_k", this sampler's 0; below -1 is invalid. */
        if (v >= -1) out->top_k = (int)(v < 0 ? 0 : v);
        else RAD_WARN("%s states top_k %lld, which is not a count; ignoring it", src, (long long)v);
    }

    /* The windowed no-repeat-n-gram ban. Its checkpoint name in transformers is
     * no_repeat_ngram_size / no_repeat_ngram_window_size (Unlimited-OCR ships 35 over the last
     * 128); this sampler's own names are taken too. The reference's negative window means the
     * whole sequence, which this sampler spells 0. */
    it = j.find("no_repeat_ngram_size");
    if (it == j.end() || !it->is_number_integer()) it = j.find("ngram_ban_n");
    if (it != j.end() && it->is_number_integer())
        take_i("no_repeat_ngram_size", src, it->get<int64_t>(), 0, 256, &out->ngram_ban_n);

    it = j.find("no_repeat_ngram_window_size");
    if (it == j.end() || !it->is_number_integer()) it = j.find("ngram_window");
    if (it != j.end() && it->is_number_integer()) {
        const int64_t v = it->get<int64_t>();
        take_i("no_repeat_ngram_window_size", src, v < 0 ? 0 : v, 0, 131072, &out->ngram_window);
    }

    /* The ban's whitelist, a JSON array of token ids (the reference's whitelist_token_ids).
     * Values are warned about and dropped, like every other field here: an id is a vocabulary
     * coordinate, so a negative one is not a value at all. */
    it = j.find("no_repeat_ngram_whitelist");
    if (it != j.end() && !it->is_array()) {
        /* The legacy key is only a problem when nothing rescues it: a config that states a bad
         * reference name beside a good sampler-spelled one reads the good one, silently. */
        const auto own = j.find("ngram_whitelist");
        if (own == j.end() || !own->is_array())
            RAD_WARN("%s states a no_repeat_ngram_whitelist that is not an array; ignoring it",
                     src);
        it = own;
    }
    if (it != j.end() && it->is_array()) {
        for (const auto& v : *it) {
            const int64_t id = v.is_number_integer() ? v.get<int64_t>() : -1;
            if (id < 0 || id > INT32_MAX) {
                RAD_WARN("%s states a no_repeat_ngram_whitelist entry outside [0, %d]; ignoring it",
                         src, INT32_MAX);
                continue;
            }
            out->ngram_whitelist.push_back((int)id);
        }
    }

    /* transformers' suppress_tokens, a JSON array of ids. Dropped entry by entry like the
     * whitelist above. */
    it = j.find("suppress_tokens");
    if (it != j.end() && !it->is_array()) {
        RAD_WARN("%s states suppress_tokens that is not an array; ignoring it", src);
    } else if (it != j.end()) {
        for (const auto& v : *it) {
            const int64_t id = v.is_number_integer() ? v.get<int64_t>() : -1;
            if (id < 0 || id > INT32_MAX) {
                RAD_WARN("%s states a suppress_tokens entry outside [0, %d]; ignoring it",
                         src, INT32_MAX);
                continue;
            }
            out->suppress.push_back((int)id);
        }
    }
    return out->any();
}

bool gen_sampling_from_file(const std::string& path, bool required, GenSampling* out) {
    std::ifstream f(path);
    if (!f) {
        if (required) RAD_ERR("%s: cannot be read", path.c_str());
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return gen_sampling_parse(text, path.c_str(), out);
}

bool gen_sampling_from_kv(const std::function<const char*(const char*)>& get, GenSampling* out) {
    /* Rebuilt as JSON and passed to gen_sampling_parse, so container metadata and the original
     * file share one set of range checks and one reading of `do_sample`. */
    Json j = Json::object();
    static const char* const kFields[] = { "do_sample", "temperature", "top_p", "min_p", "top_k",
                                           "no_repeat_ngram_size", "no_repeat_ngram_window_size",
                                           "no_repeat_ngram_whitelist", "suppress_tokens" };
    for (const char* k : kFields) {
        const char* v = get(k);
        if (!v || !*v) continue;
        if (std::strcmp(k, "do_sample") == 0) {
            /* rad-convert writes a JSON boolean as "1" or "0". */
            j[k] = (std::strcmp(v, "0") != 0);
            continue;
        }
        if (std::strcmp(k, "no_repeat_ngram_whitelist") == 0) {
            /* Not a number: the container's metadata is string-valued, and the whitelist is a
             * JSON array spelled as JSON text. */
            try {
                j[k] = Json::parse(v);
            } catch (const std::exception& e) {
                RAD_WARN("the container states generation.%s as \"%s\", which is not a JSON "
                         "array of ids (%s); ignoring it", k, v, e.what());
            }
            continue;
        }
        if (std::strcmp(k, "suppress_tokens") == 0) {
            /* An array of scalars is flattened to a space-separated string (flatten_json). */
            Json ids = Json::array();
            char* p = const_cast<char*>(v);
            for (char* e = nullptr;; p = e) {
                const long long id = std::strtoll(p, &e, 10);
                if (e == p) break;
                ids.push_back(id);
            }
            j[k] = std::move(ids);
            continue;
        }
        char* end = nullptr;
        const double d = std::strtod(v, &end);
        if (end == v) {
            RAD_WARN("the container states generation.%s as \"%s\", which is not a number; "
                     "ignoring it", k, v);
            continue;
        }
        if (std::strcmp(k, "top_k") == 0) j[k] = (int64_t)d;
        else if (std::strncmp(k, "no_repeat_ngram", 15) == 0) j[k] = (int64_t)d;
        else                              j[k] = d;
    }
    if (j.empty()) return false;
    return gen_sampling_parse(j.dump(), "the container's generation metadata", out);
}

std::string gen_sidecar_path(const std::string& model_path) {
    const size_t slash = model_path.find_last_of("/\\");
    const size_t dot   = model_path.find_last_of('.');
    const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash);
    return (has_ext ? model_path.substr(0, dot) : model_path) + ".generation.json";
}

}  /* namespace rad */
