/* gen_config.h -- the sampling a checkpoint states for itself (generation_config.json), read in
 * its three forms: a file named on the command line, a sidecar beside the container, and the
 * `generation.*` metadata rad-convert flattened into the container.
 *
 * These values set what a request sending no sampler fields gets. Without them SamplingParams'
 * defaults sample the whole vocabulary at temperature 1, and a model tuned with a truncated tail
 * occasionally draws end-of-turn early: replies stop short, and agent turns arrive with text and
 * no tool call. A separate component so this can be tested.
 *
 * Nothing here refuses: values come with the model, so an out-of-range field is warned about and
 * dropped. Command-line overrides are checked in core/config.cpp.
 */
#pragma once
#include "rad_core.h"

#include <functional>
#include <string>
#include <vector>

namespace rad {

/* One config's sampler fields. -1 means "not stated", since 0 is a legal value of each (greedy
 * temperature, top_k off, min_p off, n-gram ban off). */
struct GenSampling {
    float temp = -1.0f, top_p = -1.0f, min_p = -1.0f;
    int   top_k = -1;
    /* The windowed no-repeat-n-gram ban (transformers' no_repeat_ngram_size /
     * no_repeat_ngram_window_size; Unlimited-OCR ships 35 over the last 128). `ngram_window` is
     * normalized to 0, this sampler's "the whole sequence", when the config states it negative. */
    int   ngram_ban_n = -1, ngram_window = -1;
    /* The ban's whitelist: token ids the ban must never ban (the reference processor's
     * `whitelist_token_ids`, removed from the banned set before the -inf writes). Transformers
     * has no field for it; a container states it as `generation.no_repeat_ngram_whitelist`, a
     * JSON array of ids. Empty means not stated. */
    std::vector<int32_t> ngram_whitelist;
    /* transformers' suppress_tokens: ids never sampled. Empty means not stated. */
    std::vector<int32_t> suppress;

    bool any() const {
        return temp >= 0.0f || top_p >= 0.0f || min_p >= 0.0f || top_k >= 0 ||
               ngram_ban_n >= 0 || ngram_window >= 0 || !ngram_whitelist.empty() ||
               !suppress.empty();
    }

    /* Overlay onto `base` field by field; unstated fields keep base's, so sources stack. */
    SamplingParams onto(SamplingParams base) const;

    /* "temperature 0.70, top_k 20 (off), top_p 0.95, min_p 0.00, ngram ban n 35 over the last 128
     * tokens", for the startup log line (the last part only when a ban is on). */
    static std::string describe(const SamplingParams& sp);
};

/* Where each of the four served fields came from, for the startup report (core/diag/diag.h): one
 * entry per field no flag set, with its option, the served value and why. `stated` is what the
 * source in force read and `src` its name ("" for none). Reasons are per field, since a source may
 * state only some fields and the rest take SamplingParams' defaults. */
struct GenFieldSource { std::string flag, value, why; };
std::vector<GenFieldSource> gen_sampling_sources(const GenSampling& stated, const std::string& src,
                                                 const SamplingParams& served, const Config& cfg);

/* Parse the sampler fields from generation_config.json text. `src` names the origin in warnings.
 * Returns whether anything was taken; unparseable text warns and takes nothing. */
bool gen_sampling_parse(const std::string& json_text, const char* src, GenSampling* out);

/* The same from a file. Absent is not an error unless `required`. */
bool gen_sampling_from_file(const std::string& path, bool required, GenSampling* out);

/* The same from the container's `generation.*` metadata (string values). `get` takes a bare field
 * name ("top_k") and returns null when absent. */
bool gen_sampling_from_kv(const std::function<const char*(const char*)>& get, GenSampling* out);

/* `<model>.rad` -> `<model>.generation.json`: a sidecar that supplies or corrects a container's
 * sampling without reconverting it. */
std::string gen_sidecar_path(const std::string& model_path);

}  /* namespace rad */
