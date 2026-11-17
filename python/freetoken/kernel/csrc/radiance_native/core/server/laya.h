/* laya.h -- POST /v1/systemone for Convai Innovations' Laya decision models: each question's
 * sequence, its words for the plugin, and the answer from the plugin's option logits.
 *
 * Laya reads one sequence a question, `[CLS] <type> question: <instructions> [SEP] [MASK] opt0
 * [MASK] opt1 ... [SEP] <state> [SEP]`, and scores each option at its [MASK] (arch/laya). This
 * file is a C++ rewrite of _check_question, _to_internal, render_options, build_head,
 * build_sequence and _decode_answers in laya's own package (laya/agent.py, laya/common.py), its
 * strings kept as they are: the model was trained on exactly this text.
 *
 * As the package does, a state longer than the room a question leaves is cut (to its start, or to
 * its end when the state is a list of turns), and `usage` says so with the package's own keys:
 * state_tokens, state_tokens_dropped, truncated and truncated_questions.
 *
 * Served only when the container declares laya (`decision.type` = `laya`, which rad-convert sets
 * for a Laya checkpoint directory).
 */
#pragma once

#include "oai.h"

#include <map>
#include <string>
#include <vector>

namespace rad {
namespace server {

/* The container's Laya settings (decision.laya.*, which rad-convert writes from
 * rl_agent_config.json and tokenizer/; the engine copies them here). */
struct LayaSettings {
    int64_t     max_len = 0, head_max_len = 0;
    int32_t     cls = -1, sep = -1, mask = -1;
    std::string mask_token;
    double      temperature[3] = { 1.0, 1.0, 1.0 };            /* choice, score, noul */
    std::map<std::string, double> temperature_by_options;      /* "choice:3-5" -> t */
    bool ok() const {
        return max_len > 0 && head_max_len > 0 && cls >= 0 && sep >= 0 && mask >= 0 &&
               !mask_token.empty();
    }
};

struct LayaQuestion {
    std::string name, type;                  /* type: choice | score | noul */
    int         type_id = 0;                 /* RAD_LAYA_CHOICE, _SCORE or _NOUL (abi/rad_laya.h) */
    /* One entry an option, in the order the model scores them: a choice's labels as the request
     * gives them (`keys` the response's keys, `labels` the values `choice` answers with), a score's
     * levels "0".."n-1", noul false then true. */
    std::vector<std::string> keys;
    std::vector<json>        labels;
    std::vector<std::string> legend;         /* score: each level as the model read it */
    std::vector<int32_t>     tokens;         /* the sequence */
    std::vector<int32_t>     markers;        /* each option's [MASK] position in it */
    int64_t                  state_used = 0; /* state tokens the sequence holds */
};

struct LayaRequest {
    std::vector<LayaQuestion> questions;
    int64_t state_tokens = 0;                /* the state, tokenized whole */
};

/* Parse, validate and build every question's sequence. 400 for what the package refuses, and for
 * options that do not fit the head's budget. */
int parse_laya_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                       const std::string& model_id, const LayaSettings& s, LayaRequest* out,
                       ApiError* err);

/* The question's words for the plugin (abi/rad_laya.h): its tokens, type and markers. */
std::vector<int32_t> laya_words(const LayaQuestion& q);

/* One question's answer from the plugin's pooled row (its first k values are the option logits):
 * the fitted temperature, clamped to [0.5, 5] as the package clamps it, a softmax, and the
 * package's rounding and confidence. A non-finite logit is an error. */
int laya_answer(const LayaQuestion& q, const std::vector<float>& pooled, const LayaSettings& s,
                json* out, ApiError* err);

/* The request's `usage`: input_tokens (every sequence) and the package's truncation report. */
json laya_usage(const LayaRequest& r);


}  /* namespace server */
}  /* namespace rad */
