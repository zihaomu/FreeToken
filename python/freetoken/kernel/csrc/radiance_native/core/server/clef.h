/* clef.h -- POST /v1/systemone for Cloudflare's clef and clef-flash: the prompt, its spans, and
 * the answer read off the joint schema head's per-option logits.
 *
 * clef is not read at one position the way d1 is (systemone.h). Its joint schema head reads the
 * final hidden state of EVERY prompt position, the mean over each question's instruction span and
 * each option's span, and the lm_head rows of each option's tokens, and gives one logit an
 * allowed option. So the prompt here comes with those spans, and the answer is a softmax over the
 * head's logits. This file is a C++ rewrite of encode_record, question_options and
 * systemone_answer in the release's joint_schema_model.py, with its strings kept as they are: the
 * head was trained on exactly this text.
 *
 * Served only when the container declares clef (`decision.type` = `clef`, set at convert with
 * `--set decision.type=clef`); the HF config does not declare it.
 */
#pragma once

#include "oai.h"

#include <string>
#include <utility>
#include <vector>

namespace rad {
namespace server {

using TokenSpan = std::pair<int64_t, int64_t>;   /* [start, end) prompt positions */

struct ClefQuestion {
    std::string name, type;          /* type: noul | choice | score */
    int         type_id = 0;         /* noul 0, choice 1, score 2: the head's type embedding */
    TokenSpan   question;            /* the rendered instructions */
    /* One entry an allowed option, in the order the head scores them: true/false, the choice
     * names sorted, or the levels "0".."n-1". */
    std::vector<std::string> options;
    std::vector<TokenSpan>   option_spans;
    /* The response's own order and wording: a choice lists its options as the request gave them,
     * a score carries its legend: each level's criterion as given, any JSON value, dumped. */
    std::vector<std::string> response_order;
    std::vector<std::string> legend;
};

struct ClefRequest {
    std::vector<int32_t>      tokens;
    std::vector<ClefQuestion> questions;
};

/* Parse and render. The whole prompt must fit `lim.max_ctx`: a state too long is refused, where
 * joint_schema_model.py would cut it short, so an answer never reads a state the caller did not
 * send whole. */
int parse_clef_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                       const std::string& model_id, ClefRequest* out, ApiError* err);

/* One question's answer from the head's logits, one an option in `options` order
 * (systemone_answer: a softmax, rounded to four places as the release rounds). A non-finite logit
 * is an error. */
int clef_answer(const ClefQuestion& q, const std::vector<float>& logits, json* out, ApiError* err);

/* joint_schema_model.py's render: a string as it is, anything else as compact JSON with its keys
 * sorted and non-ASCII kept. Exposed for the tests. */
std::string clef_render(const json& v);

/* The request's words for the head on the card (abi/rad_clef.h, Request::plugin_words): the
 * prompt cut into segments at every span's edges, which segment each question and option is,
 * each option's question and tokens, each question's type. 400 past the head's bounds. */
int clef_words(const ClefRequest& r, std::vector<int32_t>* out, ApiError* err);

/* The options a request has in all, in head order: the head returns three values for each. */
int64_t clef_option_count(const ClefRequest& r);

/* The logits-row positions the head's three values an option land at (prior, cosine, residual;
 * each stretch RAD_CLEF_MAX_OPTIONS wide), as TokenScores asks for them. */
std::vector<int32_t> clef_score_ids(int64_t n_options);

/* The head's learned scalars, from the container header (rad-convert writes them from
 * joint_head.safetensors; the engine copies them to ServerOptions). */
struct ClefScalars {
    double prior_logit_scale = 0.0, joint_logit_scale = 0.0, residual_gate = 0.0;
};

/* One logit an option from the head's three values (`scores` in clef_score_ids' order):
 * exp(prior_logit_scale) * prior + sigmoid(residual_gate) * (exp(joint_logit_scale) * cosine +
 * residual), the scales clamped at 100 as the release clamps them. */
std::vector<float> clef_logits(const std::vector<float>& scores, int64_t n_options,
                               const ClefScalars& s);

}  /* namespace server */
}  /* namespace rad */
