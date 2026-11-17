/* systemone.h -- POST /v1/systemone: LiquidAI d1's decisions, a distribution read off one position.
 *
 * d1 is a decision model. It is given a state (text, any JSON value, pictures, or a mix) and named
 * questions; each answer is a softmax over a few tokens at the position right after the assistant
 * header: `yes` / `no` for a noul, one code per option for a choice, one digit per level for a
 * score. Nothing is generated. The API and prompt are the model's own (prompt.py and api.py on the
 * Hub). The engine sees only Request::token_scores: read these ids' raw logits at the prompt's last
 * position.
 *
 * Served only when the container declares d1 (`decision.type` = `d1`, set at convert with
 * `--set decision.type=d1`); the HF config does not declare it.
 */
#pragma once

#include "oai.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace rad {
namespace server {

struct DecisionQuestion {
    std::string name, type;                     /* type: noul | choice | score */
    /* One entry per answer, in response order: "true"/"false", the option names, or the levels
     * "0".."n-1". */
    std::vector<std::string> labels;
    std::vector<std::string> legend;            /* score: each level's description */
    /* The ids read at the answer slot, and per answer the indices of its spellings among them;
     * an answer scores as its best spelling. */
    std::vector<int32_t>             candidates;
    std::vector<std::vector<size_t>> groups;
    std::vector<int32_t>             tokens;    /* the whole prompt */
};

struct DecisionRequest {
    std::vector<DecisionQuestion> questions;
    /* The pictures, placed in the shared head of every question's prompt. */
    std::shared_ptr<mm::PromptMedia> media;
    /* How many leading tokens every question's prompt shares: the state, tokenised once. 0 for
     * a single question, which d1 tokenises as one text. */
    int32_t trunk = 0;
    /* Input tokens as d1 counts them: the shared head once plus each question's tail. */
    int64_t input_tokens = 0;
};

/* Parse and render. `max_ctx` bounds every question's prompt (one position stays free for the
 * token the request samples and discards). */
/* Shared by the System One models (d1, clef, laya, mica), whose prompts render request JSON as
 * their Python reference implementations do. */
/* A key of `o` outside `allowed` is refused, named at `path`. */
bool only_keys(const json& o, std::initializer_list<const char*> allowed, const std::string& path,
               ApiError* err);
/* json.dumps(v, ensure_ascii=False): the value's own key order, ", " and ": ". */
std::string py_json_dumps(const json& v);
/* json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")). */
std::string py_json_compact_sorted(const json& v);
/* Empty or whitespace only, as Python's `not s.strip()`. */
bool blank_text(const std::string& s);

int parse_decision_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                           const std::string& model_id, DecisionRequest* out, ApiError* err);

/* One question's answer from its candidates' raw logits: each answer scores its best spelling and
 * a softmax over the answers is the distribution. A non-finite logit is an error. */
int decision_answer(const DecisionQuestion& q, const std::vector<float>& logits, json* out,
                    ApiError* err);

}  /* namespace server */
}  /* namespace rad */
