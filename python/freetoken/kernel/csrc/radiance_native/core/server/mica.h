/* mica.h -- POST /v1/systemone for Mica (sky7350/Mica-v0.5-4B), a decision model fine-tuned from
 * Qwen3.5-4B and served by the qwen35 plugins as they are.
 *
 * Mica reads every answer out of one forward pass: one prompt a question (a fixed system message,
 * the state, the question and the candidates under labels), and a softmax over the label tokens'
 * logits at the prompt's last position, `No`/`Yes` for a noul and `A`, `B`, ... for a choice or a
 * score. That is d1's readout (systemone.h), so the requests run the same way: one engine request
 * a question, the state shared through the prefix cache, each reading its labels' logits
 * (Request::token_scores).
 *
 * This file is a C++ rewrite of Mica's own contract.py (rows_from_request, the multi-answer
 * expansion, the response envelope) and native.py (prompt_for, typed_result), checked against the
 * fixtures Mica's repository ships (tests/data/mica/contract_cases.json): every rendered prompt byte
 * for byte, every response from the same probabilities. Mica's deployed runtime also has a
 * "long-input mode" that approximates prompts over about 1k tokens; this is its exact "plain"
 * profile.
 *
 * Served only when the container declares mica (`--set decision.type=mica` at convert).
 */
#pragma once

#include "oai.h"

#include <string>
#include <vector>

namespace rad {
namespace server {

/* The container's calibration (decision.mica.*, from Mica's calibration.json; the engine copies
 * them here): one temperature for every question type, and a bias on a noul's true logit. */
struct MicaSettings {
    double temperature = 1.0;
    double noul_logit_bias = 0.0;
};

struct MicaCandidate {
    std::string id, text;
    int ordinal = -1;                   /* score levels: 0..n-1 */
};

/* One question as Mica scores it (a row of rows_from_request), or one inclusion question of a
 * multi-answer choice. */
struct MicaRow {
    std::string id, kind;               /* kind: noul | choice | score */
    std::string question;
    std::vector<MicaCandidate> candidates;
    bool cannot_grade = false, multi = false;
    double abstain = -1.0;              /* abstain_threshold, or -1 */
    std::string multi_of, multi_option; /* an inclusion row: its question and its candidate */
};

struct MicaRequest {
    /* The state as every prompt shows it: a string as it is, any other value as JSON text, with
     * chat-control tokens escaped (native.py). */
    std::string state;
    /* The rows scored: the plain questions in request order, then every multi-answer question's
     * inclusion rows. `multi` keeps the multi-answer questions themselves, for their answers. */
    std::vector<MicaRow> rows;
    size_t n_plain = 0;
    std::vector<MicaRow> multi;
};

/* rows_from_request and the multi-answer expansion. 400 for what Mica refuses. */
int parse_mica_request(const std::string& body, const OaiLimits& lim, const std::string& model_id,
                       MicaRequest* out, ApiError* err);

/* prompt_for: the row's prompt over the rendered state, as Mica's chat template renders it. */
std::string mica_prompt(const MicaRow& row, const std::string& state);

/* The labels a row is read at: `No`, `Yes` for a noul, the first n of Mica's 255 for the rest. */
std::vector<std::string> mica_labels(const MicaRow& row);

/* calibration.distribution: the label logits over the temperature (and a noul's bias), softmaxed. */
std::vector<double> mica_distribution(const MicaRow& row, const std::vector<float>& logits,
                                      const MicaSettings& s);

/* answer_request's packaging: each plain row's typed answer (with confidence_pmax and abstain
 * where asked), then each multi-answer question's inclusions, from one distribution a row. */
int mica_answers(const MicaRequest& r, const std::vector<std::vector<double>>& dist, json* out,
                 ApiError* err);

}  /* namespace server */
}  /* namespace rad */
