/* clef.cpp -- clef's prompt, spans and answer (clef.h).
 *
 * Every string below is the release's own, from joint_schema_model.py: the system prompt, the
 * state and schema framing, the field and option lines, and the noul defaults. The prompt is
 * tokenised piece by piece and the ids concatenated, as encode_record does, because each span is
 * the ids of one piece: BPE over the joined text could merge across a span's edge.
 */
#include "clef.h"
#include "systemone.h"

#include "rad_clef.h"

#include "util/json.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rad {
namespace server {

namespace {

const char* const kSystemPrompt =
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options.";

/* question_options: each allowed option and its description (null for none). */
int question_options(const std::string& path, const std::string& type, const json& criteria,
                     ClefQuestion* q, std::vector<json>* descs, ApiError* err) {
    if (type == "noul") {
        json c = { { "true", "The proposition is true or the answer is yes." },
                   { "false", "The proposition is false or the answer is no." } };
        if (!criteria.is_null()) {
            if (!criteria.is_object()) {
                *err = err_bad(path + ".criteria", "must be an object with `true` and `false`, or absent");
                return RAD_E_INVAL;
            }
            if (!only_keys(criteria, { "true", "false" }, path + ".criteria.", err)) return RAD_E_INVAL;
            for (const auto& kv : criteria.items()) c[kv.key()] = kv.value();
        }
        q->options = { "true", "false" };
        *descs = { c["true"], c["false"] };
        q->response_order = q->options;
        return RAD_OK;
    }
    if (type == "choice") {
        if (!criteria.is_object() || criteria.empty()) {
            *err = err_bad(path + ".criteria", "must be a non-empty object of option names to descriptions");
            return RAD_E_INVAL;
        }
        std::vector<std::pair<std::string, json>> o;
        for (const auto& kv : criteria.items()) {
            o.emplace_back(kv.key(), kv.value());
            q->response_order.push_back(kv.key());
        }
        /* Python sorts str keys by code point, which UTF-8 byte order preserves. */
        std::sort(o.begin(), o.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (auto& p : o) {
            q->options.push_back(p.first);
            descs->push_back(std::move(p.second));
        }
        return RAD_OK;
    }
    if (!criteria.is_array() || criteria.empty()) {
        *err = err_bad(path + ".criteria", "must be a non-empty array of level descriptions, lowest first");
        return RAD_E_INVAL;
    }
    for (size_t i = 0; i < criteria.size(); ++i) {
        q->options.push_back(std::to_string(i));
        descs->push_back(criteria[i]);
    }
    q->response_order = q->options;
    for (const json& c : criteria) q->legend.push_back(c.dump());
    return RAD_OK;
}

double round4(double v) { return std::round(v * 10000.0) / 10000.0; }

}  /* namespace */

std::string clef_render(const json& v) {
    return v.is_string() ? v.get<std::string>() : py_json_compact_sorted(v);
}

int parse_clef_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                       const std::string& model_id, ClefRequest* out, ApiError* err) {
    *out = ClefRequest{};
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    if (!only_keys(b, { "model", "state", "questions", "images" }, "", err)) return RAD_E_INVAL;
    if (b.contains("model") && (!b["model"].is_string() || b["model"] != model_id)) {
        *err = err_bad("model", "must name the loaded model, " + model_id);
        return RAD_E_INVAL;
    }
    if (!b.contains("state") || (b["state"].is_string() && b["state"].get<std::string>().empty())) {
        *err = err_bad("state", "required: a non-empty string or any JSON value");
        return RAD_E_INVAL;
    }
    if (b.contains("images") && !(b["images"].is_array() && b["images"].empty())) {
        *err = err_not_implemented("images: this server reads clef's state as text only");
        return RAD_E_UNSUPPORTED;
    }
    const json& qs = b.contains("questions") ? b["questions"] : json();
    if (!qs.is_object() || qs.empty()) {
        *err = err_bad("questions", "must be a non-empty object of named questions");
        return RAD_E_INVAL;
    }

    const ITokenizer& tok = *d.tok;
    auto ids = [&](const std::string& text) { return tok.encode(text, false, true); };
    auto append = [](std::vector<int32_t>* to, const std::vector<int32_t>& v) {
        to->insert(to->end(), v.begin(), v.end());
    };

    /* encode_record: the schema first, with spans relative to its own start. */
    std::vector<int32_t> schema = ids("\n\nSCHEMA FIELDS:\n");
    size_t index = 0;
    for (const auto& kv : qs.items()) {
        const std::string path = "questions." + kv.key();
        const json& src = kv.value();
        if (!src.is_object()) { *err = err_bad(path, "must be an object"); return RAD_E_INVAL; }
        if (!only_keys(src, { "type", "instructions", "criteria" }, path + ".", err)) return RAD_E_INVAL;
        const auto ty = src.find("type");
        if (ty == src.end() || !ty->is_string() ||
            (*ty != "noul" && *ty != "choice" && *ty != "score")) {
            *err = err_bad(path + ".type", "must be noul, choice or score");
            return RAD_E_INVAL;
        }
        ClefQuestion q;
        q.name = kv.key();
        q.type = ty->get<std::string>();
        q.type_id = q.type == "noul" ? 0 : q.type == "choice" ? 1 : 2;
        std::vector<json> descs;
        RAD_TRY(question_options(path, q.type, src.contains("criteria") ? src.at("criteria") : json(),
                                 &q, &descs, err));

        append(&schema, ids("\nFIELD " + std::to_string(++index) + "\nID: " + q.name + "\nTYPE: " +
                            q.type + "\nINSTRUCTION: "));
        json instructions = src.contains("instructions") ? src.at("instructions") : json();
        if (instructions.is_null() || instructions == "") instructions = q.name;
        q.question.first = (int64_t)schema.size();
        append(&schema, ids(clef_render(instructions)));
        q.question.second = (int64_t)schema.size();
        append(&schema, ids("\nALLOWED OPTIONS:\n"));
        for (size_t i = 0; i < q.options.size(); ++i) {
            append(&schema, ids("OPTION " + std::to_string(i + 1) + ": "));
            json semantics = { { "option_id", q.options[i] } };
            if (!descs[i].is_null()) semantics["description"] = descs[i];
            const int64_t start = (int64_t)schema.size();
            append(&schema, ids(clef_render(semantics)));
            q.option_spans.emplace_back(start, (int64_t)schema.size());
            append(&schema, ids("\n"));
        }
        append(&schema, ids("END FIELD\n"));
        /* A span the tokenizer left empty would make the head average nothing. */
        if (q.question.first == q.question.second) {
            *err = err_bad(path + ".instructions", "renders to no tokens");
            return RAD_E_INVAL;
        }
        out->questions.push_back(std::move(q));
    }

    const std::vector<int32_t> prefix = ids(std::string("<|im_start|>system\n") + kSystemPrompt +
                                            "<|im_end|>\n<|im_start|>user\nSTATE:\n");
    const std::vector<int32_t> state = ids(clef_render(b["state"]));
    const std::vector<int32_t> suffix =
        ids("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");
    const int64_t n = (int64_t)(prefix.size() + state.size() + schema.size() + suffix.size());
    /* The context, or the head's own bound where that is less (abi/rad_clef.h). */
    const int64_t cap = lim.max_ctx > 0 && lim.max_ctx < RAD_CLEF_MAX_PROMPT ? lim.max_ctx
                                                                             : RAD_CLEF_MAX_PROMPT;
    if (n > cap) {
        ApiError e = err_bad("state", "the prompt is " + std::to_string(n) + " tokens; " +
                             (cap == lim.max_ctx ? "this deployment's context holds "
                                                 : "clef's head takes at most ") +
                             std::to_string(cap));
        e.code = "context_length_exceeded";
        e.n_prompt_tokens = n;
        e.n_ctx = cap;
        *err = e;
        return RAD_E_INVAL;
    }
    const int64_t offset = (int64_t)(prefix.size() + state.size());
    for (ClefQuestion& q : out->questions) {
        q.question.first += offset;
        q.question.second += offset;
        for (TokenSpan& s : q.option_spans) { s.first += offset; s.second += offset; }
    }
    out->tokens = prefix;
    append(&out->tokens, state);
    append(&out->tokens, schema);
    append(&out->tokens, suffix);
    return RAD_OK;
}

int64_t clef_option_count(const ClefRequest& r) {
    int64_t o = 0;
    for (const ClefQuestion& q : r.questions) o += (int64_t)q.options.size();
    return o;
}

int clef_words(const ClefRequest& r, std::vector<int32_t>* out, ApiError* err) {
    const int64_t N = (int64_t)r.tokens.size(), Q = (int64_t)r.questions.size();
    const int64_t O = clef_option_count(r);
    int64_t L = 0;
    for (const ClefQuestion& q : r.questions)
        for (const TokenSpan& s : q.option_spans) L += s.second - s.first;
    if (Q > RAD_CLEF_MAX_QUESTIONS) {
        *err = err_bad("questions", "at most " + std::to_string(RAD_CLEF_MAX_QUESTIONS) +
                                    " questions a request");
        return RAD_E_INVAL;
    }
    if (O > RAD_CLEF_MAX_OPTIONS) {
        *err = err_bad("questions", "at most " + std::to_string(RAD_CLEF_MAX_OPTIONS) +
                                    " options a request, every question together");
        return RAD_E_INVAL;
    }
    if (L > RAD_CLEF_MAX_LEX) {
        *err = err_bad("questions", "the options render to " + std::to_string(L) +
                                    " tokens; at most " + std::to_string(RAD_CLEF_MAX_LEX) +
                                    " a request");
        return RAD_E_INVAL;
    }
    /* The prompt cut at every span's edges. Spans are disjoint, so each is one whole segment. */
    std::vector<int32_t> cuts = { 0, (int32_t)N };
    for (const ClefQuestion& q : r.questions) {
        cuts.push_back((int32_t)q.question.first);
        cuts.push_back((int32_t)q.question.second);
        for (const TokenSpan& s : q.option_spans) {
            cuts.push_back((int32_t)s.first);
            cuts.push_back((int32_t)s.second);
        }
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    const int64_t S = (int64_t)cuts.size() - 1;
    auto seg_at = [&](int64_t start) {
        return (int32_t)(std::lower_bound(cuts.begin(), cuts.end(), (int32_t)start) - cuts.begin());
    };

    const RadClefWords w = rad_clef_words_layout((int32_t)N, (int32_t)Q, (int32_t)O, (int32_t)L,
                                                 (int32_t)S);
    std::vector<int32_t>& v = *out;
    v.assign((size_t)w.total, 0);
    v[RAD_CLEF_W_N] = (int32_t)N;
    v[RAD_CLEF_W_Q] = (int32_t)Q;
    v[RAD_CLEF_W_O] = (int32_t)O;
    v[RAD_CLEF_W_L] = (int32_t)L;
    v[RAD_CLEF_W_S] = (int32_t)S;
    std::copy(cuts.begin(), cuts.end(), v.begin() + w.seg_cu);
    int64_t o = 0, l = 0;
    for (int64_t qi = 0; qi < Q; ++qi) {
        const ClefQuestion& q = r.questions[(size_t)qi];
        v[(size_t)(w.q_seg + qi)]  = seg_at(q.question.first);
        v[(size_t)(w.type + qi)]   = q.type_id;
        v[(size_t)(w.opt_cu + qi)] = (int32_t)o;
        for (const TokenSpan& s : q.option_spans) {
            v[(size_t)(w.o_seg + o)]  = seg_at(s.first);
            v[(size_t)(w.o_q + o)]    = (int32_t)qi;
            v[(size_t)(w.lex_cu + o)] = (int32_t)l;
            for (int64_t t = s.first; t < s.second; ++t) v[(size_t)(w.lex_ids + l++)] = r.tokens[(size_t)t];
            ++o;
        }
    }
    v[(size_t)(w.opt_cu + Q)] = (int32_t)O;
    v[(size_t)(w.lex_cu + O)] = (int32_t)L;
    v[(size_t)w.last]         = (int32_t)(N - 1);
    v[(size_t)w.cu_q]     = 0; v[(size_t)w.cu_q + 1] = (int32_t)Q;
    v[(size_t)w.cu_o]     = 0; v[(size_t)w.cu_o + 1] = (int32_t)O;
    v[(size_t)w.cu_k]     = 0; v[(size_t)w.cu_k + 1] = (int32_t)N;
    return RAD_OK;
}

std::vector<int32_t> clef_score_ids(int64_t n_options) {
    std::vector<int32_t> ids;
    ids.reserve((size_t)(3 * n_options));
    for (int k = 0; k < 3; ++k)
        for (int64_t o = 0; o < n_options; ++o) ids.push_back((int32_t)(k * RAD_CLEF_MAX_OPTIONS + o));
    return ids;
}

std::vector<float> clef_logits(const std::vector<float>& scores, int64_t n_options,
                               const ClefScalars& s) {
    const double cap = std::log(100.0);
    const double ps = std::exp(std::min(s.prior_logit_scale, cap));
    const double js = std::exp(std::min(s.joint_logit_scale, cap));
    const double gate = 1.0 / (1.0 + std::exp(-s.residual_gate));
    std::vector<float> out((size_t)n_options, 0.0f);
    if ((int64_t)scores.size() < 3 * n_options) return out;
    for (int64_t o = 0; o < n_options; ++o) {
        const double prior = scores[(size_t)o], cosine = scores[(size_t)(n_options + o)];
        const double residual = scores[(size_t)(2 * n_options + o)];
        out[(size_t)o] = (float)(ps * prior + gate * (js * cosine + residual));
    }
    return out;
}

int clef_answer(const ClefQuestion& q, const std::vector<float>& logits, json* out, ApiError* err) {
    if (logits.size() != q.options.size()) {
        *err = err_internal("the head returned no logits for question " + q.name);
        return RAD_E_STATE;
    }
    double peak = -std::numeric_limits<double>::infinity();
    for (float l : logits) {
        if (!std::isfinite(l)) {
            *err = err_internal("question " + q.name + " scored a non-finite logit");
            return RAD_E_STATE;
        }
        peak = std::max(peak, (double)l);
    }
    std::vector<double> p;
    double total = 0.0;
    for (float l : logits) total += p.emplace_back(std::exp((double)l - peak));
    for (double& v : p) v /= total;
    auto prob = [&](const std::string& option) {
        return p[(size_t)(std::find(q.options.begin(), q.options.end(), option) - q.options.begin())];
    };

    /* systemone_answer. */
    json a = json::object();
    a["type"] = q.type;
    if (q.type == "noul") {
        a["noul"] = round4(prob("true"));
        *out = std::move(a);
        return RAD_OK;
    }
    json probs = json::object();
    if (q.type == "choice") {
        std::string best = q.response_order[0];
        for (const std::string& o : q.response_order) if (prob(o) > prob(best)) best = o;
        a["choice"] = best;
        a["confidence"] = round4(prob(best));
        for (const std::string& o : q.response_order) probs[o] = round4(prob(o));
        a["probabilities"] = std::move(probs);
    } else {
        double expected = 0.0, top = 0.0;
        json legend = json::object();
        for (size_t i = 0; i < q.options.size(); ++i) {
            expected += (double)i * p[i];
            top = std::max(top, p[i]);
            legend[q.options[i]] = json::parse(q.legend[i]);
            probs[q.options[i]] = round4(p[i]);
        }
        a["score"] = round4(expected);
        a["confidence"] = round4(top);
        a["legend"] = std::move(legend);
        a["probabilities"] = std::move(probs);
    }
    *out = std::move(a);
    return RAD_OK;
}

}  /* namespace server */
}  /* namespace rad */
