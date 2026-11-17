/* laya.cpp -- Laya's sequences, words and answers (laya.h).
 *
 * Every string below is laya's own, from laya/common.py: "<type> question: ", the option forms
 * ("label: description", "level i: description", "false: ..."/"true: ...") and the noul defaults.
 * Each piece is tokenized on its own and the ids concatenated, as build_head does, so a marker
 * always sits at its own option's first position.
 */
#include "laya.h"
#include "systemone.h"

#include "rad_laya.h"

#include "util/json.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

namespace rad {
namespace server {

namespace {

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    for (size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size()))
        s.replace(at, from.size(), to);
    return s;
}

/* render_criterion: a string as it is, anything else as JSON text. */
std::string criterion(const json& v) { return v.is_string() ? v.get<std::string>() : py_json_dumps(v); }

/* A choice label as Python's str() writes it (the option text) and as its json module writes a
 * dict key (the response's key): they differ only for a bool. */
std::string label_text(const json& v) {
    if (v.is_boolean()) return v.get<bool>() ? "True" : "False";
    return v.is_string() ? v.get<std::string>() : py_json_dumps(v);
}
std::string label_key(const json& v) {
    return v.is_string() ? v.get<std::string>() : py_json_dumps(v);
}

/* Python compares 1, 1.0 and True as one dict key. */
bool same_label(const json& a, const json& b) {
    const bool na = a.is_number() || a.is_boolean(), nb = b.is_number() || b.is_boolean();
    if (na && nb) {
        auto num = [](const json& v) { return v.is_boolean() ? (v.get<bool>() ? 1.0 : 0.0) : v.get<double>(); };
        return num(a) == num(b);
    }
    return a == b;
}

/* _check_question and _to_internal, then render_options: the option texts in model order. */
int question(const std::string& path, const json& src, LayaQuestion* q, std::string* ins,
             std::vector<std::string>* opts, ApiError* err) {
    if (!src.is_object()) { *err = err_bad(path, "must be an object"); return RAD_E_INVAL; }
    if (!only_keys(src, { "type", "instructions", "criteria" }, path + ".", err)) return RAD_E_INVAL;
    const auto ty = src.find("type");
    if (ty == src.end() || !ty->is_string() ||
        (*ty != "choice" && *ty != "score" && *ty != "noul")) {
        *err = err_bad(path + ".type", "must be choice, score or noul");
        return RAD_E_INVAL;
    }
    q->type = ty->get<std::string>();
    q->type_id = q->type == "choice" ? RAD_LAYA_CHOICE : q->type == "score" ? RAD_LAYA_SCORE : RAD_LAYA_NOUL;

    const auto in = src.find("instructions");
    if (in == src.end() || in->is_null() || (in->is_string() && blank_text(in->get<std::string>())) ||
        ((in->is_array() || in->is_object()) && in->empty())) {
        *err = err_bad(path + ".instructions", "required: the text the model should answer");
        return RAD_E_INVAL;
    }
    *ins = in->is_string() ? in->get<std::string>() : py_json_dumps(*in);

    const json crit = src.contains("criteria") ? src.at("criteria") : json();
    if (q->type == "choice") {
        if (!(crit.is_object() || crit.is_array()) || crit.empty()) {
            *err = err_bad(path + ".criteria", "must be a non-empty object of label to description, "
                                               "or a non-empty array of labels");
            return RAD_E_INVAL;
        }
        if (crit.is_object()) {
            for (const auto& kv : crit.items()) {
                const json& d = kv.value();
                q->keys.push_back(kv.key());
                q->labels.push_back(kv.key());
                opts->push_back(d.is_null() || d == "" ? kv.key() : kv.key() + ": " + criterion(d));
            }
        } else {
            for (size_t i = 0; i < crit.size(); ++i) {
                const json& l = crit[i];
                if (!(l.is_string() || l.is_number() || l.is_boolean())) {
                    *err = err_bad(path + ".criteria", "label " + std::to_string(i) + " must be a "
                                   "string, number or bool: a label is the option's text and its "
                                   "answer key");
                    return RAD_E_INVAL;
                }
                for (size_t j = 0; j < i; ++j)
                    if (same_label(crit[j], l)) {
                        *err = err_bad(path + ".criteria", "label " + std::to_string(i) +
                                       " repeats label " + std::to_string(j) + "; every option "
                                       "needs its own answer key");
                        return RAD_E_INVAL;
                    }
                q->keys.push_back(label_key(l));
                q->labels.push_back(l);
                opts->push_back(label_text(l));
            }
        }
        return RAD_OK;
    }
    if (q->type == "score") {
        if (!crit.is_array() || crit.empty()) {
            *err = err_bad(path + ".criteria", "must be a non-empty array of level descriptions, "
                                               "level 0 first");
            return RAD_E_INVAL;
        }
        for (size_t i = 0; i < crit.size(); ++i) {
            if (crit[i].is_null()) {
                *err = err_bad(path + ".criteria", "level " + std::to_string(i) + " is null; give "
                                                   "every level a description");
                return RAD_E_INVAL;
            }
            q->keys.push_back(std::to_string(i));
            q->legend.push_back(criterion(crit[i]));
            opts->push_back("level " + std::to_string(i) + ": " + q->legend.back());
        }
        return RAD_OK;
    }
    /* noul: false then true, so the second probability is P(true). */
    json t, f;
    if (!crit.is_null()) {
        if (!crit.is_object()) {
            *err = err_bad(path + ".criteria", "must be an object with optional `true` and `false` "
                                               "descriptions, or absent");
            return RAD_E_INVAL;
        }
        for (const auto& kv : crit.items()) {
            std::string k = kv.key();
            for (char& ch : k) ch = (char)std::tolower((unsigned char)ch);
            if (k == "true") t = kv.value();
            else if (k == "false") f = kv.value();
            else {
                *err = err_bad(path + ".criteria", "takes only `true` and `false`, got `" + kv.key() + "`");
                return RAD_E_INVAL;
            }
        }
    }
    q->keys = { "false", "true" };
    opts->push_back("false: " + (f.is_null() || f == "" ? std::string("no, the statement does not hold")
                                                        : criterion(f)));
    opts->push_back("true: " + (t.is_null() || t == "" ? std::string("yes, the statement holds")
                                                       : criterion(t)));
    return RAD_OK;
}

double round4(double v) { return std::round(v * 10000.0) / 10000.0; }

/* temp_bucket, and the clamp the package applies to every fitted temperature as it loads. */
double temperature_of(const LayaQuestion& q, size_t k, const LayaSettings& s) {
    const char* size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
    const auto it = s.temperature_by_options.find(q.type + ":" + size);
    const double t = it != s.temperature_by_options.end() ? it->second : s.temperature[q.type_id];
    return std::isfinite(t) ? std::min(5.0, std::max(0.5, t)) : 1.0;
}

/* confidence_from_probs: 1 - H(p) / log(k), within [0, 1]; 1 below two options. */
double entropy_confidence(const std::vector<float>& p) {
    const size_t k = p.size();
    if (k < 2) return 1.0;
    float ent = 0.0f;
    for (float v : p) ent += v * std::log(std::min(1.0f, std::max(1e-12f, v)));
    const double c = 1.0 - (double)(-ent) / std::log((double)k);
    return c < 0.0 ? 0.0 : c > 1.0 ? 1.0 : c;
}

}  /* namespace */

int parse_laya_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                       const std::string& model_id, const LayaSettings& s, LayaRequest* out,
                       ApiError* err) {
    *out = LayaRequest{};
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    if (!only_keys(b, { "model", "state", "questions", "images" }, "", err)) return RAD_E_INVAL;
    if (b.contains("model") && (!b["model"].is_string() || b["model"] != model_id)) {
        *err = err_bad("model", "must name the loaded model, " + model_id);
        return RAD_E_INVAL;
    }
    if (!b.contains("state") || b["state"].is_null() ||
        (b["state"].is_string() && b["state"].get<std::string>().empty())) {
        *err = err_bad("state", "required: a non-empty string or any JSON value");
        return RAD_E_INVAL;
    }
    if (b.contains("images") && !(b["images"].is_array() && b["images"].empty())) {
        *err = err_not_implemented("images: Laya reads its state as text only");
        return RAD_E_UNSUPPORTED;
    }
    const json& qs = b.contains("questions") ? b["questions"] : json();
    if (!qs.is_object() || qs.empty()) {
        *err = err_bad("questions", "must be a non-empty object of named questions");
        return RAD_E_INVAL;
    }

    const ITokenizer& tok = *d.tok;
    auto ids = [&](const std::string& text) {
        return tok.encode(replace_all(text, s.mask_token, " "), false, true);
    };
    /* The state is tokenized once and cut per question; a list of turns keeps its end. */
    const json& st = b["state"];
    const bool keep_end = st.is_array();
    const std::vector<int32_t> state = ids(st.is_string() ? st.get<std::string>() : py_json_dumps(st));
    out->state_tokens = (int64_t)state.size();

    for (const auto& kv : qs.items()) {
        const std::string path = "questions." + kv.key();
        if (blank_text(kv.key())) {
            *err = err_bad("questions", "a question's name must not be empty");
            return RAD_E_INVAL;
        }
        LayaQuestion q;
        q.name = kv.key();
        std::string ins;
        std::vector<std::string> opts;
        RAD_TRY(question(path, kv.value(), &q, &ins, &opts, err));
        const int64_t n = (int64_t)opts.size();
        if (n > RAD_LAYA_MAX_OPTIONS) {
            *err = err_bad(path + ".criteria", "at most " + std::to_string(RAD_LAYA_MAX_OPTIONS) +
                                               " options a question");
            return RAD_E_INVAL;
        }

        /* build_head: options capped at 48 tokens, all of them shrunk evenly when they leave the
         * instruction fewer than 16 of head_max_len, and the instruction cut to what is left. */
        std::vector<std::vector<int32_t>> opt;
        int64_t used = 0;
        for (const std::string& o : opts) {
            std::vector<int32_t> t = ids(" " + o);
            if (t.size() > 48) t.resize(48);
            t.insert(t.begin(), s.mask);
            used += (int64_t)t.size();
            opt.push_back(std::move(t));
        }
        int64_t budget = s.head_max_len - used;
        if (budget < 16) {
            const int64_t per = std::max<int64_t>(4, (s.head_max_len - 16) / std::max<int64_t>(1, n));
            used = 0;
            for (auto& t : opt) {
                if ((int64_t)t.size() > per) t.resize((size_t)per);
                used += (int64_t)t.size();
            }
            budget = s.head_max_len - used;
        }
        std::vector<int32_t> head = ids(q.type + " question: " + ins);
        head.resize((size_t)std::min<int64_t>((int64_t)head.size(), std::max<int64_t>(8, budget)));

        std::vector<int32_t>& seq = q.tokens;
        seq.push_back(s.cls);
        seq.insert(seq.end(), head.begin(), head.end());
        seq.push_back(s.sep);
        for (const auto& t : opt) {
            q.markers.push_back((int32_t)seq.size());
            seq.insert(seq.end(), t.begin(), t.end());
        }
        seq.push_back(s.sep);
        /* build_sequence: the state gets the room left before the closing [SEP]. */
        const int64_t room = std::max<int64_t>(0, s.max_len - (int64_t)seq.size() - 1);
        const int64_t take = std::min<int64_t>(room, (int64_t)state.size());
        if (keep_end) seq.insert(seq.end(), state.end() - take, state.end());
        else          seq.insert(seq.end(), state.begin(), state.begin() + take);
        seq.push_back(s.sep);
        q.state_used = take;
        if ((int64_t)seq.size() > s.max_len) seq.resize((size_t)s.max_len);
        q.markers.erase(std::remove_if(q.markers.begin(), q.markers.end(),
                                       [&](int32_t m) { return m >= s.max_len; }),
                        q.markers.end());
        if ((int64_t)q.markers.size() != n) {
            *err = err_bad(path + ".criteria", std::to_string(n) + " options do not fit this "
                           "model's " + std::to_string(s.max_len) + "-token sequence (head_max_len " +
                           std::to_string(s.head_max_len) + "); fewer or shorter options fit");
            return RAD_E_INVAL;
        }
        out->questions.push_back(std::move(q));
    }
    return RAD_OK;
}

std::vector<int32_t> laya_words(const LayaQuestion& q) {
    std::vector<int32_t> w((size_t)RAD_LAYA_W_MARKERS, 0);
    w[RAD_LAYA_W_N] = (int32_t)q.tokens.size();
    w[RAD_LAYA_W_TYPE] = q.type_id;
    w[RAD_LAYA_W_K] = (int32_t)q.markers.size();
    w.insert(w.end(), q.markers.begin(), q.markers.end());
    return w;
}

int laya_answer(const LayaQuestion& q, const std::vector<float>& pooled, const LayaSettings& s,
                json* out, ApiError* err) {
    const size_t k = q.keys.size();
    if (pooled.size() < k) {
        *err = err_internal("the plugin returned no logits for question " + q.name);
        return RAD_E_STATE;
    }
    /* _decode_answers, in float32 as the package's numpy rows are. */
    const float t = (float)temperature_of(q, k, s);
    std::vector<float> p(k);
    float peak = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < k; ++i) {
        if (!std::isfinite(pooled[i])) {
            *err = err_internal("question " + q.name + " scored a non-finite logit");
            return RAD_E_STATE;
        }
        p[i] = pooled[i] / t;
        peak = std::max(peak, p[i]);
    }
    float total = 0.0f;
    for (float& v : p) total += (v = std::exp(v - peak));
    for (float& v : p) v /= total;

    json a = json::object();
    a["type"] = q.type;
    if (q.type == "noul") {
        a["noul"] = round4(p[1]);
        *out = std::move(a);
        return RAD_OK;
    }
    json probs = json::object();
    for (size_t i = 0; i < k; ++i) probs[q.keys[i]] = round4(p[i]);
    if (q.type == "choice") {
        const size_t best = (size_t)(std::max_element(p.begin(), p.end()) - p.begin());
        a["choice"] = q.labels[best];
        a["confidence"] = round4(entropy_confidence(p));
        a["probabilities"] = std::move(probs);
    } else {
        double expected = 0.0;
        json legend = json::object();
        for (size_t i = 0; i < k; ++i) {
            expected += (double)i * (double)p[i];
            legend[q.keys[i]] = q.legend[i];
        }
        a["score"] = round4(expected);
        a["confidence"] = round4(entropy_confidence(p));
        a["probabilities"] = std::move(probs);
        a["legend"] = std::move(legend);
    }
    *out = std::move(a);
    return RAD_OK;
}

json laya_usage(const LayaRequest& r) {
    int64_t input = 0, dropped = 0;
    json cut = json::array();
    for (const LayaQuestion& q : r.questions) {
        input += (int64_t)q.tokens.size();
        dropped = std::max(dropped, r.state_tokens - q.state_used);
        if (q.state_used < r.state_tokens) cut.push_back(q.name);
    }
    return json{ { "input_tokens", input }, { "output_tokens", 0 },
                 { "state_tokens", r.state_tokens }, { "state_tokens_dropped", dropped },
                 { "truncated", dropped > 0 }, { "truncated_questions", std::move(cut) } };
}

}  /* namespace server */
}  /* namespace rad */
