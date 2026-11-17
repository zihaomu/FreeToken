/* systemone.cpp -- d1's prompt and readout (systemone.h).
 *
 * Every prompt string below is d1's own, from prompt.py in the model's repository: the user turn,
 * the state block, the three question blocks, the option codes and their fallback codebook, and the
 * yes/no spellings. The model was trained on exactly these; a reworded instruction or a different
 * option code is a different, worse-calibrated question.
 *
 * That text is Liquid AI's, under the LFM Open License v1.0 (core/server/LICENSE.lfm, and NOTICE).
 * This file is a C++ rewrite of prompt.py's and api.py's logic, modified from them, with their
 * strings kept as they are.
 *
 * The tokenisation split also follows d1's runner: a single question's prompt is tokenised as one
 * text; with several, the shared head (state and pictures) and each question's tail are tokenised
 * separately, and the runner reads the head once and branches. BPE can merge across the join, so
 * the two give different ids.
 */
#include "systemone.h"

#include "llama/src/unicode.h"
#include "mm/processor.h"

#include "util/json.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace rad {
namespace server {

namespace {

const char* const kImStart = "<|im_start|>";
const char* const kImEnd   = "<|im_end|>";


/* The one id `text` encodes to, or -1 when it is not a single token. */
int32_t one_token(const ITokenizer& tok, const std::string& text) {
    const std::vector<int32_t> ids = tok.encode(text, false, true);
    return ids.size() == 1 ? ids[0] : -1;
}

/* An answer's spellings: the distinct single-token ones among `texts` (prompt.py, _ids). */
void add_answer(DecisionQuestion* q, const ITokenizer& tok, const std::vector<std::string>& texts) {
    std::vector<size_t> group;
    for (const std::string& t : texts) {
        const int32_t id = one_token(tok, t);
        if (id < 0) continue;
        size_t at = 0;
        while (at < q->candidates.size() && q->candidates[at] != id) ++at;
        if (at == q->candidates.size()) q->candidates.push_back(id);
        bool have = false;
        for (size_t g : group) have = have || g == at;
        if (!have) group.push_back(at);
    }
    q->groups.push_back(std::move(group));
}

/* Python's str.strip() and str.isalpha() over one stripped label. */
std::string strip(const std::string& s) {
    const std::vector<uint32_t> c = unicode_cpts_from_utf8(s);
    size_t a = 0, b = c.size();
    while (a < b && unicode_cpt_flags_from_cpt(c[a]).is_whitespace) ++a;
    while (b > a && unicode_cpt_flags_from_cpt(c[b - 1]).is_whitespace) --b;
    std::string out;
    for (size_t i = a; i < b; ++i) out += unicode_cpt_to_utf8(c[i]);
    return out;
}
bool one_letter(const std::string& s) {
    const std::vector<uint32_t> c = unicode_cpts_from_utf8(s);
    return c.size() == 1 && unicode_cpt_flags_from_cpt(c[0]).is_letter;
}

std::string two_digits(size_t i) { return (i < 10 ? "0" : "") + std::to_string(i); }

/* prompt.py, option_codes: the labels themselves when every one is a single letter, else A..Z,
 * else 00, 01, ... */
std::vector<std::string> option_codes(const std::vector<std::string>& labels) {
    std::vector<std::string> codes;
    bool letters = !labels.empty();
    for (const std::string& l : labels) letters = letters && one_letter(strip(l));
    for (size_t i = 0; i < labels.size(); ++i)
        codes.push_back(letters ? strip(labels[i])
                        : labels.size() <= 26 ? std::string(1, (char)('A' + i)) : two_digits(i));
    return codes;
}

/* prompt.py, _FALLBACK_POOL: where a code is not one token of its own, or is already taken. */
const std::vector<std::string>& fallback_pool() {
    static const std::vector<std::string> pool = [] {
        std::vector<std::string> v;
        for (char c = 'A'; c <= 'Z'; ++c) v.emplace_back(1, c);
        for (size_t i = 0; i < 100; ++i) v.push_back(two_digits(i));
        for (char c = 'a'; c <= 'z'; ++c) v.emplace_back(1, c);
        for (size_t i = 0; i < 200; ++i) v.push_back("#" + std::to_string(i));
        for (char a = 'A'; a <= 'Z'; ++a)
            for (char b = 'A'; b <= 'Z'; ++b) v.push_back(std::string{ a, b });
        return v;
    }();
    return pool;
}

/* A string criterion, or null (Python's None, which a question block writes as "None"). */
bool text_or_null(const json& v, const std::string& path, ApiError* err) {
    if (v.is_string() || v.is_null()) return true;
    *err = err_bad(path, "must be a string or null");
    return false;
}
std::string py_str(const json& v) { return v.is_null() ? "None" : v.get<std::string>(); }

/* One question: its block (prompt.py, question_block), the assistant header behind it, and the
 * ids its answers are read at (readout_ids). */
int parse_question(const std::string& name, const json& src, const ITokenizer& tok,
                   DecisionQuestion* q, std::string* suffix, ApiError* err) {
    const std::string path = "questions." + name;
    auto bad = [&](const std::string& field, const std::string& why) {
        *err = err_bad(path + field, why);
        return RAD_E_INVAL;
    };
    if (!src.is_object()) return bad("", "must be an object");
    if (!only_keys(src, { "type", "instructions", "criteria" }, path + ".", err)) return RAD_E_INVAL;
    auto it = src.find("instructions");
    if (it == src.end() || !it->is_string()) return bad(".instructions", "must be a string");
    const std::string instructions = it->get<std::string>();
    auto ty = src.find("type");
    if (ty != src.end() && !ty->is_string()) return bad(".type", "must be noul, choice or score");
    q->name = name;
    q->type = ty == src.end() ? "choice" : ty->get<std::string>();   /* prompt.py's default */
    const json criteria = src.contains("criteria") ? src.at("criteria") : json();

    std::string block;
    if (q->type == "noul") {
        if (!criteria.is_null() && !criteria.is_object())
            return bad(".criteria", "must be an object with `true` and `false`, or absent");
        if (criteria.is_object()) {
            if (!only_keys(criteria, { "true", "false" }, path + ".criteria.", err)) return RAD_E_INVAL;
            for (const auto& kv : criteria.items())
                if (!text_or_null(kv.value(), path + ".criteria." + kv.key(), err)) return RAD_E_INVAL;
        }
        block = instructions;
        if (criteria.is_object() && !criteria.empty())
            block += "\nYes: " + py_str(criteria.value("true", json())) +
                     "\nNo: " + py_str(criteria.value("false", json()));
        block += "\n\nReply with yes or no only.";
        q->labels = { "true", "false" };
        add_answer(q, tok, { "yes", "Yes", "YES" });
        add_answer(q, tok, { "no", "No", "NO" });
        if (q->groups[0].empty() || q->groups[1].empty()) {
            *err = err_not_implemented("this vocabulary has no single-token yes or no");
            return RAD_E_UNSUPPORTED;
        }
    } else if (q->type == "choice") {
        if (!criteria.is_object() || criteria.empty())
            return bad(".criteria", "must be a non-empty object of option names to descriptions");
        std::vector<std::string> descs;
        for (const auto& kv : criteria.items()) {
            if (!text_or_null(kv.value(), path + ".criteria." + kv.key(), err)) return RAD_E_INVAL;
            q->labels.push_back(kv.key());
            std::string d = kv.value().is_null() ? std::string() : kv.value().get<std::string>();
            if (d.empty()) {                 /* `desc or label.replace("_", " ")` */
                d = kv.key();
                for (char& c : d) if (c == '_') c = ' ';
            }
            descs.push_back(std::move(d));
        }
        /* prompt.py, aliases: each option's code if it is one token nobody has, else the first
         * fallback that is. Then readout_ids: the code, and " code" where that is one token. */
        std::unordered_set<int32_t> used;
        block = instructions + "\n\nOptions:\n";
        const std::vector<std::string> codes = option_codes(q->labels);
        for (size_t i = 0; i < codes.size(); ++i) {
            std::string code;
            int32_t id = one_token(tok, codes[i]);
            if (id >= 0 && !used.count(id)) code = codes[i];
            else {
                id = -1;
                for (const std::string& f : fallback_pool()) {
                    const int32_t t = one_token(tok, f);
                    if (t >= 0 && !used.count(t)) { id = t; code = f; break; }
                }
            }
            if (id < 0) return bad(".criteria", "more options than this vocabulary has distinct "
                                                "single-token codes for");
            used.insert(id);
            if (i) block += "\n";
            block += code + " " + descs[i];
            add_answer(q, tok, { code, " " + code });
        }
        block += "\n\nReply with the option code only.";
    } else if (q->type == "score") {
        if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10)
            return bad(".criteria", "must be an array of 2 to 10 level descriptions, lowest first");
        block = instructions + "\n\n";
        for (size_t i = 0; i < criteria.size(); ++i) {
            if (!criteria[i].is_string()) return bad(".criteria", "each level must be a string");
            q->labels.push_back(std::to_string(i));
            q->legend.push_back(criteria[i].get<std::string>());
            if (i) block += "\n";
            block += q->labels.back() + " " + q->legend.back();
            add_answer(q, tok, { q->labels.back() });
            if (q->groups.back().empty()) {
                *err = err_not_implemented("this vocabulary has no single-token digit " +
                                           q->labels.back());
                return RAD_E_UNSUPPORTED;
            }
        }
        block += "\n\nReply with a single digit 0-" + std::to_string(criteria.size() - 1) + " only.";
    } else {
        return bad(".type", "must be noul, choice or score");
    }
    *suffix = block + kImEnd + "\n" + kImStart + "assistant\n";
    return RAD_OK;
}

/* prompt.py, state_block at its default style: a string as it is, anything else as indented JSON
 * in the order the request gave its keys. */
std::string state_block(const json& state) {
    return (state.is_string() ? state.get<std::string>() : state.dump(2)) + "\n\n";
}

}  /* namespace */

/* ---- shared by the System One models (systemone.h) */

bool only_keys(const json& o, std::initializer_list<const char*> allowed, const std::string& path,
               ApiError* err) {
    for (const auto& kv : o.items()) {
        bool ok = false;
        for (const char* k : allowed) ok = ok || kv.key() == k;
        if (!ok) {
            *err = err_unsupported(path + kv.key(), "not a field of a System One request");
            return false;
        }
    }
    return true;
}

std::string py_json_dumps(const json& v) {
    if (v.is_object()) {
        std::string s = "{";
        bool first = true;
        for (const auto& kv : v.items()) {
            if (!first) s += ", ";
            first = false;
            s += json(kv.key()).dump(-1, ' ', false, json::error_handler_t::replace) + ": " +
                 py_json_dumps(kv.value());
        }
        return s + "}";
    }
    if (v.is_array()) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + py_json_dumps(v[i]);
        return s + "]";
    }
    return v.dump(-1, ' ', false, json::error_handler_t::replace);
}

namespace {

/* The same tree with every object's members in byte order, which is code point order for
 * UTF-8 keys: what Python's json.dumps(sort_keys=True) writes and the reference reads. This
 * replaced a round trip through a sorted-key DOM. */
json sorted_copy(const json& v) {
    if (v.is_array()) {
        json out = json::array();
        for (const auto& e : v) out.push_back(sorted_copy(e));
        return out;
    }
    if (v.is_object()) {
        /* Members appended in sorted order rather than looked up by key: a lookup in an
         * object is a scan, so filling by key made a large object quadratic. */
        std::vector<std::pair<std::string, const json*>> order;
        order.reserve(v.size());
        for (auto it = v.begin(); it != v.end(); ++it) order.emplace_back(it.key(), &it.value());
        std::sort(order.begin(), order.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        json::object_t out;
        out.reserve(order.size());
        for (auto& kv : order) out.emplace_back(std::move(kv.first), sorted_copy(*kv.second));
        return json(std::move(out));
    }
    return v;
}

}  /* namespace */

std::string py_json_compact_sorted(const json& v) {
    return sorted_copy(v).dump(-1, ' ', false, json::error_handler_t::replace);
}

bool blank_text(const std::string& s) {
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c) != 0; });
}


int parse_decision_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                           const std::string& model_id, DecisionRequest* out, ApiError* err) {
    *out = DecisionRequest{};
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    if (!only_keys(b, { "model", "state", "questions", "images" }, "", err)) return RAD_E_INVAL;
    if (b.contains("model") && (!b["model"].is_string() || b["model"] != model_id)) {
        *err = err_bad("model", "must name the loaded model, " + model_id);
        return RAD_E_INVAL;
    }
    if (!b.contains("state")) {
        *err = err_bad("state", "required: a string, any JSON value, or null with images");
        return RAD_E_INVAL;
    }
    const json& qs = b.contains("questions") ? b["questions"] : json();
    if (!qs.is_object() || qs.empty()) {
        *err = err_bad("questions", "must be a non-empty object of named questions");
        return RAD_E_INVAL;
    }

    /* Pictures are decoded here and processed when the head is tokenised. */
    std::vector<std::pair<MediaKind, std::vector<uint8_t>>> media;
    if (b.contains("images")) {
        const json& im = b["images"];
        if (!im.is_array()) { *err = err_bad("images", "must be an array of data: URLs"); return RAD_E_INVAL; }
        if (!im.empty() && (!d.mm || !d.mm->accepts(MediaKind::Image))) {
            *err = err_not_implemented("images need the model's image encoder, and this deployment "
                                       "has none loaded");
            return RAD_E_UNSUPPORTED;
        }
        for (size_t i = 0; i < im.size(); ++i) {
            const std::string field = "images[" + std::to_string(i) + "]";
            if (!im[i].is_string()) { *err = err_bad(field, "must be a data: URL"); return RAD_E_INVAL; }
            std::vector<uint8_t> bytes;
            RAD_TRY(decode_media_url(im[i].get<std::string>(), field, &bytes, err));
            media.push_back({ MediaKind::Image, std::move(bytes) });
        }
    }

    /* prompt.py, prefix_text: BOS, the user turn, the pictures as the chat template writes them
     * (one marker each, which the processor expands), then the state. A null state leaves the
     * question right behind the pictures. */
    const std::string marker = RAD_MEDIA_MARKER;
    std::string head;
    if (d.tok->bos() >= 0) head += d.tok->token_bytes(d.tok->bos());
    head += std::string(kImStart) + "user\n";
    for (size_t i = 0; i < media.size(); ++i) head += marker;
    std::string body_text;
    if (!b["state"].is_null()) body_text = state_block(b["state"]) + "\nQUESTION:\n";
    if (body_text.find(marker) != std::string::npos) {
        *err = err_bad("state", "contains \"" + marker + "\", which this server reserves");
        return RAD_E_INVAL;
    }
    head += body_text;

    std::vector<std::string> suffixes;
    for (const auto& kv : qs.items()) {
        DecisionQuestion q;
        std::string suffix;
        RAD_TRY(parse_question(kv.key(), kv.value(), *d.tok, &q, &suffix, err));
        if (!media.empty() && suffix.find(marker) != std::string::npos) {
            *err = err_bad("questions." + kv.key(), "contains \"" + marker + "\", which this "
                           "server reserves");
            return RAD_E_INVAL;
        }
        out->questions.push_back(std::move(q));
        suffixes.push_back(std::move(suffix));
    }

    /* runner.py, VISION_MAX_PIXELS: d1 caps every picture at 1024 x 1024 pixels' area before its
     * processor, so the same cap applies here. */
    MediaOpts opts;
    opts.max_pixels = 1024 * 1024;
    auto tokenize = [&](const std::string& text, std::vector<int32_t>* ids) -> int {
        if (media.empty()) { *ids = d.tok->encode(text, false, true); return RAD_OK; }
        PromptInput in;
        RAD_TRY(tokenize_with_media(text, d, media, opts, &in, err));
        *ids = std::move(in.tokens);
        out->media = std::move(in.media);
        return RAD_OK;
    };
    if (out->questions.size() == 1) {
        RAD_TRY(tokenize(head + suffixes[0], &out->questions[0].tokens));
        out->input_tokens = (int64_t)out->questions[0].tokens.size();
    } else {
        std::vector<int32_t> trunk;
        RAD_TRY(tokenize(head, &trunk));
        out->trunk = (int32_t)trunk.size();
        out->input_tokens = (int64_t)trunk.size();
        for (size_t i = 0; i < out->questions.size(); ++i) {
            std::vector<int32_t>& t = out->questions[i].tokens;
            const std::vector<int32_t> tail = d.tok->encode(suffixes[i], false, true);
            t = trunk;
            t.insert(t.end(), tail.begin(), tail.end());
            out->input_tokens += (int64_t)tail.size();
        }
    }
    /* One position stays free for the token the request samples and discards. */
    for (const DecisionQuestion& q : out->questions)
        if (lim.max_ctx > 0 && (int64_t)q.tokens.size() >= lim.max_ctx) {
            ApiError e = err_bad("questions." + q.name,
                                 "the prompt is " + std::to_string(q.tokens.size()) +
                                 " tokens; this deployment's context holds " +
                                 std::to_string(lim.max_ctx - 1) + " for a decision");
            e.code = "context_length_exceeded";
            *err = e;
            return RAD_E_INVAL;
        }
    return RAD_OK;
}

int decision_answer(const DecisionQuestion& q, const std::vector<float>& logits, json* out,
                    ApiError* err) {
    if (logits.size() != q.candidates.size() || q.groups.size() != q.labels.size()) {
        *err = err_internal("the engine returned no scores for question " + q.name);
        return RAD_E_STATE;
    }
    std::vector<double> p;
    for (const std::vector<size_t>& g : q.groups) {
        double best = -std::numeric_limits<double>::infinity();
        for (size_t i : g) best = std::max(best, (double)logits[i]);
        if (!std::isfinite(best)) {
            *err = err_internal("question " + q.name + " scored a non-finite logit");
            return RAD_E_STATE;
        }
        p.push_back(best);
    }
    const double peak = *std::max_element(p.begin(), p.end());
    double total = 0.0;
    for (double& v : p) total += (v = std::exp(v - peak));
    for (double& v : p) v /= total;

    /* api.py, answer. */
    json a = json::object();
    a["type"] = q.type;
    if (q.type == "noul") {
        a["noul"] = p[0];
        *out = std::move(a);
        return RAD_OK;
    }
    const size_t best = (size_t)(std::max_element(p.begin(), p.end()) - p.begin());
    json probs = json::object();
    for (size_t i = 0; i < p.size(); ++i) probs[q.labels[i]] = p[i];
    if (q.type == "choice") {
        a["choice"] = q.labels[best];
    } else {
        double expected = 0.0;
        for (size_t i = 0; i < p.size(); ++i) expected += (double)i * p[i];
        a["score"] = expected;
    }
    a["confidence"] = p[best];
    a["probabilities"] = std::move(probs);
    if (q.type == "score") {
        json legend = json::object();
        for (size_t i = 0; i < q.legend.size(); ++i) legend[q.labels[i]] = q.legend[i];
        a["legend"] = std::move(legend);
    }
    *out = std::move(a);
    return RAD_OK;
}

}  /* namespace server */
}  /* namespace rad */
