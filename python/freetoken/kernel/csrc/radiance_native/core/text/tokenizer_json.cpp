/* tokenizer_json.cpp -- reading a vocab in, and writing it into the .rad.
 *
 * Three sources and one sink:
 *   parse_tokenizer_json()  the HuggingFace file, interpreted rather than hashed; the preferred
 *                           path (spec §12).
 *   vocab_from_gguf()       GGUF metadata, where the chain is lost and is reconstructed from
 *                           `tokenizer.ggml.pre`, a name llama.cpp derived by hashing the
 *                           original tokenizer.json.
 *   vocab_serialize()       into the .rad's vocab section, and back.
 */
#include "text/tokenizer.h"
#include "text/tokenizer_grapheme.h"
#include "text/tokenizer_regex.h"
#include "text/gguf.h"

#include "util/json.h"

#include <cstring>
#include <fstream>
#include <new>
#include <sstream>

namespace rad {

using json = Json;

namespace {

/* HuggingFace's SplitDelimiterBehavior, in the order our iarg encodes; -1 for another name. */
int behavior_from_name(const std::string& s) {
    if (s == "Removed")            return 0;
    if (s == "Isolated")           return 1;
    if (s == "MergedWithPrevious") return 2;
    if (s == "MergedWithNext")     return 3;
    if (s == "Contiguous")         return 4;
    return -1;
}

/* Metaspace's prepend scheme, in the order our iarg encodes; -1 for another name. */
int prepend_scheme_from_name(const std::string& s) {
    if (s == "never")  return 0;
    if (s == "first")  return 1;
    if (s == "always") return 2;
    return -1;
}

/* A field value the tokenizer has no encoding for: guessing one would change the token
 * boundaries, so the vocabulary is refused like an unknown step. */
int refuse_value(const char* origin, const char* field, const std::string& v) {
    RAD_ERR("%s: %s '%s' is not supported", origin, field, v.c_str());
    return RAD_E_UNSUPPORTED;
}

/* Merges read by token text into ids: a pair naming a text that is not a token is dropped and
 * counted, since some vocabs ship merges over strings that are not tokens. */
struct MergeIds {
    std::unordered_map<std::string, uint32_t> id_of;
    size_t dropped = 0;

    explicit MergeIds(const std::vector<std::string>& tokens) {
        id_of.reserve(tokens.size() * 2);
        for (size_t i = 0; i < tokens.size(); ++i) id_of.emplace(tokens[i], (uint32_t)i);
    }
    void add(const std::string& l, const std::string& r, VocabBuild& out) {
        auto li = id_of.find(l), ri = id_of.find(r);
        if (li == id_of.end() || ri == id_of.end()) { ++dropped; return; }
        out.merges.emplace_back(li->second, ri->second);
    }
    /* The "a b" spelling, split at the first space. */
    void add_spaced(const std::string& s, VocabBuild& out) {
        const size_t sp = s.find(' ');
        if (sp == std::string::npos) { ++dropped; return; }
        add(s.substr(0, sp), s.substr(sp + 1), out);
    }
};

/* A Metaspace step over U+2581 that always prepends it, as a SentencePiece vocab reads. */
VocabStep metaspace(uint32_t kind) {
    VocabStep s;
    s.kind = kind;
    s.arg0 = "\xe2\x96\x81";
    s.iarg = 2;
    return s;
}

const json* get(const json& o, const char* k) {
    if (!o.is_object()) return nullptr;
    auto it = o.find(k);
    return it == o.end() || it->is_null() ? nullptr : &*it;
}

std::string str_or(const json& o, const char* k, const char* dflt) {
    const json* v = get(o, k);
    return v && v->is_string() ? v->get<std::string>() : std::string(dflt);
}

bool bool_or(const json& o, const char* k, bool dflt) {
    const json* v = get(o, k);
    return v && v->is_boolean() ? v->get<bool>() : dflt;
}

/* A step's regex is compiled here as well as at vocab load, so rad-convert refuses an
 * unsupported pattern with the construct named instead of writing a container that cannot
 * load. */
int check_regex(const std::string& pat, const char* what, const char* origin) {
    std::unique_ptr<TokRegex> re;
    std::string why;
    if (TokRegex::compile(pat, &re, &why) < 0) {
        RAD_ERR("%s: the %s pattern '%s' is outside the supported regex dialect: %s", origin, what,
                pat.c_str(), why.c_str());
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* HuggingFace's pattern field is either {"String": "..."} (a literal) or {"Regex": "..."}. */
bool read_pattern(const json& o, std::string* pat, bool* is_regex) {
    const json* p = get(o, "pattern");
    if (!p) return false;
    if (p->is_string()) { *pat = p->get<std::string>(); *is_regex = false; return true; }
    const json* s = get(*p, "String");
    if (s && s->is_string()) { *pat = s->get<std::string>(); *is_regex = false; return true; }
    const json* r = get(*p, "Regex");
    if (r && r->is_string()) { *pat = r->get<std::string>(); *is_regex = true;  return true; }
    return false;
}

/* A token id from the file. It indexes the token table as soon as it is read, so a negative or
 * non-integer id would write outside it and a huge one would size it. `limit` is one past the
 * largest id accepted (see parse_tokenizer_json_buf). */
bool read_id(const json& v, uint64_t limit, int32_t* out) {
    if (!v.is_number_integer()) return false;
    if (!v.is_number_unsigned() && v.get<int64_t>() < 0) return false;
    const uint64_t id = v.is_number_unsigned() ? v.get<uint64_t>() : (uint64_t)v.get<int64_t>();
    if (id >= limit || id > (uint64_t)INT32_MAX) return false;
    *out = (int32_t)id;
    return true;
}

/* --- the three chains ---------------------------------------------------------------------- */

int read_normalizer(const json& n, std::vector<VocabStep>& out, const char* origin);
int read_pretok(const json& n, std::vector<VocabStep>& out, const char* origin);
int read_decoder(const json& n, std::vector<VocabStep>& out, const char* origin);

int read_normalizer(const json& n, std::vector<VocabStep>& out, const char* origin) {
    if (n.is_null()) return RAD_OK;
    const std::string t = str_or(n, "type", "");
    VocabStep s;
    if (t == "Sequence") {
        const json* seq = get(n, "normalizers");
        if (!seq || !seq->is_array()) return RAD_OK;
        for (const json& c : *seq) RAD_TRY(read_normalizer(c, out, origin));
        return RAD_OK;
    }
    if (t == "NFC")  { s.kind = RAD_NORM_NFC;  out.push_back(s); return RAD_OK; }
    if (t == "NFKC") { s.kind = RAD_NORM_NFKC; out.push_back(s); return RAD_OK; }
    if (t == "NFD")  { s.kind = RAD_NORM_NFD;  out.push_back(s); return RAD_OK; }
    if (t == "NFKD") { s.kind = RAD_NORM_NFKD; out.push_back(s); return RAD_OK; }
    if (t == "Lowercase") { s.kind = RAD_NORM_LOWERCASE; out.push_back(s); return RAD_OK; }
    if (t == "Strip") {
        s.kind  = RAD_NORM_STRIP;
        s.flags = (bool_or(n, "strip_left", true) ? 1u : 0u) |
                  (bool_or(n, "strip_right", true) ? 2u : 0u);
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Precompiled") {
        /* SentencePiece's precompiled character map (T5's), kept base64 as the file carries it
         * and decoded at load (tokenizer_grapheme.h). */
        s.kind = RAD_NORM_PRECOMPILED;
        s.arg0 = str_or(n, "precompiled_charsmap", "");
        PrecompiledMap pm;
        std::string why;
        if (s.arg0.empty() || pm.load(s.arg0, &why) < 0) {
            RAD_ERR("%s: Precompiled normalizer: %s", origin,
                    s.arg0.empty() ? "no precompiled_charsmap" : why.c_str());
            return RAD_E_FORMAT;
        }
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Prepend") {
        s.kind = RAD_NORM_PREPEND;
        s.arg0 = str_or(n, "prepend", "");
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Replace") {
        bool is_regex = false;
        if (!read_pattern(n, &s.arg0, &is_regex)) {
            RAD_ERR("%s: Replace normalizer has no readable pattern", origin);
            return RAD_E_FORMAT;
        }
        s.kind  = RAD_NORM_REPLACE;
        s.flags = is_regex ? 1u : 0u;
        s.arg1  = str_or(n, "content", "");
        if (is_regex) RAD_TRY(check_regex(s.arg0, "Replace", origin));
        out.push_back(s);
        return RAD_OK;
    }
    /* Anything else is refused by name: ignoring a normaliser silently changes token boundaries
     * (spec §12). */
    RAD_ERR("%s: normalizer '%s' has no RadVocabStep. Refusing rather than approximating it; "
            "add a RAD_NORM_* for it if a model needs it.", origin, t.c_str());
    return RAD_E_UNSUPPORTED;
}

int read_pretok(const json& n, std::vector<VocabStep>& out, const char* origin) {
    if (n.is_null()) return RAD_OK;
    const std::string t = str_or(n, "type", "");
    VocabStep s;
    if (t == "Sequence") {
        const json* seq = get(n, "pretokenizers");
        if (!seq) seq = get(n, "pre_tokenizers");
        if (!seq || !seq->is_array()) return RAD_OK;
        for (const json& c : *seq) RAD_TRY(read_pretok(c, out, origin));
        return RAD_OK;
    }
    if (t == "ByteLevel") {
        s.kind  = RAD_PRE_BYTELEVEL;
        s.flags = (bool_or(n, "add_prefix_space", false) ? 1u : 0u) |
                  (bool_or(n, "use_regex", true) ? 2u : 0u);
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Split") {
        bool is_regex = false;
        if (!read_pattern(n, &s.arg0, &is_regex)) {
            RAD_ERR("%s: Split pre-tokenizer has no readable pattern", origin);
            return RAD_E_FORMAT;
        }
        if (!is_regex) {
            /* A literal delimiter becomes a regex with every metacharacter escaped, so the
             * runtime has one kind of Split. */
            std::string q;
            for (char c : s.arg0) {
                if (std::strchr("\\^$.|?*+()[]{}", c)) q += '\\';
                q += c;
            }
            s.arg0.swap(q);
        }
        s.kind  = RAD_PRE_SPLIT;
        s.flags = bool_or(n, "invert", false) ? 1u : 0u;
        const std::string bh = str_or(n, "behavior", "Isolated");
        s.iarg  = behavior_from_name(bh);
        if (s.iarg < 0) return refuse_value(origin, "Split behavior", bh);
        RAD_TRY(check_regex(s.arg0, "Split", origin));
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Metaspace") {
        s.kind  = RAD_PRE_METASPACE;
        s.arg0  = str_or(n, "replacement", "\xe2\x96\x81");
        const std::string ps = str_or(n, "prepend_scheme", "always");
        s.iarg  = prepend_scheme_from_name(ps);
        if (s.iarg < 0) return refuse_value(origin, "Metaspace prepend_scheme", ps);
        s.flags = bool_or(n, "split", true) ? 1u : 0u;
        /* The pre-0.14 spelling: add_prefix_space instead of prepend_scheme. */
        if (!get(n, "prepend_scheme")) s.iarg = bool_or(n, "add_prefix_space", true) ? 2 : 0;
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Whitespace" || t == "WhitespaceSplit" || t == "BertPreTokenizer") {
        s.kind  = RAD_PRE_WHITESPACE;
        s.flags = t == "Whitespace" ? kPreWhitespace
                : t == "WhitespaceSplit" ? kPreWhitespaceSplit : kPreBert;
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Punctuation") {
        s.kind = RAD_PRE_PUNCTUATION;
        const std::string bh = str_or(n, "behavior", "Isolated");
        s.iarg = behavior_from_name(bh);
        if (s.iarg < 0) return refuse_value(origin, "Punctuation behavior", bh);
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Digits") {
        s.kind  = RAD_PRE_DIGITS;
        s.flags = bool_or(n, "individual_digits", false) ? 1u : 0u;
        out.push_back(s);
        return RAD_OK;
    }
    RAD_ERR("%s: pre_tokenizer '%s' is not supported; no RAD_PRE_* step implements it",
            origin, t.c_str());
    return RAD_E_UNSUPPORTED;
}

int read_decoder(const json& n, std::vector<VocabStep>& out, const char* origin) {
    if (n.is_null()) return RAD_OK;
    const std::string t = str_or(n, "type", "");
    VocabStep s;
    if (t == "Sequence") {
        const json* seq = get(n, "decoders");
        if (!seq || !seq->is_array()) return RAD_OK;
        for (const json& c : *seq) RAD_TRY(read_decoder(c, out, origin));
        return RAD_OK;
    }
    if (t == "ByteLevel")    { s.kind = RAD_DEC_BYTELEVEL; out.push_back(s); return RAD_OK; }
    if (t == "Fuse")         { s.kind = RAD_DEC_FUSE; out.push_back(s); return RAD_OK; }
    if (t == "ByteFallback") { s.kind = RAD_DEC_BYTE_FALLBACK; out.push_back(s); return RAD_OK; }
    if (t == "Metaspace") {
        s.kind = RAD_DEC_METASPACE;
        s.arg0 = str_or(n, "replacement", "\xe2\x96\x81");
        const std::string ps = str_or(n, "prepend_scheme", "always");
        s.iarg = prepend_scheme_from_name(ps);
        if (s.iarg < 0) return refuse_value(origin, "Metaspace prepend_scheme", ps);
        if (!get(n, "prepend_scheme")) s.iarg = bool_or(n, "add_prefix_space", true) ? 2 : 0;
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Replace") {
        bool is_regex = false;
        if (!read_pattern(n, &s.arg0, &is_regex)) {
            RAD_ERR("%s: Replace decoder has no readable pattern", origin);
            return RAD_E_FORMAT;
        }
        if (is_regex) {
            RAD_ERR("%s: a regex Replace decoder is not implemented; the decoder runs per token "
                    "on the streaming path and a regex there is a per-token allocation", origin);
            return RAD_E_UNSUPPORTED;
        }
        s.kind = RAD_DEC_REPLACE;
        s.arg1 = str_or(n, "content", "");
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Strip") {
        s.kind = RAD_DEC_STRIP;
        s.arg0 = str_or(n, "content", " ");
        const json* a = get(n, "start");
        const json* b = get(n, "stop");
        const int64_t st = a && a->is_number() ? a->get<int64_t>() : 0;
        const int64_t sp = b && b->is_number() ? b->get<int64_t>() : 0;
        s.iarg = (sp << 32) | (st & 0xFFFFFFFF);
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "WordPiece") {
        /* A Replace of the continuation prefix that also joins words with a space, as
         * HuggingFace's WordPiece decoder does (kDecWordPiece). `cleanup` defaults to true. */
        VocabStep r;
        r.kind  = RAD_DEC_REPLACE;
        r.arg0  = str_or(n, "prefix", "##");
        r.flags = kDecWordPiece | (bool_or(n, "cleanup", true) ? kDecWordPieceCleanup : 0u);
        out.push_back(r);
        return RAD_OK;
    }
    if (t == "BPEDecoder") {
        /* The end-of-word suffix becomes a space, except on the last token (kDecBpeSuffix). */
        VocabStep r;
        r.kind  = RAD_DEC_REPLACE;
        r.arg0  = str_or(n, "suffix", "</w>");
        r.arg1  = " ";
        r.flags = kDecBpeSuffix;
        out.push_back(r);
        return RAD_OK;
    }
    RAD_ERR("%s: decoder '%s' has no RadVocabStep", origin, t.c_str());
    return RAD_E_UNSUPPORTED;
}

/* --- the model section --------------------------------------------------------------------- */

int read_model(const json& m, VocabBuild& out, const char* origin, uint64_t id_limit) {
    const std::string type = str_or(m, "type", "");
    const json* vocab = get(m, "vocab");
    if (!vocab) {
        RAD_ERR("%s: model has no vocab", origin);
        return RAD_E_FORMAT;
    }

    /* {text: id}, the BPE and WordPiece shape. JSON order need not match id order, so each
     * entry is placed by id; every id is checked before the table is sized from them. */
    auto read_text_to_id = [&](const char* kind) -> int {
        if (!vocab->is_object()) {
            RAD_ERR("%s: %s vocab is not an object of {text: id}", origin, kind);
            return RAD_E_FORMAT;
        }
        int32_t max_id = -1;
        for (auto it = vocab->begin(); it != vocab->end(); ++it) {
            int32_t id = 0;
            if (!read_id(it.value(), id_limit, &id)) {
                RAD_ERR("%s: %s vocab entry '%s' has id %.64s, which is not a token id",
                        origin, kind, it.key().c_str(), it.value().dump().c_str());
                return RAD_E_FORMAT;
            }
            max_id = std::max(max_id, id);
        }
        /* Two texts on one id: the byte-greatest one holds it, whatever the file's order, as when
         * the vocab was read through a key-sorted map. */
        out.tokens.assign((size_t)max_id + 1, std::string());
        std::vector<bool> taken((size_t)max_id + 1, false);
        for (auto it = vocab->begin(); it != vocab->end(); ++it) {
            int32_t id = 0;
            read_id(it.value(), id_limit, &id);
            std::string& slot = out.tokens[(size_t)id];
            if (!taken[(size_t)id] || slot < it.key()) slot = it.key();
            taken[(size_t)id] = true;
        }
        out.types.assign(out.tokens.size(), (uint8_t)RAD_TT_NORMAL);
        return RAD_OK;
    };

    if (type == "BPE") {
        out.kind = RAD_TOK_BPE;
        out.ignore_merges = bool_or(m, "ignore_merges", false);
        /* A continuation prefix changes what every merge produces, and dropout makes encoding
         * random; the encoder implements neither. */
        const std::string prefix = str_or(m, "continuing_subword_prefix", "");
        if (!prefix.empty()) return refuse_value(origin, "BPE continuing_subword_prefix", prefix);
        if (const json* d = get(m, "dropout"); d && d->is_number() && d->get<double>() > 0.0)
            return refuse_value(origin, "BPE dropout", d->dump());
        RAD_TRY(read_text_to_id("BPE"));
    } else if (type == "Unigram") {
        out.kind = RAD_TOK_UNIGRAM;
        if (!vocab->is_array()) {
            RAD_ERR("%s: Unigram vocab is not an array", origin);
            return RAD_E_FORMAT;
        }
        out.tokens.reserve(vocab->size());
        out.scores.reserve(vocab->size());
        for (const json& e : *vocab) {
            if (!e.is_array() || e.size() != 2 || !e[0].is_string() || !e[1].is_number()) {
                RAD_ERR("%s: Unigram vocab entry is not [text, score]", origin);
                return RAD_E_FORMAT;
            }
            out.tokens.push_back(e[0].get<std::string>());
            out.scores.push_back(e[1].get<float>());
        }
        out.types.assign(out.tokens.size(), (uint8_t)RAD_TT_NORMAL);
        /* The encoder emits this id for uncovered text, so it must name a token. */
        const json* u = get(m, "unk_id");
        if (u && !read_id(*u, out.tokens.size(), &out.unk)) {
            RAD_ERR("%s: Unigram unk_id %.64s is not an id in its %zu-token vocab", origin,
                    u->dump().c_str(), out.tokens.size());
            return RAD_E_FORMAT;
        }
    } else if (type == "WordPiece") {
        out.kind = RAD_TOK_WORDPIECE;
        /* The encoder spells continuations "##" and takes words of up to 100 characters,
         * HuggingFace's defaults. */
        const std::string prefix = str_or(m, "continuing_subword_prefix", "##");
        if (prefix != "##") return refuse_value(origin, "WordPiece continuing_subword_prefix", prefix);
        if (const json* mc = get(m, "max_input_chars_per_word");
            mc && !(mc->is_number_integer() && mc->get<int64_t>() == 100))
            return refuse_value(origin, "WordPiece max_input_chars_per_word", mc->dump());
        RAD_TRY(read_text_to_id("WordPiece"));
    } else {
        RAD_ERR("%s: model type '%s' is not implemented (BPE, Unigram and WordPiece are)",
                origin, type.c_str());
        return RAD_E_UNSUPPORTED;
    }

    /* The end-of-word suffix (CLIP's "</w>") is a model property in tokenizer.json but a step
     * here, as byte fallback is below: it changes the symbols a word starts as. */
    if (type == "BPE") {
        const std::string eow = str_or(m, "end_of_word_suffix", "");
        if (!eow.empty()) {
            VocabStep s;
            s.kind = RAD_PRE_END_OF_WORD;
            s.arg0 = eow;
            out.steps.push_back(s);
        }
    }

    /* Byte fallback is a model property in tokenizer.json but a step here: it changes what the
     * encoder does on a miss. */
    if (bool_or(m, "byte_fallback", false)) {
        VocabStep s;
        s.kind = RAD_PRE_BYTE_FALLBACK;
        out.steps.push_back(s);
    }

    if (type != "BPE") return RAD_OK;

    /* Merges come as ["a b", ...] or, since a token can contain a space, [["a","b"], ...]. */
    const json* merges = get(m, "merges");
    if (!merges || !merges->is_array()) return RAD_OK;

    MergeIds ids(out.tokens);
    out.merges.reserve(merges->size());
    for (const json& e : *merges) {
        if (e.is_array() && e.size() == 2 && e[0].is_string() && e[1].is_string())
            ids.add(e[0].get<std::string>(), e[1].get<std::string>(), out);
        else if (e.is_string())
            ids.add_spaced(e.get<std::string>(), out);
        else
            ++ids.dropped;
    }
    /* A dropped merge moves token boundaries; only a warning, as MergeIds says why. */
    if (ids.dropped)
        RAD_WARN("%s: %zu of %zu merges reference text that is not in the vocab and were "
                 "dropped; token boundaries may differ from the reference tokeniser",
                 origin, ids.dropped, merges->size());
    return RAD_OK;
}

/* Added tokens override the model vocab and may extend past it. `special` makes a token control
 * (isolated only when parse_special is on; see Tokenizer::encode) rather than user-defined.
 *
 * The encoder matches an added token against the raw text, exactly as written. HuggingFace's
 * AddedToken flags each change that: `single_word` matches only between word boundaries,
 * `normalized` matches against the normalised text (the same thing when there is no normaliser),
 * and `lstrip` / `rstrip` take the whitespace beside a match into it. A normalised token becomes a
 * RAD_PRE_ADDED_NORMALIZED step, placed first among the pre-tokenisers (so `<|StartOfText|>` under
 * a Lowercase is CLIP's `<|startoftext|>`); one whose spelling the normaliser itself changes would
 * be matched by that changed spelling, which is refused. Most files spell out every
 * flag, so a flag is refused only at a value that changes the result. lstrip and rstrip would
 * also refuse released models (Phi-3's chat markers carry rstrip), and the container has no field
 * to carry them, so they are warned about instead. */
int read_added(const json& arr, VocabBuild& out, const char* origin, uint64_t id_limit,
               const std::vector<VocabStep>& norm) {
    const bool has_normalizer = !norm.empty();
    std::vector<VocabStep> matched_normalized;
    if (!arr.is_array()) return RAD_OK;
    std::string stripped;
    for (const json& e : arr) {
        const json* idv = get(e, "id");
        const json* cv  = get(e, "content");
        if (!idv || !cv) continue;
        int32_t sid = 0;
        if (!read_id(*idv, id_limit, &sid) || !cv->is_string()) {
            RAD_ERR("%s: added token %.64s is not {id, content} with an id and a string",
                    origin, e.dump().c_str());
            return RAD_E_FORMAT;
        }
        const size_t id = (size_t)sid;
        if (id >= out.tokens.size()) {
            out.tokens.resize(id + 1);
            out.types.resize(id + 1, (uint8_t)RAD_TT_NORMAL);
            if (out.kind == RAD_TOK_UNIGRAM) out.scores.resize(id + 1, 0.0f);
        }
        const std::string content = cv->get<std::string>();
        const bool special = bool_or(e, "special", false);
        if (bool_or(e, "single_word", false))
            return refuse_value(origin, "added token single_word=true on", content);
        if (has_normalizer && bool_or(e, "normalized", false)) {
            std::string n = content;
            if (vocab_normalize(norm, &n) != RAD_OK || n != content)
                return refuse_value(origin, "added token normalized=true (under a normalizer that "
                                            "changes its spelling) on", content);
            if (!content.empty()) {
                VocabStep st;
                st.kind = RAD_PRE_ADDED_NORMALIZED;
                st.arg0 = content;
                st.iarg = (int64_t)id;
                matched_normalized.push_back(st);
            }
        }
        if (bool_or(e, "lstrip", false) || bool_or(e, "rstrip", false))
            stripped += (stripped.empty() ? "" : " ") + content;
        out.tokens[id] = content;
        out.types[id]  = special ? (uint8_t)RAD_TT_CONTROL : (uint8_t)RAD_TT_USER_DEFINED;
    }
    /* Ahead of the other pre-tokenisers, after the normalisers. */
    out.steps.insert(out.steps.begin() + (std::ptrdiff_t)norm.size(), matched_normalized.begin(),
                     matched_normalized.end());
    if (!stripped.empty())
        RAD_WARN("%s: added tokens %s take the whitespace beside them in HuggingFace (lstrip / "
                 "rstrip); this tokenizer keeps it as text, so prompts with whitespace next to "
                 "them tokenise differently", origin, stripped.c_str());
    return RAD_OK;
}

/* tokenizer_config.json names the special tokens; tokenizer.json does not say which is BOS.
 * The sibling is optional: when it is absent the ids stay -1 and rad-convert can fill them from
 * elsewhere. */
void read_config_json(const std::string& dir, VocabBuild& out) {
    const std::string path = dir + "tokenizer_config.json";
    std::ifstream f(path);
    if (!f) {
        RAD_INFO("tokenizer: no %s; bos/eos ids and the chat template are unset", path.c_str());
        return;
    }
    json c;
    try { f >> c; }
    catch (const std::exception& e) {
        RAD_WARN("tokenizer: %s did not parse (%s); ids left unset", path.c_str(), e.what());
        return;
    }
    std::unordered_map<std::string, int32_t> id_of;
    for (size_t i = 0; i < out.tokens.size(); ++i) id_of.emplace(out.tokens[i], (int32_t)i);

    auto token_id = [&](const char* key) -> int32_t {
        const json* v = get(c, key);
        if (!v) return -1;
        std::string text;
        const json* t = v->is_string() ? v : get(*v, "content");
        if (!t || !t->is_string()) return -1;
        text = t->get<std::string>();
        auto it = id_of.find(text);
        return it == id_of.end() ? -1 : it->second;
    };
    /* An id the post-processor already gave (interpret_tokenizer_json) stays when the config
     * names no such token. */
    const int32_t bos = token_id("bos_token"), eos = token_id("eos_token");
    if (bos >= 0 || out.bos < 0) out.bos = bos;
    if (eos >= 0 || out.eos < 0) out.eos = eos;
    out.pad = token_id("pad_token");
    out.unk = out.unk >= 0 ? out.unk : token_id("unk_token");
    out.sep = token_id("sep_token");
    out.eot = out.eos;
    /* The default is what tokenizer.json's post-processor already said (interpret_tokenizer_json). */
    out.add_bos = bool_or(c, "add_bos_token", out.add_bos);
    out.add_eos = bool_or(c, "add_eos_token", out.add_eos);

    const json* ct = get(c, "chat_template");
    if (ct && ct->is_string()) out.chat_template = ct->get<std::string>();
    else if (ct && ct->is_array() && !ct->empty()) {
        /* The multi-template form, a list of {name, template}: "default" wins, else the first,
         * as renderers do. */
        for (const json& e : *ct) {
            if (str_or(e, "name", "") == "default") { out.chat_template = str_or(e, "template", ""); break; }
        }
        if (out.chat_template.empty()) out.chat_template = str_or((*ct)[0], "template", "");
    }
}


/* Recent transformers versions write the template to a standalone chat_template.jinja and omit
 * the config key; without it, /v1/chat/completions answers 501. The file wins over the config
 * key, as in transformers. */
void read_chat_template_file(const std::string& dir, VocabBuild& out) {
    const std::string path = dir + "chat_template.jinja";
    std::ifstream f(path);
    if (!f) return;
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string t = ss.str();
    while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
    if (t.empty()) return;
    RAD_INFO("tokenizer: chat template from %s (%zu bytes)%s", path.c_str(), t.size(),
             out.chat_template.empty() ? "" : ", overriding the tokenizer_config.json key");
    out.chat_template = std::move(t);
}

void read_config(const std::string& dir, VocabBuild& out) {
    read_config_json(dir, out);
    read_chat_template_file(dir, out);
}

}  /* namespace */

int read_chat_template_path(const char* path, VocabBuild& out) {
    if (!path) return RAD_E_INVAL;
    std::ifstream f(path);
    if (!f) {
        RAD_ERR("tokenizer: cannot read chat template %s", path);
        return RAD_E_IO;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string t = ss.str();
    while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
    if (t.empty()) {
        RAD_ERR("tokenizer: chat template %s is empty", path);
        return RAD_E_IO;
    }
    RAD_INFO("tokenizer: chat template from %s (%zu bytes), overriding the checkpoint's",
             path, t.size());
    out.chat_template = std::move(t);
    return RAD_OK;
}

/* ================================================================== parse_tokenizer_json */

namespace {

int interpret_tokenizer_json(const json& j, uint64_t id_limit, const char* origin,
                             VocabBuild& out) {
    const json* model = get(j, "model");
    if (!model) {
        RAD_ERR("%s: no \"model\" section", origin);
        return RAD_E_FORMAT;
    }
    RAD_TRY(read_model(*model, out, origin, id_limit));

    /* One list, as RadVocabHeader holds it: normaliser, pre-tokeniser, then decoder steps, each
     * in file order; the kind ranges keep them separable. A ByteFallback step pushed by
     * read_model belongs with the pre-tokenisers and is moved there. */
    std::vector<VocabStep> pre_from_model;
    pre_from_model.swap(out.steps);

    std::vector<VocabStep> norm, pre, dec;
    if (const json* n = get(j, "normalizer"))     RAD_TRY(read_normalizer(*n, norm, origin));
    if (const json* p = get(j, "pre_tokenizer"))  RAD_TRY(read_pretok(*p, pre, origin));
    if (const json* d = get(j, "decoder"))        RAD_TRY(read_decoder(*d, dec, origin));
    for (VocabStep& s : pre_from_model) pre.push_back(std::move(s));

    out.steps.clear();
    out.steps.insert(out.steps.end(), norm.begin(), norm.end());
    out.steps.insert(out.steps.end(), pre.begin(),  pre.end());
    out.steps.insert(out.steps.end(), dec.begin(),  dec.end());

    if (const json* a = get(j, "added_tokens"))
        RAD_TRY(read_added(*a, out, origin, id_limit, norm));

    /* The model's unknown token, by text. The encoder emits it for text no token covers, which
     * HuggingFace does too; tokenizer_config.json's unk_token, read later, agrees or is absent. */
    if (out.unk < 0) {
        const std::string unk = str_or(*model, "unk_token", "");
        for (size_t i = 0; !unk.empty() && i < out.tokens.size(); ++i)
            if (out.tokens[i] == unk) { out.unk = (int32_t)i; break; }
    }

    /* A Strip after Fuse strips the decoded text as a whole. Its start can be stripped as text
     * streams; its end cannot be taken back once streamed, and two of them would interleave. */
    {
        bool fused = false, stripped = false;
        for (const VocabStep& s : dec) {
            if (s.kind == RAD_DEC_FUSE) fused = true;
            if (s.kind != RAD_DEC_STRIP || !fused) continue;
            if ((s.iarg >> 32) != 0 || stripped) {
                RAD_ERR("%s: a Strip decoder after Fuse that strips the end of the text, or a "
                        "second one, is not supported", origin);
                return RAD_E_UNSUPPORTED;
            }
            stripped = true;
        }
    }

    /* What `add_special_tokens` adds, when the post-processor says it. HuggingFace's encode runs
     * the post-processor, so a TemplateProcessing single template `<bos> $A <eos>` is the model's
     * add-BOS/EOS rule; Gemma states it only there, and an embedding model pools over that <eos>.
     * Only a special token before and/or after the sequence is read. The config's
     * add_bos_token / add_eos_token, read later, win where present. */
    if (const json* pp = get(j, "post_processor")) {
        const std::string ppt = str_or(*pp, "type", "");
        /* RobertaProcessing and BertProcessing frame a sequence as `cls $A sep`; they name the
         * two tokens, which stand for BOS and EOS where the config names none (CLIP's). */
        if (ppt == "RobertaProcessing" || ppt == "BertProcessing") {
            auto framing = [&](const char* k) -> int32_t {
                const json* v = get(*pp, k);
                int32_t id = -1;
                if (!v || !v->is_array() || v->size() != 2 || !read_id((*v)[1], out.tokens.size(), &id))
                    return -1;
                return id;
            };
            const int32_t cls = framing("cls"), sep = framing("sep");
            if (cls < 0 || sep < 0) {
                RAD_ERR("%s: %s names no [token, id] cls and sep in the vocabulary", origin,
                        ppt.c_str());
                return RAD_E_FORMAT;
            }
            out.bos = cls;
            out.eos = sep;
            out.add_bos = out.add_eos = true;
        }
        if (ppt == "TemplateProcessing") {
            const json* single = get(*pp, "single");
            if (single && single->is_array() && !single->empty()) {
                auto special = [](const json& e) -> bool { return e.is_object() && e.contains("SpecialToken"); };
                const bool seq_in_middle = single->size() >= 2;
                if (special(single->front()) && seq_in_middle) out.add_bos = true;
                if (special(single->back()) && seq_in_middle) out.add_eos = true;
            }
        }
    }

    if (pre.empty() && out.kind == RAD_TOK_BPE) {
        RAD_ERR("%s: a BPE model with no pre_tokenizer would tokenise the whole prompt as one "
                "word. Refusing.", origin);
        return RAD_E_FORMAT;
    }
    return RAD_OK;
}

}  /* namespace */

int parse_tokenizer_json_buf(std::string_view text, const char* origin, VocabBuild& out) {
    json j;
    try {
        j = json::parse(text);
    } catch (const std::exception& e) {
        RAD_ERR("%s: not valid JSON: %s", origin, e.what());
        return RAD_E_FORMAT;
    }
    out = VocabBuild{};
    out.source = origin;

    /* No id may exceed the file size in bytes. Ids are dense, so a valid file never needs a
     * larger one, and the bound keeps the token table proportional to the input. */
    const uint64_t id_limit = text.size();

    /* Json throws on the first get<>() of a value with the wrong JSON type. The reader checks
     * the shapes that decide where things are written; this turns any other mismatch into a
     * format error instead of std::terminate. */
    try {
        return interpret_tokenizer_json(j, id_limit, origin, out);
    } catch (const json::exception& e) {
        RAD_ERR("%s: a value has the wrong type: %s", origin, e.what());
        return RAD_E_FORMAT;
    } catch (const std::bad_alloc&) {
        RAD_ERR("%s: out of memory reading the vocabulary", origin);
        return RAD_E_NOMEM;
    }
}

int parse_tokenizer_json(const char* path, VocabBuild& out) {
    if (!path) return RAD_E_INVAL;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        RAD_ERR("tokenizer: cannot open %s", path);
        return RAD_E_IO;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    RAD_TRY(parse_tokenizer_json_buf(text, path, out));

    /* The sibling config carries the ids and the chat template. */
    const std::string p(path);
    const size_t slash = p.find_last_of('/');
    read_config(slash == std::string::npos ? std::string() : p.substr(0, slash + 1), out);

    RAD_INFO("tokenizer: %s -> %zu tokens, %zu merges, %zu chain steps",
             path, out.tokens.size(), out.merges.size(), out.steps.size());
    for (const VocabStep& s : out.steps) RAD_INFO("tokenizer:   %s", step_describe(s).c_str());
    return RAD_OK;
}

/* ================================================================== vocab_from_gguf */

namespace {

/* `tokenizer.ggml.pre` -> the pre-tokeniser regex list.
 *
 * llama.cpp's adapted patterns, copied verbatim from llm_tokenizer_bpe's constructor
 * (06938ac12). They are copied, not derived, because a GGUF tokenises the way llama.cpp runs
 * it: gpt-4o's `((?=[\p{L}])([^a-z]))*` is not o200k's `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*`. They
 * go through the same compiler as a tokenizer.json pattern.
 *
 * Exception: kimi-k2. llama.cpp's `\p{Han}+` is only the key of a hand-written splitter that
 * approximates Kimi-K2's pattern; as a regex it would split only Han runs. The entry here is the
 * real pattern from the model's tokenization_kimi.py.
 *
 * The list is not exhaustive. A GGUF whose `pre` is missing is refused, with the advice to
 * convert from tokenizer.json. */
struct PreEntry { const char* name; const char* regex[4]; bool byte_encode; };

/* The patterns several entries share. */
constexpr const char* kGpt2 =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
constexpr const char* kLlama3 =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
constexpr const char* kQwen2 =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
constexpr const char* kGpt4o =
    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
constexpr const char* kLines = "[^\\n]+|[\\n]+";

const PreEntry k_pre[] = {
    { "default", { "[\\p{P}\\$\\+<=>\\^~\\|]+", kGpt2, "\\p{N}+", "[0-9][0-9][0-9]" }, true },
    { "llama3", { kLlama3 }, true },
    { "llama-bpe", { kLlama3 }, true },
    { "llama-v3", { kLlama3 }, true },
    { "falcon3", { kLlama3 }, true },
    { "pixtral", { kLlama3 }, true },
    { "lfm2", { kLlama3 }, true },
    { "dbrx", { kLlama3 }, true },
    { "smaug-bpe", { kLlama3 }, true },
    { "glm4", { kLlama3 }, true },
    { "chatglm-bpe", { kLlama3 }, true },
    { "gpt-2", { kGpt2 }, true },
    { "phi-2", { kGpt2 }, true },
    { "mpt", { kGpt2 }, true },
    { "olmo", { kGpt2 }, true },
    { "jais", { kGpt2 }, true },
    { "trillion", { kGpt2 }, true },
    { "granite-docling", { kGpt2 }, true },
    { "starcoder", { "\\p{N}", kGpt2 }, true },
    { "refact", { "\\p{N}", kGpt2 }, true },
    { "command-r", { "\\p{N}", kGpt2 }, true },
    { "smollm", { "\\p{N}", kGpt2 }, true },
    { "codeshell", { "\\p{N}", kGpt2 }, true },
    { "exaone", { "\\p{N}", kGpt2 }, true },
    { "minerva-7b", { "\\p{N}", kGpt2 }, true },
    { "qwen2", { kQwen2 }, true },
    { "stablelm2", { kQwen2 }, true },
    { "deepseek-r1-qwen", { kQwen2 }, true },
    { "hunyuan", { kQwen2 }, true },
    { "solar-open", { kQwen2 }, true },
    { "grok-2", { kQwen2 }, true },
    { "qwen35", { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "gpt-4o", { kGpt4o }, true },
    { "llama4", { kGpt4o }, true },
    { "tekken", { "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "seed-coder", { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "kimi-k2", { "[\\p{Han}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "whitespace", { "\\S+" }, false },
    { "gemma4", { kLines }, false },
    { "sarvam-moe", { kLines }, false },
};

}  /* namespace */

void vocab_textless(VocabBuild& out) {
    out = VocabBuild{};
    out.kind = RAD_TOK_BPE;
    out.tokens = { "<|no-text|>" };
    out.types = { RAD_TT_CONTROL };
    out.source = "none: the model reads no text";
}

int vocab_from_gguf(const GgufFile& g, VocabBuild& out) {
    out = VocabBuild{};
    out.source = g.path();

    std::string model;
    if (!g.get_str("tokenizer.ggml.model", &model)) {
        RAD_ERR("gguf: %s has no tokenizer.ggml.model", g.path().c_str());
        return RAD_E_FORMAT;
    }
    if (!g.get_strs("tokenizer.ggml.tokens", &out.tokens) || out.tokens.empty()) {
        RAD_ERR("gguf: %s has no tokenizer.ggml.tokens", g.path().c_str());
        return RAD_E_FORMAT;
    }

    std::vector<int64_t> types;
    if (g.get_i64s("tokenizer.ggml.token_type", &types) && types.size() == out.tokens.size()) {
        out.types.resize(types.size());
        for (size_t i = 0; i < types.size(); ++i) out.types[i] = (uint8_t)types[i];
    } else {
        out.types.assign(out.tokens.size(), (uint8_t)RAD_TT_NORMAL);
    }
    g.get_f32s("tokenizer.ggml.scores", &out.scores);

    auto id = [&](const char* k, int32_t* dst) {
        int64_t v = 0;
        if (g.get_i64(k, &v)) *dst = (int32_t)v;
    };
    id("tokenizer.ggml.bos_token_id", &out.bos);
    id("tokenizer.ggml.eos_token_id", &out.eos);
    id("tokenizer.ggml.eot_token_id", &out.eot);
    id("tokenizer.ggml.padding_token_id", &out.pad);
    id("tokenizer.ggml.unknown_token_id", &out.unk);
    id("tokenizer.ggml.seperator_token_id", &out.sep);
    if (out.eot < 0) out.eot = out.eos;
    int64_t b = 0;
    if (g.get_i64("tokenizer.ggml.add_bos_token", &b)) out.add_bos = b != 0;
    if (g.get_i64("tokenizer.ggml.add_eos_token", &b)) out.add_eos = b != 0;
    g.get_str("tokenizer.chat_template", &out.chat_template);

    if (model == "gpt2") {
        out.kind = RAD_TOK_BPE;
        std::string pre;
        if (!g.get_str("tokenizer.ggml.pre", &pre)) pre = "default";

        const PreEntry* e = nullptr;
        for (const PreEntry& p : k_pre) if (pre == p.name) { e = &p; break; }
        if (!e) {
            /* Refused per spec §12; llama.cpp falls back to `default` and splits wrongly. */
            RAD_ERR("gguf: tokenizer.ggml.pre = '%s' is not in this build's table, and guessing "
                    "would give a silently different split. Convert from the model's "
                    "tokenizer.json instead, which needs no table.", pre.c_str());
            return RAD_E_UNSUPPORTED;
        }
        for (const char* r : e->regex) {
            if (!r) break;
            VocabStep s;
            s.kind = RAD_PRE_SPLIT;
            s.arg0 = r;
            s.iarg = 1;   /* Isolated */
            out.steps.push_back(s);
        }
        if (e->byte_encode) {
            VocabStep s;
            s.kind  = RAD_PRE_BYTELEVEL;
            s.flags = 0;   /* the split above already carries the regex */
            out.steps.push_back(s);
            VocabStep d;
            d.kind = RAD_DEC_BYTELEVEL;
            out.steps.push_back(d);
        } else {
            /* SPM-shaped BPE: the vocab is raw UTF-8 with U+2581 for space. */
            out.steps.push_back(metaspace(RAD_PRE_METASPACE));
            out.steps.push_back(metaspace(RAD_DEC_METASPACE));
        }

        std::vector<std::string> merges;
        g.get_strs("tokenizer.ggml.merges", &merges);
        MergeIds ids(out.tokens);
        out.merges.reserve(merges.size());
        for (const std::string& m : merges) ids.add_spaced(m, out);
        if (ids.dropped)
            RAD_WARN("gguf: %zu of %zu merges dropped (one side of the merge is not a token); "
                     "token boundaries may differ from the reference tokeniser",
                     ids.dropped, merges.size());
    } else if (model == "llama") {
        out.kind = RAD_TOK_BPE;
        /* SPM: whitespace escaped to U+2581, byte fallback for the rest, expressed as steps so
         * the runtime has no per-model branch. */
        out.steps.push_back(metaspace(RAD_PRE_METASPACE));
        VocabStep bf;
        bf.kind = RAD_PRE_BYTE_FALLBACK;
        out.steps.push_back(bf);
        out.steps.push_back(metaspace(RAD_DEC_METASPACE));
        RAD_WARN("gguf: '%s' is a SentencePiece vocab, tokenised as BPE over its pieces rather "
                 "than by the SPM bigram algorithm; convert from tokenizer.json for an exact match",
                 model.c_str());
    } else if (model == "t5") {
        out.kind = RAD_TOK_UNIGRAM;
        out.steps.push_back(metaspace(RAD_PRE_METASPACE));
        out.steps.push_back(metaspace(RAD_DEC_METASPACE));
    } else if (model == "bert") {
        out.kind = RAD_TOK_WORDPIECE;
        VocabStep n;
        n.kind = RAD_NORM_LOWERCASE;
        out.steps.push_back(n);
        /* BertPreTokenizer and the WordPiece decoder, as a BERT tokenizer.json declares them. */
        VocabStep w;
        w.kind  = RAD_PRE_WHITESPACE;
        w.flags = kPreBert;
        out.steps.push_back(w);
        VocabStep d;
        d.kind  = RAD_DEC_REPLACE;
        d.arg0  = "##";
        d.flags = kDecWordPiece | kDecWordPieceCleanup;
        out.steps.push_back(d);
    } else {
        RAD_ERR("gguf: tokenizer.ggml.model = '%s' is not implemented", model.c_str());
        return RAD_E_UNSUPPORTED;
    }

    RAD_INFO("gguf: vocab %zu tokens, %zu merges, %zu chain steps",
             out.tokens.size(), out.merges.size(), out.steps.size());
    for (const VocabStep& s : out.steps) RAD_INFO("gguf:   %s", step_describe(s).c_str());
    return RAD_OK;
}

/* ================================================================== serialise */

namespace {
template <class T> void put(std::vector<uint8_t>& b, const T& v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
void pad_to(std::vector<uint8_t>& b, size_t base, size_t align) {
    while ((base + b.size()) % align) b.push_back(0);
}
}  /* namespace */

int vocab_serialize(const VocabBuild& in, StringSink& strings, uint64_t off,
                    std::vector<uint8_t>& out) {
    if (in.tokens.empty()) return RAD_E_INVAL;
    if (in.tokens.size() > UINT32_MAX || in.merges.size() > UINT32_MAX ||
        in.steps.size() > UINT32_MAX) return RAD_E_INVAL;

    /* Strings go into the file's one blob and only their offsets are stored here (rad_format.h),
     * so the section is fixed-stride and mmap-walkable. */
    std::vector<rad_stroff> tok_off(in.tokens.size());
    for (size_t i = 0; i < in.tokens.size(); ++i) tok_off[i] = strings.intern(in.tokens[i]);

    std::vector<RadVocabStep> steps(in.steps.size());
    for (size_t i = 0; i < in.steps.size(); ++i) {
        steps[i].kind  = in.steps[i].kind;
        steps[i].flags = in.steps[i].flags;
        steps[i].arg0  = strings.intern(in.steps[i].arg0);
        steps[i].arg1  = strings.intern(in.steps[i].arg1);
        steps[i].iarg  = in.steps[i].iarg;
    }
    const rad_stroff tmpl = strings.intern(in.chat_template);

    out.clear();
    RadVocabHeader h{};
    h.kind     = in.kind;
    h.n_tokens = (uint32_t)in.tokens.size();
    h.n_merges = (uint32_t)in.merges.size();
    h.n_steps  = (uint32_t)in.steps.size();
    h.bos = in.bos; h.eos = in.eos; h.eot = in.eot;
    h.pad_id = in.pad; h.unk = in.unk; h.sep = in.sep;
    h.add_bos = in.add_bos ? 1u : 0u;
    h.add_eos = in.add_eos ? 1u : 0u;
    h.chat_template = tmpl;
    h.flags = in.ignore_merges ? RAD_VOCAB_IGNORE_MERGES : 0;
    out.resize(sizeof(RadVocabHeader), 0);

    /* Each array is aligned to its element type so a mapped reader can use a typed pointer.
     * The largest alignment needed is 8; the container's own rule would only require
     * RAD_ALIGN_SUB. */
    auto place = [&](size_t align, const void* p, size_t bytes) -> uint64_t {
        pad_to(out, (size_t)off, align);
        const uint64_t at = off + out.size();
        const uint8_t* s = static_cast<const uint8_t*>(p);
        out.insert(out.end(), s, s + bytes);
        return at;
    };

    h.tok_text_off = place(8, tok_off.data(), tok_off.size() * sizeof(rad_stroff));
    if (!in.scores.empty())
        h.tok_score_off = place(4, in.scores.data(), in.scores.size() * sizeof(float));
    if (!in.types.empty())
        h.tok_type_off = place(1, in.types.data(), in.types.size());
    if (!in.merges.empty()) {
        std::vector<uint32_t> flat(in.merges.size() * 2);
        for (size_t i = 0; i < in.merges.size(); ++i) {
            flat[i * 2 + 0] = in.merges[i].first;
            flat[i * 2 + 1] = in.merges[i].second;
        }
        h.merge_off = place(4, flat.data(), flat.size() * sizeof(uint32_t));
    }
    if (!steps.empty())
        h.step_off = place(8, steps.data(), steps.size() * sizeof(RadVocabStep));

    std::memcpy(out.data(), &h, sizeof h);
    return RAD_OK;
}

int vocab_deserialize(const void* file_base, uint64_t file_bytes, const RadVocabHeader* h,
                      VocabBuild& out) {
    if (!file_base || !h) return RAD_E_INVAL;
    const uint8_t* base = static_cast<const uint8_t*>(file_base);
    const char* strs = nullptr;
    uint64_t str_bytes = 0;
    {
        /* The string blob's extent is read from the file header at offset 0 rather than passed
         * in, so the caller cannot pass a mismatched one. */
        if (file_bytes < sizeof(RadFileHeader)) return RAD_E_FORMAT;
        const RadFileHeader* fh = reinterpret_cast<const RadFileHeader*>(base);
        if (fh->str_off > file_bytes || fh->str_bytes > file_bytes - fh->str_off)
            return RAD_E_FORMAT;
        strs = reinterpret_cast<const char*>(base + fh->str_off);
        str_bytes = fh->str_bytes;
    }
    auto S = [&](rad_stroff o) -> std::string {
        if (o >= str_bytes) return {};
        const char* p = strs + o;
        const size_t max = (size_t)(str_bytes - o);
        return std::string(p, strnlen(p, max));
    };
    auto in_file = [&](uint64_t o, uint64_t n) { return o <= file_bytes && n <= file_bytes - o; };

    out = VocabBuild{};
    out.kind = h->kind;
    out.bos = h->bos; out.eos = h->eos; out.eot = h->eot;
    out.pad = h->pad_id; out.unk = h->unk; out.sep = h->sep;
    out.add_bos = h->add_bos != 0;
    out.add_eos = h->add_eos != 0;
    out.ignore_merges = (h->flags & RAD_VOCAB_IGNORE_MERGES) != 0;
    out.chat_template = S(h->chat_template);

    if ((h->n_tokens && !h->tok_text_off) ||
        !in_file(h->tok_text_off, (uint64_t)h->n_tokens * sizeof(rad_stroff)))
        return RAD_E_FORMAT;
    const rad_stroff* toff = reinterpret_cast<const rad_stroff*>(base + h->tok_text_off);
    out.tokens.resize(h->n_tokens);
    for (uint32_t i = 0; i < h->n_tokens; ++i) out.tokens[i] = S(toff[i]);

    if (h->tok_score_off) {
        if (!in_file(h->tok_score_off, (uint64_t)h->n_tokens * sizeof(float))) return RAD_E_FORMAT;
        const float* p = reinterpret_cast<const float*>(base + h->tok_score_off);
        out.scores.assign(p, p + h->n_tokens);
    }
    if (h->tok_type_off) {
        if (!in_file(h->tok_type_off, h->n_tokens)) return RAD_E_FORMAT;
        const uint8_t* p = base + h->tok_type_off;
        out.types.assign(p, p + h->n_tokens);
    }
    if (h->merge_off) {
        if (!in_file(h->merge_off, (uint64_t)h->n_merges * 2 * sizeof(uint32_t)))
            return RAD_E_FORMAT;
        const uint32_t* p = reinterpret_cast<const uint32_t*>(base + h->merge_off);
        out.merges.resize(h->n_merges);
        for (uint32_t i = 0; i < h->n_merges; ++i) out.merges[i] = { p[i * 2], p[i * 2 + 1] };
    }
    if (h->step_off) {
        if (!in_file(h->step_off, (uint64_t)h->n_steps * sizeof(RadVocabStep)))
            return RAD_E_FORMAT;
        const RadVocabStep* p = reinterpret_cast<const RadVocabStep*>(base + h->step_off);
        out.steps.resize(h->n_steps);
        for (uint32_t i = 0; i < h->n_steps; ++i) {
            out.steps[i].kind  = p[i].kind;
            out.steps[i].flags = p[i].flags;
            out.steps[i].arg0  = S(p[i].arg0);
            out.steps[i].arg1  = S(p[i].arg1);
            out.steps[i].iarg  = p[i].iarg;
        }
    }
    return RAD_OK;
}

}  /* namespace rad */
