#include "text/tokenizer.h"
#include "text/tokenizer_bpe.h"
#include "util/simd.h"
#include "text/tokenizer_grapheme.h"
#include "text/tokenizer_nfc.h"
#include "text/tokenizer_regex.h"
#include "text/unicode_norm.h"
#include "llama/src/unicode.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <unordered_map>

namespace rad {

/* ================================================================== byte level
 *
 * GPT-2's byte alphabet. Printable ASCII, Latin-1 punctuation and the accented Latin-1 letters
 * map to themselves; the remaining 68 bytes map to U+0100 upward in ascending byte order. A
 * mistake here does not crash: vocab lookups silently miss and prompts grow longer. Reproduced
 * from llama.cpp's unicode_byte_to_utf8_map() (HuggingFace's table), not called, because its
 * inverse throws std::out_of_range on an unmapped string and errors here are values
 * (docs/IMPLEMENTATION.md). */
namespace {

struct ByteLevelAlphabet {
    std::string to_utf8[256];                         /* byte -> its 1-2 byte UTF-8 spelling */
    std::unordered_map<std::string, uint8_t> from;    /* and back */
    uint32_t cpt_of[256];
    int16_t  byte_of_cpt[0x144];                      /* codepoint -> byte, or -1 */

    ByteLevelAlphabet() {
        bool taken[256] = { false };
        for (int16_t& b : byte_of_cpt) b = -1;
        auto give = [&](int b, uint32_t cpt) {
            to_utf8[b] = unicode_cpt_to_utf8(cpt);
            cpt_of[b]  = cpt;
            from[to_utf8[b]] = (uint8_t)b;
            byte_of_cpt[cpt] = (int16_t)b;
            taken[b] = true;
        };
        for (int b = 0x21; b <= 0x7E; ++b) give(b, (uint32_t)b);   /* '!' .. '~'  */
        for (int b = 0xA1; b <= 0xAC; ++b) give(b, (uint32_t)b);   /* U+00A1..AC  */
        for (int b = 0xAE; b <= 0xFF; ++b) give(b, (uint32_t)b);   /* U+00AE..FF  */
        uint32_t n = 0;
        for (int b = 0; b < 256; ++b) if (!taken[b]) give(b, 256 + n++);
    }
};

const ByteLevelAlphabet& alphabet() {
    static const ByteLevelAlphabet a;
    return a;
}

std::string byte_encode(std::string_view raw) {
    const ByteLevelAlphabet& a = alphabet();
    std::string out;
    out.reserve(raw.size() * 2);
    for (unsigned char c : raw) out += a.to_utf8[c];
    return out;
}

/* Inverse. An unmapped codepoint passes through as its own UTF-8, as a user-defined token spelt
 * in real text needs; the mapped set is closed, so the two cannot be confused. */
std::string byte_decode(const std::string& enc) {
    /* Unit by unit, a unit's length given by its lead byte and cut short at the end. A unit is
     * mapped iff it is one ASCII byte or a two-byte sequence (lead C2..DF) below U+0144, so the
     * codepoint table gives the same answer as looking the bytes up in `from`, without building
     * a string per character. Anything else passes through as its own bytes. */
    const ByteLevelAlphabet& a = alphabet();
    const uint8_t* p = (const uint8_t*)enc.data();
    const size_t   n = enc.size();
    std::string out;
    out.reserve(n);
    for (size_t i = 0; i < n; ) {
        const size_t len = std::min(unicode_len_utf8(enc[i]), n - i);
        int b = -1;
        if (len == 1 && p[i] < 0x80) {
            b = a.byte_of_cpt[p[i]];
        } else if (len == 2 && p[i] >= 0xC2 && (p[i + 1] & 0xC0) == 0x80) {
            const uint32_t c = ((uint32_t)(p[i] & 0x1F) << 6) | (p[i + 1] & 0x3F);
            if (c < 0x144) b = a.byte_of_cpt[c];
        }
        if (b >= 0) out += (char)b;
        else        out.append(enc, i, len);
        i += len;
    }
    return out;
}

/* The raw bytes a byte-level token stands for, when every character of it is in the alphabet.
 * Only such a token can equal a mapped word, so a table keyed by these raw bytes answers every
 * lookup the mapped text would, and the encoder can merge raw bytes without building the mapped
 * text. */
bool byte_decode_strict(const std::string& enc, std::string* out) {
    const ByteLevelAlphabet& a = alphabet();
    out->clear();
    const uint8_t* p = (const uint8_t*)enc.data();
    for (size_t i = 0; i < enc.size(); ) {
        uint32_t c;
        if (p[i] < 0x80)                                          { c = p[i]; i += 1; }
        else if (p[i] >= 0xC2 && p[i] < 0xE0 && i + 1 < enc.size() && (p[i + 1] & 0xC0) == 0x80) {
            c = ((p[i] & 0x1F) << 6) | (p[i + 1] & 0x3F);
            i += 2;
        } else {
            return false;
        }
        if (c >= 0x144 || a.byte_of_cpt[c] < 0) return false;
        *out += (char)a.byte_of_cpt[c];
    }
    return true;
}

constexpr const char* k_metaspace = "\xe2\x96\x81";   /* U+2581 LOWER ONE EIGHTH BLOCK */

void replace_all(std::string& s, std::string_view from, std::string_view to) {
    if (from.empty()) return;
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    for (;;) {
        size_t j = s.find(from, i);
        if (j == std::string::npos) { out.append(s, i, std::string::npos); break; }
        out.append(s, i, j - i);
        out.append(to);
        i = j + from.size();
    }
    s.swap(out);
}

/* The GPT-2 pattern HuggingFace's ByteLevel pre-tokeniser carries inside itself when
 * use_regex is set. */
constexpr const char* k_gpt2_pattern =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

/* HuggingFace's character-class pre-tokenisers, as the patterns they equal, so they run through
 * the Split machinery:
 *   Whitespace   \w+|[^\w\s]+ in Rust's regex crate (not Oniguruma), its matches kept and the rest
 *                removed. Rust's \w is Alphabetic, marks, Nd, Pc and the joiners: Alphabetic is
 *                the letters, Nl and a few symbol letters (circled and squared Latin), so No
 *                ("²", "½") is not a word character as it is in Oniguruma.
 *   whitespace   Rust's char::is_whitespace, the White_Space property, which is \s
 *   punctuation  char::is_ascii_punctuation or a Unicode P* category (not symbols: "$" is
 *                ASCII punctuation, "€" is not punctuation)
 *   digits       char::is_numeric, the N* categories */
#define RAD_RUST_WORD "\\p{L}\\p{Nl}\\p{M}\\p{Nd}\\p{Pc}\\x{200C}\\x{200D}\\x{24B6}-\\x{24E9}" \
                      "\\x{1F130}-\\x{1F149}\\x{1F150}-\\x{1F169}\\x{1F170}-\\x{1F189}"
constexpr const char* k_word_runs   = "[" RAD_RUST_WORD "]+|[^" RAD_RUST_WORD "\\s]+";
#undef RAD_RUST_WORD
constexpr const char* k_space_runs  = "\\s+";
constexpr const char* k_punctuation = "[\\p{P}\\x21-\\x2F\\x3A-\\x40\\x5B-\\x60\\x7B-\\x7E]";
constexpr const char* k_digit       = "\\p{N}";
constexpr const char* k_digit_runs  = "\\p{N}+";

}  /* namespace */

/* ================================================================== the compiled encoder
 *
 * Everything encode() needs that is fixed per vocab: each Split pattern compiled to its DFA, the
 * token and merge tables as open-addressed arrays keyed for the merge loop, the special-token
 * matcher and the segment cache. Built by Vocab::load and shared by every Tokenizer over the
 * vocab. Only two parts change afterwards, both safe to share: the segment cache, which locks,
 * and the self-merge memo, whose entries are written once with relaxed atomics. */

namespace {

/* --- special tokens ---------------------------------------------------------------------- */

struct SpecialFrag { int32_t id; uint32_t off, len; };

/* The text is cut at every special token before any other step sees it. Occurrences are claimed
 * by priority, not position: longest token first (then lowest id), and a token claims only
 * occurrences that overlap neither a higher-priority claim nor one of its own claims to the
 * left. This equals splitting once per token, longest first, so "<|im_start|>" is never cut by a
 * shorter token that is its prefix. Occurrences are found in one pass with a trie of all special
 * tokens, started only at bytes some token begins with. */
class SpecialMatcher {
public:
    void build(const std::vector<int32_t>& special, const std::vector<std::string>& tokens,
               const std::vector<uint8_t>& types) {
        nodes_.assign(1, Node{});
        for (size_t prio = 0; prio < special.size(); ++prio) {
            const int32_t id = special[prio];
            const std::string& t = tokens[id];
            toks_.push_back({ id, (uint32_t)t.size(),
                              types[id] == RAD_TT_CONTROL || types[id] == RAD_TT_UNKNOWN });
            uint32_t n = 0;
            for (unsigned char c : t) {
                uint32_t next = UINT32_MAX;
                for (const auto& k : nodes_[n].kids) if (k.first == c) { next = k.second; break; }
                if (next == UINT32_MAX) {
                    next = (uint32_t)nodes_.size();
                    nodes_[n].kids.push_back({ c, next });
                    nodes_.push_back(Node{});
                }
                n = next;
            }
            nodes_[n].ends.push_back((uint32_t)prio);
            first_[(unsigned char)t[0]] = true;
        }
        int n_first = 0;
        for (int c = 0; c < 256; ++c) if (first_[c]) { ++n_first; only_first_ = c; }
        if (n_first != 1) only_first_ = -1;
    }

    void split(std::string_view text, bool parse_special, std::vector<SpecialFrag>& out) const {
        out.clear();
        const uint8_t* s = (const uint8_t*)text.data();
        const size_t n = text.size();
        struct Occ { uint32_t prio, at; };
        std::vector<Occ> occ;
        auto walk = [&](size_t i) {
            uint32_t node = 0;
            for (size_t j = i; j < n; ++j) {
                uint32_t next = UINT32_MAX;
                for (const auto& k : nodes_[node].kids) if (k.first == s[j]) { next = k.second; break; }
                if (next == UINT32_MAX) return;
                node = next;
                for (uint32_t prio : nodes_[node].ends)
                    if (parse_special || !toks_[prio].control) occ.push_back({ prio, (uint32_t)i });
            }
        };
        if (only_first_ >= 0) {
            for (const uint8_t* p = s; (p = (const uint8_t*)std::memchr(p, only_first_, (s + n) - p)); ++p)
                walk((size_t)(p - s));
        } else if (!toks_.empty()) {
            for (size_t i = 0; i < n; ++i) if (first_[s[i]]) walk(i);
        }
        if (occ.empty()) {
            out.push_back({ -1, 0, (uint32_t)n });
            return;
        }
        std::sort(occ.begin(), occ.end(), [](const Occ& a, const Occ& b) {
            return a.prio != b.prio ? a.prio < b.prio : a.at < b.at;
        });
        std::map<uint32_t, std::pair<uint32_t, int32_t>> taken;     /* start -> (end, id) */
        for (const Occ& o : occ) {
            const uint32_t b = o.at, e = o.at + toks_[o.prio].len;
            auto it = taken.lower_bound(b);
            if (it != taken.end() && it->first < e) continue;
            if (it != taken.begin() && std::prev(it)->second.first > b) continue;
            taken.emplace(b, std::make_pair(e, toks_[o.prio].id));
        }
        uint32_t pos = 0;
        for (const auto& t : taken) {
            if (t.first > pos) out.push_back({ -1, pos, t.first - pos });
            out.push_back({ t.second.second, t.first, t.second.first - t.first });
            pos = t.second.first;
        }
        if (pos < n) out.push_back({ -1, pos, (uint32_t)n - pos });
    }

private:
    struct Tok { int32_t id; uint32_t len; bool control; };
    struct Node {
        std::vector<std::pair<uint8_t, uint32_t>> kids;
        std::vector<uint32_t> ends;            /* priorities of the tokens that end here */
    };
    std::vector<Tok>  toks_;                   /* in priority order */
    std::vector<Node> nodes_;
    bool first_[256] = {};
    int  only_first_ = -1;
};

/* --- the segment cache ------------------------------------------------------------------- */

/* Segment bytes -> ids; tokenizer.h explains why a hit is exact. Keyed by the whole segment (the
 * hash picks the bucket, the bytes decide the match), bounded in bytes held (key, ids and
 * bookkeeping), LRU eviction.
 *
 * The lock covers only the index. An entry is built before it is locked in, and a hit copies the
 * ids through a shared reference after unlocking, so copying a large segment never blocks
 * another encode. */
class SegmentCache {
public:
    using Ids = std::shared_ptr<const std::vector<int32_t>>;

    bool lookup(std::string_view key, uint64_t h, std::vector<int32_t>& out) {
        Ids ids;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto range = index_.equal_range(h);
            for (auto it = range.first; it != range.second; ++it) {
                const Entry& e = *it->second;
                if (e.key.size() != key.size() ||
                    std::memcmp(e.key.data(), key.data(), key.size()) != 0)
                    continue;
                ids = e.ids;
                lru_.splice(lru_.begin(), lru_, it->second);
                break;
            }
            ++(ids ? hits_ : misses_);
        }
        if (!ids) return false;
        out.insert(out.end(), ids->begin(), ids->end());
        return true;
    }

    void insert(std::string_view key, uint64_t h, const int32_t* ids, size_t n) {
        const size_t cost = key.size() + n * sizeof(int32_t) + kOverhead;
        if (cost > limit_.load(std::memory_order_relaxed)) return;
        std::list<Entry> node;
        node.push_back(Entry{ std::string(key), std::make_shared<const std::vector<int32_t>>(ids, ids + n),
                              h, cost });
        std::list<Entry> evicted;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto range = index_.equal_range(h);
            for (auto it = range.first; it != range.second; ++it) {
                const Entry& e = *it->second;
                if (e.key.size() == key.size() && std::memcmp(e.key.data(), key.data(), key.size()) == 0)
                    return;                              /* another encode got here first */
            }
            lru_.splice(lru_.begin(), node);
            index_.emplace(h, lru_.begin());
            bytes_ += cost;
            evict(evicted);
        }
    }

    void set_limit(size_t bytes) {
        std::list<Entry> evicted;
        std::lock_guard<std::mutex> lk(mu_);
        limit_.store(bytes, std::memory_order_relaxed);
        evict(evicted);
    }

    void clear() {
        std::list<Entry> gone;
        std::lock_guard<std::mutex> lk(mu_);
        gone.swap(lru_);
        index_.clear();
        bytes_ = 0;
        hits_ = misses_ = 0;
    }

    Tokenizer::CacheStats stats() {
        std::lock_guard<std::mutex> lk(mu_);
        Tokenizer::CacheStats s;
        s.hits = hits_;
        s.misses = misses_;
        s.entries = lru_.size();
        s.bytes = bytes_;
        s.limit = limit_.load(std::memory_order_relaxed);
        return s;
    }

private:
    struct Entry {
        std::string key;
        Ids         ids;
        uint64_t    hash;
        size_t      cost;
    };
    /* A list node, a hash node, the shared vector's control block and heap headers, rounded up. */
    static constexpr size_t kOverhead = 160;

    /* Moves the least recently used entries into `out` until the bound holds; the caller frees
     * them after it unlocks. */
    void evict(std::list<Entry>& out) {
        while (bytes_ > limit_.load(std::memory_order_relaxed) && !lru_.empty()) {
            auto last = std::prev(lru_.end());
            auto range = index_.equal_range(last->hash);
            for (auto it = range.first; it != range.second; ++it) {
                if (it->second == last) { index_.erase(it); break; }
            }
            bytes_ -= last->cost;
            out.splice(out.begin(), lru_, last);
        }
    }

    std::mutex mu_;
    std::list<Entry> lru_;                                    /* front is most recent */
    std::unordered_multimap<uint64_t, std::list<Entry>::iterator> index_;
    size_t   bytes_ = 0;
    std::atomic<size_t> limit_{ Tokenizer::kSegmentCacheDefault };
    uint64_t hits_ = 0, misses_ = 0;
};

}  /* namespace */

struct TokEncoder {
    struct Step {
        VocabStep                 step;
        /* Split, a regex Replace, ByteLevel with use_regex, and the character-class
         * pre-tokenisers; `re2` is BertPreTokenizer's punctuation after its whitespace split. */
        std::unique_ptr<TokRegex> re, re2;
        /* Precompiled: the decoded character map. */
        std::unique_ptr<PrecompiledMap> pmap;
    };
    std::vector<Step> norm, pre;
    uint32_t kind = RAD_TOK_BPE;
    bool     bytelevel = false;              /* the chain has a ByteLevel pre-tokeniser */
    bool     byte_fallback = false;
    bool     ignore_merges = false;          /* VocabBuild::ignore_merges */
    bool     prepend_first = false;          /* a Metaspace step prepends to the first segment only */
    /* Index of a ByteLevel step that is the last step rewriting text; its mapping is never
     * materialised and the BPE merges raw bytes against `raw`. -1 when there is no ByteLevel or
     * a later step reads the mapped text. */
    int      defer_bytelevel = -1;

    TokTable   text;                         /* token text -> lowest id */
    TokTable   raw;                          /* raw bytes a byte-level token stands for -> id */
    int32_t    byte_sym[256];                /* raw byte -> id of its byte-level character */
    /* BPE's end-of-word suffix (RAD_PRE_END_OF_WORD), empty without one. A word's last symbol
     * is its text with the suffix: `raw_eow` holds the tokens ending in it by the raw bytes
     * before it, and `byte_sym_eow` a byte's character with it. */
    std::string eow;
    TokTable   raw_eow;
    int32_t    byte_sym_eow[256];
    int32_t    byte_tok[256];                /* raw byte -> id of "<0xNN>" */
    MergeTable merges;
    std::vector<uint8_t> types;
    int32_t    unk = -1;
    size_t     longest = 0;                  /* bytes in the longest token text */
    /* Unigram: SentencePiece's unknown-token score, the lowest score minus 10. Derived, not
     * stored in the .rad, so it cannot disagree with the scores. */
    double     unk_penalty = 0.0;

    /* Per token: whether its own text merges back to exactly itself (0 unknown, 1 yes, 2 no).
     * Most words of real text are a token and the answer never changes, so it is computed once
     * and reused by every thread for the life of the vocab. Relaxed atomics: racing threads
     * store the same value. */
    std::unique_ptr<std::atomic<uint8_t>[]> self_merges;

    SpecialMatcher       specials;
    /* The added tokens matched in the normalised text (RAD_PRE_ADDED_NORMALIZED); `specials`
     * holds the rest. */
    SpecialMatcher       nspecials;
    bool                 has_nspecials = false;
    mutable SegmentCache cache;

    int build(const VocabBuild& b, const std::vector<int32_t>& special);
};

int TokEncoder::build(const VocabBuild& b, const std::vector<int32_t>& special) {
    kind = b.kind;
    ignore_merges = b.ignore_merges;
    types = b.types;
    unk = b.unk;

    auto compile = [&](const std::string& pat, const char* what, std::unique_ptr<TokRegex>* re) {
        std::string why;
        if (TokRegex::compile(pat, re, &why) < 0) {
            RAD_ERR("vocab: the %s pattern '%s' is outside the supported regex dialect: %s. "
                    "There is no fallback matcher; a pattern that does not compile does not load.",
                    what, pat.c_str(), why.c_str());
            return RAD_E_UNSUPPORTED;
        }
        return RAD_OK;
    };

    for (const VocabStep& s : b.steps) {
        if (step_is_norm(s.kind)) {
            Step st{ s, nullptr, nullptr, nullptr };
            if (s.kind == RAD_NORM_REPLACE && (s.flags & 1)) RAD_TRY(compile(s.arg0, "Replace", &st.re));
            if (s.kind == RAD_NORM_PRECOMPILED) {
                st.pmap = std::make_unique<PrecompiledMap>();
                std::string why;
                if (st.pmap->load(s.arg0, &why) < 0) {
                    RAD_ERR("vocab: the Precompiled normaliser's map: %s", why.c_str());
                    return RAD_E_FORMAT;
                }
            }
            norm.push_back(std::move(st));
        } else if (step_is_pre(s.kind)) {
            Step st{ s, nullptr, nullptr, nullptr };
            if (s.kind == RAD_PRE_SPLIT) RAD_TRY(compile(s.arg0, "Split", &st.re));
            if (s.kind == RAD_PRE_WHITESPACE) {
                RAD_TRY(compile(s.flags == kPreWhitespace ? k_word_runs : k_space_runs,
                                "Whitespace", &st.re));
                if (s.flags == kPreBert) RAD_TRY(compile(k_punctuation, "Bert", &st.re2));
            }
            if (s.kind == RAD_PRE_PUNCTUATION) RAD_TRY(compile(k_punctuation, "Punctuation", &st.re));
            if (s.kind == RAD_PRE_DIGITS)
                RAD_TRY(compile((s.flags & 1) ? k_digit : k_digit_runs, "Digits", &st.re));
            if (s.kind == RAD_PRE_BYTELEVEL) {
                bytelevel = true;
                if (s.flags & 2) RAD_TRY(compile(k_gpt2_pattern, "ByteLevel", &st.re));
            }
            if (s.kind == RAD_PRE_BYTE_FALLBACK) byte_fallback = true;
            if (s.kind == RAD_PRE_END_OF_WORD) eow = s.arg0;
            if (s.kind == RAD_PRE_METASPACE && s.iarg == 1) prepend_first = true;
            pre.push_back(std::move(st));
        }
    }
    if (kind == RAD_TOK_BPE) {
        for (size_t k = pre.size(); k-- > 0; ) {
            const uint32_t sk = pre[k].step.kind;
            if (sk == RAD_PRE_BYTE_FALLBACK || sk == RAD_PRE_END_OF_WORD) continue;
            if (sk == RAD_PRE_BYTELEVEL) defer_bytelevel = (int)k;
            break;
        }
    }

    const size_t n = b.tokens.size();
    text.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        text.insert(b.tokens[i], (int32_t)i);
        longest = std::max(longest, b.tokens[i].size());
    }
    if (!b.scores.empty())
        unk_penalty = *std::min_element(b.scores.begin(), b.scores.end()) - 10.0;

    const ByteLevelAlphabet& a = alphabet();
    for (int c = 0; c < 256; ++c) {
        static const char* hex = "0123456789ABCDEF";
        const char buf[7] = { '<', '0', 'x', hex[c >> 4], hex[c & 15], '>', 0 };
        byte_tok[c] = text.find(buf);
        byte_sym[c] = text.find(a.to_utf8[c]);
        byte_sym_eow[c] = eow.empty() ? -1 : text.find(a.to_utf8[c] + eow);
    }
    if (defer_bytelevel >= 0) {
        raw.reserve(n);
        std::string r;
        for (size_t i = 0; i < n; ++i) {
            const std::string& t = b.tokens[i];
            if (t.empty()) continue;
            if (!eow.empty() && t.size() > eow.size() &&
                t.compare(t.size() - eow.size(), eow.size(), eow) == 0 &&
                byte_decode_strict(t.substr(0, t.size() - eow.size()), &r))
                raw_eow.insert(r, (int32_t)i);
            if (byte_decode_strict(t, &r)) raw.insert(r, (int32_t)i);
        }
    }

    /* Keyed by the pair as the vocab lists it; the lowest rank wins a duplicate. A symbol's id is
     * always the lowest id spelling its text, so a merge listed against a higher id with the same
     * text is never looked up. */
    merges.reserve(b.merges.size());
    std::string cat;
    for (size_t r = 0; r < b.merges.size(); ++r) {
        const uint32_t l = b.merges[r].first, rr = b.merges[r].second;
        if (l >= n || rr >= n) continue;
        cat.assign(b.tokens[l]);
        cat += b.tokens[rr];
        merges.insert((int32_t)l, (int32_t)rr, (uint32_t)r, text.find(cat));
    }

    /* Each matcher keeps the priority order of `special` (longest first). */
    std::vector<int32_t> raw_special, norm_special;
    for (int32_t id : special) {
        bool normalized = false;
        for (const VocabStep& s : b.steps)
            normalized |= s.kind == RAD_PRE_ADDED_NORMALIZED && s.iarg == id;
        (normalized ? norm_special : raw_special).push_back(id);
    }
    specials.build(raw_special, b.tokens, b.types);
    nspecials.build(norm_special, b.tokens, b.types);
    has_nspecials = !norm_special.empty();
    self_merges.reset(new std::atomic<uint8_t>[n]());
    /* A byte-level BPE never looks text up by its mapped spelling once `raw` exists. */
    if (defer_bytelevel >= 0) text = TokTable();
    return RAD_OK;
}

/* ================================================================== Vocab */

int Vocab::load(VocabBuild b) {
    b_ = std::move(b);
    ready_ = false;
    text_to_id_.clear();
    merge_rank_.clear();
    special_.clear();
    byte_of_.clear();

    if (b_.tokens.empty()) {
        RAD_ERR("vocab: no tokens");
        return RAD_E_FORMAT;
    }
    if (b_.kind != RAD_TOK_BPE && b_.kind != RAD_TOK_UNIGRAM && b_.kind != RAD_TOK_WORDPIECE) {
        RAD_ERR("vocab: tokenizer kind %u is not implemented (BPE, unigram and wordpiece are)",
                b_.kind);
        return RAD_E_UNSUPPORTED;
    }
    /* A step this tokenizer does not run would silently change the text or the token boundaries,
     * so a vocabulary that declares one does not load. */
    for (const VocabStep& s : b_.steps) {
        const uint32_t k = s.kind;
        if ((k >= RAD_NORM_NFC && k <= RAD_NORM_PRECOMPILED) ||
            (k >= RAD_PRE_BYTELEVEL && k <= RAD_PRE_ADDED_NORMALIZED) ||
            (k >= RAD_DEC_BYTELEVEL && k <= RAD_DEC_BYTE_FALLBACK))
            continue;
        RAD_ERR("vocab: %s is not implemented", step_describe(s).c_str());
        return RAD_E_UNSUPPORTED;
    }
    /* Values a step does not define would run as some other behaviour, so they do not load
     * either. A Strip after Fuse strips the text as a whole, which a stream can do only at its
     * start (see Detokenizer). */
    {
        bool fused = false, fused_strip = false;
        for (const VocabStep& s : b_.steps) {
            const bool behavior_step = s.kind == RAD_PRE_SPLIT || s.kind == RAD_PRE_PUNCTUATION;
            const bool bad =
                (behavior_step && (s.iarg < TOK_SPLIT_REMOVED || s.iarg > TOK_SPLIT_CONTIGUOUS)) ||
                (s.kind == RAD_PRE_WHITESPACE && s.flags > kPreBert) ||
                (s.kind == RAD_DEC_REPLACE &&
                 (s.flags & ~(kDecWordPiece | kDecWordPieceCleanup | kDecBpeSuffix)) != 0) ||
                (s.kind == RAD_DEC_STRIP && fused && ((s.iarg >> 32) != 0 || fused_strip));
            if (bad) {
                RAD_ERR("vocab: %s is not a step this tokenizer runs (flags %u)",
                        step_describe(s).c_str(), s.flags);
                return RAD_E_UNSUPPORTED;
            }
            if (s.kind == RAD_DEC_FUSE) fused = true;
            if (s.kind == RAD_DEC_STRIP && fused) fused_strip = true;
        }
    }
    /* A token matched in the normalised text is one of the vocabulary's added tokens, spelt as
     * the step says. */
    for (const VocabStep& s : b_.steps) {
        if (s.kind != RAD_PRE_ADDED_NORMALIZED) continue;
        const int32_t id = (int32_t)s.iarg;
        if (s.iarg < 0 || s.iarg >= (int64_t)b_.tokens.size() || s.arg0.empty() ||
            b_.tokens[(size_t)id] != s.arg0 ||
            (!b_.types.empty() && b_.types[(size_t)id] != RAD_TT_CONTROL &&
             b_.types[(size_t)id] != RAD_TT_USER_DEFINED)) {
            RAD_ERR("vocab: %s names no added token of that spelling", step_describe(s).c_str());
            return RAD_E_FORMAT;
        }
    }
    if (b_.types.empty()) b_.types.assign(b_.tokens.size(), (uint8_t)RAD_TT_NORMAL);
    if (b_.types.size() != b_.tokens.size()) {
        RAD_ERR("vocab: %zu tokens but %zu token types", b_.tokens.size(), b_.types.size());
        return RAD_E_FORMAT;
    }
    if (b_.kind == RAD_TOK_UNIGRAM && b_.scores.size() != b_.tokens.size()) {
        RAD_ERR("vocab: unigram needs a score per token; have %zu for %zu tokens",
                b_.scores.size(), b_.tokens.size());
        return RAD_E_FORMAT;
    }

    /* Special ids are emitted as they stand (bos/eos by encode, unk by every encoder on uncovered
     * text), so an out-of-range one would put an id outside the vocabulary into a sequence. GGUF
     * and container sources carry them as unchecked integers; every vocabulary passes through
     * here, so an out-of-range id is treated as absent, with a warning. */
    {
        const struct { const char* name; int32_t* id; } specials[] = {
            { "bos", &b_.bos }, { "eos", &b_.eos }, { "eot", &b_.eot },
            { "pad", &b_.pad }, { "unk", &b_.unk }, { "sep", &b_.sep },
        };
        for (const auto& s : specials) {
            if (*s.id >= -1 && (int64_t)*s.id < (int64_t)b_.tokens.size()) continue;
            RAD_WARN("vocab: the %s id %d names no token in a %zu-token vocabulary; treating "
                     "the vocabulary as having none", s.name, *s.id, b_.tokens.size());
            *s.id = -1;
        }
    }

    text_to_id_.reserve(b_.tokens.size() * 2);
    for (size_t i = 0; i < b_.tokens.size(); ++i) {
        /* First spelling wins. Duplicates are a producer bug but occur in llama.cpp's own vocabs,
         * and the merge table and published token counts refer to the lower id. */
        text_to_id_.emplace(b_.tokens[i], (int32_t)i);
    }

    /* Keyed by id pair rather than string pair, so a lookup hashes one integer instead of a
     * concatenated string. */
    merge_rank_.reserve(b_.merges.size() * 2);
    for (size_t r = 0; r < b_.merges.size(); ++r) {
        const uint64_t key = ((uint64_t)b_.merges[r].first << 32) | (uint64_t)b_.merges[r].second;
        merge_rank_.emplace(key, (int32_t)r);
    }

    /* A ByteFallback decoder turns "<0xNN>" pieces into their bytes whatever their type: a
     * tokenizer.json types every model token alike, so its byte pieces arrive NORMAL. */
    bool dec_byte_fallback = false;
    for (const VocabStep& s : b_.steps) dec_byte_fallback |= s.kind == RAD_DEC_BYTE_FALLBACK;

    byte_of_.assign(b_.tokens.size(), -1);
    const ByteLevelAlphabet& a = alphabet();
    for (size_t i = 0; i < b_.tokens.size(); ++i) {
        const std::string& t = b_.tokens[i];
        const bool spm_byte = dec_byte_fallback && b_.types[i] == RAD_TT_NORMAL && t.size() == 6 &&
                              t.compare(0, 3, "<0x") == 0 && t.back() == '>';
        if (b_.types[i] == RAD_TT_BYTE || spm_byte) {
            /* Two spellings: SPM's "<0xF0>" and byte-level BPE's single mapped codepoint. Both
             * decode to one byte. */
            if (t.size() == 6 && t.compare(0, 3, "<0x") == 0 && t.back() == '>') {
                const int hi = t[3], lo = t[4];
                auto hex = [](int c) { return c >= '0' && c <= '9' ? c - '0'
                                            : c >= 'a' && c <= 'f' ? c - 'a' + 10
                                            : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
                if (hex(hi) >= 0 && hex(lo) >= 0) byte_of_[i] = hex(hi) * 16 + hex(lo);
            } else {
                auto it = a.from.find(t);
                if (it != a.from.end()) byte_of_[i] = it->second;
            }
        }
        if (b_.types[i] == RAD_TT_CONTROL || b_.types[i] == RAD_TT_UNKNOWN ||
            b_.types[i] == RAD_TT_USER_DEFINED) {
            if (!t.empty()) special_.push_back((int32_t)i);
        }
    }

    /* Longest first: this is SpecialMatcher's priority order, so a shorter token that is a prefix
     * of a longer one ("<|im" vs "<|im_start|>") never cuts it. */
    std::sort(special_.begin(), special_.end(), [&](int32_t x, int32_t y) {
        const size_t nx = b_.tokens[x].size(), ny = b_.tokens[y].size();
        return nx != ny ? nx > ny : x < y;
    });

    /* ------------------------------------------------------------------ end of generation
     *
     * A model can end its turn on several tokens, but the header has only `eos` and `eot`. A Qwen
     * container declares eos_token_id: [<|im_end|>, <|endoftext|>], and its `<|endoftext|>` lands
     * in `pad_id`, which is not terminal; without it, generation runs past the turn into an
     * invented next one.
     *
     * The set is completed from control tokens whose spelling is a known end-of-generation
     * marker (llama.cpp's loader uses the same list). Only control tokens qualify, so ordinary
     * text never becomes terminal, and the list is explicit rather than a pattern, so a token
     * like `<|end_of_thinking|>` does not end generations. Every addition is logged. */
    eog_.clear();
    if (b_.eos >= 0) eog_.push_back(b_.eos);
    if (b_.eot >= 0 && b_.eot != b_.eos) eog_.push_back(b_.eot);
    {
        /* Gemma 4 ends a turn on `<turn|>`, and a tool-calling turn on `<|tool_response>` (the
         * opener of the caller's result, written after its last `<tool_call|>`), but declares
         * only `<eos>`. llama.cpp's loader lists both; both are control tokens. */
        static const char* const kEogNames[] = {
            "<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>",
            "<|end|>", "<|EOT|>", "<|eom_id|>", "<|return|>", "<eos>", "</s>",
            "<turn|>", "<|tool_response>",
        };
        /* `</s>` ends a document, not a turn. In a vocabulary with its own turn marker, a
         * control `</s>` that is not the declared eos is text a reply may write (Gemma 4 spells
         * the HTML tag with it), so it joins the set only when there is no turn marker. */
        static const char* const kTurnEnds[] = {
            "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|EOT|>", "<turn|>",
        };
        bool has_turn_end = false;
        for (int32_t id : special_) {
            if (b_.types[id] != RAD_TT_CONTROL) continue;
            for (const char* n : kTurnEnds) if (b_.tokens[id] == n) has_turn_end = true;
        }
        std::string added;
        for (int32_t id : special_) {
            if (b_.types[id] != RAD_TT_CONTROL) continue;
            if (std::find(eog_.begin(), eog_.end(), id) != eog_.end()) continue;
            bool known = false;
            for (const char* n : kEogNames) if (b_.tokens[id] == n) { known = true; break; }
            if (has_turn_end && b_.tokens[id] == "</s>") known = false;
            if (!known) continue;
            eog_.push_back(id);
            added += (added.empty() ? "" : " ") + b_.tokens[id] + "=" + std::to_string(id);
        }
        if (!added.empty())
            RAD_INFO("vocab: end-of-generation also on %s (a control token the container declares "
                     "neither as eos nor eot; without this a turn ending on it runs on)",
                     added.c_str());

        /* Plus the checkpoint's own list, generation_config.json's eos_token_id. Gemma 4 names
         * <|tool_response> there beside <eos> and <turn|>: after a call the model writes it and
         * waits, and a turn that runs past it answers its own call. Every id in range is
         * honoured, as in transformers. */
        std::string named;
        for (int32_t id : b_.eos_ids) {
            if (id < 0 || id >= (int32_t)b_.tokens.size()) continue;
            if (std::find(eog_.begin(), eog_.end(), id) != eog_.end()) continue;
            eog_.push_back(id);
            named += (named.empty() ? "" : " ") + b_.tokens[id] + "=" + std::to_string(id);
        }
        if (!named.empty())
            RAD_INFO("vocab: end-of-generation also on %s (named by the checkpoint's "
                     "generation_config.json)", named.c_str());
    }

    /* A pattern the regex compiler refuses fails the vocab load, with the construct named. */
    enc_ = std::make_shared<TokEncoder>();
    RAD_TRY(enc_->build(b_, special_));

    ready_ = true;
    RAD_DEBUG("vocab: %zu tokens, %zu merges, %zu special, %zu eog, %zu chain steps, kind %u",
              b_.tokens.size(), b_.merges.size(), special_.size(), eog_.size(), b_.steps.size(),
              b_.kind);
    return RAD_OK;
}

const std::string& Vocab::text(int32_t id) const {
    static const std::string empty;
    if (id < 0 || id >= (int32_t)b_.tokens.size()) return empty;
    return b_.tokens[id];
}

uint8_t Vocab::type(int32_t id) const {
    if (id < 0 || id >= (int32_t)b_.types.size()) return RAD_TT_UNKNOWN;
    return b_.types[id];
}

float Vocab::score(int32_t id) const {
    if (id < 0 || id >= (int32_t)b_.scores.size()) return 0.0f;
    return b_.scores[id];
}


bool Vocab::is_eog(int32_t id) const {
    return id >= 0 && std::find(eog_.begin(), eog_.end(), id) != eog_.end();
}

int32_t Vocab::find(std::string_view t) const {
    auto it = text_to_id_.find(std::string(t));
    return it == text_to_id_.end() ? -1 : it->second;
}

int32_t Vocab::merge_rank(std::string_view left, std::string_view right) const {
    const int32_t l = find(left), r = find(right);
    if (l < 0 || r < 0) return -1;
    auto it = merge_rank_.find(((uint64_t)(uint32_t)l << 32) | (uint32_t)r);
    return it == merge_rank_.end() ? -1 : it->second;
}

int32_t Vocab::token_byte(int32_t id) const {
    if (id < 0 || id >= (int32_t)byte_of_.size()) return -1;
    return byte_of_[id];
}

/* ================================================================== the step chain */

std::string step_describe(const VocabStep& s) {
    switch (s.kind) {
        case RAD_NORM_NFC:  return "NFC";
        case RAD_NORM_NFKC: return "NFKC";
        case RAD_NORM_NFD:  return "NFD";
        case RAD_NORM_NFKD: return "NFKD";
        case RAD_NORM_LOWERCASE: return "Lowercase";
        case RAD_NORM_STRIP:
            return fmt("Strip(left=%d right=%d)", (s.flags & 1) ? 1 : 0, (s.flags & 2) ? 1 : 0);
        case RAD_NORM_REPLACE:
            return fmt("Replace(%s '%s' -> '%s')", (s.flags & 1) ? "regex" : "literal",
                       s.arg0.c_str(), s.arg1.c_str());
        case RAD_NORM_PREPEND: return fmt("Prepend('%s')", s.arg0.c_str());
        case RAD_NORM_PRECOMPILED:
            return fmt("Precompiled(%zu base64 bytes)", s.arg0.size());
        case RAD_PRE_BYTELEVEL:
            return fmt("ByteLevel(add_prefix_space=%d use_regex=%d)",
                       (s.flags & 1) ? 1 : 0, (s.flags & 2) ? 1 : 0);
        case RAD_PRE_SPLIT: {
            /* Includes the compiled DFA's size, or the construct that refused it. */
            std::unique_ptr<TokRegex> re;
            std::string why;
            const std::string dfa = TokRegex::compile(s.arg0, &re, &why) < 0
                ? "[refused: " + why + "]"
                : fmt("[dfa: %zu states, %zu classes]", re->n_states(), re->n_classes());
            return fmt("Split(%s behavior=%lld invert=%d) %s",
                       s.arg0.c_str(), (long long)s.iarg, (s.flags & 1) ? 1 : 0, dfa.c_str());
        }
        case RAD_PRE_METASPACE:
            return fmt("Metaspace('%s' prepend=%lld split=%d)", s.arg0.c_str(),
                       (long long)s.iarg, (s.flags & 1) ? 1 : 0);
        case RAD_PRE_WHITESPACE:
            return s.flags == kPreWhitespaceSplit ? "WhitespaceSplit"
                 : s.flags == kPreBert            ? "BertPreTokenizer"
                 : s.flags == kPreWhitespace      ? "Whitespace"
                                                  : fmt("Whitespace(flags=%u)", s.flags);
        case RAD_PRE_PUNCTUATION:  return fmt("Punctuation(behavior=%lld)", (long long)s.iarg);
        case RAD_PRE_DIGITS:
            return fmt("Digits(individual=%d)", (s.flags & 1) ? 1 : 0);
        case RAD_PRE_BYTE_FALLBACK: return "ByteFallback";
        case RAD_PRE_END_OF_WORD:   return fmt("EndOfWordSuffix('%s')", s.arg0.c_str());
        case RAD_PRE_ADDED_NORMALIZED:
            return fmt("AddedNormalized('%s' = %lld)", s.arg0.c_str(), (long long)s.iarg);
        case RAD_DEC_BYTELEVEL:     return "decode ByteLevel";
        case RAD_DEC_METASPACE:
            return fmt("decode Metaspace('%s' prepend=%lld)", s.arg0.c_str(), (long long)s.iarg);
        case RAD_DEC_REPLACE:
            if (s.flags & kDecWordPiece)
                return fmt("decode WordPiece(prefix='%s' cleanup=%d)", s.arg0.c_str(),
                           (s.flags & kDecWordPieceCleanup) ? 1 : 0);
            if (s.flags & kDecBpeSuffix) return fmt("decode BPEDecoder(suffix='%s')", s.arg0.c_str());
            return fmt("decode Replace('%s' -> '%s')", s.arg0.c_str(), s.arg1.c_str());
        case RAD_DEC_STRIP:
            return fmt("decode Strip('%s' start=%lld stop=%lld)", s.arg0.c_str(),
                       (long long)(s.iarg & 0xFFFFFFFF), (long long)(s.iarg >> 32));
        case RAD_DEC_FUSE:          return "decode Fuse";
        case RAD_DEC_BYTE_FALLBACK: return "decode ByteFallback";
        default: return fmt("<unknown step %u>", s.kind);
    }
}

namespace {

/* --- normalisers ------------------------------------------------------------------------- */

void apply_norm(std::string& text, const TokEncoder::Step& st) {
    const VocabStep& s = st.step;
    switch (s.kind) {
        case RAD_NORM_NFC: tok_nfc(text); break;
        case RAD_NORM_NFKC: text = unorm_utf8(text, true,  true);  break;
        case RAD_NORM_NFD:  text = unorm_utf8(text, false, false); break;
        case RAD_NORM_NFKD: text = unorm_utf8(text, true,  false); break;
        case RAD_NORM_LOWERCASE: {
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(text);
            std::string out;
            out.reserve(text.size());
            /* HuggingFace lowercases a character at a time with Rust's char::to_lowercase, whose
             * one unconditional mapping to two characters is U+0130 (I with dot above) to
             * "i" U+0307; every other character maps to one. */
            for (uint32_t c : cpts) {
                if (c == 0x130) out += "i\xcc\x87";
                else out += unicode_cpt_to_utf8(unicode_tolower(c));
            }
            text.swap(out);
            break;
        }
        case RAD_NORM_STRIP: {
            /* Unicode whitespace, not isspace(): HuggingFace strips a non-breaking space too. */
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(text);
            size_t a = 0, b = cpts.size();
            if (s.flags & 1) while (a < b && unicode_cpt_flags_from_cpt(cpts[a]).is_whitespace) ++a;
            if (s.flags & 2) while (b > a && unicode_cpt_flags_from_cpt(cpts[b-1]).is_whitespace) --b;
            std::string out;
            for (size_t i = a; i < b; ++i) out += unicode_cpt_to_utf8(cpts[i]);
            text.swap(out);
            break;
        }
        case RAD_NORM_REPLACE:
            if (s.flags & 1) {
                /* HuggingFace's Replace: each match becomes the content verbatim (text, not a
                 * substitution template). */
                std::string canon, out;
                std::string_view v = text;
                if (!utf8_is_canonical(v)) { utf8_canonicalize(v, canon); v = canon; }
                tok_replace(*st.re, v, s.arg1, out);
                text.swap(out);
            } else {
                replace_all(text, s.arg0, s.arg1);
            }
            break;
        case RAD_NORM_PREPEND: text = s.arg0 + text; break;
        case RAD_NORM_PRECOMPILED: text = st.pmap->normalize(text); break;
    }
}

/* --- pre-tokenisers ----------------------------------------------------------------------
 *
 * Pieces are spans of one buffer, not a string each, to avoid an allocation per piece (a 500 KB
 * prompt is about 150,000 pieces). A step that rewrites text appends it to the buffer (or builds
 * a new buffer) and points its spans there. */

struct Pieces {
    std::string          buf;
    std::vector<TokSpan> spans;
    bool                 canonical = false;   /* every piece is known to be canonical UTF-8 */
    std::string_view at(const TokSpan& s) const {
        return std::string_view(buf).substr(s.begin, s.end - s.begin);
    }
};

struct EncodeScratch {
    Pieces               p;
    std::string          canon;
    std::vector<TokSpan> next, found, match;
    std::vector<BpeSym>  syms;
    BpeMergeScratch      merge;
};

/* One Split step over every piece. The output pieces are canonical UTF-8: a piece that is not is
 * first re-encoded by utf8_canonicalize (invalid bytes to U+FFFD, overlong forms shortest), the
 * same rules the other steps' decoder uses. */
void apply_split(Pieces& p, const TokRegex& re, int behavior, bool invert, EncodeScratch& sc) {
    if (!p.canonical) p.canonical = utf8_is_canonical(p.buf);
    const bool whole = p.canonical;
    sc.next.clear();
    for (size_t k = 0; k < p.spans.size(); ++k) {
        const TokSpan s = p.spans[k];
        if (s.end == s.begin) continue;
        uint32_t base = s.begin;
        std::string_view v = p.at(s);
        if (!whole && !utf8_is_canonical(v)) {
            sc.canon.clear();
            utf8_canonicalize(v, sc.canon);
            base = (uint32_t)p.buf.size();
            p.buf += sc.canon;
            v = std::string_view(p.buf).substr(base, sc.canon.size());
        }
        sc.found.clear();
        tok_split(re, v, behavior, invert, sc.found, sc.match);
        for (const TokSpan& f : sc.found) sc.next.push_back({ base + f.begin, base + f.end });
    }
    p.spans.swap(sc.next);
    p.canonical = true;
}

void rebuild(Pieces& p, const std::vector<std::string>& words) {
    std::string buf;
    std::vector<TokSpan> spans;
    for (const std::string& w : words) {
        spans.push_back({ (uint32_t)buf.size(), (uint32_t)(buf.size() + w.size()) });
        buf += w;
    }
    p.buf.swap(buf);
    p.spans.swap(spans);
}

std::vector<std::string> words_of(const Pieces& p) {
    std::vector<std::string> w;
    w.reserve(p.spans.size());
    for (const TokSpan& s : p.spans) w.emplace_back(p.at(s));
    return w;
}

/* A step that rewrites words: the pieces as words, `f` over them, the pieces rebuilt. */
template <class F>
void rewrite(Pieces& p, F&& f) {
    std::vector<std::string> w = words_of(p);
    f(w);
    rebuild(p, w);
}

void apply_bytelevel(Pieces& p, const TokEncoder::Step& st, bool defer, EncodeScratch& sc) {
    const bool add_prefix_space = (st.step.flags & 1) != 0;
    if (add_prefix_space && !p.spans.empty() && p.spans[0].end > p.spans[0].begin &&
        p.buf[p.spans[0].begin] != ' ') {
        const std::string first = " " + std::string(p.at(p.spans[0]));
        p.spans[0] = { (uint32_t)p.buf.size(), (uint32_t)(p.buf.size() + first.size()) };
        p.buf += first;
    }
    /* HuggingFace's ByteLevel carries the GPT-2 pattern inside it. rad-convert normally emits it
     * as an explicit Split step, but a chain that declares it inline must still work. */
    if (st.re) apply_split(p, *st.re, TOK_SPLIT_ISOLATED, false, sc);
    if (defer) return;
    rewrite(p, [](std::vector<std::string>& w) {
        for (std::string& x : w) x = byte_encode(x);
    });
}

/* `at_start`: the words are the text's first segment. HuggingFace's "first" scheme prepends only
 * there (a segment after a special token has a non-zero original offset); "always" prepends to
 * every segment. Either prepends only to a word that does not already start with the replacement
 * once its spaces are replaced: a space, or the replacement itself, which a normaliser before this
 * step (mmBERT's and Gemma's Replace " " -> U+2581) has already written. */
void apply_metaspace(std::vector<std::string>& words, const VocabStep& s, bool at_start) {
    const std::string rep = s.arg0.empty() ? std::string(k_metaspace) : s.arg0;
    const int64_t prepend = s.iarg;            /* 0 never, 1 first, 2 always */
    const bool    split   = (s.flags & 1) != 0;

    for (size_t i = 0; i < words.size(); ++i) {
        std::string& w = words[i];
        if (prepend == 2 || (prepend == 1 && i == 0 && at_start)) {
            if (w.empty() || (w.compare(0, 1, " ") != 0 && w.compare(0, rep.size(), rep) != 0))
                w.insert(0, " ");
        }
        replace_all(w, " ", rep);
    }
    if (!split) return;
    std::vector<std::string> out;
    for (const std::string& w : words) {
        size_t i = 0;
        while (i < w.size()) {
            size_t j = w.find(rep, i + (w.compare(i, rep.size(), rep) == 0 ? rep.size() : 0));
            if (j == std::string::npos) { out.push_back(w.substr(i)); break; }
            out.push_back(w.substr(i, j - i));
            i = j;
        }
        if (w.empty()) out.push_back(w);
    }
    words.swap(out);
}

/* --- models ------------------------------------------------------------------------------ */

/* Word -> ids within one encode() call. Prose and code repeat words heavily, so most words are
 * merged once per call and then copied; ids depend only on the word, so the copy is exact. */
class WordCache {
public:
    static constexpr size_t kMaxKey = 128;

    bool find(std::string_view w, uint64_t h, std::vector<int32_t>& out) const {
        if (slots_.empty()) return false;
        for (size_t i = h & mask_;; i = (i + 1) & mask_) {
            const Slot& s = slots_[i];
            if (s.klen == kEmpty) return false;
            if (s.hash == h && s.klen == w.size() &&
                std::memcmp(keys_.data() + s.koff, w.data(), w.size()) == 0) {
                out.insert(out.end(), ids_.begin() + s.ioff, ids_.begin() + s.ioff + s.ilen);
                return true;
            }
        }
    }

    void insert(std::string_view w, uint64_t h, const int32_t* ids, size_t n) {
        if ((n_ + 1) * 2 > slots_.size()) grow();
        for (size_t i = h & mask_;; i = (i + 1) & mask_) {
            Slot& s = slots_[i];
            if (s.klen != kEmpty) continue;
            s = { h, (uint32_t)keys_.size(), (uint32_t)w.size(), (uint32_t)ids_.size(), (uint32_t)n };
            keys_.append(w.data(), w.size());
            ids_.insert(ids_.end(), ids, ids + n);
            ++n_;
            return;
        }
    }

private:
    static constexpr uint32_t kEmpty = UINT32_MAX;
    struct Slot { uint64_t hash; uint32_t koff, klen, ioff, ilen; };
    std::vector<Slot>    slots_;
    std::string          keys_;
    std::vector<int32_t> ids_;
    size_t n_ = 0, mask_ = 0;

    void grow() {
        std::vector<Slot> old;
        old.swap(slots_);
        const size_t cap = old.empty() ? 1024 : old.size() * 2;
        slots_.assign(cap, Slot{ 0, 0, kEmpty, 0, 0 });
        mask_ = cap - 1;
        for (const Slot& s : old) {
            if (s.klen == kEmpty) continue;
            size_t i = s.hash & mask_;
            while (slots_[i].klen != kEmpty) i = (i + 1) & mask_;
            slots_[i] = s;
        }
    }
};

/* One word through the merge. A symbol carries the lowest id spelling its text, which is what the
 * merge table is keyed on and what the word emits. */
void bpe_word(const TokEncoder& e, std::string_view w, std::vector<int32_t>& out,
              EncodeScratch& sc) {
    const bool raw = e.defer_bytelevel >= 0;
    const bool eow = !e.eow.empty();
    const TokTable& table = raw ? (eow ? e.raw_eow : e.raw) : e.text;

    /* A user-defined token is taken whole: added tokens must round-trip, and the merge table was
     * never fitted over them, so re-merging can give a different sequence.
     *
     * Ordinary entries go through the merge loop unless the vocab sets `ignore_merges`,
     * HuggingFace's short-circuit for a pre-token that is already a token (the merges can reach a
     * different split of a rare word). The raw-byte table finds the same entry by its byte
     * spelling. A control entry is never taken whole: reaching here means the caller asked for
     * it to be read as text. */
    /* With an end-of-word suffix the whole word is its text and the suffix: `raw_eow` is keyed
     * without it, the text table with it. */
    const int32_t whole = eow && !raw ? table.find(std::string(w) + e.eow) : table.find(w);
    if (whole >= 0 && (e.types[whole] == RAD_TT_USER_DEFINED ||
                       (e.ignore_merges && e.types[whole] != RAD_TT_CONTROL))) {
        out.push_back(whole);
        return;
    }
    const uint8_t known = whole >= 0 ? e.self_merges[whole].load(std::memory_order_relaxed) : 0;
    if (known == 1) {
        out.push_back(whole);
        return;
    }

    std::vector<BpeSym>& syms = sc.syms;
    syms.clear();
    const size_t n = w.size();
    if (raw) {
        for (size_t i = 0; i < n; ++i)
            syms.push_back({ (uint32_t)i, 1, e.byte_sym[(unsigned char)w[i]] });
        if (eow) syms.back().id = e.byte_sym_eow[(unsigned char)w[n - 1]];
    } else {
        for (size_t i = 0; i < n; ) {
            const size_t len = std::min(unicode_len_utf8(w[i]), n - i);
            syms.push_back({ (uint32_t)i, (uint32_t)len,
                             eow && i + len == n ? table.find(std::string(w.substr(i, len)) + e.eow)
                                                 : table.find(w.substr(i, len)) });
            i += len;
        }
    }
    bpe_merge(syms, e.merges, sc.merge);
    if (whole >= 0 && known == 0) {
        const bool itself = syms.size() == 1 && syms[0].id == whole;
        e.self_merges[whole].store(itself ? 1 : 2, std::memory_order_relaxed);
    }

    for (const BpeSym& s : syms) {
        if (s.id >= 0) { out.push_back(s.id); continue; }
        const std::string_view str = w.substr(s.start, s.len);

        /* Miss: decompose into byte tokens, spelt per the chain. Under ByteLevel each byte is its
         * own mapped codepoint, so the piece splits at codepoint boundaries; under byte fallback
         * a byte is "<0xNN>". */
        const bool last = s.start + s.len == n;
        if (raw) {
            for (size_t k = 0; k < str.size(); ++k) {
                const unsigned char c = (unsigned char)str[k];
                const int32_t b = eow && last && k + 1 == str.size() ? e.byte_sym_eow[c]
                                                                    : e.byte_sym[c];
                if (b >= 0) out.push_back(b);
                else RAD_WARN("tokenizer: no token for byte-level piece '%s'",
                              alphabet().to_utf8[c].c_str());
            }
        } else if (e.bytelevel) {
            for (size_t k = 0; k < str.size(); ) {
                const size_t m = std::min(unicode_len_utf8(str[k]), str.size() - k);
                const int32_t b = eow && last && k + m == str.size()
                                      ? e.text.find(std::string(str.substr(k, m)) + e.eow)
                                      : e.text.find(str.substr(k, m));
                if (b >= 0) out.push_back(b);
                else RAD_WARN("tokenizer: no token for byte-level piece '%s'",
                              std::string(str.substr(k, m)).c_str());
                k += m;
            }
        } else if (e.byte_fallback) {
            for (unsigned char c : str) {
                const int32_t b = e.byte_tok[c];
                if (b >= 0) out.push_back(b);
                else if (e.unk >= 0) out.push_back(e.unk);
            }
        } else if (e.unk >= 0) {
            out.push_back(e.unk);
        } else {
            RAD_WARN("tokenizer: piece '%s' is not in the vocab and the chain declares neither "
                     "ByteLevel nor ByteFallback; it is dropped", std::string(str).c_str());
        }
    }
}

void bpe(const TokEncoder& e, const Pieces& p, std::vector<int32_t>& out, EncodeScratch& sc,
         WordCache& cache) {
    const bool raw = e.defer_bytelevel >= 0;
    for (const TokSpan& s : p.spans) {
        const std::string_view w = p.at(s);
        if (w.empty()) continue;
        /* A one-byte word of a byte-level vocab is that byte's token; the lookup and the merge
         * would both land on byte_sym. A missing one takes the warning path in bpe_word. */
        if (raw && w.size() == 1) {
            const int32_t b = e.eow.empty() ? e.byte_sym[(unsigned char)w[0]]
                                            : e.byte_sym_eow[(unsigned char)w[0]];
            if (b >= 0) {
                out.push_back(b);
                continue;
            }
        }
        if (w.size() > WordCache::kMaxKey) { bpe_word(e, w, out, sc); continue; }
        const uint64_t h = tok_hash(w);
        if (cache.find(w, h, out)) continue;
        const size_t start = out.size();
        bpe_word(e, w, out, sc);
        cache.insert(w, h, out.data() + start, out.size() - start);
    }
}

/* Unigram by Viterbi, SentencePiece's algorithm, shaped after llama.cpp's
 * llm_tokenizer_ugm_session. Scores are summed in double as the reference does; in float the
 * argmax differs on long inputs. */
void unigram(const Vocab& v, const TokEncoder& e, const std::string& text,
             std::vector<int32_t>& out) {
    if (text.empty()) return;
    const size_t n = text.size();

    struct Best { int32_t id; size_t from; double score; };
    std::vector<Best> best(n + 1, { v.unk(), 0, -DBL_MAX });
    best[0] = { v.unk(), 0, 0.0 };

    for (size_t i = 0; i < n; ) {
        const size_t cp_len = std::min(unicode_len_utf8(text[i]), n - i);
        bool matched_whole_cp = false;
        /* Prefix scan by hash lookup, up to the longest token. A trie would be faster on a large
         * vocab. */
        const size_t max_len = std::min(n - i, e.longest);
        for (size_t l = 1; l <= max_len; ++l) {
            const int32_t id = e.text.find(std::string_view(text.data() + i, l));
            if (id < 0) continue;
            if (l == cp_len) matched_whole_cp = true;
            const double sc = v.type(id) == RAD_TT_USER_DEFINED ? 0.0 : (double)v.score(id);
            const double cand = best[i].score + sc;
            if (cand > best[i + l].score) best[i + l] = { id, i, cand };
        }
        if (!matched_whole_cp) {
            const double cand = best[i].score + e.unk_penalty;
            if (cand > best[i + cp_len].score) best[i + cp_len] = { v.unk(), i, cand };
        }
        i += cp_len;
    }

    /* Without an unk token, uncovered text still takes the unk transition above, with id -1,
     * which must never reach the output. Such a span is spelt as byte tokens when the chain
     * declares byte fallback and dropped otherwise, as in bpe(). The backtrack runs back to
     * front, so the bytes go in reversed. */
    const int32_t unk = v.unk();
    const bool byte_fallback = unk < 0 && e.byte_fallback;

    const size_t start = out.size();
    size_t dropped = 0;
    bool prev_unk = false;
    for (size_t pos = n; ; ) {
        const Best& b = best[pos];
        const bool is_unk = (b.id == unk);
        if (is_unk && unk < 0) {
            for (size_t k = pos; k-- > b.from; ) {
                const unsigned char c = (unsigned char)text[k];
                const int32_t id = byte_fallback ? e.byte_tok[c] : -1;
                if (id >= 0) out.push_back(id);
                else         ++dropped;
            }
        } else if (!(prev_unk && is_unk)) {
            out.push_back(b.id);
        }
        if (b.from == 0) break;
        prev_unk = is_unk;
        pos = b.from;
    }
    std::reverse(out.begin() + start, out.end());
    if (dropped)
        RAD_WARN("tokenizer: %zu byte(s) of this text have no piece in the vocab, which has no unk "
                 "token and %s; they are dropped", dropped,
                 byte_fallback ? "lacks a byte token for them" : "declares no byte fallback");
}

/* WordPiece: greedy longest-match-first with a "##" continuation prefix, for BERT-family
 * embedding models. */
constexpr size_t kWordPieceMaxChars = 100;

void wordpiece(const Vocab& v, const TokEncoder& e, const Pieces& p, std::vector<int32_t>& out) {
    std::string cont;               /* "##" and the rest of the word, reused for every piece */
    std::vector<int32_t> piece;
    for (const TokSpan& s : p.spans) {
        const std::string_view w = p.at(s);
        if (w.empty()) continue;
        piece.clear();
        /* HuggingFace's max_input_chars_per_word: a longer word is one unknown token. */
        size_t n_chars = 0;
        for (unsigned char ch : w) n_chars += (ch & 0xC0) != 0x80;
        bool ok = n_chars <= kWordPieceMaxChars;
        for (size_t i = 0; ok && i < w.size(); ) {
            /* Longest first; a piece after the first is looked up with "##" in front. */
            std::string_view rest = w.substr(i);
            if (i > 0) {
                cont.assign("##");
                cont.append(rest);
                rest = cont;
            }
            const size_t pre = i > 0 ? 2 : 0;
            size_t len = rest.size();
            int32_t id = -1;
            for (; len > pre; --len)
                if ((id = e.text.find(rest.substr(0, len))) >= 0) break;
            if (id < 0) { ok = false; break; }
            piece.push_back(id);
            i += len - pre;
        }
        if (ok) out.insert(out.end(), piece.begin(), piece.end());
        else if (v.unk() >= 0) out.push_back(v.unk());
    }
}

/* Normalised text in `sc.p.buf` through the pre-tokenisers and the model. */
int encode_normalized(const Vocab& v, const TokEncoder& e, std::vector<int32_t>& out,
                      EncodeScratch& sc, WordCache& words, bool at_start) {
    Pieces& p = sc.p;
    if (p.buf.size() > (size_t)(UINT32_MAX / 8)) {
        RAD_ERR("tokenizer: a %zu-byte segment is past what one encode can index", p.buf.size());
        return RAD_E_INVAL;
    }
    p.spans.assign(1, TokSpan{ 0, (uint32_t)p.buf.size() });
    p.canonical = false;

    for (size_t k = 0; k < e.pre.size(); ++k) {
        const TokEncoder::Step& st = e.pre[k];
        switch (st.step.kind) {
            case RAD_PRE_SPLIT:
                apply_split(p, *st.re, (int)st.step.iarg, (st.step.flags & 1) != 0, sc);
                break;
            case RAD_PRE_BYTELEVEL:
                apply_bytelevel(p, st, (int)k == e.defer_bytelevel, sc);
                break;
            case RAD_PRE_METASPACE:
                rewrite(p, [&](std::vector<std::string>& w) { apply_metaspace(w, st.step, at_start); });
                break;
            case RAD_PRE_WHITESPACE:
                if (st.step.flags == kPreWhitespace) {
                    apply_split(p, *st.re, TOK_SPLIT_REMOVED, /*invert=*/true, sc);
                } else {
                    apply_split(p, *st.re, TOK_SPLIT_REMOVED, false, sc);
                    if (st.re2) apply_split(p, *st.re2, TOK_SPLIT_ISOLATED, false, sc);
                }
                break;
            case RAD_PRE_PUNCTUATION:
                apply_split(p, *st.re, (int)st.step.iarg, false, sc);
                break;
            case RAD_PRE_DIGITS:
                apply_split(p, *st.re, TOK_SPLIT_ISOLATED, false, sc);
                break;
            case RAD_PRE_BYTE_FALLBACK:
                /* Declares how an unknown piece is encoded, not a split; bpe_word handles it. */
                break;
            case RAD_PRE_END_OF_WORD:
                /* Declares how a word's last symbol is spelt, not a split; bpe_word handles it. */
                break;
            case RAD_PRE_ADDED_NORMALIZED:
                /* Cut out by encode_segment before the pre-tokenisers run. */
                break;
        }
    }

    switch (e.kind) {
        case RAD_TOK_BPE:       bpe(e, p, out, sc, words); break;
        case RAD_TOK_WORDPIECE: wordpiece(v, e, p, out); break;
        case RAD_TOK_UNIGRAM: {
            /* Unigram runs Viterbi over the whole segment; its pre-tokeniser only inserts the
             * metaspace, so the pieces are rejoined. */
            std::string joined;
            for (const TokSpan& s : p.spans) joined += p.at(s);
            unigram(v, e, joined, out);
            break;
        }
        default:
            RAD_ERR("tokenizer: kind %u has no encoder", e.kind);
            return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* One segment -- the text between two special tokens -- through the whole chain. `at_start`:
 * the segment begins the text (see apply_metaspace). The added tokens matched in the normalised
 * text are cut out of it after the normalisers, under the same `parse_special` rule as the raw
 * ones, and each piece between them goes through the rest of the chain on its own. */
int encode_segment(const Vocab& v, const TokEncoder& e, std::string_view seg,
                   std::vector<int32_t>& out, EncodeScratch& sc, WordCache& words, bool at_start,
                   bool parse_special) {
    Pieces& p = sc.p;
    p.buf.assign(seg.data(), seg.size());
    for (const TokEncoder::Step& st : e.norm) apply_norm(p.buf, st);
    if (!e.has_nspecials) return encode_normalized(v, e, out, sc, words, at_start);

    const std::string norm = p.buf;
    std::vector<SpecialFrag> frags;
    e.nspecials.split(norm, parse_special, frags);
    for (const SpecialFrag& f : frags) {
        if (f.id >= 0) { out.push_back(f.id); continue; }
        if (f.len == 0) continue;
        p.buf.assign(norm, f.off, f.len);
        RAD_TRY(encode_normalized(v, e, out, sc, words, at_start && f.off == 0));
    }
    return RAD_OK;
}

}  /* namespace */

int vocab_normalize(const std::vector<VocabStep>& steps, std::string* text) {
    for (const VocabStep& s : steps) {
        if (!step_is_norm(s.kind)) continue;
        TokEncoder::Step st{ s, nullptr, nullptr };
        if (s.kind == RAD_NORM_REPLACE && (s.flags & 1)) {
            std::string why;
            if (TokRegex::compile(s.arg0, &st.re, &why) < 0) return RAD_E_UNSUPPORTED;
        }
        apply_norm(*text, st);
    }
    return RAD_OK;
}

/* ================================================================== encode */

int Tokenizer::init(std::shared_ptr<const Vocab> v) {
    if (!v || !v->ready()) {
        RAD_ERR("tokenizer: vocab is not loaded");
        return RAD_E_STATE;
    }
    v_ = std::move(v);
    if (log_level() >= Log::Debug) {
        for (const VocabStep& s : v_->steps())
            RAD_DEBUG("tokenizer: step %s", step_describe(s).c_str());
    }
    return RAD_OK;
}

int Tokenizer::encode(std::string_view text, std::vector<int32_t>& out,
                      bool add_special, bool parse_special) const {
    if (!v_ || !v_->ready()) return RAD_E_STATE;
    const TokEncoder& e = *v_->enc_;

    /* Cut the text into (raw, token, raw, ...) around special tokens before any other step, so
     * the pre-tokeniser never splits one. With parse_special off, control tokens are ordinary
     * text (see encode() in tokenizer.h). */
    std::vector<SpecialFrag> frags;
    e.specials.split(text, parse_special, frags);

    /* No second BOS when the text already opens with one (a client-rendered chat template); a
     * doubled BOS is a sequence the model never saw in training. */
    if (add_special && v_->add_bos() && v_->bos() >= 0) {
        int32_t first = -1;
        for (const SpecialFrag& f : frags) {
            if (f.id < 0 && f.len == 0) continue;
            first = f.id;
            break;
        }
        if (first != v_->bos()) out.push_back(v_->bos());
    }

    EncodeScratch sc;
    WordCache words;
    for (const SpecialFrag& f : frags) {
        if (f.id >= 0) { out.push_back(f.id); continue; }
        if (f.len == 0) continue;
        const std::string_view seg = text.substr(f.off, f.len);
        /* The cache is keyed on the segment's bytes, so a segment whose ids also depend on
         * starting the text bypasses it. */
        const bool at_start = f.off == 0;
        /* A segment's ids also depend on parse_special when added tokens are matched in its
         * normalised text; the cache holds only the parse_special form. */
        if (!use_cache_ || seg.size() < kSegmentCacheMin || (at_start && e.prepend_first) ||
            (e.has_nspecials && !parse_special)) {
            RAD_TRY(encode_segment(*v_, e, seg, out, sc, words, at_start, parse_special));
            continue;
        }
        const uint64_t h = tok_hash(seg);
        if (e.cache.lookup(seg, h, out)) continue;
        const size_t start = out.size();
        RAD_TRY(encode_segment(*v_, e, seg, out, sc, words, at_start, parse_special));
        e.cache.insert(seg, h, out.data() + start, out.size() - start);
    }

    if (add_special && v_->add_eos() && v_->eos() >= 0) out.push_back(v_->eos());
    return RAD_OK;
}

Tokenizer::CacheStats Tokenizer::cache_stats() const {
    return v_ && v_->enc_ ? v_->enc_->cache.stats() : CacheStats{};
}

void Tokenizer::set_cache_limit(size_t bytes) const {
    if (v_ && v_->enc_) v_->enc_->cache.set_limit(bytes);
}

void Tokenizer::clear_cache() const {
    if (v_ && v_->enc_) v_->enc_->cache.clear();
}

/* ================================================================== decode */

std::string Tokenizer::piece(int32_t id, bool render_special) const {
    return piece_at(id, render_special, At{});
}

namespace {

/* HuggingFace's WordPiece `cleanup`, on one token: spaced punctuation and contractions closed. */
void wordpiece_cleanup(std::string& s) {
    static const char* const kPairs[][2] = {
        { " .", "." }, { " ?", "?" }, { " !", "!" }, { " ,", "," }, { " ' ", "'" },
        { " n't", "n't" }, { " 'm", "'m" }, { " do not", " don't" }, { " 's", "'s" },
        { " 've", "'ve" }, { " 're", "'re" },
    };
    for (const auto& p : kPairs) replace_all(s, p[0], p[1]);
}

/* A Strip after Fuse strips the decoded text as a whole; HuggingFace's Fuse joins the tokens into
 * one before it. Returns how many leading copies of `*content` it strips, 0 when there is none.
 * Vocab::load refuses one that strips the end, which a stream cannot take back. */
int fused_strip(const Vocab& v, std::string* content) {
    bool fused = false;
    for (const VocabStep& st : v.steps()) {
        if (st.kind == RAD_DEC_FUSE) fused = true;
        if (st.kind != RAD_DEC_STRIP || !fused) continue;
        *content = st.arg0.empty() ? std::string(" ") : st.arg0;
        return (int)(st.iarg & 0xFFFFFFFF);
    }
    return 0;
}

/* Up to `*left` leading copies of `c` removed from `s`; `*left` drops to 0 once `s` shows a
 * character that is not `c`, since HuggingFace's Strip stops there. */
void strip_leading(std::string& s, const std::string& c, int* left) {
    size_t at = 0;
    while (*left > 0 && s.size() - at >= c.size() && s.compare(at, c.size(), c) == 0) {
        at += c.size();
        --*left;
    }
    s.erase(0, at);
    if (!s.empty()) *left = 0;
}

/* Whether a decode leaves the token out, as HuggingFace's skip_special_tokens does. */
bool left_out(const Vocab& v, int32_t id, bool render_special) {
    const uint8_t t = v.type(id);
    return !render_special && (t == RAD_TT_CONTROL || t == RAD_TT_UNKNOWN);
}

bool has_bpe_suffix(const Vocab& v) {
    for (const VocabStep& st : v.steps())
        if (st.kind == RAD_DEC_REPLACE && (st.flags & kDecBpeSuffix)) return true;
    return false;
}

}  /* namespace */

std::string Tokenizer::piece_at(int32_t id, bool render_special, At at) const {
    if (!v_ || !v_->ready()) return {};
    if (id < 0 || id >= v_->n_tokens()) return {};

    const uint8_t t = v_->type(id);
    if (!render_special && (t == RAD_TT_CONTROL || t == RAD_TT_UNKNOWN)) return {};

    /* A byte token is one byte and skips the decoder chain; ByteLevel over "<0x0A>" would give
     * the literal text. */
    const int32_t b = v_->token_byte(id);
    if (b >= 0) return std::string(1, (char)b);

    std::string s = v_->text(id);
    /* Control tokens are stored as literal text. A user-defined (added, non-special) token goes
     * through the decoder chain like a normal one, as HuggingFace decodes every token it keeps:
     * Gemma's whitespace runs are added tokens spelled in U+2581 ("\u2581\u2581" is two spaces). */
    if (t == RAD_TT_CONTROL || t == RAD_TT_UNKNOWN) return s;

    /* No declared decoder means the tokens are already text (true for unigram and wordpiece, not
     * byte-level BPE). rad-convert emits the chain it read, so the absence is declared. */
    bool fused = false;
    for (const VocabStep& st : v_->steps()) {
        if (!step_is_dec(st.kind)) continue;
        switch (st.kind) {
            case RAD_DEC_BYTELEVEL: s = byte_decode(s); break;
            case RAD_DEC_METASPACE:
                replace_all(s, st.arg0.empty() ? std::string_view(k_metaspace) : st.arg0,
                            at.first && st.iarg != 0 ? "" : " ");
                break;
            case RAD_DEC_REPLACE:
                if (st.flags & kDecWordPiece) {
                    if (!at.first) {
                        if (!st.arg0.empty() && s.compare(0, st.arg0.size(), st.arg0) == 0)
                            s.erase(0, st.arg0.size());
                        else
                            s.insert(0, " ");
                    }
                    if (st.flags & kDecWordPieceCleanup) wordpiece_cleanup(s);
                } else if (st.flags & kDecBpeSuffix) {
                    replace_all(s, st.arg0, at.last ? "" : " ");
                } else {
                    replace_all(s, st.arg0, st.arg1);
                }
                break;
            case RAD_DEC_STRIP: {
                /* After Fuse it strips the whole text once (fused_strip), not every token. */
                if (fused) break;
                const std::string_view c = st.arg0.empty() ? std::string_view(" ") : st.arg0;
                int32_t start = (int32_t)(st.iarg & 0xFFFFFFFF);
                int32_t stop  = (int32_t)(st.iarg >> 32);
                while (start-- > 0 && s.size() >= c.size() && s.compare(0, c.size(), c) == 0)
                    s.erase(0, c.size());
                while (stop-- > 0 && s.size() >= c.size() &&
                       s.compare(s.size() - c.size(), c.size(), c) == 0)
                    s.erase(s.size() - c.size());
                break;
            }
            case RAD_DEC_FUSE:
                /* Fuse concatenates pieces, which streaming does anyway; it changes only how a
                 * later Strip applies. */
                fused = true;
                break;
            case RAD_DEC_BYTE_FALLBACK:
                /* Reassembling "<0xNN>" runs is Detokenizer's UTF-8 buffering. */
                break;
        }
    }
    return s;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool render_special) const {
    std::string out;
    if (!v_ || !v_->ready()) return out;
    size_t last = ids.size();
    for (size_t i = ids.size(); i-- > 0; )
        if (!left_out(*v_, ids[i], render_special)) { last = i; break; }
    bool first = true;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (left_out(*v_, ids[i], render_special)) continue;
        out += piece_at(ids[i], render_special, At{ first, i == last });
        first = false;
    }
    std::string content;
    int left = fused_strip(*v_, &content);
    if (left > 0) strip_leading(out, content, &left);
    return out;
}

/* ================================================================== Detokenizer */

Detokenizer::Detokenizer(std::shared_ptr<const Vocab> v, std::vector<std::string> stops,
                         bool render_special, const std::vector<std::string>& preserved)
    : v_(std::move(v)), stops_(std::move(stops)), render_special_(render_special) {
    /* `preserved` holds the markers the chat parser scans for (`<function name="`, `</param>`,
     * `<think>`), which are not necessarily tokens: MiniCPM5 spells the first as the control
     * token `<function` plus text. So each marker is tokenised with parse_special on, and the
     * control tokens it is built from are preserved; no other control token is. */
    if (v_ && v_->ready() && !preserved.empty()) {
        Tokenizer t;
        if (t.init(v_) >= 0) {
            std::vector<int32_t> ids;
            for (const std::string& s : preserved) {
                if (s.empty()) continue;
                ids.clear();
                if (t.encode(s, ids, /*add_special=*/false, /*parse_special=*/true) < 0) continue;
                for (int32_t id : ids) {
                    /* The turn terminator is preserved too: a derived format can fold it into its
                     * last marker (MiniCPM5's `</param></function><|im_end|>`), and without it the
                     * parser's grammar never completes. The parser consumes it on a match;
                     * core/server strips it on a miss. */
                    if (v_->type(id) == RAD_TT_CONTROL) preserved_.insert(id);
                }
            }
        }
    }
    /* Empty stops would make every position a match and hold the whole stream. */
    stops_.erase(std::remove_if(stops_.begin(), stops_.end(),
                                [](const std::string& s) { return s.empty(); }),
                 stops_.end());
    for (const std::string& s : stops_) max_stop_ = std::max(max_stop_, s.size());
    if (v_ && v_->ready()) {
        strip_left_ = fused_strip(*v_, &strip_);
        hold_space_ = has_bpe_suffix(*v_);
    }
}

/* How many bytes of pending_ are final: the smaller of two cut points, before
 *   1. a trailing incomplete UTF-8 sequence (byte-level BPE can deliver a codepoint one byte per
 *      push), and
 *   2. a trailing proper prefix of a stop string ("</tool_call>" arrives as "</", "tool", ...). */
size_t Detokenizer::safe_prefix() const {
    size_t cut = pending_.size();

    /* (1) walk back at most 3 bytes to find the start of a trailing partial sequence. */
    size_t i = pending_.size();
    size_t back = 0;
    while (i > 0 && back < 4) {
        --i; ++back;
        const unsigned char c = (unsigned char)pending_[i];
        if ((c & 0xC0) == 0x80) continue;              /* continuation, keep walking back */
        const size_t need = unicode_len_utf8((char)c);
        if (need > pending_.size() - i) cut = i;       /* the sequence is not all here yet */
        break;
    }

    /* (2) the longest suffix of pending_ that is a proper prefix of some stop string. */
    if (!stops_.empty()) {
        const size_t limit = std::min(pending_.size(), max_stop_ - 1);
        for (size_t n = limit; n > 0; --n) {
            const char* tail = pending_.data() + pending_.size() - n;
            bool is_prefix = false;
            for (const std::string& s : stops_) {
                if (s.size() > n && std::memcmp(s.data(), tail, n) == 0) { is_prefix = true; break; }
            }
            if (is_prefix) { cut = std::min(cut, pending_.size() - n); break; }
        }
    }

    /* (3) a space a BPEDecoder suffix may have produced on what turns out to be the last token. */
    if (hold_space_ && !pending_.empty() && pending_.back() == ' ')
        cut = std::min(cut, pending_.size() - 1);
    return cut;
}

/* Replacement follows WHATWG, as browsers do: one U+FFFD per maximal subpart, so a truncated
 * three-byte character becomes one replacement character, not three.
 *
 * The well-formed sequence at s[i], a non-ASCII byte: its length, or 0 with `*part` the length
 * of the ill-formed one's maximal subpart (never zero). The second byte's range rejects overlong
 * encodings, surrogates and anything past U+10FFFF without decoding. */
static inline size_t utf8_seq(std::string_view s, size_t i, size_t* part) {
    const unsigned char c = (unsigned char)s[i];
    size_t len; unsigned char lo, hi;
    if      (c >= 0xC2 && c <= 0xDF) { len = 2; lo = 0x80; hi = 0xBF; }
    else if (c == 0xE0)              { len = 3; lo = 0xA0; hi = 0xBF; }
    else if (c >= 0xE1 && c <= 0xEC) { len = 3; lo = 0x80; hi = 0xBF; }
    else if (c == 0xED)              { len = 3; lo = 0x80; hi = 0x9F; }  /* no surrogates */
    else if (c >= 0xEE && c <= 0xEF) { len = 3; lo = 0x80; hi = 0xBF; }
    else if (c == 0xF0)              { len = 4; lo = 0x90; hi = 0xBF; }
    else if (c >= 0xF1 && c <= 0xF3) { len = 4; lo = 0x80; hi = 0xBF; }
    else if (c == 0xF4)              { len = 4; lo = 0x80; hi = 0x8F; }  /* <= U+10FFFF */
    else                             { len = 0; lo = 0; hi = 0; }        /* 0x80-0xC1, 0xF5+ */

    size_t k = 1;
    if (len) {
        for (; k < len && i + k < s.size(); ++k) {
            const unsigned char d = (unsigned char)s[i + k];
            const unsigned char l = (k == 1) ? lo : 0x80, h = (k == 1) ? hi : 0xBF;
            if (d < l || d > h) break;
        }
    }
    if (len && k == len) return len;
    *part = k;
    return 0;
}

std::string utf8_sanitize(std::string_view s) {
    /* Valid runs are appended in bulk, not byte by byte. */
    std::string out;
    size_t i = 0, good = 0;                  /* [good, i) is verified and not yet appended */
    const size_t n = s.size();
    auto flush_good = [&] { out.append(s.data() + good, i - good); good = i; };

    while (i < n) {
        /* ASCII is copied verbatim either way, so jump straight to the next non-ASCII byte
         * (simd::first_non_ascii) instead of testing bytes one at a time. */
        if ((unsigned char)s[i] < 0x80) { i += simd::first_non_ascii(s.data() + i, n - i); continue; }
        size_t part = 0;
        const size_t len = utf8_seq(s, i, &part);
        if (len) { i += len; continue; }

        flush_good();
        out += "\xEF\xBF\xBD";               /* U+FFFD */
        i += part;                           /* the maximal subpart, never zero */
        good = i;
    }
    flush_good();
    return out;
}

size_t utf8_valid_prefix(std::string_view s) {
    size_t i = 0, part = 0;
    while (i < s.size()) {
        if ((unsigned char)s[i] < 0x80) { i += simd::first_non_ascii(s.data() + i, s.size() - i); continue; }
        const size_t len = utf8_seq(s, i, &part);
        if (!len) break;
        i += len;
    }
    return i;
}

std::string Detokenizer::push(int32_t token) {
    if (!v_ || !v_->ready() || stopped_) return {};

    /* One Tokenizer per stream, set up on the first push, to avoid a shared_ptr refcount round
     * trip per token; piece() depends only on the vocab. */
    if (!tok_ready_) {
        if (tok_.init(v_) < 0) return {};
        tok_ready_ = true;
    }
    const bool render = render_special_ || (!preserved_.empty() && preserved_.count(token) != 0);
    /* The stream is decoded as HuggingFace decodes the whole list (Tokenizer::piece_at): the
     * first token that is not left out reads as the first. A Strip after Fuse strips the stream's
     * start. */
    const bool counts = !left_out(*v_, token, render);
    std::string p = tok_.piece_at(token, render, Tokenizer::At{ first_ && counts, false });
    if (counts) first_ = false;
    if (strip_left_ > 0 && !p.empty()) strip_leading(p, strip_, &strip_left_);

    all_ += p;
    pending_ += p;

    /* Inside the reasoning block no stop string applies (defer_stops_until). The closer is
     * searched for incrementally in all_, from (closer length - 1) bytes before the previous end.
     * Once found, matching starts after it; the closer may still be in pending_. */
    size_t from = 0;
    if (!armed_) {
        const size_t n = arm_after_.size();
        const size_t start = arm_scanned_ >= n ? arm_scanned_ - n + 1 : 0;
        const size_t m = all_.find(arm_after_, start);
        arm_scanned_ = all_.size();
        if (m == std::string::npos) {
            from = std::string::npos;
        } else {
            armed_ = true;
            const size_t pending_at = all_.size() - pending_.size();
            from = m + n > pending_at ? m + n - pending_at : 0;
        }
    }

    /* A complete stop string in pending_ ends the stream; only the text before it is emitted. */
    if (!stops_.empty() && from != std::string::npos) {
        size_t best = std::string::npos;
        const std::string* which = nullptr;
        for (const std::string& s : stops_) {
            const size_t m = pending_.find(s, from);
            if (m != std::string::npos && m < best) { best = m; which = &s; }
        }
        if (which) {
            std::string out = pending_.substr(0, best);
            pending_.clear();
            stopped_ = true;
            stop_hit_ = *which;
            return utf8_sanitize(out);
        }
    }

    const size_t cut = safe_prefix();
    if (cut == 0) return {};
    /* Sanitised straight out of pending_, which keeps its storage for the next token. */
    std::string out = utf8_sanitize(std::string_view(pending_).substr(0, cut));
    pending_.erase(0, cut);
    return out;
}

std::string Detokenizer::flush() {
    /* End of stream: release everything held back. An incomplete UTF-8 tail (a length limit can
     * stop mid-codepoint) leaves as U+FFFD, since raw invalid bytes cannot go into a JSON
     * string. */
    std::string out;
    out.swap(pending_);
    /* The held space was a suffix on the last token, which HuggingFace's BPEDecoder drops. */
    if (hold_space_ && !out.empty() && out.back() == ' ') out.pop_back();
    return utf8_sanitize(out);
}

}  /* namespace rad */
