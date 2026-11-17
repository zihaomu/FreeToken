/* gbnf.cpp -- the GBNF grammar state machine, lifted from llama.cpp `src/llama-grammar.cpp` at
 * upstream commit 06938ac12 (MIT). See gbnf.h for the list of changes, and gbnf_mask.cpp for how a
 * token mask is produced.
 */
#include "gbnf.h"
#include "gbnf_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <new>
#include <set>
#include <unordered_map>
#include <stdexcept>

namespace rad {

using gbnf::GbnfLimit;
using gbnf::GbnfRuleSet;
using gbnf::OpWork;
using gbnf::SetBuild;
using gbnf::advance_stack;
using gbnf::decode_utf8;
using gbnf::is_end_of_sequence;
using gbnf::match_char;
using gbnf::match_token;
using gbnf::kMaxCompileBytes;
using gbnf::kMaxNesting;

namespace gbnf {

void throw_set_limit() {
    throw GbnfLimit("the grammar's parse forks into more than " + std::to_string(kMaxStacks) +
                    " stacks or " + std::to_string(kMaxSetElems) +
                    " stack elements at one position");
}

void OpWork::over() const {
    throw GbnfLimit("the grammar needs more than " + std::to_string(limit) +
                    " units of stack work at one position");
}

uint64_t stack_hash(const GbnfStack& s) {
    uint64_t h = 1469598103934665603ull;
    for (const GbnfElement* e : s) { h ^= (uint64_t)(uintptr_t)e; h *= 1099511628211ull; }
    return h ^ s.size();
}

void SetBuild::insert(GbnfStacks& out, GbnfStack s) {
    work.add(s.size() + kStackCost / 4);
    if (index.empty() && out.size() < kIndexFrom) {
        if (std::find(out.begin(), out.end(), s) != out.end()) return;
    } else {
        if (index.empty())
            for (size_t i = 0; i < out.size(); ++i) index.emplace(stack_hash(out[i]), i);
        const uint64_t h = stack_hash(s);
        const auto range = index.equal_range(h);
        for (auto it = range.first; it != range.second; ++it)
            if (out[it->second] == s) return;
        index.emplace(h, out.size());
    }
    add(s.size());
    out.push_back(std::move(s));
}

/* ================================================================== UTF-8 */

std::pair<uint32_t, const char*> decode_utf8(const char* src) {
    static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    const uint8_t first_byte = (uint8_t)*src;
    const uint8_t highbits   = first_byte >> 4;
    const int     len        = lookup[highbits];
    const uint8_t mask       = (uint8_t)((1 << (8 - len)) - 1);
    uint32_t      value      = first_byte & mask;
    const char*   end        = src + len;   /* may overrun */
    const char*   pos        = src + 1;
    for (; pos < end && *pos; pos++) value = (value << 6) + ((uint8_t)*pos & 0x3F);
    return { value, pos };
}

std::pair<std::vector<uint32_t>, PartialUtf8> decode_utf8(const std::string& src,
                                                         PartialUtf8 partial_start) {
    std::vector<uint32_t> cps;
    const PartialUtf8 p = decode_utf8(src, partial_start, &cps);
    return { std::move(cps), p };
}

PartialUtf8 decode_utf8(const std::string& src, PartialUtf8 partial_start,
                        std::vector<uint32_t>* out) {
    static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };
    const char* pos = src.c_str();
    std::vector<uint32_t>& cps = *out;
    cps.clear();
    cps.reserve(src.size() + 1);

    uint32_t value    = partial_start.value;
    int      n_remain = partial_start.n_remain;

    while (*pos != 0 && n_remain > 0) {
        const uint8_t next_byte = (uint8_t)*pos;
        if ((next_byte >> 6) != 2) {
            cps.push_back(0);
            return PartialUtf8{ 0, -1 };
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }
    if (partial_start.n_remain > 0 && n_remain == 0) cps.push_back(value);

    while (*pos != 0) {
        const uint8_t first_byte = (uint8_t)*pos;
        const uint8_t highbits   = first_byte >> 4;
        n_remain = lookup[highbits] - 1;
        if (n_remain < 0) {
            cps.clear();
            cps.push_back(0);
            return PartialUtf8{ 0, n_remain };
        }
        const uint8_t mask = (uint8_t)((1 << (7 - n_remain)) - 1);
        value = first_byte & mask;
        ++pos;
        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + ((uint8_t)*pos & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) cps.push_back(value);
    }
    cps.push_back(0);
    return PartialUtf8{ value, n_remain };
}

}  /* namespace gbnf */

/* ================================================================== the per-vocabulary cache
 *
 * Everything GBNF derives from a vocabulary (the vocabulary index and the programs compiled
 * against it) lives on the VocabView itself, with the vocabulary's lifetime. A global table keyed
 * on the view's pointer would serve a stale index or program to a new view allocated at the same
 * address, silently admitting the wrong token set. It also makes the program cap per vocabulary.
 */
struct GbnfProgram;

struct VocabGbnf {
    std::mutex mu;
    std::shared_ptr<const gbnf::VocabIndex> index;
    double index_ms = 0.0;

    /* One program per (text, root). The cache pins the most recently used ones (capped, since
     * keys come from clients) and remembers every other one still alive. Each request holds its
     * program from admission on (SamplingParams), so it always runs the program it was admitted
     * with and the scheduler thread never recompiles. */
    struct Cached {
        std::shared_ptr<const GbnfProgram> pinned;
        std::weak_ptr<const GbnfProgram>   alive;
        uint64_t                           used = 0;
    };
    std::map<std::pair<std::string, std::string>, Cached> programs;
    uint64_t clock = 0;
};

/* Created on first use under a process-wide lock; contents are guarded by the cache's own lock,
 * which is held for a whole vocabulary index build (slow on a quarter-million-piece vocabulary). */
static VocabGbnf& gbnf_cache_for(const VocabView* v) {
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    if (!v->gbnf_cache) v->gbnf_cache = std::make_shared<VocabGbnf>();
    return *static_cast<VocabGbnf*>(v->gbnf_cache.get());
}

static std::shared_ptr<const gbnf::VocabIndex> vocab_index_for(const VocabView* v) {
    VocabGbnf& C = gbnf_cache_for(v);
    std::lock_guard<std::mutex> lk(C.mu);
    if (!C.index) {
        const auto t0 = std::chrono::steady_clock::now();
        C.index = gbnf::vocab_index_build(*v);
        C.index_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count();
    }
    return C.index;
}

int gbnf_prepare_vocab(const VocabView* vocab, GbnfVocabInfo* info) {
    if (!vocab) return RAD_E_INVAL;
    try {
        const auto x = vocab_index_for(vocab);
        if (info) {
            VocabGbnf& C = gbnf_cache_for(vocab);
            std::lock_guard<std::mutex> lk(C.mu);
            info->build_ms = C.index_ms;
            info->bytes    = gbnf::vocab_index_bytes(*x);
        }
    } catch (const std::bad_alloc&) {
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

/* ================================================================== lexing helpers */

/* Where a parse error is, for its message: the rest of the line, at most 40 bytes. Not the rest
 * of the grammar, which can be megabytes for a generated one. The cut backs up to a character
 * boundary: the message reaches a JSON error body, whose encoder refuses broken UTF-8. */
static std::string where(const char* p) {
    size_t n = 0;
    while (p[n] && p[n] != '\n' && p[n] != '\r' && n < 40) ++n;
    const bool cut = p[n] && p[n] != '\n' && p[n] != '\r';
    if (cut)
        while (n > 0 && ((unsigned char)p[n] & 0xC0u) == 0x80u) --n;
    std::string s(p, n);
    if (cut) s += "...";
    return s;
}

static bool is_digit_char(char c) { return '0' <= c && c <= '9'; }
static bool is_word_char(char c) {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || is_digit_char(c);
}

static std::pair<uint32_t, const char*> parse_hex(const char* src, int size) {
    const char* pos = src;
    const char* end = src + size;
    uint32_t value = 0;
    for (; pos < end && *pos; pos++) {
        value <<= 4;
        const char c = *pos;
        if      ('a' <= c && c <= 'f') value += (uint32_t)(c - 'a' + 10);
        else if ('A' <= c && c <= 'F') value += (uint32_t)(c - 'A' + 10);
        else if ('0' <= c && c <= '9') value += (uint32_t)(c - '0');
        else break;
    }
    if (pos != end)
        throw std::runtime_error("expecting " + std::to_string(size) + " hex chars at " + where(src));
    return { value, pos };
}

static const char* parse_space(const char* src, bool newline_ok) {
    const char* pos = src;
    while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
           (newline_ok && (*pos == '\r' || *pos == '\n'))) {
        if (*pos == '#') { while (*pos && *pos != '\r' && *pos != '\n') pos++; }
        else pos++;
    }
    return pos;
}

static const char* parse_name(const char* src) {
    const char* pos = src;
    while (is_word_char(*pos)) pos++;
    if (pos == src) throw std::runtime_error(std::string("expecting name at ") + where(src));
    return pos;
}

static const char* parse_int(const char* src) {
    const char* pos = src;
    while (is_digit_char(*pos)) pos++;
    if (pos == src) throw std::runtime_error(std::string("expecting integer at ") + where(src));
    return pos;
}

/* The repetition count in [src, end), already checked to be digits. A count past 2^64 - 2 is
 * refused here (UINT64_MAX means "no maximum"); smaller ones are judged by the compile budget in
 * handle_repetitions. */
static uint64_t parse_count(const char* src, const char* end) {
    uint64_t v = 0;
    for (const char* p = src; p < end; ++p)
        if (__builtin_mul_overflow(v, 10u, &v) ||
            __builtin_add_overflow(v, (uint64_t)(*p - '0'), &v) || v == UINT64_MAX)
            throw std::runtime_error("repetition count " + std::string(src, (size_t)(end - src)) +
                                     " is past what any grammar can expand to");
    return v;
}

static std::pair<uint32_t, const char*> parse_char(const char* src) {
    if (*src == '\\') {
        switch (src[1]) {
            case 'x': return parse_hex(src + 2, 2);
            case 'u': return parse_hex(src + 2, 4);
            case 'U': return parse_hex(src + 2, 8);
            case 't': return { (uint32_t)'\t', src + 2 };
            case 'r': return { (uint32_t)'\r', src + 2 };
            case 'n': return { (uint32_t)'\n', src + 2 };
            case '\\': case '"': case '[': case ']':
                return { (uint32_t)(uint8_t)src[1], src + 2 };
            default:
                throw std::runtime_error(std::string("unknown escape at ") + where(src));
        }
    } else if (*src) {
        return decode_utf8(src);
    }
    throw std::runtime_error("unexpected end of input");
}

/* `<[id]>` is a literal token id; `<text>` resolves through the vocabulary and must tokenise to
 * exactly one token, since a rule element is one terminal. */
static std::pair<uint32_t, const char*> parse_token(const VocabView* vocab, const char* src) {
    const char* pos = src;
    if (*pos != '<') throw std::runtime_error(std::string("expecting '<' at ") + where(pos));
    pos++;

    if (*pos == '[') {
        pos++;
        const char* int_end = parse_int(pos);
        const uint32_t id = (uint32_t)std::stoul(std::string(pos, (size_t)(int_end - pos)));
        pos = int_end;
        if (*pos != ']') throw std::runtime_error(std::string("expecting ']' at ") + where(pos));
        pos++;
        if (*pos != '>') throw std::runtime_error(std::string("expecting '>' at ") + where(pos));
        pos++;
        return { id, pos };
    }

    if (!vocab) throw std::runtime_error(std::string("no vocabulary to parse token at ") + where(src));

    while (*pos != 0 && *pos != '>') pos++;
    if (*pos != '>') throw std::runtime_error(std::string("expecting '>' at ") + where(pos));
    pos++;

    std::vector<int32_t> toks;
    const std::string text(src, (size_t)(pos - src));
    if (vocab->tokenize(text, &toks) < 0 || toks.size() != 1)
        throw std::runtime_error("invalid token '" + text + "' (must be exactly one token)");
    return { (uint32_t)toks[0], pos };
}

/* ================================================================== the parser */

/* Bytes a symbol costs beyond its name: its map node and its rule slot. */
static constexpr uint64_t kSymbolOverhead = 64 + sizeof(GbnfRule);

/* Written as a subtraction so a huge `bytes` (a client count times a size) cannot wrap the sum;
 * budget_used_ never exceeds the budget, so the subtraction cannot underflow. */
void GbnfParser::charge(uint64_t bytes) {
    if (bytes > kMaxCompileBytes - budget_used_)
        throw std::runtime_error("the grammar expands past " +
                                 std::to_string(kMaxCompileBytes >> 20) + " MiB when compiled; "
                                 "reduce its repetition counts or the size of what they repeat");
    budget_used_ += bytes;
}

uint32_t GbnfParser::get_symbol_id(const char* src, size_t len) {
    const uint32_t next_id = (uint32_t)symbol_ids_.size();
    auto r = symbol_ids_.emplace(std::string(src, len), next_id);
    if (r.second) charge(len + kSymbolOverhead);
    return r.first->second;
}

/* A generated name repeats its base, so many generated rules under a long name amplify the text;
 * charged accordingly. */
uint32_t GbnfParser::generate_symbol_id(const std::string& base) {
    charge(base.size() + 12 + kSymbolOverhead);
    const uint32_t next_id = (uint32_t)symbol_ids_.size();
    symbol_ids_[base + '_' + std::to_string(next_id)] = next_id;
    return next_id;
}

void GbnfParser::add_rule(uint32_t id, const GbnfRule& rule) {
    if (rules_.size() <= id) rules_.resize(id + 1);
    rules_[id] = rule;
}

const char* GbnfParser::parse_alternates(const char* src, const std::string& rule_name,
                                         uint32_t rule_id, bool is_nested) {
    GbnfRule rule;
    const char* pos = parse_sequence(src, rule_name, rule, is_nested);
    while (*pos == '|') {
        rule.push_back({ GRE_ALT, 0 });
        pos = parse_space(pos + 1, true);
        pos = parse_sequence(pos, rule_name, rule, is_nested);
    }
    rule.push_back({ GRE_END, 0 });
    add_rule(rule_id, rule);
    return pos;
}

const char* GbnfParser::parse_sequence(const char* src, const std::string& rule_name,
                                       GbnfRule& rule, bool is_nested) {
    size_t last_sym_start = rule.size();
    const char* pos = src;

    /* S{m,n} is rewritten into rules rather than counted at match time, so the machine stays a
     * pushdown automaton without counters:
     *   S{m,n} -> S S ... (m times) S'(n-m),  S'(x) ::= S S'(x-1) |
     *   S{m,}  -> S S ... (m times) S',       S'    ::= S S' |
     *   S*     -> S{0,}     S+ -> S{1,}       S? -> S{0,1}
     *
     * The expansion is bounded by its size, charged to the compile budget before it is built, not
     * by a repetition count (upstream's cap of 2000 rejects ordinary schemas such as
     * `maxLength: 8000`). It is linear in the count, and nothing after the parse is worse than
     * linear in the rules; the mask plan gives every link past the longest token one class. So
     * `[^"]{0,8000}` costs about a megabyte and yields the same plan as `[^"]{0,1999}`.
     *
     * S'(x) depends only on S and x, so the links are shared: every bounded repetition of the same
     * operand points into one chain (rep_links_), and S{0,2000} after S{0,4000} builds nothing. */
    auto handle_repetitions = [&](uint64_t min_times, uint64_t max_times) {
        const bool no_max = max_times == UINT64_MAX;
        if (last_sym_start == rule.size())
            throw std::runtime_error(std::string("expecting preceding item to */+/?/{ at ") + where(pos));

        /* `n_opt` below is max - min, which a reversed range would wrap to ~1.8e19 rules. */
        if (!no_max && max_times < min_times)
            throw std::runtime_error("repetition {" + std::to_string(min_times) + "," +
                                     std::to_string(max_times) + "} has its minimum above its "
                                     "maximum");

        const GbnfRule prev_rule(rule.begin() + (std::ptrdiff_t)last_sym_start, rule.end());

        /* A refusal names the repetition the client wrote. */
        try {
            const uint64_t n_opt = no_max ? 1 : max_times - min_times;

            /* A bounded repetition's chain of links for this operand, shared with every earlier
             * bounded repetition of it; only the links it lacks are built. */
            std::vector<uint32_t>* links = nullptr;
            uint64_t n_new = n_opt;
            if (!no_max && n_opt > 0) {
                std::vector<std::pair<uint32_t, uint32_t>> key;
                key.reserve(prev_rule.size());
                for (const auto& e : prev_rule) key.emplace_back((uint32_t)e.type, e.value);
                links = &rep_links_[std::move(key)];
                n_new = n_opt > links->size() ? n_opt - links->size() : 0;
            }

            /* Charge the copies up front: the operand once per required count beyond the first,
             * and once per optional rule built plus its reference, ALT and END. Counts reach
             * 2^64 - 2, so overflow saturates and is refused before the loops run. Each generated
             * rule's name and slot are charged again in generate_symbol_id. */
            {
                const uint64_t w      = prev_rule.size();
                uint64_t req = 0, opt = 0, copies = 0, bytes = 0;
                const bool wraps =
                    __builtin_mul_overflow(min_times > 1 ? min_times - 1 : 0, w, &req) ||
                    __builtin_mul_overflow(n_new, w + 3, &opt) ||
                    __builtin_add_overflow(req, opt, &copies) ||
                    __builtin_mul_overflow(copies, (uint64_t)sizeof(GbnfElement), &bytes);
                charge(wraps ? UINT64_MAX : bytes);
            }

            if (min_times == 0) {
                rule.resize(last_sym_start);
            } else {
                for (uint64_t i = 1; i < min_times; i++)
                    rule.insert(rule.end(), prev_rule.begin(), prev_rule.end());
            }

            GbnfRule rec_rule(prev_rule);
            if (no_max) {
                const uint32_t rec_rule_id = generate_symbol_id(rule_name);
                rec_rule.push_back({ GRE_RULE_REF, rec_rule_id });
                rec_rule.push_back({ GRE_ALT, 0 });
                rec_rule.push_back({ GRE_END, 0 });
                add_rule(rec_rule_id, rec_rule);
                rule.push_back({ GRE_RULE_REF, rec_rule_id });
            } else if (n_opt > 0) {
                while (links->size() < n_opt) {
                    rec_rule.resize(prev_rule.size());
                    const uint32_t rec_rule_id = generate_symbol_id(rule_name);
                    if (!links->empty()) rec_rule.push_back({ GRE_RULE_REF, links->back() });
                    rec_rule.push_back({ GRE_ALT, 0 });
                    rec_rule.push_back({ GRE_END, 0 });
                    add_rule(rec_rule_id, rec_rule);
                    links->push_back(rec_rule_id);
                }
                rule.push_back({ GRE_RULE_REF, (*links)[n_opt - 1] });
            }
        } catch (const std::runtime_error& ex) {
            throw std::runtime_error("the repetition {" + std::to_string(min_times) + "," +
                                     (no_max ? std::string() : std::to_string(max_times)) + "}: " +
                                     ex.what());
        }
    };

    while (*pos) {
        if (*pos == '"') {                                   /* literal string */
            pos++;
            last_sym_start = rule.size();
            while (*pos != '"') {
                if (!*pos) throw std::runtime_error("unexpected end of input");
                auto cp = parse_char(pos);
                pos = cp.second;
                rule.push_back({ GRE_CHAR, cp.first });
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '[') {                            /* char range(s) */
            pos++;
            GreType start_type = GRE_CHAR;
            if (*pos == '^') { pos++; start_type = GRE_CHAR_NOT; }
            last_sym_start = rule.size();
            while (*pos != ']') {
                if (!*pos) throw std::runtime_error("unexpected end of input");
                auto cp = parse_char(pos);
                pos = cp.second;
                const GreType type = last_sym_start < rule.size() ? GRE_CHAR_ALT : start_type;
                rule.push_back({ type, cp.first });
                if (pos[0] == '-' && pos[1] != ']') {
                    if (!pos[1]) throw std::runtime_error("unexpected end of input");
                    auto hi = parse_char(pos + 1);
                    pos = hi.second;
                    rule.push_back({ GRE_CHAR_RNG_UPPER, hi.first });
                }
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '<' || *pos == '!') {             /* token terminal */
            GreType type = GRE_TOKEN;
            if (*pos == '!') { type = GRE_TOKEN_NOT; pos++; }
            auto tp = parse_token(vocab_, pos);
            last_sym_start = rule.size();
            rule.push_back({ type, tp.first });
            pos = parse_space(tp.second, is_nested);
        } else if (is_word_char(*pos)) {                     /* rule reference */
            const char* name_end = parse_name(pos);
            const uint32_t ref = get_symbol_id(pos, (size_t)(name_end - pos));
            pos = parse_space(name_end, is_nested);
            last_sym_start = rule.size();
            rule.push_back({ GRE_RULE_REF, ref });
        } else if (*pos == '(') {                            /* grouping */
            if (++depth_ > kMaxNesting)
                throw std::runtime_error("parentheses nest more than " +
                                         std::to_string(kMaxNesting) + " deep");
            pos = parse_space(pos + 1, true);
            const uint32_t sub_rule_id = generate_symbol_id(rule_name);
            pos = parse_alternates(pos, rule_name, sub_rule_id, true);
            --depth_;
            last_sym_start = rule.size();
            rule.push_back({ GRE_RULE_REF, sub_rule_id });
            if (*pos != ')') throw std::runtime_error(std::string("expecting ')' at ") + where(pos));
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '.') {                            /* any char */
            last_sym_start = rule.size();
            rule.push_back({ GRE_CHAR_ANY, 0 });
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '*') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, UINT64_MAX);
        } else if (*pos == '+') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(1, UINT64_MAX);
        } else if (*pos == '?') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, 1);
        } else if (*pos == '{') {
            pos = parse_space(pos + 1, is_nested);
            if (!is_digit_char(*pos))
                throw std::runtime_error(std::string("expecting an int at ") + where(pos));
            const char* int_end = parse_int(pos);
            const uint64_t min_times = parse_count(pos, int_end);
            pos = parse_space(int_end, is_nested);

            uint64_t max_times = UINT64_MAX;
            if (*pos == '}') {
                max_times = min_times;
                pos = parse_space(pos + 1, is_nested);
            } else if (*pos == ',') {
                pos = parse_space(pos + 1, is_nested);
                if (is_digit_char(*pos)) {
                    const char* e = parse_int(pos);
                    max_times = parse_count(pos, e);
                    pos = parse_space(e, is_nested);
                }
                if (*pos != '}') throw std::runtime_error(std::string("expecting '}' at ") + where(pos));
                pos = parse_space(pos + 1, is_nested);
            } else {
                throw std::runtime_error(std::string("expecting ',' at ") + where(pos));
            }
            handle_repetitions(min_times, max_times);
        } else {
            break;
        }
    }
    return pos;
}

const char* GbnfParser::parse_rule(const char* src) {
    const char* name_end = parse_name(src);
    const char* pos      = parse_space(name_end, false);
    const size_t name_len = (size_t)(name_end - src);
    const uint32_t rule_id = get_symbol_id(src, name_len);
    const std::string name(src, name_len);

    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '='))
        throw std::runtime_error(std::string("expecting ::= at ") + where(pos));
    pos = parse_space(pos + 3, true);

    /* An error inside a rule names the rule. Grammars generated from a schema or tool list name
     * rules after the tool and property, so this points the client at the failing part. */
    try {
        pos = parse_alternates(pos, name, rule_id, false);
    } catch (const std::runtime_error& ex) {
        throw std::runtime_error("rule '" + name + "': " + ex.what());
    }

    if (*pos == '\r')      pos += pos[1] == '\n' ? 2 : 1;
    else if (*pos == '\n') pos++;
    else if (*pos)         throw std::runtime_error(std::string("expecting newline or end at ") + where(pos));
    return parse_space(pos, true);
}

int GbnfParser::parse(const char* src, std::string* err) {
    if (!src) return RAD_E_INVAL;
    budget_used_ = 0;
    rep_links_.clear();
    depth_       = 0;
    try {
        const char* pos = parse_space(src, true);
        while (*pos) pos = parse_rule(pos);

        /* Every referenced rule must exist; report the missing rule by name, as upstream does. */
        for (const auto& rule : rules_) {
            if (rule.empty()) throw std::runtime_error("undefined rule");
            for (const auto& e : rule) {
                if (e.type != GRE_RULE_REF) continue;
                if (e.value < rules_.size() && !rules_[e.value].empty()) continue;
                for (const auto& kv : symbol_ids_)
                    if (kv.second == e.value)
                        throw std::runtime_error("undefined rule identifier '" + kv.first + "'");
                throw std::runtime_error("undefined rule " + std::to_string(e.value));
            }
        }
    } catch (const std::exception& ex) {
        /* The parser throws internally, as upstream's does; a failure leaves as a status and
         * message. */
        if (err) *err = ex.what();
        rules_.clear();
        return RAD_E_FORMAT;
    }
    return RAD_OK;
}

/* ================================================================== stack machinery */

namespace gbnf {

/* Each bound is an O(1) check on the existing path; a grammar that reaches one throws GbnfLimit
 * ("what a grammar may cost", gbnf_internal.h). */
void advance_stack(const GbnfRuleSet& rules, const GbnfStack& stack, GbnfStacks& out,
                   SetBuild& build) {
    auto too_wide = [] {
        return GbnfLimit("the grammar's rule references expand past " +
                         std::to_string(kMaxClosureWork) + " stack elements at one position");
    };
    std::vector<GbnfStack> todo;
    todo.push_back(stack);
    size_t pushed = stack.size() + 1;
    build.work.add(stack.size() + kStackCost);

    while (!todo.empty()) {
        GbnfStack curr = std::move(todo.back());
        todo.pop_back();
        if (!build.seen.insert(curr).second) continue;
        build.seen_elems += curr.size() + 1;
        if (build.seen_elems > kMaxClosureWork) throw too_wide();

        if (curr.empty()) {
            build.insert(out, std::move(curr));
            continue;
        }

        const GbnfElement* pos = curr.back();
        switch (pos->type) {
            case GRE_RULE_REF: {
                const size_t rule_id = (size_t)pos->value;
                const GbnfElement* subpos = rules.rule(rule_id);
                for (;;) {
                    pushed += curr.size() + 1;
                    if (pushed > kMaxClosureWork) throw too_wide();
                    build.work.add(curr.size() + kStackCost);
                    GbnfStack next(curr.begin(), curr.end() - 1);
                    if (!is_end_of_sequence(pos + 1)) next.push_back(pos + 1);
                    if (!is_end_of_sequence(subpos))  next.push_back(subpos);
                    todo.push_back(std::move(next));
                    while (!is_end_of_sequence(subpos)) subpos++;
                    if (subpos->type == GRE_ALT) subpos++;
                    else break;
                }
                break;
            }
            case GRE_CHAR: case GRE_CHAR_NOT: case GRE_CHAR_ANY:
            case GRE_TOKEN: case GRE_TOKEN_NOT:
                build.insert(out, std::move(curr));
                break;
            default:
                /* END, ALT, CHAR_ALT and CHAR_RNG_UPPER are never a stack top; reaching one is a
                 * broken invariant, so abort. */
                fatal("gbnf: stack left on element type %d, which cannot be a stack top",
                      (int)pos->type);
        }
    }
}

}  /* namespace gbnf */

/* Left recursion makes advance_stack grow a stack forever, so it is detected up front. Returns
 * the index of a rule on such a cycle, or -1.
 *
 * A reference is in a rule's left corner if everything before it in its alternative is nullable.
 * Expanding it also pushes what follows it, so it grows the stack exactly when it is not in tail
 * position. A cycle of left-corner references therefore diverges iff one of its references is not
 * in tail position; a cycle of tail references only revisits stacks the closure's `seen` set
 * already stops (`S ::= ("a"?)*` runs).
 *
 * Nullability is a fixed point (`e ::= f` is nullable because `f ::= "b" |` is; checking only for
 * an empty alternative misses `root ::= e root "c" | "a"`). It uses a worklist and the cycle search
 * an iterative SCC pass, so both are linear and non-recursive (a 100,000-rule chain would
 * otherwise recurse 100,000 frames deep). */
static int64_t find_left_recursion(const GbnfRules& rules) {
    const size_t n = rules.size();

    /* 1. Which rules can match the empty string. An alternative with a terminal never can; one of
     *    only rule references can once all of them can, tracked by a pending count. */
    std::vector<char>     nullable(n, 0);
    std::vector<uint32_t> alt_owner, alt_pending, work;
    std::vector<std::vector<uint32_t>> waiting(n);     /* rule -> alternatives it holds up */
    for (size_t r = 0; r < n; ++r) {
        const GbnfRule& rule = rules[r];
        for (size_t i = 0; i < rule.size(); ) {
            bool     terminal = false;
            uint32_t refs     = 0;
            size_t   j        = i;
            for (; !is_end_of_sequence(&rule[j]); ++j) {
                if (rule[j].type == GRE_RULE_REF) ++refs;
                else                              terminal = true;
            }
            if (!terminal) {
                if (refs == 0) {
                    if (!nullable[r]) { nullable[r] = 1; work.push_back((uint32_t)r); }
                } else {
                    const uint32_t a = (uint32_t)alt_owner.size();
                    alt_owner.push_back((uint32_t)r);
                    alt_pending.push_back(refs);
                    for (size_t k = i; k < j; ++k) waiting[rule[k].value].push_back(a);
                }
            }
            i = j + 1;
            if (rule[j].type == GRE_END) break;
        }
    }
    while (!work.empty()) {
        const uint32_t s = work.back();
        work.pop_back();
        for (uint32_t a : waiting[s]) {
            if (--alt_pending[a] != 0) continue;
            const uint32_t r = alt_owner[a];
            if (!nullable[r]) { nullable[r] = 1; work.push_back(r); }
        }
    }
    waiting = {};

    /* 2. The left-corner graph: one edge per reference that can come first, marked `grows` when
     *    not in tail position. */
    struct Edge { uint32_t to; bool grows; };
    std::vector<size_t> edge_off(n + 1, 0);
    std::vector<Edge>   edges;
    for (size_t r = 0; r < n; ++r) {
        const GbnfRule& rule = rules[r];
        bool open = true;
        for (size_t i = 0; i < rule.size(); ++i) {
            const GbnfElement& e = rule[i];
            if (is_end_of_sequence(&e)) { open = true; continue; }
            if (!open) continue;
            if (e.type != GRE_RULE_REF) { open = false; continue; }
            edges.push_back({ e.value, !is_end_of_sequence(&rule[i + 1]) });
            if (!nullable[e.value]) open = false;
        }
        edge_off[r + 1] = edges.size();
    }

    /* 3. Strongly connected components (Tarjan's, explicit call stack). A growing edge within one
     *    component, self-loops included, is the cycle. */
    std::vector<int32_t>  idx(n, -1), low(n, 0), comp(n, -1);
    std::vector<char>     on_stack(n, 0);
    std::vector<uint32_t> scc;
    struct Frame { uint32_t v; size_t e; };
    std::vector<Frame> call;
    int32_t counter = 0, n_comp = 0;
    for (size_t root = 0; root < n; ++root) {
        if (idx[root] >= 0) continue;
        idx[root] = low[root] = counter++;
        scc.push_back((uint32_t)root);
        on_stack[root] = 1;
        call.push_back({ (uint32_t)root, edge_off[root] });
        while (!call.empty()) {
            const uint32_t v = call.back().v;
            if (call.back().e < edge_off[v + 1]) {
                const uint32_t w = edges[call.back().e++].to;
                if (idx[w] < 0) {
                    idx[w] = low[w] = counter++;
                    scc.push_back(w);
                    on_stack[w] = 1;
                    call.push_back({ w, edge_off[w] });
                } else if (on_stack[w]) {
                    low[v] = std::min(low[v], idx[w]);
                }
                continue;
            }
            call.pop_back();
            if (!call.empty()) low[call.back().v] = std::min(low[call.back().v], low[v]);
            if (low[v] == idx[v]) {
                uint32_t w;
                do {
                    w = scc.back();
                    scc.pop_back();
                    on_stack[w] = 0;
                    comp[w] = n_comp;
                } while (w != v);
                ++n_comp;
            }
        }
    }

    for (size_t r = 0; r < n; ++r)
        for (size_t k = edge_off[r]; k < edge_off[r + 1]; ++k)
            if (edges[k].grows && comp[edges[k].to] == comp[r]) return (int64_t)r;
    return -1;
}

/* ================================================================== the compiled program */

/* The compiled program is shared, not per request. Its rules, initial stack set, mask plan and
 * the mask of each reachable state are a pure function of (text, root, vocabulary). A
 * GbnfGrammar per request holds only the position in it. Sharing means:
 *   - repeated calls with the same grammar parse and plan it once;
 *   - tensor-parallel ranks share one plan and mask cache, so every rank past the first hits;
 *   - a state one request reached is never recomputed by a later one;
 *   - the plan is built on the admitting thread, never the scheduler thread.
 *
 * Stacks point into `rules`, so `rules` is immutable after publication and the program is held by
 * shared_ptr: a live grammar keeps it alive after the cache unpins it. Those pointers are also
 * what makes a state comparable across requests. */
struct GbnfState {
    GbnfStacks  stacks;      /* sorted, so the key does not depend on the order a walk produced */
    PartialUtf8 partial;
    bool        allow_eog = false;
    int64_t     words     = 0;

    bool operator==(const GbnfState& o) const {
        return allow_eog == o.allow_eog && words == o.words &&
               partial.value == o.partial.value && partial.n_remain == o.partial.n_remain &&
               stacks == o.stacks;
    }
};

/* A state as fill_mask holds it, uncopied: the stacks through pointers, in the key's sorted
 * order. It hashes and compares like the GbnfState it would become, so a probe copies no stacks;
 * a key is built only to store a mask. */
struct GbnfStateView {
    const std::vector<const GbnfStack*>* stacks;
    PartialUtf8                          partial;
    bool                                 allow_eog;
    int64_t                              words;
};

struct GbnfStateHash {
    using is_transparent = void;

    template <class Stacks, class Deref>
    static size_t hash(const Stacks& stacks, Deref deref, PartialUtf8 partial, bool allow_eog,
                       int64_t words) {
        size_t h = 1469598103934665603ull;
        auto mix = [&h](size_t v) { h ^= v; h *= 1099511628211ull; };
        for (const auto& x : stacks) {
            const GbnfStack& s = deref(x);
            mix(s.size());
            for (const GbnfElement* e : s) mix((size_t)e);
        }
        mix(partial.value);
        mix((size_t)(partial.n_remain + 1));
        mix(allow_eog ? 1u : 0u);
        mix((size_t)words);
        return h;
    }
    size_t operator()(const GbnfState& k) const {
        return hash(k.stacks, [](const GbnfStack& s) -> const GbnfStack& { return s; },
                    k.partial, k.allow_eog, k.words);
    }
    size_t operator()(const GbnfStateView& k) const {
        return hash(*k.stacks, [](const GbnfStack* s) -> const GbnfStack& { return *s; },
                    k.partial, k.allow_eog, k.words);
    }
};

struct GbnfStateEq {
    using is_transparent = void;

    bool operator()(const GbnfState& a, const GbnfState& b) const { return a == b; }
    bool operator()(const GbnfStateView& v, const GbnfState& k) const {
        if (k.allow_eog != v.allow_eog || k.words != v.words ||
            k.partial.value != v.partial.value || k.partial.n_remain != v.partial.n_remain ||
            k.stacks.size() != v.stacks->size())
            return false;
        for (size_t i = 0; i < k.stacks.size(); ++i)
            if (k.stacks[i] != *(*v.stacks)[i]) return false;
        return true;
    }
    bool operator()(const GbnfState& k, const GbnfStateView& v) const { return (*this)(v, k); }
};

/* The key a view becomes, for storing. */
static GbnfState state_of(const GbnfStateView& v) {
    GbnfState k;
    k.stacks.reserve(v.stacks->size());
    for (const GbnfStack* s : *v.stacks) k.stacks.push_back(*s);
    k.partial   = v.partial;
    k.allow_eog = v.allow_eog;
    k.words     = v.words;
    return k;
}

struct GbnfProgram {
    GbnfRuleSet        rules;
    GbnfStacks         start;                /* the root's alternates, advanced once */
    gbnf::MaskPlanPtr  plan;                 /* see gbnf_mask.cpp */
    GbnfProgramInfo    info;

    /* Masks computed rather than served from the memo below (the sampler's instrument). Atomic
     * because it is used outside the lock; relaxed because nothing is ordered against it. */
    mutable std::atomic<uint64_t>   computed{0};

    /* The mask cache is shared by every rank's thread. Masks are a few KiB, copied out under the
     * lock and computed outside it; two threads racing on one state both compute the same mask,
     * which only wastes work. */
    mutable std::mutex mu;
    struct Entry { std::vector<uint32_t> mask; uint64_t used = 0; };
    mutable std::unordered_map<GbnfState, Entry, GbnfStateHash, GbnfStateEq> masks;
    mutable uint64_t clock = 0;

    /* A tool-calling schema's working set is about a hundred states; 512 leaves room for one
     * several times larger, at a few MB of host memory for a large vocabulary. Past it, the least
     * recently used entry is evicted by a linear scan, which runs only after a computed mask. */
    static constexpr size_t kMaxMasks = 512;

    bool lookup(const GbnfStateView& k, uint32_t* dst, int64_t n_words) const {
        std::lock_guard<std::mutex> lk(mu);
        auto it = masks.find(k);
        if (it == masks.end()) return false;
        it->second.used = ++clock;
        std::memcpy(dst, it->second.mask.data(), (size_t)n_words * sizeof(uint32_t));
        return true;
    }

    void store(const GbnfState& k, const uint32_t* src, int64_t n_words) const {
        std::lock_guard<std::mutex> lk(mu);
        if (masks.size() >= kMaxMasks && !masks.count(k)) {
            auto victim = masks.begin();
            for (auto it = masks.begin(); it != masks.end(); ++it)
                if (it->second.used < victim->second.used) victim = it;
            masks.erase(victim);
        }
        Entry& e = masks[k];
        e.mask.assign(src, src + n_words);
        e.used = ++clock;
    }
};

const GbnfRuleSet& gbnf::program_rules(const GbnfProgram& p) { return p.rules; }

GbnfProgramInfo gbnf_program_info(const GbnfProgram& p) { return p.info; }

namespace {

/* Programs pinned per vocabulary. Capped because keys come from clients, who must not be able to
 * grow the cache without bound. */
constexpr size_t kMaxPrograms = 32;

/* Unpin least recently used programs past either cap, and forget those nothing holds. Caller
 * holds C.mu. */
void cache_trim(VocabGbnf& C) {
    for (;;) {
        size_t pinned = 0, bytes = 0;
        auto victim = C.programs.end();
        for (auto it = C.programs.begin(); it != C.programs.end(); ++it) {
            if (!it->second.pinned) continue;
            ++pinned;
            bytes += it->second.pinned->info.bytes;
            if (victim == C.programs.end() || it->second.used < victim->second.used) victim = it;
        }
        if (pinned <= kMaxPrograms && (bytes <= gbnf::kMaxPinnedBytes || pinned <= 1)) break;
        victim->second.pinned.reset();
    }
    for (auto it = C.programs.begin(); it != C.programs.end();) {
        if (!it->second.pinned && it->second.alive.expired()) it = C.programs.erase(it);
        else ++it;
    }
}

/* The program for (text, root) if pinned or still alive, re-pinned as most recently used. Caller
 * holds C.mu. */
std::shared_ptr<const GbnfProgram> cache_find(VocabGbnf& C,
                                              const std::pair<std::string, std::string>& key) {
    auto it = C.programs.find(key);
    if (it == C.programs.end()) return nullptr;
    const bool was_pinned = (bool)it->second.pinned;
    std::shared_ptr<const GbnfProgram> p =
        was_pinned ? it->second.pinned : it->second.alive.lock();
    if (!p) {
        C.programs.erase(it);
        return nullptr;
    }
    it->second.pinned = p;
    it->second.used   = ++C.clock;
    if (!was_pinned) cache_trim(C);
    return p;
}

}  /* namespace */

static int program_for_impl(const std::string& text, const std::string& root,
                            const VocabView* vocab, std::shared_ptr<const GbnfProgram>* out,
                            std::string* err) {
    /* Keyed on (text, root) inside the vocabulary's own cache; see "the per-vocabulary cache"
     * for why not on the view's address. */
    VocabGbnf& C = gbnf_cache_for(vocab);
    const std::pair<std::string, std::string> key{ text, root };
    {
        std::lock_guard<std::mutex> lk(C.mu);
        if (auto hit = cache_find(C, key)) {
            *out = std::move(hit);
            return RAD_OK;
        }
    }

    /* Compiled outside the lock so callers do not serialise behind a plan build. If two race on
     * the same grammar, the loser's copy is dropped on insert. */
    const auto t0 = std::chrono::steady_clock::now();
    GbnfParser parser(vocab);
    std::string perr;
    const int st = parser.parse(text.c_str(), &perr);
    if (st < 0) {
        if (err) *err = "grammar parse failed: " + perr;
        return st;
    }
    if (parser.rules().empty()) {
        if (err) *err = "grammar is empty";
        return RAD_E_FORMAT;
    }

    auto sym = parser.symbol_ids().find(root);
    if (sym == parser.symbol_ids().end()) {
        if (err) *err = "grammar does not contain a '" + root + "' symbol";
        return RAD_E_NOTFOUND;
    }

    std::vector<std::string> names(parser.rules().size());
    for (const auto& kv : parser.symbol_ids())
        if (kv.second < names.size()) names[kv.second] = kv.first;

    const int64_t lr = find_left_recursion(parser.rules());
    if (lr >= 0) {
        if (err) {
            const std::string name = names[(size_t)lr].empty()
                                         ? "#" + std::to_string(lr)
                                         : "'" + names[(size_t)lr] + "'";
            *err = "left recursion through rule " + name + "; this state machine is a pushdown "
                   "automaton and cannot run one";
        }
        return RAD_E_UNSUPPORTED;
    }

    /* `rules` is final from here: stacks point into it, so a later reallocation would dangle
     * them. */
    auto p = std::make_shared<GbnfProgram>();
    for (const GbnfRule& r : parser.rules()) {
        p->rules.start.push_back((uint32_t)p->rules.elems.size());
        p->rules.elems.insert(p->rules.elems.end(), r.begin(), r.end());
    }

    const GbnfElement* pos = p->rules.rule(sym->second);
    OpWork    start_work;
    SetBuild start_set(start_work);
    for (;;) {
        GbnfStack stack;
        if (!is_end_of_sequence(pos)) stack.push_back(pos);
        advance_stack(p->rules, stack, p->start, start_set);
        while (!is_end_of_sequence(pos)) pos++;
        if (pos->type == GRE_ALT) pos++;
        else break;
    }

    gbnf::PlanStats ps;
    const int bs = gbnf::plan_build(p->rules, sym->second, vocab_index_for(vocab), &names,
                                    &p->plan, &ps, err);
    if (bs < 0) return bs;

    GbnfProgramInfo& info = p->info;
    info.rules         = p->rules.n_rules();
    info.elements      = p->rules.elems.size();
    info.plan_ms       = ps.ms;
    info.plan_work     = ps.work;
    info.keys          = ps.keys;
    info.dense_masks   = ps.dense;
    info.max_key_depth = ps.max_depth;
    info.rem_nodes     = ps.rem_nodes;
    info.max_rem_nodes = ps.max_rem;
    info.bytes         = ps.bytes + p->rules.elems.size() * sizeof(GbnfElement);
    info.compile_ms    = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0).count();
    RAD_INFO("gbnf: compiled a grammar of %zu rules in %.1f ms: %zu mask keys (%zu dense, %zu "
             "deep at most), %zu remainder nodes, %.1f MiB", info.rules, info.compile_ms,
             info.keys, info.dense_masks, info.max_key_depth, info.rem_nodes,
             (double)info.bytes / (1024.0 * 1024.0));

    std::lock_guard<std::mutex> lk(C.mu);
    if (auto hit = cache_find(C, key)) {       /* another thread won the race; use its copy */
        *out = std::move(hit);
        return RAD_OK;
    }
    C.programs[key] = VocabGbnf::Cached{ p, p, ++C.clock };
    cache_trim(C);
    *out = std::move(p);
    return RAD_OK;
}

/* No exception escapes compiling. The stack machinery throws GbnfLimit and any step can throw
 * bad_alloc; this runs on the admitting thread, where an escaping exception would terminate the
 * server. */
static int program_for(const std::string& text, const std::string& root, const VocabView* vocab,
                       std::shared_ptr<const GbnfProgram>* out, std::string* err) {
    try {
        return program_for_impl(text, root, vocab, out, err);
    } catch (const GbnfLimit& e) {
        if (err) *err = std::string("the grammar cannot be run within its bounds: ") + e.what();
        return RAD_E_UNSUPPORTED;
    } catch (const std::bad_alloc&) {
        if (err) *err = "out of memory compiling the grammar";
        return RAD_E_NOMEM;
    } catch (const std::exception& e) {
        if (err) *err = std::string("the grammar could not be compiled: ") + e.what();
        return RAD_E_FORMAT;
    }
}

int gbnf_compile(const std::string& text, const std::string& root, const VocabView* vocab,
                 std::shared_ptr<const GbnfProgram>* out, std::string* err) {
    if (!vocab || !out) return RAD_E_INVAL;
    return program_for(text, root, vocab, out, err);
}

/* ================================================================== the grammar */

int GbnfGrammar::create(const std::string& text, const std::string& root, const VocabView* vocab,
                        bool lazy,
                        const std::vector<std::string>& trigger_words,
                        const std::vector<int32_t>& trigger_tokens,
                        std::unique_ptr<GbnfGrammar>* out, std::string* err) {
    if (!out) return RAD_E_INVAL;
    /* A lazy grammar constrains nothing until a trigger fires, so without one it never would. */
    if (lazy && trigger_words.empty() && trigger_tokens.empty()) {
        if (err) *err = "a lazy grammar needs at least one word or token trigger; without one it "
                        "never constrains";
        return RAD_E_INVAL;
    }

    /* The compiled program is shared, so a repeated grammar is a cache lookup. Only the position
     * in the program and the triggers are per request. */
    std::shared_ptr<const GbnfProgram> prog;
    const int st = program_for(text, root, vocab, &prog, err);
    if (st < 0) return st;

    try {
        auto g = std::unique_ptr<GbnfGrammar>(new GbnfGrammar());
        g->vocab_  = vocab;
        g->prog_   = prog;
        g->stacks_ = prog->start;

        g->awaiting_trigger_ = lazy;
        g->trigger_tokens_   = trigger_tokens;
        if (!trigger_words.empty()) {
            auto sc = std::make_shared<WordScanner>();
            const int bs = sc->build(trigger_words, err);
            if (bs < 0) return bs;
            g->scanner_ = std::move(sc);
        }

        *out = std::move(g);
    } catch (const std::bad_alloc&) {
        if (err) *err = "out of memory creating the grammar";
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

uint64_t GbnfGrammar::masks_walked() const {
    return prog_ ? prog_->computed.load(std::memory_order_relaxed) : 0;
}

bool GbnfGrammar::complete() const {
    for (const auto& s : stacks_) if (s.empty()) return true;
    return false;
}

static void accept_chr(const GbnfRuleSet& rules, const GbnfStack& stack, uint32_t chr,
                       GbnfStacks& out, SetBuild& build) {
    if (stack.empty()) return;
    const GbnfElement* pos = stack.back();
    if (pos->type == GRE_TOKEN || pos->type == GRE_TOKEN_NOT) return;

    auto m = match_char(pos, chr);
    build.work.add((size_t)(m.second - pos));
    if (!m.first) return;

    GbnfStack next(stack.begin(), stack.end() - 1);
    if (!is_end_of_sequence(m.second)) next.push_back(m.second);
    advance_stack(rules, next, out, build);
}

/* A grammar that outgrows its bounds while accepting cannot continue, since its next state was
 * never built. It is left with no live stack, so the next mask fails the request; the status says
 * which bound was hit. A speculative walk rewinds over it like any other refusal. */
int GbnfGrammar::accept_token_piece(int32_t token, const std::string& piece) {
    try {
        return accept_token_piece_impl(token, piece);
    } catch (const GbnfLimit& e) {
        RAD_WARN("gbnf: token %d cannot be accepted: %s; the request cannot be continued",
                 token, e.what());
        stacks_.clear();
        at_ = At{};
        return RAD_E_FULL;
    } catch (const std::bad_alloc&) {
        RAD_ERR("gbnf: out of memory accepting token %d; the request cannot be continued", token);
        stacks_.clear();
        at_ = At{};
        return RAD_E_NOMEM;
    }
}

int GbnfGrammar::accept_token_piece_impl(int32_t token, const std::string& piece) {
    const PartialUtf8 partial = decode_utf8(piece, partial_utf8_, &cps_);
    const size_t      n_cps   = cps_.size() - 1;     /* the terminating 0 is not a code point */

    OpWork     work;
    GbnfStacks new_stacks;
    SetBuild   new_set(work);

    /* Code-point stacks advance together, one code point at a time, as in accept_str. Advancing
     * each through the whole piece alone gives the same set, but stacks converging on one
     * continuation (e.g. the alternatives of a wide rule) would each re-expand it, quadratic in
     * the width.
     *
     * They step through the mask plan's automaton (plan_step), which memoises each state's
     * successors per code-point interval, so a seen state costs a lookup per code point instead
     * of rebuilding closures. The result equals what accept_chr builds; its sorted order is
     * incidental (the mask cache sorts its own key).
     *
     * The new set overwrites the previous token's (`spare_`) and the two are swapped, so a
     * steady-state set allocates nothing; when the state is unchanged the set is not touched. */
    current_.clear();
    for (const auto& stack : stacks_) {
        if (stack.empty()) continue;
        const GbnfElement* pos = stack.back();

        if (pos->type == GRE_TOKEN || pos->type == GRE_TOKEN_NOT) {
            if (!match_token(pos, token)) continue;
            GbnfStack next(stack.begin(), stack.end() - 1);
            if (!is_end_of_sequence(pos + 1)) next.push_back(pos + 1);
            advance_stack(prog_->rules, next, new_stacks, new_set);
        } else {
            current_.push_back(&stack);
        }
    }
    GbnfStacks& out = spare_;
    if (n_cps > 0 && !current_.empty()) {
        /* `at_` names a state only when every stack steps over code points, so when it is known no
         * stack matched a token terminal above and the set is the stepped stacks alone. */
        gbnf::PlanState at{ at_.gen, at_.id };
        const bool same = gbnf::plan_step(*prog_->plan, current_, &at, cps_.data(), n_cps, &out,
                                          work);
        at_ = At{ at.gen, at.id };
        if (same) {
            partial_utf8_ = partial;
            return RAD_OK;
        }
    } else {
        out.resize(current_.size());
        for (size_t i = 0; i < current_.size(); ++i) out[i].assign(current_[i]->begin(),
                                                                   current_[i]->end());
        at_ = At{};
    }
    /* Stacks from token terminals go first, then the rest, de-duplicated against them. With no
     * token match, `out` is already duplicate-free (an automaton state, or a subset of the
     * previous set). */
    if (!new_stacks.empty()) {
        for (auto& s : out) new_set.insert(new_stacks, std::move(s));
        out.swap(new_stacks);
        at_ = At{};
    }
    stacks_.swap(out);
    partial_utf8_ = partial;

    if (stacks_.empty()) {
        /* Debug level: the speculative walk (Sampler::build_step) reaches this routinely when a
         * drafter proposes a forbidden token, and handles it by setting `dead`. An emitted token
         * that is forbidden is a defect, but only Engine::step can tell, so it reports this
         * status on the post-verification accept; the status must not be dropped. */
        RAD_DEBUG("gbnf: token %d (\"%s\") left no live stack", token, piece.c_str());
        return RAD_E_STATE;
    }
    return RAD_OK;
}

int GbnfGrammar::accept_token(int32_t token) {
    if (!vocab_) return RAD_E_STATE;
    const std::string& piece = vocab_->token_piece(token);

    if (awaiting_trigger_) {
        if (std::find(trigger_tokens_.begin(), trigger_tokens_.end(), token)
            != trigger_tokens_.end()) {
            awaiting_trigger_ = false;
            recent_.clear();
            return accept_token_piece(token, piece);
        }
        if (!scanner_) return RAD_OK;

        recent_.push_back(Recent{ token, scan_.off });
        const uint64_t start = scanner_->feed(scan_, piece.data(), piece.size());
        if (start == WordScanner::npos) {
            /* Keep only tokens a later word could begin in: one ending at or before
             * (fed - longest + 1) cannot hold the first byte of a word still to come. */
            const uint64_t reach = scanner_->max_len() - 1;
            const uint64_t keep  = scan_.off > reach ? scan_.off - reach : 0;
            size_t drop = 0;
            while (drop < recent_.size()) {
                const uint64_t end = drop + 1 < recent_.size() ? recent_[drop + 1].start : scan_.off;
                if (end > keep) break;
                ++drop;
            }
            if (drop) recent_.erase(recent_.begin(), recent_.begin() + (ptrdiff_t)drop);
            return RAD_OK;
        }

        /* Replay from the word's first byte, splitting the token it begins in; the bytes before it
         * are free-form output outside the grammar. */
        awaiting_trigger_ = false;
        std::vector<Recent> replay;
        replay.swap(recent_);
        for (size_t i = 0; i < replay.size(); ++i) {
            const uint64_t ts = replay[i].start;
            const uint64_t te = i + 1 < replay.size() ? replay[i + 1].start : scan_.off;
            if (te <= start) continue;
            const std::string& pc = vocab_->token_piece(replay[i].token);
            if (ts >= start) {
                RAD_TRY(accept_token_piece(replay[i].token, pc));
            } else {
                RAD_TRY(accept_token_piece(replay[i].token, pc.substr((size_t)(start - ts))));
            }
        }
        return RAD_OK;
    }

    if (vocab_->is_eog(token)) {
        if (complete()) return RAD_OK;
        /* Debug level, as for "left no live stack" in accept_token_piece. The mask admits EOG
         * only once a stack has completed, so this is normally a drafter's proposal in the
         * speculative walk; an emitted one is a defect that Engine::step reports from this
         * status. */
        RAD_DEBUG("gbnf: end-of-generation token %d offered while no stack had completed", token);
        return RAD_E_STATE;
    }

    return accept_token_piece(token, piece);
}

int GbnfGrammar::accept_str(const std::string& piece) {
    const auto decoded = decode_utf8(piece, partial_utf8_);
    const auto& cps = decoded.first;
    at_ = At{};

    try {
        OpWork work;
        for (auto it = cps.begin(), end = cps.end() - 1; it != end; ++it) {
            GbnfStacks next;
            SetBuild  next_set(work);
            for (const auto& s : stacks_) accept_chr(prog_->rules, s, *it, next, next_set);
            stacks_ = std::move(next);
            if (stacks_.empty()) {
                RAD_ERR("gbnf: the string \"%s\" is not admitted by the grammar", piece.c_str());
                return RAD_E_STATE;
            }
        }
    } catch (const GbnfLimit& e) {
        RAD_ERR("gbnf: the string \"%s\" cannot be accepted: %s", piece.c_str(), e.what());
        stacks_.clear();
        return RAD_E_FULL;
    } catch (const std::bad_alloc&) {
        RAD_ERR("gbnf: out of memory accepting the string \"%s\"", piece.c_str());
        stacks_.clear();
        return RAD_E_NOMEM;
    }
    partial_utf8_ = decoded.second;
    return RAD_OK;
}

void GbnfGrammar::push_state() {
    saved_.push_back(Snapshot{ stacks_, partial_utf8_, awaiting_trigger_, scan_, recent_, at_ });
}

int GbnfGrammar::rollback() {
    if (saved_.empty()) {
        RAD_ERR("gbnf: rollback with nothing pushed");
        return RAD_E_STATE;
    }
    Snapshot& s = saved_.back();
    stacks_                   = std::move(s.stacks);
    partial_utf8_             = s.partial;
    awaiting_trigger_         = s.awaiting_trigger;
    scan_                     = s.scan;
    recent_                   = std::move(s.recent);
    at_                       = s.at;
    saved_.pop_back();
    return RAD_OK;
}

/* The mask for the current state, with no memo. */
int GbnfGrammar::mask_compute(uint32_t* bitmask, int64_t n_words) const {
    std::fill(bitmask, bitmask + n_words, 0u);

    /* A dead (empty) stack set admits nothing. accept_token_piece leaves one after a failure, so
     * a later mask request gets a status here, not a mask. */
    if (stacks_.empty()) {
        RAD_ERR("gbnf: the grammar has no live parse left; the request cannot be continued");
        return RAD_E_STATE;
    }

    /* On the scheduler thread, so a mask past the bounds in "what a grammar may cost" becomes a
     * status that fails the request, never an escaping exception. */
    try {
        return gbnf::plan_fill(*prog_->plan, stacks_, partial_utf8_, complete(), bitmask, n_words);
    } catch (const GbnfLimit& e) {
        RAD_ERR("gbnf: no mask for this position: %s; the request cannot be continued", e.what());
        return RAD_E_FULL;
    } catch (const std::bad_alloc&) {
        RAD_ERR("gbnf: out of memory building a mask; the request cannot be continued");
        return RAD_E_NOMEM;
    }
}

int GbnfGrammar::fill_mask(uint32_t* bitmask, int64_t n_words) const {
    if (!bitmask || !vocab_) return RAD_E_INVAL;
    const int64_t n_tok = vocab_->n_tokens();
    if (n_words < bitmask_words(n_tok)) return RAD_E_SHAPE;

    /* An untriggered lazy grammar admits every token. The sampler skips the mask op for it anyway;
     * all ones keeps the driver to one code path. */
    if (awaiting_trigger_) {
        std::fill(bitmask, bitmask + n_words, 0xFFFFFFFFu);
        return RAD_OK;
    }

    /* The state cache, ahead of the plan. Keyed on everything fill_mask reads that can change
     * (stack set, carried partial, allow_eog) and compared exactly after hashing, since a hash
     * collision would return the wrong mask. Stacks are sorted so equal sets give equal keys. The
     * probe sorts pointers in thread-local storage and copies stacks only to store. */
    thread_local std::vector<const GbnfStack*> sorted;
    sorted.clear();
    bool is_sorted = true;
    for (const GbnfStack& s : stacks_) {
        if (is_sorted && !sorted.empty() && s < *sorted.back()) is_sorted = false;
        sorted.push_back(&s);
    }
    if (!is_sorted) {
        std::sort(sorted.begin(), sorted.end(),
                  [](const GbnfStack* a, const GbnfStack* b) { return *a < *b; });
    }
    const GbnfStateView key{ &sorted, partial_utf8_, complete(), n_words };
    if (prog_->lookup(key, bitmask, n_words)) return RAD_OK;

    prog_->computed.fetch_add(1, std::memory_order_relaxed);
    const int st = mask_compute(bitmask, n_words);
    if (st < 0) return st;

    /* A failed state returned above and is not cached. */
    prog_->store(state_of(key), bitmask, n_words);
    return RAD_OK;
}

}  /* namespace rad */
