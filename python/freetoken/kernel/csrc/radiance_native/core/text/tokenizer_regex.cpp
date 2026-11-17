#include "text/tokenizer_regex.h"
#include "text/tokenizer_ucd.h"
#include "rad_internal.h"
#include "util/simd.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace rad {

namespace {

constexpr uint32_t kMaxCp = 0x10FFFF;

/* ================================================================== codepoint sets
 *
 * Sorted, disjoint, merged [first, last] ranges. Every set a pattern names is one of these until
 * the alphabet is partitioned, after which the DFA only ever sees class numbers. */
using CpSet = std::vector<std::pair<uint32_t, uint32_t>>;

void set_normalize(CpSet& s) {
    std::sort(s.begin(), s.end());
    CpSet out;
    for (const auto& r : s) {
        if (!out.empty() && r.first <= out.back().second + 1) {
            out.back().second = std::max(out.back().second, r.second);
        } else {
            out.push_back(r);
        }
    }
    s.swap(out);
}

CpSet set_union(const CpSet& a, const CpSet& b) {
    CpSet r = a;
    r.insert(r.end(), b.begin(), b.end());
    set_normalize(r);
    return r;
}

CpSet set_complement(const CpSet& a) {
    CpSet r;
    uint32_t next = 0;
    for (const auto& x : a) {
        if (x.first > next) r.push_back({ next, x.first - 1 });
        next = x.second + 1;
    }
    if (next <= kMaxCp) r.push_back({ next, kMaxCp });
    return r;
}

CpSet set_intersect(const CpSet& a, const CpSet& b) {
    CpSet r;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        const uint32_t lo = std::max(a[i].first, b[j].first);
        const uint32_t hi = std::min(a[i].second, b[j].second);
        if (lo <= hi) r.push_back({ lo, hi });
        if (a[i].second < b[j].second) ++i; else ++j;
    }
    return r;
}

bool set_has(const CpSet& s, uint32_t c) {
    auto it = std::upper_bound(s.begin(), s.end(), c,
                               [](uint32_t v, const std::pair<uint32_t, uint32_t>& r) {
                                   return v < r.first;
                               });
    return it != s.begin() && c <= (it - 1)->second;
}

CpSet set_from(const ucd::CpRange* r, size_t n) {
    CpSet s;
    s.reserve(n);
    for (size_t i = 0; i < n; ++i) s.push_back({ r[i].first, r[i].last });
    set_normalize(s);
    return s;
}

/* One set per general category, built once from the run table. */
const std::vector<CpSet>& gc_sets() {
    static const std::vector<CpSet> sets = [] {
        std::vector<CpSet> v(ucd::kGcCount);
        for (size_t i = 0; i < ucd::k_gc_runs_n; ++i) {
            const uint32_t first = ucd::k_gc_runs[i].first;
            const uint32_t last = i + 1 < ucd::k_gc_runs_n ? ucd::k_gc_runs[i + 1].first - 1 : kMaxCp;
            v[ucd::k_gc_runs[i].gc].push_back({ first, last });
        }
        for (CpSet& s : v) set_normalize(s);
        return v;
    }();
    return sets;
}

CpSet gc_mask_set(uint32_t mask) {
    CpSet s;
    for (uint32_t g = 0; g < ucd::kGcCount; ++g)
        if (mask & (1u << g)) s.insert(s.end(), gc_sets()[g].begin(), gc_sets()[g].end());
    set_normalize(s);
    return s;
}

const CpSet& white_space_set() { static const CpSet s = set_from(ucd::k_white_space, ucd::k_white_space_n); return s; }
const CpSet& digit_set()       { static const CpSet s = set_from(ucd::k_digit, ucd::k_digit_n); return s; }
const CpSet& word_set()        { static const CpSet s = set_from(ucd::k_word, ucd::k_word_n); return s; }
const CpSet& word_class_set()  { static const CpSet s = set_from(ucd::k_word_class, ucd::k_word_class_n); return s; }
const CpSet& hex_set() {
    static const CpSet s = { { '0', '9' }, { 'A', 'F' }, { 'a', 'f' } };
    return s;
}

/* Case-folding orbits. `(?i)` makes a character match its whole orbit. */
struct CaseData {
    std::vector<std::vector<uint32_t>> orbits;
    std::unordered_map<uint32_t, uint32_t> orbit_of;
    CpSet multi;
};

const CaseData& case_data() {
    static const CaseData d = [] {
        CaseData c;
        for (size_t i = 0; i < ucd::k_case_orbits_n; ) {
            const uint32_t n = ucd::k_case_orbits[i++];
            std::vector<uint32_t> o(ucd::k_case_orbits + i, ucd::k_case_orbits + i + n);
            for (uint32_t m : o) c.orbit_of[m] = (uint32_t)c.orbits.size();
            c.orbits.push_back(std::move(o));
            i += n;
        }
        for (size_t i = 0; i < ucd::k_case_multi_n; ++i)
            c.multi.push_back({ ucd::k_case_multi[i], ucd::k_case_multi[i] });
        set_normalize(c.multi);
        return c;
    }();
    return d;
}

CpSet case_close(const CpSet& s) {
    CpSet add;
    for (const auto& o : case_data().orbits) {
        bool any = false;
        for (uint32_t m : o) if (set_has(s, m)) { any = true; break; }
        if (any) for (uint32_t m : o) add.push_back({ m, m });
    }
    return add.empty() ? s : set_union(s, add);
}

/* The general-category names Oniguruma accepts, short and long, after the same normalisation
 * it applies (case, spaces, underscores and hyphens ignored). */
struct GcName { const char* name; uint32_t mask; };

constexpr uint32_t bit(ucd::Gc g) { return 1u << g; }
constexpr uint32_t kL = bit(ucd::Lu) | bit(ucd::Ll) | bit(ucd::Lt) | bit(ucd::Lm) | bit(ucd::Lo);
constexpr uint32_t kM = bit(ucd::Mn) | bit(ucd::Mc) | bit(ucd::Me);
constexpr uint32_t kN = bit(ucd::Nd) | bit(ucd::Nl) | bit(ucd::No);
constexpr uint32_t kP = bit(ucd::Pc) | bit(ucd::Pd) | bit(ucd::Ps) | bit(ucd::Pe) |
                        bit(ucd::Pi) | bit(ucd::Pf) | bit(ucd::Po);
constexpr uint32_t kS = bit(ucd::Sm) | bit(ucd::Sc) | bit(ucd::Sk) | bit(ucd::So);
constexpr uint32_t kZ = bit(ucd::Zs) | bit(ucd::Zl) | bit(ucd::Zp);
constexpr uint32_t kC = bit(ucd::Cc) | bit(ucd::Cf) | bit(ucd::Cs) | bit(ucd::Co) | bit(ucd::Cn);

const GcName k_gc_names[] = {
    { "l", kL }, { "letter", kL },
    { "lu", bit(ucd::Lu) }, { "uppercaseletter", bit(ucd::Lu) },
    { "ll", bit(ucd::Ll) }, { "lowercaseletter", bit(ucd::Ll) },
    { "lt", bit(ucd::Lt) }, { "titlecaseletter", bit(ucd::Lt) },
    { "lm", bit(ucd::Lm) }, { "modifierletter", bit(ucd::Lm) },
    { "lo", bit(ucd::Lo) }, { "otherletter", bit(ucd::Lo) },
    { "m", kM }, { "mark", kM }, { "combiningmark", kM },
    { "mn", bit(ucd::Mn) }, { "nonspacingmark", bit(ucd::Mn) },
    { "mc", bit(ucd::Mc) }, { "spacingmark", bit(ucd::Mc) },
    { "me", bit(ucd::Me) }, { "enclosingmark", bit(ucd::Me) },
    { "n", kN }, { "number", kN },
    { "nd", bit(ucd::Nd) }, { "decimalnumber", bit(ucd::Nd) }, { "digit", bit(ucd::Nd) },
    { "nl", bit(ucd::Nl) }, { "letternumber", bit(ucd::Nl) },
    { "no", bit(ucd::No) }, { "othernumber", bit(ucd::No) },
    { "p", kP }, { "punctuation", kP }, { "punct", kP },
    { "pc", bit(ucd::Pc) }, { "connectorpunctuation", bit(ucd::Pc) },
    { "pd", bit(ucd::Pd) }, { "dashpunctuation", bit(ucd::Pd) },
    { "ps", bit(ucd::Ps) }, { "openpunctuation", bit(ucd::Ps) },
    { "pe", bit(ucd::Pe) }, { "closepunctuation", bit(ucd::Pe) },
    { "pi", bit(ucd::Pi) }, { "initialpunctuation", bit(ucd::Pi) },
    { "pf", bit(ucd::Pf) }, { "finalpunctuation", bit(ucd::Pf) },
    { "po", bit(ucd::Po) }, { "otherpunctuation", bit(ucd::Po) },
    { "s", kS }, { "symbol", kS },
    { "sm", bit(ucd::Sm) }, { "mathsymbol", bit(ucd::Sm) },
    { "sc", bit(ucd::Sc) }, { "currencysymbol", bit(ucd::Sc) },
    { "sk", bit(ucd::Sk) }, { "modifiersymbol", bit(ucd::Sk) },
    { "so", bit(ucd::So) }, { "othersymbol", bit(ucd::So) },
    { "z", kZ }, { "separator", kZ },
    { "zs", bit(ucd::Zs) }, { "spaceseparator", bit(ucd::Zs) },
    { "zl", bit(ucd::Zl) }, { "lineseparator", bit(ucd::Zl) },
    { "zp", bit(ucd::Zp) }, { "paragraphseparator", bit(ucd::Zp) },
    { "c", kC }, { "other", kC },
    { "cc", bit(ucd::Cc) }, { "control", bit(ucd::Cc) }, { "cntrl", bit(ucd::Cc) },
    { "cf", bit(ucd::Cf) }, { "format", bit(ucd::Cf) },
    { "cs", bit(ucd::Cs) }, { "surrogate", bit(ucd::Cs) },
    { "co", bit(ucd::Co) }, { "privateuse", bit(ucd::Co) },
    { "cn", bit(ucd::Cn) }, { "unassigned", bit(ucd::Cn) },
};

std::string prop_key(const std::string& raw) {
    std::string k;
    for (char c : raw) {
        if (c == ' ' || c == '_' || c == '-') continue;
        k += (char)((c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c);
    }
    return k;
}

bool property_set(const std::string& name, CpSet* out) {
    const std::string k = prop_key(name);
    for (const GcName& g : k_gc_names) {
        if (k == g.name) { *out = gc_mask_set(g.mask); return true; }
    }
    for (size_t i = 0; i < ucd::k_scripts_n; ++i) {
        if (k == prop_key(ucd::k_scripts[i].name)) {
            *out = set_from(ucd::k_scripts[i].ranges, ucd::k_scripts[i].n);
            return true;
        }
    }
    return false;
}

/* ================================================================== the pattern, parsed */

enum AssertKind : uint8_t {
    A_LINE_START, A_LINE_END, A_TEXT_START, A_TEXT_END, A_WORD_B, A_NOT_WORD_B,
    A_AHEAD, A_AHEAD_NOT, A_BEHIND, A_BEHIND_NOT,
};

struct Node {
    enum Kind : uint8_t { EMPTY, SET, CAT, ALT, REP, ASSERT } kind = EMPTY;
    uint8_t assert_kind = 0;
    bool    greedy = true;
    int     min = 0, max = 0;          /* REP; max < 0 is unbounded */
    int     set = -1;                  /* SET, and the lookaround ASSERTs */
    std::vector<int> kids;
};

struct Flags { bool i = false, m = false, x = false; };

/* Recursive descent over the pattern's codepoints. Every refusal goes through fail(), which
 * records the first reason and makes every caller unwind; the message names the construct and
 * the byte where it starts. */
class Parser {
public:
    explicit Parser(std::string_view pat) : pat_(pat) {}

    bool parse(int* root) {
        if (!decode_pattern()) return false;
        Flags f;
        const int r = parse_alt(f);
        if (r < 0) return false;
        if (i_ < cp_.size()) return fail("an unmatched ')'");
        *root = r;
        return true;
    }

    std::string            err;
    std::vector<Node>      nodes;
    std::vector<CpSet>     sets;

private:
    std::string_view       pat_;
    std::vector<uint32_t>  cp_;
    std::vector<size_t>    off_;       /* byte offset of each codepoint */
    size_t                 i_ = 0;
    size_t                 start_ = 0; /* where the construct being parsed began */
    /* Groups open around the construct being parsed: a possessive quantifier is taken only at
     * depth 0, where what follows its concatenation is the end of the match. */
    int                    depth_ = 0;
    /* Possessive repetitions parse_quant accepted, read as greedy; parse_concat checks that the
     * rest of their concatenation cannot take back what they consumed. */
    std::vector<int>       possessive_;

    bool decode_pattern() {
        const uint8_t* p = (const uint8_t*)pat_.data();
        const size_t n = pat_.size();
        for (size_t i = 0; i < n; ) {
            uint32_t c = p[i], len = 1;
            if (c >= 0x80) {
                if      (c >= 0xC2 && c < 0xE0) { len = 2; c &= 0x1F; }
                else if (c >= 0xE0 && c < 0xF0) { len = 3; c &= 0x0F; }
                else if (c >= 0xF0 && c < 0xF5) { len = 4; c &= 0x07; }
                else { err = fmt("the pattern is not UTF-8 (byte %zu)", i); return false; }
                if (i + len > n) { err = fmt("the pattern is not UTF-8 (byte %zu)", i); return false; }
                for (uint32_t k = 1; k < len; ++k) {
                    if ((p[i + k] & 0xC0) != 0x80) {
                        err = fmt("the pattern is not UTF-8 (byte %zu)", i);
                        return false;
                    }
                    c = (c << 6) | (p[i + k] & 0x3F);
                }
            }
            cp_.push_back(c);
            off_.push_back(i);
            i += len;
        }
        off_.push_back(n);
        return true;
    }

    bool fail(const char* what) {
        if (err.empty()) {
            const size_t at = off_.empty() ? 0 : off_[std::min(start_, off_.size() - 1)];
            const size_t end = off_.empty() ? 0 : off_[std::min(i_, off_.size() - 1)];
            /* The construct, cut at a codepoint boundary so the message stays UTF-8. */
            size_t stop = std::max(end, at + 1);
            const size_t cap = std::min(start_ + 40, off_.size() - 1);
            if (stop > off_[cap]) stop = off_[cap];
            std::string frag(pat_.substr(at, stop - at));
            if (stop < end) frag += "...";
            err = fmt("%s at byte %zu ('%s')", what, at, frag.c_str());
        }
        return false;
    }
    int failn(const char* what) { fail(what); return -1; }

    bool at_end() const { return i_ >= cp_.size(); }
    uint32_t peek(size_t k = 0) const { return i_ + k < cp_.size() ? cp_[i_ + k] : 0xFFFFFFFFu; }

    int add(Node n) { nodes.push_back(std::move(n)); return (int)nodes.size() - 1; }
    int add_set(CpSet s) { sets.push_back(std::move(s)); return (int)sets.size() - 1; }
    int mk_set(CpSet s) { Node n; n.kind = Node::SET; n.set = add_set(std::move(s)); return add(n); }
    int mk_assert(uint8_t k, int set = -1) {
        Node n; n.kind = Node::ASSERT; n.assert_kind = k; n.set = set; return add(n);
    }

    void skip_extended(const Flags& f) {
        if (!f.x) return;
        while (!at_end()) {
            const uint32_t c = peek();
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') { ++i_; continue; }
            if (c == '#') { while (!at_end() && peek() != '\n') ++i_; continue; }
            break;
        }
    }

    int parse_alt(const Flags& f) {
        std::vector<int> alts;
        for (;;) {
            const int c = parse_concat(f);
            if (c < 0) return -1;
            alts.push_back(c);
            if (!at_end() && peek() == '|') { ++i_; continue; }
            break;
        }
        if (alts.size() == 1) return alts[0];
        Node n; n.kind = Node::ALT; n.kids = std::move(alts);
        return add(n);
    }

    /* `(?imx-imx)` with no colon. Oniguruma reads the rest of the enclosing group, every later `|`
     * included, as one group with the new flags: `a(?i)b|c` is `a(?i:b|c)`, not
     * `a(?i:b)|(?i:c)`. The caller parses that rest as a nested alternation. */
    bool try_inline_flags(const Flags& f, Flags* g) {
        if (peek() != '(' || peek(1) != '?') return false;
        size_t k = 2;
        bool on = true, any = false;
        *g = f;
        for (;; ++k) {
            const uint32_t c = peek(k);
            if (c == '-') { on = false; continue; }
            if (c == 'i') { g->i = on; any = true; continue; }
            if (c == 'm') { g->m = on; any = true; continue; }
            if (c == 'x') { g->x = on; any = true; continue; }
            if (c == ')' && any) { i_ += k + 1; return true; }
            return false;
        }
    }

    int parse_concat(const Flags& f) {
        std::vector<int> items;
        for (;;) {
            skip_extended(f);
            if (at_end() || peek() == '|' || peek() == ')') break;
            Flags g;
            if (try_inline_flags(f, &g)) {
                const int rest = parse_alt(g);
                if (rest < 0) return -1;
                items.push_back(rest);
                break;
            }
            start_ = i_;
            int a = parse_atom(f);
            if (a < 0) return -1;
            a = parse_quant(a, f);
            if (a < 0) return -1;
            items.push_back(a);
        }
        if (!possessive_tails_ok(items)) return -1;
        if (items.empty()) return add(Node{});
        if (items.size() == 1) return items[0];
        Node n; n.kind = Node::CAT; n.kids = std::move(items);
        return add(n);
    }

    static bool is_digit(uint32_t c) { return c >= '0' && c <= '9'; }

    /* `{n}`, `{n,}`, `{,m}`, `{n,m}`, digits only. Anything else is a literal brace, as in Ruby. */
    bool interval_at(size_t at, int* mn, int* mx, size_t* len) const {
        if (at >= cp_.size() || cp_[at] != '{') return false;
        size_t k = at + 1;
        long a = -1, b = -1;
        auto num = [&](long* v) {
            if (k >= cp_.size() || !is_digit(cp_[k])) return;
            long x = 0;
            while (k < cp_.size() && is_digit(cp_[k])) { x = std::min(x * 10 + (long)(cp_[k] - '0'), 100000L); ++k; }
            *v = x;
        };
        num(&a);
        bool comma = false;
        if (k < cp_.size() && cp_[k] == ',') { comma = true; ++k; num(&b); }
        if (k >= cp_.size() || cp_[k] != '}') return false;
        if (a < 0 && (!comma || b < 0)) return false;
        *mn = a < 0 ? 0 : (int)a;
        *mx = comma ? (b < 0 ? -1 : (int)b) : *mn;
        *len = k + 1 - at;
        return true;
    }

    bool is_quantifier_at(size_t at) const {
        if (at >= cp_.size()) return false;
        const uint32_t c = cp_[at];
        if (c == '?' || c == '*' || c == '+') return true;
        int a, b; size_t l;
        return interval_at(at, &a, &b, &l);
    }

    static bool nullable(const std::vector<Node>& nodes, int id) {
        const Node& n = nodes[id];
        switch (n.kind) {
            case Node::EMPTY: case Node::ASSERT: return true;
            case Node::SET: return false;
            case Node::CAT:
                for (int k : n.kids) if (!nullable(nodes, k)) return false;
                return true;
            case Node::ALT:
                for (int k : n.kids) if (nullable(nodes, k)) return true;
                return false;
            case Node::REP: return n.min == 0 || nullable(nodes, n.kids[0]);
        }
        return true;
    }

    int parse_quant(int atom, const Flags& f) {
        skip_extended(f);
        if (at_end()) return atom;
        int mn = 0, mx = 0;
        size_t len = 1;
        const uint32_t c = peek();
        if (c == '?')      { mn = 0; mx = 1; }
        else if (c == '*') { mn = 0; mx = -1; }
        else if (c == '+') { mn = 1; mx = -1; }
        else if (!interval_at(i_, &mn, &mx, &len)) return atom;
        const bool brace = c == '{';
        i_ += len;
        bool greedy = true, possessive = false;
        if (!at_end() && peek() == '?') { greedy = false; ++i_; }
        else if (!at_end() && peek() == '+') {
            if (brace)
                return failn("a quantifier applied to a quantifier (Ruby reads '{n,m}+' as nested repetition)");
            /* A possessive repetition of one class at the top level is read as greedy, which
             * parse_concat proves gives the same matches; anything else has no DFA form. */
            if (depth_ > 0 || nodes[atom].kind != Node::SET)
                return failn("a possessive quantifier");
            possessive = true;
            ++i_;
        }
        if (is_quantifier_at(i_)) return failn("a quantifier applied to a quantifier");
        if (nodes[atom].kind == Node::ASSERT) return failn("a quantifier on an assertion");
        if (mx >= 0 && mn > mx) return failn("a repetition whose minimum exceeds its maximum");
        if (mn > 1000 || mx > 1000) return failn("a counted repetition above 1000");
        if (mx < 0 && nullable(nodes, atom))
            return failn("unbounded repetition of an expression that can match empty");
        Node n; n.kind = Node::REP; n.min = mn; n.max = mx; n.greedy = greedy; n.kids = { atom };
        const int r = add(n);
        if (possessive) possessive_.push_back(r);
        return r;
    }

    /* A possessive repetition of the class S never gives back, a greedy one gives back only when
     * the rest of the match fails after its longest run. At the top level that rest is the tail of
     * its concatenation and then the end of the match. When the tail is classes and repetitions of
     * classes, none of which can take a codepoint of S, a shorter run leaves the tail facing a
     * codepoint of S it cannot take, so no shorter run matches where the longest did not, and the
     * greedy reading reports exactly the possessive matches (Ling-3.0's `[^\r\n\p{L}\p{N}]?+\p{L}+`
     * and ` ?[^\s\p{L}\p{N}]++[\r\n]*`). Any other tail is refused. */
    bool possessive_tails_ok(const std::vector<int>& items) {
        for (size_t k = 0; k < items.size(); ++k) {
            if (std::find(possessive_.begin(), possessive_.end(), items[k]) == possessive_.end())
                continue;
            const CpSet& own = sets[nodes[nodes[items[k]].kids[0]].set];
            CpSet tail;
            for (size_t j = k + 1; j < items.size(); ++j) {
                const Node& n = nodes[items[j]];
                const Node& m = n.kind == Node::REP ? nodes[n.kids[0]] : n;
                if (m.kind != Node::SET)
                    return fail("a possessive quantifier followed by something other than classes");
                tail = set_union(tail, sets[m.set]);
            }
            if (!set_intersect(own, tail).empty())
                return fail("a possessive quantifier whose class the rest of the match can take");
        }
        return true;
    }

    int literal(uint32_t c, const Flags& f) {
        if (f.i) {
            if (set_has(case_data().multi, c))
                return failn("a case-insensitive character that also matches a multi-codepoint sequence");
            return mk_set(case_close({ { c, c } }));
        }
        return mk_set({ { c, c } });
    }

    /* A lookaround body must be one codepoint wide: a class, a character, or an alternation of
     * those. Its set is what the assertion tests the neighbouring codepoint against. */
    bool single_cp_set(int id, CpSet* out) {
        const Node& n = nodes[id];
        if (n.kind == Node::SET) { *out = sets[n.set]; return true; }
        if (n.kind == Node::CAT && n.kids.size() == 1) return single_cp_set(n.kids[0], out);
        if (n.kind == Node::ALT) {
            CpSet u;
            for (int k : n.kids) {
                CpSet s;
                if (!single_cp_set(k, &s)) return false;
                u = set_union(u, s);
            }
            *out = u;
            return true;
        }
        return false;
    }

    int parse_group(const Flags& outer) {
        /* `(` already consumed. */
        struct Depth { int& d; explicit Depth(int& x) : d(x) { ++d; } ~Depth() { --d; } } depth(depth_);
        Flags f = outer;
        if (peek() != '?') {
            const int r = parse_alt(f);
            if (r < 0) return -1;
            if (peek() != ')') return failn("an unterminated group");
            ++i_;
            return r;
        }
        ++i_;
        const uint32_t c = peek();
        ++i_;
        auto body = [&](Flags& g) -> int {
            const int r = parse_alt(g);
            if (r < 0) return -1;
            if (peek() != ')') return failn("an unterminated group");
            ++i_;
            return r;
        };
        auto lookaround = [&](uint8_t kind) -> int {
            const size_t at = start_;
            const int r = body(f);
            if (r < 0) return -1;
            CpSet s;
            if (!single_cp_set(r, &s)) {
                start_ = at;
                return failn("a lookaround wider than one codepoint");
            }
            return mk_assert(kind, add_set(std::move(s)));
        };
        switch (c) {
            case ':': return body(f);
            case '=': return lookaround(A_AHEAD);
            case '!': return lookaround(A_AHEAD_NOT);
            case '>': return failn("an atomic group");
            case '~': return failn("an absent operator");
            case '#': {
                while (!at_end() && peek() != ')') ++i_;
                if (at_end()) return failn("an unterminated comment");
                ++i_;
                return add(Node{});
            }
            case '<': {
                if (peek() == '=') { ++i_; return lookaround(A_BEHIND); }
                if (peek() == '!') { ++i_; return lookaround(A_BEHIND_NOT); }
                while (!at_end() && peek() != '>') ++i_;       /* a named group */
                if (at_end()) return failn("an unterminated group name");
                ++i_;
                return body(f);
            }
            case '\'': {
                while (!at_end() && peek() != '\'') ++i_;
                if (at_end()) return failn("an unterminated group name");
                ++i_;
                return body(f);
            }
            default: break;
        }
        /* `(?imx-imx:...)`. The no-colon form was consumed by try_inline_flags. */
        --i_;
        bool on = true, any = false;
        for (;;) {
            const uint32_t k = peek();
            if (k == '-') { on = false; ++i_; continue; }
            if (k == 'i') { f.i = on; any = true; ++i_; continue; }
            if (k == 'm') { f.m = on; any = true; ++i_; continue; }
            if (k == 'x') { f.x = on; any = true; ++i_; continue; }
            if (k == ':' && any) { ++i_; return body(f); }
            return failn("an unknown group option");
        }
    }

    int parse_atom(const Flags& f) {
        const uint32_t c = peek();
        ++i_;
        switch (c) {
            case '(': return parse_group(f);
            case '[': {
                CpSet s;
                if (!parse_class(f, &s)) return -1;
                return mk_set(std::move(s));
            }
            case '.': {
                CpSet s = { { 0, kMaxCp } };
                if (!f.m) s = { { 0, '\n' - 1 }, { '\n' + 1, kMaxCp } };
                return mk_set(std::move(s));
            }
            case '^': return mk_assert(A_LINE_START);
            case '$': return mk_assert(A_LINE_END);
            case '\\': return parse_escape_atom(f);
            case '?': case '*': case '+': return failn("a quantifier with nothing to repeat");
            case '{': {
                int a, b; size_t l;
                if (interval_at(i_ - 1, &a, &b, &l)) return failn("a quantifier with nothing to repeat");
                return literal(c, f);
            }
            default: return literal(c, f);
        }
    }

    /* Escapes shared by atoms and classes. `kind` says what came back. */
    enum EscKind { E_CHAR, E_SET, E_ASSERT };
    struct Esc {
        EscKind kind = E_CHAR;
        uint32_t ch = 0;
        CpSet set;
        uint8_t assert_kind = 0;
        bool prop = false;        /* the set came from \p{...} */
    };

    static int hexval(uint32_t c) {
        return c >= '0' && c <= '9' ? (int)(c - '0') : c >= 'a' && c <= 'f' ? (int)(c - 'a' + 10)
             : c >= 'A' && c <= 'F' ? (int)(c - 'A' + 10) : -1;
    }

    bool parse_escape(bool in_class, Esc* e) {
        if (at_end()) return fail("a trailing backslash");
        const uint32_t c = peek();
        ++i_;
        e->kind = E_CHAR;
        switch (c) {
            case 't': e->ch = 9;  return true;
            case 'n': e->ch = 10; return true;
            case 'r': e->ch = 13; return true;
            case 'f': e->ch = 12; return true;
            case 'v': e->ch = 11; return true;
            case 'a': e->ch = 7;  return true;
            case 'e': e->ch = 27; return true;
            case '0': {
                uint32_t v = 0;
                for (int k = 0; k < 2 && peek() >= '0' && peek() <= '7'; ++k) { v = v * 8 + (peek() - '0'); ++i_; }
                e->ch = v;
                return true;
            }
            case 'x': {
                uint32_t v = 0;
                if (peek() == '{') {
                    ++i_;
                    int nd = 0;
                    while (hexval(peek()) >= 0 && nd < 8) { v = v * 16 + hexval(peek()); ++i_; ++nd; }
                    if (nd == 0 || peek() != '}') return fail("a malformed \\x{...} escape");
                    ++i_;
                } else {
                    int nd = 0;
                    while (hexval(peek()) >= 0 && nd < 2) { v = v * 16 + hexval(peek()); ++i_; ++nd; }
                    if (nd == 0) return fail("a malformed \\x escape");
                }
                if (v > kMaxCp) return fail("a codepoint above U+10FFFF");
                e->ch = v;
                return true;
            }
            case 'u': {
                uint32_t v = 0;
                for (int k = 0; k < 4; ++k) {
                    if (hexval(peek()) < 0) return fail("a malformed \\u escape");
                    v = v * 16 + hexval(peek());
                    ++i_;
                }
                e->ch = v;
                return true;
            }
            case 'c': {
                if (at_end()) return fail("a malformed \\c escape");
                e->ch = peek() & 0x1F;
                ++i_;
                return true;
            }
            case 'd': e->kind = E_SET; e->set = digit_set(); return true;
            case 'D': e->kind = E_SET; e->set = set_complement(digit_set()); return true;
            case 's': e->kind = E_SET; e->set = white_space_set(); return true;
            case 'S': e->kind = E_SET; e->set = set_complement(white_space_set()); return true;
            case 'w': case 'W': {
                /* Oniguruma's two definitions; see tokenizer_ucd.h. */
                const CpSet& w = in_class ? word_class_set() : word_set();
                e->kind = E_SET;
                e->set = c == 'w' ? w : set_complement(w);
                return true;
            }
            case 'h': e->kind = E_SET; e->set = hex_set(); return true;
            case 'H': e->kind = E_SET; e->set = set_complement(hex_set()); return true;
            case 'p': case 'P': {
                if (peek() != '{') return fail("a property escape without braces");
                ++i_;
                bool neg = c == 'P';
                if (peek() == '^') { neg = !neg; ++i_; }
                std::string name;
                while (!at_end() && peek() != '}') {
                    if (peek() >= 0x80) return fail("an unknown property");
                    name += (char)peek();
                    ++i_;
                }
                if (at_end()) return fail("an unterminated property escape");
                ++i_;
                CpSet s;
                if (!property_set(name, &s)) return fail("a property the tables do not carry");
                e->kind = E_SET;
                e->prop = true;
                e->set = neg ? set_complement(s) : s;
                return true;
            }
            case 'b':
                if (in_class) { e->ch = 8; return true; }
                e->kind = E_ASSERT; e->assert_kind = A_WORD_B; return true;
            case 'B': case 'A': case 'z':
                if (in_class) return fail("an assertion inside a character class");
                e->kind = E_ASSERT;
                e->assert_kind = c == 'B' ? A_NOT_WORD_B : c == 'A' ? A_TEXT_START : A_TEXT_END;
                return true;
            case 'Z': return fail("\\Z (end of text or before a final newline)");
            case 'G': return fail("\\G");
            case 'R': return fail("\\R (an atomic linebreak)");
            case 'X': return fail("\\X (a grapheme cluster)");
            case 'K': return fail("\\K");
            case 'Q': return fail("\\Q...\\E quoting");
            case 'k': case 'g': return fail("a backreference or subexpression call");
            default: break;
        }
        if (c >= '1' && c <= '9') return fail("a backreference");
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return fail("an unknown escape");
        e->ch = c;             /* an escaped metacharacter, or any other symbol, is itself */
        return true;
    }

    int parse_escape_atom(const Flags& f) {
        Esc e;
        if (!parse_escape(false, &e)) return -1;
        if (e.kind == E_ASSERT) return mk_assert(e.assert_kind);
        if (e.kind == E_SET) return mk_set(std::move(e.set));
        return literal(e.ch, f);
    }

    /* `[` already consumed. Oniguruma's classes: a leading `^`, nested classes as union, `&&` as
     * intersection, `-` literal at either end. Under `(?i)` the class is closed over the case
     * orbits before it is negated, so `(?i:[^a-z])` excludes 'A'. */
    bool parse_class(const Flags& f, CpSet* out) {
        const size_t at = start_;
        const bool neg = peek() == '^';
        if (neg) ++i_;
        CpSet acc, cur;
        bool have_acc = false, first = true, used_prop = false, nested = false;
        for (;;) {
            if (at_end()) { start_ = at; return fail("an unterminated character class"); }
            const uint32_t c = peek();
            if (c == ']' && !first) { ++i_; break; }
            if (c == '&' && peek(1) == '&') {
                i_ += 2;
                acc = have_acc ? set_intersect(acc, cur) : cur;
                have_acc = true;
                cur.clear();
                first = false;
                continue;
            }
            if (c == '[') {
                if (peek(1) == ':') return fail("a POSIX bracket expression");
                ++i_;
                CpSet inner;
                if (!parse_class(f, &inner)) return false;
                cur = set_union(cur, inner);
                nested = true;
                first = false;
                continue;
            }
            uint32_t lo;
            if (c == '\\') {
                ++i_;
                Esc e;
                if (!parse_escape(true, &e)) return false;
                if (e.kind == E_SET) {
                    used_prop = used_prop || e.prop;
                    cur = set_union(cur, e.set);
                    first = false;
                    continue;
                }
                lo = e.ch;
            } else {
                lo = c;
                ++i_;
            }
            first = false;
            if (peek() == '-' && peek(1) != ']' && peek(1) != 0xFFFFFFFFu) {
                ++i_;
                uint32_t hi;
                if (peek() == '\\') {
                    ++i_;
                    Esc e;
                    if (!parse_escape(true, &e)) return false;
                    if (e.kind != E_CHAR) return fail("a class range whose end is not a character");
                    hi = e.ch;
                } else if (peek() == '[') {
                    return fail("a class range whose end is not a character");
                } else {
                    hi = peek();
                    ++i_;
                }
                if (hi < lo) return fail("a class range out of order");
                cur = set_union(cur, { { lo, hi } });
            } else {
                cur = set_union(cur, { { lo, lo } });
            }
        }
        CpSet s = have_acc ? set_intersect(acc, cur) : cur;
        if (f.i) {
            start_ = at;
            if (have_acc) return fail("a class intersection under (?i)");
            if (used_prop) return fail("a property inside a class under (?i)");
            if (nested) return fail("a nested class under (?i)");
            if (!set_intersect(s, case_data().multi).empty())
                return fail("a case-insensitive class holding a character that also matches a "
                            "multi-codepoint sequence");
            s = case_close(s);
        }
        *out = neg ? set_complement(s) : s;
        return true;
    }
};

/* ================================================================== NFA */

enum NOp : uint8_t { N_CHAR, N_SPLIT, N_EPS, N_ASSERT, N_MATCH };

struct NState {
    uint8_t op;
    int     out = -1, out1 = -1;
    int     arg = -1;         /* N_CHAR: set; N_ASSERT: node index of the assertion */
};

constexpr size_t kMaxNfaStates = 200000;
constexpr size_t kMaxDfaStates = 20000;

class NfaBuilder {
public:
    NfaBuilder(const std::vector<Node>& nodes) : nodes_(nodes) {}
    std::vector<NState> st;
    bool too_big = false;

    int build(int root) {
        Frag f = frag(root);
        const int m = add(N_MATCH);
        patch(f.outs, m);
        return f.start;
    }

private:
    struct Frag { int start; std::vector<std::pair<int, int>> outs; };
    const std::vector<Node>& nodes_;

    int add(uint8_t op, int arg = -1) {
        if (st.size() >= kMaxNfaStates) too_big = true;
        NState s; s.op = op; s.arg = arg;
        st.push_back(s);
        return (int)st.size() - 1;
    }
    void patch(const std::vector<std::pair<int, int>>& outs, int to) {
        for (const auto& o : outs) (o.second ? st[o.first].out1 : st[o.first].out) = to;
    }

    Frag frag(int id) {
        const Node& n = nodes_[id];
        if (too_big) { const int s = add(N_EPS); return { s, { { s, 0 } } }; }
        switch (n.kind) {
            case Node::EMPTY: { const int s = add(N_EPS); return { s, { { s, 0 } } }; }
            case Node::SET:   { const int s = add(N_CHAR, n.set); return { s, { { s, 0 } } }; }
            case Node::ASSERT:{ const int s = add(N_ASSERT, id); return { s, { { s, 0 } } }; }
            case Node::CAT: {
                Frag a = frag(n.kids[0]);
                for (size_t k = 1; k < n.kids.size(); ++k) {
                    Frag b = frag(n.kids[k]);
                    patch(a.outs, b.start);
                    a.outs = std::move(b.outs);
                }
                return a;
            }
            case Node::ALT: {
                /* A chain of splits, each preferring the earlier alternative. */
                Frag last = frag(n.kids.back());
                int start = last.start;
                std::vector<std::pair<int, int>> outs = last.outs;
                for (size_t k = n.kids.size() - 1; k-- > 0; ) {
                    Frag a = frag(n.kids[k]);
                    const int s = add(N_SPLIT);
                    st[s].out = a.start;
                    st[s].out1 = start;
                    outs.insert(outs.end(), a.outs.begin(), a.outs.end());
                    start = s;
                }
                return { start, outs };
            }
            case Node::REP: {
                const int kid = n.kids[0];
                Frag acc{ -1, {} };
                auto append = [&](Frag f) {
                    if (acc.start < 0) acc = std::move(f);
                    else { patch(acc.outs, f.start); acc.outs = std::move(f.outs); }
                };
                for (int k = 0; k < n.min; ++k) append(frag(kid));
                if (n.max < 0) {
                    /* The loop: the split prefers another iteration when greedy, the exit when lazy. */
                    const int s = add(N_SPLIT);
                    Frag body = frag(kid);
                    patch(body.outs, s);
                    Frag loop{ s, {} };
                    if (n.greedy) { st[s].out = body.start; loop.outs = { { s, 1 } }; }
                    else          { st[s].out1 = body.start; loop.outs = { { s, 0 } }; }
                    append(std::move(loop));
                } else if (n.max > n.min) {
                    /* Nested optionals: x{0,2} is (x(x)?)?. */
                    std::vector<std::pair<int, int>> exits;
                    int first = -1;
                    std::vector<std::pair<int, int>> pending;
                    for (int k = n.min; k < n.max; ++k) {
                        const int s = add(N_SPLIT);
                        Frag body = frag(kid);
                        if (n.greedy) { st[s].out = body.start; exits.push_back({ s, 1 }); }
                        else          { st[s].out1 = body.start; exits.push_back({ s, 0 }); }
                        if (first < 0) first = s; else patch(pending, s);
                        pending = std::move(body.outs);
                    }
                    exits.insert(exits.end(), pending.begin(), pending.end());
                    append(Frag{ first, exits });
                }
                if (acc.start < 0) { const int s = add(N_EPS); acc = { s, { { s, 0 } } }; }
                return acc;
            }
        }
        const int s = add(N_EPS);
        return { s, { { s, 0 } } };
    }
};

}  /* namespace */

/* ================================================================== DFA */

namespace {

/* Decode one codepoint of canonical UTF-8. */
inline uint32_t decode_canon(const uint8_t* p, uint32_t* len) {
    const uint32_t b = p[0];
    if (b < 0x80) { *len = 1; return b; }
    if (b < 0xE0) { *len = 2; return ((b & 0x1F) << 6) | (p[1] & 0x3F); }
    if (b < 0xF0) { *len = 3; return ((b & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); }
    *len = 4;
    return ((b & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
}

}  /* namespace */

struct TokRegex::Dfa {
    static constexpr uint32_t kMatch = 0x80000000u;

    /* States are named by their row offset, state * ncls, so the inner loop indexes the table
     * with one add. The dead state is row 0. */
    uint32_t ncls = 0;
    std::vector<uint32_t> trans;          /* row + class -> next row | kMatch */
    std::vector<uint8_t>  eoi_match;      /* per state: a match ends at the end of the text */
    uint32_t start_text = 0;              /* start row at the beginning of the text */
    std::vector<uint32_t> start_after;    /* start row after a codepoint of each class */
    bool uses_behind = false;

    /* The same automaton with rows of 128, for the scan: trans_ascii is indexed by the byte
     * itself, so an ASCII step is one load in the state chain where the class table needs two
     * (byte -> class, then row + class). Shipped pre-tokeniser patterns determinise to tens of
     * states, so the byte table is tens of KiB; past the cap it is left empty and the class
     * tables serve. trans_cls is `trans` re-encoded to rows of 128 for the non-ASCII steps.
     * State 0 is the dead row in both encodings. */
    static constexpr uint32_t kAsciiRowsMax = 512;
    std::vector<uint32_t> trans_ascii;    /* state * 128 + byte -> next state * 128 | kMatch */
    std::vector<uint32_t> trans_cls;      /* state * ncls + class -> next state * 128 | kMatch */
    uint32_t              start_text_a = 0;
    std::vector<uint32_t> start_after_a;

    uint8_t  ascii[128];
    std::vector<uint16_t> stage1;         /* cp >> 8 -> block */
    std::vector<uint8_t>  stage2;         /* block * 256 + (cp & 255) -> class */

    uint32_t cls_of(uint32_t cp) const {
        return cp < 128 ? ascii[cp] : stage2[((uint32_t)stage1[cp >> 8] << 8) | (cp & 0xFF)];
    }

    /* The class of the codepoint that ends just before byte `pos`. */
    uint32_t prev_class(const uint8_t* s, size_t pos) const {
        size_t b = pos - 1;
        while (b > 0 && (s[b] & 0xC0) == 0x80 && pos - b < 4) --b;
        uint32_t len;
        return cls_of(decode_canon(s + b, &len));
    }
};


TokRegex::~TokRegex() = default;
size_t TokRegex::n_states() const { return dfa_ ? dfa_->eoi_match.size() : 0; }
size_t TokRegex::n_classes() const { return dfa_ ? dfa_->ncls : 0; }

int TokRegex::compile(std::string_view pattern, std::unique_ptr<TokRegex>* out, std::string* why) {
    auto refuse = [&](const std::string& m) {
        if (why) *why = m;
        return RAD_E_UNSUPPORTED;
    };

    Parser p(pattern);
    int root = -1;
    if (!p.parse(&root)) return refuse(p.err);

    NfaBuilder nb(p.nodes);
    const int start = nb.build(root);
    if (nb.too_big) return refuse(fmt("the pattern expands to more than %zu NFA states", kMaxNfaStates));
    const std::vector<NState>& nfa = nb.st;

    /* ---- the sets the DFA has to tell apart, and the assertions that read them. */
    std::vector<CpSet>& sets = p.sets;
    const int nl_set = (int)sets.size();
    sets.push_back({ { '\n', '\n' } });
    const int word = (int)sets.size();
    sets.push_back(word_set());

    std::vector<int> behind;                 /* sets whose membership the state remembers */
    auto behind_bit = [&](int set) -> uint32_t {
        for (size_t k = 0; k < behind.size(); ++k)
            if (behind[k] == set) return 1u << (k + 1);
        behind.push_back(set);
        return 1u << behind.size();
    };
    struct Look { uint8_t kind; int set; uint32_t bbit; };
    std::unordered_map<int, Look> looks;     /* NFA assert state -> what it tests */
    bool uses_behind = false;
    std::vector<uint8_t> set_used(sets.size(), 0);
    for (size_t s = 0; s < nfa.size(); ++s) {
        if (nfa[s].op == N_CHAR) set_used[nfa[s].arg] = 1;
        if (nfa[s].op != N_ASSERT) continue;
        const Node& a = p.nodes[nfa[s].arg];
        Look l{ a.assert_kind, a.set, 0 };
        switch (a.assert_kind) {
            case A_LINE_START: l.set = nl_set; l.bbit = behind_bit(nl_set); uses_behind = true; break;
            case A_LINE_END:   l.set = nl_set; break;
            case A_TEXT_START: uses_behind = true; break;
            case A_TEXT_END:   break;
            case A_WORD_B: case A_NOT_WORD_B:
                l.set = word; l.bbit = behind_bit(word); uses_behind = true; break;
            case A_BEHIND: case A_BEHIND_NOT:
                l.bbit = behind_bit(a.set); uses_behind = true; break;
            default: break;
        }
        if (l.set >= 0) set_used[l.set] = 1;
        looks[(int)s] = l;
    }
    if (behind.size() > 30) return refuse("more than 30 distinct lookbehind sets");

    /* ---- the alphabet partition: codepoints no set tells apart share a class. */
    std::vector<int> used;
    for (size_t k = 0; k < sets.size(); ++k) if (set_used[k]) used.push_back((int)k);
    std::vector<uint32_t> bounds = { 0, kMaxCp + 1 };
    for (int k : used)
        for (const auto& r : sets[k]) { bounds.push_back(r.first); bounds.push_back(r.second + 1); }
    std::sort(bounds.begin(), bounds.end());
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());

    auto d = std::make_unique<Dfa>();
    std::vector<uint8_t> cls_full(kMaxCp + 1);
    std::unordered_map<std::string, uint32_t> sig_to_cls;
    std::vector<std::vector<uint8_t>> in_set;   /* class -> membership per set index */
    for (size_t b = 0; b + 1 < bounds.size(); ++b) {
        const uint32_t lo = bounds[b], hi = bounds[b + 1];
        std::string sig(used.size(), '\0');
        for (size_t k = 0; k < used.size(); ++k) sig[k] = set_has(sets[used[k]], lo) ? 1 : 0;
        auto it = sig_to_cls.find(sig);
        uint32_t c;
        if (it == sig_to_cls.end()) {
            c = (uint32_t)in_set.size();
            if (c >= 256) return refuse("the pattern needs more than 256 codepoint classes");
            sig_to_cls.emplace(sig, c);
            std::vector<uint8_t> m(sets.size(), 0);
            for (size_t k = 0; k < used.size(); ++k) m[used[k]] = (uint8_t)sig[k];
            in_set.push_back(std::move(m));
        } else {
            c = it->second;
        }
        std::fill(cls_full.begin() + lo, cls_full.begin() + hi, (uint8_t)c);
    }
    d->ncls = (uint32_t)in_set.size();
    for (int c = 0; c < 128; ++c) d->ascii[c] = cls_full[c];
    {
        std::unordered_map<std::string, uint16_t> blocks;
        d->stage1.resize((kMaxCp + 1) >> 8);
        for (uint32_t blk = 0; blk < d->stage1.size(); ++blk) {
            std::string key((const char*)cls_full.data() + blk * 256, 256);
            auto it = blocks.find(key);
            if (it == blocks.end()) {
                const uint16_t id = (uint16_t)blocks.size();
                blocks.emplace(key, id);
                d->stage2.insert(d->stage2.end(), key.begin(), key.end());
                d->stage1[blk] = id;
            } else {
                d->stage1[blk] = it->second;
            }
        }
    }

    /* What the state remembers about the codepoint just consumed. Bit 0 is "at the start". */
    std::vector<uint32_t> lb_after(d->ncls, 0);
    for (uint32_t c = 0; c < d->ncls; ++c)
        for (size_t k = 0; k < behind.size(); ++k)
            if (in_set[c][behind[k]]) lb_after[c] |= 1u << (k + 1);
    /* next < 0 is the end of the text. */
    auto holds = [&](const Look& l, uint32_t lb, int next) -> bool {
        const bool at_start = (lb & 1) != 0;
        auto in = [&](int set) { return next >= 0 && in_set[next][set] != 0; };
        switch (l.kind) {
            case A_LINE_START: return at_start || (lb & l.bbit);
            case A_LINE_END:   return next < 0 || in(l.set);
            case A_TEXT_START: return at_start;
            case A_TEXT_END:   return next < 0;
            case A_WORD_B:     return (!at_start && (lb & l.bbit)) != in(l.set);
            case A_NOT_WORD_B: return (!at_start && (lb & l.bbit)) == in(l.set);
            case A_AHEAD:      return in(l.set);
            case A_AHEAD_NOT:  return !in(l.set);
            case A_BEHIND:     return !at_start && (lb & l.bbit);
            case A_BEHIND_NOT: return at_start || !(lb & l.bbit);
        }
        return false;
    };

    /* ---- subset construction, leftmost-first.
     *
     * A DFA state is the ordered list of NFA states reached by consuming the last codepoint
     * (before their epsilon closure), plus what is remembered of that codepoint. The closure is
     * taken on the transition, when the next codepoint's class is known, so a lookahead is
     * decided exactly. Walking the closure in priority order, reaching MATCH records a match and
     * drops every lower-priority thread, which a backtracking engine would never reach. Higher-
     * priority threads survive, and a later match from one of them replaces this one. */
    std::vector<uint32_t> mark(nfa.size(), 0), mark2(nfa.size(), 0);
    uint32_t stamp = 0, stamp2 = 0;
    std::vector<int> stack, chars, next_kernel;

    auto closure = [&](const std::vector<int>& kernel, uint32_t lb, int next,
                       std::vector<int>* next_out) -> bool {
        ++stamp;
        chars.clear();
        bool matched = false;
        for (int k0 : kernel) {
            stack.clear();
            stack.push_back(k0);
            while (!stack.empty() && !matched) {
                const int s = stack.back();
                stack.pop_back();
                if (s < 0 || mark[s] == stamp) continue;
                mark[s] = stamp;
                const NState& ns = nfa[s];
                switch (ns.op) {
                    case N_CHAR:   chars.push_back(s); break;
                    case N_EPS:    stack.push_back(ns.out); break;
                    case N_SPLIT:  stack.push_back(ns.out1); stack.push_back(ns.out); break;
                    case N_ASSERT: if (holds(looks[s], lb, next)) stack.push_back(ns.out); break;
                    case N_MATCH:  matched = true; break;
                }
            }
            if (matched) break;
        }
        if (next_out) {
            next_out->clear();
            if (next >= 0) {
                ++stamp2;
                for (int s : chars) {
                    if (!in_set[next][nfa[s].arg]) continue;
                    const int t = nfa[s].out;
                    if (mark2[t] == stamp2) continue;
                    mark2[t] = stamp2;
                    next_out->push_back(t);
                }
            }
        }
        return matched;
    };

    std::unordered_map<std::string, uint32_t> ids;
    std::vector<std::vector<int>> kernels;
    std::vector<uint32_t> lbs;
    auto intern = [&](const std::vector<int>& k, uint32_t lb) -> uint32_t {
        if (k.empty()) return 0;
        std::string key((const char*)k.data(), k.size() * sizeof(int));
        key.append((const char*)&lb, sizeof lb);
        auto it = ids.find(key);
        if (it != ids.end()) return it->second;
        const uint32_t id = (uint32_t)kernels.size();
        ids.emplace(std::move(key), id);
        kernels.push_back(k);
        lbs.push_back(lb);
        return id;
    };
    kernels.push_back({});          /* 0: dead */
    lbs.push_back(0);

    d->uses_behind = uses_behind;
    d->start_text = intern({ start }, uses_behind ? 1u : 0u);
    d->start_after.assign(d->ncls, d->start_text);
    if (uses_behind)
        for (uint32_t c = 0; c < d->ncls; ++c) d->start_after[c] = intern({ start }, lb_after[c]);

    for (uint32_t s = 0; s < kernels.size(); ++s) {
        if (kernels.size() > kMaxDfaStates)
            return refuse(fmt("the pattern determinises to more than %zu states", kMaxDfaStates));
        d->trans.resize((size_t)kernels.size() * d->ncls, 0);
        d->eoi_match.resize(kernels.size(), 0);
        if (s == 0) continue;
        /* A copy: intern() may reallocate `kernels`. Row s of `trans` exists from the resize
         * above, and the states intern() adds get theirs on their own iteration. */
        const std::vector<int> kernel = kernels[s];
        const uint32_t lb = lbs[s];
        for (uint32_t c = 0; c < d->ncls; ++c) {
            const bool m = closure(kernel, lb, (int)c, &next_kernel);
            const uint32_t t = intern(next_kernel, uses_behind ? lb_after[c] : 0);
            d->trans[(size_t)s * d->ncls + c] = t | (m ? Dfa::kMatch : 0);
        }
        d->eoi_match[s] = closure(kernel, lb, -1, nullptr) ? 1 : 0;
    }
    d->trans.resize((size_t)kernels.size() * d->ncls, 0);
    d->eoi_match.resize(kernels.size(), 0);
    for (uint32_t& t : d->trans) t = ((t & ~Dfa::kMatch) * d->ncls) | (t & Dfa::kMatch);
    d->start_text *= d->ncls;
    for (uint32_t& t : d->start_after) t *= d->ncls;

    /* The byte-indexed copy for the scan (Dfa::trans_ascii), converted once here where the
     * division back to state ids costs nothing. */
    if (kernels.size() <= Dfa::kAsciiRowsMax) {
        const uint32_t ns = (uint32_t)kernels.size(), nc = d->ncls;
        d->trans_cls.assign((size_t)ns * nc, 0);
        for (uint32_t s = 0; s < ns; ++s)
            for (uint32_t c = 0; c < nc; ++c) {
                const uint32_t t = d->trans[(size_t)s * nc + c];
                d->trans_cls[(size_t)s * nc + c] =
                    ((t & ~Dfa::kMatch) / nc) * 128u | (t & Dfa::kMatch);
            }
        d->trans_ascii.assign((size_t)ns * 128, 0);
        for (uint32_t s = 0; s < ns; ++s)
            for (uint32_t b = 0; b < 128; ++b)
                d->trans_ascii[(size_t)s * 128 + b] =
                    d->trans_cls[(size_t)s * nc + d->ascii[b]];
        d->start_text_a = (d->start_text / nc) * 128u;
        d->start_after_a.assign(d->start_after.size(), 0);
        for (size_t c = 0; c < d->start_after.size(); ++c)
            d->start_after_a[c] = (d->start_after[c] / nc) * 128u;
    }

    std::unique_ptr<TokRegex> re(new TokRegex());
    re->pattern_ = std::string(pattern);
    re->dfa_ = std::move(d);
    *out = std::move(re);
    return RAD_OK;
}

int64_t TokRegex::match_at(std::string_view text, size_t pos) const {
    const Dfa& d = *dfa_;
    const uint8_t* s = (const uint8_t*)text.data();
    const size_t n = text.size();
    int64_t best = -1;
    size_t q = pos;
    if (!d.trans_ascii.empty()) {
        /* Rows of 128: an ASCII byte indexes its transition directly. */
        uint32_t state = pos == 0 || !d.uses_behind ? d.start_text_a
                                                    : d.start_after_a[d.prev_class(s, pos)];
        const uint32_t* ta = d.trans_ascii.data();
        const uint32_t* tc = d.trans_cls.data();
        while (q < n) {
            uint32_t len, t;
            const uint32_t b = s[q];
            if (b < 0x80) { t = ta[state + b]; len = 1; }
            else          { t = tc[(state >> 7) * d.ncls + d.cls_of(decode_canon(s + q, &len))]; }
            if (t & Dfa::kMatch) best = (int64_t)q;
            state = t & ~Dfa::kMatch;
            if (state == 0) return best;
            q += len;
        }
        if (d.eoi_match[state >> 7]) best = (int64_t)n;
        return best;
    }
    uint32_t state = pos == 0 || !d.uses_behind ? d.start_text : d.start_after[d.prev_class(s, pos)];
    const uint32_t* trans = d.trans.data();
    while (q < n) {
        uint32_t len, cls;
        const uint32_t b = s[q];
        if (b < 0x80) { cls = d.ascii[b]; len = 1; }
        else          { cls = d.cls_of(decode_canon(s + q, &len)); }
        const uint32_t t = trans[state + cls];
        if (t & Dfa::kMatch) best = (int64_t)q;
        state = t & ~Dfa::kMatch;
        if (state == 0) return best;
        q += len;
    }
    if (d.eoi_match[state / d.ncls]) best = (int64_t)n;
    return best;
}

void TokRegex::find_iter(std::string_view text, std::vector<TokSpan>& out) const {
    /* Oniguruma's iteration as the `onig` crate drives it: search from the end of the previous
     * match for the leftmost position with a match; an empty match exactly at the end of the
     * previous one is skipped by advancing one codepoint and searching again. */
    if (!dfa_->uses_behind) {
        /* Fused into one walk: the loop re-enters the DFA at every search position exactly as
         * the match_at loop below does -- same restarts, same rewalks of a dead run's tail,
         * same empty-match rule -- without a call and a fresh setup per piece, which at a piece
         * every few bytes was a visible share of splitting. The tag selects the byte-indexed
         * table (rows of 128) over the class table (rows of ncls); the state encoding differs,
         * hence the two instantiations. The decision at each death (or at the end of the text)
         * is the loop below's, word for word: e is what match_at(pos) would have returned. */
        const Dfa& d = *dfa_;
        auto run = [&](auto ascii_tag) {
            constexpr bool Ascii = decltype(ascii_tag)::value;
            const uint8_t* s = (const uint8_t*)text.data();
            const size_t n = text.size();
            const uint32_t* ta = Ascii ? d.trans_ascii.data() : d.trans.data();
            const uint32_t start = Ascii ? d.start_text_a : d.start_text;
            size_t pos = 0;            /* search start */
            bool have_prev = false;
            size_t prev_end = 0;
            uint32_t state = start;
            int64_t best = -1;         /* match end match_at(pos) would report so far */
            size_t q = 0;              /* walk position */
            for (;;) {
                int64_t e;
                if (q < n) {
                    uint32_t len, t;
                    const uint32_t b = s[q];
                    if (b < 0x80) {
                        t = Ascii ? ta[state + b] : ta[state + d.ascii[b]];
                        len = 1;
                    } else if (Ascii) {
                        t = d.trans_cls[(state >> 7) * d.ncls +
                                        d.cls_of(decode_canon(s + q, &len))];
                    } else {
                        t = ta[state + d.cls_of(decode_canon(s + q, &len))];
                    }
                    if (t & Dfa::kMatch) best = (int64_t)q;
                    state = t & ~Dfa::kMatch;
                    if (state != 0) { q += len; continue; }
                    e = best;              /* died: match_at(pos) returns here */
                } else {
                    const uint32_t sid = Ascii ? state >> 7 : state / d.ncls;
                    e = d.eoi_match[sid] ? (int64_t)n : best;
                }
                if (e >= 0 && !((size_t)e == pos && have_prev && prev_end == pos)) {
                    out.push_back({ (uint32_t)pos, (uint32_t)e });
                    have_prev = true;
                    prev_end = (size_t)e;
                    if ((size_t)e > pos) {
                        pos = (size_t)e;
                        q = pos;
                        state = start;
                        best = -1;
                        continue;
                    }
                }
                if (pos == n) break;
                uint32_t len;
                decode_canon(s + pos, &len);
                pos += len;
                q = pos;
                state = start;
                best = -1;
            }
        };
        if (!d.trans_ascii.empty()) run(std::true_type{});
        else                        run(std::false_type{});
        return;
    }
    const uint8_t* s = (const uint8_t*)text.data();
    const size_t n = text.size();
    size_t pos = 0;
    bool have_prev = false;
    size_t prev_end = 0;
    while (pos <= n) {
        const int64_t e = match_at(text, pos);
        if (e >= 0 && !((size_t)e == pos && have_prev && prev_end == pos)) {
            out.push_back({ (uint32_t)pos, (uint32_t)e });
            have_prev = true;
            prev_end = (size_t)e;
            if ((size_t)e > pos) { pos = (size_t)e; continue; }
        }
        if (pos == n) break;
        uint32_t len;
        decode_canon(s + pos, &len);
        pos += len;
    }
}

void tok_split(const TokRegex& re, std::string_view text, int behavior, bool invert,
               std::vector<TokSpan>& out, std::vector<TokSpan>& m) {
    m.clear();
    re.find_iter(text, m);
    const uint32_t n = (uint32_t)text.size();

    /* Isolated keeps every match and every gap as its own piece, so `invert` cannot change it. */
    if (behavior == TOK_SPLIT_ISOLATED) {
        uint32_t prev = 0;
        for (const TokSpan& x : m) {
            if (x.begin > prev) out.push_back({ prev, x.begin });
            if (x.end > x.begin) out.push_back(x);
            prev = x.end;
        }
        if (prev < n) out.push_back({ prev, n });
        return;
    }

    /* The general case follows NormalizedString::split step by step: matches and gaps in order
     * (empty matches included), flipped by `invert`, folded by the behaviour, then the removed
     * and the empty pieces dropped. */
    struct Part { uint32_t b, e; bool match; };
    std::vector<Part> parts;
    {
        uint32_t prev = 0;
        for (const TokSpan& x : m) {
            if (x.begin != prev) parts.push_back({ prev, x.begin, false });
            parts.push_back({ x.begin, x.end, true });
            prev = x.end;
        }
        if (prev != n) parts.push_back({ prev, n, false });
        if (invert) for (Part& p : parts) p.match = !p.match;
    }
    struct Keep { uint32_t b, e; bool remove; };
    std::vector<Keep> keep;
    switch (behavior) {
        case TOK_SPLIT_REMOVED:
            for (const Part& p : parts) keep.push_back({ p.b, p.e, p.match });
            break;
        case TOK_SPLIT_CONTIGUOUS: {
            bool prev_match = false;
            for (const Part& p : parts) {
                if (p.match == prev_match && !keep.empty()) keep.back().e = p.e;
                else keep.push_back({ p.b, p.e, false });
                prev_match = p.match;
            }
            break;
        }
        case TOK_SPLIT_MERGED_PREV: {
            bool prev_match = false;
            for (const Part& p : parts) {
                if (p.match && !prev_match && !keep.empty()) keep.back().e = p.e;
                else keep.push_back({ p.b, p.e, false });
                prev_match = p.match;
            }
            break;
        }
        case TOK_SPLIT_MERGED_NEXT: {
            bool prev_match = false;
            for (size_t k = parts.size(); k-- > 0; ) {
                const Part& p = parts[k];
                if (p.match && !prev_match && !keep.empty()) keep.back().b = p.b;
                else keep.push_back({ p.b, p.e, false });
                prev_match = p.match;
            }
            std::reverse(keep.begin(), keep.end());
            break;
        }
        default:
            for (const Part& p : parts) keep.push_back({ p.b, p.e, false });
            break;
    }
    for (const Keep& k : keep)
        if (!k.remove && k.e > k.b) out.push_back({ k.b, k.e });
}

void tok_replace(const TokRegex& re, std::string_view text, std::string_view content,
                 std::string& out) {
    std::vector<TokSpan> m;
    re.find_iter(text, m);
    uint32_t prev = 0;
    for (const TokSpan& x : m) {
        out.append(text.data() + prev, x.begin - prev);
        out.append(content);
        prev = x.end;
    }
    out.append(text.data() + prev, text.size() - prev);
}

bool utf8_is_canonical(std::string_view str) {
    const uint8_t* p = (const uint8_t*)str.data();
    const size_t n = str.size();
    size_t i = 0;
    while (i < n) {
        /* ASCII is canonical as it stands: jump to the next non-ASCII byte. */
        i += simd::first_non_ascii((const char*)p + i, n - i);
        if (i >= n) break;
        const uint8_t b = p[i];
        if (b < 0xC2) return false;                         /* continuation, or overlong C0/C1 */
        if (b < 0xE0) {
            if (i + 1 >= n || (p[i + 1] & 0xC0) != 0x80) return false;
            i += 2;
            continue;
        }
        if (b < 0xF0) {
            if (i + 2 >= n || (p[i + 1] & 0xC0) != 0x80 || (p[i + 2] & 0xC0) != 0x80) return false;
            if (b == 0xE0 && p[i + 1] < 0xA0) return false; /* overlong */
            i += 3;
            continue;
        }
        if (b < 0xF5) {
            if (i + 3 >= n || (p[i + 1] & 0xC0) != 0x80 || (p[i + 2] & 0xC0) != 0x80 ||
                (p[i + 3] & 0xC0) != 0x80) return false;
            if (b == 0xF0 && p[i + 1] < 0x90) return false; /* overlong */
            if (b == 0xF4 && p[i + 1] > 0x8F) return false; /* above U+10FFFF */
            i += 4;
            continue;
        }
        return false;
    }
    return true;
}

void utf8_canonicalize(std::string_view str, std::string& out) {
    const uint8_t* p = (const uint8_t*)str.data();
    const size_t n = str.size();
    auto cont = [&](size_t k) { return k < n && (p[k] & 0xC0) == 0x80; };
    auto put = [&](uint32_t c) {
        if (c < 0x80) { out += (char)c; return; }
        if (c < 0x800) { out += (char)(0xC0 | (c >> 6)); out += (char)(0x80 | (c & 0x3F)); return; }
        if (c < 0x10000) {
            out += (char)(0xE0 | (c >> 12));
            out += (char)(0x80 | ((c >> 6) & 0x3F));
            out += (char)(0x80 | (c & 0x3F));
            return;
        }
        out += (char)(0xF0 | (c >> 18));
        out += (char)(0x80 | ((c >> 12) & 0x3F));
        out += (char)(0x80 | ((c >> 6) & 0x3F));
        out += (char)(0x80 | (c & 0x3F));
    };
    /* unicode_cpts_from_utf8's rules, as used by the rest of the chain: a lead byte followed by
     * the continuations it announces decodes (overlong forms and surrogates included); anything
     * else is one U+FFFD per byte. */
    for (size_t i = 0; i < n; ) {
        const uint8_t b = p[i];
        if (b < 0x80) { out += (char)b; ++i; continue; }
        if (!(b & 0x40)) { put(0xFFFD); ++i; continue; }
        if (!(b & 0x20)) {
            if (!cont(i + 1)) { put(0xFFFD); ++i; continue; }
            put(((b & 0x1F) << 6) | (p[i + 1] & 0x3F));
            i += 2;
            continue;
        }
        if (!(b & 0x10)) {
            if (!cont(i + 1) || !cont(i + 2)) { put(0xFFFD); ++i; continue; }
            put(((b & 0x0F) << 12) | ((p[i + 1] & 0x3F) << 6) | (p[i + 2] & 0x3F));
            i += 3;
            continue;
        }
        if (!(b & 0x08)) {
            if (!cont(i + 1) || !cont(i + 2) || !cont(i + 3)) { put(0xFFFD); ++i; continue; }
            const uint32_t c = ((b & 0x07) << 18) | ((p[i + 1] & 0x3F) << 12) |
                               ((p[i + 2] & 0x3F) << 6) | (p[i + 3] & 0x3F);
            if (c > kMaxCp) { put(0xFFFD); ++i; continue; }
            put(c);
            i += 4;
            continue;
        }
        put(0xFFFD);
        ++i;
    }
}

}  /* namespace rad */
