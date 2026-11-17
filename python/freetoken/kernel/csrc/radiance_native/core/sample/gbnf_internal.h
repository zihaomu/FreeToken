/* gbnf_internal.h -- shared by the GBNF state machine's three translation units only.
 *
 *   gbnf.cpp        the lifted machine: parser, stack machinery, accepting tokens, the compiled
 *                   program and its caches.
 *   gbnf_mask.cpp   the mask engine: the vocabulary index, the stack-set automaton and the
 *                   precomputed mask plan every fill_mask is served from.
 *   tests/gbnf_oracle.cpp
 *                   upstream's candidate walk, the test oracle for the mask engine; never on a
 *                   serving path.
 *
 * The stack machinery is here so all three run the same closure and character tests: the oracle
 * reimplements the mask, not the parse, so any disagreement is a mask-engine defect.
 */
#pragma once
#include "gbnf.h"

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rad {
namespace gbnf {

/* ============================================================ what a grammar may cost

 * Grammar text is client data, compiled and walked on shared threads: the HTTP thread that
 * admits the request and the scheduler thread that serves every request. So no grammar may make
 * either use unbounded memory, time or stack. Each bound below is set well above real grammars:
 * a JSON schema of 500 tool calls compiles to about 1.3 MB of text, 22,000 rules and a few MB of
 * compiled grammar, nests parentheses two deep, and holds at most about 1,500 stacks at once; a
 * schema nested 128 objects deep holds stacks about 130 frames tall.
 *
 * Bytes a grammar may compile to beyond its own text. The parse is linear in the text except
 * where a construct amplifies it: a repetition copies its operand once per count, and a
 * generated rule's name repeats its parent's name. Those are charged here, so a 100 KB literal
 * under `{1999}` is refused before it allocates 1.6 GB, and a cached program is bounded by its
 * text plus this. */
inline constexpr uint64_t kMaxCompileBytes = 32ull << 20;

/* Maximum parenthesis nesting. The recursive-descent parser uses two frames per level, so
 * unbounded depth lets 40 KB of '(' overflow the stack. */
inline constexpr uint32_t kMaxNesting = 256;

/* Limits on one stack set (the automaton's whole state at a position): distinct stacks, and
 * elements across all of them. Ambiguity or multiplying closure paths can grow the set without
 * end, and every accept, mask and cached state copies it. */
inline constexpr size_t kMaxStacks   = 1u << 14;
inline constexpr size_t kMaxSetElems = 1u << 18;

/* Stack elements one epsilon closure may push while exploring, and one set may hold as explored.
 * Without left recursion, closure paths can still multiply (each level of
 * `r ::= s "x" | s "y"` doubles them), and the set bound above is only checked after exploring;
 * this bound stops the exploring itself. */
inline constexpr size_t kMaxClosureWork = 1u << 20;

/* Work one operation (one accepted token, one mask) may do. Size bounds do not cap time, so work
 * is counted per operation in units of roughly a nanosecond: one per stack element copied or
 * char-range element tested, plus kStackCost per stack created (an allocation and an insertion
 * into the explored set). Past the limit the request fails rather than hold the scheduler thread
 * for more than about half a second. */
inline constexpr uint64_t kStackCost = 64;
inline constexpr uint64_t kMaxOpWork = 1ull << 29;

/* Work one compile may do, in the same units. The mask plan (gbnf_mask.cpp) walks the vocabulary
 * once per place a mask can start. On a quarter-million-piece vocabulary a tool-calling grammar
 * needs tens of millions of units, 200 tools about 200 million, and the worst case (a string
 * bounded past the longest token's length) under two billion. Compiling runs on the admitting
 * thread, never the scheduler thread; a grammar past this is refused within a few seconds. */
inline constexpr uint64_t kMaxCompileWork = 1ull << 31;

/* Bytes one compiled plan may hold (precomputed masks and remainder tries), i.e. what one cached
 * grammar may cost the process. On a quarter-million-piece vocabulary a 200-tool grammar holds
 * about 16 MiB, and a string bounded past the longest token about 40 MiB. */
inline constexpr size_t kMaxPlanBytes = 64ull << 20;

/* Total bytes of the programs a vocabulary's cache pins, alongside its pin count. Bounds only
 * programs no request holds; a held program is always kept. */
inline constexpr size_t kMaxPinnedBytes = 256ull << 20;

/* The stack-set automaton is a memo that grows as states are discovered, without end for a
 * grammar whose stacks deepen every token. Past the soft bound it is dropped and rebuilt lazily
 * at the next mask (costing only warm-up); the hard bound is what one operation may add before
 * it is refused. */
inline constexpr size_t kMaxDfaElems     = 1u << 22;
inline constexpr size_t kMaxDfaWalkElems = 1u << 23;

/* A bound above was reached. Thrown from the stack machinery, which is lifted, partly recursive
 * code with no status path, and caught at every public entry point, where it becomes a status
 * that refuses or fails the request. */
struct GbnfLimit : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void throw_set_limit();

/* The work counter for one operation, passed to everything the operation calls. */
struct OpWork {
    uint64_t used  = 0;
    uint64_t limit = kMaxOpWork;

    void add(uint64_t n) {
        used += n;
        if (used > limit) over();
    }

private:
    [[noreturn]] void over() const;
};

uint64_t stack_hash(const GbnfStack& s);

struct StackLess {
    bool operator()(const GbnfStack& a, const GbnfStack& b) const {
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
            [](const GbnfElement* pa, const GbnfElement* pb) { return pa < pb; });
    }
};

/* One stack set under construction, shared by every call that adds to it: its running size (so
 * the set bounds apply to the whole set, not per call), the operation whose work each addition is
 * charged to, and the stacks already explored.
 *
 * `seen` belongs to the set, not to one closure: a stack a closure has popped already has its
 * whole closure in the set, so a later closure reaching it adds nothing. A per-call `seen` would
 * re-expand a shared continuation once per converging stack (e.g. every alternative of a wide
 * rule), quadratic in the width. The resulting set and its order are the same either way. */
struct SetBuild {
    explicit SetBuild(OpWork& w) : work(w) {}

    OpWork& work;
    size_t  stacks = 0, elems = 0;

    std::set<GbnfStack, StackLess> seen;
    size_t                         seen_elems = 0;

    void add(size_t stack_size) {
        stacks += 1;
        elems  += stack_size + 1;
        if (stacks > kMaxStacks || elems > kMaxSetElems) throw_set_limit();
    }

    /* Append `s` to `out` (this build's set) unless already present. Small sets use the lifted
     * linear search, fastest at real state sizes; past kIndexFrom the set is indexed by hash,
     * since converging stacks would make linear search quadratic. Insertion order is the same
     * either way. */
    void insert(GbnfStacks& out, GbnfStack s);

private:
    static constexpr size_t kIndexFrom = 16;
    std::unordered_multimap<uint64_t, size_t> index;
};

/* ============================================================ UTF-8 */

/* One code point from NUL-terminated text. Assumes valid UTF-8 but checks for overrun. */
std::pair<uint32_t, const char*> decode_utf8(const char* src);

/* Decode a piece, carrying an incomplete sequence in from the previous token and out to the next.
 * The returned code-point vector is 0-terminated. An invalid sequence leaves no code points and
 * a carried state of {0, -1}. */
std::pair<std::vector<uint32_t>, PartialUtf8> decode_utf8(const std::string& src,
                                                         PartialUtf8 partial_start);

/* The same into `out`, whose capacity is reused; returns the carried state. */
PartialUtf8 decode_utf8(const std::string& src, PartialUtf8 partial_start,
                        std::vector<uint32_t>* out);

/* ============================================================ elements */

inline bool is_end_of_sequence(const GbnfElement* pos) {
    return pos->type == GRE_END || pos->type == GRE_ALT;
}

inline bool is_char_elem(const GbnfElement* pos) {
    return pos->type == GRE_CHAR || pos->type == GRE_CHAR_NOT || pos->type == GRE_CHAR_ANY;
}

inline bool is_token_elem(const GbnfElement* pos) {
    return pos->type == GRE_TOKEN || pos->type == GRE_TOKEN_NOT;
}

/* True iff chr satisfies the char range at pos (regular or inverse), and the element after the
 * range. pos must point at a char range element. */
inline std::pair<bool, const GbnfElement*> match_char(const GbnfElement* pos, uint32_t chr) {
    bool found = false;
    const bool is_positive = pos->type == GRE_CHAR || pos->type == GRE_CHAR_ANY;
    do {
        if (pos[1].type == GRE_CHAR_RNG_UPPER) {
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else if (pos->type == GRE_CHAR_ANY) {
            found = true;
            pos += 1;
        } else {
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == GRE_CHAR_ALT);
    return { found == is_positive, pos };
}

/* True iff some completion of a partial UTF-8 sequence could satisfy the range at pos, so a
 * byte-level BPE token ending mid-codepoint stays admissible. */
inline bool match_partial_char(const GbnfElement* pos, PartialUtf8 partial) {
    const bool is_positive = pos->type == GRE_CHAR || pos->type == GRE_CHAR_ANY;

    const uint32_t partial_value = partial.value;
    const int      n_remain      = partial.n_remain;
    if (n_remain < 0 || (n_remain == 1 && partial_value < 2)) return false;

    uint32_t low  = partial_value << (n_remain * 6);
    const uint32_t high = low | (uint32_t)((1 << (n_remain * 6)) - 1);
    if (low == 0) {
        if (n_remain == 2)      low = 1u << 11;
        else if (n_remain == 3) low = 1u << 16;
    }

    do {
        if (pos[1].type == GRE_CHAR_RNG_UPPER) {
            if (pos->value <= high && low <= pos[1].value) return is_positive;
            pos += 2;
        } else if (pos->type == GRE_CHAR_ANY) {
            return true;
        } else {
            if (low <= pos->value && pos->value <= high) return is_positive;
            pos += 1;
        }
    } while (pos->type == GRE_CHAR_ALT);

    return !is_positive;
}

/* Elements the char range at pos spans: the cost of one match_char, which walks the whole range
 * whatever the code point. */
inline size_t char_range_len(const GbnfElement* pos) {
    return (size_t)(match_char(pos, 0).second - pos);
}

inline bool match_token(const GbnfElement* pos, int32_t token) {
    if (pos->type == GRE_TOKEN)     return pos->value == (uint32_t)token;
    if (pos->type == GRE_TOKEN_NOT) return pos->value != (uint32_t)token;
    return false;
}

/* ============================================================ the compiled rules

 * Every rule in one arena, so an element's address gives its index by subtraction. The mask
 * engine looks up each stack element's continuation class (gbnf_mask.cpp) on every mask. The
 * arena is final once the program is published, since every stack points into it. */
struct GbnfRuleSet {
    std::vector<GbnfElement> elems;
    std::vector<uint32_t>    start;     /* rule id -> index of its first element */

    const GbnfElement* rule(size_t id) const { return elems.data() + start[id]; }
    uint32_t index(const GbnfElement* e) const { return (uint32_t)(e - elems.data()); }
    size_t   n_rules() const { return start.size(); }
};

/* Expand a stack into the stacks that each end at a terminal: the epsilon closure, where rule
 * references are resolved. `build` tracks `out` under construction (SetBuild). Throws GbnfLimit
 * past the bounds above. */
void advance_stack(const GbnfRuleSet& rules, const GbnfStack& stack, GbnfStacks& out,
                   SetBuild& build);

/* The rules a compiled program's stacks point into. */
const GbnfRuleSet& program_rules(const GbnfProgram& p);

/* ============================================================ the mask engine (gbnf_mask.cpp) */

/* Everything the mask engine derives from a vocabulary: each piece decoded to code points, a
 * trie over them, and the index of pieces that resume a split character. Built once per
 * vocabulary and shared by every grammar compiled against it. */
struct VocabIndex;

/* Throws std::bad_alloc; a vocabulary with no text in it builds an empty index. */
std::shared_ptr<const VocabIndex> vocab_index_build(const VocabView& v);
size_t vocab_index_bytes(const VocabIndex& x);

/* What building a mask plan cost and what it holds, for logging and the acceptance tool. */
struct PlanStats {
    double   ms          = 0.0;   /* wall time of the plan build */
    uint64_t work        = 0;     /* units charged against kMaxCompileWork */
    size_t   classes     = 0;     /* distinct stack-entry classes */
    size_t   keys        = 0;     /* stack suffixes with a precomputed mask */
    size_t   max_depth   = 0;     /* the deepest of them, in stack entries */
    size_t   dense       = 0;     /* of them, held as a full bitmask */
    size_t   rem_nodes   = 0;     /* nodes across every remainder trie */
    size_t   max_rem     = 0;     /* the largest remainder trie */
    size_t   bytes       = 0;     /* what the plan holds */
};

/* A grammar's compiled mask plan. Its runtime part (the stack-set automaton and mask scratch) is
 * a memo under the plan's own lock, so one plan serves every request and rank using the
 * grammar. */
struct MaskPlan;
struct MaskPlanDeleter { void operator()(MaskPlan* p) const; };
using MaskPlanPtr = std::unique_ptr<MaskPlan, MaskPlanDeleter>;

/* Build the plan for a grammar whose rules are final. Returns RAD_OK, or RAD_E_UNSUPPORTED with
 * the reason in *err (naming the rule from `rule_names`, by rule id, where possible) when the
 * masks cannot be precomputed within the bounds above. Throws GbnfLimit or std::bad_alloc, which
 * the caller turns into a status. */
int plan_build(const GbnfRuleSet& rules, uint32_t root, const std::shared_ptr<const VocabIndex>& vx,
               const std::vector<std::string>* rule_names, MaskPlanPtr* out, PlanStats* stats,
               std::string* err);

/* The mask for a stack set, bit set = admissible; `n_words` covers the vocabulary. Returns
 * RAD_OK, or RAD_E_STATE if no token is admissible; throws GbnfLimit past the per-operation
 * bounds. Thread-safe (takes the plan's lock). */
int plan_fill(const MaskPlan& plan, const GbnfStacks& stacks, PartialUtf8 partial, bool allow_eog,
              uint32_t* bitmask, int64_t n_words);

/* Step a stack set over code points through the plan's automaton. `out` receives the set left by
 * accepting cps[0..n) one at a time (what accept_chr over every stack builds), sorted and
 * de-duplicated, overwriting `out` so its capacity is reused. `stacks` need not be sorted and
 * must not point into `out`.
 *
 * `at` is the caller's automaton state, if known. On entry: a state an earlier step on this plan
 * returned (skips canonicalising the set), or id -1. On return: the state reached, or id -1 when
 * it holds an empty stack or a token terminal on top (a set the caller does not step whole). Valid
 * only while the issuing automaton lives; `gen` identifies it. Returns true when the reached
 * state equals the entry `at`; `out` is then left untouched (the steady state of a free-text
 * value).
 *
 * The automaton memoises each state's successor per code-point interval, so a seen state costs a
 * lookup per code point instead of rebuilding closures. Throws GbnfLimit past the per-operation
 * bounds. Thread-safe (takes the plan's lock). */
struct PlanState {
    uint64_t gen = 0;
    int32_t  id  = -1;
};
bool plan_step(const MaskPlan& plan, const std::vector<const GbnfStack*>& stacks, PlanState* at,
               const uint32_t* cps, size_t n, GbnfStacks* out, OpWork& work);

}  /* namespace gbnf */
}  /* namespace rad */
