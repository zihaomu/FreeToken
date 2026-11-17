/* gbnf.h -- the GBNF grammar state machine.
 *
 * Lifted from llama.cpp `src/llama-grammar.{h,cpp}` at upstream commit 06938ac12 (MIT). It lives
 * in core/sample/ because the sampler owns it, but the algorithm is upstream's element for
 * element. Changes, for a future re-lift:
 *   - `llama_vocab` became `VocabView` (vocab_view.h), so the sampler does not depend on the
 *     tokeniser's concrete type and a test can state its vocabulary in a few lines.
 *   - `GGML_ASSERT` / `GGML_ABORT` became status returns and `fatal()` (errors are values,
 *     docs/IMPLEMENTATION.md). Aborts inside the stack walk stay aborts: they are broken
 *     invariants, not runtime conditions.
 *   - The recursive-descent parser still throws internally; `parse()` catches and returns a
 *     status with the message in an out-parameter.
 *   - `llama_grammar_apply_impl`, which edited a candidate array in place, became `fill_mask`,
 *     which writes a token bitmask: 19 KiB at a 151K vocab, cheap enough to upload every step
 *     where a candidate array is not (spec §13).
 *   - Everything a client grammar can make the machine do is bounded (gbnf_internal.h, "what a
 *     grammar may cost"): repetition expansion, nesting depth, stack-set size, closure
 *     exploration, work per mask, per accepted token and per compile, and automaton and plan
 *     size. A grammar past a bound is refused at compile time or fails its request at run time;
 *     it never takes the process down.
 *   - Left recursion is found with nullability as a fixed point and an iterative SCC pass.
 *     Upstream's recursive check missed recursion through an indirectly nullable rule and
 *     recursed once per rule of a chain.
 *   - A closure's explored stacks are shared by every closure that builds the same set, stacks
 *     that match code points advance together, and a large set is de-duplicated through an
 *     index. Each gives upstream's set; without them, stacks that converge on one continuation
 *     re-expand it once each.
 *   - Accepting a token steps the code-point stacks through the mask plan's stack-set automaton
 *     (gbnf_mask.cpp, plan_step) instead of a closure per stack per code point, and the grammar
 *     remembers which automaton state its stacks are. The set is the one upstream's accept
 *     builds; the test oracle keeps upstream's accept and checks every accepted token against it.
 *   - The token mask is served from a plan precomputed at compile time (gbnf_mask.cpp): per stack
 *     suffix, the tokens it admits whatever lies below it, and a small trie of the tokens that
 *     depend on what does. Upstream's candidate walk survives only as the test oracle
 *     (tests/gbnf_oracle.cpp), and every mask is checked against it bit for bit.
 *
 * Cost. The mask is built on the scheduler thread with the rank threads parked at the barrier,
 * so its cost adds to the step time of a constrained request, up to four times a step under
 * speculation. A mask that walks the vocabulary costs more than the decode step it sits in.
 *   - Compiling builds the mask plan, walking the vocabulary once per place in the grammar a mask
 *     can start: milliseconds to a few hundred of them, bounded, on the thread that admits the
 *     request (gbnf_compile), never on the scheduler thread.
 *   - A mask is then a key lookup per live stack, an OR of precomputed token sets, and a walk of
 *     the few tokens whose admissibility depends on the stack below the key, at the same cost for
 *     a new state as for a seen one. A seen state is also a memcpy from the program's mask cache,
 *     shared by every request and rank that uses the grammar.
 *   - The vocabulary index is computed once per vocabulary and shared. Nothing is per request and
 *     nothing is on the device.
 *
 * So the state machine stays on host: the walk is precomputed (XGrammar's adaptive token-mask
 * cache, adapted to this machine's stacks) rather than moved to the device (spec §13).
 */
#pragma once
#include "../rad_internal.h"
#include "vocab_view.h"
#include "word_scan.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rad {

/* ================================================================== grammar elements */

enum GreType {
    GRE_END            = 0,   /* end of rule definition */
    GRE_ALT            = 1,   /* start of an alternate definition */
    GRE_RULE_REF       = 2,   /* reference to a rule */
    GRE_CHAR           = 3,   /* terminal: a code point */
    GRE_CHAR_NOT       = 4,   /* inverse char set: [^a], [^a-b] */
    GRE_CHAR_RNG_UPPER = 5,   /* modifies the preceding CHAR into an inclusive range */
    GRE_CHAR_ALT       = 6,   /* adds an alternate char to the preceding set */
    GRE_CHAR_ANY       = 7,   /* '.' */
    GRE_TOKEN          = 8,   /* terminal: a token id, `<[id]>` */
    GRE_TOKEN_NOT      = 9    /* inverse token, `!<[id]>` */
};

struct GbnfElement {
    GreType  type;
    uint32_t value;           /* code point, rule id, or token id */
};

/* A UTF-8 sequence split across token boundaries. Byte-level BPE emits these routinely, so the
 * grammar carries one from token to token. */
struct PartialUtf8 {
    uint32_t value    = 0;    /* bits so far, unshifted */
    int      n_remain = 0;    /* bytes still needed; -1 marks an invalid sequence */
};

using GbnfRule   = std::vector<GbnfElement>;
using GbnfRules  = std::vector<GbnfRule>;
using GbnfStack  = std::vector<const GbnfElement*>;
using GbnfStacks = std::vector<GbnfStack>;

/* The compiled grammar with its mask plan and mask cache, shared by every request that uses it;
 * nothing in it is per-request (gbnf.cpp). `GbnfState` is the mask cache's key: everything
 * fill_mask reads. */
struct GbnfProgram;
struct GbnfState;

/* What compiling a grammar cost and what the compiled program holds. */
struct GbnfProgramInfo {
    double   compile_ms    = 0.0;   /* parse, checks and mask plan */
    double   plan_ms       = 0.0;   /* of which, the mask plan */
    uint64_t plan_work     = 0;     /* units the plan charged against its bound */
    size_t   rules         = 0;
    size_t   elements      = 0;
    size_t   keys          = 0;     /* stack suffixes with a precomputed token set */
    size_t   dense_masks   = 0;     /* of those, held as a whole-vocabulary bitmask */
    size_t   max_key_depth = 0;     /* the deepest suffix, in stack entries */
    size_t   rem_nodes     = 0;     /* nodes across every remainder trie */
    size_t   max_rem_nodes = 0;     /* the largest remainder trie */
    size_t   bytes         = 0;     /* rules and plan; the mask cache is extra */
};

/* What the vocabulary index cost and holds. */
struct GbnfVocabInfo {
    double build_ms = 0.0;
    size_t bytes    = 0;
};

/* ================================================================== the parser */

class GbnfParser {
public:
    explicit GbnfParser(const VocabView* vocab = nullptr) : vocab_(vocab) {}

    /* Returns RAD_OK, or RAD_E_FORMAT with the parse error in *err. Never throws out. */
    int parse(const char* src, std::string* err);

    const std::map<std::string, uint32_t>& symbol_ids() const { return symbol_ids_; }
    const GbnfRules& rules() const { return rules_; }

private:
    const VocabView*                vocab_ = nullptr;
    std::map<std::string, uint32_t> symbol_ids_;
    GbnfRules                       rules_;

    /* Bytes built beyond the text itself, and the current parenthesis depth. Both are bounded
     * (gbnf_internal.h, "what a grammar may cost") because the text is client data parsed on
     * shared threads. */
    uint64_t budget_used_ = 0;
    uint32_t depth_       = 0;
    void     charge(uint64_t bytes);

    /* The optional links of bounded repetitions, keyed by the repeated operand's elements:
     * links[k - 1] is S'(k) ::= S S'(k-1) |, so every S{m,n} over the same S shares one chain and
     * builds only the links past the longest it has (handle_repetitions). A JSON schema's
     * `maxLength` bounds all repeat `char`, so a schema with many of them costs its longest bound
     * once instead of the sum of them. */
    std::map<std::vector<std::pair<uint32_t, uint32_t>>, std::vector<uint32_t>> rep_links_;

    uint32_t get_symbol_id(const char* src, size_t len);
    uint32_t generate_symbol_id(const std::string& base);
    void     add_rule(uint32_t id, const GbnfRule& rule);

    const char* parse_alternates(const char* src, const std::string& rule_name,
                                 uint32_t rule_id, bool is_nested);
    const char* parse_sequence(const char* src, const std::string& rule_name,
                               GbnfRule& rule, bool is_nested);
    const char* parse_rule(const char* src);
};

/* ================================================================== the state machine */

class GbnfGrammar {
public:
    /* Build from grammar text; `root` names the start symbol. Returns RAD_OK, or RAD_E_FORMAT /
     * RAD_E_NOTFOUND / RAD_E_UNSUPPORTED with the reason in *err. Left recursion is
     * RAD_E_UNSUPPORTED: the grammar is well-formed but this machine cannot run it.
     *
     * A lazy grammar constrains nothing until a trigger fires, so a tool-calling template stays
     * free-form until the model starts writing a call. A trigger is a token id or a literal word
     * in the generated bytes; parsing resumes where the word begins. A lazy grammar with no
     * trigger is RAD_E_INVAL, since it would never constrain. */
    static int create(const std::string& text, const std::string& root, const VocabView* vocab,
                      bool lazy,
                      const std::vector<std::string>& trigger_words,
                      const std::vector<int32_t>& trigger_tokens,
                      std::unique_ptr<GbnfGrammar>* out, std::string* err);

    /* Advance over one accepted token. RAD_E_STATE means the token left no stack alive: the
     * sampler emitted a forbidden token, a mask bug rather than a bad request. RAD_E_FULL or
     * RAD_E_NOMEM means the parse outgrew the bounds in gbnf_internal.h; the grammar is left dead
     * and the request cannot continue. */
    int accept_token(int32_t token);

    /* Advance over a raw string, for replaying a prompt prefix into the grammar. */
    int accept_str(const std::string& piece);

    /* Fill a token bitmask, bit set = admissible; `n_words` >= ceil(n_tokens/32). RAD_E_STATE if
     * no token is admissible (a dead grammar), RAD_E_FULL / RAD_E_NOMEM if the mask outgrew its
     * bounds. Each must fail the request rather than let a random token through. */
    int fill_mask(uint32_t* bitmask, int64_t n_words) const;

    /* Some stack has run to completion, so an end-of-generation token is admissible. */
    bool complete() const;

    /* Masks this grammar's program has computed rather than served from its memo. Shared with
     * every request on the same program; see grammar.h. */
    uint64_t masks_walked() const;

    /* A lazy grammar that has not seen its trigger yet constrains nothing. */
    bool awaiting_trigger() const { return awaiting_trigger_; }

    /* The current state, read-only: what a mask is a function of. The test oracle recomputes the
     * mask from these with upstream's candidate walk. */
    const GbnfStacks&  stacks() const { return stacks_; }
    PartialUtf8        partial() const { return partial_utf8_; }
    const VocabView*   vocab() const { return vocab_; }
    const GbnfProgram& program() const { return *prog_; }

    /* Push and pop the whole mutable state, so a speculative window can be walked forward and
     * rewound (see grammar.h). What create() fixes (rules, vocabulary, trigger words) is not in the
     * snapshot; what accepting a token can change is. That includes the trigger fields, since a
     * draft token may fire the trigger mid-window. Nestable; the sampler uses one level. */
    void push_state();
    int  rollback();

private:
    /* A token that may still hold the start of a trigger word: its id and the stream offset of its
     * first byte. It ends where the next one starts, or at the scan's current offset. */
    struct Recent {
        int32_t  token;
        uint64_t start;
    };

    /* The mask plan's automaton state that `stacks_` is, when known: the automaton (`gen`) and
     * the state in it, or id -1. Lets accepting skip canonicalising the set, and skip rewriting
     * it when a token leaves the state unchanged. Anything that sets `stacks_` other than an
     * automaton step resets it, or restores it together with the stacks it describes. */
    struct At {
        uint64_t gen = 0;
        int32_t  id  = -1;
    };

    struct Snapshot {
        GbnfStacks  stacks;
        PartialUtf8 partial;
        bool        awaiting_trigger = false;
        WordScanner::State      scan;
        std::vector<Recent>     recent;
        At                      at;
    };

    const VocabView* vocab_ = nullptr;

    /* The rules are shared; the position in them is not. `stacks_` points into prog_->rules, so
     * the program must outlive every grammar positioned in it and never change once published
     * (hence shared_ptr to const). That is also what lets the mask cache key on those pointers
     * across requests. */
    std::shared_ptr<const GbnfProgram> prog_;
    GbnfStacks       stacks_;
    PartialUtf8      partial_utf8_{};
    At               at_;

    std::vector<Snapshot>   saved_;

    /* The mask from the program's plan, bypassing the memo. */
    int mask_compute(uint32_t* bitmask, int64_t n_words) const;

    /* The trigger, while a lazy grammar waits. Words are scanned as bytes arrive; only the tokens
     * a word could still be starting in are kept (at most the longest word's length of them),
     * which is all a match replays. So waiting, and snapshotting it, cost the same on every token.
     * The scanner is fixed at create() and shared by every copy of the state. */
    bool                                awaiting_trigger_ = false;
    std::vector<int32_t>                trigger_tokens_;
    std::shared_ptr<const WordScanner>  scanner_;
    WordScanner::State                  scan_;
    std::vector<Recent>                 recent_;

    int accept_token_piece(int32_t token, const std::string& piece);
    int accept_token_piece_impl(int32_t token, const std::string& piece);

    /* Scratch for accepting, so a steady-state token allocates nothing: the piece's code points,
     * the stacks that step over them, and the previous set, whose storage the next one reuses.
     * None of it is state. */
    std::vector<uint32_t>          cps_;
    std::vector<const GbnfStack*>  current_;
    GbnfStacks                     spare_;
};

/* Compile a grammar on the calling thread into the program every GbnfGrammar::create for the
 * same (text, root, vocabulary) is served from while `out`, or any other holder, keeps it alive.
 * This is all the host work before a grammar's first mask: parse, checks and mask plan. The
 * server calls it on the admitting thread, so the scheduler thread never compiles. Returns what
 * GbnfGrammar::create returns. */
int gbnf_compile(const std::string& text, const std::string& root, const VocabView* vocab,
                 std::shared_ptr<const GbnfProgram>* out, std::string* err);

GbnfProgramInfo gbnf_program_info(const GbnfProgram& p);

/* Build the vocabulary index every grammar's plan is compiled against, if not built yet. Call
 * once at startup, or the first grammar to arrive pays for it. */
int gbnf_prepare_vocab(const VocabView* vocab, GbnfVocabInfo* info);

/* Words a bitmask needs for this vocabulary: 19 KiB at 151K tokens (spec §13). */
inline int64_t bitmask_words(int64_t n_vocab) { return (n_vocab + 31) / 32; }

}  /* namespace rad */
