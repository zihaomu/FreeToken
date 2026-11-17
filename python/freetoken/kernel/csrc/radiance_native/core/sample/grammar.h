/* grammar.h -- structured output: the grammar backend interface, and the bitmask that crosses to
 * the device.
 *
 * Structured output is GBNF, with JSON Schema compiled to a grammar. The grammar state machine
 * stays on host: it is a pushdown automaton that branches per candidate, which suits a CPU. What
 * crosses is a token bitmask per sequence, 19 KiB at a 151K vocab, cheap enough to upload every
 * step (spec §13). The mask is applied by the ordinary `sample_mask` op (docs/OPS.md).
 *
 * The backend interface is kept small so an alternative (e.g. a device-side grammar) can
 * implement it without changes above it. GBNF is the only backend, so there is no registry.
 */
#pragma once
#include "../rad_internal.h"
#include "vocab_view.h"

#include <memory>
#include <string>
#include <vector>

namespace rad {

enum class GrammarKind {
    Gbnf       = 0,   /* the text is GBNF */
    JsonSchema = 1    /* the text is a JSON Schema, compiled to GBNF at create time */
};

struct GrammarSpec {
    GrammarKind kind = GrammarKind::Gbnf;
    std::string text;
    std::string root = "root";

    /* A lazy grammar constrains nothing until a trigger fires, so a tool-calling template stays
     * free-form until the model starts a call. A trigger is a token id or a literal word in the
     * emitted bytes; sampler.cpp maps a template's triggers to these. */
    bool                     lazy = false;
    std::vector<std::string> trigger_words;
    std::vector<int32_t>     trigger_tokens;

    /* Text replayed into the grammar before it constrains anything. A chat template's grammar
     * covers the whole assistant turn, so its root begins with the generation prompt (e.g.
     * `"<|im_start|>assistant\n"`), which is already in the prompt. Without the replay the mask
     * would force the model to emit it again. Empty for a caller-supplied grammar, which
     * describes only the reply. */
    std::string              prefix;
};

class GrammarBackend {
public:
    virtual ~GrammarBackend() = default;

    /* Advance over the token the sampler just emitted. RAD_E_STATE means the token was not
     * admissible: the mask and the pick disagreed, a bug here rather than a bad request.
     * RAD_E_FULL or RAD_E_NOMEM means the parse outgrew the backend's bounds; the backend is then
     * dead and its next fill_mask fails. */
    virtual int accept_token(int32_t token) = 0;

    /* Fill `bitmask` ([n_words] words, bit set = admissible). RAD_E_STATE if nothing is
     * admissible, RAD_E_FULL or RAD_E_NOMEM if the state is past the backend's bounds; each fails
     * the request rather than emit a forbidden token. */
    virtual int fill_mask(uint32_t* bitmask, int64_t n_words) const = 0;

    /* Some parse has run to completion, so the sequence may stop here. */
    virtual bool complete() const = 0;

    /* Instrument: masks computed rather than served from a memo, monotonic over the memo's
     * lifetime. Diff it around a fill_mask to classify that call. A counter, because timing cannot
     * classify: a computed mask on a small grammar can be faster than a memo hit on a large one.
     * A backend with no memo counts every mask. */
    virtual uint64_t masks_walked() const { return 0; }

    /* False while a lazy grammar waits for its trigger. Lets the driver skip the mask op for an
     * unconstrained sequence instead of applying an all-ones mask. */
    virtual bool constrains() const = 0;

    /* ---------------------------------------------------------- speculation

     * A constrained sequence speculates by walking the window and rewinding. The mask for draft
     * position i comes from the state after accepting drafts 0..i-1, so the sampler pushes the
     * state, advances over each proposal to mask the next position, then pops back.
     *
     * A backend that cannot do this returns false and its requests decode one token at a time;
     * masking a draft position against the wrong state would admit the wrong token set. */
    virtual bool can_speculate() const { return false; }

    /* Push the current state. Nestable; `rollback` pops the most recent. */
    virtual int push_state() { return RAD_E_UNSUPPORTED; }
    virtual int rollback()   { return RAD_E_UNSUPPORTED; }
};

/* Words a mask needs for a vocabulary of this size. */
int64_t grammar_mask_words(int64_t n_vocab);

/* Compile a JSON Schema to GBNF text, without building a state machine, so the grammar can be
 * inspected (e.g. `--debug-graph`). */
int grammar_from_json_schema(const std::string& schema_json, const std::string& root,
                             std::string* out_gbnf, std::string* err);

/* Build the grammar backend (always GBNF). Returns RAD_OK, or a negative status with a
 * human-readable reason in *err. */
int grammar_create(const GrammarSpec& spec, const VocabView& vocab,
                   std::unique_ptr<GrammarBackend>* out, std::string* err);

}  /* namespace rad */
