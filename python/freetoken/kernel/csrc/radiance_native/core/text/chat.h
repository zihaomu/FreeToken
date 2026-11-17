/* chat.h -- rendering a message list to a prompt, and what the template can do.
 *
 * A thin layer over the lifted `common_chat_*` API (vendor/llama/common/chat.*, jinja/, see
 * vendor/UPSTREAM). It adds three things:
 *
 *   1. Errors are values. Upstream throws std::runtime_error for a bad template, a rejected
 *      message or a malformed OAI request; a throw must not cross the server's request loop, so
 *      everything here returns a negative RAD_E_* and logs once.
 *   2. The template comes from the .rad, not a llama_model. RadVocabHeader carries the chat
 *      template, the BOS/EOS spellings and the add_bos/add_eos flags, the four things
 *      common_chat_templates_init() would otherwise read off a model.
 *   3. The lifted headers stay out of the rest of core/. Only chat.cpp, chatfmt.cpp and
 *      chatparse.cpp include them (the last only to resolve schema types for bound tools), so a
 *      vendor refresh cannot ripple into the server.
 *
 * Messages and tools are OpenAI-shaped JSON, which is what the server has and what the lift reads.
 */
#pragma once
#include "rad_internal.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "util/json.h"

namespace rad {

class Vocab;

/* What the template supports, probed rather than assumed: jinja/caps.cpp renders the template
 * against synthetic inputs and watches what it reads (spec §12, "template capability probe").
 * Reported on /props; the message adapter uses it to choose string or typed-array content. */
struct ChatCaps {
    bool tools = true;
    bool tool_calls = true;
    bool system_role = true;
    bool parallel_tool_calls = true;
    bool preserve_reasoning = false;
    bool string_content = true;
    bool typed_content = false;
    bool object_arguments = false;
    /* What the first probe render with tools threw, empty when none did. The probe ignores
     * exceptions, so a template that calls something the engine lacks reports `tools` false as a
     * template without tool support does; this is the difference. */
    std::string probe_error;
};

struct ChatOptions {
    bool        add_generation_prompt = true;
    bool        enable_thinking = true;
    bool        parallel_tool_calls = false;
    std::string tool_choice = "auto";           /* auto | none | required */
    std::string reasoning_format = "auto";      /* none | auto | deepseek | deepseek-legacy */
    std::string continue_final;                 /* "" | auto | content | reasoning_content */
    std::string grammar;                        /* a user-supplied GBNF, or empty */
    std::string json_schema;                    /* response_format json_schema, or empty */
    bool        add_bos = false;
    bool        add_eos = false;
    std::map<std::string, std::string> template_kwargs;
    bool        quiet = false;                  /* a refusal is not logged: a startup probe */
};

/* Everything the scheduler and the sampler need for one request, produced in one pass. */
struct ChatPrompt {
    std::string prompt;
    std::string generation_prompt;    /* the assistant prefix already inside `prompt` */

    std::string grammar;              /* GBNF; empty when the request is unconstrained */
    bool        grammar_lazy = false;
    /* Triggers for a lazy grammar: the grammar starts applying only once one of these is seen.
     * Kept as (type, text) so the sampler can tell a literal word from a pattern. */
    struct GrammarTrigger { int type; std::string value; };
    std::vector<GrammarTrigger> grammar_triggers;

    std::vector<std::string> preserved_tokens;    /* must survive the sampler's bans */
    std::vector<std::string> additional_stops;

    bool        supports_thinking = false;
    std::string thinking_start_tag, thinking_end_tag;

    /* The auto-parser's verdict on this render: `format` is an opaque common_chat_format value,
     * `parser` the PEG grammar it derived. The engine parses replies with chatparse.h; these are
     * kept for the differential tests that compare the two. */
    int         format = 0;
    std::string parser;

    /* Why the template refused, in its own words (a jinja `raise_exception` message such as
     * "Unexpected reasoning effort high. Supported types are ..."), so a 400 names the field at
     * fault. Set only on a failure return. The text comes from the container's jinja, so it is
     * model-controlled, not attacker-controlled. */
    std::string error;
};

class ChatTemplate {
public:
    ChatTemplate();
    ~ChatTemplate();

    /* Compile a Jinja source. `bos`/`eos` are the token texts, which templates reference as
     * `bos_token` / `eos_token`. An empty source loads the ChatML fallback with a Warn, since the
     * prompt format is then a guess. */
    int load(const std::string& jinja_source, const std::string& bos, const std::string& eos,
             bool add_bos = false, bool add_eos = false);
    /* The same, with template, bos/eos text and add flags taken from the vocab's RadVocabHeader.
     * The engine uses this path. */
    int load_from_vocab(const Vocab& v);

    bool ready() const;
    const ChatCaps& caps() const;
    const std::string& source() const;

    /* Render. `messages` and `tools` are OpenAI-shaped JSON (`tools` may be null or empty).
     * Returns RAD_OK, RAD_E_INVAL when the request does not parse, or RAD_E_UNSUPPORTED when the
     * template refuses it (e.g. a system message without a system role, or tools without tool
     * support). Replayed history is read in other servers' shapes first: a turn's reasoning under
     * `reasoning` or `thinking`, and a call whose arguments are null, blank or absent. */
    int apply(const Json& messages, const Json& tools,
              const ChatOptions& opt, ChatPrompt* out) const;

    /* The ChatML fallback, named so the server can report which template it runs. */
    static const char* chatml_source();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  /* namespace rad */
