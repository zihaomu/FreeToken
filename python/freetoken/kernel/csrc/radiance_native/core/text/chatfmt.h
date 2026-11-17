/* chatfmt.h -- how a model spells a turn (reasoning, tool calls, arguments), as data.
 *
 * Everything the engine knows about a model's reply syntax is in one struct, ChatFormat, and
 * everything that depends on it is derived from it: the streaming reply parser (chatparse.h), the
 * lazy tool-call grammar and its trigger words, and the tokens the decoder must render as text so
 * the parser can see the structure.
 *
 * Two producers, in order of precedence:
 *
 *   1. The architecture plugin, through the optional `rad_arch_chat_format` export
 *      (abi/rad_builder.h).
 *   2. Analysis of the chat template. llama.cpp's differential analyser renders the template with
 *      and without a reasoning block, a call and its arguments, and diffs the renders to find the
 *      delimiters (vendor/llama/common/chat-diff-analyzer.cpp). Its derived markers are read here;
 *      its PEG parser is not used.
 *
 * If neither yields a format the parser can run, the model still chats (the reply is content) but
 * a request with tools is refused, naming why. There is no fallback parser.
 *
 * Tagged family:
 *
 *   Qwen3.6 / Qwen3.8    <tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n
 *                        </function>\n</tool_call>
 *   MiniCPM5             <function name="NAME"><param name="ARG">VALUE</param></function>
 *   GLM-5                <tool_call>NAME<arg_key>ARG</arg_key><arg_value>VALUE</arg_value>
 *                        </tool_call>
 *
 * A call's name sits between a prefix and a suffix and the call ends with its own close; each
 * argument is a name between a prefix and a suffix, then a value ending at its own suffix; calls
 * may be wrapped one by one. A name may also have no suffix of its own (GLM): it then ends where
 * the first argument or the call's close begins, and that tag is read as what it is. Arguments
 * are typed by the tool's JSON schema: a string-typed one is raw text (optionally CDATA-wrapped),
 * any other type is JSON, and text that does not parse as JSON is taken raw, since models often
 * write Python literals into a tag.
 *
 * Object family (ChatArgs::Object): a call is its name and one object holding the arguments.
 *
 *   Gemma 4              <|tool_call>call:NAME{key:<|"|>text<|"|>,n:3}<tool_call|>
 *
 * The object is JSON or a dialect of it (strings verbatim between two declared delimiters, bare
 * keys), and the client receives it as JSON. Values carry their own types, so no schema is
 * consulted; a bare word that is not a JSON literal or number is read as a string.
 *
 * The analyser also recognises formats outside these families (a JSON object per call with the
 * name inside it, call ids). For those, chat_format_from_template keeps the reasoning split and
 * names why tool calls are refused; supporting one means adding the shape to the parser. */
#pragma once
#include "rad_internal.h"

#include "util/json.h"

#include <cstdint>
#include <string>
#include <vector>

struct RadChatFormat;

namespace rad {

/* How a value is written between its delimiters. The numbering is the ABI's (RAD_CHAT_VALUE_*). */
enum class ChatValue : uint32_t {
    Raw   = 0,   /* the text, verbatim */
    Cdata = 1,   /* the text, unless wrapped in <![CDATA[ ]]>, which is punctuation and removed */
    Json  = 2,   /* a JSON value; text that does not parse as one is taken raw */
};

/* Whether the reply starts inside a reasoning block. The numbering is the ABI's. */
enum class ChatThink : uint32_t {
    /* Search the rendered generation prompt for `reasoning_open`; what follows its first
     * occurrence is the start of the reply's text. Handles both a template that opens the block
     * for the model (Qwen ends the prompt with "<think>\n") and one that closes it again when
     * thinking is off ("<think>\n\n</think>\n\n"). */
    FromPrompt = 0,
    Always     = 1,  /* the reply is inside a reasoning block from its first byte */
    Never      = 2,  /* the model opens its own block, if any */
};

/* How a call's arguments are written. The numbering is the ABI's (RAD_CHAT_ARGS_*). */
enum class ChatArgs : uint32_t {
    Tagged = 0,  /* each argument in its own tags: arg_prefix NAME .. VALUE arg_value_suffix */
    Object = 1,  /* one object after the name: `{` key `:` value, ... `}` */
};

struct ChatFormat {
    std::string name;                 /* for logs: "plugin:<arch>" or "template" */

    /* ------------------------------------------------------------------ reasoning
     * A block runs from `reasoning_open` to `reasoning_close`. Whitespace at either end of either
     * tag is optional when matching, so "<think>\n" also matches "<think>". An empty close means
     * the format has no reasoning block. An empty open with a close means the reply starts in one
     * and the close ends it. */
    std::string reasoning_open;
    std::string reasoning_close;
    /* Literals that end an unclosed block when a call starts there, so a model that skips the
     * close and goes straight to a call does not have the call read as reasoning. When non-empty,
     * must include the call opener (`call_open`). */
    std::vector<std::string> reasoning_breaks;
    ChatThink   reasoning_in_prompt = ChatThink::FromPrompt;
    /* A block may also open later in the reply (after the answer, between calls) and is read as
     * reasoning there; a close with no block open is dropped. Gemma 4 sometimes writes an empty
     * block after its answer, or a lone close. Needs `reasoning_open`. */
    bool        reasoning_anywhere = false;
    /* A header the model writes first when it answers without reasoning; dropped when the reply
     * starts with it. Empty: none. */
    std::string content_open;

    /* ------------------------------------------------------------------ tool calls
     * `name_prefix` is what opens a call (after `call_open`, when there is one). Empty means the
     * format has no tool calls. */
    std::string section_open, section_close;  /* around the complete list of calls */
    /* Between two calls, beyond whitespace (a JSON-array format would use ", "). Tagged calls are
     * self-delimiting. Not read yet, must be empty. */
    std::string call_separator;
    /* Around each call: "<tool_call>\n", "</tool_call>". When set, a call written without it
     * (name_prefix alone) is still read as a call (chatparse.h). */
    std::string call_open, call_close;
    /* "<function=", ">\n". name_suffix may be empty: in the object family the name then ends at
     * the object's `{`; in the tagged family, where name_prefix must open the call (no
     * call_open), at the first arg_prefix or call_end, which stay part of the call's body. */
    std::string name_prefix, name_suffix;
    std::string call_end;                     /* "</function>\n"; may be empty for an object */

    /* How the arguments are written. For Object the arg_* fields below are empty and these three
     * describe the dialect: a string delimiter besides JSON's `"` with verbatim text between
     * ("<|\"|>" twice for Gemma 4; empty for JSON strings only), and whether keys may be bare. */
    ChatArgs    args_form = ChatArgs::Tagged;
    std::string object_string_open, object_string_close;
    bool        object_bare_keys = false;

    std::string arg_prefix;                   /* "<parameter=" */
    /* What ends a name and opens its value. Both may be set (GLM: "</arg_key>", "<arg_value>"):
     * they are then read as one delimiter, written back to back. */
    std::string arg_name_suffix;              /* ">\n"   (MiniCPM: empty, the value prefix ends it) */
    std::string arg_value_prefix;             /* ""      (MiniCPM: "\">") */
    /* A second value prefix, for a format whose argument tag says the value's type (DSML:
     * ` string="false">` before JSON, arg_value_prefix ` string="true">` before a string). The
     * prefix the reply writes types its value, whatever the schema says; the grammar writes the
     * one the schema calls for. Empty: the schema alone types a value. */
    std::string arg_json_value_prefix;
    std::string arg_value_suffix;             /* "</parameter>\n" */

    ChatValue   string_value = ChatValue::Cdata;  /* a string-typed argument: Raw or Cdata */
    ChatValue   other_value  = ChatValue::Json;   /* any other type: Json */
    /* One newline at each end of a value belongs to the tags, since the template writes the
     * value on its own lines. A tag that ends with its newline (Qwen's `>\n`) has already
     * consumed the first one. */
    bool        value_newlines = true;

    /* Literal words that start constrained decoding of a call. Empty means no lazy grammar: the
     * reply is unconstrained. */
    std::vector<std::string> triggers;

    bool has_reasoning() const { return !reasoning_close.empty(); }
    bool typed_tags() const { return !arg_json_value_prefix.empty(); }
    bool has_tools() const { return !name_prefix.empty(); }
    bool object_args() const { return args_form == ChatArgs::Object; }
};

/* Can the streaming parser run this format? RAD_OK, or RAD_E_UNSUPPORTED with *why naming the
 * field. A format without tool calls is valid (reasoning split only); one with tool calls must
 * make every delimiter unambiguous. */
int chat_format_check(const ChatFormat& f, std::string* why);

/* Copy the plugin's declaration out of the ABI struct. Fields past the plugin's `struct_size`
 * read as zero. RAD_E_INVAL with *why when the struct is unusable. */
int chat_format_from_abi(const RadChatFormat* in, ChatFormat* out, std::string* why);

/* Derive the format from a chat template by the vendored differential analysis.
 *
 * RAD_OK with *out filled. If the analysis finds tool calls this family cannot express, the
 * reasoning split is still returned with the tool fields empty and *tools_refused saying why;
 * requests with tools are then refused with that sentence. RAD_E_UNSUPPORTED with *why when the
 * template routes to one of llama.cpp's hand-written handlers, which the analysis does not
 * describe; RAD_E_FORMAT when the template does not compile. An empty source is the ChatML
 * fallback, which has neither reasoning nor calls. */
int chat_format_from_template(const std::string& jinja, const std::string& bos,
                              const std::string& eos, ChatFormat* out,
                              std::string* tools_refused, std::string* why);

/* Human-readable one-liner for the startup log. */
std::string chat_format_describe(const ChatFormat& f);

/* ================================================================== grammar
 *
 * The GBNF that constrains a reply to the format. For a tool request it is lazy: the reply is free
 * until one of `triggers` appears, then the grammar admits the call syntax with each argument
 * constrained by its schema. For a response format, or a request that requires a call, it covers
 * the whole reply. Rule names and structure follow the vendored generator, so both produce the
 * same grammar for a template-analysed format. */
struct ChatGrammarRequest {
    const Json* tools = nullptr;        /* the request's tools, or null */
    const Json* json_schema = nullptr;  /* a response format's schema, or null */
    std::string generation_prompt;
    bool        parallel_tool_calls = true;
    bool        extract_reasoning = true;  /* reasoning_format != "none" */
    /* tool_choice "required" or a named function: the grammar covers the whole reply (reasoning,
     * then calls, nothing after), and `only_tool` when set restricts calls to that function. */
    bool        require_call = false;
    std::string only_tool;
};

struct ChatGrammar {
    std::string text;                     /* empty when the request is unconstrained */
    bool        lazy = false;
    std::vector<std::string> trigger_words;
};

/* A tool the grammar and the reply parser both read, {"type":"function","function":{...}}; both
 * skip any other. */
bool chat_function_tool(const Json& tool);

/* RAD_OK, or RAD_E_INVAL with *why when a tool or schema cannot be expressed. */
int chat_format_grammar(const ChatFormat& f, const ChatGrammarRequest& req, ChatGrammar* out,
                        std::string* why);

/* The markers the decoder must render as text so the parser sees them; the decoder otherwise
 * suppresses control tokens, and a format built from them would lose its structure. Every
 * non-empty delimiter, whitespace-trimmed and deduplicated, in the vendored analyser's order. A
 * multi-token marker is fine: the decoder only compares markers with control-token spellings. */
std::vector<std::string> chat_format_markers(const ChatFormat& f);

}  /* namespace rad */
