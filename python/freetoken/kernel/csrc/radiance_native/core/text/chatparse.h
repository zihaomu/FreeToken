/* chatparse.h -- reading a model's reply as it streams: reasoning, content, tool calls.
 *
 * One pass, one parser. A reply is fed in pieces of any size as it is generated, and each byte is
 * examined a bounded number of times, so the whole reply costs O(length) however it is split.
 * Nothing is re-parsed, nothing is matched with a regular expression, and there is no repair
 * pass: the malformed shapes models actually write are states of this machine.
 *
 * Driven by a ChatFormat (chatfmt.h), it has three parts:
 *
 *   * Delimiter recognition: a deterministic Aho-Corasick automaton over the delimiters that can
 *     end the current region ("</think>" and the call opener in reasoning, the call opener in
 *     content, the value suffix in a value). In the root state the scan skips to the next byte
 *     that can begin a delimiter (memchr for the common one-byte case), so a long run of prose
 *     costs a memchr and a copy.
 *   * Tag structure: a transducer over those regions (call opener, name, arguments, close).
 *   * JSON-typed arguments: an incremental JSON scanner with an explicit stack.
 *
 * The output is what the format's grammar says the reply means, including where the grammar is
 * deliberately lenient: an undeclared argument is still an argument, a non-string argument that
 * is not valid JSON is text, an unclosed reasoning block ends where a call begins, and prose
 * before, between and after calls is content. Where the reply leaves the grammar (a call to a
 * function that is not a tool, text inside a call, a missing close) it recovers locally and
 * clears strict().
 *
 * Unwrapped calls. In a wrapped format (Qwen's `<tool_call>\n`) the grammar, and so constrained
 * decoding, starts only at the wrapper, so a call written as a bare `<function=NAME>`, or with the
 * wrapper run into it, is unconstrained and the most likely to be lost. In content, name_prefix
 * alone and call_open without its trailing whitespace also open a call, with every tag's
 * surrounding whitespace optional inside it. Because such an opener is also ordinary text, it
 * becomes a call only once the name tag is closed and an argument or the call's close follows;
 * until then it is held (a bounded number of bytes), and if it is not a call it is content.
 *
 * Streaming contract: what has been emitted is final. A delta is never retracted, so the
 * concatenated deltas are the finished content, reasoning and arguments. Bytes whose meaning is
 * not yet known are held, within these bounds:
 *
 *   * a possible delimiter prefix: at most the longest delimiter less one byte;
 *   * an unfinished UTF-8 sequence: at most three bytes, since a delta goes into a JSON string;
 *   * whitespace that depends on what follows: content's trailing whitespace (trimmed if nothing
 *     follows) and the whitespace opening a reasoning block (dropped if the block holds nothing
 *     else);
 *   * a tag's name while it is read, since it is a name only once closed;
 *   * a JSON-typed argument value until its closing delimiter. `30` and `30 seconds` differ from
 *     the first byte (a number, or the string "30 seconds"), so the value is emitted whole once
 *     its type is known. A value that stops being JSON part-way is streamed as text from there,
 *     except a container (`[`, `{`), which is held to its end: only then is it known whether it
 *     is JSON once its Python quoting is normalised, or becomes the JSON string of its text.
 *
 * Events are emitted in reply order; a call's arguments stream as they are parsed. */
#pragma once
#include "rad_internal.h"

#include "util/json.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rad {

struct ChatFormat;

/* One increment of a parsed reply. Consecutive increments of the same kind are merged. */
struct ReplyEvent {
    enum Kind : uint8_t {
        Reasoning,   /* text: a reasoning delta */
        Content,     /* text: a content delta */
        CallBegin,   /* text: the function name; `call` is the new call's index */
        CallArgs,    /* text: an arguments delta for call `call` */
        CallEnd,     /* call `call` closed */
    };
    Kind        kind = Content;
    uint32_t    call = 0;
    std::string text;
};

struct ReplyCall {
    std::string name;
    std::string arguments;    /* JSON text, as the OpenAI API carries it */
    bool        closed = false;
    /* Where the call's text begins in the reply (`at`), and the content before it: `content_at`
     * bytes of the message's content plus the untrimmed whitespace held after them. Used to read
     * the call back as text (ReplyStream::content_with_open_call). */
    size_t      at = 0;
    size_t      content_at = 0;
    std::string content_held;
};

/* A reply, taken apart. After finish() it is the whole reply; before, what has been emitted. */
struct ReplyMessage {
    std::string            content;
    std::string            reasoning;
    std::vector<ReplyCall> calls;
};

/* What held bytes are waiting on, by kind. See the header comment for the bound on each. */
struct ReplyHeld {
    size_t delimiter = 0;     /* a possible delimiter prefix, and an unfinished UTF-8 sequence */
    size_t space     = 0;     /* whitespace that may yet be trimmed */
    size_t structure = 0;     /* a tag name being read, or a JSON value awaiting its type */
};

/* The per-request inputs. */
struct ReplyOptions {
    /* The assistant prefix the template already rendered. It decides whether the reply starts
     * inside a reasoning block (ChatFormat::reasoning_in_prompt). */
    std::string generation_prompt;
    /* reasoning_format "none" keeps a reasoning block inline in the content. */
    bool        extract_reasoning = true;
    /* The request's tools, OpenAI-shaped; null or empty parses no calls, and call syntax in the
     * reply is then content. Read once, when the parser is created. */
    const Json* tools = nullptr;
    bool        parallel_tool_calls = true;
    /* A response_format was requested: the reply is one JSON document, optionally fenced as
     * ```json ... ```, and the document is the content. */
    bool        response_format = false;
    /* The model's turn terminators. The decoder renders one only when the format needs it; one
     * left at the end of content is not part of the answer. */
    std::vector<std::string> terminators;
};

class ReplyStream;

/* The reasoning block as one request's reply sees it: the markers with their optional
 * surrounding whitespace trimmed, and whether the reply starts inside the block (the generation
 * prompt opened it, or the format has no opener). An empty `close` means no reasoning block. */
struct ReasoningMarkers {
    std::string open, close;
    bool        starts_open = false;
};

/* A format compiled once: its automata and its delimiters. Shared by every request. */
class ReplyFormat {
public:
    ~ReplyFormat();
    /* RAD_OK, or RAD_E_UNSUPPORTED with *why when chat_format_check refuses the format. */
    static int compile(const ChatFormat& f, std::shared_ptr<const ReplyFormat>* out,
                       std::string* why);
    const ChatFormat& format() const;

    struct Impl;
    const Impl& impl() const { return *impl_; }
private:
    ReplyFormat();
    std::unique_ptr<Impl> impl_;
};

/* One request's parser: the format bound to the tools the request sent and the prompt it
 * rendered. Immutable once made; each choice opens its own stream from it. */
class ReplyParser {
public:
    ~ReplyParser();
    /* RAD_OK; RAD_E_INVAL with *why when a tool cannot be read unambiguously in this format (a
     * function or parameter name containing the delimiter that ends it); RAD_E_UNSUPPORTED when
     * the request sends tools and the format has no tool calls. */
    static int create(std::shared_ptr<const ReplyFormat> fmt, const ReplyOptions& opt,
                      std::shared_ptr<const ReplyParser>* out, std::string* why);
    std::unique_ptr<ReplyStream> open() const;
    const ChatFormat& format() const;
    /* Upper bound on ReplyHeld::delimiter for this request: a delimiter or terminator prefix in
     * each place one can be pending at once, a value's undecided head, and an unfinished UTF-8
     * sequence per output. Depends on the format, not the reply. */
    size_t hold_bound() const;
    /* Read from the generation prompt the same way a stream reads its first bytes, so a reply
     * this says starts in the block is read as reasoning from its first byte. */
    ReasoningMarkers reasoning() const;

    struct Impl;
    const Impl& impl() const { return *impl_; }
private:
    ReplyParser();
    std::unique_ptr<Impl> impl_;
};

class ReplyStream {
public:
    ~ReplyStream();

    /* Append bytes. The increments they complete are appended to *out (which may be null). */
    void feed(std::string_view bytes, std::vector<ReplyEvent>* out);
    /* The reply is complete. Whatever was held is resolved and emitted. Idempotent. */
    void finish(std::vector<ReplyEvent>* out);

    const ReplyMessage& message() const;
    /* After finish(): the content with the last call read as text. If the reply ended inside an
     * unclosed call, that is the content before the call followed by the call as `reply` (every
     * byte fed) spells it, end-trimmed like content, since an unclosed call cannot be reported as
     * a call. Otherwise (last call closed, or no calls) the message's content. */
    std::string content_with_open_call(std::string_view reply) const;
    /* False once the reply left the format's grammar and was read by a recovery rule. */
    bool strict() const;
    ReplyHeld held() const;
    /* Bytes queued so far to be read a second time. Each step of the parser consumes a byte or
     * queues bytes again, so it takes at most as many steps as the bytes fed plus these. */
    size_t reread() const;

    struct Impl;
private:
    friend class ReplyParser;
    ReplyStream();
    std::unique_ptr<Impl> impl_;
};

/* Parse a complete reply in one call. */
ReplyMessage reply_parse(const ReplyParser& p, std::string_view text, bool* strict = nullptr);

}  /* namespace rad */
