/* core/server/json_partial.h -- scanning JSON text that is still arriving.
 *
 *   json_complete       Is this a tool call? A call is reported as `finish_reason:
 *                       "tool_calls"` only if its arguments parse whole; a malformed call is
 *                       returned as ordinary content, so a tool is never called with truncated
 *                       arguments.
 *   json_stable_prefix  How much may be streamed now? A fragment is JSON-encoded into the SSE
 *                       payload, so it must end on a UTF-8 boundary; a split code point corrupts
 *                       this delta and the next.
 *   json_quote_append   Writes a string literal exactly as Json::dump does, for delta chunks.
 *
 * A scanner, not a parser: it allocates nothing and builds no DOM.
 */
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace rad {
namespace server {

/* Does `s` parse as exactly one complete JSON value, with only whitespace after it? */
bool json_complete(std::string_view s);

/* Largest n <= s.size() such that s[0,n) ends on a UTF-8 code point boundary. Invalid UTF-8 is
 * left alone: this backs out of a truncation, it does not validate. */
size_t utf8_trunc(std::string_view s);

/* Largest prefix of `s` that is safe to emit as one streaming fragment: on a UTF-8 boundary, not
 * inside a "\uXXXX" escape and not on a lone trailing backslash. The escape rule costs up to four
 * bytes of latency; it keeps a client that renders each delta separately from seeing a broken
 * escape. */
size_t json_stable_prefix(std::string_view s);

/* Append `s` to `out` as a JSON string literal, quotes included, byte for byte as Json::dump's
 * dump() writes one: `\b \t \n \f \r \" \\` as two characters, other control characters as
 * `\u00xx` in lowercase hex, everything else (DEL and non-ASCII included) as itself. Returns
 * false with `out` unchanged when `s` is not well-formed UTF-8 (including overlong forms,
 * surrogates and code points past U+10FFFF), which Json::dump refuses.
 *
 * Streamed chunks use this around a fixed frame: building and dumping a DOM per token cost more
 * than the rest of the token's host work. */
bool json_quote_append(std::string& out, std::string_view s);

}  /* namespace server */
}  /* namespace rad */
