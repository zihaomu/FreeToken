#include "text/chatfmt.h"
#include "text/chat.h"

#include "rad_builder.h"

/* The analyser and grammar builder, used at load and admission only, never while reading a
 * reply. */
#include "llama/common/chat.h"
#include "llama/common/chat-auto-parser.h"
#include "llama/common/chat-auto-parser-helpers.h"
#include "llama/common/json-schema-to-grammar.h"
#include "shim/log.h"

#include "util/json.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <functional>
#include <map>
#include <set>

namespace rad {

using ojson = Json;

namespace {

bool is_ws(unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }

std::string trim_ws(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && is_ws((unsigned char)s[a])) ++a;
    while (b > a && is_ws((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

/* `inner` occurs inside `outer` somewhere other than at its start. */
bool occurs_inside(const std::string& inner, const std::string& outer) {
    return !inner.empty() && outer.size() > inner.size() && outer.find(inner, 1) != std::string::npos;
}

/* Two delimiters an automaton scanning for both cannot report in start order: one begins or
 * occurs inside the other. */
bool confusable(const std::string& a, const std::string& b) {
    return a.starts_with(b) || b.starts_with(a) || occurs_inside(a, b) || occurs_inside(b, a);
}

std::string show(const std::string& s) { return Json(s).dump(); }

int refuse(std::string* why, const std::string& s) {
    if (why) *why = s;
    return RAD_E_UNSUPPORTED;
}

/* A byte that begins a bare word or a JSON value, or is the object's own punctuation. */
bool object_byte(unsigned char c) {
    return c == 0 || std::isalnum(c) || is_ws(c) || std::strchr("-+._'\"{}[],:", (int)c) != nullptr;
}

/* Checks an argument object against how chatparse.cpp reads it. Where a key or value may begin,
 * the parser tells three literals apart by their first bytes: the dialect's string opener, and
 * the call's close and end (a reply that leaves its object unclosed writes them there). None may
 * begin another. A bare word ends at the object's punctuation or the call's close; one automaton
 * reports both in start order only if the close contains no punctuation. */
int object_check(const ChatFormat& f, std::string* why) {
    const std::string& so = f.object_string_open;
    const std::string& sc = f.object_string_close;
    if (so.empty() != sc.empty())
        return refuse(why, "object_string_open and object_string_close are set together or not "
                           "at all");
    if (!so.empty()) {
        if (trim_ws(so) != so || trim_ws(sc) != sc)
            return refuse(why, "an object string delimiter may not begin or end with "
                               "whitespace");
        if (object_byte((unsigned char)so[0]))
            return refuse(why, "object_string_open " + show(so) + " begins with a byte that also "
                               "begins a bare word, a JSON value or the object's punctuation");
    }
    std::vector<std::string> lits;
    if (!so.empty()) lits.push_back(so);
    const std::string cc = trim_ws(f.call_close), ce = trim_ws(f.call_end);
    for (const std::string* s : { &cc, &ce }) {
        if (s->empty()) continue;
        for (char c : *s)
            if (std::strchr(",:{}[]\"", c))
                return refuse(why, "the call's close " + show(*s) + " holds " +
                                   show(std::string(1, c)) + ", which is the argument object's "
                                   "punctuation");
        if (std::find(lits.begin(), lits.end(), *s) == lits.end()) lits.push_back(*s);
        else if (!so.empty() && *s == so)
            return refuse(why, show(*s) + " both opens an object string and closes the call");
    }
    for (size_t i = 0; i < lits.size(); ++i)
        for (size_t j = i + 1; j < lits.size(); ++j)
            if (confusable(lits[i], lits[j]))
                return refuse(why, show(lits[i]) + " and " + show(lits[j]) + " cannot be told "
                                   "apart inside an argument object");
    return RAD_OK;
}

/* Content is scanned for a later reasoning opener, a stray close and the call openers in one
 * automaton, which reports them in start order only if none begins or occurs inside another. */
int reasoning_anywhere_check(const ChatFormat& f, std::string* why) {
    if (!f.reasoning_anywhere) return RAD_OK;
    std::vector<std::string> d = { trim_ws(f.reasoning_open), trim_ws(f.reasoning_close) };
    if (f.has_tools()) {
        if (!f.call_open.empty()) d.push_back(trim_ws(f.call_open));
        d.push_back(f.name_prefix);
    }
    for (size_t i = 0; i < d.size(); ++i)
        for (size_t j = i + 1; j < d.size(); ++j)
            if (confusable(d[i], d[j]))
                return refuse(why, show(d[i]) + " and " + show(d[j]) + " cannot be told apart in "
                                   "content, which a reasoning block may open anywhere in");
    return RAD_OK;
}

/* Does the template write an argument's value on lines of its own? Then one newline at each end
 * is the tags' (ChatFormat::value_newlines). A template that writes it flush against its tags
 * (`<arg_value>VALUE</arg_value>`) leaves those newlines to the value; reading them as the tags'
 * would drop a written file's last line break. A render that fails counts as own lines. */
bool value_on_own_lines(const common_chat_template& tmpl, const ChatFormat& c) {
    static const char* const kValue = "RADIANCE_VALUE_PROBE";
    ojson call = { { "id", "call_0" }, { "type", "function" } };
    call["function"] = { { "name", "probe" }, { "arguments", { { "arg", kValue } } } };
    ojson reply = { { "role", "assistant" }, { "content", "" } };
    reply["tool_calls"] = ojson::array({ call });
    ojson params = { { "type", "object" } };
    params["properties"]["arg"]["type"] = "string";
    ojson tool = { { "type", "function" } };
    tool["function"] = { { "name", "probe" }, { "description", "probe" } };
    tool["function"]["parameters"] = params;
    template_params tp;
    tp.messages = ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } }, reply });
    tp.tools = ojson::array({ tool });
    const std::string r = autoparser::apply_template(tmpl, tp);
    const std::string flush = c.arg_name_suffix + c.arg_value_prefix + kValue + c.arg_value_suffix;
    return r.find(flush) == std::string::npos;
}

}  /* namespace */

/* ================================================================== validity */

int chat_format_check(const ChatFormat& f, std::string* why) {
    auto no = [why](const std::string& s) { return refuse(why, s); };

    if (!f.reasoning_open.empty() && f.reasoning_close.empty())
        return no("reasoning_open " + show(f.reasoning_open) + " has no reasoning_close");
    if (f.has_reasoning() && trim_ws(f.reasoning_close).empty())
        return no("reasoning_close is only whitespace");
    if ((uint32_t)f.reasoning_in_prompt > (uint32_t)ChatThink::Never)
        return no("reasoning_in_prompt is not a RAD_CHAT_THINK_* value");
    if ((uint32_t)f.args_form > (uint32_t)ChatArgs::Object)
        return no("args_form is not a RAD_CHAT_ARGS_* value");
    /* A block that opens later in the reply is found by its opener, in content. */
    if (f.reasoning_anywhere && (!f.has_reasoning() || trim_ws(f.reasoning_open).empty()))
        return no("reasoning_anywhere needs a reasoning block with an opener");

    if (!f.has_tools()) {
        for (const std::string* s : { &f.section_open, &f.section_close, &f.call_separator,
                                      &f.call_open, &f.call_close, &f.name_suffix, &f.call_end,
                                      &f.arg_prefix, &f.arg_name_suffix, &f.arg_value_prefix,
                                      &f.arg_json_value_prefix, &f.arg_value_suffix,
                                      &f.object_string_open, &f.object_string_close })
            if (!s->empty()) return no("tool-call delimiters are set but name_prefix is empty");
        if (f.object_args() || f.object_bare_keys)
            return no("an argument object is described but name_prefix is empty");
        if (!f.reasoning_breaks.empty()) return no("reasoning_breaks without tool calls");
        if (!f.triggers.empty()) return no("grammar triggers without tool calls");
        return reasoning_anywhere_check(f, why);
    }

    if (f.section_open.empty() != f.section_close.empty())
        return no("section_open and section_close must be set together");
    if (!f.section_open.empty()) {
        const std::string so = trim_ws(f.section_open), sc = trim_ws(f.section_close);
        if (so.empty() || sc.empty()) return no("section_open/section_close may not be whitespace");
        if (so != f.section_open || sc != f.section_close)
            return no("section_open/section_close must not have surrounding whitespace");
        if (confusable(so, sc))
            return no("section_open " + show(so) + " and section_close " + show(sc) +
                      " cannot be told apart");
        for (const std::string* d : { &f.reasoning_open, &f.reasoning_close, &f.call_open,
                                      &f.call_close, &f.name_prefix, &f.name_suffix,
                                      &f.call_end, &f.arg_prefix, &f.arg_name_suffix,
                                      &f.arg_value_prefix, &f.arg_value_suffix }) {
            if (!d->empty() && confusable(so, trim_ws(*d)))
                return no("section_open " + show(so) + " overlaps delimiter " + show(*d));
            if (!d->empty() && confusable(sc, trim_ws(*d)))
                return no("section_close " + show(sc) + " overlaps delimiter " + show(*d));
        }
    }
    if (!f.call_separator.empty())
        return no("a separator between calls (call_separator " + show(f.call_separator) +
                  ") is not read by the streaming parser; tagged calls separate on whitespace");
    if (f.call_open.empty() && !f.call_close.empty())
        return no("call_close without call_open");
    const bool object = f.object_args();
    if (object && !f.section_open.empty())
        return no("section-wrapped object arguments are not supported");
    /* Without a suffix the name ends at what follows it: the object's `{`, or a tagged call's
     * first argument or close. */
    if (!f.name_suffix.empty()) {
        if (trim_ws(f.name_suffix).empty())
            return no("name_suffix must end the function name with something besides whitespace");
        if (is_ws((unsigned char)f.name_suffix[0]))
            return no("name_suffix " + show(f.name_suffix) + " may not begin with whitespace");
    }
    if (!object) {
        if (f.call_end.empty()) return no("call_end is empty: a call must close with its own tag");
        if (f.arg_prefix.empty()) return no("arg_prefix is empty");
        if (f.arg_value_suffix.empty()) return no("arg_value_suffix is empty");
        if (f.arg_name_suffix.empty() && f.arg_value_prefix.empty())
            return no("nothing ends an argument name: arg_name_suffix and arg_value_prefix are "
                      "empty");
        if (f.name_suffix.empty()) {
            /* The name runs to the first arg_prefix or call_end. A wrapped format also reads
             * name_prefix alone as a call, told from prose only by its closed name tag, and
             * without a suffix there is none. */
            if (!f.call_open.empty())
                return no("name_suffix is empty, which a tagged call allows only when name_prefix "
                          "opens it, and call_open is set");
            if (is_ws((unsigned char)f.arg_prefix[0]))
                return no("arg_prefix " + show(f.arg_prefix) + " may not begin with whitespace "
                          "when it is what ends the function name");
        }
        if (f.string_value != ChatValue::Raw && f.string_value != ChatValue::Cdata)
            return no("string_value must be RAD_CHAT_VALUE_RAW or RAD_CHAT_VALUE_CDATA");
        if ((uint32_t)f.other_value > (uint32_t)ChatValue::Json)
            return no("other_value is not a RAD_CHAT_VALUE_* value");
        if (!f.object_string_open.empty() || !f.object_string_close.empty() ||
            f.object_bare_keys)
            return no("the object_* fields describe an argument object, and args_form is "
                      "RAD_CHAT_ARGS_TAGGED");
        if (f.typed_tags()) {
            /* The two tags are told apart where the name ends: one may not begin the other. */
            if (f.arg_value_prefix.empty())
                return no("arg_json_value_prefix is set and arg_value_prefix, the string tag, is "
                          "empty");
            const std::string s = f.arg_name_suffix + f.arg_value_prefix;
            const std::string j = f.arg_name_suffix + f.arg_json_value_prefix;
            if (confusable(s, j))
                return no("arg_value_prefix " + show(f.arg_value_prefix) +
                          " and arg_json_value_prefix " + show(f.arg_json_value_prefix) +
                          " cannot be told apart");
            if (f.other_value != ChatValue::Json)
                return no("arg_json_value_prefix opens a JSON value, and other_value is not "
                          "RAD_CHAT_VALUE_JSON");
        }
    } else {
        for (const std::string* s : { &f.arg_prefix, &f.arg_name_suffix, &f.arg_value_prefix,
                                      &f.arg_json_value_prefix, &f.arg_value_suffix })
            if (!s->empty())
                return no("the arg_* fields describe tagged arguments, and args_form is "
                          "RAD_CHAT_ARGS_OBJECT");
        if (!f.call_end.empty() && trim_ws(f.call_end).empty())
            return no("call_end is only whitespace");
        const int st = object_check(f, why);
        if (st < 0) return st;
    }

    /* A wrapped format's content is scanned for the wrapper's core and for name_prefix alone,
     * which opens a call written without its wrapper; the two must not be confusable. */
    if (!f.call_open.empty()) {
        const std::string co = trim_ws(f.call_open);
        if (co.empty()) return no("call_open is only whitespace");
        if (confusable(co, f.name_prefix))
            return no("call_open " + show(f.call_open) + " and name_prefix " +
                      show(f.name_prefix) + " cannot be told apart");
    }
    const std::string opener = f.call_open.empty() ? f.name_prefix : f.call_open;
    const std::string section_opener = trim_ws(f.section_open);
    for (const std::string& b : f.reasoning_breaks)
        if (b != opener && (section_opener.empty() || b != section_opener))
            return no("reasoning break " + show(b) + " is not a call or section opener");
    if (!f.reasoning_breaks.empty() && !f.has_reasoning())
        return no("reasoning_breaks without a reasoning block");
    for (const std::string& t : f.triggers)
        if (t.empty()) return no("an empty grammar trigger");

    /* Delimiters scanned together must be reported in start order, which an automaton does only
     * when no delimiter occurs inside another. */
    if (f.has_reasoning()) {
        const std::string close = trim_ws(f.reasoning_close);
        for (const std::string& b : f.reasoning_breaks)
            if (confusable(close, b))
                return no("reasoning break " + show(b) + " overlaps reasoning_close " +
                          show(close));
    }
    if (!object) {
        /* Inside a call the next tag is an argument or the close. The close also matches by its
         * trimmed core, since a reply that strays from the format may drop its whitespace. */
        const std::string end = trim_ws(f.call_end);
        if (end.empty()) return no("call_end is only whitespace");
        for (const std::string* e : { &f.call_end, &end })
            if (confusable(f.arg_prefix, *e))
                return no("arg_prefix " + show(f.arg_prefix) + " and call_end " +
                          show(f.call_end) + " cannot be told apart");
    }
    return reasoning_anywhere_check(f, why);
}

/* ================================================================== the plugin's declaration */

int chat_format_from_abi(const RadChatFormat* in, ChatFormat* out, std::string* why) {
    if (!in || !out) {
        if (why) *why = "no format";
        return RAD_E_INVAL;
    }
    if (in->struct_size < offsetof(RadChatFormat, name) + sizeof(in->name)) {
        if (why) *why = "struct_size " + std::to_string(in->struct_size) + " is too small";
        return RAD_E_INVAL;
    }
    RadChatFormat c;
    std::memset(&c, 0, sizeof c);
    std::memcpy(&c, in, std::min<size_t>(in->struct_size, sizeof c));

    auto s = [](const char* p) { return p ? std::string(p) : std::string(); };
    auto list = [](const char* const* v, uint32_t n, std::vector<std::string>* o) {
        o->clear();
        for (uint32_t i = 0; v && i < n; ++i)
            if (v[i]) o->emplace_back(v[i]);
    };

    ChatFormat f;
    f.name            = s(c.name);
    f.reasoning_open  = s(c.reasoning_open);
    f.reasoning_close = s(c.reasoning_close);
    list(c.reasoning_breaks, c.n_reasoning_breaks, &f.reasoning_breaks);
    f.reasoning_in_prompt = (ChatThink)c.reasoning_in_prompt;
    f.section_open    = s(c.section_open);
    f.section_close   = s(c.section_close);
    f.call_separator  = s(c.call_separator);
    f.call_open       = s(c.call_open);
    f.call_close      = s(c.call_close);
    f.name_prefix     = s(c.name_prefix);
    f.name_suffix     = s(c.name_suffix);
    f.call_end        = s(c.call_end);
    f.arg_prefix      = s(c.arg_prefix);
    f.arg_name_suffix = s(c.arg_name_suffix);
    f.arg_value_prefix = s(c.arg_value_prefix);
    f.arg_value_suffix = s(c.arg_value_suffix);
    f.string_value    = (ChatValue)c.string_value;
    f.other_value     = (ChatValue)c.other_value;
    f.value_newlines  = c.value_newlines != 0;
    list(c.triggers, c.n_triggers, &f.triggers);
    f.args_form       = (ChatArgs)c.args_form;
    f.object_string_open  = s(c.object_string_open);
    f.object_string_close = s(c.object_string_close);
    f.object_bare_keys    = c.object_bare_keys != 0;
    f.reasoning_anywhere  = c.reasoning_anywhere != 0;
    f.arg_json_value_prefix = s(c.arg_json_value_prefix);
    f.content_open        = s(c.content_open);
    *out = std::move(f);
    return RAD_OK;
}

/* ================================================================== the template's format */

int chat_format_from_template(const std::string& jinja, const std::string& bos,
                              const std::string& eos, ChatFormat* out,
                              std::string* tools_refused, std::string* why) {
    std::string refused_unused, why_unused;
    if (!tools_refused) tools_refused = &refused_unused;
    if (!why) why = &why_unused;
    tools_refused->clear();
    const std::string src = jinja.empty() ? std::string(ChatTemplate::chatml_source()) : jinja;

    ChatFormat f;
    f.name = "template";
    try {
        common_chat_template tmpl(src, bos, eos);

        /* A template llama.cpp reads with a hand-written handler is one the analysis never
         * describes, and its markers would be read as if it had. */
        autoparser::generation_params probe;
        probe.messages = ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } });
        probe.add_generation_prompt = true;
        if (common_chat_try_specialized_template(tmpl, src, probe)) {
            *why = "the chat template is one llama.cpp reads with a hand-written handler, whose "
                   "reply format the template analysis does not describe";
            return RAD_E_UNSUPPORTED;
        }

        autoparser::autoparser a;
        a.analyze_template(tmpl);

        if (a.reasoning.mode != autoparser::reasoning_mode::NONE && !a.reasoning.end.empty()) {
            f.reasoning_open  = a.reasoning.start;
            f.reasoning_close = a.reasoning.end;
        }
        if (a.content.mode != autoparser::content_mode::PLAIN) {
            *why = "the template wraps its answer in content markers, which the streaming parser "
                   "does not read";
            return RAD_E_UNSUPPORTED;
        }

        const auto& t = a.tools;
        const bool calls = t.format.mode != autoparser::tool_format::NONE &&
                           a.jinja_caps.supports_tool_calls;
        std::string refuse;
        if (calls) {
            if (t.format.mode == autoparser::tool_format::JSON_NATIVE)
                refuse = "the template writes each call as a JSON object";
            else if (t.format.mode == autoparser::tool_format::TAG_WITH_JSON)
                refuse = "the template writes a call's arguments as one JSON object inside a tag";
            else if (t.call_id.pos != autoparser::call_id_position::NONE)
                refuse = "the template writes a call id";
            else if (!t.arguments.start.empty() || !t.arguments.end.empty())
                refuse = "the template wraps a call's arguments in " + show(t.arguments.start) +
                         " .. " + show(t.arguments.end);
            else if (!t.arguments.separator.empty())
                refuse = "the template separates arguments with " + show(t.arguments.separator);
        }
        if (calls && refuse.empty()) {
            ChatFormat c = f;
            c.section_open     = t.format.section_start;
            c.section_close    = t.format.section_end;
            c.call_open        = t.format.per_call_start;
            c.call_close       = t.format.per_call_end;
            c.name_prefix      = t.function.name_prefix;
            c.name_suffix      = t.function.name_suffix;
            c.call_end         = t.function.close;
            c.arg_prefix       = t.arguments.name_prefix;
            c.arg_name_suffix  = t.arguments.name_suffix;
            c.arg_value_prefix = t.arguments.value_prefix;
            c.arg_value_suffix = t.arguments.value_suffix;
            /* A name the per-call opener itself opens, with no tag of its own around it
             * (GLM's `<tool_call>NAME<arg_key>`): the opener is the name's prefix and the
             * per-call close the call's own, so the name ends at the first argument or close. */
            if (c.name_prefix.empty() && !c.call_open.empty() && c.call_end.empty()) {
                c.name_prefix = c.call_open;
                c.call_end    = c.call_close;
                c.call_open.clear();
                c.call_close.clear();
            }
            c.string_value     = ChatValue::Cdata;
            c.other_value      = ChatValue::Json;
            c.value_newlines   = value_on_own_lines(tmpl, c);
            /* The section or per-call opener both triggers the lazy grammar and ends an
             * unclosed reasoning block. A format with neither gets no grammar and no break,
             * since its calls open with the name prefix, which a model also writes in prose. */
            const std::string& marker = !t.format.section_start.empty() ? t.format.section_start
                                                                         : t.format.per_call_start;
            if (!marker.empty()) {
                c.triggers.push_back(marker);
                if (c.has_reasoning()) c.reasoning_breaks.push_back(marker);
            }
            std::string bad;
            if (chat_format_check(c, &bad) == RAD_OK) f = std::move(c);
            else refuse = bad;
        }
        if (!refuse.empty()) *tools_refused = refuse;
    } catch (const std::exception& e) {
        *why = std::string("the chat template could not be analysed: ") + e.what();
        return RAD_E_FORMAT;
    }
    *out = std::move(f);
    return RAD_OK;
}

std::string chat_format_describe(const ChatFormat& f) {
    std::string s = f.name + ":";
    if (f.has_reasoning())
        s += " reasoning " + show(f.reasoning_open) + ".." + show(f.reasoning_close) +
             (f.reasoning_anywhere ? " anywhere" : "");
    else
        s += " no reasoning";
    if (!f.content_open.empty()) s += "; content " + show(f.content_open);
    if (f.has_tools() && !f.section_open.empty())
        s += "; section " + show(f.section_open) + ".." + show(f.section_close);
    if (f.has_tools() && f.object_args()) {
        s += "; calls " + show(f.call_open) + show(f.name_prefix) + "NAME" + show(f.name_suffix) +
             " {OBJECT";
        if (!f.object_string_open.empty())
            s += ", strings " + show(f.object_string_open) + ".." + show(f.object_string_close);
        if (f.object_bare_keys) s += ", bare keys";
        s += "} " + show(f.call_end) + show(f.call_close);
    } else if (f.has_tools()) {
        s += "; calls " + show(f.call_open) + show(f.name_prefix) + "NAME" + show(f.name_suffix) +
             " " + show(f.arg_prefix) + "ARG" + show(f.arg_name_suffix) + show(f.arg_value_prefix) +
             (f.typed_tags() ? "|" + show(f.arg_json_value_prefix) : std::string()) +
             "VALUE" + show(f.arg_value_suffix) + " " + show(f.call_end) + show(f.call_close);
    }
    if (f.has_tools()) {
        if (!f.triggers.empty()) {
            s += "; grammar on";
            for (const std::string& t : f.triggers) s += " " + show(t);
        } else {
            s += "; no grammar";
        }
    } else {
        s += "; no tool calls";
    }
    return s;
}

std::vector<std::string> chat_format_markers(const ChatFormat& f) {
    std::vector<std::string> out;
    for (const std::string* s : { &f.reasoning_open, &f.reasoning_close, &f.content_open,
                                  &f.section_open,
                                  &f.section_close, &f.call_open, &f.call_close, &f.name_prefix,
                                  &f.name_suffix, &f.call_end, &f.arg_prefix, &f.arg_name_suffix,
                                  &f.arg_value_prefix, &f.arg_json_value_prefix,
                                  &f.arg_value_suffix, &f.object_string_open,
                                  &f.object_string_close }) {
        const std::string t = trim_ws(*s);
        if (!t.empty() && std::find(out.begin(), out.end(), t) == out.end()) out.push_back(t);
    }
    return out;
}

/* ================================================================== grammar
 *
 * The grammar is an expression tree rendered by the vendored PEG-to-GBNF rules, so a
 * template-derived format yields the same text as the vendored generator. Sequences and
 * alternations flatten as they are built. Parentheses go around a sequence or alternation inside
 * a sequence, an alternation inside an alternation, and a repeated sequence or alternation. */
namespace {

struct G {
    enum K : uint8_t { Eps, Lit, Seq, Alt, Rep, Leaf, Wrap } k = Eps;
    std::string s;               /* Lit: the text; Leaf: rendered GBNF */
    std::vector<G> kids;
    int min = 0, max = -1;
};

G eps() { return G{}; }
G lit(const std::string& s) { G g; g.k = G::Lit; g.s = s; return g; }
G leaf(const std::string& s) { G g; g.k = G::Leaf; g.s = s; return g; }
G space() { return leaf("space"); }
G rule_ref(const std::string& name) { return leaf(name); }

/* A tag or atomic around a node: renders and parenthesises as the node, but an enclosing
 * sequence does not flatten into it. */
G wrap(G inner) { G g; g.k = G::Wrap; g.kids.push_back(std::move(inner)); return g; }

/* A sequence or alternation of `parts`, a part of the same kind spliced in flat. */
G flat(G::K k, std::vector<G> parts) {
    G g; g.k = k;
    for (G& p : parts) {
        if (p.k == k) for (G& c : p.kids) g.kids.push_back(std::move(c));
        else g.kids.push_back(std::move(p));
    }
    return g;
}
G seq(std::vector<G> parts) { return flat(G::Seq, std::move(parts)); }
G alt(std::vector<G> parts) { return flat(G::Alt, std::move(parts)); }
G rep(G child, int min, int max) {
    G g; g.k = G::Rep; g.min = min; g.max = max; g.kids.push_back(std::move(child)); return g;
}
G opt(G child) { return rep(std::move(child), 0, 1); }

const G& effective(const G& g) { return g.k == G::Wrap ? effective(g.kids[0]) : g; }

std::string render(const G& g);

std::string render_rep_suffix(int min, int max) {
    if (min == 0 && max == 1) return "?";
    if (min == 0 && max == -1) return "*";
    if (min == 1 && max == -1) return "+";
    if (max == -1) return "{" + std::to_string(min) + ",}";
    if (min == max) return min == 1 ? "" : "{" + std::to_string(min) + "}";
    return "{" + std::to_string(min) + "," + std::to_string(max) + "}";
}

std::string render(const G& g) {
    switch (g.k) {
    case G::Eps: return "";
    case G::Lit: return gbnf_format_literal(g.s);
    case G::Leaf: return g.s;
    case G::Wrap: return render(g.kids[0]);
    case G::Seq: {
        std::string s;
        for (const G& c : g.kids) {
            const std::string r = render(c);
            if (r.empty()) continue;
            if (!s.empty()) s += " ";
            const G& e = effective(c);
            if (e.k == G::Seq || e.k == G::Alt) s += "(" + r + ")";
            else s += r;
        }
        return s;
    }
    case G::Alt: {
        std::string s;
        for (const G& c : g.kids) {
            if (!s.empty()) s += " | ";
            const std::string r = render(c);
            if (effective(c).k == G::Alt) s += "(" + r + ")";
            else s += r;
        }
        return s;
    }
    case G::Rep: {
        std::string r = render(g.kids[0]);
        const G& e = effective(g.kids[0]);
        if (e.k == G::Seq || e.k == G::Alt) r = "(" + r + ")";
        return r + render_rep_suffix(g.min, g.max);
    }
    }
    return "";
}

/* A character in a GBNF class, escaped as the vendored generator escapes it. */
std::string class_char(uint32_t c) {
    if (c == '-' || c == ']' || c == '[' || c == '\\') return std::string("\\") + (char)c;
    if (c == '\n') return "\\n";
    if (c == '\t') return "\\t";
    if (c == '\r') return "\\r";
    if (c >= 0x20 && c <= 0x7E) return std::string(1, (char)c);
    static const char* hex = "0123456789ABCDEF";
    const int digits = c <= 0xFF ? 2 : c <= 0xFFFF ? 4 : 8;
    std::string s = digits == 2 ? "\\x" : digits == 4 ? "\\u" : "\\U";
    for (int i = digits - 1; i >= 0; --i) s += hex[(c >> (4 * i)) & 15];
    return s;
}

std::vector<uint32_t> codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        uint32_t cp = c;
        size_t   n = 1;
        if (c >= 0xF0)      { cp = c & 0x07; n = 4; }
        else if (c >= 0xE0) { cp = c & 0x0F; n = 3; }
        else if (c >= 0xC0) { cp = c & 0x1F; n = 2; }
        if (i + n > s.size()) break;
        for (size_t k = 1; k < n; ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        out.push_back(cp);
        i += n;
    }
    return out;
}

/* A rule name as the vendored builder spells it: every run of characters outside
 * [a-zA-Z0-9-] becomes one '-'. */
std::string rule_name(const std::string& s) {
    std::string out;
    bool run = false;
    for (char ch : s) {
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') || ch == '-';
        if (ok) { out += ch; run = false; }
        else if (!run) { out += '-'; run = true; }
    }
    return out;
}

std::string utf8(const std::vector<uint32_t>& cps) {
    std::string s;
    for (uint32_t c : cps) {
        if (c < 0x80) s += (char)c;
        else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) {
            s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F));
            s += (char)(0x80 | (c & 0x3F));
        } else {
            s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F));
            s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F));
        }
    }
    return s;
}

/* "Any text not containing one of these", as a repetition over the trie of the delimiters:
 * for each proper prefix, that prefix followed by a character that does not continue it. */
std::string until_pattern(const std::vector<std::string>& delims) {
    if (delims.empty()) return ".*";
    struct Node { std::map<uint32_t, size_t> kids; bool word = false; };
    std::vector<Node> t(1);
    for (const std::string& d : delims) {
        size_t cur = 0;
        for (uint32_t c : codepoints(d)) {
            auto it = t[cur].kids.find(c);
            if (it == t[cur].kids.end()) {
                t.emplace_back();
                t[cur].kids[c] = t.size() - 1;
                cur = t.size() - 1;
            } else {
                cur = it->second;
            }
        }
        t[cur].word = true;
    }
    std::string pattern;
    std::vector<uint32_t> prefix;
    std::function<void(size_t)> walk = [&](size_t n) {
        if (!t[n].word && !t[n].kids.empty()) {
            std::string cls;
            for (const auto& kv : t[n].kids) cls += class_char(kv.first);
            if (!pattern.empty()) pattern += " | ";
            if (!prefix.empty()) pattern += gbnf_format_literal(utf8(prefix)) + " [^" + cls + "]";
            else pattern += "[^" + cls + "]";
        }
        for (const auto& kv : t[n].kids) {
            prefix.push_back(kv.first);
            walk(kv.second);
            prefix.pop_back();
        }
    };
    walk(0);
    return "(" + pattern + ")*";
}

/* The generic JSON rules the vendored generator emits whenever a schema types an argument or
 * document. Nothing references them (the schema's own rules constrain), but they are emitted so
 * the output compares byte for byte with the vendored grammar. */
void add_json_rules(const common_grammar_builder& b) {
    b.add_rule("json-array", "\"[\" space (\"]\" | json-value (\",\" space json-value)* space "
                             "\"]\") space");
    b.add_rule("json-bool", "(\"true\" | \"false\") space");
    b.add_rule("json-null", "\"null\" space");
    b.add_rule("json-number", "\"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ((\"e\" | \"E\") "
                              "[+-]? [0-9]+)? space");
    b.add_rule("json-object", "\"{\" space (\"}\" | json-string space \":\" space json-value "
                              "(space \",\" space json-string space \":\" space json-value)* "
                              "space \"}\") space");
    b.add_rule("json-string", "\"\\\"\" ( [^\"\\\\] | \"\\\\\" ( [\"\\\\/ bfnrt] | \"u\" "
                              "[0-9a-fA-F]{4} ) )* \"\\\"\" space");
    b.add_rule("json-value", "json-object | json-array | json-string | json-number | json-bool | "
                             "json-null");
}

/* The object family's argument object in its dialect: JSON, or strings between the declared
 * delimiters and bare keys. Like llama.cpp's Gemma 4 grammar, it fixes the syntax and leaves
 * values unconstrained, since a tool's schema describes JSON and the dialect is not JSON.
 * Whitespace is allowed only after a separator and inside the brackets. */
void add_object_rules(const common_grammar_builder& b, const ChatFormat& f) {
    const std::string& so = f.object_string_open;
    if (!so.empty())
        b.add_rule("obj-string", gbnf_format_literal(so) + " " +
                                 until_pattern({ f.object_string_close }) + " " +
                                 gbnf_format_literal(f.object_string_close));
    else
        b.add_rule("obj-string", "\"\\\"\" ( [^\"\\\\] | \"\\\\\" ( [\"\\\\/ bfnrt] | \"u\" "
                                 "[0-9a-fA-F]{4} ) )* \"\\\"\"");
    b.add_rule("obj-number", "\"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?");
    /* A dialect with its own strings also admits Python-quoted `'...'` values, which Gemma 4
     * writes for arguments it was given as JSON. Without this the model takes the next allowed
     * token, `-`, and the string becomes a number. The parser reads the quoted text verbatim
     * except for an escaped quote. */
    const char* sq = so.empty() ? "" : " | obj-sqstring";
    if (!so.empty()) b.add_rule("obj-sqstring", "\"'\" ( [^'\\\\] | \"\\\\\" [^\\x00] )* \"'\"");
    b.add_rule("obj-value", std::string("obj-object | obj-array | obj-string") + sq +
                            " | obj-number | \"true\" | \"false\" | \"null\"");
    b.add_rule("obj-array", "\"[\" space (\"]\" | obj-value (\",\" space obj-value)* space "
                            "\"]\")");
    if (f.object_bare_keys) {
        /* A bare key: no whitespace, no punctuation, and not the string opener's first byte. */
        std::string cls = "[^ \\t\\n\\r:,{}\\[\\]\"";
        if (!so.empty()) cls += class_char((unsigned char)so[0]);
        cls += "]+";
        b.add_rule("obj-key", cls + " | obj-string");
    } else {
        b.add_rule("obj-key", "obj-string");
    }
    b.add_rule("obj-member", "obj-key \":\" space obj-value");
    b.add_rule("obj-object", "\"{\" space (\"}\" | obj-member (\",\" space obj-member)* space "
                             "\"}\")");
}

G optspace(const std::string& tag) {
    size_t a = 0, b = tag.size();
    while (a < b && is_ws((unsigned char)tag[a])) ++a;
    while (b > a && is_ws((unsigned char)tag[b - 1])) --b;
    std::vector<G> parts{ eps() };
    for (size_t i = 0; i < a; ++i) parts.push_back(opt(lit(std::string(1, tag[i]))));
    parts.push_back(lit(tag.substr(a, b - a)));
    for (size_t i = b; i < tag.size(); ++i) parts.push_back(opt(lit(std::string(1, tag[i]))));
    return seq(std::move(parts));
}

/* The reasoning block as the vendored generator builds it. */
G reasoning_expr(const ChatFormat& f, const std::vector<std::string>& breaks) {
    const std::string end = trim_ws(f.reasoning_close);
    if (!breaks.empty()) {
        std::vector<std::string> stops{ end };
        stops.insert(stops.end(), breaks.begin(), breaks.end());
        G unclosed = seq({ wrap(leaf(until_pattern(stops))), eps() });
        G closed = seq({ wrap(leaf(until_pattern({ end }))), optspace(f.reasoning_close) });
        if (!f.reasoning_open.empty())
            return opt(alt({ seq({ optspace(f.reasoning_open), unclosed }),
                             seq({ optspace(f.reasoning_open), closed }) }));
        return opt(alt({ unclosed, closed }));
    }
    if (!f.reasoning_open.empty())
        return opt(seq({ optspace(f.reasoning_open), wrap(leaf(until_pattern({ end }))),
                         optspace(f.reasoning_close) }));
    return opt(seq({ wrap(leaf(until_pattern({ end }))), optspace(f.reasoning_close) }));
}

}  /* namespace */

bool chat_function_tool(const ojson& tool) {
    return tool.is_object() && tool.contains("type") && tool.at("type") == "function" &&
           tool.contains("function");
}

int chat_format_grammar(const ChatFormat& f, const ChatGrammarRequest& req, ChatGrammar* out,
                        std::string* why) {
    *out = ChatGrammar{};
    const ojson none;
    const ojson& tools = req.tools ? *req.tools : none;
    const bool has_tools = f.has_tools() && tools.is_array() && !tools.empty();
    const bool has_rf = req.json_schema && req.json_schema->is_object() && !req.json_schema->empty();
    if (req.require_call && (has_rf || !has_tools)) {
        if (why)
            *why = has_rf ? "a required tool call and a response format cannot both be the "
                            "whole reply"
                          : "a tool call is required, but there are no tools this format can call";
        return RAD_E_INVAL;
    }
    if (!has_rf && !(has_tools && !f.triggers.empty())) return RAD_OK;

    const std::string opener = f.call_open.empty() ? f.name_prefix : f.call_open;
    std::vector<std::string> breaks;
    if (has_tools && req.extract_reasoning && f.has_reasoning()) breaks = f.reasoning_breaks;

    try {
        out->text = build_grammar([&](const common_grammar_builder& b) {
            for (const auto& tool : tools) {
                if (!chat_function_tool(tool)) continue;
                const auto& fn = tool.at("function");
                auto schema = fn.contains("parameters") ? fn.at("parameters") : ojson::object();
                b.resolve_refs(schema);
            }
            if (has_rf) {
                auto schema = *req.json_schema;
                b.resolve_refs(schema);
            }

            /* A grammar that is the whole reply starts where the generation prompt does, so the
             * prompt's assistant prefix and the reasoning block come first. */
            auto whole_reply = [&]() {
                std::string gp = req.generation_prompt;
                if (f.has_reasoning() && !f.reasoning_open.empty()) {
                    const size_t at = gp.find(f.reasoning_open);
                    if (at != std::string::npos) gp = gp.substr(0, at);
                }
                std::vector<G> root{ gp.empty() ? eps() : lit(gp), space() };
                if (f.has_reasoning() && req.extract_reasoning) root.push_back(reasoning_expr(f, breaks));
                root.push_back(space());
                return root;
            };
            if (has_rf) {
                /* The whole reply: what follows the generation prompt, the reasoning block, and
                 * the document, fenced or not. */
                b.add_rule("response-format", b.add_schema("response-format-schema",
                                                           *req.json_schema));
                add_json_rules(b);
                std::vector<G> root = whole_reply();
                root.push_back(alt({ seq({ lit("```json"), space(), rule_ref("response-format"), space(),
                                           lit("```") }),
                                     rule_ref("response-format") }));
                b.add_rule("root", render(seq(std::move(root))));
                return;
            }

            /* The lazy call grammar: only what follows a trigger. */
            const std::string name_end = f.arg_name_suffix.empty() ? f.arg_value_prefix
                                                                   : f.arg_name_suffix;
            std::vector<G> choices;
            bool typed = false;
            for (const auto& tool : tools) {
                if (!chat_function_tool(tool)) continue;
                const auto& fn = tool.at("function");
                const std::string name = fn.at("name");
                if (!req.only_tool.empty() && name != req.only_tool) continue;
                if (f.object_args()) {
                    /* The name, then the object; the object's own closing brace may be all
                     * that ends the call. */
                    G open = wrap(seq({ lit(f.name_prefix), wrap(lit(name)), lit(f.name_suffix) }));
                    G func = f.call_end.empty()
                                 ? seq({ open, rule_ref("obj-object") })
                                 : seq({ open, rule_ref("obj-object"), space(),
                                         wrap(lit(f.call_end)) });
                    b.add_rule(rule_name("tool-" + name), render(func));
                    choices.push_back(rule_ref(rule_name("tool-" + name)));
                    continue;
                }
                auto params = fn.contains("parameters") ? fn.at("parameters") : ojson::object();
                const auto properties = params.contains("properties") ? params.at("properties")
                                                                     : ojson::object();
                std::set<std::string> required;
                if (params.contains("required")) params.at("required").get_to(required);
                common_schema_info info;
                info.resolve_refs(params);

                std::vector<G> req_arms, opt_arms;
                for (const auto& [pname, pschema] : properties.items()) {
                    /* An optional argument no value satisfies (`false`, or the `not` of an
                     * always-true schema, e.g. Arktype or TypeBox `never`) gets no arm; a model
                     * that writes it reaches the undeclared arm. The template path does the same
                     * (vendor/patches/0012, 0014). A required one goes through the schema and is
                     * refused there by name. */
                    if (!required.count(pname) && info.unsatisfiable(pschema)) continue;
                    G value;
                    /* A format whose tag types the value writes the JSON tag before a JSON
                     * value; the string tag otherwise. */
                    const std::string* vprefix = &f.arg_value_prefix;
                    if (f.other_value != ChatValue::Json || info.resolves_to_string(pschema)) {
                        value = wrap(rule_ref("until-suffix"));
                    } else {
                        typed = true;
                        const std::string sch = b.add_schema(
                            "tool-" + name + "-arg-" + pname + "-schema", pschema);
                        value = alt({ seq({ wrap(leaf(sch)), space() }), wrap(rule_ref("until-suffix")) });
                        if (f.typed_tags()) vprefix = &f.arg_json_value_prefix;
                    }
                    G arm = wrap(seq({ wrap(seq({ lit(f.arg_prefix), wrap(lit(pname)),
                                                  lit(f.arg_name_suffix) })),
                                       lit(*vprefix), value,
                                       wrap(lit(f.arg_value_suffix)) }));
                    const std::string rule = rule_name("tool-" + name + "-arg-" + pname);
                    b.add_rule(rule, render(arm));
                    (required.count(pname) ? req_arms : opt_arms).push_back(rule_ref(rule));
                }
                std::vector<G> any = req_arms;
                any.insert(any.end(), opt_arms.begin(), opt_arms.end());
                /* The generic arm for an undeclared name; not emitted for a tool that declares
                 * no parameters, since it takes no arguments. */
                if (!any.empty()) {
                    G tag = f.typed_tags() ? alt({ lit(f.arg_value_prefix),
                                                   lit(f.arg_json_value_prefix) })
                                           : lit(f.arg_value_prefix);
                    G arm = wrap(seq({ wrap(seq({ lit(f.arg_prefix),
                                                  wrap(leaf(until_pattern({ name_end }))),
                                                  lit(f.arg_name_suffix) })),
                                       tag, wrap(rule_ref("until-suffix")),
                                       wrap(lit(f.arg_value_suffix)) }));
                    const std::string rule = rule_name("tool-" + name + "-arg-undeclared");
                    b.add_rule(rule, render(arm));
                    any.push_back(rule_ref(rule));
                }
                G any_arg = alt(any);
                G args = eps();
                if (!req_arms.empty()) args = seq({ any_arg, rep(seq({ space(), any_arg }), 0, -1) });
                else if (!opt_arms.empty()) args = seq({ eps(), rep(seq({ space(), any_arg }), 0, -1) });

                /* Without a suffix the name ends where its arguments or close begin. */
                G open = wrap(seq({ lit(f.name_prefix), wrap(lit(name)),
                                    f.name_suffix.empty() ? eps() : lit(f.name_suffix) }));
                G func = seq({ open, eps(), space(), args, space(), wrap(lit(f.call_end)) });
                b.add_rule(rule_name("tool-" + name), render(func));
                choices.push_back(rule_ref(rule_name("tool-" + name)));
            }
            if (choices.empty() && !req.only_tool.empty())
                throw std::invalid_argument("no tool named \"" + req.only_tool + "\" to call");
            if (typed) add_json_rules(b);
            if (!choices.empty() && f.object_args()) add_object_rules(b, f);
            else if (!choices.empty()) b.add_rule("until-suffix", until_pattern({ f.arg_value_suffix }));

            G choice = alt(choices);
            G calls;
            if (!f.call_open.empty()) {
                G wrapped = seq({ lit(f.call_open), space(), choice, space(), lit(f.call_close) });
                calls = req.parallel_tool_calls
                            ? seq({ wrapped, rep(seq({ space(), wrapped }), 0, -1), space() })
                            : seq({ wrapped, space() });
            } else {
                /* Section-wrapped dialects contain only calls and whitespace. In particular,
                 * inter-call prose must not consume a section close and admit later calls. */
                G next = f.section_open.empty()
                             ? seq({ space(), opt(wrap(leaf(until_pattern({ opener })))), choice })
                             : seq({ space(), choice });
                calls = req.parallel_tool_calls
                            ? seq({ choice, rep(next, 0, -1), space() })
                            : seq({ choice, space() });
            }
            if (!f.section_open.empty())
                calls = seq({ lit(f.section_open), space(), calls, space(), lit(f.section_close) });
            b.add_rule("tool-call", render(calls));
            if (req.require_call) {
                /* Nothing after the calls: a complete grammar admits only the end of the turn,
                 * so the reply stops at the last call's closer. */
                std::vector<G> root = whole_reply();
                root.push_back(rule_ref("tool-call"));
                b.add_rule("root", render(seq(std::move(root))));
            } else {
                b.add_rule("root", "tool-call");
            }
        });
    } catch (const std::exception& e) {
        if (why) *why = e.what();
        return RAD_E_INVAL;
    }
    out->lazy = !has_rf && !req.require_call;
    if (out->lazy) out->trigger_words = f.triggers;
    return RAD_OK;
}

}  /* namespace rad */
