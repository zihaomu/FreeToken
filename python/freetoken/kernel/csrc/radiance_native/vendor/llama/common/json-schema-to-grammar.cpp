#include "json-schema-to-grammar.h"
#include "common.h"

#include "util/json.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = rad::Json;

static std::string build_repetition(const std::string & item_rule, int min_items, int max_items, const std::string & separator_rule = "") {
    auto has_max = max_items != std::numeric_limits<int>::max();

    if (max_items == 0) {
        return "";
    }
    if (min_items == 0 && max_items == 1) {
        return item_rule + "?";
    }

    if (separator_rule.empty()) {
        if (min_items == 1 && !has_max) {
            return item_rule + "+";
        }
        if (min_items == 0 && !has_max) {
            return item_rule + "*";
        }
        return item_rule + "{" + std::to_string(min_items) + "," + (has_max ? std::to_string(max_items) : "") + "}";
    }

    auto result = item_rule + " " + build_repetition("(" + separator_rule + " " + item_rule + ")", min_items == 0 ? 0 : min_items - 1, has_max ? max_items - 1 : max_items);
    if (min_items == 0) {
        result = "(" + result + ")?";
    }
    return result;
}

static void build_min_max_int(int64_t min_value, int64_t max_value, std::stringstream & out, int decimals_left = 16, bool top_level = true) {
    auto has_min = min_value != std::numeric_limits<int64_t>::min();
    auto has_max = max_value != std::numeric_limits<int64_t>::max();

    auto digit_range = [&](char from, char to) {
        out << "[";
        if (from == to) {
            out << from;
        } else {
            out << from << "-" << to;
        }
        out << "]";
    };
    auto more_digits = [&](int min_digits, int max_digits) {
        out << "[0-9]";
        if (min_digits == max_digits && min_digits == 1) {
            return;
        }
        out << "{";
        out << min_digits;
        if (max_digits != min_digits) {
            out << ",";
            if (max_digits != std::numeric_limits<int>::max()) {
                out << max_digits;
            }
        }
        out << "}";
    };
    std::function<void(const std::string_view &, const std::string_view &)> uniform_range =
        [&](const std::string_view & from, const std::string_view & to) {
            size_t i = 0;
            while (i < from.length() && i < to.length() && from[i] == to[i]) {
                i++;
            }
            if (i > 0) {
                out << "\"" << from.substr(0, i) << "\"";
            }
            if (i < from.length() && i < to.length()) {
                if (i > 0) {
                    out << " ";
                }
                auto sub_len = from.length() - i - 1;
                if (sub_len > 0) {
                    auto from_sub = from.substr(i + 1);
                    auto to_sub = to.substr(i + 1);
                    auto sub_zeros = string_repeat("0", sub_len);
                    auto sub_nines = string_repeat("9", sub_len);

                    auto to_reached = false;
                    out << "(";
                    if (from_sub == sub_zeros) {
                        digit_range(from[i], to[i] - 1);
                        out << " ";
                        more_digits(sub_len, sub_len);
                    } else {
                        out << "[" << from[i] << "] ";
                        out << "(";
                        uniform_range(from_sub, sub_nines);
                        out << ")";
                        if (from[i] < to[i] - 1) {
                            out << " | ";
                            if (to_sub == sub_nines) {
                                digit_range(from[i] + 1, to[i]);
                                to_reached = true;
                            } else {
                                digit_range(from[i] + 1, to[i] - 1);
                            }
                            out << " ";
                            more_digits(sub_len, sub_len);
                        }
                    }
                    if (!to_reached) {
                        out << " | ";
                        digit_range(to[i], to[i]);
                        out << " ";
                        uniform_range(sub_zeros, to_sub);
                    }
                    out << ")";
                } else {
                    out << "[" << from[i] << "-" << to[i] << "]";
                }
            }
        };

    if (has_min && has_max) {
        if (min_value < 0 && max_value < 0) {
            out << "\"-\" (";
            build_min_max_int(-max_value, -min_value, out, decimals_left, /* top_level= */ true);
            out << ")";
            return;
        }

        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(0, -min_value, out, decimals_left, /* top_level= */ true);
            out << ") | ";
            min_value = 0;
        }

        auto min_s = std::to_string(min_value);
        auto max_s = std::to_string(max_value);
        auto min_digits = min_s.length();
        auto max_digits = max_s.length();

        for (auto digits = min_digits; digits < max_digits; digits++) {
            uniform_range(min_s, string_repeat("9", digits));
            min_s = "1" + string_repeat("0", digits);
            out << " | ";
        }
        uniform_range(min_s, max_s);
        return;
    }

    auto less_decimals = std::max(decimals_left - 1, 1);

    if (has_min) {
        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(std::numeric_limits<int64_t>::min(), -min_value, out, decimals_left, /* top_level= */ false);
            out << ") | [0] | [1-9] ";
            more_digits(0, decimals_left - 1);
        } else if (min_value == 0) {
            if (top_level) {
                out << "[0] | [1-9] ";
                more_digits(0, less_decimals);
            } else {
                more_digits(1, decimals_left);
            }
        } else if (min_value <= 9) {
            char c = '0' + min_value;
            auto range_start = top_level ? '1' : '0';
            if (c > range_start) {
                digit_range(range_start, c - 1);
                out << " ";
                more_digits(1, less_decimals);
                out << " | ";
            }
            digit_range(c, '9');
            out << " ";
            more_digits(0, less_decimals);
        } else {
            auto min_s = std::to_string(min_value);
            auto len = min_s.length();
            auto c = min_s[0];

            if (c > '1') {
                digit_range(top_level ? '1' : '0', c - 1);
                out << " ";
                more_digits(len, less_decimals);
                out << " | ";
            }
            digit_range(c, c);
            out << " (";
            build_min_max_int(std::stoll(min_s.substr(1)), std::numeric_limits<int64_t>::max(), out, less_decimals, /* top_level= */ false);
            out << ")";
            if (c < '9') {
                out << " | ";
                digit_range(c + 1, '9');
                out << " ";
                more_digits(len - 1, less_decimals);
            }
        }
        return;
    }

    if (has_max) {
        if (max_value >= 0) {
            if (top_level) {
                out << "\"-\" [1-9] ";
                more_digits(0, less_decimals);
                out << " | ";
            }
            build_min_max_int(0, max_value, out, decimals_left, /* top_level= */ true);
        } else {
            out << "\"-\" (";
            build_min_max_int(-max_value, std::numeric_limits<int64_t>::max(), out, decimals_left, /* top_level= */ false);
            out << ")";
        }
        return;
    }

    throw std::runtime_error("At least one of min_value or max_value must be set");
}

const std::string SPACE_RULE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

struct BuiltinRule {
    std::string content;
    std::vector<std::string> deps;
};

static std::unordered_map<std::string, BuiltinRule> PRIMITIVE_RULES = {
    {"boolean", {"(\"true\" | \"false\") space", {}}},
    {"decimal-part", {"[0-9]{1,16}", {}}},
    {"integral-part", {"[0] | [1-9] [0-9]{0,15}", {}}},
    {"number", {"(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)? space", {"integral-part", "decimal-part"}}},
    {"integer", {"(\"-\"? integral-part) space", {"integral-part"}}},
    {"value", {"object | array | string | number | boolean | null", {"object", "array", "string", "number", "boolean", "null"}}},
    {"object", {"\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? \"}\" space", {"string", "value"}}},
    {"array", {"\"[\" space ( value (\",\" space value)* )? \"]\" space", {"value"}}},
    {"uuid", {"\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\" space", {}}},
    {"char",   {"[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", {}}},
    {"string", {"\"\\\"\" char* \"\\\"\" space", {"char"}}},
    {"null", {"\"null\" space", {}}},
};

static std::unordered_map<std::string, BuiltinRule> STRING_FORMAT_RULES = {
    {"date", {"[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", {}}},
    {"time", {"([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", {}}},
    {"date-time", {"date \"T\" time", {"date", "time"}}},
    {"date-string", {"\"\\\"\" date \"\\\"\" space", {"date"}}},
    {"time-string", {"\"\\\"\" time \"\\\"\" space", {"time"}}},
    {"date-time-string", {"\"\\\"\" date-time \"\\\"\" space", {"date-time"}}}
};

static bool is_reserved_name(const std::string & name) {
    static const std::unordered_set<std::string> RESERVED_NAMES = [] {
        std::unordered_set<std::string> s;
        s.insert("root");
        for (const auto & p : PRIMITIVE_RULES) {
            s.insert(p.first);
        }
        for (const auto & p : STRING_FORMAT_RULES) {
            s.insert(p.first);
        }
        return s;
    }();
    return RESERVED_NAMES.find(name) != RESERVED_NAMES.end();
}

static std::regex INVALID_RULE_CHARS_RE("[^a-zA-Z0-9-]+");
static std::regex GRAMMAR_LITERAL_ESCAPE_RE("[\r\n\"\\\\]");
static std::regex GRAMMAR_RANGE_LITERAL_ESCAPE_RE("[\r\n\"\\]\\-\\\\]");
static std::unordered_map<char, std::string> GRAMMAR_LITERAL_ESCAPES = {
    {'\r', "\\r"}, {'\n', "\\n"}, {'"', "\\\""}, {'-', "\\-"}, {']', "\\]"}, {'\\', "\\\\"}
};

static std::unordered_set<char> NON_LITERAL_SET = {'|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?'};

// RADIANCE: the escapes GBNF can actually spell. Anything else in a pattern is refused rather
// than copied into a grammar that will not load. See the use site in _visit_pattern.
static inline bool is_gbnf_escape(char c) {
    switch (c) {
        case 'x': case 'u': case 'U': case 't': case 'r': case 'n':
        case '\\': case '"': case '[': case ']':
            return true;
        default:
            return false;
    }
}

// RADIANCE: a JSON Schema `pattern` is unanchored (ECMA-262 search semantics, JSON Schema
// validation 6.3.3): "[0-9]+" admits any string holding a digit. _visit_pattern() reads only
// ^...$, so a missing anchor is written out as "any string text" on that side. That filler is
// not ".*": '.' here matches '"', and the string would run on past its closing quote. A
// one-sided anchor beside a top-level '|' binds to one branch only, so that pattern is
// returned as it came and refused.
static std::string anchor_pattern(const std::string & p) {
    static const std::string any = "(?:[^\"\\\\]|\\\\.)*";
    const bool starts = !p.empty() && p.front() == '^';
    size_t slashes = 0; // backslashes before a final '$': an odd count makes it a literal
    if (!p.empty() && p.back() == '$') {
        for (size_t k = p.size() - 1; k > 0 && p[k - 1] == '\\'; --k) slashes++;
    }
    const bool ends = p.size() > (starts ? 1u : 0u) && p.back() == '$' && slashes % 2 == 0;
    if (starts && ends) {
        return p;
    }
    const size_t from = starts ? 1 : 0;
    const std::string core = p.substr(from, p.size() - from - (ends ? 1 : 0));
    if (core.empty()) {
        return "^" + any + "$";
    }
    if (starts != ends) {
        int depth = 0;
        bool in_class = false;
        for (size_t k = 0; k < core.size(); ++k) {
            const char c = core[k];
            if (c == '\\') { ++k; continue; }
            if (in_class) { if (c == ']') in_class = false; continue; }
            if (c == '[') in_class = true;
            else if (c == '(') depth++;
            else if (c == ')') depth--;
            else if (c == '|' && depth == 0) return p;
        }
    }
    return "^" + (starts ? "" : any) + "(?:" + core + ")" + (ends ? "" : any) + "$";
}

// RADIANCE: \d \w \s and their negations, which GBNF cannot spell, written out as the classes
// ECMA-262 gives them, before the scanner reads the pattern. A JSON string holds text, not raw
// bytes: the negated forms leave out '"', '\' and the control characters, which a string can only
// carry escaped, and admit any JSON escape instead, so a match never runs past the closing quote;
// \s adds the escapes \t \n \f \r for the same reason. In a class a positive shorthand becomes its
// ranges and [\s\S] and its kin become any text; a negated one mixed into a class has no spelling,
// so false is returned and the pattern refused. Identity escapes of punctuation, which JavaScript
// writes freely ("\/" in a path, "\-" in a class), become characters GBNF can read. \t \n \r \f
// \b (in a class) \v \0 are control characters, which a JSON string carries only as escapes, so
// they match the escape. Any other escape in a class (\p{..}, \c, a backreference) has no GBNF
// spelling: `bad` names it and false is returned, rather than copying it into a grammar that
// would then fail to load.
static bool is_identity_punct(char c) {
    return c > ' ' && c < 0x7F && c != '\\' && !std::isalnum(static_cast<unsigned char>(c));
}

// The JSON escape for the control character a regex escape names, or "" if it names none.
static std::string json_control_escape(char n) {
    switch (n) {
        case 't': case 'n': case 'r': case 'f': case 'b': return std::string("\\\\") + n;
        case 'v': return "\\\\u000[bB]";
        case '0': return "\\\\u0000";
        default: return "";
    }
}

static std::string hex_escape(char n) {
    std::string s = "\\x";
    s += "0123456789ABCDEF"[(n >> 4) & 0xF];
    s += "0123456789ABCDEF"[n & 0xF];
    return s;
}

static bool expand_shorthands(const std::string & p, std::string & out, char & bad) {
    static const std::string sp = " \\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000\\uFEFF";
    static const std::string esc = "\\\\(?:[\"\\\\/bfnrt]|u[0-9A-Fa-f]{4})";
    static const std::string raw = "\"\\\\\\x00-\\x1F";
    auto negated = [&](const std::string & cls) { return "(?:[^" + cls + raw + "]|" + esc + ")"; };
    out.clear();
    for (size_t i = 0; i < p.size(); ++i) {
        const char c = p[i];
        if (c == '\\' && i + 1 < p.size()) {
            const char n = p[++i];
            switch (n) {
                case 'd': out += "[0-9]"; break;
                case 'w': out += "[A-Za-z0-9_]"; break;
                case 's': out += "(?:[" + sp + "]|\\\\[tnfr])"; break;
                case 'D': out += negated("0-9"); break;
                case 'W': out += negated("A-Za-z0-9_"); break;
                case 'S': out += negated(sp); break;
                default:
                    // An identity escape of punctuation GBNF has no escape for ("\/", "\-") is the
                    // bare character; regex syntax characters and '"' keep their backslash.
                    if (is_identity_punct(n) && std::string("^$.[](){}|*+?\"").find(n) == std::string::npos) {
                        out += n;
                    } else if (n != 'b' && !json_control_escape(n).empty()) {
                        // A group, so a following quantifier takes the whole escape. Outside a class
                        // \b is a word boundary, which the scanner refuses.
                        out += "(?:" + json_control_escape(n) + ")";
                    } else if (n == ' ' || static_cast<unsigned char>(n) >= 0x80) {
                        out += n;
                    } else {
                        out += c;
                        out += n;
                    }
                    break;
            }
            continue;
        }
        if (c != '[') {
            out += c;
            continue;
        }
        // The class runs to the first unescaped ']', as the scanner reads it.
        size_t j = i + 1;
        std::string body;
        while (j < p.size() && p[j] != ']') {
            if (p[j] == '\\' && j + 1 < p.size()) {
                body += p.substr(j, 2);
                j += 2;
            } else {
                body += p[j++];
            }
        }
        if (body == "\\s\\S" || body == "\\S\\s" || body == "\\d\\D" || body == "\\D\\d" ||
            body == "\\w\\W" || body == "\\W\\w") {
            out += "(?:[^" + raw + "]|" + esc + ")";
            i = j;
            continue;
        }
        std::string cls = "[";
        // A positive class's control characters can only appear as JSON escapes, which are two
        // or six characters long, so they become alternatives beside the class.
        std::vector<std::string> escapes;
        const bool negated_cls = !body.empty() && body[0] == '^';
        std::string excluded;  // the control characters a negated class leaves out, by escape letter
        for (size_t k = 0; k < body.size(); ++k) {
            if (body[k] == '\\' && k + 1 < body.size()) {
                const char n = body[++k];
                if (n == 'd') cls += "0-9";
                else if (n == 'w') cls += "A-Za-z0-9_";
                else if (n == 's') {
                    cls += sp;
                    if (!negated_cls) escapes.push_back("\\\\[tnfr]");
                    else excluded += "tnfr";
                }
                else if (n == 'D' || n == 'W' || n == 'S') { bad = n; return false; }
                else if (!json_control_escape(n).empty()) {
                    if (n == '0' && k + 1 < body.size() && std::isdigit(static_cast<unsigned char>(body[k + 1]))) {
                        bad = n;  // "\01" is a legacy octal escape
                        return false;
                    }
                    if (negated_cls) excluded += n;
                    else escapes.push_back(json_control_escape(n));
                }
                else if (is_identity_punct(n) && n != '"' && n != '[' && n != ']') {
                    // "\-" "\." "\^" and the like: GBNF has no such escape, and bare they could
                    // read as a range or a negation, so they are written as \xHH.
                    cls += hex_escape(n);
                }
                else if (n == 'x' || n == 'u') {
                    // Copied for GBNF to read, so the hex digits must be there.
                    const size_t len = n == 'x' ? 2 : 4;
                    for (size_t h = 1; h <= len; ++h) {
                        if (k + h >= body.size() || !std::isxdigit(static_cast<unsigned char>(body[k + h]))) {
                            bad = n;
                            return false;
                        }
                    }
                    cls += '\\';
                    cls += n;
                }
                else if (std::string("\"[]\\").find(n) != std::string::npos) { cls += '\\'; cls += n; }
                else if (n == ' ' || static_cast<unsigned char>(n) >= 0x80) cls += n;
                else { bad = n; return false; }
            } else {
                cls += body[k];
            }
        }
        if (j < p.size()) {
            cls += ']';
        }
        if (negated_cls && !excluded.empty() && j < p.size()) {
            // A string holds those characters only as escapes, so the class becomes text that is
            // neither a listed character nor one of those escapes. Their \u forms (\u0009 for \t)
            // are still admitted.
            std::string letters;
            for (char e : std::string("bfnrt")) {
                if (excluded.find(e) == std::string::npos) letters += e;
            }
            out += "(?:[^" + cls.substr(2, cls.size() - 3) + raw + "]|\\\\(?:[\"\\\\/" + letters +
                   "]|u[0-9A-Fa-f]{4}))";
        } else if (!escapes.empty() && j < p.size()) {
            out += "(?:";
            if (cls != "[]") out += cls + "|";
            for (size_t e = 0; e < escapes.size(); ++e) out += (e ? "|" : "") + escapes[e];
            out += ")";
        } else {
            out += cls;
        }
        i = j;
    }
    return true;
}

static std::unordered_set<char> ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = {'^', '$', '.', '[', ']', '(', ')', '|', '{', '}', '*', '+', '?'};

static std::string replacePattern(const std::string & input, const std::regex & regex, const std::function<std::string(const std::smatch  &)> & replacement) {
    std::smatch match;
    std::string result;

    std::string::const_iterator searchStart(input.cbegin());
    std::string::const_iterator searchEnd(input.cend());

    while (std::regex_search(searchStart, searchEnd, match, regex)) {
        result.append(searchStart, searchStart + match.position());
        result.append(replacement(match));
        searchStart = match.suffix().first;
    }

    result.append(searchStart, searchEnd);

    return result;
}

static std::string format_literal(const std::string & literal) {
    std::string escaped = replacePattern(literal, GRAMMAR_LITERAL_ESCAPE_RE, [&](const std::smatch & match) {
        char c = match.str()[0];
        return GRAMMAR_LITERAL_ESCAPES.at(c);
    });
    return "\"" + escaped + "\"";
}

std::string gbnf_format_literal(const std::string & literal) { return format_literal(literal); }

class common_schema_converter {
private:
    friend class common_schema_info;
    friend std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options);
    std::function<json(const std::string &)> _fetch_json;
    bool _dotall;
    std::map<std::string, std::string> _rules;
    std::unordered_map<std::string, json> _refs;
    std::unordered_set<std::string> _refs_being_resolved;
    std::vector<std::string> _errors;
    std::vector<std::string> _warnings;

    std::string _add_rule(const std::string & name, const std::string & rule) {
        std::string esc_name = regex_replace(name, INVALID_RULE_CHARS_RE, "-");
        if (_rules.find(esc_name) == _rules.end() || _rules[esc_name] == rule) {
            _rules[esc_name] = rule;
            return esc_name;
        }
        int i = 0;
        while (_rules.find(esc_name + std::to_string(i)) != _rules.end() && _rules[esc_name + std::to_string(i)] != rule) {
            i++;
        }
        std::string key = esc_name + std::to_string(i);
        _rules[key] = rule;
        return key;
    }

    std::string _generate_union_rule(const std::string & name, const std::vector<json> & alt_schemas) {
        std::vector<std::string> rules;
        rules.reserve(alt_schemas.size());
        for (size_t i = 0; i < alt_schemas.size(); i++) {
            rules.push_back(visit(alt_schemas[i], name + (name.empty() ? "alternative-" : "-") + std::to_string(i)));
        }
        return string_join(rules, " | ");
    }

    std::string _visit_pattern(const std::string & client_pattern, const std::string & name) {
        std::string expanded;
        char bad = 0;
        if (!expand_shorthands(client_pattern, expanded, bad)) {
            _errors.push_back(std::string("Unsupported regex escape '\\") + bad +
                              "' inside a character class: GBNF cannot spell it");
            return "";
        }
        const std::string pattern = anchor_pattern(expanded);
        // RADIANCE: front()/back() on an empty string is UB, and `pattern` is client data straight
        // off the wire. The size test has to come first.
        if (pattern.size() < 2 || !(pattern.front() == '^' && pattern.back() == '$')) {
            _errors.push_back("Pattern must start with '^' and end with '$'");
            return "";
        }
        std::string sub_pattern = pattern.substr(1, pattern.length() - 2);
        std::unordered_map<std::string, std::string> sub_rule_ids;

        size_t i = 0;
        size_t length = sub_pattern.length();

        using literal_or_rule = std::pair<std::string, bool>;
        auto to_rule = [&](const literal_or_rule & ls) {
            auto is_literal = ls.second;
            auto s = ls.first;
            return is_literal ? "\"" + s + "\"" : s;
        };
        std::function<literal_or_rule()> transform = [&]() -> literal_or_rule {
            size_t start = i;
            std::vector<literal_or_rule> seq;

            auto get_dot = [&]() {
                std::string rule;
                if (_dotall) {
                    rule = "[\\U00000000-\\U0010FFFF]";
                } else {
                    rule = "[^\\x0A\\x0D]";
                }
                return _add_rule("dot", rule);
            };

            // Joins the sequence, merging consecutive literals together.
            auto join_seq = [&]() {
                std::vector<literal_or_rule> ret;

                std::string literal;
                auto flush_literal = [&]() {
                    if (literal.empty()) {
                        return false;
                    }
                    ret.emplace_back(literal, true);
                    literal.clear();
                    return true;
                };

                for (const auto & item : seq) {
                    auto is_literal = item.second;
                    if (is_literal) {
                        literal += item.first;
                    } else {
                        flush_literal();
                        ret.push_back(item);
                    }
                }
                flush_literal();

                std::vector<std::string> results;
                results.reserve(ret.size());
                for (const auto & item : ret) {
                    results.push_back(to_rule(item));
                }
                return std::make_pair(string_join(results, " "), false);
            };

            while (i < length) {
                // RADIANCE: every arm below must consume at least one byte. The literal arm does not when
                // it meets a byte that is in NON_LITERAL_SET but that no arm above claims -- a
                // stray ']' or '}' -- so it breaks out with `literal` empty, nothing is pushed,
                // and the loop reconsiders the same byte forever. `pattern` is client data, so
                // that is a one-string denial of service. Checked at the bottom of the body.
                const size_t progress_at = i;
                char c = sub_pattern[i];
                if (c == '.') {
                    seq.emplace_back(get_dot(), false);
                    i++;
                } else if (c == '(') {
                    i++;
                    if (i < length && sub_pattern[i] == '?') {
                        if (i + 1 < length && sub_pattern[i + 1] == ':') {
                            i += 2; // skip "?:" for non-capturing group, treat as regular group
                        } else {
                            // lookahead/lookbehind (?=, ?!, ?<=, ?<!) - not supported
                            _warnings.push_back("Unsupported pattern syntax");
                            // skip to matching ')' to avoid UB on empty seq
                            int depth = 1;
                            while (i < length && depth > 0) {
                                if (sub_pattern[i] == '\\' && i + 1 < length) {
                                    i += 2; // skip escaped character
                                } else {
                                    if (sub_pattern[i] == '(') depth++;
                                    else if (sub_pattern[i] == ')') depth--;
                                    i++;
                                }
                            }
                            continue;
                        }
                    }
                    seq.emplace_back("(" + to_rule(transform()) + ")", false);
                } else if (c == ')') {
                    i++;
                    if (start > 0 && sub_pattern[start - 1] != '(' && (start < 2 || sub_pattern[start - 2] != '?' || sub_pattern[start - 1] != ':')) {
                        _errors.push_back("Unbalanced parentheses");
                    }
                    return join_seq();
                } else if (c == '[') {
                    std::string square_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != ']') {
                        if (sub_pattern[i] == '\\') {
                            square_brackets += sub_pattern.substr(i, 2);
                            i += 2;
                        } else {
                            square_brackets += sub_pattern[i];
                            i++;
                        }
                    }
                    if (i >= length) {
                        _errors.push_back("Unbalanced square brackets");
                    }
                    square_brackets += ']';
                    i++;
                    seq.emplace_back(square_brackets, false);
                } else if (c == '|') {
                    seq.emplace_back("|", false);
                    i++;
                } else if (c == '*' || c == '+' || c == '?') {
                    if (seq.empty()) {
                        // RADIANCE: a leading quantifier, "^*$". back() on an empty vector is UB.
                        _errors.push_back("Quantifier with nothing to repeat");
                        break;
                    }
                    seq.back() = std::make_pair(to_rule(seq.back()) + c, false);
                    i++;
                } else if (c == '{') {
                    std::string curly_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != '}') {
                        curly_brackets += sub_pattern[i];
                        i++;
                    }
                    if (i >= length) {
                        _errors.push_back("Unbalanced curly brackets");
                    }
                    curly_brackets += '}';
                    i++;
                    auto nums = string_split(curly_brackets.substr(1, curly_brackets.length() - 2), ",");
                    int min_times = 0;
                    int max_times = std::numeric_limits<int>::max();
                    try {
                        if (nums.size() == 1) {
                            min_times = max_times = std::stoi(nums[0]);
                        } else if (nums.size() != 2) {
                            _errors.push_back("Wrong number of values in curly brackets");
                        } else {
                            if (!nums[0].empty()) {
                                min_times = std::stoi(nums[0]);
                            }
                            if (!nums[1].empty()) {
                                max_times = std::stoi(nums[1]);
                            }
                        }
                    } catch (const std::invalid_argument & e) {
                        _errors.push_back("Invalid number in curly brackets");
                        return std::make_pair("", false);
                    }
                    if (seq.empty()) {
                        // RADIANCE: a leading "{n,m}" has nothing before it to repeat; back() would be UB.
                        _errors.push_back("Quantifier with nothing to repeat");
                        break;
                    }
                    auto &last = seq.back();
                    auto &sub = last.first;
                    auto sub_is_literal = last.second;

                    if (!sub_is_literal) {
                        std::string & sub_id = sub_rule_ids[sub];
                        if (sub_id.empty()) {
                            sub_id = _add_rule(name + "-" + std::to_string(sub_rule_ids.size()), sub);
                        }
                        sub = sub_id;
                    }
                    seq.back().first = build_repetition(
                        sub_is_literal ? "\"" + sub + "\"" : sub,
                        min_times,
                        max_times,
                        ""
                    );
                    seq.back().second = false;
                } else {
                    std::string literal;
                    auto is_non_literal = [&](char c) {
                        return NON_LITERAL_SET.find(c) != NON_LITERAL_SET.end();
                    };
                    while (i < length) {
                        if (sub_pattern[i] == '\\' && i < length - 1) {
                            char next = sub_pattern[i + 1];
                            if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.find(next) != ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.end()) {
                                i++;
                                literal += sub_pattern[i];
                                i++;
                            } else {
                                // RADIANCE: GBNF knows \x \u \U \t \r \n \\ \" \[ \] and nothing
                                // else. Copying an escape it cannot spell (\d \w \s \b \/ \-, a
                                // backreference) produces a grammar that returns success here and
                                // then fails to LOAD, which the caller sees as an internal error
                                // on a schema we told it was fine.
                                if (!is_gbnf_escape(next)) {
                                    _errors.push_back(
                                        std::string("Unsupported regex escape '\\") + next +
                                        "' in a pattern: GBNF cannot spell it, so the grammar "
                                        "would not load");
                                    return std::make_pair(std::string(), true);
                                }
                                literal += sub_pattern.substr(i, 2);
                                i += 2;
                            }
                        } else if (sub_pattern[i] == '"') {
                            literal += "\\\"";
                            i++;
                        } else if (!is_non_literal(sub_pattern[i]) &&
                                (i == length - 1 || literal.empty() || sub_pattern[i + 1] == '.' || !is_non_literal(sub_pattern[i + 1]))) {
                            literal += sub_pattern[i];
                            i++;
                        } else {
                            break;
                        }
                    }
                    if (!literal.empty()) {
                        seq.emplace_back(literal, true);
                    }
                }
                if (i == progress_at) {
                    _errors.push_back("Unsupported pattern syntax");
                    break;
                }
            }
            return join_seq();
        };
        return _add_rule(name, "\"\\\"\" (" + to_rule(transform()) + ") \"\\\"\" space");
    }

    /*
        Returns a rule that matches a JSON string that is none of the provided strings

        not_strings({"a"})
            -> ["] ( [a] char+ | [^"a] char* )? ["] space
        not_strings({"and", "also"})
            -> ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["] space
    */
    std::string _not_strings(const std::vector<std::string> & strings) {

        struct TrieNode {
            std::map<char, TrieNode> children;
            bool is_end_of_string;

            TrieNode() : is_end_of_string(false) {}

            void insert(const std::string & string) {
                auto *node = this;
                for (char c : string) {
                    node = &node->children[c];
                }
                node->is_end_of_string = true;
            }
        };

        TrieNode trie;
        for (const auto & s : strings) {
            trie.insert(s);
        }

        std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
        std::ostringstream out;
        out << "[\"] ( ";
        std::function<void(const TrieNode &)> visit = [&](const TrieNode & node) {
            std::ostringstream rejects;
            auto first = true;
            for (const auto & kv : node.children) {
                rejects << kv.first;
                if (first) {
                    first = false;
                } else {
                    out << " | ";
                }
                out << "[" << kv.first << "]";
                if (!kv.second.children.empty()) {
                    out << " (";
                    visit(kv.second);
                    out << ")";
                } else if (kv.second.is_end_of_string) {
                    out << " " << char_rule << "+";
                }
            }
            if (!node.children.empty()) {
                if (!first) {
                    out << " | ";
                }
                out << "[^\"" << rejects.str() << "] " << char_rule << "*";
            }
        };
        visit(trie);

        out << " )";
        if (!trie.is_end_of_string) {
            out << "?";
        }
        out << " [\"] space";
        return out.str();
    }

    std::string _resolve_ref(const std::string & ref) {
        auto it = ref.find('#');
        std::string ref_fragment = it != std::string::npos ? ref.substr(it + 1) : ref;
        static const std::regex nonalphanumeric_regex(R"([^a-zA-Z0-9-]+)");
        std::string ref_name = "ref" + std::regex_replace(ref_fragment, nonalphanumeric_regex, "-");
        if (_rules.find(ref_name) == _rules.end() && _refs_being_resolved.find(ref) == _refs_being_resolved.end()) {
            _refs_being_resolved.insert(ref);
            json resolved = _refs[ref];
            ref_name = visit(resolved, ref_name);
            _refs_being_resolved.erase(ref);
        }
        return ref_name;
    }

    std::string _build_object_rule(
        const std::vector<std::pair<std::string, json>> & properties,
        const std::unordered_set<std::string> & required,
        const std::string & name,
        const json & additional_properties)
    {
        std::vector<std::string> required_props;
        std::vector<std::string> optional_props;
        std::unordered_map<std::string, std::string> prop_kv_rule_names;
        std::vector<std::string> prop_names;
        for (const auto & kv : properties) {
            const auto &prop_name = kv.first;
            const auto &prop_schema = kv.second;

            // RADIANCE: a property no value satisfies is a key the object never carries, which the
            // grammar states exactly by having no rule for it. Its name stays in prop_names, so an
            // additionalProperties rule cannot readmit the key with a value of its own. Required,
            // it makes the whole object unsatisfiable, and that is refused.
            if (is_unsatisfiable(prop_schema)) {
                if (required.find(prop_name) != required.end()) {
                    _errors.push_back("Property \"" + prop_name + "\" is required, but its schema "
                                      "accepts no value");
                }
                prop_names.push_back(prop_name);
                continue;
            }

            std::string prop_rule_name = visit(prop_schema, name + (name.empty() ? "" : "-") + prop_name);
            prop_kv_rule_names[prop_name] = _add_rule(
                name + (name.empty() ? "" : "-") + prop_name + "-kv",
                format_literal(json(prop_name).dump()) + " space \":\" space " + prop_rule_name
            );
            if (required.find(prop_name) != required.end()) {
                required_props.push_back(prop_name);
            } else {
                optional_props.push_back(prop_name);
            }
            prop_names.push_back(prop_name);
        }
        if ((additional_properties.is_boolean() && additional_properties.get<bool>()) || additional_properties.is_object()) {
            std::string sub_name = name + (name.empty() ? "" : "-") + "additional";
            std::string value_rule =
                additional_properties.is_object() ? visit(additional_properties, sub_name + "-value")
                : _add_primitive("value", PRIMITIVE_RULES.at("value"));

            auto key_rule =
                prop_names.empty() ? _add_primitive("string", PRIMITIVE_RULES.at("string"))
                : _add_rule(sub_name + "-k", _not_strings(prop_names));
            std::string kv_rule = _add_rule(sub_name + "-kv", key_rule + " \":\" space " + value_rule);
            prop_kv_rule_names["*"] = kv_rule;
            optional_props.push_back("*");
        }

        std::string rule = "\"{\" space ";
        for (size_t i = 0; i < required_props.size(); i++) {
            if (i > 0) {
                rule += " \",\" space ";
            }
            rule += prop_kv_rule_names[required_props[i]];
        }

        if (!optional_props.empty()) {
            rule += " (";
            if (!required_props.empty()) {
                rule += " \",\" space ( ";
            }

            std::function<std::string(const std::vector<std::string> &, bool)> get_recursive_refs = [&](const std::vector<std::string> & ks, bool first_is_optional) {
                std::string res;
                if (ks.empty()) {
                    return res;
                }
                const std::string& k = ks[0];
                std::string kv_rule_name = prop_kv_rule_names[k];
                std::string comma_ref = "( \",\" space " + kv_rule_name + " )";
                if (first_is_optional) {
                    res = comma_ref + (k == "*" ? "*" : "?");
                } else {
                    res = kv_rule_name + (k == "*" ? " " + comma_ref + "*" : "");
                }
                if (ks.size() > 1) {
                    res += " " + _add_rule(
                        name + (name.empty() ? "" : "-") + k + "-rest",
                        get_recursive_refs(std::vector<std::string>(ks.begin() + 1, ks.end()), true)
                    );
                }
                return res;
            };

            for (size_t i = 0; i < optional_props.size(); i++) {
                if (i > 0) {
                    rule += " | ";
                }
                rule += get_recursive_refs(std::vector<std::string>(optional_props.begin() + i, optional_props.end()), false);
            }
            if (!required_props.empty()) {
                rule += " )";
            }
            rule += " )?";
        }

        rule += " \"}\" space";

        return rule;
    }

    std::string _add_primitive(const std::string & name, const BuiltinRule & rule) {
        auto n = _add_rule(name, rule.content);
        for (const auto & dep : rule.deps) {
            BuiltinRule dep_rule;
            auto it = PRIMITIVE_RULES.find(dep);
            if (it == PRIMITIVE_RULES.end()) {
                it = STRING_FORMAT_RULES.find(dep);
                if (it == STRING_FORMAT_RULES.end()) {
                    _errors.push_back("Rule " + dep + " not known");
                    continue;
                }
            }
            if (_rules.find(dep) == _rules.end()) {
                _add_primitive(dep, it->second);
            }
        }
        return n;
    }

public:
    common_schema_converter(
        const std::function<json(const std::string &)> & fetch_json,
        bool dotall)
          : _fetch_json(fetch_json), _dotall(dotall)
    {
        _rules["space"] = SPACE_RULE;
    }

    void resolve_refs(json & schema, const std::string & url) {
        /*
        * Resolves all $ref fields in the given schema, fetching any remote schemas,
        * replacing each $ref with absolute reference URL and populates _refs with the
        * respective referenced (sub)schema dictionaries.
        */
        std::function<void(json &)> visit_refs = [&](json & n) {
            if (n.is_array()) {
                for (auto & x : n) {
                    visit_refs(x);
                }
            } else if (n.is_object()) {
                if (n.contains("$ref")) {
                    std::string ref = n["$ref"];
                    // RADIANCE: a sampler that reaches the network to decide what tokens are legal
                    // is not a sampler we will ship. Upstream fetches through an injected callback
                    // which the public entry point then stubs out to return null -- so a remote
                    // $ref silently degrades into "accepts any object". Refuse, and fetch nothing.
                    if (ref.rfind("http://", 0) == 0 || ref.rfind("https://", 0) == 0) {
                        throw std::runtime_error(
                            "remote $ref is not supported: '" + ref + "'. Inline the referenced "
                            "schema; this converter does not fetch over the network.");
                    }
                    if (_refs.find(ref) == _refs.end()) {
                        json target;
                        if (ref.find("https://") == 0) {
                            std::string base_url = ref.substr(0, ref.find('#'));
                            auto it = _refs.find(base_url);
                            if (it != _refs.end()) {
                                target = it->second;
                            } else {
                                // Fetch the referenced schema and resolve its refs
                                auto referenced = _fetch_json(ref);
                                resolve_refs(referenced, base_url);
                                _refs[base_url] = referenced;
                            }
                            if (ref.find('#') == std::string::npos || ref.substr(ref.find('#') + 1).empty()) {
                                return;
                            }
                        } else if (ref.find("#/") == 0) {
                            target = schema;
                            n["$ref"] = url + ref;
                            ref = url + ref;
                        } else {
                            _errors.push_back("Unsupported ref: " + ref);
                            return;
                        }
                        std::string pointer = ref.substr(ref.find('#') + 1);
                        std::vector<std::string> tokens = string_split(pointer, "/");
                        for (size_t i = 1; i < tokens.size(); ++i) {
                            const std::string& sel = tokens[i];
                            if (target.is_object() && target.contains(sel)) {
                                target = target[sel];
                            } else if (target.is_array()) {
                                size_t sel_index;
                                try {
                                    sel_index = std::stoull(sel);
                                } catch (const std::invalid_argument & e) {
                                    sel_index = target.size();
                                }
                                if (sel_index >= target.size()) {
                                    _errors.push_back("Error resolving ref " + ref + ": " + sel + " not in " + target.dump());
                                    return;
                                }
                                target = target[sel_index];
                            } else {
                                _errors.push_back("Error resolving ref " + ref + ": " + sel + " not in " + target.dump());
                                return;
                            }
                        }
                        _refs[ref] = target;
                    }
                } else {
                    for (const auto & kv : n.items()) {
                        visit_refs(kv.value());
                    }
                }
            }
        };

        visit_refs(schema);
    }

    // RADIANCE: a count -- minLength, maxLength, minItems, maxItems -- as the int build_repetition
    // takes, or `dflt` where the schema has none. get<int>() casts, so a bound past INT_MAX wrapped:
    // Number.MAX_SAFE_INTEGER, a common way to write "no limit", came out as -1 and the grammar did
    // not parse. A count at or past INT_MAX is INT_MAX, which build_repetition reads as no maximum;
    // no context the engine serves holds a string or an array that long, so the bound changes
    // nothing a generation can write. A count that is not a non-negative number is refused.
    int _schema_count(const json & schema, const char * key, int dflt, const std::string & name) {
        if (!schema.contains(key)) {
            return dflt;
        }
        const json & v = schema[key];
        const double d = v.is_number() ? v.get<double>() : -1.0;
        if (!(d >= 0)) {
            _errors.push_back(std::string(key) + " of \"" + (name.empty() ? std::string("root") : name) +
                              "\" must be a non-negative number");
            return dflt;
        }
        return d >= (double) std::numeric_limits<int>::max() ? std::numeric_limits<int>::max() : (int) d;
    }

    // RADIANCE: the schema every value satisfies, whose `not` is therefore the schema none does.
    // `{}` and `true` are it, and so is a type list, or a union of bare types, that between them
    // name all six JSON types -- the way Arktype exports the complement of `never` for an optional
    // property: {"not": {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"},
    // {"type": "object"}, {"type": "array"}, {"type": "null"}]}}. Annotations may sit beside; any
    // other keyword narrows the schema, and its `not` is a real complement, refused below. A union
    // alternative that constrains more than its type is not counted, so a union is only ever read
    // as universal when it is.
    static bool is_universal(const json & s) {
        if (s.is_boolean()) return s.get<bool>();
        if (!s.is_object()) return false;
        static const std::unordered_set<std::string> annotations = {
            "description", "title", "default", "examples", "$comment", "deprecated", "readOnly",
            "writeOnly",
        };
        // Whether every key of `o` is an annotation or `allowed`.
        auto only = [&](const json & o, const char * allowed) {
            for (auto it = o.begin(); it != o.end(); ++it) {
                if (annotations.count(it.key()) == 0 && (allowed == nullptr || it.key() != allowed)) {
                    return false;
                }
            }
            return true;
        };
        // The types a bare `type` names, into `out`; false for anything that is not one.
        auto types_of = [](const json & t, std::unordered_set<std::string> & out) {
            if (t.is_string()) {
                out.insert(t.get<std::string>());
                return true;
            }
            if (!t.is_array()) return false;
            for (const auto & e : t) {
                if (!e.is_string()) return false;
                out.insert(e.get<std::string>());
            }
            return true;
        };
        auto covers = [](const std::unordered_set<std::string> & ts) {
            for (const char * t : { "string", "number", "boolean", "object", "array", "null" }) {
                if (ts.count(t) == 0) return false;
            }
            return true;
        };
        if (only(s, nullptr)) return true;
        std::unordered_set<std::string> ts;
        if (s.contains("type") && only(s, "type")) {
            return types_of(s["type"], ts) && covers(ts);
        }
        if (s.contains("anyOf") && only(s, "anyOf") && s["anyOf"].is_array()) {
            for (const auto & a : s["anyOf"]) {
                if (is_universal(a)) return true;
                if (a.is_object() && a.contains("type") && only(a, "type")) {
                    types_of(a["type"], ts);
                }
            }
            return covers(ts);
        }
        return false;
    }

    // RADIANCE: the schema no value satisfies. JSON Schema spells it `false`, and TypeBox's
    // Type.Never and Arktype's "never" export it as a `not` of the universal schema above, the
    // latter for an optional property no call may set. Such a `not` makes its whole schema
    // unsatisfiable whatever sits beside it, so the sibling keywords do not matter.
    static bool is_unsatisfiable(const json & schema) {
        if (schema.is_boolean()) return !schema.get<bool>();
        if (!schema.is_object() || !schema.contains("not")) return false;
        return is_universal(schema["not"]);
    }

    static std::string _generate_constant_rule(const json & value) {
        return format_literal(value.dump());
    }

    std::string visit(const json & schema, const std::string & name) {
        json schema_type = schema.contains("type") ? schema["type"] : json();
        std::string schema_format = schema.contains("format") ? schema["format"].get<std::string>() : "";
        std::string rule_name = is_reserved_name(name) ? name + "-" : name.empty() ? "root" : name;

        // RADIANCE: an unsatisfiable schema in a place that must hold a value -- the root, an array's
        // items, a $ref target -- has no grammar: GBNF has no empty language to give it. An object
        // property and a union alternative of this kind are handled where they occur.
        if (is_unsatisfiable(schema)) {
            _errors.push_back("Schema \"" + (name.empty() ? std::string("root") : name) + "\" accepts no "
                              "value, where a value is required");
            return "";
        }

        // RADIANCE: any other "not" has no case below, so it would fall through to a rule that
        // accepts anything -- a constraint the caller asked for, did not get, and is not told
        // about. Complementing a regular language is representable but the blowup is real;
        // refusing beats pretending.
        if (schema.is_object() && schema.contains("not")) {
            _errors.push_back("Unsupported schema keyword \"not\": this converter cannot "
                              "complement a grammar, and accepting the schema without it would "
                              "silently drop the constraint");
            return "";
        }

        if (schema.contains("$ref")) {
            return _add_rule(rule_name, _resolve_ref(schema["$ref"]));
        }
        if (schema.contains("oneOf") || schema.contains("anyOf")) {
            std::vector<json> alt_schemas = schema.contains("oneOf") ? schema["oneOf"].get<std::vector<json>>() : schema["anyOf"].get<std::vector<json>>();
            // RADIANCE: an alternative no value satisfies contributes nothing to the union.
            alt_schemas.erase(std::remove_if(alt_schemas.begin(), alt_schemas.end(), is_unsatisfiable),
                              alt_schemas.end());
            if (alt_schemas.empty()) {
                _errors.push_back("Schema \"" + (name.empty() ? std::string("root") : name) + "\": no "
                                  "alternative of its union accepts a value");
                return "";
            }
            return _add_rule(rule_name, _generate_union_rule(name, alt_schemas));
        }
        if (schema_type.is_array()) {
            std::vector<json> schema_types;
            for (const auto & t : schema_type) {
                json schema_copy(schema);
                schema_copy["type"] = t;
                schema_types.push_back(schema_copy);
            }
            return _add_rule(rule_name, _generate_union_rule(name, schema_types));
        }
        if (schema.contains("const")) {
            return _add_rule(rule_name, _generate_constant_rule(schema["const"]) + " space");
        }
        if (schema.contains("enum")) {
            std::vector<std::string> enum_values;
            for (const auto & v : schema["enum"]) {
                enum_values.push_back(_generate_constant_rule(v));
            }
            return _add_rule(rule_name, "(" + string_join(enum_values, " | ") + ") space");
        }
        if ((schema_type.is_null() || schema_type == "object")
                && (schema.contains("properties") ||
                    (schema.contains("additionalProperties") && schema["additionalProperties"] != true))) {
            std::unordered_set<std::string> required;
            if (schema.contains("required") && schema["required"].is_array()) {
                for (const auto & item : schema["required"]) {
                    if (item.is_string()) {
                        required.insert(item.get<std::string>());
                    }
                }
            }
            std::vector<std::pair<std::string, json>> properties;
            if (schema.contains("properties")) {
                for (const auto & prop : schema["properties"].items()) {
                    properties.emplace_back(prop.key(), prop.value());
                }
            }
            return _add_rule(rule_name,
                _build_object_rule(
                    properties, required, name,
                    schema.contains("additionalProperties") ? schema["additionalProperties"] : json()));
        }
        if ((schema_type.is_null() || schema_type == "object" || schema_type == "string") && schema.contains("allOf")) {
            std::unordered_set<std::string> required;
            std::vector<std::pair<std::string, json>> properties;
            std::map<std::string, size_t> enum_values;
            const std::string& hybrid_name = name;
            std::function<void(const json &, bool)> add_component = [&](const json & comp_schema, bool is_required) {
                if (comp_schema.contains("$ref")) {
                    add_component(_refs[comp_schema["$ref"]], is_required);
                } else if (comp_schema.contains("properties")) {
                    for (const auto & prop : comp_schema["properties"].items()) {
                        properties.emplace_back(prop.key(), prop.value());
                        if (is_required) {
                            required.insert(prop.key());
                        }
                    }
                } else if (comp_schema.contains("enum")) {
                    for (const auto & v : comp_schema["enum"]) {
                        const auto rule = _generate_constant_rule(v);
                        if (enum_values.find(rule) == enum_values.end()) {
                            enum_values[rule] = 0;
                        }
                        enum_values[rule] += 1;
                    }
                } else {
                  // todo warning
                }
            };
            for (const auto & t : schema["allOf"]) {
                if (t.contains("anyOf")) {
                    for (const auto & tt : t["anyOf"]) {
                        add_component(tt, false);
                    }
                } else {
                    add_component(t, true);
                }
            }
            if (!enum_values.empty()) {
                std::vector<std::string> enum_intersection;
                for (const auto & p : enum_values) {
                    if (p.second == schema["allOf"].size()) {
                        enum_intersection.push_back(p.first);
                    }
                }
                if (!enum_intersection.empty()) {
                    return _add_rule(rule_name, "(" + string_join(enum_intersection, " | ") + ") space");
                }
            }
            return _add_rule(rule_name, _build_object_rule(properties, required, hybrid_name, json()));
        }
        if ((schema_type.is_null() || schema_type == "array") && (schema.contains("items") || schema.contains("prefixItems"))) {
            json items = schema.contains("items") ? schema["items"] : schema["prefixItems"];
            if (items.is_array()) {
                std::string rule = "\"[\" space ";
                for (size_t i = 0; i < items.size(); i++) {
                    if (i > 0) {
                        rule += " \",\" space ";
                    }
                    rule += visit(items[i], name + (name.empty() ? "" : "-") + "tuple-" + std::to_string(i));
                }
                rule += " \"]\" space";
                return _add_rule(rule_name, rule);
            }
            std::string item_rule_name = visit(items, name + (name.empty() ? "" : "-") + "item");
            int min_items = _schema_count(schema, "minItems", 0, name);
            int max_items = _schema_count(schema, "maxItems", std::numeric_limits<int>::max(), name);

            return _add_rule(rule_name, "\"[\" space " + build_repetition(item_rule_name, min_items, max_items, "\",\" space") + " \"]\" space");
        }
        if ((schema_type.is_null() || schema_type == "string") && schema.contains("pattern")) {
            return _visit_pattern(schema["pattern"], rule_name);
        }
        if ((schema_type.is_null() || schema_type == "string") && std::regex_match(schema_format, std::regex("^uuid[1-5]?$"))) {
            return _add_primitive(rule_name == "root" ? "root" : schema_format, PRIMITIVE_RULES.at("uuid"));
        }
        if ((schema_type.is_null() || schema_type == "string") && STRING_FORMAT_RULES.find(schema_format + "-string") != STRING_FORMAT_RULES.end()) {
            auto prim_name = schema_format + "-string";
            return _add_rule(rule_name, _add_primitive(prim_name, STRING_FORMAT_RULES.at(prim_name)));
        }
        if (schema_type == "string" && (schema.contains("minLength") || schema.contains("maxLength"))) {
            std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
            int min_len = _schema_count(schema, "minLength", 0, name);
            int max_len = _schema_count(schema, "maxLength", std::numeric_limits<int>::max(), name);
            return _add_rule(rule_name, "\"\\\"\" " + build_repetition(char_rule, min_len, max_len) + " \"\\\"\" space");
        }
        if (schema_type == "integer" && (schema.contains("minimum") || schema.contains("exclusiveMinimum") || schema.contains("maximum") || schema.contains("exclusiveMaximum"))) {
            int64_t min_value = std::numeric_limits<int64_t>::min();
            int64_t max_value = std::numeric_limits<int64_t>::max();
            if (schema.contains("minimum")) {
                min_value = schema["minimum"].get<int64_t>();
            } else if (schema.contains("exclusiveMinimum")) {
                min_value = schema["exclusiveMinimum"].get<int64_t>() + 1;
            }
            if (schema.contains("maximum")) {
                max_value = schema["maximum"].get<int64_t>();
            } else if (schema.contains("exclusiveMaximum")) {
                max_value = schema["exclusiveMaximum"].get<int64_t>() - 1;
            }
            std::stringstream out;
            out << "(";
            build_min_max_int(min_value, max_value, out);
            out << ") space";
            return _add_rule(rule_name, out.str());
        }
        if (schema.empty() || schema_type == "object") {
            return _add_rule(rule_name, _add_primitive("object", PRIMITIVE_RULES.at("object")));
        }
        if (schema_type.is_null() && schema.is_object()) {
            // No type constraint and no recognized structural keywords (e.g. {"description": "..."}).
            // Per JSON Schema semantics this is equivalent to {} and accepts any value.
            return _add_rule(rule_name, _add_primitive("value", PRIMITIVE_RULES.at("value")));
        }
        if (!schema_type.is_string() || PRIMITIVE_RULES.find(schema_type.get<std::string>()) == PRIMITIVE_RULES.end()) {
            _errors.push_back("Unrecognized schema: " + schema.dump());
            return "";
        }
        // TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at least for zero
        return _add_primitive(rule_name == "root" ? "root" : schema_type.get<std::string>(), PRIMITIVE_RULES.at(schema_type.get<std::string>()));
    }

    void check_errors() {
        if (!_errors.empty()) {
            throw std::invalid_argument("JSON schema conversion failed:\n" + string_join(_errors, "\n"));
        }
        if (!_warnings.empty()) {
            fprintf(stderr, "WARNING: JSON schema conversion was incomplete: %s\n", string_join(_warnings, "; ").c_str());
        }
    }

    std::string format_grammar() {
        std::stringstream ss;
        for (const auto & kv : _rules) {
            ss << kv.first << " ::= " << kv.second << '\n';
        }
        return ss.str();
    }
};

// common_schema_info implementation (pimpl)

common_schema_info::common_schema_info()
    : impl_(std::make_unique<common_schema_converter>(
        [](const std::string &) { return json(); },
        false)) {}

common_schema_info::~common_schema_info() = default;

common_schema_info::common_schema_info(common_schema_info &&) noexcept = default;
common_schema_info & common_schema_info::operator=(common_schema_info &&) noexcept = default;

void common_schema_info::resolve_refs(rad::Json & schema) {
    impl_->resolve_refs(schema, "");
}

bool common_schema_info::unsatisfiable(const rad::Json & schema) const {
    return common_schema_converter::is_unsatisfiable(schema);
}

// Determines if a JSON schema can resolve to a string type through any path.
// Some models emit raw string values rather than JSON-encoded strings for string parameters.
// If any branch of the schema (via oneOf, anyOf, $ref, etc.) permits a string, this returns
// true, allowing callers to handle the value as a raw string for simplicity.
bool common_schema_info::resolves_to_string(const rad::Json & schema) {
    std::unordered_set<std::string> visited_refs;

    std::function<bool(const json &)> check = [&](const json & s) -> bool {
        if (!s.is_object()) {
            return false;
        }

        // Handle $ref
        if (s.contains("$ref")) {
            const std::string & ref = s["$ref"];
            if (visited_refs.find(ref) != visited_refs.end()) {
                // Circular reference, assume not a string to be safe
                return false;
            }
            visited_refs.insert(ref);
            auto it = impl_->_refs.find(ref);
            if (it != impl_->_refs.end()) {
                return check(it->second);
            }
            return false;
        }

        // Check type field
        if (s.contains("type")) {
            const json & schema_type = s["type"];
            if (schema_type.is_string()) {
                if (schema_type == "string") {
                    return true;
                }
            } else if (schema_type.is_array()) {
                // Type can be an array like ["string", "null"]
                for (const auto & t : schema_type) {
                    if (t == "string") {
                        return true;
                    }
                }
            }
        }

        // Check oneOf/anyOf - if any alternative can be a string
        if (s.contains("oneOf")) {
            for (const auto & alt : s["oneOf"]) {
                if (check(alt)) {
                    return true;
                }
            }
        }
        if (s.contains("anyOf")) {
            for (const auto & alt : s["anyOf"]) {
                if (check(alt)) {
                    return true;
                }
            }
        }

        // Check allOf - all components must be compatible with string type
        if (s.contains("allOf")) {
            bool all_string = true;
            for (const auto & component : s["allOf"]) {
                if (!check(component)) {
                    all_string = false;
                    break;
                }
            }
            if (all_string) {
                return true;
            }
        }

        // Check const - if the constant value is a string
        if (s.contains("const")) {
            if (s["const"].is_string()) {
                return true;
            }
        }

        // Check enum - if any enum value is a string
        if (s.contains("enum")) {
            for (const auto & val : s["enum"]) {
                if (val.is_string()) {
                    return true;
                }
            }
        }

        // String-specific keywords imply string type
        if (s.contains("pattern") || s.contains("minLength") || s.contains("maxLength")) {
            return true;
        }

        // Check format - many formats imply string
        if (s.contains("format")) {
            const std::string & fmt = s["format"];
            if (fmt == "date" || fmt == "time" || fmt == "date-time" ||
                fmt == "uri" || fmt == "email" || fmt == "hostname" ||
                fmt == "ipv4" || fmt == "ipv6" || fmt == "uuid" ||
                fmt.find("uuid") == 0) {
                return true;
            }
        }

        return false;
    };

    return check(schema);
}

std::string json_schema_to_grammar(const json & schema, bool force_gbnf) {
#ifdef LLAMA_USE_LLGUIDANCE
    if (!force_gbnf) {
        return "%llguidance {}\nstart: %json " + schema.dump();
    }
#else
    (void)force_gbnf;
#endif // LLAMA_USE_LLGUIDANCE
    return build_grammar([&](const common_grammar_builder & callbacks) {
        auto copy = schema;
        callbacks.resolve_refs(copy);
        callbacks.add_schema("", copy);
    });
}

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options) {
    common_schema_converter converter([&](const std::string &) { return json(); }, options.dotall);
    common_grammar_builder builder {
        /* .add_rule = */ [&](const std::string & name, const std::string & rule) {
            return converter._add_rule(name, rule);
        },
        /* .add_schema = */ [&](const std::string & name, const rad::Json & schema) {
            return converter.visit(schema, name == "root" ? "" : name);
        },
        /* .resolve_refs = */ [&](rad::Json & schema) {
            converter.resolve_refs(schema, "");
        }
    };
    cb(builder);
    converter.check_errors();
    return converter.format_grammar();
}
