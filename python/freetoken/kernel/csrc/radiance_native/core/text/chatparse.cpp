#include "text/chatparse.h"
#include "text/chatfmt.h"

/* Bind time only: the schema resolution that decides which arguments are strings. The grammar
 * generator uses the same function, so parser and grammar agree on every type. Nothing from
 * vendor/ runs while a reply is read. */
#include "llama/common/json-schema-to-grammar.h"

#include "util/json.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <deque>
#include <set>
#include <unordered_map>

namespace rad {

using ojson = Json;

namespace {

/* The whitespace set std::isspace accepts in the C locale, which is the set the format's grammar
 * skips between tags. */
inline bool is_ws(unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }

/* The whitespace a reasoning block may consist of and still count as empty. */
inline bool is_blank(unsigned char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }

/* ================================================================== delimiter automaton
 *
 * A deterministic Aho-Corasick automaton over a few delimiters. The state is the longest suffix
 * of the scanned text that is a prefix of some delimiter, so its depth is the number of trailing
 * bytes a streaming scan must hold, and a state whose string is a delimiter is a match.
 *
 * Reporting matches this way is exact only because chat_format_check refuses delimiter sets in
 * which one delimiter occurs inside another at a non-zero offset. A match then ends at the first
 * byte where any delimiter completes with no earlier-starting match in progress, so matches come
 * out in start order, the order the grammar reads them. */
class Dfa {
public:
    void build(const std::vector<std::string>& pats) {
        pats_ = pats;
        struct Node { int next[256]; int depth; int word; };
        std::vector<Node> t(1);
        std::fill(std::begin(t[0].next), std::end(t[0].next), -1);
        t[0].depth = 0; t[0].word = -1;
        for (size_t i = 0; i < pats.size(); ++i) {
            int cur = 0;
            for (unsigned char c : pats[i]) {
                if (t[(size_t)cur].next[c] < 0) {
                    Node n;
                    std::fill(std::begin(n.next), std::end(n.next), -1);
                    n.depth = t[(size_t)cur].depth + 1;
                    n.word  = -1;
                    t.push_back(n);
                    t[(size_t)cur].next[c] = (int)t.size() - 1;
                }
                cur = t[(size_t)cur].next[c];
            }
            if (cur != 0 && t[(size_t)cur].word < 0) t[(size_t)cur].word = (int)i;
        }
        const size_t n = t.size();
        tr_.assign(n * 256, 0);
        depth_.assign(n, 0);
        word_.assign(n, -1);
        std::vector<int> fail(n, 0), order;
        order.reserve(n);
        for (int c = 0; c < 256; ++c) {
            const int v = t[0].next[c];
            if (v > 0) { fail[(size_t)v] = 0; tr_[(size_t)c] = (uint16_t)v; order.push_back(v); }
        }
        for (size_t k = 0; k < order.size(); ++k) {
            const int u = order[k];
            for (int c = 0; c < 256; ++c) {
                const int v = t[(size_t)u].next[c];
                const uint16_t via = tr_[(size_t)fail[(size_t)u] * 256 + (size_t)c];
                if (v > 0) {
                    fail[(size_t)v] = via;
                    tr_[(size_t)u * 256 + (size_t)c] = (uint16_t)v;
                    order.push_back(v);
                } else {
                    tr_[(size_t)u * 256 + (size_t)c] = via;
                }
            }
        }
        for (size_t i = 0; i < n; ++i) {
            depth_[i] = (uint16_t)t[i].depth;
            word_[i]  = (int16_t)t[i].word;
        }
        std::fill(std::begin(first_), std::end(first_), false);
        n_first_ = 0;
        for (const std::string& p : pats)
            if (!p.empty() && !first_[(unsigned char)p[0]]) {
                first_[(unsigned char)p[0]] = true;
                only_ = (unsigned char)p[0];
                ++n_first_;
            }
    }

    uint16_t step(uint16_t s, unsigned char c) const { return tr_[(size_t)s * 256 + c]; }
    size_t   depth(uint16_t s) const { return depth_[s]; }
    int      word(uint16_t s) const { return word_[s]; }
    const std::string& pattern(int i) const { return pats_[(size_t)i]; }

    /* The first byte in [p, e) that can begin a delimiter. Plain text between delimiters is
     * skipped here without running the automaton at all. */
    const char* skip(const char* p, const char* e) const {
        if (n_first_ == 1) {
            const void* q = std::memchr(p, only_, (size_t)(e - p));
            return q ? (const char*)q : e;
        }
        if (n_first_ == 0) return e;
        while (p < e && !first_[(unsigned char)*p]) ++p;
        return p;
    }

private:
    std::vector<std::string> pats_;
    std::vector<uint16_t>    tr_;
    std::vector<uint16_t>    depth_;
    std::vector<int16_t>     word_;
    bool                     first_[256] = {};
    int                      n_first_ = 0;
    unsigned char            only_ = 0;
};

/* ================================================================== JSON
 *
 * A JSON value, recognised one byte at a time with an explicit stack. It accepts exactly what the
 * format's grammar accepts as a JSON-typed argument, quirks included, because the answer decides
 * whether the argument is typed or text:
 *
 *   * no whitespace before the value, any whitespace after and inside it;
 *   * a number may not be followed by a character that could have continued it (`01`, `1.`, `1-`
 *     are not numbers), while `true` and `null` may be followed by anything;
 *   * a raw control character or a non-JSON escape fails a string, as the schema's string rule
 *     does (a client's JSON parser rejects a raw newline in a string).
 *
 * An accepted value is still checked whole by a JSON parser before it is written as JSON
 * (json_value_done), since the grammar admits a few spellings a client cannot read.
 *
 * push() returns More while the bytes are still a JSON value or its trailing whitespace, Fail at
 * the first byte that cannot be, and Done at the first byte after a complete value, which is not
 * consumed. */
class JsonScan {
public:
    enum Res : uint8_t { More, Fail, Done };

    void reset() { st_ = Value; stack_.clear(); lit_ = nullptr; lpos_ = 0; need_ = 0; }

    Res push(unsigned char c) {
        for (;;) {
            switch (st_) {
            case Value:
                return start_value(c);
            case ValueWs:
                if (is_ws(c)) return More;
                st_ = Value;
                continue;
            case Str:
                if (need_ > 0) {
                    if ((c & 0xC0) != 0x80) return Fail;
                    --need_;
                    return More;
                }
                if (c == '"') { st_ = key_ ? KeyAfter : After; return More; }
                if (c == '\\') { st_ = StrEsc; return More; }
                if (c < 0x20) return Fail;
                if (c < 0x80) return More;
                if (!(c & 0x40)) return Fail;
                if (!(c & 0x20)) { need_ = 1; return More; }
                if (!(c & 0x10)) { need_ = 2; return More; }
                if (!(c & 0x08)) { need_ = 3; return More; }
                return Fail;
            case StrEsc:
                if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' ||
                    c == 'r' || c == 't') { st_ = Str; return More; }
                if (c == 'u') { st_ = StrHex; need_ = 4; return More; }
                return Fail;
            case StrHex:
                if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
                    return Fail;
                if (--need_ == 0) st_ = Str;
                return More;
            case NumMinus:
                if (c == '0') { st_ = NumZero; return More; }
                if (c >= '1' && c <= '9') { st_ = NumInt; return More; }
                return Fail;
            case NumZero:
                if (c == '.') { st_ = NumDot; return More; }
                if (c == 'e' || c == 'E') { st_ = NumE; return More; }
                if (num_continues(c)) return Fail;
                st_ = After;
                continue;
            case NumInt:
                if (c >= '0' && c <= '9') return More;
                if (c == '.') { st_ = NumDot; return More; }
                if (c == 'e' || c == 'E') { st_ = NumE; return More; }
                if (num_continues(c)) return Fail;
                st_ = After;
                continue;
            case NumDot:
                if (c >= '0' && c <= '9') { st_ = NumFrac; return More; }
                return Fail;
            case NumFrac:
                if (c >= '0' && c <= '9') return More;
                if (c == 'e' || c == 'E') { st_ = NumE; return More; }
                if (num_continues(c)) return Fail;
                st_ = After;
                continue;
            case NumE:
                if (c == '+' || c == '-') { st_ = NumSign; return More; }
                if (c >= '0' && c <= '9') { st_ = NumExp; return More; }
                return Fail;
            case NumSign:
                if (c >= '0' && c <= '9') { st_ = NumExp; return More; }
                return Fail;
            case NumExp:
                if (c >= '0' && c <= '9') return More;
                if (num_continues(c)) return Fail;
                st_ = After;
                continue;
            case Lit:
                if ((unsigned char)lit_[lpos_] != c) return Fail;
                if (lit_[++lpos_] == '\0') st_ = After;
                return More;
            case After:
                if (is_ws(c)) return More;
                if (stack_.empty()) return Done;
                if (stack_.back() == '{') {
                    if (c == ',') { st_ = ObjKeyWs; return More; }
                    if (c == '}') { stack_.pop_back(); st_ = After; return More; }
                    return Fail;
                }
                if (c == ',') { st_ = ValueWs; return More; }
                if (c == ']') { stack_.pop_back(); st_ = After; return More; }
                return Fail;
            case ObjOpen:
                if (is_ws(c)) return More;
                if (c == '}') { stack_.pop_back(); st_ = After; return More; }
                if (c == '"') { st_ = Str; key_ = true; need_ = 0; return More; }
                return Fail;
            case ObjKeyWs:
                if (is_ws(c)) return More;
                if (c == '"') { st_ = Str; key_ = true; need_ = 0; return More; }
                return Fail;
            case KeyAfter:
                if (is_ws(c)) return More;
                if (c == ':') { st_ = ValueWs; return More; }
                return Fail;
            case ArrOpen:
                if (is_ws(c)) return More;
                if (c == ']') { stack_.pop_back(); st_ = After; return More; }
                st_ = Value;
                continue;
            }
            return Fail;
        }
    }

private:
    enum St : uint8_t {
        Value, ValueWs, Str, StrEsc, StrHex, NumMinus, NumZero, NumInt, NumDot, NumFrac, NumE,
        NumSign, NumExp, Lit, After, ObjOpen, ObjKeyWs, KeyAfter, ArrOpen,
    };
    /* Deeper than any schema describes. The recognizer does not recurse, so this bounds memory
     * rather than the stack; a value nested past it is read as text. */
    static constexpr size_t kMaxDepth = 1u << 16;

    static bool num_continues(unsigned char c) {
        return (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-';
    }

    Res start_value(unsigned char c) {
        switch (c) {
        case '{':
            if (stack_.size() >= kMaxDepth) return Fail;
            stack_.push_back('{'); st_ = ObjOpen; return More;
        case '[':
            if (stack_.size() >= kMaxDepth) return Fail;
            stack_.push_back('['); st_ = ArrOpen; return More;
        case '"': st_ = Str; key_ = false; need_ = 0; return More;
        case '-': st_ = NumMinus; return More;
        case '0': st_ = NumZero; return More;
        case 't': st_ = Lit; lit_ = "true";  lpos_ = 1; return More;
        case 'f': st_ = Lit; lit_ = "false"; lpos_ = 1; return More;
        case 'n': st_ = Lit; lit_ = "null";  lpos_ = 1; return More;
        default:
            if (c >= '1' && c <= '9') { st_ = NumInt; return More; }
            return Fail;
        }
    }

    St                st_ = Value;
    std::vector<char> stack_;
    const char*       lit_ = nullptr;
    size_t            lpos_ = 0;
    int               need_ = 0;
    bool              key_ = false;
};

/* ================================================================== JSON text helpers */

/* A string's JSON encoding without its quotes, byte for byte what Json::dump writes:
 * two-character escapes where JSON has one, \u00XX in lower-case hex for other control
 * characters, and everything else (DEL and UTF-8 included) verbatim. */
void json_escape_into(std::string& out, const char* p, size_t n) {
    static const char hex[] = "0123456789abcdef";
    const char* run = p;
    const char* e = p + n;
    for (; p < e; ++p) {
        const unsigned char c = (unsigned char)*p;
        if (c >= 0x20 && c != '"' && c != '\\') continue;
        out.append(run, (size_t)(p - run));
        run = p + 1;
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    out.append(run, (size_t)(e - run));
}

std::string json_quote(std::string_view s) {
    std::string out = "\"";
    json_escape_into(out, s.data(), s.size());
    out += '"';
    return out;
}

/* How many trailing bytes of `s` (from `from` on) form an unfinished UTF-8 sequence. A delta
 * becomes a JSON string on the wire, where half a code point is malformed. */
size_t utf8_tail(const std::string& s, size_t from) {
    const size_t n = s.size();
    for (size_t back = 1; back <= 3 && back <= n - from; ++back) {
        const unsigned char c = (unsigned char)s[n - back];
        if ((c & 0xC0) == 0x80) continue;
        size_t need = 1;
        if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        return back < need ? back : 0;
    }
    return 0;
}

/* The grammar's value transforms, on a whole value: used for values short enough to hold entire
 * and for the held ends of long ones. */
std::string_view strip_leading_newline(std::string_view v) {
    if (!v.empty() && v.front() == '\n') v.remove_prefix(1);
    else if (v.size() >= 2 && v[0] == '\r' && v[1] == '\n') v.remove_prefix(2);
    return v;
}

std::string_view strip_trailing_newline(std::string_view v) {
    if (!v.empty() && v.back() == '\n') {
        v.remove_suffix(1);
        if (!v.empty() && v.back() == '\r') v.remove_suffix(1);
    }
    return v;
}

constexpr std::string_view kCdOpen  = "<![CDATA[";
constexpr std::string_view kCdClose = "]]>";
/* A response-format document may be fenced as a Markdown code block. */
constexpr std::string_view kFenceOpen  = "```json";
constexpr std::string_view kFenceClose = "```";

/* A CDATA section is punctuation, not content; its two ends are stripped independently. A value
 * still arriving holds back what could become the opener, and a trailing `]` or `]]` that could
 * start the closer. */
void strip_cdata_close(std::string_view& v, bool partial) {
    if (v.size() >= kCdClose.size() &&
        v.compare(v.size() - kCdClose.size(), kCdClose.size(), kCdClose) == 0) {
        v.remove_suffix(kCdClose.size());
        return;
    }
    if (!partial) return;
    size_t hold = 0;
    while (hold + 1 < kCdClose.size() && v.size() > hold && v[v.size() - 1 - hold] == ']') ++hold;
    v.remove_suffix(hold);
}

void unwrap_cdata(std::string_view& v, bool partial) {
    if (v.size() < kCdOpen.size()) {
        if (partial && !v.empty() && kCdOpen.compare(0, v.size(), v) == 0) v = std::string_view();
        return;
    }
    if (v.compare(0, kCdOpen.size(), kCdOpen) != 0) return;
    v.remove_prefix(kCdOpen.size());
    strip_cdata_close(v, partial);
}

/* An argument name as the grammar reads it: without one leading whitespace character and all
 * trailing ones. */
std::string_view trim_arg_name(std::string_view v) {
    if (!v.empty() && is_ws((unsigned char)v.front())) v.remove_prefix(1);
    while (!v.empty() && is_ws((unsigned char)v.back())) v.remove_suffix(1);
    return v;
}

std::string_view trim_trailing_ws(std::string_view v) {
    while (!v.empty() && is_ws((unsigned char)v.back())) v.remove_suffix(1);
    return v;
}

/* ================================================================== an argument value
 *
 * One value's bytes, as the delimiters bound them, turned into its JSON encoding in the arguments
 * object: a string (quoted, escaped), or raw JSON text (a value that parsed as JSON, or a
 * non-string argument that did not).
 *
 * A non-string argument that did not parse is converted to JSON, never passed through: one bare
 * token makes the client's arguments object unparseable, and the call cannot be reported at all.
 * The grammar admits text for a typed argument, and models write Python there: `True`, `False`,
 * `None` and single-quoted containers.
 *
 *   * Scalar: held while it could still be a Python literal (`True` as true, `False`, `None` as
 *     null) or a JSON number or literal, up to kScalar bytes, then written as that. Anything
 *     else, or longer, streams as a JSON string of its text. An empty value is null.
 *   * Container (first byte after at most four blanks is `[` or `{`): held whole, because only at
 *     its end is it known whether it is JSON once its quotes and Python literals are normalised.
 *     If it is not, it becomes a JSON string of its text.
 *   * A value the reply was cut off inside is written as it arrived, since there is no end to
 *     read and its call is incomplete. A value the JSON scanner accepted (json_value_done) is
 *     written as is.
 *
 * The transforms touch only the ends (one newline at each end, the CDATA opener and closer), so
 * the middle streams through and only the first kHead bytes (until the head is decided) and the
 * last kTail are held. */
class ValueOut {
public:
    enum Mode : uint8_t { String, Raw };
    static constexpr size_t kHead = 16;   /* a newline pair and the CDATA opener, and the tail */
    static constexpr size_t kTail = 5;    /* a newline pair and the CDATA closer */
    /* The longest scalar held while it could still be a literal or a number: a few blanks and
     * a 64-bit integer, or a double in its shortest round-trip spelling. */
    static constexpr size_t kScalar = 28;

    /* `lead`, `trail`: a newline at that end is punctuation. `json`: the value already parsed as
     * JSON, so a Raw scalar is written as it is. */
    void begin(Mode m, bool cdata, bool lead, bool trail, std::string& out, bool json = false) {
        mode_ = m; cdata_ = cdata; lead_ = lead; trail_ = trail; json_ = json; str_ = false;
        buf_.clear(); head_ = false; opened_ = false;
        container_ = false;
        scalar_.clear(); cont_.clear();
        if (mode_ == String) out += '"';
    }

    void add(const char* p, size_t n, std::string& out) {
        buf_.append(p, n);
        if (!head_) {
            if (buf_.size() < kHead) return;
            std::string_view v(buf_);
            size_t drop = 0;
            if (lead_) drop = v.size() - strip_leading_newline(v).size();
            if (cdata_ && v.compare(drop, kCdOpen.size(), kCdOpen) == 0) {
                opened_ = true;
                drop += kCdOpen.size();
            }
            buf_.erase(0, drop);
            head_ = true;
        }
        if (buf_.size() > kTail) {
            const size_t k = buf_.size() - kTail;
            emit(buf_.data(), k, out);
            buf_.erase(0, k);
        }
    }

    void finish(bool partial, std::string& out) {
        std::string_view v(buf_);
        if (!head_) {
            if (lead_) v = strip_leading_newline(v);
            if (trail_) v = strip_trailing_newline(v);
            if (cdata_) unwrap_cdata(v, partial);
        } else {
            if (trail_) v = strip_trailing_newline(v);
            if (opened_) strip_cdata_close(v, partial);
        }
        emit(v.data(), v.size(), out);
        if (mode_ == Raw && !json_) {
            if (container_) container_out(partial, out);
            else if (str_) { if (!partial) out += '"'; }
            else if (partial) { out += scalar_; scalar_.clear(); }   /* cut off: as it arrived */
            else scalar_out(out);
        }
        buf_.clear();
    }

    /* Bytes held while the value's ends are decided, and a scalar while it could be a literal. */
    size_t held() const { return buf_.size() + scalar_.size(); }
    /* A container, held whole until its end. */
    size_t held_container() const { return cont_.size(); }

private:
    void emit(const char* p, size_t n, std::string& out) {
        if (n == 0) return;
        if (mode_ == String) { json_escape_into(out, p, n); return; }
        if (json_) { out.append(p, n); return; }
        if (str_) { json_escape_into(out, p, n); return; }
        if (container_) { cont_.append(p, n); return; }
        scalar_.append(p, n);
        size_t ws = 0;
        while (ws < scalar_.size() && is_ws((unsigned char)scalar_[ws])) ++ws;
        if (ws <= 4 && ws < scalar_.size() && (scalar_[ws] == '[' || scalar_[ws] == '{')) {
            container_ = true;
            cont_.assign(scalar_, ws, std::string::npos);
            scalar_.clear();
            return;
        }
        if (scalar_.size() > kScalar || !scalar_prefix(scalar_)) {
            out += '"';
            json_escape_into(out, scalar_.data(), scalar_.size());
            scalar_.clear();
            str_ = true;
        }
    }

    /* Could `v` still become a Python or JSON literal or a JSON number, after at most a few
     * bytes of blanks on either side? */
    static bool scalar_prefix(std::string_view v) {
        size_t ws = 0;
        while (ws < v.size() && is_ws((unsigned char)v[ws])) ++ws;
        if (ws > 4) return false;
        v.remove_prefix(ws);
        size_t tw = 0;
        while (tw < v.size() && is_ws((unsigned char)v[v.size() - 1 - tw])) ++tw;
        if (tw > 4) return false;
        if (tw > 0 && tw == v.size()) return true;
        v.remove_suffix(tw);
        if (v.empty()) return true;
        for (std::string_view l : { std::string_view("True"), std::string_view("False"),
                                    std::string_view("None"), std::string_view("true"),
                                    std::string_view("false"), std::string_view("null") })
            if (v.size() <= l.size() && l.compare(0, v.size(), v) == 0) return true;
        return number_prefix(v);
    }

    /* A prefix of -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)? */
    static bool number_prefix(std::string_view v) {
        size_t i = 0;
        if (i < v.size() && v[i] == '-') ++i;
        if (i == v.size()) return true;
        if (v[i] == '0') ++i;
        else if (v[i] >= '1' && v[i] <= '9') while (i < v.size() && std::isdigit((unsigned char)v[i])) ++i;
        else return false;
        if (i < v.size() && v[i] == '.') {
            ++i;
            while (i < v.size() && std::isdigit((unsigned char)v[i])) ++i;
        }
        if (i < v.size() && (v[i] == 'e' || v[i] == 'E')) {
            ++i;
            if (i < v.size() && (v[i] == '+' || v[i] == '-')) ++i;
            while (i < v.size() && std::isdigit((unsigned char)v[i])) ++i;
        }
        return i == v.size();
    }

    /* A Python literal is JSON's under another spelling; anything else is as written. */
    static const char* json_literal(std::string_view w) {
        if (w == "True") return "true";
        if (w == "False") return "false";
        if (w == "None") return "null";
        return nullptr;
    }

    void scalar_out(std::string& out) {
        std::string_view v = trim_trailing_ws(scalar_);
        while (!v.empty() && is_ws((unsigned char)v.front())) v.remove_prefix(1);
        if (v.empty()) out += "null";
        else if (const char* lit = json_literal(v)) out += lit;
        else if (Json::accept(v)) out.append(v.data(), v.size());
        else out += json_quote(scalar_);
        scalar_.clear();
    }

    /* The held container, normalised; JSON if that made it JSON, else a JSON string of its text.
     * One the reply was cut off inside is written normalised as far as it arrived. */
    void container_out(bool partial, std::string& out) {
        std::string j;
        j.reserve(cont_.size() + 8);
        normalise(cont_, j);
        if (partial || Json::accept(j)) out += j;
        else out += json_quote(cont_);
        cont_.clear();
    }

    /* Python-style quoting to JSON: a single-quoted string becomes a double-quoted one, with the
     * quotes inside it re-escaped to match and its raw control characters escaped, and a bare
     * Python literal outside every string is respelled as JSON's. Whatever is no JSON after that
     * is left as written, for the JSON parser to refuse. */
    static void normalise(std::string_view v, std::string& out) {
        bool in_single = false, in_double = false;
        for (size_t i = 0; i < v.size(); ++i) {
            const char c = v[i];
            if (in_single || in_double) {
                if (c == '\\' && i + 1 < v.size()) {
                    const char d = v[++i];
                    if (d == '\'') out += '\'';          /* `\'` is a quote in either quoting */
                    else if (in_single && d == '"') out += "\\\"";
                    else { out += '\\'; out += d; }
                } else if (c == (in_single ? '\'' : '"')) {
                    in_single = in_double = false;
                    out += '"';
                } else if (c == '"') {
                    out += "\\\"";
                } else if ((unsigned char)c < 0x20) {
                    /* A raw line break or tab inside a string, which JSON spells as an escape. */
                    json_escape_into(out, &c, 1);
                } else {
                    out += c;
                }
                continue;
            }
            if (std::isalnum((unsigned char)c) || c == '_') {
                size_t e = i + 1;
                while (e < v.size() && (std::isalnum((unsigned char)v[e]) || v[e] == '_')) ++e;
                const std::string_view w = v.substr(i, e - i);
                if (const char* lit = json_literal(w)) out += lit;
                else out.append(w.data(), w.size());
                i = e - 1;
                continue;
            }
            if (c == '\'') in_single = true;
            else if (c == '"') in_double = true;
            out += c == '\'' ? '"' : c;
        }
    }

    Mode        mode_ = String;
    bool        cdata_ = true, lead_ = true, trail_ = true;
    std::string buf_;
    bool        head_ = false, opened_ = false;
    bool        container_ = false;
    std::string scalar_;         /* a scalar while it could still be a Python literal */
    std::string cont_;           /* a container, held whole */
    bool        json_ = false;   /* the value parsed as JSON: written as it is */
    bool        str_ = false;    /* the scalar is no literal and streams as a JSON string */
};

/* ================================================================== the answer text
 *
 * Content is trimmed at the end: trailing spaces and newlines, then one turn terminator. On a
 * stream that means holding the trailing whitespace and any whole or partial terminator until a
 * non-whitespace byte follows. The terminator automaton tracks the longest suffix that could
 * still become one. */
class ContentOut {
public:
    void init(const Dfa* terms, const std::vector<std::string>* list) { terms_ = terms; list_ = list; }

    void add(const char* p, size_t n, std::string& full) {
        for (size_t i = 0; i < n; ++i) {
            const unsigned char c = (unsigned char)p[i];
            if (c == ' ' || c == '\n') {
                if (ws_ == 0)
                    term_ = terms_ && terms_->word(ts_) >= 0 ? terms_->depth(ts_) : 0;
                ++ws_;
            } else {
                ws_ = 0;
            }
            if (terms_) ts_ = terms_->step(ts_, c);
        }
        pend_.append(p, n);
        const size_t hold = ws_ ? ws_ + term_ : (terms_ ? terms_->depth(ts_) : 0);
        if (pend_.size() > hold) {
            full.append(pend_, 0, pend_.size() - hold);
            pend_.erase(0, pend_.size() - hold);
        }
    }

    /* The exact trim: trailing spaces and newlines, then the first terminator in the list that
     * ends what is left. */
    void finish(std::string& full) {
        trim(pend_);
        full += pend_;
        pend_.clear();
    }

    size_t held_space() const { return std::min(ws_, pend_.size()); }
    /* What is held: the content after `full` so far, untrimmed. */
    const std::string& pending() const { return pend_; }
    /* The same trim, on a whole text. */
    void trim(std::string& s) const {
        while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
        if (list_)
            for (const std::string& t : *list_)
                if (!t.empty() && s.size() >= t.size() &&
                    s.compare(s.size() - t.size(), t.size(), t) == 0) {
                    s.erase(s.size() - t.size());
                    break;
                }
    }
    size_t held_other() const { return pend_.size() - held_space(); }

private:
    const Dfa*                      terms_ = nullptr;
    const std::vector<std::string>* list_ = nullptr;
    std::string                     pend_;
    size_t                          ws_ = 0, term_ = 0;
    uint16_t                        ts_ = 0;
};

}  /* namespace */

/* ================================================================== the compiled format */

struct ReplyFormat::Impl {
    ChatFormat  f;
    /* Each tag split into its core and the whitespace around it that is optional to match. */
    std::string open_core, open_trail, close_core, close_trail, ns_core, ns_trail, end_core,
                end_trail, co_core, co_trail, section_open_core, section_close_core;
    /* What begins a call in content. First the format's own opener: call_open's core, or
     * name_prefix when calls are not wrapped. A wrapped format also takes name_prefix alone,
     * second, since models often write a call without its wrapper; a wrapper written without
     * its trailing whitespace is read too. */
    std::vector<std::string> openers, text_markers;
    size_t section_open_hit = (size_t)-1, section_close_hit = (size_t)-1;
    size_t think_open_hit = (size_t)-1, think_close_hit = (size_t)-1;
    size_t body_section_close_hit = (size_t)-1;
    /* What ends an argument name: arg_name_suffix and arg_value_prefix, back to back. With
     * typed tags (arg_json_value_prefix) also `name_end_json`, which opens a JSON value; the
     * arg_name automata report it as pattern 1. */
    std::string name_end, name_end_json;
    /* The same, whitespace-trimmed. A loose call is one the grammar never opened, so it was
     * written unconstrained; its argument tags, like its name and close, are read with their
     * surrounding whitespace optional. */
    std::string name_end_core, name_end_json_core, value_core;
    /* call_close's core, when distinct from what else may follow inside a call, so a wrapped
     * call missing its own close is ended by its wrapper's. */
    std::string cc_core;
    Dfa think, think_breaks, text, arg_name, value, junk, arg_name_loose, value_loose;
    size_t longest = 0;       /* the longest delimiter */

    /* Object family. `object`: a call's arguments are one object. `ns_object`: the object's own
     * `{` ends the name (name_suffix empty), and ns_core is "{". */
    bool object = false, ns_object = false;
    /* Tagged family, name_suffix empty: a call's name ends where arg_prefix or end_core begins,
     * and that tag is the first of the call's body. */
    bool name_open = false;
    /* Where a key or value may begin, besides the object's punctuation: the dialect string's
     * opener, and the call's close and end (written there by a reply that leaves its object
     * unclosed). Matched anchored, by first byte; chat_format_check keeps them distinct. */
    enum : uint8_t { LitString, LitClose, LitEnd };
    std::vector<std::string> obj_lits;
    std::vector<uint8_t>     obj_lit_kind;
    /* A dialect string runs to its close. A bare value ends at `,` `}` `]` and a bare key at `:`
     * `,` `}` (patterns 0..2), or at the call's close or end, whose pattern indices are below
     * (-1: none). */
    Dfa obj_str, obj_bare, obj_key;
    /* A value opened with a plain `"` (or `'`, obj_sqstr) in a dialect with strings of its own:
     * read verbatim, like the dialect's strings, up to a quote after which the object's syntax
     * continues. Patterns: the quote, the escaped quote, the dialect's string close, the call's
     * close (if any). */
    Dfa obj_qstr, obj_sqstr;
    int bare_cc = -1, bare_ce = -1, key_cc = -1, key_ce = -1;

    /* reasoning_anywhere: content is also scanned for a later block's opener and a stray close,
     * after the call openers (text_think) or alone when the request has no calls (think_only). */
    Dfa text_think, think_only;
};

ReplyFormat::ReplyFormat() : impl_(std::make_unique<Impl>()) {}
ReplyFormat::~ReplyFormat() = default;
const ChatFormat& ReplyFormat::format() const { return impl_->f; }

static void split_ws(const std::string& tag, std::string* core, std::string* trail) {
    size_t a = 0, b = tag.size();
    while (a < b && is_ws((unsigned char)tag[a])) ++a;
    while (b > a && is_ws((unsigned char)tag[b - 1])) --b;
    *core = tag.substr(a, b - a);
    *trail = tag.substr(b);
}

int ReplyFormat::compile(const ChatFormat& f, std::shared_ptr<const ReplyFormat>* out,
                         std::string* why) {
    const int st = chat_format_check(f, why);
    if (st < 0) return st;
    std::shared_ptr<ReplyFormat> r(new ReplyFormat());
    Impl& m = *r->impl_;
    m.f = f;
    split_ws(f.reasoning_open, &m.open_core, &m.open_trail);
    split_ws(f.reasoning_close, &m.close_core, &m.close_trail);
    split_ws(f.name_suffix, &m.ns_core, &m.ns_trail);
    split_ws(f.call_end, &m.end_core, &m.end_trail);
    split_ws(f.call_open, &m.co_core, &m.co_trail);
    if (f.call_open.empty()) m.openers = { f.name_prefix };
    else m.openers = { m.co_core, f.name_prefix };
    m.text_markers = m.openers;
    if (!f.section_open.empty()) {
        std::string section_trail;
        split_ws(f.section_open, &m.section_open_core, &section_trail);
        split_ws(f.section_close, &m.section_close_core, &section_trail);
        m.section_open_hit = m.text_markers.size();
        m.text_markers.push_back(m.section_open_core);
        m.section_close_hit = m.text_markers.size();
        m.text_markers.push_back(m.section_close_core);
    }
    m.name_end = f.arg_name_suffix + f.arg_value_prefix;
    if (f.typed_tags()) m.name_end_json = f.arg_name_suffix + f.arg_json_value_prefix;
    m.object = f.has_tools() && f.object_args();
    m.name_open = f.has_tools() && !m.object && f.name_suffix.empty();
    if (m.object && f.name_suffix.empty()) {
        m.ns_object = true;
        m.ns_core = "{";
        m.ns_trail.clear();
    }

    if (f.has_reasoning()) {
        m.think.build({ m.close_core });
        std::vector<std::string> b{ m.close_core };
        for (const std::string& s : f.reasoning_breaks) b.push_back(s);
        m.think_breaks.build(b);
    }
    if (f.reasoning_anywhere && f.has_reasoning()) {
        std::vector<std::string> t = m.text_markers;
        m.think_open_hit = t.size();
        t.push_back(m.open_core);
        m.think_close_hit = t.size();
        t.push_back(m.close_core);
        m.text_think.build(t);
        m.think_only.build({ m.open_core, m.close_core });
    }
    if (m.object) {
        m.text.build(m.text_markers);
        std::string trail;
        split_ws(f.call_close, &m.cc_core, &trail);
        std::vector<std::string> bare{ ",", "}", "]" }, key{ ":", ",", "}" };
        if (!f.object_string_open.empty()) {
            m.obj_lits.push_back(f.object_string_open);
            m.obj_lit_kind.push_back(Impl::LitString);
            /* The call's close inside a string: where a model left the string unclosed. */
            std::vector<std::string> sp{ f.object_string_close };
            std::vector<std::string> qp{ "\"", "\\\"", f.object_string_close };
            std::vector<std::string> sq{ "'", "\\'", f.object_string_close };
            if (!m.cc_core.empty()) {
                sp.push_back(m.cc_core);
                qp.push_back(m.cc_core);
                sq.push_back(m.cc_core);
            }
            m.obj_str.build(sp);
            m.obj_qstr.build(qp);
            m.obj_sqstr.build(sq);
        }
        if (!m.cc_core.empty()) {
            m.obj_lits.push_back(m.cc_core);
            m.obj_lit_kind.push_back(Impl::LitClose);
            m.bare_cc = (int)bare.size(); bare.push_back(m.cc_core);
            m.key_cc = (int)key.size(); key.push_back(m.cc_core);
        }
        if (!m.end_core.empty() && m.end_core != m.cc_core) {
            m.obj_lits.push_back(m.end_core);
            m.obj_lit_kind.push_back(Impl::LitEnd);
            m.bare_ce = (int)bare.size(); bare.push_back(m.end_core);
            m.key_ce = (int)key.size(); key.push_back(m.end_core);
        }
        m.obj_bare.build(bare);
        m.obj_key.build(key);
    } else if (f.has_tools()) {
        m.text.build(m.text_markers);
        std::string trail;
        split_ws(m.name_end, &m.name_end_core, &trail);
        split_ws(m.name_end_json, &m.name_end_json_core, &trail);
        split_ws(f.arg_value_suffix, &m.value_core, &trail);
        if (f.typed_tags()) {
            m.arg_name.build({ m.name_end, m.name_end_json });
            m.arg_name_loose.build({ m.name_end_core, m.name_end_json_core });
        } else {
            m.arg_name.build({ m.name_end });
            m.arg_name_loose.build({ m.name_end_core });
        }
        m.value.build({ f.arg_value_suffix });
        m.value_loose.build({ m.value_core });
        split_ws(f.call_close, &m.cc_core, &trail);
        auto apart = [](const std::string& a, const std::string& b) {
            return a.find(b) == std::string::npos && b.find(a) == std::string::npos;
        };
        if (!apart(m.cc_core, f.arg_prefix) || !apart(m.cc_core, m.end_core)) m.cc_core.clear();
        std::vector<std::string> junk{ f.arg_prefix, m.end_core };
        if (!m.cc_core.empty()) junk.push_back(m.cc_core);
        if (!m.section_close_core.empty()) {
            m.body_section_close_hit = junk.size();
            junk.push_back(m.section_close_core);
        }
        m.junk.build(junk);
    }
    const std::string* const delims[] = { &f.reasoning_open, &f.reasoning_close, &f.section_open,
                                          &f.section_close, &f.call_open, &f.call_close,
                                          &f.name_prefix, &f.name_suffix,
                                          &f.call_end, &f.arg_prefix, &m.name_end,
                                          &m.name_end_json, &f.arg_value_suffix,
                                          &f.object_string_open,
                                          &f.object_string_close };
    for (const std::string* s : delims) m.longest = std::max(m.longest, s->size());
    for (const std::string& s : f.reasoning_breaks) m.longest = std::max(m.longest, s.size());
    *out = std::move(r);
    return RAD_OK;
}

/* ================================================================== the bound request */

namespace {
struct Param {
    bool typed = false;       /* read as JSON when it parses as JSON */
};
struct Tool {
    std::string name;
    std::unordered_map<std::string, Param> params;
    /* The declared names in the order the grammar tries them: required ones, then the rest,
     * each in schema order. */
    std::vector<std::string> arms;
    bool has_props = false;
    bool has_required = false;
};
}  /* namespace */

struct ReplyParser::Impl {
    std::shared_ptr<const ReplyFormat> fmt;
    ReplyOptions opt;
    std::string  tail;             /* the part of the generation prompt the reply continues */
    bool         calls = false;    /* the request sent tools and the format has calls */
    bool         breaks = false;   /* an unclosed reasoning block ends at a call */
    bool         pure = false;     /* no reasoning, calls or document: every byte is content */
    /* A reasoning block may open later in the reply too (ChatFormat::reasoning_anywhere), and
     * this request extracts reasoning. */
    bool         anywhere = false;
    std::vector<Tool> tools;
    std::unordered_map<std::string, size_t> by_name;
    std::vector<std::string> terminators;
    Dfa          terms;
};

ReplyParser::ReplyParser() : impl_(std::make_unique<Impl>()) {}
ReplyParser::~ReplyParser() = default;
const ChatFormat& ReplyParser::format() const { return impl_->fmt->format(); }

size_t ReplyParser::hold_bound() const {
    size_t term = 0;
    for (const std::string& t : impl_->terminators) term = std::max(term, t.size());
    return impl_->fmt->impl().longest + std::max(term, kFenceOpen.size()) + ValueOut::kHead + 1 +
           3 * 3 + ValueOut::kScalar;
}

ReasoningMarkers ReplyParser::reasoning() const {
    const ReplyFormat::Impl& F = impl_->fmt->impl();
    ReasoningMarkers m;
    if (!F.f.has_reasoning()) return m;
    m.open  = F.open_core;
    m.close = F.close_core;
    /* `tail` starts at the opener when the prompt holds one; a prompt that also closed the block
     * (thinking off) leaves the reply after it. With no opener the stream starts in the block
     * (after_lead). */
    const std::string& t = impl_->tail;
    m.starts_open = F.open_core.empty() ||
                    (!t.empty() && t.find(F.close_core, F.open_core.size()) == std::string::npos);
    return m;
}

/* Read the tools the way the grammar generator reads them, so a name or a type means the same
 * thing to the parser as to the grammar that constrained the reply. */
static int bind_tools(const ReplyFormat::Impl& F, const ojson& tools, ReplyParser::Impl* P,
                      std::string* why) {
    for (const auto& tool : tools) {
        if (!chat_function_tool(tool)) continue;
        try {
            const auto& func = tool.at("function");
            Tool t;
            t.name = func.at("name").get<std::string>();
            auto params = func.contains("parameters") ? func.at("parameters") : ojson::object();
            const auto properties = params.contains("properties") ? params.at("properties")
                                                                 : ojson::object();
            std::set<std::string> required;
            if (params.contains("required")) params.at("required").get_to(required);
            common_schema_info info;
            info.resolve_refs(params);
            std::vector<std::string> optional;
            for (const auto& [pname, pschema] : properties.items()) {
                /* An optional argument no value satisfies has no grammar arm and is read by the
                 * undeclared arm as text (chatfmt.cpp); the parser does the same. */
                if (!required.count(pname) && info.unsatisfiable(pschema)) continue;
                Param p;
                p.typed = F.f.other_value == ChatValue::Json && !info.resolves_to_string(pschema);
                t.has_props = true;
                if (required.count(pname)) {
                    t.has_required = true;
                    t.arms.push_back(pname);
                } else {
                    optional.push_back(pname);
                }
                /* A name its own delimiter could end early would be read as a different name.
                 * Object keys are delimited by the object's punctuation instead, and a key
                 * containing it is written as a string. */
                for (const std::string* ne : { &F.name_end, &F.name_end_json }) {
                    if (ne->empty()) continue;
                    const std::string probe = pname + *ne;
                    if (!F.object && probe.find(*ne) != pname.size()) {
                        *why = "parameter '" + pname + "' of tool '" + t.name + "' contains '" +
                               *ne + "', which this model's tool-call format uses to end an "
                               "argument name";
                        return RAD_E_INVAL;
                    }
                }
                t.params.emplace(pname, p);
            }
            t.arms.insert(t.arms.end(), optional.begin(), optional.end());
            if (F.name_open) {
                for (const std::string* d : { &F.f.arg_prefix, &F.end_core })
                    if ((t.name + *d).find(*d) != t.name.size()) {
                        *why = "tool '" + t.name + "' has a name running into '" + *d +
                               "', which this model's tool-call format uses to end a function "
                               "name";
                        return RAD_E_INVAL;
                    }
            } else if (t.name.find(F.ns_core) != std::string::npos ||
                (!F.ns_object &&
                 (t.name + F.f.name_suffix).find(F.f.name_suffix) != t.name.size())) {
                *why = "tool '" + t.name + "' has a name containing '" + F.ns_core +
                       "', which this model's tool-call format uses to end a function name";
                return RAD_E_INVAL;
            }
            if (!P->by_name.count(t.name)) P->by_name.emplace(t.name, P->tools.size());
            P->tools.push_back(std::move(t));
        } catch (const std::exception& e) {
            *why = std::string("a tool definition could not be read: ") + e.what();
            return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

int ReplyParser::create(std::shared_ptr<const ReplyFormat> fmt, const ReplyOptions& opt,
                        std::shared_ptr<const ReplyParser>* out, std::string* why) {
    std::string dummy;
    if (!why) why = &dummy;
    if (!fmt) { *why = "no reply format"; return RAD_E_INVAL; }
    std::shared_ptr<ReplyParser> r(new ReplyParser());
    Impl& P = *r->impl_;
    const ReplyFormat::Impl& F = fmt->impl();
    P.fmt = fmt;
    P.opt = opt;
    P.opt.tools = nullptr;

    const bool want_calls = opt.tools && opt.tools->is_array() && !opt.tools->empty();
    if (want_calls && !F.f.has_tools()) {
        *why = "this model's reply format has no tool calls";
        return RAD_E_UNSUPPORTED;
    }
    if (want_calls) {
        const int st = bind_tools(F, *opt.tools, &P, why);
        if (st < 0) return st;
    }
    /* A response format replaces the call grammar outright; the calls are not parsed. */
    P.calls  = want_calls && !opt.response_format;
    P.breaks = want_calls && F.f.has_reasoning() && !F.f.reasoning_breaks.empty();
    P.pure   = !F.f.has_reasoning() && !want_calls && !opt.response_format &&
               F.f.content_open.empty();
    P.anywhere = F.f.reasoning_anywhere && F.f.has_reasoning() && opt.extract_reasoning &&
                 !opt.response_format;

    if (F.f.has_reasoning()) {
        switch (F.f.reasoning_in_prompt) {
        case ChatThink::FromPrompt:
            if (!F.f.reasoning_open.empty() && !opt.generation_prompt.empty()) {
                const size_t at = opt.generation_prompt.find(F.f.reasoning_open);
                if (at != std::string::npos) P.tail = opt.generation_prompt.substr(at);
            }
            break;
        case ChatThink::Always: P.tail = F.f.reasoning_open; break;
        case ChatThink::Never:  break;
        }
    }
    for (const std::string& t : opt.terminators)
        if (!t.empty()) P.terminators.push_back(t);
    P.terms.build(P.terminators);
    *out = std::move(r);
    return RAD_OK;
}

/* ================================================================== the stream */

namespace {
enum class St : uint8_t {
    LeadWs,       /* whitespace before the reply proper: dropped */
    ThinkOpen,    /* the core of reasoning_open */
    ThinkOpenWs,  /* the optional whitespace that ends reasoning_open */
    ContentOpen,  /* content_open, the header of a reply that does not reason */
    Think,        /* reasoning text */
    ThinkCloseWs, /* the optional whitespace that ends reasoning_close */
    Text,         /* content, watching for the call opener when calls are on */
    OpenCoTail,   /* the whitespace that ends call_open, after its core */
    OpenWs,       /* after call_open: whitespace */
    OpenPrefix,   /* name_prefix, after call_open */
    OpenName,     /* the function name, to the core of name_suffix (or, without one, to the
                     first argument or the call's end) */
    OpenTail,     /* the whitespace that ends name_suffix */
    OpenPeek,     /* a loose call's name is read: whitespace, then an argument or the close */
    OpenPeekArg,  /* ... and the first argument's name, which must close as a name */
    Body,         /* inside a call: whitespace, then an argument or the call's end */
    BodyLit,      /* arg_prefix or call_end */
    ArgName,      /* an argument name, to its end */
    ValText,      /* a string (or raw) value, to arg_value_suffix */
    ValJson,      /* a JSON-typed value */
    ValJsonEnd,   /* arg_value_suffix, exactly, after a complete JSON value */
    EndTail,      /* the whitespace that ends call_end */
    Junk,         /* text inside a call that is no argument: content */
    AfterEnd,     /* after call_end: whitespace, then call_close */
    AfterEndLit,  /* call_close */
    AfterCall,    /* after a call: whitespace, dropped */
    RfWs,         /* before the document: whitespace, dropped */
    RfFence,      /* "```json" */
    RfFenceWs,    /* whitespace after the fence, dropped */
    RfJson,       /* the document */
    RfClose,      /* "```" */
    RfEnd,        /* after the closing fence */
    RfRaw,        /* a reply that is not the document it was asked for: content */
    ObjStart,     /* after a name_suffix: whitespace, then the argument object's `{` */
    ObjKey,       /* where a key may begin, or the object's `}` */
    ObjKeyBare,   /* a bare key, to its `:` */
    ObjColon,     /* after a quoted key: whitespace, then `:` */
    ObjValue,     /* where a value may begin, or an empty array's `]` */
    ObjLit,       /* the dialect's string opener, or the call's close or end, being matched */
    ObjStr,       /* a dialect string, to its close */
    ObjQStr,      /* a value opened with a plain quote in a dialect with strings of its own */
    ObjQLook,     /* after a plain quote there: does the object's syntax go on? */
    ObjJStr,      /* a JSON string, to its quote */
    ObjBare,      /* a bare value: a number, a literal, or a word read as a string */
    ObjAfter,     /* after a value: whitespace, then `,` or a closing bracket */
    ObjEnd,       /* after the object: whitespace, then call_end */
};

/* A function name the grammar does not know is still read as a call, up to this long. */
constexpr size_t kMaxLooseName = 1024;
/* And what a loose opener may hold in all before it is known to be a call. */
constexpr size_t kMaxLooseHold = 2 * kMaxLooseName;
}  /* namespace */

struct ReplyStream::Impl {
    const ReplyParser::Impl& P;
    const ReplyFormat::Impl& F;
    explicit Impl(const ReplyParser::Impl& p) : P(p), F(p.fmt->impl()) {
        content.init(P.terminators.empty() ? nullptr : &P.terms, &P.terminators);
        st = P.pure ? St::Text : St::LeadWs;
        think_hold_on = P.calls && !F.f.reasoning_breaks.empty();
        if (P.anywhere) {
            text_dfa = P.calls ? &F.text_think : &F.think_only;
            n_open = P.calls ? F.openers.size() : 0;
        } else if (P.calls) {
            text_dfa = &F.text;
            n_open = F.openers.size();
        }
    }

    /* What content is scanned with: the call openers when calls are on, then (reasoning_anywhere)
     * a later block's opener and a stray close. Null when every byte is content. */
    const Dfa* text_dfa = nullptr;
    size_t     n_open = 0;

    ReplyMessage msg;
    ContentOut   content;
    bool         think_text = false;    /* the reasoning has had a non-blank byte */
    std::string  think_pend;            /* its leading blanks, until then */
    std::vector<ReplyEvent>* out = nullptr;

    St   st;
    bool strict = true, finished = false, fed_tail = false;
    bool post_call = false;             /* content after a call */
    bool section_active = false;
    size_t section_calls_at_open = 0;
    /* The last completed inner call is not executable until another call starts or the outer
     * section closes. Deferring CallEnd keeps truncation consistent with the one-shot parse. */
    size_t pending_section_call = (size_t)-1;

    /* delimiter scan */
    uint16_t    ds = 0;
    std::string hold;
    /* literal match */
    std::string lhold;
    size_t      lpos = 0, trail = 0;
    /* a call being opened */
    St          origin = St::Text;
    /* `loose`: opened by something the grammar does not open a call with (name_prefix alone, or
     * call_open without its trailing whitespace). Such an opener is also ordinary text often
     * enough ("the `<function=f>` syntax") that it becomes a call only once an argument or the
     * call's close follows the name tag. Until then it is held; if it is not a call it is
     * content, as the grammar says, and the reply stays strict. `wrapped`: opened with
     * call_open, so its close is followed by call_close. */
    bool        loose = false, wrapped = false;
    size_t      tag_end = 0;            /* where a loose call's name tag ends in `att` */
    bool        name_lt = false;        /* a loose call's argument name holds a '<' */
    std::string att, name;
    int         tool = -1;
    /* the open call */
    size_t      argc = 0, calls = 0;
    int         depth = 0;              /* unclosed braces in the arguments so far */
    bool        in_str = false, esc = false;
    const Param* param = nullptr;
    /* The argument's tag was the JSON one (typed tags): its value is read as JSON. */
    bool        json_tag = false;
    /* a value */
    ValueOut    val;
    ValueOut::Mode vmode = ValueOut::String;
    JsonScan    js;
    std::string vheld;
    size_t      vs_pos = 0;
    bool        fenced = false;
    /* Channel-scoped protocols (Muse): with tools on, a trailing `to=NAME<|message|>` while
     * reasoning streams is the handoff to a tool call, not reasoning text. The reasoning's
     * tail is held until it cannot grow into framing or a header; a completed tool-channel
     * header is dropped along with everything after it -- the section opener is where the
     * call machinery takes over. Without the hold the header lands in the reasoning stream,
     * and a model that hesitates at the handoff repeats it (observed on Muse Glimmer: three
     * or four `to=read<|message|>` headers before the section, under speculative decoding). */
    bool        think_hold_on = false;
    std::string think_hold;
    bool        think_handoff = false;

    /* The argument object being read: its open brackets; whether a key (not a value) is being
     * read; a separator read but not yet written (written only once the next member shows the
     * object continues, so a trailing one never reaches the JSON); whether the container was
     * just opened; and whether a key's `:` went out with no value yet. */
    std::vector<char> ostack;
    bool        okey = false, ocomma = false, ofirst = false, vpending = false;
    /* A bare value's first bytes, to tell at its end whether it was a JSON number or literal
     * (which the grammar admits) or a word read as a string (which it does not). */
    std::string bare_txt;
    /* A high surrogate's escape, held until the low one shows it is a pair. The escape being
     * read (the backslash and a "u" escape's four hex digits) is in vheld. */
    std::string jhi;
    /* Inside a string: `srun` holds its trailing run of whitespace, closing brackets and quotes
     * (the brackets are the object's own if the call's close follows); `nq`, `nb`, `ns` count the
     * double quotes, backticks and single quotes it has written; `qtail` holds the bytes after a
     * plain quote while they decide what it was; `qmode` is how a plain-quoted value writes its
     * quotes: 0 not yet seen, 1 as \" (one is unescaped), 2 as they are. */
    std::string srun, qtail;
    size_t      nq = 0, nb = 0, ns = 0;
    uint8_t     qmode = 0;
    char        qc = '"';

    /* published offsets, for the deltas */
    size_t sent_r = 0, sent_c = 0, sent_a = 0;
    int    cur = -1;                    /* the channel last published: 0 r, 1 c, 2 args */

    /* Bytes to read again, ahead of the rest. They are always the last bytes read, so each
     * segment knows its first byte's position in the stream (the generation prompt's tail, then
     * the reply). */
    struct Seg { const char* p; size_t n; size_t at; };
    std::vector<Seg>        stack;
    const char*             seg_p = nullptr;   /* the segment being read */
    size_t                  seg_at = 0;        /* and where its first byte is in the stream */
    size_t                  fed = 0;           /* the bytes the stream has been given */
    size_t                  att_at = 0;        /* where the opener being tried begins */
    size_t here(const char* p) const { return seg_at + (size_t)(p - seg_p); }
    std::deque<std::string> owned;
    std::string             refeed;
    size_t                  reread = 0;        /* every byte ever queued in refeed */

    void lenient() { strict = false; }

    /* ------------------------------------------------------------ output channels */

    void event(ReplyEvent::Kind k, uint32_t call, std::string_view text) {
        if (!out) return;
        if ((k == ReplyEvent::Reasoning || k == ReplyEvent::Content || k == ReplyEvent::CallArgs) &&
            !out->empty() && out->back().kind == k && out->back().call == call) {
            out->back().text.append(text);
            return;
        }
        ReplyEvent e;
        e.kind = k;
        e.call = call;
        e.text.assign(text);
        out->push_back(std::move(e));
    }

    void publish(int ch, bool all) {
        std::string* s; size_t* sent; ReplyEvent::Kind k; uint32_t call = 0;
        if (ch == 0) { s = &msg.reasoning; sent = &sent_r; k = ReplyEvent::Reasoning; }
        else if (ch == 1) { s = &msg.content; sent = &sent_c; k = ReplyEvent::Content; }
        else {
            if (msg.calls.empty()) return;
            s = &msg.calls.back().arguments; sent = &sent_a; k = ReplyEvent::CallArgs;
            call = (uint32_t)msg.calls.size() - 1;
        }
        const size_t end = all ? s->size() : s->size() - utf8_tail(*s, *sent);
        if (end > *sent) {
            event(k, call, std::string_view(*s).substr(*sent, end - *sent));
            *sent = end;
        }
    }

    void touch(int ch) {
        if (cur != ch && cur >= 0) publish(cur, false);
        cur = ch;
    }

    void think_out(const char* p, size_t n) {
        if (n == 0) return;
        if (think_handoff) return;   /* past the tool-channel header: call framing, dropped */
        touch(0);
        if (!think_hold_on) {
            if (think_text) { msg.reasoning.append(p, n); return; }
            for (size_t i = 0; i < n; ++i)
                if (!is_blank((unsigned char)p[i])) { think_text = true; break; }
            think_pend.append(p, n);
            if (think_text) { msg.reasoning += think_pend; think_pend.clear(); }
            return;
        }
        touch(0);
        think_hold.append(p, n);
        flush_think_hold(false);
    }

    void emit_reasoning_upto(size_t emit) {
        if (emit == 0) return;
        const std::string part = think_hold.substr(0, emit);
        if (think_text) { msg.reasoning += part; }
        else {
            for (size_t i = 0; i < part.size(); ++i)
                if (!is_blank((unsigned char)part[i])) { think_text = true; break; }
            think_pend += part;
            if (think_text) { msg.reasoning += think_pend; think_pend.clear(); }
        }
    }

    /* The handoff position from the last `<|eom|>`: the tail reads as the tool-channel
     * header -- optional `<|start|>assistant`, ws, `to=`, recipient name bytes (never
     * `user`: that transition matches the reply close above and is consumed by it), then
     * `<|message|>` or a prefix of it while the header streams. npos: no handoff. */
    size_t tool_handoff_at() const {
        static const std::string eom = "<|eom|>", start = "<|start|>assistant", to = "to=";
        /* The framing is bounded (terminator + assistant + `to=` + a recipient + the message
         * marker); scanning only the tail keeps a byte-at-a-time stream linear. */
        const size_t wstart = think_hold.size() - std::min(kHoldWindow, think_hold.size());
        size_t p = std::string::npos;
        for (size_t q = think_hold.find(eom, wstart); q != std::string::npos;
             q = think_hold.find(eom, q + 1))
            p = q;
        if (p == std::string::npos) return std::string::npos;
        size_t k = p + eom.size();
        if (think_hold.compare(k, start.size(), start) == 0) k += start.size();
        while (k < think_hold.size() && is_ws((unsigned char)think_hold[k])) ++k;
        if (think_hold.compare(k, to.size(), to) != 0) return std::string::npos;
        return p;   /* everything from `<|eom|>` on is the tool call's framing */
    }

    /* Muse's channel protocol, per the reference parsers (vLLM's muse reasoning/tool
     * parsers; llama.cpp PR #26841): reasoning streams until `<|eom|>`, and the model
     * sometimes leaves the analysis channel WITHOUT it, opening a tool channel directly
     * (`<|start|>assistant to=read<|message|>`) -- a documented model-side defect. So the
     * reasoning's tail is held while it could still grow into that framing, and once the
     * handoff is detected everything from the header on is dropped: the call machinery
     * (the section + invoke tags) has already produced the tool_calls events. */
    void flush_think_hold(bool final_feed) {
        if (think_handoff) return;   /* past the handoff: framing, dropped */
        size_t emit = think_hold.size();
        const size_t handoff = tool_handoff_at();
        if (handoff != std::string::npos) {
            emit = handoff;
            think_handoff = true;
        } else if (!final_feed) {
            /* Hold a tail that could still grow into the handoff framing: a partial
             * `<|eom|>` / `<|start|>` / `<|message|>` marker, or a trailing ` to=NAME`. */
            static const std::string markers[] = {"<|eom|>", "<|start|>", "<|message|>"};
            for (const std::string& mk : markers)
                for (size_t k = std::min(mk.size(), think_hold.size()); k > 0; --k)
                    if (think_hold.compare(think_hold.size() - k, k, mk, 0, k) == 0)
                        emit = std::min(emit, think_hold.size() - k);
            const size_t wstart_t = think_hold.size() - std::min(kHoldWindow, think_hold.size());
            size_t at = std::string::npos;
            for (size_t q = think_hold.find("to=", wstart_t); q != std::string::npos;
                 q = think_hold.find("to=", q + 1))
                at = q;
            if (at != std::string::npos) {
                bool tail = true;
                for (size_t k = at + 3; k < think_hold.size(); ++k)
                    if (!is_name_byte((unsigned char)think_hold[k])) { tail = false; break; }
                if (tail) {
                    size_t a = at;
                    while (a > 0 && is_ws((unsigned char)think_hold[a - 1])) --a;
                    emit = std::min(emit, a);
                }
            }
            if (emit >= think_hold.size()) {
                emit_reasoning_upto(emit);
                think_hold.clear();   /* nothing held: everything goes out now */
                return;
            }
        }
        if (emit > 0) emit_reasoning_upto(emit);
        think_hold.erase(0, emit);
        if (final_feed) think_hold.clear();
    }

    /* The tail the framing scans read; a longer tail is real reasoning, never framing. */
    static constexpr size_t kHoldWindow = 160;

    static bool is_name_byte(unsigned char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
               c == '_' || c == '.' || c == '-';
    }

    void text_out(const char* p, size_t n) {
        if (n == 0) return;
        if (post_call && (!F.f.call_open.empty() || !P.opt.parallel_tool_calls)) lenient();
        touch(1);
        content.add(p, n, msg.content);
    }

    void args_out(std::string_view s) {
        std::string& a = msg.calls.back().arguments;
        for (char ch : s) {
            if (esc) { esc = false; continue; }
            if (ch == '\\' && in_str) { esc = true; continue; }
            if (ch == '"') { in_str = !in_str; continue; }
            if (!in_str) { if (ch == '{') ++depth; else if (ch == '}') --depth; }
        }
        touch(2);
        a.append(s);
    }

    /* The value encoder writes into a scratch string; the arguments see it through args_out so
     * the brace count stays right. */
    std::string scratch;
    void flush_scratch() { if (!scratch.empty()) { args_out(scratch); scratch.clear(); } }

    /* ------------------------------------------------------------ scanning */

    /* Scan text under `d` from the current state, handing plain text to `sink`. Returns the
     * position after a matched delimiter with *hit its index, or `e` with *hit = -1. */
    template <class Sink>
    const char* scan(const Dfa& d, const char* p, const char* e, int* hit, Sink&& sink) {
        *hit = -1;
        while (p < e) {
            if (ds == 0) {
                const char* q = d.skip(p, e);
                if (q > p) sink(p, (size_t)(q - p));
                p = q;
                if (p == e) break;
            }
            const unsigned char c = (unsigned char)*p++;
            const uint16_t ns = d.step(ds, c);
            const size_t keep = d.depth(ns);
            const size_t win = hold.size() + 1;
            if (keep < win) {
                const size_t rel = win - keep;
                if (rel <= hold.size()) {
                    sink(hold.data(), rel);
                    hold.erase(0, rel);
                    hold.push_back((char)c);
                } else {
                    if (!hold.empty()) sink(hold.data(), hold.size());
                    hold.clear();
                    const char cc = (char)c;
                    sink(&cc, 1);
                }
            } else {
                hold.push_back((char)c);
            }
            ds = ns;
            if (d.word(ds) >= 0) {
                *hit = d.word(ds);
                ds = 0;
                hold.clear();
                return p;
            }
        }
        return p;
    }

    void read_again(std::string_view bytes) {
        refeed.append(bytes);
        reread += bytes.size();
    }

    /* ------------------------------------------------------------ calls */

    bool declared_prefix(std::string_view n) const {
        for (const Tool& t : P.tools) {
            const std::string full = t.name + (F.ns_object ? F.ns_core : F.f.name_suffix);
            if (full.size() >= n.size() && full.compare(0, n.size(), n) == 0) return true;
        }
        return false;
    }

    /* A suffixless name (ReplyFormat::Impl::name_open) ends where arg_prefix or end_core begins.
     * Returns which one ended it (0, 1), with *len the name's length before it; or -1, with *len
     * the bytes that are the name's for certain, the rest possibly beginning either. */
    int name_stop(size_t* len) const {
        const std::string* const stops[] = { &F.f.arg_prefix, &F.end_core };
        size_t pend = 0;
        for (int i = 0; i < 2; ++i) {
            const std::string& t = *stops[i];
            if (name.size() >= t.size() && name.compare(name.size() - t.size(), t.size(), t) == 0) {
                *len = name.size() - t.size();
                return i;
            }
            for (size_t k = std::min(t.size() - 1, name.size()); k > pend; --k)
                if (name.compare(name.size() - k, k, t, 0, k) == 0) { pend = k; break; }
        }
        *len = name.size() - pend;
        return -1;
    }

    /* Could these bytes, after any leading whitespace, still be a function name? One holds no
     * '<' and no whitespace before more text, unless a declared name does. Without a tag of its
     * own to close it, a name read from prose that mentions the opener would otherwise be held
     * up to kMaxLooseName bytes. */
    bool name_plausible(std::string_view n) const {
        size_t i = 0;
        while (i < n.size() && is_ws((unsigned char)n[i])) ++i;
        bool gap = false;
        for (; i < n.size(); ++i) {
            const unsigned char c = (unsigned char)n[i];
            if (c == '<') return declared_prefix(n);
            if (is_ws(c)) gap = true;
            else if (gap) return declared_prefix(n);
        }
        return true;
    }

    /* OpenName for a suffixless name. The tag that ends it is read here, as the first of the
     * call's body: an argument's opener, or the call's end. Whitespace around the name is
     * dropped; before it, the grammar does not admit it. */
    const char* open_name_byte(const char* p) {
        size_t len = 0;
        const int stop = name_stop(&len);
        if (stop < 0) {
            if (len > kMaxLooseName || !name_plausible(std::string_view(name).substr(0, len)))
                fail_attempt();
            return p + 1;
        }
        size_t a = 0;
        while (a < len && is_ws((unsigned char)name[a])) ++a;
        const std::string nm(trim_trailing_ws(std::string_view(name).substr(a, len - a)));
        auto it = P.by_name.find(nm);
        tool = it == P.by_name.end() ? -1 : (int)it->second;
        if (tool < 0 && (nm.empty() || nm.find('<') != std::string::npos)) {
            fail_attempt();
            return p + 1;
        }
        name = nm;
        begin_call(a == 0);
        lpos = 0;
        lhold.clear();
        if (stop == 0) {
            start_arg();
        } else {
            finish_call_strictness();
            end_core_done();
        }
        return p + 1;
    }

    void complete_section_call() {
        if (pending_section_call == (size_t)-1) return;
        msg.calls[pending_section_call].closed = true;
        event(ReplyEvent::CallEnd, (uint32_t)pending_section_call, {});
        pending_section_call = (size_t)-1;
    }

    void open_section() {
        if (section_active) { lenient(); return; }
        section_active = true;
        section_calls_at_open = calls;
    }

    void close_section() {
        if (!section_active) { lenient(); return; }
        if (calls == section_calls_at_open) lenient();
        complete_section_call();
        section_active = false;
    }

    /* A section close before call_end leaves the last call incomplete: it must never be exposed
     * as an executable tool call. Keep its bytes available to content_with_open_call(). */
    void abandon_call_at_section_close() {
        lenient();
        if (cur == 2) publish(2, true);
        cur = -1;
        section_active = false;
        post_call = false;
        ds = 0;
        hold.clear();
        lhold.clear();
        st = St::Text;
    }

    /* The call opener has matched at `att`; whether a call follows is decided below. */
    void begin_attempt(St from) {
        origin = from;
        name.clear();
        tool = -1;
        lpos = 0;
        loose = false;
        wrapped = !F.f.call_open.empty();
        if (!P.calls) { fail_attempt(); return; }
        st = F.f.call_open.empty() ? St::OpenName : St::OpenWs;
    }

    /* An opener matched in content. In a wrapped format the first is call_open's core, whose
     * trailing whitespace decides whether the grammar opened a call; the second is name_prefix
     * without the wrapper, which is always loose. */
    void begin_text_attempt(int which) {
        origin = St::Text;
        name.clear();
        tool = -1;
        lpos = 0;
        trail = 0;
        att = F.openers[(size_t)which];
        if (F.f.call_open.empty()) {
            loose = false;
            wrapped = false;
            st = St::OpenName;
        } else if (which == 0) {
            loose = false;
            wrapped = true;
            st = St::OpenCoTail;
        } else {
            loose = true;
            wrapped = false;
            st = St::OpenName;
        }
    }

    /* Whitespace after a loose opener: any byte at or below a space, which covers a model's stray
     * control characters. */
    static bool loose_ws(unsigned char c) { return c <= ' '; }

    /* Not a call after all: the opener's first byte is text of the region it was found in, and
     * everything after it is read again from there. */
    void fail_attempt() {
        /* The grammar committed to a call at its own opener, so a call that does not follow
         * leaves it; a loose opener that turns out to be text is text, within it. */
        if (!loose) lenient();
        loose = false;
        st = origin;
        ds = 0;
        hold.clear();
        lhold.clear();
        if (origin == St::Think) think_out(att.data(), 1);
        else text_out(att.data(), 1);
        read_again(std::string_view(att).substr(1));
        att.clear();
    }

    void begin_call(bool exact) {
        complete_section_call();
        if (!exact || tool < 0) lenient();
        if (!P.opt.parallel_tool_calls && calls > 0) lenient();
        if (cur >= 0) publish(cur, false);
        if (cur == 2) publish(2, true);
        const std::string_view nm = trim_trailing_ws(std::string_view(name).substr(
            0, name.size() - F.ns_core.size()));
        ReplyCall c;
        c.name.assign(nm);
        c.at = att_at >= P.tail.size() ? att_at - P.tail.size() : 0;
        c.content_at = msg.content.size();
        c.content_held = content.pending();
        msg.calls.push_back(std::move(c));
        sent_a = 0;
        cur = -1;
        event(ReplyEvent::CallBegin, (uint32_t)msg.calls.size() - 1, msg.calls.back().name);
        depth = 0; in_str = false; esc = false;
        argc = 0;
        ++calls;
        att.clear();
        post_call = false;
        if (F.object) {
            /* The arguments are the object's, and its `{` is written now: a reply that ends
             * here is a call with no arguments yet, as in the tagged family. */
            obj_out("{");
            ostack.assign(1, '{');
            okey = ocomma = vpending = false;
            ofirst = true;
            jhi.clear();
            st = F.ns_object ? St::ObjKey : St::ObjStart;
            return;
        }
        args_out("{");
        st = St::Body;
    }

    /* ------------------------------------------------------------ the argument object
     *
     * The object family's argument object, read as it streams and written as JSON as it is read:
     * brackets and separators as they come, a key once its `:` shows it is one, a string as its
     * bytes arrive, and a bare value like a typed tagged value (ValueOut, Raw): held while it
     * could still be a literal or a number, else a string. Dialect spellings (string delimiters,
     * bare keys) are respelled. Departures from the dialect (a missing or trailing separator, a
     * mismatched bracket, an object cut off by the call's close) are repaired locally and clear
     * strict(). A closed call's arguments are always JSON. */

    void obj_out(std::string_view s) {
        if (s.empty()) return;
        touch(2);
        msg.calls.back().arguments.append(s.data(), s.size());
    }
    void obj_flush() { if (!scratch.empty()) { obj_out(scratch); scratch.clear(); } }

    /* A member or an element begins: the separator before it, if one was read, goes out now. */
    void obj_sep() {
        if (ocomma) { obj_out(","); ocomma = false; }
        ofirst = false;
    }

    /* The outermost bracket closed: what follows the object. */
    void obj_done() {
        if (!F.end_core.empty()) { lpos = 0; lhold.clear(); st = St::ObjEnd; }
        else end_call();
    }

    /* A closing bracket where a key, a value or a separator may stand. A separator before it is
     * dropped, and a bracket of the other kind closes the open container all the same. */
    void obj_close_bracket(char c) {
        if (ocomma) { lenient(); ocomma = false; }
        const char top = ostack.back();
        if ((top == '{') != (c == '}')) lenient();
        if (vpending) { lenient(); obj_out("null"); vpending = false; }
        ostack.pop_back();
        ofirst = false;
        obj_out(top == '{' ? "}" : "]");
        if (ostack.empty()) obj_done();
        else st = St::ObjAfter;
    }

    /* The punctuation after a value. */
    void obj_struct(char c) {
        if (c != ',') { obj_close_bracket(c); return; }
        if (ocomma) lenient();          /* `,,`: one separator */
        ocomma = true;
        ofirst = false;
        okey = ostack.back() == '{';
        st = okey ? St::ObjKey : St::ObjValue;
    }

    bool obj_lit_start(unsigned char c) const {
        for (const std::string& l : F.obj_lits)
            if ((unsigned char)l[0] == c) return true;
        return false;
    }

    /* Bytes that open no string and close no call: a bare key or a bare value. */
    void obj_bare_begin() {
        ds = 0;
        hold.clear();
        if (okey) {
            if (!F.f.object_bare_keys) lenient();
            name.clear();
            st = St::ObjKeyBare;
            return;
        }
        obj_sep();
        vpending = false;
        bare_txt.clear();
        val.begin(ValueOut::Raw, /*cdata=*/false, /*lead=*/false, /*trail=*/false, scratch);
        obj_flush();
        st = St::ObjBare;
    }

    static constexpr size_t kBareKeep = 64;   /* longer than any number a model writes */

    /* A bare value ended: one that is no JSON number or literal was read by the recovery rule. */
    void bare_done() {
        const std::string_view v = trim_trailing_ws(bare_txt);
        const bool json = bare_txt.size() < kBareKeep && !v.empty() &&
                          (v == "true" || v == "false" || v == "null" ||
                           ((v[0] == '-' || (v[0] >= '0' && v[0] <= '9')) &&
                            Json::accept(v)));
        if (!json) lenient();
    }

    /* A JSON string, which is the dialect's own only when it declares no string delimiters. */
    void json_string_begin() {
        if (!F.f.object_string_open.empty()) lenient();
        obj_out("\"");
        vheld.clear();
        st = St::ObjJStr;
    }

    /* One of the anchored literals matched where a key or a value may begin. */
    void obj_lit_done(uint8_t kind) {
        if (kind == ReplyFormat::Impl::LitString) {
            obj_sep();
            if (!okey) vpending = false;
            obj_out("\"");
            ds = 0;
            hold.clear();
            str_begin();
            st = St::ObjStr;
            return;
        }
        /* The call's close or end where the object wanted more: the reply left the object
         * unclosed, and the call ends here with it closed. */
        lenient();
        ocomma = false;
        if (kind == ReplyFormat::Impl::LitClose) end_call_at_close();
        else end_core_done();
    }

    /* A JSON string's text goes out as is, with its escapes checked, since a client's JSON parser
     * refuses all three of: a non-JSON escape (written as the backslash and letter as text), a
     * raw control character (escaped), and a surrogate escape that is not half of a pair
     * (written as U+FFFD). */
    void jstr_char(const char* q, size_t n) {
        if (n == 0) return;
        if (!jhi.empty()) { lenient(); jhi.clear(); obj_out("\\ufffd"); }
        touch(2);
        json_escape_into(msg.calls.back().arguments, q, n);
    }

    void jstr_escape_done(const std::string& e) {
        /* e: a complete two-character escape, or a "u" escape with its four digits. */
        if (e.size() == 6) {
            const unsigned long cp = std::strtoul(e.c_str() + 2, nullptr, 16);
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                if (!jhi.empty()) { lenient(); obj_out("\\ufffd"); }
                jhi = e;
                return;
            }
            if (cp >= 0xDC00 && cp <= 0xDFFF) {
                if (jhi.empty()) { lenient(); obj_out("\\ufffd"); return; }
                obj_out(jhi + e);
                jhi.clear();
                return;
            }
        }
        if (!jhi.empty()) { lenient(); jhi.clear(); obj_out("\\ufffd"); }
        obj_out(e);
    }

    static constexpr size_t kRunKeep  = 16;   /* the closers a string's end may hold back */
    static constexpr size_t kLookKeep = 32;   /* what deciding a close may read ahead */

    static bool run_byte(char c) {
        return c == '}' || c == ']' || c == '"' || c == '`' || c == '\'' || is_ws((unsigned char)c);
    }

    void str_begin() { srun.clear(); nq = nb = ns = 0; }
    void str_emit(const char* q, size_t n) {
        if (n == 0) return;
        nq += (size_t)std::count(q, q + n, '"');
        nb += (size_t)std::count(q, q + n, '`');
        ns += (size_t)std::count(q, q + n, '\'');
        touch(2);
        json_escape_into(msg.calls.back().arguments, q, n);
    }
    void str_flush() { str_emit(srun.data(), srun.size()); srun.clear(); }

    /* A string's text: all of it goes out but a trailing run of whitespace and closers. */
    void str_text(const char* q, size_t n) {
        size_t k = n;
        while (k > 0 && run_byte(q[k - 1])) --k;
        if (k > 0) { str_flush(); str_emit(q, k); }
        srun.append(q + k, n - k);
        if (srun.size() > kRunKeep) {
            const size_t out = srun.size() - kRunKeep;
            str_emit(srun.data(), out);
            srun.erase(0, out);
        }
    }

    /* Where the held run stops being text: the closers after that point close every open bracket
     * of the object, innermost first, as a model writes them before the call's close (read from
     * the end, the last closes the outermost). npos if they do not. */
    size_t run_cut() const {
        size_t i = srun.size(), k = 0;
        while (i > 0 && k < ostack.size()) {
            const char c = srun[i - 1];
            if (is_ws((unsigned char)c)) { --i; continue; }
            if (c != (ostack[k] == '{' ? '}' : ']')) break;
            --i;
            ++k;
        }
        return k == ostack.size() && k > 0 ? i : std::string::npos;
    }

    /* The call's close in a string whose held run closes the whole object: the model left the
     * string open (here it writes the close delimiter with p=0.86, `}` with p=0.08), and the call
     * ends here, as vLLM's and HF's Gemma 4 parsers end it at the first `}<tool_call|>`. The
     * closers are the object's. A quote or backtick just before them that leaves an odd count of
     * its kind in the string was the string's close in the wrong spelling. */
    void cut_begin() {
        const size_t cut = run_cut();
        if (cut == std::string::npos) { str_flush(); str_emit(F.cc_core.data(), F.cc_core.size()); return; }
        lenient();
        size_t keep = cut;
        if (keep > 0 && (srun[keep - 1] == '"' || srun[keep - 1] == '`' || srun[keep - 1] == '\'')) {
            const char c = srun[keep - 1];
            const size_t n = (c == '"' ? nq : c == '`' ? nb : ns) +
                             (size_t)std::count(srun.begin(), srun.begin() + (long)keep, c);
            if (n % 2 == 1) --keep;
        }
        str_emit(srun.data(), keep);
        srun.clear();
        jhi.clear();
        obj_out(okey ? "\":null" : "\"");
        vpending = false;
        ocomma = false;
        /* As after a call's close of its own: a repeated close goes with it. */
        end_call();
    }

    /* After a plain quote in a plain-quoted value: 0 not yet known; 1 it closed the value (the
     * object goes on with a member, an element, its closers or the call's close); 2 it was text. */
    int q_verdict() const {
        const std::string& q = qtail;
        if (q.size() > kLookKeep) return 2;
        const size_t n = q.size();
        size_t i = 0, top = ostack.size();
        auto ws = [&] { while (i < n && is_ws((unsigned char)q[i])) ++i; };
        ws();
        while (i < n && top > 0 && q[i] == (ostack[top - 1] == '{' ? '}' : ']')) { --top; ++i; ws(); }
        if (i == n) return 0;
        const std::string& so = F.f.object_string_open;
        std::string_view r(q.data() + i, n - i);
        if (!F.cc_core.empty()) {
            if (r.starts_with(F.cc_core)) return 1;
            if (std::string_view(F.cc_core).starts_with(r)) return 0;
        }
        if (top == 0 || q[i] != ',') return 2;
        ++i;
        ws();
        if (i == n) return 0;
        r = std::string_view(q.data() + i, n - i);
        const unsigned char c = (unsigned char)q[i];
        if (r.starts_with(so)) return 1;
        if (std::string_view(so).starts_with(r)) return 0;
        if (ostack[top - 1] == '{') {
            if (!(std::isalpha(c) || c == '_')) return 2;
            while (i < n && (std::isalnum((unsigned char)q[i]) || q[i] == '_')) ++i;
            ws();
            if (i == n) return 0;
            return q[i] == ':' ? 1 : 2;
        }
        return (c == '"' || c == '\'' || c == '{' || c == '[' || c == '-' || std::isdigit(c)) ? 1 : 2;
    }

    void q_done(bool closed) {
        std::string again;
        again.swap(qtail);
        if (closed) {
            str_flush();
            obj_string_done();
        } else {
            if (qmode == 0) qmode = 2;
            str_text(&qc, 1);
            st = St::ObjQStr;
        }
        read_again(again);
    }

    /* A string or a key closed. */
    void obj_string_done() {
        if (!jhi.empty()) { lenient(); jhi.clear(); obj_out("\\ufffd"); }
        obj_out("\"");
        st = okey ? St::ObjColon : St::ObjAfter;
    }

    /* The core of call_end has matched. The whitespace the format writes after it is part of the
     * tag, so the call closes once that arrives too (a reply that ends in between leaves the call
     * open, as the grammar does); a reply that goes straight on without it still closes the call. */
    void end_core_done() {
        trail = 0;
        if (F.end_trail.empty() || loose) end_call();
        else st = St::EndTail;
    }

    /* The call's close tag: the arguments object is closed with as many braces as it has open
     * outside strings. A string-typed value is always closed by its own suffix before this can
     * match, so no quote is ever pending here; a raw value that left a string open is passed
     * through as the model wrote it. */
    void end_call() {
        if (F.object) {
            /* What the object left open is closed, so the arguments are JSON: a key whose value
             * never came is null, and each bracket gets its partner. */
            std::string close = vpending ? "null" : "";
            vpending = false;
            for (auto it = ostack.rbegin(); it != ostack.rend(); ++it) close += *it == '{' ? '}' : ']';
            ostack.clear();
            obj_out(close);
        } else {
            std::string close((size_t)std::max(depth, 0), '}');
            args_out(close);
        }
        publish(2, true);
        if (section_active) {
            pending_section_call = msg.calls.size() - 1;
        } else {
            msg.calls.back().closed = true;
            event(ReplyEvent::CallEnd, (uint32_t)msg.calls.size() - 1, {});
        }
        cur = -1;
        lpos = 0;
        lhold.clear();
        if (F.f.call_open.empty()) st = St::AfterCall;
        else if (wrapped) st = St::AfterEnd;
        else after_loose_call();
    }

    /* The wrapper's close inside a wrapped call whose own close is missing ends both the call and
     * the wrapper. Lenient, since the grammar requires call_end first, but it must be read as a
     * close or the next call's arguments would be read into this one. */
    void end_call_at_close() {
        lenient();
        end_call();
        if (loose) after_loose_call();
        else st = St::AfterCall;
    }

    /* A loose call whose argument name runs into a tag was a call only as far as its arguments
     * before it: the call ends there, and the argument's opener and all after it are content. */
    void end_loose_call_at_arg(bool closed_name) {
        std::string replay = F.f.arg_prefix + name;
        if (closed_name) replay += name_end();
        name.clear();
        name_lt = false;
        wrapped = false;
        end_call();
        read_again(replay);
    }

    /* After a loose call the text goes on as content, whitespace included: the grammar never
     * opened the call, so nothing of its punctuation claims the spacing around it. */
    void after_loose_call() {
        loose = false;
        post_call = true;
        ds = 0;
        hold.clear();
        st = St::Text;
    }

    const Tool* cur_tool() const {
        return tool >= 0 ? &P.tools[(size_t)tool] : nullptr;
    }

    /* The argument name is complete. */
    void begin_value() {
        std::string_view nm = trim_arg_name(
            std::string_view(name).substr(0, name.size() - name_end().size()));
        param = nullptr;
        if (const Tool* t = cur_tool()) {
            auto it = t->params.find(std::string(nm));
            const bool exact = nm.size() + name_end().size() == name.size();
            /* The grammar binds a declared name to its own arm by the raw spelling; a name that
             * only trims to a declared one is read by the generic arm, as text. */
            if (it != t->params.end() && exact) {
                param = &it->second;
            } else if (it == t->params.end() && exact) {
                /* A name that repeats its own tag's opening, `<parameter=parameter=command>`: the
                 * grammar's undeclared arm takes it, and the call would reach the tool without
                 * the argument it names. Read as that declared argument, leniently. */
                const std::string_view rep = std::string_view(F.f.arg_prefix).substr(1);
                std::string_view rest = nm;
                while (!rep.empty() && rest.size() > rep.size() &&
                       rest.compare(0, rep.size(), rep) == 0)
                    rest.remove_prefix(rep.size());
                if (rest.size() < nm.size()) {
                    auto jt = t->params.find(std::string(rest));
                    if (jt != t->params.end()) {
                        lenient();
                        nm = rest;
                        param = &jt->second;
                    }
                }
            }
        }
        std::string key = argc > 0 ? "," : "";
        key += json_quote(nm);
        key += ':';
        args_out(key);
        ++argc;
        name.clear();
        /* With typed tags the reply's own tag says JSON or string, for any name; else the
         * declared schema does. */
        if (F.f.typed_tags() ? json_tag : (param && param->typed)) {
            js.reset();
            vheld.clear();
            st = St::ValJson;
        } else {
            start_text_value(ValueOut::String);
        }
    }

    /* What ends an argument name and a value in the call being read: the tags as the format
     * writes them, or their cores in a loose call. */
    const std::string& name_end() const {
        if (json_tag) return loose ? F.name_end_json_core : F.name_end_json;
        return loose ? F.name_end_core : F.name_end;
    }
    const std::string& value_suffix() const { return loose ? F.value_core : F.f.arg_value_suffix; }

    /* A CDATA section is punctuation around any value the format lets carry one, typed or not. */
    bool cdata() const { return F.f.string_value == ChatValue::Cdata; }

    /* The newline that opens a value is punctuation unless the tag before it has written it
     * already: Qwen's `<parameter=NAME>\n` ends with the newline the template puts before the
     * value, and a second one is the value's own. */
    bool lead_newline() const {
        const std::string& ne = name_end();
        return F.f.value_newlines && (ne.empty() || ne.back() != '\n');
    }

    /* `lit` continues the literal held in lhold (lpos bytes) with `c`. */
    bool extends(const std::string& lit, char c) const {
        return lpos < lit.size() && lit[lpos] == c && lit.compare(0, lpos, lhold) == 0;
    }

    void begin_value_out(ValueOut::Mode m, bool json = false) {
        val.begin(m, cdata(), lead_newline(), F.f.value_newlines, scratch, json);
    }

    void start_text_value(ValueOut::Mode m) {
        vmode = m;
        begin_value_out(m);
        flush_scratch();
        ds = 0;
        hold.clear();
        st = St::ValText;
    }

    /* A JSON-typed value that parsed and then met its suffix: the JSON text and the whitespace
     * after it. A client's JSON parser is slightly stricter than JsonScan's reading of the
     * grammar's JSON (a number past a double's range such as `1e400`, an unpaired UTF-16
     * surrogate escape, a vertical tab or form feed after the value). Such a value is read by the
     * grammar's other arm, as text, so the arguments stay JSON. */
    void json_value_done(std::string_view text) {
        begin_value_out(ValueOut::Raw, /*json=*/Json::accept(text));
        val.add(text.data(), text.size(), scratch);
        val.finish(false, scratch);
        flush_scratch();
    }

    /* ------------------------------------------------------------ the machine
     *
     * One step consumes at least one byte, or queues bytes to be read again. */
    const char* step(const char* p, const char* e) {
        int hit = -1;
        switch (st) {
        case St::LeadWs:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p < e) st = after_lead();
            return p;

        case St::ThinkOpen: {
            const unsigned char c = (unsigned char)*p;
            if (lpos < F.open_core.size() && (unsigned char)F.open_core[lpos] == c) {
                lhold.push_back((char)c);
                if (++lpos == F.open_core.size()) { lhold.clear(); trail = 0; st = St::ThinkOpenWs; }
                return p + 1;
            }
            /* No block: what was held is read again as whatever follows one. */
            st = after_no_block();
            read_again(lhold);
            lhold.clear();
            return p;
        }

        case St::ContentOpen: {
            const unsigned char c = (unsigned char)*p;
            if (lpos < F.f.content_open.size() && (unsigned char)F.f.content_open[lpos] == c) {
                lhold.push_back((char)c);
                if (++lpos == F.f.content_open.size()) { lhold.clear(); st = after_think(); }
                return p + 1;
            }
            /* Not the header: what was held is content (or the start of the document). */
            st = after_think();
            read_again(lhold);
            lhold.clear();
            return p;
        }

        case St::ThinkOpenWs:
            if (trail >= F.open_trail.size()) { st = St::Think; ds = 0; hold.clear(); return p; }
            if (*p == F.open_trail[trail]) { ++trail; return p + 1; }
            ++trail;
            return p;

        case St::Think: {
            const Dfa& d = P.breaks ? F.think_breaks : F.think;
            p = scan(d, p, e, &hit, [this](const char* q, size_t n) { think_out(q, n); });
            if (hit == 0) { flush_think_hold(true); trail = 0; st = St::ThinkCloseWs; }
            else if (hit > 0) {
                const std::string& marker = d.pattern(hit);
                if (!F.section_open_core.empty() && marker == F.section_open_core) {
                    flush_think_hold(true);
                    open_section();
                    st = St::Text;
                } else if (!F.section_open_core.empty() && !section_active) {
                    /* A call-looking tag cannot bypass the required outer section, including
                     * when it appears in an unclosed reasoning block. */
                    think_out(marker.data(), marker.size());
                } else {
                    att = marker;
                    att_at = here(p) - att.size();
                    begin_attempt(St::Think);
                }
            }
            return p;
        }

        case St::ThinkCloseWs:
            if (trail >= F.close_trail.size()) { st = after_think(); return p; }
            if (*p == F.close_trail[trail]) { ++trail; return p + 1; }
            ++trail;
            return p;

        case St::Text:
            if (!text_dfa) {
                text_out(p, (size_t)(e - p));
                return e;
            }
            p = scan(*text_dfa, p, e, &hit, [this](const char* q, size_t n) { text_out(q, n); });
            if (hit < 0) return p;
            if (P.anywhere && !P.calls) {
                if (hit == 0) { trail = 0; st = St::ThinkOpenWs; }
                else lenient();  /* A reasoning close with no block open. */
                return p;
            }
            if ((size_t)hit < n_open) {
                if (!F.section_open_core.empty() && !section_active) {
                    /* The call-looking tag is ordinary content outside the required wrapper. */
                    const std::string& marker = F.text_markers[(size_t)hit];
                    text_out(marker.data(), marker.size());
                } else {
                    att_at = here(p) - F.openers[(size_t)hit].size();
                    begin_text_attempt(hit);
                }
            } else if ((size_t)hit == F.section_open_hit) {
                open_section();
            } else if ((size_t)hit == F.section_close_hit) {
                if (!section_active) {
                    lenient();
                    const std::string& marker = F.text_markers[(size_t)hit];
                    text_out(marker.data(), marker.size());
                } else {
                    close_section();
                }
            } else if ((size_t)hit == F.think_open_hit) {
                /* A reasoning block opens again, later in the reply. */
                trail = 0;
                st = St::ThinkOpenWs;
            } else if ((size_t)hit == F.think_close_hit) {
                /* Its close with no block open: not part of the answer. */
                lenient();
            }
            return p;

        case St::OpenCoTail:
            if (trail < F.co_trail.size() && *p == F.co_trail[trail]) {
                att.push_back(*p);
                if (++trail == F.co_trail.size()) st = St::OpenWs;
                return p + 1;
            }
            /* call_open's core without its whitespace: a loose opener. */
            loose = trail < F.co_trail.size();
            st = St::OpenWs;
            return p;

        case St::OpenWs:
            if (loose) {
                while (p < e && loose_ws((unsigned char)*p)) att.push_back(*p++);
            } else {
                while (p < e && is_ws((unsigned char)*p)) att.push_back(*p++);
            }
            if (p < e) { lpos = 0; st = St::OpenPrefix; }
            return p;

        case St::OpenPrefix: {
            const char c = *p;
            att.push_back(c);
            if (c != F.f.name_prefix[lpos]) { fail_attempt(); return p + 1; }
            if (++lpos == F.f.name_prefix.size()) { name.clear(); st = St::OpenName; }
            return p + 1;
        }

        case St::OpenName: {
            const char c = *p;
            att.push_back(c);
            name.push_back(c);
            if (F.name_open) return open_name_byte(p);
            const size_t k = F.ns_core.size();
            if (name.size() >= k && name.compare(name.size() - k, k, F.ns_core) == 0) {
                const std::string nm = name.substr(0, name.size() - k);
                auto it = P.by_name.find(nm);
                tool = it == P.by_name.end() ? -1 : (int)it->second;
                if (tool < 0 && (nm.empty() || nm.find('<') != std::string::npos)) {
                    fail_attempt();
                    return p + 1;
                }
                /* The call exists from the moment its opening tag is complete: a reply that ends
                 * right there is a call with no arguments yet. A loose call waits for what
                 * follows the tag. */
                trail = 0;
                lpos = 0;
                lhold.clear();
                tag_end = att.size();
                if (loose && F.object) {
                    /* A call written without its wrapper, in the object family: a declared
                     * tool's name and the object's opening are what make it one. */
                    if (tool < 0) { fail_attempt(); return p + 1; }
                    begin_call(false);
                } else if (loose) st = St::OpenPeek;
                else if (F.ns_trail.empty()) begin_call(true);
                else st = St::OpenTail;
                return p + 1;
            }
            if (name.size() > kMaxLooseName || (c == '<' && !declared_prefix(name)) ||
                (loose && F.object && !declared_prefix(name))) {
                fail_attempt();
                return p + 1;
            }
            return p + 1;
        }

        case St::OpenTail:
            if (trail < F.ns_trail.size() && *p == F.ns_trail[trail]) {
                att.push_back(*p);
                if (++trail == F.ns_trail.size()) begin_call(true);
                return p + 1;
            }
            begin_call(trail == F.ns_trail.size());
            return p;

        case St::OpenPeek: {
            if (lpos == 0) {
                while (p < e && loose_ws((unsigned char)*p) && att.size() <= kMaxLooseHold)
                    att.push_back(*p++);
                if (att.size() > kMaxLooseHold) { fail_attempt(); return p; }
                if (p == e) return p;
            }
            const char c = *p;
            att.push_back(c);
            const std::string& ap = F.f.arg_prefix;
            const std::string& fc = F.end_core;
            const bool a_ok = extends(ap, c);
            const bool f_ok = extends(fc, c);
            if (!a_ok && !f_ok) { fail_attempt(); return p + 1; }
            lhold.push_back(c);
            ++lpos;
            if (a_ok && lpos == ap.size()) {
                lhold.clear();
                st = St::OpenPeekArg;
            } else if (f_ok && lpos == fc.size()) {
                begin_call(false);
                end_call();
            }
            return p + 1;
        }

        case St::OpenPeekArg: {
            /* The first argument's name is where prose that only looks like a call shows itself:
             * a name that runs into a tag is no name. Once the name closes, the call is read by
             * replaying the bytes after the name tag through it. */
            const char c = *p;
            att.push_back(c);
            lhold.push_back(c);
            if (lhold.size() > kMaxLooseName) { fail_attempt(); return p + 1; }
            const std::string* ne = &F.name_end_core;
            auto ends = [&](const std::string& s) {
                return !s.empty() && lhold.size() >= s.size() &&
                       lhold.compare(lhold.size() - s.size(), s.size(), s) == 0;
            };
            if (!ends(*ne) && ends(F.name_end_json_core)) ne = &F.name_end_json_core;
            if (ends(*ne)) {
                const std::string_view nm(lhold.data(), lhold.size() - ne->size());
                if (nm.empty() || nm.find('<') != std::string_view::npos) {
                    fail_attempt();
                    return p + 1;
                }
                const std::string replay = att.substr(tag_end);
                lhold.clear();
                begin_call(false);
                read_again(replay);
            }
            return p + 1;
        }

        case St::Body:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p < e) { lpos = 0; lhold.clear(); st = St::BodyLit; }
            return p;

        case St::BodyLit: {
            const char c = *p;
            const std::string& ap = F.f.arg_prefix;
            const std::string& fc = F.end_core;
            const std::string& cc = F.cc_core;
            const std::string& sc = F.section_close_core;
            const bool a_ok = extends(ap, c);
            const bool f_ok = extends(fc, c);
            const bool c_ok = wrapped && extends(cc, c);
            const bool s_ok = !sc.empty() && extends(sc, c);
            if (!a_ok && !f_ok && !c_ok && !s_ok) {
                lenient();
                st = St::Junk;
                ds = 0;
                hold.clear();
                lhold.push_back(c);
                read_again(lhold);
                lhold.clear();
                return p + 1;
            }
            lhold.push_back(c);
            ++lpos;
            if (a_ok && lpos == ap.size()) { start_arg(); return p + 1; }
            if (f_ok && lpos == fc.size()) { finish_call_strictness(); end_core_done(); return p + 1; }
            if (c_ok && lpos == cc.size()) { end_call_at_close(); return p + 1; }
            if (s_ok && lpos == sc.size()) { abandon_call_at_section_close(); return p + 1; }
            return p + 1;
        }

        case St::ArgName:
            p = scan(loose ? F.arg_name_loose : F.arg_name, p, e, &hit,
                     [this](const char* q, size_t n) {
                         name.append(q, n);
                         if (loose && std::memchr(q, '<', n)) name_lt = true;
                     });
            json_tag = hit == 1;
            if (name_lt) { end_loose_call_at_arg(hit >= 0); return p; }
            if (hit >= 0) { name += name_end(); begin_value(); }
            return p;

        case St::ValText:
            p = scan(loose ? F.value_loose : F.value, p, e, &hit, [this](const char* q, size_t n) {
                val.add(q, n, scratch);
                flush_scratch();
            });
            if (hit >= 0) {
                val.finish(false, scratch);
                if (vmode == ValueOut::String) scratch += '"';
                flush_scratch();
                st = St::Body;
            }
            return p;

        case St::ValJson: {
            for (; p < e; ++p) {
                const unsigned char c = (unsigned char)*p;
                const JsonScan::Res r = js.push(c);
                if (r == JsonScan::More) { vheld.push_back((char)c); continue; }
                if (r == JsonScan::Fail) {
                    /* Not JSON: the argument is its text, read again from its first byte. */
                    vheld.push_back((char)c);
                    start_text_value(ValueOut::Raw);
                    read_again(vheld);
                    vheld.clear();
                    return p + 1;
                }
                vs_pos = 0;
                st = St::ValJsonEnd;
                return p;
            }
            return p;
        }

        case St::ValJsonEnd: {
            const std::string& vs = value_suffix();
            const char c = *p;
            vheld.push_back(c);
            if (c == vs[vs_pos]) {
                if (++vs_pos == vs.size()) {
                    json_value_done(std::string_view(vheld).substr(0, vheld.size() - vs.size()));
                    vheld.clear();
                    st = St::Body;
                }
                return p + 1;
            }
            /* JSON followed by something else: the grammar reads the argument as text, all of
             * it, from its first byte. */
            start_text_value(ValueOut::String);
            read_again(vheld);
            vheld.clear();
            return p + 1;
        }

        case St::EndTail:
            if (trail < F.end_trail.size() && *p == F.end_trail[trail]) {
                if (++trail == F.end_trail.size()) end_call();
                return p + 1;
            }
            /* The close's own whitespace is missing: still the close, read leniently. */
            lenient();
            end_call();
            return p;

        case St::Junk:
            p = scan(F.junk, p, e, &hit, [this](const char* q, size_t n) { text_out(q, n); });
            if (hit == 0) start_arg();
            else if (hit == 1) end_core_done();
            else if (hit >= 0 && (size_t)hit == F.body_section_close_hit)
                abandon_call_at_section_close();
            else if (hit == 2 && wrapped) end_call_at_close();
            else if (hit == 2) text_out(F.cc_core.data(), F.cc_core.size());
            return p;

        case St::AfterEnd:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p < e) {
                lpos = 0;
                lhold.clear();
                st = F.f.call_close.empty() ? St::AfterCall : St::AfterEndLit;
            }
            return p;

        case St::AfterEndLit: {
            const char c = *p;
            if (c == F.f.call_close[lpos]) {
                lhold.push_back(c);
                if (++lpos == F.f.call_close.size()) {
                    lhold.clear();
                    if (loose) after_loose_call();
                    else st = St::AfterCall;
                }
                return p + 1;
            }
            lenient();
            if (loose) after_loose_call();
            else st = St::AfterCall;
            read_again(lhold);
            lhold.clear();
            return p;
        }

        case St::AfterCall:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p < e) { post_call = true; ds = 0; hold.clear(); st = St::Text; }
            return p;

        case St::ObjStart:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p == e) return p;
            if (*p == '{') { ofirst = true; okey = true; st = St::ObjKey; return p + 1; }
            /* No object after the name: the call takes none, and what stands there instead is
             * read as what follows a call. */
            lenient();
            obj_done();
            return p;

        case St::ObjKey: {
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p == e) return p;
            const char c = *p;
            okey = true;
            if (c == '}' || c == ']') { obj_close_bracket(c); return p + 1; }
            if (c == ',') { lenient(); return p + 1; }   /* an empty member */
            if (c == '"') {
                obj_sep();
                json_string_begin();
                return p + 1;
            }
            if (obj_lit_start((unsigned char)c)) { lhold.clear(); st = St::ObjLit; return p; }
            obj_bare_begin();
            return p;
        }

        case St::ObjKeyBare:
            p = scan(F.obj_key, p, e, &hit, [this](const char* q, size_t n) { name.append(q, n); });
            if (hit < 0) return p;
            if (hit <= 2) {
                const std::string_view k = trim_trailing_ws(name);
                if (k.empty()) lenient();
                obj_sep();
                std::string out = json_quote(k);
                name.clear();
                if (hit == 0) {
                    obj_out(out + ":");
                    vpending = true;
                    okey = false;
                    st = St::ObjValue;
                    return p;
                }
                /* A key with no value: it is null. */
                lenient();
                obj_out(out + ":null");
                obj_struct(hit == 1 ? ',' : '}');
                return p;
            }
            /* The call's close or end inside a key: the key had no value, and goes. */
            name.clear();
            lenient();
            ocomma = false;
            if (hit == F.key_cc) end_call_at_close();
            else end_core_done();
            return p;

        case St::ObjColon:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p == e) return p;
            vpending = true;
            okey = false;
            st = St::ObjValue;
            if (*p == ':') { obj_out(":"); return p + 1; }
            lenient();
            obj_out(":");
            return p;

        case St::ObjValue: {
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p == e) return p;
            const char c = *p;
            const bool in_array = ostack.back() == '[';
            okey = false;
            if (c == '{' || c == '[') {
                obj_sep();
                vpending = false;
                obj_out(c == '{' ? "{" : "[");
                ostack.push_back(c);
                ofirst = true;
                okey = c == '{';
                st = c == '{' ? St::ObjKey : St::ObjValue;
                return p + 1;
            }
            if (c == ']' && in_array && ofirst) { obj_close_bracket(c); return p + 1; }
            if (c == '}' || c == ']' || c == ',') {
                /* No value where one was due: a member's is null, and an array's gap is no
                 * element. */
                lenient();
                if (!in_array) {
                    obj_out("null");
                    vpending = false;
                    st = St::ObjAfter;
                    return p;
                }
                if (c == ',') return p + 1;
                obj_close_bracket(c);
                return p + 1;
            }
            if ((c == '"' || c == '\'') && !F.f.object_string_open.empty()) {
                /* Gemma 4 follows such a quote (`"`, or `'` as Python and JavaScript write
                 * strings) with the text verbatim, or with only its quotes escaped, never with
                 * JSON's other escapes. */
                obj_sep();
                vpending = false;
                lenient();
                obj_out("\"");
                qc = c;
                qmode = 0;
                str_begin();
                st = St::ObjQStr;
                return p + 1;
            }
            if (c == '"') {
                obj_sep();
                vpending = false;
                json_string_begin();
                return p + 1;
            }
            if (obj_lit_start((unsigned char)c)) { lhold.clear(); st = St::ObjLit; return p; }
            obj_bare_begin();
            return p;
        }

        case St::ObjLit: {
            const char c = *p;
            lhold.push_back(c);
            bool part = false;
            for (size_t i = 0; i < F.obj_lits.size(); ++i) {
                const std::string& l = F.obj_lits[i];
                if (lhold.size() > l.size() || l.compare(0, lhold.size(), lhold) != 0) continue;
                if (lhold.size() < l.size()) { part = true; continue; }
                lhold.clear();
                obj_lit_done(F.obj_lit_kind[i]);
                return p + 1;
            }
            if (part) return p + 1;
            /* None of them: the bytes begin a bare key or value. */
            std::string again;
            again.swap(lhold);
            obj_bare_begin();
            read_again(again);
            return p + 1;
        }

        case St::ObjStr:
            p = scan(F.obj_str, p, e, &hit, [this](const char* q, size_t n) { str_text(q, n); });
            if (hit < 0) return p;
            if (hit == 0) { str_flush(); obj_string_done(); }
            else cut_begin();
            return p;

        case St::ObjQStr:
            p = scan(qc == '"' ? F.obj_qstr : F.obj_sqstr, p, e, &hit,
                     [this](const char* q, size_t n) { str_text(q, n); });
            if (hit < 0) return p;
            if (hit == 0) { qtail.clear(); st = St::ObjQLook; }
            else if (hit == 1) {
                if (qmode == 0) qmode = 1;
                const char esc[2] = { '\\', qc };
                if (qmode == 1) str_text(&qc, 1);
                else str_text(esc, 2);
            } else if (hit == 2) { lenient(); str_flush(); obj_string_done(); }
            else cut_begin();
            return p;

        case St::ObjQLook: {
            qtail.push_back(*p);
            const int v = q_verdict();
            if (v) q_done(v == 1);
            return p + 1;
        }

        case St::ObjJStr: {
            const char* run = p;
            while (p < e) {
                const unsigned char c = (unsigned char)*p;
                if (vheld.size() == 1) {                /* after the backslash */
                    if (c && std::strchr("\"\\/bfnrt", (int)c)) {
                        vheld.push_back((char)c);
                        jstr_escape_done(vheld);
                        vheld.clear();
                        run = ++p;
                        continue;
                    }
                    if (c == 'u') { vheld.push_back('u'); run = ++p; continue; }
                    /* No JSON escape: the backslash is text, and what follows it is read as
                     * the string's next byte. */
                    lenient();
                    vheld.clear();
                    jstr_char("\\", 1);
                    run = p;
                    continue;
                }
                if (!vheld.empty()) {                   /* a "u" escape's digits */
                    if (std::isxdigit(c)) {
                        vheld.push_back((char)c);
                        if (vheld.size() == 6) { jstr_escape_done(vheld); vheld.clear(); }
                        run = ++p;
                        continue;
                    }
                    lenient();
                    const std::string text = vheld;
                    vheld.clear();
                    jstr_char(text.data(), text.size());
                    run = p;
                    continue;
                }
                if (c != '"' && c != '\\') { ++p; continue; }
                jstr_char(run, (size_t)(p - run));
                ++p;
                if (c == '"') { obj_string_done(); return p; }
                vheld.assign(1, '\\');
                run = p;
            }
            jstr_char(run, (size_t)(p - run));
            return p;
        }

        case St::ObjBare:
            p = scan(F.obj_bare, p, e, &hit, [this](const char* q, size_t n) {
                if (bare_txt.size() < kBareKeep)
                    bare_txt.append(q, std::min(n, kBareKeep - bare_txt.size()));
                val.add(q, n, scratch);
                obj_flush();
            });
            if (hit < 0) return p;
            val.finish(false, scratch);
            obj_flush();
            bare_done();
            if (hit <= 2) { obj_struct(",}]"[hit]); return p; }
            lenient();
            if (hit == F.bare_cc) end_call_at_close();
            else end_core_done();
            return p;

        case St::ObjAfter: {
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p == e) return p;
            const char c = *p;
            if (c == ',' || c == '}' || c == ']') { obj_struct(c); return p + 1; }
            /* A key or a value with no separator before it: one is supplied. */
            lenient();
            ocomma = true;
            ofirst = false;
            okey = ostack.back() == '{';
            st = okey ? St::ObjKey : St::ObjValue;
            return p;
        }

        case St::ObjEnd: {
            if (lpos == 0) {
                while (p < e && is_ws((unsigned char)*p)) ++p;
                if (p == e) return p;
            }
            const char c = *p;
            if (c == F.end_core[lpos]) {
                lhold.push_back(c);
                if (++lpos == F.end_core.size()) { lhold.clear(); lpos = 0; end_core_done(); }
                return p + 1;
            }
            /* The object is not followed by the call's end: the call ends with it all the same,
             * and what stands there is read as what follows a call. */
            lenient();
            std::string again;
            again.swap(lhold);
            lpos = 0;
            end_call();
            read_again(again);
            return p;
        }

        case St::RfWs:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p < e) {
                if (*p == '`') { lpos = 0; lhold.clear(); st = St::RfFence; }
                else { js.reset(); fenced = false; st = St::RfJson; }
            }
            return p;

        case St::RfFence: {
            const char c = *p;
            if (c == kFenceOpen[lpos]) {
                lhold.push_back(c);
                if (++lpos == kFenceOpen.size()) { lhold.clear(); st = St::RfFenceWs; }
                return p + 1;
            }
            lenient();
            st = St::RfRaw;
            read_again(lhold);
            lhold.clear();
            return p;
        }

        case St::RfFenceWs:
            while (p < e && is_ws((unsigned char)*p)) ++p;
            if (p < e) { js.reset(); fenced = true; st = St::RfJson; }
            return p;

        case St::RfJson: {
            const char* run = p;
            for (; p < e; ++p) {
                const JsonScan::Res r = js.push((unsigned char)*p);
                if (r == JsonScan::More) continue;
                text_out(run, (size_t)(p - run));
                if (r == JsonScan::Fail || !fenced) { lenient(); st = St::RfRaw; return p; }
                lpos = 0;
                lhold.clear();
                st = St::RfClose;
                return p;
            }
            text_out(run, (size_t)(p - run));
            return p;
        }

        case St::RfClose: {
            const char c = *p;
            if (c == kFenceClose[lpos]) {
                lhold.push_back(c);
                if (++lpos == kFenceClose.size()) { lhold.clear(); st = St::RfEnd; }
                return p + 1;
            }
            lenient();
            st = St::RfRaw;
            read_again(lhold);
            lhold.clear();
            return p;
        }

        case St::RfEnd:
            lenient();
            st = St::RfRaw;
            return p;

        case St::RfRaw:
            text_out(p, (size_t)(e - p));
            return e;
        }
        return e;
    }

    St after_lead() {
        if (F.f.has_reasoning() && P.opt.extract_reasoning)
            return F.open_core.empty() ? St::Think : St::ThinkOpen;
        return after_no_block();
    }
    /* The reply did not open a reasoning block: its content_open header, when the format has one,
     * is matched from the first byte (the bytes of a failed reasoning opener are read again). */
    St after_no_block() {
        if (F.f.content_open.empty()) return after_think();
        lpos = 0;
        return St::ContentOpen;
    }
    St after_think() const { return P.opt.response_format ? St::RfWs : St::Text; }

    /* An argument opener matched. The generic argument arm exists only for a tool that declares
     * parameters; a tool that declares none takes none. */
    void start_arg() {
        const Tool* t = cur_tool();
        if (t && !t->has_props) lenient();
        name.clear();
        name_lt = false;
        ds = 0;
        hold.clear();
        st = St::ArgName;
    }

    /* The reply ended inside an argument name. A name counts only once closed, except where the
     * format writes nothing between a name and its value prefix (arg_name_suffix empty): there
     * the grammar's opening tag for a declared argument is complete once the declared name is,
     * so `<param name="path` already carries the key. The key is the first declared name, in
     * grammar order, that the text spells followed by at most part of the value prefix. */
    void finish_arg_name() {
        const Tool* t = cur_tool();
        if (!t || !F.f.arg_name_suffix.empty()) return;
        for (const std::string& a : t->arms) {
            if (name.size() < a.size() || name.compare(0, a.size(), a) != 0) continue;
            const std::string_view rest = std::string_view(name).substr(a.size());
            if (rest.size() >= F.name_end.size() ||
                F.name_end.compare(0, rest.size(), rest) != 0)
                continue;
            std::string key = argc > 0 ? "," : "";
            key += json_quote(trim_arg_name(a));
            key += ':';
            args_out(key);
            ++argc;
            return;
        }
    }

    /* call_end matched: a tool with required parameters needed at least one argument first. */
    void finish_call_strictness() {
        const Tool* t = cur_tool();
        if (t && t->has_required && argc == 0) lenient();
    }

    /* ------------------------------------------------------------ driving */

    /* Read `in`, whose first byte is at `at` in the stream. */
    void run(std::string_view in, size_t at) {
        stack.push_back({ in.data(), in.size(), at });
        while (!stack.empty()) {
            Seg s = stack.back();
            stack.pop_back();
            seg_p = s.p;
            seg_at = s.at;
            const char* p = s.p;
            const char* e = s.p + s.n;
            while (p < e) {
                p = step(p, e);
                if (!refeed.empty()) {
                    const size_t now = here(p);
                    if (p < e) stack.push_back({ p, (size_t)(e - p), now });
                    owned.push_back(std::move(refeed));
                    refeed.clear();
                    const size_t n = owned.back().size();
                    stack.push_back({ owned.back().data(), n, now >= n ? now - n : 0 });
                    break;
                }
            }
        }
        owned.clear();
    }

    void feed_tail() {
        if (fed_tail) return;
        fed_tail = true;
        if (!P.tail.empty()) run(P.tail, 0);
        fed = P.tail.size();
    }

    void feed(std::string_view bytes) {
        run(bytes, fed);
        fed += bytes.size();
    }

    /* Would the grammar still be waiting for more, rather than have failed, on what an opener
     * has held? Asked only at the end of the reply. */
    bool open_viable() const {
        switch (st) {
        case St::OpenWs:
        case St::OpenPrefix:
            return true;
        case St::OpenName: {
            if (!F.name_open) return declared_prefix(name);
            size_t len = 0;
            name_stop(&len);
            return declared_prefix(std::string_view(name).substr(0, len));
        }
        case St::OpenTail:
            return tool >= 0;
        default:
            return true;
        }
    }

    /* The reply ended inside a loose opener, so it was never a call: its bytes are content, read
     * again from the second byte (which may hold another opener, and so on). */
    bool in_loose_attempt() const {
        switch (st) {
        case St::OpenWs: case St::OpenPrefix: case St::OpenName: case St::OpenPeek:
        case St::OpenPeekArg:
            return loose;
        default:
            return false;
        }
    }

    void finish_all() {
        /* Reads the bytes a decision queued to be read again. */
        auto rerun = [this] {
            std::string again = std::move(refeed);
            refeed.clear();
            run(again, fed >= again.size() ? fed - again.size() : 0);
        };
        /* The reply ended while a close was being decided: it was one. */
        while (st == St::ObjQLook) {
            q_done(true);
            rerun();
        }
        if (st == St::ObjStr || st == St::ObjQStr) str_flush();
        /* The reply ended inside a header that never completed: its bytes are the reply. */
        while (!F.f.content_open.empty() && !lhold.empty() &&
               (st == St::ContentOpen || st == St::ThinkOpen)) {
            st = st == St::ThinkOpen ? after_no_block() : after_think();
            read_again(lhold);
            lhold.clear();
            rerun();
        }
        while (in_loose_attempt()) {
            fail_attempt();
            rerun();
        }
        switch (st) {
        case St::OpenWs: case St::OpenPrefix: case St::OpenName: case St::OpenTail:
            if (!open_viable()) lenient();
            break;
        case St::BodyLit: {
            const Tool* t = cur_tool();
            const bool args_ok = !t || t->has_props;
            const bool end_ok  = !t || !t->has_required || argc > 0;
            const bool a = args_ok && F.f.arg_prefix.compare(0, lhold.size(), lhold) == 0;
            const bool f = end_ok && F.end_core.compare(0, lhold.size(), lhold) == 0;
            if (t && !a && !f) lenient();
            break;
        }
        case St::ArgName:
            finish_arg_name();
            break;
        case St::ValText:
            /* A partly-matched suffix at the end means the value itself was complete. */
            val.finish(hold.empty(), scratch);
            flush_scratch();
            break;
        case St::ValJson:
            begin_value_out(ValueOut::Raw);
            val.add(vheld.data(), vheld.size(), scratch);
            val.finish(true, scratch);
            flush_scratch();
            break;
        case St::ValJsonEnd:
            json_value_done(std::string_view(vheld).substr(0, vheld.size() - vs_pos));
            break;
        case St::ObjBare:
            /* A bare value the reply was cut off inside: as it arrived. The call is open. */
            val.finish(true, scratch);
            obj_flush();
            break;
        default:
            break;
        }
        if (section_active) {
            lenient();
            section_active = false;
        }
        hold.clear();
        ds = 0;
        if (think_hold_on) flush_think_hold(true);
        if (!think_text) think_pend.clear();
        content.finish(msg.content);
        if (cur >= 0) publish(cur, true);
        publish(0, true);
        publish(1, true);
        publish(2, true);
    }
};

ReplyStream::ReplyStream() = default;
ReplyStream::~ReplyStream() = default;

std::unique_ptr<ReplyStream> ReplyParser::open() const {
    std::unique_ptr<ReplyStream> s(new ReplyStream());
    s->impl_ = std::make_unique<ReplyStream::Impl>(*impl_);
    return s;
}

void ReplyStream::feed(std::string_view bytes, std::vector<ReplyEvent>* out) {
    Impl& m = *impl_;
    if (m.finished) return;
    m.out = out;
    m.feed_tail();
    if (!bytes.empty()) m.feed(bytes);
    if (m.cur >= 0) m.publish(m.cur, false);
    m.out = nullptr;
}

void ReplyStream::finish(std::vector<ReplyEvent>* out) {
    Impl& m = *impl_;
    if (m.finished) return;
    m.out = out;
    m.feed_tail();
    m.finish_all();
    m.finished = true;
    m.out = nullptr;
}

const ReplyMessage& ReplyStream::message() const { return impl_->msg; }

std::string ReplyStream::content_with_open_call(std::string_view reply) const {
    const Impl& m = *impl_;
    if (m.msg.calls.empty() || m.msg.calls.back().closed) return m.msg.content;
    const ReplyCall& c = m.msg.calls.back();
    std::string s(m.msg.content, 0, std::min(c.content_at, m.msg.content.size()));
    s += c.content_held;
    if (c.at < reply.size()) s.append(reply.substr(c.at));
    m.content.trim(s);
    return s;
}

bool ReplyStream::strict() const { return impl_->strict; }

size_t ReplyStream::reread() const { return impl_->reread; }

ReplyHeld ReplyStream::held() const {
    const Impl& m = *impl_;
    ReplyHeld h;
    /* A loose call's first argument name is read into lhold too, and is a tag name: its bytes
     * are in `att` already, counted as structure. */
    const size_t lhold = m.st == St::OpenPeekArg ? 0 : m.lhold.size();
    h.delimiter = m.hold.size() + lhold + m.val.held() + m.content.held_other() +
                  (m.msg.reasoning.size() - m.sent_r) + (m.msg.content.size() - m.sent_c) +
                  (m.msg.calls.empty() ? 0 : m.msg.calls.back().arguments.size() - m.sent_a) +
                  m.srun.size() + m.qtail.size();
    h.space = m.content.held_space() + m.think_pend.size();
    h.structure = m.att.size() + m.name.size() + m.vheld.size() + m.val.held_container() +
                  m.jhi.size();
    return h;
}

ReplyMessage reply_parse(const ReplyParser& p, std::string_view text, bool* strict) {
    std::unique_ptr<ReplyStream> s = p.open();
    s->feed(text, nullptr);
    s->finish(nullptr);
    if (strict) *strict = s->strict();
    return s->message();
}

}  /* namespace rad */
