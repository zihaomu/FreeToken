/* util/json.cpp -- rad::Json's text half: yyjson reads, this walks and writes.
 *
 * Parse converts yyjson's tree value for value (see util/json.h for the model being kept).
 * The walk is iterative -- a pre-order collect, then a bottom-up build -- so a deeply nested
 * document fails its checks instead of the stack. The bottom-up order doubles as the
 * counting order parse_bounded promises: scalars as they come, a container after its
 * children, matching the event order the predecessor's callback parser counted in.
 *
 * dump() writes the predecessor's bytes with one known exception: floats are shortest-round-trip
 * spellings (std::to_chars), where the predecessor's grisu2 occasionally prints one digit more for
 * the same bits. Same value on every re-read, sometimes a shorter spelling. Everything else
 * matches: compact separators, the same escapes, ".0" kept on integral floats, non-finite
 * floats as null, and invalid UTF-8 refused (or replaced, one U+FFFD per ill-formed maximal
 * subpart, as the predecessor's DFA places them). Error TEXTS are not the predecessor's; nothing may
 * parse them.
 */
#include "util/json.h"

#include "util/simd.h"

#include <yyjson/yyjson.h>

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <ostream>
#include <streambuf>
#include <unordered_map>

namespace rad {
namespace {

/* ------------------------------------------------------------------ allocation budget
 * As in core/server's bounded read before it moved here: yyjson allocates an input copy
 * plus one value buffer grown by halves, and the cap covers both at their worst for any
 * body the value budget accepts. Refusing past the cap reads as a parse failure. */
struct YyBudget {
    size_t live = 0;
    size_t cap = 0;
};

void* yy_malloc(void* ctx, size_t size) {
    YyBudget* b = static_cast<YyBudget*>(ctx);
    if (size > b->cap - b->live || size > SIZE_MAX - 16) return nullptr;
    void* raw = std::malloc(size + 16);
    if (!raw) return nullptr;
    *static_cast<size_t*>(raw) = size;
    b->live += size;
    return static_cast<char*>(raw) + 16;
}

void* yy_realloc(void* ctx, void* ptr, size_t /*old_size*/, size_t size) {
    if (!ptr) return yy_malloc(ctx, size);
    YyBudget* b = static_cast<YyBudget*>(ctx);
    char* raw = static_cast<char*>(ptr) - 16;
    const size_t old = *reinterpret_cast<size_t*>(raw);
    if (size > old && size - old > b->cap - b->live) return nullptr;
    if (size > SIZE_MAX - 16) return nullptr;
    void* grown = std::realloc(raw, size + 16);
    if (!grown) return nullptr;
    *static_cast<size_t*>(grown) = size;
    b->live = b->live - old + size;
    return static_cast<char*>(grown) + 16;
}

void yy_free(void* ctx, void* ptr) {
    if (!ptr) return;
    YyBudget* b = static_cast<YyBudget*>(ctx);
    char* raw = static_cast<char*>(ptr) - 16;
    b->live -= *reinterpret_cast<size_t*>(raw);
    std::free(raw);
}

size_t yy_cap(size_t body, int64_t budget) {
    __int128 cap = (__int128)body * 4 + (__int128)budget * 48 + (1 << 20);
    return cap > (__int128)SIZE_MAX ? SIZE_MAX : (size_t)cap;
}

Json yy_scalar(yyjson_val* v) {
    switch (yyjson_get_type(v)) {
    case YYJSON_TYPE_NULL: return nullptr;
    case YYJSON_TYPE_BOOL: return yyjson_get_bool(v);
    case YYJSON_TYPE_STR: return std::string(yyjson_get_str(v), yyjson_get_len(v));
    case YYJSON_TYPE_NUM:
        switch (yyjson_get_subtype(v)) {
        case YYJSON_SUBTYPE_UINT: return yyjson_get_uint(v);
        case YYJSON_SUBTYPE_SINT: return yyjson_get_sint(v);
        default: return yyjson_get_real(v);
        }
    default: return nullptr;
    }
}

/* Convert, counting against budget (containers after their children) and depth. Returns
 * discarded when either bound trips after setting the matching flag. */
Json yy_convert(yyjson_val* root, int64_t budget, int max_depth, bool* over, bool* deep) {
    *over = false;
    *deep = false;
    struct Frame {
        yyjson_val* v;
        int depth;
    };
    std::vector<Frame> order;
    std::vector<Frame> stack;
    stack.push_back({ root, 0 });
    while (!stack.empty()) {
        const Frame f = stack.back();
        stack.pop_back();
        if (f.depth > max_depth) *deep = true;
        order.push_back(f);
        const yyjson_type t = yyjson_get_type(f.v);
        if (t == YYJSON_TYPE_ARR) {
            size_t idx, max;
            yyjson_val* el;
            yyjson_arr_foreach(f.v, idx, max, el) stack.push_back({ el, f.depth + 1 });
        } else if (t == YYJSON_TYPE_OBJ) {
            size_t idx, max;
            yyjson_val *k, *val;
            yyjson_obj_foreach(f.v, idx, max, k, val) stack.push_back({ val, f.depth + 1 });
        }
    }
    if (*deep) return Json(Json::value_t::discarded);

    int64_t n = 0;
    std::vector<Json> st;
    st.reserve(64);
    for (size_t i = order.size(); i-- > 0;) {
        yyjson_val* v = order[i].v;
        const yyjson_type t = yyjson_get_type(v);
        if (t != YYJSON_TYPE_ARR && t != YYJSON_TYPE_OBJ) {
            if (++n > budget) {
                *over = true;
                return Json(Json::value_t::discarded);
            }
            st.push_back(yy_scalar(v));
            continue;
        }
        const size_t k = t == YYJSON_TYPE_ARR ? yyjson_arr_size(v) : yyjson_obj_size(v);
        /* Children sit on the stack in reverse document order. */
        const size_t base = st.size() - k;
        Json d;
        if (t == YYJSON_TYPE_ARR) {
            Json::array_t a;
            a.reserve(k);
            for (size_t c = 0; c < k; ++c) a.push_back(std::move(st[base + c]));
            d = Json(std::move(a));
        } else {
            /* Append members, never find-then-insert per member: operator[] over the
             * ordered vector is a linear scan, which made a flat object quadratic to
             * parse (a 262k-member tokenizer vocab took 27 s). A duplicate key keeps
             * its first position and takes the last value, as operator[] would have
             * left it; small objects find duplicates by scanning what was appended,
             * large ones through a hash of key views (the yyjson document outlives
             * this walk). */
            Json::object_t o;
            o.reserve(k);
            size_t idx, max, c = 0;
            yyjson_val *key, *val;
            if (k <= 48) {
                yyjson_obj_foreach(v, idx, max, key, val) {
                    Json child = std::move(st[base + c++]);
                    const std::string_view kv(yyjson_get_str(key), yyjson_get_len(key));
                    size_t at = o.size();
                    for (size_t m = 0; m < o.size(); ++m)
                        if (o[m].first == kv) { at = m; break; }
                    if (at < o.size()) o[at].second = std::move(child);
                    else o.emplace_back(std::string(kv), std::move(child));
                }
            } else {
                std::unordered_map<std::string_view, size_t> seen;
                seen.reserve(k * 2);
                yyjson_obj_foreach(v, idx, max, key, val) {
                    Json child = std::move(st[base + c++]);
                    const std::string_view kv(yyjson_get_str(key), yyjson_get_len(key));
                    const auto [it, fresh] = seen.try_emplace(kv, o.size());
                    if (fresh) o.emplace_back(std::string(kv), std::move(child));
                    else o[it->second].second = std::move(child);
                }
            }
            d = Json(std::move(o));
        }
        st.resize(base);
        if (++n > budget) {
            *over = true;
            return Json(Json::value_t::discarded);
        }
        st.push_back(std::move(d));
    }
    return std::move(st.back());
}

struct YyDoc {
    YyBudget b;
    yyjson_doc* doc = nullptr;
    yyjson_read_err err{};
    explicit YyDoc(std::string_view text, int64_t budget, bool comments) {
        b.cap = yy_cap(text.size(), budget);
        yyjson_alc alc{ yy_malloc, yy_realloc, yy_free, &b };
        yyjson_read_flag flg = YYJSON_READ_ALLOW_BOM;
        if (comments) flg |= YYJSON_READ_ALLOW_COMMENTS;
        doc = yyjson_read_opts(const_cast<char*>(text.data()), text.size(), flg, &alc, &err);
    }
    ~YyDoc() {
        if (doc) yyjson_doc_free(doc);
    }
    YyDoc(const YyDoc&) = delete;
    YyDoc& operator=(const YyDoc&) = delete;
};

}  /* namespace */

Json Json::parse_bounded(std::string_view text, int64_t budget, int max_depth, bool* over,
                         bool* deep) {
    YyDoc d(text, budget, false);
    *over = false;
    *deep = false;
    if (!d.doc) return Json(value_t::discarded);
    Json out = yy_convert(yyjson_doc_get_root(d.doc), budget, max_depth, over, deep);
    if (*over || *deep) return Json();
    return out;
}

Json Json::parse(std::string_view text) { return parse(text, nullptr, true, false); }

Json Json::parse(std::istream& in) {
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return parse(std::string_view(text));
}

Json Json::parse(std::istream& in, std::nullptr_t cb, bool allow_exceptions,
                 bool ignore_comments) {
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return parse(std::string_view(text), cb, allow_exceptions, ignore_comments);
}

std::istream& operator>>(std::istream& in, Json& j) {
    j = Json::parse(in);
    return in;
}

std::ostream& operator<<(std::ostream& out, const Json& j) { return out << j.dump(); }

Json Json::parse(std::string_view text, std::nullptr_t, bool allow_exceptions,
                 bool ignore_comments) {
    YyDoc d(text, INT64_MAX, ignore_comments);
    if (!d.doc) {
        if (!allow_exceptions) return Json(value_t::discarded);
        std::string msg = "json: parse error at byte ";
        msg += std::to_string((unsigned long long)d.err.pos);
        if (d.err.msg) {
            msg += ": ";
            msg += d.err.msg;
        }
        throw exception(msg);
    }
    bool over = false, deep = false;
    Json out = yy_convert(yyjson_doc_get_root(d.doc), INT64_MAX, INT32_MAX, &over, &deep);
    return out;
}

bool Json::accept(std::string_view text, bool ignore_comments) {
    YyDoc d(text, INT64_MAX, ignore_comments);
    return d.doc != nullptr;
}

/* ------------------------------------------------------------------ dump */

namespace {

/* Well-formed UTF-8 per the Unicode table the predecessor's DFA encodes (restricted second byte
 * after E0/ED/F0/F4). utf8_expected gives a lead byte's sequence length, 0 when the byte
 * cannot lead; utf8_second_ok applies the restriction; utf8_next decodes one sequence.
 * On failure nothing is consumed by these helpers -- the caller decides how much of the
 * ill-formed maximal subpart to skip (lead plus structurally valid continuations), which
 * is where the predecessor places its single replacement character. */
int utf8_expected(unsigned char c) {
    if (c >= 0xC2 && c <= 0xDF) return 2;
    if (c >= 0xE0 && c <= 0xEF) return 3;
    if (c >= 0xF0 && c <= 0xF4) return 4;
    return 0;
}

bool utf8_second_ok(unsigned char lead, unsigned char b) {
    if ((b & 0xC0) != 0x80) return false;
    switch (lead) {
    case 0xE0: return b >= 0xA0;
    case 0xED: return b <= 0x9F;
    case 0xF0: return b >= 0x90;
    case 0xF4: return b <= 0x8F;
    default: return true;
    }
}

bool utf8_next(const unsigned char* s, size_t left, uint32_t* cp, size_t* len) {
    const int want = utf8_expected(s[0]);
    if (want == 0 || left < (size_t)want) return false;
    if (!utf8_second_ok(s[0], s[1])) return false;
    uint32_t v = s[0] & (want == 2 ? 0x1Fu : want == 3 ? 0x0Fu : 0x07u);
    for (int i = 1; i < want; ++i) {
        if ((s[i] & 0xC0) != 0x80) return false;
        v = (v << 6) | (s[i] & 0x3Fu);
    }
    *cp = v;
    *len = (size_t)want;
    return true;
}

/* Bytes to skip past an ill-formed start: the lead and any continuations that belong to
 * its maximal subpart, so one replacement covers E2 82 (truncated) but ED (bad second
 * byte range) stops before A0, which stands on its own. */
size_t utf8_skip_ill(const unsigned char* s, size_t left) {
    const int want = utf8_expected(s[0]);
    if (want == 0) return 1;
    size_t n = 1;
    if (n < left && utf8_second_ok(s[0], s[n])) {
        ++n;
        while (n < left && n < (size_t)want && (s[n] & 0xC0) == 0x80) ++n;
    }
    return n;
}

struct Dumper {
    std::string out;
    int indent;  /* <0 compact */
    char indent_char;
    bool ensure_ascii;
    Json::error_handler_t eh;

    void newline_indent(int level) {
        if (indent < 0) return;
        out += '\n';
        out.append((size_t)(indent * level), indent_char);
    }

    void write_string(const std::string& s) {
        out += '"';
        const unsigned char* p = (const unsigned char*)s.data();
        const size_t n = s.size();
        size_t i = 0;
        while (i < n) {
            size_t stop = i + simd::first_json_special(s.data() + i, n - i);
            if (ensure_ascii) {
                /* DEL is escaped too there, as the predecessor escaped every code point from
                 * 0x7f up; first_json_special passes it. */
                const void* del = std::memchr(s.data() + i, 0x7f, stop - i);
                if (del) stop = (size_t)((const char*)del - s.data());
            }
            out.append(s.data() + i, stop - i);
            i = stop;
            if (i >= n) break;
            const unsigned char c = p[i];
            if (c < 0x80) {
                switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default: {
                    char buf[8];
                    snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                    break;
                }
                }
                ++i;
                continue;
            }
            uint32_t cp = 0;
            size_t len = 0;
            if (!utf8_next(p + i, n - i, &cp, &len)) {
                if (eh == Json::error_handler_t::strict)
                    throw Json::exception("json: dump() found invalid UTF-8");
                if (eh == Json::error_handler_t::replace)
                    out += ensure_ascii ? "\\ufffd" : "\xef\xbf\xbd";
                i += utf8_skip_ill(p + i, n - i);
                continue;
            }
            if (ensure_ascii) {
                char buf[16];
                if (cp <= 0xFFFF) {
                    snprintf(buf, sizeof buf, "\\u%04x", cp);
                    out += buf;
                } else {
                    const uint32_t v = cp - 0x10000;
                    snprintf(buf, sizeof buf, "\\u%04x\\u%04x", 0xD800 + (v >> 10),
                             0xDC00 + (v & 0x3FF));
                    out += buf;
                }
            } else {
                out.append((const char*)p + i, len);
            }
            i += len;
        }
        out += '"';
    }

    void write_double(double d) { json_append_double(out, d); }

    void write(const Json& j, int level) {
        switch (j.type()) {
        case Json::value_t::null: out += "null"; return;
        case Json::value_t::boolean: out += j.get<bool>() ? "true" : "false"; return;
        case Json::value_t::number_integer: {
            char buf[24];
            const auto r = std::to_chars(buf, buf + sizeof buf, j.get<int64_t>());
            out.append(buf, r.ptr);
            return;
        }
        case Json::value_t::number_unsigned: {
            char buf[24];
            const auto r = std::to_chars(buf, buf + sizeof buf, j.get<uint64_t>());
            out.append(buf, r.ptr);
            return;
        }
        case Json::value_t::number_float: write_double(j.get<double>()); return;
        case Json::value_t::string: {
            write_string(j.get_ref<const std::string&>());
            return;
        }
        case Json::value_t::array: {
            if (j.empty()) {
                out += "[]";
                return;
            }
            out += '[';
            bool first = true;
            for (const Json& e : j) {
                if (!first) out += ',';
                first = false;
                newline_indent(level + 1);
                write(e, level + 1);
            }
            newline_indent(level);
            out += ']';
            return;
        }
        case Json::value_t::object: {
            if (j.empty()) {
                out += "{}";
                return;
            }
            out += '{';
            bool first = true;
            for (auto it = j.begin(); it != j.end(); ++it) {
                if (!first) out += ',';
                first = false;
                newline_indent(level + 1);
                write_string(it.key());
                out += indent < 0 ? ":" : ": ";
                write(*it, level + 1);
            }
            newline_indent(level);
            out += '}';
            return;
        }
        case Json::value_t::discarded:
            throw Json::exception("json: cannot dump a discarded value");
        }
    }
};

}  /* namespace */

namespace {

/* The shortest digits that read back as d in its own type, laid out as nlohmann's dump laid
 * them out: fixed while the decimal point falls in (-4, 15] places from the first digit
 * (100000.0, 0.0001), else d.ddde+XX with at least two exponent digits (1e+16, 1.5e-05), and
 * ".0" on an integral fixed value. Schemas and tool arguments are dumped into grammars and
 * prompts, so the spelling of a number is part of what a model is given. */
template <typename T>
void append_shortest(std::string& out, T d) {
    if (!std::isfinite(d)) {
        out += "null";
        return;
    }
    if (d == 0) {
        out += std::signbit(d) ? "-0.0" : "0.0";
        return;
    }
    char buf[40];
    const auto r = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);
    const char* p = buf;
    const char* end = r.ptr;
    /* The spelling is laid out in o and appended once: a value is a handful of bytes, and one
     * append a value is what an embeddings array of hundreds of thousands of them pays. */
    char o[48];
    int w = 0;
    if (*p == '-') {
        o[w++] = '-';
        ++p;
    }
    char dig[24];
    int k = 0;
    for (; p < end && *p != 'e'; ++p)
        if (*p != '.') dig[k++] = *p;
    /* p at 'e' (to_chars' scientific form always has one), then a sign and the digits; the
     * buffer is not NUL-terminated, so the exponent is read up to end only. */
    const char* q = p + 1;
    const bool neg_e = q < end && *q == '-';
    if (q < end && (*q == '-' || *q == '+')) ++q;
    int e = 0;
    for (; q < end; ++q) e = e * 10 + (*q - '0');
    if (neg_e) e = -e;
    const int n = e + 1;              /* the decimal point's place after the first digit */
    if (k <= n && n <= 15) {
        std::memcpy(o + w, dig, (size_t)k);
        w += k;
        std::memset(o + w, '0', (size_t)(n - k));
        w += n - k;
        o[w++] = '.';
        o[w++] = '0';
    } else if (0 < n && n <= 15) {
        std::memcpy(o + w, dig, (size_t)n);
        w += n;
        o[w++] = '.';
        std::memcpy(o + w, dig + n, (size_t)(k - n));
        w += k - n;
    } else if (-4 < n && n <= 0) {
        o[w++] = '0';
        o[w++] = '.';
        std::memset(o + w, '0', (size_t)-n);
        w += -n;
        std::memcpy(o + w, dig, (size_t)k);
        w += k;
    } else {
        o[w++] = dig[0];
        if (k > 1) {
            o[w++] = '.';
            std::memcpy(o + w, dig + 1, (size_t)(k - 1));
            w += k - 1;
        }
        o[w++] = 'e';
        o[w++] = e < 0 ? '-' : '+';
        const int a = e < 0 ? -e : e;
        if (a >= 100) o[w++] = (char)('0' + a / 100);
        o[w++] = (char)('0' + a / 10 % 10);
        o[w++] = (char)('0' + a % 10);
    }
    out.append(o, (size_t)w);
}

}  /* namespace */

void json_append_double(std::string& out, double d) { append_shortest(out, d); }

void json_append_float(std::string& out, float f) { append_shortest(out, f); }

std::string Json::dump(int indent, char indent_char, bool ensure_ascii,
                       error_handler_t eh) const {
    Dumper d;
    d.out.reserve(256);
    d.indent = indent;
    d.indent_char = indent_char;
    d.ensure_ascii = ensure_ascii;
    d.eh = eh;
    d.write(*this, 0);
    return std::move(d.out);
}

}  /* namespace rad */
