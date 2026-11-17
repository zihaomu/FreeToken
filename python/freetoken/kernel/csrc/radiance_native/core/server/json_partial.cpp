#include "json_partial.h"
#include "util/simd.h"

namespace rad {
namespace server {

namespace {

struct Scanner {
    std::string_view s;
    size_t i = 0;

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }
    bool eof() const { return i >= s.size(); }
    char peek() const { return s[i]; }

    bool lit(std::string_view w) {
        if (s.size() - i < w.size()) return false;
        if (s.compare(i, w.size(), w) != 0) return false;
        i += w.size();
        return true;
    }

    bool str() {
        if (eof() || s[i] != '"') return false;
        ++i;
        while (!eof()) {
            char c = s[i];
            if (c == '\\') {
                if (i + 1 >= s.size()) return false;
                char e = s[i + 1];
                if (e == 'u') {
                    if (i + 5 >= s.size()) return false;
                    i += 6;
                } else {
                    i += 2;
                }
                continue;
            }
            if (c == '"') { ++i; return true; }
            ++i;
        }
        return false;   /* unterminated */
    }

    bool num() {
        size_t start = i;
        if (!eof() && s[i] == '-') ++i;
        size_t d0 = i;
        while (!eof() && s[i] >= '0' && s[i] <= '9') ++i;
        if (i == d0) { i = start; return false; }
        if (!eof() && s[i] == '.') {
            ++i;
            size_t d1 = i;
            while (!eof() && s[i] >= '0' && s[i] <= '9') ++i;
            if (i == d1) { i = start; return false; }
        }
        if (!eof() && (s[i] == 'e' || s[i] == 'E')) {
            ++i;
            if (!eof() && (s[i] == '+' || s[i] == '-')) ++i;
            size_t d2 = i;
            while (!eof() && s[i] >= '0' && s[i] <= '9') ++i;
            if (i == d2) { i = start; return false; }
        }
        return true;
    }

    bool value(int depth) {
        /* Depth cap: the input is model output, and a runaway "[[[[..." would otherwise recurse
         * until the stack overflows. 256 is far past any real tool schema. */
        if (depth > 256) return false;
        ws();
        if (eof()) return false;
        char c = peek();
        if (c == '"') return str();
        if (c == '{') return obj(depth + 1);
        if (c == '[') return arr(depth + 1);
        if (c == 't') return lit("true");
        if (c == 'f') return lit("false");
        if (c == 'n') return lit("null");
        return num();
    }

    bool obj(int depth) {
        ++i;                       /* '{' */
        ws();
        if (eof()) return false;
        if (peek() == '}') { ++i; return true; }
        for (;;) {
            ws();
            if (!str()) return false;
            ws();
            if (eof() || peek() != ':') return false;
            ++i;
            if (!value(depth)) return false;
            ws();
            if (eof()) return false;
            if (peek() == ',') { ++i; continue; }
            if (peek() == '}') { ++i; return true; }
            return false;
        }
    }

    bool arr(int depth) {
        ++i;                       /* '[' */
        ws();
        if (eof()) return false;
        if (peek() == ']') { ++i; return true; }
        for (;;) {
            if (!value(depth)) return false;
            ws();
            if (eof()) return false;
            if (peek() == ',') { ++i; continue; }
            if (peek() == ']') { ++i; return true; }
            return false;
        }
    }
};

}  /* namespace */

bool json_complete(std::string_view s) {
    Scanner sc{s};
    if (!sc.value(0)) return false;
    sc.ws();
    return sc.eof();
}

size_t utf8_trunc(std::string_view s) {
    size_t n = s.size();
    if (n == 0) return 0;
    /* Walk back over at most three continuation bytes to find the lead byte, then decide whether
     * the sequence it starts is complete. */
    size_t k = n;
    size_t back = 0;
    while (k > 0 && back < 4) {
        unsigned char c = (unsigned char)s[k - 1];
        if ((c & 0xC0) == 0x80) { --k; ++back; continue; }
        size_t need = 1;
        if ((c & 0x80) == 0x00)      need = 1;
        else if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        else return n;               /* not a lead byte: not our truncation to fix */
        size_t have = n - (k - 1);
        return have >= need ? n : k - 1;
    }
    return n;
}

size_t json_stable_prefix(std::string_view s) {
    size_t n = utf8_trunc(s);
    if (n == 0) return 0;

    /* Back out of an unfinished escape. Count the run of backslashes immediately before the cut:
     * an odd run means the last one is an opening escape whose payload has not arrived. */
    size_t bs = 0;
    while (bs < n && s[n - 1 - bs] == '\\') ++bs;
    if (bs % 2 == 1) return n - 1;

    /* Back out of an unfinished "\uXXXX". `k` counts the characters after the 'u'; the escape is
     * complete at k == 4. At most six bytes are examined. */
    for (size_t k = 0; k <= 4 && n >= k + 2; ++k) {
        if (s[n - 1 - k] != 'u') continue;
        if (s[n - 2 - k] != '\\') break;
        size_t bs2 = 0, j = n - 2 - k;
        while (j > 0 && s[j - 1] == '\\') { ++bs2; --j; }
        if (bs2 % 2 == 1) break;          /* the backslash was itself escaped */
        if (k < 4) return n - 2 - k;      /* fewer than four hex digits have arrived */
        break;
    }
    return n;
}

bool json_quote_append(std::string& out, std::string_view s) {
    static const char kHex[] = "0123456789abcdef";
    const size_t mark = out.size();
    out.reserve(mark + s.size() + 2);
    out.push_back('"');
    const unsigned char* p = (const unsigned char*)s.data();
    const unsigned char* e = p + s.size();
    const unsigned char* run = p;     /* [run, p) is verified and goes out as it is */
    while (p < e) {
        /* Ordinary bytes go out as they are: jump over them a vector at a time to the next
         * byte that needs an escape or UTF-8 validation. */
        p += simd::first_json_special((const char*)p, (size_t)(e - p));
        if (p >= e) break;
        const unsigned char c = *p;
        if (c >= 0x80) {
            /* Sequence length and the allowed range of the second byte, which rejects overlong
             * forms, surrogates and code points past U+10FFFF without decoding. */
            size_t len;
            unsigned char lo = 0x80, hi = 0xBF;
            if (c >= 0xC2 && c <= 0xDF)      len = 2;
            else if (c == 0xE0)              { len = 3; lo = 0xA0; }
            else if (c >= 0xE1 && c <= 0xEC) len = 3;
            else if (c == 0xED)              { len = 3; hi = 0x9F; }
            else if (c >= 0xEE && c <= 0xEF) len = 3;
            else if (c == 0xF0)              { len = 4; lo = 0x90; }
            else if (c >= 0xF1 && c <= 0xF3) len = 4;
            else if (c == 0xF4)              { len = 4; hi = 0x8F; }
            else                             len = 0;
            bool ok = len != 0 && (size_t)(e - p) >= len && p[1] >= lo && p[1] <= hi;
            for (size_t k = 2; ok && k < len; ++k) ok = p[k] >= 0x80 && p[k] <= 0xBF;
            if (!ok) {
                out.resize(mark);
                return false;
            }
            p += len;
            continue;
        }
        if (c >= 0x20 && c != '"' && c != '\\') {
            ++p;
            continue;
        }
        out.append((const char*)run, (size_t)(p - run));
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\t': out += "\\t"; break;
            case '\n': out += "\\n"; break;
            case '\f': out += "\\f"; break;
            case '\r': out += "\\r"; break;
            default: {
                const char u[6] = { '\\', 'u', '0', '0', kHex[c >> 4], kHex[c & 15] };
                out.append(u, 6);
            }
        }
        run = ++p;
    }
    out.append((const char*)run, (size_t)(p - run));
    out.push_back('"');
    return true;
}

}  /* namespace server */
}  /* namespace rad */
