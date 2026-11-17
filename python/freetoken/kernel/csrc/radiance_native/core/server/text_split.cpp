/* text_split.cpp -- sentence and clause units of text that arrives in pieces (text_split.h). */
#include "text_split.h"

#include <cctype>
#include <cstdint>

namespace rad {
namespace server {

namespace {

/* The code point at `i` and its length in bytes; a byte that starts no valid sequence is read as
 * itself, one byte long, so malformed text still advances. */
uint32_t cp_at(const std::string& s, size_t i, size_t* len) {
    const unsigned char c = (unsigned char)s[i];
    auto cont = [&](size_t k) {
        return i + k < s.size() && ((unsigned char)s[i + k] & 0xC0) == 0x80;
    };
    if (c < 0x80) { *len = 1; return c; }
    if ((c & 0xE0) == 0xC0 && cont(1)) {
        *len = 2;
        return ((uint32_t)(c & 0x1F) << 6) | ((unsigned char)s[i + 1] & 0x3F);
    }
    if ((c & 0xF0) == 0xE0 && cont(1) && cont(2)) {
        *len = 3;
        return ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)((unsigned char)s[i + 1] & 0x3F) << 6) |
               ((unsigned char)s[i + 2] & 0x3F);
    }
    if ((c & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
        *len = 4;
        return ((uint32_t)(c & 0x07) << 18) | ((uint32_t)((unsigned char)s[i + 1] & 0x3F) << 12) |
               ((uint32_t)((unsigned char)s[i + 2] & 0x3F) << 6) | ((unsigned char)s[i + 3] & 0x3F);
    }
    *len = 1;
    return c;
}

bool ends_sentence(uint32_t c) {
    switch (c) {
    case '.': case '!': case '?':
    case 0x3002:   /* 。 */
    case 0xFF01:   /* ！ */
    case 0xFF1F:   /* ？ */
    case 0xFF0E:   /* ． */
    case 0x2026:   /* … */
    case 0x0964:   /* । */
    case 0x0965:   /* ॥ */
    case 0x061F:   /* ؟ */
        return true;
    default:
        return false;
    }
}

bool ends_clause(uint32_t c) {
    switch (c) {
    case ',': case ';':
    case 0xFF0C:   /* ， */
    case 0xFF1B:   /* ； */
    case 0x060C:   /* ، */
    case 0x061B:   /* ؛ */
        return true;
    default:
        return false;
    }
}

/* What may follow a unit's closing punctuation and still belong to it. */
bool closes(uint32_t c) {
    switch (c) {
    case ')': case ']': case '}': case '"': case '\'':
    case 0x201D: case 0x2019: case 0x00BB:              /* ” ’ » */
    case 0x300D: case 0x300F: case 0xFF09: case 0x3011:  /* 」 』 ） 】 */
    case 0x300B: case 0x3009: case 0x3015: case 0xFF3D:  /* 》 〉 〕 ］ */
    case 0xFF5D:                                          /* ｝ */
        return true;
    default:
        return false;
    }
}

bool space_cp(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v' ||
           c == 0x00A0 || c == 0x3000;
}

/* A full stop that ends no sentence: one closing a common abbreviation or an initial. `word` is
 * the text from the last whitespace to the stop, the stop included. */
bool no_stop(std::string word) {
    while (!word.empty() && (word[0] == '(' || word[0] == '"' || word[0] == '\'' ||
                             word[0] == '['))
        word.erase(0, 1);
    std::string low;
    for (char ch : word) low += (char)std::tolower((unsigned char)ch);
    static const char* const kAbbrev[] = {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "mt.", "vs.", "etc.", "e.g.",
        "i.e.", "inc.", "ltd.", "co.", "no.", "fig.", "approx.", "dept.", "est.",
    };
    for (const char* a : kAbbrev)
        if (low == a) return true;
    /* An initial or a run of them: "J." or "J.R.", each an upper-case letter and a stop. */
    if (word.size() >= 2 && word.size() % 2 == 0 && std::isupper((unsigned char)word[0])) {
        bool initials = true;
        for (size_t k = 0; k < word.size(); k += 2)
            initials = initials && std::isalpha((unsigned char)word[k]) && word[k + 1] == '.';
        if (initials) return true;
    }
    return false;
}

std::string trim(const std::string& s, size_t a, size_t b) {
    while (a < b) {
        size_t n = 1;
        if (!space_cp(cp_at(s, a, &n))) break;
        a += n;
    }
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

/* Whether a unit carries anything to say: some character that is no punctuation or space. */
bool says_something(const std::string& u) {
    for (size_t i = 0; i < u.size();) {
        size_t n = 1;
        const uint32_t c = cp_at(u, i, &n);
        i += n;
        if (space_cp(c) || ends_sentence(c) || ends_clause(c) || closes(c)) continue;
        if (c < 0x80 && std::ispunct((int)c)) continue;
        return true;
    }
    return false;
}

}  /* namespace */

std::vector<std::string> take_text_units(std::string* pending, TextSplit mode, bool final) {
    std::vector<std::string> out;
    std::string& s = *pending;
    auto push = [&](std::string u) {
        if (says_something(u)) out.push_back(std::move(u));
    };
    size_t start = 0;
    if (mode != TextSplit::None) {
        size_t i = 0;
        while (i < s.size()) {
            size_t n = 1;
            const uint32_t c = cp_at(s, i, &n);
            const bool term = ends_sentence(c) || (mode == TextSplit::Clause && ends_clause(c));
            if (!term) { i += n; continue; }
            /* The punctuation run: further stops and closing quotes or brackets. */
            size_t j = i + n;
            while (j < s.size()) {
                size_t m = 1;
                const uint32_t d = cp_at(s, j, &m);
                if (!(ends_sentence(d) || ends_clause(d) || closes(d))) break;
                j += m;
            }
            if (j == s.size()) {
                /* Whether the unit ended shows only in what comes next. */
                if (!final) break;
                i = j;
                continue;
            }
            size_t m = 1;
            const uint32_t next = cp_at(s, j, &m);
            const bool ascii = c < 0x80;
            bool boundary = ascii ? space_cp(next) : true;
            if (boundary && c == '.') {
                size_t w = i;
                while (w > start && !std::isspace((unsigned char)s[w - 1])) --w;
                boundary = !no_stop(s.substr(w, i + 1 - w));
            }
            if (!boundary) { i = j; continue; }
            push(trim(s, start, j));
            /* The next unit starts at its first character. */
            while (j < s.size()) {
                size_t k = 1;
                if (!space_cp(cp_at(s, j, &k))) break;
                j += k;
            }
            start = j;
            i = j;
        }
    }
    if (final) {
        push(trim(s, start, s.size()));
        s.clear();
    } else {
        s.erase(0, start);
    }
    return out;
}

}  /* namespace server */
}  /* namespace rad */
