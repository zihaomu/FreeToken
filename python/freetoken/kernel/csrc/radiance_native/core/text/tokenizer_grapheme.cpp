/* tokenizer_grapheme.cpp -- extended grapheme clusters and the precompiled normaliser
 * (tokenizer_grapheme.h). */
#include "text/tokenizer_grapheme.h"
#include "rad_internal.h"

#include <algorithm>
#include <cstring>

namespace rad {

namespace {

enum Gcb : uint32_t { kOther, kCR, kLF, kControl, kExtend, kZWJ, kRI, kPrepend, kSpacingMark,
                      kL, kV, kT, kLV, kLVT };
enum InCB : uint32_t { kInNone, kInLinker, kInConsonant, kInExtend };

uint32_t lookup(const ucd::GbRange* r, size_t n, uint32_t c) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (r[mid].last < c) lo = mid + 1;
        else hi = mid;
    }
    return lo < n && r[lo].first <= c ? r[lo].v : 0;
}

uint32_t gcb(uint32_t c)  { return lookup(ucd::k_gcb, ucd::k_gcb_n, c); }
bool     pict(uint32_t c) { return lookup(ucd::k_ext_pict, ucd::k_ext_pict_n, c) != 0; }
uint32_t incb(uint32_t c) { return lookup(ucd::k_incb, ucd::k_incb_n, c); }

/* One codepoint off well-formed UTF-8 at `p`, its byte length in *len. A byte that starts no
 * well-formed sequence is taken alone. */
uint32_t utf8_at(const unsigned char* p, size_t left, size_t* len) {
    const unsigned char b = p[0];
    auto cont = [&](size_t k) { return k < left && (p[k] & 0xC0) == 0x80; };
    if (b < 0x80) { *len = 1; return b; }
    if ((b & 0xE0) == 0xC0 && cont(1)) { *len = 2; return ((b & 0x1Fu) << 6) | (p[1] & 0x3Fu); }
    if ((b & 0xF0) == 0xE0 && cont(1) && cont(2)) {
        *len = 3;
        return ((b & 0x0Fu) << 12) | ((p[1] & 0x3Fu) << 6) | (p[2] & 0x3Fu);
    }
    if ((b & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
        *len = 4;
        return ((b & 0x07u) << 18) | ((p[1] & 0x3Fu) << 12) | ((p[2] & 0x3Fu) << 6) |
               (p[3] & 0x3Fu);
    }
    *len = 1;
    return 0xFFFD;
}

int b64_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

}  /* namespace */

/* UAX #29's rules GB3 to GB13 in their order, the lookbehind of GB9c (a conjunct's consonant,
 * linker and consonant), GB11 (an emoji ZWJ sequence) and GB12/13 (regional indicator pairs)
 * carried as state from the cluster's start, where no such sequence can be cut. */
size_t grapheme_end(const uint32_t* cps, size_t n, size_t i) {
    uint32_t prev = gcb(cps[i]);
    /* The text so far ends in Extended_Pictographic Extend*; and did before `prev`. */
    bool ep = pict(cps[i]), ep_before = false;
    /* Regional indicators ending at `prev`. */
    int ri = prev == kRI ? 1 : 0;
    /* GB9c: 0 nothing, 1 a consonant then Extend/Linker without a linker, 2 with one. */
    int conj = 0;
    {
        const uint32_t in = incb(cps[i]);
        conj = in == kInConsonant ? 1 : 0;
    }
    size_t j = i + 1;
    for (; j < n; ++j) {
        const uint32_t c = cps[j], cur = gcb(c);
        bool join;
        if (prev == kCR && cur == kLF) join = true;                                   /* GB3 */
        else if (prev == kControl || prev == kCR || prev == kLF) join = false;         /* GB4 */
        else if (cur == kControl || cur == kCR || cur == kLF) join = false;            /* GB5 */
        else if (prev == kL && (cur == kL || cur == kV || cur == kLV || cur == kLVT)) join = true;
        else if ((prev == kLV || prev == kV) && (cur == kV || cur == kT)) join = true;  /* GB7 */
        else if ((prev == kLVT || prev == kT) && cur == kT) join = true;               /* GB8 */
        else if (cur == kExtend || cur == kZWJ) join = true;                           /* GB9 */
        else if (cur == kSpacingMark) join = true;                                     /* GB9a */
        else if (prev == kPrepend) join = true;                                        /* GB9b */
        else if (conj == 2 && incb(c) == kInConsonant) join = true;                    /* GB9c */
        else if (prev == kZWJ && ep_before && pict(c)) join = true;                    /* GB11 */
        else if (prev == kRI && cur == kRI && (ri & 1)) join = true;                   /* GB12/13 */
        else join = false;
        if (!join) break;
        ep_before = ep;
        ep = pict(c) || (ep && cur == kExtend);
        ri = cur == kRI ? ri + 1 : 0;
        const uint32_t in = incb(c);
        if (in == kInConsonant) conj = 1;
        else if (in == kInLinker && conj >= 1) conj = 2;
        else if (in == kInExtend && conj >= 1) { /* kept */ }
        else conj = 0;
        prev = cur;
    }
    return j;
}

/* ================================================================== the precompiled map */
/* The charsmap is a little-endian u32 byte count of the trie, the trie's u32 units (Darts' double
 * array), then the replacement strings, each NUL-terminated. */
int PrecompiledMap::load(std::string_view b64, std::string* why) {
    std::string raw;
    raw.reserve(b64.size() / 4 * 3);
    uint32_t acc = 0;
    int bits = 0;
    for (char ch : b64) {
        if (ch == '=') break;
        const int v = b64_value(ch);
        if (v < 0) {
            if (why) *why = "the precompiled charsmap is not base64";
            return RAD_E_FORMAT;
        }
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            raw.push_back((char)((acc >> bits) & 0xFF));
        }
    }
    if (raw.size() < 4) {
        if (why) *why = "the precompiled charsmap is shorter than its header";
        return RAD_E_FORMAT;
    }
    uint32_t tbytes = 0;
    std::memcpy(&tbytes, raw.data(), 4);
    if (tbytes % 4 || (size_t)tbytes > raw.size() - 4 || tbytes == 0) {
        if (why) *why = "the precompiled charsmap's trie does not fit it";
        return RAD_E_FORMAT;
    }
    trie_.resize(tbytes / 4);
    std::memcpy(trie_.data(), raw.data() + 4, tbytes);
    blob_.assign(raw.data() + 4 + tbytes, raw.size() - 4 - tbytes);
    return RAD_OK;
}

bool PrecompiledMap::transform(std::string_view s, std::string_view* out) const {
    auto offset = [](uint32_t u) { return (u >> 10) << ((u & (1u << 9)) >> 6); };
    auto label  = [](uint32_t u) { return u & ((1u << 31) | 0xFFu); };
    auto leaf   = [](uint32_t u) { return ((u >> 8) & 1u) != 0; };
    auto value  = [](uint32_t u) { return u & ((1u << 31) - 1); };
    size_t pos = 0;
    uint32_t unit = trie_[0];
    pos ^= offset(unit);
    for (unsigned char c : s) {
        if (c == 0) break;
        pos ^= c;
        if (pos >= trie_.size()) return false;
        unit = trie_[pos];
        if (label(unit) != c) return false;
        pos ^= offset(unit);
        if (pos >= trie_.size()) return false;
        if (leaf(unit)) {
            const size_t at = value(trie_[pos]);
            if (at >= blob_.size()) return false;
            const size_t end = blob_.find('\0', at);
            *out = std::string_view(blob_).substr(at, (end == std::string::npos ? blob_.size()
                                                                                 : end) - at);
            return true;
        }
    }
    return false;
}

/* HuggingFace's Precompiled::normalize: a cluster under six bytes goes whole to its shortest
 * mapped prefix's replacement when it has one; otherwise each character goes to its own, or
 * stays. */
std::string PrecompiledMap::normalize(std::string_view text) const {
    std::vector<uint32_t> cps;
    std::vector<size_t> at;
    cps.reserve(text.size());
    at.reserve(text.size() + 1);
    const unsigned char* p = (const unsigned char*)text.data();
    for (size_t k = 0; k < text.size();) {
        size_t len = 1;
        cps.push_back(utf8_at(p + k, text.size() - k, &len));
        at.push_back(k);
        k += len;
    }
    at.push_back(text.size());
    std::string out;
    out.reserve(text.size());
    std::string_view rep;
    for (size_t i = 0; i < cps.size();) {
        const size_t e = grapheme_end(cps.data(), cps.size(), i);
        const std::string_view g = text.substr(at[i], at[e] - at[i]);
        if (g.size() < 6 && transform(g, &rep)) {
            out.append(rep);
        } else {
            for (size_t k = i; k < e; ++k) {
                const std::string_view ch = text.substr(at[k], at[k + 1] - at[k]);
                if (transform(ch, &rep)) out.append(rep);
                else out.append(ch);
            }
        }
        i = e;
    }
    return out;
}

}  /* namespace rad */
