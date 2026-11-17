#include "text/tokenizer_nfc.h"
#include "text/unicode_norm.h"
#include "util/simd.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace rad {

namespace {

/* NFC, applied only where it can change something.
 *
 * UAX #15 quick check: a codepoint with NFC_QC Yes and combining class 0 is a boundary. Nothing
 * before it can compose with it or reorder across it, and nothing it becomes reaches back past
 * it, so normalising the text between two boundaries alone gives the same bytes as normalising
 * the whole text. The scan finds the non-boundary codepoints (NFC_QC No or Maybe, a mark out of
 * canonical order, ill-formed UTF-8) and runs unorm_utf8 only from the boundary before each one
 * to the boundary after it; the rest is copied. The properties are derived from the tables
 * unorm_utf8 uses ("No": NFC changes the codepoint on its own; "Maybe": it can compose with what
 * precedes it), so the scan agrees with that implementation. Invalid bytes are never boundaries,
 * so their replacement happens inside a normalised stretch, exactly as for the whole text. */
class NfcQuick {
public:
    NfcQuick() : not_yes_(kWords, 0), marked_(kWords, 0) {
        for (size_t i = 0; i < k_decomp_rows_n; ++i) {
            const DecompRow& r = k_decomp_rows[i];
            if (r.len_c == 0) continue;
            std::vector<uint32_t> v{ r.cpt };
            unorm_decompose(v, false);
            unorm_compose(v);
            if (v != std::vector<uint32_t>{ r.cpt }) set(not_yes_, r.cpt);
        }
        for (size_t i = 0; i < k_compose_n; ++i) set(not_yes_, k_compose[i].b);
        for (uint32_t c = 0x1161; c <= 0x1175; ++c) set(not_yes_, c);   /* Hangul V, composes after L */
        for (uint32_t c = 0x11A8; c <= 0x11C2; ++c) set(not_yes_, c);   /* Hangul T, composes after LV */
        for (size_t i = 0; i < k_ccc_n; ++i)
            for (uint32_t c = k_ccc[i].first; c <= k_ccc[i].last; ++c) set(marked_, c);
    }

    /* Normalise `text` in place; false when it was already NFC and is untouched. */
    bool apply(std::string& text) const {
        const uint8_t* p = (const uint8_t*)text.data();
        const size_t n = text.size();
        std::string out;
        size_t copied = 0;          /* text[0, copied) is in `out` */
        size_t boundary = 0;        /* start of the last boundary codepoint seen */
        uint32_t last = 0;          /* combining class of the previous codepoint */
        for (size_t i = 0; i < n; ) {
            /* ASCII codepoints are all boundaries: skip a run at vector width. */
            const size_t run = simd::first_non_ascii((const char*)p + i, n - i);
            if (run) { last = 0; boundary = i + run - 1; i += run; continue; }
            uint32_t len, ccc;
            const int k = classify(p + i, n - i, &len, &ccc);
            if (k == kBoundary) { last = 0; boundary = i; i += len; continue; }
            if (k == kKeep && !(ccc != 0 && last > ccc)) { last = ccc; i += len; continue; }
            /* Something NFC may change: normalise from the boundary before it to the next one. */
            size_t end = i + len;
            while (end < n) {
                uint32_t l2, c2;
                if (classify(p + end, n - end, &l2, &c2) == kBoundary) break;
                end += l2;
            }
            const size_t start = std::max(boundary, copied);
            out.append(text, copied, start - copied);
            out += unorm_utf8(text.substr(start, end - start), false, true);
            copied = end;
            boundary = end;
            last = 0;
            i = end;
        }
        if (copied == 0) return false;
        out.append(text, copied, std::string::npos);
        text.swap(out);
        return true;
    }

private:
    enum { kBoundary, kKeep, kChange };
    static constexpr uint32_t kInvalidLen = UINT32_MAX;
    static constexpr size_t kWords = (0x110000 + 63) / 64;
    static void set(std::vector<uint64_t>& b, uint32_t c) { b[c >> 6] |= 1ull << (c & 63); }
    static bool get(const std::vector<uint64_t>& b, uint32_t c) { return (b[c >> 6] >> (c & 63)) & 1; }

    /* kBoundary: a well-formed codepoint with class 0 and NFC_QC Yes. kKeep: well-formed, NFC_QC
     * Yes, a nonzero class (in `ccc`) the caller checks for order. kChange: anything else; an
     * invalid byte reports a length of 1 and kInvalidLen in `ccc`. */
    int classify(const uint8_t* p, size_t n, uint32_t* len, uint32_t* ccc) const {
        const uint32_t b = p[0];
        *ccc = 0;
        if (b < 0x80) { *len = 1; return kBoundary; }
        uint32_t c;
        auto cont = [&](size_t k) { return k < n && (p[k] & 0xC0) == 0x80; };
        if (b >= 0xC2 && b < 0xE0 && cont(1)) {
            c = ((b & 0x1F) << 6) | (p[1] & 0x3F);
            *len = 2;
        } else if (b >= 0xE0 && b < 0xF0 && cont(1) && cont(2)) {
            c = ((b & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
            *len = 3;
            if (c < 0x800 || (c >= 0xD800 && c <= 0xDFFF)) { *len = 1; *ccc = kInvalidLen; return kChange; }
        } else if (b >= 0xF0 && b < 0xF5 && cont(1) && cont(2) && cont(3)) {
            c = ((b & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
            *len = 4;
            if (c < 0x10000 || c > 0x10FFFF) { *len = 1; *ccc = kInvalidLen; return kChange; }
        } else {
            *len = 1;
            *ccc = kInvalidLen;
            return kChange;
        }
        if (get(not_yes_, c)) { *ccc = get(marked_, c) ? unorm_ccc(c) : 0; return kChange; }
        if (!get(marked_, c)) return kBoundary;
        *ccc = unorm_ccc(c);
        return *ccc ? kKeep : kBoundary;
    }

    std::vector<uint64_t> not_yes_;     /* NFC_QC is No or Maybe */
    std::vector<uint64_t> marked_;      /* canonical combining class is not 0 */
};

}  /* namespace */

bool tok_nfc(std::string& text) {
    static const NfcQuick q;
    return q.apply(text);
}

}  /* namespace rad */
