#include "text/unicode_norm.h"
#include "llama/src/unicode.h"

#include <algorithm>

namespace rad {

/* Hangul syllables are (de)composed arithmetically per UAX #15 instead of with 11172 table rows. */
static constexpr uint32_t k_s_base = 0xAC00, k_l_base = 0x1100, k_v_base = 0x1161,
                          k_t_base = 0x11A7, k_l_count = 19, k_v_count = 21, k_t_count = 28,
                          k_n_count = k_v_count * k_t_count,          /* 588 */
                          k_s_count = k_l_count * k_n_count;          /* 11172 */

uint32_t unorm_ccc(uint32_t cpt) {
    /* Binary search over runs. Class 0, the common case, is not in the table. */
    size_t lo = 0, hi = k_ccc_n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cpt < k_ccc[mid].first)      hi = mid;
        else if (cpt > k_ccc[mid].last)  lo = mid + 1;
        else                             return k_ccc[mid].ccc;
    }
    return 0;
}

static const DecompRow* decomp_row(uint32_t cpt) {
    size_t lo = 0, hi = k_decomp_rows_n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cpt < k_decomp_rows[mid].cpt)      hi = mid;
        else if (cpt > k_decomp_rows[mid].cpt) lo = mid + 1;
        else                                   return &k_decomp_rows[mid];
    }
    return nullptr;
}

/* Canonical ordering: a stable sort of each run of non-starters by combining class. Insertion
 * sort rather than std::stable_sort, because runs are almost always of length 0 or 1. */
static void canonical_order(std::vector<uint32_t>& v) {
    for (size_t i = 1; i < v.size(); ++i) {
        const uint32_t ccc_i = unorm_ccc(v[i]);
        if (ccc_i == 0) continue;
        size_t j = i;
        while (j > 0) {
            const uint32_t ccc_p = unorm_ccc(v[j - 1]);
            if (ccc_p == 0 || ccc_p <= ccc_i) break;
            std::swap(v[j], v[j - 1]);
            --j;
        }
    }
}

void unorm_decompose(std::vector<uint32_t>& cpts, bool compat) {
    std::vector<uint32_t> out;
    out.reserve(cpts.size() + cpts.size() / 4);
    for (uint32_t c : cpts) {
        if (c - k_s_base < k_s_count) {                 /* Hangul syllable */
            const uint32_t s = c - k_s_base;
            out.push_back(k_l_base + s / k_n_count);
            out.push_back(k_v_base + (s % k_n_count) / k_t_count);
            const uint32_t t = s % k_t_count;
            if (t) out.push_back(k_t_base + t);
            continue;
        }
        const DecompRow* r = decomp_row(c);
        /* The pool holds fully expanded sequences, so one lookup suffices (no fixed-point loop).
         * A row with only a canonical decomposition serves the K forms too. */
        const uint32_t off = (compat && r && r->len_k) ? r->off_k : (r ? r->off_c : 0);
        const uint32_t len = (compat && r && r->len_k) ? r->len_k : (r ? r->len_c : 0);
        if (!r || len == 0) { out.push_back(c); continue; }
        for (uint32_t i = 0; i < len; ++i) out.push_back(k_decomp_pool[off + i]);
    }
    canonical_order(out);
    cpts.swap(out);
}

static bool compose_pair(uint32_t a, uint32_t b, uint32_t* out) {
    /* Hangul, the only algorithmic composition, is computed before the table lookup. */
    if (a - k_l_base < k_l_count && b - k_v_base < k_v_count) {
        *out = k_s_base + ((a - k_l_base) * k_v_count + (b - k_v_base)) * k_t_count;
        return true;
    }
    if (a - k_s_base < k_s_count && (a - k_s_base) % k_t_count == 0 &&
        b - k_t_base < k_t_count && b != k_t_base) {
        *out = a + (b - k_t_base);
        return true;
    }
    size_t lo = 0, hi = k_compose_n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        const ComposeRow& r = k_compose[mid];
        if (a < r.a || (a == r.a && b < r.b))      hi = mid;
        else if (a > r.a || (a == r.a && b > r.b)) lo = mid + 1;
        else { *out = r.c; return true; }
    }
    return false;
}

/* Canonical composition, UAX #15 D117. Input must already be in NFD/NFKD order. */
void unorm_compose(std::vector<uint32_t>& cpts) {
    if (cpts.empty()) return;
    std::vector<uint32_t> out;
    out.reserve(cpts.size());
    out.push_back(cpts[0]);
    size_t starter = 0;                 /* index in `out` of the last starter */
    uint32_t last_ccc = unorm_ccc(cpts[0]);
    /* A leading non-starter is never a composition target (the standard treats it as blocked). */
    bool have_starter = (last_ccc == 0);

    for (size_t i = 1; i < cpts.size(); ++i) {
        const uint32_t c = cpts[i];
        const uint32_t ccc = unorm_ccc(c);
        uint32_t composed = 0;
        /* Blocked when a character of equal or greater combining class sits between the starter
         * and this one. */
        const bool blocked = (last_ccc != 0 && last_ccc >= ccc) || (ccc == 0 && last_ccc != 0);
        if (have_starter && !blocked && compose_pair(out[starter], c, &composed)) {
            out[starter] = composed;
            continue;                   /* last_ccc unchanged: the pair collapsed */
        }
        out.push_back(c);
        if (ccc == 0) { starter = out.size() - 1; have_starter = true; }
        last_ccc = ccc;
    }
    cpts.swap(out);
}

std::string unorm_utf8(const std::string& in, bool compat, bool compose) {
    /* ASCII is unchanged by every normalisation form and is most of every prompt, so pure-ASCII
     * text skips the decode/normalise/encode round trip. */
    bool ascii = true;
    for (unsigned char ch : in) { if (ch >= 0x80) { ascii = false; break; } }
    if (ascii) return in;

    std::vector<uint32_t> cpts = unicode_cpts_from_utf8(in);
    unorm_decompose(cpts, compat);
    if (compose) unorm_compose(cpts);
    std::string out;
    out.reserve(in.size());
    for (uint32_t c : cpts) out += unicode_cpt_to_utf8(c);
    return out;
}

}  /* namespace rad */
