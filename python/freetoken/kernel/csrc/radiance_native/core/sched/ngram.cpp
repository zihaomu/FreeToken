/* ngram.cpp -- the prompt-lookup drafter's index. See ngram.h. */
#include "ngram.h"

#include <algorithm>

namespace rad {

void NgramIndex::reset(uint64_t owner) {
    std::vector<uint32_t>().swap(tab_);
    mask_ = 0;
    filled_ = 0;
    entries_ = 0;
    owner_ = owner;
    nmin_ = nmax_ = 0;
}

uint64_t NgramIndex::hash(const TokenSeq& s, int64_t end, int n) {
    uint64_t h = 0x9E3779B97F4A7C15ull * (uint64_t)(n + 1);
    for (int64_t i = end - n + 1; i <= end; ++i) {
        h ^= (uint64_t)(uint32_t)s[i];
        h *= 0xff51afd7ed558ccdull;
        h ^= h >> 32;
    }
    h *= 0xc4ceb9fe1a85ec53ull;
    return h ^ (h >> 29);
}

bool NgramIndex::same(const TokenSeq& s, int64_t e1, int64_t e2, int n) {
    for (int k = 0; k < n; ++k)
        if (s[e1 - k] != s[e2 - k]) return false;
    return true;
}

/* Ends arrive in increasing order, so an n-gram already present is overwritten by a later
 * occurrence: the entry is always the latest. */
void NgramIndex::insert(const TokenSeq& s, int64_t end, int n) {
    const uint32_t v = ((uint32_t)(end + 1) << 4) | (uint32_t)n;
    for (uint64_t i = hash(s, end, n) & mask_;; i = (i + 1) & mask_) {
        const uint32_t t = tab_[(size_t)i];
        if (t == 0) {
            tab_[(size_t)i] = v;
            ++entries_;
            return;
        }
        if ((int)(t & 15u) == n && same(s, (int64_t)(t >> 4) - 1, end, n)) {
            tab_[(size_t)i] = v;
            return;
        }
    }
}

/* At most half full, so a probe ends within a few slots. Doubling rehashes from the sequence's
 * own tokens; amortised over the tokens that filled it, a constant per token. */
void NgramIndex::grow(const TokenSeq& s, int64_t want_entries) {
    uint64_t cap = 1024;
    while ((int64_t)cap < 2 * want_entries) cap <<= 1;
    if (cap <= (uint64_t)tab_.size()) return;
    std::vector<uint32_t> old;
    old.swap(tab_);
    tab_.assign((size_t)cap, 0u);
    mask_ = cap - 1;
    entries_ = 0;
    for (const uint32_t t : old)
        if (t) insert(s, (int64_t)(t >> 4) - 1, (int)(t & 15u));
}

void NgramIndex::index_to(const TokenSeq& s, int64_t upto) {
    if (upto <= filled_) return;
    const int nn = nmax_ - nmin_ + 1;
    grow(s, entries_ + (upto - filled_) * nn);
    for (int64_t e = filled_; e < upto; ++e)
        for (int n = nmin_; n <= nmax_ && n <= e + 1; ++n) insert(s, e, n);
    filled_ = upto;
}

int64_t NgramIndex::find_suffix(const TokenSeq& s, int n) const {
    const int64_t end = s.size() - 1;
    if (tab_.empty() || n < 1 || n > end) return -1;
    for (uint64_t i = hash(s, end, n) & mask_;; i = (i + 1) & mask_) {
        const uint32_t t = tab_[(size_t)i];
        if (t == 0) return -1;
        const int64_t e = (int64_t)(t >> 4) - 1;
        if ((int)(t & 15u) == n && e < end && same(s, e, end, n)) return e;
    }
}

int NgramIndex::propose(const TokenSeq& seq, int nmin, int nmax, int depth, int32_t* out,
                        int* match, int match_cap) {
    const int64_t len = seq.size();
    if (depth <= 0 || len < 2) return 0;
    nmin = std::max(1, std::min(nmin, (int)kMaxN));
    nmax = std::max(nmin, std::min(nmax, (int)kMaxN));
    /* Only n-grams with a token after them are indexed: the last one is the pattern sought. */
    const int64_t upto = std::min(len - 1, kMaxLen);
    if (nmin != nmin_ || nmax != nmax_ || filled_ > upto) {
        const uint64_t own = owner_;
        reset(own);
        nmin_ = nmin;
        nmax_ = nmax;
    }
    index_to(seq, upto);
    for (int n = (int)std::min<int64_t>(nmax, len - 1); n >= nmin; --n) {
        const int64_t e = find_suffix(seq, n);
        if (e < 0) continue;
        for (int d = 0; d < depth; ++d) {
            const int64_t i = e + 1 + d;
            out[d] = i < len ? seq[i] : out[i - len];
        }
        if (match) {
            int m = n;
            while (m < match_cap && e - m >= 0 && seq[e - m] == seq[len - 1 - m]) ++m;
            *match = m;
        }
        return depth;
    }
    return 0;
}

}  /* namespace rad */
