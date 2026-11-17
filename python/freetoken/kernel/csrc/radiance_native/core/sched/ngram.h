/* ngram.h -- the prompt-lookup drafter's index: where each n-gram of a sequence last occurred.
 *
 * Prompt lookup (Scheduler::draft_ngram) proposes, for a sequence of L tokens, what followed the
 * most recent earlier occurrence of its last n tokens, for n from the longest pattern down to the
 * shortest. Scanning the sequence for that occurrence costs O(L) a step on a miss, which at a
 * 400k-token context is a millisecond of host time per sequence per step. This index answers it in
 * O(n) and is updated incrementally: each token committed adds the n-grams ending at the token
 * before it, so a sequence is indexed once over its lifetime, prompt included.
 *
 * The table is open-addressed and stores no keys. An entry is the end position of the n-gram's
 * latest occurrence and its length n, and its tokens are the sequence's own, compared on every
 * probe, so the index is exact: a hit is a true earlier occurrence, a miss means there is none.
 * One table holds every length; it doubles past half full, so memory is about 8 bytes per token
 * per pattern length (16 MiB for 400k tokens and lengths 2..4), allocated as the sequence grows.
 *
 * The proposal itself is a guess like any drafter's: the verify step emits the target's own
 * tokens whatever it proposes (core/sample/accept.h), so the index changes speed, never text.
 */
#pragma once
#include <cstdint>
#include <vector>

namespace rad {

/* A sequence as prompt ++ output, the two vectors a Request keeps, without copying either. */
struct TokenSeq {
    const int32_t* a = nullptr;
    int64_t        na = 0;
    const int32_t* b = nullptr;
    int64_t        nb = 0;
    int64_t size() const { return na + nb; }
    int32_t operator[](int64_t i) const { return i < na ? a[i] : b[i - na]; }
};

class NgramIndex {
public:
    /* The longest pattern an entry can name (4 bits), and the longest sequence it indexes: past
     * 2^27 tokens the index stops growing and only earlier occurrences are found. */
    enum { kMaxN = 15 };
    static constexpr int64_t kMaxLen = ((int64_t)1 << 27) - 1;

    /* Empties the index and releases its table. `owner` names the sequence it is built for (a
     * request id); a lookup for any other owner starts over. */
    void reset(uint64_t owner = 0);
    uint64_t owner() const { return owner_; }

    /* Proposes up to `depth` tokens into `out` for the sequence `seq`, indexing what it added
     * since the last call. For n = nmax down to nmin: if the last n tokens occurred earlier, the
     * tokens after their latest earlier occurrence; past the sequence's end the proposal goes on
     * with its own tokens, so a pattern that repeats with a period shorter than `depth` is
     * proposed whole. Returns how many were proposed, 0 when no pattern of those lengths recurs.
     *
     * The sequence must only grow between calls under one owner; a shorter one (or other pattern
     * lengths) rebuilds the index. 1 <= nmin <= nmax <= kMaxN.
     *
     * `match`, if given, receives how far the match behind the proposal runs back: the number of
     * tokens before the end that equal those before the occurrence, at least the pattern's n and
     * at most `match_cap` (counted only that far). The verify gate (lookup.h) prices a long match
     * apart from a recurring bigram. */
    int propose(const TokenSeq& seq, int nmin, int nmax, int depth, int32_t* out,
                int* match = nullptr, int match_cap = 0);

    /* The latest end position e < seq.size() - 1 of an earlier occurrence of the sequence's last
     * `n` tokens, or -1. Visible for tests; propose() calls it after indexing. */
    int64_t find_suffix(const TokenSeq& seq, int n) const;

    int64_t indexed() const { return filled_; }   /* n-grams ending before this are in */
    int64_t entries() const { return entries_; }
    size_t  bytes() const { return tab_.capacity() * sizeof(uint32_t); }

private:
    std::vector<uint32_t> tab_;   /* 0 empty, else (end + 1) << 4 | n */
    uint64_t mask_ = 0;
    int64_t  filled_ = 0;         /* every n-gram ending at e < filled_ is indexed */
    int64_t  entries_ = 0;
    uint64_t owner_ = 0;
    int      nmin_ = 0, nmax_ = 0;

    static uint64_t hash(const TokenSeq& s, int64_t end, int n);
    static bool same(const TokenSeq& s, int64_t e1, int64_t e2, int n);
    void insert(const TokenSeq& s, int64_t end, int n);
    void grow(const TokenSeq& s, int64_t want_entries);
    void index_to(const TokenSeq& s, int64_t upto);
};

}  /* namespace rad */
