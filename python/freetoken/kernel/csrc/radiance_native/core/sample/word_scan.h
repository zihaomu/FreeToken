/* word_scan.h -- find the first of a set of literal words in a byte stream that arrives in pieces.
 *
 * A lazy grammar waits for its trigger word (`<tool_call>\n` on the Qwen templates), and every
 * generated token passes through here until it appears. Looking back over the stream would make a
 * long reply quadratic host work, so the scan keeps one integer of state and does one table load
 * per byte; replaying a match needs at most the longest word.
 *
 * Automaton: Aho-Corasick compiled to a full DFA over byte classes. Failure links are built
 * breadth-first and every missing edge is resolved through them at build time, so the scan never
 * follows a failure chain. Bytes in no word share one class, so the table has
 * states x (distinct word bytes + 1) entries: a few hundred for a real trigger set, so it stays
 * in L1.
 *
 * Match reported: the leftmost start among occurrences that complete inside one fed piece. Since
 * the caller stops at the first match, nothing completed earlier, so this equals a search of the
 * whole stream so far. Each state carries the length of the longest word ending there, which is
 * the one that starts earliest.
 */
#pragma once
#include "../rad_internal.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rad {

class WordScanner {
public:
    static constexpr uint64_t npos = ~(uint64_t)0;

    /* Automaton state and bytes fed so far. Plain data, so a speculative window can copy and
     * restore it. */
    struct State {
        uint32_t s   = 0;
        uint64_t off = 0;
    };

    /* Compile `words`. Every word must be non-empty; RAD_E_INVAL names the first that is not.
     * An empty set builds a scanner that never matches. */
    int build(const std::vector<std::string>& words, std::string* err);

    bool     empty()   const { return max_len_ == 0; }
    uint32_t max_len() const { return max_len_; }

    /* Advance over `n` bytes. Returns the stream offset at which the leftmost occurrence that
     * completes within these bytes begins, or npos when none completes. */
    uint64_t feed(State& st, const char* p, size_t n) const {
        uint64_t best = npos;
        uint32_t s    = st.s;
        uint64_t off  = st.off;
        for (size_t i = 0; i < n; ++i) {
            s = next_[(size_t)s * ncls_ + cls_[(uint8_t)p[i]]];
            ++off;
            const uint32_t len = out_len_[s];
            if (len && off - len < best) best = off - len;
        }
        st.s   = s;
        st.off = off;
        return best;
    }

private:
    uint8_t               cls_[256] = {};
    uint32_t              ncls_     = 1;
    uint32_t              max_len_  = 0;
    std::vector<uint32_t> next_{ 0u };      /* [state * ncls_ + class] -> state */
    std::vector<uint32_t> out_len_{ 0u };   /* longest word ending in the state, 0 for none */
};

}  /* namespace rad */
