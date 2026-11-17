/* tokenizer_bpe.h -- the tables and the merge loop under Tokenizer's BPE.
 *
 * The merge is the textbook one every reference implements: repeatedly merge the adjacent pair
 * with the lowest merge rank, leftmost on a tie, until no adjacent pair has a rank. Linear-time
 * encoders (GitHub's `bpe` crate) are exact only for vocabularies whose every token re-encodes to
 * itself, which is not checked; this loop is exact for all of them.
 *
 * Symbols are token ids, and a rank lookup is one probe of an open-addressed table keyed by the
 * id pair. Words of up to kBpeLinearMax symbols are merged by scanning a small array, several
 * times cheaper than a heap for the 4-8 symbol words of real text; longer words use the heap.
 * Both pick the same pair at every step (lowest rank, then lowest position), so they produce the
 * same merges.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace rad {

/* 64-bit hash of a byte string: eight bytes a round, a multiply-xorshift finish. */
inline uint64_t tok_hash(const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 0x9E3779B97F4A7C15ull ^ (n * 0xC2B2AE3D27D4EB4Full);
    while (n >= 8) {
        uint64_t w;
        std::memcpy(&w, p, 8);
        h = (h ^ w) * 0xFF51AFD7ED558CCDull;
        h ^= h >> 32;
        p += 8;
        n -= 8;
    }
    if (n) {
        uint64_t w = 0;
        std::memcpy(&w, p, n);
        h = (h ^ w) * 0xC4CEB9FE1A85EC53ull;
    }
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    return h;
}
inline uint64_t tok_hash(std::string_view s) { return tok_hash(s.data(), s.size()); }

/* Byte string -> id, open-addressed, keys packed in one buffer. First insert wins, as for every
 * lookup in the tokeniser: when two ids spell the same text, the merge table and every published
 * count refer to the lower one. */
class TokTable {
public:
    void    reserve(size_t n);
    bool    insert(std::string_view key, int32_t id);     /* false when the key was present */
    int32_t find(std::string_view key) const;
    size_t  size() const { return n_; }

private:
    struct Slot { uint32_t off = 0, len = 0; int32_t id = -1; uint32_t tag = 0; };
    std::vector<Slot> slots_;
    std::string       keys_;
    size_t            n_ = 0;
    size_t            mask_ = 0;
    void grow();
};

/* (left id, right id) -> (rank, id of the merged text or -1). First insert wins here too: a
 * duplicated merge keeps its lowest rank. */
class MergeTable {
public:
    void reserve(size_t n);
    bool insert(int32_t l, int32_t r, uint32_t rank, int32_t merged);
    /* rank, or UINT32_MAX when the pair does not merge. */
    uint32_t find(int32_t l, int32_t r, int32_t* merged) const {
        if (l < 0 || r < 0 || slots_.empty()) return UINT32_MAX;
        const uint64_t key = ((uint64_t)(uint32_t)l << 32) | (uint32_t)r;
        for (size_t i = mix(key) & mask_;; i = (i + 1) & mask_) {
            const Slot& s = slots_[i];
            if (s.key == key) { *merged = s.merged; return s.rank; }
            if (s.key == kEmpty) return UINT32_MAX;
        }
    }
    size_t size() const { return n_; }

private:
    static constexpr uint64_t kEmpty = ~0ull;
    struct Slot { uint64_t key = kEmpty; uint32_t rank = 0; int32_t merged = -1; };
    static uint64_t mix(uint64_t k) {
        k ^= k >> 33;
        k *= 0xFF51AFD7ED558CCDull;
        k ^= k >> 33;
        return k;
    }
    std::vector<Slot> slots_;
    size_t n_ = 0, mask_ = 0;
    void grow();
};

/* One symbol of a word being merged: a byte span of the word and the id its text spells, or -1
 * when it spells no token (it then merges with nothing). */
struct BpeSym { uint32_t start, len; int32_t id; };

/* Words up to this many symbols are merged by linear scan; longer ones by heap. */
constexpr size_t kBpeLinearMax = 64;

/* Reusable storage for the heap merge's linked symbols and candidates. Long pre-tokens (URLs,
 * base64 and other unbroken runs) would otherwise allocate both arrays for every word in a call. */
struct BpeMergeScratch {
    struct Node { int32_t prev, next; uint32_t ver; bool alive; };
    struct Cand {
        uint32_t rank, left, right, vl, vr;
        int32_t  merged;
        /* std::*_heap is a max-heap: the pair that should come out first compares greatest. */
        bool operator<(const Cand& o) const {
            return rank != o.rank ? rank > o.rank : left > o.left;
        }
    };
    std::vector<Node> nodes;
    std::vector<Cand> heap;
};

/* Merge `syms` in place until no adjacent pair has a rank. `force_heap` lets a test check that
 * both strategies agree. */
void bpe_merge(std::vector<BpeSym>& syms, const MergeTable& merges, BpeMergeScratch& scratch,
               bool force_heap = false);
void bpe_merge(std::vector<BpeSym>& syms, const MergeTable& merges, bool force_heap = false);

}  /* namespace rad */
