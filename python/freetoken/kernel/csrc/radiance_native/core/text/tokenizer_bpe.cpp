#include "text/tokenizer_bpe.h"

#include <algorithm>

namespace rad {

/* ================================================================== TokTable */

void TokTable::reserve(size_t n) {
    size_t cap = 16;
    while (cap < n * 2) cap <<= 1;
    if (cap <= slots_.size()) return;
    std::vector<Slot> old;
    old.swap(slots_);
    slots_.assign(cap, Slot{});
    mask_ = cap - 1;
    for (const Slot& s : old) {
        if (s.id < 0) continue;
        const uint64_t h = tok_hash(keys_.data() + s.off, s.len);
        size_t i = h & mask_;
        while (slots_[i].id >= 0) i = (i + 1) & mask_;
        slots_[i] = s;
    }
}

void TokTable::grow() { reserve(std::max<size_t>(16, (n_ + 1) * 2)); }

bool TokTable::insert(std::string_view key, int32_t id) {
    if ((n_ + 1) * 2 > slots_.size()) grow();
    const uint64_t h = tok_hash(key);
    const uint32_t tag = (uint32_t)(h >> 32);
    for (size_t i = h & mask_;; i = (i + 1) & mask_) {
        Slot& s = slots_[i];
        if (s.id < 0) {
            s.off = (uint32_t)keys_.size();
            s.len = (uint32_t)key.size();
            s.id = id;
            s.tag = tag;
            keys_.append(key.data(), key.size());
            ++n_;
            return true;
        }
        if (s.tag == tag && s.len == key.size() &&
            std::memcmp(keys_.data() + s.off, key.data(), key.size()) == 0)
            return false;
    }
}

int32_t TokTable::find(std::string_view key) const {
    if (slots_.empty()) return -1;
    const uint64_t h = tok_hash(key);
    const uint32_t tag = (uint32_t)(h >> 32);
    for (size_t i = h & mask_;; i = (i + 1) & mask_) {
        const Slot& s = slots_[i];
        if (s.id < 0) return -1;
        if (s.tag == tag && s.len == key.size() &&
            std::memcmp(keys_.data() + s.off, key.data(), key.size()) == 0)
            return s.id;
    }
}

/* ================================================================== MergeTable */

void MergeTable::reserve(size_t n) {
    size_t cap = 16;
    while (cap < n * 2) cap <<= 1;
    if (cap <= slots_.size()) return;
    std::vector<Slot> old;
    old.swap(slots_);
    slots_.assign(cap, Slot{});
    mask_ = cap - 1;
    for (const Slot& s : old) {
        if (s.key == kEmpty) continue;
        size_t i = mix(s.key) & mask_;
        while (slots_[i].key != kEmpty) i = (i + 1) & mask_;
        slots_[i] = s;
    }
}

void MergeTable::grow() { reserve(std::max<size_t>(16, (n_ + 1) * 2)); }

bool MergeTable::insert(int32_t l, int32_t r, uint32_t rank, int32_t merged) {
    if (l < 0 || r < 0) return false;
    if ((n_ + 1) * 2 > slots_.size()) grow();
    const uint64_t key = ((uint64_t)(uint32_t)l << 32) | (uint32_t)r;
    for (size_t i = mix(key) & mask_;; i = (i + 1) & mask_) {
        Slot& s = slots_[i];
        if (s.key == key) return false;
        if (s.key == kEmpty) {
            s.key = key;
            s.rank = rank;
            s.merged = merged;
            ++n_;
            return true;
        }
    }
}

/* ================================================================== the merge */

namespace {

void merge_linear(std::vector<BpeSym>& syms, const MergeTable& merges) {
    uint32_t rank[kBpeLinearMax];
    int32_t  merged[kBpeLinearMax];
    size_t len = syms.size();
    for (size_t i = 0; i + 1 < len; ++i) rank[i] = merges.find(syms[i].id, syms[i + 1].id, &merged[i]);
    rank[len - 1] = UINT32_MAX;
    for (;;) {
        uint32_t best = UINT32_MAX;
        size_t bi = 0;
        for (size_t i = 0; i + 1 < len; ++i) {
            if (rank[i] < best) { best = rank[i]; bi = i; }
        }
        if (best == UINT32_MAX) break;
        syms[bi].len += syms[bi + 1].len;
        syms[bi].id = merged[bi];
        /* The right symbol and the pair it started are gone; the pairs either side of the new
         * symbol are recomputed. */
        for (size_t k = bi + 1; k + 1 < len; ++k) {
            syms[k] = syms[k + 1];
            rank[k] = rank[k + 1];
            merged[k] = merged[k + 1];
        }
        --len;
        if (bi > 0) rank[bi - 1] = merges.find(syms[bi - 1].id, syms[bi].id, &merged[bi - 1]);
        rank[bi] = bi + 1 < len ? merges.find(syms[bi].id, syms[bi + 1].id, &merged[bi]) : UINT32_MAX;
    }
    syms.resize(len);
}

void merge_heap(std::vector<BpeSym>& syms, const MergeTable& merges, BpeMergeScratch& sc) {
    using Node = BpeMergeScratch::Node;
    using Cand = BpeMergeScratch::Cand;
    const size_t n = syms.size();
    std::vector<Cand>& heap = sc.heap;
    heap.clear();
    /* Bulk-build the initial heap: candidates all carry the nodes' initial version, so they can
     * be collected before the node array exists, and a word with no merging pair returns without
     * allocating or initialising it. make_heap is linear where pushing every pair is n log n. */
    for (size_t i = 0; i + 1 < n; ++i) {
        int32_t m = -1;
        const uint32_t rank = merges.find(syms[i].id, syms[i + 1].id, &m);
        if (rank != UINT32_MAX)
            heap.push_back({ rank, (uint32_t)i, (uint32_t)(i + 1), 0, 0, m });
    }
    if (heap.empty()) return;
    std::make_heap(heap.begin(), heap.end());
    std::vector<Node>& node = sc.nodes;
    node.resize(n);
    for (size_t i = 0; i < n; ++i)
        node[i] = { (int32_t)i - 1, i + 1 < n ? (int32_t)(i + 1) : -1, 0, true };
    auto offer = [&](int32_t l, int32_t r) {
        if (l < 0 || r < 0) return;
        int32_t m = -1;
        const uint32_t rank = merges.find(syms[l].id, syms[r].id, &m);
        if (rank == UINT32_MAX) return;
        heap.push_back({ rank, (uint32_t)l, (uint32_t)r, node[l].ver, node[r].ver, m });
        std::push_heap(heap.begin(), heap.end());
    };
    while (!heap.empty()) {
        const Cand c = heap.front();
        std::pop_heap(heap.begin(), heap.end());
        heap.pop_back();
        Node& l = node[c.left];
        Node& r = node[c.right];
        if (!l.alive || !r.alive || l.ver != c.vl || r.ver != c.vr) continue;
        syms[c.left].len += syms[c.right].len;
        syms[c.left].id = c.merged;
        l.next = r.next;
        if (r.next >= 0) node[r.next].prev = (int32_t)c.left;
        r.alive = false;
        ++r.ver;
        ++l.ver;
        offer(l.prev, (int32_t)c.left);
        offer((int32_t)c.left, l.next);
    }
    size_t w = 0;
    for (size_t i = 0; i < n; ++i) if (node[i].alive) syms[w++] = syms[i];
    syms.resize(w);
}

}  /* namespace */

void bpe_merge(std::vector<BpeSym>& syms, const MergeTable& merges, BpeMergeScratch& scratch,
               bool force_heap) {
    if (syms.size() < 2) return;
    if (syms.size() <= kBpeLinearMax && !force_heap) merge_linear(syms, merges);
    else                                              merge_heap(syms, merges, scratch);
}

void bpe_merge(std::vector<BpeSym>& syms, const MergeTable& merges, bool force_heap) {
    BpeMergeScratch scratch;
    bpe_merge(syms, merges, scratch, force_heap);
}

}  /* namespace rad */
