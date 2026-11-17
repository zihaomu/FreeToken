/* word_scan.cpp -- the Aho-Corasick build behind word_scan.h. */
#include "word_scan.h"

#include <algorithm>

namespace rad {

int WordScanner::build(const std::vector<std::string>& words, std::string* err) {
    for (size_t i = 0; i < words.size(); ++i) {
        if (words[i].empty()) {
            if (err) *err = "trigger word " + std::to_string(i) + " is empty";
            return RAD_E_INVAL;
        }
    }

    /* Byte classes: one per distinct byte the words use, and class 0 for every other byte. */
    std::fill(cls_, cls_ + 256, (uint8_t)0);
    ncls_ = 1;
    for (const std::string& w : words)
        for (unsigned char c : w)
            if (cls_[c] == 0) {
                if (ncls_ == 256) {
                    if (err) *err = "the trigger words use more distinct bytes than a class holds";
                    return RAD_E_INVAL;
                }
                cls_[c] = (uint8_t)ncls_++;
            }

    /* The trie with dense edges; a missing edge is kUnset until the breadth-first pass resolves
     * it through the failure link. */
    constexpr uint32_t kUnset = ~0u;
    next_.assign(ncls_, kUnset);
    out_len_.assign(1, 0u);
    max_len_ = 0;
    for (const std::string& w : words) {
        uint32_t s = 0;
        for (unsigned char c : w) {
            uint32_t& e = next_[(size_t)s * ncls_ + cls_[c]];
            if (e == kUnset) {
                e = (uint32_t)out_len_.size();
                out_len_.push_back(0u);
                next_.resize(next_.size() + ncls_, kUnset);
            }
            s = next_[(size_t)s * ncls_ + cls_[c]];
        }
        out_len_[s] = (uint32_t)w.size();
        max_len_    = std::max<uint32_t>(max_len_, (uint32_t)w.size());
    }

    /* Failure links in breadth-first order, so a state's link is final before a deeper state
     * reads it. A state's longest match is its own word if it ends one, else its failure
     * state's. */
    const uint32_t n_states = (uint32_t)out_len_.size();
    std::vector<uint32_t> fail(n_states, 0u);
    std::vector<uint32_t> queue;
    queue.reserve(n_states);
    for (uint32_t c = 0; c < ncls_; ++c) {
        uint32_t& e = next_[c];
        if (e == kUnset) { e = 0; continue; }
        fail[e] = 0;
        queue.push_back(e);
    }
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const uint32_t s = queue[qi];
        if (out_len_[s] == 0) out_len_[s] = out_len_[fail[s]];
        for (uint32_t c = 0; c < ncls_; ++c) {
            uint32_t& e = next_[(size_t)s * ncls_ + c];
            const uint32_t via_fail = next_[(size_t)fail[s] * ncls_ + c];
            if (e == kUnset) { e = via_fail; continue; }
            fail[e] = via_fail;
            queue.push_back(e);
        }
    }
    return RAD_OK;
}

}  /* namespace rad */
