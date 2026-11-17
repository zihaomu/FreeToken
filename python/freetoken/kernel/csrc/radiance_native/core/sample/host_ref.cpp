/* host_ref.cpp -- llama.cpp's sampling chain, ported to plain C++ over a float array.
 *
 * Source: llama.cpp `src/llama-sampler.cpp` at upstream commit 06938ac12 (MIT). Ported rather
 * than linked, so the oracle carries no second engine as a dependency. host_ref.h lists the two
 * deliberate deviations: chain order and RNG.
 */
#include "host_ref.h"

#include "rad_sample.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace rad {

/* ================================================================== the candidate set */

void CandidateSet::from_logits(const float* logits, int64_t n_vocab) {
    c.resize((size_t)n_vocab);
    for (int64_t i = 0; i < n_vocab; ++i) {
        c[(size_t)i].id    = (int32_t)i;
        c[(size_t)i].logit = logits[i];
        c[(size_t)i].p     = 0.0f;
    }
    sorted = false;
    selected = -1;
}

/* Upstream sorts on the logit alone, leaving ties in arbitrary order; on the device that order
 * depends on the block size. The id tie-break makes it a total order both sides share. */
void CandidateSet::sort_desc() {
    if (sorted) return;
    std::stable_sort(c.begin(), c.end(), [](const Candidate& a, const Candidate& b) {
        if (a.logit != b.logit) return a.logit > b.logit;
        return a.id < b.id;
    });
    sorted = true;
}

/* ================================================================== reproducible randomness */

/* The stream's key from abi/rad_sample.h. An unknown stream gets a key of its own, never the
 * pick's. */
static uint64_t stream_key(uint32_t stream) {
    switch (stream) {
        case RAD_RNG_PICK:   return RAD_SAMPLE_KEY_PICK;
        case RAD_RNG_XTC:    return RAD_SAMPLE_KEY_XTC;
        default:             return rad_sample_splitmix64((uint64_t)stream);
    }
}

float rng_uniform(uint64_t seed, uint64_t pos, uint32_t stream) {
    /* 24 bits: 32 bits divided by 2^32 can round to 1.0f, which walks off the end of an inverse
     * CDF. Calls rad_sample_uniform01 itself so host and device cannot drift apart. */
    return rad_sample_uniform01(seed ^ stream_key(stream), pos);
}

/* ================================================================== individual samplers */

void host_softmax(CandidateSet& s, bool do_sort) {
    if (s.c.empty()) return;
    if (do_sort) s.sort_desc();

    float max_l = s.c[0].logit;
    if (!s.sorted)
        for (size_t i = 1; i < s.c.size(); ++i) max_l = std::max(max_l, s.c[i].logit);

    double cum = 0.0;
    for (auto& e : s.c) {
        const float p = std::exp(e.logit - max_l);
        e.p = p;
        cum += p;
    }
    if (cum <= 0.0) return;   /* everything masked; p stays zero and the pick reports it */
    for (auto& e : s.c) e.p = (float)(e.p / cum);
}

/* Index of the highest logit, ties to the lowest id whatever the set's order, as the argmax
 * kernel breaks them. The set must not be empty. */
static size_t best_index(const CandidateSet& s) {
    size_t best = 0;
    for (size_t i = 1; i < s.c.size(); ++i) {
        if (s.c[i].logit > s.c[best].logit) best = i;
        else if (s.c[i].logit == s.c[best].logit && s.c[i].id < s.c[best].id) best = i;
    }
    return best;
}

void host_temp(CandidateSet& s, float temp) {
    if (s.c.empty()) return;

    /* temp <= 0 is greedy, so a temperature-0 request matches a greedy run. */
    if (temp <= 0.0f) {
        const size_t max_i = best_index(s);
        for (size_t i = 0; i < s.c.size(); ++i)
            if (i != max_i) s.c[i].logit = -INFINITY;
        return;
    }

    for (auto& e : s.c) e.logit /= temp;
}

void host_top_k(CandidateSet& s, int32_t k) {
    if (k <= 0) return;
    if ((size_t)k >= s.c.size()) { s.sort_desc(); return; }
    s.sort_desc();
    s.c.resize((size_t)k);
}

void host_top_p(CandidateSet& s, float p, size_t min_keep) {
    /* p <= 0 is off, as in sample_topp; upstream would keep min_keep candidates. */
    if (p >= 1.0f || !(p > 0.0f) || s.c.empty()) return;

    s.sort_desc();
    host_softmax(s, false);

    /* Upstream's adaptive partial sort is a performance path that yields the same set; the
     * reference does a full sort and the device kernel a radix threshold. */
    float  cum = 0.0f;
    size_t last_idx = s.c.size();
    for (size_t i = 0; i < s.c.size(); ++i) {
        cum += s.c[i].p;
        if (cum >= p && i + 1 >= min_keep) { last_idx = i + 1; break; }
    }
    s.c.resize(last_idx);
}

void host_min_p(CandidateSet& s, float p, size_t min_keep) {
    if (p <= 0.0f || s.c.empty()) return;

    /* Upstream also has an unsorted fast path. It selects the same set but leaves a different
     * order, which a device kernel cannot reproduce, so the reference always sorts. */
    s.sort_desc();

    const float min_logit = s.c[0].logit + std::log(p);   /* p_i >= p * p_max */
    size_t i = 1;                                          /* the mode always survives */
    for (; i < s.c.size(); ++i)
        if (s.c[i].logit < min_logit && i >= min_keep) break;

    s.c.resize(i);
}

void host_typical(CandidateSet& s, float p, size_t min_keep) {
    if (p >= 1.0f || !(p > 0.0f) || s.c.empty()) return;

    host_softmax(s, true);

    float entropy = 0.0f;
    for (const auto& e : s.c)
        if (e.p > 0.0f) entropy += -e.p * std::log(e.p);

    /* Distance between each candidate's surprisal and the entropy; typical sampling keeps the
     * tokens closest to the expected surprisal. A zero-probability candidate has none. */
    std::vector<float> shifted(s.c.size());
    for (size_t i = 0; i < s.c.size(); ++i) {
        const float pi = s.c[i].p;
        shifted[i] = pi > 0.0f ? std::fabs(-std::log(pi) - entropy) : INFINITY;
    }

    std::vector<size_t> idx(s.c.size());
    std::iota(idx.begin(), idx.end(), (size_t)0);
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        if (shifted[a] != shifted[b]) return shifted[a] < shifted[b];
        return s.c[a].id < s.c[b].id;      /* total order, same reason as sort_desc */
    });

    /* The cut is a distance, not a count: the smallest distance whose candidates at or below it
     * carry at least p, with every candidate at that distance kept. sample_typical keeps the tied
     * set the same way, so the result does not depend on an order among equal distances. With no
     * such distance (rounding leaves the whole mass below p) nothing is cut. */
    float  cum = 0.0f;
    size_t keep = idx.size();
    for (size_t i = 0; i < idx.size(); ++i) {
        if (!(s.c[idx[i]].p > 0.0f)) break;
        cum += s.c[idx[i]].p;
        const bool tie_next = i + 1 < idx.size() && shifted[idx[i + 1]] == shifted[idx[i]];
        if (cum >= p && !tie_next) { keep = i + 1; break; }
    }
    keep = std::max(keep, std::min(min_keep, idx.size()));

    std::vector<Candidate> kept;
    kept.reserve(keep);
    for (size_t i = 0; i < keep; ++i) kept.push_back(s.c[idx[i]]);
    s.c = std::move(kept);
    s.sorted = false;                       /* surprisal order is not logit order */
    s.sort_desc();
}

void host_penalties(CandidateSet& s, const int32_t* history, int64_t n_history,
                    int32_t last_n, float rep, float freq, float pres) {
    if (last_n == 0) return;
    if (rep == 1.0f && freq == 0.0f && pres == 0.0f) return;
    if (!history || n_history <= 0) return;

    const int64_t window = last_n < 0 ? n_history : std::min<int64_t>(last_n, n_history);
    std::unordered_map<int32_t, int> count;
    count.reserve((size_t)window * 2);
    for (int64_t i = n_history - window; i < n_history; ++i) count[history[i]]++;

    for (auto& e : s.c) {
        auto it = count.find(e.id);
        if (it == count.end()) continue;
        const int n = it->second;

        /* The paper always divides, which makes a negative logit more likely; upstream
         * multiplies below zero instead, and the reference follows upstream. */
        if (e.logit <= 0.0f) e.logit *= rep;
        else                 e.logit /= rep;

        e.logit -= (float)n * freq + (n > 0 ? 1.0f : 0.0f) * pres;
    }
    s.sorted = false;
}

/* ------------------------------------------------------------------ DRY */

void DryBreakers::index(int32_t n_vocab) {
    const size_t words = n_vocab > 0 ? ((size_t)n_vocab + 31) / 32 : 0;
    heads.assign(words, 0u);
    singles.assign(words, 0u);
    for (const auto& e : seqs) {
        const int32_t t = e.first;
        if (t < 0 || t >= n_vocab) continue;
        heads[(size_t)(t >> 5)] |= 1u << ((uint32_t)t & 31u);
        if (e.second.empty()) singles[(size_t)(t >> 5)] |= 1u << ((uint32_t)t & 31u);
    }
}

bool DryBreakers::is_head(int32_t t) const {
    if (heads.empty()) return seqs.count(t) > 0;
    return bit(heads, t);
}

bool DryBreakers::is_single(int32_t t) const {
    if (singles.empty()) {
        auto range = seqs.equal_range(t);
        for (auto it = range.first; it != range.second; ++it)
            if (it->second.empty()) return true;
        return false;
    }
    return bit(singles, t);
}

/* The longest prefix of `s` of at most `max` bytes that does not end inside a UTF-8 sequence.
 * Upstream cuts at the byte, which can leave a tail the tokeniser refuses, failing the request. */
static std::string utf8_prefix(const std::string& s, size_t max) {
    if (s.size() <= max) return s;
    size_t n = max;
    while (n > 0 && ((unsigned char)s[n] & 0xC0u) == 0x80u) --n;
    return s.substr(0, n);
}

int dry_prepare_breakers(const VocabView& vocab, const std::vector<std::string>& breakers,
                         DryBreakers* out, int max_tail_len) {
    if (!out) return RAD_E_INVAL;
    out->seqs.clear();
    out->heads.clear();
    out->singles.clear();
    if (breakers.empty()) return RAD_OK;
    if (breakers.size() > kDryMaxBreakers) {
        RAD_WARN("dry: %zu sequence breakers, more than the %zu the breaker scan accepts",
                 breakers.size(), kDryMaxBreakers);
        return RAD_E_INVAL;
    }

    /* Upstream: get_overlapping_token_sequences. A full vocabulary scan per breaker, so this runs
     * once per request, never on the step path. */
    const int32_t n = vocab.n_tokens();
    for (const std::string& given : breakers) {
        const std::string str = utf8_prefix(given, kDryBreakerMaxBytes);
        if (str.empty()) continue;

        /* The tail depends only on the split point i, not on which token covered the head, so
         * each split is tokenised once. A breaker starting with a common byte matches thousands
         * of tokens, and tokenising is the expensive half of the scan. */
        std::vector<std::vector<int32_t>> tail_at(str.size());
        std::vector<char> have(str.size(), 0);

        for (int32_t id = 0; id < n; ++id) {
            const std::string& word = vocab.token_piece(id);
            if (word.find(str) != std::string::npos) {
                out->seqs.emplace(id, std::vector<int32_t>());
                continue;
            }
            size_t pos = std::string::npos;
            while ((pos = word.find(str[0], pos + 1)) != std::string::npos) {
                bool   match = true;
                size_t i = 1;
                for (; i < str.size() && i + pos < word.size(); ++i) {
                    if (word[pos + i] != str[i]) { match = false; break; }
                }
                if (!match) continue;

                if (!have[i]) {
                    std::vector<int32_t>& fresh = tail_at[i];
                    const int st = vocab.tokenize(str.substr(i), &fresh);
                    if (st < 0) {
                        /* Fail loudly rather than run DRY with a breaker missing. */
                        RAD_WARN("dry: sequence breaker \"%s\" does not tokenise (%s)",
                                 str.c_str(), rad_strerror(st));
                        out->seqs.clear();
                        return st;
                    }
                    if (max_tail_len >= 0 && (int)fresh.size() > max_tail_len)
                        fresh.resize((size_t)max_tail_len);
                    have[i] = 1;
                }
                const std::vector<int32_t>& tail = tail_at[i];

                bool dup = false;
                auto range = out->seqs.equal_range(id);
                for (auto it = range.first; it != range.second; ++it)
                    if (it->second == tail) { dup = true; break; }
                if (!dup) out->seqs.emplace(id, tail);
            }
        }
    }
    out->index(n);
    return RAD_OK;
}

int dry_rep_limit(const DryBreakers& breakers, const int32_t* window, int n) {
    if (!window || n <= 0) return n > 0 ? n : 0;
    if (breakers.empty()) return n;

    /* rat(i): the i-th token counting back from the end of the window. */
    auto rat = [&](int i) -> int32_t { return window[n - 1 - i]; };

    for (int i = 0; i < n; ++i) {
        const int32_t tok = rat(i);
        if (!breakers.is_head(tok)) continue;

        int longest = -1;
        auto its = breakers.seqs.equal_range(tok);
        for (auto it = its.first; it != its.second; ++it) {
            const int seq_len = (int)it->second.size();
            if (seq_len <= longest || seq_len > i) continue;
            bool match = true;
            for (int off = 0; off < seq_len; ++off) {
                if (it->second[(size_t)off] != rat(i - off - 1)) { match = false; break; }
            }
            if (match) longest = seq_len;
        }
        if (longest >= 0) return i - longest;
    }
    return n;
}

void host_dry(CandidateSet& s, const int32_t* history, int64_t n_history,
              int64_t total_context, const DryParams& p, const DryBreakers& breakers) {
    if (p.multiplier == 0.0f || p.base < 1.0f || p.penalty_last_n == 0) return;
    if (!history || n_history <= 0) return;
    if (total_context <= 0) total_context = n_history;

    /* Any negative window is the whole context, as in sample_dry; upstream reads only -1 so. */
    const int64_t effective = p.penalty_last_n < 0 ? total_context : (int64_t)p.penalty_last_n;
    const int n_rep = (int)std::min(std::min(n_history, effective), total_context);
    if (n_rep <= p.allowed_length) return;

    /* rat(i): the i-th token counting back from the end of the window. Upstream reads a ring
     * buffer; here the history is the scheduler's flat array. */
    auto rat = [&](int i) -> int32_t { return history[n_history - 1 - i]; };

    std::vector<int> repeat_count((size_t)n_rep, 0);
    std::unordered_map<int32_t, int> max_token_repeat;

    /* Step 1: the most recent complete breaker sequence caps the counted repeat length. */
    const int rep_limit = dry_rep_limit(breakers, history + (n_history - n_rep), n_rep);
    if (rep_limit < p.allowed_length) return;

    /* Step 2: the Z-algorithm in reverse, giving for every position the length of the suffix
     * that also ends there. Linear: lt/rt bound each token to one visit. */
    {
        const int last = n_rep - 1;
        int rt = 0, lt = 0;
        for (int k = 1; k < n_rep; ++k) {
            if (k > rt) {
                int n = 0;
                while (n + k < n_rep && rat(n) == rat(n + k)) ++n;
                repeat_count[(size_t)(last - k)] = std::min(n, rep_limit);
                if (n > 0) { lt = k; rt = k + n - 1; }
            } else {
                const int pi = k - lt;
                const int right_part_len = rt - k + 1;
                if (repeat_count[(size_t)(last - pi)] < right_part_len) {
                    repeat_count[(size_t)(last - k)] =
                        std::min(repeat_count[(size_t)(last - pi)], rep_limit);
                } else {
                    int i = rt + 1;
                    while (i < n_rep && rat(i) == rat(i - k)) ++i;
                    repeat_count[(size_t)(last - k)] = std::min(i - k, rep_limit);
                    lt = k; rt = i - 1;
                }
            }
        }
    }

    /* Step 3: the token after each long-enough repeat would extend it; record the longest
     * repeat per such token. */
    for (int i = 0; i < n_rep - 1; ++i) {
        const int len = repeat_count[(size_t)i];
        if (len < p.allowed_length) continue;
        const int32_t tok = rat(n_rep - 2 - i);
        auto it = max_token_repeat.find(tok);
        if (it == max_token_repeat.end() || it->second < len) max_token_repeat[tok] = len;
    }

    /* Step 4: a penalty exponential in the excess length, with the exponent clamped so pow()
     * cannot overflow a float (log(FLT_MAX) = 88.72). */
    const float kFloatMaxLog = 88.7228391f;
    int max_exponent = 0;
    if (p.base > 1.000001f) max_exponent = (int)(kFloatMaxLog / std::log(p.base));

    for (auto& e : s.c) {
        auto it = max_token_repeat.find(e.id);
        if (it == max_token_repeat.end()) continue;

        /* A single-token breaker is exempt: it is what ends the repetition. */
        if (breakers.is_single(e.id)) continue;

        int exp_n = it->second - p.allowed_length;
        if (max_exponent > 0 && exp_n > max_exponent) exp_n = max_exponent;
        e.logit -= p.multiplier * std::pow(p.base, (float)exp_n);
    }
    s.sorted = false;
}

/* ------------------------------------------------------------------ XTC */

void host_xtc(CandidateSet& s, float probability, float threshold, size_t min_keep,
              uint64_t seed, uint64_t pos) {
    /* threshold <= 0 is off, as in sample_xtc. Upstream fires and keeps only the least probable
     * candidate, since every candidate is at or above 0. */
    if (probability <= 0.0f || !(threshold > 0.0f) || threshold > 0.5f || s.c.size() < 2) return;

    /* A separate RNG stream, so the pick is not correlated with XTC firing. Fires when the
     * uniform is below the probability, the comparison `sample_xtc` makes, so a draw exactly on
     * it agrees. */
    if (rng_uniform(seed, pos, RAD_RNG_XTC) >= probability) return;

    host_softmax(s, true);

    size_t pos_last = 0;
    for (size_t i = 0; i < s.c.size(); ++i) {
        if (s.c[i].p >= threshold) pos_last = i;
        else break;
    }

    /* Drop everything above the threshold except the least probable of them. */
    if (s.c.size() - pos_last >= min_keep && pos_last > 0)
        s.c.erase(s.c.begin(), s.c.begin() + (std::ptrdiff_t)pos_last);
}

/* ------------------------------------------------------------------ the grammar mask */

void host_mask(CandidateSet& s, const uint32_t* bitmask, int64_t n_words) {
    if (!bitmask) return;
    for (auto& e : s.c) {
        const int64_t w = (int64_t)e.id >> 5;
        if (w >= n_words) { e.logit = -INFINITY; continue; }
        if (!(bitmask[w] & (1u << ((uint32_t)e.id & 31u)))) e.logit = -INFINITY;
    }
    s.sorted = false;
}

/* ------------------------------------------------------------------ the pick */

int32_t host_pick(CandidateSet& s, uint64_t seed, uint64_t pos) {
    if (s.c.empty()) { s.selected = -1; return -1; }

    float max_l = -INFINITY;
    for (const auto& e : s.c) max_l = std::max(max_l, e.logit);
    if (!(max_l > -INFINITY)) {
        /* Fully masked: the grammar admits nothing. The caller fails the request. */
        s.selected = -1;
        return -1;
    }
    if (s.c.size() == 1) { s.selected = 0; s.c[0].p = 1.0f; return s.c[0].id; }

    /* Walk in descending logit, ties to the lower id, independent of the set's current order.
     * This is the order sample_pick walks, since sample_topk leaves its candidates in it. */
    std::vector<size_t> order(s.c.size());
    std::iota(order.begin(), order.end(), (size_t)0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (s.c[a].logit != s.c[b].logit) return s.c[a].logit > s.c[b].logit;
        return s.c[a].id < s.c[b].id;
    });

    double total = 0.0;
    for (size_t i : order) total += std::exp((double)s.c[i].logit - (double)max_l);
    if (!(total > 0.0)) { s.selected = -1; return -1; }

    const double want = (double)rng_uniform(seed, pos, RAD_RNG_PICK) * total;

    /* A -inf candidate is never chosen. A draw at the top of the range can pass every
     * comparison in floating point, so the fall-through is the last finite candidate. */
    double  acc = 0.0;
    size_t  chosen = order.front();
    for (size_t i : order) {
        if (!(s.c[i].logit > -INFINITY)) continue;
        chosen = i;
        acc += std::exp((double)s.c[i].logit - (double)max_l);
        if (acc >= want) break;
    }

    for (size_t i = 0; i < s.c.size(); ++i)
        s.c[i].p = (float)(std::exp((double)s.c[i].logit - (double)max_l) / total);

    s.selected = (int)chosen;
    return s.c[chosen].id;
}

int32_t host_argmax(const float* logits, int64_t n_vocab) {
    if (!logits || n_vocab <= 0) return -1;
    int64_t best = 0;
    for (int64_t i = 1; i < n_vocab; ++i)
        if (logits[i] > logits[best]) best = i;      /* strict >, so a tie keeps the lowest id */
    return (int32_t)best;
}

void host_ngram_ban(CandidateSet& s, const int32_t* history, int64_t n_history, int32_t n,
                    int32_t window, const uint32_t* whitelist, int64_t wl_words) {
    if (n <= 0 || !history || n_history < (int64_t)n) return;
    const int64_t len = n_history;
    const int64_t win = (window > 0 && window < len) ? window : len;
    const int64_t start = len - win, end = len - n + 1;
    if (end <= start) return;
    const int64_t plen = (int64_t)n - 1, p0 = len - plen;

    /* The set of tokens a continuation would complete an n-gram with, as the reference builds it:
     * every n-gram in the window whose first n-1 tokens equal the current prefix contributes the
     * token after them. */
    std::unordered_set<int32_t> banned;
    for (int64_t idx = start; idx < end; ++idx) {
        bool match = true;
        for (int64_t j = 0; j < plen; ++j) {
            if (history[idx + j] != history[p0 + j]) { match = false; break; }
        }
        if (match) banned.insert(history[idx + plen]);
    }

    /* The reference's `banned.difference_update(self.whitelist)`, before any -inf: a token whose
     * whitelist bit is set is never banned. A bit past the plane is not set, so a token outside
     * it stays banned, as an id the reference's set does not contain. */
    if (whitelist && wl_words > 0) {
        for (auto it = banned.begin(); it != banned.end();) {
            const int64_t id = *it;
            if (id >= 0 && (id >> 5) < wl_words &&
                (whitelist[(size_t)(id >> 5)] >> ((uint32_t)id & 31u)) & 1u)
                it = banned.erase(it);
            else
                ++it;
        }
    }

    if (banned.empty()) return;
    s.sorted = false;   /* a -infinity candidate belongs after every finite one */
    for (Candidate& c : s.c)
        if (banned.count(c.id) != 0) c.logit = -INFINITY;
}

/* ================================================================== the chain */
int32_t host_sample_into(CandidateSet& s, const SamplingParams& sp, const HostRefInput& in,
                         ChainOrder order) {
    /* Penalties first in both orders: they act on raw logits, the cuts on probabilities. */
    host_penalties(s, in.history, in.n_history, sp.penalty_last_n,
                   sp.rep_penalty, sp.freq_penalty, sp.pres_penalty);

    /* DRY runs without breakers, as the device chain does; breakers only cap the repeat length.
     * An allowed length below 1 is 2, as sample_dry reads it (admission refuses one). */
    if (sp.dry_multiplier > 0.0f) {
        static const DryBreakers kNone;
        DryParams dp;
        dp.multiplier     = sp.dry_multiplier;
        dp.base           = sp.dry_base;
        dp.allowed_length = sp.dry_allowed_length > 0 ? sp.dry_allowed_length : 2;
        dp.penalty_last_n = sp.dry_penalty_last_n;
        host_dry(s, in.history, in.n_history, in.total_context, dp,
                 in.dry_breakers ? *in.dry_breakers : kNone);
    }

    /* The n-gram ban, after the penalties and before the greedy split: the device chain issues it
     * there, so a greedy request sees the same masked distribution. Off at ngram_ban_n 0. */
    host_ngram_ban(s, in.history, in.n_history, sp.ngram_ban_n, sp.ngram_window,
                   in.ngram_whitelist, in.ngram_whitelist_words);

    /* Greedy skips the chain: penalties -> mask -> argmax. Temperature 0 collapses every other
     * logit to -inf, so a mask applied after it could leave nothing. */
    if (sp.greedy()) {
        host_mask(s, in.bitmask, in.n_words);
        if (s.c.empty()) return -1;
        const size_t best = best_index(s);
        if (!(s.c[best].logit > -INFINITY)) return -1;   /* everything masked out */
        s.selected = (int)best;
        return s.c[best].id;
    }

    if (order == ChainOrder::Radiance) {
        /* `temp_last` moves temperature behind the cuts, as the device chain moves it onto the
         * candidate plane (Sampler::run). */
        if (!sp.temp_last) host_temp(s, sp.temp);
        host_mask(s, in.bitmask, in.n_words);
        /* sample_topk narrows every row to the candidate width; top_k 0 is that width. */
        const int64_t width = in.max_candidates > 0 ? in.max_candidates : (int64_t)s.c.size();
        host_top_k(s, (int32_t)(sp.top_k > 0 ? std::min<int64_t>(sp.top_k, width) : width));
        host_top_p(s, sp.top_p, 1);
        host_min_p(s, sp.min_p, 1);
        host_typical(s, sp.typical_p, 1);
        host_xtc(s, sp.xtc_probability, sp.xtc_threshold, 1, sp.seed, in.pos);
        if (sp.temp_last) host_temp(s, sp.temp);
    } else {
        /* llama.cpp's default order, for checking this port against llama.cpp. Top-n-sigma is
         * not in SamplingParams, so a comparison that enables it upstream will disagree. */
        host_mask(s, in.bitmask, in.n_words);
        host_top_k(s, sp.top_k);
        host_typical(s, sp.typical_p, 1);
        host_top_p(s, sp.top_p, 1);
        host_min_p(s, sp.min_p, 1);
        host_xtc(s, sp.xtc_probability, sp.xtc_threshold, 1, sp.seed, in.pos);
        host_temp(s, sp.temp);
    }

    return host_pick(s, sp.seed, in.pos);
}

int32_t host_sample(const float* logits, int64_t n_vocab, const SamplingParams& sp,
                    const HostRefInput& in, ChainOrder order) {
    if (!logits || n_vocab <= 0) return -1;
    CandidateSet s;
    s.from_logits(logits, n_vocab);
    return host_sample_into(s, sp, in, order);
}

}  /* namespace rad */
