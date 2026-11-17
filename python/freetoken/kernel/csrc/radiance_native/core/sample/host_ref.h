/* host_ref.h -- the host reference sampling chain.
 *
 * Sampling runs on device (spec §13) through the declared sampler ops, whose oracle is libref
 * (spec §17). This is a second, independent formulation of the same chain: llama.cpp's samplers
 * ported to plain C++ over a float array, with no llama.cpp or ggml dependency, following
 * `src/llama-sampler.cpp` at upstream commit 06938ac12 (MIT). sample_test holds the sampler
 * driver's chain (Sampler over libref) to it row by row, and Sampler::logprobs_for uses host_dry to
 * undo DRY.
 *
 * Where upstream and the op chain differ, this follows the chain, which is the contract:
 *
 * 1. Chain order. llama.cpp's default is penalties, DRY, top-n-sigma, top-k, typical, top-p,
 *    min-p, XTC, temperature last. Radiance follows the spec and vLLM (see ChainOrder::Radiance).
 *    Top-p, min-p and typical cut on probabilities, so running temperature first changes which
 *    tokens survive. ChainOrder::LlamaCpp keeps upstream's order for checking this port against
 *    llama.cpp itself.
 *
 * 2. RNG. Upstream keeps a stateful std::mt19937 per sampler, which a kernel cannot hold per
 *    sequence and which makes results depend on batch composition. Here randomness is a pure
 *    function of (seed, position, stream): abi/rad_sample.h's draw, bit for bit, with the same
 *    per-stream keys, so the host sees exactly the uniforms the device stages see.
 *
 * 3. Values. top_p, typical_p or xtc_threshold at or below 0 turns the stage off; typical keeps
 *    every candidate tied at its cut; a negative DRY window is the whole context; and top_k 0 is
 *    the candidate width (HostRefInput::max_candidates), not the whole vocabulary.
 */
#pragma once
#include "../rad_core.h"
#include "vocab_view.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

/* ================================================================== the candidate set */

/* One candidate. `p` is meaningless until a softmax has run over the set; the samplers that need
 * it call one themselves, as upstream's do. */
struct Candidate {
    int32_t id    = 0;
    float   logit = 0.0f;
    float   p     = 0.0f;
};

/* The set the chain narrows. `sorted` tracks descending-by-logit order; many upstream samplers
 * take a different path when it is set, so it must stay accurate. */
struct CandidateSet {
    std::vector<Candidate> c;
    bool sorted   = false;
    int  selected = -1;      /* index into c, set by the pick; -1 until then */

    void from_logits(const float* logits, int64_t n_vocab);
    void sort_desc();        /* descending logit, ties broken by ascending id */
    size_t size() const { return c.size(); }
};

/* ================================================================== reproducible randomness */

/* Counter-based generator: bits are a pure function of the three coordinates, so host and device
 * draw the same number in any batch order or size. It is rad_sample_uniform01 from
 * abi/rad_sample.h, splitmix64 over ((seed ^ key) * golden ratio + pos).
 *
 * `pos` is the output position (token index within this request's generation), so draws depend on
 * where a request is, not when it was scheduled. `stream` separates independent draws at the same
 * position; each maps to its RAD_SAMPLE_KEY_* key. */
enum : uint32_t {
    RAD_RNG_PICK    = 0,   /* the final categorical draw */
    RAD_RNG_XTC     = 1    /* XTC's "does this fire at all" coin */
};

/* Uniform in [0,1) with 24 bits, the most a float resolves; more would let host and device
 * disagree in the last place. */
float    rng_uniform(uint64_t seed, uint64_t pos, uint32_t stream);

/* ================================================================== individual samplers */

/* Upstream: llama_sampler_softmax_impl. Fills `p` and normalises. */
void host_softmax(CandidateSet& s, bool do_sort);

/* Upstream: llama_sampler_temp_impl. temp <= 0 collapses to the argmax by setting every other
 * logit to -inf, keeping a temperature-0 request identical to a greedy one. */
void host_temp(CandidateSet& s, float temp);

/* Upstream: llama_sampler_top_k_impl. k <= 0 is off; host_sample passes the candidate width for
 * top_k 0 (see HostRefInput::max_candidates). */
void host_top_k(CandidateSet& s, int32_t k);

/* Upstream: llama_sampler_top_p_apply. Keeps the shortest prefix in descending logit order, ties
 * to the lower id, whose mass reaches p, so a tie at the threshold is split by id as sample_topp
 * splits it. p <= 0 is off. */
void host_top_p(CandidateSet& s, float p, size_t min_keep);

/* Upstream: llama_sampler_min_p_apply. A floor relative to the mode, not an absolute one. */
void host_min_p(CandidateSet& s, float p, size_t min_keep);

/* Upstream: llama_sampler_typical_apply, with sample_typical's cut. Keeps the candidates whose
 * surprisal is closest to the distribution's entropy, not the most probable ones: everything at or
 * below the smallest distance whose candidates carry at least p, ties at that distance included
 * (upstream cuts a prefix at mass > p). Zero-probability candidates go. p <= 0 is off. The result
 * is in descending logit order. */
void host_typical(CandidateSet& s, float p, size_t min_keep);

/* Upstream: llama_sampler_penalties_apply, with the history passed in instead of kept in a ring
 * buffer; the scheduler owns each request's token history. `history` is oldest first; only the
 * last `last_n` tokens are counted. */
void host_penalties(CandidateSet& s, const int32_t* history, int64_t n_history,
                    int32_t last_n, float rep, float freq, float pres);

/* DRY's sequence breakers, resolved to token sequences once per request (upstream's
 * `dry_processed_breakers`). Each breaker costs a full vocabulary scan, so this stays off the
 * step path.
 *
 * `seqs` maps a head token to the tails that complete a breaker after it; an empty tail means the
 * head token holds a whole breaker. The bitsets answer the step path's two per-token questions
 * (does a breaker start here, is this token a whole breaker) with one load instead of a hash
 * lookup. */
struct DryBreakers {
    std::unordered_multimap<int32_t, std::vector<int32_t>> seqs;
    std::vector<uint32_t> heads;     /* bit t: token t heads at least one breaker sequence */
    std::vector<uint32_t> singles;   /* bit t: token t is a whole breaker by itself */

    bool empty() const { return seqs.empty(); }

    /* Build the two bitsets from `seqs`. dry_prepare_breakers calls it; a set filled by hand is
     * answered from the multimap until indexed, so both answer the same. */
    void index(int32_t n_vocab);
    bool is_head(int32_t t) const;
    bool is_single(int32_t t) const;

private:
    static bool bit(const std::vector<uint32_t>& v, int32_t t) {
        if (t < 0 || (size_t)(t >> 5) >= v.size()) return false;
        return (v[(size_t)(t >> 5)] >> ((uint32_t)t & 31u)) & 1u;
    }
};

/* Bounds on the breaker scan. Each breaker is a full-vocabulary pass on the scheduler thread,
 * once per rank, stalling every sequence while it runs. Upstream truncates a breaker to 40 bytes
 * and its tail to 20 tokens (llama_sampler_init_dry). The count bound is this engine's own, well
 * above the four breakers clients send by default; a request over it is refused at admission. */
constexpr size_t kDryMaxBreakers      = 16;
constexpr size_t kDryBreakerMaxBytes  = 40;
constexpr int    kDryBreakerMaxTail   = 20;

/* Upstream: get_overlapping_token_sequences, per breaker, truncating each breaker to
 * kDryBreakerMaxBytes (cut back to a UTF-8 boundary so the tail still tokenises) and each tail to
 * `max_tail_len` tokens. More than kDryMaxBreakers is RAD_E_INVAL. RAD_E_UNSUPPORTED if the view
 * has no tokeniser, rather than silently running DRY without breakers. */
int dry_prepare_breakers(const VocabView& vocab, const std::vector<std::string>& breakers,
                         DryBreakers* out, int max_tail_len = kDryBreakerMaxTail);

/* Step 1 of upstream's llama_sampler_dry_apply: walking back from the end of `window` (n tokens,
 * oldest first), the first token that heads a complete breaker sequence caps the counted repeat
 * length. Returns that cap, or n when no breaker completes in the window. Shared by host_dry and
 * the sampler's per-step staging, so reference and device use the same limit. */
int dry_rep_limit(const DryBreakers& breakers, const int32_t* window, int n);

struct DryParams {
    float   multiplier     = 0.0f;
    float   base           = 1.75f;
    int32_t allowed_length = 2;
    int32_t penalty_last_n = -1;    /* -1 = the whole context */
};

/* Upstream: llama_sampler_dry_apply, Z-algorithm and all. Ported from Koboldcpp originally
 * (LostRuins/koboldcpp#982, author pi6am) by way of llama.cpp. */
void host_dry(CandidateSet& s, const int32_t* history, int64_t n_history,
              int64_t total_context, const DryParams& p, const DryBreakers& breakers);

/* Upstream: llama_sample_xtc_apply. Fires with probability `probability`; when it fires it drops
 * every candidate above the threshold except the least probable of them, removing the obvious
 * continuations. threshold <= 0 is off, as in sample_xtc. */
void host_xtc(CandidateSet& s, float probability, float threshold, size_t min_keep,
              uint64_t seed, uint64_t pos);

/* The windowed no-repeat-ngram ban (`sample_ngram_ban`, Unlimited-OCR's
 * SlidingWindowNoRepeatNgramProcessor): a candidate whose token would complete an `n`-gram the
 * last `window` tokens of `history` already contain gets -infinity. `n` <= 0 is off, `window` <= 0
 * is the whole history. It runs after the penalties and DRY and before every cut, so the greedy
 * path sees it too.
 *
 * `whitelist` is the reference's whitelist (a SET bit means exempt from the ban, the same plane
 * form as host_mask's, word w bit b covering token 32w + b): ids removed from the banned set
 * before anything is written, as `banned.difference_update(self.whitelist)` does. Null or zero
 * words is off -- the reference's empty whitelist, the default serving form. */
void host_ngram_ban(CandidateSet& s, const int32_t* history, int64_t n_history, int32_t n,
                    int32_t window, const uint32_t* whitelist = nullptr, int64_t wl_words = 0);

/* The grammar mask. One bit per token, set means admissible; a cleared bit sets the logit to -inf
 * instead of removing the candidate, so nothing downstream is renumbered. `n_words` is
 * ceil(n_vocab/32). */
void host_mask(CandidateSet& s, const uint32_t* bitmask, int64_t n_words);

/* The pick. Inverse CDF over the surviving candidates in descending logit order, ties to the
 * lower id. A total order makes the result depend only on the uniform, and it is the order
 * `sample_pick` walks top-k's output (and upstream's dist sampler walks a sorted set). A -inf
 * (grammar-masked) candidate is never returned, even by the fall-through at the top of the range.
 * Returns the token id, or -1 for an empty or fully masked set. */
int32_t host_pick(CandidateSet& s, uint64_t seed, uint64_t pos);

/* Argmax over raw logits, lowest id winning a tie. The greedy path skips the chain, so this must
 * agree with host_temp(0) + host_pick. */
int32_t host_argmax(const float* logits, int64_t n_vocab);

/* ================================================================== the chain */

/* The candidate width every stage after top-k works in, SamplerConfig::max_candidates' default. */
constexpr int64_t kSampleCandidates = 1024;
/* Under a split vocabulary the merge (sample_merge_topk) holds world_size * n_cand gathered pairs
 * in one workgroup's LDS; libr4d's row serves 4096 at most, so the per-rank width shrinks with
 * the world (512 at eight ranks). */
constexpr int64_t kSampleMergeCandidates = 4096;

enum class ChainOrder {
    /* penalties, DRY, n-gram ban, temperature, mask, top-k, top-p/min-p/typical, XTC, pick */
    Radiance = 0,
    LlamaCpp = 1    /* penalties, DRY, top-k, typical, top-p, min-p, XTC, temperature, pick */
};

/* Per-step inputs the chain needs beyond SamplingParams, which is frozen in rad_core.h and holds
 * per-request settings. */
struct HostRefInput {
    const int32_t* history   = nullptr;   /* the sequence's tokens so far, oldest first */
    int64_t        n_history = 0;
    int64_t        total_context = 0;     /* DRY's clamp; 0 means use n_history */
    const uint32_t* bitmask  = nullptr;   /* grammar mask, or null */
    int64_t        n_words   = 0;
    /* The n-gram ban's whitelist bitmask (a SET bit exempts a token, host_ngram_ban), null =
     * no exemptions. */
    const uint32_t* ngram_whitelist       = nullptr;
    int64_t         ngram_whitelist_words = 0;
    uint64_t       pos       = 0;         /* output position: the RNG's second coordinate */
    const DryBreakers* dry_breakers = nullptr;   /* null runs DRY with no breakers */
    /* The chain's candidate width (SamplerConfig::max_candidates): top-k narrows every row to
     * at most this many, so top_k 0 samples from the most likely kSampleCandidates tokens. 0 is
     * the whole vocabulary. Read by ChainOrder::Radiance only. */
    int64_t        max_candidates = kSampleCandidates;
};

/* Run the whole chain over one row of logits and return the token. */
int32_t host_sample(const float* logits, int64_t n_vocab, const SamplingParams& sp,
                    const HostRefInput& in, ChainOrder order = ChainOrder::Radiance);

/* As host_sample, leaving the narrowed candidate set for inspection. */
int32_t host_sample_into(CandidateSet& s, const SamplingParams& sp, const HostRefInput& in,
                         ChainOrder order = ChainOrder::Radiance);

}  /* namespace rad */
