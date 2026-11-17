/* accept.h -- speculative acceptance. Verification is a declared op; acceptance is core sampler
 * logic (spec §10), so it lives beside the chain that produced the distributions it compares.
 *
 * Agreement with the target's own token. A drafted token is accepted when it equals the token the
 * target produced at that position: the argmax at temperature 0, the target's sampled draw above
 * it. Both are exact, provided each verified position draws from its own RNG coordinate
 * (`p.pos = n_out + k` in sampler.cpp): the first emitted token is the target's draw whether the
 * draft agreed or not, and every later position is conditioned on a prefix accepted only because
 * it matched. Exact given the same logits; a verify step computes logits at a different M than a
 * decode step, so text matches a non-speculative run exactly at temperature 0 and mostly above it
 * (see Engine::collect_step). The cost is acceptance rate, P(y == x) instead of rejection
 * sampling's min(1, p/q), and no probability planes are needed.
 *
 * Accepting on argmax at a positive temperature is not exact: it replaces the target's
 * distribution with its mode. Do not conflate it with the rule above.
 *
 * Consumers of `n_accepted`: the scheduler frees paged KV blocks past the accepted position. The
 * linear-attention kernels do not recompute: the recurrent state and conv window have absorbed the
 * rejected tokens, and libr4d keeps the conv state as a rolling window of `conv_width-1 + n_spec`
 * entries read at the slot the last accepted token left. A rejection is a read-offset change, which
 * is why `num_accepted` is an operand of `gdn_conv_update` and `gdn_recurrent_update`
 * (docs/OPS.md, spec §10).
 */
#pragma once
#include "../rad_internal.h"

#include <cstdint>

namespace rad {

/* Acceptance by agreement over precomputed target tokens. `target` is [n_draft+1]: the token the
 * target produced at each verified position plus the bonus position past the last draft token.
 * It is the argmax at temperature 0 and the sampled draw above it; the name describes the
 * temperature-0 case, but the rule is exact for both (see the file header).
 *
 * Writes n_accepted+1 tokens into `out` (capacity n_draft+1) and sets *n_accepted. Returns
 * RAD_OK. Always emits at least one token: the target's own token after the last accepted one. */
int accept_greedy(const int32_t* draft, int n_draft, const int32_t* target,
                  int32_t* out, int32_t* n_accepted);

/* The same rule over a tree (rad_drafter_tree). Node 0 is the verify's first row (the bonus token
 * it starts from); node j, 1 <= j <= n_draft, is draft[j - 1], verified at row j, and its parent
 * is node parent[j - 1] (< j), or node j - 1 when `parent` is null (a chain). `target[r]` is the
 * target's token after node r's row.
 *
 * From node 0, take the lowest-numbered child whose token is the target's at the current node,
 * while there is one. Writes the walk's nodes to path[0..n_accepted) and the emitted tokens to
 * out[0..n_accepted]: out[t] is the target's token at path[t - 1] (node 0 for t = 0), which equals
 * the accepted draft there, and out[n_accepted] is the bonus. The card's acceptance
 * (sched/advance_impl.h, ADV_ACCEPT) walks the same way. */
inline int accept_tree(const int32_t* draft, const int32_t* parent, int n_draft,
                       const int32_t* target, int32_t* out, int32_t* path, int32_t* n_accepted) {
    if (!draft || !target || !out || !path || !n_accepted || n_draft < 0) return RAD_E_INVAL;
    int32_t cur = 0, acc = 0;
    for (;;) {
        int32_t nxt = -1;
        for (int32_t j = cur + 1; j <= n_draft; ++j) {
            int32_t p = parent ? parent[j - 1] : j - 1;
            if (p < 0 || p >= j) p = j - 1;     /* as the builder and ADV_TRUNK read it */
            if (p == cur && draft[j - 1] == target[cur]) { nxt = j; break; }
        }
        if (nxt < 0) break;
        path[acc++] = nxt;
        cur = nxt;
    }
    for (int32_t t = 0; t <= acc; ++t) out[t] = target[t == 0 ? 0 : path[t - 1]];
    *n_accepted = acc;
    return RAD_OK;
}

/* Where a verify step's emission ends: at the first end-of-generation token among `out`'s
 * n_accepted + 1 tokens (accept_greedy's output), which is emitted, or after all of them. Returns
 * the number of accepted drafts to commit, at most `n_accepted`. A drafter that predicts the stop
 * token agrees with the target, so without the cut the run would continue into the next turn.
 * The card's acceptance (sched/advance_impl.h, ADV_ACCEPT) applies the same rule, and the engine
 * refuses a step where the two disagree. */
template <class IsEog>
inline int32_t accept_cut_at_eog(const int32_t* out, int32_t n_accepted, IsEog&& is_eog) {
    for (int32_t j = 0; j <= n_accepted; ++j)
        if (is_eog(out[j])) return j;
    return n_accepted;
}

}  /* namespace rad */
