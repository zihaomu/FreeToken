/* geometry.cpp -- reconciles the linear-state chunk length with the attention block size.
 * See geometry.h.
 */
#include "geometry.h"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace rad {

/* lcm, with a non-positive side meaning "no constraint yet". The sides are plugin-declared; an lcm
 * past int64 saturates, so the quantum check refuses it rather than wrapping to a small one. */
static int64_t lcm64(int64_t a, int64_t b) {
    if (a <= 0) return b;
    if (b <= 0) return a;
    const int64_t q = a / std::gcd(a, b);
    return q > INT64_MAX / b ? INT64_MAX : q * b;
}

/* The op's sequence-chunk parameter, found by role (RAD_PROLE_SEQ_CHUNK) in its schema. Steps
 * must be cut on a multiple of it or a linear-state chunk straddles a step boundary. Matching
 * parameter names instead would silently miss an op that spells it differently. */
static const char* chunk_key_of(const RadOpSchema* s) {
    if (!s) return nullptr;
    for (int i = 0; i < s->n_params; ++i)
        if (s->params[i].role == RAD_PROLE_SEQ_CHUNK) return s->params[i].key;
    return nullptr;
}

/* Appends one contributor to the explanation. */
static void note(std::string& why, const char* what, int64_t v, const char* from) {
    if (!why.empty()) why += "; ";
    why += fmt("%s=%lld from %s", what, (long long)v, from);
}

int64_t prefill_shared_tok(const Config& cfg) {
    if (cfg.prefill_span <= 1 || cfg.checkpoint_interval <= 0) return cfg.max_tok;
    const int64_t I = cfg.checkpoint_interval;
    return std::max(std::min(cfg.max_tok, I), cfg.max_tok - (cfg.prefill_span - 1) * I);
}

/* What a prefix hit restores beside the cached blocks: linear state, or a trunk's sliding window.
 * Only the trunk's window makes a model take window snapshots; a block drafter's rides in them
 * where they exist (KVManager::set_drafter_window_snapshot) and is recomputed by draft refill
 * where they do not. A RAD_KV_RING group has no snapshot at all
 * (KVManager::declare_runtime_window): its window is each sequence's own prompt length, which no
 * slot could describe, so it is sized for the context plus a ring and is never checkpointed. */
static void restored_state(const Program& prog, bool* linear, bool* window) {
    *linear = *window = false;
    for (const KVGroupInfo& kg : prog.kv_groups) {
        if (kg.decl.kind != RAD_KV_FULL && kg.decl.kind != RAD_KV_WINDOW &&
            kg.decl.kind != RAD_KV_RING)
            *linear = true;
        else if (kg.decl.kind == RAD_KV_WINDOW && kg.block_size > 0 && !kg.layers.empty() &&
                 !kg.drafter)
            *window = true;
    }
}

bool window_snapshots(const Program& prog, const Config& cfg) {
    bool linear = false, window = false;
    restored_state(prog, &linear, &window);
    return !linear && window && cfg.prefix_cache && cfg.checkpoint_interval > 0;
}

int chunk_geometry_resolve(const Program& prog, const Config& cfg, ChunkGeometry* out) {
    if (!out) return RAD_E_INVAL;
    ChunkGeometry g;

    /* ---------------------------------------------------------------- attention block size */
    /* lcm across every paged group. libr4d uses 16 for all of them, but a sliding-window group
     * served by a different kernel may differ (§7.2). */
    bool has_linear = false, has_window = false;
    restored_state(prog, &has_linear, &has_window);
    for (const KVGroupInfo& kg : prog.kv_groups) {
        if ((kg.decl.kind == RAD_KV_FULL || kg.decl.kind == RAD_KV_WINDOW ||
             kg.decl.kind == RAD_KV_RING) && kg.block_size > 0) {
            g.attn_block = lcm64(g.attn_block, kg.block_size);
            note(g.why, "attn_block", kg.block_size, kg.name.c_str());
        }
    }

    /* ---------------------------------------------------------------- linear-state chunk */
    /* Two sources: the geometry the plugin declared the op with, and the constraints of the
     * kernel selected for it. The lcm satisfies both. */
    for (const OpInfo& op : prog.ops) {
        const char* ckey = chunk_key_of(op.schema);
        if (!ckey) continue;
        for (const Band& b : op.bands) {
            for (int d = 0; d < RAD_N_DOMAINS; ++d) {
                const Resolved& r = b.dom[d];
                if (!r) continue;

                long long v = 0;
                if (r.geom.get_i(ckey, &v) && v > 0) {
                    g.state_chunk = lcm64(g.state_chunk, (int64_t)v);
                    note(g.why, ckey, v, op.op.c_str());
                }

                const RadKernelInfo* ki = r.row ? r.row->info : nullptr;
                if (!ki) continue;
                for (int i = 0; i < ki->n_constraints; ++i) {
                    const RadConstraint& c = ki->constraints[i];
                    if (!c.key || std::strcmp(c.key, ckey) != 0 || c.ival <= 0) continue;
                    switch (c.op) {
                        /* DIV and EQ mean "a legal chunk is a multiple of this". GE folds in the
                         * same way, since a multiple of N is >= N. */
                        case RAD_C_DIV:
                        case RAD_C_EQ:
                        case RAD_C_GE:
                            g.state_chunk = lcm64(g.state_chunk, (int64_t)c.ival);
                            note(g.why, c.key, c.ival, ki->name);
                            break;
                        case RAD_C_LE:
                            g.max_chunk = g.max_chunk ? std::min<int64_t>(g.max_chunk, c.ival)
                                                      : (int64_t)c.ival;
                            break;
                        default: break;
                    }
                }
            }
        }
    }

    /* ---------------------------------------------------------------- reconcile */
    g.quantum = lcm64(g.attn_block, g.state_chunk);
    if (g.quantum < 1) g.quantum = 1;

    if (g.max_chunk > 0 && g.quantum > g.max_chunk) {
        /* No chunk length satisfies both kernels; report it instead of serving a chunk that is
         * wrong for one of them. */
        out->why = fmt("chunk quantum %lld exceeds the %lld a resolved kernel accepts (%s)",
                       (long long)g.quantum, (long long)g.max_chunk, g.why.c_str());
        out->quantum = g.quantum;
        out->min_max_tok = g.quantum;
        return RAD_E_SHAPE;
    }

    /* Round the operator's interval up to a multiple of the quantum, so a split at the interval
     * is a legal chunk split. Up, not down: down can reach zero, and a longer interval costs only
     * retention.
     *
     * Without linear state there are no checkpoints, except for a trunk window under a prefix
     * cache, which snapshots the window at interval boundaries and the prompt's tail
     * (window_ckpt). */
    if ((has_linear || (has_window && cfg.prefix_cache)) && cfg.checkpoint_interval > 0)
        g.checkpoint_interval = align_up(cfg.checkpoint_interval, g.quantum);
    else
        g.checkpoint_interval = 0;
    g.window_ckpt = !has_linear && g.checkpoint_interval > 0;
    g.window_periodic = g.window_ckpt && cfg.checkpoint_policy != "tip-only";
    /* Window checkpoints do not support skipping snapshots inside a prefill span. */
    g.span = (g.checkpoint_interval > 0 && !g.window_ckpt && cfg.prefill_span > 1)
           ? cfg.prefill_span : 1;
    g.shared_tok = prefill_shared_tok(cfg);

    /* One full quantum must fit in one step; otherwise no chunk ends on a boundary and
     * checkpoints land wherever a step ends (§7.3). */
    g.min_max_tok = g.quantum;

    if (cfg.max_tok < g.min_max_tok) {
        *out = g;
        RAD_ERR("--max-num-batched-tokens is %lld but this model's chunk geometry needs at least "
                "%lld, so a chunk can end on a linear-state boundary: %s",
                (long long)cfg.max_tok, (long long)g.min_max_tok, g.why.c_str());
        return RAD_E_INVAL;
    }

    /* The resume grid (--prefix-resume exact): where an idle server's prefill chunks start. A
     * prefill alone in its step takes the budget, rounded down to the quantum. Where chunks split
     * at the interval (linear state, or periodic window snapshots) it stops at every interval
     * boundary it reaches, and a span carries h intervals from the second one on. An interval past
     * the budget must be a multiple of the budget's chunk, or the chunk after a checkpoint would
     * start off the cold run's grid; it is rounded up, as the quantum rounds it, since a longer
     * interval costs only retention. */
    const int64_t I0 = g.checkpoint_interval;
    const bool splits = I0 > 0 && (!g.window_ckpt || g.window_periodic);
    const int64_t budget_chunk = cfg.max_tok / g.quantum * g.quantum;
    g.exact = cfg.prefix_exact && cfg.prefix_cache;
    if (g.exact && splits && I0 > budget_chunk && I0 % budget_chunk != 0) {
        g.checkpoint_interval = align_up(I0, budget_chunk);
        RAD_WARN("--checkpoint-interval %lld resolves to %lld: --prefix-resume exact needs it a "
                 "multiple of the %lld-token prefill chunk (--max-num-batched-tokens %lld) so "
                 "every checkpoint is where a cold prefill's chunk ends; give a multiple of %lld, "
                 "or --prefix-resume nearest to keep %lld", (long long)cfg.checkpoint_interval,
                 (long long)g.checkpoint_interval, (long long)budget_chunk, (long long)cfg.max_tok,
                 (long long)budget_chunk, (long long)I0);
    }
    if (g.exact) {
        const int64_t I = g.checkpoint_interval;
        if (splits && I <= budget_chunk) {
            const int64_t h = g.span > 1 ? std::max<int64_t>(1, std::min(g.span, cfg.max_tok / I))
                                         : 1;
            g.grid = h * I;
            g.grid_origin = h > 1 ? I : 0;
        } else {
            g.grid = budget_chunk;
            g.grid_origin = 0;
        }
    }

    *out = g;

    /* Log the effective chunk length. `plan_chunk` splits at the checkpoint interval first, so
     * when the interval is below --max-num-batched-tokens, raising the budget does not change the
     * chunk; the log line makes that visible. tip-only window snapshots do not
     * split at the interval, so their chunk is bounded by the token budget. */
    const int64_t span_tok = g.checkpoint_interval * g.span;
    const bool periodic = !g.window_ckpt || g.window_periodic;
    const int64_t eff = (g.checkpoint_interval > 0 && periodic && span_tok < cfg.max_tok)
                      ? span_tok : cfg.max_tok;
    RAD_INFO("chunk geometry: quantum %lld (attn block %lld, state chunk %lld), checkpoint "
             "interval %lld, max_tok %lld -> prefill chunks of at most %lld%s",
             (long long)g.quantum, (long long)g.attn_block, (long long)g.state_chunk,
             (long long)g.checkpoint_interval, (long long)cfg.max_tok, (long long)eff,
             eff >= cfg.max_tok ? ""
             : g.span > 1 ? " (the checkpoint interval times the span, not the budget)"
                          : " (the checkpoint interval, not the budget)");
    if (g.exact)
        RAD_INFO("prefix resume: exact -- a hit resumes only where an idle server's cold prefill "
                 "of the same prompt starts a chunk (every %lld tokens%s) and a prompt's tail "
                 "snapshot is taken where its last chunk starts, so the hit recomputes the cold "
                 "run's own chunks and answers bit-identically; a turn re-prefills up to one chunk "
                 "more than --prefix-resume nearest", (long long)g.grid,
                 g.grid_origin > 0 ? fmt(" from %lld", (long long)g.grid_origin).c_str() : "");
    if (cfg.prefill_span > 1) {
        /* What the span does here, said once: the intervals the budget holds can be fewer than
         * the flag asks for, and a model without an interval grid spans nothing. */
        const int64_t held = g.span > 1 ? std::min(g.span, cfg.max_tok / g.checkpoint_interval) : 0;
        if (g.span <= 1)
            RAD_WARN("--prefill-span %lld: this model has no checkpoint interval grid (%s), so no "
                     "chunk spans; a prefill alone in its step may still take the whole budget, "
                     "and every other step is planned at %lld tokens", (long long)cfg.prefill_span,
                     g.checkpoint_interval > 0 ? "its snapshots are taken at a prompt's end"
                                               : "it keeps no linear state",
                     (long long)g.shared_tok);
        else if (held < 2)
            RAD_WARN("--prefill-span %lld does nothing: --max-num-batched-tokens %lld holds one "
                     "checkpoint interval of %lld tokens, and a span of %lld needs %lld",
                     (long long)cfg.prefill_span, (long long)cfg.max_tok,
                     (long long)g.checkpoint_interval, (long long)cfg.prefill_span,
                     (long long)(cfg.prefill_span * g.checkpoint_interval));
        else
            RAD_INFO("chunk geometry: a prefill alone in its step spans up to %lld intervals%s "
                     "from its second interval on; every other step is planned at %lld tokens",
                     (long long)held,
                     held < g.span ? fmt(" (all the budget holds of the %lld asked)",
                                         (long long)g.span).c_str() : "",
                     (long long)g.shared_tok);
    }
    return RAD_OK;
}

}  /* namespace rad */
