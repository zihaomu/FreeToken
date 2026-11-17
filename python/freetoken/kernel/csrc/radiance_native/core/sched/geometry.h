/* geometry.h -- the prefill chunk geometry, computed from the resolved kernels.
 *
 * The attention block size comes from the resolved attention kernel (§7.2) and the linear-state
 * chunk length from every resolved kernel that marks a parameter RAD_PROLE_SEQ_CHUNK, so the
 * operator never has to supply a minimum --max-num-batched-tokens by hand (as vLLM requires for
 * hybrid models). If the budget is below what the geometry needs, startup prints the number and
 * refuses: a smaller budget means chunks never land on a checkpoint boundary and the linear
 * prefix hit rate is zero.
 */
#pragma once
#include "rad_core.h"

namespace rad {

struct ChunkGeometry {
    /* Attention block size, the lcm across every paged KV group. libr4d's paged attention is
     * compiled for 16 only. */
    int64_t attn_block = 1;

    /* Tile length of a linear-state scan (64 for libr4d's gated delta net). Taken from the
     * declared geometry and kernel constraints of every op whose schema marks a parameter
     * RAD_PROLE_SEQ_CHUNK. */
    int64_t state_chunk = 1;

    /* lcm(attn_block, state_chunk); every non-final prefill chunk is a multiple of it, so no
     * kernel is handed a partial tile. */
    int64_t quantum = 1;

    /* Linear-state checkpoint interval (§7.3): the operator's request rounded up to a multiple
     * of `quantum`, so a split at the interval is a legal chunk split. Zero without linear state
     * (see window_ckpt for the exception). */
    int64_t checkpoint_interval = 0;

    /* How many intervals one prefill chunk may carry from an interval boundary past the first
     * (Config::prefill_span); 1 without an interval grid. plan_chunk is where it acts. */
    int64_t span = 1;

    /* The budget of every step a span does not take (prefill_shared_tok). A span is for a prefill
     * alone in its step (Scheduler::step); a step carrying anything else is planned at this
     * budget as a deployment without a span plans it, so the span's extra rows never lengthen a
     * step other requests wait on. */
    int64_t shared_tok = 0;

    /* The smallest --max-num-batched-tokens this geometry can be served with. */
    int64_t min_max_tok = 1;

    /* Smallest upper bound a kernel places on the chunk length, or 0. Used only to report a
     * contradiction with `quantum`, never to clamp. */
    int64_t max_chunk = 0;

    /* Every contributor by name, for the startup line and the refusal. */
    std::string why;

    /* No linear state but a sliding window: checkpoints snapshot the window behind an
     * interval boundary or the prompt's tail (KVManager::has_window_snapshot). Intermediate
     * snapshots let an edited prompt resume below its divergence instead of from zero.
     * tip-only skips interval snapshots and keeps the original tail-only chunking. */
    bool window_ckpt = false;
    /* Window checkpoints at interval boundaries too, not only at the tail: every policy but
     * tip-only, which keeps the tail alone and so splits no chunk at the interval. */
    bool window_periodic = false;

    /* --prefix-resume exact (Config::prefix_exact): a prefix hit resumes only where a cold prefill
     * of the same prompt starts a chunk, so the tokens it recomputes are cut into the cold run's
     * chunks. A kernel may round a row differently at another chunk length (a split-K count chosen
     * from M, a small-M path); in the same chunks, on an idle server, the hit's output is the cold
     * run's bit for bit. The grid is where an idle server's prefill chunks start: 0, then
     * grid_origin + k * grid. Where chunks split at the interval and it fits the budget, `grid` is
     * the interval times the span the budget holds, from grid_origin = the interval when that span
     * is over 1. Otherwise `grid` is --max-num-batched-tokens rounded down to the quantum, and the
     * interval is rounded up to a multiple of it. Zero when not exact. */
    bool    exact = false;
    int64_t grid = 0;
    int64_t grid_origin = 0;

    bool checkpoints() const { return checkpoint_interval > 0; }

    /* The deepest grid position at or below `pos`; `pos` itself when not exact. */
    int64_t grid_floor(int64_t pos) const {
        if (pos <= 0) return 0;
        if (!exact || grid <= 0) return pos;
        if (pos < grid_origin) return 0;
        return grid_origin + (pos - grid_origin) / grid * grid;
    }
    bool on_grid(int64_t pos) const { return grid_floor(pos) == pos; }
};

/* The budget of a step a span does not take (ChunkGeometry::shared_tok), from the configuration
 * alone: --max-num-batched-tokens less the span's extra intervals, at least one interval, and the
 * whole budget without a span. The engine states it to the plugin before declare
 * (RadBuildCtx::max_mixed_tok) and the scheduler plans by it, so the two agree. */
int64_t prefill_shared_tok(const Config& cfg);

/* ChunkGeometry::window_ckpt from the program and the configuration alone, without resolving or
 * reporting the geometry: a trunk's sliding window, no linear state, a prefix cache and an
 * interval. Bring-up asks it before the geometry is resolved (Engine::drafter_window_snap). */
bool window_snapshots(const Program& prog, const Config& cfg);

/* Resolve the geometry from the declared program and the operator's budgets.
 *
 * Returns RAD_E_INVAL when cfg.max_tok is too small (the required number is in
 * out->min_max_tok, the reason in out->why; the caller stops), and RAD_E_SHAPE when two kernels'
 * constraints contradict each other, which is a kernel-set bug. */
int chunk_geometry_resolve(const Program& prog, const Config& cfg, ChunkGeometry* out);

}  /* namespace rad */
