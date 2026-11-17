/* residency.h -- the seam the placement component plugs into.
 *
 * The run phase takes weight handles, never pointers (spec §3.2), and resolves them at issue time,
 * which is what lets the mover relocate a weight while serving. The run phase asks a non-blocking
 * question; the placement component answers from a table it owns.
 *
 * In an all-in-VRAM deployment no ResidencyTable is installed and the lookup collapses to
 * WeightInfo::ptr, already baked into the per-weight tensor template: no virtual call, no cost.
 */
#pragma once
#include "../rad_core.h"

#include <vector>

namespace rad {

/* Where a weight's bytes are now, and whether they have arrived.
 *
 * `ready` is the mover's event. Null means resident and settled (the common case, even under
 * expert offload). Non-null means a transfer into `ptr` is in flight and the compute stream must
 * wait on it -- a stream wait, never a host sync (spec §5.4).
 *
 * Invariant: the mover bumps `generation` whenever a new transfer into the slot begins. The run
 * phase waits once per (weight, generation) and skips later waits, since a stream wait orders
 * everything issued after it. Reusing an event without bumping the generation silently loses the
 * ordering and corrupts output.
 */
struct WeightSlot {
    void*    ptr;
    RadEvent ready;
    uint32_t generation;
};

class ResidencyTable {
public:
    virtual ~ResidencyTable() = default;

    /* Never blocks: called once per weight operand per issue, and a lock would serialise the
     * mover against compute. Null means no entry; the run phase falls back to WeightInfo::ptr. */
    virtual const WeightSlot* lookup(rad_weight) const = 0;

    /* Changes whenever any slot's pointer, event or generation does. Lookups resolved at one
     * epoch stay valid while it stands, so a routed layer's expert table is not re-walked at
     * every issue. */
    virtual uint64_t epoch() const = 0;

    /* The slots whose pointer or event changed since the last clear_changes(), each named once.
     * A step moves a handful of experts, so the run phase updates only these entries instead of
     * re-resolving every table when the epoch moves. A slot that only settled (same pointer) is
     * not named.
     *
     * One reader: this rank's run phase, on the thread that also drives the writers, so the two
     * calls do not race. */
    virtual const std::vector<rad_weight>& changes() const = 0;
    virtual void clear_changes() = 0;
};

}  /* namespace rad */
