/* arena.h -- one contiguous allocation, sized by the buffer plan at load and never allocated from
 * at runtime.
 *
 * There is deliberately no allocate(): an allocator on the step path is the dynamic behaviour this
 * engine exists to remove (spec §3.1). Offsets come from the buffer plan, which computes liveness
 * over the declared op list for the worst step; the arena only owns the block they index into.
 *
 * Scratch sits at the tail and is shared by every op on a stream, which is safe because launches
 * on one stream execute in order. A context's second lane is a second stream, so a program that
 * declares one gets a second region per lane (Program::scratch_regions). A lane used without that
 * declaration shares region 0, and Ctx::issue orders a scratch user behind an unordered one on the
 * other lane. Program::scratch_bytes is the largest single kernel's requirement; a kernel asking
 * for more is refused by name in prepare.
 */
#pragma once
#include "../rad_core.h"

namespace rad {

class Arena {
public:
    Arena() = default;
    ~Arena() { release(); }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    /* `kind` is a RAD_MEM_* class: RAD_MEM_DEVICE for the device arena, RAD_MEM_HOST for the
     * second arena a program with RAD_DOMAIN_HOST buffers gets. `regions` scratch regions sit at the
     * tail, one per lane: region 0 of `scratch_bytes`, the others of `scratch_bytes1` (-1: the same;
     * Program::scratch_bytes_lane1). */
    int  init(int64_t plan_bytes, int64_t scratch_bytes, int kind, int regions = 1,
              int64_t scratch_bytes1 = -1);
    void release();

    char*   base()          const { return base_; }
    /* Region `r`, or region 0 when the arena holds fewer. */
    void*   scratch(int r = 0) const {
        if (!base_) return nullptr;
        return base_ + scratch_off_ +
               (r > 0 && r < scratch_regions_ ? scratch_pitch_ + (r - 1) * scratch_pitch1_ : 0);
    }
    /* Region 0's size (the program's largest kernel), or region `r`'s. */
    int64_t scratch_bytes() const { return scratch_bytes_; }
    int64_t scratch_bytes(int r) const {
        return r > 0 && r < scratch_regions_ ? scratch_bytes1_ : scratch_bytes_;
    }
    int     scratch_regions() const { return scratch_regions_; }
    int64_t bytes()         const { return bytes_; }
    bool    live()          const { return base_ != nullptr; }

private:
    char*   base_ = nullptr;
    int     kind_ = RAD_MEM_DEVICE;
    int64_t bytes_ = 0;
    int64_t scratch_off_ = 0;
    int64_t scratch_bytes_ = 0;
    int64_t scratch_pitch_ = 0;     /* scratch_bytes_ padded to RAD_ALIGN_UNIT */
    int64_t scratch_bytes1_ = 0;    /* each region past 0 */
    int64_t scratch_pitch1_ = 0;
    int     scratch_regions_ = 1;
};


/* The arena's size, computed without allocating it. Shared by Ctx::bind_arena (allocates at the
 * end of startup) and the VRAM budget resolver (subtracts it before the pools are allocated), so
 * the two cannot drift apart and overcommit the card.
 *
 * `dev_out` is the plan_bytes the device arena's init() receives; `total_out` is what it
 * allocates, plan and scratch both padded to RAD_ALIGN_UNIT as Arena::init pads them.
 * Returns the total. */
int64_t arena_plan_bytes(const Program& P, int64_t* dev_out, int64_t* total_out);

/* The host activation arena's size (the RAD_DOMAIN_HOST buffers), which Ctx::bind_arena pins and
 * maps at the end of startup; 0 when the program declares none. */
int64_t arena_host_bytes(const Program& P);

/* The arena level a step of `rows` tokens runs at: the smallest level that holds it, or 0 (the
 * full plan) when none does. `level_rows[k]` is the largest step level k serves; entry 0 is the
 * full plan and is never chosen by size. */
int arena_level_for(const std::vector<int64_t>& level_rows, int64_t rows);
}  /* namespace rad */
