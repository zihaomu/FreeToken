/* arena.cpp -- one allocation, sized once. See arena.h for why there is no allocate(). */
#include "arena.h"

#include <algorithm>

namespace rad {

int Arena::init(int64_t plan_bytes, int64_t scratch_bytes, int kind, int regions,
                int64_t scratch_bytes1) {
    release();
    if (plan_bytes < 0 || scratch_bytes < 0 || regions < 1) return RAD_E_INVAL;
    if (scratch_bytes1 < 0 || scratch_bytes1 > scratch_bytes) scratch_bytes1 = scratch_bytes;

    /* Scratch starts on a page boundary, so a workspace gets 4 KiB alignment for at most one
     * page of padding. */
    const int64_t off   = align_up(plan_bytes, RAD_ALIGN_UNIT);
    const int64_t pitch  = align_up(scratch_bytes, RAD_ALIGN_UNIT);
    const int64_t pitch1 = align_up(scratch_bytes1, RAD_ALIGN_UNIT);
    const int64_t tail   = pitch + pitch1 * (regions - 1);
    int64_t       total = off + tail;

    /* Never zero bytes: a null base would turn a plan bug into an anonymous segfault. */
    if (total == 0) total = RAD_ALIGN_UNIT;

    void* p = rad_dev_alloc(total, kind);
    if (!p) {
        RAD_ERR("arena: %s of %s failed: %s",
                kind == RAD_MEM_DEVICE ? "device allocation" : "host allocation",
                humanb(total).c_str(), rad_dev_last_error());
        return RAD_E_NOMEM;
    }

    base_          = static_cast<char*>(p);
    kind_          = kind;
    bytes_         = total;
    scratch_off_   = off;
    scratch_bytes_ = scratch_bytes;
    scratch_pitch_ = pitch;
    scratch_bytes1_ = regions > 1 ? scratch_bytes1 : 0;
    scratch_pitch1_ = regions > 1 ? pitch1 : 0;
    scratch_regions_ = regions;
    return RAD_OK;
}

void Arena::release() {
    if (base_) rad_dev_free(base_, kind_);
    base_          = nullptr;
    bytes_         = 0;
    scratch_off_   = 0;
    scratch_bytes_ = 0;
    scratch_pitch_ = 0;
    scratch_regions_ = 1;
}


int64_t arena_host_bytes(const Program& P) {
    int64_t host_hw = 0;
    for (size_t i = 1; i < P.buffers.size(); ++i) {
        const BufferInfo& bi = P.buffers[i];
        if (bi.arena_offset < 0 || bi.decl.domain != RAD_DOMAIN_HOST) continue;
        int64_t bytes = bi.bytes;
        if (bytes <= 0) {
            int64_t n = 1;
            for (uint32_t d = 0; d < bi.decl.rank; ++d) n *= bi.decl.shape[d];
            bytes = rad_dtype_bytes(bi.decl.dtype, n);
        }
        host_hw = std::max(host_hw, bi.arena_offset + bytes);
    }
    return host_hw;
}

/* Assumes the plan gives host-domain buffers their own offset space (BufferInfo carries one
 * arena_offset). If it interleaved them with device buffers, both arenas would come out sparse:
 * correct, but paying VRAM for host activations. */
int64_t arena_plan_bytes(const Program& P, int64_t* dev_out, int64_t* total_out) {
    int64_t dev_hw = 0, host_hw = 0;
    for (size_t i = 1; i < P.buffers.size(); ++i) {
        const BufferInfo& bi = P.buffers[i];
        if (bi.arena_offset < 0) continue;      /* the plan did not run; bind_arena refuses by name */
        int64_t bytes = bi.bytes;
        if (bytes <= 0) {
            int64_t n = 1;
            for (uint32_t d = 0; d < bi.decl.rank; ++d) n *= bi.decl.shape[d];
            bytes = rad_dtype_bytes(bi.decl.dtype, n);
        }
        int64_t& hw = (bi.decl.domain == RAD_DOMAIN_HOST) ? host_hw : dev_hw;
        hw = std::max(hw, bi.arena_offset + bytes);
    }
    /* With no host-domain buffers the plan's own total is authoritative: it may carry alignment
     * slack the high-water mark does not see. */
    int64_t dev_bytes = host_hw > 0 ? dev_hw : std::max(dev_hw, P.arena_bytes);
    /* The staging region sits above every buffer, so it is a separate term. */
    if (P.arena_stage_bytes > 0 && P.arena_stage_off >= 0)
        dev_bytes = std::max(dev_bytes, P.arena_stage_off + P.arena_stage_bytes);
    const int64_t s1 = P.scratch_bytes_lane1 < 0 || P.scratch_bytes_lane1 > P.scratch_bytes
                           ? P.scratch_bytes : P.scratch_bytes_lane1;
    int64_t total = align_up(dev_bytes, RAD_ALIGN_UNIT) + align_up(P.scratch_bytes, RAD_ALIGN_UNIT) +
                    align_up(s1, RAD_ALIGN_UNIT) * (std::max<int64_t>(P.scratch_regions, 1) - 1);
    if (total == 0) total = RAD_ALIGN_UNIT;     /* Arena::init's floor, for the same reason */
    if (dev_out)   *dev_out = dev_bytes;
    if (total_out) *total_out = total;
    return total;
}
int arena_level_for(const std::vector<int64_t>& level_rows, int64_t rows) {
    int level = 0;
    for (size_t k = 1; k < level_rows.size(); ++k)
        if (level_rows[k] >= rows && (level == 0 || level_rows[k] < level_rows[(size_t)level]))
            level = (int)k;
    return level;
}
}  /* namespace rad */
