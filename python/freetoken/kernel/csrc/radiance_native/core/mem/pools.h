/* pools.h -- explicit pool budgets (spec §6).
 *
 * Every pool is budgeted explicitly: no profiling pass, no utilisation fraction. The KV vs
 * weights split depends on the workload, and an engine that re-optimises it on its own does not
 * produce reproducible benchmarks.
 *
 * Two structural properties:
 *
 *   1. A Pool holds a base, a size and a cursor, and no pointer to any other Pool. The mover is
 *      handed Pools::weights() only, so expert promotion (spec §5.4) cannot reach a KV block.
 *
 *   2. A reservation that does not fit returns null and records the shortfall by name and size.
 *      It never falls through to another pool, shrinks an earlier reservation or rounds down;
 *      the engine refuses to start instead.
 */
#pragma once
#include "rad_core.h"

#include "../device/device.h"

#include <string>
#include <string_view>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ the allocator seam */
/* Pools allocate through this interface rather than rad_dev_alloc so they run without ROCm: the
 * host backend is a primary target (docs/IMPLEMENTATION.md). The device allocator is the default;
 * the malloc one serves the no-device build and the tests. */
class MemAllocator {
public:
    virtual ~MemAllocator() = default;
    virtual void* alloc(int64_t bytes, int kind) = 0;   /* kind is RAD_MEM_* */
    virtual void  free(void* p, int kind) = 0;
    virtual const char* name() const = 0;
};

MemAllocator& rad_device_allocator();   /* rad_dev_alloc / rad_dev_free */
MemAllocator& rad_malloc_allocator();   /* aligned_alloc; host-only builds and tests */

/* ------------------------------------------------------------------ shortfalls */
/* What did not fit, by name and by how much. Every refusal here produces one, not just a bare
 * RAD_E_NOMEM. */
struct Shortfall {
    std::string pool;       /* which budget: "vram_weights", "vram_kv", ... */
    std::string what;       /* what was being reserved when it ran out */
    int64_t     wanted = 0;
    int64_t     available = 0;
    /* Set when the allocator refused a backing allocation: its reason (out of memory, pinned
     * limit, mapping failure). `available` is meaningless then. */
    std::string refused;
    int64_t     missing() const { return wanted - available; }
    std::string str() const;
};

/* ------------------------------------------------------------------ Pool */
/* One reservation, made once, that never grows, never shrinks and never borrows. */
class Pool {
public:
    Pool() = default;
    ~Pool();
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    const char* name()  const { return name_.c_str(); }
    int64_t     bytes() const { return bytes_; }
    int64_t     used()  const { return used_; }
    int64_t     avail() const { return bytes_ - used_; }
    void*       base()  const { return base_; }
    int         kind()  const { return kind_; }

    /* Carve `bytes` at `align`. Returns the pointer, or null with a recorded shortfall if it does
     * not fit.
     * `off_out` receives the byte offset from base, which fixed-stride slabs use because the base
     * may be a device pointer the host cannot dereference. */
    void* reserve(std::string_view what, int64_t bytes, int64_t align, int64_t* off_out = nullptr);

    /* True when nothing was refused. Checked once at the end of startup so the report names every
     * failure, not only the first (spec §3.1). */
    bool ok() const { return shortfalls_.empty(); }

    /* ---- elastic pools ----------------------------------------------------------------------
     *
     * An elastic pool reserves its whole address range and backs only the parts that are
     * committed. Offsets from reserve() are final, so strides computed against the full range
     * stay valid while unused pages go back to the card for another pool. Addresses cannot move
     * between pools; pages can.
     *
     * Used by the KV cache and the expert slab. A KV block can be released only once nothing
     * holds it; an expert slot is a copy of a weight that also lives in host memory or the
     * container, so the mover can release it at will. That policy lives with the callers.
     *
     * Without virtual memory management (VMem in core/device/device.h), init_elastic() allocates
     * the whole range like init() and commit()/decommit() succeed as no-ops, so callers run
     * unchanged against a fixed pool that holds its maximum (allowed by spec §6). */
    bool    elastic()   const { return elastic_; }
    int64_t committed() const { return elastic_ ? vm_.committed() : bytes_; }

    /* Back [off, off+bytes) so a kernel may touch it. Rounds outward to the commit granule and is
     * idempotent: a pool growing to n may commit [0, n) each time and pays only for the new part. */
    int  commit(int64_t off, int64_t bytes);

    /* Release the pages. Rounds inward, so a granule with any byte still in use stays backed. */
    int  decommit(int64_t off, int64_t bytes);

    const std::vector<Shortfall>& shortfalls() const { return shortfalls_; }

    std::string report() const;

private:
    /* Reserve the address range and back nothing. `bytes` is the maximum the pool may ever hold
     * and what reserve() carves against; committed() is what it costs the card. Falls back to a
     * whole allocation without virtual memory management. */
    int init_elastic(std::string_view nm, int64_t bytes, int kind, MemAllocator* a);

    friend class Pools;
    int init(std::string_view nm, int64_t bytes, int kind, MemAllocator* a);
    /* release(), then name the pool and set its budget with nothing reserved. */
    void reset(std::string_view nm, int64_t bytes, int kind, MemAllocator* a);
    void release();

    std::string   name_;
    int64_t       bytes_ = 0;
    int64_t       used_  = 0;
    void*         base_  = nullptr;
    int           kind_  = RAD_MEM_DEVICE;
    bool          elastic_ = false;
    VMem          vm_;
    MemAllocator* alloc_ = nullptr;
    std::vector<Shortfall> shortfalls_;
    struct Res { std::string what; int64_t off, bytes; };
    std::vector<Res> res_;
};

/* ------------------------------------------------------------------ Pools */
class Pools {
public:
    /* `vram_capacity` is what the device reports free; 0 skips the capacity check (the host
     * backend has no meaningful answer).
     *
     * Refuses, by name and size, rather than shrinking anything:
     *   - if the weight or KV budget is zero (the resolver did not run);
     *   - if weights + kv exceeds the card;
     *   - if a pool's backing allocation fails.
     */
    int configure(const Config& cfg, int64_t vram_capacity,
                  MemAllocator* device = nullptr, MemAllocator* host = nullptr);

    Pool& weights()     { return weights_; }
    Pool& kv()          { return kv_; }
    Pool& host_pinned() { return host_; }

    const Pool& weights()     const { return weights_; }
    const Pool& kv()          const { return kv_; }
    const Pool& host_pinned() const { return host_; }

    bool configured() const { return configured_; }

    /* Printed at startup and on refusal. Populated on success too, for the placement report's
     * pool totals (spec §16). */
    const std::string& fit_report() const { return fit_report_; }

    std::vector<Shortfall> shortfalls() const;

private:
    Pool weights_, kv_, host_;
    std::string fit_report_;
    bool configured_ = false;
};

}  /* namespace rad */
