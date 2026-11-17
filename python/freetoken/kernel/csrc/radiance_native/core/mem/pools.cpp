/* pools.cpp -- see pools.h. */
#include <algorithm>
#include "mem/pools.h"

#include <cstdlib>
#include <cstring>
#include <new>

namespace rad {

/* ------------------------------------------------------------------ allocators */

namespace {

class DeviceAllocator final : public MemAllocator {
public:
    void* alloc(int64_t bytes, int kind) override { return rad_dev_alloc(bytes, kind); }
    void  free(void* p, int kind) override { rad_dev_free(p, kind); }
    const char* name() const override { return "device"; }
};

/* aligned_alloc at the 2 MiB pool alignment (the .rad pool alignment, huge-page mappable, spec
 * §4.2), so fixed-stride slots in a slab stay aligned. Used by the no-ROCm build and the tests. */
class MallocAllocator final : public MemAllocator {
public:
    void* alloc(int64_t bytes, int kind) override {
        (void)kind;
        if (bytes <= 0) return nullptr;
        int64_t n = align_up(bytes, (int64_t)RAD_ALIGN_POOL);
        void* p = ::aligned_alloc((size_t)RAD_ALIGN_POOL, (size_t)n);
        return p;
    }
    void free(void* p, int kind) override { (void)kind; ::free(p); }
    const char* name() const override { return "malloc"; }
};

}  /* namespace */

MemAllocator& rad_device_allocator() { static DeviceAllocator a; return a; }
MemAllocator& rad_malloc_allocator() { static MallocAllocator a; return a; }

/* ------------------------------------------------------------------ Shortfall */

std::string Shortfall::str() const {
    if (!refused.empty())
        return fmt("%s: %s of %s refused: %s", pool.c_str(), what.c_str(),
                   humanb(wanted).c_str(), refused.c_str());
    return fmt("%s: %s wanted %s, %s available, short by %s",
               pool.c_str(), what.c_str(), humanb(wanted).c_str(),
               humanb(available).c_str(), humanb(missing()).c_str());
}

/* ------------------------------------------------------------------ Pool */

Pool::~Pool() { release(); }

void Pool::release() {
    /* An elastic pool's base belongs to vm_, not the allocator, so it must not reach
     * MemAllocator::free. */
    if (elastic_) {
        vm_.release();
        elastic_ = false;
    } else if (base_ && alloc_) {
        alloc_->free(base_, kind_);
    }
    base_ = nullptr;
    bytes_ = used_ = 0;
}

void Pool::reset(std::string_view nm, int64_t bytes, int kind, MemAllocator* a) {
    release();
    name_ = std::string(nm);
    bytes_ = bytes;
    kind_ = kind;
    alloc_ = a;
    shortfalls_.clear();
    res_.clear();
}

int Pool::init(std::string_view nm, int64_t bytes, int kind, MemAllocator* a) {
    reset(nm, bytes, kind, a);
    if (bytes_ <= 0) return RAD_OK;

    base_ = a->alloc(bytes_, kind);
    if (!base_) {
        /* Refuse; never retry smaller (spec §6). Record the driver's reason. */
        const char* why = a == &rad_device_allocator() ? rad_dev_last_error() : nullptr;
        shortfalls_.push_back({name_, "backing allocation", bytes_, 0,
                               why && *why ? why : "the allocator returned nothing"});
        bytes_ = 0;
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

void* Pool::reserve(std::string_view what, int64_t bytes, int64_t align, int64_t* off_out) {
    if (bytes < 0 || align <= 0) return nullptr;
    int64_t off = align_up(used_, align);
    if (off + bytes > bytes_) {
        shortfalls_.push_back({name_, std::string(what), bytes, bytes_ - off});
        return nullptr;
    }
    used_ = off + bytes;
    res_.push_back({std::string(what), off, bytes});
    if (off_out) *off_out = off;
    return (char*)base_ + off;
}

/* ------------------------------------------------------------------ elastic */

int Pool::init_elastic(std::string_view nm, int64_t bytes, int kind, MemAllocator* a) {
    /* Without virtual memory management this is init(): commits succeed against memory that is
     * already there and the pool holds its maximum for the whole run. */
    if (!VMem::supported() || kind != RAD_MEM_DEVICE)
        return init(nm, bytes, kind, a);

    reset(nm, bytes, kind, a);
    if (bytes_ <= 0) return RAD_OK;

    const int rc = vm_.reserve(bytes_, RAD_ALIGN_POOL);
    if (rc < 0) {
        /* Refuse; never retry smaller. A failed address reservation on a 64-bit card is a bug,
         * not a budget limit. */
        const char* why = rad_dev_last_error();
        shortfalls_.push_back({ name_, "address reservation", bytes_, 0,
                                *why ? why : "the device layer gave no reason" });
        bytes_ = 0;
        return rc;
    }
    base_ = vm_.base();
    /* bytes_ stays the requested budget, not vm_.size() (rounded up to a granule): using the
     * rounded size would let reserve() grant more than the split allows. The rounded tail stays
     * reserved so the last reservation's commit can round outward into it. */
    elastic_ = true;
    return RAD_OK;
}

int Pool::commit(int64_t off, int64_t bytes) {
    if (!elastic_) {
        /* A fixed pool is backed everywhere inside its range and nowhere outside it. */
        return (off >= 0 && bytes >= 0 && off + bytes <= bytes_) ? RAD_OK : RAD_E_INVAL;
    }
    return vm_.commit(off, bytes);
}

int Pool::decommit(int64_t off, int64_t bytes) {
    if (!elastic_) return RAD_OK;       /* nothing to give back */
    return vm_.decommit(off, bytes);
}

std::string Pool::report() const {
    std::string s = fmt("%-14s %10s budget, %10s used, %10s free\n",
                        name_.c_str(), humanb(bytes_).c_str(), humanb(used_).c_str(),
                        humanb(avail()).c_str());
    for (const auto& r : res_)
        s += fmt("    %-24s %10s @ +%lld\n", r.what.c_str(), humanb(r.bytes).c_str(),
                 (long long)r.off);
    for (const auto& f : shortfalls_) s += "    " + f.str() + "\n";
    return s;
}

/* ------------------------------------------------------------------ Pools */

int Pools::configure(const Config& cfg, int64_t vram_capacity,
                     MemAllocator* device, MemAllocator* host) {
    configured_ = false;
    fit_report_.clear();

    if (!device) device = &rad_device_allocator();
    if (!host)   host   = device;

    const int64_t MiB = 1024 * 1024;
    const int64_t w  = cfg.vram_weights_mib  * MiB;
    const int64_t kv = cfg.vram_kv_mib       * MiB;
    const int64_t hp = cfg.host_pool_total_mib() * MiB;

    /* The budgets were resolved by core/mem/vram_budget.cpp (from the card, the measured driver
     * residency and the computed arena); this refuses rather than re-deciding them (spec §6). A
     * zero means the resolver never ran, which is a wiring bug: one that leaves the cache nothing
     * refuses there, with the budget. */
    std::string missing;
    if (cfg.vram_weights_mib <= 0) missing += "  vram_weights resolved to 0 MiB\n";
    if (cfg.vram_kv_mib      <= 0) missing += "  vram_kv resolved to 0 MiB\n";
    if (!missing.empty()) {
        fit_report_ = "VRAM pool budgets are empty: the budget resolver (core/mem/vram_budget.cpp) "
                      "did not run before the pools were configured:\n" + missing;
        RAD_ERR("%s", fit_report_.c_str());
        return RAD_E_INVAL;
    }

    const int64_t want = w + kv;
    if (vram_capacity > 0 && want > vram_capacity) {
        fit_report_ = fmt(
            "VRAM budgets do not fit:\n"
            "  vram_weights   %10s\n"
            "  vram_kv        %10s\n"
            "  ------------------------\n"
            "  total          %10s\n"
            "  card total     %10s\n"
            "  over by        %10s\n"
            "The resolved split is larger than the card, which the resolver does not produce: a "
            "wiring bug (spec §6).\n",
            humanb(w).c_str(), humanb(kv).c_str(),
            humanb(want).c_str(), humanb(vram_capacity).c_str(),
            humanb(want - vram_capacity).c_str());
        RAD_ERR("%s", fit_report_.c_str());
        return RAD_E_NOMEM;
    }

    int st = RAD_OK;
    int s;
    /* Order matters only for diagnostics: the two big pools go first so a failure names the one
     * that could not be served. */
    if ((s = weights_.init("vram_weights", w,  RAD_MEM_DEVICE, device)) < 0) st = s;
    /* The KV pool is always elastic. Its budget still caps what the cache may address; elasticity
     * only decides whether the unused part has VRAM behind it. What the cache releases, the
     * expert plane can promote into, so the split follows actual demand. There is no fixed-pool
     * option: it would hold every block whether used or not while experts stream over the link.
     * Without virtual memory management init_elastic() allocates the range whole. */
    if ((s = kv_.init_elastic("vram_kv", std::max(kv, cfg.kv_reserve), RAD_MEM_DEVICE, device)) < 0)
        st = s;
    /* Mapped, not merely pinned. The host pool holds the mover's staging buffers and O_DIRECT
     * landing zone, and the zero-copy execution site (spec §5.1), where kernels read weights over
     * the link and so need a kernel-addressable base: with hipHostMallocDefault,
     * rad_dev_device_ptr returns null in Mover::init and Site::DeviceZeroCopy experts get an
     * unusable pointer. On the host backend all memory kinds are the same aligned pages. */
    if ((s = host_   .init("host_pool",   hp, RAD_MEM_HOST_MAPPED, host)) < 0) st = s;

    fit_report_  = weights_.report();
    fit_report_ += kv_.report();
    fit_report_ += host_.report();

    if (st < 0) {
        RAD_ERR("pool allocation refused:\n%s", fit_report_.c_str());
        return st;
    }
    configured_ = true;
    return RAD_OK;
}

std::vector<Shortfall> Pools::shortfalls() const {
    std::vector<Shortfall> out;
    for (const Pool* p : { &weights_, &kv_, &host_ })
        for (const auto& f : p->shortfalls()) out.push_back(f);
    return out;
}

}  /* namespace rad */
