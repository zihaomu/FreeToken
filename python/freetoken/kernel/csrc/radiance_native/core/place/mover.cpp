/* mover.cpp -- the mover's mechanics; mover.h has the design.
 *
 * Everything here runs on the dispatch thread. Transfers run asynchronously on the mover's
 * stream, gated behind an event recorded after the layer's launches, but every table edit is
 * applied by this thread at a layer boundary (see Mover::Move).
 */
#include "mover.h"
#include "../device/device.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <queue>
#include <unordered_map>

namespace rad {

/* ------------------------------------------------------------------ the file tier */
int FileTier::open(const std::string& path) {
    /* No RAD_DIO_ALLOW_BUFFERED: a silent buffered fallback would make the tier a page-cache pump
     * that is several times slower, evicts the container's own mapped pages and does not show in
     * a profile. A tier pointed at tmpfs gets an error instead. */
    const int rc = f_.open(path.c_str(), RAD_DIO_READ);
    if (rc < 0)
        RAD_ERR("file tier: cannot open %s with O_DIRECT. The tier needs a filesystem that "
                "supports it -- tmpfs and some overlay mounts do not.", path.c_str());
    else
        size_ = f_.size();
    return rc;
}

int FileTier::read(uint64_t offset, int64_t bytes, void* dst, int64_t dst_cap) {
    if (!f_.is_open()) return RAD_E_STATE;
    if (!dst || bytes <= 0) return RAD_E_INVAL;
    /* The extent rounded up to the alignment: the unit stride already covers it, and reading
     * one extra block is cheaper than a branch on the step path -- unless the rounding crosses
     * the container's end: the final unit's bytes can end at an unaligned EOF, and pread_exact
     * calls that short read an error. Clamp to the file, which sends the tail through read_at's
     * bounce path (it tolerates an EOF short read) instead of the exact fast path. */
    int64_t want = direct_align(bytes);
    if (size_ >= 0 && (int64_t)offset + want > size_) want = size_ - (int64_t)offset;
    if (want > dst_cap) {
        RAD_ERR("file tier: a %s read needs a %s aligned buffer, the staging slot is %s",
                humanb(bytes).c_str(), humanb(want).c_str(), humanb(dst_cap).c_str());
        return RAD_E_SCRATCH;
    }
    RAD_TRY(f_.read_at(dst, (int64_t)offset, want));
    return RAD_OK;
}

/* ------------------------------------------------------------------ residency */
void Residency::init(const Program& prog) {
    slot_.assign(prog.weights.size(), WeightSlot{});
    noted_.assign(prog.weights.size(), 0);
    changed_.clear();
    ++epoch_;
}

/* The generation bump is the whole staleness protocol, with no lock: at this point both copies of
 * the bytes are valid (the mover wrote a shadow slot and has not freed the old one), so a reader
 * on either side of the write names correct bytes. The quarantine keeps that true. Why both
 * writers bump: mover.h. */
void Residency::begin_transfer(rad_weight w, void* ptr, RadEvent ready) {
    if ((size_t)w >= slot_.size()) return;
    slot_[w].ptr = ptr;
    slot_[w].ready = ready;
    ++slot_[w].generation;
    ++epoch_;
    note(w);
}

/* ------------------------------------------------------------------ the mover */
Mover::~Mover() {
    /* Drain the stream first: its queued gate waits, event records and copies refer to the
     * events and memory freed below. */
    if (mover_) {
        (void)flush_moves();
        rad_stream_sync(mover_);
        rad_stream_destroy(mover_);
    }
    for (Move& m : moves_) if (m.ev) rad_event_destroy(m.ev);
    for (RadEvent e : lead_) if (e) rad_event_destroy(e);
    for (RadEvent e : win_ev_) if (e) rad_event_destroy(e);
    if (gate_) rad_event_destroy(gate_);
    /* Only what this object allocated; a Pool reservation belongs to the Pool. */
    if (owns_memory_) {
        for (ClassPools& c : pools_) {
            if (c.slab_base) rad_dev_free(c.slab_base, RAD_MEM_DEVICE);
            if (c.pool_base) rad_dev_free(c.pool_base, RAD_MEM_HOST_PINNED);
        }
        if (static_vram_) rad_dev_free(static_vram_, RAD_MEM_DEVICE);
        if (stage_) rad_dev_free(stage_, RAD_MEM_HOST_PINNED);
        if (file_vram_) rad_dev_free(file_vram_, RAD_MEM_DEVICE);
        if (windows_) rad_dev_free(windows_, RAD_MEM_HOST_PINNED);
    }
    /* The pageable region is always ours: Pools has no pageable pool, since pageable memory is
     * not a device-visible budget and nothing else contends for it. */
    if (pageable_) rad_dev_free(pageable_, RAD_MEM_HOST);
}

/* The one place the mover asks for memory. Device and pinned memory come out of the resolved
 * budgets (spec §6); the direct allocation is for the placement test, which runs without a Pools
 * (init() warns). */
void* Mover::reserve(const char* what, int64_t bytes, int kind) {
    if (bytes <= 0) return nullptr;
    if (budget_) {
        Pool& p = kind == RAD_MEM_DEVICE ? budget_->weights() : budget_->host_pinned();
        return p.reserve(what, bytes, RAD_ALIGN_UNIT);
    }
    owns_memory_ = true;
    return rad_dev_alloc(bytes, kind);
}

int Mover::init(const Program& prog, const Plan& plan, const MoverConfig& cfg, Pools* budget) {
    prog_ = &prog;
    plan_ = &plan;
    cfg_ = cfg;
    budget_ = budget;
    stats_ = MoverCounters{};
    if (!budget_)
        RAD_WARN("mover: no Pools given; allocating the slab and host pool directly (test only)");

    unit_.assign(plan.units.size(), UnitState{});
    copying_.assign(plan.units.size(), 0);
    pool_layer_.assign(plan.classes.size(), {});
    donors_.assign(plan.classes.size(), DonorRank{});
    n_layers_ = 0;
    for (const MoveUnit& mu : plan.units) n_layers_ = std::max(n_layers_, mu.layer + 1);
    pools_.clear();
    pools_.resize(plan.classes.size());

    /* ---- how many slots each class needs, counted before anything is allocated ------------- */
    std::vector<int64_t> n_slab(plan.classes.size(), 0), n_pool(plan.classes.size(), 0);
    int64_t static_need = 0, pageable_need = 0;
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const MoveUnit& mu = plan.units[u];
        UnitState& us = unit_[u];
        us.tier = mu.tier;
        us.site = mu.site;
        us.klass = mu.slab_class;
        us.bytes = mu.bytes;
        if (mu.slab_class < 0) continue;
        if (mu.tier == Tier::VRAM && mu.slab_slot >= 0) {
            n_slab[(size_t)mu.slab_class] = std::max(n_slab[(size_t)mu.slab_class],
                                                     (int64_t)mu.slab_slot + 1);
        } else if (mu.tier == Tier::VRAM) {
            us.static_off = align_up(static_need, RAD_ALIGN_UNIT);
            static_need = us.static_off + mu.bytes;
        } else if (mu.tier == Tier::HostPinned) {
            us.pool_slot = n_pool[(size_t)mu.slab_class]++;
            pool_layer_[(size_t)mu.slab_class].push_back(mu.layer);
        } else if (mu.tier == Tier::HostPageable) {
            us.static_off = align_up(pageable_need, RAD_ALIGN_UNIT);
            pageable_need = us.static_off + mu.bytes;
        }
    }

    /* ---- allocate ------------------------------------------------------------------------- */
    int64_t total_shadow = 0, total_turn = 0, total_slab = 0, total_pool = 0;
    for (size_t c = 0; c < plan.classes.size(); ++c) {
        const SlabClass& sc = plan.classes[c];
        ClassPools& cp = pools_[c];
        cp.stride = sc.stride;

        /* Shadow slots follow the resident ones, so resident indices do not depend on the engine;
         * a static plan has none. Use the plan's count, which the planner budgeted. */
        const int64_t shadow = plan.dynamic ? sc.n_shadow : 0;
        /* Shadow and staging slots share one free list: both are somewhere to copy into that
         * nothing reads. */
        const int64_t slab_slots = n_slab[c] + shadow + sc.n_stage;
        /* Flex slots are added later by lend_from(), once the lender's memory is carved. */
        cp.n_fixed = slab_slots;
        cp.n_work = shadow + sc.n_stage;
        cp.n_turn = std::min<int64_t>(std::max(cfg.turnover_slots, 0),
                                      shadow * RAD_PLACE_TURNOVER_SLOTS / RAD_PLACE_SHADOW_SLOTS);
        total_turn += cp.n_turn;
        cp.slab.init(slab_slots, sc.stride);
        total_slab += slab_slots;
        total_pool += n_pool[c];
        if (slab_slots > 0) {
            /* Tail padding: a kernel assembling a packed code window from a wide read may read a
             * few bytes past the last slot; without padding that runs off the allocation and
             * faults the card. */
            cp.slab_base = reserve(sc.name.c_str(), slab_slots * sc.stride + RAD_ALIGN_UNIT,
                                   RAD_MEM_DEVICE);
            if (!cp.slab_base) {
                RAD_ERR("mover: no VRAM for the '%s' slab: %lld slots of %s",
                        sc.name.c_str(), (long long)slab_slots, humanb(sc.stride).c_str());
                return RAD_E_NOMEM;
            }
        }
        for (int64_t s = n_slab[c]; s < slab_slots; ++s) cp.slab.add_free(s);
        total_shadow += shadow + sc.n_stage;

        cp.pool.init(n_pool[c], sc.stride);
        if (n_pool[c] > 0) {
            cp.pool_base = reserve(sc.name.c_str(), n_pool[c] * sc.stride, RAD_MEM_HOST_PINNED);
            if (!cp.pool_base) {
                RAD_ERR("mover: no pinned host memory for the '%s' pool: %lld slots of %s",
                        sc.name.c_str(), (long long)n_pool[c], humanb(sc.stride).c_str());
                return RAD_E_NOMEM;
            }
            /* The pool's base as the card addresses it (equal to the host address on a unified
             * backend). Without one there is no zero-copy tier. */
            cp.pool_dev = rad_dev_device_ptr(cp.pool_base);
            /* Warned, not refused: a pool nothing zero-copies out of is not an error. The host
             * pointer is valid on the device only where the two coincide. A deployment's pool is
             * RAD_MEM_HOST_MAPPED (core/mem/pools.cpp), so this fires only on the test path. */
            if (!cp.pool_dev) {
                RAD_WARN("mover: the '%s' host pool has no device address; a kernel reading it "
                         "zero-copy would fault", sc.name.c_str());
                cp.pool_dev = cp.pool_base;
            }
        }
        /* The pool free list starts empty: every pooled unit occupies its slot. The engine primes
         * off the shadow slots and then alternates, a promotion freeing the pool slot a demotion
         * needs.
         *
         * Nothing is memset; the load phase writes every weight. rad_memset_async on
         * RAD_MEM_HOST fails on a real card but succeeds on the host backend, so use std::memset
         * if a host buffer ever needs clearing. The pool's cache cycle takes slots
         * on demand: a deposit with no free slot relabels the coldest idle unit back to the
         * file tier, copyless. */
    }

    if (static_need > 0) {
        static_vram_ = reserve("static weights", static_need, RAD_MEM_DEVICE);
        if (!static_vram_) {
            /* Log the budget arithmetic too: the slab, with its shadow and staging slots, was
             * reserved first, so this failure means the budget covered the resident set but not
             * the static weights beside it. */
            if (budget_)
                RAD_ERR("mover: the weight budget is %s and the slab already took %s of it, "
                        "leaving %s",
                        humanb(budget_->weights().bytes()).c_str(),
                        humanb(budget_->weights().used()).c_str(),
                        humanb(budget_->weights().avail()).c_str());
            RAD_ERR("mover: no VRAM for the %s of statically placed weights",
                    humanb(static_need).c_str());
            return RAD_E_NOMEM;
        }
        static_bytes_ = static_need;
    }
    if (pageable_need > 0) {
        pageable_ = rad_dev_alloc(pageable_need, RAD_MEM_HOST);
        if (!pageable_) {
            RAD_ERR("mover: no pageable host memory for %s of weights",
                    humanb(pageable_need).c_str());
            return RAD_E_NOMEM;
        }
        pageable_bytes_ = pageable_need;
    }

    /* ---- the file tier -------------------------------------------------------------------- */
    bool want_file = false;
    for (const MoveUnit& mu : plan.units) if (mu.tier == Tier::SSD) want_file = true;
    if (want_file) {
        if (cfg_.container_path.empty()) {
            RAD_ERR("mover: the plan puts weights on the file tier but no container path was "
                    "given");
            return RAD_E_INVAL;
        }
        RAD_TRY(file_.open(cfg_.container_path));
        /* Staging slots are sized by the widest unit actually on the file tier, not the widest
         * class. The widest class is usually the embedding table (gigabytes, and never on the
         * file tier), and sizing on it would reserve pinned memory for a weight that never
         * arrives, possibly failing the file tier for lack of memory. */
        int64_t widest = 0;
        for (const MoveUnit& mu : plan.units) {
            if (mu.tier != Tier::SSD || mu.slab_class < 0) continue;
            widest = std::max(widest, plan.classes[(size_t)mu.slab_class].stride);
        }
        stage_stride_ = FileTier::direct_align(widest);
        if (cfg_.stage_slots > 0 && stage_stride_ > 0) {
            stage_ = reserve("file-tier staging", (int64_t)cfg_.stage_slots * stage_stride_, RAD_MEM_HOST_PINNED);
            if (!stage_) {
                RAD_ERR("mover: no pinned memory for the file tier's %d staging slots of %s",
                        cfg_.stage_slots, humanb(stage_stride_).c_str());
                return RAD_E_NOMEM;
            }
        }
    }

    /* ---- the file stager's buffers ---------------------------------------------------------
     *
     * Exactly what the plan charged: two VRAM buffers from the weight budget and the read windows
     * from the host pool's file-staging reserve. The plan sizes them only when a routed unit is
     * on the file tier. */
    if (plan.file_stage_vram > 0) {
        if (!file_.ok()) {
            RAD_ERR("mover: the plan reads routed experts from the container and the file tier "
                    "did not open");
            return RAD_E_STATE;
        }
        file_vram_ = (char*)reserve("file-tier layer buffers", plan.file_stage_vram, RAD_MEM_DEVICE);
        if (!file_vram_) {
            RAD_ERR("mover: no VRAM for the two %s buffers a routed layer is read into from the "
                    "container", humanb(plan.file_layer_max).c_str());
            return RAD_E_NOMEM;
        }
        /* Two halves of what was charged, each at least the largest layer and aligned, so the
         * tail pad the plan added stays past the second buffer's end. */
        file_buf_bytes_ = (plan.file_stage_vram - RAD_ALIGN_UNIT) / 2 / RAD_ALIGN_UNIT * RAD_ALIGN_UNIT;
        if (file_buf_bytes_ < plan.file_layer_max) {
            RAD_ERR("mover: the plan charged %s for two file-tier layer buffers of %s",
                    humanb(plan.file_stage_vram).c_str(), humanb(plan.file_layer_max).c_str());
            return RAD_E_STATE;
        }
        if (plan.file_windows < (int64_t)RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES) {
            RAD_ERR("mover: the plan reads routed experts from the container but reserved no read "
                    "windows for them");
            return RAD_E_STATE;
        }
        windows_ = (char*)reserve("file-tier read windows",
                                  (int64_t)RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES,
                                  RAD_MEM_HOST_PINNED);
        if (!windows_) {
            RAD_ERR("mover: no pinned memory for the file tier's %d read windows of %s",
                    RAD_PLACE_FILE_WINDOWS, humanb(RAD_PLACE_FILE_WINDOW_BYTES).c_str());
            return RAD_E_NOMEM;
        }
        n_windows_ = RAD_PLACE_FILE_WINDOWS;

        /* A zeroed region the skipped units' table entries point at. A dead row that reads
         * through a dummy (a scale plane re-read, a dropped slot) sees zeros, which dequantise
         * finite -- pointing it at the staging buffer's base handed it real weight bytes whose
         * reinterpretation produced NaNs. */
        calm_ = (char*)reserve("file-tier dummy region", 8 << 20, RAD_MEM_HOST_PINNED);
        if (!calm_) {
            RAD_ERR("mover: no pinned memory for the file tier's dummy region");
            return RAD_E_NOMEM;
        }
        std::memset(calm_, 0, 8 << 20);
    }

    /* ---- the file tier's cache cycle (stage_file_units) -------------------------------------
     *
     * The stager covers the plan's file-tier units from the start; it may cover more
     * (cover_file_units). Each layer's file-tier bytes start as the plan's, which the plan sized
     * the stager's buffers for. */
    file_ok_.assign(plan.units.size(), 0);
    layer_file_.assign((size_t)std::max(n_layers_, 1), 0);
    file_cap_ = file_buf_bytes_;
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const MoveUnit& mu = plan.units[u];
        if (mu.tier != Tier::SSD || mu.access != RAD_ACCESS_CONDITIONAL) continue;
        file_ok_[u] = 1;
        if (mu.layer >= 0) layer_file_[(size_t)mu.layer] += align_up(mu.bytes, RAD_ALIGN_UNIT);
    }
    victims_.assign(plan.classes.size(), VictimRank{});

    /* ---- streams, events, move records ------------------------------------------------------ */
    RAD_TRY(rad_stream_create(&mover_, /*high_priority=*/0));
    /* Every event here is local: they order the mover's queue against this card's compute queue
     * and tell the host when a slot is free, and the host never reads the ordered bytes through
     * them. A copy out to host memory carries system scope on its own packets. */
    RAD_TRY(rad_event_create_local(&gate_));
    /* A window's event is waited on by the HOST, before the window is read into again. */
    win_ev_.assign((size_t)n_windows_, nullptr);
    win_rec_.assign((size_t)n_windows_, 0);
    for (RadEvent& e : win_ev_) RAD_TRY(rad_event_create_as(&e, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));

    const int lead = std::max(1, cfg_.max_lead_dispatches);
    lead_.assign((size_t)lead + 1, nullptr);
    for (RadEvent& e : lead_) RAD_TRY(rad_event_create_as(&e, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));

    /* Two records per free slot (shadow or turnover): a swap is two transfers, and with one
     * record per slot the demotion could never be in flight beside its promotion, so the resident
     * set would only grow. The floor lets a one-shadow plan still hold a swap. */
    const int64_t n_records = std::max<int64_t>(2 * (total_shadow + total_turn), 4);
    moves_.assign((size_t)n_records, Move{});
    for (Move& m : moves_) RAD_TRY(rad_event_create_local(&m.ev));

    /* ---- publish the initial addresses ------------------------------------------------------ */
    res_.init(prog);
    for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u) {
        unit_[(size_t)u].slab_slot = plan.units[(size_t)u].slab_slot;
        republish(u);
    }

    RAD_INFO("mover: %lld slab slots (%lld shadow, %lld more kept free out of a loan), %lld pinned "
             "pool slots, %s static vram%s",
             (long long)total_slab, (long long)total_shadow, (long long)total_turn,
             (long long)total_pool,
             humanb(static_bytes_).c_str(), file_.ok() ? ", file tier open" : "");
    /* The pageable region is charged to no budget, so this is the only place it is reported. */
    if (pageable_bytes_ > 0)
        RAD_INFO("mover: %s of pageable host memory for the units no tier could hold",
                 humanb(pageable_bytes_).c_str());
    return RAD_OK;
}

int Mover::bind_compute_stream(RadStream s) {
    if (!s) return RAD_E_INVAL;
    compute_ = s;
    return RAD_OK;
}

void* Mover::unit_base(int32_t unit) const {
    if (unit < 0 || (size_t)unit >= unit_.size()) return nullptr;
    const UnitState& us = unit_[(size_t)unit];
    if (us.klass < 0) return nullptr;
    const ClassPools& cp = pools_[(size_t)us.klass];
    if (us.slab_slot >= 0 && (cp.is_flex(us.slab_slot) || cp.slab_base))
        return cp.slot_ptr(us.slab_slot);
    if (us.tier == Tier::VRAM && us.static_off >= 0 && static_vram_)
        return (char*)static_vram_ + us.static_off;
    if (us.tier == Tier::HostPinned && us.pool_slot >= 0 && cp.pool_dev)
        return (char*)cp.pool_dev + us.pool_slot * cp.stride;
    if (us.tier == Tier::HostPageable && us.static_off >= 0 && pageable_)
        return (char*)pageable_ + us.static_off;
    /* An unstaged file-tier unit has no address; a kernel cannot read from disk, so a caller
     * that gets null must stage the unit or refuse. */
    return nullptr;
}

void Mover::republish(int32_t unit) {
    const MoveUnit& mu = plan_->units[(size_t)unit];
    char* base = (char*)unit_base(unit);
    for (size_t i = 0; i < mu.weights.size(); ++i)
        res_.publish(mu.weights[i], base ? base + mu.offset[i] : nullptr);
}

/* Whether a copy into this unit's slot is still in flight. A slot whose staging copy has not
 * landed cannot be released: the quarantine barrier is on the compute stream and says nothing
 * about an H2D copy on the mover's. Only Copying counts; a record past it has its bytes in place,
 * and counting it would hold the slot until the record recycles. */
bool Mover::unit_in_flight(int32_t unit) const {
    return unit >= 0 && (size_t)unit < copying_.size() && copying_[(size_t)unit] > 0;
}

bool Mover::stageable(int32_t unit) const {
    const UnitState& us = unit_[(size_t)unit];
    return us.tier == Tier::HostPinned && us.pool_slot >= 0 && us.slab_slot < 0 && us.klass >= 0 &&
           !unit_in_flight(unit);
}

int Mover::free_move_record() const {
    const size_t n = moves_.size();
    for (size_t k = 0; k < n; ++k) {
        const size_t i = (rec_cursor_ + k) % n;
        if (moves_[i].state == Move::State::Free) { rec_cursor_ = (i + 1) % n; return (int)i; }
    }
    return -1;
}

/* The ordering event, recorded on the compute stream once per dispatch and only when a move goes
 * out. Everything the mover issues this dispatch waits on it, and it is downstream of every
 * launch so far; re-recording it per move would restate the same ordering. */
int Mover::record_gate() {
    if (!compute_) {
        RAD_ERR("mover: no compute stream bound; a transfer cannot be ordered against the "
                "kernels that read the slot it targets");
        return RAD_E_STATE;
    }
    if (gate_dispatch_ == dispatch_) return RAD_OK;
    RAD_TRY(rad_event_record(gate_, compute_));
    gate_dispatch_ = dispatch_;
    return RAD_OK;
}

int Mover::begin_dispatch(int dispatch, HeatEngine* heat) {
    dispatch_ = dispatch;
    heat_ = heat;
    ++stats_.dispatches;
    RAD_TRY(retire(heat));

    /* The lead gate: a host wait bounding how far ahead of the cards this thread queues work,
     * so quarantine events on the compute stream signal soon enough to recycle slots
     * (MoverConfig::max_lead_dispatches). It never waits on a transfer. */
    if (compute_ && cfg_.max_lead_dispatches > 0) {
        RadEvent old = lead_[(size_t)lead_next_];
        if (stats_.dispatches > (uint64_t)lead_.size()) RAD_TRY(rad_event_sync(old));
        RAD_TRY(rad_event_record(old, compute_));
        lead_next_ = (lead_next_ + 1) % (int)lead_.size();
    }
    return RAD_OK;
}

/* ---- the exact prefetch: layer offload's whole policy -----------------------------------------
 *
 * Declare recorded every op with its weight operands in order, so each staged unit's deadline is
 * known. Transfers start as soon as a slot is free and stop at the first unit that cannot get
 * one rather than skipping ahead: the schedule is in deadline order, and serving a later unit
 * first would turn the prefetch into a cache.
 */
int Mover::prefetch_to_op(int32_t op_index) {
    if (!plan_) return RAD_E_STATE;

    /* A backwards op index means the next step started; the schedule is per step, so the cursor
     * rewinds. */
    if (op_index < last_op_) prefetch_cursor_ = 0;
    last_op_ = op_index;

    /* Release first, so slots this op finished with are free for the units it needs next. A
     * staged unit's slot goes back once its last_use_op has passed; the declared order rules out
     * a read outside [first_use_op, last_use_op]. */
    for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
        const UnitState& us = unit_[(size_t)u];
        if (us.site != Site::DeviceStaged || us.slab_slot < 0) continue;
        if (unit_in_flight(u)) continue;   /* its copy has not landed; the slot is not ours yet */
        const MoveUnit& mu = plan_->units[(size_t)u];
        if (mu.last_use_op >= 0 && mu.last_use_op < op_index) release_slot(u);
    }

    while (prefetch_cursor_ < (int32_t)plan_->prefetch.size()) {
        const Plan::Prefetch& pf = plan_->prefetch[(size_t)prefetch_cursor_];
        UnitState& us = unit_[(size_t)pf.unit];
        if (us.slab_slot >= 0) { ++prefetch_cursor_; continue; }   /* already in a slot */
        if (us.klass < 0) { ++prefetch_cursor_; continue; }

        const int64_t slot = pools_[(size_t)us.klass].slab.take();
        if (slot < 0) { ++stats_.starved_slab; break; }
        const int rc = start_stage(pf.unit, slot);
        if (rc < 0) {
            pools_[(size_t)us.klass].slab.add_free(slot);
            return rc;
        }
        ++prefetch_cursor_;
    }
    return RAD_OK;
}

int Mover::start_stage(int32_t unit, int64_t slab_slot) {
    const int rec = free_move_record();
    if (rec < 0) { ++stats_.no_record; return RAD_E_FULL; }
    RAD_TRY(record_gate());

    UnitState& us = unit_[(size_t)unit];
    ClassPools& cp = pools_[(size_t)us.klass];
    char* dst = cp.slot_ptr(slab_slot);

    const void* src = nullptr;
    if (us.tier == Tier::HostPinned && us.pool_slot >= 0)
        src = (char*)cp.pool_base + us.pool_slot * cp.stride;
    else if (us.tier == Tier::HostPageable && us.static_off >= 0)
        src = (char*)pageable_ + us.static_off;
    else if (us.tier == Tier::SSD) {
        /* A blocking host read inside the dispatch, counted in ssd_reads. */
        if (!file_.ok() || !stage_) return RAD_E_STATE;
        /* The ring has wrapped: the copies out of its slots may still be queued on the mover's
         * stream, and the pread below would overwrite a source they have not read yet. */
        if (stage_next_ == 0 && stats_.ssd_reads > 0) RAD_TRY(rad_stream_sync(mover_));
        char* stage = (char*)stage_ + (int64_t)stage_next_ * stage_stride_;
        stage_next_ = (stage_next_ + 1) % std::max(1, cfg_.stage_slots);
        /* Weight by weight, each to its own offset in the unit. A container entry holds a
         * weight's planes at 256-byte alignment while a unit lays weights out at their declared
         * alignment, so one read of the unit's span would misplace every weight after the first. */
        const MoveUnit& mu = plan_->units[(size_t)unit];
        for (size_t k = 0; k < mu.weights.size(); ++k) {
            const WeightInfo& w = prog_->weights[mu.weights[k]];
            RAD_TRY(file_.read(w.file_offset, w.stored_bytes, stage + mu.offset[k],
                               stage_stride_ - mu.offset[k]));
        }
        ++stats_.ssd_reads;
        stats_.ssd_bytes += (uint64_t)us.bytes;
        src = stage;
    }
    if (!src) return RAD_E_STATE;

    RAD_TRY(rad_event_wait(mover_, gate_));
    dev_page_invalidate(dst, us.bytes);
    RAD_TRY(rad_memcpy_async(dst, src, us.bytes, mover_));
    RAD_TRY(rad_event_record(moves_[(size_t)rec].ev, mover_));

    Move& m = moves_[(size_t)rec];
    m.state = Move::State::Copying;
    m.kind = Move::Kind::Stage;
    m.unit = unit;
    ++copying_[(size_t)unit];
    m.slab_slot = slab_slot;
    m.pool_slot = -1;
    stats_.h2d_bytes += (uint64_t)us.bytes;
    stats_.prefetch_bytes += (uint64_t)us.bytes;
    ++stats_.prefetched;

    /* Publish the destination now, with the event, not at retire. Staging is the one move whose
     * destination the run phase reads: work already issued uses the old address (the pinned
     * pool), and work issued from here on names the slab slot and waits on `ev` on the compute
     * stream. That wait is the only guard against a race; begin_transfer bumps the generation so
     * the run phase performs it. */
    us.slab_slot = slab_slot;
    char* base = (char*)unit_base(unit);
    const MoveUnit& mu = plan_->units[(size_t)unit];
    for (size_t i = 0; i < mu.weights.size(); ++i)
        res_.begin_transfer(mu.weights[i], base + mu.offset[i], m.ev);
    return RAD_OK;
}

/* ---- staging within a pass (see mover.h) ---------------------------------------------------- */
int Mover::stage_units(const std::vector<int32_t>& units, char* dst, int64_t cap, RadEvent after,
                       RadEvent done, std::vector<int32_t>* staged) {
    const StageSpan one{dst, cap};
    return stage_units(units, &one, 1, after, done, staged);
}

int Mover::stage_units(const std::vector<int32_t>& units, const StageSpan* spans, int n_spans,
                       RadEvent after, RadEvent done, std::vector<int32_t>* staged) {
    if (!plan_ || !spans || n_spans < 1 || n_spans > 2 || !staged || !done) return RAD_E_INVAL;
    for (int k = 0; k < n_spans; ++k)
        if (!spans[k].dst) return RAD_E_INVAL;
    staged->clear();
    stage_picks_.clear();
    for (int32_t u : units) {
        if (u < 0 || (size_t)u >= unit_.size()) return RAD_E_INVAL;
        if (!stageable(u)) continue;
        const UnitState& us = unit_[(size_t)u];
        const ClassPools& cp = pools_[(size_t)us.klass];
        const char* src = (const char*)cp.pool_base + us.pool_slot * cp.stride;
        stage_picks_.push_back(StagePick{src, u, us.bytes, align_up(us.bytes, RAD_ALIGN_UNIT),
                                         nullptr});
    }
    if (stage_picks_.empty()) return RAD_OK;
    /* Spans are filled in pool order, so units on neighbouring pool slots land side by side and
     * go as one copy. A layer's pooled experts were assigned slots together and mostly keep them,
     * so its ~90 units take far fewer copies, each a launch or DMA packet whatever its size. A
     * class's stride is its units' aligned size, so a run's source and destination advance alike.
     * A unit that does not fit the current span starts the next; one that fits no span is read
     * across the link in place. */
    std::sort(stage_picks_.begin(), stage_picks_.end(),
              [](const StagePick& a, const StagePick& b) { return a.src < b.src; });
    int k = 0;
    int64_t off = 0;
    size_t taken = 0;
    for (StagePick& p : stage_picks_) {
        if (off + p.need > spans[k].cap && k + 1 < n_spans && p.need <= spans[k + 1].cap) {
            ++k;
            off = 0;
        }
        if (off + p.need > spans[k].cap) { ++stats_.stage_overflow; continue; }
        p.dst = spans[k].dst + off;
        off += p.need;
        ++taken;
    }
    if (taken == 0) return RAD_OK;
    if (after) RAD_TRY(rad_event_wait(mover_, after));
    stage_segs_.clear();
    for (const StagePick& p : stage_picks_) {
        if (!p.dst) continue;
        RadCopySeg* run = stage_segs_.empty() ? nullptr : &stage_segs_.back();
        if (run && p.src - (const char*)run->src == p.dst - (char*)run->dst)
            run->bytes = (int64_t)(p.dst - (char*)run->dst) + p.bytes;
        else
            stage_segs_.push_back(RadCopySeg{p.src, p.dst, p.bytes});
        staged->push_back(p.u);
        stats_.h2d_bytes += (uint64_t)p.bytes;
        stats_.staged_bytes += (uint64_t)p.bytes;
    }
    RAD_TRY(rad_memcpy_list_async(stage_segs_.data(), (int)stage_segs_.size(), RAD_COPY_YIELD,
                                  mover_));
    RAD_TRY(rad_event_record(done, mover_));
    for (const StagePick& p : stage_picks_) {
        if (!p.dst) continue;
        const MoveUnit& mu = plan_->units[(size_t)p.u];
        for (size_t i = 0; i < mu.weights.size(); ++i) res_.publish(mu.weights[i], p.dst + mu.offset[i]);
    }
    ++stats_.stage_layers;
    stats_.staged_units += (uint64_t)staged->size();
    stats_.stage_copies += (uint64_t)stage_segs_.size();
    return RAD_OK;
}

void Mover::unstage_units(const std::vector<int32_t>& staged) {
    for (int32_t u : staged) republish(u);
}

/* The same choice of units as stage_units, without a cap. */
int64_t Mover::stage_need(const std::vector<int32_t>& units, int64_t* largest) const {
    int64_t total = 0, most = 0;
    for (int32_t u : units) {
        if (u < 0 || (size_t)u >= unit_.size() || !stageable(u)) continue;
        const int64_t need = align_up(unit_[(size_t)u].bytes, RAD_ALIGN_UNIT);
        total += need;
        most = std::max(most, need);
    }
    if (largest) *largest = most;
    return total;
}

/* ---- a routed layer read from the container (see mover.h) ---------------------------------- */
namespace {

/* The largest gap a run reads through instead of starting another pread. A routed layer's
 * weights sit together in the container, each on RAD_ALIGN_UNIT, so a gap between two file-tier
 * weights is either alignment padding (cheaper to read than a second pread) or units on another
 * tier (megabytes read for nothing). 256 KiB is above every padding and below every expert this
 * engine serves. */
constexpr int64_t kReadThrough = (int64_t)256 << 10;

}  /* namespace */

bool Mover::on_file_tier(const std::vector<int32_t>& units) const {
    for (int32_t u : units) {
        if (u < 0 || (size_t)u >= unit_.size()) continue;
        const UnitState& us = unit_[(size_t)u];
        if (us.tier == Tier::SSD && us.slab_slot < 0 && us.klass >= 0) return true;
    }
    return false;
}

bool Mover::routes_any(const std::vector<int32_t>& units, const int32_t* ecnt,
                       int32_t n_expert) const {
    if (!plan_ || !ecnt) return !units.empty();
    for (int32_t u : units) {
        if (u < 0 || (size_t)u >= plan_->units.size()) continue;
        const int32_t e = plan_->units[(size_t)u].expert;
        if (e < 0 || e >= n_expert || ecnt[e] != 0) return true;
    }
    return false;
}

void Mover::cover_file_units(const std::vector<int32_t>& more) {
    /* A unit whose container bytes are not its kernel's layout cannot be read back. */
    for (int32_t u : more)
        if (u >= 0 && (size_t)u < file_ok_.size() && plan_->units[(size_t)u].file_direct)
            file_ok_[(size_t)u] = 1;
}

/* A full pool's relabel victim for a deposit of class `klass`: its coldest pooled unit that the
 * file stager reads back (Mover::file_ok_), with nothing in flight, outside the `protect` layers,
 * and whose layer's file-tier part still fits the stager's buffer with it. -1 when none, or when
 * that unit is not colder than `below` (the unit being read): trading a pooled unit for a colder
 * one would only make the next step read the warmer one from the drive.
 *
 * The protected layers are the one being read and any whose ops are about to be issued: a
 * relabelled unit's weights are published at the zero region at once, so taking one of theirs
 * would hand the next GEMM zeros for a routed expert. Every other layer is safe: one already
 * issued this pass read the pool slot, which the quarantine keeps until those kernels are done,
 * and a later one is read back by its own trigger, since the stager covers the unit. The
 * routing counts protect nothing here: they are the counts of the layer being read, and indexing
 * them by another layer's expert number protected a random set while a whole-layer read (no
 * counts) protected nothing at all.
 *
 * The cap keeps every read of a layer within the stager's buffer: the plan sized it for the
 * largest layer's file-tier part, and a relabel adds to its victim's. */
int32_t Mover::pick_victim(int32_t klass, const int32_t* protect, int n_protect, float below) {
    if (klass < 0 || (size_t)klass >= victims_.size() || !heat_) return -1;
    VictimRank& vr = victims_[(size_t)klass];
    if (vr.at != dispatch_) {
        vr.at = dispatch_;
        vr.by_heat.clear();
        for (int32_t v = 0; v < (int32_t)unit_.size(); ++v) {
            const UnitState& vs = unit_[(size_t)v];
            if (vs.klass != klass || vs.tier != Tier::HostPinned || vs.pool_slot < 0 ||
                vs.slab_slot >= 0 || !file_ok_[(size_t)v])
                continue;
            vr.by_heat.push_back({ heat_->heat(v), v });
        }
        /* Hottest first; on equal heat the higher unit first, so the back is the coldest and,
         * among equally cold ones, the lowest: the unit a scan keeping its first strict minimum
         * would pick. */
        std::sort(vr.by_heat.begin(), vr.by_heat.end(),
                  [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                      return a.first != b.first ? a.first > b.first : a.second > b.second;
                  });
    }
    for (size_t k = vr.by_heat.size(); k-- > 0;) {
        const int32_t v = vr.by_heat[k].second;
        if (v < 0) continue;
        const UnitState& vs = unit_[(size_t)v];
        /* Within a dispatch a unit leaves the pool only by a relabel, so one that is not there any
         * more is gone for the rest of it. A copy in flight or a protected layer only waits. */
        if (vs.klass != klass || vs.tier != Tier::HostPinned || vs.pool_slot < 0 ||
            vs.slab_slot >= 0) {
            vr.by_heat[k].second = -1;
            continue;
        }
        if (copying_[(size_t)v]) continue;
        const int32_t layer = plan_->units[(size_t)v].layer;
        bool held = layer < 0 || (size_t)layer >= layer_file_.size();
        for (int p = 0; p < n_protect && !held; ++p) held = protect[p] >= 0 && protect[p] == layer;
        if (held) continue;
        if (layer_file_[(size_t)layer] + align_up(vs.bytes, RAD_ALIGN_UNIT) > file_cap_) continue;
        /* Every candidate further in is at least as warm. */
        if (!(vr.by_heat[k].first < below)) return -1;
        vr.by_heat[k].second = -1;
        while (!vr.by_heat.empty() && vr.by_heat.back().second < 0) vr.by_heat.pop_back();
        return v;
    }
    return -1;
}

int Mover::stage_file_units(const std::vector<int32_t>& units, char* dst, int64_t cap,
                            RadEvent after, RadEvent done, FileRead* rd, const int32_t* ecnt,
                            int32_t n_expert, const int32_t* protect, int n_protect) {
    if (!plan_ || !dst || !rd || !done) return RAD_E_INVAL;
    if (!file_.ok() || !windows_ || n_windows_ <= 0) return RAD_E_STATE;

    /* The layout in `dst`, unit by unit in the order given after what earlier calls for the
     * layer packed, and every weight's extent. */
    fext_.clear();
    fat_.clear();
    fbase_.clear();
    frec_.clear();
    const size_t staged0 = rd->staged.size(), skipped0 = rd->skipped.size(),
                 promoted0 = rd->promoted.size();
    const int64_t used0 = rd->used;
    int64_t off = rd->used, hot_bytes = 0;
    for (int32_t u : units) {
        if (u < 0 || (size_t)u >= unit_.size()) return RAD_E_INVAL;
        UnitState& us = unit_[(size_t)u];
        if (us.tier != Tier::SSD || us.slab_slot >= 0 || us.klass < 0) continue;
        const MoveUnit& mu = plan_->units[(size_t)u];
        /* Routing-aware reads: `ecnt` holds the counts of the rows about to run, taken at the
         * layer's trigger behind its router. An expert counted zero has no sorted rows, so no
         * GEMM run names it and nothing reads its weights: leave its bytes on the drive. The
         * dummy publication below keeps every table entry valid for the pass. */
        if (ecnt && mu.expert >= 0 && mu.expert < n_expert && ecnt[mu.expert] == 0) {
            rd->skipped.push_back(u);
            continue;
        }
        /* The hot VRAM cache: heat past the admission bar and a free
         * budgeted slot, and this pass's read lands in the slot and stays there. The read is
         * one the stager makes anyway; the slot is the whole cost, and eviction is
         * bookkeeping (the container is ground truth), so a mistake costs one future
         * re-read. No free slot, no heat, or no move record: the buffer path below, as
         * before. */
        if (plan_->dynamic && heat_ && heat_->admit_hot(u) && !unit_in_flight(u)) {
            ClassPools& cp = pools_[(size_t)us.klass];
            const int rec = free_move_record();
            int64_t slot = -1;
            if (rec < 0) ++stats_.no_record;
            else slot = cp.slab.take_where([&](int64_t s) { return s < cp.n_fixed; });
            if (slot >= 0 && record_gate() < 0) { cp.slab.add_free(slot); slot = -1; }
            if (slot >= 0) {
                char* const base = cp.slot_ptr(slot);
                Move& m = moves_[(size_t)rec];
                m.state = Move::State::Copying;
                m.kind = Move::Kind::Admit;
                m.unit = u;
                m.slab_slot = slot;
                m.pool_slot = -1;
                ++copying_[(size_t)u];
                us.slab_slot = slot;
                us.hot = true;
                heat_->set_busy(u, true);
                ++stats_.ssd_promotions;
                for (size_t k = 0; k < mu.weights.size(); ++k) {
                    const WeightInfo& w = prog_->weights[mu.weights[k]];
                    if (w.stored_bytes <= 0) continue;  /* nothing to read, nothing to place */
                    fext_.push_back(FileExt{ (int64_t)w.file_offset, w.stored_bytes,
                                             base + mu.offset[k], true, false });
                }
                rd->promoted.push_back(u);
                fbase_.push_back(base);
                frec_.push_back((int32_t)rec);
                hot_bytes += us.bytes;
                continue;
            }
        }
        /* A free pool slot and this read deposits the unit into the pool: its window's copy list
         * carries it into the slot, and the unit becomes a pool unit -- the GEMM reads it
         * zero-copy this step, heat promotes it at PCIe speed, and its demotion is a copyless
         * drop. The pool is all cache: with no free slot, its coldest unit the stager covers is
         * relabelled back to the file tier, copyless (the container holds every unit's bytes),
         * provided it is colder than this one (pick_victim). The slot it gives up feeds a later
         * deposit after the quarantine; this read takes the buffer path. */
        if (plan_->dynamic && heat_ && !unit_in_flight(u)) {
            ClassPools& cp = pools_[(size_t)us.klass];
            const int rec = free_move_record();
            int64_t g = -1;
            if (rec < 0) ++stats_.no_record;
            else if (record_gate() >= 0) g = cp.pool.take();
            if (g < 0 && rec >= 0) {
                /* The deposit's own record is still free (nothing uses it this read), so the
                 * relabel may take the same one. */
                const int r2 = free_move_record();
                const int32_t victim =
                    r2 >= 0 ? pick_victim(us.klass, protect, n_protect, heat_->heat(u)) : -1;
                if (victim >= 0) {
                    UnitState& vs = unit_[(size_t)victim];
                    const MoveUnit& mv = plan_->units[(size_t)victim];
                    Move& m2 = moves_[(size_t)r2];
                    m2.state = Move::State::Published;  /* nothing to copy: the drive has it */
                    m2.kind = Move::Kind::Evict;
                    m2.unit = victim;
                    m2.slab_slot = -1;
                    m2.pool_slot = vs.pool_slot;
                    for (size_t k = 0; k < mv.weights.size(); ++k)
                        res_.publish(mv.weights[k], calm_);
                    vs.tier = Tier::SSD;
                    vs.site = Site::DeviceStaged;
                    vs.pool_slot = -1;
                    layer_file_[(size_t)mv.layer] += align_up(vs.bytes, RAD_ALIGN_UNIT);
                    heat_->set_tier(victim, Tier::SSD);
                    /* No pool slot to promote from until a read deposits it again, which makes
                     * it movable once more; its layer's trigger reads it meanwhile. */
                    heat_->set_movable(victim, false);
                    ++stats_.pool_evictions;
                }
            }
            if (g >= 0) {
                Move& m = moves_[(size_t)rec];
                m.state = Move::State::Copying;
                m.kind = Move::Kind::Deposit;
                m.unit = u;
                m.slab_slot = -1;
                m.pool_slot = g;
                ++copying_[(size_t)u];
                us.pool_slot = g;
                us.tier = Tier::HostPinned;
                us.site = Site::DeviceZeroCopy;
                if (mu.layer >= 0 && (size_t)mu.layer < layer_file_.size())
                    layer_file_[(size_t)mu.layer] -= align_up(us.bytes, RAD_ALIGN_UNIT);
                heat_->set_tier(u, Tier::HostPinned);
                heat_->set_movable(u, true);   /* the slot makes it a PCIe-speed promotion candidate */
                heat_->set_busy(u, true);
                ++stats_.ssd_deposits;
                char* const slot = (char*)cp.pool_base + g * cp.stride;
                for (size_t k = 0; k < mu.weights.size(); ++k) {
                    const WeightInfo& w = prog_->weights[mu.weights[k]];
                    if (w.stored_bytes <= 0) continue;   /* nothing to read, nothing to place */
                    fext_.push_back(FileExt{ (int64_t)w.file_offset, w.stored_bytes,
                                             slot + mu.offset[k], false, true });
                }
                for (size_t k = 0; k < mu.weights.size(); ++k)
                    res_.publish(mu.weights[k],
                                 (char*)cp.pool_dev + g * cp.stride + mu.offset[k]);
                frec_.push_back((int32_t)rec);
                continue;
            }
        }
        const int64_t need = align_up(us.bytes, RAD_ALIGN_UNIT);
        if (off + need > cap) {
            RAD_ERR("file stager: a routed layer's file-tier units need more than the %s buffer "
                    "the plan sized for the largest one", humanb(cap).c_str());
            return RAD_E_SCRATCH;
        }
        for (size_t k = 0; k < mu.weights.size(); ++k) {
            const WeightInfo& w = prog_->weights[mu.weights[k]];
            if (w.stored_bytes <= 0) continue;          /* nothing to read, and nothing to place */
            fext_.push_back(FileExt{ (int64_t)w.file_offset, w.stored_bytes,
                                     dst + off + mu.offset[k], false, false });
        }
        rd->staged.push_back(u);
        fat_.push_back(off);
        off += need;
    }
    rd->used = off;

    /* Publish before the reads: nothing resolves these addresses until the caller's layer op,
     * which waits on `done` first. Skipped units' entries must not name stale addresses either:
     * they point at the zero region, which no zero-count expert reads through (a dead row reads
     * zeros and discards them). Published even when EVERY unit of the layer was skipped -- a
     * decode step can route nothing to a layer -- because the kernels still issue the layer's
     * table and a null entry fails the step. The caller nulls them again with the staged units.
     * Hot-cache units publish at their slots and stay published: nothing unstages them, and
     * retire() finishes their admission when the copies land. */
    for (size_t k = skipped0; k < rd->skipped.size(); ++k) {
        const int32_t u = rd->skipped[k];
        const MoveUnit& mu = plan_->units[(size_t)u];
        for (size_t j = 0; j < mu.weights.size(); ++j) res_.publish(mu.weights[j], calm_);
        stats_.ssd_skipped_bytes += (uint64_t)unit_[(size_t)u].bytes;
    }
    stats_.ssd_skipped_units += (uint64_t)(rd->skipped.size() - skipped0);
    for (size_t k = staged0; k < rd->staged.size(); ++k) {
        const int32_t u = rd->staged[k];
        const MoveUnit& mu = plan_->units[(size_t)u];
        for (size_t j = 0; j < mu.weights.size(); ++j)
            res_.publish(mu.weights[j], dst + fat_[k - staged0] + mu.offset[j]);
        stats_.ssd_read_bytes += (uint64_t)unit_[(size_t)u].bytes;
    }
    for (size_t k = promoted0; k < rd->promoted.size(); ++k) {
        const int32_t u = rd->promoted[k];
        const MoveUnit& mu = plan_->units[(size_t)u];
        for (size_t j = 0; j < mu.weights.size(); ++j)
            res_.publish(mu.weights[j], fbase_[k - promoted0] + mu.offset[j]);
        stats_.ssd_read_bytes += (uint64_t)unit_[(size_t)u].bytes;
    }
    if (fext_.empty()) return RAD_OK;
    /* Container order, so a window is one long run of the drive rather than a seek per weight. */
    std::stable_sort(fext_.begin(), fext_.end(),
                     [](const FileExt& a, const FileExt& b) { return a.foff < b.foff; });

    /* The dashboard's SSD panel and /stats ssd_reads read these: the file tier's container
     * reads are the expert SSD traffic the metric names. The scheduled prefetch path counted
     * its own link reads here; the file stager's were invisible, so the panels showed zero
     * however much the drive streamed. */
    stats_.ssd_reads += (uint64_t)fext_.size();
    for (const FileExt& e : fext_) {
        stats_.ssd_bytes += (uint64_t)e.bytes;
        /* The copies below move the same bytes from the windows to their destination, which
         * crosses the link either way: into the layer buffer when the unit is discarded at the
         * next layer (ssd_staging_bytes), into a hot-cache or host-pool slot when it stays
         * (ssd_promoted_bytes and ssd_deposit_bytes, the kept-copy half of the split). */
        if (e.hot) stats_.ssd_promoted_bytes += (uint64_t)e.bytes;
        else if (e.deposit) stats_.ssd_deposit_bytes += (uint64_t)e.bytes;
        else       stats_.ssd_staging_bytes += (uint64_t)e.bytes;
    }

    if (after) RAD_TRY(rad_event_wait(mover_, after));

    const int64_t A = DirectFile::alignment();
    const int64_t W = RAD_PLACE_FILE_WINDOW_BYTES;
    struct Seg { int64_t fs = 0, fe = 0, ws = 0; };     /* file [fs, fe) read to window offset ws */
    struct Cp  { int64_t wo = 0, bytes = 0; char* dst = nullptr; };
    std::vector<Seg> segs;
    std::vector<Cp>  cps;
    size_t  i = 0;
    int64_t piece = 0;                                  /* bytes of fext_[i] already placed */
    while (i < fext_.size()) {
        const int w = win_next_;
        win_next_ = (win_next_ + 1) % n_windows_;
        char* win = windows_ + (int64_t)w * W;
        if (win_rec_[(size_t)w]) {
            /* The window's copies from the round before must have left it. */
            const auto t0 = std::chrono::steady_clock::now();
            RAD_TRY(rad_event_sync(win_ev_[(size_t)w]));
            stats_.ssd_wait_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - t0).count();
            win_rec_[(size_t)w] = 0;
        }

        /* Pack the window: extend the current run while the next weight starts within the
         * read-through gap of its end, start another run otherwise, and split a weight the window
         * cannot finish across this window and the next. */
        segs.clear();
        cps.clear();
        while (i < fext_.size()) {
            const FileExt& e = fext_[i];
            const int64_t fo = e.foff + piece, left = e.bytes - piece;
            int64_t wo;
            if (!segs.empty() && fo >= segs.back().fs && fo <= segs.back().fe + kReadThrough) {
                wo = segs.back().ws + (fo - segs.back().fs);
            } else {
                const int64_t ws = segs.empty()
                    ? 0 : segs.back().ws + align_up(segs.back().fe - segs.back().fs, A);
                const int64_t fs = fo / A * A;
                wo = ws + (fo - fs);
                if (wo >= W) break;
                segs.push_back(Seg{ fs, fs, ws });
            }
            const int64_t take = std::min(left, W - wo);
            if (take <= 0) break;
            Seg& s = segs.back();
            s.fe = std::max(s.fe, fo + take);
            cps.push_back(Cp{ wo, take, e.dst + piece });
            piece += take;
            if (piece == e.bytes) { ++i; piece = 0; }
            if (take < left) break;                     /* the window is full */
        }
        if (cps.empty()) return RAD_E_STATE;            /* a window that takes nothing cannot end */

        const auto t0 = std::chrono::steady_clock::now();
        for (const Seg& s : segs) {
            RAD_TRY(file_.read((uint64_t)s.fs, s.fe - s.fs, win + s.ws, W - s.ws));
            ++stats_.ssd_preads;
        }
        stats_.ssd_read_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - t0).count();
        for (const Cp& c : cps) {
            dev_page_invalidate(c.dst, c.bytes);
            RAD_TRY(rad_memcpy_async(c.dst, win + c.wo, c.bytes, mover_));
        }
        RAD_TRY(rad_event_record(win_ev_[(size_t)w], mover_));
        win_rec_[(size_t)w] = 1;
    }
    RAD_TRY(rad_event_record(done, mover_));
    /* The hot-cache admissions and deposits are copies like the staged units': their records
     * land behind them, so retire() finishes a move when its bytes are in its slot. */
    for (int32_t rec : frec_) RAD_TRY(rad_event_record(moves_[(size_t)rec].ev, mover_));
    rd->copied = true;

    ++stats_.ssd_layers;
    stats_.ssd_units += (uint64_t)((rd->staged.size() - staged0) + (rd->promoted.size() - promoted0));
    stats_.h2d_bytes += (uint64_t)(off - used0 + hot_bytes);
    return RAD_OK;
}

void Mover::release_slot(int32_t unit) {
    const int rec = free_move_record();
    if (rec < 0) { ++stats_.no_record; return; }
    UnitState& us = unit_[(size_t)unit];
    Move& m = moves_[(size_t)rec];
    m.state = Move::State::Published;    /* nothing to copy; it is already "landed" */
    m.kind = Move::Kind::Release;
    m.unit = unit;
    m.slab_slot = us.slab_slot;
    m.pool_slot = -1;
    us.slab_slot = -1;
    republish(unit);                     /* the pointer stops naming the slot before the barrier */
}

int Mover::issue(const Swap& s, HeatEngine* heat) {
    if (!plan_ || !plan_->dynamic) return RAD_OK;

    auto one = [&](int32_t unit, bool promote) -> bool {
        if (unit < 0) return false;
        UnitState& us = unit_[(size_t)unit];
        if (us.klass < 0) return false;
        ClassPools& cp = pools_[(size_t)us.klass];

        /* A unit that kept its host copy is dropped, not demoted: its pool slot still holds its
         * bytes, so there is nothing to copy and no second pool slot to take. */
        if (!promote && us.pool_slot >= 0 && us.slab_slot >= 0) return drop(unit, heat) >= 0;

        /* A hot-cache unit has no pool slot to demote into and needs none:
         * the container holds its bytes, so the eviction is bookkeeping — free the slot,
         * publish null, and the file stager reads the expert again if routing asks for it.
         * The quarantine below is the same as any demotion's: kernels issued in earlier
         * layers may still name the slot. */
        if (!promote && us.hot && us.slab_slot >= 0) {
            const int rec = free_move_record();
            if (rec < 0) { ++stats_.no_record; return false; }
            if (record_gate() < 0) return false;
            Move& m = moves_[(size_t)rec];
            m.state = Move::State::Published;    /* nothing to copy: the drive still has it */
            m.kind = Move::Kind::Evict;
            m.unit = unit;
            m.slab_slot = us.slab_slot;
            m.pool_slot = -1;
            us.slab_slot = -1;
            us.tier = Tier::SSD;
            us.site = Site::DeviceStaged;
            us.hot = false;
            republish(unit);                     /* the pointer stops naming the slot first */
            ++stats_.demotions;
            ++stats_.dropped;                    /* a copyless demotion, like a lent unit's drop */
            if (heat) {
                heat->set_tier(unit, Tier::SSD);
                heat->set_busy(unit, true);
            }
            return true;
        }

        if (promote) {
            const int64_t slot = cp.slab.take();
            if (slot < 0) { ++stats_.starved_slab; return false; }
            if (promote_to(unit, slot) < 0) { cp.slab.add_free(slot); return false; }
            return true;
        }

        /* Every refusal comes before pool_slot_for, which may take a donor's host copy
         * (donor_slot) that a failed demotion could not give back. */
        if (us.slab_slot < 0) return false;
        const int rec = free_move_record();
        if (rec < 0) { ++stats_.no_record; return false; }
        if (record_gate() < 0) return false;
        const int64_t g = pool_slot_for(unit, heat);
        if (g < 0) { ++stats_.starved_pool; return false; }
        /* D2H out of the slab; the slab layout is the pool layout, so a straight copy. It is not
         * free even though zero-copy reads use the other direction: both directions share one
         * budget and a demotion costs about what a promotion does, the exchange rate heat.h's
         * min_gain is tuned against. */
        if (rad_event_wait(mover_, gate_) < 0) { cp.pool.add_free(g); return false; }
        queue_move_copy((char*)cp.pool_base + g * cp.stride, cp.slot_ptr(us.slab_slot), us.bytes,
                        rec);
        Move& m = moves_[(size_t)rec];
        m.kind = Move::Kind::Demote;
        m.slab_slot = us.slab_slot;
        m.pool_slot = g;
        stats_.d2h_bytes += (uint64_t)us.bytes;
        m.state = Move::State::Copying;
        m.unit = unit;
        ++copying_[(size_t)unit];
        return true;
    };

    const bool up = one(s.promote, true);
    const bool down = one(s.demote, false);
    /* A half-issued swap is legal: the halves supply each other, so the missing one goes out on
     * a later dispatch. A unit no move will publish must not stay busy, or the heat engine would
     * never consider it again, with no counter showing why. */
    if (heat) {
        if (!up) heat->set_busy(s.promote, false);
        if (!down) heat->set_busy(s.demote, false);
    }
    return (up || down) ? RAD_OK : RAD_E_FULL;
}

/* ------------------------------------------------------------------ the loan */
static const char* lender_name(int lender) {
    return lender == Mover::kLendKV ? "KV cache" : lender == Mover::kLendArena ? "activation arena"
                                                                             : "fleet's free VRAM";
}

static char* align_down_ptr(char* p) {
    const uintptr_t v = (uintptr_t)p;
    return (char*)(v - v % (uintptr_t)RAD_ALIGN_UNIT);
}

int Mover::lend_from(int lender, const std::vector<LoanStrip>& strips) {
    if (!plan_ || !plan_->dynamic) return RAD_OK;      /* a static plan never moves a unit */
    if (lender < 0 || lender >= kLenders) return RAD_E_INVAL;
    for (const ClassPools& cp : pools_)
        for (const ClassPools::Lent& l : cp.lent)
            if (l.lender == lender) {
                RAD_ERR("mover: lend_from called twice for one lender; its slots are carved once");
                return RAD_E_STATE;
            }

    /* Slots go to classes with promotable units in the pinned pool, in proportion to their
     * bytes, and never more slots than such units, since a slot no unit can fill is wasted. */
    const size_t nc = pools_.size();
    std::vector<int64_t> want(nc, 0), got(nc, 0);
    for (size_t u = 0; u < unit_.size(); ++u) {
        const UnitState& us = unit_[u];
        if (us.klass < 0 || us.tier != Tier::HostPinned || us.pool_slot < 0) continue;
        if (!plan_->units[u].movable) continue;
        ++want[(size_t)us.klass];
    }
    std::vector<double> weight(nc, 0.0);
    for (size_t c = 0; c < nc; ++c)
        weight[c] = pools_[c].stride > 0 ? (double)want[c] * (double)pools_[c].stride : 0.0;

    /* Every stretch, top down. Classes take turns in proportion to their weight, so each stretch
     * holds each class's share regardless of the order the stretches arrive in. */
    struct Cut { int64_t need; char* p; };
    std::vector<std::vector<Cut>> cuts(nc);
    for (const LoanStrip& st : strips) {
        if (!st.top || st.bytes <= (int64_t)RAD_ALIGN_UNIT || st.unit_bytes <= 0 ||
            st.unit_tokens <= 0)
            continue;
        char* const bottom = st.top - st.bytes;
        /* The top alignment unit stays empty. A kernel assembling a packed code window reads a
         * few bytes past its slot; past the stretch's top is memory that is not the slab's (the
         * next layer's blocks, or at the end of the carve possibly unmapped memory). */
        char* cur = align_down_ptr(st.top - RAD_ALIGN_UNIT);
        for (;;) {
            int best = -1;
            double best_k = 0.0;
            for (size_t c = 0; c < nc; ++c) {
                if (weight[c] <= 0.0 || got[c] >= want[c]) continue;
                if (cur - bottom < pools_[c].stride) continue;
                if (align_down_ptr(cur - pools_[c].stride) < bottom) continue;
                const double k = (double)(got[c] + 1) * (double)pools_[c].stride / weight[c];
                if (best < 0 || k < best_k) { best = (int)c; best_k = k; }
            }
            if (best < 0) break;
            char* p = align_down_ptr(cur - pools_[(size_t)best].stride);
            /* The loan that covers this slot reaches its lowest byte. */
            const int64_t depth = st.top - p;
            const int64_t need = (depth + st.unit_bytes - 1) / st.unit_bytes * st.unit_tokens;
            cuts[(size_t)best].push_back(Cut{ need, p });
            ++got[(size_t)best];
            cur = p;
        }
    }

    /* Smallest loan first, so any loan covers a prefix of the slots and the deepest slot, which
     * the owner wants back first, is given back first. */
    int64_t n_new = 0, bytes_new = 0;
    for (size_t c = 0; c < nc; ++c) {
        if (cuts[c].empty()) continue;
        ClassPools& cp = pools_[c];
        std::stable_sort(cuts[c].begin(), cuts[c].end(),
                         [](const Cut& x, const Cut& y) { return x.need < y.need; });
        ClassPools::Lent l;
        l.lender = lender;
        l.seg = cp.slab.add_segment((int64_t)cuts[c].size());
        l.first = cp.slab.seg_lo(l.seg);
        /* Segments are appended in the same order as their pointers, so slot_ptr's
         * `slot - n_fixed` indexes flex_ptr directly. */
        if (l.first != cp.n_fixed + (int64_t)cp.flex_ptr.size()) {
            RAD_ERR("mover: a lent segment starts at slot %lld with %zu lent slots before it",
                    (long long)l.first, cp.flex_ptr.size());
            return RAD_E_STATE;
        }
        for (const Cut& k : cuts[c]) { cp.flex_ptr.push_back(k.p); l.need.push_back(k.need); }
        cp.lent.push_back(std::move(l));
        n_new += (int64_t)cuts[c].size();
        bytes_new += (int64_t)cuts[c].size() * cp.stride;
    }
    if (n_new <= 0) return RAD_OK;

    /* A record for every lendable slot. A fill promotes into every free slot at once and a recall
     * drops every unit past the loan at once, each holding a record until its barrier clears;
     * with only the shadow count, records rather than the link would limit how fast a loan is
     * used. */
    const size_t had = moves_.size();
    moves_.resize(had + (size_t)n_new);
    for (size_t i = had; i < moves_.size(); ++i) RAD_TRY(rad_event_create_local(&moves_[i].ev));

    RAD_INFO("mover: %lld slots can be lent out of the %s (%s), filled as it lends them",
             (long long)n_new, lender_name(lender),
             humanb(bytes_new).c_str());
    count_flex();
    return RAD_OK;
}

int64_t Mover::lendable_bytes() const {
    if (!plan_ || !plan_->dynamic) return 0;
    std::vector<int64_t> n(pools_.size(), 0);
    for (size_t u = 0; u < unit_.size(); ++u) {
        const UnitState& us = unit_[u];
        if (us.klass < 0 || us.tier != Tier::HostPinned || us.pool_slot < 0) continue;
        if (!plan_->units[u].movable) continue;
        ++n[(size_t)us.klass];
    }
    int64_t b = 0;
    for (size_t c = 0; c < pools_.size(); ++c)
        if (pools_[c].stride > 0) b += n[c] * (pools_[c].stride + (int64_t)RAD_ALIGN_UNIT);
    return b > 0 ? b + 2 * (int64_t)RAD_ALIGN_UNIT : 0;
}

int Mover::set_loan(int lender, int64_t amount) {
    if (lender < 0 || lender >= kLenders) return RAD_E_INVAL;
    loan_[lender] = amount > 0 ? amount : 0;
    for (ClassPools& cp : pools_)
        for (const ClassPools::Lent& l : cp.lent) {
            if (l.lender != lender) continue;
            const int64_t n = std::upper_bound(l.need.begin(), l.need.end(), loan_[lender]) -
                              l.need.begin();
            if (cp.slab.target(l.seg) != l.first + n) {
                if (l.first + n < cp.slab.target(l.seg)) cp.strand_check = true;
                cp.slab.set_target(l.seg, l.first + n);
            }
        }
    count_flex();
    return RAD_OK;
}

int64_t Mover::loan_held(int lender) const {
    int64_t t = 0;
    for (const ClassPools& cp : pools_)
        for (const ClassPools::Lent& l : cp.lent) {
            if (l.lender != lender) continue;
            const int64_t top = cp.slab.top_held(l.seg);
            if (top >= l.first) t = std::max(t, l.need[(size_t)(top - l.first)]);
        }
    return t;
}

int64_t Mover::flex_capacity() const {
    int64_t n = 0;
    for (const ClassPools& cp : pools_) n += (int64_t)cp.flex_ptr.size() * cp.stride;
    return n;
}

int64_t Mover::loan_ceiling(int lender) const {
    int64_t t = 0;
    for (const ClassPools& cp : pools_)
        for (const ClassPools::Lent& l : cp.lent)
            if (l.lender == lender && !l.need.empty()) t = std::max(t, l.need.back());
    return t;
}

void Mover::count_flex() {
    int64_t used = 0, offered = 0;
    for (const ClassPools& cp : pools_) {
        if (cp.flex_ptr.empty()) continue;
        used += cp.slab.held_from(cp.n_fixed) * cp.stride;
        for (const ClassPools::Lent& l : cp.lent)
            offered += (cp.slab.target(l.seg) - l.first) * cp.stride;
    }
    flex_used_.store(used, std::memory_order_relaxed);
    flex_offered_.store(offered, std::memory_order_relaxed);
}

int Mover::drop(int32_t unit, HeatEngine* heat) {
    const int rec = free_move_record();
    if (rec < 0) { ++stats_.no_record; return RAD_E_FULL; }
    UnitState& us = unit_[(size_t)unit];
    Move& m = moves_[(size_t)rec];
    m.state = Move::State::Published;       /* nothing to copy: the host copy never went away */
    m.kind = Move::Kind::Drop;
    m.unit = unit;
    m.slab_slot = us.slab_slot;
    m.pool_slot = -1;
    us.slab_slot = -1;
    us.tier = Tier::HostPinned;
    us.site = Site::DeviceZeroCopy;
    republish(unit);                        /* the pointer stops naming the slot before the barrier */
    ++stats_.demotions;
    ++stats_.dropped;
    if (heat) {
        heat->set_tier(unit, Tier::HostPinned);
        heat->set_busy(unit, true);
    }
    return RAD_OK;
}

/* The pool is sized to its units, so it has a free slot only while a promotion passes through
 * quarantine. Without donors a cold budgeted unit stays and the engine's swap drops a warmer lent
 * unit instead; replaying mixed routing at TP1, the hit rate is 72.7% that way, 76.7% with up to
 * n_turn donors per class, and 78.5% for one resident set with no regions.
 *
 * A donor loses its cheap recall: a recall or shrink must move it into a free budgeted slot or
 * back across the link (reclaim, balance). There is always room once quarantine has emptied,
 * since each donation left a budgeted slot free. Donors are capped at the turnover slots and come
 * only from the KV cache's loan, so the arena's per-prompt recall stays an address change.
 *
 * The hottest unit (of the victim's layer, when asked) donates, since a donor later demoted needs
 * a pool slot of its own. Its pool slot has no reader: the unit is resident and the copy into the
 * slot waits on the gate. */
int64_t Mover::donor_slot(size_t klass, int32_t layer, HeatEngine* heat) {
    ClassPools& cp = pools_[klass];
    if (cp.n_donated >= cp.n_turn) return -1;
    /* Not while the turnover slots are held (around a prompt): a donation releases a starved
     * swap, two copies on the stream the next prompt stages on. A decode run reaches the donor
     * count within a few steps anyway. */
    if (hold_turnover_) return -1;
    DonorRank& d = donors_[klass];
    if (d.at != dispatch_) {
        d.at = dispatch_;
        d.by_layer.assign((size_t)std::max(n_layers_, 1), {});
        for (const ClassPools::Lent& l : cp.lent) {
            if (l.lender != kLendKV) continue;
            const int64_t lo = cp.slab.seg_lo(l.seg), hi = cp.slab.target(l.seg);
            for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
                const UnitState& us = unit_[(size_t)u];
                if (us.klass != (int32_t)klass || us.slab_slot < lo || us.slab_slot >= hi) continue;
                const int32_t ul = plan_->units[(size_t)u].layer;
                if (ul < 0 || ul >= n_layers_) continue;
                d.by_layer[(size_t)ul].push_back({ heat ? heat->heat(u) : 0.0f, u });
            }
        }
        for (auto& v : d.by_layer) std::sort(v.begin(), v.end());
    }
    for (;;) {
        /* The hottest candidate of the layer asked for, or of every layer. */
        std::vector<std::pair<float, int32_t>>* pick = nullptr;
        if (layer >= 0) {
            if (layer < (int32_t)d.by_layer.size() && !d.by_layer[(size_t)layer].empty())
                pick = &d.by_layer[(size_t)layer];
        } else {
            for (auto& v : d.by_layer)
                if (!v.empty() && (!pick || v.back() > pick->back())) pick = &v;
        }
        if (!pick) return -1;
        const int32_t u = pick->back().second;
        pick->pop_back();
        UnitState& us = unit_[(size_t)u];
        if (us.klass != (int32_t)klass || us.pool_slot < 0 || us.slab_slot < 0) continue;
        if (us.tier != Tier::VRAM || us.site != Site::Device || !cp.slab.covered(us.slab_slot))
            continue;
        if ((heat && heat->busy(u)) || unit_in_flight(u)) continue;
        const int64_t g = us.pool_slot;
        us.pool_slot = -1;
        ++cp.n_donated;
        ++stats_.donated;
        return g;
    }
}

/* The prefill stager copies runs of neighbouring pool slots, and the pool starts in unit order,
 * so a demotion into another layer's slot costs an extra copy in both layers on every later
 * prompt. Prefer a slot the unit's own layer started in, then a same-layer donor, then any free
 * slot, then any donor. At TP1 (26K-token prompts, 256 decoded tokens between) this takes a
 * layer's staging from 162 copies to 112. */
int64_t Mover::pool_slot_for(int32_t unit, HeatEngine* heat) {
    const UnitState& us = unit_[(size_t)unit];
    ClassPools& cp = pools_[(size_t)us.klass];
    const int32_t layer = plan_->units[(size_t)unit].layer;
    const std::vector<int32_t>& home = pool_layer_[(size_t)us.klass];
    int64_t g = -1;
    if (layer >= 0) {
        g = cp.pool.take_where([&](int64_t p) { return (size_t)p < home.size() && home[(size_t)p] == layer; });
        if (g < 0) g = donor_slot((size_t)us.klass, layer, heat);
    }
    if (g < 0) g = cp.pool.take();
    if (g < 0) g = donor_slot((size_t)us.klass, -1, heat);
    return g;
}

bool Mover::migrate(int32_t unit, HeatEngine* heat) {
    UnitState& us = unit_[(size_t)unit];
    ClassPools& cp = pools_[(size_t)us.klass];
    const int64_t to = cp.slab.take();
    if (to < 0) return false;
    if (to >= cp.n_fixed) {
        cp.slab.add_free(to);
        return false;
    }
    const int rec = free_move_record();
    if (rec < 0 || record_gate() < 0) {
        if (rec < 0) ++stats_.no_record;
        cp.slab.add_free(to);
        return false;
    }
    if (rad_event_wait(mover_, gate_) < 0) {
        cp.slab.add_free(to);
        return false;
    }
    queue_move_copy(cp.slot_ptr(to), cp.slot_ptr(us.slab_slot), us.bytes, rec);
    Move& m = moves_[(size_t)rec];
    m.state = Move::State::Copying;
    m.kind = Move::Kind::Migrate;
    m.unit = unit;
    m.slab_slot = us.slab_slot;
    m.pool_slot = -1;
    m.to_slot = to;
    ++copying_[(size_t)unit];
    if (heat) heat->set_busy(unit, true);
    return true;
}

int Mover::promote_to(int32_t unit, int64_t slot) {
    UnitState& us = unit_[(size_t)unit];
    ClassPools& cp = pools_[(size_t)us.klass];
    const int rec = free_move_record();
    if (rec < 0) { ++stats_.no_record; return RAD_E_FULL; }
    RAD_TRY(record_gate());
    if (us.pool_slot < 0) return RAD_E_STATE;
    const void* src = (const char*)cp.pool_base + us.pool_slot * cp.stride;
    Move& m = moves_[(size_t)rec];
    RAD_TRY(rad_event_wait(mover_, gate_));
    queue_move_copy(cp.slot_ptr(slot), src, us.bytes, rec);
    m.kind = Move::Kind::Promote;
    m.slab_slot = slot;
    /* Into a lent slot the host copy stays: the loan can be recalled at any time, and a unit
     * that kept its pool slot goes back by a change of address instead of needing a pool slot and
     * a link copy just as the cache wants its memory. The quarantine then has no pool slot to
     * free. Every pool unit keeps its slot on promotion: the pool is one cache, so the later demotion is a copyless drop for natives and deposits alike. */
    m.pool_slot = cp.is_flex(slot) ? -1 : us.pool_slot;
    m.state = Move::State::Copying;
    m.unit = unit;
    ++copying_[(size_t)unit];
    stats_.h2d_bytes += (uint64_t)us.bytes;
    return RAD_OK;
}

int Mover::fill(size_t klass, int64_t room, HeatEngine* heat) {
    if (room <= 0) return RAD_OK;
    struct Cand { float h; int32_t layer; int32_t unit; };
    std::vector<Cand> cand;
    for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
        const UnitState& us = unit_[(size_t)u];
        if (us.klass != (int32_t)klass || us.tier != Tier::HostPinned || us.pool_slot < 0) continue;
        if (!plan_->units[(size_t)u].movable) continue;
        if (heat && heat->busy(u)) continue;
        cand.push_back(Cand{ heat ? heat->heat(u) : 0.0f, plan_->units[(size_t)u].layer, u });
    }
    std::stable_sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.h > b.h; });

    /* Hottest first; on an exact tie, the layer with the fewest resident units. Every promotion
     * here is issued before any retires, so the heat engine's resident count does not move during
     * the batch and `added` tracks it instead; otherwise a fill of a cold slab (all heats zero)
     * would put the whole batch into whichever layer was thinnest at the start. */
    std::unordered_map<int32_t, int32_t> added;
    int64_t issued = 0;
    for (size_t i = 0; i < cand.size() && issued < room;) {
        size_t j = i;
        while (j < cand.size() && cand[j].h == cand[i].h) ++j;
        std::map<int32_t, std::vector<int32_t>> by_layer;            /* one heat, per layer */
        for (size_t k = j; k-- > i;) by_layer[cand[k].layer].push_back(cand[k].unit);
        using Key = std::pair<int32_t, int32_t>;                      /* (residents, layer) */
        std::priority_queue<Key, std::vector<Key>, std::greater<Key>> q;
        for (auto& kv : by_layer) {
            const int32_t rc = heat ? heat->layer_residents(kv.second.back()) : 0;
            q.push({ rc + added[kv.first], kv.first });
        }
        while (!q.empty() && issued < room) {
            const Key k = q.top();
            q.pop();
            std::vector<int32_t>& us = by_layer[k.second];
            const int32_t u = us.back();
            us.pop_back();
            /* A lent slot, never a working one. Shadow and staging slots are what let a swap
             * happen, and a unit filled into one sits in the budgeted range where no loan shrink
             * returns it, so the reserve would be lost. */
            const int64_t slot = pools_[klass].slab.take_at_least(pools_[klass].n_fixed);
            if (slot < 0) return RAD_OK;
            if (heat) heat->set_busy(u, true);
            if (promote_to(u, slot) < 0) {                   /* no record left: next time */
                pools_[klass].slab.add_free(slot);
                if (heat) heat->set_busy(u, false);
                return RAD_OK;
            }
            ++issued;
            ++added[k.second];
            if (!us.empty()) q.push({ k.first + 1, k.second });
        }
        i = j;
    }
    return RAD_OK;
}

int Mover::balance(HeatEngine* heat, bool turnover) {
    if (!plan_ || !plan_->dynamic) return RAD_OK;

    for (size_t c = 0; c < pools_.size(); ++c) {
        ClassPools& cp = pools_[c];
        if (cp.flex_ptr.empty()) continue;
        bool drained = false;

        /* ---- Empty what the loan no longer covers, all of it, first: drops cost no link
         * traffic and the lender is waiting. A busy unit is still on its way in and goes on a
         * later call. The sweep walks every unit, so it runs only after a shrink
         * (ClassPools::strand_check). */
        if (cp.strand_check) {
            bool left = false;
            for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
                const UnitState& us = unit_[(size_t)u];
                if (us.klass != (int32_t)c || us.slab_slot < 0 || cp.slab.covered(us.slab_slot)) continue;
                /* Anything else past the loan (an unretired promotion, a staged unit) becomes
                 * droppable later, so the sweep must come back for it. */
                if (us.tier != Tier::VRAM || us.site != Site::Device || (heat && heat->busy(u))) {
                    left = true;
                    continue;
                }
                /* A donor has no host copy to drop to: it moves into a free budgeted slot, or
                 * else back across the link, and stays past the loan until the copy lands. */
                if (us.pool_slot < 0) {
                    if (migrate(u, heat)) {
                        drained = true;
                    } else {
                        if (heat) heat->set_busy(u, true);
                        Swap s;
                        s.demote = u;
                        if (issue(s, heat) >= 0) drained = true;
                    }
                    left = true;
                    continue;
                }
                if (drop(u, heat) < 0) { left = true; break; }
                drained = true;
            }
            cp.strand_check = left;
        }

        /* ---- The working reserve. Shadow and staging slots must stay free whatever the loan,
         * or the mover cannot turn over; after a shrink the coldest lent units are dropped to
         * restore them. A slot already on its way back (`pending`) counts as free, or the drain
         * would repeat and empty the slab.
         *
         * The turnover slots are rebuilt the same way. They are the top of the lent range, which
         * a recall takes first, and the loan regrows only a prefill chunk at a time, so the fill
         * alone would leave them gone. A swap whose demotion found no pool slot is completed the
         * same way. */
        int64_t pending = 0;
        for (const Move& m : moves_) {
            if (m.state == Move::State::Free || m.unit < 0 || m.slab_slot < 0) continue;
            if (unit_[(size_t)m.unit].klass != (int32_t)c || !cp.slab.covered(m.slab_slot)) continue;
            if (m.frees_slab()) ++pending;
        }
        const int64_t turn = turnover ? cp.n_turn : 0;
        int64_t short_by = cp.n_work + turn - (cp.slab.n_free() + pending);
        if (short_by > 0) {
            drop_cand_.clear();
            for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
                const UnitState& us = unit_[(size_t)u];
                if (us.klass != (int32_t)c || us.slab_slot < 0 || !cp.is_flex(us.slab_slot)) continue;
                if (us.tier != Tier::VRAM || us.site != Site::Device || us.pool_slot < 0) continue;
                if (heat && heat->busy(u)) continue;
                drop_cand_.push_back({ heat ? heat->heat(u) : 0.0f, u });
            }
            /* The coldest first, and on a tie the lower unit, so the choice is the same every run. */
            auto colder = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                return a.first != b.first ? a.first < b.first : a.second < b.second;
            };
            const size_t k = std::min(drop_cand_.size(), (size_t)short_by);
            std::partial_sort(drop_cand_.begin(), drop_cand_.begin() + (ptrdiff_t)k, drop_cand_.end(),
                              colder);
            for (size_t i = 0; i < k; ++i) {
                if (drop(drop_cand_[i].second, heat) < 0) break;
                drained = true;
                --short_by;
            }
        }
        /* With no lent unit left to drop, demote the coldest budgeted unit (a real copy, one per
         * call), for the working reserve only: link traffic to keep turnover slots would cost what
         * they save. A shrinking loan can take the free slots a swap left in the lent range, so
         * the reserve must be rebuildable from the budgeted slots. */
        if (short_by > turn) {
            int32_t victim = -1;
            float coldest = 0.0f;
            for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
                const UnitState& us = unit_[(size_t)u];
                if (us.klass != (int32_t)c || us.slab_slot < 0 || cp.is_flex(us.slab_slot)) continue;
                if (us.tier != Tier::VRAM || us.site != Site::Device || us.pool_slot >= 0) continue;
                if (!plan_->units[(size_t)u].movable && !unit_[(size_t)u].hot) continue;
                if (heat && heat->busy(u)) continue;
                const float h = heat ? heat->heat(u) : 0.0f;
                if (victim < 0 || h < coldest) { victim = u; coldest = h; }
            }
            if (victim >= 0) {
                if (heat) heat->set_busy(victim, true);
                Swap s;
                s.demote = victim;
                if (issue(s, heat) >= 0) drained = true;
            }
        }
        /* A drain and a fill in the same call fight: the fill would take the slots the drain is
         * trying to hand back. */
        if (drained) continue;

        /* ---- Then fill every free slot past the working reserve and the turnover slots. */
        RAD_TRY(fill(c, cp.slab.n_free() - cp.n_work - turn, heat));
    }
    count_flex();
    return flush_moves();
}

int Mover::idle_tick(HeatEngine* heat) {
    if (!plan_ || !plan_->dynamic) return RAD_OK;
    RAD_TRY(retire(heat));
    return balance(heat, !hold_turnover_);
}

int Mover::reclaim(int lender, int64_t amount, HeatEngine* heat) {
    RAD_TRY(set_loan(lender, amount));
    if (loan_held(lender) <= loan_[lender]) return RAD_OK;
    if (!compute_ || !mover_) {
        RAD_ERR("mover: a loan cannot be recalled with no compute stream bound; nothing orders "
                "the kernels that may still read the slots");
        return RAD_E_STATE;
    }
    /* No copy may still be writing a slot: drain the mover's stream, then retire to publish
     * promotions that landed in slots about to be taken back. */
    RAD_TRY(flush_moves());
    RAD_TRY(rad_stream_sync(mover_));
    RAD_TRY(retire(heat));
    /* Drain quarantine too: the slots owed to the donors below come back through it (see
     * donor_slot). Nothing goes on the compute stream from here to the end of the recall. */
    RAD_TRY(rad_stream_sync(compute_));
    RAD_TRY(retire(heat));

    /* Every unit past the loan goes back to its host copy. A donor has none, so it moves into a
     * free budgeted slot or else across the link into a free pool slot, published once the
     * copies land. A donor with nowhere to go stays and the recall finishes short by its slot
     * (a later recall or balance() moves it); aborting instead would leak the slots already
     * taken off their units. */
    std::vector<std::pair<size_t, int64_t>> freed;
    std::vector<int32_t> moved;
    int stuck = 0;
    for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
        UnitState& us = unit_[(size_t)u];
        if (us.klass < 0 || us.slab_slot < 0) continue;
        ClassPools& cp = pools_[(size_t)us.klass];
        if (!cp.is_flex(us.slab_slot) || cp.slab.covered(us.slab_slot)) continue;
        if (us.pool_slot < 0) {
            const int64_t from = us.slab_slot;
            int64_t to = cp.slab.take();
            if (to >= cp.n_fixed) {
                cp.slab.add_free(to);
                to = -1;
            }
            const int64_t g = to < 0 ? cp.pool.take() : -1;
            if (to >= 0) {
                dev_page_invalidate(cp.slot_ptr(to), us.bytes);
                RAD_TRY(rad_memcpy_async(cp.slot_ptr(to), cp.slot_ptr(from), us.bytes, mover_));
                us.slab_slot = to;
            } else if (g >= 0) {
                RAD_TRY(rad_memcpy_async((char*)cp.pool_base + g * cp.stride, cp.slot_ptr(from),
                                         us.bytes, mover_));
                us.pool_slot = g;
                us.slab_slot = -1;
                us.tier = Tier::HostPinned;
                us.site = Site::DeviceZeroCopy;
                stats_.d2h_bytes += (uint64_t)us.bytes;
                ++stats_.demotions;
                if (heat) {
                    heat->set_tier(u, Tier::HostPinned);
                    heat->set_busy(u, false);
                }
            } else {
                ++stuck;
                continue;
            }
            --cp.n_donated;
            ++stats_.relocated;
            freed.push_back({ (size_t)us.klass, from });
            moved.push_back(u);
            continue;
        }
        freed.push_back({ (size_t)us.klass, us.slab_slot });
        us.slab_slot = -1;
        us.tier = Tier::HostPinned;
        us.site = Site::DeviceZeroCopy;
        republish(u);
        ++stats_.demotions;
        ++stats_.dropped;
        if (heat) {
            heat->set_tier(u, Tier::HostPinned);
            heat->set_busy(u, false);
        }
    }

    if (!moved.empty()) {
        RAD_TRY(rad_stream_sync(mover_));
        for (int32_t u : moved) republish(u);
    }

    /* No kernel still names a freed slot: the compute stream is the slab's only reader (the
     * quarantine relies on the same fact), it drained above with nothing issued since, and work
     * issued from here reads the host copies. */
    for (const auto& f : freed) pools_[f.first].slab.add_free(f.second);

    /* Every record waiting on one of those slots is done: its barrier is behind the drain. A
     * promotion's record has nothing to free (its unit kept the pool slot); a drop's frees the
     * slot it left. */
    for (Move& m : moves_) {
        if (m.state == Move::State::Free || m.unit < 0 || m.slab_slot < 0) continue;
        const int32_t klass = unit_[(size_t)m.unit].klass;
        if (klass < 0) continue;
        ClassPools& cp = pools_[(size_t)klass];
        if (!cp.is_flex(m.slab_slot) || cp.slab.covered(m.slab_slot)) continue;
        if (m.frees_slab()) cp.slab.add_free(m.slab_slot);
        if (heat && m.kind != Move::Kind::Stage && m.kind != Move::Kind::Release)
            heat->set_busy(m.unit, false);
        if (m.state == Move::State::Copying) --copying_[(size_t)m.unit];
        m.clear();
    }
    count_flex();
    if (stuck > 0) {
        stats_.stranded += (uint64_t)stuck;
        RAD_ERR("mover: %d unit(s) that gave their host copy away found no free slot to move to; "
                "the %s loan stops at %lld instead of %lld until they move", stuck,
                lender_name(lender), (long long)loan_held(lender),
                (long long)loan_[lender]);
        return RAD_E_FULL;
    }
    if (loan_held(lender) > loan_[lender]) {
        RAD_ERR("mover: recalled the %s loan to %lld and a slot at %lld is still held",
                lender_name(lender), (long long)loan_[lender],
                (long long)loan_held(lender));
        return RAD_E_STATE;
    }
    return RAD_OK;
}

int Mover::drop_slab(HeatEngine* heat, std::vector<std::pair<const void*, int64_t>>* free_ranges) {
    if (pools_.empty()) return RAD_OK;
    if (!compute_ || !mover_) return RAD_OK;   /* nothing ever moved: the slab is as loaded */
    RAD_TRY(flush_moves());
    RAD_TRY(rad_stream_sync(mover_));
    RAD_TRY(retire(heat));
    RAD_TRY(rad_stream_sync(compute_));
    RAD_TRY(retire(heat));
    std::vector<std::pair<size_t, int64_t>> freed;
    for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
        UnitState& us = unit_[(size_t)u];
        if (us.klass < 0 || us.slab_slot < 0) continue;
        ClassPools& cp = pools_[(size_t)us.klass];
        if (cp.is_flex(us.slab_slot)) continue;   /* the loans were recalled; nothing is here */
        if (us.pool_slot >= 0) {
            us.tier = Tier::HostPinned;
            us.site = Site::DeviceZeroCopy;
        } else if (us.hot || ((size_t)u < file_ok_.size() && file_ok_[(size_t)u])) {
            /* The container is its copy, and the file stager reads it back when routing needs it
             * (a hot-cache unit, or one the stager covers). */
            us.tier = Tier::SSD;
            us.hot = false;
        } else {
            continue;   /* its only copy is this slot */
        }
        freed.push_back({ (size_t)us.klass, us.slab_slot });
        us.slab_slot = -1;
        republish(u);
        ++stats_.dropped;
        if (heat) {
            heat->set_tier(u, us.tier);
            heat->set_busy(u, false);
        }
    }
    for (const auto& f : freed) pools_[f.first].slab.add_free(f.second);
    {
        int64_t kept = 0, gone = 0;
        for (const UnitState& us : unit_)
            if (us.slab_slot >= 0 && us.klass >= 0 && !pools_[(size_t)us.klass].is_flex(us.slab_slot)) kept += us.bytes;
        for (const auto& f : freed) gone += pools_[f.first].stride;
        RAD_INFO("mover: the slab gave back %s for a park; %s stays (units with no other copy)",
                 humanb(gone).c_str(), humanb(kept).c_str());
    }
    /* What stays in a budgeted slot, per class; every other slot's stretch is free. */
    for (size_t k = 0; k < pools_.size(); ++k) {
        ClassPools& cp = pools_[k];
        if (!cp.slab_base || cp.stride <= 0 || cp.n_fixed <= 0) continue;
        std::vector<char> used((size_t)cp.n_fixed, 0);
        for (const UnitState& us : unit_)
            if ((size_t)us.klass == k && us.slab_slot >= 0 && us.slab_slot < cp.n_fixed) used[(size_t)us.slab_slot] = 1;
        for (int64_t a = 0; a < cp.n_fixed;) {
            if (used[(size_t)a]) { ++a; continue; }
            int64_t b = a;
            while (b < cp.n_fixed && !used[(size_t)b]) ++b;
            free_ranges->push_back({ (char*)cp.slab_base + a * cp.stride, (b - a) * cp.stride });
            a = b;
        }
    }
    count_flex();
    return RAD_OK;
}

void Mover::queue_move_copy(void* dst, const void* src, int64_t bytes, int rec) {
    dev_page_invalidate(dst, bytes);   /* a parked copy of that slot is stale from here */
    move_segs_.push_back(RadCopySeg{ src, dst, bytes });
    move_recs_.push_back((int32_t)rec);
}

/* Swap copies go to the DMA engine as one list. Copy kernels would compete with the step's
 * kernels: at 64 swaps a step, Qwen3.8-Flash-Next on one card spends about 10 ms of a 36 ms step
 * in them. Without a DMA queue of this process's own, rad_memcpy_list_async falls back to copy
 * kernels. The list yields to the link's collectives, like the prefill stager's. */
int Mover::flush_moves() {
    if (move_segs_.empty()) return RAD_OK;
    int rc = rad_memcpy_list_async(move_segs_.data(), (int)move_segs_.size(), RAD_COPY_YIELD,
                                   mover_);
    for (int32_t rec : move_recs_) {
        const int er = rad_event_record(moves_[(size_t)rec].ev, mover_);
        if (rc >= 0) rc = er;
    }
    move_segs_.clear();
    move_recs_.clear();
    return rc;
}

int Mover::retire(HeatEngine* heat) {
    RAD_TRY(flush_moves());
    for (Move& m : moves_) {
        if (m.state != Move::State::Copying) continue;
        const int q = rad_event_query(m.ev);
        if (q < 0) return RAD_E_DEVICE;
        if (q == 0) continue;

        --copying_[(size_t)m.unit];
        UnitState& us = unit_[(size_t)m.unit];
        /* The copy has landed: a parked copy of the slot made while it ran is stale too. */
        {
            ClassPools& cp = pools_[(size_t)us.klass];
            if (m.slab_slot >= 0) dev_page_invalidate(cp.slot_ptr(m.slab_slot), cp.stride);
            if (m.to_slot >= 0) dev_page_invalidate(cp.slot_ptr(m.to_slot), cp.stride);
        }
        switch (m.kind) {
            case Move::Kind::Promote:
                us.slab_slot = m.slab_slot;
                /* The source pool slot is freed after quarantine, unless the destination was
                 * lent, in which case the unit keeps it (see promote_to). */
                if (m.pool_slot >= 0) us.pool_slot = -1;
                us.tier = Tier::VRAM;
                us.site = Site::Device;
                ++stats_.promotions;
                if (heat) heat->set_tier(m.unit, Tier::VRAM);
                break;
            case Move::Kind::Demote:
                /* Out of a lent slot, the unit was a donor (donor_slot): any other unit there is
                 * dropped, not demoted. */
                if (pools_[(size_t)us.klass].is_flex(m.slab_slot)) --pools_[(size_t)us.klass].n_donated;
                us.pool_slot = m.pool_slot;
                us.slab_slot = -1;
                us.tier = Tier::HostPinned;
                /* The unit joins the zero-copy tier: the GEMM streams it from the pinned pool as
                 * it computes. */
                us.site = Site::DeviceZeroCopy;
                ++stats_.demotions;
                if (heat) heat->set_tier(m.unit, Tier::HostPinned);
                break;
            case Move::Kind::Stage:
                /* The destination was published at issue with the event. The copy has landed, so
                 * clear the event and later reads of the staged unit cost no runtime call. No
                 * generation bump: the address did not change, and a bump would make the run
                 * phase wait again on an event that has already signalled. */
                for (rad_weight w : plan_->units[(size_t)m.unit].weights) res_.settle(w);
                /* Straight to Free, skipping quarantine: a staging copy returns no slot (the
                 * later Release does, and is quarantined). Quarantining it would only hold the
                 * record for two more dispatches and stall the prefetch on a real card. */
                m.clear();
                continue;
            case Move::Kind::Admit:
                /* The hot cache's copy landed: like a Stage, the slot was
                 * published at issue and the compute stream waited on the event, which now
                 * clears; and like a Stage it frees nothing, so no quarantine — the slot is
                 * the unit's home from here on. The tier change makes heat rank the unit as
                 * resident (its next miss costs nothing to stream), and evictable: a later
                 * swap frees the slot by bookkeeping (Mover::issue's Evict). */
                for (rad_weight w : plan_->units[(size_t)m.unit].weights) res_.settle(w);
                us.tier = Tier::VRAM;
                us.site = Site::Device;
                ++stats_.promotions;
                if (heat) {
                    heat->set_tier(m.unit, Tier::VRAM);
                    heat->set_busy(m.unit, false);
                }
                m.clear();
                continue;
            case Move::Kind::Deposit:
                /* The deposit's copy landed: like an Admit, the slot was
                 * published at issue and frees nothing — no quarantine. The tier change at issue
                 * made heat rank the unit as a promotion candidate; the pool slot is the unit's
                 * home, so its demotion is a copyless relabel. */
                for (rad_weight w : plan_->units[(size_t)m.unit].weights) res_.settle(w);
                us.tier = Tier::HostPinned;
                us.site = Site::DeviceZeroCopy;
                if (heat) heat->set_busy(m.unit, false);
                m.clear();
                continue;
            case Move::Kind::Migrate:
                us.slab_slot = m.to_slot;
                --pools_[(size_t)us.klass].n_donated;
                ++stats_.relocated;
                break;
            case Move::Kind::Release:
            case Move::Kind::Drop:
            case Move::Kind::Evict:   /* published at issue, never Copying */
                break;
        }
        /* Safe without synchronisation: both copies of the bytes are valid now (see
         * Residency::publish). */
        republish(m.unit);
        m.state = Move::State::Published;
    }

    /* Quarantine, as a separate pass. Kernels issued in earlier layers may still name the freed
     * slot. Re-recording the move's event on the compute stream is the barrier: when it signals,
     * every launch made before the publish has retired and the slot is safe to write. It must be
     * downstream of the publish in stream order, so it is not folded into the loop above. */
    for (Move& m : moves_) {
        if (m.state != Move::State::Published) continue;
        if (!compute_) {
            RAD_ERR("mover: no compute stream to quarantine the slot just freed by unit %d",
                    (int)m.unit);
            return RAD_E_STATE;
        }
        RAD_TRY(rad_event_record(m.ev, compute_));
        m.state = Move::State::Quarantine;
        ++stats_.quarantined;
    }

    for (Move& m : moves_) {
        if (m.state != Move::State::Quarantine) continue;
        const int q = rad_event_query(m.ev);
        if (q < 0) return RAD_E_DEVICE;
        if (q == 0) continue;

        /* A Stage record never gets here: it is freed as soon as it lands. */
        ClassPools& cp = pools_[(size_t)unit_[(size_t)m.unit].klass];
        if (m.kind == Move::Kind::Promote) {
            /* Every promotion keeps its pool slot: pool_slot is -1, nothing to free. */
            if (m.pool_slot >= 0) cp.pool.add_free(m.pool_slot);
        }
        else if (m.kind == Move::Kind::Evict && m.pool_slot >= 0)
            cp.pool.add_free(m.pool_slot);   /* pool relabel: the slot, named, is freed */
        else if (m.frees_slab())           cp.slab.add_free(m.slab_slot);
        if (heat && m.kind != Move::Kind::Release) heat->set_busy(m.unit, false);
        m.clear();
    }
    count_flex();
    return RAD_OK;
}

std::string Mover::report() const {
    const double gib = 1.0 / 1073741824.0;
    std::string s = fmt("mover: %llu dispatches, %llu promotions, %llu demotions, %llu staged "
                        "(%.2f GiB h2d, %.2f GiB d2h)\n",
                        (unsigned long long)stats_.dispatches,
                        (unsigned long long)stats_.promotions,
                        (unsigned long long)stats_.demotions,
                        (unsigned long long)stats_.prefetched,
                        (double)stats_.h2d_bytes * gib, (double)stats_.d2h_bytes * gib);
    /* Supply counters: they separate a mover at its budget from one that never got to spend it. */
    s += fmt("  %llu found no free move record, %llu starved of a slab slot, %llu of a pool "
             "slot; %llu slots quarantined\n",
             (unsigned long long)stats_.no_record, (unsigned long long)stats_.starved_slab,
             (unsigned long long)stats_.starved_pool, (unsigned long long)stats_.quarantined);
    if (stats_.donated)
        s += fmt("  %llu host copies given by lent units to budgeted victims, %llu such units moved "
                 "by a recall, %llu left past one with no free slot\n",
                 (unsigned long long)stats_.donated, (unsigned long long)stats_.relocated,
                 (unsigned long long)stats_.stranded);
    if (stats_.ssd_reads)
        s += fmt("  file tier: %llu demand reads (%.2f GiB), each a blocking pread in the middle "
                 "of a dispatch. Raise --host-pool-mib.\n",
                 (unsigned long long)stats_.ssd_reads, (double)stats_.ssd_bytes * gib);
    if (stats_.ssd_promotions)
        s += fmt("  hot cache: %llu expert units read into VRAM slots (%.2f GiB) and left there; "
                 "they stream no more\n",
                 (unsigned long long)stats_.ssd_promotions,
                 (double)stats_.ssd_promoted_bytes * gib);
    if (stats_.ssd_deposits)
        s += fmt("  ssd deposits: %llu (%.2f GiB), %llu copyless evictions of cold pool "
                 "units back to the file tier\n",
                 (unsigned long long)stats_.ssd_deposits,
                 (double)stats_.ssd_deposit_bytes * gib,
                 (unsigned long long)stats_.pool_evictions);
    /* The file stager's cost to the issuing thread: read is time in the drive, wait is time on a
     * window whose copies had not drained when it came round again. */
    if (stats_.ssd_layers) {
        const double rs = (double)stats_.ssd_read_ns * 1e-9;
        s += fmt("  file stager: %llu routed layers read, %llu units, %.2f GiB in %llu preads; "
                 "%.2f s reading (%.2f GB/s), %.2f s waiting on windows\n",
                 (unsigned long long)stats_.ssd_layers, (unsigned long long)stats_.ssd_units,
                 (double)stats_.ssd_read_bytes * gib, (unsigned long long)stats_.ssd_preads, rs,
                 rs > 0 ? (double)stats_.ssd_read_bytes / rs * 1e-9 : 0.0,
                 (double)stats_.ssd_wait_ns * 1e-9);
        if (stats_.ssd_skipped_units)
            s += fmt("  file stager: routing-aware reads skipped %llu units (%.2f GiB) whose "
                     "experts were not routed this pass\n",
                     (unsigned long long)stats_.ssd_skipped_units,
                     (double)stats_.ssd_skipped_bytes * gib);
    }
    return s;
}

}  /* namespace rad */
