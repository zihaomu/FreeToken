/* planner.cpp -- the placement planner's mechanics; planner.h has the design.
 *
 * The planner is pure: it reads the Program, the Config and the container's profile and writes a
 * Plan. It queries no device, clock or environment, which is what makes Config::deterministic
 * possible (spec §17) and keeps the resident set identical on every TP rank (planner.h, MoveUnit).
 *
 * Do not add a device query. Budgets come from Config (spec §6). rad_dev_count() also enumerates
 * integrated GPUs, which report tens of GiB of "VRAM" that is system RAM, so summing capacity
 * overstates the budget. Any code that does read RadDeviceProps must be per device and filter on
 * RadDeviceProps::arch, never on index or capacity.
 */
#include "planner.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>

namespace rad {

/* ------------------------------------------------------------------ names */
constexpr int kSiteCount = (int)Site::Host + 1;

/* Declared in rad_core.h, defined here because placement owns what a tier and a site mean. */
const char* tier_name(Tier t) {
    switch (t) {
        case Tier::VRAM:         return "vram";
        case Tier::HostPinned:   return "pinned";
        case Tier::HostPageable: return "pageable";
        case Tier::SSD:          return "ssd";
        case Tier::Mapped:       return "mapped";
    }
    return "?";
}

const char* table_placement_name(TablePlacement p) {
    switch (p) {
        case TablePlacement::Auto: return "auto";
        case TablePlacement::Vram: return "vram";
        case TablePlacement::Ram:  return "ram";
        case TablePlacement::Disk: return "disk";
    }
    return "?";
}

const char* site_name(Site s) {
    switch (s) {
        case Site::Device:         return "device";
        case Site::DeviceStaged:   return "staged";
        case Site::DeviceZeroCopy: return "zero-copy";
        case Site::Host:           return "host";
    }
    return "?";
}

const char* strategy_name(Strategy s) {
    switch (s) {
        case Strategy::AllVram:      return "all_vram";
        case Strategy::LayerOffload: return "layer_offload";
        case Strategy::ExpertTiered: return "expert_tiered";
        case Strategy::Auto:         return "auto";
    }
    return "?";
}

int strategy_parse(const char* s, Strategy* out) {
    if (!s || !out) return RAD_E_INVAL;
    if (!std::strcmp(s, "all_vram"))      { *out = Strategy::AllVram;      return RAD_OK; }
    if (!std::strcmp(s, "layer_offload")) { *out = Strategy::LayerOffload; return RAD_OK; }
    if (!std::strcmp(s, "expert_tiered")) { *out = Strategy::ExpertTiered; return RAD_OK; }
    if (!std::strcmp(s, "auto"))          { *out = Strategy::Auto;         return RAD_OK; }
    return RAD_E_INVAL;
}

Budgets Budgets::from(const Config& c) {
    Budgets b;
    b.vram_weights = c.vram_weights_mib * 1024 * 1024;
    b.host_pool    = c.host_pool_total_mib() * 1024 * 1024;
    b.ssd          = c.weights_disk_tier;
    return b;
}

/* ------------------------------------------------------------------ sizes and keys */
int64_t weight_bytes(const WeightInfo& w) {
    /* The stored size occupies the tier and differs from the logical size for a relayout. Before
     * the layout pass (as in planner unit tests), fall back to the declared dtype and shape. */
    if (w.stored_bytes > 0) return w.stored_bytes;
    if (w.logical_bytes > 0) return w.logical_bytes;
    int64_t n = w.decl.rank ? 1 : 0;
    for (uint32_t i = 0; i < w.decl.rank; ++i) n *= w.decl.shape[i];
    return rad_dtype_bytes(w.decl.dtype, n);
}


int64_t file_stage_reserve(int64_t widest, bool routed) {
    return (int64_t)RAD_PLACE_FILE_STAGE_SLOTS * widest +
           (routed ? (int64_t)RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES : 0);
}

int64_t expert_vram_floor(const WeightFootprint& wf, int64_t host_room) {
    if (wf.experts <= 0) return 0;
    const int64_t all = wf.experts + wf.expert_spare;
    return std::min(all, std::max<int64_t>(all + 2 * wf.expert_unit_max - host_room, 0));
}

/* A draft head's routed experts, pinned in the host pool before any strategy runs, leave the trunk
 * the whole of `host_room`. The card then holds only the trunk's floor against that room. The
 * drafter's units share the trunk's slab classes, so the spare the trunk's floor already reserves
 * covers them; with no trunk experts the drafter is the whole plane and pinning it leaves the card
 * nothing to hold. */
int64_t expert_vram_floor_draft_hosted(const WeightFootprint& wf, int64_t host_room) {
    if (wf.draft_experts <= 0) return expert_vram_floor(wf, host_room);
    const int64_t trunk = wf.experts - wf.draft_experts;
    if (trunk <= 0) return 0;
    WeightFootprint t = wf;
    t.experts = trunk;
    t.draft_experts = 0;
    return expert_vram_floor(t, host_room);
}

int plan_check_weights_pool(const Plan& p, int64_t pool_bytes) {
    if (p.vram_budget == pool_bytes) return RAD_OK;
    RAD_ERR("placement: rank %d planned against a %s weights budget (%lld bytes) and its weights "
            "pool holds %s (%lld bytes): the planner and the pool were given different splits",
            p.rank, humanb(p.vram_budget).c_str(), (long long)p.vram_budget,
            humanb(pool_bytes).c_str(), (long long)pool_bytes);
    return RAD_E_STATE;
}

/* ------------------------------------------------------------------ site availability */
/* A site is available for an op only if every band resolved a kernel in its domain (spec §5.1),
 * read from Band::dom rather than a flag. Every band, not just the one that runs: the same op
 * serves M=1 at decode and M=4096 at prefill (spec §2.2), and placement is decided once for all
 * steps.
 *
 * Host availability is an option the budgets may take. Device availability decides whether the
 * weight may be on a card at all; without it pin_mapped keeps it in the container's mapping.
 * Declare rejects only ops with no kernel in either domain, so a host-only op (e.g.
 * `embed_lookup_q` with no device kernel loaded) does reach the planner. */
static void resolve_sites(const Program& prog, std::vector<uint8_t>* host_ok,
                          std::vector<uint8_t>* device_ok) {
    host_ok->assign(prog.weights.size(), 1u);
    device_ok->assign(prog.weights.size(), 1u);
    for (const OpInfo& o : prog.ops) {
        bool host = !o.bands.empty();
        bool dev  = !o.bands.empty();
        for (const Band& b : o.bands) {
            if (!b.dom[RAD_DOMAIN_HOST]) host = false;
            if (!b.dom[RAD_DOMAIN_DEVICE]) dev = false;
        }
        for (rad_weight w : o.weights) {
            if (w == RAD_NULL_HANDLE || w >= host_ok->size()) continue;
            if (!host) (*host_ok)[w] = 0u;
            /* No device kernel leaves host as the only option. Declare rejects a band with neither
             * kernel, so `!dev` here implies `host`. */
            if (!dev) (*device_ok)[w] = 0u;
        }
    }
}

/* Planner::build_units and pin_mapped replayed without a Plan, for the budget resolver that runs
 * before a Planner exists. Keep in step with those two; resolve_sites is shared. */
void weight_footprint(const Program& prog, WeightFootprint* out, TablePlacement embedding,
                      TablePlacement ngram, bool host_backend, bool container) {
    if (!out) return;
    *out = WeightFootprint{};

    std::vector<uint8_t> host_ok, device_ok;
    resolve_sites(prog, &host_ok, &device_ok);

    /* build_units: a model-level weight is its own unit, everything else groups by the declared
     * (layer, expert). */
    struct U { int32_t layer = -1, expert = -1; int64_t bytes = 0; bool mapped = false;
               bool emb = false, ngram = false, host = true, direct = true;
               int access = -1; };
    std::map<std::pair<int32_t, int32_t>, size_t> by_group;
    std::vector<U> units;

    for (rad_weight w = 1; (size_t)w < prog.weights.size(); ++w) {
        const WeightInfo& wi = prog.weights[w];
        const int32_t layer = wi.decl.group.layer, expert = wi.decl.group.expert;
        size_t u;
        if (layer < 0) {
            u = units.size();
            units.push_back(U{});
        } else {
            auto key = std::make_pair(layer, expert);
            auto it = by_group.find(key);
            if (it == by_group.end()) {
                u = units.size();
                by_group.emplace(key, u);
                units.push_back(U{});
                units.back().layer = layer;
                units.back().expert = expert;
            } else {
                u = it->second;
            }
        }
        U& mu = units[u];
        const int64_t a = wi.align > 0 ? wi.align : RAD_ALIGN_UNIT;
        mu.bytes = align_up(mu.bytes, a) + weight_bytes(wi);
        /* pin_mapped: the whole unit is mapped, not the one weight. */
        if (w < device_ok.size() && !device_ok[w]) mu.mapped = true;
        if (!weight_file_direct(wi, prog.ctx.world_size)) mu.direct = false;
        if (w < host_ok.size() && !host_ok[w]) mu.host = false;
        /* build_units: the group's strongest access class, which says whether it is a tower's. */
        if (mu.access < 0 || access_strength(wi.decl.access) > access_strength(mu.access))
            mu.access = wi.decl.access;
        /* The placement flags' tables: model-level, so the unit is this one weight. */
        if (layer < 0) {
            mu.emb   = is_embedding_table(prog, w);
            mu.ngram = is_ngram_table(prog, w);
        }
    }

    for (U& mu : units) mu.mapped = mapped_unit(mu.mapped, container && mu.direct, host_backend);
    std::map<int32_t, int64_t> layer_experts;
    std::set<int64_t> expert_strides;
    bool ngram_host = true;
    for (const U& mu : units) if (mu.ngram && !mu.host) ngram_host = false;
    for (const U& mu : units) {
        /* Each unit is rounded up to RAD_ALIGN_UNIT, as Mover::init aligns before every unit it
         * places; summing raw bytes would come up short by one alignment per unit. */
        const int64_t b = align_up(mu.bytes, RAD_ALIGN_UNIT);
        /* pin_embedding moves a table out of the statics only where nothing has mapped it. */
        if (mu.emb && !mu.mapped) out->embedding += b;
        if (mu.ngram) out->ngram += b;
        if (mu.ngram && !mu.mapped) out->ngram_device = true;
        /* pin_towers moves an encoder tower's unit out of the statics on the same test, except a
         * lookup table pin_embedding has already pinned (tower_pin_bytes). */
        if (mu.access == RAD_ACCESS_PER_REQUEST && !mu.mapped && mu.expert < 0) {
            out->tower += b;
            if (mu.emb) out->tower_lookup += b;
        }
        /* pin_mapped: an n-gram table a device kernel reads stays off the card unless it was
         * placed in VRAM. */
        if (mu.mapped || (mu.ngram && ngram != TablePlacement::Vram)) out->mapped += b;
        else if (mu.emb && embedding == TablePlacement::Ram) out->hosted += b;
        else if (mu.expert >= 0) {
            out->experts += b;
            layer_experts[mu.layer] += b;
            out->expert_unit_max = std::max(out->expert_unit_max, b);
            expert_strides.insert(b);
            /* A draft head's routed experts: layers at meta.n_layers and above, and only when the
             * program declares a drafter. The trunk's own layers stay in the elastic plane a
             * strategy places; --draft-expert-placement host takes this part out of it. */
            if (prog.drafter.kind != RAD_DRAFT_NONE && prog.meta.n_layers > 0 &&
                mu.layer >= (int32_t)prog.meta.n_layers) {
                out->draft_experts += b;
                out->draft_unit_max = std::max(out->draft_unit_max, b);
            }
        }
        else                     out->statics += b;
        if (mu.layer >= 0 && !mu.mapped) out->layer_unit_max = std::max(out->layer_unit_max, b);
        if (mu.expert >= 0 && mu.layer >= 0) out->routed = true;
    }
    /* build_classes: one expert slab class a stride. */
    for (int64_t s : expert_strides)
        out->expert_spare += (int64_t)(RAD_PLACE_SHADOW_SLOTS + RAD_PLACE_STAGE_RING) * s +
                             RAD_ALIGN_UNIT;
    for (const auto& le : layer_experts)
        out->expert_layer_max = std::max(out->expert_layer_max, le.second);
    out->ngram_host = out->ngram > 0 && ngram_host;
    out->n_units = (int64_t)units.size();
}
std::string weight_class_key(const std::string& name) {
    std::string out;
    out.reserve(name.size());
    for (size_t i = 0; i < name.size();) {
        if (name[i] >= '0' && name[i] <= '9') {
            out += '*';
            while (i < name.size() && name[i] >= '0' && name[i] <= '9') ++i;
        } else {
            out += name[i++];
        }
    }
    return out;
}

/* ------------------------------------------------------------------ the mapped tier */
/* A weight no device kernel can read is pinned to the container's mapping before any budget is
 * consulted, and so is an n-gram table --ngram-placement put in RAM or on disk. This is not a
 * fallback tier: these bytes are never copied into a pool, so no pool is charged for them.
 *
 * The whole unit is mapped, not the one weight. `device_ok_` is cleared for every operand of a
 * host-only op, so the unit's other members are normally that op's scales and constants. A plugin
 * that groups a host-only weight with a device one collapses the group onto the host, and gets a
 * warning. */
void Planner::pin_mapped(const Program& prog, Plan* p) {
    const TablePlacement ng = in_.config->ngram_resolved;
    for (MoveUnit& mu : p->units) {
        /* `stored`: every weight is in the container exactly as its kernel reads it. */
        bool host_only = false, stored = true;
        for (rad_weight w : mu.weights) {
            if (w < device_ok_.size() && !device_ok_[w]) host_only = true;
            if (w < prog.weights.size() &&
                !weight_file_direct(prog.weights[w], prog.ctx.world_size)) stored = false;
        }
        /* See mapped_unit: on the host backend a non-direct unit goes to the weights pool. */
        host_only = mapped_unit(host_only, in_.container && stored, in_.host_backend);
        /* --ngram-placement keeps an n-gram table off the card unless it says vram, even when a
         * device kernel could read it; otherwise the strategies would spend VRAM on a huge table
         * read a few rows a token. */
        const bool ngram = mu.layer < 0 && !mu.weights.empty() &&
                           is_ngram_table(prog, mu.weights.front());
        if (!host_only && !(ngram && ng != TablePlacement::Vram)) continue;

        mu.tier = Tier::Mapped;
        mu.movable = false;
        if (host_only) {
            mu.site = Site::Host;
            mu.reason = "no device kernel for its op; read from the container's mapping";
        } else if (ng == TablePlacement::Ram) {
            mu.site = Site::DeviceZeroCopy;
            mu.reason = "--ngram-placement ram: a pinned copy in RAM the card reads in place, made "
                        "at load";
        } else {
            mu.site = Site::Host;
            mu.reason = "--ngram-placement disk: the host gather reads it from the container";
            if (!host_kernel(mu))
                note_unfit(p, mu, "--ngram-placement disk needs a host kernel for its gather, "
                                  "and none was loaded");
        }
        /* --ngram-placement ram: the load phase copies the table out of the mapping into locked
         * memory (host gather) or pinned memory (device gather). Pools still see the mapped tier;
         * the report shows the copy's cost. */
        if (ngram && ng == TablePlacement::Ram) {
            if (host_only) mu.reason = "--ngram-placement ram: a locked copy in RAM, made at load";
            else           p->ngram_copy_card = true;
            p->ngram_copy_bytes += mu.bytes;
        }
        /* Only a weight stored in the container exactly as its kernel reads it can be read from,
         * or copied out of, the mapping; a load-time relayout or a shard cannot (spec §5.1). */
        if (!in_.container)
            note_unfit(p, mu, "it must be read from a container's mapping, and a checkpoint served "
                              "directly has none; convert it to a container");
        else if (!stored)
            note_unfit(p, mu, "it must be read from the container's mapping, and its stored form "
                              "is not the container's (a relayout, a dtype conversion or a shard)");
        if (host_only)
            for (rad_weight w : mu.weights)
                if (w < device_ok_.size() && device_ok_[w])
                    RAD_WARN("placement: '%s' is grouped with a weight only a host kernel can "
                             "read, so the whole group stays in the container's mapping",
                             prog.weights[w].name.c_str());
    }
}

/* ------------------------------------------------------------------ units */
int access_strength(int access) {
    switch (access) {
        case RAD_ACCESS_PER_TOKEN:   return 4;
        case RAD_ACCESS_VOCAB:       return 4;
        case RAD_ACCESS_PER_REQUEST: return 2;
        case RAD_ACCESS_CONDITIONAL: return 1;
        case RAD_ACCESS_RARE:        return 0;
        default:                     return 4;   /* unknown: never risk a demotion */
    }
}

void Planner::build_units(const Program& prog, Plan* p) {
    p->unit_of_weight.assign(prog.weights.size(), -1);
    p->units.clear();

    /* Keyed on the declared RadWeightGroup. Ordered, not hashed: the unit order must depend only
     * on the model, or TP ranks could disagree about which experts are resident. */
    std::map<std::pair<int32_t, int32_t>, int32_t> by_group;

    for (rad_weight w = 1; (size_t)w < prog.weights.size(); ++w) {
        const WeightInfo& wi = prog.weights[w];
        const int32_t layer = wi.decl.group.layer;
        const int32_t expert = wi.decl.group.expert;

        /* A model-level weight is its own unit; grouping them would tie, e.g., the embedding table
         * and the output norm together despite very different sizes and access classes. */
        int32_t u;
        if (layer < 0) {
            u = (int32_t)p->units.size();
            p->units.push_back(MoveUnit{});
        } else {
            auto key = std::make_pair(layer, expert);
            auto it = by_group.find(key);
            if (it == by_group.end()) {
                u = (int32_t)p->units.size();
                by_group.emplace(key, u);
                p->units.push_back(MoveUnit{});
                p->units.back().layer = layer;
                p->units.back().expert = expert;
            } else {
                u = it->second;
            }
        }

        MoveUnit& mu = p->units[(size_t)u];
        const int64_t a = wi.align > 0 ? wi.align : RAD_ALIGN_UNIT;
        const int64_t at = align_up(mu.bytes, a);
        mu.weights.push_back(w);
        mu.offset.push_back(at);
        mu.bytes = at + weight_bytes(wi);
        if (!in_.container || !weight_file_direct(wi, prog.ctx.world_size)) mu.file_direct = false;
        /* The drafter's own weights (the "mtp." namespace) run on every draft pass of every
         * decode step; the expert fill pins them in the host pool ahead of everything ranked. */
        if (wi.name.rfind("mtp.", 0) == 0) mu.draft_head = true;
        if (wi.first_use_op >= 0 && (mu.first_use_op < 0 || wi.first_use_op < mu.first_use_op))
            mu.first_use_op = wi.first_use_op;
        if (wi.last_use_op > mu.last_use_op) mu.last_use_op = wi.last_use_op;
        if (mu.weights.size() == 1 ||
            access_strength(wi.decl.access) > access_strength(mu.access))
            mu.access = wi.decl.access;
        p->unit_of_weight[w] = u;
    }

    /* Warm start from the profile: it only decides which experts begin resident (spec §5.3). It is
     * worth about a point of hit rate, since imatrix popularity is not decode popularity and heat
     * re-ranks from real routing within a few dispatches. */
    if (in_.profile && in_.n_profile > 0) {
        std::map<std::pair<int32_t, int32_t>, float> share;
        for (int64_t i = 0; i < in_.n_profile; ++i)
            share[std::make_pair(in_.profile[i].layer, in_.profile[i].expert)] =
                in_.profile[i].share;
        for (MoveUnit& mu : p->units) {
            auto it = share.find(std::make_pair(mu.layer, mu.expert));
            if (it != share.end()) mu.profile_share = it->second;
        }
    }
}

/* ------------------------------------------------------------------ slab classes */
/* One class per (unit kind, byte size). Slots are interchangeable within a class, so a promotion
 * is an integer swap, and they are layer-agnostic: heat swaps the hottest pooled expert for the
 * coldest resident one wherever they sit, which is what makes a global ranking legal.
 */
void Planner::build_classes(Plan* p) {
    std::map<std::pair<int, int64_t>, int32_t> seen;

    for (MoveUnit& mu : p->units) {
        /* A mapped unit has no class and no slot: it occupies no pool, so its slot_bytes stays 0
         * and no budget is charged for it. */
        if (mu.tier == Tier::Mapped) continue;
        const int kind = mu.expert >= 0 ? 2 : (mu.layer >= 0 ? 1 : 0);
        const int64_t stride = align_up(mu.bytes, RAD_ALIGN_UNIT);
        auto key = std::make_pair(kind, stride);
        auto it = seen.find(key);
        if (it == seen.end()) {
            const int32_t c = (int32_t)p->classes.size();
            seen.emplace(key, c);
            SlabClass sc;
            sc.name = kind == 2 ? "expert" : (kind == 1 ? "layer" : "model");
            sc.name += "/" + humanb(stride);
            sc.stride = stride;
            p->classes.push_back(sc);
            mu.slab_class = c;
        } else {
            mu.slab_class = it->second;
        }
        mu.slot_bytes = p->classes[(size_t)mu.slab_class].stride;
        p->classes[(size_t)mu.slab_class].n_units++;
    }
}

/* ------------------------------------------------------------------ the ranking */
/* Rank-invariant: every sort key comes from the model or the container, never from the rank,
 * a clock or the device, so all TP ranks get the same order (planner.h, MoveUnit). */
void Planner::rank_units(Plan* p, std::vector<int32_t>* order) const {
    order->clear();
    order->reserve(p->units.size());
    for (int32_t i = 0; i < (int32_t)p->units.size(); ++i) order->push_back(i);
    std::stable_sort(order->begin(), order->end(), [&](int32_t a, int32_t b) {
        const MoveUnit& ua = p->units[(size_t)a];
        const MoveUnit& ub = p->units[(size_t)b];
        /* The drafter's weights run on every draft pass of every decode step -- more often than
         * anything the profile could rank -- so they take the pool first, before heat or the
         * share order gets a vote. */
        if (ua.draft_head != ub.draft_head) return ua.draft_head > ub.draft_head;
        if (ua.profile_share != ub.profile_share) return ua.profile_share > ub.profile_share;
        /* Expert-major tie-break, which is the whole policy without a profile (every share is
         * 0.0). Layer-major would fill layer 0, then layer 1, and leave the layers past the
         * budget cut with nothing resident; those layers then stream most of their experts over
         * the link inside the GEMM every step, and heat's hysteresis repairs the skew only slowly.
         * Expert-major puts expert 0 of every layer before expert 1 of any, so the cut lands at
         * the same expert index everywhere and residency starts balanced. Layers that run more
         * than once a step (a draft head) are not weighted here; heat's counters see them. */
        if (ua.expert != ub.expert) return ua.expert < ub.expert;
        return ua.layer < ub.layer;
    });
}

/* ------------------------------------------------------------------ assignment */
namespace {

/* The off-device fallback, in the order the planner tries it, shared by the layer and expert
 * paths so they cannot drift. */
struct HostFall {
    int64_t host_used = 0;
    int64_t host_budget = 0;
    bool    ssd = false;

    /* Returns true and sets tier/site, or false when nothing below VRAM will take it.
     *
     * The site decides the tier: only a unit the device reads needs pinned memory. Pinned memory
     * is scarce -- past a driver-dependent total, registration returns success and leaves the
     * process broken -- so host-site weights stay pageable. Both kinds are charged to
     * --host-pool-mib so no tier is unbudgeted. */
    bool place(MoveUnit& mu, bool host_kernel, const char** reason) {
        if (host_used + mu.slot_bytes <= host_budget) {
            host_used += mu.slot_bytes;
            if (host_kernel && mu.access == RAD_ACCESS_PER_TOKEN) {
                /* llama.cpp -ngl: bytes and arithmetic both on the host, so the weight never moves;
                 * the cost is the activation crossing at each run boundary. */
                mu.tier = Tier::HostPageable;
                mu.site = Site::Host;
                *reason = "host kernel resolved; host run";
                return true;
            }
            /* The device reads everything below out of host memory, so it must be pinned. */
            mu.tier = Tier::HostPinned;
            if (mu.access == RAD_ACCESS_CONDITIONAL) {
                /* A non-resident routed expert is not copied to VRAM first: the GEMM streams the
                 * shard from the pinned pool as it computes, so a miss costs one shard and no
                 * slab slot or eviction. */
                mu.site = Site::DeviceZeroCopy;
                *reason = "pooled expert; streamed over the link";
            } else {
                /* Staged rather than zero-copy: zero-copy would re-stream the same bytes every
                 * step, while the exact prefetch schedule hides a staged copy. */
                mu.site = Site::DeviceStaged;
                *reason = "no host kernel; staged into a slab slot";
            }
            return true;
        }
        /* The file tier, for a unit stored in the container as its kernel reads it. A Tier::SSD
         * unit has no address until staged. A scheduled unit is staged by Plan::prefetch. A routed
         * one cannot be scheduled, since a grouped MoE GEMM needs every expert's pointer at issue,
         * so the file stager reads all of a routed layer's file-tier experts into a VRAM buffer
         * before the layer's first expert op, every pass (place/stager.h, Plan::file_stage_vram).
         * Prefill reads nearly all of them anyway; decode runs at drive speed, which is the trade
         * --weights-disk-tier asks for. File-tier units are never movable: heat promotes out of
         * the pinned pool, and they have no pool slot. */
        if (ssd && mu.file_direct) {
            mu.tier = Tier::SSD;
            mu.site = Site::DeviceStaged;
            *reason = mu.access == RAD_ACCESS_CONDITIONAL
                          ? "host pool exhausted; read from the container a routed layer at a time"
                          : "host pool exhausted; file tier";
            return true;
        }
        return false;
    }
};

}  /* namespace */

int Planner::assign_all_vram(Plan* p) {
    int64_t used = 0;
    for (MoveUnit& mu : p->units) {
        /* No strategy may move a preplaced unit; every assignment pass starts with this check. */
        if (mu.preplaced()) continue;
        mu.tier = Tier::VRAM;
        mu.site = Site::Device;
        mu.movable = false;      /* the degenerate case: the mover never runs */
        used += mu.slot_bytes;
    }
    if (p->vram_budget > 0 && used > p->vram_budget) {
        p->notes.push_back(fmt("all_vram needs %s of weights against a weights pool "
                                  "of %s: %s short",
                                  humanb(used).c_str(), humanb(p->vram_budget).c_str(),
                                  humanb(used - p->vram_budget).c_str()));
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

/* Picks which layers leave VRAM; schedule_prefetch then stages them exactly.
 *
 * It spills a suffix of the execution order, which is one contiguous host run (spec §5.3; see
 * Plan::host_runs), and a suffix also gives the mover the longest lead: a staged prefix would have
 * to land before the step starts, a staged suffix has the whole step to arrive.
 */
int Planner::assign_layer_offload(Plan* p) {
    for (MoveUnit& mu : p->units) {
        if (mu.preplaced()) continue;
        mu.tier = Tier::VRAM; mu.site = Site::Device;
    }

    int64_t total = 0;
    for (const MoveUnit& mu : p->units) if (!mu.preplaced()) total += mu.slot_bytes;
    if (p->vram_budget <= 0 || total <= p->vram_budget) return RAD_OK;

    /* Layers in execution order (first use), which a plugin's layer numbering need not follow. */
    std::vector<int32_t> layers;
    std::map<int32_t, int64_t> bytes_of;
    std::map<int32_t, int32_t> first_of;
    for (const MoveUnit& mu : p->units) {
        if (mu.layer < 0) continue;
        if (!bytes_of.count(mu.layer)) { layers.push_back(mu.layer); first_of[mu.layer] = -1; }
        bytes_of[mu.layer] += mu.slot_bytes;
        int32_t& first = first_of[mu.layer];
        if (mu.first_use_op >= 0 && (first < 0 || mu.first_use_op < first)) first = mu.first_use_op;
    }
    std::stable_sort(layers.begin(), layers.end(), [&](int32_t a, int32_t b) {
        if (first_of[a] != first_of[b]) return first_of[a] < first_of[b];
        return a < b;
    });

    /* Reserve the staging ring before choosing the cut: a spilled layer without a host kernel is
     * staged and needs ring slots. Whether any layer lacks a host kernel is asked before the cut,
     * which may over-reserve; over-reserving spills one layer too many, under-reserving gives a
     * plan that does not load.
     *
     * One ring plus tail pad per layer slab class, matching what Mover::init allocates and
     * finish() charges. A hybrid model has several layer strides (attention vs linear layers). */
    bool any_stage = ssd_;
    std::map<int32_t, int64_t> stride_of;
    for (const MoveUnit& mu : p->units) {
        if (mu.layer < 0 || mu.slab_class < 0) continue;
        stride_of[mu.slab_class] = p->classes[(size_t)mu.slab_class].stride;
        if (!host_kernel(mu)) any_stage = true;
    }
    int64_t reserve = 0;
    if (any_stage)
        for (const auto& kv : stride_of)
            reserve += (int64_t)RAD_PLACE_STAGE_RING * kv.second + RAD_ALIGN_UNIT;

    int64_t need = total + reserve - p->vram_budget;
    /* `cut` is the first layer, in execution order, to leave VRAM; the spilled set is a suffix. */
    size_t cut = layers.size();
    int64_t freed = 0;
    while (cut > 0 && freed < need) { --cut; freed += bytes_of[layers[cut]]; }
    if (freed < need) {
        p->notes.push_back(fmt("layer_offload cannot fit: every layer spilled frees %s against "
                                  "%s needed; the model-level weights alone exceed the budget",
                                  humanb(freed).c_str(), humanb(need).c_str()));
    }

    HostFall fall{0, p->host_budget, ssd_};
    int rc = RAD_OK;
    for (size_t i = cut; i < layers.size(); ++i) {
        for (MoveUnit& mu : p->units) {
            if (mu.layer != layers[i] || mu.preplaced()) continue;
            if (!fall.place(mu, host_kernel(mu), &mu.reason)) {
                mu.reason = "fits in no tier";
                note_unfit(p, mu, ssd_ ? "neither VRAM nor the host pool has room"
                                       : "the host pool is full and no --weights-disk-tier was given");
                rc = RAD_E_NOMEM;
            }
        }
    }
    return rc;
}

/* The initial expert residency. Non-conditional units go to VRAM first, since they are read
 * every pass; conditional units take the rest in rank_units order, and heat re-ranks from real
 * routing from the first dispatch.
 */
int Planner::assign_expert_tiered(Plan* p) {
    int64_t fixed = 0;
    for (MoveUnit& mu : p->units) {
        if (mu.preplaced()) continue;
        mu.tier = Tier::VRAM;
        mu.site = Site::Device;
        if (mu.access != RAD_ACCESS_CONDITIONAL) fixed += mu.slot_bytes;
    }

    if (p->vram_budget > 0 && fixed > p->vram_budget) {
        /* The dense part alone does not fit: delegate to layer offload rather than duplicating
         * its run-picking logic. */
        RAD_WARN("placement: the non-routed weights alone need %s against a %s weight budget; "
                 "falling through to layer offload for them",
                 humanb(fixed).c_str(), humanb(p->vram_budget).c_str());
        RAD_TRY(assign_layer_offload(p));
        fixed = 0;
        for (const MoveUnit& mu : p->units)
            if (mu.access != RAD_ACCESS_CONDITIONAL && mu.tier == Tier::VRAM) fixed += mu.slot_bytes;
    }
    /* Host pool already spent by layer offload on dense units. The expert fill below re-places
     * the routed experts but keeps the dense units, so it spends only the remainder. */
    int64_t host_spent = 0;
    for (const MoveUnit& mu : p->units)
        if (!mu.preplaced() && mu.access != RAD_ACCESS_CONDITIONAL &&
            (mu.tier == Tier::HostPinned || mu.tier == Tier::HostPageable))
            host_spent += mu.slot_bytes;

    std::vector<int32_t> order;
    rank_units(p, &order);

    /* The spare slots come out of the same weights pool as the residents, or the mover's
     * allocation would overrun it. Two passes suffice: the first pass had more room, so its
     * spare count is an upper bound for the second. */
    HostFall fall{host_spent, p->host_budget, ssd_};
    int rc = RAD_OK;

    /* One fill: conditional units in rank order, at most `cap` resident within `room`, the rest
     * to the host pool or file tier. Returns the VRAM cost: residents, shadow slots and staging
     * ring, and the file stager's two buffers. */
    struct Cost { int64_t used = 0, spare = 0, file = 0, residents = 0; };
    auto fill = [&](int64_t room, int64_t cap) -> Cost {
        Cost k;
        for (int32_t ui : order) {
            MoveUnit& mu = p->units[(size_t)ui];
            /* A PREPLACED UNIT IS ALREADY PLACED: pin_mapped put it in the container's mapping
             * (host-only op) and build_classes gave it no class and a 0-byte slot. Re-tiering it
             * here would strand it -- VRAM tier with no slab slot behind it -- and the mover
             * would find no address for it at init. */
            if (mu.access != RAD_ACCESS_CONDITIONAL || mu.preplaced()) continue;
            mu.tier = Tier::VRAM;
            mu.site = Site::Device;
            mu.reason = "fits in vram";
        }
        for (SlabClass& sc : p->classes) sc.n_shadow = 0;
        p->unfit.clear();
        fall = HostFall{host_spent, p->host_budget, ssd_};
        rc = RAD_OK;

        for (int32_t ui : order) {
            MoveUnit& mu = p->units[(size_t)ui];
            if (mu.access != RAD_ACCESS_CONDITIONAL || mu.preplaced()) continue;
            if (k.residents < cap && k.used + mu.slot_bytes <= room) {
                k.used += mu.slot_bytes;
                ++k.residents;
                mu.tier = Tier::VRAM;
                mu.site = Site::Device;
                mu.reason = "resident: ranked by the container's profile";
                /* Only movable units are touched by heat; under Config::deterministic none are,
                 * so this resident set holds for the whole run. */
                mu.movable = p->dynamic;
                continue;
            }
            if (!fall.place(mu, host_kernel(mu), &mu.reason)) {
                mu.tier = Tier::SSD;
                mu.site = Site::DeviceStaged;
                mu.reason = "fits in no tier";
                /* With the file tier on, only a non-file-direct unit gets here (HostFall::place). */
                note_unfit(p, mu, ssd_ ? "neither VRAM nor the host pool has room, and its "
                                         "stored form is not the container's (a relayout, a "
                                         "dtype conversion or a shard), so the file tier cannot "
                                         "read it in place"
                                       : "the host pool is full and no --weights-disk-tier was "
                                         "given");
                rc = RAD_E_NOMEM;
                continue;
            }
            /* A pooled expert is movable, since heat promotes from the pool. A file-tier one has
             * no pool slot to promote from; the file stager reads it a layer at a time. */
            mu.movable = p->dynamic && mu.tier != Tier::SSD;
        }

        /* Slots used per class, which sizes the spares. A static plan reserves no shadow slots. */
        std::vector<int64_t> resident(p->classes.size(), 0), staged(p->classes.size(), 0),
            on_disk(p->classes.size(), 0);
        for (const MoveUnit& mu : p->units) {
            if (mu.slab_class < 0) continue;
            if (mu.movable && mu.tier == Tier::VRAM) ++resident[(size_t)mu.slab_class];
            if (mu.tier == Tier::SSD) ++on_disk[(size_t)mu.slab_class];
            /* The staging ring is slab memory too. finish() sizes it from the staged units,
             * known only after the fill, so it is reserved the same two-pass way as the shadow
             * slots. Routed file-tier units are excluded; the file stager has its own buffers. */
            if (mu.site == Site::DeviceStaged && mu.access != RAD_ACCESS_CONDITIONAL)
                ++staged[(size_t)mu.slab_class];
        }
        /* The hot VRAM cache's budget: with file-tier units in a class, the shadow slots
         * may be filled with hot experts — read a routed expert into a shadow slab slot
         * permanently, instead of the stager's per-pass buffer, when heat says the expert
         * keeps routing. Eviction is bookkeeping (the container is ground truth). The
         * budget is the full shadow cap rather than the swap formula's resident/4 share.
         * Bounded by the class's residents so the two-pass fill converges — a later pass
         * with fewer residents reserves fewer — and charged to the weights budget through
         * k.spare like any shadow slot. */
        for (size_t c = 0; c < p->classes.size(); ++c) {
            const int64_t ring = std::min<int64_t>(RAD_PLACE_STAGE_RING, staged[c]);
            const int64_t shadow = (p->dynamic && resident[c] > 0)
                ? std::min<int64_t>(RAD_PLACE_SHADOW_SLOTS, std::max<int64_t>(1, resident[c] / 4))
                : 0;
            const int64_t hot = (p->dynamic && on_disk[c] > 0 && resident[c] > 0)
                ? std::min<int64_t>(RAD_PLACE_SHADOW_SLOTS, resident[c]) : 0;
            p->classes[c].n_shadow = std::max(shadow, hot);
            if (p->classes[c].n_shadow + ring <= 0) continue;
            /* Plus the slab's tail pad: a kernel assembling a packed code window from a wide read
             * reads a few bytes past the last slot, which would fault at the allocation's end. */
            k.spare += (p->classes[c].n_shadow + ring) * p->classes[c].stride + RAD_ALIGN_UNIT;
        }

        /* The file stager's two buffers, each the largest routed layer's file-tier units at the
         * stager's packing, plus a tail pad. */
        std::map<int32_t, int64_t> layer_bytes;
        for (const MoveUnit& mu : p->units)
            if (mu.tier == Tier::SSD && mu.access == RAD_ACCESS_CONDITIONAL && mu.layer >= 0)
                layer_bytes[mu.layer] += align_up(mu.bytes, RAD_ALIGN_UNIT);
        p->file_layer_max = 0;
        for (const auto& kv : layer_bytes) p->file_layer_max = std::max(p->file_layer_max, kv.second);
        k.file = p->file_layer_max > 0 ? 2 * p->file_layer_max + RAD_ALIGN_UNIT : 0;
        p->file_stage_vram = k.file;
        return k;
    };

    const int64_t uncapped = INT64_MAX;
    const int64_t room0 = p->vram_budget > 0 ? p->vram_budget - fixed : INT64_MAX;
    Cost k = fill(room0, uncapped);
    if (p->vram_budget > 0) k = fill(room0 - k.spare, uncapped);

    /* The file buffers come out of the same budget, which the two-pass fill did not charge. Each
     * resident given up frees room but may grow a layer's file-tier share and so the buffers, so
     * the resident cap steps down in rank order until residents, spares and buffers fit together.
     * Rank order spreads residency over layers (rank_units), keeping the largest share small. */
    if (p->vram_budget > 0 && k.file > 0) {
        int64_t cap = k.residents;
        while (fixed + k.used + k.spare + k.file > p->vram_budget && cap > 0)
            k = fill(uncapped, --cap);
        if (fixed + k.used + k.spare + k.file > p->vram_budget) {
            p->notes.push_back(fmt("the file tier's two layer buffers need %s, beside %s of "
                                   "fixed weights, against a weights pool of %s",
                                   humanb(k.file).c_str(), humanb(fixed).c_str(),
                                   humanb(p->vram_budget).c_str()));
            return RAD_E_NOMEM;
        }
    }
    return rc;
}

bool Planner::host_kernel(const MoveUnit& mu) const {
    for (rad_weight w : mu.weights)
        if (w < host_ok_.size() && !host_ok_[w]) return false;
    return true;
}

void Planner::note_unfit(Plan* p, const MoveUnit& mu, const char* why) const {
    const std::string what = mu.weights.empty()
        ? std::string("<no weights>")
        : weight_class_key(in_.program->weights[mu.weights.front()].name);
    for (Plan::Unfit& u : p->unfit)
        if (u.what == what && u.why == why) { ++u.count; u.bytes += mu.bytes; return; }
    p->unfit.push_back(Plan::Unfit{what, why, 1, mu.bytes});
}

/* ------------------------------------------------------------------ op-level agreement */
/* An op runs in exactly one domain, so its weight operands must agree on the site: a host kernel
 * cannot read VRAM and a device kernel cannot read pageable memory. A disagreement collapses to
 * the device side, staged, since staging works from every tier. The repair is logged; it usually
 * means a plugin put a per-layer weight in a model-level group. */
void Planner::enforce_op_site_agreement(const Program& prog, Plan* p) {
    /* On the host backend the weights pool is host memory and every op runs its host kernel, which
     * reads the pool and the container's mapping alike: an op reading a mapped weight beside a
     * converted one in the pool (hc_read's norm gain and its joined E4M3 mixing matrices) has no
     * disagreement to repair. */
    if (in_.host_backend) return;
    for (const OpInfo& o : prog.ops) {
        bool any_host = false, any_dev = false;
        for (rad_weight w : o.weights) {
            if (w == RAD_NULL_HANDLE || w >= p->unit_of_weight.size()) continue;
            const int32_t u = p->unit_of_weight[w];
            if (u < 0) continue;
            if (p->units[(size_t)u].site == Site::Host) any_host = true;
            else                                        any_dev = true;
        }
        if (!(any_host && any_dev)) continue;
        for (rad_weight w : o.weights) {
            if (w == RAD_NULL_HANDLE || w >= p->unit_of_weight.size()) continue;
            const int32_t u = p->unit_of_weight[w];
            if (u < 0 || p->units[(size_t)u].site != Site::Host) continue;
            MoveUnit& mu = p->units[(size_t)u];
            /* A mapped unit cannot be demoted: no device kernel exists to read a staged copy, so
             * the op is unrunnable and the plan says so. This needs a plugin bug: a weight grouped
             * with a host-only one, or an op reading both kinds. */
            if (mu.tier == Tier::Mapped) {
                p->notes.push_back(fmt("op '%s' reads a weight only a host kernel can read "
                                       "alongside one placed on the device; no site serves both",
                                       o.op.c_str()));
                RAD_ERR("placement: op '%s' mixes a mapped weight with a device one. The mapped "
                        "weight has no device kernel to be staged for.", o.op.c_str());
                continue;
            }
            mu.site = Site::DeviceStaged;
            mu.reason = "op mixes host and device operands; demoted to staged";
            /* A repair, not a misfit: the plan still fits, so it is warned about rather than
             * listed as unfit. The report shows the `staged` site. */
            RAD_WARN("placement: op '%s' mixes host-site and device-site weight operands; layer "
                     "%d demoted to staged", o.op.c_str(), (int)mu.layer);
        }
    }
}

/* ------------------------------------------------------------------ prefetch */
/* The exact prefetch schedule: each staged, non-routed unit once, at the op that first reads it.
 * `at_op` is a deadline, not a launch time. When to start the copy is the mover's decision, since
 * it depends on slot supply (a slot frees only when its quarantine event signals), which the
 * planner cannot see. */
void Planner::schedule_prefetch(Plan* p) {
    p->prefetch.clear();
    for (int32_t i = 0; i < (int32_t)p->units.size(); ++i) {
        const MoveUnit& mu = p->units[(size_t)i];
        if (mu.site != Site::DeviceStaged) continue;
        if (mu.access == RAD_ACCESS_CONDITIONAL) continue;   /* predicted, not scheduled */
        p->prefetch.push_back(Plan::Prefetch{mu.first_use_op < 0 ? 0 : mu.first_use_op, i});
    }
    std::stable_sort(p->prefetch.begin(), p->prefetch.end(),
                     [](const Plan::Prefetch& a, const Plan::Prefetch& b) {
                         if (a.at_op != b.at_op) return a.at_op < b.at_op;
                         return a.unit < b.unit;
                     });
}

/* ------------------------------------------------------------------ finish */
void Planner::finish(const Program& prog, Plan* p) {
    /* Only a movable unit gets a slab slot; a static resident lives at a fixed offset
     * (slab_slot == -1). */
    std::vector<int64_t> next(p->classes.size(), 0);
    std::vector<int64_t> staged(p->classes.size(), 0);
    for (MoveUnit& mu : p->units) {
        if (mu.movable && mu.tier == Tier::VRAM && mu.slab_class >= 0) {
            mu.slab_slot = (int32_t)next[(size_t)mu.slab_class]++;
        } else {
            mu.slab_slot = -1;
        }
        /* The ring serves the exact prefetch; routed file-tier units use the file stager's
         * buffers (Plan::file_stage_vram). */
        if (mu.site == Site::DeviceStaged && mu.slab_class >= 0 &&
            mu.access != RAD_ACCESS_CONDITIONAL)
            ++staged[(size_t)mu.slab_class];
    }
    for (size_t c = 0; c < p->classes.size(); ++c) {
        p->classes[c].n_slots = next[c];
        p->classes[c].movable = next[c] > 0;
        /* The staging ring, capped at the number of staged units in the class. */
        p->classes[c].n_stage = std::min<int64_t>(RAD_PLACE_STAGE_RING, staged[c]);
        /* n_shadow comes from the assignment pass, which reserved it out of the budget; do not
         * recompute it here, or the mover and the budget would see different numbers. */
        if (!p->classes[c].movable) p->classes[c].n_shadow = 0;
        const int64_t slots = p->classes[c].n_slots + p->classes[c].n_shadow +
                              p->classes[c].n_stage;
        p->slab_overhead += (p->classes[c].n_shadow + p->classes[c].n_stage) * p->classes[c].stride;
        if (slots > 0) p->slab_overhead += RAD_ALIGN_UNIT;   /* the slab's tail pad */
    }

    /* Host runs, from the final assignment. */
    std::vector<std::pair<int32_t, int64_t>> host_layers;
    for (const MoveUnit& mu : p->units)
        if (mu.site == Site::Host && mu.layer >= 0) host_layers.push_back({mu.layer, mu.bytes});
    std::sort(host_layers.begin(), host_layers.end());
    for (size_t i = 0; i < host_layers.size();) {
        Plan::HostRun r;
        r.first_layer = host_layers[i].first;
        r.last_layer = host_layers[i].first;
        while (i < host_layers.size() &&
               (host_layers[i].first == r.last_layer || host_layers[i].first == r.last_layer + 1)) {
            r.last_layer = host_layers[i].first;
            r.bytes += host_layers[i].second;
            ++i;
        }
        p->host_runs.push_back(r);
    }
    /* Two crossings a run: to the host at its first layer, back after its last. */
    p->link_crossings = 2 * (int32_t)p->host_runs.size();

    /* Copy each unit's decision onto its weights, which is what the rest of the engine reads. */
    p->weight.assign(prog.weights.size(), Placement{});
    p->totals = Plan::Totals{};
    for (int32_t i = 0; i < (int32_t)p->units.size(); ++i) {
        const MoveUnit& mu = p->units[(size_t)i];
        for (rad_weight w : mu.weights) {
            Placement& pl = p->weight[w];
            pl.tier = mu.tier;
            pl.site = mu.site;
            pl.slab_slot = mu.slab_slot;
            pl.unit = i;
            pl.reason = mu.reason;
            pl.bytes = weight_bytes(prog.weights[w]);
            p->totals.bytes[(int)mu.tier] += pl.bytes;
            if (mu.host_forced) p->forced_host_used += pl.bytes;
        }
    }
}

/* ------------------------------------------------------------------ the lookup tables */
bool read_only_by(const Program& prog, rad_weight w, const char* op) {
    bool read = false;
    for (const OpInfo& o : prog.ops)
        for (rad_weight ow : o.weights)
            if (ow == w) {
                if (o.op != op) return false;
                read = true;
            }
    return read;
}

/* --embedding-placement ram: the token embedding goes to the pinned host pool before any strategy
 * runs. The host pool is mapped (Pools::configure), so at the DeviceZeroCopy site the lookup
 * kernel reads its M rows over the link in place, a few KiB a decode step, and the table's VRAM
 * goes to the KV cache. Strategies skip it via MoveUnit::preplaced(), and weight_footprint counts
 * it out of the statics with the same test, so budget and plan agree. Only model-level weights
 * qualify (is_embedding_table). */
int Planner::pin_embedding(const Program& prog, const Config& cfg, Plan* p) const {
    if (cfg.embedding_resolved != TablePlacement::Ram) return RAD_OK;
    int rc = RAD_OK, found = 0;
    for (MoveUnit& mu : p->units) {
        /* A model-level unit holds exactly one weight (build_units). */
        if (mu.layer >= 0 || mu.weights.empty() || mu.tier == Tier::Mapped) continue;
        const rad_weight w = mu.weights.front();
        if (!is_embedding_table(prog, w)) continue;
        ++found;
        const std::string& name = prog.weights[w].name;
        if (mu.slot_bytes > p->host_budget) {
            RAD_ERR("placement: --embedding-placement ram: '%s' needs %s of the pinned host pool "
                    "and %s of --host-pool-mib is left%s", name.c_str(),
                    humanb(mu.slot_bytes).c_str(), humanb(p->host_budget).c_str(),
                    p->file_stage_reserve ? " after the file tier's staging" : "");
            p->notes.push_back(fmt("--embedding-placement ram: '%s' (%s) does not fit the host "
                                   "pool", name.c_str(), humanb(mu.slot_bytes).c_str()));
            rc = RAD_E_NOMEM;
            continue;
        }
        p->host_budget -= mu.slot_bytes;
        p->embedding_host_bytes += mu.slot_bytes;
        mu.tier = Tier::HostPinned;
        mu.site = Site::DeviceZeroCopy;
        mu.movable = false;
        mu.host_forced = true;
        mu.reason = "--embedding-placement ram; read over the link in place";
        RAD_INFO("placement: '%s' (%s) in the pinned host pool, read in place "
                 "(--embedding-placement ram%s)", name.c_str(), humanb(mu.bytes).c_str(),
                 cfg.embedding_placement_given ? "" : ", the default");
    }
    /* Engine::plan resolves Ram only where weight_footprint found a table by the same test, so
     * finding none means the plan and the budget saw different programs. */
    if (!found) {
        RAD_ERR("placement: --embedding-placement ram found no model-level weight that only "
                "embed_lookup reads");
        p->notes.push_back("--embedding-placement ram: no token embedding table to place");
        rc = RAD_E_STATE;
    }
    return rc;
}

/* --vision-tower-placement ram: every encoder-tower unit goes to the pinned host pool before any
 * strategy runs, read in place at the DeviceZeroCopy site once per image or clip; text-only
 * requests never touch it. A tower unit is a non-expert unit with access RAD_ACCESS_PER_REQUEST,
 * the same test weight_footprint uses. It is tested by access class, not level, because a tower
 * has both model-level weights (a stem, a merger) and layer units (its blocks). */
int Planner::pin_towers(const Config& cfg, Plan* p) const {
    if (cfg.tower_resolved != TablePlacement::Ram) return RAD_OK;
    auto tower = [](const MoveUnit& mu) {
        return mu.access == RAD_ACCESS_PER_REQUEST && mu.expert < 0 && mu.tier != Tier::Mapped &&
               !mu.host_forced;
    };
    int64_t need = 0, n = 0;
    for (const MoveUnit& mu : p->units)
        if (tower(mu)) { need += mu.slot_bytes; ++n; }
    /* Engine::plan resolves Ram only where weight_footprint found a tower by the same test. */
    if (!n) {
        RAD_ERR("placement: --vision-tower-placement ram found no weight read once per request");
        p->notes.push_back("--vision-tower-placement ram: no encoder tower to place");
        return RAD_E_STATE;
    }
    if (need > p->host_budget) {
        RAD_ERR("placement: --vision-tower-placement ram: the encoder towers need %s of the pinned "
                "host pool and %s of it is left%s", humanb(need).c_str(),
                humanb(p->host_budget).c_str(),
                p->file_stage_reserve ? " after the file tier's staging" : "");
        p->notes.push_back(fmt("--vision-tower-placement ram: the encoder towers (%s) do not fit "
                               "the host pool", humanb(need).c_str()));
        return RAD_E_NOMEM;
    }
    for (MoveUnit& mu : p->units) {
        if (!tower(mu)) continue;
        p->host_budget -= mu.slot_bytes;
        p->tower_host_bytes += mu.slot_bytes;
        mu.tier = Tier::HostPinned;
        mu.site = Site::DeviceZeroCopy;
        mu.movable = false;
        mu.host_forced = true;
        mu.reason = "--vision-tower-placement ram; read over the link in place";
    }
    RAD_INFO("placement: the encoder towers, %lld units (%s), in the pinned host pool, read in place "
             "(--vision-tower-placement ram)", (long long)n, humanb(need).c_str());
    return RAD_OK;
}

/* --draft-expert-placement host: a draft head's routed experts (layers at meta.n_layers and above,
 * and only when the program declares a drafter) go to the pinned host pool before any strategy
 * runs. The card reads them in place at the DeviceZeroCopy site, as it reads a pooled expert, so
 * a drafter's experts fit beside a trunk that already fills the card. They are not movable: heat
 * never promotes a draft expert onto the card and into the trunk's slab. */
int Planner::pin_draft_experts(const Program& prog, const Config& cfg, Plan* p) const {
    if (cfg.draft_expert_placement != "host") return RAD_OK;
    if (prog.drafter.kind == RAD_DRAFT_NONE || prog.meta.n_layers == 0) {
        if (cfg.draft_expert_placement_given) {
            RAD_ERR("placement: --draft-expert-placement host: this model declares no draft head");
            p->notes.push_back("--draft-expert-placement host: no draft head to place");
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }
    const int32_t trunk = (int32_t)prog.meta.n_layers;
    auto draft = [&](const MoveUnit& mu) {
        return mu.expert >= 0 && mu.layer >= trunk && mu.tier != Tier::Mapped && !mu.host_forced;
    };
    int64_t need = 0, n = 0, widest = 0;
    for (const MoveUnit& mu : p->units)
        if (draft(mu)) { need += mu.slot_bytes; widest = std::max(widest, mu.slot_bytes); ++n; }
    if (!n) {
        RAD_ERR("placement: --draft-expert-placement host: the draft head declares no routed "
                "experts");
        p->notes.push_back("--draft-expert-placement host: no drafter experts to place");
        return RAD_E_STATE;
    }
    if (need > p->host_budget) {
        RAD_ERR("placement: --draft-expert-placement host: the drafter's experts need %s of the "
                "pinned host pool and %s of it is left%s", humanb(need).c_str(),
                humanb(p->host_budget).c_str(),
                p->file_stage_reserve ? " after the file tier's staging" : "");
        p->notes.push_back(fmt("--draft-expert-placement host: the drafter's experts (%s) do not "
                               "fit the host pool", humanb(need).c_str()));
        return RAD_E_NOMEM;
    }
    for (MoveUnit& mu : p->units) {
        if (!draft(mu)) continue;
        p->host_budget -= mu.slot_bytes;
        p->draft_host_bytes += mu.slot_bytes;
        mu.tier = Tier::HostPinned;
        mu.site = Site::DeviceZeroCopy;
        mu.movable = false;
        mu.host_forced = true;
        mu.reason = "--draft-expert-placement host; read over the link in place";
    }
    RAD_INFO("placement: the drafter's experts, %lld units (%s, widest %s), in the pinned host "
             "pool, read in place (--draft-expert-placement host)", (long long)n,
             humanb(need).c_str(), humanb(widest).c_str());
    return RAD_OK;
}

/* --expert-pin-top F: the hottest fraction of the routed experts, ranked by the profile the planner
 * was given (the container's, or --expert-profile's decode counts), stay resident. They are marked
 * unmovable, so the heat engine never demotes them and the decode-hot set never pays the host
 * pool's PCIe read. The cut keeps the hottest ceil(F * n), and a tie at the cut stays movable: the
 * threshold is the share just below the hot set, and only a share strictly above it pins. Runs
 * after the strategy, which has already put the hot set in VRAM; a hot unit the strategy had to
 * leave on the host stays there and is counted, since pinning it would need VRAM the card does
 * not have. */
void Planner::pin_hot(const Config& cfg, Plan* p) const {
    if (cfg.expert_pin_top <= 0.0f) return;
    std::vector<float> shares;
    for (const MoveUnit& mu : p->units)
        if (mu.expert >= 0 && mu.tier != Tier::Mapped) shares.push_back(mu.profile_share);
    if (shares.empty()) return;
    std::sort(shares.begin(), shares.end());
    /* The cut sits after the coldest (1 - F) of the plane, so the hottest F are pinned. */
    const double keep = (double)cfg.expert_pin_top * (double)shares.size();
    const size_t hot = keep <= 0.0 ? 0 : std::min(shares.size(), (size_t)std::ceil(keep));
    const float threshold = hot == 0 ? shares.back() + 1.0f
                                     : shares[shares.size() - hot] - 1.0f;
    p->pin_threshold = threshold;
    p->profile_units = (int64_t)shares.size();
    int64_t pinned = 0, left = 0;
    for (MoveUnit& mu : p->units) {
        if (mu.expert < 0 || mu.tier == Tier::Mapped || mu.host_forced) continue;
        if (mu.profile_share <= threshold) continue;
        if (mu.tier != Tier::VRAM) { ++left; continue; }
        mu.pinned = true;
        mu.movable = false;
        ++pinned;
    }
    p->pinned_units = pinned;
    RAD_INFO("placement: --expert-pin-top %.4g pins %lld of %lld routed experts resident (share "
             "above %.4g); %lld of the hot set had no VRAM slot and stay where the strategy put "
             "them", (double)cfg.expert_pin_top, (long long)pinned, (long long)shares.size(),
             (double)threshold, (long long)left);
}

/* ------------------------------------------------------------------ plan */
int Planner::plan(Plan* out) {
    if (!out || !in_.program || !in_.config) return RAD_E_INVAL;
    const Program& prog = *in_.program;
    const Config& cfg = *in_.config;

    *out = Plan{};
    out->rank = in_.rank;
    out->world_size = in_.world_size;
    out->dynamic = !cfg.deterministic;

    const Budgets b = Budgets::from(cfg);
    out->vram_budget = b.vram_weights;
    out->host_budget = b.host_pool;
    ssd_ = b.ssd;

    if (strategy_parse(cfg.placement.c_str(), &out->strategy) < 0) {
        RAD_ERR("placement: unknown strategy '%s'; expected all_vram, layer_offload, "
                "expert_tiered or auto", cfg.placement.c_str());
        return RAD_E_INVAL;
    }

    resolve_sites(prog, &host_ok_, &device_ok_);
    build_units(prog, out);
    /* Before build_classes and assign: mapped units get no class and no budget. */
    pin_mapped(prog, out);
    build_classes(out);

    /* After build_classes: a unit's slot_bytes is its class's stride, and the pins below charge
     * the host pool that. Before it the charge would be zero and the pool would not hold them.
     * Not RAD_TRY: a refusal still leaves `out` filled in far enough to print. */
    int hrc = pin_embedding(prog, cfg, out);
    const int trc = pin_towers(cfg, out);
    if (hrc >= 0) hrc = trc;
    if (hrc >= 0) hrc = pin_draft_experts(prog, cfg, out);

    /* Charge the file tier's staging pool (RAD_PLACE_FILE_STAGE_SLOTS) before anything is placed
     * in the host pool, or the fill could exhaust it and the mover refuse the plan. Sized from the
     * widest class carrying a layer unit, the only kind either offload strategy spills; this
     * upper-bounds what the mover computes. A model with routed units also reserves the read
     * windows, since which units reach the file tier is decided only by the fill below. */
    if (ssd_) {
        int64_t widest = 0;
        bool routed = false;
        for (const MoveUnit& mu : out->units) {
            if (mu.access == RAD_ACCESS_CONDITIONAL && mu.layer >= 0) routed = true;
            if (mu.layer < 0 || mu.slab_class < 0) continue;
            widest = std::max(widest, out->classes[(size_t)mu.slab_class].stride);
        }
        const int64_t windows = routed ? (int64_t)RAD_PLACE_FILE_WINDOWS *
                                         RAD_PLACE_FILE_WINDOW_BYTES : 0;
        const int64_t need = file_stage_reserve(widest, routed);
        if (need > 0) {
            if (need >= out->host_budget) {
                RAD_ERR("placement: the file tier needs %s of pinned staging (%d slots of %s%s) "
                        "and --host-pool-mib is %s. Raise it, or drop --weights-disk-tier.",
                        humanb(need).c_str(), RAD_PLACE_FILE_STAGE_SLOTS, humanb(widest).c_str(),
                        windows ? fmt(", and %d read windows of %s", RAD_PLACE_FILE_WINDOWS,
                                      humanb(RAD_PLACE_FILE_WINDOW_BYTES).c_str()).c_str() : "",
                        humanb(out->host_budget).c_str());
                return RAD_E_NOMEM;
            }
            out->host_budget -= need;
            out->file_stage_reserve = need;
            out->file_windows = windows;
        }
    }

    int rc = assign(out);
    enforce_op_site_agreement(prog, out);
    schedule_prefetch(out);
    /* After the strategy: it has put the hot set in VRAM, and pinning marks those resident. */
    pin_hot(cfg, out);
    finish(prog, out);
    return hrc < 0 ? hrc : rc;
}

int Planner::assign(Plan* p) {
    int64_t total = 0;
    bool has_conditional = false;
    for (const MoveUnit& mu : p->units) {
        if (!mu.preplaced()) total += mu.slot_bytes;
        if (mu.access == RAD_ACCESS_CONDITIONAL) has_conditional = true;
    }

    Strategy s = p->strategy;
    if (s == Strategy::Auto) {
        /* `auto` picks one of the three named strategies: all_vram if it fits, else expert
         * offload when there are routed experts (read only when selected), else layer offload. */
        if (p->vram_budget <= 0 || total <= p->vram_budget) s = Strategy::AllVram;
        else if (has_conditional)                           s = Strategy::ExpertTiered;
        else                                                s = Strategy::LayerOffload;
        p->strategy = s;
        p->strategy_why =
            s == Strategy::AllVram
                ? fmt("%s of weights fit the %s weights pool", humanb(total).c_str(),
                      humanb(p->vram_budget).c_str())
            : s == Strategy::ExpertTiered
                ? fmt("%s of weights against a %s weights pool: routed experts, read only when "
                      "selected, are what spills to host memory", humanb(total).c_str(),
                      humanb(p->vram_budget).c_str())
                : fmt("%s of weights against a %s weights pool and no routed experts to spill: "
                      "whole layers are offloaded", humanb(total).c_str(),
                      humanb(p->vram_budget).c_str());
        RAD_DEBUG("placement: auto chose %s (%s of weights against a %s budget)",
                  strategy_name(s), humanb(total).c_str(), humanb(p->vram_budget).c_str());
    }

    if (s != Strategy::AllVram && p->vram_budget <= 0) {
        /* Offload needs an explicit budget (spec §6); inferring one from free VRAM would make
         * runs irreproducible. */
        RAD_ERR("placement: %s needs a weights budget and none was resolved; there is nothing to "
                "spill against",
                strategy_name(s));
        return RAD_E_INVAL;
    }

    switch (s) {
        case Strategy::AllVram:      return assign_all_vram(p);
        case Strategy::LayerOffload: return assign_layer_offload(p);
        case Strategy::ExpertTiered: return assign_expert_tiered(p);
        default:                     return RAD_E_INVAL;
    }
}

/* ------------------------------------------------------------------ the report (spec §16) */
std::string Plan::report(const Program& prog) const {
    /* One line per weight class (weight_class_key), not per weight. */
    struct Row {
        int64_t n = 0, bytes = 0;
        int64_t per_tier[kTierCount] = {0};
        int64_t per_site[kSiteCount] = {0};
    };
    std::map<std::string, Row> rows;
    std::vector<std::string> order;

    for (rad_weight w = 1; (size_t)w < prog.weights.size() && (size_t)w < weight.size(); ++w) {
        const std::string key = weight_class_key(prog.weights[w].name);
        if (!rows.count(key)) order.push_back(key);
        Row& r = rows[key];
        ++r.n;
        r.bytes += weight[w].bytes;
        r.per_tier[(int)weight[w].tier] += 1;
        r.per_site[(int)weight[w].site] += 1;
    }

    size_t kw = 5;
    for (const std::string& k : order) kw = std::max(kw, k.size());

    std::string s;
    s += fmt("placement plan  rank %d/%d  strategy %s  %s\n\n",
             rank, world_size, strategy_name(strategy),
             dynamic ? "dynamic" : "static (--deterministic)");

    s += fmt("  %-*s %8s %12s  %-34s %s\n", (int)kw, "class", "count", "bytes", "tier", "site");
    for (const std::string& k : order) {
        const Row& r = rows[k];
        std::string tiers, sites;
        for (int t = 0; t < kTierCount; ++t)
            if (r.per_tier[t])
                tiers += fmt("%s%s %lld", tiers.empty() ? "" : " / ", tier_name((Tier)t),
                             (long long)r.per_tier[t]);
        for (int t = 0; t < kSiteCount; ++t)
            if (r.per_site[t])
                sites += fmt("%s%s %lld", sites.empty() ? "" : " / ", site_name((Site)t),
                             (long long)r.per_site[t]);
        s += fmt("  %-*s %8lld %12s  %-34s %s\n", (int)kw, k.c_str(), (long long)r.n,
                 humanb(r.bytes).c_str(), tiers.c_str(), sites.c_str());
    }

    s += "\n  pool                budget          used\n";
    s += fmt("  vram weights  %12s  %12s\n",
             vram_budget > 0 ? humanb(vram_budget).c_str() : "uncapped",
             humanb(totals.bytes[(int)Tier::VRAM]).c_str());
    s += fmt("  host pinned   %12s  %12s\n",
             host_budget > 0 ? humanb(host_budget).c_str() : "none",
             humanb(totals.bytes[(int)Tier::HostPinned] - forced_host_used).c_str());
    if (totals.bytes[(int)Tier::HostPageable])
        s += fmt("  host pageable %12s  %12s\n", "-",
                 humanb(totals.bytes[(int)Tier::HostPageable]).c_str());
    s += fmt("  ssd tier      %12s  %12s\n", "unbounded",
             humanb(totals.bytes[(int)Tier::SSD]).c_str());
    /* No budget: a mapped weight occupies only page cache, which the OS manages. */
    if (totals.bytes[(int)Tier::Mapped])
        s += fmt("  mapped        %12s  %12s   (the container's mapping; no pool)\n", "-",
                 humanb(totals.bytes[(int)Tier::Mapped]).c_str());
    /* Taken out of --host-pool-mib before placement, so the `host pinned` budget above is smaller
     * than the flag by this much. */
    if (file_stage_reserve)
        s += fmt("  file staging  %12s  %12s   (%d slots%s, pinned)\n", "-",
                 humanb(file_stage_reserve).c_str(), RAD_PLACE_FILE_STAGE_SLOTS,
                 file_windows ? fmt(" and %d read windows", RAD_PLACE_FILE_WINDOWS).c_str() : "");
    /* Likewise taken out of --host-pool-mib: the embedding and the encoder towers. The n-gram copy
     * is in no pool; the load phase fills it. */
    if (embedding_host_bytes)
        s += fmt("  embedding     %12s  %12s   (--embedding-placement ram, pinned, read in place)\n",
                 "-", humanb(embedding_host_bytes).c_str());
    if (tower_host_bytes)
        s += fmt("  encoder tower %12s  %12s   (--vision-tower-placement ram, pinned, read in place)\n",
                 "-", humanb(tower_host_bytes).c_str());
    if (draft_host_bytes)
        s += fmt("  drafter       %12s  %12s   (--draft-expert-placement host, pinned, read in place)\n",
                 "-", humanb(draft_host_bytes).c_str());
    if (pinned_units)
        s += fmt("  pinned hot    %12s  %12lld   (--expert-pin-top: resident, heat never demotes them, "
                 "share above %.4g of %lld)\n", "-", (long long)pinned_units, (double)pin_threshold,
                 (long long)profile_units);
    if (ngram_copy_bytes)
        s += fmt("  n-gram copy   %12s  %12s   (--ngram-placement ram, %s, filled at load)\n",
                 "-", humanb(ngram_copy_bytes).c_str(),
                 ngram_copy_card ? "pinned RAM the card reads" : "locked RAM");
    /* The file stager's two VRAM buffers, out of the weight budget but not among the residents. */
    if (file_stage_vram)
        s += fmt("  file buffers  %12s  %12s   (two of %s: the most one routed layer reads from "
                 "the container)\n", "-", humanb(file_stage_vram).c_str(),
                 humanb(file_layer_max).c_str());

    if (slab_overhead)
        s += fmt("  slab overhead %12s  %12s   (shadow slots + the staging ring)\n", "-",
                 humanb(slab_overhead).c_str());

    bool any_slab = false;
    for (const SlabClass& c : classes) if (c.movable || c.n_stage) any_slab = true;
    if (any_slab) {
        s += "\n  slab               stride       slots     shadow      stage\n";
        for (const SlabClass& c : classes) {
            if (!c.movable && !c.n_stage) continue;
            s += fmt("  %-16s %10s  %10lld %10lld %10lld\n", c.name.c_str(),
                     humanb(c.stride).c_str(), (long long)c.n_slots, (long long)c.n_shadow,
                     (long long)c.n_stage);
        }
    }

    if (!host_runs.empty()) {
        s += fmt("\n  host runs: %d run%s, %d link crossing%s a step\n", (int)host_runs.size(),
                 host_runs.size() == 1 ? "" : "s", link_crossings,
                 link_crossings == 1 ? "" : "s");
        for (const HostRun& r : host_runs)
            s += fmt("    layers %d-%d  %s\n", (int)r.first_layer, (int)r.last_layer,
                     humanb(r.bytes).c_str());
    }

    if (!prefetch.empty())
        s += fmt("\n  %d staged unit%s with an exact prefetch deadline; the first is op %d\n",
                 (int)prefetch.size(), prefetch.size() == 1 ? "" : "s", (int)prefetch[0].at_op);

    if (!notes.empty() || !unfit.empty()) {
        s += "\n  DID NOT FIT\n";
        for (const std::string& n : notes) s += "    " + n + "\n";
        /* Aggregated by class and reason: the count and bytes are what sizing the next run needs. */
        for (const Unfit& u : unfit)
            s += fmt("    %lld x %s (%s): %s\n", (long long)u.count, u.what.c_str(),
                     humanb(u.bytes).c_str(), u.why.c_str());
    }
    return s;
}

}  /* namespace rad */
