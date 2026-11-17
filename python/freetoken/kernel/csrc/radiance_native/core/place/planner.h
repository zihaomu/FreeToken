/* planner.h -- the placement planner. Runs once, after declare and before load, and assigns every
 * weight a tier and an execution site against explicit pool budgets (spec §5.3).
 *
 * All-in-VRAM, layer offload and expert offload are one mechanism at three granularities, over
 * the same tiers. They differ in whether the access is predictable:
 *
 *   Layer offload is a scheduling problem. WeightInfo::first_use_op is the full access sequence,
 *   known statically from declare, so Plan::prefetch is an exact schedule with no predictor.
 *
 *   Expert offload is a prediction problem. Routing is data-dependent, so the planner only ranks
 *   an initial residency and heat.h re-ranks from real routing.
 *
 *   All-in-VRAM is the degenerate case: every weight on tier 0 and the mover never runs.
 *
 * Tier (where the bytes live) and site (where the arithmetic runs) are independent axes
 * (spec §5.1):
 *
 *   VRAM       + Device          vLLM
 *   host/SSD   + DeviceStaged    staged into a slab slot, exact or predicted
 *   host       + DeviceZeroCopy  the GEMM streams the weight over the link as it computes; no
 *                                slab slot, so a read-once weight costs no residency
 *   host       + Host            llama.cpp -ngl
 *
 * The host site needs a RAD_DOMAIN_HOST kernel; the planner reads its availability from
 * Band::dom[RAD_DOMAIN_HOST] rather than from a flag.
 */
#pragma once
#include "../rad_core.h"
#include "rad_format.h"

#include <string>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ strategy */
enum class Strategy { AllVram = 0, LayerOffload = 1, ExpertTiered = 2, Auto = 3 };

const char* strategy_name(Strategy s);
/* RAD_E_INVAL and *out untouched if the string is not one of the four. */
int         strategy_parse(const char* s, Strategy* out);

/* ------------------------------------------------------------------ budgets */
/* Every pool is budgeted explicitly: no profiling pass, no utilisation fraction (spec §6). Zero
 * vram_weights means "no cap", which only all_vram accepts; an offload strategy needs a number
 * to spill against.
 */
struct Budgets {
    int64_t vram_weights = 0;    /* bytes; 0 = uncapped, legal only when everything fits */
    int64_t host_pool = 0;       /* bytes; the pinned host tier. 0 = no host tier */
    bool    ssd = false;         /* --weights-disk-tier was given */

    static Budgets from(const Config& c);
};

/* ------------------------------------------------------------------ the movement unit */
/* The unit of placement is a weight group, not a tensor. An expert's gate/up/down share one
 * RadWeightGroup because a dispatch missing one of them cannot compute; a layer's weights share one
 * because layer offload moves contiguous runs (spec §5.3). Slab stride, mover copies and heat
 * ranking are all per unit, and a weight's placement is its unit's placement.
 *
 * Under tensor parallel a unit is this rank's shard: dimensions are declared already divided
 * (spec §9), so `bytes` is per rank. Invariant: the resident set must be identical on every rank,
 * or a slab slot holds different experts on different cards. The ranking therefore uses only the
 * container's profile and the unit's identity -- no free-VRAM query, no timing, no device
 * enumeration.
 */
struct MoveUnit {
    int32_t layer = -1;          /* RadWeightGroup::layer; -1 for a model-level weight */
    int32_t expert = -1;         /* RadWeightGroup::expert; -1 for a non-expert group */
    int32_t slab_class = -1;     /* index into Plan::classes */
    int64_t bytes = 0;           /* this rank's shard of the whole group */
    int32_t first_use_op = -1;   /* min over the group; the prefetch schedule keys on it */
    int32_t last_use_op = -1;
    int     access = RAD_ACCESS_PER_TOKEN;   /* the group's strongest access class */
    float   profile_share = 0.0f;            /* the container's popularity profile, or 0 */

    Tier    tier = Tier::VRAM;
    Site    site = Site::Device;
    int32_t slab_slot = -1;
    bool    movable = false;     /* may the heat engine relocate it after load */
    bool    pinned = false;      /* a decode-hot resident --expert-pin-top kept: heat never demotes it */
    bool    host_forced = false; /* a placement flag put it in the pinned pool: the embedding's,
                                  * the encoder tower's or the draft head's experts' */

    /* Placed before any strategy runs, so every assignment pass skips it and no VRAM total counts
     * it: the mapped tier, --embedding-placement ram and --vision-tower-placement ram. */
    bool preplaced() const { return tier == Tier::Mapped || host_forced; }
    const char* reason = "fits in vram";   /* a fixed string for the report */

    /* The group's internal layout, computed once here: `offset[i]` is where weights[i] sits in the
     * unit at its declared alignment, and `bytes` is the padded total. The mover's copy size, pool
     * offset and published pointer all derive from it. */
    std::vector<rad_weight> weights;
    std::vector<int64_t>    offset;

    /* What the unit costs a pool: its class's slot stride, not `bytes`. The planner budgets and
     * the mover allocates this same number, so they cannot drift into a weights-pool overrun. The
     * report still attributes `bytes`, since padding is not weight. */
    int64_t slot_bytes = 0;

    /* Every weight lies in the container exactly as its kernel reads it (weight_file_direct), so
     * the file tier can stage it with a read. A unit that is not never goes to the file tier. */
    bool    file_direct = true;

    /* The drafter's own weights (the "mtp." namespace). They run on every draft pass of every
     * decode step -- up to depth passes plus the verify pass, each re-staging them from the drive
     * if they land in the file tier -- so the expert fill pins them in the host pool ahead of
     * everything ranked. One layer's experts, a fraction of the pool. */
    bool    draft_head = false;
};

/* The strongest access class in a group, "read most surely", decides the group's: a group with
 * one PER_TOKEN weight is read every forward pass, and calling it conditional would let heat
 * demote something the next token needs. VOCAB ranks with PER_TOKEN; lm_head and the embedding
 * are read every token and differ only in size and the gather (spec §5.2). */
int access_strength(int access);

/* ------------------------------------------------------------------ the slab class */
/* Fixed-stride VRAM slab slots per weight class (spec §5.4). Slots are interchangeable within a
 * class, so the mover's bookkeeping is an integer, not an allocator. A class is keyed on the unit
 * kind and the unit size rounded to the movement-unit alignment, so every unit of a class has
 * the class's stride and the slab never fragments.
 */
struct SlabClass {
    std::string name;            /* "expert", "layer", "model" plus the byte size */
    int64_t     stride = 0;      /* bytes per slot, aligned */
    int64_t     n_units = 0;     /* units of this class in the model */
    int64_t     n_slots = 0;     /* resident slots the planner bought in VRAM */
    int64_t     n_shadow = 0;    /* spare slots for in-flight promotions; see mover.h */
    int64_t     n_stage = 0;     /* the staging ring; see RAD_PLACE_STAGE_RING */
    bool        movable = false;
};

/* Staging ring: slab slots that belong to no unit, which a DeviceStaged unit is copied into
 * before its op runs. Two is double buffering (one slot read by the running op, one landing for
 * the next), the minimum that overlaps anything. Deeper hides more at more VRAM; if it needs
 * tuning it should become an explicit budget flag (spec §6), not a heuristic from free VRAM. */
#define RAD_PLACE_STAGE_RING 2

/* The file tier's staging pool, in pinned host slots: a unit read from the container lands in a
 * pinned buffer the mover can DMA out of without a bounce (spec §5.4). Defined here because the
 * planner charges it to --host-pool-mib and the mover must allocate exactly that; a mover-side
 * default would let the planner fill the pool first and the mover then refuse the plan. */
#define RAD_PLACE_FILE_STAGE_SLOTS 4

/* Pinned read windows for routed experts on the file tier. A layer's file-tier experts are read
 * as long sequential runs of the container: one window fills while the previous one drains to the
 * card. Four of 32 MiB keep a drive at its sequential rate (a 32 MiB pread is ~5 ms at 7 GB/s,
 * the copy out under 2 ms) for 128 MiB pinned, charged to --host-pool-mib. */
#define RAD_PLACE_FILE_WINDOWS 4
#define RAD_PLACE_FILE_WINDOW_BYTES ((int64_t)32 << 20)

/* Heat engine move budget: swaps per step per slab class. HeatParams::moves_per_step defaults to
 * it and the free slots below are sized from it, since a swap without a free slot does not happen.
 * The engine proposes once a step (Engine::tiering_dispatch) for every layer. Replaying a mixed
 * workload on Qwen3.8-Flash-Next TP1 with 40% of experts resident, the decode hit rate is 63.8% at
 * 8, 74.1% at 32, 77.1% at 64 and 77.8% at 128; an oracle held to 64 reaches 78.0%, so past 64
 * the move rate, not the ranking, bounds the hit rate. */
#define RAD_PLACE_MOVES_PER_STEP 64

/* Shadow slots: allocated slab slots holding no unit. A promotion writes into one and only then
 * publishes, so it never DMAs over a slot a kernel may still be reading (mover.h). They come out
 * of the weights pool, not leftover VRAM: every pool is budgeted (spec §6) and --gpu-headroom-mib
 * is untouchable. The expert floor charges for them (expert_vram_floor), so the KV cache never
 * gets them; the rest of the move budget is RAD_PLACE_TURNOVER_SLOTS. The quarter-of-the-slab cap
 * binds only on very small slabs. */
#define RAD_PLACE_SHADOW_SLOTS 32

/* Turnover slots: free slab slots the mover keeps out of a loan (Mover::lend_from) beyond the
 * shadow slots. A swap holds a slot for about three steps (copy behind the proposing step,
 * publish at the next step's retire, then the demoted slot's quarantine), so
 * RAD_PLACE_MOVES_PER_STEP needs three times as many free slots. At 64 a step, 32 slots in all
 * give a 65.9% hit rate against 77.9% with 192.
 *
 * They come out of the loan because a lent slot left free costs the cache nothing; charged to the
 * expert floor they would shrink the cache's largest size. When the cache reclaims memory, the
 * free slots at the loan's top go first and the mover rebuilds them by dropping its coldest lent
 * units, which keep host copies (Mover::balance). With nothing lent, only the shadow slots
 * remain. */
#define RAD_PLACE_TURNOVER_SLOTS (3 * RAD_PLACE_MOVES_PER_STEP - RAD_PLACE_SHADOW_SLOTS)

/* ------------------------------------------------------------------ per-weight result */
struct Placement {
    Tier    tier = Tier::VRAM;
    Site    site = Site::Device;
    int32_t slab_slot = -1;
    int32_t unit = -1;
    int64_t bytes = 0;
    /* Why this and not VRAM/device, as a fixed string for the report. */
    const char* reason = "";
};

/* ------------------------------------------------------------------ the plan */
class Plan {
public:
    int      rank = 0, world_size = 1;
    Strategy strategy = Strategy::Auto;
    /* Why `strategy` is the one it is, when `auto` chose it; empty when it was asked for. */
    std::string strategy_why;

    /* False under Config::deterministic: residency sets the summation order, so dynamic placement
     * breaks bitwise reproducibility (spec §17). The heat engine stays off and the mover allocates
     * no shadow slots when it is false. */
    bool     dynamic = true;

    std::vector<Placement> weight;         /* by rad_weight; index 0 is the sentinel */
    std::vector<MoveUnit>  units;
    std::vector<int32_t>   unit_of_weight; /* by rad_weight; -1 for the sentinel */
    std::vector<SlabClass> classes;

    /* Host-site layers in execution order, as runs. A host-site op leaves its activation in host
     * memory, so every run costs two link crossings and alternating per layer is expensive.
     * Contiguity is an explicit objective; `link_crossings` is the per-step cost. */
    struct HostRun { int32_t first_layer = 0, last_layer = 0; int64_t bytes = 0; };
    std::vector<HostRun> host_runs;
    int32_t link_crossings = 0;

    /* The exact prefetch schedule for staged, non-routed units, sorted by `at_op`: the op index by
     * which the unit's bytes must have landed. */
    struct Prefetch { int32_t at_op = 0; int32_t unit = 0; };
    std::vector<Prefetch> prefetch;

    struct Totals { int64_t bytes[kTierCount] = {0}; };   /* indexed by Tier */
    Totals  totals;
    /* `host_budget` is what --host-pool-mib leaves for weights after the file tier's staging pool.
     * `file_stage_reserve` is that pool, reported so the difference is visible: the unit slots,
     * plus the read windows when a routed expert can land on the file tier. */
    int64_t vram_budget = 0, host_budget = 0;
    int64_t file_stage_reserve = 0;
    int64_t file_windows = 0;          /* the read windows' part of it */
    /* --embedding-placement ram and --vision-tower-placement ram: host-pool bytes each flag's units
     * took before any strategy ran, reported apart from the pooled units. `forced_host_used` is
     * both flags' weight bytes, as the tier totals count them; the pooled figures exclude it. */
    int64_t embedding_host_bytes = 0;
    int64_t tower_host_bytes = 0;
    int64_t draft_host_bytes = 0;
    int64_t forced_host_used = 0;
    /* --expert-profile and --expert-pin-top: the units the planner pinned resident, the profiled
     * units, and the share threshold it derived from the fraction (Plan::report). */
    int64_t pinned_units = 0;
    int64_t profile_units = 0;
    float   pin_threshold = 0.0f;
    /* --ngram-placement ram: RAM for the n-gram tables the load phase copies out of the mapping.
     * Every pool still sees them on the mapped tier. `ngram_copy_card` selects pinned memory a
     * device kernel reads over the link; otherwise locked memory the host gather reads. */
    int64_t ngram_copy_bytes = 0;
    bool    ngram_copy_card = false;
    /* Routed experts on the file tier have no address of their own: a routed layer's are read into
     * one of two VRAM buffers before its first expert op and stay published until the next routed
     * layer starts (place/stager.h, FileStager). `file_layer_max` is the largest layer's share,
     * each unit rounded to RAD_ALIGN_UNIT as the stager packs them; `file_stage_vram` is both
     * buffers, taken from the weights pool before residents. Both 0 when no routed unit is on the
     * file tier. */
    int64_t file_layer_max = 0;
    int64_t file_stage_vram = 0;
    /* Slab VRAM no weight occupies: shadow slots (the cost of dynamic placement) and the staging
     * ring (the cost of offload). Reported separately from the tier totals. */
    int64_t slab_overhead = 0;

    /* What did not fit. `notes` holds whole-plan statements ("the budget is 3 GiB short");
     * `unfit` holds per-unit misfits aggregated by class and reason, so hundreds of stranded
     * experts print as one line (spec §16). */
    std::vector<std::string> notes;
    struct Unfit {
        std::string what;    /* the weight class key */
        std::string why;
        int64_t     count = 0;
        int64_t     bytes = 0;
    };
    std::vector<Unfit> unfit;
    bool fits() const { return notes.empty() && unfit.empty(); }

    /* The placement plan report (spec §16): tier and site per weight class (not per weight), the
     * pool totals, and what did not fit. */
    std::string report(const Program& p) const;
};

/* ------------------------------------------------------------------ the planner */
struct PlannerInput {
    const Program* program = nullptr;
    const Config*  config = nullptr;
    int rank = 0;
    int world_size = 1;

    /* The container's expert-popularity profile, from the imatrix's activation counts. Null is
     * fine and costs about a point of hit rate. See Planner::rank_units. */
    const RadFileProfile* profile = nullptr;
    int64_t               n_profile = 0;
    /* The weights are in a container, which the file and mapped tiers read in place. A checkpoint
     * served directly is neither grouped by movement unit nor aligned for O_DIRECT, so with this
     * false both tiers are closed (spec §4.3). */
    bool                  container = true;
    /* No card is visible and the device backend is host memory (device_is_host()), so a host
     * kernel can read the weights pool as well as the mapping. See mapped_unit(). */
    bool                  host_backend = false;
};

/* Whether a host-only unit stays in the container's mapping. On a card, always: a host kernel
 * cannot read VRAM. On the host backend the weights pool is host memory, so a unit whose stored
 * form differs from the container's (a load-time widening or relayout, or a checkpoint served
 * directly) takes a converted copy in the pool instead. Shared by the planner and the budget. */
inline bool mapped_unit(bool host_only, bool direct, bool host_backend) {
    return host_only && (direct || !host_backend);
}

class Planner {
public:
    explicit Planner(const PlannerInput& in) : in_(in) {}

    /* Returns RAD_OK, or a negative status with `out` still filled in far enough to print the
     * report. */
    int plan(Plan* out);

private:
    const PlannerInput& in_;

    /* Per-weight facts derived from declare, indexed by rad_weight. */
    std::vector<uint8_t> host_ok_;     /* every op reading it resolved a RAD_DOMAIN_HOST kernel */
    std::vector<uint8_t> device_ok_;   /* ...and a RAD_DOMAIN_DEVICE one. Cleared means the weight
                                        * can only be read on the host; see pin_mapped. */
    bool                 ssd_ = false; /* --weights-disk-tier was given */

    /* Whether every weight of the unit has a host kernel. */
    bool host_kernel(const MoveUnit& mu) const;
    void build_units(const Program& prog, Plan* p);
    void pin_mapped(const Program& prog, Plan* p);
    int  pin_embedding(const Program& prog, const Config& cfg, Plan* p) const;
    int  pin_towers(const Config& cfg, Plan* p) const;
    int  pin_draft_experts(const Program& prog, const Config& cfg, Plan* p) const;
    void pin_hot(const Config& cfg, Plan* p) const;
    void build_classes(Plan* p);
    void rank_units(Plan* p, std::vector<int32_t>* order) const;
    int  assign(Plan* p);
    int  assign_all_vram(Plan* p);
    int  assign_layer_offload(Plan* p);
    int  assign_expert_tiered(Plan* p);
    void enforce_op_site_agreement(const Program& prog, Plan* p);
    void schedule_prefetch(Plan* p);
    void finish(const Program& prog, Plan* p);
    /* Folds one stranded unit into the list aggregated by (class, reason). */
    void note_unfit(Plan* p, const MoveUnit& mu, const char* why) const;
};

/* Bytes of a declared weight as this rank holds them: the stored size once the layout pass has
 * run, the logical size before it. Shared so the mover and the report agree on budgets. */
int64_t weight_bytes(const WeightInfo& w);

/* ------------------------------------------------------------------ what the weights would cost
 *
 * One rank's declared weights, split into the fixed part and the elastic part. The VRAM budget
 * needs this before the planner runs. Static (non-expert) weights must be resident for the model
 * to run at a sensible speed, so they come off the top; only the expert plane trades against the
 * KV cache. A single combined total would let the cache demote attention projections to host.
 *
 * Values are per rank, which under tensor parallel is the shard: declare divides every dimension
 * per rank (spec §9).
 *
 * It must agree with the planner, which is why it lives here: it shares the planner's site test
 * and repeats the unit grouping and alignment of Planner::build_units and the mapped-tier rule of
 * Planner::pin_mapped. Any drift overcommits the card by the difference. */
struct WeightFootprint {
    int64_t statics = 0;   /* non-expert and device-readable -- the floor under any budget */
    int64_t experts = 0;   /* the elastic plane: what streams from host when it does not fit */
    int64_t expert_layer_max = 0;   /* the most expert bytes any one routed layer holds */
    int64_t expert_unit_max  = 0;   /* the largest expert unit's slot */
    /* A draft head's routed experts (layers at meta.n_layers and above, when the program has a
     * drafter): the part of `experts` a host-pinned --draft-expert-placement takes out of the host
     * pool's room for the trunk, and the largest such unit. */
    int64_t draft_experts    = 0;
    int64_t draft_unit_max   = 0;
    /* Upper bound on an expert-tiered plan's spare slab: per expert slab class, the shadow slots,
     * staging ring and tail pad (Planner::assign_expert_tiered). Same weight budget as residents. */
    int64_t expert_spare     = 0;
    int64_t layer_unit_max   = 0;   /* the widest per-layer unit's slot: the file tier's staging */
    bool    routed           = false;   /* a layer has experts: the file tier's read windows */
    int64_t mapped  = 0;   /* no device kernel can read them; they never enter a pool at all */
    int64_t hosted  = 0;   /* --embedding-placement ram: in the pinned host pool, off the card */
    /* The tables the placement flags move, already counted in a bucket above: `embedding` in
     * statics or hosted; `ngram` in statics when a device kernel reads it and it is placed in
     * VRAM, else in mapped; `tower` (units the architecture declared read once per request) in
     * statics. */
    int64_t embedding = 0;
    int64_t ngram     = 0;
    int64_t tower     = 0;
    /* The part of `tower` that is also `embedding`: a tower's model-level table read once a
     * request and only by embed_lookup (Gemma 4's patch-position tables). See tower_pin_bytes. */
    int64_t tower_lookup = 0;
    /* A device kernel reads the n-gram table, so it can be on the card or read over the link from
     * pinned host memory; without one only the host gather reads it, from RAM or the container. */
    bool    ngram_device = false;
    /* A host kernel gathers it, so it can stay in the container (--ngram-placement disk). */
    bool    ngram_host = false;
    int64_t n_units = 0;
};
/* `embedding` is the token embedding's resolved placement; only Ram moves it out of the statics.
 * `ngram` is the n-gram table's; only Vram puts it in the statics, and only where a device kernel
 * reads it. `host_backend` and `container` mean what they do in PlannerInput; every caller passes
 * them so the budget and the planner count alike. */
void weight_footprint(const Program& prog, WeightFootprint* out, TablePlacement embedding,
                      TablePlacement ngram, bool host_backend, bool container);

/* The bytes --vision-tower-placement ram moves off one rank's card. With the embedding in RAM,
 * pin_embedding pins the towers' lookup tables first and pin_towers skips them, so they count
 * once, with the embedding; a split that takes both `embedding` and `tower` out of the statics
 * comes up short by those tables. */
inline int64_t tower_pin_bytes(const WeightFootprint& wf, TablePlacement embedding) {
    return embedding == TablePlacement::Ram ? wf.tower - wf.tower_lookup : wf.tower;
}

/* Pinned staging the file tier takes out of --host-pool-mib before any unit is placed:
 * RAD_PLACE_FILE_STAGE_SLOTS of the widest per-layer unit, plus the read windows when a routed
 * layer can stream from the container. Shared by Planner::plan and the budget. */
int64_t file_stage_reserve(int64_t widest, bool routed);

/* Expert bytes the card must hold when `host_room` bytes of the pinned pool are free for experts:
 * what the pool cannot take, the spare slab slots, and one unit of first-fit slack on each side
 * (the resident fill and the pool fill can each leave up to one unit's room unspent). Capped at
 * all experts plus spares. A weight budget of statics plus this places every expert in VRAM or
 * the host pool. */
int64_t expert_vram_floor(const WeightFootprint& wf, int64_t host_room);

/* The same floor once --draft-expert-placement host has taken the drafter's experts out of the
 * plane the strategy places: the trunk's floor against the whole host room. The drafter's units
 * share the trunk's slab classes, so no spare is added. With no drafter it is expert_vram_floor. */
int64_t expert_vram_floor_draft_hosted(const WeightFootprint& wf, int64_t host_room);

/* Checks that the plan and this rank's weights pool used the same split: Plan::vram_budget must
 * equal the pool Pools::configure allocated, or the slab and the pool disagree silently. Returns
 * RAD_E_STATE, logging the rank and both sizes, when they differ. */
int plan_check_weights_pool(const Plan& p, int64_t pool_bytes);

/* The tables the placement flags move, identified by their readers rather than by name: a
 * model-level weight read by at least one op, and only by `embed_lookup`, is a token embedding;
 * only by `embed_lookup_q`, an n-gram table. A lookup reads M rows a step, so a tier below the
 * card is usable; a weight a GEMM also reads (an output head tied to the embedding) would stream
 * whole every step and is neither. Shared by the budget resolver and the planner. */
bool read_only_by(const Program& prog, rad_weight w, const char* op);
inline bool is_embedding_table(const Program& prog, rad_weight w) {
    return read_only_by(prog, w, "embed_lookup");
}
inline bool is_ngram_table(const Program& prog, rad_weight w) {
    return read_only_by(prog, w, "embed_lookup_q");
}

/* "blk.12.ffn_gate_exps.3.weight" -> "blk.*.ffn_gate_exps.*.weight": the report's per-class
 * grouping key (spec §16). */
std::string weight_class_key(const std::string& name);

}  /* namespace rad */
