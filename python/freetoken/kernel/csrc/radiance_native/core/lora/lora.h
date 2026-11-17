/* lora.h -- runtime LoRA: which declared GEMMs an adapter lands in, and the bytes of one adapter for
 * one rank. Two host-only steps that read nothing but the declared Program:
 *
 *   build_sites()   after declare, every LoRA-capable trunk GEMM whose weight a target names becomes
 *                   a Site: the weight's per-rank geometry (K slice, output width, where each PEFT
 *                   module's rows land in y), read from what the plugin recorded on the weight
 *                   (shard mode, row parts, shard and part spans, name-map sources), never from a
 *                   plugin's own split rule. One Site per GEMM op; the Sites of one declared weight
 *                   share one pool.
 *   load_adapter()  a PEFT directory -> one PackedSite per pool, the bytes of one slot.
 *
 * Nothing here touches a device; the caller declares the pools and copies.
 *
 * Targets are HF module leaf names ("q_proj", "gate_proj", "in_proj_qkv", ...). A declared weight is
 * targeted when one of its name-map sources is `<prefix>.<leaf>.weight` with `leaf` in the set and
 * the weight is in the trunk; weight_targeted() is the one predicate everything here uses.
 */
#pragma once
#include "../rad_core.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace rad {
struct StTensor;
namespace lora {

/* HF module leaf names, ordered so every caller builds the same set. Empty targets nothing. */
using TargetSet = std::set<std::string>;

/* One contiguous run of one PEFT module's rows inside this site's y, on this rank. A module may
 * need several: in_proj_qkv covers three row parts of the delta net's fused ssm_inz and a rank
 * holds a slice of each, so its B rows there are three runs at three y columns. Every part of one
 * module has the same t_off. */
struct SitePart {
    std::string module;            /* HF module path without ".weight", as the name map spells it */
    int64_t src_lo = 0, src_hi = 0;/* rows [lo,hi) of the module's full output (= rows of lora_B) */
    int64_t y_col = 0;             /* column of this site's per-rank y where row src_lo lands */
    int64_t t_off = 0;             /* first t column of the module's block = module slot * max_rank */
    int32_t row_part = 0;          /* the weight's row part this run lies in (0: a single part) */
};

/* One name-map source of the site's weight, in concatenation order, targeted or not: an untargeted
 * source still occupies rows of the fused weight. */
struct SiteSource {
    std::string module;            /* source without ".weight" */
    bool        targeted = false;
    int32_t     slot = -1;         /* module slot in the site (t_off = slot * max_rank); -1 untargeted */
    /* The run of row parts [part_lo, part_hi) the source occupies, or -1/-1 when the declared
     * facts do not decide it (two sources over four parts) and the adapter's B rows must. */
    int32_t     part_lo = -1, part_hi = -1;
};

struct Site {
    uint32_t gemm = 0;                          /* the gemm op handle */
    std::string weight;                         /* declared weight name */
    int     shard = 0;                          /* RAD_SHARD_ROW / COL / NONE of the base weight */
    int64_t K_r = 0, k_lo = 0, K_full = 0;      /* x columns on this rank; COL: A columns
                                                 * [k_lo, k_lo + K_r). K_full 0 = not recorded on
                                                 * the weight (a COL span): taken from lora_A. */
    int64_t N_r = 0;                            /* per-rank y width the gemm writes */
    int64_t R = 0;                              /* capacity: n_modules * max_rank */
    std::vector<SitePart> parts;                /* empty while !parts_resolved */
    int32_t     pool = -1;           /* index of the weight in build_sites' per-weight pool list */
    std::string op;                  /* the gemm's op name */
    int32_t     n_modules = 0;       /* targeted sources */
    int32_t     max_rank = 0;
    bool        parts_resolved = true;   /* false: the fused layout is decided by B rows at load */
    std::vector<SiteSource> sources;
    /* Per row part g of the weight (P = max(1, n_row_parts) entries): the part's rows in the
     * container (0 = not recorded: an equal-part single span), this rank's rows [lo, hi) of it in
     * the part's own units, and the y column where row `lo` lands. */
    std::vector<int64_t> part_total, part_lo, part_hi, part_y;
};

/* One adapter, one POOL (= one declared weight), one rank -- exactly the bytes of one slot. */
struct PackedSite {
    std::vector<uint16_t> a;       /* [R, K_r] bf16, zero rows past the module's rank in each block */
    std::vector<uint16_t> b;       /* [N_r, R] bf16, zero outside the module's rows / rank block */
    std::vector<float>    scale;   /* [N_r]: the module's alpha/r (alpha/sqrt(r) rsLoRA) on its rows */
    int32_t pool = -1;
    bool    empty = true;          /* no module of this adapter lands in this pool */
    /* Packed compact (pack_modules): the ranks the modules take, from j = 0; R otherwise. */
    int64_t r_used = 0;
};

/* What an adapter is, independent of the name it is served under. */
struct AdapterInfo {
    struct Module {
        std::string name;          /* adapter's module path (base_model.model. stripped) */
        std::string base;          /* the base's spelling it matched on this rank ("" = absent here) */
        int64_t     rank = 0;      /* from the tensors (== rank_pattern / r) */
        double      alpha = 0.0;   /* alpha_pattern / lora_alpha */
        float       scale = 0.0f;  /* alpha / rank, or alpha / sqrt(rank) with rsLoRA */
        int32_t     pool = -1;     /* pool it was packed into on this rank, -1 = absent here */
    };
    std::string dir;
    std::string base_model;        /* base_model_name_or_path */
    std::string peft_type;         /* "LORA" */
    int64_t     r = 0;             /* config r */
    double      alpha = 0.0;       /* config lora_alpha */
    bool        rslora = false;
    std::vector<std::string> target_modules;                    /* config, as written */
    std::vector<std::pair<std::string, int64_t>> rank_pattern;  /* config order */
    std::vector<std::pair<std::string, double>>  alpha_pattern;
    std::vector<Module>      modules;     /* sorted by name */
    std::set<std::string>    leaves;      /* leaf names present in the tensors (targets present) */
    std::vector<std::string> absent;      /* modules no site on THIS rank took (see check_placed) */
    int64_t                  max_module_rank = 0;
};

/* A targeted weight read by an op that is not a LoRA site. The caller refuses at startup. */
struct UnsupportedUse {
    std::string weight;            /* declared weight */
    std::string module;            /* the targeted source module(s), comma separated */
    std::string op;                /* the op that reads it */
    uint32_t    op_index = 0;
    std::string why;
};

/* ---------------------------------------------------------------- the one predicate
 * Is declared weight `w` a LoRA target? Works on a PARTIALLY declared Program (the builder calls
 * it from decl_op): it reads only the weight's own declaration and the name-map entry (or CONCAT
 * pieces) whose `declared` is the weight's source -- false when that does not exist yet.
 *
 * SCOPE RULE (never targets):
 *   - access RAD_ACCESS_VOCAB (embedding, lm_head) or RAD_ACCESS_PER_REQUEST (encoder towers);
 *   - a layer group at or above meta.n_layers (the drafter's / MTP head's layers, the rule
 *     KVGroupInfo::drafter uses) when meta.n_layers > 0;
 *   - a declared name in a non-trunk builder scope: first component draft / vision / audio / mtp;
 *   - a source whose first component is `mtp` or `draft`, or that passes through a vision/audio
 *     tower component (visual, vision_tower, vision_model, audio_tower, audio_model, embed_vision,
 *     embed_audio, multi_modal_projector, mm_projector), or whose leaf is embed_tokens / lm_head.
 * A routed expert IS targeted when its leaf is (so its moe op is reported, never skipped). */
bool weight_targeted(const Program& prog, rad_weight w, const TargetSet& targets);

/* Is this op a LoRA-capable GEMM over weight handle `w` (as its `b` operand)? `why` says why not.
 * gemm_nt; gemm_nt_bias without `act`; gemm_nt_q without `pb1`/`pb2`, with K == the weight's K,
 * over a weight whose known encoding has no `transform`; gemm_nt_q_bias (same encoding rule). */
bool op_is_site(const Program& prog, const OpInfo& op, rad_weight* b_out, std::string* why);

/* Every op that reads a targeted weight (any view of it) and is not a capable site over it. */
int find_unsupported_uses(const Program& prog, const TargetSet& targets,
                          std::vector<UnsupportedUse>* out);

/* The Sites of one rank's Program. `pool_weights` (optional) receives the per-weight pool list:
 * Site::pool indexes it, and its order is the order the pools' first sites appear in. RAD_OK or a
 * negative code with `err` naming the weight. Pool handles are left 0. */
int build_sites(const Program& prog, const TargetSet& targets, int max_rank,
                std::vector<Site>* out, std::string* err,
                std::vector<std::string>* pool_weights = nullptr);

/* One adapter for one rank: `packed` gets one PackedSite per pool (index = Site::pool), in the
 * slot layout of the pool's sites. Refusals return a negative code and name module, expected and
 * found. A module with no site on THIS rank is not refused here -- an attention-zero rank holds no
 * q/k/v/o -- it is listed in info->absent; check_placed() refuses one no rank took. */
int load_adapter(const std::string& dir, const std::vector<Site>& sites, int max_rank,
                 AdapterInfo* info, std::vector<PackedSite>* packed, std::string* err);

/* One module's factors as the packer reads them: lora_A [rank, K] and lora_B [rows, rank] in any
 * dtype a safetensors tensor carries (BF16, F16, F32), and the scale its rows run at. `name` is the
 * module path the base's name map spells (without ".weight"). */
struct AdModule {
    std::string     name;
    const StTensor* A = nullptr;
    const StTensor* B = nullptr;
    int64_t         rank = 0;
    double          alpha = 0.0;
    float           scale = 0.0f;
};

/* load_adapter's second half, for any source of factors: `mods` (by name) packed into one
 * PackedSite per pool of `sites`, in the slot layout of the pool's sites. Fills info->modules and
 * info->absent. `compact`: the modules of a pool take consecutive ranks from j = 0, in source
 * order, instead of a block of max_rank each, so a pass may run only the first r_used of them. */
int pack_modules(const std::vector<Site>& sites, int max_rank,
                 const std::map<std::string, AdModule>& mods, AdapterInfo* info,
                 std::vector<PackedSite>* packed, std::string* err, bool compact = false);

/* After load_adapter on every rank: refuse a module that no rank placed. With `prog`/`targets`
 * (any one rank's) the message says why it is not a site (base lacks it, scope, target list, op). */
int check_placed(const std::vector<AdapterInfo>& per_rank, std::string* err,
                 const Program* prog = nullptr, const TargetSet* targets = nullptr);

/* Why a module path is not a site on this Program, in one sentence ("" when it is one). */
std::string explain_module(const Program& prog, const TargetSet& targets, const std::string& module);

}  /* namespace lora */
}  /* namespace rad */
