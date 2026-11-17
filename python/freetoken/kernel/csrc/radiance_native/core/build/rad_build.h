/* rad_build.h -- the declare phase (spec §3.1): receives everything the architecture plugin will
 * ever do.
 *
 *   Declare always completes. A missing kernel does not stop the walk: every op, band and domain
 *   that resolved to nothing is collected with the constraint that refused it and reported at the
 *   end, which shows whether a plugin or a model is missing.
 *
 *   Nothing is decided twice. After declare the model is static (every op, its kernel, the
 *   weights it touches, the order). The bucket table, weight manifest, buffer plan and KV block
 *   sizes are built once here, so the step path indexes tables.
 *
 * The kernel hierarchy, op schemas, constraint matching and bucket boundaries belong to
 * core/plugin/. The Builder calls Registry::validate and Registry::build_bands rather than
 * reimplementing them, so selector and builder cannot drift apart.
 *
 * Lifetime: every KernelRow, Resolved and RadOpSchema in the Program points into a plugin's static
 * tables, which the Registry dlclose()s. Neither the Builder nor the Program may outlive the
 * Registry.
 */
#pragma once
#include "plugin/registry.h"

#include <cstdio>
#include <functional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

/* The plugin's handle. rad_builder.h only forward-declares it; as a base of rad::Builder, the C
 * shims reach the Builder with a static_cast. */
struct RadBuilder { };

namespace rad {

/* Process-lifetime storage for declared names. RadNameMap and RadKernelInfo hold raw
 * `const char*` and the Program outlives the Builder, so the strings are interned: a few thousand,
 * deduplicated, never freed. Owning them in RadNameMap would be an ABI change. */
const char* intern(std::string_view s);

/* Names for the enums the graph dump prints; `bld_` prefixed to keep them apart from the
 * planner's tier_name and site_name in rad_core.h. */
const char* bld_domain_name(int domain);
const char* bld_access_name(int access);
const char* bld_buf_kind_name(int kind);
const char* bld_kv_kind_name(int kind);

/* ------------------------------------------------------------------ the buffer plan */
/* Packing results for the --debug-graph report. `sum_bytes` (no sharing) and `peak_live` (a
 * perfect packer) bracket the arena and show what the greedy pass left unused. */
struct BufPlanReport {
    int64_t transient_bytes = 0;   /* the packed, shared region */
    int64_t persist_bytes   = 0;   /* PERSIST + DERIVED: their own non-overlapping region */
    int64_t host_bytes      = 0;   /* domain=HOST buffers; a separate address space (see below) */
    int64_t peak_live       = 0;   /* max over op index of the bytes live at it */
    int64_t sum_bytes       = 0;   /* every transient, unshared */
    int     n_unknown_live  = 0;   /* transients whose liveness nothing declared */
};

/* Liveness must already be in BufferInfo::first_def / last_use; a transient with first_def < 0 is
 * treated as live for the whole program, which is sound.
 *
 * Assigns BufferInfo::arena_offset for every buffer and sets Program::arena_bytes. Returns RAD_OK,
 * or RAD_E_INVAL if a buffer could not be sized (a plugin-private dtype the core cannot measure). */
int plan_buffers(Program& p, BufPlanReport* rep = nullptr);

/* The arena layout for steps of at most `rows` tokens: the real plan's buffers, the fixed region
 * unchanged, and the transients repacked, so the arena past `end` is unused by such a step and
 * can be lent. `offset` and `decl` are indexed by buffer handle. */
struct ArenaLevel {
    int64_t rows = 0;
    int64_t end = 0;
    std::vector<int64_t>    offset;
    std::vector<RadBufDecl> decl;
};
/* Lay out `probe`'s buffers (a sizing declare at `rows`) over `full`'s. Refused, with the reason
 * in `why`, unless both declares describe the same buffers with only fewer rows; otherwise they
 * cannot share the real plan's fixed region and kernels. */
int plan_level(const Program& full, const Program& probe, int64_t rows, ArenaLevel* out,
               std::string* why);

/* Bytes a declared buffer occupies. 0 means the core cannot size it: a plugin-private dtype, legal
 * for a weight (the kernel's layout hook sizes it) but not for an activation. */
int64_t buffer_bytes(const RadBufDecl& d);

/* ------------------------------------------------------------------ the model behind a declare */
/* What the model holds for a logical weight: its encoding and logical extents. rad_weight_encoding
 * answers with it and every declaration is checked against it (spec §2.4). The engine answers from
 * the container or checkpoint it loads, rad-convert from its recipe. `prog` is the declare so far,
 * whose name map is used to search a checkpoint. RAD_OK, or RAD_E_NOTFOUND for a weight the model
 * does not have. */
struct WeightSource {
    RadEncoding enc{};
    uint32_t    rank = 0;
    int64_t     shape[RAD_MAX_RANK] = {0};
};
using WeightSourceFn = std::function<int(const std::string& name, const Program& prog,
                                         WeightSource* out)>;

/* ------------------------------------------------------------------ the builder */
class Builder : public RadBuilder {
public:
    /* The Registry must outlive the Builder and its Program (see the file header). */
    Builder(Registry& reg, const RadModelMeta& meta, const RadBuildCtx& ctx);

    /* Where rad_weight_encoding's answers come from. Unset (tests, synthetic tools), every source
     * is RAD_E_NOTFOUND and a weight is read as plain in its declared dtype. */
    void set_sources(WeightSourceFn f) { sources_ = std::move(f); }

    /* The C surface, one method per rad_builder.h entry point. */
    rad_weight  decl_weight  (const char* name, const RadWeightDecl* d);
    int         weight_encoding(const char* source, RadEncoding* out, int64_t* shape,
                                uint32_t* rank);
    rad_buf     decl_buffer  (const char* name, const RadBufDecl* d);
    rad_kvgroup decl_kv_group(const char* name, const RadKVGroupDecl* d);
    int64_t     kv_block_size(rad_kvgroup g);
    int64_t     kv_layer_bytes(rad_kvgroup g);
    int         bind_layer_kv(int layer, rad_kvgroup g);
    rad_op      decl_op      (const char* op, const RadParam* p, int n_p,
                              const rad_weight* w, int n_w);
    int         op_resolved  (rad_op h) const;
    int         op_declared  (const char* op) const;
    int         decl_name_map(const RadNameMap* m);
    int         declare_logits(rad_buf b);
    int         declare_scores(rad_buf hidden, rad_weight table);
    int         declare_pooled(rad_buf b);
    int         declare_plugin_words(int64_t max_words);
    int         declare_drafter(const RadDrafterDecl* d);
    int         drafter_shared_kv(int pos_offset);
    int         lora_root(const char* native, const char* spellings);
    int         lora_alias(const char* native, const char* from, const char* to, int64_t from_lo,
                           int64_t from_hi, int64_t to_lo);
    int         drafter_short_block(int64_t block);
    int         drafter_tree(rad_buf parents);
    int         route_table(rad_buf table);
    int         declare_encoder(const RadEncoderDecl* d);
    int         declare_generator(const RadGeneratorDecl* d);
    int         declare_audio_generator(const RadAudioGenDecl* d);
    int         declare_video_generator(const RadVideoGenDecl* d);
    int         declare_shape_generator(const RadShapeGenDecl* d);
    void        note_v       (const char* f, va_list ap);

    /* An op's buffer operands (rad_op_reads/rad_op_writes in rad_builder.h), which give the
     * buffer planner activation liveness. A transient no op names is live for the whole
     * program; the --debug-graph buffer report counts those. */
    int op_reads (rad_op h, const rad_buf* b, int n);
    int op_writes(rad_op h, const rad_buf* b, int n);
    int buf_concurrent(rad_buf b);
    int lane1_window(bool end);
    int weight_shard_span(rad_weight w, int64_t lo, int64_t hi);
    int weight_shard_span_parts(rad_weight w, const int64_t* lo, const int64_t* hi,
                                const int64_t* total, int n);
    int kv_group_zero(rad_kvgroup g);
    int kv_group_dense_read(rad_kvgroup g);
    int kv_group_replicated(rad_kvgroup g);

    /* Finalise: weight first/last use, KV group geometry, buffer liveness and packing. Returns
     * RAD_OK, or the first structural error class. Resolution misses are not errors here: an
     * unresolved op returns a null handle and the plugin should offer a fallback (spec §2.3); it
     * becomes an error only if the run phase issues it. */
    int finish();

    /* A sizing declare (RadBuildCtx::shape_probe): each op takes the bands `ref` resolved at the
     * same position, and finish() initialises nothing. The plugin sees the real declare's answers
     * (which ops resolved, each group's block size), so it declares the same graph and nothing is
     * selected or set up twice. An op that differs from `ref`'s at that position is an error: a
     * different graph cannot be laid over the real buffers. */
    void set_reference(const Program* ref) { ref_ = ref; }

    /* Off: finish() selects every kernel and initialises none, for callers that launch nothing.
     * rad-convert needs only the kernels' layouts; a kernel's init depends on the running process
     * (a ring, a peer's buffers), and libavx's n-gram gather refuses a process that cannot create
     * an io_uring ring, which must not stop a sandboxed conversion. */
    void set_instances(bool on) { instances_ = on; }

    /* Runtime LoRA (core/lora/): the HF module leaves an adapter may target. Set before the plugin
     * declares, so a fused GEMM over a targeted weight (the gated gate_up fold), which leaves no
     * output for the delta, is declined as it is declared and the plugin emits the unfused chain it
     * has for a library without the fused kernel. Null: nothing is declined. */
    void set_lora_targets(const std::set<std::string>* t) { lora_targets_ = t; }
    /* The read and write sets an op declared (rad_op_reads / rad_op_writes): the LoRA declare reads
     * the dtypes of a targeted GEMM's activation and output from them. */
    const std::vector<rad_buf>& op_read_set (rad_op h) const { return op_bufs_[h].reads; }
    const std::vector<rad_buf>& op_write_set(rad_op h) const { return op_bufs_[h].writes; }

    Program&       program()       { return prog_; }
    const Program& program() const { return prog_; }

    /* Structural failures (a bad schema, two ranged parameters, a duplicate name), all of them,
     * like the miss list. */
    const std::vector<std::string>& errors() const { return errors_; }
    int  status() const { return status_; }
    const BufPlanReport& buf_report() const { return bufrep_; }

private:
    /* Record a structural error and its class. Returns 0 so a decl_ function can
     * `return err(...)` and yield a null handle. */
    uint32_t err(int code, const char* f, ...) __attribute__((format(printf, 3, 4)));
    std::string scoped(const char* name) const;

    void resolve_bands(OpInfo& oi);
    void record_tune_request(const Resolved& r);
    /* init() for every surviving instance, after the whole graph has resolved, so a failing init
     * cannot abort declare before the miss report is assembled. */
    void finish_instances();
    void fill_tune_instances();
    int32_t find_kv_consumer(rad_kvgroup g);
    void finish_layouts();
    void finish_kv_groups();
    void finish_weight_uses();
    void finish_buffer_liveness();

    Registry& reg_;
    Program prog_;
    const Program* ref_ = nullptr;
    bool instances_ = true;

    std::vector<std::string> errors_;
    int  status_ = RAD_OK;
    bool finished_ = false;

    std::string scope_;

    /* Parallel to prog_.ops: the buffers each op declared it reads and writes. */
    struct OpBufUse { std::vector<rad_buf> reads, writes; };
    std::vector<OpBufUse> op_bufs_;
    const std::set<std::string>* lora_targets_ = nullptr;

    /* Buffers touched from the second lane (rad_buf_concurrent). Their declared op range does not
     * bound when their bytes are in use, so they are live for the whole program. */
    std::set<rad_buf> concurrent_bufs_;
    /* The second lane's window (rad_lane1_window_begin), first and last op handle; -1 when none. */
    int32_t lane1_w0_ = -1, lane1_w1_ = -1;

    /* Parallel to prog_.kv_groups: where in the op list the group was declared, and the op found
     * to consume it. Declaration order is the only link between an op and a group; see
     * find_kv_consumer(). */
    std::vector<int32_t> kv_decl_at_;
    std::vector<int32_t> kv_consumer_;

    /* Declared name -> index of its first prog_.name_map entry, so a repeated declaration is
     * checked in constant time; a routed model declares tens of thousands of weights. */
    std::unordered_map<std::string, size_t> name_map_first_;

    /* Declared weight name -> handle, for the same reason (Flash-Next declares about 100K weights
     * per rank). */
    std::unordered_map<std::string, rad_weight> weight_by_name_;

    /* tune_key() -> index in prog_.tune_requests: one request per kernel instantiation. */
    std::unordered_map<std::string, size_t> tune_index_;

    /* The model's answers, cached per source: a routed model declares each expert's codes and
     * scales as two weights over one source, and a checkpoint lookup walks the name map. */
    WeightSourceFn sources_;
    std::unordered_map<std::string, std::pair<int, WeightSource>> source_cache_;
    int lookup_source(const std::string& name, WeightSource* out);

    BufPlanReport bufrep_;
};

/* ------------------------------------------------------------------ --debug-graph (spec §16) */
/* The entire declared graph: every op in declared order with its bucket table (one line per band:
 * the resolving kernel, variant and plugin, the matched constraints, the weights touched and the
 * buffer written); then the weight manifest, buffer plan, KV groups, tuning requests and miss
 * report. The bucket-table block is exactly the format in spec §2.2. */
std::string dump_graph_str(const Program& p, const std::vector<std::string>* errors = nullptr);

/* The same walk as JSON, for the server's Model view, grouped by (op, band table): an op declared
 * once per layer with one bucket table is one row with a count. */
/* `reg` is the registry the program was selected through (an engine's share), null for the process
 * registry; the dump replays selections against its device facts. */
std::string dump_graph_json(const Program& p, Registry* reg = nullptr);

}  /* namespace rad */
