/* registry.h -- the plugin layer: dlopen, the configured hierarchy, the op-schema registry, the
 * constraint selector, bucket tables and the selection table. spec.md §2 and §16.
 *
 * Rules enforced here and nowhere else:
 *
 *   The hierarchy is absolute. Plugins are queried in configured order and the first plugin with
 *   any matching kernel supplies it; a lower-ranked plugin never outbids a higher one, however
 *   specialised its kernel. That is what makes a user override an override (§2.1).
 *
 *   Specificity is declared, not derived. Within one plugin the highest `priority` among matching
 *   rows wins, ties by declaration order. Counting matched constraints gets it backwards:
 *   {head_dim<=1024, gqa>=1, block%16==0, causal in {0,1}} matches four, {head_dim==256, gqa==6}
 *   matches two, and the second is more specialised.
 *
 *   Schemas are fixed by first declaration. The first plugin in hierarchy order to declare an op
 *   fixes its schema; a later plugin declaring it differently is refused at load with both named.
 *   Arguments are positional, so disagreeing schemas would give the same call different meanings
 *   depending on which plugin won selection (§2.3).
 *
 *   Bucket boundaries are the constraint values: the union of the LE, GE and EQ values candidate
 *   kernels place on the ranged parameter, clamped to the declared range (§2.2).
 *
 *   A reference library is last. A library that declares itself one (RadPluginReferenceFn)
 *   matches every op it declares, so anything ranked behind it would be unreachable. An unnamed
 *   plugin sorts after every named one but ahead of a reference, so a kernel library for a new
 *   card works with no configuration (§18); naming it in the hierarchy overrides that. The rule
 *   is keyed on the declaration, never on a name, so bundled libraries load like third-party ones.
 *
 * Lifetime: a Resolved, a Band and a KernelRow point into a plugin's static tables and are valid
 * only as long as the plugin set that holds them; the last close() of a set dlcloses it. The core
 * keeps one Registry for the process (rad_tools_registry); a tool that loads two kernel sets in
 * turn must finish with the first.
 *
 * Sharing: several engines in one process (docs/MULTIMODEL.md §3.3) select from one loaded set,
 * because a plugin's .so can be loaded once per process and its rows and schemas are the same for
 * every model. What differs per engine is held per Registry object: the facts its cards report
 * (P2P), its selection table and the kernel instances it initialised. share() makes such an object
 * over the same set; its close() finis only its own instances and leaves the set to the others.
 *
 * Threading: loading is single-threaded, before the rank threads exist. select() and
 * init_instance() run on every rank thread during declare, so the selection log and instance list
 * share one mutex. It is never taken on the step path.
 */
#pragma once
#include "rad_core.h"
#include "rad_quant.h"
#include "../format/tunecache.h"

#include <deque>
#include <mutex>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rad {

inline const char* domain_name(int d) { return d == RAD_DOMAIN_HOST ? "host" : "device"; }


/* ------------------------------------------------------------------ one loaded .so */
struct Plugin {
    std::string name;          /* RadPluginInfo::name -- what the hierarchy list names */
    std::string file;          /* the .so's stem; the hierarchy list accepts this spelling too */
    std::string path;
    std::string version;
    std::string description;
    std::string build_target;

    uint32_t    kind   = 0;    /* one of RAD_PLUGIN_KERNEL, _ARCH, _QUANT */
    int         order  = 0;    /* hierarchy position; 0 is highest */
    void*       handle = nullptr;

    int n_kernels = 0;
    int n_schemas = 0;
    /* A kernel library whose fat binaries hold no code object the card runs: its device rows were
     * left out at load (loader.cpp, commit_), and only its host rows are selectable. */
    bool device_code_absent = false;
    /* A kernel library that declared itself a reference implementation (RadPluginReferenceFn). */
    bool reference = false;

    /* Architecture plugins are loaded and identified here; the build component drives them. The
     * registry only guarantees that no two plugins claim the same (architecture id, quantisation)
     * pair (§2.4): qwen35_bf16 and qwen35_fp8 both answer "qwen35" at two weight formats.
     * `arch_quant` is "" for a plugin that serves an unquantised container, including one that
     * predates the export. */
    std::string arch_id;
    std::string arch_quant;
    int  (*arch_declare)(RadBuilder*, const RadModelMeta*, const RadBuildCtx*) = nullptr;
    void (*arch_step)(RadCtx*, const RadBatch*) = nullptr;
    uint64_t (*arch_pass_key)(RadCtx*, const RadBatch*, const int32_t*) = nullptr;
    /* Optional (rad_builder.h); null means every question answers zero. Asked before declare, so
     * model facts that are inputs to declaration are still read by the plugin, not the core. */
    int  (*arch_probe)(const RadModelMeta*, RadArchProbe*) = nullptr;
    /* Optional (rad_builder.h); null means the reply format is derived from the chat template. */
    const RadChatFormat* (*arch_chat_format)(const RadModelMeta*) = nullptr;
    /* Optional (rad_builder.h); null means the model serves no speech route. */
    const RadAudioFormat* (*arch_audio_format)(const RadModelMeta*) = nullptr;
    /* Optional (rad_builder.h); null means a request's texts go into the audio generator's
     * prompt as given. */
    int64_t (*arch_text_prepare)(int, const char*, char*, int64_t) = nullptr;
    /* Required of a video generator (rad_builder.h); null for every other plugin. */
    int64_t (*arch_video_positions)(int, const RadVideoGeom*, float*, int64_t) = nullptr;
    /* Optional (rad_builder.h, rad_arch_serves_imports); absent is true. False keeps the plugin
     * out of the pick for an imported checkpoint (startup.h). */
    bool arch_serves_imports = true;
    /* Optional (rad_builder.h, rad_arch_multi_instance); absent is false: the plugin backs one
     * engine at a time (rad_engine_claim). */
    bool arch_multi_instance = false;
    /* A video generator's reference conditioning (rad_builder.h); null for one without. */
    int64_t (*arch_video_pack)(const RadVideoGeom*, const int32_t*, int32_t*, int32_t*, int32_t*,
                               int32_t*, float*, int64_t) = nullptr;
    int64_t (*arch_encode_geom)(int32_t, const int64_t*, int32_t*, int64_t) = nullptr;

    RadPluginCloseFn close = nullptr;
    RadPluginFreedFn freed = nullptr;   /* rad_plugin_freed, optional */
};

/* ------------------------------------------------------------------ one selection question */
/* What select() did, in enough detail to print §16's table and explain why a kernel was not
 * picked. `candidates` holds every row considered: not rows of the other domain, and not rows
 * behind a plugin that already supplied a match. */
struct SelectionTrace {
    struct Candidate {
        std::string plugin;
        std::string kernel;
        int         priority = 0;
        bool        ok = false;
        std::string why;        /* the constraint that refused it, or "ok" */
    };

    std::string op;
    std::string query;          /* the geometry as asked */
    int         domain = RAD_DOMAIN_DEVICE;

    std::string kernel;         /* empty on a miss */
    std::string plugin;
    int         priority = 0;
    std::string reason;         /* on a miss: the constraint(s) that refused it */

    /* A RAD_DERIVED parameter the caller did not supply: its constraints were skipped during
     * matching and the winner's RAD_C_EQ value for the key was written into the resolved
     * geometry. `known` is false when the winner has no EQ on the key (e.g. ref's
     * `block_size >= 1`); the caller must report that, not read a zero. */
    struct DerivedParam {
        std::string key;
        long long   value = 0;
        bool        known = false;
    };
    std::vector<DerivedParam> derived;

    std::vector<Candidate> candidates;
    bool other_domain_has_rows = false;   /* the op exists, just not in the domain asked for */
};

/* ------------------------------------------------------------------ the registry */
class Registry {
public:
    Registry();
    ~Registry();

    /* A registry over this one's loaded set, with its own device facts, selection table and
     * instances (see Sharing above). The set stays loaded while any registry sharing it lives.
     * Loading into a set that is shared is refused, so every sharer sees the same plugins. */
    std::unique_ptr<Registry> share() const;
    /* True while another registry shares this one's set. */
    bool shared() const { return set_.use_count() > 1; }
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;

    /* ---------------------------------------------------------- loading (§2, §18) */
    /* Scan a directory of .so files and load them in hierarchy order. `hierarchy` is an ordered
     * list of plugin names (RadPluginInfo::name, or the file as written: `libr4d`, `r4d`,
     * `libr4d.so`), so substituting a kernel means dropping an .so in and naming it ahead of the
     * default. A non-empty list is exclusive: a kernel library it does not name is closed before
     * it registers anything, and an entry that matches nothing is warned about. An empty list
     * loads every kernel library in filename order (reproducible), a reference library last.
     * Architecture and quantiser plugins are not ranked and always load.
     *
     * Always completes: every loadable plugin is loaded, every refusal goes to load_errors(), and
     * the first negative status is returned, so one run reports every failure (§3.1). */
    int load_dir(const std::string& dir, const std::vector<std::string>& hierarchy);
    /* The same over several directories (a $RADIANCE_HOME search path) as one set: the hierarchy
     * orders plugins across all of them, and a file name an earlier directory holds shadows later
     * ones. */
    int load_dirs(const std::vector<std::string>& dirs, const std::vector<std::string>& hierarchy);

    /* One plugin at a known hierarchy position (rad-info, rad-kbench); the engine uses load_dir.
     * Returns 1, with nothing registered, when rad_plugin_open answers RAD_E_UNSUPPORTED. */
    int load_plugin(const std::string& path, int order);

    /* Every kernel row's op must have a schema somewhere in the loaded set. An override plugin
     * may ship kernels for an op whose schema a later-loading plugin declares, so this runs once
     * the set is complete: load_dir calls it, a caller assembling a set by hand calls it last. */
    int check_complete();

    /* The device a kernel library's device code is checked against at load: the first card the
     * caller will run on. Default 0; the engine sets it after choosing its cards, before loading. */
    void set_code_device(int device) { set_->code_device = device; }

    /* Set by the engine when serving on the host backend, where no GPU code object runs: device
     * rows are left out at load and every op takes its host kernel. Only the engine sets it; tools
     * that select without serving (rad-convert's recipe check, rad-kbench, constraint tests) see
     * every row, as on the serving machine. */
    void set_no_device_code(bool v) { set_->no_device_code = v; }
    /* Measured facts about the ranks' devices, for constraints on "device." keys (rad_types.h).
     * Set before the first selection by whoever chose the devices; without a fact, a kernel
     * constrained on it is not selected. */
    void set_device_fact(std::string_view key, long long v) { device_facts_.set_i(key, v); }

    /* Fini this registry's instances and forget its selections. The last registry holding the
     * set also runs the plugins' close hooks, dlcloses them and forgets everything. */
    void close();

    const std::vector<Plugin>&      plugins() const { return set_->plugins; }
    const Plugin*                   plugin(std::string_view name) const;
    const std::vector<std::string>& load_errors() const { return set_->load_errors; }

    /* Every quantiser the loaded quantiser plugins offer, in load order. Names are unique (a
     * duplicate is refused at load); quantizer() is null when nothing offers the name. */
    struct QuantRow {
        const RadQuantizerInfo* info = nullptr;
        std::string             plugin;
    };
    const std::deque<QuantRow>& quantizers() const { return set_->quants; }
    const QuantRow*             quantizer(std::string_view name) const;

    /* Every kernel row, in (hierarchy order, declaration order). rad-info prints this. */
    const std::deque<KernelRow>&    rows() const { return set_->rows; }
    /* The rows implementing one op, in the order select() considers them. Empty if none. */
    const std::vector<const KernelRow*>* rows_for(std::string_view op) const;

    /* ---------------------------------------------------------- op schemas (§2.3) */
    const RadOpSchema* schema(std::string_view op) const;
    const Plugin*      schema_owner(std::string_view op) const;
    std::vector<std::string> op_names() const;      /* sorted; the vocabulary this set implements */

    /* Check a declared parameter set against the op's schema; a misspelled op fails with the
     * near-miss named. Returns RAD_OK, RAD_E_NOSCHEMA (no such op) or RAD_E_SCHEMA, with `err`
     * written for an architecture-plugin author. */
    int validate(std::string_view op, const RadParam* p, int n_p, std::string* err) const;

    /* The closest known op name within a small edit distance, or "". Also used by the builder. */
    std::string near_miss(std::string_view op) const;

    /* ---------------------------------------------------------- the tuning cache (§15) */
    /* Load $RADIANCE_HOME/tune/<machine>.tune (written by rad-tune). resolve() then moves a shape
     * off each axis's declared default wherever the file has a row for it. A missing file is not
     * an error; every shape stays on its default. The core owns the cache and the policy
     * (tunecache.h, applied in select.cpp). Without this call the file is silently ignored.
     *
     * Returns the number of rows loaded, or a negative status. */
    int  load_tune(const std::string& radiance_home, const std::string& machine = std::string());
    const TuneCache& tune() const { return set_->tune; }

    /* ---------------------------------------------------------- selection (§2.1) */
    /* The first plugin in hierarchy order with any matching row supplies the kernel; within that
     * plugin the highest declared priority wins, ties on declaration order. Null on a miss, with
     * the refusing constraint in `why->reason`. Every call is recorded in the selection table.
     *
     * RAD_DERIVED parameters (rad_abi.h) are supplied by the kernel, so a constraint on one is
     * skipped when the caller did not name it, and the winner's RAD_C_EQ value comes back in
     * `why->derived` (resolve() writes it into the geometry). This breaks the §7.2 cycle: the
     * block manager reads the block size off the resolved attention kernel, and a constraint whose
     * key the geometry lacks does not hold. A caller that supplies the key pins it and matches
     * normally, so a forced block size is refused by name rather than substituted.
     *
     * `record` is false for questions the engine did not ask (the graph dump replaying bands), so
     * the selection table records only what this deployment resolved.
     *
     * `accept`, when given, is asked of every row whose constraints hold; a non-empty answer drops
     * the row with that reason. This is how a weight's encoding takes part in selection: a kernel
     * whose layout hook rejects the operand's encoding (RAD_E_DTYPE) is skipped and the next one
     * chosen, so a bf16 lm_head goes to the bf16 GEMM rather than being refused by the fp8 one. */
    using RowFilter = std::function<std::string(const KernelRow&, const Geometry&)>;
    const KernelRow* select(std::string_view op, const Geometry& g, int domain,
                            SelectionTrace* why = nullptr, bool record = true,
                            const RowFilter* accept = nullptr);

    /* The row select() would choose if `plugin` were the whole hierarchy: the same constraint
     * match, derived-key skip, device facts and filter, restricted to one plugin and never
     * recorded. rad-kbench asks each library this way so a replay runs the row the engine would
     * run with that library alone. `why->derived` carries the winner's derived values. */
    const KernelRow* select_in(std::string_view plugin, std::string_view op, const Geometry& g,
                               int domain, const RowFilter* accept = nullptr,
                               SelectionTrace* why = nullptr) const;

    /* Selection plus per-instance bookkeeping: the tuned axes and their source, and the frozen
     * geometry (the query plus the winner's derived keys, so rad_kv_block_size() and RadArgs.p
     * both see them). Does not call init(); see init_instance(). */
    Resolved resolve(std::string_view op, const Geometry& g, int domain, SelectionTrace* why = nullptr,
                     const RowFilter* accept = nullptr);

    /* ---------------------------------------------------------- bucket tables (§2.2) */
    /* Resolve a ranged parameter into a bucket table: one resolution per band per domain, built
     * at declare and indexed by the actual value at issue. Each band is resolved at its upper
     * bound, the band's worst case and so the only value safe to size and select on.
     *
     * An empty `ranged_key` returns one band with hi = INT64_MAX, which the issue path's band scan
     * always finds.
     *
     * Both domains are resolved so the execution site can change at runtime without re-resolving.
     * A band with no device kernel is fatal; one with no host kernel only removes host-site
     * execution. Neither aborts here: the caller reports misses once declare completes. */
    std::vector<Band> build_bands(std::string_view op, std::string_view ranged_key,
                                  int64_t lo, int64_t hi, const Geometry& base,
                                  const RowFilter* accept = nullptr);

    /* ---------------------------------------------------------- instances (§2.1) */
    /* Calls init() once per resolved instance (one op, band and domain) with the frozen geometry,
     * rank and world size, off the hot path: e.g. libr4d's all-reduce sets up peer signal buffers
     * and IPC handles here. A negative return aborts startup; the caller names the kernel. */
    int  init_instance(Resolved& r, int rank, int world_size);
    /* The tensor-parallel group this registry's engine is (its first instance slot,
     * rad_engine_claim). Set, init_instance hands it to every kernel's init as the `tp_group`
     * parameter, so a collective's ranks rendezvous only with their own engine's; unset (-1), as
     * for the tools, no parameter is added. */
    void set_tp_group(long long g) { tp_group_ = g; }
    void fini_instances();
    size_t n_instances() const;

    /* ---------------------------------------------------------- the selection table (§16) */
    /* Every distinct question the selector was asked, in first-asked order, with its answer and,
     * for a miss, the refusing constraint. Repeats are folded into a count, which shows how much
     * of the model resolved to each kernel. */
    std::string selection_table() const;
    size_t      n_selections() const;
    void        clear_selections();

private:
    struct Staged {
        void*                handle = nullptr;
        std::string          path, file, name;
        const RadPluginInfo* info = nullptr;
        /* The device backend's fat-binary registrations before and after this plugin's dlopen:
         * [code_from, code_to) is the device code it brought in. */
        size_t               code_from = 0, code_to = 0;
        bool                 reference = false;   /* RadPluginReferenceFn answered nonzero */
    };

    struct SchemaEntry {
        const RadOpSchema* schema = nullptr;
        std::string        plugin;      /* who fixed it, for the conflict message */
    };

    struct SelectionRecord {
        std::string op, query, key, kernel, plugin, reason;
        std::string derived;      /* "block_size:=16", kept apart from the query because it was
                                   * not part of the question asked */
        int         domain = 0;
        int         priority = 0;
        long long   count = 1;
    };

    /* What loading produced: shared by every registry made with share(). */
    struct Set {
        int                      code_device = 0;
        bool                     no_device_code = false;
        TuneCache                tune;
        std::vector<Plugin>      plugins;
        std::deque<KernelRow>    rows;          /* deque: KernelRow* handed out must stay valid */
        std::deque<QuantRow>     quants;
        std::unordered_map<std::string, std::vector<const KernelRow*>> by_op;
        std::unordered_map<std::string, SchemaEntry> schemas;
        std::vector<std::string> load_errors;
    };
    explicit Registry(std::shared_ptr<Set> set) : set_(std::move(set)) {}
    std::shared_ptr<Set> set_;

    Geometry device_facts_;
    long long tp_group_ = -1;

    int  stage_(const std::string& path, Staged* out);
    int  commit_(Staged& s, int order);
    void unstage_(Staged& s);
    void index_rows_();
    void record_(const std::string& op, int domain, const Geometry& g, const SelectionTrace& t);
    void err_(const char* fmt, ...) __attribute__((format(printf, 2, 3)));

    mutable std::mutex           mu_;
    std::vector<SelectionRecord> log_;
    std::unordered_map<std::string, size_t> log_at_;   /* SelectionRecord::key -> index in log_ */
    std::vector<std::pair<const RadKernelInfo*, void*>> instances_;
    std::deque<Geometry> init_geoms_;   /* deque: init may keep pointers into an entry */
};

/* The value a kernel's own constraints place on one key, e.g. the paged block size the resolved
 * attention kernel is compiled for; the block manager sizes itself from it (§7.2). Returns false
 * when the kernel has no constraint on that key with that operator; that is not a default, and
 * the caller decides. */
bool kernel_constraint(const KernelRow& r, std::string_view key, int c_op, long long* out);

/* "an integer", "a string", ... for a RAD_P_* parameter kind, for diagnostics. */
const char* param_kind_name(int kind);

/* Why kernel `ki` cannot read weight operand `operand` (named `name`) stored as encoding `enc` with
 * planes `sel` described by `planes` (dtype and extents; data unused), or "" when it can. A row
 * whose layout hook answers RAD_E_DTYPE does not read the encoding, and a device row with no hook
 * reads a single plane as stored. A hookless host row reads canonically. The builder's band
 * resolution and rad-kbench's replay both ask this, so they choose the same row. */
std::string weight_refusal(const RadKernelInfo* ki, const Geometry& g, int operand,
                           const RadEncoding& enc, const int* sel, const RadTensor* planes,
                           int n_sel, std::string_view name);

}  /* namespace rad */
