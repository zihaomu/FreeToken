// Modified for FreeToken: native integration, diagnostics and Qwen FP8 KV support.
/* engine.h -- the startup sequence and the step loop: what happens, in order, between a command
 * line and a served token. Kept thin; each step is one component doing its own job.
 *
 * Startup, per spec §1:
 *
 *   1. Load kernel plugins from $RADIANCE_HOME/kernels/, in the configured hierarchy order.
 *   2. Load the model's metadata only: architecture id, dimensions, quantisation, vocab.
 *   3. Look up the architecture id in $RADIANCE_HOME/architectures/. No match, no model.
 *   4. Declare. The architecture plugin enumerates every weight, buffer and op it will ever
 *      issue, and the core resolves each op against the kernel hierarchy. This phase always
 *      completes and reports every resolution failure, not just the first.
 *   5. Plan. The placement planner assigns every weight a tier and an execution site against
 *      explicit pool budgets.
 *   6. Load. Weights are mapped and staged according to the plan.
 *   7. Serve.
 *
 * After step 4 the whole model is known statically: every op, its kernel, the weights it touches,
 * and the order. The rest of the engine relies on that.
 */
#pragma once
#include "rad_core.h"
#include "rad_spechub.h"   /* RadMtpCoupleDraw/State: the coupled MTP wire rows */
#include "cvec.h"
#include <map>
#include <set>
#include "build/startup.h"
#include "diag/diag.h"
#include "mem/hostarena.h"
#include "runtime/ctx.h"   /* KVStageRun */
#include "sched/mem_iface.h"
#include "runtime/batch_state.h"
#include "runtime/storage_view.h"
#include "sample/vocab_view.h"
#include "runtime/rank_barrier.h"
#include "text/tokenizer.h"
#include "format/radfile.h"
#include "format/table_copy.h"
#include "format/table_share.h"
#include "lora/adapter.h"
#include "lora/slots.h"
#include "fleet/broker.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace rad {

class ExternalStepControl;
struct ExternalPermit;

/* While on, the sweeps' pinned staging is kept for the next sweep, for a process that restores
 * models again and again; off frees what is kept. */
void sweep_buffers_keep(bool on);

class Kld;
class StallWatch;
namespace gen { class Driver; class AudioDriver; class VideoDriver; class ShapeDriver; struct IPassDriver; }
namespace server { class Sink; class Front; class Server; }
struct TierPump;
struct TierJob;
struct StepSlot;
class ExpertStats;   /* place/xstats.h */
class XsTally;

struct VramFacts;   /* mem/vram_budget.h */
struct WeightFootprint;   /* place/planner.h */

class DeviceRuntime {
public:
    explicit DeviceRuntime(Config cfg);
    DeviceRuntime(Config cfg, std::unique_ptr<BatchState> batches);
    ~DeviceRuntime();

    /* Steps 1..6. Returns negative after printing what failed and why; missing kernels are
     * reported as a full list. */
    int start();

    /* Step 7. Blocks until the server is asked to stop. A device fault is not recoverable
     * in-process, so it exits non-zero rather than serving from a wedged queue (spec §17).
     * `watch`, when given, is the startup's stall watch (util/stallwatch.h): held until the
     * server listens, or the KL mode starts scoring, so the stretch from start() to there
     * reports where it waits like start() itself. */
    int serve(std::unique_ptr<StallWatch> watch);

    using HeadlessFront = std::function<int(BatchState&, std::shared_ptr<const Vocab>, const Json&)>;
    int serve_headless(HeadlessFront front);
    void set_external_step_control(std::shared_ptr<ExternalStepControl> control) {
        external_steps_ = std::move(control);
    }
    void set_program_lowerer(std::shared_ptr<const ProgramLowerer> lowerer) {
        program_lowerer_ = std::move(lowerer);
    }
    int write_headless_manifest(const std::string& directory) const;
    Json headless_stats() const;
    std::vector<StorageRankView> storage_rank_views();
    void set_placement_factory(std::function<std::unique_ptr<HeatEngine>(int)> factory) {
        placement_factory_=std::move(factory);
    }

    /* Serve behind `front` (server/front.h) instead of listening: serve() attaches this model's
     * server under its name and `aliases` once it is ready, and returns after the front stops or
     * request_stop() is called, detaching first so no response outlives the server. Set before
     * serve(). */
    void set_front(server::Front* front, std::vector<std::string> aliases) {
        front_ = front;
        front_aliases_ = std::move(aliases);
    }
    /* Ends serve() behind a front: the model stops taking requests, its in-flight responses are
     * waited for, and serve() returns. Safe from any thread. */
    void request_stop(int64_t drain_ms = 600000) {
        stop_drain_ms_.store(drain_ms, std::memory_order_relaxed);
        stop_requested_.store(true, std::memory_order_release);
    }
    /* The fleet that grants this engine its part of the cards (fleet/broker.h); null splits the
     * cards as a process of one model does. Set before start(). */
    void set_broker(VramBroker* b) { broker_ = b; }
    /* For the broker while its grant waits for room: other engines' bring-ups go on meanwhile,
     * since only this engine's own steps need the bring-up lock (engine_bringup.cpp). Called on
     * the starting thread, inside the grant. */
    /* What this engine keeps pinned in host memory: its pools, its tables and its prefix cache's
     * host tier, as configured. */
    int64_t host_pinned_bytes() const {
        return ((int64_t)cfg_.tp * (cfg_.host_pool_mib + cfg_.embedding_host_mib + cfg_.tower_host_mib) +
                cfg_.prefix_cache_mib) << 20;
    }
    void bringup_pause()  { if (bringup_lk_ && bringup_lk_->owns_lock()) bringup_lk_->unlock(); }
    void bringup_resume() { if (bringup_lk_ && !bringup_lk_->owns_lock()) bringup_lk_->lock(); }

    /* ---- parking (docs/MULTIMODEL.md §7). park() stops taking requests (the ones handed out
     * finish within `drain_ms`, then are ended), brings the step loop to rest, and pages the
     * engine's memory out to pinned host memory at the same addresses (dev_page_out): the engine
     * keeps its threads, queues, server, kernels and recorded passes, and holds only its small
     * allocations on the card. unpark() pages it back in and takes requests again. From the
     * fleet's thread, while serve() runs behind a front. RAD_E_UNSUPPORTED when the card has no
     * virtual memory manager, since nothing could leave at the same address.
     *
     * `req` says how much to take off each card (empty: everything), the host memory the copies
     * may take and the disk tier for the rest (device.h, PageOut); `got` what left. A parked
     * engine may be asked again for more. */
    int  park(int64_t drain_ms, const PageOut& req, PageResult* got, std::string* why);
    int  unpark(std::string* why);
    /* unpark() that takes requests at once: the weights parked in host memory stay there, read in
     * place over the link, and dev_page_promote moves them onto the card while the engine serves
     * (`left` is how much stays for now). */
    int  unpark_live(std::string* why, int64_t* left);
    /* Every request is one step: an embedding program's pass over the prompt, nothing generated.
     * Set at bring-up. */
    bool one_step() const { return embedding_; }
    bool parked() const { return parked_.load(std::memory_order_acquire); }
    /* While parked, the linear-state snapshot store's pinned host memory goes back to the host:
     * the park emptied the prefix cache, so the store holds no snapshot, and the next unpark pins
     * it again before paging in (4 GiB pins in about 0.3 s, far less than the host copies of
     * weights it makes room for would cost to read back from the drive). The bytes released; 0
     * when not parked, already released, or a snapshot is still held. From the fleet's thread. */
    int64_t release_host_spare();
    /* While parked: up to about `max_bytes` of the parked memory with no host copy comes into
     * host memory at its own addresses, so the next restore copies it over the link: chunks
     * whose copy is on the drive are read back, and the weights pool's chunks are filled from the
     * container, whole weights at a time. The bytes filled, or a negative status. From the fleet's
     * thread, with the model idle. */
    int64_t fill_host(int64_t max_bytes);
    /* Host memory a refill (a restore's or a host fill's) holds while it runs, at most: the
     * relayout pool's staged inputs (kRefillStaging), its workers' output buffers and the sweep's
     * pinned windows. The fleet keeps this much room for it. */
    static constexpr int64_t kRefillStaging = (int64_t)2 << 30;
    static constexpr int64_t kRefillHost = (int64_t)4 << 30;
    /* What release_host_spare gave back and unpark will pin again. */
    int64_t host_spare_released() const { return ck_released_.load(std::memory_order_acquire) ? ck_cap_ : 0; }
    /* Runs `fn` on the step loop between steps, with nothing of the engine's in flight on any rank
     * (every stream drained), and returns its result; for mapping changes under a serving engine.
     * RAD_E_STATE when the engine stops first. From any thread but the step loop's. */
    int  run_quiet(const std::function<int()>& fn);
    /* The engine's owner id for the device layer (engine_owner): every thread it starts files its
     * allocations under it. */
    int  owner() const { return owner_; }

    /* ---- flex (docs/MULTIMODEL.md §6). What the cache and the expert slab hold on each card
     * beyond the claim, and giving it back: the step loop recalls the slab's loan first, then
     * takes the free top of the cache, the prefix cache's entries and (with `preempt`) the running
     * requests, which go back to the queue by recompute, and unbacks it. Blocks the calling thread
     * until the step loop has done it, at most `wait_ms`. The bytes given back on each card. */
    int64_t flex_bytes() const;
    /* Jobs running with no open request (a video, an async shape): the model is at work. */
    int64_t background_jobs() const { return background_jobs_.load(std::memory_order_acquire); }
    int64_t flex_release(int64_t bytes, bool preempt, int64_t wait_ms, bool kv = true);
    /* Expert flex (§6.7): the bytes on each card the expert slab holds past the claim, lent by the
     * fleet; they go first in flex_release, and with `kv` false they are all that goes. */
    int64_t expert_flex_bytes() const { return xflex_bytes_.load(std::memory_order_acquire); }
    /* What the slab could still grow by on each card while the engine is busy and the fleet had
     * no free memory for it; 0 when it is idle, at its ceiling or still filling what it has. */
    int64_t expert_flex_wanted() const { return xflex_want_.load(std::memory_order_acquire); }
    /* True once serve() has attached to its front (or listens): the model takes requests. */
    bool serving() const { return serving_.load(std::memory_order_acquire); }

    /* --debug-graph and friends: print everything declare and plan learned, without serving. A
     * first-class mode rather than a verbose-flag side effect, so what runs is always visible. */
    int dump();

    const Program& program() const;
    const Config&  config()  const { return cfg_; }

    /* The startup report (core/diag/diag.h): every option with its value and source, the
     * machine, and what on it costs speed or correctness. serve() prints it with RAD_OK before
     * listening; main prints it with the error start() or serve() failed with, so a server that
     * never came up still logs its setup. The report then repeats the error on an E line, so the
     * log's tail shows it. */
    void print_startup_report(int status);
    /* The report as JSON, with the hardware re-read, since link and clock state only mean
     * something under load (/diagnostics). The reading is reused for a second and covers only the
     * serving cards' power-management files (hardware_facts). */
    std::string diagnostics_json() const;
    bool startup_report_printed() const { return report_printed_; }
    /* The cards an engine may run on, as start() filters them (one architecture, discrete before
     * integrated); the fleet places models on them. */
    static int usable_cards(std::vector<int>* out);

    /* --proceed-with-incompatible-configuration's banner (diag::unsupported_banner) on W lines;
     * main prints it before anything else, so it heads the log. */
    static void print_unsupported_banner();

    /* ---- the control vector's public surface (core/cvec.h).
     *
     * `cvec_loaded` is whether a vector was found and validated for this container; a request
     * asking for the feature on a deployment without one is refused by name rather than silently
     * served unprojected. `cvec_default_on` is the runtime default for requests that name no
     * choice, and the setter is what POST /admin/control-vector calls: it flips a flag, builds
     * nothing, reloads nothing, and leaves every running request alone. */
    bool cvec_loaded() const { return cvec_.loaded(); }
    bool cvec_default_on() const { return cvec_default_.load(); }
    void set_cvec_default(bool on) { cvec_default_.store(on); }
    /* {available, default, mode, first, last, n_embd, n_layers, scale, file} for /stats, the
     * admin route and the dashboard. */
    std::string cvec_json() const;
    /* The value the block cache is keyed by for a step with this mode: the vector's identity when
     * on, 0 when off. Zero with no vector loaded, so a deployment without the feature hashes
     * exactly as before. */
    uint64_t cvec_salt_for(bool on) const { return on ? cvec_salt_ : 0; }

private:
    /* One thread per tensor-parallel rank, each owning one device. Ranks share the address
     * space, so there is no IPC or serialisation and the step batch is shared by pointer rather
     * than broadcast (spec §1). */
    struct Rank;
    std::vector<std::unique_ptr<Rank>> ranks_;
    std::function<std::unique_ptr<HeatEngine>(int)> placement_factory_;
    /* Per rank and KV group, that rank's stream view or null; the batch builder takes these, since it
     * cannot reach each rank's manager. */
    std::vector<std::vector<const RadKVStream*>> rank_stream_views_;
    /* Per KV group, the ranges a staged pass of the next step copies (kvstream_step_runs), refilled
     * in place each step. */
    std::vector<std::vector<KVStageRun>> stage_runs_;
    /* The tiers' work in flight: the host-memory copies the transfer streams are running. */
    std::unique_ptr<TierPump> pump_;
    /* serve()'s startup stall watch, released when the server listens (run_server) or the KL
     * mode starts (run_kld). */
    std::unique_ptr<StallWatch> startup_watch_;
    /* rad_mono_ns at the top of start(), so serve() can report how long bringup took. The same
     * span the startup watch covers (main.cpp creates it here and ends it at the listen). */
    uint64_t              startup_begin_ns_ = 0;
    std::vector<std::thread>           rank_threads_;
    std::shared_ptr<ExternalStepControl> external_steps_;

    Config cfg_;
    /* The VRAM split each rank actually takes, filled by DeviceRuntime::plan. Equal to cfg_ on every
     * rank whose cache costs as much per token as the dearest rank's. A cheaper rank (an
     * attention-zero rank) gets a KV pool sized for the common token count and the difference in
     * its weights pool. Pools::configure and the planner must read the rank's entry, never cfg_;
     * plan() refuses a rank whose plan and weights pool disagree. */
    std::vector<Config> rank_cfg_;
    int64_t kv_planned_tokens_ = 0;   /* the common token count plan() sized the re-split pools for */

    /* The model's source, one of the two. A container stays mapped for the engine's life because
     * the mapped and file tiers read it in place (spec §4.3). A checkpoint served directly also
     * stays mapped and is read once, at load (spec §4.3). */
    std::unique_ptr<RadFile> file_;
    std::unique_ptr<Checkpoint> ckpt_;
    std::unique_ptr<CheckpointSources> ckpt_src_;
    /* The source rad_weight_encoding answers from: container entries or checkpoint tensors. */
    WeightSourceFn sources() const;

    /* --expert-stats FILE (place/xstats.h); null without the flag. */
    std::unique_ptr<ExpertStats> xstats_;

    RadModelMeta   meta_{};
    LoadedPlugins  plugins_;
    std::shared_ptr<const ProgramLowerer> program_lowerer_;
    /* This engine's share of the process plugin set (plugin/registry.h, Sharing): its device facts,
     * selection table and kernel instances. Made by load_plugins; plugins_.registry points at it. */
    std::unique_ptr<Registry> reg_;
    /* The first of this engine's instance slots (RadBuildCtx::slot_base), held from load_plugins
     * to the destructor (rad_engine_claim); -1 while none is held. */
    int slot_base_ = -1;
    bool cards_claimed_ = false;    /* rad_dev_cards_claim, for an engine of more than one rank */

    /* RadModelMeta holds bare char*, so the strings behind its key/value table must outlive
     * every declare. */
    std::vector<std::string> meta_kv_k_, meta_kv_v_;
    std::vector<const char*> meta_kv_kp_, meta_kv_vp_;

    void (*arch_step_)(RadCtx*, const RadBatch*) = nullptr;
    uint64_t (*arch_pass_key_)(RadCtx*, const RadBatch*, const int32_t*) = nullptr;

    int load_plugins();
    int read_metadata();
    int read_checkpoint_metadata();
    /* --rope-scaling over the free-form keys, then the startup line and the context checks;
     * runs on meta_kv_k_/meta_kv_v_ before their pointer arrays are built. */
    int apply_rope_scaling();
    int declare();
    int plan();
    /* Runtime LoRA (core/engine_lora.cpp). Targets are fixed before declare; adapters are loaded
     * into slots after every rank's Ctx is bound, and a request's `model` names one. */
    int lora_targets();
    int lora_start();
public:
    int lora_load(const std::string& name, const std::string& dir, std::string* err);
    /* Unpublish an adapter (lora::SlotTable::unload). Idle, its slot is freed at once; in use, it
     * drains: *in_flight says how many requests still hold it, no new request may name it, and the
     * last of them frees the slot, which cools before reuse (lora/slots.h). */
    int lora_unload(const std::string& name, std::string* err, int* in_flight);
    /* RAD_OK with the slot, the uid of its load and a hold: the adapter is not freed while the
     * hold lives (the server keeps it in the request's Generation). RAD_E_NOTFOUND for a name no
     * slot holds, or one that is draining. */
    int lora_acquire(const std::string& name, int32_t* slot, uint64_t* uid,
                     std::shared_ptr<void>* hold);
    std::vector<std::string> lora_names() const;
    /* A generator's adapters: a request names one or more adapter files at their strengths, and
     * gets the slot holding exactly that combination, composed into a free (or the least recently
     * used idle) slot when none holds it yet. `note` says what the composition left out or
     * approximated, for the reply. RAD_E_NOTFOUND names an adapter neither --lora nor --lora-dir
     * provides; RAD_E_FULL means every slot is held by running requests. */
    struct LoraUse { std::string name; double weight = 1.0; };
    int gen_lora_acquire(const std::vector<LoraUse>& uses, int32_t* slot,
                         std::shared_ptr<void>* hold, std::string* note, std::string* err);
    /* The adapter files a generator request may name: --lora's and those under --lora-dir. */
    std::vector<std::string> gen_lora_names() const;
private:
    /* A generator's adapter files by name (--lora, POST /v1/load_lora_adapter, --lora-dir), opened
     * once and kept mapped; compositions read them when a combination is put in a slot. */
    int gen_lora_open(const std::string& name, std::shared_ptr<const lora::AdapterFile>* out,
                      std::string* err);
    int gen_lora_start();
    /* Every rank's packed bytes into `slot` of its pools, each rank on a stream of its own. */
    int lora_copy(int slot, const std::vector<std::vector<lora::PackedSite>>& packed,
                  std::string* err);
    bool                        lora_gen_ = false;
    std::unique_ptr<lora::KeyMap> lora_keys_;
    mutable std::mutex          lora_lib_mu_;
    std::map<std::string, std::pair<std::string, std::shared_ptr<const lora::AdapterFile>>> lora_lib_;
private:
    std::atomic<uint64_t>      lora_epoch_{0};   /* step-loop iterations, for slot reuse */
    lora::SlotTable            lora_table_;      /* reads lora_epoch_, so declared after it */
    /* Serialises loads: a slot is reserved, filled on every rank and published by one load at a
     * time. Requests and unloads go to the table alone and never wait for a load's copy. */
    std::mutex                 lora_mu_;
    std::set<std::string>      lora_targets_;
    /* --host-headroom-mib in bytes: the host RAM startup leaves free for the system when it
     * pins host memory (engine_bringup.cpp). 4 GiB suits a busy server; a small host wants to
     * leave less to make room for its tiers, which is why it is a flag and not a constant. */
    int64_t host_headroom_bytes() const { return cfg_.host_headroom_mib * ((int64_t)1 << 20); }
    /* What a table's or a stream's pin leaves free: the headroom, and in a fleet what the fleet
     * keeps (VramBroker::host_margin). */
    int64_t host_margin_bytes() const;
    /* Why a fleet's load put an embedding left to its default on the card; empty if it did not. */
    std::string emb_on_card_why_;
    /* Resolves --embedding-placement, --vision-tower-placement and --ngram-placement into Config
     * before the card is split (engine_bringup.cpp describes how auto chooses). `wfs` is every
     * rank's footprint, read for its towers. `host_late` is the pinned host memory startup
     * allocates after the tables and outside every pool (the checkpoint store, the host
     * activation arenas), which a table pinned in host memory must leave room for. */
    int resolve_table_placement(const std::vector<VramFacts>& facts,
                                const std::vector<int64_t>& emb_bytes,
                                const std::vector<WeightFootprint>& wfs, int64_t kv_floor,
                                int64_t ngram_bytes, bool ngram_device, bool ngram_host,
                                int64_t host_late);
    /* --ngram-placement ram: one copy of the container's [src, src + n) at file offset `foff`,
     * shared by every rank. Locked for the host gather, or pinned for a device kernel to read
     * over the link (`card`); with ram=DIR the pinned copy is a file other engines map too. */
    int copy_table(const uint8_t* src, int64_t n, int64_t foff, bool card, void** out,
                   std::string* why);
    /* An auto-chosen pinned copy failed: move every rank's n-gram gather to the host. */
    bool ngram_gather_to_host();
    /* The table copies: locked ones as [base, base + bytes), pinned ones as rad_dev_alloc
     * RAD_MEM_HOST_MAPPED ranges, and the container address each copies. Freed in ~DeviceRuntime after
     * the ranks that read them are gone. */
    std::vector<TableCopy> table_copies_;
    std::vector<void*> card_copies_;
    /* The files behind the card copies made under --ngram-placement ram=DIR. Closed after the
     * copies are freed, because the holder that leaves last deletes the file. */
    std::vector<TableShare> table_shares_;
    std::vector<std::pair<const void*, void*>> table_copy_of_;
    /* --ngram-placement disk=DIR: the table's own file, mapped read-only in place of the
     * container's range (table_file_map); unmapped in ~DeviceRuntime. */
    std::vector<TableCopy> table_files_;
    std::vector<const void*> table_file_src_;   /* the container address each of table_files_ replaces */
    int table_file_map(const uint8_t* src, int64_t n, int64_t foff, void** out, std::string* why);
    /* --expert-profile: the decode ranking loaded once, before any rank plans, and handed to every
     * rank's planner in place of the container's activation profile. */
    std::vector<RadFileProfile> expert_profile_;
    int load_weights();
    /* After a page-in: every weight with a byte in `spans` (memory the page-in mapped back empty,
     * dev_page_refilled) is read from the container and laid out again at its current address,
     * through the load's sweep and relayout pool. */
    /* `before_writes`, when set, runs once before the refill's first write into the spans. */
    int refill_weights(const std::vector<PageSpan>& spans, const char* what = "restore",
                       const std::function<int()>& before_writes = {});
    /* dev_page_in (or the live form with `left`) and the refill of what it mapped back empty. */
    int page_back_in(int64_t* left);
    int configure_kv();
    /* Resolves --checkpoint-slots auto for snapshots of `slot` bytes (`what` names their kind). */
    int size_checkpoint_store(int64_t slot, const char* what);
    /* Maps the step's logits rows back to the sequences that wanted them. Not the identity: a
     * mid-prompt prefill chunk produces no row, and a speculative verify produces 1 + n_spec. */
    struct StepOut {
        std::vector<Request*> reqs;       /* the requests that wanted a row, in row order */
        std::vector<int32_t>  first_row;  /* [n_seq] that entry's first row, or -1 */
        /* Leading logits rows the sampler reads. RadBatch::n_out also counts the KL mode's scored
         * rows after them (Request::score). */
        int32_t               n_rows = 0;
        std::vector<int32_t>  rows;       /* [n_out] the device's answer, copied back */
        int32_t*              rb = nullptr;   /* the pinned readback collect_issue filled */
        std::vector<float>    lg;         /* the first row's logits, at -vv */
        std::vector<int32_t>  token;      /* [n_seq] */
        std::vector<int32_t>  accepted;   /* [n_seq] */
        std::vector<uint8_t>  eos;        /* [n_seq] */
        /* [n_rows] each sampled row's logprobs, for the rows of requests that asked for them;
         * empty on a step with none. */
        std::vector<TokenLogprobs> lp;
        /* The ids' copy recorded tok_end_ ahead of the drafter's passes: collect_step waits for
         * it alone, and the step loop for step_end_ after the commit. */
        bool                  early = false;
        /* [n_seq][RAD_SCHED_MAX_SPEC] on a tree step (StepPlan::spec_is_tree): each entry's
         * accepted nodes in commit order (accept_tree's path). Empty on a chain step. */
        std::vector<int32_t> path;
    };

    /* The text frontend, loaded from the container. The sampler's grammars walk `vocab_view_`,
     * which references `vocab_` and must not outlive it. */
    std::shared_ptr<const Vocab> vocab_;
    std::unique_ptr<VocabView>   vocab_view_;
    /* The container's second vocabulary (rad-convert --tokenizer2), which an image generator that
     * declares `aux_tokens` reads its prompt through; null without one. */
    std::shared_ptr<const Vocab> aux_vocab_;

    /* The chunk geometry, resolved once and shared: the prefix cache needs the checkpoint interval
     * at configure time and the scheduler the whole geometry at init. Resolving it separately
     * would let the cache use the raw interval while the scheduler uses the rounded one. */
    ChunkGeometry                 geo_;

    /* One scheduler for every rank: one decision per step, one batch, shared by pointer. */
    std::unique_ptr<BatchState>    batch_state_owner_;
    BatchState&                   sched_;
    std::unique_ptr<IKVManager>   kv_iface_;
    std::unique_ptr<IPrefixCache> pc_iface_;

    /* The release point. Width is tp + 1: the scheduler thread also takes part. */
    std::unique_ptr<RankBarrier>  barrier_;
    std::atomic<const RadBatch*>  step_batch_{nullptr};
    /* What a release asks the ranks to issue: the forward pass and its sampler (kIssueAll), the
     * forward pass alone (kIssueModel: the next step's, issued before this one is read back), or
     * that pass's sampler alone (kIssueSampler, once the host has committed its inputs). Every
     * release must set it; a stale value would make a draft pass issue only its sampler. */
    enum { kIssueAll = 0, kIssueModel = 1, kIssueSampler = 2 };
    std::atomic<int>              issue_what_{kIssueAll};
    std::atomic<int>              step_status_{RAD_OK};
    std::atomic<bool>             stopping_{false};
    /* The error the step loop ended on; RAD_OK while running or after a clean stop. Written once
     * by fail_loop(); read by the server watcher to stop HTTP and by serve() to return it. */
    std::atomic<int>              loop_status_{RAD_OK};
    server::Front*                front_ = nullptr;
    std::vector<std::string>      front_aliases_;
    std::atomic<bool>             stop_requested_{false};
    /* How long a stop lets responses already handed out finish before the server ends them. */
    std::atomic<int64_t>          stop_drain_ms_{600000};
    VramBroker*                   broker_ = nullptr;
    std::unique_lock<std::mutex>* bringup_lk_ = nullptr;   /* start()'s, while it runs */
    const int                     owner_;
    std::atomic<bool>             parked_{false};
    /* The step loop's rest for a park: the fleet sets hold_req_, the loop answers with hold_ack_
     * once it is idle with nothing in flight and waits there until hold_req_ clears. */
    std::atomic<bool>             hold_req_{false};
    std::atomic<bool>             hold_ack_{false};
    /* hold_point's wait: woken when the hold clears (hold_release) or the engine stops
     * (wake_hold). */
    std::mutex                    hold_mu_;
    std::condition_variable       hold_cv_;
    void                          hold_point(StallWatch* watch = nullptr);
    void                          hold_release();
    void                          wake_hold();
    int                           unpark_with(std::string* why, int64_t* left);
    void                          drain_streams();
    std::vector<int>              rank_devices() const;   /* each rank's card, in rank order */
    server::Server*               srv_ = nullptr;   /* run_server's, while it runs */
    std::atomic<int64_t>          background_jobs_{0};  /* server::Deps::background */
    /* flex_release's handoff to the step loop (flex_service). */
    int64_t                       flex_floor_bytes_ = 0;   /* rank 0's KV backing after the carve */
    std::mutex                    flex_mu_;
    std::condition_variable       flex_cv_;
    std::atomic<bool>             flex_pending_{false};
    int64_t                       flex_want_ = 0;
    bool                          flex_preempt_ = false;
    bool                          flex_kv_ = true;          /* the cache's flex may go too */
    int64_t                       flex_got_ = -1;
    uint64_t                      flex_room_ns_ = 0;
    void                          flex_service();
    void                          flex_room_tick(uint64_t now_ns);
    /* Expert flex (§6.7). Each rank's slab borrows from its own reserved range (Rank::xflex), the
     * top `xflex_bytes_` of it backed; the ceiling is the loan that covers every slot carved there.
     * xflex_tick grows it on the step loop while the engine is busy, a GiB at a time once the
     * slab has filled what it holds; xflex_give recalls and unbacks it, between steps. */
    std::atomic<int64_t>          xflex_bytes_{0};
    std::atomic<int64_t>          xflex_want_{0};
    int64_t                       xflex_cap_ = 0;
    int64_t                       xflex_claimed_ = 0;       /* the claim's expert bytes a card */
    uint64_t                      xflex_ns_ = 0, xflex_busy_ns_ = 0, xflex_retry_ns_ = 0;
    void                          xflex_tick(bool busy);
    int64_t                       xflex_give(int64_t bytes);
    int64_t                       xflex_settle();
    /* run_quiet's handoff to the step loop (quiet_service). */
    std::mutex                    quiet_mu_;
    std::condition_variable       quiet_cv_;
    std::atomic<bool>             quiet_pending_{false};
    const std::function<int()>*   quiet_fn_ = nullptr;
    int                           quiet_rc_ = 0;
    bool                          quiet_done_ = false;
    void                          quiet_service();
    std::atomic<bool>             serving_{false};
    int  fail_loop(int st);

    /* This step's scored rows (Request::token_scores): a logits row and the result it fills.
     * Written by the scheduler thread while the ranks are parked, read by them during the step,
     * consumed by collect_step. */
    std::vector<std::pair<int32_t, TokenScores*>> score_rows_;
    /* A scored step (RadBatch::score_only): the plugin ran no lm_head, and the candidates' logits
     * come from the hidden rows and the head table's rows at `score_ids_` (every scored row's
     * candidates, sorted and unique), both read back into `score_host`. */
    bool                 score_only_ = false;
    std::vector<int32_t> score_ids_;
    int64_t              score_embd_ = 0;
    int  stage_scored_rows(const StepPlan& plan, const StepOut& so, std::vector<uint64_t>& cancel);
    int  issue_scored_rows(Rank& r);
    int  collect_scored_rows();

    /* Gate for the sampler's host work, which runs while the device executes the model.
     * Staging sampler parameters (and, with a grammar, a whole-vocabulary mask per draft
     * position) touches no stream: Sampler::run issues every upload on the rank thread. So the
     * ranks enqueue the model first, the scheduler runs build_step meanwhile, and the ranks wait
     * here before enqueueing the sampler chain. One-shot per step, reset while the ranks are
     * parked at the release, so no rank can see a stale ready flag. */
    std::mutex                    build_mu_;
    std::condition_variable       build_cv_;
    bool                          build_ready_  = false;
    int                           build_status_ = RAD_OK;

    /* Scratch for the checkpoint copies: one sequence's state slots, reused so the step path
     * does not allocate. See state_slots_of. */
    std::vector<int32_t>          ckpt_slots_;
    std::vector<int32_t>          ckpt_win_;    /* KVManager::window_rows, same reuse */
    bool state_slots_of(uint64_t seq, std::vector<int32_t>* out) const;

    int  load_vocab();
    int  declare_sampler(int rank, RadBuilder* b, int64_t logits_width, bool sizing = false);
    /* The build context for every declare of rank `rank` at `max_tok`; the real declare and the
     * arena's sizing declares differ only in max_tok. */
    RadBuildCtx build_ctx(int rank, int64_t max_tok) const;
    void probe_arena_levels();
    int serve_impl(std::unique_ptr<StallWatch> watch, const HeadlessFront& front);
    int  step_loop();
    /* Issue the copy of the step's sampled ids to the host, then wait for and interpret them.
     * Split so the copy is issued before draft passes overwrite the ids. With `fence` the copy
     * records tok_end_, so the wait for the ids skips the passes (StepOut::early). */
    int  collect_issue(StepOut& so, bool fence = false);
    int  collect_step(const RadBatch* b, StepOut& so);
    /* `row`'s logprobs for `token`, filed at `at` (default `row`): a tree verify emits a token from
     * its parent node's row but files it as the entry's j-th emitted token. */
    int  row_logprobs(StepOut& so, int32_t row, int32_t token, int32_t at = -1);
    /* The step's end on the card: after its last draft pass and the host readbacks, before the
     * step issued ahead. The engine waits on this event, not the stream, so the card runs the
     * next step while the host commits this one. */
    int  mark_step_end(const RadBatch* b, StepOut& so);
    RadEvent step_end_ = nullptr;
    /* The sampled ids' copy, ahead of the drafter's passes (collect_issue's `fence`). */
    RadEvent tok_end_ = nullptr;
    /* Whether the ids may be read before the drafter's passes end: not in KL mode, whose scored
     * rows are read from the logits after collection, nor for an embedding program. */
    bool     early_ids_ = false;

    /* ---- KV streaming's counters (engine.cpp, "KV streaming's counters"). Pinned, because a
     * device-to-host copy into pageable memory blocks the stream. */
    struct StreamRead {
        int64_t  resolves = 0, lookups = 0, misses = 0;
        int64_t  max_union = 0, max_misses = 0, max_claims = 0;
        int64_t  slots = 0, block_bytes = 0;
        bool     overflow = false;
        int32_t  group = -1;
    };
    std::vector<StreamRead> stream_prev_;   /* what metrics have already been charged */
    /* Pinned landing area for the copies: [n_kv_groups][16] i32, grown on first use. */
    int32_t*                stream_ctl_p_ = nullptr;
    size_t                  stream_ctl_n_ = 0;
    int  stream_probe_issue(const RadBatch* b);
    int  stream_probe_take(const RadBatch* b, bool prefill_done, bool request_end);
    void stream_report(const char* when, const StreamRead& r, const StreamRead& prev);

    /* The draft head as declared (rad_declare_drafter), zeroed when no speculative window was
     * asked for. `kind` selects which of the two drivers below runs. `drafter_name_` owns the
     * string because RadDrafterDecl::name points into the Program. */
    RadDrafterDecl       drafter_{};
    std::string          drafter_name_;
    /* The block drafter's shorter block (rad_drafter_short_block), 0 when it has none or will not
     * run it (--draft-short-from 0, or auto for a chain). */
    int64_t              drafter_short_ = 0;
    bool                 short_on_ = false;     /* the short block runs now (block_shape) */
    /* --draft-short-from resolved at bring-up (auto: 0 for a chain, 2 for a tree). */
    int64_t              short_from_ = 0;

    /* Serial driver (spec §10): `depth` dependent passes after the step commits, each embedding
     * the previous proposal. MTP, EAGLE and Medusa-serial have this shape. */
    std::vector<int32_t> draft_out_;   /* [rows][depth], what each pass proposed */

    /* Host side of every step-path readback (sampled tokens, draft proposals). Pinned, because a
     * device-to-host copy into pageable memory bounces synchronously through a staging buffer and
     * blocks on the whole stream; into pinned memory it is one more stream packet and the step's
     * own wait is the only wait. Grown on first use to the largest readback requested. */
    struct Readback {
        int32_t* p = nullptr;
        int64_t  n = 0;
        int32_t* reserve(int64_t count);
        ~Readback();
    };
    Readback             readback_;
    /* Embedding program (rad_declare_pooled): no sampler; a step's answer is the pooled buffer's
     * first n_seq rows. `pooled_host_` is their pinned landing area, [max_seqs, pooled_dim_] f32;
     * collect_pooled copies them behind the step and hands each row to its sink after the
     * step's one wait. */
    bool                 embedding_ = false;
    int64_t              pooled_dim_ = 0;
    float*               pooled_host_ = nullptr;
    int  collect_pooled_issue(const RadBatch* b, StepOut& so);
    int  collect_pooled(const RadBatch* b, StepOut& so);
    /* A generator (rad_declare_generator, rad_declare_audio_generator,
     * rad_declare_video_generator, rad_declare_shape_generator): the driver that turns requests
     * into passes and the loop that runs them, replacing the scheduler and step loop
     * (engine_gen.cpp). Null for every other program. `img_gen_`, `aud_gen_`, `vid_gen_` and
     * `shp_gen_` are the same driver by its kind, for the endpoints; one of them is set when
     * `gen_` is. */
    std::unique_ptr<gen::IPassDriver> gen_;
    gen::Driver*                      img_gen_ = nullptr;
    gen::AudioDriver*                 aud_gen_ = nullptr;
    gen::VideoDriver*                 vid_gen_ = nullptr;
    gen::ShapeDriver*                 shp_gen_ = nullptr;
    int  gen_loop();
    /* Zeroes the state slots the KV manager granted since the last step, on every rank's stream,
     * before the step or pass that first reads them. */
    void clear_fresh_states();
    bool serial_drafting() const;
    int  advance_all(int mode, int round, bool from_proposal, int rows = -1);
    int  issue_serial_drafter(int step, StepOut& so);
    /* The next step's forward pass, issued before this one is read back; null when the scheduler
     * wants this step committed first. See Scheduler::step_next. */
    int  issue_ahead(const RadBatch** out, ExternalPermit* permit = nullptr);
    /* The arena level and the followers' pools, before any release that runs a forward pass. */
    int  prepare_release(const RadBatch* b);
    /* Whether a step may be issued ahead: requires that the drafter, if any, runs on the card (the
     * prompt-lookup drafter reads the committed sequence). */
    bool pipelining_ = false;
    int  finish_serial_drafter(const StepOut& so);
    /* Handed from issue_serial_drafter to finish_serial_drafter: the drafting rows, the rounds
     * asked for, and the rounds that ran. */
    std::vector<BatchState::SerialRow> serial_row_;
    int  serial_rounds_ = 0, serial_rows_ = 0, serial_done_ = 0;
    /* The StepSlots and end-of-generation ids on every rank, and the pinned host copy of rank 0's
     * slots the step reads back (sched/advance.h). */
    int  step_state_init();
    void step_state_fini();
    StepSlot*            slots_host_ = nullptr;

    /* --draft-coupling (engine_draft.cpp). The head declares `mtp_couple_draws` and
     * `mtp_couple_state`; the engine stages one draw per drafting row before the serial passes and
     * reads the states back once, then binds each to the verify row it will be coupled at. */
    bool     mtp_couple_ = false;
    rad_buf  couple_draws_buf_ = 0, couple_state_buf_ = 0;
    int64_t  couple_rows_ = 0;                       /* the state plane's rows per round */
    RadMtpCoupleDraw*  couple_draw_stage_ = nullptr;   /* pinned [max_seqs] */
    RadMtpCoupleState* couple_readback_  = nullptr;    /* pinned [rounds, couple_rows_] */
    std::vector<RadMtpCoupleDraw> couple_by_card_;      /* the plan's draw per card slot */
    RadMtpCoupleState  couple_bound_[RAD_MTP_COUPLE_MAX_ROUNDS] = {};
    bool couple_evidence_ = false;   /* the completed-verify line was written (once a process) */
    bool couple_host_note_ = false;  /* the host-domain counterpart was written (once) */
    bool couple_malformed_note_ = false;   /* a status -1 round was reported (once) */
    /* `rows` are serial_row_ or block_row_ (Scheduler::SerialRow / BlockRow). */
    template <class Row> int stage_couple_draws(const Row* rows, int n);
    int  bind_couple(uint64_t seq, int i, int done, bool drafted, int32_t card);
    /* One INFO line at the first coupled verify that completed: the step, the live requests whose
     * rows carried a bound state, the kernel and its domain. Read from the sampler's build
     * snapshot after collect_step's wait, so it reports a finished verify, not an enqueue. */
    void couple_verify_evidence(int step);

    /* Step-time attribution, Debug only (see the barrier in step_loop): time per scheduler phase.
     * The nine phases sum to the whole step, including the readback, commit and drafter after the
     * join. */
    enum Ph { PH_PLAN, PH_BEGIN, PH_BUILD, PH_PRE, PH_RANKS, PH_SAVE, PH_COLLECT, PH_COMMIT,
              PH_DRAFT, PH_N };
    double phase_us_[PH_N] = {};
    int    phase_n_ = 0;
    std::chrono::steady_clock::time_point phase_t_{};
    void   phase_mark(int p);

    /* Block driver (spec §10): a context pass mirroring the step just run, then one pass that
     * fills every drafted position at once. DFlash2 has this shape. issue_block_drafter queues
     * both behind the step before its readback and finish_block_drafter publishes the proposals
     * after the commit, so the step can be issued ahead. */
    bool block_drafting() const;
    /* RadBuildCtx::draft_tree for every declare: decided once in declare() from the plugin's probe
     * (RadArchProbe::tree_pays). */
    bool draft_tree_ = false;
    int  block_pass(const RadBatch* db, const char* what, int step);
    /* A program's block drafter rides its window in the window snapshot: it stores its context
     * in the context pass right behind the trunk step, and the snapshot is saved after that. Only
     * where the trunk's window is snapshotted at all (window_snapshots): a full-attention trunk
     * takes no snapshot for a drafter's window alone. */
    bool drafter_window_snap(const Program& p) const {
        return cfg_.n_spec > 0 && p.drafter.kind == RAD_DRAFT_BLOCK && p.drafter.depth > 0 &&
               window_snapshots(p, cfg_);
    }
    /* The drafter's window rides in the window snapshot (KVManager::drafter_window_snapshot), so
     * the checkpoint save waits for its context pass (save_checkpoints). */
    bool drafter_snap_ = false;
    int  save_checkpoints();
    // FreeToken diagnostic: compare logical cached pages against the saved trunk prefix.
    int audit_prefix_pages(uint64_t seq, int64_t tokens, int32_t checkpoint, bool save);
    std::map<std::string, std::vector<uint64_t>> prefix_page_hashes_;
    /* A tree drafter (rad_drafter_tree), driven on the card only: ADV_BTAKE takes its parents,
     * ADV_COMPACT moves the accepted path's K and V after the acceptance, and ADV_CTXTREE stages
     * the context pass by that path. */
    bool    tree_ = false;
    rad_buf tree_parents_ = 0;
    int     tree_compact(int rows);
    int  issue_block_drafter(int step, StepOut& so);
    int  finish_block_drafter(const StepOut& so);
    /* The block and proposal count this step's query pass runs at: the drafter's own, or its short
     * block once --draft-short-from sequences decode, back to its own two below or at one sequence
     * (Scheduler::short_block_on). */
    void block_shape(int* block, int* steps);
    /* Handed from issue_block_drafter to finish_block_drafter and issue_ahead: the drafting rows,
     * in the query pass's order, and how many there are (0 when no query pass ran). */
    std::vector<BatchState::BlockRow> block_row_;
    int  block_rows_ = 0;
    /* The proposals a row of the query pass took (block_shape), for finish_block_drafter and
     * issue_ahead's expect_draft. */
    int  block_steps_ = 0;
    /* Encoder driver (spec §11, core/engine_mm.cpp): encodes every media item the scheduler waits
     * on, in passes of whole segments, and reads it back into the item. Runs between steps, never
     * inside one. */
    int  run_encodes();
    int32_t enc_step_ = 0;
    void drain_completed();
    /* Terminal ids drain_completed() took, reused so steady state allocates nothing. */
    std::vector<uint64_t> done_;
    /* The tiers' per-iteration work: collect a landed copy, start the next, make room in a full
     * arena. Never waits on the device; the scheduler's lock covers only the bookkeeping. */
    void tier_tick();
    /* --prefix-cache-ttl, once a second, with or without the tiers (PrefixCache::expire). */
    void expire_tick(uint64_t now_ns);
    uint64_t expire_last_ns_ = 0;
    /* Issue and land every waiting host copy now. Called under the scheduler's lock by an
     * eviction that would otherwise erase what the copies cover. True when any landed. */
    bool flush_copies();
    /* The KV loan: what the cache is not using, lent to every rank's expert slab at addresses the
     * cache already owns (KVManager, the loan block). Lives here, not in the scheduler, because
     * rank 0's cache sets the amount while every rank's mover fills or empties slots, and a mover
     * may only be driven while its rank thread is parked, which this thread knows. `idle` also
     * runs each mover's own fill, which a busy engine gets from the step. */
    void loan_tick(bool idle);
    bool     lending_ = false;
    uint64_t loan_last_ns_ = 0;
    /* The activation arena's loan (DeviceRuntime::arena_step). `arena_rows_[k]` is the largest step Ctx
     * level k serves (level 0 is the full plan); empty when the arena lends nothing. The floor is
     * the largest recent step: the loan grows past it only after no step that large has run for a
     * while, so a run of prefill chunks does not refill and recall the slab between chunks. */
    int  arena_step(const RadBatch* b);
    void arena_tick(uint64_t now_ns);
    int  arena_level(int64_t rows) const;
    std::vector<int64_t> arena_rows_;
    /* The span's shared level (DeviceRuntime::probe_arena_levels): the rows of the level a step of one
     * interval runs at under --prefill-span, where the prefill stager's ring is the level's own
     * (stager.h); 0 without one. */
    int64_t  arena_span_rows_ = 0;
    int64_t  arena_floor_ = 0;
    int64_t  arena_last_ = 0;
    uint64_t arena_floor_ns_ = 0;
    /* VRAM the budget reserves for allocations after start(), on top of --gpu-headroom-mib: the
     * step batch's staging (allocated by serve()) and what is built on first use, if ever (the KV
     * tiers' copy staging, the second compute lane's queue). Bring-up's final VRAM reading is
     * checked against both. */
    int64_t  late_step_bytes_ = 0;
    int64_t  late_lazy_bytes_ = 0;
    /* Linear-state snapshot store: pinned host memory, --checkpoint-slots slots, each holding every
     * rank's shard of one snapshot (KVManager::set_checkpoint_store). Off the card because no
     * kernel reads a snapshot; it is only copied back into a sequence's state. */
    HostArena ck_store_;
    std::atomic<bool> ck_released_{ false };   /* release_host_spare closed it; unpark opens it */
    int64_t           ck_slot_ = 0, ck_cap_ = 0;
    int  repin_host_spare(std::string* why);
    int  run_server();
    /* KL mode (core/kld.h), replacing run_server: submits the corpus documents as scoring
     * requests and hands every step's scored rows, copied to the host, to `kld_`. One pinned
     * block per rank, holding a step's tokens of that rank's logits columns. */
    int  run_kld();
    int  run_docs(std::vector<std::unique_ptr<Request>>& reqs,
                  std::vector<std::unique_ptr<server::Sink>>& sinks, bool fit_pool,
                  const char* tag, int every_s,
                  const std::function<void(size_t, double)>& progress, int* bad);
    /* --diagnose (core/engine_diagnose.cpp), replacing run_server: runs a fixed workload on this
     * engine -- decode by prompt category, a concurrency sweep, a prefill sweep, interference --
     * and writes DIR/diagnose.json and DIR/diagnose.md instead of listening. */
    int  run_diagnose();
    int  score_rows(const RadBatch* b, const StepOut& so);
    std::unique_ptr<Kld> kld_;
    std::vector<float*>  kld_host_;
    std::vector<int64_t> kld_vocab0_, kld_width_;   /* per rank read: its logits columns */
    /* --calib-corpus, replacing run_server: the corpus, loaded before declare so the plugin is told
     * its token count (RadBuildCtx::calib_tokens), submitted as plain prefill requests. */
    int  run_calib();
    std::unique_ptr<Kld> calib_;
    /* --kld-record beside --calib-dir: the KL corpus's tokens, which the plugins record over. */
    long long calib_tokens_ = 0;
    /* The engine half of /server_info, as a JSON object. See the definition in engine.cpp. */
    std::string engine_info() const;
    /* The sampler a request that sends no settings runs with: the container's generation defaults
     * with the operator's flags over them (resolve_sampling_defaults, engine.cpp). A mode that
     * builds requests with no client -- --diagnose -- samples with what the server would. */
    SamplingParams sampling_defaults() const;

    /* ---- startup report inputs (engine_diag.cpp). `decisions_` holds each value startup resolved
     * and why, recorded where it was resolved; the rest are facts only startup saw. */
    std::vector<diag::Decision> decisions_;
    void decide(const std::string& key, const std::string& value, const std::string& why) {
        diag::decide(&decisions_, key, value, why);
    }
    /* The architecture's RadArchProbe, asked once in declare(): a property of the model, so every
     * RadBuildCtx carries the same answer (RadBuildCtx::kv_stream_ok). */
    RadArchProbe arch_probe_{};
    std::string devnote_;            /* the devices usable_devices left out, and why */
    int         p2p_measured_ = -1;  /* 1 peer to peer, 0 through host memory, -1 not asked */
    std::string p2p_blocked_;
    bool        has_ngram_ = false;  /* the model carries an n-gram table */
    std::string prefix_off_why_;     /* why the engine itself turned prefix caching off, or "" */

    /* ---- control vectors (core/cvec.h). `cvec_` is the host table, loaded
     * once at bringup and empty when the feature is off; `cvec_views_` is, per rank, that table
     * uploaded to the rank's card and the ABI view build_ctx hands to a declaring plugin (the
     * view's `dir` is the allocation, so one vector holds both). `cvec_salt_` partitions the
     * prefix cache between the mode on and off, so a conversation's KV and linear state are never
     * reused by a step that would compute different hidden states from the same tokens. */
    ControlVector                  cvec_;
    std::vector<RadControlVector>  cvec_views_;
    uint64_t                       cvec_salt_ = 0;
    /* Runtime default for requests that name no choice (flippable through
     * POST /admin/control-vector). Atomic: the server's admin thread writes it while
     * request threads read it; nothing about the graph changes with it. */
    std::atomic<bool>              cvec_default_{false};
    std::vector<std::string> env_set_;   /* the RADIANCE_* names set when the engine was made */
    bool        report_printed_ = false;
    /* Runtime device identity (PCI address, name, integrated flag, runtime version), read once in
     * start() beside usable_devices. Later reports make no runtime call: after a failed start
     * there may have been a device fault, and on an HTTP thread (/diagnostics) a runtime call can
     * wait on a lock the launch path holds (see DeviceBackend::vram_used). */
    struct DevId { int device = -1; std::string bdf, name; bool integrated = false; };
    std::vector<DevId> dev_ids_;
    std::string        runtime_version_;
    void identify_devices(const std::vector<std::string>& names);
    /* The machine settings startup refuses (diag::incompatible) for the cards the ranks take, asked
     * before anything loads: RAD_E_UNSUPPORTED with each one and its fix on an E line, or under
     * --proceed-with-incompatible-configuration each on a W line and the start goes on. */
    int check_configuration(const std::vector<int>& devs);
    /* Syscall-only facts (diag::probe_process), read once at the first report: the memlock limit
     * cannot change under a running process, and the io_uring probe need not repeat per
     * /diagnostics request. */
    bool        probed_ = false;
    int64_t     memlock_soft_ = -1;
    std::string io_uring_;
    /* The last /diagnostics hardware reading, reused for kDiagReuseNs. The SMU answers the card
     * files under the driver's power-management lock, so a polling client must not become a
     * stream of SMU requests beside the server. */
    mutable std::mutex  diag_mu_;
    mutable diag::Facts diag_facts_;
    mutable uint64_t    diag_read_ns_ = 0;
    /* `serving`: read power-management files only for the cards the ranks serve on
     * (diag::collect's `managed`). */
    diag::Facts    hardware_facts(bool serving) const;
    diag::RunFacts run_facts(bool failed) const;

    /* One dispatch of the tiering seam: retire what landed, credit last step's routing, issue
     * what the heat engine proposes. A member because Rank is private; a no-op on a static plan.
     * The definition explains why a step is the unit. */
    void tiering_dispatch(Rank& r, int step);
    /* The routing histograms landed since the last drain (tiering_dispatch), and the drain alone
     * for --expert-stats where the heat engine is off. */
    void find_histograms(Rank& r, int before_step);
    void xstats_drain(Rank& r, XsTally* xs, int before_step);
    /* --expert-stats: the map and the file, opened before the rank threads start; closed after they
     * stop, crediting what the ring still holds once the streams are drained. */
    int  xstats_open();
    void xstats_close();
    void rank_loop(int rank);
};

}  /* namespace rad */
