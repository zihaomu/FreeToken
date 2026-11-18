#pragma once
// Runtime-facing batch state. Scheduling policy is supplied by the framework adapter.
#include "sched/batch.h"
#include "sched/draft.h"
#include "sched/lookup.h"
#include "sched/metrics.h"
#include "diag/diagnose.h"
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
namespace rad {
struct BatchCommand {
    uint64_t sequence = 0;
    bool prefill = false;
    int max_committed = 0;
    int draft_depth = 0;
    int64_t reserved_through = 0;
    std::vector<int32_t> input_ids;
};
class BatchState {
public:
    virtual bool supports_issue_ahead() const { return true; }
    virtual bool externally_planned() const { return false; }
    virtual int authorize(BatchCommand, bool = false) { return RAD_E_UNSUPPORTED; }
    struct StepResult {
        int             n_seq = 0;
        const int32_t*  n_accepted = nullptr;  /* [n_seq] drafted tokens accepted, 0..n_spec */
        const int32_t*  token = nullptr;       /* [n_seq] token sampled at the accepted position */
        const uint8_t*  eos = nullptr;         /* [n_seq] 1 when the sequence ended */
        int             n_prefill_done = 0;   /* prompts this step fully placed */
        int             n_finished = 0;       /* requests that ended during this step */
        /* Logprobs, null on a step where no request asked: entry i's j-th emitted token (its
         * accepted drafts first, then the sampled token) is lp[first_row[i] + j]. */
        const TokenLogprobs* lp = nullptr;
        const int32_t*       first_row = nullptr;  /* [n_seq] */
        /* A tree step's accepted nodes, [n_seq][RAD_SCHED_MAX_SPEC]: entry i's t-th accepted
         * draft is draft[path[i][t] - 1]. Null on a chain step, where it is draft[t]. */
        const int32_t*       path = nullptr;
    };
    struct CommittedSeq {
        uint64_t seq = 0;
        uint64_t n_out = 0;
        bool     live = false;
    };
    struct ReqStat {
        uint64_t id = 0;
        int      state = 0;            /* ReqState, as an int -- the caller has the enum */
        bool     running = false;
        int64_t  prompt_tokens = 0;
        int64_t  computed_tokens = 0;  /* of the prompt, how much is in the KV: prefill progress */
        int64_t  cached_tokens = 0;    /* of those, how many came from the prefix cache */
        int64_t  output_tokens = 0;
        int64_t  max_tokens = 0;
        int64_t  ctx_tokens = 0;       /* prompt ++ output */
        /* This request's share of the paged pools: the blocks it holds and the tokens they cover.
         * It differs from `ctx_tokens` (what it uses) by a partial block and by whatever a
         * rejected draft left. */
        int64_t  kv_blocks = 0;
        int64_t  kv_tokens = 0;
        int32_t  n_draft = 0;          /* the proposal standing for the next step */
        int32_t  slot = -1;
        double   age_s = 0;
        double   ttft_s = -1;          /* negative until the first token */
        double   prefill_s = -1;       /* first token minus admission; negative until both */
        double   decode_tps = 0;       /* this request's own, since its first token */
    };
    struct SerialRow {
        uint64_t seq = 0;
        int32_t  slot = -1;
        int32_t  card = -1;
    };
    struct BlockRow {
        uint64_t seq = 0;
        int32_t  slot = -1;
        int32_t  card = -1;
    };
    virtual ~BatchState() = default;
    virtual int init(Program&, const Config&, IKVManager*, IPrefixCache*,
                     const std::vector<BatchBuilder::RankIO>&, const ChunkGeometry* = nullptr) = 0;
    virtual const RadBatch* batch_for(int) const = 0;
    virtual const std::vector<uint64_t>* step_seqs(const RadBatch*) const = 0;
    virtual AdvanceArgs advance_args(int) const = 0;
    virtual void fini() = 0;
    virtual int add(std::unique_ptr<Request>) = 0;
    virtual void add_all(std::unique_ptr<Request>*, size_t, int*) = 0;
    virtual uint64_t wake_seq() = 0;
    virtual void wait_idle(uint64_t, std::chrono::microseconds) = 0;
    virtual void wake() = 0;
    virtual int cancel(uint64_t) = 0;
    virtual Request* find(uint64_t) = 0;
    virtual int fail_prefilling(const char*) = 0;
    virtual bool running(uint64_t) const = 0;
    virtual bool committed_seq(uint64_t, uint64_t*, bool*) const = 0;
    virtual void take_completed(std::vector<uint64_t>*) = 0;
    virtual int reap(uint64_t) = 0;
    virtual uint64_t step_record_count() const = 0;
    virtual size_t step_record_capacity() const = 0;
    virtual uint64_t step_records(uint64_t, std::vector<diag::StepRecord>*) const = 0;
    virtual bool has_deferred() const = 0;
    virtual void release_deferred() = 0;
    virtual void shutdown(const char*, int = RAD_E_STATE) = 0;
    virtual int queue_snapshot(uint64_t*, int) const = 0;
    virtual int requests(ReqStat*, int) const = 0;
    virtual int set_draft(uint64_t, const int32_t*, int, const int32_t* = nullptr) = 0;
    virtual bool encodes_pending() const = 0;
    virtual void take_encodes(std::vector<std::shared_ptr<mm::Item>>*) = 0;
    virtual const RadBatch* build_encoder(const std::vector<BatchBuilder::EncoderPart>&, int,
                                           const RadEncoderDecl&) = 0;
    virtual int64_t reset_prefix_cache() = 0;
    virtual int64_t evict_prefix_until(const std::function<bool()>&) = 0;
    virtual int preempt_until(const std::function<bool()>&) = 0;
    virtual void set_eog(std::vector<int32_t>) = 0;
    virtual int release_session(const std::string&, uint64_t, int64_t, SessionRelease*) = 0;
    virtual const RadBatch* step() = 0;
    virtual void expect_draft(uint64_t, int) = 0;
    virtual const RadBatch* step_next() = 0;
    virtual void promote_next() = 0;
    virtual int commit(StepResult&) = 0;
    virtual void step_finished() = 0;
    virtual int serial_begin(int*, SerialRow*, int) = 0;
    virtual const RadBatch* serial_batch(int, int) = 0;
    virtual void serial_end(int) = 0;
    virtual int block_begin(int, int, int, int32_t, BlockRow*, int, bool = false) = 0;
    virtual int decoded_seqs() const = 0;
    virtual const RadBatch* block_context(int) = 0;
    virtual const RadBatch* block_query(int) = 0;
    virtual void block_end() = 0;
    virtual const StepPlan& plan() const = 0;
    virtual const StepPlan& next_plan() const = 0;
    virtual const ChunkGeometry& geometry() const = 0;
    virtual DraftController& draft() = 0;
    virtual LookupGate& lookup_gate() = 0;
    virtual SchedMetrics metrics() const = 0;
    virtual void publish(bool) = 0;
    virtual SchedMetrics observed_metrics() const = 0;
    virtual int observed_requests(ReqStat*, int) const = 0;
    virtual void set_stage_rows(int64_t) = 0;
    virtual void set_span_min(int64_t) = 0;
    virtual bool prefill_continues() const = 0;
    template<class Fn> auto locked(Fn&& fn) {
        std::lock_guard<std::mutex> lock(batch_mutex()); return fn();
    }
    std::unique_lock<std::mutex> hold() { return std::unique_lock<std::mutex>(batch_mutex()); }
    static bool short_block_on(bool was_on, int64_t decoded, int64_t from) {
        if (decoded >= from) return true;
        if (decoded <= 1 || decoded + 2 <= from) return false;
        return was_on;
    }
#ifdef RAD_ENABLE_KV_ROLLBACK_TEST_HOOK
    virtual void set_kv_rollback_test_hook(std::function<void(uint64_t,int,int,int64_t,bool)>) = 0;
#endif
protected:
    virtual std::mutex& batch_mutex() = 0;

};
}
