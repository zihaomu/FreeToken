#pragma once
#include "runtime/batch_state.h"
#include <condition_variable>
#include <optional>

namespace freetoken {
// Physical B1 state for explicit framework batches. It has no admission queue or prefix policy.
class ExplicitBatchState final : public rad::BatchState {
public:
    bool supports_issue_ahead() const override { return true; }
    bool externally_planned() const override { return true; }
    struct SnapshotPosition { int64_t tokens=0,draft_filled=0; int conv_cursor=0,accepted=0; };
    SnapshotPosition snapshot_position(uint64_t);
    void restore_position(uint64_t,const SnapshotPosition&);
    int authorize(rad::BatchCommand,bool) override;
    int init(rad::Program&,const rad::Config&,rad::IKVManager*,rad::IPrefixCache*,
             const std::vector<rad::BatchBuilder::RankIO>&,const rad::ChunkGeometry*) override;
    const RadBatch* batch_for(int rank) const override { return builder_.batch_for(rank); }
    const std::vector<uint64_t>* step_seqs(const RadBatch* batch) const override { return builder_.step_seqs(batch); }
    rad::AdvanceArgs advance_args(int rank) const override { return builder_.advance_args(rank); }
    void fini() override;
    int add(std::unique_ptr<rad::Request>) override;
    void add_all(std::unique_ptr<rad::Request>* rows,size_t n,int* rc) override {
        for(size_t i=0;i<n;++i) rc[i]=add(std::move(rows[i]));
    }
    uint64_t wake_seq() override;
    void wait_idle(uint64_t,std::chrono::microseconds) override;
    void wake() override;
    int cancel(uint64_t) override;
    rad::Request* find(uint64_t id) override;
    int fail_prefilling(const char* why) override;
    bool running(uint64_t) const override;
    bool committed_seq(uint64_t,uint64_t*,bool*) const override;
    void take_completed(std::vector<uint64_t>*) override;
    int reap(uint64_t) override;
    uint64_t step_record_count() const override { return 0; }
    size_t step_record_capacity() const override { return 0; }
    uint64_t step_records(uint64_t,std::vector<rad::diag::StepRecord>*) const override { return 0; }
    bool has_deferred() const override { return false; }
    void release_deferred() override {}
    void shutdown(const char*,int) override;
    int queue_snapshot(uint64_t*,int) const override { return 0; }
    int requests(ReqStat*,int) const override;
    int set_draft(uint64_t,const int32_t*,int,const int32_t*) override;
    bool encodes_pending() const override { return false; }
    void take_encodes(std::vector<std::shared_ptr<rad::mm::Item>>* out) override { out->clear(); }
    const RadBatch* build_encoder(const std::vector<rad::BatchBuilder::EncoderPart>&,int,const RadEncoderDecl&) override { return nullptr; }
    int64_t reset_prefix_cache() override { return RAD_E_UNSUPPORTED; }
    int64_t evict_prefix_until(const std::function<bool()>&) override { return 0; }
    int preempt_until(const std::function<bool()>&) override { return 0; }
    void set_eog(std::vector<int32_t> ids) override { eog_=std::move(ids); }
    int release_session(const std::string&,uint64_t,int64_t,rad::SessionRelease*) override { return RAD_E_UNSUPPORTED; }
    const RadBatch* step() override;
    void expect_draft(uint64_t,int) override;
    const RadBatch* step_next() override;
    void promote_next() override;
    int commit(StepResult&) override;
    void step_finished() override {}
    int serial_begin(int*,SerialRow*,int) override;
    const RadBatch* serial_batch(int,int) override;
    void serial_end(int) override;
    int block_begin(int,int,int,int32_t,BlockRow*,int,bool) override { return 0; }
    int decoded_seqs() const override { return plan_.n_seq_decode; }
    const RadBatch* block_context(int) override { return nullptr; }
    const RadBatch* block_query(int) override { return nullptr; }
    void block_end() override {}
    const rad::StepPlan& plan() const override { return plan_; }
    const rad::StepPlan& next_plan() const override { return next_; }
    const rad::ChunkGeometry& geometry() const override { return geometry_; }
    rad::DraftController& draft() override { return draft_; }
    rad::LookupGate& lookup_gate() override { return lookup_; }
    rad::SchedMetrics metrics() const override;
    void publish(bool) override {}
    rad::SchedMetrics observed_metrics() const override { return metrics(); }
    int observed_requests(ReqStat* out,int max) const override { return requests(out,max); }
    void set_stage_rows(int64_t) override {}
    void set_span_min(int64_t) override {}
    bool prefill_continues() const override;
#ifdef RAD_ENABLE_KV_ROLLBACK_TEST_HOOK
    void set_kv_rollback_test_hook(std::function<void(uint64_t,int,int,int64_t,bool)> hook) override { rollback_hook_=std::move(hook); }
#endif
private:
    std::mutex& batch_mutex() override { return mu_; }
    void finish(rad::ReqState,const char*);
    bool live() const;
    rad::Config config_{};
    rad::IKVManager* kv_=nullptr;
    rad::ChunkGeometry geometry_{};
    rad::BatchBuilder builder_;
    std::unique_ptr<rad::Request> request_;
    rad::StepPlan plan_,next_,serial_plan_;
    std::optional<rad::BatchCommand> prepared_,next_prepared_;
    rad::BatchCommand next_active_;
    rad::BatchCommand active_;
    int step_=0,conv_cursor_=0,conv_window_=1,n_draft_=0,expected_draft_=0;
    int32_t draft_ids_[rad::RAD_SCHED_MAX_SPEC]={};
    int64_t draft_filled_=0;
    bool open_=false,next_open_=false,serial_held_=false,closed_=false;
    struct History {
        int64_t position=0,base_position=0;
        int rows=0,hidden_row=0,token_row=0,output_row=-1,lead=0;
        bool device=false,drafting=false;
    } history_;
    CommittedSeq committed_{};
    std::vector<uint64_t> completed_;
    std::vector<int32_t> eog_;
    rad::DraftController draft_;
    rad::LookupGate lookup_;
    rad::SchedMetrics metrics_{};
    mutable std::mutex mu_;
    std::condition_variable cv_;
    uint64_t woke_=0;
#ifdef RAD_ENABLE_KV_ROLLBACK_TEST_HOOK
    std::function<void(uint64_t,int,int,int64_t,bool)> rollback_hook_;
#endif
};
}
