#include "batch_state.h"
#include "server/sink.h"
#include <algorithm>
#include <stdexcept>

namespace freetoken {
using namespace rad;
bool ExplicitBatchState::live() const { return request_ && request_->state==ReqState::Running; }
int ExplicitBatchState::init(Program& p,const Config& c,IKVManager* kv,IPrefixCache*,
        const std::vector<BatchBuilder::RankIO>& ranks,const ChunkGeometry* geometry) {
    if(c.max_seqs!=1 || !kv || (c.n_spec!=0 && c.n_spec!=3)) return RAD_E_UNSUPPORTED;
    config_=c;kv_=kv;
    if(geometry) geometry_=*geometry; else RAD_TRY(chunk_geometry_resolve(p,c,&geometry_));
    KVGeom dimensions=KVGeom::of(p.kv_groups);
    for(size_t i=0;i<dimensions.g.size();++i) {
        const auto width=kv->state_index_width(int32_t(i));
        if(width>0) dimensions.g[i].sidx_width=width;
    }
    for(const auto& group:p.kv_groups) if(group.name=="kv_gdn_conv")
        conv_window_=int(conv_window_entries(group.decl,c.n_spec));
    if(conv_window_<1)return RAD_E_SHAPE;
    DraftController::Params params;params.max_depth=c.n_spec;
    RAD_TRY(draft_.configure(params));
    return builder_.init(p,c,geometry_,dimensions,ranks);
}
void ExplicitBatchState::fini() {
    if(kv_ && request_) kv_->free_sequence(request_->id);
    request_.reset();builder_.fini();kv_=nullptr;
}
ExplicitBatchState::SnapshotPosition ExplicitBatchState::snapshot_position(uint64_t id) {
    std::lock_guard<std::mutex> lock(mu_);
    if(!live() || request_->id!=id || open_ || next_open_ || prepared_ || next_prepared_ ||
       serial_held_ || !active_.prefill || !request_->output.empty() || request_->n_computed%4)
        throw std::runtime_error("snapshot requires a fenced, whole-page prefill boundary");
    return {request_->n_computed,draft_filled_,conv_cursor_,request_->n_accepted};
}
void ExplicitBatchState::restore_position(uint64_t id,const SnapshotPosition& p) {
    std::lock_guard<std::mutex> lock(mu_);
    if(!live() || request_->id!=id || request_->n_computed || open_ || prepared_ ||
       p.tokens<=0 || p.tokens>=int64_t(request_->prompt.size()) || p.tokens%4)
        throw std::runtime_error("restore requires a fresh lease and a partial prompt prefix");
    request_->n_computed=p.tokens;request_->n_accepted=p.accepted;
    conv_cursor_=p.conv_cursor;draft_filled_=p.draft_filled;n_draft_=0;
}
int ExplicitBatchState::add(std::unique_ptr<Request> request) {
    std::lock_guard<std::mutex> lock(mu_);
    if(closed_ || request_ || !request || request->prompt.empty() || request->max_tokens<1 ||
       request->prompt.size()+1>size_t(config_.max_ctx) || request->sp.n_probs>=0)
        return RAD_E_STATE;
    if(!kv_->can_ever_fit(int64_t(request->prompt.size())+1)) return RAD_E_FULL;
    RAD_TRY(kv_->add_sequence(request->id));
    request->state=ReqState::Running;request->n_computed=0;
    request->output.reserve(size_t(request->max_tokens));
    request->may_speculate=config_.n_spec>0;request->speculation_fixed=true;
    request_=std::move(request);conv_cursor_=n_draft_=0;draft_filled_=0;
    prepared_.reset();next_prepared_.reset();open_=next_open_=serial_held_=false;expected_draft_=0;history_={};committed_={};
    ++metrics_.admitted;++woke_;cv_.notify_all();return RAD_OK;
}
int ExplicitBatchState::authorize(BatchCommand command,bool ahead) {
    std::lock_guard<std::mutex> lock(mu_);
    if(!request_ || command.sequence!=request_->id || (!ahead && (prepared_ || open_)) ||
       command.reserved_through<0 || command.reserved_through>config_.max_ctx ||
       command.max_committed<0 || command.max_committed>4 ||
       (command.draft_depth!=0 && command.draft_depth!=3) || command.draft_depth>config_.n_spec)
        return RAD_E_STATE;
    if(!command.prefill && (!command.input_ids.empty() || command.max_committed<1))
        return RAD_E_INVAL;
    if(ahead) {
        if(!live() || !open_ || next_open_ || next_prepared_ || command.prefill ||
           plan_.e.empty() || !plan_.e[0].produces_token) return 1;
        const auto& e=plan_.e[0];
        const int drafts=command.draft_depth?expected_draft_:0;
        if(drafts<0 || drafts>command.draft_depth) return RAD_E_STATE;
        const int64_t bound=request_->n_computed+e.n_tokens;
        if(bound+1+drafts+std::max(command.draft_depth-1,0)>command.reserved_through)
            return RAD_E_FULL;
        RAD_TRY(kv_->ensure_from(request_->id,command.reserved_through,request_->n_computed));
        next_prepared_=std::move(command);return RAD_OK;
    }
    if(!live()) return RAD_OK;
    const auto& r=*request_;
    if(command.prefill) {
        if(command.input_ids.empty() || command.input_ids.size()>size_t(config_.max_tok) || command.max_committed>1 ||
           r.n_computed+int64_t(command.input_ids.size())>int64_t(r.prompt.size()) ||
           !std::equal(command.input_ids.begin(),command.input_ids.end(),r.prompt.begin()+r.n_computed))
            return RAD_E_INVAL;
        const bool last=r.n_computed+int64_t(command.input_ids.size())==int64_t(r.prompt.size());
        if((command.max_committed>0)!=last) return RAD_E_INVAL;
    } else if(!command.input_ids.empty() || r.output.empty() ||
              r.n_computed>=int64_t(r.prompt.size()+r.output.size())) return RAD_E_STATE;
    const int64_t rows=command.prefill?int64_t(command.input_ids.size()):1+(command.draft_depth?n_draft_:0);
    const int64_t tail=command.max_committed>0?std::max(command.draft_depth-1,0):0;
    if(r.n_computed+rows+tail>command.reserved_through) return RAD_E_FULL;
    RAD_TRY(kv_->ensure_from(r.id,command.reserved_through,r.n_computed));
    prepared_=std::move(command);return RAD_OK;
}
const RadBatch* ExplicitBatchState::step() {
    std::lock_guard<std::mutex> lock(mu_);
    if(!live() || !prepared_ || open_) return nullptr;
    active_=std::move(*prepared_);prepared_.reset();plan_.reset();
    auto& r=*request_;
    StepEntry e;e.req=&r;e.seq=r.id;e.slot=e.card=0;
    e.ctx_len=int32_t(r.n_computed);e.is_prefill=active_.prefill;
    e.n_spec=e.is_prefill?0:(active_.draft_depth?n_draft_:0);
    e.n_tokens=e.is_prefill?int32_t(active_.input_ids.size()):1+e.n_spec;
    e.produces_token=active_.max_committed>0;e.n_accepted=1+r.n_accepted;e.conv_cursor=conv_cursor_;
    e.draft=e.n_spec?draft_ids_:nullptr;e.n_draft=e.n_spec;
    e.dev_tok=e.n_tokens-1-e.n_spec;e.dev_out=e.produces_token?0:-1;e.dev_aux=e.n_spec;
    r.n_draft=e.n_spec;r.draft=e.draft;
    plan_.e.push_back(e);plan_.n_tok=e.n_tokens;plan_.n_spec=e.n_spec;
    plan_.phase=e.is_prefill?RAD_PHASE_PREFILL:RAD_PHASE_DECODE;
    plan_.n_seq_decode=e.is_prefill?0:1;plan_.n_tok_decode=e.is_prefill?0:e.n_tokens;
    plan_.max_q_len=e.n_tokens;plan_.max_ctx_len=e.ctx_len;
    const auto* batch=builder_.build(plan_,*kv_,step_);
    if(!batch) throw std::runtime_error("explicit batch staging failed");
    ++step_;open_=true;return batch;
}
void ExplicitBatchState::expect_draft(uint64_t id,int count) {
    std::lock_guard<std::mutex> lock(mu_);
    if(request_ && request_->id==id) expected_draft_=std::clamp(count,0,config_.n_spec);
}
const RadBatch* ExplicitBatchState::step_next() {
    std::lock_guard<std::mutex> lock(mu_);
    if(!live() || !open_ || next_open_ || !next_prepared_) return nullptr;
    const auto& current=plan_.e.at(0);auto& r=*request_;
    next_active_=std::move(*next_prepared_);next_prepared_.reset();next_.reset();
    StepEntry e;e.req=&r;e.seq=r.id;e.slot=e.card=0;e.is_prefill=false;e.produces_token=true;
    e.ctx_len=int32_t(r.n_computed+current.n_tokens);
    e.n_spec=next_active_.draft_depth?expected_draft_:0;e.n_tokens=1+e.n_spec;
    e.n_accepted=1;e.conv_cursor=conv_cursor_;e.draft=draft_ids_;e.n_draft=e.n_spec;
    e.dev=true;e.dev_tok=0;e.dev_out=0;e.dev_aux=e.n_spec;
    next_.e.push_back(e);next_.phase=RAD_PHASE_DECODE;next_.n_tok=e.n_tokens;
    next_.n_spec=e.n_spec;next_.n_seq_decode=1;next_.n_tok_decode=e.n_tokens;
    next_.max_q_len=e.n_tokens;next_.max_ctx_len=e.ctx_len;
    const auto* batch=builder_.build(next_,*kv_,step_);
    if(!batch) throw std::runtime_error("explicit issue-ahead staging failed");
    next_open_=true;++metrics_.steps_ahead;return batch;
}
void ExplicitBatchState::promote_next() {
    std::lock_guard<std::mutex> lock(mu_);
    if(!next_open_) return;
    plan_=std::move(next_);next_.reset();active_=std::move(next_active_);
    next_open_=false;open_=true;++step_;
}
void ExplicitBatchState::finish(ReqState state,const char* reason) {
    if(!request_ || request_->state!=ReqState::Running) return;
    auto& r=*request_;r.state=state;r.finish_reason=reason;completed_.push_back(r.id);
    const auto type=state==ReqState::Cancelled?server::Finish::Cancelled:
        state==ReqState::Failed?server::Finish::Error:
        std::string(reason)=="length"?server::Finish::Length:server::Finish::Stop;
    server::sink_finish(r.sink,type);
    if(state==ReqState::Cancelled) ++metrics_.cancelled;
    else if(state==ReqState::Failed) ++metrics_.failed;
    else ++metrics_.finished;
}
int ExplicitBatchState::commit(StepResult& result) {
    std::lock_guard<std::mutex> lock(mu_);
    if(!open_ || !request_ || result.n_seq!=1) return RAD_E_STATE;
    open_=false;result.n_prefill_done=result.n_finished=0;
    auto& r=*request_;const auto& e=plan_.e.at(0);
    if(!live()) { committed_={r.id,uint64_t(r.output.size()),false};return RAD_OK; }
    const int accepted=e.is_prefill?0:std::clamp(result.n_accepted?result.n_accepted[0]:0,0,e.n_spec);
    const auto valid=r.n_computed+(e.is_prefill?e.n_tokens:1+accepted);
    if(!e.is_prefill && accepted<e.n_spec && !next_open_) {
#ifdef RAD_ENABLE_KV_ROLLBACK_TEST_HOOK
        if(rollback_hook_) rollback_hook_(r.id,e.n_spec,accepted,valid,true);
#endif
        RAD_TRY(kv_->rollback(r.id,valid));
#ifdef RAD_ENABLE_KV_ROLLBACK_TEST_HOOK
        if(rollback_hook_) rollback_hook_(r.id,e.n_spec,accepted,valid,false);
#endif
    }
    // Conv read cursors follow the ABI window, not the number of emitted (possibly trimmed) tokens.
    const int conv_window=conv_window_;
    if(e.produces_token) conv_cursor_=(conv_cursor_+1+accepted)%conv_window;
    r.n_computed=valid;r.n_accepted=accepted;
    const size_t before=r.output.size();bool overflow=false;
    const int64_t timestamp=server::sink_now_ns();
    auto emit=[&](int32_t token,bool draft) {
        if(r.output.size()>=size_t(r.max_tokens) || r.prompt.size()+r.output.size()>=size_t(config_.max_ctx)) return;
        if(r.output.size()-before>=size_t(active_.max_committed))
            throw std::runtime_error("explicit step exceeded framework commit credit");
        r.output.push_back(token);if(!server::sink_push(r.sink,token,nullptr,timestamp,draft)) overflow=true;
    };
    if(e.produces_token) {
        for(int i=0;i<accepted;++i) emit(draft_ids_[i],true);
        if(result.token) emit(result.token[0],false);
    }
    if(e.is_prefill) {
        metrics_.prefill_tokens+=e.n_tokens;++metrics_.steps_prefill;
        server::sink_prompt_progress(r.sink,int32_t(r.n_computed),int32_t(r.prompt.size()),0);
        if(r.n_computed>=int64_t(r.prompt.size())) {
            server::sink_prompt_stats(r.sink,int32_t(r.prompt.size()),0);result.n_prefill_done=1;
        }
    } else {
        metrics_.decode_tokens+=1+accepted;++metrics_.steps_decode;
        if(e.n_spec) {metrics_.draft_tokens+=e.n_spec;metrics_.draft_accepted+=accepted;
            draft_.observe(e.n_spec,accepted);server::sink_drafts(r.sink,e.n_spec,accepted);}
    }
    n_draft_=0;r.n_draft=0;r.draft=nullptr;++metrics_.steps;
    if(overflow) finish(ReqState::Failed,"sink_overflow");
    else if(server::sink_cancelled(r.sink)) finish(ReqState::Cancelled,"cancelled");
    else if(r.output.size()>=size_t(r.max_tokens) || r.prompt.size()+r.output.size()>=size_t(config_.max_ctx))
        finish(ReqState::Finished,"length");
    else if(result.eos && result.eos[0] && !r.ignore_eos) finish(ReqState::Finished,"stop");
    result.n_finished=live()?0:1;committed_={r.id,uint64_t(r.output.size()),live()};return RAD_OK;
}
int ExplicitBatchState::serial_begin(int* rounds,SerialRow* out,int max) {
    std::lock_guard<std::mutex> lock(mu_);history_={};serial_held_=false;
    if(!rounds || *rounds<=0 || !out || max<1 || !live() || plan_.e.empty()) return 0;
    const auto& e=plan_.e.at(0);auto& r=*request_;
    const int64_t verified_start=int64_t(e.ctx_len)+e.n_tokens-1-e.n_spec;
    const int64_t head_max=verified_start+e.n_spec;
    const int64_t owed=e.produces_token?verified_start:int64_t(e.ctx_len)+e.n_tokens;
    const int64_t start=std::min(std::max(draft_filled_,int64_t(e.ctx_len)),owed);
    history_.position=start;history_.lead=int(owed-start);
    history_.rows=history_.lead+(e.produces_token?1+e.n_spec:0);
    history_.hidden_row=int(start-e.ctx_len);history_.token_row=int(verified_start-e.ctx_len);
    history_.output_row=e.dev_out;history_.device=e.produces_token;history_.base_position=head_max;
    history_.drafting=e.produces_token && active_.draft_depth>0;
    *rounds=history_.drafting?std::min(*rounds,active_.draft_depth):1;
    if(history_.drafting && head_max+*rounds>active_.reserved_through)
        throw std::runtime_error("MTP head exceeded framework state lease");
    serial_held_=history_.rows>0;
    if(history_.drafting) {out[0]={r.id,0,0};return 1;}
    return 0;
}
const RadBatch* ExplicitBatchState::serial_batch(int round,int step) {
    std::lock_guard<std::mutex> lock(mu_);
    if(!request_ || !serial_held_ || round<0 || round==1 || (round>0 && !history_.drafting)) return nullptr;
    serial_plan_.reset();serial_plan_.phase=RAD_PHASE_DECODE;serial_plan_.n_spec=config_.n_spec;
    serial_plan_.draft_pass=round==0?-1:round;serial_plan_.tok_shift=1;
    StepEntry e;e.req=request_.get();e.seq=request_->id;e.slot=e.card=0;
    e.n_tokens=round==0?history_.rows:1;
    e.ctx_len=int32_t(round==0?history_.position:history_.base_position+round-1);
    e.n_accepted=1;e.produces_token=round>0;e.hidden_row=round==0?history_.hidden_row:-1;
    e.draft_out=round==0 && history_.drafting;e.dev=history_.device;
    e.dev_tok=history_.token_row;e.dev_out=history_.output_row;e.dev_aux=round==0?history_.lead:0;
    serial_plan_.e.push_back(e);serial_plan_.n_tok=e.n_tokens;
    return builder_.build(serial_plan_,*kv_,step);
}
void ExplicitBatchState::serial_end(int rounds_done) {
    std::lock_guard<std::mutex> lock(mu_);
    if(request_ && serial_held_ && (!history_.device || rounds_done>=1 || !history_.drafting))
        draft_filled_=std::max(draft_filled_,history_.device?request_->n_computed:history_.position+history_.rows);
    serial_held_=false;
}
int ExplicitBatchState::set_draft(uint64_t id,const int32_t* tokens,int n,const int32_t* parents) {
    std::lock_guard<std::mutex> lock(mu_);
    if(!request_ || request_->id!=id || parents || n<0 || n>config_.n_spec) return RAD_E_INVAL;
    if(next_open_) {
        const int expected=next_.e.at(0).n_spec;
        if(n<expected)return RAD_E_STATE;
        n=expected;
    }
    n_draft_=n;std::copy_n(tokens,n,draft_ids_);request_->n_draft=n;
    request_->draft=n?draft_ids_:nullptr;return RAD_OK;
}
uint64_t ExplicitBatchState::wake_seq() {std::lock_guard<std::mutex> l(mu_);return woke_;}
void ExplicitBatchState::wait_idle(uint64_t seen,std::chrono::microseconds wait) {
    std::unique_lock<std::mutex> l(mu_);cv_.wait_for(l,wait,[&]{return woke_!=seen;});
}
void ExplicitBatchState::wake() {std::lock_guard<std::mutex> l(mu_);++woke_;cv_.notify_all();}
int ExplicitBatchState::cancel(uint64_t id) {
    std::lock_guard<std::mutex> l(mu_);if(!request_ || request_->id!=id) return RAD_E_NOTFOUND;
    finish(ReqState::Cancelled,"cancelled");prepared_.reset();++woke_;cv_.notify_all();return RAD_OK;
}
Request* ExplicitBatchState::find(uint64_t id) {std::lock_guard<std::mutex> l(mu_);return request_&&request_->id==id?request_.get():nullptr;}
bool ExplicitBatchState::running(uint64_t id) const {std::lock_guard<std::mutex> l(mu_);return live()&&request_->id==id;}
bool ExplicitBatchState::committed_seq(uint64_t id,uint64_t* count,bool* running) const {
    std::lock_guard<std::mutex> l(mu_);if(committed_.seq!=id)return false;
    *count=committed_.n_out;*running=committed_.live;return true;
}
void ExplicitBatchState::take_completed(std::vector<uint64_t>* out) {std::lock_guard<std::mutex> l(mu_);out->clear();out->swap(completed_);}
int ExplicitBatchState::reap(uint64_t id) {
    std::lock_guard<std::mutex> l(mu_);
    if(!request_ || request_->id!=id || open_ || next_open_ || serial_held_ || live()) return RAD_E_STATE;
    kv_->free_sequence(id);request_.reset();prepared_.reset();return RAD_OK;
}
int ExplicitBatchState::fail_prefilling(const char* why) {shutdown(why,RAD_E_STATE);return 1;}
void ExplicitBatchState::shutdown(const char* why,int status) {
    std::lock_guard<std::mutex> l(mu_);closed_=true;
    if(live()) {server::sink_fail(request_->sink,status,why);finish(ReqState::Failed,"error");}
    ++woke_;cv_.notify_all();
}
int ExplicitBatchState::requests(ReqStat* out,int max) const {
    std::lock_guard<std::mutex> l(mu_);if(!request_ || max<1)return 0;
    out[0]={};out[0].id=request_->id;out[0].running=live();out[0].state=int(request_->state);
    out[0].prompt_tokens=request_->prompt.size();out[0].computed_tokens=request_->n_computed;
    out[0].output_tokens=request_->output.size();out[0].max_tokens=request_->max_tokens;return 1;
}
SchedMetrics ExplicitBatchState::metrics() const {std::lock_guard<std::mutex> l(mu_);auto m=metrics_;m.running=live()?1:0;return m;}
bool ExplicitBatchState::prefill_continues() const {
    std::lock_guard<std::mutex> l(mu_);return live() && request_->n_computed<int64_t(request_->prompt.size());
}
}
