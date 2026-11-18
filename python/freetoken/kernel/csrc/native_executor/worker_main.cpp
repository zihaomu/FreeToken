/* Bounded device execution. FreeToken owns graph, request, state and placement policy. */
#include "engine.h"
#include "qwen_plan.h"
#ifdef FREETOKEN_EXPLICIT_BATCHES
#include "batch_state.h"
#include "state_storage.h"
#include "residency.h"
#endif
#include <fstream>
#include "device/device.h"
#include "fleet/config.h"
#include "runtime/external_steps.h"
#include "server/sink.h"
#include <cerrno>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <sys/eventfd.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace ft_executor {
using rad::Json;
constexpr uint32_t magic=0x32585446, max_payload=8*1024*1024;
#ifdef FREETOKEN_EXPLICIT_BATCHES
constexpr bool bridge_only=false;
#else
constexpr bool bridge_only=true;
#endif
enum Kind : uint16_t { Ready=1,Compile=2,Reserve=3,Submit=4,Complete=5,
                      Revoke=6,Release=7,Inspect=8,Shutdown=9,Error=10,
                      Snapshot=11,Restore=12,DropSnapshot=13 };
struct Header {
    uint32_t signature=magic;
    uint16_t version=3,kind=0;
    uint32_t bytes=0,reserved=0;
    uint64_t uid=0,epoch=0;
};
static_assert(sizeof(Header)==32);

bool transfer(int fd,void* data,size_t size,bool sending) {
    auto* p=static_cast<char*>(data);
    while(size) {
        const auto n=sending?send(fd,p,size,MSG_NOSIGNAL):recv(fd,p,size,0);
        if(n<0&&errno==EINTR)continue;
        if(n<=0)return false;
        p+=n;size-=size_t(n);
    }
    return true;
}
bool emit(int fd,Kind kind,const Json& value,uint64_t uid=0,uint64_t epoch=0) {
    auto bytes=value.dump();if(bytes.size()>max_payload)return false;
    Header h;h.kind=kind;h.bytes=uint32_t(bytes.size());h.uid=uid;h.epoch=epoch;
    return transfer(fd,&h,sizeof(h),true)&&transfer(fd,bytes.data(),bytes.size(),true);
}
uint64_t number(const Json& object,const char* key,uint64_t minimum=0) {
    const auto& value=object.at(key);
    if(!value.is_number_integer())throw std::runtime_error("integer field required");
    if(!value.is_number_unsigned()&&value.get<int64_t>()<0)
        throw std::runtime_error("integer out of range");
    const auto n=value.get<uint64_t>();
    if(n<minimum)throw std::runtime_error("integer out of range");
    return n;
}
Json cursor(int64_t committed,int64_t computed,int32_t pending=-1) {
    Json out={{"committed_tokens",committed},{"computed_tokens",computed}};
    out["pending_token"]=pending<0?Json(nullptr):Json(pending);return out;
}
struct Active {
    uint64_t uid=0,epoch=0,native_id=0,state_id=0;
    rad::Request* request=nullptr;
    std::shared_ptr<rad::server::Sink> sink;
};
struct Credit { Json plan;std::shared_ptr<Active> active; };

class Controller final : public rad::ExternalStepControl {
public:
    Controller(uint64_t generation,int window):generation_(generation),window_(window),
        event_(eventfd(0,EFD_CLOEXEC|EFD_NONBLOCK)) {
        if(event_<0||window<1||window>2)throw std::runtime_error("invalid credit window");
    }
    ~Controller() override { close(event_); }
    int event()const{return event_;}
    uint64_t generation()const{return generation_;}
    bool pipeline()const override{return window_==2;}
    Json handle(uint64_t id,uint64_t version=1)const {
        return {{"worker_generation",generation_},{"model_generation",1},
                {"resource_id",id},{"version",version}};
    }
    void bind(std::shared_ptr<Active> active) {
        std::lock_guard<std::mutex> lock(mu_);
        if(held_||active_)throw std::runtime_error("state allocation is busy");
        active_=std::move(active);revoked_=false;last_id_=0;state_version_=1;cursor_=cursor(0,0);
    }
    void unbind() {
        std::lock_guard<std::mutex> lock(mu_);
        if(held_||issued_)throw std::runtime_error("state has in-flight references");
        active_.reset();
    }
    void submit(Json plan) {
        std::lock_guard<std::mutex> lock(mu_);
        if(!active_||revoked_||closed_||held_>=size_t(window_))
            throw std::runtime_error("credit window exhausted or revoked");
        const auto& ticket=plan.at("ticket");const auto& request=ticket.at("request");
        if(number(request,"uid")!=active_->uid||number(request,"epoch",1)!=active_->epoch||
           number(ticket,"worker_generation",1)!=generation_||number(ticket,"credit_epoch",1)!=1)
            throw std::runtime_error("stale executor request generation");
        const auto id=number(ticket,"step_id",1);
        if(id!=last_id_+1)throw std::runtime_error("non-contiguous executor step id");
        const auto& state=plan.at("state");const auto& executable=plan.at("executable");
        for(const auto* h:{&state,&executable})
            if(number(*h,"worker_generation",1)!=generation_||number(*h,"model_generation",1)!=1)
                throw std::runtime_error("foreign executor resource");
        if(number(state,"resource_id",1)!=active_->state_id||number(executable,"resource_id",1)!=1)
            throw std::runtime_error("unknown executor resource");
        if(number(executable,"version",1)!=1)throw std::runtime_error("stale executable version");
        const auto state_version=number(state,"version",1);
        if(!plan.at("predecessor").is_null()){
            const auto& predecessor=plan.at("predecessor");
            if(number(predecessor,"step_id",1)!=last_id_||
               number(predecessor,"worker_generation",1)!=generation_||
               number(predecessor,"credit_epoch",1)!=1||
               number(predecessor.at("request"),"uid")!=active_->uid||
               number(predecessor.at("request"),"epoch",1)!=active_->epoch)
                throw std::runtime_error("invalid executor predecessor");
        }else if(state_version!=state_version_){
            throw std::runtime_error("stale state version without a predecessor");
        }
        if(number(plan,"max_committed")>4||number(plan,"reserved_through")>34816)
            throw std::runtime_error("executor token bound exceeded");
        const auto phase=plan.at("phase").get<std::string>();
        if(phase!="prefill"&&phase!="decode"&&phase!="mtp")
            throw std::runtime_error("unknown executor phase");
        if(phase=="mtp"&&number(plan,"draft_depth")!=3)
            throw std::runtime_error("unsupported MTP depth");
        queued_.push_back({id,std::make_shared<Credit>(Credit{std::move(plan),active_})});
        ++held_;last_id_=id;cv_.notify_one();
    }
    bool acquire(rad::ExternalPermit& out,bool wait,const std::atomic<bool>& stopping)override {
        std::unique_lock<std::mutex> lock(mu_);
        while(wait&&queued_.empty()&&!closed_&&!stopping.load(std::memory_order_acquire))
            cv_.wait_for(lock,std::chrono::milliseconds(10));
        if(queued_.empty()||closed_||stopping.load(std::memory_order_acquire))return false;
        out=std::move(queued_.front());queued_.pop_front();++issued_;return true;
    }
    int prepare(const rad::ExternalPermit& permit,rad::BatchState& state,bool ahead)override {
        if(!state.externally_planned())return RAD_OK;
        const auto credit=std::static_pointer_cast<Credit>(permit.payload);
        const auto& p=credit->plan;
        rad::BatchCommand command;
        command.sequence=credit->active->native_id;
        command.prefill=p.at("phase").get<std::string>()=="prefill";
        command.input_ids=p.at("input_ids").get<std::vector<int32_t>>();
        command.max_committed=int(number(p,"max_committed"));
        command.reserved_through=int64_t(number(p,"reserved_through"));
        command.draft_depth=int(number(p,"draft_depth"));
        return state.authorize(std::move(command),ahead);
    }
    void defer(rad::ExternalPermit permit)override {
        std::lock_guard<std::mutex> lock(mu_);
        --issued_;
        if(revoked_ || closed_) {
            completed_.push_back(cancelled(std::static_pointer_cast<Credit>(permit.payload)));
            notify();
        } else queued_.push_front(std::move(permit));
        cv_.notify_one();
    }
    int validate(const rad::ExternalPermit& permit,const rad::StepPlan& plan,const RadBatch& batch)override {
        auto credit=std::static_pointer_cast<Credit>(permit.payload);
        if(!credit||plan.e.size()!=1||plan.e[0].seq!=credit->active->native_id)return RAD_E_STATE;
        const auto&e=plan.e[0];const auto&p=credit->plan;const auto phase=p.at("phase").get<std::string>();
        if(e.is_prefill!=(phase=="prefill"))return RAD_E_STATE;
        if(e.is_prefill){
            const auto ids=p.at("input_ids").get<std::vector<int32_t>>();
            if(ids.size()!=size_t(e.n_tokens)||e.ctx_len<0||
               size_t(e.ctx_len)+ids.size()>e.req->prompt.size())return RAD_E_STATE;
            if(!std::equal(ids.begin(),ids.end(),e.req->prompt.begin()+e.ctx_len))return RAD_E_STATE;
        }
        if(!e.is_prefill&&(e.n_tokens<1||e.n_tokens>4))return RAD_E_STATE;
        if(phase=="decode"&&e.n_spec!=0)return RAD_E_STATE;
        if(uint64_t(e.ctx_len)+uint64_t(e.n_tokens)>number(p,"reserved_through"))return RAD_E_STATE;
        return batch.n_seq==1?RAD_OK:RAD_E_STATE;
    }
    int completed(const rad::ExternalPermit& permit,const rad::StepPlan& plan,const RadBatch& batch,int proposed,int accepted)override {
        try {
            auto credit=std::static_pointer_cast<Credit>(permit.payload);auto a=credit->active;
            rad::server::SinkToken buffer[8];const auto count=a->sink->read(buffer,8);
            if(count>number(credit->plan,"max_committed"))return RAD_E_STATE;
            std::vector<int32_t> tokens;std::vector<bool> flags;
            for(size_t i=0;i<count;i++){tokens.push_back(buffer[i].token);flags.push_back(buffer[i].accepted_draft);}
            std::lock_guard<std::mutex> lock(mu_);
            const auto before=cursor_;
            const auto logical=number(before,"committed_tokens")+(plan.e[0].is_prefill?plan.e[0].n_tokens:0)+count;
            const int64_t computed=a->request->n_computed;
            Json result={{"ticket",credit->plan.at("ticket")},{"state",handle(a->state_id,++state_version_)},
                {"before",before},{"tokens",tokens},{"draft_flags",flags},{"cancelled",false},
                {"proposed_drafts",proposed},{"verified_drafts",plan.e[0].n_spec},
                {"accepted_drafts",std::max(accepted,0)},
                {"terminal_frontier",nullptr},{"terminal_reason",nullptr},{"retired_computed_tokens",nullptr}};
            if(a->sink->done()) {
                result["after"]=nullptr;result["terminal_frontier"]=logical;
                auto reason=std::string(rad::server::finish_reason_oai(a->sink->finish_reason()));
                if(reason=="abort")reason="cancelled";
                result["terminal_reason"]=reason;result["retired_computed_tokens"]=computed;
            }else {
                if(computed<0||logical<uint64_t(computed)||logical-uint64_t(computed)>1)return RAD_E_STATE;
                const int32_t pending=logical==uint64_t(computed)?-1:
                    (a->request->output.empty()?-1:a->request->output.back());
                cursor_=cursor(logical,computed,pending);result["after"]=cursor_;
            }
            --issued_;completed_.push_back(std::move(result));++native_steps_;last_native_step_=batch.step;notify();return RAD_OK;
        }catch(...){return RAD_E_STATE;}
    }
    void revoke() {
        std::lock_guard<std::mutex> lock(mu_);revoked_=true;
        while(!queued_.empty()){
            auto c=std::static_pointer_cast<Credit>(queued_.front().payload);
            completed_.push_back(cancelled(c));
            queued_.pop_front();
        }
        notify();cv_.notify_all();
    }
    void fault(int status)override {
        std::lock_guard<std::mutex> lock(mu_);failure_=status;closed_=true;notify();cv_.notify_all();
    }
    void shutdown(){fault(RAD_OK);}
    bool pop(Json& result) {
        std::lock_guard<std::mutex> lock(mu_);
        if(completed_.empty())return false;
        result=std::move(completed_.front());completed_.pop_front();--held_;return true;
    }
    Json stats(){
        std::lock_guard<std::mutex> lock(mu_);
        return {{"bridge_only",bridge_only},{"batch_executor",!bridge_only},{"window",window_},{"held",held_},{"queued",queued_.size()},
            {"issued",issued_},{"completed_waiting",completed_.size()},{"native_steps_completed",native_steps_},
            {"last_native_step",last_native_step_},
            {"revoked",revoked_},{"failure",failure_}};
    }
    void require_idle(const Json* state=nullptr) {
        std::lock_guard<std::mutex> lock(mu_);
        if(held_ || issued_ || !queued_.empty()) throw std::runtime_error("storage operation requires the final step fence");
        if(state && (!active_ || revoked_ || number(*state,"worker_generation",1)!=generation_ ||
           number(*state,"model_generation",1)!=1 || number(*state,"resource_id",1)!=active_->state_id ||
           number(*state,"version",1)!=state_version_)) throw std::runtime_error("stale state storage handle");
    }
    Json restored(int64_t tokens) {
        std::lock_guard<std::mutex> lock(mu_);
        if(!active_ || held_ || last_id_)throw std::runtime_error("restore requires a fresh resource lease");
        cursor_=cursor(tokens,tokens);++state_version_;
        return {{"state",handle(active_->state_id,state_version_)},{"cursor",cursor_}};
    }
    void require_fresh() {
        std::lock_guard<std::mutex> lock(mu_);
        if(!active_ || held_ || last_id_ || number(cursor_,"computed_tokens"))
            throw std::runtime_error("restore requires an unused state lease");
    }
    int failure(){std::lock_guard<std::mutex> lock(mu_);return failure_;}
private:
    Json cancelled(const std::shared_ptr<Credit>& c) {
        return {{"ticket",c->plan.at("ticket")},{"state",handle(c->active->state_id,++state_version_)},
            {"before",cursor_},{"after",cursor_},{"tokens",Json::array()},{"draft_flags",Json::array()},
            {"cancelled",true},{"terminal_frontier",nullptr},{"terminal_reason",nullptr},
            {"retired_computed_tokens",nullptr}};
    }
    void notify(){
        int status;
        do{status=eventfd_write(event_,1);}while(status<0&&errno==EINTR);
        if(status<0&&errno!=EAGAIN){failure_=RAD_E_IO;closed_=true;}
    }
    uint64_t generation_,last_id_=0,state_version_=1;
    int window_,event_,failure_=0,native_steps_=0,last_native_step_=-1;
    size_t held_=0,issued_=0;bool revoked_=false,closed_=false;
    Json cursor_=cursor(0,0);std::shared_ptr<Active> active_;
    std::mutex mu_;std::condition_variable cv_;
    std::deque<rad::ExternalPermit> queued_;std::deque<Json> completed_;
};

int run(int fd,rad::BatchState& sched,std::shared_ptr<Controller> control,Json facts, const Json& composition,
        const std::function<Json()>& runtime_stats
#ifdef FREETOKEN_EXPLICIT_BATCHES
        ,freetoken::StateStorage* storage
#endif
        ) {
    if(!bridge_only) {
        facts["backend"]="native-kernels";
        facts["prefix_policy_owner"]="freetoken.kvcache";
        facts["placement_owner"]="freetoken.moe";
    }
    if(!emit(fd,Ready,{{"bridge_only",bridge_only},{"batch_executor",!bridge_only},{"graph_owner",composition.is_null()?"v1":"freetoken"},{"scheduler_owner",sched.externally_planned()?"freetoken":"v1"},
        {"pid",int(getpid())},
        {"worker_generation",control->generation()},{"engine",facts},
        {"max_credit_window",control->pipeline()?2:1},{"physical_issue_ahead",sched.supports_issue_ahead()}}))return RAD_E_IO;
    std::shared_ptr<Active> active;uint64_t next_id=1;bool connected=true;
    while(connected) {
        if(control->failure()<0){
            emit(fd,Error,{{"error","native step execution failed"},{"status",control->failure()}});
            break;
        }
        Json result;
        while(control->pop(result)){
            const auto&r=result.at("ticket").at("request");
            if(!emit(fd,Complete,result,number(r,"uid"),number(r,"epoch"))){connected=false;break;}
        }
        if(!connected)break;
        pollfd fds[2]={{fd,POLLIN,0},{control->event(),POLLIN,0}};
        if(poll(fds,2,100)<0){if(errno==EINTR)continue;break;}
        if(fds[1].revents&POLLIN){
            eventfd_t n;int status;
            do{status=eventfd_read(control->event(),&n);}while(status<0&&errno==EINTR);
            if(status<0&&errno!=EAGAIN)throw std::runtime_error("executor event channel failed");
        }
        if(fds[0].revents&(POLLHUP|POLLERR|POLLNVAL))break;
        if(!(fds[0].revents&POLLIN))continue;
        Header h;if(!transfer(fd,&h,sizeof(h),false))break;
        if(h.signature!=magic||h.version!=3||h.reserved||h.bytes>max_payload)
            throw std::runtime_error("invalid executor frame");
        std::string bytes(h.bytes,'\0');if(!transfer(fd,bytes.data(),bytes.size(),false))break;
        try {
            const auto m=bytes.empty()?Json::object():Json::parse(bytes);
            if(h.kind==Shutdown)break;
            if(h.kind==Compile){
                if(composition.is_null()) {
                    if(!m.value("reference_program",false))throw std::runtime_error("reference plan required");
                } else if(!m.contains("composition") || m.at("composition") != composition) {
                    throw std::runtime_error("compile composition differs from loaded model plan");
                }
                connected=emit(fd,Compile,{{"executable",control->handle(1)},{"bridge_only",bridge_only},{"batch_executor",!bridge_only}},h.uid,h.epoch);
            }else if(h.kind==Reserve){
                if(active)throw std::runtime_error("one state bundle at a time");
                auto a=std::make_shared<Active>();a->uid=h.uid;a->epoch=h.epoch;
                if(!a->epoch)throw std::runtime_error("request epoch must be positive");
                a->native_id=next_id++;a->state_id=100+a->native_id;a->sink=std::make_shared<rad::server::Sink>();
                auto request=std::make_unique<rad::Request>();request->id=a->native_id;
                request->prompt=m.at("input_ids").get<std::vector<int32_t>>();
                request->max_tokens=m.value("max_tokens",256);
                if(request->max_tokens<1)throw std::runtime_error("invalid output budget");
                request->sp.temp=m.value("temperature",0.0f);request->sp.top_k=m.value("top_k",0);
                request->sp.top_p=m.value("top_p",1.0f);request->ignore_eos=m.value("ignore_eos",false);
                request->sp.freq_penalty=m.value("frequency_penalty",0.0f);
                request->sp.pres_penalty=m.value("presence_penalty",0.0f);
                request->sink=a->sink.get();a->request=request.get();control->bind(a);
                const int rc=sched.add(std::move(request));
                if(rc<0){a->sink->fail(rc,"reference state reservation failed");control->unbind();throw std::runtime_error("state reservation failed");}
                active=a;connected=emit(fd,Reserve,{{"state",control->handle(a->state_id)},{"cursor",cursor(0,0)}},h.uid,h.epoch);
            }else if(h.kind==Submit){
                const auto& request=m.at("ticket").at("request");
                if(number(request,"uid")!=h.uid||number(request,"epoch",1)!=h.epoch)
                    throw std::runtime_error("step frame identity mismatch");
                control->submit(m);
            }else if(h.kind==Revoke){
                if(!active||h.uid!=active->uid||h.epoch!=active->epoch)throw std::runtime_error("stale cancellation");
                control->revoke();active->sink->request_cancel();sched.cancel(active->native_id);
                connected=emit(fd,Revoke,{{"revoked",true}},h.uid,h.epoch);
            }else if(h.kind==Release){
                if(!active||h.uid!=active->uid||h.epoch!=active->epoch)throw std::runtime_error("stale release");
                if(!active->sink->done())throw std::runtime_error("revoke the live state before release");
                control->unbind();sched.reap(active->native_id);active.reset();
                connected=emit(fd,Release,{{"released",true}},h.uid,h.epoch);
            }else if(h.kind==Inspect){
                auto stats=control->stats();stats["physical_steps_ahead"]=sched.metrics().steps_ahead;
                stats["runtime"]=runtime_stats();
#ifdef FREETOKEN_EXPLICIT_BATCHES
                if(storage) stats["storage"]=storage->inspect();
#endif
                connected=emit(fd,Inspect,stats,h.uid,h.epoch);
#ifdef FREETOKEN_EXPLICIT_BATCHES
            }else if(h.kind==Snapshot || h.kind==Restore || h.kind==DropSnapshot){
                if(!storage)throw std::runtime_error("prefix storage is disabled");
                if(h.kind!=DropSnapshot && (!active || active->uid!=h.uid || active->epoch!=h.epoch))
                    throw std::runtime_error("storage request identity mismatch");
                control->require_idle(h.kind==DropSnapshot?nullptr:&m.at("state"));
                const auto snapshot_id=[&](const Json& handle) {
                    if(number(handle,"worker_generation",1)!=control->generation() ||
                       number(handle,"model_generation",1)!=1 || number(handle,"version",1)!=1)
                        throw std::runtime_error("stale snapshot generation");
                    return number(handle,"resource_id",100001)-100000;
                };
                Json reply;
                if(h.kind==Snapshot) {
                    const auto parent=m.at("parent").is_null()?0:snapshot_id(m.at("parent"));
                    const auto id=storage->snapshot(active->native_id,active->request->prompt,parent);
                    reply={{"snapshot",control->handle(100000+id)}};
                }else if(h.kind==Restore) {
                    control->require_fresh();
                    const auto tokens=storage->restore(active->native_id,active->request->prompt,snapshot_id(m.at("snapshot")));
                    reply=control->restored(tokens);
                }else { storage->drop(snapshot_id(m.at("snapshot")));reply={{"dropped",true}}; }
                connected=emit(fd,Kind(h.kind),reply,h.uid,h.epoch);
#endif
            }else throw std::runtime_error("unsupported executor command");
        }catch(const std::exception&e){connected=emit(fd,Error,{{"error",e.what()}},h.uid,h.epoch);}
    }
    control->shutdown();
    if(active){active->sink->request_cancel();sched.cancel(active->native_id);sched.reap(active->native_id);}
    return RAD_OK;
}
} // namespace ft_executor

int main(int argc,char**argv){
    using namespace ft_executor;
    const auto parent=getppid();
    prctl(PR_SET_PDEATHSIG,SIGTERM);std::signal(SIGPIPE,SIG_IGN);
    if(getppid()!=parent)return 1;
    try {
        std::string config,model_plan,placement_plan;int fd=-1,window=1;
        for(int i=1;i<argc;i++){
            if(i+1==argc)throw std::runtime_error("missing argument value");
            const std::string key=argv[i];
            if(key=="--config")config=argv[++i];
            else if(key=="--model-plan")model_plan=argv[++i];
            else if(key=="--placement-plan")placement_plan=argv[++i];
            else if(key=="--ipc-fd")fd=std::stoi(argv[++i]);
            else if(key=="--credit-window")window=std::stoi(argv[++i]);
            else throw std::runtime_error("unknown bridge argument");
        }
        if(config.empty()||fd<0)throw std::runtime_error("config and IPC fd required");
        rad::fleet::FleetConfig fleet;
        if(rad::fleet::load(config,&fleet)<0||fleet.models.size()!=1)throw std::runtime_error("one model required");
        auto cfg=std::move(fleet.models[0].cfg);
        if(cfg.max_seqs!=1||cfg.kld()||cfg.calib()||cfg.diagnose_mode())
            throw std::runtime_error("batch executor requires B1 ordinary text mode");
#ifdef FREETOKEN_EXPLICIT_BATCHES
        const bool prefix=cfg.prefix_cache;
        const int64_t prefix_budget=prefix?(cfg.prefix_cache_mib>0?cfg.prefix_cache_mib:1024)*(1LL<<20):0;
        cfg.prefix_cache_mib=0;
        cfg.checkpoint_slots=prefix?2:0;
#else
        if(cfg.prefix_cache)throw std::runtime_error("the reference bridge requires prefix off");
#endif
        const char*expected=std::getenv("FREETOKEN_NATIVE_EXPECTED_PCIS");
        if(!expected)throw std::runtime_error("reserved PCI vector required");
        const auto pcis=Json::parse(expected).get<std::vector<std::string>>();
        if(pcis.size()!=size_t(cfg.tp)||rad_dev_count()!=cfg.tp)throw std::runtime_error("device count mismatch");
        for(int i=0;i<cfg.tp;i++)if(rad::dev_pci_address(i)!=pcis[size_t(i)])throw std::runtime_error("PCI mismatch");
        const auto now=std::chrono::steady_clock::now().time_since_epoch();
        const auto generation=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
        setenv("FREETOKEN_NATIVE_MANIFEST_DIR",std::filesystem::path(config).parent_path().c_str(),1);
        auto control=std::make_shared<Controller>(generation,window);
#ifdef FREETOKEN_EXPLICIT_BATCHES
        if(model_plan.empty())throw std::runtime_error("FreeToken model composition required");
        auto batches=std::make_unique<freetoken::ExplicitBatchState>();auto* batch_state=batches.get();
        rad::DeviceRuntime engine(std::move(cfg),std::move(batches));
#else
        rad::Engine engine(std::move(cfg));
#endif
        engine.set_external_step_control(control);
#ifdef FREETOKEN_EXPLICIT_BATCHES
        if(placement_plan.empty())throw std::runtime_error("FreeToken placement plan required");
        std::ifstream placement_file(placement_plan);
        const auto placement=Json::parse(std::string((std::istreambuf_iterator<char>(placement_file)),{}));
        engine.set_placement_factory([placement](int rank){return std::make_unique<freetoken::ExpertPlacement>(placement,rank);});
#endif
        Json composition;
        if(!model_plan.empty()) {
            engine.set_program_lowerer(freetoken::load_qwen_plan(model_plan));
            std::ifstream file(model_plan);std::string bytes((std::istreambuf_iterator<char>(file)),{});
            composition=Json::parse(bytes);
#ifdef FREETOKEN_EXPLICIT_BATCHES
            if(placement.at("metadata_sha256")!=composition.at("model_metadata_sha256") ||
               placement.at("tensor_parallel").get<size_t>()!=composition.at("ranks").size())
                throw std::runtime_error("placement plan belongs to another model/rank set");
#endif
        }
        if(engine.start()<0)throw std::runtime_error("reference model startup failed");
#ifdef FREETOKEN_EXPLICIT_BATCHES
        std::unique_ptr<freetoken::StateStorage> storage;
        if(prefix)storage=std::make_unique<freetoken::StateStorage>(engine.storage_rank_views(),*batch_state,prefix_budget);
#endif
        return engine.serve_headless([&](rad::BatchState&s,std::shared_ptr<const rad::Vocab>,const Json&facts){
            return run(fd,s,control,facts,composition,[&] {return engine.headless_stats();}
#ifdef FREETOKEN_EXPLICIT_BATCHES
                ,storage.get()
#endif
            );})<0?1:0;
    }catch(const std::exception&e){std::fprintf(stderr,"executor bridge: %s\n",e.what());return 1;}
}
