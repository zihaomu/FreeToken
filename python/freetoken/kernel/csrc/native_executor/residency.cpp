#include "residency.h"
#include <stdexcept>

namespace freetoken {
using namespace rad;
int ExpertPlacement::plan_initial(const PlannerInput& input,Plan* out) {
    if(input.world_size!=config_.at("tensor_parallel").get<int>() || input.rank!=rank_)return RAD_E_STATE;
    if(!input.config)return RAD_E_INVAL;
    expert_host_budget_=input.config->host_pool_mib*(1LL<<20);
    auxiliary_host_budget_=(input.config->embedding_host_mib+input.config->tower_host_mib)*(1LL<<20);
    if(expert_host_budget_*input.world_size>config_.at("host_pool_mib").get<int64_t>()*(1LL<<20))return RAD_E_FULL;
    ++initial_plans_;
    return Planner(input).plan(out);
}
int ExpertPlacement::init(const Plan& plan,const HeatParams&) {
    if(config_.at("owner").get<std::string>()!="freetoken.moe" || config_.at("schema").get<int>()!=1 ||
       config_.at("algorithm").get<std::string>()!="lfu_decay" || !config_.at("direct_host_read").get<bool>())
        return RAD_E_UNSUPPORTED;
    const int64_t cap=config_.at("gpu_memory_mib").get<int64_t>();
    if((cap && plan.vram_budget>cap*(1LL<<20)) ||
       plan.host_budget+plan.file_stage_reserve+plan.embedding_host_bytes+plan.tower_host_bytes>
           (expert_host_budget_?expert_host_budget_:config_.at("host_pool_mib").get<int64_t>()*(1LL<<20))+auxiliary_host_budget_)
        return RAD_E_FULL;
    HeatParams params;
    params.decay=config_.at("decay").get<float>();params.hysteresis=config_.at("hysteresis").get<float>();
    params.min_gain=config_.at("min_gain").get<float>();params.moves_per_step=config_.at("moves_per_class").get<int>();
    params.credit_cap=config_.at("credit_cap").get<float>();params.admit_heat=config_.at("admit_heat").get<float>();
    if(params.moves_per_step!=RAD_PLACE_MOVES_PER_STEP || params.decay<=0 || params.decay>1 ||
       params.hysteresis<1 || params.credit_cap<=0)return RAD_E_INVAL;
    RAD_TRY(HeatEngine::init(plan,params));
    locations_.clear();locations_.reserve(plan.units.size());
    for(const auto& unit:plan.units)locations_.push_back({unit.tier,unit.tier,unit.slab_slot});
    vram_budget_=plan.vram_budget;
    host_budget_=plan.host_budget+plan.file_stage_reserve+plan.embedding_host_bytes+plan.tower_host_bytes;
    return RAD_OK;
}
void ExpertPlacement::observe(int32_t layer,const int32_t* counts,int32_t n,float weight) {
    HeatEngine::observe(layer,counts,n,weight);++feedback_;
}
void ExpertPlacement::desire(int32_t unit,Tier tier) {
    if(unit<0 || size_t(unit)>=locations_.size())return;
    auto& location=locations_[size_t(unit)];location.desired=tier;location.intent=++intents_;
}
int ExpertPlacement::step(std::vector<Swap>* out) {
    ++rounds_;const int result=HeatEngine::step(out);
    if(result>=0)for(const auto& swap:*out) { desire(swap.promote,Tier::VRAM);desire(swap.demote,Tier::HostPinned); }
    return result;
}
void ExpertPlacement::set_tier(int32_t unit,Tier tier) {
    HeatEngine::set_tier(unit,tier);
    if(unit<0 || size_t(unit)>=locations_.size())return;
    auto& location=locations_[size_t(unit)];location.committed=tier;
    if(!location.busy)location.desired=tier;
}
void ExpertPlacement::set_busy(int32_t unit,bool busy) {
    HeatEngine::set_busy(unit,busy);
    if(unit<0 || size_t(unit)>=locations_.size())return;
    auto& location=locations_[size_t(unit)];location.busy=busy;
    if(!busy) {location.ready=nullptr;location.desired=location.committed;}
}
void ExpertPlacement::order_candidates(std::vector<std::pair<float,int32_t>>& units,bool hottest,size_t limit) {
    ++rankings_;HeatEngine::order_candidates(units,hottest,limit);
}
std::vector<int32_t> ExpertPlacement::fill_order(const std::vector<int32_t>& units,const std::vector<int32_t>& layers,int64_t room) {
    ++rankings_;auto result=HeatEngine::fill_order(units,layers,room);
    for(auto unit:result)desire(unit,Tier::VRAM);
    return result;
}
void ExpertPlacement::published(int32_t unit,Tier tier,int64_t slot,uint64_t generation,RadEvent ready) {
    if(unit<0 || size_t(unit)>=locations_.size())throw std::runtime_error("unknown placement publication");
    auto& location=locations_[size_t(unit)];
    if(generation<=location.generation)throw std::runtime_error("stale placement publication");
    location.generation=generation;location.slot=slot;location.ready=ready;
    if(!ready)location.committed=tier;
    ++published_;
}
bool ExpertPlacement::authorize_move(int32_t unit,Tier tier,uint64_t generation) {
    if(unit<0 || size_t(unit)>=locations_.size() || locations_[size_t(unit)].generation!=generation)return false;
    desire(unit,tier);return true;
}
bool ExpertPlacement::authorize_loan(int lender,int64_t bytes) {
    if(lender<0 || lender>=3 || bytes<0)return false;
    if((lender==0 && !config_.at("lend_kv").get<bool>()) ||
       (lender==1 && !config_.at("lend_arena").get<bool>()))return false;
    const auto cap=config_.at("gpu_memory_mib").get<int64_t>()*(1LL<<20);
    int64_t total=bytes;
    for(int i=0;i<3;++i)if(i!=lender)total+=loans_[i];
    if(cap && total>cap)return false;
    loans_[lender]=bytes;return true;
}
std::string ExpertPlacement::control_report() const {
    uint64_t pending=0,ready=0,unsettled=0,unpublished=0;
    for(const auto& location:locations_) {
        pending+=location.busy;ready+=location.ready!=nullptr;unsettled+=location.desired!=location.committed;
        unpublished+=location.generation==0;
    }
    return Json({{"owner",owner()},{"rank",rank_},{"units",locations_.size()},
        {"initial_plans",initial_plans_},{"loan_grants_bytes",{loans_[0],loans_[1],loans_[2]}},
        {"vram_budget_bytes",vram_budget_},{"host_budget_bytes",host_budget_},
        {"expert_host_budget_bytes",expert_host_budget_},{"auxiliary_host_budget_bytes",auxiliary_host_budget_},
        {"feedback_reports",feedback_},{"policy_rounds",rounds_},{"intents",intents_},
        {"rankings",rankings_},{"publications",published_},{"pending_units",pending},
        {"ready_fences",ready},{"unsettled_targets",unsettled},{"unpublished_units",unpublished},
        {"direct_host_read",true}}).dump();
}
}
