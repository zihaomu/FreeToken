#pragma once
#include "place/heat.h"
#include "util/json.h"

namespace freetoken {
// A CPU FreeToken policy instance per GPU rank; HeatEngine supplies the qualified LFU math.
class ExpertPlacement final : public rad::HeatEngine {
public:
    ExpertPlacement(rad::Json config,int rank):config_(std::move(config)),rank_(rank) {}
    int init(const rad::Plan&,const rad::HeatParams&) override;
    void observe(int32_t,const int32_t*,int32_t,float) override;
    int step(std::vector<rad::Swap>*) override;
    void set_tier(int32_t,rad::Tier) override;
    void set_busy(int32_t,bool) override;
    void order_candidates(std::vector<std::pair<float,int32_t>>&,bool,size_t) override;
    std::vector<int32_t> fill_order(const std::vector<int32_t>&,const std::vector<int32_t>&,int64_t) override;
    const char* owner() const override { return "freetoken.moe"; }
    std::string control_report() const override;
    void published(int32_t,rad::Tier,int64_t,uint64_t,RadEvent) override;
    bool authorize_move(int32_t,rad::Tier,uint64_t) override;
    bool authorize_loan(int,int64_t) override;
    int plan_initial(const rad::PlannerInput&,rad::Plan*) override;
private:
    struct Location {
        rad::Tier desired,committed;
        int64_t slot=-1;
        uint64_t generation=0,intent=0;
        RadEvent ready=nullptr;
        bool busy=false;
    };
    rad::Json config_;
    int rank_;
    std::vector<Location> locations_;
    uint64_t feedback_=0,rounds_=0,intents_=0,rankings_=0,published_=0;
    int64_t vram_budget_=0,host_budget_=0;
    int64_t expert_host_budget_=0,auxiliary_host_budget_=0;
    int64_t loans_[3]={};
    uint64_t initial_plans_=0;
    void desire(int32_t,rad::Tier);
};
}
