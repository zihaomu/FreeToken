/* A temporary P2 seam: an external scheduler authorizes each model step. */
#pragma once

#include "rad_types.h"
#include <atomic>
#include <cstdint>
#include <memory>

struct RadBatch;

namespace rad {
class BatchState;

struct StepPlan;

struct ExternalPermit {
    uint64_t step_id = 0;
    std::shared_ptr<void> payload;
};

class ExternalStepControl {
public:
    virtual int prepare(const ExternalPermit&, BatchState&, bool = false) { return RAD_OK; }
    virtual ~ExternalStepControl() = default;
    virtual bool acquire(ExternalPermit& out, bool wait,
                         const std::atomic<bool>& stopping) = 0;
    virtual void defer(ExternalPermit permit) = 0;
    virtual int validate(const ExternalPermit& permit, const StepPlan& plan,
                         const RadBatch& batch) = 0;
    virtual int completed(const ExternalPermit& permit, const StepPlan& plan,
                          const RadBatch& batch,int proposed,int accepted) = 0;
    virtual void fault(int status) = 0;
    virtual bool pipeline() const = 0;
};

} // namespace rad
