// The whole-request scheduler belongs only to the v1/reference constructor.
#include "engine.h"
#include "sched/scheduler.h"
namespace rad {
DeviceRuntime::DeviceRuntime(Config cfg) : DeviceRuntime(std::move(cfg), std::make_unique<Scheduler>()) {}
}
