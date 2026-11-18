#pragma once
#include "build/startup.h"

namespace freetoken {
std::shared_ptr<const rad::ProgramLowerer> load_qwen_plan(const std::string& path);
}
