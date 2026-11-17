/* lora/declare.h -- the runtime-LoRA half of declare: after the architecture plugin has declared
 * its program, build the sites (lora.h), declare the adapter pools and the two ops each targeted
 * GEMM is followed by, and record what the issue path needs (Program::lora_sites).
 *
 * Runs inside rad_tools_declare, after rad_arch_declare and before the sampler, identically on the
 * real declare and on every sizing declare, which must produce the same op list (rad_builder.cpp
 * checks it op by op). */
#pragma once
#include "lora/config.h"
#include "lora/lora.h"

#include <vector>

namespace rad {

class Builder;

struct LoraDeclare {
    const LoraConfig*        cfg = nullptr;
    lora::TargetSet          targets;
    /* The real declare's sites, for the adapter loader; null on a sizing declare. */
    std::vector<lora::Site>* sites_out = nullptr;
};

/* RAD_OK, or a negative code after RAD_ERR lines naming every refusal. */
int lora_declare(Builder& b, const LoraDeclare& d);

}  /* namespace rad */
