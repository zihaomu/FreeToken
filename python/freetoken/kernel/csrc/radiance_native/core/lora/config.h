/* lora/config.h -- what runtime LoRA was started with (--lora, --lora-dir, --max-loras,
 * --max-lora-rank, --lora-target-modules). Read by declare (which sites and pools exist), by the
 * adapter manager (what may be loaded into them) and by nothing else.
 *
 * Off unless asked for: with no adapter named, no --lora-dir and no --max-loras, `enabled()` is
 * false, declare appends nothing, the program has no pool and no extra op, and no step carries an
 * adapter slot.
 *
 * A text model's slot holds one adapter; a generator's holds the combination one request asks for
 * (several adapter files at their strengths, core/lora/adapter.h), so its slots are not tied to
 * the number of adapters named. */
#pragma once
#include <string>
#include <utility>
#include <vector>

namespace rad {

struct LoraConfig {
    /* NAME=PATH pairs to load at startup, in the order given: a PEFT directory, or for a
     * generator one .safetensors file. */
    std::vector<std::pair<std::string, std::string>> adapters;
    /* A generator's adapter directory: a request may name any NAME.safetensors under it. */
    std::string dir;
    /* Adapter slots resident at once. 0: as many as were named, and two with --lora-dir alone. */
    int max_loras = 0;
    /* The rank every slot is sized for; an adapter of lower rank is zero-padded into it. 0: the
     * default, kDefaultRank for a text model and kDefaultGenRank for a generator, whose adapter
     * files run to rank 32 and are stacked. */
    int max_rank = 0;
    static constexpr int kDefaultRank = 16, kDefaultGenRank = 64;
    int rank(bool generator) const {
        return max_rank > 0 ? max_rank : generator ? kDefaultGenRank : kDefaultRank;
    }
    /* HF module leaf names (q_proj, gate_proj, in_proj_qkv, ...) the sites are built for, sorted
     * and unique. Empty: the union of the startup adapters' target_modules. */
    std::vector<std::string> targets;

    bool enabled() const { return max_loras > 0 || !adapters.empty() || !dir.empty(); }
    int  slots() const {
        const int named = (int)adapters.size();
        if (max_loras > 0) return max_loras;
        return named > 0 ? named : (dir.empty() ? 0 : 2);
    }
};

}  /* namespace rad */
