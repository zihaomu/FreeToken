#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace freetoken {
enum class QwenOp {
    Embed, PleIds, Enter, Rope, Ple, MixRead, Gdn, Qsa, MixWrite,
    FfnRead, Moe, FfnWrite, Output, Logits
};
struct QwenInstruction { QwenOp op; int64_t layer = -1; };
struct QwenState {
    std::string name, kind, encoding;
    std::vector<int64_t> shape;
    std::vector<int> layers;
    int64_t block_tokens = 0;
};
struct QwenRank {
    std::array<int64_t, 2> vocab, query_heads, kv_heads, gdn_key_heads, gdn_value_heads;
    std::array<std::array<int64_t, 2>, 2> expert_parity;
    std::array<int64_t, 2> shared;
    std::array<int64_t, 2> protected_span;
    std::map<std::string, QwenState> states;
};
struct QwenExpertOrder { std::vector<int64_t> regular, protected_ids; };
struct QwenGraph {
    std::map<std::string, int64_t> geometry;
    std::vector<bool> full;
    std::vector<QwenRank> ranks;
    std::vector<QwenInstruction> trunk;
    std::vector<QwenExpertOrder> experts;
    int64_t ple_layer = -1;
    int mtp_depth = 0;
    bool piece_pipeline = false;
};
}
