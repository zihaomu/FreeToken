#include "qwen_plan.h"
#include <arch/freetoken_qwen_blocks.h>
#include "util/json.h"

#include <fstream>
#include <stdexcept>

namespace freetoken {
namespace {
using rad::Json;
int64_t integer(const Json& value) {
    if (!value.is_number_integer() || value.get<int64_t>() < 0)
        throw std::runtime_error("invalid Qwen plan integer");
    return value.get<int64_t>();
}
RadKVGroupDecl state_decl(const QwenState& state, int mtp_depth) {
    RadKVGroupDecl d{};
    d.dtype = rad_dtype_parse(state.encoding.c_str());
    if (d.dtype == RAD_DT_INVALID || state.shape.size() != 3)
        throw std::runtime_error("unsupported native state encoding or rank");
    const auto& x = state.shape;
    if (state.kind == "kv" || state.kind == "index") {
        if (x[0] != 2) throw std::runtime_error("paged state needs K and V axes");
        d.kind = RAD_KV_FULL; d.n_head_kv = x[1]; d.head_dim = x[2];
    } else if (state.kind == "recurrent") {
        d.kind = RAD_KV_LINEAR; d.n_head_kv = x[0];
        d.state_dim[0] = x[1]; d.state_dim[1] = x[2];
    } else if (state.kind == "conv" || state.kind == "ple_history") {
        d.kind = RAD_KV_CONV; d.n_head_kv = x[0]; d.head_dim = x[1];
        d.conv_width = x[2] - mtp_depth + 1;
        if (d.conv_width <= 1) throw std::runtime_error("invalid native history window");
    } else throw std::runtime_error("unsupported native state kind");
    return d;
}
bool same_state(const RadKVGroupDecl& a, const RadKVGroupDecl& b) {
    return a.kind == b.kind && a.dtype == b.dtype && a.window == b.window &&
        a.n_head_kv == b.n_head_kv && a.head_dim == b.head_dim &&
        a.state_dim[0] == b.state_dim[0] && a.state_dim[1] == b.state_dim[1] &&
        a.conv_width == b.conv_width;
}
std::array<int64_t, 2> span(const Json& value) {
    if (!value.is_array() || value.size() != 2)
        throw std::runtime_error("invalid Qwen plan span");
    std::array<int64_t, 2> out{integer(value.at(0)), integer(value.at(1))};
    if (out[1] <= out[0]) throw std::runtime_error("empty Qwen plan span");
    return out;
}
}

std::shared_ptr<const rad::ProgramLowerer> load_qwen_plan(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() > (1 << 20))
        throw std::runtime_error("missing or oversized FreeToken model composition");
    file.seekg(0);
    std::string bytes((std::istreambuf_iterator<char>(file)), {});
    const Json doc = Json::parse(bytes);
    if (doc.at("schema").get<int>() != 1 ||
        doc.at("owner").get<std::string>() != "freetoken.models.qwen4_exp" ||
        doc.at("fusion").get<std::string>() != "hc_quant_moe_gather_p2p")
        throw std::runtime_error("unsupported FreeToken Qwen composition");
    auto graph = std::make_shared<QwenGraph>();
    const auto schedule = doc.at("prefill_schedule").get<std::string>();
    if (schedule != "piece_pipeline" && schedule != "serial")
        throw std::runtime_error("unsupported prefill fusion schedule");
    graph->piece_pipeline = schedule == "piece_pipeline";
    for (const auto& [key, value] : doc.at("geometry").items()) {
        const auto n = integer(value);
        if (!n) throw std::runtime_error("zero Qwen geometry");
        graph->geometry.emplace(key, n);
    }
    const auto layers = graph->geometry.at("n_layer");
    if (layers > 256 || doc.at("layer_types").size() != size_t(layers))
        throw std::runtime_error("invalid Qwen layer count");
    for (const auto& kind : doc.at("layer_types")) {
        const auto text = kind.get<std::string>();
        if (text != "full_attention" && text != "linear_attention")
            throw std::runtime_error("unknown Qwen mixer");
        graph->full.push_back(text == "full_attention");
    }
    graph->ple_layer = integer(doc.at("ple_layer"));
    if (graph->ple_layer >= layers || graph->full.at(size_t(graph->ple_layer)))
        throw std::runtime_error("invalid PLE injection layer");
    graph->mtp_depth = int(integer(doc.at("mtp_depth")));
    if (graph->mtp_depth != 0 && graph->mtp_depth != 3)
        throw std::runtime_error("unsupported MTP graph depth");
    for (const auto& row : doc.at("ranks")) {
        if (integer(row.at("rank")) != int64_t(graph->ranks.size()))
            throw std::runtime_error("unordered Qwen rank plan");
        QwenRank rank{};
        rank.vocab = span(row.at("vocab"));
        rank.query_heads = span(row.at("query_heads"));
        rank.kv_heads = span(row.at("kv_heads"));
        rank.gdn_key_heads = span(row.at("gdn_key_heads"));
        rank.gdn_value_heads = span(row.at("gdn_value_heads"));
        rank.shared = span(row.at("shared"));
        rank.protected_span = span(row.at("protected"));
        if (row.at("expert_parity").size() != 2)
            throw std::runtime_error("two expert parity classes required");
        for (size_t p = 0; p < 2; ++p) rank.expert_parity[p] = span(row.at("expert_parity").at(p));
        for (const auto& item : row.at("states")) {
            QwenState state;
            state.name = item.at("name").get<std::string>();
            state.kind = item.at("kind").get<std::string>();
            state.encoding = item.at("encoding").get<std::string>();
            state.shape = item.at("local_shape").get<std::vector<int64_t>>();
            state.layers = item.at("layer_ids").get<std::vector<int>>();
            state.block_tokens = integer(item.at("block_tokens"));
            for (auto n : state.shape) if (n <= 0) throw std::runtime_error("invalid state dimension");
            if (state.layers.empty() || !std::is_sorted(state.layers.begin(),state.layers.end()) ||
                std::adjacent_find(state.layers.begin(),state.layers.end()) != state.layers.end() ||
                state.layers.front() < 0 || state.layers.back() > layers)
                throw std::runtime_error("invalid state layer binding");
            (void)state_decl(state,graph->mtp_depth);
            if (!rank.states.emplace(state.name,state).second)
                throw std::runtime_error("duplicate state group");
        }
        graph->ranks.push_back(rank);
    }
    if (graph->ranks.empty() || graph->ranks.size() > 2)
        throw std::runtime_error("unsupported Qwen TP world");
    for (const auto& row : doc.at("experts")) {
        if (integer(row.at("layer")) != int64_t(graph->experts.size()))
            throw std::runtime_error("unordered expert layer");
        QwenExpertOrder order;
        order.regular = row.at("regular").get<std::vector<int64_t>>();
        order.protected_ids = row.at("protected").get<std::vector<int64_t>>();
        auto ids = order.regular;
        ids.insert(ids.end(), order.protected_ids.begin(), order.protected_ids.end());
        std::sort(ids.begin(), ids.end());
        if (ids.size() != size_t(graph->geometry.at("n_expert")) || order.protected_ids.size() > 8)
            throw std::runtime_error("invalid expert class partition");
        for (size_t i = 0; i < ids.size(); ++i)
            if (ids[i] != int64_t(i)) throw std::runtime_error("duplicate or missing expert ID");
        graph->experts.push_back(std::move(order));
    }
    if (graph->experts.size() != size_t(layers + (graph->mtp_depth > 0)))
        throw std::runtime_error("expert layers differ from graph layers");
    const std::map<std::string, QwenOp> ops = {
        {"embed", QwenOp::Embed}, {"ple_ids", QwenOp::PleIds}, {"hc_enter", QwenOp::Enter},
        {"rope", QwenOp::Rope}, {"ple", QwenOp::Ple}, {"hc_mix_read", QwenOp::MixRead},
        {"gdn", QwenOp::Gdn}, {"qsa", QwenOp::Qsa}, {"hc_mix_write", QwenOp::MixWrite},
        {"hc_ffn_read", QwenOp::FfnRead}, {"moe", QwenOp::Moe},
        {"hc_ffn_write", QwenOp::FfnWrite}, {"hc_output", QwenOp::Output}, {"logits", QwenOp::Logits},
    };
    for (const auto& row : doc.at("trunk")) {
        if (!row.at("layer").is_number_integer()) throw std::runtime_error("invalid block layer");
        graph->trunk.push_back({ops.at(row.at("op").get<std::string>()), row.at("layer").get<int64_t>()});
    }
    // This lowering preserves the in-place HC liveness. Other orderings need a new arena plan.
    size_t index = 0;
    auto take = [&](QwenOp op, int64_t layer = -1) {
        if (index == graph->trunk.size() || graph->trunk[index].op != op ||
            graph->trunk[index].layer != layer)
            throw std::runtime_error("unsupported block dependency/fusion sequence");
        ++index;
    };
    take(QwenOp::Embed); take(QwenOp::PleIds); take(QwenOp::Enter); take(QwenOp::Rope);
    for (int64_t l = 0; l < layers; ++l) {
        if (l == graph->ple_layer) take(QwenOp::Ple, l);
        take(QwenOp::MixRead, l); take(graph->full[size_t(l)] ? QwenOp::Qsa : QwenOp::Gdn, l);
        take(QwenOp::MixWrite, l); take(QwenOp::FfnRead, l);
        take(QwenOp::Moe, l); take(QwenOp::FfnWrite, l);
    }
    take(QwenOp::Output); take(QwenOp::Logits);
    if (index != graph->trunk.size()) throw std::runtime_error("unconsumed Qwen graph blocks");
    auto lowerer = std::make_shared<rad::ProgramLowerer>();
    lowerer->name = "freetoken.qwen4.blocks.v2";
    lowerer->declare = [graph](RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
        return freetoken_qwen_blocks::lower_blocks(b, meta, ctx, graph.get());
    };
    lowerer->probe = [graph](const RadModelMeta*, RadArchProbe* probe) {
        probe->kv_stream_ok = probe->kv_fp8_ok = probe->shape_probe_ok = 1;
        probe->draft_depth = graph->mtp_depth;
        probe->chain_pays = RAD_CHAIN_ALWAYS;
        return RAD_OK;
    };
    lowerer->state_layout = [graph](int rank, const char* name, RadKVGroupDecl* declared) {
        const auto& states = graph->ranks.at(size_t(rank)).states;
        auto it = states.find(name);
        if (it == states.end()) return RAD_E_NOTFOUND;
        auto planned = state_decl(it->second,graph->mtp_depth);
        if (!same_state(planned,*declared)) return RAD_E_SHAPE;
        *declared = planned;
        return RAD_OK;
    };
    lowerer->validate = [graph](const rad::Program& program, int rank,bool sizing) {
        const auto& states = graph->ranks.at(size_t(rank)).states;
        if (program.kv_groups.size() != states.size()+1) return RAD_E_SHAPE;
        for (size_t i=1;i<program.kv_groups.size();++i) {
            const auto& group = program.kv_groups[i];
            auto it = states.find(group.name);
            if(it==states.end() || !same_state(state_decl(it->second,graph->mtp_depth),group.decl))return RAD_E_SHAPE;
            // A sizing declaration reuses the real program's op bands. It does not build KV
            // bindings; its state declarations are checked above and plan_level checks buffers.
            if(sizing)continue;
            if (it->second.layers != group.layers ||
                it->second.block_tokens != group.block_size) return RAD_E_SHAPE;
        }
        return RAD_OK;
    };
    lowerer->step = freetoken_qwen_blocks::step;
    lowerer->pass_key = freetoken_qwen_blocks::pass_key;
    return lowerer;
}
}
