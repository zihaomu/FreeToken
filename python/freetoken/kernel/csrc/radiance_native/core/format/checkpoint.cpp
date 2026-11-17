/* checkpoint.cpp -- see checkpoint.h. */
#include "checkpoint.h"
#include "ggml_planes.h"
#include "rad_plugin.h"

#include "util/json.h"

#include <algorithm>
#include <filesystem>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <map>

namespace rad {

/* ================================================================== metadata */

void MetaOwn::set(const std::string& k, const std::string& v) {
    for (size_t i = 0; i < keys.size(); ++i)
        if (keys[i] == k) { vals[i] = v; return; }
    kv(k, v);
}

bool MetaOwn::has(const std::string& k) const {
    return std::find(keys.begin(), keys.end(), k) != keys.end();
}

const char* MetaOwn::get(const std::string& k) const {
    for (size_t i = 0; i < keys.size(); ++i)
        if (keys[i] == k) return vals[i].c_str();
    return nullptr;
}

void MetaOwn::finish() {
    kp.clear();
    vp.clear();
    for (auto& s : keys) kp.push_back(s.c_str());
    for (auto& s : vals) vp.push_back(s.c_str());
    m.arch_id = arch.c_str();
    m.name    = name.c_str();
    m.quant   = quant.c_str();
    m.n_kv    = (int)keys.size();
    m.kv_key  = kp.data();
    m.kv_val  = vp.data();
}

/* ---------------------------------------------------------------- config.json
 *
 * A safetensors checkpoint's architecture and dimensions live in config.json. The core parses it,
 * since the plugin only sees RadModelMeta, whose typed fields size everything at declare.
 *
 * A text-only model puts its dimensions at the top level; a multimodal one nests them under
 * `text_config` beside `vision_config` (NVIDIA's omni wrappers: `llm_config`). Every key is
 * flattened into the free-form list by dotted path, and typed fields are read from
 * `text_config.X`, then `llm_config.X`, falling back to `X`, so a plugin finds
 * `vision_config.depth` under that name either way.
 */
/* An array's scalars appended to `s` space-separated, nested arrays flattened row-major (a
 * vocoder's per-resnet dilations, [[1, 3, 5], [1, 3, 5]], read as "1 3 5 1 3 5"); false for an
 * array holding an object or a null. */
static bool join_scalars(const Json& v, std::string* s) {
    for (const auto& e : v) {
        if (e.is_array()) {
            if (!join_scalars(e, s)) return false;
            continue;
        }
        if (!s->empty()) *s += ' ';
        if (e.is_number_integer()) *s += fmt("%lld", (long long)e.get<int64_t>());
        else if (e.is_number())    *s += fmt("%.17g", e.get<double>());
        else if (e.is_string())    *s += e.get<std::string>();
        else if (e.is_boolean())   *s += e.get<bool>() ? "1" : "0";
        else return false;
    }
    return true;
}

static void flatten_json(const Json& j, const std::string& prefix, MetaOwn* out) {
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string key = prefix.empty() ? it.key() : prefix + "." + it.key();
        const Json& v = it.value();
        /* Retain otherwise invisible Quark weight overrides so import validation can reject
         * null/empty schemes instead of accidentally inheriting global MXFP4 packing. */
        if (key.rfind("quantization_config.layer_quant_config.", 0) == 0 &&
            it.key() == "weight" && (v.is_null() || (v.is_object() && v.empty()))) {
            out->kv(key, v.is_null() ? "null" : "{}");
            continue;
        }
        if (v.is_object())            flatten_json(v, key, out);
        else if (v.is_boolean())      out->kv(key, v.get<bool>() ? "1" : "0");
        else if (v.is_number_integer()) out->kv(key, fmt("%lld", (long long)v.get<int64_t>()));
        else if (v.is_number())       out->kv(key, fmt("%.17g", v.get<double>()));
        else if (v.is_string())       out->kv(key, v.get<std::string>());
        else if (v.is_array()) {
            /* An array of scalars becomes a space-separated string (mrope_section "11 11 10",
             * layer_types "linear_attention full_attention ..."), nested arrays row-major. An
             * array of objects is flattened element by element under its index (Nemotron-H
             * Puzzle's per-layer `block_configs.1.n_routed_experts`); other mixed arrays are
             * skipped. */
            std::string s;
            if (join_scalars(v, &s)) {
                out->kv(key, s);
            } else {
                bool objects = !v.empty();
                for (const auto& e : v) objects = objects && e.is_object();
                if (objects)
                    for (size_t i = 0; i < v.size(); ++i)
                        flatten_json(v[i], key + "." + std::to_string(i), out);
            }
        }
    }
}

/* A second config.json (a drafter's), flattened under a prefix only. No typed extraction, which
 * would size the trunk from the draft head; the plugin reads `draft.` keys from the free-form
 * table. No `text_config` aliasing either: a drafter is a text model, and aliasing would collide
 * with the target's bare keys. */
int meta_merge_config_json(const std::string& dir, const std::string& prefix, MetaOwn* out) {
    const std::string path = dir + "/config.json";
    std::ifstream f(path);
    if (!f) {
        RAD_ERR("%s: no config.json. A draft checkpoint carries its own geometry there -- the "
                "block size it was trained for, which trunk layers it reads, the selector's rank "
                "-- and none of it is derivable from the target.", dir.c_str());
        return RAD_E_NOTFOUND;
    }
    Json j;
    try { f >> j; }
    catch (const std::exception& e) { RAD_ERR("%s: %s", path.c_str(), e.what()); return RAD_E_FORMAT; }
    flatten_json(j, prefix, out);
    return RAD_OK;
}

/* ---------------------------------------------------------------- generation_config.json
 *
 * The checkpoint's sampling defaults, carried into the container so serving does not depend on a
 * separate file. Optional. Without them a model tuned with a truncated tail occasionally samples
 * far down its distribution, including end-of-turn, so replies stop early. Flattened under
 * `generation.` to avoid colliding with architecture keys such as `temperature` or `top_k`. */
int meta_merge_generation_config(const std::string& dir, MetaOwn* out) {
    const std::string path = dir + "/generation_config.json";
    std::ifstream f(path);
    if (!f) return RAD_OK;               /* optional */
    Json j;
    try { f >> j; }
    catch (const std::exception& e) {
        RAD_WARN("%s: %s; no sampling defaults are carried", path.c_str(), e.what());
        return RAD_OK;
    }
    if (!j.is_object()) {
        RAD_WARN("%s: not a JSON object; no sampling defaults are carried", path.c_str());
        return RAD_OK;
    }
    flatten_json(j, "generation", out);
    return RAD_OK;
}

/* ---------------------------------------------------------------- the preprocessor configs
 *
 * How images and video become patches, carried into the container like the sampling defaults.
 * core/mm reads pixel bands, normalisation and frame rate from these; the encoder's geometry is
 * in config.json's vision_config. Each file has its own prefix (`size.shortest_edge` differs
 * between them). Optional. */
int meta_merge_preprocessor_configs(const std::string& dir, MetaOwn* out) {
    /* processor_config.json is the newer single file, one object per sub-processor
     * (`image_processor`, `video_processor`, `feature_extractor`) beside its own keys
     * (EmbeddingGemma 2's image budget and audio token rate are only there). */
    static const struct { const char* file; const char* prefix; } kFiles[] = {
        { "preprocessor_config.json", "preprocessor" },
        { "video_preprocessor_config.json", "video_preprocessor" },
        { "processor_config.json", "processor" },
        /* sentence-transformers: an embedding model's named prompts and default. */
        { "config_sentence_transformers.json", "sentence_transformers" },
    };
    for (const auto& f : kFiles) {
        const std::string path = dir + "/" + f.file;
        std::ifstream in(path);
        if (!in) continue;
        Json j;
        try { in >> j; }
        catch (const std::exception& e) {
            RAD_WARN("%s: %s; it is not carried", path.c_str(), e.what());
            continue;
        }
        if (j.is_object()) flatten_json(j, f.prefix, out);
    }
    return RAD_OK;
}

int meta_from_config_json(const std::string& dir, MetaOwn* out) {
    const std::string path = dir + "/config.json";
    std::ifstream f(path);
    if (!f) {
        RAD_ERR("%s: no config.json. A safetensors checkpoint carries its architecture and its "
                "dimensions there; without it nothing can be sized.", dir.c_str());
        return RAD_E_NOTFOUND;
    }
    Json j;
    try { f >> j; }
    catch (const std::exception& e) {
        RAD_ERR("%s: %s", path.c_str(), e.what());
        return RAD_E_FORMAT;
    }

    flatten_json(j, "", out);

    /* Alias text_config.X (or llm_config.X) to a bare X where no bare X exists, so plugins read
     * the same free-form keys (e.g. `partial_rotary_factor`) with or without a vision tower. A
     * bare key wins where both exist. */
    {
        const size_t n = out->keys.size();
        for (const char* pre : { "text_config.", "llm_config." }) {
            const size_t len = std::strlen(pre);
            for (size_t i = 0; i < n; ++i) {
                const std::string& k = out->keys[i];
                if (k.rfind(pre, 0) != 0) continue;
                const std::string bare = k.substr(len);
                bool have = false;
                for (size_t j2 = 0; j2 < out->keys.size(); ++j2)
                    if (out->keys[j2] == bare) { have = true; break; }
                if (!have) out->kv(bare, out->vals[i]);
            }
        }
    }

    auto gs = [&](const char* k) -> const char* { return out->get(k); };
    /* text_config first, then llm_config, then the top level. */
    auto nested = [&](const char* k) -> const char* {
        const char* v = gs((std::string("text_config.") + k).c_str());
        if (!v) v = gs((std::string("llm_config.") + k).c_str());
        return v ? v : gs(k);
    };
    auto gi = [&](const char* k, int64_t dflt) -> int64_t {
        const char* v = nested(k);
        return v ? strtoll(v, nullptr, 10) : dflt;
    };
    auto gf = [&](const char* k, double dflt) -> double {
        const char* v = nested(k);
        return v ? strtod(v, nullptr) : dflt;
    };

    /* HuggingFace's model_type minus underscores is the architecture id, matching GGUF's spelling
     * (`qwen3_5` is `qwen35`). */
    if (const char* mt = gs("model_type")) {
        out->arch.clear();
        for (const char* p = mt; *p; ++p) if (*p != '_') out->arch += *p;
        /* LFM2-VL names the wrapper, not the text model; one plugin serves LFM2 with or without
         * its tower. `model_type` stays in the free-form keys. */
        if (out->arch == "lfm2vl") out->arch = "lfm2";
        /* Gemma 4 12B (`gemma4_unified`): its text model is Gemma 4's, served by the gemma4
         * plugins; `model_type` tells them apart. */
        if (out->arch == "gemma4unified") out->arch = "gemma4";
        /* Nemotron-H Omni wraps a Nemotron-H Puzzle text model (`llm_config`) the same way. */
        if (out->arch == "nemotronhomni") out->arch = "nemotronh";
    }
    if (const char* nm = gs("_name_or_path")) out->name = nm;

    RadModelMeta& m = out->m;
    /* Nomic BERT states its geometry with n_layer, n_embd, n_head, n_inner, n_positions and
     * moe_top_k. They are the last fallbacks in each chain, so a config carrying both spellings
     * reads the HuggingFace one. */
    m.n_layers    = gi("num_hidden_layers", gi("n_layer", 0));
    /* Nemotron-H Puzzle states no count: HF derives it from the per-layer list. */
    if (m.n_layers == 0)
        if (const char* types = nested("layers_block_type"))
            for (const char* p = types; *p;) {
                while (*p == ' ') ++p;
                if (!*p) break;
                ++m.n_layers;
                while (*p && *p != ' ') ++p;
            }
    m.n_embd      = gi("hidden_size", gi("n_embd", 0));
    m.n_head      = gi("num_attention_heads", gi("n_head", 0));
    m.n_head_kv   = gi("num_key_value_heads", m.n_head);
    /* Read head_dim; derive it only as a fallback. On qwen35 it is 256 while
     * hidden_size/num_attention_heads is 213. */
    m.head_dim    = gi("head_dim", m.n_head ? m.n_embd / m.n_head : 0);
    /* A routed config has no `intermediate_size`, only `moe_intermediate_size` (one expert's
     * width, which is what a GEMM sees and what n_ff means) and
     * `shared_expert_intermediate_size` (read by the plugin from the free-form table). Nomic
     * spells the same width n_inner. */
    m.n_ff        = gi("intermediate_size", gi("moe_intermediate_size", gi("n_inner", 0)));
    m.n_vocab     = gi("vocab_size", 0);
    m.n_ctx_train = gi("max_position_embeddings", gi("n_positions", 0));
    /* The DeepSeek spelling: Unlimited-OCR's decoder is a DeepSeek-V2 MoE and states its expert
     * count as `n_routed_experts`; a routed config names the count under one of the three and
     * never under more than one, so the chain is a rename, not a guess. */
    m.n_expert        = gi("num_experts", gi("num_local_experts", gi("n_routed_experts", 0)));
    m.n_expert_used   = gi("num_experts_per_tok", gi("moe_top_k", 0));
    m.n_expert_shared = gi("n_shared_experts", gi("num_shared_experts",
                           gi("shared_expert_intermediate_size", 0) ? 1 : 0));
    m.rms_eps     = (float)gf("rms_norm_eps", gf("norm_eps", gf("layer_norm_epsilon", 1e-6)));
    m.rope_theta  = (float)gf("rope_parameters.rope_theta",
                              gf("rope_theta", gf("rotary_emb_base", 10000.0)));
    /* Rotary scaling stays in the free-form keys with its type (arch/common/rad_arch.h,
     * yarn_from); a bare factor here would wrongly mean linear scaling. */
    m.rope_scale  = 1.0f;

    out->quant = quant_descriptor(*out);
    out->finish();
    return RAD_OK;
}

/* The quantisation descriptor, which with the architecture selects the plugin (spec §2.4): the
 * quant_method plus its fmt ("fp8_e4m3"), so the checkpoint, not a flag, distinguishes fp8 from
 * bf16 releases. transformers' fine-grained FP8 states no fmt (Qwen3.8-Flash-Next-FP8's config)
 * and is always E4M3, so a bare "fp8" is "fp8_e4m3", as a release that states it is.
 *
 * A Gemma 4 E2B/E4B checkpoint states no quantization_config: it is a bf16 release that
 * arch/gemma4_bf16 serves under "compressed-tensors", so it needs no
 * `--set quantization_config.quant_method=compressed-tensors`. Its per-layer embeddings or shared
 * KV layers name the E form; arch/gemma4 serves only the 26B-A4B/31B shape and refuses those two
 * keys, so the bare (gemma4, "") descriptor must not win. */
std::string quant_descriptor(const MetaOwn& m) {
    std::string q;
    if (const char* qm = m.get("quantization_config.quant_method")) {
        q = qm;
        const char* fm = m.get("quantization_config.fmt");
        if (!fm && q == "fp8") fm = "e4m3";
        if (fm) { q += "_"; q += fm; }
        return q;
    }
    if (m.arch == "gemma4") {
        auto gi = [&](const char* k) -> int64_t {
            const char* v = m.get(std::string("text_config.") + k);
            if (!v) v = m.get(k);
            return v ? std::strtoll(v, nullptr, 10) : 0;
        };
        if (gi("hidden_size_per_layer_input") > 0 || gi("num_kv_shared_layers") > 0)
            q = "compressed-tensors";
    }
    return q;
}

int meta_from_gguf(GgufFile& g, MetaOwn* out) {
    if (!g.get_str("general.architecture", &out->arch)) {
        RAD_ERR("the GGUF has no general.architecture, so no architecture plugin can be chosen");
        return RAD_E_FORMAT;
    }
    g.get_str("general.name", &out->name);

    /* Dimensions come from GGUF's `<arch>.<field>` keys, so one plugin covers a family; other keys
     * pass through in the free-form list. */
    const std::string a = out->arch;
    if (a == "hy_v3") out->arch = "hyv3";
    if (a == "bailingmoe3") out->arch = "bailinghybrid";
    auto gi = [&](const char* suffix, int64_t dflt) -> int64_t {
        int64_t v = 0;
        return g.get_i64(a + "." + suffix, &v) ? v : dflt;
    };
    auto gf = [&](const char* suffix, double dflt) -> double {
        double v = 0;
        return g.get_f64(a + "." + suffix, &v) ? v : dflt;
    };

    RadModelMeta& m = out->m;
    m.n_layers    = gi("block_count", 0);
    m.n_embd      = gi("embedding_length", 0);
    m.n_head      = gi("attention.head_count", 0);
    m.n_head_kv   = gi("attention.head_count_kv", m.n_head);
    m.head_dim    = gi("attention.key_length", m.n_head ? m.n_embd / m.n_head : 0);
    m.n_ff        = gi("feed_forward_length", 0);
    m.n_ctx_train = gi("context_length", 0);
    m.n_expert       = gi("expert_count", 0);
    m.n_expert_used  = gi("expert_used_count", 0);
    m.n_expert_shared= gi("expert_shared_count", 0);
    m.rms_eps     = (float)gf("attention.layer_norm_rms_epsilon", 1e-6);
    m.rope_theta  = (float)gf("rope.freq_base", 10000.0);
    m.rope_scale  = 1.0f;

    /* Rotary scaling, rewritten as config.json keys so one reader handles both formats
     * (arch/common/rad_arch.h, yarn_from). llama.cpp's YaRN attn_factor, ext_factor and log
     * multiplier have no config.json spelling, so a non-neutral value is refused rather than
     * served as plain YaRN. */
    {
        /* As llama.cpp reads it: type defaults to linear, the factor is rope.scaling.factor or
         * rope.scale_linear, and 0 or 1 means none. */
        std::string type = "linear";
        g.get_str(a + ".rope.scaling.type", &type);
        const double factor = gf("rope.scaling.factor", gf("rope.scale_linear", 0.0));
        const bool none = type == "none" || (type == "linear" && (factor == 0.0 || factor == 1.0));
        if (!none) {
            const struct { const char* key; double neutral; } kOff[] = {
                { "rope.scaling.attn_factor", 1.0 }, { "rope.scaling.yarn_attn_factor", 1.0 },
                { "rope.scaling.yarn_log_multiplier", 0.0 } };
            for (const auto& k : kOff)
                if (gf(k.key, k.neutral) != k.neutral) {
                    RAD_ERR("the GGUF sets %s.%s = %g; that modifier of the rotary's scaling is "
                            "not implemented", a.c_str(), k.key, gf(k.key, k.neutral));
                    return RAD_E_UNSUPPORTED;
                }
            const double ext = gf("rope.scaling.yarn_ext_factor", -1.0);
            if (ext >= 0.0 && ext != 1.0) {
                RAD_ERR("the GGUF sets %s.rope.scaling.yarn_ext_factor = %g; YaRN is implemented "
                        "at extrapolation factor 1", a.c_str(), ext);
                return RAD_E_UNSUPPORTED;
            }
            out->kv("rope_scaling.type", type);
            if (factor != 0.0) out->kv("rope_scaling.factor", fmt("%.17g", factor));
            const int64_t orig = gi("rope.scaling.original_context_length", 0);
            if (orig > 0) out->kv("rope_scaling.original_max_position_embeddings",
                                  fmt("%lld", (long long)orig));
            for (const char* beta : { "beta_fast", "beta_slow" }) {
                const double v = gf((std::string("rope.scaling.yarn_") + beta).c_str(), 0.0);
                if (v != 0.0) out->kv(std::string("rope_scaling.") + beta, fmt("%.17g", v));
            }
        }
    }

    std::vector<std::string> toks;
    if (g.get_strs("tokenizer.ggml.tokens", &toks)) m.n_vocab = (int64_t)toks.size();
    if (m.n_vocab == 0) m.n_vocab = gi("vocab_size", 0);
    if (a == "hy_v3") {
        /* GGUF keeps reserved vocabulary rows beyond the tokenizer string array. The model matrices
         * are authoritative for the row count (Hy-Memory has 120832 rows and 120818 strings). */
        const GgufTensor* emb = g.tensor("token_embd.weight");
        if (emb && emb->n_dims >= 2) m.n_vocab = emb->ne[1];

        int64_t gating = 0, norm = 0;
        double route_scale = 1.0;
        g.get_i64("hy_v3.expert_gating_func", &gating);
        g.get_i64("hy_v3.expert_weights_norm", &norm);
        g.get_f64("hy_v3.expert_weights_scale", &route_scale);
        const int64_t expert_width = gi("expert_feed_forward_length", 0);
        const int64_t shared_width = gi("expert_shared_feed_forward_length", 0);
        out->kv("qk_norm", g.tensor("blk.0.attn_q_norm.weight") &&
                             g.tensor("blk.0.attn_k_norm.weight") ? "1" : "0");
        out->kv("moe_router_use_sigmoid", gating == 2 ? "1" : "0");
        out->kv("moe_router_enable_expert_bias", g.tensor("blk.1.exp_probs_b") ? "1" : "0");
        out->kv("enable_moe_fp32_combine", shared_width > 0 ? "1" : "0");
        out->kv("route_norm", norm ? "1" : "0");
        out->kv("router_scaling_factor", fmt("%.17g", route_scale));
        out->kv("first_k_dense_replace", g.tensor("blk.0.ffn_gate.weight") &&
                                           g.tensor("blk.1.ffn_gate_inp.weight") ? "1" : "0");
        out->kv("num_shared_experts", shared_width > 0 ? "1" : "0");
        out->kv("expert_hidden_dim", fmt("%lld", (long long)expert_width));
        out->kv("moe_intermediate_size", fmt("%lld", (long long)(2 * expert_width)));
        out->kv("num_experts", fmt("%lld", (long long)m.n_expert));
        out->kv("num_experts_per_tok", fmt("%lld", (long long)m.n_expert_used));
        out->kv("attention_bias", "0");
        out->kv("mlp_bias", "0");
    }
    if (a == "bailingmoe3") {
        /* Ling 3.0 (the arch alias above): the release GGUF spells the family's keys
         * bailingmoe3.*, and the plugin (arch/bailing_hybrid) reads them as the checkpoint's
         * config.json names, so the reader re-spells them here and one plugin serves both
         * formats. eps and rope_theta take the generic paths above. Four of the family's
         * design choices the GGUF does not carry are Ling 3.0's, stated as constants. */
        if (gi("kda.head_dim", 0) <= 0 || gi("rope.dimension_count", 0) <= 0 ||
            gi("attention.key_length_mla", 0) <= gi("rope.dimension_count", 0))
            return RAD_E_FORMAT;
        out->kv("num_attention_heads", fmt("%lld", (long long)m.n_head));
        /* The typed head_dim the generic pass fell back to (the MLA query width) is not the
         * KDA head the plugin asserts; rad-convert publishes the typed fields before the kv
         * table, and lookups take the first row, so the typed field must carry the 128. */
        out->m.head_dim = (int)gi("kda.head_dim", 0);
        out->kv("head_dim", fmt("%lld", (long long)gi("kda.head_dim", 0)));
        /* The GGUF's key_length_mla is the full query (nope + rotary); the plugin's
         * qk_nope_head_dim is the rotary part subtracted. */
        out->kv("qk_nope_head_dim",
                fmt("%lld", (long long)(gi("attention.key_length_mla", 0) - gi("rope.dimension_count", 0))));
        out->kv("qk_rope_head_dim", fmt("%lld", (long long)gi("rope.dimension_count", 0)));
        out->kv("v_head_dim", fmt("%lld", (long long)gi("attention.value_length_mla", 0)));
        out->kv("kv_lora_rank", fmt("%lld", (long long)gi("attention.kv_lora_rank", 0)));
        out->kv("q_lora_rank", fmt("%lld", (long long)gi("attention.q_lora_rank", 0)));
        out->kv("num_experts", fmt("%lld", (long long)m.n_expert));
        out->kv("num_experts_per_tok", fmt("%lld", (long long)m.n_expert_used));
        out->kv("n_group", fmt("%lld", (long long)gi("expert_group_count", 0)));
        out->kv("topk_group", fmt("%lld", (long long)gi("expert_group_used_count", 0)));
        out->kv("moe_intermediate_size", fmt("%lld", (long long)gi("expert_feed_forward_length", 0)));
        out->kv("moe_shared_expert_intermediate_size",
                fmt("%lld", (long long)gi("expert_shared_feed_forward_length", 0)));
        out->kv("num_shared_experts", fmt("%lld", (long long)m.n_expert_shared));
        out->kv("intermediate_size", fmt("%lld", (long long)m.n_ff));
        out->kv("first_k_dense_replace", fmt("%lld", (long long)gi("leading_dense_block_count", 0)));
        out->kv("short_conv_kernel_size", fmt("%lld", (long long)gi("ssm.conv_kernel", 0)));
        out->kv("kda_lower_bound", fmt("%.17g", gf("kda.gate_lower_bound", 0.0)));
        out->kv("routed_scaling_factor", fmt("%.17g", gf("expert_weights_scale", 1.0)));
        out->kv("norm_topk_prob", fmt("%lld", (long long)gi("expert_weights_norm", 0)));
        out->kv("kda_safe_gate", fmt("%lld", (long long)gi("kda.safe_gate", 0)));
        out->kv("layer_group_size", "4");
        out->kv("no_kda_lora", "1");
        out->kv("rope_interleave", "1");
        out->kv("gated_attention_proj_granularity_type", "head_wise");
        out->kv("num_nextn_predict_layers", "0");
        /* The plugin picks its checkpoint names from this: the GGUF spells the family's tensors
         * blk. and ssm. names (and carries the layouts bailingmoe3_layouts rebuilds), the safetensors
         * their config.json spellings. */
        out->kv("gguf_source", "1");
        std::string gating = "sigmoid";
        g.get_str("bailingmoe3.expert_gating_func", &gating);
        out->kv("score_function", gating);
    }

    /* Everything else, stringified, for the plugin. Arrays are summarised (small integer arrays
     * spelled out); tokenizer keys are skipped, since the vocab has its own section. */
    for (const GgufKV& kv : g.kv()) {
        if (kv.key.rfind("tokenizer.", 0) == 0) continue;
        std::string v;
        if (kv.is_array) {
            v = fmt("[%zu values]", kv.type == GGUF_STR ? kv.s.size()
                                  : !kv.i.empty()       ? kv.i.size() : kv.f.size());
            if (kv.type != GGUF_STR && kv.i.size() <= 8 && !kv.i.empty()) {
                v.clear();
                for (size_t j = 0; j < kv.i.size(); ++j)
                    v += fmt("%s%lld", j ? "," : "", (long long)kv.i[j]);
            }
        } else if (!kv.s.empty()) v = kv.s.front();
        else if (!kv.f.empty() && kv.type == GGUF_F32) v = fmt("%g", kv.f.front());
        else if (!kv.f.empty() && kv.type == GGUF_F64) v = fmt("%g", kv.f.front());
        else if (!kv.i.empty()) v = fmt("%lld", (long long)kv.i.front());
        if (v.size() > 240) v.resize(240);
        out->kv(kv.key, v);
    }

    /* The descriptor a safetensors release would carry. A float file type (0 all F32, 1 mostly
     * F16, 32 mostly BF16) is unquantised and selects the bf16 plugin; any other names its
     * quantisation, which no plugin serves directly. */
    int64_t ftype = 0;
    const bool typed = g.get_i64("general.file_type", &ftype);
    out->quant = !typed || ftype == 0 || ftype == 1 || ftype == 32
                     ? std::string() : fmt("gguf_ftype_%lld", (long long)ftype);
    out->finish();
    return RAD_OK;
}

/* ================================================================== the checkpoint */

int Checkpoint::open_gguf(const std::string& path) {
    RAD_TRY(gguf_.open(path.c_str()));
    kind_ = Gguf;
    for (const GgufTensor& t : gguf_.tensors()) {
        SrcTensor s;
        s.name  = t.name;
        s.ggml  = t.ggml_type;
        s.rad_dtype = gguf_to_rad_dtype(t.ggml_type);
        /* GGUF is fastest-first; everything from here on is row-major. */
        for (int d = (int)t.n_dims - 1; d >= 0; --d) s.shape.push_back(t.ne[d]);
        s.data  = gguf_.tensor_data(t);
        s.bytes = (int64_t)t.bytes;
        if (!s.data) {
            RAD_ERR("%s: '%s' has no readable extent", path.c_str(), t.name.c_str());
            return RAD_E_FORMAT;
        }
        index_.emplace(s.name, tensors_.size());
        tensors_.push_back(std::move(s));
    }
    RAD_TRY(meta_from_gguf(gguf_, &meta_));
    if (meta_.arch == "bailinghybrid") RAD_TRY(bailingmoe3_layouts());
    return RAD_OK;
}

int Checkpoint::add_shard(const std::string& p, const std::string& prefix,
                          const std::unordered_map<std::string, std::string>* owner) {
    shards_.push_back(std::make_unique<SafeTensorsFile>());
    RAD_TRY(shards_.back()->open(p));
    const std::string file = p.substr(p.find_last_of('/') + 1);
    for (const auto& t : shards_.back()->tensors()) {
        /* The index decides which shard a tensor is read from: a directory merging two
         * checkpoints (a finetune's shards beside links to the base model's) holds some names
         * twice. */
        if (owner) {
            auto it = owner->find(t.name);
            if (it != owner->end() && it->second != file) continue;
        }
        if (index_.count(prefix + t.name)) {
            RAD_ERR("%s: '%s' is also in %s, and no index entry says which copy to read", p.c_str(),
                    t.name.c_str(), shards_[index_shard_[prefix + t.name]]->path().c_str());
            return RAD_E_FORMAT;
        }
        if (t.dtype == RAD_DT_INVALID) {
            RAD_ERR("%s: '%s' has dtype %s, which has no RAD equivalent", p.c_str(),
                    t.name.c_str(), t.dtype_str.c_str());
            return RAD_E_DTYPE;
        }
        SrcTensor s;
        s.name      = prefix + t.name;
        s.rad_dtype = t.dtype;
        s.shape     = t.shape;         /* safetensors is already row-major */
        s.data      = t.data;
        s.bytes     = (int64_t)t.bytes;
        index_shard_.emplace(s.name, shards_.size() - 1);
        index_.emplace(s.name, tensors_.size());
        tensors_.push_back(std::move(s));
    }
    return RAD_OK;
}

namespace {

/* Every shard an index names, once each and in name order, or the one model.safetensors; `map`
 * is the index's weight_map, empty without one. */
int shard_list(const std::string& dir, const std::string& index, bool index_given,
               std::vector<std::string>* out, std::unordered_map<std::string, std::string>* map) {
    const int s = st_load_index(index, map);
    if (s == RAD_OK) {
        for (auto& [t, f] : *map)
            if (std::find(out->begin(), out->end(), f) == out->end()) out->push_back(f);
        std::sort(out->begin(), out->end());
        for (auto& f : *out) f = dir + "/" + f;
        return RAD_OK;
    }
    if (s != RAD_E_NOTFOUND || index_given) return s;
    out->push_back(dir + "/model.safetensors");
    return RAD_OK;
}

}  /* namespace */


/* A JSON file flattened under `prefix`; absent is not an error when `optional`. */
static int meta_merge_json_file(const std::string& path, const std::string& prefix, bool optional,
                                MetaOwn* out) {
    std::ifstream f(path);
    if (!f) {
        if (optional) return RAD_OK;
        RAD_ERR("%s: cannot read it", path.c_str());
        return RAD_E_NOTFOUND;
    }
    Json j;
    try { f >> j; }
    catch (const std::exception& e) { RAD_ERR("%s: %s", path.c_str(), e.what()); return RAD_E_FORMAT; }
    if (!j.is_object()) {
        RAD_ERR("%s: not a JSON object", path.c_str());
        return RAD_E_FORMAT;
    }
    flatten_json(j, prefix, out);
    return RAD_OK;
}

static bool path_exists(const std::string& p) { std::ifstream f(p); return (bool)f; }

/* Whether an FP8 piece of `rows` rows has one scale for the whole tensor: a rank-0 scale
 * (ModelOpt's) or a one-element [1] (compressed-tensors' tensor strategy). A [1, 1] scale is a
 * block plane. */
static bool per_tensor_scale(const SrcTensor& s, int64_t rows) {
    return s.shape.empty() || (s.shape.size() == 1 && s.shape[0] == 1 && rows > 1);
}

/* "X.weight" -> "X"; false for a name with no such suffix. */
static bool weight_base(const std::string& name, std::string* base) {
    static const std::string suffix = ".weight";
    if (name.size() <= suffix.size() ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;
    *base = name.substr(0, name.size() - suffix.size());
    return true;
}

/* A diffusers pipeline: model_index.json (or a modular pipeline's modular_model_index.json) at the
 * top and one folder per component, each with its own config and weights, read as one checkpoint:
 *
 *   text_encoder/ or language_model/   the transformers model: its config.json is the
 *                   container's model config -- the typed dimensions and the bare keys -- and its
 *                   tensors keep their names;
 *   every other folder holding a diffusers model (config.json beside
 *                   diffusion_pytorch_model*.safetensors): tensors under `<folder>.`, config.json
 *                   under `<folder>.` keys (transformer/, vae/; MiniMax-Music3's
 *                   rvq_depth_decoder/, condition_encoder/, vocoder/);
 *   scheduler/      scheduler_config.json under `scheduler.` keys (any folder holding one, under
 *                   its own name);
 *   processor/ (or tokenizer/)  the tokenizer, the chat template and the preprocessor configs;
 *   <name>.json     any other JSON object beside the index under `<name>.` keys (a distilled
 *                   release's inference settings: its trained timesteps, its attention's sparsity).
 *
 * The index goes under `pipeline.` keys, and its `_class_name` names the architecture
 * (`QwenImage21Pipeline` is `qwenimage21`, `MiniMaxMusic3ModularPipeline` is `minimaxmusic3`). A
 * component folder may link to another pipeline's (a distilled release sharing its base's text
 * encoder). */
int Checkpoint::open_diffusers() {
    kind_ = SafeT;
    const bool modular = !path_exists(dir_ + "/model_index.json");
    RAD_TRY(meta_merge_json_file(dir_ + (modular ? "/modular_model_index.json" : "/model_index.json"),
                                 "pipeline", false, &meta_));
    {
        std::vector<std::string> tops;
        std::error_code ec;
        for (const auto& de : std::filesystem::directory_iterator(dir_, ec)) {
            if (!de.is_regular_file(ec) || de.path().extension() != ".json") continue;
            const std::string nm = de.path().filename().string();
            if (nm == "model_index.json" || nm == "modular_model_index.json") continue;
            tops.push_back(nm);
        }
        std::sort(tops.begin(), tops.end());
        for (const std::string& nm : tops)
            RAD_TRY(meta_merge_json_file(dir_ + "/" + nm, nm.substr(0, nm.size() - 5), false,
                                         &meta_));
    }
    std::string lm_dir;
    for (const char* sub : { "text_encoder", "language_model" })
        if (path_exists(dir_ + "/" + sub + "/config.json")) {
            lm_dir = dir_ + "/" + sub;
            break;
        }
    if (!lm_dir.empty()) {
        std::vector<std::string> shards;
        std::unordered_map<std::string, std::string> owner;
        RAD_TRY(shard_list(lm_dir, lm_dir + "/model.safetensors.index.json", false, &shards,
                           &owner));
        for (const auto& sh : shards) RAD_TRY(add_shard(sh, "", owner.empty() ? nullptr : &owner));
        RAD_TRY(meta_from_config_json(lm_dir, &meta_));
        RAD_TRY(meta_merge_generation_config(lm_dir, &meta_));
    }
    /* The diffusers models, in name order so the namespace does not depend on the directory's. */
    std::vector<std::string> comps;
    {
        std::error_code ec;
        for (const auto& de : std::filesystem::directory_iterator(dir_, ec)) {
            if (!de.is_directory(ec)) continue;
            const std::string nm = de.path().filename().string();
            const std::string d = dir_ + "/" + nm;
            if (d == lm_dir || !path_exists(d + "/config.json")) continue;
            if (path_exists(d + "/diffusion_pytorch_model.safetensors") ||
                path_exists(d + "/diffusion_pytorch_model.safetensors.index.json"))
                comps.push_back(nm);
        }
        std::sort(comps.begin(), comps.end());
    }
    for (const std::string& comp : comps) {
        const std::string d = dir_ + "/" + comp;
        const std::string idx = d + "/diffusion_pytorch_model.safetensors.index.json";
        std::vector<std::string> shards;
        std::unordered_map<std::string, std::string> owner;
        if (path_exists(idx)) {
            RAD_TRY(shard_list(d, idx, true, &shards, &owner));
        } else {
            shards.push_back(d + "/diffusion_pytorch_model.safetensors");
        }
        for (const auto& sh : shards)
            RAD_TRY(add_shard(sh, comp + ".", owner.empty() ? nullptr : &owner));
        RAD_TRY(meta_merge_json_file(d + "/config.json", comp, false, &meta_));
    }
    /* Every scheduler folder under its own name: scheduler/, and a second schedule's (MiniMax-H3's
     * audio_scheduler/). */
    {
        std::vector<std::string> scheds;
        std::error_code ec;
        for (const auto& de : std::filesystem::directory_iterator(dir_, ec))
            if (de.is_directory(ec) &&
                path_exists(de.path().string() + "/scheduler_config.json"))
                scheds.push_back(de.path().filename().string());
        std::sort(scheds.begin(), scheds.end());
        for (const std::string& sc : scheds)
            RAD_TRY(meta_merge_json_file(dir_ + "/" + sc + "/scheduler_config.json", sc, true,
                                         &meta_));
    }
    /* The tokenizer's own folder, or the text encoder's when the release keeps it there (FLUX.3
     * [klein]'s modular pipeline points its tokenizer at text_encoder/). */
    for (const char* sub : { "processor", "tokenizer", "text_encoder" }) {
        const std::string d = dir_ + "/" + sub;
        if (!path_exists(d + "/tokenizer.json")) continue;
        tok_dir_ = d;
        RAD_TRY(meta_merge_preprocessor_configs(d, &meta_));
        break;
    }
    /* The architecture is the pipeline's, whatever the language model calls itself. */
    if (const char* cn = meta_.get("pipeline._class_name")) {
        std::string a = cn;
        for (const std::string suf : { "ModularPipeline", "Pipeline" })
            if (a.size() > suf.size() && a.compare(a.size() - suf.size(), suf.size(), suf) == 0) {
                a.resize(a.size() - suf.size());
                break;
            }
        for (char& ch : a) ch = (char)std::tolower((unsigned char)ch);
        meta_.arch = a;
    }
    if (tensors_.empty()) {
        RAD_ERR("%s: a diffusers pipeline with no component weights", dir_.c_str());
        return RAD_E_NOTFOUND;
    }
    meta_.finish();
    return RAD_OK;
}

/* config.yaml's maps of scalars as dotted keys, a list's items space-separated, booleans as 1 and
 * 0 and a trailing comment dropped: the subset Hunyuan3D-2's configs are written in. Anything else
 * (a multi-line value, a list of maps) is refused, not guessed at. */
static int meta_from_yaml(const std::string& path, MetaOwn* out) {
    std::ifstream f(path);
    if (!f) {
        RAD_ERR("%s: cannot read", path.c_str());
        return RAD_E_IO;
    }
    std::vector<std::pair<size_t, std::string>> stack;   /* indentation, key */
    std::string line;
    int ln = 0;
    auto trim = [](std::string s) {
        const size_t a = s.find_first_not_of(" \t\r"), b = s.find_last_not_of(" \t\r");
        return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
    };
    while (std::getline(f, line)) {
        ++ln;
        if (const size_t h = line.find(" #"); h != std::string::npos) line.resize(h);
        const std::string body = trim(line);
        if (body.empty() || body[0] == '#') continue;
        const size_t ind = line.find_first_not_of(' ');
        const size_t colon = body.find(':');
        if (colon == std::string::npos || body[0] == '-') {
            RAD_ERR("%s:%d: not a `key: value` line of the YAML this importer reads", path.c_str(), ln);
            return RAD_E_FORMAT;
        }
        while (!stack.empty() && stack.back().first >= ind) stack.pop_back();
        const std::string key = trim(body.substr(0, colon));
        std::string val = trim(body.substr(colon + 1));
        std::string full;
        for (const auto& s : stack) full += s.second + ".";
        full += key;
        if (val.empty()) {
            stack.emplace_back(ind, key);
            continue;
        }
        if (val.front() == '[') {
            if (val.back() != ']') {
                RAD_ERR("%s:%d: a list must close on its line", path.c_str(), ln);
                return RAD_E_FORMAT;
            }
            std::string items;
            for (char& ch : val) if (ch == '[' || ch == ']' || ch == ',') ch = ' ';
            std::istringstream is(val);
            for (std::string it; is >> it;) items += (items.empty() ? "" : " ") + it;
            val = items;
        } else if (val == "True" || val == "true") {
            val = "1";
        } else if (val == "False" || val == "false") {
            val = "0";
        } else if (val.size() >= 2 && (val.front() == '"' || val.front() == '\'') &&
                   val.back() == val.front()) {
            val = val.substr(1, val.size() - 2);
        }
        out->set(full, val);
    }
    return RAD_OK;
}

/* A Hunyuan3D-2 shape model (tencent/Hunyuan3D-2's hunyuan3d-dit-v2-0 and its kin): a folder of
 * config.yaml and one safetensors holding three networks under `model.` (the DiT),
 * `conditioner.` (DINOv2's image encoder) and `vae.` (the shape VAE), the names kept. config.yaml's
 * keys go into the metadata by dotted path (model.params.hidden_size, vae.params.scale_factor,
 * conditioner.params.main_image_encoder.kwargs.config.num_hidden_layers, ...). The architecture
 * is `hunyuan3d2`, and the model reads no text (`radiance.text` none). */
int Checkpoint::open_hy3d(const std::string& weights) {
    kind_ = SafeT;
    RAD_TRY(add_shard(weights));
    RAD_TRY(meta_from_yaml(dir_ + "/config.yaml", &meta_));
    meta_.arch = "hunyuan3d2";
    meta_.set("radiance.text", "none");
    /* The typed dimensions are the denoiser's: its blocks, width and heads, read off its tensors;
     * the vocabulary is the one token of a model that reads no text, and the context a step's
     * rows (both branches of the 1370 conditioning rows and the latent). */
    {
        RadModelMeta& m = meta_.m;
        const SrcTensor* lat = find("model.latent_in.weight");
        const SrcTensor* mlp = find("model.double_blocks.0.img_mlp.0.weight");
        const SrcTensor* qn = find("model.double_blocks.0.img_attn.norm.query_norm.scale");
        const SrcTensor* vae = find("vae.post_kl.weight");
        int64_t layers = 0;
        while (find("model.double_blocks." + std::to_string(layers) + ".img_mod.lin.weight"))
            ++layers;
        for (int64_t s = 0; find("model.single_blocks." + std::to_string(s) + ".linear1.weight"); ++s)
            ++layers;
        if (!lat || lat->shape.size() != 2 || !mlp || mlp->shape.size() != 2 || !qn ||
            qn->shape.size() != 1 || qn->shape[0] <= 0 || !vae || !layers) {
            RAD_ERR("%s: Hunyuan3D-2's DiT or shape VAE is incomplete", weights.c_str());
            return RAD_E_FORMAT;
        }
        m.n_layers = layers;
        m.n_embd = lat->shape[0];
        m.head_dim = qn->shape[0];
        m.n_head = m.n_head_kv = m.n_embd / m.head_dim;
        m.n_ff = mlp->shape[0];
        m.n_vocab = 1;
        m.n_ctx_train = 2 * (1370 + 3072);
        m.rms_eps = 1e-6f;
        m.rope_theta = 10000.0f;
        m.rope_scale = 1.0f;
    }
    meta_.finish();
    return RAD_OK;
}

/* Whether `file` is a Hunyuan3D-2 shape checkpoint: its DiT and image encoder by name. */
static bool hy3d_weights(const std::string& file) {
    SafeTensorsFile probe;
    if (probe.open(file) != RAD_OK) return false;
    bool dit = false, enc = false;
    for (const auto& t : probe.tensors()) {
        dit = dit || t.name == "model.latent_in.weight";
        enc = enc || t.name.rfind("conditioner.main_image_encoder.", 0) == 0;
    }
    return dit && enc;
}

/* A single-file Stable Diffusion XL checkpoint in Stability's (sgm) layout, the form SDXL and its
 * finetunes (Illustrious, NoobAI, Pony and the rest) are published in: one safetensors holding
 *
 *   model.diffusion_model.*                the UNet, under the names it is trained and adapted
 *                                          (LoRA) under, kept as they are;
 *   conditioner.embedders.0.transformer.*  CLIP ViT-L's text model, already in transformers'
 *                                          CLIPTextModel names, kept;
 *   conditioner.embedders.1.model.*        OpenCLIP ViT-bigG's text tower in open_clip's names,
 *                                          read as transformers' CLIPTextModelWithProjection
 *                                          under conditioner.embedders.1.transformer.: each
 *                                          block's fused in_proj split into q/k/v_proj (row
 *                                          views, nothing copied), c_fc / c_proj as fc1 / fc2,
 *                                          ln_1 / ln_2 / ln_final as layer_norm1 / layer_norm2 /
 *                                          final_layer_norm, and the (in, out) text_projection
 *                                          transposed into a linear's (out, in)
 *                                          `text_projection.weight`;
 *   first_stage_model.*                    the VAE, its decoder read in diffusers' AutoencoderKL
 *                                          names under vae. (its 1 x 1 attention convolutions
 *                                          keep their rank-4 shape).
 *
 * A `v_pred` tensor marks a v-prediction model and `ztsnr` one trained to a zero terminal SNR
 * (ComfyUI's convention): they become the `sdxl.prediction` and `sdxl.zero_snr` keys. The
 * architecture is `sdxl`. The checkpoint holds no tokenizer; rad-convert takes CLIP's with
 * --tokenizer. */
int Checkpoint::open_sgm() {
    kind_ = SafeT;
    RAD_TRY(add_shard(path_));
    bool vpred = false, zsnr = false;
    std::vector<SrcTensor> add;
    std::vector<std::string> drop;
    auto rename = [&](const SrcTensor& t, const std::string& name) {
        SrcTensor v = t;
        v.name = name;
        add.push_back(std::move(v));
        drop.push_back(t.name);
    };
    const std::string g_old = "conditioner.embedders.1.model.";
    const std::string g_new = "conditioner.embedders.1.transformer.";
    const std::string tm = g_new + "text_model.";
    const std::string v_old = "first_stage_model.";
    for (const SrcTensor& t : tensors_) {
        const std::string& n = t.name;
        if (n == "v_pred") { vpred = true; continue; }
        if (n == "ztsnr") { zsnr = true; continue; }
        if (n.rfind(g_old, 0) == 0) {
            const std::string r = n.substr(g_old.size());
            if (r == "token_embedding.weight") {
                rename(t, tm + "embeddings.token_embedding.weight");
            } else if (r == "positional_embedding") {
                rename(t, tm + "embeddings.position_embedding.weight");
            } else if (r.rfind("ln_final.", 0) == 0) {
                rename(t, tm + "final_layer_norm." + r.substr(std::strlen("ln_final.")));
            } else if (r == "text_projection") {
                /* x @ P in open_clip; a linear's weight is P^T. */
                if (t.shape.size() != 2 || !t.data ||
                    (t.rad_dtype != RAD_F16 && t.rad_dtype != RAD_BF16 && t.rad_dtype != RAD_F32)) {
                    RAD_ERR("%s: '%s' is not a 2-D float matrix", path_.c_str(), n.c_str());
                    return RAD_E_FORMAT;
                }
                const int64_t a = t.shape[0], b = t.shape[1];
                const int64_t es = rad_dtype_bytes(t.rad_dtype, 1);
                owned_.push_back(std::make_unique<uint8_t[]>((size_t)(a * b * es)));
                uint8_t* dst = owned_.back().get();
                const uint8_t* src = (const uint8_t*)t.data;
                for (int64_t i = 0; i < a; ++i)
                    for (int64_t j = 0; j < b; ++j)
                        std::memcpy(dst + (size_t)((j * a + i) * es),
                                    src + (size_t)((i * b + j) * es), (size_t)es);
                SrcTensor v = t;
                v.name = g_new + "text_projection.weight";
                v.shape = { b, a };
                v.data = dst;
                add.push_back(std::move(v));
                drop.push_back(n);
            } else if (r.rfind("transformer.resblocks.", 0) == 0) {
                /* transformer.resblocks.N.<rest> */
                const size_t p0 = std::strlen("transformer.resblocks.");
                const size_t dot = r.find('.', p0);
                if (dot == std::string::npos) continue;
                const std::string layer = tm + "encoder.layers." + r.substr(p0, dot - p0) + ".";
                const std::string rest = r.substr(dot + 1);
                if (rest == "attn.in_proj_weight" || rest == "attn.in_proj_bias") {
                    const bool w = rest == "attn.in_proj_weight";
                    if (t.shape.empty() || t.shape[0] % 3 || !t.data) {
                        RAD_ERR("%s: '%s' does not split into q, k and v", path_.c_str(),
                                n.c_str());
                        return RAD_E_FORMAT;
                    }
                    const int64_t d = t.shape[0] / 3, rb = t.bytes / t.shape[0];
                    const char* part[3] = { "q_proj", "k_proj", "v_proj" };
                    for (int k = 0; k < 3; ++k) {
                        SrcTensor v = t;
                        v.name = layer + "self_attn." + part[k] + (w ? ".weight" : ".bias");
                        v.shape[0] = d;
                        v.bytes = rb * d;
                        v.data = (const uint8_t*)t.data + (size_t)(k * d * rb);
                        add.push_back(std::move(v));
                    }
                    drop.push_back(n);
                } else {
                    static const std::pair<const char*, const char*> kMap[] = {
                        { "attn.out_proj.", "self_attn.out_proj." }, { "ln_1.", "layer_norm1." },
                        { "ln_2.", "layer_norm2." }, { "mlp.c_fc.", "mlp.fc1." },
                        { "mlp.c_proj.", "mlp.fc2." } };
                    for (const auto& [from, to] : kMap)
                        if (rest.rfind(from, 0) == 0) {
                            rename(t, layer + to + rest.substr(std::strlen(from)));
                            break;
                        }
                }
            }
            continue;
        }
        if (n.rfind(v_old + "decoder.", 0) == 0 || n.rfind(v_old + "post_quant_conv.", 0) == 0) {
            /* diffusers' convert_ldm_vae_checkpoint, the decoder's half. */
            std::string r = n.substr(v_old.size());
            auto sub = [&](const std::string& from, const std::string& to) {
                const size_t at = r.find(from);
                if (at != std::string::npos) r.replace(at, from.size(), to);
            };
            sub("decoder.mid.block_1.", "decoder.mid_block.resnets.0.");
            sub("decoder.mid.block_2.", "decoder.mid_block.resnets.1.");
            sub("decoder.mid.attn_1.norm.", "decoder.mid_block.attentions.0.group_norm.");
            sub("decoder.mid.attn_1.q.", "decoder.mid_block.attentions.0.to_q.");
            sub("decoder.mid.attn_1.k.", "decoder.mid_block.attentions.0.to_k.");
            sub("decoder.mid.attn_1.v.", "decoder.mid_block.attentions.0.to_v.");
            sub("decoder.mid.attn_1.proj_out.", "decoder.mid_block.attentions.0.to_out.0.");
            sub("decoder.norm_out.", "decoder.conv_norm_out.");
            if (r.rfind("decoder.up.", 0) == 0) {
                /* up.i runs at the i-th resolution from the bottom of the encoder's pyramid;
                 * up_blocks count in the order the decoder runs them. */
                const size_t p0 = std::strlen("decoder.up."), dot = r.find('.', p0);
                if (dot == std::string::npos) continue;
                const int i = std::atoi(r.substr(p0, dot - p0).c_str());
                std::string rest = r.substr(dot + 1);
                if (rest.rfind("block.", 0) == 0) rest = "resnets." + rest.substr(6);
                else if (rest.rfind("upsample.", 0) == 0) rest = "upsamplers.0." + rest.substr(9);
                r = "decoder.up_blocks." + std::to_string(3 - i) + "." + rest;
            }
            sub(".nin_shortcut.", ".conv_shortcut.");
            rename(t, "vae." + r);
        }
    }
    for (const std::string& n : drop) index_.erase(n);
    for (SrcTensor& v : add) {
        if (index_.count(v.name)) {
            RAD_ERR("%s: '%s' is read from two tensors", path_.c_str(), v.name.c_str());
            return RAD_E_FORMAT;
        }
        index_.emplace(v.name, tensors_.size());
        tensors_.push_back(std::move(v));
    }
    meta_.arch = "sdxl";
    const size_t slash = path_.find_last_of('/');
    meta_.name = slash == std::string::npos ? path_ : path_.substr(slash + 1);
    meta_.kv("sdxl.prediction", vpred ? "v" : "eps");
    meta_.kv("sdxl.zero_snr", zsnr ? "1" : "0");
    /* The typed dimensions are the conditioning network's, OpenCLIP bigG's text tower (as a
     * diffusers pipeline's are its text encoder's), read off its tensors. */
    {
        RadModelMeta& m = meta_.m;
        const SrcTensor* tok = find(tm + "embeddings.token_embedding.weight");
        const SrcTensor* fc1 = find(tm + "encoder.layers.0.mlp.fc1.weight");
        int64_t layers = 0;
        while (find(tm + "encoder.layers." + std::to_string(layers) + ".layer_norm1.weight"))
            ++layers;
        if (!tok || tok->shape.size() != 2 || !fc1 || fc1->shape.size() != 2 || !layers) {
            RAD_ERR("%s: OpenCLIP bigG's text tower is incomplete", path_.c_str());
            return RAD_E_FORMAT;
        }
        m.n_layers = layers;
        m.n_embd = tok->shape[1];
        m.head_dim = 64;
        m.n_head = m.n_head_kv = m.n_embd / 64;
        m.n_ff = fc1->shape[0];
        m.n_vocab = tok->shape[0];
        m.n_ctx_train = 77;
        m.rms_eps = 1e-5f;
        m.rope_theta = 10000.0f;
        m.rope_scale = 1.0f;
    }
    meta_.finish();
    return RAD_OK;
}

int Checkpoint::open_safetensors(const std::string& path) {
    kind_ = SafeT;
    /* One shard, an index, or the directory holding either, as one namespace. */
    std::vector<std::string> shards;
    std::unordered_map<std::string, std::string> owner;
    const bool is_index = path.size() > 10 && path.compare(path.size() - 10, 10, "index.json") == 0;
    const bool is_file = path.size() > 12 &&
                         path.compare(path.size() - 12, 12, ".safetensors") == 0;
    /* A pipeline index at the top: a diffusers pipeline. A modular pipeline may also carry a
     * config.json there (MiniMax-Music3's names its model class), which is not a model's. */
    if (!is_file && !is_index &&
        ((path_exists(dir_ + "/model_index.json") && !path_exists(dir_ + "/config.json")) ||
         path_exists(dir_ + "/modular_model_index.json")))
        return open_diffusers();
    /* A Hunyuan3D-2 shape model: config.yaml beside its one safetensors (the fp16 one when the
     * folder holds both). */
    if (!is_index && path_exists(dir_ + "/config.yaml")) {
        std::string w = path;
        if (!is_file)
            for (const char* f : { "/model.fp16.safetensors", "/model.safetensors" })
                if (path_exists(dir_ + f)) { w = dir_ + f; break; }
        if (w != path || is_file)
            if (hy3d_weights(w)) return open_hy3d(w);
    }
    if (is_file) {
        shards.push_back(path);
        /* A single-file Stable Diffusion XL checkpoint, as Stability and every SDXL finetune
         * ship one: read by its own importer. */
        SafeTensorsFile probe;
        if (probe.open(path) == RAD_OK) {
            bool unet = false, bigg = false;
            for (const auto& t : probe.tensors()) {
                unet = unet || t.name == "model.diffusion_model.label_emb.0.0.weight";
                bigg = bigg ||
                       t.name.rfind("conditioner.embedders.1.model.transformer.resblocks.", 0) == 0;
            }
            if (unet && bigg) return open_sgm();
        }
    } else {
        RAD_TRY(shard_list(dir_, is_index ? path : dir_ + "/model.safetensors.index.json",
                           is_index, &shards, &owner));
    }
    for (const auto& sh : shards) RAD_TRY(add_shard(sh, "", owner.empty() ? nullptr : &owner));

    const std::string& md = meta_dir_.empty() ? dir_ : meta_dir_;
    RAD_TRY(meta_from_config_json(md, &meta_));
    RAD_TRY(meta_merge_generation_config(md, &meta_));
    RAD_TRY(meta_merge_preprocessor_configs(md, &meta_));
    if (!meta_dir_.empty() && !path_exists(dir_ + "/tokenizer.json")) tok_dir_ = meta_dir_;
    if (!is_file && !is_index) {
        RAD_TRY(add_companions());
        RAD_TRY(add_torch_files());
    }
    /* HYV3's FP8 release indexes its static per-layer K/V cache descales as scalar F32 tensors in
     * kv_cache_scales.safetensors. They are runtime attention parameters, not model weights; carry
     * them through metadata so the architecture can bake them into kv_store and descale operands. */
    if (meta_.arch == "hyv3") {
        for (int64_t l = 0; l < meta_.m.n_layers; ++l) {
            for (const char* which : {"k", "v"}) {
                const std::string source = "model.layers." + std::to_string(l) +
                                           ".self_attn." + which + "_cache.scale";
                const SrcTensor* t = find(source);
                if (!t) continue;
                if (t->rad_dtype != RAD_F32 || !t->data || t->bytes < 4 ||
                    (!t->shape.empty() && t->numel() != 1)) {
                    RAD_ERR("%s: HYV3 KV cache scale must be a scalar F32", source.c_str());
                    return RAD_E_FORMAT;
                }
                float scale = 0.0f;
                std::memcpy(&scale, t->data, sizeof scale);
                if (!std::isfinite(scale) || scale <= 0.0f) {
                    RAD_ERR("%s: HYV3 KV cache scale must be positive and finite", source.c_str());
                    return RAD_E_FORMAT;
                }
                char value[32];
                std::snprintf(value, sizeof value, "%.9g", (double)scale);
                meta_.set("hyv3.kv_cache." + std::to_string(l) + "." + which + "_scale", value);
            }
        }
    }
    /* FP8's block, where the checkpoint states one. */
    if (const char* bs = meta_.get("quantization_config.weight_block_size")) {
        long long r = 0, c = 0;
        if (std::sscanf(bs, "%lld %lld", &r, &c) == 2 && r > 0 && c > 0) {
            fp8_br_ = r;
            fp8_bc_ = c;
        }
    }
    detect_import();
    meta_.finish();
    return RAD_OK;
}

/* Companion models: a folder beside the model's own weights holding a model of its own (its
 * config.json and safetensors), such as Breeze-TTS-2's audio_tokenizer/ (the codec its frames are
 * played through). Each is read as part of the checkpoint, its tensors under `<folder>.` and its
 * config under `<folder>.` keys, in name order so the namespace does not depend on the
 * directory's; a plugin that has no use for one maps none of it. */
int Checkpoint::add_companions() {
    std::vector<std::string> comps;
    std::error_code ec;
    for (const auto& de : std::filesystem::directory_iterator(dir_, ec)) {
        if (!de.is_directory(ec)) continue;
        const std::string d = de.path().string();
        if (path_exists(d + "/config.json") &&
            (path_exists(d + "/model.safetensors") ||
             path_exists(d + "/model.safetensors.index.json")))
            comps.push_back(de.path().filename().string());
    }
    std::sort(comps.begin(), comps.end());
    for (const std::string& c : comps) {
        const std::string d = dir_ + "/" + c;
        std::vector<std::string> shards;
        std::unordered_map<std::string, std::string> owner;
        RAD_TRY(shard_list(d, d + "/model.safetensors.index.json", false, &shards, &owner));
        for (const auto& sh : shards)
            RAD_TRY(add_shard(sh, c + ".", owner.empty() ? nullptr : &owner));
        RAD_TRY(meta_merge_config_json(d, c, &meta_));
    }
    return RAD_OK;
}

/* PyTorch checkpoints beside the safetensors (torch_zip.h): a model whose part ships only as a
 * torch.save file, such as Fish Audio drama-3's codec.pth, read as data, never unpickled. Each is
 * read under `<stem>.`, in name order. A file the weights-only reader refuses (a pickled training
 * state, an optimiser) is not an error of the checkpoint, only of a plugin that maps its tensors,
 * which then finds them absent; the warning says why. A torch copy of the model's own weights
 * (`pytorch_model*.pth`, `consolidated*.pth`) is passed over, as the safetensors are read. */
int Checkpoint::add_torch_files() {
    std::vector<std::string> files;
    std::error_code ec;
    for (const auto& de : std::filesystem::directory_iterator(dir_, ec)) {
        if (!de.is_regular_file(ec)) continue;
        const std::string f = de.path().filename().string();
        if (f.size() <= 4 || f.compare(f.size() - 4, 4, ".pth") != 0) continue;
        if (f.rfind("pytorch_model", 0) == 0 || f.rfind("consolidated", 0) == 0) continue;
        files.push_back(f);
    }
    std::sort(files.begin(), files.end());
    for (const std::string& f : files) {
        auto tz = std::make_unique<TorchZipFile>();
        std::string why;
        const std::string path = dir_ + "/" + f;
        if (tz->open(path, &why) < 0) {
            RAD_WARN("%s: not read (%s); a plugin that maps its tensors will not find them",
                     path.c_str(), why.c_str());
            continue;
        }
        const std::string prefix = f.substr(0, f.size() - 4) + ".";
        for (const TorchTensor& t : tz->tensors()) {
            if (index_.count(prefix + t.name)) {
                RAD_ERR("%s: '%s%s' is already in the checkpoint", path.c_str(), prefix.c_str(),
                        t.name.c_str());
                return RAD_E_FORMAT;
            }
            SrcTensor s;
            s.name      = prefix + t.name;
            s.rad_dtype = t.dtype;
            s.shape     = t.shape;
            s.data      = t.data;
            s.bytes     = (int64_t)t.bytes;
            index_.emplace(s.name, tensors_.size());
            tensors_.push_back(std::move(s));
        }
        RAD_INFO("%s: %zu tensors under '%s', read from the pickle as data", path.c_str(),
                 tz->tensors().size(), prefix.c_str());
        torch_.push_back(std::move(tz));
    }
    return RAD_OK;
}

/* Which published quantisation this is, from quantization_config alone; import_piece checks the
 * tensors weight by weight, refusing a file at the first weight that disagrees. */
void Checkpoint::detect_import() {
    auto qc = [&](const std::string& k) {
        const char* v = meta_.get("quantization_config." + k);
        return v ? std::string(v) : std::string();
    };
    auto any_under = [&](const std::string& prefix) {
        for (const std::string& k : meta_.keys)
            if (k.rfind("quantization_config." + prefix, 0) == 0) return true;
        return false;
    };
    const std::string qm = qc("quant_method");
    if (qm == "auto-round") {
        /* AutoRound's GPTQ-compatible export; its own `auto_round` packing is not read. */
        const std::string pf = qc("packing_format");
        if (pf == "auto_round:auto_gptq") import_ = Import::Gptq;
        else import_why_ = "AutoRound's packing_format '" + pf + "' is not one this reads; its "
                           "auto_round:auto_gptq export is";
    } else if (qm == "gptq") {
        /* v1 (auto_gptq's) stores the zero point less one; gptq_v2 does not, and reading it as v1
         * would shift every code by one. */
        const std::string cf = qc("checkpoint_format");
        if (cf.empty() || cf == "gptq") import_ = Import::Gptq;
        else import_why_ = "GPTQ's checkpoint_format '" + cf + "' is not one this reads; 'gptq' is";
    } else if (qm == "quark") {
        const std::string w = "global_quant_config.weight.";
        /* Quark can retain block-FP8 attention/MTP weights alongside MXFP4 experts.
         * Validate every override rather than rejecting every nonempty override map.
         * Keep a lookup here: large exports have tens of thousands of flattened fields. */
        std::map<std::string, std::string> fields;
        for (size_t i = 0; i < meta_.keys.size(); ++i) fields.emplace(meta_.keys[i], meta_.vals[i]);
        const std::string lp = "quantization_config.layer_quant_config.";
        const std::string suffix = ".weight.dtype";
        std::map<std::string, bool> layers;
        for (const auto& [k, v] : fields)
            if (k.rfind(lp, 0) == 0 && k.size() > lp.size() + suffix.size() &&
                k.compare(k.size() - suffix.size(), suffix.size(), suffix) == 0)
                layers.emplace(k.substr(0, k.size() - suffix.size()) + ".", true);
        bool overrides_ok = true;
        for (const auto& [layer, unused] : layers) {
            auto f = [&](const char* key) {
                auto it = fields.find(layer + "weight." + key);
                return it == fields.end() ? std::string() : it->second;
            };
            const bool mx = f("dtype") == "fp4" && f("qscheme") == "per_group" &&
                            f("group_size") == "32" && f("scale_format") == "e8m0";
            const bool fp8 = f("dtype") == "fp8_e4m3" && f("qscheme") == "per_block" &&
                             f("block_size") == "128 128";
            overrides_ok = overrides_ok && (mx || fp8);
        }
        /* A partial/null weight override must not disappear merely because it has no dtype. */
        for (const auto& [k, v] : fields) if (k.rfind(lp, 0) == 0) {
            bool covered = false;
            for (size_t dot = k.rfind('.'); dot != std::string::npos;
                 dot = dot ? k.rfind('.', dot - 1) : std::string::npos)
                if (layers.count(k.substr(0, dot + 1))) { covered = true; break; }
            overrides_ok = overrides_ok && covered;
        }
        if (qc(w + "dtype") == "fp4" && qc(w + "qscheme") == "per_group" &&
            qc(w + "group_size") == "32" && qc(w + "scale_format") == "e8m0" &&
            overrides_ok && !any_under("layer_type_quant_config."))
            import_ = Import::QuarkMx;
        else
            import_why_ = "Quark requires global MXFP4 (fp4, per_group 32, e8m0 scales), with "
                          "only MXFP4 or 128x128 E4M3 block-FP8 layer overrides";
    } else if (qm == "compressed-tensors") {
        /* compressed-tensors states each scheme as a config group with a format. Three are read:
         *   pack-quantized        grouped int4, symmetric (e.g. Google's Gemma 4 QAT w4a16-ct)
         *                         or asymmetric with an integer zero point per group (W4A16_ASYM,
         *                         as llm-compressor's AWQ writes it); every group must be one,
         *                         and one group-reordering act-order, or a dynamic or activation
         *                         scheme, would change what a code means
         *   nvfp4-pack-quantized  NVFP4: e2m1 codes, an e4m3 scale per 16 of K, an f32 global
         *                         scale per tensor
         *   float-quantized       8-bit float weights, read by the FP8 path
         * Act-order `static` (an alias of `weight`) quantises the columns in activation order but
         * keeps each group's columns contiguous, so the weight reads as one without act-order;
         * `group` (alias `dynamic`) reorders the groups through X.weight_g_idx, which the int4
         * kernels do not follow. NVFP4 and float groups may be mixed (a mixed-precision release
         * ships NVFP4 MLPs beside FP8 linears); the tensors confirm each weight's form at
         * import_piece and ckpt_resolve. */
        std::vector<std::string> groups;
        const std::string cg = "quantization_config.config_groups.";
        for (const std::string& k : meta_.keys)
            if (k.rfind(cg, 0) == 0) {
                const std::string g = k.substr(cg.size(), k.find('.', cg.size()) - cg.size());
                if (std::find(groups.begin(), groups.end(), g) == groups.end()) groups.push_back(g);
            }
        std::string why;
        int64_t group = 0;
        int n_int4 = 0, n_nvfp4 = 0, n_fp8 = 0, n_asym = 0;
        if (groups.empty()) why = "its quantization_config names no config group";
        for (const std::string& g : groups) {
            if (!why.empty()) break;
            const std::string w = "config_groups." + g + ".weights.";
            /* A config without per-group formats states the one format at the top level. */
            std::string f = qc("config_groups." + g + ".format");
            if (f.empty()) f = qc("format");
            if (f == "nvfp4-pack-quantized") {
                if (qc(w + "num_bits") != "4" || qc(w + "type") != "float" ||
                    qc(w + "group_size") != "16")
                    why = "config group '" + g + "' is nvfp4-pack-quantized but its weights are not "
                          "4-bit floats in groups of 16";
                ++n_nvfp4;
            } else if (f == "float-quantized") {
                if (qc(w + "num_bits") != "8" || qc(w + "type") != "float")
                    why = "config group '" + g + "' is float-quantized but its weights are not "
                          "8-bit floats";
                ++n_fp8;
            } else if (f == "pack-quantized") {
                const std::string gs = qc(w + "group_size");
                const long long gn = gs.empty() ? 0 : std::atoll(gs.c_str());
                const std::string ao = qc(w + "actorder");
                if (qc("format") != "pack-quantized")
                    why = "its format is '" + qc("format") + "', and an int4 group is read only "
                          "in a pack-quantized config";
                else if (ao == "group" || ao == "dynamic")
                    why = "config group '" + g + "' has act-order '" + ao + "', which reorders "
                          "the columns' groups through weight_g_idx; the int4 GEMMs read groups "
                          "in column order, so only act-order 'static' or 'weight' (groups kept "
                          "contiguous) or none is read";
                else if (qc(w + "num_bits") != "4" || qc(w + "type") != "int" ||
                         (qc(w + "symmetric") != "1" && qc(w + "symmetric") != "0") ||
                         qc(w + "strategy") != "group" || qc(w + "dynamic") != "0" || gn <= 0 ||
                         (!ao.empty() && ao != "static" && ao != "weight") ||
                         any_under("config_groups." + g + ".input_activations."))
                    why = "config group '" + g + "' is not 4-bit int weights in groups, packed, "
                          "with unquantised activations, which is the one this reads";
                else if (group && gn != group)
                    why = "its config groups state different group sizes";
                else
                    group = gn;
                if (qc(w + "symmetric") == "0") ++n_asym;
                ++n_int4;
            } else {
                why = "config group '" + g + "' is in the format '" + f + "'";
            }
        }
        if (why.empty() && n_int4 && (n_nvfp4 || n_fp8))
            why = "it mixes pack-quantized int4 groups with float ones";
        if (why.empty() && n_asym && n_asym != n_int4)
            why = "it mixes symmetric and asymmetric int4 groups";
        if (!why.empty()) {
            import_why_ = "compressed-tensors: " + why + "; the formats read are pack-quantized "
                          "(int4, symmetric or with a zero point), nvfp4-pack-quantized and "
                          "float-quantized (8-bit float)";
        } else if (n_int4) {
            import_ = Import::CtInt4;
            ct_group_ = group;
            ct_asym_ = n_asym > 0;
        } else {
            import_ = n_nvfp4 ? Import::Nvfp4 : Import::CtFp8;
        }
    } else if (qm == "modelopt") {
        /* NVIDIA ModelOpt's export, one algorithm (`quant_algo`) or one per layer
         * (MIXED_PRECISION, `quantized_layers.<layer>.quant_algo`). Two are read:
         *   NVFP4  `X.weight` U8 [N, K/2] e2m1 codes, `X.weight_scale` e4m3 [N, K/16] and
         *          `X.weight_scale_2` f32, one per tensor, a multiplier of the e4m3 scale
         *   FP8    `X.weight` E4M3 [N, K] and `X.weight_scale` f32, one per tensor
         * `X.input_scale` (a static activation scale) is not read: activations are quantised
         * from their own values (the GEMM's activation quantiser). Any other algorithm (an
         * AWQ-scaled one ships a pre_quant_scale this does not apply) is refused by name. */
        std::vector<std::string> algos;
        const std::string ql = "quantization_config.quantized_layers.";
        for (size_t i = 0; i < meta_.keys.size(); ++i) {
            const std::string& k = meta_.keys[i];
            const std::string tail = ".quant_algo";
            if (k.rfind(ql, 0) == 0 && k.size() > ql.size() + tail.size() &&
                k.compare(k.size() - tail.size(), tail.size(), tail) == 0)
                algos.push_back(meta_.vals[i]);
            const std::string gtail = ".group_size";
            if (k.rfind(ql, 0) == 0 && k.size() > ql.size() + gtail.size() &&
                k.compare(k.size() - gtail.size(), gtail.size(), gtail) == 0 &&
                meta_.vals[i] != "16" && import_why_.empty())
                import_why_ = "ModelOpt's '" + k.substr(ql.size(), k.size() - ql.size() -
                                                         gtail.size()) +
                              "' has group_size " + meta_.vals[i] + "; NVFP4 is read in groups of 16";
        }
        const std::string top = qc("quant_algo");
        if (top != "MIXED_PRECISION") algos.push_back(top);
        if (top == "NVFP4" && !qc("group_size").empty() && qc("group_size") != "16" &&
            import_why_.empty())
            import_why_ = "ModelOpt's NVFP4 here has group_size " + qc("group_size") +
                          "; it is read in groups of 16";
        if (algos.empty() && import_why_.empty())
            import_why_ = "ModelOpt's quantization_config names no quant_algo";
        /* W4A16_NVFP4 is the same NVFP4 weight with bf16 activations, which is what this reader
         * serves either way: activations are quantised from their own values, never from a stored
         * input scale. NVIDIA's Muse Glimmer release states it per layer beside FP8. */
        for (const std::string& a : algos)
            if (a != "NVFP4" && a != "W4A16_NVFP4" && a != "FP8" && import_why_.empty())
                import_why_ = "ModelOpt's quant_algo '" + a + "' is not one this reads; NVFP4, "
                              "W4A16_NVFP4 and FP8 (per tensor), alone or per layer, are";
        if (import_why_.empty()) import_ = Import::ModelOpt;
        else import_why_ = "modelopt: " + import_why_;
    } else if (qm == "paroquant") {
        /* z-lab's ParoQuant at integer widths: "int5-bitplane" (AWQ nibbles for the low four bits
         * plus a fifth-bit plane) or AWQ-packed int4, group 128, asymmetric, with the learned
         * rotations beside. */
        if (qc("format") == "int5-bitplane" && qc("bits") == "5" && qc("group_size") == "128")
            import_ = Import::ParoI5;
        else if (qc("format").empty() && qc("bits") == "4" && qc("group_size") == "128")
            import_ = Import::ParoI4;       /* z-lab's own release: AWQ-packed int4, asymmetric */
        else
            import_why_ = "ParoQuant here is neither the int5-bitplane format (5-bit codes) nor z-lab's "
                          "AWQ-packed int4, in groups of 128, which are the ones this reads";
    } else if (qm == "paroquant_mxfp4") {
        /* Quark's MXFP4 planes in a rotated basis (ParoQuant retrained for the MXFP4 grid): e2m1
         * codes, an e8m0 exponent per 32 of K, and per projection the learned Givens rotations
         * (`pairs`, `theta`) and channel scales in groups of 128. */
        if (qc("format") == "mxfp4" && qc("mxfp4_block") == "32" && qc("group_size") == "128")
            import_ = Import::ParoMx;
        else
            import_why_ = "ParoQuant's MXFP4 here is not format mxfp4 with 32-element blocks and "
                          "rotation groups of 128, which is the one this reads";
    } else if (qm == "paroquant_m2xfp4") {
        /* The same rotated planes on libquant's `m2xfp4` grid: e2m1 codes, an e8m0 exponent per 32
         * of K, and one metadata byte a group whose four 2-bit subgroup multipliers refine that
         * exponent to (1 + k/4) . 2^E. `mx_block` states the group the planes are cut in and
         * `subgroup` the columns one multiplier covers. */
        if (qc("format") == "m2xfp4" && qc("mx_block") == "32" && qc("subgroup") == "8" &&
            qc("group_size") == "128")
            import_ = Import::ParoM2;
        else
            import_why_ = "ParoQuant's M2XFP4 here is not format m2xfp4 with 32-element groups, "
                          "8-element subgroups and rotation groups of 128, which is the one this "
                          "reads";
    }
    /* Rename the tool's quantization_config: plugins read that key as the served format
     * (qwen35_fp8 checks quant_method is "fp8"), while an imported checkpoint states each weight's
     * encoding. It stays in the container as provenance. */
    if (import_ != Import::None)
        for (std::string& k : meta_.keys)
            if (k.rfind("quantization_config.", 0) == 0) k = "imported_" + k;
}

int Checkpoint::import_piece(const std::string& name, const SrcTensor** codes,
                             const SrcTensor** scale, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    const Import im = imports_at(name);   /* a drafter's own form under its prefix */
    /* CtFp8 has no imported form: its FP8 weights read stock (ckpt_resolve). */
    if (im == Import::None || im == Import::CtFp8) return RAD_E_NOTFOUND;
    if (auto hit = imported_.find(name); hit != imported_.end()) {
        *codes = hit->second.first;
        *scale = hit->second.second;
        return RAD_OK;
    }

    if (im == Import::ParoI5 || im == Import::ParoI4) {
        const bool i5 = im == Import::ParoI5;
        /* Paro int5: `X.weight` is X.qweight/.qweight_hi (codes), X.scales (f16 [G, N]) and
         * X.qzeros/.qzeros_hi (zero points). The view describes the logical [N, K] u5 matrix;
         * ckpt_plane_rows re-lays the planes row by row. */
        std::string base;
        if (!weight_base(name, &base)) return RAD_E_NOTFOUND;
        const SrcTensor* qw = find(base + ".qweight");
        if (!qw) return RAD_E_NOTFOUND;
        const SrcTensor* hi = i5 ? find(base + ".qweight_hi") : nullptr;
        const SrcTensor* sc = find(base + ".scales");
        const SrcTensor* qz = find(base + ".qzeros");
        const SrcTensor* qzh = i5 ? find(base + ".qzeros_hi") : nullptr;
        if ((i5 && (!hi || !qzh)) || !sc || !qz || qw->rad_dtype != RAD_I32 ||
            qz->rad_dtype != RAD_I32 || sc->rad_dtype != RAD_F16 || qw->shape.size() != 2 ||
            qz->shape.size() != 2 || sc->shape.size() != 2 ||
            (i5 && (hi->rad_dtype != RAD_I32 || qzh->rad_dtype != RAD_I32 ||
                    hi->shape.size() != 2 || qzh->shape.size() != 2)))
            return refuse(RAD_E_FORMAT, "'" + qw->name + "' lacks an int32 " +
                                        (i5 ? "qweight_hi, qzeros_hi, " : "") +
                                        "qzeros and an f16 scales beside it");
        const int64_t K = qw->shape[0], N = qw->shape[1] * 8, G = sc->shape[0];
        if (N % 32 || K % 128 || G * 128 != K || sc->shape[1] != N || qz->shape[0] != G ||
            qz->shape[1] != N / 8 ||
            (i5 && (hi->shape[0] != K || hi->shape[1] != N / 32 || qzh->shape[0] != G ||
                    qzh->shape[1] != N / 32)))
            return refuse(RAD_E_SHAPE, "'" + qw->name + "' planes do not fit a [" +
                                       std::to_string(N) + ", " + std::to_string(K) + "] weight");
        SrcTensor c = *qw;
        c.name = name;
        c.rad_dtype = i5 ? RAD_U5 : RAD_U4;
        c.shape = { N, K };
        c.pbits = i5 ? 5 : 4;
        c.aux[0] = hi; c.aux[1] = qz; c.aux[2] = qzh;
        sliced_.push_back(std::move(c));
        *codes = &sliced_.back();
        *scale = sc;
        imported_[name] = { *codes, *scale };
        return RAD_OK;
    }

    if (im == Import::QuarkMx || im == Import::ParoMx || im == Import::ParoM2) {
        /* Quark's MXFP4 is `fp4_e2m1*e8m0[1x32]` as stored: two codes per byte (element 2j in the
         * low nibble), one biased exponent per 32 of K. The view only renames dtypes and the
         * logical width. Excluded layers are plain tensors. ParoQuant's M2XFP4 grid carries one
         * more plane (`weight_meta`, [N, K/32] u8) that is handed to the weight through
         * import_meta: its bytes are the encoding's third plane as they ship. */
        const bool m2 = im == Import::ParoM2;
        const SrcTensor* w = find(name);
        const SrcTensor* e = w ? find(name + "_scale") : nullptr;
        if (!w || !e || w->rad_dtype != RAD_U8) return RAD_E_NOTFOUND;
        if (e->rad_dtype != RAD_U8 || w->shape.size() != 2 || e->shape.size() != 2 ||
            e->shape[0] != w->shape[0] || e->shape[1] * 32 != w->shape[1] * 2)
            return refuse(RAD_E_SHAPE, "'" + e->name + "' is not the [N, K/32] exponent plane of '" +
                                       w->name + "'");
        const SrcTensor* mt = m2 ? find(name + "_meta") : nullptr;
        if (m2 && (!mt || mt->rad_dtype != RAD_U8 || mt->shape != e->shape))
            return refuse(RAD_E_FORMAT, "'" + w->name + "' has no [N, K/32] u8 '" + name +
                                        "_meta' beside its exponent plane, which the checkpoint's "
                                        "m2xfp4 grid stores the subgroup multipliers in");
        SrcTensor c = *w;
        c.rad_dtype = RAD_FP4E2M1;
        c.shape = { w->shape[0], w->shape[1] * 2 };
        SrcTensor x = *e;
        x.rad_dtype = RAD_E8M0;
        sliced_.push_back(std::move(c));
        *codes = &sliced_.back();
        sliced_.push_back(std::move(x));
        *scale = &sliced_.back();
        imported_[name] = { *codes, *scale };
        if (mt) imported_meta_[name] = mt;
        return RAD_OK;
    }

    if (im == Import::CtInt4) {
        /* CT-INT4: `X.weight` ships as X.weight_packed (i32 [N, K/8], eight codes per word from the
         * low bits, each the signed code plus 8) and X.weight_scale ([N, K/group]), and when
         * asymmetric X.weight_zero_point (see SrcTensor::ct_zero). compressed-tensors dequantises
         *     w = (q - zero_point) * scale
         * with q and zero_point the signed values (zero_point 0 when symmetric). The words are
         * already the [N, K] code matrix in row order: a symmetric code is the stored nibble less
         * 8 (i4), an asymmetric one the stored nibble (u4) under the stored zero (zero_point + 8).
         * An ignored layer is an ordinary X.weight. */
        std::string base;
        if (!weight_base(name, &base)) return RAD_E_NOTFOUND;
        const SrcTensor* pk = find(name + "_packed");
        if (!pk) return RAD_E_NOTFOUND;
        const SrcTensor* sc = find(name + "_scale");
        const SrcTensor* zp = find(name + "_zero_point");
        const std::string what = "'" + pk->name + "'";
        if (pk->rad_dtype != RAD_I32 || pk->shape.size() != 2)
            return refuse(RAD_E_FORMAT, what + " is not an int32 [N, K/8] matrix");
        if (zp && !ct_asym_)
            return refuse(RAD_E_FORMAT, what + " has a zero point beside it, which its symmetric "
                                       "config does not state");
        if (!zp && ct_asym_)
            return refuse(RAD_E_FORMAT, what + " has no '" + name + "_zero_point' beside it, "
                                       "which its asymmetric config states");
        const int64_t N = pk->shape[0], K = pk->shape[1] * 8;
        if (K % ct_group_)
            return refuse(RAD_E_SHAPE, what + " has " + std::to_string(K) + " columns, not whole "
                                       "groups of " + std::to_string(ct_group_));
        if (!sc || sc->shape.size() != 2 || sc->shape[0] != N || sc->shape[1] != K / ct_group_ ||
            (sc->rad_dtype != RAD_F16 && sc->rad_dtype != RAD_BF16 && sc->rad_dtype != RAD_F32))
            return refuse(RAD_E_FORMAT, what + " has no float [" + std::to_string(N) + ", " +
                                        std::to_string(K / ct_group_) + "] '" + name +
                                        "_scale' beside it");
        const int64_t G = K / ct_group_;
        if (zp && (zp->rad_dtype != RAD_I32 || zp->shape.size() != 2 ||
                   zp->shape[0] != (N + 7) / 8 || zp->shape[1] != G))
            return refuse(RAD_E_FORMAT, "'" + zp->name + "' is not the int32 [" +
                                        std::to_string((N + 7) / 8) + ", " + std::to_string(G) +
                                        "] zero points of " + what + ", packed along its rows");
        /* Act-order 'static' / 'weight' ships no group index; one that does must keep every
         * column in its own group. */
        if (const SrcTensor* gi = find(name + "_g_idx")) {
            if ((gi->rad_dtype != RAD_I32 && gi->rad_dtype != RAD_I64) || gi->numel() != K)
                return refuse(RAD_E_FORMAT, "'" + gi->name + "' is not an integer [K] group index");
            for (int64_t k = 0; k < K; ++k)
                if ((int64_t)rad_load_f32(gi->data, gi->rad_dtype, k) != k / ct_group_)
                    return refuse(RAD_E_UNSUPPORTED, "'" + gi->name + "' reorders the columns' "
                                                     "groups (act-order); only an in-order index "
                                                     "is read");
        }
        if (const SrcTensor* ws = find(name + "_shape")) {
            int64_t dims[2] = { 0, 0 };
            for (int i = 0; i < 2 && i < (int)ws->numel(); ++i)
                dims[i] = (int64_t)rad_load_f32(ws->data, ws->rad_dtype, i);
            if (ws->numel() != 2 || dims[0] != N || dims[1] != K)
                return refuse(RAD_E_SHAPE, "'" + ws->name + "' does not state the [" +
                                           std::to_string(N) + ", " + std::to_string(K) +
                                           "] its codes hold");
        }
        SrcTensor c = *pk;
        c.rad_dtype = zp ? RAD_U4 : RAD_I4;
        c.shape = { N, K };
        c.ct_group = ct_group_;
        c.ct_zero = zp;
        sliced_.push_back(std::move(c));
        *codes = &sliced_.back();
        *scale = sc;
        imported_[name] = { *codes, *scale };
        return RAD_OK;
    }

    if (im == Import::ModelOpt) {
        /* ModelOpt's NVFP4: `X.weight` U8 [N, K/2] (two e2m1 codes a byte, the even k in the low
         * nibble), `X.weight_scale` e4m3 [N, K/16] and `X.weight_scale_2` f32, one per tensor.
         * ModelOpt dequantises
         *     w = e2m1(code) * (float(weight_scale) * weight_scale_2)
         * so the second level multiplies (compressed-tensors' global scale divides). The f32 scale
         * plane is that product, one per 16 of K, owned here; the code times it is ModelOpt's f32
         * value. A weight whose `X.weight` is not U8 (an FP8 or bf16 linear) reads stock. */
        const SrcTensor* p = find(name);
        if (!p || p->rad_dtype != RAD_U8) return RAD_E_NOTFOUND;
        std::string base;
        if (!weight_base(name, &base) || p->shape.size() != 2)
            return refuse(RAD_E_FORMAT, "'" + name + "' is a U8 tensor and not the [N, K/2] codes "
                                        "of a ModelOpt NVFP4 linear");
        const SrcTensor* s = find(base + ".weight_scale");
        const SrcTensor* s2 = find(base + ".weight_scale_2");
        const int64_t N = p->shape[0], K = p->shape[1] * 2;
        if (!s || s->rad_dtype != RAD_F8E4M3 || s->shape.size() != 2 || s->shape[0] != N ||
            K % 16 || s->shape[1] != K / 16)
            return refuse(RAD_E_FORMAT, "'" + name + "' has no e4m3 [" + std::to_string(N) + ", " +
                                        std::to_string(K / 16) + "] weight_scale beside its "
                                        "packed codes, which is ModelOpt's NVFP4");
        if (!s2 || s2->rad_dtype != RAD_F32 || s2->bytes != 4)
            return refuse(RAD_E_FORMAT, "'" + name + "' has no single f32 weight_scale_2 beside "
                                        "its packed codes, which is ModelOpt's NVFP4");
        float m;
        std::memcpy(&m, s2->data, 4);
        if (!(m > 0.0f) || !std::isfinite(m))
            return refuse(RAD_E_FORMAT, "'" + s2->name + "' is not a usable f32 multiplier: " +
                                        std::to_string(m));
        const int64_t nsc = N * (K / 16);
        auto raw = std::make_unique<uint8_t[]>((size_t)nsc * 4);
        float* tab = reinterpret_cast<float*>(raw.get());
        const uint8_t* sb = (const uint8_t*)s->data;
        for (int64_t i = 0; i < nsc; ++i) {
            tab[i] = rad_fp8e4m3_to_f32(sb[(size_t)i]) * m;
            if (!std::isfinite(tab[i]))
                return refuse(RAD_E_FORMAT, "'" + s->name + "' holds a scale that is not finite "
                                            "at element " + std::to_string(i));
        }
        owned_.push_back(std::move(raw));
        SrcTensor sv = *s;
        sv.rad_dtype = RAD_F32;
        sv.data      = owned_.back().get();
        sv.bytes     = nsc * 4;
        SrcTensor pv = *p;
        pv.rad_dtype = RAD_FP4E2M1;
        pv.shape     = { N, K };
        sliced_.push_back(std::move(pv));
        *codes = &sliced_.back();
        sliced_.push_back(std::move(sv));
        *scale = &sliced_.back();
        imported_[name] = { *codes, *scale };
        return RAD_OK;
    }

    if (im == Import::Nvfp4) {
        /* NVFP4 (compressed-tensors' nvfp4-pack-quantized): `X.weight_packed` U8 [N, K/2], two
         * e2m1 codes per byte with the even k in the low nibble; `X.weight_scale` e4m3 [N, K/16];
         * `X.weight_global_scale` f32, one per tensor. The global scale is a divisor: the value is
         * code * (e4m3 / global), as compressed-tensors' dequantise and vLLM's loader (which
         * stores 1 / global) compute it. The affine scheme multiplies, so the scale plane is that
         * quotient in f32, one per 16 of K, owned here. A weight without `X.weight_packed` is not
         * NVFP4 (a mixed release ships FP8 and bf16 linears beside it) and reads stock. */
        std::string base;
        if (!weight_base(name, &base)) return RAD_E_NOTFOUND;
        const SrcTensor* p = find(base + ".weight_packed");
        if (!p) return RAD_E_NOTFOUND;
        if (p->rad_dtype != RAD_U8 || p->shape.size() != 2)
            return refuse(RAD_E_FORMAT, "'" + p->name + "' is not the U8 [N, K/2] packed tensor the"
                                        " NVFP4 form ships beside its scales");
        const SrcTensor* s = find(base + ".weight_scale");
        if (!s || s->rad_dtype != RAD_F8E4M3 || s->shape.size() != 2)
            return refuse(RAD_E_FORMAT, "'" + name + "' has no e4m3 [N, K/16] weight_scale beside "
                                        "its packed codes, which is the form this reads");
        const SrcTensor* g = find(base + ".weight_global_scale");
        if (!g || g->rad_dtype != RAD_F32 || g->bytes != 4)
            return refuse(RAD_E_FORMAT, "'" + name + "' has no single f32 weight_global_scale "
                                        "beside its packed codes, which is the form this reads");
        const int64_t N = p->shape[0], K = p->shape[1] * 2;
        if (N <= 0 || p->shape[1] <= 0 || K % 16)
            return refuse(RAD_E_SHAPE, "'" + p->name + "' packs " + std::to_string(K) +
                                       " columns of K, which the 16-wide e4m3 scale does not "
                                       "divide");
        if (s->shape[0] != N || s->shape[1] != K / 16)
            return refuse(RAD_E_SHAPE, "'" + s->name + "' is not the [" + std::to_string(N) +
                                       ", " + std::to_string(K / 16) + "] e4m3 scale of '" +
                                       p->name + "'");
        if (const SrcTensor* ws = find(base + ".weight_shape")) {
            int64_t dims[2] = { 0, 0 };
            for (int i = 0; i < 2 && i < (int)ws->numel(); ++i)
                dims[i] = (int64_t)rad_load_f32(ws->data, ws->rad_dtype, i);
            if (ws->numel() != 2 || dims[0] != N || dims[1] != K)
                return refuse(RAD_E_SHAPE, "'" + ws->name + "' does not state the [" +
                                           std::to_string(N) + ", " + std::to_string(K) +
                                           "] its codes hold");
        }
        float d;
        std::memcpy(&d, g->data, 4);
        if (!(d > 0.0f) || !std::isfinite(d))
            return refuse(RAD_E_FORMAT, "'" + g->name + "' is not a usable f32 divisor: " +
                                        std::to_string(d));
        /* e4m3 is exact in f32, so each entry is one f32 division, as in the reference dequant. */
        const int64_t nsc = N * (K / 16);
        auto raw = std::make_unique<uint8_t[]>((size_t)nsc * 4);
        float* tab = reinterpret_cast<float*>(raw.get());
        const uint8_t* sb = (const uint8_t*)s->data;
        for (int64_t i = 0; i < nsc; ++i) {
            tab[i] = rad_fp8e4m3_to_f32(sb[(size_t)i]) / d;
            if (!std::isfinite(tab[i]))
                return refuse(RAD_E_FORMAT, "'" + s->name + "' holds a scale that is not finite "
                                            "at element " + std::to_string(i));
        }
        owned_.push_back(std::move(raw));
        SrcTensor sv = *s;
        sv.rad_dtype = RAD_F32;
        sv.data      = owned_.back().get();
        sv.bytes     = nsc * 4;
        SrcTensor pv = *p;
        pv.rad_dtype = RAD_FP4E2M1;
        pv.shape     = { N, K };
        sliced_.push_back(std::move(pv));
        *codes = &sliced_.back();
        sliced_.push_back(std::move(sv));
        *scale = &sliced_.back();
        imported_[name] = { *codes, *scale };
        return RAD_OK;
    }

    /* GPTQ-packed: `X.weight` ships as X.qweight, X.scales, X.qzeros and optionally X.g_idx. The
     * width is read from the tensors, not quantization_config, because a release may mix widths
     * (int4 experts beside int8 projections) and state them only per layer. */
    std::string base;
    if (!weight_base(name, &base)) return RAD_E_NOTFOUND;
    const SrcTensor* qw = find(base + ".qweight");
    if (!qw) return RAD_E_NOTFOUND;
    const SrcTensor* sc = find(base + ".scales");
    const SrcTensor* qz = find(base + ".qzeros");
    const SrcTensor* gi = find(base + ".g_idx");
    const std::string what = "'" + qw->name + "'";
    if (qw->rad_dtype != RAD_I32 || qw->shape.size() != 2)
        return refuse(RAD_E_FORMAT, what + " is not an int32 [K*bits/32, N] matrix");
    if (!sc || sc->shape.size() != 2 ||
        (sc->rad_dtype != RAD_F16 && sc->rad_dtype != RAD_BF16 && sc->rad_dtype != RAD_F32))
        return refuse(RAD_E_FORMAT, what + " has no float [groups, N] '" + base + ".scales' beside it");
    if (!qz || qz->rad_dtype != RAD_I32 || qz->shape.size() != 2)
        return refuse(RAD_E_FORMAT, what + " has no int32 '" + base + ".qzeros' beside it");
    const int64_t N = qw->shape[1];
    const int64_t bits = N > 0 && (qz->shape[1] * 32) % N == 0 ? qz->shape[1] * 32 / N : 0;
    if (bits != 4 && bits != 8)
        return refuse(RAD_E_UNSUPPORTED, what + " packs " + std::to_string(bits) + "-bit codes "
                                         "(from its qzeros' width); 4 and 8 are read");
    const int64_t K = qw->shape[0] * 32 / bits;
    const int64_t G = sc->shape[0];
    if (sc->shape[1] != N || G <= 0 || K % G)
        return refuse(RAD_E_SHAPE, "'" + sc->name + "' is not a [groups, " + std::to_string(N) +
                                   "] scale tensor for " + std::to_string(K) + " columns");
    const int64_t group = K / G;
    /* The served scale covers 128 of K: wider groups (per-channel included) repeat it; narrower
     * groups cannot be represented. */
    if (K % 128 || group % 128)
        return refuse(RAD_E_UNSUPPORTED, what + " has groups of " + std::to_string(group) +
                                         " columns; the imported form scales a 128 of K, so the "
                                         "group must be a multiple of 128");
    if (qz->shape[0] != G)
        return refuse(RAD_E_SHAPE, "'" + qz->name + "' has " + std::to_string(qz->shape[0]) +
                                   " rows for " + std::to_string(G) + " groups");
    /* Symmetric only: every zero point must be v1's 2^(bits-1) - 1, so a code offsets to its
     * signed value. An asymmetric weight would need a zero plane. */
    const uint32_t zero = bits == 4 ? 0x77777777u : 0x7f7f7f7fu;
    for (int64_t i = 0; i < qz->numel(); ++i) {
        uint32_t v;
        std::memcpy(&v, (const uint8_t*)qz->data + i * 4, 4);
        if (v != zero)
            return refuse(RAD_E_UNSUPPORTED, "'" + qz->name + "' holds a zero point other than "
                                             "the symmetric one; an asymmetric weight is not read");
    }
    /* In order only: act-order (desc_act) permutes column groups, which a [1x128] scale cannot
     * follow, so g_idx must be absent or the identity. */
    if (gi) {
        if (gi->rad_dtype != RAD_I32 || gi->numel() != K)
            return refuse(RAD_E_FORMAT, "'" + gi->name + "' is not an int32 [K] group index");
        for (int64_t k = 0; k < K; ++k) {
            int32_t v;
            std::memcpy(&v, (const uint8_t*)gi->data + k * 4, 4);
            if (v != k / group)
                return refuse(RAD_E_UNSUPPORTED, "'" + gi->name + "' reorders the columns' groups "
                                                 "(act-order); only an in-order index is read");
        }
    }
    SrcTensor c = *qw;
    c.rad_dtype = bits == 4 ? RAD_I4 : RAD_I8;
    c.shape = { N, K };
    c.gptq_bits = (int32_t)bits;
    c.gptq_group = group;
    sliced_.push_back(std::move(c));
    *codes = &sliced_.back();
    *scale = sc;
    imported_[name] = { *codes, *scale };
    return RAD_OK;
}

const SrcTensor* Checkpoint::import_meta(const std::string& name) const {
    auto it = imported_meta_.find(name);
    return it == imported_meta_.end() ? nullptr : it->second;
}

int Checkpoint::open(const std::string& path) {
    path_ = path;
    const bool gguf = path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0;
    dir_ = path;
    const size_t slash = path.find_last_of('/');
    auto ends = [&](const char* s) {
        const size_t n = std::strlen(s);
        return path.size() > n && path.compare(path.size() - n, n, s) == 0;
    };
    const bool is_file = gguf || ends(".safetensors") || ends("index.json");
    if (is_file) dir_ = slash == std::string::npos ? "." : path.substr(0, slash);
    return gguf ? open_gguf(path) : open_safetensors(path);
}

int Checkpoint::add_draft(const std::string& dir, const std::string& prefix) {
    if (kind_ != SafeT) {
        RAD_ERR("a drafter is merged into a safetensors target's namespace; %s is not one",
                path_.c_str());
        return RAD_E_UNSUPPORTED;
    }
    RAD_TRY(meta_merge_config_json(dir, prefix.empty() || prefix.back() != '.'
                                            ? prefix
                                            : prefix.substr(0, prefix.size() - 1), &meta_));
    std::vector<std::string> shards;
    std::unordered_map<std::string, std::string> owner;
    RAD_TRY(shard_list(dir, dir + "/model.safetensors.index.json", false, &shards, &owner));
    for (const auto& sh : shards) RAD_TRY(add_shard(sh, prefix, owner.empty() ? nullptr : &owner));
    /* The drafter's own quantisation. One is read: ModelOpt's NVFP4 in groups of 16 (LiLiCorr
     * ships `modelopt_fp4` / `W4A16_NVFP4`: NVFP4 weights, bf16 activations, which a recipe rule
     * decodes like the target's NVFP4). Its names take that form (imports_at); anything else
     * quantised is refused by name rather than read as plain tensors. */
    const std::string dq = prefix + "quantization_config.";
    auto qd = [&](const char* k) {
        const char* v = meta_.get(dq + k);
        return v ? std::string(v) : std::string();
    };
    const std::string qm = qd("quant_method");
    if (!qm.empty()) {
        const std::string qa = qd("quant_algo"), gs = qd("group_size");
        const bool modelopt = qm == "modelopt" || qm == "modelopt_fp4";
        if (!modelopt || (qa != "NVFP4" && qa != "W4A16_NVFP4") || (!gs.empty() && gs != "16")) {
            RAD_ERR("drafter %s: quantization_config %s / %s (group %s) is not one this reads; "
                    "ModelOpt's NVFP4 in groups of 16 is", dir.c_str(), qm.c_str(), qa.c_str(),
                    gs.empty() ? "unstated" : gs.c_str());
            return RAD_E_UNSUPPORTED;
        }
        if (import_ != Import::None && import_ != Import::ModelOpt) {
            RAD_ERR("drafter %s: an NVFP4 drafter beside a target imported in another form is not "
                    "read", dir.c_str());
            return RAD_E_UNSUPPORTED;
        }
        draft_import_ = Import::ModelOpt;
        draft_prefix_ = prefix;
        /* As the target's: the tool's config is provenance, not a served format. */
        for (std::string& k : meta_.keys)
            if (k.rfind(dq, 0) == 0) k = prefix + "imported_" + k.substr(prefix.size());
    }
    meta_.finish();
    return RAD_OK;
}

/* PEFT exports wrap HF module paths in the name of the net the adapter trained: ai-toolkit spells
 * a DiT module `diffusion_model.<path>`, diffusers' wrappers `transformer.` or `unet.`. */
static const char* const kLoraPrefixes[] = { "diffusion_model.", "transformer.", "unet." };

/* One `modules` rule of the map: the module pattern split at its single `*`, and the target
 * patterns into which the matched run is substituted, in B-row order. */
struct LoraRule {
    bool                     star = false;
    std::string              pre, suf;
    std::vector<std::string> targets;
    bool match(const std::string& m, std::string* tok) const {
        if (!star) {
            if (m != pre) return false;
            tok->clear();
            return true;
        }
        if (m.size() < pre.size() + suf.size() || m.compare(0, pre.size(), pre) != 0) return false;
        if (m.compare(m.size() - suf.size(), suf.size(), suf) != 0) return false;
        *tok = m.substr(pre.size(), m.size() - pre.size() - suf.size());
        return !tok->empty();
    }
};

/* Read the map file: {"strip": [...], "modules": {"<pat>": ["<t>", ...], ...}}. The module
 * order is preserved (the DOM keeps insertion order), so the first rule whose
 * pattern matches a module wins. */
static int lora_map_read(const std::string& file, std::vector<std::string>* strip,
                         std::vector<LoraRule>* rules) {
    std::ifstream f(file);
    if (!f) { RAD_ERR("lora: --lora-map %s cannot be read", file.c_str()); return RAD_E_NOTFOUND; }
    Json j;
    try {
        j = Json::parse(f);
    } catch (const std::exception& e) {
        RAD_ERR("lora: --lora-map %s: %s", file.c_str(), e.what());
        return RAD_E_FORMAT;
    }
    if (!j.is_object()) { RAD_ERR("lora: --lora-map %s is not a JSON object", file.c_str());
                          return RAD_E_FORMAT; }
    if (j.contains("strip")) {
        if (!j["strip"].is_array()) { RAD_ERR("lora: --lora-map 'strip' is not a list");
                                      return RAD_E_FORMAT; }
        for (const auto& s : j["strip"]) {
            if (!s.is_string()) { RAD_ERR("lora: --lora-map 'strip' holds a non-string");
                                  return RAD_E_FORMAT; }
            strip->push_back(s.get<std::string>());
        }
    }
    if (!j.contains("modules") || !j["modules"].is_object()) {
        RAD_ERR("lora: --lora-map %s has no 'modules' object", file.c_str());
        return RAD_E_FORMAT;
    }
    for (const auto& [pat, t] : j["modules"].items()) {
        if (!t.is_array() || t.empty()) {
            RAD_ERR("lora: --lora-map rule '%s' is not a non-empty list", pat.c_str());
            return RAD_E_FORMAT;
        }
        const size_t star = pat.find('*');
        LoraRule r;
        r.star = star != std::string::npos;
        r.pre  = r.star ? pat.substr(0, star) : pat;
        r.suf  = r.star ? pat.substr(star + 1) : "";
        for (const auto& x : t) {
            if (!x.is_string()) {
                RAD_ERR("lora: --lora-map rule '%s' holds a non-string target", pat.c_str());
                return RAD_E_FORMAT;
            }
            r.targets.push_back(x.get<std::string>());
        }
        rules->push_back(std::move(r));
    }
    return RAD_OK;
}

int Checkpoint::add_lora(const std::string& path, double strength,
                         std::vector<std::string>* merged, const std::string* map_json) {
    if (kind_ != SafeT) {
        RAD_ERR("lora: an adapter merges into a safetensors source; %s is not one", path_.c_str());
        return RAD_E_UNSUPPORTED;
    }
    if (import_ != Import::None) {
        RAD_ERR("lora: this checkpoint ships its weights packed by a quantiser; an adapter merges "
                "into the plain tensors its training saw, not into the codes");
        return RAD_E_UNSUPPORTED;
    }
    std::vector<std::string> files;
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) {
        for (const auto& e : std::filesystem::directory_iterator(path, ec))
            if (e.path().extension() == ".safetensors") files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
    } else
        files.push_back(path);
    if (files.empty()) {
        RAD_ERR("lora: %s holds no .safetensors file", path.c_str());
        return RAD_E_NOTFOUND;
    }

    /* The module lookup strips wrapper prefixes: the map's first, then the builtin ones. */
    std::vector<std::string> strips;
    std::vector<LoraRule>    rules;
    if (map_json && lora_map_read(*map_json, &strips, &rules) < 0) return RAD_E_FORMAT;
    for (const char* pre : kLoraPrefixes) strips.push_back(pre);

    /* One pair's A/B/alpha, by module; `module` keeps the wrapper prefix, so the lookup below
     * strips it. */
    struct Pair { std::string module; const StTensor *a = nullptr, *b = nullptr, *al = nullptr; };

    int n_merged = 0;
    for (const std::string& f : files) {
        SafeTensorsFile st;
        if (st.open(f) != RAD_OK) return RAD_E_FORMAT;
        std::vector<Pair> pairs;
        std::map<std::string, size_t> by_module;
        for (const StTensor& t : st.tensors()) {
            if (t.name.size() > 12 && t.name.compare(t.name.size() - 12, 12, ".dora.scalar") == 0) {
                RAD_ERR("lora: '%s' is a DoRA adapter; only plain LoRA merges (a magnitude vector "
                        "has no place in a weight merge)", t.name.c_str());
                return RAD_E_UNSUPPORTED;
            }
            const char* suf = nullptr;
            if (t.name.size() > 14 && t.name.compare(t.name.size() - 14, 14, ".lora_A.weight") == 0)
                suf = ".lora_A.weight";
            else if (t.name.size() > 18 &&
                     t.name.compare(t.name.size() - 18, 18, ".lora_down.weight") == 0)
                suf = ".lora_down.weight";
            if (suf) {
                Pair p;
                p.module = t.name.substr(0, t.name.size() - std::strlen(suf));
                p.a      = &t;
                by_module[p.module] = pairs.size();
                pairs.push_back(std::move(p));
            } else if (t.name.size() > 14 &&
                       t.name.compare(t.name.size() - 14, 14, ".lora_B.weight") == 0) {
                auto it = by_module.find(t.name.substr(0, t.name.size() - 14));
                if (it != by_module.end()) pairs[it->second].b = &t;
            } else if (t.name.size() > 6 &&
                       t.name.compare(t.name.size() - 6, 6, ".alpha") == 0) {
                auto it = by_module.find(t.name.substr(0, t.name.size() - 6));
                if (it != by_module.end()) pairs[it->second].al = &t;
            }
        }
        if (pairs.empty()) {
            RAD_ERR("lora: %s holds no .lora_A.weight / .lora_down.weight tensors", f.c_str());
            return RAD_E_NOTFOUND;
        }
        for (Pair& p : pairs) {
            if (!p.b) {
                RAD_ERR("lora: '%s' ships its A but no B", p.module.c_str());
                return RAD_E_FORMAT;
            }
            /* The targets: `<module>.weight`, else the same with a wrapper prefix stripped, else
             * the tensors the first map rule the stripped module matches names. */
            std::string norm = p.module;
            for (const std::string& s : strips)
                if (norm.size() > s.size() && norm.compare(0, s.size(), s) == 0) {
                    norm.erase(0, s.size());
                    break;
                }
            std::vector<const SrcTensor*> tg;
            if (const SrcTensor* W = find(p.module + ".weight")) tg.push_back(W);
            if (tg.empty() && norm != p.module)
                if (const SrcTensor* W = find(norm + ".weight")) tg.push_back(W);
            for (const LoraRule& ru : rules) {
                std::string tok;
                if (!ru.match(norm, &tok)) continue;
                for (const std::string& pat : ru.targets) {
                    std::string nm = pat;
                    const size_t star = nm.find('*');
                    if (star != std::string::npos) nm.replace(star, 1, tok);
                    const SrcTensor* W = find(nm);
                    if (!W) {
                        RAD_ERR("lora: the map maps '%s' onto '%s', which this checkpoint does "
                                "not ship", p.module.c_str(), nm.c_str());
                        return RAD_E_NOTFOUND;
                    }
                    tg.push_back(W);
                }
                break;
            }
            if (tg.empty()) {
                RAD_ERR("lora: '%s' names a module this checkpoint does not ship as "
                        "'<module>.weight' (nor stripped of a known wrapper prefix)%s",
                        p.module.c_str(),
                        map_json ? " and no --lora-map rule matches it" : "");
                return RAD_E_NOTFOUND;
            }
            const StTensor& A = *p.a;
            const StTensor& B = *p.b;
            if (A.shape.size() != 2 || B.shape.size() != 2 || A.shape[0] <= 0 ||
                A.shape[0] != B.shape[1]) {
                RAD_ERR("lora: '%s' is not a rank-%lld pair of matrices: [a %lld x %lld, "
                        "b %lld x %lld]",
                        p.module.c_str(), (long long)A.shape[0], (long long)A.shape[0],
                        (long long)A.shape[1], (long long)B.shape[0], (long long)B.shape[1]);
                return RAD_E_FORMAT;
            }
            const auto plain = [](uint32_t d) {
                return d == RAD_BF16 || d == RAD_F16 || d == RAD_F32;
            };
            if (!plain(A.dtype) || !plain(B.dtype)) {
                RAD_ERR("lora: '%s' ships A/B in dtypes the merge cannot read", p.module.c_str());
                return RAD_E_UNSUPPORTED;
            }
            const int64_t K = A.shape[1], r = A.shape[0];
            int64_t rows = 0;
            for (const SrcTensor* W : tg) {
                if (W->gptq_bits || W->ct_group || W->rad_dtype == RAD_DT_INVALID ||
                    !plain(W->rad_dtype)) {
                    RAD_ERR("lora: '%s' is not a plain bf16/f16/f32 matrix, so an adapter cannot "
                            "merge into it", W->name.c_str());
                    return RAD_E_UNSUPPORTED;
                }
                if (W->shape.size() != 2 || W->shape[1] != K) {
                    RAD_ERR("lora: '%s' [a %lld x %lld, b %lld x %lld] does not fit '%s' "
                            "[%lld x %lld]",
                            p.module.c_str(), (long long)A.shape[0], (long long)A.shape[1],
                            (long long)B.shape[0], (long long)B.shape[1], W->name.c_str(),
                            (long long)W->shape[0], (long long)W->shape[1]);
                    return RAD_E_FORMAT;
                }
                rows += W->shape[0];
            }
            if (rows != B.shape[0]) {
                RAD_ERR("lora: '%s' merges into %zu target(s) whose rows sum to %lld, but its B "
                        "has %lld", p.module.c_str(), tg.size(), (long long)rows,
                        (long long)B.shape[0]);
                return RAD_E_FORMAT;
            }
            float scale = (float)strength;
            if (p.al) {
                const int64_t n_el = p.al->shape.empty() ? 1 : p.al->numel();
                if (n_el != 1 || (p.al->dtype != RAD_F32 && p.al->dtype != RAD_BF16 &&
                                  p.al->dtype != RAD_F16)) {
                    RAD_ERR("lora: '%s.alpha' is not a single number", p.module.c_str());
                    return RAD_E_FORMAT;
                }
                scale *= rad_load_f32(p.al->data, p.al->dtype, 0) / (float)r;
            }
            /* The module is the targets' concatenation: target i takes B's rows [off, off + N_i).
             * A row at a time, rank-1 updates into an f32 row, with A taken to f32 once a pair so
             * the sweep reads plain f32 (a dtype switch an element is hours on a large adapter). */
            std::vector<float> Af((size_t)(r * K));
            for (int64_t i = 0; i < r * K; ++i) Af[(size_t)i] = rad_load_f32(A.data, A.dtype, i);
            std::vector<float> acc((size_t)K);
            int64_t off = 0;
            for (const SrcTensor* W : tg) {
                const int64_t N = W->shape[0];
                auto out = std::make_unique<uint8_t[]>((size_t)W->bytes);
                for (int64_t n = 0; n < N; ++n) {
                    for (int64_t j = 0; j < r; ++j) {
                        const float b = scale * rad_load_f32(B.data, B.dtype, (off + n) * r + j);
                        if (b == 0.f) continue;
                        const float* ar = Af.data() + (size_t)j * K;
                        for (int64_t k = 0; k < K; ++k) acc[(size_t)k] += b * ar[k];
                    }
                    for (int64_t k = 0; k < K; ++k)
                        rad_store_f32(out.get(), W->rad_dtype, n * K + k,
                                      rad_load_f32(W->data, W->rad_dtype, n * K + k) +
                                          acc[(size_t)k]);
                    std::fill(acc.begin(), acc.end(), 0.f);
                }
                SrcTensor& t = tensors_[index_[W->name]];
                t.data = out.get();
                owned_.push_back(std::move(out));
                if (merged) merged->push_back(t.name);
                ++n_merged;
                off += N;
            }
        }
    }
    RAD_INFO("lora     %s: %d tensor(s) merged at strength %g", path.c_str(), n_merged, strength);
    return RAD_OK;
}

int Checkpoint::add_file(const std::string& path) {
    if (kind_ != SafeT) {
        RAD_ERR("%s is merged into a safetensors checkpoint's namespace; %s is not one",
                path.c_str(), path_.c_str());
        return RAD_E_UNSUPPORTED;
    }
    RAD_TRY(add_shard(path));
    meta_.finish();
    return RAD_OK;
}

const SrcTensor* Checkpoint::find(const std::string& n) const {
    auto it = index_.find(n);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

/* One expert's slice of a stacked tensor, as a view: slicing dim 0 of a row-major tensor is an
 * offset and a shorter shape, with nothing copied. It is exact for block-quantised sources too:
 * ggml packs blocks along the row, so a dim-0 slice is whole blocks and `bytes / shape[0]` is the
 * exact byte count. Slicing other dimensions is not offered. */
const SrcTensor* Checkpoint::slice(const SrcTensor& st, int index, const char* declared) {
    if (st.shape.size() < 2) {
        RAD_ERR("'%s' asks for sub-tensor %d of '%s', which has rank %zu. Only a STACKED tensor "
                "-- rank 2 or more, with the stack on dim 0 -- has sub-tensors to take.",
                declared, index, st.name.c_str(), st.shape.size());
        return nullptr;
    }
    const int64_t n = st.shape[0];
    if (index < 0 || index >= n) {
        RAD_ERR("'%s' asks for sub-tensor %d of '%s', which holds %lld of them",
                declared, index, st.name.c_str(), (long long)n);
        return nullptr;
    }
    if (!st.data || st.bytes <= 0 || st.bytes % n != 0) {
        RAD_ERR("'%s': '%s' is %lld byte(s) over %lld sub-tensor(s), which does not divide -- the "
                "slice would straddle two of them",
                declared, st.name.c_str(), (long long)st.bytes, (long long)n);
        return nullptr;
    }
    SrcTensor v;
    /* The name carries the index so diagnostics distinguish the slices. */
    v.name      = st.name + "[" + std::to_string(index) + "]";
    v.ggml      = st.ggml;
    v.rad_dtype = st.rad_dtype;
    v.shape.assign(st.shape.begin() + 1, st.shape.end());
    v.bytes     = st.bytes / n;
    v.data      = (const uint8_t*)st.data + (size_t)index * (size_t)v.bytes;
    sliced_.push_back(std::move(v));
    return &sliced_.back();
}

const SrcTensor* Checkpoint::rows(const SrcTensor& st, int64_t r0, int64_t n, const char* declared) {
    if (st.shape.empty() || st.rad_dtype == RAD_DT_INVALID || st.gptq_bits || st.ct_group ||
        st.pbits) {
        RAD_ERR("'%s' asks for rows of '%s', which is not a plainly stored tensor of rows",
                declared, st.name.c_str());
        return nullptr;
    }
    const int64_t total = st.shape[0];
    if (r0 < 0 || n <= 0 || r0 + n > total || !st.data || st.bytes % total != 0) {
        RAD_ERR("'%s' asks for rows [%lld, %lld) of '%s', which holds %lld", declared,
                (long long)r0, (long long)(r0 + n), st.name.c_str(), (long long)total);
        return nullptr;
    }
    const int64_t rb = st.bytes / total;
    SrcTensor v;
    v.name      = st.name + "[" + std::to_string(r0) + ":" + std::to_string(r0 + n) + "]";
    v.ggml      = st.ggml;
    v.rad_dtype = st.rad_dtype;
    v.shape     = st.shape;
    v.shape[0]  = n;
    v.bytes     = rb * n;
    v.data      = (const uint8_t*)st.data + (size_t)(r0 * rb);
    sliced_.push_back(std::move(v));
    return &sliced_.back();
}

/* ================================================================== logical weights */

namespace {

}  /* namespace */

std::string ckpt_tensor_of(const SrcTensor& t) {
    const size_t br = t.name.size() && t.name.back() == ']' ? t.name.rfind('[') : std::string::npos;
    return br == std::string::npos ? t.name : t.name.substr(0, br);
}

bool ckpt_reads_any(const CkptWeight& w, const std::unordered_set<std::string>& names,
                    std::unordered_set<std::string>* hit) {
    bool any = false;
    for (const auto* list : { &w.pieces, &w.scales, &w.pq_pairs, &w.pq_theta, &w.pq_scale,
                              &w.pq_meta })
        for (const SrcTensor* t : *list) {
            if (!t) continue;
            const std::string n = ckpt_tensor_of(*t);
            if (names.count(n)) {
                any = true;
                if (hit) hit->insert(n);
            }
        }
    return any;
}

namespace {

/* ParoQuant's rotation group: pairs address columns within a group of 128. */
constexpr int64_t kParoGroup = 128;

/* "model.layers.0.mlp.down_proj.weight" -> "model.layers.0.mlp.down_proj" */
std::string paro_base(const SrcTensor& piece) {
    const std::string n = ckpt_tensor_of(piece);
    std::string base;
    return weight_base(n, &base) ? base : n;
}

}  /* namespace */

/* Stages applied in table order (stage 0 first) as a' = a cos t + b sin t, b' = b cos t - a sin t
 * (libref's loop), in f64 throughout. The error (~1e-15) is far below an f16 step, so the output is
 * the exact composition rounded to f16 via f32 regardless of libm or summation order. */
void paro_compose_dense(int64_t S, const int32_t* ia, const int32_t* ib, const float* th,
                        uint16_t* out) {
    constexpr int64_t G = kParoGroup, H = kParoGroup / 2;
    std::vector<double> cs((size_t)(S * H)), sn((size_t)(S * H)), v((size_t)G);
    for (int64_t o = 0; o < S * H; ++o) {
        cs[(size_t)o] = std::cos((double)th[o]);
        sn[(size_t)o] = std::sin((double)th[o]);
    }
    for (int64_t i = 0; i < G; ++i) {
        std::fill(v.begin(), v.end(), 0.0);
        v[(size_t)i] = 1.0;
        for (int64_t o = 0; o < S * H; ++o) {
            const double va = v[(size_t)ia[o]], vb = v[(size_t)ib[o]];
            v[(size_t)ia[o]] = va * cs[(size_t)o] + vb * sn[(size_t)o];
            v[(size_t)ib[o]] = vb * cs[(size_t)o] - va * sn[(size_t)o];
        }
        for (int64_t j = 0; j < G; ++j) out[j * G + i] = rad_f32_to_f16((float)v[(size_t)j]);
    }
}

namespace {

/* Each piece of a rotated weight has its own rotation: a fused gate_up or q|k|v is several
 * projections with their own pairs, angles and channel scales, so the fused weight has one row
 * part per piece (abi/rad_encoding.h, "givens"). Adds the tables (t.pairs and t.angle stage-major
 * per part, t.scale one row per part, t.part each part's end row, t.dense) and refuses a rotation
 * the kernels cannot read: a pair leaving its group of 128, or a stage that does not use every
 * column of a group exactly once. */
int paro_attach(Checkpoint& ck, CkptWeight* out, std::string* why) {
    auto refuse = [&](std::string m) {
        if (why) *why = std::move(m);
        return (int)RAD_E_FORMAT;
    };
    const int64_t K = out->cols, P = (int64_t)out->pieces.size();
    if (K % kParoGroup) return refuse("'" + out->name + "' has " + std::to_string(K) +
                                      " columns, which is not whole rotation groups of 128");
    int64_t S = 0;
    out->pq_pairs.clear(); out->pq_theta.clear(); out->pq_scale.clear();
    for (const SrcTensor* piece : out->pieces) {
        const std::string base = paro_base(*piece);
        const SrcTensor* pr = ck.find(base + ".pairs");
        const SrcTensor* th = ck.find(base + ".theta");
        const SrcTensor* cs = ck.find(base + ".channel_scales");
        if (!pr || !th || !cs)
            return refuse("'" + piece->name + "' is a ParoQuant weight and '" + base +
                          "' lacks its pairs, theta or channel_scales");
        if (pr->rad_dtype != RAD_I16 || th->rad_dtype != RAD_F16 || cs->rad_dtype != RAD_F16 ||
            pr->shape.size() != 2 || th->shape.size() != 2 || cs->shape.size() != 2 ||
            pr->shape[1] != K || th->shape[0] != pr->shape[0] || th->shape[1] != K / 2 ||
            cs->shape[0] != 1 || cs->shape[1] != K || pr->shape[0] < 1)
            return refuse("'" + base + "' pairs/theta/channel_scales do not fit a " +
                          std::to_string(K) + "-column weight (int16 [S, K], f16 [S, K/2], f16 [1, K])");
        if (S == 0) S = pr->shape[0];
        if (pr->shape[0] != S)
            return refuse("the pieces of '" + out->name + "' have different numbers of rotation stages");
        /* Every stage must be a perfect matching of each group's 128 columns. */
        const int16_t* pv = (const int16_t*)pr->data;
        for (int64_t st = 0; st < S; ++st)
            for (int64_t g = 0; g < K / kParoGroup; ++g) {
                bool seen[kParoGroup] = {};
                for (int64_t i = 0; i < kParoGroup; ++i) {
                    const int v = pv[st * K + g * kParoGroup + i];
                    if (v < 0 || v >= kParoGroup || seen[v])
                        return refuse("'" + pr->name + "' stage " + std::to_string(st) + " group " +
                                      std::to_string(g) + " is not a pairing of its 128 columns");
                    seen[v] = true;
                }
            }
        out->pq_pairs.push_back(pr);
        out->pq_theta.push_back(th);
        out->pq_scale.push_back(cs);
    }
    RadEncoding& e = out->enc;
    rad_enc_copy_str(e.transform, "givens");
    rad_enc_add_table(&e, "t.pairs", RAD_I32, P * S, K);
    rad_enc_add_table(&e, "t.angle", RAD_F32, P * S, K / 2);
    rad_enc_add_table(&e, "t.scale", RAD_F32, P, K);
    if (P > 1) rad_enc_add_table(&e, "t.part", RAD_I32, 1, P);
    /* The dense matrices are composed and stored here, since the prefill quantiser
     * (givens_dense_quant) applies a (part, group) as one 128 x 128 matrix and a kernel library
     * must not compute weight values at load (abi/rad_abi.h, weight layout). The pairs and angles
     * stay for the 16-bit GEMMs' stage-wise rotation. */
    if (rad_enc_add_table(&e, "t.dense", RAD_F16, P * K / kParoGroup, kParoGroup * kParoGroup) < 0)
        return refuse("'" + out->name + "' has more planes than an encoding holds");
    if (!rad_enc_valid(&e)) return refuse("the rotation tables of '" + out->name + "' are malformed");
    out->imported = true;
    out->paro = true;
    out->trivial = false;
    out->why = "it is ParoQuant MXFP4, which rad-convert imports with its rotation tables -- "
               "serve the container it writes";
    return RAD_OK;
}

}  /* namespace */

namespace {
int piece_f32(const SrcTensor& t, int64_t e, int64_t n, float* out);
}  /* namespace */

/* RAD_MAP_WNORM: src[0] the magnitude g ([d0, 1, ...]), src[1] the direction v ([d0, ...]). The
 * weight is v's shape, and each dim-0 slice's factor g_i / ||v_i|| is computed here, in double,
 * from v as the checkpoint stores it. */
static int resolve_wnorm(Checkpoint& ck, const RadNameMap& m, bool optional, CkptWeight* out,
                         std::string* why) {
    auto refuse = [&](int rc, std::string msg) {
        if (why) *why = std::move(msg);
        return rc;
    };
    if (m.n_src != 2)
        return refuse(RAD_E_INVAL, "a weight-norm map names its magnitude and its direction");
    const SrcTensor* g = ck.find(m.src[0]);
    const SrcTensor* v = ck.find(m.src[1]);
    if (!g || !v) {
        if (optional && !g && !v) return refuse(RAD_E_NOTFOUND, "none of its tensors is there");
        return refuse(RAD_E_NOTFOUND, std::string("it needs checkpoint tensors '") + m.src[0] +
                                      "' and '" + m.src[1] + "', which are not both in " + ck.path());
    }
    if (v->shape.empty() || v->shape.size() > RAD_MAX_RANK || g->shape.empty() ||
        g->shape[0] != v->shape[0])
        return refuse(RAD_E_SHAPE, "'" + g->name + "' and '" + v->name + "' are not a weight "
                                   "norm's magnitude and direction");
    int64_t gn = 1, vn = 1;
    for (int64_t x : g->shape) gn *= x;
    for (int64_t x : v->shape) vn *= x;
    const int64_t d0 = v->shape[0];
    if (gn != d0 || vn % d0)
        return refuse(RAD_E_SHAPE, "'" + g->name + "' holds " + std::to_string(gn) + " values for "
                                   "the " + std::to_string(d0) + " slices of '" + v->name + "'");
    const int64_t per = vn / d0;
    std::vector<float> vf((size_t)vn), gf((size_t)d0);
    if (piece_f32(*v, 0, vn, vf.data()) < 0 || piece_f32(*g, 0, d0, gf.data()) < 0)
        return refuse(RAD_E_FORMAT, "'" + v->name + "' or '" + g->name + "' cannot be read as f32");
    out->wnorm_scale.assign((size_t)d0, 0.0f);
    for (int64_t i = 0; i < d0; ++i) {
        double ss = 0;
        for (int64_t k = 0; k < per; ++k) {
            const double x = vf[(size_t)(i * per + k)];
            ss += x * x;
        }
        out->wnorm_scale[(size_t)i] = ss > 0 ? (float)((double)gf[(size_t)i] / std::sqrt(ss)) : 0.0f;
    }
    out->wnorm = true;
    out->pieces.assign(1, v);
    out->scales.assign(1, nullptr);
    out->concat_dim = 0;
    out->rank = (uint32_t)v->shape.size();
    for (uint32_t i = 0; i < out->rank; ++i) out->shape[i] = v->shape[i];
    rad_enc_view(out->rank, out->shape, &out->rows, &out->cols);
    out->trivial = false;
    out->why = "it is weight-normalised ('" + g->name + "' times '" + v->name + "' over its norm), "
               "folded at convert -- a recipe rule stores it";
    return RAD_OK;
}

/* RAD_MAP_TRANSPOSE (src[0] [.., a, b] read as [.., b, a]), RAD_MAP_EMA (src[0] the embedding
 * sums [bins, ...], src[1] the cluster usage [bins]; each row over max(usage, 1e-5)), RAD_MAP_TAP
 * (one index of one dimension of src[0]) and RAD_MAP_COLS (one column range of src[0]), computed in
 * f32 into `derived`. */
static int resolve_derived(Checkpoint& ck, const std::vector<const RadNameMap*>& maps,
                           bool optional, CkptWeight* out, std::string* why) {
    auto refuse = [&](int rc, std::string msg) {
        if (why) *why = std::move(msg);
        return rc;
    };
    /* Stacked EMA codebooks: each map's rows in turn, every one the first's width. */
    if (maps.size() > 1) {
        std::vector<float> all;
        int64_t rows = 0, per = -1;
        std::string names;
        for (const RadNameMap* pm : maps) {
            CkptWeight one;
            const std::vector<const RadNameMap*> single{ pm };
            const int rc = resolve_derived(ck, single, false, &one, why);
            if (rc < 0) return rc;
            const int64_t w = (int64_t)one.derived.size() / one.shape[0];
            if (per >= 0 && w != per)
                return refuse(RAD_E_SHAPE, "the codebooks stacked as '" + out->name + "' differ "
                                           "in width");
            per = w;
            rows += one.shape[0];
            all.insert(all.end(), one.derived.begin(), one.derived.end());
            if (out->pieces.empty()) {
                out->pieces = one.pieces;
                out->rank = one.rank;
                for (uint32_t i = 0; i < one.rank; ++i) out->shape[i] = one.shape[i];
            }
            names += (names.empty() ? "'" : ", '") + std::string(pm->src[0]) + "'";
        }
        out->shape[0] = rows;
        out->derived = std::move(all);
        out->scales.assign(1, nullptr);
        out->concat_dim = 0;
        rad_enc_view(out->rank, out->shape, &out->rows, &out->cols);
        out->trivial = false;
        out->why = "it is " + std::to_string(maps.size()) + " EMA codebooks stacked (" + names +
                   "), divided at convert -- a recipe rule stores it";
        return RAD_OK;
    }
    const RadNameMap& m = *maps.front();
    const bool ema = m.mode == RAD_MAP_EMA;
    const SrcTensor* a = ck.find(m.src[0]);
    const SrcTensor* u = ema ? ck.find(m.src[1]) : nullptr;
    if (!a || (ema && !u)) {
        if (optional && !a && !u) return refuse(RAD_E_NOTFOUND, "none of its tensors is there");
        return refuse(RAD_E_NOTFOUND, std::string("it needs checkpoint tensor") +
                                      (ema ? "s '" + std::string(m.src[0]) + "' and '" +
                                                 m.src[1] + "'" : " '" + std::string(m.src[0]) +
                                                                  "'") +
                                      ", not all in " + ck.path());
    }
    /* A transposed expert table is one sub-tensor of a stack, transposed. A stack kept as a
     * rank-3 tensor takes src_index[0] as its dim-0 index. A stack flattened into the rows of a
     * rank-2 tensor takes src_rows[0] as the sub-tensor's row count, with src_index[0] selecting
     * the block: Nomic's experts.mlp.w2 is [n_expert * n_ff, n_embd]. */
    if (m.mode == RAD_MAP_TRANSPOSE && m.src_index[0] >= 0) {
        if (m.src_rows[0] > 0 && a->shape.size() == 2) {
            const int64_t r0 = (int64_t)m.src_index[0] * m.src_rows[0];
            a = ck.rows(*a, r0, m.src_rows[0], out->name.c_str());
        } else {
            a = ck.slice(*a, m.src_index[0], out->name.c_str());
        }
        if (!a) return refuse(RAD_E_SHAPE, "a sub-tensor it names is not there");
    }
    const size_t rank = a->shape.size();
    if (rank < 2 || rank > RAD_MAX_RANK)
        return refuse(RAD_E_SHAPE, "'" + a->name + "' has rank " + std::to_string(rank) +
                                   "; the map needs a matrix");
    int64_t n = 1;
    for (int64_t x : a->shape) n *= x;
    std::vector<float> src((size_t)n);
    if (piece_f32(*a, 0, n, src.data()) < 0)
        return refuse(RAD_E_FORMAT, "'" + a->name + "' cannot be read as f32");
    out->rank = (uint32_t)rank;
    for (size_t i = 0; i < rank; ++i) out->shape[i] = a->shape[i];
    if (m.mode == RAD_MAP_COLS) {
        const int64_t cb = a->shape[rank - 1], c0 = m.src_index[0], nc = m.src_rows[0];
        if (c0 < 0 || nc <= 0 || c0 + nc > cb)
            return refuse(RAD_E_SHAPE, "columns [" + std::to_string(c0) + ", " +
                                       std::to_string(c0 + nc) + ") of '" + a->name + "', which "
                                       "has " + std::to_string(cb));
        const int64_t outer = n / cb;
        out->derived.resize((size_t)(outer * nc));
        for (int64_t o = 0; o < outer; ++o)
            std::memcpy(&out->derived[(size_t)(o * nc)], &src[(size_t)(o * cb + c0)],
                        (size_t)nc * sizeof(float));
        out->shape[rank - 1] = nc;
        out->why = "it is columns [" + std::to_string(c0) + ", " + std::to_string(c0 + nc) +
                   ") of '" + a->name + "', at convert -- a recipe rule stores it";
    } else if (m.mode == RAD_MAP_TAP) {
        const int d = m.concat_dim;
        if (d < 0 || (size_t)d >= rank || rank < 3)
            return refuse(RAD_E_SHAPE, "'" + a->name + "' has rank " + std::to_string(rank) +
                                       "; a tap of dimension " + std::to_string(d) +
                                       " leaves no matrix");
        const int64_t k = a->shape[(size_t)d];
        const int64_t t = m.src_index[0] < 0 ? k + m.src_index[0] : m.src_index[0];
        if (t < 0 || t >= k)
            return refuse(RAD_E_SHAPE, "tap " + std::to_string(m.src_index[0]) + " of '" +
                                       a->name + "' dimension " + std::to_string(d) + " of " +
                                       std::to_string(k));
        int64_t outer = 1, inner = 1;
        for (int i = 0; i < d; ++i) outer *= a->shape[(size_t)i];
        for (size_t i = (size_t)d + 1; i < rank; ++i) inner *= a->shape[i];
        out->derived.resize((size_t)(outer * inner));
        for (int64_t o = 0; o < outer; ++o)
            std::memcpy(&out->derived[(size_t)(o * inner)], &src[(size_t)((o * k + t) * inner)],
                        (size_t)inner * sizeof(float));
        out->rank = (uint32_t)(rank - 1);
        for (size_t i = (size_t)d; i + 1 < rank; ++i) out->shape[i] = a->shape[i + 1];
        out->why = "it is tap " + std::to_string(t) + " of '" + a->name + "' dimension " +
                   std::to_string(d) + ", at convert -- a recipe rule stores it";
    } else if (ema) {
        const int64_t bins = a->shape[0], per = n / bins;
        int64_t un = 1;
        for (int64_t x : u->shape) un *= x;
        if (un != bins)
            return refuse(RAD_E_SHAPE, "'" + u->name + "' holds " + std::to_string(un) +
                                       " usage counts for the " + std::to_string(bins) +
                                       " rows of '" + a->name + "'");
        std::vector<float> uf((size_t)bins);
        if (piece_f32(*u, 0, bins, uf.data()) < 0)
            return refuse(RAD_E_FORMAT, "'" + u->name + "' cannot be read as f32");
        for (int64_t i = 0; i < bins; ++i) {
            const float d = std::max(uf[(size_t)i], 1e-5f);
            for (int64_t k = 0; k < per; ++k) src[(size_t)(i * per + k)] /= d;
        }
        out->derived = std::move(src);
        out->why = "it is an EMA codebook ('" + a->name + "' over '" + u->name + "'), divided at "
                   "convert -- a recipe rule stores it";
    } else {
        const int64_t ra = a->shape[rank - 2], cb = a->shape[rank - 1], per = ra * cb;
        out->shape[rank - 2] = cb;
        out->shape[rank - 1] = ra;
        out->derived.assign((size_t)n, 0.0f);
        for (int64_t s = 0; s < n / per; ++s)
            for (int64_t i = 0; i < ra; ++i)
                for (int64_t j = 0; j < cb; ++j)
                    out->derived[(size_t)(s * per + j * ra + i)] =
                        src[(size_t)(s * per + i * cb + j)];
        out->why = "it is '" + a->name + "' transposed, at convert -- a recipe rule stores it";
    }
    out->pieces.assign(1, a);
    out->scales.assign(1, nullptr);
    out->concat_dim = 0;
    rad_enc_view(out->rank, out->shape, &out->rows, &out->cols);
    out->trivial = false;
    return RAD_OK;
}

int ckpt_resolve(Checkpoint& ck, const Program& prog, const std::string& name, bool optional,
                 CkptWeight* out, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    *out = CkptWeight{};
    out->name = name;

    /* Every map for the name in declaration order. Only a CONCAT map may be declared in pieces
     * (declare refused other repeats); the pieces join end to end. */
    std::vector<const RadNameMap*> maps;
    for (const auto& nm : prog.name_map)
        if (nm.declared && name == nm.declared) maps.push_back(&nm);
    if (maps.empty()) return refuse(RAD_E_NOTFOUND, "the name map has no entry for it");
    if (maps.front()->mode == RAD_MAP_WNORM)
        return resolve_wnorm(ck, *maps.front(), optional, out, why);
    if (maps.front()->mode == RAD_MAP_TRANSPOSE || maps.front()->mode == RAD_MAP_EMA ||
        maps.front()->mode == RAD_MAP_TAP || maps.front()->mode == RAD_MAP_COLS)
        return resolve_derived(ck, maps, optional, out, why);
    struct Src1 { const char* name; int index; int64_t rows; };
    std::vector<Src1> srcs;
    const bool row_pieces = maps.front()->mode == RAD_MAP_ROWS;
    for (const RadNameMap* pm : maps)
        for (int k = 0; k < pm->n_src; ++k)
            srcs.push_back({ pm->src[k], pm->src_index[k], pm->src_rows[k] });
    out->concat_dim = row_pieces ? 0 : maps.front()->concat_dim;

    /* The mode decides what several sources mean:
     *
     *   RAD_MAP_CONCAT   all, joined (gate and up into gate_up).
     *   RAD_MAP_COPY     alternatives, first present wins (a tied lm_head is output.weight when
     *                    shipped, else the embedding). */
    const bool alternatives = maps.front()->mode == RAD_MAP_COPY && srcs.size() > 1;
    /* Pieces whose scale is one `X.weight_scale` their shards share (below). */
    bool shard_scaled = false;
    for (const Src1& s1 : srcs) {
        const SrcTensor* st = ck.find(s1.name);
        /* An FP8 weight's scale plane: `X.weight_scale_inv` (block-scaled releases) or
         * `X.weight_scale` (compressed-tensors' and Quark's naming, one per row or per block).
         * Only an FP8 tensor takes `X.weight_scale`, so a float weight beside an unrelated tensor
         * of that name reads plain. An imported weight's planes come from import_piece below. */
        const SrcTensor* sc = st ? ck.find(std::string(s1.name) + "_scale_inv") : nullptr;
        if (!sc && st && (st->rad_dtype == RAD_F8E4M3 || st->rad_dtype == RAD_F8E5M2))
            sc = ck.find(std::string(s1.name) + "_scale");
        /* A table split into shards, `X.shard_<i>.weight`, under one scale for all of them,
         * `X.weight_scale` (Qwen3.8-Flash-Next-FP8's n-gram table). */
        if (!sc && st && (st->rad_dtype == RAD_F8E4M3 || st->rad_dtype == RAD_F8E5M2)) {
            const std::string n = s1.name;
            const size_t at = n.rfind(".shard_");
            const size_t end = n.size() >= 7 ? n.size() - 7 : 0;
            if (at != std::string::npos && end > at + 7 && n.compare(end, 7, ".weight") == 0 &&
                std::all_of(n.begin() + (long)at + 7, n.begin() + (long)end,
                            [](char c) { return c >= '0' && c <= '9'; }))
                if ((sc = ck.find(n.substr(0, at) + ".weight_scale"))) shard_scaled = true;
        }
        /* A weight shipped in an imported form is read through that form's views. */
        {
            const SrcTensor *ic = nullptr, *is = nullptr;
            std::string iw;
            const int ir = ck.import_piece(s1.name, &ic, &is, &iw);
            if (ir == RAD_OK) {
                if (s1.index >= 0)
                    return refuse(RAD_E_UNSUPPORTED, "it is a slice of '" + std::string(s1.name) +
                                                     "', which the checkpoint ships quantised; an "
                                                     "imported tensor is read whole");
                st = ic;
                sc = is;
            } else if (ir != RAD_E_NOTFOUND) {
                return refuse(ir, iw);
            }
        }
        /* A row piece (RAD_MAP_ROWS) of a plain tensor; a scale plane has no rows to match. */
        if (st && row_pieces) {
            if (sc)
                return refuse(RAD_E_UNSUPPORTED, "it takes rows of '" + std::string(s1.name) +
                                                 "', which the checkpoint ships block-scaled");
            st = ck.rows(*st, s1.index, s1.rows, name.c_str());
            if (!st) return refuse(RAD_E_SHAPE, "the rows it names are not there");
        }
        /* A sub-tensor index is always applied (ignoring it would give every expert the whole
         * stack). Its scale plane is sliced at the same index. A stack flattened into the rows of
         * a rank-2 tensor takes src_rows as the sub-tensor's row count. */
        else if (st && s1.index >= 0) {
            if (s1.rows > 0 && st->shape.size() == 2)
                st = ck.rows(*st, (int64_t)s1.index * s1.rows, s1.rows, name.c_str());
            else
                st = ck.slice(*st, s1.index, name.c_str());
            if (!st) return refuse(RAD_E_SHAPE, "a sub-tensor it names is not there");
            if (sc && !(sc = ck.slice(*sc, s1.index, name.c_str())))
                return refuse(RAD_E_SHAPE, "its scale plane cannot be sliced at the same index");
        }
        if (!st) {
            if (alternatives) continue;
            if (optional) { out->pieces.clear(); break; }
            return refuse(RAD_E_NOTFOUND, std::string("it needs checkpoint tensor '") + s1.name +
                                          "', which is not in " + ck.path());
        }
        out->pieces.push_back(st);
        out->scales.push_back(sc);
        if (alternatives) break;
    }
    if (out->pieces.empty()) {
        if (optional) return refuse(RAD_E_NOTFOUND, "none of its tensors is in the checkpoint");
        std::string tried;
        for (size_t k = 0; k < srcs.size(); ++k) tried += (k ? " or " : "") + std::string(srcs[k].name);
        return refuse(RAD_E_NOTFOUND, "it needs one of " + tried + ", and " + ck.path() +
                                      " holds none of them");
    }

    /* The logical shape: pieces joined along concat_dim, other extents equal. A rank-0 tensor
     * (safetensors' scalar, e.g. Gemma 4's clipped-linear bounds) reads as shape [1]. */
    const SrcTensor& p0 = *out->pieces[0];
    const int d = out->concat_dim;
    auto shape_of = [](const SrcTensor& t) {
        return t.shape.empty() ? std::vector<int64_t>{ 1 } : t.shape;
    };
    const std::vector<int64_t> s0 = shape_of(p0);
    if (s0.size() > RAD_MAX_RANK || d < 0 || d >= (int)s0.size())
        return refuse(RAD_E_SHAPE, "'" + p0.name + "' cannot be joined along dim " +
                                   std::to_string(d));
    out->rank = (uint32_t)s0.size();
    for (uint32_t i = 0; i < out->rank; ++i) out->shape[i] = s0[i];
    for (size_t k = 1; k < out->pieces.size(); ++k) {
        const SrcTensor& pk = *out->pieces[k];
        const std::vector<int64_t> sk = shape_of(pk);
        bool fits = sk.size() == s0.size();
        for (uint32_t i = 0; fits && i < out->rank; ++i)
            if ((int)i != d && sk[i] != s0[i]) fits = false;
        if (!fits)
            return refuse(RAD_E_SHAPE, "'" + pk.name + "' and '" + p0.name +
                                       "' do not join along dim " + std::to_string(d));
        out->shape[d] += sk[(size_t)d];
    }
    rad_enc_view(out->rank, out->shape, &out->rows, &out->cols);

    /* The pieces' encoding, when trivial (spec §4.3), or imported. */
    const uint32_t dt = p0.rad_dtype;
    bool same = true, scaled = true, unscaled = true;
    for (size_t k = 0; k < out->pieces.size(); ++k) {
        if (out->pieces[k]->rad_dtype != dt) same = false;
        if (out->scales[k]) unscaled = false; else scaled = false;
    }
    if (dt == RAD_DT_INVALID) {
        out->why = std::string("'") + p0.name + "' is a GGUF " +
                   (gguf_dtype(p0.ggml) ? gguf_dtype(p0.ggml)->name : "block") + " tensor";
        /* Q8_0, Q4_K and Q6_K become their planes when every piece is of that type and they join
         * along rows, so each row of the weight is one source row of blocks. rad-convert imports
         * them; the loader refuses them, since the blocks interleave what the planes hold apart. */
        bool one_type = true;
        for (const SrcTensor* pk : out->pieces) one_type = one_type && pk->ggml == p0.ggml;
        RadEncoding enc;
        if (!one_type) {
            out->why = "its GGUF pieces are of different block types";
        } else if (ggml_plane_encoding(p0.ggml, &enc)) {
            if (out->pieces.size() > 1 && (d != 0 || out->rank < 2)) {
                out->why = "its GGUF pieces are joined along a column, across their blocks";
            } else {
                out->imported = true;
                out->ggml_blocks = true;
                out->enc = enc;
                out->why = std::string("it is a GGUF ") + gguf_dtype(p0.ggml)->name +
                           " tensor, which rad-convert imports as its planes -- serve the "
                           "container it writes";
            }
        }
    } else if (!same) {
        out->why = "its pieces are stored in different dtypes";
    } else if (p0.pbits) {
        /* Paro int5-bitplane: u5 codes, an f16 scale and u8 zero point per 128 of K, then the
         * rotation tables. Joined only along rows (gate|up), as every imported form. */
        bool fits = d == 0 || out->pieces.size() == 1;
        for (const SrcTensor* pk : out->pieces) fits = fits && pk->pbits == p0.pbits;
        if (!fits) {
            out->why = "its int5 pieces are joined along a column, across their groups";
        } else {
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", p0.pbits == 5 ? RAD_U5 : RAD_U4, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", RAD_F16, 1, 128);
            rad_enc_add_plane(&out->enc, "zero", RAD_U8, 1, 128);
            const int rc = paro_attach(ck, out, why);
            if (rc != RAD_OK) return rc;
            out->why = std::string("it is ParoQuant int") + (p0.pbits == 5 ? "5" : "4") +
                       ", which rad-convert imports with its zero points and rotation tables -- "
                       "serve the container it writes";
        }
    } else if (p0.gptq_bits) {
        /* GPTQ-packed: one width and group throughout, joined only along rows (a column join
         * would put two pieces' groups in one scale block). */
        out->imported = true;
        for (size_t k = 1; k < out->pieces.size() && out->imported; ++k)
            if (out->pieces[k]->gptq_bits != p0.gptq_bits ||
                out->pieces[k]->gptq_group != p0.gptq_group) {
                out->imported = false;
                out->why = "its GPTQ-packed pieces differ in width or group";
            }
        if (out->imported && d != 0 && out->pieces.size() > 1) {
            out->imported = false;
            out->why = "its GPTQ-packed pieces are joined along a column, across their groups";
        }
        if (out->imported) {
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", dt, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", RAD_BF16, 1, 128);
            out->why = "it is GPTQ-packed int" + std::to_string(p0.gptq_bits) + ", which "
                       "rad-convert imports with its scales rounded to bf16 -- serve the "
                       "container it writes";
        }
    } else if (dt == RAD_FP4E2M1 && scaled &&
               (ck.imports_at(p0.name) == Checkpoint::Import::Nvfp4 ||
                ck.imports_at(p0.name) == Checkpoint::Import::ModelOpt)) {
        /* NVFP4 as import_piece reads it: e2m1 codes beside the f32 scale the e4m3 one and the
         * second level make, one per 16 of K, each piece with its own second level. Not trivial
         * (the scale is derived), so the loader refuses it and rad-convert decodes it for a
         * recipe. Joined only along rows: the planes and the f32 read follow the pieces row by
         * row. */
        bool f32 = true;
        for (const SrcTensor* s : out->scales) f32 = f32 && s->rad_dtype == RAD_F32;
        if (!f32) {
            out->why = "its FP4 pieces are not all NVFP4";
        } else if (d != 0 && out->pieces.size() > 1) {
            out->why = "its NVFP4 pieces are joined along a column, across their global scales";
        } else {
            out->imported = true;
            out->why = "it is NVFP4 (e2m1 codes, an e4m3 scale per 16 of K and an f32 global "
                       "scale per tensor), which no kernel serves; rad-convert decodes it for a "
                       "recipe rule";
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", RAD_FP4E2M1, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", RAD_F32, 1, 16);
        }
    } else if (dt == RAD_FP4E2M1 && scaled && ck.imports() == Checkpoint::Import::ParoM2) {
        /* ParoQuant's M2XFP4: the MXFP4 planes byte for byte -- e2m1 codes, an e8m0 exponent per
         * 32 of K -- plus the scheme's metadata byte a group (one per 32, four 2-bit subgroup
         * multipliers), in the rotated basis the checkpoint trained, with the rotation beside them
         * (paro_attach). Joined only along rows: every plane follows the pieces row by row. */
        bool fits = d == 0 || out->pieces.size() == 1;
        for (size_t k = 0; k < out->pieces.size() && fits; ++k) {
            const SrcTensor* mt = ck.import_meta(out->pieces[k]->name);
            fits = mt && out->scales[k] && out->scales[k]->rad_dtype == RAD_E8M0;
            if (fits) out->pq_meta.push_back(mt);
        }
        if (!fits) {
            out->pq_meta.clear();
            out->why = "its M2XFP4 pieces are joined along a column, or a piece has no metadata "
                       "plane beside its exponents";
        } else {
            out->imported = true;
            out->trivial = false;
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "m2xfp4");
            rad_enc_add_plane(&out->enc, "codes", RAD_FP4E2M1, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", RAD_E8M0, 1, 32);
            rad_enc_add_plane(&out->enc, "meta", RAD_U8, 1, 32);
            const int rc = paro_attach(ck, out, why);
            if (rc != RAD_OK) return rc;
            out->why = "it is ParoQuant M2XFP4, which rad-convert imports with its rotation tables "
                       "-- serve the container it writes";
        }
    } else if (dt == RAD_FP4E2M1 && scaled) {
        /* Quark's MXFP4 byte for byte: codes and exponents are the planes of
         * `fp4_e2m1*e8m0[1x32]`. Joined only along rows (gate and up). */
        out->trivial = d == 0 || out->pieces.size() == 1;
        if (!out->trivial) out->why = "its MXFP4 pieces are joined along a column";
        if (out->trivial) {
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", RAD_FP4E2M1, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", RAD_E8M0, 1, 32);
            if (ck.imports() == Checkpoint::Import::ParoMx) {
                const int rc = paro_attach(ck, out, why);
                if (rc != RAD_OK) return rc;
            }
        }
    } else if ((dt == RAD_F8E4M3 || dt == RAD_F8E5M2) && scaled) {
        /* FP8 codes with a scale plane of one dtype, in one of three shapes per piece:
         *   block       [ceil(N/br), ceil(K/bc)], the checkpoint's block (128x128 unless stated):
         *               `fp8_e4m3*<s>[brxbc]`
         *   per row     [N] or [N, 1] (compressed-tensors' channel strategy, ParoQuant's heads):
         *               `fp8_e4m3*<s>[1x*]`
         *   per tensor  one element (ModelOpt's FP8): read as per row, the scale once a row, so
         *               pieces with their own scales join; not byte for byte, so imported
         * The first two are byte for byte when joined along rows, and a block join only where each
         * piece but the last fills its last row block. A row join that ends inside a block is
         * still read piece by piece, so it keeps its encoding without being trivial. Mixed shapes
         * and column joins state no encoding: ckpt_rows_f32 reads scales along rows only, so it
         * refuses them with `why`. */
        const int64_t br = ck.fp8_block_rows(), bc = ck.fp8_block_cols();
        const uint32_t sdt = out->scales[0]->rad_dtype;
        int n_row = 0, n_blk = 0, n_one = 0;
        bool bad = false;
        for (size_t k = 0; k < out->pieces.size() && !bad; ++k) {
            const SrcTensor& pk = *out->pieces[k];
            const SrcTensor& sk = *out->scales[k];
            int64_t r = 0, c = 0;
            rad_enc_view((uint32_t)pk.shape.size(), pk.shape.data(), &r, &c);
            const int64_t want_r = (r + br - 1) / br, want_c = (c + bc - 1) / bc;
            int64_t sr = 0, scl = 0;
            rad_enc_view((uint32_t)sk.shape.size(), sk.shape.data(), &sr, &scl);
            /* A rank-1 plane is per row; a wider one is a block plane first, then [N, 1]. One
             * element for a piece of several rows is the whole tensor's. */
            const bool one = per_tensor_scale(sk, r);
            const bool row1 = sk.shape.size() == 1 && sk.shape[0] == r;
            const bool blk = sk.shape.size() >= 2 && sr == want_r && scl == want_c;
            const bool row2 = sk.shape.size() >= 2 && sr == r && scl == 1;
            if (sk.rad_dtype != sdt) {
                bad = true;
                out->why = "its FP8 pieces' scale planes differ in dtype";
            } else if (one) {
                ++n_row;
                ++n_one;
            } else if (row1) {
                ++n_row;
            } else if (blk) {
                ++n_blk;
                if (out->why.empty() && k + 1 < out->pieces.size() && r % br)
                    out->why = "'" + pk.name + "' ends inside a scale block, so the next piece's "
                               "scales cannot follow it";
            } else if (row2) {
                ++n_row;
            } else {
                bad = true;
                out->why = "'" + sk.name + "' is neither the [" + std::to_string(want_r) + ", " +
                           std::to_string(want_c) + "] " + std::to_string(br) + "x" +
                           std::to_string(bc) + " scale plane of '" + pk.name + "' nor one "
                           "scale per row";
            }
        }
        if (!bad && n_row && n_blk) {
            bad = true;
            out->why = "its FP8 pieces disagree: some scale per row, some per " +
                       std::to_string(br) + "x" + std::to_string(bc) + " block";
        }
        if (!bad && d != 0 && out->pieces.size() > 1) {
            bad = true;
            out->why = "its FP8 pieces are joined along a column, across their scale planes";
        }
        /* Shards under the one scale they share are one tensor with one scale, byte for byte:
         * `fp8_e4m3*<s>[*x*]`. */
        bool one_scale = !bad && shard_scaled && n_one == (int)out->pieces.size();
        for (size_t k = 1; one_scale && k < out->scales.size(); ++k)
            one_scale = out->scales[k] == out->scales[0];
        if (one_scale) {
            out->trivial = out->why.empty();
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", dt, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", sdt, 0, 0);
        } else if (!bad) {
            out->trivial = out->why.empty() && n_one == 0;
            /* A per-tensor scale is written once a row (ckpt_plane_rows). */
            if (n_one) {
                out->imported = true;
                if (out->why.empty())
                    out->why = "it is FP8 with one scale per tensor, which no kernel reads as it "
                               "is; a recipe's import rule (block=128x128 scale=bf16) or a "
                               "quantiser stores it";
            }
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", dt, 1, 1);
            if (n_row) rad_enc_add_plane(&out->enc, "scale", sdt, 1, 0);
            else rad_enc_add_plane(&out->enc, "scale", sdt, br, bc);
        }
    } else if (p0.ct_group) {
        /* CT-INT4: one group throughout, joined only along rows, as GPTQ. A symmetric weight's
         * codes become signed i4 (the offset removed at relayout); an asymmetric one keeps its u4
         * codes and has a u4 zero plane, one per group of K. The scale plane is as shipped. */
        out->imported = true;
        for (size_t k = 1; k < out->pieces.size() && out->imported; ++k)
            if (out->pieces[k]->ct_group != p0.ct_group ||
                out->scales[k]->rad_dtype != out->scales[0]->rad_dtype ||
                !out->pieces[k]->ct_zero != !p0.ct_zero) {
                out->imported = false;
                out->why = "its CT-INT4 pieces differ in group, scale dtype or zero point";
            }
        if (out->imported && d != 0 && out->pieces.size() > 1) {
            out->imported = false;
            out->why = "its CT-INT4 pieces are joined along a column, across their groups";
        }
        if (out->imported) {
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", p0.ct_zero ? RAD_U4 : RAD_I4, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", out->scales[0]->rad_dtype, 1, p0.ct_group);
            if (p0.ct_zero) rad_enc_add_plane(&out->enc, "zero", RAD_U4, 1, p0.ct_group);
            out->why = "it is CT-INT4, which rad-convert imports -- serve the container it writes";
        }
    } else if (!unscaled) {
        out->why = "some of its pieces carry a scale plane and some do not";
    } else if (dt == RAD_F8E4M3 || dt == RAD_F8E5M2) {
        out->why = "it is FP8 with no scale plane beside it";
    } else {
        out->trivial = rad_dtype_bits(dt) >= 8;
        if (out->trivial) out->enc = rad_enc_plain(dt);
        else out->why = std::string("it is ") + rad_dtype_name(dt);
    }
    return RAD_OK;
}

namespace {

/* ---------------------------------------------------------------- GGUF block types
 *
 * ggml block types decodable to f32. Their geometry lives in core/text; decoding lives here, where
 * weights are read. Q4_K, Q5_K and Q6_K are the K-quants the GGUF releases ship, IQ4_NL the
 * non-linear 4-bit. */
enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7,
       GGML_Q8_0 = 8, GGML_Q4_K = 12, GGML_Q5_K = 13, GGML_Q6_K = 14, GGML_BF16 = 30,
       GGML_IQ4_NL = 20 };

/* IQ4_NL's non-linear levels (llama.cpp's kvalues_iq4nl), one per 4-bit code. */
static const int8_t kIq4NlLevels[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};


int ggml_rows_f32(uint32_t type, const uint8_t* p, int64_t n, float* dst) {
    auto half = [&](const uint8_t* q) { uint16_t h; std::memcpy(&h, q, 2); return rad_f16_to_f32(h); };
    switch (type) {
        case GGML_Q8_0:
            for (int64_t b = 0; b < n / 32; ++b, p += 34) {
                const float d = half(p);
                for (int j = 0; j < 32; ++j) dst[b * 32 + j] = d * (float)(int8_t)p[2 + j];
            }
            return RAD_OK;
        case GGML_Q4_0:
            for (int64_t b = 0; b < n / 32; ++b, p += 18) {
                const float d = half(p);
                for (int j = 0; j < 16; ++j) {
                    dst[b * 32 + j]      = d * (float)((p[2 + j] & 0x0F) - 8);
                    dst[b * 32 + j + 16] = d * (float)((p[2 + j] >> 4)   - 8);
                }
            }
            return RAD_OK;
        case GGML_Q4_1:
            for (int64_t b = 0; b < n / 32; ++b, p += 20) {
                const float d = half(p), m = half(p + 2);
                for (int j = 0; j < 16; ++j) {
                    dst[b * 32 + j]      = d * (float)(p[4 + j] & 0x0F) + m;
                    dst[b * 32 + j + 16] = d * (float)(p[4 + j] >> 4)   + m;
                }
            }
            return RAD_OK;
        case GGML_Q5_0:
            for (int64_t b = 0; b < n / 32; ++b, p += 22) {
                const float d = half(p);
                uint32_t qh; std::memcpy(&qh, p + 2, 4);
                for (int j = 0; j < 16; ++j) {
                    const int lo = (int)((p[6 + j] & 0x0F) | (((qh >> j) & 1u) << 4)) - 16;
                    const int hi = (int)((p[6 + j] >> 4)   | (((qh >> (j + 16)) & 1u) << 4)) - 16;
                    dst[b * 32 + j]      = d * (float)lo;
                    dst[b * 32 + j + 16] = d * (float)hi;
                }
            }
            return RAD_OK;
        case GGML_Q5_1:
            for (int64_t b = 0; b < n / 32; ++b, p += 24) {
                const float d = half(p), m = half(p + 2);
                uint32_t qh; std::memcpy(&qh, p + 4, 4);
                for (int j = 0; j < 16; ++j) {
                    const int lo = (int)((p[8 + j] & 0x0F) | (((qh >> j) & 1u) << 4));
                    const int hi = (int)((p[8 + j] >> 4)   | (((qh >> (j + 16)) & 1u) << 4));
                    dst[b * 32 + j]      = d * (float)lo + m;
                    dst[b * 32 + j + 16] = d * (float)hi + m;
                }
            }
            return RAD_OK;
        case GGML_Q4_K:
            for (int64_t b = 0; b < n / 256; ++b, p += 144) {
                const float d = half(p), dmin = half(p + 2);
                const uint8_t* scales = p + 4;
                const uint8_t* qs = p + 16;
                for (int j = 0; j < 8; ++j) {
                    uint32_t sc, mn;
                    ggml_planes::k4_scale_min(j, scales, &sc, &mn);
                    const float ds = d * (float)sc, ms = dmin * (float)mn;
                    for (int l = 0; l < 32; ++l) {
                        const int v = j * 32 + l;
                        const uint8_t q = qs[32 * (v / 64) + v % 32];
                        const int code = (v % 64) < 32 ? q & 15 : q >> 4;
                        dst[b * 256 + v] = ds * (float)code - ms;
                    }
                }
            }
            return RAD_OK;
        case GGML_Q5_K:
            /* block: d, dmin, scales[12], qh[32], qs[128]. The fifth bit of value v is bit
             * (v/64)*2 + (v%64)/32 of qh's byte (v/64)*32 + v%32. */
            for (int64_t b = 0; b < n / 256; ++b, p += 176) {
                const float d = half(p), dmin = half(p + 2);
                const uint8_t* scales = p + 4;
                const uint8_t* qh = p + 16;
                const uint8_t* qs = p + 48;
                for (int j = 0; j < 256; j += 64) {
                    const int hb = j / 64;
                    const uint32_t u1 = 1u << (2 * hb), u2 = 1u << (2 * hb + 1);
                    uint32_t sc1, m1, sc2, m2;
                    ggml_planes::k4_scale_min(j / 32, scales, &sc1, &m1);
                    ggml_planes::k4_scale_min(j / 32 + 1, scales, &sc2, &m2);
                    const float d1 = d * (float)sc1, m1f = dmin * (float)m1;
                    const float d2 = d * (float)sc2, m2f = dmin * (float)m2;
                    const uint8_t* ql = qs + (j / 2);
                    /* The 32 qh bytes are shared across all four 64-blocks; only the mask moves. */
                    for (int l = 0; l < 32; ++l)
                        dst[b * 256 + j + l] = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1f;
                    for (int l = 0; l < 32; ++l)
                        dst[b * 256 + j + 32 + l] = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2f;
                }
            }
            return RAD_OK;
        case GGML_Q6_K:
            /* block: ql[128], qh[64], scales[16], d. */
            for (int64_t b = 0; b < n / 256; ++b, p += 210) {
                const uint8_t* ql = p;
                const uint8_t* qh = p + 128;
                const int8_t* scales = (const int8_t*)(p + 192);
                const float d = half(p + 208);
                for (int k = 0; k < 256; ++k) {
                    const int n128 = k / 128, t = (k % 128) / 32, l = k % 32;
                    const uint8_t lo = ql[64 * n128 + l + 32 * (t & 1)];
                    const int code = (int)((t < 2 ? lo & 15 : lo >> 4) |
                                           (((qh[32 * n128 + l] >> (2 * t)) & 3) << 4));
                    dst[b * 256 + k] = d * (float)scales[k / 16] * (float)(code - 32);
                }
            }
            return RAD_OK;
        case GGML_IQ4_NL:
            for (int64_t b = 0; b < n / 32; ++b, p += 18) {
                const float d = half(p);
                const uint8_t* qs = p + 2;
                for (int j = 0; j < 16; ++j) {
                    dst[b * 32 + j]      = d * (float)kIq4NlLevels[qs[j] & 0x0F];
                    dst[b * 32 + j + 16] = d * (float)kIq4NlLevels[qs[j] >> 4];
                }
            }
            return RAD_OK;
        default:
            return RAD_E_UNSUPPORTED;
    }
}

/* Where flat element `i` of the joined tensor lives: which piece, and the element in it. A join
 * along dim d repeats, for each index of the dims before d, every piece's [d..] extent in turn. */
struct Joined {
    std::vector<int64_t> inner;   /* each piece's elements per outer index */
    int64_t total_inner = 0;
    int     n = 0;
    void init(const std::vector<const SrcTensor*>& pieces, int d) {
        n = (int)pieces.size();
        inner.assign((size_t)n, 1);
        total_inner = 0;
        for (int k = 0; k < n; ++k) {
            for (size_t i = (size_t)d; i < pieces[(size_t)k]->shape.size(); ++i)
                inner[(size_t)k] *= pieces[(size_t)k]->shape[i];
            total_inner += inner[(size_t)k];
        }
    }
    /* Visit [a, b) as runs that lie in one piece: f(piece, element in it, count, at). */
    template <class F> int walk(int64_t a, int64_t b, F f) const {
        while (a < b) {
            const int64_t o = a / total_inner;
            int64_t w = a % total_inner, k = 0;
            while (w >= inner[(size_t)k]) w -= inner[(size_t)k++];
            const int64_t run = std::min<int64_t>(b - a, inner[(size_t)k] - w);
            const int rc = f((int)k, o * inner[(size_t)k] + w, run);
            if (rc < 0) return rc;
            a += run;
        }
        return RAD_OK;
    }
};

/* An unsigned integer code dtype (u1 .. u12, u8, u16). */
bool unsigned_int_dt(uint32_t dt) {
    int sgn = 0;
    return dt == RAD_U8 || dt == RAD_U16 || (rad_dtype_packed_int(dt, &sgn) > 0 && !sgn);
}

/* The stored zero point (the signed one plus 8) of row `n`, group `g` of an asymmetric CT-INT4
 * piece: nibble n % 8 of word [n / 8, g]. */
int ct_zero_at(const SrcTensor& t, int64_t n, int64_t g) {
    const SrcTensor& z = *t.ct_zero;
    uint32_t w;
    std::memcpy(&w, (const uint8_t*)z.data + ((n / 8) * z.shape[1] + g) * 4, 4);
    return (int)((w >> ((n % 8) * 4)) & 0xFu);
}

/* `n` elements of a stored-dtype piece from element `e`, as f32. */
int piece_f32(const SrcTensor& t, int64_t e, int64_t n, float* out) {
    if (t.ct_group) {
        /* Element (row, k) of the logical [N, K] matrix is nibble k % 8 of word [row, k / 8]: less
         * 8 for a symmetric weight (the signed code), as stored for an asymmetric one (the zero
         * point is subtracted with the scale, ckpt_rows_f32). */
        const int64_t K = t.shape[1];
        const int off = t.ct_zero ? 0 : 8;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t row = (e + i) / K, k = (e + i) % K;
            uint32_t w;
            std::memcpy(&w, (const uint8_t*)t.data + (row * (K / 8) + k / 8) * 4, 4);
            out[i] = (float)((int)((w >> ((k % 8) * 4)) & 0xFu) - off);
        }
        return RAD_OK;
    }
    if (t.gptq_bits) {
        /* Element (row, k) of the logical [N, K] matrix is field k % (32/bits) of word
         * [k / (32/bits), row], less the zero point. */
        const int64_t N = t.shape[0], K = t.shape[1];
        const int64_t per = 32 / t.gptq_bits;
        const uint32_t mask = (1u << t.gptq_bits) - 1u;
        const int zp = 1 << (t.gptq_bits - 1);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t row = (e + i) / K, k = (e + i) % K;
            uint32_t w;
            std::memcpy(&w, (const uint8_t*)t.data + ((k / per) * N + row) * 4, 4);
            out[i] = (float)((int)((w >> ((k % per) * t.gptq_bits)) & mask) - zp);
        }
        return RAD_OK;
    }
    if (t.rad_dtype == RAD_DT_INVALID) {
        /* A GGUF block type: whole rows only (a row is whole blocks). */
        const int64_t cols = t.shape.back();
        if (cols <= 0 || e % cols || n % cols) return RAD_E_SHAPE;
        const int64_t rows = t.numel() / cols, rb = t.bytes / (rows ? rows : 1);
        return ggml_rows_f32(t.ggml, (const uint8_t*)t.data + (e / cols) * rb, n, out);
    }
    const uint8_t* p = (const uint8_t*)t.data;
    switch (t.rad_dtype) {
        case RAD_F32:
            std::memcpy(out, (const float*)t.data + e, (size_t)n * 4);
            return RAD_OK;
        case RAD_BF16: {
            /* bf16 is the top half of an f32: a shift in a loop the compiler vectorises (the
             * per-element switch below does not). Most checkpoints are mostly bf16. */
            const uint16_t* q = (const uint16_t*)t.data + e;
            for (int64_t i = 0; i < n; ++i) {
                const uint32_t u = (uint32_t)q[i] << 16;
                std::memcpy(out + i, &u, 4);
            }
            return RAD_OK;
        }
        case RAD_F16: case RAD_F8E4M3: case RAD_F8E5M2: case RAD_FP4E2M1:
        case RAD_I8: case RAD_U8: case RAD_I16: case RAD_I32: case RAD_U32: case RAD_U16:
            for (int64_t i = 0; i < n; ++i) out[i] = rad_load_f32(p, t.rad_dtype, e + i);
            return RAD_OK;
        default:
            return RAD_E_UNSUPPORTED;
    }
}

}  /* namespace */

int ckpt_rows_f32(const CkptWeight& w, int64_t r0, int64_t nr, float* out, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    if (r0 < 0 || nr < 0 || r0 + nr > w.rows) return refuse(RAD_E_SHAPE, "rows out of range");
    /* A rotated weight's codes are in the rotated basis; its f32 value needs the container's tables
     * to decode, and nothing re-quantises it. */
    if (w.paro)
        return refuse(RAD_E_UNSUPPORTED, "'" + w.name + "' is a ParoQuant weight, stored rotated; "
                                         "it is kept as it is and has no f32 view here");
    if (!w.derived.empty()) {
        std::memcpy(out, w.derived.data() + r0 * w.cols, (size_t)(nr * w.cols) * sizeof(float));
        return RAD_OK;
    }
    if (w.wnorm) {
        const SrcTensor& v = *w.pieces[0];
        if (piece_f32(v, r0 * w.cols, nr * w.cols, out) < 0)
            return refuse(RAD_E_FORMAT, "'" + v.name + "' cannot be read as f32");
        /* View rows per dim-0 slice: the dims between the first and the last. */
        const int64_t rps = w.rows / (int64_t)w.wnorm_scale.size();
        for (int64_t r = 0; r < nr; ++r) {
            const float f = w.wnorm_scale[(size_t)((r0 + r) / rps)];
            for (int64_t c = 0; c < w.cols; ++c) out[r * w.cols + c] *= f;
        }
        return RAD_OK;
    }
    Joined j;
    j.init(w.pieces, w.concat_dim);
    /* Runs arrive in order, so the cursor is the count already written. */
    int64_t at = 0;
    const int rc = j.walk(r0 * w.cols, (r0 + nr) * w.cols, [&](int k, int64_t e, int64_t n) {
        const int s = piece_f32(*w.pieces[(size_t)k], e, n, out + at);
        at += n;
        return s;
    });
    if (rc < 0)
        return refuse(rc, "'" + w.pieces[0]->name + "' is stored in a form that cannot be read "
                          "as f32 (" + (w.pieces[0]->rad_dtype == RAD_DT_INVALID
                                            ? std::string(gguf_dtype(w.pieces[0]->ggml)
                                                              ? gguf_dtype(w.pieces[0]->ggml)->name
                                                              : "?")
                                            : std::string(rad_dtype_name(w.pieces[0]->rad_dtype))) +
                          ")");
    /* An imported code times its group's shipped scale: GPTQ's scales are [groups, N] (row r's are
     * a column), CT-INT4's are [N, groups] (a row), and an asymmetric CT-INT4 code has its
     * group's zero point subtracted first, (q - zero) * scale. Both factors are small integers
     * times one float, so the f32 value is exact. An NVFP4 import has neither group and takes the
     * scale-plane form below. */
    if (w.imported && (w.pieces[0]->gptq_group || w.pieces[0]->ct_group)) {
        int64_t base = 0;
        for (size_t k = 0; k < w.pieces.size(); ++k) {
            const SrcTensor& pk = *w.pieces[k];
            const SrcTensor& sk = *w.scales[k];
            const int64_t pr = pk.shape[0];
            for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + pr); ++r)
                for (int64_t c = 0; c < w.cols; ++c) {
                    float& v = out[(r - r0) * w.cols + c];
                    if (pk.ct_zero) v -= (float)ct_zero_at(pk, r - base, c / pk.ct_group);
                    v *= rad_load_f32(
                        sk.data, sk.rad_dtype,
                        pk.ct_group ? (r - base) * (w.cols / pk.ct_group) + c / pk.ct_group
                                    : (c / pk.gptq_group) * pr + (r - base));
                }
            base += pr;
        }
        return RAD_OK;
    }
    /* A code times its scale-plane element (FP8's block or row scale, MXFP4's exponent, NVFP4's
     * folded scale). Pieces join along rows, so logical row r takes its covering piece's plane.
     * A weight whose planes ckpt_resolve could not state an encoding for (mixed shapes, a
     * column join, a piece without a plane) is refused rather than read with a guessed block. */
    bool has_scale = false;
    for (const SrcTensor* s : w.scales)
        if (s) { has_scale = true; break; }
    if (has_scale) {
        if (w.enc.n_planes < 2)
            return refuse(RAD_E_UNSUPPORTED, "'" + w.name + "' has a scale plane with no stated "
                                             "encoding" + (w.why.empty() ? "" : ": " + w.why));
        /* A block extent of 0 is the whole extent (one scale per row or per column). */
        const int64_t br0 = w.enc.plane[1].block[0], bc0 = w.enc.plane[1].block[1];
        int64_t base = 0;
        for (size_t k = 0; k < w.pieces.size(); ++k) {
            const SrcTensor& pk = *w.pieces[k];
            const SrcTensor& sk = *w.scales[k];
            int64_t pr = 0, pc = 0;
            rad_enc_view((uint32_t)pk.shape.size(), pk.shape.data(), &pr, &pc);
            const int64_t br = br0 > 0 ? br0 : pr, bc = bc0 > 0 ? bc0 : pc;
            const int64_t sc = (pc + bc - 1) / bc;
            /* A piece's single scale (FP8 per tensor) covers every one of its rows. */
            const bool one = per_tensor_scale(sk, pr);
            for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + pr); ++r)
                for (int64_t c = 0; c < w.cols; ++c)
                    out[(r - r0) * w.cols + c] *=
                        rad_load_f32(sk.data, sk.rad_dtype,
                                     one ? 0 : ((r - base) / br) * sc + c / bc);
            base += pr;
        }
    }
    return RAD_OK;
}

int ckpt_import_form_parse(const std::vector<std::pair<std::string, std::string>>& options,
                           ImportForm* f, std::string* why) {
    auto refuse = [&](std::string m) {
        if (why) *why = std::move(m);
        return (int)RAD_E_INVAL;
    };
    for (const auto& [k, v] : options) {
        uint32_t* dt = k == "codes" ? &f->codes : k == "zero" ? &f->zero
                     : k == "scale" ? &f->scale : nullptr;
        if (dt) {
            *dt = rad_dtype_parse(v.c_str());
            if (*dt == RAD_DT_INVALID) return refuse("import's " + k + "=" + v + " is not a dtype");
        } else if (k == "block") {
            long long a = 0, b = 0;
            char x = 0, tail = 0;
            if (std::sscanf(v.c_str(), "%lld%c%lld%c", &a, &x, &b, &tail) != 3 || x != 'x' ||
                a < 1 || b < 1)
                return refuse("import's block=" + v + " is not ROWSxCOLS");
            f->block[0] = a;
            f->block[1] = b;
        } else {
            return refuse("the import rule takes codes=, zero=, scale= and block=, not " + k);
        }
    }
    return RAD_OK;
}

int ckpt_import_form(CkptWeight* w, const ImportForm& f, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    char en[160];
    rad_enc_format(&w->enc, en, sizeof en);
    if (w->pieces.empty() || w->paro || w->wnorm || !w->derived.empty())
        return refuse(RAD_E_FORMAT, "'" + w->name + "' is " + en + ", which the import rule does "
                                    "not re-express");
    if (rad_enc_is(&w->enc, "plain"))
        return refuse(RAD_E_UNSUPPORTED, "'" + w->name + "' ships as plain " + en +
                                         ", with nothing quantised to re-express");
    const SrcTensor& p0 = *w->pieces[0];
    const RadEncPlane* sp = rad_enc_plane(&w->enc, "scale");
    auto float_dt = [](uint32_t d) { return d == RAD_F16 || d == RAD_BF16 || d == RAD_F32; };
    if (f.scale != RAD_DT_INVALID && !float_dt(f.scale))
        return refuse(RAD_E_INVAL, std::string("the import rule's scale=") +
                                   rad_dtype_name(f.scale) + " is not f16, bf16 or f32");
    const uint32_t sdt = f.scale != RAD_DT_INVALID ? f.scale
                       : sp ? sp->dtype : (uint32_t)RAD_DT_INVALID;

    /* Every shipped scale once, as f32, with whether `sdt` holds it. */
    auto count = [&](const SrcTensor& s) {
        const int64_t n = s.shape.empty() ? 1 : s.numel();
        for (int64_t i = 0; i < n; ++i) {
            const float v = rad_load_f32(s.data, s.rad_dtype, i);
            uint8_t b[4];
            rad_store_f32(b, sdt, 0, v);
            const float r = rad_load_f32(b, sdt, 0);
            if (r != v) {
                ++w->rounded;
                if (v != 0.0f) w->max_rel = std::max(w->max_rel, (double)std::fabs((r - v) / v));
            }
        }
    };
    w->rounded = 0;
    w->max_rel = 0;
    RadEncoding e{};
    rad_enc_copy_str(e.scheme, "affine");

    if (w->imported && (p0.gptq_group || p0.ct_group)) {
        /* An integer grid: GPTQ's or CT-INT4's codes, a scale per group and, when asymmetric, a
         * zero point. */
        const int bits = p0.ct_group ? 4 : p0.gptq_bits;
        const bool asym = p0.ct_zero != nullptr;
        if (f.block[0] || f.block[1])
            return refuse(RAD_E_INVAL, "'" + w->name + "' is an integer grid (" + en + "); the "
                                       "import rule's block= re-grids an FP8 scale, not a group");
        int sgn = 0;
        const bool unsign = asym || f.codes != RAD_DT_INVALID;
        if (f.codes != RAD_DT_INVALID &&
            (rad_dtype_packed_int(f.codes, &sgn) != bits || sgn) && !(bits == 8 && f.codes == RAD_U8))
            return refuse(RAD_E_INVAL, std::string("the import rule's codes=") +
                                       rad_dtype_name(f.codes) + " is not the unsigned " +
                                       std::to_string(bits) + "-bit dtype of '" + w->name + "' (" +
                                       en + ")");
        if (!unsign && f.zero != RAD_DT_INVALID)
            return refuse(RAD_E_INVAL, "'" + w->name + "' is a symmetric grid (" + en + ") and has "
                                       "no zero point for zero= to store; codes=u" +
                                       std::to_string(bits) + " makes one");
        const uint32_t zdt = f.zero != RAD_DT_INVALID ? f.zero
                           : asym ? (uint32_t)RAD_U4 : (uint32_t)RAD_DT_INVALID;
        if (unsign && zdt == RAD_DT_INVALID)
            return refuse(RAD_E_INVAL, "the import rule's codes=" + std::string(rad_dtype_name(f.codes)) +
                                       " makes '" + w->name + "' unsigned and needs zero= for the "
                                       "zero point " + std::to_string(1 << (bits - 1)));
        if (unsign && !(unsigned_int_dt(zdt) && rad_dtype_bits(zdt) >= bits))
            return refuse(RAD_E_INVAL, std::string("the import rule's zero=") + rad_dtype_name(zdt) +
                                       " does not hold the " + std::to_string(bits) +
                                       "-bit zero points of '" + w->name + "'");
        const int64_t gs = sp->block[1];
        const uint32_t cdt = unsign ? (uint32_t)(bits == 4 ? RAD_U4 : RAD_U8) : w->enc.plane[0].dtype;
        rad_enc_add_plane(&e, "codes", cdt, 1, 1);
        rad_enc_add_plane(&e, "scale", sdt, 1, gs);
        if (unsign) rad_enc_add_plane(&e, "zero", zdt, 1, gs);
        for (const SrcTensor* s : w->scales) count(*s);
    } else if ((w->enc.plane[0].dtype == RAD_F8E4M3 || w->enc.plane[0].dtype == RAD_F8E5M2) &&
               sp && sp->block[0] == 1 && sp->block[1] == 0) {
        /* FP8 with a scale per row or per tensor. */
        if (f.codes != RAD_DT_INVALID || f.zero != RAD_DT_INVALID)
            return refuse(RAD_E_INVAL, "'" + w->name + "' is FP8 (" + en + "), with no integer "
                                       "codes or zero point for codes= or zero=");
        const int64_t br = f.block[0] ? f.block[0] : 1, bc = f.block[1];
        if (br < 1 || bc < 0)
            return refuse(RAD_E_INVAL, "the import rule's block= is not rows x columns");
        /* Each block's rows must share one scale: a per-tensor piece's rows all do, and a block
         * may not straddle two pieces; per-row scales must agree within a block. */
        int64_t base = 0;
        for (size_t p = 0; p < w->pieces.size(); ++p) {
            const SrcTensor& s = *w->scales[p];
            const int64_t pr = w->pieces[p]->shape[0];
            const bool one = per_tensor_scale(s, pr);
            if (br > 1 && (base % br || (p + 1 < w->pieces.size() && (base + pr) % br)))
                return refuse(RAD_E_SHAPE, "'" + w->pieces[p]->name + "' starts or ends inside a " +
                                           std::to_string(br) + "-row block of '" + w->name + "'");
            if (!one)
                for (int64_t r = 0; r < pr; ++r)
                    if (rad_load_f32(s.data, s.rad_dtype, r) !=
                        rad_load_f32(s.data, s.rad_dtype, r - r % br))
                        return refuse(RAD_E_FORMAT, "'" + s.name + "' gives rows " +
                                                    std::to_string(r - r % br) + " and " +
                                                    std::to_string(r) + " of one " +
                                                    std::to_string(br) + "-row block different "
                                                    "scales, which one block scale cannot hold");
            count(s);
            base += pr;
        }
        rad_enc_add_plane(&e, "codes", w->enc.plane[0].dtype, 1, 1);
        rad_enc_add_plane(&e, "scale", sdt, br, bc);
    } else {
        return refuse(RAD_E_FORMAT, "'" + w->name + "' is " + en + "; the import rule re-expresses "
                                    "an integer grid (GPTQ, CT-INT4) or an FP8 weight with a scale "
                                    "per row or per tensor");
    }
    if (!rad_enc_valid(&e))
        return refuse(RAD_E_FORMAT, "the import rule's form of '" + w->name + "' is malformed");
    w->enc = e;
    w->form = f;
    w->imported = true;
    w->trivial = false;
    w->why = "it is re-expressed by an import rule";
    return RAD_OK;
}


int Checkpoint::bailingmoe3_layouts() {
    /* The release GGUF keeps layouts the plugin's names cannot spell through the row and slice
     * maps. The MLA layers' latent projections are head-last: k_b [128, 512, heads] and
     * v_b [512, 128, heads] (a 128-row k block and a 128-row v block a head), where the plugin
     * reads one interleaved kv_b [heads * 2 * 128, 512] -- head h's k rows [2 * 128 * h,
     * + 128), its v rows the next 128. The MoE layers' routed stacks are expert-last (gate and
     * up [ff, emb, E], down [emb, ff, E]); the reader's fastest-first flip orients them, and
     * the block's split-stacked source slices one expert off the flipped [E, ...] stack, so
     * they stay where they are. Everything the flip already hands over in the plugin's
     * orientation (the shared expert, the KDA and the norms) needs only a name, which the
     * plugin takes from the gguf_source meta. */
    const int64_t n_layer = meta_.m.n_layers;
    const char* grp_s = meta_.get("layer_group_size");
    const int64_t group = grp_s ? std::stoll(grp_s) : 4;
    const char* dense_s = meta_.get("first_k_dense_replace");
    const int64_t dense = dense_s ? std::stoll(dense_s) : 1;
    const char* heads_s = meta_.get("num_attention_heads");
    const int64_t heads = heads_s ? std::stoll(heads_s) : 16;
    const char* latent_s = meta_.get("kv_lora_rank");
    const int64_t latent = latent_s ? std::stoll(latent_s) : 512;
    const char* ff_s = meta_.get("moe_intermediate_size");
    const int64_t ff = ff_s ? std::stoll(ff_s) : 512;
    /* The experts' third axis is the embedding width, not the dense feed-forward's. */
    const int64_t emb = meta_.m.n_embd;
    const char* nexp_s = meta_.get("num_experts");
    const int64_t n_expert = nexp_s ? std::stoll(nexp_s) : 0;
    if (n_layer < 1 || group < 2 || heads < 1 || latent < 1 || ff < 1 || emb < 1)
        return RAD_E_FORMAT;

    /* Decode a whole source tensor to f32: a block type row by row, a plain one a copy. */
    auto whole = [&](const SrcTensor* t, std::vector<float>& dst) -> int {
        const int64_t n = t->numel();
        dst.resize((size_t)n);
        return piece_f32(*t, 0, n, dst.data());
    };
    auto add = [&](const std::string& name, std::vector<int64_t> shape,
                   std::vector<float>&& v) -> int {
        if (index_.count(name)) return RAD_OK;   /* the checkpoint already spells it */
        SrcTensor s;
        s.name      = name;
        s.rad_dtype = RAD_F32;
        s.shape     = std::move(shape);
        s.owned     = std::move(v);
        s.data      = s.owned.data();
        s.bytes     = (int64_t)(s.owned.size() * sizeof(float));
        index_.emplace(s.name, tensors_.size());
        tensors_.push_back(std::move(s));
        return RAD_OK;
    };

    /* The exporter's ssm_a is exp(A_log) (llama.cpp's bailingmoe3 modify_tensors); the KDA
     * kernel exponentiates internally, so the container wants the log values back. */
    for (int64_t l = 0; l < n_layer; ++l) {
        if ((l + 1) % group == 0) continue;   /* the MLA layers carry no ssm_a */
        const SrcTensor* sa = find(fmt("blk.%lld.ssm_a", (long long)l));
        if (!sa) continue;
        std::vector<float> a3, raw;
        RAD_TRY(whole(sa, a3));
        raw.resize(a3.size());
        for (size_t i = 0; i < a3.size(); ++i)
            raw[i] = std::logf(a3[i] > 0.0f ? a3[i] : 1.0f);
        RAD_TRY(add(fmt("blk.%lld.ssm_a_raw", (long long)l), sa->shape, std::move(raw)));
    }

    for (int64_t l = 1; l < n_layer; ++l) {
        if ((l + 1) % group) continue;   /* the MLA layers */
        const std::string kn = fmt("blk.%lld.attn_k_b.weight", (long long)l);
        const std::string vn = fmt("blk.%lld.attn_v_b.weight", (long long)l);
        const SrcTensor* k = find(kn);
        const SrcTensor* v = find(vn);
        if (!k || !v) continue;
        if (k->shape.size() != 3 || v->shape.size() != 3 || k->shape[0] != heads ||
            k->shape[1] != latent || k->shape[2] != 128 || v->shape[0] != heads ||
            v->shape[1] != 128 || v->shape[2] != latent)
            return RAD_E_SHAPE;
        std::vector<float> k3, v3, fused;
        RAD_TRY(whole(k, k3));
        RAD_TRY(whole(v, v3));
        fused.assign((size_t)(heads * (2 * 128) * latent), 0.0f);
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t d = 0; d < 128; ++d) {
                /* k_b's head block [latent, 128] transposed into the 128 k rows. */
                for (int64_t l2 = 0; l2 < latent; ++l2)
                    fused[((h * 2 + 0) * 128 + d) * latent + l2] = k3[((h * latent + l2) * 128) + d];
                /* v_b's head block [128, latent] already runs its v rows. */
                std::memcpy(&fused[((h * 2 + 1) * 128 + d) * latent],
                            &v3[((h * 128 + d) * latent)], (size_t)latent * sizeof(float));
            }
        RAD_TRY(add(fmt("blk.%lld.attn_kv_b.weight", (long long)l),
                    { heads * 2 * 128, latent }, std::move(fused)));
        /* q_b is one [heads * (nope + rope), q_lora] matrix the plugin row-maps; a quantised
         * source cannot be row-mapped, so split it into the plain nope and rope blocks. The
         * per-head boundary comes from the meta the reader republished, so the split only runs
         * when the geometry is the one the plugin will consume. */
        const char* nps = meta_.get("qk_nope_head_dim"), *rps = meta_.get("qk_rope_head_dim");
        const int64_t qn = nps ? std::stoll(nps) : 0, qr = rps ? std::stoll(rps) : 0;
        const SrcTensor* qb = find(fmt("blk.%lld.attn_q_b.weight", (long long)l));
        if (qb && qb->shape.size() == 2 && qb->shape[1] > 0 && qn > 0 && qr > 0 &&
            qb->shape[0] == heads * (qn + qr)) {
            const int64_t ql = qb->shape[1], w = qn + qr;
            std::vector<float> q3, nblk, rblk;
            RAD_TRY(whole(qb, q3));
            nblk.assign((size_t)(heads * qn) * ql, 0.0f);
            rblk.assign((size_t)(heads * qr) * ql, 0.0f);
            for (int64_t h = 0; h < heads; ++h) {
                std::memcpy(&nblk[(h * qn) * ql], &q3[(h * w) * ql], (size_t)(qn * ql) * sizeof(float));
                std::memcpy(&rblk[(h * qr) * ql], &q3[(h * w + qn) * ql], (size_t)(qr * ql) * sizeof(float));
            }
            RAD_TRY(add(fmt("blk.%lld.attn_q_b_nope.weight", (long long)l), { heads * qn, ql }, std::move(nblk)));
            RAD_TRY(add(fmt("blk.%lld.attn_q_b_rope.weight", (long long)l), { heads * qr, ql }, std::move(rblk)));
        }
    }
    for (int64_t l = dense; l < n_layer; ++l) {
        if (n_expert < 1) continue;
        /* The routed stacks stay where they are: the block's split-stacked source
         * (gate, up, down per expert, a dim-0 slice each) slices the reader's flipped
         * [E, ff, emb] / [E, emb, ff] tensors directly. Only the shared expert needs
         * re-spelling under the block's prefix: copies, the reader's flip already
         * orienting them. */
        const char* sh[3] = { "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp" };
        const char* shd[3] = { "mlp.shared_experts.gate_proj.weight",
                               "mlp.shared_experts.up_proj.weight",
                               "mlp.shared_experts.down_proj.weight" };
        for (int q = 0; q < 3; ++q) {
            const SrcTensor* t = find(fmt("blk.%lld.%s.weight", (long long)l, sh[q]));
            if (!t) continue;
            std::vector<float> v;
            RAD_TRY(whole(t, v));
            RAD_TRY(add(fmt("blk.%lld.%s", (long long)l, shd[q]), t->shape, std::move(v)));
        }
    }
    return RAD_OK;
}

int ckpt_plane_rows(const CkptWeight& w, int k, int64_t r0, int64_t nr, void* dst,
                    std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    if (!(w.trivial || w.imported) || k < 0 || k >= w.enc.n_planes)
        return refuse(RAD_E_STATE, "not a plane of a trivial or imported encoding");
    const std::vector<const SrcTensor*>& src = k == 0 ? w.pieces : w.scales;
    int64_t pr = 0, pc = 0;
    rad_enc_plane_dims(&w.enc.plane[k], w.rows, w.cols, &pr, &pc);
    if (r0 < 0 || nr < 0 || r0 + nr > pr) return refuse(RAD_E_SHAPE, "plane rows out of range");
    const uint32_t dt = w.enc.plane[k].dtype;
    const int64_t eb = rad_dtype_bits(dt) / 8;
    uint8_t* o = (uint8_t*)dst;
    if (w.paro && w.pieces[0]->pbits && k <= 2) {
        /* Paro int5-bitplane, re-laid one row of the logical [N, K] matrix at a time. The
         * checkpoint packs along N: qweight [K, N/8] holds eight low nibbles per word in AWQ order
         * (column c of a group of eight is in nibble nib(c)), and qweight_hi [K, N/32] the fifth
         * bit of 32 columns at bit n % 32. Zero points use the same two packings over [G, N]. */
        auto nib = [](int c) { return c % 2 == 0 ? c / 2 : 4 + c / 2; };      /* AWQ order */
        auto u32 = [](const SrcTensor& t, int64_t i) {
            uint32_t v;
            std::memcpy(&v, (const uint8_t*)t.data + i * 4, 4);
            return v;
        };
        int64_t base = 0;
        for (size_t pi = 0; pi < w.pieces.size(); ++pi) {
            const SrcTensor* pc = w.pieces[pi];
            const int64_t Np = pc->shape[0], K = pc->shape[1], G = K / 128;
            const SrcTensor *hip = pc->aux[0], *qzp = pc->aux[1], *qzhp = pc->aux[2];
            const SrcTensor& sc = *w.scales[pi];
            for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + Np); ++r) {
                const int64_t n = r - base;
                const int cn = (int)(n % 8);
                if (k == 0) {
                    const int pb = pc->pbits;
                    const int64_t rb = (K * pb + 7) / 8;
                    std::memset(o, 0, (size_t)rb);
                    for (int64_t kk = 0; kk < K; ++kk) {
                        const uint32_t lo = (u32(*pc, kk * (Np / 8) + n / 8) >> (4 * nib(cn))) & 15u;
                        const uint32_t h = pb == 5 ? (u32(*hip, kk * (Np / 32) + n / 32) >> (n % 32)) & 1u : 0u;
                        const uint32_t code = lo | (h << 4);
                        const int64_t bit = kk * pb;
                        const uint32_t sh = (uint32_t)(bit & 7);
                        const uint32_t v = code << sh;
                        o[bit >> 3] |= (uint8_t)(v & 0xFFu);
                        if (sh + pb > 8) o[(bit >> 3) + 1] |= (uint8_t)(v >> 8);
                    }
                    o += rb;
                } else if (k == 1) {
                    for (int64_t g = 0; g < G; ++g) {
                        std::memcpy(o, (const uint8_t*)sc.data + (g * Np + n) * 2, 2);
                        o += 2;
                    }
                } else {
                    for (int64_t g = 0; g < G; ++g) {
                        const uint32_t lo = (u32(*qzp, g * (Np / 8) + n / 8) >> (4 * nib(cn))) & 15u;
                        const uint32_t h = pc->pbits == 5 ? (u32(*qzhp, g * (Np / 32) + n / 32) >> (n % 32)) & 1u : 0u;
                        *o++ = (uint8_t)(lo | (h << 4));
                    }
                }
            }
            base += Np;
        }
        return RAD_OK;
    }
    if (w.paro && std::strncmp(w.enc.plane[k].role, "meta", RAD_ENC_STR) == 0) {
        /* The M2XFP4 grid's metadata byte a group, as stored: [N][K/32], one byte a group, the
         * subgroup multipliers packed in it (subgroup s in bits [2s, 2s+2)). The checkpoint packs
         * along N the way the plane does, so a row is a memcpy. */
        int64_t base = 0;
        for (size_t p = 0; p < w.pieces.size() && p < w.pq_meta.size(); ++p) {
            const SrcTensor& m = *w.pq_meta[p];
            const int64_t Np = w.pieces[p]->shape[0];
            /* The pieces view is the logical [N, K] codes; one metadata byte covers 32 of K. */
            const int64_t ng = w.pieces[p]->shape[1] / 32;
            if (m.shape.size() != 2 || m.shape[0] != Np || m.shape[1] != ng)
                return refuse(RAD_E_SHAPE, "'" + m.name + "' is not the [" + std::to_string(Np) +
                                           ", " + std::to_string(ng) + "] metadata plane of '" +
                                           w.pieces[p]->name + "'; a stacked form it cannot be "
                                           "sliced out of is refused rather than packed");
            const int64_t lo = std::max(r0, base), hi = std::min(r0 + nr, base + Np);
            if (hi > lo) {
                std::memcpy(o, (const uint8_t*)m.data + (lo - base) * ng,
                            (size_t)((hi - lo) * ng));
                o += (hi - lo) * ng;
            }
            base += Np;
        }
        return RAD_OK;
    }
    if (w.paro && std::strncmp(w.enc.plane[k].role, "t.", 2) == 0) {
        /* A rotation table, built a row at a time: pairs converted from group-local to whole-row
         * columns, angles widened, the channel scale inverted (the checkpoint multiplies the
         * activation by it; t.scale is the factor the stored row was multiplied by), and the part
         * ends. */
        const RadEncPlane& pl = w.enc.plane[k];
        const int64_t P = (int64_t)w.pieces.size(), S = w.pq_pairs[0]->shape[0], K = w.cols;
        const std::string role = pl.role;
        for (int64_t r = r0; r < r0 + nr; ++r) {
            if (role == "t.pairs") {
                const int16_t* pv = (const int16_t*)w.pq_pairs[(size_t)(r / S)]->data + (r % S) * K;
                int32_t* d = (int32_t*)o;
                for (int64_t c = 0; c < K; ++c)
                    d[c] = (int32_t)((c / kParoGroup) * kParoGroup + pv[c]);
                o += K * 4;
            } else if (role == "t.angle") {
                const SrcTensor& th = *w.pq_theta[(size_t)(r / S)];
                float* d = (float*)o;
                for (int64_t c = 0; c < K / 2; ++c)
                    d[c] = rad_load_f32(th.data, th.rad_dtype, (r % S) * (K / 2) + c);
                o += K / 2 * 4;
            } else if (role == "t.scale") {
                const SrcTensor& cs = *w.pq_scale[(size_t)r];
                float* d = (float*)o;
                for (int64_t c = 0; c < K; ++c) d[c] = 1.0f / rad_load_f32(cs.data, cs.rad_dtype, c);
                o += K * 4;
            } else if (role == "t.part") {
                int32_t* d = (int32_t*)o;
                int64_t end = 0;
                for (int64_t q = 0; q < P; ++q) {
                    end += w.pieces[(size_t)q]->shape[0];
                    d[q] = (int32_t)end;
                }
                o += P * 4;
            } else if (role == "t.dense") {
                /* Row r is part r / (K/128), group r % (K/128): its group-local stages composed. */
                const int64_t nkb = K / kParoGroup, H = kParoGroup / 2, q = r / nkb, g = r % nkb;
                const int16_t* pv = (const int16_t*)w.pq_pairs[(size_t)q]->data;
                const SrcTensor& th = *w.pq_theta[(size_t)q];
                std::vector<int32_t> ia((size_t)(S * H)), ib((size_t)(S * H));
                std::vector<float> an((size_t)(S * H));
                for (int64_t st = 0; st < S; ++st)
                    for (int64_t j = 0; j < H; ++j) {
                        const int64_t at = st * H + j;
                        ia[(size_t)at] = pv[st * K + g * kParoGroup + 2 * j];
                        ib[(size_t)at] = pv[st * K + g * kParoGroup + 2 * j + 1];
                        an[(size_t)at] = rad_load_f32(th.data, th.rad_dtype, st * (K / 2) + g * H + j);
                    }
                paro_compose_dense(S, ia.data(), ib.data(), an.data(), (uint16_t*)o);
                o += kParoGroup * kParoGroup * 2;
            } else {
                return refuse(RAD_E_STATE, "plane '" + role + "' of a ParoQuant weight is not a table");
            }
        }
        return RAD_OK;
    }
    if (w.imported && !w.paro && (w.pieces[0]->gptq_group || w.pieces[0]->ct_group)) {
        /* GPTQ-packed and CT-INT4, re-laid row by row. Both store each code plus 2^(bits-1)
         * (GPTQ's symmetric zero point, CT's packing offset), so a signed code is the stored field
         * with its top bit flipped and an unsigned one (CT's asymmetric codes, or an `import`
         * rule's codes=uN) is the field as stored.
         *   codes  GPTQ: a column of the qweight, its words' bytes in order (int4: two codes a
         *          byte, low nibble first); CT: a row of the packed words, already in that order
         *   scale  GPTQ: a column of the scale tensor, one per 128 of K; CT: a row of the scale
         *          tensor, one per group. Stored in the plane's dtype, rounded to nearest (bf16
         *          by default for GPTQ, as shipped for CT)
         *   zero   CT's asymmetric zero point as stored, or 2^(bits-1) for a symmetric grid an
         *          `import` rule made unsigned */
        const std::string role = w.enc.plane[k].role;
        const bool unsign = unsigned_int_dt(w.enc.plane[0].dtype);
        int64_t base = 0;
        for (size_t p = 0; p < w.pieces.size(); ++p) {
            const SrcTensor& c = *w.pieces[p];
            const SrcTensor& s = *w.scales[p];
            const int64_t N = c.shape[0], K = c.shape[1];
            const int bits = c.ct_group ? 4 : c.gptq_bits;
            const uint8_t flip = unsign ? 0 : bits == 4 ? 0x88 : 0x80;
            /* Scales a row: one per CT group, or one per 128 of K for GPTQ. */
            const int64_t gs = c.ct_group ? c.ct_group : 128, G = K / gs;
            for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + N); ++r) {
                const int64_t n = r - base;
                if (role == "codes" && c.ct_group) {
                    const uint8_t* q = (const uint8_t*)c.data + n * (K / 2);
                    for (int64_t i = 0; i < K / 2; ++i) *o++ = q[i] ^ flip;
                } else if (role == "codes") {
                    const uint8_t* q = (const uint8_t*)c.data;
                    for (int64_t wi = 0; wi < K * c.gptq_bits / 32; ++wi) {
                        uint32_t v;
                        std::memcpy(&v, q + (wi * N + n) * 4, 4);
                        for (int b = 0; b < 4; ++b) *o++ = (uint8_t)(v >> (8 * b)) ^ flip;
                    }
                } else if (role == "scale") {
                    for (int64_t g = 0; g < G; ++g)
                        rad_store_f32(o, dt, g, rad_load_f32(s.data, s.rad_dtype,
                                                             c.ct_group ? n * G + g
                                                                        : ((g * 128) / c.gptq_group) * N + n));
                    o += G * eb;
                } else if (role == "zero") {
                    const int64_t zb = rad_enc_row_bytes(dt, G);
                    std::memset(o, 0, (size_t)zb);
                    for (int64_t g = 0; g < G; ++g)
                        rad_store_code(o, dt, g, c.ct_zero ? ct_zero_at(c, n, g) : 1 << (bits - 1));
                    o += zb;
                } else {
                    return refuse(RAD_E_STATE, "plane '" + role + "' is not one of an integer grid");
                }
            }
            base += N;
        }
        return RAD_OK;
    }
    if (w.ggml_blocks) {
        /* GGUF blocks, split one row at a time into every plane's row: a call pays the whole row
         * for the one plane it keeps, and rad-convert asks plane by plane. */
        int64_t at[RAD_ENC_MAX_PLANES + 1] = { 0 };
        for (int i = 0; i < w.enc.n_planes; ++i) {
            int64_t qr = 0, qc = 0;
            rad_enc_plane_dims(&w.enc.plane[i], 1, w.cols, &qr, &qc);
            at[i + 1] = at[i] + rad_enc_row_bytes(w.enc.plane[i].dtype, qc);
        }
        std::vector<uint8_t> split((size_t)at[w.enc.n_planes]);
        uint8_t* row[RAD_ENC_MAX_PLANES] = { nullptr };
        for (int i = 0; i < w.enc.n_planes; ++i) row[i] = split.data() + at[i];
        int64_t base = 0;
        for (const SrcTensor* piece : w.pieces) {
            const int64_t prows = piece->numel() / w.cols, rb = piece->bytes / prows;
            for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + prows); ++r) {
                ggml_plane_row(piece->ggml, (const uint8_t*)piece->data + (r - base) * rb, w.cols,
                               row);
                std::memcpy(o, row[k], (size_t)(at[k + 1] - at[k]));
                o += at[k + 1] - at[k];
            }
            base += prows;
        }
        return RAD_OK;
    }
    if (k == 0) {
        /* Codes: the pieces joined, bytes as stored. Sub-byte codes (MXFP4) join along rows of even
         * width, so every run is byte-aligned. */
        const int64_t bits = rad_dtype_bits(dt);
        Joined j;
        j.init(src, w.concat_dim);
        return j.walk(r0 * pc, (r0 + nr) * pc, [&](int p, int64_t e, int64_t n) {
            if ((e * bits) % 8 || (n * bits) % 8) return (int)RAD_E_SHAPE;
            std::memcpy(o, (const uint8_t*)src[(size_t)p]->data + e * bits / 8, (size_t)(n * bits / 8));
            o += n * bits / 8;
            return (int)RAD_OK;
        });
    }
    /* An FP8 scale shipped once per row or per tensor and written per row (a per-tensor piece's
     * scale repeated), or over the block grid an `import` rule asks for, rounded to the plane's
     * dtype. ckpt_import_form checked that every block's rows share their scale. */
    bool any_one = false;
    for (size_t p = 0; p < src.size() && k > 0; ++p)
        any_one = any_one || per_tensor_scale(*src[p], w.pieces[p]->shape[0]);
    if (w.imported && k > 0 &&
        (dt != src[0]->rad_dtype || w.enc.plane[k].block[0] != 1 || any_one)) {
        const int64_t br = w.enc.plane[k].block[0] > 0 ? w.enc.plane[k].block[0] : w.rows;
        for (int64_t i = r0; i < r0 + nr; ++i) {
            /* Logical row i * br's piece, and its scale. */
            int64_t r = i * br, base = 0;
            size_t p = 0;
            while (p + 1 < w.pieces.size() && r >= base + w.pieces[p]->shape[0])
                base += w.pieces[p++]->shape[0];
            const SrcTensor& s = *src[p];
            const bool one = per_tensor_scale(s, w.pieces[p]->shape[0]);
            const float v = rad_load_f32(s.data, s.rad_dtype, one ? 0 : r - base);
            for (int64_t j = 0; j < pc; ++j) rad_store_f32(o, dt, j, v);
            o += pc * eb;
        }
        return RAD_OK;
    }
    /* A scale plane: the pieces' scale rows in sequence. */
    int64_t base = 0;
    for (const SrcTensor* s : src) {
        int64_t sr = 0, scl = 0;
        rad_enc_view((uint32_t)s->shape.size(), s->shape.data(), &sr, &scl);
        if (s->shape.size() == 1) { sr = s->shape[0]; scl = 1; }       /* one scale per row */
        const int64_t lo = std::max(r0, base), hi = std::min(r0 + nr, base + sr);
        if (hi > lo) {
            std::memcpy(o, (const uint8_t*)s->data + (lo - base) * scl * eb,
                        (size_t)((hi - lo) * scl * eb));
            o += (hi - lo) * scl * eb;
        }
        base += sr;
    }
    return RAD_OK;
}

}  /* namespace rad */
