/* startup.cpp -- the tools' seam onto core/plugin and core/build, and the shared CLI plumbing.
 * The tools use the same Registry and Builder as the engine (see startup.h), so they resolve the
 * same program.
 */
#include "build/startup.h"
#include "build/rad_build.h"
#include "lora/declare.h"
#include "format/tunecache.h"
#include "plugin/registry.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace rad {

/* One Registry for the life of the process: Resolved, Band and KernelRow point into plugin
 * static tables, and the Program outlives the call that built it.
 *
 * Never destroyed. A function-local static would be destroyed after the plugins' own static
 * destructors, and its close would call each instance's fini into destroyed plugin state (e.g.
 * libr4d's router erasing from its destroyed global map: a double free). A caller that needs the
 * finis closes the registry while plugins are alive, as Engine's shutdown does. */
Registry& rad_tools_registry() {
    static Registry* r = new Registry();
    return *r;
}

/* ------------------------------------------------------------------ engines in one process */

namespace {
std::mutex                 g_claim_mu;
bool                       g_slot_held[RAD_MAX_SLOTS] = {};
std::map<std::string, int> g_arch_engines;   /* arch plugin -> engines holding it */
}  // namespace

int rad_engine_claim(const std::string& arch_plugin, int n_ranks, int* slot_base) {
    std::lock_guard<std::mutex> lk(g_claim_mu);
    const Plugin* ap = rad_tools_registry().plugin(arch_plugin);
    if (ap && !ap->arch_multi_instance && g_arch_engines[arch_plugin] > 0) {
        RAD_ERR("architecture plugin '%s' keeps one model's state per process and already backs "
                "a running model; it does not export rad_arch_multi_instance, so a second model on "
                "it cannot run beside the first", arch_plugin.c_str());
        return RAD_E_STATE;
    }
    for (int base = 0; base + n_ranks <= RAD_MAX_SLOTS; ++base) {
        bool free = true;
        for (int i = 0; i < n_ranks && free; ++i) free = !g_slot_held[base + i];
        if (!free) continue;
        for (int i = 0; i < n_ranks; ++i) g_slot_held[base + i] = true;
        ++g_arch_engines[arch_plugin];
        *slot_base = base;
        return RAD_OK;
    }
    RAD_ERR("no %d free instance slot(s) in a row: the process holds at most %d ranks across its "
            "models (RAD_MAX_SLOTS)", n_ranks, RAD_MAX_SLOTS);
    return RAD_E_FULL;
}

void rad_engine_release(const std::string& arch_plugin, int slot_base, int n_ranks) {
    std::lock_guard<std::mutex> lk(g_claim_mu);
    for (int i = 0; i < n_ranks && slot_base + i < RAD_MAX_SLOTS; ++i) g_slot_held[slot_base + i] = false;
    auto it = g_arch_engines.find(arch_plugin);
    if (it != g_arch_engines.end() && --it->second <= 0) g_arch_engines.erase(it);
}

/* ------------------------------------------------------------------ the reference-kernel gate */

/* An op is "on the reference" when no band in any domain offers a non-reference kernel
 * (RadPluginReferenceFn). A reference library usually serves the host domain for every op, so
 * only an op with no other implementation is flagged: wherever it is placed, it runs naive code.
 * Counted once per declared instance; `out` lists distinct op names. */
int rad_reference_ops(const Program& p, std::vector<std::string>* out) {
    int n = 0;
    for (size_t i = 1; i < p.ops.size(); ++i) {
        const OpInfo& oi = p.ops[i];
        if (oi.lora_declined) continue;   /* never handed out, so never issued (rad_builder.cpp) */
        bool real = false;
        for (const Band& b : oi.bands)
            for (int d = 0; d < RAD_N_DOMAINS && !real; ++d)
                if (b.dom[d] && b.dom[d].row && !b.dom[d].row->reference)
                    real = true;
        if (real) continue;
        ++n;
        /* Deduplicated by op name. */
        if (out && std::find(out->begin(), out->end(), oi.op) == out->end())
            out->push_back(oi.op);
    }
    return n;
}

int rad_gate_reference_kernels(const Program& p, bool allow, const char* what) {
    std::vector<std::string> ops;
    const int n = rad_reference_ops(p, &ops);
    if (n == 0) return RAD_OK;

    const std::string list = join(ops);
    /* The reference libraries involved, by plugin name. */
    std::vector<std::string> refs;
    for (size_t i = 1; i < p.ops.size(); ++i)
        for (const Band& b : p.ops[i].bands)
            for (int d = 0; d < RAD_N_DOMAINS; ++d)
                if (b.dom[d] && b.dom[d].row && b.dom[d].row->reference &&
                    std::find(refs.begin(), refs.end(), b.dom[d].row->plugin) == refs.end())
                    refs.push_back(b.dom[d].row->plugin);
    const std::string libs = join(refs);

    if (allow) {
        RAD_WARN("%d declared op(s) resolve to a reference implementation (%s) and "
                 "--debug-accept-reference-kernels was passed, so %s continues. This is not a "
                 "production configuration: %zu distinct op(s) run the naive generic "
                 "implementation -- %s", n, libs.c_str(), what, ops.size(), list.c_str());
        return RAD_OK;
    }

    RAD_ERR("%d declared op(s) resolve only to a reference implementation (%s), which runs "
            "orders of magnitude slower; %s refuses them (spec §17).\n"
            "    %zu distinct op(s): %s\n"
            "  Supply a kernel plugin that covers these ops, or pass "
            "--debug-accept-reference-kernels to accept the cost.",
            n, libs.c_str(), what, ops.size(), list.c_str());
    return RAD_E_NOKERNEL;
}

std::vector<std::string> rad_home_dirs(const std::string& home) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= home.size()) {
        size_t e = home.find(':', i);
        if (e == std::string::npos) e = home.size();
        if (e > i) out.push_back(home.substr(i, e - i));
        i = e + 1;
    }
    if (out.empty()) out.push_back(".");
    return out;
}

std::string rad_home_file(const std::string& home, const std::string& rel) {
    const std::vector<std::string> dirs = rad_home_dirs(home);
    for (const std::string& d : dirs) {
        struct stat st {};
        if (stat((d + "/" + rel).c_str(), &st) == 0) return d + "/" + rel;
    }
    return dirs.front() + "/" + rel;
}

/* The homes on the path that have the directory `sub`. */
static std::vector<std::string> home_subdirs(const std::string& home, const char* sub) {
    std::vector<std::string> out;
    for (const std::string& d : rad_home_dirs(home)) {
        struct stat st {};
        const std::string p = d + "/" + sub;
        if (stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) out.push_back(p);
    }
    return out;
}

/* The architecture plugin for (arch_id, quant) among a loaded set, and the kernel libraries it
 * selects from, into *out. */
static int pick_arch(Registry& r, const std::string& home, const std::string& arch_id,
                     const std::string& quant, LoadedPlugins* out, bool imported) {
    /* The selection key is the (arch_id, quant) pair (spec §2.4), matched exactly with no
     * fallback; a mismatched plugin would only fail later at the name map with a worse message. */
    for (const Plugin& p : r.plugins()) {
        if (p.kind == RAD_PLUGIN_KERNEL) {
            out->kernels.push_back({ p.name, p.version });
        } else if (p.kind == RAD_PLUGIN_ARCH && !arch_id.empty() && p.arch_id == arch_id &&
                   p.arch_quant == quant) {
            out->arch_plugin  = p.name;
            out->arch_version = p.version;
            out->arch_id      = p.arch_id;
            out->arch_quant   = p.arch_quant;
        }
    }

    /* An imported quantisation is served by the family's quantised plugin (startup.h). A plugin
     * that declines imports (rad_arch_serves_imports) is left out of the family here: neither
     * counted as a second quantised plugin nor picked as the only one. Block-scaled fp8 reads
     * stock weight by weight, so a family with no plugin claiming it is served the same way: the
     * plugin declares each weight at the encoding it finds (Qwen3.8-Flash-Next-FP8: fp8 routed
     * experts beside bf16 linears, served by qwen4exp_fp8). */
    const bool stock = quant == "fp8_e4m3";
    if ((imported || stock) && !arch_id.empty() && out->arch_plugin.empty()) {
        std::vector<const Plugin*> fam, quantised;
        for (const Plugin& p : r.plugins())
            if (p.kind == RAD_PLUGIN_ARCH && p.arch_id == arch_id && p.arch_serves_imports) {
                fam.push_back(&p);
                if (!p.arch_quant.empty()) quantised.push_back(&p);
            }
        const Plugin* pick = quantised.size() == 1 ? quantised[0]
                           : quantised.empty() && fam.size() == 1 ? fam[0] : nullptr;
        if (!pick && quantised.size() > 1) {
            std::string names;
            for (const Plugin* p : quantised) names += (names.empty() ? "" : ", ") + p->arch_quant;
            RAD_ERR("'%s' is quantised by %s and imported weight by weight, and '%s' has more than "
                    "one quantised plugin (%s) to serve it; none is chosen for it",
                    arch_id.c_str(), quant.c_str(), arch_id.c_str(), names.c_str());
            return RAD_E_NOTFOUND;
        }
        if (pick) {
            out->arch_plugin  = pick->name;
            out->arch_version = pick->version;
            out->arch_id      = pick->arch_id;
            out->arch_quant   = pick->arch_quant;
            RAD_INFO("quantisation '%s' is read weight by weight; '%s' serves it as %s",
                     quant.c_str(), pick->name.c_str(),
                     pick->arch_quant.empty() ? "(unquantised)" : pick->arch_quant.c_str());
        }
    }

    if (out->kernels.empty()) {
        RAD_ERR("no kernel plugin loaded from %s/kernels. Every weight's layout comes from the "
                "kernel that will read it, so there is nothing to lay a container out for.",
                home.c_str());
        return RAD_E_NOKERNEL;
    }
    if (!arch_id.empty() && out->arch_plugin.empty()) {
        /* No match, no model (spec §1). List what is installed with its quantisation, to tell an
         * unsupported architecture from one supported only at another weight format. */
        std::string have, same;
        for (const Plugin& p : r.plugins()) {
            if (p.kind != RAD_PLUGIN_ARCH) continue;
            const std::string q = p.arch_quant.empty() ? "(unquantised)" : p.arch_quant;
            if (!have.empty()) have += ", ";
            have += p.arch_id + " @ " + q;
            if (p.arch_id == arch_id) { if (!same.empty()) same += ", "; same += q; }
        }
        if (!same.empty())
            RAD_ERR("no architecture plugin claims '%s' at quantisation '%s'. '%s' is served at: "
                    "%s. Either the container's quantisation descriptor is not one this project "
                    "has a plugin for, or the container was written before the descriptor was "
                    "recorded -- rad-info prints it.",
                    arch_id.c_str(), quant.empty() ? "(unquantised)" : quant.c_str(),
                    arch_id.c_str(), same.c_str());
        else
            RAD_ERR("no architecture plugin claims '%s'. %s/architectures offers: %s",
                    arch_id.c_str(), home.c_str(), have.empty() ? "(none)" : have.c_str());
        return RAD_E_NOTFOUND;
    }
    return RAD_OK;
}

int rad_tools_load_plugins(const std::string& home, const std::vector<std::string>& hierarchy,
                           const std::string& arch_id, const std::string& quant,
                           LoadedPlugins* out, bool imported) {
    if (!out) return RAD_E_INVAL;
    *out = LoadedPlugins{};

    Registry& r = rad_tools_registry();

    /* One plugin set serves every engine in the process (registry.h, Sharing): the first load
     * reads the homes, a later one finds the set loaded and only picks its architecture. The home
     * and the hierarchy are server-level settings, so every engine asks for the same set; a
     * different one is refused rather than served from the wrong libraries. */
    static std::string loaded_home;
    static std::vector<std::string> loaded_hierarchy;
    if (!r.plugins().empty()) {
        if (home != loaded_home || hierarchy != loaded_hierarchy) {
            RAD_ERR("the plugin set is already loaded from %s; every model in a process uses the "
                    "same radiance-home and kernels", loaded_home.c_str());
            return RAD_E_STATE;
        }
        return pick_arch(r, home, arch_id, quant, out, imported);
    }
    loaded_home = home;
    loaded_hierarchy = hierarchy;

    /* Kernels first, in hierarchy order, then architectures (unordered: no two may claim the same
     * id). A home need not have every directory, but kernels/ and architectures/ must each exist
     * somewhere on the path, or the scan reports the first home's as missing. */
    const std::string first = rad_home_dirs(home).front();
    std::vector<std::string> kdirs = home_subdirs(home, "kernels");
    if (kdirs.empty()) kdirs.push_back(first + "/kernels");
    int s = r.load_dirs(kdirs, hierarchy);
    if (s < 0) return s;

    std::vector<std::string> adirs = home_subdirs(home, "architectures");
    if (adirs.empty()) adirs.push_back(first + "/architectures");
    s = r.load_dirs(adirs, {});
    if (s < 0) return s;

    /* Quantisers: rad-convert drives them, and the oracle decodes through them for schemes the
     * core does not define. Optional. */
    const std::vector<std::string> qdirs = home_subdirs(home, "quantizers");
    if (!qdirs.empty()) {
        s = r.load_dirs(qdirs, {});
        if (s < 0) return s;
    }

    RAD_TRY(r.check_complete());

    /* The tuning cache, once the kernel set is known. A missing file leaves every shape on its
     * plugin's first variant. Logged when it has rows; --debug-graph shows the per-shape
     * source. */
    const std::string tune = rad_home_file(home, TuneCache::rel_path(rad_machine_key()));
    const std::string tune_home = tune.substr(0, tune.size() - TuneCache::rel_path(rad_machine_key()).size() - 1);
    const int n_tuned = r.load_tune(tune_home);
    if (n_tuned > 0) RAD_INFO("tune cache: %d row(s) from %s", n_tuned, tune.c_str());

    return pick_arch(r, home, arch_id, quant, out, imported);
}

/* Optional export. Without it, or on failure, every field is zero and the caller uses its own
 * defaults. A failed probe is not an error: the plugin can still refuse in declare, where it has
 * the whole configuration and can say why. */
int rad_tools_probe(const LoadedPlugins& lp, const RadModelMeta& meta, RadArchProbe* out,
                    int64_t gen_max_pixels, int64_t gen_max_refs) {
    if (!out) return RAD_E_INVAL;
    *out = RadArchProbe{};
    out->gen_max_pixels = gen_max_pixels;
    out->gen_max_refs = gen_max_refs;
    const Plugin* ap = rad_tools_registry().plugin(lp.arch_plugin);
    if (!ap || !ap->arch_probe) return RAD_E_UNSUPPORTED;
    const int s = ap->arch_probe(&meta, out);
    if (s < 0) {
        RAD_WARN("%s: rad_arch_probe returned %s; the core will use its own defaults",
                 lp.arch_plugin.c_str(), rad_strerror(s));
        *out = RadArchProbe{};
        out->gen_max_pixels = gen_max_pixels;
        out->gen_max_refs = gen_max_refs;
        return s;
    }
    return RAD_OK;
}

const RadChatFormat* rad_tools_chat_format(const LoadedPlugins& lp, const RadModelMeta& meta) {
    const Plugin* ap = rad_tools_registry().plugin(lp.arch_plugin);
    if (!ap || !ap->arch_chat_format) return nullptr;
    return ap->arch_chat_format(&meta);
}

int rad_tools_text_prepare(const LoadedPlugins& lp, int field, const std::string& in,
                           std::string* out) {
    const Plugin* ap = rad_tools_registry().plugin(lp.arch_plugin);
    if (!ap || !ap->arch_text_prepare) {
        *out = in;
        return RAD_OK;
    }
    std::string buf(in.size() * 2 + 64, '\0');
    for (int tries = 0; tries < 2; ++tries) {
        const int64_t n = ap->arch_text_prepare(field, in.c_str(), &buf[0], (int64_t)buf.size());
        if (n < 0) {
            /* The plugin's reason, when it left one in the (zeroed) buffer. */
            buf.resize(std::strlen(buf.c_str()));
            *out = std::move(buf);
            return (int)n;
        }
        if (n < (int64_t)buf.size()) {
            buf.resize((size_t)n);
            *out = std::move(buf);
            return RAD_OK;
        }
        buf.assign((size_t)n + 1, '\0');
    }
    return RAD_E_STATE;
}

VideoPositionsFn rad_tools_video_positions(const LoadedPlugins& lp) {
    const Plugin* ap = rad_tools_registry().plugin(lp.arch_plugin);
    return ap ? ap->arch_video_positions : nullptr;
}

VideoPackFn rad_tools_video_pack(const LoadedPlugins& lp) {
    const Plugin* ap = rad_tools_registry().plugin(lp.arch_plugin);
    return ap ? ap->arch_video_pack : nullptr;
}

EncodeGeomFn rad_tools_encode_geom(const LoadedPlugins& lp) {
    const Plugin* ap = rad_tools_registry().plugin(lp.arch_plugin);
    return ap ? ap->arch_encode_geom : nullptr;
}

const RadAudioFormat* rad_tools_audio_format(const LoadedPlugins& lp, const RadModelMeta& meta) {
    const Plugin* ap = rad_tools_registry().plugin(lp.arch_plugin);
    if (!ap || !ap->arch_audio_format) return nullptr;
    const RadAudioFormat* f = ap->arch_audio_format(&meta);
    if (!f) return nullptr;
    /* The plugin's struct may be older than ours: it is copied into a zeroed one up to the size
     * it states, so a field it predates reads as null (rad_builder.h). */
    static thread_local RadAudioFormat copy;
    copy = RadAudioFormat{};
    const size_t n = f->struct_size < sizeof copy ? f->struct_size : sizeof copy;
    std::memcpy(&copy, f, n);
    copy.struct_size = (uint32_t)sizeof copy;
    return &copy;
}

WeightSourceFn rad_container_sources(const RadFile& f) {
    return [&f](const std::string& name, const Program&, WeightSource* out) {
        const RadFileEntry* e = f.find(name);
        if (!e) return RAD_E_NOTFOUND;
        out->enc = f.encoding(*e);
        out->rank = e->rank;
        for (int i = 0; i < RAD_MAX_RANK; ++i) out->shape[i] = e->shape[i];
        return RAD_OK;
    };
}

WeightSourceFn CheckpointSources::fn() {
    return [this](const std::string& name, const Program& prog, WeightSource* out) {
        return answer(name, prog, out);
    };
}

int CheckpointSources::answer(const std::string& name, const Program& prog, WeightSource* out) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = got_.find(name);
    if (it == got_.end()) {
        CkptWeight w;
        std::string why;
        const int st = ckpt_resolve(ck_, prog, name, false, &w, &why);
        /* Not-found is not cached, since declare may map the name later, but its reason is kept
         * for the loader to report. */
        if (st == RAD_E_NOTFOUND) {
            missing_[name] = why;
            return st;
        }
        missing_.erase(name);
        if (st < 0) {
            refused_[name] = why;
            return st;
        }
        it = got_.emplace(name, std::move(w)).first;
    }
    const CkptWeight& w = it->second;
    if (!w.trivial) {
        refused_[name] = w.why;
        return RAD_E_UNSUPPORTED;
    }
    out->enc = w.enc;
    out->rank = w.rank;
    for (int i = 0; i < RAD_MAX_RANK; ++i) out->shape[i] = w.shape[i];
    return RAD_OK;
}

const CkptWeight* CheckpointSources::find(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = got_.find(name);
    return it == got_.end() || !it->second.trivial ? nullptr : &it->second;
}

std::string CheckpointSources::missing(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = missing_.find(name);
    return it == missing_.end() ? std::string() : it->second;
}

std::vector<std::string> CheckpointSources::refused() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> out;
    for (const auto& [n, why] : refused_) out.push_back(n + ": " + why);
    return out;
}

int rad_tools_declare(const LoadedPlugins& lp, const RadModelMeta& meta, const RadBuildCtx& ctx,
                      Program* out, const std::function<int(RadBuilder*, int64_t)>& also,
                      const Program* ref, const WeightSourceFn& sources, bool instances,
                      const LoraDeclare* lora) {
    if (!out) return RAD_E_INVAL;

    Registry& r = lp.registry ? *lp.registry : rad_tools_registry();
    const Plugin* ap = r.plugin(lp.arch_plugin);
    if (!ap || !ap->arch_declare) {
        RAD_ERR("architecture plugin '%s' is not loaded or exports no rad_arch_declare",
                lp.arch_plugin.c_str());
        return RAD_E_NOTFOUND;
    }

    Builder b(r, meta, ctx);
    if (ref) b.set_reference(ref);
    if (sources) b.set_sources(sources);
    b.set_instances(instances);
    const bool with_lora = lora && lora->cfg && lora->cfg->enabled();
    if (with_lora) b.set_lora_targets(&lora->targets);

    int s = ap->arch_declare(&b, &meta, &ctx);
    if (s < 0) {
        RAD_ERR("%s: rad_arch_declare returned %s", lp.arch_plugin.c_str(), rad_strerror(s));
        for (const auto& e : b.errors()) RAD_ERR("    %s", e.c_str());
        return s;
    }

    /* Runtime LoRA, after the plugin and before the sampler: the sites read the plugin's finished
     * graph, and the appended ops go through the same resolution as every other op. */
    if (with_lora) {
        b.set_lora_targets(nullptr);
        s = lora_declare(b, *lora);
        if (s < 0) {
            for (const auto& e : b.errors()) RAD_ERR("    %s", e.c_str());
            return s;
        }
    }

    /* The sampler declares into the same builder before finish(). An unresolved sampler op is not
     * a declare failure (a deployment that never sends `typical_p` needs no typical kernel), so a
     * negative return here is structural. Skipped for embedding programs and generators, which
     * sample through no token sampler (an audio generator samples its frames itself). */
    if (also && b.program().pooled_buf == RAD_NULL_HANDLE && !b.program().is_pass_driven()) {
        /* This rank's vocabulary width, taken from the declared logits buffer rather than
         * re-derived from n_vocab and world_size, which would break when a plugin does not split
         * the vocabulary (arch/common/rad_arch.h does not). A wrong width does not crash; it
         * samples the wrong tokens. */
        int64_t lw = 0;
        if (b.program().logits_buf != RAD_NULL_HANDLE &&
            (size_t)b.program().logits_buf < b.program().buffers.size()) {
            const RadBufDecl& d = b.program().buffers[b.program().logits_buf].decl;
            if (d.rank > 0) lw = d.shape[d.rank - 1];
        }
        s = also(&b, lw);
        if (s < 0) {
            for (const auto& e : b.errors()) RAD_ERR("    %s", e.c_str());
            return s;
        }
    }

    s = b.finish();
    /* Declare always completes; every structural error is reported (spec §3.1). */
    for (const auto& e : b.errors()) RAD_ERR("    %s", e.c_str());
    if (s < 0) return s;
    if (b.status() < 0) return b.status();

    *out = std::move(b.program());
    return RAD_OK;
}

}  /* namespace rad */
