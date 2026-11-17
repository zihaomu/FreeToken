/* startup.h -- steps 1 through 4 of spec §1's startup sequence, shared by the engine and the
 * tools.
 *
 *   1. Load kernel plugins from $RADIANCE_HOME/kernels/, in the configured hierarchy order.
 *   3. Look up the architecture id in $RADIANCE_HOME/architectures/.
 *   4. Declare.
 *
 * In core because rad-convert, rad-check and rad-tune are the engine's startup with a different
 * last step (spec §4.4, §15, §17). A second implementation could disagree on hierarchy order,
 * which decides the layout baked into a container.
 */
#pragma once
#include "rad_core.h"
#include "format/radfile.h"
#include "format/checkpoint.h"
#include "plugin/registry.h"
#include "build/rad_build.h"

#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace rad {

/* A loaded plugin, as a tool names it. */
struct PluginId { std::string name, version; };

struct LoadedPlugins {
    /* Kernel plugins in hierarchy order, which decides selection. */
    std::vector<PluginId>       kernels;
    std::string                 arch_plugin;
    std::string                 arch_version;
    std::string                 arch_id;
    /* The quantisation half of the selection key, "" for an unquantised container. */
    std::string                 arch_quant;
    /* The registry declare selects through: an engine's own share of the process set
     * (Registry::share), or null for the process registry itself, as the tools use. */
    Registry*                   registry = nullptr;

    /* `name@version`, space separated, in hierarchy order; what a benchmark records as the
     * libraries measured. */
    std::string libraries() const {
        std::string s;
        for (const auto& p : kernels) {
            if (!s.empty()) s += ' ';
            s += p.name + "@" + (p.version.empty() ? "0" : p.version);
        }
        return s;
    }
};

/* $RADIANCE_HOME is a ':'-separated search path, first home first, so a directory of one's own
 * plugins can precede an installation. Each home may hold kernels/, architectures/ and
 * quantizers/; a plugin earlier on the path shadows one with the same file name later. Non-plugin
 * files (the tuning cache, rad-kbench fixtures) are read from the first home that has them and
 * written to the first home. */
std::vector<std::string> rad_home_dirs(const std::string& home);
/* `<home>/<rel>` for the first home on the path that has it; the first home's when none does. */
std::string rad_home_file(const std::string& home, const std::string& rel);

/* Loads into the process-lifetime registry (rad_tools_registry).
 *
 * `arch_id` and `quant` together are the selection key (spec §2.4, rad_builder.h): a family has
 * one plugin per weight format, all answering the same architecture id. `quant` is the container's
 * descriptor (RadModelMeta::quant, RadFileHeader::quant); "" is unquantised and matches only a
 * plugin that says "".
 *
 * `imported`: the checkpoint's quantisation is imported weight by weight (Checkpoint::imports), so
 * its descriptor names the producing tool and no plugin claims it. The family's quantised plugin
 * serves it: the one plugin claiming a descriptor, or the only plugin if none does. Two quantised
 * plugins are refused by name. A plugin whose rad_arch_serves_imports returns 0 is left out of
 * both counts. Block-scaled fp8 (`fp8_e4m3`) that no plugin of the family claims is served the
 * same way, since its weights read stock. `out->arch_quant` is the chosen descriptor, which a
 * container converted from it records. */
int rad_tools_load_plugins(const std::string& radiance_home,
                           const std::vector<std::string>& hierarchy,
                           const std::string& arch_id,
                           const std::string& quant,
                           LoadedPlugins* out,
                           bool imported = false);

/* The architecture plugin's rad_arch_probe answers, given before declare. */
/* `gen_max_pixels`, `gen_max_refs`: the probe's inputs (RadArchProbe), the deployment's
 * --image-gen-max-pixels and --image-gen-max-refs; 0 the model's own. */
int rad_tools_probe(const LoadedPlugins& plugins, const RadModelMeta& meta, RadArchProbe* out,
                    int64_t gen_max_pixels = 0, int64_t gen_max_refs = 0);

/* The architecture plugin's declared reply format (rad_arch_chat_format), or null when it
 * declares none and the format is to be derived from the chat template. */
const RadChatFormat* rad_tools_chat_format(const LoadedPlugins& plugins, const RadModelMeta& meta);

/* The architecture plugin's declared audio format (rad_arch_audio_format), or null when it
 * declares none -- a model the speech routes do not serve. A format whose struct_size is too
 * small to hold a field reads that field as null. */
const RadAudioFormat* rad_tools_audio_format(const LoadedPlugins& plugins, const RadModelMeta& meta);

/* A request's text rewritten by the architecture plugin for the audio generator's prompt
 * (rad_arch_text_prepare): RAD_OK with the result in `*out` (the text unchanged when the plugin
 * states no rewrite), or the plugin's negative status. */
int rad_tools_text_prepare(const LoadedPlugins& plugins, int field, const std::string& in,
                           std::string* out);   /* on a refusal, `out` is its reason or "" */

/* The architecture plugin's video positions (rad_arch_video_positions), or null when it exports
 * none. */
using VideoPositionsFn = int64_t (*)(int, const RadVideoGeom*, float*, int64_t);
VideoPositionsFn rad_tools_video_positions(const LoadedPlugins& plugins);

/* Its rad_arch_video_pack and rad_arch_encode_geom, null when it does not export them. */
using VideoPackFn = int64_t (*)(const RadVideoGeom*, const int32_t*, int32_t*, int32_t*, int32_t*,
                                int32_t*, float*, int64_t);
VideoPackFn rad_tools_video_pack(const LoadedPlugins& plugins);
using EncodeGeomFn = int64_t (*)(int32_t, const int64_t*, int32_t*, int64_t);
EncodeGeomFn rad_tools_encode_geom(const LoadedPlugins& plugins);

/* Run declare.
 *
 * `also` declares into the same builder after the architecture plugin, which is how the sampler
 * chain joins the Program: its ops resolve through the same hierarchy, so a missing sampler kernel
 * is in the same miss list rather than found at the first request (spec §3.1, §13). Empty for the
 * tools, which have no sampler.
 *
 * `ref` makes this a sizing declare (RadBuildCtx::shape_probe, Builder::set_reference): the plugin
 * declares against `ref`'s bands, nothing is selected or initialised, and the result is used only
 * for its buffers.
 *
 * `sources` answers rad_weight_encoding (Builder::set_sources): container entries, checkpoint
 * tensors, or a recipe. Empty: every weight is plain in its declared dtype.
 *
 * `instances` false selects every kernel and initialises none (Builder::set_instances), for
 * callers that launch nothing, such as rad-convert. */
int rad_tools_declare(const LoadedPlugins& plugins, const RadModelMeta& meta,
                      const RadBuildCtx& ctx, Program* out,
                      const std::function<int(RadBuilder*, int64_t logits_width)>& also = {},
                      const Program* ref = nullptr, const WeightSourceFn& sources = {},
                      bool instances = true, const struct LoraDeclare* lora = nullptr);

/* A container's entries as the source of every declaration's encoding. `f` must outlive the
 * declare. */
WeightSourceFn rad_container_sources(const RadFile& f);

/* A checkpoint served directly as the source (spec §4.3). Each declared weight is resolved
 * through the name map once and kept for the loader, which reads the same tensors. Locked, since
 * every rank declares concurrently.
 *
 * Only trivial encodings are answered (bf16, f16, f32, block-scaled FP8) because nothing is
 * quantised at load. Anything else is recorded with its reason and answered RAD_E_UNSUPPORTED;
 * refused() lists them after declare, and the remedy is a container. */
class CheckpointSources {
public:
    explicit CheckpointSources(Checkpoint& ck) : ck_(ck) {}
    WeightSourceFn fn();
    /* After declare: what `name` resolved to, or null. */
    const CkptWeight* find(const std::string& name) const;
    /* Why `name` resolved to nothing, as the name map last answered it; empty if it did. */
    std::string missing(const std::string& name) const;
    /* Every weight that is not served as it stands, "name: why", sorted. */
    std::vector<std::string> refused() const;

private:
    int answer(const std::string& name, const Program& prog, WeightSource* out);
    Checkpoint& ck_;
    mutable std::mutex mu_;
    std::map<std::string, CkptWeight> got_;
    std::map<std::string, std::string> refused_;
    std::map<std::string, std::string> missing_;
};

/* ------------------------------------------------------------------ the reference-kernel gate */
/* A reference library (RadPluginReferenceFn) is both the slow fallback for an unsupported model
 * and the oracle rad-kbench measures against (spec §17). As a silent fallback it runs orders of
 * magnitude below the machine with no visible sign, so resolving to one is fatal unless
 * --debug-accept-reference-kernels is passed. */

int rad_reference_ops(const Program& p, std::vector<std::string>* out);

/* Report and refuse, or report and continue when `allow`. `what` names the caller ("serve",
 * "the container's layout") so the message states the consequence. */
int rad_gate_reference_kernels(const Program& p, bool allow, const char* what);

Registry& rad_tools_registry();

/* Engines in one process (docs/MULTIMODEL.md §3.3). Each holds, while it lives, a contiguous block
 * of `n_ranks` instance slots, whose first is its RadBuildCtx::slot_base, and a claim on its
 * architecture plugin. A plugin that does not export rad_arch_multi_instance backs one engine at a
 * time: a second claim on it is refused by name. RAD_OK with *slot_base set, or a negative status
 * after a RAD_ERR line saying why. Release returns the slots and the claim. */
int  rad_engine_claim(const std::string& arch_plugin, int n_ranks, int* slot_base);
void rad_engine_release(const std::string& arch_plugin, int slot_base, int n_ranks);

}  /* namespace rad */
