/* cvec.h -- control vectors: one GGUF direction per layer, projected out of the residual stream.
 *
 * The format is llama.cpp's `controlvector` GGUF: `direction.<l>` tensors of f32, n_embd wide. The
 * operation is llama.cpp's `--cvec-mode project`
 *
 *     h <- h - s_l (h . v_l) v_l          v_l = d_l / |d_l|,  s_l = |d_l| * package scale
 *
 * applied by an architecture plugin that declares RadArchProbe::control_vector_ok (qwen4exp_fp8
 * issues `hc_project` on every stream of its gated residual). The layer range and scale come from
 * the package's JSON beside the file; without one, every direction the file carries steers at
 * scale 1.
 *
 * The vector shipped in data/vectors/ is a refusal-direction projection for Qwen3.8-Flash-Next:
 * with it on, the model declines far fewer requests. It is loaded only with --control-vector, and
 * each request can name its own choice.
 *
 * A file is refused unless
 *
 *   - `general.architecture` is `controlvector`,
 *   - `controlvector.model_hint`, when present, is this model's architecture id,
 *   - every tensor is `direction.<l>`, F32, one dimension, exactly the model's `n_embd`, and
 *     1 <= l < the model's layer count,
 *   - the layer range lies inside the model and every layer in it has a direction.
 */
#pragma once
#include "rad_internal.h"

#include <string>
#include <vector>

namespace rad {

/* What a loaded vector keeps on the host, before it is uploaded per rank. `dir` is
 * [n_layers * n_embd] f32, unit rows, row 0 unused (the format has no direction for layer 0);
 * `scales[l]` is |d_l| * user scale, or 0 where the layer is outside the loaded range. */
struct ControlVector {
    std::vector<float> dir;
    std::vector<float> scales;
    int64_t n_embd = 0;
    int64_t n_layers = 0;
    int64_t n_dirs = 0;               /* directions the file carried (47 for the published one) */
    int     first = -1, last = -1;    /* inclusive layer range */
    float   user_scale = 1.0f;
    std::string path;

    bool loaded() const { return n_layers > 0 && n_embd > 0; }
    /* Whether layer `l` is steered, i.e. a kernel has something to project. Mirror of the loader's
     * range check, used by the architecture plugin at declare. */
    bool covers(int64_t l) const {
        return loaded() && l >= first && l <= last && l < (int64_t) scales.size() &&
               scales[(size_t) l] != 0.0f;
    }
};

/* Read and validate `path` against a model of `n_embd` and `n_model_layers`. `path`'s own
 * `<stem>.json` supplies `layer_range` and `scale`; one that is not JSON or names a mode other than
 * `project` is refused. Without it, every layer the file carries steers at scale 1. */
int control_vector_load(const std::string& path, const std::string& arch, int64_t n_embd,
                        int64_t n_model_layers, ControlVector* out, std::string* err);

/* Where a vector for `model_path` is looked for:
 * the container's sidecar, then one generic name in the container's directory, then each
 * $RADIANCE_HOME's `vectors/` directory. Mirrors gen_sidecar_path's naming so an operator who
 * already knows the generation-config convention knows this one. */
std::string control_vector_sidecar_path(const std::string& model_path);

/* Sets *found to the first candidate that exists and is non-empty, or to "" when none does. *tried receives the candidates, for the log line that says why the feature is
 * unavailable on this deployment. Discovery is a filesystem test only; whether the file is
 * compatible is control_vector_load's answer. */
int control_vector_discover(const std::string& model_path, const std::string& arch,
                            const std::string& radiance_home, std::string* found,
                            std::string* tried);

/* Build the ABI view for one rank. `dir_device` is that rank's [n_layers, n_embd] f32 copy. */
RadControlVector control_vector_view(const ControlVector& cv, const float* dir_device);

/* Upload the host table to a fresh device buffer on rank `rank`, or null on failure. The caller
 * owns the buffer and frees it with rad_dev_free. */
const float* control_vector_upload(const ControlVector& cv, int rank);

/* The layer salt a cache uses so that blocks and linear-state snapshots computed with the mode on
 * are never handed to a request with it off (or to another vector's). 0 for "off", which is also
 * the value the engine uses with no vector loaded, so a deployment without the feature hashes
 * exactly as it did before it existed. */
uint64_t control_vector_salt(const ControlVector& cv);

}  /* namespace rad */
