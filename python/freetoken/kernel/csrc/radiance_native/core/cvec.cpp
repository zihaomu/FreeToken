/* cvec.cpp -- loading and validating a GGUF control vector (core/cvec.h). The reader is
 * core/text/gguf.cpp, which maps the file and sizes its tensors. */
#include "cvec.h"

#include "text/gguf.h"

#include <algorithm>
#include <cmath>
#include <fstream>

#include "util/json.h"

namespace rad {
namespace {

/* FNV-1a over the table, so two different vectors never share a cache partition. */
uint64_t hash_table(const ControlVector& cv) {
    uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*) p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    mix(cv.dir.data(), cv.dir.size() * sizeof(float));
    mix(cv.scales.data(), cv.scales.size() * sizeof(float));
    return h ? h : 1ull;
}

}  /* namespace */

std::string control_vector_sidecar_path(const std::string& model_path) {
    const size_t slash = model_path.find_last_of("/\\");
    const size_t dot   = model_path.find_last_of('.');
    const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash);
    return (has_ext ? model_path.substr(0, dot) : model_path) + ".control-vector.gguf";
}

namespace {

bool file_present(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    return f.tellg() > 0;
}

}  /* namespace */

int control_vector_discover(const std::string& model_path, const std::string& arch,
                            const std::string& radiance_home, std::string* found,
                            std::string* tried) {
    if (!found) return RAD_E_INVAL;
    found->clear();
    if (tried) tried->clear();

    std::vector<std::string> cand;
    const size_t slash = model_path.find_last_of("/\\");
    const std::string dir = slash == std::string::npos ? std::string(".") : model_path.substr(0, slash);
    if (!model_path.empty()) {
        cand.push_back(control_vector_sidecar_path(model_path));
        /* One file for a directory of containers: each quant of the family shares the directions,
         * since the file names the architecture and its width, not the quantisation. */
        cand.push_back(dir + "/control-vector.gguf");
    }
    if (!arch.empty() && !radiance_home.empty()) {
        size_t at = 0;
        while (at <= radiance_home.size()) {
            const size_t colon = radiance_home.find(':', at);
            const std::string home = radiance_home.substr(
                at, colon == std::string::npos ? std::string::npos : colon - at);
            if (!home.empty())
                cand.push_back(home + "/vectors/" + arch + "-control-vector.gguf");
            if (colon == std::string::npos) break;
            at = colon + 1;
        }
    }

    for (const std::string& p : cand) {
        if (tried) {
            if (!tried->empty()) *tried += ", ";
            *tried += p;
        }
        if (file_present(p)) { *found = p; return RAD_OK; }
    }
    return RAD_OK;
}


/* The package's own JSON beside the vector (`<stem>.json`), the only place the operation's
 * parameters are stated: `{"mode": "project", "scale": 1.0, "layer_range": [4, 44]}` for the
 * published Flash-Next vector. Absent, the defaults apply. A file that is not JSON or names a mode
 * other than `project` is refused: the directions were tuned for that operation. */
namespace {

struct PackageJson {
    int   first = 0, last = -1;   /* -1: not stated */
    float scale = 1.0f;
};

int read_package_json(const std::string& gguf_path, PackageJson* out, std::string* why) {
    *out = PackageJson{};
    std::string stem = gguf_path;
    const size_t dot = stem.find_last_of('.');
    const size_t slash = stem.find_last_of("/\\");
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
        stem = stem.substr(0, dot);
    const std::string path = stem + ".json";
    std::ifstream f(path);
    if (!f) return RAD_OK;
    try {
        const Json j = Json::parse(f);
        if (const auto m = j.find("mode"); m != j.end() && m->is_string() &&
                                          m->get<std::string>() != "project") {
            *why = "control vector " + path + ": mode '" + m->get<std::string>() +
                   "'; only 'project' is implemented";
            return RAD_E_UNSUPPORTED;
        }
        if (const auto r = j.find("layer_range"); r != j.end() && r->is_array() && r->size() == 2 &&
                                                  (*r)[0].is_number_integer() &&
                                                  (*r)[1].is_number_integer()) {
            out->first = (*r)[0].get<int>();
            out->last  = (*r)[1].get<int>();
        }
        if (const auto s = j.find("scale"); s != j.end() && s->is_number())
            out->scale = (float) s->get<double>();
    } catch (const std::exception& e) {
        *why = "control vector " + path + ": not readable JSON (" + e.what() + ")";
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

}  /* namespace */

int control_vector_load(const std::string& path, const std::string& arch, int64_t n_embd,
                        int64_t n_model_layers, ControlVector* out, std::string* err) {
    const auto fail = [err](const std::string& why) {
        if (err) *err = why;
        return RAD_E_INVAL;
    };
    if (!out) return RAD_E_INVAL;
    *out = ControlVector{};

    GgufFile f;
    const int rc = f.open(path.c_str());
    if (rc != RAD_OK)
        return fail("control vector " + path + ": cannot read it as a GGUF container");

    std::string a;
    if (!f.get_str("general.architecture", &a) || a != "controlvector")
        return fail("control vector " + path + ": general.architecture is '" + a +
                    "', not 'controlvector' -- this is not a control vector");

    /* The hint is optional in llama.cpp's generator, but when a file states one it must be this
     * model's architecture: a direction is a basis vector inferred from that architecture's
     * residual stream, and nothing else. */
    std::string hint;
    if (f.get_str("controlvector.model_hint", &hint) && !hint.empty() && !arch.empty() &&
        hint != arch)
        return fail("control vector " + path + ": the vector is for architecture '" + hint +
                    "', this container is '" + arch + "' -- a direction from another model's "
                    "residual basis would move hidden states along a meaningless axis");

    /* The file's directions: direction.<l>, one dimensional, F32, n_embd wide. Anything else is
     * refused rather than skipped, because a quietly missing layer is a differently steered model. */
    int64_t n_dir = 0, dir_first = -1, dir_last = -1;
    for (const GgufTensor& t : f.tensors()) {
        if (t.name.rfind("direction.", 0) != 0)
            return fail("control vector " + path + ": tensor '" + t.name +
                        "' is not a direction.<layer> entry");
        int64_t layer = 0;
        try {
            layer = std::stoll(t.name.substr(10));
        } catch (...) {
            return fail("control vector " + path + ": tensor name '" + t.name +
                        "' does not end in a layer number");
        }
        /* 1 .. n_layers-1: the table is indexed by layer (row 0 is unused, the format carrying no
         * direction for it), so direction.<n_layers> would write one row past it. */
        if (layer < 1 || layer >= n_model_layers)
            return fail("control vector " + path + ": direction." + std::to_string(layer) +
                        " is outside this model's " + std::to_string(n_model_layers) + " layers");
        if (t.n_dims != 1 || t.ne[0] != n_embd)
            return fail("control vector " + path + ": " + t.name + " is shape [" +
                        std::to_string(t.ne[0]) + "] with rank " + std::to_string(t.n_dims) +
                        "; this model's residual streams are " + std::to_string(n_embd) +
                        " wide and a direction is one row of that");
        if (gguf_to_rad_dtype(t.ggml_type) != RAD_F32)
            return fail("control vector " + path + ": " + t.name +
                        " is not F32 -- only an f32 direction is read");
        if (dir_first < 0 || layer < dir_first) dir_first = layer;
        if (layer > dir_last)                  dir_last  = layer;
        ++n_dir;
    }
    if (n_dir == 0) return fail("control vector " + path + ": the file holds no directions");

    /* The package's parameters: its JSON, else every direction the file carries at scale 1, as
     * llama.cpp applies a control vector given no layer range. */
    PackageJson pj;
    std::string pj_why;
    if (const int prc = read_package_json(path, &pj, &pj_why); prc != RAD_OK) return fail(pj_why);
    int first = pj.first, last = pj.last;
    float scale = pj.scale;
    if (last < 0) { first = (int) dir_first; last = (int) dir_last; }
    if (first < 1 || last < first)
        return fail("control vector " + path + ": the layer range " + std::to_string(first) + ":" +
                    std::to_string(last) + " is empty");
    if (last >= n_model_layers)
        return fail("control vector " + path + ": the layer range " + std::to_string(first) + ":" +
                    std::to_string(last) + " runs past this model's " +
                    std::to_string(n_model_layers) + " layers");
    if (!(scale > 0.0f) || !std::isfinite(scale))
        return fail("control vector " + path + ": its scale is not a positive number");

    out->n_embd = n_embd;
    /* Indexed by layer: rows 0..n_layers-1. Row 0 stays zero, the format carrying no direction
     * for layer 0, and every row a model can name exists. */
    out->n_layers = n_model_layers;
    out->n_dirs = n_dir;
    out->first = first;
    out->last = last;
    out->user_scale = scale;
    out->path = path;
    out->dir.assign((size_t) (out->n_layers * n_embd), 0.0f);
    out->scales.assign((size_t) out->n_layers, 0.0f);

    double worst = 0.0;
    for (int64_t l = 1; l <= n_model_layers; ++l) {
        const GgufTensor* t = f.tensor("direction." + std::to_string(l));
        if (!t) continue;
        const float* d = (const float*) f.tensor_data(*t);
        if (!d) return fail("control vector " + path + ": " + t->name + " lies outside the file");

        double nn = 0.0;
        for (int64_t i = 0; i < n_embd; ++i) nn += (double) d[i] * (double) d[i];
        const double norm = std::sqrt(nn);
        if (!(norm > 0.0) || !std::isfinite(norm))
            return fail("control vector " + path + ": " + t->name + " has no direction");

        float* dst = &out->dir[(size_t) (l * n_embd)];
        const float inv = (float) (1.0 / norm);
        for (int64_t i = 0; i < n_embd; ++i) dst[i] = d[i] * inv;
        /* Zero outside the range: a layer is steered exactly when its scale is nonzero
         * (abi/rad_builder.h, RadControlVector), and a file carries directions for layers the
         * package's JSON leaves out. The row is kept so a range change needs no new upload. */
        out->scales[(size_t) l] = (l >= first && l <= last) ? (float) norm * scale : 0.0f;

        /* A layer inside the range with no direction would silently be unsteered. */
        if (l >= first && l <= last) worst = std::max(worst, std::fabs(norm - 1.0));
    }
    for (int64_t l = first; l <= last; ++l)
        if (out->scales[(size_t) l] == 0.0f)
            return fail("control vector " + path + ": the range " + std::to_string(first) + ":" +
                        std::to_string(last) + " includes layer " + std::to_string(l) +
                        ", which has no direction in the file");

    if (worst > 1e-2)
        RAD_WARN("control vector: %s: the directions are not unit-normalised (worst |norm - 1| "
                 "= %.4f); each is divided by its own norm, so the operation is unchanged",
                 path.c_str(), worst);
    return RAD_OK;
}

RadControlVector control_vector_view(const ControlVector& cv, const float* dir_device) {
    RadControlVector v{};
    v.dir = dir_device;
    v.scales = cv.scales.data();
    v.n_embd = cv.n_embd;
    v.n_layers = cv.n_layers;
    return v;
}

const float* control_vector_upload(const ControlVector& cv, int rank) {
    if (!cv.loaded()) return nullptr;
    const int64_t bytes = (int64_t) cv.dir.size() * 4;
    void* d = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
    if (!d) {
        RAD_ERR("control vector: rank %d: could not allocate %.2f MiB on the card for the "
                "directions", rank, (double) bytes / (1024.0 * 1024.0));
        return nullptr;
    }
    RadStream s = nullptr;
    if (rad_stream_create(&s, 0) < 0) {
        rad_dev_free(d, RAD_MEM_DEVICE);
        return nullptr;
    }
    const int rc = rad_memcpy_async(d, cv.dir.data(), bytes, s);
    rad_stream_sync(s);
    rad_stream_destroy(s);
    if (rc < 0) {
        RAD_ERR("control vector: rank %d: uploading the directions failed: %s", rank,
                rad_dev_last_error());
        rad_dev_free(d, RAD_MEM_DEVICE);
        return nullptr;
    }
    return (const float*) d;
}

uint64_t control_vector_salt(const ControlVector& cv) {
    if (!cv.loaded()) return 0;
    return hash_table(cv) | 1ull;
}

}  /* namespace rad */
