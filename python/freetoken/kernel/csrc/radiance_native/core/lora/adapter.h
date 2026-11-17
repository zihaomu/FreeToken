/* lora/adapter.h -- adapter files as image and video generators receive them, and the bytes of a
 * slot holding one or several of them at chosen strengths.
 *
 * An adapter for a generator is one safetensors file, written by whichever tool trained or
 * converted it, so the same delta arrives under several spellings:
 *
 *   LoRA     `<m>.lora_down.weight` / `<m>.lora_up.weight` (+ `<m>.alpha`): kohya, ComfyUI;
 *            `<m>.lora_A[.default].weight` / `<m>.lora_B[...]` (+ `.alpha`): PEFT, diffusers, ai-toolkit;
 *            `<m>.lora.down.weight` / `<m>.lora.up.weight`; `<m>.lora_linear_layer.down/up.weight`.
 *            delta = alpha / rank * up @ down; no alpha is a scale of 1. A convolution's down is
 *            [rank, cin, kh, kw] and its up [cout, rank, 1, 1], read as the flattened matrices.
 *   LoHa     `<m>.hada_w1_a/_w1_b/_w2_a/_w2_b` (+ alpha): delta = alpha / rank1 *
 *            (w1_a @ w1_b) * (w2_a @ w2_b), elementwise; exactly a rank1 * rank2 product of
 *            row-wise Kronecker factors.
 *   LoKr     `<m>.lokr_w1` or `_w1_a`/`_w1_b`, `<m>.lokr_w2` or `_w2_a`/`_w2_b` (+ alpha):
 *            delta = scale * kron(w1, w2), scale alpha / rank of a decomposed factor, else 1.
 *   full     `<m>.diff`: the delta itself.
 *
 * Every module becomes a Delta, a linear map with its strength folded in. A slot holds, for each
 * checkpoint module the adapters reach, the sum of the deltas that land there: the low-rank ones
 * side by side while their ranks fit the slot's rank (exact), otherwise the sum's best rank-R
 * approximation (a randomized SVD; the fraction of the delta's energy it keeps is reported).
 *
 * Where a module lands is the generator's own declaration (rad_lora_root, rad_lora_alias): a
 * key's prefix names the network, its path the checkpoint module directly or through an alias,
 * kohya's underscore form matched against the same names. */
#pragma once
#include "lora.h"
#include "../format/safetensors.h"

#include <functional>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {
namespace lora {

/* One module of an adapter file, before it is mapped onto the base. */
struct FileModule {
    enum Kind { LORA, LOHA, LOKR, FULL };
    std::string     key;           /* the module as the file spells it, tensor suffix removed */
    Kind            kind = LORA;
    /* LORA: down, up. LOHA: w1_a, w1_b, w2_a, w2_b. LOKR: w1, w1_a, w1_b, w2, w2_a, w2_b.
     * FULL: diff. */
    const StTensor* t[6] = {};
    const StTensor* alpha = nullptr;
};

class AdapterFile {
public:
    /* Opens and groups the file's tensors by module. Refuses a file holding a form whose effect
     * this engine cannot reproduce (DoRA's magnitude, OFT, Tucker cores), naming it. */
    int open(const std::string& path, std::string* err);
    const std::string&             path() const { return st_.path(); }
    const std::vector<FileModule>& modules() const { return mods_; }
    /* Tensors that change no linear (a trained bias, a norm's weight), with why; reported, not
     * applied. */
    const std::vector<std::pair<std::string, std::string>>& skipped() const { return skipped_; }
    /* The most common kind among the modules, for messages ("LoRA", "LoHa", ...). */
    const char* kind_name() const;

private:
    SafeTensorsFile st_;
    std::vector<FileModule> mods_;
    std::vector<std::pair<std::string, std::string>> skipped_;
};

/* A module's delta as a linear map [N, K] with its scale and strength folded in. Low-rank forms
 * keep their factors (B [N, r] and A [r, K], row-major); the rest are applied as operators to
 * blocks of `l` column vectors (X [K, l] -> Y [N, l], and the transpose), all f32. */
struct Delta {
    int64_t N = 0, K = 0;
    int64_t rank_bound = 0;
    double  energy = -1.0;          /* ||delta||_F^2, -1 when not known */
    bool    low_rank = false;
    int64_t r = 0;
    std::vector<float> B, A;
    std::function<void(const float* X, int64_t l, float* Y)> apply, apply_t;
};

/* The delta of one module at strength `w`. RAD_E_SHAPE / RAD_E_DTYPE / RAD_E_FORMAT name the
 * tensor. */
int module_delta(const FileModule& m, double w, Delta* out, std::string* err);

/* Rank-`r` factors of a delta given as an operator (randomized range finder, two power
 * iterations): B [N, r], A [r, K]. `kept` is the fraction of the squared Frobenius norm kept, of
 * Delta::energy when known and of what the sketch holds otherwise (1 when the delta's rank is at
 * most r). */
void truncate(const Delta& d, int64_t r, std::vector<float>* B, std::vector<float>* A,
              double* kept);

/* Where an adapter module lands in the base: rows [from_lo, from_hi) of its output (from_hi 0 =
 * all) at rows from `to_lo` of the checkpoint module `module`. */
struct Target {
    std::string module;
    int64_t     from_lo = 0, from_hi = 0, to_lo = 0;
};

/* The generator's key names over one rank's Program and sites. */
class KeyMap {
public:
    KeyMap(const Program& prog, const std::vector<Site>& sites);
    /* The targets of a file's module key; empty with `why` when it lands nowhere adaptable. */
    std::vector<Target> resolve(const std::string& key, std::string* why) const;
    /* A checkpoint module's full rows and columns as the sites record them, or -1 when the
     * declaration leaves them open. */
    int64_t rows(const std::string& module) const;
    int64_t cols(const std::string& module) const;
    /* The row pieces a module is cut into (RAD_MAP_ROWS; core/lora/lora.cpp names each
     * `<module>@<first row>+<rows>`), empty for a module the sites read whole; and whether a site
     * also reads it whole. */
    struct Piece { std::string name; int64_t lo = 0, n = 0; };
    const std::vector<Piece>& pieces(const std::string& module) const;
    bool whole(const std::string& module) const { return whole_.count(module) != 0; }

private:
    struct Root {
        std::string native;
        std::vector<std::string> spellings;
        std::unordered_map<std::string, std::string> kohya;   /* underscored rel -> module */
    };
    struct Alias {
        size_t root;
        std::string from, to;
        int64_t from_lo, from_hi, to_lo;
        std::regex dotted, kohya;
    };
    bool lookup(const Root& r, const std::string& rel, bool kohya, std::vector<Target>* out,
                std::string* why) const;

    const Program* prog_;
    std::vector<Root> roots_;
    std::vector<Alias> aliases_;
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> adaptable_;   /* -> rows, cols */
    std::unordered_map<std::string, std::string> known_;   /* module -> why not adaptable */
    std::unordered_map<std::string, std::vector<Piece>> pieces_;
    std::set<std::string> whole_;
};

/* One adapter at a strength, for compose(). */
struct Strength {
    std::string                        name;
    std::shared_ptr<const AdapterFile> file;
    double                             weight = 1.0;
};

/* What compose() did, for the log and the client. */
struct ComposeReport {
    int64_t modules = 0;             /* checkpoint modules the slot carries */
    int64_t exact = 0;               /* ... held exactly */
    double  worst_kept = 1.0;        /* lowest kept energy among the truncated */
    std::string worst_module;
    std::vector<std::string> unplaced;   /* "<adapter>: <key>: why" */
    std::vector<std::string> skipped;    /* "<adapter>: <tensor>: why" */
};

/* The factors of a composite, per checkpoint module: B [rows, r] and A [r, K], f32, held by the
 * caller while the packer reads them. */
struct Composite {
    int64_t rows = 0, K = 0, r = 0;
    std::vector<float> B, A;
    StTensor tA, tB;
    std::vector<StTensor> tPieces;   /* B's row ranges, one per piece of the module */
};

/* The sum of the adapters' deltas per checkpoint module, at most `max_rank` each. Unplaced
 * modules are reported, not refused, as UIs do. RAD_E_SHAPE when a module's shape disagrees with
 * the base's. */
int compose(const KeyMap& km, const std::vector<Strength>& ads, int max_rank,
            std::map<std::string, Composite>* out, ComposeReport* rep, std::string* err);

/* The packer's view of a composite (scale 1: the strengths and alphas are folded into B): a
 * module under its own name where a site reads it whole, and each of its row pieces under the
 * piece's name. */
std::map<std::string, AdModule> as_modules(std::map<std::string, Composite>& c, const KeyMap& km);

}  /* namespace lora */
}  /* namespace rad */
