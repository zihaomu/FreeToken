/* checkpoint.h -- a safetensors or GGUF checkpoint as a source of logical weights.
 *
 * Read by rad-convert (to write a container) and by the loader (to serve a checkpoint with
 * trivial encodings directly, spec §4.3). Both need:
 *
 *   - the tensors, as one row-major namespace across shards and drafters;
 *   - the metadata, from config.json and its siblings or the GGUF header;
 *   - the logical weight for a declared name: the tensors the plugin's name map resolves it to
 *     (concatenations, alternatives, expert slices), with their encoding when trivial, and its rows
 *     on demand.
 *
 * Trivial (spec §4.3): plain f32, f16 and bf16 (GGUF's too); FP8 with a block or per-row scale
 * plane (`X.weight_scale_inv`, or `X.weight_scale`); and MXFP4 as AMD Quark ships it (`X.weight`
 * U8 codes, `X.weight_scale` U8 exponents), byte for byte `fp4_e2m1*e8m0[1x32]`. Anything else
 * needs a recipe ahead of time, or for a published release in a form listed under
 * Checkpoint::Import an import: the shipped codes and scales re-laid, or decoded for a recipe, at
 * convert time.
 *
 * An imported quantisation's descriptor (`auto-round`, `gptq`, `quark`, `compressed-tensors`,
 * `paroquant`, `modelopt`) names the tool, not a served form, so no plugin claims it;
 * Checkpoint::imports() describes how the plugin is chosen.
 */
#pragma once
#include "../rad_internal.h"
#include "../rad_core.h"
#include "safetensors.h"
#include "torch_zip.h"
#include "../text/gguf.h"

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rad {

class Program;

/* ================================================================== tensors */
/* Shapes are row-major, shape[0] slowest. GGUF stores ne[] fastest-first, so Checkpoint::open
 * reverses it, once; getting it wrong transposes every matrix. */
struct SrcTensor {
    std::string          name;
    uint32_t             ggml = 0;
    uint32_t             rad_dtype = RAD_DT_INVALID;   /* INVALID: a GGUF block type */
    std::vector<int64_t> shape;                        /* row-major */
    const void*          data = nullptr;
    int64_t              bytes = 0;
    /* Owned payload for a derived tensor: a checkpoint's layout rebuilt at load time (the Ling
     * GGUF's head-last k_b/v_b fused into the plugin's interleaved kv_b, its expert-last stacks
     * regrouped per expert). When non-empty `data` points here and `rad_dtype` is RAD_F32. */
    std::vector<float>   owned;
    /* GPTQ-packed (auto_gptq's layout, shared by AutoRound's `auto_round:auto_gptq` export):
     * `data` is the qweight's int32 words [K * bits / 32, N], each packing 32 / bits codes along K
     * from the low bits up, unsigned with zero point 2^(bits-1). The tensor described is the
     * logical [N, K] code matrix, `rad_dtype` RAD_I4 or RAD_I8; `gptq_group` is the K columns one
     * scale row covers. 0 for other tensors. */
    int32_t              gptq_bits = 0;
    int64_t              gptq_group = 0;
    /* CT-INT4 (compressed-tensors' pack-quantized int4): `data` is X.weight_packed's int32 words
     * [N, K / 8], eight codes per word along K from the low bits up, each the signed code plus 8.
     * The tensor described is the logical [N, K] code matrix, `rad_dtype` RAD_I4 (symmetric) or
     * RAD_U4 (asymmetric, the stored nibbles); `ct_group` is the K columns one scale of
     * X.weight_scale [N, K / group] covers. 0 for other tensors. */
    int64_t              ct_group = 0;
    /* An asymmetric CT-INT4 weight's X.weight_zero_point: int32 [ceil(N / 8), G], eight zero
     * points per word along N from the low bits up, each the signed zero point plus 8
     * (compressed-tensors packs it along dim 0). Null for a symmetric weight. */
    const SrcTensor*     ct_zero = nullptr;
    /* ParoQuant's int5-bitplane (Import::ParoI5): `data` is the AWQ-packed low nibbles qweight
     * [K, N/8] and the tensor described is the logical [N, K] u5 code matrix; `aux` are the fifth-bit
     * plane qweight_hi [K, N/32], and the zero points qzeros [G, N/8] and qzeros_hi [G, N/32].
     * z-lab's int4 (Import::ParoI4) is the same without the fifth-bit planes (aux[0], aux[2] null). */
    int                  pbits = 0;            /* 5 for int5-bitplane, 4 for z-lab's plain AWQ-packed int4; 0 otherwise */
    const SrcTensor*     aux[3] = { nullptr, nullptr, nullptr };

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return shape.empty() ? 0 : n;
    }
};

/* ================================================================== metadata */
/* The model's metadata as RadModelMeta, with the strings it points at owned here. */
struct MetaOwn {
    std::string arch, name, quant;
    std::vector<std::string> keys, vals;
    std::vector<const char*> kp, vp;
    RadModelMeta m{};

    void kv(const std::string& k, const std::string& v) { keys.push_back(k); vals.push_back(v); }
    /* Set a key that must win over an existing one: lookups take the first row with a key, so
     * kv() would leave the old value in effect. */
    void set(const std::string& k, const std::string& v);
    bool has(const std::string& k) const;
    const char* get(const std::string& k) const;     /* null when absent */
    /* Republish RadModelMeta's pointers. kv()/set() may move the strings, so call this after the
     * last of them and before reading `m`. */
    void finish();
};

int meta_from_config_json(const std::string& dir, MetaOwn* out);
/* quantization_config's quant_method and fmt as the plugin descriptor ("fp8_e4m3"), "" without. */
std::string quant_descriptor(const MetaOwn& m);
int meta_merge_config_json(const std::string& dir, const std::string& prefix, MetaOwn* out);
int meta_merge_generation_config(const std::string& dir, MetaOwn* out);
int meta_merge_preprocessor_configs(const std::string& dir, MetaOwn* out);
int meta_from_gguf(GgufFile& g, MetaOwn* out);

/* ================================================================== the checkpoint */
class Checkpoint {
public:
    Checkpoint() = default;
    Checkpoint(const Checkpoint&) = delete;
    Checkpoint& operator=(const Checkpoint&) = delete;

    /* A .gguf, a .safetensors, a model.safetensors.index.json, or the directory holding one,
     * plus the metadata beside it. */
    int open(const std::string& path);
    /* Merge a second checkpoint under `prefix`: a drafter shipped as its own repository (DFlash2)
     * borrows the target's embedding and head, so it goes into the target's container with its
     * tensors and config.json keys under `draft.`. */
    int add_draft(const std::string& dir, const std::string& prefix);
    /* Merge one more safetensors file, its tensor names kept: a release that ships its networks
     * as separate single files (ComfyUI's split_files: a diffusion model, a text encoder and a
     * VAE). A name already present is refused. */
    int add_file(const std::string& path);
    /* Before open(): read config.json, generation_config.json and the tokenizer from `dir` rather
     * than from beside the checkpoint, for a single-file checkpoint that ships none of them (the
     * text encoder's repository describes such a release). */
    void set_meta_dir(const std::string& dir) { meta_dir_ = dir; }

    /* Merge PEFT adapters into the tensors they name: `path` is a .safetensors file or a
     * directory of them, of `<module>.lora_A.weight` [r, K] and `<module>.lora_B.weight` [N, r]
     * pairs (or ComfyUI's `.lora_down` / `.lora_up`) and an optional `<module>.alpha`. The target
     * is `<module>.weight`, or that with a wrapper prefix ("diffusion_model.", "transformer.",
     * "unet.") stripped; a module the checkpoint does not have is refused by name. In f32,
     * W += strength (alpha / r, or 1 without alpha) B A, stored in the target's dtype, a plain
     * bf16, f16 or f32 matrix. Call right after open(): a merged tensor's bytes move, and views
     * taken before read the old ones. `merged` receives the targets' names.
     *
     * `map_json` names a file of rules for adapters trained on other module names (a fused qkv,
     * renamed blocks): `{"strip": ["prefix."], "modules": {"<pattern>": ["<target>", ...]}}`,
     * a `*` in a pattern matching one run (the block index) that fills each target's `*`. With
     * several targets the module is their concatenation, B's rows split along them in order. A
     * rule's target the checkpoint lacks is refused. */
    int add_lora(const std::string& path, double strength, std::vector<std::string>* merged,
                 const std::string* map_json = nullptr);

    const SrcTensor* find(const std::string& n) const;
    const std::vector<SrcTensor>& tensors() const { return tensors_; }
    bool               is_gguf() const { return kind_ == Gguf; }
    GgufFile&          gguf() { return gguf_; }
    const std::string& path() const { return path_; }
    const std::string& dir() const { return dir_; }   /* where config.json and tokenizer.json sit */
    /* Where tokenizer.json, tokenizer_config.json and the chat template sit: dir() for a model
     * repository, the `processor/` (or `tokenizer/`) folder of a diffusers pipeline. */
    const std::string& tok_dir() const { return tok_dir_.empty() ? dir_ : tok_dir_; }
    MetaOwn&           meta() { return meta_; }
    const MetaOwn&     meta() const { return meta_; }

    /* Sub-tensor `index` along dim 0 of a stacked (rank >= 2) tensor, as a view owned here and
     * valid for the checkpoint's life. Null, with the reason logged, if there is no such slice. */
    const SrcTensor* slice(const SrcTensor& st, int index, const char* declared);
    /* Rows [r0, r0 + n) along dim 0 of a plainly stored tensor, as a view owned here (a
     * RAD_MAP_ROWS piece). Null, with the reason logged, when the rows are not there. */
    const SrcTensor* rows(const SrcTensor& st, int64_t r0, int64_t n, const char* declared);

    /* FP8's block, from quantization_config.weight_block_size; 128x128 when it says nothing. */
    int64_t fp8_block_rows() const { return fp8_br_; }
    int64_t fp8_block_cols() const { return fp8_bc_; }

    /* The checkpoint's quantisation imported weight by weight, or None; read from
     * quantization_config at open. Gptq: AutoRound's `auto_round:auto_gptq` export and GPTQ v1
     * (zero point stored less one). QuarkMx: Quark's MXFP4 (fp4 weights, per_group 32, e8m0
     * scales, no per-layer overrides). ParoMx: ParoQuant's `paroquant_mxfp4` (Quark's codes and
     * exponents in a rotated basis, rotations beside them). ParoM2: ParoQuant's
     * `paroquant_m2xfp4` (the same plus the m2xfp4 metadata byte a group). CtInt4, Nvfp4, CtFp8:
     * compressed-tensors' pack-quantized int4 (symmetric, or asymmetric with a zero point), a
     * release with an nvfp4-pack-quantized group, and one whose groups are all float-quantized FP8
     * (detect_import). ModelOpt: NVIDIA ModelOpt's NVFP4, FP8, or both per layer
     * (MIXED_PRECISION). In an Nvfp4, CtFp8 or ModelOpt release every weight without the NVFP4
     * tensors reads stock, so FP8 and bf16 linears beside the NVFP4 ones read by their own form.
     * An unreadable form of these stays None, with the reason in import_refusal(). The
     * quantization_config keys are then renamed `imported_quantization_config.*`, since they
     * describe the tool, not a served format. */
    enum class Import { None, Gptq, QuarkMx, CtInt4, ParoMx, ParoM2, ParoI5, ParoI4, Nvfp4, CtFp8,
                        ModelOpt };
    Import imports() const { return import_; }
    /* The form `name` is read in: a drafter merged by add_draft ships its own quantisation (a
     * ModelOpt NVFP4 drafter beside a bf16 target), so names under its prefix take that one. */
    Import imports_at(const std::string& name) const {
        return draft_import_ != Import::None && name.rfind(draft_prefix_, 0) == 0 ? draft_import_
                                                                                  : import_;
    }
    const std::string& import_refusal() const { return import_why_; }

    /* The pieces an imported `name` (a `.weight`) is read from, as views owned here: GPTQ-packed
     * codes and their scales, Quark's codes and exponents as RAD dtypes, or NVFP4's codes as fp4
     * beside an f32 scale plane derived from the e4m3 scales and the second level (a divisor in
     * compressed-tensors, a multiplier in ModelOpt). RAD_E_NOTFOUND when `name` is an ordinary
     * tensor or absent; another error, with `why`, for an unreadable imported form. */
    int import_piece(const std::string& name, const SrcTensor** codes, const SrcTensor** scale,
                     std::string* why);

    /* The metadata plane an imported `name` carries, when its form has one (ParoM2): [N, K/32] u8,
     * one byte a group, as `weight_meta` ships it. Null for every other form. */
    const SrcTensor* import_meta(const std::string& name) const;

private:
    /* `owner` is the index's weight_map, if any: a tensor it places in another shard is skipped
     * here. */
    int add_shard(const std::string& p, const std::string& prefix = "",
                  const std::unordered_map<std::string, std::string>* owner = nullptr);
    int open_safetensors(const std::string& path);
    int open_diffusers();
    int open_sgm();
    int open_hy3d(const std::string& weights);
    int add_companions();
    int add_torch_files();
    int open_gguf(const std::string& path);
    void detect_import();
    /* The Ling GGUF's layouts the plugin's names expect, rebuilt at load so the container's
     * recipe never sees the source layout: the KDA layers' ssm_a (the exporter's exp(A_log))
     * logged back to ssm_a_raw, the MLA layers' head-last k_b and v_b fused into one
     * interleaved kv_b, the quantised q_b split into the plugin's nope and rope blocks, and
     * the shared experts re-spelled under the block's mlp.shared_experts prefix. The routed
     * stacks stay where the reader's fastest-first flip leaves them. */
    int bailingmoe3_layouts();

    enum Kind { None, Gguf, SafeT } kind_ = None;
    std::string path_, dir_, tok_dir_, meta_dir_;
    GgufFile    gguf_;
    MetaOwn     meta_;
    std::vector<std::unique_ptr<SafeTensorsFile>> shards_;
    std::vector<std::unique_ptr<TorchZipFile>>    torch_;
    std::vector<SrcTensor> tensors_;
    std::unordered_map<std::string, size_t> index_;
    std::unordered_map<std::string, size_t> index_shard_;   /* name -> shards_ position */
    std::deque<SrcTensor> sliced_;            /* views; a deque never moves what it holds */
    /* Bytes a view's data points at when they are derived, not mapped (NVFP4's f32 scale
     * plane); they live as long as the views. */
    std::vector<std::unique_ptr<uint8_t[]>> owned_;
    int64_t fp8_br_ = 128, fp8_bc_ = 128;
    Import      import_ = Import::None;
    std::string import_why_;
    /* add_draft's drafter, when it ships a quantisation of its own: the form and the prefix. */
    Import      draft_import_ = Import::None;
    std::string draft_prefix_;
    int64_t     ct_group_ = 0;   /* CtInt4: the group every weight's config states */
    bool        ct_asym_  = false;   /* CtInt4: the config states asymmetric weights */
    /* import_piece's answers by name, so a weight resolved twice reuses its views. */
    std::unordered_map<std::string, std::pair<const SrcTensor*, const SrcTensor*>> imported_;
    /* The metadata plane of an imported form that has one (ParoM2), by name. */
    std::unordered_map<std::string, const SrcTensor*> imported_meta_;
};

/* ================================================================== logical weights */
/* Another spelling of the numbers a weight ships, asked for by a recipe's `import` rule
 * (ckpt_import_form). Each field left at its default keeps that part as the import writes it.
 *
 *   codes   an unsigned dtype of the shipped codes' width: a symmetric grid's signed codes c are
 *           written c + 2^(bits-1) under a zero point of 2^(bits-1), so (u - z) * s == c * s;
 *   zero    the zero plane's dtype (a shipped or re-expressed zero point), which must hold it;
 *   scale   the scale plane's dtype, each shipped scale rounded to nearest (counted when that
 *           changes it);
 *   block   a scale shipped once per row or per tensor (FP8) repeated over this block grid,
 *           which every block's rows must share. */
struct ImportForm {
    uint32_t codes = RAD_DT_INVALID, zero = RAD_DT_INVALID, scale = RAD_DT_INVALID;
    int64_t  block[2] = { 0, 0 };
};

/* A declared weight's source, as the name map resolves it: pieces joined along `concat_dim`, and
 * for block-scaled FP8 each piece's scale plane beside it. */
struct CkptWeight {
    std::string name;                         /* the logical (declared) name */
    std::vector<const SrcTensor*> pieces;     /* the codes, in concatenation order */
    std::vector<const SrcTensor*> scales;     /* FP8 only: one a piece */
    int      concat_dim = 0;
    uint32_t rank = 0;
    int64_t  shape[RAD_MAX_RANK] = {0};       /* the logical shape */
    int64_t  rows = 0, cols = 0;              /* its [rows, cols] view (rad_enc_view) */
    /* `enc` is the pieces' encoding when trivial; otherwise `why` says what is not. */
    bool        trivial = false;
    /* Imported pieces, read into `enc` by ckpt_plane_rows and ckpt_rows_f32. GPTQ-packed: `i4` or
     * `i8` codes transposed and offset to signed, f16 scales rounded to bf16 per 128 of K.
     * NVFP4: fp4 codes beside the f32 scale e4m3 / global per 16 of K. Not trivial (the scales
     * are changed or derived), so the loader refuses it; rad-convert writes it as `enc` or
     * quantises it by a recipe rule. A non-trivial FP8 weight may also state `enc` when it
     * still reads piece by piece (a row join ending inside a scale block); one that cannot be
     * read leaves `enc` unstated, and ckpt_rows_f32 refuses it with `why`. */
    bool        imported = false;
    /* ParoQuant MXFP4 (Import::ParoMx), also `imported`: written plane by plane through
     * ckpt_plane_rows. Planes 0 and 1 are Quark's codes and exponents byte for byte; the rest are
     * the "givens" transform's tables, built from `pq_pairs`/`pq_theta`/`pq_scale`, one set per
     * piece (a row part: gate and up, or q, k and v, each rotated separately). Import::ParoM2 is
     * the same with a third plane before the tables, the m2xfp4 scheme's metadata byte a group,
     * from `pq_meta`. */
    bool        paro = false;
    std::vector<const SrcTensor*> pq_pairs, pq_theta, pq_scale, pq_meta;
    /* GGUF Q8_0, Q4_K or Q6_K blocks (also `imported`): ckpt_plane_rows splits each row's blocks
     * into the planes of `enc` (ggml_planes.h), and ckpt_rows_f32 dequantises them. */
    bool        ggml_blocks = false;
    /* A weight-normalised tensor (RAD_MAP_WNORM): `pieces` is its direction v alone, and every
     * element of v with first index i reads times wnorm_scale[i] = g_i / ||v_i||. Not trivial
     * (the values change), so a recipe rule stores it. */
    bool        wnorm = false;
    std::vector<float> wnorm_scale;
    /* A weight computed from its sources at resolve (RAD_MAP_TRANSPOSE, EMA, TAP, COLS): its f32
     * values in the logical shape, which ckpt_rows_f32 serves. Not trivial, so a recipe rule
     * stores it. */
    std::vector<float> derived;
    /* An `import` rule's form, applied by ckpt_plane_rows (the weight is then `imported`).
     * `rounded` counts the shipped scales `form.scale` does not hold exactly and `max_rel` is their
     * largest relative change, both found by ckpt_import_form. */
    ImportForm  form{};
    int64_t     rounded = 0;
    double      max_rel = 0;
    RadEncoding enc{};
    std::string why;
};

/* Resolve `name` through `prog`'s name map. RAD_OK; RAD_E_NOTFOUND when the name map has no entry
 * for it yet (declare may add one later) or, for an optional weight, none of its tensors is
 * there; anything else with the reason in `why`. */
int ckpt_resolve(Checkpoint& ck, const Program& prog, const std::string& name, bool optional,
                 CkptWeight* out, std::string* why);

/* The quantiser name of a recipe rule rad-convert handles itself through ckpt_import_form. */
constexpr const char* kImportRule = "import";

/* An import rule's options (codes=DTYPE, zero=DTYPE, scale=DTYPE, block=RxC) as an ImportForm.
 * RAD_E_INVAL, with the option named in `why`, for an option it does not take or a value it
 * cannot read. */
int ckpt_import_form_parse(const std::vector<std::pair<std::string, std::string>>& options,
                           ImportForm* f, std::string* why);

/* Re-express a resolved `w` as `f` asks (a recipe's `import` rule): RAD_OK with `w->enc` the new
 * encoding and the form recorded for ckpt_plane_rows. RAD_E_UNSUPPORTED, with `why`, when `w` is
 * a plain float weight with nothing shipped to re-express (the rule declines it); any other error,
 * with `why`, when its shipped form cannot take `f`: a form other than an integer grid (GPTQ,
 * CT-INT4) or an FP8 weight with one scale per row or per tensor, a code or zero dtype that does
 * not hold the shipped values, or a block whose rows carry different scales. Rounding a scale to
 * `f.scale` is allowed and counted in `w->rounded`. */
int ckpt_import_form(CkptWeight* w, const ImportForm& f, std::string* why);

/* The checkpoint tensor a piece or scale reads: its name without the "[index]" a slice of a
 * stacked tensor carries, so an expert reads the stack it was cut from. */
std::string ckpt_tensor_of(const SrcTensor& t);

/* Whether `w` reads any tensor named in `names` (checkpoint names, as ckpt_tensor_of gives them),
 * through a piece or a scale. Each name it reads is added to `hit` when that is given. */
bool ckpt_reads_any(const CkptWeight& w, const std::unordered_set<std::string>& names,
                    std::unordered_set<std::string>* hit = nullptr);

/* One (part, group) of a ParoQuant rotation as a dense matrix: `S` stages of 64 pairs, each with
 * group-local columns `ia`/`ib` and angle `th`, stage-major; out[j * 128 + i] (f16) is output j for
 * unit vector e_i. This is the "givens" encoding's "t.dense" table. */
void paro_compose_dense(int64_t S, const int32_t* ia, const int32_t* ib, const float* th, uint16_t* out);

/* Logical rows [r0, r0 + nr) of `w` as f32 [nr, cols], whatever the pieces are stored as: a
 * plain float widened, FP8 times its block scale, a GGUF block type dequantised. */
int ckpt_rows_f32(const CkptWeight& w, int64_t r0, int64_t nr, float* out, std::string* why);

/* Rows [r0, r0 + nr) of plane `k` of a trivial or imported `w`, in the plane's own elements, dense
 * into `dst`: the bytes a container entry holds. */
int ckpt_plane_rows(const CkptWeight& w, int k, int64_t r0, int64_t nr, void* dst,
                    std::string* why);

}  /* namespace rad */
