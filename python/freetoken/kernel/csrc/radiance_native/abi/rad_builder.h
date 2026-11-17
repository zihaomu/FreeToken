/* rad_builder.h -- the declare phase. An architecture plugin states everything it will do before
 * doing any of it: every weight, every buffer, every op, in order. See spec.md §3.1.
 *
 * After declare, the whole model is known statically (every op, the kernel serving it, the weights
 * it touches, and the order), and the rest of the engine relies on that.
 */
#ifndef RAD_BUILDER_H
#define RAD_BUILDER_H

#include "rad_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RadBuilder RadBuilder;

/* ================================================================== model metadata */
/* What the loader read out of the container's header before any plugin ran. Dimensions come from
 * here, so one plugin covers every size in a family: qwen3_dense_bf16.so serves 2B, 9B and 27B. */
typedef struct RadModelMeta {
    const char* arch_id;         /* the string that selected this plugin */
    const char* name;
    const char* quant;           /* the quantisation descriptor, e.g. "w4a8_symz_g128" */

    int64_t n_layers;
    int64_t n_embd;
    int64_t n_head;
    int64_t n_head_kv;
    int64_t head_dim;
    int64_t n_ff;
    int64_t n_vocab;
    int64_t n_ctx_train;

    int64_t n_expert;            /* 0 for a dense model */
    int64_t n_expert_used;
    int64_t n_expert_shared;

    float   rms_eps;
    float   rope_theta;
    /* Always 1. Rotary scaling is a type with parameters (linear, YaRN, llama3, longrope) carried
     * in the free-form keys below (rope_parameters.* / rope_scaling.*, with text_config.* and
     * draft.* forms); arch/common/rad_arch.h's yarn_from reads them and refuses unknown schemes by
     * name. A bare factor here would read every scheme as a linear position divide. */
    float   rope_scale;

    /* Free-form key/value from the container header, for anything the struct above does not
     * name. A plugin reads what it needs; the core never interprets these. */
    int         n_kv;
    const char* const* kv_key;
    const char* const* kv_val;
} RadModelMeta;

long long   rad_meta_geti(const RadModelMeta* m, const char* key, long long dflt);
double      rad_meta_getf(const RadModelMeta* m, const char* key, double dflt);
const char* rad_meta_gets(const RadModelMeta* m, const char* key, const char* dflt);

/* ================================================================== build context */
/* A paged cache group's dtype says how a token's row in a head is laid out, at head_dim HD:
 *
 *   RAD_BF16, 4 * HD bytes: K then V, as the model computes them.
 *   RAD_I8, 2 * HD + 4 bytes: K and V as int8 codes with one f16 scale each per (token, head):
 *     [0, HD)            K's int8 codes (two's complement, -127..127)
 *     [HD, 2HD)          V's int8 codes
 *     [2HD, 2HD + 2)     K's scale (the head's absmax / 127)
 *     [2HD + 2, 2HD + 4) V's scale (absmax / 127)
 *   The plugin declares HD; the core adds the scales' bytes (rad_kv_row_elems). */

/* Elements of a token's row in one head of a paged group declared at `dtype` and `head_dim`: K
 * then V, and under the int8 cache the two f16 scales after them. */
static inline int64_t rad_kv_row_elems(uint32_t dtype, int64_t head_dim) {
    return 2 * head_dim + (dtype == RAD_I8 ? 4 : 0);
}

/* RadArchProbe::chain_pays. */
enum {
    RAD_CHAIN_NONE   = 0,
    RAD_CHAIN_SPEC   = 1,   /* with a drafter decoding speculatively */
    RAD_CHAIN_ALWAYS = 2    /* with or without one */
};

typedef struct RadBuildCtx {
    int      rank;            /* declare runs once per rank; the plugin declares dimensions */
    int      world_size;      /* already divided -- the core inserts no collectives (spec §9) */

    int64_t  max_tok;         /* the largest token count one step may carry */
    int64_t  max_seqs;        /* the largest number of sequences one step may carry */
    int64_t  max_ctx;         /* context bound this deployment was configured for */
    int      max_spec;        /* speculative window bound, 0 if speculation is off */

    /* The tensor-parallel wire. 0 (the zero-initialised default) is the exact all-reduce. Nonzero
     * accepts a lossy wire: at two ranks libr4d serves a Walsh-Hadamard-rotated 6-bit payload (6
     * bits an element plus a bf16 scale per group of 64, so a prefill chunk's message shrinks
     * 2.4x), at three and four ranks a tiered int8 two-shot. The plugin inverts it into
     * `all_reduce`'s `exact` parameter.
     * The polarity makes a context that forgets the field exact, not lossy (spec §9). */
    int      tp_wire_lossy;

    /* The all-reduce message size, in bytes, from which the lossy wire is used. Below it the
     * plugin must use the exact kernel even when `tp_wire_lossy` is set: compression pays only
     * once the transfer is bandwidth-bound, and a decode step's few residual rows cost more to
     * rotate/pack/unpack than they save. 0 means the plugin's default, 128 KiB. */
    int64_t  tp_wire_min_bytes;


    /* The KV cache dtype, a deployment choice rather than a property of the checkpoint.
     *
     * RAD_DT_INVALID (the zero-initialised default) means the plugin's default, bf16 for every
     * architecture in this tree. RAD_I8 is the other value every paged attention serves, the
     * int8 cache above: the attention kernels band on it and the store writes each token's codes
     * and scales. RAD_F8E4M3 reaches only a plugin whose probe sets RadArchProbe::kv_fp8_ok.
     *
     * An ABI field rather than an environment variable because it changes what the model
     * computes, so it must have one visible answer: --kv-cache-dtype. */
    int      kv_dtype;        /* RAD_DT_INVALID = the plugin's default (bf16); RAD_I8; RAD_F8E4M3 */
    int      is_draft;        /* this builder scope is a drafter (spec §10) */
    const char* scope;        /* "", "draft", "vision", "audio" -- prefixes declared names */

    /* Nonzero: a declare that only sizes the activations, at a `max_tok` smaller than the
     * deployment's. The core runs one per level of the activation arena it lends out; the result
     * gives every buffer's size when no step carries more than `max_tok` tokens, and the arena
     * above that is lent to the expert slab while steps are that small.
     *
     * Invariant: the plugin must not touch its own state while this is set. Its step function
     * reads state the real declare wrote at the real `max_tok`; overwriting it would run every
     * later step against the smaller configuration. Declare into scratch storage instead: nothing
     * a sizing declare produces is issued, and its op handles are the real declare's.
     *
     * Only asked of a plugin that set RadArchProbe::shape_probe_ok. Trailing and zero-defaulted,
     * so a plugin built against the older struct is unchanged. */
    int      shape_probe;

    /* The most patches one encoder pass may carry; 0 means the deployment serves no media, the
     * plugin declares no encoder, and a tower the container carries stays off the card. It is the
     * encoder's `max_tok` (its activations are sized against it) and bounds the largest single
     * image accepted, since one image is one attention segment and is never split across passes. */
    int64_t  max_enc_patches;

    /* The most logits rows one step may name, when that exceeds the plugin's own rule of one per
     * sequence (1 + max_spec at a speculative step); 0 is the rule. The engine's KL mode sets it to
     * a step's tokens plus a row per sequence, since it reads a full-vocabulary row at every prompt
     * position; the logits buffer and the lm_head's row range are sized from it. The engine checks
     * the declared logits buffer and refuses the mode when it is shorter, so a plugin that ignores
     * the field is refused rather than overrun.
     *
     * Trailing and zero-defaulted, so a plugin built against the older struct is unchanged. */
    int64_t  max_out_rows;

    /* The most frames one audio encoder pass may carry; 0 means no audio and no audio encoder.
     * It is the audio encoder's `max_tok`, as max_enc_patches is a vision tower's, and bounds the
     * longest clip, since one clip is one segment. The engine takes it from
     * RadArchProbe::audio_frames.
     *
     * Trailing and zero-defaulted, so a plugin built against the older struct is unchanged. */
    int64_t  max_audio_frames;

    /* The most tokens a step of more than one sequence may carry, when that is less than
     * max_tok; 0 means max_tok. Under --prefill-span a prefill alone in its step may take max_tok
     * and every other step is planned at a shared budget, so a step wider than this is one
     * sequence. A plugin may size a buffer holding a row per query of every sequence in a step at
     * this, and serve a wider step a window at a time.
     *
     * Trailing and zero-defaulted, so a plugin built against the older struct is unchanged. */
    int64_t  max_mixed_tok;

    /* ---- appended in ABI 30 (KV streaming) */
    /* --kv-resident in tokens, 0 when nothing streams: the ring a streamed group carves is this in
     * whole blocks, raised to the engine's floors and clamped to the context. A plugin states it as
     * its `kv_resolve`'s ring depth. */
    int64_t  kv_resident_blocks;
    /* RadArchProbe::kv_stream_ok. The streamed-only ops (`kv_store_streamed`, its folded form,
     * `kv_resolve`) are declared only when this is set and kv_resident_blocks asks for a ring;
     * otherwise rad_op_declared says no and the plugin takes its resident path. */
    int      kv_stream_ok;

    /* ---- appended in ABI 27 (control vectors) */
    /* A control vector loaded from a GGUF `controlvector` file found for this model, or null.
     *
     * The engine loads and validates it once, at bringup, and the plugin only reads it: the
     * direction table lives in device memory on every rank, in the activations' own float form,
     * and a per-request switch reaches the step through a derived buffer the plugin declares
     * itself (a row mask, `projection_on_rows`). Nothing is reloaded to turn the feature on or
     * off, and with the mask zero the model computes what it computes without a vector.
     *
     * The plugin decides where in its own graph the projection applies; for the gated-residual
     * architectures that is after each block's write into the wide stream, which is the point
     * llama.cpp's control-vector hook uses for `qwen4exp`. A plugin that does not declare an op
     * reading this table ignores it.
     *
     * Trailing and zero-defaulted, so a plugin built against the older struct is unchanged. */
    const struct RadControlVector* cvec;

    /* ---- appended in ABI 27 (row-split prefill attention) */
    /* 1 from --tp-attn-split rows: from the plugin's split threshold, a prefill-sized step under
     * tensor parallelism runs attention by rows, each rank all heads of its run of the step's
     * rows and an all_gather of the runs, instead of by heads. It is faster where the attention
     * dominates, costs the full-width projections on every rank, and is not bit-identical to the
     * head split. 0 (the zero-initialised default) is the head split. The engine sets it only
     * above one rank and only for a plugin whose probe sets RadArchProbe::tp_attn_rows_ok, and
     * refuses the flag otherwise.
     *
     * Trailing and zero-defaulted, so a plugin built against the older struct is unchanged. */
    int      tp_attn_rows;

    /* ---- appended in ABI 27 (recurrent state at f16) */
    /* RAD_F16 from --state-cache-dtype f16: the recurrent state caches (RAD_KV_LINEAR groups) are
     * stored at f16 instead of the plugin's f32, which halves the state pool, its checkpoints and
     * its traffic and changes the generated text. 0 (the zero-initialised default) is the
     * plugin's own width. The engine sets it only for a plugin whose probe has
     * RadArchProbe::state_f16_ok and refuses the flag otherwise.
     *
     * Trailing and zero-defaulted, so a plugin built against the older struct is unchanged. */
    uint32_t state_dtype;

    /* ---- appended in ABI 27 (draft coupling) */
    /* Nonzero from --draft-coupling on: a linear serial drafter samples each round's proposal from
     * its own shortlist law, or a block drafter's walk each position's (DFlash2), and declares the
     * planes and ops the coupled verify reads (abi/rad_spechub.h, docs/OPS.md `mtp_couple_propose`,
     * `dflash_couple_propose`). 0, the zero-initialised default,
     * declares the greedy drafter. A plugin that cannot couple returns RAD_E_UNSUPPORTED from
     * declare when it is set; the engine refuses a model whose program carries no coupled planes.
     *
     * Trailing and zero-defaulted, so a plugin built against the older struct is unchanged. */
    int      draft_coupling;

    /* ---- appended in ABI 28 (weight prefetch) */
    /* Nonzero from --weight-prefetch off: the plugin declares none of its weight prefetches, the
     * reads of a following GEMM's weight beside a collective (arch/common/rad_arch.h
     * prefetch_beside). An engine that shares its cards with another server sets it: where the
     * read goes on the second lane, the two servers hold more compute queues than a card has
     * slots, and the join from the first lane puts a queue switch into every decode step. 0, the
     * zero-initialised default, prefetches where the plugin does.
     *
     * Trailing and zero-defaulted, so a plugin built against the older struct is unchanged. */
    int      weight_prefetch_off;

    /* ---- appended in ABI 31 (tree verification) */
    /* Nonzero: grow the block drafter's proposals into a tree (rad_drafter_tree) where the plugin
     * can. The engine sets it where the plugin's probe said trees pay (RadArchProbe::tree_pays)
     * and the drafter decodes speculatively; 0 keeps the chain. A plugin that grows no trees
     * ignores it. 64-bit so it starts past the older
     * struct's tail padding, where a plugin built against it would read nothing.
     *
     * Trailing and zero-defaulted, so a plugin built against the older struct is unchanged. */
    int64_t  draft_tree;

    /* ---- appended in ABI 32 (calibration) */
    /* --calib-dir DIR: the deployment records calibration statistics (the second moment of each
     * linear's input, GPTQ's Hessian) into DIR; null records nothing. A plugin that keeps none
     * ignores it. `calib_layer_lo`..`calib_layer_hi` (--calib-layers, hi exclusive; both 0 for
     * every layer) bound the layers recorded in one run, so a model whose statistics do not fit
     * in memory at once is recorded in several runs over the same corpus. `calib_tokens` is the
     * corpus's token count under --calib-corpus: the plugin writes its files once it has seen
     * that many rows; 0 when the deployment drives no corpus.
     *
     * Trailing and zero-defaulted, so a plugin built against the older struct is unchanged. */
    const char* calib_dir;
    int32_t  calib_layer_lo;
    int32_t  calib_layer_hi;
    int64_t  calib_tokens;

    /* ---- appended in ABI 32 (an image generator's bounds) */
    /* --image-gen-max-pixels: the largest image an image generator makes, in pixels; and
     * --image-gen-max-refs: the reference pictures a request at most. 0: the model's own. A
     * generator sizes its per-image and conditioning buffers for them and states them in its
     * RadGeneratorDecl (max_tokens, max_refs), which the core holds requests to; one that keeps
     * fixed buffers ignores them, and the core says so at startup. */
    int64_t  gen_max_pixels;
    int64_t  gen_max_refs;

    /* ---- appended in ABI 37 (where a routed expert may live) */
    /* --weights-disk-tier: nonzero when weights may stay in the container. An expert read from the
     * drive has to be stored whole, so a plugin that can lay its routed experts out either way
     * keeps them whole for this; 0 leaves it free to split them across the ranks by width. */
    int      expert_file_tier;

    /* ---- appended in ABI 38 (several engines in one process) */
    /* The first of this engine's slots. Each engine in the process holds a contiguous block of
     * slots, one per rank, so an instance's slot, `slot_base + rank` (rad_build_slot), is unique
     * among the ranks of every engine alive, where `rank` is unique only within one engine. A
     * plugin keeps per-instance state (handles, geometry, its step's view of the model) indexed by
     * the slot, and reads it back in step() with rad_slot(). Zero, the default, makes the slot the
     * rank: one engine in a process, or a test driving a plugin alone. Two engines may serve the
     * same plugin at once only when it says so (rad_arch_multi_instance). */
    int      slot_base;
} RadBuildCtx;

/* The most instances (engine ranks) one process holds at once; every slot is below it. */
#define RAD_MAX_SLOTS 32

/* The declaring instance's slot (see RadBuildCtx::slot_base). */
static inline int rad_build_slot(const RadBuildCtx* c) { return c->slot_base + c->rank; }

/* ================================================================== control vectors */
/* A direction per layer, in the shape llama.cpp's `--cvec-mode project` applies it:
 *
 *     h <- h - s_l (h . v_l) v_l            v_l = d_l / |d_l|, s_l = |d_l| (times the user scale)
 *
 * with `v_l` unit and `s_l` the norm the file stored (so a unit vector's scale is 1). The engine
 * normalises the rows when it uploads them, so a kernel reads `dir` as the unit direction and
 * takes `s` from `scales`; a zero scale means "this layer is not steered".
 *
 * `dir` is [n_layers, n_embd] f32, row `l` that layer's direction, row 0 unused (layer 0 has no
 * direction in the format). Each rank has its own copy: the pointer a plugin receives is that
 * rank's, and the residual stream is replicated, so no collective is involved.
 *
 * `n_layers` is the model's layer count, which may be longer than the file's (the published
 * Flash-Next vector carries 47 of 48 rows). The steered range is not a separate field: a layer is
 * steered exactly when its `scales` entry is nonzero, so a plugin asks that and cannot disagree
 * with the loader. */
typedef struct RadControlVector {
    const float* dir;        /* device, [n_layers, n_embd] f32, unit rows */
    const float* scales;     /* host,    [n_layers]     f32, |d_l| * user scale; 0 = not steered */
    int64_t      n_embd;
    int64_t      n_layers;
} RadControlVector;

/* ================================================================== weights */
/* Access class, declared per weight because only the architecture plugin knows it. It tells the
 * placement planner whether prefetch can be exact (spec §5.2). */
enum {
    RAD_ACCESS_PER_TOKEN   = 0,  /* read every forward pass -- dense weights */
    RAD_ACCESS_PER_REQUEST = 1,  /* read once per request -- an encoder tower, with an image */
    RAD_ACCESS_CONDITIONAL = 2,  /* read only if selected -- routed experts */
    RAD_ACCESS_RARE        = 3,  /* may not be read at all in a given deployment */
    RAD_ACCESS_VOCAB       = 4   /* lm_head and the embedding table: read every token, very
                                  * large, and one is a gather rather than a GEMM. Neither
                                  * PER_TOKEN's prefetch nor CONDITIONAL's prediction fits it
                                  * (spec §5.2). */
};

/* How a weight is split across tensor-parallel ranks. The plugin says; the core does not infer. */
enum {
    RAD_SHARD_NONE = 0,   /* replicated on every rank */
    RAD_SHARD_ROW  = 1,   /* split along dim 0 -- column-parallel in the usual naming */
    RAD_SHARD_COL  = 2    /* split along dim 1 -- row-parallel, needs a following all-reduce */
};

/* Grouping key: the unit of movement. An expert's gate/up/down share one, because the unit of
 * transfer is an expert, not a tensor (spec §4.2). Layer grouping makes layer offload a contiguous
 * run rather than a scatter. */
typedef struct RadWeightGroup {
    int32_t layer;    /* -1 for a model-level weight */
    int32_t expert;   /* -1 for a non-expert weight */
    int32_t slot;     /* free index within the group, e.g. gate=0 up=1 down=2 */
} RadWeightGroup;

/* Note: these are C compound literals, which C++ accepts only as a GNU extension. A C++ plugin
 * should use the equivalent grp_layer()/grp_expert()/grp_model() inline functions in
 * `<arch/rad_arch.h>`, as the in-tree plugins do. */
#define rad_group_layer(l)      ((RadWeightGroup){ (int32_t)(l), -1, 0 })
#define rad_group_expert(l, e)  ((RadWeightGroup){ (int32_t)(l), (int32_t)(e), 0 })
#define rad_group_model()       ((RadWeightGroup){ -1, -1, 0 })

/* The most projections one stored tensor may stack along dim 0. The bf16 delta net needs five
 * (q, k, v, a, b); eight leaves room without making the decl big. */
enum { RAD_MAX_ROW_PARTS = 8 };

typedef struct RadWeightDecl {
    uint32_t       dtype;       /* what the kernel reads; the stored layout comes from the kernel */
    uint32_t       rank;
    int64_t        shape[RAD_MAX_RANK];   /* this rank's extent of what `planes` select (below) */
    int            access;      /* RAD_ACCESS_* */
    int            shard;       /* RAD_SHARD_* */
    RadWeightGroup group;
    int            optional;    /* absent in the container is not an error */
    /* What is stacked in dim 0, for a RAD_SHARD_ROW weight whose stored tensor concatenates
     * several projections ([gate|up], the delta net's [q|k|v]). Rows of this rank, in stored
     * order, summing to shape[0]; 0 parts means dim 0 is one projection.
     *
     * A tensor-parallel rank owns a slice of each stacked projection, not a contiguous slice of
     * the stack: rows [rank*D, (rank+1)*D) of a fused gate_up would give rank 0 the whole gate and
     * rank 1 the whole up, and the model would produce fluent wrong text (spec §9). The parts
     * cannot be inferred from the container, which holds only the fused result, and a checkpoint
     * tensor may already be a fusion the name map does not mention (in_proj_qkv is three head
     * groups in one [10240, 5120]). Only the plugin knows them. */
    int            n_row_parts;
    int64_t        row_parts[RAD_MAX_ROW_PARTS];

    /* Which stored weight this is a view of, and which of its planes. spec.md §4.3.
     *
     * A container entry or checkpoint tensor is a logical weight: the planes of its encoding
     * (rad_encoding.h). A block-fp8 linear's codes and scale plane are one weight, not two. A
     * kernel takes them as whatever operands its schema has: gemm_nt_q takes codes and scales as
     * two, the fp8 lm_head one operand holding both. So a declared weight names its `source` (null
     * for the logical weight of its own name) and the `planes` it takes, by role, comma-separated:
     * "codes", "scale", "codes,scale". Null takes every plane, which for a plain weight is one.
     *
     * One declared weight is one stored buffer. Its dtype and shape describe what is selected: a
     * single plane's own dtype and extents, or for several, the codes' dtype and the logical
     * extents. A dtype other than the plane's is legal only as an exact widening (a bf16 norm read
     * as f32), which the loader performs; anything lossy is the quantiser's job, done before the
     * container is written. */
    const char*    source;
    const char*    planes;
} RadWeightDecl;

rad_weight rad_decl_weight(RadBuilder* b, const char* name, const RadWeightDecl* d);

/* A stored weight's encoding, its logical extents in `shape` (RAD_MAX_RANK entries, may be null)
 * and its rank in `rank` (may be null). RAD_OK, or RAD_E_NOTFOUND when the model has no such
 * weight.
 *
 * The answer comes from whatever the model is loaded from (a container entry, a checkpoint tensor,
 * or in rad-convert the recipe that will quantise it), so a plugin chooses its ops the same way at
 * convert and at load: an fp8 GEMM for a block-fp8 weight, the rotated expert kernels for
 * `i4*bf16[1x128]/fwht128`. Ask after the weight's name map entry (rad_decl_name_map), which is how
 * a checkpoint is searched for it. A builder with no model behind it (a test, a synthetic tool)
 * answers RAD_E_NOTFOUND, and a weight declared there is read as plain in its declared dtype. */
int rad_weight_encoding(RadBuilder* b, const char* source, RadEncoding* out, int64_t* shape,
                        uint32_t* rank);

/* ================================================================== buffers */
/* Activations are declared too, so the buffer plan can analyse them. Liveness is computed over the
 * declared op list, so scratch and intermediates come from one arena sized once instead of an
 * allocator called per step (spec §3.1). */
enum {
    RAD_BUF_TRANSIENT = 0,  /* liveness-analysed, may share storage with another transient */
    RAD_BUF_PERSIST   = 1,  /* survives across steps: a KV-adjacent scratch, a routing histogram */
    RAD_BUF_DERIVED   = 2   /* computed once per step by the core from the batch (spec §8) */
};

typedef struct RadBufDecl {
    uint32_t    dtype;
    uint32_t    rank;
    int64_t     shape[RAD_MAX_RANK];   /* sized at max_tok, which is the worst step */
    int         kind;                  /* RAD_BUF_* */
    int         domain;                /* RAD_DOMAIN_DEVICE or RAD_DOMAIN_HOST */
    const char* derive;                /* RAD_BUF_DERIVED: the derivation the core runs */
    /* The KV group a derivation is about, or 0 for the only one of its kind.
     *
     * `conv_state_index` computes a read cursor inside a rolling window, so it needs that group's
     * slot per sequence and window depth, and a model may declare several (Qwen4-Exp's PLE layer
     * has two conv states of different depths and dtypes beside the gated delta net's). 0 with
     * more than one group of the kind a derivation reads is refused: the wrong window depth gives
     * a plausible cursor and a convolution that reads history it never wrote.
     *
     * Trailing and zero-defaulted, so a plugin built against the older struct is unchanged. */
    rad_kvgroup kv;
} RadBufDecl;

rad_buf rad_decl_buffer(RadBuilder* b, const char* name, const RadBufDecl* d);

/* ================================================================== KV groups */
/* Each distinct state kind is a KV group with its own block geometry, pool and slot mapping in the
 * step batch. Linear-attention state is a KV group too: not paged the same way, but allocated,
 * mapped and freed by the same manager, so there is one record of what a sequence owns
 * (spec §7.2). */
enum {
    RAD_KV_FULL    = 0,   /* paged, per-token, growing -- full attention */
    RAD_KV_WINDOW  = 1,   /* paged, per-token, bounded by the window */
    RAD_KV_LINEAR  = 2,   /* per-sequence, fixed size -- GDN recurrent state */
    RAD_KV_CONV    = 3,   /* per-sequence, a rolling window of width-1 + num_spec entries */
    /* Paged exactly like RAD_KV_WINDOW, but the window's floor is per sequence and set at
     * runtime (R-SWA, spec §7.2): `window` is the ring's W, and a sequence keeps everything from
     * position 0 up to its own prompt (prefill and image rows) plus a ring of W slots after it,
     * rewritten in place. So the group is sized for the whole context plus one ring a sequence,
     * never checkpointed, and never reused from the prefix cache. A model that serves a fixed
     * sliding window declares RAD_KV_WINDOW instead. */
    RAD_KV_RING    = 4
};

typedef struct RadKVGroupDecl {
    int      kind;            /* RAD_KV_* */
    uint32_t dtype;
    int64_t  window;          /* RAD_KV_WINDOW: the window; RAD_KV_RING: the ring's W; 0 otherwise */
    int64_t  n_head_kv;
    int64_t  head_dim;
    int64_t  state_dim[2];    /* RAD_KV_LINEAR: the per-head state geometry, e.g. {128,128} */
    int64_t  conv_width;      /* RAD_KV_CONV */
    /* Block size is not declared. It is taken from the resolved attention kernel's constraints at
     * declare, so the core needs no edit when a kernel changes (spec §7.2). Read it back with
     * rad_kv_block_size() after declaring the attention op that consumes this group. */
} RadKVGroupDecl;

/* `rad_kvgroup` is in rad_types.h with the other handles -- RadBufDecl above carries one. */
rad_kvgroup rad_decl_kv_group(RadBuilder* b, const char* name, const RadKVGroupDecl* d);
/* Valid only after the op that reads this group has been declared. 0 before that. */
int64_t     rad_kv_block_size(RadBuilder* b, rad_kvgroup g);
/* Bytes one block of one layer of a paged group occupies: its pool slot's pitch and its host
 * mirror row's. 0 for a group that is not paged, or before the block size is known. Additive
 * export. */
int64_t     rad_kv_layer_bytes(RadBuilder* b, rad_kvgroup g);

/* Bind a layer to a group, so the block manager knows how many state instances a sequence owns. */
int rad_bind_layer_kv(RadBuilder* b, int layer, rad_kvgroup g);

/* ================================================================== ops */
/* Declared with their parameters and weight operands. The core resolves each against the kernel
 * hierarchy, per band, per domain. This phase always completes, and every op that failed to
 * resolve is reported together at the end. */
rad_op rad_decl_op(RadBuilder* b, const char* op,
                   const RadParam* params, int n_params,
                   const rad_weight* weights, int n_weights);

/* Convenience wrappers so a declare function reads like a list of what the model does.
 *   rad_op h = RAD_OP(b, "gemm_w4a8_nt",
 *                     RAD_PARAMS(RAD_RANGE("M",1,max_tok), RAD_INT("N",8192), RAD_INT("K",5120)),
 *                     RAD_WEIGHTS(w));
 */
#define RAD_PARAMS(...)   ((const RadParam[]){ __VA_ARGS__ }), \
                          (int)(sizeof((const RadParam[]){ __VA_ARGS__ }) / sizeof(RadParam))
#define RAD_NOPARAMS      ((const RadParam*)0), 0
#define RAD_WEIGHTS(...)  ((const rad_weight[]){ __VA_ARGS__ }), \
                          (int)(sizeof((const rad_weight[]){ __VA_ARGS__ }) / sizeof(rad_weight))
#define RAD_NOWEIGHTS     ((const rad_weight*)0), 0
#define RAD_OP(b, op, ...) rad_decl_op((b), (op), __VA_ARGS__)

/* An op's buffer operands. rad_decl_op names only weights, which placement moves; the buffer
 * planner needs liveness, i.e. which ops read and write which buffers. A transient with no
 * declared use is assumed live for the whole program, which is sound (a plugin may issue any
 * handle in any order) but wasteful; --debug-graph prints how many buffers fell into that case.
 * Separate calls rather than four more parameters on every rad_decl_op. */
int rad_op_reads (RadBuilder* b, rad_op h, const rad_buf* bufs, int n);
int rad_op_writes(RadBuilder* b, rad_op h, const rad_buf* bufs, int n);

/* A buffer the architecture touches from the second lane, the one case where the op range a
 * buffer's declarations imply does not bound when its bytes are in use.
 *
 * The planner packs transients from one linear order (spec §3.1): two whose declared lives do not
 * overlap share arena bytes. `rad_lane` breaks that: lane 1's ops sit at op indices inside lane
 * 0's, so a shared-arm scratch live over ops 41..42 can overlay a routed-arm scratch live over ops
 * 37..38, and on two streams those runs overlap in time, corrupting data silently.
 *
 * A buffer named here is live for the whole program, like a buffer with no declared use. It is
 * per-buffer rather than a mode because unsharing every buffer costs ~220 MiB on a large model,
 * against a few MiB for one block's lane-1 transients. Call it at declare for every buffer the
 * second lane reads or writes; the lane-0 buffers they could share with are covered, since
 * sharing needs both sides.
 *
 * A no-op for a PERSIST or DERIVED buffer, which is live everywhere already. */
int rad_buf_concurrent(RadBuilder* b, rad_buf buf);

/* ---- appended in ABI 27 (the second lane's window) */
/* The second lane's window: every op the architecture issues on lane 1 is declared between
 * rad_lane1_window_begin and rad_lane1_window_end. With a window a rad_buf_concurrent buffer is live
 * over the window and its own declared uses instead of the whole program, so buffers used only
 * outside the window (a drafter's, the heads') can share its bytes. The runtime holds the
 * architecture to it: an op issued on lane 1 outside the window fails the step by name, and an op
 * issued on lane 0 outside the window joins lane 1 first when lane 1 has work pending. Optional, one
 * window a program; call both at declare, in order. */
int rad_lane1_window_begin(RadBuilder* b);
int rad_lane1_window_end(RadBuilder* b);

/* An uneven shard: this rank's slice of a RAD_SHARD_ROW or RAD_SHARD_COL weight's split dimension
 * is [lo, hi) rather than an even 1/tp. Rows of each stacked row part for ROW, columns for COL, in
 * the container's unsharded units. The declared extent must already be `hi - lo` (per row part,
 * for ROW).
 *
 * For a width that does not divide into equal whole blocks across ranks: a 640-column projection
 * with 128-column scale blocks is five blocks, split three and two. The stacked parts of a ROW
 * weight must be equal in the container, since only the rank's own part sizes are declared.
 * Refused for a container converted for more than one rank (its entries are already even shares)
 * and for a stored layout whose row runs or column groups the span would cut. */
int rad_weight_shard_span(RadBuilder* b, rad_weight w, int64_t lo, int64_t hi);

/* A stacked weight's span, one [lo, hi) per row part, with each part's whole extent in the
 * container, for parts of unequal width that rad_weight_shard_span's single span cannot express
 * (the delta net's [q|k|v|z] is 2048, 2048, 6144, 6144 rows on Flash-Next). `n` is the declared
 * part count; container part g is `total[g]` rows, of which this rank takes [lo[g], hi[g]) and
 * declared `hi[g] - lo[g]`; the totals must sum to the entry's rows. An uneven share of the heads
 * (6 of 16 K heads, with three ranks taking 6, 5 and 5) is declared this way, without padding.
 * Additive export. */
int rad_weight_shard_span_parts(RadBuilder* b, rad_weight w, const int64_t* lo, const int64_t* hi,
                                const int64_t* total, int n);

/* This rank keeps no cache for the group. The group is still declared as on every other rank (the
 * scheduler indexes groups by position and one block table serves every rank, so it must exist
 * with the same block size and layers), but no pool is carved for it. For a rank the geometry
 * gives no attention heads (rad_arch.h Geom::attn_zero), whose attention is a `fill` of zeros.
 * After this, no op on this rank may name the group as a RAD_KV operand; the engine refuses one
 * at issue. Additive export. */
int rad_kv_group_zero(RadBuilder* b, rad_kvgroup g);

/* Every rank holds the same bytes for this paged group, so a tier stores it once. Under tensor
 * parallelism a host slot for a cached entry is otherwise every rank's shard end to end; a group
 * declared replicated takes one window at the front of the slot, written by rank 0 and read by
 * every rank on a restore. The plugin asserts the replication (an MLA latent cache that every
 * rank computes whole, an indexer's key cache every rank pools into); the engine moves bytes and
 * cannot know. RADIANCE_DEBUG_TIER_VERIFY checks the claim: each follower hashes what it would
 * have written and the leader compares. Paged groups only, and never a group this rank declared
 * zero. No effect at one rank, and none on a group that streams, whose host mirror is its tier.
 * Additive export. */
int rad_kv_group_replicated(RadBuilder* b, rad_kvgroup g);

/* The group is read over its whole history every decode step (an indexer's block keys), so it
 * cannot stream. While a sibling group streams, its carve holds every sequence's full context
 * rather than the split's share, as the sibling's host mirror holds --max-num-seqs sequences.
 * Additive export. */
int rad_kv_group_dense_read(RadBuilder* b, rad_kvgroup g);

#define RAD_BUFS(...)  ((const rad_buf[]){ __VA_ARGS__ }), \
                       (int)(sizeof((const rad_buf[]){ __VA_ARGS__ }) / sizeof(rad_buf))

/* Did the last rad_decl_op resolve? A plugin asks for the fused op first and emits the unfused
 * sequence when it does not, which makes fusion a selection rather than a compiler pass
 * (spec §2.3). An unresolved handle is RAD_NULL_HANDLE and becomes an error only if issued. */
int rad_op_resolved(RadBuilder* b, rad_op h);

/* Does any loaded kernel library declare op `op`? Declaring an op outside the vocabulary is an
 * error, since that is what a misspelt name looks like; an op that is declared but resolves to no
 * kernel is not (rad_op_resolved). A plugin asks this first for a fused op only an optional
 * library defines, and emits the chain when the answer is 0. Additive export. */
int rad_op_declared(RadBuilder* b, const char* op);

/* ================================================================== checkpoint name map */
/* The plugin owns the map from checkpoint tensor names to declared weight names, including
 * convert-time fusions (q/k/v into one qkv, gate and up into gate_up). It is the only component
 * that knows both sides, and rad-convert reads the map from it (spec §2.4). */
enum { RAD_MAP_COPY = 0, RAD_MAP_CONCAT = 1, RAD_MAP_WNORM = 2, RAD_MAP_ROWS = 3,
       RAD_MAP_TRANSPOSE = 4, RAD_MAP_EMA = 5, RAD_MAP_TAP = 6, RAD_MAP_COLS = 7 };

typedef struct RadNameMap {
    const char* declared;        /* the name given to rad_decl_weight */
    int         mode;            /* RAD_MAP_COPY | RAD_MAP_CONCAT */
    int         n_src;
    const char* src[8];          /* checkpoint tensor names */
    int         concat_dim;

    /* A sub-tensor index within src[i], or -1 for the whole tensor. GGUF stores a MoE layer's
     * experts stacked as one ffn_gate_exps of shape [.., n_expert], while placement needs one
     * declared weight per expert; this carries the expert index so rad-convert need not parse it
     * out of the declared name. */
    int         src_index[8];

    /* RAD_MAP_ROWS only: piece i is rows [src_index[i], src_index[i] + src_rows[i]) of the 2-D (or
     * higher, rows along dim 0) tensor src[i]; the pieces join along dim 0. RAD_MAP_COLS: columns
     * [src_index[0], src_index[0] + src_rows[0]) of src[0]'s last dimension. RAD_MAP_TRANSPOSE with
     * src_index[0] >= 0: rows [src_index[0] * src_rows[0], (src_index[0] + 1) * src_rows[0]) when
     * src[0] is rank 2 and src_rows[0] > 0 (a stack flattened over rows); otherwise src_index[0] is
     * the dim-0 sub-tensor of a higher-rank stack. */
    int64_t     src_rows[8];
} RadNameMap;

/* RAD_MAP_COPY with n_src > 1: the first source present in the container wins. This expresses a
 * tied lm_head (output.weight if present, else token_embd.weight), so one plugin covers family
 * members that differ only in tying. RAD_MAP_CONCAT with n_src > 1 concatenates all sources along
 * concat_dim and requires every one.
 *
 * RAD_MAP_WNORM folds a weight-normalised tensor (torch.nn.utils.weight_norm) into the plain
 * weight it stands for: src[0] is the magnitude `weight_g` ([d0, 1, ...]), src[1] the direction
 * `weight_v` ([d0, ...]), and the result is g * v / ||v||, the norm taken over every dimension but
 * the first. Folded in f32 and stored as the recipe says. A model that applies the
 * parametrisation at every forward computes the same weight each time; doing it once at convert
 * leaves the plugin an ordinary weight.
 *
 * RAD_MAP_ROWS keeps only row ranges of its sources (see src_rows), joined: a head read over a
 * fraction of a vocabulary (an audio model's codes, a few rows of a text model's 200K) stores that
 * fraction rather than the whole table. Successive ROWS maps for one name extend it in call order,
 * as CONCAT maps do, so a weight may take more than eight pieces (an output projection whose rows
 * are permuted, one piece a row).
 *
 * RAD_MAP_COLS keeps one column range of src[0] (see src_rows): a linear whose input carries
 * channels the runtime never feeds (Cosmos's padding-mask channel, always zero) is read over the
 * columns it is fed.
 *
 * RAD_MAP_TRANSPOSE swaps the last two dimensions of src[0] in each leading slice, [.., a, b] to
 * [.., b, a]: a linear a checkpoint keeps as (in, out), where the GEMMs read (out, in). With
 * src_index[0], it first takes one expert's sub-tensor (see src_rows): a stacked expert table kept
 * [n_expert, in, out] or flattened [n_expert * in, out].
 *
 * RAD_MAP_EMA is an exponential-moving-average codebook as EnCodec-style quantisers keep it: src[0]
 * the embedding sums [bins, ...], src[1] the cluster usage [bins], and the codebook is each row
 * divided by max(usage, 1e-5), in f32. Successive EMA maps for one name stack their codebooks
 * along dim 0, as CONCAT maps extend (a residual quantiser's codebooks as one table).
 *
 * RAD_MAP_TAP keeps one index of one dimension of src[0], dropping that dimension: index
 * src_index[0] (negative counts from the end) of dimension `concat_dim`. A causal 3-D convolution
 * run on a single frame reads only its last temporal tap, so [cout, cin, kt, kh, kw] read at tap -1
 * of dimension 2 is the 2-D convolution it then is.
 *
 * TRANSPOSE, EMA, TAP and COLS are computed at convert in f32, like WNORM, and stored as the recipe
 * says. */

int rad_decl_name_map(RadBuilder* b, const RadNameMap* m);


/* ================================================================== the step's output */
/* Which buffer holds this step's logits. The core samples from it (the sampler chain is core logic
 * whose ops resolve through the same hierarchy as the plugin's, spec §13), and only the plugin
 * knows which buffer it is. A name match would be wrong: a drafter or encoder tower declares into
 * a scope (RadBuildCtx::scope, spec §10, §11), so "the buffer called logits" can be two buffers in
 * one Program.
 *
 * Its extent along dim 0 bounds RadBatch::n_out: one row per sampled position, not per token of
 * the step. Declaring it [max_tok, n_vocab] is legal but, at a 248K vocabulary, 8 GiB.
 *
 * A Program with no logits buffer cannot serve, and the engine says so at startup. An
 * embedding-only or encoder-only plugin declares none. */
int rad_declare_logits(RadBuilder* b, rad_buf logits);

/* Scored steps (RadBatch::score_only). A plugin whose lm_head is the bf16 table `table`
 * ([n_vocab, n_embd], e.g. a tied embedding, whatever encoding the head itself is read at) and
 * whose final-normed rows land in `hidden` ([rows, n_embd] bf16, row k the step's k-th sampled row,
 * as the lm_head reads them) declares both here. On a step where every sampled row is a scored row
 * the core sets RadBatch::score_only, the plugin issues everything but the lm_head, and the core
 * reads the hidden rows and the candidates' table rows back and takes the dot products on the host
 * in f64: the logits a scored request returns, at no head read on the card (half a gigabyte a step
 * at a 128K bf16 vocabulary). Optional; one rank only (the core ignores it at --tp > 1).
 * Additive export; RAD_ABI_VERSION does not move. */
int rad_declare_scores(RadBuilder* b, rad_buf hidden, rad_weight table);

/* Which buffer holds this step's pooled vectors, what an embedding model returns instead of
 * logits. Row s is sequence s's whole input pooled into one vector, with whatever pooling the
 * model was trained with (a mean, the last token, a CLS row). The core reads rows [0, n_seq) back
 * after the step and hands each to its request. It does not normalise them; that is the API's
 * choice, not the model's.
 *
 * Declaring it makes the Program an embedding program:
 *   - every sequence in a step is a whole input. The scheduler never splits one across steps,
 *     since a bidirectional encoder cannot be fed half its input, and refuses an input longer
 *     than one step rather than truncating it;
 *   - there is no sampler, drafter or prefix cache, and nothing is generated: a request ends at
 *     the step that carried it.
 *
 * f32, device, [>= max_seqs, dim]: `dim` is the vector width the core reads. A Program declares
 * this or logits, never both. */
int rad_declare_pooled(RadBuilder* b, rad_buf pooled);

/* The most integer words one request may hand the plugin (RadBatch::plugin_words), 0 for none.
 * The engine stages that many a sequence in the step batch and refuses a longer request before
 * it is admitted, so the plugin reads its words at the size it declared. Additive export. */
int rad_declare_plugin_words(RadBuilder* b, int64_t max_words);


/* ================================================================== the drafter */
/* The plugin's draft head. The core drives the extra passes, and only the plugin knows what it
 * is, so it is declared rather than detected from weight names (`mtp.fc.weight`) with the core
 * re-reading the drafter's configuration keys the plugin already read.
 *
 * The kind is a drive shape, not a model. SERIAL runs `depth` dependent passes, each embedding the
 * last one's proposal, so tokens arrive one at a time and a pass that declines ends the chain
 * (MTP, EAGLE, Medusa-serial). BLOCK runs a context pass, then one pass that fills every drafted
 * position at once (DFlash2 and other block-diffusion drafters).
 *
 * `proposal` is the buffer a pass writes its token ids into; 0 means the sampler's token buffer,
 * which suits a head whose logits go through the ordinary sampler chain.
 *
 * Declaring no drafter is the normal case; the scheduler's prompt-lookup drafter still runs. */
typedef enum {
    RAD_DRAFT_NONE   = 0,
    RAD_DRAFT_SERIAL = 1,   /* `depth` dependent passes, one proposal each */
    RAD_DRAFT_BLOCK  = 2,   /* a context pass, then one pass that fills the whole block */
} RadDraftKind;

typedef struct RadDrafterDecl {
    const char* name;            /* "mtp", "dflash2" -- for messages and the graph dump */
    int         kind;            /* RAD_DRAFT_* */
    int         depth;           /* tokens proposed a step; must equal RadBuildCtx::max_spec */

    /* Where a pass leaves its proposal. 0 = the sampler's token buffer. */
    rad_buf     proposal;
    int64_t     proposal_pitch;  /* elements a row in `proposal`; 0 means 1, and for a BLOCK
                                      * drafter `depth` (less is refused) */

    /* BLOCK only. `mask_token` is the id every unfilled block position embeds; a block drafter
     * without one cannot run. `window` is the drafter's own sliding window over its own cache, 0
     * for unbounded. The core stages the block pass against it, so it must match the window the
     * plugin gated its attention with. */
    int32_t     mask_token;
    int64_t     window;

    /* BLOCK only: how many rows the query pass runs over, per sequence. 0 means `depth + 1`.
     *
     * The core stages the block as an anchor (the token the sampler just chose) followed by
     * `block - 1` copies of `mask_token`, and the head runs over the last `depth` rows. That covers
     * both training conventions:
     *
     *   block = depth + 1    the anchor row predicts nothing; each mask row predicts the token at
     *                        its own position. The head sees rows 1..depth. DFlash2.
     *   block = depth        every row, the anchor included, predicts the token at the next
     *                        position (the ordinary next-token relation). The head sees rows
     *                        0..depth-1. DSpark, whose reference takes `hidden[:, -block_size:, :]`.
     *
     * A wrong value is not visible in the output: both run, every draft is verified so the text
     * stays correct, but the proposals shift by one position and acceptance drops to about chance.
     * Hence it is declared by the plugin, which knows how the checkpoint was trained. */
    int64_t     block;
} RadDrafterDecl;

int rad_declare_drafter(RadBuilder* b, const RadDrafterDecl* d);

/* A serial head with no cache of its own: each of its attention layers reads one trunk layer's K
 * and V and writes none. Gemma 4's assistant head, whose layers have only a query projection,
 * reads the last sliding and last global trunk layer. Called once, after rad_declare_drafter.
 * Additive export; RAD_ABI_VERSION does not move.
 *
 * Effect on the serial driver, all following from the head writing nothing:
 *
 *   No history pass, since there is no K and V of its own to fill. The pass carries round 1
 *   alone: one row per drafting sequence, at the committed position, with the token the trunk
 *   chose after it and the trunk's hidden state there (out_ids). A sequence that does not draft
 *   gets no row, and a step with nothing to draft no pass.
 *
 *   Every round runs at that same position over the same context. The trunk's cache past the
 *   committed position holds K and V of drafts the last verify rejected, or nothing, so every
 *   round's `seqused` ends at the committed position; the head's proposals travel forward in its
 *   hidden state and next input token instead.
 *
 * `pos_offset` is the rotary position of those rows relative to the committed position: 0 is the
 * position of the trunk hidden state the head continues from, 1 that of the token it embeds. It
 * moves nothing else (the context ends at the committed position either way), and which one a
 * checkpoint was trained with is the plugin's to know. */
int rad_drafter_shared_kv(RadBuilder* b, int pos_offset);

/* A block drafter whose one pass proposes a tree rather than a chain. Called once, after
 * rad_declare_drafter. `parents` is an i32 buffer at the proposal's pitch: row i's j-th entry is
 * the parent of the proposal's j-th token as a node index, 0 for the token the verify starts from
 * (the bonus token) and j' + 1 for the proposal's j'-th token, with every parent before its child.
 * Additive export; RAD_ABI_VERSION does not move.
 *
 * Each verify then runs the proposal as a tree in the same rows: node j at row j, rotated at the
 * committed end + its depth, its K and V stored at the committed end + j, and every trunk step
 * carries RadBatch::spec_parent, which the plugin hands its attention so a row sees, inside its
 * own block, only its ancestors. The acceptance follows the deepest root path the target agrees
 * with; the core then moves that path's K and V into place in every paged group. A proposal from
 * prompt lookup is verified as the chain-shaped tree.
 *
 * Requirements on the plugin: the trunk's attention honours spec_parent on every decode row; the
 * drafter's context pass stores K and V through the slot mapping only (it is staged per node, the
 * rejected ones at slot -1); and the step is issued ahead, behind which the block driver queues
 * the drafter's passes (the KL mode drafts nothing, so trees do not apply there). */
int rad_drafter_tree(RadBuilder* b, rad_buf parents);

/* The model's routing counts live in `table`: an i32 RAD_BUF_PERSIST device buffer of at least
 * [layers, n_expert], one row per layer that reports (rad_route_report's `layer`, so a head past
 * the trunk counts) and n_expert the model's expert count. Each layer's router writes its counts
 * into its own row and the report names that row as RadRouting::expert_count; the core takes it
 * in place and copies the pass's rows to the host once, at the end of the pass. Without a table
 * the core keeps the rows itself and a report costs one copy dispatch per routed layer on the
 * compute stream, since a plugin cannot bind the core's rows as operands of a recorded pass. A
 * report naming any other buffer is still copied, into the layer's row of the table.
 *
 * A row per layer also means a consumer that reads a layer's counts later in stream order (the
 * file stager, at the layer's first expert op) reads that layer's, which one buffer shared by
 * every layer does not guarantee once a later layer's router has run.
 *
 * Called at most once. The row count is checked when a context binds, where the reporting layers
 * are known. Additive export; RAD_ABI_VERSION does not move. */
int rad_route_table(RadBuilder* b, rad_buf table);

/* Whether this engine's launches chain (RadKernelChainFn, abi/rad_abi.h): the core's decision, on
 * the AQL backend only, which keeps a completion counter per queue, where the model's plugin
 * measured a gain with its speculative drafter (RadArchProbe::chain_pays) and that drafter decodes
 * speculatively. A plugin that places work by whether a kernel waits in its prologue behind the
 * one before it -- a weight read issued on lane 0 right after its GEMM -- asks this, so it follows
 * that decision. Additive export; RAD_ABI_VERSION does not move. */
int rad_chain_on(void);

/* A block drafter that also serves a shorter block: the plugin declared its query pass at `block`
 * rows a sequence as well as at RadDrafterDecl::block, with the same anchor convention, so the
 * core may stage a step's block at either. The short block proposes
 * `block - (RadDrafterDecl::block - depth)` tokens; a tree drafter's (rad_drafter_tree, declared
 * first) a tree of its own over the shorter pass, at most `block - 1` nodes, which needs only that
 * the short block be 2 rows or more. Called once, after rad_declare_drafter.
 *
 * The block is proposed whole, so its length is paid for by every drafting sequence whatever each
 * verifies: at one sequence a long block's extra drafts pay, at many its rows and its weaker first
 * positions cost more than they bring. The core picks per step (--draft-short-from; by default
 * a tree's from two decoding sequences on, a chain's never). */
/* ---- appended in ABI 27 (a block drafter's short block) */
int rad_drafter_short_block(RadBuilder* b, int64_t block);

/* ================================================================== the encoder */
/* A media encoder the plugin runs on an encoder pass (RadBatch::enc). The core drives the pass and
 * reads its result, so the plugin declares it. A plugin that declares none serves text only, and
 * the server refuses a media part by name.
 *
 * This is the shape of the interface, not the model. The core stages patches of `patch_dim`
 * elements and reads back one `n_embd`-wide row per `merge` of them; the tower itself is the
 * plugin's business. How an image or video becomes patches is the processor's (core/mm), which
 * reads its geometry from the container; the core refuses a program whose encoder disagrees with
 * the processor about `patch_dim` or `merge`. */
/* A program declares at most two: one serving images and video, one serving audio. An audio
 * encoder's "patch" is one feature frame (a mel row), its `merge` is frames per output row, and
 * its `max_patches` is RadBuildCtx::max_audio_frames. RadBatch::enc names which one a pass runs. */
enum {
    RAD_MM_IMAGE = 1u << 0,
    RAD_MM_VIDEO = 1u << 1,
    RAD_MM_AUDIO = 1u << 2,
};

typedef struct RadEncoderDecl {
    const char* name;            /* "vision" -- for messages */
    uint32_t    modalities;      /* RAD_MM_* this encoder serves: IMAGE|VIDEO, or AUDIO alone */
    int64_t     patch_dim;       /* elements in one input patch row */
    int64_t     merge;           /* input patches per output row */
    int64_t     max_patches;     /* must equal RadBuildCtx::max_enc_patches (max_audio_frames) */
    int64_t     n_embd;          /* the width of an output row, the model's embedding width */
    /* [max_patches / merge, n_embd] bf16: where a pass leaves its rows. The core reads it after
     * the pass and before any other pass runs, so the planner may share its bytes with buffers
     * no encoder pass touches. */
    rad_buf     out;

    /* ---- appended in ABI 26 (one grid a pass) */
    /* 1: every segment of a pass has one grid (Item segment height), for an encoder whose
     * position tables or kernels are per geometry; the core ends a pass where the grid changes.
     * 0 packs segments of any grid into a pass. */
    int32_t     one_grid;

    /* ---- appended in ABI 32 (runs the language model reads both ways) */
    /* 1: the language model attends the placeholders of one picture or video frame to each other
     * both ways (Gemma 4's use_bidirectional_attention "vision"), text before and after them
     * causally. The core then ends no prefill chunk inside such a run and states each trunk pass's
     * runs (RadBatch::n_bidi). 0 for an audio encoder. */
    int32_t     bidirectional;
} RadEncoderDecl;

int rad_declare_encoder(RadBuilder* b, const RadEncoderDecl* d);

/* ================================================================== the image generator */
/* A program that makes images: a conditioning network reads a prompt (and pictures), a denoiser
 * turns a noise latent into an image latent over a number of steps, and a decoder turns the
 * latent into pixels. The core drives the passes, so the plugin declares them.
 *
 * Declaring one makes the program a generator, driven by passes rather than token steps
 * (RadBatch::gen):
 *
 *   RAD_GEN_PASS_COND     one conditioning sequence (the prompt and any reference pictures)
 *                         encoded into the generator's KV group. Once per request, twice with
 *                         guidance (the negative prompt is its own sequence).
 *   RAD_GEN_PASS_DENOISE  one step of every image the pass carries: each image's latent rows,
 *                         read from and written back to the plugin's state at the image's lane,
 *                         against its conditioning's KV.
 *   RAD_GEN_PASS_DECODE   one image's latent to pixels, in `image_out`, which the core reads back.
 *
 * A program declares exactly one of this, the other generators, logits or pooled vectors.
 *
 * The fields are the model's contract with a client: the prompt it was trained on, the sizes it
 * makes, its sampler's schedule. The core turns a request into passes from them without naming
 * any model. */
enum { RAD_GEN_PASS_COND = 1, RAD_GEN_PASS_DENOISE = 2, RAD_GEN_PASS_DECODE = 3 };

/* How the generator's sequence is laid out; the core builds it from the conditioning prompt.
 *
 *   JOINT_GRID   the prompt's rows after `drop_prefix`, each media slot standing for `slot_rows`
 *                rows that a reference picture's latent tokens take in raster order, then the
 *                image being made. Rotary positions are three components: a text row counts up
 *                in all three; a picture of h x w tokens holds the count in the first, and its row
 *                and column, centred on zero (-(n - n/2) .. n/2 - 1), in the other two, after which
 *                the count moves on by max(h, w). Text rows attend causally, a picture's rows to
 *                each other both ways, every row to every row before it.
 *
 *   JOINT_STEP   the conditioning is not cached as keys and values but rides in every denoising
 *                step beside the image, since the model updates it as it goes. The conditioning
 *                is the prompt's rows after `drop_prefix`, then `pad_token` rows up to a multiple
 *                of `pad_multiple`, then each reference picture's latent tokens in raster order
 *                (the pictures are not in the prompt; `tmpl_image` is unused). No KV group is
 *                declared: the COND pass leaves the conditioning's rows in the plugin's own state,
 *                once for every lane the request's images run in, and each DENOISE entry is that
 *                conditioning followed by the image (RadGenBatch, "the joint layout"). Rotary
 *                positions are four f32 components (t, h, w, l): a text or padding row is (0, 0,
 *                0, l), l its index with `rope_text_l` and 0 without; a picture's token at row h,
 *                column w (from 0) is (0, s h, s w, l) with s `rope_hw_scale`, l -(1 + k) for
 *                the k-th reference picture and 0 for the image being made. */
enum { RAD_GEN_LAYOUT_JOINT_GRID = 1, RAD_GEN_LAYOUT_JOINT_STEP = 2 };

typedef struct RadGeneratorDecl {
    const char* name;             /* "qwen-image" -- for messages */
    int32_t     layout;           /* RAD_GEN_LAYOUT_* */

    /* The conditioning prompt, as text the core tokenises. `{prompt}` is replaced by the user's
     * prompt; in `tmpl_edit`, `{images}` by one `tmpl_image` per reference picture, joined by
     * `tmpl_image_sep`. In `tmpl_image`, `{n}` is the picture's 1-based number and `{image}` marks
     * where the picture's own tokens go (the media processor's run for it, markers included). */
    const char* tmpl_text;
    const char* tmpl_edit;
    const char* tmpl_image;
    const char* tmpl_image_sep;
    /* The tokens of this text at the start of the prompt are encoded but not handed to the
     * generator; the conditioning starts after them (e.g. a fixed system turn). */
    const char* drop_prefix;
    /* Prompt for a transparent image, in the words the model was trained on; `{prompt}` is the
     * user's description. Empty when the model makes no alpha. */
    const char* tmpl_transparent;

    /* Geometry. One latent token covers `latent_scale` x `latent_scale` pixels and holds
     * `latent_channels` values; a side is a multiple of `size_multiple` pixels; an image has
     * `pixel_channels` channels (4: RGBA). A media slot of the conditioning prompt (one encoder
     * output row) stands for `slot_rows` rows of the generator's sequence. */
    int64_t latent_channels;
    int64_t latent_scale;
    int64_t size_multiple;
    int64_t pixel_channels;
    int64_t slot_rows;
    /* The largest image, in latent tokens; the plugin sizes its per-image buffers for it. */
    int64_t max_tokens;
    /* Reference pictures: at most `max_refs` a request, each resized to `ref_area` pixels. */
    int64_t max_refs;
    int64_t ref_area;
    /* The size a request that names none is made at. */
    int64_t default_width, default_height;

    /* The sampler: flow matching, Euler steps from sigma 1 towards 0.
     *   A fixed schedule: `n_sigmas` values, the terminal 0 implied; a request may not change the
     *   step count. Otherwise `default_steps` sigmas from 1 to 1/steps (or off `sigma_grid`'s
     *   table), shifted:
     *     dyn_shift 0   sigma' = shift * s / (1 + (shift - 1) * s)
     *     dyn_shift 1   mu linear in the image's token count, 256 -> base_shift .. max_seq ->
     *                   max_shift; shift_exp 1: exp(mu) / (exp(mu) + (1/s - 1)), 0: mu / (mu + ...)
     *   then, with shift_terminal > 0, stretched so the last sigma is shift_terminal. */
    int64_t       default_steps;
    int32_t       n_sigmas;
    const double* sigmas;
    double        shift;
    int32_t       dyn_shift;
    int32_t       shift_exp;
    double        base_shift, max_shift;
    int64_t       base_seq, max_seq;
    double        shift_terminal;
    /* Guidance a request gets when it names none; 1 is off. */
    double        default_cfg;

    /* `max_lanes` images may be in flight at once, each in a lane of the plugin's own state.
     * `image_out` is [max_tokens * latent_scale^2, pixel_channels] u8 on the device: row-major
     * pixels of the image a decode pass made. */
    int64_t max_lanes;
    rad_buf image_out;

    /* ---- the JOINT_STEP layout (zero for JOINT_GRID) */
    /* The most rows one conditioning lays out (text, padding and reference latents together):
     * the plugin keeps `cond_rows` rows a lane a branch, and a longer prompt is refused. */
    int64_t     cond_rows;
    double      rope_hw_scale;
    int32_t     rope_text_l;
    int32_t     pad_multiple;
    int32_t     pad_token;
    /* Guidance as the model's card states its scale s: 1 when s means cond + s (cond - uncond),
     * which the core runs as s + 1, 0 when it means uncond + s (cond - uncond). `default_cfg` is
     * in the card's terms. */
    int32_t     cfg_original;
    /* The negative prompt guidance uses when a request names none ("" is a prompt); null when a
     * guided request must name its own. */
    const char* default_negative;
    /* 1: a reference picture is shrunk to `ref_area` only when larger and cut to `size_multiple`
     * by a centred crop; 0: resized to `ref_area` at its own aspect. */
    int32_t     ref_shrink_only;

    /* ---- appended in ABI 25 (Anima, SDXL) */
    /* `aux_tokens` nonzero: the conditioning also reads the user's prompt through the container's
     * second tokenizer (rad-convert --tokenizer2), as that tokenizer encodes it alone, special
     * tokens included, and the COND pass carries the ids (RadGenBatch::aux_ids). A prompt whose
     * second tokenisation is longer than `aux_max_tokens` is refused. */
    int64_t     aux_max_tokens;
    /* The unshifted sigmas of a schedule that is not fixed: 0 is diffusers' linspace(1, 1 / steps,
     * steps); N > 0 picks them off a table of N timesteps as ComfyUI's `simple` scheduler does,
     * step i at (N - floor(i N / steps)) / N, each an f32 quotient of the table's own. */
    int32_t     sigma_grid;
    /* A prompt shorter than this many tokens after `drop_prefix` is padded with `pad_token` up to
     * it, the padding read as prompt (a reference tokenizer's minimum length: an empty prompt is
     * one padding token, and stays empty rather than becoming " "). 0: no minimum. */
    int32_t     min_tokens;
    int32_t     aux_tokens;
    /* ---- a discrete-time denoiser (train_steps 0: the flow-matching sampler above)
     * A model trained on `train_steps` timesteps of a variance-preserving schedule, betas from
     * `beta_start` to `beta_end` (RAD_GEN_BETAS_*), sampled as k-diffusion samples it (Stable
     * Diffusion's family). Its schedule is in sigmas, sigma_t = sqrt((1 - abar_t) / abar_t) for
     * the cumulative product abar of 1 - beta; `zero_snr` rescales it to a zero terminal SNR
     * (the last abar 4.8973451890853435e-08, as ComfyUI does). A request names a sampler
     * (RAD_GEN_SAMPLER_*, one of those `samplers` holds as 1 << id, which the plugin's step
     * serves) and a scheduler (RAD_GEN_SCHED_*), which places the request's steps between the
     * table's largest and smallest sigma, 0 last. A DENOISE pass carries each entry's sigma,
     * next and previous sigma in these units, the timestep the denoiser is conditioned on, the
     * sampler and the step (RadGenBatch). n_sigmas, shift and the dynamic shift are unused. */
    double      beta_start, beta_end;
    int32_t     train_steps;
    int32_t     beta_schedule;
    int32_t     zero_snr;
    uint32_t    samplers;
    int32_t     default_sampler;
    int32_t     default_scheduler;

    /* ---- the prompt in windows (a CLIP text encoder's fixed context)
     * Nonzero: the prompt's tokens after `drop_prefix` are cut into runs of at most
     * `chunk_tokens`, each laid out as `chunk_bos`, the run, `chunk_eos`, then `pad_token` up to
     * chunk_tokens + 2 rows, the windows end to end; a prompt with no tokens is one window of
     * the two markers and padding. The JOINT_STEP layout only, with `pad_multiple` 0, and
     * `cond_rows` a whole number of windows. */
    int32_t     chunk_tokens;
    int32_t     chunk_bos, chunk_eos;

    /* ---- appended in ABI 26 (Krea 2's rebalance) */
    /* The RAD_GEN_COND_* options (rad_runtime.h) the plugin's COND pass serves, as a mask; a
     * request asking for another is refused. 0: none. */
    uint32_t    cond_opts;
} RadGeneratorDecl;

/* RadGeneratorDecl::beta_schedule: betas linear in sqrt(beta) (Stable Diffusion's
 * "scaled_linear"), or linear in beta. */
enum { RAD_GEN_BETAS_SCALED_LINEAR = 0, RAD_GEN_BETAS_LINEAR = 1 };
/* A discrete-time model's samplers, as k-diffusion (and ComfyUI) name and define them. */
enum {
    RAD_GEN_SAMPLER_EULER           = 0,   /* "euler" */
    RAD_GEN_SAMPLER_EULER_ANCESTRAL = 1,   /* "euler_ancestral", eta 1 */
    RAD_GEN_SAMPLER_DPMPP_2M        = 2,   /* "dpmpp_2m" */
    RAD_GEN_SAMPLER_COUNT           = 3
};
/* ... and its schedulers, ComfyUI's: "normal" (timesteps evenly spaced, the last at the smallest
 * sigma), "karras" (rho 7), "exponential", "sgm_uniform" (normal without its last point) and
 * "simple" (the table read at even strides). */
enum {
    RAD_GEN_SCHED_NORMAL      = 0,
    RAD_GEN_SCHED_KARRAS      = 1,
    RAD_GEN_SCHED_EXPONENTIAL = 2,
    RAD_GEN_SCHED_SGM_UNIFORM = 3,
    RAD_GEN_SCHED_SIMPLE      = 4,
    RAD_GEN_SCHED_COUNT       = 5
};

int rad_declare_generator(RadBuilder* b, const RadGeneratorDecl* d);

/* ================================================================== adapter key names */
/* How LoRA files written for this model name its linears, for an image or video generator served
 * with runtime adapters (--lora, --lora-dir, --max-loras; core/lora/). Adapter files from training
 * tools and UIs name a module by the model's original code, not by the checkpoint the container
 * was converted from, and wrap it in a prefix of their own: `diffusion_model.blocks.3.attn.wq`,
 * kohya's `lora_unet_blocks_3_attn_wq`, a bare `blocks.3.attn.wq`. Both calls are optional,
 * made before the GEMMs they cover are declared, and only a program that declares a root carries
 * adapters at all. Additive exports; RAD_ABI_VERSION does not move.
 *
 * rad_lora_root declares one network the adapters may reach (the denoiser, a text encoder): every
 * LoRA-capable GEMM over a checkpoint tensor whose name starts with `native` gets an adapter path.
 * `spellings` lists, space separated, the prefixes adapter keys of this network start with in
 * place of `native`; one ending in '_' is kohya's form, in which the rest of the key is the module
 * path with every '.' written '_'. `native` itself is always accepted, and "-" stands for a key
 * with no prefix at all. A key is tried against the longest matching spelling first.
 *
 * rad_lora_alias maps an adapter module that the checkpoint names differently. `from` and `to`
 * are module paths relative to the root's prefixes (no ".weight"), '#' standing for a decimal
 * index that `to` repeats in the same order. Rows [from_lo, from_hi) of the adapter module's
 * output (its lora_B rows; from_hi 0 = all of them) land at rows from `to_lo` of the checkpoint
 * module `to`. Several aliases with one `from` split a fused adapter module over separate
 * checkpoint tensors (a `qkv` over `to_q`, `to_k`, `to_v`); several `from` into one `to` at
 * different `to_lo` place separate adapter modules into one fused tensor. A module path that is
 * the checkpoint's own needs no alias. */
int rad_lora_root(RadBuilder* b, const char* native, const char* spellings);
int rad_lora_alias(RadBuilder* b, const char* native, const char* from, const char* to,
                   int64_t from_lo, int64_t from_hi, int64_t to_lo);

/* ================================================================== the audio generator */
/* A program that makes audio from text: an autoregressive language model writes the piece frame by
 * frame, with classifier-free guidance (each request runs a conditional and an unconditional
 * branch in step), and either a denoiser turns the frames' hidden states into audio latents window
 * by window, which a vocoder turns into samples, or (`voc_chunk` > 0) a vocoder plays the frames
 * themselves a few at a time as they are made. The core drives the passes (RadBatch::gen,
 * RadAudioGenBatch):
 *
 *   RAD_GEN_PASS_PROMPT  the prompt of one or more requests, both branches, as an ordinary causal
 *                        prefill into the KV group: two sequences a request, the conditional one
 *                        first. A long prompt takes several passes, each a chunk of both branches,
 *                        unless the declaration asks for whole prompts (`prompt_whole`). The pass
 *                        that ends a prompt leaves the branches' last hidden states in the
 *                        request's lane.
 *   RAD_GEN_PASS_FRAME   one frame of every request in its autoregressive stage: the plugin samples
 *                        the frame from the lane's last hidden states, keeps what the denoiser
 *                        conditions on, and runs the language model over the frame's embedding
 *                        (one row a branch, appended to that branch's KV). It writes `status` for
 *                        each lane: >= 0 a frame was made, -1 the model ended the piece instead.
 *   RAD_GEN_PASS_WINDOW  one denoising step of one or more windows, each a run of a request's frames
 *                        (two entries a window with guidance, conditional and unconditional).
 *   RAD_GEN_PASS_VOCODE  one finished window's latents to samples, in `audio_out`, which the core
 *                        reads back; with `voc_chunk`, the frames [voc_frame0, voc_frame0 +
 *                        voc_latents) of one lane, of which the first `voc_left` are context and
 *                        only the rest are played.
 *
 * Windows overlap: window k covers frames [k win_hop, k win_hop + win_frames) and is stepped once
 * the language model has made all of them, or ended. A window joins its predecessor through the
 * latents they share, so a request's windows run in order, and window k + 1 does not start before
 * window k is vocoded. The last window ends at the last frame; there is a window k >= 1 only while
 * frames remain past k win_hop + (win_frames - win_hop), so no window is all overlap.
 *
 * A program declares exactly one of this, the other generators, logits or pooled vectors. */
enum { RAD_GEN_PASS_PROMPT = 4, RAD_GEN_PASS_FRAME = 5, RAD_GEN_PASS_WINDOW = 6,
       RAD_GEN_PASS_VOCODE = 7 };

typedef struct RadAudioGenDecl {
    const char* name;             /* "minimax-music3" -- the served model id and messages */

    /* The prompt, as text the core tokenises: `{instructions}` is replaced by the request's
     * instructions (a piece's description, a voice's) and `{input}` by its input text (lyrics, the
     * words to speak), each first rewritten by the plugin's rad_arch_text_prepare when it exports
     * one. Special tokens are spelled as text. NULL for a model that takes no instructions, which
     * then declares `tmpl_plain` and has a request's instructions refused. */
    const char* tmpl;
    /* The unconditional branch is the conditional prompt with every token but the first
     * `neg_keep_head` and the last `neg_keep_tail` replaced by `neg_token`, so the two branches
     * are the same length and step together. */
    int32_t     neg_token;
    int32_t     neg_keep_head, neg_keep_tail;
    /* The longest prompt in tokens, and the most frames a request makes. */
    int64_t     max_prompt;
    int64_t     max_frames;
    /* frame_rate_num / frame_rate_den frames a second. */
    int64_t     frame_rate_num, frame_rate_den;

    /* Windows: `win_frames` frames, a new one every `win_hop`. A window of n frames has
     * floor(n * lat_num / lat_den) latents of `latent_channels`, and each latent is
     * `samples_per_latent` samples of `channels` channels at `sample_rate`. Where two windows meet,
     * the earlier one's last `win_crop_right` latents and the later one's first `win_crop_left` are
     * not played; the first window keeps its start and the last its end. */
    int64_t     win_frames, win_hop;
    /* Rows the denoiser puts ahead of a window's latents in each entry of a WINDOW pass (a
     * timestep token); the core counts them into the pass's rows. */
    int64_t     win_lead;
    int64_t     lat_num, lat_den;
    int64_t     latent_channels;
    int64_t     win_crop_left, win_crop_right;
    int64_t     samples_per_latent;
    int64_t     sample_rate;
    int64_t     channels;

    /* The denoiser's sampler: `default_steps` Euler steps of flow matching from t = 0 (noise) to
     * t = 1 (data), t_i = i / steps; guidance `default_cfg` (1 is off), v = u + cfg (c - u). */
    int64_t     default_steps;
    double      default_cfg;

    /* `max_lanes` requests may be in flight at once, each in a lane of the plugin's state.
     * `status` is [>= max_lanes] i32 on the device, indexed by lane (see FRAME above); `audio_out`
     * is [>= the samples of the longest window, channels] f32 on the device, interleaved, in
     * [-1, 1]: what a VOCODE pass plays, its crops already applied. */
    int64_t     max_lanes;
    rad_buf     status;
    rad_buf     audio_out;

    /* The prompt of a request without instructions, `{input}` its only slot; NULL makes the
     * instructions required. */
    const char* tmpl_plain;
    /* The unconditional branch's own template, rendered from the request's texts as `tmpl` is;
     * NULL derives it from the conditional prompt by `neg_token` as above. With it the branches
     * may differ in length, and a request whose guidance is 1 runs the conditional prompt in both
     * (the guided logits are then the conditional ones). */
    const char* neg_tmpl;
    /* Nonzero: a prompt is never split across PROMPT passes (an encoder that reads it at once);
     * the pass rows must hold both branches of the longest. Required with `neg_tmpl`. */
    int32_t     prompt_whole;
    /* The first frame a request keeps: frames before it only move the model past the prompt. */
    int32_t     first_frame;
    /* Nonzero: no denoiser and no windows. The vocoder plays the kept frames as they are made, at
     * most `voc_chunk` a VOCODE pass, reading `voc_context` frames before them again (and not
     * playing them) so that a causal vocoder needs no state between passes. A frame is then one
     * latent: `lat_num` = `lat_den` = 1, `samples_per_latent` samples a frame. */
    int64_t     voc_chunk;
    int64_t     voc_context;

    /* ---- appended in ABI 27 (one-branch speech over a hybrid model, references) */
    /* Sequences a request runs: 0 or 2, the conditional and unconditional branches above; 1, the
     * conditional one alone, for a model without guidance (no unconditional prompt is rendered,
     * a request's guidance must be 1, and every per-entry array of a PROMPT or FRAME pass has one
     * entry a lane, its `slot` the lane itself). The KV, the prompt's rows and a frame's rows then
     * cost half. */
    int32_t     branches;
    /* Nonzero: the `{input}` slot is tokenised with its special tokens, which the template's own
     * text always is, so a caller can write the model's speaker tags. The plugin's
     * rad_arch_text_prepare then refuses the control tokens a caller must not write. */
    int32_t     input_special;
    /* References (voice cloning): with `tmpl_ref`, a request may carry up to `ref_max` of them,
     * each a transcript and a clip of the voice saying it, which the plugin encodes into frames
     * in the request's PROMPT pass (whole prompts only: `prompt_whole`). `tmpl_ref` is the prompt
     * of a request with references, `{references}` each of them rendered from `ref_item` in turn
     * and `{input}` as in `tmpl_plain`. `ref_item` is one reference: `{index}` its number from 1,
     * `{text}` its transcript (rewritten by rad_arch_text_prepare as RAD_TEXT_REFERENCE) and
     * `{audio}` its frames, one prompt row each (RadAudioGenBatch::ref_frame); the text before
     * `{audio}` is tokenised as one segment with its special tokens (the transcript's own, which
     * prepare() has limited to those a caller may write), and so is the text after it. A clip is
     * mono f32 in [-1, 1] at `ref_rate`, right-padded with zeros to a whole number of `ref_hop`
     * samples, each `ref_hop` samples one frame; a request's clips hold at most `ref_max_samples`
     * samples together. With two branches, references require default_cfg 1 and a plain
     * neg_tmpl: guidance 1 repeats the conditional prompt and frame map in both branches.
     * Reference direction is opt-in through the templates appended below. NULL `tmpl_ref`:
     * references are refused. */
    const char* tmpl_ref;
    const char* ref_item;
    int64_t     ref_rate;
    int64_t     ref_hop;
    int64_t     ref_max_samples;
    int32_t     ref_max;
    /* ABI 40: reference-plus-instructions uses tmpl_ref_instruct ({references},
     * {instructions}, {input} once each). Its guidance baseline is neg_tmpl_ref
     * ({references}, {input}, no instructions). Both branches retain the same clips;
     * their text lengths and reference-frame maps may differ. NULL: direction refused. */
    const char* tmpl_ref_instruct;
    const char* neg_tmpl_ref;
} RadAudioGenDecl;

int rad_declare_audio_generator(RadBuilder* b, const RadAudioGenDecl* d);

/* ================================================================== the video generator */
/* A program that makes a video and its soundtrack from text: a text encoder reads the prompt, one
 * denoiser steps a packed sequence holding the prompt's rows, the soundtrack's latents and the
 * video's latents together (flow matching, one forward a step, no guidance), and two decoders turn
 * the latents into frames and samples. The core drives the passes (RadBatch::gen,
 * RadVideoGenBatch), one request a pass:
 *
 *   RAD_GEN_PASS_VCOND    the request's prompt, an ordinary non-paged pass over its tokens: the
 *                         text encoder, and what the denoiser reads of the prompt at every step,
 *                         into the request's lane. It carries the request's whole schedule, so
 *                         what depends only on a step's timesteps can be computed once here.
 *   RAD_GEN_PASS_VSTEP    one denoising step of one lane over the packed sequence
 *                         [text | audio | video]: the prompt's rows, then the soundtrack's
 *                         `audio_channels` blocks of latents (channel-major, one row a latent),
 *                         then the video's patches (frame-major, then row-major). The first step
 *                         draws the lane's noise.
 *   RAD_GEN_PASS_ADECODE  the lane's soundtrack to samples in `audio_out`, which the core reads
 *                         back. Runs once, after the last step.
 *   RAD_GEN_PASS_VDECODE  one chunk of the lane's video latents to frames in `frames_out`, which
 *                         the core reads back. Chunks run in order after the soundtrack; each joins
 *                         its predecessor through state the plugin keeps in the lane.
 *
 * Geometry. A video of n chunks has `frames_per_chunk n + frames_tail` frames at
 * `fps_num / fps_den` a second and `latents_per_chunk n + latents_tail` latent frames, each of
 * (height / latent_scale) x (width / latent_scale) latents of `latent_channels`; the denoiser
 * reads them in `patch` x `patch` patches. Chunk c of n is `frames_per_chunk` frames, the last
 * `frames_per_chunk + frames_tail`. A side is a multiple of `size_multiple` pixels, at least
 * `min_side`, the area at most `max_area`, and the long side over the short at most `max_aspect`.
 * The soundtrack has round(frames / fps * audio_rate) latents a channel, each
 * `sample_rate / audio_rate` samples.
 *
 * The rotary positions of a pass's rows are the plugin's (rad_arch_video_positions), which a video
 * generator must export.
 *
 * The sampler: a request's `steps` are sigma grid points, the terminal 0 included, as diffusers'
 * MiniMax-H3 counts them: sigma_i = shift s / (1 + (shift - 1) s) over s = linspace(1, 0, steps)
 * in f32 (consecutive duplicates dropped), and a model evaluation at each timestep
 * t_i = 1 - sigma_i but the last; one schedule for the video (`shift_video`) and one for the
 * soundtrack (`shift_audio`), stepped together. A few-step (distilled) checkpoint instead declares
 * the base times it was trained on (`rungs`, below): s = rungs then 0, each taken to f32, the same
 * shifts applied, and the step count fixed at n_rungs + 1.
 *
 * Latent upscaling, for a plugin with a network that enlarges a video latent (`upscale_max` > 1):
 * a request with a factor f > 1 is denoised at a smaller size first. Its video is W x H; each side
 * divided by f and rounded to the nearest multiple of `size_multiple` (at least one multiple, at
 * most the side) gives the small size W' x H'. The request's evaluations but its last `refine` run
 * at W' x H', the last of them stepping to sigma 0 (its clean estimate) instead of the next grid
 * point. Then one pass:
 *
 *   RAD_GEN_PASS_VUPSCALE  the lane's video latent from W' x H' to W x H through the plugin's
 *                          network, told the factor (W / W' + H / H') / 2; then both latents
 *                          re-noised to the sigmas of the first evaluation at W x H, each with a
 *                          stream of its own seed other than the one its first step drew.
 *
 * The last `refine` evaluations run at W x H, and the decodes follow as for any request. Every
 * pass carries the geometry its latent has at that point: a step before the VUPSCALE pass is at
 * W' x H'. The schedule is the request's own, whatever size each evaluation runs at.
 *
 * Reference conditioning, for a plugin with the VAE encoders (`max_enc_side`, `max_enc_samples`,
 * below): before the prompt pass each of a request's reference media and keyframes is encoded,
 * one a pass, into the lane, where the steps read it. Nothing an encode makes is read back.
 *
 *   RAD_GEN_PASS_VENCODE  one picture or video, its frames in `enc_pixels` (RadVideoGenBatch),
 *                         through the video VAE's encoder.
 *   RAD_GEN_PASS_AENCODE  one soundtrack, its samples in `enc_samples`, through the audio VAE's
 *                         encoder.
 *
 * `enc_kind` (RAD_ENC_*, rad_runtime.h) and `enc_slot`, the media's 1-based number within its kind,
 * say where the latents go. A latent-upscaled request's keyframes, pinned on the grid its steps
 * run on, are encoded at W' x H' before the prompt pass, then again at W x H into the same slots
 * after the VUPSCALE pass and before the first evaluation at W x H. References, on their own grid,
 * are encoded once. The media's shape is the plugin's to accept: rad_arch_encode_geom
 * (below) refuses what the encoders cannot take, and the step's packed sequence with the encoded
 * media in it is the plugin's rad_arch_video_pack.
 *
 * A program declares exactly one of this, the other generators, logits or pooled vectors. */
enum { RAD_GEN_PASS_VCOND = 8, RAD_GEN_PASS_VSTEP = 9, RAD_GEN_PASS_ADECODE = 10,
       RAD_GEN_PASS_VDECODE = 11, RAD_GEN_PASS_VUPSCALE = 12,
       RAD_GEN_PASS_VENCODE = 13, RAD_GEN_PASS_AENCODE = 14 };

typedef struct RadVideoGenDecl {
    const char* name;             /* "minimax-h3" -- the served model id and messages */

    /* The prompt as text the core tokenises, `{prompt}` replaced by the request's; special tokens
     * are spelled as text. No beginning-of-sequence token is added. */
    const char* tmpl;
    int64_t     max_text;         /* the longest prompt, in tokens */

    /* Geometry (above). */
    int64_t     fps_num, fps_den;
    int64_t     frames_per_chunk, frames_tail;
    int64_t     latents_per_chunk, latents_tail, dec_latents;
    int64_t     min_chunks, max_chunks;
    int64_t     latent_scale, latent_channels, patch;
    int64_t     size_multiple, min_side, max_area, max_aspect;
    /* The size and length a request that names none is made at. */
    int64_t     default_width, default_height, default_chunks;

    /* The soundtrack. */
    int64_t     sample_rate, audio_rate, audio_channels, audio_latent_channels;

    /* The sampler (above). */
    int64_t     default_steps, max_steps;
    double      shift_video, shift_audio;

    /* `max_lanes` requests may hold state at once. `frames_out` is [>= the frames of the largest
     * chunk, height, width, 3] u8 on the device, the frames a VDECODE pass made, row-major RGB;
     * `audio_out` is [>= the samples of the longest soundtrack, audio_channels] f32, interleaved,
     * what an ADECODE pass made. */
    int64_t     max_lanes;
    rad_buf     frames_out;
    rad_buf     audio_out;

    /* A trained schedule when `n_rungs` > 0: base times strictly decreasing in (0, 1] (a
     * trainer's timesteps over its 1000-step clock), each modality's sigmas their shifts as above.
     * `default_steps` and `max_steps` are then both n_rungs + 1 and a request may not name another
     * count. 0 is the linspace schedule. */
    int32_t       n_rungs;
    const double* rungs;

    /* Latent upscaling (above): the largest factor a request may ask for, 0 when the plugin has no
     * upscaler; and the evaluations at the full size a request that names none runs, at least 1
     * and fewer than the default schedule's. */
    double        upscale_max;
    int32_t       upscale_refine;

    /* ---- appended in ABI 24 (LTX-2.5) */
    /* The sampler's form. 0: the grid points above. 1: diffusers' FlowMatchEulerDiscrete as
     * LTX-2 runs it -- a request's `steps` are its evaluations n, between 1 and max_steps; sigmas
     * linspace(1, 1/n, n), computed in f64 and taken to f32, and with `dyn_shift` 1 shifted by mu
     * as exp(mu) / (exp(mu) + 1/s - 1) in f32, mu linear in the video's latent tokens from
     * `base_seq` (`base_shift`) to `max_seq` (`max_shift`); then, with `shift_terminal` > 0,
     * stretched so the last is shift_terminal; and a closing 0. A trained schedule (n_rungs > 0)
     * is the rungs taken to f32 and a closing 0, with default_steps and max_steps both n_rungs.
     * Both modalities step the same sigmas; shift_video and shift_audio are unused. */
    int32_t       sched_kind;
    int32_t       dyn_shift;
    double        base_shift, max_shift;
    int64_t       base_seq, max_seq;
    double        shift_terminal;
    /* With latent upscaling under a trained schedule of form 1: the sigmas the evaluations after
     * the upscale step through (a second stage's own schedule, its first the sigma the upscaled
     * latents are re-noised to), instead of the schedule's tail. The small size then runs every
     * rung, its last to the clean estimate, and a request's refinement is n_refine_rungs. */
    int32_t       n_refine_rungs;
    const double* refine_rungs;
    /* The text rows of every VSTEP pass, whatever the prompt's length (0: the prompt's tokens):
     * a model whose text encoder writes a fixed number of rows. */
    int64_t       step_text_rows;
    /* Guidance terms (RadVideoGenBatch::guidance_h) when `guidance` is 1: the defaults a request
     * that names none gets, the video's scale, spatio-temporal scale, modality scale and rescale,
     * then the soundtrack's four. A request whose video or soundtrack scale is not 1 is guided
     * against a negative prompt: its own, else `default_negative` (null: ""). The VCOND pass then
     * carries both prompts, the positive first (two cu_seqlens entries). */
    int32_t       guidance;
    double        guidance_default[8];
    const char*   default_negative;
    /* Image to video: 1 when the model can start from a first frame (OpenAI's
     * `input_reference`), handed to the VCOND pass resized to the size the first evaluations run
     * at (RadVideoGenBatch `image`), after an H.264 round trip at `image_crf` when that is > 0
     * (the compression the model was trained against). */
    int32_t       image_cond;
    int32_t       image_crf;
    /* A left-padded text encoder: the VCOND pass's positions count so that each prompt ends at
     * `text_left_pad` - 1, the place it holds in a sequence padded on the left to that length (0:
     * from 0). */
    int64_t       text_left_pad;
    /* The soundtrack's latent rows a step carries for each latent (0: audio_channels, a stream of
     * rows a channel); a model whose latent holds every channel at once carries 1. */
    int32_t       audio_streams;
    /* A latent upscaler of one fixed factor, upscale_max: a request's sides must then be
     * multiples of size_multiple x upscale_max, so the small size is exactly the full size over
     * the factor. */
    int32_t       upscale_exact;
    /* A length head: a model that predicts from the prompt how long its video should be. With
     * `duration_out` set, every VCOND pass writes to it (f32 [1], device) the natural log of that
     * length in seconds. For a request that names no length the core takes its exponent, clamps it
     * to [duration_min, duration_max] seconds, rounds it to frames at the frame rate and floors
     * that to the chunk grid (frames_per_chunk n + frames_tail; one chunk more when the floor falls
     * under duration_min and duration_max allows it), then holds it to min_chunks..max_chunks and
     * to what a pass carries. Until that VCOND pass is done such a request is shaped at
     * `default_chunks`, so the pass may not depend on the length: its geometry carries the
     * provisional frame count, and `t_video` / `t_audio` the provisional schedule. 0: a request
     * that names no length is made at default_chunks. `duration_dtype` is the precision the model
     * takes the exponent in: RAD_F32, or RAD_BF16 for a head run in bf16, whose length is
     * exp(x) in f32 rounded to bf16. Frames round half to even, as Python's round(). */
    rad_buf       duration_out;
    double        duration_min, duration_max;
    uint32_t      duration_dtype;
    /* Decoders: a model with more than one way to turn its latents into frames names them,
     * space-separated, its default first (LTX-2.5: "vae diffusion"). A request may name one; every
     * VDECODE pass carries its index (RadVideoGenBatch::decoder). `decoder_max_area`, when set,
     * holds each decoder's largest frame in pixels (0: max_area), and a request over it is refused.
     * Null: one decoder. */
    const char*    decoders;
    const int64_t* decoder_max_area;

    /* ---- appended in ABI 31 (MiniMax-H3 reference conditioning, above) */
    /* The reference media a request may carry: pictures, videos (each with an optional soundtrack
     * of its own) and standalone soundtracks. 0 for none of a kind; a plugin that declares any
     * exports rad_arch_video_pack. */
    int64_t       max_ref_images, max_ref_videos, max_ref_audios;
    /* What one encode pass carries: a video's frames after rad_arch_encode_geom pads them to whole
     * clips, a frame's longer side and a soundtrack's samples. `max_enc_side` 0: no video encoder,
     * `max_enc_samples` 0: no audio encoder. A picture's sides are multiples of `enc_ratio`. */
    int64_t       max_enc_frames, max_enc_side, max_enc_samples;
    int32_t       enc_ratio;
    /* The distinct modulations a step's groups may read (rad_arch_video_pack's `mod`, the highest
     * cell + 1); a request needing more is refused at submit. 0: no limit. */
    int32_t       max_mod_cells;
} RadVideoGenDecl;

int rad_declare_video_generator(RadBuilder* b, const RadVideoGenDecl* d);

/* ================================================================== the shape generator */
/* A program that makes a 3-D shape from one picture: an image encoder reads the picture, a
 * denoiser steps a set of latent rows by flow matching, guided against an unconditioned branch,
 * and a decoder turns the latent into an occupancy field that the core samples at points of its
 * choosing and meshes. The core drives the passes (RadBatch::gen, RadShapeGenBatch), one request
 * a pass:
 *
 *   RAD_GEN_PASS_SCOND    the request's picture through the image encoder into the request's
 *                         lane: what the denoiser reads of it at every step, for both branches
 *                         (the picture's and the unconditioned one).
 *   RAD_GEN_PASS_SSTEP    one denoising step of one lane over both branches, `step_rows` rows:
 *                         each branch's conditioning and latent, laid out as the plugin likes, the
 *                         latents two entries of `latent_rows` (the conditioned first) for the
 *                         image generator's sampler ops. The first step draws the lane's noise.
 *   RAD_GEN_PASS_SLATENT  the lane's final latent through the decoder's network, into the lane:
 *                         what every query pass of the request reads. Runs once, after the last
 *                         step.
 *   RAD_GEN_PASS_SQUERY   the field at `n_query` points (RadShapeGenBatch::query, at most
 *                         `max_query`), one f32 a point in `field_out`, which the core reads
 *                         back.
 *
 * The picture. The core decodes it with its alpha (opaque without one) and prepares it as
 * Hunyuan3D-2's ImageProcessorV2 and image encoder do: the alpha's bounding box (its last row and
 * column left out, as the reference's slice leaves them) scaled with OpenCV's INTER_AREA to
 * (1 - border_ratio) of the longer side of the square that holds the picture, centred on it and
 * flattened onto white; that square resized to `canvas` x `canvas` with INTER_CUBIC; then as 0..1
 * values bilinearly resized to `enc_size` x `enc_size` (half-pixel centres, torchvision's), each
 * channel normalised by `pix_mean` and `pix_std`, and cut into (enc_size / enc_patch)^2 patches of
 * enc_patch x enc_patch pixels, row-major, each patch's values channel-major then row-major (a
 * convolution's weight order). An SCOND pass carries those patches.
 *
 * The sampler: a request's `steps` are sigma grid points, sigma_i = i / (steps - 1) for i in
 * [0, steps), numpy's linspace(0, 1) taken to f32; the latent starts as noise at sigma 0 and an
 * evaluation at each sigma but the last steps it to the next (the reference evaluates at the
 * last too, against itself, a step of zero). The model's timestep is sigma, the guided velocity
 * neg + cfg (pos - neg), and the step x + (sigma_next - sigma) v.
 *
 * The field: the decoder's logit at a point, inside the shape above `level`. The core samples the
 * cube [-bounds, bounds]^3 on a grid of `resolution` cells a side ((resolution + 1)^3 points,
 * numpy's linspace in f32 along each axis), coarse to fine as FlashVDM's hierarchical decoding
 * does: the whole grid at the coarsest level (resolution halved while it stays at least 63), then
 * at each finer level only the points near where the coarser one crosses `level` or is within
 * 0.95 of it. The surface is marching cubes over the finest grid, a vertex at grid index g placed
 * at g 2 bounds / (resolution + 1) - bounds, as the reference places it.
 *
 * A program declares exactly one of this, the other generators, logits or pooled vectors. */
enum { RAD_GEN_PASS_SCOND = 15, RAD_GEN_PASS_SSTEP = 16, RAD_GEN_PASS_SLATENT = 17,
       RAD_GEN_PASS_SQUERY = 18 };

typedef struct RadShapeGenDecl {
    const char* name;             /* "hunyuan3d-2" -- the served model id and messages */

    /* The picture (above). */
    int64_t     canvas;
    double      border_ratio;
    int64_t     enc_size, enc_patch;
    double      pix_mean[3], pix_std[3];

    /* The latent: `latent_rows` rows of `latent_channels` a branch; a step's rows, both branches
     * with their conditioning. */
    int64_t     latent_rows, latent_channels, step_rows;

    /* The sampler (above) and the guidance a request that names none runs at. */
    int64_t     default_steps, max_steps;
    double      default_guidance;

    /* The field (above): the cube's half side, the iso value, and the grid's resolution a request
     * that names none is sampled at, with the range a request may name. */
    double      bounds;
    double      level;
    int64_t     default_resolution, min_resolution, max_resolution;

    /* `max_lanes` requests may hold state at once. `field_out` is a device f32 buffer of at least
     * `max_query` elements. */
    int64_t     max_lanes;
    int64_t     max_query;
    rad_buf     field_out;
} RadShapeGenDecl;

int rad_declare_shape_generator(RadBuilder* b, const RadShapeGenDecl* d);

/* ================================================================== diagnostics */
/* A plugin may annotate the graph dump. Free-form, printed under --debug-graph. */
void rad_note(RadBuilder* b, const char* fmt, ...);

/* ================================================================== the arch plugin exports */
const char* rad_arch_id(void);

/* The quantisation this plugin serves: the second half of the key that selects it.
 *
 * spec §2.4 is one plugin per (architecture, quantisation): a w4a8 model wants
 * rmsnorm_had_quant_i8 where a bf16 model wants a plain rmsnorm, and branching on a quant
 * descriptor inside one plugin obscures what actually runs. So `qwen35_bf16` and `qwen35_fp8` both
 * answer "qwen35" to rad_arch_id and are told apart here, by RadModelMeta::quant.
 *
 * The match is exact, with no fallback. Unquantised containers and plugins both say ""; the
 * symbol is optional and a plugin without it reads as "". A fallback to a dense plugin would fail
 * later at the name map with a worse message than "no plugin claims qwen35 at fp8_e4m3; qwen35 is
 * served at: (unquantised)". */
const char* rad_arch_quant(void);

/* Whether this plugin may serve a checkpoint whose quantisation was imported weight by weight
 * (Quark, ParoQuant, GPTQ, NVFP4, ...). Such a descriptor names the producing tool, which no plugin
 * claims, so the family's one quantised plugin serves it, and two are refused (startup.h). A
 * family's second quantised plugin that reads only its own containers returns 0 and leaves imports
 * to the first. Optional: absent or non-zero is yes. Additive export; RAD_ABI_VERSION does not
 * move. */
int rad_arch_serves_imports(void);

/* Nonzero when every piece of per-instance state the plugin keeps is indexed by
 * rad_build_slot() / rad_slot(), so several engines in one process may run it at once
 * (docs/MULTIMODEL.md §3.3). Optional: absent or zero, the core runs the plugin for one engine at
 * a time and refuses a second engine on it by name. arch/common/rad_arch.h writes it with
 * RAD_ARCH_MULTI_INSTANCE(). */
int rad_arch_multi_instance(void);

/* What the plugin can say before it declares anything. Optional: a plugin that does not export
 * it answers zero to everything, each field's default.
 *
 * The main case is the draft depth, a model property that is an input to declaration:
 * `RadBuildCtx::max_spec` sizes buffers, bands kernels and sets the scheduler's window, so it is
 * fixed before rad_arch_declare runs, yet only the plugin reads the drafter's configuration. 0
 * means the container carries no drafter.
 *
 * --num-speculative-tokens overrides the answer, and the plugin sees the result in
 * RadBuildCtx::max_spec; a plugin whose checkpoint cannot serve that depth refuses in declare. */
typedef struct RadArchProbe {
    int draft_depth;

    /* Nonzero when `draft_depth` is the only depth the drafter serves. A serial head's depth is a
     * throughput operating point (each extra pass costs one pass and acceptance decays, but it
     * still runs). A block drafter's depth is the block length its checkpoint was trained at (its
     * convolution masks the position within the block), so any other depth is refused.
     *
     * `--num-speculative-tokens auto` takes `draft_depth` either way. rad-convert does not: a
     * container carries everything the model can serve, so it widens to `draft_depth` only when
     * this says the drafter has no choice; otherwise a deployment wanting a deeper serial draft
     * would have to convert again. */
    int draft_depth_fixed;

    /* The plugin honours RadBuildCtx::shape_probe: a sizing declare leaves the state its step
     * function reads untouched. Zero means the core does not ask, and the activation arena is not
     * lent. */
    int shape_probe_ok;

    /* The encoder's merge, input patches per output row (RadEncoderDecl::merge), when the plugin
     * declares an encoder. A pass carries whole output rows, so the core rounds the pass size it
     * picks itself (`--mm-max-patches auto`) down to a multiple of this; an operator-named size is
     * kept, and the plugin refuses one that does not divide. 0 is no opinion. */
    int64_t enc_merge;

    /* The longest clip the audio encoder takes, in feature frames (a whole number of its merge);
     * the engine sizes the audio pass with it (RadBuildCtx::max_audio_frames). 0 is no audio. */
    int64_t audio_frames;

    /* Nonzero for a generator (rad_declare_generator, rad_declare_audio_generator): the rows its
     * largest pass carries. The engine sizes the step for it before declare (an image's denoise
     * pass carries all of its latent rows at once, twice with guidance; an audio window's step all
     * of the window's) and turns off what a generator does not use. 0 is a program that
     * generates text or embeds. */
    int64_t gen_rows;
    /* The most patches the generator's conditioning pass encodes: every reference picture a
     * request may carry, at the model's resize target. This is the encoder's pass size
     * (max_enc_patches). */
    int64_t gen_patches;
    /* Nonzero for an audio generator, which the engine sizes before declare as it does an image
     * generator, with these differences: two sequences a request (its branches), and `gen_ctx`
     * the longest a sequence grows (the whole prompt and every frame), which is the context the
     * KV group is sized for. */
    int32_t gen_audio;
    int64_t gen_ctx;
    /* Nonzero for a video generator, which the engine sizes as it does an image generator from
     * `gen_rows` (its largest pass: a step's packed sequence at the largest size and length). It
     * keeps no KV. */
    int32_t gen_video;
    /* Nonzero when the plugin serves an E4M3 cache for this model (--kv-cache-dtype fp8,
     * RadBuildCtx::kv_dtype RAD_F8E4M3): GLM-5.3's latent attention. The engine refuses the flag
     * for a plugin that leaves it zero, naming the int8 cache every paged attention serves. */
    int32_t kv_fp8_ok;

    /* ---- appended in ABI 26 (chained launches, control vectors) */
    /* How this plugin measured chained launches (RadKernelChainFn: a dependent kernel waits in its
     * own prologue rather than behind the queue's barrier bit), a RAD_CHAIN_*. SPEC: its
     * speculative step is faster with them, and the core takes them on the AQL backend while a
     * drafter decodes speculatively. ALWAYS: its plain decode is faster too, and the core takes
     * them with or without a drafter. Zero keeps them off: some models' plain decode measured
     * slower chained, and a plugin that never measured them asks for nothing. */
    int32_t chain_pays;
    /* ---- appended in ABI 30 (KV streaming) */
    /* Nonzero when the architecture can stream a KV group between host memory and a VRAM ring: its
     * attention is sparse and block-addressed and it declares a streamed store and `kv_resolve`.
     * --kv-resident on an architecture that says 0 is turned off at startup, and nothing streamed
     * is declared (RadBuildCtx::kv_stream_ok). */
    int32_t kv_stream_ok;
    /* Nonzero when the plugin applies RadBuildCtx::cvec for this model. --control-vector is
     * refused at startup with the plugin named when this is 0. */
    int32_t control_vector_ok;

    /* ---- appended in ABI 27 (row-split prefill attention) */
    /* Nonzero when the plugin serves RadBuildCtx::tp_attn_rows for this model. --tp-attn-split
     * rows is refused at startup with the plugin named when this is 0. */
    int32_t tp_attn_rows_ok;

    /* ---- appended in ABI 27 (recurrent state at f16) */
    /* Nonzero when the plugin declares its recurrent state at f16 under RadBuildCtx::state_dtype
     * and its kernels read that width. --state-cache-dtype f16 is refused at startup with the
     * plugin named when this is 0. */
    int32_t state_f16_ok;

    /* ---- appended in ABI 31 (tree verification) */
    /* Nonzero when this plugin can grow its block drafter's proposals into a tree
     * (rad_drafter_tree) and measured its speculative step faster with one. The core then asks for
     * a tree (RadBuildCtx::draft_tree) wherever that drafter decodes speculatively. Zero keeps the
     * chain: a plugin that never measured trees asks for none. 64-bit so it starts past the older
     * struct's tail padding, which a plugin built against it may copy over. */
    int64_t tree_pays;

    /* ---- appended in ABI 32 (inputs: an image generator's bounds) */
    /* Set by the core before the call: --image-gen-max-pixels and --image-gen-max-refs, as
     * RadBuildCtx carries them to declare (0: the model's own). A generator's gen_rows and
     * gen_patches answer for them. */
    int64_t gen_max_pixels;
    int64_t gen_max_refs;

    /* ---- appended in ABI 33 (the shape generator) */
    /* Nonzero for a shape generator, which the engine sizes as it does a video generator from
     * `gen_rows` (its largest pass: a step's two branches, or a query pass's points). It keeps no
     * KV. */
    int32_t gen_shape;
} RadArchProbe;

int rad_arch_probe(const RadModelMeta* meta, RadArchProbe* out);

/* How this model spells a turn, as data. Optional: without it the engine derives the format by
 * analysing the chat template.
 *
 * One streaming parser driven by this description splits a reply into reasoning, content and tool
 * calls, and the lazy grammar that constrains a call and its trigger words are generated from it,
 * so a model whose format the template analysis misses or gets wrong is supported by stating it
 * here. It describes the tagged family:
 *
 *   [reasoning_open] reasoning [reasoning_close]  content
 *   [call_open] name_prefix NAME name_suffix
 *       arg_prefix ARG arg_name_suffix arg_value_prefix VALUE arg_value_suffix  ...
 *   call_end [call_close]
 *
 * An argument's tag may also say its value's type, with arg_json_value_prefix written in place of
 * arg_value_prefix before a JSON value. DeepSeek-V4.1's DSML, inside a section
 * "<｜DSML｜ calls>" / "</｜DSML｜ calls>", is "<｜DSML｜ invoke name=\"" / "\">\n" / "</｜DSML｜ invoke>",
 * arguments "<｜DSML｜ parameter name=\"" / "\"" / " string=\"true\">" (or
 * " string=\"false\">") / "</｜DSML｜ parameter>".
 *
 * Qwen3.8, for example, is "<think>\n" / "\n</think>\n\n", calls "<tool_call>\n" / "</tool_call>",
 * "<function=" / ">\n" / "</function>\n", arguments "<parameter=" / ">\n" / "" / "</parameter>\n".
 * Whitespace at the ends of a tag is optional when matching. Every string is NUL-terminated
 * UTF-8; NULL reads as "". The engine copies what it needs while loading, so static storage in
 * the plugin is enough.
 *
 * The other family is a call written as a name and one object holding its arguments
 * (args_form = RAD_CHAT_ARGS_OBJECT):
 *
 *   [call_open] name_prefix NAME name_suffix { key: value, ... } [call_end] [call_close]
 *
 * The object is JSON, or a dialect of it: strings between object_string_open and
 * object_string_close, their text verbatim, and keys written bare (object_bare_keys). Gemma 4 is
 * "<|tool_call>" / "<tool_call|>", "call:" / "" / "", strings "<|\"|>" / "<|\"|>", bare keys --
 * `<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>` -- and the reply's JSON
 * arguments are that object respelled as JSON. The arg_* fields describe tagged arguments only
 * and are empty in this family.
 *
 * The struct only grows at its end. `struct_size` is sizeof(RadChatFormat) as the plugin was
 * compiled; the engine reads only the fields that size covers and treats the rest as zero, so a
 * plugin built against an older header keeps loading, and a new field must make zero mean "as
 * before". So this export does not move RAD_ABI_VERSION. A field reordered, retyped or removed is
 * a layout change and bumps RAD_ABI_VERSION like any other. */
enum {
    RAD_CHAT_VALUE_RAW   = 0,  /* the text between the delimiters, verbatim */
    RAD_CHAT_VALUE_CDATA = 1,  /* verbatim, unless wrapped in <![CDATA[ ... ]]>, which is removed */
    RAD_CHAT_VALUE_JSON  = 2   /* a JSON value; text that does not parse as JSON is taken raw */
};

enum {
    /* The generation prompt shows whether the reply starts inside a reasoning block: whatever
     * follows the first `reasoning_open` in it is the start of the reply's own text. */
    RAD_CHAT_THINK_FROM_PROMPT = 0,
    RAD_CHAT_THINK_ALWAYS      = 1,  /* the reply is inside a reasoning block from its first byte */
    RAD_CHAT_THINK_NEVER       = 2   /* the model opens its own block, if any */
};

enum {
    RAD_CHAT_ARGS_TAGGED = 0,  /* each argument in its own tags: arg_prefix .. arg_value_suffix */
    RAD_CHAT_ARGS_OBJECT = 1   /* one object after the name, in JSON or the object_* dialect */
};

typedef struct RadChatFormat {
    uint32_t    struct_size;
    const char* name;                /* for the startup log; NULL is the plugin's name */

    const char* reasoning_open;      /* "" when the model does not reason */
    const char* reasoning_close;
    /* Literals that end a reasoning block the model forgot to close, when a call starts there.
     * Each must be a call opener (`call_open`, or `name_prefix` without one) or section_open. */
    const char* const* reasoning_breaks;
    uint32_t    n_reasoning_breaks;
    uint32_t    reasoning_in_prompt; /* RAD_CHAT_THINK_* */

    /* Around the complete list of calls. section_open and section_close must be set together.
     * call_separator is additional text between calls beyond whitespace; it is not supported. */
    const char* section_open;
    const char* section_close;
    const char* call_separator;

    const char* call_open;           /* around each call; "" when calls are not wrapped */
    const char* call_close;
    const char* name_prefix;         /* "" when the model has no tool calls */
    /* "" when nothing of its own ends the name: then the object's `{` does, or, for tagged
     * arguments in an unwrapped call, the first arg_prefix or call_end, which begin the body */
    const char* name_suffix;
    const char* call_end;
    const char* arg_prefix;
    /* "" when arg_value_prefix alone ends the name; both set are written back to back */
    const char* arg_name_suffix;
    const char* arg_value_prefix;
    const char* arg_value_suffix;

    uint32_t    string_value;        /* RAD_CHAT_VALUE_RAW or _CDATA: a string-typed argument */
    uint32_t    other_value;         /* RAD_CHAT_VALUE_JSON: any other type (or _RAW, as text) */
    /* Non-zero: one newline at each end of a value is punctuation, not part of the value -- at
     * its start only where the tag before it does not end with that newline itself. */
    uint32_t    value_newlines;

    /* The literal words that start constrained decoding of a call. None: no grammar, and the
     * reply is read unconstrained. */
    const char* const* triggers;
    uint32_t    n_triggers;

    /* RAD_CHAT_ARGS_*: how a call's arguments are written. Zero is the tagged family above. With
     * RAD_CHAT_ARGS_OBJECT, name_suffix may be empty -- the name then ends where the object
     * begins -- and so may call_end, when the object's own closing brace ends the call. */
    uint32_t    args_form;
    /* RAD_CHAT_ARGS_OBJECT: a string delimiter of the object's dialect besides JSON's `"`. What
     * stands between the two is the string, verbatim -- the dialect has no escapes. NULL: the
     * object writes JSON strings only. */
    const char* object_string_open;
    const char* object_string_close;
    /* RAD_CHAT_ARGS_OBJECT: non-zero when the dialect writes a key bare, `key:value`. */
    uint32_t    object_bare_keys;

    /* Non-zero: a reasoning block may open again later in the reply (after its answer, between
     * calls) and is read as reasoning there too, and a reasoning_close with no block open is not
     * part of the answer. Zero: only a block the reply starts with is reasoning. */
    uint32_t    reasoning_anywhere;

    /* Tagged arguments whose tag says the value's type: written in place of arg_value_prefix,
     * this opens a JSON value, and arg_value_prefix then opens a string. The tag the reply writes
     * decides how its value is read, whatever the schema says; the grammar writes the one the
     * schema calls for. NULL: the schema alone types a value. */
    const char* arg_json_value_prefix;

    /* A header the model writes first when it answers without reasoning, in place of
     * reasoning_open: the reply then starts with this literal, which is dropped, and the rest is
     * content. A reply that starts with anything else, or with only part of it, is read as
     * written. When the model does reason, its header after the block is reasoning_close's.
     * NULL or "": the reply starts with its content. */
    const char* content_open;
} RadChatFormat;

/* Return the format, or NULL for "derive it from the chat template". Called once at startup. */
const RadChatFormat* rad_arch_chat_format(const RadModelMeta* meta);

/* How a speech model's prompt carries audio, as data. Optional: a plugin that does not export it
 * is a model the speech routes do not serve.
 *
 * It states the tokens that open and close a clip, the turn a transcription is asked in, how a
 * language is named and forced, and where the transcript starts in the reply, so the core's
 * routes know nothing of any one model. Qwen3-ASR, for example:
 *
 *   audio_open "<|audio_start|>"  audio_close "<|audio_end|>"
 *   transcribe_prompt  "<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n{audio}"
 *                      "<|im_end|>\n<|im_start|>assistant\n{language}"
 *   language_prompt    "language {language}<asr_text>"
 *   transcript_open "<asr_text>"   language_lead "language "
 *   languages { "en", "English", "zh", "Chinese", ... }
 *
 * Every string is NUL-terminated UTF-8 and NULL reads as "". The engine copies what it needs while
 * loading, so static storage in the plugin is enough.
 *
 * The struct only grows at its end, as RadChatFormat does: `struct_size` is sizeof(RadAudioFormat)
 * as the plugin was compiled, the engine reads only the fields that size covers, and a field added
 * later reads as zero for an older plugin. So this export does not move RAD_ABI_VERSION. */
typedef struct RadAudioFormat {
    uint32_t    struct_size;

    /* The literal tokens around a clip's placeholder run, as the vocabulary spells them, for a
     * model whose config names no id for them. NULL: the config's own (boa/eoa_token_id). */
    const char* audio_open;
    const char* audio_close;

    /* The transcription prompt (POST /v1/audio/transcriptions), with three slots: {context} is
     * the request's `prompt` text, {audio} the clip, and {language} `language_prompt` with the
     * request's language named in it, or "" when the request names none and the model is to say
     * which it heard. Each slot appears once. NULL: this model does not transcribe. */
    const char* transcribe_prompt;
    const char* language_prompt;

    /* A reply whose language was not forced names it first: `language_lead` and the language's
     * name, then `transcript_open`, then the transcript. "" for a model whose reply is the
     * transcript alone. */
    const char* transcript_open;
    const char* language_lead;

    /* The languages the model names, as n_languages (ISO 639-1 code, the model's own name) pairs:
     * a request's `language` matches either spelling, and the name is what {language} takes. */
    const char* const* languages;
    uint32_t    n_languages;
} RadAudioFormat;

/* Return the format, or NULL when this model serves no speech route. Called once at startup. */
const RadAudioFormat* rad_arch_audio_format(const RadModelMeta* meta);

/* A request's free text rewritten into the form the model was trained on, before it goes into the
 * audio generator's prompt template (RadAudioGenDecl::tmpl). Optional: absent, the text goes in
 * as given. Training data has conventions a client does not know -- section tags on lines of
 * their own and lower case, no markdown in a description -- and only the plugin knows them.
 *
 * `field` is RAD_TEXT_*, `in` NUL-terminated UTF-8: the instructions, the input, or a
 * reference's transcript (RadAudioGenDecl::ref_item's `{text}`). Writes at most `cap` bytes, the
 * NUL included, and returns the length the whole result needs without its NUL (call again with
 * more room when it is >= cap), or a negative status for text the model cannot take, with the
 * reason in `out` (NUL-terminated within `cap`) when it has one to give the caller. Called on
 * HTTP threads, so it keeps no state. */
enum { RAD_TEXT_INSTRUCTIONS = 1, RAD_TEXT_INPUT = 2, RAD_TEXT_REFERENCE = 3 };
int64_t rad_arch_text_prepare(int field, const char* in, char* out, int64_t cap);

/* The layout of a video generator's passes (rad_declare_video_generator), which only the model
 * knows. `which` names the pass: RAD_VPOS_STEP, the packed sequence of every VSTEP pass of a
 * request of geometry `g` -- its rows, and their rotary positions written to `pos` (f32 [3, rows]
 * component-major) when it is non-null and `cap` rows hold them; RAD_VPOS_DECODE, the rows of one
 * of its VDECODE passes, whose positions the plugin makes itself (`pos` is not written). Returns
 * the rows, or a negative status for a geometry the model cannot take. Called on any thread, so it
 * keeps no state. Required of a video generator. */
typedef struct RadVideoGeom {
    int32_t n_text;               /* the prompt's tokens */
    int32_t frames;               /* the video's frames */
    int32_t lat_frames;           /* its latent frames */
    int32_t lat_h, lat_w;         /* latents a frame, down and across */
    int32_t n_audio;              /* the soundtrack's latents a channel */
} RadVideoGeom;
enum { RAD_VPOS_STEP = 1, RAD_VPOS_DECODE = 2 };
int64_t rad_arch_video_positions(int which, const RadVideoGeom* g, float* pos, int64_t cap);

/* The geometry of an encode pass (RAD_GEN_PASS_VENCODE, RAD_GEN_PASS_AENCODE): the int32s the
 * driver stages verbatim in RadVideoGenBatch::enc_geom_d for the plugin's kernels. `kind` is the
 * pass; `in` is {frames or samples, height, width, channels}, 0 where unused. With `out` null it
 * returns the count, with `out` it fills out[cap]; negative for media the encoders cannot take.
 * Entry 0 of an Aencode geometry is the samples padded to the encoder's hop, entry 23 of a Vencode
 * geometry the frames padded to whole clips and entry 24 its latent frames: the driver stages the
 * media padded to those lengths. A plugin with encoders exports it. Called on any thread. */
int64_t rad_arch_encode_geom(int32_t kind, const int64_t* in, int32_t* out, int64_t cap);

/* The packed sequence of a step with reference media or keyframes in: `ref_tab` is
 * RadVideoGenBatch::ref_tab. Returns the sequence's rows, the rows RAD_VPOS_STEP would position, or
 * a negative status. `seg` receives the segment boundaries (seg[0] = 0, the last the rows) and
 * `mod` the boundaries of the modulation groups, then the cell each group reads (groups that read
 * the same modulation share a cell, numbered from 0); `*n_seg` and `*n_mod` are their capacities
 * in and their counts out, and a null array asks for the count alone. `pos` receives the rotary
 * positions, f32 [3, rows], when `cap_pos` holds 3 x rows. A plugin that declares reference media
 * exports it. Called on any thread. */
int64_t rad_arch_video_pack(const RadVideoGeom* g, const int32_t* ref_tab, int32_t* seg,
                            int32_t* n_seg, int32_t* mod, int32_t* n_mod, float* pos,
                            int64_t cap_pos);

int         rad_arch_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx);
/* Forward declaration; the signature is in rad_runtime.h. */
struct RadCtx;
struct RadBatch;
void        rad_arch_step(struct RadCtx* c, const struct RadBatch* batch);
/* Optional. Return a stable fingerprint of actual dispatch geometry whose grid depends on live
 * per-group seqused values. `kv_used_max_h[g]` is the host maximum for batch->kv[g], and
 * rad_ctx_wgp_count(c) supplies the device's WGP count. Return 0 when exact geometry is unknown;
 * that pass runs live and is not recorded. Only prefill and mixed launches read the live maximum:
 * in a decode, verify or draft pass every launch reads the step's keyed context bound
 * (RadArgs::kv_used_max_h), so such a pass needs no geometry here and returns nonzero. */
uint64_t    rad_arch_pass_key(struct RadCtx* c, const struct RadBatch* batch,
                              const int32_t* kv_used_max_h);

#ifdef __cplusplus
}   /* extern "C" */
#endif
#endif /* RAD_BUILDER_H */
