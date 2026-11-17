/* rad_quant.h -- the quantiser plugin ABI. spec.md §4.5.
 *
 * A quantiser turns a weight, handed to it in f32, into the planes of an encoding (rad_encoding.h).
 * It is the third plugin kind beside kernels and architectures, loaded from
 * $RADIANCE_HOME/quantizers, and the only place a weight's numbers are chosen: a kernel only
 * rearranges the planes it is given (rad_abi.h, RadRelayoutFn).
 *
 * rad-convert drives quantisers through a recipe: ordered rules from a logical weight name to a
 * quantiser and its options (spec §4.4). For each weight a rule picks, it asks the quantiser which
 * encoding it would write (`encoding`), lets the architecture plugin declare against that, then
 * hands the weight over in row blocks (`row_block`, `quantize`), so a 51 GB table streams.
 *
 * Input: the weight's logical name, shape and movement group, and the column importances when the
 * converter has an imatrix. Other calibration data (a GPTQ Hessian, AutoRound's activations) is an
 * option the quantiser reads itself (`calib=DIR`); its format is the quantiser's business.
 *
 * Output: the canonical planes only, so every kernel library that accepts the encoding and the
 * core's own decoder can read it. That is how rad-convert measures the error without knowing the
 * method.
 *
 * Pure C, like the rest of abi/.
 */
#ifndef RAD_QUANT_H
#define RAD_QUANT_H

#include "rad_abi.h"
#include "rad_encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

/* What a quantiser is told about the weight it is quantising. */
typedef struct RadQuantWeight {
    const char* name;                 /* the logical (declared) name: "blk.7.ffn_down_exps.3.weight" */
    uint32_t    rank;
    int64_t     shape[RAD_MAX_RANK];  /* the logical shape */
    int64_t     rows, cols;           /* the [rows, cols] view every plane is laid over */
    /* The movement group; -1 for none. Also -1 when `encoding` is asked at declare, before the
     * group exists, so an encoding may depend on the name and shape but not on these; rad-convert
     * refuses a quantiser whose answer changes once it is told. */
    int32_t     layer, expert;
    /* The imatrix's column importances for this weight (the second moment of its input, one per
     * column), or null when the converter has none. Per expert where the imatrix has that form. */
    const float* importance;
} RadQuantWeight;

/* One option a quantiser takes, for `rad-convert --list-quantizers` and for typing a recipe's
 * `key=value` text before the quantiser sees it: an option declared RAD_P_INT arrives as an int,
 * RAD_P_F64 as a double, RAD_P_STR as the text. An option a recipe gives and the table does not
 * name is refused by the converter, by name, before anything is quantised. */
typedef struct RadQuantOption {
    const char* key;
    int         type;                 /* RAD_P_INT | RAD_P_F64 | RAD_P_STR */
    const char* deflt;                /* the default, as text; null for none */
    const char* doc;
} RadQuantOption;

/* The encoding this quantiser writes for `w` under options `o`. RAD_E_UNSUPPORTED declines (a
 * weight it does not quantise, like a rank-1 norm) and the converter moves on to the next rule;
 * any other failure is a recipe error reported with the weight's name. */
typedef int (*RadQuantEncodingFn)(const RadParam* o, int n_o, const RadQuantWeight* w,
                                  RadEncoding* out);

/* The rows one `quantize` call may take: every call but the last gets a multiple of this, starting
 * on a multiple of it. 0 means the whole weight in one call, needed for any cross-row decision (a
 * derived per-tensor scale, a learned rotation, a shared Hessian). Must be a multiple of every
 * TILED plane's block rows. */
typedef int64_t (*RadQuantRowBlockFn)(const RadParam* o, int n_o, const RadQuantWeight* w);

/* Quantise logical rows [row0, row0 + rows) of `w`, given as f32 [rows, cols] in `src`, into the
 * planes. `planes[i]` points at the element row of plane i that covers logical row `row0` (row
 * row0 / br for a TILED plane of block rows br; the plane's start for a whole-extent or TABLE
 * plane), and the call writes exactly the rows its block covers.
 * The buffers arrive zeroed. `src` may be modified. */
typedef int (*RadQuantizeFn)(const RadParam* o, int n_o, const RadQuantWeight* w,
                             float* src, int64_t row0, int64_t rows, void* const* planes);

/* Decode rows [row0, row0 + rows) back to f32 [rows, cols], for an encoding whose scheme the core
 * does not decode. Null for a quantiser that writes only "plain" and "affine", which the core
 * decodes itself. `planes[i]` points at plane i's start.
 *
 * The oracle, rad-kbench and rad-convert's error measurement all read a quantiser's own scheme
 * through it. The core asks every loaded quantiser that has one, in load order, and takes the
 * first answer that is not RAD_E_UNSUPPORTED, so a decoder must return RAD_E_UNSUPPORTED for an
 * encoding it did not write. `w` names the weight where the caller knows it; a decoder must need
 * nothing beyond the encoding, `rows` and `cols`, because the oracle knows no more. */
typedef int (*RadQuantDecodeFn)(const RadEncoding* e, const RadQuantWeight* w,
                                const void* const* planes, int64_t row0, int64_t rows,
                                float* out);

typedef struct RadQuantizerInfo {
    const char*           name;      /* what a recipe names: "fp8-block", "int", "gptq" */
    const char*           doc;
    const RadQuantOption* options;
    int                   n_options;
    RadQuantEncodingFn    encoding;
    RadQuantRowBlockFn    row_block;
    RadQuantizeFn         quantize;
    RadQuantDecodeFn      decode;    /* optional */
} RadQuantizerInfo;

/* Every quantiser plugin exports these, beside rad_plugin_abi_version and rad_plugin_info (kind
 * RAD_PLUGIN_QUANT). Two plugins naming the same quantiser are refused at load, both named, since
 * a recipe naming it would otherwise give different numbers depending on load order. */
int                     rad_quant_count(void);
const RadQuantizerInfo* rad_quant_at(int i);

/* Optional: the cards a quantiser may run on, as the HIP runtime numbers them (after
 * HIP_VISIBLE_DEVICES); none is the host alone, the state before any call. rad-convert calls it
 * once, after loading and before the first weight, with its --gpus list. A quantiser that uses a
 * card writes the same bytes it writes on the host, so a container does not depend on the machine
 * that converted it, and --reuse matches it either way. RAD_E_UNSUPPORTED: this build has no
 * device path; any other negative status refuses the list (a card that is not there). Symbol
 * name: `rad_quant_devices`. */
typedef int (*RadQuantDevicesFn)(const int* ids, int n);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* RAD_QUANT_H */
