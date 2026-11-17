/* lq_ggml.h -- the `ggml` quantiser: llama.cpp's legacy, K- and I-quant formats, chosen by
 * llama.cpp's own searches and stored as the canonical affine planes. lq_ggml.cpp says how each
 * format's blocks become planes; lq_ggml_port.h is the search itself.
 */
#pragma once
#include "rad_quant.h"

namespace lq {

/* The quantiser's table row, for lq_registry.cpp. Its initialiser is a constant expression, so it
 * is in place before any table that copies it is built. */
extern const RadQuantizerInfo ggml_quantizer;

}  /* namespace lq */
