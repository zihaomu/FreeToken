/* derive.h -- per-step metadata the core computes once, by name, into the arena.
 *
 * A plugin that needs linear-state chunk boundaries or conv-state indices declares a
 * RAD_BUF_DERIVED buffer naming the derivation. The core fills it once per step, not per layer,
 * and layers that share a geometry share the buffer. The step batch is the natural owner of
 * "once per step"; vLLM has no such object and keys it off a module-level counter.
 *
 * Adding a derivation is a row in the table in derive.cpp.
 */
#pragma once
#include "batch.h"

namespace rad {

/* Everything a derivation may read. All host memory, already staged for this step, so no
 * derivation adds a device sync to the step path. */
struct DeriveInput {
    const StepPlan* plan = nullptr;
    const int32_t*  cu_seqlens = nullptr;   /* [n_seq+1] */
    const int32_t*  positions = nullptr;    /* [n_tok] absolute position per token */
    const int32_t*  conv_slot = nullptr;    /* [n_seq] the conv group's state slot, -1 if none */
    int64_t         n_tok = 0;
    int64_t         n_seq = 0;
    int64_t         state_chunk = 1;        /* the resolved linear-state chunk length */
    int32_t         conv_window = 1;        /* conv_width - 1 + max_spec */
};

/* Writes at most `cap` int32 elements to `out` and reports how many. Returns RAD_OK or negative;
 * negative aborts the step rather than let an op read a half-filled buffer. */
typedef int (*DeriveFn)(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out);

struct DeriveRow {
    const char* name;
    DeriveFn    fn;
    const char* doc;
};

/* -1 for an unknown name. Reported at declare with the known list, like a misspelled op
 * (spec §2.3). */
int              derive_index(const char* name);
const DeriveRow* derive_row(int index);
int              derive_count();

}  /* namespace rad */
