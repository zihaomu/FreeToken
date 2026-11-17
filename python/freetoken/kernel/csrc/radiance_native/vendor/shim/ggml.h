/* vendor/shim/ggml.h -- NOT a lift. This file is ours.
 *
 * The lifted chat and PEG-parser sources include "ggml.h" for exactly two macros. Taking ggml to
 * get them would take the thing spec §12 says we do not take, and editing the lifts to drop the
 * include would make them unpatchable. A shim is the third option: the lifted files compile
 * BYTE-FOR-BYTE unchanged, and everything they needed from outside is here where it is visible.
 */
#pragma once

#include <cstdio>
#include <cstdlib>

/* GGML_ABORT is reached only on an invariant the lifted code believes cannot break. It behaves
 * the way rad::fatal() does -- print and exit non-zero rather than continue from a broken state
 * (spec §17) -- and is spelt without including rad_internal.h so that vendor/ stays free of any
 * dependency on core/. */
#define GGML_ABORT(...)                                                     \
    do {                                                                    \
        fprintf(stderr, "fatal: %s:%d: ", __FILE__, __LINE__);              \
        fprintf(stderr, __VA_ARGS__);                                       \
        fputc('\n', stderr);                                                \
        abort();                                                            \
    } while (0)

#define GGML_ASSERT(x)                                                      \
    do {                                                                    \
        if (!(x)) GGML_ABORT("assertion failed: %s", #x);                   \
    } while (0)
