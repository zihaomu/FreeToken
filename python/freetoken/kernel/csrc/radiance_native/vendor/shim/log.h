/* vendor/shim/log.h -- NOT a lift. This file is ours.
 *
 * llama.cpp's log.h is a whole logging subsystem with a background thread and its own verbosity
 * plumbing. The lifted files use four macros from it. Routing those four at radiance's logger
 * would make vendor/ depend on core/, so they go to stderr behind one runtime switch, and
 * core/text/chat.cpp turns that switch on when rad's own level reaches Debug.
 *
 * The cost, named: a lifted file's log line does not carry rad's level, file or timestamp. These
 * lines exist for debugging a template the auto-parser could not crack, which is a developer
 * activity and not a production one.
 */
#pragma once

#include "ggml.h"   /* upstream's log.h includes ggml.h for ggml_log_level, and lifted files rely
                     * on that transitively -- json-partial.cpp uses GGML_ASSERT without
                     * including ggml.h itself. Carrying the same transitive include keeps
                     * those files unedited. */
#include <cstdio>

namespace rad_vendor_log {
/* Off by default. Anything below Warn from a vendored file is noise during normal operation. */
inline bool& verbose() { static bool v = false; return v; }
}

#define LOG_ERR(...)  fprintf(stderr, __VA_ARGS__)
#define LOG_WRN(...)  fprintf(stderr, __VA_ARGS__)
#define LOG_INF(...)  do { if (::rad_vendor_log::verbose()) fprintf(stderr, __VA_ARGS__); } while (0)
#define LOG_DBG(...)  do { if (::rad_vendor_log::verbose()) fprintf(stderr, __VA_ARGS__); } while (0)
#define LOG_CNT(...)  do { if (::rad_vendor_log::verbose()) fprintf(stderr, __VA_ARGS__); } while (0)
