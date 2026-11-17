/* fleet/fleet.h -- the models of one configuration, served by one process (docs/MULTIMODEL.md).
 *
 * Every model the YAML names is known to the Front (server/front.h) from the start and loaded when
 * a request first needs it: the fleet's scheduler thread builds its Engine on cards it chooses,
 * making room by unloading models the policy lets go (fleet/policy.h). A model with an idle `ttl`
 * is unloaded after that long without requests; a pinned one is loaded at start and never leaves;
 * a one-model configuration loads its model at start, as a process of one model always has.
 *
 * Engines start one after another, since bring-up loads plugins and declares against process-wide
 * tables; they serve side by side. Unloading destroys the engine; its hardware queues stay with the
 * process for the next engine on that card (core/device/aql.cpp, g_spare) until the card is left
 * empty or a model needs their memory.
 *
 * Signals: SIGINT and SIGTERM drain and stop every model; SIGHUP re-reads the file (§10).
 */
#pragma once
#include "fleet/config.h"

#include <string>

namespace rad::fleet {

/* Serve every model until SIGINT or SIGTERM. The process's exit code. */
int run(FleetConfig& fc);

/* The keys this build reads but does not act on yet, as one message naming them by YAML path, or
 * "". A file setting them is refused rather than half-honoured. */
std::string unsupported(const FleetConfig& fc);

}  // namespace rad::fleet
