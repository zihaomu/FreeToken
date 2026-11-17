/* residency_iface.h -- forwards the seam between placement and the run phase.
 *
 * core/runtime/residency.h owns the declarations; placement keeps no copy, since two definitions
 * of a struct that crosses a component boundary are an ODR violation. The contract is documented
 * there, notably: begin_transfer and publish are the only writers and both must bump the
 * generation, because the run phase memoises the stream wait per generation and a new event at
 * the old generation would skip the wait.
 */
#pragma once
#include "runtime/residency.h"
