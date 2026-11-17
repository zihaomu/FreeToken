/* kvbind.h -- the KV group pools as the run phase addresses them (KVPoolBinding, ctx.h). */
#pragma once
#include "ctx.h"

#include <functional>
#include <vector>

namespace rad {

struct KVGroupPlan;

/* One binding per KV group of `p`, from the block manager's plan for it (`plan_of(g)`, null for a
 * group with none). An operand names an absolute model layer while a pool holds only its group's
 * layers, so each bound layer maps to its position in the group: in a group of layers 3, 7, ...,
 * layer 7 is the second; subtracting the first bound layer would give the fifth. */
void bind_kv_pools(const Program& p, const std::function<const KVGroupPlan*(int32_t)>& plan_of,
                   std::vector<KVPoolBinding>* out);

}  /* namespace rad */
