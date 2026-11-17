/* accept.cpp -- acceptance for speculative decoding. See accept.h for why the rule is exact.
 */
#include "accept.h"

namespace rad {

int accept_greedy(const int32_t* draft, int n_draft, const int32_t* target,
                  int32_t* out, int32_t* n_accepted) {
    if (!target || !out || !n_accepted || n_draft < 0) return RAD_E_INVAL;
    if (n_draft > 0 && !draft) return RAD_E_INVAL;

    int i = 0;
    for (; i < n_draft; ++i) {
        if (draft[i] != target[i]) break;
        out[i] = draft[i];
    }
    /* The target's token at position i is correct to emit either way: the bonus token on a full
     * run, the correction on a rejection. */
    out[i] = target[i];
    *n_accepted = i;
    return RAD_OK;
}

}  /* namespace rad */
