/* kvbind.cpp -- see kvbind.h. */
#include "kvbind.h"

#include "kvstream/bind.h"
#include "mem/kv.h"

namespace rad {

void bind_kv_pools(const Program& p, const std::function<const KVGroupPlan*(int32_t)>& plan_of,
                   std::vector<KVPoolBinding>* out) {
    out->assign(p.kv_groups.size(), KVPoolBinding{});
    for (size_t g = 0; g < p.kv_groups.size(); ++g) {
        const KVGroupPlan* gp = plan_of((int32_t)g);
        if (!gp) continue;
        KVPoolBinding& b = (*out)[g];
        /* A paged group's are then set from its active view by kv_bind_views. */
        b.base         = gp->base;
        b.layer_stride = gp->layer_stride();
        b.bytes        = gp->pool_bytes;
        b.dtype        = p.kv_groups[g].decl.dtype;

        /* The position of every bound layer, indexed by its absolute number. */
        const std::vector<int32_t>& ls = p.kv_groups[g].layers;
        int32_t hi = -1;
        for (int32_t l : ls) if (l > hi) hi = l;
        b.layer_slot.assign((size_t)(hi + 1), -1);
        for (size_t i = 0; i < ls.size(); ++i) b.layer_slot[(size_t)ls[i]] = (int32_t)i;

        /* One layer's cache shape, in the layout libref/ref_ops.h defines. Built here because only
         * this point has both the plugin's declaration (heads, head_dim, state_dim, conv_width)
         * and the block manager's decisions (block_size, n_blocks, n_states, conv_slots); a
         * kernel cannot derive the conv depth or slot count from op parameters. */
        const RadKVGroupDecl& d = p.kv_groups[g].decl;
        switch (gp->kind) {
        case RAD_KV_FULL:
        case RAD_KV_WINDOW:
        case RAD_KV_RING:
            b.rank = 4;
            /* Set again per view by kv_bind_views. */
            b.shape[0] = gp->pool_blocks();
            b.shape[1] = d.n_head_kv;
            b.shape[2] = gp->block_size;
            b.shape[3] = rad_kv_row_elems(d.dtype, d.head_dim);
            kv_bind_views(*gp, &b);
            break;
        case RAD_KV_LINEAR:
            b.rank = 4;
            b.shape[0] = gp->n_states;
            b.shape[1] = d.n_head_kv;
            b.shape[2] = d.state_dim[0];
            b.shape[3] = d.state_dim[1];
            break;
        case RAD_KV_CONV:
            b.rank = 3;
            b.shape[0] = gp->n_states;
            b.shape[1] = d.n_head_kv * d.head_dim;
            b.shape[2] = gp->conv_slots;
            break;
        default:
            b.rank = 0;   /* the null sentinel at handle 0; nothing issues against it */
            break;
        }
    }
}

}  /* namespace rad */
