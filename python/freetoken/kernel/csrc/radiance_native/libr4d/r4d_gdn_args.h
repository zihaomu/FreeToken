/* r4d_gdn_args.h -- argument helpers shared by the gated-delta-net kernels (and the DFlash ops):
 * dtype checks, num_seqs from cu_seqlens, the compiled-geometry check and a row-pitch reader.
 */
#ifndef R4D_GDN_ARGS_H
#define R4D_GDN_ARGS_H

#include "r4d.h"
#include "r4d_args.h"


/* Every `const void*` libr4d takes has one element type, cast inside the kernel (e.g.
 * r4d_gdn_conv_prep_w4_h128_bf16 reads `wgt` as bf16 and `A_log` as f32); the signature does not
 * say so. A plugin declaring the `gdn_kkt` buffer as f32 would be read as
 * bf16 on the device, while libref (which honours each operand's dtype) stays correct, so the bug
 * would show only on the device path. The shim therefore checks dtypes by name (spec §2.1). */
static inline int r4d_dt(const RadTensor* t, uint32_t want) {
    if (!t) return RAD_OK;                       /* an absent optional operand has no dtype */
    return t->dtype == want ? RAD_OK : RAD_E_DTYPE;
}
#define R4D_DT(t, want) \
    do { const int _d = r4d_dt((t), (want)); if (_d != RAD_OK) return R4D_NO(_d); } while (0)

/* A raw r4d.h entry point's result as a RAD status. Those return 0, a negative code for a
 * geometry they were not compiled for, or the positive hipError_t of a failed launch. */
static inline int r4d_raw_status(int rc, const char* file, int line) {
    if (rc == 0) return RAD_OK;
    return r4d_refused(rc < 0 ? RAD_E_SHAPE : RAD_E_DEVICE, file, line);
}
#define R4D_RAW(rc) r4d_raw_status((rc), __FILE__, __LINE__)

/* num_seqs from the [N+1] cu_seqlens tensor, the only operand that carries N. Requiring rank 1
 * keeps an off-by-one N from reaching kernels that index cu[N] unconditionally. */
static inline int seqs_of(const RadTensor* cu, int* out) {
    if (!cu || cu->rank != 1 || cu->shape[0] < 2) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);
    *out = (int)(cu->shape[0] - 1);
    return RAD_OK;
}

/* libr4d's compiled-in geometry, so a rebuild at a different head size cannot silently change what
 * the rows mean. */
static inline int gdn_dims_ok(void) {
    int hk = 0, hv = 0, ch = 0;
    r4d_gdn_dims(&hk, &hv, &ch);
    return (hk == 128 && hv == 128 && ch == 64) ? RAD_OK : RAD_E_SHAPE;
}


/* Elements between consecutive `width`-wide rows: the second-to-last stride of a [.., heads, width]
 * tensor, or `width` for a dense flattened [tokens, heads*width] one. Negative otherwise. */
static inline long long gn_row_pitch(const RadTensor* t, long long width) {
    if (!t || t->rank < 1) return -1;
    if (t->shape[t->rank - 1] == width) {
        if (t->stride[t->rank - 1] != 1) return -1;
        return t->rank >= 2 ? t->stride[t->rank - 2] : width;
    }
    return r4d_contig(t) ? width : -1;
}

#endif  /* R4D_GDN_ARGS_H */
