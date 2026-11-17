/* r4d_expert_scale_bf16.hip -- expert_scale: each routing weight times its expert's own scale, in place.
 *
 * Gemma 4's router renormalises its top-k, then multiplies each weight by its expert's
 * per_expert_scale. Unfused that is two launches per MoE layer (`embed_lookup` of the [n_expert, 1]
 * scale table by expert id, then `mul`), each a few dozen elements at decode. Fused:
 *
 *     ew[t, k] = bf16( ew[t, k] * scale[ids[t, k]] )
 *
 * Bit-identical to the pair: one f32 product of two bf16 values, rounded once to nearest even. An
 * id outside the table reads a zero scale, as embed_lookup writes a zero row, so the product's
 * sign matches too. One thread per element.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "rad_plugin.h"

namespace r4d_escale {

struct EscaleArgs {
    const int32_t*  ids;  long long ids_ld;
    const uint16_t* sc;                         /* [n_expert], bf16 */
    uint16_t*       ew;   long long ew_ld;
    long long       rows;
    int             k, n_expert;
};

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((unsigned)h << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
    const unsigned u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__global__ __launch_bounds__(256) void expert_scale_kernel(EscaleArgs A) {
    const long long e = (long long)blockIdx.x * 256 + threadIdx.x;
    if (e >= A.rows * A.k) return;
    const long long r = e / A.k;
    const int c = (int)(e - r * A.k);
    const int id = A.ids[r * A.ids_ld + c];
    const float s = (id >= 0 && id < A.n_expert) ? bf2f(A.sc[id]) : 0.0f;
    uint16_t* w = A.ew + r * A.ew_ld + c;
    *w = f2bf(bf2f(*w) * s);
}

/* [rows, k] with a dense last axis: its rows and row pitch, or false */
static bool rows_k(const RadTensor* t, uint32_t dt, int64_t k, int64_t* rows, int64_t* ld) {
    if (!t || t->dtype != dt || t->rank < 1) return false;
    if (rad_tensor_lastdim(t) != k || t->stride[t->rank - 1] != 1) return false;
    *rows = rad_tensor_rows(t);
    *ld = rad_tensor_row_pitch(t);
    return *ld >= k || *rows <= 1;
}

}  /* namespace r4d_escale */

extern "C" {

/* operands: ids, w_e (the [n_expert] scale table), ew (inout) */
int r4d_rad_expert_scale_bf16(const RadArgs* a, RadStream stream) {
    using namespace r4d_escale;
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const int64_t k = rad_args_geti_or(a, "top_k", 0), ne = rad_args_geti_or(a, "n_expert", 0);
    if (k <= 0 || k > 64 || ne <= 0 || ne > (1 << 20)) return RAD_E_SHAPE;
    const RadTensor *I = rad_arg_in(a, 0), *S = rad_arg_in(a, 1), *W = rad_arg_in(a, 2);
    if (!I || !S || !W) return RAD_E_INVAL;
    if (S->dtype != RAD_BF16 || rad_tensor_numel(S) < ne) return RAD_E_DTYPE;
    int64_t ri = 0, li = 0, rw = 0, lw = 0;
    if (!rows_k(I, RAD_I32, k, &ri, &li) || !rows_k(W, RAD_BF16, k, &rw, &lw)) return RAD_E_SHAPE;
    const int64_t M = rw;                            /* the step's rows: the stream's view */
    if (M <= 0) return RAD_OK;
    if (ri < M || M * k > 0x7fffffff) return RAD_E_SHAPE;
    EscaleArgs P{};
    P.ids = (const int32_t*)I->data; P.ids_ld = li;
    P.sc = (const uint16_t*)S->data;
    P.ew = (uint16_t*)W->data;       P.ew_ld = lw;
    P.rows = M; P.k = (int)k; P.n_expert = (int)ne;
    const unsigned blocks = (unsigned)((M * k + 255) / 256);
    R4D_KLAUNCH(R4D_KNAME(expert_scale_kernel), dim3(blocks), dim3(256), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* extern "C" */
