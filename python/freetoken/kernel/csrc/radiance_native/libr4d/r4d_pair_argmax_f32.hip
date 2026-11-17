/* r4d_pair_argmax_f32.hip -- pair_argmax: the global argmax of a row of gathered (id, value) pairs,
 * written as one contiguous column of token ids.
 *
 * A vocab-sharded draft head finds each rank's best candidates with `row_topk` and moves them with
 * `all_gather`, which moves whole 16-byte words per row; a pair is 8 bytes, so a rank contributes
 * at least R = 2 (rad_block_mtp_hc.h states the same bound). `row_topk_merge` then returns two
 * columns per row, but a block drafter's proposal is `steps` contiguous ids per sequence, and
 * nothing in libr4d copies a strided i32 column. This reduces the row to its top-1 and writes it
 * where the proposal reads it.
 *
 * Same rules as row_topk_merge at R = 1, with a strided output: the largest value wins, a tie goes
 * to the lower id, and a negative id (a slot a shard could not fill) is skipped. A row with no
 * valid pair writes id -1 and value -inf. One wave per row, eight rows per workgroup.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "rad_plugin.h"

namespace r4d_pick {

struct PickArgs {
    const float* pairs; long long p_rs;      /* [M, n, 2]: int32 id bits, then f32 value */
    int*         tok;   long long t_rs;
    float*       val;   long long v_rs;      /* optional */
    long long    rows;
    int          n;
};

__device__ __forceinline__ int pk_lane() { return (int)(threadIdx.x & 31u); }
__device__ __forceinline__ int pk_xor_i(int v, int m) {
    return __builtin_amdgcn_ds_bpermute((pk_lane() ^ m) << 2, v);
}
__device__ __forceinline__ float pk_xor_f(float v, int m) {
    return __int_as_float(pk_xor_i(__float_as_int(v), m));
}
/* (v, id) beats (bv, bid)? Larger value, then lower id; an invalid id (< 0) never wins. */
__device__ __forceinline__ bool pk_better(float v, int id, float bv, int bid) {
    if (id < 0) return false;
    if (bid < 0) return true;
    return v > bv || (v == bv && id < bid);
}

__global__ __launch_bounds__(256) void pick_kernel(PickArgs A) {
    const long long r = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= A.rows) return;                              /* uniform across the wave */
    const int lane = pk_lane();
    const float* row = A.pairs + r * A.p_rs;
    float bv = -__builtin_inff();
    int bid = -1;
    for (int j = lane; j < A.n; j += 32) {
        const int id = __float_as_int(row[2 * j]);
        const float v = row[2 * j + 1];
        if (pk_better(v, id, bv, bid)) { bv = v; bid = id; }
    }
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) {
        const float ov = pk_xor_f(bv, m);
        const int oid = pk_xor_i(bid, m);
        if (pk_better(ov, oid, bv, bid)) { bv = ov; bid = oid; }
    }
    if (lane == 0) {
        A.tok[r * A.t_rs] = bid;
        if (A.val) A.val[r * A.v_rs] = bv;
    }
}

}  /* namespace r4d_pick */

/* pairs as [M, n, 2] (rank 3) or [M, 2n] (rank 2), last axes dense; the outputs one element a row
 * at their own row pitch. */
extern "C" int r4d_rad_pair_argmax_f32(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const RadTensor* p = rad_arg_in(a, 0);
    const RadTensor* t = rad_arg_in(a, 1);
    const RadTensor* v = rad_arg_in(a, 2);
    if (!p || !t) return RAD_E_INVAL;
    if (p->dtype != RAD_F32 || t->dtype != RAD_I32 || (v && v->dtype != RAD_F32)) return RAD_E_DTYPE;
    const long long n = rad_args_geti_or(a, "n", 0);
    if (n <= 0 || p->rank < 2) return RAD_E_SHAPE;
    if (p->rank == 3) {
        if (p->shape[1] < n || p->shape[2] != 2 || p->stride[2] != 1 || p->stride[1] != 2) return RAD_E_SHAPE;
    } else if (p->rank == 2) {
        if (p->shape[1] < 2 * n || p->stride[1] != 1) return RAD_E_SHAPE;
    } else {
        return RAD_E_SHAPE;
    }
    const int64_t rows = p->shape[0];
    if (rows == 0) return RAD_OK;
    if (t->rank < 1 || t->shape[0] < rows || (v && (v->rank < 1 || v->shape[0] < rows))) return RAD_E_SHAPE;
    r4d_pick::PickArgs A{};
    A.pairs = (const float*)p->data; A.p_rs = p->stride[0];
    /* A row's element is stride[0] apart at any rank: a rank-1 output may be a strided column. */
    A.tok = (int*)t->data;           A.t_rs = t->stride[0];
    A.val = v ? (float*)v->data : nullptr;
    A.v_rs = v ? v->stride[0] : 0;
    A.rows = rows;
    A.n = (int)n;
    R4D_KLAUNCH(R4D_KNAME(r4d_pick::pick_kernel), dim3((unsigned)((rows + 7) / 8)), dim3(256),
               (hipStream_t)stream, A);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
