/* r4d_norm_join_bf16.hip -- Gemma 4's sandwich-norm joins, each one kernel instead of a chain.
 *
 * A 26B-A4B decode step launches ~250 rmsnorms and ~90 adds, each a few hundred bytes of work
 * behind a full launch (about 2.0 and 0.85 ms of a 16.6 ms step). Two places per layer are pure
 * chains of them:
 *
 *   norm_add_norm    the attention's join:   t = norm(a) * w_a;  x += t;  y = norm(x) * w_y
 *                    (post_attention_layernorm, the residual add, pre_feedforward_layernorm)
 *   join_add_scale   the feed-forward's:     d' = norm(d) * w_d;  r' = norm(r) * w_r;
 *                    j = norm(d' + r') * w_j;  x = (x + j) * s;  and optionally y = norm(x) * w_n
 *                    (post_feedforward_layernorm_1 and _2, the arms' sum, post_feedforward_layernorm,
 *                    the residual add, layer_scalar, and the next layer's input_layernorm or the
 *                    final norm, which read the x this writes)
 *
 * Same function as the chain, rounding for rounding: every value the chain stores as bf16 is
 * rounded here at the same point, with libr4d's own expressions (a norm is
 * bf16((v * rsqrtf(mean(v^2) + eps)) * g), an add bf16(u + v), a scale bf16(x * s)). Only the
 * order of the sum of squares can differ (libr4d picks its block width from a table), moving the
 * f32 norm factor by at most an ulp and, rarely, a bf16 output by one. The arch plugin falls back
 * to the chain where nothing serves these (gemma4.cpp).
 *
 * norm_add_norm_route also writes the routed arm's two inputs: the router's norm(x) * w_r *
 * hidden^-0.5 and the experts' norm(x) * w_e quantised to E4M3, which would otherwise be two more
 * launches (`rmsnorm`, `rmsnorm_quant_fp8`) re-reading x and redoing its sum of squares. The router
 * row uses libr4d's rmsnorm expression, ((x * s) * w) * mult; the codes match rmsnorm_quant_fp8:
 * the bf16-rounded norm, a scale of amax * (1 / 448) per 128 columns (1 for an all-zero group),
 * value times reciprocal clamped to +-448, one v_cvt_pk_fp8_f32. Element i is thread i % 256 at
 * slot i / 256, so a group is four waves at one slot: a wave max, then the four.
 * norm_add_norm_route_i8g writes the experts' input as rmsnorm_had_quant_i8g does instead: int8
 * codes of the bf16-rounded norm rotated by a 128-wide Hadamard, a scale of amax * (1 / 127).
 *
 * One 256-thread block per row: a decode step is a handful of rows, so the row is latency bound
 * and three block reductions beat seven launches. Values a later stage reuses stay in registers
 * (PER per thread: 12 up to n = 3072, kMaxPer = 24 up to 6144; the row table enforces it).
 *
 * Every load is issued first. A barrier is a fence, so a load written after a reduction waits for
 * it and pays a global round trip per phase. All of the row's operands are in flight at once, the
 * two arm norms share one barrier pair, and a thread holds only the elements it touches. Sums over
 * the padding past n add +0.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "r4d_q8.h"       // norm_add_norm_route_i8g's rotated int8 codes
#include "rad_plugin.h"

namespace r4d_join {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;
constexpr int kMaxPer = 24;         /* elements a thread holds: n <= 6144 */

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((unsigned)h << 16); }
/* round to nearest even, as libr4d's f32_to_bf16 */
__device__ __forceinline__ uint16_t f2bf(float f) {
    const unsigned u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }
__device__ __forceinline__ int lane_id() { return (int)(threadIdx.x & 31u); }
__device__ __forceinline__ float shfl_xor(float v, int m) {
    return __int_as_float(__builtin_amdgcn_ds_bpermute((lane_id() ^ m) << 2, __float_as_int(v)));
}
/* the block's sum, every thread the same value: a wave butterfly, then the waves in order */
__device__ __forceinline__ float block_sum(float v, float* lds) {
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) v += shfl_xor(v, d);
    const int lane = lane_id(), wave = (int)(threadIdx.x >> 5);
    __syncthreads();                                 /* lds is free: the last read of it is done */
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < kWaves; ++w) s += lds[w];
    return s;
}
/* two block sums for one barrier pair: each value takes block_sum's butterfly and wave order */
__device__ __forceinline__ void block_sum2(float& a, float& b, float* lds) {
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) { a += shfl_xor(a, d); b += shfl_xor(b, d); }
    const int lane = lane_id(), wave = (int)(threadIdx.x >> 5);
    __syncthreads();
    if (lane == 0) { lds[wave] = a; lds[kWaves + wave] = b; }
    __syncthreads();
    float sa = 0.0f, sb = 0.0f;
#pragma unroll
    for (int w = 0; w < kWaves; ++w) { sa += lds[w]; sb += lds[kWaves + w]; }
    a = sa;
    b = sb;
}
__device__ __forceinline__ float norm_factor(float ss, int n, float eps) {
    return rsqrtf(ss / (float)n + eps);
}

struct NormArgs {
    const uint16_t* a;  long long a_ld;     /* norm_add_norm: the block output */
    const uint16_t* d;  long long d_ld;     /* join_add_scale: the dense partial */
    const uint16_t* r;  long long r_ld;     /*                 the routed partial */
    uint16_t*       x;  long long x_ld;     /* the residual stream, read and written */
    uint16_t*       y;  long long y_ld;     /* norm_add_norm: the next norm's output */
    const float*    w0;                     /* w_a, or w_d */
    const float*    w1;                     /* w_y, or w_r */
    const float*    w2;                     /*        w_j */
    const float*    w3;                     /*        w_n, the next norm's gain (or null) */
    const uint16_t* s;                      /*        the layer scalar, bf16, one element */
    /* norm_add_norm_route: the router's input, the experts' codes and their scales; the router's
     * gain is w2 and the experts' w3 */
    uint16_t*       rt; long long rt_ld;
    uint8_t*        q;  long long q_ld;
    float*          qs; long long qs_ld;
    float mult;
    int   n;
    float eps;
};

/* t = norm(a) * w_a;  x += t;  y = norm(x) * w_y -- and with ROUTE, from the same norm of x, the
 * router's input norm(x) * w_r * mult and the experts' norm(x) * w_e as codes and scales: E4M3
 * (ROUTE 1), or at ROUTE 2 had_quant_act_i8g's rotated int8. The rotation's butterfly wants a
 * group in one wave, four columns a lane, where this kernel holds a column a thread, so the
 * experts' bf16 row goes through LDS to that mapping (r4d_q8.h, q8_row_had_i8g). */
template <int PER, int ROUTE>
__global__ __launch_bounds__(kThreads) void norm_add_norm_kernel(NormArgs A) {
    __shared__ float lds[kWaves];
    __shared__ float gmax[ROUTE == 1 ? PER * kWaves : 1];
    __shared__ uint16_t erow[ROUTE == 2 ? PER * kThreads : 1];
    const int tid = (int)threadIdx.x, n = A.n;
    const long long m = (long long)blockIdx.x;
    const uint16_t* ar = A.a + m * A.a_ld;
    uint16_t* xr = A.x + m * A.x_ld;
    uint16_t* yr = A.y + m * A.y_ld;

    /* every operand of the row in flight at once: the block output, the residual, the gains */
    float av[PER], xv[PER], wa[PER], wy[PER], wr[ROUTE ? PER : 1], we[ROUTE ? PER : 1];
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int i = tid + k * kThreads;
        const bool in = i < n;
        av[k] = in ? bf2f(ar[i]) : 0.0f;
        xv[k] = in ? bf2f(xr[i]) : 0.0f;
        wa[k] = in ? A.w0[i] : 0.0f;
        wy[k] = in ? A.w1[i] : 0.0f;
        if constexpr (ROUTE) {
            wr[k] = in ? A.w2[i] : 0.0f;
            we[k] = in ? A.w3[i] : 0.0f;
        }
    }
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < PER; ++k) ss += av[k] * av[k];
    const float sa = norm_factor(block_sum(ss, lds), n, A.eps);
    ss = 0.0f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int i = tid + k * kThreads;
        if (i < n) {
            const float t = rbf((av[k] * sa) * wa[k]);
            xv[k] = rbf(xv[k] + t);
            xr[i] = f2bf(xv[k]);
        }
        ss += xv[k] * xv[k];
    }
    const float sx = norm_factor(block_sum(ss, lds), n, A.eps);
    uint16_t* rr = ROUTE ? A.rt + m * A.rt_ld : nullptr;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int i = tid + k * kThreads;
        if (i < n) {
            yr[i] = f2bf((xv[k] * sx) * wy[k]);
            if constexpr (ROUTE) {
                rr[i] = f2bf(((xv[k] * sx) * wr[k]) * A.mult);
                const uint16_t he = f2bf((xv[k] * sx) * we[k]);
                av[k] = bf2f(he);                    /* the experts' row, as quantised */
                if constexpr (ROUTE == 2) erow[i] = he;
            }
        } else if constexpr (ROUTE) {
            av[k] = 0.0f;
        }
    }
    if constexpr (ROUTE == 2) {
        __syncthreads();
        q8_row_had_i8g<kThreads>(erow, (signed char*)(A.q + m * A.q_ld), A.qs + m * A.qs_ld,
                                 (uint32_t)n);
    }
    if constexpr (ROUTE == 1) {
        /* group 2k + tid / 128 at slot k is waves 0..3 or 4..7: each wave's max, then its four */
        const int wave = tid >> 5, half = tid >> 7;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            float mx = fabsf(av[k]);
#pragma unroll
            for (int d = 16; d >= 1; d >>= 1) mx = fmaxf(mx, shfl_xor(mx, d));
            if (lane_id() == 0) gmax[k * kWaves + wave] = mx;
        }
        __syncthreads();
        uint8_t* qr = A.q + m * A.q_ld;
        float* sr = A.qs + m * A.qs_ld;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int i = tid + k * kThreads;
            if (i >= n) continue;                    /* n is whole groups: a group is in or out */
            const float* g = gmax + k * kWaves + 4 * half;
            const float am = fmaxf(fmaxf(g[0], g[1]), fmaxf(g[2], g[3]));
            const float sc = am > 0.0f ? am * (1.0f / 448.0f) : 1.0f;
            const float inv = 1.0f / sc;
            const float v = fminf(fmaxf(av[k] * inv, -448.0f), 448.0f);
            qr[i] = (uint8_t)(__builtin_amdgcn_cvt_pk_fp8_f32(v, v, 0, false) & 0xff);
            if ((tid & 127) == 0) sr[i >> 7] = sc;
        }
    }
}

/* x = (x + norm(norm(d) * w_d + norm(r) * w_r) * w_j) * s, and with NEXT y = norm(x) * w_n: the
 * next layer's input norm taken over the bf16 x this stores, which is what that norm would read. */
template <int PER, bool NEXT>
__global__ __launch_bounds__(kThreads) void join_add_scale_kernel(NormArgs A) {
    __shared__ float lds[2 * kWaves];
    const int tid = (int)threadIdx.x, n = A.n;
    const long long m = (long long)blockIdx.x;
    const uint16_t* dr = A.d + m * A.d_ld;
    const uint16_t* rr = A.r + m * A.r_ld;
    uint16_t* xr = A.x + m * A.x_ld;

    /* every operand of the row in flight at once: both arms, the residual, the three gains (four
     * with the next norm's) and the layer scalar */
    float dv[PER], rv[PER], xo[PER], wd[PER], wr[PER], wj[PER], wn[NEXT ? PER : 1];
    const float sc = bf2f(*A.s);
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int i = tid + k * kThreads;
        const bool in = i < n;
        dv[k] = in ? bf2f(dr[i]) : 0.0f;
        rv[k] = in ? bf2f(rr[i]) : 0.0f;
        xo[k] = in ? bf2f(xr[i]) : 0.0f;
        wd[k] = in ? A.w0[i] : 0.0f;
        wr[k] = in ? A.w1[i] : 0.0f;
        wj[k] = in ? A.w2[i] : 0.0f;
        if constexpr (NEXT) wn[k] = in ? A.w3[i] : 0.0f;
    }
    float sd2 = 0.0f, sr2 = 0.0f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        sd2 += dv[k] * dv[k];
        sr2 += rv[k] * rv[k];
    }
    block_sum2(sd2, sr2, lds);
    const float sd = norm_factor(sd2, n, A.eps);
    const float sr = norm_factor(sr2, n, A.eps);
    /* the join, kept in dv */
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int i = tid + k * kThreads;
        if (i < n) {
            const float u = rbf((dv[k] * sd) * wd[k]);
            const float v = rbf((rv[k] * sr) * wr[k]);
            dv[k] = rbf(u + v);
        }
        ss += dv[k] * dv[k];
    }
    const float sj = norm_factor(block_sum(ss, lds), n, A.eps);
    ss = 0.0f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int i = tid + k * kThreads;
        if (i < n) {
            const float j = rbf((dv[k] * sj) * wj[k]);
            const float x1 = rbf(xo[k] + j);
            const uint16_t xn = f2bf(x1 * sc);
            xr[i] = xn;
            rv[k] = bf2f(xn);                        /* the new x, kept for the next norm */
        } else {
            rv[k] = 0.0f;
        }
        if (NEXT) ss += rv[k] * rv[k];
    }
    if (NEXT) {
        const float sn = norm_factor(block_sum(ss, lds), n, A.eps);
        uint16_t* yr = A.y + m * A.y_ld;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int i = tid + k * kThreads;
            if (i < n) yr[i] = f2bf((rv[k] * sn) * wn[k]);
        }
    }
}

/* ------------------------------------------------------------------ host: the launch adapters */

/* [rows..., n] of `dt` with a dense last axis: its row count and pitch, or false */
static bool rows_as(const RadTensor* t, uint32_t dt, int64_t n, int64_t* rows, int64_t* ld) {
    if (!t || t->dtype != dt || t->rank < 1) return false;
    if (rad_tensor_lastdim(t) != n || (t->stride[t->rank - 1] != 1)) return false;
    *rows = rad_tensor_rows(t);
    *ld = rad_tensor_row_pitch(t);
    return *ld >= n || *rows <= 1;
}
static bool rows_of(const RadTensor* t, int64_t n, int64_t* rows, int64_t* ld) {
    return rows_as(t, RAD_BF16, n, rows, ld);
}
static bool gain(const RadTensor* w, int64_t n) {
    return w && w->dtype == RAD_F32 && rad_tensor_numel(w) == n &&
           (w->rank == 0 || w->stride[w->rank - 1] == 1);
}

static int params(const RadArgs* a, int64_t* n, float* eps) {
    *n = rad_args_geti_or(a, "n", 0);
    *eps = (float)rad_args_getf_or(a, "eps", 1e-6);
    if (*n <= 0 || *n > (int64_t)kThreads * kMaxPer) return RAD_E_SHAPE;
    return RAD_OK;
}

}  /* namespace r4d_join */

extern "C" {

/* operands: a, w_a, x (inout), w_y, y */
int r4d_rad_norm_add_norm_bf16(const RadArgs* a, RadStream stream) {
    using namespace r4d_join;
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    int64_t n = 0;
    float eps = 0.0f;
    if (int rc = params(a, &n, &eps)) return rc;
    const RadTensor *A_ = rad_arg_in(a, 0), *Wa = rad_arg_in(a, 1), *X = rad_arg_in(a, 2),
                    *Wy = rad_arg_in(a, 3), *Y = rad_arg_in(a, 4);
    if (!A_ || !X || !Y) return RAD_E_INVAL;
    if (!gain(Wa, n) || !gain(Wy, n)) return RAD_E_DTYPE;
    int64_t ra = 0, la = 0, rx = 0, lx = 0, ry = 0, ly = 0;
    if (!rows_of(A_, n, &ra, &la) || !rows_of(X, n, &rx, &lx) || !rows_of(Y, n, &ry, &ly))
        return RAD_E_SHAPE;
    const int64_t M = rx;                            /* the step's rows: the stream's view */
    if (M <= 0) return RAD_OK;
    if (ra < M || ry < M || M > 0x7fffffff) return RAD_E_SHAPE;
    NormArgs P{};
    P.a = (const uint16_t*)A_->data; P.a_ld = la;
    P.x = (uint16_t*)X->data;        P.x_ld = lx;
    P.y = (uint16_t*)Y->data;        P.y_ld = ly;
    P.w0 = (const float*)Wa->data;   P.w1 = (const float*)Wy->data;
    P.n = (int)n; P.eps = eps;
    if (n <= (int64_t)kThreads * 12)
        R4D_KLAUNCH(R4D_KNAME(norm_add_norm_kernel<12, 0>), dim3((unsigned)M), dim3(kThreads),
                   (hipStream_t)stream, P);
    else
        R4D_KLAUNCH(R4D_KNAME(norm_add_norm_kernel<kMaxPer, 0>), dim3((unsigned)M),
                   dim3(kThreads), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* extern "C" */

/* operands: a, w_a, x (inout), w_y, y, w_r, r, w_e, q, qs; `i8` is norm_add_norm_route_i8g, whose
 * q is int8 */
static int norm_add_norm_route(const RadArgs* a, RadStream stream, bool i8) {
    using namespace r4d_join;
    if (!a || a->n_t < 10) return RAD_E_INVAL;
    int64_t n = 0;
    float eps = 0.0f;
    if (int rc = params(a, &n, &eps)) return rc;
    if (n % 128 || rad_args_geti_or(a, "group", 0) != 128) return RAD_E_SHAPE;
    const RadTensor *A_ = rad_arg_in(a, 0), *Wa = rad_arg_in(a, 1), *X = rad_arg_in(a, 2),
                    *Wy = rad_arg_in(a, 3), *Y = rad_arg_in(a, 4), *Wr = rad_arg_in(a, 5),
                    *R = rad_arg_in(a, 6), *We = rad_arg_in(a, 7), *Q = rad_arg_in(a, 8),
                    *S = rad_arg_in(a, 9);
    if (!A_ || !X || !Y || !R || !Q || !S) return RAD_E_INVAL;
    if (!gain(Wa, n) || !gain(Wy, n) || !gain(Wr, n) || !gain(We, n)) return RAD_E_DTYPE;
    int64_t ra = 0, la = 0, rx = 0, lx = 0, ry = 0, ly = 0, rr = 0, lr = 0, rq = 0, lq = 0,
            rs = 0, ls = 0;
    if (!rows_of(A_, n, &ra, &la) || !rows_of(X, n, &rx, &lx) || !rows_of(Y, n, &ry, &ly) ||
        !rows_of(R, n, &rr, &lr) || !rows_as(Q, i8 ? RAD_I8 : RAD_F8E4M3, n, &rq, &lq) ||
        !rows_as(S, RAD_F32, n / 128, &rs, &ls))
        return RAD_E_SHAPE;
    const int64_t M = rx;                            /* the step's rows: the stream's view */
    if (M <= 0) return RAD_OK;
    if (ra < M || ry < M || rr < M || rq < M || rs < M || M > 0x7fffffff) return RAD_E_SHAPE;
    NormArgs P{};
    P.a = (const uint16_t*)A_->data; P.a_ld = la;
    P.x = (uint16_t*)X->data;        P.x_ld = lx;
    P.y = (uint16_t*)Y->data;        P.y_ld = ly;
    P.rt = (uint16_t*)R->data;       P.rt_ld = lr;
    P.q = (uint8_t*)Q->data;         P.q_ld = lq;
    P.qs = (float*)S->data;          P.qs_ld = ls;
    P.w0 = (const float*)Wa->data;   P.w1 = (const float*)Wy->data;
    P.w2 = (const float*)Wr->data;   P.w3 = (const float*)We->data;
    P.mult = (float)rad_args_getf_or(a, "mult", 1.0);
    P.n = (int)n; P.eps = eps;
    const bool narrow = n <= (int64_t)kThreads * 12;
    if (narrow && i8)
        R4D_KLAUNCH(R4D_KNAME(norm_add_norm_kernel<12, 2>), dim3((unsigned)M), dim3(kThreads),
                   (hipStream_t)stream, P);
    else if (narrow)
        R4D_KLAUNCH(R4D_KNAME(norm_add_norm_kernel<12, 1>), dim3((unsigned)M), dim3(kThreads),
                   (hipStream_t)stream, P);
    else if (i8)
        R4D_KLAUNCH(R4D_KNAME(norm_add_norm_kernel<kMaxPer, 2>), dim3((unsigned)M),
                   dim3(kThreads), (hipStream_t)stream, P);
    else
        R4D_KLAUNCH(R4D_KNAME(norm_add_norm_kernel<kMaxPer, 1>), dim3((unsigned)M),
                   dim3(kThreads), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" {

int r4d_rad_norm_add_norm_route_bf16(const RadArgs* a, RadStream stream) {
    return norm_add_norm_route(a, stream, false);
}
int r4d_rad_norm_add_norm_route_i8g_bf16(const RadArgs* a, RadStream stream) {
    return norm_add_norm_route(a, stream, true);
}

/* operands: d, w_d, r, w_r, w_j, x (inout), s, w_n?, y? -- the last two both or neither */
int r4d_rad_join_add_scale_bf16(const RadArgs* a, RadStream stream) {
    using namespace r4d_join;
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    int64_t n = 0;
    float eps = 0.0f;
    if (int rc = params(a, &n, &eps)) return rc;
    const RadTensor *D = rad_arg_in(a, 0), *Wd = rad_arg_in(a, 1), *R = rad_arg_in(a, 2),
                    *Wr = rad_arg_in(a, 3), *Wj = rad_arg_in(a, 4), *X = rad_arg_in(a, 5),
                    *Sc = rad_arg_in(a, 6);
    if (!D || !R || !X || !Sc) return RAD_E_INVAL;
    if (!gain(Wd, n) || !gain(Wr, n) || !gain(Wj, n)) return RAD_E_DTYPE;
    if (Sc->dtype != RAD_BF16 || rad_tensor_numel(Sc) != 1) return RAD_E_DTYPE;
    const RadTensor *Wn = a->n_t > 7 ? rad_arg_in(a, 7) : nullptr,
                    *Y  = a->n_t > 8 ? rad_arg_in(a, 8) : nullptr;
    if (!Wn != !Y) return RAD_E_INVAL;
    if (Wn && !gain(Wn, n)) return RAD_E_DTYPE;
    int64_t rd = 0, ld = 0, rr = 0, lr = 0, rx = 0, lx = 0, ry = 0, ly = 0;
    if (!rows_of(D, n, &rd, &ld) || !rows_of(R, n, &rr, &lr) || !rows_of(X, n, &rx, &lx))
        return RAD_E_SHAPE;
    if (Y && !rows_of(Y, n, &ry, &ly)) return RAD_E_SHAPE;
    const int64_t M = rx;                            /* the step's rows: the stream's view */
    if (M <= 0) return RAD_OK;
    if (rd < M || rr < M || (Y && ry < M) || M > 0x7fffffff) return RAD_E_SHAPE;
    NormArgs P{};
    P.d = (const uint16_t*)D->data;  P.d_ld = ld;
    P.r = (const uint16_t*)R->data;  P.r_ld = lr;
    P.x = (uint16_t*)X->data;        P.x_ld = lx;
    P.w0 = (const float*)Wd->data;   P.w1 = (const float*)Wr->data;   P.w2 = (const float*)Wj->data;
    P.s = (const uint16_t*)Sc->data;
    if (Y) { P.w3 = (const float*)Wn->data; P.y = (uint16_t*)Y->data; P.y_ld = ly; }
    P.n = (int)n; P.eps = eps;
    const bool narrow = n <= (int64_t)kThreads * 12;
    if (narrow && Y)
        R4D_KLAUNCH(R4D_KNAME(join_add_scale_kernel<12, true>), dim3((unsigned)M), dim3(kThreads),
                   (hipStream_t)stream, P);
    else if (narrow)
        R4D_KLAUNCH(R4D_KNAME(join_add_scale_kernel<12, false>), dim3((unsigned)M), dim3(kThreads),
                   (hipStream_t)stream, P);
    else if (Y)
        R4D_KLAUNCH(R4D_KNAME(join_add_scale_kernel<kMaxPer, true>), dim3((unsigned)M),
                   dim3(kThreads), (hipStream_t)stream, P);
    else
        R4D_KLAUNCH(R4D_KNAME(join_add_scale_kernel<kMaxPer, false>), dim3((unsigned)M),
                   dim3(kThreads), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* extern "C" */
