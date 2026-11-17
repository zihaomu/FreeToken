// r4d_ple_gate_bf16.hip -- Per-Layer Embedding's dot-product gate.
//
//   K[c] = grouped_rmsnorm(k)[c] * (w_key   + wadd)     k is key_proj(embedding)
//   Q[c] = grouped_rmsnorm(q)[c] * (w_query + wadd)     q is the incoming hc stream
//   s[c] = sum_d K[c][d] * Q[c][d] / sqrt(n)
//   s[c] = sign(s[c]) * sqrt(max(|s[c]|, gate_eps))     <- the signed square root
//   gv[c] = sigmoid(s[c]) * v                           v is value_proj(embedding), one row
//   gvn   = grouped_rmsnorm(gv) * (w_conv + wadd)
//
// libref/ref_ple.cpp is the oracle and docs/OPS.md the contract. The sign comes from the original
// value, not the clamped magnitude: torch's sign(0) is 0, so an exactly-zero dot (as a quantised
// activation row produces) stays zero instead of becoming sqrt(gate_eps).
//
// One workgroup per (row, stream): every reduction (two norms, the dot, the gate's rms) is over
// one stream's `n`, and a workgroup is the widest unit that reduces without a second launch. At
// decode that is four workgroups, which is fine: the op moves 107 KB a row and is latency bound
// well before occupancy matters.
//
// Each thread keeps its n/threads elements of k and q in registers across the norm and the dot,
// and `gv` from its store to its rms, so nothing is read twice. `gv` is narrowed to bf16 before
// its rms, because the reference norms the bf16 tensor it wrote; keeping the f32 value would give
// a different number.
//
// Invariant: gv[c] is a scalar times one shared v row and norm_conv's rms is per stream, so the
// gate cancels out of `gvn`. This holds only in exact arithmetic (gv is narrowed first), so it is a
// property to check a fused kernel against, never one to fold.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_common.h"

#include <cstdint>

namespace {

enum { kPgThreads = 256 };

/* A value narrowed to bf16 and read back, as the reference's `.type_as(x)` after every norm. */
__device__ __forceinline__ float pg_narrow(float v) {
    return bf16_to_f32(f32_to_bf16(v));
}

/* expf, not __expf: this gate is compared against libref's std::exp with a tolerance meant for
 * summation order. */
__device__ __forceinline__ float pg_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* A workgroup-wide sum through LDS. The trailing barrier lets the next call reuse the array. */
__device__ __forceinline__ float pg_reduce(float v, float* lds) {
    const uint32_t t = threadIdx.x;
    lds[t] = v;
    __syncthreads();
    for (uint32_t w = kPgThreads / 2; w > 0; w >>= 1) {
        if (t < w) lds[t] += lds[t + w];
        __syncthreads();
    }
    const float r = lds[0];
    __syncthreads();
    return r;
}

#define PLE_GATE_PARAMS                                                                         \
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ q,                             \
    const uint16_t* __restrict__ v, const float* __restrict__ wk,                               \
    const float* __restrict__ wq, const float* __restrict__ wc, uint16_t* __restrict__ gv,      \
    uint16_t* __restrict__ gvn, uint32_t n, uint32_t hc, uint32_t k_ld, uint32_t q_ld,          \
    uint32_t v_ld, uint32_t g_ld, uint32_t gn_ld, float eps, float wadd, float geps,            \
    uint32_t per_thread
__device__ __forceinline__ void ple_gate_body(PLE_GATE_PARAMS) {

    __shared__ float lds[kPgThreads];
    /* At most 16 elements per thread: n <= 16 * kPgThreads = 4096, which the row constrains. */
    float mine_k[16], mine_q[16], mine_gv[16];

    const uint32_t row = blockIdx.x / hc;
    const uint32_t c   = blockIdx.x % hc;
    const uint32_t t   = threadIdx.x;
    const uint32_t base = c * n;                 /* this stream's first column of the wide row */

    /* ---- the two norms, from values each thread then keeps ------------------------------------ */
    float sk = 0.0f, sq = 0.0f;
    for (uint32_t i = 0; i < per_thread; ++i) {
        const uint32_t d = i * kPgThreads + t;
        if (d >= n) break;
        const float kv = bf16_to_f32(k[(size_t)row * k_ld + base + d]);
        const float qv = bf16_to_f32(q[(size_t)row * q_ld + base + d]);
        mine_k[i] = kv;
        mine_q[i] = qv;
        sk += kv * kv;
        sq += qv * qv;
    }
    const float rk = rsqrtf(pg_reduce(sk, lds) / (float)n + eps);
    const float rq = rsqrtf(pg_reduce(sq, lds) / (float)n + eps);

    /* ---- the dot between them, over the NARROWED normed values -------------------------------- */
    float dot = 0.0f;
    for (uint32_t i = 0; i < per_thread; ++i) {
        const uint32_t d = i * kPgThreads + t;
        if (d >= n) break;
        const float kk = pg_narrow(mine_k[i] * rk * (wk[base + d] + wadd));
        const float qq = pg_narrow(mine_q[i] * rq * (wq[base + d] + wadd));
        dot += kk * qq;
    }
    float s = pg_reduce(dot, lds) * rsqrtf((float)n);

    /* The signed square root in the reference's order: clamp the magnitude, take the root, take the
     * sign from the original. */
    const float mag = sqrtf(fabsf(s) < geps ? geps : fabsf(s));
    s = s > 0.0f ? mag : (s < 0.0f ? -mag : 0.0f);
    const float gate = pg_sigmoid(s);

    /* ---- the gated value, narrowed, and its own rms from the narrowed bytes ------------------- */
    float sg = 0.0f;
    for (uint32_t i = 0; i < per_thread; ++i) {
        const uint32_t d = i * kPgThreads + t;
        if (d >= n) break;
        const float x = pg_narrow(gate * bf16_to_f32(v[(size_t)row * v_ld + d]));
        mine_gv[i] = x;
        gv[(size_t)row * g_ld + base + d] = f32_to_bf16(x);
        sg += x * x;
    }
    const float rg = rsqrtf(pg_reduce(sg, lds) / (float)n + eps);

    for (uint32_t i = 0; i < per_thread; ++i) {
        const uint32_t d = i * kPgThreads + t;
        if (d >= n) break;
        gvn[(size_t)row * gn_ld + base + d] =
            f32_to_bf16(pg_narrow(mine_gv[i] * rg * (wc[base + d] + wadd)));
    }
}

/* The kernel around the body, whose blocks return where their work ends: CH `r4dchain` is the
 * chained form (r4d_chain.h), taken at decode heights. */
template <class CH>
__global__ __launch_bounds__(kPgThreads)
void ple_gate_kernel(PLE_GATE_PARAMS) {
    if (CH::on) r4d_chain_wait();
    ple_gate_body(k, q, v, wk, wq, wc, gv, gvn, n, hc, k_ld, q_ld, v_ld, g_ld, gn_ld, eps, wadd, geps, per_thread);
    if (CH::on) r4d_chain_done();
}
#undef PLE_GATE_PARAMS

enum { PG_K = 0, PG_Q, PG_V, PG_WK, PG_WQ, PG_WC, PG_GV, PG_GVN };

}  // namespace

extern "C" int r4d_rad_ple_gate(const RadArgs* a, RadStream s) {
    const RadTensor* k   = r4d_opt(a, PG_K);
    const RadTensor* q   = r4d_opt(a, PG_Q);
    const RadTensor* v   = r4d_opt(a, PG_V);
    const RadTensor* wk  = r4d_opt(a, PG_WK);
    const RadTensor* wq  = r4d_opt(a, PG_WQ);
    const RadTensor* wc  = r4d_opt(a, PG_WC);
    const RadTensor* gv  = r4d_opt(a, PG_GV);
    const RadTensor* gvn = r4d_opt(a, PG_GVN);
    if (!k || !q || !v || !wk || !wq || !wc || !gv || !gvn) return R4D_NO(RAD_E_INVAL);
    if (k->dtype != RAD_BF16 || q->dtype != RAD_BF16 || v->dtype != RAD_BF16 ||
        gv->dtype != RAD_BF16 || gvn->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (wk->dtype != RAD_F32 || wq->dtype != RAD_F32 || wc->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);

    const long long n  = r4d_geti_or(a, "n", 0);
    const long long hc = r4d_geti_or(a, "hc", 0);
    if (n <= 0 || hc <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long hn = hc * n;
    const long long per_thread = (n + kPgThreads - 1) / kPgThreads;
    if (per_thread > 16) return R4D_NO(RAD_E_UNSUPPORTED);

    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);
    const float geps = (float)r4d_getf(a, "gate_eps", 1e-6);

    long long T = 0, tq = 0, tv = 0, tg = 0, tn = 0;
    const long long k_ld  = r4d_rows_pitch(k, hn, &T);
    const long long q_ld  = r4d_rows_pitch(q, hn, &tq);
    const long long v_ld  = r4d_rows_pitch(v, n, &tv);
    const long long g_ld  = r4d_rows_pitch(gv, hn, &tg);
    const long long gn_ld = r4d_rows_pitch(gvn, hn, &tn);
    if (k_ld < 0 || q_ld < 0 || v_ld < 0 || g_ld < 0 || gn_ld < 0) return R4D_NO(RAD_E_STRIDE);
    /* The row count comes from the operands, and they must agree: `M` is ranged, so the arg holds
     * the band's upper bound and a grid sized by it would run off the arena. */
    if (T <= 0 || tq != T || tv != T || tg != T || tn != T) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(wk) < hn || r4d_numel(wq) < hn || r4d_numel(wc) < hn)
        return R4D_NO(RAD_E_SHAPE);

#define R4D_PG(CH_)                                                                             \
    hipLaunchKernelGGL(ple_gate_kernel<CH_>, dim3((uint32_t)(T * hc)), dim3(kPgThreads), 0,      \
                       (hipStream_t)s,                                                           \
                       (const uint16_t*)r4d_addr(k), (const uint16_t*)r4d_addr(q),               \
                       (const uint16_t*)r4d_addr(v), (const float*)r4d_addr(wk),                 \
                       (const float*)r4d_addr(wq), (const float*)r4d_addr(wc),                   \
                       (uint16_t*)r4d_addr(gv), (uint16_t*)r4d_addr(gvn),                        \
                       (uint32_t)n, (uint32_t)hc, (uint32_t)k_ld, (uint32_t)q_ld, (uint32_t)v_ld, \
                       (uint32_t)g_ld, (uint32_t)gn_ld, eps, wadd, geps, (uint32_t)per_thread)
    if (r4d_geti_or(a, "chain", 0) != 0 && T <= 64) R4D_PG(r4dchain);
    else                                          R4D_PG(R4dUnchained);
#undef R4D_PG
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}
