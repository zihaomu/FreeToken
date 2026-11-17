/* r4d_lili.hip -- LiLiCorr's candidate correlator (libref/ref_lili.cpp has the contracts):
 * lili_cand, lili_attn, lili_factor and lili_walk, at a draft step's size (seven positions of eight
 * candidates a sequence, a 1024-wide lattice). Each is a few microseconds of a draft step, so the
 * kernels are plain: one workgroup a row (lili_cand, lili_factor), a (sequence, head) pair
 * (lili_attn) or a sequence (lili_walk), shared memory and serial loops where a step is short, f32
 * arithmetic in libref's order and one rounding to bf16 at each store.
 *
 * Activations are bf16; lili_cand's statistics, LayerNorm and up projection are f32 and its slot and
 * rank tables bf16; lili_attn's biases f32. Rows may have their own pitch, the last axis dense.
 * Nothing here allocates or synchronises (docs/PLUGIN.md §5).
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "rad_plugin.h"

namespace r4d_lili {

constexpr int kThr = 256;
constexpr int kMaxK = 64;        /* candidates a position (lili_cand, lili_walk) */
constexpr int kMaxW = 16;        /* ranks (lili_cand) */
constexpr int kMaxL = 64;        /* lattice rows a sequence (lili_attn) */
constexpr int kMaxHd = 128;      /* head width (lili_attn) */
constexpr int kMaxWalk = 512;    /* steps * k * k scores a sequence (lili_walk) */

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }
__device__ __forceinline__ uint16_t tobf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);   /* NaN stays NaN */
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

/* ------------------------------------------------------------------ lili_cand */
struct CandArgs {
    const float*    st;   long long st_rs;
    const uint16_t* pas;  long long pas_rs;
    const float *lnw, *lnb, *upw, *upb;
    const uint16_t *slot, *rank;
    int*            cand; long long cand_rs;
    uint16_t*       feat; long long feat_rs;
    uint16_t*       base; long long base_rs;
    int k, row_k, world, steps, n;
    float eps;
};

__global__ __launch_bounds__(kThr) void cand_kernel(CandArgs A) {
    __shared__ float s_y[kMaxK][5];
    const long long m = blockIdx.x;
    const int k = A.k, w = 2 + 2 * A.row_k;
    if (threadIdx.x == 0) {
        const float* row = A.st + m * A.st_rs;
        float mx = -__builtin_inff();
        for (int r = 0; r < A.world; ++r) mx = __builtin_fmaxf(mx, row[r * w]);
        float sum = 0.0f;
        for (int r = 0; r < A.world; ++r)
            if (row[r * w] > -__builtin_inff()) sum += row[r * w + 1] * expf(row[r * w] - mx);
        const float lse = mx + logf(sum);
        /* A world-way merge of lists sorted by value, ties to the lower id. */
        int at[kMaxW];
        for (int r = 0; r < A.world; ++r) at[r] = 0;
        float lp[kMaxK];
        for (int j = 0; j < k; ++j) {
            int br = -1, bid = 0;
            float bv = 0.0f;
            for (int r = 0; r < A.world; ++r) {
                while (at[r] < k && (int)row[r * w + 2 + 2 * at[r]] < 0) ++at[r];
                if (at[r] >= k) continue;
                const float v = row[r * w + 3 + 2 * at[r]];
                const int id = (int)row[r * w + 2 + 2 * at[r]];
                if (br < 0 || v > bv || (v == bv && id < bid)) { br = r; bv = v; bid = id; }
            }
            lp[j] = br < 0 ? -1e4f : bv - lse;
            if (br >= 0) ++at[br];
            A.cand[m * A.cand_rs + j] = br < 0 ? 0 : bid;
        }
        for (int j = 0; j < k; ++j) {
            const float f[5] = { lp[j], expf(lp[j]), lp[j] - lp[0], (float)j / (float)(k - 1),
                                 j == 0 ? 1.0f : 0.0f };
            float mean = 0.0f;
            for (int i = 0; i < 5; ++i) mean += f[i];
            mean /= 5.0f;
            float var = 0.0f;
            for (int i = 0; i < 5; ++i) var += (f[i] - mean) * (f[i] - mean);
            var /= 5.0f;
            const float rs = 1.0f / __builtin_sqrtf(var + A.eps);
            for (int i = 0; i < 5; ++i) s_y[j][i] = (f[i] - mean) * rs * A.lnw[i] + A.lnb[i];
        }
    }
    __syncthreads();
    const int s = (int)(m % A.steps);
    for (int c = (int)threadIdx.x; c < A.n; c += kThr) {
        const float pv = bf(A.pas[m * A.pas_rs + c]) + bf(A.slot[(long long)s * A.n + c]);
        float uw[5];
        for (int i = 0; i < 5; ++i) uw[i] = A.upw[(long long)c * 5 + i];
        const float ub = A.upb[c];
        for (int j = 0; j < k; ++j) {
            float u = ub;
            for (int i = 0; i < 5; ++i) u += uw[i] * s_y[j][i];
            const long long row = m * k + j;
            A.feat[row * A.feat_rs + c] = tobf(u / (1.0f + expf(-u)));
            A.base[row * A.base_rs + c] = tobf(pv + bf(A.rank[(long long)j * A.n + c]));
        }
    }
}

/* ------------------------------------------------------------------ lili_attn */
struct AttnArgs {
    const uint16_t* qkv; long long q_rs;
    const float *rel, *same; int rel_w;
    uint16_t* out; long long o_rs;
    int steps, k, nh, hd;
    float scale;
};

__global__ __launch_bounds__(kThr) void attn_kernel(AttnArgs A) {
    __shared__ uint16_t s_k[kMaxL][kMaxHd], s_v[kMaxL][kMaxHd];
    __shared__ float s_q[kMaxHd], s_sc[kMaxL], s_sum;
    const int L = A.steps * A.k, hd = A.hd, n = A.nh * hd, h = (int)blockIdx.y;
    const long long r0 = (long long)blockIdx.x * L;
    const int tid = (int)threadIdx.x;
    for (int e = tid; e < L * hd; e += kThr) {
        const int j = e / hd, d = e - j * hd;
        const uint16_t* row = A.qkv + (r0 + j) * A.q_rs;
        s_k[j][d] = row[n + h * hd + d];
        s_v[j][d] = row[2 * n + h * hd + d];
    }
    const float sm = A.same[h];
    for (int i = 0; i < L; ++i) {
        __syncthreads();
        if (tid < hd) s_q[tid] = bf(A.qkv[(r0 + i) * A.q_rs + h * hd + tid]);
        __syncthreads();
        const int si = i / A.k;
        if (tid < L) {
            float dot = 0.0f;
            for (int d = 0; d < hd; ++d) dot += s_q[d] * bf(s_k[tid][d]);
            const int sj = tid / A.k;
            float b = A.rel[(long long)h * A.rel_w + (si - sj + A.steps)];
            if (si == sj) b += sm;
            s_sc[tid] = dot * A.scale + b;
        }
        __syncthreads();
        if (tid == 0) {
            float mx = s_sc[0];
            for (int j = 1; j < L; ++j) mx = __builtin_fmaxf(mx, s_sc[j]);
            float sum = 0.0f;
            for (int j = 0; j < L; ++j) { s_sc[j] = expf(s_sc[j] - mx); sum += s_sc[j]; }
            s_sum = sum;
        }
        __syncthreads();
        if (tid < hd) {
            float acc = 0.0f;
            for (int j = 0; j < L; ++j) acc += s_sc[j] * bf(s_v[j][tid]);
            A.out[(r0 + i) * A.o_rs + h * hd + tid] = tobf(acc / s_sum);
        }
    }
}

/* ------------------------------------------------------------------ lili_factor */
struct FactorArgs {
    const uint16_t* h; long long h_rs;
    const uint16_t* a; long long a_rs;
    uint16_t* y; long long y_rs;
    int L, n;
};

__global__ __launch_bounds__(kThr) void factor_kernel(FactorArgs A) {
    const long long i = blockIdx.x;
    const uint16_t* hr = A.h + i * A.h_rs;
    const uint16_t* ar = A.a + (i / A.L) * A.a_rs;
    uint16_t* yr = A.y + i * A.y_rs;
    for (int c = (int)threadIdx.x; c < A.n; c += kThr) {
        const float hv = bf(hr[c]), av = bf(ar[c]);
        yr[c] = hr[c];
        yr[A.n + c] = ar[c];
        yr[2 * A.n + c] = tobf(hv * av);
    }
}

/* ------------------------------------------------------------------ lili_walk */
struct WalkArgs {
    const uint16_t* ed; long long e_rs;
    const uint16_t* ao; long long a_rs;
    const int* cand; long long c_rs;
    int* tok; long long t_rs;
    float* scores;                     /* [M, steps, k, k] dense, or null */
    int steps, k, n;
    float eps, scale;
};

__global__ __launch_bounds__(kThr) void walk_kernel(WalkArgs A) {
    /* norms: [0, R) outgoing, [R, 2R) incoming, 2R the anchor's */
    __shared__ float s_nrm[2 * kMaxK * 8 + 1];
    __shared__ float s_sc[kMaxWalk];
    const long long m = blockIdx.x;
    const int k = A.k, R = A.steps * k, n = A.n, tid = (int)threadIdx.x;
    const uint16_t* E = A.ed + m * R * A.e_rs;
    const uint16_t* an = A.ao + m * A.a_rs;
    for (int v = tid; v < 2 * R + 1; v += kThr) {
        const uint16_t* p = v < 2 * R ? E + (long long)(v % R) * A.e_rs + (v < R ? 0 : n) : an;
        float ss = 0.0f;
        for (int c = 0; c < n; ++c) { const float x = bf(p[c]); ss += x * x; }
        s_nrm[v] = __builtin_fmaxf(__builtin_sqrtf(ss), A.eps);
    }
    __syncthreads();
    /* start (k entries, shared by every i) and the pairs: dots of the raw vectors over their norms */
    const int npair = (A.steps - 1) * k * k;
    for (int e = tid; e < k + npair; e += kThr) {
        const uint16_t *x, *y;
        float nx, ny;
        if (e < k) {
            x = an; nx = s_nrm[2 * R];
            y = E + (long long)e * A.e_rs + n; ny = s_nrm[R + e];
        } else {
            const int q = e - k, s = q / (k * k) + 1, i = (q / k) % k, j = q % k;
            x = E + (long long)((s - 1) * k + i) * A.e_rs; nx = s_nrm[(s - 1) * k + i];
            y = E + (long long)(s * k + j) * A.e_rs + n;   ny = s_nrm[R + s * k + j];
        }
        float d = 0.0f;
        for (int c = 0; c < n; ++c) d += bf(x[c]) * bf(y[c]);
        const float v = A.scale * (d / nx / ny);
        if (e < k) {
            for (int i = 0; i < k; ++i) s_sc[i * k + e] = v;
        } else {
            s_sc[k * k + (e - k)] = v;
        }
    }
    __syncthreads();
    if (tid == 0) {
        int prev = 0;
        for (int s = 0; s < A.steps; ++s) {
            const float* row = s_sc + (s * k + prev) * k;
            int best = 0;
            for (int j = 1; j < k; ++j) if (row[j] > row[best]) best = j;
            A.tok[m * A.t_rs + s] = A.cand[(m * A.steps + s) * A.c_rs + best];
            prev = best;
        }
    }
    if (A.scores)
        for (int e = tid; e < R * k; e += kThr) A.scores[m * R * k + e] = s_sc[e];
}

/* A tensor read as [rows, cols] with a row pitch, last axis dense: the pitch, and the rows in *M. */
static long long pitch_of(const RadTensor* t, long long cols, long long* M) {
    if (!t || cols <= 0) return -1;
    if (t->rank == 1) {
        if (t->stride[0] != 1 || t->shape[0] % cols) return -1;
        *M = t->shape[0] / cols;
        return cols;
    }
    const uint32_t r = t->rank;
    if (r < 2 || t->stride[r - 1] != 1) return -1;
    long long rows = 1;
    for (uint32_t i = 0; i + 1 < r; ++i) rows *= t->shape[i];
    if (t->shape[r - 1] == cols) {
        for (uint32_t i = 0; i + 2 < r; ++i)
            if (t->shape[i] > 1 && t->stride[i] != t->stride[i + 1] * t->shape[i + 1]) return -1;
        *M = rows;
        return t->stride[r - 2];
    }
    /* A flat [rows, a * cols] read at a narrower width is not a pitch. */
    if (r == 2 && t->stride[0] == t->shape[1] && t->shape[1] % cols == 0) {
        *M = rows * (t->shape[1] / cols);
        return cols;
    }
    return -1;
}

static bool dense(const RadTensor* t) {
    int64_t s = 1;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        if (t->shape[i] != 1 && t->stride[i] != s) return false;
        s *= t->shape[i];
    }
    return true;
}

}  /* namespace r4d_lili */

using namespace r4d_lili;

extern "C" {

int r4d_rad_lili_cand(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 11) return RAD_E_INVAL;
    const RadTensor* t[11];
    for (int i = 0; i < 11; ++i) if (!(t[i] = rad_arg_in(a, i))) return RAD_E_INVAL;
    const long long k = rad_args_geti_or(a, "k", 0), world = rad_args_geti_or(a, "world", 1);
    const long long steps = rad_args_geti_or(a, "steps", 0), n = rad_args_geti_or(a, "n", 0);
    const long long rk = rad_args_geti_or(a, "row_k", k);
    if (k < 2 || k > kMaxK || rk < k || world < 1 || world > kMaxW || steps < 1 || n <= 0)
        return RAD_E_SHAPE;
    if (t[0]->dtype != RAD_F32 || t[1]->dtype != RAD_BF16 || t[8]->dtype != RAD_I32 ||
        t[9]->dtype != RAD_BF16 || t[10]->dtype != RAD_BF16 || t[6]->dtype != RAD_BF16 ||
        t[7]->dtype != RAD_BF16)
        return RAD_E_DTYPE;
    for (int i = 2; i <= 5; ++i) if (t[i]->dtype != RAD_F32) return RAD_E_DTYPE;
    for (int i = 2; i <= 7; ++i) if (!dense(t[i])) return RAD_E_STRIDE;
    if (rad_tensor_numel(t[2]) < 5 || rad_tensor_numel(t[3]) < 5 || rad_tensor_numel(t[4]) < n * 5 ||
        rad_tensor_numel(t[5]) < n || rad_tensor_numel(t[6]) < steps * n ||
        rad_tensor_numel(t[7]) < k * n)
        return RAD_E_SHAPE;
    long long M = 0, Mp = 0, Mc = 0, Mf = 0, Mb = 0;
    CandArgs A{};
    A.st_rs = pitch_of(t[0], world * (2 + 2 * rk), &M);
    A.pas_rs = pitch_of(t[1], n, &Mp);
    A.cand_rs = pitch_of(t[8], k, &Mc);
    A.feat_rs = pitch_of(t[9], n, &Mf);
    A.base_rs = pitch_of(t[10], n, &Mb);
    if (A.st_rs < 0 || A.pas_rs < 0 || A.cand_rs < 0 || A.feat_rs < 0 || A.base_rs < 0)
        return RAD_E_STRIDE;
    if (M <= 0 || Mp < M || Mc < M || Mf < M * k || Mb < M * k) return RAD_E_SHAPE;
    A.st = (const float*)t[0]->data;     A.pas = (const uint16_t*)t[1]->data;
    A.lnw = (const float*)t[2]->data;    A.lnb = (const float*)t[3]->data;
    A.upw = (const float*)t[4]->data;    A.upb = (const float*)t[5]->data;
    A.slot = (const uint16_t*)t[6]->data; A.rank = (const uint16_t*)t[7]->data;
    A.cand = (int*)t[8]->data;           A.feat = (uint16_t*)t[9]->data;
    A.base = (uint16_t*)t[10]->data;
    A.k = (int)k; A.row_k = (int)rk; A.world = (int)world; A.steps = (int)steps; A.n = (int)n;
    A.eps = (float)rad_args_getf_or(a, "eps", 1e-5);
    R4D_KLAUNCH(R4D_KNAME(cand_kernel), dim3((unsigned)M), dim3(kThr), (hipStream_t)stream, A);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_lili_attn(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4) return RAD_E_INVAL;
    const RadTensor* q = rad_arg_in(a, 0);
    const RadTensor* rel = rad_arg_in(a, 1);
    const RadTensor* same = rad_arg_in(a, 2);
    const RadTensor* o = rad_arg_in(a, 3);
    if (!q || !rel || !same || !o) return RAD_E_INVAL;
    const long long steps = rad_args_geti_or(a, "steps", 0), k = rad_args_geti_or(a, "k", 0);
    const long long nh = rad_args_geti_or(a, "n_head", 0), hd = rad_args_geti_or(a, "head_dim", 0);
    if (steps < 1 || k < 1 || nh < 1 || hd < 1 || hd > kMaxHd || steps * k > kMaxL)
        return RAD_E_SHAPE;
    if (q->dtype != RAD_BF16 || o->dtype != RAD_BF16 || rel->dtype != RAD_F32 ||
        same->dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (!dense(rel) || !dense(same)) return RAD_E_STRIDE;
    const long long rw = rel->rank ? rel->shape[rel->rank - 1] : 0;
    if (rw < 2 * steps - 1 || rad_tensor_numel(rel) < nh * rw || rad_tensor_numel(same) < nh)
        return RAD_E_SHAPE;
    const long long L = steps * k, n = nh * hd;
    long long M = 0, Mo = 0;
    AttnArgs A{};
    A.q_rs = pitch_of(q, 3 * n, &M);
    A.o_rs = pitch_of(o, n, &Mo);
    if (A.q_rs < 0 || A.o_rs < 0) return RAD_E_STRIDE;
    if (M <= 0 || M % L || Mo < M) return RAD_E_SHAPE;
    A.qkv = (const uint16_t*)q->data; A.out = (uint16_t*)o->data;
    A.rel = (const float*)rel->data;  A.same = (const float*)same->data; A.rel_w = (int)rw;
    A.steps = (int)steps; A.k = (int)k; A.nh = (int)nh; A.hd = (int)hd;
    A.scale = (float)rad_args_getf_or(a, "scale", 1.0 / __builtin_sqrt((double)hd));
    R4D_KLAUNCH(R4D_KNAME(attn_kernel), dim3((unsigned)(M / L), (unsigned)nh), dim3(kThr),
               (hipStream_t)stream, A);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_lili_factor(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const RadTensor* h = rad_arg_in(a, 0);
    const RadTensor* an = rad_arg_in(a, 1);
    const RadTensor* y = rad_arg_in(a, 2);
    if (!h || !an || !y) return RAD_E_INVAL;
    if (h->dtype != RAD_BF16 || an->dtype != RAD_BF16 || y->dtype != RAD_BF16) return RAD_E_DTYPE;
    const long long L = rad_args_geti_or(a, "L", 0), n = rad_args_geti_or(a, "n", 0);
    if (L < 1 || n < 1) return RAD_E_SHAPE;
    long long M = 0, Ma = 0, My = 0;
    FactorArgs A{};
    A.h_rs = pitch_of(h, n, &M);
    A.a_rs = pitch_of(an, n, &Ma);
    A.y_rs = pitch_of(y, 3 * n, &My);
    if (A.h_rs < 0 || A.a_rs < 0 || A.y_rs < 0) return RAD_E_STRIDE;
    if (M <= 0 || M % L || Ma < M / L || My < M) return RAD_E_SHAPE;
    A.h = (const uint16_t*)h->data; A.a = (const uint16_t*)an->data; A.y = (uint16_t*)y->data;
    A.L = (int)L; A.n = (int)n;
    R4D_KLAUNCH(R4D_KNAME(factor_kernel), dim3((unsigned)M), dim3(kThr), (hipStream_t)stream, A);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int r4d_rad_lili_walk(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4) return RAD_E_INVAL;
    const RadTensor* ed = rad_arg_in(a, 0);
    const RadTensor* ao = rad_arg_in(a, 1);
    const RadTensor* cd = rad_arg_in(a, 2);
    const RadTensor* tk = rad_arg_in(a, 3);
    const RadTensor* sc = a->n_t > 4 ? rad_arg_in(a, 4) : nullptr;
    if (!ed || !ao || !cd || !tk) return RAD_E_INVAL;
    const long long steps = rad_args_geti_or(a, "steps", 0), k = rad_args_geti_or(a, "k", 0);
    const long long n = rad_args_geti_or(a, "n", 0);
    if (steps < 1 || k < 1 || k > kMaxK || n < 1 || steps * k * k > kMaxWalk ||
        steps * k > kMaxK * 8)
        return RAD_E_SHAPE;
    if (ed->dtype != RAD_BF16 || ao->dtype != RAD_BF16 || cd->dtype != RAD_I32 ||
        tk->dtype != RAD_I32 || (sc && sc->dtype != RAD_F32))
        return RAD_E_DTYPE;
    if (sc && !dense(sc)) return RAD_E_STRIDE;
    long long M = 0, Me = 0, Mc = 0, Mt = 0;
    WalkArgs A{};
    A.a_rs = pitch_of(ao, n, &M);
    A.e_rs = pitch_of(ed, 2 * n, &Me);
    A.c_rs = pitch_of(cd, k, &Mc);
    A.t_rs = pitch_of(tk, steps, &Mt);
    if (A.a_rs < 0 || A.e_rs < 0 || A.c_rs < 0 || A.t_rs < 0) return RAD_E_STRIDE;
    if (M <= 0 || Me < M * steps * k || Mc < M * steps || Mt < M) return RAD_E_SHAPE;
    if (sc && rad_tensor_numel(sc) < M * steps * k * k) return RAD_E_SHAPE;
    A.ed = (const uint16_t*)ed->data; A.ao = (const uint16_t*)ao->data;
    A.cand = (const int*)cd->data;    A.tok = (int*)tk->data;
    A.scores = sc ? (float*)sc->data : nullptr;
    A.steps = (int)steps; A.k = (int)k; A.n = (int)n;
    A.eps = (float)rad_args_getf_or(a, "eps", 1e-4);
    A.scale = (float)rad_args_getf_or(a, "scale", 1.0);
    R4D_KLAUNCH(R4D_KNAME(walk_kernel), dim3((unsigned)M), dim3(kThr), (hipStream_t)stream, A);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* extern "C" */
