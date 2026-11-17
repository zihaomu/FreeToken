// Model-local BF16 seams. Attention recomputes QK in three bounded passes: maximum, f32
// denominator, then BF16 probabilities times V. Online rescaling would round unnormalised P
// and change this op's eager contract. No quadratic score buffer, allocation or host barrier.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_dt16.h"
#include "rad_yarn.h"
#include <climits>
#include <algorithm>
#include <new>

namespace {
constexpr int NT = 256;
struct RotaryFrequency {
  float f[64] = {};
};
__device__ float bf2f(uint16_t x) {
  return __builtin_bit_cast(float, (uint32_t)x << 16);
}
__device__ float narrow(float x) {
  return bf2f(f32_to_bf16(x));
}

// Metadata is device resident. Validate on the same queue before any data writes. A malformed
// index faults the pass (asynchronously), rather than returning a successful partial embedding.
__global__ void validate(const int* x, int64_t stride, int64_t plane, int M, int mode, int extra,
                         int limit, int* valid) {
  if (threadIdx.x || blockIdx.x) return;
  *valid = 0;
  bool ok = true;
  if (mode == 0 || mode == 1) {
    for (int r = 0; r < M; ++r) {
      ok = ok && x[(int64_t)r * stride] >= 0;
      if (mode == 1) ok = ok && x[plane + (int64_t)r * stride] >= 0;
    }
  } else if (mode == 2) {
    ok = x[0] == 0 && x[(int64_t)extra * stride] == M;
    for (int s = 0; s < extra; ++s) {
      const int64_t b = x[(int64_t)s * stride], e = x[(int64_t)(s + 1) * stride];
      ok = ok && b >= 0 && e > b && e <= M && e - b <= limit;
    }
  } else {
    for (int64_t p = 0; ok && p < M;) {
      const int64_t h = x[2 * plane + p * stride], w = x[3 * plane + p * stride];
      ok = h > 0 && w > 0 && h % extra == 0 && w % extra == 0 && h <= (M - p) / w;
      for (int64_t j = 0; ok && j < h * w; ++j)
        ok = x[(p + j) * stride] == j / w && x[plane + (p + j) * stride] == j % w &&
             x[2 * plane + (p + j) * stride] == h && x[3 * plane + (p + j) * stride] == w;
      if (ok) p += h * w;
    }
  }
  if (!ok) __builtin_trap();
  *valid = 1;
}

__global__ void pooled_mean(const uint16_t* x, int64_t xl, int64_t xs, const int* cu, int64_t cs,
                            uint16_t* y, int64_t yl, int64_t ys, int S, int N, const int* valid) {
  if (*valid != 1)
    return;
  const int64_t c = (int64_t)blockIdx.x * NT + threadIdx.x;
  if (c >= N)
    return;
  for (int64_t s = blockIdx.y; s < S; s += gridDim.y) {
    const int begin = cu[s * cs], end = cu[(s + 1) * cs];
    float sum = 0;
    for (int r = begin; r < end; ++r)
      sum += bf2f(x[(int64_t)r * xl + c * xs]);
    y[s * yl + c * ys] = f32_to_bf16(narrow(sum) / (float)(end - begin));
  }
}

__global__ void scale(uint16_t* x, int64_t ld, int64_t cs, const int* pos, int64_t ps, int M, int n,
                      int original, float beta, const int* valid) {
  if (*valid != 1) return;
  for (int64_t r = blockIdx.y; r < M; r += gridDim.y) {
    const int64_t c = (int64_t)blockIdx.x * NT + threadIdx.x;
    if (c < n) {
      const float g = narrow(1.f + beta * logf(1.f + (float)(pos[r * ps] / original)));
      x[r * ld + c * cs] = f32_to_bf16(bf2f(x[r * ld + c * cs]) * g);
    }
  }
}

__global__ void rope(uint16_t* x, int64_t ld, int64_t cs, const int* pos, int64_t ps, int64_t plane,
                     int M, int D, int H, int K, RotaryFrequency frequency, float magnitude,
                     bool pixtral, const int* valid) {
  if (*valid != 1) return;
  const int i = threadIdx.x % (D / 2), h = threadIdx.x / (D / 2) + blockIdx.x * (NT / (D / 2));
  if (h >= H + K) return;
  for (int64_t r = blockIdx.y; r < M; r += gridDim.y) {
    const int axis = pixtral ? i / (D / 4) : 0;
    const float angle = (float)pos[axis * plane + r * ps] * frequency.f[i];
    const float c = narrow(cosf(angle) * magnitude), s = narrow(sinf(angle) * magnitude);
    const int64_t u = r * ld + (h * D + i) * cs, v = u + (D / 2) * cs;
    const float a = bf2f(x[u]), b = bf2f(x[v]);
    x[u] = f32_to_bf16(narrow(a * c) - narrow(b * s));
    x[v] = f32_to_bf16(narrow(b * c) + narrow(a * s));
  }
}

__global__ void merge(const uint16_t* x, int64_t ld, int64_t cs, const int* rc, int64_t rs,
                      int64_t ps, uint16_t* y, int64_t yl, int64_t ys, int M, int N, int K,
                      const int* valid) {
  if (*valid != 1) return;
  for (int64_t p = blockIdx.y; p < M; p += gridDim.y) {
    const int row = rc[p * ps], col = rc[rs + p * ps], w = rc[3 * rs + p * ps];
    if (row % K || col % K) continue;
    const int64_t base = p - (int64_t)row * w - col;
    const int64_t dst = base / (K * K) + (row / K) * (w / K) + col / K;
    for (int64_t n = (int64_t)blockIdx.x * NT + threadIdx.x; n < N; n += (int64_t)gridDim.x * NT)
      for (int dy = 0; dy < K; ++dy)
        for (int dx = 0; dx < K; ++dx)
          y[dst * yl + ((int64_t)n * K * K + dy * K + dx) * ys] =
            x[(p + (int64_t)dy * w + dx) * ld + (int64_t)n * cs];
  }
}

template <int D>
__global__ __launch_bounds__(32) void attention(const uint16_t* q, const uint16_t* k,
                                                const uint16_t* v, uint16_t* out, const int* cu,
                                                int64_t cus, int M, int H, int G, int64_t ql,
                                                int64_t kl, int64_t vl, int64_t ol, float scale,
                                                const int* valid) {
  if (*valid != 1) return;
  using BF = DT16<0>;
  const int lane = threadIdx.x, c = lane & 15, h = lane >> 4;
  const int begin = cu[(int64_t)blockIdx.z * cus], end = cu[(int64_t)(blockIdx.z + 1) * cus];
  const int qi = blockIdx.x * 16 + c, head = blockIdx.y, kh = head / G;
  if ((int)blockIdx.x * 16 >= end - begin) return;
  const int qr = begin + min(qi, end - begin - 1);
  typename BF::frag qf[D / 16];
  for (int t = 0; t < D / 16; ++t) {
    typename BF::frag f;
    for (int e = 0; e < 8; ++e)
      f[e] = __builtin_bit_cast(__bf16, q[(int64_t)qr * ql + head * D + t * 16 + h * 8 + e]);
    qf[t] = f;
  }
  __shared__ uint16_t sv[D][24];
  __shared__ float sp[16][16];
  v8f acc[D / 16];
  for (int t = 0; t < D / 16; ++t)
    acc[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
  float mx = -INFINITY, denom = 0;
  for (int pass = 0; pass < 3; ++pass) {
    for (int kb = begin; kb < end; kb += 16) {
      v8f s = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
      if constexpr (D == 64) {
        // Pixtral's small heads: preserve the reference dot order, including very small
        // components beside the learned large coordinates. BF16 midpoint crossings
        // otherwise compound over the vision tower before the text encoder sees it.
        for (int e = 0; e < 8; ++e) {
          const int kr = min(kb + h * 8 + e, end - 1);
          float dot = 0;
          for (int d = 0; d < D; ++d)
            dot +=
              bf2f(q[(int64_t)qr * ql + head * D + d]) * bf2f(k[(int64_t)kr * kl + kh * D + d]);
          s[e] = dot;
        }
      } else {
        for (int t = 0; t < D / 16; ++t) {
          typename BF::frag f;
          const int kr = min(kb + c, end - 1);
          for (int e = 0; e < 8; ++e)
            f[e] = __builtin_bit_cast(__bf16, k[(int64_t)kr * kl + kh * D + t * 16 + h * 8 + e]);
          s = BF::wmma(f, qf[t], s);
        }
      }
      float p[8];
      for (int e = 0; e < 8; ++e) {
        const float score = narrow(narrow(s[e]) * scale);
        p[e] = kb + h * 8 + e < end ? score : -INFINITY;
        if (pass == 0)
          mx = fmaxf(mx, p[e]);
        else
          p[e] = expf(p[e] - mx);
        if (pass == 1 && D != 64) denom += p[e];
      }
      if constexpr (D == 64) {
        if (pass == 1) {
          for (int e = 0; e < 8; ++e)
            denom += __shfl(p[e], c);
          for (int e = 0; e < 8; ++e)
            denom += __shfl(p[e], c + 16);
        }
      }
      if (pass != 2) continue;
      if constexpr (D == 64) {
        for (int e = 0; e < 8; ++e)
          sp[c][h * 8 + e] = narrow(p[e] / denom);
        __syncthreads();
        for (int j = 0; j < 16 && kb + j < end; ++j)
          for (int t = 0; t < D / 16; ++t)
            for (int e = 0; e < 8; ++e)
              acc[t][e] += sp[c][j] * bf2f(v[(int64_t)(kb + j) * vl + kh * D + t * 16 + h * 8 + e]);
        __syncthreads();
        continue;
      }
      for (int z = lane; z < D * 16; z += 32) {
        const int d = z / 16, key = z % 16;
        sv[d][key] = kb + key < end ? v[(int64_t)(kb + key) * vl + kh * D + d] : 0;
      }
      __syncthreads();
      typename BF::frag pf;
      for (int e = 0; e < 8; ++e)
        pf[e] = __builtin_bit_cast(__bf16, f32_to_bf16(p[e] / denom));
      for (int t = 0; t < D / 16; ++t) {
        typename BF::frag vf;
        for (int e = 0; e < 8; ++e)
          vf[e] = __builtin_bit_cast(__bf16, sv[t * 16 + c][h * 8 + e]);
        acc[t] = BF::wmma(vf, pf, acc[t]);
      }
      __syncthreads();
    }
    if (pass == 0) mx = fmaxf(mx, __shfl_xor(mx, 16));
    if (pass == 1 && D != 64) denom += __shfl_xor(denom, 16);
  }
  if (qi >= end - begin) return;
  for (int t = 0; t < D / 16; ++t)
    for (int e = 0; e < 8; ++e)
      out[(int64_t)(begin + qi) * ol + head * D + t * 16 + h * 8 + e] = f32_to_bf16(acc[t][e]);
}

// Pixtral's RGB 14x14 patch projection has K=588, outside the existing K%16 GEMMs.
// Keep the oracle's ordered f32 dot for this small stem: a WMMA reduction can cross a BF16
// rounding midpoint. No existing GEMM row accepts this geometry, so no old row is replaced.
__global__ __launch_bounds__(NT) void patch_projection(const uint16_t* a, const uint16_t* w,
                                                       uint16_t* y, int M, int N, int64_t al,
                                                       int64_t wl, int64_t yl) {
  const int n = blockIdx.x * NT + threadIdx.x;
  if (n >= N) return;
  for (int64_t r = blockIdx.y; r < M; r += gridDim.y) {
    float sum = 0;
    for (int d = 0; d < 588; ++d)
      sum += bf2f(a[r * al + d]) * bf2f(w[(int64_t)n * wl + d]);
    y[r * yl + n] = f32_to_bf16(sum);
  }
}

bool matrix(const RadTensor& t, int64_t m, int64_t n) {
  constexpr int64_t max_elements = INT64_MAX / 2;
  if (!t.data || t.dtype != RAD_BF16 || t.rank != 2 || t.shape[0] != m || t.shape[1] != n ||
      m <= 0 || n <= 0 || t.stride[1] <= 0 || t.stride[0] <= 0 || n > max_elements / t.stride[1])
    return false;
  const int64_t columns = (n - 1) * t.stride[1];
  return t.stride[0] >= columns + 1 && (m - 1) <= (max_elements - columns) / t.stride[0];
}
bool indices(const RadTensor& t, int64_t n, int planes = 1) {
  constexpr int64_t max_elements = INT64_MAX / 4;
  if (!t.data || t.dtype != RAD_I32 || t.rank != (planes == 1 ? 1u : 2u) || n <= 0 ||
      t.stride[t.rank - 1] <= 0 || n > max_elements / t.stride[t.rank - 1])
    return false;
  if (planes == 1) return t.shape[0] == n;
  const int64_t columns = (n - 1) * t.stride[1];
  return t.shape[0] == planes && t.shape[1] == n && t.stride[0] >= columns + 1 &&
         t.stride[0] <= (max_elements - columns) / (planes - 1);
}
int prepare(const RadArgs* a, RadStream stream, const RadTensor& idx, int M, int mode,
            int extra = 0, int limit = 32768) {
  if (!a->scratch || a->scratch_bytes < 4) return RAD_E_SCRATCH;
  hipLaunchKernelGGL(validate, dim3(1), dim3(1), 0, (hipStream_t)stream, (const int*)idx.data,
                     idx.stride[idx.rank - 1], idx.rank == 2 ? idx.stride[0] : 0, M, mode, extra,
                     limit, (int*)a->scratch);
  return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
int finish() {
  return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
}

// Frequencies are shape constants. Resolve them once with the host oracle's powf, rather than
// let a device pow approximation perturb long-position phases before BF16 narrowing.
static int rotary_init(const RadParam* p, int np, void** out, bool pix) {
  if (!out) return RAD_E_INVAL;
  RadArgs a{};
  a.p = p;
  a.n_p = np;
  const int64_t D = r4d_geti_or(&a, "head_dim", 0);
  const float theta = r4d_getf(&a, "theta", 0);
  const RadYarn yarn = rad_yarn_args(&a);
  if ((D != 64 && D != 128) || !std::isfinite(theta) || theta <= 0 || !rad_yarn_valid(yarn))
    return RAD_E_SHAPE;
  auto* f = new (std::nothrow) RotaryFrequency;
  if (!f) return RAD_E_NOMEM;
  for (int i = 0; i < D / 2; ++i) {
    const int axis = pix ? i / (D / 4) : 0, j = pix ? i % (D / 4) : i;
    const float inv = 1.f / std::pow(theta, (float)(pix ? 4 * j + 2 * axis : 2 * i) / (float)D);
    f->f[i] = pix ? inv : yarn.frequency(inv, i);
  }
  *out = f;
  return RAD_OK;
}
extern "C" int r4d_ministral_init(const RadParam* p, int n, int, int, void** out) {
  return rotary_init(p, n, out, false);
}
extern "C" int r4d_pixtral_init(const RadParam* p, int n, int, int, void** out) {
  return rotary_init(p, n, out, true);
}
extern "C" void r4d_rotary_fini(void* p) {
  delete (RotaryFrequency*)p;
}
extern "C" int64_t r4d_nemotron_scratch(const RadArgs*) {
  return 4;
}
extern "C" int r4d_rad_seg_mean_bf16_sum(const RadArgs* a, RadStream stream) {
  if (!rad_args_have(a, 3))
    return RAD_E_INVAL;
  const auto &x = a->t[0], &cu = a->t[1], &y = a->t[2];
  const int64_t M = x.rank == 2 ? x.shape[0] : 0, N = r4d_geti_or(a, "n", 0),
                S = cu.rank == 1 && cu.shape[0] > 1 ? cu.shape[0] - 1 : 0;
  if (M <= 0 || M > INT_MAX || M > r4d_geti_or(a, "M", 0) || N <= 0 || N > INT_MAX || S <= 0 ||
      S > INT_MAX || !matrix(x, M, N) || !matrix(y, S, N) || !indices(cu, S + 1))
    return RAD_E_SHAPE;
  int rc = prepare(a, stream, cu, (int)M, 2, (int)S, INT_MAX);
  if (rc < 0)
    return rc;
  hipLaunchKernelGGL(pooled_mean, dim3((N + NT - 1) / NT, std::min<int64_t>(S, 65535)), dim3(NT), 0,
                     (hipStream_t)stream, (const uint16_t*)x.data, x.stride[0], x.stride[1],
                     (const int*)cu.data, cu.stride[0], (uint16_t*)y.data, y.stride[0], y.stride[1],
                     (int)S, (int)N, (const int*)a->scratch);
  return finish();
}
extern "C" int r4d_rad_position_scale(const RadArgs* a, RadStream stream) {
  if (!rad_args_have(a, 2)) return RAD_E_INVAL;
  const auto &x = a->t[0], &p = a->t[1];
  const int64_t M = x.rank == 2 ? x.shape[0] : 0, n = r4d_geti_or(a, "n", 0),
                orig = r4d_geti_or(a, "original", 0);
  const float beta = r4d_getf(a, "beta", -1);
  if (M <= 0 || M > INT_MAX || M > r4d_geti_or(a, "M", 0) || n <= 0 || n > INT_MAX || orig <= 0 ||
      orig > INT_MAX || !std::isfinite(beta) || beta < 0 || !matrix(x, M, n) || !indices(p, M))
    return RAD_E_SHAPE;
  int rc = prepare(a, stream, p, (int)M, 0);
  if (rc < 0) return rc;
  hipLaunchKernelGGL(scale, dim3((n + NT - 1) / NT, std::min<int64_t>(M, 65535)), dim3(NT), 0,
                     (hipStream_t)stream, (uint16_t*)x.data, x.stride[0], x.stride[1],
                     (const int*)p.data, p.stride[0], (int)M, (int)n, (int)orig, beta,
                     (const int*)a->scratch);
  return finish();
}
static int rotary(const RadArgs* a, RadStream stream, bool pix) {
  if (!rad_args_have(a, 2)) return RAD_E_INVAL;
  const auto &x = a->t[0], &p = a->t[1];
  const int64_t M = x.rank == 2 ? x.shape[0] : 0, D = r4d_geti_or(a, "head_dim", 0),
                H = r4d_geti_or(a, "n_head", 0), K = pix ? H : r4d_geti_or(a, "n_head_kv", 0);
  const float theta = r4d_getf(a, "theta", 0);
  const RadYarn yarn = rad_yarn_args(a);
  if (M <= 0 || M > INT_MAX || M > r4d_geti_or(a, "M", 0) || (D != 64 && D != 128) || H <= 0 ||
      H > 1024 || K <= 0 || K > H || H % K || !std::isfinite(theta) || theta <= 0 ||
      !rad_yarn_valid(yarn) || !matrix(x, M, (H + 2 * K) * D) || !indices(p, M, pix ? 2 : 1))
    return RAD_E_SHAPE;
  if (!a->instance) return RAD_E_STATE;
  int rc = prepare(a, stream, p, (int)M, pix ? 1 : 0);
  if (rc < 0) return rc;
  hipLaunchKernelGGL(rope, dim3(((H + K) * (D / 2) + NT - 1) / NT, std::min<int64_t>(M, 65535)),
                     dim3(NT), 0, (hipStream_t)stream, (uint16_t*)x.data, x.stride[0], x.stride[1],
                     (const int*)p.data, p.stride[p.rank - 1], pix ? p.stride[0] : 0, (int)M,
                     (int)D, (int)H, (int)K, *(const RotaryFrequency*)a->instance, yarn.magnitude,
                     pix, (const int*)a->scratch);
  return finish();
}
extern "C" int r4d_rad_ministral_rope(const RadArgs* a, RadStream s) {
  return rotary(a, s, false);
}
extern "C" int r4d_rad_pixtral_rope(const RadArgs* a, RadStream s) {
  return rotary(a, s, true);
}
extern "C" int r4d_rad_pixtral_merge(const RadArgs* a, RadStream stream) {
  if (!rad_args_have(a, 3)) return RAD_E_INVAL;
  const auto &x = a->t[0], &p = a->t[1], &y = a->t[2];
  const int64_t M = x.rank == 2 ? x.shape[0] : 0, N = r4d_geti_or(a, "n", 0),
                K = r4d_geti_or(a, "merge", 0);
  if (M <= 0 || M > INT_MAX || M > r4d_geti_or(a, "M", 0) || N <= 0 || N > INT_MAX || K != 2 ||
      M % 4 || !matrix(x, M, N) || !indices(p, M, 4) || !matrix(y, M / 4, N * 4))
    return RAD_E_SHAPE;
  int rc = prepare(a, stream, p, (int)M, 3, 2);
  if (rc < 0) return rc;
  hipLaunchKernelGGL(merge, dim3((N + NT - 1) / NT, std::min<int64_t>(M, 65535)), dim3(NT), 0,
                     (hipStream_t)stream, (const uint16_t*)x.data, x.stride[0], x.stride[1],
                     (const int*)p.data, p.stride[0], p.stride[1], (uint16_t*)y.data, y.stride[0],
                     y.stride[1], (int)M, (int)N, (int)K, (const int*)a->scratch);
  return finish();
}
extern "C" int r4d_rad_embed_attn(const RadArgs* a, RadStream stream) {
  if (!rad_args_have(a, 5)) return RAD_E_INVAL;
  const auto &q = a->t[0], &k = a->t[1], &v = a->t[2], &cu = a->t[3], &o = a->t[4];
  const int64_t M = q.rank == 2 ? q.shape[0] : 0, D = r4d_geti_or(a, "head_dim", 0),
                G = r4d_geti_or(a, "gqa", 0), H = D > 0 ? q.shape[1] / D : 0,
                S = cu.rank == 1 ? cu.shape[0] - 1 : 0;
  if (M <= 0 || M > INT_MAX || M > r4d_geti_or(a, "M", 0) || (D != 64 && D != 128) || G <= 0 ||
      H <= 0 || H > 65535 || H % G || S <= 0 || S > 65535 || r4d_geti_or(a, "causal", 0) != 0 ||
      !matrix(q, M, H * D) || !matrix(k, M, H / G * D) || !matrix(v, M, H / G * D) ||
      !matrix(o, M, H * D) || !indices(cu, S + 1) || q.stride[1] != 1 || k.stride[1] != 1 ||
      v.stride[1] != 1 || o.stride[1] != 1)
    return RAD_E_SHAPE;
  const float sc = r4d_getf(a, "scale", 1.0 / std::sqrt((float)D));
  if (!std::isfinite(sc) || sc <= 0) return RAD_E_INVAL;
  const int64_t limit = std::min<int64_t>({M, 32768, r4d_geti_or(a, "max_seqlen", 32768)});
  if (limit <= 0) return RAD_E_SHAPE;
  int rc = prepare(a, stream, cu, (int)M, 2, (int)S, (int)limit);
  if (rc < 0) return rc;
  // Each segment is at most 32768; combined M may exceed that. The grid uses a bounded
  // per-segment extent without carving a request into independent attention states.
  dim3 grid((limit + 15) / 16, H, S);
#define LAUNCH(DIM)                                                                                \
  hipLaunchKernelGGL((attention<DIM>), grid, dim3(32), 0, (hipStream_t)stream,                     \
                     (const uint16_t*)q.data, (const uint16_t*)k.data, (const uint16_t*)v.data,    \
                     (uint16_t*)o.data, (const int*)cu.data, cu.stride[0], (int)M, (int)H, (int)G, \
                     q.stride[0], k.stride[0], v.stride[0], o.stride[0], sc,                       \
                     (const int*)a->scratch)
  if (D == 64) {
    LAUNCH(64);
  } else {
    LAUNCH(128);
  }
#undef LAUNCH
  return finish();
}

extern "C" int r4d_rad_pixtral_patch(const RadArgs* a, RadStream stream) {
  if (!rad_args_have(a, 3)) return RAD_E_INVAL;
  const auto &x = a->t[0], &w = a->t[1], &y = a->t[2];
  const int64_t M = x.rank == 2 ? x.shape[0] : 0, N = r4d_geti_or(a, "N", 0);
  if (M <= 0 || M > INT_MAX || M > r4d_geti_or(a, "M", 0) || N <= 0 || N > INT_MAX ||
      r4d_geti_or(a, "K", 0) != 588 || !matrix(x, M, 588) || !matrix(w, N, 588) ||
      !matrix(y, M, N) || x.stride[1] != 1 || w.stride[1] != 1 || y.stride[1] != 1)
    return RAD_E_SHAPE;
  hipLaunchKernelGGL(patch_projection, dim3((N + NT - 1) / NT, std::min<int64_t>(M, 65535)),
                     dim3(NT), 0, (hipStream_t)stream, (const uint16_t*)x.data,
                     (const uint16_t*)w.data, (uint16_t*)y.data, (int)M, (int)N, x.stride[0],
                     w.stride[0], y.stride[0]);
  return finish();
}
