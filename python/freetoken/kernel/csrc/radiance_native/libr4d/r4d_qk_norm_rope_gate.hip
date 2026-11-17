// Modified for FreeToken: native integration, diagnostics and Qwen FP8 KV support.
// r4d_qk_norm_rope_gate.hip -- r4d_qk_norm_rope_gate: the whole gated-attention QKV preamble in one
// pass: split [q|gate] per head, QK RMS norm over the head, partial NeoX RoPE, and the gate copy.
//
// vLLM's equivalent fusion, `fused_qk_rmsnorm_rope_gate`, is gated on `is_cuda()`, and its Triton
// kernel does not compile on gfx1201 (Triton's AMD backend fails in
// `TritonAMDGPUCanonicalizePointers`). The unfused ROCm path is four kernels per attention layer
// (split, two GemmaRMSNorms, RoPE and gate chunk): 64 dispatches a decode step over 16 layers for
// ~6.3 us of work.
//
// Bit-identity: the reference stores the normed value as bf16 and RoPEs what it reads back, so the
// norm output is rounded to bf16 before the rotation. Skipping the round trip would be more
// accurate and a different function; vLLM's kernel has `.to(INPUT_DTYPE).to(tl.float32)` for the
// same reason.
//
// `wadd` is added to every gain: Qwen3NextRMSNorm is GemmaRMSNorm, `x_hat * (1 + w)` over a
// zero-centred gamma in fp32. This takes the stored gain and the offset and adds them in fp32 after
// the load, which matches vLLM's `weight.float() + 1.0` without a per-layer fp32 copy.
//
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "rad_yarn.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>

// One block per (token, head) and one element per lane, required for bit-equality.
//
// Eight consecutive elements per lane with one wave per head (no LDS, rotary partner one shuffle
// away) computes a different function. r4d_rmsnorm_bf16's partition decides which columns a lane
// folds into the sum of squares, and a last-bit difference in the norm becomes a few percent where
// a rotated element's two products nearly cancel, enough to change the engine's text within a few
// dozen steps.
//
// The reference's fold is easy to reproduce: `rmsnorm_bf16` folds a row over kR4dSumLanes' 256
// lanes, lane v adding columns v, v+256, ... (r4d_common.h), and every head_dim here is at most
// 256, so it always folds exactly one element per lane. Lanes past the head add exact zeros, so
// `block_sum` over 8 or 32 wave partials gives the same number.
// Hence one element a lane, 256 threads, `block_sum` in ascending wave order.
//
// Cost: at rot 64 the rotary partner is in another wave. The lane re-derives it instead of an LDS
// exchange: it loads the partner's input (an L1-hot read) and norms it with the block's `inv`.
/* The paged cache the KV form writes: kv_store's operands and parameters in one argument, to stay
 * inside the emitter's parameter budget. */
struct QkKvStore {
  const __hip_bfloat16* v;
  const int32_t*        slot;
  void*                 cache;
  int   v_pitch, block_size, n_blocks;
  float k_inv = 1.0f, v_inv = 1.0f;
  /* The streamed forms only: the group's host mirror and per-layer residency map, as
   * kv_store_streamed takes them (libr4d/r4d_model_bf16.hip). `blocks` is the mirror's block
   * count (the id space) while `n_blocks` stays the ring depth. */
  const int32_t*        res;
  void*                 host;
  int                   blocks;
};

/* The cache forms KV selects: bf16 or the int8 cache, each resident or streamed (the same store
 * into the host mirror and, where the block is resident, the ring). */
enum { QK_KV_NONE = 0, QK_KV_BF16 = 2, QK_KV_STREAMED_BF16 = 6, QK_KV_I8 = 7,
       QK_KV_STREAMED_I8 = 8, QK_KV_FP8 = 9, QK_KV_STREAMED_FP8 = 10 };

/* KV folds kv_store in: a k head's block, holding the token's rotated k one element per lane, also
 * writes that element and the head's v element into the cache at the token's slot, as bf16 like
 * kv_store_bf16 or as the int8 cache's codes like kv_store_i8 (see the store at the end). */
#define R4D_QK_NORM_ROPE_GATE_PARAMS                                                            \
    const __hip_bfloat16* __restrict__ QG, const __hip_bfloat16* __restrict__ K,                \
    __hip_bfloat16* __restrict__ QO, __hip_bfloat16* __restrict__ KO,                           \
    __hip_bfloat16* __restrict__ GO, const void* __restrict__ WQ,                               \
    const void* __restrict__ WK, const void* __restrict__ CS, const void* __restrict__ POS,     \
    long cs_rows, int nq, int nkv, int rot, float eps, float wadd, int qg_pitch,                \
    int k_pitch, int cs_f32, int pos_i64, int w_f32, QkKvStore kvs
template <int HD, int KV = QK_KV_NONE>
__device__ __forceinline__ void r4d_qk_norm_rope_gate_body(R4D_QK_NORM_ROPE_GATE_PARAMS) {
  constexpr int THR = 256;
  __shared__ float lds[THR / 32];
  const int tok = blockIdx.x, h = blockIdx.y;
  const int t = threadIdx.x;
  const long p = pos_i64 ? ((const long*)POS)[tok] : (long)((const int*)POS)[tok];
  const int half = rot >> 1;
  const bool isk = h >= nq;
  const int lh = isk ? h - nq : h;
  const __hip_bfloat16* in = isk ? (K + (size_t)tok * k_pitch + (size_t)lh * HD)
                                 : (QG + (size_t)tok * qg_pitch + (size_t)lh * 2 * HD);
  // The gain is f32 in this engine's plugins and bf16 in vLLM's; its width follows the operand.
  const void* w = isk ? WK : WQ;
  __hip_bfloat16* out = isk ? (KO + ((size_t)tok * nkv + lh) * HD)
                            : (QO + ((size_t)tok * nq + lh) * HD);
  const float* csf = (const float*)CS;
  const __hip_bfloat16* csb = (const __hip_bfloat16*)CS;

  // Spelled once so this lane's element and its rotary partner cannot use different widths.
  auto gain = [&](int i) -> float {
    return w_f32 ? ((const float*)w)[i] : __bfloat162float(((const __hip_bfloat16*)w)[i]);
  };
  // Rounded to bf16 before the rotation; see the bit-identity note at the top of this file.
  auto normed = [&](float v, float g, float inv) -> float {
    return __bfloat162float(__float2bfloat16(v * inv * (g + wadd)));
  };

  // Everything the rotation reads is loaded before the reduction, so it is in flight during the
  // sum of squares: this lane's element and gain, its partner's, and its cos and sin. Indices are
  // clamped instead of guarding the loads (a lane past the head reads its last element; a lane past
  // the rotary range reads its own element and table column 0, which exists since rot >= 2), and
  // the values are used only where the guards below allow.
  const int tc = t < HD ? t : HD - 1;
  const bool rl = t < rot, first = t < half;
  const int pt = rl ? (first ? t + half : t - half) : tc;
  const int ci = rl ? (first ? t : t - half) : 0;
  // The table row is clamped: a padded row (negative) or a position past the table's height has no
  // row rope_table wrote, and must not read outside the plane.
  const size_t co = (size_t)(p < 0 ? 0 : (p < cs_rows ? p : cs_rows - 1)) * rot;
  const float vt = __bfloat162float(in[tc]), gt = gain(tc);
  const float vp = __bfloat162float(in[pt]), gp = gain(pt);
  const float c = cs_f32 ? csf[co + ci] : __bfloat162float(csb[co + ci]);
  const float s = cs_f32 ? csf[co + half + ci] : __bfloat162float(csb[co + half + ci]);

  float ss = 0.0f;
  if (t < HD) ss = vt * vt;
  // rmsnorm_bf16's block_sum, transcribed: the wave butterfly, then the wave partials summed in
  // ascending order. Any other order breaks the bit-equality above. The barrier orders only the LDS
  // partials, so the loads above stay in flight across it.
  ss = r4d_wave_sum(ss);
  if ((t & 31) == 0) lds[t >> 5] = ss;
  lds_barrier();
  float sum = 0.0f;
#pragma unroll
  for (int wv = 0; wv < THR / 32; ++wv) sum += lds[wv];
  const float inv = rsqrtf(sum / (float)HD + eps);

  float res = 0.0f;
  if (t < HD) {
    res = normed(vt, gt, inv);
    if (rl) {
      const float xp = normed(vp, gp, inv);
      res = first ? (res * c - xp * s) : (xp * s + res * c);
    }
  }
  // Every lane must read its input before any lane writes back: the engine rotates `k` in place,
  // and a lane's partner is a memory read, so without this barrier it may already be overwritten.
  // The failure is a ~1% scatter of wrong elements that looks like rounding; r4d_selftest's
  // byte-equality row catches it.
  __syncthreads();
  if (t < HD) {
    out[t] = __float2bfloat16(res);
    // `gate_out` is optional: a caller that reads the gate straight from the interleaved projection
    // passes none.
    if (!isk && GO) GO[((size_t)tok * nq + lh) * HD + t] = in[HD + t];
  }
  // The cache write kv_store would make next, from the same bytes: the rotated k as just stored
  // and the v row as the projection left it. Layout [n_blocks, kv_heads, block_size, row], k then
  // v; a negative or out-of-range slot stores nothing.
  //
  // A streamed group's cache is the ring and its truth is the host mirror: the token goes to the
  // mirror's block row always and to the ring slot only where the residency map says the block is
  // resident, exactly as kv_store_streamed writes it (libref/ref_attn.cpp states why the test
  // belongs in the writer). The block id bounds the mirror, the map bounds the ring, and a slot
  // names a ring block only through `res`.
  if constexpr (KV == QK_KV_I8 || KV == QK_KV_STREAMED_I8) {
    // The int8 cache (kv_store_i8's bytes): the head's k and v each as int8 codes against one f16
    // scale, the head's absmax / 127, taken over the block through LDS. `isk` is uniform over the
    // block, so every thread of a k head's block meets the barrier.
    if (isk) {
      __shared__ float s_m[2][THR / 32];
      const bool in = t < HD;
      const float kf = in ? __bfloat162float(__float2bfloat16(res)) : 0.0f;
      const float vf =
          in ? __bfloat162float(kvs.v[(size_t)tok * kvs.v_pitch + (size_t)lh * HD + t]) : 0.0f;
      float km = r4d_wave_max(fabsf(kf)), vm = r4d_wave_max(fabsf(vf));
      if ((t & 31) == 0) { s_m[0][t >> 5] = km; s_m[1][t >> 5] = vm; }
      __syncthreads();
      km = 0.0f; vm = 0.0f;
#pragma unroll
      for (int wv = 0; wv < THR / 32; ++wv) {
        km = fmaxf(km, s_m[0][wv]);
        vm = fmaxf(vm, s_m[1][wv]);
      }
      const int32_t sl = kvs.slot[tok];
      constexpr bool STREAMED = KV == QK_KV_STREAMED_I8;
      const uint32_t id_space = (uint32_t)(STREAMED ? kvs.blocks : kvs.n_blocks);
      if (in && sl >= 0 && (uint32_t)sl / (uint32_t)kvs.block_size < id_space) {
        const uint32_t blk = (uint32_t)sl / (uint32_t)kvs.block_size;
        const uint32_t off = (uint32_t)sl % (uint32_t)kvs.block_size;
        const size_t rowb = (size_t)(2 * HD + 4);
        const __fp16 k16 = (__fp16)fminf(km / 127.0f, 515.0f);
        const __fp16 v16 = (__fp16)fminf(vm / 127.0f, 515.0f);
        const float ksc = (float)k16, vsc = (float)v16;
        const float kr = ksc > 0.0f ? rintf(kf / ksc) : 0.0f;
        const float vr = vsc > 0.0f ? rintf(vf / vsc) : 0.0f;
        const unsigned char kc = (unsigned char)(signed char)(int)fminf(fmaxf(kr, -127.0f), 127.0f);
        const unsigned char vc = (unsigned char)(signed char)(int)fminf(fmaxf(vr, -127.0f), 127.0f);
        auto put = [&](unsigned char* c) {
          c[t]      = kc;
          c[HD + t] = vc;
          if (t == 0) {
            *(uint16_t*)(c + 2 * HD)     = __builtin_bit_cast(uint16_t, k16);
            *(uint16_t*)(c + 2 * HD + 2) = __builtin_bit_cast(uint16_t, v16);
          }
        };
        if constexpr (STREAMED) {
          const size_t o = ((size_t)lh * kvs.block_size + off) * rowb;
          const size_t brow = (size_t)nkv * kvs.block_size * rowb;
          put((unsigned char*)kvs.host + (size_t)blk * brow + o);
          const int32_t rs = kvs.res[blk];
          if (rs >= 0 && (uint32_t)rs < (uint32_t)kvs.n_blocks)
            put((unsigned char*)kvs.cache + (size_t)rs * brow + o);
        } else {
          put((unsigned char*)kvs.cache + (((size_t)blk * nkv + lh) * kvs.block_size + off) * rowb);
        }
      }
    }
  } else if constexpr (KV == QK_KV_FP8 || KV == QK_KV_STREAMED_FP8) {
    if (isk && t < HD) {
      const int32_t sl = kvs.slot[tok];
      constexpr bool STREAMED = KV == QK_KV_STREAMED_FP8;
      const uint32_t blocks = (uint32_t)(STREAMED ? kvs.blocks : kvs.n_blocks);
      if (sl >= 0 && (uint32_t)sl / (uint32_t)kvs.block_size < blocks) {
        const uint32_t blk = (uint32_t)sl / (uint32_t)kvs.block_size;
        const uint32_t off = (uint32_t)sl % (uint32_t)kvs.block_size;
        const float kf = __bfloat162float(__float2bfloat16(res)) * kvs.k_inv;
        const float vf = __bfloat162float(kvs.v[(size_t)tok * kvs.v_pitch +
                                                   (size_t)lh * HD + t]) * kvs.v_inv;
        const uint32_t packed = __builtin_amdgcn_cvt_pk_fp8_f32(
            fminf(fmaxf(kf, -448.0f), 448.0f), fminf(fmaxf(vf, -448.0f), 448.0f), 0, false);
        auto put = [&](uint8_t* row) {
          row[t] = (uint8_t)packed;
          row[HD + t] = (uint8_t)(packed >> 8);
        };
        const size_t pitch = (size_t)nkv * kvs.block_size * 2 * HD;
        const size_t at = ((size_t)lh * kvs.block_size + off) * 2 * HD;
        if constexpr (STREAMED) {
          put((uint8_t*)kvs.host + (size_t)blk * pitch + at);
          const int32_t rs = kvs.res[blk];
          if (rs >= 0 && (uint32_t)rs < (uint32_t)kvs.n_blocks)
            put((uint8_t*)kvs.cache + (size_t)rs * pitch + at);
        } else {
          put((uint8_t*)kvs.cache + (size_t)blk * pitch + at);
        }
      }
    }
  } else if constexpr (KV != QK_KV_NONE) {
    if (isk && t < HD) {
      const int32_t sl = kvs.slot[tok];
      const size_t rowb = (size_t)(2 * HD);
      const __hip_bfloat16 kb = __float2bfloat16(res);
      const __hip_bfloat16 vb = kvs.v[(size_t)tok * kvs.v_pitch + (size_t)lh * HD + t];
      if constexpr (KV == QK_KV_STREAMED_BF16) {
        if (sl >= 0 && (uint32_t)sl / (uint32_t)kvs.block_size < (uint32_t)kvs.blocks) {
          const uint32_t blk = (uint32_t)sl / (uint32_t)kvs.block_size;
          const uint32_t off = (uint32_t)sl % (uint32_t)kvs.block_size;
          const size_t o = ((size_t)lh * kvs.block_size + off) * rowb;
          const size_t brow = (size_t)nkv * kvs.block_size * rowb;
          __hip_bfloat16* const m = (__hip_bfloat16*)kvs.host + (size_t)blk * brow + o;
          m[t]      = kb;
          m[HD + t] = vb;
          const int32_t rs = kvs.res[blk];
          if (rs >= 0 && (uint32_t)rs < (uint32_t)kvs.n_blocks) {
            __hip_bfloat16* const c = (__hip_bfloat16*)kvs.cache + (size_t)rs * brow + o;
            c[t]      = kb;
            c[HD + t] = vb;
          }
        }
      } else if (sl >= 0 && (uint32_t)sl / (uint32_t)kvs.block_size < (uint32_t)kvs.n_blocks) {
        const uint32_t blk = (uint32_t)sl / (uint32_t)kvs.block_size;
        const uint32_t off = (uint32_t)sl % (uint32_t)kvs.block_size;
        const size_t at = ((size_t)blk * nkv * kvs.block_size + off) * rowb +
                          (size_t)lh * kvs.block_size * rowb;
        __hip_bfloat16* c = (__hip_bfloat16*)kvs.cache + at;
        c[t]      = kb;
        c[HD + t] = vb;
      }
    }
  }
}

// The kernel around the body: CH `r4dchain` is the chained form (r4d_chain.h).
template <int HD, int KV, class CH>
__global__ __launch_bounds__(256)
void r4d_qk_norm_rope_gate_kernel(R4D_QK_NORM_ROPE_GATE_PARAMS) {
  if (CH::on) r4d_chain_wait();
  r4d_qk_norm_rope_gate_body<HD, KV>(QG, K, QO, KO, GO, WQ, WK, CS, POS, cs_rows, nq, nkv, rot, eps,
                                     wadd, qg_pitch, k_pitch, cs_f32, pos_i64, w_f32, kvs);
  if (CH::on) r4d_chain_done();
}
#undef R4D_QK_NORM_ROPE_GATE_PARAMS

// ============================================================ the cos/sin table
//
// The fused kernel reads a precomputed table instead of theta, and this fills it: one launch per
// step for every attention layer, instead of a rope launch per layer (~7 us of dispatch for ~1 us
// of work at head_dim 256).
//
// Each position is written at its own row, so the table has a row per nameable position and only
// the rows a step touches are filled (kilobytes at decode, not the whole ~6 MB table). Rows nothing
// addresses are never read.
//
// The arithmetic is r4d_model_bf16.hip's rope_kernel line for line, which makes the fusion
// bit-identical: `1.0f / powf(theta, 2i/den)` with `den` the ladder's width (`rot` for `rotary`,
// `head_dim` for `head`), the position divided by `rscale` in f32, `sincosf`, and the products
// consumed in f32 by the caller. Changing any of it changes the model's output.
#define R4D_QK_ROPE_TABLE_PARAMS                                                                \
    const void* __restrict__ POS, void* __restrict__ CS, long rows, int rot, int den,           \
    float theta, float rscale, int cs_f32, int pos_i64, RadYarn yarn
__device__ __forceinline__ void r4d_qk_rope_table_body(R4D_QK_ROPE_TABLE_PARAMS) {
  const int m = blockIdx.x;
  const long pp = pos_i64 ? ((const long*)POS)[m] : (long)((const int*)POS)[m];
  // A padded row addresses no table row, and a position at or past the table's height writes
  // nothing (it would land in whatever the arena placed after the plane). libref and libavx skip
  // the same rows.
  if (pp < 0 || pp >= rows) return;
  const int half = rot >> 1;
  const float p = (float)pp / (rscale != 0.0f ? rscale : 1.0f);
  const size_t o = (size_t)pp * rot;
  for (int i = threadIdx.x; i < half; i += (int)blockDim.x) {
    const float inv = yarn.frequency(1.0f / powf(theta, (float)(2 * i) / (float)den), i);
    float c, s;
    sincosf(p * inv, &s, &c);
    if (yarn.factor != 0.0f) { c *= yarn.magnitude; s *= yarn.magnitude; }
    if (cs_f32) {
      ((float*)CS)[o + i] = c;
      ((float*)CS)[o + half + i] = s;
    } else {
      ((__hip_bfloat16*)CS)[o + i] = __float2bfloat16(c);
      ((__hip_bfloat16*)CS)[o + half + i] = __float2bfloat16(s);
    }
  }
}

// The kernel around the body: CH `r4dchain` is the chained form (r4d_chain.h).
template <class CH>
__global__ __launch_bounds__(64)
void r4d_qk_rope_table_kernel(R4D_QK_ROPE_TABLE_PARAMS) {
  if (CH::on) r4d_chain_wait();
  r4d_qk_rope_table_body(POS, CS, rows, rot, den, theta, rscale, cs_f32, pos_i64, yarn);
  if (CH::on) r4d_chain_done();
}
#undef R4D_QK_ROPE_TABLE_PARAMS

// The two launches, as the plugin ABI's RadLaunchFn. See the note at the head of
// r4d_quant_act_i8.hip on the argument list and status return.
enum { QK_QGATE = 0, QK_K, QK_COSSIN, QK_POS, QK_WQ, QK_WK, QK_QOUT, QK_KOUT, QK_GOUT,
       QK_V, QK_SLOT, QK_CACHE, QK_RES, QK_HOST };

/* The body of both forms; KVF is the kv_store fold and `kvs` its cache, already checked. */
template <int KVF>
static int qk_body(const RadArgs* args, RadStream stream, const QkKvStore& kvs) {
  const bool chain = r4d_geti_or(args, "chain", 0) != 0;   // the chained form (r4d_chain.h)
  const RadTensor* qg = r4d_opt(args, QK_QGATE);
  const RadTensor* k  = r4d_opt(args, QK_K);
  const RadTensor* cs = r4d_opt(args, QK_COSSIN);
  const RadTensor* po = r4d_opt(args, QK_POS);
  const RadTensor* wq = r4d_opt(args, QK_WQ);
  const RadTensor* wk = r4d_opt(args, QK_WK);
  const RadTensor* qo = r4d_opt(args, QK_QOUT);
  const RadTensor* ko = r4d_opt(args, QK_KOUT);
  const RadTensor* go = r4d_opt(args, QK_GOUT);
  // The k-only form: `n_head` 0 means there is no [q|gate] projection to split (each head is a
  // plain `head_dim` run), so `q_gate`, `q_w` and `q_out` are absent and every head is served off
  // `k`. The kernel needs nothing for it: `isk` is `h >= nq`, so at nq 0 every block takes the k
  // path and the absent operands (null via r4d_p / r4d_wp) are never dereferenced.
  if (!k || !cs || !po || !wk || !ko) return R4D_NO(RAD_E_INVAL);
  if (!r4d_contig(cs) || !r4d_contig(po) || !r4d_contig(wk)) return R4D_NO(RAD_E_STRIDE);
  if (!r4d_contig(ko)) return R4D_NO(RAD_E_STRIDE);
  if (go && !r4d_contig(go)) return R4D_NO(RAD_E_STRIDE);

  const long long hd  = r4d_geti_or(args, "head_dim", 0);
  const long long nq  = r4d_geti_or(args, "n_head", 0);
  const long long nkv = r4d_geti_or(args, "n_head_kv", 0);
  const long long rot = r4d_geti_or(args, "rot", 0);
  if (nq < 0 || nkv < 1) return R4D_NO(RAD_E_SHAPE);
  if (nq >= 1 && (!qg || !wq || !qo)) return R4D_NO(RAD_E_INVAL);
  if (qg && (!r4d_contig(wq) || !r4d_contig(qo))) return R4D_NO(RAD_E_STRIDE);

  const long long qg_pitch = qg ? r4d_row_pitch(qg) : 0;
  const long long k_pitch  = r4d_row_pitch(k);
  if (qg_pitch < 0 || k_pitch < 0) return R4D_NO(RAD_E_STRIDE);

  // The row count is the operand's; which operand carries it depends on the form.
  const long long M = qg ? qg->shape[0] : k->shape[0];
  if (M < 1) return R4D_NO(RAD_E_SHAPE);
  if (hd != 64 && hd != 128 && hd != 256) return R4D_NO(RAD_E_SHAPE);   // the compiled set
  if (rot < 2 || (rot & 1) || rot > hd) return R4D_NO(RAD_E_SHAPE);
  const long long cs_rows = r4d_numel(cs) / rot;
  if (cs_rows < 1 || r4d_numel(cs) % rot) return R4D_NO(RAD_E_SHAPE);

  const float eps  = (float)r4d_getf(args, "eps", 1e-6);
  const float wadd = (float)r4d_getf(args, "wadd", 0.0);
  // The width flags come from the operands: a cos/sin table in f32 and positions in int64 are what
  // the tensors say they are.
  const int cs_f32  = (cs->dtype == RAD_F32) ? 1 : 0;
  const int pos_i64 = (po->dtype == RAD_I64) ? 1 : 0;
  // The norm gain's width is the operand's too (f32 in every architecture plugin here, bf16 in
  // vLLM). One flag serves both gains, so they must agree.
  if (wq && wq->dtype != wk->dtype) return R4D_NO(RAD_E_DTYPE);
  if (wk->dtype != RAD_F32 && wk->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
  const int w_f32 = (wk->dtype == RAD_F32) ? 1 : 0;

  const int nh = (int)(nq + nkv);
  // One block per (token, head), 256 threads, one element per lane. The width is fixed at 256, not
  // fitted to the head count, so the sum of squares matches r4d_rmsnorm_bf16 (see the kernel).
#define R4D_QKG_E(H, CH_) emit_rc = R4D_EMIT(stream, (r4d_qk_norm_rope_gate_kernel<H, KVF, CH_>), \
    dim3((int)M, nh), dim3(256), 0, (const __hip_bfloat16*)r4d_p(qg), \
    (const __hip_bfloat16*)r4d_p(k), (__hip_bfloat16*)r4d_wp(qo), (__hip_bfloat16*)r4d_wp(ko), \
    (__hip_bfloat16*)r4d_wp(go), r4d_p(wq), r4d_p(wk), r4d_p(cs), r4d_p(po), (long)cs_rows, \
    (int)nq, (int)nkv, (int)rot, eps, wadd, (int)qg_pitch, (int)k_pitch, cs_f32, pos_i64, w_f32, \
    kvs)
#define R4D_QKG_L(H) \
    do { if (chain) R4D_QKG_E(H, r4dchain); else R4D_QKG_E(H, R4dUnchained); } while (0)
  int emit_rc = RAD_OK;
  switch (hd) {
    case 64:  R4D_QKG_L(64); break;
    case 128: R4D_QKG_L(128); break;
    default:  R4D_QKG_L(256); break;
  }
#undef R4D_QKG_L
#undef R4D_QKG_E
  return emit_rc;
}

extern "C" int r4d_rad_qk_norm_rope_gate(const RadArgs* args, RadStream stream) {
  return qk_body<QK_KV_NONE>(args, stream, QkKvStore{});
}

/* The prologue with the KV store folded in: qk_norm_rope_gate's operands, then kv_store's `v`,
 * `slot_mapping` and `kv_cache`, and its parameters. The cache is bf16 or the int8 cache
 * ([n_blocks, kv_heads, block_size, 2 * HD + 4] bytes) as `kv_dtype` names, checked as kv_store
 * checks it, one slot per prologue row. The k kv_store would read is the prologue's own output, so
 * it is not a separate operand. `streamed` adds the residency map and the host mirror (docs/OPS.md
 * "kv_store_streamed"): each token's rotated k and its v go to the mirror's block row always and
 * to `kv_cache[res[block]]` only where the block is resident, one launch where qk_norm_rope_gate +
 * kv_store_streamed were two, and the same bytes. */
static int qk_kv_store(const RadArgs* args, RadStream stream, bool streamed) {
  const RadTensor* k   = r4d_opt(args, QK_K);
  const RadTensor* qg  = r4d_opt(args, QK_QGATE);
  const RadTensor* v   = r4d_opt(args, QK_V);
  const RadTensor* sm  = r4d_opt(args, QK_SLOT);
  const RadTensor* kv  = r4d_opt(args, QK_CACHE);
  const RadTensor* res = streamed ? r4d_opt(args, QK_RES) : nullptr;
  const RadTensor* ho  = streamed ? r4d_opt(args, QK_HOST) : nullptr;
  if (!k || !v || !sm || !kv || (streamed && (!res || !ho))) return R4D_NO(RAD_E_INVAL);
  const char* kvd = r4d_gets(args, "kv_dtype");
  const uint32_t want = kvd ? rad_dtype_parse(kvd) : (uint32_t)RAD_DT_INVALID;
  if (v->dtype != RAD_BF16 || sm->dtype != RAD_I32 || kv->dtype != want ||
      (kv->dtype != RAD_I8 && kv->dtype != RAD_BF16 && kv->dtype != RAD_F8E4M3))
    return R4D_NO(RAD_E_DTYPE);
  if (streamed && (res->dtype != RAD_I32 || ho->dtype != RAD_U8)) return R4D_NO(RAD_E_DTYPE);
  if (!r4d_contig(sm) || !r4d_contig(kv)) return R4D_NO(RAD_E_STRIDE);
  if (streamed && (!r4d_contig(res) || !r4d_contig(ho) || ho->rank != 2))
    return R4D_NO(RAD_E_STRIDE);
  const bool i8 = kv->dtype == RAD_I8;
  const bool fp8 = kv->dtype == RAD_F8E4M3;

  const long long hd  = r4d_geti_or(args, "head_dim", 0);
  const long long nkv = r4d_geti_or(args, "n_head_kv", 0);
  const long long bs  = r4d_geti_or(args, "block_size", 0);
  if (hd <= 0 || nkv <= 0 || bs <= 0) return R4D_NO(RAD_E_SHAPE);
  const long long rowe = i8 ? 2 * hd + 4 : 2 * hd;       /* a token's row in a head, in elements */
  if (kv->rank != 4 || kv->shape[1] != nkv || kv->shape[2] != bs || kv->shape[3] != rowe)
    return R4D_NO(RAD_E_SHAPE);
  const long long M = qg ? qg->shape[0] : k->shape[0];
  long long Mv = 0;
  const long long v_ld = r4d_rows_pitch(v, nkv * hd, &Mv);
  if (v_ld < 0) return R4D_NO(RAD_E_STRIDE);
  if (Mv < M || r4d_numel(sm) != M) return R4D_NO(RAD_E_SHAPE);

  QkKvStore kvs{};
  kvs.v          = (const __hip_bfloat16*)r4d_p(v);
  kvs.slot       = (const int32_t*)r4d_p(sm);
  kvs.cache      = r4d_wp(kv);
  kvs.v_pitch    = (int)v_ld;
  kvs.block_size = (int)bs;
  kvs.n_blocks   = (int)kv->shape[0];
  const float ks = (float)r4d_getf(args, "k_scale", 1.0);
  const float vs = (float)r4d_getf(args, "v_scale", 1.0);
  if (fp8 && (!(ks > 0.0f) || !(vs > 0.0f))) return R4D_NO(RAD_E_INVAL);
  kvs.k_inv = 1.0f / ks;
  kvs.v_inv = 1.0f / vs;
  if (!streamed && fp8) return qk_body<QK_KV_FP8>(args, stream, kvs);
  if (!streamed)
    return i8 ? qk_body<QK_KV_I8>(args, stream, kvs) : qk_body<QK_KV_BF16>(args, stream, kvs);

  /* One block of one layer: the mirror's row pitch and the ring's slot pitch are the same bytes,
   * so the store addresses both with one row arithmetic. */
  const long long row    = nkv * bs * rowe * (i8 || fp8 ? 1 : 2);
  const long long blocks = ho->shape[0];
  if (blocks <= 0 || kvs.n_blocks <= 0 || row <= 0) return R4D_NO(RAD_E_SHAPE);
  if (ho->shape[1] != row || ho->stride[0] != row) return R4D_NO(RAD_E_STRIDE);
  if (r4d_numel(res) < blocks) return R4D_NO(RAD_E_SHAPE);
  kvs.res    = (const int32_t*)r4d_p(res);
  kvs.host   = r4d_wp(ho);
  kvs.blocks = (int)blocks;
  if (fp8) return qk_body<QK_KV_STREAMED_FP8>(args, stream, kvs);
  return i8 ? qk_body<QK_KV_STREAMED_I8>(args, stream, kvs)
            : qk_body<QK_KV_STREAMED_BF16>(args, stream, kvs);
}
extern "C" int r4d_rad_qk_norm_rope_gate_kv_store(const RadArgs* args, RadStream stream) {
  return qk_kv_store(args, stream, false);
}
extern "C" int r4d_rad_qk_norm_rope_gate_kv_store_streamed(const RadArgs* args, RadStream stream) {
  return qk_kv_store(args, stream, true);
}

// The cos/sin plane the fused prologue reads: one launch a step, against `rope`'s one per layer.
// `qk_norm_rope_gate` takes a table, not a theta, so this row is what makes it usable.
//
// Only the rows `positions` names are written, each at its own row of a [rows, rot] plane whose
// height is the operand's. A row nothing addresses keeps its contents, which is correct because
// nothing reads it.
extern "C" int r4d_rad_rope_table(const RadArgs* args, RadStream stream) {
  const RadTensor* po = r4d_opt(args, 0);
  const RadTensor* cs = r4d_opt(args, 1);
  if (!po || !cs) return R4D_NO(RAD_E_INVAL);
  if (!r4d_contig(po) || !r4d_contig(cs)) return R4D_NO(RAD_E_STRIDE);
  if (cs->dtype != RAD_F32 && cs->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
  if (po->dtype != RAD_I32 && po->dtype != RAD_I64) return R4D_NO(RAD_E_DTYPE);

  // The row count is the operand's: `M` is declared as a band ceiling, but the call carries a
  // step's rows.
  const long long M   = r4d_numel(po);
  const long long rot = r4d_geti_or(args, "rot", 0);
  if (M < 1 || rot < 2 || (rot & 1)) return R4D_NO(RAD_E_SHAPE);
  if (r4d_numel(cs) % rot) return R4D_NO(RAD_E_SHAPE);

  /* `ladder`: `rotary` (default) measures the base ladder in `rot`; `head` in `head_dim`, the
   * proportional variant. `rope_table` has no head_dim by default, so `head` needs the parameter. */
  const long long head_dim = r4d_geti_or(args, "head_dim", 0);
  const char* lad = r4d_gets(args, "ladder");
  long long den = rot;
  if (lad) {
      if (!std::strcmp(lad, "head")) {
          if (head_dim <= 0) return R4D_NO(RAD_E_SHAPE);
          den = head_dim;
      } else if (std::strcmp(lad, "rotary")) {
          return R4D_NO(RAD_E_UNSUPPORTED);
      }
  }

  const float theta  = (float)r4d_getf(args, "theta", 10000.0);
  const float rscale = (float)r4d_getf(args, "scale", 1.0);
  const RadYarn yarn = rad_yarn_args(args);
  if (!rad_yarn_valid(yarn) || (yarn.factor != 0.0f && rscale != 1.0f)) return R4D_NO(RAD_E_INVAL);
#define R4D_RT(CH_)                                                                              \
  R4D_EMIT(stream, r4d_qk_rope_table_kernel<CH_>, dim3((int)M), dim3(64), 0,                     \
           r4d_p(po), r4d_wp(cs), (long)(r4d_numel(cs) / rot), (int)rot, (int)den, theta,        \
           rscale, cs->dtype == RAD_F32 ? 1 : 0, po->dtype == RAD_I64 ? 1 : 0, yarn)
  const int rc = r4d_geti_or(args, "chain", 0) != 0 && M <= 64 ? R4D_RT(r4dchain)
                                                               : R4D_RT(R4dUnchained);
#undef R4D_RT
  return rc;
}
