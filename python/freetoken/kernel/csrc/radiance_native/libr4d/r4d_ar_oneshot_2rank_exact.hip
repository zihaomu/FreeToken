// r4d_ar_oneshot_2rank_exact.hip: r4d_ar_oneshot_2rank_exact -- multi-block one-shot P2P-BAR
// all-reduce, exactly two ranks, for dual R9700 (gfx1201/RDNA4, PCIe). One-shot: each rank pushes
// its whole input, so scratch is sized by the full message and there is no ring or reduce-scatter
// stage. Two ranks is a hard constraint: the handshake is one peer flag per block. Pure HIP behind
// the C ABI in r4d.h; the caller passes raw pointers, element count, dtype and stream.
//
// Design:
//  * Push: each rank writes its input into the peer's IPC scratch, then reduces local input +
//    peer data (now in its own scratch) into out. Only scratch and flags are IPC-shared; input and
//    output stay local, so graph capture needs no buffer registration.
//  * Multi-block with per-chunk flags and no grid barrier: block b owns a contiguous chunk of 16B
//    words, pushes it, sets peer_flags[b], spins on my_flags[b] and reduces its chunk. This
//    spreads the push (PCIe) and the reduce (memory) across CUs.
//  * Graph-safe: seq is a device-resident per-block counter the kernel increments. Both ranks run
//    the same launch sequence (SPMD), so seq_ctr[b] stays in lockstep across ranks.
//  * Reuse: scratch is double-buffered by a per-launch parity (r4d_ar_seq.h explains why not
//    seq&1); same-stream serialisation plus the handshake drain a slot before reuse.
//  * Fine-grained memory (hipDeviceMallocFinegrained) is uncached and posted straight to the
//    fabric, so system-scope release/acquire skip the L2 writeback. The drain/acq fences are
//    selectable (default drain=1/acq=0 on fine-grained).
#include "r4d.h"
#include "r4d_chain.h"
#include "r4d_ar_seq.h"
#include "r4d_ar_hc.h"      // the gather word and the hc write word, shared with the rotated wire
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#define R4D_AR_SPIN_MAX 4000000000ULL
#define R4D_AR_MAX_BLOCKS 512
#define HIPOK(e) do { hipError_t _e = (e); if (_e != hipSuccess) \
    throw std::runtime_error(std::string("hip: ") + hipGetErrorString(_e)); } while (0)

__device__ __forceinline__ void store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int load_sys_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float rd_to_f(const float& x) { return x; }
__device__ __forceinline__ float rd_to_f(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float rd_to_f(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void rd_from_f(float v, float& o) { o = v; }
__device__ __forceinline__ void rd_from_f(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void rd_from_f(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// Vectorised 128-bit add: an int4 packs LANES elements of T, summed in fp32.
template <typename T, int LANES>
__device__ __forceinline__ int4 vadd(const int4& a, const int4& b) {
  int4 r;
  const T* pa = reinterpret_cast<const T*>(&a);
  const T* pb = reinterpret_cast<const T*>(&b);
  T* pr = reinterpret_cast<T*>(&r);
#pragma unroll
  for (int j = 0; j < LANES; ++j) rd_from_f(rd_to_f(pa[j]) + rd_to_f(pb[j]), pr[j]);
  return r;
}

// Multi-block one-shot all-reduce. Each block owns words [start,end) of the 16B-word space and
// drives its own flag handshake; no cross-block sync.
// CH `r4dchain`: the chained form (r4d_chain.h); no block returns early.
template <typename T, int LANES, class CH>
__global__ void r4d_ar_oneshot_2rank_exact_kernel(
    const int4* __restrict__ in4, int4* peer_base, const int4* my_base, int4* __restrict__ out4,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int n16, int slot_stride16, int drain, int acq, unsigned int nbw, long stage) {
  if (CH::on) r4d_chain_wait();
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;
  const long md = r4d_archk_md(stage, my_base);   // the host mailbox's VRAM mirror (r4d_ar_seq.h)

  const int w = (n16 + nb - 1) / nb;      // 16B words per block
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;
  unsigned int hck = 0u;   // the push check (r4d_ar_seq.h)

  // drain 4, tests only (r4d_ar_late_push_pin, r4d_selftest): the flag goes out before the push
  // and the push ~20 us after it, so the peer sees a flag whose bytes have not landed -- what a
  // push over PCIe P2P now and then does on its own -- on every launch.
  if (drain == 4) {
    if (tid == 0) store_sys_rel(&peer_flags[b], s);
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < 2000ull) __builtin_amdgcn_s_sleep(8);   // 100 MHz: 20 us
  }
  // Push the chunk to the peer slot, 4x unrolled for memory-level parallelism.
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      int4 v0 = in4[k], v1 = in4[k + nt], v2 = in4[k + 2 * nt], v3 = in4[k + 3 * nt];
      ps[k] = v0; ps[k + nt] = v1; ps[k + 2 * nt] = v2; ps[k + 3 * nt] = v3;
      hck += r4d_archk_u4(v0, 4u * (unsigned)k) + r4d_archk_u4(v1, 4u * (unsigned)(k + nt)) +
             r4d_archk_u4(v2, 4u * (unsigned)(k + 2 * nt)) + r4d_archk_u4(v3, 4u * (unsigned)(k + 3 * nt));
    }
    for (; k < end; k += nt) { const int4 v = in4[k]; ps[k] = v; hck += r4d_archk_u4(v, 4u * (unsigned)k); }
  }
  // Drain modes, ordering the pushes before the release flag so the peer sees data first:
  //  0 = none (relies on PCIe posted-write ordering and the release flag)
  //  1 = all-thread __threadfence_system: safe, but an L2 writeback-invalidate per wave, wasted
  //      on fine-grained memory and costly at high nt*nb
  //  2 = single-thread __threadfence_system: cheap, but drains only wave 0's stores
  //  3 = all-thread s_wait_storecnt 0: every wave drains its posted writes with no L2
  //      writeback-invalidate; PCIe keeps posted writes to one peer ordered, so the release flag
  //      lands after the data. Correct only for fine-grained memory (coarse leaves data in L2).
  r4d_archk_put(hck, peer_flags, nbw, (unsigned)slot, (unsigned)b, s, drain == 4);
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    if (drain != 4) store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    // A handshake that cannot complete is a lockstep violation; trap rather than sum stale bytes.
    while (r4d_ar_seq_behind(load_sys_acq(&my_flags[b]), s)) {
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  r4d_archk_wait(my_flags, nbw, (unsigned)slot, (unsigned)b, s, [&](unsigned int& h) {
    for (int k = start + tid; k < end; k += nt)
      h += r4d_archk_u4(r4d_archk_keep(&ms[k], md), 4u * (unsigned)k);
  });
  ms = r4d_archk_moved(ms, md);
  // Reduce the chunk, 4x unrolled: out = local input + peer input (staged in local scratch).
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      out4[k]          = vadd<T, LANES>(in4[k],          ms[k]);
      out4[k + nt]     = vadd<T, LANES>(in4[k + nt],     ms[k + nt]);
      out4[k + 2 * nt] = vadd<T, LANES>(in4[k + 2 * nt], ms[k + 2 * nt]);
      out4[k + 3 * nt] = vadd<T, LANES>(in4[k + 3 * nt], ms[k + 3 * nt]);
    }
    for (; k < end; k += nt) out4[k] = vadd<T, LANES>(in4[k], ms[k]);
  }
  if (CH::on) r4d_chain_done();
}

// The same all-reduce with the gated residual write as its epilogue. `y` is the message: each word
// is reduced as above and stored back, then the hc streams take
// h[r][c*n + i] += inj[r][c] * y[r][i] in hc_write's arithmetic (r4d_hc_bf16.hip) over the stored
// bf16 value. The output matches the two-kernel pair byte for byte without the second launch (a
// dependent launch gap is about 3 us on this part, against a 1-2 us write).
//
// n is a multiple of eight, so a word never straddles a row and the block that reduces it owns
// every stream element behind it.
// CH `r4dchain`: the chained form (r4d_chain.h); no block returns early.
template <class CH>
__global__ void r4d_ar_hc_write_kernel(
    int4* __restrict__ y4, int4* peer_base, const int4* my_base, unsigned int* peer_flags,
    unsigned int* my_flags, unsigned int* seq_ctrs, const unsigned short* __restrict__ inj,
    unsigned short* __restrict__ h, int n16, int n, int hc, long irow, long hrow,
    int slot_stride16, int drain, int acq, unsigned int nbw, long long* gate,
    unsigned int* gate_ctr, long stage) {
  if (CH::on) r4d_chain_wait();
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;
  const long md = r4d_archk_md(stage, my_base);   // the host mailbox's VRAM mirror (r4d_ar_seq.h)

  const int w = (n16 + nb - 1) / nb;      // 16B words per block
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;
  unsigned int hck = 0u;   // the push check (r4d_ar_seq.h)

  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      int4 v0 = y4[k], v1 = y4[k + nt], v2 = y4[k + 2 * nt], v3 = y4[k + 3 * nt];
      ps[k] = v0; ps[k + nt] = v1; ps[k + 2 * nt] = v2; ps[k + 3 * nt] = v3;
      hck += r4d_archk_u4(v0, 4u * (unsigned)k) + r4d_archk_u4(v1, 4u * (unsigned)(k + nt)) +
             r4d_archk_u4(v2, 4u * (unsigned)(k + 2 * nt)) + r4d_archk_u4(v3, 4u * (unsigned)(k + 3 * nt));
    }
    for (; k < end; k += nt) { const int4 v = y4[k]; ps[k] = v; hck += r4d_archk_u4(v, 4u * (unsigned)k); }
  }
  r4d_archk_put(hck, peer_flags, nbw, (unsigned)slot, (unsigned)b, s);
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    while (r4d_ar_seq_behind(load_sys_acq(&my_flags[b]), s)) {
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  r4d_archk_wait(my_flags, nbw, (unsigned)slot, (unsigned)b, s, [&](unsigned int& h) {
    for (int k = start + tid; k < end; k += nt)
      h += r4d_archk_u4(r4d_archk_keep(&ms[k], md), 4u * (unsigned)k);
  });
  ms = r4d_archk_moved(ms, md);
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  const int wpr = n >> 3;                 // 16B words a row
  for (int k = start + tid; k < end; k += nt) {
    const int4 r = vadd<__hip_bfloat16, 8>(y4[k], ms[k]);
    y4[k] = r;
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    r4d_ar_hc_write_word(inj, h, row, i0, n, hc, irow, hrow,
                         reinterpret_cast<const unsigned short*>(&r));
  }
  if (CH::on) r4d_chain_done();
}

// The same write with the MoE gather as its prologue. The routed experts' rows go back to token
// order, y[t][i] = sum over t's slots, ascending, of w[t][j] * y_expert[slot][i] rounded to bf16,
// with the shared arm folded in as moe_gather does; the word is pushed to the peer and kept as this
// rank's message. The gather uses moe_gather's round-to-nearest-even narrows (r4d_common.h), the hc
// write keeps hc_write's, so the bytes match the three separate kernels with two fewer launches.
//
// A block's words sit in at most R4D_ARG_ROWS rows (the launcher sizes the grid for it), and their
// slots are found once, in LDS, in one pass over the sorted list.
//
// That table does not suit a prefill chunk: eight rows per block means hundreds of blocks at 2048
// rows, each scanning the whole sorted list and paying its own flag round trip, where a few dozen
// already saturate the fabric. With INV the slots come from an inverse table instead
// (inv[token * top_k + slot] = sorted row holding it, or -1), built by r4d_ar_gather_inv_kernel
// beforehand, and the block count stays the all-reduce's. The gather's reads then overlap the push
// of words already summed. The arithmetic is identical to the table form.

/* inv[f] = i for every sorted row i holding slot f, -1 for a slot no row holds. One block, so the
 * clear and the scatter are separated by a barrier. */
__global__ void r4d_ar_gather_inv_kernel(int* __restrict__ inv, const int* __restrict__ sorted,
                                         int T) {
  for (int x = threadIdx.x; x < T; x += blockDim.x) inv[x] = -1;
  __syncthreads();
  for (int i = threadIdx.x; i < T; i += blockDim.x) {
    const int f = sorted[i];
    if (f >= 0 && f < T) inv[f] = i;
  }
}

// CH `r4dchain`: the chained form (r4d_chain.h); no block returns early.
template <bool INV, class CH>
__global__ void r4d_ar_gather_hc_write_kernel(
    int4* __restrict__ y4, int4* peer_base, const int4* my_base, unsigned int* peer_flags,
    unsigned int* my_flags, unsigned int* seq_ctrs, const unsigned short* __restrict__ inj,
    unsigned short* __restrict__ h, int n16, int n, int hc, long irow, long hrow,
    int slot_stride16, int drain, int acq, unsigned int nbw, R4dArGather G,
    const int* __restrict__ inv, long long* gate, unsigned int* gate_ctr, long stage) {
  if (CH::on) r4d_chain_wait();
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  __shared__ int   s_i[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_w[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_g[R4D_ARG_ROWS];
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;
  const long md = r4d_archk_md(stage, my_base);   // the host mailbox's VRAM mirror (r4d_ar_seq.h)

  const int w = (n16 + nb - 1) / nb;      // 16B words per block
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;
  unsigned int hck = 0u;   // the push check (r4d_ar_seq.h)
  const int wpr = n >> 3;                 // 16B words a row
  const int row_lo = start / wpr;
  const int nrow = end > start ? (end - 1) / wpr - row_lo + 1 : 0;
  const int K = G.top_k;

  /* Slot table. A slot no sorted row claims reads -1, so clear and scan are separated by a
   * barrier, as in moe_gather. */
  if (!INV) {
    for (int x = tid; x < nrow * K; x += nt) s_i[x] = -1;
    __syncthreads();
    {
      const int lo = row_lo * K, hi = (row_lo + nrow) * K;
      for (int i = tid; i < G.T; i += nt) {
        const int f = G.sorted[i];
        if (f >= lo && f < hi) s_i[f - lo] = i;
      }
    }
    for (int x = tid; x < nrow * K; x += nt)
      s_w[x] = bf2f(G.ew[(long)(row_lo + x / K) * G.ew_ld + x % K]);
    if (G.sh)
      for (int r = tid; r < nrow; r += nt) {
        const float v0 = bf2f(G.sg[(long)(row_lo + r) * G.sg_ld]);
        s_g[r] = G.sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
      }
    __syncthreads();
  }

  /* Gather one word per thread per pass; push it and keep it. */
  for (int k = start + tid; k < end; k += nt) {
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    const int4 out = r4d_ar_gather_word<INV>(G, inv, s_i, s_w, s_g, row, row - row_lo, i0);
    y4[k] = out;
    ps[k] = out;
    hck += r4d_archk_u4(out, 4u * (unsigned)k);
  }
  r4d_archk_put(hck, peer_flags, nbw, (unsigned)slot, (unsigned)b, s);
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    while (r4d_ar_seq_behind(load_sys_acq(&my_flags[b]), s)) {
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  r4d_archk_wait(my_flags, nbw, (unsigned)slot, (unsigned)b, s, [&](unsigned int& h) {
    for (int k = start + tid; k < end; k += nt)
      h += r4d_archk_u4(r4d_archk_keep(&ms[k], md), 4u * (unsigned)k);
  });
  ms = r4d_archk_moved(ms, md);
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  for (int k = start + tid; k < end; k += nt) {
    const int4 r = vadd<__hip_bfloat16, 8>(y4[k], ms[k]);
    y4[k] = r;
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    r4d_ar_hc_write_word(inj, h, row, i0, n, hc, irow, hrow,
                         reinterpret_cast<const unsigned short*>(&r));
  }
  if (CH::on) r4d_chain_done();
}

// The IPC handle goes into a caller-owned buffer of R4D_AR_HANDLE_BYTES so this stays a C ABI.
static_assert(sizeof(hipIpcMemHandle_t) <= R4D_AR_HANDLE_BYTES, "ipc handle larger than the ABI buffer");

long r4d_ar_ipc_alloc(long size, int finegrained, char* out_handle, int* out_len) {
  void* p = nullptr;
  if (finegrained) HIPOK(hipExtMallocWithFlags(&p, size, hipDeviceMallocFinegrained));
  else HIPOK(hipMalloc(&p, size));
  HIPOK(hipMemset(p, 0, size));
  // The zeroes must land before the pointer reaches a peer, or its first push or flag store could
  // race the queued memset.
  HIPOK(hipStreamSynchronize(nullptr));
  // The handle is for an out-of-process peer. libr4d's ranks are threads of one process and share
  // the pointer itself (r4d_ar_entry.hip), so a refused export is not fatal: beside another engine
  // on the same cards hipIpcGetMemHandle answers invalid argument while the allocation is fine.
  hipIpcMemHandle_t h;
  if (hipIpcGetMemHandle(&h, p) == hipSuccess) {
    std::memcpy(out_handle, &h, sizeof(h));
    *out_len = (int)sizeof(h);
  } else {
    (void)hipGetLastError();
    std::memset(out_handle, 0, R4D_AR_HANDLE_BYTES);
    *out_len = 0;
  }
  return (long)p;
}
// r4d_p2p_copy2d as a kernel with a bounded workgroup count.
//
// hipMemcpy2DAsync is not a DMA engine on this part: it runs as
// `__amd_rocclr_copyBufferRectAligned`, one 16-thread workgroup per 16 bytes of a row (62080
// workgroups for 64 x 248 KB). Alone it reaches link rate, but beside the GEMM it should hide
// under, it fills every scheduling slot and slows that GEMM by about 60%.
//
// A peer copy needs only a small grid: 32 workgroups of 64 threads is within 0.1% of the best
// configuration at every size. The grid is an argument and defaults to at most 64 workgroups.
//
// 16-byte accesses: every pitch, the width and both base addresses must be 16-byte aligned; the
// caller checks and falls back to r4d_p2p_copy2d otherwise.
typedef unsigned r4d_u4v __attribute__((ext_vector_type(4)));

__global__ __launch_bounds__(256) void r4d_p2p_copy2d_k(const r4d_u4v* __restrict__ src,
                                                        r4d_u4v* __restrict__ dst, long spitch16,
                                                        long dpitch16, long wpr, int height) {
  const long cstride = (long)gridDim.x * blockDim.x;
  for (int r = blockIdx.y; r < height; r += gridDim.y) {
    const r4d_u4v* s = src + (long)r * spitch16;
    r4d_u4v* d = dst + (long)r * dpitch16;
    for (long c = (long)blockIdx.x * blockDim.x + threadIdx.x; c < wpr; c += cstride) d[c] = s[c];
  }
}

int r4d_p2p_copy2d_kernel(long dst, long src, long dpitch, long spitch, long width, long height,
                          int bx, int by, int threads, long stream) {
  if ((dst | src | dpitch | spitch | width) & 15L) return -1;
  if (height <= 0 || width <= 0) return 0;
  if (threads <= 0 || threads > 256 || (threads & 31)) return -2;
  const long wpr = width / 16;
  if (bx <= 0) bx = (int)((wpr + threads - 1) / threads);
  if (bx > 64) bx = 64;
  if (by <= 0) by = (int)height;
  if (by > (int)height) by = (int)height;
  hipLaunchKernelGGL(r4d_p2p_copy2d_k, dim3(bx, by), dim3(threads), 0, (hipStream_t)stream,
                     (const r4d_u4v*)src, (r4d_u4v*)dst, spitch / 16, dpitch / 16, wpr,
                     (int)height);
  hipError_t e = hipGetLastError();
  return e == hipSuccess ? 0 : -100 - (int)e;
}

// A system-scope release on `stream`, for when this device has written into peer memory that
// another agent will read. A peer write from a copy or kernel is posted and retires long before it
// lands; only a system-scope release orders it against a later handshake on a different path. One
// thread suffices, since the fence is a memory-system operation.
__global__ void r4d_p2p_fence_kernel() { __threadfence_system(); }

int r4d_p2p_fence(long stream) {
  hipLaunchKernelGGL(r4d_p2p_fence_kernel, dim3(1), dim3(1), 0, (hipStream_t)stream);
  hipError_t e = hipGetLastError();
  return e == hipSuccess ? 0 : -100 - (int)e;
}

// A strided device-to-device copy: a two-rank all-gather once the peer's buffer is IPC-mapped.
// The runtime's copy path is the fast one for peer traffic on this part (13.52 GB/s per direction
// with both directions busy, against 9.30 for a b128-load kernel and 10.1 for RCCL).
// `dpitch`, `spitch` and `width` are bytes; `height` is rows.
int r4d_p2p_copy2d(long dst, long src, long dpitch, long spitch, long width, long height,
                   long stream) {
  hipError_t e = hipMemcpy2DAsync((void*)dst, (size_t)dpitch, (const void*)src, (size_t)spitch,
                                  (size_t)width, (size_t)height, hipMemcpyDeviceToDevice,
                                  (hipStream_t)stream);
  return e == hipSuccess ? 0 : -100 - (int)e;
}

void r4d_ar_hc_write(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                     long seq_ctrs, long slot_stride16, long y, long inj, long h, long M, long n,
                     long hc, long irow, long hrow, long stream_i, long nblocks, long nthreads,
                     long drain, long acq, long gate, long gate_ctr, long stage, long chain) {
  hipStream_t st = (hipStream_t)stream_i;
  if (n <= 0 || M <= 0 || n % 8) throw std::runtime_error("ar_hc_write: n must be a positive multiple of 8");
  if (hc < 1 || hrow % 8) throw std::runtime_error("ar_hc_write: the streams must be whole 16B words");
  if (M * n * 2 / 16 > slot_stride16) throw std::runtime_error("ar_hc_write: message larger than a slot");
  const int n16 = (int)(M * n * 2 / 16);
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_MAX_BLOCKS) nb = R4D_AR_MAX_BLOCKS;
  const unsigned nbw = (unsigned)r4d_ar_max_blocks();
  if ((unsigned)nb > nbw) nb = (int)nbw;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  nt &= ~31;   // whole waves: the push check's wave sum reads all 32 lanes
#define R4D_ARHW(CH_)                                                                           \
  r4d_ar_hc_write_kernel<CH_><<<nb, nt, 0, st>>>(                                               \
      (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned*)peer_flags,           \
      (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h, \
      n16, (int)n, (int)hc, irow, hrow, (int)slot_stride16, (int)drain, (int)acq, nbw,         \
      (long long*)gate, (unsigned int*)gate_ctr, stage)
  if (chain) R4D_ARHW(r4dchain); else R4D_ARHW(R4dUnchained);
#undef R4D_ARHW
}

int r4d_ar_gather_hc_write(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                           long seq_ctrs, long slot_stride16, long y, long inj, long h, long M,
                           long n, long hc, long irow, long hrow, long ye, long ew, long sorted,
                           long sh, long sg, long ye_ld, long ew_ld, long sh_ld, long sg_ld,
                           long top_k, long sig, long stream_i, long nblocks, long nthreads,
                           long drain, long acq, long inv, long gate, long gate_ctr, long stage,
                           long chain) {
  hipStream_t st = (hipStream_t)stream_i;
  if (n <= 0 || M <= 0 || n % 8 || hc < 1 || hrow % 8) return -1;
  if (top_k < 1 || top_k > R4D_ARG_SLOTS) return -2;
  if (ye_ld % 8 || (sh && sh_ld < n)) return -3;
  if (M * n * 2 / 16 > slot_stride16) return -5;      /* larger than a slot */
  const int n16 = (int)(M * n * 2 / 16);
  const int wpr = (int)(n / 8);
  const unsigned nbw = (unsigned)r4d_ar_max_blocks();
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  /* Enough blocks that none spans more than R4D_ARG_ROWS rows: a block of w words touches at most
   * (w - 1) / wpr + 2 rows. Both ranks compute the same count from the same M and n, as the
   * per-block handshake requires. */
  const int wmax = (R4D_ARG_ROWS - 2) * wpr + 1;
  const int nb_min = (n16 + wmax - 1) / wmax;
  /* Use the inverse table when the LDS table would need more blocks than requested (a prefill
   * chunk, never a decode step); both ranks decide from the same M and n. The block count stays
   * the requested one: at a 2048-row chunk, 48, 96 and 192 blocks are no faster than 24. */
  const bool by_inv = inv != 0 && nb_min > nb;
  if (!by_inv && nb < nb_min) nb = nb_min;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_MAX_BLOCKS || (unsigned)nb > nbw) return -4;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  nt &= ~31;   // whole waves: the push check's wave sum reads all 32 lanes
  /* Zeroed including padding: the struct is passed by value, so its tail padding is part of the
   * kernel-argument bytes a recorded pass's audit compares. */
  R4dArGather G;
  std::memset(&G, 0, sizeof G);
  G.ye = (const unsigned short*)ye;
  G.ew = (const unsigned short*)ew;
  G.sorted = (const int*)sorted;
  G.sh = (const unsigned short*)sh;
  G.sg = (const unsigned short*)sg;
  G.ye_ld = ye_ld; G.ew_ld = ew_ld; G.sh_ld = sh_ld; G.sg_ld = sg_ld;
  G.top_k = (int)top_k; G.T = (int)(M * top_k); G.sig = (int)sig;
  /* The inverse table's two launches stay plain; the decode form's one has a chained form. */
#define R4D_ARGHW(INV_, CH_, inv_)                                                              \
    r4d_ar_gather_hc_write_kernel<INV_, CH_><<<nb, nt, 0, st>>>(                                \
        (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned*)peer_flags,         \
        (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h, \
        n16, (int)n, (int)hc, irow, hrow, (int)slot_stride16, (int)drain, (int)acq, nbw, G,    \
        inv_, (long long*)gate, (unsigned int*)gate_ctr, stage)
  if (by_inv) {
    r4d_ar_gather_inv_kernel<<<1, 1024, 0, st>>>((int*)inv, (const int*)sorted, G.T);
    R4D_ARGHW(true, R4dUnchained, (const int*)inv);
  } else if (chain) {
    R4D_ARGHW(false, r4dchain, nullptr);
  } else {
    R4D_ARGHW(false, R4dUnchained, nullptr);
  }
#undef R4D_ARGHW
  return (int)hipGetLastError();
}

/* The inverse slot table alone, for the wide fused gather (r4d_ar_oneshot_Nrank_exact.hip); same
 * kernel and launch as above. */
int r4d_ar_gather_inv(long inv, long sorted, long T, long stream_i) {
  if (!inv || !sorted || T < 1) return -1;
  r4d_ar_gather_inv_kernel<<<1, 1024, 0, (hipStream_t)stream_i>>>((int*)inv, (const int*)sorted,
                                                                  (int)T);
  return (int)hipGetLastError();
}

void r4d_ar_ipc_free(long p) { if (p) (void)hipFree((void*)p); }

// Host mailbox. Coherent so a peer's stores are not held in a CPU cache line, mapped so cards can
// address it, portable so every card can (not only the current one). Cards reach it over the same
// fine-grained path as a peer's VRAM scratch, so the kernels' fences apply unchanged. The card
// address must equal the CPU address, because one pointer is published to every rank.
long r4d_ar_host_alloc(long size) {
  void* p = nullptr;
  HIPOK(hipHostMalloc(&p, size, hipHostMallocCoherent | hipHostMallocMapped |
                                    hipHostMallocPortable));
  std::memset(p, 0, (size_t)size);
  void* d = nullptr;
  const hipError_t e = hipHostGetDevicePointer(&d, p, 0);
  if (e != hipSuccess || d != p) {
    (void)hipHostFree(p);
    throw std::runtime_error(e != hipSuccess
        ? std::string("host mailbox: no card address: ") + hipGetErrorString(e)
        : std::string("host mailbox: the card address differs from the CPU address"));
  }
  return (long)p;
}
void r4d_ar_host_free(long p) { if (p) (void)hipHostFree((void*)p); }

// Enable direct peer access. Ranks are threads of one process sharing an address space, so a peer
// pointer is valid once this has run and no IPC handle exchange is needed (see r4d_ar_entry.hip).
void r4d_ar_ipc_enable_peer(long peer) {
  hipError_t e = hipDeviceEnablePeerAccess((int)peer, 0);
  if (e == hipErrorPeerAccessAlreadyEnabled) { (void)hipGetLastError(); return; }
  if (e != hipSuccess)
    throw std::runtime_error(std::string("enable_peer: ") + hipGetErrorString(e));
}

/* r4d_ar_late_push_pin: 1 makes every 2-rank exact all-reduce send its flags before its pushes
 * (drain 4). A test hook (r4d_selftest); nothing else sets it. */
static int g_ar_late_push = 0;
extern "C" void r4d_ar_late_push_pin(int on) { g_ar_late_push = on; }

void r4d_ar_oneshot_2rank_exact(long peer_scratch, long my_scratch, long peer_flags,
                            long my_flags, long seq_ctrs, long slot_stride16,
                            long inp, long out, long n_elem, long dtype,
                            long stream_i, long nblocks, long nthreads,
                            long drain, long acq, long stage, long chain) {
  hipStream_t st = (hipStream_t)stream_i;
  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_reduce: n_elem*esize not 16B-aligned");
  if (nbytes / 16 > slot_stride16) throw std::runtime_error("all_reduce: message larger than a slot");
  const int n16 = (int)(nbytes / 16);
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_MAX_BLOCKS) nb = R4D_AR_MAX_BLOCKS;
  // Block capacity of the counter allocation. Every 2-rank kernel on an instance must agree on it,
  // because the parity counters start that many words in (r4d_ar_seq.h).
  const unsigned nbw = (unsigned)r4d_ar_max_blocks();
  if ((unsigned)nb > nbw) nb = (int)nbw;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  nt &= ~31;   // whole waves: the push check's wave sum reads all 32 lanes
  // Tests only: every push lands after its flag (drain 4 above). Not in the tuning table, so no
  // tuner can pick it.
  if (g_ar_late_push) drain = 4;
#define LK(T, LN, CH_) r4d_ar_oneshot_2rank_exact_kernel<T, LN, CH_><<<nb, nt, 0, st>>>(   \
    (const int4*)inp, (int4*)peer_scratch, (const int4*)my_scratch, (int4*)out,        \
    (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, n16,              \
    (int)slot_stride16, (int)drain, (int)acq, nbw, stage)
#define L(T, LN) do { if (chain) LK(T, LN, r4dchain); else LK(T, LN, R4dUnchained); } while (0)
  if (dtype == 0) L(__hip_bfloat16, 8);
  else if (dtype == 1) L(__half, 8);
  else if (dtype == 2) L(float, 4);
  else throw std::runtime_error("all_reduce: bad dtype");
#undef L
#undef LK
}

