// r4d_ar_gather_2rank.hip: r4d_ar_gather_2rank -- multi-block one-shot P2P all-GATHER, exactly two
// ranks, for dual R9700 (gfx1201/RDNA4, PCIe).
//
// A kernel rather than r4d_p2p_copy2d: a copy moves the data but carries no handshake, so the
// reader cannot know the writer's bytes have landed, and ordering it with a fence plus a barrier
// collective costs three launches and a second rendezvous. This is the one-shot all-reduce minus
// the reduction (same push, per-block flag protocol and double-buffered slot); it is a separate
// file because the output shapes differ (numel vs world_size * numel).
//
// Used by the vocab-parallel sampler. lm_head is row-sharded, so each rank reduces its vocabulary
// slice to k candidates and only the candidate sets are combined (32 KiB at 4 rows x 1024
// candidates, against ~1 GiB for full logits). The payload is (u32 token id, f32 logit) pairs,
// so the bytes are moved, never summed.
//
// `dtype` only converts n_elem to bytes.
#include "r4d.h"
#include "r4d_ar_seq.h"
#include <cstdint>
#include <stdexcept>
#include <string>

#define R4D_AR_GATHER_SPIN_MAX 4000000000ULL
#define R4D_AR_GATHER_MAX_BLOCKS 512

__device__ __forceinline__ void r4d_g_store_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int r4d_g_load_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Where a rank's words land. `row16` is one row of this rank's contribution, in 16-byte words.
//
//   row16 <= 0 or row16 == n16  -- one row: rank r's whole slice at r * n16 (plain concatenation).
//   otherwise                   -- row-interleaved: input [rows][row16], output
//                                 [rows][world][row16], so row m holds every rank's words for
//                                 that row side by side.
//
// The vocab-parallel sampler's merge reads one row of gathered candidates per position at a single
// row stride, which the concatenated form cannot express. Interleaving costs one division per word
// and saves a transpose launch. `dp16` is the output's row pitch in words: 2 * row16 for a tight
// [rows][2][row16], more when the gathered rows are the leading columns of a wider buffer.
__device__ __forceinline__ int r4d_g_dst(int k, int row16, int n16, int who, int dp16) {
  if (row16 <= 0 || row16 >= n16) return who * n16 + k;
  const int m = k / row16, c = k - m * row16;
  return m * dp16 + who * row16 + c;
}

// Where word k of this rank's contribution is read: `sp16` is the input's row pitch in words,
// row16 when its rows are tight, more when they are a window of a wider row.
__device__ __forceinline__ int r4d_g_src(int k, int row16, int n16, int sp16) {
  if (row16 <= 0 || row16 >= n16 || sp16 == row16) return k;
  const int m = k / row16, c = k - m * row16;
  return m * sp16 + c;
}

// Block b owns words [start,end) of the 16B-word space and drives its own flag handshake; no
// cross-block sync, as in the all-reduce.
__global__ void r4d_ar_gather_2rank_kernel(
    const int4* __restrict__ in4, int4* peer_base, const int4* my_base, int4* __restrict__ out4,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int n16, int slot_stride16, int rank, int row16, int dp16, int sp16, int drain, int acq,
    unsigned int nbw, long stage) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;
  const long md = r4d_archk_md(stage, my_base);   // the host mailbox's VRAM mirror (r4d_ar_seq.h)

  const int w = (n16 + nb - 1) / nb;
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;
  unsigned int hck = 0u;   // the push check (r4d_ar_seq.h)

  // Push the chunk to the peer slot and place this rank's own slice (a local store, no handshake)
  // while the push is in flight.
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      int4 v0 = in4[r4d_g_src(k, row16, n16, sp16)], v1 = in4[r4d_g_src(k + nt, row16, n16, sp16)],
           v2 = in4[r4d_g_src(k + 2 * nt, row16, n16, sp16)],
           v3 = in4[r4d_g_src(k + 3 * nt, row16, n16, sp16)];
      ps[k] = v0; ps[k + nt] = v1; ps[k + 2 * nt] = v2; ps[k + 3 * nt] = v3;
      hck += r4d_archk_u4(v0, 4u * (unsigned)k) + r4d_archk_u4(v1, 4u * (unsigned)(k + nt)) +
             r4d_archk_u4(v2, 4u * (unsigned)(k + 2 * nt)) + r4d_archk_u4(v3, 4u * (unsigned)(k + 3 * nt));
      out4[r4d_g_dst(k,          row16, n16, rank, dp16)] = v0;
      out4[r4d_g_dst(k + nt,     row16, n16, rank, dp16)] = v1;
      out4[r4d_g_dst(k + 2 * nt, row16, n16, rank, dp16)] = v2;
      out4[r4d_g_dst(k + 3 * nt, row16, n16, rank, dp16)] = v3;
    }
    for (; k < end; k += nt) {
      int4 v = in4[r4d_g_src(k, row16, n16, sp16)];
      ps[k] = v;
      hck += r4d_archk_u4(v, 4u * (unsigned)k);
      out4[r4d_g_dst(k, row16, n16, rank, dp16)] = v;
    }
  }
  // Drain modes, as in the all-reduce: order the pushes before the release flag so the peer cannot
  // see the flag before the data.
  r4d_archk_put(hck, peer_flags, nbw, (unsigned)slot, (unsigned)b, s);
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    r4d_g_store_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    // A handshake that cannot complete is a lockstep violation; trap rather than pass on stale
    // bytes as the peer's.
    while (r4d_ar_seq_behind(r4d_g_load_acq(&my_flags[b]), s)) {
      if (++z > R4D_AR_GATHER_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  r4d_archk_wait(my_flags, nbw, (unsigned)slot, (unsigned)b, s, [&](unsigned int& h) {
    for (int k = start + tid; k < end; k += nt)
      h += r4d_archk_u4(r4d_archk_keep(&ms[k], md), 4u * (unsigned)k);
  });
  ms = r4d_archk_moved(ms, md);

  // Place the peer's slice, now staged in this rank's scratch.
  {
    const int who = 1 - rank;
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      out4[r4d_g_dst(k,          row16, n16, who, dp16)] = ms[k];
      out4[r4d_g_dst(k + nt,     row16, n16, who, dp16)] = ms[k + nt];
      out4[r4d_g_dst(k + 2 * nt, row16, n16, who, dp16)] = ms[k + 2 * nt];
      out4[r4d_g_dst(k + 3 * nt, row16, n16, who, dp16)] = ms[k + 3 * nt];
    }
    for (; k < end; k += nt) out4[r4d_g_dst(k, row16, n16, who, dp16)] = ms[k];
  }
}

void r4d_ar_gather_2rank(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                         long seq_ctrs, long slot_stride16, long inp, long out, long n_elem,
                         long dtype, long rank, long row_elems, long stream_i, long nblocks,
                         long nthreads, long drain, long acq, long dpitch, long spitch,
                         long stage) {
  hipStream_t st = (hipStream_t)stream_i;
  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_gather: n_elem*esize not 16B-aligned");
  if (rank != 0 && rank != 1) throw std::runtime_error("all_gather: rank must be 0 or 1");
  if (nbytes / 16 > slot_stride16) throw std::runtime_error("all_gather: message larger than a slot");
  const int n16 = (int)(nbytes / 16);
  // One row of this rank's contribution, in 16-byte words. Zero means one row (plain
  // concatenation); otherwise rows must be whole 16-byte words and divide the message.
  long rowb = (long)row_elems * esize;
  if (row_elems > 0 && (rowb % 16 != 0 || nbytes % rowb != 0))
    throw std::runtime_error("all_gather: row is not a whole number of 16B words");
  const int row16 = row_elems > 0 ? (int)(rowb / 16) : 0;
  // The output's row pitch in elements, 0 for a tight [rows][2][row]: whole words, at least the
  // two ranks' row.
  const long dpb = (long)dpitch * esize;
  if (dpitch > 0 && (row16 <= 0 || dpb % 16 != 0 || dpitch < 2 * row_elems))
    throw std::runtime_error("all_gather: the output pitch is not a whole number of 16B words "
                             "at least two rows wide");
  const int dp16 = dpitch > 0 ? (int)(dpb / 16) : 2 * row16;
  // The input's row pitch in elements, 0 for tight rows: whole words, at least a row.
  const long spb = (long)spitch * esize;
  if (spitch > 0 && (row16 <= 0 || spb % 16 != 0 || spitch < row_elems))
    throw std::runtime_error("all_gather: the input pitch is not a whole number of 16B words "
                             "at least a row wide");
  const int sp16 = spitch > 0 ? (int)(spb / 16) : row16;
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_GATHER_MAX_BLOCKS) nb = R4D_AR_GATHER_MAX_BLOCKS;
  // Block capacity of the counter allocation shared with the all-reduce (r4d_ar_seq.h).
  const unsigned nbw = (unsigned)r4d_ar_max_blocks();
  if ((unsigned)nb > nbw) nb = (int)nbw;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  nt &= ~31;   // whole waves: the push check's wave sum reads all 32 lanes
  r4d_ar_gather_2rank_kernel<<<nb, nt, 0, st>>>(
      (const int4*)inp, (int4*)peer_scratch, (const int4*)my_scratch, (int4*)out,
      (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, n16,
      (int)slot_stride16, (int)rank, row16, dp16, sp16, (int)drain, (int)acq, nbw, stage);
}
