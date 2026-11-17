// r4d_ar_oneshot_2rank_wht6.hip: translation unit for r4d_ar_oneshot_2rank_wht6, the all-reduce whose wire
// payload is Walsh-Hadamard rotated and quantised to 6 bits. Device code and launcher live in
// r4d_ar_oneshot_2rank_wht6.h so they also compile standalone without r4d; this file exports them
// through the library's C ABI with the packing constants for sizing scratch. It also holds the
// rotated wire's fused ar_hc_write and ar_gather_hc_write, which need r4d_ar_hc.h.
#include "r4d.h"
#include "r4d_ar_hc.h"   // the gather word and the hc write word, shared with the exact wire
#include "r4d_ar_oneshot_2rank_wht6.h"
#include <cstring>

void r4d_ar_oneshot_2rank_wht6(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                               long seq_ctrs, long loc_pack, long slot_stride_bytes,
                               long scale_off_bytes, long inp, long out, long n_elem, long dtype,
                               long stream, long nblocks, long nthreads, long drain, long acq,
                               long stage) {
  r4d_ar_wht6_launch(peer_scratch, my_scratch, peer_flags, my_flags, seq_ctrs, loc_pack,
                     slot_stride_bytes, scale_off_bytes, inp, out, n_elem, dtype, stream, nblocks,
                     nthreads, drain, acq, stage);
}

// Block cap of both all-reduce kernels (the flag array is sized from it).
int r4d_ar_max_blocks(void) { return R4D_AR_MAX_BLOCKS; }

// Rotated 6-bit wire format dimensions, for sizing the caller's packed scratch.
void r4d_ar_wht6_dims(int* group, int* bits, int* chunk_elems) {
  if (group)       *group       = GROUP;
  if (bits)        *bits        = BITS;
  if (chunk_elems) *chunk_elems = CHUNK_ELEMS;
}

// ---------------------------------------------------------------------------------------------
// The rotated wire with the gated residual write as epilogue and, with GATHER, the MoE gather as
// prologue: 6-bit counterparts of r4d_ar_hc_write_kernel and r4d_ar_gather_hc_write_kernel
// (r4d_ar_oneshot_2rank_exact.hip). Each matches the unfused chain (moe_gather,
// r4d_ar_oneshot_2rank_wht6, hc_write) byte for byte because every stage is the same code: the
// gather and write words from r4d_ar_hc.h, and wht6_push_chunk / wht6_reduce_chunk at a smaller
// chunk, which changes byte ownership but not values. Gathered words are rounded to bf16 as
// moe_gather stores them; the reduced sum is rounded to bf16, stored into `y` and read back as
// such by the write.
//
// A lane's EPL elements are one word, so a word and its write never straddle a row even where a
// group or chunk does: the rotation spans the flattened message, the write the word's row.
//
// The gather reads its slots from an LDS table the block builds in one pass over the sorted list,
// covering every row it owns (whole wave-chunks, so tens of rows at a prefill chunk); unlike the
// exact form it is sized by the launch, not fixed at R4D_ARG_ROWS. Reads of the routed rows then
// depend only on their own addresses, where an inverse table adds a dependent slot load before
// each. The launcher widens the grid until a block's table fits.
static_assert(EPL == 8, "a lane's share of a group is one 16B word of the message");

// The LDS a block's slot table may take; the handshake's two words sit beside it.
static constexpr size_t kWht6TableBytes = 60 * 1024;

// The fused forms use 8-group chunks (512 elements, two group-iterations per wave) instead of the
// standalone 32. A wave walks its groups serially, each waiting on its own reads (routed rows,
// streams), so a quarter-size chunk puts four times as many waves on the message: at a 2048-row
// chunk the gather form takes 403 us against 500 at 32 groups and 416 at 16. Both ranks run the
// same kernel, so they push and reduce under the same chunking.
static constexpr int kWht6FusedCG = 8;

// The rows block b's chunks of `ce` elements touch: [row_lo, row_lo + nrow).
__device__ __forceinline__ void wht6_block_rows(int c0, int c1, int ce, int n_groups, int n,
                                                int& row_lo, int& nrow) {
  const long e0 = (long)c0 * ce;
  long e1 = (long)c1 * ce;
  if (e1 > (long)n_groups * GROUP) e1 = (long)n_groups * GROUP;
  row_lo = (int)(e0 / n);
  nrow = e1 > e0 ? (int)((e1 - 1) / n) - row_lo + 1 : 0;
}

template <bool GATHER, int CG>
__global__ void r4d_ar_hc_write_wht6_kernel(
    unsigned short* __restrict__ y, char* peer_scratch, const char* my_scratch, char* loc_pack,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    const unsigned short* __restrict__ inj, unsigned short* __restrict__ h, int n, int hc,
    long irow, long hrow, int n_groups, int n_chunks, long slot_stride_bytes,
    long scale_off_bytes, int drain, int acq, unsigned int nbw, R4dArGather G, int rows_max,
    long long* gate, unsigned int* gate_ctr, long stage) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int lane = tid & (WAVE - 1), wid = tid / WAVE, nwarp = nt / WAVE;
  const int lig = lane & (LPG - 1), gw = lane / LPG;
  __shared__ unsigned int s_seq, s_slot;
  extern __shared__ int s_tab[];   // slots, weights: [rows_max * top_k]; gates: [rows_max]
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  Wht6Slot w = wht6_slot(peer_scratch, my_scratch, loc_pack, (int)s_slot,
                         slot_stride_bytes, scale_off_bytes);
  const long md = r4d_archk_md(stage, my_scratch);   // the host mailbox's VRAM mirror

  const int cpb = (n_chunks + nb - 1) / nb;                 // block b owns chunks [c0,c1)
  const int c0 = b * cpb; int c1 = c0 + cpb; if (c1 > n_chunks) c1 = n_chunks;

  /* Slot table, as in the exact form: a slot no sorted row claims reads -1, so clear and scan are
   * separated by a barrier. */
  int* s_i = s_tab;
  float* s_w = reinterpret_cast<float*>(s_tab + rows_max * G.top_k);
  float* s_g = s_w + rows_max * G.top_k;
  int row_lo = 0, nrow = 0;
  if (GATHER) {
    wht6_block_rows(c0, c1, Wht6Chunk<CG>::ELEMS, n_groups, n, row_lo, nrow);
    const int K = G.top_k;
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

  const auto ld = [&](long off, float* v) {
    int4 q;
    if (GATHER) {
      const int row = (int)(off / n), i0 = (int)(off - (long)row * n);
      q = r4d_ar_gather_word<false>(G, nullptr, s_i, s_w, s_g, row, row - row_lo, i0);
    } else {
      q = *reinterpret_cast<const int4*>(y + off);
    }
    const unsigned short* e = reinterpret_cast<const unsigned short*>(&q);
#pragma unroll
    for (int i = 0; i < EPL; ++i) v[i] = bf2f(e[i]);
  };
  const auto st = [&](long off, const float* v) {
    int4 q;
    __hip_bfloat16* e = reinterpret_cast<__hip_bfloat16*>(&q);
#pragma unroll
    for (int i = 0; i < EPL; ++i) rd_from_f(v[i], e[i]);
    *reinterpret_cast<int4*>(y + off) = q;
    const int row = (int)(off / n), i0 = (int)(off - (long)row * n);
    r4d_ar_hc_write_word(inj, h, row, i0, n, hc, irow, hrow,
                         reinterpret_cast<const unsigned short*>(&q));
  };
  unsigned int hck = 0u;   // the push check (r4d_ar_seq.h)
  for (int c = c0 + wid; c < c1; c += nwarp)
    wht6_push_chunk<CG>(ld, c, n_groups, lane, lig, gw, w.p_pay, w.p_sc, w.l_pay, w.l_sc, hck);
  r4d_archk_put(hck, peer_flags, nbw, s_slot, (unsigned)b, s);

  wht6_handshake(peer_flags, my_flags, b, s, tid, drain, acq);
  r4d_archk_wait(my_flags, nbw, s_slot, (unsigned)b, s, [&](unsigned int& h) {
    for (int c = c0 + wid; c < c1; c += nwarp)
      wht6_rehash_chunk<CG>(h, c, n_groups, lane, lig, gw, w.m_pay, w.m_sc, md);
  });
  w.moved(md);
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  for (int c = c0 + wid; c < c1; c += nwarp)
    wht6_reduce_chunk<CG>(st, c, n_groups, lane, lig, gw, w.m_pay, w.m_sc, w.l_pay, w.l_sc);
}

int r4d_ar_hc_write_wht6(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                         long seq_ctrs, long loc_pack, long slot_stride_bytes,
                         long scale_off_bytes, long y, long inj, long h, long M, long n, long hc,
                         long irow, long hrow, long ye, long ew, long sorted, long sh, long sg,
                         long ye_ld, long ew_ld, long sh_ld, long sg_ld, long top_k, long sig,
                         long stream_i, long nblocks, long nthreads, long drain, long acq,
                         long gate, long gate_ctr, long stage) {
  hipStream_t st = (hipStream_t)stream_i;
  if (n <= 0 || M <= 0 || n % 8 || hc < 1 || hrow % 8) return -1;
  if ((M * n) % GROUP) return -1;
  const bool gather = ye != 0;
  if (gather && (!ew || !sorted || top_k < 1 || top_k > R4D_ARG_SLOTS || ye_ld % 8 ||
                 (sh && sh_ld < n)))
    return -2;
  const int n_groups = (int)(M * n / GROUP);
  const int ce = Wht6Chunk<kWht6FusedCG>::ELEMS;
  const int n_chunks = (n_groups + kWht6FusedCG - 1) / kWht6FusedCG;
  int nb = clamp_blocks((int)nblocks, n_chunks);
  const int nt = clamp_threads((int)nthreads);
  /* Widen the grid until the table fits: a block of cpb chunks touches at most
   * (cpb * ce - 1) / n + 2 rows. Both ranks compute the same grid from the same M and n, as the
   * per-block handshake requires. */
  int rows_max = 0;
  size_t lds = 0;
  if (gather) {
    for (;;) {
      const long cpb = (n_chunks + nb - 1) / nb;
      rows_max = (int)((cpb * ce - 1) / n + 2);
      lds = (size_t)rows_max * (size_t)top_k * 8 + (size_t)rows_max * 4;
      if (lds <= kWht6TableBytes || nb >= n_chunks) break;
      ++nb;
    }
    if (lds > kWht6TableBytes || nb > R4D_AR_MAX_BLOCKS) return -3;
  }
  /* Zeroed including padding, as in r4d_ar_gather_hc_write: the tail padding is part of the
   * kernel-argument bytes a recorded pass's audit compares. */
  R4dArGather G;
  std::memset(&G, 0, sizeof G);
  if (gather) {
    G.ye = (const unsigned short*)ye;
    G.ew = (const unsigned short*)ew;
    G.sorted = (const int*)sorted;
    G.sh = (const unsigned short*)sh;
    G.sg = (const unsigned short*)sg;
    G.ye_ld = ye_ld; G.ew_ld = ew_ld; G.sh_ld = sh_ld; G.sg_ld = sg_ld;
    G.top_k = (int)top_k; G.T = (int)(M * top_k); G.sig = (int)sig;
  }
#define LF(GA, CG) r4d_ar_hc_write_wht6_kernel<GA, CG><<<nb, nt, lds, st>>>( \
    (unsigned short*)y, (char*)peer_scratch, (const char*)my_scratch, (char*)loc_pack, \
    (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, \
    (unsigned short*)h, (int)n, (int)hc, irow, hrow, n_groups, n_chunks, slot_stride_bytes, \
    scale_off_bytes, (int)drain, (int)acq, (unsigned)R4D_AR_MAX_BLOCKS, G, rows_max, \
    (long long*)gate, (unsigned int*)gate_ctr, stage)
  if (gather) LF(true, kWht6FusedCG);
  else LF(false, kWht6FusedCG);
#undef LF
  return (int)hipGetLastError();
}
