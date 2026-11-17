// r4d_ar_gather_Nrank.hip: r4d_ar_gather_3rank, r4d_ar_gather_4rank, r4d_ar_gather_8rank -- one-shot P2P all-GATHER,
// 3, 4 or 8 ranks, gfx1201. WS is compile-time; the 2-rank gather has its own file, as does the
// 2-rank one-shot all-reduce.
//
// r4d_ar_gather_2rank.hip handshakes with one peer flag per block, so it serves two ranks only. The
// vocab-parallel sampler declares an all_gather whenever the vocabulary is sharded; without this
// kernel only libref's host ref_all_gather resolves at wider worlds and the serve is refused. It is
// the wide one-shot all-reduce with the reduction removed: same push, per-block handshake and
// double-buffered (parity, source) scratch, with the fp32 sum replaced by placement.
//
// The payload is (u32 token id, f32 logit) pairs, so it is moved and never summed. `dtype` only
// converts n_elem to bytes.
//
// Placement: row16 is one row of this rank's contribution in 16-byte words. r4d_g_dst_ws
// concatenates when there is one row, and otherwise interleaves rows (input [rows][row16], output
// [rows][WS][row16]) so the merge reads each position's candidates at a single row stride (see
// r4d_ar_gather_2rank.hip).
//
// The IPC helpers and the handshake live in r4d_ar_wide.h; r4d_ar_entry.hip owns the instance and
// the peer arrays.
#include "r4d.h"
#include "r4d_ar_wide.h"
#include "r4d_chain.h"
#include <cstdint>
#include <stdexcept>
#include <string>

// Peer arguments are in ascending global rank with the current rank removed; `out` is
// world_size * n_elem elements. Declared extern "C" here so the definitions below get C linkage;
// r4d_ar_entry.hip repeats the prototypes at its call site, since r4d.h does not declare them.
extern "C" void r4d_ar_gather_3rank(long peer_scratch0, long peer_scratch1,
                                    long peer_flags0, long peer_flags1,
                                    long my_scratch, long my_flags, long seq_ctrs,
                                    long slot_stride16, long inp, long out, long n_elem,
                                    long dtype, long rank, long row_elems, long stream,
                                    long nblocks, long nthreads, long drain, long acq, long pub,
                                    long dpitch, long spitch);
extern "C" void r4d_ar_gather_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                    long peer_flags0, long peer_flags1, long peer_flags2,
                                    long my_scratch, long my_flags, long seq_ctrs,
                                    long slot_stride16, long inp, long out, long n_elem,
                                    long dtype, long rank, long row_elems, long stream,
                                    long nblocks, long nthreads, long drain, long acq, long pub,
                                    long gate, long gate_ctr, long dpitch, long spitch, long chain);
extern "C" void r4d_ar_gather_8rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                    long peer_scratch3, long peer_scratch4, long peer_scratch5,
                                    long peer_scratch6, long peer_flags0, long peer_flags1,
                                    long peer_flags2, long peer_flags3, long peer_flags4,
                                    long peer_flags5, long peer_flags6, long my_scratch,
                                    long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                    long out, long n_elem, long dtype, long rank, long row_elems,
                                    long stream, long nblocks, long nthreads, long drain, long acq,
                                    long pub, long dpitch, long spitch);

// Where a rank's words land: the 2-rank r4d_g_dst generalised to WS. Named differently because two
// __device__ definitions of one name in the two units would violate the ODR.
// `dp16` is the output's row pitch in words, WS * row16 when the gathered rows are tight.
__device__ __forceinline__ int r4d_g_dst_ws(int k, int row16, int n16, int who, int dp16) {
  if (row16 <= 0 || row16 >= n16) return who * n16 + k;
  const int m = k / row16, c = k - m * row16;
  return m * dp16 + who * row16 + c;
}

// Where word k of this rank's contribution is read: `sp16` is the input's row pitch in words,
// row16 when its rows are tight, more when they are a window of a wider row.
__device__ __forceinline__ int r4d_g_src_ws(int k, int row16, int n16, int sp16) {
  if (row16 <= 0 || row16 >= n16 || sp16 == row16) return k;
  const int m = k / row16, c = k - m * row16;
  return m * sp16 + c;
}

// Block b owns words [start,end) of the 16B-word space and drives its own handshake; no cross-block
// sync, as in the all-reduce.
// The body; the plain kernel and its chained form (r4d_chain.h) wrap it below. The chained form
// waits before the sequence take, so a block of the next launch on the instance never starts
// ahead of this one's; no block returns early.
template <int WS>
__device__ __forceinline__ void r4d_ar_gather_wide_body(
    const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers,
    const int4* __restrict__ my_base, int4* __restrict__ out4,
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ seq_ctrs,
    int n16, int slot_stride16, int rank, int row16, int dp16, int sp16, int drain, int acq,
    int pub, long long* gate, unsigned int* gate_ctr) {
  constexpr int NPEERS = WS - 1;
  // Placement holds WS source words live per output word, so the 2x unroll spills at ws=8 (same
  // limit as the one-shot all-reduce).
  constexpr int RU = (WS <= 4) ? 2 : 1;

  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);  // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;

  const int w = (n16 + nb - 1) / nb;
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  // Every peer lays its scratch out identically, with one slot per source rank. Our words go into
  // the slot numbered by our rank in every peer's scratch.
  int4* ps[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    ps[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;

  // Push, 4x unrolled; loaded words are reused across peers so the input is read once. This rank's
  // own slice is placed after the handshake with all the others, from one src[] table.
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      const int4 v0 = in4[r4d_g_src_ws(k, row16, n16, sp16)];
      const int4 v1 = in4[r4d_g_src_ws(k + nt, row16, n16, sp16)];
      const int4 v2 = in4[r4d_g_src_ws(k + 2 * nt, row16, n16, sp16)];
      const int4 v3 = in4[r4d_g_src_ws(k + 3 * nt, row16, n16, sp16)];
#pragma unroll
      for (int p = 0; p < NPEERS; ++p) {
        ps[p][k] = v0; ps[p][k + nt] = v1; ps[p][k + 2 * nt] = v2; ps[p][k + 3 * nt] = v3;
      }
    }
    for (; k < end; k += nt) {
      const int4 v = in4[r4d_g_src_ws(k, row16, n16, sp16)];
#pragma unroll
      for (int p = 0; p < NPEERS; ++p) ps[p][k] = v;
    }
  }

  // Publish 2*seq to every peer's flag word [b][our rank], then wait on our own [b][source] for
  // every source. The protocol is r4d_ar_wide.h's, shared with the all-reduces on these buffers; a
  // handshake that cannot complete traps rather than pass on stale bytes.
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
  // One handshake, and the placement after it reads this card's scratch and input only: the last
  // block here ends the engine's hold, as in the one-shot all-reduce this rides beside.
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  // Place every rank's slice. Slot [parity][our rank] of our own scratch is never written, so, as
  // in the all-reduce, our rank's source is the input itself, read at its own pitch.
  const int4* src[WS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    src[r] = (r == rank) ? in4
                         : (my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16);
  {
    int k = start + tid;
    if (RU == 2) {
      for (; k + nt < end; k += 2 * nt) {
        const int k0 = r4d_g_src_ws(k, row16, n16, sp16), k1 = r4d_g_src_ws(k + nt, row16, n16, sp16);
#pragma unroll
        for (int r = 0; r < WS; ++r) {
          out4[r4d_g_dst_ws(k,      row16, n16, r, dp16)] = src[r][r == rank ? k0 : k];
          out4[r4d_g_dst_ws(k + nt, row16, n16, r, dp16)] = src[r][r == rank ? k1 : k + nt];
        }
      }
    }
    for (; k < end; k += nt) {
      const int k0 = r4d_g_src_ws(k, row16, n16, sp16);
#pragma unroll
      for (int r = 0; r < WS; ++r)
        out4[r4d_g_dst_ws(k, row16, n16, r, dp16)] = src[r][r == rank ? k0 : k];
    }
  }
}
template <int WS>
__global__ void r4d_ar_gather_wide_kernel(
    const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers,
    const int4* __restrict__ my_base, int4* __restrict__ out4,
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ seq_ctrs,
    int n16, int slot_stride16, int rank, int row16, int dp16, int sp16, int drain, int acq,
    int pub, long long* gate, unsigned int* gate_ctr) {
    r4d_ar_gather_wide_body<WS>(in4, peers, my_base, out4, my_flags, seq_ctrs, n16, slot_stride16,
                                rank, row16, dp16, sp16, drain, acq, pub, gate, gate_ctr);
}
template <int WS>
__global__ void r4dchain_ar_gather_wide(
    const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers,
    const int4* __restrict__ my_base, int4* __restrict__ out4,
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ seq_ctrs,
    int n16, int slot_stride16, int rank, int row16, int dp16, int sp16, int drain, int acq,
    int pub, long long* gate, unsigned int* gate_ctr) {
    r4d_chain_wait();
    r4d_ar_gather_wide_body<WS>(in4, peers, my_base, out4, my_flags, seq_ctrs, n16, slot_stride16,
                                rank, row16, dp16, sp16, drain, acq, pub, gate, gate_ctr);
    r4d_chain_done();
}

// --- host side ---
// `row_elems` is the row width in elements, 0 for plain concatenation. The entry point checks it
// already; it is rechecked here because this is the half that throws.
template <int WS>
static void ar_gather_wide_launch(const long* peer_scratch, const long* peer_flags, long my_scratch,
                                  long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                  long out, long n_elem, long dtype, long rank, long row_elems,
                                  long stream_i, long nblocks, long nthreads, long drain, long acq,
                                  long pub, long gate, long gate_ctr, long dpitch, long spitch,
                                  bool chain = false) {
  constexpr int NPEERS = WS - 1;
  static_assert(NPEERS <= R4D_AR_WIDE_MAX_PEERS, "npeers exceeds the peer bundle bound");

  if (rank < 0 || rank >= WS)
    throw std::runtime_error("all_gather: rank " + std::to_string(rank) +
                             " out of range for the " + std::to_string(WS) +
                             " ranks this entry point is compiled for");

  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_gather: n_elem*esize not 16B-aligned");
  const long n16l = nbytes / 16;
  if (n16l > (long)slot_stride16)
    throw std::runtime_error("all_gather: message (" + std::to_string(n16l) +
                             " 16B words) exceeds the per-source slot stride (" +
                             std::to_string(slot_stride16) + ")");
  const int n16 = (int)n16l;

  long rowb = (long)row_elems * esize;
  if (row_elems > 0 && (rowb % 16 != 0 || nbytes % rowb != 0))
    throw std::runtime_error("all_gather: row is not a whole number of 16B words");
  const int row16 = row_elems > 0 ? (int)(rowb / 16) : 0;
  // The output's row pitch in elements, 0 for a tight [rows][WS][row]: whole words, at least the
  // WS ranks' row (the gathered rows as the leading columns of a wider buffer).
  const long dpb = (long)dpitch * esize;
  if (dpitch > 0 && (row16 <= 0 || dpb % 16 != 0 || dpitch < (long)WS * row_elems))
    throw std::runtime_error("all_gather: the output pitch is not a whole number of 16B words "
                             "at least " + std::to_string(WS) + " rows wide");
  const int dp16 = dpitch > 0 ? (int)(dpb / 16) : WS * row16;
  // The input's row pitch in elements, 0 for tight rows: whole words, at least a row (a rank's
  // window of a wider row).
  const long spb = (long)spitch * esize;
  if (spitch > 0 && (row16 <= 0 || spb % 16 != 0 || spitch < row_elems))
    throw std::runtime_error("all_gather: the input pitch is not a whole number of 16B words "
                             "at least a row wide");
  const int sp16 = spitch > 0 ? (int)(spb / 16) : row16;

  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;

  R4DArWidePeers<int4, NPEERS> peers{};
  for (int i = 0; i < NPEERS; ++i) {
    if (!peer_scratch[i] || !peer_flags[i])
      throw std::runtime_error("all_gather: null peer scratch/flag pointer at index " +
                               std::to_string(i));
    peers.scratch[i] = (int4*)peer_scratch[i];
    peers.flags[i] = (unsigned int*)peer_flags[i];
    peers.rank[i] = (i < (int)rank) ? i : i + 1;
  }

  // As in the all-reduce sharing this instance, check against the allocations: scratch backs 2*WS
  // slots of the stride, flags one word per (block, source rank), seq a handshake and a parity
  // word per block of capacity.
  const long slot_bytes = 2L * WS * (long)slot_stride16 * 16;
  const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
  r4d_ar_check_extent(my_scratch, slot_bytes, "all_gather", "my_scratch");
  r4d_ar_check_extent(my_flags, flag_bytes, "all_gather", "my_flags");
  r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                      "all_gather", "seq_ctrs");
  for (int i = 0; i < NPEERS; ++i) {
    r4d_ar_check_extent(peer_scratch[i], slot_bytes, "all_gather", "peer scratch");
    r4d_ar_check_extent(peer_flags[i], flag_bytes, "all_gather", "peer flags");
  }

  hipStream_t st = (hipStream_t)stream_i;
  (chain ? r4dchain_ar_gather_wide<WS> : r4d_ar_gather_wide_kernel<WS>)<<<nb, nt, 0, st>>>(
      (const int4*)inp, peers, (const int4*)my_scratch, (int4*)out, (unsigned int*)my_flags,
      (unsigned int*)seq_ctrs, n16, (int)slot_stride16, (int)rank, row16, dp16, sp16, (int)drain,
      (int)acq, (int)pub, (long long*)gate, (unsigned int*)gate_ctr);
}

/* Three ranks, e.g. Qwen3.8-Flash-Next padded to 36 heads / 3 KV heads (docs/TP3.md). */
void r4d_ar_gather_3rank(long peer_scratch0, long peer_scratch1,
                         long peer_flags0, long peer_flags1, long my_scratch,
                         long my_flags, long seq_ctrs, long slot_stride16, long inp, long out,
                         long n_elem, long dtype, long rank, long row_elems, long stream,
                         long nblocks, long nthreads, long drain, long acq, long pub, long dpitch,
                         long spitch) {
  const long ps[2] = {peer_scratch0, peer_scratch1};
  const long pf[2] = {peer_flags0, peer_flags1};
  /* No link gate at three ranks: the row answers 1, the engine releases by barrier. */
  ar_gather_wide_launch<3>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                           dtype, rank, row_elems, stream, nblocks, nthreads, drain, acq, pub,
                           0L, 0L, dpitch, spitch);
}

void r4d_ar_gather_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                         long peer_flags0, long peer_flags1, long peer_flags2, long my_scratch,
                         long my_flags, long seq_ctrs, long slot_stride16, long inp, long out,
                         long n_elem, long dtype, long rank, long row_elems, long stream,
                         long nblocks, long nthreads, long drain, long acq, long pub, long gate,
                         long gate_ctr, long dpitch, long spitch, long chain) {
  const long ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const long pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  ar_gather_wide_launch<4>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                           dtype, rank, row_elems, stream, nblocks, nthreads, drain, acq, pub,
                           gate, gate_ctr, dpitch, spitch, chain != 0);
}

void r4d_ar_gather_8rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                         long peer_scratch3, long peer_scratch4, long peer_scratch5,
                         long peer_scratch6, long peer_flags0, long peer_flags1, long peer_flags2,
                         long peer_flags3, long peer_flags4, long peer_flags5, long peer_flags6,
                         long my_scratch, long my_flags, long seq_ctrs, long slot_stride16,
                         long inp, long out, long n_elem, long dtype, long rank, long row_elems,
                         long stream, long nblocks, long nthreads, long drain, long acq,
                         long pub, long dpitch, long spitch) {
  const long ps[7] = {peer_scratch0, peer_scratch1, peer_scratch2, peer_scratch3,
                      peer_scratch4, peer_scratch5, peer_scratch6};
  const long pf[7] = {peer_flags0, peer_flags1, peer_flags2, peer_flags3,
                      peer_flags4, peer_flags5, peer_flags6};
  /* No link gate at eight ranks: the row answers 1, the engine releases by barrier. */
  ar_gather_wide_launch<8>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                           dtype, rank, row_elems, stream, nblocks, nthreads, drain, acq, pub,
                           0L, 0L, dpitch, spitch);
}
