// r4d_ar_twoshot_Nrank_ti8.hip: r4d_ar_twoshot_{3,4}rank_ti8 -- reduce-scatter + all-gather with
// the tiered-int8 wire on both hops, at 3 or 4 ranks. Codec (tiers 1/2/3) from r4d_ar_ti8.h,
// transport from r4d_ar_twoshot_Nrank_exact.hip. The second hop costs 38/64 of its exact bytes.
// The own contribution is read exact from the input at fold time.
//
// Cross-rank identity: the phase-2 encode input is the deterministic fp32 fold, and the owner
// stores the decode of the words it published, never the exact fold, so every rank holds the same
// bytes for every shard.
//
// Slot: region A = ti8 wire (38*sg bytes, codes | rec | scales), region B at align16(A end)
// a second wire image of the same layout; the stride must hold both.
//
// Uneven shards, in whole groups (the exact two-shot's rule applied to groups): ng = numel/32
// groups split ng/ws per shard and the first ng % ws take one more, so shard s starts at group
// s*base + min(s, rem). Every slot is laid out for the longest shard (sg_max), so writer and reader
// agree on sg_max alone. At 4 ranks the entry requires numel to tile 32 x 4, so the split is even;
// at 3, a 2048-wide row is 64 groups.
#include "r4d.h"
#include "r4d_ar_ti8.h"
#include "r4d_ar_wide.h"
#include <stdexcept>
#include <string>

template <typename T, int WS>
__global__ void r4d_ar_twoshot_wide_ti8_kernel(
    const T* __restrict__ in, R4DArWidePeers<char, WS - 1> peers, char* __restrict__ my_base,
    unsigned int* __restrict__ my_flags, T* __restrict__ out,
    unsigned int* __restrict__ seq_ctrs, int n_groups, long slot_stride_bytes,
    long b_off_bytes, int rank, int drain, int acq, int pub,
    long long* gate, unsigned int* gate_ctr) {
  constexpr int NPEERS = WS - 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);  // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;

  // shard s = groups [s*gbase + min(s, grem), +gbase + (s < grem)); sg is the longest
  const int gbase = n_groups / WS, grem = n_groups % WS;
  const int sg = gbase + (grem ? 1 : 0);
#define R4D_TI8_OFF(s) ((size_t)((s) * gbase + ((s) < grem ? (s) : grem)) * TI8_GROUP)
#define R4D_TI8_LEN(s) (gbase + ((s) < grem ? 1 : 0))
  const int rec_off = TI8_REC_OFF * sg, sc_off = TI8_SC_OFF * sg;  // wire layout: codes | rec | scales
  const int gpb = (sg + nb - 1) / nb;      // block b owns shard groups [gA, gB)
  const int gA = b * gpb;
  const int gB = (gA + gpb < sg) ? gA + gpb : sg;

  char* pslot[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    pslot[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride_bytes;

  // ---- phase 1: encode my contribution to each peer-owned shard into its owner's region A. My
  // own shard's contribution stays exact, read from the input at fold time.
  for (int p = 0; p < NPEERS; ++p) {
    const T* src = in + R4D_TI8_OFF(peers.rank[p]);
    const int gE = gB < R4D_TI8_LEN(peers.rank[p]) ? gB : R4D_TI8_LEN(peers.rank[p]);
    for (int k = 4 * gA + tid; k < 4 * gE; k += nt) {
      const int lig = k & 3;
      const int g = k >> 2;
      uint2 codes;
      unsigned int rec;
      __half scale;
      ti8_encode_group<T>(src + k * TI8_EPL, lig, &codes, &rec, &scale);
      ((uint2*)pslot[p])[k] = codes;
      if (lig == 0) {
        ((unsigned int*)(pslot[p] + rec_off))[g] = rec;
        ((__half*)(pslot[p] + sc_off))[g] = scale;
      }
    }
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s - 1u, drain, acq, pub);

  // ---- reduce my shard: own contribution exact, peers decoded from region A, fp32 fold in
  // ascending rank order. The reduced group is re-encoded to every peer's region B and the owner
  // keeps its decode.
  const T* my_in = in + R4D_TI8_OFF(rank);
  T* my_out = out + R4D_TI8_OFF(rank);
  const int gM = gB < R4D_TI8_LEN(rank) ? gB : R4D_TI8_LEN(rank);
  for (int k = 4 * gA + tid; k < 4 * gM; k += nt) {
    const int lig = k & 3;
    const int g = k >> 2;
    float acc[TI8_EPL];
#pragma unroll
    for (int i = 0; i < TI8_EPL; ++i) acc[i] = 0.f;
    for (int r = 0; r < WS; ++r) {  // ascending source order, own exact from the input
      float v[TI8_EPL];
      if (r == rank) {
        ti8_load8(my_in + k * TI8_EPL, v);
      } else {
        const char* base =
            my_base + ((size_t)parity * WS + r) * (size_t)slot_stride_bytes;
        const uint2 c = ((const uint2*)base)[k];
        const unsigned int rc = ((const unsigned int*)(base + rec_off))[g];
        const float scl = __half2float(((const __half*)(base + sc_off))[g]);
        ti8_decode_group<T>(c, rc, scl, lig, v);
      }
#pragma unroll
      for (int i = 0; i < TI8_EPL; ++i) acc[i] += v[i];
    }
    uint2 codes;
    unsigned int rc2;
    __half scl2;
    ti8_encode_group_v<T>(acc, lig, &codes, &rc2, &scl2);
#pragma unroll
    for (int p = 0; p < NPEERS; ++p) {
      char* bB = pslot[p] + b_off_bytes;
      ((uint2*)bB)[k] = codes;
      if (lig == 0) {
        ((unsigned int*)(bB + rec_off))[g] = rc2;
        ((__half*)(bB + sc_off))[g] = scl2;
      }
    }
    float v2[TI8_EPL];
    ti8_decode_group<T>(codes, rc2, __half2float(scl2), lig, v2);
    ti8_store8(my_out + k * TI8_EPL, v2);
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
  /* As in the exact two-shot, the hold ends after the second handshake: the reduce above is fused
   * with the phase-2 wire push. Only the local decode-assemble is left (r4d_ar_seq.h). */
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  // ---- assemble: decode each peer-owned shard from region B, matching the owner's roundtrip
  // store.
  for (int p = 0; p < NPEERS; ++p) {
    const int owner = peers.rank[p];
    const char* baseB = my_base +
        ((size_t)parity * WS + owner) * (size_t)slot_stride_bytes + b_off_bytes;
    T* dst = out + R4D_TI8_OFF(owner);
    const int gE = gB < R4D_TI8_LEN(owner) ? gB : R4D_TI8_LEN(owner);
    for (int k = 4 * gA + tid; k < 4 * gE; k += nt) {
      const int lig = k & 3;
      const int g = k >> 2;
      const uint2 c = ((const uint2*)baseB)[k];
      const unsigned int rc2 = ((const unsigned int*)(baseB + rec_off))[g];
      const float scl = __half2float(((const __half*)(baseB + sc_off))[g]);
      float v[TI8_EPL];
      ti8_decode_group<T>(c, rc2, scl, lig, v);
      ti8_store8(dst + k * TI8_EPL, v);
    }
  }
#undef R4D_TI8_OFF
#undef R4D_TI8_LEN
}

// ---------------------------------------------------------------------------------- host side
template <int WS>
static void ar_twoshot_ti8_launch(const long* peer_scratch, const long* peer_flags,
                                  long my_scratch, long my_flags, long seq_ctrs,
                                  long slot_stride_bytes, long inp, long out, long n_elem,
                                  long dtype, long rank, long stream_i, long nblocks,
                                  long nthreads, long drain, long acq, long pub, long gate,
                                  long gate_ctr) {
  constexpr int NPEERS = WS - 1;
  if (rank < 0 || rank >= WS)
    throw std::runtime_error("ti8 ar: rank " + std::to_string(rank) +
                             " out of range for the " + std::to_string(WS) +
                             " ranks this entry point is compiled for");
  if (dtype != 0 && dtype != 1) throw std::runtime_error("ti8 ar: only bf16/fp16 payloads");
  if (n_elem % TI8_GROUP != 0)
    throw std::runtime_error("ti8 ar: n_elem must be whole " + std::to_string(TI8_GROUP) +
                             "-elem groups");
  const long n_groups = n_elem / TI8_GROUP;
  if (n_groups < WS)
    throw std::runtime_error("ti8 ar: fewer groups than ranks");
  const int sg = (int)((n_groups + WS - 1) / WS);  // the longest shard, the slot layout's size
  const long b_off = ((long)TI8_GROUP_BYTES * sg + 15) / 16 * 16;  // one wire image, 16B-aligned
  if (2 * b_off > slot_stride_bytes)
    throw std::runtime_error("ti8 ar: slot stride " + std::to_string(slot_stride_bytes) +
                             " too small for two wire images (needs " +
                             std::to_string(2 * b_off) + " bytes)");

  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > sg) nb = sg;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  // A wire group is TI8_LPG adjacent lanes (codec shuffles use masks below TI8_LPG), so it must
  // land on lanes 4j..4j+3. The kernel walks k = 4*gA + tid in steps of nt, which keeps
  // k%TI8_LPG == lane%TI8_LPG only while TI8_LPG divides nt; otherwise the results are silently
  // wrong, so refuse rather than clamp.
  if (nt % TI8_LPG)
    throw std::runtime_error("ti8 ar: nthreads must be a multiple of " +
                             std::to_string(TI8_LPG) + " (a wire group is that many "
                             "adjacent lanes); got " + std::to_string(nt));

  R4DArWidePeers<char, NPEERS> peers{};
  for (int i = 0; i < NPEERS; ++i) {
    if (!peer_scratch[i] || !peer_flags[i])
      throw std::runtime_error("ti8 ar: null peer scratch/flag pointer at index " +
                               std::to_string(i));
    peers.scratch[i] = (char*)peer_scratch[i];
    peers.flags[i] = (unsigned int*)peer_flags[i];
    peers.rank[i] = (i < (int)rank) ? i : i + 1;
  }

  // Checked against the allocations: scratch backs 2*WS slots of the stride, flags one word per
  // (block, source rank), seq a handshake and a parity word per block of capacity (r4d_ar_seq.h).
  const long slot_bytes = 2L * WS * slot_stride_bytes;
  const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
  r4d_ar_check_extent(my_scratch, slot_bytes, "ti8 ar", "my_scratch");
  r4d_ar_check_extent(my_flags, flag_bytes, "ti8 ar", "my_flags");
  r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                      "ti8 ar", "seq_ctrs");
  for (int i = 0; i < NPEERS; ++i) {
    r4d_ar_check_extent(peer_scratch[i], slot_bytes, "ti8 ar", "peer scratch");
    r4d_ar_check_extent(peer_flags[i], flag_bytes, "ti8 ar", "peer flags");
  }

  hipStream_t st = (hipStream_t)stream_i;
#define R4D_AR_TS8_LAUNCH(T)                                                            \
  r4d_ar_twoshot_wide_ti8_kernel<T, WS><<<nb, nt, 0, st>>>(                             \
      (const T*)inp, peers, (char*)my_scratch, (unsigned int*)my_flags, (T*)out,        \
      (unsigned int*)seq_ctrs, (int)n_groups, (long)slot_stride_bytes, b_off, (int)rank, \
      (int)drain, (int)acq, (int)pub, (long long*)gate, (unsigned int*)gate_ctr)
  if (dtype == 0) R4D_AR_TS8_LAUNCH(__hip_bfloat16);
  else R4D_AR_TS8_LAUNCH(__half);
#undef R4D_AR_TS8_LAUNCH
}

void r4d_ar_twoshot_3rank_ti8(long peer_scratch0, long peer_scratch1, long peer_flags0,
                              long peer_flags1, long my_scratch, long my_flags, long seq_ctrs,
                              long slot_stride_bytes, long inp, long out, long n_elem,
                              long dtype, long rank, long stream, long nblocks,
                              long nthreads, long drain, long acq, long pub) {
  const long ps[2] = {peer_scratch0, peer_scratch1};
  const long pf[2] = {peer_flags0, peer_flags1};
  /* No link gate at three ranks: the row answers 1, the engine releases by barrier. */
  ar_twoshot_ti8_launch<3>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride_bytes, inp,
                           out, n_elem, dtype, rank, stream, nblocks, nthreads, drain,
                           acq, pub, 0L, 0L);
}

void r4d_ar_twoshot_4rank_ti8(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                              long peer_flags0, long peer_flags1, long peer_flags2,
                              long my_scratch, long my_flags, long seq_ctrs,
                              long slot_stride_bytes, long inp, long out, long n_elem,
                              long dtype, long rank, long stream, long nblocks,
                              long nthreads, long drain, long acq, long pub, long gate,
                              long gate_ctr) {
  const long ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const long pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  ar_twoshot_ti8_launch<4>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride_bytes, inp,
                           out, n_elem, dtype, rank, stream, nblocks, nthreads, drain,
                           acq, pub, gate, gate_ctr);
}

// Tiered-int8 wire format dimensions, for sizing the caller's packed scratch.
void r4d_ar_ti8_dims(int* group, int* group_bytes) {
  if (group)       *group       = TI8_GROUP;
  if (group_bytes) *group_bytes = TI8_GROUP_BYTES;
}
