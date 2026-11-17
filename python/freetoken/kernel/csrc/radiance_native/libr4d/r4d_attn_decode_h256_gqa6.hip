// R4D split-KV decode (also the MTP-verify shape). Same core as the prefill kernel (lazy
// reference-max softmax, f16 operands, folded query scale, contiguous-k fragments, prefetch +
// scheduling barrier, dot2 row sum), plus two decode-only options:
//
//   KLDS0  reads K straight from the paged cache (bf16 or fp8), widening once per warp instead of
//          once per workgroup. Costs VALU on a bandwidth-bound leg and frees sK, which at TILE 16
//          is 8.3 KB of the 14.5 KB that limits resident workgroups. Best at every depth.
//   PF16   split-KV partials stored as f16. At 16K context the partial round trip is ~14 MB
//          against ~17 MB of KV. Storing O/l instead of O bounds values by max|v|, so f16 cannot
//          overflow; the 11-bit mantissa costs ~5e-4 relative.
//
// No fp8-P leg and no non-CONTIG alternative here; r4d_attn_prefill_h256_gqa6.hip documents the
// full option set.
#include "r4d.h"
#include "r4d_common.h"
#include "r4d_dt16.h"
#include "r4d_q8.h"     // the gate quantiser's 8-bit codes: E4M3, or int8 (GQ 2 and 3)
#include "r4d_chain.h"
#include <algorithm>
#include <type_traits>

#define O_F16    1
#define O_MSKIP  2
#define O_FOLDQ  8
#define O_KLDS0 16
#define O_PF16  32
#define O_CONTIG 512
#define O_PF2   1024
#define O_PF4   2048
#define O_PF8   3072
#define O_SGB   4096
#define O_DOT2 16384
// Tile-staging options, shared with r4d_attn_prefill_h256_gqa6.hip:
//   BTS   the tile's block-table entries in SGPRs, loaded a tile ahead. With KLDS0 this also takes
//         the dependent bt[kk/BS] load out of the m-tile loop.
//   GPREV tile ti+1's V fetched into registers before the m-tile loop.
//   LDSB  an LDS-scoped barrier, so the barrier does not invalidate the vector cache.
#define O_BTS   262144
#define O_GPREV 1048576
#define O_LDSB  2097152
// Bit 4194304 is free: a non-temporal hint on the K load gained nothing on the target GPU (the KV
// stream does not evict the neighbouring GEMMs' activations).
// O_V8: an fp8 cache's V tile staged in LDS as codes (6 KiB instead of 12 KiB of f16), converted
// when a fragment is read. For a one-wave workgroup the f16 tile is what caps residency (ten
// workgroups per WGP). At one wave the conversion count is unchanged and the conversion is exact,
// so results are identical; with more waves each would convert the whole tile, so the launcher
// uses it only at one wave.
#define O_V8    8388608
// O_KRAW: with K read from the paged cache (KLDS0, CONTIG), issue all HEAD_DIM/16 fragment loads
// of an m-tile in one round and convert each fragment just before its WMMA. LLVM 24 honours the PF
// groups and scheduling barriers literally, so with PF 2 K arrives in HEAD_DIM/32 serial round
// trips (16-19% on the head_dim-128 drafter leg, bf16 cache). Raw cache bytes are held (8 per
// fragment for fp8, 16 for bf16), costing 2 or 4 x HEAD_DIM/16 VGPRs; PF = HEAD_DIM/16 would hold
// converted fragments and cost occupancy. Results are unchanged.
#define O_KRAW  16777216
// O_V4: the fp8_v4 cache (E4M3 K, int4 V in groups of 32 with an f16 scale), and with O_K6 the
// k6_v4 cache (int6 K, int4 V). No row instantiates either without O_I8.
#define O_V4    33554432
#define O_K6    67108864
// O_I8 (with O_V4 | O_K6, whose staging it takes): the i8 cache, int8 K and V with one f16 scale
// each per token and head (a RAD_I8 group, abi/rad_builder.h). The codes are a byte wide and a key has
// one scale where k6_v4 has one per 32 dims; they widen to value * scale the same way.
#define O_I8    1073741824
// The two views of R4DArgs (r4d.h), compiled in only where a row asks for them, since the h256
// instances sit at the 256-VGPR bound and a runtime pitch and head map spill there:
//   QLD    q rows at R4DArgs::q_ld (attn_paged_qrow), not dense.
//   KVREP  query group g reads cache head g / R4DArgs::kv_rep, descales per cache head.
// Neither changes the arithmetic.
#define O_QLD   134217728
#define O_KVREP 268435456
// TREE: a tree verify (attn_paged_qrow_tree; R4DArgs::spec_parent). A row whose entry is not -2
// sees, inside its sequence's query block, only itself and its ancestors, and its window is
// measured from the block's first position plus its depth. Compiled in only where a row asks.
#define O_TREE  536870912
__host__ __device__ __forceinline__ int r4d_attn_tiles(int ctx, int tile) { return (ctx + tile - 1) / tile; }
__host__ __device__ __forceinline__ int r4d_attn_tps(int ctx, int tile, int splits) {
    int t = r4d_attn_tiles(ctx, tile); return (t + splits - 1) / splits;
}
// Segments the combine reads for output row `tok`. The decode kernel cuts the tiles of the launch's
// context (the last row's) into segments of `tps` tiles. A causal row ends earlier than that: its
// own context is ctx - q_len + 1 + row, and the segments past it hold nothing it needs. Reading
// them anyway changes how the merge groups the live ones (the chains' tail), so a verify row
// would differ in the last bit from the single-row decode at its position, and greedy text with a
// drafter from the text without. `nef` is the launch's tile count after the window's floor, `win`
// whether this is a windowed row (kept as the launch's: its floor is row 0's). Same segments as
// before for a plain decode, a window, a non-causal row and a tree.
__device__ __forceinline__ int r4d_row_segments(const R4DArgs& a, int seq, int tok, int ctx, int tile,
                                                int tps, int nef, int win) {
    const int cz = a.causal_seq ? (int)((const unsigned char*)a.causal_seq)[seq] : a.causal;
    if (win || !cz || a.spec_parent) return (nef + tps - 1) / tps;
    const int qlen = r4d_q_len(a.cu_q, seq, a.q_len);
    const int row  = tok - r4d_q_off(a.cu_q, seq, a.q_len);
    const int rctx = ctx - qlen + 1 + row;
    const int tiles = rctx < ctx ? r4d_attn_tiles(rctx > 0 ? rctx : 1, tile) : nef;
    return (tiles + tps - 1) / tps;
}

// First KV position query row `qrow` may attend to under a sliding window: the row's
// causal limit, less the window, floored at zero. window <= 0 means no window.
__host__ __device__ __forceinline__ int r4d_attn_wlo(int ctx, int q_len, int qrow, int window) {
    if (window <= 0) return 0;
    const int lo = ctx - q_len + qrow - window + 1;
    return lo > 0 ? lo : 0;
}

// WIN enables a sliding window (a.window positions ending at each row's causal limit). As a
// template parameter defaulting to 0, full attention compiles with no window code.
// The body: its blocks return early (empty slots, tiles past the window), so the chained form
// (r4d_chain.h) wraps it below rather than bracketing it in place.
template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT,
         int WIN = 0>
__device__ __forceinline__
void r4d_attn_decode_body(const R4DArgs& a, int splits)
{
    constexpr int F16   = (OPT & O_F16)   ? 1 : 0;
    constexpr int MSKIP = (OPT & O_MSKIP) ? 1 : 0;
    constexpr int FOLDQ = (OPT & O_FOLDQ) ? 1 : 0;
    constexpr int KLDS  = (OPT & O_KLDS0) ? 0 : 1;
    constexpr int PF16  = (OPT & O_PF16)  ? 1 : 0;
    constexpr int CTG   = (OPT & O_CONTIG) ? 1 : 0;
    constexpr int PF    = 1 << ((OPT >> 10) & 3);
    constexpr int SGB   = (OPT & O_SGB) ? 1 : 0;
    constexpr int DOT2  = ((OPT & O_DOT2) && (OPT & O_F16)) ? 1 : 0;
    constexpr int BTS   = (OPT & O_BTS)  ? 1 : 0;
    constexpr int LDSB  = (OPT & O_LDSB) ? 1 : 0;
    constexpr int V4    = ((OPT & O_V4) && !KVP) ? 1 : 0;
    constexpr int K6    = ((OPT & O_K6) && V4) ? 1 : 0;
    constexpr int I8    = ((OPT & O_I8) && K6) ? 1 : 0;
    // A token's row in a page, and where in it V's codes, V's scales and K's scales start.
    constexpr int ROWB  = I8 ? 2 * HEAD_DIM + 4
                        : K6 ? HEAD_DIM * 22 / 16 : V4 ? HEAD_DIM * 25 / 16 : 2 * HEAD_DIM;
    constexpr int VOFF  = I8 ? HEAD_DIM : K6 ? HEAD_DIM * 3 / 4 : HEAD_DIM;
    constexpr int VSOFF = I8 ? 2 * HEAD_DIM + 2 : K6 ? HEAD_DIM * 21 / 16 : HEAD_DIM * 3 / 2;
    constexpr int KSOFF = I8 ? 2 * HEAD_DIM : HEAD_DIM * 5 / 4;
    constexpr int V8    = ((OPT & O_V8) && !KVP && !V4) ? 1 : 0;
    static_assert(!V8 || (OPT & O_CONTIG), "the code tile is read in the contiguous layout");
    constexpr int KRAW  = ((OPT & O_KRAW) && !KLDS) ? 1 : 0;
    static_assert(!KRAW || (OPT & O_CONTIG), "the raw K round reads the contiguous layout");
    static_assert(!K6 || (!KLDS && (OPT & O_CONTIG) && !KRAW),
                  "k6_v4 K is widened from the paged cache, a lane's eight dims contiguous");
    static_assert(!V4 || (OPT & O_F16), "fp8_v4 widens to f16");
    constexpr int QLD   = (OPT & O_QLD)   ? 1 : 0;
    constexpr int KVREP = (OPT & O_KVREP) ? 1 : 0;
    constexpr int TREE  = (OPT & O_TREE)  ? 1 : 0;
    typedef DT16<F16> D;
    typedef typename D::frag frag16;

    constexpr int NKS   = HEAD_DIM / 16;
    constexpr int MT    = TILE / 16;
    constexpr int KSTR  = HEAD_DIM + (CTG ? 8 : 4);
    constexpr int VSTR  = TILE + (CTG ? 8 : 4);
    constexpr int NTHR  = NWARPS * 32;
    constexpr int NB    = TILE / BS;
    constexpr int NTR   = (TILE / 8) * (HEAD_DIM / (KVP ? 32 : 64));
    constexpr int VREGS = (NTR + NWARPS - 1) / NWARPS;
    // A V prefetch costs VREGS x 4 VGPRs on top of a 128-VGPR accumulator and a 64-VGPR query.
    // Up to three uint4 it fits in 255 VGPRs without spilling (two: +6-8% at depth; three, the
    // 48-row shape q_len 8 x gqa 6 at NWARPS 3: a further 2-15%). At four or more (NWARPS 1-2, or a
    // bf16 cache) it spills and loses ~10%, so the geometry gates it, not just the flag.
    constexpr int GPREV = ((OPT & O_GPREV) && VREGS <= 3 && !V4) ? 1 : 0;
    constexpr float PSHIFT = D::SHIFT;
    constexpr float PGROW  = D::GROW;

    __shared__ uint16_t sK[KLDS ? TILE * KSTR : 1];
    // V4 at TILE 16: dim octet o's 8 rows start S(o) = 16 (o & 3) + 8 ((o >> 2) & 1) + 128 (o >> 2)
    // halfwords later, so the 4 (8) octets of a storeV4 store land on distinct banks (unskewed they
    // share them: 8 rows = 96 dwords = 0 mod 32). The 128 (o >> 2) term (64 dwords, unseen by the
    // banks) keeps octet o's last row clear of octet o + 1's first; multiples of 8 halfwords keep the
    // b128 PV read aligned. NWARPS 1 keeps the plain tile: its 10 workgroups a WGP would not fit.
    constexpr int VSK  = (V4 && TILE == 16 && CTG && NWARPS >= 2) ? 1 : 0;
    constexpr int VSKN = HEAD_DIM / 8 - 1;
    constexpr int VSKX = VSK ? 16 * (VSKN & 3) + 8 * ((VSKN >> 2) & 1) + 128 * (VSKN >> 2) : 0;
    __shared__ uint16_t sV16[V8 ? 1 : HEAD_DIM * VSTR + VSKX];
    /* A code-tile row is 16 keys + 8 padding = 24 bytes (six dwords): the sixteen lanes of a
     * fragment read hit distinct banks, and a lane's eight codes are one aligned 8-byte load. */
    constexpr int VSTR8 = TILE + 8;
    __shared__ uint8_t sV8[V8 ? HEAD_DIM * VSTR8 : 1];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // The launch is NTHR threads. Stated (k6_v4), so the V4 rounds wholly below NP4 lose their range
    // test: as exec-masked blocks, each round's loads issued only after the previous round's returned.
    // fp8_v4 keeps the tested rounds: batched there, its 4-row steps ran 6 % slower.
    if constexpr (K6) __builtin_assume(tid < NTHR);
    const int c = lane & 15, h = lane >> 4;

    const int sp = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z;
    // kvh is the query group; under KVREP the cache head it reads is kvh / kv_rep
    const int kvc = KVREP ? kvh / a.kv_rep : kvh;
    const int ctx = a.seqused_k[seq];
    if (ctx <= 0) return;

    // Ragged batches (r4d.h, R4DArgs::cu_q): this sequence's q/out rows and query length. With
    // cu_q, a.q_len is the batch maximum that sized the wave count; rows past qlen are not live.
    const int qoff = r4d_q_off(a.cu_q, seq, a.q_len);
    const int qlen = r4d_q_len(a.cu_q, seq, a.q_len);
    if (qlen <= 0) return;                 // a slot carrying no query rows this step

    const int ntl  = r4d_attn_tiles(ctx, TILE);
    // A sliding-window layer must not read below the window: those blocks may already be freed
    // and reassigned to another sequence. The walk starts at the tile holding row 0's first
    // window token (row 0 reaches furthest back), and the splits divide that range.
    const int wlo0 = WIN ? r4d_attn_wlo(ctx, qlen, 0, a.window) : 0;
    const int tb   = WIN ? wlo0 / TILE : 0;
    const int neff = ntl - tb;
    const int tps  = (neff + splits - 1) / splits;
    const int t_lo = tb + sp * tps;
    if (t_lo >= ntl) return;
    const int t_hi = min(t_lo + tps, ntl);

    // TREE: the block's parents staged once, so each row's walk below reads LDS rather than a
    // chain of dependent global loads. Every early return above is uniform across the block.
    __shared__ int s_tpar[TREE ? 64 : 1];
    if constexpr (TREE) {
        if (a.spec_parent) {
            for (int r = tid; r < qlen && r < 64; r += NTHR) s_tpar[r] = a.spec_parent[qoff + r];
            __syncthreads();
        }
    }

    const int row = warp * 16 + c;
    const int qi = row / GQA, hi = row - qi * GQA;
    const bool live = (qi < qlen);
    const int qrow = live ? qi : (qlen - 1);
    const int qhead = kvh * GQA + hi;

    const int* __restrict__ bt = a.block_table + (size_t)seq * a.max_blocks;
    const float kdesc = a.k_descale
                      ? a.k_descale[seq * (KVREP ? a.kv_heads / a.kv_rep : a.kv_heads) + kvc]
                      : 1.0f;
    const float mul   = a.scale * kdesc * 1.44269504089f;

    frag16 qf[NKS];
    {
        const uint16_t* qp = QLD ? (const uint16_t*)a.q + (size_t)(qoff + qrow) * a.q_ld
                                                        + (size_t)qhead * HEAD_DIM
                                 : (const uint16_t*)a.q
                                   + ((size_t)(qoff + qrow) * a.q_heads + qhead) * HEAD_DIM;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) {
            const uint2 l0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h : 4 * h));
            const uint2 h0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h + 4 : 8 + 4 * h));
            qf[t] = D::mk(make_uint2(r4d_qcvt<D, FOLDQ, F16>(l0.x, mul),
                                     r4d_qcvt<D, FOLDQ, F16>(l0.y, mul)),
                          make_uint2(r4d_qcvt<D, FOLDQ, F16>(h0.x, mul),
                                     r4d_qcvt<D, FOLDQ, F16>(h0.y, mul)));
        }
    }

    // The row's absolute position. Causal masking bounds the key by it; non-causal masking
    // bounds it by the sequence, with the window applied on both sides (a block-diffusion
    // drafter's own block).
    const int pos = ctx - qlen + qrow;
    const int cz  = a.causal_seq ? (int)((const unsigned char*)a.causal_seq)[seq] : a.causal;
    const int klimit = cz ? pos
                          : ((WIN && a.window > 0) ? min(ctx - 1, pos + a.window - 1)
                                                   : ctx - 1);
    // A tree verify's row (TREE): a bit per row of the block for itself and each ancestor, and
    // its depth, from which its window is measured. Parents precede children, so the walk ends.
    const int tbase = ctx - qlen;
    uint64_t tmask = 0;
    int tdep = 0;
    bool trow = false;
    if (TREE && a.spec_parent && qrow < 64) {
        int pj = s_tpar[qrow];
        if (pj != -2) {
            trow  = true;
            tmask = 1ull << qrow;
            for (int at = qrow; pj >= 0 && pj < at; ) {
                tmask |= 1ull << pj;
                ++tdep;
                at = pj;
                pj = s_tpar[pj];
            }
        }
    }
    const int wpos   = (TREE && trow) ? tbase + tdep : pos;
    const int wlo    = (WIN && a.window > 0) ? (wpos - a.window + 1 > 0
                                               ? wpos - a.window + 1 : 0) : 0;
    float m_ref = -3.0e38f;
    float off   = PSHIFT + 3.0e38f;
    float l_i   = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0,0,0,0,0,0,0,0};

    const int vg = lane >> 3, vj = lane & 7;
    uint4 vreg[VREGS];

    // ---- fp8_v4 / k6_v4: V's int4 codes into sV16 ([dim][key]) as the prefill kernel stages them
    // (r4d_attn_prefill_h256_gqa6.hip: a key pair's 8 dims a thread, two code words and the pair's
    // scale, widened with packed f16 into one dword a dim), fetched in the tile: no prefetch.
    constexpr int NP4 = (TILE / 2) * (HEAD_DIM / 8);
    constexpr int U4  = V4 ? (NP4 + NTHR - 1) / NTHR : 1;
    // The pair's two scales stay apart until storeV4 packs them: packed here, the fetch is waited on
    // right after it is issued. fp8_v4 packs here, as the prefill kernel does.
    constexpr int OLDV4 = V4 && !K6;
    // i8: a key's 8 dims are two code words, so a pair's are four.
    uint32_t v4c[U4][I8 ? 4 : 2], v4s[U4][2];
    auto fetchV4 = [&](int k0, const int* blks) {
        #pragma unroll
        for (int r = 0; r < U4; ++r) {
            const int u = tid + r * NTHR;
            // k6_v4, a partial round (NWARPS 3: the third, warps 0-1): u < NP4 <=> warp < (NP4 - r NTHR)
            // / 32 when NP4 is a multiple of 32, tested on the wave-uniform warp: a scalar branch.
            if (NP4 % NTHR == 0 || (K6 ? ((r + 1) * NTHR <= NP4
                || (NP4 % 32 == 0 ? __builtin_amdgcn_readfirstlane(warp) * 32 < NP4 - r * NTHR
                                  : u < NP4)) : u < NP4)) {
                const int kp = u % (TILE / 2), o = u / (TILE / 2);
                uint32_t sc[2];
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int kk = min(k0 + 2 * kp + j, ctx - 1);
                    if (WIN && kk < wlo0) kk = wlo0;
                    const int bk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                    const uint8_t* base = (const uint8_t*)a.kv + (size_t)bk * a.kv_block_stride
                                        + (size_t)kvc * a.kv_head_stride + (size_t)(kk % BS) * ROWB;
                    if constexpr (I8) {
                        const uint2 cw = *(const uint2*)(base + VOFF + o * 8);
                        v4c[r][2 * j]     = cw.x;
                        v4c[r][2 * j + 1] = cw.y;
                        sc[j]             = *(const uint16_t*)(base + VSOFF);
                    } else {
                        v4c[r][j] = *(const uint32_t*)(base + VOFF + o * 4);
                        sc[j]     = *(const uint16_t*)(base + VSOFF + (o >> 2) * 2);
                    }
                }
                if constexpr (OLDV4) { v4s[r][0] = sc[0] | (sc[1] << 16); v4s[r][1] = 0; }
                else                 { v4s[r][0] = sc[0]; v4s[r][1] = sc[1]; }
            }
        }
    };
    auto storeV4 = [&]() {
        typedef _Float16 h2 __attribute__((ext_vector_type(2)));
        #pragma unroll
        for (int r = 0; r < U4; ++r) {
            const int u = tid + r * NTHR;
            // k6_v4, a partial round (NWARPS 3: the third, warps 0-1): u < NP4 <=> warp < (NP4 - r NTHR)
            // / 32 when NP4 is a multiple of 32, tested on the wave-uniform warp: a scalar branch.
            if (NP4 % NTHR == 0 || (K6 ? ((r + 1) * NTHR <= NP4
                || (NP4 % 32 == 0 ? __builtin_amdgcn_readfirstlane(warp) * 32 < NP4 - r * NTHR
                                  : u < NP4)) : u < NP4)) {
                const int kp = u % (TILE / 2), o = u / (TILE / 2);
                // Two's complement nibbles; ^ 8 makes them offset binary (value + 8). i8: bytes,
                // ^ 0x80 (value + 128); `n` is a key's dims 0-3 of the octet and `m` its dims 4-7.
                const uint32_t n0 = I8 ? v4c[r][0] ^ 0x80808080u
                                       : (v4c[r][0] & 0x0F0F0F0Fu) ^ 0x08080808u;
                const uint32_t m0 = I8 ? v4c[r][1] ^ 0x80808080u
                                       : ((v4c[r][0] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
                const uint32_t n1 = I8 ? v4c[r][I8 ? 2 : 1] ^ 0x80808080u
                                       : (v4c[r][1] & 0x0F0F0F0Fu) ^ 0x08080808u;
                const uint32_t m1 = I8 ? v4c[r][I8 ? 3 : 1] ^ 0x80808080u
                                       : ((v4c[r][1] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
                const h2 s2 = __builtin_bit_cast(h2, OLDV4 ? v4s[r][0] : (v4s[r][0] | (v4s[r][1] << 16)));
                const h2 bias = { (_Float16)(I8 ? 1152.0f : 1032.0f), (_Float16)(I8 ? 1152.0f : 1032.0f) };
                uint16_t* dst = &sV16[(o * 8) * VSTR + 2 * kp
                                      + (VSK ? 16 * (o & 3) + 8 * ((o >> 2) & 1) + 128 * (o >> 2) : 0)];
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const uint32_t sel = (uint32_t)b | 0x0c00u | ((uint32_t)(4 + b) << 16) | 0x0c000000u;
                    const uint32_t lo = __builtin_amdgcn_perm(n1, n0, sel) | 0x64006400u;
                    const uint32_t hi = __builtin_amdgcn_perm(m1, m0, sel) | 0x64006400u;
                    const h2 vl = (__builtin_bit_cast(h2, lo) - bias) * s2;
                    const h2 vh = (__builtin_bit_cast(h2, hi) - bias) * s2;
                    // The octet's rows: nibbles interleave the dims (2b, 2b+1), bytes run b, 4+b.
                    *(uint32_t*)(dst + (I8 ? b : 2 * b) * VSTR)         = __builtin_bit_cast(uint32_t, vl);
                    *(uint32_t*)(dst + (I8 ? 4 + b : 2 * b + 1) * VSTR) = __builtin_bit_cast(uint32_t, vh);
                }
            }
        }
    };

    auto fetchV = [&](int k0, const int* blks) {
        #pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int idx = warp + r * NWARPS;
            if (NTR % NWARPS == 0 || idx < NTR) {
                constexpr int DG = HEAD_DIM / (KVP ? 32 : 64);
                const int kg = idx / DG, dg = idx - kg * DG;
                int kk = min(k0 + kg * 8 + vj, ctx - 1);
                if (WIN && kk < wlo0) kk = wlo0;
                const int d0 = dg * (KVP ? 32 : 64) + vg * (KVP ? 8 : 16);
                const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)blk * a.kv_block_stride
                               + (size_t)kvc * a.kv_head_stride
                               + (size_t)(kk % BS) * ROWB + HEAD_DIM + d0;
                const void* p = KVP ? (const void*)((const uint16_t*)a.kv + o)
                                    : (const void*)((const uint8_t*)a.kv + o);
                vreg[r] = __builtin_bit_cast(uint4, load_tr_b128(p));
            }
        }
    };

    auto storeV = [&]() {
        #pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int idx = warp + r * NWARPS;
            if (NTR % NWARPS == 0 || idx < NTR) {
                constexpr int DG = HEAD_DIM / (KVP ? 32 : 64);
                const int kg = idx / DG, dg = idx - kg * DG;
                const int d0 = dg * (KVP ? 32 : 64) + vg * (KVP ? 8 : 16);
                const uint4 v = vreg[r];
                if (KVP) {
                    uint16_t* dst = &sV16[(d0 + vj) * VSTR + kg * 8];
                    *(uint2*)(dst)     = make_uint2(D::from_bf16w(v.x), D::from_bf16w(v.y));
                    *(uint2*)(dst + 4) = make_uint2(D::from_bf16w(v.z), D::from_bf16w(v.w));
                } else if (V8) {
                    const uint32_t lo0 = byte_gather(v.x, v.y, 0), lo1 = byte_gather(v.z, v.w, 0);
                    const uint32_t hi0 = byte_gather(v.x, v.y, 1), hi1 = byte_gather(v.z, v.w, 1);
                    *(uint2*)&sV8[(d0 + 2 * vj)     * VSTR8 + kg * 8] = make_uint2(lo0, lo1);
                    *(uint2*)&sV8[(d0 + 2 * vj + 1) * VSTR8 + kg * 8] = make_uint2(hi0, hi1);
                } else {
                    const uint32_t lo0 = byte_gather(v.x, v.y, 0), lo1 = byte_gather(v.z, v.w, 0);
                    const uint32_t hi0 = byte_gather(v.x, v.y, 1), hi1 = byte_gather(v.z, v.w, 1);
                    uint32_t wl[4], wh[4];
                    fp8x8_to_16x4w<D>(make_uint2(lo0, lo1), wl);
                    fp8x8_to_16x4w<D>(make_uint2(hi0, hi1), wh);
                    uint16_t* a0 = &sV16[(d0 + 2 * vj)     * VSTR + kg * 8];
                    uint16_t* a1 = &sV16[(d0 + 2 * vj + 1) * VSTR + kg * 8];
                    *(uint2*)(a0)     = make_uint2(wl[0], wl[1]);
                    *(uint2*)(a0 + 4) = make_uint2(wl[2], wl[3]);
                    *(uint2*)(a1)     = make_uint2(wh[0], wh[1]);
                    *(uint2*)(a1 + 4) = make_uint2(wh[2], wh[3]);
                }
            }
        }
    };

    // A tile's NB entries are read whole, and a multi-block tile can run past the sequence's last
    // block (on the last row, past the end of the table). Keys are clamped to ctx - 1, so entries
    // past lastb are never used; only tiles reaching past lastb take the clamped load, and the
    // rest keep one wide scalar load. With NB == 1 the test folds away.
    const int lastb = (ctx - 1) / BS;
    auto load_bt = [&](int t0, int* out) {
        if (NB > 1 && t0 + NB - 1 > lastb) {
            #pragma unroll
            for (int i = 0; i < NB; ++i) out[i] = bt[min(t0 + i, lastb)];
        } else {
            #pragma unroll
            for (int i = 0; i < NB; ++i) out[i] = bt[t0 + i];
        }
    };
    int blk[NB];
    if (BTS) load_bt(t_lo * NB, blk);
    if (GPREV) fetchV(t_lo * TILE, blk);

    for (int ti = t_lo; ti < t_hi; ++ti) {
        const int k0 = ti * TILE;
        int blkn[NB];
        if (BTS && ti + 1 < t_hi) {
            load_bt((ti + 1) * NB, blkn);
        }
        if (LDSB) lds_barrier(); else __syncthreads();

        if (KLDS) {
            #pragma unroll 1
            for (int i = tid; i < TILE * (HEAD_DIM / 8); i += NTHR) {
                const int key = i / (HEAD_DIM / 8), ch = i - key * (HEAD_DIM / 8);
                int kk = min(k0 + key, ctx - 1);
                if (WIN && kk < wlo0) kk = wlo0;
                const int kb = BTS ? blk[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)kb * a.kv_block_stride
                               + (size_t)kvc * a.kv_head_stride + (size_t)(kk % BS) * ROWB;
                uint16_t* kd = &sK[key * KSTR];
                uint32_t w4[4];
                if (KVP) {
                    const uint4 r = *(const uint4*)((const uint16_t*)a.kv + o + ch * 8);
                    w4[0] = D::from_bf16w(r.x); w4[1] = D::from_bf16w(r.y);
                    w4[2] = D::from_bf16w(r.z); w4[3] = D::from_bf16w(r.w);
                } else {
                    fp8x8_to_16x4w<D>(*(const uint2*)((const uint8_t*)a.kv + o + ch * 8), w4);
                }
                *(uint2*)(kd + ch * 8)     = make_uint2(w4[0], w4[1]);
                *(uint2*)(kd + ch * 8 + 4) = make_uint2(w4[2], w4[3]);
            }
        }
        if constexpr (V4) { fetchV4(k0, blk); storeV4(); }
        else { if (!GPREV) fetchV(k0, blk); storeV(); }
        if (LDSB) lds_barrier(); else __syncthreads();
        if (GPREV && ti + 1 < t_hi) fetchV(k0 + TILE, blkn);

        #pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            const uint8_t* grow = nullptr;
            if (!KLDS) {
                int kk = min(k0 + mt * 16 + c, ctx - 1);
                if (WIN && kk < wlo0) kk = wlo0;
                const int kb = BTS ? blk[(kk - k0) / BS] : bt[kk / BS];
                grow = (const uint8_t*)a.kv + ((size_t)kb * a.kv_block_stride
                     + (size_t)kvc * a.kv_head_stride + (size_t)(kk % BS) * ROWB)
                     * (KVP ? 2 : 1);
            }
            // i8: the key's one K scale, read once an m-tile.
            const uint32_t ksb = I8 ? (uint32_t)*(const uint16_t*)(grow + KSOFF) : 0u;
            // A fragment's raw K: 8 elements of 1 byte (fp8) or 2 (bf16).
            typedef typename std::conditional<KVP != 0, uint4, uint2>::type kraw_t;
            kraw_t kraw[KRAW ? NKS : 1];
            if constexpr (KRAW) {
                #pragma unroll
                for (int t = 0; t < NKS; ++t)
                    kraw[t] = *(const kraw_t*)(grow + (16 * t + 8 * h) * (KVP ? 2 : 1));
            }
            v8f s = (v8f){0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int g = 0; g < NKS; g += PF) {
                frag16 kb[PF];
                #pragma unroll
                for (int j = 0; j < PF; ++j) {
                    const int t = g + j;
                    if (KLDS) {
                        const uint16_t* kp = &sK[(mt * 16 + c) * KSTR + 16 * t];
                        if (CTG) {
                            const uint4 u = *(const uint4*)(kp + 8 * h);
                            kb[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        } else {
                            kb[j] = D::mk(*(const uint2*)(kp + 4 * h), *(const uint2*)(kp + 8 + 4 * h));
                        }
                    } else if (KVP) {
                        const uint16_t* kp = (const uint16_t*)grow + 16 * t;
                        uint2 l0, h0;
                        if constexpr (KRAW && KVP != 0) {
                            const uint4 u = kraw[t];
                            l0 = make_uint2(u.x, u.y);
                            h0 = make_uint2(u.z, u.w);
                        } else {
                            l0 = *(const uint2*)(kp + (CTG ? 8 * h : 4 * h));
                            h0 = *(const uint2*)(kp + (CTG ? 8 * h + 4 : 8 + 4 * h));
                        }
                        kb[j] = D::mk(make_uint2(D::from_bf16w(l0.x), D::from_bf16w(l0.y)),
                                      make_uint2(D::from_bf16w(h0.x), D::from_bf16w(h0.y)));
                    } else if constexpr (I8) {
                        // i8: this lane's eight dims (16t + 8h ..) as bytes, times the key's scale.
                        uint32_t w4[4];
                        i8x8_to_f16x4w(*(const uint2*)(grow + 16 * t + 8 * h), ksb | (ksb << 16), w4);
                        kb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                    } else if constexpr (K6) {
                        // k6_v4: this lane's eight dims (16t + 8h ..) from their low nibbles, the
                        // chunk's high-bit word and the group's scale.
                        uint32_t w4[4];
                        const uint32_t sb = *(const uint16_t*)(grow + KSOFF + (t >> 1) * 2);
                        k6x8_to_f16x4w(*(const uint32_t*)(grow + 8 * t + 4 * h),
                                       *(const uint32_t*)(grow + HEAD_DIM / 2 + 4 * t), h,
                                       sb | (sb << 16), w4);
                        kb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                    } else {
                        const uint8_t* kp = grow + 16 * t;
                        uint32_t w4[4];
                        // CONTIG: one 8-byte load; the split map needs two 4-byte loads.
                        uint2 raw;
                        if constexpr (KRAW && KVP == 0) raw = kraw[t];
                        else raw = CTG ? *(const uint2*)(kp + 8 * h)
                                       : make_uint2(*(const uint32_t*)(kp + 4 * h),
                                                    *(const uint32_t*)(kp + 8 + 4 * h));
                        fp8x8_to_16x4w<D>(raw, w4);
                        kb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                    }
                }
                if (SGB) sched_barrier();
                #pragma unroll
                for (int j = 0; j < PF; ++j) s = D::wmma(kb[j], qf[g + j], s);
            }
            float sc[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) sc[e] = FOLDQ ? s[e] : s[e] * mul;

            const int kbase = k0 + mt * 16 + 8 * h;
            if (!MSKIP || wave_any(kbase + 7 > klimit) || (WIN && wave_any(kbase < wlo)) ||
                (TREE && wave_any(trow && kbase + 7 >= tbase))) {
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    sc[e] = ((kbase + e <= klimit) && (!WIN || kbase + e >= wlo) &&
                             (!TREE || !trow || kbase + e < tbase ||
                              ((tmask >> (kbase + e - tbase)) & 1ull)))
                          ? sc[e] : -INFINITY;
            }
            float smax = sc[0];
            #pragma unroll
            for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);

            if (wave_any(smax > m_ref + PGROW)) {
                const float rmax  = fmaxf(smax, swap16(smax));
                const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
                const float alpha = ex2(m_ref - mnew);
                #pragma unroll
                for (int dt = 0; dt < NKS; ++dt)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[dt][e] *= alpha;
                l_i *= alpha;
                m_ref = mnew;
                off   = PSHIFT - mnew;
            }

            float pf[8], lsum = 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float p = ex2(sc[e] + off); pf[e] = p;
                if (!DOT2) lsum += p;
            }
            if (!DOT2) l_i += lsum;

            {
                uint32_t pw[4];
                #pragma unroll
                for (int e = 0; e < 8; e += 2) pw[e >> 1] = D::pk(pf[e], pf[e + 1]);
                if (DOT2) {
                    const v2hf one = {(__fp16)1.0f, (__fp16)1.0f};
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                        l_i = __builtin_amdgcn_fdot2(__builtin_bit_cast(v2hf, pw[e]), one, l_i, false);
                }
                const uint2 lo  = make_uint2(pw[0], pw[1]);
                const uint2 hi2 = make_uint2(pw[2], pw[3]);
                frag16 p16;
                if (CTG) {
                    p16 = D::mk(lo, hi2);          // QK already left the keys lane-local
                } else {
                    const uint2 y = swap16_u2p((h == 0) ? hi2 : lo);
                    p16 = (h == 0) ? D::mk(lo, y) : D::mk(y, hi2);
                }
                // Skewed tile: row 16 G + c is in octet 2 G + (c >> 3), so its S is 16 (c >> 3) here
                // plus 32 (G & 1) + 8 ((G >> 1) & 1) + 128 (G >> 1), a compile-time offset below.
                const uint16_t* vbase = &sV16[c * VSTR + mt * 16] + (CTG ? 8 * h : 0)
                                      + (VSK ? 16 * (c >> 3) : 0);
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    frag16 vb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        const uint16_t* vp = vbase + (g + j) * 16 * VSTR
                            + (VSK ? 32 * ((g + j) & 1) + 8 * (((g + j) >> 1) & 1) + 128 * ((g + j) >> 1)
                                   : 0);
                        if (V8) {
                            uint32_t w4[4];
                            fp8x8_to_16x4w<D>(*(const uint2*)&sV8[(c + (g + j) * 16) * VSTR8 +
                                                                 mt * 16 + 8 * h], w4);
                            vb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                        } else if (CTG) {
                            const uint4 u = *(const uint4*)vp;
                            vb[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        } else {
                            vb[j] = D::mk(*(const uint2*)(vp + 4 * h),
                                          *(const uint2*)(vp + 8 + 4 * h));
                        }
                    }
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) acc[g + j] = D::wmma(vb[j], p16, acc[g + j]);
                }
            }
        }
        if (BTS) {
            #pragma unroll
            for (int i = 0; i < NB; ++i) blk[i] = blkn[i];
        }
    }

    if (!live) return;
    // fetch-inactive: the f32 partials read lrow under `if (h == 0)` only (see the prefill kernel's
    // epilogue).
    const float lrow = l_i + swap16_fi(l_i);
    const int tok = qoff + qrow;
    const size_t idx = (size_t)(tok * a.q_heads + qhead) * splits + sp;
    const size_t nrow = (size_t)r4d_q_total(a) * a.q_heads * splits;
    if (PF16) {
        // O/l in f16: bounded by max|v|, so no overflow, and the merge weight becomes w*l.
        uint16_t* so = (uint16_t*)a.scratch;
        float* sm = (float*)(so + nrow * HEAD_DIM);
        float* sl = sm + nrow;
        uint16_t* op = so + idx * HEAD_DIM;
        const float inv = (lrow > 0.0f) ? (1.0f / lrow) : 0.0f;
        #pragma unroll
        for (int dt = 0; dt < NKS; ++dt) {
            uint32_t tw[4];
            #pragma unroll
            for (int e = 0; e < 8; e += 2) tw[e >> 1] = DT16<1>::pk(acc[dt][e] * inv,
                                                                    acc[dt][e + 1] * inv);
            *(uint4*)(op + dt * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
        }
        if (h == 0) { sm[idx] = m_ref; sl[idx] = lrow; }
    } else {
        float* so = (float*)a.scratch;
        float* sm = so + nrow * HEAD_DIM;
        float* sl = sm + nrow;
        float* op = so + idx * HEAD_DIM;
        #pragma unroll
        for (int dt = 0; dt < NKS; ++dt) *(v8f*)(op + dt * 16 + 8 * h) = acc[dt];
        if (h == 0) { sm[idx] = m_ref; sl[idx] = lrow; }
    }
}

// ---- Output gate and fp8 quantiser as the merge's epilogue (attn_paged_gate_quant) ----
//
// A gated attention's merged row feeds gate_quant_fp8: bf16(sigmoid(gate)) times the row, rounded
// to bf16, then E4M3 codes with one scale per 128 columns. A merge row is one (token, head) pair
// and a scale group is 128 columns, so the `_gq` forms finish the row in registers instead of in a
// second dispatch.
//
// Byte-identical to gate_quant_fp8 (r4d_fused_quant_fp8.hip): gate rounded to bf16 before the
// multiply, product rounded, scale = group absmax / 448 (1 at zero), codes clamped to +-448, and
// the bf16 product stored to `out`. A padded slot merges to zeros and takes the same epilogue.
// GQ == 2 is the int8 form (scale amax / 127, round to nearest-even, r4d_q8.h).
// GQ == 3 (attn_paged_gate_had_quant) is the int8 form with the width-128 Hadamard between the
// bf16 product and the absmax, byte-identical to gate_had_quant_i8g: `out` keeps the unrotated
// product, and the stages run by ascending stride as r4d_fwht_wave's do, whose butterflies give
// the same bits under any thread map.
struct R4DGateQuant {
    const uint16_t* g;          // gate, row `job` at g + job * g_ld
    uint8_t*        q;          // E4M3 (or int8, `i8`) codes, row `job` at q + job * q_ld
    float*          s;          // one scale per 128 columns, row `job` at s + job * s_ld
    unsigned        g_ld, q_ld, s_ld;
    int             i8;         // read by the launcher only: which combine to instantiate
    int             rot;        // likewise: the rotated int8 form (GQ 3)
};
constexpr int kGqGroup = 128;

__device__ __forceinline__ uint16_t r4d_gq_gate(uint16_t gb, uint16_t xb) {
    const float    gv = bf16_to_f32(gb);
    const uint16_t ab = f32_to_bf16(1.0f / (1.0f + expf(-gv)));
    return f32_to_bf16(bf16_to_f32(ab) * bf16_to_f32(xb));
}
__device__ __forceinline__ float r4d_gq_clamp(float v) { return fminf(fmaxf(v, -448.0f), 448.0f); }

// The plain decode kernel, and the chained one (r4dchain*, r4d_chain.h) over the same body.
template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT,
         int WIN = 0>
__global__ __launch_bounds__(NWARPS * 32)
void r4d_attn_decode_kernel(const R4DArgs a, int splits)
{
    r4d_attn_decode_body<NWARPS, TILE, HEAD_DIM, GQA, BS, KVP, OPT, WIN>(a, splits);
}
template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT,
         int WIN = 0>
__global__ __launch_bounds__(NWARPS * 32)
void r4dchain_attn_decode(const R4DArgs a, int splits)
{
    r4d_chain_wait();
    r4d_attn_decode_body<NWARPS, TILE, HEAD_DIM, GQA, BS, KVP, OPT, WIN>(a, splits);
    r4d_chain_done();
}

// Vectorised merge, used by the h256 launcher whenever the split count fits a wave. Bit-identical
// to splitkv_combine_body below; only the thread map differs. One wave owns one output row and
// each lane HEAD_DIM/32 dims, so a segment is one 16-byte load per lane; a block carries JPB rows.
// No barrier and no LDS, so rows in one block may have different contexts and segment counts.
//
// SPAN: merge only rows [cu_q[0], cu_q[num_seqs]). A launch over part of a step (ArPieces' pieces,
// a mixed step's prefill half) gets q and out whole with cu_q at its own first row; r4d_q_seq
// would map any other row onto this launch's first or last sequence and overwrite it. The decode
// launchers never offset cu_q and do not instantiate it.
template<int HEAD_DIM, int CHAINS, int PF16, int JPB, int WIN, int GQ, int SPAN = 0>
__device__ __forceinline__
void splitkv_combine_vec_body(const R4DArgs& a, int splits, int tile, const R4DGateQuant& gq)
{
    constexpr int DPL = HEAD_DIM / 32;                 // dims a lane owns
    const int lane = threadIdx.x & 31;
    const int job  = blockIdx.x * JPB + (int)(threadIdx.x >> 5);
    if (job >= r4d_q_total(a) * a.q_heads) return;
    const int tok = job / a.q_heads;
    if (SPAN && a.cu_q && (tok < a.cu_q[0] || tok >= a.cu_q[a.num_seqs])) return;
    const int seq = r4d_q_seq(a.cu_q, a.num_seqs, a.q_len, tok);
    const int ctx = a.seqused_k[seq];
    uint16_t* outp = (uint16_t*)a.out + (size_t)job * HEAD_DIM + lane * DPL;

    /* Plain store, or gated + quantised. A lane's DPL columns lie in one 128-column group of
     * 128 / DPL lanes, so the group absmax is a butterfly over those lanes only. */
    auto finish = [&](const uint16_t (&ob)[DPL]) {
        if constexpr (!GQ) {
            union { uint4 u4; uint2 u2; uint16_t w[DPL]; } o_;
            #pragma unroll
            for (int j = 0; j < DPL; ++j) o_.w[j] = ob[j];
            if (DPL == 8) *(uint4*)outp = o_.u4; else *(uint2*)outp = o_.u2;
        } else {
            constexpr int LPG = kGqGroup / DPL;        // lanes a scale group
            static_assert(DPL == 8 || DPL == 4, "a lane's columns are one or two dwords of codes");
            const uint16_t* gp = gq.g + (size_t)job * gq.g_ld + lane * DPL;
            union { uint4 u4; uint2 u2; uint16_t w[DPL]; } gi, o_;
            if (DPL == 8) gi.u4 = *(const uint4*)gp; else gi.u2 = *(const uint2*)gp;
            float v[DPL];
            #pragma unroll
            for (int j = 0; j < DPL; ++j) {
                o_.w[j] = r4d_gq_gate(gi.w[j], ob[j]);
                v[j] = bf16_to_f32(o_.w[j]);
            }
            if (DPL == 8) *(uint4*)outp = o_.u4; else *(uint2*)outp = o_.u2;
            if constexpr (GQ == 3) r4d_fwht_part<DPL, LPG>(v, lane);
            float m = 0.0f;
            #pragma unroll
            for (int j = 0; j < DPL; ++j) m = fmaxf(m, fabsf(v[j]));
            if constexpr (LPG > 16) m = fmaxf(m, r4d_xor_lane<16>(m));
            if constexpr (LPG > 8)  m = fmaxf(m, r4d_xor_lane<8>(m));
            m = fmaxf(m, r4d_xor_lane<4>(m));
            m = fmaxf(m, r4d_xor_lane<2>(m));
            m = fmaxf(m, r4d_xor_lane<1>(m));
            const float sc  = m > 0.0f ? m * (1.0f / q8_top<GQ >= 2>()) : 1.0f;
            const float inv = 1.0f / sc;
            if ((lane % LPG) == 0) gq.s[(size_t)job * gq.s_ld + (lane * DPL) / kGqGroup] = sc;
            uint32_t pk[DPL / 4];
            #pragma unroll
            for (int w = 0; w < DPL / 4; ++w)
                pk[w] = q8_pack4<GQ >= 2>(v[4 * w + 0], v[4 * w + 1], v[4 * w + 2], v[4 * w + 3],
                                          inv);
            uint8_t* qp = gq.q + (size_t)job * gq.q_ld + lane * DPL;
            if (DPL == 8) *(uint2*)qp = make_uint2(pk[0], pk[DPL / 4 - 1]);
            else          *(uint32_t*)qp = pk[0];
        }
    };

    if (ctx <= 0) {                                    // padded slot: zeros, read nothing
        uint16_t ob[DPL];
        #pragma unroll
        for (int i = 0; i < DPL; ++i) ob[i] = 0;
        finish(ob);
        return;
    }
    const int nef = r4d_attn_tiles(ctx, tile)
                  - (WIN ? r4d_attn_wlo(ctx, r4d_q_len(a.cu_q, seq, a.q_len), 0, a.window) / tile : 0);
    const int tp_ = (nef + splits - 1) / splits;
    const int u   = r4d_row_segments(a, seq, tok, ctx, tile, tp_, nef, WIN);
    const size_t nrow = (size_t)r4d_q_total(a) * a.q_heads * splits;
    const size_t base = (size_t)job * splits;

    const float* sm; const float* sl;
    if (PF16) { const uint16_t* so = (const uint16_t*)a.scratch;
                sm = (const float*)(so + nrow * HEAD_DIM); sl = sm + nrow; }
    else      { const float* so = (const float*)a.scratch;
                sm = so + nrow * HEAD_DIM; sl = sm + nrow; }

    // max is order-independent and wave-uniform, so it is a scalar walk. The sum reproduces the
    // LDS tree's association (stride-32 pre-accumulation, then xor butterfly 16..1) so results
    // match the non-vector merge bit for bit.
    float gmax = -INFINITY;
    for (int s = 0; s < u; ++s) gmax = fmaxf(gmax, sm[base + s]);
    float tot = 0.0f;
    for (int s = lane; s < u; s += 32) {
        const float mm = sm[base + s];
        const float w  = (mm <= -3.0e38f) ? 0.0f : ex2(mm - gmax);
        tot += w * sl[base + s];
    }
    tot = r4d_wave_sum(tot);

    float ch[CHAINS][DPL];
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i)
        #pragma unroll
        for (int j = 0; j < DPL; ++j) ch[i][j] = 0.0f;

    auto wgt = [&](int sx) {
        const float mm = sm[base + sx];
        const float w  = (mm <= -3.0e38f) ? 0.0f : ex2(mm - gmax);
        return PF16 ? w * sl[base + sx] : w;
    };

    int s = 0;
    if (PF16) {
        const uint16_t* op = (const uint16_t*)a.scratch + base * HEAD_DIM + lane * DPL;
        // One aligned access per lane per segment: 16 bytes at head_dim 256, 8 at 128.
        union { uint4 u4; uint2 u2; __fp16 h[DPL]; } r;
        auto rd = [&](int sx) {
            if (DPL == 8) r.u4 = *(const uint4*)(op + (size_t)sx * HEAD_DIM);
            else          r.u2 = *(const uint2*)(op + (size_t)sx * HEAD_DIM);
        };
        for (; s + CHAINS <= u; s += CHAINS) {
            #pragma unroll
            for (int i = 0; i < CHAINS; ++i) {
                rd(s + i);
                const float w = wgt(s + i);
                #pragma unroll
                for (int j = 0; j < DPL; ++j) ch[i][j] = fmaf(w, (float)r.h[j], ch[i][j]);
            }
        }
        for (; s < u; ++s) {
            rd(s);
            const float w = wgt(s);
            #pragma unroll
            for (int j = 0; j < DPL; ++j) ch[0][j] = fmaf(w, (float)r.h[j], ch[0][j]);
        }
    } else {
        const float* op = (const float*)a.scratch + base * HEAD_DIM + lane * DPL;
        for (; s + CHAINS <= u; s += CHAINS) {
            #pragma unroll
            for (int i = 0; i < CHAINS; ++i) {
                const float* p = op + (size_t)(s + i) * HEAD_DIM;
                const float w = wgt(s + i);
                #pragma unroll
                for (int j = 0; j < DPL; ++j) ch[i][j] = fmaf(w, p[j], ch[i][j]);
            }
        }
        for (; s < u; ++s) {
            const float* p = op + (size_t)s * HEAD_DIM;
            const float w = wgt(s);
            #pragma unroll
            for (int j = 0; j < DPL; ++j) ch[0][j] = fmaf(w, p[j], ch[0][j]);
        }
    }

    const int kvh = (job % a.q_heads) / (a.q_heads / a.kv_heads);   /* the query group */
    const int krep = a.kv_rep > 1 ? a.kv_rep : 1;                    /* its cache head: kvh / krep */
    const float vd = a.v_descale ? a.v_descale[seq * (a.kv_heads / krep) + kvh / krep] : 1.0f;
    uint16_t ob[DPL];
    #pragma unroll
    for (int j = 0; j < DPL; ++j) {
        float av = 0.0f;
        #pragma unroll
        for (int i = 0; i < CHAINS; ++i) av += ch[i][j];
        // (av * vd) / tot in this order; folding vd/tot first rounds differently (one bf16 ULP).
        ob[j] = f32_to_bf16(tot > 0.0f ? av * vd / tot : 0.0f);
    }
    finish(ob);
}

template<int HEAD_DIM, int CHAINS, int PF16, int JPB, int WIN = 0, int SPAN = 0>
__global__ __launch_bounds__(32 * JPB)
void r4d_attn_splitkv_combine_vec_kernel(const R4DArgs a, int splits, int tile)
{
    splitkv_combine_vec_body<HEAD_DIM, CHAINS, PF16, JPB, WIN, 0, SPAN>(a, splits, tile,
                                                                         R4DGateQuant{});
}
template<int HEAD_DIM, int CHAINS, int PF16, int JPB, int WIN = 0>
__global__ __launch_bounds__(32 * JPB)
void r4dchain_attn_splitkv_combine_vec(const R4DArgs a, int splits, int tile)
{
    r4d_chain_wait();
    splitkv_combine_vec_body<HEAD_DIM, CHAINS, PF16, JPB, WIN, 0>(a, splits, tile,
                                                                   R4DGateQuant{});
    r4d_chain_done();
}
template<int HEAD_DIM, int CHAINS, int PF16, int JPB, int WIN = 0, int GQ = 1>
__global__ __launch_bounds__(32 * JPB)
void r4d_attn_splitkv_combine_vec_gq_kernel(const R4DArgs a, int splits, int tile,
                                            const R4DGateQuant gq)
{
    splitkv_combine_vec_body<HEAD_DIM, CHAINS, PF16, JPB, WIN, GQ>(a, splits, tile, gq);
}

// Merge. PF16=1 reads normalised f16 partials and folds l into the weight; PF16=0 reads
// unnormalised f32 partials with l summed separately. Both walk segments in fixed index order so
// rounding never depends on workgroup scheduling. SPAN as in the vectorised form (the return is
// block-uniform, ahead of every barrier).
template<int HEAD_DIM, int CHAINS, int PF16, int WIN, int GQ, int SPAN = 0>
__device__ __forceinline__
void splitkv_combine_body(const R4DArgs& a, int splits, int tile, const R4DGateQuant& gq)
{
    extern __shared__ float sw[];
    __shared__ float red[HEAD_DIM];
    const int job = blockIdx.x, d = threadIdx.x;
    const int tok = job / a.q_heads;
    if (SPAN && a.cu_q && (tok < a.cu_q[0] || tok >= a.cu_q[a.num_seqs])) return;
    const int seq = r4d_q_seq(a.cu_q, a.num_seqs, a.q_len, tok);
    const int ctx = a.seqused_k[seq];

    /* One thread per column, so a 128-column scale group is four waves, combined through `red`.
     * Reusing `red` is safe: its last read (the total) is behind a barrier before the merge loop,
     * and a padded slot never used it. */
    auto finish = [&](uint16_t ob) {
        uint16_t* op = (uint16_t*)a.out + (size_t)job * HEAD_DIM + d;
        if constexpr (!GQ) {
            *op = ob;
        } else {
            constexpr int WPG = kGqGroup / 32;         // waves a scale group
            const uint16_t h = r4d_gq_gate(gq.g[(size_t)job * gq.g_ld + d], ob);
            *op = h;
            float v = bf16_to_f32(h);
            if constexpr (GQ == 3) {
                /* Strides 1 to 16 pair lanes of the wave; 32 and 64 pair waves of the group, two
                 * exchanges through `red`, each read behind the barrier after its store and ahead
                 * of the next. */
                r4d_fwht_part<1, 32>(&v, d & 31);
                #pragma unroll
                for (int hs = 32; hs < kGqGroup; hs <<= 1) {
                    red[d] = v;
                    __syncthreads();
                    const float o = red[d ^ hs];
                    v = __int_as_float(__float_as_int(v) ^ ((d & hs) ? 0x80000000u : 0u)) + o;
                    __syncthreads();
                }
            }
            const float m = r4d_wave_max(fabsf(v));
            if ((d & 31) == 0) red[d / 32] = m;
            __syncthreads();
            const int g0 = (d / kGqGroup) * WPG;
            float gm = red[g0];
            #pragma unroll
            for (int w = 1; w < WPG; ++w) gm = fmaxf(gm, red[g0 + w]);
            const float sc  = gm > 0.0f ? gm * (1.0f / q8_top<GQ >= 2>()) : 1.0f;
            const float inv = 1.0f / sc;
            if ((d % kGqGroup) == 0) gq.s[(size_t)job * gq.s_ld + d / kGqGroup] = sc;
            if constexpr (GQ >= 2) {
                gq.q[(size_t)job * gq.q_ld + d] =
                    (uint8_t)(int)rintf(fminf(fmaxf(v * inv, -127.0f), 127.0f));
            } else {
                const int p = __builtin_amdgcn_cvt_pk_fp8_f32(r4d_gq_clamp(v * inv), 0.0f, 0, false);
                gq.q[(size_t)job * gq.q_ld + d] = (uint8_t)(p & 0xff);
            }
        }
    };

    // A padded slot (ctx 0) would divide by zero below; write zeros and stop. ctx is
    // block-uniform, so the barriers below stay matched.
    if (ctx <= 0) { finish(0); return; }
    // Segment count, derived exactly as in the decode kernel, including the window's tile floor;
    // otherwise the merge would read partials no segment wrote.
    const int nef = r4d_attn_tiles(ctx, tile)
                  - (WIN ? r4d_attn_wlo(ctx, r4d_q_len(a.cu_q, seq, a.q_len), 0, a.window) / tile : 0);
    const int tp_ = (nef + splits - 1) / splits;
    const int u   = r4d_row_segments(a, seq, tok, ctx, tile, tp_, nef, WIN);
    const size_t nrow = (size_t)r4d_q_total(a) * a.q_heads * splits;
    const size_t base = (size_t)job * splits;

    const float* sm; const float* sl;
    if (PF16) { const uint16_t* so = (const uint16_t*)a.scratch;
                sm = (const float*)(so + nrow * HEAD_DIM); sl = sm + nrow; }
    else      { const float* so = (const float*)a.scratch;
                sm = so + nrow * HEAD_DIM; sl = sm + nrow; }

    float gmax = -INFINITY;
    for (int s = d; s < u; s += HEAD_DIM) gmax = fmaxf(gmax, sm[base + s]);
    red[d] = gmax; __syncthreads();
    for (int off = HEAD_DIM / 2; off; off >>= 1) {
        if (d < off) red[d] = fmaxf(red[d], red[d + off]);
        __syncthreads();
    }
    gmax = red[0]; __syncthreads();

    float tot = 0.0f;
    for (int s = d; s < u; s += HEAD_DIM) {
        const float mm = sm[base + s];
        float w = (mm <= -3.0e38f) ? 0.0f : ex2(mm - gmax);
        if (PF16) { w *= sl[base + s]; sw[s] = w; tot += w; }
        else      { sw[s] = w;         tot += w * sl[base + s]; }
    }
    red[d] = tot; __syncthreads();
    for (int off = HEAD_DIM / 2; off; off >>= 1) {
        if (d < off) red[d] += red[d + off];
        __syncthreads();
    }
    tot = red[0]; __syncthreads();

    float ch[CHAINS];
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i) ch[i] = 0.0f;
    int s = 0;
    if (PF16) {
        const uint16_t* op = (const uint16_t*)a.scratch + base * HEAD_DIM + d;
        for (; s + CHAINS <= u; s += CHAINS) {
            #pragma unroll
            for (int i = 0; i < CHAINS; ++i)
                ch[i] = fmaf(sw[s + i], (float)__builtin_bit_cast(
                    __fp16, op[(size_t)(s + i) * HEAD_DIM]), ch[i]);
        }
        for (; s < u; ++s)
            ch[0] = fmaf(sw[s], (float)__builtin_bit_cast(__fp16, op[(size_t)s * HEAD_DIM]), ch[0]);
    } else {
        const float* op = (const float*)a.scratch + base * HEAD_DIM + d;
        for (; s + CHAINS <= u; s += CHAINS) {
            #pragma unroll
            for (int i = 0; i < CHAINS; ++i)
                ch[i] = fmaf(sw[s + i], op[(size_t)(s + i) * HEAD_DIM], ch[i]);
        }
        for (; s < u; ++s) ch[0] = fmaf(sw[s], op[(size_t)s * HEAD_DIM], ch[0]);
    }
    float av = 0.0f;
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i) av += ch[i];

    const int kvh = (job % a.q_heads) / (a.q_heads / a.kv_heads);   /* the query group */
    const int krep = a.kv_rep > 1 ? a.kv_rep : 1;                    /* its cache head: kvh / krep */
    const float vd = a.v_descale ? a.v_descale[seq * (a.kv_heads / krep) + kvh / krep] : 1.0f;
    finish(f32_to_bf16(tot > 0.0f ? av * vd / tot : 0.0f));
}

template<int HEAD_DIM, int CHAINS, int PF16, int WIN = 0, int SPAN = 0>
__global__ __launch_bounds__(HEAD_DIM)
void r4d_attn_splitkv_combine_kernel(const R4DArgs a, int splits, int tile)
{
    splitkv_combine_body<HEAD_DIM, CHAINS, PF16, WIN, 0, SPAN>(a, splits, tile, R4DGateQuant{});
}
template<int HEAD_DIM, int CHAINS, int PF16, int WIN = 0>
__global__ __launch_bounds__(HEAD_DIM)
void r4dchain_attn_splitkv_combine(const R4DArgs a, int splits, int tile)
{
    r4d_chain_wait();
    splitkv_combine_body<HEAD_DIM, CHAINS, PF16, WIN, 0>(a, splits, tile, R4DGateQuant{});
    r4d_chain_done();
}
template<int HEAD_DIM, int CHAINS, int PF16, int WIN = 0, int GQ = 1>
__global__ __launch_bounds__(HEAD_DIM)
void r4d_attn_splitkv_combine_gq_kernel(const R4DArgs a, int splits, int tile,
                                        const R4DGateQuant gq)
{
    splitkv_combine_body<HEAD_DIM, CHAINS, PF16, WIN, GQ>(a, splits, tile, gq);
}

// ---- host helpers shared by the paged decode dispatches -----------------------------------------

// Split law of the full-attention legs (h256/gqa6 and the gqa8 family): as many workgroups as fit
// under 128 (grid = splits * kv_heads * num_seqs), at least two tiles per segment. It does not
// depend on gqa or head_dim.
//
// floor(128 / units), not a power of two: a power of two leaves 96-112 workgroups at 3, 5, 6 and
// 7 requests, and floor is 2-20% faster there. Staying under a multiple of 64 matters (at five
// requests, 13 splits = 130 workgroups is three rounds of 64 and slower than 12).
//
// Fewer splits mainly save partial-buffer traffic (seqs x q_len x q_heads x splits rows of
// head_dim f16, written once and read once): at eight requests 12.8 MB at splits 32 vs 3.2 MB at
// 8, 19% of a 16K step's KV but 1.2% at 128K. The effect fades with depth, never enough to need a
// context term.
//
// The target is 256 workgroups where a workgroup has one or two waves (q_len * gqa <= 32: a plain
// decode, or a 4-row verify of the DFlash2 short block), 128 otherwise: a one- or two-wave
// workgroup keeps too few KV bytes in flight to load the card's DRAM at 128 (Qwen3.8-27B TP2,
// k6_v4: one row over 100k 264 -> 175 us, a decode step at speculation 7 -4.5 %; 4x and 8x the
// target were worse than 2x). At three waves (an 8-row verify) more workgroups only add
// contention (+9 %). The split count already moves with the batch, so the merge sees no new
// partition.
//
// Except that a three-wave workgroup reads the step's depth (r4d_decode_depth): past
// kVerifyMinFrom it keeps at least kVerifySplitsMin segments, what 128 gives one sequence of two
// KV heads, and past kVerifyDeepFrom it targets 512 workgroups (at most kVerifyDeepMax segments).
// At depth the 128 target starves an 8-row verify, each segment walking 2-18k tokens. Qwen3.8-27B
// at TP2 on the R9700, k6_v4, speculation 7, ms a decode step, forced segment counts
// (RADIANCE_TUNE=attn_paged:splits), the 128 target (16-21 at x3-x4, 10 at x6) -> 64: x3 45k
// 27.83 -> 27.56, 90k 32.61 -> 31.79, 180k 42.31 -> 40.47; x4 45k 29.96 -> 29.48, 90k 36.03 ->
// 34.89, 180k 48.74 -> 46.26; x6 45k 37.39 -> 36.11, 90k 47.74 -> 44.70. Deeper still two
// sequences want 128 (90k 26.73 -> 26.36, 180k 33.00 -> 32.01) and one 256 (180k 25.25 -> 24.93,
// 90k level), while four stay at 64 (128: 180k 45.94 vs 46.32, 90k 35.06 vs 35.00) and six lose at
// 128 (90k 44.91 vs 44.70): about 512 workgroups. Shallow the minimum costs up to half a percent
// (1k: x2 20.98 -> 21.01, x4 24.01 -> 24.15, x6 28.71 -> 28.86; x4 16k 25.80 -> 25.85), hence
// the floor. The kernel benchmark above (+9 % at x2) did not carry to the server. 64 is also the
// vectorised merge's limit and the count one sequence already splits at.
// Served against the 128 target, ms a decode step: x1 90k 22.07 -> 22.03, 180k 25.35 -> 24.88;
// x2 90k 26.83 -> 26.36, 180k 33.18 -> 31.99; x3 180k 42.41 -> 40.33; x4 90k 36.20 -> 34.96. A
// plain decode, the 4-row short block (16k x16 38.94 -> 38.90) and a verify up to 16k (1k x1-x8)
// keep their segment count. Greedy output is unchanged.
static constexpr int kVerifySplitsMin = 64;
// The floor is also bounded by the batch: at most kVerifyFloorWorkgroups workgroups (segments x KV
// heads x sequences). The scratch is sized at declare for the worst sequence count up to max_seqs,
// and an unbounded floor reserves 64 segments for each sequence: 487.5 MiB a card for h256/gqa12 at
// 256 sequences and max_ctx 200000, 7.62 MiB under the bound. Up to eight sequences of two KV
// heads the floor stays 64; at 16 it is 32, at 256 two: the region stays at what eight need.
static constexpr int kVerifyFloorWorkgroups = 1024;
static constexpr int kVerifyMinFrom   = 16384;
static constexpr int kVerifyDeepFrom  = 65536;
static constexpr int kVerifyDeepMax   = 256;

// The deepest context the step can reach. At issue the block table is the step's: its width moves
// with the batch's longest sequence, rounded up (core/sched/batch.cpp, bound_bucket), while
// `max_ctx` is the declared bound, the model's context. Declare passes no table (max_blocks 0), so
// the scratch is sized at the declared bound, the deepest the law can be asked for; the law grows
// with depth, so an issue never needs more.
static inline int r4d_decode_depth(const R4DArgs* a) {
    if (a->max_blocks > 0 && a->block_size > 0) {
        const long long t = (long long)a->max_blocks * a->block_size;
        if (t < a->max_ctx) return (int)t;
    }
    return a->max_ctx;
}

static inline int r4d_decode_splits(const R4DArgs* a, int tile) {
    if (a->splits > 0) return a->splits;
    const int tiles = (a->max_ctx + tile - 1) / tile;
    const int units = std::max(1, a->kv_heads * a->num_seqs);
    const int gqa = a->kv_heads > 0 ? std::max(1, a->q_heads / a->kv_heads) : 1;
    int s = 256 / units;
    if (a->q_len * gqa > 32) {
        const int depth = r4d_decode_depth(a);
        s = (depth > kVerifyDeepFrom ? 512 : 128) / units;
        if (s > kVerifyDeepMax) s = kVerifyDeepMax;
        if (depth > kVerifyMinFrom) {
            const int floor = std::min(kVerifySplitsMin, kVerifyFloorWorkgroups / units);
            if (s < floor) s = floor;
        }
    }
    const int cap = tiles / 2;
    if (s > cap) s = cap;
    if (s > tiles) s = tiles;
    if (s < 1) s = 1;
    return s;
}

// Split law of the windowed legs (the DFlash2 drafter at h128/gqa4, Gemma 4 at h256/gqa2): target
// 192 workgroups, round down to a power of two (min 16), keep at least two tiles per segment, then
// step up off splits * units == 256, a slow spot for these geometries. The tile count comes from
// the window when one is set, since that is the span the kernel walks.
static inline int r4d_decode_splits_win(const R4DArgs* a, int tile) {
    if (a->splits > 0) return a->splits;
    const int span  = (a->window > 0 && a->window + a->q_len < a->max_ctx)
                    ? a->window + a->q_len : a->max_ctx;
    const int tiles = (span + tile - 1) / tile;
    const int units = std::max(1, a->kv_heads * a->num_seqs);
    const int raw   = 192 / units;
    int s = 128;
    while (s > 16 && s > raw) s >>= 1;
    const int cap = tiles / 2;
    while (s > 16 && s > cap) s >>= 1;
    if (s > tiles) s = tiles;
    if (s < 1) s = 1;
    if (s * units == 256 && s * 2 <= tiles) s <<= 1;
    return s;
}

// Bytes of split-KV partials one decode launch needs, the same for both cache dtypes: f16 O/l plus
// m and l under O_PF16, else f32 O plus m and l.
static inline long r4d_decode_scratch(const R4DArgs* a, int splits, int opt) {
    const long rows = (long)r4d_q_total(*a) * a->q_heads;
    return (opt & O_PF16) ? rows * splits * (long)(a->head_dim * 2 + 8)
                          : rows * splits * (long)(a->head_dim + 2) * 4L;
}

// The windowed legs' decode (WIN = 1): the attention kernel at the gqa-aligned height, then the
// merge, one wave per row when there are enough rows and the segment count fits a wave, else the
// LDS-tree form; both give identical bytes. Tile 16, paged block 16, at most 64 query rows, and
// the full-attention legs' merge constants (two rows a block, the wave form from 256 rows).
// Returns 0, a negative code for a shape it does not serve, or -100 - hipError_t.
template <int HEAD_DIM, int GQA, int OPT, int KVP>
static int r4d_decode_launch_win(const R4DArgs* a, hipStream_t s) {
    constexpr int TILE = 16, BS = 16, JPB = 2, VECMIN = 256, PF16 = (OPT & O_PF16) ? 1 : 0;
    if (a->head_dim != HEAD_DIM || a->block_size != BS) return -1;
    if (a->q_heads / a->kv_heads != GQA) return -2;
    const int rows = a->q_len * GQA;
    if (rows > 64) return -4;
    const int nw = (rows + 15) / 16;
    const int splits = r4d_decode_splits_win(a, TILE);
#define R4D_WIN_LAUNCH(NW) do { if (a->chain)                                                  \
    hipLaunchKernelGGL((r4dchain_attn_decode<NW, TILE, HEAD_DIM, GQA, BS, KVP, OPT, 1>),        \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, s, *a, splits); \
    else                                                                                      \
    hipLaunchKernelGGL((r4d_attn_decode_kernel<NW, TILE, HEAD_DIM, GQA, BS, KVP, OPT, 1>),      \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, s, *a, splits); \
    } while (0)
    if      (nw == 1) R4D_WIN_LAUNCH(1);
    else if (nw == 2) R4D_WIN_LAUNCH(2);
    else if (nw == 3) R4D_WIN_LAUNCH(3);
    else              R4D_WIN_LAUNCH(4);
#undef R4D_WIN_LAUNCH
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -100 - (int)e;

    const int jobs3 = r4d_q_total(*a) * a->q_heads;
    if (splits <= 64 && jobs3 >= VECMIN && a->chain)
        hipLaunchKernelGGL((r4dchain_attn_splitkv_combine_vec<HEAD_DIM, 4, PF16, JPB, 1>),
                           dim3((jobs3 + JPB - 1) / JPB), dim3(32 * JPB), 0, s, *a, splits, TILE);
    else if (splits <= 64 && jobs3 >= VECMIN)
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_vec_kernel<HEAD_DIM, 4, PF16, JPB, 1>),
                           dim3((jobs3 + JPB - 1) / JPB), dim3(32 * JPB), 0, s, *a, splits, TILE);
    else if (a->chain)
        hipLaunchKernelGGL((r4dchain_attn_splitkv_combine<HEAD_DIM, 4, PF16, 1>), dim3(jobs3),
                           dim3(HEAD_DIM), splits * sizeof(float), s, *a, splits, TILE);
    else
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<HEAD_DIM, 4, PF16, 1>), dim3(jobs3),
                           dim3(HEAD_DIM), splits * sizeof(float), s, *a, splits, TILE);
    e = hipGetLastError();
    return e == hipSuccess ? 0 : -100 - (int)e;
}
