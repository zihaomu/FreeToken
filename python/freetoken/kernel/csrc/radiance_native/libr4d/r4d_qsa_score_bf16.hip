// The QSA indexer's block score: how much one query wants each compressed block of key history.
//
//   score[m][b] = ( sum over heads h of relu( q[m][h] . bk[bt[seq(m)][b]] ) ) / sqrt(n)
//
// The top budget/ratio scores per query select the blocks attention reads (docs/QSA.md;
// r4d_qsa_block_key_bf16.hip builds `bk`).
//
// The relu is per (block, head), then summed over heads, as in the reference
// (`relu(scores).sum(dim=-1)`); summing first would let one head's negative cancel another's
// positive and still rank plausibly.
//
// `bt` holds page ids, not block indices: the attention KV group is paged at `ratio` tokens, so a
// compress block is a page, block keys are stored per page, and the sequence's attention block
// table is the list scored. The top-k is then directly the block table attention reads.
//
// An absent entry scores -inf, not zero: tables are padded, and a real score can be zero.
//
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr int kWave    = 32;
constexpr int kThreads = 256;
constexpr int kWaves   = kThreads / kWave;
constexpr int kMaxH    = 8;               // indexer query heads; 4 on this model
constexpr int kMaxVPL  = 8;               // n / kWave, so n <= 256
// Scalar form: one wave per block, and a workgroup covers a tile of queries, so the key history
// (the only real traffic) is read once per tile instead of once per query row.
//
// kQT = 4 matches a decode row group (1 + n_spec rows at depth 3), so tiles usually align with
// sequences; a tile straddling two sequences is processed per part, reading the keys once per
// part. The tile lives in registers (see the scoring loop).
constexpr int kBPW     = 4;               // blocks a WAVE; a workgroup owns kWaves * kBPW = 32
constexpr int kQT      = 4;               // query rows a workgroup, and the grid's third axis

/* The scoring loop spells out r4d_wave_sum's five butterfly steps so they can be hoisted out of
 * the head loop (see there); same lanes, order and bits. */

// `m` is a query row, not a sequence: a prefill chunk has many queries over one history, each
// selecting its own set (docs/QSA.md, "Per-query selection"). A query finds its sequence's block-table row by
// scanning `cu`.
//
// A lane owns n / 32 consecutive dims of the query heads and the block key, so a head's dot is one
// wave reduction and each block key is read once for all heads and all kQT queries.
//
// `nc` (complete blocks per sequence) bounds the block loop; nb is a declared width sized for
// max_ctx and would mostly score padding. nc is an upper bound for every query in the chunk;
// qsa_select applies the exact per-query bound from `pos`.
template <int VPL, int HEADS>
__global__ __launch_bounds__(kThreads) void qsa_score_kernel(
    const __hip_bfloat16* __restrict__ q, const __hip_bfloat16* __restrict__ bk,
    const int32_t* __restrict__ bt, const int32_t* __restrict__ cu,
    const int32_t* __restrict__ nc, float* __restrict__ out,
    int n, int heads, int nb, int nseq, int q_ld, int bk_ld, int bt_ld, int out_ld) {

    __shared__ int sh_seq;
    const int lane = threadIdx.x & (kWave - 1), wave = threadIdx.x >> 5;
    const float inv = rsqrtf((float)n);
    /* Load width: the widest vector dividing both VPL and the row pitch (a KV cache row pitch
     * need not be a multiple of anything). */
    const int kvec = ((VPL & 3) == 0 && (bk_ld & 3) == 0) ? 4
                   : ((VPL & 1) == 0 && (bk_ld & 1) == 0) ? 2 : 1;
    const int m0 = (int)blockIdx.y * kQT;

    /* The tile walks its queries one sequence at a time (block table row, complete-block count
     * and block bound are per sequence). Usually one iteration. */
    for (int m = m0; m < m0 + kQT; ) {
        /* The sequence of query `m`. Exactly one i matches, so the store is deterministic. A query
         * past the batch matches nothing and ends the tile. */
        if (threadIdx.x == 0) sh_seq = -1;
        __syncthreads();
        for (int i = (int)threadIdx.x; i < nseq; i += kThreads)
            if (m >= cu[i] && m < cu[i + 1]) sh_seq = i;
        __syncthreads();
        const int seq = sh_seq;
        if (seq < 0) return;
        const int mend = cu[seq + 1] < m0 + kQT ? cu[seq + 1] : m0 + kQT;
        const int nq   = mend - m;

        const int nbs = nc[seq] < nb ? nc[seq] : nb;
        /* The block axis is sized from the declared width, so most workgroups on a short context
         * skip here, before the query load. A skip, not a return: a straddling tile may be past
         * the first sequence's blocks but inside the second's. The bound is uniform, so barriers
         * stay matched.
         *
         * The skip path still needs the barrier: the next part starts with thread 0 resetting
         * sh_seq, and a wave that has not read this part's value would leave early. */
        const int b0 = (int)blockIdx.x * (kWaves * kBPW);
        if (nbs <= 0 || b0 >= nbs) {
            m = mend;
            if (m < m0 + kQT) __syncthreads();
            continue;
        }
        const int b1 = (b0 + kWaves * kBPW) < nbs ? (b0 + kWaves * kBPW) : nbs;

        /* The query tile in registers (kQT * HEADS * VPL, 64 at this model's shape), loaded once
         * for all of this workgroup's blocks; in LDS every block would re-read it, and LDS issue
         * is this kernel's bottleneck. A query past the part reads its last row, unscored. */
        float qv[kQT][HEADS][VPL];
#pragma unroll
        for (int mi = 0; mi < kQT; ++mi) {
            const int row = m + (mi < nq ? mi : nq - 1);
#pragma unroll
            for (int h = 0; h < HEADS; ++h) {
                const __hip_bfloat16* qp = q + (size_t)row * q_ld + (size_t)h * n + lane * VPL;
#pragma unroll
                for (int jj = 0; jj < VPL; ++jj) qv[mi][h][jj] = __bfloat162float(qp[jj]);
            }
        }

        /* Issue all of this wave's loads before scoring: each block is two dependent reads (page
         * id, then key), so all page ids are read first, then all keys, instead of 2 * kBPW
         * serial latencies. A block past the end or without a page reads page 0's key (always
         * present) and is skipped below. */
        int   pg[kBPW];
        float kvs[kBPW][VPL];
#pragma unroll
        for (int i = 0; i < kBPW; ++i) {
            const int b = b0 + wave + i * kWaves;
            pg[i] = b < b1 ? bt[(size_t)seq * bt_ld + b] : -1;
        }
#pragma unroll
        for (int i = 0; i < kBPW; ++i) {
            float* kv = kvs[i];
            /* One vector load per lane instead of VPL two-byte loads, each of which would touch
             * the wave's whole 256-byte row (about 145 GB/s on a 621 GB/s part). Same bytes and
             * registers, so the sum is bit-identical. */
            const uint16_t* kp = (const uint16_t*)bk + (size_t)(pg[i] >= 0 ? pg[i] : 0) * bk_ld
                               + lane * VPL;
            if (kvec == 4) {
#pragma unroll
                for (int j = 0; j < VPL; j += 4) {
                    const uint2 v = *(const uint2*)(kp + j);
                    kv[j + 0] = __uint_as_float((v.x & 0xFFFFu) << 16);
                    kv[j + 1] = __uint_as_float(v.x & 0xFFFF0000u);
                    kv[j + 2] = __uint_as_float((v.y & 0xFFFFu) << 16);
                    kv[j + 3] = __uint_as_float(v.y & 0xFFFF0000u);
                }
            } else if (kvec == 2) {
#pragma unroll
                for (int j = 0; j < VPL; j += 2) {
                    const uint32_t v = *(const uint32_t*)(kp + j);
                    kv[j + 0] = __uint_as_float((v & 0xFFFFu) << 16);
                    kv[j + 1] = __uint_as_float(v & 0xFFFF0000u);
                }
            } else {
#pragma unroll
                for (int j = 0; j < VPL; ++j)
                    kv[j] = __uint_as_float((uint32_t)kp[j] << 16);
            }
        }

#pragma unroll
        for (int i = 0; i < kBPW; ++i) {
            const int b = b0 + wave + i * kWaves;
            if (b >= b1) break;
            if (pg[i] < 0) {
                if (lane == 0)
                    for (int mi = 0; mi < nq; ++mi)
                        out[(size_t)(m + mi) * out_ld + b] = -INFINITY;
                continue;
            }
            const float* kv = kvs[i];
#pragma unroll
            for (int mi = 0; mi < kQT; ++mi) {
                if (mi >= nq) break;
                /* Every head's dot first, then one butterfly over all of them: five dependent steps
                 * with HEADS independent shuffles each, instead of a HEADS * 5 chain. Same values in
                 * the same order, so bit-identical. HEADS is a template parameter so dv[] and qv
                 * stay in registers. */
                float dv[HEADS];
#pragma unroll
                for (int h = 0; h < HEADS; ++h) {
                    float d = 0.0f;
#pragma unroll
                    for (int j = 0; j < VPL; ++j) d = fmaf(qv[mi][h][j], kv[j], d);
                    dv[h] = d;
                }
#pragma unroll
                for (int h = 0; h < HEADS; ++h) dv[h] += swap16(dv[h]);       /* xor 16 */
#pragma unroll
                for (int h = 0; h < HEADS; ++h) dv[h] = r4d_dpp_xor_add<0x168>(dv[h]);   /* xor 8 */
#pragma unroll
                for (int h = 0; h < HEADS; ++h) dv[h] = r4d_dpp_xor_add<0x164>(dv[h]);   /* xor 4 */
#pragma unroll
                for (int h = 0; h < HEADS; ++h) dv[h] = r4d_dpp_xor_add<0x162>(dv[h]);   /* xor 2 */
#pragma unroll
                for (int h = 0; h < HEADS; ++h) dv[h] = r4d_dpp_xor_add<0x161>(dv[h]);   /* xor 1 */
                /* Every lane holds the full reduction; lane 0 stores. */
                float s = 0.0f;
#pragma unroll
                for (int h = 0; h < HEADS; ++h) s += fmaxf(dv[h], 0.0f);
                if (lane == 0) out[(size_t)(m + mi) * out_ld + b] = s * inv;
            }
        }
        m = mend;
        /* The next part resets sh_seq, so all reads of it must be done. */
        __syncthreads();
    }
}

/* ================================================================== the matrix form
 *
 * The same score as a matrix product. Per head, 16 queries x 16 blocks is one 16x16 tile of
 * K_tile . Q_h^T (eight bf16 WMMA steps over 128 dims); a lane then holds one query against eight
 * consecutive blocks of every head's tile, so relu and head sum happen in registers.
 *
 * The scalar form reads the key history once per four query rows and does a lane reduction per
 * head per block, so it is linear in M (a 2048-row chunk at 80K context: 9.3 ms per call). Here
 * the history is read once per 16 queries with no reductions. It is used for every batch it fits,
 * since it was faster at every measured shape except a single short-context query (by ~0.25 us),
 * e.g. at 200K context: M=1 29.9 -> 12.7 us, M=32 437 -> 27.2 us, M=2048 27.7 -> 1.36 ms.
 *
 * One form for all batch sizes also keeps a score a function of its query: the two forms sum in
 * different orders and differ in the last bit (rel_l2 ~4e-8), which could flip an exact tie.
 *
 * One wave owns 16 query rows (in registers: HEADS x 8 fragments, 128 VGPRs at four heads) and a
 * range of `tpw` block tiles, each one page-id read and eight fragment loads. A workgroup is four
 * waves on consecutive block ranges of the same query rows, sharing them through the WGP cache.
 * No LDS, no barrier. A tile with rows of two sequences is scored per sequence, as in the scalar
 * form (decode rows next to a prefill chunk). */
constexpr int kMxWaves = 4;     // waves a workgroup, each on its own block range
constexpr int kMxQ     = 16;    // query rows a wave: the WMMA tile's N
constexpr int kMxKS    = 8;     // 16-wide K steps: n = 128, the only width this form is built for
/* Target grid size in waves for sizing a wave's block range: a couple of rounds of resident
 * capacity (64 CUs x 4 SIMDs x ~6 waves at this form's register count) ... */
constexpr long long kMxWavesTarget = 4096;
/* ... and at least this many tiles per wave, since loading the 16 query rows (16 KiB at four
 * heads) would otherwise dominate. */
constexpr long long kMxMinTiles    = 4;

#define QSA_SCORE_MX_PARAMS                                                                     \
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ bk,                \
    const int32_t* __restrict__ bt, const int32_t* __restrict__ cu,                             \
    const int32_t* __restrict__ nc, float* __restrict__ out, int M, int nb, int nseq,           \
    int tpw, int q_ld, int bk_ld, int bt_ld, int out_ld, int vec
template <int HEADS>
__device__ __forceinline__ void qsa_score_mx_body(QSA_SCORE_MX_PARAMS) {

    const int lane = threadIdx.x & (kWave - 1), wave = threadIdx.x >> 5;
    const int half = lane >> 4, col = lane & 15;
    const int m0 = (int)blockIdx.y * kMxQ;
    const int wb0 = ((int)blockIdx.x * kMxWaves + wave) * tpw * 16;
    if (m0 >= M || wb0 >= nb) return;       /* wave-uniform, and nothing below synchronises */
    const float inv = rsqrtf((float)(kMxKS * 16));

    /* B operand: query row `col` of the tile, every head and K step. A row past the batch reads
     * the last row and is never stored. */
    v8bf qf[HEADS][kMxKS];
    {
        const int qr = m0 + col < M ? m0 + col : M - 1;
        const unsigned short* qp = q + (size_t)qr * q_ld + 4 * half;
#pragma unroll
        for (int h = 0; h < HEADS; ++h)
#pragma unroll
            for (int ks = 0; ks < kMxKS; ++ks) qf[h][ks] = fragP(qp + h * kMxKS * 16 + ks * 16);
    }

    const int mlim = m0 + kMxQ < M ? m0 + kMxQ : M;
    const int row = m0 + col;
    for (int m = m0; m < mlim; ) {
        /* The sequence of query `m`: exactly one entry of `cu` brackets it. */
        int seq = -1;
        for (int i0 = 0; i0 < nseq; i0 += kWave) {
            const int i = i0 + lane;
            const unsigned long long hit = __ballot(i < nseq && m >= cu[i] && m < cu[i + 1]);
            if (hit) { seq = i0 + __ffsll((long long)hit) - 1; break; }
        }
        if (seq < 0) return;                /* past the batch: so is every later row */
        const int mend = cu[seq + 1] < mlim ? cu[seq + 1] : mlim;
        const bool live = row >= m && row < mend;
        const int nbs  = nc[seq] < nb ? nc[seq] : nb;
        const int bend = wb0 + tpw * 16 < nbs ? wb0 + tpw * 16 : nbs;
        const int32_t* tab = bt + (size_t)seq * bt_ld;

        /* Read the next tile's page id while this tile's key loads, hiding one dependent latency
         * per tile. */
        int pg_next = wb0 + col < bend ? tab[wb0 + col] : -1;
        for (int b0 = wb0; b0 < bend; b0 += 16) {
            /* A operand: block b0 + col's key. A block past the end or without a page reads page
             * 0's key (always present) and is masked or skipped at the store. */
            const int b  = b0 + col;
            const int pg = pg_next;
            const unsigned short* kp = bk + (size_t)(pg >= 0 ? pg : 0) * bk_ld + 4 * half;
            v8bf kf[kMxKS];
#pragma unroll
            for (int ks = 0; ks < kMxKS; ++ks) kf[ks] = fragP(kp + ks * 16);
            pg_next = b + 16 < bend ? tab[b + 16] : -1;
            /* Which of the tile's sixteen blocks have no page: lanes 0-15 each read one. */
            const unsigned absent = (unsigned)__ballot(pg < 0) & 0xFFFFu;

            v8f acc[HEADS];
#pragma unroll
            for (int h = 0; h < HEADS; ++h)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[h][e] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < kMxKS; ++ks)
#pragma unroll
                for (int h = 0; h < HEADS; ++h) acc[h] = mma(kf[ks], qf[h][ks], acc[h]);

            /* D[block][query]: this lane holds query `col` against blocks b0 + 8*half + e. */
            if (!live) continue;
            float v[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float s = 0.0f;
#pragma unroll
                for (int h = 0; h < HEADS; ++h) s += fmaxf(acc[h][e], 0.0f);
                v[e] = ((absent >> (8 * half + e)) & 1u) ? -INFINITY : s * inv;
            }
            float* op = out + (size_t)row * out_ld + b0 + 8 * half;
            if (vec && b0 + 16 <= bend) {
                *(float4*)op       = make_float4(v[0], v[1], v[2], v[3]);
                *(float4*)(op + 4) = make_float4(v[4], v[5], v[6], v[7]);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    if (b0 + 8 * half + e < bend) op[e] = v[e];
            }
        }
        m = mend;
    }
}

/* The kernel around the body, whose blocks and threads return where their work ends: CH `r4dchain`
 * is the chained form (r4d_chain.h). */
template <int HEADS, class CH>
__global__ __launch_bounds__(kMxWaves * kWave)
void qsa_score_mx_kernel(QSA_SCORE_MX_PARAMS) {
    if (CH::on) r4d_chain_wait();
    qsa_score_mx_body<HEADS>(q, bk, bt, cu, nc, out, M, nb, nseq, tpw, q_ld, bk_ld, bt_ld, out_ld,
                             vec);
    if (CH::on) r4d_chain_done();
}
#undef QSA_SCORE_MX_PARAMS

/* Form override for tests (r4d_selftest compares the two): 0 = automatic, 1 = scalar,
 * 2 = matrix. Like r4d_gemm_fp8a8_pin_mw. */
int g_qsa_form = 0;

}  // namespace

extern "C" void r4d_qsa_score_pin_form(int form) { g_qsa_form = form; }

//   q    [M][heads*n]    bf16, `q_ld` elements a row (0 = heads*n); M is query rows
//   bk   [pages][n]      bf16, `bk_ld` elements a row (0 = n)
//   bt   [nseq][nb]      i32, `bt_ld` elements a row (0 = nb); negative = no block
//   cu   [nseq+1]        i32, cumulative query counts: which sequence a query row belongs to
//   nc   [nseq]          i32, complete blocks a sequence; the block loop's bound
//   out  [M][nb]         f32, `out_ld` elements a row (0 = nb)
static int r4d_qsa_score_bf16_emit(long q, long bk, long bt, long cu, long nc, long out,
                                  int M, int n, int heads, int nb, int nseq,
                                  int q_ld, int bk_ld, int bt_ld, int out_ld, RadStream stream,
                                  bool chain) {
    int emit_rc = RAD_OK;
    if (!q || !bk || !bt || !cu || !nc || !out) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || n <= 0 || heads <= 0 || nb <= 0 || nseq <= 0) return R4D_NO(RAD_E_SHAPE);
    /* A lane owns n/32 dims, at most 8. */
    if (n % kWave || n / kWave > kMaxVPL || heads > kMaxH) return R4D_NO(RAD_E_SHAPE);
    if (!q_ld) q_ld = heads * n;
    if (!bk_ld) bk_ld = n;
    if (!bt_ld) bt_ld = nb;
    if (!out_ld) out_ld = nb;
    /* Grid axes are sized from declared widths (live counts are device data); surplus workgroups
     * exit early. The matrix form is used whenever it fits (n = 128, at most four heads, so the
     * query rows fit in registers); the scalar form serves the rest. */
    const bool mx_fits = n == kMxKS * 16 && heads <= 4;
    if (g_qsa_form == 2 && !mx_fits) return R4D_NO(RAD_E_SHAPE);
    const bool mx = g_qsa_form == 1 ? false : mx_fits;
    if (mx) {
        /* Tiles per wave: about kMxWavesTarget waves in the grid, at least kMxMinTiles each. */
        const long long qt = (M + kMxQ - 1) / kMxQ, tiles = (nb + 15) / 16;
        long long tpw = (tiles * qt + kMxWavesTarget - 1) / kMxWavesTarget;
        if (tpw < kMxMinTiles) tpw = kMxMinTiles;
        const unsigned gx = (unsigned)((tiles + tpw * kMxWaves - 1) / (tpw * kMxWaves));
        const int vec = (out_ld % 4 == 0) && ((out & 15) == 0);
#define RAD_QSA_MX_E(H, CH_)                                                                      \
        emit_rc = R4D_EMIT(stream, (qsa_score_mx_kernel<H, CH_>), dim3(gx, (unsigned)qt),         \
                           dim3(kMxWaves * kWave), 0, (const unsigned short*)q,                   \
                           (const unsigned short*)bk,                                             \
                           (const int32_t*)bt, (const int32_t*)cu, (const int32_t*)nc,            \
                           (float*)out, M, nb, nseq, (int)tpw, q_ld, bk_ld, bt_ld, out_ld, vec)
#define RAD_QSA_MX(H)                                                                             \
        do { if (chain) RAD_QSA_MX_E(H, r4dchain); else RAD_QSA_MX_E(H, R4dUnchained); } while (0)
        switch (heads) {
            case 1: RAD_QSA_MX(1); break;
            case 2: RAD_QSA_MX(2); break;
            case 3: RAD_QSA_MX(3); break;
            case 4: RAD_QSA_MX(4); break;
            default: return R4D_NO(RAD_E_SHAPE);
        }
#undef RAD_QSA_MX
#undef RAD_QSA_MX_E
        return emit_rc;
    }
    const unsigned nbg = (unsigned)((nb + kWaves * kBPW - 1) / (kWaves * kBPW));
    const unsigned nmg = (unsigned)((M + kQT - 1) / kQT);
    /* Dims per lane (n / 32) are a template parameter: as a runtime value the dot is a rolled loop
     * and the block key a runtime-indexed array in scratch. The instantiations cover every n and
     * head count the checks admit (n % 32 == 0, n <= 256, heads <= 8). */
#define RAD_QSA_GO(V, H)                                                                       \
    emit_rc = R4D_EMIT(stream, (qsa_score_kernel<V, H>), dim3(nbg, nmg), dim3(kThreads), 0,      \
                       (const __hip_bfloat16*)q, (const __hip_bfloat16*)bk, (const int32_t*)bt, \
                       (const int32_t*)cu, (const int32_t*)nc,                                  \
                       (float*)out, n, heads, nb, nseq, q_ld, bk_ld, bt_ld, out_ld)
#define RAD_QSA_HEADS(V)                                        \
    switch (heads) {                                            \
        case 1: RAD_QSA_GO(V, 1); break;                        \
        case 2: RAD_QSA_GO(V, 2); break;                        \
        case 3: RAD_QSA_GO(V, 3); break;                        \
        case 4: RAD_QSA_GO(V, 4); break;                        \
        case 5: RAD_QSA_GO(V, 5); break;                        \
        case 6: RAD_QSA_GO(V, 6); break;                        \
        case 7: RAD_QSA_GO(V, 7); break;                        \
        case 8: RAD_QSA_GO(V, 8); break;                        \
        default: return R4D_NO(RAD_E_SHAPE);                    \
    }
    switch (n / kWave) {
        case 1: RAD_QSA_HEADS(1); break;
        case 2: RAD_QSA_HEADS(2); break;
        case 3: RAD_QSA_HEADS(3); break;
        case 4: RAD_QSA_HEADS(4); break;
        case 5: RAD_QSA_HEADS(5); break;
        case 6: RAD_QSA_HEADS(6); break;
        case 7: RAD_QSA_HEADS(7); break;
        case 8: RAD_QSA_HEADS(8); break;
        default: return R4D_NO(RAD_E_SHAPE);
    }
#undef RAD_QSA_HEADS
#undef RAD_QSA_GO
    return emit_rc;
}

// ============================================================== the plugin ABI entry point

enum { QS_Q = 0, QS_BK = 1, QS_BT = 2, QS_CU = 3, QS_NC = 4, QS_OUT = 5 };

extern "C" int r4d_rad_qsa_score(const RadArgs* a, RadStream stream) {
    const RadTensor* q  = r4d_opt(a, QS_Q);
    const RadTensor* bk = r4d_opt(a, QS_BK);
    const RadTensor* bt = r4d_opt(a, QS_BT);
    const RadTensor* cu = r4d_opt(a, QS_CU);
    const RadTensor* nc = r4d_opt(a, QS_NC);
    const RadTensor* o  = r4d_opt(a, QS_OUT);
    if (!q || !bk || !bt || !cu || !nc || !o) return R4D_NO(RAD_E_INVAL);
    /* bk may be a rank-4 KV cache (see r4d_qsa_block_key_bf16.hip); a page holds n contiguous
     * values either way. */
    if (bk->rank < 2 || bt->rank != 2 || o->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long n     = r4d_geti_or(a, "n", 0);
    const long long heads = r4d_geti_or(a, "heads", 0);
    if (n <= 0 || heads <= 0) return R4D_NO(RAD_E_INVAL);
    const long long M = o->shape[0];
    /* nb = min(table width, score-plane width). Both are sized for max_ctx / ratio (by the KV
     * manager and the architecture respectively, and may round differently), so the minimum is
     * never below the live count. qsa_select bounds its reads by nc regardless. */
    const long long nb = bt->shape[1] < o->shape[1] ? bt->shape[1] : o->shape[1];
    if (M < 1 || nb < 1) return R4D_NO(RAD_E_SHAPE);
    /* The sequence count comes from cu, not the table height (max_seqs), as in qsa_work and
     * qsa_tail_store. */
    if (cu->rank != 1 || cu->shape[0] < 2) return R4D_NO(RAD_E_SHAPE);
    const long long nseq = cu->shape[0] - 1;
    if (bt->shape[0] < nseq || r4d_numel(nc) < nseq) return R4D_NO(RAD_E_SHAPE);
    if (bk->shape[0] < 1 || r4d_numel(bk) / bk->shape[0] != n) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(q) < M * heads * n) return R4D_NO(RAD_E_SHAPE);
    const long long qp = r4d_row_pitch(q), kp = r4d_row_pitch(bk);
    const long long tp = r4d_row_pitch(bt), op = r4d_row_pitch(o);
    if (qp < 0 || kp < 0 || tp < 0 || op < 0) return R4D_NO(RAD_E_STRIDE);
    return r4d_qsa_score_bf16_emit(r4d_addr(q), r4d_addr(bk), r4d_addr(bt), r4d_addr(cu),
                                   r4d_addr(nc), r4d_addr(o),
                                   (int)M, (int)n, (int)heads, (int)nb, (int)nseq,
                                   (int)qp, (int)kp, (int)tp, (int)op, stream,
                                   r4d_geti_or(a, "chain", 0) != 0);
}

