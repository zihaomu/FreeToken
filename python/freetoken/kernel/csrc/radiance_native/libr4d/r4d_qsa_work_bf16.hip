// The QSA indexer's work list: which blocks complete this step, where their raw keys are, and
// where their block key goes.
//
// qsa_block_key reads each block's `ratio` raw keys from consecutive rows of one plane, but a
// block completing at decode has its keys partly in the tail and partly in this step, and blocks
// belong to different sequences. This op finds the completed blocks, gathers each one's raw keys
// into consecutive rows of a staging plane, and writes its destination page and first-token
// position: exactly qsa_block_key's three operands (docs/QSA.md). The copy is small (1 KiB per
// completing block) and keeps qsa_block_key single-source.
//
// The tail is a per-layer linear KV state ([slots, 1, ring, n]; `sidx` names the sequence's slot,
// not the step-local batch row); see r4d_qsa_tail_store_bf16.hip, including why ring is
// ratio + n_spec. Keys are read at row q % ring to match.
//
// The launch is bounded on the host (ceil(T / ratio) + n_seq: one block per `ratio` tokens plus
// one straddling block per sequence). Work items past the real count get page = -1, which
// qsa_block_key skips.
//
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_emit.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 128;
/* The sequence loop's bound, checked by the launcher against the real count from cu_seqlens. Not a
 * row constraint on the declared seqs (rad-convert declares 256, which must still resolve). */
constexpr int kMaxSeq  = 256;

#define QSA_WORK_PARAMS                                                                         \
    const uint16_t* __restrict__ k, const int32_t* __restrict__ pos,                            \
    const int32_t* __restrict__ cu, const uint16_t* __restrict__ state,                         \
    const int32_t* __restrict__ sidx, const int32_t* __restrict__ bt,                           \
    uint16_t* __restrict__ stage, int32_t* __restrict__ page, int32_t* __restrict__ bpos,       \
    int32_t* __restrict__ nc, int n, int ratio, int ring, int seqs, int max_work, int k_ld,     \
    int s_slot, int s_row, int sidx_ld, int bt_ld, int g_ld,                                    \
    const int32_t* __restrict__ rope, int rplane, int bstep
__device__ __forceinline__ void qsa_work_body(QSA_WORK_PARAMS) {

    const int w = (int)blockIdx.x;
    /* Complete blocks per sequence (needed by the scorer and the selection), written by the first
     * `seqs` workgroups instead of a separate launch. */
    if (w < seqs && threadIdx.x == 0) {
        const int t1 = cu[w + 1];
        nc[w] = t1 > cu[w] ? (pos[t1 - 1] + 1) / ratio : 0;
    }
    if (w >= max_work) return;

    /* The sequence owning work item w, by a running count recomputed in every workgroup (seqs is
     * small; a prefix would need a second launch). */
    int s = -1, j = 0;
    int run = 0;
    for (int i = 0; i < seqs && i < kMaxSeq; ++i) {
        const int t0 = cu[i], t1 = cu[i + 1];
        if (t1 <= t0) continue;
        const int p0 = pos[t0], p1 = pos[t1 - 1];
        /* A block completes this step when its last token, b * ratio + ratio - 1, is in [p0, p1].
         * Both bounds use non-negative numerators: C division truncates toward zero, so
         * (p1 - (ratio - 1)) / ratio would be 0, not -1, for p1 < ratio - 1. */
        const int lo = p0 / ratio;                                  /* ceil((p0-ratio+1)/ratio) */
        const int hi = (p1 + 1) / ratio - 1;                        /* floor((p1-ratio+1)/ratio) */
        const int cnt = hi >= lo ? hi - lo + 1 : 0;
        if (w < run + cnt) { s = i; j = w - run + lo; break; }
        run += cnt;
    }
    if (s < 0) {
        /* Past the real count: page = -1 makes the host's bound safe. */
        if (threadIdx.x == 0) {
            page[w] = -1;
            for (int c = 0; c < (rope ? 3 : 1); ++c) bpos[(size_t)w * bstep + c] = 0;
        }
        return;
    }

    const int b  = j;                       /* the block index within the sequence */
    const int t0 = cu[s];
    const int p0 = pos[t0];
    /* Column 0 is the committed slot; -1 marks a sequence with no state, whose blocks cannot be
     * built (page -1, no gather). */
    const int si = sidx[(size_t)s * sidx_ld];

    if (threadIdx.x == 0) {
        page[w] = si < 0 ? -1 : bt[(size_t)s * bt_ld + b];
        if (!rope) {
            bpos[(size_t)w * bstep] = b * ratio;
        } else {
            /* Rotary position of the block's first token, three components side by side, items
             * bstep apart. A first token in this step reads its own components. One from an
             * earlier step is text (only a decode row or a prompt's last chunk leaves a block open,
             * and a prompt ends in text), so it rotates at its index plus the shift carried by
             * this step's first row of the sequence. */
            const int f = b * ratio;
            for (int c = 0; c < 3; ++c)
                bpos[(size_t)w * bstep + c] =
                    f >= p0 ? rope[(size_t)c * rplane + t0 + (f - p0)] : f + (rope[t0] - p0);
        }
    }
    if (si < 0) return;

    /* The block's raw keys, each from this step or from the tail. */
    for (int i = 0; i < ratio; ++i) {
        const int q = b * ratio + i;                 /* this key's position */
        const uint16_t* src;
        if (q >= p0) src = k + (size_t)(t0 + (q - p0)) * k_ld;
        else         src = state + (size_t)si * s_slot + (size_t)(q % ring) * s_row;
        uint16_t* dst = stage + (size_t)(w * ratio + i) * g_ld;
        for (int d = (int)threadIdx.x; d < n; d += kThreads) dst[d] = src[d];
    }
}

/* The kernel around the body, whose blocks and threads return where their work ends: CH `r4dchain`
 * is the chained form (r4d_chain.h). */
template <class CH>
__global__ __launch_bounds__(kThreads)
void qsa_work_kernel(QSA_WORK_PARAMS) {
    if (CH::on) r4d_chain_wait();
    qsa_work_body(k, pos, cu, state, sidx, bt, stage, page, bpos, nc, n, ratio, ring, seqs,
                  max_work, k_ld, s_slot, s_row, sidx_ld, bt_ld, g_ld, rope, rplane, bstep);
    if (CH::on) r4d_chain_done();
}
#undef QSA_WORK_PARAMS

}  // namespace

//   k      [T][n]                  bf16, this step's raw index keys
//   pos    [T]                     i32
//   cu     [seqs + 1]              i32
//   state  [slots][ring][n]        bf16, the tail group's cache for this layer
//   sidx   [seqs][sidx_ld]         i32, the tail group's state row; column 0 is the slot
//   bt     [seqs][bt_ld]           i32, the block-key group's block table
//   stage  [max_work * ratio][n]   bf16
//   page   [max_work]              i32, -1 past the real count
//   bpos   [max_work][bstep]       i32, a work item's first-token position at [w * bstep]
//   rope   [3][rplane]             i32, optional: this step's rotary components (RadBatch::
//                                  rope_pos); present, a work item's three components are
//                                  bpos[w * bstep + 0..2]
static int r4d_qsa_work_bf16_emit(long k, long pos, long cu, long state, long sidx, long bt,
                                 long stage, long page, long bpos, long nc, int n, int ratio,
                                 int ring, int seqs, int max_work, int k_ld, int s_slot,
                                 int s_row, int sidx_ld, int bt_ld, int g_ld, long rope,
                                 int rplane, int bstep, RadStream stream, bool chain) {
    if (!k || !pos || !cu || !state || !sidx || !bt || !stage || !page || !bpos || !nc)
        return R4D_NO(RAD_E_INVAL);
    if (n <= 0 || ratio <= 0 || ring < ratio || seqs <= 0 || max_work <= 0 || seqs > kMaxSeq)
        return R4D_NO(RAD_E_SHAPE);
    if (!k_ld) k_ld = n;
    if (!s_row) s_row = n;
    if (!g_ld) g_ld = n;
    if (!sidx_ld) sidx_ld = 1;
    /* Each form named at its own launch: a test reads the kernel a launch site names. */
#define R4D_QWORK(CH_)                                                                        \
    R4D_EMIT(stream, qsa_work_kernel<CH_>, dim3((unsigned)max_work), dim3(kThreads), 0,       \
             (const uint16_t*)k, (const int32_t*)pos, (const int32_t*)cu,                     \
             (const uint16_t*)state, (const int32_t*)sidx, (const int32_t*)bt,                \
             (uint16_t*)stage, (int32_t*)page, (int32_t*)bpos, (int32_t*)nc, n, ratio, ring,  \
             seqs, max_work, k_ld, s_slot, s_row, sidx_ld, bt_ld, g_ld, (const int32_t*)rope, \
             rplane, bstep)
    const int rc = chain ? R4D_QWORK(r4dchain) : R4D_QWORK(R4dUnchained);
#undef R4D_QWORK
    return rc;
}

// ============================================================== the plugin ABI entry point

enum { QW_K = 0, QW_POS, QW_CU, QW_STATE, QW_SIDX, QW_BT, QW_ROPE, QW_STAGE, QW_PAGE, QW_BPOS,
       QW_NC };

extern "C" int r4d_rad_qsa_work(const RadArgs* a, RadStream stream) {
    const RadTensor* k  = r4d_opt(a, QW_K);
    const RadTensor* po = r4d_opt(a, QW_POS);
    const RadTensor* cu = r4d_opt(a, QW_CU);
    const RadTensor* tl = r4d_opt(a, QW_STATE);
    const RadTensor* sx = r4d_opt(a, QW_SIDX);
    const RadTensor* bt = r4d_opt(a, QW_BT);
    const RadTensor* rp = r4d_opt(a, QW_ROPE);       /* optional: multi-component positions */
    const RadTensor* st = r4d_opt(a, QW_STAGE);
    const RadTensor* pg = r4d_opt(a, QW_PAGE);
    const RadTensor* bp = r4d_opt(a, QW_BPOS);
    const RadTensor* nc = r4d_opt(a, QW_NC);
    if (!k || !po || !cu || !tl || !sx || !bt || !st || !pg || !bp || !nc)
        return R4D_NO(RAD_E_INVAL);
    if (k->rank != 2 || bt->rank != 2 || st->rank != 2) return R4D_NO(RAD_E_SHAPE);
    /* The tail state arrives as its group declared it: [slots, heads, ring, n], heads 1. See
     * r4d_qsa_tail_store_bf16.hip. */
    if (tl->rank != 4 || tl->shape[1] != 1) return R4D_NO(RAD_E_SHAPE);
    if (tl->stride[3] != 1) return R4D_NO(RAD_E_STRIDE);
    /* bt is the block-key group's table, not the attention group's; both are paged at `ratio`,
     * so they share block indices (docs/QSA.md). */
    const long long n     = r4d_geti_or(a, "n", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    const long long ring  = r4d_geti_or(a, "ring", 0);
    if (n <= 0 || ratio <= 0 || ring < ratio) return R4D_NO(RAD_E_INVAL);
    const long long seqs = r4d_numel(cu) - 1;
    const long long T    = k->shape[0];
    if (seqs < 1 || T < 1 || r4d_numel(po) < T) return R4D_NO(RAD_E_SHAPE);
    if (k->shape[1] != n || st->shape[1] != n) return R4D_NO(RAD_E_SHAPE);
    if (tl->shape[2] < ring || tl->shape[3] != n) return R4D_NO(RAD_E_SHAPE);
    if (bt->shape[0] < seqs) return R4D_NO(RAD_E_SHAPE);
    const long long sxp = sx->rank >= 2 ? r4d_row_pitch(sx) : 1;
    if (sxp < 1 || r4d_dim(sx, 0) < seqs) return R4D_NO(RAD_E_SHAPE);
    /* The work count comes from `page`, and `stage` must hold ratio rows per item. */
    const long long mw = r4d_numel(pg);
    if (mw < 1 || st->shape[0] < mw * ratio) return R4D_NO(RAD_E_SHAPE);
    /* bpos is one position per work item, or a rank-2 [>= work, 1 or 3] column slice of a
     * [max_work, 3] buffer (components side by side, items a row pitch apart); three columns are
     * required with `rope`. rope is [3, rplane] component-major, rplane >= the step's tokens. */
    long long bstep = 1;
    if (bp->rank == 2) {
        if (bp->shape[0] < mw || bp->stride[1] != 1 || bp->stride[0] < bp->shape[1] ||
            (rp && bp->shape[1] < 3))
            return R4D_NO(RAD_E_SHAPE);
        bstep = bp->stride[0];
    } else if (rp || r4d_numel(bp) < mw) {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (rp && (rp->rank != 2 || rp->shape[0] < 3 || rp->shape[1] < T || !r4d_contig(rp) ||
               rp->dtype != RAD_I32))
        return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(nc) < seqs) return R4D_NO(RAD_E_SHAPE);
    /* nc is written by the first `seqs` workgroups, so the launch must be at least that wide. */
    if (mw < seqs) return R4D_NO(RAD_E_SHAPE);
    const long long kp = r4d_row_pitch(k);
    const long long bp2 = r4d_row_pitch(bt), sp = r4d_row_pitch(st);
    if (kp < 0 || bp2 < 0 || sp < 0) return R4D_NO(RAD_E_STRIDE);
    if (k->dtype != RAD_BF16 || tl->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    return r4d_qsa_work_bf16_emit(r4d_addr(k), r4d_addr(po), r4d_addr(cu), r4d_addr(tl),
                                  r4d_addr(sx), r4d_addr(bt), r4d_addr(st), r4d_addr(pg),
                                  r4d_addr(bp), r4d_addr(nc), (int)n, (int)ratio, (int)ring,
                                  (int)seqs, (int)mw, (int)kp, (int)tl->stride[0],
                                  (int)tl->stride[2], (int)sxp, (int)bp2, (int)sp,
                                  r4d_addr(rp), rp ? (int)rp->shape[1] : 0, (int)bstep, stream,
                                  r4d_geti_or(a, "chain", 0) != 0);
}

