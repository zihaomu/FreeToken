// R4D prefill. Transposed core: S^T = K.Q^T, one lane owns one query row, ROWS = 16*NWARPS,
// 128-VGPR accumulator. The query is bf16 and the KV cache is fp8 or bf16; the 16-bit operand
// format for both matmuls is f16 (see r4d_dt16.h for why).
//
// Two families: all-16-bit (bf16 KV) and 16/8 mixed (bf16 query, fp8, fp8_v4 or k6_v4 KV). The
// mixed family also has lossy e4m3 legs (QK8, PV8), launched only by the rows a caller's
// `prefill_fp8` selects.
//
// Each option below can be turned off on its own after a geometry change:
//
//   LAZY  (always on) Softmax against a reference max instead of a running one:
//         p = 2^(s - m_ref + SHIFT), with m_ref updated and the 128-VGPR accumulator rescaled only
//         when a row max exceeds m_ref by more than the P format absorbs (DT16::GROW octaves),
//         instead of on every m-tile. Same relative weighting in an f32 accumulator.
//   F16   f16 as the 16-bit operand format. gfx1201 has no bf16 convert (software RTNE is ~6 VALU);
//         v_cvt_pkrtz_f16_f32 converts two values in one. Exact for these operands (fp8 has 3
//         mantissa bits, bf16 7, f16 10).
//   MSKIP Causal masking only on m-tiles that straddle the limit.
//   FOLDQ scale * k_descale * log2(e) folded into the query once at load.
//   CONTIG The WMMA k axis is only contracted, so relabelling it on both operands is free. Reading
//         eight contiguous elements at 8*h gives one aligned ds_read_b128 per fragment, removes 16
//         address adds and makes the P fragment lane-local.
//   PF/SGB Load PF fragments before the matching PF WMMAs, with a scheduling barrier between the
//         groups. Without the barrier PF has no effect.
//   DOT2  v_dot2_f32_f16 row sum from the packed f16 P: 4 instructions instead of 8, and the
//         denominator uses the p values the PV matmul used.
//   BTS/KWIDE/GPREV/KLATE  tile staging; see each define.
//   QK8   Lossy. Q rounded to e4m3 once at load, after the folded scale; K as e4m3 bytes in LDS
//         (half the tile) and Q.K^T runs as fp8 WMMA at twice the f16 rate.
//   PV8   Lossy. P rounded to e4m3 and V as e4m3 bytes in LDS, P.V as fp8 WMMA. GROW is 8
//         octaves (e4m3's largest value is 448 = 2^8.8), and DOT2 is off: the denominator sums the
//         p values before rounding.
//         The LDS bytes are the cache's own for the fp8 cache and fp8_v4's K. k6_v4's K and the
//         int4 V of fp8_v4/k6_v4 are widened to value * group scale, as for the f16 legs, and
//         rounded once to e4m3 at staging. QK8 and PV8 need an 8-bit cache and CONTIG; a
//         static_assert holds both.
#include "r4d.h"
#include "r4d_common.h"
#include "r4d_dt16.h"

#define O_F16     1
#define O_MSKIP   2
#define O_QK8    32
#define O_PV8    64
#define O_FOLDQ   8
#define O_CONTIG  512
#define O_PF2    1024
#define O_PF4    2048
#define O_PF8    3072
#define O_SGB    4096
#define O_DOT2  16384
// BTS: a tile's TILE/BS block-table entries are at a wave-uniform index, so load them into SGPRs
// once per tile. Otherwise each staging pass does two serialised memory round trips (block table,
// then KV) inside a barrier-bounded phase with nothing else to run.
#define O_BTS   262144
// KWIDE: one 16-byte K fetch per thread instead of two 8-byte ones. 768 threads x 16 B is exactly
// a 48-key fp8 tile, so staging is straight-line with half the loads and block-table lookups.
// key = tid/16 would make a warp write one row at a 32-byte stride (4-way LDS bank conflict, since
// 8*ch mod 32 takes four values). key = 2*warp + (lane&1), ch = (lane&31)>>1 interleaves two keys
// so the writes cover all 32 banks, while global reads stay two contiguous 256-byte runs.
#define O_KWIDE 524288
// GPREV: fetch tile ti+1's V into registers right after the second barrier, so its latency is
// hidden by tile ti's 96 WMMAs and only the LDS stores remain between barriers. V only: at
// 240 VGPRs a K prefetch would be spilled, and a spilled prefetch is waited on immediately.
#define O_GPREV 1048576
// LDSB: an LDS-scoped barrier instead of __syncthreads() (lds_barrier() in r4d_common.h). The
// default fence invalidates the vector cache twice per tile, though only LDS needs ordering.
#define O_LDSB  2097152
// KLATE: on the fp8 and k6_v4 caches (the wide K fetch), tile ti+1's K is fetched once tile ti's
// last m-tile has its scores (its K fragments are dead there, so kreg adds no live range across the
// WMMAs), and the staging between the barriers only widens and stores. Otherwise every wave of the
// workgroup (the only one on its WGP) waits out a global round trip each tile. With the branch-free
// K fetch and the V scales packed at the store, k6_v4 over 100k keys runs 25.96 -> 21.48 ms (2048
// queries, r4d_selftest --perfpaged) and fp8 24.6 -> 22.7; the gain is largest once the KV no
// longer fits the 64 MB Infinity Cache.
#define O_KLATE 4194304
// O_V4: the fp8_v4 cache, O_V4 | O_K6 the k6_v4 cache (see r4d_attn_decode_h256_gqa6.hip).
#define O_V4    33554432
#define O_K6    67108864
// O_I8 (with O_V4 | O_K6): the i8 cache (see r4d_attn_decode_h256_gqa6.hip).
#define O_I8    1073741824
// The int8 cache's bits together, for a launcher's option set.
#define O_KVI8  (O_V4 | O_K6 | O_I8)
// O_PSPLIT: a KV split for launches with too few workgroups to fill the card (r4d_prefill_splits
// in r4d_attn_prefill_split.h). blockIdx.x is qtile * a.splits + split, and the workgroup walks
// only its segment of the sequence's tiles, cut as r4d_attn_tps cuts them (the partition the
// split-KV merge recomputes from ctx), clipped at its causal end. Instead of bf16 out it stores the
// unnormalised f32 O, m_ref and l in the decode kernel's partial layout (r4d_attn_decode_body,
// !PF16); r4d_attn_splitkv_combine* finishes the rows, so only the summation order changes.
#define O_PSPLIT 536870912

typedef int v2i32_r4d __attribute__((ext_vector_type(2)));   // one fp8 WMMA operand (8 e4m3)
// pack 4 floats -> 4 e4m3 bytes in element order (RNE, saturating), 2 instructions
__device__ __forceinline__ uint32_t pk_fp8x4(float a, float b, float c, float d) {
    const uint32_t w = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false);
    return __builtin_amdgcn_cvt_pk_fp8_f32(c, d, w, true);
}
// two f16 dwords (four values in element order) -> four e4m3 bytes
__device__ __forceinline__ uint32_t pk_fp8x4_h(uint32_t lo, uint32_t hi) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const h2 a = __builtin_bit_cast(h2, lo), b = __builtin_bit_cast(h2, hi);
    return pk_fp8x4((float)a[0], (float)a[1], (float)b[0], (float)b[1]);
}
// one f16 dword -> two e4m3 bytes in the low half
__device__ __forceinline__ uint32_t pk_fp8x2_h(uint32_t w) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const h2 a = __builtin_bit_cast(h2, w);
    return (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32((float)a[0], (float)a[1], 0u, false) & 0xffffu;
}

// bf16 dword -> 16-bit dword in the target format, optionally scaled. Used once per kernel on Q.
template<class D, int FOLDQ, int F16>
__device__ __forceinline__ uint32_t r4d_qcvt(uint32_t w, float s) {
    if (!FOLDQ) return F16 ? D::from_bf16w(w) : w;
    return D::pk(__builtin_bit_cast(float, w << 16) * s,
                 __builtin_bit_cast(float, w & 0xffff0000u) * s);
}

// O_PSPLIT's store: segment `sp` of row (tok, qhead) in the decode kernel's f32 partial layout,
// O at [idx][HEAD_DIM] then m[nrow] and l[nrow], idx = (tok * q_heads + qhead) * splits + sp. Lane
// (c, h) holds dims 16 dt + 8 h .. + 7 of its row, as at the bf16 store. An empty segment stores
// O = 0, m = -3e38 (at the merge's zero-weight cut) and l = 0.
template<int NKS, int HEAD_DIM>
__device__ __forceinline__ void r4d_prefill_store_partial(const R4DArgs& a, int tok, int qhead,
                                                          int sp, const v8f* acc, float m, float l,
                                                          int h) {
    const size_t idx  = (size_t)(tok * a.q_heads + qhead) * a.splits + sp;
    const size_t nrow = (size_t)r4d_q_total(a) * a.q_heads * a.splits;
    float* so = (float*)a.scratch;
    float* op = so + idx * HEAD_DIM;
    #pragma unroll
    for (int dt = 0; dt < NKS; ++dt)
        *(v8f*)(op + dt * 16 + 8 * h) = acc[dt];
    if (h == 0) { so[nrow * HEAD_DIM + idx] = m; so[nrow * HEAD_DIM + nrow + idx] = l; }
}

// WIN enables a sliding window (a.window positions ending at each row's causal limit), as in the
// decode kernel. Keys below the window are masked, and the KV walk starts at the tile holding the
// workgroup's earliest window token. window <= 0 (or WIN off) is full attention.
template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT, int WIN = 0>
__global__ __launch_bounds__(NWARPS * 32)
void r4d_attn_prefill_kernel(const R4DArgs a)
{
    constexpr int F16   = (OPT & O_F16)   ? 1 : 0;
    constexpr int MSKIP = (OPT & O_MSKIP) ? 1 : 0;
    constexpr int FOLDQ = (OPT & O_FOLDQ) ? 1 : 0;
    constexpr int CTG   = (OPT & O_CONTIG) ? 1 : 0;
    constexpr int PF    = 1 << ((OPT >> 10) & 3);
    constexpr int SGB   = (OPT & O_SGB) ? 1 : 0;
    constexpr int DOT2  = ((OPT & O_DOT2) && (OPT & O_F16) && !(OPT & O_PV8)) ? 1 : 0;
    constexpr int BTS   = (OPT & O_BTS)   ? 1 : 0;
    constexpr int V4    = ((OPT & O_V4) && !KVP) ? 1 : 0;
    constexpr int K6    = ((OPT & O_K6) && V4) ? 1 : 0;
    constexpr int I8    = ((OPT & O_I8) && K6) ? 1 : 0;
    static_assert(!I8 || !(OPT & (O_QK8 | O_PV8)), "the i8 cache has no e4m3 legs");
    // A token's row in a page, and where in it V's codes, V's scales and K's scales start.
    constexpr int ROWB  = I8 ? 2 * HEAD_DIM + 4
                        : K6 ? HEAD_DIM * 22 / 16 : V4 ? HEAD_DIM * 25 / 16 : 2 * HEAD_DIM;
    constexpr int VOFF  = I8 ? HEAD_DIM : K6 ? HEAD_DIM * 3 / 4 : HEAD_DIM;
    constexpr int VSOFF = I8 ? 2 * HEAD_DIM + 2 : K6 ? HEAD_DIM * 21 / 16 : HEAD_DIM * 3 / 2;
    constexpr int KSOFF = I8 ? 2 * HEAD_DIM : HEAD_DIM * 5 / 4;
    constexpr int GPREV = ((OPT & O_GPREV) && !V4) ? 1 : 0;
    constexpr int LDSB  = (OPT & O_LDSB)  ? 1 : 0;
    constexpr int KWIDE = (OPT & O_KWIDE) ? 1 : 0;
    // fp8_v4 takes none of the three (KLATE, the assume, the scale pack): they gain at 100k but lose
    // 3-7 % at 25k and 50k.
    constexpr int OLDV4 = V4 && !K6;
    constexpr int KLATE = ((OPT & O_KLATE) && KWIDE && !KVP && !OLDV4) ? 1 : 0;
    constexpr int QK8   = ((OPT & O_QK8) && !KVP) ? 1 : 0;
    constexpr int PV8   = ((OPT & O_PV8) && !KVP) ? 1 : 0;
    static_assert(!(OPT & (O_QK8 | O_PV8)) || !KVP,
                  "the e4m3 legs need an 8-bit cache, or they compile to the exact form");
    constexpr int SPLIT = (OPT & O_PSPLIT) ? 1 : 0;
    static_assert(!SPLIT || !WIN, "the split partition has no window floor (the merge's WIN has)");
    static_assert(!QK8 || KWIDE, "QK8 raw-byte staging is written for the wide K fetch");
    typedef DT16<F16> D;
    typedef typename D::frag frag16;

    constexpr int ROWS    = NWARPS * 16;
    constexpr int BLOCK_Q = ROWS / GQA;
    constexpr int NKS     = HEAD_DIM / 16;
    constexpr int MT      = TILE / 16;
    constexpr int KSTR    = HEAD_DIM + (CTG ? 8 : 4);
    constexpr int VSTR    = TILE + (CTG ? 8 : 4);
    constexpr int NTHR    = NWARPS * 32;
    constexpr int NB      = TILE / BS;                       // block-table entries per tile
    constexpr int KEL     = KWIDE ? 16 : 8;                  // K elements fetched per thread-pass
    constexpr int KITEMS  = TILE * (HEAD_DIM / KEL);          // K passes needed for one tile
    constexpr int KPASS   = (KITEMS + NTHR - 1) / NTHR;        // passes per thread
    constexpr int KREGS   = KPASS * (KVP ? 2 : 1);             // uint4 of raw K per thread
    constexpr int NTR     = (TILE / 8) * (HEAD_DIM / (KVP ? 32 : 64));
    constexpr int VREGS   = (NTR + NWARPS - 1) / NWARPS;        // uint4 of raw V per thread
    constexpr float PSHIFT = D::SHIFT;
    constexpr float PGROW  = PV8 ? 8.0f : D::GROW;   // e4m3 max is 448 = 2^8.8
    static_assert(!KWIDE || KPASS == 1, "the wide K fetch is one straight-line pass");
    static_assert(!K6 || I8 || KWIDE, "k6_v4 K is staged sixteen dims a thread");
    static_assert((!QK8 && !PV8) || CTG, "the 8-bit fragment maps are written for CONTIG");

    constexpr int KSTR8 = HEAD_DIM + 8, VSTR8 = TILE + 8;
    __shared__ uint16_t sK[QK8 ? (TILE * KSTR8 + 1) / 2 : TILE * KSTR];
    __shared__ uint16_t sV16[PV8 ? (HEAD_DIM * VSTR8 + 1) / 2 : HEAD_DIM * VSTR];
    uint8_t* const sK8 = (uint8_t*)sK;
    uint8_t* const sV8 = (uint8_t*)sV16;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // The launch is NTHR threads. Stated, so the wide K fetch's `key >= TILE` test (never true:
    // KITEMS == NTHR) folds: as a branch its join made the compiler wait on every load in flight right
    // after the K fetch, the V stores' cover included.
    if constexpr (!OLDV4) __builtin_assume(tid < NTHR);
    const int c = lane & 15, h = lane >> 4;

    // SPLIT: blockIdx.x = qtile * splits + segment; wave-uniform, so the divide is scalar.
    const int qb = SPLIT ? (int)blockIdx.x / a.splits : (int)blockIdx.x;
    const int sp = SPLIT ? (int)blockIdx.x - qb * a.splits : 0;
    const int kvh = blockIdx.y, seq = blockIdx.z;
    const int ctx = a.seqused_k[seq];
    if (ctx <= 0) return;

    // Ragged batches: this sequence owns rows [qoff, qoff+qlen) of q and out. grid.x is sized from
    // the batch maximum q_len, so blocks past a shorter sequence's end exit here.
    const int qoff = r4d_q_off(a.cu_q, seq, a.q_len);
    const int qlen = r4d_q_len(a.cu_q, seq, a.q_len);
    if (qb * BLOCK_Q >= qlen) return;

    const int row = warp * 16 + c;
    const int qi = row / GQA, hi = row - qi * GQA;
    const int qpos = qb * BLOCK_Q + qi;
    const bool live = (qpos < qlen);
    const int qrow = live ? qpos : (qlen - 1);
    const int qhead = kvh * GQA + hi;

    const int* __restrict__ bt = a.block_table + (size_t)seq * a.max_blocks;

    // SPLIT: tps tiles a segment, as r4d_attn_tps cuts tiles(ctx). A segment at or past the merge's
    // segment count u = ceil(tiles / tps) is never read: exit. One wholly past this workgroup's
    // causal end must still be written, since the merge reads all u segments of every row: its walk
    // is empty, so the epilogue stores the empty partial (O = 0, m_ref = -3e38, l = 0), and the
    // prologue's prefetch of tile tstart stays inside the sequence. (A separate early store spills
    // k6_v4's next-tile fetch addresses inside the tile loop.)
    const int sp_tps = SPLIT ? ((ctx + TILE - 1) / TILE + a.splits - 1) / a.splits : 0;
    if constexpr (SPLIT) {
        if (sp * sp_tps >= (ctx + TILE - 1) / TILE) return;
    }

    const float kdesc = a.k_descale ? a.k_descale[seq * a.kv_heads + kvh] : 1.0f;
    const float vdesc = a.v_descale ? a.v_descale[seq * a.kv_heads + kvh] : 1.0f;
    const float mul   = a.scale * kdesc * 1.44269504089f;

    frag16 qf[NKS];
    v2i32_r4d qf8[QK8 ? NKS : 1];
    {
        const uint16_t* qp = (const uint16_t*)a.q
                           + ((size_t)(qoff + qrow) * a.q_heads + qhead) * HEAD_DIM;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) {
            // CONTIG reads the 8 dims as one run; the split map takes them 4+4.
            const uint2 l0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h : 4 * h));
            const uint2 h0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h + 4 : 8 + 4 * h));
            if constexpr (QK8) {
                const float m_ = FOLDQ ? mul : 1.0f;
                #define R4D_QB(w) __builtin_bit_cast(float, (w) << 16) * m_, \
                                  __builtin_bit_cast(float, (w) & 0xffff0000u) * m_
                qf8[t] = (v2i32_r4d){(int)pk_fp8x4(R4D_QB(l0.x), R4D_QB(l0.y)),
                                     (int)pk_fp8x4(R4D_QB(h0.x), R4D_QB(h0.y))};
                #undef R4D_QB
            } else {
                qf[t] = D::mk(make_uint2(r4d_qcvt<D, FOLDQ, F16>(l0.x, mul),
                                         r4d_qcvt<D, FOLDQ, F16>(l0.y, mul)),
                              make_uint2(r4d_qcvt<D, FOLDQ, F16>(h0.x, mul),
                                         r4d_qcvt<D, FOLDQ, F16>(h0.y, mul)));
            }
        }
    }

    const int klimit = ctx - qlen + qpos;
    // First key in this row's window, floored at zero. Same formula as r4d_attn_wlo, which is
    // defined in the decode unit, included after this one.
    int wlo = (WIN && a.window > 0) ? (ctx - qlen + qpos - a.window + 1) : 0;
    if (WIN && a.window > 0 && wlo < 0) wlo = 0;

    float m_ref = -3.0e38f;
    float off   = PSHIFT + 3.0e38f;
    float l_i   = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0,0,0,0,0,0,0,0};

    const int qmax_cta   = min(qb * BLOCK_Q + BLOCK_Q - 1, qlen - 1);
    const int klimit_cta = ctx - qlen + qmax_cta;
    const int ctiles = (min(klimit_cta + 1, ctx) + TILE - 1) / TILE;
    // SPLIT: the segment's tiles, clipped at the causal end (empty when it starts past it).
    const int ntiles = SPLIT ? min((sp + 1) * sp_tps, ctiles) : ctiles;
    // First window tile, taken at this workgroup's first query row (the one reaching furthest
    // back), not the chunk's row 0, so fully masked tiles are never walked. Starts at 0 without a
    // window.
    const int wlo0 = (WIN && a.window > 0) ? (ctx - qlen + qb * BLOCK_Q - a.window + 1) : 0;
    const int tstart = SPLIT ? sp * sp_tps
                     : (WIN && a.window > 0 && wlo0 > 0) ? (wlo0 / TILE) : 0;

    // ---- staging, split into a global fetch (registers) and an LDS store ------------------------
    // The split lets GPREV move the fetch out from between the barriers.
    const int vg = lane >> 3, vj = lane & 7;
    uint4 kreg[KREGS], vreg[VREGS];
    // i8: the K scale of the key a pass stages.
    uint32_t ksc8[I8 ? KPASS : 1];

    // ---- fp8_v4 / k6_v4: V is int4 in groups of 32 with an f16 scale (absmax / 7): 16 bytes of
    // codes a group at VOFF (dim d at bits 4*(d%8) of word d/8), the group scales at VSOFF. A thread stages one key pair's 8 dims: two code words and the pair's scale,
    // fetched a tile ahead like GPREV's V, so the loads are in flight across the m-tile loop. They
    // widen into sV16 ([dim][key]) as a dword per dim holding both keys: with u = code ^ 8 (offset
    // binary), 0x6400 | u is the f16 1024 + u, so (x - 1032) * s is the value times s, rounded
    // once. Key pairs run fastest across lanes,
    // which keeps the dword stores of one dim row in distinct banks.
    static_assert(!V4 || F16, "fp8_v4 widens V to f16");
    constexpr int NP4 = (TILE / 2) * (HEAD_DIM / 8);
    constexpr int U4  = V4 ? (NP4 + NTHR - 1) / NTHR : 1;
    // The pair's two scales stay apart until storeV4 packs them: packed here, the prefetch a tile
    // ahead would be waited on right after it is issued (fp8_v4 packs here, see KLATE).
    // i8: V's codes are bytes, two words a key's 8 dims, and a key has one scale; the rest is the
    // same with u = code ^ 0x80 and (x - 1152) * s.
    uint32_t v4c[U4][I8 ? 4 : 2], v4s[U4][2];
    auto fetchV4 = [&](int k0, const int* blks) {
        #pragma unroll
        for (int r = 0; r < U4; ++r) {
            const int u = tid + r * NTHR;
            if (NP4 % NTHR == 0 || u < NP4) {
                const int kp = u % (TILE / 2), o = u / (TILE / 2);
                uint32_t sc[2];
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int kk = min(k0 + 2 * kp + j, ctx - 1);
                    const int bk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                    const uint8_t* base = (const uint8_t*)a.kv + (size_t)bk * a.kv_block_stride
                                        + (size_t)kvh * a.kv_head_stride + (size_t)(kk % BS) * ROWB;
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
            if (NP4 % NTHR == 0 || u < NP4) {
                const int kp = u % (TILE / 2), o = u / (TILE / 2);
                // Codes are two's complement nibbles; ^ 8 makes them offset binary (value + 8). i8:
                // bytes, ^ 0x80; `n` is a key's dims 0-3 of the octet and `m` its dims 4-7.
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
                uint16_t* dst = &sV16[(o * 8) * VSTR + 2 * kp];
                uint8_t* dst8 = &sV8[(o * 8) * VSTR8 + 2 * kp];
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    // Byte b of the first key's word low, of the second key's word in the high half:
                    // perm selects source bytes 0-3 from its second operand, 4-7 from its first,
                    // and 0x0c is a zero byte.
                    const uint32_t sel = (uint32_t)b | 0x0c00u | ((uint32_t)(4 + b) << 16) | 0x0c000000u;
                    const uint32_t lo = __builtin_amdgcn_perm(n1, n0, sel) | 0x64006400u;   // dim 2b
                    const uint32_t hi = __builtin_amdgcn_perm(m1, m0, sel) | 0x64006400u;   // dim 2b+1
                    const h2 vl = (__builtin_bit_cast(h2, lo) - bias) * s2;
                    const h2 vh = (__builtin_bit_cast(h2, hi) - bias) * s2;
                    if constexpr (PV8) {
                        // [dim][key] bytes: the pair's two keys as one 16-bit store a dim
                        *(uint16_t*)(dst8 + (2 * b) * VSTR8)     = (uint16_t)pk_fp8x2_h(__builtin_bit_cast(uint32_t, vl));
                        *(uint16_t*)(dst8 + (2 * b + 1) * VSTR8) = (uint16_t)pk_fp8x2_h(__builtin_bit_cast(uint32_t, vh));
                        continue;
                    }
                    // The octet's rows: nibbles interleave the dims (2b, 2b+1), bytes run b, 4+b.
                    *(uint32_t*)(dst + (I8 ? b : 2 * b) * VSTR)         = __builtin_bit_cast(uint32_t, vl);
                    *(uint32_t*)(dst + (I8 ? 4 + b : 2 * b + 1) * VSTR) = __builtin_bit_cast(uint32_t, vh);
                }
            }
        }
    };

    auto fetchK = [&](int k0, const int* blks) {
        #pragma unroll
        for (int p = 0; p < KPASS; ++p) {
            const int i = tid + p * NTHR;
            if (KITEMS % NTHR == 0 || i < KITEMS) {
                const int key = KWIDE ? (2 * (i >> 5) + (i & 1)) : (i / (HEAD_DIM / KEL));
                const int ch  = KWIDE ? ((i & 31) >> 1) : (i - key * (HEAD_DIM / KEL));
                if (KWIDE && key >= TILE) continue;
                const int kk  = min(k0 + key, ctx - 1);
                const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)blk * a.kv_block_stride
                               + (size_t)kvh * a.kv_head_stride + (size_t)(kk % BS) * ROWB;
                if (KVP) {
                    const uint16_t* g = (const uint16_t*)a.kv + o + ch * KEL;
                    #pragma unroll
                    for (int q = 0; q < (KVP ? (KEL / 8) : 1); ++q)
                        kreg[p * (KEL / 8) + q] = *(const uint4*)(g + 8 * q);
                } else if (I8) {
                    // i8: the chunk's codes (sixteen, or eight without the wide fetch) and the
                    // key's scale.
                    const uint8_t* rp = (const uint8_t*)a.kv + o;
                    if constexpr (KWIDE) {
                        kreg[p] = *(const uint4*)(rp + ch * 16);
                    } else {
                        const uint2 r = *(const uint2*)(rp + ch * 8);
                        kreg[p] = make_uint4(r.x, r.y, 0, 0);
                    }
                    ksc8[p] = *(const uint16_t*)(rp + KSOFF);
                } else if (K6) {
                    // k6_v4: the chunk's sixteen low nibbles, its high-bit word, its group's scale.
                    const uint8_t* rp = (const uint8_t*)a.kv + o;
                    const uint2 lo = *(const uint2*)(rp + ch * 8);
                    kreg[p] = make_uint4(lo.x, lo.y, *(const uint32_t*)(rp + HEAD_DIM / 2 + ch * 4),
                                         *(const uint16_t*)(rp + KSOFF + (ch >> 1) * 2));
                } else if (KWIDE) {
                    kreg[p] = *(const uint4*)((const uint8_t*)a.kv + o + ch * 16);
                } else {
                    const uint2 r = *(const uint2*)((const uint8_t*)a.kv + o + ch * 8);
                    kreg[p] = make_uint4(r.x, r.y, 0, 0);
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
                const int kk = min(k0 + kg * 8 + vj, ctx - 1);
                const int d0 = dg * (KVP ? 32 : 64) + vg * (KVP ? 8 : 16);
                const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)blk * a.kv_block_stride
                               + (size_t)kvh * a.kv_head_stride
                               + (size_t)(kk % BS) * ROWB + HEAD_DIM + d0;
                const void* p = KVP ? (const void*)((const uint16_t*)a.kv + o)
                                    : (const void*)((const uint8_t*)a.kv + o);
                vreg[r] = __builtin_bit_cast(uint4, load_tr_b128(p));
            }
        }
    };

    auto storeK = [&]() {
        #pragma unroll
        for (int p = 0; p < KPASS; ++p) {
            const int i = tid + p * NTHR;
            if (KITEMS % NTHR == 0 || i < KITEMS) {
                const int key = KWIDE ? (2 * (i >> 5) + (i & 1)) : (i / (HEAD_DIM / KEL));
                const int ch  = KWIDE ? ((i & 31) >> 1) : (i - key * (HEAD_DIM / KEL));
                if (KWIDE && key >= TILE) continue;
                if constexpr (QK8 && K6) {
                    // k6_v4: widen to f16 as below, then round each value once to e4m3
                    const uint4 r = kreg[p];
                    #pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        uint32_t w4[4];
                        k6x8_to_f16x4w(q ? r.y : r.x, r.z, q, r.w | (r.w << 16), w4);
                        *(uint2*)(&sK8[key * KSTR8 + ch * KEL + 8 * q]) =
                            make_uint2(pk_fp8x4_h(w4[0], w4[1]), pk_fp8x4_h(w4[2], w4[3]));
                    }
                    continue;
                } else if constexpr (QK8) {
                    *(uint4*)(&sK8[key * KSTR8 + ch * KEL]) = kreg[p];
                    continue;
                }
                uint16_t* kd = &sK[key * KSTR + ch * KEL];
                #pragma unroll
                for (int q = 0; q < KEL / 8; ++q) {
                    uint32_t w4[4];
                    if (KVP) {
                        const uint4 r = kreg[p * (KEL / 8) + q];
                        w4[0] = D::from_bf16w(r.x); w4[1] = D::from_bf16w(r.y);
                        w4[2] = D::from_bf16w(r.z); w4[3] = D::from_bf16w(r.w);
                    } else {
                        const uint4 r = kreg[p];
                        if constexpr (I8)
                            i8x8_to_f16x4w(q ? make_uint2(r.z, r.w) : make_uint2(r.x, r.y),
                                           ksc8[p] | (ksc8[p] << 16), w4);
                        else if constexpr (K6) k6x8_to_f16x4w(q ? r.y : r.x, r.z, q, r.w | (r.w << 16), w4);
                        else fp8x8_to_16x4w<D>(q ? make_uint2(r.z, r.w) : make_uint2(r.x, r.y), w4);
                    }
                    *(uint2*)(kd + 8 * q)     = make_uint2(w4[0], w4[1]);
                    *(uint2*)(kd + 8 * q + 4) = make_uint2(w4[2], w4[3]);
                }
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
                } else {
                    // tr_b128 transposes 16-bit lanes, so each short holds two fp8 from adjacent
                    // dims; v_perm_b32 splits them into two dim rows.
                    const uint32_t lo0 = byte_gather(v.x, v.y, 0), lo1 = byte_gather(v.z, v.w, 0);
                    const uint32_t hi0 = byte_gather(v.x, v.y, 1), hi1 = byte_gather(v.z, v.w, 1);
                    if constexpr (PV8) {
                        *(uint2*)(&sV8[(d0 + 2 * vj)     * VSTR8 + kg * 8]) = make_uint2(lo0, lo1);
                        *(uint2*)(&sV8[(d0 + 2 * vj + 1) * VSTR8 + kg * 8]) = make_uint2(hi0, hi1);
                        continue;
                    }
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

    // Tile ti's block-table entries are loaded during tile ti-1, off the KV address chain.
    // A multi-block tile can run past the sequence's last block (on the last row, past the end of
    // the table). Keys are clamped to ctx - 1, so entries past lastb are never used; only tiles
    // reaching past lastb take the clamped load. With NB == 1 the test folds away.
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
    if (BTS) load_bt(tstart * NB, blk);
    if (GPREV) fetchV(tstart * TILE, blk);
    if (V4) fetchV4(tstart * TILE, blk);
    if (KLATE) fetchK(tstart * TILE, blk);

    for (int ti = tstart; ti < ntiles; ++ti) {
        const int k0 = ti * TILE;
        int blkn[NB];
        if (BTS && ti + 1 < ntiles) {
            load_bt((ti + 1) * NB, blkn);
        }
        if (LDSB) lds_barrier(); else __syncthreads();
        if (!KLATE) fetchK(k0, blk);
        // V first: already in registers.
        if constexpr (V4) { storeV4(); } else { if (!GPREV) fetchV(k0, blk); storeV(); }
        storeK();
        if (LDSB) lds_barrier(); else __syncthreads();
        // Issue the next tile's V loads before the WMMAs below; they are consumed after the next
        // barrier.
        if (GPREV && ti + 1 < ntiles) fetchV(k0 + TILE, blkn);
        if (V4 && ti + 1 < ntiles) fetchV4(k0 + TILE, blkn);

        #pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            v8f s = (v8f){0,0,0,0,0,0,0,0};
            if constexpr (QK8) {
                const uint8_t* krow8 = &sK8[(mt * 16 + c) * KSTR8] + 8 * h;
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    v2i32_r4d kb8[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        kb8[j] = *(const v2i32_r4d*)(krow8 + 16 * (g + j));
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        s = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(kb8[j], qf8[g + j], s);
                }
            } else {
                const uint16_t* krow = &sK[(mt * 16 + c) * KSTR] + (CTG ? 8 * h : 0);
                // Load PF fragments before issuing their WMMAs, so each WMMA waits on
                // s_wait_dscnt PF-1 instead of a full drain.
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    frag16 kb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        const uint16_t* kp = krow + 16 * (g + j);
                        if (CTG) {
                            const uint4 u = *(const uint4*)kp;
                            kb[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        } else {
                            kb[j] = D::mk(*(const uint2*)(kp + 4 * h),
                                          *(const uint2*)(kp + 8 + 4 * h));
                        }
                    }
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) s = D::wmma(kb[j], qf[g + j], s);
                }
            }
            if (KLATE && mt == MT - 1 && ti + 1 < ntiles) fetchK(k0 + TILE, blkn);
            float sc[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) sc[e] = FOLDQ ? s[e] : s[e] * mul;

            const int kbase = k0 + mt * 16 + 8 * h;
            // Mask keys past klimit and, with WIN, before wlo. MSKIP skips tiles straddling neither.
            if (!MSKIP || wave_any(kbase + 7 > klimit) || (WIN && wave_any(kbase < wlo))) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) sc[e] = ((kbase + e <= klimit) && (!WIN || kbase + e >= wlo))
                                                    ? sc[e] : -INFINITY;
            }

            float smax = sc[0];
            #pragma unroll
            for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
            // Both half-waves hold the same query and different keys. The branch must be
            // wave-uniform, but each query may change scale only on its own growth threshold;
            // otherwise other live query rows change its rounding when the chunk length changes.
            const float rmax = fmaxf(smax, swap16(smax));
            const bool row_grow = live && rmax > m_ref + PGROW;
            if (wave_any(row_grow)) {
                const float mnew = row_grow ? fmaxf(fmaxf(m_ref, rmax), -3.0e38f) : m_ref;
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
                const float p = ex2(sc[e] + off);
                pf[e] = p;
                if (!DOT2) lsum += p;
            }
            if (!DOT2) l_i += lsum;

            if constexpr (PV8) {
                const v2i32_r4d p8 = (v2i32_r4d){(int)pk_fp8x4(pf[0], pf[1], pf[2], pf[3]),
                                                 (int)pk_fp8x4(pf[4], pf[5], pf[6], pf[7])};
                const uint8_t* vbase8 = &sV8[c * VSTR8 + mt * 16 + 8 * h];
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    v2i32_r4d vb8[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        vb8[j] = *(const v2i32_r4d*)(vbase8 + (size_t)(g + j) * 16 * VSTR8);
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        acc[g + j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(vb8[j], p8, acc[g + j]);
                }
            } else {
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
                // QK leaves lane (c,h) holding keys 8h+e. Under CONTIG that is already the B
                // fragment; the split map needs a cross-half exchange.
                frag16 p16;
                if (CTG) {
                    p16 = D::mk(lo, hi2);
                } else {
                    const uint2 y = swap16_u2p((h == 0) ? hi2 : lo);
                    p16 = (h == 0) ? D::mk(lo, y) : D::mk(y, hi2);
                }
                const uint16_t* vbase = &sV16[c * VSTR + mt * 16] + (CTG ? 8 * h : 0);
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    frag16 vb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        const uint16_t* vp = vbase + (g + j) * 16 * VSTR;
                        if (CTG) {
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
    // fetch-inactive: the split store reads l_i under `if (h == 0)` only, so the exchange may be sunk
    // into that half; swap16_fi still reads the partner's half of the row sum.
    l_i += swap16_fi(l_i);
    if constexpr (SPLIT) {
        // Unnormalised: the merge weighs segments by 2^(m - max m) and applies v_descale itself.
        r4d_prefill_store_partial<NKS, HEAD_DIM>(a, qoff + qpos, qhead, sp, acc, m_ref, l_i, h);
        return;
    }
    const float inv = (l_i > 0.0f) ? (vdesc / l_i) : 0.0f;
    uint16_t* op = (uint16_t*)a.out
                 + ((size_t)(qoff + qpos) * a.q_heads + qhead) * HEAD_DIM;
    #pragma unroll
    for (int dt = 0; dt < NKS; ++dt) {
        uint32_t tw[4];
        #pragma unroll
        for (int e = 0; e < 8; e += 2)
            tw[e >> 1] = (uint32_t)f32_to_bf16(acc[dt][e] * inv)
                       | ((uint32_t)f32_to_bf16(acc[dt][e + 1] * inv) << 16);
        *(uint4*)(op + dt * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
    }
}
