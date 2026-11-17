// R4D vision-encoder attention: dense, non-causal, head_dim 72, bf16 in and out.
//
// Same transposed core as the paged prefill kernel: S^T = K.Q^T, so each lane of the wave32
// column-distributed accumulator owns one query row and the softmax state is lane-private. The C
// fragment of S^T (lane holds query c, keys 8h+i) is the B fragment of P.V (P^T), so P never goes
// through LDS; with that round trip this shape does not beat torch SDPA.
//
// head_dim 72 runs as 5 k-tiles of 16 = 80, with dims 72..79 zero: the last Q tile is zeroed on
// the h==1 half, the last 8 columns of every staged K row are zeroed once, and only the h==0 half
// stores the last output tile (h==1 would run past the 72-element row). 11% padded WMMA work.
//
// Differences from the paged kernel:
//   * K and V are contiguous [tokens, heads, 72] bf16: no block table, descale or fp8 widening.
//     K is fetched 16 bytes per thread; V is transposed into LDS with global_load_tr_b128.
//   * Non-causal: only the last key tile is masked, with a workgroup-uniform branch.
//   * Varlen: one launch covers every image in the batch (grid.z over cu_seqlens segments).
//
// Operands are f16 (r4d_dt16.h: gfx1201 has no bf16 convert, so packing P costs 1 instruction per
// pair instead of ~6). A bf16 build (V_O_F16 off) makes K/V staging a byte copy but is slower
// overall; it is kept so the trade can be rechecked after a geometry change.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_dt16.h"

// Optimisation steps, each separable so any one can be turned off after a geometry change.
#define V_O_F16        1   // f16 rather than bf16 as the 16-bit operand format
#define V_O_FOLDQ      8   // scale * log2(e) folded into the query once at load
#define V_O_PF2     1024   // fragment prefetch depth: 2
#define V_O_PF4     2048   //                          4
#define V_O_SGB     4096   // hard scheduling barrier between the prefetch group and its WMMAs
#define V_O_DOT2   16384   // v_dot2_f32_f16 row sum off the packed P
#define V_O_GPREV 1048576  // next tile's V fetched into registers across the m-tile loop
#define V_O_LDSB  2097152  // LDS-scoped barrier instead of __syncthreads()
// GPREK: register prefetch for K too. The paged kernel (240 VGPRs) would spill it; this one runs at
// ~130 and the K fetch is three uint4. Without it the K load sits between the barriers with only
// the V LDS stores to cover it.
#define V_O_GPREK 4194304
// MIXFMT: bf16 operands for QK, f16 for PV. Staging bf16 K into an f16 LDS costs a convert per
// element (3 VALU per dword), while packing f32 P to bf16 costs ~6 VALU vs one v_cvt_pkrtz_f16_f32.
// So K is copied raw as bf16 (query converted once), and V and P stay f16. Both widenings are exact.
#define V_O_MIXFMT 8388608


// bf16 dword -> 16-bit dword in the target format, with the query scale folded in.
template<class D, int FOLDQ, int F16>
__device__ __forceinline__ uint32_t r4d_vit_qcvt(uint32_t w, float s) {
    if (!FOLDQ) return F16 ? D::from_bf16w(w) : w;
    return D::pk(__builtin_bit_cast(float, w << 16) * s,
                 __builtin_bit_cast(float, w & 0xffff0000u) * s);
}

template<int NWARPS, int TILE, int QB, int OPT>
__global__ __launch_bounds__(NWARPS * 32)
void r4d_attn_vit_kernel(const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
                         const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
                         const int* __restrict__ cu, int heads, float scale,
                         int q_ld, int k_ld, int v_ld, int o_ld)
{
    constexpr int F16   = (OPT & V_O_F16)   ? 1 : 0;
    constexpr int FOLDQ = (OPT & V_O_FOLDQ) ? 1 : 0;
    constexpr int PF    = (OPT & V_O_PF4) ? 4 : ((OPT & V_O_PF2) ? 2 : 1);
    constexpr int SGB   = (OPT & V_O_SGB)   ? 1 : 0;
    constexpr int DOT2  = ((OPT & V_O_DOT2) && F16) ? 1 : 0;
    constexpr int GPREV = (OPT & V_O_GPREV) ? 1 : 0;
    constexpr int LDSB  = (OPT & V_O_LDSB)  ? 1 : 0;
    constexpr int GPREK = (OPT & V_O_GPREK) ? 1 : 0;
    constexpr int MIXF  = (OPT & V_O_MIXFMT) ? 1 : 0;
    typedef DT16<F16> D;                       // V and P: the format the softmax packs into
    typedef DT16<MIXF ? 0 : F16> DK;           // K and Q: bf16 under MIXFMT, else the same
    typedef typename D::frag  frag16;
    typedef typename DK::frag fragK;

    constexpr int HD    = 72;               // the real head dim
    constexpr int HDP   = 80;               // padded to whole 16-wide WMMA k-tiles
    constexpr int NKS   = HDP / 16;         // 5
    constexpr int DGRP  = HD / 8;           // 9 eight-element dim groups spanning the real dims
    constexpr int ROWS  = NWARPS * 16 * QB;
    constexpr int MT    = TILE / 16;
    constexpr int NTHR  = NWARPS * 32;
    // Row strides padded by 8 shorts: 16-byte aligned (an 8-byte-aligned 16-byte LDS access splits
    // into two b64) and off the bank period, so both fragment reads hit 8 distinct bank groups.
    constexpr int KSTR  = HDP + 8;
    constexpr int VSTR  = TILE + 8;
    constexpr int KGRP  = TILE / 8;         // eight-key groups per tile; a power of two
    constexpr int NTR   = KGRP * DGRP;      // transposed V units, one per 8 lanes
    constexpr int LGRP  = NTHR / 8;
    constexpr int VPASS = (NTR + LGRP - 1) / LGRP;
    constexpr int KIA   = TILE * 8;         // K fetch items, dims 0..63 (16 B each)
    constexpr int KIB   = TILE;             // K fetch items, dims 64..71
    constexpr int KPA   = (KIA + NTHR - 1) / NTHR;
    constexpr int KPB   = (KIB + NTHR - 1) / NTHR;
    constexpr float PSHIFT = D::SHIFT;
    constexpr float PGROW  = D::GROW;
    static_assert((TILE & (TILE - 1)) == 0 && TILE >= 16, "TILE must be a power of two >= 16");

    __shared__ __align__(16) uint16_t sK[TILE * KSTR];
    __shared__ __align__(16) uint16_t sV[HDP * VSTR];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c = lane & 15, h = lane >> 4;
    const int qb = blockIdx.x, hd = blockIdx.y, seq = blockIdx.z;

    const int s0 = cu[seq], slen = cu[seq + 1] - s0;
    if (slen <= 0 || qb * ROWS >= slen) return;

    // q_ld, k_ld, v_ld, o_ld are token pitches in elements: heads * HD for a dense tensor, or the
    // projection row for a column slice of a fused [q|k|v] one.
    const float mul = scale * 1.44269504089f;

    // ---- the query fragments, held for the life of the workgroup -----------------------------
    // A warp owns QB tiles of 16 query rows, so each K/V fragment read from LDS feeds QB WMMAs. At
    // QB = 1 there is one ds_load_b128 per WMMA, and LDS delivers less per cycle than the matrix
    // units consume. The accumulator is 8 VGPRs per output tile, so QB > 1 is affordable here.
    int  qpos[QB];
    bool live[QB];
    fragK qf[QB][NKS];
    #pragma unroll
    for (int b = 0; b < QB; ++b) {
        qpos[b] = qb * ROWS + (warp * QB + b) * 16 + c;
        live[b] = qpos[b] < slen;
        const int qrow = live[b] ? qpos[b] : slen - 1;
        const uint16_t* qp = gq + (long)(s0 + qrow) * q_ld + (long)hd * HD;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) {
            // Padded half of the last k-tile (dims 72..79). K is zeroed too: the WMMA contracts,
            // so a NaN * 0 there would poison the whole score row.
            if (t == NKS - 1 && h == 1) { qf[b][t] = DK::mk(make_uint2(0, 0), make_uint2(0, 0)); continue; }
            const uint4 u = *(const uint4*)(qp + 16 * t + 8 * h);
            qf[b][t] = DK::mk(make_uint2(r4d_vit_qcvt<DK, FOLDQ, MIXF ? 0 : F16>(u.x, mul),
                                         r4d_vit_qcvt<DK, FOLDQ, MIXF ? 0 : F16>(u.y, mul)),
                              make_uint2(r4d_vit_qcvt<DK, FOLDQ, MIXF ? 0 : F16>(u.z, mul),
                                         r4d_vit_qcvt<DK, FOLDQ, MIXF ? 0 : F16>(u.w, mul)));
        }
    }

    // Zero the padding of the staging buffers once; staging only fills dims 0..71.
    #pragma unroll 1
    for (int i = tid; i < TILE; i += NTHR) *(uint4*)&sK[i * KSTR + HD] = make_uint4(0, 0, 0, 0);
    #pragma unroll 1
    for (int i = tid; i < (HDP - HD) * VSTR; i += NTHR) sV[HD * VSTR + i] = 0;

    float m_ref[QB], soff[QB], l_i[QB];
    v8f acc[QB][NKS];
    #pragma unroll
    for (int b = 0; b < QB; ++b) {
        m_ref[b] = -3.0e38f;
        soff[b]  = PSHIFT + 3.0e38f;
        l_i[b]   = 0.0f;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) acc[b][t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    }

    // ---- staging, split into a global fetch (to registers) and an LDS store -------------------
    const int vj = tid & 7, lgi = tid >> 3;
    uint4 kregA[KPA], kregB[KPB], vreg[VPASS];

    auto fetchK = [&](int k0) {
        #pragma unroll
        for (int p = 0; p < KPA; ++p) {
            const int i = tid + p * NTHR;
            if (KIA % NTHR == 0 || i < KIA) {
                const int key = i >> 3, g = i & 7;
                const int kk = (k0 + key < slen) ? (k0 + key) : (slen - 1);
                kregA[p] = *(const uint4*)(gk + (long)(s0 + kk) * k_ld + (long)hd * HD + g * 8);
            }
        }
        #pragma unroll
        for (int p = 0; p < KPB; ++p) {
            const int i = tid + p * NTHR;
            if (KIB % NTHR == 0 || i < KIB) {
                const int kk = (k0 + i < slen) ? (k0 + i) : (slen - 1);
                kregB[p] = *(const uint4*)(gk + (long)(s0 + kk) * k_ld + (long)hd * HD + 64);
            }
        }
    };

    auto storeK = [&]() {
        #pragma unroll
        for (int p = 0; p < KPA; ++p) {
            const int i = tid + p * NTHR;
            if (KIA % NTHR == 0 || i < KIA) {
                const uint4 r = kregA[p];
                *(uint4*)&sK[(i >> 3) * KSTR + (i & 7) * 8] =
                    make_uint4(DK::from_bf16w(r.x), DK::from_bf16w(r.y),
                               DK::from_bf16w(r.z), DK::from_bf16w(r.w));
            }
        }
        #pragma unroll
        for (int p = 0; p < KPB; ++p) {
            const int i = tid + p * NTHR;
            if (KIB % NTHR == 0 || i < KIB) {
                const uint4 r = kregB[p];
                *(uint4*)&sK[i * KSTR + 64] =
                    make_uint4(DK::from_bf16w(r.x), DK::from_bf16w(r.y),
                               DK::from_bf16w(r.z), DK::from_bf16w(r.w));
            }
        }
    };

    // V is read transposed from global: a group's eight lanes name eight key rows at the same 8
    // dims, and global_load_tr_b128 gives lane j dim d0+j across all eight, one 16-byte row of V^T.
    auto fetchV = [&](int k0) {
        #pragma unroll
        for (int r = 0; r < VPASS; ++r) {
            const int u = lgi + r * LGRP;
            if (NTR % LGRP == 0 || u < NTR) {
                const int kg = u & (KGRP - 1), dg = u / KGRP;
                const int kk0 = k0 + kg * 8 + vj;
                const int kk = (kk0 < slen) ? kk0 : (slen - 1);
                vreg[r] = __builtin_bit_cast(uint4, load_tr_b128(
                    gv + (long)(s0 + kk) * v_ld + (long)hd * HD + dg * 8));
            }
        }
    };

    auto storeV = [&]() {
        #pragma unroll
        for (int r = 0; r < VPASS; ++r) {
            const int u = lgi + r * LGRP;
            if (NTR % LGRP == 0 || u < NTR) {
                const int kg = u & (KGRP - 1), dg = u / KGRP;
                const uint4 v = vreg[r];
                *(uint4*)&sV[(dg * 8 + vj) * VSTR + kg * 8] =
                    make_uint4(D::from_bf16w(v.x), D::from_bf16w(v.y),
                               D::from_bf16w(v.z), D::from_bf16w(v.w));
            }
        }
    };

    const int ntiles = (slen + TILE - 1) / TILE;
    if (GPREV) fetchV(0);
    if (GPREK) fetchK(0);

    for (int ti = 0; ti < ntiles; ++ti) {
        const int k0 = ti * TILE;
        const bool tail = (k0 + TILE > slen);
        if (LDSB) lds_barrier(); else __syncthreads();
        if (!GPREK) fetchK(k0);
        if (!GPREV) fetchV(k0);
        storeV();              // V first: already in registers, so it covers K's load wait
        storeK();
        if (LDSB) lds_barrier(); else __syncthreads();
        // Issued before the WMMAs below; consumed after the next barrier.
        if (GPREV && ti + 1 < ntiles) fetchV(k0 + TILE);
        if (GPREK && ti + 1 < ntiles) fetchK(k0 + TILE);

        #pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            v8f s[QB];
            #pragma unroll
            for (int b = 0; b < QB; ++b) s[b] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
            {
                const uint16_t* krow = &sK[(mt * 16 + c) * KSTR] + 8 * h;
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    fragK kb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        if (g + j < NKS) {
                            const uint4 u = *(const uint4*)(krow + 16 * (g + j));
                            kb[j] = DK::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        }
                    }
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        if (g + j < NKS)
                            #pragma unroll
                            for (int b = 0; b < QB; ++b) s[b] = DK::wmma(kb[j], qf[b][g + j], s[b]);
                }
            }

            frag16 p16[QB];
            #pragma unroll
            for (int b = 0; b < QB; ++b) {
                float sc[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) sc[e] = FOLDQ ? s[b][e] : s[b][e] * mul;

                // Non-causal: only the last tile can hold keys past the segment end.
                if (tail) {
                    const int kbase = k0 + mt * 16 + 8 * h;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) sc[e] = (kbase + e < slen) ? sc[e] : -INFINITY;
                }

                float smax = sc[0];
                #pragma unroll
                for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
                if (wave_any(smax > m_ref[b] + PGROW)) {
                    const float rmax  = fmaxf(smax, swap16(smax));
                    const float mnew  = fmaxf(fmaxf(m_ref[b], rmax), -3.0e38f);
                    const float alpha = ex2(m_ref[b] - mnew);
                    #pragma unroll
                    for (int dt = 0; dt < NKS; ++dt)
                        #pragma unroll
                        for (int e = 0; e < 8; ++e) acc[b][dt][e] *= alpha;
                    l_i[b] *= alpha;
                    m_ref[b] = mnew;
                    soff[b]  = PSHIFT - mnew;
                }

                float pv[8], lsum = 0.0f;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float p = ex2(sc[e] + soff[b]);
                    pv[e] = p;
                    if (!DOT2) lsum += p;
                }
                if (!DOT2) l_i[b] += lsum;

                uint32_t pw[4];
                #pragma unroll
                for (int e = 0; e < 8; e += 2) pw[e >> 1] = D::pk(pv[e], pv[e + 1]);
                if (DOT2) {
                    const v2hf one = {(__fp16)1.0f, (__fp16)1.0f};
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                        l_i[b] = __builtin_amdgcn_fdot2(__builtin_bit_cast(v2hf, pw[e]), one,
                                                        l_i[b], false);
                }
                // Lane (c,h) holds keys 8h+e of query c, which is already the B fragment of P^T.
                p16[b] = D::mk(make_uint2(pw[0], pw[1]), make_uint2(pw[2], pw[3]));
            }

            {
                const uint16_t* vbase = &sV[c * VSTR + mt * 16] + 8 * h;
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    frag16 vb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        if (g + j < NKS) {
                            const uint4 u = *(const uint4*)(vbase + (g + j) * 16 * VSTR);
                            vb[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        }
                    }
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        if (g + j < NKS)
                            #pragma unroll
                            for (int b = 0; b < QB; ++b)
                                acc[b][g + j] = D::wmma(vb[j], p16[b], acc[b][g + j]);
                }
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < QB; ++b) {
        if (!live[b]) continue;
        const float ls = l_i[b] + swap16(l_i[b]);
        const float inv = (ls > 0.0f) ? (1.0f / ls) : 0.0f;
        uint16_t* op = go + (long)(s0 + qpos[b]) * o_ld + (long)hd * HD;
        #pragma unroll
        for (int dt = 0; dt < NKS; ++dt) {
            // The last tile's h==1 half (dims 72..79) is padding and would store past the row.
            if (dt == NKS - 1 && h == 1) continue;
            uint32_t tw[4];
            #pragma unroll
            for (int e = 0; e < 8; e += 2)
                tw[e >> 1] = (uint32_t)f32_to_bf16(acc[b][dt][e] * inv)
                           | ((uint32_t)f32_to_bf16(acc[b][dt][e + 1] * inv) << 16);
            *(uint4*)(op + dt * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
        }
    }
}

// ---- entry point -----------------------------------------------------------------------------
// Two instantiations of the same kernel, chosen by segment length; they differ in query rows per
// workgroup. ROWS 256 is fastest from 1K patches up but quantises badly below (at 576 patches a
// workgroup is two thirds idle: 1.33x Triton vs 1.76x for ROWS 128). The crossover is between 576
// and 1024; V_SPLIT 768 is the midpoint.
#define V_NWARPS 16          // large instantiation: ROWS = 256
#define V_TILE 128
#define V_QB 1
#define V_SNWARPS 8          // small instantiation: ROWS = 128
#define V_STILE 64
#define V_SQB 1
#define V_SPLIT 768
// F16 | FOLDQ | PF4 | SGB | DOT2 | GPREV | LDSB | GPREK. MIXFMT is off: same speed, and one
// operand format is simpler.
#define V_OPT 7362569

// q, k, v, o are [total_tokens, heads, 72] contiguous bf16; cu_seqlens is [num_seqs + 1] int32, one
// image (or attention window) per segment. max_seqlen is a host-side bound on the lengths; the
// kernel reads the device-side cu_seqlens.
extern "C" int r4d_attn_vit_h72_bf16(const void* q, const void* k, const void* v, void* o,
                                     const void* cu_seqlens, int num_seqs, int max_seqlen,
                                     int heads, int head_dim, float scale, void* stream)
{
    const int ld = heads * 72;
    return r4d_attn_vit_h72_strided_bf16(q, k, v, o, cu_seqlens, num_seqs, max_seqlen, heads,
                                         head_dim, scale, ld, ld, ld, ld, stream);
}

// As above, each operand at its own token pitch in elements (e.g. a column slice of a fused
// [q|k|v] projection). Pitches and head offsets must be multiples of 8 elements for the 16-byte
// K and V fetches.
extern "C" int r4d_attn_vit_h72_strided_bf16(const void* q, const void* k, const void* v, void* o,
                                             const void* cu_seqlens, int num_seqs, int max_seqlen,
                                             int heads, int head_dim, float scale, int q_ld,
                                             int k_ld, int v_ld, int o_ld, void* stream)
{
    if (head_dim != 72) return -1;
    if (heads <= 0) return -2;
    if ((q_ld | k_ld | v_ld | o_ld) & 7) return -3;
    if (num_seqs <= 0 || max_seqlen <= 0) return 0;

#define V_LAUNCH(NW, TL, QBS)                                                                  \
    do {                                                                                       \
        constexpr int ROWS = (NW) * 16 * (QBS);                                                \
        dim3 grid((max_seqlen + ROWS - 1) / ROWS, heads, num_seqs);                             \
        hipLaunchKernelGGL((r4d_attn_vit_kernel<(NW), (TL), (QBS), V_OPT>), grid,               \
                           dim3((NW) * 32), 0, (hipStream_t)stream, (const uint16_t*)q,         \
                           (const uint16_t*)k, (const uint16_t*)v, (uint16_t*)o,                \
                           (const int*)cu_seqlens, heads, scale, q_ld, k_ld, v_ld, o_ld);       \
    } while (0)

    if (max_seqlen < V_SPLIT) V_LAUNCH(V_SNWARPS, V_STILE, V_SQB);
    else                      V_LAUNCH(V_NWARPS,  V_TILE,  V_QB);
#undef V_LAUNCH

    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -100 - (int)e;
    return 0;
}

// Compiled geometry: head_dim, the two query-block heights, and the switch length.
extern "C" void r4d_attn_vit_dims(int* head_dim, int* rows_large, int* rows_small, int* split) {
    if (head_dim)    *head_dim    = 72;
    if (rows_large)  *rows_large  = V_NWARPS * 16 * V_QB;
    if (rows_small)  *rows_small  = V_SNWARPS * 16 * V_SQB;
    if (split)       *split       = V_SPLIT;
}

// ============================================================== the plugin ABI entry points

/* ------------------------------------------------------------------ dense (vision) */

enum { AD_Q = 0, AD_K, AD_V, AD_CU, AD_OUT };

extern "C" int r4d_rad_attn_vit(const RadArgs* a, RadStream s) {
    const RadTensor* q  = r4d_opt(a, AD_Q);
    const RadTensor* k  = r4d_opt(a, AD_K);
    const RadTensor* v  = r4d_opt(a, AD_V);
    const RadTensor* cu = r4d_opt(a, AD_CU);
    const RadTensor* o  = r4d_opt(a, AD_OUT);
    if (!q || !k || !v || !cu || !o) return R4D_NO(RAD_E_INVAL);
    if (!r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);

    /* [tokens, heads, 72] dense, or [tokens, heads * 72] at its own row pitch (a column slice of
     * a fused projection). */
    const long long head_dim = r4d_geti_or(a, "head_dim", q->rank == 3 ? q->shape[2] : 72);
    if (head_dim != 72) return R4D_NO(RAD_E_SHAPE);
    /* The kernel attends every key of a segment, its bounds read as they stand, so a window,
     * blocks or a cu_seqlens in finer units are refused. */
    if (r4d_geti_or(a, "window", 0) != 0 || r4d_geti_or(a, "block", 0) != 0 ||
        r4d_geti_or(a, "cu_div", 1) != 1)
        return R4D_NO(RAD_E_INVAL);
    long long total = 0, heads = 0, q_ld = 0;
    long long tk = 0, hk = 0, k_ld = 0, tv = 0, hv = 0, v_ld = 0, to = 0, ho = 0, o_ld = 0;
    if (!r4d_head_rows(q, head_dim, &total, &heads, &q_ld) ||
        !r4d_head_rows(k, head_dim, &tk, &hk, &k_ld) ||
        !r4d_head_rows(v, head_dim, &tv, &hv, &v_ld) ||
        !r4d_head_rows(o, head_dim, &to, &ho, &o_ld))
        return R4D_NO(RAD_E_STRIDE);
    if (hk != heads || hv != heads || ho != heads) return R4D_NO(RAD_E_SHAPE);
    if (tk < total || tv < total || to < total) return R4D_NO(RAD_E_SHAPE);
    if (cu->rank != 1 || cu->shape[0] < 2) return R4D_NO(RAD_E_SHAPE);

    const int num_seqs = (int)(cu->shape[0] - 1);

    /* max_seqlen is a bound that selects the query-block height (speed only); the kernel reads
     * device-side cu_seqlens. Defaults to the total token count. */
    long long msl = r4d_geti_or(a, "max_seqlen", 0);
    if (msl <= 0) msl = total;

    const float scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)head_dim));

    const int e = r4d_attn_vit_h72_strided_bf16(q->data, k->data, v->data, o->data, cu->data,
                                                num_seqs, (int)msl, (int)heads, (int)head_dim,
                                                scale, (int)q_ld, (int)k_ld, (int)v_ld, (int)o_ld,
                                                s);
    if (e == 0) return RAD_OK;
    return R4D_NO(e == -3 ? RAD_E_STRIDE : e <= -100 ? RAD_E_DEVICE : RAD_E_SHAPE);
}
