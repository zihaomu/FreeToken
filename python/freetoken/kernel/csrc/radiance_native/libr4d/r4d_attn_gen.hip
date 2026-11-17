// R4D attention over a paged prefix and the pass's own rows (attn_prefix_self), and the causal
// dense attention a text encoder's prompt pass needs (attn_dense, causal 1, head_dim 128).
//
// The dense kernel (r4d_attn_dense_bf16.hip) with a second key source. S^T = K.Q^T, so a lane
// owns one query row and its softmax state, the C fragment of S^T is the B fragment of P.V, and P
// never goes through LDS. A workgroup owns one KV head and every query head reading it.
//
//   Prefix: a sequence's first `used[s]` keys are in the paged cache ([blocks, kv_heads, 16,
//   2 * HD], K then V per slot); its own rows' keys are dense beside the queries. Prefix tiles are
//   walked first, then own tiles; each tile has one source. Every prefix key is visible.
//
//   Causal own keys: with `causal` a query sees its own rows up to itself (a conditioning
//   sequence's text segment, a text encoder's prompt). Tiles wholly below every query skip the mask.
//
// bf16 operands (DT16<0>): diffusion transformer values carry outliers that could overflow f16,
// and the reference also takes P to bf16 for P.V (rounded there, truncated here; see the softmax).
//
// Int8 Q.K form (`I8`, op operand `k_i8` present): scores run on v_wmma_i32_16x16x16_iu8 (half the
// cycles of bf16); P.V stays bf16. Codes follow quant_act_i8g: per (row, head) scale amax / 127
// (1 for an all-zero row), code rint(x / scale) clamped to +-127. Own keys arrive quantised
// (k_i8, k_scale, from the preceding op); the query is quantised here once per workgroup, and
// prefix tiles (a few hundred conditioning rows) as they are staged. A score is (q . k) * qs * ks
// with an exact integer dot product. At a DiT's 16384 rows this is ~38 -> ~32 ms, with image
// error below the spread between exact attention implementations.
//
// FP8 P.V form (`PV8`, op parameter `pv` "fp8", prefix-free and one query head a KV head): V as
// E4M3 codes with a scale per column (quant_cols_fp8's), P.V on v_wmma_f32_16x16x16_fp8_fp8 (half
// the cycles of bf16) with P as E4M3 codes of 448 p. E4M3 ends at 448, so the softmax keeps an
// exact running max, taken over a whole 64-key tile so that its test runs once a tile; the row
// sum takes the unrounded 448 p and the epilogue multiplies by the column scale. The V tile is
// staged as the transposing load over pairs of codes, split by byte permutes into V^T rows.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_dt16.h"

#include <cmath>
#include <type_traits>

typedef int v2i_ps __attribute__((ext_vector_type(2)));
typedef int v8i_ps __attribute__((ext_vector_type(8)));

/* quant_act_i8g codes for four values against `inv` = 1 / scale, packed into one dword. */
__device__ __forceinline__ uint32_t ps_q8x4(float a, float b, float c, float d, float inv) {
    auto q = [inv](float x) {
        return (uint32_t)((int)rintf(fminf(fmaxf(x * inv, -127.0f), 127.0f)) & 0xff);
    };
    return q(a) | (q(b) << 8) | (q(c) << 16) | (q(d) << 24);
}
__device__ __forceinline__ float ps_lo(uint32_t w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float ps_hi(uint32_t w) { return __builtin_bit_cast(float, w & 0xffff0000u); }
__device__ __forceinline__ float ps_amax8(uint4 u) {
    return fmaxf(fmaxf(fmaxf(fabsf(ps_lo(u.x)), fabsf(ps_hi(u.x))),
                       fmaxf(fabsf(ps_lo(u.y)), fabsf(ps_hi(u.y)))),
                 fmaxf(fmaxf(fabsf(ps_lo(u.z)), fabsf(ps_hi(u.z))),
                       fmaxf(fabsf(ps_lo(u.w)), fabsf(ps_hi(u.w)))));
}
// log2(448) less 2^-16: exp2 of it is 448 (1 - 1.1e-5), under 448 with room for the hardware
// exp2's error, and still the E4M3 code of 448.
constexpr float kPsLog448 = 8.8073397f;

__device__ __forceinline__ uint2 ps_q8x8(uint4 u, float inv) {
    return make_uint2(ps_q8x4(ps_lo(u.x), ps_hi(u.x), ps_lo(u.y), ps_hi(u.y), inv),
                      ps_q8x4(ps_lo(u.z), ps_hi(u.z), ps_lo(u.w), ps_hi(u.w), inv));
}

template<int HD, int GQA, int NQB, int TILE, bool I8, bool PV8>
__global__ __launch_bounds__(NQB * GQA * 32)
void r4d_attn_ps_kernel(const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
                        const int8_t* __restrict__ gk8, const float* __restrict__ gks,
                        const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
                        const int* __restrict__ cu, const int* __restrict__ cuk, int num_seqs,
                        const int* __restrict__ used,
                        const int* __restrict__ bt, int bt_pitch,
                        const uint16_t* __restrict__ kvc, int kv_heads,
                        float scale, int causal, int q_ld, int k_ld, int k8_ld, int ks_ld,
                        int v_ld, int o_ld, const float* __restrict__ gvs)
{
    typedef DT16<0> D;
    typedef D::frag frag16;
    typedef typename std::conditional<I8, v2i_ps, frag16>::type qfrag;

    constexpr int NWARPS = NQB * GQA;
    constexpr int NTHR   = NWARPS * 32;
    constexpr int BQ     = NQB * 16;
    constexpr int NKS    = HD / 16;
    constexpr int MT     = TILE / 16;
    constexpr int KSTR   = HD + 8;
    constexpr int VSTR   = TILE + 8;
    constexpr int DG     = HD / 8;
    constexpr int KGRP   = TILE / 8;
    constexpr int KITEMS = TILE * DG;
    constexpr int KP     = (KITEMS + NTHR - 1) / NTHR;
    constexpr int NTR    = KGRP * DG;
    constexpr int LGRP   = NTHR / 8;
    constexpr int VP     = (NTR + LGRP - 1) / LGRP;
    constexpr float GROW = D::GROW;
    constexpr int BS     = 16;                 // the paged block, libr4d's (block_size == 16)
    static_assert(HD % 16 == 0 && TILE % 16 == 0, "geometry");
    // Int8 key tile: HD bytes per row padded by 8 (34 dwords), so a fragment read's sixteen rows
    // hit distinct bank pairs; written as 8-byte halves since odd rows are only 8-byte aligned.
    // A prefix key spans DG lanes of 8 dims, which must sit in one half-wave for the row absmax.
    constexpr int KSTR8  = HD + 8;
    constexpr int K8G    = HD / 16;            // 16-byte pieces of an own key's row
    constexpr int K8P    = (TILE * K8G + NTHR - 1) / NTHR;
    static_assert(!I8 || (DG == 16 && NTHR % 32 == 0), "an int8 prefix key is one half-wave");
    // E4M3 V^T tile: HD rows of TILE key bytes padded by 8, so a fragment read's sixteen rows
    // (8 bytes a lane) fall on distinct bank pairs. One 16-byte unit (16 dims of 8 keys) a thread:
    // unit u is key group u / DG8 and dims 16 (u % DG8), so a wave reads whole half-rows.
    constexpr int VSTR8  = TILE + 8;
    constexpr int DG8    = HD / 16;
    static_assert(!PV8 || (GQA == 1 && KGRP * DG8 == LGRP), "one E4M3 V unit a thread");

    __shared__ __align__(16) uint16_t sK[I8 ? 8 : TILE * KSTR];
    __shared__ __align__(16) int8_t   sK8[I8 ? TILE * KSTR8 : 16];
    __shared__ __align__(16) float    sKs[I8 ? TILE : 4];
    __shared__ __align__(16) uint16_t sV[PV8 ? 8 : HD * VSTR];
    __shared__ __align__(16) uint8_t  sV8[PV8 ? HD * VSTR8 : 16];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c = lane & 15, h = lane >> 4;
    const int g   = warp % GQA;
    const int qbi = warp / GQA;
    const int kvh = blockIdx.y;
    const int hq  = kvh * GQA + g;

    // Flat grid over all sequences' query blocks: block b is the b-th BQ-row block counting through
    // the sequences in order, instead of (longest sequence) x (sequences). Few sequences, so a
    // linear walk.
    int seq = -1, q0 = 0;
    for (int s = 0, b = blockIdx.x; s < num_seqs; ++s) {
        const int nb = (cu[s + 1] - cu[s] + BQ - 1) / BQ;
        if (b < nb) { seq = s; q0 = b * BQ; break; }
        b -= nb;
    }
    if (seq < 0) return;
    const int s0 = cu[seq], slen = cu[seq + 1] - s0;
    if (slen <= 0 || q0 >= slen) return;
    const int qlast = min(q0 + BQ, slen) - 1;
    const int P = used ? used[seq] : 0;
    // Own keys: beside the queries, or (`cuk`, attn_cross) a run of their own; a query sees key j
    // of its run, counted from the run's start, when j <= its own row with `causal`.
    const int ks0 = cuk ? cuk[seq] : s0, klen = cuk ? cuk[seq + 1] - ks0 : slen;
    const int ke = causal ? min(qlast + 1, klen) : klen;   // own keys any query here can see

    const float mul = scale * 1.44269504089f;
    const int  qpos = q0 + qbi * 16 + c;
    const bool live = qpos < slen;

    // The query: bf16 fragments pre-multiplied by the softmax scale, or int8 codes against the
    // row absmax (a lane holds half the head's dims, lane + 16 the other half) with the scale in
    // `qs`, applied to the integer score.
    qfrag qf[NKS];
    float qs = mul;
    {
        const int qrow = live ? qpos : slen - 1;
        const uint16_t* qp = gq + (long)(s0 + qrow) * q_ld + (long)hq * HD;
        if constexpr (I8) {
            uint4 qu[NKS];
            float am = 0.0f;
            #pragma unroll
            for (int t = 0; t < NKS; ++t) {
                qu[t] = *(const uint4*)(qp + 16 * t + 8 * h);
                am = fmaxf(am, ps_amax8(qu[t]));
            }
            am = fmaxf(am, swap16(am));
            const float sq = am > 0.0f ? am * (1.0f / 127.0f) : 1.0f;
            const float inv = 1.0f / sq;
            qs = sq * mul;
            #pragma unroll
            for (int t = 0; t < NKS; ++t) {
                const uint2 w = ps_q8x8(qu[t], inv);
                qf[t] = (v2i_ps){ (int)w.x, (int)w.y };
            }
        } else {
            #pragma unroll
            for (int t = 0; t < NKS; ++t) {
                const uint4 u = *(const uint4*)(qp + 16 * t + 8 * h);
                auto cv = [&](uint32_t w) {
                    return D::pk(__builtin_bit_cast(float, w << 16) * mul,
                                 __builtin_bit_cast(float, w & 0xffff0000u) * mul);
                };
                qf[t] = D::mk(make_uint2(cv(u.x), cv(u.y)), make_uint2(cv(u.z), cv(u.w)));
            }
        }
    }

    float m_ref = -3.0e38f, soff = 3.0e38f, l_i = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};

    const int vj = tid & 7, lgi = tid >> 3;
    const uint16_t* kbase = gk + (long)kvh * HD;
    const uint16_t* vbase = gv + (long)kvh * HD;
    const int* btrow = bt ? bt + (long)seq * bt_pitch : nullptr;

    // The K row of key `kk` of the current source; its V row is `vrow`.
    auto krow = [&](bool pre, int kk) -> const uint16_t* {
        if (!pre) return kbase + (long)(ks0 + kk) * k_ld;
        const int blk = btrow[kk / BS];
        return kvc + ((long)(blk * kv_heads + kvh) * BS + (kk % BS)) * (2 * HD);
    };
    auto vrow = [&](bool pre, int kk) -> const uint16_t* {
        if (!pre) return vbase + (long)(ks0 + kk) * v_ld;
        const int blk = btrow[kk / BS];
        return kvc + ((long)(blk * kv_heads + kvh) * BS + (kk % BS)) * (2 * HD) + HD;
    };

    const int ntp = (P + TILE - 1) / TILE, nts = (ke + TILE - 1) / TILE;
    for (int ti = 0; ti < ntp + nts; ++ti) {
        const bool pre = ti < ntp;
        const int k0 = pre ? ti * TILE : (ti - ntp) * TILE;
        const int klim = pre ? P : ke;
        const bool mask = (k0 + TILE > klim) || (!pre && causal && k0 + TILE - 1 > q0);

        // An int8 own tile is K8P 16-byte pieces per thread plus a scale from each of the first
        // TILE threads; otherwise bf16 rows (an int8 prefix tile is quantised below).
        const bool own8 = I8 && !pre;
        uint4 kr[KP], vr[VP];
        float ksr = 0.0f;
        if (own8) {
            #pragma unroll
            for (int p = 0; p < K8P; ++p) {
                const int i = tid + p * NTHR;
                if ((TILE * K8G) % NTHR == 0 || i < TILE * K8G) {
                    const int key = i / K8G, gg = i % K8G;
                    const int kk = (k0 + key < klim) ? k0 + key : klim - 1;
                    kr[p] = *(const uint4*)(gk8 + (long)(ks0 + kk) * k8_ld + (long)kvh * HD + gg * 16);
                }
            }
            if (tid < TILE) {
                const int kk = (k0 + tid < klim) ? k0 + tid : klim - 1;
                ksr = gks[(long)(ks0 + kk) * ks_ld + kvh];
            }
        } else {
            #pragma unroll
            for (int p = 0; p < KP; ++p) {
                const int i = tid + p * NTHR;
                if (KITEMS % NTHR == 0 || i < KITEMS) {
                    const int key = i / DG, gg = i % DG;
                    const int kk = (k0 + key < klim) ? k0 + key : klim - 1;
                    kr[p] = *(const uint4*)(krow(pre, kk) + gg * 8);
                }
            }
        }
        if constexpr (PV8) {
            const int kk0 = k0 + (lgi / DG8) * 8 + vj;
            const int kk = (kk0 < klim) ? kk0 : klim - 1;
            const uint8_t* v8 = (const uint8_t*)gv + (long)(ks0 + kk) * v_ld + (long)kvh * HD;
            vr[0] = __builtin_bit_cast(uint4, load_tr_b128(v8 + (lgi % DG8) * 16));
        } else {
            #pragma unroll
            for (int r = 0; r < VP; ++r) {
                const int u = lgi + r * LGRP;
                if (NTR % LGRP == 0 || u < NTR) {
                    const int kg = u % KGRP, dg = u / KGRP;
                    const int kk0 = k0 + kg * 8 + vj;
                    const int kk = (kk0 < klim) ? kk0 : klim - 1;
                    vr[r] = __builtin_bit_cast(uint4, load_tr_b128(vrow(pre, kk) + dg * 8));
                }
            }
        }
        // Int8 prefix key: row absmax over its DG lanes (one half-wave, DPP butterfly), then eight
        // codes per lane.
        uint2 kq[KP];
        float kqs[KP];
        if (I8 && pre) {
            #pragma unroll
            for (int p = 0; p < KP; ++p) {
                float am = ps_amax8(kr[p]);
                am = fmaxf(am, r4d_xor_lane<8>(am));
                am = fmaxf(am, r4d_xor_lane<4>(am));
                am = fmaxf(am, r4d_xor_lane<2>(am));
                am = fmaxf(am, r4d_xor_lane<1>(am));
                kqs[p] = am > 0.0f ? am * (1.0f / 127.0f) : 1.0f;
                kq[p] = ps_q8x8(kr[p], 1.0f / kqs[p]);
            }
        }

        lds_barrier();
        if constexpr (PV8) {
            // Lane j holds dims 2 j and 2 j + 1 of the group's eight keys interleaved, a key a
            // 16-bit pair: even bytes are one V^T row, odd bytes the next.
            const int row = (lgi % DG8) * 16 + 2 * vj, col = (lgi / DG8) * 8;
            const uint4 w = vr[0];
            *(uint2*)&sV8[row * VSTR8 + col] =
                make_uint2(__builtin_amdgcn_perm(w.y, w.x, 0x06040200u),
                           __builtin_amdgcn_perm(w.w, w.z, 0x06040200u));
            *(uint2*)&sV8[(row + 1) * VSTR8 + col] =
                make_uint2(__builtin_amdgcn_perm(w.y, w.x, 0x07050301u),
                           __builtin_amdgcn_perm(w.w, w.z, 0x07050301u));
        } else {
            #pragma unroll
            for (int r = 0; r < VP; ++r) {
                const int u = lgi + r * LGRP;
                if (NTR % LGRP == 0 || u < NTR) {
                    const int kg = u % KGRP, dg = u / KGRP;
                    *(uint4*)&sV[(dg * 8 + vj) * VSTR + kg * 8] = vr[r];
                }
            }
        }
        if constexpr (I8) {
            if (!pre) {
                #pragma unroll
                for (int p = 0; p < K8P; ++p) {
                    const int i = tid + p * NTHR;
                    if ((TILE * K8G) % NTHR == 0 || i < TILE * K8G) {
                        int8_t* d = &sK8[(i / K8G) * KSTR8 + (i % K8G) * 16];
                        *(uint2*)d       = make_uint2(kr[p].x, kr[p].y);
                        *(uint2*)(d + 8) = make_uint2(kr[p].z, kr[p].w);
                    }
                }
                if (tid < TILE) sKs[tid] = ksr;
            } else {
                #pragma unroll
                for (int p = 0; p < KP; ++p) {
                    const int i = tid + p * NTHR;
                    if (KITEMS % NTHR == 0 || i < KITEMS) {
                        *(uint2*)&sK8[(i / DG) * KSTR8 + (i % DG) * 8] = kq[p];
                        if (i % DG == 0) sKs[i / DG] = kqs[p];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int p = 0; p < KP; ++p) {
                const int i = tid + p * NTHR;
                if (KITEMS % NTHR == 0 || i < KITEMS)
                    *(uint4*)&sK[(i / DG) * KSTR + (i % DG) * 8] = kr[p];
            }
        }
        lds_barrier();

        if constexpr (PV8) {
            // The whole tile's scores, one exact running max over them, P as E4M3 codes of 448 p,
            // then the tile's fp8 WMMAs. The max is kept in the scores' own units (the int8
            // form's are unscaled, `qs` being one positive factor a row), so s - max is at most 0
            // exactly and x = exp2((s - max) qs + kPsLog448) cannot pass exp2(kPsLog448) < 448,
            // where v_cvt_pk_fp8_f32 would give NaN. x is 448 p up to a factor common to every
            // key, which the row sum shares.
            float sc[MT][8];
            #pragma unroll
            for (int mt = 0; mt < MT; mt += (I8 ? 2 : 1)) {
                if constexpr (I8) {
                    // Two blocks' int8 chains interleaved: each WMMA waits on the one two back.
                    v8i_ps s0 = (v8i_ps){0, 0, 0, 0, 0, 0, 0, 0}, s1 = s0;
                    const int8_t* r0 = &sK8[(mt * 16 + c) * KSTR8] + 8 * h;
                    const int8_t* r1 = r0 + 16 * KSTR8;
                    #pragma unroll
                    for (int t0 = 0; t0 < NKS; t0 += 4) {
                        v2i_ps k0[4], k1[4];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const uint2 u0 = *(const uint2*)(r0 + 16 * (t0 + j));
                            const uint2 u1 = *(const uint2*)(r1 + 16 * (t0 + j));
                            k0[j] = (v2i_ps){ (int)u0.x, (int)u0.y };
                            k1[j] = (v2i_ps){ (int)u1.x, (int)u1.y };
                        }
                        sched_barrier();
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            s0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                                true, k0[j], true, qf[t0 + j], s0, false);
                            s1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                                true, k1[j], true, qf[t0 + j], s1, false);
                        }
                    }
                    #pragma unroll
                    for (int b = 0; b < 2; ++b) {
                        const float4 ka = *(const float4*)&sKs[(mt + b) * 16 + 8 * h];
                        const float4 kb = *(const float4*)&sKs[(mt + b) * 16 + 8 * h + 4];
                        const float ks[8] = { ka.x, ka.y, ka.z, ka.w, kb.x, kb.y, kb.z, kb.w };
                        #pragma unroll
                        for (int e = 0; e < 8; ++e)
                            sc[mt + b][e] = (float)(b ? s1[e] : s0[e]) * ks[e];
                    }
                } else {
                    v8f s = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
                    const uint16_t* kr2 = &sK[(mt * 16 + c) * KSTR] + 8 * h;
                    #pragma unroll
                    for (int t0 = 0; t0 < NKS; t0 += 4) {
                        frag16 kf[4];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const uint4 u = *(const uint4*)(kr2 + 16 * (t0 + j));
                            kf[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        }
                        sched_barrier();
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) s = D::wmma(kf[j], qf[t0 + j], s);
                    }
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) sc[mt][e] = s[e];
                }
            }
            if (mask) {
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const int kq0 = k0 + mt * 16 + 8 * h;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int key = kq0 + e;
                        const bool ok = key < klim && (!causal || key <= qpos);
                        sc[mt][e] = ok ? sc[mt][e] : -INFINITY;
                    }
                }
            }
            float smax = sc[0][0];
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                #pragma unroll
                for (int e = 0; e < 8; ++e) smax = fmaxf(smax, sc[mt][e]);
            if (wave_any(smax > m_ref)) {
                const float rmax  = fmaxf(smax, swap16(smax));
                const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
                const float alpha = ex2(I8 ? (m_ref - mnew) * qs : m_ref - mnew);
                #pragma unroll
                for (int t = 0; t < NKS; ++t)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[t][e] *= alpha;
                l_i *= alpha;
                m_ref = mnew;
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                float x[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float d = sc[mt][e] - m_ref;
                    x[e] = ex2(I8 ? __builtin_fmaf(d, qs, kPsLog448) : d + kPsLog448);
                }
                l_i += ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7]));
                // The first conversion's high half is dead (the second writes it), so any live
                // register serves as its old value: no zeroing moves.
                int w0 = __builtin_amdgcn_cvt_pk_fp8_f32(x[0], x[1], __builtin_bit_cast(int, x[0]),
                                                         false);
                w0 = __builtin_amdgcn_cvt_pk_fp8_f32(x[2], x[3], w0, true);
                int w1 = __builtin_amdgcn_cvt_pk_fp8_f32(x[4], x[5], __builtin_bit_cast(int, x[4]),
                                                         false);
                w1 = __builtin_amdgcn_cvt_pk_fp8_f32(x[6], x[7], w1, true);
                const v2i_ps p8 = (v2i_ps){ w0, w1 };
                const uint8_t* vr2 = &sV8[c * VSTR8 + mt * 16 + 8 * h];
                #pragma unroll
                for (int t0 = 0; t0 < NKS; t0 += 4) {
                    v2i_ps vf[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint2 u = *(const uint2*)(vr2 + (t0 + j) * 16 * VSTR8);
                        vf[j] = (v2i_ps){ (int)u.x, (int)u.y };
                    }
                    sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        acc[t0 + j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                            vf[j], p8, acc[t0 + j]);
                }
            }
            continue;
        }

        #pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            // Scores of 16 keys, in log2 units once multiplied by `qs` (int8 only; the bf16 query
            // already carries the scale).
            float sc[8];
            if constexpr (I8) {
                v8i_ps si = (v8i_ps){0, 0, 0, 0, 0, 0, 0, 0};
                const int8_t* kr2 = &sK8[(mt * 16 + c) * KSTR8] + 8 * h;
                #pragma unroll
                for (int t0 = 0; t0 < NKS; t0 += 4) {
                    v2i_ps kf[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) {
                            const uint2 u = *(const uint2*)(kr2 + 16 * (t0 + j));
                            kf[j] = (v2i_ps){ (int)u.x, (int)u.y };
                        }
                    sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS)
                            si = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, kf[j], true,
                                                                                 qf[t0 + j], si, false);
                }
                const float4 ka = *(const float4*)&sKs[mt * 16 + 8 * h];
                const float4 kb = *(const float4*)&sKs[mt * 16 + 8 * h + 4];
                const float ks[8] = { ka.x, ka.y, ka.z, ka.w, kb.x, kb.y, kb.z, kb.w };
                #pragma unroll
                for (int e = 0; e < 8; ++e) sc[e] = (float)si[e] * ks[e];
            } else {
                v8f s = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
                const uint16_t* kr2 = &sK[(mt * 16 + c) * KSTR] + 8 * h;
                #pragma unroll
                for (int t0 = 0; t0 < NKS; t0 += 4) {
                    frag16 kf[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) {
                            const uint4 u = *(const uint4*)(kr2 + 16 * (t0 + j));
                            kf[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        }
                    sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) s = D::wmma(kf[j], qf[t0 + j], s);
                }
                #pragma unroll
                for (int e = 0; e < 8; ++e) sc[e] = s[e];
            }
            if (mask) {
                const int kq0 = k0 + mt * 16 + 8 * h;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int key = kq0 + e;
                    const bool ok = key < klim && (pre || !causal || key <= qpos);
                    sc[e] = ok ? sc[e] : -INFINITY;
                }
            }

            // qs > 0, so the max of the scaled scores is the scaled max.
            float smax = sc[0];
            #pragma unroll
            for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
            if (I8) smax *= qs;
            if (wave_any(smax > m_ref + GROW)) {
                const float rmax  = fmaxf(smax, swap16(smax));
                const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
                const float alpha = ex2(m_ref - mnew);
                #pragma unroll
                for (int t = 0; t < NKS; ++t)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[t][e] *= alpha;
                l_i *= alpha;
                m_ref = mnew;
                soff  = -mnew;
            }

            // P to bf16 by truncation: one v_perm per pair, where gfx1201's software
            // round-to-nearest costs ~6 VALU per value on a SIMD shared with the WMMA (4-7% of the
            // kernel at 16384 rows). The downward bias cancels because the row sum below uses the
            // same truncated values.
            uint32_t pw[4];
            #pragma unroll
            for (int e = 0; e < 8; e += 2) {
                const float a0 = I8 ? __builtin_fmaf(sc[e], qs, soff) : sc[e] + soff;
                const float a1 = I8 ? __builtin_fmaf(sc[e + 1], qs, soff) : sc[e + 1] + soff;
                const uint32_t lo = __builtin_bit_cast(uint32_t, ex2(a0));
                const uint32_t hi = __builtin_bit_cast(uint32_t, ex2(a1));
                pw[e >> 1] = __builtin_amdgcn_perm(hi, lo, 0x07060302u);
            }
            // Row sum from the packed P, so the denominator matches the values P.V used.
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                l_i += __builtin_bit_cast(float, pw[e] << 16) +
                       __builtin_bit_cast(float, pw[e] & 0xffff0000u);
            const frag16 p16 = D::mk(make_uint2(pw[0], pw[1]), make_uint2(pw[2], pw[3]));

            {
                const uint16_t* vr2 = &sV[c * VSTR + mt * 16] + 8 * h;
                #pragma unroll
                for (int t0 = 0; t0 < NKS; t0 += 4) {
                    frag16 vf[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) {
                            const uint4 u = *(const uint4*)(vr2 + (t0 + j) * 16 * VSTR);
                            vf[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        }
                    sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) acc[t0 + j] = D::wmma(vf[j], p16, acc[t0 + j]);
                }
            }
        }
    }

    if (!live) return;
    const float ls = l_i + swap16(l_i);
    const float inv = (ls > 0.0f) ? (1.0f / ls) : 0.0f;
    uint16_t* op = go + (long)(s0 + qpos) * o_ld + (long)hq * HD;
    #pragma unroll
    for (int t = 0; t < NKS; ++t) {
        uint32_t tw[4];
        if constexpr (PV8) {
            // The column scales of the lane's eight dims.
            const float4 sa = *(const float4*)(gvs + (long)hq * HD + t * 16 + 8 * h);
            const float4 sb = *(const float4*)(gvs + (long)hq * HD + t * 16 + 8 * h + 4);
            const float vsc[8] = { sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w };
            #pragma unroll
            for (int e = 0; e < 8; ++e) acc[t][e] *= vsc[e];
        }
        #pragma unroll
        for (int e = 0; e < 8; e += 2)
            tw[e >> 1] = (uint32_t)f32_to_bf16(acc[t][e] * inv)
                       | ((uint32_t)f32_to_bf16(acc[t][e + 1] * inv) << 16);
        *(uint4*)(op + t * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
    }
}

template<int HD, int GQA, int NQB, int TILE, bool I8, bool PV8>
static int r4d_attn_ps_launch(const void* q, const void* k, const void* k8, const float* ks,
                              const void* v, void* o, const int* cu, const int* cuk,
                              const int* used, const int* bt, int bt_pitch, const void* kvc,
                              int num_seqs, int max_q, int n_head_kv, float scale, int causal,
                              int q_ld, int k_ld, int k8_ld, int ks_ld, int v_ld, int o_ld,
                              const float* vs, hipStream_t s)
{
    constexpr int BQ = NQB * 16;
    // At most total / BQ full blocks plus one partial block per sequence.
    dim3 grid((max_q + BQ - 1) / BQ + num_seqs, n_head_kv, 1);
    hipLaunchKernelGGL((r4d_attn_ps_kernel<HD, GQA, NQB, TILE, I8, PV8>), grid,
                       dim3(NQB * GQA * 32), 0, s,
                       (const uint16_t*)q, (const uint16_t*)k, (const int8_t*)k8, ks,
                       (const uint16_t*)v, (uint16_t*)o, cu, cuk, num_seqs, used, bt, bt_pitch,
                       (const uint16_t*)kvc, n_head_kv, scale, causal, q_ld, k_ld, k8_ld, ks_ld,
                       v_ld, o_ld, vs);
    const hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -100 - (int)e;
}

/* head_dim 128 at gqa 1, 2, 4 and 8, and 64 (an audio stream's heads) without a prefix or the
 * int8 and fp8 forms. Null `used`/`bt`/`kvc` means no prefix (the dense form).
 * `max_q` is the total query rows over all sequences and bounds the flat grid. Non-null `k8`
 * selects the int8 Q.K form (own-key codes, `k8_ld` bytes per row; a scale per (row, head) in
 * `ks`, `ks_ld` floats per row), gqa 1 only. Non-null `vs` selects the fp8 P.V form (`v` as E4M3
 * codes, `v_ld` bytes per row; `vs` their column scales), gqa 1 and no prefix only. */
static int r4d_attn_ps(const void* q, const void* k, const void* k8, const float* ks,
                       const void* v, void* o, const int* cu,
                       const int* used, const int* bt, int bt_pitch, const void* kvc,
                       int num_seqs, int max_q, int n_head, int n_head_kv, int head_dim,
                       float scale, int causal, int q_ld, int k_ld, int k8_ld, int ks_ld,
                       int v_ld, int o_ld, hipStream_t s, const float* vs = nullptr,
                       const int* cuk = nullptr)
{
    if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv) return -2;
    if ((q_ld | k_ld | v_ld | o_ld) & 7) return -3;
    if (k8 && ((k8_ld & 15) || ((uintptr_t)k8 & 15) || !ks)) return -3;
    if (num_seqs <= 0 || max_q <= 0) return 0;
    const int gqa = n_head / n_head_kv;
    if (head_dim == 64) {
        if (k8 || vs || used || bt || kvc) return -1;
#define R4D_PS64(G, NQB)                                                                          \
        return r4d_attn_ps_launch<64, G, NQB, 64, false, false>(q, k, k8, ks, v, o, cu, cuk, used, \
                                                                bt, bt_pitch, kvc, num_seqs,      \
                                                                max_q, n_head_kv, scale, causal,  \
                                                                q_ld, k_ld, k8_ld, ks_ld, v_ld,   \
                                                                o_ld, nullptr, s)
        switch (gqa) {
            case 1: R4D_PS64(1, 16);
            case 2: R4D_PS64(2, 4);
            case 4: R4D_PS64(4, 2);
            case 8: R4D_PS64(8, 1);
            default: return -1;
        }
#undef R4D_PS64
    }
    if (head_dim != 128) return -1;
#define R4D_PS(G, NQB, T, I8)                                                                     \
    return r4d_attn_ps_launch<128, G, NQB, T, I8, false>(q, k, k8, ks, v, o, cu, cuk, used, bt,   \
                                                         bt_pitch, kvc, num_seqs, max_q,          \
                                                         n_head_kv, scale, causal, q_ld, k_ld,    \
                                                         k8_ld, ks_ld, v_ld, o_ld, nullptr, s)
    /* Sixteen query tiles per workgroup at gqa 1: a workgroup streams its head's whole K and V
     * once, so more rows per workgroup divide that traffic. At 16384 rows a head's K/V (8 MB) do
     * not fit L2; 128 rows per workgroup gave 66 TFLOP/s, 256 gave 111, 512 gave 102. */
    if (vs) {
        if (gqa != 1 || used || bt || kvc || (v_ld & 15) || ((uintptr_t)v & 15)) return -1;
        if (k8)
            return r4d_attn_ps_launch<128, 1, 16, 64, true, true>(
                q, k, k8, ks, v, o, cu, cuk, used, bt, bt_pitch, kvc, num_seqs, max_q, n_head_kv,
                scale, causal, q_ld, k_ld, k8_ld, ks_ld, v_ld, o_ld, vs, s);
        return r4d_attn_ps_launch<128, 1, 16, 64, false, true>(
            q, k, k8, ks, v, o, cu, cuk, used, bt, bt_pitch, kvc, num_seqs, max_q, n_head_kv, scale,
            causal, q_ld, k_ld, k8_ld, ks_ld, v_ld, o_ld, vs, s);
    }
    if (k8) {
        if (gqa != 1) return -1;
        R4D_PS(1, 16, 64, true);
    }
    switch (gqa) {
        case 1: R4D_PS(1, 16, 64, false);
        case 2: R4D_PS(2, 4, 64, false);
        case 4: R4D_PS(4, 2, 64, false);
        case 8: R4D_PS(8, 1, 64, false);
        default: return -1;
    }
#undef R4D_PS
}

// ============================================================== the plugin ABI entry points
enum { PS_Q = 0, PS_K, PS_V, PS_KV, PS_BT, PS_USED, PS_CU, PS_KDES, PS_VDES, PS_K8, PS_KS,
       PS_OUT, PS_VS };

extern "C" int r4d_rad_attn_prefix_self(const RadArgs* a, RadStream s) {
    const RadTensor* q  = r4d_opt(a, PS_Q);
    const RadTensor* k  = r4d_opt(a, PS_K);
    const RadTensor* v  = r4d_opt(a, PS_V);
    const RadTensor* kv = r4d_opt(a, PS_KV);
    const RadTensor* bt = r4d_opt(a, PS_BT);
    const RadTensor* su = r4d_opt(a, PS_USED);
    const RadTensor* cu = r4d_opt(a, PS_CU);
    const RadTensor* o  = r4d_opt(a, PS_OUT);
    if (!q || !k || !v || !cu || !o) return R4D_NO(RAD_E_INVAL);
    /* The paged prefix is the cache, its block table and the per-sequence prefix lengths, all
     * three or none: without them the op is self attention over each sequence's rows. */
    const bool pre = kv != nullptr;
    if (pre != (bt != nullptr) || pre != (su != nullptr)) return R4D_NO(RAD_E_INVAL);
    if (r4d_opt(a, PS_KDES) || r4d_opt(a, PS_VDES)) return R4D_NO(RAD_E_UNSUPPORTED);
    if ((pre && kv->dtype != RAD_BF16) || q->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    /* The fp8 P.V form (`pv` "fp8"): `v` as E4M3 codes [tokens, kv_heads * head_dim] and their
     * column scales `v_scale` [kv_heads * head_dim], as quant_cols_fp8 writes them; prefix-free,
     * one query head a KV head. */
    const RadTensor* vs = r4d_opt(a, PS_VS);
    const char* pvs = r4d_gets(a, "pv");
    const bool pv8 = pvs && !std::strcmp(pvs, "fp8");
    if (pvs && !pv8 && std::strcmp(pvs, "bf16")) return R4D_NO(RAD_E_INVAL);
    if (pv8 != (vs != nullptr)) return R4D_NO(RAD_E_INVAL);
    if (pv8 && pre) return R4D_NO(RAD_E_UNSUPPORTED);
    if (pv8 && (v->dtype != RAD_F8E4M3 || vs->dtype != RAD_F32)) return R4D_NO(RAD_E_DTYPE);
    if (pv8 && r4d_geti_or(a, "gqa", 1) != 1) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long hd = r4d_geti_or(a, "head_dim", 0), gqa = r4d_geti_or(a, "gqa", 1);
    if (hd != 128 || gqa <= 0) return R4D_NO(RAD_E_SHAPE);
    if (pre && (kv->rank != 4 || kv->shape[2] != 16 || kv->shape[3] != 2 * hd))
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(cu) || (pre && !r4d_contig(su)) || cu->rank != 1 || cu->shape[0] < 2)
        return R4D_NO(RAD_E_SHAPE);
    const int num_seqs = (int)(cu->shape[0] - 1);
    if (pre && r4d_numel(su) < num_seqs) return R4D_NO(RAD_E_SHAPE);
    long long total = 0, nh = 0, q_ld = 0, tk = 0, hk = 0, k_ld = 0, tv = 0, hv = 0, v_ld = 0;
    long long to = 0, ho = 0, o_ld = 0;
    if (!r4d_head_rows(q, hd, &total, &nh, &q_ld) ||
        !r4d_head_rows(k, hd, &tk, &hk, &k_ld) ||
        !r4d_head_rows(v, hd, &tv, &hv, &v_ld) ||
        !r4d_head_rows(o, hd, &to, &ho, &o_ld))
        return R4D_NO(RAD_E_STRIDE);
    if (hk != hv || ho != nh || hk * gqa != nh || (pre && hk != kv->shape[1]))
        return R4D_NO(RAD_E_SHAPE);
    if (tk < total || tv < total || to < total) return R4D_NO(RAD_E_SHAPE);
    const long long bt_pitch = !pre ? 0 : bt->rank >= 2 ? bt->shape[1] : r4d_numel(bt) / num_seqs;
    if (pre && bt->rank >= 2 && bt->stride[1] != 1) return R4D_NO(RAD_E_STRIDE);
    const long long mq = total;
    /* Int8 Q.K form (`qk` "i8"): own-key codes [tokens, kv_heads * head_dim] and scales
     * [tokens, kv_heads], as quant_act_i8g writes them at group head_dim. */
    const RadTensor* k8 = r4d_opt(a, PS_K8);
    const RadTensor* ks = r4d_opt(a, PS_KS);
    const char* qk = r4d_gets(a, "qk");
    const bool i8 = qk && !std::strcmp(qk, "i8");
    if (qk && !i8 && std::strcmp(qk, "bf16")) return R4D_NO(RAD_E_INVAL);
    if (i8 != (k8 != nullptr) || i8 != (ks != nullptr)) return R4D_NO(RAD_E_INVAL);
    long long k8_ld = 0, ks_ld = 0;
    if (k8) {
        if (k8->dtype != RAD_I8 || ks->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
        long long t8 = 0, h8 = 0, ts = 0;
        if (!r4d_head_rows(k8, hd, &t8, &h8, &k8_ld)) return R4D_NO(RAD_E_STRIDE);
        ks_ld = r4d_rows_pitch(ks, hk, &ts);
        if (ks_ld < 0) return R4D_NO(RAD_E_STRIDE);
        if (h8 != hk || t8 < total || ts < total) return R4D_NO(RAD_E_SHAPE);
    }
    if (pv8) {
        if (!r4d_contig(vs) || ((uintptr_t)vs->data & 15) || (v_ld & 15) ||
            ((uintptr_t)v->data & 15))
            return R4D_NO(RAD_E_STRIDE);
        if (r4d_numel(vs) < hk * hd) return R4D_NO(RAD_E_SHAPE);
    }
    const float scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)hd));
    const int e = r4d_attn_ps(q->data, k->data, k8 ? k8->data : nullptr,
                              ks ? (const float*)ks->data : nullptr, v->data, o->data,
                              (const int*)cu->data, pre ? (const int*)su->data : nullptr,
                              pre ? (const int*)bt->data : nullptr, (int)bt_pitch,
                              pre ? kv->data : nullptr, num_seqs, (int)mq, (int)nh, (int)hk, (int)hd,
                              scale, (int)r4d_geti_or(a, "causal", 0), (int)q_ld, (int)k_ld,
                              (int)k8_ld, (int)ks_ld, (int)v_ld, (int)o_ld, (hipStream_t)s,
                              pv8 ? (const float*)vs->data : nullptr);
    if (e == -1 || e == -2) return R4D_NO(RAD_E_SHAPE);
    if (e == 0) return RAD_OK;
    return R4D_NO(e == -3 ? RAD_E_STRIDE : e <= -100 ? RAD_E_DEVICE : RAD_E_SHAPE);
}

/* attn_dense at head_dim 128, causal or not: the same kernel with no prefix. */
extern "C" int r4d_rad_attn_dense_h128(const RadArgs* a, RadStream s) {
    const RadTensor* q  = r4d_opt(a, 0);
    const RadTensor* k  = r4d_opt(a, 1);
    const RadTensor* v  = r4d_opt(a, 2);
    const RadTensor* cu = r4d_opt(a, 3);
    const RadTensor* o  = r4d_opt(a, 4);
    if (!q || !k || !v || !cu || !o) return R4D_NO(RAD_E_INVAL);
    if (!r4d_contig(cu) || cu->rank != 1 || cu->shape[0] < 2) return R4D_NO(RAD_E_SHAPE);
    const long long hd = r4d_geti_or(a, "head_dim", 0), gqa = r4d_geti_or(a, "gqa", 1);
    if (hd != 128 || gqa <= 0) return R4D_NO(RAD_E_SHAPE);
    /* Every key of a segment, its bounds read as they stand: no window, blocks or cu_div. */
    if (r4d_geti_or(a, "window", 0) != 0 || r4d_geti_or(a, "block", 0) != 0 ||
        r4d_geti_or(a, "cu_div", 1) != 1)
        return R4D_NO(RAD_E_UNSUPPORTED);
    long long total = 0, nh = 0, q_ld = 0, tk = 0, hk = 0, k_ld = 0, tv = 0, hv = 0, v_ld = 0;
    long long to = 0, ho = 0, o_ld = 0;
    if (!r4d_head_rows(q, hd, &total, &nh, &q_ld) ||
        !r4d_head_rows(k, hd, &tk, &hk, &k_ld) ||
        !r4d_head_rows(v, hd, &tv, &hv, &v_ld) ||
        !r4d_head_rows(o, hd, &to, &ho, &o_ld))
        return R4D_NO(RAD_E_STRIDE);
    if (hk != hv || ho != nh || hk * gqa != nh) return R4D_NO(RAD_E_SHAPE);
    if (tk < total || tv < total || to < total) return R4D_NO(RAD_E_SHAPE);
    const int num_seqs = (int)(cu->shape[0] - 1);
    const float scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)hd));
    const int e = r4d_attn_ps(q->data, k->data, nullptr, nullptr, v->data, o->data,
                              (const int*)cu->data, nullptr, nullptr, 0, nullptr, num_seqs,
                              (int)total, (int)nh, (int)hk, (int)hd, scale,
                              (int)r4d_geti_or(a, "causal", 0), (int)q_ld, (int)k_ld, 0, 0,
                              (int)v_ld, (int)o_ld, (hipStream_t)s);
    if (e == -1 || e == -2) return R4D_NO(RAD_E_SHAPE);
    if (e == 0) return RAD_OK;
    return R4D_NO(e == -3 ? RAD_E_STRIDE : e <= -100 ? RAD_E_DEVICE : RAD_E_SHAPE);
}

/* attn_cross at head_dim 64 or 128: the dense kernel with each entry's keys a run of their own
 * (`cu_k`), causal counting from each run's start. */
extern "C" int r4d_rad_attn_cross(const RadArgs* a, RadStream s) {
    const RadTensor* q  = r4d_opt(a, 0);
    const RadTensor* k  = r4d_opt(a, 1);
    const RadTensor* v  = r4d_opt(a, 2);
    const RadTensor* cu = r4d_opt(a, 3);
    const RadTensor* ck = r4d_opt(a, 4);
    const RadTensor* o  = r4d_opt(a, 5);
    if (!q || !k || !v || !cu || !ck || !o) return R4D_NO(RAD_E_INVAL);
    if (!r4d_contig(cu) || cu->rank != 1 || cu->shape[0] < 2 || !r4d_contig(ck) ||
        r4d_numel(ck) < cu->shape[0])
        return R4D_NO(RAD_E_SHAPE);
    const long long hd = r4d_geti_or(a, "head_dim", 0), gqa = r4d_geti_or(a, "gqa", 1);
    if ((hd != 64 && hd != 128) || gqa <= 0) return R4D_NO(RAD_E_SHAPE);
    long long total = 0, nh = 0, q_ld = 0, tk = 0, hk = 0, k_ld = 0, tv = 0, hv = 0, v_ld = 0;
    long long to = 0, ho = 0, o_ld = 0;
    if (!r4d_head_rows(q, hd, &total, &nh, &q_ld) ||
        !r4d_head_rows(k, hd, &tk, &hk, &k_ld) ||
        !r4d_head_rows(v, hd, &tv, &hv, &v_ld) ||
        !r4d_head_rows(o, hd, &to, &ho, &o_ld))
        return R4D_NO(RAD_E_STRIDE);
    if (hk != hv || ho != nh || hk * gqa != nh || tk != tv || to < total) return R4D_NO(RAD_E_SHAPE);
    const int num_seqs = (int)(cu->shape[0] - 1);
    const float scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)hd));
    const int e = r4d_attn_ps(q->data, k->data, nullptr, nullptr, v->data, o->data,
                              (const int*)cu->data, nullptr, nullptr, 0, nullptr, num_seqs,
                              (int)total, (int)nh, (int)hk, (int)hd, scale,
                              (int)r4d_geti_or(a, "causal", 0), (int)q_ld, (int)k_ld, 0, 0,
                              (int)v_ld, (int)o_ld, (hipStream_t)s, nullptr,
                              (const int*)ck->data);
    if (e == -1 || e == -2) return R4D_NO(RAD_E_SHAPE);
    if (e == 0) return RAD_OK;
    return R4D_NO(e == -3 ? RAD_E_STRIDE : e <= -100 ? RAD_E_DEVICE : RAD_E_SHAPE);
}

// ============================================================== wide heads (attn_dense, h384 .. h1152)
// Image decoder mid-block attention: one head 384 to 1152 wide over every latent pixel (16384 rows
// at 2048 x 2048). A lane cannot hold a 1152-wide output row (576 accumulator VGPRs), so the head
// is split across the workgroup's eight waves: a wave owns HD/8 dims of the 16-query block.
//
//   S  = Q.K^T in eight partial tiles (one per wave), summed through LDS in fixed wave order so
//        the result is run-to-run deterministic.
//   P  formed by sixteen threads (one per query row), rounded to bf16; the row sum uses the
//        rounded values.
//   O  += P.V over the wave's own dims: V^T straight from global via the transposing load, P^T
//        from LDS.
//
// S^T = K.Q^T as above, so a partial tile's C layout is (query = lane % 16, key = 8 (lane / 16) + e)
// for both the reduction and the P^T fragment.
template<int HD>
__global__ __launch_bounds__(256)
void r4d_attn_wide_kernel(const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
                          const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
                          const int* __restrict__ cu, int num_seqs, float scale, int q_ld,
                          int k_ld, int v_ld, int o_ld)
{
    typedef DT16<0> D;
    constexpr int DW = HD / 8;          // dims a wave
    constexpr int NS = DW / 16;         // its 16-wide steps
    constexpr int KT = 64;              // keys a tile
    constexpr int PP = KT + 8;          // P's LDS row pitch, in bf16
    static_assert(HD % 128 == 0, "eight waves of whole 16-wide steps");

    __shared__ float sPart[8 * KT * 16];
    __shared__ float sS[KT * 16];
    __shared__ __align__(16) uint16_t sP[16 * PP];
    __shared__ float sAlpha[16], sL[16];

    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int c = lane & 15, h = lane >> 4;
    const int head = blockIdx.y;

    int seq = -1, q0 = 0;
    for (int s = 0, b = blockIdx.x; s < num_seqs; ++s) {
        const int nb = (cu[s + 1] - cu[s] + 15) / 16;
        if (b < nb) { seq = s; q0 = b * 16; break; }
        b -= nb;
    }
    if (seq < 0) return;
    const int s0 = cu[seq], slen = cu[seq + 1] - s0;
    if (slen <= 0 || q0 >= slen) return;

    const int d0 = wave * DW;
    const uint16_t* qb = gq + (long)head * HD + d0;
    const uint16_t* kb = gk + (long)head * HD + d0;
    const uint16_t* vb = gv + (long)head * HD + d0;
    const float mul = scale * 1.44269504089f;

    D::frag qf[NS];
    {
        const int qr = min(q0 + c, slen - 1);
        const uint16_t* p = qb + (long)(s0 + qr) * q_ld + 4 * h;
        #pragma unroll
        for (int t = 0; t < NS; ++t) {
            const uint2 lo = *(const uint2*)(p + 16 * t), hi = *(const uint2*)(p + 16 * t + 8);
            qf[t] = D::mk(lo, hi);
        }
    }
    v8f acc[NS];
    #pragma unroll
    for (int t = 0; t < NS; ++t) acc[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    float m_run = -INFINITY, l_run = 0.0f;

    for (int k0 = 0; k0 < slen; k0 += KT) {
        // ---- this wave's share of S^T for the tile's four 16-key blocks
        #pragma unroll
        for (int j = 0; j < KT / 16; ++j) {
            v8f sp = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
            const int kr = min(k0 + 16 * j + c, slen - 1);
            const uint16_t* p = kb + (long)(s0 + kr) * k_ld + 4 * h;
            #pragma unroll
            for (int t = 0; t < NS; ++t) {
                const uint2 lo = *(const uint2*)(p + 16 * t), hi = *(const uint2*)(p + 16 * t + 8);
                sp = D::wmma(D::mk(lo, hi), qf[t], sp);
            }
            #pragma unroll
            for (int e = 0; e < 8; ++e) sPart[wave * KT * 16 + (16 * j + 8 * h + e) * 16 + c] = sp[e];
        }
        __syncthreads();
        for (int u = tid; u < KT * 16; u += 256) {
            float t = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) t += sPart[w * KT * 16 + u];
            sS[u] = t;
        }
        __syncthreads();
        // ---- the softmax, a thread a query row
        if (tid < 16) {
            const int nk = min(KT, slen - k0);
            float mx = m_run;
            for (int key = 0; key < nk; ++key) mx = fmaxf(mx, sS[key * 16 + tid] * mul);
            const float alpha = ex2(m_run - mx);
            float lsum = 0.0f;
            for (int key = 0; key < KT; ++key) {
                const uint16_t pb = key < nk ? f32_to_bf16(ex2(sS[key * 16 + tid] * mul - mx)) : 0;
                sP[tid * PP + key] = pb;
                lsum += __builtin_bit_cast(float, (uint32_t)pb << 16);
            }
            l_run = l_run * alpha + lsum;
            m_run = mx;
            sAlpha[tid] = alpha;
        }
        __syncthreads();
        // ---- O^T += V^T . P^T over the wave's dims
        const float al = sAlpha[c];
        #pragma unroll
        for (int t = 0; t < NS; ++t)
            #pragma unroll
            for (int e = 0; e < 8; ++e) acc[t][e] *= al;
        #pragma unroll
        for (int ks = 0; ks < KT / 16; ++ks) {
            const uint4 pu = *(const uint4*)&sP[c * PP + 16 * ks + 8 * h];
            const D::frag pf = D::mk(make_uint2(pu.x, pu.y), make_uint2(pu.z, pu.w));
            const int kr = min(k0 + 16 * ks + 8 * h + (lane & 7), slen - 1);
            const uint16_t* vp = vb + (long)(s0 + kr) * v_ld + 8 * ((lane >> 3) & 1);
            #pragma unroll
            for (int t = 0; t < NS; ++t) {
                const uint4 u = __builtin_bit_cast(uint4, load_tr_b128(vp + 16 * t));
                acc[t] = D::wmma(D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w)), pf, acc[t]);
            }
        }
    }
    if (tid < 16) sL[tid] = l_run;
    __syncthreads();
    if (q0 + c >= slen) return;
    const float inv = sL[c] > 0.0f ? 1.0f / sL[c] : 0.0f;
    uint16_t* op = go + (long)(s0 + q0 + c) * o_ld + (long)head * HD + d0;
    #pragma unroll
    for (int t = 0; t < NS; ++t) {
        uint32_t tw[4];
        #pragma unroll
        for (int e = 0; e < 8; e += 2)
            tw[e >> 1] = (uint32_t)f32_to_bf16(acc[t][e] * inv)
                       | ((uint32_t)f32_to_bf16(acc[t][e + 1] * inv) << 16);
        *(uint4*)(op + 16 * t + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
    }
}

/* attn_dense at head_dim 768 or 1152 (image encoder/decoder mid block): one query head per KV
 * head, bidirectional. */
extern "C" int r4d_rad_attn_dense_wide(const RadArgs* a, RadStream s) {
    const RadTensor* q  = r4d_opt(a, 0);
    const RadTensor* k  = r4d_opt(a, 1);
    const RadTensor* v  = r4d_opt(a, 2);
    const RadTensor* cu = r4d_opt(a, 3);
    const RadTensor* o  = r4d_opt(a, 4);
    if (!q || !k || !v || !cu || !o) return R4D_NO(RAD_E_INVAL);
    if (!r4d_contig(cu) || cu->rank != 1 || cu->shape[0] < 2) return R4D_NO(RAD_E_SHAPE);
    const long long hd = r4d_geti_or(a, "head_dim", 0);
    if ((hd != 384 && hd != 512 && hd != 768 && hd != 1152) || r4d_geti_or(a, "gqa", 1) != 1 ||
        r4d_geti_or(a, "causal", 0) != 0 ||
        r4d_geti_or(a, "window", 0) != 0 || r4d_geti_or(a, "block", 0) != 0 ||
        r4d_geti_or(a, "cu_div", 1) != 1)
        return R4D_NO(RAD_E_UNSUPPORTED);
    long long total = 0, nh = 0, q_ld = 0, tk = 0, hk = 0, k_ld = 0, tv = 0, hv = 0, v_ld = 0;
    long long to = 0, ho = 0, o_ld = 0;
    if (!r4d_head_rows(q, hd, &total, &nh, &q_ld) ||
        !r4d_head_rows(k, hd, &tk, &hk, &k_ld) ||
        !r4d_head_rows(v, hd, &tv, &hv, &v_ld) ||
        !r4d_head_rows(o, hd, &to, &ho, &o_ld))
        return R4D_NO(RAD_E_STRIDE);
    if (hk != nh || hv != nh || ho != nh) return R4D_NO(RAD_E_SHAPE);
    if (tk < total || tv < total || to < total) return R4D_NO(RAD_E_SHAPE);
    if ((q_ld | k_ld | v_ld | o_ld) & 7) return R4D_NO(RAD_E_STRIDE);
    const int num_seqs = (int)(cu->shape[0] - 1);
    const float scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)hd));
    dim3 grid((unsigned)((total + 15) / 16 + num_seqs), (unsigned)nh);
#define R4D_WIDE(HD)                                                                             \
    hipLaunchKernelGGL((r4d_attn_wide_kernel<HD>), grid, dim3(256), 0, (hipStream_t)s,           \
                       (const uint16_t*)q->data, (const uint16_t*)k->data,                        \
                       (const uint16_t*)v->data, (uint16_t*)o->data, (const int*)cu->data,        \
                       num_seqs, scale, (int)q_ld, (int)k_ld, (int)v_ld, (int)o_ld)
    if (hd == 384)      R4D_WIDE(384);
    else if (hd == 512) R4D_WIDE(512);
    else if (hd == 768) R4D_WIDE(768);
    else                R4D_WIDE(1152);
#undef R4D_WIDE
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}
