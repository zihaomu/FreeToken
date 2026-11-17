/* avx_had.h -- Hadamard rotation, fp8 group quantiser and the shared fp8-fusion epilogue, for the
 * kernels that emit a rotated or block-scaled fp8 row: had_quant_act_fp8, the fused quantisers
 * (avx_quant.cpp), gemm_nt_q_gated (avx_gemm.cpp) and hc_read (avx_hc.cpp).
 * Compiled per level with its includer; include after avx_vec.h and avx_common.h, with
 * `using namespace avx` in effect. */
#ifndef RAD_AVX_HAD_H
#define RAD_AVX_HAD_H

static inline bool pow2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }

/* Widest rotation supported. libr4d caps `had` at 512 (the largest power of two dividing a
 * per-rank K of 8704); 4096 matches libref's cap so both refuse the same requests. A wider request
 * returns RAD_E_UNSUPPORTED rather than rotating a narrower block, whose output would look
 * plausible but be wrong. */
enum { AVX_HAD_MAX = 4096 };

/* ================================================================== the butterfly
 * One stride `h` of the transform over a g-element block; strides run ascending from 1. */
#if AVX_LEVEL >= 1
/* In-register stride for h < VF_N: permute each lane with its partner at stride h, then lanes with
 * (i & h) == 0 take self + partner and the others partner - self. */
static inline void fwht_small(float* v, int64_t g, int64_t h) {
    for (int64_t b = 0; b < g; b += VF_N) {
        vf x = vf_loadu(v + b);
        vf p;
#if AVX_LEVEL == 3
        __mmask16 m;
        switch (h) {
            case 1: p = _mm512_permute_ps(x, 0xB1);            m = 0x5555; break;
            case 2: p = _mm512_permute_ps(x, 0x4E);            m = 0x3333; break;
            case 4: p = _mm512_shuffle_f32x4(x, x, 0xB1);      m = 0x0F0F; break;
            default:p = _mm512_shuffle_f32x4(x, x, 0x4E);      m = 0x00FF; break;  /* h == 8 */
        }
        vf_storeu(v + b, _mm512_mask_blend_ps(m, _mm512_sub_ps(p, x), _mm512_add_ps(x, p)));
#else
        switch (h) {
            case 1:  p = _mm256_permute_ps(x, 0xB1); break;
            case 2:  p = _mm256_permute_ps(x, 0x4E); break;
            default: p = _mm256_permute2f128_ps(x, x, 0x01); break;   /* h == 4 */
        }
        const __m256 sum = _mm256_add_ps(x, p), dif = _mm256_sub_ps(p, x);
        __m256 r;
        switch (h) {
            case 1:  r = _mm256_blend_ps(dif, sum, 0x55); break;
            case 2:  r = _mm256_blend_ps(dif, sum, 0x33); break;
            default: r = _mm256_blend_ps(dif, sum, 0x0F); break;
        }
        vf_storeu(v + b, r);
#endif
    }
}
#endif

static void fwht(float* v, int64_t g) {
    for (int64_t h = 1; h < g; h <<= 1) {
#if AVX_LEVEL >= 1
        if (g >= VF_N && h >= VF_N) {
            for (int64_t i = 0; i < g; i += VF_N)
                if ((i & h) == 0) {
                    const vf a = vf_loadu(v + i), b = vf_loadu(v + i + h);
                    vf_storeu(v + i, vf_add(a, b));
                    vf_storeu(v + i + h, vf_sub(a, b));
                }
            continue;
        }
        if (g >= VF_N) { fwht_small(v, g, h); continue; }
#endif
        for (int64_t i = 0; i < g; ++i)
            if ((i & h) == 0) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
    }
}

/* Rotate a whole row of `n` elements in blocks of `g`, in place. */
static inline void fwht_row(float* v, int64_t n, int64_t g) {
    for (int64_t b = 0; b < n; b += g) fwht(v + b, g);
}

/* ================================================================== the fp8 group
 * out[k0, k1) = v[k0, k1) / scale, clamped to +-top, with scale = amax / top (1 for an all-zero
 * group): `top` 448 is E4M3's finite range, 127 int8's. */
static inline void fp8_group(const float* v, float* out, int64_t k0, int64_t k1, float* scale_out,
                             float top = 448.0f) {
    const float amax = row_amax(v + k0, k1 - k0);
    const float sc = amax > 0.0f ? amax * (1.0f / top) : 1.0f;
    const float inv = 1.0f / sc;
    const vf vi = vf_set1(inv), hi = vf_set1(top), lo = vf_set1(-top);
    int64_t i = k0;
    for (; i + VF_N <= k1; i += VF_N)
        vf_storeu(out + i, vf_min(vf_max(vf_mul(vf_loadu(v + i), vi), lo), hi));
    for (; i < k1; ++i) {
        float t = v[i] * inv;
        out[i] = t > top ? top : (t < -top ? -top : t);
    }
    *scale_out = sc;
}

/* ================================================================== the fp8 fusions' epilogue
 *
 * A fused fp8 op replaces a pair that writes a bf16 tensor and then quantises it. So the codes are
 * taken over the bf16-narrowed value whatever the activation dtype, and `out_bf16` holds that
 * value (libref's contract, ref_quant.cpp). The helpers below therefore narrow before quantising. */

/* Round v[0, n) through bf16 in place. `tmp` holds n floats (room for the n bf16). */
static inline void narrow_bf16(float* v, int64_t n, float* tmp) {
    row_from_f32(RAD_BF16, v, tmp, n);
    row_to_f32(RAD_BF16, tmp, v, n);
}

/* out = act(gate) * up, narrowed to bf16. act 0 is silu (down_proj's input), 1 is sigmoid (a
 * gated attention's o_proj input). `tmp` holds n floats. */
static inline void gate_bf16(const float* gv, const float* uv, float* out, int64_t n, int act,
                             float* tmp) {
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) {
        const vf gt = vf_loadu(gv + i);
        vf_storeu(out + i, vf_mul(act == 1 ? vf_sigmoid(gt) : vf_silu(gt), vf_loadu(uv + i)));
    }
    for (; i < n; ++i) {
        const float sg = 1.0f / (1.0f + std::exp(-gv[i]));
        out[i] = (act == 1 ? sg : gv[i] * sg) * uv[i];
    }
    narrow_bf16(out, n, tmp);
}

/* Writes row r of the codes `q` and group scales `s` over v[0, n), in groups of `g` (the last may
 * be short), and v itself into `ob` when given. `codes` holds n floats. An int8 `q` takes
 * quant_act_i8g's codes (libref/ref_quant.cpp: fp8_row). */
static inline void fp8_emit(const float* v, float* codes, int64_t n, int64_t g, const RadTensor* q,
                            const RadTensor* s, const RadTensor* ob, int64_t r) {
    if (ob) out_row(ob, rowoff(ob, r, n), n, v);
    const int64_t ngrp = (n + g - 1) / g;
    const float top = q->dtype == RAD_I8 ? 127.0f : 448.0f;
    for (int64_t gi = 0; gi < ngrp; ++gi) {
        const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
        float sc;
        fp8_group(v, codes, k0, k1, &sc, top);
        stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
    }
    out_row(q, rowoff(q, r, n), n, codes);
}

#endif /* RAD_AVX_HAD_H */
