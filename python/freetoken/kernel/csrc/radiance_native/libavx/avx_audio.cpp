/* avx_audio.cpp -- the audio generator's ops (docs/OPS.md, "Audio generation"): the vocoder's 1-D
 * convolutions and snake activation, the guided frame sampler and the frame embeddings, the window
 * states of the latent denoiser, and the index helpers that feed them.
 *
 * Bit-exact with libref (libref/ref_audio.cpp) for all but the three vocoder ops: the sampler (one
 * pick decides every later frame), the window noise and its blends (a seed gives the same piece on
 * every plugin), the embeddings, the copies and the index helpers. Where libref multiplies and then
 * adds in two roundings, `keep` stops the compiler from fusing the pair (this plugin is built with
 * -ffp-contract=fast, libref without contraction). conv1d and conv_t1d are GEMMs over bands of
 * patch rows and snake takes a polynomial sine; those three are held to the tolerance table.
 *
 * The per-entry ops (frame_mix_store, end_status, frame_store, win_input, win_prep, win_finish, and
 * cfg_topk_sample with a history) check every entry before writing any, where libref writes the
 * entries before a bad one; they return libref's code, and refuse with RAD_E_SHAPE an entry libref
 * would read or write past an operand for.
 */
#include "avx_vec.h"
#include "avx_common.h"
#include "rad_philox.h"

#include <algorithm>
#include <functional>
#include <vector>

using namespace avx;

namespace avx {
inline namespace AVX_SUFFIX {
/* gemm_nt's product over operands assembled by a composite kernel (avx_gemm.cpp). */
int gemm_nt_into(const RadTensor* A, const RadTensor* B, const RadTensor* bias, const RadTensor* Y,
                 int64_t M, int64_t N, int64_t K);
}  /* inline namespace AVX_SUFFIX */
}  /* namespace avx */

namespace {

/* A window entry's flags (RadAudioGenBatch, RAD_AGEN_F_*), restated as libref restates them, since
 * a kernel library compiles against rad_abi.h only. */
enum { kFirst = 1, kLast = 8, kJoin = 16 };

/* Element `i` of an index operand in logical order (libref's ldi), and entry `i` of a cu_seqlens
 * operand. */
inline int64_t at(const RadTensor* t, int64_t i) { return ldi(t, offlin(t, i)); }
inline int64_t cu_at(const RadTensor* cu, int64_t i) { return at(cu, i); }
/* The sequence that owns row `r` (libref's linear scan). */
inline int64_t seq_of(const RadTensor* cu, int64_t n_seq, int64_t r) {
    int64_t s = 0;
    while (s + 1 < n_seq && cu_at(cu, s + 1) <= r) ++s;
    return s;
}

inline float rbf(float v) { return narrow1(RAD_BF16, v); }

/* v[0, n) rounded through dtype `dt` in place, as a store and reload rounds it. `tmp` holds n
 * floats. */
inline void narrow_row(uint32_t dt, float* v, int64_t n, float* tmp) {
    if (dt == RAD_F32) return;
    row_from_f32(dt, v, tmp, n);
    row_to_f32(dt, tmp, v, n);
}

/* A value the compiler may not fuse into the next add. */
#if AVX_LEVEL == 0
inline vf keep(vf x) { __asm__("" : "+x"(x)); return x; }
#elif AVX_LEVEL == 3
inline vf keep(vf x) { __asm__("" : "+v"(x)); return x; }
inline float keep(float x) { __asm__("" : "+v"(x)); return x; }
#else
inline vf keep(vf x) { __asm__("" : "+x"(x)); return x; }
inline float keep(float x) { __asm__("" : "+x"(x)); return x; }
#endif

/* Round to the nearest integer, ties to even, lane by lane. */
#if AVX_LEVEL == 0
inline vf vf_round(vf x) { return std::nearbyint(x); }
#elif AVX_LEVEL == 3
inline vf vf_round(vf x) {
    return _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}
#else
inline vf vf_round(vf x) { return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
#endif

/* n elements of row `base` of `t` as f32 into dst. */
inline void load_row(const RadTensor* t, int64_t base, int64_t n, float* dst) {
    const float* s = in_row(t, base, n, dst);
    if (s != dst) std::memcpy(dst, s, (size_t)n * sizeof(float));
}

/* A row copied between operands, converted through f32 as libref's element loop is. Whole-byte
 * float rows of one dtype are copied as bytes, which is the same thing for every non-NaN value. */
inline void copy_row(const RadTensor* src, int64_t sb, const RadTensor* dst, int64_t db, int64_t n,
                     float* scr) {
    const uint32_t dt = src->dtype;
    const bool flt = dt == RAD_F32 || dt == RAD_BF16 || dt == RAD_F16 || dt == RAD_F8E4M3 ||
                     dt == RAD_F8E5M2;
    if (flt && dt == dst->dtype && laststride(src) == 1 && laststride(dst) == 1) {
        const size_t w = (size_t)(rad_dtype_bits(dt) / 8);
        std::memcpy((char*)dst->data + (size_t)db * w, (const char*)src->data + (size_t)sb * w,
                    (size_t)n * w);
        return;
    }
    out_row(dst, db, n, in_row(src, sb, n, scr));
}

inline uint64_t seed_of(const RadTensor* seed, int64_t e) {
    return (uint64_t)(uint32_t)at(seed, 2 * e) | ((uint64_t)(uint32_t)at(seed, 2 * e + 1) << 32);
}

/* Elements [e0, e0 + cnt) of the normal stream (seed, stream) into out (abi/rad_philox.h): the same
 * arithmetic as rad_philox_normal_s, with one Philox block and one log per element pair instead of
 * one each per element. */
void philox_normals_s(uint64_t seed, uint64_t stream, uint64_t e0, int64_t cnt, float* out) {
    uint64_t e = e0;
    int64_t i = 0;
    while (i < cnt) {
        const uint64_t g = e >> 2;
        uint32_t c[4] = { (uint32_t)g, (uint32_t)(g >> 32), (uint32_t)stream,
                          (uint32_t)(stream >> 32) };
        rad_philox4x32_10(c, (uint32_t)seed, (uint32_t)(seed >> 32));
        for (int pair = (int)((e & 3u) >> 1); pair < 2 && i < cnt; ++pair) {
            const uint32_t wa = c[pair * 2], wb = c[pair * 2 + 1];
            const float ua = ((float)(wa >> 8) + 0.5f) * (1.0f / 16777216.0f);
            const float ub = ((float)(wb >> 8) + 0.5f) * (1.0f / 16777216.0f);
            const float r = sqrtf(-2.0f * logf(ua));
            const float t = 6.28318530717958647692f * ub;
            if ((e & 1u) == 0) { out[i++] = r * cosf(t); ++e; }
            if (i < cnt) { out[i++] = r * sinf(t); ++e; }
        }
    }
}

/* Patch rows a GEMM band holds: at least 8 Mi elements and eight times the weight, so the GEMM's
 * per-call weight pack is amortised, at most 32 Mi (64 MiB of bf16 the buffer keeps). */
inline int64_t band_rows(int64_t KK, int64_t cout, int64_t rows) {
    const int64_t want = std::min<int64_t>((int64_t)1 << 25,
                                           std::max<int64_t>((int64_t)1 << 23, 8 * cout * KK));
    return std::max<int64_t>(1, std::min(rows, want / std::max<int64_t>(1, KK)));
}

/* A dense rank-2 view of `rows` x `cols` elements of dtype `dt` at `data`. */
inline RadTensor mat_view(void* data, uint32_t dt, int64_t rows, int64_t cols) {
    RadTensor t{};
    t.data = data;
    t.dtype = dt;
    t.rank = 2;
    t.shape[0] = rows; t.shape[1] = cols;
    t.stride[0] = cols; t.stride[1] = 1;
    return t;
}

/* The weight a GEMM reads, re-laid from PyTorch's flat order: element `src(co, j)` of w goes to
 * row co, column j of a [cout, KK] matrix, kept bf16 when w is bf16 (exact copies) and f32
 * otherwise. */
template <class Src>
void relay_weight(const RadTensor* w, int64_t cout, int64_t KK, std::vector<uint8_t>& buf,
                  RadTensor* out, Src src) {
    const bool b16 = w->dtype == RAD_BF16;
    const size_t el = b16 ? 2 : 4;
    buf.resize((size_t)(cout * KK) * el + 64);
    uint8_t* wp = buf.data();
    AVX_PARALLEL_FOR_IF(cout * KK >= AVX_PAR_MIN)
    for (int64_t co = 0; co < cout; ++co)
        for (int64_t j = 0; j < KK; ++j) {
            const int64_t s = src(co, j);
            if (b16) ((uint16_t*)wp)[co * KK + j] = ((const uint16_t*)w->data)[s];
            else     ((float*)wp)[co * KK + j] = ld_elem(w->data, w->dtype, s);
        }
    *out = mat_view(wp, b16 ? (uint32_t)RAD_BF16 : (uint32_t)RAD_F32, cout, KK);
}

/* `cin` channels of input row `ir` into a patch slot of dtype bf16 (x bf16 and dense, copied) or
 * f32; zeros when `valid` is false. */
inline void patch_slot(const RadTensor* x, int64_t ir, int64_t cin, bool valid, bool b16, char* d) {
    if (!valid) {
        std::memset(d, 0, (size_t)cin * (b16 ? 2 : 4));
        return;
    }
    const int64_t base = rowoff(x, ir, cin);
    if (b16) std::memcpy(d, (const uint16_t*)x->data + base, (size_t)cin * 2);
    else     load_row(x, base, cin, (float*)d);
}

/* sin(v)^2 by reduction to [-pi/2, pi/2] (the sign of the reduced sine does not matter once it is
 * squared) and an odd Taylor polynomial through r^13, whose truncation at pi/2 is 7e-10. The
 * three-part pi keeps k * P1 exact while |k| < 2^16; callers send |v| above kSinBig to the libm
 * sine instead. */
constexpr float kSinBig = 8192.0f;
inline vf sin_sq(vf v) {
    const vf k = vf_round(vf_mul(v, vf_set1(0.318309886183790671538f)));
    vf r = vf_sub(v, vf_mul(k, vf_set1(3.140625f)));
    r = vf_sub(r, vf_mul(k, vf_set1(9.67502593994140625e-4f)));
    r = vf_sub(r, vf_mul(k, vf_set1(1.509957990978376432e-7f)));
    const vf r2 = vf_mul(r, r);
    vf p = vf_set1(1.6059043836821614599e-10f);
    p = vf_fma(p, r2, vf_set1(-2.5052108385441718775e-8f));
    p = vf_fma(p, r2, vf_set1(2.7557319223985890653e-6f));
    p = vf_fma(p, r2, vf_set1(-1.9841269841269841270e-4f));
    p = vf_fma(p, r2, vf_set1(8.3333333333333333333e-3f));
    p = vf_fma(p, r2, vf_set1(-1.6666666666666666667e-1f));
    const vf s = vf_fma(vf_mul(r, r2), p, r);
    return vf_mul(s, s);
}

/* sin_sq for one element, the same steps. */
inline float sin_sq1(float v) {
    const float k = std::nearbyint(v * 0.318309886183790671538f);
    float r = v - k * 3.140625f;
    r = r - k * 9.67502593994140625e-4f;
    r = r - k * 1.509957990978376432e-7f;
    const float r2 = r * r;
    float p = 1.6059043836821614599e-10f;
    p = p * r2 + -2.5052108385441718775e-8f;
    p = p * r2 + 2.7557319223985890653e-6f;
    p = p * r2 + -1.9841269841269841270e-4f;
    p = p * r2 + 8.3333333333333333333e-3f;
    p = p * r2 + -1.6666666666666666667e-1f;
    const float sn = r * r2 * p + r;
    return sn * sn;
}

}  /* namespace */

/* ------------------------------------------------------------------ conv1d */
/* A 1-D convolution over sequences, channels innermost, as GEMMs over bands of patch rows. The
 * weight [cout, cin, k] is re-laid once a launch as [cout][k][cin], so an output row's patch is its
 * k input rows' channel runs back to back, zeros outside its own sequence (padded on both sides,
 * or with `causal` on the left only). A band's patches are one gemm_nt_into call into f32 rows (the
 * bias added there); `res` and the tanh or the clamp to [-1, 1] are applied to the f32 rows, which
 * are rounded once into y. A 1-wide kernel over a dense x reads x itself as the patches. With a
 * `stride` (causal only) output row o's patch starts at input row o stride - pad, the pad the
 * reach less the stride, and a sequence of L rows makes L / stride. `cu_div`: the bounds in units
 * that many input rows apart, as libref reads them. */
AVX_KERNEL(conv1d) {
    if (!rad_args_have(a, 3) || a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *W = &a->t[1], *cu = &a->t[2];
    const RadTensor *res = rad_arg_in(a, 3), *Y = &a->t[4], *B = rad_arg_in(a, 5);
    if (!Y->data) return RAD_E_INVAL;
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0), k = p_int(a, "k", 0);
    const int64_t dil = p_int(a, "dil", 1), causal = p_int(a, "causal", 0);
    const int64_t st = p_int(a, "stride", 1), ep = p_int(a, "pad", -1), cd = p_int(a, "cu_div", 1);
    const char* act = p_str(a, "act", "none");
    const bool tanh_out = act && !std::strcmp(act, "tanh");
    const bool clamp_out = act && !std::strcmp(act, "clamp");
    if (!tanh_out && !clamp_out && act && std::strcmp(act, "none")) return RAD_E_INVAL;
    /* A stride takes one of two forms: causal (bounds in cu_div units, each sequence a multiple of
     * the stride) or, with `pad`, PyTorch's zero-padded one (bf16, one sequence, cu_div 1). */
    const bool padded = ep >= 0 && st > 1;
    if (cin <= 0 || cout <= 0 || k <= 0 || (!causal && st <= 1 && (k & 1) == 0) || dil <= 0 ||
        causal < 0 || causal > 1 || st < 1 || (ep >= 0 && (st <= 1 || causal || cd != 1)) ||
        (st > 1 && ep < 0 && (!causal || dil * (k - 1) + 1 < st)) || cd < 1)
        return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    const int64_t cdv = padded ? 1 : cd;
    for (int64_t e = 0; !padded && e <= n_seq; ++e)
        if (cu_at(cu, e) % (cd * st)) return RAD_E_SHAPE;
    const int64_t x0 = cu_at(cu, 0) / cdv, x1 = cu_at(cu, n_seq) / cdv;
    if (x0 < 0 || x1 <= x0 || rows_of(X, cin) < x1) return RAD_E_SHAPE;
    if (numel(W) < cout * cin * k || (B && numel(B) < cout)) return RAD_E_SHAPE;
    /* r0, r1: the output rows. A causal stride takes each sequence of L rows to L / stride; the
     * padded form has PyTorch's (L + 2 pad - dil (k - 1) - 1) / stride + 1, y exactly that many. */
    int64_t r0 = x0 / st, r1 = x1 / st;
    if (padded) {
        if (X->dtype != RAD_BF16) return RAD_E_DTYPE;
        if (n_seq != 1 || x0 != 0 || x1 != rows_of(X, cin)) return RAD_E_SHAPE;
        r0 = 0;
        r1 = (x1 + 2 * ep - dil * (k - 1) - 1) / st + 1;
    }
    if (r0 >= r1) return RAD_E_SHAPE;
    if (padded ? rows_of(Y, cout) != r1 : rows_of(Y, cout) < r1) return RAD_E_SHAPE;
    if (res && rows_of(res, cout) < r1) return RAD_E_SHAPE;
    const int64_t pad = padded ? ep : (causal ? dil * (k - 1) + 1 - st : dil * (k - 1) / 2),
                  KK = k * cin;
    static thread_local std::vector<uint8_t> wbuf, pbuf;
    static thread_local std::vector<float> ybuf;

    RadTensor wt;
    if (k == 1 && (W->dtype == RAD_BF16 || W->dtype == RAD_F32))
        wt = mat_view(W->data, W->dtype, cout, KK);
    else
        relay_weight(W, cout, KK, wbuf, &wt, [cin, k](int64_t co, int64_t j) {
            return (co * cin + j % cin) * k + j / cin;
        });

    const bool b16 = X->dtype == RAD_BF16 && laststride(X) == 1;
    const uint32_t adt = b16 ? (uint32_t)RAD_BF16 : (uint32_t)RAD_F32;
    const size_t ael = b16 ? 2 : 4;
    /* A 1-wide kernel's patch row is x's own row; a strided op reads rows `stride` apart, so it
     * always builds patches. */
    const bool direct = k == 1 && st == 1 && X->rank == 2 && X->shape[1] == cin &&
                        X->stride[1] == 1 && (X->dtype == RAD_BF16 || X->dtype == RAD_F32);
    const int64_t band = band_rows(KK, cout, r1 - r0);
    if (!direct) pbuf.resize((size_t)(band * KK) * ael + 64);
    ybuf.resize((size_t)(band * cout) + 16);
    char* pp = (char*)pbuf.data();
    float* yp = ybuf.data();
    for (int64_t b0 = r0; b0 < r1; b0 += band) {
        const int64_t nb = std::min(band, r1 - b0);
        RadTensor pt;
        if (direct) {
            pt = mat_view((char*)X->data + (size_t)(b0 * X->stride[0]) * rad_dtype_bits(X->dtype) / 8,
                          X->dtype, nb, KK);
            pt.stride[0] = X->stride[0];
        } else {
            pt = mat_view(pp, adt, nb, KK);
            AVX_PARALLEL_FOR_IF(nb * KK >= AVX_PAR_MIN)
            for (int64_t p = 0; p < nb; ++p) {
                const int64_t r = b0 + p, e = seq_of(cu, n_seq, r * st * cdv);
                const int64_t s0 = cu_at(cu, e) / cdv, s1 = cu_at(cu, e + 1) / cdv;
                for (int64_t t = 0; t < k; ++t) {
                    const int64_t ir = r * st + t * dil - pad;
                    patch_slot(X, ir, cin, ir >= s0 && ir < s1, b16,
                               pp + (size_t)(p * KK + t * cin) * ael);
                }
            }
        }
        const RadTensor yt = mat_view(yp, RAD_F32, nb, cout);
        const int st = gemm_nt_into(&pt, &wt, B, &yt, nb, cout, KK);
        if (st < 0) return st;
        AVX_PARALLEL_FOR_IF(nb * cout >= AVX_PAR_MIN)
        for (int64_t p = 0; p < nb; ++p) {
            float* row = yp + p * cout;
            if (res) {
                const float* rr = in_row(res, rowoff(res, b0 + p, cout), cout,
                                         scratch_f32((size_t)cout + 16));
                for (int64_t c = 0; c < cout; ++c) row[c] += rr[c];
            }
            if (tanh_out)
                for (int64_t c = 0; c < cout; ++c) row[c] = sc_tanh(row[c]);
            /* A NaN stays NaN, as libref's comparisons leave it. */
            if (clamp_out)
                for (int64_t c = 0; c < cout; ++c)
                    row[c] = row[c] < -1.0f ? -1.0f : (row[c] > 1.0f ? 1.0f : row[c]);
            out_row(Y, rowoff(Y, b0 + p, cout), cout, row);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv_t1d */
/* A transposed 1-D convolution of kernel k >= stride as `stride` GEMMs, one a phase. Output row lo
 * of a sequence is q s + r - pad for phase r (pad (k - stride) / 2 by default, 0 the causal form)
 * and reads input rows q - t against taps r + t stride for each t with r + t stride < k, so a
 * phase's rows are one GEMM over patch rows [x[q] | x[q - 1] | ..] (zeros past the sequence) and
 * the weight [cin, cout, k] re-laid per phase as [cout][taps][cin], the phases back to back. Each
 * phase is computed into f32 rows (the bias added there) and rounded once into y's rows, `stride`
 * apart. */
AVX_KERNEL(conv_t1d) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *W = &a->t[1], *B = &a->t[2], *cu = &a->t[3], *Y = &a->t[4];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0), s = p_int(a, "stride", 0);
    /* Without `pad`, the symmetric (k - stride) / 2, which needs k - stride even. */
    const int64_t K = p_int(a, "k", 2 * s), pad = p_int(a, "pad", -1);
    const int64_t P = pad >= 0 ? pad : (K - s) / 2;
    if (cin <= 0 || cout <= 0 || s <= 0 || K < s || (pad < 0 && ((K - s) & 1))) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    const int64_t r0 = cu_at(cu, 0), r1 = cu_at(cu, n_seq);
    if (r0 < 0 || r1 <= r0 || rows_of(X, cin) < r1 || rows_of(Y, cout) < r1 * s) return RAD_E_SHAPE;
    if (numel(W) < cin * cout * K || numel(B) < cout) return RAD_E_SHAPE;
    /* Phase r's taps, and the most any phase has (phase 0's). */
    auto taps_of = [K, s](int64_t r) { return (K - r + s - 1) / s; };
    const int64_t KKmax = taps_of(0) * cin;
    static thread_local std::vector<uint8_t> wbuf, pbuf;
    static thread_local std::vector<float> ybuf;
    static thread_local std::vector<int64_t> woff;

    /* Phase r's [cout, taps cin] block starts at element woff[r]; its row co, column t cin + ci
     * is w[ci, co, r + t s]. Kept bf16 when w is bf16 (exact copies), f32 otherwise. */
    const bool w16 = W->dtype == RAD_BF16;
    const size_t wel = w16 ? 2 : 4;
    woff.assign((size_t)s + 1, 0);
    for (int64_t r = 0; r < s; ++r) woff[(size_t)r + 1] = woff[(size_t)r] + cout * taps_of(r) * cin;
    wbuf.resize((size_t)woff[(size_t)s] * wel + 64);
    uint8_t* wp = wbuf.data();
    const int64_t* wo = woff.data();
    AVX_PARALLEL_FOR_IF(wo[s] >= AVX_PAR_MIN)
    for (int64_t rc = 0; rc < s * cout; ++rc) {
        const int64_t r = rc / cout, co = rc % cout, KKr = taps_of(r) * cin;
        for (int64_t j = 0; j < KKr; ++j) {
            const int64_t src = ((j % cin) * cout + co) * K + r + (j / cin) * s;
            const int64_t dst = wo[r] + co * KKr + j;
            if (w16) ((uint16_t*)wp)[dst] = ((const uint16_t*)W->data)[src];
            else     ((float*)wp)[dst] = ld_elem(W->data, W->dtype, src);
        }
    }

    const bool b16 = X->dtype == RAD_BF16 && laststride(X) == 1;
    const uint32_t adt = b16 ? (uint32_t)RAD_BF16 : (uint32_t)RAD_F32;
    const size_t ael = b16 ? 2 : 4;
    int64_t longest = 0;
    for (int64_t e = 0; e < n_seq; ++e)
        longest = std::max(longest, cu_at(cu, e + 1) - cu_at(cu, e));
    const int64_t band = band_rows(KKmax, cout, std::max<int64_t>(1, longest));
    pbuf.resize((size_t)(band * KKmax) * ael + 64);
    ybuf.resize((size_t)(band * cout) + 16);
    char* pp = (char*)pbuf.data();
    float* yp = ybuf.data();
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t i0 = cu_at(cu, e), i1 = cu_at(cu, e + 1), Lx = i1 - i0;
        if (Lx <= 0) continue;
        for (int64_t r = 0; r < s; ++r) {
            /* Phase r's rows are the Lx rows lo = q s + r - pad in [0, Lx s), q from
             * ceil((pad - r) / s) when pad > r, else from 0. */
            const int64_t q0 = P > r ? (P - r + s - 1) / s : 0;
            const int64_t taps = taps_of(r), KK = taps * cin;
            const RadTensor wt = mat_view(wp + (size_t)wo[r] * wel,
                                          w16 ? (uint32_t)RAD_BF16 : (uint32_t)RAD_F32, cout, KK);
            for (int64_t j0 = 0; j0 < Lx; j0 += band) {
                const int64_t nb = std::min(band, Lx - j0);
                AVX_PARALLEL_FOR_IF(nb * KK >= AVX_PAR_MIN)
                for (int64_t p = 0; p < nb; ++p) {
                    const int64_t q = q0 + j0 + p;
                    char* d = pp + (size_t)(p * KK) * ael;
                    for (int64_t t = 0; t < taps; ++t)
                        patch_slot(X, i0 + q - t, cin, q - t >= 0 && q - t < Lx, b16,
                                   d + (size_t)(t * cin) * ael);
                }
                const RadTensor pt = mat_view(pp, adt, nb, KK);
                const RadTensor yt = mat_view(yp, RAD_F32, nb, cout);
                const int st = gemm_nt_into(&pt, &wt, B, &yt, nb, cout, KK);
                if (st < 0) return st;
                AVX_PARALLEL_FOR_IF(nb * cout >= AVX_PAR_MIN)
                for (int64_t p = 0; p < nb; ++p) {
                    const int64_t o = i0 * s + (q0 + j0 + p) * s + r - P;
                    out_row(Y, rowoff(Y, o, cout), cout, yp + p * cout);
                }
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ snake */
/* y = x + sin(alpha x)^2 / (alpha + 1e-9), or SnakeBeta's x + sin(alpha x)^2 / (beta + 1e-9) with
 * `beta` (its reciprocal taken once a channel, as libref takes it), alpha and beta exponentiated
 * first with `log` 1; the sine squared by sin_sq. A row holding an argument past kSinBig (or a
 * non-finite one) is computed again with the libm sine, as libref computes it. */
AVX_KERNEL(snake) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *AL = &a->t[1], *Y = &a->t[2], *BE = rad_arg_in(a, 3);
    const int64_t n = p_int(a, "n", 0), lg = p_int(a, "log", 0);
    if (n <= 0 || numel(AL) < n || (BE && numel(BE) < n) || lg < 0 || lg > 1) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;
    /* alpha and the divisor (alpha + 1e-9) for every channel, once a launch, before the parallel
     * region; with beta, the multiplier 1 / (beta + 1e-9) in its place. */
    float* al = (float*)scratch_raw((size_t)(2 * n + 32) * sizeof(float));
    float* den = al + n + 16;
    for (int64_t c = 0; c < n; ++c) {
        al[c] = lg ? (float)std::exp((double)ldt(AL, c)) : ldt(AL, c);
        if (BE) {
            const float b = lg ? (float)std::exp((double)ldt(BE, c)) : ldt(BE, c);
            den[c] = 1.0f / (b + 1e-9f);
        } else {
            den[c] = al[c] + 1e-9f;
        }
    }
    const bool mul = BE != nullptr;
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const float* x = in_row(X, rowoff(X, r, n), n, scratch_f32((size_t)n + 16));
        float* y = scratch_f32_b((size_t)n + 16);
        vf big = vf_zero();
        int64_t c = 0;
        for (; c + VF_N <= n; c += VF_N) {
            const vf xv = vf_loadu(x + c), v = vf_mul(vf_loadu(al + c), xv);
            /* |v| first: a NaN lane keeps the running maximum (vmaxps returns its second
             * operand on NaN). */
            big = vf_max(vf_abs(v), big);
            const vf s2 = sin_sq(v), dv = vf_loadu(den + c);
            vf_storeu(y + c, vf_add(xv, mul ? vf_mul(dv, s2) : vf_div(s2, dv)));
        }
        bool slow = !(vf_hmax(big) <= kSinBig);
        for (; c < n; ++c) {
            const float v = al[c] * x[c];
            slow = slow || !(std::fabs(v) <= kSinBig);
            y[c] = x[c] + (mul ? den[c] * sin_sq1(v) : sin_sq1(v) / den[c]);
        }
        if (slow)
            for (c = 0; c < n; ++c) {
                const float sn = sc_sin(al[c] * x[c]);
                y[c] = x[c] + (mul ? den[c] * (sn * sn) : sn * sn / den[c]);
            }
        out_row(Y, rowoff(Y, r, n), n, y);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ snake_aa */
/* BigVGAN's alias-free SnakeBeta over each sequence (libref's snake_aa): the replicate-padded
 * sequence upsampled `ratio` times by the transposed f_up filter, SnakeBeta, and the
 * replicate-padded result filtered by f_down at stride `ratio`, in f32. Channels are innermost, so
 * every step is vector work along a row. A task is a band of output rows: it computes the
 * upsampled rows the band reads (each a function of its index alone, so two bands' overlaps
 * agree) into scratch, then filters them down. Each sum runs in libref's order; the sine is
 * sin_sq, a row holding an argument past kSinBig taken again with the libm sine, as `snake`
 * does. */
AVX_KERNEL(snake_aa) {
    if (!rad_args_have(a, 7)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *AL = &a->t[1], *BE = &a->t[2], *FU = &a->t[3];
    const RadTensor *FD = &a->t[4], *cu = &a->t[5], *Y = &a->t[6];
    const int64_t n = p_int(a, "n", 0), R = p_int(a, "ratio", 0), k = p_int(a, "k", 0);
    const int64_t lg = p_int(a, "log", 0);
    if (n <= 0 || R <= 0 || k < R || lg < 0 || lg > 1) return RAD_E_SHAPE;
    if (numel(AL) < n || numel(BE) < n || numel(FU) < k || numel(FD) < k) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    const int64_t r0 = cu_at(cu, 0), r1 = cu_at(cu, n_seq);
    if (r0 < 0 || r1 <= r0 || rows_of(X, n) < r1 || rows_of(Y, n) < r1) return RAD_E_SHAPE;
    const int64_t P = k / R - 1, crop = P * R + (k - R) / 2, pl = k / 2 - (k % 2 == 0 ? 1 : 0);
    /* alpha, the multiplier 1 / (beta + 1e-9) and both filters, once a launch. */
    float* tab = (float*)scratch_raw((size_t)(2 * n + 2 * k + 64) * sizeof(float));
    float *al = tab, *mb = tab + n, *fup = tab + 2 * n, *fdn = fup + k;
    for (int64_t c = 0; c < n; ++c) {
        al[c] = lg ? sc_exp(ldt(AL, c)) : ldt(AL, c);
        mb[c] = 1.0f / ((lg ? sc_exp(ldt(BE, c)) : ldt(BE, c)) + 1e-9f);
    }
    for (int64_t j = 0; j < k; ++j) {
        fup[j] = ldt(FU, offlin(FU, j));
        fdn[j] = ldt(FD, offlin(FD, j));
    }
    auto clampi = [](int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : (v > hi ? hi : v); };
    constexpr int64_t kBand = 64;     /* output rows a task */
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t s0 = cu_at(cu, e), L = cu_at(cu, e + 1) - s0;
        if (L <= 0) continue;
        const int64_t LR = L * R, bands = (L + kBand - 1) / kBand;
        /* The padded input rows an upsampled row m sums over, ascending. */
        auto i_lo = [=](int64_t m) { return crop + m < k ? 0 : (crop + m - k) / R + 1; };
        auto i_hi = [=](int64_t m) { return std::min((crop + m) / R, L + 2 * P - 1); };
        AVX_PARALLEL_FOR_IF(L * n * k >= AVX_PAR_MIN)
        for (int64_t b = 0; b < bands; ++b) {
            const int64_t o0 = b * kBand, o1 = std::min(L, o0 + kBand);
            const int64_t m_lo = clampi(o0 * R - pl, 0, LR - 1);
            const int64_t m_hi = clampi((o1 - 1) * R + k - 1 - pl, 0, LR - 1);
            const int64_t ia = i_lo(m_lo), ib = i_hi(m_hi);
            float* xb = scratch_f32((size_t)((ib - ia + 1) * n + 16));
            float* ab = scratch_f32_b((size_t)((m_hi - m_lo + 1) * n + 16));
            float* yr = scratch_f32_c((size_t)n + 16);
            for (int64_t i = ia; i <= ib; ++i)
                load_row(X, rowoff(X, s0 + clampi(i - P, 0, L - 1), n), n, xb + (i - ia) * n);
            for (int64_t m = m_lo; m <= m_hi; ++m) {
                const int64_t o = crop + m, il = i_lo(m), ih = i_hi(m);
                float* am = ab + (m - m_lo) * n;
                const vf vR = vf_set1((float)R);
                vf big = vf_zero();
                int64_t c = 0;
                for (; c + VF_N <= n; c += VF_N) {
                    vf acc = vf_zero();
                    for (int64_t i = il; i <= ih; ++i)
                        acc = vf_fma(vf_set1(fup[o - i * R]), vf_loadu(xb + (i - ia) * n + c), acc);
                    const vf u = vf_mul(vR, acc), v = vf_mul(u, vf_loadu(al + c));
                    /* |v| first: a NaN lane keeps the running maximum. */
                    big = vf_max(vf_abs(v), big);
                    vf_storeu(am + c, vf_add(u, vf_mul(vf_loadu(mb + c), sin_sq(v))));
                }
                bool slow = !(vf_hmax(big) <= kSinBig);
                for (; c < n; ++c) {
                    float acc = 0.0f;
                    for (int64_t i = il; i <= ih; ++i) acc += fup[o - i * R] * xb[(i - ia) * n + c];
                    const float u = (float)R * acc, v = u * al[c];
                    slow = slow || !(std::fabs(v) <= kSinBig);
                    am[c] = u + mb[c] * sin_sq1(v);
                }
                if (slow)
                    for (c = 0; c < n; ++c) {
                        float acc = 0.0f;
                        for (int64_t i = il; i <= ih; ++i)
                            acc += fup[o - i * R] * xb[(i - ia) * n + c];
                        const float u = (float)R * acc, sn = sc_sin(u * al[c]);
                        am[c] = u + mb[c] * (sn * sn);
                    }
            }
            for (int64_t o = o0; o < o1; ++o) {
                int64_t c = 0;
                for (; c + VF_N <= n; c += VF_N) {
                    vf acc = vf_zero();
                    for (int64_t j = 0; j < k; ++j) {
                        const int64_t m = clampi(o * R + j - pl, 0, LR - 1);
                        acc = vf_fma(vf_set1(fdn[j]), vf_loadu(ab + (m - m_lo) * n + c), acc);
                    }
                    vf_storeu(yr + c, acc);
                }
                for (; c < n; ++c) {
                    float acc = 0.0f;
                    for (int64_t j = 0; j < k; ++j)
                        acc += fdn[j] * ab[(clampi(o * R + j - pl, 0, LR - 1) - m_lo) * n + c];
                    yr[c] = acc;
                }
                out_row(Y, rowoff(Y, s0 + o, n), n, yr);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ cfg_topk_sample */
/* One pick a lane, libref's arithmetic step for step: g = u + cfg (c - u) in two roundings (the
 * lane's own guidance when `cfg` is bound), the repetition penalty over the lane's history (its
 * row cleared at frame 0, the pick recorded after), the suppressed ids and the candidates at or
 * above the conditional's cond_k-th largest left out, the k largest g with ties to the lower index,
 * the softmax over temp in double, and the inverse CDF against
 * rad_philox_uniform_s(seed, 2^32 + frame, stream). Every entry's history row and candidate set is
 * checked before anything is written. Lanes run in parallel without a history; with one they run
 * in order, as two entries may share a lane's row. */
AVX_KERNEL(cfg_topk_sample) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *seed = &a->t[1], *fr = &a->t[2], *ids = &a->t[3];
    const RadTensor *cfgs = rad_arg_in(a, 4), *lane = rad_arg_in(a, 5), *hist = rad_arg_in(a, 6);
    const int64_t V = p_int(a, "n_vocab", 0), K = p_int(a, "k", 0), rep = p_int(a, "rep", 1);
    const int64_t ck = p_int(a, "cond_k", 0), sub = p_int(a, "stream", 0);
    const int64_t sup_lo = p_int(a, "sup_lo", 0), sup_hi = p_int(a, "sup_hi", 0);
    const float cfg = p_f32(a, "cfg", 1.0f), temp = p_f32(a, "temp", 1.0f);
    const float pen = p_f32(a, "penalty", 1.0f);
    const double top_p = (double)p_f32(a, "top_p", 1.0f);
    if (V <= 0 || K <= 0 || K > V || rep < 1 || rep > 2 || ck < 0 || ck > V || !(temp > 0) ||
        sup_lo < 0 || sup_hi < sup_lo || sup_hi > V || !(pen > 0) || (!hist != !lane) ||
        !(top_p >= 0))
        return RAD_E_SHAPE;
    const int64_t n = numel(ids) / rep;
    if (n <= 0 || rows_of(lg, V) < rep * n || numel(fr) < n || numel(seed) < 2 * n ||
        (cfgs && numel(cfgs) < n) || (lane && numel(lane) < n))
        return RAD_E_SHAPE;
    auto open = [sup_lo, sup_hi](int64_t j) { return j < sup_lo || j >= sup_hi; };
    /* Before anything is written: each entry's history row inside `hist`, and a candidate left
     * for each lane once the suppressed ids are out. */
    for (int64_t i = 0; hist && i < n; ++i) {
        const int64_t L = at(lane, i);
        if (L < 0 || (L + 1) * V > numel(hist)) return RAD_E_SHAPE;
    }
    if (sup_hi - sup_lo >= V) return RAD_E_SHAPE;
    if (ck > 0 && sup_hi > sup_lo)
        for (int64_t i = 0; i < n; ++i) {
            float* c = scratch_f32((size_t)V + 16);
            float* cc = scratch_f32_b((size_t)V + 16);
            load_row(lg, rowoff(lg, rep * i, V), V, c);
            std::memcpy(cc, c, (size_t)V * sizeof(float));
            std::nth_element(cc, cc + (ck - 1), cc + V, std::greater<float>());
            const float thr = cc[ck - 1];
            bool any = false;
            for (int64_t j = 0; j < V && !any; ++j) any = c[j] >= thr && open(j);
            if (!any) return RAD_E_SHAPE;
        }
    auto one = [&](int64_t i) {
        float* c = scratch_f32((size_t)V + 16);
        float* g = scratch_f32_b((size_t)V + 16);
        char* raw = (char*)scratch_raw((size_t)V * (2 * sizeof(int64_t) + sizeof(float)) + 256);
        int64_t* idx = (int64_t*)raw;
        double* e = (double*)(raw + (size_t)V * sizeof(int64_t));
        float* cc = (float*)(raw + (size_t)V * 2 * sizeof(int64_t));
        load_row(lg, rowoff(lg, rep * i, V), V, c);
        if (rep == 2) {
            const float gi = cfgs ? ldt(cfgs, offlin(cfgs, i)) : cfg;
            load_row(lg, rowoff(lg, rep * i + 1, V), V, g);
            for (int64_t j = 0; j < V; ++j) g[j] = g[j] + keep(gi * (c[j] - g[j]));
        } else {
            std::memcpy(g, c, (size_t)V * sizeof(float));
        }
        int64_t hb = -1;
        if (hist) {
            hb = at(lane, i) * V;
            if (at(fr, i) == 0)
                for (int64_t j = 0; j < V; ++j)
                    st_int_dt(hist->dtype, hist->data, offlin(hist, hb + j), 0);
            for (int64_t j = 0; j < V; ++j)
                if (at(hist, hb + j) != 0) g[j] = g[j] > 0.0f ? g[j] / pen : g[j] * pen;
        }
        int64_t nc = 0;
        if (ck > 0) {
            std::memcpy(cc, c, (size_t)V * sizeof(float));
            std::nth_element(cc, cc + (ck - 1), cc + V, std::greater<float>());
            const float thr = cc[ck - 1];
            for (int64_t j = 0; j < V; ++j) if (c[j] >= thr && open(j)) idx[nc++] = j;
        } else {
            for (int64_t j = 0; j < V; ++j) if (open(j)) idx[nc++] = j;
        }
        /* top_p's normaliser over every candidate in index order (idx still is), as libref. */
        double z = 0;
        if (top_p < 1.0) {
            float gm = g[idx[0]];
            for (int64_t t = 1; t < nc; ++t) gm = std::max(gm, g[idx[t]]);
            const float m0 = gm / temp;
            for (int64_t t = 0; t < nc; ++t) z += std::exp((double)(g[idx[t]] / temp - m0));
        }
        int64_t kk = std::min(K, nc);
        std::partial_sort(idx, idx + kk, idx + nc, [g](int64_t p, int64_t q) {
            return g[p] > g[q] || (g[p] == g[q] && p < q);
        });
        const float m = g[idx[0]] / temp;
        if (top_p < 1.0) {
            double cum = 0;
            int64_t keep = 1;
            for (int64_t t = 0; t < kk; ++t) {
                cum += std::exp((double)(g[idx[t]] / temp - m)) / z;
                if (t > 0 && cum > top_p) break;
                keep = t + 1;
            }
            kk = keep;
        }
        double tot = 0;
        for (int64_t t = 0; t < kk; ++t) {
            e[t] = std::exp((double)(g[idx[t]] / temp - m));
            tot += e[t];
        }
        const double u = (double)rad_philox_uniform_s(seed_of(seed, i),
                                                      ((uint64_t)1 << 32) + (uint64_t)at(fr, i),
                                                      (uint64_t)sub) * tot;
        int64_t pick = idx[kk - 1];
        double acc = 0;
        for (int64_t t = 0; t < kk; ++t) {
            acc += e[t];
            if (u < acc) { pick = idx[t]; break; }
        }
        for (int64_t r = 0; r < rep; ++r)
            st_int_dt(ids->dtype, ids->data, offlin(ids, rep * i + r), pick);
        if (hist) st_int_dt(hist->dtype, hist->data, offlin(hist, hb + pick), 1);
    };
    if (hist) {
        for (int64_t i = 0; i < n; ++i) one(i);
    } else {
        AVX_PARALLEL_FOR_IF(n * V >= AVX_PAR_MIN)
        for (int64_t i = 0; i < n; ++i) one(i);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rvq_embed */
/* y = bf16(bf16(t0[c0 + off0] + bf16(sum_k t1[c_k + (k - 1) vocab])) * scale), the sum over the
 * codebooks in libref's order, a whole row at a time; with `round` 1 one f32 sum from t0's row on,
 * times scale, rounded once into y. A row whose first code is out of range is not written, as in
 * libref. */
AVX_KERNEL(rvq_embed) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *codes = &a->t[0], *t0 = &a->t[1], *t1 = &a->t[2], *Y = &a->t[3];
    const int64_t D = p_int(a, "n_embd", 0), K = p_int(a, "n_codes", 0);
    const int64_t voc = p_int(a, "vocab", 0), off0 = p_int(a, "off0", 0);
    const float scale = p_f32(a, "scale", 1.0f);
    const int64_t round1 = p_int(a, "round", 0), clamp = p_int(a, "clamp", 0);
    if (D <= 0 || K < 1 || voc <= 0 || round1 < 0 || round1 > 1 || clamp < 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, D);
    if (rows <= 0 || numel(codes) < K * rows) return RAD_E_SHAPE;
    const int64_t n0 = rows_of(t0, D), n1 = rows_of(t1, D);
    auto cl = [clamp](int64_t c) { return clamp > 0 && c > clamp - 1 ? clamp - 1 : c; };
    /* Code (k, r): a codebook's row at the plane's own pitch, a frame's code at its own stride. */
    const int64_t kp = codes->rank >= 2 ? codes->stride[codes->rank - 2] : rows;
    const int64_t rp = codes->rank >= 1 ? codes->stride[codes->rank - 1] : 1;
    AVX_PARALLEL_FOR_IF(rows * D * K >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t c0 = cl(ld_int_dt(codes->dtype, codes->data, r * rp)) + off0;
        if (c0 < 0 || c0 >= n0) continue;
        float* acc = scratch_f32((size_t)D + 16);
        float* buf = scratch_f32_b((size_t)D + 16);
        float* tmp = scratch_f32_c((size_t)D + 16);
        if (round1) {
            const float* e0 = in_row(t0, rowoff(t0, c0, D), D, buf);
            for (int64_t d = 0; d < D; ++d) acc[d] = e0[d];
        } else {
            for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
        }
        for (int64_t k = 1; k < K; ++k) {
            const int64_t cb =
                cl(ld_int_dt(codes->dtype, codes->data, k * kp + r * rp)) + (k - 1) * voc;
            if (cb < 0 || cb >= n1) continue;
            const float* tr = in_row(t1, rowoff(t1, cb, D), D, buf);
            for (int64_t d = 0; d < D; ++d) acc[d] += tr[d];
        }
        if (round1) {
            for (int64_t d = 0; d < D; ++d) acc[d] *= scale;
            out_row(Y, rowoff(Y, r, D), D, acc);
            continue;
        }
        narrow_row(RAD_BF16, acc, D, tmp);
        const float* e0 = in_row(t0, rowoff(t0, c0, D), D, buf);
        for (int64_t d = 0; d < D; ++d) acc[d] = e0[d] + acc[d];
        narrow_row(RAD_BF16, acc, D, tmp);
        for (int64_t d = 0; d < D; ++d) acc[d] *= scale;
        out_row(Y, rowoff(Y, r, D), D, acc);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vq_encode */
/* libref's step a row at a time: e as dim dot products over x's row, the codebook normalised once
 * a launch (F.normalize's division) and scored against e normalised, ties to the lower index, the
 * chosen raw row's out-projection subtracted and the row rounded once into x. */
AVX_KERNEL(vq_encode) {
    if (!rad_args_have(a, 7)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *iw = &a->t[1], *ib = &a->t[2], *cb = &a->t[3];
    const RadTensor *ow = &a->t[4], *ob = &a->t[5], *codes = &a->t[6];
    const int64_t n = p_int(a, "n", 0), dim = p_int(a, "dim", 0), bins = p_int(a, "bins", 0);
    if (n <= 0 || dim <= 0 || bins <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(X, n);
    if (rows <= 0 || numel(codes) < rows || numel(iw) < dim * n || numel(ib) < dim ||
        numel(cb) < bins * dim || numel(ow) < n * dim || numel(ob) < n)
        return RAD_E_SHAPE;
    /* The weights dense in f32, held by this thread for the launch's parallel region. */
    auto dense = [](const RadTensor* t, int64_t m) {
        std::vector<float> v((size_t)m);
        for (int64_t i = 0; i < m; ++i) v[(size_t)i] = ldt(t, i);
        return v;
    };
    const std::vector<float> wi = dense(iw, dim * n), bi = dense(ib, dim);
    const std::vector<float> raw = dense(cb, bins * dim), wo = dense(ow, n * dim), bo = dense(ob, n);
    std::vector<float> cn((size_t)(bins * dim));
    for (int64_t c = 0; c < bins; ++c) {
        const float* cr = raw.data() + c * dim;
        float ss = 0.0f;
        for (int64_t j = 0; j < dim; ++j) ss += cr[j] * cr[j];
        const float nrm = std::max(std::sqrt(ss), 1e-12f);
        for (int64_t j = 0; j < dim; ++j) cn[(size_t)(c * dim + j)] = cr[j] / nrm;
    }
    const float *Wi = wi.data(), *Bi = bi.data(), *Raw = raw.data(), *Cn = cn.data();
    const float *Wo = wo.data(), *Bo = bo.data();
    AVX_PARALLEL_FOR_IF(rows * (2 * n + bins) * dim >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, n);
        const float* xv = in_row(X, xb, n, scratch_f32((size_t)n + 16));
        float* e = scratch_f32_c((size_t)dim + 16);
        float ss = 0.0f;
        for (int64_t j = 0; j < dim; ++j) {
            e[j] = dotf(Wi + j * n, xv, n) + Bi[j];
            ss += e[j] * e[j];
        }
        const float nrm = std::max(std::sqrt(ss), 1e-12f);
        for (int64_t j = 0; j < dim; ++j) e[j] /= nrm;
        int64_t best = 0;
        float bs = -INFINITY;
        for (int64_t c = 0; c < bins; ++c) {
            float s = 0.0f;
            for (int64_t j = 0; j < dim; ++j) s += e[j] * Cn[c * dim + j];
            if (s > bs) { bs = s; best = c; }
        }
        st_int_dt(codes->dtype, codes->data, offlin(codes, r), best);
        float* y = scratch_f32_b((size_t)n + 16);
        const float* row = Raw + best * dim;
        for (int64_t d = 0; d < n; ++d) {
            float q = 0.0f;
            for (int64_t j = 0; j < dim; ++j) q += Wo[d * dim + j] * row[j];
            y[d] = xv[d] - (q + Bo[d]);
        }
        out_row(X, xb, n, y);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vq_euclidean */
/* libref's step a row at a time: the squared distance to every centroid summed in channel order,
 * the nearest taken (the lower index on a tie), its code written and the centroid subtracted from
 * the row, rounded once into x. The codebook is read dense in f32 once a launch. */
AVX_KERNEL(vq_euclidean) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *cb = &a->t[1], *codes = &a->t[2];
    const int64_t n = p_int(a, "n", 0), bins = p_int(a, "bins", 0);
    if (n <= 0 || bins <= 0 || numel(cb) != bins * n || codes->dtype != RAD_I32)
        return RAD_E_SHAPE;
    const int64_t rows = rows_of(X, n);
    if (rows < 1 || numel(codes) < rows) return RAD_E_SHAPE;
    std::vector<float> dense((size_t)(bins * n));
    for (int64_t i = 0; i < bins * n; ++i) dense[(size_t)i] = ldt(cb, offlin(cb, i));
    const float* C = dense.data();
    AVX_PARALLEL_FOR_IF(rows * bins * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(X, r, n);
        const float* xv = in_row(X, xb, n, scratch_f32((size_t)n + 16));
        float best = INFINITY;
        int64_t pick = 0;
        for (int64_t k = 0; k < bins; ++k) {
            const float* c = C + k * n;
            float d = 0.0f;
            for (int64_t j = 0; j < n; ++j) {
                const float delta = xv[j] - c[j];
                d += delta * delta;
            }
            if (d < best) { best = d; pick = k; }
        }
        st_int_dt(codes->dtype, codes->data, offlin(codes, r), pick);
        float* y = scratch_f32_b((size_t)n + 16);
        const float* c = C + pick * n;
        for (int64_t j = 0; j < n; ++j) y[j] = xv[j] - c[j];
        out_row(X, xb, n, y);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ elu */
/* y = x >= 0 ? x : expm1(x) (alpha 1), a row at a time, as libref. */
AVX_KERNEL(elu) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(X) < rows * n) return RAD_E_SHAPE;
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sy = scratch_f32_b((size_t)n);
        const float* v = in_row(X, rowoff(X, r, n), n, scratch_f32((size_t)n));
        for (int64_t i = 0; i < n; ++i) sy[i] = v[i] >= 0.0f ? v[i] : std::expm1(v[i]);
        out_row(Y, rowoff(Y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ frame_mix_store */
/* m = bf16(bf16(sum_l w_l h_l) * bf16(scale)), w = bf16(softmax(bf16(logits))) in libref's double
 * arithmetic, the planes summed in order with each product rounded before its add (`keep`). Entries
 * are stored one after another, a row's channels a vector at a time. */
AVX_KERNEL(frame_mix_store) {
    if (!rad_args_have(a, 7)) return RAD_E_INVAL;
    const RadTensor *hid = &a->t[0], *lw = &a->t[1], *sc = &a->t[2], *lane = &a->t[3];
    const RadTensor *fr = &a->t[4], *st = &a->t[5], *dst = &a->t[6];
    const int64_t D = p_int(a, "n_embd", 0), L = p_int(a, "L", 0), stride = p_int(a, "stride", 0);
    const int64_t rep = p_int(a, "rep", 1);
    if (D <= 0 || L <= 0 || stride <= 0 || rep < 1 || numel(lw) < L || numel(sc) < 1)
        return RAD_E_SHAPE;
    const int64_t n = numel(lane), plane = rows_of(hid, D) / L;
    if (n <= 0 || plane < rep * n || numel(fr) < n) return RAD_E_SHAPE;
    const int64_t drows = rows_of(dst, D), nst = numel(st);
    auto stores = [&](int64_t i) {
        const int64_t ln = at(lane, i), f = at(fr, i);
        return !(f < 1 || ln < 0 || ln >= nst || at(st, ln) < 0);
    };
    for (int64_t i = 0; i < n; ++i) {
        if (!stores(i)) continue;
        const int64_t f = at(fr, i);
        if (f > stride || at(lane, i) * stride + f - 1 >= drows) return RAD_E_SHAPE;
    }
    float* wl = (float*)scratch_raw((size_t)(L + 16) * sizeof(float));
    float mx = -INFINITY;
    for (int64_t l = 0; l < L; ++l) { wl[l] = rbf(ldt(lw, l)); mx = std::max(mx, wl[l]); }
    double tot = 0;
    for (int64_t l = 0; l < L; ++l) tot += std::exp((double)(wl[l] - mx));
    for (int64_t l = 0; l < L; ++l) wl[l] = rbf((float)(std::exp((double)(wl[l] - mx)) / tot));
    const float scale = rbf(ldt(sc, 0));
    float* acc = scratch_f32((size_t)D + 16);
    float* buf = scratch_f32_b((size_t)D + 16);
    float* tmp = scratch_f32_c((size_t)D + 16);
    for (int64_t i = 0; i < n; ++i) {
        if (!stores(i)) continue;
        const int64_t row = at(lane, i) * stride + at(fr, i) - 1;
        for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
        for (int64_t l = 0; l < L; ++l) {
            const float* h = in_row(hid, rowoff(hid, l * plane + rep * i, D), D, buf);
            const vf w = vf_set1(wl[l]);
            int64_t d = 0;
            for (; d + VF_N <= D; d += VF_N)
                vf_storeu(acc + d, vf_add(vf_loadu(acc + d), keep(vf_mul(w, vf_loadu(h + d)))));
            for (; d < D; ++d) acc[d] = acc[d] + keep(wl[l] * h[d]);
        }
        narrow_row(RAD_BF16, acc, D, tmp);
        for (int64_t d = 0; d < D; ++d) acc[d] *= scale;
        out_row(dst, rowoff(dst, row, D), D, acc);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ras_pick */
/* libref's rule: the first draw unless it lies in [lo, hi) and the lane's window of its last `win`
 * picks holds it, then the alternative; the pick enters the window. Lanes are checked before
 * anything is written and run in order, as two entries may share a lane's window. */
AVX_KERNEL(ras_pick) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *first = &a->t[0], *alt = &a->t[1], *lane = &a->t[2], *hist = &a->t[3];
    const RadTensor* ids = &a->t[4];
    const int64_t W = p_int(a, "win", 0), lo = p_int(a, "lo", 0), hi = p_int(a, "hi", 0);
    const int64_t rep = p_int(a, "rep", 1), clear = p_int(a, "clear", 0);
    const int64_t n = numel(lane);
    if (W <= 0 || lo < 0 || hi < lo || rep < 1 || clear < 0 || clear > 1 || n <= 0 ||
        numel(first) < rep * n || numel(alt) < rep * n || numel(ids) < rep * n)
        return RAD_E_SHAPE;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t L = at(lane, i);
        if (L < 0 || (L + 1) * W > numel(hist)) return RAD_E_SHAPE;
    }
    for (int64_t i = 0; i < n; ++i) {
        const int64_t hb = at(lane, i) * W;
        if (clear)
            for (int64_t w = 0; w < W; ++w) st_int_dt(hist->dtype, hist->data, offlin(hist, hb + w), -1);
        const int64_t f = at(first, rep * i);
        int64_t pick = f;
        if (f >= lo && f < hi)
            for (int64_t w = 0; w < W; ++w)
                if (at(hist, hb + w) == f) { pick = at(alt, rep * i); break; }
        for (int64_t r = 0; r < rep; ++r)
            st_int_dt(ids->dtype, ids->data, offlin(ids, rep * i + r), pick);
        for (int64_t w = 0; w + 1 < W; ++w)
            st_int_dt(hist->dtype, hist->data, offlin(hist, hb + w), at(hist, hb + w + 1));
        st_int_dt(hist->dtype, hist->data, offlin(hist, hb + W - 1), pick);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ end_status */
/* status[lane[i]] = ids[rep i] == end ? -1 : ids[rep i]. */
AVX_KERNEL(end_status) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *ids = &a->t[0], *lane = &a->t[1], *st = &a->t[2];
    const int64_t end = p_int(a, "end", -1), rep = p_int(a, "rep", 1);
    const int64_t n = numel(lane);
    if (n <= 0 || rep < 1 || numel(ids) < rep * n) return RAD_E_SHAPE;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t ln = at(lane, i);
        if (ln < 0 || ln >= numel(st)) return RAD_E_SHAPE;
    }
    for (int64_t i = 0; i < n; ++i) {
        const int64_t v = at(ids, rep * i);
        st_int_dt(st->dtype, st->data, offlin(st, at(lane, i)), v == end ? -1 : v);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ frame_store */
/* dst[(lane n_codes + k) stride + frame - skip] = codes[k, rep i] for each entry whose frame is at
 * least `skip` and whose lane has not ended, every entry checked first. codes is read at libref's
 * linear offsets (a codebook's row at the plane's pitch, a frame's code at its stride). */
AVX_KERNEL(frame_store) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *codes = &a->t[0], *lane = &a->t[1], *fr = &a->t[2], *st = &a->t[3];
    const RadTensor *dst = &a->t[4];
    const int64_t K = p_int(a, "n_codes", 0), stride = p_int(a, "stride", 0);
    const int64_t rep = p_int(a, "rep", 1), skip = p_int(a, "skip", 0);
    if (K <= 0 || stride <= 0 || rep < 1 || skip < 0) return RAD_E_SHAPE;
    const int64_t n = numel(lane);
    if (n <= 0 || numel(fr) < n || numel(codes) < K * rep * n) return RAD_E_SHAPE;
    const int64_t kp = codes->rank >= 2 ? codes->stride[codes->rank - 2] : rep * n;
    const int64_t rp = codes->rank >= 1 ? codes->stride[codes->rank - 1] : 1;
    const int64_t nst = numel(st), nd = numel(dst);
    auto stores = [&](int64_t i) {
        return at(fr, i) - skip >= 0 && at(st, at(lane, i)) >= 0;
    };
    for (int64_t i = 0; i < n; ++i) {
        const int64_t L = at(lane, i);
        if (L < 0 || L >= nst) return RAD_E_SHAPE;
        if (!stores(i)) continue;
        if (at(fr, i) - skip >= stride || (L + 1) * K * stride > nd) return RAD_E_SHAPE;
    }
    for (int64_t i = 0; i < n; ++i) {
        if (!stores(i)) continue;
        const int64_t L = at(lane, i), f = at(fr, i) - skip;
        for (int64_t k = 0; k < K; ++k)
            st_int_dt(dst->dtype, dst->data, offlin(dst, (L * K + k) * stride + f),
                      ld_int_dt(codes->dtype, codes->data, k * kp + rep * i * rp));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ resample_nearest */
/* Output row j is input row min(floor(j * (in / out)), in - 1), the ratio in f32 as libref forms
 * it. */
AVX_KERNEL(resample_nearest) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t in = rows_of(X, n), out = rows_of(Y, n);
    if (in <= 0 || out <= 0) return RAD_E_SHAPE;
    const float scale = (float)in / (float)out;
    AVX_PARALLEL_FOR_IF(out * n >= AVX_PAR_MIN)
    for (int64_t j = 0; j < out; ++j) {
        const int64_t src = std::min<int64_t>((int64_t)std::floor((float)j * scale), in - 1);
        copy_row(X, rowoff(X, src, n), Y, rowoff(Y, j, n), n, scratch_f32((size_t)n + 16));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ fourier_embed */
/* [cos(2 pi t w) | sin(2 pi t w)] with libref's libm calls and, with `round` 1, its bf16 roundings
 * of t, 2 pi t and the product. */
AVX_KERNEL(fourier_embed) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *T = &a->t[0], *W = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    const bool rnd = p_int(a, "round", 0) != 0;
    if (n <= 0 || (n & 1) || numel(W) < n / 2) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(T) < rows) return RAD_E_SHAPE;
    const int64_t h = n / 2;
    const float two_pi = 6.28318530717958647692f;
    float* w = (float*)scratch_raw((size_t)(h + 16) * sizeof(float));
    for (int64_t i = 0; i < h; ++i) w[i] = ldt(W, i);
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN / 16)
    for (int64_t r = 0; r < rows; ++r) {
        float* row = scratch_f32((size_t)n + 16);
        float tv = ldt(T, offlin(T, r));
        tv = rnd ? rbf(rbf(tv) * two_pi) : tv * two_pi;
        for (int64_t i = 0; i < h; ++i) {
            float ang = tv * w[i];
            if (rnd) ang = rbf(ang);
            row[i] = sc_cos(ang);
            row[h + i] = sc_sin(ang);
        }
        out_row(Y, rowoff(Y, r, n), n, row);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ win_input */
/* y[cu[e] + k] = [lat | zeros | cond] of row k of lane win_lane[e], zeros for the conditioning of an
 * unconditional entry (pair -2). */
AVX_KERNEL(win_input) {
    if (!rad_args_have(a, 6)) return RAD_E_INVAL;
    const RadTensor *lat = &a->t[0], *cond = &a->t[1], *lcu = &a->t[2], *wl = &a->t[3];
    const RadTensor *pair = &a->t[4], *Y = &a->t[5];
    const int64_t C = p_int(a, "n_lat", 0), Z = p_int(a, "n_zero", 0), D = p_int(a, "n_cond", 0);
    const int64_t stride = p_int(a, "stride", 0), W = C + Z + D;
    if (C <= 0 || Z < 0 || D <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t S = numel(lcu) - 1;
    if (S <= 0 || numel(wl) < S || numel(pair) < S) return RAD_E_SHAPE;
    const int64_t y_rows = rows_of(Y, W);
    if (y_rows < cu_at(lcu, S)) return RAD_E_SHAPE;
    for (int64_t e = 0; e < S; ++e) {
        const int64_t L = at(wl, e), neg = at(pair, e) == -2;
        const int64_t b0 = cu_at(lcu, e), len = cu_at(lcu, e + 1) - b0;
        if (L < 0 || len > stride || (L + 1) * stride > rows_of(lat, C)) return RAD_E_SHAPE;
        if (len > 0 && (b0 < 0 || b0 + len > y_rows)) return RAD_E_SHAPE;
        if (len > 0 && !neg && (L + 1) * stride > rows_of(cond, D)) return RAD_E_SHAPE;
    }
    for (int64_t e = 0; e < S; ++e) {
        const int64_t L = at(wl, e), neg = at(pair, e) == -2;
        const int64_t b0 = cu_at(lcu, e), len = cu_at(lcu, e + 1) - b0;
        AVX_PARALLEL_FOR_IF(len * W >= AVX_PAR_MIN)
        for (int64_t k = 0; k < len; ++k) {
            float* row = scratch_f32((size_t)W + 16);
            load_row(lat, rowoff(lat, L * stride + k, C), C, row);
            for (int64_t c = 0; c < Z; ++c) row[C + c] = 0.0f;
            if (neg) for (int64_t c = 0; c < D; ++c) row[C + Z + c] = 0.0f;
            else     load_row(cond, rowoff(cond, L * stride + k, D), D, row + C + Z);
            out_row(Y, rowoff(Y, b0 + k, W), W, row);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ win_prep */
/* A window step's start for each entry that moves its lane: FIRST draws the window's noise
 * (rad_philox_normal_s(seed, 1 + win_index, k n + c), as bf16) into the lane and, with JOIN, into
 * `noise`; JOIN binds the first min(ov, rows) rows to the previous window's end,
 * bf16(bf16(wn noise) + bf16(t prev)). A row's draw and its binding depend on that row alone, so
 * an entry's rows run in parallel, each doing both. */
AVX_KERNEL(win_prep) {
    if (!rad_args_have(a, 10)) return RAD_E_INVAL;
    const RadTensor *fl = &a->t[0], *T = &a->t[1], *seed = &a->t[2], *widx = &a->t[3];
    const RadTensor *wl = &a->t[4], *pair = &a->t[5], *lcu = &a->t[6], *prev = &a->t[7];
    const RadTensor *noise = &a->t[8], *lat = &a->t[9];
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0), ov = p_int(a, "ov", 0);
    if (n <= 0 || stride <= 0 || ov < 0 || ov > stride) return RAD_E_SHAPE;
    const int64_t S = numel(lcu) - 1;
    if (S <= 0 || numel(fl) < S || numel(T) < S || numel(widx) < S) return RAD_E_SHAPE;
    if (numel(pair) < S || numel(wl) < S) return RAD_E_SHAPE;
    for (int64_t e = 0; e < S; ++e) {
        if (at(pair, e) == -2) continue;
        const int64_t f = at(fl, e), L = at(wl, e);
        const int64_t len = cu_at(lcu, e + 1) - cu_at(lcu, e);
        if (L < 0 || len > stride || (L + 1) * stride > rows_of(lat, n) ||
            (ov && (L + 1) * ov > rows_of(noise, n)))
            return RAD_E_SHAPE;
        if ((f & kFirst) && numel(seed) < 2 * (e + 1)) return RAD_E_SHAPE;
        if ((f & kJoin) && ov && len > 0 && (L + 1) * ov > rows_of(prev, n)) return RAD_E_SHAPE;
    }
    for (int64_t e = 0; e < S; ++e) {
        if (at(pair, e) == -2) continue;
        const int64_t f = at(fl, e), L = at(wl, e);
        const int64_t len = cu_at(lcu, e + 1) - cu_at(lcu, e);
        const bool first = (f & kFirst) != 0, join = (f & kJoin) != 0;
        const uint64_t sd = first ? seed_of(seed, e) : 0;
        const uint64_t sm = 1 + (uint64_t)at(widx, e);
        const float tb = join ? rbf(ldt(T, offlin(T, e))) : 0.0f;
        const float wn = rbf(1.0f - rbf((1.0f - 1e-6f) * tb));
        const int64_t m = std::min(ov, len);
        const int64_t todo = first ? len : (join ? m : 0);
        AVX_PARALLEL_FOR_IF(todo * n >= AVX_PAR_MIN / 4)
        for (int64_t k = 0; k < todo; ++k) {
            float* v = scratch_f32((size_t)n + 16);
            float* tmp = scratch_f32_c((size_t)n + 16);
            if (first) {
                philox_normals_s(sd, sm, (uint64_t)(k * n), n, v);
                narrow_row(RAD_BF16, v, n, tmp);
                out_row(lat, rowoff(lat, L * stride + k, n), n, v);
                if (join && k < ov) out_row(noise, rowoff(noise, L * ov + k, n), n, v);
            }
            if (join && k < m) {
                float* pv = scratch_f32_b((size_t)n + 16);
                load_row(noise, rowoff(noise, L * ov + k, n), n, v);
                load_row(prev, rowoff(prev, L * ov + k, n), n, pv);
                for (int64_t c = 0; c < n; ++c) v[c] = wn * v[c];
                narrow_row(RAD_BF16, v, n, tmp);
                for (int64_t c = 0; c < n; ++c) pv[c] = tb * pv[c];
                narrow_row(RAD_BF16, pv, n, tmp);
                for (int64_t c = 0; c < n; ++c) v[c] = v[c] + pv[c];
                narrow_row(RAD_BF16, v, n, tmp);
                out_row(lat, rowoff(lat, L * stride + k, n), n, v);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ win_finish */
/* A window's end for each entry that moves its lane and carries LAST: with JOIN its first
 * min(ov, rows) rows become the previous window's end; then rows [max(0, rows - 2 ov),
 * max(.., rows - ov)) of the latent and the conditioning are kept as the next window's prev and
 * prev_cond. The two copies run one after the other, since the second may read rows the first
 * wrote. */
AVX_KERNEL(win_finish) {
    if (!rad_args_have(a, 8)) return RAD_E_INVAL;
    const RadTensor *fl = &a->t[0], *pair = &a->t[1], *wl = &a->t[2], *lcu = &a->t[3];
    const RadTensor *cond = &a->t[4], *lat = &a->t[5], *prev = &a->t[6], *pc = &a->t[7];
    const int64_t n = p_int(a, "n", 0), D = p_int(a, "n_cond", 0);
    const int64_t stride = p_int(a, "stride", 0), ov = p_int(a, "ov", 0);
    if (n <= 0 || D <= 0 || stride <= 0 || ov <= 0 || ov > stride) return RAD_E_SHAPE;
    const int64_t S = numel(lcu) - 1;
    if (S <= 0 || numel(fl) < S || numel(wl) < S) return RAD_E_SHAPE;
    if (numel(pair) < S) return RAD_E_SHAPE;
    for (int64_t e = 0; e < S; ++e) {
        const int64_t f = at(fl, e);
        if (at(pair, e) == -2 || !(f & kLast)) continue;
        const int64_t L = at(wl, e), len = cu_at(lcu, e + 1) - cu_at(lcu, e);
        if (L < 0 || len > stride || (L + 1) * stride > rows_of(lat, n) ||
            (L + 1) * ov > rows_of(prev, n) || (L + 1) * ov > rows_of(pc, D))
            return RAD_E_SHAPE;
        if (len > ov && (L + 1) * stride > rows_of(cond, D)) return RAD_E_SHAPE;
    }
    for (int64_t e = 0; e < S; ++e) {
        const int64_t f = at(fl, e);
        if (at(pair, e) == -2 || !(f & kLast)) continue;
        const int64_t L = at(wl, e), len = cu_at(lcu, e + 1) - cu_at(lcu, e);
        if (f & kJoin) {
            const int64_t m = std::min(ov, len);
            AVX_PARALLEL_FOR_IF(m * n >= AVX_PAR_MIN)
            for (int64_t k = 0; k < m; ++k)
                copy_row(prev, rowoff(prev, L * ov + k, n), lat, rowoff(lat, L * stride + k, n), n,
                         scratch_f32((size_t)n + 16));
        }
        const int64_t s0 = std::max<int64_t>(0, len - 2 * ov);
        const int64_t s1 = std::max<int64_t>(s0, len - ov);
        AVX_PARALLEL_FOR_IF((s1 - s0) * (n + D) >= AVX_PAR_MIN)
        for (int64_t k = s0; k < s1; ++k) {
            float* scr = scratch_f32((size_t)std::max(n, D) + 16);
            copy_row(lat, rowoff(lat, L * stride + k, n), prev, rowoff(prev, L * ov + k - s0, n), n,
                     scr);
            copy_row(cond, rowoff(cond, L * stride + k, D), pc, rowoff(pc, L * ov + k - s0, D), D,
                     scr);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ chan_split */
/* y[p rows + r] = x[r, p n .. p n + n): channel groups as sequences of their own. */
AVX_KERNEL(chan_split) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), P = p_int(a, "parts", 0);
    if (n <= 0 || P <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n) / P;
    if (rows <= 0 || rows_of(X, P * n) < rows) return RAD_E_SHAPE;
    const int64_t xs = laststride(X);
    AVX_PARALLEL_FOR_IF(P * rows * n >= AVX_PAR_MIN)
    for (int64_t q = 0; q < P * rows; ++q) {
        const int64_t p = q / rows, r = q % rows;
        copy_row(X, rowoff(X, r, P * n) + p * n * xs, Y, rowoff(Y, q, n), n,
                 scratch_f32((size_t)n + 16));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ interleave2 */
/* y [rows, 2] = (a[r], b[r]). */
AVX_KERNEL(interleave2) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x0 = &a->t[0], *x1 = &a->t[1], *Y = &a->t[2];
    const int64_t rows = rows_of(Y, 2);
    if (rows <= 0 || numel(x0) < rows || numel(x1) < rows) return RAD_E_SHAPE;
    const int64_t ys = laststride(Y);
    AVX_PARALLEL_FOR_IF(rows >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t yb = rowoff(Y, r, 2);
        stt(Y, yb, ldt(x0, offlin(x0, r)));
        stt(Y, yb + ys, ldt(x1, offlin(x1, r)));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ iota */
/* y[i] = start + i * step over y's elements. */
AVX_KERNEL(iota) {
    if (!rad_args_have(a, 1)) return RAD_E_INVAL;
    const RadTensor* Y = &a->t[0];
    const int64_t start = p_int(a, "start", 0), step = p_int(a, "step", 1);
    const int64_t period = p_int(a, "period", 0);
    const int64_t n = numel(Y);
    if (n <= 0 || period < 0) return RAD_E_SHAPE;
    for (int64_t i = 0; i < n; ++i)
        st_int_dt(Y->dtype, Y->data, offlin(Y, i), start + (period ? i % period : i) * step);
    return RAD_OK;
}

/* ------------------------------------------------------------------ int_affine */
/* y[j] = (src[j] < 0 ? 0 : x[src[j]]) * mul[j] + add[j] over three integer lists. */
static int avx_parse_ints(const char* s, int64_t* v, int cap) {
    int n = 0;
    while (s && *s) {
        char* e = nullptr;
        const long long x = std::strtoll(s, &e, 10);
        if (e == s) break;
        if (n == cap) return -1;
        v[n++] = x;
        s = e;
    }
    return n;
}

AVX_KERNEL(int_affine) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    int64_t src[16], mul[16], add[16];
    const int n = avx_parse_ints(p_str(a, "src", ""), src, 16);
    if (n <= 0 || avx_parse_ints(p_str(a, "mul", ""), mul, 16) != n ||
        avx_parse_ints(p_str(a, "add", ""), add, 16) != n || numel(Y) < n)
        return RAD_E_SHAPE;
    for (int j = 0; j < n; ++j) {
        if (src[j] >= numel(X)) return RAD_E_SHAPE;
        const int64_t v = src[j] < 0 ? 0 : at(X, src[j]);
        st_int_dt(Y->dtype, Y->data, offlin(Y, j), v * mul[j] + add[j]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sub_i32 */
/* y[i] = x[i] - s[0], int32. */
AVX_KERNEL(sub_i32) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *S = &a->t[1], *Y = &a->t[2];
    const int64_t n = numel(Y);
    if (n <= 0 || numel(X) < n || numel(S) < 1) return RAD_E_SHAPE;
    const int64_t s0 = at(S, 0);
    for (int64_t i = 0; i < n; ++i)
        st_int_dt(Y->dtype, Y->data, offlin(Y, i), (int64_t)(int32_t)(at(X, i) - s0));
    return RAD_OK;
}

/* ------------------------------------------------------------------ cu_repeat */
/* y[j] = j * len[0] * mult over y's elements: the bounds of equal sequences. */
AVX_KERNEL(cu_repeat) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *len = &a->t[0], *Y = &a->t[1];
    const int64_t mult = p_int(a, "mult", 1);
    const int64_t n = numel(Y);
    if (n <= 0 || numel(len) < 1 || mult <= 0) return RAD_E_SHAPE;
    const int64_t l = at(len, 0);
    for (int64_t j = 0; j < n; ++j) st_int_dt(Y->dtype, Y->data, offlin(Y, j), j * l * mult);
    return RAD_OK;
}
