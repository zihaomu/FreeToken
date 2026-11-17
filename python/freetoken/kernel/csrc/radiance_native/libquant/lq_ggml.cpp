/* lq_ggml.cpp -- the `ggml` quantiser: llama.cpp's formats, llama.cpp's codes, radiance's planes.
 *
 * A row is quantised by llama.cpp's own quantize_<type> (lq_ggml_port.h), with the imatrix's
 * column importances as quant_weights, and the resulting GGUF blocks are unpacked into the affine
 * planes of rad_encoding.h. A llama.cpp model and a radiance container quantised from the same
 * weights and imatrix therefore hold the same numbers, but the container stores a generic encoding
 * instead of a block layout tied to one kernel family.
 *
 * ===================================================================== The planes
 *
 * Each type's planes, in order. [R x C] is a TILED plane's block, {R x C} a TABLE's extents, * a
 * whole extent; d and dmin are the block's f16 super-scales, copied bit for bit.
 *
 *   q8_0     codes i8; scale f16 [1x32] = d
 *   q4_0     codes i4 = q-8; scale f16 [1x32] = d
 *   q5_0     codes i5 = q-16; scale f16 [1x32] = d
 *   q4_1     codes u4; scale f16 [1x32] = d; min f16 [1x32] = -m
 *   q5_1     codes u5; scale f16 [1x32] = d; min f16 [1x32] = -m
 *   q2_K     codes u2; scale u4 [1x16]; scale.1 f16 [1x256] = d; min u4 [1x16]; min.1 f16 = dmin
 *   q3_K     codes i3 = q-4 (two low bits and the hmask bit); scale i6 [1x16] = sc-32;
 *            scale.1 f16 [1x256] = d
 *   q4_K     codes u4; scale u6 [1x32]; scale.1 f16 [1x256] = d; min u6 [1x32]; min.1 f16 = dmin
 *   q5_K     codes u5; otherwise as q4_K
 *   q6_K     codes i6 = q-32; scale i8 [1x16]; scale.1 f16 [1x256] = d
 *   iq4_nl   codes u4; table i8 {1x16} = kvalues_iq4nl; scale f16 [1x32] = d
 *   iq4_xs   codes u4; table i8 {1x16}; scale i6 [1x32] = ls-32; scale.1 f16 [1x256] = d
 *   iq2_xxs  codes u8 [1x8]; grid u8 {256x8}; signs u1; scale u5 [1x32] = 2ls+1;
 *            scale.1 f16 [1x256] = d; scale.2 f32 [*x*] = 1/8
 *   iq2_xs   codes u9 [1x8]; grid u8 {512x8}; signs u1; scale u5 [1x16] = 2ls+1; scale.1, scale.2
 *            as iq2_xxs
 *   iq2_s    codes u10 [1x8]; grid u8 {1024x8}; signs u1; scale u5 [1x16] = 2ls+1; scale.1,
 *            scale.2 as iq2_xxs
 *   iq3_xxs  codes u8 [1x4]; grid u8 {256x4}; signs u1; scale u5 [1x32] = 2ls+1;
 *            scale.1 f16 [1x256] = d; scale.2 f32 [*x*] = 1/4
 *   iq3_s    codes u9 [1x4]; grid u8 {512x4}; signs u1; scale u5 [1x32] = 2ls+1;
 *            scale.1 f16 [1x256] = d
 *   iq1_s    codes u11 [1x8]; grid i8 {2048x8}; scale u4 [1x32] = 2ls+1; scale.1 f16 [1x256] = d;
 *            zero bf16 [1x32] = -delta
 *   iq1_m    codes u11 [1x8]; grid i8 {2048x8}; scale u4 [1x16] = 2ls+1; scale.1 f16 [1x256] = d
 *            (assembled from the scale words' top nibbles); zero bf16 [1x8] = -delta
 *
 * Each line is llama.cpp's dequantize_row_<type> rewritten in the affine order (q = code | grid |
 * table, negated by its sign, less its zero, times every scale level, less every min level):
 * q4_1's `q*d + m` is `q*d - (-m)`, iq1's `dl*(grid + delta)` is `(grid - (-delta))*scale*d`, and
 * the IQ2/IQ3 `d*(0.5 + ls)*0.25` is `(2ls+1)*d/8`. Nothing is rounded: every plane holds a block
 * value exactly, so the planes convert back to the same GGUF blocks.
 *
 * The 1/8 and 1/4 are a per-tensor level, not folded into d. d/8 in f16 is exact only while it
 * stays normal (d >= 2^-11), and an IQ2 d is about a super-block's largest value over 155. On
 * Gaussian weights with std 0.02, typical for transformer linears, 98-100% of IQ2 and IQ3_XXS
 * super-blocks have d/8 or d/4 below f16's normal range and would lose up to three bits of scale.
 * As an f32 [*x*] constant the factor costs four bytes per tensor and multiplies exactly.
 *
 * Every product in the decode is exact in f32, so the affine order (code * sub-scale * d, where
 * llama.cpp forms d * sub-scale first) reproduces llama.cpp's dequantised values bit for bit. d is
 * an f16 with 11 significant bits and a code or grid value times its sub-scale has at most 12
 * (q6_K's 31 x 127 and iq4_xs's 127 x 31 are the widest), so no product rounds in either order.
 * Only the min subtraction (q4_1, q5_1, q2_K, q4_K, q5_K) rounds, once, on the same two operands.
 */
#include "lq_ggml.h"
#include "lq_ggml_port.h"
#include "lq_quant.h"
#include "rad_plugin.h"

#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace lq {

namespace {

using namespace lq::ggml;

/* ================================================================== writing a plane row */

/* Writes a plane row front to back, `bits` per element, lowest bits first (rad_encoding.h's bit
 * stream). 32 bits go out at a time and the tail on flush(), so each byte is stored exactly once
 * and never read back; this does not rely on the row arriving zeroed. */
class Bits {
  public:
    explicit Bits(uint8_t* row) : p_(row) {}
    void put(uint32_t v, int bits) {
        acc_ |= (uint64_t)(v & ((1u << bits) - 1u)) << n_;
        n_ += bits;
        if (n_ >= 32) {
            p_[0] = (uint8_t)acc_;
            p_[1] = (uint8_t)(acc_ >> 8);
            p_[2] = (uint8_t)(acc_ >> 16);
            p_[3] = (uint8_t)(acc_ >> 24);
            p_ += 4;
            acc_ >>= 32;
            n_ -= 32;
        }
    }
    void flush() {
        for (; n_ > 0; n_ -= 8, acc_ >>= 8) *p_++ = (uint8_t)acc_;
        n_ = 0;
    }

  private:
    uint8_t* p_;
    uint64_t acc_ = 0;
    int      n_ = 0;
};

/* Element i of a 16-bit plane row, the f16 or bf16 bits as they are. */
inline void put16(uint8_t* row, int64_t i, uint16_t h) {
    row[2 * i] = (uint8_t)h;
    row[2 * i + 1] = (uint8_t)(h >> 8);
}

/* Two 4-bit codes per byte, element 2k low; q[] holds a block's codes in value order. */
inline void put_nibbles(uint8_t* row, const uint8_t* q, int n) {
    for (int k = 0; k < n / 2; ++k)
        row[k] = (uint8_t)((q[2 * k] & 15) | ((q[2 * k + 1] & 15) << 4));
}

inline uint32_t le32(const uint8_t* p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

/* iq1's delta as the zero the affine order subtracts: zero = -delta, delta = -1/8 when the bit is
 * set and +1/8 when it is not. 0x3E00 and 0xBE00 are +-0.125 in bf16, exactly. */
inline uint16_t iq1_zero(bool negative_delta) { return negative_delta ? 0x3E00 : 0xBE00; }

/* ================================================================== unpacking a row's blocks
 *
 * Each unpack_<type> reads the `nb` blocks of one row and writes plane i's row at row[i]. A TABLE
 * or per-tensor plane's slot is null; write_shared writes those once per call. The index
 * arithmetic inverts the type's dequantize_row_<type>, value by value. */

void unpack_q8_0(const void* v, int64_t nb, uint8_t* const* row) {
    const block_q8_0* b = (const block_q8_0*)v;
    for (int64_t i = 0; i < nb; ++i) {
        std::memcpy(row[0] + QK8_0 * i, b[i].qs, QK8_0);
        put16(row[1], i, b[i].d);
    }
}

/* q4_0 and iq4_nl share the 32-value nibble order: value j of a block is the low nibble of qs[j],
 * value j+16 the high one. */
inline void nibbles32(const uint8_t* qs, uint8_t* q) {
    for (int j = 0; j < 16; ++j) {
        q[j] = qs[j] & 15;
        q[j + 16] = qs[j] >> 4;
    }
}

void unpack_q4_0(const void* v, int64_t nb, uint8_t* const* row) {
    const block_q4_0* b = (const block_q4_0*)v;
    uint8_t q[QK4_0];
    for (int64_t i = 0; i < nb; ++i) {
        nibbles32(b[i].qs, q);
        for (int j = 0; j < QK4_0; ++j) q[j] ^= 8;   /* q - 8 in four bits */
        put_nibbles(row[0] + 16 * i, q, QK4_0);
        put16(row[1], i, b[i].d);
    }
}

void unpack_q4_1(const void* v, int64_t nb, uint8_t* const* row) {
    const block_q4_1* b = (const block_q4_1*)v;
    uint8_t q[QK4_1];
    for (int64_t i = 0; i < nb; ++i) {
        nibbles32(b[i].qs, q);
        put_nibbles(row[0] + 16 * i, q, QK4_1);
        put16(row[1], i, b[i].d);
        put16(row[2], i, (uint16_t)(b[i].m ^ 0x8000u));
    }
}

/* q5_0 and q5_1: the fifth bit of value j is bit j of qh. */
inline void fives32(const uint8_t* qs, const uint8_t* qh4, uint8_t* q) {
    const uint32_t qh = le32(qh4);
    nibbles32(qs, q);
    for (int j = 0; j < 32; ++j) q[j] |= (uint8_t)(((qh >> j) & 1u) << 4);
}

void unpack_q5_0(const void* v, int64_t nb, uint8_t* const* row) {
    const block_q5_0* b = (const block_q5_0*)v;
    Bits c(row[0]);
    uint8_t q[QK5_0];
    for (int64_t i = 0; i < nb; ++i) {
        fives32(b[i].qs, b[i].qh, q);
        for (int j = 0; j < QK5_0; ++j) c.put((uint32_t)(q[j] - 16), 5);
        put16(row[1], i, b[i].d);
    }
    c.flush();
}

void unpack_q5_1(const void* v, int64_t nb, uint8_t* const* row) {
    const block_q5_1* b = (const block_q5_1*)v;
    Bits c(row[0]);
    uint8_t q[QK5_1];
    for (int64_t i = 0; i < nb; ++i) {
        fives32(b[i].qs, b[i].qh, q);
        for (int j = 0; j < QK5_1; ++j) c.put(q[j], 5);
        put16(row[1], i, b[i].d);
        put16(row[2], i, (uint16_t)(b[i].m ^ 0x8000u));
    }
    c.flush();
}

/* q2_K and q3_K: value v's two low bits are bits 2*((v%128)/32) of qs[32*(v/128) + v%32]. */
inline uint32_t k_low2(const uint8_t* qs, int v) {
    return (uint32_t)(qs[32 * (v / 128) + v % 32] >> (2 * ((v % 128) / 32))) & 3u;
}

void unpack_q2_K(const void* v, int64_t nb, uint8_t* const* row) {
    const block_q2_K* b = (const block_q2_K*)v;
    Bits c(row[0]), s(row[1]), m(row[3]);
    for (int64_t i = 0; i < nb; ++i) {
        for (int k = 0; k < QK_K; ++k) c.put(k_low2(b[i].qs, k), 2);
        for (int j = 0; j < QK_K / 16; ++j) {
            s.put(b[i].scales[j] & 15u, 4);
            m.put(b[i].scales[j] >> 4, 4);
        }
        put16(row[2], i, b[i].d);
        put16(row[4], i, b[i].dmin);
    }
    c.flush();
    s.flush();
    m.flush();
}

void unpack_q3_K(const void* v, int64_t nb, uint8_t* const* row) {
    const block_q3_K* b = (const block_q3_K*)v;
    Bits c(row[0]), s(row[1]);
    for (int64_t i = 0; i < nb; ++i) {
        const block_q3_K& x = b[i];
        /* The high bit of value k is bit k/32 of hmask[k%32]. */
        for (int k = 0; k < QK_K; ++k)
            c.put(k_low2(x.qs, k) + 4u * ((x.hmask[k % 32] >> (k / 32)) & 1u) - 4u, 3);
        /* The sixteen 6-bit scales, unpacked as quantize_row_q3_K_ref re-reads them. */
        for (int j = 0; j < QK_K / 16; ++j) {
            const int lo = j < 8 ? x.scales[j] & 0xF : x.scales[j - 8] >> 4;
            const int sc = lo | (((x.scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4);
            s.put((uint32_t)(sc - 32), 6);
        }
        put16(row[2], i, x.d);
    }
    c.flush();
    s.flush();
}

/* q4_K and q5_K: value v's low four bits are a nibble of qs[32*(v/64) + v%32], the low one for
 * the first 32 of each 64; q5_K's fifth is bit v/32 of qh[v%32]. The 6-bit scale and min of
 * sub-block j are get_scale_min_k4's. */
inline uint32_t k_low4(const uint8_t* qs, int v) {
    const uint8_t q = qs[32 * (v / 64) + v % 32];
    return (uint32_t)((v % 64) < 32 ? q & 15 : q >> 4);
}

inline void k4_scale_min(int j, const uint8_t* q, uint32_t* d, uint32_t* m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0xFu) | ((uint32_t)(q[j - 4] >> 6) << 4);
        *m = (uint32_t)(q[j + 4] >> 4) | ((uint32_t)(q[j - 0] >> 6) << 4);
    }
}

void unpack_q4_K(const void* v, int64_t nb, uint8_t* const* row) {
    const block_q4_K* b = (const block_q4_K*)v;
    Bits s(row[1]), m(row[3]);
    uint8_t q[QK_K];
    for (int64_t i = 0; i < nb; ++i) {
        for (int k = 0; k < QK_K; ++k) q[k] = (uint8_t)k_low4(b[i].qs, k);
        put_nibbles(row[0] + (QK_K / 2) * i, q, QK_K);
        for (int j = 0; j < QK_K / 32; ++j) {
            uint32_t sc, mn;
            k4_scale_min(j, b[i].scales, &sc, &mn);
            s.put(sc, 6);
            m.put(mn, 6);
        }
        put16(row[2], i, b[i].d);
        put16(row[4], i, b[i].dmin);
    }
    s.flush();
    m.flush();
}

void unpack_q5_K(const void* v, int64_t nb, uint8_t* const* row) {
    const block_q5_K* b = (const block_q5_K*)v;
    Bits c(row[0]), s(row[1]), m(row[3]);
    for (int64_t i = 0; i < nb; ++i) {
        for (int k = 0; k < QK_K; ++k)
            c.put(k_low4(b[i].qs, k) + 16u * ((b[i].qh[k % 32] >> (k / 32)) & 1u), 5);
        for (int j = 0; j < QK_K / 32; ++j) {
            uint32_t sc, mn;
            k4_scale_min(j, b[i].scales, &sc, &mn);
            s.put(sc, 6);
            m.put(mn, 6);
        }
        put16(row[2], i, b[i].d);
        put16(row[4], i, b[i].dmin);
    }
    c.flush();
    s.flush();
    m.flush();
}

/* q6_K: in each 128 values, quarter t of 32 takes its low four bits from ql[64n + l (+32 for odd
 * t)] (the high nibble for t >= 2) and its top two from bits 2t of qh[32n + l]. */
void unpack_q6_K(const void* v, int64_t nb, uint8_t* const* row) {
    const block_q6_K* b = (const block_q6_K*)v;
    Bits c(row[0]);
    for (int64_t i = 0; i < nb; ++i) {
        const block_q6_K& x = b[i];
        for (int k = 0; k < QK_K; ++k) {
            const int n = k / 128, t = (k % 128) / 32, l = k % 32;
            const uint8_t lo = x.ql[64 * n + l + 32 * (t & 1)];
            const uint32_t q = (uint32_t)(t < 2 ? lo & 15 : lo >> 4) |
                               (uint32_t)((x.qh[32 * n + l] >> (2 * t)) & 3) << 4;
            c.put(q - 32u, 6);
        }
        std::memcpy(row[1] + (QK_K / 16) * i, x.scales, QK_K / 16);
        put16(row[2], i, x.d);
    }
    c.flush();
}

void unpack_iq4_nl(const void* v, int64_t nb, uint8_t* const* row) {
    const block_iq4_nl* b = (const block_iq4_nl*)v;
    uint8_t q[QK4_NL];
    for (int64_t i = 0; i < nb; ++i) {
        nibbles32(b[i].qs, q);
        put_nibbles(row[0] + 16 * i, q, QK4_NL);
        put16(row[2], i, b[i].d);
    }
}

void unpack_iq4_xs(const void* v, int64_t nb, uint8_t* const* row) {
    const block_iq4_xs* b = (const block_iq4_xs*)v;
    Bits s(row[2]);
    uint8_t q[32];
    for (int64_t i = 0; i < nb; ++i) {
        const block_iq4_xs& x = b[i];
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            nibbles32(x.qs + 16 * ib, q);
            put_nibbles(row[0] + (QK_K / 2) * i + 16 * ib, q, 32);
            const int ls = ((x.scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) |
                           (((x.scales_h >> 2 * ib) & 3) << 4);
            s.put((uint32_t)(ls - 32), 6);
        }
        put16(row[3], i, x.d);
    }
    s.flush();
}

/* IQ2 and IQ3: a sign byte covers eight values, bit j for value j. That is the u1 plane's own
 * order, so a group of eight's signs is one byte of it. */
void unpack_iq2_xxs(const void* v, int64_t nb, uint8_t* const* row) {
    const block_iq2_xxs* b = (const block_iq2_xxs*)v;
    Bits s(row[3]);
    for (int64_t i = 0; i < nb; ++i) {
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const uint16_t* q = b[i].qs + 4 * ib32;
            const uint32_t a0 = (uint32_t)q[0] | (uint32_t)q[1] << 16;
            const uint32_t a1 = (uint32_t)q[2] | (uint32_t)q[3] << 16;
            for (int l = 0; l < 4; ++l) {
                row[0][32 * i + 4 * ib32 + l] = (uint8_t)(a0 >> 8 * l);
                row[2][32 * i + 4 * ib32 + l] = ksigns_iq2xs[(a1 >> 7 * l) & 127];
            }
            s.put(2 * (a1 >> 28) + 1, 5);
        }
        put16(row[4], i, b[i].d);
    }
    s.flush();
}

void unpack_iq2_xs(const void* v, int64_t nb, uint8_t* const* row) {
    const block_iq2_xs* b = (const block_iq2_xs*)v;
    Bits c(row[0]), s(row[3]);
    for (int64_t i = 0; i < nb; ++i) {
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            for (int l = 0; l < 4; ++l) {
                const uint16_t q = b[i].qs[4 * ib32 + l];
                c.put(q & 511u, 9);
                row[2][32 * i + 4 * ib32 + l] = ksigns_iq2xs[q >> 9];
            }
            s.put(2u * (b[i].scales[ib32] & 0xfu) + 1u, 5);
            s.put(2u * (b[i].scales[ib32] >> 4) + 1u, 5);
        }
        put16(row[4], i, b[i].d);
    }
    c.flush();
    s.flush();
}

void unpack_iq2_s(const void* v, int64_t nb, uint8_t* const* row) {
    const block_iq2_s* b = (const block_iq2_s*)v;
    Bits c(row[0]), s(row[3]);
    for (int64_t i = 0; i < nb; ++i) {
        const block_iq2_s& x = b[i];
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            for (int l = 0; l < 4; ++l) {
                c.put(x.qs[4 * ib32 + l] | ((x.qh[ib32] << (8 - 2 * l)) & 0x300u), 10);
                row[2][32 * i + 4 * ib32 + l] = x.qs[QK_K / 8 + 4 * ib32 + l];
            }
            s.put(2u * (x.scales[ib32] & 0xfu) + 1u, 5);
            s.put(2u * (x.scales[ib32] >> 4) + 1u, 5);
        }
        put16(row[4], i, x.d);
    }
    c.flush();
    s.flush();
}

void unpack_iq3_xxs(const void* v, int64_t nb, uint8_t* const* row) {
    const block_iq3_xxs* b = (const block_iq3_xxs*)v;
    Bits s(row[3]);
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* qs = b[i].qs;
        std::memcpy(row[0] + (QK_K / 4) * i, qs, QK_K / 4);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const uint32_t a = le32(qs + QK_K / 4 + 4 * ib32);
            for (int l = 0; l < 4; ++l)
                row[2][32 * i + 4 * ib32 + l] = ksigns_iq2xs[(a >> 7 * l) & 127];
            s.put(2 * (a >> 28) + 1, 5);
        }
        put16(row[4], i, b[i].d);
    }
    s.flush();
}

void unpack_iq3_s(const void* v, int64_t nb, uint8_t* const* row) {
    const block_iq3_s* b = (const block_iq3_s*)v;
    Bits c(row[0]), s(row[3]);
    for (int64_t i = 0; i < nb; ++i) {
        const block_iq3_s& x = b[i];
        /* The ninth bit of 4-group g's grid index is bit g%8 of qh[g/8]. */
        for (int g = 0; g < QK_K / 4; ++g)
            c.put(x.qs[g] | (uint32_t)((x.qh[g / 8] >> (g % 8)) & 1) << 8, 9);
        std::memcpy(row[2] + (QK_K / 8) * i, x.signs, QK_K / 8);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32)
            s.put(1u + 2u * ((x.scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xfu), 5);
        put16(row[4], i, x.d);
    }
    c.flush();
    s.flush();
}

void unpack_iq1_s(const void* v, int64_t nb, uint8_t* const* row) {
    const block_iq1_s* b = (const block_iq1_s*)v;
    Bits c(row[0]), s(row[2]);
    for (int64_t i = 0; i < nb; ++i) {
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const uint16_t h = b[i].qh[ib];
            for (int l = 0; l < 4; ++l)
                c.put(b[i].qs[4 * ib + l] | (((uint32_t)h >> 3 * l) & 7u) << 8, 11);
            s.put(2u * ((h >> 12) & 7u) + 1u, 4);
            put16(row[4], (QK_K / 32) * i + ib, iq1_zero(h & 0x8000));
        }
        put16(row[3], i, b[i].d);
    }
    c.flush();
    s.flush();
}

void unpack_iq1_m(const void* v, int64_t nb, uint8_t* const* row) {
    const block_iq1_m* b = (const block_iq1_m*)v;
    Bits c(row[0]), s(row[2]);
    for (int64_t i = 0; i < nb; ++i) {
        const block_iq1_m& x = b[i];
        uint16_t sc[4];
        for (int k = 0; k < 4; ++k) sc[k] = (uint16_t)(x.scales[2 * k] | x.scales[2 * k + 1] << 8);
        const uint16_t d = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                                      ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const uint8_t* qs = x.qs + 4 * ib;
            const uint8_t h0 = x.qh[2 * ib], h1 = x.qh[2 * ib + 1];
            c.put(qs[0] | ((h0 << 8) & 0x700u), 11);
            c.put(qs[1] | ((h0 << 4) & 0x700u), 11);
            c.put(qs[2] | ((h1 << 8) & 0x700u), 11);
            c.put(qs[3] | ((h1 << 4) & 0x700u), 11);
            const int64_t g = (QK_K / 8) * i + 4 * ib;
            put16(row[4], g + 0, iq1_zero(h0 & 0x08));
            put16(row[4], g + 1, iq1_zero(h0 & 0x80));
            put16(row[4], g + 2, iq1_zero(h1 & 0x08));
            put16(row[4], g + 3, iq1_zero(h1 & 0x80));
            s.put(2u * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7u) + 1u, 4);
            s.put(2u * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7u) + 1u, 4);
        }
        put16(row[3], i, d);
    }
    c.flush();
    s.flush();
}

/* ================================================================== the types */

typedef size_t (*QuantizeFn)(const float*, void*, int64_t, int64_t, const float*);
typedef void (*UnpackFn)(const void* blocks, int64_t nb, uint8_t* const* row);

struct Kind {
    const char* name;      /* as llama.cpp spells it */
    ggml_type   type;
    int64_t     block;     /* values one GGUF block covers */
    QuantizeFn  quantize;
    UnpackFn    unpack;
};

const Kind kKinds[] = {
    { "q4_0",    GGML_TYPE_Q4_0,    QK4_0,  quantize_q4_0,    unpack_q4_0 },
    { "q4_1",    GGML_TYPE_Q4_1,    QK4_1,  quantize_q4_1,    unpack_q4_1 },
    { "q5_0",    GGML_TYPE_Q5_0,    QK5_0,  quantize_q5_0,    unpack_q5_0 },
    { "q5_1",    GGML_TYPE_Q5_1,    QK5_1,  quantize_q5_1,    unpack_q5_1 },
    { "q8_0",    GGML_TYPE_Q8_0,    QK8_0,  quantize_q8_0,    unpack_q8_0 },
    { "q2_K",    GGML_TYPE_Q2_K,    QK_K,   quantize_q2_K,    unpack_q2_K },
    { "q3_K",    GGML_TYPE_Q3_K,    QK_K,   quantize_q3_K,    unpack_q3_K },
    { "q4_K",    GGML_TYPE_Q4_K,    QK_K,   quantize_q4_K,    unpack_q4_K },
    { "q5_K",    GGML_TYPE_Q5_K,    QK_K,   quantize_q5_K,    unpack_q5_K },
    { "q6_K",    GGML_TYPE_Q6_K,    QK_K,   quantize_q6_K,    unpack_q6_K },
    { "iq4_nl",  GGML_TYPE_IQ4_NL,  QK4_NL, quantize_iq4_nl,  unpack_iq4_nl },
    { "iq4_xs",  GGML_TYPE_IQ4_XS,  QK_K,   quantize_iq4_xs,  unpack_iq4_xs },
    { "iq2_xxs", GGML_TYPE_IQ2_XXS, QK_K,   quantize_iq2_xxs, unpack_iq2_xxs },
    { "iq2_xs",  GGML_TYPE_IQ2_XS,  QK_K,   quantize_iq2_xs,  unpack_iq2_xs },
    { "iq2_s",   GGML_TYPE_IQ2_S,   QK_K,   quantize_iq2_s,   unpack_iq2_s },
    { "iq3_xxs", GGML_TYPE_IQ3_XXS, QK_K,   quantize_iq3_xxs, unpack_iq3_xxs },
    { "iq3_s",   GGML_TYPE_IQ3_S,   QK_K,   quantize_iq3_s,   unpack_iq3_s },
    { "iq1_s",   GGML_TYPE_IQ1_S,   QK_K,   quantize_iq1_s,   unpack_iq1_s },
    { "iq1_m",   GGML_TYPE_IQ1_M,   QK_K,   quantize_iq1_m,   unpack_iq1_m },
};

/* The encoding a type writes, plane for plane as listed in the file header. */
RadEncoding encoding_of(ggml_type t) {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    auto plane = [&](const char* role, uint32_t dt, int64_t bc) {
        rad_enc_add_plane(&e, role, dt, 1, bc);
    };
    switch (t) {
        case GGML_TYPE_Q8_0:
            plane("codes", RAD_I8, 1);
            plane("scale", RAD_F16, 32);
            break;
        case GGML_TYPE_Q4_0:
            plane("codes", RAD_I4, 1);
            plane("scale", RAD_F16, 32);
            break;
        case GGML_TYPE_Q5_0:
            plane("codes", RAD_I5, 1);
            plane("scale", RAD_F16, 32);
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_1:
            plane("codes", t == GGML_TYPE_Q4_1 ? RAD_U4 : RAD_U5, 1);
            plane("scale", RAD_F16, 32);
            plane("min", RAD_F16, 32);
            break;
        case GGML_TYPE_Q2_K:
            plane("codes", RAD_U2, 1);
            plane("scale", RAD_U4, 16);
            plane("scale.1", RAD_F16, 256);
            plane("min", RAD_U4, 16);
            plane("min.1", RAD_F16, 256);
            break;
        case GGML_TYPE_Q3_K:
            plane("codes", RAD_I3, 1);
            plane("scale", RAD_I6, 16);
            plane("scale.1", RAD_F16, 256);
            break;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            plane("codes", t == GGML_TYPE_Q4_K ? RAD_U4 : RAD_U5, 1);
            plane("scale", RAD_U6, 32);
            plane("scale.1", RAD_F16, 256);
            plane("min", RAD_U6, 32);
            plane("min.1", RAD_F16, 256);
            break;
        case GGML_TYPE_Q6_K:
            plane("codes", RAD_I6, 1);
            plane("scale", RAD_I8, 16);
            plane("scale.1", RAD_F16, 256);
            break;
        case GGML_TYPE_IQ4_NL:
            plane("codes", RAD_U4, 1);
            rad_enc_add_table(&e, "table", RAD_I8, 1, 16);
            plane("scale", RAD_F16, 32);
            break;
        case GGML_TYPE_IQ4_XS:
            plane("codes", RAD_U4, 1);
            rad_enc_add_table(&e, "table", RAD_I8, 1, 16);
            plane("scale", RAD_I6, 32);
            plane("scale.1", RAD_F16, 256);
            break;
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S: {
            const int n = t == GGML_TYPE_IQ2_XXS ? 256 : t == GGML_TYPE_IQ2_XS ? 512 : 1024;
            plane("codes", n == 256 ? RAD_U8 : n == 512 ? RAD_U9 : RAD_U10, 8);
            rad_enc_add_table(&e, "grid", RAD_U8, n, 8);
            plane("signs", RAD_U1, 1);
            plane("scale", RAD_U5, t == GGML_TYPE_IQ2_XXS ? 32 : 16);
            plane("scale.1", RAD_F16, 256);
            rad_enc_add_plane(&e, "scale.2", RAD_F32, 0, 0);
            break;
        }
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            plane("codes", t == GGML_TYPE_IQ3_XXS ? RAD_U8 : RAD_U9, 4);
            rad_enc_add_table(&e, "grid", RAD_U8, t == GGML_TYPE_IQ3_XXS ? 256 : 512, 4);
            plane("signs", RAD_U1, 1);
            plane("scale", RAD_U5, 32);
            plane("scale.1", RAD_F16, 256);
            if (t == GGML_TYPE_IQ3_XXS) rad_enc_add_plane(&e, "scale.2", RAD_F32, 0, 0);
            break;
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            plane("codes", RAD_U11, 8);
            rad_enc_add_table(&e, "grid", RAD_I8, NGRID_IQ1S, 8);
            plane("scale", RAD_U4, t == GGML_TYPE_IQ1_S ? 32 : 16);
            plane("scale.1", RAD_F16, 256);
            plane("zero", RAD_BF16, t == GGML_TYPE_IQ1_S ? 32 : 8);
            break;
    }
    return e;
}

/* Grid row k of an I type's codebook: the bytes llama.cpp's dequantiser reads at
 * (const uint8_t *)(grid + k), lowest first. */
template <class W>
void fill_grid(uint8_t* out, const W* grid, int n) {
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < (int)sizeof(W); ++j)
            out[k * (int)sizeof(W) + j] = (uint8_t)(grid[k] >> (8 * j));
}

/* The planes all rows share (a codebook, IQ4's levels, IQ2/IQ3's 1/8 or 1/4). Every call writes
 * them in full, because rad-convert decodes each call's rows as soon as they are written. */
void write_shared(ggml_type t, const RadEncoding& e, void* const* planes) {
    for (int i = 0; i < e.n_planes; ++i) {
        const RadEncPlane& p = e.plane[i];
        uint8_t* out = (uint8_t*)planes[i];
        if (!std::strcmp(p.role, "table")) {
            std::memcpy(out, kvalues_iq4nl, 16);
        } else if (!std::strcmp(p.role, "grid")) {
            switch (t) {
                case GGML_TYPE_IQ2_XXS: fill_grid(out, iq2xxs_grid, 256); break;
                case GGML_TYPE_IQ2_XS:  fill_grid(out, iq2xs_grid, 512); break;
                case GGML_TYPE_IQ2_S:   fill_grid(out, iq2s_grid, 1024); break;
                case GGML_TYPE_IQ3_XXS: fill_grid(out, iq3xxs_grid, 256); break;
                case GGML_TYPE_IQ3_S:   fill_grid(out, iq3s_grid, 512); break;
                default:                fill_grid(out, iq1s_grid, NGRID_IQ1S); break;
            }
        } else if (!std::strcmp(p.role, "scale.2")) {
            const float k = t == GGML_TYPE_IQ3_XXS ? 0.25f : 0.125f;
            uint32_t u;
            std::memcpy(&u, &k, 4);
            for (int j = 0; j < 4; ++j) out[j] = (uint8_t)(u >> (8 * j));
        }
    }
}

/* ================================================================== the quantiser */

bool same_name(const char* a, const char* b) {
    for (; *a && *b; ++a, ++b)
        if (std::tolower((unsigned char)*a) != std::tolower((unsigned char)*b)) return false;
    return *a == *b;
}

/* The type the options name, or RAD_E_INVAL with a message naming the weight. */
int kind_of(const RadParam* o, int n_o, const RadQuantWeight* w, const Kind** out) {
    const char* t = rad_param_gets(o, n_o, "type", nullptr);
    if (!t)
        return refuse(w, "type= is required: the llama.cpp type to write, e.g. q4_K, q8_0, iq4_xs");
    for (const Kind& k : kKinds)
        if (same_name(t, k.name)) {
            *out = &k;
            return RAD_OK;
        }
    std::string all;
    for (const Kind& k : kKinds) all += std::string(all.empty() ? "" : " ") + k.name;
    return refuse(w, ("type=" + std::string(t) + " is not a llama.cpp type this quantiser writes (" +
                      all + ")").c_str());
}

/* A rank-1 weight, or one whose rows do not divide into the type's blocks, is declined so the next
 * rule takes it (a norm, or a 2880-wide projection a q4_K pattern happened to match). */
int ggml_encoding(const RadParam* o, int n_o, const RadQuantWeight* w, RadEncoding* out) {
    if (declines(w)) return RAD_E_UNSUPPORTED;
    const Kind* k = nullptr;
    const int st = kind_of(o, n_o, w, &k);
    if (st != RAD_OK) return st;
    if (w->cols <= 0 || w->rows <= 0) return RAD_E_SHAPE;
    if (w->cols % k->block) return RAD_E_UNSUPPORTED;
    *out = encoding_of(k->type);
    return RAD_OK;
}

/* Rows are independent in every type: a block never spans rows, and the imatrix is per column. */
int64_t ggml_row_block(const RadParam*, int, const RadQuantWeight*) { return 1; }

int ggml_quantize(const RadParam* o, int n_o, const RadQuantWeight* w, float* src, int64_t row0,
                  int64_t rows, void* const* planes) {
    const Kind* k = nullptr;
    const int st = kind_of(o, n_o, w, &k);
    if (st != RAD_OK) return st;
    if (!w || w->rank < 2 || w->cols <= 0 || w->rows <= 0 || w->cols % k->block || row0 < 0 ||
        rows < 0 || row0 + rows > w->rows)
        return RAD_E_SHAPE;
    /* llama.cpp's iq2_xxs, iq2_xs and iq1_s require the imatrix and have no other path;
     * substituting one would write, under llama.cpp's type name, data llama.cpp could not have
     * produced. */
    if (!w->importance && ggml_quantize_requires_imatrix(k->type))
        return refuse(w, ("type=" + std::string(k->name) + " needs an imatrix, and the converter "
                          "has none for this weight (rad-convert --imatrix FILE)").c_str());
    ggml_quantize_init(k->type);

    const RadEncoding e = encoding_of(k->type);
    const int64_t C = w->cols, nb = C / k->block;
    int64_t row_bytes[RAD_ENC_MAX_PLANES] = {};
    bool per_row[RAD_ENC_MAX_PLANES] = {};
    for (int i = 0; i < e.n_planes; ++i) {
        const RadEncPlane& p = e.plane[i];
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&p, w->rows, C, &pr, &pc);
        row_bytes[i] = rad_enc_row_bytes(p.dtype, pc);
        per_row[i] = p.kind == RAD_PLANE_TILED && p.block[0] == 1;
    }
    write_shared(k->type, e, planes);

    const size_t rsz = ggml_row_size(k->type, C);
    const float* imp = w->importance;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<uint8_t> blocks(rsz);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int64_t r = 0; r < rows; ++r) {
            k->quantize(src + r * C, blocks.data(), 1, C, imp);
            uint8_t* row[RAD_ENC_MAX_PLANES] = {};
            for (int i = 0; i < e.n_planes; ++i)
                if (per_row[i]) row[i] = (uint8_t*)planes[i] + r * row_bytes[i];
            k->unpack(blocks.data(), nb, row);
        }
    }
    return RAD_OK;
}

const RadQuantOption kOptions[] = {
    { "type", RAD_P_STR, nullptr,
      "the llama.cpp type, any case: q4_0 q4_1 q5_0 q5_1 q8_0 q2_K q3_K q4_K q5_K q6_K iq4_nl "
      "iq4_xs iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq1_s iq1_m. iq2_xxs, iq2_xs and iq1_s need an "
      "imatrix" },
};

}  /* namespace */

const RadQuantizerInfo ggml_quantizer = {
    "ggml",
    "llama.cpp's legacy, K- and I-quants, the codes and scales chosen by llama.cpp's own search "
    "(with the imatrix when the converter has one) and stored as affine planes",
    kOptions, (int)(sizeof kOptions / sizeof kOptions[0]),
    ggml_encoding, ggml_row_block, ggml_quantize, nullptr,
};

}  /* namespace lq */
