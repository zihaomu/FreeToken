/* ggml_planes.h -- llama.cpp's Q8_0, Q4_K and Q6_K blocks as the affine planes libquant/lq_ggml.cpp
 * writes them, for reading a GGUF without a recipe. The core cannot call libquant, so this is the
 * reader's copy; quant_ggml_test holds the two to the same encoding, and checkpoint_test decodes
 * the planes against the block definition.
 *
 * The planes are bit for bit what lq_ggml's unpack_<type> writes from the same blocks (its header
 * lists them), so a container holds the same numbers whichever way a GGUF reached it. The blocks
 * are read as raw bytes at their GGUF offsets, so no struct layout is assumed, and gguf_dtype gives
 * their size.
 */
#pragma once
#include "rad_encoding.h"
#include "../text/gguf.h"

#include <cstdint>
#include <cstring>

namespace rad {

namespace ggml_planes {

constexpr int kQK8_0 = 32, kQK_K = 256;

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

/* Element i of a 16-bit plane row: an f16 super-scale's two bytes as the block holds them. */
inline void copy16(uint8_t* row, int64_t i, const uint8_t* half) {
    row[2 * i] = half[0];
    row[2 * i + 1] = half[1];
}

/* Two 4-bit codes per byte, element 2k low; q[] holds a block's codes in value order. */
inline void put_nibbles(uint8_t* row, const uint8_t* q, int n) {
    for (int k = 0; k < n / 2; ++k)
        row[k] = (uint8_t)((q[2 * k] & 15) | ((q[2 * k + 1] & 15) << 4));
}

/* Q4_K: value v is a nibble of qs[32*(v/64) + v%32], the low one for the first 32 of each 64. */
inline uint32_t k_low4(const uint8_t* qs, int v) {
    const uint8_t q = qs[32 * (v / 64) + v % 32];
    return (uint32_t)((v % 64) < 32 ? q & 15 : q >> 4);
}

/* Q4_K: the 6-bit scale and min of sub-block j, as llama.cpp's get_scale_min_k4 unpacks them. */
inline void k4_scale_min(int j, const uint8_t* q, uint32_t* d, uint32_t* m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0xFu) | ((uint32_t)(q[j - 4] >> 6) << 4);
        *m = (uint32_t)(q[j + 4] >> 4) | ((uint32_t)(q[j] >> 6) << 4);
    }
}

/* block_q8_0: d f16 @0, qs i8[32] @2. */
inline void split_q8_0(const uint8_t* blocks, int64_t nb, size_t bb, uint8_t* const* row) {
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = blocks + bb * i;
        std::memcpy(row[0] + kQK8_0 * i, b + 2, kQK8_0);
        copy16(row[1], i, b);
    }
}

/* block_q4_K: d f16 @0, dmin f16 @2, scales u8[12] @4, qs u8[128] @16. */
inline void split_q4_K(const uint8_t* blocks, int64_t nb, size_t bb, uint8_t* const* row) {
    Bits s(row[1]), m(row[3]);
    uint8_t q[kQK_K];
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = blocks + bb * i;
        for (int k = 0; k < kQK_K; ++k) q[k] = (uint8_t)k_low4(b + 16, k);
        put_nibbles(row[0] + (kQK_K / 2) * i, q, kQK_K);
        for (int j = 0; j < kQK_K / 32; ++j) {
            uint32_t sc, mn;
            k4_scale_min(j, b + 4, &sc, &mn);
            s.put(sc, 6);
            m.put(mn, 6);
        }
        copy16(row[2], i, b);
        copy16(row[4], i, b + 2);
    }
    s.flush();
    m.flush();
}

/* block_q6_K: ql u8[128] @0, qh u8[64] @128, scales i8[16] @192, d f16 @208. In each 128 values,
 * quarter t of 32 takes its low four bits from ql[64n + l (+32 for odd t)] (the high nibble for
 * t >= 2) and its top two from bits 2t of qh[32n + l]. */
inline void split_q6_K(const uint8_t* blocks, int64_t nb, size_t bb, uint8_t* const* row) {
    Bits c(row[0]);
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* b = blocks + bb * i;
        const uint8_t *ql = b, *qh = b + 128;
        for (int k = 0; k < kQK_K; ++k) {
            const int n = k / 128, t = (k % 128) / 32, l = k % 32;
            const uint8_t lo = ql[64 * n + l + 32 * (t & 1)];
            const uint32_t q = (uint32_t)(t < 2 ? lo & 15 : lo >> 4) |
                               (uint32_t)((qh[32 * n + l] >> (2 * t)) & 3) << 4;
            c.put(q - 32u, 6);
        }
        std::memcpy(row[1] + (kQK_K / 16) * i, b + 192, kQK_K / 16);
        copy16(row[2], i, b + 208);
    }
    c.flush();
}

/* A plane as lq_ggml's encoding_of lists it: one row high, `block` values wide. */
struct Plane {
    const char* role;
    uint32_t    dtype;
    int64_t     block;
};

/* A block type this header splits: ggml's type number, its planes in encoding order (the rest
 * empty), and the split of a row's `nb` blocks of `bb` bytes into one row of each. */
struct Type {
    uint32_t ggml;
    Plane    planes[RAD_ENC_MAX_PLANES];
    void   (*split)(const uint8_t* blocks, int64_t nb, size_t bb, uint8_t* const* row);
};

inline constexpr Type kTypes[] = {
    { 8, { { "codes", RAD_I8, 1 }, { "scale", RAD_F16, 32 } }, split_q8_0 },
    { 12,
      { { "codes", RAD_U4, 1 }, { "scale", RAD_U6, 32 }, { "scale.1", RAD_F16, 256 },
        { "min", RAD_U6, 32 }, { "min.1", RAD_F16, 256 } },
      split_q4_K },
    { 14, { { "codes", RAD_I6, 1 }, { "scale", RAD_I8, 16 }, { "scale.1", RAD_F16, 256 } },
      split_q6_K },
};

inline const Type* type_of(uint32_t ggml_type) {
    for (const Type& t : kTypes)
        if (t.ggml == ggml_type) return &t;
    return nullptr;
}

}  /* namespace ggml_planes */

/* The encoding a GGUF block type is read as, plane for plane as lq_ggml's encoding_of lists it.
 * False for a type this header does not split. */
inline bool ggml_plane_encoding(uint32_t ggml_type, RadEncoding* e) {
    const ggml_planes::Type* t = ggml_planes::type_of(ggml_type);
    if (!t) return false;
    rad_enc_clear(e);
    rad_enc_copy_str(e->scheme, "affine");
    for (const ggml_planes::Plane& p : t->planes) {
        if (!p.role) break;
        rad_enc_add_plane(e, p.role, p.dtype, 1, p.block);
    }
    return true;
}

/* One row of `cols` values held as GGUF blocks, split into one row of every plane of
 * ggml_plane_encoding's encoding: plane i's row goes to row[i], sized rad_enc_plane_dims and
 * rad_enc_row_bytes give for a one-row weight. `cols` is a whole number of blocks, and the type one
 * ggml_plane_encoding accepts. */
inline void ggml_plane_row(uint32_t ggml_type, const uint8_t* blocks, int64_t cols,
                           uint8_t* const* row) {
    const GgufDtype& gd = *gguf_dtype(ggml_type);
    ggml_planes::type_of(ggml_type)->split(blocks, cols / gd.blck, gd.block_bytes, row);
}

}  /* namespace rad */
