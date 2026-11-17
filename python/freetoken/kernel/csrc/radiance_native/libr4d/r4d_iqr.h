/* r4d_iqr.h -- the narrow int8-grid weights (IQR) as libr4d keeps them in VRAM: the stored codes
 * reordered for the GEMMs that decode them, never widened. Pure C++ (no HIP), shared by the layout
 * hooks (r4d_iqr_layout.cpp) and the kernels (r4d_gemm_iqr.hip).
 *
 * Every form is an int8 value per weight times a scale per (row, 128 K), the int8 grid IQ8R's GEMMs
 * read, and every form's int8 value is a function of its codes alone, so a kernel decodes it where
 * it reads it:
 *   - Signed   iB codes, the value itself (B = 2..7);
 *   - Zero     uB codes less a zero per (row, 128) (u6 codes under a u6 zero in IQ6R);
 *   - Table    uB codes indexing a table of up to 2^B int8 values (IQR-V1's scalar codebooks);
 *   - Grid     one uB code a V-vector of K, indexing a table of 2^B int8 vectors (V1's vector
 *              codebooks: V = 4, B = 8..12);
 *   - Trellis  uB codes a bitshift trellis shifts in down each column: row r's value is the table
 *              entry of state_r = (state_{r-1} << B | code_r) mod 2^L, tail-biting over runs of T
 *              rows (IQR-V2, L = 12; IQR-V3, L = 14 on the tcq_rb grid, below);
 *   - TrellisMad the same trellis whose state values are computed, not read: state st's value is
 *              bytesum((st * A + B) mod 2^32 & 0x3F3F3F3F) - 126 (libquant's tcq_code=mad), one
 *              multiply-add, a mask and a byte-sum dot a value, so the GEMMs read no table and
 *              L = 16 costs nothing more than L = 14.
 * The scale is a bf16 per (row, 128), or an E4M3 per (row, 128) under an f32 for the tensor or the
 * row, whose product rounded to bf16 is the row's scale (libquant's grid_tcq_settle). A trellis on
 * libquant's tcq_rb grid (rb > 1) reads one scale per rb rows, the largest of the block's own, and
 * a state's value v becomes rint(v * own / B) against it, the int8 the codes were chosen against.
 *
 * ---- the form word ------------------------------------------------------------------------------
 *
 * The launch sees a weight only as its stored dtype and declared [N, K], so the form rides in the
 * stored dtype: R4D_DT_IQR_W | word for the weight buffer and R4D_DT_IQR_S | word for its scales.
 *
 * ---- the codes: lane segments ------------------------------------------------------------------
 *
 * The unit is a 16-row x 128-K tile, one scale block of one 16-row tile (tile rt, block kb at unit
 * rt * K/128 + kb). A unit is 32 lane segments of lane_bits() each, back to back: lane l holds
 * columns 4l .. 4l + 3 of the block for the tile's 16 rows, so a lane's four values of a row are
 * one sudot4 against four activation bytes, and a 16-row x 128-K int8 tile in WMMA fragment order
 * is one LDS transpose away (r4d_iqr_dec.h). Within a segment:
 *   - scalar forms: row r's four codes at codes 4r .. 4r + 3, B bits each, the first in the low
 *     bits;
 *   - Grid: row r's 4/V codes at codes (4/V) r .., B bits each (V = 2 or 4);
 *   - Trellis: column e's 16 codes at codes 16e .. 16e + 15, bottom-up (row 15 of the tile first,
 *     row 0 last). The codes after a column's last are then the first ones of the same column in
 *     the unit above (rt - 1, or the run's last tile at a run's first, the run being tail-biting),
 *     so a row's state is the L bits starting at its own code in that stream, its code lowest.
 * A unit is 4 * lane_bits() bytes, and the codes plane is exactly N*K*B/8 bytes (N*K*B/(8V) for a
 * grid): a reorder of the stored bits.
 *
 * ---- the rest ------------------------------------------------------------------------------------
 *
 * After the codes, at aux_off(), the form's table (Table: 256 int8, zero-padded; Grid: 2^B * V;
 * Trellis: 2^L; TrellisMad: A and B alone as two u32, the values being computed from them where
 * they are read and the stored table regenerated on the way back) or, for Zero, its zero plane as
 * u8 in the scale order below. The scale buffer is
 * tile-major, [N/16][K/128][16]: the 16 rows of a tile side by side for one 128-K block, as the
 * int8 GEMMs' (r4d_fp8_layout.cpp): bf16, or E4M3 bytes followed at s2_off() by the second level as
 * f32 (one, or one a row) and, with rb > 1, at blk_off() the bf16 block scale B per [N/rb][K/128].
 */
#pragma once
#include <cstdint>

#if defined(__HIPCC__) || defined(__HIP__)
#define R4D_IQR_HD __host__ __device__
#else
#define R4D_IQR_HD
#endif

namespace r4d_iqr {

enum Kind : uint32_t { kSigned = 1, kZero = 2, kTable = 3, kGrid = 4, kTrellis = 5, kTrellisMad = 6 };
/* Scale forms. kRowF32: one f32 a row over the whole K times one f32 a 128-K block shared by every
 * row (libquant's tcq_colf; 1 without it): a GEMM folds the block's factor with the activation's
 * per-128 scale and the row's in its epilogue, so its K loop reads no weight scale a row. */
enum Scale : uint32_t { kBf16 = 0, kE4m3Tensor = 1, kE4m3Row = 2, kRowF32 = 3 };

/* The private dtype bases (rad_types.h's RAD_DT_PLUGIN_BASE and up are the plugin's). */
constexpr uint32_t R4D_DT_IQR_W = 0x40000000u, R4D_DT_IQR_S = 0x50000000u;
constexpr uint32_t kWordMask = 0x0FFFFFFFu;

struct Form {
    uint32_t kind = 0;    /* Kind */
    uint32_t bits = 0;    /* B: code bits */
    uint32_t lv = 0;      /* Grid: log2 V */
    uint32_t L = 0;       /* Trellis: state bits */
    uint32_t lrb = 0;     /* Trellis: log2 rows a scale (tcq_rb), 0 for one a row */
    uint32_t scale = 0;   /* Scale */
    uint32_t lrun = 0;    /* Trellis: log2 rows a run */
    uint32_t rot = 0;     /* the embedding's forms: each 128 of a row rotated (fwht128) */
};

R4D_IQR_HD constexpr uint32_t word(const Form& f) {
    return (f.kind & 15u) | (f.bits & 31u) << 4 | (f.lv & 7u) << 9 | (f.L & 31u) << 12 |
           (f.lrb & 7u) << 17 | (f.scale & 3u) << 20 | (f.lrun & 15u) << 22 | (f.rot & 1u) << 26;
}
R4D_IQR_HD constexpr Form form(uint32_t w) {
    Form f;
    f.kind = w & 15u;
    f.bits = (w >> 4) & 31u;
    f.lv = (w >> 9) & 7u;
    f.L = (w >> 12) & 31u;
    f.lrb = (w >> 17) & 7u;
    f.scale = (w >> 20) & 3u;
    f.lrun = (w >> 22) & 15u;
    f.rot = (w >> 26) & 1u;
    return f;
}
inline bool is_weight_dtype(uint32_t dt) { return (dt & ~kWordMask) == R4D_DT_IQR_W; }
inline bool is_scale_dtype(uint32_t dt) { return (dt & ~kWordMask) == R4D_DT_IQR_S; }

/* A trellis kind: the codes' stream, the run and the continuation into the unit above. */
R4D_IQR_HD constexpr bool is_trellis(uint32_t kind) { return kind == kTrellis || kind == kTrellisMad; }

/* Bits of one lane's segment of a unit. */
R4D_IQR_HD constexpr uint32_t lane_bits(const Form& f) {
    return f.kind == kGrid ? (64u >> f.lv) * f.bits : 64u * f.bits;
}
R4D_IQR_HD constexpr int64_t unit_bytes(const Form& f) { return 4 * (int64_t)lane_bits(f); }
R4D_IQR_HD constexpr int64_t codes_bytes(const Form& f, int64_t N, int64_t K) {
    return (N / 16) * (K / 128) * unit_bytes(f);
}
R4D_IQR_HD constexpr int64_t align256(int64_t b) { return (b + 255) / 256 * 256; }
/* The codes plane is followed by 16 bytes a lane's last dword read may reach. */
R4D_IQR_HD constexpr int64_t aux_off(const Form& f, int64_t N, int64_t K) {
    return align256(codes_bytes(f, N, K) + 16);
}
R4D_IQR_HD constexpr int64_t aux_bytes(const Form& f, int64_t N, int64_t K) {
    return f.kind == kTable ? 256
         : f.kind == kGrid ? ((int64_t)1 << f.bits) << f.lv
         : f.kind == kTrellis ? (int64_t)1 << f.L
         : f.kind == kTrellisMad ? 8
         : f.kind == kZero ? N * (K / 128)
         : 0;
}
R4D_IQR_HD constexpr int64_t weight_bytes(const Form& f, int64_t N, int64_t K) {
    return aux_off(f, N, K) + align256(aux_bytes(f, N, K) + 16);
}

/* The scale buffer. kRowF32: the rows' f32 scales, then at cf_off() the blocks' f32 factors. */
R4D_IQR_HD constexpr int64_t s1_bytes(const Form& f, int64_t N, int64_t K) {
    return f.scale == kRowF32 ? N * 4 : N * (K / 128) * (f.scale == kBf16 ? 2 : 1);
}
R4D_IQR_HD constexpr int64_t cf_off(const Form& f, int64_t N, int64_t K) {
    return align256(s1_bytes(f, N, K));
}
R4D_IQR_HD constexpr int64_t s2_off(const Form& f, int64_t N, int64_t K) {
    return align256(s1_bytes(f, N, K));
}
R4D_IQR_HD constexpr int64_t s2_bytes(const Form& f, int64_t N) {
    return f.scale == kE4m3Tensor ? 16 : f.scale == kE4m3Row ? 4 * N : 0;
}
R4D_IQR_HD constexpr int64_t blk_off(const Form& f, int64_t N, int64_t K) {
    return s2_off(f, N, K) + align256(s2_bytes(f, N));
}
R4D_IQR_HD constexpr int64_t scale_bytes(const Form& f, int64_t N, int64_t K) {
    return f.scale == kRowF32 ? cf_off(f, N, K) + align256((K / 128) * 4)
         : blk_off(f, N, K) + (f.lrb ? align256((N >> f.lrb) * (K / 128) * 2) : 0);
}

/* The tile-major index of (row r, 128-K block kb). */
R4D_IQR_HD constexpr int64_t tile_major(int64_t r, int64_t kb, int64_t nkb) {
    return ((r / 16) * nkb + kb) * 16 + r % 16;
}

/* The forms libr4d's kernels are built for, as (kind, B, log2 V, L, log2 rb, scale): one list, so
 * the layout hook refuses at declare a form no kernel reads. */
#define R4D_IQR_FORMS(X)                                                                          \
    /* IQR-V1: scalar tables, vector grids, 6-bit codes under a zero */                           \
    X(kTable, 2, 0, 0, 0, kBf16) X(kTable, 3, 0, 0, 0, kBf16) X(kTable, 4, 0, 0, 0, kBf16)        \
    X(kTable, 5, 0, 0, 0, kBf16)                                                                  \
    X(kGrid, 8, 2, 0, 0, kBf16) X(kGrid, 9, 2, 0, 0, kBf16) X(kGrid, 10, 2, 0, 0, kBf16)          \
    X(kGrid, 12, 2, 0, 0, kBf16)                                                                  \
    X(kZero, 6, 0, 0, 0, kBf16)                                                                   \
    /* IQR-V2: the 4096-state trellis, E4M3 scales under one f32 */                               \
    X(kTrellis, 2, 0, 12, 0, kE4m3Tensor) X(kTrellis, 3, 0, 12, 0, kE4m3Tensor)                   \
    X(kTrellis, 4, 0, 12, 0, kE4m3Tensor) X(kTrellis, 5, 0, 12, 0, kE4m3Tensor)                   \
    X(kTrellis, 6, 0, 12, 0, kE4m3Tensor) X(kTrellis, 7, 0, 12, 0, kE4m3Tensor)                   \
    /* IQR-V3: the 16384-state trellis on the tcq_rb=32 grid */                                   \
    X(kTrellis, 2, 0, 14, 5, kE4m3Tensor) X(kTrellis, 3, 0, 14, 5, kE4m3Tensor)                   \
    X(kTrellis, 4, 0, 14, 5, kE4m3Tensor) X(kTrellis, 5, 0, 14, 5, kE4m3Tensor)                   \
    X(kTrellis, 6, 0, 14, 5, kE4m3Tensor) X(kTrellis, 7, 0, 14, 5, kE4m3Tensor)                   \
    /* the computed 65536-state trellis on the tcq_rb=32 grid */                                  \
    X(kTrellisMad, 2, 0, 16, 5, kE4m3Tensor) X(kTrellisMad, 3, 0, 16, 5, kE4m3Tensor)             \
    X(kTrellisMad, 4, 0, 16, 5, kE4m3Tensor) X(kTrellisMad, 5, 0, 16, 5, kE4m3Tensor)             \
    X(kTrellisMad, 6, 0, 16, 5, kE4m3Tensor) X(kTrellisMad, 7, 0, 16, 5, kE4m3Tensor)             \
    /* the computed trellis with a scale a row (no rescale a value) */                            \
    X(kTrellisMad, 2, 0, 16, 0, kE4m3Tensor) X(kTrellisMad, 3, 0, 16, 0, kE4m3Tensor)             \
    X(kTrellisMad, 4, 0, 16, 0, kE4m3Tensor) X(kTrellisMad, 5, 0, 16, 0, kE4m3Tensor)             \
    X(kTrellisMad, 6, 0, 16, 0, kE4m3Tensor) X(kTrellisMad, 7, 0, 16, 0, kE4m3Tensor)             \
    /* the computed trellis under a bf16 scale a (row, 128): the fold reads it in place */        \
    X(kTrellisMad, 4, 0, 16, 0, kBf16) X(kTrellisMad, 5, 0, 16, 0, kBf16)                         \
    X(kTrellisMad, 6, 0, 16, 0, kBf16) X(kTrellisMad, 7, 0, 16, 0, kBf16)                         \
    /* the computed trellis under one scale a row and a factor a 128-K block */                  \
    X(kTrellisMad, 2, 0, 16, 0, kRowF32) X(kTrellisMad, 3, 0, 16, 0, kRowF32)                     \
    X(kTrellisMad, 4, 0, 16, 0, kRowF32) X(kTrellisMad, 5, 0, 16, 0, kRowF32)                     \
    X(kTrellisMad, 6, 0, 16, 0, kRowF32) X(kTrellisMad, 7, 0, 16, 0, kRowF32)                     \
    X(kTrellisMad, 2, 0, 15, 0, kRowF32)

/* The form word without its run, which the kernels take as an argument. */
R4D_IQR_HD constexpr uint32_t kernel_word(uint32_t w) {
    return word(Form{ form(w).kind, form(w).bits, form(w).lv, form(w).L, form(w).lrb,
                      form(w).scale, 0, form(w).rot });
}
R4D_IQR_HD constexpr bool built(uint32_t w) {
#define R4D_IQR_IS(k, b, lv, l, lrb, sc) \
    kernel_word(w) == word(Form{ k, b, lv, l, lrb, sc, 0 }) ||
    return R4D_IQR_FORMS(R4D_IQR_IS) false;
#undef R4D_IQR_IS
}

/* ---- the embedding table ---------------------------------------------------------------------
 *
 * embed_lookup reads a handful of rows a step, so its table keeps the container's row-major planes
 * as they are (each row's codes from a byte, B bits a code), with the form's table or, for Zero,
 * its zero plane as u8 [rows][K/128], then the scale plane as stored ([rows][K/128] bf16 or E4M3),
 * its second level as f32 and, with rb > 1, the block scale B per [rows/rb][K/128]. The forms
 * built, as R4D_IQR_FORMS plus rot: */
#define R4D_IQR_EMBED_FORMS(X)                                                                    \
    X(kTable, 2, 0, 0, 0, kBf16, 0) X(kTable, 3, 0, 0, 0, kBf16, 0)                              \
    X(kTable, 4, 0, 0, 0, kBf16, 0) X(kTable, 5, 0, 0, 0, kBf16, 0)                              \
    X(kTable, 2, 0, 0, 0, kBf16, 1) X(kTable, 3, 0, 0, 0, kBf16, 1)                              \
    X(kTable, 4, 0, 0, 0, kBf16, 1) X(kTable, 5, 0, 0, 0, kBf16, 1)                              \
    X(kZero, 6, 0, 0, 0, kBf16, 0) X(kZero, 7, 0, 0, 0, kBf16, 0)                                \
    X(kZero, 8, 0, 0, 0, kBf16, 0)                                                               \
    X(kTrellis, 2, 0, 12, 0, kE4m3Tensor, 1) X(kTrellis, 3, 0, 12, 0, kE4m3Tensor, 1)             \
    X(kTrellis, 4, 0, 12, 0, kE4m3Tensor, 1) X(kTrellis, 5, 0, 12, 0, kE4m3Tensor, 1)             \
    X(kTrellis, 6, 0, 12, 0, kE4m3Tensor, 1) X(kTrellis, 7, 0, 12, 0, kE4m3Tensor, 1)

R4D_IQR_HD constexpr bool embed_built(uint32_t w) {
#define R4D_IQR_EIS(k, b, lv, l, lrb, sc, r) \
    kernel_word(w) == word(Form{ k, b, lv, l, lrb, sc, 0, r }) ||
    return R4D_IQR_EMBED_FORMS(R4D_IQR_EIS) false;
#undef R4D_IQR_EIS
}

struct EmbedOff { int64_t codes_row, aux, sc, s2, blk, bytes; };
R4D_IQR_HD constexpr int64_t code_cols(const Form& f, int64_t K) {
    return f.kind == kGrid ? K >> f.lv : K;
}
R4D_IQR_HD constexpr EmbedOff embed_off(const Form& f, int64_t rows, int64_t K) {
    EmbedOff o{};
    o.codes_row = (code_cols(f, K) * f.bits + 7) / 8;
    o.aux = align256(rows * o.codes_row + 16);
    const int64_t aux_b = f.kind == kZero ? rows * (K / 128) : aux_bytes(f, rows, K);
    o.sc = o.aux + align256(aux_b + 16);
    o.s2 = o.sc + align256(rows * (K / 128) * (f.scale == kBf16 ? 2 : 1));
    o.blk = o.s2 + align256(s2_bytes(f, rows));
    o.bytes = o.blk + (f.lrb ? align256((rows >> f.lrb) * (K / 128) * 2) : 0);
    return o;
}

/* ---- the lm_head ------------------------------------------------------------------------------
 *
 * logits_gemm's row form (r4d_fp8_lm.h) at B bits a code: each row its K codes packed as the
 * container stores them, then its bf16 scale per 128 columns, padded to 16 bytes; the table of
 * 2^B int8 entries after the last row, 256-aligned. Table forms of 2 to 6 bits, unrotated. */
R4D_IQR_HD constexpr int64_t lm_row(const Form& f, int64_t K) {
    return (K / 8 * f.bits + 2 * (K / 128) + 15) / 16 * 16;
}
R4D_IQR_HD constexpr int64_t lm_tab(const Form& f, int64_t N, int64_t K) {
    return align256(N * lm_row(f, K));
}
R4D_IQR_HD constexpr int64_t lm_bytes(const Form& f, int64_t N, int64_t K) {
    return lm_tab(f, N, K) + 256;
}
R4D_IQR_HD constexpr bool lm_built(uint32_t w) {
    return form(w).kind == kTable && form(w).bits >= 2 && form(w).bits <= 6 &&
           form(w).scale == kBf16 && !form(w).rot && !form(w).lrb;
}

/* A trellis's codes a state spans. */
R4D_IQR_HD constexpr uint32_t span(const Form& f) { return (f.L + f.bits - 1) / f.bits; }

}  // namespace r4d_iqr
