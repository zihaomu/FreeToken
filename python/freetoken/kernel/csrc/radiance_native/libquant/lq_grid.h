/* lq_grid.h -- a quantisation grid: the codes a weight may take and how its scales sit over it.
 *
 * libquant separates three things a recipe names independently (spec.md §4.5): the grid (code type
 * plus the scale, zero and transform structure around it); the algorithm that picks codes on it
 * (round-to-nearest, GPTQ, AWQ, ...); and the stored encoding, which follows from the grid alone.
 * Every algorithm takes the same grid options.
 *
 * A grid is parsed from a rule's options:
 *
 *   codes=DT        the code dtype: i2..i8, i16, i32 (symmetric), u1..u8, u16 with zero=
 *                   (asymmetric), or a float code: fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2,
 *                   fp4_e2m1. Required, except beside table=, which implies it.
 *   table=T         a scalar codebook: each code indexes a table of values, which a block's scale
 *                   multiplies. nf4 (QLoRA's NormalFloat), iq4nl (llama.cpp's IQ4_NL values),
 *                   w4nl (the rotated expert grid libr4d's w4nla8h kernel decodes), w5nl (its
 *                   32-level form, libr4d's w5nl64a8h),
 *                   binary {-1, +1}, ternary {-1, 0, +1}, or the values themselves, `;`-separated.
 *                   The codes are the narrowest unsigned width that indexes it.
 *   group=N         one scale per N columns of a row -- block [1 x N]
 *   block=RxC       the scale block in general; `*` is the whole extent: 128x128, 1x*, *x*
 *                   (default: one scale a row, 1x*)
 *   scale=DT        the scale dtype: f32, bf16, f16, fp8_e4m3, e8m0 (default bf16); under scale2=
 *                   also an integer, u4..u8, i6, i8 -- a K-quant's sub-block scale
 *   scale2=DT       a second scale level the first is relative to: f32, bf16, f16, fp8_e4m3, e8m0.
 *                   NVFP4 is fp4 codes, an fp8_e4m3 scale per [1 x 16] and an f32 scale2 per tensor.
 *   scale2_value=X  the second level fixed at X for all its blocks instead of derived from the
 *                   largest first-level scale under it; must be exact in scale2=. For E4M3
 *                   sub-scales over a model whose scales fit E4M3's range, only a known offset is
 *                   needed (libr4d's w4nl64a8h experts). A first-level value it would push past its
 *                   dtype's top is clamped there, reported once on stderr.
 *   group2=N        the second level's block, [1 x N]; or
 *   block2=RxC      in general (default: the whole tensor, *x*). A multiple of the first level's
 *                   block in each extent.
 *   zero=DT         an integer zero point a block, making the grid asymmetric: u8, u4 ...
 *   transform=fwhtN rotate each row by the unnormalised Hadamard before quantising
 *   rule=R          how a block's scale is chosen: absmax (the default), mx (the OCP shared
 *                   exponent, the default for an e8m0 scale), fixed (scale_value=, every block),
 *                   search (the scale, or the asymmetric range, of least squared error over a
 *                   sweep of candidates and their least-squares refits, weighted by the imatrix's
 *                   column importances when the converter has them)
 *   scale_value=X   the fixed scale for rule=fixed
 *   clamp=full|sym  a signed code's range: [-2^(b-1), 2^(b-1)-1], or symmetric about zero
 *
 * The rules are exactly those the containers in service were made with (the order of a division,
 * which number is rounded to the scale dtype before codes are chosen, what an all-zero block gets),
 * so a recipe reproduces those containers byte for byte. Each is stated where it is implemented.
 * Forms no container in service uses (a codebook, a second level, rule=search, 16- and 32-bit
 * codes) take a separate general path, which leaves those rules untouched.
 */
#pragma once
#include "rad_quant.h"

#include <string>
#include <vector>

namespace lq {

struct Grid {
    uint32_t codes = RAD_DT_INVALID;
    bool     is_float = false;     /* a float code: fp8, fp6, fp4 */
    bool     asym = false;         /* a zero plane */
    bool     wide = false;         /* a 16- or 32-bit integer code, chosen in double */
    uint32_t scale_dt = RAD_BF16;
    uint32_t zero_dt = RAD_DT_INVALID;
    int64_t  br = 1, bc = 0;       /* the scale block; 0 = the whole extent */
    int64_t  fwht = 0;             /* the transform's order, or 0 */
    int      qmin = 0, qmax = 0;   /* an integer code's range */
    float    fmax = 0.0f;          /* a float code's largest magnitude */
    int      emax = 0;             /* a float code's largest normal exponent, for rule=mx */
    enum Rule { ABSMAX, MX, FIXED, SEARCH } rule = ABSMAX;
    double   fixed = 0.0;

    /* A scalar codebook: the value of each code, in code order. */
    std::vector<float> table;
    uint32_t table_dt = RAD_DT_INVALID;
    std::vector<int>   table_order;  /* the codes, by ascending value */
    float    tpeak = 0.0f;           /* the entry of largest magnitude, signed */
    /* The extremes are not mirror images (iq4nl: -127 and +113), so a block's scale takes the
     * sign of its largest-magnitude value and that value lands on `tpeak` exactly. */
    bool     tsigned = false;

    /* A second scale level, which the first is relative to. */
    bool     lv2 = false;
    uint32_t scale2_dt = RAD_DT_INVALID;
    int64_t  br2 = 0, bc2 = 0;       /* its block; 0 = the whole extent */
    double   fixed2 = 0.0;           /* scale2_value: every block's stored value, or 0 */
    bool     l1_int = false;         /* the first level is an integer sub-scale */
    int      l1_lo = 0, l1_hi = 0;   /* its range */
    float    l1max = 0.0f;           /* the largest first-level value its dtype holds */

    /* A vector codebook (vq=FILE): one code per `vdim` consecutive columns, indexing `vq`
     * (`ventries` rows of `vdim` integer values in -127..127), stored as the encoding's `grid`
     * table. A block's scale matches its rms to the codebook's: a value v is the entry over
     * `vq_unit`, the units a unit-rms vector the codebook was fitted to spans. */
    std::vector<float> vq;
    int64_t  vdim = 0, ventries = 0;
    double   vq_unit = 1.0;
    /* The search's layout: vq_soa[j * ventries + k] = entry k's j-th value, vq_n2[k] its squared
     * norm, so the distances to every entry vectorise across entries. */
    std::vector<float> vq_soa, vq_n2;

    /* A trellis (tcq=L): the codes of a column, one per row, are the bits a bitshift trellis
     * shifts in. Row r's value is tcq_lut[state_r] times its block's scale, where state_r =
     * (state_{r-1} << k | code_r) mod 2^L, k the code's bits. Each run of tcq_rows rows is
     * tail-biting: the state before its first row is the one after its last, which its last
     * ceil(L/k) codes make. A block's scale matches its rms to the table's, as a vector grid's
     * does (vq_unit). */
    int      tcq_L = 0, tcq_k = 0;
    int64_t  tcq_rows = 0;
    std::vector<float> tcq_lut;           /* 2^L integer values in -127..127 */
    float    tcq_peak = 0.0f;             /* tcq_peak=1: the table's largest magnitude, which no
                                             block's largest value may pass; else 0 */
    /* tcq_rb=R: the values on the int8 grid an int8 GEMM reads with one scale per R rows (and the
     * block's 128 columns), the largest of the R rows' scales: row r's value is
     * B * rint(lut[state] * (s_r / B)), B that largest. The row holding it keeps its values; the
     * others' are rounded onto B's grid, inside the trellis's search. 0: off. */
    int64_t  tcq_rb = 0;
    /* tcq_colf=1 (one scale a row, block=1x*): each 128-column block's values also take a factor
     * shared by every row, its share of the weight's energy, so a value's scale is its row's times
     * its block's. A GEMM folds the factor into the activation's per-128 scale, which it multiplies
     * anyway, and the row's in its epilogue. Chosen from the weight before GPTQ. */
    bool     tcq_colf = false;
    /* tcq_code=mad: the states' values are computed, not chosen: with h = (st * A + B) mod 2^32
     * (the product of the factors' low 24 bits), state st's value is bytesum(h & 0x3F3F3F3F) - 126
     * (tcq_lut_mad). A reader needs no table: a multiply-add, a mask and a byte-sum dot a value. The
     * table is still stored, A and B beside it ("trellis.mad"). */
    bool     tcq_mad = false;
    uint32_t tcq_a = 0, tcq_b = 0;
    /* tcq_code=mad16: the same hash with every byte whole, bytesum(h) - 510 (sigma ~148, where mad's
     * is ~37): values a 16-bit reader takes exactly (f16 holds integers to 2048), so the code is not
     * held to the int8 grid's ~42 dB floor. No int8 encoding can carry it; gptqsim measures it. */
    bool     tcq_wide = false;

    /* The planes of the encoding, by index; -1 when the grid has none. */
    int plane_zero() const { return asym ? 2 : -1; }
    int plane_grid() const {
        return vq.empty() && !tcq_L ? -1 : 2 + (asym ? 1 : 0) + (lv2 ? 1 : 0);
    }
    int plane_scale2() const { return lv2 ? (asym ? 3 : 2) : -1; }
    int plane_mad() const { return tcq_mad ? plane_grid() + 1 : -1; }
    /* A trellis under one scale a row (block=1x*) carries the 128-column blocks' factors beside it,
     * tcq_colf's or ones, so a reader has one form for both (libr4d's kRowF32). */
    bool row_scaled() const { return tcq_L && br == 1 && bc == 0 && !lv2; }
    int plane_colf() const { return row_scaled() ? plane_grid() + (tcq_mad ? 2 : 1) : -1; }
    int plane_table() const {
        return table.empty() ? -1 : 2 + (asym ? 1 : 0) + (lv2 ? 1 : 0);
    }
    /* Whether this grid takes the general path (see the file header). */
    bool general() const { return lv2 || rule == SEARCH || !table.empty() || wide; }
};

/* Parse a grid from options. RAD_OK, or RAD_E_INVAL with `why` naming the option. */
int grid_parse(const RadParam* o, int n_o, Grid* g, std::string* why);

/* The encoding a grid writes. */
RadEncoding grid_encoding(const Grid& g);

/* The rows one quantize call may take for this grid: a multiple of every level's block rows, or 0
 * when a scale covers every row and has to see them all. */
int64_t grid_row_block(const Grid& g);

/* One block's quantiser state: the scale the codes are chosen against, its stored form, the
 * reciprocal, and for an asymmetric grid the zero. */
struct BlockScale {
    float   s = 1.0f;        /* the value codes are chosen against: the stored scale(s), widened */
    float   inv = 1.0f;
    double  inv_d = 1.0;     /* rule=fixed reads its reciprocal in double; see grid.cpp */
    int     zero = 0;
    int     e8 = 0;          /* rule=mx: the shared exponent's E8M0 code */
    float   stored = 1.0f;   /* what the scale plane holds: s divided by a transform's order, or
                                under a second level the first level's own value */
    float   lo = 0.0f;       /* an asymmetric block's low end, which the zero is taken against */
};

/* The scale of the block covering logical rows [r0, r1) and columns [c0, c1) of `w`, f32 [*, cols],
 * already transformed. `imp` is a column's importance for rule=search, or null. Under a second
 * level this is the first level's ideal scale, unrounded; grid_scales finishes it. */
BlockScale grid_block_scale(const Grid& g, const float* w, int64_t cols, int64_t r0, int64_t r1,
                            int64_t c0, int64_t c1, const float* imp = nullptr);

/* One value onto the grid: its code, and the value the code decodes to (for error feedback). */
int   grid_code(const Grid& g, const BlockScale& b, float v);
/* A vector grid's code for the `vdim` values at `v`: the entry nearest v times b.inv. */
int   grid_vq_code(const Grid& g, const BlockScale& b, const float* v);
float grid_decode(const Grid& g, const BlockScale& b, int code);

/* tcq_code=mad's value of every one of the 2^L states (Grid::tcq_mad); with `wide`, mad16's
 * (Grid::tcq_wide). */
std::vector<float> tcq_lut_mad(int L, uint32_t a, uint32_t b, bool wide = false);

/* A trellis grid's search state for one run, reused across runs. */
struct TcqWork {
    std::vector<float>   cost, next, best, rx, rs, rr;
    std::vector<uint8_t> back;
    std::vector<int64_t> path;
};
/* A trellis grid's codes for a run of n <= tcq_rows rows of one column: a tail-biting path (the
 * state before the first row is the state after the last) of least sum over the rows of
 * (x_r - v_r)^2, v_r = s_r * lut[state_r], its codes into `codes` and the values they decode to
 * into `dq`. With `ratio` (tcq_rb), v_r = s_r * rint(lut[state_r] * ratio_r) instead: s_r is then
 * the row block's scale and ratio_r the row's own scale over it. */
void grid_tcq_run(const Grid& g, const float* x, const float* s, int64_t n, int32_t* codes,
                  float* dq, TcqWork& w, const float* ratio = nullptr);

/* A trellis grid's two-level scales are read as one bf16 a block: the bf16 of the first level
 * (b->stored) times the second's stored value s2, which is what libr4d widens them to for its
 * int8 GEMMs. Settle b's scale to that. */
void grid_tcq_settle(const Grid& g, float s2, BlockScale* b);
/* A trellis block's two-level scale from its ideal (unrounded) scale under the second level's
 * stored s2, settled: dyn=1's choice when GPTQ reaches the block. */
BlockScale grid_tcq_block_lv2(const Grid& g, double ideal, float s2);

/* Rows [row0, row0 + rows) of a "trellis" weight of rows_all x cols back to f32 in its logical
 * basis (the transform undone), from the plane starts: abi/rad_quant.h's decode hook. Whole runs
 * only, but for the weight's last. RAD_E_UNSUPPORTED for any other encoding. */
int grid_tcq_decode(const RadEncoding* e, int64_t rows_all, int64_t cols, const void* const* planes,
                    int64_t row0, int64_t rows, float* out);

/* Write a scale (and a zero) into element `i` of its plane row. */
void grid_store_scale(const Grid& g, const BlockScale& b, void* scale_row, void* zero_row,
                      int64_t i);

/* The geometry of a quantize call: where each plane's rows for this call start, as the ABI hands
 * them over, and the plane row widths in bytes. */
struct Planes {
    uint8_t* codes = nullptr;
    uint8_t* scale = nullptr;
    uint8_t* zero = nullptr;
    uint8_t* scale2 = nullptr;
    uint8_t* table = nullptr;
    uint8_t* grid = nullptr;
    uint8_t* mad = nullptr;
    uint8_t* colf = nullptr;
    int64_t  colf_cols = 0;
    int64_t  codes_row = 0, scale_row = 0, zero_row = 0, scale2_row = 0;   /* bytes */
    int64_t  scale_cols = 0, scale2_cols = 0;                             /* elements a row */
};
Planes grid_planes(const Grid& g, int64_t cols, void* const* planes);

/* Rotate rows in place when the grid carries a transform. */
void grid_transform_rows(const Grid& g, float* w, int64_t rows, int64_t cols);

/* Every scale of a call's rows, both levels: the first level block by block in plane order, the
 * second level's stored values in theirs. */
struct Scales {
    std::vector<BlockScale> b;
    int64_t nr = 0, nc = 0;            /* first-level blocks */
    int64_t br = 1, bc = 1;            /* their extents, resolved against the call */
    std::vector<float> s2;             /* the second level as stored */
    int64_t nr2 = 0, nc2 = 0;
    std::vector<float> colf;           /* tcq_colf: each 128-column block's factor, all rows' */
    const BlockScale& at(int64_t r, int64_t c) const {
        return b[(size_t)((r / br) * nc + c / bc)];
    }
};

/* The whole scale grid of rows [0, rows) of `w`, without choosing codes: what round-to-nearest and
 * the static-groups form of an error-feedback algorithm quantise against. `imp` is the columns'
 * importances for rule=search, or null. */
Scales grid_scales(const Grid& g, const float* w, int64_t rows, int64_t cols,
                   const float* imp = nullptr);

/* Write every scale plane, the zeros and the codebook of a call. */
void grid_store_scales(const Grid& g, const Scales& s, const Planes& p);

/* Store one code into element `i` of a code row, any width this grid writes. */
void grid_store_code(const Grid& g, uint8_t* row, int64_t i, int code);

/* Quantise and decode rows [0, rows) of `w` into `out` [rows, cols]: the weight a reader would see,
 * for algorithms that measure a candidate. `rows` covers whole blocks. */
void grid_fake(const Grid& g, const float* w, int64_t rows, int64_t cols, float* out,
               const float* imp = nullptr);

/* A block's state from an unrounded scale `s` and the asymmetric low end `lo` the zero is taken
 * against, rounded as the general path rounds. For algorithms that choose the scale themselves;
 * not for grids with a second level, which have their own pass. */
BlockScale grid_finish(const Grid& g, double s, float lo);

/* Round-to-nearest over rows [0, rows) of `w` (logical rows [row0, row0+rows)). The block scales
 * are taken over each block's own rows, so `rows` must cover whole blocks. */
int grid_rtn(const Grid& g, const float* w, int64_t row0, int64_t rows, int64_t cols,
             const Planes& p, const float* imp = nullptr);

}  /* namespace lq */
