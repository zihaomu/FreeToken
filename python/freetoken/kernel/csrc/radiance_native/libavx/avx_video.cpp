/* avx_video.cpp -- the video generator's ops around its decoder (docs/OPS.md, "Video
 * generation"): the latent's patch rows unfolded to voxels, the tiles a video decoder runs over,
 * gathered from the latent and stitched back into frames, and the rotary positions of a tile's
 * tokens; the encoder's plan, cut and blend for a picture too big for one volume (docs/OPS.md,
 * "Video generation"); and its latent upscaler's (docs/OPS.md, "The latent upscaler"): the 3-D
 * convolution,
 * the group norm, the temporal depthwise convolution, the per-frame resize, the fold back into
 * patch rows and the re-noising.
 *
 * Bit-exact with libref (libref/ref_video.cpp) but for conv3d_ndhwc and group_norm: they are
 * copies, roundings and short sums in libref's order, and a seed must give the same video bytes
 * on both host libraries. Where libref multiplies and then adds in two roundings, `keep` stops the
 * compiler from fusing the pair (this plugin is built with -ffp-contract=fast, libref without
 * contraction); the stitch's bf16 blends are passes of whole rows through x's dtype
 * (`narrow_row`), so a product is rounded before the sum reads it. conv3d_ndhwc is GEMMs over
 * patch rows and group_norm sums its statistics over fixed row blocks; both are held to the
 * tolerance table.
 */
#include "avx_vec.h"
#include "avx_common.h"
#include "rad_philox.h"

#include <algorithm>
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

inline int64_t at(const RadTensor* t, int64_t i) { return ldi(t, offlin(t, i)); }

/* v[0, n) rounded through dtype `dt` in place, as a store and reload rounds it. `tmp` holds n
 * floats. */
inline void narrow_row(uint32_t dt, float* v, int64_t n, float* tmp) {
    if (dt == RAD_F32) return;
    row_from_f32(dt, v, tmp, n);
    row_to_f32(dt, tmp, v, n);
}

/* A value the compiler may not fuse into the next add. At the scalar level a vector is a float. */
#if AVX_LEVEL == 0
inline float keep(float x) { __asm__("" : "+x"(x)); return x; }
#elif AVX_LEVEL == 3
inline float keep(float x) { __asm__("" : "+v"(x)); return x; }
inline vf keep(vf x) { __asm__("" : "+v"(x)); return x; }
#else
inline float keep(float x) { __asm__("" : "+x"(x)); return x; }
inline vf keep(vf x) { __asm__("" : "+x"(x)); return x; }
#endif

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

/* Elements [e0, e0 + cnt) of the normal stream (seed, stream) into out (abi/rad_philox.h): the same
 * arithmetic as rad_philox_normal_s, with one Philox block and one log per element pair instead of
 * one each per element (avx_audio.cpp's). */
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

/* The f32 values of a space-separated list (libref's parse_f32_list): up to `cap` into out, the
 * count returned whatever it is. */
int64_t parse_f32(const char* s, float* out, int64_t cap) {
    int64_t k = 0;
    while (s && *s) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end == s) break;
        if (k < cap) out[k] = (float)v;
        ++k;
        s = end;
    }
    return k;
}

/* libref's split_tiles: the tile starts and overlaps over `len` latent units. */
bool split_tiles(int64_t len, int64_t tile, int64_t overlap, std::vector<int64_t>* start,
                 std::vector<int64_t>* ov) {
    start->clear();
    ov->clear();
    if (len <= 0 || tile <= 0 || overlap < 0 || overlap >= tile) return false;
    if (tile >= len) {
        start->push_back(0);
        return len == tile;
    }
    int64_t nt = (len + tile - 1) / tile;
    while (tile * nt - overlap * (nt - 1) - len < 0) ++nt;
    ov->assign((size_t)(nt - 1), overlap);
    const int64_t rem = tile * nt - overlap * (nt - 1) - len;
    for (int64_t i = 0; i < rem; ++i) (*ov)[(size_t)(i % (nt - 1))] += 1;
    start->push_back(0);
    for (int64_t i = 0; i + 1 < nt; ++i) start->push_back(start->back() + tile - (*ov)[(size_t)i]);
    return true;
}

/* libref's split_tiles_axis: the decoder's split, except that an axis no longer than a tile is one
 * tile, short if need be. */
bool split_tiles_axis(int64_t len, int64_t tile, int64_t overlap, std::vector<int64_t>* start,
                      std::vector<int64_t>* ov) {
    if (tile >= len) {
        start->clear();
        ov->clear();
        start->push_back(0);
        return len > 0 && tile > 0 && overlap >= 0 && overlap < tile;
    }
    return split_tiles(len, tile, overlap, start, ov);
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

/* out[e] = rd(rd(a[e] wa) + rd(b[e] wb)) over e < cnt, pixels of three values with a weight pair
 * each (`ws` 1) or one pair for every pixel (`ws` 0), rd a pass through `dt`. `out` may be `b`;
 * `p2` and `tmp` hold cnt floats. */
void blend_rgb(const float* a, const float* wa, const float* b, const float* wb, int64_t ws,
               int64_t cnt, float* out, float* p2, float* tmp, uint32_t dt) {
    for (int64_t e = 0; e < cnt; ++e) {
        p2[e] = b[e] * wb[(e / 3) * ws];
        out[e] = a[e] * wa[(e / 3) * ws];
    }
    narrow_row(dt, out, cnt, tmp);
    narrow_row(dt, p2, cnt, tmp);
    for (int64_t e = 0; e < cnt; ++e) out[e] += p2[e];
    narrow_row(dt, out, cnt, tmp);
}

}  /* namespace */

/* ------------------------------------------------------------------ patch_unfold */
/* Patch rows to denormalised voxel rows, f32(f32(x * mul[c]) + add[c]) (libref): a task is a
 * patch, its row read once and dealt into its pt ph pw voxel rows. */
AVX_KERNEL(patch_unfold) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *Y = &a->t[2];
    const int64_t C = p_int(a, "C", 0), pt = p_int(a, "pt", 0);
    const int64_t ph = p_int(a, "ph", 0), pw = p_int(a, "pw", 0);
    const char* ms = p_str(a, "mul", ""), *as = p_str(a, "add", "");
    /* Each number takes at least two characters but the last, so this bounds the count. */
    const int64_t cap = (int64_t)((std::strlen(ms) + std::strlen(as)) / 2 + 2);
    float* lists = (float*)scratch_raw((size_t)(2 * cap + 16) * sizeof(float));
    float *mul = lists, *add = lists + cap;
    if (C <= 0 || pt <= 0 || ph <= 0 || pw <= 0 || parse_f32(ms, mul, cap) != C ||
        parse_f32(as, add, cap) != C || numel(G) < 3)
        return RAD_E_SHAPE;
    const int64_t F = at(G, 0), Hp = at(G, 1), Wp = at(G, 2);
    if (F <= 0 || Hp <= 0 || Wp <= 0) return RAD_E_SHAPE;
    const int64_t vol = pt * ph * pw, Hv = Hp * ph, Wv = Wp * pw, K = C * vol;
    if (rows_of(Y, C) < F * pt * Hv * Wv || rows_of(X, K) < F * Hp * Wp) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(F * Hp * Wp * K >= AVX_PAR_MIN)
    for (int64_t pi = 0; pi < F * Hp * Wp; ++pi) {
        float* buf = scratch_f32((size_t)(K + C + 32));
        float *xs = buf, *row = buf + K + 16;
        const float* xr = in_row(X, rowoff(X, pi, K), K, xs);
        const int64_t f = pi / (Hp * Wp), i = (pi / Wp) % Hp, j = pi % Wp;
        for (int64_t sub = 0; sub < vol; ++sub) {
            const int64_t ta = sub / (ph * pw), yb = (sub / pw) % ph, xd = sub % pw;
            for (int64_t c = 0; c < C; ++c) row[c] = keep(xr[c * vol + sub] * mul[c]) + add[c];
            const int64_t r = ((f * pt + ta) * Hv + i * ph + yb) * Wv + j * pw + xd;
            out_row(Y, rowoff(Y, r, C), C, row);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ patch_fold */
/* Voxel rows folded into normalised patch rows, f32(f32(x * mul[c]) + add[c]) (libref): a task
 * is a patch, its pt ph pw voxel rows read once each and dealt into its row. */
AVX_KERNEL(patch_fold) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *Y = &a->t[2];
    const int64_t C = p_int(a, "C", 0), pt = p_int(a, "pt", 0);
    const int64_t ph = p_int(a, "ph", 0), pw = p_int(a, "pw", 0);
    const char* ms = p_str(a, "mul", ""), *as = p_str(a, "add", "");
    /* Each number takes at least two characters but the last, so this bounds the count. */
    const int64_t cap = (int64_t)((std::strlen(ms) + std::strlen(as)) / 2 + 2);
    float* lists = (float*)scratch_raw((size_t)(2 * cap + 16) * sizeof(float));
    float *mul = lists, *add = lists + cap;
    if (C <= 0 || pt <= 0 || ph <= 0 || pw <= 0 || parse_f32(ms, mul, cap) != C ||
        parse_f32(as, add, cap) != C || numel(G) < 3)
        return RAD_E_SHAPE;
    const int64_t F = at(G, 0), Hp = at(G, 1), Wp = at(G, 2);
    if (F <= 0 || Hp <= 0 || Wp <= 0) return RAD_E_SHAPE;
    const int64_t vol = pt * ph * pw, Hv = Hp * ph, Wv = Wp * pw, K = C * vol;
    const int64_t rows = F * Hp * Wp;
    if (rows_of(Y, K) < rows || rows_of(X, C) < F * pt * Hv * Wv) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * K >= AVX_PAR_MIN)
    for (int64_t pi = 0; pi < rows; ++pi) {
        float* buf = scratch_f32((size_t)(K + C + 32));
        float *row = buf, *xs = buf + K + 16;
        const int64_t f = pi / (Hp * Wp), i = (pi / Wp) % Hp, j = pi % Wp;
        for (int64_t sub = 0; sub < vol; ++sub) {
            const int64_t ta = sub / (ph * pw), tb = (sub / pw) % ph, td = sub % pw;
            const int64_t r = ((f * pt + ta) * Hv + i * ph + tb) * Wv + j * pw + td;
            const float* xr = in_row(X, rowoff(X, r, C), C, xs);
            for (int64_t c = 0; c < C; ++c) row[c * vol + sub] = keep(xr[c] * mul[c]) + add[c];
        }
        out_row(Y, rowoff(Y, pi, K), K, row);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ grid_pos3 */
/* A tile's rotary positions, f32(scale) g(c, n), g(c, n) = f32(f32(2 f32((c + 0.5) / n)) - 1)
 * (libref): one value an axis index, taken once a launch. */
AVX_KERNEL(grid_pos3) {
    if (!rad_args_have(a, 1)) return RAD_E_INVAL;
    const RadTensor* Y = &a->t[0];
    const int64_t T = p_int(a, "T", 0), H = p_int(a, "H", 0), W = p_int(a, "W", 0);
    const int64_t period = p_int(a, "period", 0);
    const float scale = p_f32(a, "scale", 1.0f);
    if (T <= 0 || H <= 0 || W <= 0 || period < T * H * W || Y->rank != 2 || Y->shape[0] < 3)
        return RAD_E_SHAPE;
    const int64_t M = Y->shape[1], rs = Y->stride[0], cs = Y->stride[1];
    if (M <= 0) return RAD_E_SHAPE;
    float* g = (float*)scratch_raw((size_t)(T + H + W + 16) * sizeof(float));
    auto axis = [scale](int64_t c, int64_t n) {
        const float q = ((float)c + 0.5f) / (float)n;
        return scale * (keep(2.0f * q) - 1.0f);
    };
    for (int64_t c = 0; c < T; ++c) g[c] = axis(c, T);
    for (int64_t c = 0; c < H; ++c) g[T + c] = axis(c, H);
    for (int64_t c = 0; c < W; ++c) g[T + H + c] = axis(c, W);
    AVX_PARALLEL_FOR_IF(M * 3 >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t k = m % period;
        const bool tok = k < T * H * W;
        stt(Y, m * cs, tok ? g[k / (H * W)] : 0.0f);
        stt(Y, rs + m * cs, tok ? g[T + (k / W) % H] : 0.0f);
        stt(Y, 2 * rs + m * cs, tok ? g[T + H + k % W] : 0.0f);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vtile_gather */
/* A video decoder's tiles out of the latent's voxel rows, then the register rows and zeros
 * (libref): one copied row a y row. */
AVX_KERNEL(vtile_gather) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *R = &a->t[1], *G = &a->t[2], *Y = &a->t[3];
    const int64_t n = p_int(a, "n", 0), tile = p_int(a, "tile", 0);
    const int64_t overlap = p_int(a, "overlap", 0), frames = p_int(a, "frames", 0);
    const int64_t period = p_int(a, "period", 0), n_reg = p_int(a, "n_reg", 0);
    const int64_t vox = frames * tile * tile;
    if (n <= 0 || tile <= 0 || frames <= 0 || n_reg < 0 || period < vox + n_reg ||
        numel(G) < 3 || rows_of(R, n) < n_reg)
        return RAD_E_SHAPE;
    const int64_t H = at(G, 1), W = at(G, 2);
    std::vector<int64_t> sy, sx, oy, ox;
    if (at(G, 0) != frames || !split_tiles(H, tile, overlap, &sy, &oy) ||
        !split_tiles(W, tile, overlap, &sx, &ox))
        return RAD_E_SHAPE;
    const int64_t nw = (int64_t)sx.size(), rows = (int64_t)sy.size() * nw * period;
    if (rows_of(Y, n) < rows || rows_of(X, n) < frames * H * W) return RAD_E_SHAPE;
    const int64_t* py = sy.data();
    const int64_t* px = sx.data();

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t row = 0; row < rows; ++row) {
        float* scr = scratch_f32((size_t)n + 16);
        const int64_t k = row / period, p = row % period, yb = rowoff(Y, row, n);
        if (p < vox) {
            const int64_t t = p / (tile * tile), ty = (p / tile) % tile, tx = p % tile;
            const int64_t src = (t * H + py[k / nw] + ty) * W + px[k % nw] + tx;
            copy_row(X, rowoff(X, src, n), Y, yb, n, scr);
        } else if (p < vox + n_reg) {
            copy_row(R, rowoff(R, p - vox, n), Y, yb, n, scr);
        } else {
            for (int64_t c = 0; c < n; ++c) scr[c] = 0.0f;
            out_row(Y, yb, n, scr);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vtile_stitch */
/* A decoded chunk's tiles stitched into frames, the chunk overlap blended and kept, RGB bytes out
 * (libref's rule and roundings). A task is one pixel row of one frame: each tile column's line of
 * the owner tile is read out of x, blended with the tile above's line and then the left tile's
 * (both as decoded), a whole line at a time, every product and sum a pass through x's dtype. The
 * kept frames are made first, reading `prev`; the frames that refill it after, in a second pass,
 * so no task reads a row another has rewritten. */
AVX_KERNEL(vtile_stitch) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *PV = &a->t[2], *Y = &a->t[3];
    const int64_t tile = p_int(a, "tile", 0), overlap = p_int(a, "overlap", 0);
    const int64_t S = p_int(a, "scale", 0), pt = p_int(a, "pt", 0), skip = p_int(a, "skip", 0);
    const int64_t keep_n = p_int(a, "keep", 0), blend = p_int(a, "blend", 0);
    const int64_t period = p_int(a, "period", 0);
    float mean[3], stdv[3];
    if (tile <= 0 || S <= 0 || pt <= 0 || skip < 0 || keep_n <= 0 || blend < 0 ||
        blend > keep_n || parse_f32(p_str(a, "mean", ""), mean, 3) != 3 ||
        parse_f32(p_str(a, "std", ""), stdv, 3) != 3 || numel(G) < 5 || Y->dtype != RAD_U8)
        return RAD_E_SHAPE;
    const int64_t frames = at(G, 0), H = at(G, 1), W = at(G, 2);
    const bool first = at(G, 3) != 0, last = at(G, 4) != 0;
    std::vector<int64_t> sy, sx, oy, ox;
    if (frames <= 0 || 2 * skip + keep_n + blend > frames * pt || period < frames * tile * tile ||
        !split_tiles(H, tile, overlap, &sy, &oy) || !split_tiles(W, tile, overlap, &sx, &ox))
        return RAD_E_SHAPE;
    const int64_t nh = (int64_t)sy.size(), nw = (int64_t)sx.size(), cols = 3 * pt * S * S;
    if (rows_of(X, cols) < nh * nw * period) return RAD_E_SHAPE;
    const int64_t Hp = H * S, Wp = W * S, npx = Hp * Wp;
    const int64_t fout = keep_n + (last ? blend : 0);
    if (rows_of(Y, 3) < fout * npx || (blend > 0 && rows_of(PV, 3) < blend * npx))
        return RAD_E_SHAPE;
    std::vector<int64_t> own_y((size_t)Hp);
    for (int64_t Yp = 0, i = 0; Yp < Hp; ++Yp) {
        while (i + 1 < nh && Yp >= sy[(size_t)(i + 1)] * S) ++i;
        own_y[(size_t)Yp] = i;
    }
    const uint32_t dt = X->dtype;
    const int64_t TW = tile * S, RW = Wp * 3, TW3 = TW * 3;
    const int64_t *psy = sy.data(), *psx = sx.data(), *poy = oy.data(), *pox = ox.data();
    const int64_t* pown = own_y.data();
    auto wb = [dt](int64_t p, int64_t e) { return narrow1(dt, (float)p / (float)e); };
    auto wa = [dt](int64_t p, int64_t e) {
        return narrow1(dt, 1.0f - narrow1(dt, (float)p / (float)e));
    };

    /* Tile (i, j)'s decoded line at row ly of clip frame fc: out[lx * 3 + c], lx < TW. */
    auto tile_line = [&](int64_t i, int64_t j, int64_t ly, int64_t fc, float* out) {
        const int64_t base = (i * nw + j) * period + ((fc / pt) * tile + ly / S) * tile;
        const int64_t xs = laststride(X);
        for (int64_t tx = 0; tx < tile; ++tx) {
            const int64_t xb = rowoff(X, base + tx, cols);
            for (int64_t c = 0; c < 3; ++c) {
                const int64_t col0 = ((c * pt + fc % pt) * S + ly % S) * S;
                for (int64_t d = 0; d < S; ++d)
                    out[(tx * S + d) * 3 + c] = ldt(X, xb + (col0 + d) * xs);
            }
        }
    };
    /* The stitched pixel row Yp of clip frame fc into row[0, RW). `l0`, `l1` hold TW3 floats,
     * `w` 2 TW, `p2` and `tmp` RW. */
    auto stitch_row = [&](int64_t fc, int64_t Yp, float* row, float* l0, float* l1, float* w,
                          float* p2, float* tmp) {
        const int64_t i = pown[Yp], ly = Yp - psy[i] * S;
        const int64_t ev = i > 0 ? poy[i - 1] * S : 0;
        const float wv[2] = { ly < ev ? wa(ly, ev) : 0.0f, ly < ev ? wb(ly, ev) : 0.0f };
        for (int64_t j = 0; j < nw; ++j) {
            const int64_t x0 = psx[j] * S, x1 = j + 1 < nw ? psx[j + 1] * S : Wp;
            tile_line(i, j, ly, fc, l0);
            if (ly < ev) {
                tile_line(i - 1, j, ly + TW - ev, fc, l1);
                blend_rgb(l1, &wv[0], l0, &wv[1], 0, TW3, l0, p2, tmp, dt);
            }
            const int64_t eh = j > 0 ? std::min(pox[j - 1] * S, x1 - x0) : 0;
            if (eh > 0) {
                const int64_t full = pox[j - 1] * S;
                tile_line(i, j - 1, ly, fc, l1);
                for (int64_t lx = 0; lx < eh; ++lx) {
                    w[lx] = wa(lx, full);
                    w[TW + lx] = wb(lx, full);
                }
                blend_rgb(l1 + (TW - full) * 3, w, l0, w + TW, 1, eh * 3, l0, p2, tmp, dt);
            }
            std::memcpy(row + x0 * 3, l0, (size_t)((x1 - x0) * 3) * sizeof(float));
        }
    };
    /* Frame f's pixel row Yp as bytes. */
    auto bytes = [&](int64_t f, int64_t Yp, const float* row) {
        for (int64_t X0 = 0; X0 < Wp; ++X0) {
            const int64_t yb = rowoff(Y, f * npx + Yp * Wp + X0, 3);
            for (int64_t c = 0; c < 3; ++c) {
                float p = keep(row[X0 * 3 + c] * stdv[c]) + mean[c];
                p = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
                stt(Y, yb + c * laststride(Y), p * 255.0f);
            }
        }
    };
    const size_t need = (size_t)(3 * RW + 2 * TW3 + 2 * TW + 96);

    AVX_PARALLEL_FOR_IF(keep_n * npx * 3 * 8 >= AVX_PAR_MIN)
    for (int64_t task = 0; task < keep_n * Hp; ++task) {
        const int64_t k = task / Hp, Yp = task % Hp;
        float* buf = scratch_f32(need);
        float *row = buf, *p2 = row + RW + 16, *tmp = p2 + RW + 16, *l0 = tmp + RW + 16;
        float *l1 = l0 + TW3 + 16, *w = l1 + TW3 + 16;
        stitch_row(skip + k, Yp, row, l0, l1, w, p2, tmp);
        if (!first && k < blend) {
            float* pr = l0;   /* the overlap row, a line's worth at a time */
            const float pw[2] = { wa(k, blend), wb(k, blend) };
            for (int64_t x0 = 0; x0 < Wp; x0 += TW) {
                const int64_t cnt = std::min(TW, Wp - x0);
                for (int64_t e = 0; e < cnt * 3; ++e) {
                    const int64_t pxi = k * npx + Yp * Wp + x0 + e / 3;
                    pr[e] = ldt(PV, rowoff(PV, pxi, 3) + (e % 3) * laststride(PV));
                }
                blend_rgb(pr, &pw[0], row + x0 * 3, &pw[1], 0, cnt * 3, row + x0 * 3, p2, tmp, dt);
            }
        }
        bytes(k, Yp, row);
    }

    AVX_PARALLEL_FOR_IF(blend * npx * 3 * 8 >= AVX_PAR_MIN)
    for (int64_t task = 0; task < blend * Hp; ++task) {
        const int64_t k = task / Hp, Yp = task % Hp;
        float* buf = scratch_f32(need);
        float *row = buf, *p2 = row + RW + 16, *tmp = p2 + RW + 16, *l0 = tmp + RW + 16;
        float *l1 = l0 + TW3 + 16, *w = l1 + TW3 + 16;
        stitch_row(2 * skip + keep_n + k, Yp, row, l0, l1, w, p2, tmp);
        if (last) bytes(keep_n + k, Yp, row);
        for (int64_t X0 = 0; X0 < Wp; ++X0) {
            const int64_t pb = rowoff(PV, k * npx + Yp * Wp + X0, 3);
            for (int64_t c = 0; c < 3; ++c) stt(PV, pb + c * laststride(PV), row[X0 * 3 + c]);
        }
    }
    return RAD_OK;
}

/* ================================================================== the tiled encoder
 * The plan, the cut and the blend of vae.py's `tiled_encode` (libref's section comment). */

/* ------------------------------------------------------------------ vtile_plan */
/* The plan one frame's encode runs over (libref): geom = (clip, H, W) the frame's extent in
 * pixels, split by split_tiles_axis over the spatial axes in latent units; row t of y (i32
 * [M, 22]) the tile's first pixel, its pixel extent, and the six level volumes. */
/* The tile count comes from G; the ranged `M` is a ceiling only (band_rows). */
AVX_KERNEL(vtile_plan) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *G = &a->t[0], *Y = &a->t[1];
    const int64_t band = p_int(a, "M", 0), tile = p_int(a, "tile", 0);
    const int64_t overlap = p_int(a, "overlap", 0), ratio = p_int(a, "ratio", 0);
    if (tile <= 0 || overlap < 0 || overlap >= tile || ratio <= 0 || numel(G) < 3)
        return RAD_E_SHAPE;
    const int64_t H = at(G, 1), W = at(G, 2);
    std::vector<int64_t> sy, sx, oy, ox;
    if (H <= 0 || W <= 0 || H % ratio || W % ratio ||
        !split_tiles_axis(H / ratio, tile, overlap, &sy, &oy) ||
        !split_tiles_axis(W / ratio, tile, overlap, &sx, &ox))
        return RAD_E_SHAPE;
    const int64_t nh = (int64_t)sy.size(), nw = (int64_t)sx.size();
    const int64_t M = band_rows(nh * nw, band);
    if (M <= 0 || rows_of(Y, 22) < M) return RAD_E_SHAPE;
    const int64_t* psy = sy.data(), *psx = sx.data();
    AVX_PARALLEL_FOR_IF(M >= AVX_PAR_MIN)
    for (int64_t t = 0; t < M; ++t) {
        const int64_t i = t / nw, j = t % nw;
        const int64_t th = std::min(tile, H / ratio - psy[(size_t)i]) * ratio;
        const int64_t tw = std::min(tile, W / ratio - psx[(size_t)j]) * ratio;
        const int64_t yb = rowoff(Y, t, 22), ys = laststride(Y);
        const int64_t v[22] = { psy[(size_t)i] * ratio, psx[(size_t)j] * ratio, th, tw,
                                1, th, tw,   1, th / 2, tw / 2,   1, th / 4, tw / 4,
                                1, th / 8, tw / 8,   1, th / 16, tw / 16,   1, th / 16, tw / 16 };
        for (int64_t c = 0; c < 22; ++c) st_int_dt(Y->dtype, Y->data, yb + c * ys, v[c]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vtile_crop */
/* One frame's pixel plane cut into the plan's tiles (libref): y f32 [M, 3] holds the tiles'
 * pixels tile-major, M the tiles' pixel count. Only the row the tile stands on is read. */
AVX_KERNEL(vtile_crop) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *Y = &a->t[2];
    const int64_t band = p_int(a, "M", 0), tile = p_int(a, "tile", 0);
    const int64_t overlap = p_int(a, "overlap", 0), ratio = p_int(a, "ratio", 0);
    if (tile <= 0 || overlap < 0 || overlap >= tile || ratio <= 0 || numel(G) < 3)
        return RAD_E_SHAPE;
    const int64_t H = at(G, 1), W = at(G, 2);
    std::vector<int64_t> sy, sx, oy, ox;
    if (H <= 0 || W <= 0 || H % ratio || W % ratio ||
        !split_tiles_axis(H / ratio, tile, overlap, &sy, &oy) ||
        !split_tiles_axis(W / ratio, tile, overlap, &sx, &ox))
        return RAD_E_SHAPE;
    const int64_t nh = (int64_t)sy.size(), nw = (int64_t)sx.size();
    std::vector<int64_t> off((size_t)(nh * nw + 1), 0);
    for (int64_t i = 0; i < nh; ++i) {
        const int64_t th = std::min(tile, H / ratio - sy[(size_t)i]) * ratio;
        for (int64_t j = 0; j < nw; ++j)
            off[(size_t)(i * nw + j + 1)] =
                off[(size_t)(i * nw + j)] +
                th * std::min(tile, W / ratio - sx[(size_t)j]) * ratio;
    }
    const int64_t M = band_rows(off.back(), band);
    if (M <= 0 || rows_of(X, 3) < H * W || rows_of(Y, 3) < M) return RAD_E_SHAPE;
    const int64_t* psy = sy.data(), *psx = sx.data(), *poff = off.data();
    AVX_PARALLEL_FOR_IF(M * 3 >= AVX_PAR_MIN)
    for (int64_t r = 0; r < M; ++r) {
        const int64_t t =
            (int64_t)(std::upper_bound(poff, poff + nh * nw + 1, r) - poff) - 1;
        const int64_t i = t / nw, j = t % nw;
        const int64_t tw = std::min(tile, W / ratio - psx[(size_t)j]) * ratio;
        const int64_t k = r - poff[t];
        const int64_t src = (psy[(size_t)i] * ratio + k / tw) * W + psx[(size_t)j] * ratio + k % tw;
        const int64_t xb = rowoff(X, src, 3), yb = rowoff(Y, r, 3), xs = laststride(X);
        const int64_t ys = laststride(Y);
        for (int64_t c = 0; c < 3; ++c) stt(Y, yb + c * ys, ldt(X, xb + c * xs));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vtile_join */
/* The tiles' moments blended into one latent plane (libref): unblended neighbours, the top blend
 * before the left one, every product and sum rounded to x's dtype. */
AVX_KERNEL(vtile_join) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *Y = &a->t[2];
    const int64_t band = p_int(a, "M", 0), tile = p_int(a, "tile", 0);
    const int64_t overlap = p_int(a, "overlap", 0), n = p_int(a, "n", 0);
    if (tile <= 0 || overlap < 0 || overlap >= tile || n <= 0 || numel(G) < 3)
        return RAD_E_SHAPE;
    const int64_t H = at(G, 1), W = at(G, 2);
    std::vector<int64_t> sy, sx, oy, ox;
    if (H <= 0 || W <= 0 || !split_tiles_axis(H, tile, overlap, &sy, &oy) ||
        !split_tiles_axis(W, tile, overlap, &sx, &ox))
        return RAD_E_SHAPE;
    const int64_t nh = (int64_t)sy.size(), nw = (int64_t)sx.size();
    const int64_t M = band_rows(H * W, band);
    std::vector<int64_t> off((size_t)(nh * nw + 1), 0);
    for (int64_t i = 0; i < nh; ++i) {
        const int64_t th = std::min(tile, H - sy[(size_t)i]);
        for (int64_t j = 0; j < nw; ++j)
            off[(size_t)(i * nw + j + 1)] =
                off[(size_t)(i * nw + j)] + th * std::min(tile, W - sx[(size_t)j]);
    }
    if (M <= 0 || rows_of(X, n) < off.back() || rows_of(Y, n) < M) return RAD_E_SHAPE;
    /* The owner tile of every canvas row and column. */
    std::vector<int64_t> own_y((size_t)H), own_x((size_t)W);
    for (int64_t Yp = 0, i = 0; Yp < H; ++Yp) {
        while (i + 1 < nh && Yp >= sy[(size_t)(i + 1)]) ++i;
        own_y[(size_t)Yp] = i;
    }
    for (int64_t X0 = 0, j = 0; X0 < W; ++X0) {
        while (j + 1 < nw && X0 >= sx[(size_t)(j + 1)]) ++j;
        own_x[(size_t)X0] = j;
    }
    const uint32_t dt = X->dtype;
    auto rd = [dt](float v) { return narrow1(dt, v); };
    auto wb = [&](int64_t p, int64_t e) { return rd((float)p / (float)e); };
    auto wa = [&](int64_t p, int64_t e) { return rd(1.0f - rd((float)p / (float)e)); };
    const int64_t* psy = sy.data(), *psx = sx.data(), *poff = off.data();
    const int64_t* pown_y = own_y.data(), *pown_x = own_x.data();
    const int64_t xs = laststride(X), ys = laststride(Y);
    auto T = [&](int64_t i, int64_t j, int64_t ly, int64_t lx, int64_t c) {
        const int64_t row = poff[i * nw + j] + ly * std::min(tile, W - psx[(size_t)j]) + lx;
        return ldt(X, rowoff(X, row, n) + c * xs);
    };
    AVX_PARALLEL_FOR_IF(M * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < M; ++r) {
        const int64_t Yp = r / W, X0 = r % W;
        const int64_t i = pown_y[Yp], j = pown_x[X0];
        const int64_t ly = Yp - psy[(size_t)i], lx = X0 - psx[(size_t)j];
        const int64_t ev = i > 0 ? oy[(size_t)(i - 1)] : 0, eh = j > 0 ? ox[(size_t)(j - 1)] : 0;
        const int64_t yb = rowoff(Y, r, n);
        for (int64_t c = 0; c < n; ++c) {
            float v = T(i, j, ly, lx, c);
            if (ly < ev)
                v = rd(rd(T(i - 1, j, ly + tile - ev, lx, c) * wa(ly, ev)) +
                       rd(v * wb(ly, ev)));
            if (lx < eh)
                v = rd(rd(T(i, j - 1, ly, lx + tile - eh, c) * wa(lx, eh)) +
                       rd(v * wb(lx, eh)));
            stt(Y, yb + c * ys, v);
        }
    }
    return RAD_OK;
}

/* ================================================================== the latent upscaler
 * Voxel rows (t, i, j) of a T x H x W volume, channels innermost, the volume in `geom`. */

/* Division rounding toward negative infinity, as PyTorch's output-length formula takes it. */
static int64_t idiv_floor(int64_t x, int64_t d) {
    const int64_t q = x / d;
    return (x % d && (x < 0) != (d < 0)) ? q - 1 : q;
}

/* ------------------------------------------------------------------ conv3d_ndhwc */
/* A 3-D convolution as GEMMs over bands of patch rows, as conv2d_nhwc and conv1d take theirs. The
 * weight [cout, cin, kt, k, k] is re-laid once a launch as [cout][dt][kh][kw][cin] (bf16 copies
 * when it is bf16, f32 otherwise), so an output voxel's patch row is its kt k k input voxels'
 * channel runs back to back, zeros outside the volume (or `reflect`'s mirror): bf16 copies when x
 * is bf16 with dense rows, f32 otherwise. A band is one gemm_nt_into into f32 rows, the bias
 * added there; each row is then rounded once into y, or with `res` rounded to y's dtype, its
 * residual row added and rounded again. A band's residual rows are read before its rows of y are
 * written, row by row, so `res` may be y. The patches and the weight live in thread_local vectors
 * that keep their high-water size, leaving the GEMM's scratch arenas to it.
 *
 * The optional `stride`, `stride_t`, `causal`, `pad` and `reflect` are a causal video VAE's
 * convolution forms (libref's ref_conv3d_ndhwc, docs/OPS.md): output voxel (t, i, j) of the
 * T2 x H2 x W2 grid reads the input voxels from (t sz - pz0, i sy - py0, j sy - px0). */
AVX_KERNEL(conv3d_ndhwc) {
    if (a->n_t < 6 || !rad_args_have(a, 4) || !rad_arg_in(a, 5)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *W = &a->t[1], *B = &a->t[2], *G = &a->t[3];
    const RadTensor *R = rad_arg_in(a, 4), *Y = &a->t[5];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0);
    const int64_t k = p_int(a, "k", 0), kt = p_int(a, "kt", 0), tpre = p_int(a, "tpre", 0);
    /* The frames' padding (libref's): zero, the edge frame repeated, or causal with `tpre` more
     * copies of the first frame in front (T + tpre frames out). */
    const char* tps = p_str(a, "tpad", "zero");
    const int tpad = !std::strcmp(tps, "zero") ? 0 : !std::strcmp(tps, "replicate") ? 1
                   : !std::strcmp(tps, "causal") ? 2 : -1;
    if (tpad < 0 || (tpre && tpad != 2)) return RAD_E_INVAL;
    if (cin <= 0 || cout <= 0 || k <= 0 || kt <= 0 || !(k & 1) || !(kt & 1) || tpre < 0 ||
        numel(G) < 3)
        return RAD_E_SHAPE;
    const int64_t sy = p_int(a, "stride", 1), sz = p_int(a, "stride_t", 1);
    const int64_t causal = p_int(a, "causal", 0), reflect = p_int(a, "reflect", 0);
    const char* pads = p_str(a, "pad", "same");
    const bool down = pads && !std::strcmp(pads, "down");
    if (sy <= 0 || sz <= 0 || (causal && causal != 1) || (reflect && reflect != 1) ||
        (causal && tpad) /* `causal` pads with zeros, `tpad` with copies */ ||
        !(!pads || !std::strcmp(pads, "same") || down))
        return RAD_E_SHAPE;
    const bool gen = sy != 1 || sz != 1 || causal || down || reflect;
    if (gen && (k != 3 || kt != 3)) return RAD_E_SHAPE;
    const int64_t T = at(G, 0), H = at(G, 1), Wd = at(G, 2);
    if (T <= 0 || H <= 0 || Wd <= 0) return RAD_E_SHAPE;
    /* The pads before (0) and after (1) each axis. `tpad` replicate centres the window as `zero`
     * does and clamps the reads to the volume; `tpad` causal puts kt - 1 + tpre copies of the
     * first frame in front, T + tpre frames out, the grid `causal` takes with zeros. */
    const int64_t pz0 = causal ? kt - 1 : tpad == 2 ? kt - 1 + tpre : kt / 2;
    const int64_t pz1 = (causal || tpad == 2) ? 0 : kt / 2;
    const int64_t py0 = down ? 0 : k / 2, py1 = down ? 1 : k / 2;
    const int64_t px0 = down ? 0 : k / 2, px1 = down ? 1 : k / 2;
    if (reflect && (std::max(py0, py1) >= H || std::max(px0, px1) >= Wd)) return RAD_E_SHAPE;
    const int64_t T2 = idiv_floor(T + pz0 + pz1 - kt, sz) + 1;
    const int64_t H2 = idiv_floor(H + py0 + py1 - k, sy) + 1;
    const int64_t W2 = idiv_floor(Wd + px0 + px1 - k, sy) + 1;
    if (T2 <= 0 || H2 <= 0 || W2 <= 0) return RAD_E_SHAPE;
    const int64_t rows = T * H * Wd, orows = T2 * H2 * W2, taps = kt * k * k, KK = taps * cin;
    if (rows_of(Y, cout) < orows || rows_of(X, cin) < rows || (R && rows_of(R, cout) < orows) ||
        numel(W) < cout * KK || numel(B) < cout)
        return RAD_E_SHAPE;
    static thread_local std::vector<uint8_t> wbuf, pbuf;
    static thread_local std::vector<float> ybuf;

    /* ---- the weight, [cout][dt][kh][kw][cin] */
    const bool w16 = W->dtype == RAD_BF16;
    wbuf.resize((size_t)(cout * KK) * (w16 ? 2 : 4) + 64);
    uint8_t* wp = wbuf.data();
    AVX_PARALLEL_FOR_IF(cout * KK >= AVX_PAR_MIN)
    for (int64_t co = 0; co < cout; ++co)
        for (int64_t ci = 0; ci < cin; ++ci)
            for (int64_t tp = 0; tp < taps; ++tp) {
                const int64_t src = (co * cin + ci) * taps + tp, dst = co * KK + tp * cin + ci;
                if (w16) ((uint16_t*)wp)[dst] = ((const uint16_t*)W->data)[src];
                else     ((float*)wp)[dst] = ld_elem(W->data, W->dtype, src);
            }
    const RadTensor wt = mat_view(wp, w16 ? (uint32_t)RAD_BF16 : (uint32_t)RAD_F32, cout, KK);

    /* ---- bands of patch rows: at least 8 Mi elements and eight times the weight, so the GEMM's
     * per-call weight pack is amortised, at most 32 Mi (64 MiB of bf16 kept after the call) */
    const bool x16 = X->dtype == RAD_BF16 && laststride(X) == 1;
    const size_t ael = x16 ? 2 : 4;
    const int64_t want = std::min<int64_t>((int64_t)1 << 25,
                                           std::max<int64_t>((int64_t)1 << 23, 8 * cout * KK));
    const int64_t band = std::max<int64_t>(1, std::min(orows, want / KK));
    pbuf.resize((size_t)(band * KK) * ael + 64);
    ybuf.resize((size_t)(band * cout) + 16);
    char* pp = (char*)pbuf.data();
    float* yp = ybuf.data();
    for (int64_t r0 = 0; r0 < orows; r0 += band) {
        const int64_t nb = std::min(band, orows - r0);
        AVX_PARALLEL_FOR_IF(nb * KK >= AVX_PAR_MIN)
        for (int64_t br = 0; br < nb; ++br) {
            const int64_t r = r0 + br, t = r / (H2 * W2), i = (r / W2) % H2, j = r % W2;
            char* d = pp + (size_t)(br * KK) * ael;
            for (int64_t dz = 0; dz < kt; ++dz)
                for (int64_t kh = 0; kh < k; ++kh)
                    for (int64_t kw = 0; kw < k; ++kw, d += (size_t)cin * ael) {
                        int64_t tz = t * sz + dz - pz0;
                        if (tpad)   /* copies of the edge frame; `zero` and `causal` read zeros */
                            tz = std::min<int64_t>(std::max<int64_t>(tz, 0), T - 1);
                        int64_t iy = i * sy + kh - py0;
                        int64_t ix = j * sy + kw - px0;
                        if (reflect) {   /* PyTorch's reflect: the edge itself is not repeated */
                            if (iy < 0) iy = -iy; else if (iy >= H) iy = 2 * H - 2 - iy;
                            if (ix < 0) ix = -ix; else if (ix >= Wd) ix = 2 * Wd - 2 - ix;
                        }
                        if (tz < 0 || tz >= T || iy < 0 || iy >= H || ix < 0 || ix >= Wd) {
                            std::memset(d, 0, (size_t)cin * ael);
                            continue;
                        }
                        const int64_t xo = rowoff(X, (tz * H + iy) * Wd + ix, cin);
                        if (x16) {
                            std::memcpy(d, (const uint16_t*)X->data + xo, (size_t)cin * 2);
                        } else {
                            const float* s = in_row(X, xo, cin, (float*)d);
                            if (s != (const float*)d) std::memcpy(d, s, (size_t)cin * 4);
                        }
                    }
        }
        const RadTensor pt = mat_view(pp, x16 ? (uint32_t)RAD_BF16 : (uint32_t)RAD_F32, nb, KK);
        const RadTensor yt = mat_view(yp, RAD_F32, nb, cout);
        const int st = gemm_nt_into(&pt, &wt, B, &yt, nb, cout, KK);
        if (st < 0) return st;
        AVX_PARALLEL_FOR_IF(nb * cout >= AVX_PAR_MIN)
        for (int64_t br = 0; br < nb; ++br) {
            float* row = yp + br * cout;
            if (R) {
                float* buf = scratch_f32((size_t)(2 * cout + 32));
                float *tmp = buf, *rs = buf + cout + 16;
                narrow_row(Y->dtype, row, cout, tmp);
                const float* rr = in_row(R, rowoff(R, r0 + br, cout), cout, rs);
                for (int64_t c = 0; c < cout; ++c) row[c] += rr[c];
            }
            out_row(Y, rowoff(Y, r0 + br, cout), cout, row);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ group_norm */
/* A group norm over all rows (libref's definition). The statistics are f64 sums over fixed blocks
 * of 64 rows, each block's per-group sums in row-major order and the blocks combined in block
 * order, so the bytes do not depend on the thread count; two passes, the variance about the f64
 * mean. The rows are then normalised in f32 from per-channel tables (the group's mean and rstd,
 * w, b, 1 + scale and shift), each product kept apart from the add after it as libref rounds
 * them, and rounded once. stats[0, 2 groups) gets the mean and rstd; the rest of `stats` is not
 * touched, the block sums being kept in this plugin's own memory. */
AVX_KERNEL(group_norm) {
    if (a->n_t < 6 || !rad_args_have(a, 3) || !rad_arg_in(a, 4) || !rad_arg_in(a, 5))
        return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *W = &a->t[1], *B = &a->t[2], *MD = rad_arg_in(a, 3);
    const RadTensor *ST = &a->t[4], *Y = &a->t[5];
    const int64_t n = p_int(a, "n", 0), G = p_int(a, "groups", 0);
    const float eps = p_f32(a, "eps", 1e-5f);
    static const char* const kActs[] = { "none", "silu" };
    const int act = p_enum(a, "act", kActs, 2, 0);
    if (act < 0 || act > 1) return RAD_E_INVAL;
    if (n <= 0 || G <= 0 || n % G || numel(W) < n || numel(B) < n ||
        (MD && numel(MD) < 2 * n) || numel(ST) < G * (2 + 3 * 1024))
        return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n), cg = n / G;
    if (rows <= 0 || rows_of(X, n) < rows) return RAD_E_SHAPE;
    constexpr int64_t kBlock = 64;
    const int64_t nblk = (rows + kBlock - 1) / kBlock;
    const double cnt = (double)rows * (double)cg;
    static thread_local std::vector<double> part;
    part.assign((size_t)(nblk * G + 2 * G), 0.0);
    double *ps = part.data(), *mu = ps + nblk * G, *var = mu + G;

    /* Block bk's per-group sums of x, or with `dev` of (x - mu)^2, into ps[bk G, (bk + 1) G). */
    for (const bool dev : { false, true }) {
        AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
        for (int64_t bk = 0; bk < nblk; ++bk) {
            float* xs = scratch_f32((size_t)n + 16);
            double* s = ps + bk * G;
            const int64_t r1 = std::min(rows, (bk + 1) * kBlock);
            for (int64_t r = bk * kBlock; r < r1; ++r) {
                const float* x = in_row(X, rowoff(X, r, n), n, xs);
                for (int64_t g = 0; g < G; ++g) {
                    const float* xg = x + g * cg;
                    double acc = s[g];
                    if (dev) {
                        for (int64_t c = 0; c < cg; ++c) {
                            const double d = (double)xg[c] - mu[g];
                            acc += d * d;
                        }
                    } else {
                        for (int64_t c = 0; c < cg; ++c) acc += (double)xg[c];
                    }
                    s[g] = acc;
                }
            }
        }
        for (int64_t g = 0; g < G; ++g) {
            double tot = 0.0;
            for (int64_t bk = 0; bk < nblk; ++bk) {
                tot += ps[bk * G + g];
                ps[bk * G + g] = 0.0;
            }
            (dev ? var : mu)[g] = tot / cnt;
        }
    }

    /* Per-channel tables: mean, rstd, w, b, 1 + scale and shift. */
    float* tab = (float*)scratch_raw((size_t)(6 * n + 16) * sizeof(float));
    float *mc = tab, *rc = mc + n, *wc = rc + n, *bc = wc + n, *sc = bc + n, *sh = sc + n;
    for (int64_t g = 0; g < G; ++g) {
        const float m = (float)mu[g], rs = 1.0f / std::sqrt((float)var[g] + eps);
        stt(ST, offlin(ST, 2 * g), m);
        stt(ST, offlin(ST, 2 * g + 1), rs);
        for (int64_t c = g * cg; c < (g + 1) * cg; ++c) { mc[c] = m; rc[c] = rs; }
    }
    for (int64_t c = 0; c < n; ++c) {
        wc[c] = ld_elem(W->data, W->dtype, c);
        bc[c] = ld_elem(B->data, B->dtype, c);
        sc[c] = MD ? 1.0f + ldt(MD, offlin(MD, c)) : 1.0f;
        sh[c] = MD ? ldt(MD, offlin(MD, n + c)) : 0.0f;
    }
    const bool modded = MD != nullptr;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(2 * n + 32));
        float *xs = buf, *v = buf + n + 16;
        const float* x = in_row(X, rowoff(X, r, n), n, xs);
        const vf one = vf_set1(1.0f);
        int64_t c = 0;
        for (; c + VF_N <= n; c += VF_N) {
            vf t = vf_mul(vf_sub(vf_loadu(x + c), vf_loadu(mc + c)), vf_loadu(rc + c));
            t = vf_add(keep(vf_mul(t, vf_loadu(wc + c))), vf_loadu(bc + c));
            if (modded) t = vf_add(keep(vf_mul(t, vf_loadu(sc + c))), vf_loadu(sh + c));
            if (act) t = vf_div(t, vf_add(one, vf_exp(vf_sub(vf_zero(), t))));
            vf_storeu(v + c, t);
        }
        for (; c < n; ++c) {
            float t = (x[c] - mc[c]) * rc[c];
            t = keep(t * wc[c]) + bc[c];
            if (modded) t = keep(t * sc[c]) + sh[c];
            if (act) t = t / (1.0f + sc_exp(-t));
            v[c] = t;
        }
        out_row(Y, rowoff(Y, r, n), n, v);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ tconv_dw */
/* A depthwise convolution along the frames, libref's sum bit for bit: each output row adds its
 * in-volume taps in ascending order, every product rounded before its add, then the bias, a
 * vector across the channels. The taps are re-laid once a launch as [k][n] f32 beside the bias. */
AVX_KERNEL(tconv_dw) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *W = &a->t[1], *B = &a->t[2], *G = &a->t[3], *Y = &a->t[4];
    const int64_t n = p_int(a, "n", 0), k = p_int(a, "k", 0);
    if (n <= 0 || k <= 0 || !(k & 1) || numel(G) < 3 || numel(W) < n * k || numel(B) < n)
        return RAD_E_SHAPE;
    const int64_t T = at(G, 0), H = at(G, 1), Wd = at(G, 2);
    if (T <= 0 || H <= 0 || Wd <= 0) return RAD_E_SHAPE;
    const int64_t P = H * Wd, rows = T * P;
    if (rows_of(Y, n) < rows || rows_of(X, n) < rows) return RAD_E_SHAPE;
    float* tab = (float*)scratch_raw((size_t)((k + 1) * n + 16) * sizeof(float));
    for (int64_t c = 0; c < n; ++c) {
        for (int64_t j = 0; j < k; ++j) tab[j * n + c] = ld_elem(W->data, W->dtype, c * k + j);
        tab[k * n + c] = ld_elem(B->data, B->dtype, c);
    }
    const float* bias = tab + k * n;

    AVX_PARALLEL_FOR_IF(rows * n * k >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)(2 * n + 32));
        float *xs = buf, *acc = buf + n + 16;
        const int64_t t = r / P, p = r % P;
        for (int64_t c = 0; c < n; ++c) acc[c] = 0.0f;
        for (int64_t j = 0; j < k; ++j) {
            const int64_t tz = t + j - k / 2;
            if (tz < 0 || tz >= T) continue;
            const float* x = in_row(X, rowoff(X, tz * P + p, n), n, xs);
            const float* w = tab + j * n;
            int64_t c = 0;
            for (; c + VF_N <= n; c += VF_N)
                vf_storeu(acc + c, vf_add(vf_loadu(acc + c),
                                          keep(vf_mul(vf_loadu(w + c), vf_loadu(x + c)))));
            for (; c < n; ++c) acc[c] = acc[c] + keep(w[c] * x[c]);
        }
        int64_t c = 0;
        for (; c + VF_N <= n; c += VF_N)
            vf_storeu(acc + c, vf_add(vf_loadu(acc + c), vf_loadu(bias + c)));
        for (; c < n; ++c) acc[c] = acc[c] + bias[c];
        out_row(Y, rowoff(Y, r, n), n, acc);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ resize_bilinear */
/* Each frame resized as torch's align_corners=False bilinear at a size, libref's arithmetic bit
 * for bit: the source rows, columns and weights of every output row and column are taken once a
 * launch (libref's f32 steps, the product before the subtraction kept apart), and an output row
 * is its four source rows blended, a vector across the channels, each product kept apart from
 * the add after it. */
AVX_KERNEL(resize_bilinear) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || numel(G) < 5) return RAD_E_SHAPE;
    const int64_t T = at(G, 0), h = at(G, 1), w = at(G, 2), H = at(G, 3), Wd = at(G, 4);
    if (T <= 0 || h <= 0 || w <= 0 || H <= 0 || Wd <= 0) return RAD_E_SHAPE;
    const int64_t rows = T * H * Wd;
    if (rows_of(Y, n) < rows || rows_of(X, n) < T * h * w) return RAD_E_SHAPE;
    /* Per output index of each axis: two source indices, then two weights. */
    const int64_t E = H + Wd;
    int64_t* idx =
        (int64_t*)scratch_raw((size_t)E * (2 * sizeof(int64_t) + 2 * sizeof(float)) + 64);
    float* lam = (float*)(idx + 2 * E);
    auto axis = [&](int64_t e, int64_t o, int64_t in, int64_t out) {
        const float s = (float)in / (float)out;
        float src = keep(s * ((float)o + 0.5f)) - 0.5f;
        if (src < 0.0f) src = 0.0f;
        const int64_t i0 = (int64_t)src;
        idx[2 * e] = i0;
        idx[2 * e + 1] = i0 + (i0 < in - 1 ? 1 : 0);
        lam[2 * e + 1] = src - (float)i0;
        lam[2 * e] = 1.0f - lam[2 * e + 1];
    };
    for (int64_t i = 0; i < H; ++i) axis(i, i, h, H);
    for (int64_t j = 0; j < Wd; ++j) axis(H + j, j, w, Wd);

    AVX_PARALLEL_FOR_IF(rows * n * 4 >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t t = r / (H * Wd), i = (r / Wd) % H, j = r % Wd;
        const int64_t i0 = idx[2 * i], i1 = idx[2 * i + 1];
        const int64_t j0 = idx[2 * (H + j)], j1 = idx[2 * (H + j) + 1];
        const float l0y = lam[2 * i], l1y = lam[2 * i + 1];
        const float l0x = lam[2 * (H + j)], l1x = lam[2 * (H + j) + 1];
        float* buf = scratch_f32((size_t)(5 * n + 80));
        float *s0 = buf, *s1 = s0 + n + 16, *s2 = s1 + n + 16, *s3 = s2 + n + 16, *v = s3 + n + 16;
        const float* x00 = in_row(X, rowoff(X, (t * h + i0) * w + j0, n), n, s0);
        const float* x01 = in_row(X, rowoff(X, (t * h + i0) * w + j1, n), n, s1);
        const float* x10 = in_row(X, rowoff(X, (t * h + i1) * w + j0, n), n, s2);
        const float* x11 = in_row(X, rowoff(X, (t * h + i1) * w + j1, n), n, s3);
        const vf a0 = vf_set1(l0x), a1 = vf_set1(l1x), b0 = vf_set1(l0y), b1 = vf_set1(l1y);
        int64_t c = 0;
        for (; c + VF_N <= n; c += VF_N) {
            const vf top = vf_add(keep(vf_mul(a0, vf_loadu(x00 + c))),
                                  keep(vf_mul(a1, vf_loadu(x01 + c))));
            const vf bot = vf_add(keep(vf_mul(a0, vf_loadu(x10 + c))),
                                  keep(vf_mul(a1, vf_loadu(x11 + c))));
            vf_storeu(v + c, vf_add(keep(vf_mul(b0, top)), keep(vf_mul(b1, bot))));
        }
        for (; c < n; ++c) {
            const float top = keep(l0x * x00[c]) + keep(l1x * x01[c]);
            const float bot = keep(l0x * x10[c]) + keep(l1x * x11[c]);
            v[c] = keep(l0y * top) + keep(l1y * bot);
        }
        out_row(Y, rowoff(Y, r, n), n, v);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ flow_renoise */
/* x = f32(f32(f32(1 - s) x) + f32(s z)) over x's rows, z the (seed, stream) normal stream at
 * k n + c (libref's, bit for bit): a row's draws come a Philox block at a time. */
AVX_KERNEL(flow_renoise) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *SG = &a->t[0], *SD = &a->t[1], *X = &a->t[2];
    const int64_t n = p_int(a, "n", 0), sm = p_int(a, "stream", 0);
    if (n <= 0 || sm < 0 || numel(SG) < 1 || numel(SD) < 2) return RAD_E_SHAPE;
    const int64_t rows = rows_of(X, n);
    if (rows <= 0) return RAD_E_SHAPE;
    const float s = ldt(SG, offlin(SG, 0)), oms = 1.0f - s;
    const uint64_t sd = (uint64_t)(uint32_t)at(SD, 0) | ((uint64_t)(uint32_t)at(SD, 1) << 32);

    AVX_PARALLEL_FOR_IF(rows * n * 8 >= AVX_PAR_MIN)
    for (int64_t k = 0; k < rows; ++k) {
        float* buf = scratch_f32((size_t)(3 * n + 48));
        float *xs = buf, *z = xs + n + 16, *o = z + n + 16;
        const float* x = in_row(X, rowoff(X, k, n), n, xs);
        philox_normals_s(sd, (uint64_t)sm, (uint64_t)(k * n), n, z);
        const vf vo = vf_set1(oms), vs = vf_set1(s);
        int64_t c = 0;
        for (; c + VF_N <= n; c += VF_N)
            vf_storeu(o + c, vf_add(keep(vf_mul(vo, vf_loadu(x + c))),
                                    keep(vf_mul(vs, vf_loadu(z + c)))));
        for (; c < n; ++c) o[c] = keep(oms * x[c]) + keep(s * z[c]);
        out_row(X, rowoff(X, k, n), n, o);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ cond_aug */
/* A visual condition's rows blended with noise (libref's ref_cond_aug, bit for bit):
 * x = f32(f32(a x) + f32(oma z)), oma the f32 of the f64 1 - aug, z the (seed, stream) normal
 * stream at k n + c. */
AVX_KERNEL(cond_aug) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *SD = &a->t[0], *X = &a->t[1];
    const int64_t n = p_int(a, "n", 0), sm = p_int(a, "stream", 0);
    const RadParam* ap = p_find(a, "aug");
    if (!ap || ap->kind != RAD_P_F64) return RAD_E_SHAPE;
    const double aug = ap->dval;
    if (n <= 0 || sm < 0 || !(aug >= 0.0 && aug < 1.0) || numel(SD) < 2) return RAD_E_SHAPE;
    const int64_t rows = rows_of(X, n);
    if (rows <= 0) return RAD_E_SHAPE;
    const float av = (float)aug, omv = (float)(1.0 - aug);
    const uint64_t sd = (uint64_t)(uint32_t)at(SD, 0) | ((uint64_t)(uint32_t)at(SD, 1) << 32);

    AVX_PARALLEL_FOR_IF(rows * n * 8 >= AVX_PAR_MIN)
    for (int64_t k = 0; k < rows; ++k) {
        float* buf = scratch_f32((size_t)(3 * n + 48));
        float *xs = buf, *z = xs + n + 16, *o = z + n + 16;
        const float* x = in_row(X, rowoff(X, k, n), n, xs);
        philox_normals_s(sd, (uint64_t)sm, (uint64_t)(k * n), n, z);
        const vf va = vf_set1(av), vo = vf_set1(omv);
        int64_t c = 0;
        for (; c + VF_N <= n; c += VF_N)
            vf_storeu(o + c, vf_add(keep(vf_mul(va, vf_loadu(x + c))),
                                    keep(vf_mul(vo, vf_loadu(z + c)))));
        for (; c < n; ++c) o[c] = keep(av * x[c]) + keep(omv * z[c]);
        out_row(X, rowoff(X, k, n), n, o);
    }
    return RAD_OK;
}
