/* ref_video.cpp -- the video generator's ops around its decoder (docs/OPS.md, "Video
 * generation"): the denoised latent's patch rows unfolded to voxels, the tiles a video decoder
 * runs over, gathered from the latent and stitched back into frames, and the rotary positions of
 * a tile's tokens; and the latent upscaler's (docs/OPS.md, "The latent upscaler"): voxel rows
 * folded back into patch rows, the 3-D and temporal depthwise convolutions, the group norm, the
 * per-frame bilinear resize and the re-noising of the upscaled latent.
 *
 * Per-request geometry (a video's frames, height and width) comes from a `geom` operand the run
 * phase writes on the device, never from the parameters, which are fixed at declare. The tiles
 * are diffusers' AutoencoderKLMiniMaxH3 tiling in latent units, and the stitch rounds where its
 * bf16 blends round. The upscaler's ops round once into their output, where a fused kernel would.
 */
#include "ref_common.h"
#include "ref_ops.h"
#include "rad_philox.h"

#include <algorithm>
#include <vector>

using namespace ref;

namespace {

inline int64_t ldi(const RadTensor* t, int64_t i) {
    return ld_int(t->dtype, t->data, offlin(t, i));
}

/* A space-separated list of floats, kept in f32: a video's latent and pixel statistics are f32
 * tensors upstream, unlike the bf16 lists chan_affine reads. */
int parse_f32_list(const char* s, std::vector<float>* out) {
    out->clear();
    while (s && *s) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end == s) break;
        out->push_back((float)v);
        s = end;
    }
    return (int)out->size();
}

/* The tiles over one axis of `len` latent units, diffusers' _split_tiles in latent units: a tile
 * at least as long as the axis is the whole axis (and must be exactly as long, every tile being
 * tile x tile); otherwise the fewest tiles whose union covers the axis with every overlap at
 * least `overlap`, the slack added to the overlaps one unit at a time round-robin from the first,
 * and start[i + 1] = start[i] + tile - ov[i]. False where no such layout exists. */
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

/* The encoder's split over one axis (comfy/ldm/minimax/vae.py, `split_tiles`): split_tiles,
 * except that an axis no longer than a tile is one tile, short if need be, where the decoder's
 * split demands an exact fit. */
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

}  /* namespace */

/* ------------------------------------------------------------------ patch_unfold */
/* A video latent's patch rows unfolded to voxel rows and denormalised. x [F Hp Wp, C pt ph pw]
 * holds patch (f, i, j), frame-major then row-major, its column (c, a, b, d) at
 * c pt ph pw + a ph pw + b pw + d; y [(F pt) (Hp ph) (Wp pw), C] holds voxel (t, yy, xx)
 * row-major:
 *
 *   y[(t, yy, xx), c] = f32(f32(x[(t / pt, yy / ph, xx / pw), (c, t % pt, yy % ph, xx % pw)]
 *                               * mul[c]) + add[c])
 *
 * `geom` is (F, Hp, Wp), the request's latent grid in patches. `mul` and `add` are C values each,
 * kept in f32 (the reference denormalises f32 latents with an f32 deviation and mean). y holds at
 * least the voxels' rows; rows past them are not written. */
int ref_patch_unfold(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *geom = &a->t[1], *y = &a->t[2];
    const int64_t C = p_int(a, "C", 0), pt = p_int(a, "pt", 0);
    const int64_t ph = p_int(a, "ph", 0), pw = p_int(a, "pw", 0);
    std::vector<float> mul, add;
    if (C <= 0 || pt <= 0 || ph <= 0 || pw <= 0 || parse_f32_list(p_str(a, "mul", ""), &mul) != C ||
        parse_f32_list(p_str(a, "add", ""), &add) != C || numel(geom) < 3)
        return RAD_E_SHAPE;
    const int64_t F = ldi(geom, 0), Hp = ldi(geom, 1), Wp = ldi(geom, 2);
    if (F <= 0 || Hp <= 0 || Wp <= 0) return RAD_E_SHAPE;
    const int64_t Hv = Hp * ph, Wv = Wp * pw, K = C * pt * ph * pw, rows = F * pt * Hv * Wv;
    if (rows_of(y, C) < rows || rows_of(x, K) < F * Hp * Wp) return RAD_E_SHAPE;
    REF_PAR_FOR(rows * C)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t t = r / (Hv * Wv), yy = (r / Wv) % Hv, xx = r % Wv;
        const int64_t patch = ((t / pt) * Hp + yy / ph) * Wp + xx / pw;
        const int64_t sub = ((t % pt) * ph + yy % ph) * pw + xx % pw;
        const int64_t xb = rowoff(x, patch, K), xs = laststride(x);
        const int64_t yb = rowoff(y, r, C), ys = laststride(y);
        for (int64_t c = 0; c < C; ++c)
            stt(y, yb + c * ys,
                ldt(x, xb + (c * pt * ph * pw + sub) * xs) * mul[(size_t)c] + add[(size_t)c]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ patch_fold */
/* patch_unfold's inverse, normalising instead of denormalising: voxel rows
 * x [(F pt) (Hp ph) (Wp pw), C] folded into patch rows y [F Hp Wp, C pt ph pw], column
 * (c, a, b, d) of patch (f, i, j) at c pt ph pw + a ph pw + b pw + d:
 *
 *   y[(f, i, j), (c, a, b, d)] = f32(f32(x[(f pt + a, i ph + b, j pw + d), c] * mul[c]) + add[c])
 *
 * `geom` is (F, Hp, Wp); `mul` and `add` are C values each, kept in f32. y holds at least the
 * patches' rows; rows past them are not written. */
int ref_patch_fold(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *geom = &a->t[1], *y = &a->t[2];
    const int64_t C = p_int(a, "C", 0), pt = p_int(a, "pt", 0);
    const int64_t ph = p_int(a, "ph", 0), pw = p_int(a, "pw", 0);
    std::vector<float> mul, add;
    if (C <= 0 || pt <= 0 || ph <= 0 || pw <= 0 || parse_f32_list(p_str(a, "mul", ""), &mul) != C ||
        parse_f32_list(p_str(a, "add", ""), &add) != C || numel(geom) < 3)
        return RAD_E_SHAPE;
    const int64_t F = ldi(geom, 0), Hp = ldi(geom, 1), Wp = ldi(geom, 2);
    if (F <= 0 || Hp <= 0 || Wp <= 0) return RAD_E_SHAPE;
    const int64_t vol = pt * ph * pw, Hv = Hp * ph, Wv = Wp * pw, K = C * vol, rows = F * Hp * Wp;
    if (rows_of(y, K) < rows || rows_of(x, C) < F * pt * Hv * Wv) return RAD_E_SHAPE;
    REF_PAR_FOR(rows * K)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t f = r / (Hp * Wp), i = (r / Wp) % Hp, j = r % Wp;
        const int64_t yb = rowoff(y, r, K), ys = laststride(y), xs = laststride(x);
        for (int64_t sub = 0; sub < vol; ++sub) {
            const int64_t ta = sub / (ph * pw), tb = (sub / pw) % ph, td = sub % pw;
            const int64_t vox = ((f * pt + ta) * Hv + i * ph + tb) * Wv + j * pw + td;
            const int64_t xb = rowoff(x, vox, C);
            for (int64_t c = 0; c < C; ++c)
                stt(y, yb + (c * vol + sub) * ys,
                    ldt(x, xb + c * xs) * mul[(size_t)c] + add[(size_t)c]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ grid_pos3 */
/* A video decoder's rotary positions over consecutive tiles: row m is token k = m % period of its
 * tile, the tile's T x H x W tokens first ((t, i, j) row-major) and the period's remaining rows
 * (register tokens) at position 0. Each axis is the token's centre normalised to [-1, 1) and
 * scaled, in f32 as torch's 2 * (arange(0.5, n) / n) - 1 makes it:
 *
 *   g(c, n) = f32(f32(2 f32((c + 0.5) / n)) - 1)     y[axis][m] = f32(f32(scale) g)
 *
 * y is f32 [3, >= M], component-major, each component a row at the operand's own pitch (a [3, M]
 * view of a wider buffer); M is its row length. */
int ref_grid_pos3(const RadArgs* a, RadStream) {
    if (!have(a, 1)) return RAD_E_INVAL;
    const RadTensor* y = &a->t[0];
    const int64_t T = p_int(a, "T", 0), H = p_int(a, "H", 0), W = p_int(a, "W", 0);
    const int64_t period = p_int(a, "period", 0);
    const float scale = p_f32(a, "scale", 1.0f);
    if (T <= 0 || H <= 0 || W <= 0 || period < T * H * W || y->rank != 2 || y->shape[0] < 3)
        return RAD_E_SHAPE;
    const int64_t M = y->shape[1], rs = y->stride[0], cs = y->stride[1];
    if (M <= 0) return RAD_E_SHAPE;
    auto g = [](int64_t c, int64_t n) {
        const float q = ((float)c + 0.5f) / (float)n;
        return 2.0f * q - 1.0f;
    };
    REF_PAR_FOR(M * 3)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t k = m % period;
        float p[3] = { 0.0f, 0.0f, 0.0f };
        if (k < T * H * W) {
            p[0] = scale * g(k / (H * W), T);
            p[1] = scale * g((k / W) % H, H);
            p[2] = scale * g(k % W, W);
        }
        for (int c = 0; c < 3; ++c) stt(y, c * rs + m * cs, p[c]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vtile_gather */
/* The tiles a video decoder runs over, gathered from voxel rows x [frames H W, n] ((t, yy, xx)
 * row-major), `geom` = (frames, H, W) in latent units, its frames those of the `frames`
 * parameter. Each spatial axis is split by split_tiles (`tile`, `overlap`) and the tiles are taken
 * row-major over (row band, column band). Tile k is rows [k period, (k + 1) period) of y: its
 * frames tile tile voxels (t, ty, tx) row-major, from x's row (t, start_y + ty, start_x + tx);
 * then reg's first `n_reg` rows (a decoder's register tokens) in y's dtype; then zero rows to the
 * period. y holds at least n_tiles period rows. */
int ref_vtile_gather(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *reg = &a->t[1], *geom = &a->t[2], *y = &a->t[3];
    const int64_t n = p_int(a, "n", 0), tile = p_int(a, "tile", 0);
    const int64_t overlap = p_int(a, "overlap", 0), frames = p_int(a, "frames", 0);
    const int64_t period = p_int(a, "period", 0), n_reg = p_int(a, "n_reg", 0);
    const int64_t vox = frames * tile * tile;
    if (n <= 0 || tile <= 0 || frames <= 0 || n_reg < 0 || period < vox + n_reg ||
        numel(geom) < 3 || rows_of(reg, n) < n_reg)
        return RAD_E_SHAPE;
    const int64_t H = ldi(geom, 1), W = ldi(geom, 2);
    std::vector<int64_t> sy, sx, oy, ox;
    if (ldi(geom, 0) != frames || !split_tiles(H, tile, overlap, &sy, &oy) ||
        !split_tiles(W, tile, overlap, &sx, &ox))
        return RAD_E_SHAPE;
    const int64_t nw = (int64_t)sx.size(), rows = (int64_t)sy.size() * nw * period;
    if (rows_of(y, n) < rows || rows_of(x, n) < frames * H * W) return RAD_E_SHAPE;
    REF_PAR_FOR(rows * n)
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t k = row / period, p = row % period;
        const int64_t yb = rowoff(y, row, n), ys = laststride(y);
        if (p < vox) {
            const int64_t t = p / (tile * tile), ty = (p / tile) % tile, tx = p % tile;
            const int64_t src = (t * H + sy[(size_t)(k / nw)] + ty) * W + sx[(size_t)(k % nw)] + tx;
            const int64_t xb = rowoff(x, src, n), xs = laststride(x);
            for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, ldt(x, xb + c * xs));
        } else if (p < vox + n_reg) {
            const int64_t rb = rowoff(reg, p - vox, n), rs = laststride(reg);
            for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, ldt(reg, rb + c * rs));
        } else {
            for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, 0.0f);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vtile_stitch */
/* A decoded chunk's tiles stitched into frames, blended with the previous chunk's overlap, and
 * written as RGB bytes.
 *
 * x is the decoder's output over vtile_gather's tiles, [>= n_tiles period, 3 pt scale scale]:
 * tile k's row k period + (lt tile + ty) tile + tx, column
 * c pt scale^2 + a scale^2 + b scale + d, is colour c of pixel (ty scale + b, tx scale + d) of clip
 * frame lt pt + a. `geom` = (frames, H, W, first, last): the chunk's latent geometry, and whether
 * it is its video's first and last chunk (nonzero). Tiles are split_tiles' over H and W.
 *
 * Every product and sum below rounds to x's dtype (rd), as the reference's bf16 blends do, with
 * wa(p, e) = rd(1 - rd(p / e)) and wb(p, e) = rd(p / e). Pixel (Y, X) of clip frame fc belongs to
 * the tile (i, j) whose kept range holds it, [start_i scale, start_{i+1} scale) (the last tile to
 * its end), at (ly, lx) within it; T(i, j, ly, lx) is that tile's decoded value. The spatial
 * stitch blends the top ev = ov_y[i-1] scale rows with the tile above and then the left
 * eh = ov_x[j-1] scale columns with the tile to the left, both neighbours as decoded (never
 * blended):
 *
 *   V = rd(rd(T(i-1, j, ly + tile scale - ev, lx) wa(ly, ev)) + rd(T(i, j, ly, lx) wb(ly, ev)))
 *       when i > 0 and ly < ev, else T(i, j, ly, lx)
 *   S = rd(rd(T(i, j-1, ly, lx + tile scale - eh) wa(lx, eh)) + rd(V wb(lx, eh)))
 *       when j > 0 and lx < eh, else V
 *
 * Output frame k < keep is clip frame skip + k, its first `blend` frames blended with `prev`
 * unless the chunk is its video's first, v = rd(rd(prev[k] wa(k, blend)) + rd(S wb(k, blend)));
 * prev then becomes the frames S(2 skip + keep + k'), k' < blend, at prev's dtype; with `last`
 * those frames follow the kept ones in y. A byte is round-half-even(255 clamp(f32(f32(v std[c]) +
 * mean[c]), 0, 1)), `mean` and `std` three f32 values each. y is u8 [frames_out, H scale,
 * W scale, 3] and prev [blend, H scale, W scale, 3]; y holds at least the frames' rows. */
int ref_vtile_stitch(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *geom = &a->t[1], *prev = &a->t[2], *y = &a->t[3];
    const int64_t tile = p_int(a, "tile", 0), overlap = p_int(a, "overlap", 0);
    const int64_t S = p_int(a, "scale", 0), pt = p_int(a, "pt", 0), skip = p_int(a, "skip", 0);
    const int64_t keep = p_int(a, "keep", 0), blend = p_int(a, "blend", 0);
    const int64_t period = p_int(a, "period", 0);
    std::vector<float> mean, stdv;
    if (tile <= 0 || S <= 0 || pt <= 0 || skip < 0 || keep <= 0 || blend < 0 || blend > keep ||
        parse_f32_list(p_str(a, "mean", ""), &mean) != 3 ||
        parse_f32_list(p_str(a, "std", ""), &stdv) != 3 || numel(geom) < 5 || y->dtype != RAD_U8)
        return RAD_E_SHAPE;
    const int64_t frames = ldi(geom, 0), H = ldi(geom, 1), W = ldi(geom, 2);
    const bool first = ldi(geom, 3) != 0, last = ldi(geom, 4) != 0;
    std::vector<int64_t> sy, sx, oy, ox;
    if (frames <= 0 || 2 * skip + keep + blend > frames * pt || period < frames * tile * tile ||
        !split_tiles(H, tile, overlap, &sy, &oy) || !split_tiles(W, tile, overlap, &sx, &ox))
        return RAD_E_SHAPE;
    const int64_t nh = (int64_t)sy.size(), nw = (int64_t)sx.size(), cols = 3 * pt * S * S;
    if (rows_of(x, cols) < nh * nw * period) return RAD_E_SHAPE;
    const int64_t Hp = H * S, Wp = W * S, npx = Hp * Wp;
    const int64_t fout = keep + (last ? blend : 0);
    if (rows_of(y, 3) < fout * npx || (blend > 0 && rows_of(prev, 3) < blend * npx))
        return RAD_E_SHAPE;
    /* The owner tile and the offset in it of every pixel row and column. */
    std::vector<int64_t> own_y((size_t)Hp), own_x((size_t)Wp);
    for (int64_t Y = 0, i = 0; Y < Hp; ++Y) {
        while (i + 1 < nh && Y >= sy[(size_t)(i + 1)] * S) ++i;
        own_y[(size_t)Y] = i;
    }
    for (int64_t X = 0, j = 0; X < Wp; ++X) {
        while (j + 1 < nw && X >= sx[(size_t)(j + 1)] * S) ++j;
        own_x[(size_t)X] = j;
    }
    const uint32_t dt = x->dtype;
    auto rd = [dt](float v) { return round_dt(dt, v); };
    auto wb = [&](int64_t p, int64_t e) { return rd((float)p / (float)e); };
    auto wa = [&](int64_t p, int64_t e) { return rd(1.0f - rd((float)p / (float)e)); };
    auto T = [&](int64_t i, int64_t j, int64_t ly, int64_t lx, int64_t fc, int64_t c) {
        const int64_t row = (i * nw + j) * period + ((fc / pt) * tile + ly / S) * tile + lx / S;
        const int64_t col = ((c * pt + fc % pt) * S + ly % S) * S + lx % S;
        return ldt(x, rowoff(x, row, cols) + col * laststride(x));
    };
    auto stitch = [&](int64_t fc, int64_t Y, int64_t X, int64_t c) {
        const int64_t i = own_y[(size_t)Y], j = own_x[(size_t)X];
        const int64_t ly = Y - sy[(size_t)i] * S, lx = X - sx[(size_t)j] * S;
        float v = T(i, j, ly, lx, fc, c);
        const int64_t ev = i > 0 ? oy[(size_t)(i - 1)] * S : 0;
        if (ly < ev) {
            const float up = T(i - 1, j, ly + tile * S - ev, lx, fc, c);
            v = rd(rd(up * wa(ly, ev)) + rd(v * wb(ly, ev)));
        }
        const int64_t eh = j > 0 ? ox[(size_t)(j - 1)] * S : 0;
        if (lx < eh) {
            const float left = T(i, j - 1, ly, lx + tile * S - eh, fc, c);
            v = rd(rd(left * wa(lx, eh)) + rd(v * wb(lx, eh)));
        }
        return v;
    };
    auto byte = [&](int64_t f, int64_t px, int64_t c, float v) {
        float p = v * stdv[(size_t)c] + mean[(size_t)c];
        p = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        stt(y, rowoff(y, f * npx + px, 3) + c * laststride(y), p * 255.0f);
    };
    REF_PAR_FOR(npx * 3 * (fout + blend) * 4)
    for (int64_t px = 0; px < npx; ++px) {
        const int64_t Y = px / Wp, X = px % Wp;
        for (int64_t c = 0; c < 3; ++c) {
            for (int64_t k = 0; k < keep; ++k) {
                float v = stitch(skip + k, Y, X, c);
                if (!first && k < blend) {
                    const int64_t pb = rowoff(prev, k * npx + px, 3);
                    const float pv = ldt(prev, pb + c * laststride(prev));
                    v = rd(rd(pv * wa(k, blend)) + rd(v * wb(k, blend)));
                }
                byte(k, px, c, v);
            }
            for (int64_t k = 0; k < blend; ++k) {
                const float v = stitch(2 * skip + keep + k, Y, X, c);
                if (last) byte(keep + k, px, c, v);
                stt(prev, rowoff(prev, k * npx + px, 3) + c * laststride(prev), v);
            }
        }
    }
    return RAD_OK;
}

/* ================================================================== the latent upscaler
 * A 3-D convolutional network over a video latent's voxel rows: row (t, i, j) of a T x H x W
 * volume is (t H + i) W + j, channels innermost (NDHWC), every operand at its own row pitch.
 * `geom` carries the request's volume; an output holds at least the rows the volume writes. */

/* ------------------------------------------------------------------ conv3d_ndhwc */
/* A 3-D convolution over the volume geom = (T, H, W). The defaults are stride 1 with `same` zero
 * padding: kt / 2 frames and k / 2 rows and columns on each side (kt and k odd); x
 * [>= T H W, cin] and y and `res` [>= rows out, cout]; w is PyTorch's Conv3d weight
 * [cout, cin, kt, k, k] and b [cout]:
 *
 *   acc = sum over (dt, kh, kw, ci) of w[o, ci, dt, kh, kw] x[(t + dt - kt/2, i + kh - k/2,
 *                                                              j + kw - k/2), ci]
 *   y[(t, i, j), o] = rd(acc + b[o])                    rd(rd(acc + b[o]) + res) with `res`
 *
 * terms outside the volume zero. acc is f32, the taps in (dt, kh, kw) order and each tap's
 * channels ascending; rd is y's dtype. With `res` the convolution's output is rounded before the
 * residual is added, as a residual block's skip(x) + h adds two bf16 tensors. `res` may be y
 * itself: each output element reads its own residual before its write, and no other.
 *
 * The optional `stride` (spatial) and `stride_t` pad first and then stride, `causal` 1 takes the
 * temporal padding front-only (kt - 1 zeros, none behind), `pad` `down` pads none above and left
 * and one below and right, and `reflect` 1 mirrors the spatial padding instead of zeroing it;
 * these CausalConv3d forms need k = kt = 3. Output (t, i, j) then reads
 * x[(t stride_t + dt - ft, i stride + kh - pt, j stride + kw - pl), ci] over the lengths
 *
 *   ((T + ft + bk - kt) / stride_t + 1) ((H + pt + pb - k) / stride + 1)
 *   ((W + pl + pr - k) / stride + 1)                                   (floor divisions)
 *
 * with ft / bk and pt / pb / pl / pr the front / back and top / bottom / left / right pads. The
 * time axis is zero-padded unless `tpad` says otherwise (below); `reflect` mirrors each spatial
 * axis once, torch's `reflect` (x[-n] = x[n], x[D + n] = x[D - 2 - n]), so each pad must be below
 * the axis it pads.
 *
 * `tpad` sets the frames' padding (rows and columns stay zero-padded): `zero` (the default),
 * `replicate` (the edge frame repeated, kt/2 a side: LTX-2's decoder) or `causal` (the first
 * frame repeated kt - 1 times in front, nothing after: LTX-2's encoder). With `causal`, `tpre`
 * more copies of the first frame stand in front of those, and y has T + tpre frames (LTX-2's
 * temporal downsampler, which prepends its first frame before its convolution). It folds into
 * the same grid math: replicate keeps ft = kt / 2 and clamps the time reads to the volume;
 * causal-`tpre` takes ft = kt - 1 + tpre and bk = 0, clamping below. `tpad` `causal` repeats
 * where `causal` 1 zeroes, and the two are refused together. */
static int64_t idiv_floor(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b && (a < 0) != (b < 0)) ? q - 1 : q;
}

int ref_conv3d_ndhwc(const RadArgs* a, RadStream) {
    if (a->n_t < 6 || !have(a, 4) || !t_in(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *b = &a->t[2], *geom = &a->t[3];
    const RadTensor *res = t_in(a, 4), *y = &a->t[5];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0);
    const int64_t k = p_int(a, "k", 0), kt = p_int(a, "kt", 0), tpre = p_int(a, "tpre", 0);
    static const char* const kPads[] = { "zero", "replicate", "causal" };
    const int tpad = p_enum(a, "tpad", kPads, 3, 0);
    if (tpad < 0 || (tpre && tpad != 2)) return RAD_E_INVAL;
    if (cin <= 0 || cout <= 0 || k <= 0 || kt <= 0 || !(k & 1) || !(kt & 1) || tpre < 0 ||
        numel(geom) < 3)
        return RAD_E_SHAPE;
    const int64_t sp = p_int(a, "stride", 1), st = p_int(a, "stride_t", 1);
    const int64_t causal = p_int(a, "causal", 0), reflect = p_int(a, "reflect", 0);
    const char* pads = p_str(a, "pad", "same");
    const bool down = pads && !std::strcmp(pads, "down");
    if (sp <= 0 || st <= 0 || (causal && causal != 1) || (reflect && reflect != 1) ||
        (causal && tpad) /* zero-causal and replicated-causal contradict */ ||
        !(!pads || !std::strcmp(pads, "same") || down))
        return RAD_E_SHAPE;
    const bool gen = sp != 1 || st != 1 || causal || down || reflect;
    if (gen && (k != 3 || kt != 3)) return RAD_E_SHAPE;
    const int64_t T = ldi(geom, 0), H = ldi(geom, 1), W = ldi(geom, 2);
    if (T <= 0 || H <= 0 || W <= 0) return RAD_E_SHAPE;
    const int64_t ft = causal ? kt - 1 : tpad == 2 ? kt - 1 + tpre : kt / 2;
    const int64_t bk = (causal || tpad == 2) ? 0 : kt / 2;
    const int64_t pt = down ? 0 : k / 2, pb = down ? 1 : k / 2;
    const int64_t pl = down ? 0 : k / 2, pr = down ? 1 : k / 2;
    if (reflect && (std::max(pt, pb) >= H || std::max(pl, pr) >= W)) return RAD_E_SHAPE;
    const int64_t T2 = idiv_floor(T + ft + bk - kt, st) + 1;
    const int64_t H2 = idiv_floor(H + pt + pb - k, sp) + 1;
    const int64_t W2 = idiv_floor(W + pl + pr - k, sp) + 1;
    if (T2 <= 0 || H2 <= 0 || W2 <= 0) return RAD_E_SHAPE;
    const int64_t rows = T * H * W, orows = T2 * H2 * W2, taps = kt * k * k;
    if (rows_of(y, cout) < orows || rows_of(x, cin) < rows ||
        (res && rows_of(res, cout) < orows) || numel(w) < cout * cin * taps || numel(b) < cout)
        return RAD_E_SHAPE;
    const int64_t xs = laststride(x), ys = laststride(y);
    const int64_t rs = res ? laststride(res) : 0;
    const uint32_t dt = y->dtype;
    REF_PAR_FOR(orows * cout * cin * taps)
    for (int64_t r = 0; r < orows; ++r) {
        const int64_t t = r / (H2 * W2), i = (r / W2) % H2, j = r % W2;
        const int64_t yb = rowoff(y, r, cout), rb = res ? rowoff(res, r, cout) : 0;
        for (int64_t o = 0; o < cout; ++o) {
            float acc = 0.0f;
            for (int64_t dz = 0; dz < kt; ++dz) {
                int64_t tz = t * st + dz - ft;
                if (tpad)   /* replicate and causal-tpre clamp; `zero`/`causal` pad */
                    tz = std::min<int64_t>(std::max<int64_t>(tz, 0), T - 1);
                if (tz < 0 || tz >= T) continue;
                for (int64_t kh = 0; kh < k; ++kh) {
                    int64_t iy = i * sp + kh - pt;
                    if (reflect) iy = iy < 0 ? -iy : (iy >= H ? 2 * H - 2 - iy : iy);
                    if (iy < 0 || iy >= H) continue;
                    for (int64_t kw = 0; kw < k; ++kw) {
                        int64_t ix = j * sp + kw - pl;
                        if (reflect) ix = ix < 0 ? -ix : (ix >= W ? 2 * W - 2 - ix : ix);
                        if (ix < 0 || ix >= W) continue;
                        const int64_t xb = rowoff(x, (tz * H + iy) * W + ix, cin);
                        const int64_t wb = o * cin * taps + (dz * k + kh) * k + kw;
                        for (int64_t ci = 0; ci < cin; ++ci)
                            acc += ldt(w, wb + ci * taps) * ldt(x, xb + ci * xs);
                    }
                }
            }
            const float v = acc + ldt(b, o);
            stt(y, yb + o * ys, res ? round_dt(dt, v) + ldt(res, rb + o * rs) : v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ group_norm */
/* PyTorch's nn.GroupNorm(groups, n) over [1, n, rows]: group g is channels
 * [g n / groups, (g + 1) n / groups) of every row, rows being y's. Its mean and biased variance
 * are over the group's rows n / groups values, summed in f64 in row-major order (two passes, the
 * variance about the f64 mean); then, in f32,
 *
 *   mean_g = f32(mean)   rstd_g = 1 / sqrt(f32(var) + eps)
 *   v = (x - mean_g) rstd_g w[c] + b[c]            each product and sum one f32 rounding
 *   v = v (1 + scale[c]) + shift[c]                with `mod`, one row [scale | shift] of 2n
 *   v = v / (1 + exp(-v))                          with `act` silu
 *
 * and v is rounded once into y. stats[2g] and stats[2g + 1] receive mean_g and rstd_g; `stats`
 * must hold groups (2 + 3 1024) values, the rest being a device kernel's scratch (partial sums of
 * up to 1024 parts a group), which this implementation does not touch. y may be x: the statistics
 * are complete before any row is written, and each element is read before its own write. */
int ref_group_norm(const RadArgs* a, RadStream) {
    if (a->n_t < 6 || !have(a, 3) || !t_in(a, 4) || !t_in(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *b = &a->t[2], *mod = t_in(a, 3);
    const RadTensor *st = &a->t[4], *y = &a->t[5];
    const int64_t n = p_int(a, "n", 0), G = p_int(a, "groups", 0);
    const float eps = p_f32(a, "eps", 1e-5f);
    static const char* const kActs[] = { "none", "silu" };
    const int act = p_enum(a, "act", kActs, 2, 0);
    if (act < 0 || act > 1) return RAD_E_INVAL;
    if (n <= 0 || G <= 0 || n % G || numel(w) < n || numel(b) < n ||
        (mod && numel(mod) < 2 * n) || numel(st) < G * (2 + 3 * 1024))
        return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n), cg = n / G;
    if (rows <= 0 || rows_of(x, n) < rows) return RAD_E_SHAPE;
    const int64_t xs = laststride(x), ys = laststride(y);
    const double cnt = (double)rows * (double)cg;
    std::vector<float> mean((size_t)G), rstd((size_t)G);
    REF_PAR_FOR(rows * n * 2)
    for (int64_t g = 0; g < G; ++g) {
        double s = 0.0;
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t xb = rowoff(x, r, n) + g * cg * xs;
            for (int64_t c = 0; c < cg; ++c) s += (double)ldt(x, xb + c * xs);
        }
        const double mu = s / cnt;
        double q = 0.0;
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t xb = rowoff(x, r, n) + g * cg * xs;
            for (int64_t c = 0; c < cg; ++c) {
                const double d = (double)ldt(x, xb + c * xs) - mu;
                q += d * d;
            }
        }
        mean[(size_t)g] = (float)mu;
        rstd[(size_t)g] = 1.0f / std::sqrt((float)(q / cnt) + eps);
    }
    for (int64_t g = 0; g < G; ++g) {
        stt(st, offlin(st, 2 * g), mean[(size_t)g]);
        stt(st, offlin(st, 2 * g + 1), rstd[(size_t)g]);
    }
    REF_PAR_FOR(rows * n * 4)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), yb = rowoff(y, r, n);
        for (int64_t c = 0; c < n; ++c) {
            const size_t g = (size_t)(c / cg);
            float v = (ldt(x, xb + c * xs) - mean[g]) * rstd[g];
            v = v * ldt(w, c) + ldt(b, c);
            if (mod) {
                const float sc = 1.0f + ldt(mod, offlin(mod, c));
                v = v * sc + ldt(mod, offlin(mod, n + c));
            }
            if (act) v = v / (1.0f + ref_exp(-v));
            stt(y, yb + c * ys, v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ tconv_dw */
/* A depthwise convolution along a volume's frames (a Conv3d of kernel (k, 1, 1), padding k / 2
 * frames, one group a channel): geom = (T, H, W), P = H W rows a frame, x and y [>= T P, n], w
 * PyTorch's [n, 1, k, 1, 1] (tap j of channel c at c k + j) and b [n], k odd:
 *
 *   y[(t, p), c] = rd(f32(sum_j w[c, j] x[(t + j - k/2, p), c]) + b[c])
 *
 * frames outside [0, T) zero, the sum f32 in ascending j. */
int ref_tconv_dw(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *b = &a->t[2], *geom = &a->t[3], *y = &a->t[4];
    const int64_t n = p_int(a, "n", 0), k = p_int(a, "k", 0);
    if (n <= 0 || k <= 0 || !(k & 1) || numel(geom) < 3 || numel(w) < n * k || numel(b) < n)
        return RAD_E_SHAPE;
    const int64_t T = ldi(geom, 0), H = ldi(geom, 1), W = ldi(geom, 2);
    if (T <= 0 || H <= 0 || W <= 0) return RAD_E_SHAPE;
    const int64_t P = H * W, rows = T * P;
    if (rows_of(y, n) < rows || rows_of(x, n) < rows) return RAD_E_SHAPE;
    const int64_t xs = laststride(x), ys = laststride(y);
    REF_PAR_FOR(rows * n * k)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t t = r / P, p = r % P, yb = rowoff(y, r, n);
        for (int64_t c = 0; c < n; ++c) {
            float acc = 0.0f;
            for (int64_t j = 0; j < k; ++j) {
                const int64_t tz = t + j - k / 2;
                if (tz < 0 || tz >= T) continue;
                acc += ldt(w, c * k + j) * ldt(x, rowoff(x, tz * P + p, n) + c * xs);
            }
            stt(y, yb + c * ys, acc + ldt(b, c));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ resize_bilinear */
/* Each frame of a volume resized on its own (the frame axis kept), as torch's
 * upsample_bilinear2d computes align_corners=False for an output size (not a scale factor):
 * geom = (T, h, w, H, W), x [>= T h w, n] and y [>= T H W, n]. For output row i of H, in f32,
 *
 *   sh = h / H   src = max(sh (i + 0.5) - 0.5, 0)   i0 = floor(src)   i1 = i0 + (i0 < h - 1)
 *   l1 = src - i0   l0 = 1 - l1
 *
 * and likewise for column j of W against w; then, every product and sum one f32 rounding and the
 * result rounded once into y,
 *
 *   v = l0y (l0x x[i0, j0] + l1x x[i0, j1]) + l1y (l0x x[i1, j0] + l1x x[i1, j1])
 *
 * MiniMax-H3's upscaler resizes trilinearly at an unchanged frame count, whose temporal weights
 * are exactly 1 and 0, so this is that resize. Any sizes, shrinking included. */
int ref_resize_bilinear(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *geom = &a->t[1], *y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || numel(geom) < 5) return RAD_E_SHAPE;
    const int64_t T = ldi(geom, 0), h = ldi(geom, 1), w = ldi(geom, 2);
    const int64_t H = ldi(geom, 3), W = ldi(geom, 4);
    if (T <= 0 || h <= 0 || w <= 0 || H <= 0 || W <= 0) return RAD_E_SHAPE;
    const int64_t rows = T * H * W;
    if (rows_of(y, n) < rows || rows_of(x, n) < T * h * w) return RAD_E_SHAPE;
    /* Output index `o` of an axis resized from `in` to `out`: its two source indices and
     * weights. */
    auto axis = [](int64_t o, int64_t in, int64_t out, int64_t* i0, int64_t* i1, float* l0,
                   float* l1) {
        const float s = (float)in / (float)out;
        float src = s * ((float)o + 0.5f) - 0.5f;
        if (src < 0.0f) src = 0.0f;
        *i0 = (int64_t)src;
        *i1 = *i0 + (*i0 < in - 1 ? 1 : 0);
        *l1 = src - (float)*i0;
        *l0 = 1.0f - *l1;
    };
    const int64_t xs = laststride(x), ys = laststride(y);
    REF_PAR_FOR(rows * n * 8)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t t = r / (H * W), i = (r / W) % H, j = r % W;
        int64_t i0, i1, j0, j1;
        float l0y, l1y, l0x, l1x;
        axis(i, h, H, &i0, &i1, &l0y, &l1y);
        axis(j, w, W, &j0, &j1, &l0x, &l1x);
        const int64_t b00 = rowoff(x, (t * h + i0) * w + j0, n);
        const int64_t b01 = rowoff(x, (t * h + i0) * w + j1, n);
        const int64_t b10 = rowoff(x, (t * h + i1) * w + j0, n);
        const int64_t b11 = rowoff(x, (t * h + i1) * w + j1, n);
        const int64_t yb = rowoff(y, r, n);
        for (int64_t c = 0; c < n; ++c) {
            const float top = l0x * ldt(x, b00 + c * xs) + l1x * ldt(x, b01 + c * xs);
            const float bot = l0x * ldt(x, b10 + c * xs) + l1x * ldt(x, b11 + c * xs);
            stt(y, yb + c * ys, l0y * top + l1y * bot);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ flow_renoise */
/* A clean latent taken back to noise level s with a fresh stream, as a sampler restarts from an
 * upscaled latent: over x's rows, in f32,
 *
 *   x[k, c] = f32(f32(f32(1 - s) x[k, c]) + f32(s z(k n + c)))
 *
 * z(e) = rad_philox_normal_s(seed, stream, e) (abi/rad_philox.h), `seed` two u32 words low first,
 * `stream` the parameter, s = sigma (one f32 value). x is stored at its own dtype (f32 in
 * MiniMax-H3) and row pitch. */
int ref_flow_renoise(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *sg = &a->t[0], *seed = &a->t[1], *x = &a->t[2];
    const int64_t n = p_int(a, "n", 0), stream = p_int(a, "stream", 0);
    if (n <= 0 || stream < 0 || numel(sg) < 1 || numel(seed) < 2) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0) return RAD_E_SHAPE;
    const float s = ldt(sg, offlin(sg, 0)), oms = 1.0f - s;
    const uint64_t sd = (uint64_t)(uint32_t)ldi(seed, 0) | ((uint64_t)(uint32_t)ldi(seed, 1) << 32);
    const int64_t xs = laststride(x);
    REF_PAR_FOR(rows * n * 8)
    for (int64_t k = 0; k < rows; ++k) {
        const int64_t xb = rowoff(x, k, n);
        for (int64_t c = 0; c < n; ++c) {
            const float z = rad_philox_normal_s(sd, (uint64_t)stream, (uint64_t)(k * n + c));
            stt(x, xb + c * xs, oms * ldt(x, xb + c * xs) + s * z);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ cond_aug */
/* A visual condition's rows blended with noise, as ComfyUI's `_cond_video_rows`
 * (comfy/ldm/minimax/model.py) does before they become condition rows. Element e = k n + c:
 *
 *   x[k, c] = f32(f32(a x[k, c]) + f32(oma z(e)))
 *
 * a = `aug` (VISUAL_COND_TIMESTEP, 0.999), oma the f32 of the f64 1 - aug (ComfyUI subtracts in
 * double), z(e) = rad_philox_normal_s(seed, stream, e) (abi/rad_philox.h), `seed` the request's
 * two u32 words, low first. x is f32 [M, n] at its row pitch. ComfyUI restarts the generator for
 * every condition and every forward, so the blend is a fixed function of the request and is
 * applied once, per media item. */
int ref_cond_aug(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *seed = &a->t[0], *x = &a->t[1];
    const int64_t n = p_int(a, "n", 0), stream = p_int(a, "stream", 0);
    const RadParam* ap = p_find(a, "aug");
    if (!ap || ap->kind != RAD_P_F64) return RAD_E_SHAPE;
    const double aug = ap->dval;
    if (n <= 0 || stream < 0 || !(aug >= 0.0 && aug < 1.0) || numel(seed) < 2) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0) return RAD_E_SHAPE;
    const float av = (float)aug, omv = (float)(1.0 - aug);
    const uint64_t sd = (uint64_t)(uint32_t)ldi(seed, 0) | ((uint64_t)(uint32_t)ldi(seed, 1) << 32);
    const int64_t xs = laststride(x);
    REF_PAR_FOR(rows * n * 8)
    for (int64_t k = 0; k < rows; ++k) {
        const int64_t xb = rowoff(x, k, n);
        for (int64_t c = 0; c < n; ++c) {
            const float z = rad_philox_normal_s(sd, (uint64_t)stream, (uint64_t)(k * n + c));
            stt(x, xb + c * xs, av * ldt(x, xb + c * xs) + omv * z);
        }
    }
    return RAD_OK;
}

/* ================================================================== the tiled encoder
 * A one-frame media larger than a tile (a reference picture or a keyframe) is encoded as vae.py's
 * `tiled_encode` does: the pixel plane is cut into split_tiles_axis' tiles, each goes through the
 * encoder alone, and the tiles' moments are blended into one latent plane. These ops are the plan,
 * the cut and the blend; the encoder is the per-level ops, issued per tile. Tile starts, extents
 * and counts come from `geom`, the request's (clip, H, W) on the device, never from the declared
 * parameters. */

/* ------------------------------------------------------------------ vtile_plan */
/* The tiles of one frame of `geom` = (clip, H, W) pixels: split_tiles_axis over each spatial axis
 * in latent units (`tile` a side, `overlap` the least blend, `ratio` pixels a latent), numbered
 * row-major. Row t of y (i32 [M, 22]):
 *
 *   [0] [1]     the tile's first pixel (y0, x0)
 *   [2] [3]     its pixel extent (th, tw): `tile` x ratio, except a lone short tile
 *   3 L + 4..6  level L's volume (1, th / 2^min(L, 4), tw / 2^min(L, 4)), L = 0..5: venc_geom's
 *               level layout for one frame (levels 4 and 5 share a size)
 *
 * `M` is the band's bound, a ceiling only (band_rows); y must hold the tiles. */
int ref_vtile_plan(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *geom = &a->t[0], *y = &a->t[1];
    const int64_t band = p_int(a, "M", 0), tile = p_int(a, "tile", 0);
    const int64_t overlap = p_int(a, "overlap", 0), ratio = p_int(a, "ratio", 0);
    if (tile <= 0 || overlap < 0 || overlap >= tile || ratio <= 0 || numel(geom) < 3)
        return RAD_E_SHAPE;
    const int64_t H = ldi(geom, 1), W = ldi(geom, 2);
    std::vector<int64_t> sy, sx, oy, ox;
    if (H <= 0 || W <= 0 || H % ratio || W % ratio ||
        !split_tiles_axis(H / ratio, tile, overlap, &sy, &oy) ||
        !split_tiles_axis(W / ratio, tile, overlap, &sx, &ox))
        return RAD_E_SHAPE;
    const int64_t nh = (int64_t)sy.size(), nw = (int64_t)sx.size();
    const int64_t M = band_rows(nh * nw, band);
    if (M <= 0 || rows_of(y, 22) < M) return RAD_E_SHAPE;
    REF_PAR_FOR(M)
    for (int64_t t = 0; t < M; ++t) {
        const int64_t i = t / nw, j = t % nw;
        const int64_t th = std::min(tile, H / ratio - sy[(size_t)i]) * ratio;
        const int64_t tw = std::min(tile, W / ratio - sx[(size_t)j]) * ratio;
        const int64_t yb = rowoff(y, t, 22), ys = laststride(y);
        const int64_t v[22] = { sy[(size_t)i] * ratio, sx[(size_t)j] * ratio, th, tw,
                                1, th, tw,   1, th / 2, tw / 2,   1, th / 4, tw / 4,
                                1, th / 8, tw / 8,   1, th / 16, tw / 16,   1, th / 16, tw / 16 };
        for (int64_t c = 0; c < 22; ++c) st_int(y->dtype, y->data, yb + c * ys, v[c]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vtile_crop */
/* One frame's pixel plane cut into vtile_plan's tiles: x f32 [>= H W, 3] is the plane (pixel
 * (Y, X) at row Y W + X), geom = (clip, H, W) in pixels; y f32 [M, 3] holds the tiles' pixels
 * tile-major, each tile's rows in plane order. The tiles overlap, so y is larger than x and a
 * buffer of its own. `M` is a ceiling only (band_rows). */
int ref_vtile_crop(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *geom = &a->t[1], *y = &a->t[2];
    const int64_t band = p_int(a, "M", 0), tile = p_int(a, "tile", 0);
    const int64_t overlap = p_int(a, "overlap", 0), ratio = p_int(a, "ratio", 0);
    if (tile <= 0 || overlap < 0 || overlap >= tile || ratio <= 0 || numel(geom) < 3)
        return RAD_E_SHAPE;
    const int64_t H = ldi(geom, 1), W = ldi(geom, 2);
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
    if (M <= 0 || rows_of(x, 3) < H * W || rows_of(y, 3) < M) return RAD_E_SHAPE;
    REF_PAR_FOR(M * 3)
    for (int64_t r = 0; r < M; ++r) {
        const int64_t t = (int64_t)(std::upper_bound(off.begin(), off.end(), r) - off.begin()) - 1;
        const int64_t i = t / nw, j = t % nw;
        const int64_t tw = std::min(tile, W / ratio - sx[(size_t)j]) * ratio;
        const int64_t k = r - off[(size_t)t];
        const int64_t src = (sy[(size_t)i] * ratio + k / tw) * W + sx[(size_t)j] * ratio + k % tw;
        const int64_t xb = rowoff(x, src, 3), yb = rowoff(y, r, 3), xs = laststride(x);
        const int64_t ys = laststride(y);
        for (int64_t c = 0; c < 3; ++c) stt(y, yb + c * ys, ldt(x, xb + c * xs));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vtile_join */
/* The tiles' moments blended into one latent plane, as vae.py's `tiled_encode` does: x holds the
 * tiles' n-channel moments tile-major (vtile_crop's layout at latent extent), geom = (clip, lat_h,
 * lat_w) in latents, and y f32 or bf16 [M, n] is the canvas, M = lat_h lat_w. Position (Y, X)
 * belongs to the tile (i, j) whose range [start, next start) holds it, at (ly, lx) within it;
 * T(i, j, ly, lx) is that tile's value. The neighbours are read unblended and the top blend comes
 * before the left one; every product and sum rounds to x's dtype (`rd`), with
 * wa(p, e) = rd(1 - rd(p / e)) and wb(p, e) = rd(p / e):
 *
 *   v = T(i, j, ly, lx)
 *   v = rd(rd(T(i - 1, j, ly + tile - ev) * wa(ly, ev)) + rd(v * wb(ly, ev)))  ev = ov_y[i - 1]
 *   v = rd(rd(T(i, j - 1, lx + tile - eh) * wa(lx, eh)) + rd(v * wb(lx, eh)))  eh = ov_x[j - 1]
 *
 * the first when i > 0 and ly < ev, the second when j > 0 and lx < eh. Only a lone tile is short,
 * and it has no neighbour, so a neighbour is always `tile` long on the blended axis. `M` is a
 * ceiling only (band_rows). */
int ref_vtile_join(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *geom = &a->t[1], *y = &a->t[2];
    const int64_t band = p_int(a, "M", 0), tile = p_int(a, "tile", 0);
    const int64_t overlap = p_int(a, "overlap", 0), n = p_int(a, "n", 0);
    if (tile <= 0 || overlap < 0 || overlap >= tile || n <= 0 || numel(geom) < 3)
        return RAD_E_SHAPE;
    const int64_t H = ldi(geom, 1), W = ldi(geom, 2);
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
    if (M <= 0 || rows_of(x, n) < off.back() || rows_of(y, n) < M) return RAD_E_SHAPE;
    /* The owner tile of every canvas row and column. */
    std::vector<int64_t> own_y((size_t)H), own_x((size_t)W);
    for (int64_t Y = 0, i = 0; Y < H; ++Y) {
        while (i + 1 < nh && Y >= sy[(size_t)(i + 1)]) ++i;
        own_y[(size_t)Y] = i;
    }
    for (int64_t X = 0, j = 0; X < W; ++X) {
        while (j + 1 < nw && X >= sx[(size_t)(j + 1)]) ++j;
        own_x[(size_t)X] = j;
    }
    const uint32_t dt = x->dtype;
    auto rd = [dt](float v) { return round_dt(dt, v); };
    auto wb = [&](int64_t p, int64_t e) { return rd((float)p / (float)e); };
    auto wa = [&](int64_t p, int64_t e) { return rd(1.0f - rd((float)p / (float)e)); };
    auto T = [&](int64_t i, int64_t j, int64_t ly, int64_t lx, int64_t c) {
        const int64_t row = off[(size_t)(i * nw + j)] + ly * std::min(tile, W - sx[(size_t)j]) + lx;
        return ldt(x, rowoff(x, row, n) + c * laststride(x));
    };
    const int64_t ys = laststride(y);
    REF_PAR_FOR(M * n)
    for (int64_t r = 0; r < M; ++r) {
        const int64_t Y = r / W, X = r % W;
        const int64_t i = own_y[(size_t)Y], j = own_x[(size_t)X];
        const int64_t ly = Y - sy[(size_t)i], lx = X - sx[(size_t)j];
        const int64_t ev = i > 0 ? oy[(size_t)(i - 1)] : 0, eh = j > 0 ? ox[(size_t)(j - 1)] : 0;
        const int64_t yb = rowoff(y, r, n);
        for (int64_t c = 0; c < n; ++c) {
            float v = T(i, j, ly, lx, c);
            if (ly < ev)
                v = rd(rd(T(i - 1, j, ly + tile - ev, lx, c) * wa(ly, ev)) +
                       rd(v * wb(ly, ev)));
            if (lx < eh)
                v = rd(rd(T(i, j - 1, ly, lx + tile - eh, c) * wa(lx, eh)) +
                       rd(v * wb(lx, eh)));
            stt(y, yb + c * ys, v);
        }
    }
    return RAD_OK;
}
