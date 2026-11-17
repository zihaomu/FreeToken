/* rad_block_sam.h -- Unlimited-OCR's DeepEncoder: SAM ViT-B/16, then CLIP ViT-L/14 on its map.
 *
 * The reference is the checkpoint's own deepencoder.py, used by
 * modeling_unlimitedocr.py's image branch:
 *
 *   sam_model(SxS picture, S in {1024, 640})                      -> [1, S/64, S/64, 1024]
 *     patch_embed: conv 16x16 stride 16 (768 out), + pos_embed resampled to the patch grid:
 *       get_abs_pos_sam returns the [1,64,64,768] table unchanged at 64, and bicubic-antialias
 *       resamples it (fp32, cast back to the model dtype) at any other grid -- 40 for a 640 tile
 *     12 blocks, w 768, 12 heads x 64: LayerNorm, qkv, attention with DECOMPOSED RELATIVE
 *       POSITION (rel_pos_h/rel_pos_w, 27 rows) inside a 14x14 window of the patch grid
 *       (padded to a multiple of 14: 70x70 for a 64 grid, 42x42 for a 40 grid), and one
 *       whole-grid window on blocks 2, 5, 8 and 11 (tables of 2*grid-1 rows; get_rel_pos
 *       resizes the 127-row table to 79 linearly at a 40 grid -- the `sam_attn` op does it);
 *       proj; LayerNorm; mlp 3072 exact gelu; mlp proj
 *     neck: conv 1x1 256 (no bias), LayerNorm2d 256, conv 3x3 256 (no bias), LayerNorm2d 256
 *     net_2: conv 3x3 stride 2 -> 512 at half (no bias); net_3: -> 1024 at a quarter (no bias)
 *   vision_model(patches=the sam map)                             -> CLIP ViT-L/14, M*M+1 tokens
 *     class token + the map as patch_embeds (NO pixels) + position_embedding, whose 257 rows
 *       (cls + a 16x16 grid) get_abs_pos resamples bicubic-antialias to the map's grid (cls row 0
 *       passthrough) -> 101 rows at a 640 tile;
 *     pre_layrnorm eps 1e-5; 24 layers, w 1024, 16 heads x 64: LayerNorm, fused qkv, dense
 *       attention over all tokens, out_proj; LayerNorm; fc1 4096 quick_gelu; fc2
 *   fuse: cat(clip rows 1:, the map) -> 2048 -> the projector linear -> 1280
 *   layout: every map line of M rows is closed by `image_newline`, the view by
 *           `view_seperator` -> M*(M+1)+1 rows (273 for a 1024 view, 110 for a 640 tile)
 *
 * ============================== The one op the engine must add ==============================
 *
 * SAM's attention logits are
 *
 *     scale * q.k  +  q.Rh[i_h, j_h]  +  q.Rw[i_w, j_w]
 *
 * with Rh/Rw read from the relative-position tables at the (query, key) grid offsets. The
 * per-pair term is a dot of the query with a vector that depends on BOTH rows, so it folds into
 * neither q, k nor v, and no declared op takes a bias operand (attn_dense's operands are
 * q, k, v, cu, out). This block therefore declares its attention as `sam_attn`:
 *
 *   params    M(range), head_dim, n_head, grid_h, grid_w, window, scale(f64), dtype
 *   operands  q, k, v, cu_seqlens (i32), out     -- rows n_head*head_dim each
 *   weights   rel_pos_h, rel_pos_w               -- [2*win-1 or 2*grid-1, head_dim]
 *   for each cu segment of grid_h*grid_w rows in row-major order: with `window` > 0 pad the grid
 *   to multiples of `window` with zero rows (deepencoder's window_partition pads AFTER norm1;
 *   the pad rows take part in the softmax exactly as the reference's do), cut it into
 *   window x window tiles, each tile attending to all of its rows; with `window` 0 one tile is
 *   the whole segment. logits as above, softmax in f32 over the tile's keys, out = sum_j p v_j,
 *   written back in the segment's row order with the pad rows dropped. Rh is deepencoder's
 *   get_rel_pos(table, q_h, k_h): the table at (i_h - j_h + len - 1), linearly interpolated when
 *   its length is not 2*max(q_h,k_h)-1 (the 127-row global table resizes to 79 at a 40 grid).
 *
 * probe() answers whether that op resolves. When it does not, declare() still maps and declares
 * every weight (a converted container carries the tower) but declares NO tower op and NO encoder,
 * which leaves the plugin text-only. Dropping the tables instead is not a usable
 * approximation: without them the tower's rows reach only cosine 0.35-0.77 to the reference's.
 *
 * ============================== Rows and patches: two geometries ==============================
 *
 * core/mm stages one view as a G x (G + 4) grid of 16x16 patch rows (G = 64 for a 1024 view,
 * 40 for a 640 tile; a gundam page stages its tiles and its global view in one item): the last
 * kDown = 4 patch columns of every row are zeros the encoder must not read. The patch embed
 * therefore runs one grid row at a time -- a G-row GEMM reading that row's first G staged patch
 * rows where they lie -- so the tower sees exactly the reference's raster GxG grid. A view's
 * rows are then written with an `image_newline` row closing each map line, and the view
 * separator's segment (an empty kDown x kDown grid) one `view_seperator` row.
 *
 * The two geometries are declared as two towers over the same weights (`Geom64`, `Geom40`);
 * step() picks one per pass from the pass's segment heights, which the pass packer keeps
 * uniform (core/engine_mm.cpp breaks a pass when a segment's grid differs). The 40 tower's
 * absolute position tables are the 64/16 tables resampled bicubic+antialias (arch/common/
 * rad_resample.h, pinned by tests/resample_golden.py against torch), computed host-side on a
 * rank's first tile pass from its own `sam.pos_embed` / `clip.pos_embed` weight planes and
 * staged into PERSIST buffers; everything else is the same graph at a G of 40.
 */
#ifndef RAD_BLOCK_SAM_H
#define RAD_BLOCK_SAM_H

#include <arch/rad_arch.h>
#include <arch/rad_resample.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace rad {
namespace arch {

struct DeepEncoder {
    /* ---- the reference's geometry, shared by both view sizes. */
    static constexpr int64_t kPatch     = 16;   /* SAM's patch size in pixels */
    static constexpr int64_t kDown      = 4;    /* patch columns per map token (net_2 x net_3) */
    static constexpr int64_t kWin       = 14;   /* SAM's window on the patch grid */
    static constexpr int64_t kSamW      = 768;
    static constexpr int64_t kSamHeads  = 12;
    static constexpr int64_t kSamHd     = 64;
    static constexpr int64_t kSamDepth  = 12;
    static constexpr int64_t kSamFf     = 3072; /* mlp_ratio 4 */
    static constexpr int64_t kClipW     = 1024;
    static constexpr int64_t kClipHeads = 16;
    static constexpr int64_t kClipHd    = 64;
    static constexpr int64_t kClipDepth = 24;
    static constexpr int64_t kClipFf    = 4096;
    static constexpr int64_t kClipSrcMap = 16;  /* the checkpoint's 16x16 CLIP table grid */
    static constexpr int64_t kClipSrcTok = kClipSrcMap * kClipSrcMap + 1;   /* 257 */
    static constexpr int64_t kFused     = 2048;
    static constexpr int64_t kNEmbd     = 1280;
    static constexpr int64_t kPatchDim  = 3 * kPatch * kPatch;        /* 768: a staged patch row */
    static constexpr int64_t kMerge     = kDown * kDown;              /* 16 patches a row */
    /* Tiles one tower invocation batches. max_patches/kSamTok bounds a pass (10 at the 40 grid,
     * 4 at 64 for the 16384-patch default); the cap keeps the activation set modest. */
    static constexpr int64_t kBatchCap  = 8;

    static bool global_block(int64_t i) { return i == 2 || i == 5 || i == 8 || i == 11; }

    /* ---- one view geometry. The 1024 view is the checkpoint's trained grid; the 640 tile is
     * the gundam mode's crop / the base mode's page, whose position tables are resampled. */
    struct Geom64 {
        static constexpr int64_t kGrid  = 64;   /* 1024 / 16 patches a side */
        static constexpr int64_t kMap   = kGrid / kDown;                 /* 16 */
        static constexpr int64_t kPadCols = (kWin - kGrid % kWin) % kWin; /* 6 */
        static constexpr int64_t kGridP = kGrid + kPadCols;               /* 70 */
        static constexpr int64_t kSamTok  = kGrid * kGrid;                /* 4096 */
        static constexpr int64_t kSamTokP = kGridP * kGridP;              /* 4900 */
        static constexpr int64_t kClipTok = kMap * kMap + 1;              /* 257 */
        static constexpr int64_t kRowsView  = kMap * (kMap + 1) + 1;      /* 273 */
        static constexpr int64_t kPatchView = kGrid * (kGrid + kDown);    /* 4352 */
        static constexpr int64_t kSegView   = kPatchView + kDown * kDown; /* 4368 */
    };
    struct Geom40 {
        static constexpr int64_t kGrid  = 40;   /* 640 / 16 patches a side */
        static constexpr int64_t kMap   = kGrid / kDown;                  /* 10 */
        static constexpr int64_t kPadCols = (kWin - kGrid % kWin) % kWin; /* 2 */
        static constexpr int64_t kGridP = kGrid + kPadCols;               /* 42 */
        static constexpr int64_t kSamTok  = kGrid * kGrid;                /* 1600 */
        static constexpr int64_t kSamTokP = kGridP * kGridP;              /* 1764 */
        static constexpr int64_t kClipTok = kMap * kMap + 1;              /* 101 */
        static constexpr int64_t kRowsView  = kMap * (kMap + 1) + 1;      /* 111: 110 + sep row */
        static constexpr int64_t kPatchView = kGrid * (kGrid + kDown);    /* 1760 */
        static constexpr int64_t kSegView   = kPatchView + kDown * kDown; /* 1776 */
    };

    template <typename G>
    struct SamLayerT {
        rad_weight w_n1 = 0, b_n1 = 0, w_qkv = 0, b_qkv = 0, w_proj = 0, b_proj = 0;
        rad_weight w_rh = 0, w_rw = 0;
        rad_weight w_n2 = 0, b_n2 = 0, w_l1 = 0, b_l1 = 0, w_l2 = 0, b_l2 = 0;
        rad_op op_n1 = 0, op_qkv = 0, op_attn = 0, op_proj = 0, op_n2 = 0, op_l1 = 0, op_l2 = 0;
        rad_op op_padd = 0, op_ladd = 0;   /* the residual adds, rounded apart from the GEMM */
        rad_op op_qkvw = 0, op_attnw = 0;   /* the windowed path, over the padded grid */
        rad_op op_gpad = 0, op_greal = 0;   /* into window_partition's order, and back */
        int64_t rel_rows = 0;
        int64_t window = 0;      /* 0 = the whole grid */
        /* The 40 tower's layers read the same checkpoint weights as the 64 tower's. */
        void copy_weights_from(const SamLayerT<Geom64>& o) {
            w_n1 = o.w_n1; b_n1 = o.b_n1; w_qkv = o.w_qkv; b_qkv = o.b_qkv; w_proj = o.w_proj;
            b_proj = o.b_proj; w_rh = o.w_rh; w_rw = o.w_rw; w_n2 = o.w_n2; b_n2 = o.b_n2;
            w_l1 = o.w_l1; b_l1 = o.b_l1; w_l2 = o.w_l2; b_l2 = o.b_l2;
            rel_rows = o.rel_rows; window = o.window;
        }
    };
    template <typename G>
    struct ClipLayerT {
        rad_weight w_n1 = 0, b_n1 = 0, w_qkv = 0, b_qkv = 0, w_out = 0, b_out = 0;
        rad_weight w_n2 = 0, b_n2 = 0, w_fc1 = 0, b_fc1 = 0, w_fc2 = 0, b_fc2 = 0;
        rad_op op_n1 = 0, op_qkv = 0, op_attn = 0, op_out = 0;
        rad_op op_cadd = 0, op_madd = 0;   /* the residual adds, rounded apart from the GEMM */
        rad_op op_n2 = 0, op_fc1 = 0, op_qg = 0, op_fc2 = 0;
        void copy_weights_from(const ClipLayerT<Geom64>& o) {
            w_n1 = o.w_n1; b_n1 = o.b_n1; w_qkv = o.w_qkv; b_qkv = o.b_qkv; w_out = o.w_out;
            b_out = o.b_out; w_n2 = o.w_n2; b_n2 = o.b_n2; w_fc1 = o.w_fc1; b_fc1 = o.b_fc1;
            w_fc2 = o.w_fc2; b_fc2 = o.b_fc2;
        }
    };

    /* ------------------------------------------------------------------ one tower a geometry
     * All of the tower's OPS and ACTIVATIONS; the weights live in DeepEncoder (one checkpoint
     * tensor a name, both geometries read the same planes). */
    /* One staged tile of a pass: segment `patch0`/`gw`, tile `t` of `n` in it, its output row
     * block, and whether this tile closes a line (the newline row follows its map). */
    struct Tile {
        int64_t patch0 = 0, gw = 0, t = 0, n = 0, rowbase = 0;
        bool nl = false;
    };

    template <typename G>
    struct Tower {
        std::vector<SamLayerT<G>>  sam;
        std::vector<ClipLayerT<G>> clip;

        /* pass constants */
        rad_op op_cu_sam = 0, op_cu_clip = 0, op_cu_pad = 0;   /* segment bounds tables */
        rad_buf b_cu_sam = 0, b_cu_clip = 0, b_cu_pad = 0;

        /* stem, neck, layout ops */
        rad_op op_patch = 0, op_sampos = 0, op_sampadd = 0;
        rad_op op_neck0 = 0, op_nk1 = 0, op_unf3 = 0, op_nk2 = 0, op_nk3 = 0;
        rad_op op_unf_net2 = 0, op_net2 = 0, op_unf_net3 = 0, op_net3 = 0;
        rad_op op_cls = 0, op_cmap = 0, op_cpos = 0, op_cadd = 0, op_playr = 0;
        rad_op op_ccopy = 0; /* pre_layrnorm output into the residual stream */
        rad_op op_fuse_clip = 0, op_fuse_sam = 0, op_proj = 0;
        rad_op op_dbg1024 = 0, op_dbg1280 = 0, op_dbg768 = 0, op_dbg256 = 0;
        rad_op op_newline = 0, op_sep = 0;

        /* activations, one batched run's: `B` tiles, tile-major (tile t's rows at
         * [t*kSamTok, (t+1)*kSamTok)). */
        rad_buf b_x = 0, b_h = 0, b_qkv = 0, b_att = 0, b_ff = 0, b_sampos = 0;
        rad_buf b_nlrow = 0;    /* the newline row, computed once, pitched into the layout */
        rad_buf b_pixstg = 0;   /* a tile's patch rows, packed for the single patch-embed GEMM */
        /* The windowed path's padded grid: h gathered into window_partition's order with the pad
         * rows zero (so the qkv linear yields the reference's bias on them), then qkv, then
         * attention. `hw`/`attw` hold kSamTokP rows. */
        rad_buf b_hw = 0, b_qkvw = 0, b_attw = 0;
        rad_buf b_ids_pad = 0, b_ids_real = 0;   /* pad position -> real row (-1 pad); the inverse */
        rad_buf b_n0 = 0, b_n1 = 0, b_unf3 = 0, b_n2 = 0, b_n3 = 0;
        rad_buf b_unf_2 = 0, b_map2 = 0, b_unf_3 = 0, b_map = 0;
        rad_buf b_c = 0, b_ch = 0, b_cq = 0, b_ca = 0, b_cf = 0, b_cg2 = 0;
        rad_buf b_cpos = 0, b_fuse = 0, b_projbuf = 0;
        rad_buf b_ct = 0;   /* a CLIP block's projection output, before its residual add */
        /* CLIP's pre_layrnorm output: the first layer's norm1 reads this, every later one the
         * residual stream (deepencoder's VitModel normalises once, before the transformer). */
        rad_buf b_cp = 0;
        rad_buf b_ids_sam = 0, b_ids_clip = 0;

        /* The 40 tower's resampled absolute tables (PERSIST, staged once a rank by fill_tables;
         * the gathers read them as plain operands). The 64 tower gathers straight from the
         * checkpoint's weight planes and leaves these null. */
        rad_buf b_pos_tab = 0, b_cpos_tab = 0;
        rad_buf p_one = 0, p_one_c = 0, p_out = 0;   /* shared planes, set by declare */

        mutable uint32_t tabs_done = 0;   /* bit per rank: the resampled tables are staged */
        mutable uint32_t ids_done = 0;    /* bit per rank: the position/index tables are staged */
        int64_t Bmax = 1;   /* tiles one invocation batches; every per-view plane is Bmax of them */

        static constexpr int64_t kGrid      = G::kGrid;
        static constexpr int64_t kMap       = G::kMap;
        static constexpr int64_t kGridP     = G::kGridP;
        static constexpr int64_t kSamTok    = G::kSamTok;
        static constexpr int64_t kSamTokP   = G::kSamTokP;
        static constexpr int64_t kClipTok   = G::kClipTok;
        static constexpr int64_t kRowsView  = G::kRowsView;
        static constexpr int64_t kPatchView = G::kPatchView;
        static constexpr int64_t kSegView   = G::kSegView;

        int declare(RadBuilder* b, Names& nm, DeepEncoder& w, rad_buf out,
                    rad_buf one, rad_buf one_c);
        void fill_tables(RadCtx* c, const DeepEncoder& w) const;
        void stage_ids(RadCtx* c) const;
        /* One batched tower run over `nt` (<= Bmax) staged tiles. */
        void batch(RadCtx* c, const RadBatch* batch, const Tile* tiles, int64_t nt,
                   bool tap, const DeepEncoder& w);
    };

    Tower<Geom64> t64;
    Tower<Geom40> t40;

    /* stems, neck, projector, layout -- one checkpoint tensor a name, read by both towers. */
    rad_weight w_patch = 0, b_patch = 0, w_pos = 0;
    rad_weight w_neck0 = 0, w_nk1 = 0, b_nk1 = 0, w_nk2 = 0, w_nk3 = 0, b_nk3 = 0;
    rad_weight w_net2 = 0, w_net3 = 0;
    rad_weight w_cls = 0, w_cpos = 0, w_playr = 0, b_playr = 0;
    rad_weight w_proj = 0, b_proj = 0, w_newline = 0, w_sep = 0;

    /* shared constants: the three 1-row planes and their fills, issued by every pass. */
    rad_op op_fill_one = 0, op_fill_one_c = 0;
    rad_buf b_one = 0, b_one_c = 0;
    rad_buf b_out = 0;
    int64_t max_patches = 0;
    int64_t max_views40 = 0;
    int64_t max_views64 = 0;
    bool    on = false;
    /* The stage taps belong to the first pass only. step()/fill_tables() run on the rank's own
     * step thread (one thread per DeepEncoder), so the check-then-set is safe as it stands; the
     * atomic makes it safe by construction too (debug-only path, so the cost is irrelevant). The
     * holder wraps the assignment in a relaxed store so the struct stays assignable: declare()
     * resets a rank's Model with `m = Model{}`, which a bare std::atomic<bool> member deletes. */
    struct TapFlag {
        mutable std::atomic<bool> v{false};
        TapFlag() = default;
        TapFlag(const TapFlag& o) : v(o.v.load(std::memory_order_relaxed)) {}
        TapFlag& operator=(const TapFlag& o) {
            v.store(o.v.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }
        TapFlag& operator=(bool x) {
            v.store(x, std::memory_order_relaxed);
            return *this;
        }
        operator bool() const { return v.load(std::memory_order_relaxed); }
    };
    mutable TapFlag tapped_;

    /* ------------------------------------------------------------------ stage taps
     * RADIANCE_DEBUG_MM_BUFFERS names intermediate stages (comma separated; a name matches a tag
     * by substring) and RADIANCE_DEBUG_MM_DUMP's directory receives <tag>.bf16: raw bf16 rows, one
     * file a stage, for a stage-by-stage comparison against a reference encoder. The copy is
     * issued on the pass's stream behind the producing op and synced, so it reads the op's output
     * (unlike the after-pass dump in core/engine_mm.cpp, which sees each buffer's last writer). */
    static const char* tap_dir() {
        static const char* const dir = std::getenv("RADIANCE_DEBUG_MM_DUMP");
        return dir;
    }
    static bool tap_names(const char** list, size_t* n) {
        static const char* const env = std::getenv("RADIANCE_DEBUG_MM_BUFFERS");
        const char* const dir = tap_dir();
        *list = env;
        *n = 0;
        if (!env || !*env || !dir || !*dir) return false;
        return true;
    }
    static bool tap_wants(const char* tag) {
        const char* list; size_t n;
        if (!tap_names(&list, &n)) return false;
        const size_t tl = std::strlen(tag);
        for (const char* s = list; *s; ) {
            const char* e = s;
            while (*e && *e != ',') ++e;
            const size_t l = (size_t)(e - s);
            if (l) {
                for (size_t i = 0; i + l <= tl; ++i)
                    if (std::strncmp(tag + i, s, l) == 0) return true;
            }
            s = *e ? e + 1 : e;
        }
        return false;
    }
    void tap(RadCtx* c, const char* tag, rad_buf b, int64_t rows, int64_t cols,
             int64_t off = 0) const {
        if (rows <= 0 || cols <= 0 || rad_rank(c) != 0 || !tap_wants(tag)) return;
        void* p = rad_buf_ptr(c, b);
        if (!p) return;
        std::vector<uint16_t> h((size_t)(rows * cols));
        if (rad_memcpy_async(h.data(), (const char*)p + off * cols * 2,
                             (size_t)(rows * cols * 2), rad_stream(c)) < 0) return;
        rad_stream_sync(rad_stream(c));
        char path[512];
        std::snprintf(path, sizeof path, "%s/tap.%s.bf16", tap_dir(), tag);
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(h.data(), sizeof(uint16_t), h.size(), f);
            std::fclose(f);
        }
    }

    /* ------------------------------------------------------------------ probe / weights */
    /* SAM's attention has no op unless one is added (the note above states it). */
    static bool probe(RadBuilder* b) { return rad_op_declared(b, "sam_attn") != 0; }

    static int wcopy(RadBuilder* b, const char* dn, const char* ck,
                     std::initializer_list<int64_t> sh, rad_weight* w, RadWeightGroup g,
                     int access = RAD_ACCESS_PER_REQUEST) {
        RAD_ARCH_TRY(map_copy(b, dn, ck));
        *w = decl_w(b, dn, RAD_BF16, sh, access, RAD_SHARD_NONE, g);
        return *w ? RAD_OK : RAD_E_INVAL;
    }

    /* Every vision tensor, at the checkpoint's names. Always declared, whether or not the tower
     * runs, so a converted container is complete for the pass that will. */
    int declare_weights(RadBuilder* b, Names& nm, int grp_base) {
        int rc;
        const RadWeightGroup gm = grp_model();
        if ((rc = wcopy(b, nm.f("sam.patch_embed.weight"),
                        nm.ckpt("model.sam_model.patch_embed.proj.weight"), {kSamW, kPatchDim},
                        &w_patch, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.patch_embed.bias"),
                        nm.ckpt("model.sam_model.patch_embed.proj.bias"), {kSamW}, &b_patch,
                        gm)) != RAD_OK) return rc;
        /* pos_embed is [1, 64, 64, 768]: one bf16 plane reshaped, which decl_weight admits.
         * VOCAB access (never streamed out): the 40 tower reads it back host-side to resample
         * its tile table. */
        if ((rc = wcopy(b, nm.f("sam.pos_embed"), nm.ckpt("model.sam_model.pos_embed"),
                        {Geom64::kSamTok, kSamW}, &w_pos, gm, RAD_ACCESS_VOCAB)) != RAD_OK)
            return rc;

        t64.sam.assign((size_t)kSamDepth, SamLayerT<Geom64>{});
        t40.sam.assign((size_t)kSamDepth, SamLayerT<Geom40>{});
        for (int64_t i = 0; i < kSamDepth; ++i) {
            SamLayerT<Geom64>& l = t64.sam[(size_t)i];
            SamLayerT<Geom40>& l40 = t40.sam[(size_t)i];
            const RadWeightGroup g = grp_layer(grp_base + (int)i);
            const long long L = (long long)i;
            /* The windowed blocks' 27-row tables are exact at any grid padded to 14; the global
             * blocks' 127-row tables resize to 2*G-1 (79 at a 40 grid) inside sam_attn. */
            l.rel_rows = global_block(i) ? 2 * Geom64::kGrid - 1 : 2 * kWin - 1;
            l.window   = global_block(i) ? 0 : kWin;
            /* One checkpoint tensor, its declared name and shape, shared by both towers. */
            auto SW = [&](const char* dn, const char* ck, std::initializer_list<int64_t> sh,
                          rad_weight* w) -> int {
                char d[96], k[160];
                std::snprintf(d, sizeof d, "sam.blk.%lld.%s", L, dn);
                std::snprintf(k, sizeof k, "model.sam_model.blocks.%lld.%s", L, ck);
                return wcopy(b, nm.f("%s", d), nm.ckpt("%s", k), sh, w, g);
            };
            if ((rc = SW("attn_norm.weight", "norm1.weight", {kSamW}, &l.w_n1)) != RAD_OK) return rc;
            if ((rc = SW("attn_norm.bias", "norm1.bias", {kSamW}, &l.b_n1)) != RAD_OK) return rc;
            if ((rc = SW("attn.qkv.weight", "attn.qkv.weight", {3 * kSamW, kSamW},
                         &l.w_qkv)) != RAD_OK) return rc;
            if ((rc = SW("attn.qkv.bias", "attn.qkv.bias", {3 * kSamW}, &l.b_qkv)) != RAD_OK)
                return rc;
            if ((rc = SW("attn.proj.weight", "attn.proj.weight", {kSamW, kSamW},
                         &l.w_proj)) != RAD_OK) return rc;
            if ((rc = SW("attn.proj.bias", "attn.proj.bias", {kSamW}, &l.b_proj)) != RAD_OK)
                return rc;
            if ((rc = SW("attn.rel_pos_h", "attn.rel_pos_h", {l.rel_rows, kSamHd},
                         &l.w_rh)) != RAD_OK) return rc;
            if ((rc = SW("attn.rel_pos_w", "attn.rel_pos_w", {l.rel_rows, kSamHd},
                         &l.w_rw)) != RAD_OK) return rc;
            if ((rc = SW("mlp_norm.weight", "norm2.weight", {kSamW}, &l.w_n2)) != RAD_OK) return rc;
            if ((rc = SW("mlp_norm.bias", "norm2.bias", {kSamW}, &l.b_n2)) != RAD_OK) return rc;
            if ((rc = SW("mlp.lin1.weight", "mlp.lin1.weight", {kSamFf, kSamW},
                         &l.w_l1)) != RAD_OK) return rc;
            if ((rc = SW("mlp.lin1.bias", "mlp.lin1.bias", {kSamFf}, &l.b_l1)) != RAD_OK) return rc;
            if ((rc = SW("mlp.lin2.weight", "mlp.lin2.weight", {kSamW, kSamFf},
                         &l.w_l2)) != RAD_OK) return rc;
            if ((rc = SW("mlp.lin2.bias", "mlp.lin2.bias", {kSamW}, &l.b_l2)) != RAD_OK) return rc;
            l40.copy_weights_from(l);
        }

        if ((rc = wcopy(b, nm.f("sam.neck.0.weight"), nm.ckpt("model.sam_model.neck.0.weight"),
                        {256, kSamW}, &w_neck0, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.neck.1.weight"), nm.ckpt("model.sam_model.neck.1.weight"),
                        {256}, &w_nk1, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.neck.1.bias"), nm.ckpt("model.sam_model.neck.1.bias"),
                        {256}, &b_nk1, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.neck.2.weight"), nm.ckpt("model.sam_model.neck.2.weight"),
                        {256, 256 * 9}, &w_nk2, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.neck.3.weight"), nm.ckpt("model.sam_model.neck.3.weight"),
                        {256}, &w_nk3, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.neck.3.bias"), nm.ckpt("model.sam_model.neck.3.bias"),
                        {256}, &b_nk3, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.net_2.weight"), nm.ckpt("model.sam_model.net_2.weight"),
                        {512, 256 * 9}, &w_net2, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.net_3.weight"), nm.ckpt("model.sam_model.net_3.weight"),
                        {1024, 512 * 9}, &w_net3, gm)) != RAD_OK) return rc;

        const char* cp = "model.vision_model.";
        if ((rc = wcopy(b, nm.f("clip.cls"), nm.ckpt("%sembeddings.class_embedding", cp),
                        {kClipW}, &w_cls, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("clip.pos_embed"),
                        nm.ckpt("%sembeddings.position_embedding.weight", cp),
                        {kClipSrcTok, kClipW}, &w_cpos, gm, RAD_ACCESS_VOCAB)) != RAD_OK)
            return rc;
        if ((rc = wcopy(b, nm.f("clip.pre_norm.weight"), nm.ckpt("%spre_layrnorm.weight", cp),
                        {kClipW}, &w_playr, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("clip.pre_norm.bias"), nm.ckpt("%spre_layrnorm.bias", cp),
                        {kClipW}, &b_playr, gm)) != RAD_OK) return rc;

        t64.clip.assign((size_t)kClipDepth, ClipLayerT<Geom64>{});
        t40.clip.assign((size_t)kClipDepth, ClipLayerT<Geom40>{});
        for (int64_t i = 0; i < kClipDepth; ++i) {
            ClipLayerT<Geom64>& l = t64.clip[(size_t)i];
            ClipLayerT<Geom40>& l40 = t40.clip[(size_t)i];
            const RadWeightGroup g = grp_layer(grp_base + 32 + (int)i);
            const long long L = (long long)i;
            auto CW = [&](const char* dn, const char* ck, std::initializer_list<int64_t> sh,
                          rad_weight* w) -> int {
                char d[96], k[192];
                std::snprintf(d, sizeof d, "clip.blk.%lld.%s", L, dn);
                std::snprintf(k, sizeof k, "model.vision_model.transformer.layers.%lld.%s", L, ck);
                return wcopy(b, nm.f("%s", d), nm.ckpt("%s", k), sh, w, g);
            };
            if ((rc = CW("attn_norm.weight", "layer_norm1.weight", {kClipW}, &l.w_n1)) != RAD_OK)
                return rc;
            if ((rc = CW("attn_norm.bias", "layer_norm1.bias", {kClipW}, &l.b_n1)) != RAD_OK)
                return rc;
            if ((rc = CW("attn.qkv.weight", "self_attn.qkv_proj.weight", {3 * kClipW, kClipW},
                         &l.w_qkv)) != RAD_OK) return rc;
            if ((rc = CW("attn.qkv.bias", "self_attn.qkv_proj.bias", {3 * kClipW},
                         &l.b_qkv)) != RAD_OK) return rc;
            if ((rc = CW("attn.out.weight", "self_attn.out_proj.weight", {kClipW, kClipW},
                         &l.w_out)) != RAD_OK) return rc;
            if ((rc = CW("attn.out.bias", "self_attn.out_proj.bias", {kClipW}, &l.b_out)) != RAD_OK)
                return rc;
            if ((rc = CW("mlp_norm.weight", "layer_norm2.weight", {kClipW}, &l.w_n2)) != RAD_OK)
                return rc;
            if ((rc = CW("mlp_norm.bias", "layer_norm2.bias", {kClipW}, &l.b_n2)) != RAD_OK)
                return rc;
            if ((rc = CW("mlp.fc1.weight", "mlp.fc1.weight", {kClipFf, kClipW}, &l.w_fc1)) != RAD_OK)
                return rc;
            if ((rc = CW("mlp.fc1.bias", "mlp.fc1.bias", {kClipFf}, &l.b_fc1)) != RAD_OK) return rc;
            if ((rc = CW("mlp.fc2.weight", "mlp.fc2.weight", {kClipW, kClipFf}, &l.w_fc2)) != RAD_OK)
                return rc;
            if ((rc = CW("mlp.fc2.bias", "mlp.fc2.bias", {kClipW}, &l.b_fc2)) != RAD_OK) return rc;
            l40.copy_weights_from(l);
        }

        if ((rc = wcopy(b, nm.f("proj.weight"), nm.ckpt("model.projector.layers.weight"),
                        {kNEmbd, kFused}, &w_proj, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("proj.bias"), nm.ckpt("model.projector.layers.bias"),
                        {kNEmbd}, &b_proj, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("layout.image_newline"), nm.ckpt("model.image_newline"),
                        {kNEmbd}, &w_newline, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("layout.view_seperator"), nm.ckpt("model.view_seperator"),
                        {kNEmbd}, &w_sep, gm)) != RAD_OK) return rc;
        return RAD_OK;
    }

    /* ------------------------------------------------------------------ op builders */
    static rad_op gemm_b(RadBuilder* b, int64_t max_m, int64_t N, int64_t K, const char* act,
                         rad_weight w, rad_weight bias) {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, max_m), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("act", act), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w, bias));
    }
    static rad_op gemm_u(RadBuilder* b, int64_t max_m, int64_t N, int64_t K, rad_weight w) {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, max_m), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    }
    static rad_op lnorm(RadBuilder* b, int64_t max_m, int64_t n, float eps, rad_weight w,
                        rad_weight bias) {
        return RAD_OP(b, "layernorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, max_m), RAD_INT("n", n), RAD_F64("eps", eps),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w, bias));
    }
    /* A bias-free 3x3 convolution as conv2d_unfold + gemm_nt: the unfolded rows are exactly what
     * a GEMM against the checkpoint's [cout, cin, 3, 3] weight (read as [cout, cin*9]) needs. */
    static rad_op unfold(RadBuilder* b, int64_t h, int64_t w, int64_t cin, int64_t stride,
                         int64_t cu_div) {
        return RAD_OP(b, "conv2d_unfold",
                      RAD_PARAMS(RAD_INT("M", h), RAD_INT("width", w), RAD_INT("cin", cin),
                                 RAD_INT("kh", 3), RAD_INT("kw", 3), RAD_INT("stride", stride),
                                 RAD_INT("pad", 1), RAD_INT("cu_div", cu_div),
                                 RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    }

    /* ------------------------------------------------------------------ declare */
    /* Maps and declares the tower's weights always; declares the ops and the encoder only when
     * SAM's attention op exists. `grp_base` is a free movement-group number. */
    int declare(RadBuilder* b, const RadBuildCtx* ctx, Names& nm, int64_t n_embd_model,
                int grp_base) {
        RAD_ARCH_TRY(declare_weights(b, nm, grp_base));
        if (n_embd_model != kNEmbd) {
            fprintf(stderr, "radiance: unlimitedocr: the decoder's hidden size %lld is not the "
                            "DeepEncoder's %lld\n", (long long)n_embd_model, (long long)kNEmbd);
            return RAD_E_UNSUPPORTED;
        }
        if (!probe(b)) {
            rad_note(b, "the DeepEncoder tower is NOT declared: SAM's attention needs the "
                        "`sam_attn` op (windowed 14x14 / global grid-wide attention with decomposed "
                        "rel_pos_h/rel_pos_w), which no loaded kernel library provides. Its "
                        "weights are mapped and declared, so a converted container carries them; "
                        "images are refused until the op lands");
            on = false;
            return RAD_OK;
        }
        max_patches = ctx->max_enc_patches;
        if (max_patches < Geom64::kSegView) {
            /* --mm-max-patches 0 (text only) or a bound below one 1024x1024 view: the tower stays
             * off and the trunk serves text, which is what the flag asks for. */
            rad_note(b, "the DeepEncoder tower is NOT declared: one 1024x1024 view is %lld staged "
                        "patches and --mm-max-patches is %lld; images are refused",
                     (long long)Geom64::kSegView, (long long)max_patches);
            on = false;
            return RAD_OK;
        }
        /* The towers size their activations for the tiles one invocation batches. A pass is
         * max_patches/(G::kSamTok) whole tiles at most; kBatchCap keeps the set modest. */
        max_views64 = max_patches / Geom64::kSamTok;
        max_views40 = max_patches / Geom40::kSamTok;
        if (max_views64 > kBatchCap) max_views64 = kBatchCap;
        if (max_views40 > kBatchCap) max_views40 = kBatchCap;
        if (max_views64 < 1) max_views64 = 1;
        if (max_views40 < 1) max_views40 = 1;
        b_out = decl_b(b, "v.out", RAD_BF16, {max_patches / kMerge, kNEmbd});

        /* The pass's shared constants, declared FIRST because they are issued first. Liveness is
         * computed over this op order, so an op declared after its readers leaves its buffer with
         * a live range that begins at the reader: the packer then aliases that buffer with an
         * activation written in between, and the value is gone by the time it is read (this is
         * exactly what corrupted the cu_seqlens tables while they were declared at the bottom). */
        b_one    = decl_b(b, "v.one",      RAD_BF16, {1, kNEmbd});
        b_one_c  = decl_b(b, "v.one.clip", RAD_BF16, {1, kClipW});
        op_fill_one = rw(b, RAD_OP(b, "fill",
                           RAD_PARAMS(RAD_INT("M", 1), RAD_INT("n", kNEmbd),
                                      RAD_STR("dtype", "bf16"), RAD_F64("value", 1.0)),
                           RAD_NOWEIGHTS),
                         {}, {b_one});
        op_fill_one_c = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_INT("M", 1), RAD_INT("n", kClipW),
                                        RAD_STR("dtype", "bf16"), RAD_F64("value", 1.0)),
                             RAD_NOWEIGHTS),
                           {}, {b_one_c});
        RAD_ARCH_TRY(t64.declare(b, nm, *this, b_out, b_one, b_one_c));
        RAD_ARCH_TRY(t40.declare(b, nm, *this, b_out, b_one, b_one_c));
        if (!ctx->shape_probe) {
            RadEncoderDecl e{};
            e.name        = "deepencoder";
            e.modalities  = RAD_MM_IMAGE;
            e.patch_dim   = kPatchDim;
            e.merge       = kMerge;
            e.max_patches = max_patches;
            e.n_embd      = kNEmbd;
            e.out         = b_out;
            /* Each pass runs one of the two towers, chosen by the segment grid. */
            e.one_grid    = 1;
            RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        }
        on = true;
        rad_note(b, "DeepEncoder: SAM ViT-B/16 on two grids (%lld blocks of %lld, %lld heads x "
                    "%lld, window %lld on a 64x64/40x40 grid, global on 2/5/8/11, rel_pos tables "
                    "27/127 rows, the 127 resizing to 79 at a 40 grid) -> map -> CLIP ViT-L/14 "
                    "(%lld layers of %lld, %lld heads x %lld, 257/101 tokens) -> %lld -> "
                    "projector -> %lld; a view is 4368 (1024) or 1776 (640) staged patches -> "
                    "273/111 rows; the 40 grid's position tables are bicubic+antialias resamples",
                 (long long)kSamDepth, (long long)kSamW, (long long)kSamHeads, (long long)kSamHd,
                 (long long)kWin, (long long)kClipDepth, (long long)kClipW, (long long)kClipHeads,
                 (long long)kClipHd, (long long)kFused, (long long)kNEmbd);
        return RAD_OK;
    }

    /* ------------------------------------------------------------------ step */
    /* A pass is one view grid plus the square one-cell separators that close its views. A view
     * segment is G x (n*G + kDown) patches: n tiles of G real patch rows side by side, with the
     * reference's single newline patch column at the end of the line (only the first G patches of
     * a row are read). Its output is kMap lines of n*kMap features plus one newline cell. The
     * separator segment (kDown x kDown patches, no pixels) is one learned view_seperator row. */
    void step(RadCtx* c, const RadBatch* batch) {
        if (!on) {
            rad_step_fail(c, "unlimitedocr: an image pass reached the DeepEncoder, which this "
                             "program did not declare");
            return;
        }
        const int64_t n_patch = batch->enc_n_patch;
        const int64_t n_seg   = batch->enc_n_seg;
        const int32_t* sh = batch->enc_seg_h;
        const int32_t* sw = batch->enc_seg_w;
        if (n_patch <= 0 || n_seg <= 0 || !sh || !sw) {
            rad_step_fail(c, "unlimitedocr: an encoder pass the DeepEncoder does not serve: it "
                             "wants its segments' grids");
            return;
        }
        /* Validate the pass: every view segment one grid, every separator exactly one merged
         * cell. A view row is a line chunk: n = w/h tiles, its last chunk carrying the line's
         * newline column (w = n*h + kDown), an earlier chunk not (w = n*h). */
        int64_t n_tiles = 0;
        int32_t grid = 0;
        for (int64_t s = 0; s < n_seg; ++s) {
            const int32_t h = sh[s], w = sw[s];
            if (h <= 0 || w <= 0) {
                rad_step_fail(c, "unlimitedocr: an encoder pass with an empty segment grid");
                return;
            }
            if (h == kDown && w == kDown) continue;     /* the one-cell view separator */
            const int32_t rem = w % h;
            if (w < h || (rem != 0 && rem != (int32_t)kDown)) {
                char msg[192];
                std::snprintf(msg, sizeof msg,
                              "unlimitedocr: a %dx%d-patch view segment is not a line of whole "
                              "%d-patch tiles (optionally plus the newline column)", (int)h,
                              (int)w, (int)h);
                rad_step_fail(c, msg);
                return;
            }
            if (grid == 0) grid = h;
            if (h != grid) {
                rad_step_fail(c, "unlimitedocr: an encoder pass mixed view grids; the pass packer "
                                 "must break a pass at a geometry change");
                return;
            }
            n_tiles += w / h;
        }
        if (n_tiles > 0 && grid != Geom64::kGrid && grid != Geom40::kGrid) {
            char msg[192];
            std::snprintf(msg, sizeof msg,
                          "unlimitedocr: an encoder pass at a %d-patch grid the DeepEncoder "
                          "does not serve (it declares 64 and 40)", (int)grid);
            rad_step_fail(c, msg);
            return;
        }
        /* Shared constants. */
        RAD_ISSUE(c, op_fill_one, RAD_B(b_one));
        RAD_ISSUE(c, op_fill_one_c, RAD_B(b_one_c));
        /* Walk the segments into a flat tile list. Every view tile is batched with the pass's
         * other tiles (chunked at the tower's Bmax); only the separator rows are issued on their
         * own. */
        std::vector<Tile> tiles;
        std::vector<int64_t> sep_rows;
        int64_t pc = 0;
        for (int64_t s = 0; s < n_seg; ++s) {
            const int64_t h = sh[s], sw_ = sw[s];
            const int64_t rowbase = pc / kMerge;
            if (h == kDown && sw_ == kDown) {
                sep_rows.push_back(rowbase);
                pc += h * sw_;
                continue;
            }
            const int64_t n = sw_ / h;
            const bool nl = (sw_ % h) != 0;     /* the line's last chunk holds the newline col */
            for (int64_t t = 0; t < n; ++t) {
                Tile tl;
                tl.patch0 = pc;
                tl.gw     = sw_;
                tl.t      = t;
                tl.n      = n;
                tl.rowbase = rowbase;
                tl.nl     = nl && t == n - 1;
                tiles.push_back(tl);
            }
            pc += h * sw_;
        }
        const bool do_tap = !tapped_;
        bool tapped_view = false;
        for (size_t i = 0; i < tiles.size(); ) {
            const int64_t bm = (grid == Geom64::kGrid) ? t64.Bmax : t40.Bmax;
            int64_t nb = (int64_t)(tiles.size() - i);
            if (nb > bm) nb = bm;
            const bool tap0 = do_tap && !tapped_view;
            if (grid == Geom64::kGrid)
                t64.batch(c, batch, &tiles[i], nb, tap0, *this);
            else
                t40.batch(c, batch, &tiles[i], nb, tap0, *this);
            tapped_view = true;
            i += (size_t)nb;
        }
        /* One learned separator row; geometry-independent, so the 64 tower carries the op. */
        for (int64_t rowbase : sep_rows)
            RAD_ISSUE(c, t64.op_sep, RAD_B(b_one), RAD_W(w_sep),
                      brow_slice(b_out, rowbase, 1, kNEmbd));
        if (tapped_view) tapped_ = true;
    }
};

/* ------------------------------------------------------------------ tower declare */
template <typename G>
int DeepEncoder::Tower<G>::declare(RadBuilder* b, Names& nm, DeepEncoder& w, rad_buf out,
                                   rad_buf one, rad_buf one_c) {
    static_assert(G::kGrid == DeepEncoder::Geom64::kGrid ||
                  G::kGrid == DeepEncoder::Geom40::kGrid, "the tower serves two grids");
    const bool is40 = G::kGrid == DeepEncoder::Geom40::kGrid;
    const char* sfx = is40 ? "40" : "64";
    p_one = one; p_one_c = one_c; p_out = out;
    char bn[64];
#define RAD_VN(fmtv) std::snprintf(bn, sizeof bn, fmtv, sfx)

    /* Tiles one invocation batches: the pass bound over this grid's tokens, capped. */
    Bmax = w.max_patches / kSamTok;
    if (Bmax < 1) Bmax = 1;
    if (Bmax > DeepEncoder::kBatchCap) Bmax = DeepEncoder::kBatchCap;
    const int64_t B = Bmax;

    /* ---- activations, B tiles' worth, tile-major. */
    RAD_VN("v%s.sam.x");     b_x   = decl_b(b, bn, RAD_BF16, {B * kSamTok, DeepEncoder::kSamW});
    RAD_VN("v%s.sam.h");     b_h   = decl_b(b, bn, RAD_BF16, {B * kSamTok, DeepEncoder::kSamW});
    RAD_VN("v%s.sam.qkv");   b_qkv = decl_b(b, bn, RAD_BF16, {B * kSamTok, 3 * DeepEncoder::kSamW});
    RAD_VN("v%s.sam.att");   b_att = decl_b(b, bn, RAD_BF16, {B * kSamTok, DeepEncoder::kSamW});
    RAD_VN("v%s.sam.ff");    b_ff  = decl_b(b, bn, RAD_BF16, {B * kSamTok, DeepEncoder::kSamFf});
    RAD_VN("v%s.sam.pos");   b_sampos  = decl_b(b, bn, RAD_BF16, {B * kSamTok, DeepEncoder::kSamW});
    RAD_VN("v%s.sam.ids");   b_ids_sam = decl_b(b, bn, RAD_I32,  {B * kSamTok}, RAD_BUF_PERSIST);
    RAD_VN("v%s.sam.hw");    b_hw    = decl_b(b, bn, RAD_BF16, {B * kSamTokP, DeepEncoder::kSamW});
    RAD_VN("v%s.sam.qkvw");  b_qkvw  = decl_b(b, bn, RAD_BF16, {B * kSamTokP, 3 * DeepEncoder::kSamW});
    RAD_VN("v%s.sam.attw");  b_attw  = decl_b(b, bn, RAD_BF16, {B * kSamTokP, DeepEncoder::kSamW});
    RAD_VN("v%s.sam.idsp");  b_ids_pad  = decl_b(b, bn, RAD_I32, {B * kSamTokP}, RAD_BUF_PERSIST);
    RAD_VN("v%s.sam.idsr");  b_ids_real = decl_b(b, bn, RAD_I32, {B * kSamTok}, RAD_BUF_PERSIST);

    RAD_VN("v%s.n0");        b_n0   = decl_b(b, bn, RAD_BF16, {B * kSamTok, 256});
    RAD_VN("v%s.n1");        b_n1   = decl_b(b, bn, RAD_BF16, {B * kSamTok, 256});
    RAD_VN("v%s.nfu");       b_unf3 = decl_b(b, bn, RAD_BF16, {B * kSamTok, 256 * 9});
    RAD_VN("v%s.n2");        b_n2   = decl_b(b, bn, RAD_BF16, {B * kSamTok, 256});
    RAD_VN("v%s.n3");        b_n3   = decl_b(b, bn, RAD_BF16, {B * kSamTok, 256});
    RAD_VN("v%s.n2u");       b_unf_2 = decl_b(b, bn, RAD_BF16, {B * kSamTok / 4, 256 * 9});
    RAD_VN("v%s.m2");        b_map2  = decl_b(b, bn, RAD_BF16, {B * kSamTok / 4, 512});
    RAD_VN("v%s.n3u");       b_unf_3 = decl_b(b, bn, RAD_BF16, {B * kMap * kMap, 512 * 9});
    RAD_VN("v%s.map");       b_map   = decl_b(b, bn, RAD_BF16, {B * kMap * kMap, DeepEncoder::kClipW});

    RAD_VN("v%s.cx");        b_c   = decl_b(b, bn, RAD_BF16, {B * kClipTok, DeepEncoder::kClipW});
    RAD_VN("v%s.ch");        b_ch  = decl_b(b, bn, RAD_BF16, {B * kClipTok, DeepEncoder::kClipW});
    RAD_VN("v%s.cq");        b_cq  = decl_b(b, bn, RAD_BF16, {B * kClipTok, 3 * DeepEncoder::kClipW});
    RAD_VN("v%s.ca");        b_ca  = decl_b(b, bn, RAD_BF16, {B * kClipTok, DeepEncoder::kClipW});
    RAD_VN("v%s.cf");        b_cf  = decl_b(b, bn, RAD_BF16, {B * kClipTok, DeepEncoder::kClipFf});
    RAD_VN("v%s.cg2");       b_cg2 = decl_b(b, bn, RAD_BF16, {B * kClipTok, DeepEncoder::kClipFf});
    RAD_VN("v%s.ct");        b_ct  = decl_b(b, bn, RAD_BF16, {B * kClipTok, DeepEncoder::kClipW});
    RAD_VN("v%s.cp");        b_cp  = decl_b(b, bn, RAD_BF16, {B * kClipTok, DeepEncoder::kClipW});
    RAD_VN("v%s.cids");      b_ids_clip = decl_b(b, bn, RAD_I32, {B * kClipTok}, RAD_BUF_PERSIST);
    RAD_VN("v%s.cpos");      b_cpos = decl_b(b, bn, RAD_BF16, {B * kClipTok, DeepEncoder::kClipW});

    RAD_VN("v%s.fuse");      b_fuse    = decl_b(b, bn, RAD_BF16, {B * kMap * kMap, DeepEncoder::kFused});
    RAD_VN("v%s.proj");      b_projbuf = decl_b(b, bn, RAD_BF16, {B * kMap * kMap, DeepEncoder::kNEmbd});
    RAD_VN("v%s.nlrow");     b_nlrow   = decl_b(b, bn, RAD_BF16, {1, DeepEncoder::kNEmbd});
    RAD_VN("v%s.pixstg");    b_pixstg  = decl_b(b, bn, RAD_BF16, {B * kSamTok, DeepEncoder::kPatchDim});

    /* One cu_seqlens table each, a bound a tile: the batched SAM/CLIP attention reads B segments
     * and the three unfolds divide by kGrid (net_3 by 2*kGrid). */
    RAD_VN("v%s.cus");
    b_cu_sam = decl_b(b, bn, RAD_I32, {B + 1});
    op_cu_sam = rw(b, RAD_OP(b, "iota",
                     RAD_PARAMS(RAD_RANGE("M", 2, B + 1), RAD_INT("start", 0),
                                RAD_INT("step", kSamTok)),
                     RAD_NOWEIGHTS),
                   {}, {b_cu_sam});
    RAD_VN("v%s.cuc");
    b_cu_clip = decl_b(b, bn, RAD_I32, {B + 1});
    op_cu_clip = rw(b, RAD_OP(b, "iota",
                      RAD_PARAMS(RAD_RANGE("M", 2, B + 1), RAD_INT("start", 0),
                                 RAD_INT("step", kClipTok)),
                      RAD_NOWEIGHTS),
                    {}, {b_cu_clip});
    RAD_VN("v%s.cup");
    b_cu_pad = decl_b(b, bn, RAD_I32, {B + 1});
    op_cu_pad = rw(b, RAD_OP(b, "iota",
                     RAD_PARAMS(RAD_RANGE("M", 2, B + 1), RAD_INT("start", 0),
                                RAD_INT("step", kSamTokP)),
                     RAD_NOWEIGHTS),
                   {}, {b_cu_pad});

    /* The 40 tower stages its resampled absolute tables into PERSIST planes (never packed,
     * never aliased); fill_tables() writes them once on the rank's first tile pass and the
     * gathers below read them as plain operands. The 64 tower gathers straight from the
     * checkpoint's weight planes. */
    if (is40) {
        RAD_VN("v%s.pos40");
        b_pos_tab  = decl_b(b, bn, RAD_BF16, {kSamTok, DeepEncoder::kSamW}, RAD_BUF_PERSIST);
        RAD_VN("v%s.cpos40");
        b_cpos_tab = decl_b(b, bn, RAD_BF16, {kClipTok, DeepEncoder::kClipW}, RAD_BUF_PERSIST);
    }

    /* ---- SAM stem. The patch embed is one GEMM over the whole staged tile (the processor
     * stages kDown zero patch columns at the end of each row, skipped by the pitched pack into
     * `pixstg`), then the position table added in: a row gather, from the 64x64 plane at the 64
     * grid (exact, unchanged) and the staged 40x40 resample at 40. */
    op_patch = rw(b, gemm_b(b, B * kSamTok, DeepEncoder::kSamW, DeepEncoder::kPatchDim, "none",
                            w.w_patch, w.b_patch), {b_pixstg}, {b_x});
    if (!is40)
        op_sampos = rw(b, RAD_OP(b, "embed_lookup",
                         RAD_PARAMS(RAD_RANGE("M", 1, B * kSamTok),
                                    RAD_INT("n_embd", DeepEncoder::kSamW),
                                    RAD_INT("n_vocab", kSamTok), RAD_STR("dtype", "bf16")),
                         RAD_WEIGHTS(w.w_pos)),
                     {b_ids_sam}, {b_sampos});
    else
        op_sampos = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, B * kSamTok),
                                    RAD_INT("n", DeepEncoder::kSamW),
                                    RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                     {b_pos_tab, b_ids_sam}, {b_sampos});
    op_sampadd = rw(b, RAD_OP(b, "add",
                              RAD_PARAMS(RAD_RANGE("M", 1, B * kSamTok),
                                         RAD_INT("n", DeepEncoder::kSamW),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                   {b_x, b_sampos}, {b_x});

    for (size_t i = 0; i < sam.size(); ++i) {
        SamLayerT<G>& l = sam[i];
        l.op_n1 = rw(b, lnorm(b, B * kSamTok, DeepEncoder::kSamW, 1e-6f, l.w_n1, l.b_n1),
                     {b_x}, {b_h});
        /* The windowed path: `h` into window_partition's padded order with zero pad rows
         * (a negative gather index is a zero row), so the qkv linear gives the pad rows the
         * reference's bias -- deepencoder pads AFTER norm1 and runs the whole attention on the
         * padded window, so a pad key's k and v are the qkv bias, not zero. */
        l.op_gpad = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, B * kSamTokP), RAD_INT("n", DeepEncoder::kSamW),
                                    RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                       {b_h, b_ids_pad}, {b_hw});
        l.op_qkv = rw(b, gemm_b(b, B * kSamTok, 3 * DeepEncoder::kSamW, DeepEncoder::kSamW, "none",
                                l.w_qkv, l.b_qkv), {b_h}, {b_qkv});
        l.op_qkvw = rw(b, gemm_b(b, B * kSamTokP, 3 * DeepEncoder::kSamW, DeepEncoder::kSamW, "none",
                                 l.w_qkv, l.b_qkv), {b_hw}, {b_qkvw});
        l.op_attn = rw(b, RAD_OP(b, "sam_attn",
                         RAD_PARAMS(RAD_RANGE("M", 1, B * kSamTok),
                                    RAD_INT("head_dim", DeepEncoder::kSamHd),
                                    RAD_INT("n_head", DeepEncoder::kSamHeads),
                                    RAD_INT("grid_h", kGrid), RAD_INT("grid_w", kGrid),
                                    RAD_INT("window", l.window),
                                    RAD_F64("scale", 1.0 / 8.0), RAD_STR("dtype", "bf16")),
                         RAD_WEIGHTS(l.w_rh, l.w_rw)),
                     {b_qkv, b_cu_sam}, {b_att});
        l.op_attnw = rw(b, RAD_OP(b, "sam_attn",
                          RAD_PARAMS(RAD_RANGE("M", 1, B * kSamTokP),
                                     RAD_INT("head_dim", DeepEncoder::kSamHd),
                                     RAD_INT("n_head", DeepEncoder::kSamHeads),
                                     RAD_INT("grid_h", kGridP), RAD_INT("grid_w", kGridP),
                                     RAD_INT("window", DeepEncoder::kWin),
                                     RAD_F64("scale", 1.0 / 8.0), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(l.w_rh, l.w_rw)),
                      {b_qkvw, b_cu_pad}, {b_attw});
        l.op_greal = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, B * kSamTok), RAD_INT("n", DeepEncoder::kSamW),
                                    RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                       {b_attw, b_ids_real}, {b_att});
        l.op_proj = rw(b, gemm_b(b, B * kSamTok, DeepEncoder::kSamW, DeepEncoder::kSamW, "none",
                                 l.w_proj, l.b_proj), {b_att}, {b_h});
        /* deepencoder rounds the projection before adding the shortcut (`x + attn(norm(x))`
         * is two bf16 operations), so the residual add is its own op: fusing it into the
         * GEMM's f32 accumulator skips a rounding and drifts from the reference. */
        l.op_padd = rw(b, RAD_OP(b, "add",
                         RAD_PARAMS(RAD_RANGE("M", 1, B * kSamTok), RAD_INT("n", DeepEncoder::kSamW),
                                    RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                       {b_h, b_x}, {b_x});
        l.op_n2 = rw(b, lnorm(b, B * kSamTok, DeepEncoder::kSamW, 1e-6f, l.w_n2, l.b_n2),
                     {b_x}, {b_h});
        l.op_l1 = rw(b, gemm_b(b, B * kSamTok, DeepEncoder::kSamFf, DeepEncoder::kSamW, "gelu",
                               l.w_l1, l.b_l1), {b_h}, {b_ff});
        l.op_l2 = rw(b, gemm_b(b, B * kSamTok, DeepEncoder::kSamW, DeepEncoder::kSamFf, "none",
                               l.w_l2, l.b_l2), {b_ff}, {b_h});
        l.op_ladd = rw(b, RAD_OP(b, "add",
                         RAD_PARAMS(RAD_RANGE("M", 1, B * kSamTok), RAD_INT("n", DeepEncoder::kSamW),
                                    RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                       {b_h, b_x}, {b_x});
    }

    /* ---- neck. The 1x1 is a GEMM; the 3x3s are unfold + gemm because they carry no bias and
     * conv2d_nhwc's `b` is required. LayerNorm2d is a layernorm across a pixel's channels,
     * which are innermost. */
    op_neck0 = rw(b, gemm_u(b, B * kSamTok, 256, DeepEncoder::kSamW, w.w_neck0), {b_x}, {b_n0});
    op_nk1 = rw(b, lnorm(b, B * kSamTok, 256, 1e-6f, w.w_nk1, w.b_nk1), {b_n0}, {b_n1});
    op_unf3 = rw(b, unfold(b, kGrid, kGrid, 256, 1, kGrid), {b_n1, b_cu_sam}, {b_unf3});
    op_nk2 = rw(b, gemm_u(b, B * kSamTok, 256, 256 * 9, w.w_nk2), {b_unf3}, {b_n2});
    op_nk3 = rw(b, lnorm(b, B * kSamTok, 256, 1e-6f, w.w_nk3, w.b_nk3), {b_n2}, {b_n3});
    op_unf_net2 = rw(b, unfold(b, kGrid, kGrid, 256, 2, kGrid), {b_n3, b_cu_sam}, {b_unf_2});
    op_net2 = rw(b, gemm_u(b, B * kSamTok / 4, 512, 256 * 9, w.w_net2), {b_unf_2}, {b_map2});
    op_unf_net3 = rw(b, unfold(b, kGrid / 2, kGrid / 2, 512, 2, kGrid * 2),
                     {b_map2, b_cu_sam}, {b_unf_3});
    op_net3 = rw(b, gemm_u(b, B * kMap * kMap, DeepEncoder::kClipW, 512 * 9, w.w_net3),
                 {b_unf_3}, {b_map});

    /* ---- CLIP. The class row goes first and the map's rows below it: absolute position
     * pairs row for row, and attention is permutation-equivariant, so the class row's place
     * is free. (The reference cats class first too.) The 40 tower gathers its staged 101-row
     * table; the 64 tower the checkpoint's 257 rows. */
    op_cls = rw(b, RAD_OP(b, "mul_weight",
                          RAD_PARAMS(RAD_INT("M", 1), RAD_INT("n", DeepEncoder::kClipW),
                                     RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w.w_cls)),
                  {p_one_c}, {b_c});
    op_cmap = rw(b, RAD_OP(b, "cast",
                           RAD_PARAMS(RAD_INT("M", kMap * kMap), RAD_INT("n", DeepEncoder::kClipW),
                                      RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                           RAD_NOWEIGHTS),
                    {b_map}, {b_c});
    if (!is40)
        op_cpos = rw(b, RAD_OP(b, "embed_lookup",
                               RAD_PARAMS(RAD_RANGE("M", 1, B * kClipTok),
                                          RAD_INT("n_embd", DeepEncoder::kClipW),
                                          RAD_INT("n_vocab", kClipTok), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(w.w_cpos)),
                     {b_ids_clip}, {b_cpos});
    else
        op_cpos = rw(b, RAD_OP(b, "gather_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, B * kClipTok),
                                          RAD_INT("n", DeepEncoder::kClipW),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     {b_cpos_tab, b_ids_clip}, {b_cpos});
    op_cadd = rw(b, RAD_OP(b, "add",
                           RAD_PARAMS(RAD_RANGE("M", 1, B * kClipTok),
                                      RAD_INT("n", DeepEncoder::kClipW),
                                      RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                 {b_c, b_cpos}, {b_c});
    op_playr = rw(b, lnorm(b, B * kClipTok, DeepEncoder::kClipW, 1e-5f, w.w_playr, w.b_playr),
                  {b_c}, {b_cp});
    /* deepencoder normalises the embedding once (pre_layrnorm) and the transformer's residual
     * stream starts there, not at the embedding: layer 0's norm1 and its residual add both
     * read the pre_layrnorm output. */
    op_ccopy = rw(b, RAD_OP(b, "cast",
                            RAD_PARAMS(RAD_INT("M", kClipTok), RAD_INT("n", DeepEncoder::kClipW),
                                       RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                            RAD_NOWEIGHTS),
                  {b_cp}, {b_c});

    for (ClipLayerT<G>& l : clip) {
        l.op_n1 = rw(b, lnorm(b, B * kClipTok, DeepEncoder::kClipW, 1e-5f, l.w_n1, l.b_n1),
                     {b_c}, {b_ch});
        l.op_qkv = rw(b, gemm_b(b, B * kClipTok, 3 * DeepEncoder::kClipW, DeepEncoder::kClipW,
                                "none", l.w_qkv, l.b_qkv), {b_ch}, {b_cq});
        l.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, B * kClipTok),
                                            RAD_INT("head_dim", DeepEncoder::kClipHd),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                            RAD_F64("scale", 1.0 / 8.0), RAD_INT("cu_div", 1),
                                            RAD_INT("max_seqlen", kClipTok),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                      {b_cq, b_cu_clip}, {b_ca});
        l.op_out = rw(b, gemm_b(b, B * kClipTok, DeepEncoder::kClipW, DeepEncoder::kClipW, "none",
                                l.w_out, l.b_out), {b_ca}, {b_ct});
        l.op_cadd = rw(b, RAD_OP(b, "add",
                                 RAD_PARAMS(RAD_RANGE("M", 1, B * kClipTok),
                                            RAD_INT("n", DeepEncoder::kClipW),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       {b_ct, b_c}, {b_c});
        l.op_n2 = rw(b, lnorm(b, B * kClipTok, DeepEncoder::kClipW, 1e-5f, l.w_n2, l.b_n2),
                     {b_c}, {b_ch});
        l.op_fc1 = rw(b, gemm_b(b, B * kClipTok, DeepEncoder::kClipFf, DeepEncoder::kClipW, "none",
                                l.w_fc1, l.b_fc1), {b_ch}, {b_cf});
        /* quick_gelu: fc1's output times sigmoid(1.702 * itself), fused into one act. The
         * kernel rounds the scaled value to bf16 before the sigmoid, which is exactly the
         * store the unfused mul did, so the fusion is bit-identical to the pair it replaces
         * (and saves a 257x4096 bf16 round-trip per layer x 24 layers x tile). */
        l.op_qg = rw(b, RAD_OP(b, "act_mul",
                               RAD_PARAMS(RAD_RANGE("M", 1, B * kClipTok),
                                          RAD_INT("n", DeepEncoder::kClipFf),
                                          RAD_STR("act", "quick_gelu"), RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     {b_cf, b_cf}, {b_cg2});
        l.op_fc2 = rw(b, gemm_b(b, B * kClipTok, DeepEncoder::kClipW, DeepEncoder::kClipFf,
                                "none", l.w_fc2, l.b_fc2), {b_cg2}, {b_ct});
        l.op_madd = rw(b, RAD_OP(b, "add",
                                 RAD_PARAMS(RAD_RANGE("M", 1, B * kClipTok),
                                            RAD_INT("n", DeepEncoder::kClipW),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       {b_ct, b_c}, {b_c});
    }

    /* ---- fuse, project. */
    op_fuse_clip = rw(b, RAD_OP(b, "cast",
                                RAD_PARAMS(RAD_INT("M", kMap * kMap),
                                           RAD_INT("n", DeepEncoder::kClipW),
                                           RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                                RAD_NOWEIGHTS),
                      {b_c}, {b_fuse});
    op_fuse_sam = rw(b, RAD_OP(b, "cast",
                               RAD_PARAMS(RAD_INT("M", kMap * kMap),
                                          RAD_INT("n", DeepEncoder::kClipW),
                                          RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                               RAD_NOWEIGHTS),
                     {b_map}, {b_fuse});
    op_proj = rw(b, gemm_b(b, B * kMap * kMap, DeepEncoder::kNEmbd, DeepEncoder::kFused, "none",
                           w.w_proj, w.b_proj), {b_fuse}, {b_projbuf});
    /* Declared and never issued: they name the stage buffers as read and the out plane as
     * written, and removing them changes the buffer plan, while the layout below fills the out
     * plane with copies outside the op stream. */
    op_dbg1024 = rw(b, RAD_OP(b, "cast",
                              RAD_PARAMS(RAD_INT("M", 256), RAD_INT("n", DeepEncoder::kClipW),
                                         RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                              RAD_NOWEIGHTS),
                   {b_map, b_c, b_unf_2}, {p_out});
    op_dbg768 = rw(b, RAD_OP(b, "cast",
                             RAD_PARAMS(RAD_INT("M", 256), RAD_INT("n", DeepEncoder::kSamW),
                                        RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                             RAD_NOWEIGHTS),
                  {b_x, b_att}, {p_out});
    op_dbg256 = rw(b, RAD_OP(b, "cast",
                             RAD_PARAMS(RAD_INT("M", 256), RAD_INT("n", 256),
                                        RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                             RAD_NOWEIGHTS),
                  {b_n0, b_n1, b_n2, b_n3, b_map2, b_unf_2}, {p_out});
    op_dbg1280 = rw(b, RAD_OP(b, "cast",
                              RAD_PARAMS(RAD_INT("M", 256), RAD_INT("n", DeepEncoder::kNEmbd),
                                         RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                              RAD_NOWEIGHTS),
                   {b_projbuf}, {p_out});

    /* ---- the two layout rows. A projected map line reaches its layout rows as a pitched
     * copy in view(); the newline row is computed once here and pitched out by segment(). */
    op_newline = rw(b, RAD_OP(b, "mul_weight",
                              RAD_PARAMS(RAD_INT("M", 1), RAD_INT("n", DeepEncoder::kNEmbd),
                                         RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(w.w_newline)),
                    {p_one}, {b_nlrow});
    op_sep = rw(b, RAD_OP(b, "mul_weight",
                          RAD_PARAMS(RAD_INT("M", 1), RAD_INT("n", DeepEncoder::kNEmbd),
                                     RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w.w_sep)),
                  {p_one}, {p_out});
#undef RAD_VN
    return RAD_OK;
}

/* ------------------------------------------------------------------ tile position tables
 * The 40 tower's absolute position planes, computed once a rank on its first tile pass: the
 * checkpoint's bf16 tables read back, resampled fp32 bicubic+antialias exactly as the
 * reference's get_abs_pos_sam / get_abs_pos do (torch interpolates in fp32 and casts back),
 * staged into the PERSIST planes the gathers read. The CLIP table's class row passes through
 * (the reference cats it first). */
template <typename G>
void DeepEncoder::Tower<G>::fill_tables(RadCtx* c, const DeepEncoder& w) const {
    const int rk = rad_rank(c);
    if (rk < 0 || rk >= 32) return;
    if (tabs_done & (1u << rk)) return;
    void* pw = rad_weight_ptr(c, w.w_pos);
    void* cw = rad_weight_ptr(c, w.w_cpos);
    void* pt = rad_buf_ptr(c, b_pos_tab);
    void* ct = rad_buf_ptr(c, b_cpos_tab);
    if (!pw || !cw || !pt || !ct) {
        rad_step_fail(c, "unlimitedocr: a position table of the DeepEncoder did not bind");
        return;
    }
    const int64_t src_grid = DeepEncoder::Geom64::kGrid;     /* SAM's 64x64 table */
    const int64_t csrc = DeepEncoder::kClipSrcMap;           /* CLIP's 16x16 grid */
    std::vector<uint16_t> hpos((size_t)(src_grid * src_grid * DeepEncoder::kSamW));
    std::vector<uint16_t> hclip((size_t)(csrc * csrc + 1) * DeepEncoder::kClipW);
    if (rad_memcpy_async(hpos.data(), pw, hpos.size() * sizeof(uint16_t), rad_stream(c)) < 0 ||
        rad_memcpy_async(hclip.data(), cw, hclip.size() * sizeof(uint16_t), rad_stream(c)) < 0 ||
        rad_stream_sync(rad_stream(c)) < 0) {
        rad_step_fail(c, "unlimitedocr: the DeepEncoder's position tables could not be read back");
        return;
    }
    /* SAM: [64,64,768] -> [40,40,768], fp32, cast back to bf16. */
    std::vector<float> f((size_t)(src_grid * src_grid * DeepEncoder::kSamW)),
                       g((size_t)(kGrid * kGrid * DeepEncoder::kSamW));
    for (size_t i = 0; i < f.size(); ++i) f[i] = rad_bf16_to_f32(hpos[i]);
    resample_bicubic_aa(f.data(), (int)src_grid, (int)src_grid, (int)kGrid, (int)kGrid,
                        (int)DeepEncoder::kSamW, g.data());
    std::vector<uint16_t> out(g.size());
    for (size_t i = 0; i < g.size(); ++i) out[i] = rad_f32_to_bf16(g[i]);

    if (rad_memcpy_async(pt, out.data(), out.size() * sizeof(uint16_t), rad_stream(c)) < 0) {
        rad_step_fail(c, "unlimitedocr: the SAM tile position table could not be staged");
        return;
    }

    /* CLIP: row 0 is the class row, rows 1: are the 16x16 grid -> cls + a 10x10 resample.
     * f/g are the SAM resample's buffers, reused here as the CLIP resample's scratch; that is
     * safe only while SAM's f covers CLIP's source grid and SAM's g covers CLIP's map. Pinned
     * so retuning a geometry fails at compile time, not as a heap overrun. */
    static_assert((size_t)DeepEncoder::Geom64::kGrid * DeepEncoder::Geom64::kGrid *
                          DeepEncoder::kSamW >=
                      (size_t)DeepEncoder::kClipSrcMap * DeepEncoder::kClipSrcMap *
                          DeepEncoder::kClipW,
                  "fill_tables reuses SAM's f scratch for the CLIP source grid");
    static_assert((size_t)kGrid * kGrid * DeepEncoder::kSamW >=
                      (size_t)kMap * kMap * DeepEncoder::kClipW,
                  "fill_tables reuses SAM's g scratch for the CLIP map grid");
    std::vector<uint16_t> cout((size_t)kClipTok * DeepEncoder::kClipW);
    std::memcpy(cout.data(), hclip.data(), (size_t)DeepEncoder::kClipW * sizeof(uint16_t));
    const size_t ngrid = (size_t)(csrc * csrc * DeepEncoder::kClipW);
    for (size_t i = 0; i < ngrid; ++i) f[i] = rad_bf16_to_f32(hclip[DeepEncoder::kClipW + i]);
    resample_bicubic_aa(f.data(), (int)csrc, (int)csrc, (int)kMap, (int)kMap,
                        (int)DeepEncoder::kClipW, g.data());
    for (size_t i = 0; i < (size_t)kMap * kMap * DeepEncoder::kClipW; ++i)
        cout[(size_t)DeepEncoder::kClipW + i] = rad_f32_to_bf16(g[i]);
    if (rad_memcpy_async(ct, cout.data(), cout.size() * sizeof(uint16_t), rad_stream(c)) < 0 ||
        rad_stream_sync(rad_stream(c)) < 0) {
        rad_step_fail(c, "unlimitedocr: the CLIP tile position table could not be staged");
        return;
    }
    tabs_done |= (1u << rk);
}

/* ------------------------------------------------------------------ tower batch step */
/* The position and window tables do not depend on the pass: tile t reads its own copy of the
 * same kSamTok (kClipTok) rows, so Bmax copies are staged once a rank and every batch run
 * gathers the first B copies. The window tables carry the tile's own row offset. */
template <typename G>
void DeepEncoder::Tower<G>::stage_ids(RadCtx* c) const {
    const int rk = rad_rank(c);
    if (rk < 0 || rk >= 32 || (ids_done & (1u << rk))) return;
    std::vector<int32_t> ids_s((size_t)(Bmax * kSamTok));
    std::vector<int32_t> ids_c((size_t)(Bmax * kClipTok));
    std::vector<int32_t> pad((size_t)(Bmax * kSamTokP), -1);
    std::vector<int32_t> real((size_t)(Bmax * kSamTok));
    for (int64_t t = 0; t < Bmax; ++t) {
        for (int64_t r = 0; r < kSamTok; ++r)
            ids_s[(size_t)(t * kSamTok + r)] = (int32_t)r;
        for (int64_t r = 0; r < kClipTok; ++r)
            ids_c[(size_t)(t * kClipTok + r)] = (int32_t)r;
        for (int64_t r = 0; r < kGrid; ++r)
            for (int64_t cc = 0; cc < kGrid; ++cc) {
                pad[(size_t)(t * kSamTokP + r * kGridP + cc)] =
                    (int32_t)(t * kSamTok + r * kGrid + cc);
                real[(size_t)(t * kSamTok + r * kGrid + cc)] =
                    (int32_t)(t * kSamTokP + r * kGridP + cc);
            }
    }
    void* p_ids_s = rad_buf_ptr(c, b_ids_sam);
    void* p_ids_c = rad_buf_ptr(c, b_ids_clip);
    void* p_pad   = rad_buf_ptr(c, b_ids_pad);
    void* p_real  = rad_buf_ptr(c, b_ids_real);
    if (!p_ids_s || !p_ids_c || !p_pad || !p_real) {
        rad_step_fail(c, "unlimitedocr: a staged index table of the DeepEncoder did not bind");
        return;
    }
    if (rad_memcpy_async(p_ids_s, ids_s.data(), (int64_t)(ids_s.size() * sizeof(int32_t)),
                         rad_stream(c)) < 0 ||
        rad_memcpy_async(p_ids_c, ids_c.data(), (int64_t)(ids_c.size() * sizeof(int32_t)),
                         rad_stream(c)) < 0 ||
        rad_memcpy_async(p_pad, pad.data(), (int64_t)(pad.size() * sizeof(int32_t)),
                         rad_stream(c)) < 0 ||
        rad_memcpy_async(p_real, real.data(), (int64_t)(real.size() * sizeof(int32_t)),
                         rad_stream(c)) < 0 ||
        rad_stream_sync(rad_stream(c)) < 0) {
        rad_step_fail(c, "unlimitedocr: the staged index tables of the DeepEncoder could not be "
                         "filled");
        return;
    }
    ids_done |= (1u << rk);
}

/* One batched tower run over `nt` staged tiles. Tile t's rows are [t*kSamTok, (t+1)*kSamTok) of
 * every SAM plane (and its CLIP block at [t*kClipTok, (t+1)*kClipTok)); the segment-bounded ops
 * read a B+1-entry cu table, so attention and the unfolds never see across a tile. */
template <typename G>
void DeepEncoder::Tower<G>::batch(RadCtx* c, const RadBatch* batch, const Tile* tiles,
                                  int64_t nt, bool tap, const DeepEncoder& w) {
    if (nt < 1 || nt > Bmax) {
        rad_step_fail(c, "unlimitedocr: a DeepEncoder batch exceeded the declared tile bound");
        return;
    }
    /* The 40 tower stages its resampled position tables once on the rank's first tile pass. */
    if (b_pos_tab) fill_tables(c, w);
    stage_ids(c);

    const int64_t BT  = nt * kSamTok;
    const int64_t BTP = nt * kSamTokP;
    const int64_t BCT = nt * kClipTok;
    const int64_t BM  = nt * kMap * kMap;

    /* The patch embed: one pitched copy a tile packs its grid rows out of the line's rows (the
     * kDown zero patch columns at each row's end are the copy's dead space), then ONE GEMM over
     * all B tiles. The copies ride the pass's stream, ordered with the ops. */
    char* stg = (char*)rad_buf_ptr(c, b_pixstg);
    const int64_t rowb = kGrid * DeepEncoder::kPatchDim * 2;
    if (!stg) {
        rad_step_fail(c, "unlimitedocr: the patch rows of the DeepEncoder could not be packed");
        return;
    }
    for (int64_t t = 0; t < nt; ++t) {
        const char* pix = (const char*)batch->enc_pixels +
                          (tiles[t].patch0 + tiles[t].t * kGrid) * DeepEncoder::kPatchDim * 2;
        char* dst = stg + t * kSamTok * DeepEncoder::kPatchDim * 2;
        if (rad_memcpy_2d_async(dst, rowb, pix, tiles[t].gw * DeepEncoder::kPatchDim * 2,
                                rowb, kGrid, rad_stream(c)) < 0) {
            rad_step_fail(c, "unlimitedocr: the patch rows of the DeepEncoder could not be packed");
            return;
        }
    }
    RAD_ISSUE_N(c, op_patch, BT,
                RAD_P_T2(stg, RAD_BF16, BT, DeepEncoder::kPatchDim),
                RAD_W(w.w_patch), RAD_W(w.b_patch), RAD_NONE,
                brow_slice(b_x, 0, BT, DeepEncoder::kSamW));

    /* The pass's segment tables: B tiles, a bound each. */
    RAD_ISSUE_N(c, op_cu_sam,  nt + 1, brows(b_cu_sam, nt + 1));
    RAD_ISSUE_N(c, op_cu_clip, nt + 1, brows(b_cu_clip, nt + 1));
    RAD_ISSUE_N(c, op_cu_pad,  nt + 1, brows(b_cu_pad, nt + 1));

    /* The position table, a row for a token (a gather, exact at 64 and from the staged
     * resample at 40), added in. The ids repeat per tile, so every tile sees the same table. */
    if (tap) w.tap(c, "st_emb", b_x, 256, DeepEncoder::kSamW);
    if (b_pos_tab)
        RAD_ISSUE_N(c, op_sampos, BT, RAD_B(b_pos_tab), brows(b_ids_sam, BT),
                    brows(b_sampos, BT));
    else
        RAD_ISSUE_N(c, op_sampos, BT, brows(b_ids_sam, BT), RAD_W(w.w_pos),
                    brows(b_sampos, BT));
    if (tap) w.tap(c, "st_sampos", b_sampos, 256, DeepEncoder::kSamW);
    RAD_ISSUE_N(c, op_sampadd, BT, brows(b_x, BT), brows(b_sampos, BT), brows(b_x, BT));
    if (tap) w.tap(c, "st_x0", b_x, 256, DeepEncoder::kSamW);


    for (size_t li = 0; li < sam.size(); ++li) {
        const SamLayerT<G>& l = sam[li];
        RAD_ISSUE_N(c, l.op_n1, BT, brows(b_x, BT), RAD_W(l.w_n1), RAD_W(l.b_n1),
                    brows(b_h, BT));
        if (l.window > 0) {
            /* The reference's window path: pad the normed grid to a multiple of 14, run qkv
             * over the pad rows too (they carry the qkv bias), attend inside each 14x14 window,
             * and take the real rows back out. The pad rows take softmax mass with the bias as v. */
            RAD_ISSUE_N(c, l.op_gpad, BTP, brows(b_h, BT), brows(b_ids_pad, BTP),
                        brows(b_hw, BTP));
            RAD_ISSUE_N(c, l.op_qkvw, BTP, brows(b_hw, BTP), RAD_W(l.w_qkv),
                        RAD_W(l.b_qkv), RAD_NONE, brows(b_qkvw, BTP));
            RAD_ISSUE_N(c, l.op_attnw, BTP,
                        bcol(b_qkvw, 0, DeepEncoder::kSamW, BTP),
                        bcol(b_qkvw, DeepEncoder::kSamW, DeepEncoder::kSamW, BTP),
                        bcol(b_qkvw, 2 * DeepEncoder::kSamW, DeepEncoder::kSamW, BTP),
                        brows(b_cu_pad, nt + 1), brows(b_attw, BTP),
                        RAD_W(l.w_rh), RAD_W(l.w_rw));
            RAD_ISSUE_N(c, l.op_greal, BT, brows(b_attw, BTP), brows(b_ids_real, BT),
                        brows(b_att, BT));
        } else {
            RAD_ISSUE_N(c, l.op_qkv, BT, brows(b_h, BT), RAD_W(l.w_qkv),
                        RAD_W(l.b_qkv), RAD_NONE, brows(b_qkv, BT));
            RAD_ISSUE_N(c, l.op_attn, BT,
                        bcol(b_qkv, 0, DeepEncoder::kSamW, BT),
                        bcol(b_qkv, DeepEncoder::kSamW, DeepEncoder::kSamW, BT),
                        bcol(b_qkv, 2 * DeepEncoder::kSamW, DeepEncoder::kSamW, BT),
                        brows(b_cu_sam, nt + 1), brows(b_att, BT),
                        RAD_W(l.w_rh), RAD_W(l.w_rw));
        }
        char tag[40];
        std::snprintf(tag, sizeof tag, "sam_att%lld_raw", (long long)li);
        if (tap) w.tap(c, tag, b_att, 256, DeepEncoder::kSamW);
        RAD_ISSUE_N(c, l.op_proj, BT, brows(b_att, BT), RAD_W(l.w_proj),
                    RAD_W(l.b_proj), RAD_NONE, brows(b_h, BT));
        RAD_ISSUE_N(c, l.op_padd, BT, brows(b_h, BT), brows(b_x, BT), brows(b_x, BT));
        std::snprintf(tag, sizeof tag, "st_pre%lld", (long long)li);
        if (tap) w.tap(c, tag, b_x, 256, DeepEncoder::kSamW);
        RAD_ISSUE_N(c, l.op_n2, BT, brows(b_x, BT), RAD_W(l.w_n2), RAD_W(l.b_n2),
                    brows(b_h, BT));
        RAD_ISSUE_N(c, l.op_l1, BT, brows(b_h, BT), RAD_W(l.w_l1), RAD_W(l.b_l1),
                    RAD_NONE, brows(b_ff, BT));
        RAD_ISSUE_N(c, l.op_l2, BT, brows(b_ff, BT), RAD_W(l.w_l2), RAD_W(l.b_l2),
                    RAD_NONE, brows(b_h, BT));
        RAD_ISSUE_N(c, l.op_ladd, BT, brows(b_h, BT), brows(b_x, BT), brows(b_x, BT));
        std::snprintf(tag, sizeof tag, "st_b%lld", (long long)li);
        if (tap) w.tap(c, tag, b_x, 256, DeepEncoder::kSamW);
        if (li < 3) {
            std::snprintf(tag, sizeof tag, "st_full%lld", (long long)li);
            if (tap) w.tap(c, tag, b_x, kSamTok, DeepEncoder::kSamW);
        }
    }

    /* neck + net_2 + net_3. The 3x3s are a dense view of the map rows, unfolded; the cu table
     * gives each tile its own grid. */
    RAD_ISSUE_N(c, op_neck0, BT, brows(b_x, BT), RAD_W(w.w_neck0), brows(b_n0, BT));
    if (tap) w.tap(c, "st_n0", b_n0, 256, 256);
    RAD_ISSUE_N(c, op_nk1, BT, brows(b_n0, BT), RAD_W(w.w_nk1), RAD_W(w.b_nk1),
                brows(b_n1, BT));
    if (tap) w.tap(c, "st_n1", b_n1, 256, 256);
    RAD_ISSUE(c, op_unf3,
              RAD_P_T2(rad_buf_ptr(c, b_n1), RAD_BF16, nt * kGrid, kGrid * 256),
              brows(b_cu_sam, nt + 1), RAD_NONE, brows(b_unf3, BT));
    RAD_ISSUE_N(c, op_nk2, BT, brows(b_unf3, BT), RAD_W(w.w_nk2), brows(b_n2, BT));
    if (tap) w.tap(c, "st_n2", b_n2, 256, 256);
    RAD_ISSUE_N(c, op_nk3, BT, brows(b_n2, BT), RAD_W(w.w_nk3), RAD_W(w.b_nk3),
                brows(b_n3, BT));
    if (tap) w.tap(c, "st_n3", b_n3, 256, 256);
    RAD_ISSUE(c, op_unf_net2,
              RAD_P_T2(rad_buf_ptr(c, b_n3), RAD_BF16, nt * kGrid, kGrid * 256),
              brows(b_cu_sam, nt + 1), RAD_NONE, brows(b_unf_2, nt * kSamTok / 4));
    RAD_ISSUE_N(c, op_net2, nt * kSamTok / 4, brows(b_unf_2, nt * kSamTok / 4),
                RAD_W(w.w_net2), brows(b_map2, nt * kSamTok / 4));
    if (tap) w.tap(c, "st_map2", b_map2, 256, 512);
    RAD_ISSUE(c, op_unf_net3,
              RAD_P_T2(rad_buf_ptr(c, b_map2), RAD_BF16, nt * kGrid / 2, (kGrid / 2) * 512),
              brows(b_cu_sam, nt + 1), RAD_NONE, brows(b_unf_3, BM));
    RAD_ISSUE_N(c, op_net3, BM, brows(b_unf_3, BM), RAD_W(w.w_net3), brows(b_map, BM));
    if (tap) w.tap(c, "st_map", b_map, kMap * kMap, DeepEncoder::kClipW);

    /* CLIP: the class row and the map, tile by tile (each block's rows are strided in the
     * tile-major plane), then the batched transformer and absolute table. */
    for (int64_t t = 0; t < nt; ++t) {
        RAD_ISSUE(c, op_cls, RAD_B(p_one_c), RAD_W(w.w_cls),
                  brow_slice(b_c, t * kClipTok, 1, DeepEncoder::kClipW));
        RAD_ISSUE(c, op_cmap,
                  brow_slice(b_map, t * kMap * kMap, kMap * kMap, DeepEncoder::kClipW),
                  brow_slice(b_c, t * kClipTok + 1, kMap * kMap, DeepEncoder::kClipW));
    }
    if (b_cpos_tab)
        RAD_ISSUE_N(c, op_cpos, BCT, RAD_B(b_cpos_tab), brows(b_ids_clip, BCT),
                    brows(b_cpos, BCT));
    else
        RAD_ISSUE_N(c, op_cpos, BCT, brows(b_ids_clip, BCT), RAD_W(w.w_cpos),
                    brows(b_cpos, BCT));
    RAD_ISSUE_N(c, op_cadd, BCT, brows(b_c, BCT), brows(b_cpos, BCT), brows(b_c, BCT));
    if (tap) w.tap(c, "clip_x", b_c, kClipTok, DeepEncoder::kClipW);
    RAD_ISSUE_N(c, op_playr, BCT, brows(b_c, BCT), RAD_W(w.w_playr),
                RAD_W(w.b_playr), brows(b_cp, BCT));
    RAD_ISSUE_N(c, op_ccopy, BCT, brows(b_cp, BCT), brows(b_c, BCT));

    for (size_t ci = 0; ci < clip.size(); ++ci) {
        const ClipLayerT<G>& l = clip[ci];
        char ctag[40];
        RAD_ISSUE_N(c, l.op_n1, BCT, brows(b_c, BCT), RAD_W(l.w_n1),
                    RAD_W(l.b_n1), brows(b_ch, BCT));
        std::snprintf(ctag, sizeof ctag, "clip_pre%lld", (long long)ci);
        if (tap) w.tap(c, ctag, b_ch, kClipTok, DeepEncoder::kClipW);
        RAD_ISSUE_N(c, l.op_qkv, BCT, brows(b_ch, BCT), RAD_W(l.w_qkv),
                    RAD_W(l.b_qkv), RAD_NONE, brows(b_cq, BCT));
        RAD_ISSUE_N(c, l.op_attn, BCT,
                    bcol(b_cq, 0, DeepEncoder::kClipW, BCT),
                    bcol(b_cq, DeepEncoder::kClipW, DeepEncoder::kClipW, BCT),
                    bcol(b_cq, 2 * DeepEncoder::kClipW, DeepEncoder::kClipW, BCT),
                    brows(b_cu_clip, nt + 1), brows(b_ca, BCT));
        std::snprintf(ctag, sizeof ctag, "clip_att%lld", (long long)ci);
        if (tap) w.tap(c, ctag, b_ca, kClipTok, DeepEncoder::kClipW);
        RAD_ISSUE_N(c, l.op_out, BCT, brows(b_ca, BCT), RAD_W(l.w_out),
                    RAD_W(l.b_out), RAD_NONE, brows(b_ct, BCT));
        RAD_ISSUE_N(c, l.op_cadd, BCT, brows(b_ct, BCT), brows(b_c, BCT), brows(b_c, BCT));
        std::snprintf(ctag, sizeof ctag, "clip_h%lld", (long long)ci);
        if (tap) w.tap(c, ctag, b_c, kClipTok, DeepEncoder::kClipW);
        RAD_ISSUE_N(c, l.op_n2, BCT, brows(b_c, BCT), RAD_W(l.w_n2),
                    RAD_W(l.b_n2), brows(b_ch, BCT));
        RAD_ISSUE_N(c, l.op_fc1, BCT, brows(b_ch, BCT), RAD_W(l.w_fc1),
                    RAD_W(l.b_fc1), RAD_NONE, brows(b_cf, BCT));
        RAD_ISSUE_N(c, l.op_qg, BCT, brows(b_cf, BCT), brows(b_cf, BCT),
                    brows(b_cg2, BCT));
        RAD_ISSUE_N(c, l.op_fc2, BCT, brows(b_cg2, BCT), RAD_W(l.w_fc2),
                    RAD_W(l.b_fc2), RAD_NONE, brows(b_ct, BCT));
        RAD_ISSUE_N(c, l.op_madd, BCT, brows(b_ct, BCT), brows(b_c, BCT), brows(b_c, BCT));
        std::snprintf(ctag, sizeof ctag, "clip_b%lld", (long long)ci);
        if (tap) w.tap(c, ctag, b_c, kClipTok, DeepEncoder::kClipW);
    }

    if (tap) w.tap(c, "clip_out", b_c, kMap * kMap, DeepEncoder::kClipW, 1);

    /* Fuse (CLIP's patch rows | the map) and project, tile by tile into the tile-major fuse. */
    for (int64_t t = 0; t < nt; ++t) {
        RAD_ISSUE(c, op_fuse_clip,
                  brow_slice(b_c, t * kClipTok + 1, kMap * kMap, DeepEncoder::kClipW),
                  bcol_at(b_fuse, t * kMap * kMap, DeepEncoder::kFused, 0,
                          DeepEncoder::kClipW, kMap * kMap));
        RAD_ISSUE(c, op_fuse_sam,
                  brow_slice(b_map, t * kMap * kMap, kMap * kMap, DeepEncoder::kClipW),
                  bcol_at(b_fuse, t * kMap * kMap, DeepEncoder::kFused, DeepEncoder::kClipW,
                          DeepEncoder::kClipW, kMap * kMap));
    }
    if (tap) w.tap(c, "fuse", b_fuse, kMap * kMap, DeepEncoder::kFused);
    RAD_ISSUE_N(c, op_proj, BM, brows(b_fuse, BM), RAD_W(w.w_proj),
                RAD_W(w.b_proj), RAD_NONE, brows(b_projbuf, BM));
    if (tap) w.tap(c, "proj", b_projbuf, kMap * kMap, DeepEncoder::kNEmbd);


    /* The layout: each tile's map goes to columns [t*kMap, (t+1)*kMap) of every map row -- a
     * pitched copy of the projected tile into the plane (the layout pitch is the line's row
     * count plus the newline column), one copy replacing kMap cast-launches. The newline cell
     * after a line's last tile belongs to the line and follows that tile's copy. */
    char* po = (char*)rad_buf_ptr(c, p_out);
    const char* proj = (const char*)rad_buf_ptr(c, b_projbuf);
    const int64_t wbytes = kMap * DeepEncoder::kNEmbd * 2;
    if (!po || !proj) {
        rad_step_fail(c, "unlimitedocr: a layout plane of the DeepEncoder did not bind");
        return;
    }
    for (int64_t t = 0; t < nt; ++t) {
        const int64_t rowbase = tiles[t].rowbase;
        const int64_t tt = tiles[t].t, n = tiles[t].n;
        if (rad_memcpy_2d_async(po + (rowbase + tt * kMap) * DeepEncoder::kNEmbd * 2,
                                (n * kMap + 1) * DeepEncoder::kNEmbd * 2,
                                proj + t * kMap * kMap * DeepEncoder::kNEmbd * 2,
                                wbytes, wbytes, kMap, rad_stream(c)) < 0) {
            rad_step_fail(c, "unlimitedocr: the projected map of the DeepEncoder could not be "
                             "laid out");
            return;
        }
        if (tiles[t].nl) {
            /* The reference expands image_newline over every map row of the line
             * (modeling_unlimitedocr.py: `image_newline.expand(h, 1, n_dim)`), so the one
             * learned row is replicated into each newline cell. Copy one cell at a time:
             * b_nlrow is a single row, and a 2D copy with a zero source pitch is not a
             * portable device operation. */
            RAD_ISSUE(c, op_newline, RAD_B(p_one), RAD_W(w.w_newline), RAD_B(b_nlrow));
            const char* nl = (const char*)rad_buf_ptr(c, b_nlrow);
            if (!nl) {
                rad_step_fail(c, "unlimitedocr: the newline row of the DeepEncoder did not bind");
                return;
            }
            for (int64_t r = 0; r < kMap; ++r) {
                char* dst = po + ((rowbase + n * kMap) + r * (n * kMap + 1)) *
                                   DeepEncoder::kNEmbd * 2;
                if (rad_memcpy_async(dst, nl, DeepEncoder::kNEmbd * 2, rad_stream(c)) < 0) {
                    rad_step_fail(c, "unlimitedocr: the newline row of the DeepEncoder could "
                                     "not be laid out");
                    return;
                }
            }
        }
    }
}

}  /* namespace arch */
}  /* namespace rad */

#endif  /* RAD_BLOCK_SAM_H */
