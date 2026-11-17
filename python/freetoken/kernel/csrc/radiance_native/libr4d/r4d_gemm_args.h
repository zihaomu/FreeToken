/* r4d_gemm_args.h -- the skinny GEMMs' operand lists and their split law.
 *
 * The launch law (ladders, caps, `pick_sk`, a bound rule per weight grid) lives beside the kernels
 * so an out-of-tree kernel library carries its own rule. The tunable axes (r4d_rows.cpp) are what
 * a tuner walks; this resolves a shape left on the axis defaults.
 *
 * Eleven kernels read one of two operand lists: [A, B, Y] for bf16 and
 * [A, a_scale, B, b_scale, Y, a_sum, B_ref] for every quantised grid.
 */
#ifndef R4D_GEMM_ARGS_H
#define R4D_GEMM_ARGS_H

#include "r4d.h"
#include "r4d_args.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <exception>


/* K per scale group of each quantised weight layout. A mismatch between weight file and kernel
 * would be silently wrong, so each is checked against r4d_args.h's layout constant. */
#define R4D_GEMM_W4_GROUP   128
#define R4D_GEMM_W8_GROUP   128
#define R4D_GEMM_W4A8_GROUP 128
#define R4D_GEMM_W8A8_GROUP 128
#define R4D_GEMM_W2A8_GROUP 128
static_assert(R4D_W4_GROUP == R4D_GEMM_W4_GROUP,   "w4 group differs from the layout constant");
static_assert(R4D_W8_GROUP == R4D_GEMM_W8_GROUP,   "w8 group differs from the layout constant");
static_assert(R4D_W4_GROUP == R4D_GEMM_W4A8_GROUP, "w4a8 group differs from the layout constant");
static_assert(R4D_W8_GROUP == R4D_GEMM_W8A8_GROUP, "w8a8 group differs from the layout constant");
static_assert(R4D_W2_GROUP == R4D_GEMM_W2A8_GROUP, "w2a8 group differs from the layout constant");


static inline int gemm_caught(const char* who, const char* what) {
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "[r4d] %s threw: %s\n", who, what ? what : "(null)");
    }
    return R4D_NO(RAD_E_DEVICE);
}

#define R4D_TRY(who, ...)                                                          \
    do {                                                                           \
        try { __VA_ARGS__; }                                                       \
        catch (const std::exception& e_) { return gemm_caught(who, e_.what()); }   \
        catch (...) { return gemm_caught(who, "unknown exception"); }              \
    } while (0)

/* ------------------------------------------------------------------ the split count
 *
 * The skinny GEMMs split K inside the workgroup: SK k-splits per column tile, reduced in LDS, to
 * fill a 64-CU part at M=1. Legality is K % (SK * unit) == 0, so no single SK is legal at every K,
 * and an illegal default would make an untuned install fail instead of run slowly (spec §15).
 * The `sk` axis therefore defaults to 0, meaning "the largest legal split for this K" from a
 * descending ladder of split counts tuned configurations use. A 0 in --debug-graph means the law
 * decided, not a literal split.
 */
/* The launch bound is part of a split's legality. The skinny kernels' __launch_bounds__ tighten as
 * MB and NPW grow (w2a8: (MB >= 4) ? 256 : ((MB >= 2 || NPW >= 4) ? 512 : 1024)), and a block of
 * WV*SK*32 threads over its bound fails to launch, surfacing later as a device error at the next
 * sync. MB comes from M, so this only bites under concurrency: on the DFlash2 2-bit draft head
 * (N = 248320), M >= 49 gives a bound of 256 where sk = 10 needs 320 threads.
 *
 * Each caller names its own rule because one conservative rule costs throughput: bf16, w4a16,
 * w8a16 and mxfp4 are a flat 1024; w4a8 tightens on NPW and w8a8 on MB*NPW (a register budget).
 * Where a build flag can disable a kernel's bound, the tight rule is named anyway: over-constraining
 * loses a little parallelism, under-constraining fails to launch.
 */
enum BoundRule { BR_FLAT = 0, BR_W2A8, BR_NPW, BR_MBNPW };

static inline int block_bound(BoundRule r, int mb, int npw) {
    switch (r) {
        case BR_W2A8:  return (mb >= 4) ? 256 : (((mb >= 2) || (npw >= 4)) ? 512 : 1024);
        case BR_NPW:   return (npw >= 8) ? 256 : ((npw >= 4) ? 512 : 1024);
        case BR_MBNPW: return (mb * npw >= 8) ? 256 : ((mb * npw >= 4) ? 512 : 1024);
        default:       return 1024;
    }
}

/* Where N already supplies the parallelism, the largest legal split is wrong: it only adds
 * reduction work. At K = 5120 the fastest split falls as N grows: largest legal at N = 48 (3 N
 * tiles), 4 at N = 256 (16 tiles), 1 or 2 at N = 1280 (80 tiles). So with `n_tiles` (the shape's
 * 16-row N tiles) this picks the smallest split whose `n_tiles * sk` reaches kFlatTileTarget
 * workgroups. Without it, or when none reaches, it picks the largest legal split. kFlatTileTarget
 * is the same quantity as kSplitTarget in the fp8 kernel's `narrow_ksplit`. */
enum { kFlatTileTarget = 128 };

static inline int pick_sk(long long n_tiles, long long K, long long unit, int wv, int bound) {
    /* The top entry is the largest split an untuned shape gets. 20 divides K = 5120 and is the
     * fastest legal split there on the GDN a|b projection; 32 is left out on purpose (legal at
     * K = 5120 but slower). `cap` keeps the block within its bound (wv = 2 cannot reach 20). */
    static const int ladder[] = { 1, 2, 4, 5, 8, 10, 16, 20 };
    const int cap = (wv > 0) ? (bound / (wv * 32)) : 1;   /* threads/block <= this MB/NPW's bound */
    int best = 1;
    for (int s : ladder) {
        if (s > cap || unit <= 0 || (K % ((long long)s * unit))) continue;
        best = s;
        if (n_tiles > 0 && n_tiles * s >= (long long)kFlatTileTarget) return s;
    }
    return best;
}

/* Resolve the tuned point into (WV, SK, MB, NPW, NT) for a shape whose split unit is `unit`. */
struct Cfg { int wv, sk, mb, npw, nt; };

/* `n_tiles` is the shape's number of 16-row N tiles, for pick_sk. Zero keeps the largest legal
 * split, right wherever the split is the only source of parallelism. */
static inline int cfg_for(const RadArgs* a, long long K, long long M, long long unit, BoundRule br,
                          Cfg* out, long long n_tiles = 0) {
    R4dGemmCfg cv;
    r4d_cfg_gemm(a, &cv);
    const R4dGemmCfg* c = &cv;
    out->wv  = c->wv  > 0 ? c->wv  : 1;
    /* MB is the number of row tiles one block covers; mb <= 0 means all of them. grid.y is
     * ceil(ceil(M/16)/MB) and each grid.y row of blocks reads the whole weight, so MB = 1 at M = 64
     * reads it four times. Covering every row tile is right when the weight outweighs the
     * activation. */
    out->mb  = c->mb  > 0 ? c->mb : (int)((M + 15) / 16);
    if (out->mb < 1) out->mb = 1;
    if (out->mb > 4) out->mb = 4;
    /* In the decode band (M <= 64, four row tiles) the grid is the cost, not the weight: at
     * N = 48, MB = ceil(M/16) gives a three-block grid on a 32-WGP part. MB = 1 is fastest at
     * every decode (N, M) these callers run despite re-reading the weight. Prefill (hundreds of
     * row tiles) is excluded by the bound.
     *
     * BR_FLAT only: its bound ignores MB, so the split is unchanged and every output element keeps
     * the same accumulation order, hence the same bits. Under BR_W2A8 or BR_MBNPW, MB would move
     * the split. */
    if (c->mb <= 0 && br == BR_FLAT && ((M + 15) / 16) <= 4) out->mb = 1;
    /* Also past the decode band when N is narrow. Above 64 rows, N >= 128 goes to the tiled row
     * (r4d_gemm_bf16_nt_tiled.hip); what stays here are narrow projections (GDN a|b at 48, a
     * shared-expert gate at 1) whose weight is small against the activation. At K = 2560,
     * M 80..2048, N 16..112, MB = 1 is 2-5x faster. BR_FLAT keeps the bits as above. */
    if (c->mb <= 0 && br == BR_FLAT && n_tiles > 0 && n_tiles < 8) out->mb = 1;
    out->npw = c->npw > 0 ? c->npw : 1;
    out->nt  = c->nt;
    /* W2A8 past one row tile needs a wider block, not a taller one. Its only caller (the DFlash2
     * draft head, N = 124160, K = 5120) runs M = n_seq * steps, so M > 16 from the second
     * concurrent sequence on. At NPW = 1 the kernel then drops to about a third of memory
     * bandwidth; NPW = 4 gives each block four N tiles to hide the dequantise behind. MB is pinned
     * at 2: MB = 4 with NPW = 4 spills (MT * NPW accumulators), and MB = 3 (12 accumulators vs 8)
     * is slower even where it moves the same bytes. Reading the weight twice is cheaper than any
     * shape that reads it once; see rad_block_dflash2.h for the vocabulary shard.
     *
     * The split is pinned to 8: pick_sk would answer 10 (divisors of 40 up to 16), and 8 is faster
     * across this M band. Only mb = 0 is touched, so a tuned mb survives. A pin this K cannot
     * divide or that overruns the bound falls back to pick_sk, since the auto path must serve
     * every shape. */
    int sk_pin = 0;
    if (br == BR_W2A8 && c->mb <= 0 && out->mb >= 2) {
        out->mb  = 2;
        out->npw = 4;
        sk_pin   = 8;
    }
    /* W2A8 at one row tile is bound by the weight stream (a vocabulary head of 80 to 170 MB), and
     * its N tiles already fill the card, so a deep split only adds reduction work: the ladder's
     * answer of 20 at K = 2560 runs the 124160-row head in 144 us where a split of 2 takes 133. */
    if (br == BR_W2A8 && c->mb <= 0 && out->mb == 1) sk_pin = 2;
    const int bound = block_bound(br, out->mb, out->npw);
    if (sk_pin && (out->wv * sk_pin * 32 > bound ||
                   (unit > 0 && (K % ((long long)sk_pin * unit)))))
        sk_pin = 0;
    out->sk  = c->sk > 0 ? c->sk : (sk_pin ? sk_pin : pick_sk(n_tiles, K, unit, out->wv, bound));
    /* A named split that overruns the bound is refused, so the tuner never records the candidate
     * and the selector cannot hand a cached name to a shape whose MB has grown. */
    if (out->wv * out->sk * 32 > bound) return R4D_NO(RAD_E_UNSUPPORTED);
    /* A named split this K cannot divide is refused rather than clamped, so the tuner never
     * records a configuration that did not run. */
    if (unit > 0 && (K % ((long long)out->sk * unit))) return R4D_NO(RAD_E_SHAPE);
    /* R4D_GEMMCFG=1 prints each distinct (tuned point, K, M, n_tiles) resolution once. */
    static const bool dbg = [] { const char* e = std::getenv("R4D_GEMMCFG");
                                 return e && e[0] && e[0] != '0'; }();
    if (dbg) {
        /* At most 64 keys: this runs on every issue of every GEMM row. */
        static uint64_t seen[64];
        static int nseen = 0;
        const uint64_t key = ((uint64_t)(unsigned)(c->wv * 41 + c->sk * 7 + c->mb * 3 + c->npw) << 56)
                           ^ ((uint64_t)(unsigned)br << 52)
                           ^ ((uint64_t)(unsigned)n_tiles << 26) ^ (uint64_t)(unsigned)M
                           ^ ((uint64_t)(unsigned)K << 13);
        int i = 0;
        while (i < nseen && seen[i] != key) ++i;
        if (i == nseen && nseen < 64) {
            seen[nseen++] = key;
            std::fprintf(stderr,
                "[gemmcfg] K=%lld M=%lld tiles=%lld br=%d -> wv=%d sk=%d mb=%d npw=%d nt=%d\n",
                (long long)K, (long long)M, (long long)n_tiles, (int)br,
                out->wv, out->sk, out->mb, out->npw, out->nt);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ operand plumbing */

enum { G_A = 0, G_B, G_Y };                                     /* gemm_nt          */
enum { Q_A = 0, Q_ASCALE, Q_B, Q_BSCALE, Q_Y, Q_ASUM, Q_BREF };  /* gemm_nt_q       */

struct GemmOps {
    /* Flattened views of a rank-3 `a` / `y`; see gemm_ops. Held by value because the operand
     * array belongs to the core. */
    RadTensor a_flat, y_flat;
    const RadTensor *a, *ascale, *b, *bscale, *y, *asum, *bref;
    long long M, N, K;
};

/* The MXFP4 GEMMs' activation scale grid, read off the operand. One f32 per row (`[M]` or
 * `[M][1]`, a per-token quantiser) is group 0; one per (row, 128 of K) (`[M][ceil(K/128)]`, the
 * block-scaled fp8 quantisers every fused producer in arch/common writes) is group 128, read at the
 * operand's row stride. Anything else is refused. */
static inline int mx_ascale_grid(const GemmOps& g, int* group, int* rs, int parts = 1) {
    const RadTensor* s = g.ascale;
    if (!s || g.M <= 0) return R4D_NO(RAD_E_INVAL);
    long long cols = 1, rstride = 1;
    if (s->rank >= 2) {
        cols    = s->shape[s->rank - 1];
        rstride = s->stride[s->rank - 2];
        if (s->shape[s->rank - 2] < g.M) return R4D_NO(RAD_E_SHAPE);
    } else if (s->rank == 1) {
        if (s->shape[0] < g.M) return R4D_NO(RAD_E_SHAPE);
        cols = s->shape[0] / g.M;
        rstride = cols;
    }
    const long long kb = (g.K + 127) / 128;
    /* With parts, the activation's copies sit side by side in a row, so its scale row is `parts`
     * times as long; each part reads its own kb entries. */
    if (parts == 1 && cols == 1 && (s->rank < 2 || rstride == 1)) { *group = 0; *rs = 1; return RAD_OK; }
    if (cols == kb * parts && rstride >= kb * parts) { *group = 128; *rs = (int)rstride; return RAD_OK; }
    return R4D_NO(RAD_E_SHAPE);
}

/* The parts of a partitioned MXFP4 weight (gemm_nt_q's `pb1`/`pb2`), as launches.
 *
 * A fused projection whose pieces were rotated apart (ParoQuant: gate|up, q|k|v) has one copy of
 * the activation per piece, side by side in a row of `a` ([M, parts * K]), and its scales likewise;
 * weight rows [0, pb1), [pb1, pb2), [pb2, N) each meet their own copy. A part is the ordinary
 * product over a window of the planes: activation from byte p * K of each row, scales from
 * p * ceil(K/128), fragment-order weight from tile n0 / 16 (byte n0 * K / 2), exponents, reference
 * and output from column n0 at the full planes' row pitch. The kernels learn only row pitches
 * (`lda`, `ldws`, `ldc`), and each part is one launch of the kernel the shape would pick anyway.
 *
 * `pb1` = 0 is one part and `pb2` = 0 ends the second part at N, as libref reads them. Bounds are
 * whole 16-row tiles. Fills `p`, corrects K to the op's own (gemm_ops read parts * K) and sets the
 * activation's row pitch in `lda`. */
struct MxParts {
    int       n = 1;
    long long n0[3] = { 0, 0, 0 }, n1[3] = { 0, 0, 0 };
    long long lda = 0;                  /* 0: the plane's own K */
};

/* The bounds `pb1`/`pb2` name over N rows, as every reader takes them: { 0, e1, e2, N }, part i
 * being [bound[i], bound[i+1]). Returns how many copies of the activation they imply, 1..3. */
static inline int mx_part_bounds(long long N, long long pb1, long long pb2, long long bound[4]) {
    const long long e1 = pb1 > 0 && pb1 < N ? pb1 : N;
    long long e2 = pb2 > 0 && pb2 < N ? pb2 : N;
    if (e2 < e1) e2 = e1;
    bound[0] = 0;
    bound[1] = e1;
    bound[2] = e2;
    bound[3] = N;
    return e2 < N ? 3 : (e1 < N ? 2 : 1);
}

static inline int mx_parts(const RadArgs* a, GemmOps* g, MxParts* p) {
    p->n = 1;
    p->n0[0] = 0;
    p->n1[0] = g->N;
    p->lda = 0;
    long long bound[4];
    const int segs = mx_part_bounds(g->N, r4d_geti_or(a, "pb1", 0), r4d_geti_or(a, "pb2", 0), bound);
    if (segs == 1) return RAD_OK;
    const long long K = r4d_geti_or(a, "K", 0);
    if (K <= 0) return R4D_NO(RAD_E_INVAL);
    if (g->K != (long long)segs * K || (bound[1] % 16) || (bound[2] % 16)) return R4D_NO(RAD_E_SHAPE);
    p->n = 0;
    for (int i = 0; i < 3; ++i)
        if (bound[i + 1] > bound[i]) {
            p->n0[p->n] = bound[i];
            p->n1[p->n] = bound[i + 1];
            ++p->n;
        }
    if (p->n != segs) return R4D_NO(RAD_E_SHAPE);        /* an empty middle part has no copy to meet */
    p->lda = g->K;
    g->K = K;
    return RAD_OK;
}

/* Folds a contiguous rank-3+ operand to rank 2 (rows x the product of the rest), held by value in
 * `flat` because the operand array is the core's; a rank-2 operand is left as it is. */
static inline int fold_rows(const RadTensor** t, RadTensor* flat) {
    if ((*t)->rank < 2) return R4D_NO(RAD_E_SHAPE);
    if ((*t)->rank == 2) return RAD_OK;
    if (!r4d_contig(*t)) return R4D_NO(RAD_E_STRIDE);
    *flat = **t;
    int64_t k = 1;
    for (uint32_t d = 1; d < (*t)->rank; ++d) k *= (*t)->shape[d];
    flat->rank = 2;
    flat->shape[1] = k;  flat->stride[1] = 1;
    flat->stride[0] = k;
    *t = flat;
    return RAD_OK;
}

/* Common shape and layout checks. A is [M,K] row-major, C is [M,N] row-major. The weight is an
 * opaque packed blob whose size the layout hook owns (RadLayout.bytes), so only its presence is
 * checked here.
 *
 * `parts`: the caller applies a partitioned weight's `pb1`/`pb2` (mx_parts). Any other caller
 * refuses them; ignoring them would multiply every weight row by the first activation copy. */
static inline int gemm_ops(const RadArgs* a, GemmOps* g, int quantised, int parts = 0) {
    std::memset(g, 0, sizeof(*g));
    if (quantised && !parts && (r4d_geti_or(a, "pb1", 0) > 0 || r4d_geti_or(a, "pb2", 0) > 0))
        return R4D_NO(RAD_E_UNSUPPORTED);
    if (quantised) {
        g->a      = r4d_opt(a, Q_A);
        g->ascale = r4d_opt(a, Q_ASCALE);
        g->b      = r4d_opt(a, Q_B);
        g->bscale = r4d_opt(a, Q_BSCALE);
        g->y      = r4d_opt(a, Q_Y);
        g->asum   = r4d_opt(a, Q_ASUM);
        g->bref   = r4d_opt(a, Q_BREF);
        if (!g->a || !g->b || !g->bscale || !g->y) return R4D_NO(RAD_E_INVAL);
    } else {
        g->a = r4d_opt(a, G_A);
        g->b = r4d_opt(a, G_B);
        g->y = r4d_opt(a, G_Y);
        if (!g->a || !g->b || !g->y) return R4D_NO(RAD_E_INVAL);
    }
    /* The gated delta net hands its [tokens, heads, head_v] output (as `gdn_chunk_scan` declares
     * `o`) to `gemm_nt`, so K is the product of the trailing dims, not shape[1]. Gated attention
     * writes q|gate into [tokens, heads, 2*head_dim], so q and gate are column slices of the last
     * dim and N is the product of the trailing two. A strided rank-3 tensor is refused. */
    int rc = fold_rows(&g->a, &g->a_flat);
    if (rc != RAD_OK) return rc;
    rc = fold_rows(&g->y, &g->y_flat);
    if (rc != RAD_OK) return rc;
    if (!r4d_contig(g->a) || !r4d_contig(g->y)) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_aligned(g->a->data, 16) || !r4d_aligned(g->b->data, 16)) return R4D_NO(RAD_E_ALIGN);

    g->M = g->a->shape[0];
    g->K = g->a->shape[1];
    g->N = g->y->shape[1];
    if (g->y->shape[0] != g->M) return R4D_NO(RAD_E_SHAPE);
    if (g->M < 1 || g->N < 1 || g->K < 1) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}



/* The asymmetric grid is a different weight format: the nibble is read unsigned and the per-group
 * zero is corrected with the activation's per-group row sums. It is selected by `dtype`, never by
 * whether a_sum is present: inferring it would silently give the symmetric answer to a caller
 * that forgot the sums. */
static inline int is_asym(const RadArgs* a) {
    const char* dt = r4d_gets(a, "dtype");
    return dt && std::strstr(dt, "_asym") != nullptr;
}

/* The tiled and prefill kernels take an opaque tile variant into libr4d's tile switch. N must be a
 * multiple of the variant's BN, which the row cannot express since BN varies; the row constrains
 * N % 256 (the default BN) and this refuses a variant whose BN does not divide N. */
static inline int tile_ok(const R4dTileCfg* t, long long N) {
    if (!t) return R4D_NO(RAD_E_INVAL);
    if (t->bn > 0 && (N % t->bn)) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}


#endif  /* R4D_GEMM_ARGS_H */
