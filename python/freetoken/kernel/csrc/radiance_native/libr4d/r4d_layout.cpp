/* r4d_layout.cpp -- the stored form of libr4d's integer and MXFP4 GEMM weights, and the relayouts
 * that produce it from the weight's canonical planes (spec.md §4.3). Pure host C++: nothing here
 * includes r4d.h or <hip/hip_runtime.h>, so packing can be tested without ROCm.
 *
 * A container holds each weight's encoding (codes, scales and zero points, canonical and
 * row-major), and the kernel that reads it rearranges it at load, so a fragment order can change
 * without reconverting models. The numbers are chosen by a quantiser (libquant), never here.
 *
 * Fragment order. gfx1201's wave32 WMMA fragment layout is
 *
 *     idx = lane % 16 ,  k = 8*(e>>2) + 4*(lane>>4) + (e&3)
 *
 * so a B fragment maps lane l onto output row l&15. In canonical order the sixteen lanes of a
 * half-wave read sixteen rows K/2 bytes apart, a fully strided read on the operand that dominates
 * traffic. Rearranged, a wave's weight load for four k steps is one 512-byte global_load_b128 with
 * a constant stride and no address arithmetic in the loop.
 *
 * Every element->slot map below is transcribed from the kernel that reads it; each libr4d GEMM's
 * header states it in closed form.
 */

#include "r4d_plugin.h"
#include "r4d_hsimd.h"
#include "r4d_par.h"
#include "r4d_enc.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

/* ------------------------------------------------------------------ half-precision
 *
 * Written out so the file builds without ROCm or a float16 extension: __fp16 is not portable and
 * _Float16 is not available on every host compiler. */

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu)                    /* inf / nan */
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);  /* overflow -> inf */
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;            /* underflow -> zero */
        man |= 0x800000u;                                /* subnormal: shift in the hidden bit */
        const int shift = 14 - exp;
        const uint32_t v = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        uint32_t out = v;
        if (rem > half || (rem == half && (v & 1u))) ++out;   /* round to nearest even */
        return (uint16_t)(sign | out);
    }
    uint32_t v = ((uint32_t)exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1u))) ++v;
    return (uint16_t)(sign | v);
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t man  = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) out = sign;
        else {
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; ++e; } while (!(m & 0x400u));
            out = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

/* The scale dword: a group's f16 scale in the low half and the f16 of -(1024 + zero) in the high
 * half, broadcast into a v_pk_add_f16. The kernel builds 1024 + code and adds that constant, which
 * subtracts an integer zero point exactly. A symmetric 4-bit grid uses zero == 8, so one packed
 * weight feeds both the f16 and the int8 kernel. */
static uint32_t scale_zero_dword(float scale, int zero) {
    const uint32_t lo = f32_to_f16(scale);
    const uint32_t hi = f32_to_f16(-(1024.0f + (float)zero));
    return lo | (hi << 16);
}

/* Geometry the layout hooks read out of the declared parameters. */
struct WGeom { long long N, K, group; };

static int wgeom(const RadParam* p, int n_p, long long dflt_group, WGeom* g) {
    g->N = r4d_pgeti_or(p, n_p, "N", 0);
    g->K = r4d_pgeti_or(p, n_p, "K", 0);
    g->group = r4d_pgeti_or(p, n_p, "group", dflt_group);
    if (g->N <= 0 || g->K <= 0 || g->group <= 0) return RAD_E_INVAL;
    if (g->N % R4D_NTILE) return RAD_E_SHAPE;
    if (g->K % g->group) return RAD_E_SHAPE;
    return RAD_OK;
}

/* Operand indices in `gemm_nt_q` (see oGemmNtQ in r4d_rows.cpp). */
enum { Q_B = 2, Q_BSCALE = 3, Q_BREF = 6 };

static void fill(RadLayout* o, const char* tag, uint32_t dtype, long long r0, long long r1,
                 long long bytes) {
    std::memset(o, 0, sizeof(*o));
    o->tag = tag;
    o->dtype = dtype;
    o->rank = 2;
    o->shape[0] = r0;
    o->shape[1] = r1;
    /* Every stream is read with 128-bit loads; 256 bytes is a cacheline on the target
     * architectures and the .rad plane alignment. */
    o->align = 256;
    o->bytes = bytes;
}

/* The asymmetric 4-bit grid is a different weight, and `dtype` says which. `w4a8_asym` kernels
 * read the nibble unsigned and subtract a per-group integer zero from the scale dword's high half;
 * the symmetric grid stores a two's-complement nibble with the zero fixed at 8. Packing, extents
 * and byte counts are identical, so a plane read as the other grid decodes eight levels off. This
 * is the kernels' own test (r4d_gemm_args.h is_asym), so the layout hook accepts only the matching
 * encoding (i4 without a zero, u4 with one) and refuses the other at declare. */
static bool w4_asym(const RadParam* p, int n_p) {
    const char* d = r4d_pgets(p, n_p, "dtype");
    return d && std::strstr(d, "_asym") != nullptr;
}

/* ================================================================== moe w4
 *
 * The grouped MoE GEMM's 4-bit expert weight. Unlike the dense w4 it is not permuted into fragment
 * order: the grouped kernel stages its weight through LDS (eight M tiles share one N tile), so a
 * permutation would only have to be undone.
 *
 *   weight  [N][K/2] bytes, row-major, two's complement codes -8..7: an ordinary symmetric int4
 *           grid, as GPTQ writes.
 *   scale   [N][K/group] bf16, one per row per 128 of K. w ~= code * scale.
 *
 * Decode: V_PERM_B32 permutes the bytes of an eight-byte source, so one table holds eight entries
 * and a two's complement nibble needs sixteen. Two perms index a positive and a negative table
 * with the low three bits, and a third selects on bit 3 by adding 4 to a selector byte (which
 * cannot carry): all sixteen levels for five more VALU per eight weights than a fifteen-level
 * sign-magnitude grid. See r4d_moe.hip w4_e4m3.
 *
 * The integers -8..7 are exact in E4M3 (three mantissa bits reach 16), so expanding a code to an
 * E4M3 byte is lossless and the weight runs on the existing f32_16x16x16_fp8_fp8 inner loop; only
 * the staging differs.
 *
 * Nibble order: byte b of a packed dword holds element b in its low nibble and element b+4 in its
 * high nibble, so the two masks yield elements 0..3 and 4..7 already in order, as in the dense
 * w4a8 kernel.
 *
 * Not shardable: a packed row is K/2 bytes and a column shard could cut a byte in half. Under
 * expert parallelism an expert is whole on one rank, the only configuration that declares this
 * format; rad_block_moe_fp8.h refuses a width-sharded declaration.
 */
/* `dtype=w4a8h` serves `i4*bf16[1x128]/fwht128`: the same bytes, but every group of K was
 * multiplied by a `group`-wide Hadamard before quantisation. Nothing in packing, extents or byte
 * count shows this, and a weight served the other way reads as noise, so the layout hook accepts
 * only the encoding the dtype names.
 *
 * The kernel does not branch on it: <Hw, Ha> = group * <w, a>, and the quantiser folds the
 * residual factor into the stored scale (docs/MOE-W4.md). What must agree is the activation, which
 * had_quant_act_fp8 quantises with the same rotation width. That pairing is a property of the
 * graph and is not checked here; the arch block that declares one declares the other. */
/* `dtype=w4nla8h` serves `u4*bf16[1x128]/fwht128` whose codes index a sixteen-entry `table` plane
 * (libquant `table=w4nl`), and the kernel decodes only its own table (r4d_args.h). The codes view
 * selects the table plane and the relayout compares it with the kernel's value for value, so a
 * container quantised against another table is refused at load. */
static int moe_w4_nl(const RadParam* p, int n_p) {
    const char* d = r4d_pgets(p, n_p, "dtype");
    return d && (std::strcmp(d, "w4nla8h") == 0 || std::strcmp(d, "w4nl64a8h") == 0 ||
                 std::strcmp(d, "w4nl32a8h") == 0 || std::strcmp(d, "w5nl64a8h") == 0 ||
                 std::strcmp(d, "w4nl64a8i8h") == 0 || std::strcmp(d, "w4nl32a8i8h") == 0);
}
/* The `a8i8` names are the same stored planes met by an int8 activation (moe_gemm_w4a8i8):
 * `w4a8i8` and `w4a8i8h` are w4a8's and w4a8h's, `w4nl64a8i8h` and `w4nl32a8i8h` the subgroup-scaled
 * codebook's. The hooks answer for them as for the names they stand beside. */
/* `w5nl64a8h` is w4nl64a8h at five bits: `u5*fp8_e4m3[1x64]*f32[*x*]/fwht128`, codes indexing
 * libquant's 32-entry `table=w5nl`. Its codes view relays each row into R4D_DT_MOEW5 (the low four
 * bits of every code as w4nla8h's nibble plane, then each code's high bit, the sign, one dword per
 * 32 elements in moew5_row's order) and checks the table against kW5nlTable. Its scale view is
 * w4nl64a8h's. */
static bool moe_w5(const RadParam* p, int n_p) {
    const char* d = r4d_pgets(p, n_p, "dtype");
    return d && std::strcmp(d, "w5nl64a8h") == 0;
}
/* `w4nl64a8h` is the codebook form with one E4M3 scale per (row, 64 of K) under a fixed f32
 * second level, kW4G64Scale, stored once per weight (`u4*fp8_e4m3[1x64]*f32[*x*]/fwht128` with the
 * table). Codes are stored as w4nla8h's. The scale view selects the E4M3 plane and the second
 * level; its relayout checks the second level against the kernel's constant and stores the E4M3
 * plane unchanged. `w4nl32a8h` is the same with one scale per (row, 32 of K).
 *
 * Returns the scale groups per 128 of K (the bf16 forms have one): 2, 4, or 0 for neither. */
static int moe_w4_g64(const RadParam* p, int n_p) {
    const char* d = r4d_pgets(p, n_p, "dtype");
    if (!d) return 0;
    return (std::strcmp(d, "w4nl64a8h") == 0 || std::strcmp(d, "w5nl64a8h") == 0 ||
            std::strcmp(d, "w4nl64a8i8h") == 0) ? 2
           : (std::strcmp(d, "w4nl32a8h") == 0 || std::strcmp(d, "w4nl32a8i8h") == 0) ? 4 : 0;
}
static int moe_w4_rot(const RadParam* p, int n_p) {
    const char* d = r4d_pgets(p, n_p, "dtype");
    return d && (std::strcmp(d, "w4a8h") == 0 || std::strcmp(d, "w4a8i8h") == 0 ||
                 moe_w4_nl(p, n_p));
}

/* ================================================================== moe_gemm_q's two classes
 *
 * `moe_gemm_q` takes its experts as two tables by parity: operands 2 and 3 hold the even experts
 * at [N, K], operands 6 and 7 the odd ones at [N_odd, K_odd] (see the schema in r4d_rows.cpp). A
 * hook asked about 6 or 7 answers as for 2 or 3 at the odd geometry, so both classes share one
 * format definition. */
namespace {
struct MoeView {
    RadParam p[32];
    int      n = 0;
    int      operand = 0;
};

int moe_view(const RadParam* p, int n_p, int operand, MoeView* v) {
    if (n_p < 0 || n_p > 32 || (n_p > 0 && !p)) return RAD_E_INVAL;
    v->n = n_p;
    v->operand = operand;
    for (int i = 0; i < n_p; ++i) v->p[i] = p[i];
    if (operand != 6 && operand != 7) return RAD_OK;
    v->operand = operand - 4;
    const long long n1 = r4d_pgeti_or(p, n_p, "N_odd", -1);
    const long long k1 = r4d_pgeti_or(p, n_p, "K_odd", -1);
    for (int i = 0; i < n_p; ++i) {
        if (!v->p[i].key) continue;
        if (n1 > 0 && std::strcmp(v->p[i].key, "N") == 0) v->p[i].ival = n1;
        if (k1 > 0 && std::strcmp(v->p[i].key, "K") == 0) v->p[i].ival = k1;
    }
    return RAD_OK;
}

/* `moe_gemm_q_mod4`'s four classes: operands 6/7, 8/9 and 10/11 are classes 1, 2 and 3 at
 * N_1/K_1, N_2/K_2 and N_3/K_3, answered as 2/3 at that geometry. */
int moe_view_mod4(const RadParam* p, int n_p, int operand, MoeView* v) {
    if (n_p < 0 || n_p > 32 || (n_p > 0 && !p)) return RAD_E_INVAL;
    v->n = n_p;
    v->operand = operand;
    for (int i = 0; i < n_p; ++i) v->p[i] = p[i];
    if (operand < 6 || operand > 11) return RAD_OK;
    static const char* const kN[4] = { "N", "N_1", "N_2", "N_3" };
    static const char* const kK[4] = { "K", "K_1", "K_2", "K_3" };
    const int q = (operand - 4) / 2;                 /* 6/7 -> 1, 8/9 -> 2, 10/11 -> 3 */
    v->operand = 2 + (operand & 1);
    const long long nq = r4d_pgeti_or(p, n_p, kN[q], -1);
    const long long kq = r4d_pgeti_or(p, n_p, kK[q], -1);
    for (int i = 0; i < n_p; ++i) {
        if (!v->p[i].key) continue;
        if (nq > 0 && std::strcmp(v->p[i].key, "N") == 0) v->p[i].ival = nq;
        if (kq > 0 && std::strcmp(v->p[i].key, "K") == 0) v->p[i].ival = kq;
    }
    return RAD_OK;
}
}  /* namespace */

/* ================================================================== the hooks
 *
 * Codes and scales arrive chosen by a quantiser; the hooks only arrange them as the kernel reads
 * them (fragment order, the (scale, zero) dword, a tile-major scale plane, a nibble order). Each
 * refuses an encoding its kernel does not read with RAD_E_DTYPE. */
namespace {

using r4d_enc::selects;
using r4d_enc::dims;

/* A selected plane's dense bytes. */
const uint8_t* pdata(const RadTensor& t) { return (const uint8_t*)t.data; }

/* The f16 scale of a canonical plane as f32; scale_zero_dword narrows it back exactly. */
float f16_at(const RadTensor& t, long long i) { return f16_to_f32(((const uint16_t*)t.data)[i]); }

/* Zero point `gi` of row `n` of a [N, nsz] u4 or u8 plane. Rows start on a byte (rad_encoding.h),
 * so a u4 row of an odd number of groups ends in a pad nibble and the next row starts after it. */
uint8_t* zero_row(const RadTensor& t, long long n, long long nsz) {
    return (uint8_t*)t.data + n * rad_enc_row_bytes(t.dtype, nsz);
}
int zero_at(const RadTensor& t, long long n, long long nsz, long long gi) {
    return (int)rad_load_f32(zero_row(t, n, nsz), t.dtype, gi);
}

/* The dense w4/w8/w2 families' scale operand: (scale) for a symmetric grid, (scale, zero) for an
 * asymmetric one, tile-major [N/16][(K/group)*16] in dwords or, for w8a8, halves. */
int sz_layout(const WGeom& g, const RadEncoding* enc, const int* sel, const RadTensor* pl, int n,
              bool asym, const char* tag, uint32_t dt, int bytes_each, RadLayout* out) {
    const long long nsz = g.K / g.group;
    if (asym ? !selects(enc, sel, n, "scale", "zero") : !selects(enc, sel, n, "scale"))
        return RAD_E_SHAPE;
    for (int i = 0; i < n; ++i)
        if (!dims(pl[i], g.N, nsz)) return RAD_E_SHAPE;
    fill(out, tag, dt, g.N / R4D_NTILE, nsz * R4D_NTILE, g.N * nsz * bytes_each);
    return RAD_OK;
}

/* Write the tile-major (scale, zero) dwords. `zfixed` >= 0 is a symmetric grid's constant zero. */
void sz_write(const WGeom& g, const RadTensor* pl, int zfixed, uint32_t* sz) {
    const long long nt = g.N / R4D_NTILE, nsz = g.K / g.group;
    for (long long t = 0; t < nt; ++t)
        for (long long r = 0; r < R4D_NTILE; ++r)
            for (long long gi = 0; gi < nsz; ++gi) {
                const long long n = t * R4D_NTILE + r;
                const int z = zfixed >= 0 ? zfixed : zero_at(pl[1], n, nsz, gi);
                const float sc = f16_at(pl[0], n * nsz + gi);
                sz[(t * nsz + gi) * R4D_NTILE + r] = scale_zero_dword(sc, z);
            }
}

/* Whether every zero point fits in `bits` bits. The int8 prefill kernel (r4d_gemm_wpa8i8_tiled.hip)
 * reads five bits of the zero out of the scale dword where the f16 kernel reads the whole value,
 * so a larger zero would make the two disagree. */
bool sz_zeros_fit(const WGeom& g, const RadTensor& zp, int bits) {
    const long long nsz = g.K / g.group;
    for (long long n = 0; n < g.N; ++n)
        for (long long gi = 0; gi < nsz; ++gi) {
            const int z = zero_at(zp, n, nsz, gi);
            if (z < 0 || z >= (1 << bits)) return false;
        }
    return true;
}

void sz_read(const WGeom& g, const uint32_t* sz, const RadTensor* pl, bool asym) {
    const long long nt = g.N / R4D_NTILE, nsz = g.K / g.group;
    for (long long t = 0; t < nt; ++t)
        for (long long r = 0; r < R4D_NTILE; ++r)
            for (long long gi = 0; gi < nsz; ++gi) {
                const long long n = t * R4D_NTILE + r;
                const uint32_t d = sz[(t * nsz + gi) * R4D_NTILE + r];
                ((uint16_t*)pl[0].data)[n * nsz + gi] = (uint16_t)(d & 0xFFFFu);
                if (asym) {
                    const int z = (int)(-f16_to_f32((uint16_t)(d >> 16)) - 1024.0f + 0.5f);
                    rad_store_code(zero_row(pl[1], n, nsz), pl[1].dtype, gi, z);
                }
            }
}

/* A rotated asymmetric grid: codes of dtype `codes`, an f16 scale and a zero point per group,
 * optionally with ParoQuant's `givens` transform and its tables, or libquant's `fwht128` (the
 * scale carrying its 1/128). The bytes are the unrotated grid's; the rotation is applied to the
 * activation. */
static bool paro_asym_enc(const RadEncoding* e, uint32_t codes, long long group) {
    if (!e || !rad_enc_is(e, "affine") || e->n_planes < 3) return false;
    const bool giv = std::strncmp(e->transform, "givens", RAD_ENC_STR) == 0;
    const bool fwht = std::strncmp(e->transform, "fwht128", RAD_ENC_STR) == 0;
    if (!giv && !fwht && e->transform[0]) return false;
    if (!giv && e->n_planes != 3) return false;
    if (giv && (!rad_enc_valid(e) || rad_enc_find(e, "t.pairs") < 0)) return false;
    if (std::strncmp(e->plane[0].role, "codes", RAD_ENC_STR) != 0 || e->plane[0].dtype != codes ||
        e->plane[0].block[0] != 1 || e->plane[0].block[1] != 1) return false;
    const int s = rad_enc_find(e, "scale"), z = rad_enc_find(e, "zero");
    if (s != 1 || z != 2) return false;
    return e->plane[s].dtype == RAD_F16 && e->plane[s].kind == RAD_PLANE_TILED &&
           e->plane[s].block[0] == 1 && e->plane[s].block[1] == group &&
           (e->plane[z].dtype == RAD_U8 || e->plane[z].dtype == codes) &&
           e->plane[z].kind == RAD_PLANE_TILED && e->plane[z].block[0] == 1 &&
           e->plane[z].block[1] == group;
}

/* ---- w4: a symmetric grid is i4 codes with an f16 scale; `_asym` is u4 codes with a zero. */
int w4_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, WGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    WGeom g;
    int rc = wgeom(p, n_p, R4D_W4_GROUP, &g);
    if (rc != RAD_OK) return rc;
    if (g.group != R4D_W4_GROUP) return RAD_E_DTYPE;
    if (g.K % R4D_W4_KPB) return RAD_E_SHAPE;
    const bool asym = w4_asym(p, n_p);
    const bool ok = asym ? (r4d_enc::affine(enc, RAD_U4, RAD_F16, 1, g.group, RAD_U8) ||
                            r4d_enc::affine(enc, RAD_U4, RAD_F16, 1, g.group, RAD_U4) ||
                            paro_asym_enc(enc, RAD_U4, g.group))
                         : r4d_enc::affine(enc, RAD_I4, RAD_F16, 1, g.group);
    if (!ok) return RAD_E_DTYPE;
    if (gout) *gout = g;
    if (operand == Q_BSCALE)
        return sz_layout(g, enc, sel, pl, n, asym, asym ? "r4d.w4u.sz.g128" : "r4d.w4.sz.g128",
                         R4D_DT_W4_SZ, 4, out);
    if (operand != Q_B) return RAD_E_UNSUPPORTED;
    if (!selects(enc, sel, n, "codes") || !dims(pl[0], g.N, g.K)) return RAD_E_SHAPE;
    fill(out, asym ? "r4d.w4u.frag.g128" : "r4d.w4.frag.g128", R4D_DT_W4_FRAG, g.N, g.K,
         g.N * g.K / 2);
    return RAD_OK;
}

/* Where element (n, k) of a w4 plane lands: the 32-bit word and the nibble within it. */
void w4_slot(const WGeom& g, long long n, long long k, long long* word, int* nib) {
    const long long kpb = g.K / R4D_W4_KPB;
    const long long t = n / R4D_NTILE, r = n % R4D_NTILE;
    const long long kb = k / R4D_W4_KPB, kin = k % R4D_W4_KPB;
    const long long s = kin / 16, kk = kin % 16;
    const long long lh = (kk % 8) / 4, e = (kk / 8) * 4 + (kk % 4);
    *nib = (int)((e < 4) ? (2 * e) : (2 * (e - 4) + 1));
    *word = ((t * kpb + kb) * R4D_WAVE + lh * 16 + r) * 4 + s;
}

}  /* namespace */

extern "C" int r4d_rad_layout_w4(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                     int n, RadLayout* out) {
    return w4_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int r4d_rad_relayout_w4(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                   const int* sel, const RadTensor* pl, int n, void* dst,
                                   int64_t dst_bytes) {
    RadLayout L;
    WGeom g;
    const int rc = w4_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    std::memset(dst, 0, (size_t)dst_bytes);
    if (operand == Q_BSCALE) {
        const bool asym = w4_asym(p, n_p);
        /* A rotated (ParoQuant) grid's zero must fit a 4-bit code. */
        if (asym && std::strncmp(enc->transform, "givens", RAD_ENC_STR) == 0 &&
            !sz_zeros_fit(g, pl[1], 4)) return RAD_E_FORMAT;
        sz_write(g, pl, asym ? -1 : 8, (uint32_t*)dst);
        return RAD_OK;
    }
    uint32_t* wq = (uint32_t*)dst;
    for (long long nn = 0; nn < g.N; ++nn) {
        const uint8_t* row = pdata(pl[0]) + nn * (g.K / 2);
        for (long long k = 0; k < g.K; ++k) {
            long long w;
            int nib;
            w4_slot(g, nn, k, &w, &nib);
            wq[w] |= (uint32_t)((row[k >> 1] >> (4 * (k & 1))) & 0xF) << (4 * nib);
        }
    }
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_w4(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const void* src,
                                     int64_t src_bytes, const RadTensor* pl, int n) {
    RadLayout L;
    WGeom g;
    const int rc = w4_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes) return RAD_E_SHAPE;
    if (operand == Q_BSCALE) {
        sz_read(g, (const uint32_t*)src, pl, w4_asym(p, n_p));
        return RAD_OK;
    }
    const uint32_t* wq = (const uint32_t*)src;
    uint8_t* d = (uint8_t*)pl[0].data;
    std::memset(d, 0, (size_t)(g.N * g.K / 2));
    for (long long nn = 0; nn < g.N; ++nn)
        for (long long k = 0; k < g.K; ++k) {
            long long w;
            int nib;
            w4_slot(g, nn, k, &w, &nib);
            const uint32_t q = (wq[w] >> (4 * nib)) & 0xF;
            d[nn * (g.K / 2) + (k >> 1)] |= (uint8_t)(q << (4 * (k & 1)));
        }
    return RAD_OK;
}

/* ---- w5: ParoQuant's asymmetric int5: u5 codes, an f16 scale and a u8 zero point (0..31) per
 * 128 of K. The blob is w4's fragment order for the low four bits of every code (offset binary, no
 * sign flip), then 32-bit words of fifth bits, one per (n-tile, 64 of K, lane): byte s is k step s
 * of the block and bit e is fragment element e, whose nibble is 2e (e < 4) or 2(e-4)+1 in dword s
 * of that lane's 16 bytes (r4d_gemm_w5a16_nt_m64.hip). The scale operand is w4u's (scale, zero)
 * dword plane. A rotated weight (a `givens` transform and its tables beside the first three
 * planes) has the same bytes; the rotation is the activation's. */
namespace {
bool w5_enc(const RadEncoding* e, long long group) { return paro_asym_enc(e, RAD_U5, group); }

int w5_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, WGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    WGeom g;
    int rc = wgeom(p, n_p, R4D_W4_GROUP, &g);
    if (rc != RAD_OK) return rc;
    if (g.group != R4D_W4_GROUP) return RAD_E_DTYPE;
    if (g.K % R4D_W4_KPB) return RAD_E_SHAPE;
    if (!w5_enc(enc, g.group)) return RAD_E_DTYPE;
    if (gout) *gout = g;
    if (operand == Q_BSCALE)
        return sz_layout(g, enc, sel, pl, n, true, "r4d.w5.sz.g128", R4D_DT_W4_SZ, 4, out);
    if (operand != Q_B) return RAD_E_UNSUPPORTED;
    if (!selects(enc, sel, n, "codes") || !dims(pl[0], g.N, g.K)) return RAD_E_SHAPE;
    fill(out, "r4d.w5.frag.g128", R4D_DT_W4_FRAG, g.N, g.K, g.N * g.K * 5 / 8);
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_w5(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                 const int* sel, const RadTensor* pl, int n, RadLayout* out) {
    return w5_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int r4d_rad_relayout_w5(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                   const int* sel, const RadTensor* pl, int n, void* dst,
                                   int64_t dst_bytes) {
    RadLayout L;
    WGeom g;
    const int rc = w5_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    std::memset(dst, 0, (size_t)dst_bytes);
    if (operand == Q_BSCALE) {
        if (!sz_zeros_fit(g, pl[1], 5)) return RAD_E_FORMAT;
        sz_write(g, pl, -1, (uint32_t*)dst);
        return RAD_OK;
    }
    uint32_t* wq = (uint32_t*)dst;
    uint32_t* hi = wq + g.N * g.K / 8;                    /* after the N*K/2 bytes of nibbles */
    const long long rb = (g.K * 5 + 7) / 8;
    for (long long nn = 0; nn < g.N; ++nn) {
        const uint8_t* row = pdata(pl[0]) + nn * rb;
        for (long long k = 0; k < g.K; ++k) {
            const uint32_t q = rad_load_bits(row, 5, k);
            long long w;
            int nib;
            w4_slot(g, nn, k, &w, &nib);
            wq[w] |= (q & 15u) << (4 * nib);
            /* the fifth bit: word (t, kb, lane) = w / 4 (w4_slot's dword index is that * 4 + s);
             * byte s = w % 4, bit e from the nibble position */
            const long long hw = w / 4;
            const int s = (int)(w % 4);
            const int e = nib % 2 == 0 ? nib / 2 : 4 + nib / 2;     /* nibble 2e / 2(e-4)+1 -> e */
            hi[hw] |= (q >> 4) << (8 * s + e);
        }
    }
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_w5(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                     const int* sel, const void* src, int64_t src_bytes,
                                     const RadTensor* pl, int n) {
    RadLayout L;
    WGeom g;
    const int rc = w5_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes) return RAD_E_SHAPE;
    if (operand == Q_BSCALE) {
        sz_read(g, (const uint32_t*)src, pl, true);
        return RAD_OK;
    }
    const uint32_t* wq = (const uint32_t*)src;
    const uint32_t* hi = wq + g.N * g.K / 8;
    const long long rb = (g.K * 5 + 7) / 8;
    uint8_t* d = (uint8_t*)pl[0].data;
    std::memset(d, 0, (size_t)(g.N * rb));
    for (long long nn = 0; nn < g.N; ++nn)
        for (long long k = 0; k < g.K; ++k) {
            long long w;
            int nib;
            w4_slot(g, nn, k, &w, &nib);
            const int e = nib % 2 == 0 ? nib / 2 : 4 + nib / 2;     /* as r4d_rad_relayout_w5 */
            const uint32_t q = ((wq[w] >> (4 * nib)) & 15u) | (((hi[w / 4] >> (8 * (w % 4) + e)) & 1u) << 4);
            rad_store_bits(d + nn * rb, 5, k, q);
        }
    return RAD_OK;
}

/* ---- givens pairs: ParoQuant's rotation pairs (i32 [P*S, K]), stored unchanged but checked. The
 * rotate kernels index a group's 128 LDS slots with them unchecked, so a stage that is not a
 * pairing within its own group (an index outside the group, or a column twice) is refused at load.
 * Operand 1 of the givens rows; the other operands are stored as they are. */
namespace {
constexpr int GV_PAIRS_OPERAND = 1;
constexpr long long GV_GROUP = 128;

}  /* namespace */

/* ---- givens_dense: ParoQuant's rotation as one dense matrix per (part, group).
 *
 * The dense kernel (r4d_givens_dense.hip) applies a part's Givens stages over each group of 128
 * columns as T (x / D), T being their composition. rad-convert composes T and stores it as the
 * "givens" encoding's "t.dense" table, f16 [P * K/128, 128 * 128]: in row p * (K/128) + g, element
 * j * 128 + i is output j for unit input e_i (abi/rad_encoding.h). The kernel stages a (part,
 * group), 32 KiB, whole in that order, so the layout is the table as stored. Operand 1 of
 * givens_dense_quant; everything else stays as stored. */
namespace {
constexpr int GD_OPERAND = 1;
constexpr long long GD_GROUP = 128;

int gd_geom(const RadParam* p, int n_p, const RadEncoding* enc, const int* sel, const RadTensor* pl, int n,
            long long* P, long long* K) {
    if (n != 1 || std::strncmp(r4d_enc::role(enc, sel, 0), "t.dense", RAD_ENC_STR) != 0) return RAD_E_SHAPE;
    *K = r4d_pgeti_or(p, n_p, "n", 0);
    *P = r4d_pgeti_or(p, n_p, "parts", 1);
    if (*K <= 0 || *K % GD_GROUP || *P < 1) return RAD_E_SHAPE;
    if (!dims(pl[0], *P * (*K / GD_GROUP), GD_GROUP * GD_GROUP) || pl[0].dtype != RAD_F16) return RAD_E_SHAPE;
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_gdense(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                     const int* sel, const RadTensor* pl, int n, RadLayout* out) {
    if (!out || !pl) return RAD_E_INVAL;
    if (operand != GD_OPERAND) return RAD_E_UNSUPPORTED;
    long long P, K;
    const int rc = gd_geom(p, n_p, enc, sel, pl, n, &P, &K);
    if (rc != RAD_OK) return rc;
    fill(out, "r4d.givens.dense.g128", RAD_F16, P * K, GD_GROUP, P * K * GD_GROUP * 2);
    return RAD_OK;
}

extern "C" int r4d_rad_relayout_gdense(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                       const int* sel, const RadTensor* pl, int n, void* dst,
                                       int64_t dst_bytes) {
    RadLayout L;
    int rc = r4d_rad_layout_gdense(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    std::memcpy(dst, pdata(pl[0]), (size_t)L.bytes);
    return RAD_OK;
}

/* ---- w8: i8 codes, symmetric about zero, an f16 scale per group. The f16 kernel (a16) reads the
 * code as offset binary with zero 128 in the shared dword; the int8 kernel (a8) reads it signed
 * with a bare f16 scale plane. */
namespace {
int w8_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, WGeom* gout, int a16) {
    if (!out || !pl) return RAD_E_INVAL;
    WGeom g;
    int rc = wgeom(p, n_p, R4D_W8_GROUP, &g);
    if (rc != RAD_OK) return rc;
    if (g.group != R4D_W8_GROUP) return RAD_E_DTYPE;
    if (g.K % R4D_W8_KPB) return RAD_E_SHAPE;
    if (!r4d_enc::affine(enc, RAD_I8, RAD_F16, 1, g.group)) return RAD_E_DTYPE;
    if (gout) *gout = g;
    if (operand == Q_BSCALE)
        return a16 ? sz_layout(g, enc, sel, pl, n, false, "r4d.w8a16.sz.g128", R4D_DT_W4_SZ, 4, out)
                   : sz_layout(g, enc, sel, pl, n, false, "r4d.w8a8.s.g128", R4D_DT_W8_S, 2, out);
    if (operand != Q_B) return RAD_E_UNSUPPORTED;
    if (!selects(enc, sel, n, "codes") || !dims(pl[0], g.N, g.K)) return RAD_E_SHAPE;
    fill(out, a16 ? "r4d.w8a16.frag.g128" : "r4d.w8a8.frag.g128", R4D_DT_W8_FRAG, g.N, g.K,
         g.N * g.K);
    return RAD_OK;
}

long long w8_slot(const WGeom& g, long long n, long long k) {
    const long long kpb = g.K / R4D_W8_KPB;
    const long long t = n / R4D_NTILE, r = n % R4D_NTILE;
    const long long kb = k / R4D_W8_KPB, kin = k % R4D_W8_KPB;
    const long long s = kin / 16, kk = kin % 16;
    const long long lh = (kk % 8) / 4, e = (kk / 8) * 4 + (kk % 4);
    return ((t * kpb + kb) * R4D_WAVE + lh * 16 + r) * 16 + s * 8 + e;
}

int w8_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                const RadTensor* pl, int n, void* dst, int64_t dst_bytes, int a16) {
    RadLayout L;
    WGeom g;
    const int rc = w8_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g, a16);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    std::memset(dst, 0, (size_t)dst_bytes);
    if (operand == Q_BSCALE) {
        if (a16) {
            sz_write(g, pl, 128, (uint32_t*)dst);
        } else {
            const long long nt = g.N / R4D_NTILE, nsz = g.K / g.group;
            uint16_t* s16 = (uint16_t*)dst;
            for (long long t = 0; t < nt; ++t)
                for (long long r = 0; r < R4D_NTILE; ++r)
                    for (long long gi = 0; gi < nsz; ++gi)
                        s16[(t * nsz + gi) * R4D_NTILE + r] =
                            ((const uint16_t*)pl[0].data)[(t * R4D_NTILE + r) * nsz + gi];
        }
        return RAD_OK;
    }
    uint8_t* wq = (uint8_t*)dst;
    for (long long nn = 0; nn < g.N; ++nn)
        for (long long k = 0; k < g.K; ++k) {
            const int8_t q = ((const int8_t*)pl[0].data)[nn * g.K + k];
            wq[w8_slot(g, nn, k)] = a16 ? (uint8_t)(q + 128) : (uint8_t)q;
        }
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_w8a16(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* pl, int n, RadLayout* out) {
    return w8_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr, 1);
}
extern "C" int r4d_rad_layout_w8a8(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, RadLayout* out) {
    return w8_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr, 0);
}
extern "C" int r4d_rad_relayout_w8a16(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                      int n, void* dst, int64_t dst_bytes) {
    return w8_relayout(p, n_p, operand, enc, sel, pl, n, dst, dst_bytes, 1);
}
extern "C" int r4d_rad_relayout_w8a8(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                     int n, void* dst, int64_t dst_bytes) {
    return w8_relayout(p, n_p, operand, enc, sel, pl, n, dst, dst_bytes, 0);
}

/* ---- w2a8: u2 codes, asymmetric, an f16 scale and an integer zero per group, both streams
 * tile-major. */
namespace {
int w2_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, WGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    WGeom g;
    int rc = wgeom(p, n_p, R4D_W2_GROUP, &g);
    if (rc != RAD_OK) return rc;
    if (g.group != R4D_W2_GROUP) return RAD_E_DTYPE;
    if (g.K % 128) return RAD_E_SHAPE;
    if (!(r4d_enc::affine(enc, RAD_U2, RAD_F16, 1, g.group, RAD_U8) ||
          r4d_enc::affine(enc, RAD_U2, RAD_F16, 1, g.group, RAD_U4))) return RAD_E_DTYPE;
    if (gout) *gout = g;
    if (operand == Q_BSCALE)
        return sz_layout(g, enc, sel, pl, n, true, "r4d.w2a8.sz.g128", R4D_DT_W4_SZ, 4, out);
    if (operand != Q_B) return RAD_E_UNSUPPORTED;
    if (!selects(enc, sel, n, "codes") || !dims(pl[0], g.N, g.K)) return RAD_E_SHAPE;
    fill(out, "r4d.w2a8.frag.g128", R4D_DT_W2_FRAG, g.N, g.K, g.N * g.K / 4);
    return RAD_OK;
}

void w2_slot(const WGeom& g, long long n, long long k, long long* word, int* field) {
    const long long kpb = g.K / 128;
    const long long t = n / R4D_NTILE, r = n % R4D_NTILE;
    const long long kb = k / 128, kin = k % 128;
    const long long step = kin / 16, d = step / 2, h = step % 2;
    const long long kk = kin % 16, lh = (kk % 8) / 4, e = (kk / 8) * 4 + (kk % 4);
    *field = (int)(4 * (e % 4) + (e / 4) + 2 * h);
    *word = ((t * kpb + kb) * R4D_WAVE + lh * 16 + r) * 4 + d;
}
}  /* namespace */

extern "C" int r4d_rad_layout_w2a8(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, RadLayout* out) {
    return w2_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int r4d_rad_relayout_w2a8(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                     int n, void* dst, int64_t dst_bytes) {
    RadLayout L;
    WGeom g;
    const int rc = w2_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    if (operand == Q_BSCALE) {
        std::memset(dst, 0, (size_t)dst_bytes);
        sz_write(g, pl, -1, (uint32_t*)dst);
        return RAD_OK;
    }
    /* A tile at a time (w2_slot's order, r4d_hsimd::w2a8_tile), every stored byte written: a
     * draft head of 2-bit codes is a 300 MiB weight that a code at a time took 2.7 s, the last
     * piece of a load or restore to finish. */
    const long long kpb = g.K / 128, rb = g.K / 4, tile = R4D_NTILE * rb;
    const uint8_t* codes = pdata(pl[0]);
    uint8_t* out = (uint8_t*)dst;
    r4d_par::par_for(g.N / R4D_NTILE, std::max(1ll, (16ll << 20) / tile), [&](long long lo, long long hi) {
        for (long long t = lo; t < hi; ++t)
            r4d_hsimd::w2a8_tile(codes + t * tile, rb, kpb, out + t * tile);
    });
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_w2a8(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel, const void* src,
                                       int64_t src_bytes, const RadTensor* pl, int n) {
    RadLayout L;
    WGeom g;
    const int rc = w2_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes) return RAD_E_SHAPE;
    if (operand == Q_BSCALE) {
        sz_read(g, (const uint32_t*)src, pl, true);
        return RAD_OK;
    }
    const uint32_t* wq = (const uint32_t*)src;
    uint8_t* d = (uint8_t*)pl[0].data;
    std::memset(d, 0, (size_t)(g.N * g.K / 4));
    for (long long nn = 0; nn < g.N; ++nn)
        for (long long k = 0; k < g.K; ++k) {
            long long w;
            int f;
            w2_slot(g, nn, k, &w, &f);
            d[nn * (g.K / 4) + (k >> 2)] |= (uint8_t)(((wq[w] >> (2 * f)) & 3) << (2 * (k & 3)));
        }
    return RAD_OK;
}

/* ---- MXFP4: fp4 e2m1 codes with an E8M0 exponent per 32. The codes go to fragment order, the
 * exponents to [K/32][N], and b_ref is each row's largest exponent: a reduction any reader of the
 * plane could compute, so not a choice. */
namespace {
/* A rotated MXFP4 weight (ParoQuant): the same two planes stored in the basis the "givens"
 * transform names, with the rotation's tables beside them. The GEMM reads codes and exponents as
 * for an unrotated weight; the rotation is applied to the activation (givens_dense_quant), whose
 * tables are separate weight operands. This only checks that the tables are present and are the
 * rotation the quantiser applies; any other transform would decode to other numbers. */
bool mx_paro_enc(const RadEncoding* e, long long K) {
    if (!e || !rad_enc_is(e, "affine") || std::strncmp(e->transform, "givens", RAD_ENC_STR) != 0) return false;
    if (e->n_planes < 5 || !rad_enc_valid(e)) return false;
    if (std::strncmp(e->plane[0].role, "codes", RAD_ENC_STR) != 0 || e->plane[0].dtype != RAD_FP4E2M1 ||
        e->plane[0].block[0] != 1 || e->plane[0].block[1] != 1) return false;
    const int s = rad_enc_find(e, "scale");
    if (s != 1 || e->plane[s].dtype != RAD_E8M0 || e->plane[s].kind != RAD_PLANE_TILED ||
        e->plane[s].block[0] != 1 || e->plane[s].block[1] != R4D_MXFP4_GROUP) return false;
    const int p = rad_enc_find(e, "t.pairs"), an = rad_enc_find(e, "t.angle"), sc = rad_enc_find(e, "t.scale");
    if (p < 0 || an < 0 || sc < 0) return false;
    if (e->plane[p].dtype != RAD_I32 || e->plane[an].dtype != RAD_F32 || e->plane[sc].dtype != RAD_F32)
        return false;
    /* The quantiser rotates whole groups of 128 columns, one scale per group. */
    return K % 128 == 0 && e->plane[p].extent[1] == K;
}

int mx_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, WGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    WGeom g;
    int rc = wgeom(p, n_p, R4D_MXFP4_GROUP, &g);
    if (rc != RAD_OK) return rc;
    if (g.group != R4D_MXFP4_GROUP || g.K % 16) return RAD_E_SHAPE;
    if (!r4d_enc::affine(enc, RAD_FP4E2M1, RAD_E8M0, 1, R4D_MXFP4_GROUP) &&
        !mx_paro_enc(enc, g.K)) return RAD_E_DTYPE;
    if (gout) *gout = g;
    const long long nblk = g.K / R4D_MXFP4_GROUP;
    if (operand == Q_BSCALE || operand == Q_BREF) {
        if (!selects(enc, sel, n, "scale") || !dims(pl[0], g.N, nblk)) return RAD_E_SHAPE;
        if (operand == Q_BSCALE) fill(out, "r4d.mxfp4.e8m0", R4D_DT_E8M0, nblk, g.N, nblk * g.N);
        else                     fill(out, "r4d.mxfp4.wref", R4D_DT_E8M0, g.N, 1, g.N);
        return RAD_OK;
    }
    if (operand != Q_B) return RAD_E_UNSUPPORTED;
    if (!selects(enc, sel, n, "codes") || !dims(pl[0], g.N, g.K)) return RAD_E_SHAPE;
    fill(out, "r4d.mxfp4.frag.g32", R4D_DT_MXFP4_FRAG, g.N, g.K, g.N * g.K / 2);
    return RAD_OK;
}

}  /* namespace */

extern "C" int r4d_rad_layout_mxfp4(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* pl, int n, RadLayout* out) {
    return mx_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int r4d_rad_relayout_mxfp4(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                      int n, void* dst, int64_t dst_bytes) {
    RadLayout L;
    WGeom g;
    const int rc = mx_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    const long long nblk = g.K / R4D_MXFP4_GROUP;
    uint8_t* d = (uint8_t*)dst;
    if (operand == Q_BSCALE || operand == Q_BREF) {
        const uint8_t* e = pdata(pl[0]);
        if (operand == Q_BSCALE) {
            r4d_hsimd::byte_transpose(e, g.N, nblk, d);
            return RAD_OK;
        }
        for (long long nn = 0; nn < g.N; ++nn) {
            uint8_t mx = 0;
            for (long long b = 0; b < nblk; ++b) mx = std::max(mx, e[nn * nblk + b]);
            d[nn] = mx;
        }
        return RAD_OK;
    }
    /* mx_slot moves each 16-code group's halves, four bytes each, whole: half lh of group ksi of
     * row r is dword 2 ksi + lh of the row, and goes to dword (2 ksi + lh) * 16 + r of the n-tile.
     * So a tile is a transpose of its 16 rows of K / 8 dwords, every stored byte written. */
    const long long words = g.K / 8, tile = R4D_NTILE * words;
    const uint32_t* codes = (const uint32_t*)pdata(pl[0]);
    uint32_t* out = (uint32_t*)dst;
    r4d_par::par_for(g.N / R4D_NTILE, std::max(1ll, (4ll << 20) / tile), [&](long long lo, long long hi) {
        for (long long t = lo; t < hi; ++t)
            r4d_hsimd::dword_transpose16(codes + t * tile, words, words, out + t * tile);
    });
    return RAD_OK;
}

/* ---- M2XFP4: the M2 analogue of MXFP4 above, the paper's weight-side Sg-EM (arXiv 2601.19213
 * §4.4.2). The quantiser's own scheme "m2xfp4", three TILED planes over the same 32-column groups:
 *
 *   codes  fp4_e2m1  [1 x 1]   element 2j in bits 0-3 of byte j -- radiance MXFP4's codes
 *   scale  e8m0      [1 x 32]  per group, the adaptive bias already folded into the stored byte
 *   meta   u8        [1 x 32]  one byte a group: subgroup s = columns [8s, 8s+8), bits [2s, 2s+2)
 *
 * The exact element is e2m1mag[c&7] . sign(c) . 2^(scale-127) . (1 + k_s/4). What the kernels read
 * is the mxfp4 arrangement with the meta plane riding the stored weight's tail, in the scale
 * plane's [K/32][N] order: codes in fragment order, then K/32 rows of N meta bytes, so both GEMM
 * rows read it with the exponents' own geometry (r4d_gemm_m2xfp4a8_*.hip). b_ref stays each row's
 * largest exponent byte, the same derivation a reader of the plane could compute -- and it is what
 * bounds the fold: d >= 0 for every block, so the largest e4m3 a table cell holds is the x1.75
 * magnitude 10.5 scaled by 2^-d, a normal at every d >= 0.
 *
 * Anything else is refused: this kernel library decodes one scheme, and a weight whose meta plane
 * said something else would decode to other numbers.
 *
 * A rotated weight (ParoQuant's M2XFP4) carries the same three planes in the basis its "givens"
 * transform names, with the rotation's tables beside them -- the rotated MXFP4 weight's shape plus
 * the meta plane. The GEMM reads the planes as for an unrotated weight; the rotation is applied to
 * the activation (givens_dense_quant), whose tables are separate weight operands. */
namespace {

}  /* namespace */

/* ---- the MoE experts. I4 at BF16 scales per 128 K uses 16-column/64-K fragment tiles,
 * coalescing a wave's register loads without another copy of the weight. The staged prefill
 * kernels read the same tiles back in row segments. Subgroup-scaled codebooks keep their
 * row-major nibble permutation, and FP8 is unchanged. `w4a8h` has the same packing with the weight Hadamard-rotated per group,
 * accepted only from an encoding that says so. */
namespace {
int moew4_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                     const int* sel, const RadTensor* pl, int n, RadLayout* out, WGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    WGeom g;
    const int rc = wgeom(p, n_p, R4D_W4_GROUP, &g);
    if (rc != RAD_OK) return rc;
    if (g.K % 8) return RAD_E_SHAPE;
    char fw[RAD_ENC_STR] = "";
    if (moe_w4_rot(p, n_p)) std::snprintf(fw, sizeof fw, "fwht%lld", g.group);
    const bool nl = moe_w4_nl(p, n_p);
    const bool w5 = moe_w5(p, n_p);
    const int  g64 = moe_w4_g64(p, n_p);             /* scale groups a 128 of K, or 0 */
    if (w5 && g.K % 32) return RAD_E_SHAPE;           /* a sign word is 32 elements */
    if (g64 ? !r4d_enc::affine_table2(enc, w5 ? RAD_U5 : RAD_U4, RAD_F8E4M3, 1, g.group / g64,
                                      RAD_F32, RAD_F32, w5 ? 32 : 16, fw)
       : nl ? !r4d_enc::affine_table(enc, RAD_U4, RAD_BF16, 1, g.group, RAD_F32, 16, fw)
            : !r4d_enc::affine(enc, RAD_I4, RAD_BF16, 1, g.group, RAD_DT_INVALID, fw))
        return RAD_E_DTYPE;
    if (gout) *gout = g;
    if (operand == Q_BSCALE) {
        if (g64) {
            if (!selects(enc, sel, n, "scale", "scale.1") ||
                !dims(pl[0], g.N, g64 * g.K / g.group) || !dims(pl[1], 1, 1))
                return RAD_E_SHAPE;
            fill(out, g64 == 4 ? "r4d.moew4nh32.s" : "r4d.moew4nh64.s", RAD_F8E4M3, g.N,
                 g64 * g.K / g.group, g.N * g64 * g.K / g.group);
            return RAD_OK;
        }
        if (!selects(enc, sel, n, "scale") || !dims(pl[0], g.N, g.K / g.group)) return RAD_E_SHAPE;
        return RAD_E_UNSUPPORTED;                    /* bf16 a row a group, as it is */
    }
    if (operand != Q_B) return RAD_E_UNSUPPORTED;
    if (nl ? !selects(enc, sel, n, "codes", "table") || !dims(pl[1], 1, w5 ? 32 : 16)
           : !selects(enc, sel, n, "codes"))
        return RAD_E_SHAPE;
    if (!dims(pl[0], g.N, g.K)) return RAD_E_SHAPE;
    if (!g64 && (g.N % 16 || g.K % 64)) return RAD_E_SHAPE;
    if (w5) {
        fill(out, "r4d.moew5nh64.w", R4D_DT_MOEW5, g.N, g.K, g.N * (g.K / 8) * 5);
        return RAD_OK;
    }
    fill(out, g64 == 4 ? "r4d.moew4nh32.w" : g64 ? "r4d.moew4nh64.w"
                    : nl ? "r4d.moew4nh.w.t16k64.g128"
                    : moe_w4_rot(p, n_p) ? "r4d.moew4h.w.t16k64.g128" : "r4d.moew4.w.t16k64.g128",
         R4D_DT_MOEW4, g.N, g.K, g.N * g.K / 2);
    return RAD_OK;
}

/* A w4nla8h weight's table plane against the one the kernel decodes, value for value. A mismatch
 * is RAD_E_DTYPE, which the loader reports with the weight's name, its encoding and the tag. */
bool moew4_table_is_kernels(const RadTensor& t) {
    const float* v = (const float*)t.data;
    for (int i = 0; i < 16; ++i)
        if (v[i] != kW4nlTable[i]) return false;
    return true;
}
/* And w5nl64a8h's against kW5nlTable. */
bool moew5_table_is_kernels(const RadTensor& t) {
    const float* v = (const float*)t.data;
    for (int i = 0; i < 32; ++i)
        if (v[i] != kW5nlTable[i]) return false;
    return true;
}

/* The expansion's nibble order, applied to a 64-bit word (two dwords). Canonical element 8g+i is
 * nibble i of dword g; stored, byte b holds element b low and b+4 high (nibbles n0 n4 n1 n5 n2 n6
 * n3 n7, a perfect shuffle of the dword's halves). Swapping the middle bytes, then each half's
 * middle nibbles, produces it; each step swaps two fields, so the steps in reverse order undo it.
 * The load's forward pass is r4d_hsimd's vector form of the same steps; this one serves the
 * inverse of the fragment tiles. */
inline uint64_t moew4_bytes(uint64_t x) {
    return (x & 0xFF0000FFFF0000FFull) | ((x & 0x0000FF000000FF00ull) << 8) |
           ((x >> 8) & 0x0000FF000000FF00ull);
}
inline uint64_t moew4_nibbles(uint64_t x) {
    return (x & 0xF00FF00FF00FF00Full) | ((x & 0x00F000F000F000F0ull) << 4) |
           ((x >> 4) & 0x00F000F000F000F0ull);
}

/* Permutes `bytes` of codes (a multiple of four) from `s` to `d`, forward or back: vectorised
 * by CPU level (r4d_hsimd.h), since every routed expert passes through here at load. */
inline void moew4_permute(const uint8_t* s, uint8_t* d, int64_t bytes, bool forward) {
    r4d_hsimd::moew4_permute(s, d, bytes, forward);
}
/* Signed-I4 fragment tiles: [N/16][K/64][lane=32][step=4] dwords. A lane owns column
 * lane%16 and K bytes step*16 + (lane/16)*8 .. +7, in the nibble order above. Two adjacent
 * canonical dwords supply the low/high lane halves; the inverse restores the exact source codes.
 * No precision or byte-count change, and no second resident weight plane. */
void moew4_frag_permute(const uint8_t* src, uint8_t* dst, int64_t N, int64_t K, bool forward) {
    if (forward) {
        /* A row at a time: its codes permuted in one vector pass, then its dwords dealt to the
         * lanes, low halves at `packed` and high ones 256 bytes on. */
        std::vector<uint8_t> row((size_t)(K / 2));
        for (int64_t r = 0; r < N; ++r) {
            moew4_permute(src + (size_t)(r * (K / 2)), row.data(), K / 2, true);
            const int64_t nt = r / 16, n = r % 16;
            for (int64_t kb = 0; kb < K / 64; ++kb)
                for (int64_t s = 0; s < 4; ++s) {
                    const size_t packed = (size_t)(((nt * (K / 64) + kb) * 32 + n) * 16 + s * 4);
                    const uint8_t* x = row.data() + kb * 32 + s * 8;
                    std::memcpy(dst + packed, x, 4);
                    std::memcpy(dst + packed + 256, x + 4, 4);
                }
        }
        return;
    }
    /* Back (the oracle's inverse): a dword pair at a time. */
    for (int64_t nt = 0; nt < N / 16; ++nt)
        for (int64_t kb = 0; kb < K / 64; ++kb)
            for (int64_t n = 0; n < 16; ++n)
                for (int64_t s = 0; s < 4; ++s) {
                    const size_t canonical = (size_t)((nt * 16 + n) * (K / 2) + kb * 32 + s * 8);
                    const size_t packed = (size_t)(((nt * (K / 64) + kb) * 32 + n) * 16 + s * 4);
                    uint32_t lo, hi;
                    std::memcpy(&lo, src + packed, sizeof lo);
                    std::memcpy(&hi, src + packed + 256, sizeof hi);
                    const uint64_t x = moew4_bytes(moew4_nibbles((uint64_t)lo | ((uint64_t)hi << 32)));
                    std::memcpy(dst + canonical, &x, sizeof x);
                }
}

/* One w5nl64a8h row between container and kernel. Stored, a row is K five-bit codes, LSB first
 * (rad_plugin.h packing), so every eight codes are five bytes. Served, it is the codes' low nibbles
 * in canonical 4-bit order (two per byte, element 2i low) put through moew4_permute, then their
 * high bits, one dword per 32 elements: K/2 + K/8 bytes, the same count. w4nla8h does not use this
 * row: per-128 BF16-scale I4, including that codebook, uses the 16-column/64-K fragment tiles.
 *
 * The sign dword is transposed for the kernel's merge: element 4m+b of a 32 (byte b of the
 * expansion's output dword m) is bit 7-m of byte b, so `sg << m` aligns dword m's four signs with
 * E4M3's sign bits at once (r4d_moe.hip, w5_expand). `nib` is a scratch row of K/2 bytes. */
inline uint32_t moew5_sign_bit(int64_t j) {        /* j: the element's index within its 32 */
    return 8u * (uint32_t)(j & 3) + 7u - (uint32_t)(j >> 2);
}
void moew5_row(const uint8_t* s, uint8_t* d, int64_t K, uint8_t* nib, bool forward) {
    if (forward) {
        uint8_t* sg = d + K / 2;
        for (int64_t c0 = 0; c0 < K; c0 += 32) {
            uint32_t w = 0;
            for (int64_t k = c0; k < c0 + 32; k += 8) {
                uint64_t x = 0;
                std::memcpy(&x, s + k / 8 * 5, 5);
                uint32_t lo = 0;
                for (int i = 0; i < 8; ++i) {
                    const uint32_t c = (uint32_t)(x >> (5 * i)) & 31u;
                    lo |= (c & 15u) << (4 * i);
                    w |= (c >> 4) << moew5_sign_bit(k - c0 + i);
                }
                std::memcpy(nib + k / 2, &lo, 4);
            }
            std::memcpy(sg + c0 / 8, &w, 4);
        }
        moew4_permute(nib, d, K / 2, true);
        return;
    }
    moew4_permute(s, nib, K / 2, false);
    const uint8_t* ss = s + K / 2;
    for (int64_t c0 = 0; c0 < K; c0 += 32) {
        uint32_t w = 0;
        std::memcpy(&w, ss + c0 / 8, 4);
        for (int64_t k = c0; k < c0 + 32; k += 8) {
            uint32_t lo = 0;
            std::memcpy(&lo, nib + k / 2, 4);
            uint64_t x = 0;
            for (int i = 0; i < 8; ++i) {
                const uint64_t c = ((lo >> (4 * i)) & 15u) |
                                   (uint64_t)(((w >> moew5_sign_bit(k - c0 + i)) & 1u) << 4);
                x |= c << (5 * i);
            }
            std::memcpy(d + k / 8 * 5, &x, 5);
        }
    }
}
}  /* namespace */

namespace {
using MoeViewFn = int (*)(const RadParam*, int, int, MoeView*);

int moew4_layout_by(MoeViewFn view, const RadParam* p, int n_p, int operand,
                    const RadEncoding* enc, const int* sel, const RadTensor* pl, int n,
                    RadLayout* out) {
    MoeView v;
    const int rc = view(p, n_p, operand, &v);
    return rc != RAD_OK ? rc : moew4_enc_layout(v.p, v.n, v.operand, enc, sel, pl, n, out, nullptr);
}

int moew4_relayout_by(MoeViewFn view, const RadParam* p, int n_p, int operand,
                      const RadEncoding* enc, const int* sel, const RadTensor* pl, int n,
                      void* dst, int64_t dst_bytes) {
    MoeView v;
    RadLayout L;
    WGeom g;
    int rc = view(p, n_p, operand, &v);
    if (rc == RAD_OK) rc = moew4_enc_layout(v.p, v.n, v.operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    if (v.operand == Q_BSCALE) {                     /* g64 forms: the second level is fixed */
        float s2 = 0.0f;
        std::memcpy(&s2, pdata(pl[1]), sizeof s2);
        if (s2 != kW4G64Scale) return RAD_E_DTYPE;
        std::memcpy(dst, pdata(pl[0]), (size_t)L.bytes);
        return RAD_OK;
    }
    if (moe_w5(v.p, v.n)) {
        if (!moew5_table_is_kernels(pl[1])) return RAD_E_DTYPE;
        const int64_t rb = g.K / 8 * 5;
        std::vector<uint8_t> nib((size_t)(g.K / 2));
        for (long long nn = 0; nn < g.N; ++nn)
            moew5_row(pdata(pl[0]) + nn * rb, (uint8_t*)dst + nn * rb, g.K, nib.data(), true);
        return RAD_OK;
    }
    if (moe_w4_nl(v.p, v.n) && !moew4_table_is_kernels(pl[1])) return RAD_E_DTYPE;
    if (moe_w4_g64(v.p, v.n)) moew4_permute(pdata(pl[0]), (uint8_t*)dst, g.N * (g.K / 2), true);
    else moew4_frag_permute(pdata(pl[0]), (uint8_t*)dst, g.N, g.K, true);
    return RAD_OK;
}

int moew4_unrelayout_by(MoeViewFn view, const RadParam* p, int n_p, int operand,
                        const RadEncoding* enc, const int* sel, const void* src,
                        int64_t src_bytes, const RadTensor* pl, int n) {
    MoeView v;
    RadLayout L;
    WGeom g;
    int rc = view(p, n_p, operand, &v);
    if (rc == RAD_OK) rc = moew4_enc_layout(v.p, v.n, v.operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes) return RAD_E_SHAPE;
    if (v.operand == Q_BSCALE) {
        std::memcpy(pl[0].data, src, (size_t)L.bytes);
        std::memcpy(pl[1].data, &kW4G64Scale, sizeof kW4G64Scale);
        return RAD_OK;
    }
    if (moe_w5(v.p, v.n)) {
        const int64_t rb = g.K / 8 * 5;
        std::vector<uint8_t> nib((size_t)(g.K / 2));
        for (long long nn = 0; nn < g.N; ++nn)
            moew5_row((const uint8_t*)src + nn * rb, (uint8_t*)pl[0].data + nn * rb, g.K,
                      nib.data(), false);
        std::memcpy(pl[1].data, kW5nlTable, sizeof kW5nlTable);
        return RAD_OK;
    }
    if (moe_w4_g64(v.p, v.n))
        moew4_permute((const uint8_t*)src, (uint8_t*)pl[0].data, g.N * (g.K / 2), false);
    else moew4_frag_permute((const uint8_t*)src, (uint8_t*)pl[0].data, g.N, g.K, false);
    /* The table the stored codes index is the kernel's: the relayout refused any other. */
    if (moe_w4_nl(v.p, v.n)) std::memcpy(pl[1].data, kW4nlTable, sizeof kW4nlTable);
    return RAD_OK;
}

}  /* namespace */

extern "C" int r4d_rad_layout_moe_w4(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel,
                                     const RadTensor* pl, int n, RadLayout* out) {
    return moew4_layout_by(moe_view, p, n_p, operand, enc, sel, pl, n, out);
}
extern "C" int r4d_rad_relayout_moe_w4(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, void* dst, int64_t dst_bytes) {
    return moew4_relayout_by(moe_view, p, n_p, operand, enc, sel, pl, n, dst, dst_bytes);
}
extern "C" int r4d_rad_unrelayout_moe_w4(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel, const void* src,
                                         int64_t src_bytes, const RadTensor* pl, int n) {
    return moew4_unrelayout_by(moe_view, p, n_p, operand, enc, sel, src, src_bytes, pl, n);
}
extern "C" int r4d_rad_layout_moe_w4_mod4(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, RadLayout* out) {
    return moew4_layout_by(moe_view_mod4, p, n_p, operand, enc, sel, pl, n, out);
}
extern "C" int r4d_rad_relayout_moe_w4_mod4(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const RadTensor* pl, int n, void* dst,
                                            int64_t dst_bytes) {
    return moew4_relayout_by(moe_view_mod4, p, n_p, operand, enc, sel, pl, n, dst, dst_bytes);
}
extern "C" int r4d_rad_unrelayout_moe_w4_mod4(const RadParam* p, int n_p, int operand,
                                              const RadEncoding* enc, const int* sel,
                                              const void* src, int64_t src_bytes,
                                              const RadTensor* pl, int n) {
    return moew4_unrelayout_by(moe_view_mod4, p, n_p, operand, enc, sel, src, src_bytes, pl, n);
}

/* moe_gemm_i8a8's experts: each class's int8 codes and row scales as stored. */
extern "C" int r4d_rad_layout_moe_i8(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel,
                                     const RadTensor* pl, int n, RadLayout* out) {
    MoeView v;
    const int rc = moe_view(p, n_p, operand, &v);
    return rc != RAD_OK ? rc : r4d_rad_layout_i8_rowmajor(v.p, v.n, v.operand, enc, sel, pl, n,
                                                          out);
}
extern "C" int r4d_rad_layout_moe_i8_mod4(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, RadLayout* out) {
    MoeView v;
    const int rc = moe_view_mod4(p, n_p, operand, &v);
    return rc != RAD_OK ? rc : r4d_rad_layout_i8_rowmajor(v.p, v.n, v.operand, enc, sel, pl, n,
                                                          out);
}

/* ---- conv2d_nhwc: PyTorch's [cout][cin][k][k] stored tap-major, [cout][k][k][cin].
 *
 * The implicit-GEMM convolution (r4d_gen.hip) contracts over K = (tap, channel) with the channel
 * innermost, so eight consecutive K are eight channels of one input pixel, one 16-byte read of an
 * NHWC row. The weight's K must run the same way; PyTorch's is channel-major. Operand 1 of
 * conv2d_nhwc; the bias is read as stored. */
namespace {
constexpr int CONV_W_OPERAND = 1;

int conv_geom(const RadParam* p, int n_p, const RadTensor* pl, int n, long long* cout,
              long long* cin, long long* k) {
    *cout = r4d_pgeti_or(p, n_p, "cout", 0);
    *cin = r4d_pgeti_or(p, n_p, "cin", 0);
    *k = r4d_pgeti_or(p, n_p, "k", 0);
    if (n != 1 || !pl || *cout <= 0 || *cin <= 0 || *k <= 0) return RAD_E_SHAPE;
    if (pl[0].dtype != RAD_BF16 || pl[0].rank != 2 ||
        pl[0].shape[0] * pl[0].shape[1] != *cout * *cin * *k * *k)
        return RAD_E_SHAPE;
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_conv_tap(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                       int n, RadLayout* out) {
    (void)sel;
    if (!out) return RAD_E_INVAL;
    if (operand != CONV_W_OPERAND) return RAD_E_UNSUPPORTED;
    if (!enc || !rad_enc_is(enc, "plain")) return RAD_E_DTYPE;
    long long co, ci, k;
    const int rc = conv_geom(p, n_p, pl, n, &co, &ci, &k);
    if (rc != RAD_OK) return rc;
    fill(out, "r4d.conv.tap_major", RAD_BF16, co, k * k * ci, co * k * k * ci * 2);
    return RAD_OK;
}

extern "C" int r4d_rad_relayout_conv_tap(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* pl, int n, void* dst,
                                         int64_t dst_bytes) {
    RadLayout L;
    const int rc = r4d_rad_layout_conv_tap(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    long long co, ci, k;
    conv_geom(p, n_p, pl, n, &co, &ci, &k);
    const uint16_t* s = (const uint16_t*)pl[0].data;
    uint16_t* d = (uint16_t*)dst;
    const long long kk = k * k;
    /* Written in order, a tap's channels at a time: an output channel's [cin][k * k] block is a
     * few KiB the reads stay inside, where storing a channel's taps cin apart missed every time. */
    for (long long o = 0; o < co; ++o) {
        const uint16_t* so = s + o * ci * kk;
        uint16_t* dt = d + o * kk * ci;
        for (long long t = 0; t < kk; ++t, dt += ci)
            for (long long c = 0; c < ci; ++c) dt[c] = so[c * kk + t];
    }
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_conv_tap(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const void* src, int64_t src_bytes,
                                           const RadTensor* pl, int n) {
    RadLayout L;
    const int rc = r4d_rad_layout_conv_tap(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    long long co, ci, k;
    conv_geom(p, n_p, pl, n, &co, &ci, &k);
    const uint16_t* s = (const uint16_t*)src;
    uint16_t* d = (uint16_t*)pl[0].data;
    const long long kk = k * k;
    for (long long o = 0; o < co; ++o)
        for (long long c = 0; c < ci; ++c)
            for (long long t = 0; t < kk; ++t)
                d[(o * ci + c) * kk + t] = s[(o * kk + t) * ci + c];
    return RAD_OK;
}

/* ---- conv3d_ndhwc: PyTorch's [cout][cin][kt][k][k] stored tap-major, [cout][kt][k][k][cin].
 *
 * conv2d_nhwc's reordering with the frame tap outermost: the implicit GEMM (r4d_upscaler.hip)
 * contracts over K = (tap, channel), tap = (dt k + dy) k + dx, the channel innermost, so a 16-byte
 * read of a voxel row is eight consecutive K. Operand 1 of conv3d_ndhwc; the bias is read as
 * stored. */
namespace {
int conv3d_geom(const RadParam* p, int n_p, const RadTensor* pl, int n, long long* cout,
                long long* cin, long long* taps) {
    *cout = r4d_pgeti_or(p, n_p, "cout", 0);
    *cin = r4d_pgeti_or(p, n_p, "cin", 0);
    const long long k = r4d_pgeti_or(p, n_p, "k", 0), kt = r4d_pgeti_or(p, n_p, "kt", 0);
    if (n != 1 || !pl || *cout <= 0 || *cin <= 0 || k <= 0 || kt <= 0) return RAD_E_SHAPE;
    *taps = kt * k * k;
    if (pl[0].dtype != RAD_BF16 || rad_tensor_numel(&pl[0]) != *cout * *cin * *taps)
        return RAD_E_SHAPE;
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_conv3d_tap(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* pl, int n, RadLayout* out) {
    (void)sel;
    if (!out) return RAD_E_INVAL;
    if (operand != CONV_W_OPERAND) return RAD_E_UNSUPPORTED;
    if (!enc || !rad_enc_is(enc, "plain")) return RAD_E_DTYPE;
    long long co, ci, taps;
    const int rc = conv3d_geom(p, n_p, pl, n, &co, &ci, &taps);
    if (rc != RAD_OK) return rc;
    fill(out, "r4d.conv3d.tap_major", RAD_BF16, co, taps * ci, co * taps * ci * 2);
    return RAD_OK;
}

extern "C" int r4d_rad_relayout_conv3d_tap(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const RadTensor* pl, int n, void* dst,
                                           int64_t dst_bytes) {
    RadLayout L;
    const int rc = r4d_rad_layout_conv3d_tap(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    long long co, ci, taps;
    conv3d_geom(p, n_p, pl, n, &co, &ci, &taps);
    const uint16_t* s = (const uint16_t*)pl[0].data;
    uint16_t* d = (uint16_t*)dst;
    for (long long o = 0; o < co; ++o)
        for (long long c = 0; c < ci; ++c)
            for (long long t = 0; t < taps; ++t)
                d[(o * taps + t) * ci + c] = s[(o * ci + c) * taps + t];
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_conv3d_tap(const RadParam* p, int n_p, int operand,
                                             const RadEncoding* enc, const int* sel,
                                             const void* src, int64_t src_bytes,
                                             const RadTensor* pl, int n) {
    RadLayout L;
    const int rc = r4d_rad_layout_conv3d_tap(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    long long co, ci, taps;
    conv3d_geom(p, n_p, pl, n, &co, &ci, &taps);
    const uint16_t* s = (const uint16_t*)src;
    uint16_t* d = (uint16_t*)pl[0].data;
    for (long long o = 0; o < co; ++o)
        for (long long c = 0; c < ci; ++c)
            for (long long t = 0; t < taps; ++t)
                d[(o * ci + c) * taps + t] = s[(o * taps + t) * ci + c];
    return RAD_OK;
}

/* ---- conv1d and conv_t1d: a vocoder's 1-D weights, reordered so K runs over (tap, channel) with
 * the channel innermost, as conv2d_nhwc's (r4d_audio.hip). Operand 1 of both.
 *
 *   conv1d     PyTorch's Conv1d [cout][cin][k]            stored [cout][k][cin]
 *   conv_t1d   PyTorch's ConvTranspose1d [cin][cout][2s]  stored [s][cout][2][cin], phase r's
 *              taps (r, r + s) one [cout][2 cin] GEMM operand
 */
namespace {
int conv1d_geom(const RadParam* p, int n_p, const RadTensor* pl, int n, long long* cout,
                long long* cin, long long* k) {
    *cout = r4d_pgeti_or(p, n_p, "cout", 0);
    *cin = r4d_pgeti_or(p, n_p, "cin", 0);
    *k = r4d_pgeti_or(p, n_p, "k", 0);
    if (n != 1 || !pl || *cout <= 0 || *cin <= 0 || *k <= 0) return RAD_E_SHAPE;
    if (pl[0].dtype != RAD_BF16 || rad_tensor_numel(&pl[0]) != *cout * *cin * *k) return RAD_E_SHAPE;
    return RAD_OK;
}
/* The transposed convolution's kernel is `k` = 2 stride (the default) or stride: `taps` = k /
 * stride input rows feed each output row. */
int convt_geom(const RadParam* p, int n_p, const RadTensor* pl, int n, long long* cout,
               long long* cin, long long* s, long long* taps) {
    *cout = r4d_pgeti_or(p, n_p, "cout", 0);
    *cin = r4d_pgeti_or(p, n_p, "cin", 0);
    *s = r4d_pgeti_or(p, n_p, "stride", 0);
    const long long k = r4d_pgeti_or(p, n_p, "k", 2 * *s);
    if (n != 1 || !pl || *cout <= 0 || *cin <= 0 || *s <= 0 || (k != *s && k != 2 * *s))
        return RAD_E_SHAPE;
    *taps = k / *s;
    if (pl[0].dtype != RAD_BF16 || rad_tensor_numel(&pl[0]) != *cin * *cout * k)
        return RAD_E_SHAPE;
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_conv1d_tap(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                         int n, RadLayout* out) {
    (void)sel;
    if (!out) return RAD_E_INVAL;
    if (operand != CONV_W_OPERAND) return RAD_E_UNSUPPORTED;
    if (!enc || !rad_enc_is(enc, "plain")) return RAD_E_DTYPE;
    long long co, ci, k;
    const int rc = conv1d_geom(p, n_p, pl, n, &co, &ci, &k);
    if (rc != RAD_OK) return rc;
    fill(out, "r4d.conv1d.tap_major", RAD_BF16, co, k * ci, co * k * ci * 2);
    return RAD_OK;
}

extern "C" int r4d_rad_relayout_conv1d_tap(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const RadTensor* pl, int n, void* dst,
                                           int64_t dst_bytes) {
    RadLayout L;
    const int rc = r4d_rad_layout_conv1d_tap(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    long long co, ci, k;
    conv1d_geom(p, n_p, pl, n, &co, &ci, &k);
    const uint16_t* s = (const uint16_t*)pl[0].data;
    uint16_t* d = (uint16_t*)dst;
    for (long long o = 0; o < co; ++o)
        for (long long c = 0; c < ci; ++c)
            for (long long t = 0; t < k; ++t) d[(o * k + t) * ci + c] = s[(o * ci + c) * k + t];
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_conv1d_tap(const RadParam* p, int n_p, int operand,
                                             const RadEncoding* enc, const int* sel,
                                             const void* src, int64_t src_bytes,
                                             const RadTensor* pl, int n) {
    RadLayout L;
    const int rc = r4d_rad_layout_conv1d_tap(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    long long co, ci, k;
    conv1d_geom(p, n_p, pl, n, &co, &ci, &k);
    const uint16_t* s = (const uint16_t*)src;
    uint16_t* d = (uint16_t*)pl[0].data;
    for (long long o = 0; o < co; ++o)
        for (long long c = 0; c < ci; ++c)
            for (long long t = 0; t < k; ++t) d[(o * ci + c) * k + t] = s[(o * k + t) * ci + c];
    return RAD_OK;
}

extern "C" int r4d_rad_layout_convt_phase(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, RadLayout* out) {
    (void)sel;
    if (!out) return RAD_E_INVAL;
    if (operand != CONV_W_OPERAND) return RAD_E_UNSUPPORTED;
    if (!enc || !rad_enc_is(enc, "plain")) return RAD_E_DTYPE;
    long long co, ci, st, tp;
    const int rc = convt_geom(p, n_p, pl, n, &co, &ci, &st, &tp);
    if (rc != RAD_OK) return rc;
    fill(out, "r4d.convt1d.phase_major", RAD_BF16, st * co, tp * ci, st * co * tp * ci * 2);
    return RAD_OK;
}

extern "C" int r4d_rad_relayout_convt_phase(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const RadTensor* pl, int n, void* dst,
                                            int64_t dst_bytes) {
    RadLayout L;
    const int rc = r4d_rad_layout_convt_phase(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    long long co, ci, st, tp;
    convt_geom(p, n_p, pl, n, &co, &ci, &st, &tp);
    const uint16_t* s = (const uint16_t*)pl[0].data;
    uint16_t* d = (uint16_t*)dst;
    const long long K = tp * st;
    for (long long r = 0; r < st; ++r)
        for (long long o = 0; o < co; ++o)
            for (long long t = 0; t < tp; ++t)
                for (long long c = 0; c < ci; ++c)
                    d[((r * co + o) * tp + t) * ci + c] = s[(c * co + o) * K + r + t * st];
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_convt_phase(const RadParam* p, int n_p, int operand,
                                              const RadEncoding* enc, const int* sel,
                                              const void* src, int64_t src_bytes,
                                              const RadTensor* pl, int n) {
    RadLayout L;
    const int rc = r4d_rad_layout_convt_phase(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    long long co, ci, st, tp;
    convt_geom(p, n_p, pl, n, &co, &ci, &st, &tp);
    const uint16_t* s = (const uint16_t*)src;
    uint16_t* d = (uint16_t*)pl[0].data;
    const long long K = tp * st;
    for (long long r = 0; r < st; ++r)
        for (long long o = 0; o < co; ++o)
            for (long long t = 0; t < tp; ++t)
                for (long long c = 0; c < ci; ++c)
                    d[(c * co + o) * K + r + t * st] = s[((r * co + o) * tp + t) * ci + c];
    return RAD_OK;
}

/* ---- the f32 rows of conv1d and conv_t1d (r4d_audio.hip's f32 implicit GEMMs), a vocoder kept
 * in f32 throughout:
 *
 *   conv1d     PyTorch's Conv1d [cout][cin][k]                stored [cout][k][cin], as bf16's
 *   conv_t1d   PyTorch's ConvTranspose1d [cin][cout][k]       stored [s][cout][taps][cin],
 *              taps = ceil(k / s): phase r's taps r, r + s, .. below k, the rest zeros, so any
 *              k of at least a stride is one [cout][taps cin] GEMM operand a phase
 */
namespace {
int conv1d_geom_f32(const RadParam* p, int n_p, const RadTensor* pl, int n, long long* cout,
                    long long* cin, long long* k) {
    *cout = r4d_pgeti_or(p, n_p, "cout", 0);
    *cin = r4d_pgeti_or(p, n_p, "cin", 0);
    *k = r4d_pgeti_or(p, n_p, "k", 0);
    if (n != 1 || !pl || *cout <= 0 || *cin <= 0 || *k <= 0) return RAD_E_SHAPE;
    if (pl[0].dtype != RAD_F32 || rad_tensor_numel(&pl[0]) != *cout * *cin * *k) return RAD_E_SHAPE;
    return RAD_OK;
}
int convt_geom_f32(const RadParam* p, int n_p, const RadTensor* pl, int n, long long* cout,
                   long long* cin, long long* s, long long* k, long long* taps) {
    *cout = r4d_pgeti_or(p, n_p, "cout", 0);
    *cin = r4d_pgeti_or(p, n_p, "cin", 0);
    *s = r4d_pgeti_or(p, n_p, "stride", 0);
    *k = r4d_pgeti_or(p, n_p, "k", 2 * *s);
    if (n != 1 || !pl || *cout <= 0 || *cin <= 0 || *s <= 0 || *k < *s) return RAD_E_SHAPE;
    *taps = (*k + *s - 1) / *s;
    if (pl[0].dtype != RAD_F32 || rad_tensor_numel(&pl[0]) != *cin * *cout * *k)
        return RAD_E_SHAPE;
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_conv1d_tap_f32(const RadParam* p, int n_p, int operand,
                                             const RadEncoding* enc, const int* sel,
                                             const RadTensor* pl, int n, RadLayout* out) {
    (void)sel;
    if (!out) return RAD_E_INVAL;
    if (operand != CONV_W_OPERAND) return RAD_E_UNSUPPORTED;
    if (!enc || !rad_enc_is(enc, "plain")) return RAD_E_DTYPE;
    long long co, ci, k;
    const int rc = conv1d_geom_f32(p, n_p, pl, n, &co, &ci, &k);
    if (rc != RAD_OK) return rc;
    fill(out, "r4d.conv1d.tap_major.f32", RAD_F32, co, k * ci, co * k * ci * 4);
    return RAD_OK;
}

extern "C" int r4d_rad_relayout_conv1d_tap_f32(const RadParam* p, int n_p, int operand,
                                               const RadEncoding* enc, const int* sel,
                                               const RadTensor* pl, int n, void* dst,
                                               int64_t dst_bytes) {
    RadLayout L;
    const int rc = r4d_rad_layout_conv1d_tap_f32(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    long long co, ci, k;
    conv1d_geom_f32(p, n_p, pl, n, &co, &ci, &k);
    const float* s = (const float*)pl[0].data;
    float* d = (float*)dst;
    for (long long o = 0; o < co; ++o)
        for (long long c = 0; c < ci; ++c)
            for (long long t = 0; t < k; ++t) d[(o * k + t) * ci + c] = s[(o * ci + c) * k + t];
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_conv1d_tap_f32(const RadParam* p, int n_p, int operand,
                                                 const RadEncoding* enc, const int* sel,
                                                 const void* src, int64_t src_bytes,
                                                 const RadTensor* pl, int n) {
    RadLayout L;
    const int rc = r4d_rad_layout_conv1d_tap_f32(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    long long co, ci, k;
    conv1d_geom_f32(p, n_p, pl, n, &co, &ci, &k);
    const float* s = (const float*)src;
    float* d = (float*)pl[0].data;
    for (long long o = 0; o < co; ++o)
        for (long long c = 0; c < ci; ++c)
            for (long long t = 0; t < k; ++t) d[(o * ci + c) * k + t] = s[(o * k + t) * ci + c];
    return RAD_OK;
}

extern "C" int r4d_rad_layout_convt_phase_f32(const RadParam* p, int n_p, int operand,
                                              const RadEncoding* enc, const int* sel,
                                              const RadTensor* pl, int n, RadLayout* out) {
    (void)sel;
    if (!out) return RAD_E_INVAL;
    if (operand != CONV_W_OPERAND) return RAD_E_UNSUPPORTED;
    if (!enc || !rad_enc_is(enc, "plain")) return RAD_E_DTYPE;
    long long co, ci, st, k, tp;
    const int rc = convt_geom_f32(p, n_p, pl, n, &co, &ci, &st, &k, &tp);
    if (rc != RAD_OK) return rc;
    fill(out, "r4d.convt1d.phase_major.f32", RAD_F32, st * co, tp * ci, st * co * tp * ci * 4);
    return RAD_OK;
}

extern "C" int r4d_rad_relayout_convt_phase_f32(const RadParam* p, int n_p, int operand,
                                                const RadEncoding* enc, const int* sel,
                                                const RadTensor* pl, int n, void* dst,
                                                int64_t dst_bytes) {
    RadLayout L;
    const int rc = r4d_rad_layout_convt_phase_f32(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    long long co, ci, st, k, tp;
    convt_geom_f32(p, n_p, pl, n, &co, &ci, &st, &k, &tp);
    const float* s = (const float*)pl[0].data;
    float* d = (float*)dst;
    for (long long r = 0; r < st; ++r)
        for (long long o = 0; o < co; ++o)
            for (long long t = 0; t < tp; ++t)
                for (long long c = 0; c < ci; ++c) {
                    const long long tap = r + t * st;
                    d[((r * co + o) * tp + t) * ci + c] = tap < k ? s[(c * co + o) * k + tap] : 0.0f;
                }
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_convt_phase_f32(const RadParam* p, int n_p, int operand,
                                                  const RadEncoding* enc, const int* sel,
                                                  const void* src, int64_t src_bytes,
                                                  const RadTensor* pl, int n) {
    RadLayout L;
    const int rc = r4d_rad_layout_convt_phase_f32(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    long long co, ci, st, k, tp;
    convt_geom_f32(p, n_p, pl, n, &co, &ci, &st, &k, &tp);
    const float* s = (const float*)src;
    float* d = (float*)pl[0].data;
    for (long long r = 0; r < st; ++r)
        for (long long o = 0; o < co; ++o)
            for (long long t = 0; t < tp; ++t)
                for (long long c = 0; c < ci; ++c) {
                    const long long tap = r + t * st;
                    if (tap < k) d[(c * co + o) * k + tap] = s[((r * co + o) * tp + t) * ci + c];
                }
    return RAD_OK;
}
