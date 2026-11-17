/* r4d_iqr_layout.cpp -- the IQR weights' stored form (r4d_iqr.h) from a container's planes, and
 * back. Pure host C++, as r4d_fp8_layout.cpp.
 *
 * The weight buffer is the codes reordered into lane segments plus the form's table or zero plane;
 * the scale buffer is the scale plane in tile-major order, with its second level and, on a tcq_rb
 * trellis, the block scale B each block of rows is read at. Every bit of the codes is kept and
 * nothing is widened: VRAM holds a B-bit weight in B bits. The relayout refuses (RAD_E_FORMAT) a
 * code no entry answers and a value outside the int8 grid's -127..127, which no kernel could
 * read. */

#include "r4d_plugin.h"
#include "r4d_iqr.h"
#include "r4d_fp8_frag.h"
#include "r4d_enc.h"
#include "r4d_hsimd.h"
#include "r4d_par.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

using namespace r4d_iqr;

/* Operand indices of `gemm_nt_q` (r4d_rows.cpp). */
enum { IL_B = 2, IL_BSCALE = 3 };

namespace {

int log2_exact(int64_t v) {
    if (v <= 0 || (v & (v - 1))) return -1;
    int l = 0;
    while (((int64_t)1 << l) < v) ++l;
    return l;
}

int unsigned_bits(uint32_t dt) {
    static const uint32_t kU[] = { RAD_U1, RAD_U2, RAD_U3, RAD_U4, RAD_U5, RAD_U6, RAD_U7, RAD_U8 };
    for (int i = 0; i < 8; ++i)
        if (dt == kU[i]) return i + 1;
    return 0;
}
int signed_bits(uint32_t dt) {
    static const uint32_t kS[] = { RAD_I2, RAD_I3, RAD_I4, RAD_I5, RAD_I6, RAD_I7 };
    for (int i = 0; i < 6; ++i)
        if (dt == kS[i]) return i + 2;
    return 0;
}

/* The scale plane a form reads: one bf16 per [1 x 128], or E4M3 per [1 x 128] under an f32 or bf16
 * second level for the tensor or the row. */
bool scale_form(const RadEncoding* enc, int want_planes, uint32_t* out) {
    const RadEncPlane* sc = rad_enc_plane(enc, "scale");
    const RadEncPlane* s2 = rad_enc_plane(enc, "scale.1");
    /* one f32 a row (block [1 x whole]), with a "scale.col" f32 a [whole x 128] or none */
    if (sc && sc->kind == RAD_PLANE_TILED && sc->block[0] == 1 && sc->block[1] == 0 &&
        sc->dtype == RAD_F32 && !s2) {
        const RadEncPlane* cf = rad_enc_plane(enc, "scale.col");
        if (cf && (cf->kind != RAD_PLANE_TILED || cf->dtype != RAD_F32 || cf->block[0] != 0 ||
                   cf->block[1] != 128))
            return false;
        if (enc->n_planes != want_planes + (cf ? 1 : 0)) return false;
        *out = kRowF32;
        return true;
    }
    if (!sc || sc->kind != RAD_PLANE_TILED || sc->block[0] != 1 || sc->block[1] != 128) return false;
    if (!s2) {
        if (sc->dtype != RAD_BF16 || enc->n_planes != want_planes) return false;
        *out = kBf16;
        return true;
    }
    if (sc->dtype != RAD_F8E4M3 || enc->n_planes != want_planes + 1) return false;
    if (s2->kind != RAD_PLANE_TILED || (s2->dtype != RAD_F32 && s2->dtype != RAD_BF16) ||
        s2->block[1] != 0 || (s2->block[0] != 0 && s2->block[0] != 1))
        return false;
    *out = s2->block[0] == 0 ? kE4m3Tensor : kE4m3Row;
    return true;
}

/* Which IQR form `enc` is at transform `tr`, and the entries its table holds. Int8 codes are not
 * one: the int8 GEMMs read them as they are. */
bool classify(const RadEncoding* enc, const char* tr, Form* f, int64_t* entries) {
    *f = Form{};
    *entries = 0;
    if (!enc || std::strncmp(enc->transform, tr, RAD_ENC_STR) != 0) return false;
    const RadEncPlane& cp = enc->plane[0];
    if (std::strncmp(cp.role, "codes", RAD_ENC_STR) != 0) return false;
    if (rad_enc_is(enc, "trellis")) {
        const RadEncPlane* tp = rad_enc_plane(enc, "trellis");
        /* tcq_code=mad: A and B beside the table, the values computed where they are read */
        const RadEncPlane* mp = rad_enc_plane(enc, "trellis.mad");
        const int b = unsigned_bits(cp.dtype);
        uint32_t sc = 0;
        if (!tp || !b || cp.block[0] != 1 || cp.block[1] != 1 || !scale_form(enc, mp ? 4 : 3, &sc))
            return false;
        if (mp && (mp->kind != RAD_PLANE_TABLE || mp->dtype != RAD_U32 || mp->extent[0] != 2 ||
                   mp->extent[1] != 1))
            return false;
        const int L = log2_exact(tp->extent[0]);
        const int lrun = log2_exact(tp->block[0]);
        const int lrb = tp->block[1] > 1 ? log2_exact(tp->block[1]) : 0;
        /* A table's values are read from LDS (16 KiB at L = 14), a computed code's from none; a run
         * is whole 16-row tiles. */
        if (tp->kind != RAD_PLANE_TABLE || tp->dtype != RAD_I8 || tp->extent[1] != 1 || L <= b ||
            L > (mp ? 16 : 14) || lrun < 4 || lrb < 0 || lrb > lrun)
            return false;
        f->kind = mp ? kTrellisMad : kTrellis; f->bits = (uint32_t)b; f->L = (uint32_t)L; f->lrun = (uint32_t)lrun;
        f->lrb = (uint32_t)lrb; f->scale = sc;
        *entries = tp->extent[0];
        return span(*f) <= 16;
    }
    if (!rad_enc_is(enc, "affine")) return false;
    uint32_t sc = 0;
    if (cp.block[0] != 1) return false;
    /* A vector grid: one code a V-vector (V 2 or 4: a lane's four columns hold whole vectors), an i8
     * table of V-vectors. */
    if (cp.block[1] > 1) {
        const RadEncPlane* gr = rad_enc_plane(enc, "grid");
        const int b = unsigned_bits(cp.dtype) ? unsigned_bits(cp.dtype)
                    : cp.dtype == RAD_U9 ? 9 : cp.dtype == RAD_U10 ? 10 : cp.dtype == RAD_U11 ? 11
                    : cp.dtype == RAD_U12 ? 12 : 0;
        const int lv = log2_exact(cp.block[1]);
        if (!gr || !b || lv < 1 || lv > 2 || !scale_form(enc, 3, &sc) || sc != kBf16 ||
            gr->kind != RAD_PLANE_TABLE || gr->dtype != RAD_I8 || gr->extent[1] != cp.block[1] ||
            gr->extent[0] < 1 || gr->extent[0] > ((int64_t)1 << b) ||
            ((int64_t)1 << b) * cp.block[1] > 16384)
            return false;
        f->kind = kGrid; f->bits = (uint32_t)b; f->lv = (uint32_t)lv; f->scale = kBf16;
        *entries = gr->extent[0];
        return true;
    }
    if (const int b = signed_bits(cp.dtype)) {
        if (!scale_form(enc, 2, &sc) || sc != kBf16) return false;
        f->kind = kSigned; f->bits = (uint32_t)b;
        return true;
    }
    const int b = unsigned_bits(cp.dtype);
    /* eight bits only under a zero: the embedding's u8 table (no GEMM form takes it) */
    if (!b || b > 8 || (b == 8 && !rad_enc_plane(enc, "zero"))) return false;
    if (const RadEncPlane* zp = rad_enc_plane(enc, "zero")) {
        if (!scale_form(enc, 3, &sc) || sc != kBf16 || !unsigned_bits(zp->dtype) ||
            zp->block[0] != 1 || zp->block[1] != 128)
            return false;
        f->kind = kZero; f->bits = (uint32_t)b;
        return true;
    }
    if (const RadEncPlane* tp = rad_enc_plane(enc, "table")) {
        if (!scale_form(enc, 3, &sc) || sc != kBf16 || tp->kind != RAD_PLANE_TABLE ||
            tp->dtype != RAD_I8 || tp->extent[0] != 1 || tp->extent[1] < 2 ||
            tp->extent[1] > ((int64_t)1 << b))
            return false;
        f->kind = kTable; f->bits = (uint32_t)b;
        *entries = tp->extent[1];
        return true;
    }
    return false;
}

struct Geom { long long N, K, nkb; };

/* tcq_code=mad's value of state `st`: bytesum((st * A + B) mod 2^32 & 0x3F3F3F3F) - 126, the
 * product from the low 24 bits of each factor (libquant's grid_tcq_mad_values). */
inline int mad_value(uint32_t st, uint32_t a, uint32_t b) {
    const uint32_t h = ((st & 0xFFFFFFu) * (a & 0xFFFFFFu) + b) & 0x3F3F3F3Fu;
    return (int)((h & 0xFFu) + ((h >> 8) & 0xFFu) + ((h >> 16) & 0xFFu) + (h >> 24)) - 126;
}

int geom(const RadParam* p, int n_p, Geom* g) {
    g->N = r4d_pgeti_or(p, n_p, "N", 0);
    g->K = r4d_pgeti_or(p, n_p, "K", 0);
    if (g->N <= 0 || g->K <= 0) return RAD_E_SHAPE;
    if (r4d_pgeti_or(p, n_p, "group", 128) != 128) return RAD_E_SHAPE;
    if (g->N % 16 || g->K % 128) return RAD_E_SHAPE;
    g->nkb = g->K / 128;
    return RAD_OK;
}

void lfill(RadLayout* out, const char* tag, uint32_t dt, long long d0, long long d1,
           long long bytes) {
    std::memset(out, 0, sizeof(*out));
    out->tag = tag;
    out->dtype = dt;
    out->rank = 2;
    out->shape[0] = d0;
    out->shape[1] = d1;
    out->align = 256;
    out->bytes = bytes;
}

/* The selected planes by role. */
struct Planes {
    const RadTensor *codes = nullptr, *aux = nullptr, *scale = nullptr, *scale2 = nullptr,
                    *mad = nullptr, *colf = nullptr;
};
bool planes_of(const RadEncoding* enc, const int* sel, const RadTensor* pl, int n, Planes* o) {
    if (!sel || !pl) return false;
    for (int i = 0; i < n; ++i) {
        const char* r = r4d_enc::role(enc, sel, i);
        if (!std::strncmp(r, "codes", RAD_ENC_STR)) o->codes = &pl[i];
        else if (!std::strncmp(r, "scale", RAD_ENC_STR)) o->scale = &pl[i];
        else if (!std::strncmp(r, "scale.1", RAD_ENC_STR)) o->scale2 = &pl[i];
        else if (!std::strncmp(r, "trellis.mad", RAD_ENC_STR)) o->mad = &pl[i];
        else if (!std::strncmp(r, "scale.col", RAD_ENC_STR)) o->colf = &pl[i];
        else if (!std::strncmp(r, "zero", RAD_ENC_STR) || !std::strncmp(r, "table", RAD_ENC_STR) ||
                 !std::strncmp(r, "grid", RAD_ENC_STR) || !std::strncmp(r, "trellis", RAD_ENC_STR))
            o->aux = &pl[i];
        else return false;
    }
    return true;
}

/* The weight's form and planes, checked against its geometry. */
int weight_view(const RadParam* p, int n_p, const RadEncoding* enc, const int* sel,
                const RadTensor* pl, int n, Geom* g, Form* f, int64_t* entries, Planes* v) {
    int rc = geom(p, n_p, g);
    if (rc != RAD_OK) return rc;
    const char* d = r4d_pgets(p, n_p, "dtype");
    const bool rot = d && std::strcmp(d, "i8a8h") == 0;
    if (!classify(enc, rot ? "fwht128" : "", f, entries) || !built(word(*f))) return RAD_E_DTYPE;
    if (!planes_of(enc, sel, pl, n, v) || !v->codes) return RAD_E_SHAPE;
    const bool aux = f->kind != kSigned;
    if (aux != (v->aux != nullptr) || (f->kind == kTrellisMad) != (v->mad != nullptr))
        return RAD_E_SHAPE;
    const long long kc = f->kind == kGrid ? g->K >> f->lv : g->K;
    if (!r4d_enc::dims(*v->codes, g->N, kc)) return RAD_E_SHAPE;
    if (is_trellis(f->kind) && g->N % ((long long)1 << f->lrun)) return RAD_E_SHAPE;
    if (f->kind == kZero && !r4d_enc::dims(*v->aux, g->N, g->nkb)) return RAD_E_SHAPE;
    return RAD_OK;
}

/* ---- bits ----------------------------------------------------------------------------------- */

/* `n` <= 57 bits of a plane row from bit `o`. */
inline uint64_t get_bits(const uint8_t* row, long long o, int n) {
    const uint8_t* b = row + (o >> 3);
    const int sh = (int)(o & 7);
    const int bytes = (sh + n + 7) >> 3;
    uint64_t w = 0;
    for (int i = 0; i < bytes; ++i) w |= (uint64_t)b[i] << (8 * i);
    return (w >> sh) & (n == 64 ? ~0ull : ((1ull << n) - 1ull));
}
/* OR `n` <= 57 bits into a zeroed buffer at bit `o`. */
inline void put_bits(uint8_t* dst, long long o, uint64_t v, int n) {
    uint8_t* b = dst + (o >> 3);
    const int sh = (int)(o & 7);
    const int bytes = (sh + n + 7) >> 3;
    v <<= sh;
    for (int i = 0; i < bytes; ++i) b[i] |= (uint8_t)(v >> (8 * i));
}

using r4d_par::par_for;

/* 16-row tiles a relayout thread takes: about 16 MiB of codes. */
long long tiles_a_thread(const Form& f, long long K) {
    return std::max(1ll, ((long long)16 << 20) / std::max(1ll, (long long)(K / 128) * unit_bytes(f)));
}

/* Bits [o, o + n) of a plane row, n <= 32, read with one unaligned 8-byte load (fewer bytes where
 * the plane ends within 8 bytes, at `end`). */
inline uint32_t peek_bits(const uint8_t* row, long long o, int n, const uint8_t* end) {
    const uint8_t* b = row + (o >> 3);
    uint64_t w = 0;
    if (end - b >= 8) std::memcpy(&w, b, 8);
    else std::memcpy(&w, b, (size_t)(end - b));
    return (uint32_t)((w >> (o & 7)) & ((1ull << n) - 1ull));
}

/* Appends fields of up to 32 bits to a byte-aligned stream, a dword at a time. */
struct BitOut {
    uint8_t* p;
    uint64_t acc = 0;
    int      n = 0;   /* bits held, below 32 between puts */
    explicit BitOut(uint8_t* at) : p(at) {}
    void put(uint64_t v, int w) {
        acc |= v << n;
        n += w;
        if (n >= 32) {
            const uint32_t x = (uint32_t)acc;
            std::memcpy(p, &x, 4);
            p += 4;
            acc >>= 32;
            n -= 32;
        }
    }
    /* The held bits, a whole number of bytes at a segment's end. */
    void done() {
        for (; n > 0; n -= 8, acc >>= 8) *p++ = (uint8_t)acc;
    }
};

/* Visits each code of the weight with its row-major position (row r, code column c in the codes
 * plane) and its bit in the stored codes plane: the one definition of r4d_iqr.h's order, walked
 * forwards by the relayout and backwards by the unrelayout. `per_tile(rt, visit)` gives a worker a
 * whole 16-row tile so it can read its rows once. */
template <typename V>
void walk_tile(const Form& f, long long K, long long rt, V visit) {
    const long long nkb = K / 128;
    const long long ub = unit_bytes(f) * 8;
    const int B = (int)f.bits;
    for (long long kb = 0; kb < nkb; ++kb) {
        const long long u0 = (rt * nkb + kb) * ub;
        for (int lane = 0; lane < 32; ++lane) {
            const long long l0 = u0 + (long long)lane * lane_bits(f), k0 = kb * 128 + 4 * lane;
            if (is_trellis(f.kind)) {
                for (int e = 0; e < 4; ++e)
                    for (int q = 0; q < 16; ++q)
                        visit(rt * 16 + 15 - q, k0 + e, l0 + (e * 16 + q) * B);
                continue;
            }
            const int per = f.kind == kGrid ? 4 >> f.lv : 4;   /* codes a row */
            for (int r = 0; r < 16; ++r)
                for (int e = 0; e < per; ++e)
                    visit(rt * 16 + r, f.kind == kGrid ? (k0 >> f.lv) + e : k0 + e,
                          l0 + (r * per + e) * B);
        }
    }
}

/* The value a decoded code stands for, or a code no entry answers (-1000). */
int scalar_value(const Form& f, const Planes& v, int64_t entries, long long r, long long c,
                 uint32_t code) {
    if (f.kind == kSigned) return (int)(code << (32 - f.bits)) >> (32 - f.bits);
    if (f.kind == kTable)
        return (int64_t)code < entries ? ((const int8_t*)v.aux->data)[code] : -1000;
    if (f.kind == kZero) {
        const int zb = unsigned_bits(v.aux->dtype);
        return (int)code - (int)rad_load_bits(r4d_enc::row(*v.aux, r), zb, c / 128);
    }
    return 0;
}

/* The codes in r4d_iqr.h's lane segments. A trellis tile goes to r4d_hsimd::iqr_trellis_tile;
 * for the other forms, a segment at a time, each of the tile's 16 rows gives a lane its codes
 * 4l .. 4l + 3 (4/V for a grid) as one field read with a single load, and the segment is written
 * as a stream of dwords. walk_tile's order, which the unrelayout walks, element
 * for element: relayout_iqr_test holds the two to each other on every form. A code the form cannot
 * read (no entry answers it, or its value is outside -127..127) refuses the weight. */
int weight_relayout(const Form& f, const Geom& g, int64_t entries, const Planes& v, uint8_t* d,
                    int64_t bytes) {
    if (bytes != weight_bytes(f, g.N, g.K) || !v.codes->data) return RAD_E_SHAPE;
    /* The segments cover the codes area whole; only what follows it is cleared. */
    const int64_t cb = codes_bytes(f, g.N, g.K);
    std::memset(d + cb, 0, (size_t)(bytes - cb));
    std::atomic<int> bad{ RAD_OK };
    const int B = (int)f.bits;
    const bool tr = is_trellis(f.kind);
    const int per = f.kind == kGrid ? 4 >> f.lv : 4;   /* codes a row gives a lane */
    const int w = per * B;
    const uint32_t m = (1u << B) - 1u;
    const long long rb = rad_dtype_bytes(v.codes->dtype, v.codes->shape[1]);
    const uint8_t* c0p = (const uint8_t*)v.codes->data;
    const uint8_t* end = c0p + rb * g.N;
    const long long ub = unit_bytes(f), lb = lane_bits(f) / 8;
    /* Which codes refuse, for the forms whose check is the code alone. */
    std::vector<uint8_t> refuse;
    if (f.kind == kSigned || f.kind == kTable || f.kind == kGrid) {
        refuse.assign((size_t)1 << B, 0);
        for (uint32_t c = 0; c <= m; ++c) {
            const int val = f.kind == kGrid ? 0 : scalar_value(f, v, entries, 0, 0, c);
            refuse[c] = f.kind == kGrid ? (int64_t)c >= entries : (val < -127 || val > 127);
        }
    }
    const int zb = f.kind == kZero ? unsigned_bits(v.aux->dtype) : 0;
    par_for(g.N / 16, tiles_a_thread(f, g.K), [&](long long lo, long long hi) {
        uint32_t ch[16];
        int zero[16];
        for (long long rt = lo; rt < hi; ++rt) {
            const uint8_t* rows = c0p + rt * 16 * rb;
            /* A trellis tile is a transpose of its codes, done a unit at a time in vector
             * registers (r4d_hsimd.h). */
            if (tr) {
                r4d_hsimd::iqr_trellis_tile(rows, rb, B, g.nkb, d + rt * g.nkb * ub);
                continue;
            }
            for (long long kb = 0; kb < g.nkb; ++kb) {
                uint8_t* u = d + (rt * g.nkb + kb) * ub;
                if (f.kind == kZero)
                    for (int r = 0; r < 16; ++r)
                        zero[r] = (int)rad_load_bits(r4d_enc::row(*v.aux, rt * 16 + r), zb, kb);
                for (int lane = 0; lane < 32; ++lane) {
                    const long long c = f.kind == kGrid ? (kb * 128 + 4 * lane) >> f.lv : kb * 128 + 4 * lane;
                    for (int r = 0; r < 16; ++r) ch[r] = peek_bits(rows + r * rb, c * B, w, end);
                    BitOut o(u + lane * lb);
                    {
                        if (!refuse.empty()) {
                            uint32_t any = 0;
                            for (int r = 0; r < 16; ++r)
                                for (int e = 0; e < per; ++e) any |= refuse[(ch[r] >> (e * B)) & m];
                            if (any) bad = RAD_E_FORMAT;
                        } else if (f.kind == kZero) {
                            for (int r = 0; r < 16; ++r)
                                for (int e = 0; e < per; ++e) {
                                    const int val = (int)((ch[r] >> (e * B)) & m) - zero[r];
                                    if (val < -127 || val > 127) bad = RAD_E_FORMAT;
                                }
                        }
                        for (int r = 0; r < 16; ++r) o.put(ch[r], w);
                    }
                    o.done();
                }
            }
        }
    });
    if (bad != RAD_OK) return bad;
    uint8_t* aux = d + aux_off(f, g.N, g.K);
    if (f.kind == kTable || f.kind == kGrid || is_trellis(f.kind)) {
        const long long n = f.kind == kTable ? entries
                          : f.kind == kGrid ? entries << f.lv : entries;
        const int8_t* t = (const int8_t*)v.aux->data;
        for (long long i = 0; i < n; ++i)
            if (t[i] < -127 && f.kind != kTable) return RAD_E_FORMAT;
        if (f.kind == kTrellisMad) {
            /* The kernels compute the values from A and B, which is all the layout keeps: the
             * stored table must be that code. */
            if (!v.mad || !v.mad->data) return RAD_E_SHAPE;
            const uint32_t a = ((const uint32_t*)v.mad->data)[0], b = ((const uint32_t*)v.mad->data)[1];
            for (long long st = 0; st < n; ++st)
                if (mad_value((uint32_t)st, a, b) != t[st]) return RAD_E_FORMAT;
            std::memcpy(aux, v.mad->data, 8);
        } else {
            std::memcpy(aux, t, (size_t)n);
        }
    } else if (f.kind == kZero) {
        const int zb = unsigned_bits(v.aux->dtype);
        for (long long r = 0; r < g.N; ++r)
            for (long long kb = 0; kb < g.nkb; ++kb)
                aux[tile_major(r, kb, g.nkb)] = (uint8_t)rad_load_bits(r4d_enc::row(*v.aux, r), zb, kb);
    }
    return RAD_OK;
}

int weight_unrelayout(const Form& f, const Geom& g, int64_t entries, const Planes& v,
                      const uint8_t* s, int64_t bytes) {
    if (bytes != weight_bytes(f, g.N, g.K) || !v.codes->data) return RAD_E_SHAPE;
    const int B = (int)f.bits;
    const long long rb = rad_dtype_bytes(v.codes->dtype, v.codes->shape[1]);
    std::memset(v.codes->data, 0, (size_t)(rb * g.N));
    par_for(g.N / 16, tiles_a_thread(f, g.K), [&](long long lo, long long hi) {
        for (long long rt = lo; rt < hi; ++rt)
            walk_tile(f, g.K, rt, [&](long long r, long long c, long long bit) {
                put_bits((uint8_t*)v.codes->data + r * rb, c * B, get_bits(s, bit, B), B);
            });
    });
    const uint8_t* aux = s + aux_off(f, g.N, g.K);
    if (f.kind == kTable || f.kind == kGrid || is_trellis(f.kind)) {
        const long long n = f.kind == kGrid ? entries << f.lv : entries;
        if (f.kind == kTrellisMad) {
            /* The table back from the constants the layout kept. */
            uint32_t ab[2];
            std::memcpy(ab, aux, 8);
            for (long long st = 0; st < n; ++st)
                ((int8_t*)v.aux->data)[st] = (int8_t)mad_value((uint32_t)st, ab[0], ab[1]);
            if (v.mad && v.mad->data) std::memcpy(v.mad->data, aux, 8);
        } else {
            std::memcpy(v.aux->data, aux, (size_t)n);
        }
    } else if (f.kind == kZero) {
        const int zb = unsigned_bits(v.aux->dtype);
        const long long zr = rad_dtype_bytes(v.aux->dtype, v.aux->shape[1]);
        std::memset(v.aux->data, 0, (size_t)(zr * g.N));
        for (long long r = 0; r < g.N; ++r)
            for (long long kb = 0; kb < g.nkb; ++kb)
                rad_store_bits((uint8_t*)v.aux->data + r * zr, zb, kb, aux[tile_major(r, kb, g.nkb)]);
    }
    return RAD_OK;
}

/* ---- scales --------------------------------------------------------------------------------- */

/* The bf16 bits of a row's scale for block kb: the plane's own, or the E4M3 times its second level
 * rounded to nearest even (libquant's grid_tcq_settle). */
uint16_t own_scale(const Form& f, const Planes& v, long long r, long long kb) {
    if (f.scale == kBf16) return ((const uint16_t*)r4d_enc::row(*v.scale, r))[kb];
    const uint8_t e = r4d_enc::row(*v.scale, r)[kb];
    const float s2 = rad_load_f32(v.scale2->data, v.scale2->dtype, f.scale == kE4m3Row ? r : 0);
    const float x = s2 * rad_fp8e4m3_to_f32(e);
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

int scale_view(const RadParam* p, int n_p, const RadEncoding* enc, const int* sel,
               const RadTensor* pl, int n, Geom* g, Form* f, Planes* v) {
    int rc = geom(p, n_p, g);
    if (rc != RAD_OK) return rc;
    const char* d = r4d_pgets(p, n_p, "dtype");
    const bool rot = d && std::strcmp(d, "i8a8h") == 0;
    int64_t entries = 0;
    if (!classify(enc, rot ? "fwht128" : "", f, &entries) || !built(word(*f))) return RAD_E_DTYPE;
    if (!planes_of(enc, sel, pl, n, v) || !v->scale || v->codes || v->aux) return RAD_E_SHAPE;
    if (f->scale == kRowF32) {
        if (v->scale2 || !r4d_enc::dims(*v->scale, g->N, 1)) return RAD_E_SHAPE;
        if (v->colf && !r4d_enc::dims(*v->colf, 1, g->nkb)) return RAD_E_SHAPE;
        if ((rad_enc_plane(enc, "scale.col") != nullptr) != (v->colf != nullptr)) return RAD_E_SHAPE;
        return RAD_OK;
    }
    if ((f->scale != kBf16) != (v->scale2 != nullptr)) return RAD_E_SHAPE;
    if (!r4d_enc::dims(*v->scale, g->N, g->nkb)) return RAD_E_SHAPE;
    if (v->scale2 && (v->scale2->rank != 2 || v->scale2->shape[1] != 1 ||
                      v->scale2->shape[0] != (f->scale == kE4m3Row ? g->N : 1)))
        return RAD_E_SHAPE;
    return RAD_OK;
}

int scale_relayout(const Form& f, const Geom& g, const Planes& v, uint8_t* d, int64_t bytes) {
    if (bytes != scale_bytes(f, g.N, g.K) || !v.scale->data) return RAD_E_SHAPE;
    std::memset(d, 0, (size_t)bytes);
    if (f.scale == kRowF32) {
        float* rs = (float*)d;
        float* cf = (float*)(d + cf_off(f, g.N, g.K));
        for (long long r = 0; r < g.N; ++r) rs[r] = *(const float*)r4d_enc::row(*v.scale, r);
        for (long long kb = 0; kb < g.nkb; ++kb)
            cf[kb] = v.colf ? ((const float*)v.colf->data)[kb] : 1.0f;
        return RAD_OK;
    }
    for (long long r = 0; r < g.N; ++r)
        for (long long kb = 0; kb < g.nkb; ++kb) {
            const long long i = tile_major(r, kb, g.nkb);
            if (f.scale == kBf16) ((uint16_t*)d)[i] = ((const uint16_t*)r4d_enc::row(*v.scale, r))[kb];
            else d[i] = r4d_enc::row(*v.scale, r)[kb];
        }
    if (f.scale != kBf16) {
        float* s2 = (float*)(d + s2_off(f, g.N, g.K));
        const long long n2 = f.scale == kE4m3Row ? g.N : 1;
        for (long long r = 0; r < n2; ++r) s2[r] = rad_load_f32(v.scale2->data, v.scale2->dtype, r);
    }
    if (f.lrb) {
        /* Scales are positive, so their bf16 bits order as the values do. */
        uint16_t* bp = (uint16_t*)(d + blk_off(f, g.N, g.K));
        const long long rb = 1ll << f.lrb;
        for (long long b = 0; b < g.N / rb; ++b)
            for (long long kb = 0; kb < g.nkb; ++kb) {
                uint16_t m = 0;
                for (long long r = b * rb; r < b * rb + rb; ++r) m = std::max(m, own_scale(f, v, r, kb));
                bp[b * g.nkb + kb] = m;
            }
    }
    return RAD_OK;
}

int scale_unrelayout(const Form& f, const Geom& g, const Planes& v, const uint8_t* s,
                     int64_t bytes) {
    if (bytes != scale_bytes(f, g.N, g.K) || !v.scale->data) return RAD_E_SHAPE;
    if (f.scale == kRowF32) {
        const float* rs = (const float*)s;
        const float* cf = (const float*)(s + cf_off(f, g.N, g.K));
        for (long long r = 0; r < g.N; ++r) *(float*)((uint8_t*)v.scale->data + r * 4) = rs[r];
        if (v.colf)
            for (long long kb = 0; kb < g.nkb; ++kb) ((float*)v.colf->data)[kb] = cf[kb];
        return RAD_OK;
    }
    for (long long r = 0; r < g.N; ++r)
        for (long long kb = 0; kb < g.nkb; ++kb) {
            const long long i = tile_major(r, kb, g.nkb);
            uint8_t* row = (uint8_t*)v.scale->data + r * rad_dtype_bytes(v.scale->dtype, g.nkb);
            if (f.scale == kBf16) ((uint16_t*)row)[kb] = ((const uint16_t*)s)[i];
            else row[kb] = s[i];
        }
    /* The second level is f32 here and may be bf16 in the container; a bf16 one round-trips
     * through f32 exactly. */
    if (f.scale != kBf16) {
        const float* s2 = (const float*)(s + s2_off(f, g.N, g.K));
        const long long n2 = f.scale == kE4m3Row ? g.N : 1;
        for (long long r = 0; r < n2; ++r) {
            if (v.scale2->dtype == RAD_F32) {
                ((float*)v.scale2->data)[r] = s2[r];
            } else {
                uint32_t u;
                std::memcpy(&u, &s2[r], 4);
                ((uint16_t*)v.scale2->data)[r] = (uint16_t)(u >> 16);
            }
        }
    }
    return RAD_OK;
}

}  /* namespace */

/* ==================================================================== the hooks */

extern "C" int r4d_rad_layout_iqr(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                  const int* sel, const RadTensor* pl, int n, RadLayout* out) {
    if (!out) return RAD_E_INVAL;
    Geom g;
    Form f;
    Planes v;
    if (operand == IL_B) {
        int64_t entries = 0;
        const int rc = weight_view(p, n_p, enc, sel, pl, n, &g, &f, &entries, &v);
        if (rc != RAD_OK) return rc;
        lfill(out, "r4d.iqr.w", R4D_DT_IQR_W | word(f), g.N, g.K, weight_bytes(f, g.N, g.K));
        return RAD_OK;
    }
    if (operand == IL_BSCALE) {
        const int rc = scale_view(p, n_p, enc, sel, pl, n, &g, &f, &v);
        if (rc != RAD_OK) return rc;
        lfill(out, "r4d.iqr.s", R4D_DT_IQR_S | word(f), g.N, g.nkb, scale_bytes(f, g.N, g.K));
        return RAD_OK;
    }
    return RAD_E_UNSUPPORTED;
}

extern "C" int r4d_rad_relayout_iqr(const RadParam* p, int n_p, int operand,
                                    const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                    int n, void* dst, int64_t dst_bytes) {
    Geom g;
    Form f;
    Planes v;
    if (operand == IL_B) {
        int64_t entries = 0;
        const int rc = weight_view(p, n_p, enc, sel, pl, n, &g, &f, &entries, &v);
        if (rc != RAD_OK) return rc;
        if (v.aux && !v.aux->data) return RAD_E_SHAPE;
        return weight_relayout(f, g, entries, v, (uint8_t*)dst, dst_bytes);
    }
    if (operand == IL_BSCALE) {
        const int rc = scale_view(p, n_p, enc, sel, pl, n, &g, &f, &v);
        if (rc != RAD_OK) return rc;
        if (v.scale2 && !v.scale2->data) return RAD_E_SHAPE;
        return scale_relayout(f, g, v, (uint8_t*)dst, dst_bytes);
    }
    return RAD_E_UNSUPPORTED;
}

extern "C" int r4d_rad_unrelayout_iqr(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel, const void* src,
                                      int64_t src_bytes, const RadTensor* pl, int n) {
    Geom g;
    Form f;
    Planes v;
    if (operand == IL_B) {
        int64_t entries = 0;
        const int rc = weight_view(p, n_p, enc, sel, pl, n, &g, &f, &entries, &v);
        if (rc != RAD_OK) return rc;
        if (v.aux && !v.aux->data) return RAD_E_SHAPE;
        return weight_unrelayout(f, g, entries, v, (const uint8_t*)src, src_bytes);
    }
    if (operand == IL_BSCALE) {
        const int rc = scale_view(p, n_p, enc, sel, pl, n, &g, &f, &v);
        if (rc != RAD_OK) return rc;
        if (v.scale2 && !v.scale2->data) return RAD_E_SHAPE;
        return scale_unrelayout(f, g, v, (const uint8_t*)src, src_bytes);
    }
    return RAD_E_UNSUPPORTED;
}

/* ==================================================================== embed_lookup's table
 *
 * The rows as the container stores them (r4d_iqr.h's embed_off): only the gather's handful of rows
 * a step are decoded, so nothing is reordered, and nothing is widened. A plain bf16 table is the
 * bf16 gather's. */
namespace {
enum { EL_WTE = 1 };

int embed_view(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
               const RadTensor* pl, int n, Form* f, int64_t* entries, Planes* v, long long* rows,
               long long* dim) {
    if (operand != EL_WTE) return RAD_E_UNSUPPORTED;
    *rows = r4d_pgeti_or(p, n_p, "n_vocab", 0);
    *dim = r4d_pgeti_or(p, n_p, "n_embd", 0);
    if (*rows <= 0 || *dim <= 0 || *dim % 128) return RAD_E_SHAPE;
    if (rad_enc_is(enc, "plain")) return RAD_E_DTYPE;
    const bool rot = !std::strncmp(enc->transform, "fwht128", RAD_ENC_STR);
    if (!classify(enc, rot ? "fwht128" : "", f, entries)) return RAD_E_DTYPE;
    f->rot = rot ? 1u : 0u;
    if (!embed_built(word(*f))) return RAD_E_DTYPE;
    if (!planes_of(enc, sel, pl, n, v) || !v->codes || !v->scale) return RAD_E_SHAPE;
    if ((f->kind != kSigned) != (v->aux != nullptr)) return RAD_E_SHAPE;
    if ((f->scale != kBf16) != (v->scale2 != nullptr)) return RAD_E_SHAPE;
    const long long nkb = *dim / 128;
    if (!r4d_enc::dims(*v->codes, *rows, code_cols(*f, *dim)) ||
        !r4d_enc::dims(*v->scale, *rows, nkb))
        return RAD_E_SHAPE;
    if (f->kind == kTrellis && *rows % (1ll << f->lrun)) return RAD_E_SHAPE;
    if (f->kind == kZero && !r4d_enc::dims(*v->aux, *rows, nkb)) return RAD_E_SHAPE;
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_iqr_embed(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                        int n, RadLayout* out) {
    if (!out) return RAD_E_INVAL;
    Form f;
    int64_t entries = 0;
    Planes v;
    long long rows = 0, dim = 0;
    const int rc = embed_view(p, n_p, operand, enc, sel, pl, n, &f, &entries, &v, &rows, &dim);
    if (rc != RAD_OK) return rc;
    lfill(out, "r4d.iqr.embed", R4D_DT_IQR_W | word(f), rows, dim, embed_off(f, rows, dim).bytes);
    return RAD_OK;
}

extern "C" int r4d_rad_relayout_iqr_embed(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, void* dst,
                                          int64_t dst_bytes) {
    Form f;
    int64_t entries = 0;
    Planes v;
    long long rows = 0, dim = 0;
    const int rc = embed_view(p, n_p, operand, enc, sel, pl, n, &f, &entries, &v, &rows, &dim);
    if (rc != RAD_OK) return rc;
    const EmbedOff o = embed_off(f, rows, dim);
    if (dst_bytes != o.bytes || !v.codes->data || !v.scale->data || (v.aux && !v.aux->data) ||
        (v.scale2 && !v.scale2->data))
        return RAD_E_SHAPE;
    uint8_t* d = (uint8_t*)dst;
    const long long nkb = dim / 128, crow = rad_dtype_bytes(v.codes->dtype, v.codes->shape[1]);
    if (crow != o.codes_row) return RAD_E_SHAPE;
    /* The codes are copied whole; only what follows them is cleared first. */
    std::memcpy(d, v.codes->data, (size_t)(rows * crow));
    if (dst_bytes > rows * crow) std::memset(d + rows * crow, 0, (size_t)(dst_bytes - rows * crow));
    uint8_t* aux = d + o.aux;
    if (f.kind == kTable || f.kind == kGrid || f.kind == kTrellis) {
        const long long na = f.kind == kGrid ? entries << f.lv : entries;
        const int8_t* t = (const int8_t*)v.aux->data;
        for (long long i = 0; i < na; ++i)
            if (t[i] < -127) return RAD_E_FORMAT;
        std::memcpy(aux, t, (size_t)na);
    } else if (f.kind == kZero) {
        const int zb = unsigned_bits(v.aux->dtype);
        for (long long r = 0; r < rows; ++r)
            for (long long kb = 0; kb < nkb; ++kb)
                aux[r * nkb + kb] = (uint8_t)rad_load_bits(r4d_enc::row(*v.aux, r), zb, kb);
    }
    const long long sb = f.scale == kBf16 ? 2 : 1;
    for (long long r = 0; r < rows; ++r)
        std::memcpy(d + o.sc + r * nkb * sb, r4d_enc::row(*v.scale, r), (size_t)(nkb * sb));
    if (f.scale != kBf16) {
        float* s2 = (float*)(d + o.s2);
        for (long long r = 0; r < (f.scale == kE4m3Row ? rows : 1); ++r)
            s2[r] = rad_load_f32(v.scale2->data, v.scale2->dtype, r);
    }
    if (f.lrb) {
        uint16_t* bp = (uint16_t*)(d + o.blk);
        const long long rb = 1ll << f.lrb;
        for (long long b = 0; b < rows / rb; ++b)
            for (long long kb = 0; kb < nkb; ++kb) {
                uint16_t m = 0;
                for (long long r = b * rb; r < b * rb + rb; ++r) m = std::max(m, own_scale(f, v, r, kb));
                bp[b * nkb + kb] = m;
            }
    }
    return RAD_OK;
}

/* ==================================================================== logits_gemm's head
 *
 * The rows of r4d_iqr.h's lm form: the container's packed codes and the row's scales, the table
 * after the last row. A reorder of the stored bytes, so it inverts. */
namespace {
enum { LG_W = 1 };

int lm_view(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
            const RadTensor* pl, int n, Form* f, int64_t* entries, Planes* v, long long* rows,
            long long* dim) {
    if (operand != LG_W) return RAD_E_UNSUPPORTED;
    *rows = r4d_pgeti_or(p, n_p, "n_vocab", 0);
    *dim = r4d_pgeti_or(p, n_p, "n_embd", 0);
    if (*rows <= 0 || *dim <= 0 || *dim % 128) return RAD_E_SHAPE;
    if (!classify(enc, "", f, entries) || !lm_built(word(*f))) return RAD_E_DTYPE;
    if (!planes_of(enc, sel, pl, n, v) || !v->codes || !v->scale || !v->aux) return RAD_E_SHAPE;
    if (!r4d_enc::dims(*v->codes, *rows, *dim) || !r4d_enc::dims(*v->scale, *rows, *dim / 128))
        return RAD_E_SHAPE;
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_iqr_logits(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* pl, int n, RadLayout* out) {
    if (!out) return RAD_E_INVAL;
    Form f;
    int64_t entries = 0;
    Planes v;
    long long rows = 0, dim = 0;
    const int rc = lm_view(p, n_p, operand, enc, sel, pl, n, &f, &entries, &v, &rows, &dim);
    if (rc != RAD_OK) return rc;
    lfill(out, "r4d.iqr.lm", R4D_DT_IQR_W | word(f), rows, lm_row(f, dim), lm_bytes(f, rows, dim));
    return RAD_OK;
}

extern "C" int r4d_rad_relayout_iqr_logits(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const RadTensor* pl, int n, void* dst,
                                           int64_t dst_bytes) {
    Form f;
    int64_t entries = 0;
    Planes v;
    long long rows = 0, dim = 0;
    const int rc = lm_view(p, n_p, operand, enc, sel, pl, n, &f, &entries, &v, &rows, &dim);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != lm_bytes(f, rows, dim) || !v.codes->data || !v.scale->data || !v.aux->data)
        return RAD_E_SHAPE;
    uint8_t* d = (uint8_t*)dst;
    const long long stride = lm_row(f, dim), cb = dim / 8 * f.bits, nkb = dim / 128;
    /* Each row written once: its codes, its scales, then the padding cleared; then everything
     * past the rows, the table's area included. */
    const long long used = cb + 2 * nkb;
    for (long long r = 0; r < rows; ++r) {
        uint8_t* row = d + r * stride;
        std::memcpy(row, r4d_enc::row(*v.codes, r), (size_t)cb);
        std::memcpy(row + cb, r4d_enc::row(*v.scale, r), (size_t)(2 * nkb));
        if (stride > used) std::memset(row + used, 0, (size_t)(stride - used));
    }
    if (dst_bytes > rows * stride) std::memset(d + rows * stride, 0, (size_t)(dst_bytes - rows * stride));
    const int8_t* t = (const int8_t*)v.aux->data;
    for (int64_t i = 0; i < entries; ++i)
        if (t[i] < -127) return RAD_E_FORMAT;
    std::memcpy(d + lm_tab(f, rows, dim), t, (size_t)entries);
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_iqr_logits(const RadParam* p, int n_p, int operand,
                                             const RadEncoding* enc, const int* sel,
                                             const void* src, int64_t src_bytes,
                                             const RadTensor* pl, int n) {
    Form f;
    int64_t entries = 0;
    Planes v;
    long long rows = 0, dim = 0;
    const int rc = lm_view(p, n_p, operand, enc, sel, pl, n, &f, &entries, &v, &rows, &dim);
    if (rc != RAD_OK) return rc;
    if (src_bytes != lm_bytes(f, rows, dim) || !v.codes->data || !v.scale->data || !v.aux->data)
        return RAD_E_SHAPE;
    const uint8_t* s = (const uint8_t*)src;
    const long long stride = lm_row(f, dim), cb = dim / 8 * f.bits, nkb = dim / 128;
    for (long long r = 0; r < rows; ++r) {
        std::memcpy((uint8_t*)v.codes->data + r * cb, s + r * stride, (size_t)cb);
        std::memcpy((uint8_t*)v.scale->data + r * nkb * 2, s + r * stride + cb, (size_t)(2 * nkb));
    }
    std::memcpy(v.aux->data, s + lm_tab(f, rows, dim), (size_t)entries);
    return RAD_OK;
}

/* ==================================================================== logits_gemm_vocab's classes
 *
 * Each width class of the head is a gemm_nt_q weight of its own rows (operands 1 + 3i its codes,
 * 2 + 3i its scales), stored as that row stores it, with the class's rows (`rows<i>`) and n_embd
 * as its geometry and the rotated forms (the classes are fwht128). A class's vocabulary ids
 * (3 + 3i) are read as stored. */
namespace {
int vocab_geom(const RadParam* p, int n_p, int operand, RadParam (&q)[4], int* il) {
    if (operand < 1 || operand > 12 || (operand - 1) % 3 == 2) return RAD_E_UNSUPPORTED;
    static const char* kRows[] = { "rows0", "rows1", "rows2", "rows3" };
    const int i = (operand - 1) / 3;
    std::memset(q, 0, sizeof q);
    q[0].key = "N";
    q[0].kind = RAD_P_INT;
    q[0].ival = r4d_pgeti_or(p, n_p, kRows[i], 0);
    q[1].key = "K";
    q[1].kind = RAD_P_INT;
    q[1].ival = r4d_pgeti_or(p, n_p, "n_embd", 0);
    q[2].key = "group";
    q[2].kind = RAD_P_INT;
    q[2].ival = 128;
    q[3].key = "dtype";
    q[3].kind = RAD_P_STR;
    q[3].sval = "i8a8h";
    *il = (operand - 1) % 3 == 0 ? IL_B : IL_BSCALE;
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_iqr_vocab(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                        int n, RadLayout* out) {
    RadParam q[4];
    int il = 0;
    const int rc = vocab_geom(p, n_p, operand, q, &il);
    if (rc != RAD_OK) return rc;
    Form f;
    int64_t entries = 0;
    if (!classify(enc, "fwht128", &f, &entries) || f.kind != kTrellisMad || f.scale != kRowF32 ||
        f.lrb != 0 || f.lrun != 8)
        return RAD_E_DTYPE;
    return r4d_rad_layout_iqr(q, 4, il, enc, sel, pl, n, out);
}

extern "C" int r4d_rad_relayout_iqr_vocab(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, void* dst,
                                          int64_t dst_bytes) {
    RadParam q[4];
    int il = 0;
    const int rc = vocab_geom(p, n_p, operand, q, &il);
    if (rc != RAD_OK) return rc;
    return r4d_rad_relayout_iqr(q, 4, il, enc, sel, pl, n, dst, dst_bytes);
}

extern "C" int r4d_rad_unrelayout_iqr_vocab(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const void* src, int64_t src_bytes,
                                            const RadTensor* pl, int n) {
    RadParam q[4];
    int il = 0;
    const int rc = vocab_geom(p, n_p, operand, q, &il);
    if (rc != RAD_OK) return rc;
    return r4d_rad_unrelayout_iqr(q, 4, il, enc, sel, src, src_bytes, pl, n);
}

/* ==================================================================== the routed experts
 *
 * moe_gemm_iqr's tables: every entry is one expert's IQR weight, relayouted by the dense hooks at
 * its class's geometry. The op's class operands (r4d_layout.cpp's moe_view: `w` / `w_scale` at
 * 2 / 3, the odd class at 6 / 7 with N_odd and K_odd; under four classes 6 .. 11 with N_q and K_q)
 * are answered as the dense GEMM's weight and scale operands with N and K set to the class's. */
namespace {

struct MoeParams {
    RadParam p[32];
    int      n = 0;
    int      operand = 0;
};

int moe_params(const RadParam* p, int n_p, int operand, bool mod4, MoeParams* v) {
    if (n_p < 0 || n_p > 32 || (n_p > 0 && !p)) return RAD_E_INVAL;
    v->n = n_p;
    v->operand = operand;
    for (int i = 0; i < n_p; ++i) v->p[i] = p[i];
    if (operand < 6 || operand > (mod4 ? 11 : 7)) return RAD_OK;
    static const char* const kN[4] = { "N", "N_1", "N_2", "N_3" };
    static const char* const kK[4] = { "K", "K_1", "K_2", "K_3" };
    const int q = (operand - 4) / 2;                 /* 6/7 -> 1, 8/9 -> 2, 10/11 -> 3 */
    v->operand = 2 + (operand & 1);
    const long long nq = r4d_pgeti_or(p, n_p, mod4 ? kN[q] : "N_odd", -1);
    const long long kq = r4d_pgeti_or(p, n_p, mod4 ? kK[q] : "K_odd", -1);
    for (int i = 0; i < n_p; ++i) {
        if (!v->p[i].key) continue;
        if (nq > 0 && std::strcmp(v->p[i].key, "N") == 0) v->p[i].ival = nq;
        if (kq > 0 && std::strcmp(v->p[i].key, "K") == 0) v->p[i].ival = kq;
    }
    return RAD_OK;
}

}  /* namespace */

#define R4D_MOE_IQR_HOOKS(name, mod4)                                                             \
    extern "C" int r4d_rad_layout_##name(const RadParam* p, int n_p, int operand,                \
                                         const RadEncoding* enc, const int* sel,                 \
                                         const RadTensor* pl, int n, RadLayout* out) {           \
        MoeParams v;                                                                              \
        const int rc = moe_params(p, n_p, operand, mod4, &v);                                     \
        return rc != RAD_OK ? rc : r4d_rad_layout_iqr(v.p, v.n, v.operand, enc, sel, pl, n, out); \
    }                                                                                             \
    extern "C" int r4d_rad_relayout_##name(const RadParam* p, int n_p, int operand,              \
                                           const RadEncoding* enc, const int* sel,               \
                                           const RadTensor* pl, int n, void* dst,                \
                                           int64_t dst_bytes) {                                  \
        MoeParams v;                                                                              \
        const int rc = moe_params(p, n_p, operand, mod4, &v);                                     \
        return rc != RAD_OK ? rc : r4d_rad_relayout_iqr(v.p, v.n, v.operand, enc, sel, pl, n,    \
                                                        dst, dst_bytes);                          \
    }                                                                                             \
    extern "C" int r4d_rad_unrelayout_##name(const RadParam* p, int n_p, int operand,            \
                                             const RadEncoding* enc, const int* sel,             \
                                             const void* src, int64_t src_bytes,                 \
                                             const RadTensor* pl, int n) {                       \
        MoeParams v;                                                                              \
        const int rc = moe_params(p, n_p, operand, mod4, &v);                                     \
        return rc != RAD_OK ? rc : r4d_rad_unrelayout_iqr(v.p, v.n, v.operand, enc, sel, src,    \
                                                          src_bytes, pl, n);                      \
    }
R4D_MOE_IQR_HOOKS(moe_iqr, false)
R4D_MOE_IQR_HOOKS(moe_iqr_mod4, true)
#undef R4D_MOE_IQR_HOOKS
