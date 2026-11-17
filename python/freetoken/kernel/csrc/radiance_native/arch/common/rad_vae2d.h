/* rad_vae2d.h -- an image VAE (encoder or decoder) as a recorded list of ops over NHWC images.
 *
 * Image size varies per request, so a VAE's ops are declared once for the largest image and issued
 * with each pass's extents (`Vae::issue`). Qwen-Image 2.1's residual Wan-2.2 VAE, Qwen-Image's
 * and Krea 2's Wan-2.1 one and FLUX's GroupNorm one are each a few lines of `conv`, `norm`,
 * `resblock` and `attention` calls in their plugin.
 */
#pragma once
#include <arch/rad_arch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

namespace rad {
namespace arch {

/* ================================================================== the VAE as a recorded graph
 *
 * Image size varies per request, so the VAE's ops are declared once for the largest image and
 * issued with the actual extents: every image operand is a raw [height, width * channels] view of
 * its buffer (docs/OPS.md, "Image generation"), with extents from the pass's picture size and the
 * stage's scale. `Vae` records each op with what it needs to rebuild its operands, so declare()
 * and step() walk the same list.
 *
 * Every output is its own buffer, sized for its stage; the planner shares storage between those
 * whose lives do not overlap, and no buffer is reused by hand. This is sound because ops are issued
 * in declaration order. */
struct VImg {
    rad_buf buf = 0;
    int64_t ch = 0;
    int     lvl = 0;   /* the side is the pass's base side times 2^lvl (decoder) or / 2^lvl (encoder) */
};

struct VOp {
    enum Kind { CONV, NORM, ADD, D2S, S2D, ATTN, GNORM, LIN } kind = CONV;
    rad_op     op = 0;
    VImg       x{}, y{}, z{};     /* inputs x (and z for ADD), output y */
    rad_weight w = 0, b = 0;
    int        k = 1, up = 1, stride = 1;
    int64_t    qkv_ch = 0;        /* ATTN: x is the [*, 3C] qkv rows, y the [*, C] output */
};

/* What a checkpoint stores a VAE's convolutions as. */
enum VaeConvForm {
    VAE_CONV_2D,       /* [cout, cin, k, k] */
    VAE_CONV_CAUSAL,   /* a causal 3-D convolution [cout, cin, kt, k, k], run on one frame: its
                        * last temporal tap (RAD_MAP_TAP) */
};

struct Vae {
    std::vector<VOp> ops;
    Names*      nm = nullptr;
    RadBuilder* b = nullptr;
    const char* scope = "";      /* "dec" or "enc" */
    bool        down = false;    /* an encoder: a level halves the side */
    int64_t     base_px = 0;     /* pixels at level 0, the largest any pass hands it */
    int         n_buf = 0;
    int         first_group = 0;
    /* The banded tail. From op `band_from` on, every output pixel reads a bounded window of rows
     * (3x3 convolutions, nearest upsampling, per-pixel norms and adds, depth-to-space shortcuts),
     * so a rank can run the tail on its band of level-0 rows plus `halo` rows per side and get its
     * band's pixels bit for bit: rows spoiled by a conv's zero padding at the band edge never reach
     * the band. The ops before it include the mid-block's attention, which reads every pixel, and
     * run whole; `whole` holds their outputs, read by the tail at the band's row offset. -1: no
     * tail. */
    int                  band_from = -1;
    int64_t              halo = 0;
    std::vector<rad_buf> whole;
    VaeConvForm          form = VAE_CONV_2D;
    /* group_norm's statistics scratch, one for every GroupNorm op (they run one after another). */
    rad_buf              gn_stats = 0;

    int64_t px_at(int lvl) const {
        int64_t p = base_px;
        for (int i = 0; i < lvl; ++i) p = down ? (p + 3) / 4 : p * 4;
        return p;
    }
    VImg img(int64_t ch, int lvl) {
        VImg v;
        v.ch = ch;
        v.lvl = lvl;
        v.buf = decl_b(b, nm->f("%s.t%d", scope, n_buf++), RAD_BF16, { px_at(lvl), ch });
        return v;
    }
    rad_weight wt(const char* name, const char* src, std::initializer_list<int64_t> shape,
                  bool tap = false) {
        if ((tap ? map_tap(b, name, src, 2, -1) : map_copy(b, name, src)) < 0) return 0;
        return decl_w(b, name, RAD_BF16, shape, RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE,
                      grp_layer(first_group));
    }

    /* y = conv(x): k x k, stride, `up` 2 reads x upsampled, `pad` same or down. */
    VImg conv(const VImg& x, int64_t cout, int k, int stride, int up, const char* pad,
              const char* ckpt) {
        VOp o;
        o.kind = VOp::CONV;
        o.x = x;
        o.up = up;
        o.stride = stride;
        int lvl = x.lvl;
        if (up == 2) lvl = down ? lvl - 1 : lvl + 1;
        if (stride == 2) lvl = down ? lvl + 1 : lvl - 1;
        o.y = img(cout, lvl);
        o.k = k;
        o.w = wt(nm->f("%s.weight", ckpt), nm->ckpt("%s.weight", ckpt), { cout, x.ch, k, k },
                 form == VAE_CONV_CAUSAL);
        o.b = wt(nm->f("%s.bias", ckpt), nm->ckpt("%s.bias", ckpt), { cout });
        o.op = RAD_OP(b, "conv2d_nhwc",
                      RAD_PARAMS(RAD_RANGE("M", 1, px_at(lvl)), RAD_INT("cin", x.ch),
                                 RAD_INT("cout", cout), RAD_INT("k", k),
                                 RAD_INT("stride", stride), RAD_STR("pad", pad),
                                 RAD_INT("up", up), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(o.w, o.b));
        /* An input with no buffer is the picture the pass staged, read where it lies. */
        if (x.buf) rw(b, o.op, { x.buf }, { o.y.buf });
        else       rw(b, o.op, {}, { o.y.buf });
        ops.push_back(o);
        return o.y;
    }
    /* The channel norm, `silu` or not. `gamma` is [C, 1, 1(, 1)] in the checkpoint. */
    VImg norm(const VImg& x, bool silu, const char* ckpt, int gamma_rank) {
        VOp o;
        o.kind = VOp::NORM;
        o.x = x;
        o.y = img(x.ch, x.lvl);
        const char* dn = nm->f("%s.gamma", ckpt);
        if (map_copy(b, dn, nm->ckpt("%s.gamma", ckpt)) < 0) return VImg{};
        o.w = gamma_rank == 4
                ? decl_w(b, dn, RAD_BF16, { x.ch, 1, 1, 1 }, RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, grp_layer(first_group))
                : decl_w(b, dn, RAD_BF16, { x.ch, 1, 1 }, RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, grp_layer(first_group));
        o.op = rw(b, RAD_OP(b, "l2norm_chan",
                            RAD_PARAMS(RAD_RANGE("M", 1, px_at(x.lvl)), RAD_INT("n", x.ch),
                                       RAD_STR("act", silu ? "silu" : "none"),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(o.w)),
                  { x.buf }, { o.y.buf });
        ops.push_back(o);
        return o.y;
    }
    VImg add(const VImg& x, const VImg& z) {
        VOp o;
        o.kind = VOp::ADD;
        o.x = x;
        o.z = z;
        o.y = img(x.ch, x.lvl);
        o.op = rw(b, RAD_OP(b, "add",
                            RAD_PARAMS(RAD_RANGE("M", 1, px_at(x.lvl)), RAD_INT("n", x.ch),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { x.buf, z.buf }, { o.y.buf });
        ops.push_back(o);
        return o.y;
    }
    /* y += the resampling shortcut of x (in place on y). */
    void shortcut(const VImg& x, const VImg& y, int ft) {
        VOp o;
        o.kind = down ? VOp::S2D : VOp::D2S;
        o.x = x;
        o.y = y;
        o.up = ft;
        o.op = rw(b, RAD_OP(b, down ? "s2d_avg_add" : "d2s_add",
                            RAD_PARAMS(RAD_RANGE("M", 1, px_at(y.lvl)), RAD_INT("cin", x.ch),
                                       RAD_INT("cout", y.ch), RAD_INT("ft", ft),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { x.buf, y.buf }, { y.buf });
        ops.push_back(o);
    }
    /* A residual block, in -> out channels. */
    VImg resblock(const VImg& x, int64_t out, const char* ckpt) {
        VImg h = x;
        if (x.ch != out) h = conv(x, out, 1, 1, 1, "same", nm->f("%s.conv_shortcut", ckpt));
        VImg t = norm(x, true, nm->f("%s.norm1", ckpt), 4);
        t = conv(t, out, 3, 1, 1, "same", nm->f("%s.conv1", ckpt));
        t = norm(t, true, nm->f("%s.norm2", ckpt), 4);
        t = conv(t, out, 3, 1, 1, "same", nm->f("%s.conv2", ckpt));
        return add(t, h);
    }
    /* The mid-block's single-head attention over every pixel. */
    VImg attention(const VImg& x, const char* ckpt) {
        const int64_t C = x.ch;
        VImg t = norm(x, false, nm->f("%s.norm", ckpt), 3);
        VImg qkv = conv(t, 3 * C, 1, 1, 1, "same", nm->f("%s.to_qkv", ckpt));
        VOp o;
        o.kind = VOp::ATTN;
        o.x = qkv;
        o.y = img(C, x.lvl);
        o.qkv_ch = 3 * C;
        o.op = rw(b, RAD_OP(b, "attn_dense",
                            RAD_PARAMS(RAD_RANGE("M", 1, px_at(x.lvl)), RAD_INT("head_dim", C),
                                       RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                       RAD_STR("dtype", "bf16"),
                                       RAD_F64("scale", 1.0 / std::sqrt((double)C))),
                            RAD_NOWEIGHTS),
                  { qkv.buf }, { o.y.buf });
        ops.push_back(o);
        VImg p = conv(o.y, C, 1, 1, 1, "same", nm->f("%s.proj", ckpt));
        return add(p, x);
    }

    /* PyTorch's GroupNorm over the whole picture, `silu` or not (FLUX's VAE). Not row-local, so a
     * VAE that has one runs whole (no banded tail). */
    VImg gnorm(const VImg& x, bool silu, const char* ckpt, int64_t groups, double eps) {
        VOp o;
        o.kind = VOp::GNORM;
        o.x = x;
        o.y = img(x.ch, x.lvl);
        o.w = wt(nm->f("%s.weight", ckpt), nm->ckpt("%s.weight", ckpt), { x.ch });
        o.b = wt(nm->f("%s.bias", ckpt), nm->ckpt("%s.bias", ckpt), { x.ch });
        if (!gn_stats)
            gn_stats = decl_b(b, nm->f("%s.gn_stats", scope), RAD_F32, { 64 * (2 + 3 * 1024) });
        o.k = (int)groups;
        o.op = RAD_OP(b, "group_norm",
                      RAD_PARAMS(RAD_RANGE("M", 1, px_at(x.lvl)), RAD_INT("n", x.ch),
                                 RAD_INT("groups", groups), RAD_F64("eps", eps),
                                 RAD_STR("act", silu ? "silu" : "none"), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(o.w, o.b));
        rw(b, o.op, { x.buf, gn_stats }, { o.y.buf, gn_stats });
        ops.push_back(o);
        return o.y;
    }
    /* A per-pixel linear with a bias (a 1x1 convolution kept as a Linear): `cout` from x's
     * channels. */
    VImg lin(const VImg& x, int64_t cout, const char* ckpt) {
        VOp o;
        o.kind = VOp::LIN;
        o.x = x;
        o.y = img(cout, x.lvl);
        o.w = wt(nm->f("%s.weight", ckpt), nm->ckpt("%s.weight", ckpt), { cout, x.ch });
        o.b = wt(nm->f("%s.bias", ckpt), nm->ckpt("%s.bias", ckpt), { cout });
        o.op = rw(b, RAD_OP(b, "gemm_nt_bias",
                            RAD_PARAMS(RAD_RANGE("M", 1, px_at(x.lvl)), RAD_INT("N", cout),
                                       RAD_INT("K", x.ch), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(o.w, o.b)),
                  { x.buf }, { o.y.buf });
        ops.push_back(o);
        return o.y;
    }
    /* A GroupNorm residual block (FLUX's VAE, diffusers' ResnetBlock2D): norm1, silu, conv1,
     * norm2, silu, conv2, plus the input or its 1x1 `conv_shortcut`. */
    VImg resblock_gn(const VImg& x, int64_t out, const char* ckpt, int64_t groups, double eps) {
        VImg h = x;
        if (x.ch != out) h = conv(x, out, 1, 1, 1, "same", nm->f("%s.conv_shortcut", ckpt));
        VImg t = gnorm(x, true, nm->f("%s.norm1", ckpt), groups, eps);
        t = conv(t, out, 3, 1, 1, "same", nm->f("%s.conv1", ckpt));
        t = gnorm(t, true, nm->f("%s.norm2", ckpt), groups, eps);
        t = conv(t, out, 3, 1, 1, "same", nm->f("%s.conv2", ckpt));
        return add(t, h);
    }
    /* diffusers' single-head VAE attention: GroupNorm, to_q / to_k / to_v linears with biases,
     * attention over every pixel, to_out.0, plus the input. */
    VImg attention_gn(const VImg& x, const char* ckpt, int64_t groups, double eps) {
        const int64_t C = x.ch;
        VImg t = gnorm(x, false, nm->f("%s.group_norm", ckpt), groups, eps);
        VImg q = lin(t, C, nm->f("%s.to_q", ckpt));
        VImg k = lin(t, C, nm->f("%s.to_k", ckpt));
        VImg v = lin(t, C, nm->f("%s.to_v", ckpt));
        VOp o;
        o.kind = VOp::ATTN;
        o.x = q;
        o.y = img(C, x.lvl);
        o.z = k;
        o.w = 0;
        o.qkv_ch = 0;
        o.k = 0;
        o.op = rw(b, RAD_OP(b, "attn_dense",
                            RAD_PARAMS(RAD_RANGE("M", 1, px_at(x.lvl)), RAD_INT("head_dim", C),
                                       RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                       RAD_STR("dtype", "bf16"),
                                       RAD_F64("scale", 1.0 / std::sqrt((double)C))),
                            RAD_NOWEIGHTS),
                  { q.buf, k.buf, v.buf }, { o.y.buf });
        attn_v.push_back({ ops.size(), v });
        ops.push_back(o);
        VImg p = lin(o.y, C, nm->f("%s.to_out.0", ckpt));
        return add(p, x);
    }
    /* The value image of each separate-projection attention, by op index. */
    std::vector<std::pair<size_t, VImg>> attn_v;
    const VImg* value_of(size_t i) const {
        for (const auto& e : attn_v) if (e.first == i) return &e.second;
        return nullptr;
    }

    /* The operands of op `o` for a pass whose level-0 picture is H0 x W0 pixels (or latent tokens,
     * for the decoder), in the op's schema order, and its ranged value. */
    static int64_t side(int64_t s0, int lvl, bool dn) {
        int64_t s = s0;
        for (int i = 0; i < lvl; ++i) s = dn ? s / 2 : s * 2;
        return s;
    }
    /* Declares `cu`, a buffer of the plugin's, as read by every attention: issue() hands it to
     * them. */
    int attn_reads(rad_buf cu) const {
        for (const VOp& o : ops)
            if (o.kind == VOp::ATTN) RAD_ARCH_TRY(rad_op_reads(b, o.op, &cu, 1));
        return RAD_OK;
    }
    /* `cu` is {0, pixels} at the attention's level -- the pass's latent tokens -- and `input` the
     * level-0 picture when the first op reads one the pass staged. Ops [from, to) are issued; with
     * `row0`, H0 is a band's rows and a `whole` buffer is read from its level-0 row `row0`. */
    void issue(RadCtx* c, int64_t H0, int64_t W0, const RadOperand& cu, const void* input,
               size_t from = 0, size_t to = (size_t)-1, int64_t row0 = 0) const;

    bool is_whole(rad_buf v) const {
        for (rad_buf w : whole) if (w == v) return true;
        return false;
    }
    /* The tail's halo in level-0 rows: walked back from the last op, a buffer's margin is the most
     * any reader of it needs, in its own rows -- a k x k conv adds k/2, reading through a 2x
     * upsample halves it rounding up, and the depth-to-space shortcut halves it. -1 when the tail
     * holds an op that is not row-local, or reads a whole buffer above level 0. */
    int64_t band_halo() const {
        if (band_from < 0 || ops.empty()) return -1;
        std::vector<std::pair<rad_buf, int64_t>> need{ { ops.back().y.buf, 0 } };
        auto get = [&](rad_buf v) -> int64_t {
            for (const auto& e : need) if (e.first == v) return e.second;
            return 0;
        };
        auto want = [&](rad_buf v, int64_t m) {
            for (auto& e : need) if (e.first == v) { e.second = std::max(e.second, m); return; }
            need.push_back({ v, m });
        };
        for (size_t i = ops.size(); i-- > (size_t)band_from;) {
            const VOp& o = ops[i];
            const int64_t my = get(o.y.buf);
            switch (o.kind) {
            case VOp::CONV: {
                if (o.stride != 1) return -1;
                const int64_t m = my + o.k / 2;
                want(o.x.buf, o.up == 2 ? (m + 1) / 2 : m);
                break;
            }
            case VOp::NORM: want(o.x.buf, my); break;
            case VOp::LIN:  want(o.x.buf, my); break;
            case VOp::ADD:  want(o.x.buf, my); want(o.z.buf, my); break;
            case VOp::D2S:  want(o.x.buf, (my + 1) / 2); break;
            default: return -1;
            }
        }
        int64_t h = 0;
        for (size_t i = (size_t)band_from; i < ops.size(); ++i) {
            const VOp& o = ops[i];
            for (const VImg* v : { &o.x, &o.z })
                if (v->buf && is_whole(v->buf)) {
                    if (v->lvl != 0) return -1;
                    h = std::max(h, get(v->buf));
                }
        }
        return h;
    }
};

inline void Vae::issue(RadCtx* c, int64_t H0, int64_t W0, const RadOperand& cu,
                       const void* input, size_t from, size_t to, int64_t row0) const {
    auto ptr = [&](const VImg& v) -> const void* {
        if (!v.buf) return input;
        const char* p = (const char*)rad_buf_ptr(c, v.buf);
        return row0 && is_whole(v.buf) ? p + row0 * W0 * v.ch * (int64_t)sizeof(uint16_t) : p;
    };
    auto view = [&](const VImg& v, int64_t* H, int64_t* W) {
        *H = side(H0, v.lvl, down);
        *W = side(W0, v.lvl, down);
        return RAD_P_T2(ptr(v), RAD_BF16, *H, *W * v.ch);
    };
    auto rows = [&](const VImg& v, int64_t cols) {
        const int64_t H = side(H0, v.lvl, down), W = side(W0, v.lvl, down);
        return RAD_P_T2(ptr(v), RAD_BF16, H * W * v.ch / cols, cols);
    };
    for (size_t i = from; i < to && i < ops.size(); ++i) {
        const VOp& o = ops[i];
        int64_t Hx, Wx, Hy, Wy;
        switch (o.kind) {
        case VOp::CONV: {
            const RadOperand x = view(o.x, &Hx, &Wx), y = view(o.y, &Hy, &Wy);
            RAD_ISSUE_N(c, o.op, Hy * Wy, x, RAD_W(o.w), RAD_W(o.b), y);
            break;
        }
        case VOp::NORM: {
            const int64_t H = side(H0, o.x.lvl, down), W = side(W0, o.x.lvl, down);
            RAD_ISSUE_N(c, o.op, H * W, rows(o.x, o.x.ch), RAD_W(o.w), rows(o.y, o.y.ch));
            break;
        }
        case VOp::ADD: {
            const int64_t H = side(H0, o.x.lvl, down), W = side(W0, o.x.lvl, down);
            RAD_ISSUE_N(c, o.op, H * W, rows(o.x, o.x.ch), rows(o.z, o.z.ch), rows(o.y, o.y.ch));
            break;
        }
        case VOp::D2S:
        case VOp::S2D: {
            const RadOperand x = view(o.x, &Hx, &Wx), y = view(o.y, &Hy, &Wy);
            RAD_ISSUE_N(c, o.op, Hy * Wy, x, y);
            break;
        }
        case VOp::ATTN: {
            const int64_t H = side(H0, o.x.lvl, down), W = side(W0, o.x.lvl, down);
            const int64_t C = o.y.ch, n = H * W;
            if (const VImg* v = value_of(i)) {
                /* Separate q, k and v images. */
                RAD_ISSUE_N(c, o.op, n, rows(o.x, C), rows(o.z, C), rows(*v, C), cu,
                            brows(o.y.buf, n));
            } else {
                /* q, k and v are column slices of the qkv rows, at its 3C pitch. */
                RAD_ISSUE_N(c, o.op, n, bcol(o.x.buf, 0, C, n), bcol(o.x.buf, C, C, n),
                            bcol(o.x.buf, 2 * C, C, n), cu, brows(o.y.buf, n));
            }
            break;
        }
        case VOp::GNORM: {
            const int64_t H = side(H0, o.x.lvl, down), W = side(W0, o.x.lvl, down);
            RAD_ISSUE_N(c, o.op, H * W, rows(o.x, o.x.ch), RAD_W(o.w), RAD_W(o.b), RAD_NONE,
                        RAD_P_T2(rad_buf_ptr(c, gn_stats), RAD_F32, 1, 64 * (2 + 3 * 1024)),
                        rows(o.y, o.y.ch));
            break;
        }
        case VOp::LIN: {
            const int64_t H = side(H0, o.x.lvl, down), W = side(W0, o.x.lvl, down);
            RAD_ISSUE_N(c, o.op, H * W, rows(o.x, o.x.ch), RAD_W(o.w), RAD_W(o.b), RAD_NONE,
                        rows(o.y, o.y.ch));
            break;
        }
        }
    }
}

}  /* namespace arch */
}  /* namespace rad */
