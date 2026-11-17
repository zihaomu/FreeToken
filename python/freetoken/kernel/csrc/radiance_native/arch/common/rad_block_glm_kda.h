/* GLM-5.3's KDA layer: the seven bf16 projections around the recurrent core (GlmKda), and the
 * core itself over already projected inputs (KdaState), whose ple_conv supplies exactly causal
 * depthwise convolution + SiLU, cold start and accepted-history rollback included. The two
 * low-rank gate paths have no intervening activation. The caller supplies the normalized input
 * and owns the mHC residual connection.
 *
 * Under tensor parallelism the heads split (GlmTp::heads), and with them the q/k/v rows, the
 * convolution taps, the state and the output projection's columns; the two rank-128 projections
 * into the gates are replicated. The block's all_reduce sums the output. */
#ifndef RAD_BLOCK_GLM_KDA_H
#define RAD_BLOCK_GLM_KDA_H
#include "rad_fp8.h"
#include "rad_glm_tp.h"
#include <cmath>

namespace rad {
namespace arch {
struct KdaState {
    struct Config {
        int64_t heads = 16, dim = 128, width = 4;
        double lower = -5;
    };
    struct Src {
        const char *q_conv = nullptr, *k_conv = nullptr, *v_conv = nullptr, *alog = nullptr,
                   *bias = nullptr, *norm = nullptr;
    };
    struct Wire {
        rad_buf qkv = 0, conv = 0, a = 0, beta = 0, gate = 0, out = 0;
    };
    Geom g{};
    Config c{};
    Wire w{};
    int layer = 0;
    rad_kvgroup state = 0, history = 0;
    rad_weight taps = 0, alog = 0, bias = 0, norm = 0;
    rad_op op_conv = 0, op_verify = 0, op_prefill = 0, op_norm = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup st, rad_kvgroup cv,
                const Config& cfg, const Wire& wire, const Src& src) {
        g = geom;
        c = cfg;
        w = wire;
        layer = l;
        state = st;
        history = cv;
        if (!b || !st || !cv || c.heads < 1 || c.dim != 128 || c.width != 4 || g.max_spec > 31 ||
            !std::isfinite(c.lower) || c.lower >= 0 || !w.qkv || !w.conv || !w.a || !w.beta ||
            !w.gate || !w.out || w.qkv == w.conv || !src.q_conv || !src.k_conv || !src.v_conv ||
            !src.alog || !src.bias || !src.norm)
            return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, st));
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, cv));
        const int64_t width = c.heads * c.dim;
        const char *tn = nm.f("blk.%d.ssm_conv.weight", l), *an = nm.f("blk.%d.ssm_a", l),
                   *bn = nm.f("blk.%d.ssm_dt_bias", l), *nn = nm.f("blk.%d.ssm_norm.weight", l);
        RAD_ARCH_TRY(map_concat(b, tn, { src.q_conv, src.k_conv, src.v_conv }));
        RAD_ARCH_TRY(map_copy(b, an, src.alog));
        RAD_ARCH_TRY(map_copy(b, bn, src.bias));
        RAD_ARCH_TRY(map_copy(b, nn, src.norm));
        taps = decl_w(b, tn, RAD_BF16, { 3 * width, c.width }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,
                      grp_layer(l), 0, { width, width, width });
        alog =
            decl_w(b, an, RAD_F32, { c.heads }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
        bias = decl_w(b, bn, RAD_F32, { width }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
        norm =
            decl_w(b, nn, RAD_F32, { c.dim }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        if (!taps || !alog || !bias || !norm) return RAD_E_INVAL;
        op_conv = rw(b,
                     RAD_OP(b, "ple_conv",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 3 * width),
                                       RAD_INT("width", c.width), RAD_INT("dilation", 1),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(taps)),
                     { w.qkv }, { w.conv });
        for (int verify = 0; verify < 2; ++verify) {
            const rad_op op = rw(
                b,
                RAD_OP(b, "kda_update",
                       RAD_PARAMS(
                           RAD_RANGE("M", 1, g.max_tok), RAD_INT("heads", c.heads),
                           RAD_INT("head_k", c.dim), RAD_INT("head_v", c.dim),
                           RAD_F64("lower_bound", c.lower),
                           RAD_F64("scale", 1.0 / std::sqrt((double)c.dim)),
                           RAD_F64("l2_eps", 1e-6), RAD_STR("mode", verify ? "verify" : "prefill"),
                           RAD_STR("dtype", "bf16"),
                           RAD_STR("state_form", g.max_spec > 0 ? "anchor" : "per_candidate")),
                       RAD_WEIGHTS(alog, bias)),
                { w.conv, w.a, w.beta }, { w.out });
            if (verify)
                op_verify = op;
            else
                op_prefill = op;
        }
        op_norm = rw(b,
                     RAD_OP(b, "gdn_gated_rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("channels", c.dim),
                                       RAD_F64("eps", g.eps), RAD_STR("act", "sigmoid")),
                            RAD_WEIGHTS(norm)),
                     { w.out, w.gate }, { w.out });
        return op_conv && op_verify && op_prefill && op_norm ? RAD_OK : RAD_E_INVAL;
    }
    void step(RadCtx* ctx, const RadBatch* batch) const {
        const auto *st = kv_batch(batch, state), *cv = kv_batch(batch, history);
        if (!st || !cv || !st->state_index || !cv->state_index) {
            rad_step_fail(ctx, "KDA requires recurrent and convolution state indices");
            return;
        }
        const int64_t T = batch->n_tok, S = batch->n_seq, width = c.heads * c.dim;
        const int64_t pitch = st->state_index_pitch > 0 ? st->state_index_pitch : 1;
        int64_t D = 0, DT = 0;
        batch_split(batch, &D, &DT);
        for (int verify = 1; verify >= 0; --verify) {
            const int64_t seq0 = verify ? 0 : D, seqs = verify ? D : S - D, rows = verify ? DT : T;
            if (!seqs) continue;
            const RadOperand cu = praw(batch->cu_seqlens + seq0, RAD_I32, seqs + 1);
            const RadOperand init = praw(batch->ctx_lens + seq0, RAD_I32, seqs);
            const RadOperand accepted =
                verify ? praw(batch->num_accepted, RAD_I32, seqs) : RAD_NONE;
            RAD_ISSUE_N(ctx, op_conv, rows, brows(w.qkv, rows), RAD_W(taps),
                        kv_cache(history, layer), cu, RAD_NONE,
                        praw(cv->state_index + seq0, RAD_I32, seqs), init, accepted,
                        brows(w.conv, rows));
            RAD_ISSUE_N(ctx, verify ? op_verify : op_prefill, rows, bcol(w.conv, 0, width, rows),
                        bcol(w.conv, width, width, rows), bcol(w.conv, 2 * width, width, rows),
                        brows(w.a, rows), brows(w.beta, rows), RAD_W(alog), RAD_W(bias),
                        kv_cache(state, layer),
                        praw2(st->state_index + seq0 * pitch, RAD_I32, seqs, pitch), cu, accepted,
                        init, brows(w.out, rows));
        }
        RAD_ISSUE_N(ctx, op_norm, T, brows(w.out, T), brows(w.gate, T), RAD_W(norm),
                    brows(w.out, T));
    }
};

struct GlmKda {
    struct Wire {
        rad_buf input = 0, forget_low = 0, gate_low = 0, out = 0;
        KdaState::Wire core{};
    };
    Geom g{};
    Wire w{};
    KdaState core{};
    LinearFP8 qkv{}, beta{}, forget_a{}, forget_b{}, gate_a{}, gate_b{}, output{};
    rad_op op_ar = 0;
    int declare(RadBuilder* b, Names& nm, const Geom& geom, int layer, const Wire& wire,
                const char* prefix, const KdaState::Src& src, rad_kvgroup state,
                rad_kvgroup history, bool reduce = true) {
        g = geom;
        w = wire;
        if (!b || !prefix || g.n_embd != 4096 || g.world < 1 || GlmTp::kHeads % g.world ||
            !w.input || !w.forget_low || !w.gate_low || !w.out || w.input == w.out ||
            w.forget_low == w.gate_low)
            return RAD_E_INVAL;
        // All seven projections are plain bf16 in the pinned checkpoint; under w_bf16,
        // LinearFP8::declare refuses a quantised one by name.
        Geom bg = g;
        bg.w_bf16 = true;
        const int64_t width = GlmTp::kda_width(g.world), heads = GlmTp::heads(g.world);
        const auto group = grp_layer(layer);
        auto act = [](rad_buf x) {
            ActFP8 a{};
            a.x = x;
            return a;
        };
        const auto input = act(w.input);
        RAD_ARCH_TRY(qkv.declare(
            b, nm, bg, nm.f("blk.%d.attn_qkv", layer), 3 * width, 4096, RAD_SHARD_ROW, group,
            { nm.ckpt("%s.q_proj.weight", prefix), nm.ckpt("%s.k_proj.weight", prefix),
              nm.ckpt("%s.v_proj.weight", prefix) },
            input, w.core.qkv, 0, { width, width, width }));
        RAD_ARCH_TRY(beta.declare(b, nm, bg, nm.f("blk.%d.attn_beta", layer), heads, 4096,
                                  RAD_SHARD_ROW, group, { nm.ckpt("%s.b_proj.weight", prefix) },
                                  input, w.core.beta));
        /* Both low-rank down-projections before either up-projection, in step() as here: the
         * four reads of the input then share one ordered point of a recorded pass and the two
         * up-projections the next, where forget_b between them cost a third. Declared in the
         * issue order because the buffer plan takes lifetimes from this list, and forget_low and
         * gate_low must not share bytes while both are live. */
        RAD_ARCH_TRY(forget_a.declare(
            b, nm, bg, nm.f("blk.%d.attn_forget_a", layer), 128, 4096, RAD_SHARD_NONE, group,
            { nm.ckpt("%s.f_a_proj.weight", prefix) }, input, w.forget_low));
        RAD_ARCH_TRY(gate_a.declare(b, nm, bg, nm.f("blk.%d.attn_gate_a", layer), 128, 4096,
                                    RAD_SHARD_NONE, group,
                                    { nm.ckpt("%s.g_a_proj.weight", prefix) }, input, w.gate_low));
        RAD_ARCH_TRY(forget_b.declare(
            b, nm, bg, nm.f("blk.%d.attn_forget_b", layer), width, 128, RAD_SHARD_ROW, group,
            { nm.ckpt("%s.f_b_proj.weight", prefix) }, act(w.forget_low), w.core.a));
        RAD_ARCH_TRY(gate_b.declare(b, nm, bg, nm.f("blk.%d.attn_gate_b", layer), width, 128,
                                    RAD_SHARD_ROW, group, { nm.ckpt("%s.g_b_proj.weight", prefix) },
                                    act(w.gate_low), w.core.gate));
        KdaState::Config cfg;
        cfg.heads = heads;
        RAD_ARCH_TRY(core.declare(b, nm, bg, layer, state, history, cfg, w.core, src));
        RAD_ARCH_TRY(output.declare(b, nm, bg, nm.f("blk.%d.attn_output", layer), 4096, width,
                                    RAD_SHARD_COL, group, { nm.ckpt("%s.o_proj.weight", prefix) },
                                    act(w.core.out), w.out));
        if (reduce) op_ar = GlmTp::reduce(b, g, w.out);
        return reduce && g.world > 1 && !op_ar ? RAD_E_INVAL : RAD_OK;
    }
    void step(RadCtx* ctx, const RadBatch* batch) const {
        const int64_t T = batch->n_tok;
        auto act = [](rad_buf x) {
            ActFP8 a{};
            a.x = x;
            return a;
        };
        const auto input = act(w.input);
        qkv.step(ctx, input, w.core.qkv, T);
        beta.step(ctx, input, w.core.beta, T);
        forget_a.step(ctx, input, w.forget_low, T);
        gate_a.step(ctx, input, w.gate_low, T);
        forget_b.step(ctx, act(w.forget_low), w.core.a, T);
        gate_b.step(ctx, act(w.gate_low), w.core.gate, T);
        core.step(ctx, batch);
        output.step(ctx, act(w.core.out), w.out, T);
        if (op_ar) RAD_ISSUE_N(ctx, op_ar, T * 4096, brows(w.out, T), RAD_NONE);
    }
};
}
} // namespace rad::arch
#endif
