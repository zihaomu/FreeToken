/* avx_cases.cpp -- the geometry table both harnesses run (one table, so the bench and the checker
 * cannot drift apart; tools/opshapes.h does the same for rad-kbench and rad-tune).
 *
 * Values are chosen to break hand-vectorised kernels, not for realism (a real n_embd would make the
 * checker take minutes and find nothing a 640-wide row does not):
 *
 *   Rows wider than any vector (16 floats at AVX-512), so the main body runs at every level.
 *   Rows that are not a multiple of it, where tails, masked stores and scalar loops can disagree:
 *     645 % 16 == 5, 645 % 8 == 5, 259 % 16 == 3.
 *   More than one row, to catch a kernel that computes row 0 and broadcasts it.
 *   M = 1, 7 and 65 on the GEMMs: decode, a speculative verify window, and one row past any
 *     register-block edge.
 *   Both sides of every option: `causal`, bounded and unbounded `window`, `norm`, both `a_order`
 *     spellings, both rope `mode`s.
 *
 * Ops run in f32 (tight tolerance: only summation order may differ) and in bf16 (narrowing).
 *
 * `bytes` and `flops` feed the bench's rate columns and stay 0 where the cost is not a simple
 * function of the geometry (in-place samplers, draw-dependent scatters).
 */
#include "avx_harness.h"

std::vector<Case> build_cases() {
    std::vector<Case> c;
    auto add = [&](const char* op, std::vector<P> p, double bytes = 0, double flops = 0) {
        c.push_back(Case{ op, std::move(p), bytes, flops });
    };
    /* Bytes moved by an elementwise op over `rows` x `n`, `w` bytes per element, `k` streams. */
    auto EB = [](double rows, double n, double w, double k) { return rows * n * w * k; };

    for (const char* dt : { "f32", "bf16", "f16" }) {
        const double w = std::strcmp(dt, "f32") == 0 ? 4.0 : 2.0;
        add("rmsnorm",     { HI("M", 7), HI("n", 640), HF("eps", 1e-6), HS("dtype", dt),
                             HF("wadd", 0) }, EB(7, 640, w, 2));
        add("rmsnorm",     { HI("M", 3), HI("n", 645), HF("eps", 1e-5), HS("dtype", dt),
                              HF("wadd", 1) }, EB(3, 645, w, 2));
        add("rmsnorm_p",   { HI("M", 7), HI("n", 640), HF("eps", 1e-6), HS("dtype", dt) },
                              EB(7, 640, w, 2));
        add("rmsnorm_p",   { HI("M", 3), HI("n", 645), HF("eps", 1e-5), HS("dtype", dt) },
                              EB(3, 645, w, 2));
        add("rmsnorm_add", { HI("M", 5), HI("n", 640), HF("eps", 1e-6), HS("dtype", dt),
                             HF("wadd", 0) }, EB(5, 640, w, 4));
        add("layernorm",   { HI("M", 5), HI("n", 645), HF("eps", 1e-5), HS("dtype", dt) },
                           EB(5, 645, w, 2));
        add("add",         { HI("M", 9), HI("n", 645), HS("dtype", dt) }, EB(9, 645, w, 3));
        add("add",         { HI("M", 9), HI("n", 645), HS("dtype", dt), HF("div", 3.0) },
            EB(9, 645, w, 3));
        add("mul",         { HI("M", 9), HI("n", 640), HS("dtype", dt) }, EB(9, 640, w, 3));
        add("silu_mul",    { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 3));
        add("gelu_mul",    { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 3));
        add("gelu",        { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("silu",        { HI("M", 6), HI("n", 640), HS("dtype", dt) }, EB(6, 640, w, 2));
        add("sigmoid",     { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("fill",        { HI("M", 6), HI("n", 645), HS("dtype", dt), HF("value", 1.3) },
            EB(6, 645, w, 1));
        add("logit_softcap", { HI("M", 6), HI("n", 645), HF("cap", 30.0), HS("dtype", dt) },
                              EB(6, 645, w, 2));
        add("scale",         { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("softmax",     { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("gather_rows", { HI("M", 6), HI("n", 640), HS("dtype", dt) }, EB(6, 640, w, 2));
        for (const char* act : { "silu", "gelu", "gelu_tanh", "sigmoid", "quick_gelu", "relu" })
            add("act_mul", { HI("M", 6), HI("n", 645), HS("act", act), HS("dtype", dt) },
                EB(6, 645, w, 3));
        add("mul_weight",  { HI("M", 9), HI("n", 645), HS("dtype", dt) }, EB(9, 645, w, 2));
        add("mul_weight",  { HI("M", 9), HI("n", 645), HS("dtype", dt), HI("scalar", 1) },
            EB(9, 645, w, 2));
        add("seg_mean",    { HI("M", 23), HI("n", 645), HS("dtype", dt) }, EB(23, 645, w, 1));
        add("seg_softmax_pool", { HI("M", 23), HI("n", 645), HS("dtype", dt) }, EB(23, 645, w, 2));
        if (!std::strcmp(dt, "bf16"))
            add("seg_mean_bf16_sum", { HI("M", 23), HI("n", 645), HS("dtype", dt) },
                EB(23, 645, w, 1));
        add("relu",        { HI("M", 7), HI("n", 645), HS("dtype", dt) }, EB(7, 645, w, 2));
        add("clamp",       { HI("M", 7), HI("n", 645), HS("dtype", dt) }, EB(7, 645, w, 2));
        add("rmsnorm",     { HI("M", 5), HI("n", 645), HF("eps", 1e-6), HS("dtype", dt),
                             HF("wadd", 0), HF("mult", 0.5) }, EB(5, 645, w, 2));
        /* The audio subsampler's two geometries, scaled down in time: without and with the
         * frame mask and a segment boundary (cu_div 2: cu counts in units finer than rows). */
        add("conv2d", { HI("M", 24), HI("width", 16), HI("cin", 1), HI("cout", 24), HI("kh", 3),
                        HI("kw", 3), HI("stride", 2), HI("pad", 1), HS("dtype", dt) });
        add("conv2d", { HI("M", 24), HI("width", 8), HI("cin", 20), HI("cout", 12), HI("kh", 3),
                        HI("kw", 3), HI("stride", 2), HI("pad", 1), HI("cu_div", 2),
                        HI("valid_stride", 2), HS("dtype", dt) });
        /* Qwen3-ASR's: a [mel, time] kernel (time last), a bias and gelu after it; and its
         * output projection as a 1 x 16 convolution over the frequency axis. */
        add("conv2d", { HI("M", 24), HI("width", 16), HI("cin", 6), HI("cout", 10), HI("kh", 3),
                        HI("kw", 3), HI("stride", 2), HI("pad", 1), HS("dtype", dt),
                        HI("t_last", 1), HS("act", "gelu") });
        add("conv2d", { HI("M", 8), HI("width", 16), HI("cin", 6), HI("cout", 20), HI("kh", 1),
                        HI("kw", 16), HI("stride", 1), HI("pad", 0), HS("dtype", dt) });
        /* A kernel taller than sixteen rows. */
        add("conv2d_unfold", { HI("M", 40), HI("width", 4), HI("cin", 2), HI("kh", 17),
                               HI("kw", 1), HI("stride", 1), HI("pad", 8), HS("dtype", dt) });
        for (int tl : { 0, 1 })
            add("conv2d_unfold", { HI("M", 24), HI("width", 16), HI("cin", 6), HI("kh", 3),
                                   HI("kw", 3), HI("stride", 2), HI("pad", 1),
                                   HI("cu_div", 2), HI("valid_stride", 2), HI("t_last", tl),
                                   HS("dtype", dt) });
        add("conv1d_dw", { HI("M", 26), HI("n", 645), HI("k", 5), HS("dtype", dt) });
        add("conv1d_dw", { HI("M", 24), HI("n", 64), HI("k", 5), HI("cu_div", 4), HS("dtype", dt) });
        add("rel_sinusoid", { HI("M", 13), HI("n", 1024), HF("max_timescale", 10000.0),
                              HS("dtype", dt) });
        add("pool_rows",   { HI("M", 27), HI("n", 645), HI("k", 9), HF("mult", 27.712812921102035),
                             HS("dtype", dt) }, EB(27, 645, w, 1));
        add("pool_cols",   { HI("M", 27), HI("n", 648), HI("k", 9), HF("mult", 27.712812921102035),
                             HS("dtype", dt) }, EB(27, 648, w, 1));
        add("pool_cols",   { HI("M", 40), HI("n", 256), HI("k", 8), HS("dtype", dt) },
            EB(40, 256, w, 1));
        add("point_fourier", { HI("M", 37), HI("n", 3), HI("freqs", 8), HS("dtype", dt) },
            EB(37, 51, w, 1));
        add("point_fourier", { HI("M", 5), HI("n", 4), HI("freqs", 3), HI("pi", 1),
                               HS("dtype", dt) }, EB(5, 28, w, 1));
        add("scatter_rows", { HI("M", 6), HI("n", 645), HS("dtype", dt) }, EB(6, 645, w, 2));
        add("scale_rows",  { HI("M", 6), HI("n", 645), HS("act", "sigmoid"), HS("dtype", dt) },
                           EB(6, 645, w, 2));
        add("scale_rows",  { HI("M", 6), HI("n", 640), HS("act", "none"), HS("dtype", dt) },
                           EB(6, 640, w, 2));
    }
    for (const char* dt : {"f32", "bf16"})
        add("position_scale",
            {HI("M", 7), HI("n", 259), HI("original", 16384), HF("beta", 0.1), HS("dtype", dt)});
    for (const char* dt : {"f32", "bf16"})
        add("pixtral_rope",
            {HI("M", 7), HI("head_dim", 64), HI("n_head", 2), HF("theta", 10000), HS("dtype", dt)});
    for (const char* dt : {"f32", "bf16"})
        add("ministral_rope", {HI("M", 7), HI("head_dim", 128), HI("n_head", 4), HI("n_head_kv", 1),
                               HF("theta", 1e6), HF("yarn_factor", 16), HF("yarn_low", 8),
                               HF("yarn_high", 24), HF("yarn_magnitude", 1), HS("dtype", dt)});
    for (const char* dt : {"f32", "bf16"})
        add("embed_attn", {HI("M", 7), HI("head_dim", 64), HI("n_head", 4), HI("gqa", 2),
                           HI("causal", 0), HI("n_seq", 2), HS("dtype", dt)});
    /* Every dtype pair: `cast` carries the narrowing conventions and is checked at 1e-6, i.e.
     * exactly. */
    for (const char* from : { "f32", "bf16", "f16", "fp8_e4m3", "fp8_e5m2", "i8" })
        for (const char* to : { "f32", "bf16", "f16", "fp8_e4m3", "fp8_e5m2", "i8" })
            add("cast", { HI("M", 5), HI("n", 259), HS("from", from), HS("to", to) });

    /* The wide residual stream: Qwen4-Exp's gated residual scaled down (hc 4, lowrank 8). `inject`
     * on is the trunk case, off the final mixer; they differ in the operand list. */
    for (const char* dt : { "f32", "bf16" }) {
        add("hc_enter",  { HI("M", 5), HI("n", 128), HI("hc", 4), HS("dtype", dt) });
        add("swiglu_clamped", { HI("M", 5), HI("n", 259), HF("limit", 0.5), HS("dtype", dt) });
        add("swiglu_step", { HI("M", 5), HI("n", 259), HF("limit", 0.5), HS("dtype", dt) });
        if (!std::strcmp(dt, "bf16"))
            add("dsa_pool", { HI("M", 7), HI("n", 128), HI("ratio", 4), HS("dtype", "bf16") });
        if (!std::strcmp(dt, "bf16"))
            add("dsa_query", { HI("M", 7), HI("n", 128), HS("dtype", "bf16") });
        if (!std::strcmp(dt, "bf16"))
            add("gemm_nt_f32", { HI("M", 4), HI("N", 32), HI("K", 129), HS("dtype", "bf16") });
        if (!std::strcmp(dt, "bf16"))
            for (const char* mode : { "absorb", "expand" })
                add("mla_project", { HI("M", 3), HI("heads", 2), HI("head_dim", 17),
                                     HI("latent_dim", 31), HS("mode", mode), HS("dtype", "bf16") });
        if (!std::strcmp(dt, "bf16"))
            add("dsa_score", { HI("M", 3), HI("n", 128), HI("heads", 32), HI("blocks", 17),
                               HI("seqs", 2), HS("dtype", "fp8_e4m3") });
        for (int norm : { 0, 1 })
            add("router_sigmoid_topk",
                { HI("M", 5), HI("n_expert", 288), HI("top_k", 8), HI("norm", norm),
                  HF("route_scale", 2.5), HS("dtype", dt) });
        add("mhc_pre", { HI("M", 3), HI("n", 129), HI("hc", 4), HI("iterations", 20),
                         HF("eps", 1e-5), HF("hc_eps", 1e-6), HS("dtype", dt) });
        add("mhc_pre_partial", { HI("M", 3), HI("n", 129), HI("hc", 4), HI("chunk", 129),
                                 HI("first", 1), HI("chunks", 2), HS("dtype", dt) });
        add("mhc_pre_finish",
            { HI("M", 3), HI("n", 129), HI("hc", 4), HI("iterations", 20), HF("eps", 1e-5),
              HF("hc_eps", 1e-6), HI("chunk", 129), HS("dtype", dt) });
        add("mhc_post", { HI("M", 3), HI("n", 129), HI("hc", 4), HS("dtype", dt) });
        add("mhc_contract", { HI("M", 3), HI("n", 129), HI("hc", 4), HS("dtype", dt) });
        for (int norm : { 0, 1 })
            add("router_sigmoid_group_topk",
                { HI("M", 5), HI("n_expert", 512), HI("top_k", 8), HI("n_group", 8),
                  HI("topk_group", 4), HI("norm", norm), HF("route_scale", 2.5), HS("dtype", dt) });
        add("mla_attn_paged", { HI("M", 5), HI("heads", 3), HI("latent_dim", 64), HI("rope_dim", 16),
                                HI("block_size", 4), HF("scale", 0.0721687836), HI("ctx", 23),
                                HS("dtype", dt) });
        add("mla_attn", { HI("M", 3), HI("latent_dim", 32), HI("heads", 2), HI("blocks", 7),
                          HI("block_size", 4), HF("scale", 0.125), HS("dtype", dt) });
        for (const char* mode : { "prefill", "verify" })
            add("kda_update", { HI("M", 3), HI("heads", 2), HI("head_k", 32), HI("head_v", 32),
                                HF("lower_bound", -5), HF("scale", 0.1767766952966369),
                                HF("l2_eps", 1e-6), HS("mode", mode), HS("dtype", dt) });
        /* Nemotron-H's per-rank shape at four ranks is 32 heads x 64 over 2 groups of 128. */
        for (const char* mode : { "prefill", "verify" })
            add("ssd_update", { HI("M", 3), HI("heads", 4), HI("head_dim", 16), HI("d_state", 32),
                                HI("groups", 2), HS("mode", mode), HS("dtype", dt) });
        add("hc_write",  { HI("M", 5), HI("n", 128), HI("hc", 4), HS("dtype", dt) });
        add("hc_project",{ HI("M", 5), HI("n", 128), HI("hc", 4), HF("scale", 1.0),
                           HS("dtype", dt) });
        /* The model's own width: Qwen3.8-Flash-Next's stream is 2560 wide with hc 4, and a decode
         * step is one row per sequence (with speculation, up to 1 + depth). One case at that
         * geometry is what says whether 41 of these a step is affordable. */
        add("hc_project",{ HI("M", 4), HI("n", 2560), HI("hc", 4), HF("scale", 1.0),
                           HS("dtype", "bf16") });
        add("hc_read",   { HI("M", 5), HI("n", 128), HI("hc", 4), HI("lowrank", 8),
                           HF("eps", 1e-6), HS("dtype", dt), HF("wadd", 1), HI("inject", 1),
                           HI("group", 0) });
        add("hc_read",   { HI("M", 5), HI("n", 128), HI("hc", 4), HI("lowrank", 8),
                           HF("eps", 1e-6), HS("dtype", dt), HF("wadd", 0), HI("inject", 0),
                           HI("group", 0) });
        /* The fp8 form and its Hadamard-rotated variant, 32-wide groups (four per stream). */
        add("hc_read",   { HI("M", 5), HI("n", 128), HI("hc", 4), HI("lowrank", 8),
                           HF("eps", 1e-6), HS("dtype", dt), HF("wadd", 1), HI("inject", 1),
                           HI("group", 32), HI("rotate", 1) });
        add("mtp_enter", { HI("M", 5), HI("n", 128), HI("hc", 4), HF("eps", 1e-6), HS("dtype", dt),
                           HF("wadd", 1), HI("ngroup", 4) });
        add("ple_gate",  { HI("M", 5), HI("n", 128), HI("hc", 4), HF("eps", 1e-6), HS("dtype", dt),
                           HF("wadd", 1), HF("gate_eps", 1e-6) });
        add("ple_conv",  { HI("M", 9), HI("n", 128), HI("width", 4), HI("dilation", 2),
                           HS("dtype", dt) });
        add("causal_conv1d", { HI("M", 9), HI("n", 128), HI("width", 3), HS("dtype", dt) });
    }
    add("ngram_ids", { HI("M", 9), HI("heads", 3), HI("ngram", 4), HI("eos", 2) });

    for (const char* order : { "token", "sorted" })
        add("moe_gemm_q",
            { HI("M", 9), HI("N", 64), HI("K", 128), HI("n_expert", 8), HI("top_k", 2),
              HI("group", 128), HS("a_order", order), HS("dtype", "mxfp4a8") });
    /* One scale per column over the rows: no group. */
    add("quant_cols_fp8", { HI("M", 5), HI("n", 256), HS("dtype", "bf16") }, EB(5, 256, 2, 1.5));
    add("quant_cols_fp8", { HI("M", 7), HI("n", 96), HS("dtype", "f32") }, EB(7, 96, 4, 1.25));

    /* IQ8R's activation: one scale per 256 columns, each 128-wide half rotated on its own. */
    add("had_quant_act_i8g", { HI("M", 5), HI("n", 512), HI("group", 256), HS("dtype", "bf16") });

    /* The packed four-bit weight read back: two's complement at a group of 64, a zero point at
     * 128, one fragment tile of rows each. */
    add("dequant_w4", { HI("N", 16), HI("K", 64), HI("group", 64), HI("twos", 1),
                        HS("dtype", "w4a8") });
    add("dequant_w4", { HI("N", 16), HI("K", 128), HI("group", 128), HI("twos", 0),
                        HS("dtype", "w4a8") });
    /* Quantisation. 128 is the real grid; 64 tests a second block edge. n = 256 is divisible by
     * both, which the had_* forms require (a rotation block may not straddle the row end). */
    for (int group : { 64, 128 }) {
        add("quant_act_i8",       { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") }, EB(5, 256, 2, 1.5));
        add("quant_act_i8",       { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "f32") }, EB(5, 256, 4, 1.25));
        add("quant_act_i8g",      { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") }, EB(5, 256, 2, 1.5));
        add("quant_act_i8g",      { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "f32") }, EB(5, 256, 4, 1.25));
        add("quant_act_fp8",      { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") }, EB(5, 256, 2, 1.5));
        add("had_quant_act_i8g",  { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") });
        add("had_quant_act_i8",   { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") });
        add("dequant",            { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16") });
        add("gated_had_quant_i8", { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16"), HS("act", "silu"), HI("mode", 0) });
        add("rmsnorm_had_quant_i8", { HI("M", 5), HI("n", 256), HF("eps", 1e-6),
                                      HI("group", group), HS("dtype", "bf16"), HF("wadd", 0) });
        add("rmsnorm_quant_fp8",  { HI("M", 5), HI("n", 256), HF("eps", 1e-6),
                                    HI("group", group), HS("dtype", "bf16"), HF("wadd", 0) });
        add("gated_quant_fp8",    { HI("M", 5), HI("n", 256), HI("group", group),
                                    HS("dtype", "bf16"), HS("act", "silu") });
    }
    /* The fp8 fusions off the grid: a row the group does not divide (260 = 4 x 64 + 4, so the last
     * scale covers four values), f32 activations (still narrowed to bf16), and the other gate and
     * gain. */
    add("rmsnorm_quant_fp8", { HI("M", 5), HI("n", 260), HF("eps", 1e-6), HI("group", 64),
                               HS("dtype", "f32"), HF("wadd", 1) });
    add("gated_quant_fp8",   { HI("M", 5), HI("n", 260), HI("group", 64), HS("dtype", "f32"),
                               HS("act", "sigmoid") });
    add("gram_accum", { HI("M", 5), HI("n", 128), HI("group", 128), HS("dtype", "fp8_e4m3") },
        0, 2.0 * 5 * 128 * 128);
    /* The plain bf16 activation, with no scale. */
    add("gram_accum", { HI("M", 5), HI("n", 260), HI("group", 128), HS("dtype", "bf16") },
        0, 2.0 * 5 * 260 * 260);

    /* GEMM. K = 259 is not a multiple of any vector or block width.
     *
     * gemm_nt_bias's activations run at M = 7 only. The product is computed the same way whatever
     * `act` names, and the activation and residual are applied afterwards, element by element, by
     * an epilogue with no M blocking (avx_gemm.cpp: gemm_epilogue), so M = 1 and 65 add nothing
     * for them; those two keep the plain bias form for the product's own M edges. */
    for (const char* dt : { "f32", "bf16", "f16" })
        for (int M : { 1, 7, 65 }) {
            const double fl = 2.0 * M * 96 * 259;
            add("gemm_nt",      { HI("M", M), HI("N", 96), HI("K", 259), HS("dtype", dt) }, 0, fl);
            add("gemm_nt_bias", { HI("M", M), HI("N", 96), HI("K", 259), HS("dtype", dt) }, 0, fl);
            if (M == 7)
                for (const char* act : { "none", "gelu", "gelu_tanh" })
                    add("gemm_nt_bias", { HI("M", M), HI("N", 96), HI("K", 259), HS("dtype", dt),
                                          HS("act", act) }, 0, fl);
            add("logits_gemm",  { HI("M", M), HI("n_vocab", 96), HI("n_embd", 259),
                                  HS("dtype", dt) }, 0, fl);
        }
    /* Encoder-sized M with N not a multiple of the 48-column panel, K odd and even: 37 rows take
     * the bf16 dot-product path, 133 the outer-product one (column edge, zero-paired last k,
     * five-row tail). */
    for (const char* dt : { "f32", "bf16" })
        for (int M : { 37, 133 })
            for (int K : { 259, 64 }) {
                const double fl = 2.0 * M * 100 * K;
                add("gemm_nt",      { HI("M", M), HI("N", 100), HI("K", K), HS("dtype", dt) },
                    0, fl);
                add("gemm_nt_bias", { HI("M", M), HI("N", 100), HI("K", K), HS("dtype", dt),
                                      HS("act", "gelu_tanh") }, 0, fl);
            }
    /* The formats libref describes (ref_shapes.cpp): block fp8, and the int4 and int8 weights
     * against quant_act_i8's step-ordered codes. The bias form needs `a_scale`, which fp8a16 (a
     * bf16 activation) does not have, so libref declines it there. */
    for (const char* dt : { "fp8a8", "fp8a8f32", "fp8a16", "w4a8", "w8a8" })
        for (int M : { 1, 65 }) {
            const double fl = 2.0 * M * 96 * 256;
            add("gemm_nt_q",      { HI("M", M), HI("N", 96), HI("K", 256), HI("group", 128),
                                    HS("dtype", dt) }, 0, fl);
            if (std::strcmp(dt, "fp8a16") != 0)
                add("gemm_nt_q_bias", { HI("M", M), HI("N", 96), HI("K", 256), HI("group", 128),
                                        HS("dtype", dt) }, 0, fl);
        }
    /* The int8 trunk (row-major codes, a scale per row per 128 of K on both sides; i8a8h is the
     * same operands over a rotated weight) and OCP-MXFP4 (e2m1 codes with an E8M0 exponent per 32
     * of K against the E4M3 activation). */
    for (int M : { 1, 65 }) {
        add("gemm_nt_q", { HI("M", M), HI("N", 96), HI("K", 256), HI("group", 128),
                           HS("dtype", "i8a8") }, 0, 2.0 * M * 96 * 256);
        add("gemm_nt_q", { HI("M", M), HI("N", 96), HI("K", 256), HI("group", 128),
                           HS("dtype", "i8a8h") }, 0, 2.0 * M * 96 * 256);
        /* IQ8R's grid: one scale per 256 of K on both sides. */
        add("gemm_nt_q", { HI("M", M), HI("N", 96), HI("K", 512), HI("group", 256),
                           HS("dtype", "i8a8h") }, 0, 2.0 * M * 96 * 512);
        add("gemm_nt_q", { HI("M", M), HI("N", 96), HI("K", 256), HI("group", 32),
                           HS("dtype", "mxfp4a8") }, 0, 2.0 * M * 96 * 256);
    }
    /* Block fp8 off the 128 grid: N = 200 gives the weight's scale plane a second, partial block
     * row (rows 128..199) and K = 320 a partial last group (256..319). */
    for (const char* dt : { "fp8a8", "fp8a16" })
        for (int M : { 1, 65 })
            add("gemm_nt_q", { HI("M", M), HI("N", 200), HI("K", 320), HI("group", 128),
                               HS("dtype", dt) }, 0, 2.0 * M * 200 * 320);
    add("gemm_nt_q_bias", { HI("M", 7), HI("N", 200), HI("K", 320), HI("group", 128),
                            HS("dtype", "fp8a8") }, 0, 2.0 * 7 * 200 * 320);
    /* The tile drivers' edges (avx_gemm_tile.cpp): K = 1104 is two of the bf16 driver's K chunks
     * and three of the f32 one's, the last with a partial group; K = 17408 (the 27B's down) puts
     * 490 rows in two or three super-blocks, the last one short. */
    for (const char* dt : { "fp8a8", "fp8a16", "i8a8", "mxfp4a8", "w4a8" })
        add("gemm_nt_q", { HI("M", 40), HI("N", 100), HI("K", 1104),
                           HI("group", std::strcmp(dt, "mxfp4a8") ? 128 : 32), HS("dtype", dt) },
            0, 2.0 * 40 * 100 * 1104);
    add("gemm_nt_q", { HI("M", 490), HI("N", 20), HI("K", 17408), HI("group", 128),
                       HS("dtype", "fp8a8") }, 0, 2.0 * 490 * 20 * 17408);
    for (const char* dt : { "f32", "bf16" })
        add("gemm_nt", { HI("M", 40), HI("N", 100), HI("K", 1104), HS("dtype", dt) }, 0,
            2.0 * 40 * 100 * 1104);
    add("gemm_nt", { HI("M", 250), HI("N", 20), HI("K", 17408), HS("dtype", "bf16") }, 0,
        2.0 * 250 * 20 * 17408);
    /* The gated form, which libref describes at fp8a8 only. N is the weight's rows, so the output
     * is N/2 = 96 wide. */
    for (int M : { 1, 65 })
        for (const char* act : { "silu", "sigmoid" })
            add("gemm_nt_q_gated", { HI("M", M), HI("N", 192), HI("K", 256), HI("group", 128),
                                     HS("dtype", "fp8a8"), HS("act", act) }, 0,
                2.0 * M * 192 * 256);
    add("gemm_nt_q_gated", { HI("M", 7), HI("N", 512), HI("K", 256), HI("group", 128),
                             HS("dtype", "fp8a8"), HS("act", "silu") }, 0, 2.0 * 7 * 512 * 256);
    /* Runtime LoRA. libref's description draws `slot` in [0, S], so S itself, a base row, comes
     * up in every case of more than one row (a single row draws an adapter, or the case would
     * compare nothing). M = 1 takes the shrink's team-over-columns arm and 9 and 65 its
     * team-over-rows one; K = 1000 leaves a partial scale group and a vector tail at every level;
     * R = 48 is not a power of two. The dtype is `a`'s for shrink (the coded ones with their scale
     * stream) and `y`'s for expand. */
    for (int M : { 1, 9, 65 }) {
        for (const char* dt : { "f32", "bf16", "f16", "fp8_e4m3", "i8" })
            for (int K : { 256, 1000 })
                add("lora_shrink", { HI("M", M), HI("K", K), HI("R", 48), HI("S", 4),
                                     HI("group", 128), HS("dtype", dt) }, 0, 2.0 * M * 48 * K);
        for (const char* dt : { "f32", "bf16", "f16" })
            for (const char* op : { "lora_expand", "lora_expand_mm" })
                add(op, { HI("M", M), HI("N", 645), HI("R", 48), HI("S", 4), HS("dtype", dt) }, 0,
                    2.0 * M * 645 * 48);
    }
    for (const char* dt : { "f32", "bf16" }) {
        add("embed_lookup",   { HI("M", 9), HI("n_embd", 128), HI("n_vocab", 64), HS("dtype", dt),
                                HI("vocab_offset", 0), HF("mult", 1) });
        add("embed_lookup",   { HI("M", 9), HI("n_embd", 128), HI("n_vocab", 64), HS("dtype", dt),
                                HI("vocab_offset", 0), HF("mult", 22.625) });
        add("embed_lookup_q", { HI("M", 9), HI("n_embd", 128), HI("n_vocab", 64), HS("dtype", dt),
                                HI("vocab_offset", 0), HF("wscale", 0.01) });
    }

    /* Attention. q_len 1 is decode, 17 a ragged verify window, 64 the prefill panel form
     * (q_len >= 32, uniform). `window` 0 is unbounded, 8 sliding (a different loop bound). */
    for (const char* dt : { "f32", "bf16" })
        for (int qlen : { 1, 17, 64 })
            for (int window : { 0, 8 })
                add("attn_paged", { HI("q_len", qlen), HI("head_dim", 64), HI("gqa", 4),
                                    HI("block_size", 16), HI("causal", 1), HI("window", window),
                                    HS("q_dtype", dt), HS("kv_dtype", dt), HF("scale", 0.125),
                                    HI("max_ctx", 128), HI("n_head", 8), HI("max_seqs", 2) });
    /* One anchor a sequence (DSpark's drafter): a window of 8 measured from the first query row. */
    for (const char* dt : { "f32", "bf16" })
        for (int qlen : { 1, 17 })
            add("attn_paged_anchor_window", { HI("q_len", qlen), HI("head_dim", 64), HI("gqa", 4),
                                              HI("block_size", 16), HI("causal", 1), HI("window", 8),
                                              HS("q_dtype", dt), HS("kv_dtype", dt), HF("scale", 0.125),
                                              HI("max_ctx", 128), HI("n_head", 8), HI("max_seqs", 2) });
    /* The tree form over the same shapes (its described parents are ordinary rows, -2). */
    for (const char* dt : { "f32", "bf16" })
        for (int qlen : { 1, 17 })
            add("attn_paged_tree", { HI("q_len", qlen), HI("head_dim", 64), HI("gqa", 4),
                                     HI("block_size", 16), HI("causal", 1), HI("window", 8),
                                     HS("q_dtype", dt), HS("kv_dtype", dt), HF("scale", 0.125),
                                     HI("max_ctx", 128), HI("n_head", 8), HI("max_seqs", 2) });
    /* The sink form: a block of 8 non-causal rows (LiLiCorr's query pass), a decode row and the
     * prefill panel form, with and without a window. */
    for (const char* dt : { "f32", "bf16" })
        for (int qlen : { 1, 8, 64 })
            for (int window : { 0, 16 })
                add("attn_paged_sink", { HI("q_len", qlen), HI("head_dim", 64), HI("gqa", 4),
                                         HI("block_size", 16), HI("causal", qlen == 8 ? 0 : 1),
                                         HI("window", window), HS("q_dtype", dt),
                                         HS("kv_dtype", dt), HF("scale", 0.125),
                                         HI("max_ctx", 128), HI("n_head", 8), HI("max_seqs", 2) });
    for (const char* dt : { "f32", "bf16" })
        for (int causal : { 0, 1 })
            add("attn_dense", { HI("M", 24), HI("head_dim", 64), HI("gqa", 2),
                                HI("causal", causal), HS("dtype", dt), HF("scale", 0.125),
                                HI("max_seqlen", 24), HI("window", 0) });
    /* The sliding window both ways (bidirectional |i - j| < 5, causal i - j < 5), over segments
     * longer than the window. */
    for (const char* dt : { "f32", "bf16" })
        for (int causal : { 0, 1 })
            add("attn_dense", { HI("M", 24), HI("head_dim", 64), HI("gqa", 2),
                                HI("causal", causal), HS("dtype", dt), HF("scale", 0.125),
                                HI("max_seqlen", 24), HI("window", 5) });
    /* A non-causal window reaching further right than left (a sliding encoder layer that sees 4
     * keys behind and 7 ahead). */
    for (const char* dt : { "f32", "bf16" })
        add("attn_dense", { HI("M", 24), HI("head_dim", 64), HI("gqa", 2), HI("causal", 0),
                            HS("dtype", dt), HF("scale", 0.125), HI("max_seqlen", 24),
                            HI("window", 5), HI("window_right", 8) });
    /* Blocks: each segment cut into blocks of 7 rows that attend within themselves (a chunked
     * encoder's windows), the last block of a segment short. */
    for (const char* dt : { "f32", "bf16" })
        add("attn_dense", { HI("M", 24), HI("head_dim", 64), HI("gqa", 1), HI("causal", 0),
                            HS("dtype", dt), HF("scale", 0.125), HI("max_seqlen", 24),
                            HI("block", 7) });
    /* An audio encoder's attention: head_dim 256, no GQA, causal within a window. */
    for (const char* dt : { "f32", "bf16" })
        add("attn_dense", { HI("M", 24), HI("head_dim", 256), HI("gqa", 1), HI("causal", 1),
                            HS("dtype", dt), HF("scale", 0.125), HI("max_seqlen", 24),
                            HI("window", 8) });
    /* Cross attention: each entry's keys a run of their own, both sides of `causal`. */
    for (const char* dt : { "f32", "bf16" })
        for (int causal : { 0, 1 })
            add("attn_cross", { HI("M", 24), HI("head_dim", 64), HI("gqa", 2),
                                HI("causal", causal), HS("dtype", dt), HF("scale", 0.125) });
    /* Neighborhood attention: a 4 x 4 x 4 volume (the description's), windows narrower and as wide
     * as its axes. */
    for (const char* dt : { "f32", "bf16" })
        for (int kw : { 3, 4 })
            add("attn_na3d", { HI("M", 64), HI("head_dim", 64), HI("kt", 3), HI("kh", 3),
                               HI("kw", kw), HS("dtype", dt) });
    /* Unlimited-OCR's SAM tower: window 14 on a 16x16 grid (padded to 2x2 windows of 14x14), the
     * same over a 14x21 grid (a partial window in both directions, and the padded rows take part in
     * the softmax), an exact 14x14 multiple, window 7 (21x21 after padding), the global form on a
     * square grid, and the global form on a non-square grid whose rel_pos tables must be resampled
     * along their row axis. Two segments where the case is multi-segment, so a wrong cu walk shows.
     * `n_seq` is not a schema parameter: libref's shape hook sizes cu from it (the harness hint
     * convention, avx_cases.cpp's gdn cases). */
    add("sam_attn", { HI("M", 256), HI("head_dim", 64), HI("n_head", 2), HI("grid_h", 16),
                      HI("grid_w", 16), HI("window", 14), HF("scale", 0.125),
                      HS("dtype", "bf16") });
    add("sam_attn", { HI("M", 588), HI("head_dim", 32), HI("n_head", 2), HI("grid_h", 14),
                      HI("grid_w", 21), HI("window", 14), HF("scale", 0.17677669529663687),
                      HS("dtype", "bf16"), HI("n_seq", 2) });
    add("sam_attn", { HI("M", 196), HI("head_dim", 64), HI("n_head", 2), HI("grid_h", 14),
                      HI("grid_w", 14), HI("window", 14), HF("scale", 0.125),
                      HS("dtype", "bf16") });
    add("sam_attn", { HI("M", 256), HI("head_dim", 64), HI("n_head", 2), HI("grid_h", 16),
                      HI("grid_w", 16), HI("window", 7), HF("scale", 0.125),
                      HS("dtype", "bf16") });
    add("sam_attn", { HI("M", 256), HI("head_dim", 16), HI("n_head", 4), HI("grid_h", 16),
                      HI("grid_w", 16), HI("window", 0), HF("scale", 0.25),
                      HS("dtype", "bf16") });
    add("sam_attn", { HI("M", 240), HI("head_dim", 32), HI("n_head", 2), HI("grid_h", 12),
                      HI("grid_w", 20), HI("window", 0), HF("scale", 0.17677669529663687),
                      HS("dtype", "bf16") });
    for (const char* dt : { "f32", "bf16" })
        add("seg_sinusoid", { HI("M", 30), HI("n", 64), HI("period", 13), HS("dtype", dt) });
    /* Gemma 4's audio attention: left 12, over segments longer than the window. */
    for (const char* dt : { "f32", "bf16" })
        for (int cd : { 1, 4 })
            add("attn_local_rel", { HI("M", 40), HI("head_dim", 32), HI("n_head", 3),
                                    HI("left", 12), HI("right", 0), HF("softcap", 50.0),
                                    HI("cu_div", cd), HS("dtype", dt) });
    /* The store into each cache: the values themselves, the int8 cache's codes and scales, or
     * E4M3 (GLM-5.3's latent attention). */
    for (const char* dt : { "f32", "bf16", "i8", "fp8_e4m3" })
        add("kv_store", { HI("M", 12), HI("head_dim", 64), HI("n_head_kv", 2),
                          HI("block_size", 16), HS("kv_dtype", dt) });
    /* The streamed store, on one block and one ring slot (ref_shapes.cpp says why); the resident and
     * mirror-only split is built by hand in avx_check_hand.cpp. */
    for (const char* dt : { "i8", "bf16", "fp8_e4m3" }) {
        add("kv_store_streamed", { HI("M", 12), HI("head_dim", 64), HI("n_head_kv", 2),
                                   HI("block_size", 32), HS("kv_dtype", dt) });
        add("kv_store_streamed", { HI("M", 3), HI("head_dim", 32), HI("n_head_kv", 1),
                                   HI("block_size", 4), HS("kv_dtype", dt) });
    }
    for (const char* mode : { "neox", "gptj" }) {
        add("rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 2),
                      HF("theta", 10000.0), HF("scale", 1.0), HS("mode", mode),
                      HI("rotary_dim", 64) });
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 2),
                              HF("theta", 10000.0), HF("eps", 1e-6), HF("scale", 1.0),
                              HS("mode", mode), HI("rotary_dim", 64), HF("wadd", 0) });
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 2),
                              HF("theta", 10000.0), HF("eps", 1e-6), HF("scale", 1.0),
                              HS("mode", mode), HI("rotary_dim", 64), HF("wadd", 0),
                              HI("v_norm", 1) });
    }
    /* `blocks`, the two-component vision form: x over the first half, y over the second, with the
     * components listed in the opposite order to the position planes. */
    for (const char* dt : { "f32", "bf16" }) {
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 4),
                              HF("theta", 100.0), HF("eps", 1e-6), HS("mode", "blocks"),
                              HS("sections", "16 16"), HS("comps", "1 0"), HI("v_norm", 1),
                              HS("dtype", dt) });
        /* Qwen-Image's two forms over one position component (libref describes no other):
         * the DiT's adjacent pairs in one section, and Qwen3-VL's interleaved M-RoPE, which then
         * reduces to NeoX pairs. Three components are avx_check_hand.cpp's. */
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 4),
                              HF("theta", 10000.0), HF("eps", 1e-6), HS("mode", "gptj_blocks"),
                              HS("sections", "32"), HS("dtype", dt) });
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 2),
                              HF("theta", 5e6), HF("eps", 1e-6), HS("mode", "imrope"),
                              HI("rotary_dim", 64), HS("sections", "12 10 10"), HS("dtype", dt) });
    }
    add("rope_table", { HI("M", 12), HI("rot", 64), HF("theta", 10000.0), HF("scale", 1.0),
                        HS("dtype", "f32") });
    for (const char* dt : { "f32", "bf16" }) {
        for (const char* mode : { "neox", "gptj", "imrope" })
            add("rope", { HI("M", 12), HI("head_dim", 128), HI("n_head", 4), HI("n_head_kv", 2),
                          HF("theta", 1e7), HF("scale", 1), HS("mode", mode),
                          HI("rotary_dim", 64), HS("sections", "11 11 10"), HS("dtype", dt),
                          HF("yarn_factor", 2), HF("yarn_low", 14), HF("yarn_high", 22),
                          HF("yarn_magnitude", 1.069314718) });
        add("rope_table", { HI("M", 12), HI("rot", 64), HF("theta", 1e7), HF("scale", 1),
                            HS("dtype", dt), HF("yarn_factor", 2), HF("yarn_low", 14),
                            HF("yarn_high", 22), HF("yarn_magnitude", 1.069314718) });
    }
    for (double magnitude : {2.0, (double)NAN}) {
        add("rope", { HI("M", 12), HI("head_dim", 128), HI("n_head", 4), HI("n_head_kv", 2),
                      HF("theta", 1e7), HF("scale", 1), HS("mode", "neox"),
                      HI("rotary_dim", 64), HS("dtype", "bf16"), HF("yarn_factor", 0),
                      HF("yarn_magnitude", magnitude) });
        add("rope_table", { HI("M", 12), HI("rot", 64), HF("theta", 1e7), HF("scale", 1),
                            HS("dtype", "f32"), HF("yarn_factor", 0), HF("yarn_magnitude", magnitude) });
    }
    /* QSA's compressed key at Flash-Next's geometry (n 128, ratio 4, 64-wide rotary), plain and
     * with YaRN; positions are drawn up to block count times ratio. */
    for (const char* dt : { "f32", "bf16" }) {
        add("qsa_block_key", { HI("M", 12), HI("n", 128), HI("ratio", 4), HI("rotary_dim", 64),
                               HF("eps", 1e-6), HF("theta", 1e7), HF("scale", 1), HF("wadd", 1),
                               HS("dtype", dt) });
        add("qsa_block_key", { HI("M", 12), HI("n", 128), HI("ratio", 4), HI("rotary_dim", 64),
                               HF("eps", 1e-6), HF("theta", 1e7), HF("scale", 1), HF("wadd", 0),
                               HS("dtype", dt), HF("yarn_factor", 2), HF("yarn_low", 14),
                                HF("yarn_high", 22), HF("yarn_magnitude", 1.069314718) });
    }
    /* The rest of the indexer at Flash-Next's geometry (4 heads of 128, ratio 4, a 7-row ring for
     * depth 3) with the block count cut to the checker's budget, then the edges: block counts that
     * are no multiple of any vector width, one block, several sequences, more rows than a score
     * tile, a head width that pads, and a topk that keeps every live block or splits the zeros.
     * libref's descriptions draw page ids >= 0, cu from 0 and nc in [0, blocks]; the hand-built
     * cases (avx_check_hand.cpp) cover the rest. */
    for (const char* dt : { "bf16", "f32" }) {
        add("qsa_score", { HI("M", 4), HI("n", 128), HI("heads", 4), HI("blocks", 300),
                           HI("seqs", 1), HS("dtype", dt) });
        add("qsa_tail_store", { HI("M", 5), HI("n", 128), HI("ratio", 4), HI("ring", 7),
                                HI("seqs", 5), HS("dtype", dt) });
        add("qsa_work", { HI("M", 8), HI("n", 128), HI("ratio", 4), HI("ring", 7), HI("seqs", 4),
                          HI("work", 6), HS("dtype", dt) });
    }
    add("qsa_score", { HI("M", 7), HI("n", 128), HI("heads", 4), HI("blocks", 37), HI("seqs", 3),
                       HS("dtype", "bf16") });
    add("qsa_score", { HI("M", 19), HI("n", 72), HI("heads", 3), HI("blocks", 41), HI("seqs", 2),
                       HS("dtype", "f32") });
    add("qsa_score", { HI("M", 1), HI("n", 128), HI("heads", 4), HI("blocks", 1), HI("seqs", 1),
                       HS("dtype", "bf16") });
    add("qsa_select", { HI("M", 4), HI("blocks", 300), HI("topk", 32), HI("ratio", 4),
                        HI("seqs", 1) });
    add("qsa_select", { HI("M", 4), HI("blocks", 300), HI("topk", 200), HI("ratio", 4),
                        HI("seqs", 1) });
    add("qsa_select", { HI("M", 3), HI("blocks", 300), HI("topk", 512), HI("ratio", 4),
                        HI("seqs", 1) });
    add("qsa_select", { HI("M", 7), HI("blocks", 129), HI("topk", 16), HI("ratio", 4),
                        HI("seqs", 3) });
    add("qsa_select", { HI("M", 2), HI("blocks", 1), HI("topk", 1), HI("ratio", 3),
                        HI("seqs", 1) });
    add("qsa_select", { HI("M", 4), HI("blocks", 300), HI("topk", 32), HI("ratio", 4),
                        HI("seqs", 1), HI("signed", 1) });
    add("qsa_select", { HI("M", 7), HI("blocks", 129), HI("topk", 16), HI("ratio", 4),
                        HI("seqs", 3), HI("signed", 1) });
    add("qsa_tail_store", { HI("M", 3), HI("n", 37), HI("ratio", 2), HI("ring", 2), HI("seqs", 3),
                            HS("dtype", "f32") });
    add("qsa_work", { HI("M", 3), HI("n", 40), HI("ratio", 2), HI("ring", 3), HI("seqs", 3),
                      HI("work", 3), HS("dtype", "f32") });
    /* The residency map at a decode step's geometry and at the edges a description can reach: a ring
     * shallower and deeper than the id space, one query row, a pitch of one, and a `block_bytes`
     * that is not a whole vector (the copy's tail). The described map is all-resident, so misses,
     * evictions, second chances and overflow are built by hand in avx_check_hand.cpp. */
    add("kv_resolve", { HI("M", 4), HI("blocks", 300), HI("slots", 96), HI("sel_pitch", 33),
                        HI("block_bytes", 256) });
    add("kv_resolve", { HI("M", 6), HI("blocks", 64), HI("slots", 128), HI("sel_pitch", 17),
                        HI("block_bytes", 24) });
    add("kv_resolve", { HI("M", 1), HI("blocks", 8), HI("slots", 4), HI("sel_pitch", 1),
                        HI("block_bytes", 8) });
    add("kv_resolve", { HI("M", 3), HI("blocks", 1), HI("slots", 1), HI("sel_pitch", 513),
                        HI("block_bytes", 4096) });
    /* The proportional ladder: a partial rotary (rotary_dim < head_dim) with `ladder` head, so the
     * frequency denominator is head_dim. */
    for (const char* mode : { "neox", "gptj" }) {
        add("rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 2),
                      HF("theta", 10000.0), HF("scale", 1.0), HS("mode", mode),
                      HI("rotary_dim", 32), HS("ladder", "head") });
    }
    add("rope_table", { HI("M", 12), HI("rot", 32), HF("theta", 10000.0), HF("scale", 1.0),
                        HS("dtype", "f32"), HI("head_dim", 64), HS("ladder", "head") });
    /* Llama 3's frequency scaling at a depth decoder's settings (theta 5e5, factor 32, original
     * length 16): kept, blended and divided frequencies all present. */
    for (const char* dt : { "f32", "bf16" })
        add("rope", { HI("M", 12), HI("head_dim", 128), HI("n_head", 4), HI("n_head_kv", 2),
                      HF("theta", 500000.0), HF("scale", 1.0), HS("mode", "neox"),
                      HF("llama3_factor", 32.0), HF("llama3_low", 0.001953125),
                      HF("llama3_high", 0.0078125), HF("llama3_orig", 16.0), HS("dtype", dt) });
    /* A rotary wider than 1024 (libref takes up to 2048, i.e. 1024 frequencies), plain and with
     * contiguous sections, and QSA's key at the same width. */
    add("rope", { HI("M", 3), HI("head_dim", 1280), HI("n_head", 1), HI("n_head_kv", 1),
                  HF("theta", 10000.0), HF("scale", 1.0), HS("mode", "neox"),
                  HI("rotary_dim", 1280), HS("dtype", "f32") });
    add("rope", { HI("M", 3), HI("head_dim", 1280), HI("n_head", 1), HI("n_head_kv", 1),
                  HF("theta", 10000.0), HF("scale", 1.0), HS("mode", "mrope"),
                  HI("rotary_dim", 1280), HS("sections", "214 213 213"), HS("dtype", "f32") });
    add("qsa_block_key", { HI("M", 3), HI("n", 1280), HI("ratio", 2), HI("rotary_dim", 1280),
                           HF("eps", 1e-6), HF("theta", 10000.0), HF("scale", 1), HF("wadd", 1),
                           HS("dtype", "f32") });
    /* The multi-component forms: Qwen3-VL's interleaved M-RoPE over a partial rotary, and a vision
     * tower's 2-D axial rotary over a whole 72-wide head. */
    for (const char* dt : { "f32", "bf16" }) {
        add("rope", { HI("M", 12), HI("head_dim", 128), HI("n_head", 4), HI("n_head_kv", 2),
                      HF("theta", 1e7), HF("scale", 1.0), HS("mode", "imrope"),
                      HI("rotary_dim", 64), HS("sections", "11 11 10"), HS("dtype", dt) });
        add("rope", { HI("M", 12), HI("head_dim", 72), HI("n_head", 4), HI("n_head_kv", 4),
                      HF("theta", 10000.0), HF("scale", 1.0), HS("mode", "axial"),
                      HI("rotary_dim", 72), HS("sections", "18 18"), HS("dtype", dt) });
        /* MiniMax-H3's: the DiT's 3-axis rotary over 96 of 128 channels, the video decoder's
         * over 48 of 64; and the norm alone (rotary_dim 0, no positions). f32 positions and
         * absent gains are avx_check_hand.cpp's. */
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 128), HI("n_head", 2), HI("n_head_kv", 2),
                              HF("theta", 10000.0), HF("eps", 1e-5), HS("mode", "axial"),
                              HI("rotary_dim", 96), HS("sections", "16 16 16"), HS("dtype", dt) });
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 4),
                              HF("theta", 100.0), HF("eps", 1e-5), HS("mode", "axial"),
                              HI("rotary_dim", 48), HS("sections", "8 8 8"), HS("dtype", dt) });
        add("qk_norm_rope", { HI("M", 12), HI("head_dim", 64), HI("n_head", 4), HI("n_head_kv", 2),
                              HF("theta", 10000.0), HF("eps", 1e-6), HI("rotary_dim", 0),
                              HI("v_norm", 1), HS("dtype", dt) });
        add("grid_embed", { HI("M", 20), HI("n", 72), HI("side", 6), HS("dtype", dt) });
        /* The zero-padded form: drawn grids never exceed the table, so the off-table masking is
         * covered by avx_check's hand-built case. */
        add("grid_embed_pad", { HI("M", 20), HI("n", 72), HI("side", 6), HS("dtype", dt) });
        /* Two table sides. A drawn grid never exceeds the table, so upscaling is covered by
         * avx_check's hand-built case. */
        add("grid_embed_aa", { HI("M", 20), HI("n", 72), HI("side", 16), HS("dtype", dt) });
        add("grid_embed_aa", { HI("M", 40), HI("n", 72), HI("side", 4), HS("dtype", dt) });
        add("grid_embed_cpe", { HI("M", 20), HI("n", 72), HI("side", 16), HS("dtype", dt) });
        add("grid_embed_cpe", { HI("M", 40), HI("n", 72), HI("side", 4), HS("dtype", dt) });
        add("grid_embed_cpe", { HI("M", 24), HI("n", 72), HI("side", 8), HI("n_prefix", 3),
                                HS("dtype", dt) });
    }

    /* LiLiCorr's lattice ops, at its geometry (7 positions of 8 candidates, 8 heads) narrowed in
     * width; lili_cand's rows are no draw and are avx_check_hand.cpp's. */
    for (const char* dt : { "f32", "bf16" }) {
        add("lili_attn", { HI("M", 112), HI("steps", 7), HI("k", 8), HI("n_head", 8),
                           HI("head_dim", 16), HS("dtype", dt) });
        add("lili_factor", { HI("M", 112), HI("L", 56), HI("n", 48), HS("dtype", dt) });
        add("lili_walk", { HI("M", 3), HI("steps", 7), HI("k", 8), HI("n", 40), HS("dtype", dt),
                           HF("eps", 1e-4), HF("scale", 8.0) });
    }

    /* MoE. Both `a_order` spellings, since the two grouped GEMMs of a routed block use
     * different ones. */
    for (const char* dt : { "f32", "bf16" }) {
        add("router_topk", { HI("M", 9), HI("n_expert", 8), HI("top_k", 2), HI("norm", 1),
                             HS("dtype", dt) });
        add("router_topk", { HI("M", 9), HI("n_expert", 8), HI("top_k", 2), HI("norm", 0),
                             HS("dtype", dt) });
        /* The pair and the pair behind the router GEMM: the whole-model form and a rank's slice
         * with a nonzero expert base, since the sort's slicing is the part the fuse adds. */
        add("router_topk_scatter", { HI("M", 9), HI("n_expert", 8), HI("top_k", 2),
                                     HI("norm", 1), HS("dtype", dt) });
        add("router_topk_scatter", { HI("M", 9), HI("n_expert", 8), HI("n_local", 4),
                                     HI("top_k", 2), HI("norm", 0), HI("expert_base", 2),
                                     HS("dtype", dt) });
        add("router_gemm_topk_scatter", { HI("M", 9), HI("K", 128), HI("n_expert", 8),
                                          HI("top_k", 2), HI("norm", 0), HS("dtype", dt) });
        add("router_gemm_topk_scatter", { HI("M", 9), HI("K", 128), HI("n_expert", 8),
                                          HI("n_local", 4), HI("top_k", 2), HI("norm", 1),
                                          HI("expert_base", 2), HS("dtype", dt) });
        add("moe_gather",  { HI("M", 9), HI("n", 128), HI("top_k", 2), HS("dtype", dt) });
        /* The shared-expert fold: `shared`/`shared_gate` are always described, so always
         * present; this case uses sigmoid, the one above "none". */
        add("moe_gather",  { HI("M", 9), HI("n", 128), HI("top_k", 2), HS("dtype", dt),
                             HS("act", "sigmoid") });
        add("row_topk",    { HI("M", 5), HI("N", 97), HI("R", 4), HI("vocab_off", 0),
                             HS("dtype", dt) });
        add("row_topk_merge", { HI("M", 5), HI("R", 4), HI("world_size", 2), HS("dtype", dt) });
        add("pair_argmax", { HI("M", 5), HI("n", 8), HS("dtype", dt) });
        /* "sorted", not "slot": libref refuses other spellings with RAD_E_UNSUPPORTED. */
        for (const char* order : { "token", "sorted" }) {
            add("moe_gemm",   { HI("M", 9), HI("N", 64), HI("K", 128), HI("n_expert", 8),
                                HI("top_k", 2), HS("a_order", order), HS("dtype", dt) });
            /* `fp8a8` (block fp8): libref describes a grouped GEMM's scale plane only for a
             * dtype that names its grid. */
            add("moe_gemm_q", { HI("M", 9), HI("N", 64), HI("K", 128), HI("n_expert", 8),
                                HI("top_k", 2), HI("group", 128), HS("a_order", order),
                                HS("dtype", "fp8a8") });
            add("moe_gemm_q",
                { HI("M", 9), HI("N", 128), HI("K", 256), HI("n_expert", 8), HI("top_k", 2),
                  HI("group", 128), HS("a_order", order), HS("dtype", "fp8a8f32") });
        }
        /* Two classes at different widths: a gate/up pair whose odd experts are half as wide
         * (the output's second part begins past the odd class's zeros), and a down projection
         * whose odd experts are half as long. */
        add("moe_gemm_q", { HI("M", 9), HI("N", 256), HI("K", 128), HI("N_odd", 128),
                            HI("parts", 2), HI("n_expert", 8), HI("top_k", 2), HI("group", 128),
                            HS("a_order", "token"), HS("dtype", "fp8a8") });
        add("moe_gemm_q", { HI("M", 9), HI("N", 64), HI("K", 256), HI("K_odd", 128),
                            HI("n_expert", 8), HI("top_k", 2), HI("group", 128),
                            HS("a_order", "sorted"), HS("dtype", "fp8a8") });
        /* Four classes at three widths (a four-rank slice of a five-block expert is two blocks of
         * one class and one of each other), over ten experts, so classes 2 and 3 hold one fewer
         * than 0 and 1; for the gate/up pair and the down projection. */
        add("moe_gemm_q_mod4", { HI("M", 9), HI("N", 128), HI("K", 128), HI("N_1", 256),
                                 HI("N_2", 128), HI("N_3", 128), HI("parts", 2),
                                 HI("n_expert", 10), HI("top_k", 2), HI("group", 128),
                                 HS("a_order", "token"), HS("dtype", "fp8a8") });
        add("moe_gemm_q_mod4", { HI("M", 9), HI("N", 64), HI("K", 128), HI("K_1", 128),
                                 HI("K_2", 256), HI("K_3", 128), HI("n_expert", 10),
                                 HI("top_k", 2), HI("group", 128), HS("a_order", "sorted"),
                                 HS("dtype", "fp8a8") });
    }
    add("moe_scatter", { HI("M", 9), HI("n_expert", 8), HI("top_k", 2), HI("expert_base", 0) });
    /* The draft head's rerank: f32 and bf16 activations over an f32 head, the whole vocabulary on
     * this rank and a shard of it (candidates below vocab_off become pads). */
    for (const char* dt : { "f32", "bf16" })
        add("logit_rerank", { HI("M", 5), HI("R", 8), HI("n_vocab", 300), HI("n_embd", 259),
                              HI("vocab_off", 0), HS("dtype", dt) });
    add("logit_rerank", { HI("M", 3), HI("R", 16), HI("n_vocab", 64), HI("n_embd", 128),
                          HI("vocab_off", 100), HS("dtype", "f32") });

    /* Gated delta net. head_k = head_v = 32 and chunk 8 keep the states small enough to inspect
     * while running every loop the 128/128 shape does.
     *
     * `n_head_k`, `n_head_v` and `n_seq` are not op parameters: the gdn ops read head counts from
     * their operands, so libref's shape hook cannot size them cold and declines without these
     * hints (tools/opshapes.h takes the same three, as `ShapeCtx`, from container metadata).
     * Without them every gdn case would be skipped.
     *
     * `n_seq` is 2 so that a wrong `cu` walk differs from a right one. */
    add("gdn_conv_prep", { HI("M", 16), HI("head_k", 32), HI("head_v", 32), HI("chunk", 8),
                           HI("conv_width", 4), HF("l2_eps", 1e-6), HF("softplus_thr", 20.0),
                           HI("n_head_k", 2), HI("n_head_v", 2), HI("n_seq", 2) });
    add("gdn_conv_update", { HI("q_len", 2), HI("head_k", 32), HI("head_v", 32),
                             HI("conv_width", 4), HI("max_query_len", 2),
                             HI("n_head_k", 2), HI("n_head_v", 2), HI("n_seq", 2) });
    add("gdn_kkt_solve", { HI("M", 16), HI("head_k", 32), HI("chunk", 8),
                           HI("n_head_k", 2), HI("n_head_v", 2), HI("n_seq", 2) });
    add("gdn_chunk_scan", { HI("M", 16), HI("head_k", 32), HI("head_v", 32), HI("chunk", 8),
                            HF("scale", 0.176), HI("state_fp16", 0),
                            HI("n_head_k", 2), HI("n_head_v", 2), HI("n_seq", 2) });
    add("gdn_recurrent_update", { HI("q_len", 2), HI("head_k", 32), HI("head_v", 32),
                                  HI("n_head_v", 2), HF("scale", 0.176), HF("eps", 1e-6),
                                  HF("l2_eps", 1e-6), HF("softplus_thr", 20.0), HS("act", "silu"),
                                  HI("state_fp16", 0), HI("n_head_k", 2), HI("n_seq", 2) });
    add("gdn_conv_recurrent_update", { HI("q_len", 2), HI("head_k", 32), HI("head_v", 32),
                                       HI("conv_width", 4), HI("n_head_v", 2), HF("scale", 0.176),
                                       HF("eps", 1e-6), HF("l2_eps", 1e-6),
                                       HF("softplus_thr", 20.0), HS("act", "silu"),
                                       HI("state_fp16", 0), HI("n_head_k", 2), HI("n_seq", 2) });
    for (const char* act : { "silu", "sigmoid" })
        add("gdn_gated_rmsnorm", { HI("M", 12), HI("channels", 128), HF("eps", 1e-6),
                                   HS("act", act) });

    /* Collectives at world_size 1, the only width libavx serves. */
    add("all_reduce", { HI("world_size", 1), HI("numel", 645), HS("dtype", "f32"), HI("exact", 1),
                        HI("min_bytes", 0), HI("hops", 1) });
    add("all_gather", { HI("world_size", 1), HI("numel", 645), HS("dtype", "f32"), HI("row", 0) });

    /* Sampling. n_vocab 517 exceeds every vector width and is not a multiple of any; n_cand 12 is
     * a realistic candidate set after top-k. */
    for (const char* op : { "sample_penalties", "sample_dry", "sample_ngram_ban", "sample_temp",
                            "sample_mask", "sample_argmax" })
        add(op, { HI("M", 5), HI("n_vocab", 517) });
    add("sample_topk", { HI("M", 5), HI("n_vocab", 517), HI("vocab_off", 0) });
    /* k 7 and 20 (OpenAI's top_logprobs bound) against 517 columns; the offset case checks that
     * a shard reports global ids. */
    for (int64_t k : { 7, 20 })
        add("sample_logprobs", { HI("M", 5), HI("n_vocab", 517), HI("k", k), HI("vocab_off", 0) });
    add("sample_logprobs", { HI("M", 3), HI("n_vocab", 517), HI("k", 5), HI("vocab_off", 1000) });
    /* sample_merge_topk has no table case: libref cannot describe its packed (id, logit) words
     * (ref_shapes.cpp), so avx_check_hand.cpp builds them. */
    for (const char* op : { "sample_topp", "sample_minp", "sample_typical", "sample_xtc",
                            "sample_pick" })
        add(op, { HI("M", 5), HI("n_cand", 12) });

    /* The drafter's walk: every step after the first reads its predecessor from the walk. Rank 64
     * is whole vectors at every level, 37 is not; anchor stride 2 reads every other anchor. */
    add("dflash_select", { HI("M", 3), HI("steps", 4), HI("top_k", 8), HI("rank", 64),
                           HI("n_vocab", 300), HI("anchor_stride", 1), HS("dtype", "bf16") });
    add("dflash_select", { HI("M", 5), HI("steps", 3), HI("top_k", 12), HI("rank", 37),
                           HI("n_vocab", 97), HI("anchor_stride", 2), HS("dtype", "bf16") });
    /* DSpark's full-vocabulary walk: a ragged last tile (V % 64), rank 37 not whole vectors. */
    for (const char* op : { "dflash_markov_greedy", "dflash_markov_greedy_f32" }) {
        add(op, { HI("M", 3), HI("steps", 4), HI("rank", 64), HI("n_vocab", 300),
                  HI("anchor_stride", 1), HS("dtype", "bf16") });
        add(op, { HI("M", 2), HI("steps", 3), HI("rank", 37), HI("n_vocab", 97),
                  HI("anchor_stride", 2), HS("dtype", "bf16") });
    }
    /* The same lattices grown into trees: more nodes than steps, and fewer. */
    add("dflash_tree", { HI("M", 3), HI("steps", 4), HI("top_k", 8), HI("rank", 64),
                         HI("n_vocab", 300), HI("anchor_stride", 1), HI("nodes", 6),
                         HS("dtype", "bf16") });
    add("dflash_tree", { HI("M", 5), HI("steps", 3), HI("top_k", 12), HI("rank", 37),
                         HI("n_vocab", 97), HI("anchor_stride", 2), HI("nodes", 2),
                         HS("dtype", "bf16") });
    /* DFlash2's convolution: the 27B drafter's form (2 taps, groups of 16) over a narrower row,
     * then groups that are no multiple of a vector, three taps and a one-row block. f16 rows take
     * the path through f32 copies, bf16 and f32 the fused one. */
    for (const char* dt : { "bf16", "f32" })
        add("dflash_conv", { HI("T", 16), HI("hidden_size", 640), HI("taps", 2),
                             HI("group_size", 16), HI("NG", 40), HI("block_size", 4),
                             HS("dtype", dt) });
    add("dflash_conv", { HI("T", 11), HI("hidden_size", 36), HI("taps", 3), HI("group_size", 4),
                         HI("NG", 9), HI("block_size", 5), HS("dtype", "f32") });
    add("dflash_conv", { HI("T", 6), HI("hidden_size", 48), HI("taps", 2), HI("group_size", 48),
                         HI("NG", 1), HI("block_size", 1), HS("dtype", "bf16") });
    add("dflash_conv", { HI("T", 11), HI("hidden_size", 36), HI("taps", 3), HI("group_size", 4),
                         HI("NG", 9), HI("block_size", 5), HS("dtype", "f16") });

    /* Image generation (Qwen-Image 2.1, arch/qwenimage21), scaled down: the DiT's norms over rows
     * that are not a multiple of any vector, the latent's 64 channels in lanes of 16 rows over two
     * entries, the decoder's convolutions and shortcuts on the eight-column pictures libref picks
     * (a channel count below and above the convolution's 48-column panel, every padding and
     * resampling form, and 256 pixels for the bf16 outer-product path), and the DiT's attention:
     * 40 rows (one query block and part of a second) against a 37-key prefix (two whole blocks
     * and part of a third), both score forms. Several entries, missing blocks and column slices
     * are avx_check_hand.cpp's. */
    static const std::string kMul = [] {
        std::string s;
        for (int i = 0; i < 67; ++i) s += (i ? " " : "") + std::to_string(0.9 + 0.013 * i);
        return s;
    }();
    static const std::string kAdd = [] {
        std::string s;
        for (int i = 0; i < 67; ++i) s += (i ? " " : "") + std::to_string(0.017 * i - 0.5);
        return s;
    }();
    for (const char* dt : { "f32", "bf16" }) {
        add("ln_mod",   { HI("M", 7), HI("n", 645), HF("eps", 1e-6), HS("dtype", dt) },
            EB(7, 645, dt[0] == 'f' ? 4 : 2, 2));
        add("gate_add", { HI("M", 7), HI("n", 645), HS("dtype", dt) },
            EB(7, 645, dt[0] == 'f' ? 4 : 2, 3));
        add("gate_add", { HI("M", 7), HI("n", 645), HS("dtype", dt), HS("act", "none") },
            EB(7, 645, dt[0] == 'f' ? 4 : 2, 3));
        add("table_add", { HI("M", 3), HI("n", 645), HI("L", 5), HS("dtype", dt) });
        add("mix_rows", { HI("M", 7), HI("n", 645), HI("k", 12), HS("dtype", dt) });
        for (const char* mode : { "denorm", "norm" })
            for (int var : { 0, 1 })
                add("chan_stats", { HI("M", 7), HI("n", 645), HS("mode", mode), HI("var", var),
                                    HF("eps", 1e-4), HS("dtype", dt) });
        for (int round : { 3, 2, 1, 0 })
            add("timestep_embed", { HI("M", 3), HI("n", 256), HF("scale", 1000.0),
                                    HF("max_period", 10000.0), HI("round", round),
                                    HS("dtype", dt) });
        for (int g : { 5, 1025 })
            add("timestep_embed", { HI("M", 3), HI("n", 8), HI("curve", g), HS("dtype", dt) });
        add("lane_gather", { HI("M", 24), HI("n", 64), HI("stride", 16), HI("n_seq", 2),
                             HS("dtype", dt) });
        add("lane_store",  { HI("M", 12), HI("n", 64), HI("stride", 16), HS("dtype", dt) });
        add("randn_lanes", { HI("M", 24), HI("n", 64), HI("stride", 16), HI("n_seq", 2),
                             HS("dtype", dt) });
        add("flow_euler",  { HI("M", 24), HI("n", 64), HI("stride", 16), HS("dtype", dt) });
        add("kd_input", { HI("M", 24), HI("n", 4), HI("stride", 16), HS("dtype", dt) });
        for (const char* pred : { "eps", "v" })
            add("kd_step", { HI("M", 24), HI("n", 4), HI("stride", 16), HS("pred", pred),
                             HS("dtype", dt) });
        for (const char* mode : { "fma", "inv" })
            for (int round : { 1, 0 })
                add("chan_affine", { HI("M", 9), HI("n", 67), HS("mul", kMul.c_str()),
                                     HS("add", kAdd.c_str()), HS("mode", mode), HS("dtype", dt),
                                     HI("round", round) });
        for (const char* act : { "silu", "none" })
            add("l2norm_chan", { HI("M", 9), HI("n", 645), HS("act", act), HS("dtype", dt) });
        add("to_rgba8",  { HI("M", 9), HI("n", 4), HS("dtype", dt) });
        add("gelu_tanh", { HI("M", 6), HI("n", 645), HS("dtype", dt) });
        add("gelu_quick", { HI("M", 6), HI("n", 645), HS("dtype", dt) });
        add("scatter_add_rows", { HI("M", 6), HI("n", 645), HS("dtype", dt) });
        struct Conv { int M, cin, cout, k, stride, up; const char* pad; };
        static const Conv kConv[] = {
            { 64, 20, 24, 3, 1, 1, "same" }, { 64, 20, 52, 3, 1, 1, "same" },
            { 64, 4, 52, 3, 1, 1, "same" },  { 64, 20, 52, 1, 1, 1, "same" },
            { 64, 20, 12, 1, 1, 1, "same" }, { 64, 20, 52, 3, 2, 1, "down" },
            { 64, 20, 52, 3, 1, 2, "same" }, { 64, 20, 4, 3, 1, 1, "same" },
            { 256, 20, 52, 3, 1, 1, "same" },
        };
        for (const Conv& v : kConv)
            add("conv2d_nhwc", { HI("M", v.M), HI("cin", v.cin), HI("cout", v.cout), HI("k", v.k),
                                 HI("stride", v.stride), HS("pad", v.pad), HI("up", v.up),
                                 HS("dtype", dt) },
                0, 2.0 * v.M * v.cout * v.cin * v.k * v.k);
        /* The convolutions' geometries once each: cout is not a parameter of the unfold. */
        for (const Conv& v : kConv) {
            bool seen = false;
            for (const Conv* u = kConv; u != &v; ++u)
                seen = seen || (u->M == v.M && u->cin == v.cin && u->k == v.k &&
                                u->stride == v.stride && u->up == v.up && u->pad == v.pad);
            if (seen) continue;
            add("im2col_nhwc", { HI("M", v.M), HI("cin", v.cin), HI("k", v.k),
                                 HI("stride", v.stride), HS("pad", v.pad), HI("up", v.up),
                                 HS("dtype", dt) });
        }
        for (int ft : { 1, 2 }) {
            add("d2s_add",     { HI("M", 64), HI("cin", 24), HI("cout", 24), HI("ft", ft),
                                 HS("dtype", dt) });
            add("d2s_add",     { HI("M", 64), HI("cin", 48), HI("cout", 24), HI("ft", ft),
                                 HS("dtype", dt) });
            add("s2d_avg_add", { HI("M", 64), HI("cin", 24), HI("cout", 24), HI("ft", ft),
                                 HS("dtype", dt) });
            add("s2d_avg_add", { HI("M", 64), HI("cin", 12), HI("cout", 24), HI("ft", ft),
                                 HS("dtype", dt) });
        }
        for (const char* qk : { "bf16", "i8" })
            for (int causal : { 0, 1 })
                for (int hd : { 64, 72 })
                    add("attn_prefix_self", { HI("q_len", 40), HI("head_dim", hd), HI("gqa", 2),
                                              HI("block_size", 16), HI("causal", causal),
                                              HS("q_dtype", dt), HS("kv_dtype", dt),
                                              HF("scale", 1.0 / std::sqrt((double)hd)),
                                              HI("n_head", 4), HS("qk", qk) });
    }

    /* ================================================================== video generation
     *
     * MiniMax-H3's DiT row ops at an odd width, both gated forms; its sampler step over an f32
     * state; the decoder's patch unfold, tiles, stitch and positions over the cube geometry
     * libref describes (one value in every `geom` entry). Uneven grids, a chunk between two
     * others and views of wider planes are avx_check_hand.cpp's. */
    static const std::string kStd24 = [] {
        std::string s;
        for (int i = 0; i < 24; ++i) s += (i ? " " : "") + std::to_string(0.4 + 0.07 * i);
        return s;
    }();
    static const std::string kMean24 = [] {
        std::string s;
        for (int i = 0; i < 24; ++i) s += (i ? " " : "") + std::to_string(0.031 * i - 0.37);
        return s;
    }();
    for (const char* dt : { "f32", "bf16" }) {
        for (int gated : { 1, 0 }) {
            add("rms_mod", { HI("M", 7), HI("n", 645), HF("eps", 1e-5), HI("gated", gated),
                             HS("dtype", dt) });
            /* A zero-centred gain (1 + w), the norm and gain rounded once (Krea 2's). */
            add("rms_mod", { HI("M", 7), HI("n", 645), HF("eps", 1e-5), HI("gated", gated),
                             HS("dtype", dt), HF("wadd", 1.0), HI("norm_once", 1) });
            /* A LayerNorm without a gain (FLUX's). */
            add("rms_mod", { HI("M", 7), HI("n", 645), HF("eps", 1e-6), HI("gated", gated),
                             HS("dtype", dt), HS("norm", "layer") });
        }
        add("flow_x0_step", { HI("M", 9), HI("n", 645), HS("dtype", dt) });
        for (int pt : { 1, 4 })
            add("patch_unfold", { HI("M", 8 * 4 * pt), HI("C", 24), HI("pt", pt), HI("ph", 2),
                                  HI("pw", 2), HS("mul", kStd24.c_str()),
                                  HS("add", kMean24.c_str()), HS("dtype", dt) });
        /* Three frames of 3 x 3: two tiles of 2 a side overlapping by 1, two register rows and
         * one zero row a tile. */
        add("vtile_gather", { HI("M", 60), HI("n", 37), HI("tile", 2), HI("overlap", 1),
                              HI("frames", 3), HI("period", 15), HI("n_reg", 2), HS("dtype", dt) });
    }
    /* A 3 x 3 latent of 3 frames, 2 x 2 tiles a row apart past their voxels, decoded at scale 2
     * and two clip frames a latent one: the kept frames and the overlap of a video's only
     * chunk. */
    add("vtile_stitch", { HI("M", 108), HI("tile", 2), HI("overlap", 1), HI("scale", 2),
                          HI("pt", 2), HI("skip", 1), HI("keep", 2), HI("blend", 1),
                          HI("period", 13), HS("mean", "0.485 0.456 0.406"),
                          HS("std", "0.229 0.224 0.225") });
    add("grid_pos3", { HI("M", 40), HI("T", 2), HI("H", 3), HI("W", 4), HI("period", 27),
                       HF("scale", 6.283185307179586) });

    /* The tiled encoder over the shape functions' 20 x 20 latent plane: the encoder's tile 16
     * with its 4 overlap (two tiles an axis), and a 3-overlap-1 split (ten tiles an axis, one of
     * slack round-robin) that puts a live blend on every interior row and column. crop is drawn
     * at ratio 1 -- the plane is 20 x 20 pixels, its tiles' sum 30 x 30 either way; join reads
     * the moments' 30 x 30 rows into the 400-row canvas. */
    add("vtile_plan", { HI("M", 4), HI("tile", 16), HI("overlap", 4), HI("ratio", 16) });
    add("vtile_plan", { HI("M", 100), HI("tile", 3), HI("overlap", 1), HI("ratio", 1) });
    add("vtile_crop", { HI("M", 1024), HI("tile", 16), HI("overlap", 4), HI("ratio", 1),
                        HS("dtype", "f32") });
    add("vtile_crop", { HI("M", 900), HI("tile", 3), HI("overlap", 1), HI("ratio", 1),
                        HS("dtype", "bf16") });
    add("vtile_join", { HI("M", 400), HI("tile", 16), HI("overlap", 4), HI("n", 48),
                        HS("dtype", "bf16") });
    add("vtile_join", { HI("M", 400), HI("tile", 3), HI("overlap", 1), HI("n", 5),
                        HS("dtype", "f32") });

    /* The latent upscaler over libref's cube volumes (c x c x c, the largest M holds; every geom
     * entry c, so the resize is the identity here): the 3-D convolution at odd channel counts,
     * kernels 3 x 3 x 3 and 3 x 1 x 1, with the residual (a separate buffer), and at MiniMax-H3's
     * widths 24 -> 512, 512 -> 512 and 512 -> 24 in bf16; the modulated group norm over a few rows
     * and over three of avx's 64-row blocks; the temporal convolution with taps past both ends.
     * Real resizes, an in-place residual, an unmodulated norm over y = x, uneven volumes and views
     * of wider planes are avx_check_hand.cpp's. */
    for (const char* dt : { "f32", "bf16" }) {
        struct Conv3 { int M, cin, cout, k, kt; };
        static const Conv3 kConv3[] = { { 27, 5, 13, 3, 3 }, { 64, 20, 52, 3, 3 },
                                        { 27, 24, 16, 1, 3 } };
        for (const Conv3& v : kConv3)
            add("conv3d_ndhwc", { HI("M", v.M), HI("cin", v.cin), HI("cout", v.cout),
                                  HI("k", v.k), HI("kt", v.kt), HS("dtype", dt) },
                0, 2.0 * v.M * v.cout * v.cin * v.k * v.k * v.kt);
        for (const char* act : { "silu", "none" })
            add("group_norm", { HI("M", 7), HI("n", 640), HI("groups", 32), HF("eps", 1e-5),
                                HS("act", act), HS("dtype", dt) });
        add("group_norm", { HI("M", 150), HI("n", 96), HI("groups", 4), HF("eps", 1e-5),
                            HS("act", "silu"), HS("dtype", dt) });
        /* The video encoder's first level: four channels a group, eps 1e-6. */
        add("group_norm", { HI("M", 130), HI("n", 128), HI("groups", 32), HF("eps", 1e-6),
                            HS("act", "silu"), HS("dtype", dt) });
        add("tconv_dw", { HI("M", 27), HI("n", 645), HI("k", 5), HS("dtype", dt) });
        add("tconv_dw", { HI("M", 64), HI("n", 37), HI("k", 3), HS("dtype", dt) });
        add("resize_bilinear", { HI("M", 27), HI("n", 645), HS("dtype", dt) });
        for (int pt : { 1, 4 })
            add("patch_fold", { HI("M", 8), HI("C", 24), HI("pt", pt), HI("ph", 2), HI("pw", 2),
                                HS("mul", kStd24.c_str()), HS("add", kMean24.c_str()),
                                HS("dtype", dt) });
    }
    struct Wide { int cin, cout; };
    for (const Wide v : { Wide{ 24, 512 }, Wide{ 512, 512 }, Wide{ 512, 24 } })
        add("conv3d_ndhwc", { HI("M", 27), HI("cin", v.cin), HI("cout", v.cout), HI("k", 3),
                              HI("kt", 3), HS("dtype", "bf16") },
            0, 2.0 * 27 * v.cin * v.cout * 27);
    add("flow_renoise", { HI("M", 9), HI("n", 96), HI("stream", 1), HS("dtype", "f32") });
    add("flow_renoise", { HI("M", 5), HI("n", 645), HI("stream", 7), HS("dtype", "f32") });
    add("cond_aug", { HI("M", 9), HI("n", 96), HI("stream", 2), HF("aug", 0.999),
                      HS("dtype", "f32") });
    add("cond_aug", { HI("M", 5), HI("n", 645), HI("stream", 7), HF("aug", 0.5),
                      HS("dtype", "f32") });

    /* LTX-2: the video's 3-axis rotary (two padded pairs, a rank's second head), the cross
     * attentions' 1-axis one, the per-head gate and the guided step. */
    for (const char* dt : { "f32", "bf16" }) {
        add("rope_split", { HI("M", 7), HI("n_head", 3), HI("head_dim", 128), HI("dim", 4096),
                            HI("n_dims", 3), HF("theta", 10000.0), HS("max", "20 2048 2048"),
                            HI("head0", 1), HS("dtype", dt) });
        add("rope_split", { HI("M", 5), HI("n_head", 4), HI("head_dim", 64), HI("dim", 256),
                            HI("n_dims", 1), HF("theta", 10000.0), HS("max", "20"),
                            HS("dtype", dt) });
        add("head_gate", { HI("M", 7), HI("n_head", 5), HI("head_dim", 64), HS("dtype", dt) });
        add("ltx_guide", { HI("M", 9), HI("n", 128), HS("dtype", dt) });
        add("pixel_norm", { HI("M", 26), HI("n", 96), HF("eps", 1e-6), HS("form", "rsqrt"),
                            HI("ileave", 13), HF("mult", 3.7), HS("dtype", dt) });
        add("pixel_norm", { HI("M", 7), HI("n", 256), HF("eps", 1e-8), HS("form", "div"),
                            HS("act", "silu"), HS("dtype", dt) });
        add("d2s3d", { HI("M", 27), HI("C", 5), HI("st", 2), HI("sh", 2), HI("sw", 2),
                       HS("dtype", dt) });
        add("s2d3d", { HI("M", 27), HI("C", 4), HI("st", 2), HI("sh", 1), HI("sw", 1),
                       HI("tpre", 1), HS("mode", "copy"), HS("dtype", dt) });
        add("s2d3d", { HI("M", 64), HI("C", 6), HI("st", 2), HI("sh", 2), HI("sw", 2),
                       HI("group", 4), HS("mode", "mean"), HI("add", 1), HS("dtype", dt) });
        add("pix_patch", { HI("M", 12), HI("p", 4), HI("C", 3), HS("dtype", dt) });
        add("stft_logmel", { HI("M", 6), HI("hop", 80), HI("nfft", 512), HI("n_mel", 64),
                             HI("ch", 2), HS("dtype", dt) });
        add("bwe_up", { HI("M", 600), HI("ratio", 3), HI("ch", 2), HS("dtype", dt) });
        add("tile_crop", { HI("M", 125), HI("C", 7), HS("dtype", dt) });
        add("tile_stitch", { HI("M", 4000), HI("tile", 4), HI("stride", 3), HI("scale", 2),
                             HI("p", 4), HI("C", 3), HS("dtype", dt) });
    }

    /* Video sparse attention over the cubes libref describes: a 3 x 3 x 3 grid in boxes of 2
     * after two dense segments of as many rows (P 8, V 8; sparsity 0.8 keeps 2), with and without
     * the coarse operands; and MiniMax-H3's boxes of 4 at patch 2 over a 10 x 5 x 5 grid after
     * one segment (P 4, V 12). attn_vsa at k = 1, the one selection a description draws; the
     * others are avx_check_hand.cpp's. `n_head`, `coarse` and `dtype` are the description's. */
    for (const char* dt : { "f32", "bf16" }) {
        for (int coarse : { 1, 0 }) {
            add("vsa_select", { HI("M", 81), HI("head_dim", 64), HI("tile_t", 2), HI("tile_h", 2),
                                HI("tile_w", 2), HI("patch", 1), HF("sparsity", 0.8),
                                HI("n_head", 2), HI("coarse", coarse), HS("dtype", dt) });
            add("attn_vsa", { HI("M", 81), HI("head_dim", 64), HI("tile_t", 2), HI("tile_h", 2),
                              HI("tile_w", 2), HI("patch", 1), HF("sparsity", 0.9),
                              HI("n_head", 2), HI("coarse", coarse), HS("dtype", dt) });
        }
        for (double sp : { 0.5, 0.0 })
            add("vsa_select", { HI("M", 500), HI("head_dim", 128), HI("tile_t", 4), HI("tile_h", 4),
                                HI("tile_w", 4), HI("patch", 2), HF("sparsity", sp),
                                HI("n_head", 2), HS("dtype", dt) });
        add("attn_vsa", { HI("M", 500), HI("head_dim", 128), HI("tile_t", 4), HI("tile_h", 4),
                          HI("tile_w", 4), HI("patch", 2), HF("sparsity", 0.95), HI("n_head", 2),
                          HS("dtype", dt) });
    }

    /* ================================================================== audio generation
     *
     * The vocoder's convolutions over two sequences (a stereo latent's halves as libref describes
     * them), odd channel counts, the codec's kernel widths and dilations and every stride an
     * upsampling block uses; the sampler at both branch counts, with and without the conditional
     * candidate cut; the window ops over three and four entries, so the drawn flags reach FIRST,
     * JOIN and LAST in several combinations. The first case of each op stays inside libref's four
     * lanes at twice its rows, which the zero-row check describes. */
    for (const char* dt : { "f32", "bf16", "f16" }) {
        for (int dil : { 1, 3 })
            for (const char* act : { "none", "tanh" })
                add("conv1d", { HI("M", 24), HI("cin", 20), HI("cout", 13), HI("k", 7),
                                HI("dil", dil), HS("act", act), HS("dtype", dt) });
        add("conv1d", { HI("M", 24), HI("cin", 20), HI("cout", 13), HI("k", 3), HI("dil", 9),
                        HS("dtype", dt) });
        add("conv1d", { HI("M", 9), HI("cin", 33), HI("cout", 17), HI("k", 1), HI("dil", 1),
                        HS("dtype", dt) });
        /* Causal: all of the padding on the left (odd k; an even k is a hand-built case, which
         * libref's description does not draw). */
        for (int dil : { 1, 3 })
            add("conv1d", { HI("M", 24), HI("cin", 20), HI("cout", 13), HI("k", 7),
                            HI("dil", dil), HI("causal", 1), HS("dtype", dt) });
        /* The zero-padded strided form, a DAC encoder block's downsample: kernel 2 stride, pad
         * ceil(stride / 2), one sequence. The form is bf16 only. */
        if (std::strcmp(dt, "bf16") == 0)
            for (int s : { 2, 4, 5 })
                add("conv1d", { HI("M", 24), HI("cin", 20), HI("cout", 13), HI("k", 2 * s),
                                HI("dil", 1), HI("stride", s), HI("pad", (s + 1) / 2),
                                HS("dtype", dt) });
        /* Strided, causal: Fish Audio's codec encoder downsamples with kernels of two strides
         * (2, 4, 8) and its quantiser with a kernel of one (k 2, stride 2). */
        for (int s : { 2, 4, 8 })
            add("conv1d", { HI("M", 64), HI("cin", 20), HI("cout", 13), HI("k", 2 * s),
                            HI("dil", 1), HI("causal", 1), HI("stride", s), HS("dtype", dt) });
        add("conv1d", { HI("M", 64), HI("cin", 24), HI("cout", 16), HI("k", 2), HI("dil", 1),
                        HI("causal", 1), HI("stride", 2), HS("dtype", dt) });
        for (int s : { 2, 4, 8 })
            add("conv_t1d", { HI("M", 12), HI("cin", 20), HI("cout", 13), HI("stride", s),
                              HS("dtype", dt) });
        /* The causal transposed form, a codec's upsampler: no padding, a sequence's first
         * output rows reading its first input row alone. */
        for (int s : { 2, 4, 8 })
            add("conv_t1d", { HI("M", 12), HI("cin", 20), HI("cout", 13), HI("stride", s),
                              HI("pad", 0), HS("dtype", dt) });
        /* A kernel of one stride (one tap a phase), without padding and with some. An odd stride
         * is a hand-built case, which libref's description does not draw. */
        add("conv_t1d", { HI("M", 12), HI("cin", 20), HI("cout", 13), HI("stride", 2),
                          HI("k", 2), HS("dtype", dt) });
        add("conv_t1d", { HI("M", 12), HI("cin", 20), HI("cout", 13), HI("stride", 4),
                          HI("k", 4), HI("pad", 1), HS("dtype", dt) });
        /* BigVGAN's upsamplers: kernel 9 at stride 5 (three taps in some phases, two in
         * others), and a kernel of six strides whose padding passes a stride. */
        add("conv_t1d", { HI("M", 12), HI("cin", 20), HI("cout", 13), HI("stride", 5),
                          HI("k", 9), HS("dtype", dt) });
        add("conv_t1d", { HI("M", 12), HI("cin", 20), HI("cout", 13), HI("stride", 2),
                          HI("k", 12), HS("dtype", dt) });
        add("conv1d", { HI("M", 24), HI("cin", 20), HI("cout", 13), HI("k", 7), HI("dil", 1),
                        HS("act", "clamp"), HS("dtype", dt) });
        add("snake", { HI("M", 6), HI("n", 645), HS("dtype", dt) });
        add("snake", { HI("M", 6), HI("n", 645), HI("log", 1), HS("dtype", dt) });
        /* The alias-free SnakeBeta at BigVGAN's 12-tap filters over two sequences, and an odd
         * filter at ratio 3. */
        for (int lg : { 0, 1 })
            add("snake_aa", { HI("M", 24), HI("n", 37), HI("ratio", 2), HI("k", 12),
                              HI("log", lg), HS("dtype", dt) });
        add("snake_aa", { HI("M", 10), HI("n", 645), HI("ratio", 3), HI("k", 7), HS("dtype", dt) });
        add("resample_nearest", { HI("M", 9), HI("n", 645), HS("dtype", dt) });
        add("resample_nearest", { HI("M", 4), HI("n", 64), HS("dtype", dt) });
        add("chan_split", { HI("M", 9), HI("n", 645), HI("parts", 2), HS("dtype", dt) });
        add("chan_split", { HI("M", 5), HI("n", 64), HI("parts", 3), HS("dtype", dt) });
        add("interleave2", { HI("M", 645), HS("dtype", dt) });
    }
    for (int rep : { 1, 2 })
        for (int ck : { 0, 40 })
            add("cfg_topk_sample", { HI("M", 4 * rep), HI("n_vocab", 645), HI("k", 50),
                                     HI("rep", rep), HI("cond_k", ck), HF("cfg", 1.5),
                                     HI("stream", 3), HS("dtype", "f32") });
    add("cfg_topk_sample", { HI("M", 6), HI("n_vocab", 645), HI("k", 645), HI("rep", 2),
                             HF("cfg", 1.7), HF("temp", 0.8), HI("stream", 0),
                             HS("dtype", "f32") });
    /* Fish Audio's draw: the nucleus over the whole candidate set's softmax, then the top-k cut;
     * at a small top_p the head alone, and above 1 none. */
    for (double tp : { 0.7, 0.05, 0.95 })
        add("cfg_topk_sample", { HI("M", 4), HI("n_vocab", 645), HI("k", 30), HI("rep", 1),
                                 HF("cfg", 1.0), HF("temp", 0.7), HI("stream", 2),
                                 HF("top_p", tp), HS("dtype", "f32") });
    /* A speech model's backbone pick: reserved ids suppressed (the vocabulary's last class stays a
     * candidate), a temperature, the penalty idle without a history. */
    for (int rep : { 1, 2 })
        add("cfg_topk_sample", { HI("M", 2 * rep), HI("n_vocab", 645), HI("k", 50),
                                 HI("rep", rep), HF("cfg", 4.0), HF("temp", 0.9), HI("stream", 0),
                                 HI("sup_lo", 600), HI("sup_hi", 644), HF("penalty", 1.1),
                                 HS("dtype", "f32") });
    for (const char* dt : { "f32", "bf16" }) {
        for (double sc : { 1.0, 0.7 })
            add("rvq_embed", { HI("M", 7), HI("n_embd", 645), HI("n_codes", 8), HI("vocab", 64),
                               HI("off0", 5), HF("scale", sc), HS("dtype", dt) });
        /* Codes past a codebook smaller than its heads read its last row (Fish Audio's codec). */
        add("rvq_embed", { HI("M", 7), HI("n_embd", 645), HI("n_codes", 4), HI("vocab", 64),
                           HI("off0", 0), HF("scale", 1.0), HI("clamp", 40), HS("dtype", dt) });
        /* One rounding of the whole sum: sixteen codebooks, as a speech model's frame embedding. */
        for (double sc : { 1.0, 0.7 })
            add("rvq_embed", { HI("M", 7), HI("n_embd", 645), HI("n_codes", 16), HI("vocab", 64),
                               HI("off0", 0), HF("scale", sc), HI("round", 1), HS("dtype", dt) });
        add("frame_mix_store", { HI("M", 2), HI("n_embd", 645), HI("L", 9), HI("stride", 5),
                                 HI("rep", 2), HS("dtype", dt) });
        add("frame_mix_store", { HI("M", 4), HI("n_embd", 64), HI("L", 3), HI("stride", 2),
                                 HI("rep", 1), HS("dtype", dt) });
        for (int round : { 1, 0 })
            add("fourier_embed", { HI("M", 5), HI("n", 256), HI("round", round),
                                   HS("dtype", dt) });
        add("win_input", { HI("M", 24), HI("n_lat", 37), HI("n_zero", 0), HI("n_cond", 64),
                           HI("stride", 12), HS("dtype", dt) });
        add("win_input", { HI("M", 36), HI("n_lat", 64), HI("n_zero", 3), HI("n_cond", 37),
                           HI("stride", 16), HS("dtype", dt) });
        for (int ov : { 6, 16 })
            add("win_prep", { HI("M", 32), HI("n", 64), HI("stride", 16), HI("ov", ov),
                              HS("dtype", dt) });
        add("win_prep", { HI("M", 64), HI("n", 37), HI("stride", 16), HI("ov", 4),
                          HS("dtype", dt) });
        add("win_finish", { HI("M", 32), HI("n", 64), HI("n_cond", 37), HI("stride", 16),
                            HI("ov", 6), HS("dtype", dt) });
        add("win_finish", { HI("M", 64), HI("n", 64), HI("n_cond", 37), HI("stride", 16),
                            HI("ov", 4), HS("dtype", dt) });
        add("win_finish", { HI("M", 36), HI("n", 37), HI("n_cond", 64), HI("stride", 12),
                            HI("ov", 5), HS("dtype", dt) });
    }
    add("end_status", { HI("M", 2), HI("end", 5), HI("rep", 2) });
    add("end_status", { HI("M", 4), HI("end", 3), HI("rep", 1) });
    add("ras_pick", { HI("M", 4), HI("win", 16), HI("lo", 1), HI("hi", 9), HI("rep", 1) });
    add("ras_pick", { HI("M", 2), HI("win", 4), HI("lo", 0), HI("hi", 12), HI("rep", 2),
                      HI("clear", 1) });
    /* A codec encoder's residual quantiser step: DAC's eight-wide codes over a semantic codebook
     * and a residual one, both dtypes; a dimension that is no vector's width. */
    for (const char* dt : { "f32", "bf16" }) {
        add("vq_encode", { HI("M", 7), HI("n", 64), HI("dim", 8), HI("bins", 4096),
                           HS("dtype", dt) });
        add("vq_encode", { HI("M", 5), HI("n", 128), HI("dim", 8), HI("bins", 1024),
                           HS("dtype", dt) });
    }
    add("vq_encode", { HI("M", 3), HI("n", 40), HI("dim", 5), HI("bins", 33),
                       HS("dtype", "f32") });
    /* Mimi's Euclidean quantiser step at its codebook's shape and at widths that are no
     * vector's, and the encoder's ELU over a row that is none either. */
    for (const char* dt : { "f32", "bf16" }) {
        add("vq_euclidean", { HI("M", 7), HI("n", 256), HI("bins", 2048), HS("dtype", dt) });
        add("vq_euclidean", { HI("M", 3), HI("n", 37), HI("bins", 33), HS("dtype", dt) });
        add("elu", { HI("M", 6), HI("n", 645), HS("dtype", dt) });
    }
    add("frame_store", { HI("M", 2), HI("n_codes", 16), HI("stride", 9), HI("rep", 2),
                         HI("skip", 0) });
    add("frame_store", { HI("M", 4), HI("n_codes", 3), HI("stride", 5), HI("rep", 1),
                         HI("skip", 1) });
    add("iota", { HI("M", 645), HI("start", 3), HI("step", 2) });
    add("tile_grid", { HI("M", 9), HI("tile", 16), HI("stride", 14) });
    add("vol_index", { HI("M", 125) });
    add("iota", { HI("M", 7), HI("start", -5), HI("step", 1) });
    add("cu_repeat", { HI("M", 5), HI("mult", 3) });
    add("cu_repeat", { HI("M", 9), HI("mult", 1) });
    add("int_affine", { HI("M", 4), HS("src", "0 -1 2 1"), HS("mul", "2 0 8 4"),
                        HS("add", "-1 7 -7 -3") });
    add("sub_i32", { HI("M", 1) });
    add("sub_i32", { HI("M", 37) });

    /* The row-form names: every case of the base op again under the name. */
    {
        const char* const forms[][2] = { { "rmsnorm", "rmsnorm_rows" }, { "rope", "rope_rows" } };
        const size_t n = c.size();
        for (size_t i = 0; i < n; ++i)
            for (const auto& f : forms)
                if (std::strcmp(c[i].op, f[0]) == 0) {
                    Case x = c[i];
                    x.op = f[1];
                    c.push_back(x);
                }
    }

    /* ================================================================== speed-only, at model size
     *
     * Timed by rad-avx-bench, skipped by rad-avx-check. The cases above are sized to break kernels;
     * speed depends on the cache hierarchy and needs real shapes.
     *
     * Geometry of a ~1.5B dense model (n_embd 2048, n_ff 5632, n_vocab 32000, 16 heads of 128).
     * M = 1 is decode, 64 a speculative verify window, 512 a prefill chunk.
     *
     * `logits_gemm` at 32000 x 2048 is memory-bound: its 128 MiB bf16 weight fits no cache, so
     * compare it with the DRAM triad, not the FMA ceiling. */
    auto speed = [&](const char* op, std::vector<P> p, double bytes, double flops) {
        c.push_back(Case{ op, std::move(p), bytes, flops, true });
    };
    for (int M : { 1, 64, 512 }) {
        speed("gemm_nt", { HI("M", M), HI("N", 2048), HI("K", 2048), HS("dtype", "bf16") },
              0, 2.0 * M * 2048 * 2048);
        speed("gemm_nt", { HI("M", M), HI("N", 5632), HI("K", 2048), HS("dtype", "bf16") },
              0, 2.0 * M * 5632 * 2048);
    }
    /* An encoder step (all tokens at once) at EmbeddingGemma 2's two FFN shapes. */
    for (int M : { 2048 }) {
        speed("gemm_nt", { HI("M", M), HI("N", 4096), HI("K", 512), HS("dtype", "bf16") },
              0, 2.0 * M * 4096 * 512);
        speed("gemm_nt", { HI("M", M), HI("N", 512), HI("K", 2048), HS("dtype", "bf16") },
              0, 2.0 * M * 512 * 2048);
        speed("rmsnorm", { HI("M", M), HI("n", 2048), HF("eps", 1e-6), HS("dtype", "bf16"),
                           HF("wadd", 0) }, (double)M * 2048 * 2 * 2, 0);
        speed("silu_mul", { HI("M", M), HI("n", 5632), HS("dtype", "bf16") },
              (double)M * 5632 * 3 * 2, 0);
        speed("softmax", { HI("M", M), HI("n", 2048), HS("dtype", "f32") },
              (double)M * 2048 * 4 * 2, 0);
        speed("quant_act_fp8", { HI("M", M), HI("n", 2048), HI("group", 128),
                                 HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
        speed("quant_act_i8", { HI("M", M), HI("n", 2048), HI("group", 128),
                                HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
        speed("quant_act_i8g", { HI("M", M), HI("n", 2048), HI("group", 128),
                                 HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
        speed("had_quant_act_i8g", { HI("M", M), HI("n", 2048), HI("group", 128),
                                     HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
        speed("had_quant_act_i8", { HI("M", M), HI("n", 2048), HI("group", 128),
                                    HS("dtype", "bf16") }, (double)M * 2048 * 3, 0);
    }
    /* GDN at qwen35 model shape (k128/v128, chunk 64, 16 key heads, 48 value heads): chunk_scan
     * over a 512 chunk (prefill) and recurrent_update at q_len 1 (decode). No rate: the traffic
     * depends on the state layout. libref runs these scalar, so keep --iters low. */
    speed("gdn_chunk_scan", { HI("M", 512), HI("head_k", 128), HI("head_v", 128), HI("chunk", 64),
                              HF("scale", 0.176), HI("state_fp16", 0),
                              HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    speed("gdn_chunk_scan", { HI("M", 128), HI("head_k", 128), HI("head_v", 128), HI("chunk", 64),
                              HF("scale", 0.176), HI("state_fp16", 0),
                              HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    speed("gdn_recurrent_update", { HI("q_len", 1), HI("head_k", 128), HI("head_v", 128),
                                    HI("n_head_v", 48), HF("scale", 0.176), HF("eps", 1e-6),
                                    HF("l2_eps", 1e-6), HF("softplus_thr", 20.0), HS("act", "silu"),
                                    HI("state_fp16", 0), HI("n_head_k", 16), HI("n_seq", 2) }, 0, 0);
    speed("gdn_kkt_solve", { HI("M", 512), HI("head_k", 128), HI("chunk", 64),
                             HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    /* Conv prep at model shape: conv_dim 10240 = 2*16*128 + 48*128, width 4, over a 512 chunk. */
    speed("gdn_conv_prep", { HI("M", 512), HI("head_k", 128), HI("head_v", 128), HI("chunk", 64),
                             HI("conv_width", 4), HF("l2_eps", 1e-6), HF("softplus_thr", 20.0),
                             HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    /* Conv update at decode shape: one new token per sequence, same widths. */
    speed("gdn_conv_update", { HI("q_len", 1), HI("head_k", 128), HI("head_v", 128),
                               HI("conv_width", 4), HI("max_query_len", 1),
                               HI("n_head_k", 16), HI("n_head_v", 48), HI("n_seq", 2) }, 0, 0);
    speed("logits_gemm", { HI("M", 1), HI("n_vocab", 32000), HI("n_embd", 2048),
                           HS("dtype", "bf16") }, (double)32000 * 2048 * 2, 2.0 * 32000 * 2048);
    speed("sample_argmax", { HI("M", 1), HI("n_vocab", 32000) }, (double)32000 * 4, 0);
    speed("sample_logprobs", { HI("M", 1), HI("n_vocab", 32000), HI("k", 20) }, (double)32000 * 4, 0);
    speed("sample_topk", { HI("M", 1), HI("n_vocab", 32000), HI("vocab_off", 0) }, 0, 0);
    /* Decode attention over a real context: 16 query heads in groups of 4, 2048 cached keys. */
    speed("attn_paged", { HI("q_len", 1), HI("head_dim", 128), HI("gqa", 4), HI("block_size", 16),
                          HI("causal", 1), HI("window", 0), HS("q_dtype", "bf16"),
                          HS("kv_dtype", "bf16"), HF("scale", 0.088), HI("max_ctx", 2048),
                          HI("n_head", 16), HI("max_seqs", 1) },
          /* No rate: the KV bytes read depend on the context length drawn into `seqused`,
             which this table does not control. */
          0, 0);
    /* Prefill attention: 512 queries over 512 cached keys, same heads. */
    speed("attn_paged", { HI("q_len", 512), HI("head_dim", 128), HI("gqa", 4),
                          HI("block_size", 16), HI("causal", 1), HI("window", 0),
                          HS("q_dtype", "bf16"), HS("kv_dtype", "bf16"), HF("scale", 0.088),
                          HI("max_ctx", 512), HI("n_head", 16), HI("max_seqs", 1) },
          0, 0);
    /* Qwen3.8-Flash-Next's indexer (arch/qwen4exp_fp8: 4 heads of 128, ratio 4, topk 512, a
     * 7-row ring at depth 3) over 8K, 64K and 200K tokens of context: decode at one query and at
     * one sequence's four verify rows, and a 2048-token prefill chunk at 64K. Decode streams the
     * bf16 block keys (256 bytes a block), so it is rated in GB/s; the prefill chunk is FMA work.
     * The bench sets every block complete and every query past all of them (avx_bench.cpp:
     * model_shape). Page ids are drawn at random, which a block table need not be in practice. */
    for (const int64_t blocks : { 2048, 16384, 50000 })
        for (const int64_t M : { 1, 4 })
            speed("qsa_score", { HI("M", M), HI("n", 128), HI("heads", 4), HI("blocks", blocks),
                                 HI("seqs", 1), HS("dtype", "bf16") },
                  (double)blocks * (128 * 2 + 4) + (double)M * blocks * 4, 0);
    speed("qsa_score", { HI("M", 2048), HI("n", 128), HI("heads", 4), HI("blocks", 16384),
                         HI("seqs", 1), HS("dtype", "bf16") },
          0, 2.0 * 2048 * 4 * 128 * 16384);
    for (const int64_t blocks : { 2048, 16384, 50000 })
        for (const int64_t M : { 1, 4 })
            speed("qsa_select", { HI("M", M), HI("blocks", blocks), HI("topk", 512),
                                  HI("ratio", 4), HI("seqs", 1) },
                  (double)M * blocks * 8, 0);
    speed("qsa_select", { HI("M", 2048), HI("blocks", 16384), HI("topk", 512), HI("ratio", 4),
                          HI("seqs", 1) }, 2048.0 * 16384 * 8, 0);
    /* The residency map at a decode step: an 8K-block id space over a 2048-slot ring, one query row
     * and a four-row verify, 512 blocks a row. All-resident, so this times the selection passes,
     * the claim compaction, the rewrite and the bookkeeping, not the copies. */
    for (const int64_t M : { 1, 4 })
        speed("kv_resolve", { HI("M", M), HI("blocks", 8192), HI("slots", 2048),
                              HI("sel_pitch", 513), HI("block_bytes", 2048) },
              (double)M * 513 * 8, 0);
    /* The tail and the work list: four decoding sequences, and a prefill chunk's volume (512
     * sequences of four rows each completing a block, 2048 key rows staged). */
    speed("qsa_tail_store", { HI("M", 4), HI("n", 128), HI("ratio", 4), HI("ring", 7),
                              HI("seqs", 4), HS("dtype", "bf16") }, 4.0 * 128 * 2 * 2, 0);
    speed("qsa_work", { HI("M", 4), HI("n", 128), HI("ratio", 4), HI("ring", 7), HI("seqs", 4),
                        HI("work", 5), HS("dtype", "bf16") }, 4.0 * 4 * 128 * 2 * 2, 0);
    speed("qsa_work", { HI("M", 2048), HI("n", 128), HI("ratio", 4), HI("ring", 7),
                        HI("seqs", 512), HI("work", 513), HS("dtype", "bf16") },
          2048.0 * 128 * 2 * 2, 0);
    /* The 27B DFlash2 drafter's convolution (arch/common/rad_block_dflash2.h: n_embd 5120, 2
     * taps, groups of 16, a 16-row query block): one sequence's block and four. */
    for (const int64_t T : { 16, 64 })
        speed("dflash_conv", { HI("T", T), HI("hidden_size", 5120), HI("taps", 2),
                               HI("group_size", 16), HI("NG", 320), HI("block_size", 16),
                               HS("dtype", "bf16") },
              (double)T * 5120 * 2 * 2 + (double)T * 2 * 320 * 2 + 2.0 * 5120 * 2, 0);
    /* Rope at model shape: 512 tokens, 16+4 heads of 128. Time-only (libref scalar). */
    speed("rope", { HI("M", 512), HI("head_dim", 128), HI("n_head", 16), HI("n_head_kv", 4),
                    HF("theta", 10000.0), HF("scale", 1.0), HS("mode", "neox"),
                    HI("rotary_dim", 128) }, 0, 0);
    speed("qk_norm_rope", { HI("M", 512), HI("head_dim", 128), HI("n_head", 16),
                            HI("n_head_kv", 4), HF("theta", 10000.0), HF("eps", 1e-6),
                            HF("scale", 1.0), HS("mode", "neox"), HI("rotary_dim", 128),
                            HF("wadd", 0) }, 0, 0);

    /* Qwen-Image 2.1 (docs/ARCHITECTURES.md): the VAE decoder's widths 1152..144 and the DiT's
     * 32 heads of 128 over 4096 rows. Each case is about 25 GFLOP so a one-thread run stays
     * practical; libref is not timed past 2e10 (rad-avx-bench). A 512 x 512 picture's decoder
     * runs its 1152-wide convolutions at 32 x 32 pixels, its 576-wide ones at 128 x 128 and its
     * 144-wide ones at 512 x 512; the cases cut the pixels to keep the width (libref describes an
     * eight-column picture, the same work per pixel). */
    struct ConvSpeed { int64_t M, cin, cout, k, up; };
    static const ConvSpeed kConvSpeed[] = {
        { 1024, 1152, 1152, 3, 1 },    /* mid block and up block 0, at 32 x 32 */
        { 4096, 576, 576, 3, 1 },      /* up block 2 */
        { 16384, 288, 288, 3, 2 },     /* up block 3's upsampler, nearest 2x then 3 x 3 */
        { 65536, 144, 144, 3, 1 },     /* up block 4, at the picture's resolution */
        { 262144, 144, 4, 3, 1 },      /* conv_out at 512 x 512: four output channels */
        { 1024, 1152, 3456, 1, 1 },    /* the middle attention's qkv, 1 x 1 */
    };
    for (const ConvSpeed& v : kConvSpeed)
        speed("conv2d_nhwc", { HI("M", v.M), HI("cin", v.cin), HI("cout", v.cout), HI("k", v.k),
                               HI("stride", 1), HS("pad", "same"), HI("up", v.up),
                               HS("dtype", "bf16") },
              0, 2.0 * (double)v.M * v.cout * v.cin * v.k * v.k);
    /* The denoising attention: an image's 1024 latent rows (512 x 512) against a 256-key
     * conditioning, all 32 heads; 4096 rows (1024 x 1024) over four heads (the cost is per head,
     * and four heads of 128 query blocks fill the threads). Both score forms. */
    for (const char* qk : { "bf16", "i8" }) {
        speed("attn_prefix_self", { HI("q_len", 1024), HI("head_dim", 128), HI("gqa", 1),
                                    HI("block_size", 16), HI("causal", 0), HS("q_dtype", "bf16"),
                                    HS("kv_dtype", "bf16"), HF("scale", 0.08838834764831845),
                                    HI("n_head", 32), HI("ctx", 256), HS("qk", qk) },
              0, 4.0 * 1024 * (1024 + 256) * 128 * 32);
        speed("attn_prefix_self", { HI("q_len", 4096), HI("head_dim", 128), HI("gqa", 1),
                                    HI("block_size", 16), HI("causal", 0), HS("q_dtype", "bf16"),
                                    HS("kv_dtype", "bf16"), HF("scale", 0.08838834764831845),
                                    HI("n_head", 4), HI("ctx", 256), HS("qk", qk) },
              0, 4.0 * 4096 * (4096 + 256) * 128 * 4);
    }
    /* The DiT's row ops over a 512 x 512 picture's 1024 rows of 4096, and the decoder's at its
     * last level (512 x 512 pixels): bytes are the operands' reads and writes. */
    speed("ln_mod", { HI("M", 1024), HI("n", 4096), HF("eps", 1e-6), HS("dtype", "bf16") },
          1024.0 * 4096 * 2 * 2, 0);
    speed("gate_add", { HI("M", 1024), HI("n", 4096), HS("dtype", "bf16") },
          1024.0 * 4096 * 2 * 3, 0);
    speed("qk_norm_rope", { HI("M", 1024), HI("head_dim", 128), HI("n_head", 32),
                            HI("n_head_kv", 32), HF("theta", 10000.0), HF("eps", 1e-6),
                            HS("mode", "gptj_blocks"), HS("sections", "64"),
                            HS("dtype", "bf16") }, 1024.0 * 64 * 128 * 2 * 2, 0);
    speed("l2norm_chan", { HI("M", 262144), HI("n", 144), HS("act", "silu"), HS("dtype", "bf16") },
          262144.0 * 144 * 2 * 2, 0);
    speed("d2s_add", { HI("M", 262144), HI("cin", 576), HI("cout", 288), HI("ft", 1),
                       HS("dtype", "bf16") }, 262144.0 * 288 * 2 * 2 + 65536.0 * 576 * 2, 0);
    /* Qwen3.6-35B-A3B's routed experts in block fp8 (256 experts of 512 at top 8 over n_embd
     * 2048): gate|up in two parts with the token-order activation, down with the sorted one. The
     * bytes are the experts a call routes to at most, top_k * M of them. */
    for (int M : { 1, 8, 512 }) {
        const double ex = std::min(256.0, 8.0 * M);
        speed("moe_gemm_q", { HI("M", M), HI("N", 1024), HI("K", 2048), HI("parts", 2),
                              HI("n_expert", 256), HI("top_k", 8), HI("group", 128),
                              HS("a_order", "token"), HS("dtype", "fp8a8") },
              ex * 1024 * 2048, 2.0 * M * 8 * 1024 * 2048);
        speed("moe_gemm_q", { HI("M", M), HI("N", 2048), HI("K", 512), HI("n_expert", 256),
                              HI("top_k", 8), HI("group", 128), HS("a_order", "sorted"),
                              HS("dtype", "fp8a8") },
              ex * 2048 * 512, 2.0 * M * 8 * 2048 * 512);
    }
    /* Flash-Next's gated residual read (n 2560, hc 4, lowrank 320) at decode, a verify window and
     * a prefill chunk; the bytes are the two mixing matrices in bf16, read once per launch. */
    for (int M : { 1, 4, 512 })
        speed("hc_read", { HI("M", M), HI("n", 2560), HI("hc", 4), HI("lowrank", 320),
                           HF("eps", 1e-6), HS("dtype", "bf16"), HF("wadd", 1), HI("inject", 1),
                           HI("group", 128), HI("rotate", 0) },
              2.0 * 2 * 320 * 10240, 2.0 * 2 * M * 320 * 10240);
    /* The DiT's feed-forward gate_up over 1024 rows, as the bf16 and the block-fp8 containers
     * hold it: 61% of a block's FLOPs are in its three feed-forward products. */
    speed("gemm_nt", { HI("M", 1024), HI("N", 12288), HI("K", 4096), HS("dtype", "bf16") }, 0,
          2.0 * 1024 * 12288 * 4096);
    speed("gemm_nt_q", { HI("M", 1024), HI("N", 12288), HI("K", 4096), HI("group", 128),
                         HS("dtype", "fp8a8") }, 0, 2.0 * 1024 * 12288 * 4096);
    /* The music vocoder's third stage (384 channels; a window's two streams there are 88192 rows,
     * timed over 8192): a residual unit's dilated 7-tap convolution and its 1 x 1, the stage's
     * stride-4 upsampling from 768 channels, and the snake before each. */
    speed("conv1d", { HI("M", 8192), HI("cin", 384), HI("cout", 384), HI("k", 7), HI("dil", 3),
                      HS("dtype", "bf16") }, 0, 2.0 * 8192 * 384 * 384 * 7);
    speed("conv1d", { HI("M", 8192), HI("cin", 384), HI("cout", 384), HI("k", 1), HI("dil", 1),
                      HS("dtype", "bf16") }, 0, 2.0 * 8192 * 384 * 384);
    speed("conv_t1d", { HI("M", 2048), HI("cin", 768), HI("cout", 384), HI("stride", 4),
                        HS("dtype", "bf16") }, 0, 2.0 * 8192 * 384 * 2 * 768);
    speed("snake", { HI("M", 8192), HI("n", 384), HS("dtype", "bf16") }, 8192.0 * 384 * 2 * 2, 0);

    return c;
}
