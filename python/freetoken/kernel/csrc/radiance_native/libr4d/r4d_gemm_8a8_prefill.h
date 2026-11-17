// r4d_gemm_8a8_prefill.h -- the int8 and fp8 prefill tile's body (gemm_8a8_prefill_body), shared
// by r4d_gemm_fp8a8.hip's tiles and r4d_gemm_iqr.hip's, which decodes its weight in the body's
// `begin` hook. Included inside an anonymous namespace after the includer's kWave (32) and kBlock
// (128), with r4d_fp8_wmma.h's names in scope.

// ---- the prefill tile's geometry ---------------------------------------------------------------
//
// A block owns 128 weight rows x 128 activation rows: eight waves as kPWW (weight) x kPWA
// (activation), each wave kPTW weight tiles x kPTA activation tiles of 16 x 16. Eight accumulator
// tiles a wave is a hard cap: each is two f32 sets (`raw` and `out`), so more spills. The staged K
// tile is kPFK bytes, two per 128-column scale block. See gemm_8a8_prefill_kernel for the layout
// and what was measured.
constexpr uint32_t kPTW = 2, kPTA = 4, kPWW = 4, kPWA = 2;
constexpr uint32_t kPBN = kPWW * kPTW * 16, kPBM = kPWA * kPTA * 16;   // weight x activation rows
constexpr uint32_t kPWaves = kPWW * kPWA;
constexpr uint32_t kPFK = 64;
static_assert(kPBN == kBlock, "a block's weight rows are one scale row");
// Activation bands per raster group of the prefill tile; see the kernel's grouping.
constexpr uint32_t kGroupM = 8;
/* The LDS the body's activation staging holds, which a `begin` hook may borrow. */
constexpr uint32_t kPrefillLds = 2u * kPBM * (kPFK + 8u);

// ---- the prefill tile ---------------------------------------------------------------------------
//
// 128 weight rows x 128 activation rows a block, eight waves as kPWW (weight) x kPWA (activation),
// a wave kPTW weight tiles x kPTA activation tiles. The weight is the WMMA A operand, read from the
// fragment-order plane straight into registers: a lane's 32 bytes are its four fragments for 64 k,
// so a wave's load is one contiguous KiB and the weight never touches LDS. The activation is the B
// operand, staged through LDS and read by the four waves that share its rows. A lane's accumulator
// is eight consecutive weight rows of one activation row, so the fold takes one activation scale a
// lane a tile (and the block's one weight scale), and the store is one 16-byte write.
//
// What decides the layout is LDS bandwidth. gfx1201 serves 64 bytes of LDS a clock per CU, loads
// and stores alike, and issues an fp8 WMMA every 8 clocks per SIMD, so a CU has 256 bytes of LDS
// per WMMA at the WMMA rate (libr4d/isa/wmmamix.hip). Staging both operands with one activation
// tile x eight weight tiles a wave moves 288 bytes of fragment reads and 64 of stores per WMMA,
// which alone caps a loop at ~73% of the WMMA rate (measured with nothing else in it). This layout
// reads 128 bytes and stores 32 per WMMA, and the weight's 64 bytes come through the vector cache.
//
// At the 300 W limit every GEMM shape here is power-bound (sclk 2.1-2.3 GHz against 2.55 for bare
// WMMA on random operands), so a variant's speed tracks the bytes it moves as much as the
// instructions it issues. Measured against the both-operands-through-LDS 1x8 tile at the DiT's
// shapes (M 18648) and LLM prefill shapes (M 300-4096), 160 TFLOP/s at gate_up:
//   - this layout with the per-row activation scales gathered from global, mul + add fold: +12%;
//   - the activation scales staged through LDS (`s_as`), one row a thread: +4%. A lane's gather
//     touched sixteen cachelines a load, more bytes than its weight fragments;
//   - the fold as one fma instead of a rounded multiply and an add: +4-5%. Not bit-identical to
//     the separate multiply (rel_l2 ~2e-5 against it, both well inside bf16 output);
//   - unconditional prefetch (the last tile's prefetch re-reads the last tile, discarded): +1%.
//     A conditional load joins two paths with different counts and the compiler waits for zero;
//   - the LDS-only barrier (lds_barrier) instead of __syncthreads: +1%; the latter's release waits
//     for every outstanding load, the weight prefetch included;
//   - kGroupM 8 instead of 16: +1-3%; 4 ties 8.
// Measured and rejected:
//   - 2 x 4 waves of 4 weight x 2 activation tiles (half the LDS reads, twice the weight bytes
//     through the vector cache): 3-5% slower;
//   - 8 x 1 waves (256 x 64 blocks, no weight tile read twice in a block): ties at o and gate_up;
//   - the weight two tiles ahead: 240 VGPRs and scratch spills, 25-38% slower;
//   - 12 or 16 waves (192 or 256 activation rows a block, up to a quarter less L2 traffic): ties
//     to 1% slower, a higher clock at a lower rate per clock;
//   - four threads an activation row, 16 bytes each (whole 64-byte segments a load): neutral;
//   - raw buffer loads with 32-bit offsets for both operands: 20% slower;
//   - s_setprio around the WMMA run: neutral. CU mode (-mcumode): ~10% slower.
/* RB (I8 only): the weight's scale is one per RB rows and 128 K, as a trellis on libquant's tcq_rb
 * grid widens (r4d_fp8_layout.cpp; the per-row plane holds the block's one value in each row):
 * RB 8, a lane's eight accumulator rows, one scale per lane per tile; RB 32, a wave's two tiles,
 * one scale per wave as the fp8 tile has. The fold is then the fp8 one, one fma per slot: the
 * int32 accumulator starts each 128-K block at 0x4B400000, so the slot read as an f32 is exactly
 * 12582912 + sum (|sum| <= 128 * 127 * 127 < 2^22) and needs no conversion. The scales are summed
 * in `corr` (per lane and tile at RB 8, per lane at RB 32), and 12582912 times them leaves `out`
 * every kRbFlush blocks and at the store, which keeps the f32 sum within a few bias terms of its
 * value (6.8e-4 of the output's rms against the exact fold, under bf16's own rounding).
 *
 * M = 2048, one rank, the median of 7 interleaved rounds against the fp8 tile at the same clocks:
 * RB 32 -4..0% (gate_up, down, ssm_inz, o), RB 8 -1..+3%, the per-row int8 tile +15..19%. */
constexpr int kRbBias = 0x4B400000;
constexpr float kRbBiasF = 12582912.0f;
constexpr uint32_t kRbFlush = 8;
template <int RB>
__device__ __forceinline__ f8v rb_first() {
    if constexpr (RB != 0) return __builtin_bit_cast(f8v, (i8v)(kRbBias));
    else return acc_zero();
}

/* What the tile does around its K loop besides the GEMM: `begin(bm, bn, lds)` once the block's
 * bands are known and before any weight load, with the activation's staging LDS (`kPrefillLds`
 * bytes) free to borrow; `end(bn)` after the block's stores. The int8 and fp8 tiles do nothing; the
 * IQR tile decodes its weight there (r4d_gemm_iqr.hip). */
struct PrefillPlain {
    __device__ void begin(uint32_t, uint32_t, uint8_t*) const {}
    __device__ void end(uint32_t) const {}
};

/* ROWF (RB 32): the weight's scale is one f32 a row times one f32 a 128-K block for all rows (an
 * IQR weight under kRowF32): the block's factor `Cf[kb]` is the fold's wave-uniform scale, as RB
 * 32's, and each row's `Rs[n]` scales its outputs at the store. */
/* SB (I8, RB 0): the K extent of one scale on both sides, 128 or IQ8R's 256. A block of SB
 * columns is SB / kPFK staged tiles; the int32 sums run over all of them before the one fold
 * (|sum| <= 256 * 127 * 127 < 2^24, exact in the f32 conversion), so at 256 the tile folds half as
 * often. */
template <bool I8, int RB, class Pre, bool ROWF = false, uint32_t SB = kBlock>
__device__ __forceinline__ void gemm_8a8_prefill_body(
    const uint8_t* __restrict__ A, const float* __restrict__ Sa, const uint8_t* __restrict__ W,
    const uint16_t* __restrict__ Sw, uint16_t* __restrict__ C, uint32_t M, uint32_t K, uint32_t N,
    uint32_t a_ld, uint32_t sa_ld, uint32_t sw_ld, const Pre& pre,
    const float* __restrict__ Rs = nullptr, const float* __restrict__ Cf = nullptr) {
    static_assert(!ROWF || RB == 32, "a row's scale and a block's factor fold as RB 32's one scale");
    static_assert(SB == kBlock || (I8 && RB == 0 && SB == 256u), "a 256 group is the per-row int8 tile's");
    constexpr uint32_t kTPB = SB / kPFK;   // staged tiles a scale block
    constexpr uint32_t P = lds_pitch<kPFK>();
    static_assert(kPBM * 2u == kPWaves * kWave, "two threads stage an activation row");
    // Double-buffered: one barrier a staged tile, which publishes it.
    __shared__ __attribute__((aligned(16))) uint8_t s_a[2][kPBM * P];
    static_assert(sizeof(s_a) == kPrefillLds, "the staging LDS a begin hook may borrow");
    // The activation scales of the current and the next 128-column block, one per block row.
    __shared__ float s_as[2][kPBM];

    const uint32_t tid = threadIdx.x, lane = tid % kWave;
    // Uniform per wave, and through readfirstlane known so, so what derives from it stays scalar.
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const uint32_t ww = wave % kPWW, wa = wave / kPWW;

    // The grid runs in groups of kGroupM activation bands. Blocks dispatch in flattened-id order;
    // with the plain x-fastest grid the activation is read once per 128 weight rows, which halves
    // the rate once it exceeds the 64 MB MALL. Within a group the bands stay cached across the N
    // sweep and the weight is read once per group. With kGroupM bands or fewer the plain order is
    // used and the division skipped.
    uint32_t bm = blockIdx.x, bn = blockIdx.y;
    if (gridDim.x > kGroupM) {
        const uint32_t per = kGroupM * gridDim.y;
        const uint32_t lin = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t first = (lin / per) * kGroupM, r = lin % per;
        const uint32_t gm = min(gridDim.x - first, kGroupM);
        bm = first + r % gm;
        bn = r / gm;
    }
    const uint32_t m0 = bm * kPBM, n0 = bn * kPBN;
    const uint32_t ksuper = K / kSuperK, nkb = K / SB;
    pre.begin(bm, bn, &s_a[0][0]);

    // Staging: two threads a row, 32 bytes each. Rows past M re-read row M-1, and every load is
    // issued unconditionally (the last tile's prefetch re-reads the last tile), so the loop has no
    // branch and its waits count exactly; nothing past M is stored.
    const uint32_t sr = tid / 2u, sc = (tid % 2u) * 32u;
    const uint8_t* const asrc = A + (size_t)min(m0 + sr, M - 1u) * a_ld + sc;
    // Thread t loads the scale of block row t % 128; both threads of a row write the same value.
    const float* const ssrc = Sa + (size_t)min(m0 + tid % kPBM, M - 1u) * sa_ld;
    // This wave's weight tiles, kTileBytes a 64-k supertile, `ksuper` supertiles a tile row.
    const uint32_t wt0 = n0 / 16u + ww * kPTW;
    const uint8_t* const wsrc = W + (size_t)wt0 * ksuper * kTileBytes + lane * kLaneBytes;
    // Activation rows this wave reads from the staged tile, and the lane's column in each.
    const uint32_t arow = wa * kPTA * 16u;

    static_assert(!RB || I8, "RB is the int8 tile's");
    static_assert(RB == 0 || RB == 8 || RB == 32, "a weight scale per 8 or 32 rows");
    static_assert(RB != 32 || kPTW * 16 == 32, "RB 32 is a wave's rows");
    constexpr uint32_t kCI = RB == 8 ? kPTW : 1u;   // corr's tiles: one per tile, or the wave's
    f8v raw[kPTW][kPTA], out[kPTW][kPTA];
    float corr[kCI][kPTA];
#pragma unroll
    for (uint32_t i = 0; i < kPTW; ++i)
#pragma unroll
        for (uint32_t j = 0; j < kPTA; ++j) {
            out[i][j] = acc_zero();
            if constexpr (RB != 0) raw[i][j] = acc_zero();
        }
#pragma unroll
    for (uint32_t i = 0; i < kCI; ++i)
#pragma unroll
        for (uint32_t j = 0; j < kPTA; ++j) corr[i][j] = 0.0f;

    Stage8 pa = stage8_load(asrc, true);
    Stage8 w0[kPTW], w1[kPTW];
#pragma unroll
    for (uint32_t i = 0; i < kPTW; ++i)
        w0[i] = stage8_load(wsrc + (size_t)i * ksuper * kTileBytes, true);
    s_as[0][tid % kPBM] = ssrc[0];
    float scn = ssrc[min(1u, nkb - 1u)];

    /* One staged K tile, the first or second half of scale block `kb`: publish the activation
     * tile (and in the second half the next block's scales), issue the next tile's loads, then four
     * k steps from `cur`. The first k step of a block starts from a zero accumulator, so `raw` needs
     * no clearing after the fold. */
    auto tile = [&](uint32_t kb, uint32_t half, Stage8 (&cur)[kPTW], Stage8 (&nxt)[kPTW]) {
        stage8_store<kPFK>(s_a[half % 2u], sr, sc, pa);
        if (half == kTPB - 1u) s_as[(kb + 1u) % 2u][tid % kPBM] = scn;
        lds_barrier();
        const uint32_t c1 = min(kb * SB + (half + 1u) * kPFK, K - kPFK);
        pa = stage8_load(asrc + c1, true);
#pragma unroll
        for (uint32_t i = 0; i < kPTW; ++i)
            nxt[i] = stage8_load(wsrc + ((size_t)i * ksuper + c1 / kSuperK) * kTileBytes, true);
        if (half == kTPB - 1u) scn = ssrc[min(kb + 2u, nkb - 1u)];
#pragma unroll
        for (uint32_t ks = 0; ks < kPFK / kK; ++ks) {
            i2v bf[kPTA];
#pragma unroll
            for (uint32_t j = 0; j < kPTA; ++j) bf[j] = frag<kPFK>(s_a[half % 2u], arow + j * 16u, ks * kK, lane);
#pragma unroll
            for (uint32_t i = 0; i < kPTW; ++i) {
                const i2v af = frag_of(cur[i], ks);
#pragma unroll
                for (uint32_t j = 0; j < kPTA; ++j)
                    raw[i][j] = mma8<I8>(af, bf[j],
                                         (half == 0u && ks == 0u) ? rb_first<RB>() : raw[i][j]);
            }
        }
    };

    for (uint32_t kb = 0; kb < nkb; ++kb) {
        // The weight's scales for this block, read ahead of the two tiles that hide them: one bf16
        // for the block's 128 rows, or with I8 one per row, a lane's eight rows of a tile adjacent
        // in the tile-major [N/16][K/128][16] plane.
        float ws = 0.0f;
        uint4 w8[I8 && !RB ? kPTW : 1u];
        float wb[RB == 8 ? kPTW : 1u];
        if constexpr (ROWF) {
            ws = Cf[kb];
        } else if constexpr (RB == 32) {
            /* the wave's 32 rows hold one value: the first's, a wave-uniform load */
            ws = bscale(Sw[((size_t)wt0 * nkb + kb) * 16u]);
        } else if constexpr (RB == 8) {
#pragma unroll
            for (uint32_t i = 0; i < kPTW; ++i)
                wb[i] = bscale(Sw[((size_t)(wt0 + i) * nkb + kb) * 16u + (lane >> 4) * 8u]);
        } else if constexpr (I8) {
#pragma unroll
            for (uint32_t i = 0; i < kPTW; ++i)
                w8[i] = *(const uint4*)(Sw + ((size_t)(wt0 + i) * nkb + kb) * 16u + (lane >> 4) * 8u);
        } else {
            ws = bscale(Sw[(size_t)(n0 / kBlock) * sw_ld + kb]);
        }
        tile(kb, 0u, w0, w1);
        tile(kb, 1u, w1, w0);
        if constexpr (kTPB == 4u) {
            tile(kb, 2u, w0, w1);
            tile(kb, 3u, w1, w0);
        }
        if constexpr (RB != 0) {
            if (kb % kRbFlush == 0u && kb) {   // the last kRbFlush blocks' bias, before this fold
#pragma unroll
                for (uint32_t i = 0; i < kPTW; ++i)
#pragma unroll
                    for (uint32_t j = 0; j < kPTA; ++j)
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t)
                            out[i][j][t] = __builtin_fmaf(-kRbBiasF, corr[RB == 8 ? i : 0][j],
                                                          out[i][j][t]);
#pragma unroll
                for (uint32_t i = 0; i < kCI; ++i)
#pragma unroll
                    for (uint32_t j = 0; j < kPTA; ++j) corr[i][j] = 0.0f;
            }
        }
#pragma unroll
        for (uint32_t j = 0; j < kPTA; ++j) {
            const float as = s_as[kb % 2u][arow + j * 16u + frag_c(lane)];
            if constexpr (RB == 32) corr[0][j] = __builtin_fmaf(ws, as, corr[0][j]);
#pragma unroll
            for (uint32_t i = 0; i < kPTW; ++i) {
                if constexpr (RB != 0) {
                    const float sb = (RB == 8 ? wb[i] : ws) * as;
                    if constexpr (RB == 8) corr[i][j] += sb;
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t)
                        out[i][j][t] = __builtin_fmaf(raw[i][j][t], sb, out[i][j][t]);
                    continue;
                }
                float s[8];
                if constexpr (I8) {
                    const uint32_t d[4] = { w8[i].x, w8[i].y, w8[i].z, w8[i].w };
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t)
                        s[t] = as * bscale((uint16_t)(d[t / 2] >> (16 * (t % 2))));
                } else {
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) s[t] = ws * as;
                }
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t)
                    out[i][j][t] = __builtin_fmaf(acc_val<I8>(raw[i][j][t]), s[t], out[i][j][t]);
            }
        }
    }

    // D is [weight row][activation row]: eight consecutive columns of one C row a lane a tile.
#pragma unroll
    for (uint32_t i = 0; i < kPTW; ++i)
#pragma unroll
        for (uint32_t j = 0; j < kPTA; ++j) {
            const uint32_t m = m0 + arow + j * 16u + frag_c(lane);
            const uint32_t n = n0 + (ww * kPTW + i) * 16u + acc_row(lane, 0);
            if (m < M) {
                f8v v = out[i][j];
                if constexpr (RB != 0)
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t)
                        v[t] = __builtin_fmaf(-kRbBiasF, corr[RB == 8 ? i : 0][j], v[t]);
                if constexpr (ROWF)
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) v[t] *= Rs[n + t];
                /* Packed in whole registers (bf16x2_rne, f32_to_bf16's bits). */
                *(uint4*)(C + (size_t)m * N + n) =
                    make_uint4(bf16x2_rne(v[0], v[1]), bf16x2_rne(v[2], v[3]),
                               bf16x2_rne(v[4], v[5]), bf16x2_rne(v[6], v[7]));
            }
        }
    pre.end(bn);
}
