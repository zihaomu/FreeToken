// r4d_sample_ngram_f32 -- the windowed no-repeat-ngram ban (`sample_ngram_ban`).
//
// Unlimited-OCR's generation_config asks for SlidingWindowNoRepeatNgramProcessor(n=35,
// window=128): a token that would complete an n-gram the last `window` tokens of the sequence
// already contain is banned, by setting its logit to -infinity. modeling_unlimitedocr.py:353-383
// is the reference, which SGLang ships as DeepseekOCRNoRepeatNGramLogitProcessor. A model that
// loops at the n-gram level (OCR layout rows repeat) otherwise repeats a phrase until the context
// ends.
//
// The reference's rule, with the row's history `s[0, hl)` oldest first (the committed tokens,
// including the one just emitted, plus this step's earlier drafts at a speculative verify):
//
//     search_start = max(0, hl - w)   with w from the row's {n, window} pair, w <= 0 whole span
//     search_end   = hl - n + 1
//     for idx in [search_start, search_end):
//         if s[idx .. idx + n - 2] == s[hl - n + 1 .. hl - 1):   ban s[idx + n - 1]
//
// and at n = 1 the prefix is empty, so every token in the window is banned. The window bounds the
// n-gram's START, not its end, exactly as the reference slices `sequence[search_start:search_end]`
// -- which is why an n-gram that spans the window boundary is still searched.
//
// Exactness. Nothing here rounds: the kernel compares int32 history ids and stores the constant
// -infinity, so its answer is libref's ref_sample_ngram_ban by construction, not by tolerance. The
// only ordering freedom is that two threads may store -infinity to the same column, and the store
// is idempotent.
//
// Ids are global and the logits row is this rank's vocabulary shard: a token's column is its id
// less `vocab_off`, and a token outside the shard is another rank's to mask (it is not a column
// here at all).
//
// The optional `whitelist` plane is the reference's whitelist (modeling_unlimitedocr.py:359,379):
// word w bit b of the row's own plane row covers column 32w + b, and a SET bit removes that token
// from the banned set -- it is never written. Absent, nothing is exempt, and the kernel's answer
// is the one before the whitelist existed.
//
// The scan is O(w * n) a row over a history of at most max_hist tokens, so one block a row with
// the positions dealt out across the block's threads; the n-1-token prefix is staged in LDS once
// when it fits (n <= 256), which is every request the model card asks for (35).

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"

#include <cfloat>

// The params row is the engine's wire format, included rather than restated (abi/rad_sample.h).
#include "rad_sample.h"

namespace {

constexpr uint32_t kThreads = 256;

// The prefix staged in LDS, one int32 a token: 1 KiB. A longer n falls back to reading the
// history plane, which is correct but repeats the prefix's loads per position.
constexpr uint32_t kMaxPrefix = 256;

using SP = RadSampleParams;

__global__ __launch_bounds__(kThreads) void ngram_ban_kernel(
    float* __restrict__ lg, const SP* __restrict__ sp, const int32_t* __restrict__ hist,
    const int32_t* __restrict__ ng, const uint32_t* __restrict__ wl, uint32_t wl_words,
    uint32_t n_vocab, uint32_t lg_ld, uint32_t hmax, int32_t voff) {
    __shared__ int32_t pf[kMaxPrefix];
    __shared__ int64_t s_start, s_end, s_plen, s_p0;

    const uint32_t m = blockIdx.x;
    const SP p = sp[m];
    // The row's {n-gram length, window} pair (a separate plane, so the params row keeps the wire
    // format every sampler op shares).
    const int32_t ngram = ng[2 * m];
    const int32_t win_req = ng[2 * m + 1];
    // Every test before the barrier reads the row's own planes, so the block leaves together.
    if (ngram <= 0) return;

    // The whole staged span of the row (hist_off/hist_len both come from the engine; a negative
    // hist_off is 0), clamped to the plane.
    int32_t hl = p.hist_len;
    int32_t ho = p.hist_off < 0 ? 0 : p.hist_off;
    if (ho > (int32_t)hmax) ho = (int32_t)hmax;
    if (hl > (int32_t)hmax - ho) hl = (int32_t)hmax - ho;

    if (threadIdx.x == 0) {
        int64_t start = 0, end = 0, plen = 0, p0 = 0;
        if (hl >= ngram) {
            // w <= 0 is the un-windowed processor: the whole staged history.
            const int64_t win = win_req > 0 && win_req < hl ? win_req : hl;
            start = hl - win;
            end   = (int64_t)hl - ngram + 1;
            plen  = (int64_t)ngram - 1;
            p0    = (int64_t)hl - plen;
            if (end <= start) end = start;   /* nothing to search */
        }
        s_start = start;
        s_end   = end;
        s_plen  = plen;
        s_p0    = p0;
    }
    __syncthreads();

    const int64_t start = s_start, end = s_end, plen = s_plen, p0 = s_p0;
    if (end <= start) return;

    const int32_t* __restrict__ h = hist + (size_t)m * hmax + ho;
    float* __restrict__ row = lg + (size_t)m * lg_ld;
    // The row's own whitelist words (a SET bit exempts the column, the reference's
    // `banned.difference_update(self.whitelist)`), read at the position loop below.
    const uint32_t* __restrict__ w = wl ? wl + (size_t)m * wl_words : nullptr;

    // The prefix: LDS when it fits, otherwise each position reads it from the plane.
    const bool in_lds = plen <= (int64_t)kMaxPrefix;
    if (in_lds) {
        for (int64_t j = threadIdx.x; j < plen; j += (int64_t)kThreads) pf[j] = h[p0 + j];
        __syncthreads();
    }
    auto prefix_at = [&](int64_t j) -> int32_t {
        return in_lds ? pf[j] : h[p0 + j];
    };

    for (int64_t idx = start + (int64_t)threadIdx.x; idx < end; idx += (int64_t)kThreads) {
        bool match = true;
        for (int64_t j = 0; j < plen; ++j) {
            if (h[idx + j] != prefix_at(j)) { match = false; break; }
        }
        if (!match) continue;
        const int32_t tok = h[idx + plen];
        const int64_t col = (int64_t)tok - voff;
        if (tok < 0 || col < 0 || col >= (int64_t)n_vocab) continue;
        if (wl && ((w[col >> 5] >> (uint32_t)(col & 31)) & 1u) != 0u) continue;
        // Two positions may ban one token; the store is idempotent and the value is constant.
        row[col] = -INFINITY;
    }
}

int r4d_sample_ngram_ban_f32(long lg, long params, long hist, long ng, long wl, int wl_words,
                             int M, int n_vocab, int hmax, int lg_ld, int vocab_off, long stream) {
    if (!lg || !params || !hist || !ng) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || n_vocab <= 0 || hmax <= 0) return R4D_NO(RAD_E_SHAPE);
    if (wl && wl_words <= 0) return R4D_NO(RAD_E_SHAPE);
    if (!lg_ld) lg_ld = n_vocab;
    hipLaunchKernelGGL(ngram_ban_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (float*)lg, (const SP*)params, (const int32_t*)hist, (const int32_t*)ng,
                       (const uint32_t*)wl, (uint32_t)(wl ? wl_words : 0),
                       (uint32_t)n_vocab, (uint32_t)lg_ld, (uint32_t)hmax, (int32_t)vocab_off);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

}  // namespace

// ================================================================================ entry point
//
// The shim checks what the kernel assumes: f32 logits, an i32 history contiguous in both rows and
// columns (the kernel walks it as one int32 a token), a params plane of whole RadSampleParams rows
// and the row pitches. `M` is the step's staged row count, never the band's upper bound.

#define R4D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R4D_NO(_s); } while (0)

static inline int dt_f32(const RadTensor* t) {
    return (t && t->dtype == RAD_F32) ? RAD_OK : RAD_E_DTYPE;
}

static inline int dt_i32(const RadTensor* t) {
    return (t && (t->dtype == RAD_I32 || t->dtype == RAD_U32)) ? RAD_OK : RAD_E_DTYPE;
}

static int params_ok(const RadTensor* t, long long M) {
    if (!t || !t->data) return R4D_NO(RAD_E_INVAL);
    if (t->dtype != RAD_U32 && t->dtype != RAD_I32 && t->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(t)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(t) * 4 < M * (long long)sizeof(RadSampleParams)) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}

extern "C" int r4d_rad_sample_ngram_ban(const RadArgs* a, RadStream s) {
    enum { V_LOGITS = 0, V_PARAMS, V_HIST, V_NGRAM, V_WL };
    const RadTensor* lg = r4d_opt(a, V_LOGITS);
    if (!lg) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_f32(lg));
    const long long nv = r4d_geti_or(a, "n_vocab", r4d_dim(lg, (int)lg->rank - 1));
    if (nv <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long ld = r4d_rows_pitch(lg, nv, &M);
    if (ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);

    const RadTensor* pr = r4d_opt(a, V_PARAMS);
    const RadTensor* h  = r4d_opt(a, V_HIST);
    const RadTensor* ng = r4d_opt(a, V_NGRAM);
    R4D_NEED(params_ok(pr, M));
    if (!h || !ng) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(h));
    R4D_NEED(dt_i32(ng));
    if (!r4d_contig(h) || !r4d_contig(ng)) return R4D_NO(RAD_E_STRIDE);
    const long long hmax = r4d_numel(h) / M;
    if (hmax <= 0) return R4D_NO(RAD_E_SHAPE);
    /* One {n, window} pair a row, at least M of them. */
    if (r4d_numel(ng) < 2 * M) return R4D_NO(RAD_E_SHAPE);

    /* The optional whitelist bitmask: contiguous words of 32 columns, M rows of them. */
    const RadTensor* wl = r4d_opt(a, V_WL);
    const long long wwords = (nv + 31) / 32;
    long long wl_addr = 0;
    if (wl) {
        R4D_NEED(dt_i32(wl));
        if (!r4d_contig(wl)) return R4D_NO(RAD_E_STRIDE);
        if (r4d_numel(wl) < M * wwords) return R4D_NO(RAD_E_SHAPE);
        wl_addr = r4d_addr(wl);
    }

    const long long voff = r4d_geti_or(a, "vocab_off", 0);
    return r4d_sample_ngram_ban_f32(r4d_addr(lg), r4d_addr(pr), r4d_addr(h), r4d_addr(ng),
                                    wl_addr, (int)wwords,
                                    (int)M, (int)nv, (int)hmax, (int)ld, (int)voff,
                                    (long)(uintptr_t)s);
}
