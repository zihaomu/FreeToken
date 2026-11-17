/* engine_probe.cpp -- RADIANCE_DUMP_BUF, the buffer probe. Debug-only; kept out of the step
 * loop's file because it decodes every dtype and buffer shape.
 */
#include "engine_priv.h"

#include "build/rad_build.h"
#include "device/device.h"
#include "rad_plugin.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

namespace rad {

/* ================================================================== the buffer probe
 *
 * RADIANCE_DUMP_BUF names declared buffers (comma separated); every step prints the first row of
 * each: mean, largest magnitude and first four elements. RADIANCE_DUMP_HEAD=<n> adds the first n
 * elements verbatim (integers included); RADIANCE_DUMP_TOPK=<k> adds the k largest. A name may
 * carry a row count, `frame.mix:100`, to read that many rows as one (a code-major plane holds one
 * code a row). RADIANCE_DUMP_DIR=<dir> also writes each read's raw bytes to
 * <dir>/<step>-<phase>-<name>.bin, for a comparison against another implementation's tensors.
 *
 * It shows tensors inside a step, which neither rad-kbench (one op at a time) nor --debug-graph
 * (what was declared) can. An environment variable at -vv only, because it adds a device-to-host
 * copy and a synchronisation per step. */

/* A positive integer from the environment, clamped to `hi`, or `dflt` when unset or not one. */
static int64_t env_count(const char* name, int64_t dflt, int64_t hi) {
    const char* s = std::getenv(name);
    const long long v = (s && *s) ? std::strtoll(s, nullptr, 10) : 0;
    return v > 0 ? std::min<int64_t>(v, hi) : dflt;
}

/* Bytes of a row to read back. 8 KiB covers an ordinary hidden state; a concatenated row (the
 * DFlash2 tap buffer is five hidden states wide) needs more, set with RADIANCE_DUMP_BYTES. */
static int64_t dump_cap() {
    static const int64_t cap = env_count("RADIANCE_DUMP_BYTES", 8192, INT64_MAX);
    return cap;
}

/* The floating types, which the summary and the top k decode; index planes have no meaningful
 * mean. */
static bool is_float(uint32_t dtype) {
    return dtype == RAD_F32 || dtype == RAD_F16 || dtype == RAD_BF16 || dtype == RAD_F8E4M3 ||
           dtype == RAD_F8E5M2;
}

static void dump_row(int step, const char* phase, const char* name, uint32_t dtype,
                     const std::vector<uint8_t>& raw) {
    const int bits = rad_dtype_bits(dtype);
    const int64_t cnt = bits > 0 ? (int64_t)raw.size() * 8 / bits : 0;
    if (!cnt) return;
    const bool fl = is_float(dtype);
    double sum = 0, amax = 0;
    /* Count non-finite values and the first one's index: a single NaN would poison the mean, and
     * where the garbage starts often locates the bug. */
    int64_t bad = 0, bad_first = -1;
    float first[4] = {0, 0, 0, 0};
    for (int64_t k = 0; fl && k < cnt; ++k) {
        const float v = rad_load_f32(raw.data(), dtype, k);
        if (!std::isfinite(v)) { if (bad_first < 0) bad_first = k; ++bad; }
        else { sum += v; amax = std::max(amax, (double)std::fabs(v)); }
        if (k < 4) first[k] = v;
    }
    /* FNV-1a over the raw bytes, to tell whether two runs match exactly; the mean cannot (one logit
     * moving by 1e-3 shifts a vocabulary's mean by 1e-8). */
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < raw.size(); ++i) { h ^= raw[i]; h *= 1099511628211ull; }
    const int64_t good = cnt - bad;
    if (const char* dir = std::getenv("RADIANCE_DUMP_DIR"); dir && *dir) {
        const std::string path = fmt("%s/%d-%s-%s.bin", dir, step, phase, name);
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(raw.data(), 1, raw.size(), f);
            std::fclose(f);
        }
    }
    RAD_TRACE("step %d%-8s %-28s %-5s n=%lld  hash %016llx  mean %.6f  amax %.5f  "
              "nonfinite %lld%s  [%.6f %.6f %.6f %.6f]",
              step, phase, name, rad_dtype_name(dtype), (long long)cnt, (unsigned long long)h,
              good > 0 ? sum / (double)good : 0.0, amax, (long long)bad,
              bad ? fmt(" from %lld", (long long)bad_first).c_str() : "",
              (double)first[0], (double)first[1], (double)first[2], (double)first[3]);

    /* ---- the first n elements, in the row's own type ----------------------------------------
     *
     * RADIANCE_DUMP_HEAD=<n> prints n elements verbatim, including integer planes, which the
     * summary above cannot decode. After sample_merge_topk, `sampler.cand_idx` holds global token
     * ids and `sampler.cand_val` their logits, descending: the engine's next-token distribution,
     * assembled across vocabulary shards. Renormalised over a fixed candidate set the full
     * vocabulary's partition function cancels, so these logits alone give a top-K KL divergence
     * against another engine's top-K logprobs. */
    static const int64_t head = env_count("RADIANCE_DUMP_HEAD", 0, 4096);
    if (head > 0) {
        const int64_t n = std::min<int64_t>(head, cnt);
        std::string hl;
        for (int64_t k = 0; k < n; ++k) {
            if (dtype == RAD_I32 || dtype == RAD_U32) {
                uint32_t u; std::memcpy(&u, raw.data() + k * 4, 4);
                hl += dtype == RAD_I32 ? fmt(" %d", (int)(int32_t)u) : fmt(" %u", (unsigned)u);
            } else if (fl) {
                hl += fmt(" %.6f", (double)rad_load_f32(raw.data(), dtype, k));
            } else {
                hl += " ?";
            }
        }
        RAD_TRACE("step %d%-8s %-28s head%s", step, phase, name, hl.c_str());
    }

    /* ---- the top k of the row ---------------------------------------------------------------
     *
     * RADIANCE_DUMP_TOPK=<k> adds a `topk` line per row: the k largest values and their indices.
     * It grades a lossy change from a serving engine: N distinct prompts at max_tokens = 1 give
     * every build identical inputs at one row each, and the top of that row compares next-token
     * distributions (top-1 agreement, rank displacement, the first-to-second logit gap).
     * Teacher-forced scoring is the KL mode's (core/kld.h). A generated continuation is not
     * evidence: greedy divergence is chaotic, so one continuation cannot rank two formats. */
    static const int64_t topk = env_count("RADIANCE_DUMP_TOPK", 0, 64);
    if (topk <= 0 || !fl) return;
    /* A partial selection, not a sort: n is the whole vocabulary and this runs on the step path. */
    std::vector<std::pair<float, int64_t>> best;
    best.reserve((size_t)topk + 1);
    for (int64_t k = 0; k < cnt; ++k) {
        const float v = rad_load_f32(raw.data(), dtype, k);
        if (!std::isfinite(v)) continue;
        if ((int64_t)best.size() < topk) {
            best.emplace_back(v, k);
            std::push_heap(best.begin(), best.end(), std::greater<>());
        } else if (v > best.front().first) {
            std::pop_heap(best.begin(), best.end(), std::greater<>());
            best.back() = { v, k };
            std::push_heap(best.begin(), best.end(), std::greater<>());
        }
    }
    std::sort(best.begin(), best.end(), std::greater<>());
    std::string line;
    for (const auto& b : best) line += fmt(" %lld:%.5f", (long long)b.second, (double)b.first);
    RAD_TRACE("step %d%-8s %-28s topk%s", step, phase, name, line.c_str());
}

/* Whether `name` matches `want`: the whole name or its trailing scope components ("rank0.h"
 * matches "h"), never a bare suffix ("moe_logits" must not match "logits"). */
static bool answers_to(const std::string& name, const std::string& want) {
    if (name.size() < want.size() ||
        name.compare(name.size() - want.size(), want.size(), want) != 0) return false;
    return name.size() == want.size() || name[name.size() - want.size() - 1] == '.';
}

/* Copy `n` bytes at `ptr` back and print them as one row. */
static bool dump_at(Ctx& ctx, int step, const char* phase, const std::string& name,
                    uint32_t dtype, const void* ptr, int64_t n) {
    std::vector<uint8_t> raw((size_t)n);
    if (n <= 0 || rad_memcpy_async(raw.data(), ptr, n, ctx.stream()) < 0) return false;
    rad_stream_sync(ctx.stream());
    dump_row(step, phase, name.c_str(), dtype, raw);
    return true;
}

/* A buffer is read where it lives after the pass, and the arena shares a transient's bytes with
 * every buffer whose lifetime does not overlap it. So a dump after a drafter's context pass shows
 * a tap buffer (dflash_src) already overwritten by a later one in that pass (df_c), and only the
 * step loop's dump, after the trunk, shows the taps as written. A PERSIST buffer is never shared. */
void dump_buffers(Ctx& ctx, int step, const char* phase) {
    if ((int)log_level() < (int)Log::Trace) return;
    static const char* const names = std::getenv("RADIANCE_DUMP_BUF");
    if (!names || !*names) return;
    const Program& p = *ctx.program();
    for (const char* s = names; *s; ) {
        const char* e = s;
        while (*e && *e != ',') ++e;
        std::string want(s, (size_t)(e - s));
        s = *e ? e + 1 : e;
        int64_t rows = 1;
        if (const size_t colon = want.find(':'); colon != std::string::npos) {
            rows = std::max<long long>(1, std::strtoll(want.c_str() + colon + 1, nullptr, 10));
            want.resize(colon);
        }
        if (want.empty()) continue;

        /* Weights match the same names, to check that the loader put the right checkpoint tensor
         * in each declared slot (a wrong name map still runs, but means nothing). */
        bool done = false;
        for (size_t i = 1; i < p.weights.size() && !done; ++i) {
            const WeightInfo& wi = p.weights[i];
            if (!answers_to(wi.name, want)) continue;
            void* ptr = ctx.weight_ptr((rad_weight)i);
            if (!ptr) continue;
            /* A rank-1 weight (a norm gain, A_log, dt_bias) is one row; dividing by shape[0] would
             * show a single element. */
            const int64_t row = wi.decl.rank >= 2
                                    ? wi.stored_bytes / (wi.decl.shape[0] ? wi.decl.shape[0] : 1)
                                    : wi.stored_bytes;
            const int64_t n = std::min<int64_t>(row > 0 ? row : wi.stored_bytes, dump_cap());
            done = dump_at(ctx, step, phase, wi.name, wi.decl.dtype, ptr, n);
        }

        for (size_t i = 1; i < p.buffers.size() && !done; ++i) {
            const BufferInfo& b = p.buffers[i];
            if (!answers_to(b.name, want)) continue;
            void* ptr = ctx.buf_ptr((rad_buf)i);
            if (!ptr) continue;
            const int64_t row = b.decl.rank >= 1 ? b.bytes / (b.decl.shape[0] ? b.decl.shape[0] : 1)
                                                 : b.bytes;
            done = dump_at(ctx, step, phase, b.name, b.decl.dtype, ptr,
                           row <= dump_cap() ? std::min<int64_t>(row * rows, b.bytes) : dump_cap());
        }
    }
}

}  /* namespace rad */
