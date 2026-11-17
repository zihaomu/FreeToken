/* avx_check.h -- what rad-avx-check's translation units share: the run's settings, the value
 * comparison and the parallel rerun. */
#ifndef RAD_AVX_CHECK_H
#define RAD_AVX_CHECK_H

#include "avx_harness.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

/* One run: the oracle, the library under test, and what it is held to. */
struct Check {
    const Lib* ref = nullptr;
    const Lib* lib = nullptr;
    /* The library under test is libavx and is held to libavx's contract: every op libref
     * implements (avx_check.cpp lists those not yet implemented), and the file-backed gather of
     * embed_lookup_q with its init. Any other host library is checked op for op against libref, and what it does not
     * implement is reported rather than failed. */
    bool own = false;
    /* Team size of the parallel rerun (avx_par_force), or 0 for none. */
    int threads = 0;
    bool verbose = false;
    double tol_override = -1;
    /* Only this op, or every op when empty. */
    std::string only_op;
};

/* rad-kbench's error metric: the finiteness pattern is compared before any norm. */
struct Err {
    double max_abs = 0, max_rel = 0, rel_l2 = 0;
    int64_t nonfinite_gap = 0, n = 0;
};
Err compare(const float* got, const float* ref, int64_t n);

/* One output buffer of `n` elements of `dt`, the library's against the oracle's. Token and row ids
 * (i32, u32, i64) must match exactly: widened to f32, two ids above 2^24 can compare equal and an
 * off-by-one under the relative tolerance passes. Everything else is held to `tol` in rel_l2 with
 * the finiteness pattern first. Returns false with the reason in `why`; `rel` gets the error. */
bool same_values(uint32_t dt, const void* ref, const void* got, int64_t n, double tol,
                 double* rel, std::string* why);

/* The serial run's outputs against a rerun with every parallel region forced to `threads` team
 * threads (avx_par_force). libavx partitions only outer loops and never reduces across threads
 * (avx_common.h), so the rerun must reproduce every byte. `outs` are the output buffers and
 * `bytes` their sizes; `reset` restores every output and in-out buffer to its pre-launch bytes and
 * `launch` runs the library's kernel once. Returns false with the reason in `why`. */
bool parallel_rerun(const Lib& lib, int threads, const std::vector<uint8_t*>& outs,
                    const std::vector<size_t>& bytes, const std::function<void()>& reset,
                    const std::function<int()>& launch, std::string* why);

/* The hand-built cases (avx_check_hand.cpp). Returns false on any failure, having printed it. */
bool hand_cases(const Check& ck);

/* embed_lookup_q over a file-backed table in `dir`, with two ranks and the reader racing, for
 * `calls` calls whose ids are drawn from `seed` (avx_check_rows.cpp). Each table is written once
 * per process and kept mapped, so later calls reuse it. */
bool row_gather_case(const Lib& lib, const std::string& dir, int64_t calls, uint64_t seed,
                     bool verbose, bool bf16);
/* embed_lookup_q over a prefill whose following chunks are passed ahead, as the engine passes
 * them: every row read about once, every row exact (avx_check_rows.cpp). */
bool row_ahead_case(const Lib& lib, const std::string& dir, bool verbose);
/* embed_lookup_q's init with io_uring refused, and in this process (avx_check_rows.cpp). */
bool ngram_init_case(const Lib& lib);

#endif /* RAD_AVX_CHECK_H */
