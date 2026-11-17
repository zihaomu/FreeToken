/* rowpool.h -- independent rows of a host loop spread over a pool of threads kept for the
 * process.
 *
 * A request's host work (a picture resampled, cut into patches) is a loop of rows that each write
 * only their own output. Starting threads for each such loop cost more than the rows: a resample
 * of a document page started eight threads per pass, about 32 a request, and their creation and
 * exit (stacks mapped, faulted and freed, per-thread library state set up) took twice the CPU of
 * the pixels. The pool's workers start once, on first use, and wait for loops.
 *
 * The calling thread takes rows too, so a loop never waits for a worker to wake, and concurrent
 * callers each get help: a worker serves the oldest loop that still has rows and fewer helpers than
 * it asked for. Rows are handed out one at a time from a shared counter, so which thread computes
 * a row varies; a loop whose rows are independent gives the same result however they fall.
 */
#pragma once

#include <cstdint>
#include <functional>

namespace rad {

/* Runs row(i) for every i in [0, rows), on the calling thread and at most `width - 1` pool workers,
 * and returns when all rows are done. `width` <= 1, or a loop of fewer than `min_rows` rows, runs
 * inline. row must not throw. */
void row_pool_run(int64_t rows, int width, const std::function<void(int64_t)>& row,
                  int64_t min_rows = 2);

/* Workers the pool may start: one fewer than the host's hardware threads, at most 15. */
int row_pool_workers();

}  // namespace rad
