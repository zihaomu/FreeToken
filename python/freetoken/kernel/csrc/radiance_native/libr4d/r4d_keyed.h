/* r4d_keyed.h -- device blocks a kernel keeps between launches, keyed by the buffer they serve.
 *
 * Several rows keep arrival counters that must be zero when a launch starts and that the launch
 * leaves zero: one block per (stream or device, output buffer), made at the first launch and
 * replayed by recorded passes. They cannot live in the scratch, which other ops overwrite. The
 * blocks are keyed by the output's address, so when the core frees the memory an output lived in
 * (an engine of a process that loads and unloads models goes away), the blocks for it are freed
 * too (rad_plugin_freed, abi/rad_abi.h); otherwise every load would leave its blocks behind.
 */
#pragma once
#include <cstddef>

namespace r4d {

/* Which row's blocks: rows keep blocks of different sizes and protocols for the same output, so
 * each has its own. */
enum class Keyed : int { SplitFp8 = 1, GdnConv, MxDecode, M2xDecode, MxGemv };

/* The zeroed block of `bytes` for (row, owner, out), made on first use. `owner` is the stream for
 * a row whose launches overlap across streams, or null for one block per device. Null when it
 * cannot be made; the caller then runs its unfused path.
 *
 * The lock is not held while a block is made. Zeroing it waits for the card, and under tensor
 * parallelism the card can be inside an all-reduce whose other half the peer rank has yet to issue;
 * if the peer's first launch then waited on this lock, neither rank would advance. */
unsigned* keyed_block(Keyed row, const void* owner, const void* out, size_t bytes);

/* State a row keeps under keys of its own, released when the memory [lo, hi) is freed. */
using FreedFn = void (*)(const char* lo, const char* hi);
/* Registers `f`; returns 0, so a file can register from a static initialiser. */
int on_freed(FreedFn f);

}  // namespace r4d
