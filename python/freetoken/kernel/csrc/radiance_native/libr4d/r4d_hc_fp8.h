// r4d_hc_fp8.h -- stored row geometry of hc_read's two mixing matrices at E4M3. Pure C++ (no HIP,
// no r4d.h), so the host layout hook (r4d_fp8_layout.cpp), the kernels (r4d_hc_bf16.hip) and the
// selftest share one definition.
//
// A stored row is [K E4M3 codes][K/G f32 scales][pad to 16 bytes], rows back to back. As in
// r4d_rad_layout_fp8_logits, the scales ride in their row, so a kernel finds them at a fixed
// offset from its codes.
//
// The group size is set by the kernel that reads the row, not by tuning:
//
//   mix_down  [lowrank rows, K = hc*n]   G = 128. A wave walks the row in 256-element steps with
//             eight codes a lane, so each lane's eight sit in one group and a step needs two
//             scales a wave.
//   mix_up    [hc*n rows, K = lowrank]   G = K/4. Four threads share a row, each taking one
//             contiguous quarter (80 codes at lowrank 320), so a thread's quarter is its group and
//             it applies one scale after its partial sum. lowrank must be a multiple of 64 so a
//             quarter splits into sixteen-code loads.
//
// Both are finer than the GEMMs' 128 x 128 blocks on purpose: these rows meet a normalised stream
// where every element matters, and a per-row group costs one f32 per G codes (3.1% of mix_down's
// bytes, 5% of mix_up's).
#pragma once
#include <cstdint>

namespace r4d_hc8 {

constexpr int64_t kDownGroup = 128;

// constexpr rather than inline so device code can call them: a constexpr function is implicitly
// host and device.
constexpr int64_t down_group(int64_t /*K*/) { return kDownGroup; }
constexpr int64_t up_group(int64_t K) { return K / 4; }

// Whether K can be stored with group G: the kernels load sixteen codes at a time, and a group must
// be a whole number of those.
constexpr bool group_ok(int64_t K, int64_t G) { return G > 0 && G % 16 == 0 && K % G == 0; }

// Bytes one stored row occupies, and where its pad starts. Rows start 16-byte aligned so every
// code load is a uint4 or uint2 at a 16- or 8-byte boundary.
constexpr int64_t row_used(int64_t K, int64_t G) { return K + 4 * (K / G); }
constexpr int64_t row_bytes(int64_t K, int64_t G) { return (row_used(K, G) + 15) & ~(int64_t)15; }

}  // namespace r4d_hc8
