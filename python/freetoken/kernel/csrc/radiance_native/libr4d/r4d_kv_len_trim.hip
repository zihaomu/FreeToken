// r4d_kv_len_trim.hip - r4d_rad_kv_len_trim: a batch's per-sequence context lengths less `trim`.
//
// The 31B's prefill overlap runs attention per slice of a one-sequence step: slice A's rows are the
// step's first rows, so its attention must see the context ending at its own last row, not the
// step's (rows past it are not yet in the cache, and their positions are counted from the end).
// `trim` is the rows after the slice, a runtime count. A launch sees its op's parameters collapsed
// to the band's upper bound (RadArgs.p), so the count is the extent of operand `tail`, those rows'
// positions, whose values are never read; the ranged parameter only picks the band.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include <cstdint>

namespace {

__global__ __launch_bounds__(64) void kv_len_trim_kernel(const int32_t* __restrict__ in,
                                                         int32_t* __restrict__ out, int n,
                                                         int trim) {
    const int i = (int)threadIdx.x;
    if (i < n) {
        const int v = in[i] - trim;
        out[i] = v > 0 ? v : 0;
    }
}

}  // namespace

extern "C" int r4d_rad_kv_len_trim(const RadArgs* a, RadStream stream) {
    const RadTensor* in   = r4d_opt(a, 0);
    const RadTensor* tail = r4d_opt(a, 1);
    const RadTensor* out  = r4d_opt(a, 2);
    if (!in || !tail || !out || !in->data || !out->data) return R4D_NO(RAD_E_INVAL);
    if (in->dtype != RAD_I32 || out->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_numel(in);
    if (n < 1 || n > 64 || r4d_numel(out) < n) return R4D_NO(RAD_E_SHAPE);
    const long long trim = tail->rank ? r4d_numel(tail) : 0;
    if (trim < 0 || trim > 0x7fffffffLL) return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, kv_len_trim_kernel, dim3(1), dim3(64), 0, (const int32_t*)in->data,
                    (int32_t*)out->data, (int)n, (int)trim);
}
