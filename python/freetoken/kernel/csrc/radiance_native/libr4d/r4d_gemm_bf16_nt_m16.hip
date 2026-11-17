// r4d_gemm_bf16_nt_m16.hip -- forwarding entry, no kernel.
//
// This row owns the M <= 16 band (the MoE router gate) and serves it with the M <= 64 WMMA kernel,
// which is faster than a scalar split-K kernel across the band. The result is not bit-identical to
// a scalar kernel: the reduction tree differs, which moves emitted text at N = 48 (GDN a|b
// projection). A reduction order is not a quality bar, so that is accepted.
//
// The row is kept so the band stays explicitly owned rather than falling to whatever else claims it.
#include "r4d.h"
#include "r4d_gemm_args.h"

extern "C" int r4d_rad_gemm_bf16_m64(const RadArgs*, RadStream);

extern "C" int r4d_rad_gemm_bf16_m16(const RadArgs* a, RadStream s) {
    return r4d_rad_gemm_bf16_m64(a, s);
}
