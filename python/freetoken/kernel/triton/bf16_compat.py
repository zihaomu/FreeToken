"""RDNA BF16 multiply rounding workaround for Triton #11227.

Upstream 3.6.0 through 3.8.0 release tags contain the truncating lowering.
https://github.com/triton-lang/triton/pull/11227 fixes it in 6b4b633; builds
containing that fix no longer need this workaround. Vendor backports cannot be
identified by version alone, so a compile-only probe detects the actual lowering.
CUDA and non-RDNA targets never compile the probe. The decision is cached for
FreeToken's one-process-per-GPU execution model.
"""
import functools

import triton
import triton.language as tl


@triton.jit
def _bf16_mul_probe(a, b, out):
    offsets = tl.arange(0, 128)
    tl.store(out + offsets, tl.load(a + offsets) * tl.load(b + offsets))


@functools.cache
def needs_bf16_mul_workaround() -> bool:
    from freetoken.kernel.backend import is_rocm

    if not is_rocm():
        return False
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or not str(target.arch).startswith(("gfx11", "gfx120")):
        return False

    from triton.compiler import ASTSource

    compiled = triton.compile(
        ASTSource(_bf16_mul_probe, signature={"a": "*bf16", "b": "*bf16", "out": "*bf16"}),
        target=target,
        options={"num_warps": 4},
    )
    return "llvm.amdgcn.fdot2.bf16.bf16" in compiled.asm["llir"]


@triton.jit
def bf16_mul_compat(a, b, WORKAROUND: tl.constexpr):
    if WORKAROUND and a.dtype == tl.bfloat16 and b.dtype == tl.bfloat16:
        # Triton #11227: preserve BF16 product precision, but round to nearest even.
        return (a.to(tl.float32) * b.to(tl.float32)).to(tl.bfloat16, fp_downcast_rounding="rtne")
    return a * b
