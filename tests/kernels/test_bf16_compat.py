"""RDNA BF16 product rounding, with unchanged native arithmetic elsewhere."""
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

from freetoken.kernel import backend
from freetoken.kernel.triton import bf16_compat as compat


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    compat.needs_bf16_mul_workaround.cache_clear()
    yield
    compat.needs_bf16_mul_workaround.cache_clear()


def test_cuda_does_not_query_target_or_compile_probe(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("CUDA must not query the RDNA compiler probe")

    monkeypatch.setattr(backend, "is_rocm", lambda: False)
    monkeypatch.setattr(triton.runtime, "driver", SimpleNamespace(active=SimpleNamespace(get_current_target=unexpected)))
    monkeypatch.setattr(triton, "compile", unexpected)
    assert compat.needs_bf16_mul_workaround() is False


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950", "gfx1250"])
def test_unaffected_arch_skips_probe(monkeypatch, arch):
    target = GPUTarget("hip", arch, 64)
    monkeypatch.setattr(backend, "is_rocm", lambda: True)
    monkeypatch.setattr(triton.runtime, "driver", SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: target)))

    def unexpected(*args, **kwargs):
        raise AssertionError("unaffected architectures must not compile the probe")

    monkeypatch.setattr(triton, "compile", unexpected)
    assert compat.needs_bf16_mul_workaround() is False


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1151", "gfx1201"])
@pytest.mark.parametrize("broken", [False, True])
def test_probe_distinguishes_compiler_backports_and_caches(monkeypatch, arch, broken):
    target = GPUTarget("hip", arch, 32)
    monkeypatch.setattr(backend, "is_rocm", lambda: True)
    monkeypatch.setattr(triton, "__version__", "3.8.0")
    monkeypatch.setattr(triton.runtime, "driver", SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: target)))
    calls = []

    def compile_probe(source, **kwargs):
        calls.append(kwargs["target"])
        assert source.signature == {"a": "*bf16", "b": "*bf16", "out": "*bf16"}
        ir = "call @llvm.amdgcn.fdot2.bf16.bf16" if broken else "fmul float; explicit rtne"
        return SimpleNamespace(asm={"llir": ir})

    monkeypatch.setattr(triton, "compile", compile_probe)
    assert compat.needs_bf16_mul_workaround() is broken
    assert compat.needs_bf16_mul_workaround() is broken
    assert calls == [target]


@triton.jit
def _multiply_kernel(a, b, out, COUNT: tl.constexpr, WORKAROUND: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a + offsets, offsets < COUNT, 0)
    y = tl.load(b + offsets, offsets < COUNT, 0)
    tl.store(out + offsets, compat.bf16_mul_compat(x, y, WORKAROUND), offsets < COUNT)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("capture", [False, True])
def test_product_matches_cpu_rounding_reference(dtype, capture):
    count = 1 << 16
    gen = torch.Generator().manual_seed(11227)
    a_cpu = torch.randn(count, generator=gen).to(dtype)
    b_cpu = torch.randn(count, generator=gen).to(dtype)
    expected = (a_cpu.float() * b_cpu.float()).to(dtype)
    a, b = a_cpu.cuda(), b_cpu.cuda()
    out = torch.empty_like(a)
    workaround = dtype == torch.bfloat16 and compat.needs_bf16_mul_workaround()

    def run():
        _multiply_kernel[(triton.cdiv(count, 256),)](a, b, out, count, workaround, 256)

    run()
    if capture:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="requires ROCm")
def test_compile_only_probe_is_safe_during_graph_capture():
    a = torch.full((128,), 1.0078125, dtype=torch.bfloat16, device="cuda")
    b = a.clone()
    out = torch.empty_like(a)
    workaround = compat.needs_bf16_mul_workaround()
    _multiply_kernel[(1,)](a, b, out, 128, workaround, 128)
    compat.needs_bf16_mul_workaround.cache_clear()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        observed = compat.needs_bf16_mul_workaround()
        _multiply_kernel[(1,)](a, b, out, 128, observed, 128)
    graph.replay()
    assert observed == workaround
    expected = (a.cpu().float() * b.cpu().float()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_decode_single_product_rounds_to_nearest_even():
    from freetoken.kernel import fused_moe_decode_kernel_triton

    a = torch.zeros(1, 64, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(1, 32, 64, dtype=torch.bfloat16, device="cuda")
    a[0, 0] = 0.41015625
    b[0, :, 0] = 0.0400390625
    out = torch.empty(1, 1, 32, dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    weights = torch.ones(1, 1, dtype=torch.float32, device="cuda")
    fused_moe_decode_kernel_triton(
        a, b, out, weights, ids, False, False,
        {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "num_warps": 8}, torch.bfloat16,
    )
    # Only one nonzero product: reduction order cannot explain a one-ULP error.
    value = (torch.tensor(0.41015625) * torch.tensor(0.0400390625)).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), value.expand_as(out.cpu()), atol=0, rtol=0)
