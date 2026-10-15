"""Packed GGUF ops against gguf-py's CPU dequantization and PyTorch math."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel import ggml
from freetoken.third_party.gguf import GGMLQuantizationType as QType
from freetoken.third_party.gguf import GGML_QUANT_SIZES, quants

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
QUANT_TYPES = [QType[name] for name in (
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
    "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS",
    "MXFP4", "NVFP4", "Q1_0", "Q2_0",
)]


def _random_weights(qtype, shape):
    rng = np.random.default_rng(690 + int(qtype))
    block_size, block_bytes = GGML_QUANT_SIZES[qtype]
    blocks = rng.integers(0, 256, (int(np.prod(shape)) // block_size, block_bytes), dtype=np.uint8)
    if qtype == QType.MXFP4:
        blocks[:, 0] = rng.integers(121, 125, len(blocks))
    elif qtype == QType.NVFP4:
        blocks[:, :4] = rng.integers(40, 57, (len(blocks), 4))
    else:
        offset = {QType.Q2_K: block_bytes - 4, QType.Q3_K: block_bytes - 2, QType.Q6_K: block_bytes - 2}.get(qtype, 0)
        blocks[:, offset : offset + 2] = np.array([2**-10], dtype=np.float16).view(np.uint8)
        if qtype in (QType.Q4_1, QType.Q5_1, QType.Q4_K, QType.Q5_K, QType.Q2_K):
            offset = block_bytes - 2 if qtype == QType.Q2_K else 2
            blocks[:, offset : offset + 2] = np.array([2**-12], dtype=np.float16).view(np.uint8)
    packed = blocks.reshape(*shape[:-1], shape[-1] // block_size * block_bytes)
    if qtype in (QType.Q1_0, QType.Q2_0):
        bits = 1 if qtype == QType.Q1_0 else 2
        shifts = np.arange(0, 8, bits, dtype=np.uint8)
        codes = ((blocks[:, 2:, None] >> shifts) & ((1 << bits) - 1)).astype(np.float32)
        symbols = 2 * codes - 1 if bits == 1 else codes - 1
        reference = (symbols.reshape(-1, block_size) * blocks[:, :2].copy().view(np.float16)).reshape(shape)
    else:
        reference = quants.dequantize(packed, qtype)
    assert np.isfinite(reference).all()
    storage = torch.zeros(packed.size + packed.shape[-1], dtype=torch.uint8, device="cuda")
    w = storage[: packed.size].view(packed.shape)
    w.copy_(torch.from_numpy(packed))
    return w, torch.from_numpy(reference.astype(np.float32))


def _assert_product(actual, expected, dtype):
    rtol = 8e-3 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(actual.cpu(), expected.to(dtype), atol=2e-4, rtol=rtol)


def _q4_weights(shape, seed=690):
    values = np.random.default_rng(seed).normal(0, 0.1, shape).astype(np.float32)
    packed = quants.quantize(values, QType.Q4_0)
    reference = torch.from_numpy(quants.dequantize(packed, QType.Q4_0))
    # ggml's last MMQ tile may read an extra partial row.
    storage = torch.zeros(*packed.shape[:-2], packed.shape[-2] + 1, packed.shape[-1], dtype=torch.uint8, device="cuda")
    storage[..., :-1, :].copy_(torch.from_numpy(packed))
    return storage[..., :-1, :], reference


def _activation(shape, dtype, seed=19):
    rng = np.random.default_rng(seed)
    half = rng.integers(-127, 128, (*shape[:-1], shape[-1] // 32, 16))
    half[..., 0] = 127
    # Exactly representable q8_1 inputs with zero block sums isolate weight math.
    values = np.concatenate([half, -half], axis=-1).reshape(shape).astype(np.float32) / 64
    return torch.from_numpy(values).to("cuda", dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_q4_dequantize_and_rows(dtype):
    w, ref = _q4_weights((64, 256))
    actual = ggml.dequantize(w, int(QType.Q4_0), 256, dtype)
    torch.testing.assert_close(actual.cpu(), ref.to(dtype), atol=0, rtol=0)
    ids = torch.tensor([63, 0, 17, 17], dtype=torch.int32, device="cuda")
    rows = ggml.get_rows(w, int(QType.Q4_0), 256, ids, dtype)
    torch.testing.assert_close(rows.cpu(), ref[ids.cpu().long()].to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_q4_matrix_matches_cpu(dtype, batch):
    w, ref = _q4_weights((64, 256))
    x = _activation((batch, 256), dtype)
    actual = ggml.mul_mat(w, int(QType.Q4_0), x)
    expected = F.linear(x.cpu().float(), ref).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_q4_moe_routes_match_cpu(dtype, batch):
    w, ref = _q4_weights((4, 64, 256))
    x = _activation((batch, 256), dtype)
    ids = torch.tensor([[i % 4, (i + 1) % 4] for i in range(batch)], dtype=torch.int32, device="cuda")
    actual = ggml.mul_mat_id(w, int(QType.Q4_0), x, ids)
    expected = torch.einsum("tk, tjok -> tjo", x.cpu().float(), ref[ids.cpu().long()]).to(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_q4_shared_q8_matches_internal_quantization(dtype):
    w, _ = _q4_weights((64, 256))
    x = _activation((1, 256), dtype)
    expected = ggml.mul_mat(w, int(QType.Q4_0), x)
    actual = ggml.mul_mat(w, int(QType.Q4_0), x, xq=ggml.quantize_q8_1(x))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_q4_nondefault_stream():
    w, ref = _q4_weights((64, 256))
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = _activation((1, 256), torch.float32)
        actual = ggml.mul_mat(w, int(QType.Q4_0), x)
        copied = actual.clone()
    stream.synchronize()
    torch.testing.assert_close(copied.cpu(), F.linear(x.cpu(), ref), atol=2e-4, rtol=1e-3)


@pytest.mark.parametrize("qtype", QUANT_TYPES, ids=lambda t: t.name)
@pytest.mark.parametrize("dtype", DTYPES)
def test_quantized_dequantize_and_rows(qtype, dtype):
    w, ref = _random_weights(qtype, (64, 256))
    actual = ggml.dequantize(w, int(qtype), 256, dtype)
    torch.testing.assert_close(actual.cpu(), ref.to(dtype), atol=0, rtol=0)
    ids = torch.tensor([63, 0, 17, 17], dtype=torch.int32, device="cuda")
    rows = ggml.get_rows(w, int(qtype), 256, ids, dtype)
    torch.testing.assert_close(rows.cpu(), ref[ids.cpu().long()].to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize("qtype", QUANT_TYPES, ids=lambda t: t.name)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_quantized_matrix_matches_cpu(qtype, dtype, batch):
    w, ref = _random_weights(qtype, (64, 256))
    x = _activation((batch, 256), dtype)
    actual = ggml.mul_mat(w, int(qtype), x)
    _assert_product(actual, F.linear(x.cpu().float(), ref), dtype)


@pytest.mark.parametrize("qtype", QUANT_TYPES, ids=lambda t: t.name)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_quantized_moe_routes_match_cpu(qtype, dtype, batch):
    w, ref = _random_weights(qtype, (4, 64, 256))
    x = _activation((batch, 256), dtype)
    ids = torch.tensor([[i % 4, (i + 1) % 4] for i in range(batch)], dtype=torch.int32, device="cuda")
    actual = ggml.mul_mat_id(w, int(qtype), x, ids)
    expected = torch.einsum("tk, tjok -> tjo", x.cpu().float(), ref[ids.cpu().long()])
    _assert_product(actual, expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("glu", ["swiglu", "geglu", "swiglu_oai", "swiglu_clamp"])
def test_q4_fused_glu_and_weighted_sum(dtype, glu):
    w, ref = _q4_weights((4, 128, 256))
    x = _activation((1, 256), dtype)
    ids = torch.tensor([[3, 1]], dtype=torch.int32, device="cuda")
    products = torch.einsum("tk, tjok -> tjo", x.cpu().float(), ref[ids.cpu().long()])
    gate, up = products.chunk(2, dim=-1)
    limit = 0.2 if glu == "swiglu_clamp" else 0.0
    if glu == "swiglu":
        expected = F.silu(gate) * up
    elif glu == "geglu":
        expected = F.gelu(gate, approximate="tanh") * up
    elif glu == "swiglu_oai":
        gate, up = gate.clamp(max=7), up.clamp(-7, 7)
        expected = gate * torch.sigmoid(1.702 * gate) * (up + 1)
    else:
        expected = F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
    actual = ggml.mul_mat_id_glu(w, int(QType.Q4_0), x, ids, glu, limit)
    _assert_product(actual, expected, dtype)
    weights = torch.tensor([[0.25, 0.75]], device="cuda")
    total = ggml.moe_weighted_sum(actual, weights)
    reference = (actual.cpu().float() * weights.cpu().unsqueeze(-1)).sum(1).to(dtype)
    # The GPU may fuse multiply-add while the CPU reference rounds each product.
    tolerance = 1e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(total.cpu(), reference, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_q4_graph_replay_reads_new_input(dtype, batch):
    w, ref = _q4_weights((64, 256))
    x = _activation((batch, 256), dtype)
    ggml.mul_mat(w, int(QType.Q4_0), x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ggml.mul_mat(w, int(QType.Q4_0), x)
    updated = _activation((batch, 256), dtype, seed=91)
    x.copy_(updated)
    graph.replay()
    _assert_product(actual, F.linear(updated.cpu().float(), ref), dtype)


@pytest.mark.parametrize("qtype", [QType.F32, QType.F16, QType.BF16], ids=lambda t: t.name)
@pytest.mark.parametrize("dtype", DTYPES)
def test_float_storage_dequantize_and_rows(qtype, dtype):
    source = np.random.default_rng(690).normal(0, 0.1, (64, 256)).astype(np.float32)
    raw = quants.quantize(source, qtype)
    ref = torch.from_numpy(quants.dequantize(raw, qtype)).to(dtype)
    w = torch.from_numpy(raw.view(np.uint8).reshape(64, -1)).cuda()
    torch.testing.assert_close(ggml.dequantize(w, int(qtype), 256, dtype).cpu(), ref, atol=0, rtol=0)
    ids = torch.tensor([0, 63, 12], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(ggml.get_rows(w, int(qtype), 256, ids, dtype).cpu(), ref[ids.cpu().long()], atol=0, rtol=0)


def test_nvfp4_row_lookup_graph_replay():
    w, ref = _random_weights(QType.NVFP4, (64, 256))
    ids = torch.tensor([0, 63], dtype=torch.int32, device="cuda")
    ggml.get_rows(w, int(QType.NVFP4), 256, ids, torch.float32)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = ggml.get_rows(w, int(QType.NVFP4), 256, ids, torch.float32)
    ids.copy_(torch.tensor([12, 31], dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.testing.assert_close(actual.cpu(), ref[[12, 31]], atol=0, rtol=0)


def test_q4_repeated_calls_release_temporary_buffers():
    w, _ = _q4_weights((64, 256))
    x = _activation((17, 256), torch.bfloat16)
    for _ in range(3):
        ggml.mul_mat(w, int(QType.Q4_0), x)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(32):
        ggml.mul_mat(w, int(QType.Q4_0), x)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
