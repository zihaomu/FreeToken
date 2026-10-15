"""Production GGUF methods checked against decoded CPU weights."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import freetoken.core as core
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from freetoken.layers.linear import LinearColParallelMerged
from freetoken.layers.moe import MoELayer
from freetoken.layers.quantization import GGUFConfig, NameMap
from freetoken.third_party.gguf import GGMLQuantizationType as QType
from freetoken.third_party.gguf import quants
from freetoken.utils.torch_utils import torch_dtype

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


@pytest.fixture(autouse=True)
def _runtime(monkeypatch):
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)


def _weights(shape, qtype=QType.Q4_0, seed=690):
    source = np.random.default_rng(seed).normal(0, 0.1, shape).astype(np.float32)
    packed = quants.quantize(source, qtype)
    reference = torch.from_numpy(quants.dequantize(packed, qtype))
    storage = torch.zeros(packed.size + packed.shape[-1], dtype=torch.uint8, device="cuda")
    weights = storage[: packed.size].view(packed.shape)
    weights.copy_(torch.from_numpy(packed))
    return weights, reference


def _input(batch, dtype):
    half = np.random.default_rng(batch).integers(-127, 128, (batch, 8, 16))
    half[..., 0] = 127
    source = np.concatenate([half, -half], axis=-1).reshape(batch, 256).astype(np.float32) / 64
    return torch.from_numpy(source).to("cuda", dtype)


def _config(types, name_map=None):
    return GGUFConfig(types, lambda name, _types: (name,), name_map=name_map)


def _assert_linear(actual, reference, dtype):
    torch.testing.assert_close(actual.cpu(), reference.to(dtype), atol=2e-4, rtol=8e-3 if dtype == torch.bfloat16 else 2e-3)


def _q4_q8_product(x, packed, mmq):
    x = x.reshape(*x.shape[:-1], -1, 32)
    scale = x.abs().amax(-1, keepdim=True) / 127
    codes = (x / scale.clamp_min(1e-30)).round().clamp(-127, 127)
    d8 = scale.squeeze(-1).half().float()
    sums = codes.sum(-1) * d8 if mmq else x.sum(-1).half().float()
    blocks = packed.reshape(*packed.shape[:-1], -1, 18)
    d4 = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1).float()
    qs = blocks[..., 2:].long()
    nibbles = torch.cat([qs & 15, qs >> 4], -1).float()
    dot = torch.einsum("tjobk,tjbk->tjob", nibbles, codes)
    # MMVQ uses the original fp16 block sum; RDNA WMMA centers Q4 codes before MMA.
    return ((dot * d8.unsqueeze(-2) - 8 * sums.unsqueeze(-2)) * d4).sum(-1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_gguf_embedding_and_tied_head(dtype, batch):
    cfg = _config({"embed.weight": int(QType.Q4_0)})
    with torch.device("cuda"), torch_dtype(dtype):
        embedding = VocabParallelEmbedding(128, 256, embed_scale=16.0, quant_config=cfg, prefix="embed")
        head = ParallelLMHead(128, 256, tie_word_embeddings=True, tied_embedding=embedding)
    packed, reference = _weights((128, 256))
    embedding.weight.zero_()
    embedding.weight[:128].copy_(packed)
    ids = torch.arange(batch, device="cuda")
    torch.testing.assert_close(embedding.forward(ids).cpu(), reference[:batch].to(dtype) * 16, atol=0, rtol=0)
    core._GLOBAL_CTX = SimpleNamespace(batch=SimpleNamespace(size=batch, is_prefill=False))
    x = _input(batch, dtype)
    _assert_linear(head.forward(x), F.linear(x.cpu().float(), reference), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_mixed_gguf_fused_linear(dtype, batch):
    types = [QType.Q4_0, QType.Q8_0, QType.Q4_0]
    cfg = _config({f"{name}.weight": int(t) for name, t in zip(["q", "k", "v"], types)},
                  NameMap(packed=(("qkv", ("q", "k", "v")),)))
    with torch.device("cuda"), torch_dtype(dtype):
        layer = LinearColParallelMerged(256, [64, 64, 64], False, quant_config=cfg, prefix="qkv")
    references = []
    for i, qtype in enumerate(types):
        packed, reference = _weights((64, 256), qtype, seed=690 + i)
        target = getattr(layer, "weight" if i == 0 else f"weight_{i}")
        target.zero_()
        target[:64].copy_(packed)
        references.append(reference)
    x = _input(batch, dtype)
    _assert_linear(layer.forward(x), F.linear(x.cpu().float(), torch.cat(references)), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 17])
def test_resident_gguf_moe(dtype, batch):
    cfg = _config({f"{name}.weight": int(QType.Q4_0) for name in ["gate", "up", "down"]},
                  NameMap(packed=(("experts", ("gate", "up", "down")),)))
    with torch.device("cuda"), torch_dtype(dtype):
        layer = MoELayer(4, 2, 256, 256, quant_config=cfg, prefix="experts")
    gu, gu_ref = _weights((4, 512, 256))
    down, _ = _weights((4, 256, 256), seed=691)
    layer._expert_banks = {"gate_up": gu, "down": down}
    x = _input(batch, dtype)
    ids = torch.tensor([[i % 4, (i + 1) % 4] for i in range(batch)], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]], device="cuda").expand(batch, -1).contiguous()
    core._GLOBAL_CTX = SimpleNamespace(batch=SimpleNamespace(is_prefill=batch > 1))
    actual = layer.routed_forward(x, weights, ids)
    products = torch.einsum("tk,tjok->tjo", x.cpu().float(), gu_ref[ids.cpu().long()])
    if batch > 1:
        products = products.to(dtype).float()
    gate, up = products.chunk(2, -1)
    intermediate = (F.silu(gate) * up).to(dtype).float()
    reference = _q4_q8_product(intermediate, down.cpu()[ids.cpu().long()], mmq=batch > 1).to(dtype).float()
    reference = (reference * weights.cpu().unsqueeze(-1)).sum(1)
    if dtype == torch.bfloat16 and batch > 1:
        # A BF16 rounding boundary before SiLU can change a subsequent q8 code.
        error = (actual.cpu().float() - reference.to(dtype).float()).norm() / reference.norm()
        assert error < 2e-3, error.item()
    else:
        _assert_linear(actual, reference, dtype)


@pytest.mark.skipif(torch.version.hip is None, reason="ROCm float expert reference checks")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("capture", [False, True])
def test_float_gguf_decode_matches_cpu_reference(dtype, batch, capture):
    from freetoken.layers.quantization.moe.base import ExpertView
    from freetoken.layers.quantization.moe.gguf import GgmlMoEKernel

    generator = torch.Generator().manual_seed(690)
    gate_up = (torch.randn(4, 512, 256, generator=generator) * 0.1).to(dtype).cuda()
    down = (torch.randn(4, 256, 256, generator=generator) * 0.1).to(dtype).cuda()
    x = (torch.randn(batch, 256, generator=generator) * 0.1).to(dtype).cuda()
    ids = torch.tensor([[3, 1]], device="cuda", dtype=torch.int32).repeat(batch, 1)
    weights = torch.tensor([[0.25, 0.75]], device="cuda").repeat(batch, 1)
    slots = torch.tensor([5, 1, 7, 3], device="cuda", dtype=torch.int32)
    cached_gate_up = torch.zeros(9, 512, 256, device="cuda", dtype=dtype)
    cached_down = torch.zeros(9, 256, 256, device="cuda", dtype=dtype)
    cached_gate_up[slots.long()] = gate_up
    cached_down[slots.long()] = down
    resident = ExpertView({"gate_up": gate_up, "down": down})
    cached = ExpertView({"gate_up": cached_gate_up, "down": cached_down})
    cached_ids = slots[ids.long()]
    qtype = int(QType.BF16 if dtype == torch.bfloat16 else QType.F16)

    def run(strategy, view, expert_ids):
        layer = SimpleNamespace(
            quant_method=SimpleNamespace(cfg=SimpleNamespace(strategy=strategy)),
            activation="gelu_tanh", alpha=1.0, limit=None, apply_router_weight_on_input=False,
        )
        return GgmlMoEKernel._apply_float(layer, x.clone(), weights, expert_ids, view, qtype, is_prefill=False)

    run("resident", resident, ids)
    run("offload", cached, cached_ids)
    if capture:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run("resident", resident, ids)
            cache_result = run("offload", cached, cached_ids)
        graph.replay()
    else:
        actual = run("resident", resident, ids)
        cache_result = run("offload", cached, cached_ids)
    cpu_ids = ids.cpu().long()
    gu = torch.einsum("tk,tjok->tjo", x.cpu().float(), gate_up.cpu()[cpu_ids].float()).to(dtype).float()
    gate, up = gu.chunk(2, -1)
    inter = (F.gelu(gate, approximate="tanh") * up).to(dtype).float()
    y = torch.einsum("tjk,tjok->tjo", inter, down.cpu()[cpu_ids].float())
    reference = (y * weights.cpu().unsqueeze(-1)).to(dtype).float().sum(1).to(dtype)
    torch.testing.assert_close(actual.cpu(), reference, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(cache_result.cpu(), reference, atol=2e-3, rtol=2e-2)
