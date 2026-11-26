"""Qwen3.8 text TP: local GDN/QSA heads, replicated HC/PLE/indexer and split routed experts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from freetoken.distributed import DistributedInfo, get_tp_info
from freetoken.distributed.plan import BaseParallelPlan, ExpertGroup, ParallelArgs, TensorPartition

_ROWS = re.compile(r"model\.embed_tokens\.weight|lm_head\.weight|model\.layers\.\d+\.linear_attn\.(A_log|dt_bias)")
_COLS = re.compile(r"model\.layers\.\d+\.(linear_attn\.out_proj|self_attn\.o_proj|mlp\.shared_expert\.down_proj)\.weight")
_FUSED = re.compile(r"model\.layers\.\d+\.(linear_attn\.(in_proj|conv1d)|self_attn\.qkv_proj|mlp\.shared_expert\.gate_up_proj)\.weight")
_SPLIT_MODULE = re.compile(r"model\.layers\.\d+\.(linear_attn\.(in_proj.*|out_proj)|self_attn\.(qkv_proj|o_proj)|mlp\.shared_expert\.(gate_up_proj|down_proj))\.")


def dense_tp(config) -> int:
    plan = getattr(config, "parallel_plan", None)
    return plan.dense.size if plan is not None else 1


@dataclass
class Qwen4ExpParallelPlan(BaseParallelPlan):
    segments: dict[str, tuple[int, ...]] = field(kw_only=True)
    local_heads: tuple[int, int, int, int] = field(kw_only=True)

    def __post_init__(self):
        self._rows = TensorPartition(0, self.dense.rank, self.dense.size)
        self._cols = TensorPartition(1, self.dense.rank, self.dense.size)

    def shard(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if _ROWS.fullmatch(name):
            return self._rows.slice(tensor)
        if _COLS.fullmatch(name):
            return self._cols.slice(tensor)
        if match := _FUSED.fullmatch(name):
            parts = self.segments[match[1]]
            if tensor.shape[0] != sum(parts):
                raise ValueError(f"{name}: {tensor.shape[0]} rows do not match fused segments {parts}")
            return torch.cat([self._rows.slice(part) for part in tensor.split(parts)])
        if _SPLIT_MODULE.match(name):
            raise ValueError(f"Qwen3.8 TP splits unquantized dense projections only; got {name}")
        return tensor

    def bind(self, model, cache) -> None:
        if get_tp_info() != self.dense:
            raise ValueError(f"Qwen3.8 layers use {get_tp_info()}, expected {self.dense}")
        super().bind(model, cache)
        q, kv, k, v = self.local_heads
        for layer in model.model.layers.op_list:
            op = layer.linear_attn if layer._is_linear else layer.self_attn
            actual = (op.num_k_heads, op.num_v_heads) if layer._is_linear else (op.num_q, op.num_kv)
            expected = (k, v) if layer._is_linear else (q, kv)
            if actual != expected:
                raise ValueError(f"Qwen3.8 layer {layer._layer_id} has heads {actual}, expected {expected}")


def build_parallel_plan(config, args: ParallelArgs) -> Qwen4ExpParallelPlan:
    tp = args.tp_size
    if tp != 2:
        raise ValueError("Qwen3.8 text tensor parallelism currently supports --tp 2")
    if config.is_multimodal:
        raise ValueError("Qwen3.8 TP2 currently requires --text-model-only")
    if config.qwen4_args.gdn_out_heads_tiled:
        raise ValueError("Qwen3.8 GGUF TP2 is not supported; use the original safetensors checkpoint")
    g = config.linear_attention_group()
    dims = {"QSA query heads": config.num_qo_heads, "QSA KV heads": config.num_kv_heads,
            "GDN key heads": g.num_key_heads, "GDN value heads": g.num_value_heads,
            "vocabulary": config.vocab_size, "shared expert": config.shared_expert_intermediate_size,
            "routed expert": config.moe_intermediate_size}
    for what, width in dims.items():
        if width % tp:
            raise ValueError(f"Qwen3.8 cannot split {what} ({width}) over TP={tp}")
    # Quantization blocks and scales need their own partition rules; retain TP1 support for those releases.
    quant = config.quant
    if quant is not None:
        modules = ["model.embed_tokens", "lm_head"]
        for i in range(config.num_layers):
            pre = f"model.layers.{i}"
            leaves = (["linear_attn.in_proj_qkvz", "linear_attn.in_proj_ba", "linear_attn.out_proj"]
                      if config.is_linear_layer(i) else ["self_attn.qkv_proj", "self_attn.o_proj"])
            modules += [f"{pre}.{leaf}" for leaf in [*leaves, "mlp.shared_expert.gate_up_proj", "mlp.shared_expert.down_proj"]]
        for module in modules:
            if quant.scheme_for(module) is not None:
                raise ValueError(f"Qwen3.8 TP2 requires unquantized dense weights; {module} is quantized")
    kd, vd = g.num_key_heads * g.key_head_dim, g.num_value_heads * g.value_head_dim
    shared = config.shared_expert_intermediate_size
    return Qwen4ExpParallelPlan(
        args, DistributedInfo(args.rank, tp), ExpertGroup(tp, config.moe_intermediate_size),
        segments={"linear_attn.in_proj": (kd, kd, vd, vd, g.num_value_heads, g.num_value_heads),
                  "linear_attn.conv1d": (kd, kd, vd),
                  "self_attn.qkv_proj": (2 * config.num_qo_heads * config.head_dim,
                                         config.num_kv_heads * config.head_dim, config.num_kv_heads * config.head_dim),
                  "mlp.shared_expert.gate_up_proj": (shared, shared)},
        local_heads=(config.num_qo_heads // tp, config.num_kv_heads // tp, g.num_key_heads // tp, g.num_value_heads // tp),
    )
