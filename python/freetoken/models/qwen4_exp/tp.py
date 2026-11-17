"""Keep dense GEMM shapes stable while partitioning QSA/GDN heads and expert work."""

from __future__ import annotations

import os
import re
import torch

from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearReplicated, LinearRowParallel, OPList
from freetoken.layers.quantization import QuantKind, QuantScheme, get_quant_config


def supports_tp_row_fp8(module: str, scheme: QuantScheme) -> bool:
    allowed = module == "lm_head" or re.fullmatch(
        r"model\.layers\.\d+\.(linear_attn\.(in_proj_qkv|in_proj_z|in_proj_qkvz|out_proj)|self_attn\.o_proj)",
        module,
    ) is not None
    return (allowed and scheme.kind is QuantKind.FP8_TENSOR
            and scheme.weight.elem == "e4m3" and scheme.weight.group == (1, -1)
            and scheme.roles == frozenset(("weight", "weight_scale")))


def require_tp_row_fp8(module: str, scheme: QuantScheme) -> None:
    if not supports_tp_row_fp8(module, scheme):
        raise NotImplementedError(f"qwen4_exp TP=2 unsupported dense scheme: {module} {scheme}")
    defaults = {"GDN_INPUT": "r0w", "DENSE_OUTPUT": "r0w", "COLLECTIVE": "rccl", "GDN_BF16_BACKEND": "torch"}
    if any(os.environ.get("FREETOKEN_QWEN38_TP_" + key, value) != value for key, value in defaults.items()):
        raise NotImplementedError("qwen4_exp TP=2 row-FP8 requires the R0w TP options")


class _ColumnSlice(LinearReplicated):
    def __init__(self, source: LinearColParallelMerged):
        tp = get_tp_info()
        self._rank, self._size = tp.rank, tp.size
        segments = list(source.output_sizes)
        if source.prefix.endswith(".linear_attn.in_proj"):
            conv, value, b, a = segments
            key = (conv - value) // 2
            segments = [key, key, value, value, b, a]
        elif source.prefix.endswith(".linear_attn.in_proj_qkvz"):
            conv, value = segments
            key = (conv - value) // 2
            segments = [key, key, value, value]
        self._segments = tuple(n * tp.size for n in segments)
        quant = get_quant_config() if source.quant_method.cfg.scheme is not None else None
        super().__init__(source.full_input_size, source.full_output_size, has_bias=False,
                         output_sizes=list(self._segments), quant_config=quant, prefix=source.prefix)
        if self.quant_method.cfg.scheme != source.quant_method.cfg.scheme:
            raise ValueError(f"{source.prefix}: TP wrapper changed the quantization scheme")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full = self.forward_full(x)
        return torch.cat([
            part[..., self._rank * (n // self._size):(self._rank + 1) * (n // self._size)]
            for part, n in zip(full.split(self._segments, dim=-1), self._segments)
        ], dim=-1).contiguous()

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class _RowGather(LinearReplicated):
    def __init__(self, source: LinearRowParallel):
        self._size = get_tp_info().size
        self._comm = DistributedCommunicator()
        quant = get_quant_config() if source.quant_method.cfg.scheme is not None else None
        super().__init__(source.full_input_size, source.full_output_size, has_bias=False,
                         quant_config=quant, prefix=source.prefix)
        if self.quant_method.cfg.scheme != source.quant_method.cfg.scheme:
            raise ValueError(f"{source.prefix}: TP wrapper changed the quantization scheme")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = self._comm.all_gather(x.contiguous()).view(self._size, *x.shape)
        return super().forward(torch.cat(list(parts.unbind(0)), dim=-1))


def preserve_tp_dense_math(root: BaseOP) -> None:
    """HC amplifies a few BF16 rounding changes; retain TP1 dense dot-product shapes."""
    if get_tp_info().size == 1:
        return
    if isinstance(root, OPList):
        for child in root.op_list:
            preserve_tp_dense_math(child)
        return
    for name, child in list(vars(root).items()):
        if isinstance(child, (LinearColParallelMerged, LinearRowParallel)):
            if child.bias is not None:
                raise NotImplementedError("qwen4_exp TP=2 requires bias-free dense projections")
            if child.quant_method.cfg.scheme is not None:
                require_tp_row_fp8(child.prefix, child.quant_method.cfg.scheme)
            replacement = _ColumnSlice(child) if isinstance(child, LinearColParallelMerged) else _RowGather(child)
            setattr(root, name, replacement)
        elif isinstance(child, BaseOP):
            preserve_tp_dense_math(child)
