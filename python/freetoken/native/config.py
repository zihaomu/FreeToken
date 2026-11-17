from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NativeConfig:
    model: str
    worker: str
    runtime_home: str
    tensor_parallel: int = 2
    max_context: int = 34816
    prefill_chunk: int = 2048
    mtp_depth: int = 0
    kv_dtype: str = "i8"
    wire: str = "exact"
    host_pool_mib: int = 12288
    headroom_mib: int = 96
    prefix_cache: bool = False
    ngram_share_dir: str = ""
    visible_devices: str | None = None
    expected_pcis: tuple[str, ...] = ()
    gpu_memory_mib: int = 0
    prefix_cache_mib: int = 0

    def __post_init__(self) -> None:
        if self.tensor_parallel not in (1, 2):
            raise ValueError("native Qwen backend supports TP1 or TP2")
        if self.mtp_depth not in (0, 3):
            raise ValueError("native MTP depth must be 0 or 3")
        if self.kv_dtype not in ("i8", "bf16", "fp8"):
            raise ValueError("native KV dtype must be i8, bf16 or fp8")
        if self.wire not in ("exact", "wht6"):
            raise ValueError("native wire must be exact or wht6")
        if not 1 <= self.prefill_chunk <= self.max_context:
            raise ValueError("prefill chunk must be within the model context")
        if min(self.host_pool_mib, self.headroom_mib, self.gpu_memory_mib, self.prefix_cache_mib) < 0:
            raise ValueError("native memory budgets must be nonnegative")
        if self.visible_devices is not None and (
            len(self.visible_devices.split(",")) != self.tensor_parallel or
            len(self.expected_pcis) != self.tensor_parallel or self.visible_devices in ("", "-1")
        ):
            raise ValueError("native backend needs an explicitly selected GPU vector and matching reserved PCI addresses")
        for name in ("model", "worker"):
            if not Path(getattr(self, name)).is_file():
                raise ValueError(f"native {name} does not exist: {getattr(self, name)}")
        if not Path(self.runtime_home).is_dir():
            raise ValueError(f"native plugin home does not exist: {self.runtime_home}")

    def yaml(self) -> str:
        # JSON strings are YAML strings, including colons and user supplied paths.
        values = {
            "model": str(Path(self.model).resolve()), "tp": self.tensor_parallel,
            "tp-wire": self.wire,
            "max-num-seqs": 1, "max-model-len": self.max_context,
            "max-num-batched-tokens": self.prefill_chunk,
            "placement": "expert_tiered", "host-pool-mib": self.host_pool_mib,
            "kv-cache-dtype": self.kv_dtype, "num-speculative-tokens": self.mtp_depth,
        }
        if not self.prefix_cache:
            values["no-prefix-cache"] = True
        elif self.prefix_cache_mib:
            values["prefix-cache-mib"] = self.prefix_cache_mib
        if self.ngram_share_dir:
            values["ngram-placement"] = "ram=" + self.ngram_share_dir
        memory = (f"  gpu-memory-mib: {self.gpu_memory_mib}" if self.gpu_memory_mib
                  else f"  gpu-headroom-mib: {self.headroom_mib}")
        lines = ["server:", f"  radiance-home: {json.dumps(self.runtime_home)}",
                 "memory:", memory, "models:", "  freetoken-native:"]
        lines.extend(f"    {key}: {json.dumps(value)}" for key, value in values.items())
        return "\n".join(lines) + "\n"
