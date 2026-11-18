"""CPU expert-policy configuration for the native backing provider."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class ExpertPlacementPlan:
    tensor_parallel: int
    gpu_memory_mib: int
    host_pool_mib: int
    metadata_sha256: str
    schema: int = 1
    owner: str = "freetoken.moe"
    algorithm: str = "lfu_decay"
    decay: float = 0.95
    hysteresis: float = 1.25
    min_gain: float = 1.5
    moves_per_class: int = 64
    credit_cap: float = 512.0
    admit_heat: float = 8.0
    direct_host_read: bool = True
    lend_kv: bool = True
    lend_arena: bool = True

    @classmethod
    def from_runtime(cls, config, composition):
        if config.tensor_parallel != len(composition.ranks):
            raise ValueError("placement and model rank counts differ")
        return cls(config.tensor_parallel, config.gpu_memory_mib, config.host_pool_mib,
                   composition.model_metadata_sha256)

    def to_bytes(self):
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()
