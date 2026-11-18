"""Canonical model plans emitted by FreeToken, before device-kernel resolution."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Protocol

from .backend import ResourceHandle, checked_int


def _unique_pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("duplicate model parameter")
        value[key] = item
    return value


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int | str, ...]
    encoding: str
    kind: str = "temporary"

    def __post_init__(self):
        if not self.name or not self.encoding or self.kind not in (
            "input", "temporary", "persistent", "scratch", "state"
        ):
            raise ValueError("invalid model tensor declaration")
        if not self.shape or any(
            not ((type(x) is int and x >= 1) or (type(x) is str and x.isidentifier()))
            for x in self.shape
        ):
            raise ValueError("tensor dimensions must be positive or named dynamic dimensions")


@dataclass(frozen=True)
class WeightSlice:
    source_name: str
    binding_name: str
    rows: tuple[int, int]
    columns: tuple[int, int]
    encoding_id: str

    def __post_init__(self):
        if not self.source_name or not self.binding_name or not self.encoding_id:
            raise ValueError("weight source, binding and encoding are required")
        for start, end in (self.rows, self.columns):
            checked_int(start, "slice start")
            checked_int(end, "slice end", minimum=1)
            if start >= end:
                raise ValueError("empty weight slice")


@dataclass(frozen=True)
class StateGroupSpec:
    name: str
    kind: str
    encoding: str
    layer_ids: tuple[int, ...]
    local_shape: tuple[int, ...]
    block_tokens: int = 0

    def __post_init__(self):
        if not self.name or not self.encoding or self.kind not in (
            "kv", "index", "recurrent", "conv", "ple_history", "mtp"
        ):
            raise ValueError("invalid state group")
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("state layers must be unique and nonempty")
        for value in self.layer_ids:
            checked_int(value, "layer id")
        if not self.local_shape:
            raise ValueError("state shape must be nonempty")
        for value in self.local_shape:
            checked_int(value, "local state dimension", minimum=1)
        checked_int(self.block_tokens, "state block size")


@dataclass(frozen=True)
class OpSpec:
    name: str
    op: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    parameters_json: str = "{}"
    fusion_group: str = ""

    def __post_init__(self):
        if not self.name or not self.op:
            raise ValueError("model op names are required")
        value = json.loads(self.parameters_json, object_pairs_hook=_unique_pairs)
        if not isinstance(value, dict):
            raise ValueError("op parameters must be an object")
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "parameters_json", canonical)


@dataclass(frozen=True)
class ModelPlan:
    model_identity: str
    definition: str
    rank: int
    world_size: int
    phase: str
    tensors: tuple[TensorSpec, ...]
    weights: tuple[WeightSlice, ...]
    states: tuple[StateGroupSpec, ...]
    ops: tuple[OpSpec, ...]
    outputs: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self):
        checked_int(self.world_size, "world size", minimum=1, maximum=2)
        checked_int(self.rank, "rank", maximum=self.world_size - 1)
        if self.schema_version != 1 or not self.model_identity or not self.definition:
            raise ValueError("invalid model plan identity")
        if self.phase not in ("prefill", "decode", "verify", "mtp_draft", "mtp_history"):
            raise ValueError("unsupported model plan phase")
        tensors = {t.name: t for t in self.tensors}
        if len(tensors) != len(self.tensors):
            raise ValueError("duplicate tensor declaration")
        bindings = [w.binding_name for w in self.weights]
        if len(set(bindings)) != len(bindings) or set(bindings) & tensors.keys():
            raise ValueError("duplicate weight binding")
        if len({s.name for s in self.states}) != len(self.states):
            raise ValueError("duplicate state group")
        available = set(bindings) | {
            name for name, tensor in tensors.items()
            if tensor.kind in ("input", "persistent", "state")
        }
        names = set()
        for op in self.ops:
            if op.name in names:
                raise ValueError("duplicate op name")
            names.add(op.name)
            if set(op.reads) - available or set(op.writes) - tensors.keys():
                raise ValueError(f"{op.name}: use before definition or undeclared output")
            available.update(op.writes)
        if not self.ops or not self.outputs or set(self.outputs) - available:
            raise ValueError("model outputs must have been defined")

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()


class StateStorage(Protocol):
    """Storage adapters perform operations after the scheduler grants ownership."""

    def reserve(self, specs: tuple[StateGroupSpec, ...], capacity: int) -> ResourceHandle:
        ...

    def snapshot(self, state: ResourceHandle, cursor: int) -> ResourceHandle:
        ...

    def restore(self, state: ResourceHandle, snapshot: ResourceHandle) -> None:
        ...

    def release_after(self, handle: ResourceHandle, fence: ResourceHandle) -> None:
        ...
