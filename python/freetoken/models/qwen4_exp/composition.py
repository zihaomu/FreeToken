"""Shared Qwen schedule and CPU construction of the native block graph.

Quantized block boundaries survive lowering. The device compiler expands blocks
into kernels without inventing layers, TP spans, or residual connections.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from freetoken.engine.backend import checked_int
from freetoken.engine.model_plan import StateGroupSpec


def layer_schedule(count: int, interval: int, explicit=None) -> tuple[str, ...]:
    checked_int(count, "layer count", minimum=1)
    checked_int(interval, "attention interval", minimum=2)
    aliases = {"indexed_attention": "full_attention", "qwen_sparse_attention": "full_attention"}
    if isinstance(explicit, str):
        explicit = explicit.split()
    result = tuple(aliases.get(t, t) for t in explicit) if explicit is not None else tuple(
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(count)
    )
    if len(result) != count or any(t not in ("full_attention", "linear_attention") for t in result):
        raise ValueError("invalid Qwen hybrid layer schedule")
    return result


def block_span(width: int, world: int, place: int, block: int = 128) -> tuple[int, int]:
    checked_int(world, "world", minimum=1, maximum=2)
    checked_int(place, "place", maximum=world - 1)
    if width <= 0 or width % block:
        raise ValueError("TP width must contain whole encoding blocks")
    base, rem = divmod(width // block, world)
    start = place * base + min(place, rem)
    return start * block, (start + base + (place < rem)) * block


@dataclass(frozen=True)
class BlockOp:
    op: str
    layer: int = -1


@dataclass(frozen=True)
class RankComposition:
    rank: int
    vocab: tuple[int, int]
    query_heads: tuple[int, int]
    kv_heads: tuple[int, int]
    gdn_key_heads: tuple[int, int]
    gdn_value_heads: tuple[int, int]
    expert_parity: tuple[tuple[int, int], ...]
    shared: tuple[int, int]
    protected: tuple[int, int]
    states: tuple[StateGroupSpec, ...]


@dataclass(frozen=True)
class ExpertOrder:
    layer: int
    regular: tuple[int, ...]
    protected: tuple[int, ...]


@dataclass(frozen=True)
class QwenComposition:
    model_metadata_sha256: str
    geometry: dict[str, int]
    layer_types: tuple[str, ...]
    ple_layer: int
    ranks: tuple[RankComposition, ...]
    trunk: tuple[BlockOp, ...]
    mtp_depth: int
    experts: tuple[ExpertOrder, ...]
    schema: int = 1
    owner: str = "freetoken.models.qwen4_exp"
    fusion: str = "hc_quant_moe_gather_p2p"
    prefill_schedule: str = "piece_pipeline"

    @classmethod
    def from_source(cls, source, *, world: int = 2, mtp_depth: int = 0):
        checked_int(world, "world", minimum=1, maximum=2)
        if source.architecture != "qwen4exp" or mtp_depth not in (0, 3):
            raise ValueError("unsupported native Qwen composition")

        def integer(name, fallback=None):
            for key in ("text_config." + name, name):
                if key in source.metadata:
                    return int(source.metadata[key])
            if fallback is not None:
                return int(source.metadata[fallback])
            raise ValueError(f"missing Qwen geometry: {name}")

        fields = {
            "n_layer": ("num_hidden_layers", "n_layers"),
            "n_embd": ("hidden_size", "n_embd"),
            "n_vocab": ("vocab_size", "n_vocab"),
            "n_head": ("num_attention_heads", "n_head"),
            "n_head_kv": ("num_key_value_heads", "n_head_kv"),
            "head_dim": ("head_dim", None), "hc": ("hc_count", None),
            "lowrank": ("hc_lowrank", None), "n_ff": ("moe_intermediate_size", "n_ff"),
            "n_expert": ("num_experts", "n_expert"),
            "top_k": ("num_experts_per_tok", "n_expert_used"),
            "shared_ff": ("shared_expert_intermediate_size", None),
            "gdn_k": ("linear_num_key_heads", None),
            "gdn_v": ("linear_num_value_heads", None),
            "gdn_head_k": ("linear_key_head_dim", None),
            "gdn_head_v": ("linear_value_head_dim", None),
            "conv_width": ("linear_conv_kernel_dim", None),
            "qsa_ratio": ("indexer_compress_ratio", None),
            "qsa_budget": ("indexer_budget", None),
            "qsa_head_dim": ("indexer_head_dim", None),
            "ple_ngram": ("ngram_size", None),
            "ple_conv_width": ("ple_conv_kernel_size", None),
        }
        g = {key: integer(*value) for key, value in fields.items()}
        if min(g.values()) <= 0:
            raise ValueError("nonpositive model geometry")
        interval = integer("full_attention_interval")
        types = layer_schedule(g["n_layer"], interval, source.metadata.get("text_config.layer_types"))
        ple = str(source.metadata.get("text_config.ple_layer_ids", "")).split()
        if len(ple) != 1:
            raise ValueError("native Qwen requires one PLE layer")
        ple_layer = int(ple[0]) - 1
        if not 0 <= ple_layer < len(types) or types[ple_layer] != "linear_attention":
            raise ValueError("PLE must precede a GDN layer")
        for key in ("n_vocab", "n_head", "n_head_kv", "gdn_k", "gdn_v"):
            if g[key] % world:
                raise ValueError(f"{key} cannot be evenly split at TP{world}")

        def heads(key, rank):
            size = g[key] // world
            return rank * size, (rank + 1) * size

        def states(rank):
            full = tuple(i for i, t in enumerate(types) if t == "full_attention")
            if mtp_depth:
                full += (g["n_layer"],)
            linear = tuple(i for i, t in enumerate(types) if t == "linear_attention")
            kh, vh = g["gdn_k"] // world, g["gdn_v"] // world
            return (
                StateGroupSpec("kv_attn", "kv", "fp8_e4m3", full,
                               (2, g["n_head_kv"] // world, g["head_dim"]), 4),
                StateGroupSpec("kv_gdn_state", "recurrent", "f32", linear,
                               (vh, g["gdn_head_v"], g["gdn_head_k"])),
                StateGroupSpec("kv_gdn_conv", "conv", "bf16", linear,
                               (2 * kh + vh, g["gdn_head_k"],
                                g["conv_width"] - 1 + mtp_depth)),
                StateGroupSpec("kv_qsa_tail", "recurrent", "bf16", full,
                               (1, g["qsa_ratio"] + mtp_depth, g["qsa_head_dim"])),
                StateGroupSpec("kv_qsa_bkey", "index", "bf16", full,
                               (2, 1, g["qsa_head_dim"] // (2 * g["qsa_ratio"])), g["qsa_ratio"]),
                StateGroupSpec("kv_ple_tok", "ple_history", "i32", (ple_layer,),
                               (1, 1, g["ple_ngram"] - 1 + mtp_depth)),
                StateGroupSpec("kv_ple_conv", "conv", "bf16", (ple_layer,),
                               (1, g["hc"] * g["n_embd"],
                                (g["ple_conv_width"] - 1) * g["ple_ngram"] + mtp_depth)),
            )

        ranks = tuple(RankComposition(
            rank, heads("n_vocab", rank), heads("n_head", rank), heads("n_head_kv", rank),
            heads("gdn_k", rank), heads("gdn_v", rank),
            tuple(block_span(g["n_ff"], world, (rank - parity) % world) for parity in range(2)),
            block_span(g["shared_ff"], world, rank),
            block_span(g["n_ff"], world, rank, block=64),
            states(rank),
        ) for rank in range(world))
        experts = []
        for layer in range(g["n_layer"] + bool(mtp_depth)):
            regular, protected = [], []
            for expert in range(g["n_expert"]):
                pair = [source.weights[f"blk.{layer}.ffn_{projection}_exps.{expert}.weight"]
                        for projection in ("gate_up", "down")]
                plain = [w.encoding.scheme == "plain" and len(w.encoding.planes) == 1
                         and w.encoding.planes[0].dtype == 3 for w in pair]
                if plain[0] != plain[1]:
                    raise ValueError("a protected expert needs both BF16 projections")
                (protected if plain[0] else regular).append(expert)
            if not regular or len(protected) > 8:
                raise ValueError("unsupported protected-expert class")
            experts.append(ExpertOrder(layer, tuple(regular), tuple(protected)))
        ops = [BlockOp("embed"), BlockOp("ple_ids"), BlockOp("hc_enter"), BlockOp("rope")]
        for layer, kind in enumerate(types):
            if layer == ple_layer:
                ops.append(BlockOp("ple", layer))
            ops += [BlockOp("hc_mix_read", layer),
                    BlockOp("qsa" if kind == "full_attention" else "gdn", layer),
                    BlockOp("hc_mix_write", layer), BlockOp("hc_ffn_read", layer),
                    BlockOp("moe", layer), BlockOp("hc_ffn_write", layer)]
        ops += [BlockOp("hc_output"), BlockOp("logits")]
        return cls(source.metadata_sha256, g, types, ple_layer, ranks, tuple(ops), mtp_depth,
                   tuple(experts))

    def to_bytes(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    @property
    def sha256(self):
        return hashlib.sha256(self.to_bytes()).hexdigest()
