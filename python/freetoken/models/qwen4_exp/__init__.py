"""Qwen3.8 Flash Next; CPU specifications do not import device layers."""
from importlib import import_module

_EXPORTS = {
    "parse_config": ".config",
    **{name: ".gguf" for name in (
        "gguf_mm_config", "iter_gguf_weights", "parse_gguf_config", "to_gguf",
    )},
    **{name: ".model" for name in (
        "Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration",
    )},
    **{name: ".weight" for name in (
        "ftw_side_files", "nvfp4_expert_spec", "iter_vision_weights", "iter_weights",
        "load_ple_table",
    )},
    "iter_expert_pieces": "freetoken.models.qwen3_5_moe.weight",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
