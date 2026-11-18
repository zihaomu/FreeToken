"""Model exports, loaded on demand so CPU specifications work without Torch."""
from importlib import import_module

_EXPORTS = {
    "BaseLLMModel": ".blocks",
    "get_model_class": ".register",
    "load_weight": ".weight",
    **{name: ".config" for name in (
        "AttentionGroupConfig", "BaseAttentionGroupConfig", "DSV4AttentionGroupConfig",
        "FullAttentionGroupConfig", "KVCacheGroupSpec", "LinearGatedDeltaGroupConfig",
        "ModelConfig", "RotaryConfig", "SWAAttentionGroupConfig",
    )},
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def create_model(model_config):
    from .register import get_model_class
    return get_model_class(model_config.architectures[0], model_config)


__all__ = [name for name in _EXPORTS if name != "get_model_class"] + ["create_model"]
