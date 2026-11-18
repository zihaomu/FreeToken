from importlib import import_module

__all__ = ["Engine", "EngineConfig", "ForwardOutput", "BatchSamplingArgs"]

_EXPORTS = {
    "Engine": ".engine",
    "EngineConfig": ".config",
    "ForwardOutput": ".engine",
    "BatchSamplingArgs": ".sample",
}


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
