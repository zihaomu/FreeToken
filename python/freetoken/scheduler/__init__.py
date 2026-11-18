"""Scheduler adapters and their CPU policy core."""
from importlib import import_module

__all__ = ["Scheduler", "SchedulerConfig"]
_EXPORTS = {"Scheduler": ".scheduler", "SchedulerConfig": ".config"}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
