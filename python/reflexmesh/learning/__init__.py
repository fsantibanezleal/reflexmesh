"""Versioned offline collection, fitting, calibration and export."""

from .dataset import assert_disjoint, collect, load_episodes


def __getattr__(name):
    if name not in {"train_classical", "train_recurrent", "train_transition"}:
        raise AttributeError(name)
    from . import train

    return getattr(train, name)


__all__ = [
    "assert_disjoint",
    "collect",
    "load_episodes",
    "train_classical",
    "train_recurrent",
    "train_transition",
]
