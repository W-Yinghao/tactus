"""Training loop and CLI.

    from tactus.train import Trainer, train_fold

``tactus.train.run`` is the command-line entry point::

    python -m tactus.train.run --config configs/nice_infonce.yaml

Imports are lazy so that ``import tactus.train`` does not drag in torch on a
metadata-only machine.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Trainer", "TrainResult", "train_fold", "set_seed", "TrialView", "main"]

_LAZY = {
    "Trainer": "trainer",
    "TrainResult": "trainer",
    "train_fold": "trainer",
    "set_seed": "trainer",
    "TrialView": "trainer",
    "PseudoTrialView": "trainer",
    "StratifiedVideoBatchSampler": "trainer",
    "build_model": "trainer",
    "main": "run",
    "load_config": "run",
    "aggregate": "run",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial dispatch
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f"{__name__}.{mod}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(globals()) | set(_LAZY))
