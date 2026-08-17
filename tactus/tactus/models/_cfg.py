"""Config normalisation shared by the model builders.

Kept dependency-free (no torch, no imports from the rest of the package) so that
``tactus.models.heads`` and ``tactus.models.eeg.base`` can both use it without an
import cycle through ``tactus.models.__init__``.

The builders accept several spellings of the same option because the trainer, the
YAML configs and the model classes were written against slightly different
vocabularies (``d_embed`` / ``d_out`` / ``embed_dim``; ``subject_conditioning`` /
``subject_cond``).  Normalising here means neither side has to change.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set

__all__ = [
    "to_dict",
    "unwrap_section",
    "accepted_kwargs",
    "filter_kwargs",
    "apply_aliases",
]


def to_dict(cfg: Any) -> Dict[str, Any]:
    """Normalise dict / Namespace / dataclass / OmegaConf into a plain ``dict``."""
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    try:  # OmegaConf is optional
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            return dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        pass
    if hasattr(cfg, "__dataclass_fields__"):
        from dataclasses import asdict

        return dict(asdict(cfg))
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    raise TypeError(f"cannot interpret config of type {type(cfg).__name__}")


def unwrap_section(cfg: Dict[str, Any], *section_names: str) -> Dict[str, Any]:
    """If ``cfg`` is a whole run config, descend into the first matching sub-section."""
    for name in section_names:
        sub = cfg.get(name)
        if isinstance(sub, Mapping) or (
            sub is not None and not isinstance(sub, (str, bytes, int, float, bool)) and hasattr(sub, "__dict__")
        ):
            return to_dict(sub)
    return cfg


def accepted_kwargs(target: Any) -> Optional[Set[str]]:
    """Every named keyword ``target`` can accept, unioned across its MRO.

    Walking the MRO matters: the encoder subclasses declare ``**base`` and forward it
    to :class:`~tactus.models.eeg.base.EEGEncoder`, so inspecting only the subclass
    signature would report "accepts anything" and let genuinely bogus keys through to a
    ``TypeError`` deep in the constructor.

    Returns ``None`` when the target is a plain function with ``**kwargs`` (i.e. it
    really does accept anything).
    """
    if inspect.isclass(target):
        names: Set[str] = set()
        for klass in inspect.getmro(target):
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            try:
                sig = inspect.signature(init)
            except (TypeError, ValueError):  # pragma: no cover - C-level __init__
                continue
            names |= {
                n
                for n, p in sig.parameters.items()
                if n != "self"
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            }
        return names
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):  # pragma: no cover
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return None
    return {n for n in sig.parameters if n != "self"}


def filter_kwargs(target: Any, kwargs: Dict[str, Any], context: str) -> Dict[str, Any]:
    """Drop keys ``target`` cannot accept, warning once about the whole set."""
    accepted = accepted_kwargs(target)
    if accepted is None:
        return kwargs
    kept = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        warnings.warn(
            f"{context}: ignoring config keys not accepted by "
            f"{getattr(target, '__name__', target)}: {dropped}",
            RuntimeWarning,
            stacklevel=3,
        )
    return kept


def apply_aliases(cfg: Dict[str, Any], aliases: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    """Rename alias keys onto canonical ones, in place.

    ``aliases`` maps ``canonical -> (alias, alias, ...)`` in priority order.  An
    explicit canonical key always wins; aliases are consumed (popped) either way, so
    they cannot leak through to a constructor that would reject them.
    """
    for canonical, alts in aliases.items():
        for alt in alts:
            if alt in cfg:
                value = cfg.pop(alt)
                if canonical not in cfg:
                    cfg[canonical] = value
    return cfg
