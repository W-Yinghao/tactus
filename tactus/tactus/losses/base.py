"""TACTUS loss package -- core contract, registry, and shared numerics.

This module is the *stable* half of the user's swap point.  A new contrastive
algorithm is added by writing ONE file in ``tactus/losses/`` that subclasses
:class:`ContrastiveLoss` and decorates it with :func:`register_loss`.  Nothing
else in the codebase needs to change.

The forward contract (frozen -- do not change without changing every caller)::

    class MyLoss(ContrastiveLoss):
        def forward(self, z_eeg, z_vid, meta) -> dict:
            ...
            return {"loss": <0-dim tensor>, "logs": {<str>: <float>}}

    z_eeg : (B, D) float, L2-normalized EEG embedding
    z_vid : (B, D) float, L2-normalized *projected* frozen video embedding of the
            trial's condition
    meta  : dict[str, Tensor], each (B,), keys listed in :data:`META_KEYS`

Numerical policy
----------------
Every loss in this package must satisfy three invariants, because TACTUS batches
are pathological by construction (a dense 80x360x8 design means duplicate
conditions land in the same batch constantly):

1. **No NaN when a row has no valid positive or no valid negative.**  Rows that
   cannot form a term are dropped from the mean; if *no* row is valid the loss
   is a graph-connected zero (see :meth:`ContrastiveLoss._zero_loss`) so
   ``.backward()`` still populates ``.grad`` for every parameter.
2. **Masking is done with a large finite negative**, not ``-inf``.  A row whose
   entries are all ``-inf`` produces NaN out of ``log_softmax``; a row that is
   all ``-1e9`` produces a finite uniform distribution which we then discard via
   the validity mask.  :func:`mask_value` picks a fill that underflows to zero
   in ``exp`` for the given dtype without overflowing fp16.
3. **Logs are plain Python floats**, already detached.  Never put a live tensor
   in ``logs`` -- the trainer accumulates them across steps and would leak the
   graph.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "META_KEYS",
    "ID_META_KEYS",
    "CONTINUOUS_META_KEYS",
    "ContrastiveLoss",
    "TemperatureMixin",
    "LOSS_REGISTRY",
    "register_loss",
    "get_loss",
    "list_losses",
    "build_loss",
    "get_meta",
    "pairwise_eq",
    "safe_normalize",
    "mask_value",
    "masked_log_softmax",
    "masked_logsumexp",
    "safe_pdist",
    "to_float",
    "make_dummy_batch",
]

# --------------------------------------------------------------------------- #
# meta contract
# --------------------------------------------------------------------------- #

#: Integer-valued meta keys (converted to ``torch.long`` on access).
ID_META_KEYS = (
    "video_id",  # 1..90
    "condition_id",  # 0..359, == (video_id - 1) * 4 + orientation
    "orientation",  # 0=original, 1=horflip, 2=vertflip, 3=horvertflip
    "subject_id",  # 1..80
    "sequence_id",  # 1..32, WITHIN subject (see clisa.py sequence_scope)
    "material_id",  # 0..7
    "touch_type_id",  # 0..11
    "toucher_id",  # 0=hand, 1=object
    "pain",  # 0/1
)

#: Continuous meta keys (converted to ``torch.float`` on access, z-scored upstream).
CONTINUOUS_META_KEYS = ("valence", "arousal", "threat")

#: Full meta contract.
META_KEYS = ID_META_KEYS + CONTINUOUS_META_KEYS


# --------------------------------------------------------------------------- #
# numerics helpers
# --------------------------------------------------------------------------- #


def mask_value(dtype: torch.dtype) -> float:
    """Return a finite fill value that behaves like ``-inf`` under ``softmax``.

    ``-inf`` is avoided deliberately: a fully-masked row of ``-inf`` yields NaN
    from ``log_softmax`` and the NaN then poisons the whole batch even if the
    row is later discarded.  The values below are >= 4 orders of magnitude below
    any reachable logit (``|logit| <= logit_scale_max * 1 == 100``) so ``exp``
    underflows to exactly zero, while staying inside the dtype's finite range.
    """
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize without dividing by zero.

    The contract says ``z_eeg``/``z_vid`` arrive normalized, so for well-behaved
    inputs this is a no-op; it exists so a collapsed (all-zero) embedding
    produces a zero vector instead of NaN.
    """
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def to_float(x: Union[torch.Tensor, float, int]) -> float:
    """Detach a scalar tensor into a plain Python float for the ``logs`` dict."""
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        return float(x.detach().float().mean().item())
    return float(x)


def pairwise_eq(a: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``(B, B)`` boolean matrix with ``out[i, j] = (a[i] == b[j])``."""
    if b is None:
        b = a
    return a.reshape(-1, 1) == b.reshape(1, -1)


def masked_logsumexp(
    logits: torch.Tensor, valid: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """``logsumexp`` over the entries where ``valid`` is True.

    Fully-invalid slices return ``mask_value(dtype)`` (a finite sentinel) rather
    than ``-inf``; callers must discard those slices via their own validity
    mask.
    """
    fill = mask_value(logits.dtype)
    return torch.logsumexp(logits.masked_fill(~valid, fill), dim=dim)


def masked_log_softmax(
    logits: torch.Tensor, valid: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """``log_softmax`` restricted to ``valid`` entries; never NaN."""
    fill = mask_value(logits.dtype)
    return torch.log_softmax(logits.masked_fill(~valid, fill), dim=dim)


def safe_pdist(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Pairwise Euclidean distance with a NaN-free gradient on the diagonal.

    ``torch.cdist`` (and any ``sqrt`` of an exact zero) has an infinite
    derivative at distance 0, and the diagonal of a self-distance matrix is
    always exactly 0.  Even though the diagonal is masked out downstream,
    ``0 * inf == NaN`` still propagates through ``sqrt``'s backward.  Adding
    ``eps`` *inside* the square root is the fix.
    """
    x2 = (x * x).sum(dim=-1, keepdim=True)
    d2 = x2 + x2.transpose(0, 1) - 2.0 * (x @ x.transpose(0, 1))
    return torch.sqrt(d2.clamp_min(0.0) + eps)


def get_meta(
    meta: Mapping[str, Any],
    key: str,
    *,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    required: bool = True,
    default: Optional[Any] = None,
) -> Optional[torch.Tensor]:
    """Fetch ``meta[key]`` as a 1-D tensor of the right dtype/device.

    Accepts tensors, numpy arrays, and sequences, and reshapes ``(B, 1)`` to
    ``(B,)``.  Raises a loud, actionable ``KeyError`` when a loss needs a key the
    dataloader did not supply -- silently substituting zeros here would produce
    a plausible-looking but meaningless training curve, which is far worse.
    """
    if key not in meta or meta[key] is None:
        if required:
            raise KeyError(
                f"meta['{key}'] is required by this loss but was not provided. "
                f"Available keys: {sorted(meta.keys())}. "
                f"See tactus.losses.base.META_KEYS for the full contract."
            )
        if default is None:
            return None
        value: Any = default
    else:
        value = meta[key]

    if dtype is None:
        dtype = torch.long if key in ID_META_KEYS else torch.float32

    if isinstance(value, torch.Tensor):
        t = value
    else:
        t = torch.as_tensor(value)

    t = t.reshape(-1)
    if device is not None and t.device != device:
        t = t.to(device)
    if t.dtype != dtype:
        t = t.to(dtype)
    if batch_size is not None and t.numel() != batch_size:
        raise ValueError(
            f"meta['{key}'] has {t.numel()} elements but the batch has {batch_size}."
        )
    return t


# --------------------------------------------------------------------------- #
# temperature
# --------------------------------------------------------------------------- #


class TemperatureMixin:
    """Learnable (or fixed) CLIP-style temperature.

    Follows the OpenAI CLIP convention where the stored parameter is the
    *logarithm* of the multiplicative logit scale::

        logit_scale = log(1 / temperature)
        scale       = exp(logit_scale)          # multiplies the cosine sims

    The clamp is applied inside :meth:`scale` (differentiably -- gradients are
    zero outside the range) rather than by an external ``clamp_`` in the train
    loop, so a loss cannot blow up because someone forgot the post-step hook.
    CLIP's ceiling of 100 is the default ``max_scale``.

    Mix into a :class:`ContrastiveLoss` and call :meth:`_init_temperature` from
    ``__init__`` *after* ``super().__init__()``.
    """

    #: filled in by ``_init_temperature``
    _temp_max_scale: float
    _temp_min_scale: float

    def _init_temperature(
        self,
        temperature: float = 0.07,
        learnable: bool = True,
        max_scale: float = 100.0,
        min_scale: float = 0.01,
        name: str = "logit_scale",
    ) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not (0 < min_scale < max_scale):
            raise ValueError(
                f"require 0 < min_scale < max_scale, got {min_scale}, {max_scale}"
            )
        self._temp_name = name
        self._temp_max_scale = float(max_scale)
        self._temp_min_scale = float(min_scale)
        init = torch.tensor(math.log(1.0 / float(temperature)), dtype=torch.float32)
        # clamp the *initial* value into range so scale() is not silently
        # different from the temperature the user asked for.
        lo, hi = math.log(min_scale), math.log(max_scale)
        if not (lo <= float(init) <= hi):
            raise ValueError(
                f"temperature={temperature} implies logit scale "
                f"{math.exp(float(init)):.3f}, outside [{min_scale}, {max_scale}]. "
                f"Widen min_scale/max_scale or change the temperature."
            )
        if learnable:
            setattr(self, name, nn.Parameter(init))
        else:
            self.register_buffer(name, init)  # type: ignore[attr-defined]
        self._temp_learnable = bool(learnable)

    def scale(self) -> torch.Tensor:
        """Clamped multiplicative logit scale (0-dim tensor, differentiable)."""
        raw = getattr(self, self._temp_name)
        lo = math.log(self._temp_min_scale)
        hi = math.log(self._temp_max_scale)
        return raw.clamp(min=lo, max=hi).exp()

    @property
    def temperature(self) -> float:
        """Current temperature as a plain float (``1 / scale``)."""
        return float(1.0 / self.scale().detach().item())


# --------------------------------------------------------------------------- #
# the ABC
# --------------------------------------------------------------------------- #


class ContrastiveLoss(nn.Module, ABC):
    """Base class for every TACTUS contrastive objective.

    Subclasses implement :meth:`forward` and return
    ``{"loss": Tensor(scalar), "logs": {str: float}}``.

    Class attributes let the trainer reason about a loss without instantiating
    it:

    ``requires_video``
        ``False`` for EEG-only objectives (CLISA, RnC on ``z_eeg``).  The trainer
        uses this to warn when the video projector would receive no gradient.
    ``requires_meta``
        Meta keys the loss will access.  Checked up-front by
        :meth:`check_meta` so a 40-fold sweep fails in second 1, not hour 3.
    """

    requires_video: bool = True
    requires_meta: Sequence[str] = ()

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        z_eeg: torch.Tensor,
        z_vid: torch.Tensor,
        meta: Mapping[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Compute the loss.  See the module docstring for the exact contract."""
        raise NotImplementedError

    # -- shared utilities available to every subclass ----------------------- #

    def check_meta(self, meta: Mapping[str, Any]) -> None:
        """Raise if any key in :attr:`requires_meta` is absent."""
        missing = [k for k in self.requires_meta if k not in meta or meta[k] is None]
        if missing:
            raise KeyError(
                f"{type(self).__name__} requires meta keys {missing} which are "
                f"absent. Provided: {sorted(meta.keys())}."
            )

    def _zero_loss(self, *tensors: Optional[torch.Tensor]) -> torch.Tensor:
        """A scalar zero that is still connected to the graph.

        Returned when a batch contains no usable term (e.g. every sample shares
        one condition, so masked InfoNCE has no negatives).  Multiplying the
        inputs and this module's parameters by 0 guarantees ``.backward()``
        populates ``.grad`` with zeros everywhere instead of leaving ``None``,
        which would otherwise crash optimizers configured with
        ``set_to_none=False`` or trip DDP's unused-parameter check.
        """
        parts: list[torch.Tensor] = []
        for t in tensors:
            if isinstance(t, torch.Tensor):
                parts.append(t.sum() * 0.0)
        for p in self.parameters():
            parts.append(p.sum() * 0.0)
        if not parts:
            return torch.zeros((), dtype=torch.float32)
        out = parts[0]
        for extra in parts[1:]:
            out = out + extra
        # `x.sum() * 0.0` is -0.0 when the sum is negative; adding 0.0 normalizes
        # it so logs read "0.0000" rather than a puzzling "-0.0000".
        return out.float() + 0.0

    @staticmethod
    def _prepare(
        z_eeg: torch.Tensor,
        z_vid: Optional[torch.Tensor],
        renormalize: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Validate shapes and (optionally) re-normalize defensively."""
        if z_eeg.dim() != 2:
            raise ValueError(f"z_eeg must be (B, D), got {tuple(z_eeg.shape)}")
        if z_vid is not None:
            if z_vid.dim() != 2:
                raise ValueError(f"z_vid must be (B, D), got {tuple(z_vid.shape)}")
            if z_vid.shape != z_eeg.shape:
                raise ValueError(
                    f"z_eeg {tuple(z_eeg.shape)} and z_vid {tuple(z_vid.shape)} "
                    f"must have the same shape."
                )
        if renormalize:
            z_eeg = safe_normalize(z_eeg)
            if z_vid is not None:
                z_vid = safe_normalize(z_vid)
        return z_eeg, z_vid


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

LOSS_REGISTRY: Dict[str, Type[ContrastiveLoss]] = {}


def register_loss(name: str):
    """Class decorator that registers a loss under ``name``.

    >>> @register_loss("my_loss")
    ... class MyLoss(ContrastiveLoss):
    ...     ...
    """

    def _wrap(cls: Type[ContrastiveLoss]) -> Type[ContrastiveLoss]:
        key = name.strip().lower()
        if not key:
            raise ValueError("loss name must be a non-empty string")
        existing = LOSS_REGISTRY.get(key)
        if existing is not None and existing is not cls:
            # `python -m tactus.losses.infonce` imports the package (registering
            # InfoNCE) and then re-executes infonce.py as __main__, producing a
            # second, distinct class object with the same name. Likewise for
            # importlib.reload. Neither is a real collision, so only refuse when
            # two genuinely different classes claim the same name.
            re_execution = "__main__" in (cls.__module__, existing.__module__)
            same_class = existing.__qualname__ == cls.__qualname__
            if not (re_execution or same_class):
                raise KeyError(
                    f"loss name '{key}' is already registered to "
                    f"{existing.__module__}.{existing.__name__}. "
                    f"Pick a different name."
                )
        if not issubclass(cls, ContrastiveLoss):
            raise TypeError(
                f"{cls.__name__} must subclass ContrastiveLoss to be registered."
            )
        LOSS_REGISTRY[key] = cls
        cls.loss_name = key  # type: ignore[attr-defined]
        return cls

    return _wrap


def get_loss(name: str) -> Type[ContrastiveLoss]:
    """Look up a registered loss class by name."""
    key = str(name).strip().lower()
    if key not in LOSS_REGISTRY:
        raise KeyError(
            f"unknown loss '{name}'. Registered: {sorted(LOSS_REGISTRY)}. "
            f"(If you just added a file to tactus/losses/, import it in "
            f"tactus/losses/__init__.py so the decorator runs.)"
        )
    return LOSS_REGISTRY[key]


def list_losses() -> list[str]:
    """Sorted names of all registered losses."""
    return sorted(LOSS_REGISTRY)


def build_loss(cfg: Union[str, Mapping[str, Any], ContrastiveLoss]) -> ContrastiveLoss:
    """Instantiate a loss from a config dict.

    Accepted forms::

        build_loss("infonce")
        build_loss({"name": "infonce", "temperature": 0.05})
        build_loss({"type": "protonce", "dim": 256})          # 'type' is an alias
        build_loss({"name": "composite",
                    "components": {"protonce": 1.0, "clisa": 0.3}})

    Any key other than ``name``/``type`` is forwarded to the class constructor,
    so a typo in a YAML config surfaces as an ordinary ``TypeError`` naming the
    offending keyword rather than being silently ignored.
    """
    if isinstance(cfg, ContrastiveLoss):
        return cfg
    if isinstance(cfg, str):
        cfg = {"name": cfg}
    if not isinstance(cfg, Mapping):
        raise TypeError(f"loss cfg must be a str or mapping, got {type(cfg)!r}")

    kwargs = dict(cfg)
    name = kwargs.pop("name", None)
    alias = kwargs.pop("type", None)
    if name is None:
        name = alias
    elif alias is not None and str(alias).lower() != str(name).lower():
        raise ValueError(
            f"loss cfg specifies both name='{name}' and type='{alias}'; use one."
        )
    if name is None:
        raise ValueError(
            f"loss cfg must contain 'name' (or 'type'). Got keys {sorted(kwargs)}."
        )
    cls = get_loss(str(name))
    try:
        return cls(**kwargs)
    except TypeError as exc:  # pragma: no cover - surfaced to the user verbatim
        raise TypeError(
            f"could not construct loss '{name}' from cfg {dict(cfg)!r}: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# dummy batch generator (used by every module's __main__ smoke test)
# --------------------------------------------------------------------------- #


def make_dummy_batch(
    batch_size: int = 32,
    dim: int = 256,
    n_unique_conditions: Optional[int] = None,
    n_subjects: int = 8,
    n_sequences: int = 4,
    n_videos: int = 90,
    n_orientations: int = 4,
    single_condition: bool = False,
    single_subject: bool = False,
    single_sequence: bool = False,
    seed: int = 0,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Build a *structurally faithful* random batch for smoke tests.

    Faithful in the ways that break losses:

    * ``z_vid`` is drawn from a fixed 360-entry codebook indexed by
      ``condition_id`` -- so two trials of the same condition get **identical**
      video embeddings, exactly as in TACTUS (the video tower is frozen).  This
      is what makes in-batch false negatives real rather than hypothetical.
    * ``condition_id == (video_id - 1) * 4 + orientation`` holds exactly.
    * ``material_id`` is a deterministic function of ``video_id`` (a video has
      one material), not an independent draw.
    * Continuous labels are z-scored per batch and constant within a video.

    The ``single_*`` switches produce the degenerate batches every loss must
    survive: no negatives (``single_condition``), no cross-subject positives
    (``single_subject``), and every candidate refused as a negative
    (``single_sequence`` together with ``single_subject``).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    n_conditions = n_videos * n_orientations

    if single_condition:
        condition_id = torch.full((batch_size,), 7 * n_orientations, dtype=torch.long)
    else:
        if n_unique_conditions is None:
            # ~half the batch duplicated: the realistic stratified-sampling regime
            n_unique_conditions = max(1, batch_size // 2)
        n_unique_conditions = max(1, min(int(n_unique_conditions), n_conditions))
        # distinct condition ids, then draw the batch from that pool
        pool = torch.randperm(n_conditions, generator=g)[:n_unique_conditions]
        if n_unique_conditions >= batch_size:
            # guarantee every batch element is a different condition
            pick = torch.randperm(n_unique_conditions, generator=g)[:batch_size]
        else:
            pick = torch.randint(0, n_unique_conditions, (batch_size,), generator=g)
        condition_id = pool[pick]

    # the contract's identity, satisfied exactly
    video_id = condition_id // n_orientations + 1
    orientation = condition_id % n_orientations

    if single_subject:
        subject_id = torch.ones(batch_size, dtype=torch.long)
    else:
        subject_id = torch.randint(1, n_subjects + 1, (batch_size,), generator=g)
    if single_sequence:
        sequence_id = torch.ones(batch_size, dtype=torch.long)
    else:
        sequence_id = torch.randint(1, n_sequences + 1, (batch_size,), generator=g)

    # per-video attribute tables (a video has exactly one material, etc.)
    vg = torch.Generator(device="cpu").manual_seed(12345)
    mat_of_video = torch.randint(0, 8, (n_videos + 1,), generator=vg)
    tt_of_video = torch.randint(0, 12, (n_videos + 1,), generator=vg)
    tou_of_video = torch.randint(0, 2, (n_videos + 1,), generator=vg)
    pain_of_video = torch.randint(0, 2, (n_videos + 1,), generator=vg)
    val_of_video = torch.randn(n_videos + 1, generator=vg)
    aro_of_video = torch.randn(n_videos + 1, generator=vg)
    thr_of_video = torch.randn(n_videos + 1, generator=vg)

    def _z(x: torch.Tensor) -> torch.Tensor:
        # std() with fewer than 2 elements has dof <= 0 and returns NaN, which
        # would silently hand every loss a NaN label at B == 1.
        if x.numel() < 2:
            return torch.zeros_like(x)
        return (x - x.mean()) / x.std().nan_to_num(0.0).clamp_min(1e-6)

    meta: Dict[str, torch.Tensor] = {
        "video_id": video_id,
        "condition_id": condition_id,
        "orientation": orientation,
        "subject_id": subject_id,
        "sequence_id": sequence_id,
        "material_id": mat_of_video[video_id],
        "touch_type_id": tt_of_video[video_id],
        "toucher_id": tou_of_video[video_id],
        "pain": pain_of_video[video_id],
        "valence": _z(val_of_video[video_id]),
        "arousal": _z(aro_of_video[video_id]),
        "threat": _z(thr_of_video[video_id]),
    }
    meta = {k: v.to(dev) for k, v in meta.items()}

    z_eeg = safe_normalize(torch.randn(batch_size, dim, generator=g).to(dev, dtype))
    codebook = safe_normalize(torch.randn(n_conditions, dim, generator=g).to(dev, dtype))
    z_vid = codebook[condition_id.to(dev)]

    if requires_grad:
        z_eeg = z_eeg.detach().requires_grad_(True)
        z_vid = z_vid.detach().requires_grad_(True)
    return z_eeg, z_vid, meta


if __name__ == "__main__":  # pragma: no cover
    ze, zv, m = make_dummy_batch(32, 256)
    print("z_eeg", tuple(ze.shape), "z_vid", tuple(zv.shape))
    print("unique conditions in batch:", int(m["condition_id"].unique().numel()))
    print("meta keys:", sorted(m))
    print("registered losses (base only, nothing imported yet):", list_losses())
