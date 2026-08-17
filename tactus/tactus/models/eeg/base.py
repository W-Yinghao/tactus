"""EEG encoder base class and registry (contract F).

Contract F
----------
``class EEGEncoder(nn.Module): forward(x, subject_id) -> (B, D)`` L2-normalized,
with ``x: (B, 64, T) float32``.

Everything subject-related is delegated to
:mod:`tactus.models.eeg.subject_cond`, so a new architecture only has to implement

* :meth:`EEGEncoder.extract_features` -- ``(B, C, T) -> (B, F)``, and
* :meth:`EEGEncoder.build_projector`  -- ``F' -> nn.Module`` mapping to ``(B, D)``,

then call :meth:`EEGEncoder.setup_head` at the end of its ``__init__``.  The base
class then wires conditioning in at the three contract points and guarantees the
L2-normalized output.

The forward pass is::

    x  --(conditioner.apply_input)-->  x'
       --(extract_features, inside conditioner.subject_context)-->  h
       --(conditioner.apply_features)-->  h'
       --(projector)-->  z  --(l2 normalize)-->  (B, D)

Adding an architecture::

    @register_eeg_encoder("my_net")
    class MyNet(EEGEncoder):
        def __init__(self, *, n_filters=40, **base):
            super().__init__(**base)
            self.trunk = ...
            self.setup_head()
        def extract_features(self, x): return self.trunk(x).flatten(1)
        def build_projector(self, in_dim): return ResidualProjection(in_dim, self.embed_dim)
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type

import torch
import torch.nn as nn

from .._cfg import apply_aliases, filter_kwargs, to_dict, unwrap_section
from ..heads import ResidualProjection, l2_normalize
from .subject_cond import SubjectConditioner, build_subject_conditioner

__all__ = [
    "EEGEncoder",
    "register_eeg_encoder",
    "get_eeg_encoder",
    "list_eeg_encoders",
    "count_parameters",
    "build_eeg_encoder",
]


_EEG_ENCODER_REGISTRY: Dict[str, Type["EEGEncoder"]] = {}


def register_eeg_encoder(name: str, *aliases: str):
    """Class decorator registering an :class:`EEGEncoder` subclass under ``name``."""

    def deco(cls: Type["EEGEncoder"]) -> Type["EEGEncoder"]:
        for key in (name, *aliases):
            k = key.lower()
            if k in _EEG_ENCODER_REGISTRY:
                raise KeyError(f"EEG encoder {key!r} already registered")
            _EEG_ENCODER_REGISTRY[k] = cls
        cls.encoder_name = name.lower()
        return cls

    return deco


def get_eeg_encoder(name: str) -> Type["EEGEncoder"]:
    """Look up a registered encoder class."""
    key = str(name).lower()
    if key not in _EEG_ENCODER_REGISTRY:
        raise KeyError(f"unknown EEG encoder {name!r}; available: {list_eeg_encoders()}")
    return _EEG_ENCODER_REGISTRY[key]


def list_eeg_encoders() -> List[str]:
    """Names (and aliases) of registered encoders."""
    return sorted(_EEG_ENCODER_REGISTRY)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters, de-duplicated by identity."""
    seen: Dict[int, int] = {}
    for p in module.parameters():
        if trainable_only and not p.requires_grad:
            continue
        seen[id(p)] = p.numel()
    return sum(seen.values())


class EEGEncoder(nn.Module, abc.ABC):
    """Abstract EEG tower.

    Parameters
    ----------
    n_channels:
        Number of EEG channels (64 for BioSemi ActiveTwo, contract B).
    n_times:
        Samples per epoch: 120 for the primary ``w0600`` window, 180 for the
        ``wm100_800`` sensitivity window (contract B).
    embed_dim:
        Joint-space dimensionality ``D``.
    n_subjects:
        Number of subjects owning conditioning parameters (80).
    subject_ids:
        Dataset ids per parameter row; defaults to ``1..n_subjects`` (contract A).
    subject_cond:
        ``"none" | "subject_token" | "subject_layer" | "sulora"``.
    subject_cond_kwargs:
        Extra keyword arguments for the conditioner.
    normalize_output:
        L2-normalize the returned embedding.  Required by the loss contract; only
        disable for diagnostics.
    """

    #: set by :func:`register_eeg_encoder`
    encoder_name: str = "abstract"

    #: Architecture-specific config spellings, ``canonical -> aliases``.  Applied by
    #: :func:`build_eeg_encoder` *after* the global :data:`EEG_ENCODER_ALIASES`, so each
    #: architecture can claim a generic name (``d_model``, ``depth``) for its own use.
    config_aliases: Dict[str, Sequence[str]] = {}

    def __init__(
        self,
        *,
        n_channels: int = 64,
        n_times: int = 120,
        embed_dim: int = 256,
        n_subjects: int = 80,
        subject_ids: Optional[Sequence[int]] = None,
        subject_cond: Optional[str] = "none",
        subject_cond_kwargs: Optional[Dict[str, Any]] = None,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        if n_channels <= 0 or n_times <= 0 or embed_dim <= 0:
            raise ValueError("n_channels, n_times and embed_dim must be positive")
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.embed_dim = int(embed_dim)
        self.n_subjects = int(n_subjects)
        self.normalize_output = bool(normalize_output)

        self._subject_cond_name = (subject_cond or "none").lower()
        self._subject_cond_kwargs = dict(subject_cond_kwargs or {})
        self._subject_ids = list(subject_ids) if subject_ids is not None else None

        # filled by setup_head()
        self.conditioner: Optional[SubjectConditioner] = None
        self.projector: Optional[nn.Module] = None
        self._feature_dim: Optional[int] = None

    # ---------------------------------------------------------------------------------
    # subclass API
    # ---------------------------------------------------------------------------------

    @abc.abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trunk forward: ``(B, C, T) -> (B, F)`` flattened features.

        Must work for any ``T`` compatible with the trunk's receptive field, because
        :class:`~tactus.models.heads.TimeWindowHeads` calls it on sub-windows.
        """

    def build_projector(self, in_dim: int) -> nn.Module:
        """Build the head mapping trunk features to ``(B, embed_dim)``.

        The default is the NICE-style residual projector.  ``in_dim`` already includes
        any widening from the conditioner (e.g. a concatenated subject token).
        """
        return ResidualProjection(in_dim, self.embed_dim, hidden_dim=self.embed_dim, n_blocks=1, dropout=0.5)

    # ---------------------------------------------------------------------------------
    # construction
    # ---------------------------------------------------------------------------------

    def setup_head(self) -> None:
        """Build the conditioner and projector.  Call at the end of the subclass ``__init__``.

        Order matters: the trunk must exist so its feature width can be probed; the
        conditioner is built next (it may widen that width); the projector is built
        last; and only then may SuLoRA rewrite the projector's ``Linear`` layers.
        """
        if self.projector is not None:
            raise RuntimeError("setup_head() called twice")
        feat_dim = self.feature_dim_for(self.n_times)
        self.conditioner = build_subject_conditioner(
            self._subject_cond_name,
            n_subjects=self.n_subjects,
            n_channels=self.n_channels,
            feature_dim=feat_dim,
            subject_ids=self._subject_ids,
            **self._subject_cond_kwargs,
        )
        proj_in = int(self.conditioner.transform_feature_dim(feat_dim))
        self.projector = self.build_projector(proj_in)
        self._feature_dim = feat_dim
        self._projector_in_dim = proj_in
        self.conditioner.attach(self)

    def feature_dim_for(self, n_times: int) -> int:
        """Flattened trunk width for an input of ``n_times`` samples (shape probe).

        Runs the trunk once on zeros in ``eval`` mode under ``no_grad`` so no
        BatchNorm running statistics are touched.
        """
        n_times = int(n_times)
        if n_times <= 0:
            raise ValueError(f"n_times must be positive, got {n_times}")
        try:
            ref = next(self.parameters())
            device, dtype = ref.device, ref.dtype
        except StopIteration:  # pragma: no cover - trunk with no parameters
            device, dtype = torch.device("cpu"), torch.float32
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                dummy = torch.zeros(2, self.n_channels, n_times, device=device, dtype=dtype)
                out = self.extract_features(dummy)
        except RuntimeError as exc:  # pragma: no cover - depends on architecture
            raise RuntimeError(
                f"{type(self).__name__}.extract_features failed on an input of "
                f"({self.n_channels}, {n_times}); the window is probably shorter than the "
                f"trunk's receptive field. Original error: {exc}"
            ) from exc
        finally:
            self.train(was_training)
        if out.dim() != 2:
            raise ValueError(
                f"extract_features must return (B, F); {type(self).__name__} returned {tuple(out.shape)}"
            )
        return int(out.shape[1])

    @property
    def feature_dim(self) -> int:
        """Flattened trunk width at the configured ``n_times``."""
        if self._feature_dim is None:
            raise RuntimeError("setup_head() has not been called")
        return int(self._feature_dim)

    # ---------------------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------------------

    def _check_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)  # tolerate (B, 1, C, T)
        if x.dim() != 3:
            raise ValueError(f"expected (B, {self.n_channels}, {self.n_times}), got {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {x.shape[1]}")
        if x.shape[2] != self.n_times:
            raise ValueError(
                f"expected {self.n_times} samples (window mismatch?), got {x.shape[2]}. "
                "Use TimeWindowHeads for sub-window embeddings."
            )
        return x

    def forward(self, x: torch.Tensor, subject_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x``: ``(B, 64, T)``, ``subject_id``: ``(B,)`` dataset ids -> ``(B, D)`` L2-normalized.

        ``subject_id=None`` is legal and means "no subject information": every
        conditioner falls back to its pre-registered unseen state, which is the correct
        behaviour for a pure-LOSO evaluation.
        """
        if self.projector is None or self.conditioner is None:
            raise RuntimeError(f"{type(self).__name__} did not call setup_head() in __init__")
        x = self._check_input(x)
        cond = self.conditioner
        with cond.subject_context(subject_id):
            x = cond.apply_input(x, subject_id)
            h = self.extract_features(x)
            h = cond.apply_features(h, subject_id)
            z = self.projector(h)
        return l2_normalize(z) if self.normalize_output else z

    # ---------------------------------------------------------------------------------
    # subject-conditioning passthroughs
    # ---------------------------------------------------------------------------------

    def set_train_subjects(self, subject_ids: Optional[Sequence[int]]) -> None:
        """Declare this fold's training subjects (see :meth:`SubjectConditioner.set_train_subjects`).

        Call this for every fold before training *and* before evaluation, otherwise a
        held-out subject can read its own untrained parameters instead of the
        pre-registered unseen rule.
        """
        if self.conditioner is None:
            raise RuntimeError("setup_head() has not been called")
        self.conditioner.set_train_subjects(subject_ids)

    def unseen_subject_state(self) -> Dict[str, Any]:
        """The a-priori inference rule for an unseen subject (log this every run)."""
        if self.conditioner is None:
            raise RuntimeError("setup_head() has not been called")
        return self.conditioner.unseen_subject_state()

    def relink_subject_conditioning(self) -> None:
        """Repair conditioner<->adapter links after ``copy.deepcopy`` of this encoder.

        ``deepcopy`` copies ``weakref`` objects atomically, so without this the copy's
        SuLoRA adapters keep pointing at the *original* encoder's conditioner and
        silently contribute nothing.  Call it on every deep copy.
        """
        if self.conditioner is not None:
            self.conditioner.relink(self)

    # ---------------------------------------------------------------------------------
    # reporting
    # ---------------------------------------------------------------------------------

    def n_parameters(self, trainable_only: bool = True) -> int:
        """Total (trainable) parameter count."""
        return count_parameters(self, trainable_only)

    def describe(self) -> Dict[str, Any]:
        """Compact architecture summary, suitable for the run log."""
        cond = self.conditioner
        out: Dict[str, Any] = {
            "encoder": self.encoder_name,
            "class": type(self).__name__,
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "embed_dim": self.embed_dim,
            "feature_dim": self._feature_dim,
            "projector_in_dim": getattr(self, "_projector_in_dim", None),
            "n_parameters": self.n_parameters(),
            "subject_cond": self._subject_cond_name,
            "subject_cond_kwargs": dict(self._subject_cond_kwargs),
        }
        if cond is not None:
            out["unseen_subject_rule"] = {
                k: v for k, v in cond.unseen_subject_state().items() if k != "state"
            }
            out["n_train_subjects"] = len(cond.train_subject_ids)
            if hasattr(cond, "adapter_parameter_count"):
                out["adapter_parameters"] = cond.adapter_parameter_count()  # type: ignore[attr-defined]
                out["adapter_targets"] = cond.attached_targets  # type: ignore[attr-defined]
        return out


# --------------------------------------------------------------------------------------
# builder
# --------------------------------------------------------------------------------------

#: Accepted spellings for encoder options.  ``canonical -> aliases`` in priority order.
EEG_ENCODER_ALIASES: Dict[str, Sequence[str]] = {
    "name": ("arch", "encoder", "encoder_name"),
    "embed_dim": ("d_embed", "d_out", "out_dim", "dim"),
    "n_times": ("n_samples", "T"),
    "n_channels": ("n_chans", "C"),
    "subject_cond": ("subject_conditioning", "conditioning", "subject_mechanism"),
    "subject_cond_kwargs": ("subject_conditioning_kwargs", "conditioning_kwargs"),
}


def _split_subject_cond(cfg: Dict[str, Any]) -> tuple:
    """Extract ``(mechanism_name, kwargs)`` from either a string or a nested mapping."""
    raw = cfg.pop("subject_cond", "none")
    extra = to_dict(cfg.pop("subject_cond_kwargs", None))
    is_mapping_like = isinstance(raw, Mapping) or (
        raw is not None and not isinstance(raw, (str, bytes)) and hasattr(raw, "__dict__")
    )
    if is_mapping_like:
        sub = to_dict(raw)
        name = str(sub.pop("name", None) or sub.pop("mechanism", None) or "none")
        sub.update(extra)
        return name, sub
    return str(raw or "none"), extra


def build_eeg_encoder(cfg: Any = None, **overrides: Any) -> EEGEncoder:
    """Build an EEG encoder (contract F) from a config mapping, section, or name.

    Accepts, in decreasing order of specificity:

    * a whole run config containing a ``model`` / ``eeg_encoder`` / ``encoder`` section,
    * the encoder section itself,
    * a bare architecture name (``build_eeg_encoder("tsconv", n_times=120)``),

    plus keyword ``overrides`` applied last.  Alias spellings listed in
    :data:`EEG_ENCODER_ALIASES` are normalised, and keys no constructor in the MRO
    accepts are dropped with a warning rather than raising -- one shared config can
    therefore carry keys for several architectures.

    Subject ids
    -----------
    Rows are addressed by *dataset* subject id (contract A: 1..80).  Any id outside the
    registered set -- including ``0`` and :data:`~tactus.models.eeg.subject_cond.UNSEEN_SUBJECT`
    (``-1``) -- triggers the mechanism's pre-registered unseen rule, so a caller that
    reserves index 0 for "mean/unseen subject" gets the intended behaviour with
    ``n_subjects=81`` and no further configuration.

    Remember to call ``encoder.set_train_subjects(fold.train_subjects)`` for each fold.
    """
    if isinstance(cfg, str):
        conf: Dict[str, Any] = {"name": cfg}
    else:
        conf = unwrap_section(to_dict(cfg), "model", "eeg_encoder", "encoder", "eeg")
    conf = dict(conf)
    conf.update(overrides)
    apply_aliases(conf, EEG_ENCODER_ALIASES)
    conf.pop("_target_", None)
    # a nested {"params": {...}} block is merged up
    params = conf.pop("params", None)
    if params is not None:
        merged = to_dict(params)
        merged.update({k: v for k, v in conf.items() if k != "params"})
        conf = apply_aliases(merged, EEG_ENCODER_ALIASES)

    name = str(conf.pop("name", "tsconv"))
    cls = get_eeg_encoder(name)
    apply_aliases(conf, getattr(cls, "config_aliases", {}))
    cond_name, cond_kwargs = _split_subject_cond(conf)
    kwargs = filter_kwargs(cls, conf, f"build_eeg_encoder({name!r})")
    return cls(subject_cond=cond_name, subject_cond_kwargs=cond_kwargs, **kwargs)
