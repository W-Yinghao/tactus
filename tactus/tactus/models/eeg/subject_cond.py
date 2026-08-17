"""Interchangeable subject-conditioning mechanisms for the EEG tower.

Four mechanisms behind one interface (BLUEPRINT_v2 section 4.2, the
"subject token / 1x1 conv layer / SuLoRA low-rank adapter" three-way ablation):

======================  ==================================================  ==========================
name                    what it does                                        UNSEEN-SUBJECT RULE
======================  ==================================================  ==========================
``none``                nothing; the encoder is subject-agnostic            identity (no-op)
``subject_token``       learnable per-subject embedding added to /          **mean of the training
                        concatenated with the trunk features                subjects' tokens**
``subject_layer``       per-subject ``I + dW`` mixing matrix over the 64    **identity layer**
                        input channels (Defossez 2023, brainmagick)         (``dW = 0``)
``sulora``              per-subject rank-r additive correction on chosen    **zero adapter**
                        ``Linear``/``Conv`` weights (arXiv:2510.08059)      (``B_s x A_s = 0``)
======================  ==================================================  ==========================

The unseen-subject rule is the load-bearing part.  The blueprint requires it to be
**fixed a priori** ("纯 LOSO 推断规则先验定死 ... 否则 LOSO 从头条声明中删除"), because
choosing it after seeing LOSO numbers is a hidden test-set fit.  Every conditioner
therefore exposes :meth:`SubjectConditioner.unseen_subject_state`, which returns both
a machine-readable description of the rule and the concrete tensor(s) it produces --
dump that dict into the pre-registration and into every run's log.

Two safety properties are enforced by construction:

1. **A subject id that was never in the training set can never silently use its own
   parameters.**  Call :meth:`SubjectConditioner.set_train_subjects` with the fold's
   training subjects; with ``strict_unseen=True`` (default) every other id is routed
   to the unseen rule even if a (randomly initialised, never trained) row exists for
   it.  Without this, a LOSO evaluation quietly reads noise instead of applying the
   pre-registered rule.
2. **Adapters are zero-initialised** so an untrained conditioner is exactly the
   unconditioned model; the ablation ladder starts from a common point.

Subject ids are the dataset's own 1..80 (contract A ``subject_id``), not 0-based rows;
the mapping is handled internally by a lookup buffer.  Use ``subject_id = -1`` (or any
id outside the registered set) to request the unseen rule explicitly; the constant
:data:`UNSEEN_SUBJECT` is provided for that.
"""

from __future__ import annotations

import abc
import contextlib
import contextvars
import fnmatch
import weakref
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "UNSEEN_SUBJECT",
    "SubjectConditioner",
    "NoSubjectConditioning",
    "SubjectToken",
    "SubjectLayer",
    "SuLoRA",
    "SuLoRALinear",
    "SuLoRAConv2d",
    "build_subject_conditioner",
    "list_subject_conditioners",
    "register_subject_conditioner",
]

#: Sentinel subject id meaning "not one of the training subjects -> apply the
#: pre-registered unseen rule".  Any id absent from the registered set behaves the same.
UNSEEN_SUBJECT: int = -1

#: Active subject rows, keyed by ``id(conditioner)``.
#:
#: This is deliberately a *module-level* ContextVar rather than a per-instance one:
#: ``contextvars.ContextVar`` cannot be deep-copied ("cannot pickle ContextVar"), and
#: ``TimeWindowHeads(share_trunk=False)`` deep-copies whole encoders.  Keeping the
#: variable at module level makes conditioners copyable while retaining the
#: thread-locality that ``DataParallel``'s per-device threads need.
_SUBJECT_ROWS: contextvars.ContextVar[Mapping[int, Optional[torch.Tensor]]] = contextvars.ContextVar(
    "tactus_subject_rows", default={}
)


# ======================================================================================
# base
# ======================================================================================


class SubjectConditioner(nn.Module, abc.ABC):
    """Common interface for all subject-conditioning mechanisms.

    A conditioner may intervene at three points, and each mechanism uses a subset:

    ``apply_input(x, subject_id)``
        transform the raw ``(B, C, T)`` input (used by ``subject_layer``);
    ``apply_features(h, subject_id)``
        transform the flattened trunk features ``(B, F)`` (used by ``subject_token``);
    ``attach(model)`` + ``subject_context(subject_id)``
        rewrite weights inside the trunk (used by ``sulora``); the encoder wraps its
        whole forward pass in ``subject_context`` so wrapped modules can look up the
        current batch's subject rows.

    Parameters
    ----------
    n_subjects:
        Number of subjects with their own parameters (80 for ds005662).
    n_channels:
        EEG channel count (64).
    feature_dim:
        Flattened trunk feature width, needed by feature-level mechanisms.
    subject_ids:
        The dataset ids owning each row, in row order.  Defaults to ``1..n_subjects``
        to match contract A.
    strict_unseen:
        If ``True`` (default) any id not currently marked as a training subject is
        routed to the unseen rule.  Turn this off only for deliberate transductive
        experiments, and say so in the write-up.
    """

    mechanism: str = "abstract"

    def __init__(
        self,
        *,
        n_subjects: int = 80,
        n_channels: int = 64,
        feature_dim: int = 0,
        subject_ids: Optional[Sequence[int]] = None,
        strict_unseen: bool = True,
    ) -> None:
        super().__init__()
        self.n_subjects = int(n_subjects)
        self.n_channels = int(n_channels)
        self.feature_dim = int(feature_dim)
        self.strict_unseen = bool(strict_unseen)

        ids = list(range(1, self.n_subjects + 1)) if subject_ids is None else [int(s) for s in subject_ids]
        if len(ids) != self.n_subjects:
            raise ValueError(f"subject_ids has {len(ids)} entries but n_subjects={self.n_subjects}")
        if len(set(ids)) != len(ids):
            raise ValueError("subject_ids contains duplicates")
        if min(ids) < 0:
            raise ValueError("subject ids must be non-negative (negative ids mean 'unseen')")
        lut = torch.full((max(ids) + 1,), -1, dtype=torch.long)
        for row, sid in enumerate(ids):
            lut[sid] = row
        self.register_buffer("_sid_lut", lut, persistent=True)
        self.register_buffer("_subject_ids", torch.tensor(ids, dtype=torch.long), persistent=True)
        self.register_buffer("_train_mask", torch.ones(self.n_subjects, dtype=torch.bool), persistent=True)

    # -- id bookkeeping ---------------------------------------------------------------

    @property
    def subject_ids(self) -> List[int]:
        """Dataset subject ids in row order."""
        return [int(v) for v in self._subject_ids.tolist()]

    @property
    def train_subject_ids(self) -> List[int]:
        """Dataset ids currently marked as training subjects."""
        return [int(v) for v in self._subject_ids[self._train_mask].tolist()]

    def set_train_subjects(self, subject_ids: Optional[Iterable[int]]) -> None:
        """Declare which subjects this fold trains on (call once per fold, before training).

        ``None`` resets to "all subjects are training subjects".  Ids that have no row
        raise -- silently ignoring them would produce a wrong unseen-token mean.
        """
        if subject_ids is None:
            self._train_mask.fill_(True)
            return
        ids = [int(s) for s in subject_ids]
        mask = torch.zeros_like(self._train_mask)
        for sid in ids:
            if sid < 0 or sid >= self._sid_lut.numel() or int(self._sid_lut[sid]) < 0:
                raise ValueError(f"subject id {sid} has no row in this conditioner")
            mask[int(self._sid_lut[sid])] = True
        if not bool(mask.any()):
            raise ValueError("set_train_subjects() would leave zero training subjects")
        self._train_mask.copy_(mask)

    def rows_for(self, subject_id: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Map dataset subject ids to parameter rows; ``-1`` marks the unseen rule.

        ``None`` in, ``None`` out (meaning "no subject information supplied" -- every
        mechanism then falls back to its unseen state, which is the safe default).
        """
        if subject_id is None:
            return None
        sid = torch.as_tensor(subject_id)
        sid = sid.reshape(-1).to(device=self._sid_lut.device, dtype=torch.long)
        rows = torch.full_like(sid, -1)
        in_range = (sid >= 0) & (sid < self._sid_lut.numel())
        if bool(in_range.any()):
            rows[in_range] = self._sid_lut[sid[in_range]]
        if self.strict_unseen:
            safe = rows.clamp(min=0)
            keep = (rows >= 0) & self._train_mask[safe]
            rows = torch.where(keep, rows, torch.full_like(rows, -1))
        return rows

    # -- context ----------------------------------------------------------------------

    @contextlib.contextmanager
    def subject_context(self, subject_id: Optional[torch.Tensor]):
        """Publish this batch's subject rows for weight-level mechanisms (SuLoRA).

        Re-entrant and thread-local (``contextvars``), so nested calls and
        ``DataParallel``'s per-device threads do not clobber each other.
        """
        current = dict(_SUBJECT_ROWS.get())
        current[id(self)] = self.rows_for(subject_id)
        token = _SUBJECT_ROWS.set(current)
        try:
            yield
        finally:
            _SUBJECT_ROWS.reset(token)

    def current_rows(self) -> Optional[torch.Tensor]:
        """Rows published by the innermost active :meth:`subject_context`."""
        return _SUBJECT_ROWS.get().get(id(self))

    # -- hooks (default = no-op) ------------------------------------------------------

    def transform_feature_dim(self, feature_dim: int) -> int:
        """Feature width seen by the projector after :meth:`apply_features`."""
        return int(feature_dim)

    def apply_input(self, x: torch.Tensor, subject_id: Optional[torch.Tensor]) -> torch.Tensor:
        """Transform the raw ``(B, C, T)`` input."""
        return x

    def apply_features(self, h: torch.Tensor, subject_id: Optional[torch.Tensor]) -> torch.Tensor:
        """Transform the flattened ``(B, F)`` trunk features."""
        return h

    def attach(self, model: nn.Module) -> None:
        """Perform any weight surgery on the host encoder.  Called once, after build."""
        return None

    def adapt_module(self, module: nn.Module, name: str = "") -> int:
        """Apply this mechanism to a module built *after* :meth:`attach`.

        Needed by :class:`~tactus.models.heads.TimeWindowHeads`, whose per-window
        projection heads do not exist when the encoder is constructed.  Mechanisms that
        act through :meth:`apply_input` / :meth:`apply_features` (``subject_layer``,
        ``subject_token``) already cover any downstream head and correctly do nothing
        here; only weight-level mechanisms need to reach into the new module.

        Returns the number of sub-modules adapted.
        """
        return 0

    def relink(self, model: nn.Module) -> None:
        """Re-establish links between this conditioner and modules inside ``model``.

        ``copy.deepcopy`` copies ``weakref`` objects atomically, so a deep-copied
        encoder's adapters would still point at the *original* conditioner and would
        silently go inert.  Anything that copies an encoder must call this on the copy
        (:class:`~tactus.models.heads.TimeWindowHeads` does).  A no-op for mechanisms
        that perform no weight surgery.
        """
        return None

    # -- the pre-registered inference rule --------------------------------------------

    @abc.abstractmethod
    def unseen_subject_state(self) -> Dict[str, Any]:
        """Return the a-priori inference rule for a subject with no trained parameters.

        Keys
        ----
        ``mechanism`` : str
            Conditioner name.
        ``rule`` : str
            Machine-readable rule id, e.g. ``"mean_of_train_subject_tokens"``.
        ``description`` : str
            Human-readable statement for the pre-registration.
        ``fixed_a_priori`` : bool
            Always ``True`` -- these rules are not tuned on held-out subjects.
        ``n_train_subjects`` : int
        ``state`` : dict[str, Tensor]
            The concrete tensors the rule produces (detached, on CPU).
        """

    # -- misc -------------------------------------------------------------------------

    def n_parameters(self, trainable_only: bool = True) -> int:
        """Parameter count owned by this conditioner (SuLoRA's live in the host trunk)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def extra_repr(self) -> str:  # noqa: D102
        return (
            f"mechanism={self.mechanism!r}, n_subjects={self.n_subjects}, "
            f"n_train_subjects={int(self._train_mask.sum())}, strict_unseen={self.strict_unseen}"
        )

    # -- shared helper ----------------------------------------------------------------

    @staticmethod
    def _expand_mask(rows: torch.Tensor, ndim: int) -> torch.Tensor:
        """``(B,)`` bool -> ``(B, 1, 1, ...)`` broadcastable against an ``ndim``-D tensor."""
        return (rows >= 0).reshape(-1, *([1] * (ndim - 1)))


# ======================================================================================
# registry
# ======================================================================================

_CONDITIONER_REGISTRY: Dict[str, Type[SubjectConditioner]] = {}
_CONDITIONER_ALIASES: Dict[str, str] = {}


def register_subject_conditioner(name: str, *aliases: str):
    """Class decorator registering a conditioner under ``name`` (plus optional aliases)."""

    def deco(cls: Type[SubjectConditioner]) -> Type[SubjectConditioner]:
        key = name.lower()
        if key in _CONDITIONER_REGISTRY:
            raise KeyError(f"subject conditioner {name!r} already registered")
        _CONDITIONER_REGISTRY[key] = cls
        cls.mechanism = key
        for a in aliases:
            _CONDITIONER_ALIASES[a.lower()] = key
        return cls

    return deco


def list_subject_conditioners() -> List[str]:
    """Registered conditioner names."""
    return sorted(_CONDITIONER_REGISTRY)


def build_subject_conditioner(
    name: Optional[str] = "none",
    *,
    n_subjects: int = 80,
    n_channels: int = 64,
    feature_dim: int = 0,
    subject_ids: Optional[Sequence[int]] = None,
    strict_unseen: bool = True,
    **kwargs: Any,
) -> SubjectConditioner:
    """Build a conditioner by name.  ``None``/``"none"`` gives the no-op mechanism."""
    key = (name or "none").lower()
    key = _CONDITIONER_ALIASES.get(key, key)
    if key not in _CONDITIONER_REGISTRY:
        raise KeyError(
            f"unknown subject conditioner {name!r}; available: "
            f"{list_subject_conditioners()} (aliases: {sorted(_CONDITIONER_ALIASES)})"
        )
    cls = _CONDITIONER_REGISTRY[key]
    return cls(
        n_subjects=n_subjects,
        n_channels=n_channels,
        feature_dim=feature_dim,
        subject_ids=subject_ids,
        strict_unseen=strict_unseen,
        **kwargs,
    )


# ======================================================================================
# (a) none
# ======================================================================================


@register_subject_conditioner("none", "identity", "null")
class NoSubjectConditioning(SubjectConditioner):
    """No conditioning.  The encoder is fully subject-agnostic.

    This is also the mechanism the *phenotype read-out* encoder must use
    (BLUEPRINT_v2 section 7: per-subject alignment scores may not come from a model
    that was optimised to erase subject differences).
    """

    def unseen_subject_state(self) -> Dict[str, Any]:  # noqa: D102
        return {
            "mechanism": "none",
            "rule": "identity",
            "description": "Encoder has no subject-specific parameters; unseen subjects are "
            "processed exactly like training subjects.",
            "fixed_a_priori": True,
            "n_train_subjects": int(self._train_mask.sum()),
            "state": {},
        }


# ======================================================================================
# (b) subject token
# ======================================================================================


@register_subject_conditioner("subject_token", "token", "clip_mused")
class SubjectToken(SubjectConditioner):
    """Learnable per-subject embedding injected at the projector (CLIP-MUSED style).

    Parameters
    ----------
    token_dim:
        Token width.  ``None`` means ``feature_dim`` for ``mode="add"`` and 64 for
        ``mode="concat"``.
    mode:
        ``"add"`` -- ``h + scale * token`` (token_dim must equal feature_dim);
        ``"concat"`` -- ``[h, token]``, widening the projector input.
    scale:
        Multiplier on the token in ``"add"`` mode.
    init_std:
        Std of the Gaussian token initialisation.  Small (0.02) so that an untrained
        model is close to the unconditioned one.
    detach_unseen:
        If ``True`` (default) the mean-token used for unseen subjects is detached, so
        an unseen sample sneaking into a training batch cannot drag every training
        token toward it.

    Unseen rule
    -----------
    **Mean of the training subjects' tokens.**  Note this is the mean over the rows
    marked by :meth:`set_train_subjects`, *not* over all 80 rows -- a held-out
    subject's own row is never trained, so including it would average in noise.
    """

    def __init__(
        self,
        *,
        token_dim: Optional[int] = None,
        mode: str = "add",
        scale: float = 1.0,
        init_std: float = 0.02,
        dropout: float = 0.0,
        detach_unseen: bool = True,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        mode = str(mode).lower()
        if mode not in ("add", "concat"):
            raise ValueError(f"mode must be 'add' or 'concat', got {mode!r}")
        if token_dim is None:
            token_dim = self.feature_dim if mode == "add" else 64
        token_dim = int(token_dim)
        if mode == "add" and self.feature_dim and token_dim != self.feature_dim:
            raise ValueError(
                f"mode='add' needs token_dim == feature_dim ({self.feature_dim}), got {token_dim}"
            )
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}")

        self.mode = mode
        self.token_dim = token_dim
        self.scale = float(scale)
        self.detach_unseen = bool(detach_unseen)
        self.embedding = nn.Embedding(self.n_subjects, token_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=float(init_std))
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    # ---------------------------------------------------------------------------------

    def transform_feature_dim(self, feature_dim: int) -> int:  # noqa: D102
        return int(feature_dim) + self.token_dim if self.mode == "concat" else int(feature_dim)

    def unseen_token(self) -> torch.Tensor:
        """The pre-registered token for an unseen subject: mean of the training tokens."""
        w = self.embedding.weight
        mask = self._train_mask
        tok = w[mask].mean(dim=0) if bool(mask.any()) else torch.zeros_like(w[0])
        return tok.detach() if self.detach_unseen else tok

    def tokens_for(self, subject_id: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        """``(B, token_dim)`` tokens, with the unseen rule substituted where needed."""
        unseen = self.unseen_token()
        rows = self.rows_for(subject_id)
        if rows is None:
            return unseen.unsqueeze(0).expand(batch_size, -1)
        if rows.numel() != batch_size:
            raise ValueError(f"subject_id has {rows.numel()} entries but batch is {batch_size}")
        known = (rows >= 0).unsqueeze(1)
        # clamp(min=0) keeps the gather in range; torch.where zeroes that branch's grad.
        return torch.where(known, self.embedding(rows.clamp(min=0)), unseen.unsqueeze(0))

    def apply_features(
        self, h: torch.Tensor, subject_id: Optional[torch.Tensor]
    ) -> torch.Tensor:  # noqa: D102
        tok = self.dropout(self.tokens_for(subject_id, h.shape[0])).to(dtype=h.dtype)
        if self.mode == "add":
            if tok.shape[1] != h.shape[1]:
                raise ValueError(
                    f"token_dim={tok.shape[1]} does not match feature width {h.shape[1]}; "
                    "rebuild the conditioner with the trunk's real feature_dim"
                )
            return h + self.scale * tok
        return torch.cat([h, tok], dim=1)

    def unseen_subject_state(self) -> Dict[str, Any]:  # noqa: D102
        return {
            "mechanism": "subject_token",
            "rule": "mean_of_train_subject_tokens",
            "description": (
                "An unseen subject receives the arithmetic mean of the token embeddings of the "
                f"{int(self._train_mask.sum())} training subjects of this fold "
                f"(mode={self.mode!r}, token_dim={self.token_dim})."
            ),
            "fixed_a_priori": True,
            "n_train_subjects": int(self._train_mask.sum()),
            "state": {"token": self.unseen_token().detach().cpu()},
        }

    def extra_repr(self) -> str:  # noqa: D102
        return super().extra_repr() + f", mode={self.mode!r}, token_dim={self.token_dim}"


# ======================================================================================
# (c) subject layer
# ======================================================================================


@register_subject_conditioner("subject_layer", "layer", "defossez")
class SubjectLayer(SubjectConditioner):
    """Per-subject 1x1 convolution over the 64 input channels (Defossez 2023).

    The layer is parameterised as ``x' = (I + dW_s) x`` rather than ``W_s x`` for two
    reasons: (i) ``dW = 0`` initialisation makes an untrained model identical to the
    unconditioned model, and (ii) it makes the unseen rule -- ``dW = 0``, i.e. the
    identity layer -- an exact member of the parameter family rather than an
    out-of-family fallback.

    Parameters
    ----------
    rank:
        ``None`` (default) uses a full ``(C, C)`` correction per subject
        (``80 x 64 x 64 = 327,680`` parameters -- comparable to the whole TSConv trunk,
        so consider ``rank=16`` if the ablation is capacity-confounded).  An integer
        ``r`` uses ``dW_s = U_s V_s^T`` with ``U, V`` of shape ``(C, r)``.
    init_std:
        Std for the non-zero factor.  The other factor is zero-initialised so the
        product starts at exactly zero.
    per_subject_bias:
        Add a per-subject per-channel bias (a DC offset per electrode).  Off by default:
        with no baseline correction (primary window 0-600 ms) a learned DC term can
        absorb the very drift we want the encoder to be robust to.

    Unseen rule
    -----------
    **Identity layer** (``dW = 0``): the raw 64 channels pass through untouched.
    """

    def __init__(
        self,
        *,
        rank: Optional[int] = None,
        init_std: float = 0.02,
        per_subject_bias: bool = False,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        c = self.n_channels
        self.rank = int(rank) if rank is not None else None
        if self.rank is not None and not (0 < self.rank <= c):
            raise ValueError(f"rank must be in (0, {c}], got {self.rank}")
        if self.rank is None:
            self.delta = nn.Parameter(torch.zeros(self.n_subjects, c, c))
            self.factor_u = None
            self.factor_v = None
        else:
            self.delta = None
            self.factor_u = nn.Parameter(torch.randn(self.n_subjects, c, self.rank) * float(init_std))
            self.factor_v = nn.Parameter(torch.zeros(self.n_subjects, c, self.rank))
        self.per_subject_bias = bool(per_subject_bias)
        if self.per_subject_bias:
            self.bias = nn.Parameter(torch.zeros(self.n_subjects, c))
        else:
            self.bias = None

    # ---------------------------------------------------------------------------------

    def delta_for_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """``(B, C, C)`` channel-mixing corrections; zero where ``rows < 0``."""
        idx = rows.clamp(min=0)
        if self.delta is not None:
            d = self.delta[idx]
        else:
            u = self.factor_u[idx]  # (B, C, r)
            v = self.factor_v[idx]  # (B, C, r)
            d = torch.bmm(u, v.transpose(1, 2))  # (B, C, C)
        return d * self._expand_mask(rows, d.dim()).to(d.dtype)

    def apply_input(
        self, x: torch.Tensor, subject_id: Optional[torch.Tensor]
    ) -> torch.Tensor:  # noqa: D102
        if x.dim() != 3:
            raise ValueError(f"SubjectLayer expects (B, C, T), got {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {x.shape[1]}")
        rows = self.rows_for(subject_id)
        if rows is None:
            return x  # unseen rule: identity
        if rows.numel() != x.shape[0]:
            raise ValueError(f"subject_id has {rows.numel()} entries but batch is {x.shape[0]}")
        d = self.delta_for_rows(rows).to(dtype=x.dtype)
        out = x + torch.bmm(d, x)
        if self.bias is not None:
            b = self.bias[rows.clamp(min=0)] * self._expand_mask(rows, 2).to(self.bias.dtype)
            out = out + b.to(dtype=x.dtype).unsqueeze(-1)
        return out

    def unseen_subject_state(self) -> Dict[str, Any]:  # noqa: D102
        c = self.n_channels
        return {
            "mechanism": "subject_layer",
            "rule": "identity_layer",
            "description": (
                "An unseen subject uses the identity channel-mixing layer (dW = 0, no bias); "
                f"the 64 input channels pass through unchanged (rank={self.rank})."
            ),
            "fixed_a_priori": True,
            "n_train_subjects": int(self._train_mask.sum()),
            "state": {
                "delta": torch.zeros(c, c),
                "weight": torch.eye(c),
                "bias": torch.zeros(c),
            },
        }

    def extra_repr(self) -> str:  # noqa: D102
        return super().extra_repr() + f", rank={self.rank}, per_subject_bias={self.per_subject_bias}"


# ======================================================================================
# (d) SuLoRA
# ======================================================================================


class _SuLoRABase(nn.Module):
    """Shared plumbing for the per-subject low-rank wrappers.

    The wrapper keeps a *weak* reference to the owning :class:`SuLoRA` conditioner.
    It must not be a plain attribute assignment of an ``nn.Module``, or the module
    graph would contain a cycle (encoder -> wrapper -> conditioner -> ... ) and
    ``named_modules`` would recurse forever.
    """

    def __init__(self, conditioner: "SuLoRA") -> None:
        super().__init__()
        object.__setattr__(self, "_cond_ref", weakref.ref(conditioner))

    @property
    def conditioner(self) -> "SuLoRA":
        cond = self._cond_ref()
        if cond is None:
            raise RuntimeError("SuLoRA conditioner was garbage-collected before its adapters")
        return cond

    def _active_rows(self, batch: int) -> Optional[torch.Tensor]:
        """Rows for the current batch, or ``None`` when the adapter must be skipped."""
        rows = self.conditioner.current_rows()
        if rows is None:
            return None
        if rows.numel() != batch:
            raise ValueError(
                f"SuLoRA: subject_context holds {rows.numel()} rows but the batch is {batch}; "
                "did you call encoder.forward() outside subject_context?"
            )
        if not bool((rows >= 0).any()):
            return None
        return rows


class SuLoRALinear(_SuLoRABase):
    """``nn.Linear`` with a per-subject rank-``r`` additive weight correction.

    ``y = W x + b + (alpha / r) * B_s (A_s x)``, with ``B`` zero-initialised so the
    adapter starts (and stays, for unseen subjects) at exactly zero.
    """

    def __init__(
        self,
        base: nn.Linear,
        conditioner: "SuLoRA",
        n_subjects: int,
        rank: int,
        alpha: float = 1.0,
        init_std: float = 0.02,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(conditioner)
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_a = nn.Parameter(torch.randn(n_subjects, self.rank, base.in_features) * float(init_std))
        self.lora_b = nn.Parameter(torch.zeros(n_subjects, base.out_features, self.rank))
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        out = self.base(x)
        rows = self._active_rows(x.shape[0])
        if rows is None:
            return out
        idx = rows.clamp(min=0)
        a = self.lora_a[idx].to(dtype=x.dtype)  # (B, r, in)
        b = self.lora_b[idx].to(dtype=x.dtype)  # (B, out, r)
        xd = self.dropout(x)
        if xd.dim() == 2:  # (B, in)
            h = torch.einsum("bi,bri->br", xd, a)
            delta = torch.einsum("br,bor->bo", h, b)
        elif xd.dim() == 3:  # (B, N, in) -- token sequences
            h = torch.einsum("bni,bri->bnr", xd, a)
            delta = torch.einsum("bnr,bor->bno", h, b)
        else:
            raise ValueError(f"SuLoRALinear supports 2D/3D inputs, got {tuple(xd.shape)}")
        mask = SubjectConditioner._expand_mask(rows, delta.dim()).to(delta.dtype)
        return out + self.scaling * delta * mask

    def extra_repr(self) -> str:  # noqa: D102
        return f"rank={self.rank}, scaling={self.scaling:.4g}"


class SuLoRAConv2d(_SuLoRABase):
    """``nn.Conv2d`` with a per-subject rank-``r`` additive kernel correction.

    The correction is factorised as ``conv(x, A_s)`` (r output maps, same geometry as
    the base conv) followed by a ``1x1`` conv with ``B_s``.  Per-sample kernels are
    realised with the standard grouped-convolution trick (batch folded into groups),
    which is exact but noticeably slower than the base conv -- prefer targeting
    ``Linear`` layers unless the ablation specifically needs conv adapters.

    Only ``groups == 1`` base convolutions are supported; for a depthwise base conv the
    two-step factorisation is not equivalent and :class:`SuLoRA` skips it.
    """

    def __init__(
        self,
        base: nn.Conv2d,
        conditioner: "SuLoRA",
        n_subjects: int,
        rank: int,
        alpha: float = 1.0,
        init_std: float = 0.02,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(conditioner)
        if base.groups != 1:
            raise ValueError("SuLoRAConv2d requires groups == 1 in the base convolution")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        kh, kw = base.kernel_size
        self.lora_a = nn.Parameter(
            torch.randn(n_subjects, self.rank, base.in_channels, kh, kw) * float(init_std)
        )
        self.lora_b = nn.Parameter(torch.zeros(n_subjects, base.out_channels, self.rank, 1, 1))
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        out = self.base(x)
        rows = self._active_rows(x.shape[0])
        if rows is None:
            return out
        idx = rows.clamp(min=0)
        b_sz, c_in, h_in, w_in = x.shape
        a = self.lora_a[idx].to(dtype=x.dtype)  # (B, r, Cin, kh, kw)
        bm = self.lora_b[idx].to(dtype=x.dtype)  # (B, Cout, r, 1, 1)
        xg = self.dropout(x).reshape(1, b_sz * c_in, h_in, w_in)
        ag = a.reshape(b_sz * self.rank, c_in, *self.base.kernel_size)
        hg = F.conv2d(
            xg,
            ag,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=b_sz,
        )  # (1, B*r, H', W')
        bg = bm.reshape(b_sz * self.base.out_channels, self.rank, 1, 1)
        dg = F.conv2d(hg, bg, groups=b_sz)  # (1, B*Cout, H', W')
        delta = dg.reshape(b_sz, self.base.out_channels, dg.shape[-2], dg.shape[-1])
        mask = SubjectConditioner._expand_mask(rows, delta.dim()).to(delta.dtype)
        return out + self.scaling * delta * mask

    def extra_repr(self) -> str:  # noqa: D102
        return f"rank={self.rank}, scaling={self.scaling:.4g}"


@register_subject_conditioner("sulora", "lora", "adapter")
class SuLoRA(SubjectConditioner):
    """Per-subject low-rank additive corrections on selected trunk weights.

    Parameters
    ----------
    rank:
        Adapter rank ``r``.
    alpha:
        LoRA scaling; the effective multiplier is ``alpha / rank``.
    target_patterns:
        ``fnmatch`` patterns matched against *qualified module names* of the host
        encoder.  Default ``("*projector*",)`` adapts the projection head only, which
        is the cheap end of the ablation.  ``("*projector*", "*spatial*")`` also adapts
        the spatial convolution -- note the parameter cost: a ``(40, 64, 1)`` spatial
        kernel with ``r=8`` costs ``80 x 8 x 40 x 64 = 1.6M`` parameters, dwarfing the
        trunk.  Use ``("*",)`` to adapt everything eligible.
    target_types:
        Which module classes are eligible (``"linear"``, ``"conv1d"``, ``"conv2d"``).
    init_std, dropout:
        Adapter initialisation std (on the ``A`` factor only) and adapter-input dropout.
    allow_empty:
        If ``False`` (default) attaching with zero matched modules raises.  A SuLoRA run
        that silently matched nothing is indistinguishable from the ``none`` arm, which
        would corrupt the ablation table.

    Unseen rule
    -----------
    **Zero adapter**: ``B_s A_s = 0``, so the encoder collapses to its shared weights.

    Notes
    -----
    The adapter parameters live inside the *host encoder's* module tree (they replace
    the wrapped layers in place), not under this conditioner, so
    ``conditioner.n_parameters()`` reports 0 for SuLoRA.  Use
    :meth:`adapter_parameter_count` instead.
    """

    def __init__(
        self,
        *,
        rank: int = 8,
        alpha: float = 1.0,
        target_patterns: Sequence[str] = ("*projector*",),
        target_types: Sequence[str] = ("linear",),
        init_std: float = 0.02,
        dropout: float = 0.0,
        allow_empty: bool = False,
        **base: Any,
    ) -> None:
        super().__init__(**base)
        if int(rank) <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.init_std = float(init_std)
        self.adapter_dropout = float(dropout)
        self.allow_empty = bool(allow_empty)
        self.target_patterns = tuple(str(p) for p in target_patterns)
        types = {str(t).lower() for t in target_types}
        unknown = types - {"linear", "conv1d", "conv2d"}
        if unknown:
            raise ValueError(f"unknown target_types {sorted(unknown)}")
        self.target_types = tuple(sorted(types))
        self._attached = False
        self._attached_targets: List[str] = []
        # weak refs to the wrappers, for reporting; strong refs live in the encoder tree
        object.__setattr__(self, "_adapter_refs", [])

    # ---------------------------------------------------------------------------------

    def _eligible(self, module: nn.Module) -> bool:
        if isinstance(module, nn.Linear):
            return "linear" in self.target_types
        if isinstance(module, nn.Conv1d):
            return "conv1d" in self.target_types
        if isinstance(module, nn.Conv2d):
            return "conv2d" in self.target_types and module.groups == 1
        return False

    def _wrap(self, module: nn.Module) -> _SuLoRABase:
        kw = dict(
            conditioner=self,
            n_subjects=self.n_subjects,
            rank=self.rank,
            alpha=self.alpha,
            init_std=self.init_std,
            dropout=self.adapter_dropout,
        )
        if isinstance(module, nn.Linear):
            return SuLoRALinear(module, **kw)  # type: ignore[arg-type]
        if isinstance(module, nn.Conv2d):
            return SuLoRAConv2d(module, **kw)  # type: ignore[arg-type]
        if isinstance(module, nn.Conv1d):
            # A Conv1d over (B, C, T) is a Conv2d over (B, C, 1, T); wrap by view.
            raise NotImplementedError(
                "SuLoRA on nn.Conv1d is not implemented; the TACTUS encoders use Conv2d. "
                "Reshape the layer to Conv2d or restrict target_types."
            )
        raise TypeError(f"cannot wrap {type(module).__name__}")

    def _already_wrapped_ids(self, root: nn.Module) -> set:
        """Ids of every module living *inside* an existing adapter (never re-wrap those)."""
        inner: set = set()
        for mod in root.modules():
            if isinstance(mod, _SuLoRABase):
                inner.update(id(c) for c in mod.modules())
        return inner

    def _wrap_matching(
        self, root: nn.Module, patterns: Optional[Sequence[str]], prefix: str = ""
    ) -> List[str]:
        """Wrap every eligible leaf of ``root``; ``patterns=None`` means "wrap all"."""
        skip = self._already_wrapped_ids(root) | {id(m) for m in self.modules()}
        matched: List[Tuple[str, nn.Module]] = []
        for qname, module in list(root.named_modules()):
            if not qname or id(module) in skip or not self._eligible(module):
                continue
            if patterns is not None and not any(fnmatch.fnmatch(qname, p) for p in patterns):
                continue
            matched.append((qname, module))

        names: List[str] = []
        for qname, module in matched:
            parent_name, _, child = qname.rpartition(".")
            parent = root.get_submodule(parent_name) if parent_name else root
            wrapper = self._wrap(module)
            setattr(parent, child, wrapper)
            full = f"{prefix}.{qname}" if prefix else qname
            self._attached_targets.append(full)
            self._adapter_refs.append(weakref.ref(wrapper))
            names.append(full)
        return names

    def attach(self, model: nn.Module) -> None:
        """Replace matching leaf modules of ``model`` with per-subject adapted versions."""
        if self._attached:
            raise RuntimeError("SuLoRA.attach() called twice on the same conditioner")
        self._wrap_matching(model, self.target_patterns)
        self._attached = True
        if not self._attached_targets and not self.allow_empty:
            raise RuntimeError(
                f"SuLoRA matched no modules with patterns {list(self.target_patterns)} and types "
                f"{list(self.target_types)}. Candidate module names: "
                f"{[n for n, m in model.named_modules() if self._eligible(m)][:20]}"
            )

    # ---------------------------------------------------------------------------------

    def adapt_module(self, module: nn.Module, name: str = "") -> int:
        """Adapt a head built after :meth:`attach` (e.g. a ``TimeWindowHeads`` head).

        ``target_patterns`` is deliberately *not* applied: the caller has explicitly
        designated this module, and the pattern language addresses the encoder's own
        namespace.  Without this, running the SuLoRA arm with time-window heads would
        yield window embeddings with no subject conditioning at all -- an ablation
        silently reduced to the ``none`` arm.
        """
        return len(self._wrap_matching(module, patterns=None, prefix=name))

    def relink(self, model: nn.Module) -> None:
        """Re-point every adapter inside ``model`` at *this* conditioner (post-deepcopy)."""
        refs: List[Any] = []
        names: List[str] = []
        for qname, mod in model.named_modules():
            if isinstance(mod, _SuLoRABase):
                object.__setattr__(mod, "_cond_ref", weakref.ref(self))
                refs.append(weakref.ref(mod))
                names.append(qname)
        self._adapter_refs = refs
        self._attached_targets = names
        self._attached = True

    @property
    def attached_targets(self) -> List[str]:
        """Qualified names of the modules that received an adapter."""
        return list(self._attached_targets)

    def adapter_parameters(self) -> List[nn.Parameter]:
        """The per-subject adapter parameters (they live in the host encoder tree)."""
        params: List[nn.Parameter] = []
        for ref in self._adapter_refs:
            mod = ref()
            if mod is None:
                continue
            params.extend([mod.lora_a, mod.lora_b])
        return params

    def adapter_parameter_count(self) -> int:
        """Total number of per-subject adapter parameters."""
        return sum(p.numel() for p in self.adapter_parameters())

    def unseen_subject_state(self) -> Dict[str, Any]:  # noqa: D102
        return {
            "mechanism": "sulora",
            "rule": "zero_adapter",
            "description": (
                "An unseen subject uses the zero adapter (B_s A_s = 0), i.e. the shared "
                f"backbone weights only. rank={self.rank}, alpha={self.alpha}, "
                f"targets={self._attached_targets}"
            ),
            "fixed_a_priori": True,
            "n_train_subjects": int(self._train_mask.sum()),
            "state": {"delta_w": torch.zeros(1)},
        }

    def extra_repr(self) -> str:  # noqa: D102
        return (
            super().extra_repr()
            + f", rank={self.rank}, alpha={self.alpha}, patterns={list(self.target_patterns)}, "
            f"attached={len(self._attached_targets)}"
        )
