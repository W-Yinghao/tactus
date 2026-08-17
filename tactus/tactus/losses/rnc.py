"""Rank-N-Contrast (Zha et al., NeurIPS 2023) for continuous labels.

Ordinary contrastive losses need a binary same/different verdict.  Valence,
arousal and threat are continuous, and thresholding them into classes throws
away exactly the structure worth learning.  RnC instead enforces a *ranking*:
for anchor ``i`` and any other sample ``j``, the embedding similarity
``s(i, j)`` should exceed ``s(i, k)`` for every ``k`` whose label is at least as
far from ``y_i`` as ``y_j`` is::

    L = mean over (i, j) of  -log[ exp(s_ij / T) / sum_{k in S_ij} exp(s_ik / T) ]
    S_ij = { k != i : |y_i - y_k| >= |y_i - y_j| }

The result is an embedding whose geometry is ordered by the label rather than
merely clustered, which is what the affective-axis analysis needs.

Scope discipline
----------------
The affective attributes are properties of the *video*, and on 90 clips they are
strongly inter-correlated with material and touch type (a knife on skin is
metal, threatening, and negative all at once).  This loss therefore cannot on
its own establish that the model encodes valence rather than a correlated
low-level property.  It is a representation-shaping term to be run with a small
weight inside a composite, and the attribute cross-correlation matrix is the
thing that licenses any interpretation of what it did.

Numerics
--------
``similarity="l2"`` follows the paper (negative Euclidean distance).  The
distance is computed via :func:`tactus.losses.base.safe_pdist`, which adds an
epsilon inside the square root: the diagonal of a self-distance matrix is
exactly zero, ``d/dx sqrt(x)`` is infinite there, and ``0 * inf`` is NaN even
though the diagonal is masked out downstream.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Union

import torch

from .base import (
    ContrastiveLoss,
    TemperatureMixin,
    get_meta,
    masked_logsumexp,
    register_loss,
    safe_pdist,
    to_float,
)

_VALID_LABELS = ("valence", "arousal", "threat")
_VALID_FEATURES = ("eeg", "vid", "cross")


@register_loss("rnc")
class RankNContrast(ContrastiveLoss, TemperatureMixin):
    """Ranking-based contrast over a continuous attribute.

    Parameters
    ----------
    label_key
        ``"valence"``, ``"arousal"``, ``"threat"``, or a list of them (the
        per-key losses are averaged and each is logged separately).
    feature
        Which embedding is shaped: ``"eeg"`` (default), ``"vid"``, or
        ``"cross"`` (EEG anchors against video candidates).  ``"eeg"`` gives the
        video projector no gradient, so run it inside a composite.
    similarity
        ``"l2"`` (paper: negative Euclidean distance) or ``"cosine"``.  With
        L2-normalized embeddings the two are monotonically related, so they
        differ only in how the temperature acts.
    temperature, learnable_temperature, max_scale, min_scale
        RnC typically uses a temperature near 2, i.e. a logit scale near 0.5 --
        note this is *below* the CLIP-style default range, which is why
        ``min_scale`` defaults to 0.01 here.
    chunk_size
        Anchors processed at once.  The ranking set is an ``(chunk, B, B)``
        tensor, so this caps peak memory at roughly
        ``chunk * B * B * 4`` bytes.
    label_tolerance
        Absolute tolerance when testing ``|y_i - y_k| >= |y_i - y_j|``.  Guards
        against float noise splitting exact ties.
    """

    requires_video = False

    def __init__(
        self,
        label_key: Union[str, Sequence[str]] = "valence",
        feature: str = "eeg",
        similarity: str = "l2",
        temperature: float = 2.0,
        learnable_temperature: bool = False,
        max_scale: float = 100.0,
        min_scale: float = 0.001,
        chunk_size: int = 64,
        label_tolerance: float = 1e-6,
        renormalize: bool = True,
    ) -> None:
        super().__init__()
        keys: List[str] = [label_key] if isinstance(label_key, str) else list(label_key)
        bad = [k for k in keys if k not in _VALID_LABELS]
        if bad:
            raise ValueError(f"label_key {bad} not in {_VALID_LABELS}")
        if not keys:
            raise ValueError("label_key must not be empty")
        if feature not in _VALID_FEATURES:
            raise ValueError(f"feature must be one of {_VALID_FEATURES}, got {feature!r}")
        if similarity not in ("l2", "cosine"):
            raise ValueError(f"similarity must be 'l2' or 'cosine', got {similarity!r}")

        self._init_temperature(
            temperature,
            learnable=learnable_temperature,
            max_scale=max_scale,
            min_scale=min_scale,
        )
        self.label_keys = keys
        self.feature = feature
        self.similarity = similarity
        self.chunk_size = int(chunk_size)
        self.label_tolerance = float(label_tolerance)
        self.renormalize = bool(renormalize)
        self.requires_meta = tuple(keys)
        self.requires_video = feature in ("vid", "cross")

    # ------------------------------------------------------------------ #

    def _similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.similarity == "cosine":
            return a @ b.transpose(0, 1)
        if a is b:
            return -safe_pdist(a)
        x2 = (a * a).sum(-1, keepdim=True)
        y2 = (b * b).sum(-1, keepdim=True).transpose(0, 1)
        d2 = (x2 + y2 - 2.0 * (a @ b.transpose(0, 1))).clamp_min(0.0)
        return -torch.sqrt(d2 + 1e-12)

    def _one_label_loss(
        self, sim: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
    ):
        """RnC for a single continuous label.  ``sim`` is unscaled ``(B, B)``."""
        b = y.shape[0]
        device = sim.device
        finite = torch.isfinite(y)
        eye = torch.eye(b, dtype=torch.bool, device=device)

        d = (y.view(-1, 1) - y.view(1, -1)).abs()  # (B, B)
        d = torch.nan_to_num(d, nan=float("inf"))
        s = sim * scale  # scale == 1 / temperature

        total = sim.sum() * 0.0
        n_terms = 0
        tol = self.label_tolerance

        for start in range(0, b, max(self.chunk_size, 1)):
            stop = min(start + max(self.chunk_size, 1), b)
            d_i = d[start:stop]  # (c, B) -> d_i[i, k] = |y_i - y_k|
            s_i = s[start:stop]  # (c, B)
            eye_i = eye[start:stop]  # (c, B)

            # incl[i, j, k] = ( d[i, k] >= d[i, j] - tol )  and  k != i
            incl = d_i.unsqueeze(1) >= (d_i.unsqueeze(2) - tol)  # (c, B, B)
            incl = incl & ~eye_i.unsqueeze(1)
            incl = incl & finite.view(1, 1, -1)

            log_z = masked_logsumexp(
                s_i.unsqueeze(1).expand(-1, b, -1), incl, dim=-1
            )  # (c, B)
            term = -(s_i - log_z)  # (c, B), indexed [i, j]

            # j must be a real, distinct, finitely-labeled partner, and its own
            # ranking set must be non-empty (it always contains j itself).
            valid_j = (~eye_i) & finite.view(1, -1) & finite[start:stop].view(-1, 1)
            valid_j = valid_j & incl.any(dim=-1)

            total = total + (term * valid_j.to(term.dtype)).sum()
            n_terms += int(valid_j.sum())

        if n_terms == 0:
            return sim.sum() * 0.0 + 0.0, 0
        return total / float(n_terms), n_terms

    def forward(
        self,
        z_eeg: torch.Tensor,
        z_vid: torch.Tensor,
        meta: Mapping[str, torch.Tensor],
    ) -> Dict[str, Any]:
        z_eeg, z_vid = self._prepare(
            z_eeg, z_vid if self.requires_video else None, self.renormalize
        )
        b, device = z_eeg.shape[0], z_eeg.device

        if b < 3:
            # With fewer than 3 samples the ranking set is degenerate: every
            # candidate is trivially the farthest, so the loss is a constant.
            return {
                "loss": self._zero_loss(z_eeg, z_vid),
                "logs": {"loss": 0.0, "n_terms": 0.0, "degenerate": 1.0},
            }

        if self.feature == "eeg":
            sim = self._similarity(z_eeg, z_eeg)
        elif self.feature == "vid":
            assert z_vid is not None
            sim = self._similarity(z_vid, z_vid)
        else:
            assert z_vid is not None
            sim = self._similarity(z_eeg, z_vid)

        scale = self.scale()
        total = None
        logs: Dict[str, float] = {}
        n_total = 0
        for key in self.label_keys:
            y = get_meta(
                meta, key, device=device, batch_size=b, dtype=torch.float32
            )
            assert y is not None
            loss_k, n_k = self._one_label_loss(sim, y, scale)
            logs[f"loss_{key}"] = to_float(loss_k)
            logs[f"n_terms_{key}"] = float(n_k)
            n_total += n_k
            total = loss_k if total is None else total + loss_k

        assert total is not None
        loss = total / float(len(self.label_keys))

        if n_total == 0:
            return {
                "loss": self._zero_loss(z_eeg, z_vid),
                "logs": {**logs, "loss": 0.0, "n_terms": 0.0, "degenerate": 1.0},
            }

        logs.update(
            {
                "loss": to_float(loss),
                "logit_scale": to_float(scale),
                "temperature": self.temperature,
                "n_terms": float(n_total),
                "degenerate": 0.0,
            }
        )
        return {"loss": loss, "logs": logs}

    def extra_repr(self) -> str:
        return (
            f"label_keys={self.label_keys}, feature={self.feature}, "
            f"similarity={self.similarity}, temperature={self.temperature:.3f}, "
            f"chunk_size={self.chunk_size}"
        )


if __name__ == "__main__":  # pragma: no cover
    from .base import make_dummy_batch

    torch.manual_seed(0)
    for tag, fn in [
        ("valence/eeg/l2", RankNContrast()),
        ("arousal/eeg/cos", RankNContrast(label_key="arousal", similarity="cosine")),
        ("all-three/cross", RankNContrast(label_key=list(_VALID_LABELS), feature="cross")),
        ("chunked", RankNContrast(chunk_size=7)),
    ]:
        for btag, kw in [
            ("random", {}),
            ("all-same-video", {"single_condition": True}),  # every label identical
            ("B=2", {"batch_size": 2}),
        ]:
            ze, zv, m = make_dummy_batch(dim=256, **{"batch_size": 32, **kw})
            out = fn(ze, zv, m)
            out["loss"].backward()
            grad_ok = torch.isfinite(ze.grad).all().item()
            print(
                f"[rnc/{tag:18s}/{btag:16s}] loss={out['loss'].item():.4f} "
                f"n_terms={out['logs']['n_terms']:.0f} grad_finite={grad_ok}"
            )
