"""Pairwise sigmoid loss (SigLIP, Zhai et al., ICCV 2023) with per-pair labels.

SigLIP replaces the softmax over a batch with an independent binary decision per
pair::

    L = -1/B * sum_i sum_j  logsigmoid( y_ij * (t * <x_i, y_j> + b) )

with ``y_ij = +1`` for a positive pair and ``-1`` otherwise, a learnable scale
``t`` and a learnable bias ``b`` initialized very negative to counteract the
overwhelming prior of negatives.

Why this form suits TACTUS
--------------------------
Because the decision is per-pair rather than per-row, a duplicate condition in
the batch is not a contradiction to be masked out -- it is simply *labeled
positive*.  ``positive_key`` controls that labeling, so the loss expresses the
dense-repeat structure of the design directly instead of patching around it.
There is no normalization over candidates, so the objective does not shift when
the sampler happens to draw a different number of duplicates.

The ``ignore_*`` options mark pairs as neither positive nor negative and drop
them from the sum.  Same base video across orientations is the case that
matters: calling those pairs negative trains mirror discrimination (and the eye
movements that come with it), calling them positive destroys the equivariance
signal, so abstaining is a defensible third option unavailable to softmax
losses.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    ContrastiveLoss,
    TemperatureMixin,
    get_meta,
    pairwise_eq,
    register_loss,
    to_float,
)


@register_loss("siglip")
class SigLIP(ContrastiveLoss, TemperatureMixin):
    """Sigmoid pairwise contrastive loss.

    Parameters
    ----------
    positive_key
        Meta key whose equality marks a pair positive.  ``"condition_id"``
        (default) labels same-condition pairs positive rather than negative.
        Pass ``None`` to use the identity (diagonal-only) labeling, which
        reproduces vanilla SigLIP.
    ignore_same_video
        Drop same-``video_id``/different-``condition_id`` pairs from the sum
        entirely (neither positive nor negative).
    ignore_same_material
        Same treatment for same-``material_id`` pairs.  Off by default; material
        is the shortcut axis we want measured, not hidden.
    temperature, learnable_temperature, max_scale, min_scale
        Scale ``t``; SigLIP initializes it to 10 (``temperature = 0.1``).
    bias_init
        Initial value of ``b``.  Zhai et al. use -10; with a positive rate of
        roughly ``1/B`` this cancels the class imbalance at initialization.
    learnable_bias
        Whether ``b`` is optimized.
    pos_weight
        Multiplier on positive-pair terms.  Raise it when ``positive_key``
        yields very few positives per row and the loss is dominated by the
        negatives.
    normalize_by_pairs
        ``False`` (default) reproduces the paper: sum over ``j``, mean over
        ``i``, so the loss scales with batch size.  ``True`` divides by the
        number of contributing pairs, giving a batch-size-invariant number that
        is easier to compare across ablations but is not the published loss.
    """

    requires_video = True

    def __init__(
        self,
        positive_key: Optional[str] = "condition_id",
        ignore_same_video: bool = False,
        ignore_same_material: bool = False,
        temperature: float = 0.1,
        learnable_temperature: bool = True,
        max_scale: float = 100.0,
        min_scale: float = 0.01,
        bias_init: float = -10.0,
        learnable_bias: bool = True,
        pos_weight: float = 1.0,
        normalize_by_pairs: bool = False,
        renormalize: bool = True,
    ) -> None:
        super().__init__()
        self._init_temperature(
            temperature,
            learnable=learnable_temperature,
            max_scale=max_scale,
            min_scale=min_scale,
        )
        bias = torch.tensor(float(bias_init), dtype=torch.float32)
        if learnable_bias:
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer("bias", bias)
        self.positive_key = positive_key
        self.ignore_same_video = bool(ignore_same_video)
        self.ignore_same_material = bool(ignore_same_material)
        self.pos_weight = float(pos_weight)
        self.normalize_by_pairs = bool(normalize_by_pairs)
        self.renormalize = bool(renormalize)

        needed = []
        if positive_key is not None:
            needed.append(positive_key)
        if ignore_same_video:
            needed.extend(["video_id", "condition_id"])
        if ignore_same_material:
            needed.append("material_id")
        self.requires_meta = tuple(dict.fromkeys(needed))

    def _pair_labels(
        self, meta: Mapping[str, torch.Tensor], b: int, device: torch.device
    ):
        """Return ``(labels, active, pos)``.

        ``labels`` holds the ``+1 / -1`` sign of every pair, ``active`` marks
        pairs that contribute at all, and ``pos`` is the boolean positive mask
        (kept separately for logging and for ``pos_weight``).
        """
        eye = torch.eye(b, dtype=torch.bool, device=device)
        if self.positive_key is None:
            pos = eye.clone()
        else:
            ids = get_meta(meta, self.positive_key, device=device, batch_size=b)
            assert ids is not None
            pos = pairwise_eq(ids) | eye  # the paired diagonal is always positive

        active = torch.ones(b, b, dtype=torch.bool, device=device)
        if self.ignore_same_video:
            vid = get_meta(meta, "video_id", device=device, batch_size=b)
            cond = get_meta(meta, "condition_id", device=device, batch_size=b)
            assert vid is not None and cond is not None
            same_video_diff_cond = pairwise_eq(vid) & ~pairwise_eq(cond)
            active = active & ~same_video_diff_cond
        if self.ignore_same_material:
            mat = get_meta(meta, "material_id", device=device, batch_size=b)
            assert mat is not None
            active = active & ~(pairwise_eq(mat) & ~pos)
        active = active | eye  # never ignore the true pair

        labels = pos.to(torch.float32) * 2.0 - 1.0  # True -> +1, False -> -1
        return labels, active, pos

    def forward(
        self,
        z_eeg: torch.Tensor,
        z_vid: torch.Tensor,
        meta: Mapping[str, torch.Tensor],
    ) -> Dict[str, Any]:
        z_eeg, z_vid = self._prepare(z_eeg, z_vid, self.renormalize)
        assert z_vid is not None
        b, device = z_eeg.shape[0], z_eeg.device

        scale = self.scale()
        logits = scale * (z_eeg @ z_vid.transpose(0, 1)) + self.bias  # (B, B)
        labels, active, pos = self._pair_labels(meta, b, device)

        # Per-pair weight: pos_weight on positives, 1 on negatives, 0 if ignored.
        w = torch.where(
            pos,
            torch.full_like(logits, self.pos_weight),
            torch.ones_like(logits),
        ) * active.to(logits.dtype)

        w_sum = w.sum()
        if float(w_sum) <= 0.0:
            return {
                "loss": self._zero_loss(z_eeg, z_vid),
                "logs": {"loss": 0.0, "n_valid": 0.0, "degenerate": 1.0},
            }

        per_pair = -F.logsigmoid(labels.to(logits.dtype) * logits)  # (B, B)
        if self.normalize_by_pairs:
            loss = (per_pair * w).sum() / w_sum
        else:
            loss = (per_pair * w).sum() / float(b)

        with torch.no_grad():
            n_pos = (pos & active).sum()
            n_neg = (~pos & active).sum()
            probs = torch.sigmoid(logits)
            pos_p = probs[pos & active].mean() if int(n_pos) > 0 else torch.zeros((), device=device)
            neg_p = probs[~pos & active].mean() if int(n_neg) > 0 else torch.zeros((), device=device)
            acc = (
                (logits.argmax(dim=1) == torch.arange(b, device=device)).float().mean()
            )

        return {
            "loss": loss,
            "logs": {
                "loss": to_float(loss),
                "logit_scale": to_float(scale),
                "temperature": self.temperature,
                "bias": to_float(self.bias),
                "n_pos_pairs": float(int(n_pos)),
                "n_neg_pairs": float(int(n_neg)),
                "pos_prob": to_float(pos_p),
                "neg_prob": to_float(neg_p),
                "ignored_frac": to_float((~active).float().mean()),
                "acc_e2v": to_float(acc),
                "n_valid": float(b),
                "degenerate": 0.0,
            },
        }

    def extra_repr(self) -> str:
        return (
            f"positive_key={self.positive_key}, bias={float(self.bias):.3f}, "
            f"temperature={self.temperature:.4f}, "
            f"ignore_same_video={self.ignore_same_video}"
        )


if __name__ == "__main__":  # pragma: no cover
    from .base import make_dummy_batch

    torch.manual_seed(0)
    for tag, fn in [
        ("cond-positive", SigLIP()),
        ("vanilla-diag", SigLIP(positive_key=None)),
        ("ignore-same-video", SigLIP(ignore_same_video=True)),
        ("normalized", SigLIP(normalize_by_pairs=True)),
    ]:
        for btag, kw in [
            ("random", {}),
            ("many-dups", {"n_unique_conditions": 4}),
            ("all-same-condition", {"single_condition": True}),
            ("B=1", {"batch_size": 1}),
        ]:
            ze, zv, m = make_dummy_batch(dim=256, **{"batch_size": 32, **kw})
            out = fn(ze, zv, m)
            out["loss"].backward()
            print(
                f"[siglip/{tag:18s}/{btag:20s}] loss={out['loss'].item():.4f} "
                f"n_pos={out['logs']['n_pos_pairs']:.0f} "
                f"ignored={out['logs'].get('ignored_frac', 0):.3f}"
            )
