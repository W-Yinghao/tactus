"""Supervised contrastive loss (Khosla et al., NeurIPS 2020) with multi-positive.

Positives are every other sample sharing a configurable label key, rather than
only the paired view.  On TACTUS the natural keys form a granularity ladder:

``condition_id``  360 classes -- orientation-specific, the tightest grouping
``video_id``       90 classes -- pools the 4 orientations, i.e. asks the encoder
                                 to be orientation-*invariant*
``material_id``     8 classes -- the coarse semantic axis the retrieval metric
                                 is most at risk of collapsing onto

Choosing ``video_id`` here is the same scientific fork as
``mask_same_video`` in :mod:`tactus.losses.masked_infonce`, stated positively:
it actively pulls mirrored versions of a clip together instead of merely
declining to push them apart.

The ``L_out`` formulation is used (positives outside the log), which Khosla et
al. show is the better-behaved of the two.

A TACTUS-specific wrinkle
-------------------------
In ``multiview`` mode the canonical loss treats the two views symmetrically.
Here the "second view" is a *frozen* video embedding, so two trials of the same
condition have video vectors that are numerically identical.  A video-video
positive pair therefore has similarity exactly 1.0 and saturates the softmax
denominator, drowning out every informative term.  ``drop_vid_vid_pairs``
(default ``True``) removes those pairs from both the positive set and the
denominator.  Turn it off only to reproduce a textbook SupCon.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .base import (
    ContrastiveLoss,
    TemperatureMixin,
    get_meta,
    masked_log_softmax,
    pairwise_eq,
    register_loss,
    to_float,
)

_VALID_KEYS = ("condition_id", "video_id", "material_id", "touch_type_id", "toucher_id", "pain")
_VALID_MODES = ("multiview", "cross_modal", "eeg_only")


def supcon_core(
    sim: torch.Tensor,
    pos_mask: torch.Tensor,
    cand_mask: torch.Tensor,
    anchor_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SupCon ``L_out`` given a pre-scaled similarity matrix.

    Parameters
    ----------
    sim
        ``(N, M)`` similarities already divided by the temperature.
    pos_mask
        ``(N, M)`` True where column ``j`` is a positive for anchor ``i``.
        Intersected with ``cand_mask`` internally.
    cand_mask
        ``(N, M)`` True where column ``j`` may appear in the denominator of
        anchor ``i`` (self-comparisons already removed by the caller).
    anchor_valid
        ``(N,)`` True where row ``i`` is allowed to contribute at all.

    Returns
    -------
    (loss, per_anchor, valid)
        ``loss`` is the mean over valid anchors, or a hard zero tensor if none
        are valid (callers should substitute ``_zero_loss`` in that case).
    """
    pos_mask = pos_mask & cand_mask
    logp = masked_log_softmax(sim, cand_mask, dim=1)
    n_pos = pos_mask.sum(dim=1)
    valid = anchor_valid & (n_pos > 0) & cand_mask.any(dim=1)
    per_anchor = -(logp * pos_mask.to(logp.dtype)).sum(dim=1) / n_pos.clamp_min(1)
    n_valid = valid.sum()
    if int(n_valid) == 0:
        return sim.sum() * 0.0 + 0.0, per_anchor, valid
    loss = (per_anchor * valid.to(per_anchor.dtype)).sum() / n_valid.to(per_anchor.dtype)
    return loss, per_anchor, valid


@register_loss("supcon")
class SupCon(ContrastiveLoss, TemperatureMixin):
    """Multi-positive supervised contrast.

    Parameters
    ----------
    positive_key
        Meta key whose equality defines a positive pair.  One of
        ``{"condition_id", "video_id", "material_id", "touch_type_id",
        "toucher_id", "pain"}``.
    mode
        ``"multiview"``  -- anchors are the 2B concatenation of EEG and video
                            embeddings (canonical Khosla).
        ``"cross_modal"``-- EEG anchors against video candidates and back;
                            equivalent to masked InfoNCE with multiple positives.
        ``"eeg_only"``   -- EEG against EEG; no gradient reaches the video
                            projector, so use it inside a composite.
    drop_vid_vid_pairs
        See the module docstring.  Only meaningful in ``multiview``.
    temperature, learnable_temperature, max_scale, min_scale
        See :class:`tactus.losses.base.TemperatureMixin`.
    base_temperature
        Khosla's ``tau / tau_base`` gradient rescale.  ``None`` (default)
        disables it.  With a learnable temperature the factor is taken from the
        current detached value, so it acts as a slowly-varying constant.
    renormalize
        Defensive input re-normalization.
    """

    def __init__(
        self,
        positive_key: str = "condition_id",
        mode: str = "multiview",
        drop_vid_vid_pairs: bool = True,
        temperature: float = 0.1,
        learnable_temperature: bool = True,
        max_scale: float = 100.0,
        min_scale: float = 0.01,
        base_temperature: Optional[float] = None,
        renormalize: bool = True,
    ) -> None:
        super().__init__()
        if positive_key not in _VALID_KEYS:
            raise ValueError(
                f"positive_key must be one of {_VALID_KEYS}, got {positive_key!r}"
            )
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self._init_temperature(
            temperature,
            learnable=learnable_temperature,
            max_scale=max_scale,
            min_scale=min_scale,
        )
        self.positive_key = positive_key
        self.mode = mode
        self.drop_vid_vid_pairs = bool(drop_vid_vid_pairs)
        self.base_temperature = base_temperature
        self.renormalize = bool(renormalize)
        self.requires_meta = (positive_key,)
        self.requires_video = mode != "eeg_only"

    # ------------------------------------------------------------------ #

    def forward(
        self,
        z_eeg: torch.Tensor,
        z_vid: torch.Tensor,
        meta: Mapping[str, torch.Tensor],
    ) -> Dict[str, Any]:
        z_eeg, z_vid = self._prepare(
            z_eeg, z_vid if self.mode != "eeg_only" else None, self.renormalize
        )
        b, device = z_eeg.shape[0], z_eeg.device
        labels = get_meta(meta, self.positive_key, device=device, batch_size=b)
        assert labels is not None
        scale = self.scale()

        if b < 2:
            return {
                "loss": self._zero_loss(z_eeg, z_vid),
                "logs": {"loss": 0.0, "n_valid": 0.0, "degenerate": 1.0},
            }

        if self.mode == "eeg_only":
            feats = z_eeg
            all_labels = labels
            is_vid = torch.zeros(b, dtype=torch.bool, device=device)
        elif self.mode == "multiview":
            assert z_vid is not None
            feats = torch.cat([z_eeg, z_vid], dim=0)  # (2B, D)
            all_labels = torch.cat([labels, labels], dim=0)
            is_vid = torch.cat(
                [
                    torch.zeros(b, dtype=torch.bool, device=device),
                    torch.ones(b, dtype=torch.bool, device=device),
                ]
            )
        else:  # cross_modal
            assert z_vid is not None
            feats = None  # handled below
            all_labels = labels
            is_vid = torch.zeros(b, dtype=torch.bool, device=device)

        if self.mode == "cross_modal":
            assert z_vid is not None
            sim = scale * (z_eeg @ z_vid.transpose(0, 1))  # (B, B)
            pos = pairwise_eq(labels)
            cand = torch.ones_like(pos)
            valid_anchor = torch.ones(b, dtype=torch.bool, device=device)
            loss_a, per_a, va = supcon_core(sim, pos, cand, valid_anchor)
            loss_b, per_b, vb = supcon_core(
                sim.transpose(0, 1), pos.transpose(0, 1), cand, valid_anchor
            )
            n_valid = int(va.sum()) + int(vb.sum())
            loss = 0.5 * (loss_a + loss_b)
            n_pos_mean = pos.sum(dim=1).float().mean()
        else:
            assert feats is not None
            n = feats.shape[0]
            sim = scale * (feats @ feats.transpose(0, 1))  # (N, N)
            eye = torch.eye(n, dtype=torch.bool, device=device)
            pos = pairwise_eq(all_labels) & ~eye
            cand = ~eye
            if self.mode == "multiview" and self.drop_vid_vid_pairs:
                vv = is_vid.view(-1, 1) & is_vid.view(1, -1)
                pos = pos & ~vv
                cand = cand & ~vv
            valid_anchor = torch.ones(n, dtype=torch.bool, device=device)
            loss, per_a, va = supcon_core(sim, pos, cand, valid_anchor)
            n_valid = int(va.sum())
            n_pos_mean = pos.sum(dim=1).float().mean()

        if n_valid == 0:
            # e.g. every sample has a distinct label -> no positives anywhere.
            return {
                "loss": self._zero_loss(z_eeg, z_vid),
                "logs": {
                    "loss": 0.0,
                    "n_valid": 0.0,
                    "degenerate": 1.0,
                    "mean_n_pos": to_float(n_pos_mean),
                    "logit_scale": to_float(scale),
                },
            }

        if self.base_temperature is not None:
            loss = loss * (self.temperature / float(self.base_temperature))

        with torch.no_grad():
            n_classes = int(labels.unique().numel())

        return {
            "loss": loss,
            "logs": {
                "loss": to_float(loss),
                "logit_scale": to_float(scale),
                "temperature": self.temperature,
                "mean_n_pos": to_float(n_pos_mean),
                "n_classes_in_batch": float(n_classes),
                "n_valid": float(n_valid),
                "degenerate": 0.0,
            },
        }

    def extra_repr(self) -> str:
        return (
            f"positive_key={self.positive_key}, mode={self.mode}, "
            f"temperature={self.temperature:.4f}, "
            f"drop_vid_vid_pairs={self.drop_vid_vid_pairs}"
        )


if __name__ == "__main__":  # pragma: no cover
    from .base import make_dummy_batch

    torch.manual_seed(0)
    for key in ("condition_id", "video_id", "material_id"):
        for mode in _VALID_MODES:
            fn = SupCon(positive_key=key, mode=mode)
            for tag, kw in [
                ("random", {}),
                ("all-distinct", {"n_unique_conditions": 32}),
                ("all-same-condition", {"single_condition": True}),
            ]:
                ze, zv, m = make_dummy_batch(dim=256, **{"batch_size": 32, **kw})
                out = fn(ze, zv, m)
                out["loss"].backward()
                print(
                    f"[supcon/{key:13s}/{mode:11s}/{tag:18s}] "
                    f"loss={out['loss'].item():.4f} "
                    f"n_pos={out['logs'].get('mean_n_pos', 0):.2f} "
                    f"n_valid={out['logs']['n_valid']:.0f}"
                )
