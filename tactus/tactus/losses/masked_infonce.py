"""InfoNCE with in-batch false negatives removed from the denominator.

Why this exists
---------------
TACTUS presents 360 conditions to every subject 8 times.  Any batch large
enough to be useful contains repeats, and because the video tower is frozen the
video embeddings of two trials of the same condition are *identical vectors*.
Plain InfoNCE (:mod:`tactus.losses.infonce`) then asks the encoder to make
``z_eeg[i]`` closer to ``z_vid[i]`` than to ``z_vid[j]`` when
``z_vid[i] == z_vid[j]`` -- an objective with no solution, contributing pure
gradient noise plus an irreducible floor on the reported loss.

This module drops those columns from the softmax denominator.  Two granularities
are available and they encode a *scientific* choice, not a hyperparameter:

``mask_same_condition`` (default True)
    Removes exact duplicates.  Nobody disputes this one.

``mask_same_video`` (default False)
    Also removes the other three orientations of the anchor's own base video.
    This is the "flips are positives" side of the orientation fork: it forbids
    the encoder from using mirror-discriminative information, which is where the
    eye-movement confound lives, at the cost of destroying the equivariance
    signal the orientation analysis needs.  Leave it False for the main model
    and flip it on for the orientation-policy pilot.

Masking is implemented by filling the excluded logits with a large finite
negative (see :func:`tactus.losses.base.mask_value`), not ``-inf``: a row whose
every off-diagonal entry is masked would otherwise produce NaN.  Such rows are
detected and excluded from the mean instead.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

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


@register_loss("masked_infonce")
class MaskedInfoNCE(ContrastiveLoss, TemperatureMixin):
    """Symmetric InfoNCE that refuses to treat known duplicates as negatives.

    Parameters
    ----------
    mask_same_condition
        Exclude off-diagonal pairs with equal ``condition_id``.
    mask_same_video
        Exclude off-diagonal pairs with equal ``video_id`` (i.e. also the other
        orientations).  Implies the condition-level mask.
    mask_same_material
        Additionally exclude same-``material_id`` pairs.  Off by default and
        rarely a good idea -- material is the shortcut axis we want the model to
        be *challenged* on, and masking it inflates retrieval by removing the
        hardest distractors.  Exposed only so the ablation can be run.
    extra_mask_keys
        Any further meta keys whose equality marks a pair as a false negative.
    temperature, learnable_temperature, max_scale, min_scale
        See :class:`tactus.losses.base.TemperatureMixin`.
    direction
        ``"both"``, ``"e2v"`` or ``"v2e"``.
    renormalize
        Defensive input re-normalization.
    """

    requires_video = True

    def __init__(
        self,
        mask_same_condition: bool = True,
        mask_same_video: bool = False,
        mask_same_material: bool = False,
        extra_mask_keys: Optional[list[str]] = None,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        max_scale: float = 100.0,
        min_scale: float = 0.01,
        direction: str = "both",
        renormalize: bool = True,
    ) -> None:
        super().__init__()
        self._init_temperature(
            temperature,
            learnable=learnable_temperature,
            max_scale=max_scale,
            min_scale=min_scale,
        )
        if direction not in ("both", "e2v", "v2e"):
            raise ValueError(
                f"direction must be 'both', 'e2v' or 'v2e', got {direction!r}"
            )
        self.mask_same_condition = bool(mask_same_condition)
        self.mask_same_video = bool(mask_same_video)
        self.mask_same_material = bool(mask_same_material)
        self.extra_mask_keys = list(extra_mask_keys or [])
        self.direction = direction
        self.renormalize = bool(renormalize)

        keys: list[str] = []
        if self.mask_same_condition:
            keys.append("condition_id")
        if self.mask_same_video:
            keys.append("video_id")
        if self.mask_same_material:
            keys.append("material_id")
        keys.extend(self.extra_mask_keys)
        self.mask_keys = keys
        self.requires_meta = tuple(dict.fromkeys(keys))

    # ------------------------------------------------------------------ #

    def _false_negative_mask(
        self, meta: Mapping[str, torch.Tensor], b: int, device: torch.device
    ) -> torch.Tensor:
        """``(B, B)`` True where a pair must be dropped from the denominator.

        The diagonal is forced False: the positive always stays in.
        """
        mask = torch.zeros(b, b, dtype=torch.bool, device=device)
        for key in self.mask_keys:
            ids = get_meta(meta, key, device=device, batch_size=b)
            assert ids is not None
            mask = mask | pairwise_eq(ids)
        eye = torch.eye(b, dtype=torch.bool, device=device)
        return mask & ~eye

    def forward(
        self,
        z_eeg: torch.Tensor,
        z_vid: torch.Tensor,
        meta: Mapping[str, torch.Tensor],
    ) -> Dict[str, Any]:
        z_eeg, z_vid = self._prepare(z_eeg, z_vid, self.renormalize)
        assert z_vid is not None
        b, device = z_eeg.shape[0], z_eeg.device

        if b < 2:
            return {
                "loss": self._zero_loss(z_eeg, z_vid),
                "logs": {"loss": 0.0, "n_valid": 0.0, "degenerate": 1.0},
            }

        scale = self.scale()
        logits = scale * (z_eeg @ z_vid.transpose(0, 1))  # (B, B)
        eye = torch.eye(b, dtype=torch.bool, device=device)
        drop = self._false_negative_mask(meta, b, device)

        # Candidates kept in the denominator: the positive (diagonal) plus every
        # off-diagonal column not flagged as a duplicate.
        keep = (~drop) | eye
        # A row is usable only if at least one *true* negative survives; with
        # zero negatives the softmax is trivially 1 and the term is a constant
        # with no gradient, so it must not dilute the mean.
        has_neg = (keep & ~eye).any(dim=1)  # (B,)

        target = torch.arange(b, device=device)
        logp_e2v = masked_log_softmax(logits, keep, dim=1)
        logp_v2e = masked_log_softmax(logits.transpose(0, 1), keep.transpose(0, 1), dim=1)
        row_e2v = -logp_e2v.gather(1, target.view(-1, 1)).squeeze(1)  # (B,)
        row_v2e = -logp_v2e.gather(1, target.view(-1, 1)).squeeze(1)  # (B,)

        n_valid = has_neg.sum()
        if int(n_valid) == 0:
            # Every sample in the batch shares the masked key: nothing to
            # contrast against. Happens for tiny batches or a pathological
            # sampler; must not be a NaN.
            return {
                "loss": self._zero_loss(z_eeg, z_vid),
                "logs": {
                    "loss": 0.0,
                    "n_valid": 0.0,
                    "degenerate": 1.0,
                    "masked_frac": to_float(drop.float().mean()),
                    "logit_scale": to_float(scale),
                },
            }

        denom = n_valid.clamp_min(1).to(logits.dtype)
        loss_e2v = (row_e2v * has_neg).sum() / denom
        loss_v2e = (row_v2e * has_neg).sum() / denom
        if self.direction == "both":
            loss = 0.5 * (loss_e2v + loss_v2e)
        elif self.direction == "e2v":
            loss = loss_e2v
        else:
            loss = loss_v2e

        with torch.no_grad():
            # argmax over the *kept* candidates only, so accuracy is not
            # penalised for a duplicate outranking the diagonal.
            masked_logits = logits.masked_fill(~keep, float("-inf"))
            pred = masked_logits.argmax(dim=1)
            acc = ((pred == target) & has_neg).float().sum() / denom
            off = ~eye
            n_neg = (keep & off).sum(dim=1).float()

        return {
            "loss": loss,
            "logs": {
                "loss": to_float(loss),
                "loss_e2v": to_float(loss_e2v),
                "loss_v2e": to_float(loss_v2e),
                "logit_scale": to_float(scale),
                "temperature": self.temperature,
                "acc_e2v": to_float(acc),
                "masked_frac": to_float(drop.float().sum() / off.float().sum()),
                "mean_n_neg": to_float(n_neg.mean()),
                "n_valid": float(int(n_valid)),
                "degenerate": 0.0,
            },
        }

    def extra_repr(self) -> str:
        return (
            f"mask_keys={self.mask_keys}, direction={self.direction}, "
            f"temperature={self.temperature:.4f}"
        )


if __name__ == "__main__":  # pragma: no cover
    from .base import make_dummy_batch

    torch.manual_seed(0)
    for name, fn in [
        ("cond-only", MaskedInfoNCE()),
        ("cond+video", MaskedInfoNCE(mask_same_video=True)),
    ]:
        for tag, kw in [
            ("random", {}),
            ("many-dups", {"n_unique_conditions": 4}),
            ("all-same-condition", {"single_condition": True}),
            ("B=2", {"batch_size": 2}),
        ]:
            ze, zv, m = make_dummy_batch(dim=256, **{"batch_size": 32, **kw})
            out = fn(ze, zv, m)
            out["loss"].backward()
            print(
                f"[masked_infonce/{name}/{tag:20s}] "
                f"loss={out['loss'].item():.4f} "
                f"masked_frac={out['logs'].get('masked_frac', 0):.3f} "
                f"n_valid={out['logs']['n_valid']:.0f}"
            )
