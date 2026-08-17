"""CLISA-style cross-subject contrast (Shen et al., IEEE TAC 2022), EEG only.

Positives are trials of the **same condition recorded from a different
subject**; negatives are trials of a different condition.  No video side is
involved, so this term shapes the EEG embedding directly into a
subject-invariant space rather than routing invariance through a shared video
anchor.

The non-negotiable constraint
-----------------------------
Negatives drawn from the **same recording sequence as the anchor** are refused.
EEG carries slow drift -- impedance, sweat, alertness, electrode settling -- on
a timescale far longer than the 800 ms SOA, so two trials from the same sequence
share a low-frequency signature that has nothing to do with stimulus content.
A contrastive objective is an efficient shortcut-finder: given the choice, it
will separate anchor from same-sequence negative using drift, score a low loss,
and learn nothing transferable.  Because ``sequence_id`` runs 1..32 *within* each
subject, the refusal must be scoped by ``(subject_id, sequence_id)`` jointly --
subject 3's sequence 5 and subject 40's sequence 5 are unrelated recordings and
refusing that pair would throw away good negatives for no reason.  That is what
``sequence_scope="subject"`` (the default) does.  Use ``"global"`` only if the
trial table has been rewritten with globally unique sequence ids.

Redundancy warning
------------------
This term is partly redundant with the cross-modal objective: pulling every
subject's trial toward a shared frozen video prototype *already* pulls those
trials toward each other.  Its marginal contribution has to be measured by an
ablation at fixed everything-else, not assumed.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch

from .base import (
    ContrastiveLoss,
    TemperatureMixin,
    get_meta,
    masked_logsumexp,
    pairwise_eq,
    register_loss,
    to_float,
)

_VALID_SCOPES = ("subject", "global", "none")


@register_loss("clisa")
class CLISA(ContrastiveLoss, TemperatureMixin):
    """Cross-subject, same-condition contrast on ``z_eeg``.

    For anchor ``i`` with positive set ``P(i)`` and negative set ``N(i)``::

        L_i = -1/|P(i)| * sum_{p in P(i)}
                  log[ exp(s_ip) / ( exp(s_ip) + sum_{n in N(i)} exp(s_in) ) ]

    The denominator holds one positive plus the negatives, rather than every
    non-self candidate as in SupCon's ``L_out``.  That is the right choice here
    precisely because the negative set is *restricted*: entries excluded by the
    sequence refusal must be absent from the denominator too, otherwise the
    refusal accomplishes nothing.

    Parameters
    ----------
    positive_key
        Condition identity for positives, ``"condition_id"`` (default) or
        ``"video_id"`` (pools orientations).
    require_different_subject
        Keep ``True``.  Same-subject same-condition pairs are the 8 repeats,
        which are contaminated by within-subject drift and would let the loss
        be satisfied without any cross-subject alignment.
    sequence_scope
        ``"subject"`` (default): refuse negatives sharing both ``subject_id`` and
        ``sequence_id``.  ``"global"``: refuse on ``sequence_id`` alone.
        ``"none"``: disable the refusal -- provided only so the ablation
        quantifying the drift shortcut can be run, never for a real model.
    negatives_cross_subject_only
        Additionally require negatives to come from a different subject.
    mask_same_video_negatives
        Do not use a different orientation of the anchor's own base video as a
        negative.
    temperature, learnable_temperature, max_scale, min_scale
        See :class:`tactus.losses.base.TemperatureMixin`.
    """

    requires_video = False

    def __init__(
        self,
        positive_key: str = "condition_id",
        require_different_subject: bool = True,
        sequence_scope: str = "subject",
        negatives_cross_subject_only: bool = False,
        mask_same_video_negatives: bool = False,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        max_scale: float = 100.0,
        min_scale: float = 0.01,
        renormalize: bool = True,
    ) -> None:
        super().__init__()
        if positive_key not in ("condition_id", "video_id"):
            raise ValueError(
                f"positive_key must be 'condition_id' or 'video_id', got {positive_key!r}"
            )
        if sequence_scope not in _VALID_SCOPES:
            raise ValueError(
                f"sequence_scope must be one of {_VALID_SCOPES}, got {sequence_scope!r}"
            )
        self._init_temperature(
            temperature,
            learnable=learnable_temperature,
            max_scale=max_scale,
            min_scale=min_scale,
        )
        self.positive_key = positive_key
        self.require_different_subject = bool(require_different_subject)
        self.sequence_scope = sequence_scope
        self.negatives_cross_subject_only = bool(negatives_cross_subject_only)
        self.mask_same_video_negatives = bool(mask_same_video_negatives)
        self.renormalize = bool(renormalize)

        needed = [positive_key, "subject_id"]
        if sequence_scope != "none":
            needed.append("sequence_id")
        if mask_same_video_negatives:
            needed.append("video_id")
        self.requires_meta = tuple(dict.fromkeys(needed))

    # ------------------------------------------------------------------ #

    def _masks(self, meta: Mapping[str, torch.Tensor], b: int, device: torch.device):
        """Build ``(pos, neg, refused)`` boolean ``(B, B)`` masks."""
        cond = get_meta(meta, self.positive_key, device=device, batch_size=b)
        subj = get_meta(meta, "subject_id", device=device, batch_size=b)
        assert cond is not None and subj is not None
        eye = torch.eye(b, dtype=torch.bool, device=device)

        same_cond = pairwise_eq(cond)
        same_subj = pairwise_eq(subj)

        pos = same_cond & ~eye
        if self.require_different_subject:
            pos = pos & ~same_subj

        neg = ~same_cond & ~eye

        # --- the drift refusal ------------------------------------------- #
        if self.sequence_scope == "none":
            refused = torch.zeros_like(neg)
        else:
            seq = get_meta(meta, "sequence_id", device=device, batch_size=b)
            assert seq is not None
            same_seq = pairwise_eq(seq)
            if self.sequence_scope == "subject":
                # sequence_id is 1..32 *within* a subject: only the conjunction
                # identifies one physical recording block.
                same_block = same_seq & same_subj
            else:
                same_block = same_seq
            refused = neg & same_block
            neg = neg & ~same_block

        if self.negatives_cross_subject_only:
            neg = neg & ~same_subj
        if self.mask_same_video_negatives:
            vid = get_meta(meta, "video_id", device=device, batch_size=b)
            assert vid is not None
            neg = neg & ~pairwise_eq(vid)

        return pos, neg, refused

    def forward(
        self,
        z_eeg: torch.Tensor,
        z_vid: torch.Tensor,
        meta: Mapping[str, torch.Tensor],
    ) -> Dict[str, Any]:
        z_eeg, _ = self._prepare(z_eeg, None, self.renormalize)
        b, device = z_eeg.shape[0], z_eeg.device

        if b < 2:
            return {
                "loss": self._zero_loss(z_eeg, z_vid),
                "logs": {"loss": 0.0, "n_valid": 0.0, "degenerate": 1.0},
            }

        scale = self.scale()
        sim = scale * (z_eeg @ z_eeg.transpose(0, 1))  # (B, B)
        pos, neg, refused = self._masks(meta, b, device)

        n_pos = pos.sum(dim=1)
        n_neg = neg.sum(dim=1)
        row_valid = (n_pos > 0) & (n_neg > 0)
        n_valid = int(row_valid.sum())

        if n_valid == 0:
            # Single-subject batch (no cross-subject positives) or a batch where
            # every candidate was refused. Common with a naive sampler -- if this
            # fires often, the batch sampler is not grouping conditions across
            # subjects and CLISA is contributing nothing.
            return {
                "loss": self._zero_loss(z_eeg, z_vid),
                "logs": {
                    "loss": 0.0,
                    "n_valid": 0.0,
                    "degenerate": 1.0,
                    "mean_n_pos": to_float(n_pos.float().mean()),
                    "mean_n_neg": to_float(n_neg.float().mean()),
                    "refused_frac": to_float(refused.float().mean()),
                    "logit_scale": to_float(scale),
                },
            }

        # log-denominator per (anchor, positive): exp(s_ip) + sum_neg exp(s_in)
        neg_lse = masked_logsumexp(sim, neg, dim=1)  # (B,)
        # broadcast to (B, B) over the positive index p
        both = torch.stack(
            [sim, neg_lse.unsqueeze(1).expand(-1, b)], dim=0
        )  # (2, B, B)
        log_denom = torch.logsumexp(both, dim=0)  # (B, B)
        per_pair = sim - log_denom  # (B, B), = log p(p | i)

        pos_f = pos.to(per_pair.dtype)
        per_anchor = -(per_pair * pos_f).sum(dim=1) / n_pos.clamp_min(1).to(
            per_pair.dtype
        )
        loss = (per_anchor * row_valid.to(per_anchor.dtype)).sum() / float(n_valid)

        with torch.no_grad():
            raw = sim / scale
            pos_sim = (raw * pos_f).sum() / pos_f.sum().clamp_min(1)
            neg_f = neg.to(raw.dtype)
            neg_sim = (raw * neg_f).sum() / neg_f.sum().clamp_min(1)
            n_subj = int(get_meta(meta, "subject_id", device=device).unique().numel())

        return {
            "loss": loss,
            "logs": {
                "loss": to_float(loss),
                "logit_scale": to_float(scale),
                "temperature": self.temperature,
                "mean_n_pos": to_float(n_pos.float().mean()),
                "mean_n_neg": to_float(n_neg.float().mean()),
                "refused_frac": to_float(refused.float().mean()),
                "pos_sim": to_float(pos_sim),
                "neg_sim": to_float(neg_sim),
                "n_subjects_in_batch": float(n_subj),
                "n_valid": float(n_valid),
                "degenerate": 0.0,
            },
        }

    def extra_repr(self) -> str:
        return (
            f"positive_key={self.positive_key}, "
            f"sequence_scope={self.sequence_scope}, "
            f"require_different_subject={self.require_different_subject}, "
            f"temperature={self.temperature:.4f}"
        )


if __name__ == "__main__":  # pragma: no cover
    from .base import make_dummy_batch

    torch.manual_seed(0)
    for tag, fn in [
        ("default", CLISA()),
        ("video-level", CLISA(positive_key="video_id")),
        ("no-refusal(ablation)", CLISA(sequence_scope="none")),
        ("cross-subj-negatives", CLISA(negatives_cross_subject_only=True)),
    ]:
        for btag, kw in [
            ("random", {}),
            ("few-conditions", {"n_unique_conditions": 6}),
            ("single-subject", {"single_subject": True}),
            ("single-subj+seq", {"single_subject": True, "single_sequence": True}),
            ("all-same-condition", {"single_condition": True}),
        ]:
            ze, zv, m = make_dummy_batch(dim=256, **{"batch_size": 32, **kw})
            out = fn(ze, zv, m)
            out["loss"].backward()
            print(
                f"[clisa/{tag:22s}/{btag:18s}] loss={out['loss'].item():.4f} "
                f"n_valid={out['logs']['n_valid']:.0f} "
                f"refused={out['logs']['refused_frac']:.3f} "
                f"n_pos={out['logs']['mean_n_pos']:.2f}"
            )
