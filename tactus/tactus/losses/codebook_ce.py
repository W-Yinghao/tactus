"""Live-codebook cross-entropy: ProtoNCE's bank with a gradient path (E1a-ii).

Why this exists.  ``ProtoNCE`` holds its prototypes in an EMA bank built under
``no_grad``; with ``live_positive=False`` (the only safe setting, see
``tests/test_protonce_shortcut.py``) nothing the video projector produces ever
reaches the loss with a gradient, so the projector sits at its seeded init --
verified bit-identical across four independently trained folds (D24).  The
program's E1a asks what that costs.  Making only the *positive* live is the
documented shortcut (stale negatives, live positive: the projector wins by
rotating the batch away from its own lagging copies).  The symmetric fix is to
make **every** prototype live: the trainer projects the whole frozen codebook
each step and hands it over as ``meta["codebook"]``; this loss is then a plain
cross-entropy of each EEG embedding against all 360 live condition prototypes
(and, at weight ``weights["video"]``, against the 90 video prototypes obtained
by averaging the four orientation prototypes).  No bank, no momentum, no
stale/live asymmetry -- so the shortcut has nothing to exploit, which
``tests/test_codebook_ce_shortcut.py`` checks with noise EEG.

Requires ``meta["codebook"]`` of shape ``(n_conditions, D)`` (L2 rows) -- the
trainer supplies it when the loss sets ``requires_live_codebook = True``.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from .base import ContrastiveLoss, TemperatureMixin, register_loss, to_float


@register_loss("codebook_ce")
class LiveCodebookCE(ContrastiveLoss, TemperatureMixin):
    requires_video = True
    requires_live_codebook = True

    def __init__(
        self,
        n_conditions: int = 360,
        n_orientations: int = 4,
        granularities: Sequence[str] = ("condition", "video"),
        weights: Optional[Mapping[str, float]] = None,
        temperature: float = 0.04,
        learnable_temperature: bool = True,
        max_scale: float = 100.0,
        min_scale: float = 0.01,
        label_smoothing: float = 0.0,
        renormalize: bool = True,
    ) -> None:
        super().__init__()
        self.n_conditions = int(n_conditions)
        self.n_orientations = int(n_orientations)
        self.granularities = tuple(str(g) for g in granularities)
        bad = [g for g in self.granularities if g not in ("condition", "video")]
        if bad:
            raise ValueError(f"unknown granularities {bad}")
        w = dict(weights or {"condition": 1.0, "video": 0.5})
        self.weights = {g: float(w.get(g, 1.0)) for g in self.granularities}
        self.label_smoothing = float(label_smoothing)
        self.renormalize = bool(renormalize)
        self._init_temperature(temperature, learnable=learnable_temperature,
                               max_scale=max_scale, min_scale=min_scale)

    def forward(self, z_eeg: torch.Tensor, z_vid: Optional[torch.Tensor],
                meta: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "codebook" not in meta:
            raise KeyError(
                "codebook_ce needs meta['codebook'] (live projected prototypes); "
                "the trainer supplies it when the loss sets requires_live_codebook")
        codebook = meta["codebook"]
        if codebook.shape[0] != self.n_conditions:
            raise ValueError(f"codebook has {codebook.shape[0]} rows, expected {self.n_conditions}")
        z = F.normalize(z_eeg.float(), dim=-1) if self.renormalize else z_eeg.float()
        cb = F.normalize(codebook.float(), dim=-1) if self.renormalize else codebook.float()
        scale = self.scale()
        cond = meta["condition_id"].long()
        logs: Dict[str, torch.Tensor] = {}
        total = z.new_zeros(())

        if "condition" in self.granularities:
            logits = scale * (z @ cb.t())                          # (B, 360), all live
            loss_c = F.cross_entropy(logits, cond, label_smoothing=self.label_smoothing)
            total = total + self.weights["condition"] * loss_c
            logs["cb/condition_loss"] = loss_c.detach()
            logs["cb/condition_acc"] = (logits.argmax(1) == cond).float().mean().detach()

        if "video" in self.granularities:
            n_vid = self.n_conditions // self.n_orientations
            vid_protos = F.normalize(
                cb.view(n_vid, self.n_orientations, -1).mean(dim=1), dim=-1)  # (90, D), live
            vid_target = cond // self.n_orientations
            logits_v = scale * (z @ vid_protos.t())
            loss_v = F.cross_entropy(logits_v, vid_target, label_smoothing=self.label_smoothing)
            total = total + self.weights["video"] * loss_v
            logs["cb/video_loss"] = loss_v.detach()
            logs["cb/video_acc"] = (logits_v.argmax(1) == vid_target).float().mean().detach()

        logs["cb/scale"] = scale.detach() if torch.is_tensor(scale) else torch.tensor(float(scale))
        return {"loss": total, **{k: to_float(v) for k, v in logs.items()}}
