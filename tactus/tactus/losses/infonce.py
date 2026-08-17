"""Classic CLIP-style symmetric InfoNCE -- the reference implementation.

This is deliberately the *plainest possible* cross-modal contrastive loss: the
diagonal of the (B, B) similarity matrix is positive, everything else is
negative, both directions are averaged, temperature is learnable.  No masking,
no reweighting, no tricks.

It is the baseline every other loss in this package is measured against, and it
doubles as the correctness oracle: if a fancier loss does not reduce to
something numerically close to this one when its extra machinery is disabled,
the fancier loss has a bug.

Known and accepted defect on TACTUS data
----------------------------------------
With 360 conditions and a batch of a few hundred trials, batches routinely
contain several trials of the *same* condition.  Their video embeddings are
byte-identical (the video tower is frozen), so this loss asks the model to push
apart two rows whose targets are the same vector -- an unsatisfiable objective
that puts a floor on the loss and adds gradient noise.  That is precisely what
:mod:`tactus.losses.masked_infonce` fixes.  Keep this file unmasked anyway: the
size of the gap between the two is a number worth reporting.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F

from .base import ContrastiveLoss, TemperatureMixin, register_loss, to_float


@register_loss("infonce")
class InfoNCE(ContrastiveLoss, TemperatureMixin):
    """Symmetric CLIP loss.

    .. math::

        \\mathcal{L} = \\tfrac{1}{2}\\big[
            \\mathrm{CE}(s \\cdot Z_e Z_v^\\top,\\, \\mathrm{diag}) +
            \\mathrm{CE}(s \\cdot Z_v Z_e^\\top,\\, \\mathrm{diag}) \\big]

    Parameters
    ----------
    temperature
        Initial temperature; the stored parameter is ``log(1 / temperature)``.
    learnable_temperature
        If ``True`` (default) the temperature is optimized jointly, clamped to a
        logit scale of at most ``max_scale``.
    max_scale, min_scale
        Clamp range for the multiplicative logit scale.  CLIP uses 100 as the
        ceiling; keeping it prevents the run-away sharpening that otherwise
        turns into a silent collapse.
    label_smoothing
        Passed to ``F.cross_entropy``.  Mild smoothing (0.05-0.1) is a cheap
        hedge against the duplicate-condition problem described above, but it is
        a blunt instrument compared to explicit masking.
    direction
        ``"both"`` (default, symmetric), ``"e2v"`` (EEG queries a video gallery
        -- matches the retrieval metric), or ``"v2e"``.
    renormalize
        Defensive re-normalization of the inputs.  A no-op under the contract.
    """

    requires_video = True
    requires_meta = ()

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        max_scale: float = 100.0,
        min_scale: float = 0.01,
        label_smoothing: float = 0.0,
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
        self.direction = direction
        self.label_smoothing = float(label_smoothing)
        self.renormalize = bool(renormalize)

    def forward(
        self,
        z_eeg: torch.Tensor,
        z_vid: torch.Tensor,
        meta: Mapping[str, torch.Tensor],
    ) -> Dict[str, Any]:
        z_eeg, z_vid = self._prepare(z_eeg, z_vid, self.renormalize)
        assert z_vid is not None
        b = z_eeg.shape[0]

        if b < 2:
            # A single sample has no negatives: cross-entropy over one class is
            # identically zero and carries no gradient. Return a graph-connected
            # zero rather than a misleading 0.0 that looks like convergence.
            return {
                "loss": self._zero_loss(z_eeg, z_vid),
                "logs": {"loss": 0.0, "n_valid": 0.0, "degenerate": 1.0},
            }

        scale = self.scale()
        logits_e2v = scale * (z_eeg @ z_vid.transpose(0, 1))  # (B, B)
        logits_v2e = logits_e2v.transpose(0, 1)
        target = torch.arange(b, device=z_eeg.device)

        loss_e2v = F.cross_entropy(
            logits_e2v, target, label_smoothing=self.label_smoothing
        )
        loss_v2e = F.cross_entropy(
            logits_v2e, target, label_smoothing=self.label_smoothing
        )
        if self.direction == "both":
            loss = 0.5 * (loss_e2v + loss_v2e)
        elif self.direction == "e2v":
            loss = loss_e2v
        else:
            loss = loss_v2e

        with torch.no_grad():
            sim = logits_e2v / scale
            pos = sim.diagonal()
            off = ~torch.eye(b, dtype=torch.bool, device=z_eeg.device)
            acc_e2v = (logits_e2v.argmax(dim=1) == target).float().mean()
            acc_v2e = (logits_v2e.argmax(dim=1) == target).float().mean()
            # How often the "negative" is actually the same condition: the cost
            # of not masking, measured rather than assumed.
            if "condition_id" in meta:
                cond = meta["condition_id"].reshape(-1)
                dup = (cond.reshape(-1, 1) == cond.reshape(1, -1)) & off
                dup_rate = dup.float().sum() / max(off.float().sum().item(), 1.0)
            else:
                dup_rate = torch.zeros((), device=z_eeg.device)

        return {
            "loss": loss,
            "logs": {
                "loss": to_float(loss),
                "loss_e2v": to_float(loss_e2v),
                "loss_v2e": to_float(loss_v2e),
                "logit_scale": to_float(scale),
                "temperature": self.temperature,
                "acc_e2v": to_float(acc_e2v),
                "acc_v2e": to_float(acc_v2e),
                "pos_sim": to_float(pos.mean()),
                "neg_sim": to_float(sim[off].mean()),
                "false_neg_rate": to_float(dup_rate),
                "n_valid": float(b),
            },
        }

    def extra_repr(self) -> str:
        return (
            f"direction={self.direction}, temperature={self.temperature:.4f}, "
            f"learnable={self._temp_learnable}, "
            f"label_smoothing={self.label_smoothing}"
        )


if __name__ == "__main__":  # pragma: no cover
    from .base import make_dummy_batch

    torch.manual_seed(0)
    loss_fn = InfoNCE()
    for tag, kw in [
        ("random", {}),
        ("all-same-condition", {"single_condition": True}),
        ("B=1", {"batch_size": 1}),
        ("B=2", {"batch_size": 2}),
    ]:
        ze, zv, m = make_dummy_batch(dim=256, **{"batch_size": 32, **kw})
        out = loss_fn(ze, zv, m)
        out["loss"].backward()
        print(f"[infonce/{tag:20s}] loss={out['loss'].item():.4f} logs={out['logs']}")
